# -*- coding: utf-8 -*-
"""
DNA-PAINT: binding events, sticking, kinetics and qPAINT counting.

DNA-PAINT does not blink like dSTORM, and that difference invalidates a
specific number this software reports. In dSTORM a fluorophore emits a
handful of times and bleaches, so "how many localizations are in this
cluster" is at least loosely related to how much protein is there. In
DNA-PAINT an imager strand binds a docking site, is imaged, unbinds, and
another one binds -- for the whole acquisition, without end. The number
of localizations at a site then measures HOW LONG YOU IMAGED, not how
many molecules are present.

What replaces it is qPAINT (Jungmann et al., Nat Methods 2016): the mean
DARK time between binding events at a site falls as 1/N with the number
of docking sites, because N sites capture imagers N times as fast. So

    n_units = 1 / (influx_rate * tau_dark)

and ``influx_rate`` has to be measured on a reference of known valency in
the same sample at the same imager concentration -- a single docking
strand on a DNA origami is the usual one (Stein et al., PNAS 2025, do
exactly this for tissue cryosections). Without that calibration the
formula still produces a number, and that number is meaningless in
absolute terms. This module therefore refuses to present an uncalibrated
count as a count; see ``QPaintResult.is_calibrated``.

Three things have to happen before any of that
----------------------------------------------
1. LINK. One binding event spans several consecutive frames. Until
   they are collapsed, every per-event statistic counts the same event
   several times.
2. REJECT STICKING. An imager stuck non-specifically produces a dense,
   bright cluster confined to a short stretch of the movie. It looks
   exactly like a real site to DBSCAN and nothing like one in time.
3. MEASURE THE DARK TIMES, which only exist once events do.

Relationship to the rest of MPS Explorer
----------------------------------------
Nothing here changes the geometry pipeline. It produces an alternative
set of POINTS -- one per binding event instead of one per localization,
or one per docking site -- which the existing clustering, perimeter and
occupancy code can consume unchanged. The mode has to be visible in the
output, though, because an occupancy computed over binding events is not
comparable with one computed over raw localizations.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import warnings as warnings_module
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.optimize import OptimizeWarning

# Linking radius, as a multiple of the lateral localization precision.
# Two localizations of one binding event differ by about sqrt(2) * lp, so
# 3 * lp keeps essentially all of them while staying well inside the
# distance to a neighbouring docking site. Picasso's own API default
# (0.05 camera pixels, ~5.7 nm here) is SMALLER than one precision and
# would cut most events in half; its CLI default (1.0 pixel, ~113 nm)
# would merge neighbouring sites. Neither travels between cameras, so the
# default here is derived from the data instead of fixed.
LINK_RADIUS_IN_PRECISIONS = 3.0

# Frames of tolerated darkness inside one binding event. A bound imager
# that is simply dim for one frame should not split the event in two;
# anything longer is a genuine unbinding. Kept small on purpose: this
# parameter trades directly against tau_dark, and tau_dark IS the qPAINT
# measurement.
DEFAULT_MAX_DARK_TIME = 1

# Picasso's basic frame analysis, from picasso.clusterer._frame_analysis.
_FA_MEAN_FRAME_LOW = 0.20
_FA_MEAN_FRAME_HIGH = 0.80
_FA_N_WINDOWS = 20
_FA_MAX_SHARE_IN_WINDOW = 0.80


# ===================================================================
#  1. Linking localizations into binding events
# ===================================================================
def link_localizations(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    radius_nm: float,
    max_dark_time: int = DEFAULT_MAX_DARK_TIME,
    group: Optional[NDArray[np.int64]] = None,
) -> NDArray[np.int64]:
    """
    Assign each localization to a binding event.

    Follows ``picasso.postprocess._get_link_groups`` exactly, including
    one detail worth stating: the chain takes the FIRST unassigned
    localization within ``radius_nm`` in the search window, not the
    nearest one. Taking the nearest would arguably be better, but it
    would silently disagree with ``picasso link`` on the same data, and
    being able to cross-check against Picasso is worth more than a
    marginal improvement.

    The chain is relative to the LAST localization added, not to the
    first, so a long event may wander further than ``radius_nm`` in
    total. That is the intended behaviour: a bound imager's apparent
    position random-walks with the localization error.

    Parameters
    ----------
    frame : frame index of each localization.
    x_nm, y_nm : coordinates, nanometres.
    radius_nm : maximum step between consecutive localizations of one
        event. See LINK_RADIUS_IN_PRECISIONS for how to choose it.
    max_dark_time : frames that may be missing inside one event. Search
        window is [f+1, f+max_dark_time+1] inclusive, as in Picasso.
    group : optional pre-existing labels (e.g. from clustering); events
        never span two groups.

    Returns
    -------
    link_group : event index per localization, in INPUT order.
    """
    frame = np.asarray(frame, dtype=np.int64)
    x = np.asarray(x_nm, dtype=float)
    y = np.asarray(y_nm, dtype=float)
    n = frame.size
    if n == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(frame, kind="stable")
    f_sorted = frame[order]
    x_sorted, y_sorted = x[order], y[order]
    if group is None:
        g_sorted = np.zeros(n, dtype=np.int64)
    else:
        g_sorted = np.asarray(group, dtype=np.int64)[order]

    link = np.full(n, -1, dtype=np.int64)
    radius2 = float(radius_nm) ** 2
    current = -1

    for i in range(n):
        if link[i] != -1:
            continue
        current += 1
        link[i] = current
        j = i
        while True:
            # Frames [f+1, f+max_dark_time+1], inclusive at both ends.
            lo = int(np.searchsorted(f_sorted, f_sorted[j] + 1, "left"))
            hi = int(np.searchsorted(
                f_sorted, f_sorted[j] + max_dark_time + 1, "right"))
            if hi <= lo:
                break
            window = np.arange(lo, hi)
            free = window[
                (link[lo:hi] == -1) & (g_sorted[lo:hi] == g_sorted[j])
            ]
            if free.size == 0:
                break
            d2 = (x_sorted[free] - x_sorted[j]) ** 2 + \
                 (y_sorted[free] - y_sorted[j]) ** 2
            within = np.nonzero(d2 <= radius2)[0]
            if within.size == 0:
                break
            # First in index order -- i.e. earliest frame, then input
            # order within that frame. Picasso's rule.
            j = int(free[within[0]])
            link[j] = current

    out = np.empty(n, dtype=np.int64)
    out[order] = link
    return out


@dataclass
class BindingEvents:
    """
    One row per binding event, in the order the events first appear.

    ``length`` is the span in frames (last - first + 1) and ``n_locs``
    the number of localizations in it. They differ when ``max_dark_time``
    let a frame be missing inside the event, which is why Picasso keeps
    both as ``len`` and ``n``.
    """

    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: Optional[NDArray[np.float64]]
    first_frame: NDArray[np.int64]
    last_frame: NDArray[np.int64]
    length: NDArray[np.int64]
    n_locs: NDArray[np.int64]
    photons: Optional[NDArray[np.float64]]
    photon_rate: Optional[NDArray[np.float64]]
    group: Optional[NDArray[np.int64]]
    link_group: NDArray[np.int64]     # per input localization
    radius_nm: float
    max_dark_time: int

    @property
    def n(self) -> int:
        return int(self.first_frame.size)


def build_events(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    *,
    z_nm: Optional[NDArray[np.float64]] = None,
    photons: Optional[NDArray[np.float64]] = None,
    lp_nm: Optional[NDArray[np.float64]] = None,
    group: Optional[NDArray[np.int64]] = None,
    radius_nm: float,
    max_dark_time: int = DEFAULT_MAX_DARK_TIME,
) -> BindingEvents:
    """
    Link localizations and collapse each event to a single point.

    The position is the inverse-variance weighted mean when ``lp_nm`` is
    given, otherwise the plain mean. Weighting matters: within one event
    the localizations differ in photon count by a lot, and a dim frame
    should not pull the event's position as hard as a bright one.
    """
    link = link_localizations(
        frame, x_nm, y_nm, radius_nm, max_dark_time, group
    )
    n_events = int(link.max()) + 1 if link.size else 0
    if n_events == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=float)
        return BindingEvents(
            empty_f, empty_f, None, empty_i, empty_i, empty_i, empty_i,
            None, None, None, link, radius_nm, max_dark_time,
        )

    frame = np.asarray(frame, dtype=np.int64)
    weights = (
        np.ones(link.size, dtype=float)
        if lp_nm is None
        else 1.0 / np.clip(np.asarray(lp_nm, dtype=float), 1e-6, None) ** 2
    )
    weight_sum = np.bincount(link, weights=weights, minlength=n_events)
    weight_sum[weight_sum == 0] = 1.0

    def weighted(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.bincount(
            link, weights=np.asarray(values, dtype=float) * weights,
            minlength=n_events,
        ) / weight_sum

    counts = np.bincount(link, minlength=n_events).astype(np.int64)
    first = np.full(n_events, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, link, frame)
    last = np.full(n_events, np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(last, link, frame)
    length = (last - first + 1).astype(np.int64)

    total_photons: Optional[NDArray[np.float64]] = (
        None if photons is None
        else np.asarray(
            np.bincount(link, weights=np.asarray(photons, dtype=float),
                        minlength=n_events),
            dtype=np.float64,
        )
    )
    rate = None if total_photons is None else total_photons / np.maximum(
        length, 1
    )
    event_group = None
    if group is not None:
        # Scatter-assign: every localization of an event writes its own
        # label into the event's slot. Safe because linking never lets an
        # event span two groups, so they are all writing the same value.
        event_group = np.zeros(n_events, dtype=np.int64)
        event_group[link] = np.asarray(group, dtype=np.int64)

    return BindingEvents(
        x_nm=weighted(x_nm),
        y_nm=weighted(y_nm),
        z_nm=None if z_nm is None else weighted(z_nm),
        first_frame=first,
        last_frame=last,
        length=length,
        n_locs=counts,
        photons=total_photons,
        photon_rate=rate,
        group=event_group,
        link_group=link,
        radius_nm=float(radius_nm),
        max_dark_time=int(max_dark_time),
    )


# ===================================================================
#  2. Rejecting non-specific sticking
# ===================================================================
@dataclass
class FrameAnalysis:
    """Which clusters look like docking sites over time, and which do not."""

    passed: NDArray[np.bool_]           # one per cluster label, sorted
    labels: NDArray[np.int64]           # the labels these refer to
    mean_frame_fraction: NDArray[np.float64]
    max_window_share: NDArray[np.float64]
    n_rejected: int
    warnings: List[str] = field(default_factory=list)


def frame_analysis(
    labels: NDArray[np.int64],
    frame: NDArray[np.int64],
    n_frames: Optional[int] = None,
) -> FrameAnalysis:
    """
    Reject clusters whose localizations are not spread over the movie.

    Picasso's rule (``picasso.clusterer.frame_analysis``), unchanged: a
    cluster fails if its mean frame lies outside the middle 20-80 % of
    the acquisition, or if any one twentieth of the acquisition holds
    more than 80 % of its localizations.

    The reasoning is specific to PAINT. A real docking site is sampled
    repeatedly for as long as you image, so its localizations are spread
    roughly uniformly in time and its mean frame sits near the middle.
    An imager stuck non-specifically gives a burst: dense in space,
    indistinguishable from a site to DBSCAN, and confined in time.

    It is worth running on dSTORM data too, where it catches a different
    artefact with the same signature -- a fluorescent contaminant that
    bleaches early, or a cluster that only appears once the reactivating
    405 nm illumination has been ramped up.
    """
    labels = np.asarray(labels, dtype=np.int64)
    frame = np.asarray(frame, dtype=np.int64)
    warnings: List[str] = []

    total_frames = (
        int(np.max(frame)) + 1 if n_frames is None else int(n_frames)
    )
    if total_frames <= 0:
        warnings.append("acquisition length unknown; frame analysis skipped")
        empty = np.empty(0, dtype=np.int64)
        return FrameAnalysis(
            np.empty(0, dtype=bool), empty,
            np.empty(0, dtype=float), np.empty(0, dtype=float), 0, warnings,
        )

    unique = np.unique(labels[labels >= 0])
    passed = np.ones(unique.size, dtype=bool)
    mean_fraction = np.zeros(unique.size, dtype=float)
    window_share = np.zeros(unique.size, dtype=float)
    window_edges = np.linspace(0, total_frames, _FA_N_WINDOWS + 1)

    for i, label in enumerate(unique):
        inside = frame[labels == label]
        if inside.size == 0:
            passed[i] = False
            continue
        mean_fraction[i] = float(np.mean(inside)) / total_frames
        counts, _ = np.histogram(inside, bins=window_edges)
        window_share[i] = float(counts.max()) / inside.size
        passed[i] = (
            _FA_MEAN_FRAME_LOW <= mean_fraction[i] <= _FA_MEAN_FRAME_HIGH
            and window_share[i] <= _FA_MAX_SHARE_IN_WINDOW
        )

    n_rejected = int(np.sum(~passed))
    if unique.size and n_rejected / unique.size > 0.5:
        warnings.append(
            f"{n_rejected} of {unique.size} clusters fail the frame "
            f"analysis. Above half, suspect the acquisition rather than the "
            f"sample: too short a movie leaves every site looking bursty, "
            f"and the test then rejects real structure."
        )
    return FrameAnalysis(
        passed=passed,
        labels=unique,
        mean_frame_fraction=mean_fraction,
        max_window_share=window_share,
        n_rejected=n_rejected,
        warnings=warnings,
    )


# ===================================================================
#  3. Dark times and kinetics
# ===================================================================
def dark_times(
    first_frame: NDArray[np.int64],
    last_frame: NDArray[np.int64],
    group: Optional[NDArray[np.int64]] = None,
) -> NDArray[np.int64]:
    """
    Frames of darkness before each binding event, within its group.

    For event i this is the smallest positive gap between the END of any
    earlier event in the same group and the START of i, which is what
    ``picasso.postprocess._dark_times`` computes -- there in O(n^2), here
    by sorting. An event with nothing before it gets -1.

    A "group" here is a docking site, or whatever stands in for one:
    normally a cluster label. Grouping by anything larger mixes the dark
    times of several sites and makes tau_dark meaningless.
    """
    first = np.asarray(first_frame, dtype=np.int64)
    last = np.asarray(last_frame, dtype=np.int64)
    n = first.size
    out = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return out
    g = np.zeros(n, dtype=np.int64) if group is None else np.asarray(
        group, dtype=np.int64
    )

    order = np.lexsort((first, g))
    g_s, first_s, last_s = g[order], first[order], last[order]
    # Running maximum of the end frame, within each group, over events
    # that started strictly earlier.
    best_end = np.iinfo(np.int64).min
    previous_group = None
    result = np.full(n, -1, dtype=np.int64)
    for k in range(n):
        if previous_group is None or g_s[k] != previous_group:
            previous_group = g_s[k]
            best_end = last_s[k]
            continue
        if best_end < first_s[k]:
            result[k] = first_s[k] - best_end
        best_end = max(best_end, int(last_s[k]))
    out[order] = result
    return out


def cumulative_exponential(
    x: NDArray[np.float64], a: float, t: float, c: float
) -> NDArray[np.float64]:
    """``a * (1 - exp(-x / t)) + c``; Picasso's ``lib.cumulative_exponential``."""
    return a * (1.0 - np.exp(-(x / t))) + c


def estimate_kinetic_rate(
    data: Union[Sequence[float], NDArray[Any]],
) -> float:
    """
    Mean bright or dark time, by fitting the cumulative distribution.

    Picasso's ``lib.estimate_kinetic_rate``. Fitting the CDF rather than
    averaging is what makes this usable: the dark-time distribution is
    exponential and censored -- a dark period longer than the remaining
    acquisition is never observed -- so the sample mean is biased low,
    and biased low by an amount that depends on how long you imaged. The
    fitted time constant is not.
    """
    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 2:
        return float(np.nanmean(values)) if values.size else float("nan")
    values = np.sort(values)
    if values.max() - values.min() == 0:
        return float(np.nanmean(values))
    y = np.arange(1, values.size + 1, dtype=float)
    try:
        with warnings_module.catch_warnings():
            # Only popt is used, never the covariance, so curve_fit's
            # complaint that it could not estimate one is noise here.
            warnings_module.simplefilter("ignore", OptimizeWarning)
            popt, _ = optimize.curve_fit(
                cumulative_exponential, values, y,
                p0=[values.size, float(np.mean(values)), float(values.min())],
                bounds=([0, values.min(), 0],
                        [np.inf, values.max(), np.inf]),
                maxfev=20000,
            )
    except (RuntimeError, ValueError):
        return float(np.nanmean(values))
    return float(popt[1])


@dataclass
class Kinetics:
    tau_bright_frames: float
    tau_dark_frames: float
    tau_bright_s: Optional[float]
    tau_dark_s: Optional[float]
    n_events: int
    n_dark: int
    n_sites: int
    warnings: List[str] = field(default_factory=list)


def kinetics(
    events: BindingEvents,
    *,
    site_labels: Optional[NDArray[np.int64]] = None,
    exposure_s: Optional[float] = None,
    n_frames: Optional[int] = None,
) -> Kinetics:
    """
    Mean bright and dark time, from the binding events.

    ``site_labels`` says which docking site each event belongs to. Pass
    the cluster labels; without them every event is treated as belonging
    to one site and tau_dark measures the whole field of view's binding
    rate, not a site's.
    """
    warnings: List[str] = []
    if events.n == 0:
        return Kinetics(float("nan"), float("nan"), None, None,
                        0, 0, 0, ["no binding events"])

    labels = (
        events.group if site_labels is None else
        np.asarray(site_labels, dtype=np.int64)
    )
    if labels is None:
        labels = np.zeros(events.n, dtype=np.int64)
        warnings.append(
            "no site labels: every event was treated as belonging to one "
            "site, so tau_dark describes the whole region's binding rate "
            "and cannot be turned into a count."
        )

    # Unclustered localizations carry label -1. They are not a docking
    # site, and pooling them would compute dark times between events
    # scattered across the whole region as though one site had produced
    # them all -- which shortens tau_dark and inflates every count.
    real_site = np.asarray(labels) >= 0
    if not np.any(real_site):
        warnings.append(
            "every event is unclustered (label -1), so there are no sites "
            "to measure dark times within."
        )
        return Kinetics(
            estimate_kinetic_rate(events.length.astype(float)),
            float("nan"), None, None, events.n, 0, 0, warnings,
        )

    dark = np.full(events.n, -1, dtype=np.int64)
    dark[real_site] = dark_times(
        events.first_frame[real_site],
        events.last_frame[real_site],
        np.asarray(labels)[real_site],
    )
    observed = dark[dark > 0]
    tau_bright = estimate_kinetic_rate(events.length.astype(float))
    tau_dark = (
        estimate_kinetic_rate(observed.astype(float))
        if observed.size else float("nan")
    )

    if n_frames and np.isfinite(tau_dark) and tau_dark > 0.2 * n_frames:
        warnings.append(
            f"tau_dark ({tau_dark:.0f} frames) is a large fraction of the "
            f"{n_frames}-frame acquisition. Dark periods longer than the "
            f"movie are never seen, so this estimate is a lower bound and "
            f"any count derived from it is an upper bound."
        )
    if observed.size < 20:
        warnings.append(
            f"only {observed.size} observed dark periods; tau_dark is "
            f"poorly determined."
        )

    return Kinetics(
        tau_bright_frames=tau_bright,
        tau_dark_frames=tau_dark,
        tau_bright_s=None if exposure_s is None else tau_bright * exposure_s,
        tau_dark_s=None if exposure_s is None else tau_dark * exposure_s,
        n_events=events.n,
        n_dark=int(observed.size),
        n_sites=int(np.unique(np.asarray(labels)[real_site]).size),
        warnings=warnings,
    )


# ===================================================================
#  4. qPAINT counting
# ===================================================================
@dataclass
class QPaintResult:
    """
    Docking sites per cluster, and whether that number means anything.

    ``is_calibrated`` is the field to read first. qPAINT is a RELATIVE
    measurement: it compares a cluster's binding frequency against a
    reference of known valency imaged under the same conditions. Without
    that reference the units are arbitrary, and ``n_units`` is a ranking,
    not a count.
    """

    labels: NDArray[np.int64]
    tau_dark_frames: NDArray[np.float64]
    n_units: NDArray[np.float64]
    n_events: NDArray[np.int64]
    influx_rate: float                 # per frame
    is_calibrated: bool
    calibration_source: str
    warnings: List[str] = field(default_factory=list)


def influx_from_reference(tau_dark_reference_frames: float) -> float:
    """
    Influx rate from a reference of ONE docking site.

    A single site has ``tau_dark = 1 / influx`` by definition, so the
    reference's dark time is the whole calibration. Measure it on a
    structure of known valency in the same sample, at the same imager
    concentration and the same exposure -- a DNA origami with one docking
    strand is the standard choice, and the one Stein et al. (PNAS 2025)
    use for tissue cryosections.
    """
    if not np.isfinite(tau_dark_reference_frames) or \
            tau_dark_reference_frames <= 0:
        raise ValueError(
            "The reference dark time must be a positive number of frames."
        )
    return 1.0 / float(tau_dark_reference_frames)


def qpaint(
    events: BindingEvents,
    site_labels: NDArray[np.int64],
    *,
    influx_rate: Optional[float] = None,
    tau_dark_reference_frames: Optional[float] = None,
    min_events: int = 3,
) -> QPaintResult:
    """
    Count docking sites per cluster from their binding frequency.

    ``n_units = 1 / (influx_rate * tau_dark)``, as in
    ``picasso.postprocess.pick_properties``.

    Supply EITHER ``influx_rate`` (per frame) or
    ``tau_dark_reference_frames`` from a single-site reference. Supply
    neither and the calculation still runs, with influx set so that the
    median cluster comes out at one unit -- which makes the output a
    relative ranking and nothing more. ``is_calibrated`` is False in that
    case and every caller must say so.
    """
    warnings: List[str] = []
    labels = np.asarray(site_labels, dtype=np.int64)
    unique = np.unique(labels[labels >= 0])
    dark = dark_times(events.first_frame, events.last_frame, labels)

    tau = np.full(unique.size, np.nan, dtype=float)
    counts = np.zeros(unique.size, dtype=np.int64)
    for i, label in enumerate(unique):
        inside = labels == label
        counts[i] = int(np.sum(inside))
        observed = dark[inside & (dark > 0)]
        if observed.size >= max(min_events - 1, 1):
            tau[i] = estimate_kinetic_rate(observed.astype(float))

    too_few = int(np.sum(counts < min_events))
    if too_few:
        warnings.append(
            f"{too_few} of {unique.size} clusters have fewer than "
            f"{min_events} binding events, so they have no usable dark-time "
            f"distribution and no count. Reporting them as zero units would "
            f"be wrong; they are NaN."
        )

    if influx_rate is not None and tau_dark_reference_frames is not None:
        raise ValueError(
            "Give either influx_rate or tau_dark_reference_frames, not both."
        )
    if tau_dark_reference_frames is not None:
        rate = influx_from_reference(tau_dark_reference_frames)
        calibrated, source = True, (
            f"single-site reference, tau_dark = "
            f"{tau_dark_reference_frames:g} frames"
        )
    elif influx_rate is not None:
        rate = float(influx_rate)
        calibrated, source = True, f"influx rate {influx_rate:g} per frame"
    else:
        median_tau = float(np.nanmedian(tau)) if np.any(np.isfinite(tau)) \
            else float("nan")
        rate = 1.0 / median_tau if np.isfinite(median_tau) and median_tau > 0 \
            else float("nan")
        calibrated, source = False, "uncalibrated (median cluster set to 1)"
        warnings.append(
            "No influx calibration was supplied, so these are RELATIVE "
            "units scaled to the median cluster, not docking-site counts. "
            "To get counts, image a reference of known valency (a "
            "single-docking-strand origami) in the same sample at the same "
            "imager concentration and pass its dark time."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        units = 1.0 / (rate * tau)

    return QPaintResult(
        labels=unique,
        tau_dark_frames=tau,
        n_units=units,
        n_events=counts,
        influx_rate=rate,
        is_calibrated=calibrated,
        calibration_source=source,
        warnings=warnings,
    )


# ===================================================================
#  The whole DNA-PAINT pass
# ===================================================================
@dataclass
class PaintReport:
    source: str
    n_locs: int
    events: Optional[BindingEvents] = None
    sticking: Optional[FrameAnalysis] = None
    kinetics_result: Optional[Kinetics] = None
    qpaint_result: Optional[QPaintResult] = None
    radius_nm: float = float("nan")
    missing: List[str] = field(default_factory=list)

    @property
    def compression(self) -> float:
        """Localizations per binding event; 1.0 means nothing linked."""
        if self.events is None or self.events.n == 0:
            return float("nan")
        return self.n_locs / self.events.n

    @property
    def warnings(self) -> List[str]:
        out: List[str] = []
        for part in (self.sticking, self.kinetics_result, self.qpaint_result):
            if part is not None:
                out.extend(part.warnings)
        return out


def paint_report(
    loc: Any,
    *,
    site_labels: Optional[NDArray[np.int64]] = None,
    radius_nm: Optional[float] = None,
    max_dark_time: int = DEFAULT_MAX_DARK_TIME,
    exposure_s: Optional[float] = None,
    influx_rate: Optional[float] = None,
    tau_dark_reference_frames: Optional[float] = None,
    run_qpaint: bool = True,
) -> PaintReport:
    """
    Link, reject sticking, measure kinetics and (optionally) count.

    Takes a ``tools.mps_io.Localizations``. ``radius_nm`` defaults to
    ``LINK_RADIUS_IN_PRECISIONS`` times the median lateral precision of
    this file, which is why the loader has to read lpx and lpy.
    """
    report = PaintReport(source=loc.path, n_locs=loc.n)
    frame = loc.frame
    if frame is None:
        report.missing.append(
            "no 'frame' column: nothing in DNA-PAINT analysis is possible "
            "without it"
        )
        return report

    lp = loc.lp_lateral_nm
    if radius_nm is None:
        if lp is None:
            report.missing.append(
                "no 'lpx'/'lpy' columns and no radius given: cannot choose a "
                "linking radius from the data"
            )
            return report
        radius_nm = LINK_RADIUS_IN_PRECISIONS * float(np.nanmedian(lp))
    report.radius_nm = float(radius_nm)

    report.events = build_events(
        frame, loc.x_nm, loc.y_nm,
        z_nm=loc.z_nm if loc.is_3d else None,
        photons=loc.photons,
        lp_nm=lp,
        group=site_labels,
        radius_nm=report.radius_nm,
        max_dark_time=max_dark_time,
    )

    if site_labels is not None:
        report.sticking = frame_analysis(site_labels, frame, loc.n_frames)

    # build_events already carried the per-localization labels through to
    # one label per event, so the events' own group is what kinetics needs.
    report.kinetics_result = kinetics(
        report.events,
        site_labels=report.events.group,
        exposure_s=exposure_s,
        n_frames=loc.n_frames,
    )

    if run_qpaint and report.events.group is not None:
        report.qpaint_result = qpaint(
            report.events, report.events.group,
            influx_rate=influx_rate,
            tau_dark_reference_frames=tau_dark_reference_frames,
        )
    return report
