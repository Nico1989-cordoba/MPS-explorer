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

    Follows ``picasso.postprocess._get_link_groups``, including one detail
    worth stating: the chain takes the FIRST unassigned localization
    within ``radius_nm`` in the search window, not the nearest one.
    Taking the nearest would arguably be better, but it would silently
    disagree with ``picasso link`` on the same data, and being able to
    cross-check against Picasso is worth more than a marginal improvement.

    Checked against Picasso 0.9.10 on NCtransversal_Roi2_2_1 (19,143
    localizations, 1.5 px, one dark frame): given the same input, the
    events are identical. One way the two could still differ: Picasso
    sorts by frame with an unstable quicksort, so when two localizations
    of the same frame are both within reach, which one is "first" may not
    be the same here.

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
    link_group: NDArray[np.int64]     # per input localization; -1 = no position
    radius_nm: float
    max_dark_time: int
    # Failed fits (see build_events): linked, but no weight in a position.
    n_discarded: int = 0
    # Events left out because they were already bound in the first frame
    # or still bound in the last, so their length is only a lower bound.
    n_censored: int = 0
    # Events left out because every localization in them is a failed fit.
    n_no_position: int = 0
    # Localizations without finite coordinates, which cannot be linked.
    n_unplaced: int = 0

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
    n_frames: Optional[int] = None,
    remove_ambiguous_lengths: bool = True,
) -> BindingEvents:
    """
    Link localizations and collapse each event to a single point.

    Frames are counted from 0 (``Localizations.frame`` already shifts
    ThunderSTORM's 1-based numbering).

    The position is the inverse-variance weighted mean when ``lp_nm`` is
    given, otherwise the plain mean. Weighting matters: within one event
    the localizations differ in photon count by a lot, and a dim frame
    should not pull the event's position as hard as a bright one.

    Failed fits -- a precision that is NaN, zero or negative, which is how
    ``Localizations.lp_lateral_nm`` marks them -- are linked like any
    other localization, so the event keeps its frames, but get no say in
    its position. They are counted in ``n_discarded``. An event made only
    of failed fits has no position and is dropped (``n_no_position``).
    Localizations without coordinates are not linked (``n_unplaced``).
    Picasso does otherwise:
    it keeps a zero precision, the infinite weight makes the event's
    position NaN, and saving then deletes the whole event -- mostly long
    ones (55 events averaging 20 localizations in NCtransversal_Roi2_2_1),
    which biases tau_bright low and merges the dark times on either side.

    ``remove_ambiguous_lengths`` drops events that were already bound in
    the first frame or still bound in the last. Their length is CENSORED:
    the recorded value is a lower bound, not a measurement, and keeping
    them biases tau_bright low. The last-frame half needs ``n_frames``;
    without it a warning says it was skipped. Picasso applies the same
    rule, but its end test (``last < Frames``) is always true for frames
    counted from 0, so it only ever trims the start.
    """
    frame = np.asarray(frame, dtype=np.int64)
    x_nm = np.asarray(x_nm, dtype=float)
    y_nm = np.asarray(y_nm, dtype=float)
    n_input = frame.size
    # Only a localization without a position cannot be linked at all.
    placed = np.isfinite(x_nm) & np.isfinite(y_nm)
    if lp_nm is None:
        fitted = placed.copy()
    else:
        precision = np.asarray(lp_nm, dtype=float)
        fitted = placed & np.isfinite(precision) & (precision > 0)
    n_unplaced = int(n_input - np.count_nonzero(placed))
    n_discarded = int(np.count_nonzero(placed & ~fitted))

    keep_idx = np.nonzero(placed)[0]

    def placed_only(values: Any) -> Any:
        return None if values is None else np.asarray(values)[placed]

    f = frame[placed]
    x = x_nm[placed]
    y = y_nm[placed]
    z = placed_only(z_nm)
    ph = placed_only(photons)
    g = placed_only(group)
    link = link_localizations(f, x, y, radius_nm, max_dark_time, g)
    # link_group is reported for the ORIGINAL input, with -1 where a
    # localization had no position, so a caller can still map back.
    full_link = np.full(n_input, -1, dtype=np.int64)
    full_link[keep_idx] = link
    n_events = int(link.max()) + 1 if link.size else 0

    if lp_nm is None:
        weights = np.ones(link.size, dtype=float)
    else:
        lp = np.asarray(lp_nm, dtype=float)[placed]
        good = fitted[placed]
        weights = np.zeros(link.size, dtype=float)
        weights[good] = 1.0 / lp[good] ** 2
    weight_sum = np.bincount(link, weights=weights, minlength=n_events)
    has_position = weight_sum > 0
    divisor = np.where(has_position, weight_sum, 1.0)

    def weighted(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.bincount(
            link, weights=np.asarray(values, dtype=float) * weights,
            minlength=n_events,
        ) / divisor

    counts = np.bincount(link, minlength=n_events).astype(np.int64)
    first = np.full(n_events, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, link, f)
    last = np.full(n_events, np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(last, link, f)
    length = (last - first + 1).astype(np.int64)

    total_photons: Optional[NDArray[np.float64]] = (
        None if ph is None
        else np.asarray(
            np.bincount(link, weights=np.asarray(ph, dtype=float),
                        minlength=n_events),
            dtype=np.float64,
        )
    )
    rate = None if total_photons is None else total_photons / np.maximum(
        length, 1
    )
    event_group = None
    if g is not None:
        # Scatter-assign: every localization of an event writes its own
        # label into the event's slot. Safe because linking never lets an
        # event span two groups, so they are all writing the same value.
        event_group = np.zeros(n_events, dtype=np.int64)
        event_group[link] = np.asarray(g, dtype=np.int64)

    # The link_group column keeps the ORIGINAL event numbering, so a
    # localization can still be traced back; the arrays below hold the
    # surviving events only.
    uncensored = np.ones(n_events, dtype=bool)
    if remove_ambiguous_lengths:
        uncensored &= first > 0
        if n_frames is not None:
            uncensored &= last < int(n_frames) - 1
        else:
            warnings_module.warn(
                "acquisition length unknown: events still bound in the last "
                "frame were kept, so tau_bright is biased low",
                stacklevel=2,
            )
    keep = has_position & uncensored

    return BindingEvents(
        x_nm=weighted(x)[keep],
        y_nm=weighted(y)[keep],
        z_nm=None if z is None else weighted(z)[keep],
        first_frame=first[keep],
        last_frame=last[keep],
        length=length[keep],
        n_locs=counts[keep],
        photons=None if total_photons is None else total_photons[keep],
        photon_rate=None if rate is None else rate[keep],
        group=None if event_group is None else event_group[keep],
        link_group=full_link,
        radius_nm=float(radius_nm),
        max_dark_time=int(max_dark_time),
        n_discarded=n_discarded,
        n_censored=int(np.count_nonzero(has_position & ~uncensored)),
        n_no_position=int(np.count_nonzero(~has_position)),
        n_unplaced=n_unplaced,
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


# A dark period this many frames beyond max_dark_time is almost certainly
# the same binding event flickering rather than a genuine unbinding and
# rebinding: at a site whose real dark time is hundreds of frames, an
# imager leaving and returning within three frames is vanishingly rare.
FRAGMENTATION_MARGIN = 2

# How many times more short gaps than a single exponential predicts before
# the distribution is called fragmented. Self-calibrating: the prediction
# comes from the fit to the long gaps themselves, so no absolute dark time
# has to be assumed. An excess of 3x is already far outside what sampling
# noise produces on a real exponential.
FRAGMENTATION_EXCESS = 3.0

# ...but only when the short gaps are numerous enough to matter. A handful
# of them can exceed the prediction by a large factor and change nothing.
FRAGMENTATION_MIN_FRACTION = 0.10


@dataclass
class Kinetics:
    tau_bright_frames: float
    tau_dark_frames: float
    tau_bright_s: Optional[float]
    tau_dark_s: Optional[float]
    n_events: int
    n_dark: int
    n_sites: int
    # The same fit with the very short gaps left out. When it disagrees
    # with tau_dark_frames, the events are being fragmented and the
    # headline number is measuring the fragmentation -- see
    # ``short_gap_fraction`` and the warning that goes with it.
    tau_dark_excluding_short_frames: float = float("nan")
    short_gap_fraction: float = float("nan")
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
        # Keyword arguments, not positional: two fields were inserted
        # before `warnings` when the fragmentation check was added, and a
        # positional call silently put the warning list into a float field
        # instead. The same mistake already cost this project once, in
        # intraclass_correlation.
        return Kinetics(
            tau_bright_frames=float("nan"),
            tau_dark_frames=float("nan"),
            tau_bright_s=None,
            tau_dark_s=None,
            n_events=0,
            n_dark=0,
            n_sites=0,
            warnings=["no binding events"],
        )

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
            tau_bright_frames=estimate_kinetic_rate(
                events.length.astype(float)
            ),
            tau_dark_frames=float("nan"),
            tau_bright_s=None,
            tau_dark_s=None,
            n_events=events.n,
            n_dark=0,
            n_sites=0,
            warnings=warnings,
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

    # Is tau_dark describing the binding chemistry, or the linking?
    #
    # A binding event whose imager is missed for more than max_dark_time
    # frames is split into two events, and the gap between the halves
    # enters the dark-time distribution as a very short period. Those
    # pile up near zero and, being far more numerous than the real dark
    # periods, they drag the cumulative fit down onto themselves. Since
    # tau_dark IS the qPAINT measurement, the resulting count is wrong by
    # whatever factor the fit moved -- on this project's own DNA-PAINT
    # data, 8 frames against a real 429.
    cutoff = events.max_dark_time + FRAGMENTATION_MARGIN
    short_fraction = (
        float(np.mean(observed <= cutoff)) if observed.size else float("nan")
    )
    long_gaps = observed[observed > cutoff]
    tau_dark_long = (
        estimate_kinetic_rate(long_gaps.astype(float))
        if long_gaps.size > 2 else float("nan")
    )
    # The test is whether the short gaps are more numerous than a single
    # exponential predicts -- but it has to be asked ONE CLUSTER AT A
    # TIME. A cluster of N docking sites has dark times that are
    # exponential with rate N times the single-site rate, so pooling
    # clusters of different N gives a MIXTURE of exponentials, which
    # always shows an excess of short gaps whether or not anything is
    # wrong. Asked of the pooled distribution this check fires on every
    # real dataset; asked per cluster it fires only when that cluster's
    # own dark times are not exponential, which is what fragmentation
    # looks like.
    fragmented, examined = 0, 0
    for label in np.unique(np.asarray(labels)[real_site]):
        per_site = dark[(np.asarray(labels) == label) & (dark > 0)]
        if per_site.size < 20:
            continue
        tail = per_site[per_site > cutoff]
        if tail.size < 10:
            continue
        tau_site = estimate_kinetic_rate(tail.astype(float))
        if not np.isfinite(tau_site) or tau_site <= 0:
            continue
        examined += 1
        observed_short = float(np.mean(per_site <= cutoff))
        predicted = 1.0 - float(np.exp(-cutoff / tau_site))
        if (
            observed_short > FRAGMENTATION_MIN_FRACTION
            and predicted > 0
            and observed_short / predicted > FRAGMENTATION_EXCESS
        ):
            fragmented += 1

    if examined and fragmented / examined > 0.5:
        warnings.append(
            f"{fragmented} of {examined} clusters have far more very short "
            f"dark periods ({cutoff} frames or fewer) than their OWN binding "
            f"rate predicts, so their dark times are not a single "
            f"exponential and tau_dark does not summarise them. Pooled, the "
            f"short gaps give {tau_dark:.0f} frames against "
            f"{tau_dark_long:.0f} without them, and every qPAINT count "
            f"scales with that. Three things produce it: one binding event "
            f"split in two by the linking (raise 'Max dark frames', or the "
            f"link radius if events break up in space rather than in time); "
            f"a fluorophore blinking while still bound; or clusters that are "
            f"arbitrary chunks of a continuous structure rather than "
            f"discrete docking sites, in which case qPAINT does not apply to "
            f"them at all."
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
        tau_dark_excluding_short_frames=tau_dark_long,
        short_gap_fraction=short_fraction,
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
    min_dark_frames: int = 0,
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

    ``min_dark_frames`` leaves dark periods at or below that many frames
    out of each fit. Use it when ``kinetics`` reports event
    fragmentation: those short gaps are one binding event split in two by
    the linking, and counting them as real unbindings drags tau_dark down
    and every count up by the same factor. 0 (the default) keeps
    everything.
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
        observed = dark[inside & (dark > min_dark_frames)]
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
    link_warnings: List[str] = field(default_factory=list)

    @property
    def compression(self) -> float:
        """Localizations per binding event; 1.0 means nothing linked."""
        if self.events is None or self.events.n == 0:
            return float("nan")
        return float(np.mean(self.events.n_locs))

    @property
    def warnings(self) -> List[str]:
        out: List[str] = list(self.link_warnings)
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

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        report.events = build_events(
            frame, loc.x_nm, loc.y_nm,
            z_nm=loc.z_nm if loc.is_3d else None,
            photons=loc.photons,
            lp_nm=lp,
            group=site_labels,
            radius_nm=report.radius_nm,
            max_dark_time=max_dark_time,
            n_frames=loc.n_frames,
        )
    report.link_warnings.extend(str(w.message) for w in caught)

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
