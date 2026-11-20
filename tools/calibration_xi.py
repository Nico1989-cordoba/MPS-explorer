# -*- coding: utf-8 -*-
"""
Calibration of xi: binding events per isolated label, from a sparse control.

xi is the distribution of the number of localizations (here: frame-linked
EVENTS) one fluorescent label leaves in a movie. The emitters layer
("Emitters (BaGoL-like reimplementation)"; Fazel et al. 2022) needs it as a
prior, and Fazel et al. foresee measuring it on "a dedicated control
experiment where the sample is known to be at low labeling fraction". This
module is the analysis of that control, as laid out in the lab's calibration
protocol (sections F6, F7, F11 and F12): secondary antibodies adsorbed
sparsely on glass, or isolated labels in a control section, plus an
optional sample without antibody for the background.

Steps, in order
---------------
1. LINK localizations into events with ``tools.mps_paint.build_events`` --
   the same function, and it must be the same radius and dark-frame
   tolerance as the tissue data, because ``calibrated_prior`` expects
   "localizations per emitter after the same linking and filtering as the
   data". Censored events (bound in the first or last frame) are KEPT: here
   an event is counted, its length is not used.
2. GROUP events into marks: single linkage at ``group_radius_nm`` (DBSCAN
   with min_samples = 1 is the same partition). Default radius derived from
   the precision: ``MARK_RADIUS_IN_SIGMAS * sqrt(lp_median^2 + extra^2)``.
3. ISOLATE: a mark counts only with no other mark within
   ``isolation_nm`` (0.5 um), at least ``border_nm`` (1 um) from the field
   edge, and not elongated (extent <= 2 * group radius) [protocol F6, F11.4;
   ad hoc there].
4. BACKGROUND: the control is processed identically. Its density of groups
   per um^2 times the analysed area times the fraction of groups that pass
   isolation gives the expected number of false marks among the isolated
   ones, with the control's own distribution of events per group. It enters
   the fit as a known mixture weight; the protocol's alternative (fit only
   counts >= n_min, where background is < 5 % of marks) is reported too.
5. FIT the counts by maximum likelihood: a zero-truncated, rounding-
   discretized gamma (k, theta), which is what the emitters layer evaluates
   (``gampdf`` at integer N; BaGoL_RJMCMC.m), its k = 1 special case (a
   geometric = discretized exponential), and as a check a zero-truncated
   negative binomial. The end of the movie: the "recorded" estimand
   ignores it (it is what a movie of this length gives, and the right one
   when tissue movies have the same length); the "censored" estimand
   estimates the count until bleaching, each mark contributing
   P(N = n) + s(g) P(N > n), with s(g) the empirical share of dark gaps
   inside isolated marks longer than the g frames between its last event
   and the end.
6. UNCERTAINTY: bootstrap by session, then field within session, then mark
   within field. With several sessions, the mean of the session means and
   their SD are reported too (protocol F8: the between-session spread is the
   main uncertainty).
7. QUALITY checks from the protocol's table F12 that can be computed from
   the localizations (density, aggregates, background share, drift, marks
   active at the end, number of isolated marks) plus pH and laser power
   when the session metadata carries them.

Output: one CSV row per mark, a JSON summary with the ready-to-use
arguments of ``tools.emitters_prior.calibrated_prior``, and a PNG figure.

Origin labels of every constant follow the project rule: [lit: ...],
[deriv], [ad hoc], [simulation].

Never run this on someone else's data without being asked to; the tests use
synthetic data only.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Constants, each with its origin
# ---------------------------------------------------------------------------
# Two positions of one mark, each with per-axis sd sigma, differ by a
# Rayleigh distance of scale sqrt(2)*sigma; its 99.9 % quantile is
# sqrt(2) * sqrt(2 ln 1000) * sigma = 5.26 sigma.  [deriv]
MARK_RADIUS_IN_SIGMAS = math.sqrt(2.0) * math.sqrt(2.0 * math.log(1000.0))
# Spread of one label beyond the localization precision: antibody size and
# residual drift. No measurement behind it. [ad hoc]
EXTRA_SIGMA_NM = 10.0
# The protocol's fixed grouping radius (F11.3), offered as an alternative
# and as a sensitivity check. [ad hoc, protocol F11.3]
PROTOCOL_GROUP_RADIUS_NM = 150.0
# Isolation and border rules (protocol F6, F11.4). [ad hoc]
ISOLATION_NM = 500.0
BORDER_NM = 1000.0
# Elongated = extent > ELONGATION_FACTOR * group radius (F11.4). [ad hoc]
ELONGATION_FACTOR = 2.0
# n_min rule: background expected < 5 % of the marks with that count (F11.5). [ad hoc]
N_MIN_BACKGROUND_SHARE = 0.05
# Pump-frame versions reported when the user gives none (F7). [ad hoc]
DEFAULT_PUMP_FRAMES: Tuple[int, ...] = (0, 200, 1000)
# Main estimate uses marks with >= 70 % of peak irradiance (F9). [ad hoc, after Lin et al. 2015]
MIN_REL_IRRADIANCE = 0.70
IRRADIANCE_BANDS: Tuple[Tuple[float, float], ...] = ((0.85, 1.01), (0.70, 0.85), (0.0, 0.70))
# Quality thresholds, protocol table F12. All [ad hoc].
QC_DENSITY_RANGE = (0.02, 0.20)          # marks per um^2
QC_AGGREGATE_EXCESS = 2.0                # observed / expected short 1NN
QC_BACKGROUND_SHARE = 0.10               # expected false / isolated marks
QC_DRIFT_NM = 100.0
QC_MIN_ISOLATED = 100                    # per session and dye
QC_PH_START = (7.8, 8.2)
QC_PH_DROP = 0.3
QC_PH_FLOOR = 7.5
QC_POWER_CHANGE = 0.10
# Marks active at the end: above this share the "recorded" count is mostly
# a lower bound and the censored estimand should be looked at. [ad hoc]
QC_ACTIVE_AT_END = 0.20
# Drift is estimated from marks with at least this many events. [ad hoc]
DRIFT_MIN_EVENTS = 4
DEFAULT_BOOTSTRAP = 1000
DEFAULT_SEED = 20261016

METHOD_LABEL = "Emitters (BaGoL-like reimplementation; Fazel et al. 2022)"
TABLE_VERSION = "calibration_xi v1"

# Okabe-Ito hex values, the same as tools/mps_plot_style.OKABE_ITO (not
# imported: that module needs Qt, this one runs headless).
_C_OBSERVED = "#0072b2"
_C_GAMMA = "#e69f00"
_C_GEOM = "#009e73"
_C_BACKGROUND = "#d55e00"
_C_GREY = "#7f7f7f"


# ===========================================================================
#  Data containers
# ===========================================================================
@dataclass
class FieldData:
    """One movie (one field of view) as arrays in nanometres and frames."""

    frame: NDArray[np.int64]
    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    lp_nm: Optional[NDArray[np.float64]] = None
    photons: Optional[NDArray[np.float64]] = None
    z_nm: Optional[NDArray[np.float64]] = None
    n_frames: Optional[int] = None
    session: str = "S1"
    name: str = "field"
    # (xmin, xmax, ymin, ymax) in nm; None = bounding box of the localizations.
    bounds_nm: Optional[Tuple[float, float, float, float]] = None
    # Per-movie metadata (ph_start, ph_end, power_start_mw, power_end_mw, ...).
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolved_bounds(self) -> Tuple[float, float, float, float]:
        if self.bounds_nm is not None:
            return tuple(float(v) for v in self.bounds_nm)  # type: ignore[return-value]
        ok = np.isfinite(self.x_nm) & np.isfinite(self.y_nm)
        if not ok.any():
            return (0.0, 0.0, 0.0, 0.0)
        return (float(np.min(self.x_nm[ok])), float(np.max(self.x_nm[ok])),
                float(np.min(self.y_nm[ok])), float(np.max(self.y_nm[ok])))

    def resolved_n_frames(self) -> int:
        if self.n_frames is not None:
            return int(self.n_frames)
        return int(np.max(self.frame)) + 1 if self.frame.size else 0


@dataclass
class CalibrationParams:
    """Every analysis choice, so it travels to the JSON."""

    # The linking of the tissue data, both required (decision of the user 2026-10-17): the prior must count events
    # exactly as the tissue is linked (a radius of 3 lp splits ~10 % of the cycles, +5 % events per label).
    link_radius_nm: float
    max_dark_frames: int
    group_radius_nm: Optional[float] = None
    extra_sigma_nm: float = EXTRA_SIGMA_NM
    isolation_nm: float = ISOLATION_NM
    border_nm: float = BORDER_NM
    elongation_factor: float = ELONGATION_FACTOR
    use_z_for_isolation: bool = False
    pump_frames: int = 0
    pump_frames_versions: Tuple[int, ...] = DEFAULT_PUMP_FRAMES
    end_window_frames: Optional[int] = None
    estimand: str = "recorded"           # "recorded" | "censored"
    background: str = "mixture"          # "mixture" | "n_min" | "none"
    beam_centre_nm: Optional[Tuple[float, float]] = None
    beam_sigma_nm: Optional[float] = None
    min_rel_irradiance: float = MIN_REL_IRRADIANCE
    n_bootstrap: int = DEFAULT_BOOTSTRAP
    seed: int = DEFAULT_SEED

    def validate(self) -> None:
        if not (self.link_radius_nm > 0 and math.isfinite(self.link_radius_nm)):
            raise ValueError("link_radius_nm must be > 0: use the SAME value as for the tissue data")
        if self.max_dark_frames < 0:
            raise ValueError("max_dark_frames must be >= 0")
        if self.estimand not in ("recorded", "censored"):
            raise ValueError("estimand must be 'recorded' or 'censored'")
        if self.background not in ("mixture", "n_min", "none"):
            raise ValueError("background must be 'mixture', 'n_min' or 'none'")
        if self.group_radius_nm is not None and not self.group_radius_nm > 0:
            raise ValueError("group_radius_nm must be > 0")


# ===========================================================================
#  1. Linking (reuses the program's linker) and grouping
# ===========================================================================
def link_events(fd: FieldData, params: CalibrationParams) -> Any:
    """Frame-link with ``tools.mps_paint.build_events``; censored events are kept."""
    from tools.mps_paint import build_events
    return build_events(
        fd.frame, fd.x_nm, fd.y_nm,
        z_nm=fd.z_nm, photons=fd.photons, lp_nm=fd.lp_nm,
        radius_nm=float(params.link_radius_nm),
        max_dark_time=int(params.max_dark_frames),
        n_frames=fd.resolved_n_frames(),
        # An event bound in the first or last frame is still one event of
        # its mark; only its LENGTH would be censored, and lengths are not used.
        remove_ambiguous_lengths=False,
    )


def derived_group_radius(lp_nm: Optional[NDArray[np.float64]], extra_sigma_nm: float = EXTRA_SIGMA_NM) -> float:
    """``MARK_RADIUS_IN_SIGMAS * sqrt(median lp^2 + extra^2)`` [deriv; extra ad hoc]."""
    lp_med = 0.0
    if lp_nm is not None:
        good = np.asarray(lp_nm, dtype=float)
        good = good[np.isfinite(good) & (good > 0)]
        if good.size:
            lp_med = float(np.median(good))
    if lp_med == 0.0 and extra_sigma_nm <= 0:
        raise ValueError("no precision column and extra_sigma_nm = 0: give group_radius_nm")
    return MARK_RADIUS_IN_SIGMAS * math.sqrt(lp_med ** 2 + float(extra_sigma_nm) ** 2)


def single_linkage_labels(points: NDArray[np.float64], radius: float) -> NDArray[np.int64]:
    """Connected components of the graph 'closer than radius' (= DBSCAN, min_samples 1)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = points.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(float(radius), output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
        coo_matrix((n, n))
    _, labels = connected_components(graph, directed=False)
    # Relabel in order of first appearance, so the numbering is deterministic.
    _, first = np.unique(labels, return_index=True)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    return np.asarray(remap[labels], dtype=np.int64)


# ===========================================================================
#  2. Marks: one row per group of events
# ===========================================================================
MARK_COLUMNS: Tuple[str, ...] = (
    "session", "field", "mark_id", "x_nm", "y_nm", "z_nm", "n_events", "n_locs", "first_frame", "last_frame",
    "extent_nm", "nn_nm", "border_nm", "rel_irradiance", "end_gap_frames", "p_silent_if_active", "active_at_end",
    "isolated", "in_main", "exclusion",
)


def _mark_table(fd: FieldData, params: CalibrationParams, group_radius: float,
                end_window: Optional[int]) -> Tuple[Dict[str, NDArray[Any]], Any]:
    """Link, group and measure one field. ``end_window`` None = decide later."""
    ev = link_events(fd, params)
    n_ev = ev.n
    pts = np.column_stack([ev.x_nm, ev.y_nm]) if n_ev else np.empty((0, 2))
    lab = single_linkage_labels(pts, group_radius)
    n_marks = int(lab.max()) + 1 if lab.size else 0
    cnt = np.bincount(lab, minlength=n_marks).astype(np.int64)
    nloc = np.bincount(lab, weights=ev.n_locs.astype(float), minlength=n_marks).astype(np.int64)
    div = np.maximum(cnt, 1)
    cx = np.bincount(lab, weights=ev.x_nm, minlength=n_marks) / div
    cy = np.bincount(lab, weights=ev.y_nm, minlength=n_marks) / div
    if ev.z_nm is not None:
        cz = np.bincount(lab, weights=np.nan_to_num(ev.z_nm), minlength=n_marks) / div
    else:
        cz = np.full(n_marks, np.nan)
    first = np.full(n_marks, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, lab, ev.first_frame)
    last = np.full(n_marks, np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(last, lab, ev.last_frame)
    rad = np.hypot(ev.x_nm - cx[lab], ev.y_nm - cy[lab]) if n_ev else np.empty(0)
    extent = np.zeros(n_marks)
    np.maximum.at(extent, lab, 2.0 * rad)

    # Nearest other mark (every group, background included).
    from scipy.spatial import cKDTree
    if params.use_z_for_isolation and ev.z_nm is not None:
        centres = np.column_stack([cx, cy, cz])
    else:
        centres = np.column_stack([cx, cy])
    if n_marks >= 2:
        d, _ = cKDTree(centres).query(centres, k=2)
        nn = d[:, 1]
    else:
        nn = np.full(n_marks, np.inf)
    xmin, xmax, ymin, ymax = fd.resolved_bounds()
    border = np.minimum.reduce([cx - xmin, xmax - cx, cy - ymin, ymax - cy]) if n_marks else np.empty(0)
    if params.beam_centre_nm is not None and params.beam_sigma_nm:
        r2 = (cx - params.beam_centre_nm[0]) ** 2 + (cy - params.beam_centre_nm[1]) ** 2
        rel = np.exp(-r2 / (2.0 * float(params.beam_sigma_nm) ** 2))
    else:
        rel = np.full(n_marks, np.nan)

    table: Dict[str, NDArray[Any]] = {
        "mark_id": np.arange(n_marks, dtype=np.int64), "x_nm": cx, "y_nm": cy, "z_nm": cz,
        "n_events": cnt, "n_locs": nloc, "first_frame": first, "last_frame": last, "extent_nm": extent,
        "nn_nm": nn, "border_nm": border, "rel_irradiance": rel,
        "_event_label": lab,
    }
    return table, ev


def _classify(table: Dict[str, NDArray[Any]], params: CalibrationParams, group_radius: float) -> None:
    n = table["n_events"].size
    reason = np.array([""] * n, dtype=object)
    near = table["nn_nm"] < params.isolation_nm
    edge = table["border_nm"] < params.border_nm
    long_ = table["extent_nm"] > params.elongation_factor * group_radius
    for mask, why in ((long_, "elongated"), (edge, "border"), (near, "neighbour")):
        reason[mask] = why
    isolated = ~(near | edge | long_)
    in_main = isolated.copy()
    if params.beam_centre_nm is not None and params.beam_sigma_nm:
        dim = table["rel_irradiance"] < params.min_rel_irradiance
        reason[isolated & dim] = "irradiance"
        in_main &= ~dim
    table["isolated"] = isolated
    table["in_main"] = in_main
    table["exclusion"] = reason


def _inner_area_um2(fd: FieldData, border_nm: float) -> float:
    xmin, xmax, ymin, ymax = fd.resolved_bounds()
    w = max(0.0, (xmax - xmin) - 2 * border_nm)
    h = max(0.0, (ymax - ymin) - 2 * border_nm)
    return w * h / 1e6


def _events_after(ev: Any, label: NDArray[np.int64], n_marks: int, pump: int) -> NDArray[np.int64]:
    """Events per mark whose first frame is >= pump."""
    keep = ev.first_frame >= int(pump)
    return np.bincount(label[keep], minlength=n_marks).astype(np.int64)


def _off_gaps(ev: Any, label: NDArray[np.int64]) -> NDArray[np.float64]:
    """Dark frames between the end of one event and the start of the next, inside each mark."""
    if label.size < 2:
        return np.empty(0)
    order = np.lexsort((ev.first_frame, label))
    lab_s, f_s, l_s = label[order], ev.first_frame[order], ev.last_frame[order]
    same = lab_s[1:] == lab_s[:-1]
    return np.asarray((f_s[1:] - l_s[:-1])[same], dtype=float)


# ===========================================================================
#  3. Count models and maximum likelihood
# ===========================================================================
MODELS: Tuple[str, ...] = ("geometric", "gamma", "negbin")
N_PARAMS = {"geometric": 1, "gamma": 2, "negbin": 2}


def _model_tables(model: str, theta_vec: NDArray[np.float64], n_max: int
                  ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    pmf q(n) and tail T(n) = P(N > n), n = 0..n_max, zero-truncated.

    gamma / geometric: N = round(X), X ~ Gamma(k, theta), conditioned on N >= 1
    [deriv: BaGoL evaluates gampdf at integer N, ~ F(n+1/2) - F(n-1/2)].
    negbin: Gamma-Poisson with shape r and mean mu, conditioned on N >= 1.
    """
    from scipy import special, stats
    n = np.arange(n_max + 2, dtype=float)
    if model in ("gamma", "geometric"):
        if model == "geometric":
            k, scale = 1.0, float(np.exp(theta_vec[0]))
        else:
            k, scale = float(np.exp(theta_vec[0])), float(np.exp(theta_vec[1]))
        upper = special.gammaincc(k, np.maximum(n - 0.5, 0.0) / scale)   # P(X >= n - 1/2)
        upper[0] = 1.0
        z = upper[1]                                                      # P(N >= 1)
        tail = special.gammaincc(k, (n + 0.5) / scale)                    # P(N > n)
        pmf = upper - tail
        pmf[0] = 0.0
        tail = np.minimum(tail / z, 1.0)
        pmf = pmf / z
    else:
        r, mu = float(np.exp(theta_vec[0])), float(np.exp(theta_vec[1]))
        p = r / (r + mu)
        dist = stats.nbinom(r, p)
        z = 1.0 - float(dist.pmf(0))
        pmf = np.asarray(dist.pmf(n), dtype=float) / z
        pmf[0] = 0.0
        tail = np.asarray(dist.sf(n), dtype=float) / z
    return pmf[: n_max + 1], tail[: n_max + 1]


@dataclass
class Background:
    """Known contamination of the isolated marks by background groups."""

    weight: float                                # expected false / isolated marks
    pmf: NDArray[np.float64]                     # events per background group, index = n
    expected_false: float = 0.0
    density_per_um2: float = 0.0
    n_groups: int = 0
    area_um2: float = 0.0

    def tables(self, n_max: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        pmf = np.zeros(n_max + 1)
        m = min(n_max + 1, self.pmf.size)
        pmf[:m] = self.pmf[:m]
        tail = 1.0 - np.cumsum(pmf)
        return pmf, np.clip(tail, 0.0, 1.0)


@dataclass
class FitResult:
    model: str
    k: float
    theta: float
    mean: float                 # k * theta of the continuous gamma (negbin: mu)
    loglik: float
    aic: float
    n_marks: int
    n_censored: int
    n_min: int
    converged: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "k": self.k, "theta": self.theta, "mean": self.mean,
                "loglik": self.loglik, "aic": self.aic, "n_marks": self.n_marks,
                "n_censored": self.n_censored, "n_min": self.n_min, "converged": self.converged}


def _histograms(counts: NDArray[np.int64], censored: NDArray[np.bool_], n_min: int
                ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    counts = np.asarray(counts, dtype=np.int64)
    keep = counts >= max(int(n_min), 1)
    c = counts[keep]
    cen = np.asarray(censored, dtype=bool)[keep]
    n_max = int(c.max()) if c.size else 1
    h_u = np.bincount(c[~cen], minlength=n_max + 1).astype(float)
    h_c = np.bincount(c[cen], minlength=n_max + 1).astype(float)
    return h_u, h_c


def _negloglik(theta_vec: NDArray[np.float64], model: str, h_u: NDArray[np.float64], h_c: NDArray[np.float64],
               n_min: int, bg: Optional[Background], soft_n: Optional[NDArray[np.int64]] = None,
               soft_s: Optional[NDArray[np.float64]] = None) -> float:
    n_max = h_u.size - 1
    try:
        pmf, tail = _model_tables(model, theta_vec, n_max)
    except (ValueError, FloatingPointError, OverflowError):
        return 1e300
    if n_min > 1:
        # Left truncation at n_min: renormalize on N >= n_min.
        norm = tail[n_min - 1]
        if not norm > 0:
            return 1e300
        pmf = pmf / norm
        tail = tail / norm
    elif bg is not None and bg.weight > 0:
        bpmf, btail = bg.tables(n_max)
        w = min(max(bg.weight, 0.0), 0.99)
        pmf = (1 - w) * pmf + w * bpmf
        tail = (1 - w) * tail + w * btail
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(h_u > 0, np.log(np.maximum(pmf, 1e-300)), 0.0)
        lt = np.where(h_c > 0, np.log(np.maximum(tail, 1e-300)), 0.0)
    val = -(float(np.dot(h_u, lp)) + float(np.dot(h_c, lt)))
    if soft_n is not None and soft_s is not None and soft_n.size:
        # P(exactly n seen) = P(N = n) + P(N > n) * P(no event in the rest of the movie | still active)
        val -= float(np.sum(np.log(np.maximum(pmf[soft_n] + soft_s * tail[soft_n], 1e-300))))
    return val if math.isfinite(val) else 1e300


def fit_counts(counts: ArrayLike, censored: Optional[ArrayLike] = None, *, model: str = "gamma",
               background: Optional[Background] = None, n_min: int = 1,
               start: Optional[Tuple[float, float]] = None, survival: Optional[ArrayLike] = None) -> FitResult:
    """
    Maximum-likelihood fit of events per mark.

    ``censored`` marks, KNOWN to be still active at the end, contribute
    P(N > n). ``survival`` is the soft version used by the pipeline: per
    mark, the probability that a still-active label would have produced
    no event between its last event and the end of the movie; the mark
    then contributes P(N = n) + s * P(N > n) [deriv]. All zeros (or
    neither argument) is the "recorded" estimand. ``background`` (mixture
    with a known weight) is ignored when ``n_min`` > 1, which is the
    protocol's alternative correction. ``start`` = (k, mean).
    """
    from scipy import optimize
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    c = np.asarray(counts, dtype=np.int64)
    cen = np.zeros(c.size, dtype=bool) if censored is None else np.asarray(censored, dtype=bool)
    if c.size == 0 or np.any(c < 1):
        raise ValueError("counts must be >= 1 and not empty")
    soft_n: Optional[NDArray[np.int64]] = None
    soft_s: Optional[NDArray[np.float64]] = None
    if survival is not None:
        s_all = np.clip(np.asarray(survival, dtype=float), 0.0, 1.0)
        if s_all.shape != c.shape:
            raise ValueError("survival needs one value per mark")
        soft = (s_all > 0) & ~cen & (c >= max(n_min, 1))
        soft_n, soft_s = c[soft], s_all[soft]
        h_u, h_c = _histograms(c[~soft], cen[~soft], n_min)
        size = max(h_u.size, int(soft_n.max()) + 1 if soft_n.size else 0)
        h_u, h_c = np.pad(h_u, (0, size - h_u.size)), np.pad(h_c, (0, size - h_c.size))
    else:
        h_u, h_c = _histograms(c, cen, n_min)
    n_used = int(h_u.sum() + h_c.sum()) + (0 if soft_n is None else int(soft_n.size))
    if n_used == 0:
        raise ValueError(f"no mark with at least n_min = {n_min} events")
    kept = c[c >= max(n_min, 1)]
    m0 = max(float(np.mean(kept)), 1.01)
    v0 = float(np.var(kept)) if kept.size > 1 else m0 ** 2
    k0 = min(max(m0 ** 2 / max(v0, 1e-9), 0.2), 50.0)
    if start is not None:
        k0, m0 = max(start[0], 0.05), max(start[1], 0.5)
    if model == "geometric":
        x0 = np.array([math.log(m0)])
    elif model == "gamma":
        x0 = np.array([math.log(k0), math.log(m0 / k0)])
    else:
        x0 = np.array([math.log(k0), math.log(m0)])
    res = optimize.minimize(_negloglik, x0, args=(model, h_u, h_c, n_min, background, soft_n, soft_s),
                            method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000, "maxfev": 8000})
    x = np.asarray(res.x, dtype=float)
    if model == "geometric":
        k, theta = 1.0, float(np.exp(x[0]))
        mean = theta
    elif model == "gamma":
        k, theta = float(np.exp(x[0])), float(np.exp(x[1]))
        mean = k * theta
    else:
        k, mean = float(np.exp(x[0])), float(np.exp(x[1]))
        theta = mean / k
    ll = -float(res.fun)
    return FitResult(model=model, k=k, theta=theta, mean=mean, loglik=ll, aic=2 * N_PARAMS[model] - 2 * ll,
                     n_marks=n_used, n_censored=int(h_c.sum()) + (0 if soft_s is None else int(np.sum(soft_s > 0.5))),
                     n_min=int(n_min), converged=bool(res.success))


def n_min_from_background(counts: ArrayLike, bg: Optional[Background],
                          share: float = N_MIN_BACKGROUND_SHARE) -> int:
    """Smallest n from which the expected background is < ``share`` of the marks at every count (F11.5)."""
    if bg is None or bg.expected_false <= 0:
        return 1
    c = np.asarray(counts, dtype=np.int64)
    n_max = int(c.max()) if c.size else 1
    obs = np.bincount(c, minlength=n_max + 1).astype(float)
    exp_bg = bg.expected_false * bg.tables(n_max)[0]
    bad = [n for n in range(1, n_max + 1) if exp_bg[n] >= share * max(obs[n], 1e-12) and exp_bg[n] > 0]
    return (max(bad) + 1) if bad else 1


# ===========================================================================
#  4. Background from the control
# ===========================================================================
def background_from_controls(control_tables: Sequence[Tuple[Dict[str, NDArray[Any]], float]],
                             counts_key: str = "n_events") -> Tuple[float, NDArray[np.float64], int, float]:
    """
    (density per um^2, pmf of events per group, n groups, area um^2) over
    the inner area of the controls. Each table needs an ``_inner`` mask
    (at least ``border_nm`` from the edge) and the ``counts_key`` column.
    """
    n_groups, area = 0, 0.0
    hist: NDArray[np.float64] = np.zeros(2)
    for table, inner_area in control_tables:
        sel = np.asarray(table["_inner"], dtype=bool)
        cnt = np.asarray(table[counts_key])[sel]
        cnt = cnt[cnt > 0]
        n_groups += int(cnt.size)
        area += inner_area
        if cnt.size:
            h = np.bincount(cnt).astype(float)
            if h.size > hist.size:
                hist = np.pad(hist, (0, h.size - hist.size))
            hist[: h.size] += h
    pmf = hist / hist.sum() if hist.sum() > 0 else hist
    density = n_groups / area if area > 0 else 0.0
    return density, pmf, n_groups, area


# ===========================================================================
#  5. The whole analysis
# ===========================================================================
@dataclass
class CalibrationResult:
    summary: Dict[str, Any]
    marks: List[Dict[str, Any]]
    hist_counts: NDArray[np.int64]
    fits: Dict[str, FitResult]
    background: Optional[Background]
    survival: List[Tuple[int, float]]

    def calibrated_prior_args(self) -> Dict[str, Any]:
        return dict(self.summary["calibrated_prior_args"])


def _resample_background(bg: Optional[Background], rng: np.random.Generator,
                          n_marks: int) -> Optional[Background]:
    """
    The control's uncertainty in one bootstrap replicate: its number of
    groups redrawn as Poisson, their events per group as multinomial
    [deriv: the groups of a uniform background are a Poisson process].
    """
    if bg is None or bg.n_groups <= 0 or bg.expected_false <= 0:
        return bg
    ng = int(rng.poisson(bg.n_groups))
    if ng == 0:
        return Background(weight=0.0, pmf=bg.pmf, expected_false=0.0)
    pmf = rng.multinomial(ng, bg.pmf / bg.pmf.sum()).astype(float) / ng
    expected = bg.expected_false * ng / bg.n_groups
    return Background(weight=min(expected / max(n_marks, 1), 0.99), pmf=pmf, expected_false=expected,
                      density_per_um2=bg.density_per_um2 * ng / bg.n_groups, n_groups=ng, area_um2=bg.area_um2)


def _bootstrap(units: List[Tuple[str, str, NDArray[np.int64], NDArray[np.float64]]], *, model: str,
               bg: Optional[Background], n_min: int, n_boot: int, seed: int,
               start: Tuple[float, float]) -> Dict[str, Any]:
    """Hierarchical resampling: sessions, then fields within session, then marks within field."""
    rng = np.random.default_rng(seed)
    sessions: Dict[str, List[Tuple[NDArray[np.int64], NDArray[np.float64]]]] = {}
    for sess, _, c, cen in units:
        sessions.setdefault(sess, []).append((c, cen))
    names = sorted(sessions)
    means, ks, gks = [], [], []
    for _ in range(int(n_boot)):
        pick_s = [names[i] for i in rng.integers(0, len(names), len(names))] if len(names) > 1 else names
        cs, cens = [], []
        for s in pick_s:
            fields = sessions[s]
            pick_f = rng.integers(0, len(fields), len(fields)) if len(fields) > 1 else [0]
            for fi in pick_f:
                c, cen = fields[int(fi)]
                if c.size == 0:
                    continue
                idx = rng.integers(0, c.size, c.size)
                cs.append(c[idx])
                cens.append(cen[idx])
        if not cs:
            continue
        cc, ce = np.concatenate(cs), np.concatenate(cens)
        bg_rep = _resample_background(bg, rng, cc.size)
        try:
            fr = fit_counts(cc, None, survival=ce, model=model, background=bg_rep, n_min=n_min, start=start)
        except ValueError:
            continue
        means.append(fr.mean)
        ks.append(fr.k)
        # The gamma's own k, so its interval exists even when k = 1 was selected.
        gks.append(fr.k if model == "gamma" else
                   fit_counts(cc, None, survival=ce, model="gamma", background=bg_rep, n_min=n_min,
                              start=(1.0, fr.mean)).k)
    if not means:
        return {"n_replicates": 0}
    m, k, gk = np.asarray(means), np.asarray(ks), np.asarray(gks)
    return {
        "n_replicates": int(m.size),
        "levels": ("session > field > mark" if len(names) > 1 else
                   "field > mark" if max(len(v) for v in sessions.values()) > 1 else "mark"),
        "mean_ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
        "k_ci95": [float(np.percentile(k, 2.5)), float(np.percentile(k, 97.5))],
        "mean_se": float(np.std(m, ddof=1)) if m.size > 1 else float("nan"),
        "k_se": float(np.std(k, ddof=1)) if k.size > 1 else float("nan"),
        "gamma_k_ci95": [float(np.percentile(gk, 2.5)), float(np.percentile(gk, 97.5))],
        "model": model,
    }


def _drift_nm(ev: Any, label: NDArray[np.int64], keep_marks: NDArray[np.bool_], n_frames: int) -> float:
    """Median over marks of the per-mark linear slope of event position vs frame, times the movie length."""
    slopes = []
    for m in np.nonzero(keep_marks)[0]:
        sel = label == m
        if np.count_nonzero(sel) < DRIFT_MIN_EVENTS:
            continue
        t = 0.5 * (ev.first_frame[sel] + ev.last_frame[sel]).astype(float)
        if np.ptp(t) <= 0:
            continue
        sx = np.polyfit(t, ev.x_nm[sel], 1)[0]
        sy = np.polyfit(t, ev.y_nm[sel], 1)[0]
        slopes.append((sx, sy))
    if len(slopes) < 5:
        return float("nan")
    s = np.median(np.asarray(slopes), axis=0)
    return float(math.hypot(s[0], s[1]) * n_frames)


def _metadata_checks(meta: Dict[str, Any], where: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def num(key: str) -> Optional[float]:
        try:
            return float(meta[key]) if key in meta and meta[key] not in ("", None) else None
        except (TypeError, ValueError):
            return None

    ph0, ph1 = num("ph_start"), num("ph_end")
    if ph0 is not None:
        ok = QC_PH_START[0] <= ph0 <= QC_PH_START[1]
        out.append({"check": "ph_start", "where": where, "value": ph0, "ok": ok,
                    "rule": f"{QC_PH_START[0]}-{QC_PH_START[1]} [ad hoc, protocol F5/F12]"})
    if ph0 is not None and ph1 is not None:
        ok = (ph0 - ph1) <= QC_PH_DROP and ph1 >= QC_PH_FLOOR
        out.append({"check": "ph_drop", "where": where, "value": ph0 - ph1, "ok": ok,
                    "rule": f"drop <= {QC_PH_DROP} and end >= {QC_PH_FLOOR} [ad hoc, protocol F5/F12]"})
    p0, p1 = num("power_start_mw"), num("power_end_mw")
    if p0 is not None and p1 is not None and p0 > 0:
        change = abs(p1 - p0) / p0
        out.append({"check": "laser_power_change", "where": where, "value": change, "ok": change <= QC_POWER_CHANGE,
                    "rule": f"<= {QC_POWER_CHANGE:.0%} between start and end [ad hoc, protocol F12]"})
    return out


def analyse_calibration(samples: Sequence[FieldData], params: CalibrationParams, *,
                        controls: Sequence[FieldData] = (), metadata: Optional[Dict[str, Any]] = None,
                        dye: Optional[str] = None, buffer: Optional[str] = None) -> CalibrationResult:
    """Run steps 1-7 (module docstring) on synthetic or user-supplied arrays."""
    params.validate()
    if not samples:
        raise ValueError("no sample field given")
    metadata = dict(metadata or {})
    all_lp = [s.lp_nm for s in samples if s.lp_nm is not None]
    if params.group_radius_nm is not None:
        group_radius = float(params.group_radius_nm)
        radius_origin = "given by the user"
    else:
        lp_cat = np.concatenate([np.asarray(v, dtype=float) for v in all_lp]) if all_lp else None
        group_radius = derived_group_radius(lp_cat, params.extra_sigma_nm)
        radius_origin = (f"[deriv] {MARK_RADIUS_IN_SIGMAS:.2f} x sqrt(median lp^2 + {params.extra_sigma_nm:g}^2) "
                         f"(extra sigma [ad hoc])")
    warn_list: List[str] = []

    # ---- per field: link, group, measure
    per_field = []
    for fd in samples:
        table, ev = _mark_table(fd, params, group_radius, None)
        _classify(table, params, group_radius)
        per_field.append((fd, table, ev))
    n_frames_all = {fd.resolved_n_frames() for fd, _, _ in per_field}
    if any(fd.n_frames is None for fd in samples):
        warn_list.append("movie length not given for some field: taken as the last frame + 1, which makes "
                         "'active at the end' unreliable")

    # ---- the end of the movie, from the dark gaps inside isolated marks [deriv]
    gaps_iso = []
    for _, t, ev in per_field:
        lab = t["_event_label"]
        iso_ev = t["isolated"][lab] if lab.size else np.zeros(0, dtype=bool)
        # Events of non-isolated marks get unique labels, so they make no gaps.
        gaps_iso.append(_off_gaps(ev, np.where(iso_ev, lab, -1 - np.arange(lab.size))))
    g_all = np.sort(np.concatenate(gaps_iso)) if gaps_iso else np.empty(0)
    enough_gaps = g_all.size >= 10
    if params.end_window_frames is not None:
        end_window = int(params.end_window_frames)
        end_origin = "given by the user"
    else:
        end_window = int(np.ceil(np.percentile(g_all, 95))) if enough_gaps else 0
        end_origin = ("[deriv] 95th percentile of the dark gaps between events inside isolated marks"
                      if enough_gaps else "too few within-mark gaps: 0 (no mark treated as active)")
    if not enough_gaps:
        warn_list.append("fewer than 10 dark gaps inside isolated marks: the censored estimand equals the recorded one")
    for fd, t, ev in per_field:
        gap_end = (fd.resolved_n_frames() - 1 - t["last_frame"]).astype(float)
        t["end_gap_frames"] = gap_end
        t["active_at_end"] = gap_end <= end_window
        # Empirical P(dark gap > g): the chance that a label still active would have stayed silent this long.
        t["p_silent_if_active"] = (1.0 - np.searchsorted(g_all, gap_end, "right") / g_all.size) if enough_gaps \
            else np.zeros(gap_end.size)

    # ---- background from the controls (identical processing)
    ctrl_tables = []
    for cfd in controls:
        ct, cev = _mark_table(cfd, params, group_radius, None)
        _classify(ct, params, group_radius)
        ct["_inner"] = ct["border_nm"] >= params.border_nm
        ctrl_tables.append((cfd, ct, cev))

    def background_for(pump: int, iso_share: float, area_um2: float) -> Optional[Background]:
        if not ctrl_tables:
            return None
        packed = []
        for cfd, ct, cev in ctrl_tables:
            t2 = dict(ct)
            t2["n_events_pump"] = _events_after(cev, ct["_event_label"], ct["n_events"].size, pump)
            packed.append((t2, _inner_area_um2(cfd, params.border_nm)))
        dens, pmf, ng, area = background_from_controls(packed, counts_key="n_events_pump")
        expected = dens * area_um2 * iso_share
        return Background(weight=0.0, pmf=pmf, expected_false=expected, density_per_um2=dens,
                          n_groups=ng, area_um2=area)

    # ---- assemble units and counts for one pump value
    def collect(pump: int) -> Tuple[List[Tuple[str, str, NDArray[np.int64], NDArray[np.float64]]], Dict[str, Any]]:
        units = []
        n_census, n_lost, n_inner, n_iso, area = 0, 0, 0, 0, 0.0
        for fd, t, ev in per_field:
            cnt = _events_after(ev, t["_event_label"], t["n_events"].size, pump)
            sel = t["in_main"]
            n_census += int(np.count_nonzero(sel))
            n_lost += int(np.count_nonzero(sel & (cnt == 0)))
            use = sel & (cnt > 0)
            surv = np.asarray(t["p_silent_if_active"], dtype=float)[use] if params.estimand == "censored" \
                else np.zeros(int(np.count_nonzero(use)))
            units.append((fd.session, fd.name, cnt[use], surv))
            inner = t["border_nm"] >= params.border_nm
            n_inner += int(np.count_nonzero(inner & (cnt > 0)))
            n_iso += int(np.count_nonzero(use))
            area += _inner_area_um2(fd, params.border_nm)
        info = {"n_census": n_census, "n_lost": n_lost, "n_inner_groups": n_inner, "n_isolated": n_iso,
                "inner_area_um2": area}
        return units, info

    def fit_all(units: List[Tuple[str, str, NDArray[np.int64], NDArray[np.float64]]], info: Dict[str, Any],
                pump: int) -> Tuple[Dict[str, FitResult], Optional[Background], int, NDArray[np.int64],
                                    NDArray[np.float64]]:
        c = np.concatenate([u[2] for u in units]) if units else np.empty(0, dtype=np.int64)
        cen = np.concatenate([u[3] for u in units]) if units else np.empty(0, dtype=float)
        iso_share = info["n_isolated"] / info["n_inner_groups"] if info["n_inner_groups"] else 1.0
        bg = background_for(pump, iso_share, info["inner_area_um2"])
        if bg is not None and c.size:
            bg.weight = min(bg.expected_false / c.size, 0.99)
        n_min = 1
        bg_fit: Optional[Background] = bg if params.background == "mixture" else None
        if params.background == "n_min":
            n_min = n_min_from_background(c, bg)
        fits = {m: fit_counts(c, None, survival=cen, model=m, background=bg_fit, n_min=n_min)
                for m in MODELS} if c.size else {}
        return fits, bg, n_min, c, cen

    main_units, main_info = collect(params.pump_frames)
    fits, bg, n_min, counts, cens = fit_all(main_units, main_info, params.pump_frames)
    if not fits:
        raise ValueError("no isolated mark with events: nothing to fit")
    if n_min > 1:
        warn_list.append(f"fit restricted to marks with >= n_min = {n_min} events: only the tail is seen, so the "
                         f"geometric mean is identified but the gamma's k (and with it k*theta) is not; "
                         f"prefer the mixture correction or check the geometric fit [simulation]")
    # Model choice by AIC; the gamma nests k = 1 [ad hoc rule: prefer the simpler one when AIC does not favour k].
    chosen = "gamma" if fits["gamma"].aic < fits["geometric"].aic else "geometric"
    best = fits[chosen]
    boot = _bootstrap(main_units, model=chosen, bg=bg if params.background == "mixture" else None,
                      n_min=n_min, n_boot=params.n_bootstrap, seed=params.seed, start=(best.k, best.mean))

    # Alternative corrections and estimands, for the record
    alt: Dict[str, Any] = {}
    if bg is not None:
        nm = n_min_from_background(counts, bg)
        enough = bool(np.count_nonzero(counts >= nm))
        alt["n_min_rule"] = {
            "n_min": nm,
            "geometric": fit_counts(counts, None, survival=cens, model="geometric", n_min=nm).to_dict()
            if enough else None,
            "gamma": fit_counts(counts, None, survival=cens, model="gamma", n_min=nm).to_dict() if enough else None}
        alt["no_background_correction"] = fit_counts(counts, None, survival=cens, model="gamma").to_dict()
    other = "censored" if params.estimand == "recorded" else "recorded"
    if other == "censored":
        cen_other = np.concatenate([np.asarray(t["p_silent_if_active"], dtype=float)[
            t["in_main"] & (_events_after(ev, t["_event_label"], t["n_events"].size, params.pump_frames) > 0)]
            for _, t, ev in per_field])
    else:
        cen_other = np.zeros(counts.size)
    alt[f"estimand_{other}"] = fit_counts(counts, None, survival=cen_other, model="gamma",
                                          background=bg if params.background == "mixture" else None,
                                          n_min=n_min).to_dict()

    # Pump-frame versions (point estimates)
    pump_versions = []
    for pf in sorted(set(params.pump_frames_versions) | {params.pump_frames}):
        u2, i2 = collect(pf)
        try:
            f2, _, nm2, c2, _ = fit_all(u2, i2, pf)
        except ValueError:
            f2, nm2, c2 = {}, 1, np.empty(0, dtype=np.int64)
        pump_versions.append({
            "pump_frames": pf, "n_marks": int(c2.size),
            "empirical_mean": float(np.mean(c2)) if c2.size else None,
            "gamma_mean": f2["gamma"].mean if f2 else None, "gamma_k": f2["gamma"].k if f2 else None,
            "geometric_mean": f2["geometric"].mean if f2 else None,
            "fraction_lost": (i2["n_lost"] / i2["n_census"]) if i2["n_census"] else None,
        })

    # Irradiance bands (isolated marks, regardless of the main cut)
    bands = []
    if params.beam_centre_nm is not None and params.beam_sigma_nm:
        for lo, hi in IRRADIANCE_BANDS:
            cs = []
            for fd, t, ev in per_field:
                cnt = _events_after(ev, t["_event_label"], t["n_events"].size, params.pump_frames)
                sel = t["isolated"] & (t["rel_irradiance"] >= lo) & (t["rel_irradiance"] < hi) & (cnt > 0)
                cs.append(cnt[sel])
            cb = np.concatenate(cs)
            bands.append({"rel_irradiance": [lo, min(hi, 1.0)], "n_marks": int(cb.size),
                          "empirical_mean": float(np.mean(cb)) if cb.size else None})

    # Per-session fits (mean of session means, protocol F8)
    sessions: Dict[str, List[NDArray[np.int64]]] = {}
    sess_surv: Dict[str, List[NDArray[np.float64]]] = {}
    for s, _, c, sv in main_units:
        sessions.setdefault(s, []).append(c)
        sess_surv.setdefault(s, []).append(sv)
    per_session = {}
    for s, lst in sorted(sessions.items()):
        c_s = np.concatenate(lst)
        if c_s.size:
            fr = fit_counts(c_s, None, survival=np.concatenate(sess_surv[s]), model=chosen,
                            background=bg if params.background == "mixture" else None, n_min=n_min)
            per_session[s] = {"n_marks": int(c_s.size), "mean": fr.mean, "k": fr.k,
                              "empirical_mean": float(np.mean(c_s))}
    sess_summary: Dict[str, Any] = {"per_session": per_session}
    if len(per_session) >= 2:
        from scipy import stats
        ms = np.array([v["mean"] for v in per_session.values()])
        sd = float(np.std(ms, ddof=1))
        tq = float(stats.t.ppf(0.975, ms.size - 1))
        sess_summary.update({"mean_of_session_means": float(ms.mean()), "sd_between_sessions": sd,
                             "ci95_t": [float(ms.mean() - tq * sd / math.sqrt(ms.size)),
                                        float(ms.mean() + tq * sd / math.sqrt(ms.size))],
                             "origin": "[deriv] t interval over session means (protocol F8)"})

    # ---- quality checks
    qc: List[Dict[str, Any]] = []
    for fd, t, ev in per_field:
        where = f"{fd.session}/{fd.name}"
        area = _inner_area_um2(fd, params.border_nm)
        inner = t["border_nm"] >= params.border_nm
        n_in = int(np.count_nonzero(inner))
        dens = n_in / area if area > 0 else float("nan")
        qc.append({"check": "density_marks_per_um2", "where": where, "value": dens,
                   "ok": bool(QC_DENSITY_RANGE[0] <= dens <= QC_DENSITY_RANGE[1]),
                   "rule": f"{QC_DENSITY_RANGE[0]}-{QC_DENSITY_RANGE[1]} [ad hoc, protocol F6/F12]"})
        if n_in:
            obs_short = float(np.mean(t["nn_nm"][inner] < params.isolation_nm))
            exp_short = 1.0 - math.exp(-math.pi * dens * (params.isolation_nm / 1000.0) ** 2) \
                if math.isfinite(dens) else float("nan")
            ratio = obs_short / exp_short if exp_short > 0 else float("nan")
            qc.append({"check": "aggregates_short_1nn_ratio", "where": where, "value": ratio,
                       "observed": obs_short, "expected_poisson": exp_short,
                       "ok": bool(not (ratio > QC_AGGREGATE_EXCESS)),
                       "rule": f"observed/expected share of 1NN < {params.isolation_nm:g} nm <= "
                               f"{QC_AGGREGATE_EXCESS} [ad hoc, protocol F6/F12; expected = 1-exp(-pi rho r^2), deriv]"})
        dr = _drift_nm(ev, t["_event_label"], t["isolated"], fd.resolved_n_frames())
        qc.append({"check": "drift_nm_over_movie", "where": where, "value": dr,
                   "ok": bool(not (dr > QC_DRIFT_NM)) if math.isfinite(dr) else None,
                   "rule": f"<= {QC_DRIFT_NM:g} nm (median per-mark slope x frames) [ad hoc, protocol F12]"})
        qc.extend(_metadata_checks(fd.metadata, where))
    for s, info in per_session.items():
        qc.append({"check": "isolated_marks_per_session", "where": s, "value": info["n_marks"],
                   "ok": info["n_marks"] >= QC_MIN_ISOLATED,
                   "rule": f">= {QC_MIN_ISOLATED} [ad hoc, protocol F8/F12]"})
    act_all = np.concatenate([np.asarray(t["active_at_end"])[t["in_main"]] for _, t, _ in per_field])
    act = float(np.mean(act_all)) if act_all.size else float("nan")
    qc.append({"check": "fraction_active_at_end", "where": "all", "value": act,
               "ok": bool(not (act > QC_ACTIVE_AT_END)),
               "rule": f"<= {QC_ACTIVE_AT_END:.0%}; above it the recorded count is mostly a lower bound "
                       f"[ad hoc; Dempsey et al. 2011 p. 1030 call their cycles a lower bound for this reason]"})
    if bg is not None:
        share = bg.expected_false / counts.size if counts.size else float("nan")
        qc.append({"check": "background_share", "where": "all", "value": share,
                   "ok": bool(not (share > QC_BACKGROUND_SHARE)),
                   "rule": f"expected false marks / isolated marks <= {QC_BACKGROUND_SHARE:.0%} "
                           f"[ad hoc, protocol F12]"})
    else:
        warn_list.append("no background control given: single-event background groups count as marks")
    qc.extend(_metadata_checks(metadata, "session"))

    # ---- survival of the isolated marks (fraction with an event at or after frame t)
    last_all = np.concatenate([t["last_frame"][t["in_main"]] for _, t, _ in per_field])
    nf = max(n_frames_all) if n_frames_all else 0
    grid = np.linspace(0, nf, 21).astype(int) if nf else np.zeros(1, dtype=int)
    survival = [(int(g), float(np.mean(last_all >= g)) if last_all.size else float("nan")) for g in grid]

    # ---- sensitivity to the protocol's fixed 150 nm grouping radius
    sens: Optional[Dict[str, Any]] = None
    if params.group_radius_nm is None and abs(group_radius - PROTOCOL_GROUP_RADIUS_NM) > 1e-6:
        cs = []
        for fd, _, _ in per_field:
            t2, ev2 = _mark_table(fd, params, PROTOCOL_GROUP_RADIUS_NM, None)
            _classify(t2, params, PROTOCOL_GROUP_RADIUS_NM)
            cnt2 = _events_after(ev2, t2["_event_label"], t2["n_events"].size, params.pump_frames)
            cs.append(cnt2[t2["in_main"] & (cnt2 > 0)])
        c150 = np.concatenate(cs)
        sens = {"group_radius_nm": PROTOCOL_GROUP_RADIUS_NM, "n_marks": int(c150.size),
                "empirical_mean": float(np.mean(c150)) if c150.size else None}

    # ---- the prior, ready to register
    from tools.emitters_prior import calibrated_prior
    note_bits = [
        "per isolated label (an IgG secondary when the control is antibodies on glass)",
        f"sessions: {', '.join(sorted(sessions))}", f"n = {best.n_marks} marks",
        f"link {params.link_radius_nm:g} nm / {params.max_dark_frames} dark frames",
        f"group radius {group_radius:.0f} nm", f"pump frames {params.pump_frames}",
        f"estimand {params.estimand}", f"background {params.background if bg is not None else 'none given'}",
    ]
    if "mean_ci95" in boot:
        note_bits.append(f"bootstrap 95% CI mean {boot['mean_ci95'][0]:.2f}-{boot['mean_ci95'][1]:.2f}")
    if params.beam_centre_nm is not None and params.beam_sigma_nm:
        note_bits.append(f"irradiance >= {params.min_rel_irradiance:.0%} of peak")
    for key in ("lot", "date", "filter"):
        if key in metadata:
            note_bits.append(f"{key} {metadata[key]}")
    note = "; ".join(note_bits)
    prior_mean, prior_k = float(best.mean), float(best.k)
    prior_args: Dict[str, Any] = {"mean": prior_mean, "k": prior_k, "dye": dye, "buffer": buffer, "note": note}
    prior = calibrated_prior(prior_mean, k=prior_k, dye=dye, buffer=buffer, note=note)

    # ---- mark rows
    rows: List[Dict[str, Any]] = []
    for fd, t, ev in per_field:
        for i in range(t["n_events"].size):
            rows.append({c: (fd.session if c == "session" else fd.name if c == "field" else
                             (t[c][i].item() if hasattr(t[c][i], "item") else t[c][i]))
                         for c in MARK_COLUMNS})

    emp = {"mean": float(np.mean(counts)), "median": float(np.median(counts)),
           "p_le_1": float(np.mean(counts <= 1)), "p_le_3": float(np.mean(counts <= 3))}
    summary: Dict[str, Any] = {
        "table_version": TABLE_VERSION,
        "method": METHOD_LABEL,
        "what": "events (frame-linked localizations) per isolated label",
        "metadata": metadata,
        "dye": dye, "buffer": buffer,
        "inputs": {"samples": [{"session": fd.session, "field": fd.name, "n_locs": int(fd.frame.size),
                                "n_frames": fd.resolved_n_frames(),
                                "n_frames_source": "given" if fd.n_frames is not None else "last frame + 1",
                                "bounds_nm": list(fd.resolved_bounds()), "metadata": fd.metadata}
                               for fd in samples],
                   "controls": [{"session": c.session, "field": c.name, "n_locs": int(c.frame.size)}
                                for c in controls]},
        "parameters": {
            "link_radius_nm": params.link_radius_nm, "max_dark_frames": params.max_dark_frames,
            "linker": "tools.mps_paint.build_events (Picasso's link rule), censored events kept",
            "group_radius_nm": group_radius, "group_radius_origin": radius_origin,
            "isolation_nm": params.isolation_nm, "border_nm": params.border_nm,
            "elongation_factor": params.elongation_factor, "use_z_for_isolation": params.use_z_for_isolation,
            "pump_frames": params.pump_frames, "end_window_frames": end_window, "end_window_origin": end_origin,
            "estimand": params.estimand, "background": params.background,
            "beam_centre_nm": params.beam_centre_nm, "beam_sigma_nm": params.beam_sigma_nm,
            "min_rel_irradiance": params.min_rel_irradiance, "n_bootstrap": params.n_bootstrap,
            "seed": params.seed,
        },
        "counts": {"n_marks_total": int(sum(t["n_events"].size for _, t, _ in per_field)),
                   "n_isolated_main": int(counts.size), "n_census_main": main_info["n_census"],
                   "fraction_lost_after_pump": (main_info["n_lost"] / main_info["n_census"]
                                                if main_info["n_census"] else None),
                   "empirical": emp},
        "background": None if bg is None else {
            "density_groups_per_um2": bg.density_per_um2, "n_groups": bg.n_groups, "area_um2": bg.area_um2,
            "expected_false_marks": bg.expected_false, "weight": bg.weight,
            "pmf_events_per_group": [float(v) for v in bg.pmf],
            "n_min_rule": n_min_from_background(counts, bg)},
        "fits": {m: f.to_dict() for m, f in fits.items()},
        "selected_model": chosen,
        "selection_rule": "lower AIC between gamma and geometric (k = 1) [ad hoc, protocol F11.6]",
        "result": {"mean": best.mean, "k": best.k, "theta": best.theta,
                   "p_le_1": prior.prob_at_most(1), "p_le_3": prior.prob_at_most(3),
                   "bootstrap": boot,
                   "mean_definition": "k*theta of the continuous gamma the emitters layer evaluates; the "
                                      "empirical mean of the counts is about 0.5 higher for k = 1 [deriv]"},
        "sessions": sess_summary,
        "alternatives": alt,
        "pump_versions": pump_versions,
        "irradiance_bands": bands,
        "grouping_sensitivity_150nm": sens,
        "survival_isolated": survival,
        "quality": qc,
        "quality_all_ok": all(q["ok"] is not False for q in qc),
        "warnings": warn_list,
        "calibrated_prior_args": prior_args,
        "calibrated_prior": prior.to_dict(),
    }
    return CalibrationResult(summary=summary, marks=rows, hist_counts=counts, fits=fits, background=bg,
                             survival=survival)


# ===========================================================================
#  6. Outputs
# ===========================================================================
def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _clean(o: Any) -> Any:
    """NaN/inf -> None, so the JSON is strict."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


def write_outputs(result: CalibrationResult, out_prefix: str, *, figure: bool = True) -> Dict[str, str]:
    """``<prefix>_marks.csv``, ``<prefix>_xi.json`` and (optionally) ``<prefix>_xi.png``. Never overwrites."""
    paths = {"csv": out_prefix + "_marks.csv", "json": out_prefix + "_xi.json"}
    if figure:
        paths["png"] = out_prefix + "_xi.png"
    for p in paths.values():
        if os.path.exists(p):
            raise FileExistsError(f"{p} exists: choose another output prefix (outputs are never overwritten)")
    d = os.path.dirname(os.path.abspath(out_prefix))
    os.makedirs(d, exist_ok=True)
    with open(paths["csv"], "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MARK_COLUMNS))
        w.writeheader()
        for r in result.marks:
            w.writerow({k: ("" if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()})
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(_clean(json.loads(json.dumps(result.summary, default=_json_default))), fh, indent=2,
                  ensure_ascii=False, allow_nan=False)
    if figure:
        plot_result(result, paths["png"])
    return paths


def plot_result(result: CalibrationResult, path: str) -> None:
    """Histogram of events per isolated mark with the fitted pmfs, and the survival of the marks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = result.hist_counts
    n_max = int(max(c.max(), 1))
    n = np.arange(1, n_max + 1)
    obs = np.bincount(c, minlength=n_max + 1)[1:]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax.bar(n, obs, width=0.9, color=_C_OBSERVED, alpha=0.55, label=f"isolated marks (n = {c.size})")
    bg = result.background
    w = bg.weight if bg is not None else 0.0
    for model, colour, style in (("gamma", _C_GAMMA, "-"), ("geometric", _C_GEOM, "--")):
        f = result.fits.get(model)
        if f is None:
            continue
        th = np.array([math.log(f.theta)]) if model == "geometric" else np.array([math.log(f.k), math.log(f.theta)])
        pmf, _ = _model_tables(model, th, n_max)
        lab = f"{model}: mean {f.mean:.2f}" + (f", k {f.k:.2f}" if model == "gamma" else "") + f" (AIC {f.aic:.1f})"
        exp = c.size * (1 - w) * pmf[1:]
        if bg is not None:
            exp = exp + c.size * w * bg.tables(n_max)[0][1:]
        ax.plot(n, exp, style, color=colour, lw=2, label=lab)
    if bg is not None and bg.expected_false > 0:
        ax.bar(n, bg.expected_false * bg.tables(n_max)[0][1:], width=0.5, color=_C_BACKGROUND, alpha=0.8,
               label=f"expected background ({bg.expected_false:.1f} marks)")
    ax.set_xlabel("events per mark (after frame linking)")
    ax.set_ylabel("marks")
    ax.set_title("xi calibration: " + str(result.summary.get("dye") or "dye not given"))
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    s = np.asarray(result.survival, dtype=float)
    ax2.plot(s[:, 0], s[:, 1], color=_C_GREY, lw=2)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("isolated marks with an event at or after it")
    ax2.set_ylim(0, 1.02)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
#  7. Files and command line
# ===========================================================================
def load_field(path: str, *, session: str, pixel_size_nm: Optional[float] = None,
               n_frames: Optional[int] = None) -> FieldData:
    """Read a Picasso HDF5 / ThunderSTORM CSV with ``tools.mps_io.load_localizations``."""
    from tools.mps_io import load_localizations
    loc = load_localizations(path, pixel_size_nm=pixel_size_nm)
    frame = loc.frame
    if frame is None:
        raise ValueError(f"{os.path.basename(path)} has no frame column: events cannot be linked")
    nf = n_frames if n_frames is not None else loc.n_frames
    return FieldData(frame=frame, x_nm=loc.x_nm, y_nm=loc.y_nm, lp_nm=loc.lp_lateral_nm, photons=loc.photons,
                     z_nm=loc.z_nm if getattr(loc, "is_3d", False) else None, n_frames=nf, session=session,
                     name=os.path.splitext(os.path.basename(path))[0])


def _parse_meta(items: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--meta expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.calibration_xi",
        description="Events per isolated label (xi) from a sparse calibration sample; writes a per-mark CSV, "
                    "a JSON with the calibrated_prior arguments and a PNG.")
    p.add_argument("--session", nargs="+", action="append", required=True, metavar=("LABEL", "FILE"),
                   help="a session label followed by its localization files (one per field); repeat per session")
    p.add_argument("--control", action="append", default=[], metavar="FILE",
                   help="sample without antibody, for the background (repeatable)")
    p.add_argument("--link-radius", type=float, required=True,
                   help="link radius in nm: the SAME value used for the tissue data")
    p.add_argument("--max-dark", type=int, required=True,
                   help="max dark frames inside one event: the SAME value used for the tissue data")
    p.add_argument("--group-radius", type=float, default=None,
                   help="nm; default derived from the precision (protocol's fixed value: 150)")
    p.add_argument("--extra-sigma", type=float, default=EXTRA_SIGMA_NM)
    p.add_argument("--isolation", type=float, default=ISOLATION_NM)
    p.add_argument("--border", type=float, default=BORDER_NM)
    p.add_argument("--use-z", action="store_true", help="nearest neighbour in 3D (tissue)")
    p.add_argument("--pump-frames", type=int, default=0)
    p.add_argument("--pump-versions", default=",".join(str(v) for v in DEFAULT_PUMP_FRAMES))
    p.add_argument("--end-window", type=int, default=None)
    p.add_argument("--estimand", choices=("recorded", "censored"), default="recorded")
    p.add_argument("--background", choices=("mixture", "n_min", "none"), default="mixture")
    p.add_argument("--beam-centre", default=None, help="x,y in nm")
    p.add_argument("--beam-sigma", type=float, default=None, help="nm")
    p.add_argument("--min-irradiance", type=float, default=MIN_REL_IRRADIANCE)
    p.add_argument("--pixel-size", type=float, default=None)
    p.add_argument("--n-frames", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dye", default=None)
    p.add_argument("--buffer", default=None)
    p.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                   help="session metadata stored in the JSON (lot, power_start_mw, ph_start, frames, ...)")
    p.add_argument("--out", required=True, help="output prefix (files are never overwritten)")
    p.add_argument("--no-figure", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    samples: List[FieldData] = []
    for group in args.session:
        if len(group) < 2:
            raise SystemExit("--session needs a label and at least one file")
        for path in group[1:]:
            samples.append(load_field(path, session=group[0], pixel_size_nm=args.pixel_size,
                                      n_frames=args.n_frames))
    controls = [load_field(p, session="control", pixel_size_nm=args.pixel_size, n_frames=args.n_frames)
                for p in args.control]
    beam = None
    if args.beam_centre:
        bx, by = (float(v) for v in args.beam_centre.split(","))
        beam = (bx, by)
    params = CalibrationParams(
        link_radius_nm=args.link_radius, max_dark_frames=args.max_dark, group_radius_nm=args.group_radius,
        extra_sigma_nm=args.extra_sigma, isolation_nm=args.isolation, border_nm=args.border,
        use_z_for_isolation=args.use_z, pump_frames=args.pump_frames,
        pump_frames_versions=tuple(int(v) for v in args.pump_versions.split(",") if v.strip()),
        end_window_frames=args.end_window, estimand=args.estimand, background=args.background,
        beam_centre_nm=beam, beam_sigma_nm=args.beam_sigma, min_rel_irradiance=args.min_irradiance,
        n_bootstrap=args.bootstrap, seed=args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = analyse_calibration(samples, params, controls=controls, metadata=_parse_meta(args.meta),
                                     dye=args.dye, buffer=args.buffer)
    paths = write_outputs(result, args.out, figure=not args.no_figure)
    r = result.summary["result"]
    ci = r["bootstrap"].get("mean_ci95")
    print(f"xi ({result.summary['selected_model']}): mean {r['mean']:.2f}, k {r['k']:.2f}"
          + (f", 95% CI mean {ci[0]:.2f}-{ci[1]:.2f}" if ci else "")
          + f"; n = {result.summary['counts']['n_isolated_main']} isolated marks")
    bad = [q for q in result.summary["quality"] if q["ok"] is False]
    for q in bad:
        print(f"QUALITY: {q['check']} at {q['where']} = {q['value']} (rule {q['rule']})")
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
