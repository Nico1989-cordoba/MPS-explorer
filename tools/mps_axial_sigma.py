# -*- coding: utf-8 -*-
"""
The axial localization error sigma_z measured in the file itself (a "NeNA in z" with a slow component).

Why
---
The axial leak of question C of the perimeter density analysis (a localization of one slice that lands in its
neighbour) depends on how much the z of a localization is wrong. Picasso's ``lpz`` does not say: measured on pilot
data it ranges from 0.45 x lpz close to the focus to 1.0-1.7 x lpz above z = +100 nm, so it is not a usable error per
localization. The data can say it. A molecule seen in two consecutive frames is in the same place both times, so the
difference of its two z values is only error (NeNA, Endesfelder et al. 2014, turned to the axial coordinate); and the
same molecule switched on again many frames later shows an additional slow error (focus drift, a z bias that changes
between switch-ons) that consecutive frames cannot see.

What is measured (``measure_axial_sigma``)
------------------------------------------
* ``sigma_frame``: the frame-to-frame error. Pairs of localizations in consecutive frames closer than 25 nm in x, y
  and with no other partner within 40 nm (the same molecule). The error in z is NOT Gaussian: the difference of the
  pair is fitted by the mixture that follows from "each localization is good (sd s_c) with probability 1 - w and bad
  (sd s_b) with probability w", by maximum likelihood on a fine histogram, with BIC against a single Gaussian. The
  leak is linear in the MEAN ABSOLUTE error (uniform depth, error smaller than the slice), so the equivalent sd is
  (1 - w) s_c + w s_b, not a root-mean-square.
* ``sigma_slow``: the extra error of a molecule seen again 1001 frames or more later. Pairs closer than 10 nm in x, y
  are a mixture of the same molecule (narrow) and its neighbours (wide); the narrow component at lag >= 1001 frames
  against the one at lag 1 gives delta = sqrt(s_long^2 - s_1^2).
* ``sigma_total`` adds delta in quadrature to the core and to the bad component of the frame mixture:
  (1 - w) sqrt(s_c^2 + delta^2) + w sqrt(s_b^2 + delta^2).
* Intervals (95 %): the file is resampled by blocks of frames (or by the groups the caller gives, e.g. axons, when
  there are at least six) and everything is refitted.

[real data, pilot 2026-11-17: 18 axons of one 3D-dSTORM acquisition]: s_c 24 nm, w 0.18, s_b 88 nm, sigma_frame
36 nm, delta 27 nm, sigma_total 47 nm (the width the leak of C uses when a file cannot be measured). The method is
the one of the validation study; this module makes it a function of one file.

When it CANNOT be measured (``AxialSigma.status`` and ``warnings`` say so; the caller then falls back to the pilot
width and says it)
-------------------------------------------------------------------------------------------------------------
* no ``frame`` column (``no_frame``): nothing identifies the same molecule;
* too few pairs (``too_few_pairs``): fewer than ``MIN_PAIRS_FRAME`` unique pairs or ``MIN_PAIRS_LAG1`` /
  ``MIN_PAIRS_LONG`` for the slow component; the frame error alone is then reported as ``frame_only`` and it is NOT
  a sigma_z for the leak, because it misses the slow component (it is 36 nm against 47 nm on the pilot);
* a dense acquisition (``contaminated``): localizations of different molecules in the same or the next frame within
  reach are taken for the same molecule and widen the pairs. Two controls from the data: (i) the share of
  localizations with another one in the SAME frame within 300 nm (different molecules for certain) and (ii) the
  pairs of consecutive frames at 100-300 nm, scaled to the 25 nm disc (area scale and, pessimistic, line scale).
  Above ``CONTAM_WARN`` a warning, above ``CONTAM_MAX`` no measurement.
* an interval wider than ``MAX_REL_WIDTH`` of the value (``interval_too_wide``).

What it does not see: a fixed bias per molecule (dipole orientation, local aberration that does not change between
switch-ons) and whether the slow component is error or neighbours of the same ring at < 10 nm (the validation study
could not tell them apart: 36 nm if it is contamination, 47 nm if it is error). ``lp`` and ``lpz`` are only
reported for comparison; they never enter the estimate.

Pure functions, numpy and scipy, no Qt, no I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import ndtr

# ----------------------------------------------------------------------------------------------------
# Constants. [real data, pilot] = taken from the validation study; [ad hoc] = chosen here, with a reason.
# ----------------------------------------------------------------------------------------------------
R_FRAME_NM: float = 25.0        # [real data, pilot] ~2.3 x the lateral step sd of one molecule; moves sigma +-3 nm
R_UNIQUE_NM: float = 40.0       # [real data, pilot] a pair is dropped when a member has another partner within it
R_LAG_NM: float = 10.0          # [real data, pilot] radius of the re-switch-on pairs
R_BOX_NM: float = 300.0         # [real data, pilot] reach of the same-frame / ring controls
R_RING_IN_NM: float = 100.0     # [real data, pilot] inner radius of the control ring of other molecules
LONG_LAG_FRAMES: int = 1001     # [real data, pilot] the slow error is flat from ~1 minute on (>= 1001 frames of 50 ms)
MIN_PAIRS_FRAME: int = 1500     # [ad hoc] the smallest pilot axon gave 1556 pairs and a stable fit
MIN_PAIRS_LAG1: int = 500       # [ad hoc]
MIN_PAIRS_LONG: int = 1500      # [ad hoc]
MIN_LOCS: int = 2000            # [ad hoc]
CONTAM_WARN: float = 0.10       # [ad hoc] line-scaled share of the pairs that may be other molecules
CONTAM_MAX: float = 0.30        # [ad hoc]
SAME_FRAME_WARN: float = 0.01   # [ad hoc] pilot: 0 of 18 axons
SAME_FRAME_MAX: float = 0.05    # [ad hoc]
MAX_REL_WIDTH: float = 0.6      # [ad hoc] (hi - lo) / value above which the figure is not used for the leak
N_BOOT_DEFAULT: int = 99
N_BLOCKS: int = 20              # [ad hoc] frame blocks of the resampling
MIN_GROUPS: int = 6             # [ad hoc] groups from which the resampling is by group instead of by frame block
SIGMA_MIN_NM: float = 3.0       # [ad hoc] sanity range of a usable measured width
SIGMA_MAX_NM: float = 200.0
MAX_PAIRS_SLOW: int = 40_000_000  # [ad hoc] above this the localizations are subsampled for the slow component

_DZ_EDGES: NDArray[np.float64] = np.linspace(-820.0, 820.0, 821)   # 2 nm
_SQ2 = math.sqrt(2.0)
_NAN = float("nan")

STATUS_MEASURED = "measured"
STATUS_FRAME_ONLY = "frame_only"
STATUS_NO_FRAME = "no_frame"
STATUS_NO_Z = "no_z"
STATUS_TOO_FEW_LOCS = "too_few_locs"
STATUS_TOO_FEW_PAIRS = "too_few_pairs"
STATUS_CONTAMINATED = "contaminated"
STATUS_INTERVAL_TOO_WIDE = "interval_too_wide"
STATUSES: Tuple[str, ...] = (STATUS_MEASURED, STATUS_FRAME_ONLY, STATUS_NO_FRAME, STATUS_NO_Z, STATUS_TOO_FEW_LOCS,
                             STATUS_TOO_FEW_PAIRS, STATUS_CONTAMINATED, STATUS_INTERVAL_TOO_WIDE)


@dataclass(frozen=True)
class AxialSigma:
    """What ``measure_axial_sigma`` found in one file. NaN where a value could not be computed.

    ``reliable`` (status "measured") is the only state in which ``total`` may replace the pilot width."""

    status: str
    reliable: bool
    frame_nm: float = _NAN                 # equivalent sd, frame to frame only (the "36-like" figure)
    frame_lo_nm: float = _NAN
    frame_hi_nm: float = _NAN
    slow_nm: float = _NAN                  # delta, the slow component added in quadrature
    slow_lo_nm: float = _NAN
    slow_hi_nm: float = _NAN
    total_nm: float = _NAN                 # equivalent sd with the slow component (the "47-like" figure)
    total_lo_nm: float = _NAN
    total_hi_nm: float = _NAN
    core_nm: float = _NAN                  # mixture of the frame error: core, bad share, bad sd
    bad_fraction: float = _NAN
    bad_nm: float = _NAN
    mixture_by_bic: bool = False
    n_locs: int = 0
    n_pairs_frame: int = 0                 # unique consecutive-frame pairs at < 25 nm
    n_pairs_lag1: int = 0                  # consecutive-frame pairs at < 10 nm (reference of the slow component)
    n_pairs_long: int = 0                  # pairs at < 10 nm with lag >= 1001 frames
    contamination: float = _NAN            # line-scaled share of other-molecule pairs among the 25 nm pairs
    contamination_area: float = _NAN
    same_frame_fraction: float = _NAN      # share of localizations with another one in the same frame within 300 nm
    lpz_median_nm: float = _NAN            # reported lpz, for comparison only
    ratio_to_lpz: float = _NAN             # frame_nm / lpz median
    n_boot: int = 0
    resampled_by: str = ""                 # "frame blocks" or "groups"
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def note(self) -> str:
        return " | ".join(self.warnings)


# ----------------------------------------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------------------------------------
def _bin_probs(edges: NDArray[np.float64], s: float) -> NDArray[np.float64]:
    return np.diff(ndtr(edges / s)) / (ndtr(edges[-1] / s) - ndtr(edges[0] / s))


def _fit_frame_mixture(counts: NDArray[np.float64], edges: NDArray[np.float64] = _DZ_EDGES,
                       start: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
    """Model by localization on the histogram of dz of consecutive-frame pairs: each error is N(0, sc^2) with
    probability 1 - w or N(0, sb^2) with probability w (sb >= 1.5 sc), so dz is the mixture of three Gaussians
    (1-w)^2 N(0, 2 sc^2) + 2 w (1-w) N(0, sc^2 + sb^2) + w^2 N(0, 2 sb^2). BIC against one Gaussian. ``start``
    = (sc, w, sb) of an earlier fit (resamples), one start instead of four. Returns sc, sb, w, mix_by_bic and the
    equivalent sd for the leak ``(1 - w) sc + w sb``."""
    counts = np.asarray(counts, dtype=np.float64)
    n = float(counts.sum())
    nz = counts > 0
    c = counts[nz]
    centres = 0.5 * (edges[:-1] + edges[1:])
    absc = np.abs(centres)
    order = np.argsort(absc)
    cs = np.cumsum(counts[order])
    med_abs = absc[order][min(int(np.searchsorted(cs, 0.5 * n)), absc.size - 1)]
    s0 = max(1.4826 * float(med_abs) / _SQ2, float(edges[1] - edges[0]))

    def nll1(t: Any) -> float:
        p = _bin_probs(edges, _SQ2 * math.exp(t[0]))[nz]
        return -float(np.sum(c * np.log(np.maximum(p, 1e-300))))

    o1 = minimize(nll1, [math.log(s0)], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6))
    sc1 = math.exp(float(o1.x[0]))
    bic1 = 2.0 * float(o1.fun) + math.log(n)

    def unpack(t: Any) -> Tuple[float, float, float]:
        sc = math.exp(t[0])
        w = 0.6 / (1.0 + math.exp(-t[1]))
        sb = sc * (1.5 + math.exp(t[2]))
        return sc, w, sb

    def nll3(t: Any) -> float:
        sc, w, sb = unpack(t)
        p = ((1 - w) ** 2 * _bin_probs(edges, _SQ2 * sc) + 2 * w * (1 - w) * _bin_probs(edges, math.hypot(sc, sb))
             + w * w * _bin_probs(edges, _SQ2 * sb))[nz]
        return -float(np.sum(c * np.log(np.maximum(p, 1e-300))))

    starts: List[List[float]] = []
    if start is not None and all(math.isfinite(v) and v > 0 for v in start):
        sc_s, w_s, sb_s = start
        w_s = min(max(w_s, 1e-3), 0.59)
        starts.append([math.log(sc_s), -math.log(0.6 / w_s - 1.0), math.log(max(sb_s / sc_s - 1.5, 1e-3))])
    else:
        for wi in (-2.0, 0.0):
            for ti in (0.0, 1.0):
                starts.append([math.log(s0 * 0.8), wi, ti])
    best = None
    for t0 in starts:
        o = minimize(nll3, t0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=3000))
        if best is None or o.fun < best.fun:
            best = o
    assert best is not None
    sc, w, sb = unpack(best.x)
    bic3 = 2.0 * float(best.fun) + 3.0 * math.log(n)
    mix = bic3 < bic1
    if not mix:
        sc, sb, w = sc1, sc1, 0.0
    return dict(n=int(n), sc=sc, sb=sb, w=w, mix_by_bic=bool(mix), s_eq=(1 - w) * sc + w * sb, sc1=sc1,
                raw=(unpack(best.x) if mix else (sc1, 0.05, sc1 * 2.0)))


def _fit_narrow(counts: NDArray[np.float64], edges: NDArray[np.float64] = _DZ_EDGES,
                start: Optional[Tuple[float, float, float]] = None) -> Tuple[float, float, float, Tuple[float, float, float]]:
    """Narrow component of dz of re-switch-on pairs: mixture of two Gaussians (the same molecule, narrow; its
    neighbours, wide). Returns (narrow sd / sqrt 2, narrow share, wide sd / sqrt 2, raw start for a refit)."""
    counts = np.asarray(counts, dtype=np.float64)
    nz = counts > 0
    c = counts[nz]
    if c.sum() < 200:
        return _NAN, _NAN, _NAN, (_NAN, _NAN, _NAN)

    def unpack(t: Any) -> Tuple[float, float, float]:
        s1 = math.exp(t[0])
        w = 1.0 / (1.0 + math.exp(-t[1]))
        s2 = s1 * (1.5 + math.exp(t[2]))
        return s1, w, s2

    def nll(t: Any) -> float:
        s1, w, s2 = unpack(t)
        p = ((1 - w) * _bin_probs(edges, s1) + w * _bin_probs(edges, s2))[nz]
        return -float(np.sum(c * np.log(np.maximum(p, 1e-300))))

    starts: List[List[float]]
    if start is not None and all(math.isfinite(v) for v in start):
        s1, w, s2 = start
        starts = [[math.log(s1), math.log(max(w, 1e-3) / max(1 - w, 1e-3)), math.log(max(s2 / s1 - 1.5, 1e-3))]]
    else:
        starts = [[math.log(40), -1, 1], [math.log(60), 0, 0.5], [math.log(30), 1, 1.5]]
    best = None
    for t0 in starts:
        o = minimize(nll, t0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=3000))
        if best is None or o.fun < best.fun:
            best = o
    assert best is not None
    s1, w, s2 = unpack(best.x)
    return s1 / _SQ2, 1.0 - w, s2 / _SQ2, (s1, w, s2)


def _total_sigma(core: float, bad: float, w: float, delta: float) -> float:
    """Equivalent sd with the slow error added in quadrature to both components of the mixture."""
    return (1.0 - w) * math.hypot(core, delta) + w * math.hypot(bad, delta)


# ----------------------------------------------------------------------------------------------------
# Pairs
# ----------------------------------------------------------------------------------------------------
def _hist(v: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.histogram(v, bins=_DZ_EDGES)[0].astype(np.float64)


def _block_hists(dz: NDArray[np.float64], block: NDArray[np.int64], n_blocks: int) -> NDArray[np.float64]:
    out = np.zeros((n_blocks, _DZ_EDGES.size - 1))
    if dz.size == 0:
        return out
    idx = np.clip(np.searchsorted(_DZ_EDGES, dz, side="right") - 1, 0, _DZ_EDGES.size - 2)
    inside = (dz >= _DZ_EDGES[0]) & (dz < _DZ_EDGES[-1])
    np.add.at(out, (block[inside], idx[inside]), 1.0)
    return out


def _percentile(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray([x for x in values if math.isfinite(x)], dtype=np.float64)
    if v.size < max(10, int(0.5 * len(values))):
        return _NAN, _NAN
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _empty(status: str, warnings: List[str], **kw: Any) -> AxialSigma:
    return AxialSigma(status=status, reliable=False, warnings=tuple(warnings), **kw)


def measure_axial_sigma(x_nm: Any, y_nm: Any, z_nm: Any, frame: Any = None, *, lp: Any = None, lpz: Any = None,
                        groups: Any = None, n_boot: int = N_BOOT_DEFAULT, seed: int = 0) -> AxialSigma:
    """Measure the axial error of one file from its localizations (see the module docstring).

    ``frame`` is required (None -> status ``no_frame``). ``groups`` (optional, one label per localization, e.g. the
    axon) makes the intervals resample groups instead of frame blocks when there are at least ``MIN_GROUPS``.
    ``lp`` and ``lpz`` are only reported. Never raises on data: every way of not being able to measure is a status
    with its warnings."""
    warnings: List[str] = []
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    z = np.asarray(z_nm, dtype=np.float64).reshape(-1)
    if frame is None:
        warnings.append("no frame column: the same molecule cannot be told from its neighbours, so sigma_z is not "
                        "measurable in this file")
        return _empty(STATUS_NO_FRAME, warnings, n_locs=int(x.size))
    fr_raw = np.asarray(frame, dtype=np.float64).reshape(-1)
    if not (x.size == y.size == z.size == fr_raw.size):
        warnings.append("x, y, z and frame differ in length")
        return _empty(STATUS_NO_FRAME, warnings, n_locs=int(x.size))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(fr_raw)
    if np.any(fr_raw[ok] != np.round(fr_raw[ok])):
        warnings.append("the frame column holds non-integer values")
        return _empty(STATUS_NO_FRAME, warnings, n_locs=int(x.size))
    lpz_all = None if lpz is None else np.asarray(lpz, dtype=np.float64).reshape(-1)
    lp_all = None if lp is None else np.asarray(lp, dtype=np.float64).reshape(-1)
    grp_all = None if groups is None else np.asarray(groups).reshape(-1)
    if grp_all is not None and grp_all.size != x.size:
        grp_all = None
    x, y, z = x[ok], y[ok], z[ok]
    fr = fr_raw[ok].astype(np.int64)
    n = int(x.size)
    lpz_med = _NAN
    if lpz_all is not None and lpz_all.size == ok.size:
        v = lpz_all[ok]
        v = v[np.isfinite(v) & (v > 0)]
        lpz_med = float(np.median(v)) if v.size else _NAN
    if lp_all is not None and lp_all.size == ok.size:
        v = lp_all[ok]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size and float(np.median(v)) > 0.5 * R_FRAME_NM:
            warnings.append(f"median lateral precision {float(np.median(v)):.1f} nm is above half the pairing "
                            f"radius ({R_FRAME_NM:g} nm): pairs of the same molecule are lost")
    if n < MIN_LOCS:
        warnings.append(f"{n} localizations (fewer than {MIN_LOCS})")
        return _empty(STATUS_TOO_FEW_LOCS, warnings, n_locs=n, lpz_median_nm=lpz_med)
    grp = grp_all[ok] if grp_all is not None else None

    # ---- consecutive-frame pairs and the controls against other molecules ----
    pts = np.column_stack([x, y, fr.astype(np.float64) * R_BOX_NM])
    tree = cKDTree(pts)
    pr = tree.query_pairs(R_BOX_NM, p=np.inf, output_type="ndarray")
    a, b = pr[:, 0], pr[:, 1]
    dfr = fr[b] - fr[a]
    swap = dfr < 0
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    dfr = np.abs(dfr)
    d = np.hypot(x[b] - x[a], y[b] - y[a])
    keep = (d < R_BOX_NM) & (dfr <= 1)
    a, b, dfr, d = a[keep], b[keep], dfr[keep], d[keep]
    same = dfr == 0
    n_same = int(same.sum())
    same_frame_fraction = float(np.unique(np.concatenate([a[same], b[same]])).size / n) if n_same else 0.0
    l1 = dfr == 1
    A1, B1, D1 = a[l1], b[l1], d[l1]
    close = D1 < R_UNIQUE_NM
    cnt_a = np.bincount(A1[close], minlength=n)
    cnt_b = np.bincount(B1[close], minlength=n)
    unique = close & (cnt_a[A1] == 1) & (cnt_b[B1] == 1)
    sel = unique & (D1 < R_FRAME_NM)
    n_close = int((D1 < R_FRAME_NM).sum())
    n_ring = int(((D1 >= R_RING_IN_NM) & (D1 < R_BOX_NM)).sum())
    n_pairs = int(sel.sum())
    # other molecules in the 25 nm disc, scaled from the 100-300 nm ring: area (homogeneous) and line (pessimistic)
    cont_area = n_ring * (R_FRAME_NM ** 2 / (R_BOX_NM ** 2 - R_RING_IN_NM ** 2)) / max(n_close, 1)
    cont_line = n_ring * (R_FRAME_NM / (R_BOX_NM - R_RING_IN_NM)) / max(n_close, 1)
    common: Dict[str, Any] = dict(n_locs=n, n_pairs_frame=n_pairs, contamination=cont_line,
                                  contamination_area=cont_area, same_frame_fraction=same_frame_fraction,
                                  lpz_median_nm=lpz_med)
    if same_frame_fraction > SAME_FRAME_MAX or cont_line > CONTAM_MAX:
        warnings.append(
            f"dense acquisition: {100 * same_frame_fraction:.1f} % of the localizations have another one in the same "
            f"frame within {R_BOX_NM:g} nm and the line-scaled share of other molecules among the pairs is "
            f"{100 * cont_line:.0f} %: localizations of different molecules would be taken for the same one")
        return _empty(STATUS_CONTAMINATED, warnings, **common)
    if same_frame_fraction > SAME_FRAME_WARN or cont_line > CONTAM_WARN:
        warnings.append(
            f"density is not low: {100 * same_frame_fraction:.1f} % of the localizations share a frame with another "
            f"within {R_BOX_NM:g} nm, line-scaled contamination of the pairs {100 * cont_line:.0f} % (area-scaled "
            f"{100 * cont_area:.1f} %): sigma_z may be overestimated")
    if n_pairs < MIN_PAIRS_FRAME:
        warnings.append(f"{n_pairs} unique consecutive-frame pairs (fewer than {MIN_PAIRS_FRAME})")
        return _empty(STATUS_TOO_FEW_PAIRS, warnings, **common)

    dz_frame = z[B1[sel]] - z[A1[sel]]
    t_pair = fr[A1[sel]]

    # ---- re-switch-on pairs at < 10 nm: lag 1 (reference) and lag >= 1001 (slow component) ----
    span = int(fr.max() - fr.min() + 1)
    slow_ok = span > 2 * LONG_LAG_FRAMES
    dz1 = np.empty(0)
    t1 = np.empty(0, dtype=np.int64)
    dzL = np.empty(0)
    tL = np.empty(0, dtype=np.int64)
    if not slow_ok:
        warnings.append(f"the acquisition spans {span} frames (needs more than {2 * LONG_LAG_FRAMES}): the slow "
                        "component cannot be measured")
    else:
        idx_all = np.arange(n)
        t2 = cKDTree(np.column_stack([x, y]))
        n_est = float(np.sum(t2.query_ball_point(np.column_stack([x, y]), R_LAG_NM, return_length=True)) - n) / 2.0
        if n_est > MAX_PAIRS_SLOW:
            rng_s = np.random.default_rng(seed + 1)
            keep_frac = math.sqrt(MAX_PAIRS_SLOW / n_est)
            idx_all = np.sort(rng_s.choice(n, size=max(int(n * keep_frac), 100), replace=False))
            t2 = cKDTree(np.column_stack([x[idx_all], y[idx_all]]))
            warnings.append(f"very many pairs at {R_LAG_NM:g} nm: {100 * keep_frac:.0f} % of the localizations were "
                            "used for the slow component")
        pr2 = t2.query_pairs(R_LAG_NM, output_type="ndarray")
        i2, j2 = idx_all[pr2[:, 0]], idx_all[pr2[:, 1]]
        lag = np.abs(fr[j2] - fr[i2])
        dzp = z[j2] - z[i2]
        first = np.minimum(fr[i2], fr[j2])
        m1 = lag == 1
        mL = lag >= LONG_LAG_FRAMES
        dz1, t1 = dzp[m1], first[m1]
        dzL, tL = dzp[mL], first[mL]
        del pr2, i2, j2, lag, dzp, first
    n_lag1, n_long = int(dz1.size), int(dzL.size)
    common.update(n_pairs_lag1=n_lag1, n_pairs_long=n_long)

    # ---- point estimates ----
    ff = _fit_frame_mixture(_hist(dz_frame))
    sigma_frame = float(ff["s_eq"])
    slow_measurable = slow_ok and n_lag1 >= MIN_PAIRS_LAG1 and n_long >= MIN_PAIRS_LONG
    if slow_ok and not slow_measurable:
        warnings.append(f"re-switch-on pairs: {n_lag1} at lag 1 and {n_long} at lag >= {LONG_LAG_FRAMES} "
                        f"(needs {MIN_PAIRS_LAG1} and {MIN_PAIRS_LONG}): the slow component cannot be measured")
    delta = _NAN
    raw1 = rawL = (_NAN, _NAN, _NAN)
    if slow_measurable:
        s1, _w1, _b1, raw1 = _fit_narrow(_hist(dz1))
        sL, _wL, _bL, rawL = _fit_narrow(_hist(dzL))
        if math.isfinite(s1) and math.isfinite(sL):
            delta = math.sqrt(max(sL * sL - s1 * s1, 0.0))
        else:
            slow_measurable = False
            warnings.append("the narrow component of the re-switch-on pairs could not be fitted")
    total = _total_sigma(float(ff["sc"]), float(ff["sb"]), float(ff["w"]), delta) if slow_measurable else _NAN

    # ---- intervals: resample blocks of frames (or groups) ----
    f_lo, f_hi = int(fr.min()), int(fr.max())
    span1 = max(f_hi - f_lo + 1, 1)

    def frame_block(t: NDArray[np.int64]) -> NDArray[np.int64]:
        return np.clip(((t - f_lo) * N_BLOCKS) // span1, 0, N_BLOCKS - 1).astype(np.int64)

    blk_1, blk_L = frame_block(t1), frame_block(tL)
    n_blocks_slow = N_BLOCKS
    if grp is not None and np.unique(grp).size >= MIN_GROUPS:
        # a pair belongs to the group of its first member; the slow pairs keep frame blocks (their members are not
        # tracked back to a group), and the warning says so
        labels, inv = np.unique(grp, return_inverse=True)
        n_blocks = int(labels.size)
        resampled_by = "groups"
        blk_f = inv[A1[sel]].astype(np.int64)
        warnings.append("resampled by group for the frame error and by frame block for the slow component")
    else:
        n_blocks = N_BLOCKS
        resampled_by = "frame blocks"
        blk_f = frame_block(t_pair)
    Hf = _block_hists(dz_frame, blk_f, n_blocks)
    H1 = _block_hists(dz1, blk_1, n_blocks_slow)
    HL = _block_hists(dzL, blk_L, n_blocks_slow)
    rng = np.random.default_rng(seed)
    b_frame: List[float] = []
    b_slow: List[float] = []
    b_total: List[float] = []
    start_f = ff["raw"]
    for _ in range(max(int(n_boot), 0)):
        pick = rng.integers(0, n_blocks, size=n_blocks)
        fb = _fit_frame_mixture(Hf[pick].sum(axis=0), start=(start_f[0], start_f[1], start_f[2]))
        b_frame.append(float(fb["s_eq"]))
        if slow_measurable:
            pick2 = rng.integers(0, n_blocks_slow, size=n_blocks_slow) if resampled_by == "groups" else pick
            s1b, _, _, _ = _fit_narrow(H1[pick2].sum(axis=0), start=raw1)
            sLb, _, _, _ = _fit_narrow(HL[pick2].sum(axis=0), start=rawL)
            if math.isfinite(s1b) and math.isfinite(sLb):
                db = math.sqrt(max(sLb * sLb - s1b * s1b, 0.0))
                b_slow.append(db)
                b_total.append(_total_sigma(float(fb["sc"]), float(fb["sb"]), float(fb["w"]), db))
    f_lo_ci, f_hi_ci = _percentile(b_frame) if n_boot else (_NAN, _NAN)
    d_lo, d_hi = _percentile(b_slow) if (n_boot and slow_measurable) else (_NAN, _NAN)
    t_lo, t_hi = _percentile(b_total) if (n_boot and slow_measurable) else (_NAN, _NAN)
    lpz_ratio = sigma_frame / lpz_med if math.isfinite(lpz_med) and lpz_med > 0 else _NAN
    base = dict(common, frame_nm=sigma_frame, frame_lo_nm=f_lo_ci, frame_hi_nm=f_hi_ci, slow_nm=delta,
                slow_lo_nm=d_lo, slow_hi_nm=d_hi, total_nm=total, total_lo_nm=t_lo, total_hi_nm=t_hi,
                core_nm=float(ff["sc"]), bad_fraction=float(ff["w"]), bad_nm=float(ff["sb"]),
                mixture_by_bic=bool(ff["mix_by_bic"]), ratio_to_lpz=lpz_ratio, n_boot=int(n_boot),
                resampled_by=resampled_by)
    if not slow_measurable:
        warnings.append("only the frame-to-frame error was measured: it leaves out the slow component and is not "
                        "used as the width of the leak")
        return AxialSigma(status=STATUS_FRAME_ONLY, reliable=False, warnings=tuple(warnings), **base)
    if not (SIGMA_MIN_NM <= total <= SIGMA_MAX_NM):
        warnings.append(f"measured sigma_z {total:.0f} nm is outside {SIGMA_MIN_NM:g}-{SIGMA_MAX_NM:g} nm")
        return AxialSigma(status=STATUS_INTERVAL_TOO_WIDE, reliable=False, warnings=tuple(warnings), **base)
    if not (math.isfinite(t_lo) and math.isfinite(t_hi)):
        warnings.append("no interval could be computed (resamples did not fit)")
        return AxialSigma(status=STATUS_INTERVAL_TOO_WIDE, reliable=False, warnings=tuple(warnings), **base)
    if (t_hi - t_lo) > MAX_REL_WIDTH * total:
        warnings.append(f"the interval {t_lo:.0f}-{t_hi:.0f} nm is wider than {100 * MAX_REL_WIDTH:.0f} % of the "
                        f"value {total:.0f} nm")
        return AxialSigma(status=STATUS_INTERVAL_TOO_WIDE, reliable=False, warnings=tuple(warnings), **base)
    return AxialSigma(status=STATUS_MEASURED, reliable=True, warnings=tuple(warnings), **base)


def leak_widths(res: Optional[AxialSigma]) -> Optional[Tuple[float, float, float]]:
    """(low, main, high) in nm of the width the leak of C should use, or None when the measurement is not reliable.
    The interval is clipped so that low <= main <= high."""
    if res is None or not res.reliable:
        return None
    lo = min(res.total_lo_nm, res.total_nm)
    hi = max(res.total_hi_nm, res.total_nm)
    return float(lo), float(res.total_nm), float(hi)
