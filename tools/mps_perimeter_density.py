"""
Projection onto the contour, band, arc grid and density profiles of the perimeter density analysis (package P4).

Pure, deterministic, numpy only. What this module does, in order:

1. ``project``: arc coordinate ``s`` and signed radial offset ``r`` (positive outside) of every localization
   with respect to a given closed contour (``SmoothPath``).
2. ``band_width``: half-width of the band around the contour (fixed 100 nm by default; 50 and 150 nm are the
   mandatory sensitivities and are obtained by calling ``slice_profiles`` with another ``half_width_nm``).
3. ``arc_grid`` / ``density_profile`` / ``peak_bin``: counts per arc bin on a closed uniform grid, the
   not-measured mask and the maximum with a deterministic tie rule.
4. ``slice_profiles``: the three slices on ONE grid and ONE origin (the centre of the central slice's maximum
   bin), which is the "z0_max" frame; ``centred_lags_nm`` is the straight axis centred on 0.
5. ``aligned_on_own_max``: a profile rotated to its own maximum, for question A and for display only.

Conventions
-----------
* The arc grid is circular: the last bin is adjacent to the first. Bin ``j`` is CENTRED on
  ``origin_arc_nm + j * bin_nm_effective`` (mod L), so the origin is the centre of bin 0. Rotating the origin by a
  whole number of bins rotates the profile (``np.roll``) and changes nothing else.
* An empty profile (no localization) is never divided by zero: ``relative`` is all zeros, ``measured`` all
  False, ``max_bin = -1`` and ``max_tie = False``.
* Origin tags on numbers: [director] fixed by the thesis director, [ad hoc] chosen here,
  [derivation] follows from the geometry or the statistics.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.mps_matching import SmoothPath
from tools.mps_perimeter_types import (BIN_NM, OUTSIDE_LABEL, SLICES, UNSUPPORTED_GAP_NM, BandWidth,
                                       DensityProfile, SliceProfiles)
from tools.mps_unroll import project_on_path

# [derivation] FWHM = 2 sqrt(2 ln 2) * sigma for a Gaussian: 2.3548. The radial sigma is the robust one
# (1.4826 * MAD, the MAD-to-sigma factor of a Gaussian), so a few distant localizations do not widen the band.
_FWHM_PER_SIGMA: float = 2.3548200450309493
_MAD_TO_SIGMA: float = 1.482602218505602
_MIN_BINS: int = 8                         # [ad hoc] below 8 bins a circular profile has no structure to correlate
_FRACTION_RADII_NM: Tuple[float, float, float] = (50.0, 100.0, 150.0)   # [ad hoc] radii of the reported fractions
_BAND_MODES: Tuple[str, ...] = ("fixed", "fwhm")
_EDGE_EPS: float = 1e-9                    # [ad hoc] relative slack for "bin lies inside a gap" comparisons


# ----------------------------------------------------------------------------------------------------
# Projection and band
# ----------------------------------------------------------------------------------------------------
def project(path: SmoothPath, x_nm: NDArray[np.float64],
            y_nm: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Arc coordinate s and signed radial offset r (positive outside) of every point.

    ``s`` is in [0, L) along the path's own arc. "Outside" is the side away from the polygon's interior and is
    derived from the orientation of the path (shoelace area), so clockwise and counter-clockwise paths give the
    same r. Reuses ``mps_unroll.project_on_path`` (not modified).
    """
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"project: x and y differ in length ({x.shape[0]} vs {y.shape[0]})")
    pts = np.column_stack([x, y])
    poly = np.asarray(path.points_nm, dtype=np.float64)
    area2 = float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
    if area2 == 0.0:
        raise ValueError("project: the path encloses no area, so its outside is undefined")
    outward = -1.0 if area2 > 0.0 else 1.0     # counter-clockwise tour: interior on the left, outward = -cross
    return project_on_path(path, pts, outward_sign=outward)


def band_width(r_nm: NDArray[np.float64], *, mode: str = "fixed", fixed_nm: float = 100.0,
               clip_nm: Tuple[float, float] = (50.0, 150.0)) -> BandWidth:
    """Band half-width: fixed, or from the radial FWHM clipped to ``clip_nm``.

    ``fixed``: half-width = ``fixed_nm`` [ad hoc, pilot 2026-11-16; 50 and 150 nm are the sensitivities].
    ``fwhm``: half-width = the radial FWHM of the axon itself, clipped to ``clip_nm``. The FWHM is the
    Gaussian-equivalent one, 2.3548 x 1.4826 x MAD of r [derivation]: sigma 25 nm gives 59 nm. With no usable r
    the "fwhm" mode falls back to the fixed width and says so by reporting mode "fixed".

    Always reported: radial FWHM and robust SD, the fraction of points within +-50 / 100 / 150 nm and the
    fractions beyond the half-width inside (r < -half) and outside (r > +half). All nan when r is empty.
    """
    if mode not in _BAND_MODES:
        raise ValueError(f"band_width: mode must be one of {_BAND_MODES}, got {mode!r}")
    lo, hi = float(clip_nm[0]), float(clip_nm[1])
    if not (0.0 < lo <= hi):
        raise ValueError("band_width: clip_nm must satisfy 0 < lo <= hi")
    if not (fixed_nm > 0.0):
        raise ValueError("band_width: fixed_nm must be positive")
    r = np.asarray(r_nm, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    nan = float("nan")
    if r.size == 0:
        return BandWidth(mode="fixed", half_width_nm=float(fixed_nm), fwhm_nm=nan, robust_sd_nm=nan,
                         frac_within={k: nan for k in _FRACTION_RADII_NM}, frac_inside=nan, frac_outside=nan)
    med = float(np.median(r))
    sd = _MAD_TO_SIGMA * float(np.median(np.abs(r - med)))
    fwhm = _FWHM_PER_SIGMA * sd
    if mode == "fwhm":
        half = float(min(max(fwhm, lo), hi))
    else:
        half = float(fixed_nm)
    n = float(r.size)
    within = {k: float(np.count_nonzero(np.abs(r) <= k)) / n for k in _FRACTION_RADII_NM}
    inside = float(np.count_nonzero(r < -half)) / n
    outside = float(np.count_nonzero(r > half)) / n
    return BandWidth(mode=mode, half_width_nm=half, fwhm_nm=float(fwhm), robust_sd_nm=float(sd),
                     frac_within=within, frac_inside=inside, frac_outside=outside)


# ----------------------------------------------------------------------------------------------------
# Arc grid and profiles
# ----------------------------------------------------------------------------------------------------
def arc_grid(length_nm: float, bin_nm: float = BIN_NM) -> Tuple[int, float]:
    """Return (n_bins, effective bin); n_bins = max(8, round(L / bin)).

    Rounds half up (not banker's rounding) so the result never depends on parity. The effective bin is
    L / n_bins: a uniform CLOSED grid, which circular rotation and the FFT need (a partial last bin would break
    both). It differs from the nominal bin by less than 0.5 / n_bins in relative terms once n_bins > 8.
    """
    if not (length_nm > 0.0) or not np.isfinite(length_nm):
        raise ValueError("arc_grid: length_nm must be positive and finite")
    if not (bin_nm > 0.0) or not np.isfinite(bin_nm):
        raise ValueError("arc_grid: bin_nm must be positive and finite")
    n = max(_MIN_BINS, int(np.floor(length_nm / bin_nm + 0.5)))
    return n, float(length_nm) / n


def peak_bin(counts: NDArray[np.int64]) -> Tuple[int, bool]:
    """Index of the maximum bin and whether it was a tie (tie rule: larger 3-bin sum, then lower index).

    The 3-bin sum is circular (the grid is closed). No smoothing is applied beforehand. An all-zero or empty
    array has no maximum: returns (-1, False).
    """
    c = np.asarray(counts, dtype=np.int64).reshape(-1)
    if c.size == 0 or int(c.max()) <= 0:
        return -1, False
    top = int(c.max())
    cand = np.flatnonzero(c == top)
    if cand.size == 1:
        return int(cand[0]), False
    sums = c[cand] + c[(cand - 1) % c.size] + c[(cand + 1) % c.size]
    best = int(np.argmax(sums))          # argmax returns the first (lowest index, cand is ascending) on equal sums
    return int(cand[best]), True


def _unmeasured_bins(s_sorted: NDArray[np.float64], length_nm: float, n_bins: int, bin_nm_eff: float,
                     origin_arc_nm: float, gap_nm: float) -> NDArray[np.bool_]:
    """True for the bins that lie entirely inside an arc stretch longer than ``gap_nm`` with no localization."""
    out = np.zeros(n_bins, dtype=np.bool_)
    if s_sorted.size == 0:
        out[:] = True
        return out
    nxt = np.roll(s_sorted, -1)
    gaps = nxt - s_sorted
    gaps[-1] += length_nm                # the closing gap wraps through L (one point: the gap is L itself)
    long_gaps = np.flatnonzero(gaps > gap_nm)
    if long_gaps.size == 0:
        return out
    j = np.arange(n_bins, dtype=np.float64)
    lo = origin_arc_nm + (j - 0.5) * bin_nm_eff          # arc of each bin's lower edge
    slack = _EDGE_EPS * length_nm
    for g in long_gaps:
        d = np.mod(lo - s_sorted[g], length_nm)           # offset of the bin start from the gap start
        out |= (d + bin_nm_eff <= gaps[g] + slack) & (d >= -slack)
    return out


def _check_arcs(s_nm: NDArray[np.float64], length_nm: float) -> NDArray[np.float64]:
    if not (length_nm > 0.0) or not np.isfinite(length_nm):
        raise ValueError("length_nm must be positive and finite")
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    if not np.isfinite(s).all():
        raise ValueError("non-finite arc coordinate")
    return np.mod(s, length_nm)


def density_profile(s_nm: NDArray[np.float64], length_nm: float, *, n_bins: int, origin_arc_nm: float = 0.0,
                    unsupported_gap_nm: float = UNSUPPORTED_GAP_NM) -> DensityProfile:
    """Counts per arc bin with the not-measured mask.

    Bin j is centred on ``origin_arc_nm + j * L / n_bins`` (circular). ``measured[j]`` is False when the bin lies
    entirely inside a stretch of arc longer than ``unsupported_gap_nm`` [ad hoc, 400 nm] between two consecutive
    localizations (the contour there is an interpolation, not data). Conservation: ``counts.sum() == n_locs ==``
    number of points given. Ties and empty profiles: see ``peak_bin`` and the module docstring.
    """
    if n_bins < 1:
        raise ValueError("density_profile: n_bins must be >= 1")
    s = _check_arcs(s_nm, length_nm)
    w = float(length_nm) / n_bins
    idx = np.mod(np.floor((s - origin_arc_nm) / w + 0.5).astype(np.int64), n_bins)
    counts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    top = int(counts.max()) if counts.size else 0
    relative = counts / float(top) if top > 0 else np.zeros(n_bins, dtype=np.float64)
    measured = ~_unmeasured_bins(np.sort(s), float(length_nm), n_bins, w, float(origin_arc_nm),
                                 float(unsupported_gap_nm))
    mx, tie = peak_bin(counts)
    return DensityProfile(counts=counts, relative=np.asarray(relative, dtype=np.float64), measured=measured,
                          n_bins=int(n_bins), bin_nm_effective=w, origin_arc_nm=float(origin_arc_nm) % float(length_nm),
                          max_bin=mx, max_tie=tie, n_locs=int(counts.sum()))


def slice_profiles(s_nm: NDArray[np.float64], r_nm: NDArray[np.float64], slice_label: NDArray[np.int8],
                   length_nm: float, *, half_width_nm: float, bin_nm: float = BIN_NM) -> SliceProfiles:
    """The three slices' profiles on one grid and origin (frame "z0_max").

    A localization counts in slice k when ``slice_label == k`` and ``|r| <= half_width_nm`` (the band is
    symmetric: with an asymmetric band the density per unit arc would carry a spurious curvature term
    [derivation]). The origin of all three profiles is the centre of the central slice's maximum bin, so the
    three are rotated by the same amount and their correlations do not change. If the central slice is empty
    the origin stays at arc 0. Labels other than -1/0/+1 (``OUTSIDE_LABEL``) are ignored.
    """
    if not (half_width_nm > 0.0):
        raise ValueError("slice_profiles: half_width_nm must be positive")
    s = _check_arcs(s_nm, length_nm)
    r = np.asarray(r_nm, dtype=np.float64).reshape(-1)
    lab = np.asarray(slice_label).reshape(-1)
    if not (s.shape == r.shape == lab.shape):
        raise ValueError("slice_profiles: s, r and slice_label must have the same length")
    in_band = np.isfinite(r) & (np.abs(r) <= half_width_nm)
    n_bins, w = arc_grid(length_nm, bin_nm)

    def arcs_of(k: int) -> NDArray[np.float64]:
        return s[in_band & (lab == k)]

    base = density_profile(arcs_of(0), length_nm, n_bins=n_bins, origin_arc_nm=0.0)
    origin = base.max_bin * w if base.max_bin >= 0 else 0.0
    by_slice: Dict[int, DensityProfile] = {}
    for k in SLICES:
        by_slice[k] = density_profile(arcs_of(k), length_nm, n_bins=n_bins, origin_arc_nm=origin)
    return SliceProfiles(by_slice=by_slice, length_nm=float(length_nm), frame="z0_max")


# ----------------------------------------------------------------------------------------------------
# Straight axis and own-maximum alignment
# ----------------------------------------------------------------------------------------------------
def centred_lags_nm(n_bins: int, bin_nm_effective: float) -> NDArray[np.float64]:
    """Axis in (-L/2, L/2]: the perimeter as a straight line centred on 0, ascending.

    Bin j of a profile whose bin 0 is the maximum sits at ``j * bin`` for j <= n/2 and at ``(j - n) * bin``
    beyond. This returns the ascending axis; ``centred_order(n_bins)`` gives the matching bin indices, so
    ``profile.relative[centred_order(n)]`` is plotted against ``centred_lags_nm(n, w)``. The sign of the
    distance means nothing unless the cut's orientation is known; question A folds it to |d|.
    """
    if n_bins < 1:
        raise ValueError("centred_lags_nm: n_bins must be >= 1")
    return np.asarray(np.arange(-((n_bins - 1) // 2), n_bins // 2 + 1, dtype=np.float64) * bin_nm_effective)


def centred_order(n_bins: int) -> NDArray[np.int64]:
    """Bin indices that sort a profile (bin 0 = origin) onto ``centred_lags_nm``."""
    if n_bins < 1:
        raise ValueError("centred_order: n_bins must be >= 1")
    return np.asarray(np.mod(np.arange(-((n_bins - 1) // 2), n_bins // 2 + 1, dtype=np.int64), n_bins))


def aligned_on_own_max(p: DensityProfile) -> DensityProfile:
    """Rotate a profile to its own maximum (bin 0 = maximum); for question A and for display only.

    ``origin_arc_nm`` moves to the arc of that maximum. A profile with no maximum (empty) is returned
    unchanged. A profile aligned this way must never be correlated against another slice: wrap a set of them
    with ``own_max_profiles`` so ``SliceProfiles.frame`` says "own_max".
    """
    if p.max_bin < 0 or p.n_bins < 1:
        return p
    k = int(p.max_bin)
    length = p.bin_nm_effective * p.n_bins
    return replace(p, counts=np.roll(p.counts, -k), relative=np.roll(p.relative, -k),
                   measured=np.roll(p.measured, -k), max_bin=0,
                   origin_arc_nm=float((p.origin_arc_nm + k * p.bin_nm_effective) % length))


def own_max_profiles(sp: SliceProfiles) -> SliceProfiles:
    """Each slice aligned on its own maximum, tagged frame "own_max" (not correlatable across slices)."""
    return SliceProfiles(by_slice={k: aligned_on_own_max(v) for k, v in sp.by_slice.items()},
                         length_nm=sp.length_nm, frame="own_max")


__all__ = ["project", "band_width", "arc_grid", "density_profile", "peak_bin", "slice_profiles",
           "centred_lags_nm", "centred_order", "aligned_on_own_max", "own_max_profiles", "OUTSIDE_LABEL"]
