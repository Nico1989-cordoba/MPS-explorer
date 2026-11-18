"""
Slices and focus of the perimeter density analysis (package P2).

The height z0 of the central slice can be chosen three ways (``Z0Criterion``): the localization PEAK in z
(default), the nominal FOCUS (z = 0 of the file, via the calibration record) or the empirical FOCUS (where the
median of sx - sy crosses zero). Both foci are always estimated and both distances to the peak are always
reported, signed and absolute, whatever criterion picks z0. Three contiguous slices of 170 nm (-1, 0, +1) are
then cut around z0. A lateral slice is never dropped for having few localizations: its count and its percentage
of the central slice are reported instead.

Pure functions, deterministic, numpy only (no Qt, no new dependency).
Origin tags on numbers: [pilot] real-data pilot of 2026-11-16, [ad hoc] chosen here, [derivation] follows from
the geometry, [contract] constant from ``mps_perimeter_types``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.mps_perimeter_types import (
    FOCUS_BIN_NM,
    FOCUS_MIN_PER_BIN,
    OUTSIDE_LABEL,
    SLICE_THICKNESS_NM,
    SLICES,
    FocusEstimate,
    SliceSet,
    Status,
    Z0Choice,
    Z0Criterion,
)

if TYPE_CHECKING:
    from tools.mps_zcalibration import CalibrationRecord

# [pilot] Gaussian width of the kernel density whose maximum is the z peak. The pilot (2026-11-16, 18 axons)
# found the KDE peak with 25 nm within a median of 7.7 nm (max 36 nm) of the programme's mixture-model density
# peak (``fit_z_periodicity(z).main_peak_nm``); 25 nm is under a seventh of the 170 nm period, so the peak is
# not smeared across rings, and above the ~10 nm axial precision, so it is not noise-driven.
PEAK_KDE_SIGMA_NM: float = 25.0
# [ad hoc] histogram step (nm) on which the kernel density is evaluated; 1 nm is far below the 25 nm kernel.
PEAK_GRID_NM: float = 1.0


def _finite(a: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(a, dtype=np.float64).ravel()
    return np.asarray(arr[np.isfinite(arr)], dtype=np.float64)


def z_peak_nm(z_nm: NDArray[np.float64]) -> float:
    """Height of the highest localization density in z (nm); ``nan`` without finite z.

    Gaussian kernel density (sigma ``PEAK_KDE_SIGMA_NM``) on a ``PEAK_GRID_NM`` grid; the first maximum wins a
    tie. Deterministic and numpy only, unlike the mixture model of ``fit_z_periodicity`` it stands in for (the
    pilot compares the two: median 7.7 nm apart).
    """
    z = _finite(z_nm)
    if z.size == 0:
        return float("nan")
    if z.size == 1 or float(z.max()) == float(z.min()):
        return float(z[0])
    pad = 4.0 * PEAK_KDE_SIGMA_NM                      # [derivation] the kernel is negligible beyond 4 sigma
    lo = math.floor((float(z.min()) - pad) / PEAK_GRID_NM) * PEAK_GRID_NM
    n_bins = int(math.ceil((float(z.max()) + pad - lo) / PEAK_GRID_NM)) + 1
    edges = lo + PEAK_GRID_NM * np.arange(n_bins + 1)
    counts, _ = np.histogram(z, bins=edges)
    half = int(math.ceil(pad / PEAK_GRID_NM))
    x = np.arange(-half, half + 1) * PEAK_GRID_NM
    kernel = np.exp(-0.5 * (x / PEAK_KDE_SIGMA_NM) ** 2)
    dens = np.convolve(counts.astype(np.float64), kernel, mode="same")
    return float(lo + PEAK_GRID_NM * (int(np.argmax(dens)) + 0.5))


def _empirical_focus(z: NDArray[np.float64], sx: NDArray[np.float64], sy: NDArray[np.float64],
                     bin_nm: float, min_per_bin: int) -> Tuple[Optional[float], int, List[str]]:
    """z where the per-bin median of (sx - sy) crosses zero, by linear interpolation between adjacent bins."""
    warn: List[str] = []
    diff = sx - sy
    if diff.size == 0:
        return None, 0, ["empirical focus: no localizations with finite z, sx and sy"]
    if float(np.ptp(diff)) == 0.0:
        return None, 0, ["empirical focus: sx equals sy for every localization (constant), so there is no crossing"]
    first = math.floor(float(z.min()) / bin_nm)
    idx = np.floor(z / bin_nm).astype(np.int64) - first
    n_bins = int(idx.max()) + 1
    centres: List[float] = []
    medians: List[float] = []
    bin_ids: List[int] = []
    for b in range(n_bins):
        sel = diff[idx == b]
        if sel.size >= min_per_bin:
            centres.append((first + b + 0.5) * bin_nm)
            medians.append(float(np.median(sel)))
            bin_ids.append(b)
    n_used = len(centres)
    if n_used < 2:
        return None, n_used, [f"empirical focus: fewer than 2 z bins with >= {min_per_bin} localizations"]
    roots: List[float] = []
    for k in range(n_used - 1):
        if bin_ids[k + 1] != bin_ids[k] + 1:           # a gap of empty bins: no interpolation across it
            continue
        m0, m1 = medians[k], medians[k + 1]
        if m0 == 0.0:
            roots.append(centres[k])
        elif m0 * m1 < 0.0:
            roots.append(centres[k] + (0.0 - m0) * (centres[k + 1] - centres[k]) / (m1 - m0))
    if medians[-1] == 0.0:
        roots.append(centres[-1])
    roots = sorted(set(roots))
    if not roots:
        return None, n_used, ["empirical focus: the median of sx - sy does not cross zero inside the z range"]
    root = min(roots, key=abs)
    if len(roots) > 1:
        warn.append(f"empirical focus: {len(roots)} zero crossings of sx - sy; the one nearest z = 0 was used")
    return float(root), n_used, warn


def estimate_focus(z_nm: NDArray[np.float64], sx: Optional[NDArray[np.float64]], sy: Optional[NDArray[np.float64]],
                   record: Optional["CalibrationRecord"], *, bin_nm: float = FOCUS_BIN_NM,
                   min_per_bin: int = FOCUS_MIN_PER_BIN) -> FocusEstimate:
    """Nominal focus (z = 0 of the file, via the record) and empirical focus (sx = sy crossing).

    Nominal: the record's wx = wy crossing times the magnification (Picasso multiplies z by m), in the file's z;
    ``None`` without a record, curves or magnification. Empirical: per-bin medians of sx - sy (``bin_nm``, bins
    with at least ``min_per_bin`` localizations) and the zero crossing nearest z = 0; ``None`` without sx/sy, with
    sx = sy constant (simulated files), or without a crossing. A missing focus is never replaced by another.
    """
    warnings: List[str] = []
    nominal: Optional[float] = None
    if record is None:
        warnings.append("nominal focus: no 3D calibration record")
    elif not record.has_curves:
        warnings.append("nominal focus: the calibration record has no width curves")
    elif record.magnification is None:
        warnings.append("nominal focus: the calibration record has no magnification")
    else:
        crossing = record.crossing_nm()
        if crossing is None:
            warnings.append("nominal focus: the calibration curves do not cross")
        else:
            nominal = float(crossing) * float(record.magnification)

    empirical: Optional[float] = None
    n_bins = 0
    if sx is None or sy is None:
        warnings.append("empirical focus: no sx / sy columns")
    else:
        z_all = np.asarray(z_nm, dtype=np.float64).ravel()
        sx_all = np.asarray(sx, dtype=np.float64).ravel()
        sy_all = np.asarray(sy, dtype=np.float64).ravel()
        if not (z_all.size == sx_all.size == sy_all.size):
            raise ValueError("z, sx and sy must have the same length")
        ok = np.isfinite(z_all) & np.isfinite(sx_all) & np.isfinite(sy_all)
        empirical, n_bins, w = _empirical_focus(z_all[ok], sx_all[ok], sy_all[ok], float(bin_nm), int(min_per_bin))
        warnings.extend(w)

    parts = []
    if nominal is not None:
        parts.append("record")
    if empirical is not None:
        parts.append("sx-sy")
    return FocusEstimate(nominal_nm=nominal, empirical_nm=empirical, empirical_n_bins=int(n_bins),
                         source="+".join(parts) if parts else "none", warnings=tuple(warnings))


def _minus(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else float(a - b)


def choose_z0(z_nm: NDArray[np.float64], focus: FocusEstimate, *, criterion: Z0Criterion) -> Z0Choice:
    """Pick z0 by criterion; a focus criterion without a focus gives status "no focus", never the peak.

    The peak and both distances (peak - nominal focus, peak - empirical focus) are filled whenever they can be
    known, whatever the criterion. ``peak_minus_focus_nm`` is the distance to the focus of the selected kind
    (the nominal focus when the criterion is PEAK, as the contract says).
    """
    p = z_peak_nm(z_nm)
    peak: Optional[float] = p if math.isfinite(p) else None
    d_nom = _minus(peak, focus.nominal_nm)
    d_emp = _minus(peak, focus.empirical_nm)
    d_sel = d_emp if criterion == Z0Criterion.FOCUS_EMPIRICAL else d_nom

    z0: Optional[float]
    status = Status.OK
    if criterion == Z0Criterion.PEAK:
        z0 = peak
    elif criterion == Z0Criterion.FOCUS_NOMINAL:
        z0 = focus.nominal_nm
    else:
        z0 = focus.empirical_nm
    if z0 is None:
        status = Status.TOO_FEW_LOCS if (criterion == Z0Criterion.PEAK) else Status.NO_FOCUS
    return Z0Choice(z0_nm=z0, criterion=criterion, peak_nm=peak,
                    focus_nominal_nm=focus.nominal_nm, focus_empirical_nm=focus.empirical_nm,
                    peak_minus_focus_nm=d_sel, peak_minus_focus_nominal_nm=d_nom,
                    peak_minus_focus_empirical_nm=d_emp, status=status)


def cut_slices(z_nm: NDArray[np.float64], z0_nm: float, *, thickness_nm: float = SLICE_THICKNESS_NM,
               guard_nm: float = 0.0) -> SliceSet:
    """Cut the slices -1, 0, +1; non-overlapping, never dropped for having few localizations.

    Slice 0 is [z0 - T/2, z0 + T/2); slices -1 and +1 follow contiguously, each of thickness T. Intervals are
    half-open (lower edge in, upper edge out), so a localization on a shared edge belongs to exactly one slice.
    A guard g > 0 trims g from both sides of every slice (each is then T - 2g thick, e.g. 85 nm for g = 42.5):
    localizations in the guard gaps are labelled OUTSIDE_LABEL. ``bounds_nm`` are the effective bounds.
    ``z_coverage`` is the fraction of each effective interval inside [min z, max z] of the finite localizations
    (< 1 when the file's z range truncates it). ``label`` has the input length: it is positional, so a caller
    traces back to the raw data through its own ``source_index`` array (see ``slice_members``).
    """
    t = float(thickness_nm)
    g = float(guard_nm)
    z0 = float(z0_nm)
    if not math.isfinite(z0):
        raise ValueError("z0 must be finite")
    if not (t > 0.0 and g >= 0.0 and 2.0 * g < t):
        raise ValueError("need thickness > 0, guard >= 0 and 2 * guard < thickness")
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    finite = np.isfinite(z)
    zf = z[finite]

    b: List[Tuple[float, float]] = []
    for s in SLICES:
        lo = z0 + (s - 0.5) * t + g
        hi = z0 + (s + 0.5) * t - g
        b.append((lo, hi))
    label = np.full(z.shape, OUTSIDE_LABEL, dtype=np.int8)
    counts: List[int] = []
    for s, (lo, hi) in zip(SLICES, b):
        sel = finite & (z >= lo) & (z < hi)
        label[sel] = s
        counts.append(int(sel.sum()))

    n_m1, n_0, n_p1 = counts
    pct: Tuple[float, float] = ((100.0 * n_m1 / n_0, 100.0 * n_p1 / n_0) if n_0 > 0
                                else (float("nan"), float("nan")))
    cov: List[float] = []
    for lo, hi in b:
        if zf.size == 0:
            cov.append(0.0)
        else:
            inside = min(hi, float(zf.max())) - max(lo, float(zf.min()))
            cov.append(float(min(1.0, max(0.0, inside / (hi - lo)))))
    frac = (n_m1 + n_0 + n_p1) / zf.size if zf.size else float("nan")
    return SliceSet(bounds_nm=(b[0], b[1], b[2]), label=label, n_locs=(n_m1, n_0, n_p1), pct_of_central=pct,
                    z_coverage=(cov[0], cov[1], cov[2]), frac_in_three=float(frac))


def slice_members(slices: SliceSet, source_index: Optional[NDArray[np.int64]] = None) -> Dict[int, NDArray[np.int64]]:
    """The members of each slice as source indices (or as positions when ``source_index`` is None)."""
    if source_index is None:
        ids = np.arange(slices.label.size, dtype=np.int64)
    else:
        ids = np.asarray(source_index, dtype=np.int64).ravel()
        if ids.size != slices.label.size:
            raise ValueError("source_index must have one entry per localization")
    return {s: ids[slices.label == s] for s in SLICES}
