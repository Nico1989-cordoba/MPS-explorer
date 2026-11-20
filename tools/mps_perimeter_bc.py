"""
Questions B (tortuosity vs density) and C (correlation between slices), per axon.

Two owners share this file: P5 writes the C group, P6 the B group (P6 merges after P5).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.mps_matching import SmoothPath
from tools.mps_perimeter_types import SliceProfiles, TestReadout

# =====================================================================================================
# GROUP B: tortuosity and density. Owner: P6.
# =====================================================================================================
#
# Question B of the perimeter density analysis: is the localization density of a perimeter segment related to
# how curved that segment is? Read per axon on the central slice (z0), on the common arc grid of
# ``tools.mps_perimeter_density`` (bin j is CENTRED on origin + j b, i.e. covers the arc [origin + (j - 1/2) b, origin + (j + 1/2) b) of the contour, b =
# L / n_bins, arcs measured along ``SmoothPath.cum_nm`` from the path's first row and taken modulo L).
#
# * Window: ``window_bins`` = 2 consecutive bins (200 nm nominal; exactly 2 b), step 1 bin, so there are as many
#   windows as bins and window i holds bins i, i + 1 (mod n_bins). Overlapping windows give the rotation null
#   resolution (an 8 um axon has 80 windows instead of 40; with 40 the smallest p would be 0.025).
# * Tortuosity of a window = chord between the curve points at its two ends / arc between them (<= 1, 1 =
#   straight). The arc is always 2 b by construction, also for the window that crosses the closure of the curve
#   (the arc is taken modulo L), so the tortuosity never depends on where the curve's table starts.
# * Density of a window = sum of the counts of its bins (Spearman is invariant to the scale, so counts and the
#   relative density counts / counts.max() give the same statistic). A window with a bin not measured is excluded.
#   Windows with few (or zero) localizations are NOT excluded: a low count is the signal, not a defect; the many
#   ties of a sparse profile only make the rank test conservative (checked in test_perimeter_b.py).
# * Statistic: Spearman = Pearson of the ranks (average ranks for ties), computed at every circular shift by
#   ``tools.mps_gaps.profile_correlation`` (the density ranks are rotated against the tortuosity ranks; masked bins
#   travel with the density). Two-sided p = (1 + #{|r_k| >= |r_0|}) / (1 + n_used) [lit: Phipson & Smyth 2010], no
#   direction being predicted.
# * Null ("randomized version"): circular rotation of the density profile with the curvature fixed. It keeps the
#   autocorrelation of both profiles (the window overlap included) and breaks only their alignment.
#
# What the null does NOT control (the contour-band bias): where the fitted contour cuts a corner, the contour is
# shorter than the membrane there, the localizations bunch up per unit of contour arc and the band loses a different
# share of them; that is a real tortuosity-count association that is not biology. Only simulation controls it
# (test_perimeter_b.py: noisy contour + density independent of curvature, at band half-widths 50/100/150 nm), plus
# the exported control ``tau_vs_offset`` (Spearman of tortuosity with the median radial offset per window).
#
# Resolution: for an arc of radius R, chord / arc = sin(w / 2R) / (w / 2R) ~= 1 - (w / R)^2 / 24 (next term
# (w / R)^4 / 1920) [derivation, verified numerically in the tests]: w = 200 nm and R = 600 nm give 0.9954. Ranking
# by tortuosity is ranking by the mean squared curvature of the window: the sign (convex / concave) is lost, which
# is why ``window_curvature`` exports the signed mean curvature per window as well.
#
# The imports of this group live here (not in the header) so that the C group's edits and these merge cleanly.
import math  # noqa: E402
from typing import Optional  # noqa: E402

from tools.mps_curvature import CURVATURE_RESOLVED_MIN_RADIUS_NM, curvature_of_closed_curve  # noqa: E402
from tools.mps_gaps import profile_correlation  # noqa: E402
from tools.mps_matching import _walk_polygon  # noqa: E402
from tools.mps_perimeter_types import BReadout  # noqa: E402

# Bins per window [director: 200 nm window on 100 nm bins]; step 1 bin [plan sec. 3.1: as many windows as bins].
DEFAULT_WINDOW_BINS: int = 2
# Fewest usable windows for the test [ad hoc: the shortest profile ``profile_correlation`` accepts, and its
# MIN_SHARED_BINS for masked profiles].
MIN_VALID_WINDOWS: int = 8
# |r_k| >= |r_0| is evaluated with this slack so that a null value equal to the observed one up to float rounding
# counts as "at least as extreme" (conservative) [derivation: correlations are O(1) sums of O(n) terms].
TIE_TOL: float = 1e-12


def _check_grid(n_bins: int, window_bins: int) -> Tuple[int, int]:
    n, k = int(n_bins), int(window_bins)
    if n < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if not 1 <= k <= n:
        raise ValueError(f"window_bins must be in [1, n_bins = {n}], got {window_bins}")
    return n, k


def _window_starts(length_nm: float, n_bins: int, origin_arc_nm: float) -> NDArray[np.float64]:
    """Arc of the lower edge of each bin (= the start of each window), modulo L. Bin j is CENTRED on origin + j b
    (the convention of ``tools.mps_perimeter_density.density_profile``), so its lower edge is origin + (j - 1/2) b."""
    b = float(length_nm) / n_bins
    return np.asarray(np.mod(float(origin_arc_nm) + b * (np.arange(n_bins, dtype=np.float64) - 0.5), length_nm),
                      dtype=np.float64)


def arc_bin_index(s_nm: NDArray[np.float64], *, length_nm: float, n_bins: int,
                  origin_arc_nm: float) -> NDArray[np.int64]:
    """Bin of each arc on the grid of ``tools.mps_perimeter_density``: floor((s - origin) / b + 1/2) mod n_bins,
    bin j centred on origin + j b. (P6 helper)"""
    b = float(length_nm) / int(n_bins)
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    return np.asarray(np.mod(np.floor((s - float(origin_arc_nm)) / b + 0.5).astype(np.int64), int(n_bins)),
                      dtype=np.int64)


def _point_at(path: SmoothPath, s_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Points of the path's dense table at arcs ``s_nm`` (taken modulo L), as ``point_on_path`` reads them."""
    s = np.mod(np.asarray(s_nm, dtype=np.float64), path.length_nm)
    return _walk_polygon(path.points_nm, path.edges_nm, path.cum_nm, s)


def tortuosity_windows(path: SmoothPath, *, n_bins: int, origin_arc_nm: float,
                       window_bins: int = 2) -> NDArray[np.float64]:
    """Chord / arc of each window (2 bins, step 1 bin, closed curve); <= 1, 1 = straight. (P6)

    Window i spans the arc [origin + (i - 1/2) b, origin + (i - 1/2 + window_bins) b] (mod L) with b = L / n_bins; its tortuosity
    is |P(end) - P(start)| / (window_bins b), the points interpolated on the path's dense table (the table of
    ``SmoothPath`` is 2 nm, so the polyline's chord / arc differs from the smooth curve's by ~ (2 nm / R)^2 / 24,
    < 1e-6 at R >= 400 nm). Raises ValueError for a degenerate path or grid."""
    n, k = _check_grid(n_bins, window_bins)
    length = float(path.length_nm)
    if not (math.isfinite(length) and length > 0.0):
        raise ValueError(f"tortuosity_windows: the path has no length ({path.length_nm})")
    arc = k * length / n
    starts = _window_starts(length, n, origin_arc_nm)
    p0 = _point_at(path, starts)
    p1 = _point_at(path, starts + arc)
    chord = np.hypot(p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1])
    return np.asarray(np.minimum(chord / arc, 1.0), dtype=np.float64)


def window_counts(counts: NDArray[np.int64], window_bins: int = 2) -> NDArray[np.int64]:
    """Sum of the counts of each window's bins (closed, step 1). (P6)

    ``out[i] = counts[i] + ... + counts[(i + window_bins - 1) % n]``: the same windows as ``tortuosity_windows``."""
    c = np.asarray(counts).reshape(-1)
    n, k = _check_grid(c.size, window_bins)
    out = np.zeros(n, dtype=np.int64)
    for j in range(k):
        out += np.roll(c, -j).astype(np.int64)
    return out


def window_measured(measured: NDArray[np.bool_], window_bins: int = 2) -> NDArray[np.bool_]:
    """A window is measured only when every one of its bins is (P6 helper, not in the contract)."""
    m = np.asarray(measured, dtype=bool).reshape(-1)
    n, k = _check_grid(m.size, window_bins)
    out = np.ones(n, dtype=bool)
    for j in range(k):
        out &= np.roll(m, -j)
    return out


def _average_ranks(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """1-based ranks with ties sharing their average rank (``scipy.stats.rankdata(method="average")``)."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts).astype(np.float64)
    return np.asarray((upper - (counts - 1) / 2.0)[inverse.reshape(-1)], dtype=np.float64)


def _nan_readout(n_null: int) -> TestReadout:
    nan = float("nan")
    return TestReadout(value=nan, null_mean=nan, null_sd=nan, z=nan, p=nan, n_null=int(n_null), sided="two")


def curvature_density_test(tau: NDArray[np.float64], window_density: NDArray[np.float64],
                           measured: NDArray[np.bool_], *, n_null: int = 1999, seed: int = 0) -> TestReadout:
    """Spearman(tortuosity, density) against the circular-rotation null; two-sided. (P6)

    ``tau``, ``window_density`` and ``measured`` are per window (``tortuosity_windows``, ``window_counts``,
    ``window_measured``). Windows with a non-finite tortuosity or density, or not measured, are left out of the
    ranks and of every shift. The null shifts are every non-zero shift when ``n_null >= n - 1`` (or ``n_null <=
    0``), otherwise ``n_null`` of them drawn without replacement with ``seed`` (the same draw as
    ``profile_correlation``'s one-sided p). With masked windows each profile is ranked over its OWN valid windows
    and the correlation at each shift is the Pearson correlation of those ranks over the windows both have there
    (the same statistic at every shift, so the null stays exchangeable); it equals Spearman exactly when nothing is
    masked and differs from the Spearman of the shared subset by ~1e-3 otherwise. ``value`` is r(0); ``z`` is r(0) in null standard deviations;
    ``n_null`` is the number of shifts the p was taken over. All NaN (p included) with fewer than
    ``MIN_VALID_WINDOWS`` usable windows or a constant variable: there is then no rank correlation to test."""
    t = np.asarray(tau, dtype=np.float64).reshape(-1)
    d = np.asarray(window_density, dtype=np.float64).reshape(-1)
    m = np.asarray(measured, dtype=bool).reshape(-1)
    n = t.size
    if d.size != n or m.size != n:
        raise ValueError(f"curvature_density_test: tau, density and measured differ in length "
                         f"({n}, {d.size}, {m.size})")
    ok_t = np.isfinite(t)
    ok_d = m & np.isfinite(d)
    if n < MIN_VALID_WINDOWS or int(np.count_nonzero(ok_t & ok_d)) < MIN_VALID_WINDOWS:
        return _nan_readout(0)
    rt = np.zeros(n, dtype=np.float64)
    rd = np.zeros(n, dtype=np.float64)
    rt[ok_t] = _average_ranks(t[ok_t])
    rd[ok_d] = _average_ranks(d[ok_d])
    if ok_t.all() and ok_d.all():
        pc = profile_correlation(rt, rd, n_rotations=int(n_null), seed=int(seed))
    else:
        pc = profile_correlation(rt, rd, n_rotations=int(n_null), seed=int(seed), measured_a=ok_t, measured_b=ok_d)
    corr = np.asarray(pc.correlation, dtype=np.float64)
    r0 = float(corr[0]) if corr.size else float("nan")
    if not math.isfinite(r0):
        return _nan_readout(0)
    n_rot = int(n_null)
    if n_rot <= 0 or n_rot >= n - 1:
        shifts: NDArray[np.int_] = np.arange(1, n)
    else:   # the draw of tools.mps_gaps._with_null
        shifts = np.random.default_rng(int(seed)).choice(np.arange(1, n), size=n_rot, replace=False)
    null = corr[shifts]
    null = null[np.isfinite(null)]
    if null.size == 0:
        return _nan_readout(0)
    b = int(np.count_nonzero(np.abs(null) >= abs(r0) - TIE_TOL))
    p = float(1 + b) / float(1 + null.size)
    mean = float(null.mean())
    sd = float(null.std())
    z = (r0 - mean) / sd if sd > 0 else float("nan")
    return TestReadout(value=r0, null_mean=mean, null_sd=sd, z=float(z), p=p, n_null=int(null.size), sided="two")


def chord_arc_circle(window_nm: float, radius_nm: float) -> float:
    """Exact tortuosity of an arc of length ``window_nm`` on a circle of radius ``radius_nm``: sin(x) / x,
    x = w / 2R [derivation]. (P6 helper)"""
    x = float(window_nm) / (2.0 * float(radius_nm))
    return 1.0 if x == 0.0 else math.sin(x) / x


def chord_arc_small_angle(window_nm: float, radius_nm: float) -> float:
    """The plan's approximation 1 - (w / R)^2 / 24; its error is ~ (w / R)^4 / 1920 [derivation]. (P6 helper)"""
    return 1.0 - (float(window_nm) / float(radius_nm)) ** 2 / 24.0


def window_curvature(path: SmoothPath, *, n_bins: int, origin_arc_nm: float,
                     window_bins: int = 2) -> NDArray[np.float64]:
    """Signed mean curvature (1/nm, + convex) of each window, from ``tools.mps_curvature.curvature_of_closed_curve``
    on the path's table (5 nm resampling, 10 nm central differences). It keeps the sign the tortuosity loses; for
    the long table of the export. (P6 helper)"""
    n, k = _check_grid(n_bins, window_bins)
    _, s, kappa, length = curvature_of_closed_curve(path.points_nm)
    arc = k * length / n
    starts = _window_starts(length, n, origin_arc_nm)
    rel = np.mod(s[None, :] - starts[:, None], length)          # (n, M) arc of each sample from each window start
    inside = rel < arc
    num = (inside * kappa[None, :]).sum(axis=1)
    den = inside.sum(axis=1)
    return np.asarray(np.where(den > 0, num / np.maximum(den, 1), np.nan), dtype=np.float64)


def window_radial_median(s_nm: NDArray[np.float64], r_nm: NDArray[np.float64], *, length_nm: float, n_bins: int,
                         origin_arc_nm: float, window_bins: int = 2) -> NDArray[np.float64]:
    """Median signed radial offset of the localizations (already inside the band) in each window; NaN in an empty
    window. ``s_nm`` is the arc of the projection on the contour (same origin as ``SmoothPath.cum_nm``). Input of
    the contour-bias control ``tau_vs_offset``. (P6 helper)"""
    n, k = _check_grid(n_bins, window_bins)
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    r = np.asarray(r_nm, dtype=np.float64).reshape(-1)
    if s.size != r.size:
        raise ValueError("window_radial_median: s and r differ in length")
    j = arc_bin_index(s, length_nm=float(length_nm), n_bins=n, origin_arc_nm=origin_arc_nm)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        sel = np.mod(j - i, n) < k
        if np.any(sel):
            out[i] = float(np.median(r[sel]))
    return out


def _spearman_value(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 4:
        return float("nan")
    ra = _average_ranks(a[ok])
    rb = _average_ranks(b[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def axon_b_readout(path: SmoothPath, counts: NDArray[np.int64], measured: NDArray[np.bool_], *,
                   origin_arc_nm: float, window_bins: int = 2,
                   radial_median_nm: Optional[NDArray[np.float64]] = None,
                   n_null: int = 1999, seed: int = 0) -> BReadout:
    """THE entry point of question B for one axon (the orchestration calls this; public, not in the P0 contract).

    ``counts`` and ``measured`` are the central slice's ``DensityProfile.counts`` / ``.measured`` and
    ``origin_arc_nm`` its ``origin_arc_nm`` (bin j centred on origin + j b); ``path`` is the contour the profile was
    projected on. Returns the BReadout: the test, ``tau_min`` (the smallest
    window tortuosity of the axon, i.e. its dynamic range below 1), ``resolved`` (R = L / 2 pi >= 600 nm,
    ``tools.mps_curvature.CURVATURE_RESOLVED_MIN_RADIUS_NM``) and ``tau_vs_offset`` (Spearman of the tortuosity with
    ``radial_median_nm`` per window, from ``window_radial_median``; NaN when not given)."""
    c = np.asarray(counts).reshape(-1)
    n = c.size
    tau = tortuosity_windows(path, n_bins=n, origin_arc_nm=origin_arc_nm, window_bins=window_bins)
    dens = window_counts(c, window_bins).astype(np.float64)
    wm = window_measured(measured, window_bins)
    test = curvature_density_test(tau, dens, wm, n_null=n_null, seed=seed)
    offset = float("nan")
    if radial_median_nm is not None:
        rm = np.asarray(radial_median_nm, dtype=np.float64).reshape(-1)
        if rm.size != n:
            raise ValueError("axon_b_readout: radial_median_nm must have one value per window")
        offset = _spearman_value(np.where(wm, tau, np.nan), rm)
    resolved = float(path.length_nm) / (2.0 * math.pi) >= CURVATURE_RESOLVED_MIN_RADIUS_NM
    return BReadout(test=test, tau_min=float(np.min(tau)), resolved=bool(resolved), tau_vs_offset=offset)


# =====================================================================================================
# GROUP C: correlation between slices. Owner: P5.
# =====================================================================================================


import math  # noqa: E402  (kept inside group C so the two owners' import blocks do not collide)

from tools.mps_gaps import profile_correlation  # noqa: E402

# Axial localization precision of the pilot data. [datos reales, piloto]: lpz ~ 47 nm (the plan's figure; the
# effective axial width measured in the archive is larger, sigma_z ~ 77 nm, so the leak is then bigger).
LEAK_DEFAULT_LPZ_NM: float = 47.0
# Tolerance used to call a neighbour correlation "leakage only". [ad hoc]: about one standard error of a
# Pearson r over ~100 bins (SE ~ 1/sqrt(n) = 0.1); to be swept in simulation.
LEAK_TOLERANCE: float = 0.10
# Significance level for the reading. [lit] conventional alpha.
ALPHA: float = 0.05

_PAIRS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (-1, 1))


def _nan_c_readout() -> TestReadout:
    nan = float("nan")
    return TestReadout(value=nan, null_mean=nan, null_sd=nan, z=nan, p=nan, n_null=0, sided="one")


def slice_correlations(profiles: SliceProfiles, *, n_null: int = 1999,
                       seed: int = 0) -> Dict[Tuple[int, int], TestReadout]:
    """
    r(0) and one-sided p of the pairs (-1, 0), (0, 1), (-1, 1) in the common frame.

    The three profiles must share grid and origin (frame "z0_max": origin at the maximum of the CENTRAL slice).
    Aligning each slice on its own maximum would force a peak at bin 0 in all three and destroy the very
    correlation being tested, so ``frame != "z0_max"`` is refused. Each pair goes through
    ``tools.mps_gaps.profile_correlation`` unchanged: Pearson r with no rotation, one-sided p against the
    circular-rotation null (Phipson & Smyth 2010 counting; every shift when ``n_bins - 1 <= n_null``), bins not
    measured in either profile masked out. ``z`` is r(0) in standard deviations of the null correlations.
    The same ``seed`` is used for the three pairs. Pearson r does not depend on the scale of a profile, so
    counts and counts / max give the same result.

    Neighbouring pairs carry axial leakage (see ``leak_expected_r``); the pair (-1, 1) is the main reading.
    A pair that cannot be correlated (empty or constant profile, too few common bins) returns NaN fields.
    """
    if profiles.frame != "z0_max":
        raise ValueError(
            f"Slices must be correlated in the common frame 'z0_max', got {profiles.frame!r}: aligning each "
            "slice on its own maximum destroys the correlation.")
    for k in (-1, 0, 1):
        if k not in profiles.by_slice:
            raise ValueError(f"Missing slice {k}: slice_correlations needs the slices -1, 0 and +1.")
    ref = profiles.by_slice[0]
    for k in (-1, 1):
        q = profiles.by_slice[k]
        if (q.n_bins != ref.n_bins or q.counts.size != ref.counts.size
                or abs(q.bin_nm_effective - ref.bin_nm_effective) > 1e-9
                or abs(q.origin_arc_nm - ref.origin_arc_nm) > 1e-9):
            raise ValueError(f"Slice {k} does not share grid and origin with the central slice.")

    out: Dict[Tuple[int, int], TestReadout] = {}
    for i, j in _PAIRS:
        pi, pj = profiles.by_slice[i], profiles.by_slice[j]
        pc = profile_correlation(np.asarray(pi.counts, dtype=np.float64),
                                 np.asarray(pj.counts, dtype=np.float64),
                                 n_rotations=n_null, seed=seed,
                                 measured_a=np.asarray(pi.measured, dtype=bool),
                                 measured_b=np.asarray(pj.measured, dtype=bool))
        r0 = float(pc.r_at_zero)
        if not np.isfinite(r0):
            out[(i, j)] = _nan_c_readout()
            continue
        null = np.asarray(pc.correlation, dtype=np.float64)[1:]
        null = null[np.isfinite(null)]
        mean = float(null.mean()) if null.size else float("nan")
        sd = float(null.std()) if null.size else float("nan")
        z = (r0 - mean) / sd if (np.isfinite(sd) and sd > 0) else float("nan")
        out[(i, j)] = TestReadout(value=r0, null_mean=mean, null_sd=sd, z=float(z),
                                  p=float(pc.p_rotation), n_null=int(pc.n_rotations), sided="one")
    return out


def _leak_fractions(lpz_nm: float, thickness_nm: float, guard_nm: float) -> Tuple[float, float]:
    """(a, e): fraction of a slice's molecules observed in its own core, and in ONE neighbour's core."""
    n_u = 1201
    u = [thickness_nm * i / (n_u - 1) for i in range(n_u)]       # true depth in the slice, uniform
    root2 = math.sqrt(2.0)

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / (lpz_nm * root2)))

    def mass(lo: float, hi: float) -> float:
        p = [cdf(hi - v) - cdf(lo - v) for v in u]
        h = thickness_nm / (n_u - 1)
        return float(h * (sum(p) - 0.5 * (p[0] + p[-1])) / thickness_nm)   # trapezoid, divided by thickness

    a = mass(guard_nm, thickness_nm - guard_nm)
    e = mass(thickness_nm + guard_nm, 2.0 * thickness_nm - guard_nm)
    return a, e


def leak_expected_r(lpz_nm: float, thickness_nm: float, guard_nm: float) -> float:
    """
    Correlation between NEIGHBOURING slices expected from axial leakage alone [cuenta].

    Assumptions, all explicit:
      1. [cuenta] Every molecule sits at a depth uniform inside its slice of ``thickness_nm`` (170 nm).
      2. [cuenta] Its observed z carries a Gaussian error of sd ``lpz_nm`` (47 nm [datos reales, piloto]).
      3. [cuenta] The true angular patterns of different slices are independent, with equal mean and variance.
      4. [cuenta] Counts are noise-free: Poisson noise would lower the observed r (attenuation), so this is the
         largest r that leakage alone produces.
      5. [cuenta] Only the nearest neighbour leaks (the next one needs an error of 3.6 sd; ~1e-4).
      6. ``guard_nm`` keeps only the core [guard, thickness - guard] of each slice (0 = contiguous slices).
    With a = fraction of a slice's molecules observed in its own core and e = fraction observed in one
    neighbour's core, an observed profile is a*P_k + e*(P_k-1 + P_k+1), so for independent unit-variance P:
    r(k, k+1) = 2 a e / (a^2 + 2 e^2). For 170 nm and 47 nm: a = 0.78, e = 0.11, r = 0.27.
    The pair (-1, +1) shares only the central pattern: r = e^2 / (a^2 + 2 e^2) ~ 0.02 (not returned here).
    """
    if thickness_nm <= 0:
        raise ValueError("thickness_nm must be positive.")
    if guard_nm < 0 or 2.0 * guard_nm >= thickness_nm:
        raise ValueError("guard_nm must be in [0, thickness_nm / 2).")
    if lpz_nm < 0:
        raise ValueError("lpz_nm must be >= 0.")
    if lpz_nm == 0:
        return 0.0
    a, e = _leak_fractions(float(lpz_nm), float(thickness_nm), float(guard_nm))
    den = a * a + 2.0 * e * e
    return float(2.0 * a * e / den) if den > 0 else 0.0


# ----------------------------------------------------------------------------------------------------
# When the three slices "coincide well" for question C (decision P-R3-1, calibrated 2026-11-17)
# ----------------------------------------------------------------------------------------------------
# Largest RMS misfit (nm) of the BETTER-fitting lateral slice for which C(-1, +1) keeps its nominal level.
# [own sim; Bradley 1978 robustness band 7.5 %]: see ``slices_coincide_for_c``.
C_SLICE_MATCH_MAX_NM: float = 32.0


def slices_coincide_for_c(rms_minus_nm: Optional[float], rms_plus_nm: Optional[float],
                          max_nm: float = C_SLICE_MATCH_MAX_NM) -> bool:
    """
    Whether the rings of slices -1 and +1 coincide well enough with the central one for the main reading of C.

    Rule: ``min(rms_minus_nm, rms_plus_nm) <= max_nm`` (32 nm). When either misfit could not be measured (None or
    NaN) the slices do NOT coincide ("not measurable" is never "passes"; review R3, M6).

    Why the smaller of the two: a lateral shift takes localizations out of the +-100 nm band at the same angles in a
    shifted slice, so a spurious r(-1, +1) needs BOTH laterals shifted; to first order it grows with the product of
    the two losses and the smaller one rules. With only slice -1 shifted (up to 90 nm RMS) the false-positive rate
    of C(-1, +1) never left 0.075 in 2100 simulated axons; a change of radius alone (up to 120 nm) neither. The
    lateral part alone is not enough: radius change combined with a lateral shift does inflate C.

    Where 32 nm comes from [own sim; Bradley 1978 robustness band 7.5 %]: about 27 500 simulated axons with no
    registry (independent patterns per slice), six geometries of misfit (tilt, same side, cone, cone + tilt, one
    slice, half), lateral densities 8 / 20 / 50 localizations per bin, radial sd 40 and 73 nm, the misfit MEASURED
    by ``mps_auto_contour``'s slice misfit. For every stratum the local false-positive rate of C(-1, +1) was fitted
    as an isotonic function of the statistic; the threshold is the largest misfit where it stays within Bradley's
    liberal robustness band (at most 1.5 alpha = 0.075; Bradley, J. V. 1978, Br. J. Math. Stat. Psychol. 31:
    144-152), as the 5th percentile of 2000 bootstrap crossings in the worst stratum (both laterals shifted to the
    same side, 50 localizations per bin: point 33.6 nm, 5th percentile 32.0 nm). Among the axons that pass, the
    false-positive rate was 0.011-0.064 in 29 of 30 cells. Report: CRITERIO_RODAJAS_C.md of the validation (outside
    the repository); simulation ``crit_sim.py``.

    Limits: one cloud model (Gaussian radial + background + interior) and lateral densities up to 50 localizations
    per bin. The threshold FALLS with density (63 -> 56 -> 34 nm from 8 to 50 per bin in the worst geometry): with
    denser data it must be recalibrated (re-run ``crit_sim.py`` with the new density).
    """
    vals = []
    for v in (rms_minus_nm, rms_plus_nm):
        try:
            x = float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            x = float("nan")
        if not math.isfinite(x):
            return False
        vals.append(x)
    return bool(min(vals) <= float(max_nm))


# ----------------------------------------------------------------------------------------------------
# Axial leakage with the axon's own slice fractions (the code above assumes three equal slices)
# ----------------------------------------------------------------------------------------------------
# The axial widths (sd of the error in z) the expected leak is reported for. Decided 2026-11-17 by the user after
# measuring a "NeNA in z" on the pilot (85 729 pairs of the same molecule in consecutive frames; validation report
# NENA_Z.md) [real data, pilot]:
#   47 nm  MAIN figure: the measured equivalent sd when the slow error component (~27 nm) is real error (median
#          47.0 nm across axons); it equals the lpz Picasso reports (median of the pilot).
#   36-51  its UNCERTAINTY: 36 nm = frame-to-frame error only (slow component read as contamination), 51 nm = the
#          upper end across axons with the slow component.
#   77 nm  UPPER BOUND only: the sd of the rings' mixture components in z (1.76 x lpz), which also holds the
#          physical thickness of the ring (~61 nm implied); for leakage only the localization error counts.
# These are the pilot's values; for new data sigma_z should be measured per file (not yet in the program).
LEAK_SIGMA_Z_MAIN_NM: float = 47.0
LEAK_SIGMA_Z_RANGE_NM: Tuple[float, float] = (36.0, 51.0)
LEAK_SIGMA_Z_UPPER_NM: float = 77.0
# Widths of the v20 leak columns (47 = main, 77 = upper bound) and of the v21 ones (the range 36-51 nm).
LEAK_SIGMA_Z_NM: Tuple[float, ...] = (LEAK_SIGMA_Z_MAIN_NM, LEAK_SIGMA_Z_UPPER_NM)
LEAK_SIGMA_Z_V21_NM: Tuple[float, ...] = LEAK_SIGMA_Z_RANGE_NM
LEAK_SIGMA_Z_ALL_NM: Tuple[float, ...] = LEAK_SIGMA_Z_NM + LEAK_SIGMA_Z_V21_NM


def _leak_mass(sigma_z_nm: float, thickness_nm: float, lo: float, hi: float) -> float:
    """Fraction of molecules at a depth uniform in [0, T] whose observed z (Gaussian error sd sigma) falls in
    [lo, hi]; trapezoid over 1201 depths, as ``_leak_fractions``."""
    n_u = 1201
    root2 = math.sqrt(2.0)
    s = float(sigma_z_nm)
    T = float(thickness_nm)

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / (s * root2)))

    p = []
    for i in range(n_u):
        v = T * i / (n_u - 1)
        p.append(cdf(hi - v) - cdf(lo - v))
    return float((sum(p) - 0.5 * (p[0] + p[-1])) / (n_u - 1))


def leak_matrix_uniform(sigma_z_nm: float, thickness_nm: float, guard_nm: float) -> np.ndarray:
    """
    M[k, j] (k, j = 0, 1, 2 for slices -1, 0, +1): probability that a localization of a molecule of slice k is
    observed in the core of analysis slice j [cuenta]. Molecules sit at a depth uniform inside their own slice of
    ``thickness_nm`` and nowhere else; the observed z carries a Gaussian error of sd ``sigma_z_nm``; the cores are
    [guard, T - guard] of each contiguous slice. Unlike ``leak_expected_r`` the leak to the slice after next (-1 to
    +1) is kept. A row does not sum to 1: the rest is observed outside the cores.
    """
    T = float(thickness_nm)
    g = float(guard_nm)
    if T <= 0:
        raise ValueError("thickness_nm must be positive.")
    if g < 0 or 2.0 * g >= T:
        raise ValueError("guard_nm must be in [0, thickness_nm / 2).")
    if sigma_z_nm < 0:
        raise ValueError("sigma_z_nm must be >= 0.")
    M = np.zeros((3, 3), dtype=np.float64)
    for k in range(3):
        for j in range(3):
            d = j - k
            if sigma_z_nm == 0:
                M[k, j] = (T - 2.0 * g) / T if d == 0 else 0.0
            else:
                M[k, j] = _leak_mass(float(sigma_z_nm), T, d * T + g, (d + 1) * T - g)
    return M


def leak_expected_r_pairs(n_locs: Sequence[float], sigma_z_nm: float, thickness_nm: float,
                          guard_nm: float) -> Dict[Tuple[int, int], float]:
    """
    Correlation expected from axial leakage alone for the three pairs, with the axon's OWN slice fractions [cuenta].

    ``n_locs``: localizations observed in slices -1, 0, +1, taken as the molecule masses w of the three slices
    (normalised). Inverting the leak instead (observed counts = M^T w) was tried and rejected: under this
    uniform-depth model it attributes almost every localization of a sparse lateral to the central slice's leak
    (one pilot axon, lateral at 13 % of the central: neighbour leak 1.00 at 47 nm), far from the observed
    neighbour correlations (~0.3); real depth is peaked in the central slice, which leaks less than uniform depth.
    Taking the observed fractions as masses can UNDERESTIMATE the leak when a lateral is mostly leak (its true mass
    is smaller than observed). With independent true patterns of equal relative variance, the observed profile of
    slice j is sum_k w_k M[k, j] P_k and
    r(i, j) = (A^T A)[i, j] / sqrt((A^T A)[i, i] (A^T A)[j, j]) with A[k, j] = w_k M[k, j] (the model of
    ``mps_perimeter_sim.expected_slice_r`` with rho = 0). Counts are taken noise-free (Poisson noise lowers the
    observed r), so each value is an UPPER bound of what leakage alone gives. With laterals below the central
    slice, what a lateral receives from the central slice weighs more in its profile than with equal slices: for
    laterals at 45 % of the central and sigma_z 47 nm the pair (-1, +1) gets about 0.09, against 0.02 with equal
    slices (CRITERIO_RODAJAS_C.md, sec. 6). A pair with an empty slice is NaN.
    """
    nan = float("nan")
    out: Dict[Tuple[int, int], float] = {pair: nan for pair in _PAIRS}
    try:
        n = np.asarray([float(v) for v in n_locs], dtype=np.float64)
    except (TypeError, ValueError):
        return out
    if n.size != 3 or not np.all(np.isfinite(n)) or np.any(n < 0) or n.sum() <= 0:
        return out
    if not (math.isfinite(sigma_z_nm) and math.isfinite(thickness_nm) and math.isfinite(guard_nm)):
        return out
    M = leak_matrix_uniform(float(sigma_z_nm), float(thickness_nm), float(guard_nm))
    w = n / n.sum()
    A = M * w[:, None]
    cov = A.T @ A
    idx = {-1: 0, 0: 1, 1: 2}
    for (i, j) in _PAIRS:
        a, b = idx[i], idx[j]
        if n[a] <= 0 or n[b] <= 0 or cov[a, a] <= 0 or cov[b, b] <= 0:
            continue
        out[(i, j)] = float(cov[a, b] / math.sqrt(cov[a, a] * cov[b, b]))
    return out


# Reading of r(-1, +1) against the expected leak of the pair (``m1_p1_vs_leak``; v21 vocabulary, 2026-11-17).
VS_LEAK_ABOVE_UPPER: str = "above the leak (sigma_z 47 nm, range 36-51 nm) and above its upper bound (77 nm)"
VS_LEAK_ABOVE: str = "above the leak (sigma_z 47 nm, range 36-51 nm); within the upper bound (77 nm)"
VS_LEAK_WITHIN_RANGE: str = "above the leak at sigma_z 47 nm, within its range (not above 51 nm)"
VS_LEAK_LIMIT: str = "at the limit: above the leak at sigma_z 36 nm, not at 47 nm"
VS_LEAK_NOT_ABOVE: str = "not above the leak (not even at sigma_z 36 nm)"
VS_LEAK_VALUES: Tuple[str, ...] = (VS_LEAK_ABOVE_UPPER, VS_LEAK_ABOVE, VS_LEAK_WITHIN_RANGE, VS_LEAK_LIMIT,
                                   VS_LEAK_NOT_ABOVE)


def m1_p1_vs_leak(r_m1_p1: Optional[float], leak_by_width: Dict[float, float]) -> Optional[str]:
    """r(-1, +1) observed against the leak expected for the pair (``leak_expected_r_pairs``) at the main width
    (47 nm), its measured range (36-51 nm) and the upper bound (77 nm): one of ``VS_LEAK_VALUES``, from "above the
    leak ... and above its upper bound" down to "not above the leak (not even at sigma_z 36 nm)"; "at the limit"
    = above the 36 nm leak but not the 47 nm one. None when r or one of the four leaks is missing. The leak is a
    noise-free upper bound, so "above" is the conservative side. It says nothing about significance
    (``pd_c_p_m1_p1``)."""
    try:
        r = float(r_m1_p1) if r_m1_p1 is not None else float("nan")
    except (TypeError, ValueError):
        return None
    lk = [float(leak_by_width.get(sz, float("nan"))) for sz in
          (LEAK_SIGMA_Z_RANGE_NM[0], LEAK_SIGMA_Z_MAIN_NM, LEAK_SIGMA_Z_RANGE_NM[1], LEAK_SIGMA_Z_UPPER_NM)]
    if not (math.isfinite(r) and all(math.isfinite(v) for v in lk)):
        return None
    l36, l47, l51, l77 = lk
    if r > max(l36, l47, l51, l77):
        return VS_LEAK_ABOVE_UPPER
    if r > max(l36, l47, l51):
        return VS_LEAK_ABOVE
    if r > max(l36, l47):
        return VS_LEAK_WITHIN_RANGE
    if r > l36:
        return VS_LEAK_LIMIT
    return VS_LEAK_NOT_ABOVE


def slice_reading(c: Dict[Tuple[int, int], TestReadout], expected_leak_r: float, *,
                  alpha: float = ALPHA, tol: float = LEAK_TOLERANCE) -> str:
    """
    Plain reading of the three correlations, with leakage flagged as such, against ONE leak of three equal slices.
    Since 2026-11-17 the program's per-axon reading is ``slice_reading_real_fractions`` (each pair against the leak
    of the axon's own slice fractions); this function is kept for the validation scripts and comparisons.

    * "registry between slices": r(-1, +1) positive and significant (the pair leakage cannot reach);
    * "leakage only": neighbour pairs significant with r within ``tol`` of ``expected_leak_r`` (or below) and
      r(-1, +1) not significant: the neighbour correlation is the axial leak, not registry;
    * "neighbours above leakage": a significant neighbour r exceeds the leak expectation by more than ``tol``
      while r(-1, +1) is not significant: suggestive, not conclusive;
    * "no correlation between slices"; "not computable" when a pair is NaN.
    """
    skip = c.get((-1, 1))
    n1, n2 = c.get((-1, 0)), c.get((0, 1))
    if skip is None or n1 is None or n2 is None:
        return "not computable"
    if not all(math.isfinite(x.value) and math.isfinite(x.p) for x in (skip, n1, n2)):
        return "not computable"
    if skip.p < alpha and skip.value > 0:
        return "registry between slices"
    sig = [x for x in (n1, n2) if x.p < alpha]
    if not sig:
        return "no correlation between slices"
    if any(x.value > expected_leak_r + tol for x in sig):
        return "neighbours above leakage"
    return "leakage only"


def slice_reading_pairs(c: Dict[Tuple[int, int], TestReadout], leak_by_pair: Dict[Tuple[int, int], float], *,
                        alpha: float = ALPHA, tol: float = LEAK_TOLERANCE) -> str:
    """
    ``slice_reading`` with each neighbouring pair compared against ITS OWN expected leak (``leak_expected_r_pairs``,
    the axon's slice fractions), instead of one leak of three equal slices. Same labels and the same rules:
    "registry between slices" (r(-1, +1) positive and significant), "leakage only" (every significant neighbour r
    within ``tol`` of its pair's leak or below), "neighbours above leakage" (a significant neighbour r above its
    pair's leak by more than ``tol``), "no correlation between slices", "not computable" (a pair, or the leak of a
    significant neighbour pair, is NaN).
    """
    skip = c.get((-1, 1))
    n1, n2 = c.get((-1, 0)), c.get((0, 1))
    if skip is None or n1 is None or n2 is None:
        return "not computable"
    if not all(math.isfinite(x.value) and math.isfinite(x.p) for x in (skip, n1, n2)):
        return "not computable"
    if skip.p < alpha and skip.value > 0:
        return "registry between slices"
    sig = [(pair, x) for pair, x in (((-1, 0), n1), ((0, 1), n2)) if x.p < alpha]
    if not sig:
        return "no correlation between slices"
    leaks = [float(leak_by_pair.get(pair, float("nan"))) for pair, _x in sig]
    if not all(math.isfinite(v) for v in leaks):
        return "not computable"
    if any(x.value > lk + tol for (_pair, x), lk in zip(sig, leaks)):
        return "neighbours above leakage"
    return "leakage only"


def slice_reading_real_fractions(c: Dict[Tuple[int, int], TestReadout],
                                 leaks_by_width: Dict[float, Dict[Tuple[int, int], float]], *,
                                 alpha: float = ALPHA, tol: float = LEAK_TOLERANCE) -> str:
    """
    The per-axon reading of C (``pd_c_reading``, user's decisions 2026-11-17): the neighbouring pairs are compared
    with the leak expected from the axon's OWN slice fractions (``leak_expected_r_pairs``), not with the leak of
    equal slices (``leak_expected_r``, 0.27 at 47 nm), at the MAIN axial width (47 nm, measured by the NeNA in z).
    The reading is repeated at the two ends of the measured range (36 and 51 nm); when an end gives another reading
    the label is the 47 nm one followed by
    " (within the measured range of sigma_z 36-51 nm: <reading> at 36 nm[; <reading> at 51 nm])". The 77 nm upper
    bound (it holds the rings' physical thickness) does not enter the reading. The leak is noise-free (an upper bound)
    and assumes the observed fractions are the true masses, which can underestimate it when a lateral slice is mostly
    leak (``leak_expected_r_pairs``).
    """
    nan_pairs: Dict[Tuple[int, int], float] = {pair: float("nan") for pair in _PAIRS}
    main = slice_reading_pairs(c, leaks_by_width.get(LEAK_SIGMA_Z_MAIN_NM, nan_pairs), alpha=alpha, tol=tol)
    other = []
    for sz in LEAK_SIGMA_Z_RANGE_NM:
        rd = slice_reading_pairs(c, leaks_by_width.get(sz, nan_pairs), alpha=alpha, tol=tol)
        if rd != main:
            other.append(f"{rd} at {sz:g} nm")
    if not other:
        return main
    lo, hi = LEAK_SIGMA_Z_RANGE_NM
    return f"{main} (within the measured range of sigma_z {lo:g}-{hi:g} nm: {'; '.join(other)})"
