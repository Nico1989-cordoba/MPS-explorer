"""Question B of the perimeter density analysis (P6): tortuosity of 200 nm windows against their density.

Synthetic data only, built by hand here (the shared simulator P1 does not exist yet): closed curves tabulated
densely as ``SmoothPath``, localizations placed along a true curve with a density that is either independent of the
curvature or proportional to 1 + beta kappa R, and -- for the hard controls -- a "fitted" contour that is NOT the
true curve (low-pass filtered, so it cuts the corners, plus a random low-frequency radial error), onto which the
localizations are projected and kept inside a symmetric band, as the real pipeline does.

Run one file per process: ``python -m pytest test_perimeter_b.py``. Every test takes well under 60 s.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import mps_perimeter_bc as bc  # noqa: E402
from tools.mps_matching import SmoothPath, smooth_closed_path  # noqa: E402

# ----------------------------------------------------------------------------------------------------------------
# Synthetic geometry and localizations (numbers tagged with their origin)
# ----------------------------------------------------------------------------------------------------------------
TABLE_STEP_NM = 2.0          # [derivation] = tools.mps_matching.PATH_STEP_NM, the table step of a real SmoothPath
BIN_NM = 100.0               # [director]
RADIAL_SD_NM = 25.0          # [own sim, plan sec. 2.4: radial sd 23-27 nm in simulation]
NOISY_CONTOUR_HARMONICS = 6  # [ad hoc] keeps harmonics <= 6 of r(theta): on 1-1.4 um calibres that is a ~1 um
#                              scale, coarser than the 400 nm knots of the real fit, i.e. a pessimistic corner cut
NOISY_CONTOUR_SD_NM = 15.0   # [ad hoc, plan sec. 2.3: P3's target mean error <= 25 nm] random radial error
ALPHA = 0.05


def _path_from_points(points):
    """A ``SmoothPath`` whose dense table IS the given closed polyline (exact geometry for analytic tests)."""
    p = np.asarray(points, dtype=np.float64)
    nxt = np.roll(p, -1, axis=0)
    edges = np.hypot(nxt[:, 0] - p[:, 0], nxt[:, 1] - p[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    return SmoothPath(points_nm=p, vertex_row=np.arange(p.shape[0], dtype=np.intp), edges_nm=edges,
                      cum_nm=cum, length_nm=float(cum[-1]))


def _polar_points(radius_fn, n):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = radius_fn(th)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def _circle(radius, step=TABLE_STEP_NM):
    n = int(math.ceil(2.0 * math.pi * radius / step))
    return _path_from_points(_polar_points(lambda th: np.full_like(th, radius), n))


def _ellipse_r(a, b):
    return lambda th: a * b / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)


def _wavy_r(r0):
    # harmonics 2, 3, 5 with amplitudes that keep the curve star-shaped and its concavities moderate
    return lambda th: r0 * (1.0 + 0.10 * np.cos(2 * th) + 0.05 * np.cos(3 * th + 0.7) + 0.02 * np.cos(5 * th + 1.9))


def _resample_closed(points, step=TABLE_STEP_NM):
    """Uniform arc resampling of a closed polyline (so the table step is ~``step`` everywhere)."""
    p = np.asarray(points, dtype=np.float64)
    closed = np.vstack([p, p[:1]])
    seg = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    m = int(math.ceil(cum[-1] / step))
    s = np.arange(m) * (cum[-1] / m)
    return np.column_stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])])


def _true_curve(radius_fn):
    return _path_from_points(_resample_closed(_polar_points(radius_fn, 20000)))


def _tangent_normal(path):
    p = path.points_nm
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    sign = 1.0 if area > 0 else -1.0
    normal_out = sign * np.column_stack([t[:, 1], -t[:, 0]])
    return t, normal_out


def _curvature_at_rows(path):
    _, s, kappa, _ = bc.curvature_of_closed_curve(path.points_nm)
    return np.interp(path.cum_nm[:-1], s, kappa, period=path.length_nm)


def _simulate(path, n_locs, rng, *, beta=0.0):
    """Localizations along ``path`` (true curve): arc density proportional to 1 + beta kappa R (R = L / 2 pi),
    radial scatter N(0, RADIAL_SD_NM). Returns (xy (n, 2), true arc (n,))."""
    rows = path.points_nm.shape[0]
    w = np.ones(rows)
    if beta != 0.0:
        w = np.clip(1.0 + beta * _curvature_at_rows(path) * path.length_nm / (2.0 * math.pi), 0.0, None)
    w = w * path.edges_nm
    idx = rng.choice(rows, size=n_locs, p=w / w.sum())
    frac = rng.random(n_locs)
    s = path.cum_nm[idx] + frac * path.edges_nm[idx]
    xy = path.points_nm[idx] + frac[:, None] * (path.points_nm[(idx + 1) % rows] - path.points_nm[idx])
    _, normal = _tangent_normal(path)
    xy = xy + rng.normal(0.0, RADIAL_SD_NM, n_locs)[:, None] * normal[idx]
    return xy, s


def _noisy_contour(radius_fn, rng):
    """A fitted-contour stand-in: r(theta) low-passed to NOISY_CONTOUR_HARMONICS (cuts the corners) plus a random
    radial error of harmonics 3..8 with sd NOISY_CONTOUR_SD_NM."""
    m = 4096
    th = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    spec = np.fft.rfft(radius_fn(th))
    spec[NOISY_CONTOUR_HARMONICS + 1:] = 0.0
    r = np.fft.irfft(spec, n=m)
    err = np.zeros(m)
    for k in range(3, 9):
        err += rng.normal() * np.cos(k * th) + rng.normal() * np.sin(k * th)
    err *= NOISY_CONTOUR_SD_NM / err.std()
    pts = np.column_stack([(r + err) * np.cos(th), (r + err) * np.sin(th)])
    return _path_from_points(_resample_closed(pts))


def _project(path, xy):
    """Arc of the nearest table row and signed (outward +) distance from the contour."""
    _, normal = _tangent_normal(path)
    p = path.points_nm
    idx = np.empty(xy.shape[0], dtype=np.int64)
    for a in range(0, xy.shape[0], 512):
        q = xy[a:a + 512]
        d2 = (q[:, None, 0] - p[None, :, 0]) ** 2 + (q[:, None, 1] - p[None, :, 1]) ** 2
        idx[a:a + 512] = np.argmin(d2, axis=1)
    r = np.einsum("ij,ij->i", xy - p[idx], normal[idx])
    return path.cum_nm[idx], r


def _counts(s, length, n_bins, origin=0.0):
    """The bin rule of tools.mps_perimeter_density.density_profile, replicated by hand (bin j CENTRED on
    origin + j b): j = floor((s - origin) / b + 1/2) mod n_bins."""
    b = length / n_bins
    j = np.mod(np.floor((np.asarray(s) - origin) / b + 0.5).astype(np.int64), n_bins)
    return np.bincount(j, minlength=n_bins).astype(np.int64)


def _n_bins(length):
    return max(8, int(round(length / BIN_NM)))   # the grid rule of tools.mps_perimeter_density (plan sec. 3.1)


def _test_on(path, s, n_bins, *, n_null=1999, seed=0):
    tau = bc.tortuosity_windows(path, n_bins=n_bins, origin_arc_nm=0.0)
    dens = bc.window_counts(_counts(s, path.length_nm, n_bins)).astype(float)
    return bc.curvature_density_test(tau, dens, np.ones(n_bins, dtype=bool), n_null=n_null, seed=seed)


def _square(side=3000.0, step=TABLE_STEP_NM):
    """Counter-clockwise square from (0, 0), corners at arcs 0, side, 2 side, 3 side, every corner a table row."""
    m = int(round(side / step))
    t = np.arange(m) * step
    z = np.zeros(m)
    pts = np.vstack([np.column_stack([t, z]), np.column_stack([z + side, t]),
                     np.column_stack([side - t, z + side]), np.column_stack([z, side - t])])
    return _path_from_points(pts)


# ----------------------------------------------------------------------------------------------------------------
# Geometry: analytic values, resolution, invariance, alignment with the density grid
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("radius", [400.0, 600.0, 1000.0, 2000.0])
def test_circle_tortuosity_is_sin_x_over_x(radius):
    path = _circle(radius)
    n = _n_bins(path.length_nm)
    tau = bc.tortuosity_windows(path, n_bins=n, origin_arc_nm=13.0)
    w = 2.0 * path.length_nm / n
    exact = bc.chord_arc_circle(w, radius)
    # plan sec. 4.4: error < 1e-4; the 2 nm table gives ~1e-6 (polygon sagitta), far below the range of tau
    assert np.abs(tau - exact).max() < 1e-5
    assert tau.max() - tau.min() < 2e-6          # a circle has no curvature contrast at all
    assert np.all(tau <= 1.0)


def test_small_angle_approximation_and_numerical_resolution():
    # 1 - (w/R)^2/24 against sin(x)/x: the error is the next term (w/R)^4/1920 [derivation], checked numerically
    w = 200.0
    for radius in (300.0, 400.0, 600.0, 1000.0, 5000.0):
        exact = bc.chord_arc_circle(w, radius)
        approx = bc.chord_arc_small_angle(w, radius)
        assert abs(approx - exact) <= 1.01 * (w / radius) ** 4 / 1920.0
    assert abs(bc.chord_arc_circle(w, 600.0) - 0.99538) < 1e-5      # the plan's "0.995 at R = 600 nm"
    # dynamic range on a 400-600 nm curvature scale: 5e-3 .. 1e-2 below 1, >= 3 orders above the table error
    assert 1.0 - bc.chord_arc_circle(w, 600.0) > 1e3 * 1e-6


def test_spline_path_of_a_circle_matches_the_analytic_value():
    # the real SmoothPath builder (interpolating periodic spline through 24 vertices) on a 700 nm circle
    th = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    path = smooth_closed_path(np.column_stack([700.0 * np.cos(th), 700.0 * np.sin(th)]))
    n = _n_bins(path.length_nm)
    tau = bc.tortuosity_windows(path, n_bins=n, origin_arc_nm=0.0)
    assert np.abs(tau - bc.chord_arc_circle(2.0 * path.length_nm / n, 700.0)).max() < 1e-4


def test_square_corner_windows_take_the_analytic_value():
    path = _square()
    n = 120                                      # L = 12000 nm, b = 100 nm
    origin = 2500.0                              # corner at arc 3000 is the centre of bin 5 (density grid rule)
    tau = bc.tortuosity_windows(path, n_bins=n, origin_arc_nm=origin)
    corner = math.hypot(150.0, 50.0) / 200.0     # legs 150 + 50 nm at a right angle
    expected = np.ones(n)
    for c in (5, 35, 65, 95):                    # corners at arcs 3000, 6000, 9000, 12000 = 0
        expected[[c - 1, c]] = corner
    assert np.abs(tau - expected).max() < 1e-9


def test_alignment_with_the_density_grid_rule():
    """Window i holds bins i and i + 1 of the density grid of tools.mps_perimeter_density (bin j CENTRED on
    origin + j b): a localization at the corner falls in the windows whose tortuosity sees the corner."""
    path = _square()
    n, origin = 120, 2500.0
    b = path.length_nm / n
    tau = bc.tortuosity_windows(path, n_bins=n, origin_arc_nm=origin)
    s = np.array([3000.0, 3000.0 - 0.49 * b, 3000.0 + 0.49 * b])      # all inside bin 5
    counts = _counts(s, path.length_nm, n, origin)
    assert counts[5] == 3 and counts.sum() == 3
    np.testing.assert_array_equal(bc.arc_bin_index(s, length_nm=path.length_nm, n_bins=n, origin_arc_nm=origin),
                                  [5, 5, 5])
    wc = bc.window_counts(counts)
    assert set(np.flatnonzero(wc > 0).tolist()) == {4, 5}
    assert set(np.flatnonzero(tau[:20] < 1.0).tolist()) == {4, 5}
    # window starts are the lower bin edges origin + (i - 1/2) b
    np.testing.assert_allclose(bc._window_starts(path.length_nm, n, origin)[:3],
                               origin + (np.arange(3) - 0.5) * b)
    # the radial median uses the same bins: a localization past the window's end is not in it
    r = np.array([10.0, 20.0, 30.0, 99.0])
    s4 = np.array([3000.0, 3000.0, 3000.0, 3000.0 + 1.51 * b])           # last one in bin 7
    med = bc.window_radial_median(s4, r, length_nm=path.length_nm, n_bins=n, origin_arc_nm=origin)
    assert med[4] == 20.0 and med[5] == 20.0 and med[6] == 99.0 and np.isnan(med[3])


def test_origin_invariance_including_the_window_across_the_closure():
    path = _true_curve(_wavy_r(1100.0))
    n = _n_bins(path.length_nm)
    b = path.length_nm / n
    tau = bc.tortuosity_windows(path, n_bins=n, origin_arc_nm=37.0)
    # moving the origin by k whole bins rotates the windows and changes nothing else
    for k in (1, 5, n - 1):
        np.testing.assert_allclose(bc.tortuosity_windows(path, n_bins=n, origin_arc_nm=37.0 + k * b),
                                   np.roll(tau, -k), atol=1e-12)
    # restarting the table at another row (the closure moves) with the origin compensated: same windows
    row = path.points_nm.shape[0] // 3
    shifted = _path_from_points(np.roll(path.points_nm, -row, axis=0))
    tau2 = bc.tortuosity_windows(shifted, n_bins=n, origin_arc_nm=37.0 - path.cum_nm[row])
    np.testing.assert_allclose(tau2, tau, atol=1e-9)
    # one of those windows really crosses the end of the new table
    starts = np.mod(37.0 - path.cum_nm[row] + (np.arange(n) - 0.5) * b, shifted.length_nm)
    assert np.any(starts + 2 * b > shifted.length_nm)


def test_window_counts_and_measured_wrap():
    c = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    np.testing.assert_array_equal(bc.window_counts(c), [3, 5, 7, 9, 6])
    np.testing.assert_array_equal(bc.window_counts(c, 1), c)
    np.testing.assert_array_equal(bc.window_counts(c, 3), [6, 9, 12, 10, 8])
    m = np.array([True, True, False, True, True])
    np.testing.assert_array_equal(bc.window_measured(m), [True, False, False, True, True])
    with pytest.raises(ValueError):
        bc.window_counts(c, 6)


def test_window_curvature_of_a_circle():
    path = _circle(900.0)
    n = _n_bins(path.length_nm)
    kap = bc.window_curvature(path, n_bins=n, origin_arc_nm=0.0)
    np.testing.assert_allclose(kap, 1.0 / 900.0, rtol=1e-3)


# ----------------------------------------------------------------------------------------------------------------
# The test statistic
# ----------------------------------------------------------------------------------------------------------------
def test_spearman_value_matches_the_reference_implementation():
    from tools.mps_apparent_size import spearman
    rng = np.random.default_rng(3)
    tau = 1.0 - rng.random(60) * 1e-3
    dens = rng.poisson(4.0, 60).astype(float)          # many ties
    out = bc.curvature_density_test(tau, dens, np.ones(60, bool))
    ref, _ = spearman(tau, dens)
    assert abs(out.value - ref) < 1e-12
    assert out.sided == "two" and out.n_null == 59 and 0.0 < out.p <= 1.0


def test_monotone_relation_is_detected_two_sided():
    n = 80
    tau = 1.0 - 1e-3 * np.random.default_rng(8).random(n)       # no autocorrelation: a narrow null
    up = bc.curvature_density_test(tau, tau * 1000.0, np.ones(n, bool))
    down = bc.curvature_density_test(tau, -tau * 1000.0, np.ones(n, bool))
    assert up.value == pytest.approx(1.0) and down.value == pytest.approx(-1.0)
    assert up.p == down.p == pytest.approx(1.0 / n)            # floor, both directions
    assert up.z > 3.0 and down.z < -3.0


def test_undefined_cases_return_nan():
    n = 40
    tau = np.linspace(0.99, 1.0, n)
    for res in (bc.curvature_density_test(tau, np.full(n, 5.0), np.ones(n, bool)),         # constant density
                bc.curvature_density_test(tau[:6], tau[:6], np.ones(6, bool)),             # too few windows
                bc.curvature_density_test(tau, tau, np.arange(n) < 5)):                    # too few measured
        assert math.isnan(res.value) and math.isnan(res.p)


def test_unmeasured_windows_do_not_enter():
    rng = np.random.default_rng(5)
    n = 60
    tau = 1.0 - rng.random(n) * 1e-3
    dens = rng.poisson(6.0, n).astype(float)
    m = np.ones(n, bool)
    m[[3, 4, 30]] = False
    a = bc.curvature_density_test(tau, dens, m)
    dens2 = dens.copy()
    dens2[[3, 4, 30]] = 1e6
    b = bc.curvature_density_test(tau, dens2, m)
    assert a.value == b.value and a.p == b.p
    # with a mask, r(0) is the Pearson correlation, over the shared windows, of ranks taken over each profile's own
    # valid windows (the same statistic at every shift); without a mask it is exactly Spearman
    rt = bc._average_ranks(tau)
    rd = np.zeros(n)
    rd[m] = bc._average_ranks(dens[m])
    ref = float(np.corrcoef(rt[m], rd[m])[0, 1])
    assert abs(a.value - ref) < 1e-12
    from tools.mps_apparent_size import spearman
    assert abs(a.value - spearman(tau[m], dens[m])[0]) < 0.01    # close to the subset Spearman


def test_seeded_subsampled_null_is_deterministic():
    rng = np.random.default_rng(9)
    n = 300
    tau = 1.0 - rng.random(n) * 1e-3
    dens = rng.poisson(5.0, n).astype(float)
    a = bc.curvature_density_test(tau, dens, np.ones(n, bool), n_null=99, seed=4)
    b = bc.curvature_density_test(tau, dens, np.ones(n, bool), n_null=99, seed=4)
    assert a == b and a.n_null == 99


def test_axon_b_readout_resolution_flag_and_fields():
    rng = np.random.default_rng(2)
    for radius, resolved in ((400.0, False), (1000.0, True)):
        path = _circle(radius)
        n = _n_bins(path.length_nm)
        counts = rng.poisson(20.0, n).astype(np.int64)
        out = bc.axon_b_readout(path, counts, np.ones(n, bool), origin_arc_nm=0.0)
        assert out.resolved is resolved
        assert out.tau_min == pytest.approx(bc.chord_arc_circle(2 * path.length_nm / n, radius), abs=1e-5)
        assert math.isnan(out.tau_vs_offset)


# ----------------------------------------------------------------------------------------------------------------
# Simulation controls (calibration and power). Measured values are written in the docstrings.
# ----------------------------------------------------------------------------------------------------------------
def test_false_positive_rate_true_contour():
    """Negative control: wavy axon, density independent of curvature, TRUE contour. Measured 2026-11-16: 0.034 over
    500 axons; the nominal level of p <= 0.05 with 76 shifts is 3/77 = 0.039 (p is discrete)."""
    rng = np.random.default_rng(1)
    true = _true_curve(_wavy_r(1200.0))
    n = _n_bins(true.length_nm)
    tau = bc.tortuosity_windows(true, n_bins=n, origin_arc_nm=0.0)
    ps = np.array([bc.curvature_density_test(
        tau, bc.window_counts(_counts(rng.uniform(0, true.length_nm, 8000), true.length_nm, n)).astype(float),
        np.ones(n, bool)).p for _ in range(500)])
    fpr = float(np.mean(ps <= ALPHA))
    assert 0.02 <= fpr <= 0.07, fpr


def test_false_positive_rate_with_few_localizations():
    """Segments with few localizations (60 per axon over ~77 bins: mostly 0/1/2 per window, many ties) are kept;
    the rank test stays at or below its level (ties make it conservative)."""
    rng = np.random.default_rng(11)
    true = _true_curve(_wavy_r(1200.0))
    n = _n_bins(true.length_nm)
    tau = bc.tortuosity_windows(true, n_bins=n, origin_arc_nm=0.0)
    ps = []
    for _ in range(400):
        dens = bc.window_counts(_counts(rng.uniform(0, true.length_nm, 60), true.length_nm, n)).astype(float)
        res = bc.curvature_density_test(tau, dens, np.ones(n, bool))
        assert 0.0 < res.p <= 1.0
        ps.append(res.p)
    assert float(np.mean(np.array(ps) <= ALPHA)) <= 0.07


def test_hard_negative_noisy_contour_bias_is_flagged_by_the_offset_control():
    """Hard negative (plan sec. 4.4): density independent of curvature, but the contour is a noisy fit (corner cut +
    15 nm low-frequency radial error), localizations projected and kept in a band of 50 / 100 / 150 nm.

    Measured 2026-11-16 (100 axons per shape, this generator): wavy FPR 0.09 / 0.07 / 0.07 and ellipse 0.10 / 0.11 /
    0.11 at 50 / 100 / 150 nm, mean Spearman ~ +0.07..+0.10: the contour-band bias is REAL and the rotation null does
    not see it. Decomposition (80 axons): corner cut alone at harmonics <= 6 gives FPR 0.03-0.06 and r ~ 0; a
    harsher cut (harmonics <= 3) gives r ~ -0.15 on the ellipse (FPR 0.11); the 15 nm radial error alone gives r ~
    +0.1; a 5 nm error gives r ~ 0. In every biased case the control tau_vs_offset (Spearman of tortuosity with the
    median radial offset per window) is far from 0 (+0.25 to +0.55, or -0.2), and ~0 when the bias is absent: that
    control is what flags an axon whose B cannot be read. Asserted here: the bias stays bounded and is flagged."""
    rng = np.random.default_rng(21)
    fn = _wavy_r(1200.0)
    true = _true_curve(fn)
    hits = {50.0: 0, 100.0: 0, 150.0: 0}
    offsets = []
    n_rep = 60
    for _ in range(n_rep):
        xy, _ = _simulate(true, 6000, rng)
        cont = _noisy_contour(fn, rng)
        s, r = _project(cont, xy)
        n = _n_bins(cont.length_nm)
        for hw in hits:
            keep = np.abs(r) <= hw
            res = bc.axon_b_readout(cont, _counts(s[keep], cont.length_nm, n), np.ones(n, bool), origin_arc_nm=0.0)
            hits[hw] += int(res.test.p <= ALPHA)
        keep = np.abs(r) <= 100.0
        rm = bc.window_radial_median(s[keep], r[keep], length_nm=cont.length_nm, n_bins=n, origin_arc_nm=0.0)
        res = bc.axon_b_readout(cont, _counts(s[keep], cont.length_nm, n), np.ones(n, bool), origin_arc_nm=0.0,
                                radial_median_nm=rm)
        offsets.append(res.tau_vs_offset)
    for hw, h in hits.items():
        assert h / n_rep <= 0.17, (hw, h / n_rep)
    assert float(np.nanmean(offsets)) > 0.2


def test_positive_density_follows_curvature():
    """Positive control: lambda proportional to 1 + 0.5 kappa R, 10 000 localizations, R = L / 2 pi ~ 1.2 um
    (resolved). Measured 2026-11-16 (60 axons): power 1.00 with the true contour, 0.73 with the noisy contour; the
    Spearman is negative (more curvature = lower tortuosity = more density)."""
    rng = np.random.default_rng(31)
    fn = _wavy_r(1200.0)
    true = _true_curve(fn)
    n = _n_bins(true.length_nm)
    hits_true, hits_noisy, values = 0, 0, []
    n_rep = 30
    for _ in range(n_rep):
        xy, s_true = _simulate(true, 10000, rng, beta=0.5)
        out = _test_on(true, s_true, n)
        hits_true += int(out.p <= ALPHA)
        values.append(out.value)
        cont = _noisy_contour(fn, rng)
        s, r = _project(cont, xy)
        keep = np.abs(r) <= 100.0
        n2 = _n_bins(cont.length_nm)
        res = bc.axon_b_readout(cont, _counts(s[keep], cont.length_nm, n2), np.ones(n2, bool), origin_arc_nm=0.0)
        hits_noisy += int(res.test.p <= ALPHA)
        assert res.resolved
    assert hits_true / n_rep >= 0.8
    assert hits_noisy / n_rep >= 0.5
    assert float(np.mean(values)) < -0.5


def test_symmetric_shape_limits_the_rotation_null():
    """Documented limit: on a 2-fold symmetric ellipse the tortuosity repeats every half perimeter and its rank
    profile anti-aligns at a quarter, so the rotation null is very broad: even with the density an exact, noiseless
    function of the curvature (r = -1.00) the z against the null is only ~ -1.5, and any noise lets shifts near
    a half or a quarter of the perimeter reach |r(0)|. Measured: power 0.60 (true contour, 60 axons, 10 000
    localizations) against 1.00 on the asymmetric wavy shape."""
    true = _true_curve(_ellipse_r(1400.0, 900.0))
    n = _n_bins(true.length_nm)
    tau = bc.tortuosity_windows(true, n_bins=n, origin_arc_nm=0.0)
    kap = bc.window_curvature(true, n_bins=n, origin_arc_nm=0.0)
    dens = 1.0 + 0.5 * kap * true.length_nm / (2 * math.pi)          # exact, noiseless
    res = bc.curvature_density_test(tau, dens, np.ones(n, bool))
    assert res.value < -0.9
    assert res.null_sd > 0.5 and abs(res.z) < 2.0
