"""Tests of the projection, band and density profiles of the perimeter density analysis (package P4).

Simulated minimal data only: circles and ellipses built by hand (the simulator package does not exist yet).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.mps_matching import SmoothPath  # noqa: E402
from tools.mps_perimeter_density import (aligned_on_own_max, arc_grid, band_width, centred_lags_nm,  # noqa: E402
                                         centred_order, density_profile, own_max_profiles, peak_bin, project,
                                         slice_profiles)


def make_path(points: np.ndarray) -> SmoothPath:
    """A SmoothPath from a closed dense polyline (curve order, closed by the wrap)."""
    pts = np.asarray(points, dtype=np.float64)
    edges = np.sqrt(((np.roll(pts, -1, axis=0) - pts) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    return SmoothPath(points_nm=pts, vertex_row=np.zeros(0, dtype=np.intp), edges_nm=edges, cum_nm=cum,
                      length_nm=float(cum[-1]))


def ellipse(a: float, b: float, n: int = 4000, ccw: bool = True) -> SmoothPath:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if not ccw:
        t = -t                   # same start point, opposite direction
    return make_path(np.column_stack([a * np.cos(t), b * np.sin(t)]))


# ---------------------------------------------------------------------------------------------- project
@pytest.mark.parametrize("ccw", [True, False])
def test_project_circle_sign_and_arc(ccw):
    path = ellipse(1000.0, 1000.0, ccw=ccw)
    theta = np.array([0.3, 1.7, 3.0, 5.5])
    for rad, expect in [(1050.0, 50.0), (940.0, -60.0)]:
        s, r = project(path, rad * np.cos(theta), rad * np.sin(theta))
        np.testing.assert_allclose(r, expect, atol=0.01)       # + outside, whatever the orientation
    # arc of a point is the cumulative length at the foot (up to the start at angle 0 and the direction)
    s, _ = project(path, 1000.0 * np.cos(theta), 1000.0 * np.sin(theta))
    L = path.length_nm
    want = (1000.0 * theta) % L if ccw else (L - 1000.0 * theta) % L
    d = np.abs(s - want)
    assert np.all(np.minimum(d, L - d) < 1.0)
    assert np.all((s >= 0.0) & (s < L))


def test_project_ellipse_outside_positive_and_validation():
    path = ellipse(1500.0, 800.0)
    _, r = project(path, np.array([1600.0, 0.0, 0.0]), np.array([0.0, 880.0, 0.0]))
    assert r[0] == pytest.approx(100.0, abs=0.5) and r[1] == pytest.approx(80.0, abs=0.5) and r[2] < -700.0
    with pytest.raises(ValueError):
        project(path, np.zeros(3), np.zeros(2))


# ---------------------------------------------------------------------------------------------- arc_grid
def test_arc_grid():
    n, w = arc_grid(10_050.0, 100.0)
    assert n in (100, 101) and w == pytest.approx(10_050.0 / n)
    assert abs(w - 100.0) / 100.0 < 0.5 / n + 1e-12
    n, w = arc_grid(300.0, 100.0)          # never fewer than 8 bins
    assert n == 8 and w == pytest.approx(37.5)
    assert arc_grid(250.0, 10.0)[0] == 25
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            arc_grid(bad)


# ---------------------------------------------------------------------------------------------- peak_bin
def test_peak_bin_unique_tie_and_empty():
    assert peak_bin(np.array([1, 4, 2, 0], dtype=np.int64)) == (1, False)
    # tie broken by the larger circular 3-bin sum
    c = np.array([5, 1, 0, 0, 5, 3, 0, 0], dtype=np.int64)
    assert peak_bin(c) == (4, True)
    # circular neighbours count: bin 0 has neighbours 7 and 1
    c = np.array([5, 3, 0, 0, 0, 0, 0, 5, 4, 0], dtype=np.int64)
    assert peak_bin(c)[0] == 7 and peak_bin(c)[1] is True    # 7: 0+5+4... compare with bin 0
    # full tie of sums: lowest index wins, deterministic
    c = np.array([2, 0, 0, 0, 2, 0, 0, 0], dtype=np.int64)
    assert peak_bin(c) == (0, True) and peak_bin(c.copy()) == (0, True)
    assert peak_bin(np.zeros(5, dtype=np.int64)) == (-1, False)
    assert peak_bin(np.zeros(0, dtype=np.int64)) == (-1, False)


def test_density_profile_tie_flag():
    L, n = 800.0, 8
    w = L / n
    s = np.concatenate([np.full(4, 1 * w), np.full(4, 5 * w), np.full(1, 6 * w)])
    p = density_profile(s, L, n_bins=n)
    assert p.max_tie is True and p.max_bin == 5            # 5 has the larger 3-bin sum (4 + 1)
    p2 = density_profile(s.copy(), L, n_bins=n)
    assert p2.max_bin == p.max_bin


# ---------------------------------------------------------------------------------------------- profile
def test_conservation_and_relative_max():
    rng = np.random.default_rng(1)
    s = rng.uniform(0.0, 5000.0, 700)
    p = density_profile(s, 5000.0, n_bins=50)
    assert p.counts.sum() == 700 == p.n_locs
    assert p.relative.max() == 1.0 and p.relative[p.max_bin] == 1.0
    assert p.bin_nm_effective == pytest.approx(100.0) and p.counts.dtype == np.int64
    # arcs beyond L wrap
    q = density_profile(s + 5000.0, 5000.0, n_bins=50)
    np.testing.assert_array_equal(p.counts, q.counts)


def test_empty_profile_no_division_by_zero():
    p = density_profile(np.zeros(0), 5000.0, n_bins=50)
    assert p.counts.sum() == 0 and p.n_locs == 0
    assert np.all(p.relative == 0.0) and np.isfinite(p.relative).all()
    assert p.max_bin == -1 and p.max_tie is False and not p.measured.any()
    assert aligned_on_own_max(p) is p
    with pytest.raises(ValueError):
        density_profile(np.array([1.0, np.nan]), 5000.0, n_bins=50)


def test_origin_rotation_only_rotates():
    rng = np.random.default_rng(2)
    L, n = 4000.0, 40
    w = L / n
    s = (np.floor(rng.uniform(0, n, 600)) + rng.uniform(-0.4, 0.4, 600)) * w    # far from bin edges
    base = density_profile(s, L, n_bins=n)
    for k in (1, 7, 39):
        rot = density_profile(s, L, n_bins=n, origin_arc_nm=k * w)
        np.testing.assert_array_equal(rot.counts, np.roll(base.counts, -k))
        np.testing.assert_array_equal(rot.measured, np.roll(base.measured, -k))
        assert rot.n_locs == base.n_locs
        assert rot.max_bin == (base.max_bin - k) % n


def test_not_measured_gap_and_circular_closure():
    L, n = 6000.0, 60                      # bin 100 nm
    rng = np.random.default_rng(3)
    # data on [-500, 1500) (crosses the wrap at 0) and nothing on [1500, 5500): a 4000 nm gap
    s = np.mod(rng.uniform(-500.0, 1500.0, 4000), L)
    p = density_profile(s, L, n_bins=n)
    centres = np.arange(n) * 100.0
    in_gap = (centres > 1700.0) & (centres < 5300.0)
    assert not p.measured[in_gap].any()
    assert p.measured[(centres < 1300.0) | (centres > 5700.0)].all()      # wrap-around bins are measured
    # a gap shorter than 400 nm never marks a bin
    s2 = np.concatenate([np.linspace(0, 2000, 300), np.linspace(2300, L - 1, 700)])
    assert density_profile(s2, L, n_bins=n).measured.all()
    # one single point: everything away from it is unmeasured, its own bin is not
    one = density_profile(np.array([3000.0]), L, n_bins=n)
    assert one.measured.sum() == 1 and bool(one.measured[30])


# ---------------------------------------------------------------------------------------------- slices
def _slice_data(seed: int = 4):
    rng = np.random.default_rng(seed)
    L = 6000.0
    n = 3000
    s = rng.uniform(0, L, n)
    r = rng.normal(0.0, 40.0, n)
    lab = rng.choice(np.array([-1, 0, 1, -9], dtype=np.int8), n)
    # make the central slice peak near arc 2500
    extra = rng.normal(2500.0, 150.0, 800) % L
    s = np.concatenate([s, extra])
    r = np.concatenate([r, rng.normal(0.0, 30.0, 800)])
    lab = np.concatenate([lab, np.zeros(800, dtype=np.int8)])
    return s, r, lab, L


def test_slice_profiles_common_grid_origin_and_conservation():
    s, r, lab, L = _slice_data()
    sp = slice_profiles(s, r, lab, L, half_width_nm=100.0)
    assert sp.frame == "z0_max" and set(sp.by_slice) == {-1, 0, 1}
    origins = {p.origin_arc_nm for p in sp.by_slice.values()}
    grids = {(p.n_bins, p.bin_nm_effective) for p in sp.by_slice.values()}
    assert len(origins) == 1 and len(grids) == 1
    for k, p in sp.by_slice.items():
        want = int(np.count_nonzero((lab == k) & (np.abs(r) <= 100.0)))
        assert p.counts.sum() == p.n_locs == want
        assert p.relative.max() == 1.0
    assert sp.by_slice[0].max_bin == 0                       # the origin is the central maximum
    assert abs(sp.by_slice[0].origin_arc_nm - 2500.0) < 300.0
    # wider band -> more points (50 / 150 nm sensitivities use the same call)
    n50 = sum(p.n_locs for p in slice_profiles(s, r, lab, L, half_width_nm=50.0).by_slice.values())
    n150 = sum(p.n_locs for p in slice_profiles(s, r, lab, L, half_width_nm=150.0).by_slice.values())
    assert n50 < sum(p.n_locs for p in sp.by_slice.values()) < n150


def test_slice_profiles_empty_central_and_validation():
    s = np.array([100.0, 200.0, 300.0])
    sp = slice_profiles(s, np.zeros(3), np.array([1, 1, -9], dtype=np.int8), 5000.0, half_width_nm=100.0)
    assert sp.by_slice[0].n_locs == 0 and sp.by_slice[0].max_bin == -1
    assert sp.by_slice[1].n_locs == 2 and sp.by_slice[-1].n_locs == 0
    assert sp.by_slice[0].origin_arc_nm == sp.by_slice[1].origin_arc_nm == 0.0
    with pytest.raises(ValueError):
        slice_profiles(s, np.zeros(2), np.zeros(3, dtype=np.int8), 5000.0, half_width_nm=100.0)
    with pytest.raises(ValueError):
        slice_profiles(s, np.zeros(3), np.zeros(3, dtype=np.int8), 5000.0, half_width_nm=0.0)


def test_rotating_the_contour_origin_rotates_the_common_profile_only():
    s, r, lab, L = _slice_data()
    sp = slice_profiles(s, r, lab, L, half_width_nm=100.0)
    sp2 = slice_profiles(np.mod(s + 1200.0, L), r, lab, L, half_width_nm=100.0)
    for k in (-1, 0, 1):
        np.testing.assert_array_equal(sp.by_slice[k].counts, sp2.by_slice[k].counts)
        assert sp2.by_slice[k].origin_arc_nm == pytest.approx((sp.by_slice[k].origin_arc_nm + 1200.0) % L)


def test_own_max_frame_cannot_be_confused_with_common_frame():
    s, r, lab, L = _slice_data()
    sp = slice_profiles(s, r, lab, L, half_width_nm=100.0)
    own = own_max_profiles(sp)
    assert sp.frame == "z0_max" and own.frame == "own_max"
    for k, p in own.by_slice.items():
        assert p.max_bin == 0 and p.counts[0] == p.counts.max()
        assert p.n_locs == sp.by_slice[k].n_locs
        np.testing.assert_array_equal(np.sort(p.counts), np.sort(sp.by_slice[k].counts))
    # aligning an already rotated copy gives the same profile and the same origin
    a1 = aligned_on_own_max(sp.by_slice[1])
    shifted = density_profile(np.mod(s[(lab == 1) & (np.abs(r) <= 100.0)], L), L, n_bins=sp.by_slice[1].n_bins,
                              origin_arc_nm=sp.by_slice[1].origin_arc_nm + 7 * sp.by_slice[1].bin_nm_effective)
    a2 = aligned_on_own_max(shifted)
    np.testing.assert_array_equal(a1.counts, a2.counts)
    assert a2.origin_arc_nm == pytest.approx(a1.origin_arc_nm)


# ---------------------------------------------------------------------------------------------- axis
@pytest.mark.parametrize("n", [8, 9, 100, 101])
def test_centred_axis(n):
    w = 100.0
    lags = centred_lags_nm(n, w)
    L = n * w
    assert lags.size == n and np.all(np.diff(lags) > 0)
    assert lags[0] > -L / 2 and lags[-1] <= L / 2 + 1e-9 and (0.0 in lags)
    order = centred_order(n)
    assert sorted(order.tolist()) == list(range(n))
    assert order[int(np.flatnonzero(lags == 0.0)[0])] == 0           # the origin bin sits at lag 0
    # neighbour bins sit at +-1 bin
    z = int(np.flatnonzero(lags == 0.0)[0])
    assert order[z + 1] == 1 and order[z - 1] == n - 1


# ---------------------------------------------------------------------------------------------- band
def test_band_width_gaussian_fwhm():
    rng = np.random.default_rng(5)
    r = rng.normal(7.0, 25.0, 20_000)
    b = band_width(r, mode="fwhm")
    assert abs(b.fwhm_nm - 59.0) < 5.0 and abs(b.robust_sd_nm - 25.0) < 1.5
    assert b.mode == "fwhm" and 50.0 <= b.half_width_nm <= 150.0
    assert b.half_width_nm == pytest.approx(b.fwhm_nm)
    fx = band_width(r)
    assert fx.mode == "fixed" and fx.half_width_nm == 100.0
    assert set(fx.frac_within) == {50.0, 100.0, 150.0}
    assert fx.frac_within[50.0] <= fx.frac_within[100.0] <= fx.frac_within[150.0] <= 1.0
    assert abs(fx.frac_within[100.0] - 1.0) < 0.01 or fx.frac_within[100.0] > 0.97


def test_band_width_clipping_inside_outside_and_empty():
    rng = np.random.default_rng(6)
    assert band_width(rng.normal(0, 5.0, 5000), mode="fwhm").half_width_nm == 50.0
    assert band_width(rng.normal(0, 200.0, 5000), mode="fwhm").half_width_nm == 150.0
    assert band_width(rng.normal(0, 100.0, 100), mode="fixed", fixed_nm=50.0).half_width_nm == 50.0
    r = np.array([-200.0, -10.0, 0.0, 10.0, 120.0, 300.0, 400.0])
    b = band_width(r, fixed_nm=100.0)
    assert b.frac_inside == pytest.approx(1 / 7) and b.frac_outside == pytest.approx(3 / 7)
    e = band_width(np.zeros(0), mode="fwhm")
    assert e.mode == "fixed" and e.half_width_nm == 100.0 and np.isnan(e.fwhm_nm) and np.isnan(e.frac_inside)
    with pytest.raises(ValueError):
        band_width(r, mode="bogus")


def test_symmetric_band_on_circle_gives_flat_profile():
    """Uniform areal density in a symmetric band: counts per bin are flat within Poisson (chi-square)."""
    rng = np.random.default_rng(7)
    R, half, n_pts = 1500.0, 100.0, 40_000
    # uniform per unit area in the annulus: density of r proportional to (R + r)
    u = rng.uniform(0.0, 1.0, n_pts)
    r_in = np.sqrt((R - half) ** 2 + u * ((R + half) ** 2 - (R - half) ** 2)) - R
    th = rng.uniform(0.0, 2.0 * np.pi, n_pts)
    x, y = (R + r_in) * np.cos(th), (R + r_in) * np.sin(th)
    path = ellipse(R, R, n=6000)
    s, r = project(path, x, y)
    lab = np.zeros(n_pts, dtype=np.int8)
    sp = slice_profiles(s, r, lab, path.length_nm, half_width_nm=half)
    c = sp.by_slice[0].counts
    expect = c.sum() / c.size
    chi2 = float(((c - expect) ** 2 / expect).sum())
    df = c.size - 1
    assert abs(chi2 - df) < 5.0 * np.sqrt(2.0 * df)
    assert c.sum() == n_pts
    # an asymmetric band would not be flat on this geometry: nothing to assert, the module offers none


def test_publishable_text_has_no_researcher_name():
    here = Path(__file__).resolve().parent
    for f in (here / "tools" / "mps_perimeter_density.py", Path(__file__)):
        text = f.read_text(encoding="utf-8").lower()
        assert ("unsa" + "in") not in text and ("unsa" + "ín") not in text
