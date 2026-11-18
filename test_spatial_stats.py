# -*- coding: utf-8 -*-
"""
tools/spatial_stats.py on synthetic point patterns only (never real data): CSR in a rectangle and on a
closed arc keep H inside the Monte Carlo envelopes, planted Gaussian clouds on a ring are detected at
their own scale, g(r) behaves, the edge correction matters where it should, and arc_coordinates
inverts the generation of points on a contour.

Run:  py -3 -m pytest test_spatial_stats.py     or     venv\\Scripts\\python.exe test_spatial_stats.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.spatial_stats import (  # noqa: E402
    arc_coordinates,
    csr_envelopes,
    is_axis_aligned_rectangle,
    pair_correlation_1d_arc,
    pair_correlation_2d,
    point_in_polygon,
    polygon_area,
    ripley_k_1d_arc,
    ripley_k_2d,
    simulate_csr_arc,
    simulate_csr_polygon,
)

RECT = np.array([[0.0, 0.0], [2000.0, 0.0], [2000.0, 2000.0], [0.0, 2000.0]])
R_GRID = np.arange(20.0, 401.0, 10.0)          # A9 spec sec. 2.5
RING_LENGTH_NM = 2000.0                        # "20 clusters on a 2 um ring"
N_CLUSTERS = 20
SIGMA_NM = 12.0
LOCS_PER_CLUSTER = 25


def _planted_ring_1d(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = (np.arange(N_CLUSTERS) + 0.5) * RING_LENGTH_NM / N_CLUSTERS
    s = centres[:, None] + rng.normal(0.0, SIGMA_NM, (N_CLUSTERS, LOCS_PER_CLUSTER))
    return np.mod(s.ravel(), RING_LENGTH_NM)


def _planted_slab_2d(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The unrolled band: a 2000 x 200 nm rectangle with the 20 clouds along its midline (y = 100)."""
    rng = np.random.default_rng(seed)
    centres = (np.arange(N_CLUSTERS) + 0.5) * RING_LENGTH_NM / N_CLUSTERS
    x = centres[:, None] + rng.normal(0.0, SIGMA_NM, (N_CLUSTERS, LOCS_PER_CLUSTER))
    y = 100.0 + rng.normal(0.0, SIGMA_NM, (N_CLUSTERS, LOCS_PER_CLUSTER))
    xy = np.c_[x.ravel(), y.ravel()]
    band = np.array([[0.0, 0.0], [RING_LENGTH_NM, 0.0], [RING_LENGTH_NM, 200.0], [0.0, 200.0]])
    keep = point_in_polygon(xy[:, 0], xy[:, 1], band)
    return xy[keep], band


# --------------------------------------------------------------------------- utilities
def test_polygon_area_and_rectangle_detection() -> None:
    assert abs(polygon_area(RECT) - 4.0e6) < 1e-6
    assert abs(polygon_area(RECT[::-1]) - 4.0e6) < 1e-6              # orientation does not matter
    tri = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    assert abs(polygon_area(tri) - 6.0) < 1e-12
    assert is_axis_aligned_rectangle(RECT)
    assert not is_axis_aligned_rectangle(tri)
    assert not is_axis_aligned_rectangle(np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [0.0, 1.0]]))


def test_point_in_polygon_concave() -> None:
    # an L shape: (1.5, 1.5) is inside the notch, hence outside
    poly = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
    inside = point_in_polygon(np.array([0.5, 1.5, 1.5]), np.array([0.5, 0.5, 1.5]), poly)
    assert inside.tolist() == [True, True, False]


# --------------------------------------------------------------------------- (a) CSR in 2D
def test_a_csr_rectangle_inside_envelopes() -> None:
    n = 500
    inside_fracs = []
    p_values = []
    for seed in range(10):
        xy = simulate_csr_polygon(n, RECT, np.random.default_rng(seed))
        obs = ripley_k_2d(xy, RECT, R_GRID, "translation").h
        env = csr_envelopes(lambda p: ripley_k_2d(p, RECT, R_GRID, "translation").h, n, RECT, R_GRID,
                            n_sim=99, seed=1000 + seed, observed=obs)      # replicates never share the pattern's seed
        inside_fracs.append(float(np.mean((obs >= env.pointwise_lo) & (obs <= env.pointwise_hi))))
        p_values.append(env.p_global)
        assert env.sims.shape == (99, R_GRID.size)
        assert np.all(env.global_lo <= env.pointwise_lo + 1e-9) and np.all(env.global_hi >= env.pointwise_hi - 1e-9)
    # a CSR pattern strays from the POINTWISE band somewhere in about half the cases (39 correlated
    # radii; 47 % in a 40-seed calibration, 20 % with less than 90 % of the radii inside), so the
    # pointwise criterion is asserted on the median seed and the decision rests on the global p
    assert float(np.median(inside_fracs)) >= 0.90, inside_fracs
    assert sum(p > 0.05 for p in p_values) >= 9, p_values


def test_csr_envelopes_are_deterministic() -> None:
    fn = lambda p: ripley_k_1d_arc(p, 1000.0, R_GRID).h       # noqa: E731
    e1 = csr_envelopes(fn, 100, 1000.0, R_GRID, n_sim=20, seed=7)
    e2 = csr_envelopes(fn, 100, 1000.0, R_GRID, n_sim=20, seed=7)
    assert np.array_equal(e1.sims, e2.sims)


# --------------------------------------------------------------------------- (b) CSR in 1D
def test_b_csr_closed_arc_h1_near_zero() -> None:
    length = 2000.0
    r = np.arange(10.0, 801.0, 10.0)
    for seed in range(5):
        s = simulate_csr_arc(1000, length, np.random.default_rng(300 + seed))
        res = ripley_k_1d_arc(s, length, r)
        assert np.all(np.abs(res.h) < 0.02 * length), np.abs(res.h).max()
        ratio = res.k / (2.0 * r)
        assert np.all(np.abs(ratio[r >= 50.0] - 1.0) < 0.08), ratio
    # an open arc with the 1D Ripley weight stays unbiased too
    s = simulate_csr_arc(1000, length, np.random.default_rng(42))
    res_open = ripley_k_1d_arc(s, length, r, closed=False)
    assert np.all(np.abs(res_open.h) < 0.02 * length), np.abs(res_open.h).max()


# --------------------------------------------------------------------------- (c) planted clouds
def test_c_planted_clouds_1d_detected_at_cloud_scale() -> None:
    r = np.arange(5.0, 201.0, 5.0)
    s = _planted_ring_1d(11)
    obs = ripley_k_1d_arc(s, RING_LENGTH_NM, r).h
    env = csr_envelopes(lambda p: ripley_k_1d_arc(p, RING_LENGTH_NM, r).h, s.size, RING_LENGTH_NM, r,
                        n_sim=99, seed=1, observed=obs)
    assert env.p_global < 0.05
    assert obs[r <= 40.0].min() > 0.0         # clouds 100 nm apart: H crosses zero near 45 nm (regular beyond)
    assert env.h_max > 0.0
    assert 15.0 <= env.r_star <= 60.0, env.r_star


def test_c_planted_clouds_2d_detected_at_cloud_scale() -> None:
    r = np.arange(5.0, 201.0, 5.0)
    xy, band = _planted_slab_2d(12)
    obs = ripley_k_2d(xy, band, r, "translation").h
    env = csr_envelopes(lambda p: ripley_k_2d(p, band, r, "translation").h, xy.shape[0], band, r,
                        n_sim=99, seed=2, observed=obs)
    assert env.p_global < 0.05
    assert obs[r <= 40.0].min() > 0.0         # clouds 100 nm apart: H crosses zero near 45 nm (regular beyond)
    assert env.h_max > 0.0
    assert 15.0 <= env.r_star <= 60.0, env.r_star


# --------------------------------------------------------------------------- (d) pair correlation
def test_d_pair_correlation_csr_and_clustered() -> None:
    edges = np.arange(0.0, 401.0, 20.0)
    xy = simulate_csr_polygon(2000, RECT, np.random.default_rng(5))
    g = pair_correlation_2d(xy, RECT, edges, "translation").g
    assert np.all(np.abs(g[edges[1:] > 100.0] - 1.0) < 0.15), g
    s = simulate_csr_arc(2000, 2000.0, np.random.default_rng(6))
    g1 = pair_correlation_1d_arc(s, 2000.0, edges).g
    assert np.all(np.abs(g1 - 1.0) < 0.15), g1
    xy_c, band = _planted_slab_2d(13)
    gc = pair_correlation_2d(xy_c, band, edges, "translation").g
    assert gc[0] > 1.5 and gc[1] > 1.0
    g1c = pair_correlation_1d_arc(_planted_ring_1d(14), RING_LENGTH_NM, edges).g
    assert g1c[0] > 1.5                      # sigma 12 nm: nearly all within-cloud pairs fall below 20 nm in 1D


# --------------------------------------------------------------------------- (e) edge correction
def test_e_uncorrected_k_underestimates_in_small_rectangle() -> None:
    small = np.array([[0.0, 0.0], [600.0, 0.0], [600.0, 600.0], [0.0, 600.0]])
    r = np.arange(20.0, 201.0, 20.0)
    k_none = np.zeros(r.size)
    k_ripley = np.zeros(r.size)
    k_trans = np.zeros(r.size)
    for seed in range(5):
        xy = simulate_csr_polygon(400, small, np.random.default_rng(500 + seed))
        k_none += ripley_k_2d(xy, small, r, "none").k
        k_ripley += ripley_k_2d(xy, small, r, "ripley").k
        k_trans += ripley_k_2d(xy, small, r, "translation").k
    expected = np.pi * r ** 2
    assert np.all(k_none[r >= 60.0] < k_ripley[r >= 60.0])
    assert np.all(k_none[r >= 60.0] < k_trans[r >= 60.0])
    # the corrected estimators sit near pi r^2, the uncorrected one well below at r = 200 (> 25 % low)
    assert abs(k_ripley[-1] / 5 / expected[-1] - 1.0) < 0.10
    assert abs(k_trans[-1] / 5 / expected[-1] - 1.0) < 0.10
    assert k_none[-1] / 5 / expected[-1] < 0.75


# --------------------------------------------------------------------------- (f) arc coordinates
def test_f_arc_coordinates_recovers_s() -> None:
    rng = np.random.default_rng(9)
    theta = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    radius = 500.0
    contour = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    s_true = rng.random(300) * 2.0 * np.pi * radius
    d_true = rng.normal(0.0, 20.0, 300)
    ang = s_true / radius
    xy = np.c_[(radius + d_true) * np.cos(ang), (radius + d_true) * np.sin(ang)]
    s, d, length = arc_coordinates(xy, contour)
    # polygon length is slightly below 2 pi R; compare on the polygon's own scale
    s_exp = np.mod(s_true, 2.0 * np.pi * radius) * length / (2.0 * np.pi * radius)
    err = np.abs(s - s_exp)
    err = np.minimum(err, length - err)
    assert err.max() < 2.0, err.max()
    assert np.all(np.sign(d) == np.sign(d_true))
    assert np.abs(np.abs(d) - np.abs(d_true)).max() < 1.0


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL  {name}: {exc}")
    print("all tests passed" if not failed else f"{failed} test(s) failed")
    sys.exit(1 if failed else 0)
