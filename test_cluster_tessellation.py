# -*- coding: utf-8 -*-
"""
Tests of the tessellation family (Voronoi density segmentation, SR-Tesseler / ClusterViSu), on synthetic
data only (R8):

- the geometry: Sutherland-Hodgman clipping, point-in-polygon, clipped cell areas summing to the window;
- (a) pure CSR in a rectangle: with the CSR threshold at the 0.99 quantile, at most 5 % of the points end
  up in clusters;
- (b) Gaussian clusters (sigma 12 nm, 15-30 points) on a ring over uniform noise: at least 90 % of the
  planted clusters are recovered in both threshold modes;
- (c) border cells: no infinite or NaN density, and points outside the ROI are noise, not infinities;
- (d) the interface: label length, -1 for noise, serialisable ``get_params``, determinism with the seed,
  the family registry.

Run: ``py -3 -m pytest test_cluster_tessellation.py -q``.
"""
from __future__ import annotations

import json
from typing import Tuple

import numpy as np
import pytest

from tools import cluster_tessellation as ct

SQUARE = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]])


def ring_dataset(seed: int, n_clusters: int = 20, n_noise: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters (sigma 12 nm, 15-30 points) on a ring of radius 400 nm plus uniform noise.
    Returns (xy, truth) with truth -1 for noise."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
    cx = 500.0 + 400.0 * np.cos(ang)
    cy = 500.0 + 400.0 * np.sin(ang)
    parts = []
    truth = []
    for k in range(n_clusters):
        m = int(rng.integers(15, 31))
        parts.append(np.column_stack([cx[k] + rng.normal(0.0, 12.0, m), cy[k] + rng.normal(0.0, 12.0, m)]))
        truth += [k] * m
    parts.append(rng.uniform(0.0, 1000.0, (n_noise, 2)))
    truth += [-1] * n_noise
    return np.vstack(parts), np.asarray(truth)


def recovered_fraction(labels: np.ndarray, truth: np.ndarray) -> float:
    """A planted cluster counts as recovered when one found cluster holds at least 30 % of its points
    and that found cluster is made mostly (> 50 %) of them (border points may be left out, as the
    tessellation literature reports)."""
    ids = np.unique(truth[truth >= 0])
    hits = 0
    for k in ids:
        mine = labels[truth == k]
        mine = mine[mine >= 0]
        if mine.size == 0:
            continue
        best = int(np.argmax(np.bincount(mine)))
        share = float((mine == best).sum()) / float((truth == k).sum())
        purity = float((truth[labels == best] == k).mean())
        if share >= 0.3 and purity > 0.5:
            hits += 1
    return hits / float(ids.size)


# =========================================================================== geometry
def test_clip_convex_polygon_areas() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    big = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    small = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert ct.polygon_area(ct.clip_convex_polygon(square, big)) == pytest.approx(1.0)
    assert ct.polygon_area(ct.clip_convex_polygon(square, small)) == pytest.approx(0.5)
    far = np.array([[5.0, 5.0], [6.0, 5.0], [5.0, 6.0]])
    assert ct.clip_convex_polygon(square, far).shape[0] == 0


def test_points_in_convex_polygon() -> None:
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [0.0, 0.0], [-0.1, 0.5]])
    inside = ct.points_in_convex_polygon(pts, np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    assert inside.tolist() == [True, False, True, False]


def test_clipped_cells_tile_the_window() -> None:
    rng = np.random.default_rng(3)
    pts = rng.uniform(0.0, 1000.0, (500, 2))
    cells = ct.voronoi_cells(pts, SQUARE)
    assert cells.area.sum() == pytest.approx(1.0e6, rel=1e-9)
    assert (cells.area > 0).all()
    assert cells.ridge_points.max() < 500 and cells.ridge_points.min() >= 0


# =========================================================================== (a) CSR stays unclustered
def test_csr_threshold_leaves_pure_csr_unclustered() -> None:
    rng = np.random.default_rng(11)
    pts = rng.uniform(0.0, 1000.0, (1500, 2))
    strat = ct.VoronoiDensityStrategy(threshold_mode="csr", csr_quantile=0.99, csr_iterations=20,
                                      roi_polygon=SQUARE, rng_seed=5)
    labels = strat.fit(pts)
    assert labels.shape == (1500,)
    assert float((labels >= 0).mean()) <= 0.05
    d = strat.diagnostics
    assert d is not None and d.threshold_mode == "csr"
    assert d.csr_probabilities.shape == d.csr_density_quantiles.shape == d.observed_density_quantiles.shape
    assert d.threshold_density > d.mean_density        # the 0.99 quantile of 1/area sits well above n/A
    # CSR data never leave the Monte Carlo envelope, so the first-crossing rule finds nothing to cross
    assert d.threshold_rule == "quantile fallback"


# =========================================================================== (b) planted clusters
@pytest.mark.parametrize("mode", ["factor", "csr"])
def test_planted_ring_clusters_recovered(mode: str) -> None:
    xy, truth = ring_dataset(seed=1)
    strat = ct.VoronoiDensityStrategy(threshold_mode=mode, csr_iterations=20, rng_seed=0)
    labels = strat.fit(xy)
    assert recovered_fraction(labels, truth) >= 0.9
    assert float((labels[truth == -1] >= 0).mean()) <= 0.1      # little noise swept in
    assert strat.diagnostics is not None and strat.diagnostics.n_clusters == labels.max() + 1


def test_first_rank_density_also_recovers() -> None:
    xy, truth = ring_dataset(seed=2)
    strat = ct.VoronoiDensityStrategy(threshold_mode="factor", density_rank=1)
    assert recovered_fraction(strat.fit(xy), truth) >= 0.9


# =========================================================================== (c) border cells
def test_border_cells_are_finite() -> None:
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.0, 1000.0, (800, 2))
    pts[:10, 0] = 0.0                      # points on the window edge
    pts[10:20] = pts[:10]                  # coincident points
    strat = ct.VoronoiDensityStrategy(threshold_mode="factor", roi_polygon=SQUARE)
    labels = strat.fit(pts)
    d = strat.diagnostics
    assert d is not None
    assert np.isfinite(d.density).all() and np.isfinite(d.cell_area).all()
    assert labels.shape == (800,)
    # the default window (expanded hull) also gives finite values
    strat2 = ct.VoronoiDensityStrategy(threshold_mode="factor")
    strat2.fit(pts)
    assert strat2.diagnostics is not None and np.isfinite(strat2.diagnostics.density).all()


def test_points_outside_roi_are_noise_not_infinite() -> None:
    rng = np.random.default_rng(8)
    pts = rng.uniform(0.0, 1000.0, (400, 2))
    outside = np.array([[1500.0, 1500.0], [1500.1, 1500.1], [1500.2, 1500.0], [1500.0, 1500.2], [1500.1, 1500.0]])
    strat = ct.VoronoiDensityStrategy(threshold_mode="factor", roi_polygon=SQUARE)
    labels = strat.fit(np.vstack([pts, outside]))
    d = strat.diagnostics
    assert d is not None and np.isfinite(d.density).all()
    assert (d.density[-5:] == 0.0).all() and (labels[-5:] == -1).all()


# =========================================================================== (d) interface
def test_interface_labels_params_and_determinism() -> None:
    xy, _ = ring_dataset(seed=4, n_clusters=8, n_noise=200)
    strat = ct.VoronoiDensityStrategy(threshold_mode="csr", csr_iterations=10, rng_seed=12, roi_polygon=SQUARE)
    a = strat.fit(xy)
    b = ct.VoronoiDensityStrategy(threshold_mode="csr", csr_iterations=10, rng_seed=12, roi_polygon=SQUARE).fit(xy)
    assert a.shape == (xy.shape[0],) and a.dtype.kind == "i"
    assert (a == b).all()
    assert a.min() == -1 and a.max() >= 0
    assert set(np.unique(a[a >= 0]).tolist()) == set(range(int(a.max()) + 1))     # labels 0..k-1
    params = strat.get_params()
    assert json.loads(json.dumps(params))["threshold_mode"] == "csr"
    assert "SR-Tesseler" in ct.VoronoiDensityStrategy().get_strategy_name()
    assert "CSR" in strat.get_strategy_name()
    xyz = np.column_stack([xy, np.zeros(xy.shape[0])])
    assert (ct.VoronoiDensityStrategy(threshold_mode="factor").fit(xyz)
            == ct.VoronoiDensityStrategy(threshold_mode="factor").fit(xy)).all()   # z is ignored


def test_min_cluster_points_and_small_inputs() -> None:
    xy, truth = ring_dataset(seed=5, n_clusters=6, n_noise=100)
    loose = ct.VoronoiDensityStrategy(threshold_mode="factor", min_cluster_points=1).fit(xy)
    strict = ct.VoronoiDensityStrategy(threshold_mode="factor", min_cluster_points=15).fit(xy)
    assert (loose >= 0).sum() >= (strict >= 0).sum()
    assert strict.max() + 1 <= 6
    assert (ct.VoronoiDensityStrategy().fit(np.zeros((3, 2))) == -1).all()
    with pytest.raises(ValueError):
        ct.VoronoiDensityStrategy(threshold_mode="other")
    with pytest.raises(ValueError):
        ct.VoronoiDensityStrategy().fit(np.zeros(10))


def test_family_registry() -> None:
    fam = ct.tessellation_family()
    assert set(fam) == {"voronoi_factor", "voronoi_csr"}
    for name, entry in fam.items():
        assert entry["cls"] is ct.VoronoiDensityStrategy
        assert set(entry["defaults"]) - {"threshold_mode"} == set(entry["origins"])
        strat = entry["cls"](**entry["defaults"])
        assert strat.get_params()["threshold_mode"] == entry["defaults"]["threshold_mode"]
    assert fam["voronoi_factor"]["defaults"]["density_factor"] == 2.0
    assert "Levet" in fam["voronoi_factor"]["origins"]["density_factor"]


# =========================================================================== review fixes (REVIEW_D0 A2, A3, M1)
def test_coincident_points_share_their_cell_and_label() -> None:
    """A2: Qhull keeps one copy of a repeated point; the other copy used to be an isolated noise point."""
    rng = np.random.default_rng(21)
    cluster = np.column_stack([500.0 + rng.normal(0, 12.0, 40), 500.0 + rng.normal(0, 12.0, 40)])
    noise = rng.uniform(0.0, 1000.0, (300, 2))
    xy = np.vstack([cluster, cluster[:10], noise])          # 10 exact copies of cluster points
    for kw in (dict(threshold_mode="factor"), dict(threshold_mode="factor", density_rank=0),
               dict(threshold_mode="csr", csr_iterations=10)):
        strat = ct.VoronoiDensityStrategy(roi_polygon=SQUARE, **kw)
        labels = strat.fit(xy)
        np.testing.assert_array_equal(labels[40:50], labels[:10])
        assert (labels[:40] >= 0).mean() >= 0.5
        assert strat.diagnostics is not None and strat.diagnostics.n_duplicates == 10
        np.testing.assert_array_equal(strat.diagnostics.density[40:50], strat.diagnostics.density[:10])


def test_mean_density_refers_to_the_occupied_band_not_the_hull() -> None:
    """A3: on a ring of radius 2 um the convex hull includes the empty axoplasm; the mean density of the
    occupied region is several times higher than n / hull area and is close to the density of the band."""
    rng = np.random.default_rng(22)
    th = rng.uniform(0, 2 * np.pi, 4000)
    r = 2000.0 + rng.uniform(-150.0, 150.0, th.size)
    xy = np.column_stack([r * np.cos(th), r * np.sin(th)])
    occ = ct.VoronoiDensityStrategy(threshold_mode="factor")
    occ.fit(xy)
    win = ct.VoronoiDensityStrategy(threshold_mode="factor", reference_region="window")
    win.fit(xy)
    assert occ.diagnostics is not None and win.diagnostics is not None
    band = xy.shape[0] / (np.pi * (2150.0 ** 2 - 1850.0 ** 2))
    assert occ.diagnostics.mean_density > 2.0 * win.diagnostics.mean_density
    assert 0.4 * band < occ.diagnostics.mean_density <= 1.05 * band
    # uniform band: with the hull the factor-2 rule took 74 % of the band as "clusters"; with the occupied
    # region 18 % (the dilated 100-nm grid is ~1.7x the 300-nm band, so the mean is ~0.6x the band density;
    # simulation (project, scratchpad/phaseD/fix_algo/t_tess4.py), 2026-10-16). Still not zero: honest limit.
    sel_occ = float((occ.fit(xy) >= 0).mean())
    sel_win = float((win.fit(xy) >= 0).mean())
    assert sel_occ < 0.3 and sel_occ < 0.5 * sel_win


def test_default_window_is_convex_and_holds_every_point() -> None:
    """M1: pushing hull vertices radially left reflex vertices on elongated clouds."""
    rng = np.random.default_rng(23)
    for _ in range(20):
        xy = np.column_stack([rng.uniform(0, 3000.0, 300), rng.uniform(0, 150.0, 300)])
        ang = rng.uniform(0, np.pi)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        xy = xy @ rot.T
        w = ct.VoronoiDensityStrategy()._window(xy)
        assert ct.points_in_convex_polygon(xy, w).all()
        hull = ct.convex_window(w)
        assert hull.shape[0] == w.shape[0]                  # every vertex is a hull vertex: convex
