# -*- coding: utf-8 -*-
"""
Edge-curation margin of Phase D (VERIF_DEFAULTS_D0 section 9, FIX_ALGO step 5), on SIMULATED tables only (R8).

``scale_nm`` is the method's own length scale and is informative; the margin of ``identify_bad_clusters`` is
``ClusteringStrategy.edge_margin_nm(compat_eps)``: eps for DBSCAN (the compatibility path, unchanged), and
max(compat eps, the method's separation scale) for the others -- never an intra-cluster scale (median core
distance, Voronoi cell spacing) that would curate less than DBSCAN 25 does.

Run: ``py -3 -m pytest test_phase_d_edge_margin.py -q`` (seconds).
"""
from __future__ import annotations

import numpy as np

from tools.cluster_density import FOCALStrategy, OPTICSStrategy, SklearnHDBSCANStrategy
from tools.cluster_tessellation import VoronoiDensityStrategy
from tools.cluster_topological import GraphCommunityStrategy, ToMAToStrategy
from tools.clustering_strategies import DBSCANStrategy, core_distance_max, core_distance_scale


def _ring(seed: int = 0, n_clusters: int = 24, radius_nm: float = 400.0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    xs, ys = [], []
    for a in ang:
        n = int(rng.integers(15, 40))
        xs.append(radius_nm * np.cos(a) + rng.normal(0, 9, n))
        ys.append(radius_nm * np.sin(a) + rng.normal(0, 9, n))
    xs.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, 150))
    ys.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, 150))
    return np.concatenate(xs) + 2000.0, np.concatenate(ys) + 2000.0


def test_dbscan_margin_is_eps_whatever_the_fallback() -> None:
    s = DBSCANStrategy(eps=25.0, min_samples=10)
    s.fit(np.column_stack(_ring()))
    assert s.edge_margin_nm(31.0) == 25.0 and s.scale_nm() == 25.0


def test_hierarchical_margin_is_the_largest_core_distance_not_the_median() -> None:
    xy = np.column_stack(_ring())
    h = SklearnHDBSCANStrategy(min_samples=5, min_cluster_size=10)
    lab = h.fit(xy)
    med = core_distance_scale(xy, lab, 5)
    mx = core_distance_max(xy, lab, 5)
    assert med is not None and mx is not None and mx >= med
    assert h.scale_nm() == med                                   # informative, intra-cluster
    assert h.edge_margin_nm(25.0) == max(25.0, mx)
    o = OPTICSStrategy(min_samples=10, max_eps=100.0)
    o.fit(xy)
    assert o.edge_margin_nm(25.0) >= 25.0


def test_other_methods_never_curate_less_than_eps() -> None:
    xy = np.column_stack(_ring())
    for s in (FOCALStrategy(), VoronoiDensityStrategy(), ToMAToStrategy(), GraphCommunityStrategy()):
        s.fit(xy)
        sc = s.scale_nm()
        m = s.edge_margin_nm(25.0)
        assert m >= 25.0
        if sc is not None:
            assert m == max(25.0, sc)
    v = VoronoiDensityStrategy()
    v.fit(xy)
    assert v.scale_nm() is not None and v.scale_nm() < 25.0 and v.edge_margin_nm(25.0) == 25.0


def test_analysis_records_scale_and_margin_separately() -> None:
    from tools.mps_analysis import analyze_axon
    x, y = _ring()
    z = np.zeros_like(x) + 300.0
    a = analyze_axon(x, y, z, run_randomization=False)
    assert a.cluster_scale_nm == 25.0 and a.cluster_edge_margin_nm == 25.0
    b = analyze_axon(x, y, z, run_randomization=False, cluster_method="FOCAL")
    assert b.cluster_scale_nm == 15.0 and b.cluster_edge_margin_nm == 25.0
