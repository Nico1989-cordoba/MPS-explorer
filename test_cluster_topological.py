# -*- coding: utf-8 -*-
"""
Tests of the graph / topological clustering family (``tools.cluster_topological``): ToMATo (Chazal et al., 2013;
Pike et al., 2020) and the StormGraph-inspired ``GraphCommunityStrategy`` (NOT the published StormGraph), on
SIMULATED tables only (R8).

- Ring of 20 Gaussian clouds (sigma 12 nm, 15-30 points each, 126 nm apart on a 400-nm ring) over uniform noise
  (300 points in 1 x 1 um): ToMATo with tau from the largest gap of the persistence diagram recovers >= 90 % of the
  clouds one-to-one (the majority label of a cloud covers >= 80 % of its points and is the majority label of no
  other cloud); ``GraphCommunityStrategy`` with the scanned radius recovers >= 85 % and sends >= 85 % of the noise
  to -1.
- Two clouds 150 nm apart stay two clusters; one cloud stays one.
- Pure uniform noise with a density floor gives 0 or 1 ToMATo cluster; the graph strategy clusters at most
  2 * alpha of the noise points (alpha is, by construction, the per-localisation false-positive level).
- Interface: label length, -1 convention, ``get_params`` JSON-serialisable, same labels on a second fit and on a
  second instance (determinism), ``topological_family`` registry, ``suggest_tau`` on a hand-made diagram.

Run: ``py -3 -m pytest test_cluster_topological.py -q`` (seconds).
"""
from __future__ import annotations

import json
from typing import Any, Callable, List, Tuple

import numpy as np
import pytest

from tools import cluster_topological as ct
from tools.clustering_strategies import ClusteringStrategy

SEED = 7
SIGMA_NM = 12.0
BOX_NM = 1000.0
N_NOISE = 300
N_CLOUDS = 20
RING_RADIUS_NM = 400.0
NOISE_DENSITY = N_NOISE / BOX_NM ** 2          # points per nm^2 of the uniform background (known by design)


def ring_table(seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian clouds on a ring plus uniform noise; truth is the cloud index or -1."""
    rng = np.random.default_rng(seed)
    pts = []
    truth = []
    for c, ang in enumerate(np.linspace(0.0, 2.0 * np.pi, N_CLOUDS, endpoint=False)):
        m = int(rng.integers(15, 31))
        centre = BOX_NM / 2.0 + RING_RADIUS_NM * np.array([np.cos(ang), np.sin(ang)])
        pts.append(centre + rng.normal(0.0, SIGMA_NM, (m, 2)))
        truth.append(np.full(m, c))
    pts.append(rng.uniform(0.0, BOX_NM, (N_NOISE, 2)))
    truth.append(np.full(N_NOISE, -1))
    return np.vstack(pts), np.concatenate(truth)


def recovered_fraction(labels: np.ndarray, truth: np.ndarray, coverage: float = 0.8) -> float:
    """Fraction of true clouds whose majority label covers >= ``coverage`` of the cloud and is shared by no other."""
    clouds = np.unique(truth[truth >= 0])
    majority = {}
    for c in clouds:
        lab = labels[truth == c]
        kept = lab[lab >= 0]
        if kept.size == 0:
            continue
        maj = int(np.bincount(kept).argmax())
        if np.mean(lab == maj) >= coverage:
            majority[int(c)] = maj
    counts = np.bincount(np.array(list(majority.values()), dtype=np.int64)) if majority else np.zeros(0)
    ok = sum(1 for c, maj in majority.items() if counts[maj] == 1)
    return ok / float(clouds.size)


# -- ToMATo ----------------------------------------------------------------------------------------------------

KNN_TOMATO = dict(k_graph=10, radius_nm=None, density="knn", bandwidth_nm=None)   # the pre-2026-10-16 default


def test_tomato_default_recovers_ring_clouds_and_leaves_noise_out() -> None:
    """Radius graph 20 nm + KDE (default): the clouds are recovered and most noise cannot be linked (-1); a few
    noise triples closer than 20 nm form small extra clusters (26 clusters for 20 clouds here)."""
    X, T = ring_table()
    s = ct.ToMAToStrategy()
    labels = s.fit(X)
    assert recovered_fraction(labels, T) >= 0.9
    assert np.mean(labels[T < 0] == -1) >= 0.8
    sizes = np.bincount(labels[labels >= 0])
    assert abs(int((sizes >= 10).sum()) - N_CLOUDS) <= 2


def test_tomato_recovers_ring_clouds_with_gap_tau() -> None:
    X, T = ring_table()
    s = ct.ToMAToStrategy(**KNN_TOMATO)
    labels = s.fit(X)
    assert labels.shape == (X.shape[0],)
    assert s.tau_used_ is not None and s.tau_used_ > 0
    assert recovered_fraction(labels, T) >= 0.9
    assert abs(s.n_clusters_ - N_CLOUDS) <= 2
    # The diagram shows the gap: the 20 cloud peaks are at least 10x more persistent than the next one.
    p = np.sort(s.persistences())[::-1]
    assert p.size > N_CLOUDS
    assert p[N_CLOUDS - 1] > 10.0 * p[N_CLOUDS]


def test_tomato_density_floor_sends_noise_to_minus_one() -> None:
    X, T = ring_table()
    s = ct.ToMAToStrategy(density_floor=2.0 * NOISE_DENSITY)
    labels = s.fit(X)
    assert recovered_fraction(labels, T) >= 0.9
    assert np.mean(labels[T < 0] == -1) > 0.5
    # Every point under the floor is -1.
    assert s.density_values_ is not None
    assert np.all(labels[s.density_values_ < 2.0 * NOISE_DENSITY] == -1)


def test_tomato_kde_density_variant() -> None:
    X, T = ring_table()
    s = ct.ToMAToStrategy(density="kde", bandwidth_nm=15.0)
    labels = s.fit(X)
    assert recovered_fraction(labels, T) >= 0.9


def test_tomato_two_clouds_150nm_apart_stay_two() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, SIGMA_NM, (25, 2))
    b = rng.normal(0.0, SIGMA_NM, (25, 2)) + np.array([150.0, 0.0])
    s = ct.ToMAToStrategy()
    labels = s.fit(np.vstack([a, b]))
    assert s.n_clusters_ == 2
    # radius graph: a straggler farther than 20 nm from the rest of its cloud is noise
    la, lb = labels[:25], labels[25:]
    assert np.unique(la[la >= 0]).size == 1 and np.unique(lb[lb >= 0]).size == 1
    assert np.mean(la >= 0) >= 0.8 and np.mean(lb >= 0) >= 0.8
    assert la[la >= 0][0] != lb[lb >= 0][0]
    knn = ct.ToMAToStrategy(**KNN_TOMATO)
    lk = knn.fit(np.vstack([a, b]))
    assert knn.n_clusters_ == 2 and len(np.unique(lk[:25])) == 1 and len(np.unique(lk[25:])) == 1
    one = ct.ToMAToStrategy()
    one.fit(a)
    assert one.n_clusters_ == 1


def test_tomato_pure_noise_with_floor_gives_at_most_one_cluster() -> None:
    rng = np.random.default_rng(3)
    noise = rng.uniform(0.0, BOX_NM, (N_NOISE, 2))
    s = ct.ToMAToStrategy(density_floor=2.0 * NOISE_DENSITY, **KNN_TOMATO)
    labels = s.fit(noise)
    assert s.n_clusters_ <= 1
    # default (radius graph, no floor): pure noise is mostly unlinkable
    assert np.mean(ct.ToMAToStrategy().fit(noise) == -1) > 0.8
    assert np.mean(labels == -1) > 0.5


def test_suggest_tau_largest_gap() -> None:
    # Three real peaks and a tail of wiggles: tau falls in the gap between 0.02 and 0.001.
    tau = ct.suggest_tau([0.05, 0.03, 0.02, 0.001, 0.0008, 0.0002, np.nan, np.inf, 0.0])
    assert 0.001 < tau < 0.02
    assert ct.suggest_tau([]) == 0.0
    # No tail at all (no ratio >= 2): every peak is kept.
    assert ct.suggest_tau([0.05, 0.04, 0.03]) < 0.03
    assert ct.suggest_tau([0.05]) < 0.05


def test_tomato_default_is_pikes_radius_graph_with_kde() -> None:
    """Defaults verified on 2026-10-16 (VERIF_DEFAULTS_D0 section 5): radius graph of 20 nm, KDE sigma 10, no floor;
    an isolated localisation cannot be linked and is noise (with a k-NN graph it always had neighbours)."""
    s = ct.ToMAToStrategy()
    p = s.get_params()
    assert p["k_graph"] is None and p["radius_nm"] == 20.0 and p["density"] == "kde" and p["bandwidth_nm"] == 10.0
    assert p["density_floor"] is None and s.scale_nm() == 20.0
    rng = np.random.default_rng(11)
    cloud = rng.normal(0.0, SIGMA_NM, (30, 2))
    lonely = np.array([[400.0, 400.0], [-400.0, 300.0]])
    labels = s.fit(np.vstack([cloud, lonely]))
    assert (labels[-2:] == -1).all()
    assert np.mean(labels[:30] == labels[0]) >= 0.9 and labels[0] >= 0
    knn = ct.ToMAToStrategy(k_graph=10, radius_nm=None, density="knn", bandwidth_nm=None)
    assert (knn.fit(np.vstack([cloud, lonely]))[-2:] >= 0).any()      # the k-NN graph links the lonely points


def test_graph_null_is_drawn_in_the_occupied_region() -> None:
    """M3: on a ring the bounding box is mostly empty axoplasm; random clouds drawn there are sparse and the degree
    threshold is too low to prune. The occupied region gives a higher threshold."""
    rng = np.random.default_rng(12)
    th = rng.uniform(0.0, 2.0 * np.pi, 3000)
    r = 2000.0 + rng.uniform(-100.0, 100.0, th.size)
    ring = np.column_stack([r * np.cos(th), r * np.sin(th)])
    occ = ct.GraphCommunityStrategy()
    box = ct.GraphCommunityStrategy(null_region="bbox")
    occ.fit(ring)
    box.fit(ring)
    assert occ.radius_used_ == 25.0 and occ.get_params()["null_region"] == "occupied"
    assert occ.degree_threshold_ is not None and box.degree_threshold_ is not None
    assert occ.degree_threshold_ > 1.3 * box.degree_threshold_          # 1.28 vs 0.82 on this ring (seed 12)


def test_tomato_core_full_diagram_births_are_local_maxima() -> None:
    X, _ = ring_table()
    dens = ct.knn_density(X, 10)
    a, b = ct.neighbourhood_graph(X, 10, None)
    indptr, indices = ct._adjacency_lists(X.shape[0], a, b)
    root, birth, death, peak = ct.tomato_core(dens, indptr, indices, np.inf)
    assert np.all(np.isin(root, peak))
    assert np.allclose(birth, dens[peak])
    assert np.all(np.isnan(death) | (death <= birth))
    # With tau = inf the graph's components are the clusters.
    assert len(np.unique(root)) == len(set(root.tolist()))


# -- GraphCommunity ----------------------------------------------------------------------------------------------

def test_graph_community_recovers_ring_with_scanned_radius() -> None:
    X, T = ring_table()
    g = ct.GraphCommunityStrategy(radius_nm=None)
    labels = g.fit(X)
    assert g.scan_ is not None and g.radius_used_ in g.scan_["radii"]
    assert recovered_fraction(labels, T) >= 0.85
    assert np.mean(labels[T < 0] == -1) >= 0.85
    # Compare against the same radius fixed by hand.
    fixed = ct.GraphCommunityStrategy(radius_nm=g.radius_used_)
    assert np.array_equal(fixed.fit(X), labels)


def test_graph_community_pure_noise_clusters_at_most_two_alpha() -> None:
    rng = np.random.default_rng(5)
    noise = rng.uniform(0.0, BOX_NM, (N_NOISE, 2))
    g = ct.GraphCommunityStrategy(radius_nm=30.0, alpha=0.05)
    labels = g.fit(noise)
    assert np.mean(labels >= 0) <= 2.0 * 0.05


def test_graph_community_prune_quantile_alternative() -> None:
    X, T = ring_table()
    # 40 % of the table is noise by design; the quantile below that leaves a little noise that 25 nm cannot bridge.
    g = ct.GraphCommunityStrategy(radius_nm=25.0, prune_quantile=0.35)
    labels = g.fit(X)
    assert recovered_fraction(labels, T) >= 0.85


# -- Interface ---------------------------------------------------------------------------------------------------

def _makers() -> List[Callable[[], Any]]:
    return [
        lambda: ct.ToMAToStrategy(),
        lambda: ct.ToMAToStrategy(k_graph=None, radius_nm=40.0, density="kde"),
        lambda: ct.GraphCommunityStrategy(),
        lambda: ct.ToMAToStrategy(k_graph=10, radius_nm=None, density="knn", bandwidth_nm=None),
    ]


def test_interface_contract() -> None:
    X, _ = ring_table()
    for make in _makers():
        _check_contract(make, X)


def _check_contract(make: Callable[[], Any], X: np.ndarray) -> None:
    s = make()
    labels = s.fit(X)
    assert isinstance(labels, np.ndarray) and labels.shape == (X.shape[0],)
    assert labels.dtype.kind == "i"
    assert labels.min() >= -1
    k = labels.max() + 1
    assert set(np.unique(labels[labels >= 0]).tolist()) == set(range(k))
    assert isinstance(s.get_strategy_name(), str) and s.get_strategy_name()
    params = s.get_params()
    json.dumps(params)
    # Determinism: same instance again, and a fresh instance with the same parameters.
    assert np.array_equal(s.fit(X), labels)
    assert np.array_equal(make().fit(X), labels)
    # Empty input.
    empty = s.fit(np.zeros((0, 2)))
    assert empty.shape == (0,)


def test_rejects_bad_shapes_and_options() -> None:
    with pytest.raises(ValueError):
        ct.ToMAToStrategy(density="voronoi")
    with pytest.raises(ValueError):
        ct.ToMAToStrategy(k_graph=None, radius_nm=None)
    with pytest.raises(ValueError):
        ct.GraphCommunityStrategy(null_region="hull")
    with pytest.raises(ValueError):
        ct.ToMAToStrategy().fit(np.zeros((5, 4)))


def test_family_registry_lists_both_with_origins() -> None:
    fam = ct.topological_family()
    assert set(fam) == {"ToMATo", "GraphCommunity"}
    for name, entry in fam.items():
        assert issubclass(entry["cls"], ClusteringStrategy)
        assert entry["description"]
        assert set(entry["defaults"]) == set(entry["origins"])
        json.dumps(entry["defaults"])
    assert "NOT" in ct.GraphCommunityStrategy.__doc__.upper()
    assert "StormGraph" in fam["GraphCommunity"]["description"]
