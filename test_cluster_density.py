# -*- coding: utf-8 -*-
"""
Tests of the density family of clustering strategies in
``tools.cluster_density`` (OPTICS, FOCAL, scikit-learn HDBSCAN).

Synthetic data only (rule R8): Gaussian clouds of sigma 12 nm with 10-30
localizations each, planted on a ring of 1 um perimeter, plus uniform
background noise over the field; fixed seeds. Each strategy must recover
at least 90 % of the planted clusters with reasonable parameters, report
no (or almost no) clusters on pure noise, and honour the common interface
(labels of length n, -1 for noise, JSON-serialisable ``get_params``).

Run: ``py -3 -m pytest test_cluster_density.py -q``.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pytest

from tools.cluster_density import (
    FOCALStrategy,
    OPTICSStrategy,
    SklearnHDBSCANStrategy,
    density_family,
)
from tools.clustering_strategies import ClusteringStrategy

PERIMETER_NM = 1000.0
SIGMA_NM = 12.0
N_CLUSTERS = 10
FIELD_NM = 600.0  # square field centred on the ring, for the noise
N_NOISE = 60


def ring_with_clusters(seed: int, n_clusters: int = N_CLUSTERS, n_noise: int = N_NOISE) -> Tuple[np.ndarray, np.ndarray]:
    """Planted clusters on a ring of 1 um perimeter plus uniform noise.

    Returns (xy in nm, truth) with truth = planted cluster index, -1 for noise.
    """
    rng = np.random.default_rng(seed)
    radius = PERIMETER_NM / (2.0 * math.pi)
    theta = np.linspace(0.0, 2.0 * math.pi, n_clusters, endpoint=False) + rng.uniform(-0.05, 0.05, n_clusters)
    centres = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    pts: List[np.ndarray] = []
    truth: List[np.ndarray] = []
    for k, c in enumerate(centres):
        m = int(rng.integers(10, 31))
        pts.append(c + rng.normal(0.0, SIGMA_NM, size=(m, 2)))
        truth.append(np.full(m, k, dtype=np.int64))
    if n_noise > 0:
        pts.append(rng.uniform(-FIELD_NM / 2, FIELD_NM / 2, size=(n_noise, 2)))
        truth.append(np.full(n_noise, -1, dtype=np.int64))
    xy = np.vstack(pts)
    tr = np.concatenate(truth)
    perm = rng.permutation(xy.shape[0])
    return xy[perm], tr[perm]


def pure_noise(seed: int, n: int = N_NOISE) -> np.ndarray:
    """Uniform noise at the same density as the background of ``ring_with_clusters``."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-FIELD_NM / 2, FIELD_NM / 2, size=(n, 2))


def recovered_fraction(labels: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of planted clusters whose majority label is a cluster (not -1), covers >= 50 % of the
    planted points, and is not claimed by another planted cluster (one-to-one)."""
    planted = sorted(set(truth[truth >= 0].tolist()))
    claimed: Dict[int, int] = {}
    hits = 0
    for k in planted:
        sel = labels[truth == k]
        vals, cnt = np.unique(sel[sel >= 0], return_counts=True)
        if vals.size == 0:
            continue
        best = int(vals[np.argmax(cnt)])
        if cnt.max() / sel.size < 0.5:
            continue
        if best in claimed:
            continue
        claimed[best] = k
        hits += 1
    return hits / len(planted)


def n_clusters_of(labels: np.ndarray) -> int:
    return int(len(set(labels.tolist()) - {-1}))


def strategies() -> Dict[str, ClusteringStrategy]:
    """Reasonable parameters for the planted data (sigma 12 nm, 10-30 points, gap ~30 nm between clouds).

    OPTICS-xi needs ``min_samples`` of the order of the cluster size (10, the project's value): with 5 the
    reachability plot of a 12-nm cloud has sub-steps steeper than xi and the clouds fragment (scan of
    2026-10-16: min_samples 5 recovered 0.2-0.7, 8 recovered 0.8-1.0, 10 recovered 1.0 on three seeds).
    """
    return {
        "optics_xi": OPTICSStrategy(min_samples=10, xi=0.05, max_eps=40.0),
        "optics_dbscan": OPTICSStrategy(min_samples=5, max_eps=40.0, cluster_method="dbscan", eps=25.0),
        # FOCAL at its registry defaults (15 nm / min_l 6 on the 3x3-enhanced map / min_c 3).
        "focal": FOCALStrategy(),
        # min_samples = min_cluster_size (the library default): with min_samples 5, 60 uniform
        # points were once split into two "clusters" covering 68 % of them (seed 7).
        "hdbscan_sklearn": SklearnHDBSCANStrategy(min_cluster_size=8, min_samples=8),
    }


@pytest.mark.parametrize("name", sorted(strategies().keys()))
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recovers_planted_clusters(name: str, seed: int) -> None:
    xy, truth = ring_with_clusters(seed)
    strat = strategies()[name]
    labels = strat.fit(xy)
    frac = recovered_fraction(labels, truth)
    assert frac >= 0.9, f"{name} seed {seed}: recovered {frac:.2f} of {N_CLUSTERS} clusters"
    # Not fragmented into many spurious clusters either
    assert n_clusters_of(labels) <= N_CLUSTERS + 3


@pytest.mark.parametrize("name", sorted(strategies().keys()))
def test_pure_noise_gives_no_or_few_clusters(name: str) -> None:
    # HDBSCAN always looks for the most stable split, so on uniform data it
    # may still return one or two small clusters; the others return none.
    for seed in (7, 8, 9):
        labels = strategies()[name].fit(pure_noise(seed))
        assert n_clusters_of(labels) <= 2, (name, seed)
        assert np.mean(labels == -1) >= 0.6, (name, seed)


@pytest.mark.parametrize("name", sorted(strategies().keys()))
def test_interface(name: str) -> None:
    xy, _ = ring_with_clusters(11)
    strat = strategies()[name]
    assert isinstance(strat, ClusteringStrategy)
    labels = strat.fit(xy)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (xy.shape[0],)
    assert np.issubdtype(labels.dtype, np.integer)
    assert labels.min() >= -1
    k = n_clusters_of(labels)
    assert set(labels[labels >= 0].tolist()) == set(range(k)), "cluster labels are 0..k-1"
    assert isinstance(strat.get_strategy_name(), str) and strat.get_strategy_name()
    params = strat.get_params()
    assert isinstance(params, dict)
    json.dumps(params, allow_nan=False)  # serialisable, no inf/nan
    # 3-D input (x, y, z) is accepted too
    xyz = np.column_stack([xy, np.zeros(xy.shape[0])])
    assert strat.fit(xyz).shape == (xy.shape[0],)


@pytest.mark.parametrize("name", sorted(strategies().keys()))
def test_empty_and_tiny_inputs(name: str) -> None:
    strat = strategies()[name]
    assert strat.fit(np.zeros((0, 2))).shape == (0,)
    tiny = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = strat.fit(tiny)
    assert labels.shape == (2,)
    assert np.all(labels == -1)
    with pytest.raises(ValueError):
        strat.fit(np.zeros((5, 4)))


def test_optics_reachability_plot() -> None:
    xy, _ = ring_with_clusters(5)
    strat = OPTICSStrategy(min_samples=5, max_eps=40.0)
    with pytest.raises(RuntimeError):
        strat.reachability_plot()
    strat.fit(xy)
    order, reach = strat.reachability_plot()
    assert order.shape == (xy.shape[0],) and reach.shape == (xy.shape[0],)
    assert sorted(order.tolist()) == list(range(xy.shape[0]))
    assert np.all(reach[np.isfinite(reach)] >= 0.0)
    assert strat.core_distances_ is not None and strat.predecessor_ is not None


def test_optics_validation() -> None:
    with pytest.raises(ValueError):
        OPTICSStrategy(cluster_method="leaf")
    with pytest.raises(ValueError):
        OPTICSStrategy(xi=1.5)
    assert OPTICSStrategy().get_params()["max_eps"] == 100.0  # bounded default (REVIEW_D0 M5)
    assert OPTICSStrategy(max_eps=math.inf).get_params()["max_eps"] is None  # inf written as None


def test_focal_grid_and_min_c() -> None:
    xy, truth = ring_with_clusters(3)
    strat = FOCALStrategy()
    labels = strat.fit(xy)
    assert strat.cell_index_ is not None and strat.cell_index_.shape == (xy.shape[0], 2)
    assert strat.n_core_cells_ > 0
    # A huge min_c removes every cluster; a huge min_l leaves no core cell
    assert n_clusters_of(FOCALStrategy(20.0, 3, 1000).fit(xy)) == 0
    strict = FOCALStrategy(20.0, 1000, 1)
    assert n_clusters_of(strict.fit(xy)) == 0 and strict.n_core_cells_ == 0
    # Two localizations in one cell with min_l=1, min_c=1 form one cluster
    assert n_clusters_of(FOCALStrategy(50.0, 1, 1).fit(np.array([[0.0, 0.0], [5.0, 5.0]]))) == 1
    # 4-connectivity (Nino et al. 2020: cells that share an edge): diagonal cells stay apart
    diag = np.array([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [25.0, 25.0], [26.0, 26.0], [27.0, 27.0]])
    assert n_clusters_of(FOCALStrategy(20.0, 3, 1).fit(diag)) == 2
    # Noise from the planted data is mostly left as -1 (a noise point in a cell next to a cluster becomes a
    # border cell of it, as in the paper: 0.87 of the noise stays -1 on this seed)
    assert np.mean(labels[truth == -1] == -1) >= 0.8


def test_focal_enhanced_map_rescues_cells_below_min_l() -> None:
    """The case the D0 implementation (raw per-cell count >= min_l) failed: a cluster spread over 3x3 cells of
    2 localizations each. No raw cell reaches min_l = 6, so the old rule returned no cluster; on the enhanced map
    (Nino et al. 2020, Fig 1B) every cell sums >= 8 and the whole cluster is found."""
    g = 15.0
    centres = np.array([[(i + 0.5) * g, (j + 0.5) * g] for i in range(3) for j in range(3)])
    xy = np.repeat(centres, 2, axis=0) + np.tile([[-2.0, 0.0], [2.0, 0.0]], (9, 1))
    labels = FOCALStrategy(g, 6, 3).fit(xy)
    assert n_clusters_of(labels) == 1 and bool(np.all(labels == 0))


def test_focal_border_cells_join_the_cluster() -> None:
    """Border cells (Nino et al. 2020, Fig 1C): non-empty cells below min_l that share an edge with a core cell
    carry the cluster's label instead of being noise."""
    g = 10.0
    # grid origin = (1, 1): left in cell (0, 0), core in (1, 0), right in (2, 0), far in (9, 9)
    core = np.column_stack([np.linspace(11.0, 17.0, 11), np.linspace(1.0, 7.0, 11)])
    left = np.array([[1.0, 4.0]])
    right = np.array([[21.0, 4.0]])
    far = np.array([[91.0, 91.0]])
    xy = np.vstack([core, left, right, far])
    st = FOCALStrategy(g, 13, 3)
    labels = st.fit(xy)
    # core cell: 11 + 1 + 1 = 13 >= 13; each side cell: 1 + 11 = 12 < 13 -> border, edge-adjacent
    assert st.n_core_cells_ == 1 and st.n_border_cells_ == 2
    assert bool(np.all(labels[:13] == 0)) and labels[13] == -1


def test_focal_suggest_min_l_cleans_random_data() -> None:
    xy, _ = ring_with_clusters(9)
    strat = FOCALStrategy()
    m = strat.suggest_min_l(xy, seed=1)
    assert 1 <= m <= 100
    probe = FOCALStrategy(strat.grid_nm, m, strat.min_c)
    # random copies over the occupied region are clean at m (that is the definition) ...
    from tools.clustering_strategies import OccupiedSupport
    sup = OccupiedSupport(xy)
    rng = np.random.default_rng(1)
    assert n_clusters_of(probe.fit(sup.sample(xy.shape[0], rng))) == 0
    # ... and so is a copy over the bounding box, which is sparser (the old reference, REVIEW_D0 M3)
    rnd = rng.uniform(xy.min(axis=0), xy.max(axis=0), size=xy.shape)
    assert n_clusters_of(probe.fit(rnd)) == 0
    # ... while the structured data still has most of its clusters
    assert n_clusters_of(probe.fit(xy)) >= int(0.8 * N_CLUSTERS)


def test_occupied_support_excludes_the_empty_interior_of_a_ring() -> None:
    """The reference region of the nulls (REVIEW_D0 A3 / M3): on a ring of radius 1.5 um the occupied region is a
    band, several times smaller than the bounding box, and contains every localization."""
    from tools.clustering_strategies import OccupiedSupport
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 3000)
    r = 1500.0 + rng.normal(0, 20.0, th.size)
    xy = np.column_stack([r * np.cos(th), r * np.sin(th)])
    sup = OccupiedSupport(xy)
    box = float(np.prod(xy.max(axis=0) - xy.min(axis=0)))
    assert sup.area_nm2 < box / 2
    assert bool(sup.contains(xy).all())
    assert not bool(sup.contains(np.array([[0.0, 0.0]]))[0])
    pts = sup.sample(500, np.random.default_rng(1))
    assert bool(sup.contains(pts).all())


def test_density_family_registry() -> None:
    fam = density_family()
    assert set(fam) == {"OPTICS", "FOCAL", "HDBSCAN (scikit-learn)"}
    for name, entry in fam.items():
        assert issubclass(entry["cls"], ClusteringStrategy)
        assert entry["description"] and entry["citation"]
        for p, d in entry["defaults"].items():
            assert set(d) == {"value", "origin"}, (name, p)
            assert d["origin"].split(":")[0] in ("literature", "derivation", "simulation", "ad hoc"), (name, p)
        # The defaults instantiate (None values mean "library default")
        kwargs = {p: d["value"] for p, d in entry["defaults"].items() if d["value"] is not None}
        inst = entry["cls"](**kwargs)
        assert inst.get_strategy_name() == name
    json.dumps({k: {p: d for p, d in v["defaults"].items()} for k, v in fam.items()})
