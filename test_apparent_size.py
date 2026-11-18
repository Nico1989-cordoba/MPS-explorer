# -*- coding: utf-8 -*-
"""
Headless tests of the apparent-size analysis (clustering plan A6 + the apparent-size simulation's recommendations;
docs/apparent_size.md): tools/mps_apparent_size.py, tools/mps_curvature.py, tools/mps_size_stats.py,
tools/size_plugins.py, with the generator tools/mps_size_synth.py.

Synthetic data only (R8): clusters, rings and contours drawn here, with a known truth.

  1. estimators independent of N: the SD of the cloud and the hull at fixed N = 15 do not grow with the
     localizations of the same cloud; the convex hull (the legacy "Cluster area") does; on a population with one
     label per cluster the hull ranks by N_loc and the cloud SD / r_rare do not;
  2. the r_rare flag at N_loc < 15; the rarefaction is deterministic and per cluster;
  3. the footprint pool: recovered with >= 2 labels and the right NeNA, "content only" with one label, "not
     evaluated" without events;
  4. curvature: exact on a circle, kappa R recovered at 400 nm knots on a thick ellipse, concave stretches found on
     a notched contour, "curvature not resolved" below R = 600 nm, sign independent of the curve's orientation;
  5. question (b): the concave clusters are excluded from the primary readout and reported apart;
  6. the mark-permutation diagnostic shows its documented false positives under a null with DBSCAN (and none with
     the true clusters);
  7. across axons: Fisher z combination, the mixed model's level under the null, the covariates by estimator kind;
  8. the plug-ins on real AxonAnalysis runs of two methods (multimethod), the eps sensitivity, the pipeline-null
     hook, the per-cluster export.

Run:  py -3 -m pytest test_apparent_size.py -q      (about a minute)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import mps_apparent_size as asz  # noqa: E402
from tools import mps_curvature as curv  # noqa: E402
from tools import mps_size_stats as st  # noqa: E402
from tools import mps_size_synth as syn  # noqa: E402
from tools import size_plugins as sp  # noqa: E402


def _gauss_cloud(rng: np.random.Generator, n: int, sigma: float = 10.0) -> np.ndarray:
    return rng.normal(0.0, sigma, (n, 2))


# --------------------------------------------------------------------------------------- 1. independent of N
def test_cloud_sd_and_fixed_n_hull_do_not_grow_with_n_but_the_hull_does():
    """The colleagues' point (SIZE_SIM sec. 3.3): the same Gaussian cloud (sigma 10 nm) sampled with 20 and with 200
    localizations. The cloud SD and the 15-point hull stay within 5 %; the full hull grows > 2x in area."""
    rng = np.random.default_rng(20261022)
    sd = {20: [], 200: []}
    rr = {20: [], 200: []}
    hull = {20: [], 200: []}
    for n in (20, 200):
        for i in range(300):
            pts = _gauss_cloud(rng, n)
            lab = np.zeros(n, dtype=np.int64)
            cs = asz.cluster_sizes(pts[:, 0], pts[:, 1], lab, seed=i)
            sd[n].append(cs.cloud_sd[0])
            rr[n].append(cs.hull_area_n15[0])
            hull[n].append(asz.hull_area(pts))
    m = {k: {n: float(np.median(v[n])) for n in (20, 200)} for k, v in (("sd", sd), ("rr", rr), ("hull", hull))}
    assert abs(m["sd"][200] / m["sd"][20] - 1.0) < 0.05
    assert abs(m["rr"][200] / m["rr"][20] - 1.0) < 0.05
    assert m["hull"][200] / m["hull"][20] > 2.0
    assert 9.0 < m["sd"][200] < 11.0                                      # it IS the precision with one label


def test_population_with_one_label_hull_ranks_by_n_and_the_new_estimators_do_not():
    """SIZE_SIM sec. 3.1, one label: Spearman(r_hull, log N_loc) ~ 0.8, r_rare ~ 0; the cloud SD stays well below
    the hull."""
    rng = np.random.default_rng(7)
    xs, ys, labs = [], [], []
    for c in range(400):
        d = syn.draw_cluster(rng, 25.0, 1)
        if d.t.size < 3:
            continue
        xs.append(d.t + 1000.0 * c)
        ys.append(d.n)
        labs.append(np.full(d.t.size, c))
    x, y, lab = np.concatenate(xs), np.concatenate(ys), np.concatenate(labs)
    cs = asz.cluster_sizes(x, y, lab)
    full_hull = np.array([asz.hull_area(np.column_stack([x[lab == c], y[lab == c]])) for c in cs.labels])
    rh, _ = asz.spearman(full_hull, cs.n_locs)
    rsd, _ = asz.spearman(cs.cloud_sd, cs.n_locs)
    rrare, _ = asz.spearman(cs.r_rare, cs.n_locs)
    assert rh > 0.6
    assert abs(rsd) < 0.3 and abs(rrare) < 0.2
    assert rh - abs(rsd) > 0.4


# --------------------------------------------------------------------------------------- 2. r_rare flag
def test_r_rare_undefined_below_15_and_deterministic_per_cluster():
    rng = np.random.default_rng(3)
    a, b, c = _gauss_cloud(rng, 14), _gauss_cloud(rng, 15) + 500, _gauss_cloud(rng, 60) + 1000
    pts = np.vstack([a, b, c])
    lab = np.repeat([0, 1, 2], [14, 15, 60]).astype(np.int64)
    cs = asz.cluster_sizes(pts[:, 0], pts[:, 1], lab, seed=5)
    assert list(cs.r_rare_undefined) == [True, False, False]
    assert math.isnan(cs.r_rare[0]) and math.isnan(cs.hull_area_n15[0])
    assert np.isfinite(cs.r_rare[1:]).all()
    assert cs.flag_few_locs.tolist() == [True, True, False]
    rows = sp.cluster_rows(SimpleNamespace(sizes=cs, membrane=None, arc_nm=np.full(3, np.nan),
                                           kappa_per_nm=np.full(3, np.nan), kappa_r=np.full(3, np.nan)))
    assert rows[0]["size_r_rare_undefined"] is True and rows[0]["size_r_rare_nm"] is None
    # same seed -> same value; dropping another cluster does not change this one's
    again = asz.cluster_sizes(pts[:, 0], pts[:, 1], lab, seed=5)
    only = asz.cluster_sizes(pts[:, 0], pts[:, 1], lab, seed=5, exclude=[1])
    assert again.r_rare[2] == cs.r_rare[2] and only.r_rare[-1] == cs.r_rare[2]
    assert asz.cluster_sizes(pts[:, 0], pts[:, 1], lab, seed=6).r_rare[2] != cs.r_rare[2]


def test_exclude_noise_and_flags():
    rng = np.random.default_rng(4)
    pts = np.vstack([_gauss_cloud(rng, 30), _gauss_cloud(rng, 30) + [40.0, 0.0], _gauss_cloud(rng, 30) + [900, 0]])
    lab = np.repeat([0, 1, 2], 30).astype(np.int64)
    lab[:3] = -1
    cs = asz.cluster_sizes(pts[:, 0], pts[:, 1], lab, exclude=[2])
    assert cs.labels.tolist() == [0, 1] and cs.n_locs.tolist() == [27, 30]
    assert cs.flag_close.all()                                         # 40 nm apart
    assert cs.frame == "none" and np.isnan(cs.elongation).all()        # no curve, no tangential frame
    # an elongated cloud along a horizontal membrane is flagged
    e = np.column_stack([rng.normal(0, 30, 80), rng.normal(0, 8, 80)])
    curve = np.column_stack([np.linspace(-5000, 5000, 400), np.zeros(400)])
    curve = np.vstack([curve, curve[::-1] + [0, -3000]])
    ce = asz.cluster_sizes(e[:, 0], e[:, 1], np.zeros(80, dtype=np.int64), curve=curve)
    assert ce.frame == "membrane" and ce.elongation[0] > 1.8 and ce.flag_elongated[0]


# --------------------------------------------------------------------------------------- 3. footprint pool
def _rings(seed: int, labels_mean: float, n: int = 6, rho: float = 25.0, R: float = 1400.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = syn.make_contour("circle", R, rng)
        out.append(syn.generate_ring(rng, c, labels_mean=labels_mean, rho=rho, K=60))
    return out


def _pool_of(ring: syn.SyntheticRing, nena: float, **kw: Any) -> asz.FootprintPool:
    cs = asz.cluster_sizes(ring.x, ring.y, ring.truth, event_id=ring.event, curve=ring.contour.xy)
    return asz.pooled_footprint(cs.sd_tangential, cs.n_locs, cs.n_events, nena, **kw)


def test_pool_recovers_the_footprint_with_three_labels_and_the_right_nena():
    """P-R61 sec. 3.2 / SIZE_SIM sec. 3.2: 3 labels, rho 25 nm, K = 60, NeNA exact -> 25.9 +/- 3.6 nm."""
    rhos, ses = [], []
    for ring in _rings(11, 3.0):
        p = _pool_of(ring, syn.SIGMA_LOC_NM)
        assert p.verdict == "footprint measurable" and p.labels_per_cluster >= 1.5
        rhos.append(p.rho_nm)
        ses.append(p.se_nm)
    assert 19.0 < float(np.mean(rhos)) < 31.0
    assert all(s is not None and 0 < s < 15 for s in ses)
    # a NeNA under-estimated by 2 nm inflates it (the reason the NeNA of each file is required)
    low = float(np.mean([_pool_of(r, syn.SIGMA_LOC_NM - 2.0).rho_nm for r in _rings(11, 3.0)]))
    assert low > float(np.mean(rhos)) + 3.0


def test_pool_is_content_only_with_one_label_and_not_evaluated_without_events():
    ring = _rings(12, 1.0, n=1)[0]
    p = _pool_of(ring, syn.SIGMA_LOC_NM)
    assert p.verdict == "content only" and p.rho_nm is None and "not identifiable" in p.reason
    cs = asz.cluster_sizes(ring.x, ring.y, ring.truth, curve=ring.contour.xy)
    q = asz.pooled_footprint(cs.sd_tangential, cs.n_locs, None, 10.0)
    assert q.verdict == "not evaluated" and q.rho_nm is None
    r = asz.pooled_footprint(cs.sd_tangential, cs.n_locs, np.full(cs.n_clusters, 40.0), None)
    assert r.verdict == "not evaluated"


# --------------------------------------------------------------------------------------- 4. curvature
def test_curvature_exact_on_a_circle_and_orientation_free():
    R = 1000.0
    th = np.linspace(0, 2 * np.pi, 20000, endpoint=False)          # dense: a coarse polygon has corners
    circ = np.column_stack([R * np.cos(th), R * np.sin(th)])
    for pts in (circ, circ[::-1]):
        m = curv.membrane_from_curve(pts)
        assert abs(m.radius_nm - R) / R < 0.001
        assert np.allclose(m.kappa_per_nm * R, 1.0, atol=0.01)


def _fit_and_compare(seed: int, kind: str, R: float, aspect: float, n_notch: int = 0):
    rng = np.random.default_rng(seed)
    c = syn.make_contour(kind, R, rng, aspect=aspect, n_notch=n_notch)
    ring = syn.generate_ring(rng, c, labels_mean=3.0)
    cs = asz.cluster_sizes(ring.x, ring.y, ring.truth)
    m = ring.truth >= 0
    mem = curv.fit_curvature_membrane(ring.x[m], ring.y[m], cs.centroid)
    _, _, kr = mem.at(cs.centroid)
    true = c.kappa_at_points(cs.centroid) * c.R
    return mem, kr, true


def test_kappa_r_recovered_at_400_nm_on_a_thick_ellipse():
    """SIZE_SIM sec. 6.1: thick axons (R 1200-1600), ellipse, 3 labels, 400 nm knots: Spearman 0.90, RMSE 0.22."""
    rs, rmse = [], []
    for seed in (0, 1, 2):
        mem, kr, true = _fit_and_compare(seed, "ellipse", 1400.0, 1.6)
        assert mem.source == "localizations" and mem.knot_spacing_nm == 400.0 and mem.resolved
        rs.append(asz.spearman(kr, true)[0])
        rmse.append(float(np.sqrt(np.mean((kr - true) ** 2))))
    assert min(rs) > 0.8 and float(np.mean(rs)) > 0.88
    assert float(np.mean(rmse)) < 0.3


def test_concave_stretches_found_on_a_thick_notched_contour():
    hits = []
    for seed in (0, 1, 3):
        _, kr, true = _fit_and_compare(seed, "notched", 1400.0, 1.4, n_notch=3)
        deep = true < -0.3
        if deep.any():
            hits.append(float(np.mean(kr[deep] < 0)))
        assert float(np.mean(kr[true > 0.8] > 0)) > 0.9               # convex stays convex
    assert hits and float(np.mean(hits)) > 0.6


def test_thin_axon_is_labelled_curvature_not_resolved():
    mem, _, _ = _fit_and_compare(0, "ellipse", 500.0, 1.6)
    assert mem.radius_nm < 600 and not mem.resolved
    assert mem.resolution == "curvature not resolved"
    assert any("not resolved" in w for w in mem.warnings)


# --------------------------------------------------------------------------------------- 5. concave exclusion
def test_question_b_excludes_concave_clusters_from_the_primary_readout():
    rng = np.random.default_rng(9)
    kr = np.concatenate([rng.uniform(0.2, 2.0, 60), rng.uniform(-1.5, -0.1, 20)])
    content = np.concatenate([rng.poisson(40, 60).astype(float),            # no relation where convex
                              40 + 30 * (-kr[60:]) + rng.normal(0, 2, 20)])  # a strong one in the notches
    sc = curv.size_vs_curvature(content, kr)
    assert sc.n_convex == 60 and sc.n_concave == 20
    assert abs(sc.content_rho) < 0.3
    assert sc.content_rho_concave < -0.8
    allrho = asz.spearman(content, kr)[0]
    assert abs(allrho) > abs(sc.content_rho) + 0.2                     # the artefact moved the pooled value


# --------------------------------------------------------------------------------------- 6. permutation diagnostic
def _dbscan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    return np.asarray(DBSCAN(eps=25.0, min_samples=10).fit(np.column_stack([x, y])).labels_, dtype=np.int64)


def test_mark_permutation_diagnostic_has_the_documented_false_positives_with_dbscan():
    """SIZE_SIM sec. 5.2-5.3: under a TRUE null (sizes independent of position), DBSCAN fabricates a positive
    size-neighbour association; the mark permutation flags it (combined z far above 0), and does not with the true
    clusters. That is why it is only a diagnostic and the null of question (a) is the pipeline null."""
    rng = np.random.default_rng(20261016)
    z_db, z_or = [], []
    for _ in range(10):
        c = syn.make_contour("circle", 2000.0, rng)
        ring = syn.generate_ring(rng, c, labels_mean=3.0, background_fraction=0.05)
        for lab, sink in ((_dbscan(ring.x, ring.y), z_db), (ring.truth, z_or)):
            cs = asz.cluster_sizes(ring.x, ring.y, lab, rarefy_reps=1)
            mp = asz.mark_permutation_diagnostic(cs.n_locs.astype(float), cs.d_nn, rng)
            assert mp.n_permutations == 199 and "DIAGNOSTIC" in mp.note
            sink.append(mp.z)
    stouffer_db = float(np.sum(z_db) / math.sqrt(len(z_db)))
    stouffer_or = float(np.sum(z_or) / math.sqrt(len(z_or)))
    assert stouffer_db > 2.5                       # "detected" although nothing is there: a false positive
    assert abs(stouffer_or) < 2.5                  # with the true clusters there is nothing


# --------------------------------------------------------------------------------------- 7. across axons
def test_fisher_combination_and_the_animal_level():
    z = [1.0, 1.2, 0.8, 1.1, 0.9, 1.3]
    out = st.combine_axons_fisher(z, animal=["a", "a", "b", "b", "c", "c"])
    assert out["axon_n"] == 6 and out["axon_p"] < 0.01
    assert out["animal_n"] == 3 and out["animal_p"] > out["axon_p"]    # fewer units, less power
    assert asz.fisher_z(None, 30) is None and asz.fisher_z(0.5, 3) is None
    assert abs(asz.fisher_z(0.5, 28) - math.atanh(0.5) * 5.0) < 1e-12


def test_mixed_model_holds_its_level_and_covariates_follow_the_estimator_kind():
    rng = np.random.default_rng(1)
    rej = 0
    n_rep = 60
    for _ in range(n_rep):
        animal = np.repeat(np.arange(6), 6)
        group = (animal >= 3).astype(int)
        y = rng.normal(0, 1.0, 6)[animal] + rng.normal(0, 0.5, 36)       # animal effect, no group effect
        res = st.compare_groups(y, group, animal, estimator_kind="pool")
        rej += int(res.mixed_p is not None and res.mixed_p < 0.05)
    assert rej / n_rep < 0.15                                           # ~5 % (MC error at 60 replicates)
    with pytest.raises(ValueError):
        st.compare_groups([1.0] * 4, [0, 0, 1, 1], ["a", "b", "c", "d"], estimator_kind="apparent")
    res = st.compare_groups(list(rng.normal(size=36)), np.repeat([0, 1], 18), np.repeat(np.arange(6), 6),
                            estimator_kind="apparent", nena_nm=list(rng.normal(11, 1, 36)),
                            log_n_locs=list(rng.normal(4, 0.3, 36)))
    assert res.covariates == ["nena_nm", "log_n_locs"] and res.mixed_df == 2
    assert st.COVARIATES_BY_KIND["pool"] == () and any("at least 6" in w for w in
                                                       st.compare_groups([1, 2, 3, 4], [0, 0, 1, 1],
                                                                         ["a", "b", "c", "d"],
                                                                         estimator_kind="content").warnings)
    assert all("origin" in row for row in st.design_guidance())


# --------------------------------------------------------------------------------------- 8. plug-ins
def _frames_for(ring: syn.SyntheticRing, rng: np.random.Generator, n_frames: int = 200000) -> np.ndarray:
    """A frame per localization: each event a run of consecutive frames starting at a random frame (the events of
    one label rarely overlap in a long movie), as the event layer expects."""
    frames = np.zeros(ring.x.size, dtype=np.int64)
    starts = rng.integers(0, n_frames, int(ring.event.max()) + 1)
    order = np.argsort(ring.event, kind="stable")
    ev_sorted = ring.event[order]
    first = np.searchsorted(ev_sorted, ev_sorted, side="left")
    frames[order] = starts[ev_sorted] + (np.arange(ev_sorted.size) - first)
    return frames


@pytest.fixture(scope="module")
def runs_and_mm():
    from tools import multimethod as MM
    rng = np.random.default_rng(2026)
    c = syn.make_contour("ellipse", 1400.0, rng, aspect=1.3)
    ring = syn.generate_ring(rng, c, labels_mean=3.0, background_fraction=0.03)
    x, y = ring.x + 3000.0, ring.y + 3000.0
    z = rng.normal(300.0, 20.0, x.size)
    frames = _frames_for(ring, rng)
    run_fn = MM.runner_from_arrays(x, y, z, source_name="synthetic size ring", localization_frame=frames)
    runs = MM.run_methods(run_fn, ["HDBSCAN"], n_locs=x.size, axon_id="size0")
    sp.register_with_multimethod(replace=True)
    return runs, MM


def test_size_plugin_on_two_methods(runs_and_mm):
    runs, MM = runs_and_mm
    assert [r.status for r in runs] == [MM.STATUS_OK, MM.STATUS_OK]
    rows = MM.apply_analyses(runs, names=[sp.SIZE_ANALYSIS], extras={"nena_nm": syn.SIGMA_LOC_NM})
    assert not [r for r in rows if r.get("error")]
    by = {(r["method_key"], r["readout"]): r["value"] for r in rows}
    for run in runs:
        k = run.key
        assert by[(k, "size_n_clusters")] >= 15
        assert by[(k, "size_content_unit_verdict")] == "events"
        assert by[(k, "size_curvature_verdict")] == "resolved"
        assert by[(k, "size_label_regime_verdict")] in asz.REGIME_LEVELS
        assert by[(k, "size_n_convex")] + by[(k, "size_n_concave")] == by[(k, "size_n_clusters")]
        assert by[(k, "size_nloc_nn_null_p")] is None                  # no null ran: never the permutation's p
        assert by[(k, "size_nloc_nn_markperm_diagnostic_z")] is not None
        assert 8.0 < by[(k, "size_cloud_sd_median_nm")] < 25.0
        assert by[(k, "size_eps_nm")] == (25.0 if k == MM.REFERENCE_KEY else None)
    assert "r_eff" not in " ".join(r[0] for r in sp.SIZE_READOUTS)       # never r_eff as a size
    # without the NeNA: content in localizations, no footprint
    rows2 = MM.apply_analyses(runs[:1], names=[sp.SIZE_ANALYSIS])
    by2 = {r["readout"]: r["value"] for r in rows2}
    assert by2["size_content_unit_verdict"] == "localizations" and by2["size_rho_pool_nm"] is None
    assert by2["size_label_regime_verdict"] == "not evaluated"


def test_per_cluster_rows_join_the_cluster_table(runs_and_mm):
    runs, _MM = runs_and_mm
    an = runs[0].analysis
    sa = sp.size_analysis(an, nena_nm=syn.SIGMA_LOC_NM)
    rows = sp.cluster_rows(sa)
    assert [r["cluster_label"] for r in rows] == [int(v) for v in an.areas.labels]   # same clusters as Cluster area
    assert set(rows[0]) == set(sp.SIZE_CLUSTER_COLUMNS)
    assert all(sp.describe_column(c) for c in sp.SIZE_CLUSTER_COLUMNS)
    assert [r["size_n_locs"] for r in rows] == [int(v) for v in an.areas.n_locs]
    assert all(r["size_curvature_resolved"] is True for r in rows)
    assert any(r["size_n_events"] for r in rows)


def test_eps_sensitivity_and_pipeline_null_hook(runs_and_mm):
    runs, MM = runs_and_mm
    rows = MM.apply_analyses(runs, names=[sp.SIZE_EPS_ANALYSIS])
    by = {(r["method_key"], r["readout"]): r["value"] for r in rows}
    ref = MM.REFERENCE_KEY
    assert all(by[(ref, f"size_nloc_nn_spearman_eps{e}")] is not None for e in (20, 25, 30))
    assert by[(ref, "size_nloc_nn_spearman_eps25")] == pytest.approx(sp.nloc_nn_statistic(runs[0].analysis))
    assert all(by[("HDBSCAN", f"size_nloc_nn_spearman_eps{e}")] is None for e in (20, 25, 30))

    # a hook that returns null replicates: the p is the two-sided Monte Carlo p of the observed statistic
    obs = sp.nloc_nn_statistic(runs[0].analysis)
    seen: Dict[str, Any] = {}

    def hook(run: Any, context: Any, stats: Dict[str, Any]) -> Dict[str, List[float]]:
        seen["names"] = list(stats)
        seen["value"] = stats["size_nloc_nn_spearman"](run.analysis)
        return {"size_nloc_nn_spearman": list(np.linspace(-0.5, 0.5, 199) + obs + 1.0)}   # all above obs

    rows = MM.apply_analyses(runs[:1], names=[sp.SIZE_ANALYSIS], extras={"size_pipeline_null": hook})
    p = {r["readout"]: r["value"] for r in rows}["size_nloc_nn_null_p"]
    assert seen["names"] == ["size_nloc_nn_spearman"] and seen["value"] == pytest.approx(obs)
    assert p == pytest.approx(2.0 / 200.0)
    # precomputed results of the null branch, keyed by run
    rows = MM.apply_analyses(runs[:1], names=[sp.SIZE_ANALYSIS],
                             extras={"pipeline_null_results": {ref: SimpleNamespace(
                                 p_values={"size_nloc_nn_spearman": 0.37})}})
    assert {r["readout"]: r["value"] for r in rows}["size_nloc_nn_null_p"] == pytest.approx(0.37)


def test_register_with_pipeline_null_uses_its_readout_api(monkeypatch):
    registered: List[Any] = []

    class Spec(SimpleNamespace):
        pass

    fake = SimpleNamespace(ReadoutSpec=lambda name, fn, unit, **kw: Spec(name=name, fn=fn, unit=unit, **kw),
                           register_readout=lambda spec, replace=False: registered.append(spec))
    monkeypatch.setitem(sys.modules, "tools.mps_pipeline_null", fake)
    assert sp.register_with_pipeline_null() == ["size_nloc_nn_spearman"]
    spec = registered[0]
    assert spec.name == "size_nloc_nn_spearman" and spec.needs_ring is False
    view = SimpleNamespace(analysis=SimpleNamespace(
        x_slab=np.r_[0.0, 1, 2, 100, 101, 102, 400, 401, 402, 900], y_slab=np.zeros(10),
        labels=np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, -1], bad_report=SimpleNamespace(bad_labels=set()),
        discarded_labels=frozenset()))
    assert spec.fn(view) is None                    # three clusters: fewer than four pairs, not computable
