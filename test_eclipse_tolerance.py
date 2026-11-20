# -*- coding: utf-8 -*-
"""
The eclipse tolerance of P-R61 (``tools/mps_eclipse_tolerance.py``; the per-pair rule in ``tools/mps_matching.py`` /
``tools/mps_unroll.py``; the derived tau_0 of ``config/columns_params.yaml``). SYNTHETIC DATA ONLY (R8): the P-R61
research generator (``docs/sims/eclipse_tolerance/eclipse_sim.py``) and the project's simulated axon
(``batch_columns.write_simulated_input``).

- The matching with a tolerance MATRIX: the scalar rule is the matrix with every entry tau (same pairs, same null,
  bit for bit); D-24b's trap and the monotonicity hold; wrong shapes and values are refused.
- The centre error: the formula on hand-computed cases, centroid_r95 = 2.448 sigma_c, the derived tau = 2.77 x
  median sigma_c, the simulated sum-w^2 table (deterministic, decreasing with the events).
- The footprint: pooled when >= 2 labels per cluster (the root-mean-square footprint, +-5 nm), NOT identifiable
  with one label or without events (25 nm used, 0 nm sensitivity, said so); the merged-cluster criterion on
  constructed singles and merged pairs, and on DBSCAN merges.
- The rules on synthetic rings: false-positive rate held under the null for the scalar and the per-pair rule; the
  research's power ordering (precision rules >> a wide tolerance) on a small grid.
- The inputs: calibrated c_label needs a date and then makes the per-pair rule primary.
- The exports (section 7.3), the plug-in (``tools.multimethod``, readouts ``eclipse_*``) with two methods, and
  ``build_rings(cluster_method=...)``.
- The private pilot value is gone from the repository (a grep), and tau_0 is the derived 31 nm.

Run: ``py -3 -m pytest test_eclipse_tolerance.py -q`` (about 1-2 min, one process).
"""
from __future__ import annotations

import dataclasses
import math
import os
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from tools import mps_eclipse_tolerance as et
from tools.mps_matching import assign_within_tau, match_arcs
from tools.mps_unroll import PAIR_K_ARC, arc_eclipse_test, arc_joint_null

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "docs", "sims", "eclipse_tolerance"))
import eclipse_sim as es  # noqa: E402

# The pool's estimator as the research validated it: without the event layer's fusion exclusion (user decision E2,
# which tools.footprint_pool applies by default; its own tests are in test_core_tanda1.py)
NO_E2 = et.ToleranceInputs(exclude_event_fusion_from_pool=False)


# ============================================================================== matching with a tolerance matrix
def test_scalar_rule_is_the_matrix_of_constant_tau() -> None:
    rng = np.random.default_rng(3)
    L = 6000.0
    a, b = np.sort(rng.uniform(0, L, 30)), np.sort(rng.uniform(0, L, 28))
    tau = 31.0
    sig = np.full(30, tau / (PAIR_K_ARC * math.sqrt(2.0)))
    sigb = np.full(28, tau / (PAIR_K_ARC * math.sqrt(2.0)))
    m0 = arc_eclipse_test(a, b, L, tau, n_null=99, random_seed=5)
    m1 = arc_eclipse_test(a, b, L, tau, n_null=99, random_seed=5, sigma_a_nm=sig, sigma_b_nm=sigb)
    assert m0.tolerance_rule == "scalar" and m1.tolerance_rule == "per_pair"
    assert np.array_equal(m0.i_a, m1.i_a) and np.array_equal(m0.j_b, m1.j_b)
    assert np.array_equal(m0.null_n_matched, m1.null_n_matched)
    assert np.array_equal(m0.null_shift_nm, m1.null_shift_nm) and m0.p_excess == m1.p_excess
    assert np.allclose(m1.t_nm, tau) and np.all(m0.t_nm == tau)
    j0 = arc_joint_null([0, 1, 2], [a, b, a + 7.0], L, tau, reference_ring=0, n_null=49)
    j1 = arc_joint_null([0, 1, 2], [a, b, a + 7.0], L, tau, reference_ring=0, n_null=49, sigma_nm=[sig, sigb, sig])
    assert np.array_equal(j0.null_n_matched, j1.null_n_matched) and j0.T_obs == j1.T_obs
    assert j1.tolerance_rule == "per_pair" and j0.tolerance_rule == "scalar"


def test_matrix_keeps_d24b_and_monotonicity() -> None:
    # the 3-vs-2 trap of match_rings on the circle: three pairs at 55 nm beat two at 0 nm
    a = np.array([0.0, 55.0, 110.0])
    b = np.array([-55.0, 0.0, 55.0])
    T = np.full((3, 3), 60.0)
    i, j, d = match_arcs(a, b, 1e6, 60.0, tolerance_nm=T)
    assert i.size == 3 and np.allclose(d, 55.0)
    # per-pair: with 1 nm everywhere only the two coincident pairs (0 nm) are admitted; opening (1, 1) to 56 nm adds
    # an admissible pair but no larger matching, so the minimum-distance one stays
    T2 = np.full((3, 3), 1.0)
    i, j, _ = match_arcs(a, b, 1e6, 60.0, tolerance_nm=T2)
    assert list(zip(i.tolist(), j.tolist())) == [(0, 1), (1, 2)]
    T2[1, 1] = 56.0
    i, j, _ = match_arcs(a, b, 1e6, 60.0, tolerance_nm=T2)
    assert list(zip(i.tolist(), j.tolist())) == [(0, 1), (1, 2)]
    # admitting (2, 0) (165 nm apart) on top of the two coincident pairs gives three: maximum cardinality first
    T3 = np.full((3, 3), 1e-6)
    T3[0, 1] = T3[1, 2] = 1e-9
    T3[2, 0] = 170.0
    D = np.abs(a[:, None] - b[None, :])
    assert D[0, 1] == 0.0 and D[1, 2] == 0.0                     # still admitted (0 <= t)
    assert match_arcs(a, b, 1e6, 60.0, tolerance_nm=T3)[0].size == 3
    rng = np.random.default_rng(1)
    D = rng.uniform(0, 100, (8, 9))
    base = rng.uniform(5, 40, (8, 9))
    counts = [assign_within_tau(D, base * s)[0].size for s in (0.5, 1.0, 1.5, 2.0, 3.0)]
    assert counts == sorted(counts)
    with pytest.raises(ValueError):
        assign_within_tau(D, base[:, :3])
    with pytest.raises(ValueError):
        match_arcs(a, b, 1e6, 60.0, tolerance_nm=np.full((3, 3), -1.0))
    with pytest.raises(ValueError):
        arc_eclipse_test(a, b, 1000.0, 30.0, n_null=9, sigma_a_nm=np.ones(3))


# ============================================================================== centre error
def test_sigma_c_formula_r95_and_derived_tau() -> None:
    inp = et.ToleranceInputs()
    s = et.sigma_c_nm([100], [1.0 / 3.0], 25.0, inp)[0]
    want = math.sqrt(10.0 ** 2 / 100 + (10.0 ** 2 + 25.0 ** 2 / 4) / 3.0 + 2.0 ** 2)
    assert abs(s - want) < 1e-12
    assert abs(et.K_PLANE - 2.4477) < 1e-3 and abs(et.K_SCALAR - 2.7719) < 1e-3
    assert et.scalar_tau_from_sigma([10.0, 11.3, 12.6]) == pytest.approx(et.K_SCALAR * 11.3)
    # the derived default: 2.77 x 11.3 nm rounds to 31 nm
    assert round(et.K_SCALAR * et.SIGMA_C_REFERENCE_NM) == et.TAU_SCALAR_DERIVED_NM == 31.0


def test_sum_w2_table_is_deterministic_and_decreasing() -> None:
    e1, t1 = et.sum_w2_table(14.0, 3.0)
    et.sum_w2_table.cache_clear()
    e2, t2 = et.sum_w2_table(14.0, 3.0)
    assert e1 == e2 and t1 == t2
    tab = np.asarray(t1)
    fin = tab[np.isfinite(tab)]
    assert fin[0] > 0.9 and fin[-1] < 0.5          # few events: one label; many: several
    assert np.all(np.diff(fin[:12]) < 0.02)         # (nearly) monotone over the populated bins
    w = et.sum_w2_given_events([np.nan, 2, 40, 200], 14.0, 3.0)
    assert w[0] == 1.0 and w[1] > w[2] > w[3]
    assert et.lambda_from_mean_labels(1.0) == 0.05 and et.lambda_from_mean_labels(3.2) == 3.0


# ============================================================================== footprint and merged clusters
def _ring_groups(rng: np.random.Generator, scen: str, k: int = 60, spacing: float = 120.0
                 ) -> Dict[str, Any]:
    """k separated clusters on a line along x (tangent), with events: positions far apart, centre far below."""
    sc = es.SCENARIOS[scen]
    xs, ys, evs, members = [], [], [], []
    off = n0 = 0
    vals, probs = zip(*sc["rho_mix"])
    rho = rng.choice(np.asarray(vals), k, p=np.asarray(probs))
    labels = es.draw_labels(rng, k, sc["label_mode"], sc["lam"])
    for i in range(k):
        t, n, ev = es.draw_cluster(rng, float(rho[i]), int(labels[i]))
        xs.append(i * spacing + t)
        ys.append(1e7 + n)
        evs.append(ev + off)
        off += int(ev.max()) + 1
        members.append(np.arange(n0, n0 + t.size))
        n0 += t.size
    return dict(x=np.concatenate(xs), y=np.concatenate(ys), ev=np.concatenate(evs), members=members,
                rms=float(np.sqrt(np.mean(rho ** 2))))


def test_pool_measures_the_rms_footprint_with_labels() -> None:
    rng = np.random.default_rng(11)
    for scen, truth in (("H0_rho25_ztp3", 25.0), ("H4_rho40_ztp3", 40.0)):
        got = []
        for _ in range(6):
            g = _ring_groups(rng, scen)
            # the estimator itself, without the event layer's fusion exclusion (E2 is tested in test_core_tanda1.py)
            tol = et.axon_tolerances(g["x"], g["y"], {0: g["members"]}, event_id=g["ev"],
                                     centres_nm={0: [0.0, 0.0]}, inputs=NO_E2)
            assert tol.pool.identifiable and tol.pool.rho_source == et.RHO_POOL
            assert tol.pool.rho_used_nm == tol.pool.rho_pooled_nm
            got.append(tol.pool.rho_pooled_nm)
        assert abs(float(np.mean(got)) - truth) < 5.0, (scen, got)


def test_one_label_or_no_events_is_not_identifiable() -> None:
    rng = np.random.default_rng(12)
    g = _ring_groups(rng, "H2_rho0_one_label")
    tol = et.axon_tolerances(g["x"], g["y"], {0: g["members"]}, event_id=g["ev"], centres_nm={0: [0.0, 0.0]})
    p = tol.pool
    assert not p.identifiable and p.rho_pooled_nm is None and p.rho_used_nm == 25.0 and p.rho_sensitivity_nm == 0.0
    assert p.rho_source == et.RHO_DEFAULT_NOT_IDENTIFIABLE and "not a measurement" in " ".join(tol.warnings)
    gr = tol.groups[0]
    assert np.all(gr.sigma_c_sensitivity_nm < gr.sigma_c_nm) and gr.sigma_c_source == et.SIGMA_C_PRESET
    assert np.allclose(gr.centroid_r95_nm, et.K_PLANE * gr.sigma_c_nm)
    tol2 = et.axon_tolerances(g["x"], g["y"], {0: g["members"]}, event_id=None, centres_nm={0: [0.0, 0.0]})
    assert tol2.pool.rho_source == et.RHO_DEFAULT_NO_EVENTS and np.all(tol2.groups[0].sum_w2_hat == 1.0)
    assert tol2.groups[0].sigma_c_source == et.SIGMA_C_NO_EVENTS and tol2.lambda_hat is None


def test_merged_criterion_on_constructed_singles_and_pairs() -> None:
    rng = np.random.default_rng(13)
    inp = et.ToleranceInputs()
    thr = et.merged_threshold_nm(inp)
    assert 74.0 < thr < 75.0                      # 3.29 sqrt(2) sqrt(10^2 + 25^2/4)
    singles, merged = [], []
    for _ in range(300):
        lab = int(es.draw_labels(rng, 1, "ztp", 3.0)[0])
        t, _n, ev = es.draw_cluster(rng, float(rng.choice([0.0, 25.0])), lab)
        singles.append(et.two_mode_separation(t, ev))
        t2, _n2, ev2 = es.draw_cluster(rng, 25.0, int(es.draw_labels(rng, 1, "ztp", 3.0)[0]))
        sep = rng.uniform(90.0, 130.0)               # two real clusters: hard core 80 nm + their extent
        merged.append(et.two_mode_separation(np.concatenate([t, t2 + sep]),
                                             np.concatenate([ev, ev2 + int(ev.max()) + 1])))
    fs = et.merged_cluster_flags(np.asarray(singles), inp)
    fm = et.merged_cluster_flags(np.asarray(merged), inp)
    assert fs.mean() <= 0.01, fs.mean()
    assert fm.mean() >= 0.80, fm.mean()


def test_merged_exclusion_on_dbscan_merges_reduces_the_bias() -> None:
    r = es.merged_experiment(8, scenarios=("H1_mixed_ztp3",))["H1_mixed_ztp3"]
    truth = r["rho_true_rms"][0]
    assert r["flag_rate_dbscan_singles"] <= 0.02 and r["detection_rate_dbscan_merged"] >= 0.4
    assert r["rho_pool_dbscan_all"][0] > truth + 4.0                 # the merges inflate the pool: an upper bound
    assert abs(r["rho_pool_dbscan_excluded"][0] - truth) < abs(r["rho_pool_dbscan_all"][0] - truth)


def test_switch_to_the_excluded_pool_is_recorded() -> None:
    rng = np.random.default_rng(14)
    g = _ring_groups(rng, "H1_mixed_ztp3")
    # add one merged cluster: two clusters fused
    members = list(g["members"])
    members[0] = np.concatenate([members[0], members[1]])
    del members[1]
    a = et.axon_tolerances(g["x"], g["y"], {0: members}, event_id=g["ev"], centres_nm={0: [0.0, 0.0]}, inputs=NO_E2)
    inp = dataclasses.replace(NO_E2, exclude_merged_from_pool=True)
    b = et.axon_tolerances(g["x"], g["y"], {0: members}, event_id=g["ev"], centres_nm={0: [0.0, 0.0]}, inputs=inp)
    assert a.groups[0].merged_suspect[0] and a.pool.n_excluded_merged >= 1
    assert a.pool.rho_source == et.RHO_POOL and b.pool.rho_source == et.RHO_POOL_EXCLUDED
    assert b.pool.rho_used_nm == b.pool.rho_pooled_excluding_merged_nm < a.pool.rho_used_nm


# ============================================================================== the rules on synthetic rings
def test_false_positive_rate_held_under_the_null() -> None:
    r = es.rules_experiment(60, 99, cells=(("H1_mixed_ztp3", 0.0), ("H4_rho40_ztp3", 0.0)))
    for cell, rates in r.items():
        for rule in es.RULES:
            assert rates[rule] <= 0.10, (cell, rule, rates[rule])   # nominal 0.05; SE 0.028 at 60 axons


def test_power_ordering_of_the_research() -> None:
    r = es.rules_experiment(40, 99, cells=(("H1_mixed_ztp3", 0.3),))["H1_mixed_ztp3|f=0.3"]
    assert r["scalar31"] >= r["scalar80_wide"] + 0.25, r
    assert r["per_pair"] >= r["scalar80_wide"] + 0.25, r
    assert abs(r["per_pair"] - r["scalar31"]) <= 0.25, r              # the precision rules are equivalent


# ============================================================================== inputs
def test_calibration_hook_and_primary_rule() -> None:
    inp = et.ToleranceInputs()
    assert inp.primary_rule() == et.RULE_SCALAR and not inp.calibrated
    with pytest.raises(ValueError):
        et.ToleranceInputs(c_label=20.0, c_label_source="calibrated")
    cal = et.ToleranceInputs(c_label=20.0, c_label_source="calibrated", c_label_date="2027-01-15")
    assert cal.calibrated and cal.primary_rule() == et.RULE_PER_PAIR
    assert inp.with_nena(12.5).sigma_loc_nm == 12.5 and inp.with_nena(None) is inp and inp.with_nena(-1) is inp
    with pytest.raises(ValueError):
        et.ToleranceInputs(sigma_link_nm=0.0)


# ============================================================================== the rings, the exports, determinism
@pytest.fixture(scope="module")
def sim_axon(tmp_path_factory: Any) -> Dict[str, Any]:
    import batch_columns as bc
    folder = tmp_path_factory.mktemp("eclipse_axon")
    npz = bc.write_simulated_input(str(folder / "sim.npz"), "circle", 39, 0.3)
    z = np.load(npz)
    return {k: z[k] for k in ("x", "y", "z", "frame", "lp", "lpz", "n_frames")}


def test_rings_per_pair_arc_and_exports(sim_axon: Dict[str, Any]) -> None:
    from tools.mps_columns import build_rings, rings_params_from
    from tools.mps_lumen import default_columns_params
    from tools.mps_unroll import UnrollParams, analyze_arc_columns
    cp = default_columns_params()
    z = sim_axon
    res = build_rings(z["x"], z["y"], z["z"], frame=z["frame"], lp_lateral_nm=z["lp"], lpz_nm=z["lpz"],
                      params=rings_params_from(cp), n_frames=int(z["n_frames"]))
    tol = et.tolerances_from_rings(res)
    tol2 = et.tolerances_from_rings(res)
    assert np.array_equal(tol.all_sigma_c(), tol2.all_sigma_c()) and tol.pool == tol2.pool     # determinism
    stub: Any = SimpleNamespace(include_suspect=True, tau0_nm=cp.tau0_nm, random_seed=0)
    up = UnrollParams(n_null=19)
    sc = analyze_arc_columns(res, stub, params=up, reference_curve="centroid_membrane")
    pp = et.per_pair_arc_columns(res, stub, tol, params=up, reference_curve="centroid_membrane")
    assert sc.tolerance_rule == "scalar" and pp.tolerance_rule == "per_pair" and pp.pair_k == et.K_ARC
    assert sc.tau_nm == pp.tau_nm == 31.0
    for ms, mp in zip(sc.adjacent, pp.adjacent):
        assert np.array_equal(ms.null_shift_nm, mp.null_shift_nm)              # the same rotations
        T = et.pair_tolerance_matrix(tol.sigma_of(mp.ring_a), tol.sigma_of(mp.ring_b))
        assert np.allclose(mp.t_nm, T[mp.i_a, mp.j_b]) and np.all(mp.d_nm <= mp.t_nm + 1e-9)
    crow = et.cluster_rows(tol)
    assert len(crow) == sum(len(r.clusters) for r in res.rings) and set(crow[0]) == set(et.CLUSTER_COLUMNS)
    prow = et.pair_rows(sc, pp, tol.inputs)
    assert {r["rule"] for r in prow} == {"scalar", "per_pair"} and set(prow[0]) == set(et.PAIR_COLUMNS)
    assert {r["role"] for r in prow if r["rule"] == "per_pair"} == {"exploratory"}
    arow = et.axon_row(tol, sc, pp)
    assert list(arow) == list(et.AXON_COLUMNS) and arow["primary_rule"] == "scalar" and arow["tau_scalar_nm"] == 31.0
    assert all(et.describe(c) for c in et.CLUSTER_COLUMNS + et.PAIR_COLUMNS + et.AXON_COLUMNS)
    text = "\n".join(et.summary_lines(tol, sc, pp))
    assert "PRIMARY tau = 31.0 nm" in text and "EXPLORATORY t_ij" in text


# ============================================================================== plug-in, two methods
def test_plugin_registered_and_runs_with_two_methods(sim_axon: Dict[str, Any]) -> None:
    from tools import multimethod as MM
    assert et.ECLIPSE_ANALYSIS in MM.registered_analyses()
    names = [s.name for s in MM.readout_specs([et.ECLIPSE_ANALYSIS])]
    assert tuple(names) == et.ECLIPSE_READOUTS and all(n.startswith("eclipse_") for n in names)
    # one simulated ring slab: the ring nearest z = 0 of the simulated axon (synthetic)
    rng = np.random.default_rng(15)
    g = _ring_groups(rng, "H1_mixed_ztp3", k=30)
    ang = g["x"] / (30 * 120.0) * 2 * np.pi
    r = 900.0 + (g["y"] - 1e7)
    x, y = 3000.0 + r * np.cos(ang), 3000.0 + r * np.sin(ang)
    zz = rng.normal(0.0, 20.0, x.size)
    run_fn = MM.runner_from_arrays(x, y, zz, source_name="synthetic ring", slab_override=(-200.0, 200.0))
    runs = MM.run_methods(run_fn, ["FOCAL"], n_locs=x.size, axon_id="ring")
    assert all(r.ok for r in runs), [(r.key, r.error) for r in runs]
    rows = MM.apply_analyses(runs, [et.ECLIPSE_ANALYSIS], extras={"event_id": g["ev"]})
    by = {(r["method_key"], r["readout"]): r for r in rows}
    for run in runs:
        assert not by[(run.key, "eclipse_n_clusters")]["error"], by[(run.key, "eclipse_n_clusters")]
        assert by[(run.key, "eclipse_n_clusters")]["value"] > 10
        assert by[(run.key, "eclipse_sigma_c_source")]["value"] == et.SIGMA_C_PRESET
        assert by[(run.key, "eclipse_n_ring_pairs")]["value"] is None             # no rings input
    rows2 = MM.apply_analyses(runs, [et.ECLIPSE_ANALYSIS], extras={"event_id": g["ev"]})
    assert [r["value"] for r in rows] == [r["value"] for r in rows2]            # determinism


def test_plugin_rings_path_with_another_method(sim_axon: Dict[str, Any]) -> None:
    from tools import multimethod as MM
    z = sim_axon
    ctx = MM.AnalysisContext(axon_id="a", runs=(), reference=None, n_replicates=19,
                             extras={"eclipse_rings_input": dict(x=z["x"], y=z["y"], z=z["z"], frame=z["frame"],
                                                                 lp_lateral_nm=z["lp"], lpz_nm=z["lpz"],
                                                                 n_frames=int(z["n_frames"]))})
    fake = SimpleNamespace(method="FOCAL", params={}, seconds=1.0)
    out = et._rings_readouts(fake, ctx, et.ToleranceInputs(), ctx.extras["eclipse_rings_input"])
    assert out["eclipse_n_ring_pairs"] >= 1
    for k in ("eclipse_scalar_p", "eclipse_per_pair_p"):
        assert out[k] is not None and 0.0 < out[k] <= 1.0
    slow = SimpleNamespace(method="OPTICS", params={}, seconds=1.0)
    with pytest.raises(MM.SlowMethodRefused):
        et._rings_readouts(slow, ctx, et.ToleranceInputs(), ctx.extras["eclipse_rings_input"])


# ============================================================================== the private pilot value is gone
FORBIDDEN = (re.compile(r"80[.,]5756"), re.compile(r"80[.,]58\s*nm"), re.compile(r"80[.,]6\s*nm"),
             re.compile(r"tau_?0?\w*\W{0,3}[:=]\W{0,3}80[.,]5"))


def test_the_pilot_tau_is_gone_and_tau0_is_derived() -> None:
    files = subprocess.run(["git", "ls-files"], cwd=HERE, capture_output=True, text=True, check=True).stdout.split("\n")
    hits: List[str] = []
    for rel in files:
        if not rel or rel.startswith("example data/") or rel.endswith((".png", ".ico", ".npz", ".pdf", ".tif")):
            continue
        path = os.path.join(HERE, rel)
        if not os.path.isfile(path) or rel == os.path.basename(__file__):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        for rx in FORBIDDEN:
            if rx.search(text):
                hits.append(f"{rel}: {rx.pattern}")
    assert not hits, hits
    from tools.mps_lumen import default_columns_params
    cp = default_columns_params()
    assert cp.tau0_nm == et.TAU_SCALAR_DERIVED_NM == 31.0
    assert cp.tau_sensitivity_nm == [31.0, 62.0, 100.0]
