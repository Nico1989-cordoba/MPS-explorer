"""Every analysis with several clustering methods (tools/multimethod.py): runs on the same slab with the reference
always first, slow methods refused, failures as rows, plug-ins, agreement between methods, the "does the
conclusion depend on the method?" summary, and the CSV round trip.

Synthetic data only (a ring of clusters, as test_phase_d_integration._ring_axon). Run:
py -3 -m pytest test_multimethod.py -q
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pytest

from tools import multimethod as MM
from tools.clustering_strategies import COMPAT_METHOD


def _ring_axon(seed: int = 0, n_clusters: int = 24, radius_nm: float = 400.0):
    """A ring of clusters (one axon's cross-section) with background, in nm (test_phase_d_integration)."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False) + rng.normal(0, 0.03, n_clusters)
    xs, ys, zs = [], [], []
    for a in ang:
        n = int(rng.integers(15, 40))
        xs.append(radius_nm * np.cos(a) + rng.normal(0, 9, n))
        ys.append(radius_nm * np.sin(a) + rng.normal(0, 9, n))
        zs.append(rng.normal(0, 20, n))
    nb = 150
    xs.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    ys.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    zs.append(rng.normal(0, 60, nb))
    return np.concatenate(xs) + 2000.0, np.concatenate(ys) + 2000.0, np.concatenate(zs) + 300.0


FAST = ["HDBSCAN", "FOCAL"]


@pytest.fixture(scope="module", autouse=True)
def _only_the_light_plugins():
    """These tests call ``apply_analyses`` with every registered plug-in; the clustering integration (imported by
    other test modules of a full run) registers the per-axon null and the gap measures too, 199 simulated axons per
    run. They are set aside for this module and put back after it."""
    heavy = ("pipeline_null", "gap_measures", "sweep", "size_eps")
    kept = {n: MM._ANALYSES.pop(n) for n in heavy if n in MM._ANALYSES}
    yield
    MM._ANALYSES.update(kept)


@pytest.fixture(scope="module")
def ring_runs() -> List[MM.MethodRun]:
    x, y, z = _ring_axon()
    run_fn = MM.runner_from_arrays(x, y, z, source_name="synthetic ring")
    return MM.run_methods(run_fn, FAST, n_locs=x.size, axon_id="ring0")


# ----------------------------------------------------------------------------------------------- plan
def test_plan_puts_the_reference_first_and_keys_the_runs():
    plan = MM.plan_runs(["FOCAL", ("DBSCAN", {"eps": 25.0, "min_samples": 10}), ("DBSCAN", {"eps": 30.0})], 1000)
    keys = [j[0] for j in plan.jobs]
    assert keys[0] == MM.REFERENCE_KEY and plan.jobs[0][3] is True
    assert keys.count(MM.REFERENCE_KEY) == 1                        # DBSCAN 25/10 is the reference, not a copy
    assert "FOCAL" in keys
    other = [j for j in plan.jobs if j[1] == COMPAT_METHOD and not j[3]]
    assert len(other) == 1 and other[0][2]["eps"] == 30.0 and other[0][2]["min_samples"] == 10
    assert other[0][0].startswith("DBSCAN {") and '"eps":30.0' in other[0][0]
    assert plan.seconds > 0 and not plan.refused


def test_plan_without_reference_and_with_params_for():
    plan = MM.plan_runs(["FOCAL"], 500, params_for=lambda n: {"grid_nm": 30.0}, include_reference=False)
    assert [(j[0], j[2]) for j in plan.jobs] == [("FOCAL", {"grid_nm": 30.0})]


def test_slow_methods_are_refused_unless_allowed():
    plan = MM.plan_runs(["emitters_bagol_like", "OPTICS", "FOCAL"], 2000)
    assert set(plan.refused) == {"emitters_bagol_like", "OPTICS"}
    assert "slow" in plan.refused["OPTICS"] and "allow_slow" in plan.refused["OPTICS"]
    assert plan.seconds_if_allowed > plan.seconds
    assert "refused as slow" in plan.words()
    allowed = MM.plan_runs(["emitters_bagol_like"], 2000, allow_slow=True)
    assert not allowed.refused

    called: List[str] = []

    def spy(method: str, params: Dict[str, Any]) -> Any:
        called.append(method)
        raise RuntimeError("only the spy")

    runs = MM.run_methods(spy, ["emitters_bagol_like", "FOCAL"], n_locs=2000)
    assert "emitters_bagol_like" not in called                      # never ran
    st = {r.key: r.status for r in runs}
    assert st["emitters_bagol_like"] == MM.STATUS_REFUSED
    assert st["FOCAL"] == MM.STATUS_ERROR and st[MM.REFERENCE_KEY] == MM.STATUS_ERROR


# ----------------------------------------------------------------------------------------------- runs
def test_runs_share_the_slab_and_the_reference_is_plain_dbscan(ring_runs):
    from sklearn.cluster import DBSCAN
    assert [r.key for r in ring_runs] == [MM.REFERENCE_KEY] + FAST
    assert all(r.ok for r in ring_runs), [(r.key, r.error) for r in ring_runs]
    ref = ring_runs[0]
    assert ref.is_reference and ref.analysis.eps_nm == 25.0 and ref.analysis.min_samples == 10
    pts = np.column_stack([ref.analysis.x_slab, ref.analysis.y_slab])
    np.testing.assert_array_equal(ref.labels, DBSCAN(eps=25.0, min_samples=10).fit(pts).labels_)
    for r in ring_runs[1:]:
        np.testing.assert_array_equal(r.analysis.x_slab, ref.analysis.x_slab)
        assert r.analysis.cluster_method == r.method
        assert r.seconds > 0


def test_a_failing_method_is_a_row_and_the_others_run():
    x, y, z = _ring_axon(seed=1)
    good = MM.runner_from_arrays(x, y, z)

    def run_fn(method: str, params: Dict[str, Any]) -> Any:
        if method == "FOCAL":
            raise ValueError("broken on purpose")
        return good(method, params)

    runs = MM.run_methods(run_fn, ["FOCAL", "no such method"], n_locs=x.size, axon_id="a1")
    by = {r.key: r for r in runs}
    assert by[MM.REFERENCE_KEY].ok
    assert by["FOCAL"].status == MM.STATUS_ERROR and "broken on purpose" in by["FOCAL"].error
    assert "ValueError" in by["FOCAL"].traceback
    assert by["no such method"].status == MM.STATUS_ERROR and "KeyError" in by["no such method"].error
    rows = MM.apply_analyses(runs, [MM.CORE_ANALYSIS])
    failed = [r for r in rows if r["method_key"] == "FOCAL"]
    assert failed and all(r["value"] is None and "broken" in r["error"] for r in failed)


def test_cancel_stops_before_the_next_method():
    x, y, z = _ring_axon(seed=2)
    good = MM.runner_from_arrays(x, y, z)
    done: List[str] = []

    def run_fn(method: str, params: Dict[str, Any]) -> Any:
        out = good(method, params)
        done.append(method)                    # the cancel arrives once the first method has finished
        return out

    runs = MM.run_methods(run_fn, ["FOCAL", "HDBSCAN"], n_locs=x.size, cancelled=lambda: len(done) >= 1)
    assert [r.status for r in runs] == [MM.STATUS_OK, MM.STATUS_CANCELLED, MM.STATUS_CANCELLED]
    assert done == [COMPAT_METHOD]


def test_cancel_inside_a_method_stops_it_at_its_next_check():
    # tools.mps_cancel: the analysis asks between its steps (before the clustering, the perimeter, ...)
    x, y, z = _ring_axon(seed=2)
    runs = MM.run_methods(MM.runner_from_arrays(x, y, z), ["FOCAL"], n_locs=x.size,
                          cancelled=_after_calls(2))
    assert runs[0].status == MM.STATUS_CANCELLED and "while this method ran" in runs[0].error
    assert runs[1].status == MM.STATUS_CANCELLED and "before" in runs[1].error


def _after_calls(n: int):
    calls = [0]

    def cancelled() -> bool:
        calls[0] += 1
        return calls[0] > n
    return cancelled


def test_runs_of_other_localizations_are_refused():
    xa, ya, za = _ring_axon(seed=3)
    xb, yb, zb = _ring_axon(seed=4)
    ra, rb = MM.runner_from_arrays(xa, ya, za), MM.runner_from_arrays(xb, yb, zb)
    runs = MM.run_methods(lambda m, p: ra(m, p) if m == COMPAT_METHOD else rb(m, p), ["FOCAL"], n_locs=xa.size)
    assert runs[0].ok and runs[1].status == MM.STATUS_ERROR and "same localizations" in runs[1].error


def test_determinism_with_a_seed():
    x, y, z = _ring_axon(seed=5)
    tables = []
    for _ in range(2):
        runs = MM.run_methods(MM.runner_from_arrays(x, y, z), ["FOCAL"], n_locs=x.size, axon_id="d")
        # every plug-in registered so far but the ones that repeat the clustering many times (the null, the gap
        # measures' null, the sweeps: registered by the clustering integration in a full run; the null also
        # reports its measured seconds per replicate, which no seed fixes)
        heavy = {"pipeline_null", "gap_measures", "sweep", "size_eps"}
        names = [n for n in MM.registered_analyses() if n not in heavy]
        rows = MM.apply_analyses(runs, names, seed=7)
        tables.append([{k: v for k, v in r.items() if k != "seconds"} for r in rows])
    assert tables[0] == tables[1]


def test_runner_from_analysis_repeats_the_analysis_shown():
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring_axon(seed=6)
    base = analyze_axon(x, y, z, run_randomization=False, eps_nm=30.0, min_samples=8)
    runs = MM.run_methods(MM.runner_from_analysis(base), [("DBSCAN", {"eps": 30.0, "min_samples": 8})],
                          n_locs=base.x_slab.size)
    assert runs[0].key == MM.REFERENCE_KEY and runs[0].analysis.eps_nm == 25.0
    mine = runs[1]
    assert not mine.is_reference and mine.analysis.eps_nm == 30.0
    np.testing.assert_array_equal(mine.labels, base.labels)
    assert mine.analysis.slab_zmin_nm == base.slab_zmin_nm and mine.analysis.slab_zmax_nm == base.slab_zmax_nm


# ----------------------------------------------------------------------------------------------- plug-ins
def test_the_core_readouts_are_registered_first(ring_runs):
    assert MM.registered_analyses()[:2] == [MM.CORE_ANALYSIS, MM.RANDOMIZATION_ANALYSIS]
    rows = MM.apply_analyses(ring_runs)
    ref = {r["readout"]: r["value"] for r in rows if r["method_key"] == MM.REFERENCE_KEY}
    an = ring_runs[0].analysis
    assert ref["clusters_per_um"] == pytest.approx(an.clusters_per_um)
    assert ref["median_1nn_nm"] == pytest.approx(an.median_1nn_nm)
    assert ref["median_r_eff_nm"] == pytest.approx(an.median_r_eff_nm)
    assert ref["occupancy_percent"] == pytest.approx(an.occupancy_percent)
    assert ref["n_clusters_kept"] == an.n_clusters_kept
    assert ref["ks_p_mc_median"] is None                            # randomization off in the comparison
    assert len(rows) == len(ring_runs) * len(MM.readout_specs())
    wide = MM.wide_view(rows)
    assert [w["method_key"] for w in wide] == [r.key for r in ring_runs]
    assert set(MM.WIDE_HEAD_COLUMNS) <= set(wide[0]) and "occupancy_percent" in wide[0]


def test_plugin_registration_contract(ring_runs):
    seen: Dict[str, Any] = {}

    def plug(run: MM.MethodRun, ctx: MM.AnalysisContext) -> Dict[str, Any]:
        seen[run.key] = (ctx.reference.key if ctx.reference else None, len(ctx.runs), ctx.n_replicates)
        again = ctx.rerun(run, run.analysis.x_slab, run.analysis.y_slab, run.analysis.z_slab)
        same = bool(np.array_equal(again.labels, run.labels))
        p = float(ctx.rng(run).random())
        return {"test_same_labels": int(same), "test_p": p, "test_verdict": "same" if same else "other"}

    specs = [MM.ReadoutSpec("test_same_labels", kind="count"),
             MM.ReadoutSpec("test_p", kind="p_value", tests="test_same_labels"),
             MM.ReadoutSpec("test_verdict", kind="verdict", levels=("same", "other"))]
    try:
        MM.register_analysis("test_plugin", plug, specs)
        with pytest.raises(ValueError):
            MM.register_analysis("test_plugin", plug, specs)                    # name taken
        with pytest.raises(ValueError):
            MM.register_analysis("test_other", plug, ["median_1nn_nm"])         # readout of the core
        with pytest.raises(ValueError):
            MM.ReadoutSpec("Bad Name")
        with pytest.raises(ValueError):
            MM.ReadoutSpec("x_p", kind="probability")
        assert MM.ReadoutSpec("x_p", kind="p_value").alpha == MM.DEFAULT_ALPHA
        rows = MM.apply_analyses(ring_runs, ["test_plugin"], seed=3, n_replicates=999)
        assert all(v == (MM.REFERENCE_KEY, 3, 999) for v in seen.values())
        by = {(r["method_key"], r["readout"]): r for r in rows}
        for run in ring_runs:
            assert by[(run.key, "test_same_labels")]["value"] == 1           # rerun = same method, same labels
            assert by[(run.key, "test_verdict")]["verdict"] == "same"
            p = by[(run.key, "test_p")]
            assert p["verdict"] == (MM.VERDICT_REJECT if p["value"] <= 0.05 else MM.VERDICT_NO_EVIDENCE)
        again = MM.apply_analyses(ring_runs, ["test_plugin"], seed=3)
        assert [r["value"] for r in again] == [r["value"] for r in rows]     # same seed, same draws
        assert MM.describe_column("test_p") is None or isinstance(MM.describe_column("test_p"), str)
    finally:
        MM.unregister_analysis("test_plugin")
    assert "test_plugin" not in MM.registered_analyses()


def test_a_plugin_that_fails_or_cheats_gives_error_rows(ring_runs):
    def boom(run: MM.MethodRun, ctx: MM.AnalysisContext) -> Dict[str, Any]:
        if run.is_reference:
            raise ZeroDivisionError("no")
        if run.method == "FOCAL":
            raise MM.SlowMethodRefused(MM.slow_refusal(run.method, 200))
        return {"test_a": 1.0, "undeclared": 2.0}

    try:
        MM.register_analysis("test_boom", boom, ["test_a"])
        rows = {r["method_key"]: r for r in MM.apply_analyses(ring_runs, ["test_boom"])}
        assert rows[MM.REFERENCE_KEY]["error"] == "ZeroDivisionError: no"
        assert rows["FOCAL"]["error"].startswith("not run: FOCAL is a slow method") and "200 times" in rows["FOCAL"]["error"]
        assert "undeclared" in rows["HDBSCAN"]["error"] and rows["HDBSCAN"]["value"] is None
    finally:
        MM.unregister_analysis("test_boom")


def test_cost_estimate_of_the_plugins(ring_runs):
    try:
        MM.register_analysis("test_cost", lambda r, c: {}, ["test_cost_x"], cost_seconds=lambda run: 2.0)
        assert MM.estimate_analyses_seconds(ring_runs, ["test_cost"]) == 2.0 * len(ring_runs)
        assert MM.estimate_analyses_seconds(ring_runs, [MM.CORE_ANALYSIS]) == 0.0
    finally:
        MM.unregister_analysis("test_cost")


# ----------------------------------------------------------------------------------------------- agreement
def test_identical_labels_agree_perfectly(ring_runs):
    ref = ring_runs[0]
    twin = MM.MethodRun(method="FOCAL", params={}, key="twin", axon_id=ref.axon_id, analysis=ref.analysis,
                        labels=ref.labels)
    rows = MM.pairwise_agreement([ref, twin])
    assert len(rows) == 1
    r = rows[0]
    assert r["method_a"] == MM.REFERENCE_KEY and r["a_is_reference"] is True
    assert r["ari"] == pytest.approx(1.0)
    assert r["n_a"] == r["n_b"] == r["n_one_to_one"] == ref.analysis.n_clusters_kept
    assert r["n_a_split"] == r["n_b_merged"] == r["n_a_unmatched"] == r["n_b_unmatched"] == 0


def test_agreement_between_real_methods(ring_runs):
    rows = MM.pairwise_agreement(ring_runs)
    n = len(ring_runs)
    assert len(rows) == n * (n - 1) // 2
    assert all(r["method_a"] == MM.REFERENCE_KEY for r in rows[:n - 1])
    for r in rows:
        assert r["ari"] is None or -1.0 <= r["ari"] <= 1.0
        assert r["n_one_to_one"] <= r["n_b"]


def _long(axon: str, readout: str, kind: str, values: Dict[str, Any], alpha: float = 0.05) -> List[Dict[str, Any]]:
    spec = MM.ReadoutSpec(readout, kind=kind, alpha=alpha if kind == "p_value" else None)
    return [{"axon_id": axon, "method_key": k, "is_reference": k == MM.REFERENCE_KEY, "analysis_name": "t",
             "readout": readout, "kind": kind, "value": v, "verdict": spec.verdict(v)} for k, v in values.items()]


def test_conclusion_summary_rules():
    rows = (_long("a", "x_p", "p_value", {MM.REFERENCE_KEY: 0.01, "FOCAL": 0.30, "HDBSCAN": 0.02})
            + _long("a", "y_p", "p_value", {MM.REFERENCE_KEY: 0.01, "FOCAL": 0.04})
            + _long("a", "z_p", "p_value", {MM.REFERENCE_KEY: None, "FOCAL": 0.04})
            + _long("a", "v_nm", "value", {MM.REFERENCE_KEY: 100.0, "FOCAL": 120.0, "HDBSCAN": 95.0})
            + _long("a", "w_nm", "value", {MM.REFERENCE_KEY: 100.0, "FOCAL": 105.0})
            + _long("a", "q_verdict", "verdict", {MM.REFERENCE_KEY: "regular", "FOCAL": "regular"}))
    s = {r["readout"]: r for r in MM.conclusion_summary(rows)}
    assert s["x_p"]["depends_on_method"] == "yes" and s["x_p"]["methods_disagreeing"] == "FOCAL"
    assert s["x_p"]["verdicts_agree"] is False and s["x_p"]["reference_verdict"] == MM.VERDICT_REJECT
    assert s["y_p"]["depends_on_method"] == "no" and s["y_p"]["verdicts_agree"] is True
    assert s["z_p"]["depends_on_method"] == "undetermined"
    v = s["v_nm"]
    assert (v["min_value"], v["min_method"], v["max_value"], v["max_method"]) == (95.0, "HDBSCAN", 120.0, "FOCAL")
    assert v["range_rel_reference"] == pytest.approx(0.25) and v["large_spread"] is True
    assert v["depends_on_method"] == "spread only"
    assert s["w_nm"]["large_spread"] is False
    assert s["q_verdict"]["depends_on_method"] == "no"
    over = {r["readout"]: r for r in MM.summary_over_axons(list(s.values()))}
    assert over["x_p"]["n_axons_depends"] == 1 and over["v_nm"]["n_axons_large_spread"] == 1


# ----------------------------------------------------------------------------------------------- CSV
def test_csv_round_trip_and_never_overwrite(ring_runs, tmp_path):
    rows = MM.apply_analyses(ring_runs)
    path = str(tmp_path / "comparison.csv")
    p1 = MM.write_long_csv(path, rows)
    assert p1 == path
    back = MM.read_long_csv(p1)
    assert len(back) == len(rows)
    for a, b in zip(rows, back):
        for c in MM.LONG_COLUMNS:
            if isinstance(a[c], float):
                assert b[c] == pytest.approx(a[c], rel=0, abs=0), c          # repr round-trips a float exactly
            else:
                assert b[c] == a[c], (c, a[c], b[c])
    before = open(p1, encoding="utf-8").read()
    p2 = MM.write_long_csv(path, rows[:3])
    assert p2 == str(tmp_path / "comparison_2.csv") and open(p1, encoding="utf-8").read() == before
    ag = MM.pairwise_agreement(ring_runs)
    p3 = MM.write_rows_csv(str(tmp_path / "agreement.csv"), ag, MM.AGREEMENT_COLUMNS)
    back_ag = MM.read_rows_csv(p3)
    assert back_ag[0]["n_a"] == ag[0]["n_a"] and back_ag[0]["a_is_reference"] is True
    with pytest.raises(ValueError):
        MM.read_long_csv(p3)
    summ = MM.conclusion_summary(rows)
    p4 = MM.write_rows_csv(str(tmp_path / "summary.csv"), summ, MM.SUMMARY_COLUMNS)
    assert len(MM.read_rows_csv(p4)) == len(summ)


def test_every_column_is_described():
    for cols in (MM.LONG_COLUMNS, MM.WIDE_HEAD_COLUMNS, MM.AGREEMENT_COLUMNS, MM.SUMMARY_COLUMNS,
                 MM.OVER_AXONS_COLUMNS):
        assert not [c for c in cols if not MM.describe_column(c)]
    for spec in MM.readout_specs():
        assert MM.describe_column(spec.name)


def test_module_imports_without_qt_at_import_time():
    import subprocess
    import sys
    code = ("import sys; import tools.multimethod; "
            "print('PyQt5' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.abspath(__file__)))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
