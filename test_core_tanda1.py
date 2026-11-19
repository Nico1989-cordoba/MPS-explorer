# -*- coding: utf-8 -*-
"""
The headless fixes after the design review (branch fix/core-batch-tanda1), on SYNTHETIC data only (rule R8):

1. G1 (c): failed replicates of the pipeline null counted in the worst case, ``null_failed_replicates`` and the
   ``*_ignoring_failed`` p values; more than 5 % failed -> NOT INTERPRETABLE.

(The later sections of this file test the other tasks of the branch.)

Run:  py -3 -m pytest test_core_tanda1.py -q
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import mps_gap_measures as gm  # noqa: E402
from tools import mps_pipeline_null as pn  # noqa: E402
from tools.mps_randomization import monte_carlo_p  # noqa: E402

SMALL = (1000.0, 700.0)          # perimeter ~5.4 um: ~20 molecules per ring (as test_pipeline_null)


@pytest.fixture(scope="module")
def observed_small():
    model = pn.synthetic_model(SMALL)
    an, ax = pn.observe_synthetic(model, 3)
    return model, an, ax


# ===================================================================================================== 1. G1 (c)

def test_tolerated_failures_are_five_percent_rounded_down() -> None:
    assert pn.max_tolerated_failed(49) == 2 and pn.max_tolerated_failed(199) == 9
    assert pn.max_tolerated_failed(19) == 0 and pn.max_tolerated_failed(20) == 1 and pn.max_tolerated_failed(40) == 2


def test_worst_case_p_counts_each_failure_as_extreme_in_every_tail() -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(size=48)
    for obs in (-2.5, -0.3, 0.0, 0.4, 3.0):
        for two in (False, True):
            # no failure: exactly the rank p of tools.mps_randomization
            assert pn.monte_carlo_p_worst_case(obs, v, two_sided=two, n_failed=0) == monte_carlo_p(obs, v, two)
        b_hi = int(np.count_nonzero(v >= obs))
        b_lo = int(np.count_nonzero(v <= obs))
        f = 2
        hi = (1 + b_hi + f) / (48 + f + 1)
        lo = (1 + b_lo + f) / (48 + f + 1)
        assert pn.monte_carlo_p_worst_case(obs, v, two_sided=False, n_failed=f) == pytest.approx(hi)
        assert pn.monte_carlo_p_worst_case(obs, v, two_sided=True, n_failed=f) == pytest.approx(min(1, 2 * min(lo, hi)))
        # conservative: never below the p that ignores the failures
        for two in (False, True):
            assert pn.monte_carlo_p_worst_case(obs, v, two_sided=two, n_failed=f) >= monte_carlo_p(obs, v, two)
    # the most extreme observed: 1/(B+1) without failures, (1 + f)/(B + f + 1) with them
    assert pn.monte_carlo_p_worst_case(99.0, v, two_sided=False, n_failed=2) == pytest.approx(3 / 51)
    assert pn.monte_carlo_p_worst_case(None, v, two_sided=True, n_failed=2) is None
    assert pn.monte_carlo_p_worst_case(1.0, [], two_sided=True, n_failed=2) is None


def test_global_envelope_p_worst_case_algebra() -> None:
    # p = (1 + b) / (n + 1) with b = 4 of n = 47 curves; 2 failures -> (1 + 4 + 2) / (47 + 2 + 1)
    p = (1 + 4) / 48
    assert pn.p_global_worst_case(p, 47, 2) == pytest.approx(7 / 50)
    assert pn.p_global_worst_case(p, 47, 0) == p
    assert pn.p_global_worst_case(None, 47, 2) is None


def test_gap_summary_counts_failures_in_the_tested_tail() -> None:
    v = list(np.linspace(0.2, 0.8, 47))
    for tail, two in (("two-sided", True), ("upper", False)):
        s = gm.summarise("x", 0.55, v, tail, n_failed=2)
        assert s.p == pytest.approx(pn.monte_carlo_p_worst_case(0.55, v, two_sided=two, n_failed=2))
        assert s.p_ignoring_failed == pytest.approx(monte_carlo_p(0.55, np.array(v), two))
        assert s.n_failed == 2 and s.n == 47
    low = gm.summarise("x", 0.25, v, "lower", n_failed=2)
    assert low.p == pytest.approx(pn.monte_carlo_p_worst_case(-0.25, -np.array(v), two_sided=False, n_failed=2))
    assert low.p > low.p_ignoring_failed


def _fail_some(original: Any, failing: set) -> Any:
    def run(model: Any, spec: Any, seed: int, index: int, *args: Any, **kw: Any) -> Any:
        out = original(model, spec, seed, index, *args, **kw)
        if index in failing:
            nan = np.full(np.asarray(args[0]).size, np.nan)
            return pn.ReplicateOutcome(index, False, "generator: jammed (test)", {}, nan, nan, 0, 0, float("nan"), 0.0)
        return out
    return run


def test_null_with_tolerated_failures_counts_them_and_exports_both_p(observed_small, monkeypatch) -> None:
    _model, an, _ax = observed_small
    monkeypatch.setattr(pn, "_run_replicate", _fail_some(pn._run_replicate, {3}))
    res = pn.run_pipeline_null(an, n_replicates=20, seed=11)
    assert res.n_failed == 1 and res.max_failed == 1 and res.n_accepted == 19
    assert not any(r.startswith("Only ") for r in res.not_interpretable_reasons())
    assert any("tolerated" in w for w in res.warnings)
    d = res.export_dict()
    assert d["null_failed_replicates"] == 1
    for name, spec in res.readout_specs.items():
        if name not in res.observed or res.p_values.get(name) is None:
            continue
        vals = np.array([r.get(name) for r in res.replicate_readouts if r.get(name) is not None], dtype=float)
        obs = res.observed[name]
        assert res.p_values[name] == pytest.approx(
            pn.monte_carlo_p_worst_case(obs, vals, two_sided=spec.two_sided, n_failed=1))
        assert res.p_values_ignoring_failed[name] == pytest.approx(monte_carlo_p(obs, vals, spec.two_sided))
        assert d[f"p_mc_{name}"] >= d[f"p_mc_{name}_ignoring_failed"]
    for tag in ("ripley1d", "ripley2d"):
        if d[f"{tag}_p_global"] is not None:
            assert d[f"{tag}_p_global"] >= d[f"{tag}_p_global_ignoring_failed"]
    assert "counted as at least as extreme" in res.headline() or not res.interpretable
    # more than 5 % failed: NOT INTERPRETABLE, with the reason
    worse = dataclasses.replace(res, n_requested=22)            # 3 failed of 22 (> floor(1.1) = 1)
    assert worse.n_failed == 3
    assert any(r.startswith("Only 19 of the 22") and "at most 1" in r for r in worse.not_interpretable_reasons())
    assert not worse.interpretable


def test_size_curvature_readout_e5_goes_through_the_worst_case_rule(observed_small, monkeypatch) -> None:
    # integration 3 (2026-10-29): E5's readout (feat/validations-d2-e1-e5) rides the same null, so G1 (c) applies to
    # it like to every readout: the failure counted as extreme, and its _ignoring_failed companion exported
    from tools import clustering_integration as ci
    from tools import size_curvature_null as scn
    from tools import size_plugins as sp
    sp.register_with_pipeline_null(replace=True)
    _model, an, _ax = observed_small
    monkeypatch.setattr(pn, "_run_replicate", _fail_some(pn._run_replicate, {2}))
    name = scn.NULL_STATISTIC
    res = pn.run_pipeline_null(an, n_replicates=20, seed=13, readouts=[name, "size_nloc_nn_spearman"])
    assert res.n_failed == 1
    d = res.export_dict()
    assert f"p_mc_{name}_ignoring_failed" in d
    if res.observed.get(name) is not None and res.p_values.get(name) is not None:
        vals = np.array([r.get(name) for r in res.replicate_readouts if r.get(name) is not None], dtype=float)
        assert res.p_values[name] == pytest.approx(
            pn.monte_carlo_p_worst_case(res.observed[name], vals, two_sided=True, n_failed=1))
        assert d[f"p_mc_{name}"] >= d[f"p_mc_{name}_ignoring_failed"]
    assert f"p_mc_{name}_ignoring_failed" in ci.v15_columns()


def test_integration_block_places_the_failure_columns() -> None:
    from tools import clustering_integration as ci
    from tools import column_dictionary as cd
    cols = ci.integration_columns()
    for c in ci.NULL_FAILED_COLUMNS:
        assert c in cols and cd.describe(c), c
    assert "p_mc_gap_cv_ignoring_failed" in ci.NULL_FAILED_COLUMNS
    assert ci.provisional("G1") == pn.TOLERATED_FAILED_FRACTION == gm.MATCHED_TOLERATED_FAILED_FRACTION


# ===================================================================================================== 2. tau (D4)

def _tolerances_and_arc(tau_nm: float) -> Any:
    from types import SimpleNamespace
    from tools import mps_eclipse_tolerance as et
    rng = np.random.default_rng(4)
    groups, x, y = [], [], []
    for k in range(12):
        a = 2.0 * math.pi * k / 12
        cx, cy = 900.0 * math.cos(a), 900.0 * math.sin(a)
        start = len(x)
        x.extend(rng.normal(cx, 8.0, 30).tolist())
        y.extend(rng.normal(cy, 8.0, 30).tolist())
        groups.append(np.arange(start, start + 30))
    tol = et.axon_tolerances(np.array(x), np.array(y), {0: groups})
    joint = SimpleNamespace(pairs=[(0, 1)], n_matched_obs=np.array([5]), null_n_matched=np.array([[3], [4], [2], [5]]),
                            min_K=np.array([10]))
    arc = SimpleNamespace(tau_nm=float(tau_nm), joint=joint, adjacent=[])
    return et, tol, arc


@pytest.mark.parametrize("tau, role", [(31.0, "primary"), (31.004, "primary"), (62.0, "exploratory"),
                                       (40.0, "exploratory"), (25.0, "exploratory")])
def test_any_tau_but_31_is_exploratory_in_text_and_export(tau: float, role: str) -> None:
    et, tol, arc = _tolerances_and_arc(tau)
    assert et.scalar_tau_role(tau, tol.inputs) == role
    row = et.axon_row(tol, arc, None)
    assert row["tau_scalar_role"] == role and "tau_scalar_role" in et.AXON_COLUMNS and et.describe("tau_scalar_role")
    text = "\n".join(et.summary_lines(tol, arc, None))
    if role == "primary":
        assert "PRIMARY tau = " in text and "EXPLORATORY tau" not in text
    else:
        assert "PRIMARY tau" not in text                              # the bug: "PRIMARY tau = 62 nm"
        assert f"EXPLORATORY tau = {tau:.1f} nm (not the pre-specified tau_0 = 31 nm)" in text


# ===================================================================================================== batch helpers

def _write_synthetic_file(folder: Any, name: str, seed: int, semi_axes: tuple = SMALL) -> str:
    """One synthetic axon with blinking (the pipeline null's own generator: molecules on an ellipse, labels, cycles
    of localizations in consecutive frames, background, three rings in z) written as a localization CSV."""
    model = pn.synthetic_model(semi_axes)
    ax = pn.simulate_null_axon(model, np.random.default_rng(seed))
    path = os.path.join(str(folder), name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("id,frame,x [nm],y [nm],z [nm],uncertainty_xy [nm],intensity [photon]\n")
        for i in range(ax.n_locs):
            fh.write(f"{i + 1},{int(ax.frame[i])},{ax.x_nm[i] + 3000.0:.2f},{ax.y_nm[i] + 3000.0:.2f},"
                     f"{ax.z_nm[i]:.2f},{float(ax.lp_nm[i]):.2f},1000\n")
    return path


def _batch_settings(c_label: float = 14.0) -> Any:
    from tools.mps_settings import MPSSettings
    s = MPSSettings()
    s.randomization_params = {"n_iterations": 19, "seed": 0}          # [ad hoc: a short test]
    s.c_label = c_label
    return s


@pytest.fixture(scope="module")
def batch_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("batch")
    _write_synthetic_file(folder, "synth_a.csv", 21)
    _write_synthetic_file(folder, "synth_b.csv", 22)
    return folder


# ===================================================================================================== 3. c_label

def test_c_label_is_kept_with_the_settings(tmp_path) -> None:
    from tools.mps_settings import MPSSettings, clean_c_label, load_settings, save_settings
    s = MPSSettings()
    assert s.c_label == 14.0
    s.c_label = 26.0
    assert save_settings(s, str(tmp_path))
    assert load_settings(str(tmp_path)).c_label == 26.0
    for bad in (500.0, 0.0, "auto", float("nan"), True):
        assert clean_c_label(bad) == 14.0
    assert clean_c_label("19.5") == 19.5


def test_the_batch_uses_and_records_the_edited_c_label(batch_folder) -> None:
    from tools import mps_batch as mb
    settings = mb.BatchSettings.from_settings(_batch_settings(26.0))
    assert settings.c_label == 26.0 and "c_label 26 cycles per label" in settings.describe()
    plan = mb.plan_batch(str(batch_folder))
    item = next(f for f in plan.files if f.path.endswith("synth_a.csv"))
    rec = mb.analyse_file(item, settings)
    assert rec.ok, rec.error
    assert rec.row["c_label_used"] == 26.0 and "typed" not in str(rec.row["c_label_source"])   # 26 is the beta-ME preset
    default = mb.analyse_file(item, mb.BatchSettings.from_settings(_batch_settings()))
    assert default.row["c_label_used"] == 14.0
    # the c_label reaches the analyses: the size plug-in's labels per cluster scales as 1 / c_label
    a, b = rec.row.get("size_labels_per_cluster_est"), default.row.get("size_labels_per_cluster_est")
    if a is not None and b is not None and a > 1.0 and b > 1.0:
        assert a < b


# ===================================================================================================== 4. never overwrite

def test_excel_copy_batch_log_and_comparison_never_overwrite(tmp_path) -> None:
    from tools import mps_batch as mb
    from tools.mps_identity import AxonIdentity
    from tools.results_table import excel_copy
    table = tmp_path / "axons.csv"
    table.write_text("axon_id,value\na,1.5\n", encoding="utf-8")
    first = excel_copy(str(table))
    table.write_text("axon_id,value\na,1.5\nb,2.25\n", encoding="utf-8")
    second = excel_copy(str(table))
    assert first.endswith("axons_for_excel.csv") and second.endswith("axons_for_excel_2.csv")
    assert "1,5" in open(first, encoding="utf-8-sig").read() and "2,25" not in open(first, encoding="utf-8-sig").read()
    # the batch log: a second batch exported to the same table keeps the first batch's log
    rec = mb.AxonRecord(source="x.csv", identity=AxonIdentity(), error="could not be read: test")
    log1 = mb.write_log([rec], str(table))
    log2 = mb.write_log([rec], str(table))
    assert log1.endswith("axons_batch_log.csv") and log2.endswith("axons_batch_log_2.csv")
    # the comparison table
    rows = [{"source": "x.csv", "cluster_method": "DBSCAN"}]
    p1 = mb.write_methods_comparison_file(rows, mb.comparison_path(str(table)))
    assert mb.write_methods_comparison(rows, mb.comparison_path(str(table))) == 1
    assert p1.endswith("axons_methods_comparison.csv") and os.path.exists(str(tmp_path / "axons_methods_comparison_2.csv"))


# ===================================================================================================== 5-6. batch comparison

@pytest.fixture(scope="module")
def batch_comparison(batch_folder):
    from tools import mps_batch as mb
    settings = mb.BatchSettings.from_settings(_batch_settings())
    plan = mb.plan_batch(str(batch_folder))
    items = sorted(plan.files, key=lambda f: f.path)
    records = mb.run_batch(items, settings)
    results: List[Any] = []
    rows = mb.compare_methods(items[:1], settings, ["DBSCAN", "HDBSCAN"], analyses=["per_axon"],
                              results_out=results)
    return records, rows, results


def test_comparison_axon_id_is_the_axon_tables_own(batch_comparison, tmp_path) -> None:
    from tools import mps_batch as mb
    records, rows, results = batch_comparison
    table_ids = {r.row["axon_id"] for r in records if r.ok}
    assert rows and all(r["axon_id"] in table_ids for r in rows)                 # not the full path
    assert all(os.sep not in str(r["axon_id"]) and "/" not in str(r["axon_id"]) for r in rows)
    assert results[0].axon_id in table_ids
    assert all(lr["axon_id"] in table_ids for lr in results[0].long_rows)
    written = mb.write_methods_comparison_long(results, str(tmp_path / "axons.csv"))
    import csv as _csv
    with open(written["long"], encoding="utf-8") as fh:
        ids = {r["axon_id"] for r in _csv.DictReader(fh)}
    assert ids and ids <= table_ids


def test_batch_rows_methods_block_filled_or_not_run_with_reason(batch_comparison) -> None:
    from tools import clustering_integration as ci
    from tools import mps_batch as mb
    records, _rows, results = batch_comparison
    rows = [dict(r.row) for r in records if r.ok]
    assert len(rows) == 2
    for r in rows:                                    # no comparison: 'not run' with the reason, never empty
        for name in mb.METHODS_BLOCK:
            assert r[name] == mb.BATCH_METHODS_NOT_RUN
    n = mb.fill_methods_columns(rows, results, "/somewhere/batch_methods_long.csv")
    assert n == 1                                    # only the first file was compared
    done = next(r for r in rows if r["axon_id"] == results[0].axon_id)
    other = next(r for r in rows if r["axon_id"] != results[0].axon_id)
    assert "DBSCAN" in done["methods_compared"] and "HDBSCAN" in done["methods_compared"]
    assert done["methods_long_file"] == "batch_methods_long.csv"   # whatever file name it is given
    assert done["methods_depends_on_method"] not in (None, "")
    assert other["methods_compared"] == mb.BATCH_METHODS_NOT_REACHED
    rows2 = [dict(r.row) for r in records if r.ok]
    mb.fill_methods_columns(rows2, results, "")
    assert next(r for r in rows2 if r["axon_id"] == results[0].axon_id)["methods_long_file"] == \
        ci.METHODS_TABLES_NOT_WRITTEN


# ===================================================================================================== 7. stage-2 level

def test_regularity_rows_say_their_stage2_level_and_stage1_is_the_same() -> None:
    from tools import clustering_integration as ci
    # a planted lattice (320 nm, 5 nm jitter) on a larger ring: stage 1 says "more regular" (p 0.025 with B = 39,
    # the smallest two-sided p), so stage 2 runs at each level
    model = pn.synthetic_model((1400.0, 1000.0))
    length = float(model.membrane_length_nm())
    rng = np.random.default_rng(8)
    arcs = [pn.planted_arcs("lattice", length, 0, rng, lattice_nm=320.0, jitter_nm=5.0) for _ in range(pn.N_RINGS)]
    an, _ = pn.observe_synthetic(model, 5, molecule_arcs=arcs)
    null = pn.run_pipeline_null(an, n_replicates=39, seed=2)
    small = {"steps": 2, "n_bisect": 4}
    rc = gm.analyze_gap_regularity(an, null_result=null, stage2="centres", n_matched=19, seed=2)
    rp = gm.analyze_gap_regularity(an, null_result=null, stage2="pipeline", n_matched=19, seed=2, matched_kw=small)
    dc, dp = rc.export_columns(), rp.export_columns()
    assert dc["regularity_stage2_level"] == "centres" and dp["regularity_stage2_level"] == "pipeline"
    for c in ("gap_cv", "gap_cv_p_mc", "gap_cv_z", "gap_excess_cv_p_mc", "gap_rho1_z", "bragg_s_z"):
        assert dc[c] == dp[c], c                                     # stage 1: one null, one answer
    assert dc["matched_null_kind"] == "centres_approx" and dp["matched_null_kind"] == "pipeline"   # stage 2 ran
    assert "regularity_stage2_level" in ci.integration_columns()
    from tools import column_dictionary as cd
    assert cd.describe("regularity_stage2_level")


# ===================================================================================================== 8. one footprint pool

def _synthetic_clusters(labels_mean: float, seed: int) -> Any:
    from tools import mps_apparent_size as asz
    from tools import mps_size_synth as syn
    rng = np.random.default_rng(seed)
    c = syn.make_contour("circle", 1400.0, rng)
    ring = syn.generate_ring(rng, c, labels_mean=labels_mean, rho=25.0, K=60)
    return asz.cluster_sizes(ring.x, ring.y, ring.truth, event_id=ring.event, curve=ring.contour.xy), syn.SIGMA_LOC_NM


@pytest.mark.parametrize("labels_mean, e2", [(1.0, True), (3.0, True), (3.0, False), (5.0, False)])
def test_size_and_eclipse_pool_the_same_clusters_to_the_same_number(labels_mean: float, e2: bool) -> None:
    """Finding (q): one definition (tools.footprint_pool) -> one labels-per-cluster, one verdict, one footprint."""
    from tools import footprint_pool as fp
    from tools import mps_apparent_size as asz
    from tools import mps_eclipse_tolerance as et
    cs, nena = _synthetic_clusters(labels_mean, 31)
    size = asz.pooled_footprint(cs.sd_tangential, cs.n_locs, cs.n_events, nena, exclude_event_fusion=e2)
    inp = et.ToleranceInputs(sigma_loc_nm=nena, exclude_event_fusion_from_pool=e2)
    ecl, _lam = et.pool_footprint(cs.sd_tangential, cs.n_locs, cs.n_events, np.zeros(cs.n_clusters, bool), inp)
    assert size.labels_per_cluster == pytest.approx(ecl.n_label_mean_hat)
    assert (size.verdict == fp.REGIME_LEVELS[0]) == ecl.identifiable
    if ecl.identifiable:
        assert size.rho_nm == pytest.approx(ecl.rho_pooled_nm) and size.se_nm == pytest.approx(ecl.rho_pooled_se_nm)
        assert size.n_used == ecl.n_used
    assert et.MIN_MEAN_LABELS_FOR_POOL == asz.IDENTIFIABLE_LABELS == fp.IDENTIFIABLE_LABELS == 1.5
    assert et.MIN_POOL_CLUSTERS == asz.POOL_MIN_CLUSTERS == fp.MIN_POOL_CLUSTERS


def test_e2_the_event_layers_fusions_leave_the_pool_and_its_label_filter() -> None:
    """E2 (user decision 2026-10-23), implemented in the shared pool: the probable fusions (events / c_label > 2)
    leave the pool AND the labels-per-cluster average that gates it. A one-label axon whose fused clusters lifted
    the old eclipse average over every cluster stays 'content only'."""
    from tools import footprint_pool as fp
    cs, nena = _synthetic_clusters(1.0, 32)
    ev = np.asarray(cs.n_events, dtype=float).copy()
    ev[:12] = ev[:12] * 3.0 + 40.0                      # 12 clusters made to look fused (3+ labels' worth of events)
    on = fp.footprint_pool(cs.sd_tangential, cs.n_locs, ev, nena, sigma_link_nm=10.0, c_label=14.0)
    off = fp.footprint_pool(cs.sd_tangential, cs.n_locs, ev, nena, sigma_link_nm=10.0, c_label=14.0,
                            exclude_event_fusion=False)
    assert on.n_excluded_fusion >= 10 and off.n_excluded_fusion == 0
    assert on.labels_per_cluster < 1.5 <= off.labels_per_cluster          # without E2 the fusions pass the filter
    assert on.verdict == "content only"
    # the integration turns it on for both modules with one switch
    from tools import clustering_integration as ci
    ex = ci.comparison_extras(nena_nm=12.0)
    assert ex["size_exclude_event_fusion"] is True and ex["eclipse_inputs"].exclude_event_fusion_from_pool is True


def test_e2_consequence_with_several_labels_per_cluster_is_recorded() -> None:
    """Evidence for the report (an open question for the user): the event layer's mark is ABSOLUTE (2 labels' worth
    of events), so with ~3 labels per cluster E2 removes most real clusters and the pool falls to 'content only';
    the estimator without E2 recovers the footprint (25 nm, within the research's +-6 nm)."""
    from tools import footprint_pool as fp
    cs, nena = _synthetic_clusters(3.0, 33)
    on = fp.footprint_pool(cs.sd_tangential, cs.n_locs, cs.n_events, nena, sigma_link_nm=10.0, c_label=14.0)
    off = fp.footprint_pool(cs.sd_tangential, cs.n_locs, cs.n_events, nena, sigma_link_nm=10.0, c_label=14.0,
                            exclude_event_fusion=False)
    assert on.n_excluded_fusion > cs.n_clusters // 3 and not on.measurable
    assert off.measurable and abs(float(off.rho_nm) - 25.0) < 8.0


# ===================================================================================================== 9. size stats in the Batch

def _size_rows(seed: int = 3) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for g, shift in (("KO", 6.0), ("WT", 0.0)):
        for a in range(4):
            animal_effect = rng.normal(0, 1.0)
            for _ax in range(3):
                rows.append({"animal": f"{g}{a}", "genotype": g,
                             "size_rho_pool_nm": 25.0 + shift + animal_effect + rng.normal(0, 1.5),
                             "size_cloud_sd_median_nm": 14.0 + rng.normal(0, 1.0),
                             "size_r_rare_median_nm": 30.0 + rng.normal(0, 2.0), "size_radius_gyration_median_nm": None,
                             "size_n_events_median": 30.0 + rng.normal(0, 3.0),
                             "size_n_locs_median": 60.0 + rng.normal(0, 5.0), "size_nena_nm": 10.0 + rng.normal(0, 1.0),
                             "size_nloc_nn_fisher_z": 0.2 + rng.normal(0, 0.2), "size_content_kappa_r_fisher_z": None})
    rows.append({"animal": "", "genotype": "WT", "size_rho_pool_nm": 99.0})       # no animal: left out
    return rows


def test_size_stats_reach_the_batch_with_their_columns(tmp_path) -> None:
    from tools import clustering_integration as ci
    from tools import column_dictionary as cd
    from tools import mps_size_stats as ss
    rows = _size_rows()
    summ = ci.size_batch_summary(rows)
    out = summ["rows"]
    assert all(set(r) == set(ci.SIZE_STATS_COLUMNS) for r in out)
    pool = next(r for r in out if r["size_readout"] == "size_rho_pool_nm")
    assert pool["group_a"] == "KO" and pool["group_b"] == "WT" and pool["n_animals"] == 8 and pool["n_axons"] == 24
    direct = ss.compare_groups([r["size_rho_pool_nm"] for r in rows[:-1]], [r["genotype"] for r in rows[:-1]],
                               [r["animal"] for r in rows[:-1]], estimator_kind="pool")
    assert pool["animal_p"] == pytest.approx(direct.animal_p) and pool["animal_diff"] == pytest.approx(-6.0, abs=3.0)
    cloud = next(r for r in out if r["size_readout"] == "size_cloud_sd_median_nm")
    assert cloud["covariates"] == "nena_nm, log_n_locs"
    za = [r for r in out if r["size_stat"] == "combined z" and r["size_readout"] == "size_nloc_nn_fisher_z"]
    assert {r["group"] for r in za} == {"all", "KO", "WT"}
    allz = next(r for r in za if r["group"] == "all")
    assert allz["axon_n"] == 24 and allz["animal_n"] == 8 and "not against the pipeline null" in allz["size_stat_warnings"]
    zb = [r for r in out if r["size_readout"] == "size_content_kappa_r_fisher_z"]
    assert all(r["axon_n"] is None and "no axon" in r["size_stat_warnings"] for r in zb)
    for c in ci.SIZE_STATS_COLUMNS:
        assert cd.describe(c), c
    p1 = ci.write_size_summary(str(tmp_path / "axons.csv"), summ)["size_stats"]
    p2 = ci.write_size_summary(str(tmp_path / "axons.csv"), summ)["size_stats"]
    assert p1.endswith("axons_size_stats.csv") and p2.endswith("axons_size_stats_2.csv")
    one = ci.size_batch_summary([r for r in rows if r.get("genotype") == "KO"])
    assert all(r["size_stat_warnings"].startswith("needs exactly two genotypes")
               for r in one["rows"] if r["size_stat"] == "group comparison")


# ===================================================================================================== 10. lpc by depth (B3 c)

def _events_in_z(rng: np.random.Generator, n: int, z_centre: float, mean_locs: float, frame0: int) -> Any:
    """n separate events (one blink each, consecutive frames, 3 nm scatter) with geometric sizes of the given mean at
    depth z_centre: the localizations, frames and z of a synthetic movie."""
    xs, ys, zs, fs = [], [], [], []
    f = frame0
    for i in range(n):
        k = int(rng.geometric(1.0 / mean_locs))
        x0, y0 = (i % 100) * 500.0, (i // 100) * 500.0
        xs.append(x0 + rng.normal(0, 3.0, k)); ys.append(y0 + rng.normal(0, 3.0, k))
        zs.append(z_centre + rng.normal(0, 10.0, k)); fs.append(np.arange(f, f + k))
        f += k + 3
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), np.concatenate(fs), f


def test_lpc_profile_by_depth_finds_a_slab_that_differs() -> None:
    from tools import lpc_depth
    rng = np.random.default_rng(10)
    xa, ya, za, fa, f = _events_in_z(rng, 800, 0.0, 3.0, 5)            # the slab: 3 localizations per event
    xb, yb, zb, fb, f = _events_in_z(rng, 800, 400.0, 1.5, f + 10)     # 400 nm away: 1.5
    x, y, z, fr = (np.concatenate(v) for v in ((xa, xb), (ya, yb), (za, zb), (fa, fb)))
    prof = lpc_depth.lpc_depth_profile(fr, x, y, z, nena_nm=3.0, slab=(-90.0, 90.0), n_boot=199)
    assert 700 < prof.n_slab < 900                                      # a few events split by the radius
    assert prof.lpc_slab == pytest.approx(3.0, rel=0.12) and prof.lpc_outside == pytest.approx(1.5, rel=0.12)
    assert prof.differs is True and prof.diff_ci[0] > 0 and prof.warnings
    slab_band = next(b for b in prof.bands if b.is_slab)
    assert slab_band.z_lo_nm == -90.0 and slab_band.z_hi_nm == 90.0 and slab_band.mean == pytest.approx(prof.lpc_slab)
    far = [b for b in prof.bands if b.z_lo_nm <= 400.0 < b.z_hi_nm][0]
    assert far.mean == pytest.approx(1.5, rel=0.12) and far.ci_low < far.mean < far.ci_high
    cols = lpc_depth.export_columns(prof)
    assert cols["lpc_slab_differs"] is True and "*-90..90 nm" in cols["lpc_z_profile"] and ";" not in cols["lpc_z_profile"]
    # the same sizes at every depth: no flag
    xc, yc, zc, fc, _ = _events_in_z(rng, 800, 400.0, 3.0, f + 10)
    same = lpc_depth.lpc_depth_profile(np.concatenate([fa, fc]), np.concatenate([xa, xc]), np.concatenate([ya, yc]),
                                       np.concatenate([za, zc]), nena_nm=3.0, slab=(-90.0, 90.0), n_boot=199)
    assert same.differs is False and not same.warnings


def test_lpc_profile_columns_reach_the_batch_row(batch_folder) -> None:
    from tools import clustering_integration as ci
    from tools import column_dictionary as cd
    from tools import mps_batch as mb
    from tools.lpc_depth import LPC_DEPTH_COLUMNS
    plan = mb.plan_batch(str(batch_folder))
    item = next(f for f in plan.files if f.path.endswith("synth_a.csv"))
    rec = mb.analyse_file(item, mb.BatchSettings.from_settings(_batch_settings()))
    assert rec.ok, rec.error
    assert rec.row["lpc_file"] is not None and rec.row["lpc_slab_n_events"] > 0
    assert rec.row["lpc_z_profile"] and rec.row["lpc_slab_differs"] in (True, False, None)
    # the rule's own value is unchanged and still the whole file's (B3)
    assert rec.row["loc_per_cycle_used"] is not None
    for c in LPC_DEPTH_COLUMNS:
        assert c in ci.integration_columns() and cd.describe(c), c
    # the axon window does not compute it (it needs the whole file): the note says so
    from tools import mps_pipeline_null as _pn
    an, _ = _pn.observe_synthetic(_pn.synthetic_model(SMALL), 3)
    w = ci.axon_columns(an, nena_nm=12.0)
    assert w["lpc_slab"] is None and w["lpc_depth_note"] == ci.LPC_DEPTH_NOT_HERE


# ===================================================================================================== 11. null closures (C3 / C4)

def test_null_closure_choices_and_cost_estimate() -> None:
    from types import SimpleNamespace
    from tools import clustering_integration as ci
    from tools import mps_batch as mb
    from tools.mps_settings import MPSSettings
    assert ci.null_closures("n_locs") == ["n_locs"] and ci.null_closures("k") == ["k_clusters"]
    assert ci.null_closures("both") == ["n_locs", "k_clusters"]
    with pytest.raises(ValueError):
        ci.null_closures("molecules")
    s = MPSSettings()
    assert s.batch_null_closure == "n_locs"
    s.batch_null_closure = "nonsense"
    assert s.validate().batch_null_closure == "n_locs"
    s.batch_null_closure = "both"
    assert mb.BatchSettings.from_settings(s).null_closure == "both"
    run: Any = SimpleNamespace(method="DBSCAN", analysis=SimpleNamespace(n_locs_slab=3000))
    one = ci.null_estimate_seconds(run, 199, "n_locs", "centres")
    k = ci.null_estimate_seconds(run, 199, "k", "centres")
    both = ci.null_estimate_seconds(run, 199, "both", "centres")
    assert k > one and both == pytest.approx(one + k)                  # the K closure's pilots run the pipeline
    assert ci.null_estimate_seconds(run, 199, "n_locs", "pipeline") > one   # stage 2 at the pipeline level counted
    assert mb.null_cost_per_file("DBSCAN", 3000, 199, "both") > mb.null_cost_per_file("DBSCAN", 3000, 199, "n_locs")


def test_batch_both_closures_one_row_set_each(batch_folder, tmp_path) -> None:
    import csv as _csv
    from tools import clustering_integration as ci
    from tools import mps_batch as mb
    s = _batch_settings()
    s.batch_null_closure = "both"
    settings = mb.BatchSettings.from_settings(s)
    plan = mb.plan_batch(str(batch_folder))
    item = next(f for f in plan.files if f.path.endswith("synth_a.csv"))
    rec = mb.analyse_file(item, settings, null_replicates=19)
    assert rec.ok, rec.error
    assert [r["null_closure"] for r in rec.null_rows] == ["n_locs", "k_clusters"]
    assert rec.row["null_closure"] == "n_locs"                          # the axon table's row: the default closure
    for r in rec.null_rows:
        assert r["axon_id"] == rec.row["axon_id"] and list(r) == list(ci.null_by_closure_columns())
        if r["null_status"] == "ok":
            assert r["null_B"] is not None and r["null_failed_replicates"] is not None
    k_row = rec.null_rows[1]
    if k_row["null_status"] == "ok":
        assert k_row["p_mc_k_per_um"] is None                           # conditioned on K: its p is left empty
    path = mb.write_null_by_closure([rec], str(tmp_path / "axons.csv"))
    with open(path, encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    assert {r["null_closure"] for r in rows} == {"n_locs", "k_clusters"}
    assert mb.write_null_by_closure([rec], str(tmp_path / "axons.csv")).endswith("_null_by_closure_2.csv")
    from tools import column_dictionary as cd
    assert all(cd.describe(c) for c in ci.null_by_closure_columns())
