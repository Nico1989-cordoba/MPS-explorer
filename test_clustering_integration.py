"""The clustering integration without Qt (tools/clustering_integration.py): the provisional answers, the plug-ins
registered once, the comparison with the reference first and the rule's double table, the per-method null with the
regularity reading on the same null, the axon table's v13 block, and the batch's regularity summary.

Synthetic data only (rule R8). Run: py -3 -m pytest test_clustering_integration.py -q
"""
from __future__ import annotations

import os
import types
from typing import Any, Dict, List

import numpy as np
import pytest

from tools import clustering_integration as ci
from tools import multimethod as MM
from tools import mps_pipeline_null as pn

SMALL = (900.0, 700.0)


def _ring_axon(seed: int = 0, n_clusters: int = 24, radius_nm: float = 400.0):
    """A ring of clusters with background, in nm (as test_multimethod)."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False) + rng.normal(0, 0.03, n_clusters)
    xs, ys, zs = [], [], []
    for a in ang:
        n = int(rng.integers(15, 40))
        xs.append(radius_nm * np.cos(a) + rng.normal(0, 9, n))
        ys.append(radius_nm * np.sin(a) + rng.normal(0, 9, n))
        zs.append(rng.normal(0, 20, n))
    xs.append(rng.uniform(-520, 520, 150))
    ys.append(rng.uniform(-520, 520, 150))
    zs.append(rng.normal(0, 60, 150))
    return np.concatenate(xs) + 2000.0, np.concatenate(ys) + 2000.0, np.concatenate(zs) + 300.0


@pytest.fixture(scope="module")
def observed():
    """A synthetic observed axon of the pipeline null's generator (frames and precisions kept)."""
    an, _ax = pn.observe_synthetic(pn.synthetic_model(SMALL), 3)
    return an


@pytest.fixture(scope="module")
def small_null(observed):
    run = ci.run_of(observed, "syn")
    return run, ci.run_null_for(run, n_replicates=4, nena_nm=12.0, seed=1)


# ------------------------------------------------------------------------------------------------- provisional
def test_every_open_question_has_a_value_an_origin_and_a_place():
    keys = set(ci.PROVISIONAL)
    assert {"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3",
            "D4", "E1", "E2", "E4", "E5", "G1", "G2", "G3", "G4", "G5"} <= keys
    for p in ci.PROVISIONAL.values():
        assert p.origin.startswith("[") and p.where, p.key
    assert ci.NULL_GLOBAL_TEST_FROM == "d50" and ci.provisional("C1") == "d50"
    assert pn.PILOT_AXONS_K == ci.provisional("C3") == 4
    assert abs(ci.LPC_RULE_RADIUS_IN_NENA - 3.0 * np.sqrt(2.0)) < 1e-12
    assert ci.PROVISIONAL["D1"].applied is False            # the frozen H-ECL pre-specification is not touched
    from tools import mps_gap_measures as gm
    assert gm.MATCHED_TOLERATED_FAILED_FRACTION == ci.provisional("G1")


def test_plugins_are_registered_once_with_the_regularity_hook():
    names = ci.register_all_plugins()
    for n in ("per_axon", "events", "eclipse", "size", "sweep", "size_eps", "pipeline_null", "gap_measures"):
        assert n in names
    assert ci.register_all_plugins() == names
    for r in ("size_nloc_nn_spearman", "gap_cv", "gap_rho1", "bragg_s", "elongation_p95"):
        assert r in pn.READOUTS
    assert set(ci.NULL_READOUT_NAMES) <= set(pn.READOUTS)
    assert not ({c.name for c in ci.ANALYSIS_CHOICES} - set(names))
    assert "sweep" not in ci.DEFAULT_ANALYSES and "size_eps" not in ci.DEFAULT_ANALYSES      # B5, E4


# ------------------------------------------------------------------------------------------------- comparison
def test_comparison_runs_the_reference_first_with_the_rule_double_table(tmp_path):
    x, y, z = _ring_axon()
    run_fn = MM.runner_from_arrays(x, y, z, source_name="synthetic ring")
    rule = ci.rule_for(12.0)
    rule26 = ci.rule_for(12.0, c_label=26.0)
    assert rule is not None and rule.min_samples == 6 and rule26 is not None and rule26.min_samples == 11
    methods: List[Any] = ["FOCAL"] + ci.rule_method_specs(rule, rule26)
    plan = MM.plan_runs(methods, int(x.size))
    labels = ci.rule_labels(plan, rule, rule26)
    assert sorted(labels.values()) == ["DBSCAN rule (c_label 14)", "DBSCAN rule (c_label 26)"]
    res = ci.run_comparison(run_fn, methods, n_locs=int(x.size), axon_id="ring", plan=plan, labels=labels,
                            extras=ci.comparison_extras(nena_nm=12.0))
    assert res.runs[0].key == MM.REFERENCE_KEY and res.runs[0].is_reference
    assert all(r.ok for r in res.runs), [r.error for r in res.runs]
    analyses = {r["analysis_name"] for r in res.long_rows}
    assert {"per_axon", "events", "eclipse", "size"} <= analyses
    assert not [r for r in res.long_rows if r["error"] and r["run_status"] == "ok"], \
        {r["error"] for r in res.long_rows if r["error"]}
    double = res.double_table()
    assert [d["label"] for d in double][0] == "DBSCAN 25/10 (reference)" and len(double) == 3
    paths = ci.write_comparison(str(tmp_path / "axonA"), res)
    assert os.path.basename(paths["long"]) == "axonA_methods_long.csv"
    back = MM.read_long_csv(paths["long"])
    assert len(back) == len(res.long_rows)
    again = ci.write_comparison(str(tmp_path / "axonA"), res)
    assert again["long"].endswith("axonA_methods_long_2.csv")                  # never overwritten
    for key in ("agreement", "summary", "wide"):
        assert os.path.exists(paths[key])


def test_slow_methods_are_refused_in_the_comparison_unless_ticked():
    plan = MM.plan_runs(["emitters_bagol_like", "FOCAL"], 2000)
    assert "emitters_bagol_like" in plan.refused
    assert ci.provisional("A2") is True


# ------------------------------------------------------------------------------------------------- the null
def test_null_of_a_run_repeats_its_method_and_reads_regularity_on_the_same_null(observed, small_null):
    run, out = small_null
    assert out.ok and out.result.method == observed.cluster_method
    assert out.result.global_test_from == "d50"                         # C1
    assert out.result.global_r_min_nm == pytest.approx(out.result.scale.d50_nm)
    assert len(out.result.replicate_arcs) == out.result.n_accepted      # G2
    assert out.gap is not None and out.gap.null_result is out.result    # G3: one null
    assert "gap_cv" in out.result.p_values and "size_nloc_nn_spearman" in out.result.p_values
    from tools import mps_gap_measures as gm
    bands = gm.pipeline_null_bands(out.result, int(observed.n_clusters_kept), 5000.0)
    assert bands["n"][0] == out.result.n_accepted and bands["gap_hist_band"].shape[0] == 3
    rows = ci.null_long_rows([run], {run.key: out})
    assert rows and set(rows[0]) == set(MM.LONG_COLUMNS)
    assert {r["readout"] for r in rows} >= {"null_ripley1d_p", "null_interpretable_verdict"}


def test_null_refusals_are_outcomes_not_exceptions(observed):
    slow = MM.MethodRun(method="emitters_bagol_like", params={}, key="Emitters", analysis=observed,
                        labels=np.asarray(observed.labels))
    out = ci.run_null_for(slow, n_replicates=199, nena_nm=12.0)
    assert out.status == "refused" and "slow" in out.message and out.estimate_s > 0
    failed = MM.MethodRun(method="FOCAL", params={}, key="FOCAL", status=MM.STATUS_ERROR, error="boom")
    assert ci.run_null_for(failed, n_replicates=3, nena_nm=12.0).status == "error"


def test_null_cancel_stops_the_remaining_methods(observed):
    run = ci.run_of(observed, "syn")
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    out = ci.run_nulls([run, run], n_replicates=5, nena_nm=12.0, cancelled=cancelled)
    assert [o.status for o in out.values()][-1] == "cancelled"


def test_stage2_not_interpretable_is_not_read():
    from tools import mps_gap_measures as gm
    s = gm.NullSummary
    cv = gm.summarise("gap_cv", 0.3, [0.9 + 0.01 * i for i in range(40)], "two-sided")
    rho = gm.summarise("gap_rho1", -0.5, [0.0] * 40, "lower")
    bragg = gm.summarise("bragg_s", 30.0, [3.0] * 40, "upper")
    matched = types.SimpleNamespace(kind="pipeline", d_m_nm=120.0, capped=False, n_replicates=40,
                                    interpretable=False, not_interpretable="only 30 of 40 replicates")
    rd = gm.regularity_reading(k=40, cv=cv, rho1_matched=rho, bragg_matched=bragg, matched=matched)
    assert rd.level == gm.KIND_NOT_IDENTIFIED and "NOT INTERPRETABLE" in rd.sentence
    matched.interpretable = True
    assert gm.regularity_reading(k=40, cv=cv, rho1_matched=rho, bragg_matched=bragg, matched=matched).level == gm.LATTICE
    assert s is not None


# ------------------------------------------------------------------------------------------------- the axon row
def test_axon_columns_are_described_and_fill_the_null_when_one_ran(observed, small_null):
    cols = ci.integration_columns()
    assert len(cols) == len(set(cols))
    assert not [c for c in cols if not ci.describe_column(c)]
    row = ci.axon_columns(observed, nena_nm=12.0, nena_source="this file")
    assert list(row) == list(cols)
    assert row["dbscan_mode"] == "compatibility 25/10" and row["null_type"] == "none"
    assert row["scale_d50_nm"] == pytest.approx(80.0) and row["min_samples_rule"] == 6
    assert row["eclipse_sigma_c_source"] == "preset"            # events were linked: c_label preset, not "no events"
    assert row["events_n_per_cluster_median"] is not None
    _run, out = small_null
    full = ci.axon_columns(observed, nena_nm=12.0, null=out)
    assert full["null_B"] == out.result.n_accepted and full["null_global_test_from"] == "d50"
    assert full["regularity_reading"] == out.gap.reading.level
    assert set(out.gap.export_columns()) <= set(cols)
    no_nena = ci.axon_columns(observed)
    assert no_nena["scale_nena_source"] == "paper (no NeNA)" and no_nena["eps_rule_nm"] is None


def test_dbscan_mode_reads_how_the_parameters_were_chosen(observed):
    rule = ci.rule_for(12.0)
    a = types.SimpleNamespace(cluster_method="DBSCAN", eps_nm=25.0, min_samples=6)
    assert ci.dbscan_mode(a, rule) == "rule"
    a.min_samples = 7
    assert ci.dbscan_mode(a, rule) == "typed"
    assert ci.dbscan_mode(types.SimpleNamespace(cluster_method="HDBSCAN"), rule) == "not DBSCAN"
    assert ci.dbscan_mode(observed) == "compatibility 25/10"


# ------------------------------------------------------------------------------------------------- v14: A8 / P-R103
def _polygon_on_circle(k: int, radius_nm: float, phase: float = 0.3) -> np.ndarray:
    ang = phase + np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    return np.column_stack([radius_nm * np.cos(ang), radius_nm * np.sin(ang)]) + 5000.0


@pytest.mark.parametrize("k,radius_nm", [(20, 1500.0), (40, 400.0), (7, 300.0), (60, 450.0), (100, 600.0)])
def test_arc_fraction_on_equispaced_centres_is_min_one_2dk_over_p(k, radius_nm):
    """Analytical case (derivation): K centres equally spaced on a circle are the vertices of a regular polygon of
    perimeter P = 2 K R sin(pi / K); each covers d of arc along each of its two edges, so the fraction within d is
    min(1, 2 d K / P) exactly (the chord is below 2 d in the last two cases: the whole contour is covered)."""
    from tools.mps_occupancy import arc_fraction_within
    poly = _polygon_on_circle(k, radius_nm)
    p = 2.0 * k * radius_nm * np.sin(np.pi / k)
    expected = min(1.0, 2.0 * 25.0 * k / p)
    assert arc_fraction_within(poly, poly, 25.0) == pytest.approx(expected, abs=1e-12)
    # the same centres in another order (the union does not depend on it)
    assert arc_fraction_within(poly, poly[::-1].copy(), 25.0) == pytest.approx(expected, abs=1e-12)


def test_arc_fraction_counts_overlaps_once_and_reads_off_contour_centres():
    from tools.mps_occupancy import arc_fraction_within
    square = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0]])   # P = 4000 nm
    # one centre on an edge's middle: 2 d of arc
    assert arc_fraction_within(square, np.array([[500.0, 0.0]]), 25.0) == pytest.approx(50.0 / 4000.0, abs=1e-12)
    # the same centre twice, and a second one 10 nm away: one union, never twice
    twice = np.array([[500.0, 0.0], [500.0, 0.0], [510.0, 0.0]])
    assert arc_fraction_within(square, twice, 25.0) == pytest.approx(60.0 / 4000.0, abs=1e-12)
    # a centre 15 nm off the edge covers the chord 2 sqrt(d^2 - h^2) = 40 nm (derivation)
    assert arc_fraction_within(square, np.array([[500.0, 15.0]]), 25.0) == pytest.approx(40.0 / 4000.0, abs=1e-12)
    # at a corner: d along each edge
    assert arc_fraction_within(square, np.array([[0.0, 0.0]]), 25.0) == pytest.approx(50.0 / 4000.0, abs=1e-12)
    # farther than d from every edge, or no centre at all: nothing covered
    assert arc_fraction_within(square, np.array([[500.0, 500.0]]), 25.0) == 0.0
    assert arc_fraction_within(square, np.empty((0, 2)), 25.0) == 0.0
    with pytest.raises(ValueError):
        arc_fraction_within(square[:2], square[:2], 25.0)
    with pytest.raises(ValueError):
        arc_fraction_within(square, square, 0.0)


def test_arc_fraction_does_not_change_the_occupancy_and_is_in_the_row(observed):
    """P-R103 adds a reading; occupancy_percent (the constrained Gaussian) is computed exactly as before."""
    from tools.mps_occupancy import compute_occupancy
    before = float(observed.occupancy.occupancy_percent)
    frac = ci.arc_fraction_of(observed)
    assert frac is not None and 0.0 < frac <= 1.0
    row = ci.axon_columns(observed, nena_nm=12.0, nena_source="this file")
    excluded = {int(v) for v in observed.bad_report.bad_labels} | {int(v) for v in observed.discarded_labels}
    again = compute_occupancy(observed.x_slab, observed.y_slab, observed.labels, observed.perimeter.contour,
                              exclude_labels=excluded, mahalanobis_threshold=observed.occupancy.mahalanobis_threshold,
                              sigma_cap_fraction=observed.sigma_cap_fraction, ellipse_mode=observed.ellipse_mode)
    assert observed.occupancy.occupancy_percent == before == again.occupancy_percent
    # the arc is the occupancy's own curve: the closed polygon of perimeter_um
    assert again.perimeter_length_nm == pytest.approx(observed.perimeter.perimeter_nm, rel=1e-9)
    assert len(observed.centroids) == len(observed.perimeter.contour)
    assert row["occupancy_arc_fraction_within_25nm"] == pytest.approx(frac, abs=5e-5)
    assert row["link_radius_nm"] == pytest.approx(36.0) and row["nena_method"] == ci.NENA_METHOD_MEASURED
    bare = ci.axon_columns(observed)
    assert bare["nena_method"] == ci.NENA_METHOD_NONE and bare["link_radius_nm"] is None
    tail = list(ci.REPORT_COLUMNS) + list(ci.v15_columns()) + list(ci.v16_columns())
    assert list(ci.integration_columns())[-len(tail):] == tail


def test_the_table_is_v16_and_the_report_v15_v16_columns_are_last_and_described():
    from tools import axon_export, column_dictionary
    assert axon_export.TABLE_VERSION == "axon tables v22"
    known = column_dictionary.known_columns()
    tail = list(ci.REPORT_COLUMNS) + list(ci.v15_columns()) + list(ci.v16_columns())
    from tools.perimeter_export import (PERIMETER_COLUMNS, PERIMETER_V19_COLUMNS, PERIMETER_V20_COLUMNS,
                                        PERIMETER_V21_COLUMNS, PERIMETER_V22_COLUMNS)
    from tools.unanchored_export import UNANCHORED_COLUMNS
    n_last = (len(PERIMETER_COLUMNS) + len(UNANCHORED_COLUMNS) + len(PERIMETER_V19_COLUMNS)
              + len(PERIMETER_V20_COLUMNS) + len(PERIMETER_V21_COLUMNS)
              + len(PERIMETER_V22_COLUMNS))
    assert known[:-n_last][-len(tail):] == tail   # v17: the perimeter block, v18: the unanchored block, v19: last
    for c in tail:
        assert column_dictionary.describe(c), c
    assert column_dictionary.unit_of("link_radius_nm") == "nm"
    assert column_dictionary.unit_of("occupancy_arc_fraction_within_25nm") == ""


# ------------------------------------------------------------------------------------------------- the batch
def test_regularity_batch_summary_reads_animals_and_refuses_the_calibre_below_three():
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(5)
    for animal in ("m1", "m2"):
        for i in range(3):
            rows.append({"animal": animal, "genotype": "WT", "regularity_reading": "random_with_contact",
                         "gap_cv_z": float(rng.normal()), "gap_excess_cv_z": float(rng.normal()),
                         "gap_rho1_z_matched": None, "bragg_s_z_matched": None, "n_clusters_kept": 40 + i,
                         "perimeter_um": 10.0 + i})
    out = ci.regularity_batch_summary(rows)
    assert len(out["by_animal"]) == 2 and "WT" in out["lattice"]
    assert "not fitted" in " ".join(out["lines"])
    for i in range(3):
        rows.append({"animal": "m3", "genotype": "WT", "regularity_reading": "random_with_contact",
                     "gap_cv_z": 0.1, "n_clusters_kept": 45 + 2 * i, "perimeter_um": 11.0 + 1.5 * i})
    out3 = ci.regularity_batch_summary(rows)
    assert out3["calibre"] and "not fitted" not in out3["calibre"]
    assert ci.regularity_batch_summary([{"animal": "m1"}])["lines"][0].startswith("Regularity: no axon")
