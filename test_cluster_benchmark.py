# -*- coding: utf-8 -*-
"""
Tests of the clustering benchmark harness ``tools.cluster_benchmark``
(docs/sims/clustering_benchmark/README.md). Synthetic data only (rule R8):
every sample is drawn by the harness's own generator with a fixed seed.

Run: ``py -3 -m pytest test_cluster_benchmark.py -q``.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from tools import cluster_benchmark as cb
from tools.clustering_strategies import DBSCANStrategy


def _small_sample(seed: int = 3, p_label: float = 0.6) -> cb.SyntheticSample:
    """Sparse ring of 8 um (about 40 molecules, below 300) with the study's sigma, c_label and low background."""
    sc = cb.Scenario("small_sparse", spacing_nm=200.0, p_label=p_label, L_um_range=(8.0, 8.0))
    return cb.generate(sc, np.random.default_rng(seed))


def test_generator_truth_is_consistent():
    s = _small_sample()
    n = s.xy.shape[0]
    assert s.params["n_molecules"] <= 300
    assert s.frame.shape == s.molecule_id.shape == s.label_id.shape == s.event_id.shape == (n,)
    assert np.all(np.diff(s.frame) >= 0)                     # sorted by frame, as a real file
    assert s.frame.max() < cb.N_FRAMES
    real = s.molecule_id >= 0
    assert real.sum() > 0 and (~real).sum() > 0               # molecules and background both present
    assert set(np.unique(s.molecule_id[real])) <= set(range(s.mol_xy.shape[0]))
    assert int((s.mol_n_labels > 0).sum()) == s.params["n_molecules_labelled"]


def test_oracle_scores_perfectly():
    s = _small_sample()
    m = cb.evaluate(s.oracle_labels, s)
    assert m["exact"] == pytest.approx(1.0)
    assert m["ari"] == pytest.approx(1.0)
    assert m["iou_mean"] == pytest.approx(1.0)
    assert m["merge"] == 0.0 and m["split"] == 0.0 and m["missed"] == 0.0 and m["false"] == 0.0
    assert m["n_clusters"] == m["n_mol_labelled"]


def test_dbscan_default_on_sparse_scenario_matches_report_range():
    """SIM_CLUSTERING_REPORT.md section 0.1: DBSCAN 25/10 gives 77-83 % of one-molecule clusters in the sparse
    regime (about 1.35 labels per labelled molecule at p_label 0.6). Three seeds pooled, wide band."""
    exact = []
    for seed in (3, 4, 5):
        s = _small_sample(seed)
        labels = DBSCANStrategy(eps=cb.EPS_NM_DEFAULT, min_samples=cb.MIN_SAMPLES_DEFAULT).fit(s.xy)
        m = cb.evaluate(labels, s, cb.readouts(s.xy, s.oracle_labels, s.arc_length_nm))
        exact.append(m["exact"])
        assert np.isfinite(m["bias_clusters_per_um_pct"])
    assert 0.60 <= float(np.mean(exact)) <= 0.95


def test_merged_clusters_are_detected():
    """Everything in one cluster: one merged cluster, ARI near 0, every molecule 'exact' fails."""
    s = _small_sample()
    labels = np.zeros(s.xy.shape[0], dtype=np.int64)
    m = cb.evaluate(labels, s)
    assert m["n_clusters"] == 1 and m["merge"] == 1.0 and m["exact"] == 0.0
    assert m["ari"] == pytest.approx(0.0, abs=1e-9)


def test_run_benchmark_writes_and_reads_back(tmp_path):
    sc = cb.Scenario("tiny", spacing_nm=200.0, p_label=1.0, L_um_range=(6.0, 6.0))
    paths = cb.run_and_write([sc], ["dbscan_25_10"], n_replicates=2, seed=11, out_dir=str(tmp_path))
    for p in paths.values():
        assert os.path.exists(p)
    rows = cb.read_csv(paths["long"])
    assert len(rows) == 4                                     # 2 replicates x (oracle + dbscan)
    assert {r["strategy"] for r in rows} == {"oracle", "dbscan_25_10"}
    summ = cb.read_csv(paths["summary"])
    assert len(summ) == 2 and all(r["n_replicates"] == 2 for r in summ)
    text = open(paths["markdown"], encoding="utf-8").read()
    assert "## Scenario `tiny`" in text and "dbscan_25_10" in text and "+/-" in text
    cfg = json.load(open(paths["config"], encoding="utf-8"))
    assert cfg["seed"] == 11 and cfg["n_replicates"] == 2 and cfg["scenarios"][0]["name"] == "tiny"
    # reproducible: a second run with the same seed gives the same long table
    rows2 = cb.read_csv(cb.run_and_write([sc], ["dbscan_25_10"], 2, 11, str(tmp_path / "again"))["long"])
    for a, b in zip(rows, rows2):
        assert a["exact"] == b["exact"] and a["n_locs"] == b["n_locs"]


def test_external_labels_strategy_round_trip(tmp_path):
    s = _small_sample()
    truth = s.oracle_labels
    path = tmp_path / "ext.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "label"])
        for (x, y), lab in zip(s.xy, truth):
            w.writerow([repr(float(x)), repr(float(y)), int(lab)])
    labels = cb.ExternalLabelsStrategy(str(path)).fit(s.xy)
    assert np.array_equal(labels, truth)
    assert cb.evaluate(labels, s)["exact"] == pytest.approx(1.0)


def test_callable_strategy_and_grid():
    s = _small_sample()
    st = cb.CallableStrategy(lambda xy: DBSCANStrategy(25.0, 10).fit(xy), "my_dbscan", {"eps": 25})
    assert st.get_strategy_name() == "my_dbscan"
    rows = cb.run_benchmark([cb.Scenario("g", L_um_range=(6.0, 6.0))], {"d": DBSCANStrategy(25.0, 10)}, 1, 0,
                            grid={"d": [{"eps": 20.0}, {"eps": 30.0}]})
    assert [r["strategy"] for r in rows] == ["oracle", "d", "d"]
    assert json.loads(rows[1]["params"])["eps"] == 20.0 and json.loads(rows[2]["params"])["eps"] == 30.0
    with pytest.raises(ValueError):
        cb.CallableStrategy(lambda xy: np.zeros(3, dtype=np.int64), "bad").fit(s.xy)


def test_presets_and_cli_list(capsys):
    assert "sparse_lab1" in cb.PRESETS and cb.PRESETS["sparse_lab1"].p_label == 0.5
    assert cb.main(["--list"]) == 0
    assert "sparse_lab2_c26" in capsys.readouterr().out


def test_scenarios_get_independent_seeds_and_variants_do_not_leak():
    """M2: replicate r of two scenarios no longer shares positions; B1: a grid variant runs on its own copy."""
    a = cb.Scenario("sa", spacing_nm=200.0, p_label=1.0, L_um_range=(6.0, 6.0))
    b = cb.Scenario("sb", spacing_nm=200.0, p_label=1.0, L_um_range=(6.0, 6.0))
    rows = cb.run_benchmark([a, b], {"d": DBSCANStrategy(25.0, 10)}, 1, 0, include_oracle=False)
    assert rows[0]["n_locs"] != rows[1]["n_locs"] or rows[0]["exact"] != rows[1]["exact"]
    assert cb.scenario_seed("sa") != cb.scenario_seed("sb")
    st = DBSCANStrategy(25.0, 10)
    rows = cb.run_benchmark([a], {"d": st}, 1, 0, grid={"d": [{"eps": 20.0}, {"min_samples": 5}]}, include_oracle=False)
    assert json.loads(rows[1]["params"])["eps"] == 25.0 and st.eps == 25.0
    v = cb.verdicts_from_rows(rows + [dict(r, strategy="ref", params="{}") for r in rows[:1]], reference="ref")
    assert len(v["verdicts"]) == 2                                  # one entry per parameter set


def test_external_labels_rounded_csv_raises_and_tolerance_fixes_it(tmp_path):
    """M6: a CSV with 2 decimals matched almost nothing at 1e-3 nm and returned noise without a word."""
    s = _small_sample()
    path = tmp_path / "ext2.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "label"])
        for (x, y), lab in list(zip(s.xy, s.oracle_labels))[::-1]:
            w.writerow([f"{x:.2f}", f"{y:.2f}", int(lab)])
    with pytest.raises(ValueError, match="matched"):
        cb.ExternalLabelsStrategy(str(path)).fit(s.xy)
    ok = cb.ExternalLabelsStrategy(str(path), tol_nm=0.01)
    assert np.array_equal(ok.fit(s.xy), s.oracle_labels) and ok.frac_matched_ == 1.0


def test_registry_spec_accepts_words_and_null():
    st = cb.make_strategy("reg:OPTICS:cluster_method=dbscan,eps=25")
    assert st.get_params()["cluster_method"] == "dbscan" and st.get_params()["eps"] == 25
    assert cb.make_strategy("reg:ToMATo:k_graph=null").get_params()["k_graph"] is None


def test_optics_verdict_is_diagnostic_and_status_is_verified():
    rows = []
    for rep in range(3):
        for st in ("dbscan_25_10", "reg:OPTICS"):
            rows.append({"strategy": st, "scenario": "sparse_x", "replicate": rep, "params": "{}",
                         "exact": 0.9, "merge": 0.0, "missed": 0.0, "time_s": 0.01})
    v = cb.verdicts_from_rows(rows)
    assert v["criterion"]["status"] == cb.VERDICT_STATUS
    e = v["verdicts"]["OPTICS"]["sparse"]
    assert e["label"] == "no recomendable" and "diagnóstico" in e["role"]
