"""Phase D integration (D0): registry, scale_nm, compatibility bit for bit, settings, exports, the strip's panel.

Synthetic data only (R8). Run: py -3 -m pytest test_phase_d_integration.py -q
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from tools.clustering_strategies import (
    COMPAT_METHOD, ClusteringStrategy, DBSCANStrategy, make_strategy, registry_names, strategy_params_json,
    strategy_registry)


def _ring_axon(seed: int = 0, n_clusters: int = 24, radius_nm: float = 400.0):
    """A ring of clusters (one axon's cross-section) with background, in nm."""
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
    return (np.concatenate(xs) + 2000.0, np.concatenate(ys) + 2000.0, np.concatenate(zs) + 300.0)


# ----------------------------------------------------------------- registry
def test_registry_dbscan_is_the_only_non_experimental():
    reg = strategy_registry()
    assert registry_names(include_experimental=False) == [COMPAT_METHOD]
    assert reg[COMPAT_METHOD].experimental is False
    assert all(info.experimental for name, info in reg.items() if name != COMPAT_METHOD)
    for name in ("OPTICS", "FOCAL", "voronoi_factor", "voronoi_csr", "ToMATo", "GraphCommunity", "External labels"):
        assert name in reg
    for name, info in reg.items():
        for key, d in info.defaults.items():
            assert set(d) >= {"value", "origin"}, (name, key)
            assert d["origin"], (name, key)


def test_make_strategy_defaults_and_overrides():
    st = make_strategy("FOCAL", {"grid_nm": 30.0, "unknown_key": 1})
    assert st.get_params()["grid_nm"] == 30.0
    st2 = make_strategy("OPTICS", {})            # max_eps 100 nm, registry default (REVIEW_D0 M5)
    assert st2.get_params()["max_eps"] == 100.0
    with pytest.raises(KeyError):
        make_strategy("no such method", {})


@pytest.mark.parametrize("name", [n for n in registry_names() if n != "External labels"])
def test_every_strategy_fits_and_reports_a_scale(name):
    x, y, _ = _ring_axon()
    xy = np.column_stack([x, y])
    st = make_strategy(name, {})
    labels = st.fit(xy)
    assert labels.shape == (xy.shape[0],)
    assert labels.min() >= -1
    scale = st.scale_nm()
    assert scale is None or (np.isfinite(scale) and scale > 0)
    if name == COMPAT_METHOD:
        assert scale == 25.0
    if name == "FOCAL":
        assert scale == 15.0                                 # FOCAL grid, verification run E
    json.dumps(strategy_params_json(st))


# ----------------------------------------------------------------- analysis: compatibility bit for bit
def test_compat_path_is_bit_identical_to_plain_dbscan():
    from tools.mps_analysis import analyze_axon, cluster_slab
    x, y, z = _ring_axon()
    a = analyze_axon(x, y, z, run_randomization=False)
    assert a.cluster_method == COMPAT_METHOD
    assert a.cluster_scale_nm == a.eps_nm == 25.0
    assert a.cluster_params["eps"] == 25.0 and a.cluster_params["min_samples"] == 10
    pts = np.column_stack([a.x_slab, a.y_slab])
    ref = DBSCAN(eps=a.eps_nm, min_samples=a.min_samples).fit(pts).labels_
    np.testing.assert_array_equal(a.labels, ref)
    labels, scale = cluster_slab(pts, DBSCANStrategy(eps=25.0, min_samples=10))
    np.testing.assert_array_equal(labels, ref)
    assert scale == 25.0
    # the explicit compat call equals the default call
    b = analyze_axon(x, y, z, run_randomization=False, cluster_method=COMPAT_METHOD)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_another_method_runs_and_records_its_scale():
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring_axon()
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method="FOCAL",
                     cluster_params={"grid_nm": 25.0, "min_l": 4}, cluster_methods_compared=["DBSCAN", "FOCAL"])
    assert a.cluster_method == "FOCAL"
    assert a.cluster_scale_nm == 25.0
    assert a.cluster_params["grid_nm"] == 25.0
    d = a.cluster_method_record()
    assert d["cluster_method"] == "FOCAL"
    assert json.loads(d["cluster_params"])["grid_nm"] == 25.0
    assert d["cluster_scale_nm"] == 25.0
    assert d["cluster_methods_compared"] == "DBSCAN|FOCAL"
    assert a.n_clusters_kept > 0
    # No cell may hold a ";" (Excel splits rows on it where the decimal mark is a comma, the user's locale).
    from tools.axon_export import axon_row
    row = axon_row(a)
    assert row["cluster_methods_compared"] == "DBSCAN|FOCAL"
    assert not [k for k, v in row.items() if ";" in str(v)], [k for k, v in row.items() if ";" in str(v)]


def test_fallback_scale_when_method_reports_none():
    from tools.mps_analysis import cluster_slab

    class NoScale(DBSCANStrategy):
        def scale_nm(self):
            return None

        def edge_margin_nm(self, compat_eps):        # the general rule: no scale -> the compatibility eps
            return ClusteringStrategy.edge_margin_nm(self, compat_eps)

    xy = np.column_stack(_ring_axon()[:2])
    _labels, scale = cluster_slab(xy, NoScale(eps=25.0, min_samples=10), fallback_scale_nm=31.0)
    assert scale == 31.0


# ----------------------------------------------------------------- export columns and axon_id
def test_export_columns_and_axon_id_stability():
    from tools import axon_export
    from tools.column_dictionary import describe, missing
    from tools.mps_analysis import analyze_axon
    assert axon_export.TABLE_VERSION == "axon tables v16"
    from tools.column_dictionary import known_columns
    n_em = len(axon_export.EMITTERS_COLUMNS)   # v11 (BaGoL F5): the emitters' block after the method's
    known = known_columns()[:-len(axon_export.integration_columns())]   # v13: the integration block, last
    assert known[-4 - n_em:-n_em] == list(axon_export.CLUSTER_METHOD_COLUMNS)   # v10
    for c in axon_export.CLUSTER_METHOD_COLUMNS:
        assert describe(c), c
    assert not missing(list(axon_export.CLUSTER_METHOD_COLUMNS))
    x, y, z = _ring_axon()
    a = analyze_axon(x, y, z, run_randomization=False)
    b = analyze_axon(x, y, z, run_randomization=False, cluster_method="FOCAL")
    assert axon_export.analysis_id(a) != axon_export.analysis_id(b)
    # the methods compared add nothing to a DBSCAN row's analysis_id (the id still changes with
    # TABLE_VERSION, v9 -> v10 -> v11, as at every version)
    assert axon_export.analysis_id(a) == axon_export.analysis_id(
        analyze_axon(x, y, z, run_randomization=False, cluster_methods_compared=["FOCAL"]))


# ----------------------------------------------------------------- settings
def test_settings_round_trip_and_validation():
    from tools.mps_settings import MPSSettings, load_settings, save_settings
    s = MPSSettings()
    assert s.cluster_method == "DBSCAN" and s.experimental_methods_enabled is False
    s.cluster_method = "ToMATo"
    s.cluster_params = {"k_density": 12}
    s.experimental_methods_enabled = True
    s.cluster_methods_compared = ["DBSCAN", "ToMATo", "bogus"]
    s.validate()
    assert s.cluster_methods_compared == ["DBSCAN", "ToMATo"]
    with tempfile.TemporaryDirectory() as d:
        save_settings(s, directory=d)
        back = load_settings(directory=d)
    assert back.cluster_method == "ToMATo"
    assert back.cluster_params == {"k_density": 12}
    assert back.experimental_methods_enabled is True
    s.cluster_method = "not a method"
    s.validate()
    assert s.cluster_method == "DBSCAN"


def test_batch_settings_carry_the_method():
    from tools.mps_batch import BatchSettings
    from tools.mps_settings import MPSSettings
    s = MPSSettings()
    s.cluster_method = "FOCAL"
    s.cluster_params = {"grid_nm": 30.0}
    b = BatchSettings.from_settings(s)
    assert b.cluster_method == "FOCAL" and b.cluster_params == {"grid_nm": 30.0}
    assert "FOCAL" in b.describe()
    assert "clustering method" not in BatchSettings.from_settings(MPSSettings()).describe()


# ----------------------------------------------------------------- benchmark verdicts (D11)
def test_verdicts_from_paired_rows():
    from tools.cluster_benchmark import verdicts_from_rows
    rows = []
    for rep in range(10):
        for sc in ("sparse_lab1", "medium_lab1"):
            rows.append({"scenario": sc, "replicate": rep, "strategy": "dbscan_25_10",
                         "exact": 0.80, "merge": 0.05, "missed": 0.05, "time_s": 0.01})
            rows.append({"scenario": sc, "replicate": rep, "strategy": "reg:FOCAL",
                         "exact": 0.80 + 0.001 * rep, "merge": 0.05, "missed": 0.05, "time_s": 0.02})
            rows.append({"scenario": sc, "replicate": rep, "strategy": "reg:same",
                         "exact": 0.80 + 0.01 * (rep % 2) - 0.005, "merge": 0.05, "missed": 0.05, "time_s": 0.02})
            rows.append({"scenario": sc, "replicate": rep, "strategy": "reg:slow",
                         "exact": 0.60, "merge": 0.30, "missed": 0.05, "time_s": 1.0})
    v = verdicts_from_rows(rows, reference="dbscan_25_10", delta=0.05, time_factor=10.0)
    assert v["verdicts"]["FOCAL"]["sparse"]["label"] == "preferible"
    assert v["verdicts"]["same"]["medium"]["label"] == "recomendable"
    assert v["verdicts"]["slow"]["sparse"]["label"] == "no recomendable"
    assert v["criterion"]["delta"] == 0.05


# ----------------------------------------------------------------- the strip's panel (offscreen Qt)
def test_method_panel_and_compare_dialog_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tools.cluster_method_ui import CompareMethodsDialog, MethodPanel, verdict_words
    p = MethodPanel()
    assert [p.combo.itemText(i) for i in range(p.combo.count())] == ["DBSCAN"]
    p.chk_experimental.setChecked(True)
    assert p.combo.count() > 1
    p.set_state("FOCAL", {"grid_nm": 30.0}, True, ["DBSCAN", "FOCAL"])
    assert p.current_method() == "FOCAL" and p.current_params()["grid_nm"] == 30.0
    p.set_compat()
    assert p.current_method() == "DBSCAN" and p.current_params() == {}
    assert verdict_words({}, "FOCAL", "sparse") == "no benchmark"
    assert verdict_words({"verdicts": {"FOCAL": {"sparse": {"label": "recomendable"}}}}, "FOCAL", "sparse") \
        == "recomendable"

    from tools.mps_analysis import analyze_axon
    x, y, z = _ring_axon()
    base = analyze_axon(x, y, z, run_randomization=False)
    calls = []

    def run_fn(method, params):
        calls.append(method)
        return analyze_axon(base.x_slab, base.y_slab, base.z_slab, run_randomization=False,
                            slab_override=(base.slab_zmin_nm, base.slab_zmax_nm),
                            cluster_method=method, cluster_params=params)

    dlg = CompareMethodsDialog(run_fn, int(base.x_slab.size), True, "DBSCAN", lambda n: {}, verdicts={})
    assert calls == []                      # D12: nothing ran on opening
    for name, cb in dlg.checks.items():
        cb.setChecked(name in ("DBSCAN", "FOCAL"))
    dlg.start()
    assert dlg._worker is not None
    dlg._worker.wait(60000)
    for _ in range(50):
        app.processEvents()
    assert sorted(calls) == ["DBSCAN", "FOCAL"]
    assert dlg.table.rowCount() == 2
    dlg.table.selectRow(1)
    assert dlg.selected_methods() == [dlg.table.item(1, 0).text()]
    dlg.close()


def test_set_state_emits_nothing_and_the_window_reruns_only_when_asked():
    """D12 (review of D0): showing a saved method must not re-run the analysis shown, and Compatibility 25/10
    must run it once, not twice (set_state used to unblock the combo and emit method_changed)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from tools.cluster_method_ui import MethodPanel
    p = MethodPanel()
    hits = []
    p.method_changed.connect(lambda: hits.append(p.current_method()))
    p.set_state("FOCAL", {}, False)
    assert p.current_method() == "FOCAL" and hits == []
    p.set_compat()
    assert p.current_method() == "DBSCAN" and hits == []
    assert not p.combo.signalsBlocked()
    p.combo.setCurrentIndex(p.combo.findText("ToMATo"))         # the user's own choice is emitted
    assert hits == ["ToMATo"]
    p.chk_experimental.setChecked(False)                         # unticking falls back to DBSCAN, emitted
    assert p.current_method() == "DBSCAN" and hits == ["ToMATo", "DBSCAN"]

    from tools.mps_analysis import analyze_axon
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    x, y, z = _ring_axon()
    saved = {"method": "FOCAL"}
    calls = []

    def rerun(**kw):
        calls.append(kw.get("cluster_method"))
        kw.pop("main_peak_override_nm", None)
        kw.pop("contour_guide", None)
        kw["run_randomization"] = False
        return analyze_axon(x, y, z, **kw)

    def store(values):
        if "cluster_method" in values:
            saved["method"] = values["cluster_method"]

    links = WindowLinks(store=store, method_state=lambda: (saved["method"], {}, True, []))
    w = MPSResultsWindow(analyze_axon(x, y, z, run_randomization=False, cluster_method="FOCAL"),
                         rerun_callback=rerun, links=links)
    for _ in range(20):
        app.processEvents()
    assert calls == []                      # opening the window runs nothing (D12)
    assert w.method_panel.current_method() == "FOCAL"
    w.method_panel.btn_compat.click()
    for _ in range(20):
        app.processEvents()
    assert len(calls) == 1 and saved["method"] == "DBSCAN"
    w.close()


# ----------------------------------------------------------------- batch comparison (monkeypatched analysis)
def test_batch_compare_methods_rows(monkeypatch, tmp_path):
    from tools import mps_batch as mb
    from tools.mps_settings import MPSSettings

    # The clustering integration: the file is loaded once and every method runs on its localizations through
    # tools.multimethod (DBSCAN 25/10 first, as the reference); the analysis itself is faked here.
    import types
    xs = np.arange(30, dtype=float)
    loc = types.SimpleNamespace(x_nm=xs, y_nm=xs, z_nm=xs, frame=None)
    seen = []

    def fake_analyse(loc_, item, settings):
        seen.append((settings.cluster_method, settings.eps_nm, dict(settings.cluster_params)))
        return types.SimpleNamespace(
            source_name=item.path, roi=None, cluster_method=settings.cluster_method, cluster_scale_nm=25.0,
            clusters_per_um=5.0, median_1nn_nm=190.0, median_r_eff_nm=20.0, occupancy_percent=60.0,
            n_clusters_kept=40, x_slab=xs, y_slab=xs, labels=np.zeros(30, dtype=np.int64))

    monkeypatch.setattr(mb, "_load", lambda item: (loc, None))
    monkeypatch.setattr(mb, "_analyze_loaded", fake_analyse)
    settings = mb.BatchSettings.from_settings(MPSSettings())
    item = mb.BatchFile(path=str(tmp_path / "a.hdf5"), identity=mb.AxonIdentity())
    results = []
    rows = mb.compare_methods([item], settings, ["DBSCAN", "FOCAL"], results_out=results)
    assert [r["cluster_method"] for r in rows] == ["DBSCAN", "FOCAL"]
    assert all(r["error"] == "" for r in rows)
    assert [s[0] for s in seen] == ["DBSCAN", "FOCAL"] and seen[0][1] == 25.0
    assert results and results[0].runs[0].is_reference and results[0].long_rows
    out = mb.comparison_path(str(tmp_path / "mps_axons.csv"))
    assert os.path.basename(out) == "batch_methods_comparison.csv"
    assert mb.write_methods_comparison(rows, out) == 2
    with open(out, encoding="utf-8") as fh:
        head = fh.readline().strip().split(",")
    assert head == list(mb.COMPARISON_COLUMNS)
