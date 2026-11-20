"""The interface batch in the analysis window (offscreen Qt, synthetic data only, rule R8; 2026-11-07):
A2 (the confirmation before a slow method runs, in 'Compare methods...', the null and the Batch), B1 (the two radii
in Working scale, EXPLORATORY, the export), B3 (localizations per cycle by depth in the window), C2 (the null's
membrane and the deformation indicator), C3 / C4 (the closure selector, 'both', the closure comparison, null_closure
per row) and G2 (the regularity bands: 'approx., centres' before the null, the pipeline null's after it).

Run: QT_QPA_PLATFORM=offscreen py -3 -m pytest test_ui_tanda2_gui.py -q
"""
from __future__ import annotations

import csv
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from PyQt5 import QtWidgets  # noqa: E402

from tools import mps_pipeline_null as pn  # noqa: E402
from tools import slow_confirm  # noqa: E402

SMALL = (900.0, 700.0)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def observed():
    an, _ax = pn.observe_synthetic(pn.synthetic_model(SMALL), 3)
    return an


@pytest.fixture(autouse=True)
def no_dialogs():
    slow_confirm.ASKED[:] = []
    slow_confirm.ANSWER = lambda _t: False
    yield
    slow_confirm.ANSWER = None
    slow_confirm.ASKED[:] = []


def _pump(app, cond, timeout=300.0):
    t0 = time.time()
    while cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()


def _fake_profile():
    from tools import lpc_depth
    rng = np.random.default_rng(3)
    n = 400
    frame = np.repeat(np.arange(n) * 10, 3)
    x = np.repeat(rng.uniform(0, 5000, n), 3) + rng.normal(0, 3, 3 * n)
    y = np.repeat(rng.uniform(0, 5000, n), 3) + rng.normal(0, 3, 3 * n)
    z = np.repeat(rng.uniform(-300, 300, n), 3)
    return lpc_depth.lpc_depth_profile(frame + np.tile([0, 1, 2], n), x, y, z, nena_nm=10.0, slab=(-90.0, 90.0),
                                       n_boot=99)


def _window(observed, calls=None):
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    prof = _fake_profile()

    def lpc_depth(_a, radius=None):
        if calls is not None:
            calls.append(radius)
        return prof, ""

    links = WindowLinks(nena=lambda: (12.0, "this file (synthetic)"), lpc_depth=lpc_depth,
                        rule_inputs_at=lambda r: (calls.append(("lpc", r)) if calls is not None else None))
    return MPSResultsWindow(observed, rerun_callback=lambda **kw: observed, links=links)


# ------------------------------------------------------------------------------------------------------- B1 / B3
def test_radii_in_working_scale_are_exploratory_and_exported(app, observed):
    calls = []
    w = _window(observed, calls)
    try:
        ws = w.working_scale
        assert ws.lpc_radius() is None and ws.events_radius() is None
        cols = w.integration_columns()
        assert cols["radii_exploratory"] == "no" and cols["link_radius_nena"] == 3.0
        ws.spin_events_radius.setValue(2.0)
        assert "EXPLORATORY" in ws.lbl_radii.text() and "events / fusion at 2 NeNA" in ws.lbl_radii.text()
        cols = w.integration_columns()
        assert cols["radii_exploratory"].startswith("EXPLORATORY") and cols["link_radius_nena"] == 2.0
        assert cols["link_radius_nm"] == pytest.approx(24.0)
        ws.spin_lpc_radius.setValue(3.0)
        assert ("lpc", 3.0) in calls                         # the rule re-reads lpc at the typed radius
        assert w.integration_columns()["lpc_rule_radius_nena"] == 3.0
        ws.reset_radii()
        assert ws.lbl_radii.text() == "defaults (pre-specified)"
        assert w.integration_columns()["radii_exploratory"] == "no"
    finally:
        w.close()


def test_lpc_by_depth_in_the_window(app, observed):
    w = _window(observed)
    try:
        ws = w.working_scale
        before = w.integration_columns()
        assert before["lpc_file"] is None and "press" in (before["lpc_depth_note"] or "")
        ws.btn_lpc_depth.click()
        assert "lpc of the whole file" in ws.lbl_lpc_depth.text()
        assert not ws.plot_lpc.isHidden()
        cols = w.integration_columns()
        assert cols["lpc_file"] is not None and cols["lpc_z_profile"]
        assert cols["lpc_slab_differs"] in (True, False)
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------ C2 / G2 / C3 C4
def test_membrane_indicator_bands_and_both_closures(app, observed, tmp_path):
    w = _window(observed)
    try:
        w._sync_integration()
        arr = w.arrangement
        assert arr.closure_choice() == "n_locs" and arr.membrane_choice() == "dbscan_centroids"   # defaults
        # G2 before the null: the centres-level band, labelled so
        assert arr.band_source == "approx., centres"
        title = arr.plot_s.getPlotItem().titleLabel.text
        assert "approx., centres" in title
        leg = [lab for lab, _s in arr.legend_entries("S(m) periodogram of the centres")]
        assert any("approx., centres" in lab for lab in leg), leg
        # C2: the indicator (computed when asked; here by the export)
        d = arr.deformation()
        assert d is not None and d.mean_nm >= 0.0
        assert "Centroid vs localization membrane" in arr.lbl_membrane.text()
        cols = w.integration_columns()
        assert cols["membrane_deformation_mean_nm"] is not None and cols["membrane_deformation_n_flagged"] >= 0
        # C3 / C4: both closures, a tiny B
        arr.combo_closure.setCurrentIndex(2)
        assert "both side by side" in arr.lbl_estimate.text()
        arr.combo_b.setItemData(0, 4)
        assert arr.start()
        _pump(app, arr.running)
        key = arr.runs[0].key
        assert key in arr.outcomes and key + " [K closure]" in arr.outcomes
        assert arr.outcomes[key].closure == "n_locs" and arr.outcomes[key + " [K closure]"].closure == "k_clusters"
        assert arr.table.rowCount() == 2 and arr.btn_compare_closures.isEnabled()
        dlg = arr.show_closure_comparison()
        assert dlg is not None and dlg.table.rowCount() >= 7
        dlg.close()
        # G2 after the null: the band from the pipeline null
        arr._draw(key)
        assert arr.band_source.startswith("pipeline null")
        out = arr.export_curves(str(tmp_path / "null.csv"))
        with open(out["by_closure"], encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert sorted(r["null_closure"] for r in rows) == ["k_clusters", "n_locs"]
        with open(out["nulls"], encoding="utf-8") as fh:
            summ = list(csv.DictReader(fh))
        assert {r["null_closure"] for r in summ} == {"n_locs", "k_clusters"}
        assert all(r["null_membrane"] for r in summ)
        assert os.path.exists(out["closures"])
        # the row of the export carries the N_loc null and its membrane
        cols = w.integration_columns()
        assert cols["null_closure"] == "n_locs" and cols["null_membrane"].startswith("centroids of DBSCAN 25/10")
    finally:
        w.close()


def test_null_on_the_localization_membrane(app, observed):
    w = _window(observed)
    try:
        w._sync_integration()
        arr = w.arrangement
        arr.combo_membrane.setCurrentIndex(1)
        arr.combo_b.setItemData(0, 3)
        assert arr.start()
        _pump(app, arr.running)
        o = arr.outcomes[arr.runs[0].key]
        assert o.ok and o.membrane == "localizations of the slab"
        assert "localizations" in o.result.model.origins["membrane_nm"]
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------------------ A2
def test_slow_methods_ask_before_running(app, observed):
    import dataclasses
    from tools import clustering_integration as ci
    from tools.compare_analyses_ui import CompareAnalysesDialog
    w = _window(observed)
    try:
        # the null: a slow run is refused without the box, asked with it (and nothing runs on 'Cancel')
        w._sync_integration()
        arr = w.arrangement
        slow_run = dataclasses.replace(ci.run_of(observed), method="OPTICS", key="OPTICS", is_reference=False)
        arr.runs = [slow_run]
        assert not arr.start() and arr.outcomes["OPTICS"].status == "refused" and not slow_confirm.ASKED
        arr.chk_slow.setChecked(True)
        assert not arr.start()
        assert len(slow_confirm.ASKED) == 1 and "OPTICS: about" in slow_confirm.ASKED[0]
        assert "not confirmed" in arr.banner.text() and not arr.running()
        # Compare methods: a slow method ticked asks first; Cancel runs nothing
        dlg = CompareAnalysesDialog(w._compare_run, int(observed.x_slab.size), True, "DBSCAN",
                                    lambda n: {"eps": 25.0, "min_samples": 10} if n == "DBSCAN" else {}, parent=w,
                                    analysis=observed, nena_nm=12.0, c_label=14.0)
        for name, cb in dlg.checks.items():
            cb.setChecked(name in ("DBSCAN", "OPTICS"))
        dlg.start()
        assert len(slow_confirm.ASKED) == 2 and "Compare methods" in slow_confirm.ASKED[1]
        assert not dlg.running() and "not confirmed" in dlg.lbl_status.text()
        # with only fast methods nothing is asked
        for name, cb in dlg.checks.items():
            cb.setChecked(name == "DBSCAN")
        slow_confirm.ANSWER = lambda _t: True
        dlg.start()
        _pump(app, dlg.running)
        assert len(slow_confirm.ASKED) == 2
        dlg.close()
    finally:
        w.close()


def test_batch_window_slow_box_and_confirmation(app, tmp_path):
    from tools import mps_batch as mb
    from tools.batch_window import BatchWindow
    from tools.mps_settings import MPSSettings
    s = MPSSettings()
    s.cluster_method = "OPTICS"
    w = BatchWindow(s)
    try:
        assert w.check_slow_methods.text() == "Allow slow methods" and not w.check_slow_methods.isChecked()
        settings = mb.BatchSettings.from_settings(s)
        assert not w._method_runs(settings, False) and w._method_runs(settings, True)
        assert w._confirm_slow(settings, ["DBSCAN", "voronoi_csr"], 199, 3) is False
        text = slow_confirm.ASKED[-1]
        assert "The Batch" in text and "OPTICS" in text and "its null, B = 199" in text and "voronoi_csr" in text
        slow_confirm.ASKED[:] = []
        s2 = MPSSettings()
        assert w._confirm_slow(mb.BatchSettings.from_settings(s2), ["DBSCAN"], 0, 1) is True
        assert not slow_confirm.ASKED                      # nothing slow: never asked
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------------- G4 / G5
def test_regularity_tab_warns_of_the_edge_and_says_the_label_is_descriptive(app, observed, monkeypatch):
    from tools import clustering_integration as ci
    monkeypatch.setattr(ci, "edge_note", lambda _a: "2 cluster(s) removed at the ROI edge")
    w = _window(observed)
    try:
        w._sync_integration()
        arr = w.arrangement
        assert ci.EDGE_WARNING in arr.lbl_reading.text() and "WARNING" in arr.lbl_reading.text()
        arr.combo_b.setItemData(0, 3)
        assert arr.start()
        _pump(app, arr.running)
        text = arr.lbl_reading.text()
        if arr.outcomes[arr.runs[0].key].gap is not None:
            assert ci.EDGE_WARNING in text and "descriptive only" in text and "4 of 12" in text
    finally:
        w.close()


def test_batch_window_null_closure_combo_writes_the_settings(app):
    from tools.batch_window import BatchWindow
    from tools.mps_settings import MPSSettings
    s = MPSSettings()
    saved = []
    w = BatchWindow(s, save=lambda st: saved.append(st.batch_null_closure))
    try:
        assert w.combo_null_closure.currentData() == "n_locs"
        w.combo_null_closure.setCurrentIndex(2)
        assert s.batch_null_closure == "both" and saved[-1] == "both"
    finally:
        w.close()
