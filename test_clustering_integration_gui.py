"""The clustering integration in the analysis window (offscreen Qt, synthetic data only, rule R8): the Working
scale panel, 'Compare methods...' with the analyses and the DBSCAN rule (the double table, the export of the long
table), the Perimeter arrangement tab (a null per method, re-running its own method, Cancel, Show synthetic axon,
the curves' export), the probable-fusion rows of the map, and the axon table's v13 block from the window.

Run: QT_QPA_PLATFORM=offscreen py -3 -m pytest test_clustering_integration_gui.py -q
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from PyQt5 import QtWidgets  # noqa: E402

from tools import mps_pipeline_null as pn  # noqa: E402

SMALL = (900.0, 700.0)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def observed():
    an, _ax = pn.observe_synthetic(pn.synthetic_model(SMALL), 3)
    return an


def _pump(app, cond, timeout=300.0):
    t0 = time.time()
    while cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()


def _window(observed):
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    links = WindowLinks(nena=lambda: (12.0, "this file (synthetic)"))
    return MPSResultsWindow(observed, rerun_callback=lambda **kw: observed, links=links)


def test_working_scale_rule_and_the_tab(app, observed):
    from tools.arrangement_ui import TAB_TITLE
    from tools import mps_axon_map_layers as L
    w = _window(observed)
    try:
        assert TAB_TITLE in [w.tabs.tabText(i) for i in range(w.tabs.count())]
        ws = w.working_scale
        marks = [ws.table.item(r, 0).text() for r in range(ws.table.rowCount())]
        assert "d50 (half of pairs resolved)" in marks and "near-total fusion" in marks
        assert ws.rule is not None and ws.rule.min_samples == 6 and "compatibility 25/10" in ws.lbl_rule.text()
        assert "min samples 6 with c_label 14 (11 with 26)" in ws.lbl_rule.text(), ws.lbl_rule.text()
        ws.spin_c_label.setValue(26.0)                       # c_label is editable: the rule follows
        assert ws.rule.min_samples == 11 and ws.badge_c_label.badge == "user"
        # the parenthesis shows the other c_label (manual test of 2026-10-23: it repeated 26)
        assert "min samples 11 with c_label 26 (6 with 14)" in ws.lbl_rule.text(), ws.lbl_rule.text()
        ws.spin_c_label.setValue(14.0)
        assert ws.badge_c_label.badge == "paper" and ws.rule.min_samples == 6
        groups = {g.key: g for g in L.build_map(w._map_inputs())}
        assert "events" in groups and [lay.key for lay in groups["events"].layers] == ["fus14", "fus26"]
        assert w.btn_null_shortcut.isEnabled()
        w._on_null_shortcut()
        assert w.tabs.currentWidget() is w.arrangement_scroll
        assert "B = 199" in w.arrangement.lbl_estimate.text() and "per animal" in w.arrangement.lbl_estimate.text()
    finally:
        w.close()


def test_compare_with_analyses_rule_and_null_per_method(app, observed, tmp_path):
    from tools.compare_analyses_ui import CompareAnalysesDialog
    w = _window(observed)
    try:
        ws = w.working_scale
        dlg = CompareAnalysesDialog(w._compare_run, int(observed.x_slab.size), True, "DBSCAN",
                                    lambda n: {"eps": 25.0, "min_samples": 10} if n == "DBSCAN" else {}, parent=w,
                                    analysis=observed, nena_nm=12.0, c_label=14.0, rule=ws.rule, rule26=ws.rule26)
        for name, cb in dlg.checks.items():
            cb.setChecked(name in ("DBSCAN", "FOCAL"))
        assert dlg.analysis_checks["events"].isChecked() and not dlg.analysis_checks["sweep"].isChecked()
        dlg.chk_rule.setChecked(True)
        dlg.start()
        _pump(app, dlg.running)
        res = dlg.comparison_result
        assert res is not None and res.runs[0].key == "DBSCAN 25/10"
        assert dlg.table.rowCount() == 3 and dlg.table_double.rowCount() == 2
        assert "DBSCAN rule (c_label 14)" in dlg.labels.values()
        assert dlg.run_for("DBSCAN rule (c_label 14)").params["min_samples"] == 6
        paths = dlg.export_tables(str(tmp_path / "axon"))
        assert os.path.basename(paths["long"]) == "axon_methods_long.csv"
        dlg.close()
        w.method_comparison = res
        w._sync_integration()
        arr = w.arrangement
        assert [r.key for r in arr.runs] == ["DBSCAN 25/10", "FOCAL", "DBSCAN rule (c_label 14)"]
        assert ws.double.isVisible() or ws.double.rowCount() == 2
        arr.combo_b.setItemData(0, 4)                        # a tiny B for the test
        arr.start()
        _pump(app, arr.running)
        assert arr.table.rowCount() == 3
        assert all(o.ok for o in arr.outcomes.values()), {k: o.message for k, o in arr.outcomes.items()}
        assert arr.outcomes["FOCAL"].result.method == "FOCAL"          # each method's own null
        assert "NOT INTERPRETABLE" in arr.banner.text()               # B = 4 cannot reach 0.05
        dlg2 = arr.show_synthetic()
        assert dlg2 is not None
        img = dlg2.export_image(str(tmp_path / "synth.png"))
        assert os.path.exists(img)
        dlg2.close()
        out = arr.export_curves(str(tmp_path / "null.csv"))
        assert os.path.exists(out["ripley"]) and os.path.exists(out["nulls"])
        cols = w.integration_columns()
        assert cols["null_B"] == 4 and cols["null_method"] == observed.cluster_method
        assert "DBSCAN rule (c_label 14)" in cols["methods_compared"]
        assert set(w.arrangement.figure_plots()) <= set(w._plots())
    finally:
        w.close()


def test_null_cancel_keeps_what_ran(app, observed):
    w = _window(observed)
    try:
        arr = w.arrangement
        arr.combo_b.setItemData(0, 60)
        arr.start()
        t0 = time.time()
        while arr.running() and arr.progress.value() < 2 and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.02)
        arr.cancel()
        _pump(app, arr.running)
        assert not arr.running()
        assert [o.status for o in arr.outcomes.values()] == ["cancelled"]
    finally:
        w.close()


def test_slow_methods_are_refused_in_the_tab_unless_ticked(app, observed):
    from tools import multimethod as MM
    w = _window(observed)
    try:
        arr = w.arrangement
        slow = MM.MethodRun(method="emitters_bagol_like", params={}, key="Emitters", analysis=observed,
                            labels=observed.labels)
        arr.set_context(observed, arr.runs + [slow], 12.0, 14.0)
        assert "Refused as slow" in arr.lbl_estimate.text()
        arr.runs = [slow]
        assert arr.start() is False
        assert arr.outcomes["Emitters"].status == "refused"
    finally:
        w.close()


def test_the_panel_starts_at_the_stored_c_label_and_stores_an_edit(app, observed):
    """2026-10-29: the c_label typed in the Working scale panel is written to the settings (through the store) and
    the panel of the next window starts at it; the Batch reads it from the same settings."""
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    stored: list = []
    links = WindowLinks(nena=lambda: (12.0, "this file (synthetic)"), c_label=lambda: 26.0, store=stored.append)
    w = MPSResultsWindow(observed, rerun_callback=lambda **kw: observed, links=links)
    try:
        assert w.working_scale.c_label() == 26.0 and not stored        # starting at it stores nothing
        w.working_scale.spin_c_label.setValue(19.0)
        app.processEvents()
        assert {"c_label": 19.0} in stored
    finally:
        w.close()
