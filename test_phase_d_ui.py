"""Phase D (D0) interface fixes: the 'Compare methods' worker's life cycle, the generic parameter panel, the batch's
comparison box. Synthetic data only (R8); offscreen Qt.

Run: py -3 -m pytest test_phase_d_ui.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


_APP = []


def _app():
    """The QApplication, kept alive for the whole module (a collected QApplication takes Qt down with it)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


# A crash of Qt (a QThread destroyed while it runs) kills the process, so the
# close-while-running scenario runs in a child process and the test reads
# its exit code and its last line.
_CLOSE_WHILE_RUNNING = textwrap.dedent(r"""
    import gc, os, sys, time, types
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, sys.argv[1])
    from PyQt5 import QtCore, QtWidgets
    app = QtWidgets.QApplication([])
    from tools import cluster_method_ui as ui

    calls = []

    def run_fn(method, params):
        calls.append(method)
        time.sleep(0.4)
        return types.SimpleNamespace(clusters_per_um=5.0, median_1nn_nm=190.0, median_r_eff_nm=20.0,
                                     occupancy_percent=60.0, n_clusters_kept=40, cluster_scale_nm=25.0)

    def open_run_close(close_with):
        dlg = ui.CompareMethodsDialog(run_fn, 1000, True, "DBSCAN", lambda n: {}, verdicts={})
        for name, cb in dlg.checks.items():
            cb.setChecked(name in ("DBSCAN", "FOCAL", "ToMATo", "OPTICS"))
        state = {}

        def run():
            dlg.btn_run.click()
            state["run_disabled"] = not dlg.btn_run.isEnabled()
            QtCore.QTimer.singleShot(100, close)

        def close():
            close_with(dlg)
            state["still_open_while_running"] = dlg.isVisible() and dlg.running()
            state["close_disabled"] = not dlg.btn_close.isEnabled()

        QtCore.QTimer.singleShot(50, run)
        result = dlg.exec_()
        state["result"] = result
        state["running_after"] = dlg.running()
        return dlg, state

    # 1. Close button, 2. Escape (reject), 3. the window's X (closeEvent), each followed by dropping the
    # dialog and collecting it: the old dialog used to take its running QThread with it (review A1).
    for how, fn in (("button", lambda d: d.btn_close.click()), ("reject", lambda d: d.reject()),
                    ("x", lambda d: d.close())):
        calls.clear()
        dlg, st = open_run_close(fn)
        assert st["run_disabled"], (how, st)
        assert st["still_open_while_running"] and st["close_disabled"], (how, st)
        assert st["result"] == QtWidgets.QDialog.Rejected and not st["running_after"], (how, st)
        assert len(calls) == 1, (how, calls)          # cancelled before the second method
        del dlg
        gc.collect()
        for _ in range(20):
            app.processEvents()
            time.sleep(0.01)
        assert ui.live_workers() == 0, how

    # The review's own reproduction (r7), without exec_: Run, Close, drop the dialog at once. The old code
    # destroyed the running QThread here and Qt aborted the process.
    calls.clear()
    dlg = ui.CompareMethodsDialog(run_fn, 50000, True, "DBSCAN", lambda n: {}, verdicts={})
    dlg.start()
    dlg.btn_close.click()
    dlg = None
    gc.collect()
    t0 = time.time()
    while (ui.live_workers() or time.time() - t0 < 0.6) and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.01)
    assert ui.live_workers() == 0 and len(calls) <= 1, calls      # cancelled before or after the first

    # Reopening after all that: a full run to the end, then Close.
    calls.clear()
    dlg = ui.CompareMethodsDialog(run_fn, 1000, True, "DBSCAN", lambda n: {}, verdicts={})
    for name, cb in dlg.checks.items():
        cb.setChecked(name in ("DBSCAN", "FOCAL"))
    dlg.start()
    t0 = time.time()
    while dlg.running() and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.01)
    assert not dlg.running() and dlg.table.rowCount() == 2 and dlg.btn_use.isEnabled()
    dlg.reject()
    del dlg
    gc.collect()
    print("survived")
""")


def test_closing_the_compare_dialog_while_running_never_crashes(tmp_path):
    script = tmp_path / "close_while_running.py"
    script.write_text(_CLOSE_WHILE_RUNNING, encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, str(script), HERE], capture_output=True, text=True, timeout=180, env=env)
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-2000:], proc.stderr[-4000:])
    assert proc.stdout.strip().splitlines()[-1] == "survived"


def test_worker_errors_reach_the_dialog():
    app = _app()
    import time
    from tools import cluster_method_ui as ui

    def run_fn(method, params):
        raise RuntimeError(f"boom in {method}")

    dlg = ui.CompareMethodsDialog(run_fn, 500, False, "DBSCAN", lambda n: {}, verdicts={})
    dlg.start()
    t0 = time.time()
    while dlg.running() and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.01)
    assert not dlg.running()
    assert dlg.errors and "boom in DBSCAN" in dlg.errors[0]
    assert "failed" in dlg.lbl_status.text()
    assert "Traceback" in dlg.table.item(0, 0).toolTip()
    assert dlg.selected_methods() == []            # an errored row is never 'used'
    dlg.reject()


def test_estimate_is_superlinear_for_optics_and_says_so():
    _app()
    from tools import cluster_method_ui as ui
    one = ui.estimate_seconds(["OPTICS"], 1000)
    eight = ui.estimate_seconds(["OPTICS"], 8000)
    lin = ui.estimate_seconds(["DBSCAN"], 8000) / ui.estimate_seconds(["DBSCAN"], 1000)
    assert eight / one > 10 and lin == pytest.approx(8.0)
    dlg = ui.CompareMethodsDialog(lambda m, p: None, 8000, True, "OPTICS", lambda n: {}, verdicts={})
    assert "or more" in dlg.lbl_estimate.text() and "OPTICS" in dlg.lbl_estimate.text()
    assert "lower bound" in dlg.lbl_estimate.toolTip()
    assert ui.CANCEL_NOTE in dlg.btn_cancel.toolTip()
    dlg.reject()


# ----------------------------------------------------------------- the generic parameter panel (review M5, B3)
class _FakeMethod:
    def __init__(self, n_steps: int = 3, max_reach: float = float("inf"), floor=None) -> None:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.n_steps, self.max_reach, self.floor = n_steps, max_reach, floor


def test_param_specs_come_from_the_registry_alone():
    from typing import Optional  # noqa: F401  (annotation of the fake below)
    from tools import cluster_method_ui as ui
    from tools.clustering_strategies import StrategyInfo

    class Fake(_FakeMethod):
        def __init__(self, n_steps: int = 3, max_reach: float = float("inf"),
                     floor: "Optional[float]" = None, new_knob: float = 1.5) -> None:
            super().__init__(n_steps, max_reach, floor)
            self.new_knob = new_knob

    info = StrategyInfo("fake", Fake, "test", "a fake method", {
        "n_steps": {"value": 3, "origin": "ad hoc"}, "max_reach": {"value": None, "origin": "ad hoc"},
        "floor": {"value": None, "origin": "ad hoc"}, "new_knob": {"value": 1.5, "origin": "ad hoc"},
        "mode": {"value": "x", "origin": "ad hoc"}})
    specs = {s.key: s for s in ui.param_specs(info)}
    assert set(specs) == {"n_steps", "max_reach", "floor", "new_knob"}      # a new numeric parameter appears
    assert specs["n_steps"].kind == "int" and specs["n_steps"].lo == 1
    assert specs["max_reach"].kind == "float" and specs["max_reach"].none_text == "no limit"
    assert specs["floor"].kind == "float" and specs["floor"].none_text == "auto"


def test_panel_none_is_no_limit_and_values_are_validated(monkeypatch):
    _app()
    from tools import cluster_method_ui as ui
    from tools.clustering_strategies import strategy_registry
    p = ui.MethodPanel()
    # A None default ("no limit"): OPTICS max_eps was None until 2026-10-16 (now 100 nm, REVIEW_D0 M5); the test
    # declares it None so the "no limit" step is still exercised.
    info = strategy_registry()["OPTICS"]
    monkeypatch.setitem(info.defaults, "max_eps", {"value": None, "origin": "ad hoc: test (None = no limit)"})
    p.set_state("OPTICS", {}, True)
    max_eps = p._spins["max_eps"]
    assert max_eps.text() == "no limit" and "max_eps" not in p.current_params()   # None = constructor's own
    max_eps.setValue(100.0)
    assert p.current_params()["max_eps"] == 100.0 and p.validation_error() == ""
    max_eps.setValue(max_eps.minimum())
    assert "max_eps" not in p.current_params()
    p._spins["min_samples"].setValue(1)                  # OPTICS rejects it: flagged, not applied
    assert "min_samples" in p.validation_error() and p.lbl_invalid.isVisibleTo(p)
    assert p.current_params()["min_samples"] == 10
    p._spins["min_samples"].setValue(5)
    assert p.validation_error() == "" and p.current_params()["min_samples"] == 5
    assert p._spins["xi"].maximum() < 1.0                 # fractions stop below 1

    # The registry's own default being a number (max_eps = 100 nm since 2026-10-16) needs no code in the panel.
    monkeypatch.setitem(info.defaults, "max_eps", {"value": 100.0, "origin": "ad hoc: test"})
    p.set_state("FOCAL", {}, True)
    p.set_state("OPTICS", {}, True)
    assert p._spins["max_eps"].value() == 100.0 and p._spins["max_eps"].specialValueText() == ""
    assert p.current_params()["max_eps"] == 100.0

    p.set_state("FOCAL", {"min_l": 0}, True)             # a saved value the method rejects
    assert "min_l" in p.validation_error() or p._spins["min_l"].value() >= 1
    assert p.current_params()["min_l"] >= 1
    p.set_state("voronoi_factor", {}, True)
    assert isinstance(p._spins["density_rank"], ui.QtWidgets.QComboBox)
    assert [p._spins["density_rank"].itemData(i) for i in range(2)] == [0, 1]
    p.set_state("voronoi_csr", {}, True)
    assert p.lbl_slow.isVisibleTo(p) and "slow" in p.lbl_slow.text()


# ----------------------------------------------------------------- the batch's comparison box (review of D0, M4; D12)
def test_methods_to_compare_is_the_users_choice():
    from tools import mps_batch as mb
    from tools.mps_settings import MPSSettings
    s = MPSSettings()
    assert mb.methods_to_compare(mb.BatchSettings.from_settings(s)) == []          # nothing compared yet
    b = mb.BatchSettings.from_settings(s)
    b.cluster_methods_compared = ["FOCAL", "ToMATo", "FOCAL", "no such method", "External labels"]
    assert mb.methods_to_compare(b) == ["DBSCAN", "FOCAL", "ToMATo"]               # DBSCAN as the reference
    b.cluster_methods_compared = ["ToMATo", "DBSCAN"]
    assert mb.methods_to_compare(b) == ["ToMATo", "DBSCAN"]


def test_batch_compare_box_runs_only_the_chosen_methods(monkeypatch, tmp_path):
    _app()
    from tools import mps_batch as mb
    from tools.batch_window import BatchWindow
    from tools.mps_settings import MPSSettings
    settings = MPSSettings()
    settings.experimental_methods_enabled = False
    w = BatchWindow(settings)
    box = w.check_compare_methods
    assert not box.isEnabled() and not box.isChecked()
    assert "Compare methods" in box.toolTip() and "primero" in box.toolTip()

    settings.cluster_methods_compared = ["FOCAL"]          # what 'Compare methods...' recorded
    w._show_settings()
    assert box.isEnabled() and "DBSCAN, FOCAL" in box.toolTip()

    seen = {}
    monkeypatch.setattr(mb, "run_batch", lambda items, st, **kw: None)

    def fake_compare(items, st, methods, **kw):
        seen["methods"] = list(methods)
        return []

    monkeypatch.setattr(mb, "compare_methods", fake_compare)
    w.plan = mb.BatchPlan(root=str(tmp_path), files=[
        mb.BatchFile(path=str(tmp_path / "a.hdf5"), identity=mb.AxonIdentity())])
    w._fill_table()
    w.set_ticked(0, True)
    box.setChecked(True)
    w.run(ask=lambda *a, **k: None)
    w.wait(60)
    # Never every registered method: exactly the chosen one plus the DBSCAN reference, although
    # 'Experimental clustering methods' is off (the user chose FOCAL in the comparison).
    assert seen["methods"] == ["DBSCAN", "FOCAL"]

    settings.cluster_methods_compared = []
    w._show_settings()
    assert not box.isEnabled() and not box.isChecked()
    seen.clear()
    w.run(ask=lambda *a, **k: None)
    w.wait(60)
    assert seen == {}
    w.close()


# ----------------------------------------------------------------- m1: the threshold shown, with and without a method


def _sim_analysis(**kw):
    from tools.mps_analysis import analyze_axon
    from tools.mps_simulate_axon import SimConfig, simulate_axon
    sim = simulate_axon(SimConfig(), 5)
    return analyze_axon(sim.x_nm, sim.y_nm, sim.z_nm, source_name="sim.hdf5", pixel_size_nm=113.0,
                        pixel_size_source="yaml", run_randomization=False, **kw)


def test_mahalanobis_spin_follows_the_store_or_the_analysis():
    """Review m1: the threshold shown comes from the store when there is one (never overwritten by the analysis
    because no method_state was given), and from the analysis when there is none (even with a method_state)."""
    _app()
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    a = _sim_analysis(mahalanobis_threshold=2.5)

    links = WindowLinks(store=lambda values: None, params=lambda: (a.eps_nm, int(a.min_samples),
                                                                   a.slab_half_width_nm, 4.0))
    w = MPSResultsWindow(a, links=links)
    assert w.spin_maha.value() == pytest.approx(4.0)            # the store's one value, not the analysis's 2.5
    w.close()

    links = WindowLinks(store=lambda values: None, method_state=lambda: ("DBSCAN", {}, False, []))
    w = MPSResultsWindow(a, links=links)
    assert w.spin_maha.value() == pytest.approx(2.5)            # no params: the analysis's, method or not
    w.close()


def test_compare_run_builds_the_contour_like_the_analysis_shown(monkeypatch):
    """Review m3: 'Compare methods' clusters with the contour of the analysis shown (2-opt from every start, the
    same guide), so its DBSCAN row is the analysis shown; the dialog says what it still leaves out."""
    _app()
    import tools.mps_analysis as ma
    from tools.cluster_method_ui import CompareMethodsDialog
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    a = _sim_analysis()
    w = MPSResultsWindow(a, links=WindowLinks())
    b = w._compare_run("DBSCAN", {})
    assert b.n_clusters_kept == a.n_clusters_kept
    assert b.perimeter.perimeter_nm == pytest.approx(a.perimeter.perimeter_nm)

    seen = {}
    real = ma.analyze_axon

    def spy(*args, **kw):
        seen.update(kw)
        return real(*args, **kw)

    monkeypatch.setattr(ma, "analyze_axon", spy)
    w._compare_run("DBSCAN", {})
    assert "all_starts" not in seen or seen["all_starts"] is True
    assert "contour_guide" in seen
    w.close()

    d = CompareMethodsDialog(lambda m, p: None, 1000, False, "DBSCAN", lambda n: {})
    text = " ".join(lbl.text() for lbl in d.findChildren(type(d.lbl_status)))
    assert "2-opt from every start" in text and "Picasso pick" in text
    d.close()


def test_settings_fall_back_to_dbscan_when_the_methods_package_is_missing(monkeypatch, caplog):
    """Review m4: a settings file naming HDBSCAN, read on a machine without ``hdbscan``, falls back to DBSCAN with
    a warning (it used to keep HDBSCAN, so every analysis failed with ImportError while the combo showed DBSCAN)."""
    import logging
    from tools.clustering_strategies import strategy_registry
    from tools.mps_settings import MPSSettings
    monkeypatch.setattr(strategy_registry()["HDBSCAN"], "available", False)
    s = MPSSettings()
    s.cluster_method = "HDBSCAN"
    s.cluster_params = {"min_cluster_size": 9}
    s.cluster_methods_compared = ["DBSCAN", "HDBSCAN"]
    with caplog.at_level(logging.WARNING):
        s.validate()
    assert s.cluster_method == "DBSCAN" and s.cluster_params == {}
    assert any("not installed" in r.getMessage() for r in caplog.records)
    assert s.cluster_methods_compared == ["DBSCAN", "HDBSCAN"]     # the record of what was compared is kept

    monkeypatch.setattr(strategy_registry()["HDBSCAN"], "available", True)
    s.cluster_method = "HDBSCAN"
    s.cluster_params = {"min_cluster_size": 9}
    s.validate()
    assert s.cluster_method == "HDBSCAN" and s.cluster_params == {"min_cluster_size": 9}
