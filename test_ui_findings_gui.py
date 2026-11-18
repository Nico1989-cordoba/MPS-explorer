"""Findings of the user's manual test of 2026-10-22 (branch feat/ui-findings), offscreen.

F2: the Voronoi hint under the method's parameters is scrolled into sight when the method is chosen.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_ui_findings_gui.py -q
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _pump(seconds: float = 0.2) -> None:
    app = _app()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def _settle(widget) -> None:
    """Delete the widget inside this test, so the garbage collector never destroys it while a later test runs Qt."""
    from PyQt5 import QtCore
    app = _app()
    widget.close()
    widget.deleteLater()
    for _ in range(10):
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.fixture()
def layout_dir(tmp_path, monkeypatch):
    """The layout and settings files in a temporary folder: no test writes beside the program."""
    from tools import mps_settings
    original = mps_settings.settings_path
    monkeypatch.setattr(mps_settings, "settings_path",
                        lambda directory=None: original(directory or str(tmp_path)))
    return tmp_path


def _results_window(size=(1366, 700)):
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    _app()
    w = MPSResultsWindow(None, links=WindowLinks(), rerun_callback=lambda **k: None)
    w.resize(*size)
    w.show()
    _pump()
    return w


def _choose(w, method: str) -> None:
    p = w.method_panel
    p.chk_experimental.setChecked(True)
    i = p.combo.findText(method)
    assert i >= 0, method
    p.combo.setCurrentIndex(i)
    p.combo.activated.emit(i)
    _pump(0.4)


# ---------------------------------------------------------------------------------------------------------------
# F2
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["voronoi_factor", "voronoi_csr"])
def test_f2_voronoi_hint_scrolled_into_sight(layout_dir, method):
    from PyQt5 import QtCore
    w = _results_window((1366, 700))
    try:
        _choose(w, method)
        p = w.method_panel
        assert p.lbl_hint.isVisible() and "Cells" in p.lbl_hint.text()
        vp = w.strip_scroll.viewport()
        top = p.lbl_hint.mapTo(vp, QtCore.QPoint(0, 0)).y()
        assert 0 <= top < vp.height(), (top, vp.height())
        # the first lines of the hint are in sight, not only its top pixel
        assert top + min(p.lbl_hint.height(), 30) <= vp.height(), (top, p.lbl_hint.height(), vp.height())
    finally:
        _settle(w)


def test_f2_dbscan_does_not_scroll(layout_dir):
    """A method with no hint, no slow note and no box leaves the strip where it was."""
    w = _results_window((1366, 700))
    try:
        _choose(w, "DBSCAN")
        assert w.strip_scroll.verticalScrollBar().value() == 0
    finally:
        _settle(w)


# ---------------------------------------------------------------------------------------------------------------
# F4
# ---------------------------------------------------------------------------------------------------------------
def test_f4_collapse_expand_and_remember(layout_dir):
    w = _results_window((1366, 700))
    try:
        _choose(w, "voronoi_factor")
        tall = w.strip_scroll.height()
        assert not w.method_collapsed() and w.method_panel.isVisible()
        w.method_panel.btn_collapse.click()
        _pump(0.3)
        assert w.method_collapsed() and not w.method_panel.isVisible() and w.method_bar.isVisible()
        assert "voronoi_factor" in w.method_bar.btn_expand.text()
        assert w.strip_scroll.height() < tall, (w.strip_scroll.height(), tall)
        # the panel keeps its values while collapsed
        assert w.method_panel.current_method() == "voronoi_factor"
    finally:
        _settle(w)
    # a new window opens collapsed (mps_ui_layout.json in the temporary folder)
    from tools.mps_ui_layout import layout_path
    assert os.path.dirname(layout_path()) == str(layout_dir)
    w2 = _results_window((1366, 700))
    try:
        assert w2.method_collapsed()
        w2.method_bar.btn_expand.click()
        _pump(0.2)
        assert not w2.method_collapsed() and w2.method_panel.isVisible()
    finally:
        _settle(w2)
    w3 = _results_window((1366, 700))
    try:
        assert not w3.method_collapsed()
    finally:
        _settle(w3)


def test_f4_a_run_goes_on_while_collapsing_and_expanding(layout_dir):
    import threading
    from tools import mps_background_run as bg
    from tools import mps_cancel
    from tools.mps_background_run import PreparedRun
    from tools.mps_results_window import MPSResultsWindow, WindowLinks

    release = threading.Event()
    started = []
    finished = []

    def compute():
        started.append(1)
        i = 0
        while not release.is_set():
            mps_cancel.check("perimeter, 2-opt from every start", i % 50, 50)
            i += 1
            time.sleep(0.01)
        return None

    runs = [PreparedRun(compute=compute, finish=lambda a: finished.append(1), words="a test run")]
    w = MPSResultsWindow(None, links=WindowLinks(), rerun_callback=lambda **k: None,
                         background=lambda **kw: runs.pop(0))
    w.resize(1366, 700)
    w.show()
    _pump()
    try:
        assert w.start_background({"cluster_method": "OPTICS"})
        _pump(1.3)
        assert w.busy()
        # the collapse button works during the run, the rest of the strip does not
        assert w.method_panel.btn_collapse.isEnabled() and not w.spin_eps.isEnabled()
        w.method_panel.btn_collapse.click()
        _pump(1.2)
        assert w.method_collapsed() and w.busy()
        bar = w.method_bar
        assert bar.btn_cancel.isVisible() and bar.btn_cancel.isEnabled()
        assert "Running a test run" in bar.lbl_status.text()
        assert "perimeter" in bar.lbl_status.toolTip()                 # the stage, from tools.mps_progress
        bar.btn_expand.click()
        _pump(0.3)
        assert not w.method_collapsed() and w.busy() and len(started) == 1     # not stopped, not restarted
        w.method_panel.btn_collapse.click()
        _pump(0.2)
        release.set()
        deadline = time.monotonic() + 10
        while w.busy() and time.monotonic() < deadline:
            _pump(0.05)
        assert not w.busy() and finished == [1] and len(started) == 1
        assert not w.method_bar.btn_cancel.isVisible() and w.method_bar.lbl_status.text() == ""
        assert w.spin_eps.isEnabled() and w.method_bar.isEnabled()
    finally:
        release.set()
        deadline = time.monotonic() + 5
        while bg.live_runs() and time.monotonic() < deadline:
            _pump(0.05)
        _settle(w)


def test_f4_cancel_from_the_collapsed_bar(layout_dir):
    from tools import mps_background_run as bg
    from tools import mps_cancel
    from tools.mps_background_run import PreparedRun
    from tools.mps_results_window import MPSResultsWindow, WindowLinks

    def compute():
        end = time.monotonic() + 20
        while time.monotonic() < end:
            mps_cancel.check("randomization", 1, 10)
            time.sleep(0.01)
        return None

    runs = [PreparedRun(compute=compute, finish=lambda a: None, words="a test run")]
    w = MPSResultsWindow(None, links=WindowLinks(), rerun_callback=lambda **k: None,
                         background=lambda **kw: runs.pop(0))
    w.resize(1366, 700)
    w.show()
    _pump()
    try:
        w.set_method_collapsed(True)
        assert w.start_background({"cluster_method": "OPTICS"})
        _pump(0.3)
        w.method_bar.btn_cancel.click()
        deadline = time.monotonic() + 10
        while w.busy() and time.monotonic() < deadline:
            _pump(0.05)
        assert not w.busy() and "Cancelled" in w.run_progress.label.text()
    finally:
        deadline = time.monotonic() + 5
        while bg.live_runs() and time.monotonic() < deadline:
            _pump(0.05)
        _settle(w)
