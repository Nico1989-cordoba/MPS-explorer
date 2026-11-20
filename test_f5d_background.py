"""F5d: the MPS analysis window's parameter strip scrolls (the Emitters box is reachable on a laptop screen) and a
slow method runs in a worker thread with its time and a Cancel button. Synthetic data only (R8); offscreen Qt.

Run: py -3 -m pytest test_f5d_background.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))
_APP: list = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _pump(seconds: float = 0.05) -> None:
    app = _app()
    end = time.monotonic() + seconds
    while True:
        app.processEvents()
        if time.monotonic() >= end:
            break
        time.sleep(0.005)


def _pump_until(cond, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        _pump(0.02)
        if cond():
            return True
    return False


def _ring(seed: int = 0, n_clusters: int = 16, radius_nm: float = 400.0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    xs, ys = [], []
    for a in ang:
        xs.append(radius_nm * np.cos(a) + rng.normal(0, 12, 25))
        ys.append(radius_nm * np.sin(a) + rng.normal(0, 12, 25))
    x = np.concatenate(xs) + 2000.0
    y = np.concatenate(ys) + 2000.0
    z = rng.normal(300.0, 20.0, x.size)
    return x, y, z


@pytest.fixture(scope="module")
def analysis():
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring()
    return analyze_axon(x, y, z, run_randomization=False)


@pytest.fixture
def quiet_boxes(monkeypatch):
    """Message boxes recorded instead of shown (a modal box offscreen would wait for ever)."""
    from PyQt5 import QtWidgets
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, kind,
                            staticmethod(lambda *a, _k=kind, **k: (shown.append((_k,) + a[1:3]),
                                                                    QtWidgets.QMessageBox.Ok)[1]))
    return shown


# ----------------------------------------------------------------- A: the strip scrolls
def _window(analysis=None, **kw):
    _app()
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    kw.setdefault("rerun_callback", lambda **k: analysis)
    return MPSResultsWindow(analysis, links=WindowLinks(), **kw)


def _choose(w, method: str) -> None:
    """As the user does: the experimental box, then the method in the combo (activated)."""
    p = w.method_panel
    p.chk_experimental.setChecked(True)
    i = p.combo.findText(method)
    if i < 0:
        i = p.combo.findData(method)
    assert i >= 0, [p.combo.itemText(k) for k in range(p.combo.count())]
    p.combo.setCurrentIndex(i)
    p.combo.activated.emit(i)


def test_the_strip_is_a_vertical_scroll_area_and_the_window_minimum_does_not_grow():
    from PyQt5 import QtCore, QtWidgets
    w = _window()
    w.resize(1366, 700)
    w.show()
    _pump(0.1)
    sc = w.findChild(QtWidgets.QScrollArea, "param_strip_scroll")
    assert sc is w.strip_scroll and sc.widgetResizable()
    assert sc.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
    assert sc.widget().objectName() == "param_strip"
    low_dbscan = w.minimumSizeHint().height()
    w.method_panel.set_state("emitters_bagol_like", {}, True, [])
    _pump(0.1)
    low_emitters = w.minimumSizeHint().height()
    # before F5d the Emitters box raised the window's minimum to ~1,275 px, beyond a laptop screen
    assert low_emitters == low_dbscan, (low_dbscan, low_emitters)
    assert low_emitters < 700
    assert sc.horizontalScrollBar().maximum() == 0
    assert sc.scrolls(), "the Emitters box does not fit in 700 px: the strip scrolls"
    w.close()


@pytest.mark.parametrize("size", [(1366, 700), (1280, 640), (1920, 1000)])
def test_choosing_emitters_brings_its_box_into_sight(size):
    from PyQt5 import QtCore, QtWidgets
    w = _window()
    w.resize(*size)
    w.show()
    _pump(0.1)
    _choose(w, "emitters_bagol_like")
    _pump(0.2)
    p = w.method_panel
    assert p.layer_box.isVisible() and p.lbl_slow.isVisible()
    # the tour and "Explain this" find the widgets by their names, inside the scroll area too
    for name in ("emitters_layer_box", "btn_emitters_run", "combo_emitters_buffer", "method_slow_note",
                 "combo_method"):
        assert w.findChild(QtWidgets.QWidget, name) is not None, name
    vp = w.strip_scroll.viewport()
    top = p.layer_box.mapTo(vp, QtCore.QPoint(0, 0)).y()
    # the box's title and its first rows (the imaging buffer) are inside the visible part of the strip
    assert 0 <= top and top + 100 <= vp.height(), (top, vp.height())
    w.close()


def test_a_more_line_or_a_short_window_never_pushes_the_plots_out():
    w = _window()
    w.resize(1366, 640)
    w.show()
    _pump(0.1)
    w.method_panel.set_state("emitters_bagol_like", {}, True, [])
    w.more_randomization.button.setChecked(True)
    _pump(0.1)
    assert w.height() <= 640 + 1
    assert w.splitter.height() > 300, w.splitter.height()
    w.close()


# ----------------------------------------------------------------- B: cancellation points
def test_cancel_check_is_a_noop_outside_a_run():
    from tools import mps_cancel
    mps_cancel.check("anywhere")
    assert not mps_cancel.requested()
    with mps_cancel.cancellable(lambda: False):
        mps_cancel.check()
    with mps_cancel.cancellable(lambda: True):
        assert mps_cancel.requested()
        with pytest.raises(mps_cancel.AnalysisCancelled):
            mps_cancel.check("here")
    mps_cancel.check("after the block")
    assert issubclass(mps_cancel.AnalysisCancelled, BaseException)
    assert not issubclass(mps_cancel.AnalysisCancelled, Exception), "an 'except Exception' must not swallow it"


def test_a_thread_does_not_inherit_the_cancel_scope():
    from tools import mps_cancel
    seen = []
    with mps_cancel.cancellable(lambda: True):
        t = threading.Thread(target=lambda: seen.append(mps_cancel.requested()))
        t.start()
        t.join()
    assert seen == [False]


def test_the_perimeter_and_the_randomization_stop_when_cancelled():
    from tools import mps_cancel
    from tools.mps_geometry import reconstruct_perimeter
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 2 * np.pi, 60)
    pts = np.column_stack([1000 * np.cos(a), 700 * np.sin(a)]) + rng.normal(0, 20, (60, 2))
    plain = reconstruct_perimeter(pts, refine=True, all_starts=True)
    with mps_cancel.cancellable(lambda: False):
        same = reconstruct_perimeter(pts, refine=True, all_starts=True)
    assert same.perimeter_um == plain.perimeter_um
    with mps_cancel.cancellable(lambda: True), pytest.raises(mps_cancel.AnalysisCancelled):
        reconstruct_perimeter(pts, refine=True, all_starts=True)
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring()
    calls = {"n": 0}

    def after_some() -> bool:
        calls["n"] += 1
        return calls["n"] > 4           # past the clustering, inside the steps

    with mps_cancel.cancellable(after_some), pytest.raises(mps_cancel.AnalysisCancelled):
        analyze_axon(x, y, z, run_randomization=True, n_randomizations=50)


def test_the_emitters_layer_stops_when_cancelled():
    from tools import mps_cancel
    from tools.cluster_emitters import XiPrior, fit_emitters
    rng = np.random.default_rng(2)
    xy = np.concatenate([rng.normal(c, 10, (15, 2)) for c in rng.uniform(0, 3000, (12, 2))])
    sig = np.full((xy.shape[0], 2), 8.0)
    with mps_cancel.cancellable(lambda: True), pytest.raises(mps_cancel.AnalysisCancelled):
        fit_emitters(xy, sig, XiPrior.poisson(14.0), n_burnin=50, n_trials=50, processes=1)


_EMITTERS_TWO_PROCESSES = textwrap.dedent(r"""
    import multiprocessing as mp, sys, threading, time
    import numpy as np

    def main():
        sys.path.insert(0, sys.argv[1])
        from tools import mps_cancel
        from tools.cluster_emitters import XiPrior, fit_emitters
        rng = np.random.default_rng(2)
        xy = np.concatenate([rng.normal(c, 10, (15, 2)) for c in rng.uniform(0, 20000, (40, 2))])
        sig = np.full((xy.shape[0], 2), 8.0)
        flag = {"on": False, "at": None}
        out = {}

        def run():
            try:
                with mps_cancel.cancellable(lambda: flag["on"]):
                    fit_emitters(xy, sig, XiPrior.poisson(14.0), n_burnin=200000, n_trials=200000, processes=2)
                out["result"] = "finished"
            except mps_cancel.AnalysisCancelled:
                out["result"] = "cancelled"
                out["after"] = time.monotonic() - flag["at"]

        t = threading.Thread(target=run)
        t.start()
        time.sleep(4.0)                       # the two worker processes are up and inside their chains
        assert len(mp.active_children()) == 2, mp.active_children()
        flag["at"] = time.monotonic()
        flag["on"] = True
        t.join(60)
        assert out.get("result") == "cancelled", out
        assert not mp.active_children(), mp.active_children()
        print("ABORTED", round(out["after"], 2))

    if __name__ == "__main__":
        main()
""")


def test_the_emitters_worker_processes_stop_at_once_when_cancelled(tmp_path):
    """Two processes (the registry default): the thread that waits for them sees the cancel within 0.5 s and
    terminates them, instead of waiting for the end of their chains."""
    script = tmp_path / "emitters_two_processes.py"
    script.write_text(_EMITTERS_TWO_PROCESSES, encoding="utf-8")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, str(script), HERE], capture_output=True, text=True, timeout=180, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("ABORTED"), out.stdout + out.stderr
    assert float(last.split()[1]) < 6.0, last


# ----------------------------------------------------------------- B: the window's background run
def _slow_run(analysis, result, *, seconds=0.6, cooperative=True, calls=None, gates=()):
    """A PreparedRun whose computation sleeps (checking the cancel flag when cooperative)."""
    from tools import mps_cancel
    from tools.mps_background_run import PreparedRun
    calls = calls if calls is not None else {}

    def compute():
        calls["thread"] = threading.get_ident()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if cooperative:
                mps_cancel.check("test")
            time.sleep(0.01)
        return result

    def finish(a):
        calls["finish"] = threading.get_ident()
        return a

    def fail(exc):
        calls["fail"] = exc

    return PreparedRun(compute=compute, finish=finish, fail=fail, gates=gates, words="OPTICS on a test ring")


def _background_window(analysis, prepared_runs, kw_seen=None):
    def background(**kw):
        if kw_seen is not None:
            kw_seen.append(kw)
        return prepared_runs.pop(0)
    w = _window(analysis, background=background)
    w.resize(1366, 760)
    w.show()
    _pump(0.1)
    return w


def test_a_slow_method_runs_in_a_worker_and_the_window_keeps_responding(analysis, quiet_boxes):
    from tools import mps_background_run as bg
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring(seed=5)
    new = analyze_axon(x, y, z, run_randomization=False)
    calls: dict = {}
    kw_seen: list = []
    w = _background_window(analysis, [_slow_run(analysis, new, calls=calls)], kw_seen)
    w.method_panel.set_state("OPTICS", {}, True, [])
    gui = threading.get_ident()
    ticks = []
    from PyQt5 import QtCore
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start(20)
    w._on_param_changed()                                   # what choosing OPTICS does
    assert w.busy() and w.run_progress.isVisible()
    assert "Running OPTICS on a test ring" in w.run_progress.label.text()
    assert not w.strip_scroll.widget().isEnabled() and not w.banner.isEnabled()
    assert not w.btn_export.isEnabled()
    assert kw_seen and kw_seen[0]["cluster_method"] == "OPTICS"
    assert _pump_until(lambda: not w.busy(), 10.0)
    timer.stop()
    # the event loop kept turning while the analysis ran (a frozen window would leave no ticks)
    assert len(ticks) >= 10, len(ticks)
    assert calls["thread"] != gui and calls["finish"] == gui
    assert w.analysis is new and not w.run_progress.isVisible()
    assert w.strip_scroll.widget().isEnabled() and w.banner.isEnabled()
    assert w.btn_export.isEnabled() == (w.export_callback is not None), "each control gets its own state back"
    assert not quiet_boxes
    assert _pump_until(lambda: bg.live_runs() == 0, 5.0)
    w.close()


def test_dbscan_stays_synchronous(analysis):
    seen = []
    w = _window(analysis, background=lambda **kw: seen.append(kw))
    calls = []
    w.rerun_callback = lambda **kw: (calls.append(kw), analysis)[1]
    w._on_param_changed()
    assert calls and not seen and not w.busy()
    w.close()


def test_cancel_stops_a_cooperative_run_and_keeps_the_analysis(analysis, quiet_boxes):
    from tools import mps_background_run as bg
    from tools import mps_cancel
    calls: dict = {}
    w = _background_window(analysis, [_slow_run(analysis, object(), seconds=20.0, calls=calls)])
    w.method_panel.set_state("OPTICS", {}, True, [])
    w._on_param_changed()
    _pump(0.2)
    assert w.busy()
    t0 = time.monotonic()
    w.run_progress.btn_cancel.click()
    assert "Cancelling" in w.run_progress.label.text() and not w.run_progress.btn_cancel.isEnabled()
    assert _pump_until(lambda: not w.busy(), 5.0)
    assert time.monotonic() - t0 < 3.0
    assert w.analysis is analysis, "the analysis shown is kept"
    assert isinstance(calls.get("fail"), mps_cancel.AnalysisCancelled)
    assert "Cancelled" in w.run_progress.label.text() and "previous one" in w.run_progress.label.text()
    assert w.strip_scroll.widget().isEnabled()
    assert not quiet_boxes, "a cancel is not an error"
    assert _pump_until(lambda: bg.live_runs() == 0, 5.0)
    w.close()


def test_a_result_that_arrives_after_cancel_is_discarded(analysis, quiet_boxes):
    from tools import mps_background_run as bg
    calls: dict = {}
    w = _background_window(analysis, [_slow_run(analysis, object(), seconds=0.8, cooperative=False, calls=calls)])
    w.method_panel.set_state("OPTICS", {}, True, [])
    w._on_param_changed()
    _pump(0.1)
    w.cancel_background()
    assert _pump_until(lambda: not w.busy(), 5.0)
    assert w.analysis is analysis and "finish" not in calls and "fail" in calls
    assert _pump_until(lambda: bg.live_runs() == 0, 5.0)
    w.close()


def test_an_error_in_the_worker_is_reported_and_the_window_is_usable(analysis, quiet_boxes):
    from tools.mps_background_run import PreparedRun

    def boom():
        raise ValueError("synthetic failure")
    seen: dict = {}
    w = _background_window(analysis, [PreparedRun(compute=boom, finish=lambda a: a,
                                                  fail=lambda e: seen.setdefault("fail", e))])
    w.method_panel.set_state("OPTICS", {}, True, [])
    w._on_param_changed()
    assert _pump_until(lambda: not w.busy(), 5.0)
    assert isinstance(seen["fail"], ValueError)
    assert quiet_boxes and quiet_boxes[-1][0] == "critical" and "synthetic failure" in quiet_boxes[-1][2]
    assert w.analysis is analysis and w.strip_scroll.widget().isEnabled()
    w.close()


def test_a_question_in_the_middle_is_asked_in_the_gui_thread(analysis, quiet_boxes):
    """The perimeter's gate, installed in the worker as a relay: its dialog would run in the GUI thread, and the
    worker gets the answer (here: one start)."""
    from tools import mps_perimeter_cost as cost
    from tools.mps_background_run import PreparedRun
    asked = {}

    def gate(n):
        asked["thread"] = threading.get_ident()
        asked["n"] = n
        return cost.ONE_START

    def compute():
        asked["worker"] = threading.get_ident()
        return cost.decide(5000)

    w = _background_window(analysis, [PreparedRun(compute=compute, finish=lambda a: (asked.setdefault("answer", a),
                                                                                     analysis)[1],
                                                  gates=((cost.asking, gate),))])
    w.method_panel.set_state("OPTICS", {}, True, [])
    w._on_param_changed()
    assert _pump_until(lambda: not w.busy(), 5.0)
    assert asked["n"] == 5000 and asked["answer"] == cost.ONE_START
    assert asked["thread"] == threading.get_ident() != asked["worker"]
    w.close()


def test_the_perimeter_cancel_at_the_question_is_not_an_error(analysis, quiet_boxes):
    from tools import mps_perimeter_cost as cost
    from tools.mps_background_run import PreparedRun
    def compute():
        # what analyze_axon does with the answer (tools/mps_analysis.py, before the perimeter)
        if cost.decide(5000) == cost.CANCEL:
            raise cost.PerimeterCancelled(5000, cost.estimate_perimeter_seconds(5000))
        return analysis

    w = _background_window(analysis, [PreparedRun(compute=compute, finish=lambda a: a,
                                                  gates=((cost.asking, lambda n: cost.CANCEL),))])
    w.method_panel.set_state("OPTICS", {}, True, [])
    w._on_param_changed()
    assert _pump_until(lambda: not w.busy(), 5.0)
    assert not quiet_boxes and w.analysis is analysis
    assert "previous one" in w.run_progress.label.text()
    w.close()


def test_nothing_starts_when_preparing_refuses(analysis, quiet_boxes):
    w = _background_window(analysis, [None])
    w.method_panel.set_state("OPTICS", {}, True, [])
    w._on_param_changed()
    assert not w.busy() and w.analysis is analysis
    w.close()


# A crash of Qt (a QThread destroyed while it runs) kills the process: the close-while-running scenario runs in a
# child process and the test reads its exit code and its last line (as test_phase_d_ui does for A1).
_CLOSE_WHILE_RUNNING = textwrap.dedent(r"""
    import gc, os, sys, time, threading
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, sys.argv[1])
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication([])
    from tools import mps_background_run as bg, mps_cancel
    from tools.mps_background_run import PreparedRun
    from tools.mps_results_window import MPSResultsWindow, WindowLinks

    def pump(t):
        end = time.monotonic() + t
        while time.monotonic() < end:
            app.processEvents(); time.sleep(0.005)

    def compute_cooperative():
        while True:
            mps_cancel.check("test"); time.sleep(0.01)

    def compute_stubborn():
        time.sleep(1.0)               # no check: finishes its step after the close
        return "late"

    for compute in (compute_cooperative, compute_stubborn):
        runs = [PreparedRun(compute=compute, finish=lambda a: a)]
        w = MPSResultsWindow(None, rerun_callback=lambda **k: None, links=WindowLinks(),
                             background=lambda **k: runs.pop(0))
        w.show(); pump(0.05)
        assert w.start_background({"cluster_method": "OPTICS"})
        pump(0.2)
        w.close()                     # cancels; the window only hides
        assert not w.isVisible()
        w.deleteLater(); del w; gc.collect()
        pump(0.05)
        end = time.monotonic() + 10
        while bg.live_runs() and time.monotonic() < end:
            pump(0.05)
        assert bg.live_runs() == 0, bg.live_runs()
    # shutdown with a run still going: it is cancelled and waited for
    runs = [PreparedRun(compute=compute_cooperative, finish=lambda a: a)]
    w = MPSResultsWindow(None, rerun_callback=lambda **k: None, links=WindowLinks(),
                         background=lambda **k: runs.pop(0))
    assert w.start_background({"cluster_method": "OPTICS"})
    pump(0.1)
    assert bg.shutdown(5.0) == 0
    pump(0.1)
    print("CLEAN", threading.active_count())
""")


def test_closing_the_window_while_running_leaves_no_thread(tmp_path):
    script = tmp_path / "close_while_running.py"
    script.write_text(_CLOSE_WHILE_RUNNING, encoding="utf-8")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, str(script), HERE], capture_output=True, text=True, timeout=120, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("CLEAN"), out.stdout + out.stderr
    assert int(last.split()[1]) == 1, "only the main thread is left"
