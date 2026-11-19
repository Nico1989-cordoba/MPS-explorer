"""Progress bars and size warnings (manual test of 2026-11-17): the per-axon null has its own share of the Batch bar
(before: ~99 % for the whole null), the bar says time spent and an estimate of what is left, a very large file or
selection is asked about before it costs anything, the background run's bar is a real bar, and the Batch's Cancel
reaches inside the axon being analysed.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_ui_progress_gui.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import mps_progress, mps_size_guard as guard  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_APP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


# ------------------------------------------------------------------------------------------ the null's share
def test_null_stage_has_its_own_share_and_the_bar_reaches_the_end_only_after_it():
    plain = mps_progress.file_fraction("randomization", 198, 199)[0]
    assert plain < 1.0
    with_null = mps_progress.file_fraction("randomization", 198, 199, with_null=True)[0]
    assert with_null <= mps_progress.NULL_ANALYSIS_SHARE        # the analysis is the first fifth of the file's bar
    first = mps_progress.file_fraction(mps_progress.NULL_STAGE, 0, 200, with_null=True)[0]
    mid = mps_progress.file_fraction(mps_progress.NULL_STAGE, 100, 200, with_null=True)[0]
    last = mps_progress.file_fraction(mps_progress.NULL_STAGE, 200, 200, with_null=True)[0]
    assert with_null <= first < mid < last == pytest.approx(1.0)
    share, words = mps_progress.file_fraction(mps_progress.NULL_STAGE, 4, 200, with_null=True)
    assert "replicate 5 of 200" in words and share is not None
    # a null nobody announced: words only, no share (as before), and the stages without a null are unchanged
    assert mps_progress.file_fraction(mps_progress.NULL_STAGE, 4, 200)[0] is None
    assert mps_progress.file_fraction("before the perimeter")[0] == pytest.approx(0.30)


def test_eta_words_and_elapsed():
    assert mps_progress.eta_words(0.01, 100) == ""                # too little done to say
    assert mps_progress.eta_words(0.5, 5) == ""                   # too little time to say
    assert mps_progress.eta_words(0.5, 600) == "about 10 min left"
    assert mps_progress.eta_words(0.25, 3600) == "about 3.0 h left"
    assert mps_progress.eta_words(0.5, 40) == "about 40 s left"
    assert mps_progress.eta_words(1.0, 600) == ""
    assert mps_progress.elapsed_words(75) == "01:15" and mps_progress.elapsed_words(3725) == "1:02:05"


def _batch_window():
    _app()
    from test_batch_perimeter_progress import _settings
    from tools.batch_window import BatchWindow
    return BatchWindow(_settings())


def test_batch_window_bar_moves_through_the_null_and_says_time():
    import threading
    w = _batch_window()
    w._set_bar(2, "batch")
    w._n_files, w._compare_names, w._with_null = 2, [], True
    gate = threading.Event()
    w._thread = threading.Thread(target=gate.wait, daemon=True)         # "running"
    w._thread.start()
    try:
        w._t_phase = time.monotonic() - 600.0
        w._on_stage(0, "randomization", 18, 19)
        after_analysis = w.progress.value()
        assert after_analysis <= 0.2 * bw_unit()                        # the analysis is the first fifth of a file
        w._on_stage(0, mps_progress.NULL_STAGE, 50, 200)
        mid = w.progress.value()
        assert mid > after_analysis and mid >= 0.39 * bw_unit()          # not stuck at ~99 % any more
        text = w.progress.format()
        assert "per-axon null" in text and "replicate 51 of 200" in text and "elapsed" in text
        assert "left" in text                                           # an estimate of the time remaining
        w._on_stage(0, mps_progress.NULL_STAGE, 200, 200)
        assert w.progress.value() < 2 * bw_unit() and w.progress.value() >= 0.99 * bw_unit()
    finally:
        gate.set()
        w._thread.join()
        w._thread = None
        w.close()


def bw_unit():
    from tools.batch_window import UNIT_STEPS
    return UNIT_STEPS


def test_batch_window_without_a_null_keeps_the_old_shares():
    import threading
    w = _batch_window()
    w._set_bar(2, "batch")
    w._n_files, w._compare_names, w._with_null = 2, [], False
    w._on_stage(0, "randomization", 18, 19)
    assert w.progress.value() > 0.9 * bw_unit() * 0.95                  # as before: the randomization ends the file
    w.close()


def test_cancel_reaches_inside_the_axon_being_analysed(tmp_path):
    """Before, the batch heard Cancel only between two files; now at the next check point of the analysis."""
    from test_batch_perimeter_progress import _write
    _write(tmp_path, "synth_axon1.csv", 1, 30)
    _write(tmp_path, "synth_axon2.csv", 2, 12)
    w = _batch_window()
    w.find_files(str(tmp_path))
    for row in range(len(w.plan.files)):
        w.set_ticked(row, True)
    seen = []

    def on_stage(unit, where, done, total):
        seen.append(where)
        if where == "before the perimeter":
            w._cancel.set()                     # from the worker's own thread: no race with the GUI queue

    from PyQt5 import QtCore
    w._relay.stage.connect(on_stage, QtCore.Qt.DirectConnection)
    w.run(ask=lambda *a, **k: None)
    w.wait(120)
    assert not w.running
    assert "before the randomization" not in seen                       # stopped inside the first file
    assert w.batch_records() == []                                      # nothing half-analysed was kept
    w.close()


# ------------------------------------------------------------------------------------------ the size rule
def test_thresholds_follow_the_size_of_a_cropped_axon():
    assert guard.LOAD_ASK_LOCS == 10 * guard.AXON_LOCS_TYPICAL_MAX
    assert guard.AXON_LOCS_TYPICAL_MAX < guard.ANALYSIS_BACKGROUND_LOCS < guard.ANALYSIS_ASK_LOCS < guard.LOAD_ASK_LOCS
    assert not guard.load_needs_question(15_000) and not guard.load_needs_question(None)
    assert guard.load_needs_question(guard.LOAD_ASK_LOCS + 1)
    assert not guard.analysis_needs_question(15_000) and guard.analysis_needs_question(60_000)
    assert not guard.analysis_in_background(15_000) and guard.analysis_in_background(20_000)


def test_peek_count_reads_the_header_not_the_table(tmp_path):
    import h5py
    path = str(tmp_path / "locs.hdf5")
    table = np.zeros(1234, dtype=[("frame", "<u4"), ("x", "<f4"), ("y", "<f4")])
    with h5py.File(path, "w") as fh:
        fh.create_dataset("locs", data=table)
    n, size = guard.peek_count(path)
    assert n == 1234 and size > 0
    csv = tmp_path / "locs.csv"
    csv.write_text("id,frame,x [nm],y [nm],z [nm]\n" + "".join(f"{i},{i},1.0,2.0,3.0\n" for i in range(5000)))
    n, _ = guard.peek_count(str(csv))
    assert n is not None and abs(n - 5000) < 100
    assert guard.peek_count(str(tmp_path / "missing.hdf5")) == (None, 0)
    other = tmp_path / "x.hdf5"
    other.write_bytes(b"not hdf5")
    assert guard.peek_count(str(other))[0] is None                 # a peek never raises


def test_texts_say_the_numbers_and_the_way_out():
    t = guard.load_question_text("big.hdf5", 4_000_000, 4_000_000_000)
    assert "4,000,000" in t and "4.0 GB" in t and "10.0 GB" in t and "Cancel" in t
    a = guard.analysis_question_text(60_000, "OPTICS", "30 min")
    assert "60,000" in a and "30 min" in a and "Cancel" in a


def test_confirm_hook_and_offscreen_default():
    _app()
    from tools import size_guard_ui as sg
    sg.ASKED.clear()
    assert sg.confirm(None, "t", "question") is True               # offscreen: nobody to ask, go on as before
    sg.ANSWER = lambda text: False
    try:
        assert sg.confirm(None, "t", "question 2") is False
    finally:
        sg.ANSWER = None
    assert sg.ASKED[-2:] == ["question", "question 2"]


def test_read_with_progress_offscreen_is_a_plain_call():
    _app()
    from tools import size_guard_ui as sg
    assert sg.read_with_progress(None, lambda: 7) == 7
    with pytest.raises(ZeroDivisionError):
        sg.read_with_progress(None, lambda: 1 / 0)


def test_read_dialog_runs_in_a_worker_and_cancel_discards(monkeypatch):
    _app()
    from PyQt5 import QtCore
    from tools import size_guard_ui as sg
    monkeypatch.setattr(sg, "interactive", lambda: True)           # pretend a screen: the dialog is built offscreen
    assert sg.read_with_progress(None, lambda: (time.sleep(0.3), "read")[1]) == "read"
    gate = []

    def slow():
        time.sleep(0.6)
        gate.append("ended")
        return "late"

    QtCore.QTimer.singleShot(150, lambda: _app().activeModalWidget().btn_cancel.click())
    with pytest.raises(sg.ReadCancelled):
        sg.read_with_progress(None, slow)
    deadline = time.monotonic() + 3
    while not gate and time.monotonic() < deadline:
        _app().processEvents()
        time.sleep(0.02)
    assert gate == ["ended"]                                       # the read ended by itself; nothing used it


# ------------------------------------------------------------------------------------ the background run's bar
def test_run_progress_is_a_bar_once_the_run_says_where_it_is():
    _app()
    from tools.mps_background_run import RunProgress
    rp = RunProgress()
    assert rp.bar.maximum() == 0                                   # busy until the first check point
    rp.set_fraction(0.42)
    assert rp.bar.maximum() == 1000 and rp.bar.value() == 420 and rp.bar.isTextVisible()
    rp.set_fraction(-1.0)
    assert rp.bar.maximum() == 0


def test_background_run_fraction_moves_by_the_clustering_estimate():
    _app()
    from tools.mps_background_run import BackgroundRun, PreparedRun
    run = BackgroundRun(PreparedRun(compute=lambda: None, finish=lambda r: r, clustering_estimate_s=100.0))

    class W:
        point = ("before the clustering", None, None)
        point_t = time.monotonic() - 50.0
        stage = ""
        cancelled = False

    run.worker = W()
    f = run.fraction()
    lo, hi = mps_progress.STAGES["before the clustering"][1:]
    assert lo + 0.4 * (hi - lo) < f < lo + 0.6 * (hi - lo)         # half of the estimate: half of the stage
    W.point_t = time.monotonic() - 10_000.0
    assert run.fraction() == pytest.approx(lo + 0.95 * (hi - lo))  # a lower bound: never the whole stage
    W.point = ("randomization", 50, 100)
    assert run.fraction() == pytest.approx(0.75 + 0.25 * 0.5)
    W.point = ("", None, None)
    assert run.fraction() == -1.0
    run.worker = None


# ------------------------------------------------------ the main window: the questions before reading / analysing
_SCRIPT = textwrap.dedent(r"""
    import functools, os, sys
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    HERE, WORK = sys.argv[1], sys.argv[2]
    sys.path.insert(0, HERE)
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication([])
    import golden_ui_stage2 as gold
    inputs = os.path.join(WORK, "inputs")
    described = gold.make_inputs(inputs)
    os.chdir(WORK)
    from tools import mps_settings
    settings_dir = os.path.join(WORK, "settings"); os.makedirs(settings_dir, exist_ok=True)
    original = mps_settings.settings_path
    mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)
    import MPS_explorer
    MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
    MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
    from tools import mps_size_guard as guard, size_guard_ui as sg
    mw = MPS_explorer.MPS_explorer()
    path = os.path.join(inputs, "sim_axon_A.hdf5")
    n, _ = guard.peek_count(path)
    assert n and n > 1000, n
    # 1. a file over the (lowered) load threshold: declined -> nothing is loaded, no error box
    guard.LOAD_ASK_LOCS = n - 1
    sg.ANSWER = lambda text: False
    assert mw.load_channel1(path, 0) is False and mw.xdata is None
    assert len(sg.ASKED) == 1 and f"{n:,}" in sg.ASKED[0]
    # 2. accepted: loaded as before
    sg.ANSWER = lambda text: True
    assert mw.load_channel1(path, 0) is True and mw.xdata is not None and len(mw.xdata) == n
    # 3. a file under the threshold is not asked about
    sg.ASKED.clear(); guard.LOAD_ASK_LOCS = n + 1
    assert mw.load_channel1(path, 0) is True and sg.ASKED == []
    # 4. the analysis of a selection over the (lowered) threshold: asked once per selection
    mw.scatterplot(); mw.update_ROI()
    guard.ANALYSIS_ASK_LOCS = 10
    sg.ASKED.clear(); sg.ANSWER = lambda text: False
    assert mw.run_mps_analysis(show_window=False) is None and len(sg.ASKED) == 1
    assert "not run" in mw._perimeter_cancelled
    sg.ANSWER = lambda text: True
    assert mw.run_mps_analysis(show_window=False) is not None and len(sg.ASKED) == 2
    mw.run_mps_analysis(show_window=False)
    assert len(sg.ASKED) == 2, "the same selection is not asked again"
    # 5. big selections go to the background, whatever the method
    guard.ANALYSIS_BACKGROUND_LOCS = 10
    mw.mps_window = mw._make_mps_window(mw.mps_analysis)
    assert mw._selection_is_big() and mw.mps_window.big_run()
    print("OK")
""")


def test_main_window_asks_before_reading_and_before_analysing(tmp_path):
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, HERE, str(tmp_path)], capture_output=True, text=True,
                          timeout=540)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout[-2000:] + proc.stderr[-3000:]
