"""The warning before a long perimeter (tools/mps_perimeter_cost.py, tools/perimeter_cost_ui.py).

The estimate is a pure function fitted to this project's synthetic timings; the analysis asks the gate installed by
``asking`` only when a warning is due, and nothing non-interactive (no gate, offscreen, a worker thread, the batch)
ever shows a dialog. Synthetic data only (R8); offscreen Qt.

Run: py -3 -m pytest test_perimeter_cost.py -q
"""
from __future__ import annotations

import dataclasses
import json
import os
import threading

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import mps_perimeter_cost as pc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TIMINGS = os.path.join(HERE, "docs", "sims", "perimeter", "perimeter_time.jsonl")


# ----------------------------------------------------------------- the estimate
def test_estimate_is_zero_without_a_contour_and_grows_as_the_fit():
    assert pc.estimate_perimeter_seconds(0) == 0.0 and pc.estimate_perimeter_seconds(2) == 0.0
    assert pc.estimate_one_start_seconds(2) == 0.0
    sizes = [3, 10, 100, 300, 600, 1000, 2000]
    every = [pc.estimate_perimeter_seconds(n) for n in sizes]
    one = [pc.estimate_one_start_seconds(n) for n in sizes]
    assert all(b > a for a, b in zip(every, every[1:]))
    assert all(b > a for a, b in zip(one, one[1:]))
    assert pc.estimate_perimeter_seconds(1000) / pc.estimate_perimeter_seconds(100) == pytest.approx(
        10.0 ** pc.EVERY_START_B)
    # One start is the cheaper option wherever a warning is due.
    for n in (pc.PERIMETER_WARN_CLUSTERS, 1000, 2000):
        assert pc.estimate_one_start_seconds(n) < pc.estimate_perimeter_seconds(n)


def test_estimate_follows_the_committed_timings():
    """The constants are the fit of docs/sims/perimeter/perimeter_time.jsonl (slowest timing at each n): within a
    factor 2 of it at every measured size from 300 clusters, where a warning can be due."""
    rows = [json.loads(line) for line in open(TIMINGS, encoding="utf-8") if line.strip()]
    for mode, estimate in (("every", pc.estimate_perimeter_seconds), ("one", pc.estimate_one_start_seconds)):
        slowest = {}
        for r in rows:
            if r["mode"] == mode:
                slowest[r["n"]] = max(slowest.get(r["n"], 0.0), r["seconds"])
        assert {300, 600, 1000} <= set(slowest), mode
        for n, seconds in slowest.items():
            if n >= 300:
                assert 0.5 <= seconds / estimate(n) <= 2.0, (mode, n, seconds, estimate(n))


def test_threshold_is_where_the_estimate_reaches_the_seconds():
    n = pc.PERIMETER_WARN_CLUSTERS
    assert pc.estimate_perimeter_seconds(n - 1) < pc.PERIMETER_WARN_SECONDS <= pc.estimate_perimeter_seconds(n)
    assert not pc.needs_warning(n - 1) and pc.needs_warning(n)
    assert pc.clusters_for_seconds(pc.PERIMETER_WARN_SECONDS) == n


def test_minutes_in_words():
    assert pc.format_minutes(10) == "under 1 min"
    assert pc.format_minutes(60) == "about 1 min"
    assert pc.format_minutes(11 * 60 + 5) == "about 11 min"
    text = pc.note(1000)
    assert text.startswith("perimeter: 1,000 clusters, estimated about ") and "from every start" in text
    assert "one start" in pc.note(1000, pc.ONE_START)


# ----------------------------------------------------------------- the gate
def test_decide_asks_only_when_a_warning_is_due_and_a_gate_is_installed():
    asked = []

    def gate(n):
        asked.append(n)
        return pc.ONE_START

    big = pc.PERIMETER_WARN_CLUSTERS
    assert pc.decide(big) == pc.EVERY_START                       # no gate: never asks
    with pc.asking(gate):
        assert pc.decide(big - 1) == pc.EVERY_START               # below the threshold: not asked
        assert asked == []
        assert pc.decide(big) == pc.ONE_START
        assert asked == [big]
    assert pc.current_gate() is None                              # the block took it away
    with pc.asking(lambda n: "maybe"):
        with pytest.raises(ValueError):
            pc.decide(big)


def test_a_worker_thread_never_inherits_the_gate():
    seen = {}

    def gate(n):
        raise AssertionError("asked from a worker thread")

    def work():
        seen["gate"] = pc.current_gate()
        seen["answer"] = pc.decide(pc.PERIMETER_WARN_CLUSTERS + 10)

    with pc.asking(gate):
        t = threading.Thread(target=work)
        t.start()
        t.join()
    assert seen == {"gate": None, "answer": pc.EVERY_START}


# ----------------------------------------------------------------- the analysis
@pytest.fixture(scope="module")
def sim():
    from tools.mps_simulate_axon import SimConfig, simulate_axon
    return simulate_axon(SimConfig(), 5)


def _analyse(sim, **kw):
    from tools.mps_analysis import analyze_axon
    return analyze_axon(sim.x_nm, sim.y_nm, sim.z_nm, source_name="sim.hdf5", pixel_size_nm=113.0,
                        pixel_size_source="yaml", run_randomization=False, **kw)


def _notes(analysis):
    return [w for w in analysis.warnings if w.startswith("perimeter: ")]


def test_default_threshold_leaves_a_small_axon_alone(sim):
    def gate(n):
        raise AssertionError("asked for a small axon")
    with pc.asking(gate):
        a = _analyse(sim)
    assert a.n_clusters_kept < pc.PERIMETER_WARN_CLUSTERS
    assert _notes(a) == [] and a.contour_2opt == "every start"


def test_without_a_gate_the_analysis_runs_from_every_start_and_notes_it(sim, monkeypatch):
    monkeypatch.setattr(pc, "PERIMETER_WARN_CLUSTERS", 3)
    a = _analyse(sim)
    assert a.contour_2opt == "every start"
    assert _notes(a) == [pc.note(a.n_clusters_kept)]
    assert not any(w.startswith("perimeter: ") for w in a.upstream_warnings)


def test_the_gate_chooses_one_start_or_cancels(sim, monkeypatch):
    monkeypatch.setattr(pc, "PERIMETER_WARN_CLUSTERS", 3)
    asked = []

    def one(n):
        asked.append(n)
        return pc.ONE_START

    with pc.asking(one):
        a = _analyse(sim)
    assert asked == [a.n_clusters_kept]
    assert a.contour_2opt == "one start" and a.perimeter.n_starts == 1
    assert _notes(a) == [pc.note(a.n_clusters_kept, pc.ONE_START)]

    # A later re-run from every start (the axoplasm panel's) is not what the note describes: it is not carried.
    from tools.mps_analysis import with_every_start
    every = with_every_start(a)
    assert every.contour_2opt == "every start" and _notes(every) == []

    with pc.asking(lambda n: pc.CANCEL):
        with pytest.raises(pc.PerimeterCancelled) as info:
            _analyse(sim)
    assert info.value.n_clusters == a.n_clusters_kept


def test_no_question_when_no_2opt_from_every_start_would_run(sim, monkeypatch):
    monkeypatch.setattr(pc, "PERIMETER_WARN_CLUSTERS", 3)

    def gate(n):
        raise AssertionError("asked although no 2-opt from every start runs")

    with pc.asking(gate):
        one = _analyse(sim, all_starts=False)
        assert _notes(one) == [] and one.contour_2opt == "one start"
        c = np.asarray(one.centroids, float)
        mx, my = c.mean(axis=0)
        r = float(np.mean(np.hypot(c[:, 0] - mx, c[:, 1] - my)))
        t = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
        guided = _analyse(sim, contour_guide=np.column_stack([mx + r * np.cos(t), my + r * np.sin(t)]))
        assert _notes(guided) == [] and guided.contour_2opt is None


# ----------------------------------------------------------------- the batch: never asks, writes the note
def test_batch_writes_the_note_and_never_asks(tmp_path, monkeypatch):
    import golden_ui_stage2 as gold
    from tools import mps_batch as mb
    from tools import perimeter_cost_ui as ui
    from tools.mps_identity import AxonIdentity
    from tools.mps_settings import MPSSettings

    gold.make_inputs(str(tmp_path))
    monkeypatch.setattr(pc, "PERIMETER_WARN_CLUSTERS", 3)

    def no_dialog(*a, **k):
        raise AssertionError("the batch showed the perimeter dialog")
    monkeypatch.setattr(ui, "ask", no_dialog)
    settings = dataclasses.replace(mb.BatchSettings.from_settings(MPSSettings()), n_randomizations=19)
    item = mb.BatchFile(path=str(tmp_path / "sim_axon_A.hdf5"), identity=AxonIdentity())
    record = mb.analyse_file(item, settings)
    assert record.error in ("", None), record.error
    assert "perimeter: " in str(record.row["warnings"]) and "from every start" in str(record.row["warnings"])
    assert record.row["contour_2opt"] == "every start"


# ----------------------------------------------------------------- the dialog (offscreen)
_APP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def test_gate_shows_nothing_offscreen(monkeypatch):
    _app()
    from tools import perimeter_cost_ui as ui
    assert not ui.interactive()

    def no_dialog(*a, **k):
        raise AssertionError("dialog shown offscreen")
    monkeypatch.setattr(ui, "ask", no_dialog)
    assert ui.make_gate(None)(5000) == pc.EVERY_START


def test_gate_asks_when_interactive_and_puts_the_cursors_back(monkeypatch):
    _app()
    from PyQt5 import QtCore, QtWidgets

    from tools import perimeter_cost_ui as ui
    monkeypatch.setattr(ui, "interactive", lambda: True)
    seen = {}

    def answer(parent, n):
        seen["n"] = n
        seen["cursor_during"] = QtWidgets.QApplication.overrideCursor()
        return pc.ONE_START
    monkeypatch.setattr(ui, "ask", answer)
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    try:
        assert ui.make_gate(None)(1234) == pc.ONE_START
        assert seen["n"] == 1234 and seen["cursor_during"] is None
        assert QtWidgets.QApplication.overrideCursor() is not None
        QtWidgets.QApplication.restoreOverrideCursor()
        assert QtWidgets.QApplication.overrideCursor() is not None   # both were put back
        QtWidgets.QApplication.restoreOverrideCursor()
    finally:
        while QtWidgets.QApplication.overrideCursor() is not None:
            QtWidgets.QApplication.restoreOverrideCursor()
    assert QtWidgets.QApplication.overrideCursor() is None


def test_dialog_buttons_name_their_answers(monkeypatch):
    _app()
    from PyQt5 import QtWidgets

    from tools import perimeter_cost_ui as ui
    box = ui.build_dialog(None, 1000)
    text = box.text()
    assert "1,000 clusters" in text and "every start" in text and "one start" in text.lower()
    assert pc.format_minutes(pc.estimate_perimeter_seconds(1000)) in text
    answers = {b.text(): b.property("perimeter_answer") for b in box.buttons()}
    assert answers == {"Continue (every start)": pc.EVERY_START, "One start (faster)": pc.ONE_START,
                       "Cancel": pc.CANCEL}
    assert box.escapeButton().property("perimeter_answer") == pc.CANCEL
    box.close()

    # ask(): the clicked button's answer; closing the box without one is Cancel.
    for label, expected in (("One start (faster)", pc.ONE_START), (None, pc.CANCEL)):
        def fake_exec(self, _label=label):
            for b in self.buttons():
                if b.text() == _label:
                    b.click()
            return 0
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", fake_exec)
        assert ui.ask(None, 1000) == expected
