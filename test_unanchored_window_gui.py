"""The unanchored clusters window (P15): preview, decisions by hand, Apply, decisions file (offscreen Qt).

Simulated axons only (a ring of clusters with two lumps planted inside, as test_unanchored.py builds them); the
randomization is off and the rule's noise check short, so every run takes seconds.

Run (one file per process): QT_QPA_PLATFORM=offscreen py -3 -m pytest test_unanchored_window_gui.py -q
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import mps_unanchored as mu  # noqa: E402
from tools import mps_unanchored_window as muw  # noqa: E402
from test_unanchored import circle, membrane_on, with_lumps  # noqa: E402

REPS = 3                           # [ad hoc] a short noise check: the window, not the statistics


@pytest.fixture(autouse=True)
def _no_run_outlives_its_test():
    """A test that closes its window while the background run goes (apply, cancel) leaves the worker in the
    process-wide count of tools.mps_background_run.live_runs() until its release event is processed. The next
    test files in the same pytest process assert that count is 0 (test_f5d_background) and read this one's worker
    as theirs. Wait here, where the cause is, until the count is back to what it was."""
    from tools import mps_background_run as bg
    before = bg.live_runs()
    yield
    end = time.monotonic() + 30
    while bg.live_runs() > before and time.monotonic() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def analysis():
    from tools.mps_analysis import analyze_axon
    rng = np.random.default_rng(11)
    x, y, cents = membrane_on(circle(1500), rng, locs_per_cluster=40, cluster_sd_nm=12.0, diffuse=0)
    x, y, _ = with_lumps(x, y, cents, [(0.0, 0.0), (300.0, -200.0)], rng)
    z = rng.normal(0.0, 30.0, x.size)
    return analyze_axon(x, y, z, source_name="sim_lumps", pixel_size_nm=130.0, pixel_size_source="override",
                        run_randomization=False)


def _window(analysis, data_path, reasons=None):
    asked = [] if reasons is None else reasons
    inputs = muw.UnanchoredInputs(analysis=analysis, data_path=str(data_path))
    w = muw.UnanchoredWindow(inputs=inputs, noise_replicates=REPS,
                             ask_reason=lambda action, label: (asked.append((action, label)) or "glia"))
    return w, asked


def _run(w, **kw):
    assert w.analysis_changed(**kw)
    assert w.wait_run(120)


def _deepest(w):
    return int(np.nanargmax(w.result.depth_nm))


def test_opens_with_the_decided_defaults_and_one_save_button(app, analysis, tmp_path):
    w, _ = _window(analysis, tmp_path / "axon1.hdf5")
    try:
        assert w.method() == mu.METHOD_AUTOMATIC and w.reference() == mu.REFERENCE_POLAR
        assert w.spin_exclude.value() == 400.0 and w.spin_doubt.value() == 250.0
        texts = [b.text() for b in w.findChildren(QtWidgets.QPushButton)]
        assert texts.count("Save...") == 1 and not [t for t in texts if t.lower().startswith(("browse", "export"))]
        assert not w.btn_cancel.isEnabled() and w.progress.isHidden()
        for t in w.one_line_texts():
            assert "\n" not in t
        for b in (w.combo_method, w.combo_reference, w.spin_exclude, w.spin_doubt, w.btn_apply, w.btn_save):
            assert 0 < len(b.toolTip()) <= 100, b.toolTip()
    finally:
        w.close()


def test_preview_marks_the_lumps_and_changes_nothing_until_apply(app, analysis, tmp_path):
    w, _ = _window(analysis, tmp_path / "axon1.hdf5")
    seen = []
    w.applied.connect(lambda *a: seen.append(a))
    try:
        _run(w)
        assert w.result is not None and w.result.evaluated, w.result.failure
        assert w.flags.n_excluded == 2
        assert w.line_preview.full().startswith("Would exclude 2 of")
        names = [it.name() for it in w._click_items]
        assert any(n.startswith("excluded") and n.endswith("(2)") for n in names)
        assert any(n.startswith("doubtful") for n in names)
        assert seen == [] and w.comparison is None                       # nothing applied yet
        assert w.btn_apply.isEnabled()
    finally:
        w.close()


def test_apply_runs_in_the_background_and_hands_both_columns(app, analysis, tmp_path):
    w, _ = _window(analysis, tmp_path / "axon1.hdf5")
    seen = []
    w.applied.connect(lambda a, f, c: seen.append((a, f, c)))
    lines = []
    w.progress_changed.connect(lambda line, value, running: lines.append((value, running)))
    try:
        _run(w)
        assert w.apply()
        assert not w.progress.isHidden() and w.btn_cancel.isEnabled()
        assert w.wait_run(120)
        assert len(seen) == 1
        a, flags, cmp = seen[0]
        assert a is analysis and flags.n_excluded == 2
        assert cmp.discard_applied.n_clusters_kept == analysis.n_clusters_kept - 2
        assert cmp.discard_applied.discard_source.startswith("method automatic")
        assert lines and lines[-1][1] is False                           # the progress line was told it ended
        assert not w.apply()                                             # same flags: nothing to do again
    finally:
        w.close()


def test_auto_apply_is_what_the_main_window_uses(app, analysis, tmp_path):
    w, _ = _window(analysis, tmp_path / "axon1.hdf5")
    seen = []
    w.applied.connect(lambda *a: seen.append(a))
    try:
        _run(w, auto_apply=True)
        end = time.monotonic() + 60
        while not seen and time.monotonic() < end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
        assert seen and w.comparison is not None
    finally:
        w.close()


def test_click_excludes_with_a_reason_and_click_again_takes_it_back(app, analysis, tmp_path):
    w, asked = _window(analysis, tmp_path / "axon1.hdf5")
    try:
        _run(w)
        kept_row = int(np.flatnonzero(~w.flags.excluded)[0])
        assert w.toggle_cluster(kept_row)
        assert asked[-1][0] == mu.ACTION_EXCLUDE
        assert w.flags.excluded[kept_row] and w.flags.source[kept_row] == mu.SOURCE_MANUAL
        assert w.table.rowCount() == 1 and w.table.item(0, 5).text() == "exclude"
        assert w.table.item(0, 6).text() == "glia" and w.dirty
        assert w.toggle_cluster(kept_row)                                 # a decision there: taken back
        assert not w.flags.excluded[kept_row] and w.table.rowCount() == 0
        deep = _deepest(w)                                                # excluded by the rule
        assert w.toggle_cluster(deep)
        assert asked[-1][0] == mu.ACTION_RESTORE
        assert not w.flags.excluded[deep] and w.flags.source[deep] == mu.SOURCE_RESTORED
        w.undo()
        assert w.flags.excluded[deep]
    finally:
        w.close()


def test_a_click_on_the_map_reaches_the_cluster(app, analysis, tmp_path):
    w, asked = _window(analysis, tmp_path / "axon1.hdf5")
    try:
        _run(w)
        item = next(it for it in w._click_items if it.name().startswith("centres"))
        spot = item.points()[0]
        row = int(spot.data())
        w._on_points_clicked(item, [spot])
        assert asked and w.flags.excluded[row]
    finally:
        w.close()


def test_cancelled_reason_changes_nothing(app, analysis, tmp_path):
    inputs = muw.UnanchoredInputs(analysis=analysis, data_path=str(tmp_path / "axon1.hdf5"))
    w = muw.UnanchoredWindow(inputs=inputs, noise_replicates=REPS, ask_reason=lambda a, lab: None)
    try:
        _run(w)
        before = w.flags.excluded.copy()
        assert not w.toggle_cluster(int(np.flatnonzero(~before)[0]))
        assert np.array_equal(w.flags.excluded, before) and len(w.decisions) == 0
    finally:
        w.close()


def test_decisions_file_round_trip_beside_the_data_and_by_position(app, analysis, tmp_path):
    data = tmp_path / "axon1.hdf5"
    w, _ = _window(analysis, data)
    try:
        _run(w)
        row = int(np.flatnonzero(~w.flags.excluded)[0])
        w.toggle_cluster(row)
        default = w.default_decisions_path()
        assert default == mu.decisions_path(str(data))
        written = w.save_to(default)
        assert written == default and not w.dirty and "loads by itself" in w.line_file.full()
        stored = json.loads(Path(default).read_text(encoding="utf-8"))
        assert stored["format"] == mu.DECISIONS_FORMAT and stored["source"] == "axon1.hdf5"
        assert len(stored["decisions"]) == 1 and stored["decisions"][0]["reason"] == "glia"
    finally:
        w.close()
    # a new window on the same file reads it by itself, and the decision lands on the same cluster
    w2, _ = _window(analysis, data)
    try:
        _run(w2)
        assert len(w2.decisions) == 1 and w2.flags.excluded[row] and w2.flags.source[row] == mu.SOURCE_MANUAL
        assert "read from axon1_unanchored.json" in w2.line_file.full()
    finally:
        w2.close()


def test_save_under_another_name_is_respected_and_said(app, analysis, tmp_path, monkeypatch):
    data = tmp_path / "axon1.hdf5"
    w, _ = _window(analysis, data)
    try:
        _run(w)
        w.toggle_cluster(int(np.flatnonzero(~w.flags.excluded)[0]))
        chosen = str(tmp_path / "other_name.json")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (chosen, muw.SAVE_FILTER)))
        w.btn_save.click()
        assert os.path.exists(chosen) and not os.path.exists(mu.decisions_path(str(data)))
        assert "only axon1_unanchored.json loads by itself" in w.line_file.full()
        n = w.load_from(chosen)
        assert n == 1
    finally:
        w.close()


def test_a_foreign_file_at_the_decisions_path_is_never_overwritten(app, analysis, tmp_path):
    data = tmp_path / "axon1.hdf5"
    foreign = Path(mu.decisions_path(str(data)))
    foreign.write_text('{"something": "else"}', encoding="utf-8")
    w, _ = _window(analysis, data)
    try:
        _run(w)
        assert "not read" in w.line_file.full()
        with pytest.raises(mu.ManualDecisionsError):
            w.save_to(str(foreign))
        assert foreign.read_text(encoding="utf-8") == '{"something": "else"}'
    finally:
        w.close()


def test_reference_and_thresholds_rerun_the_rule(app, analysis, tmp_path):
    w, _ = _window(analysis, tmp_path / "axon1.hdf5")
    try:
        _run(w)
        n400 = w.flags.n_excluded
        w.spin_exclude.setValue(3000.0)
        w.spin_doubt.setValue(2900.0)
        w.recompute()
        assert w.wait_run(120)
        assert w.flags.n_excluded == 0 and n400 == 2
        w.combo_reference.setCurrentIndex(1)
        w.recompute()
        assert w.wait_run(120)
        assert w.result.reference == mu.REFERENCE_SMOOTH
        w.spin_doubt.setValue(3000.0)                                     # doubtful not below exclude: refused
        w.recompute()
        assert "must be below" in w.status.full()
    finally:
        w.close()


def test_off_method_keeps_only_the_decisions_and_widefield_needs_flags(app, analysis, tmp_path):
    w, _ = _window(analysis, tmp_path / "axon1.hdf5")
    try:
        _run(w)
        w.combo_method.setCurrentIndex(2)                                 # off
        assert w.wait_run(30)
        assert w.flags.method == mu.METHOD_OFF and w.flags.n_excluded == 0
        assert not w.combo_reference.isEnabled()
        model = w.combo_method.model()
        assert not model.item(1).isEnabled()                              # no widefield flags here
    finally:
        w.close()


def test_cancel_stops_the_rule(app, analysis, tmp_path):
    inputs = muw.UnanchoredInputs(analysis=analysis, data_path=str(tmp_path / "axon1.hdf5"))
    w = muw.UnanchoredWindow(inputs=inputs, noise_replicates=200, ask_reason=lambda a, lab: "x")
    try:
        assert w.analysis_changed()
        assert w.running() and w.btn_cancel.isEnabled()
        w.cancel()
        assert w.wait_run(60)
        assert "Cancelled" in w.status.full() and w.result is None and w.flags is None
    finally:
        w.close()


def test_the_tour_exists_and_its_targets_resolve(app, analysis, tmp_path):
    from tools import mps_tour
    w, _ = _window(analysis, tmp_path / "axon1.hdf5")
    try:
        steps = mps_tour.steps_for(muw.TOUR_ID)
        assert len(steps) >= 3
        for s in steps:
            if s.target:
                assert mps_tour.find_target(w, s.target) is not None, s.target
    finally:
        w.close()


def test_the_name_of_the_researcher_is_not_in_the_new_files():
    for name in ("tools/mps_unanchored_window.py", "test_unanchored_window_gui.py"):
        text = (Path(__file__).resolve().parent / name).read_text(encoding="utf-8").lower()
        assert ("un" + "sain") not in text and ("un" + "saín") not in text


# ------------------------------------------------------------------------------------------------ results window
def test_results_window_has_one_unanchored_button_and_a_progress_line(app):
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    opened, cancelled = [], []
    w = MPSResultsWindow(None, links=WindowLinks(open_unanchored=lambda: opened.append(1),
                                                 cancel_unanchored=lambda: cancelled.append(1)))
    try:
        buttons = [b for b in w.findChildren(QtWidgets.QPushButton) if "unanchored" in b.text().lower()]
        assert [b.text() for b in buttons] == ["Unanchored..."] and buttons[0].isEnabled()
        buttons[0].click()
        assert opened == [1]
        w.show_side_progress("Unanchored clusters - Rule (1 s): noise...", 400, True)
        assert w.side_progress.bar.value() == 400 and w.side_progress.btn_cancel.isEnabled()
        w.side_progress.btn_cancel.click()
        assert cancelled == [1]
        w.show_side_progress("Applied.", -1, False)
        assert not w.side_progress.btn_cancel.isVisibleTo(w)
    finally:
        w.close()


def test_results_window_without_the_link_disables_the_button(app):
    from tools.mps_results_window import MPSResultsWindow
    w = MPSResultsWindow(None)
    try:
        assert not w.btn_unanchored.isEnabled()
    finally:
        w.close()


def test_export_image_offers_the_legend_on_the_regularity_plots(app, monkeypatch):
    """Manual test 6.3: the legend box was ticked but the regularity PNG came out without it -- the dialog was told
    that only the map, the axial view and the 1NN plot carry a legend."""
    import tools.figure_export_ui as feu
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    seen = {}

    def fake_ask(plots, suggested, **kw):
        seen.update(kw)
        seen["plots"] = list(plots)
        return None
    monkeypatch.setattr(feu, "ask", fake_ask)
    w = MPSResultsWindow(None, links=WindowLinks(root_name=lambda: "x"))
    try:
        w._on_export_image()
        regularity = set(w.arrangement.figure_plots())
        assert regularity and regularity <= set(seen["legend"])
    finally:
        w.close()


def test_results_window_says_when_the_discard_leaves_nothing_out_and_who_discarded(app, analysis):
    """Manual test of 2026-11-19: with nothing left out 'Measured' and 'Discard applied' were the same and the choice
    looked broken; a cluster excluded by hand was named 'Discarded by the axoplasm panel' on the map."""
    from tools import mps_analysis as ma
    from tools.mps_results_window import _COL_DISCARD, _COL_MEASURED, MPSResultsWindow
    res = mu.auto_for_analysis(analysis, noise_replicates=0, compare_references=False)
    k = analysis.n_clusters_kept
    held = {"cmp": mu.compare_unanchored(analysis, mu.final_flags(res, method="off"))}
    w = MPSResultsWindow(analysis, discard_callback=lambda a: held["cmp"])
    try:
        w.refresh()
        assert w.radio_measured.text() == f"measured ({k} clusters)"
        assert w.radio_discard.text() == f"discard applied ({k}, nothing left out)", w.radio_discard.text()
        assert "Nothing is left out now" in w.radio_discard.toolTip()
        assert "Nothing is left out now" in w.table.horizontalHeaderItem(_COL_DISCARD).toolTip()
        assert not w.table.isColumnHidden(_COL_DISCARD) and not w.table.isColumnHidden(_COL_MEASURED)
        # the rule's two and one by hand left out: the count alone on the button, who left them out in the tooltip
        flags = mu.final_flags(res)
        anchored = int(np.flatnonzero(~flags.excluded)[0])
        hand = mu.add_decision(mu.ManualDecisions(), *np.asarray(analysis.centroids)[anchored], mu.ACTION_EXCLUDE,
                               "glia")
        held["cmp"] = mu.compare_unanchored(analysis, mu.final_flags(res, hand))
        w.refresh()
        n_out = k - held["cmp"].discard_applied.n_clusters_kept
        assert n_out == 3
        assert w.radio_discard.text() == f"discard applied ({k - n_out})"
        tip = w.radio_discard.toolTip()
        now = tip.split("\n\n")[-1]                  # after the common text, which names both sources
        assert f"Now {n_out} of the {k} clusters are left out" in now and "Unanchored clusters window" in now
        head = w.table.horizontalHeaderItem(_COL_DISCARD).toolTip()
        assert "Nothing is left out" not in head and "Unanchored clusters window" in head
        assert "axoplasm panel\ndiscarded" not in head
        w.radio_discard.setChecked(True)
        w.refresh()
        box = w.axon_map.layers.checkbox("c_discarded")
        assert box.text() == f"Discarded (unanchored rule / by hand) ({n_out})", box.text()
        # the Axoplasm panel's own discard keeps its name
        held["cmp"] = ma.compare_discard(analysis, flags.excluded, margin_nm=400.0)
        w.refresh()
        assert w.axon_map.layers.checkbox("c_discarded").text().startswith("Discarded by the axoplasm panel")
        assert "Axoplasm panel" in w.radio_discard.toolTip().split("\n\n")[-1]
    finally:
        w.close()


LONG_REASON = "debris beside the ring, probably a glial process seen in the widefield tubulin image"


def _window_with_two_decisions(analysis, tmp_path, width, room_for_the_record=False):
    """The window as it opens (1200 x 820) with two decisions by hand of a long reason, then ``width`` wide (or its
    minimum, if wider: offscreen every character is a box about twice as wide as on a Windows screen, so the top
    bar alone asks for more than 1920 px there - pixel widths of a real screen cannot be reproduced; the rule can).
    ``room_for_the_record``: at least wide enough that half of it holds the record (the case of a 1920 px screen)."""
    inputs = muw.UnanchoredInputs(analysis=analysis, data_path=str(tmp_path / "axon1.hdf5"))
    w = muw.UnanchoredWindow(inputs=inputs, noise_replicates=REPS, ask_reason=lambda action, label: LONG_REASON)
    w.resize(1200, 820)
    w.show()
    _run(w)
    for row in np.flatnonzero(~w.flags.excluded)[:2]:
        assert w.toggle_cluster(int(row))
    width = max(width, w.minimumSizeHint().width())
    if room_for_the_record:
        width = max(width, 2 * (w.log_width_needed() + muw.MARGIN) + 20)
    w.resize(width, 900)
    for _ in range(5):
        QtWidgets.QApplication.processEvents()
    assert w.width() == width
    return w


def test_decisions_table_shows_the_date_without_a_horizontal_bar(app, analysis, tmp_path):
    """Manual test of 2026-11-19: maximized at 1920 px the record's last column (Date) was cut off and a horizontal
    bar appeared. Every column but the reason fits its text, the reason takes what is left (elided, whole in its
    tooltip), and the splitter gives the record the width it needs when the map keeps half."""
    w = _window_with_two_decisions(analysis, tmp_path, 1920, room_for_the_record=True)
    try:
        t = w.table
        assert t.rowCount() == 2
        left, side = w.split.sizes()
        assert w.log_width_needed() <= (left + side) // 2, "the case where it fits"
        assert side >= w.log_width_needed() and left >= side, w.split.sizes()
        date = muw.LOG_HEADERS.index("Date")
        right = t.columnViewportPosition(date) + t.columnWidth(date)
        assert right <= t.viewport().width(), (right, t.viewport().width(), w.split.sizes())
        assert not t.horizontalScrollBar().isVisible()
        assert t.columnWidth(date) >= t.sizeHintForColumn(date) and t.item(0, date).text()
        for j in range(t.columnCount()):
            if j != muw.LOG_STRETCH:                   # whole: no column but the reason is ever squeezed
                assert t.columnWidth(j) >= max(t.sizeHintForColumn(j), t.horizontalHeader().sectionSizeHint(j))
        reason = t.item(0, muw.LOG_STRETCH)
        assert reason.text() == LONG_REASON and reason.toolTip() == LONG_REASON
        # a narrower side the user dragged is widened again only up to what the record needs
        w.split.setSizes([left + side - 400, 400])
        w.resize(w.width() + 10, w.height())
        QtWidgets.QApplication.processEvents()
        assert w.split.sizes()[1] >= w.log_width_needed()
    finally:
        w.close()


def test_decisions_table_never_takes_more_than_half_the_window(app, analysis, tmp_path):
    """When the record needs more than half the width (here: a table font three times as large), the map keeps its
    half and the record gets the other half."""
    w = _window_with_two_decisions(analysis, tmp_path, 1366)
    try:
        f = w.table.font()
        f.setPointSizeF(f.pointSizeF() * 3)
        w.table.setFont(f)
        w._fill_log()
        QtWidgets.QApplication.processEvents()
        left, side = w.split.sizes()
        assert w.log_width_needed() > (left + side) // 2, "the case where it does not fit"
        assert abs(left - side) <= 1, w.split.sizes()
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------ the batch
def test_batch_row_carries_both_columns_and_the_block(app, analysis, tmp_path):
    from tools import mps_batch as mb
    data = tmp_path / "axon1.hdf5"
    measured, discard, block = mb.unanchored_for_file(analysis, str(data))
    assert discard is not None and discard.n_clusters_kept == analysis.n_clusters_kept - 2
    assert block["unanchored_status"] == "ok" and block["unanchored_n_final"] == 2
    # a decision by hand saved beside the file is read by the batch too
    rows = np.flatnonzero(~mu.auto_for_analysis(analysis, noise_replicates=0).unanchored)
    c = np.asarray(analysis.centroids)[int(rows[0])]
    mu.save_decisions(mu.add_decision(mu.ManualDecisions(source="axon1.hdf5"), float(c[0]), float(c[1]),
                                      mu.ACTION_EXCLUDE, "glia"), mu.decisions_path(str(data)))
    _m, discard2, block2 = mb.unanchored_for_file(analysis, str(data))
    assert block2["unanchored_n_excluded_manual"] == 1 and discard2.n_clusters_kept == analysis.n_clusters_kept - 3
    os.environ["MPS_UNANCHORED_AUTO"] = "0"
    try:
        assert mb.unanchored_for_file(analysis, str(data)) == (analysis, None, None)
    finally:
        del os.environ["MPS_UNANCHORED_AUTO"]
