"""The perimeter density window (P12): one button, one window, one box in the Batch (offscreen Qt).

Simulated data only (``tools.mps_perimeter_sim``). The automatic contour is injected (the simulator's true curve)
so these tests do not depend on the contour package; a short null keeps every run in seconds.

Run (one file per process): QT_QPA_PLATFORM=offscreen py -3 -m pytest test_perimeter_window_gui.py -q
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import mps_perimeter_types as pdt  # noqa: E402
from tools import mps_perimeter_window as mpw  # noqa: E402
from tools.mps_perimeter_sim import simulate_scenario  # noqa: E402

SOURCE = "C:/data/26.1.15_ROI 1/26.1.15_ROI1_axon2.hdf5"      # synthetic path: year 26, ROI 1, axon 2
FAST = replace(pdt.PerimeterParams(), n_null=99)             # [ad hoc] a short null: the window, not the statistics


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope="module")
def sim():
    return simulate_scenario("z_peak_focus", n_locs=6000, seed=1)


def _true_contour(sim):
    from test_perimeter_analysis import true_contour
    return true_contour(sim)


def _slow_contour(sim, seconds=1.5):
    fn = _true_contour(sim)

    def slow(x, y, **kw):
        time.sleep(seconds)
        return fn(x, y, **kw)
    return slow


@pytest.fixture(scope="module")
def inputs(sim):
    return mpw.PerimeterInputs(x_nm=sim.x_nm, y_nm=sim.y_nm, z_nm=sim.z_nm, columns={"sx": sim.sx, "sy": sim.sy},
                               info=[], source_path=SOURCE)


@pytest.fixture(scope="module")
def window(app, sim, inputs):
    w = mpw.MPSPerimeterWindow(inputs, params=FAST, contour_fn=_true_contour(sim))
    w.show()
    assert w.start_run()
    assert w.wait_run(120)
    yield w
    w.close()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _settle():
    end = time.monotonic() + 30
    while mpw.live_runs() and time.monotonic() < end:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.01)


# ------------------------------------------------------------------------------------------------ the entrance
def test_one_button_in_the_results_window_and_search_finds_it(app, inputs):
    from tools import ui_search
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    w = MPSResultsWindow(None, links=WindowLinks(perimeter_inputs=lambda: inputs))
    try:
        buttons = [b for b in w.findChildren(QtWidgets.QPushButton) if "perimeter" in b.text().lower()]
        assert [b.text() for b in buttons] == ["Perimeter..."]
        assert buttons[0] is w.btn_perimeter and buttons[0].isEnabled()
        for word in ("perimeter", "density", "slices", "tortuosity"):
            hits = ui_search.find_matches(w, word)
            assert any(getattr(m, "widget", None) is w.btn_perimeter for m in hits), word
        # no second way in: nothing else in the window opens it (no menu entry)
        assert not [a for a in w.findChildren(QtWidgets.QAction) if "perimeter" in a.text().lower()]
    finally:
        w.close()


def test_button_is_disabled_without_an_axon_provider(app):
    from tools.mps_results_window import MPSResultsWindow
    w = MPSResultsWindow(None)
    try:
        assert not w.btn_perimeter.isEnabled()
    finally:
        w.close()


def test_button_opens_one_window_and_its_notes_reach_analysis_notes(app, sim, inputs):
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    w = MPSResultsWindow(None, links=WindowLinks(perimeter_inputs=lambda: inputs))
    try:
        w.btn_perimeter.click()
        first = w._perimeter_window
        assert isinstance(first, mpw.MPSPerimeterWindow)
        w.btn_perimeter.click()
        assert w._perimeter_window is first                    # pressed again: raised, not duplicated
        w.set_perimeter_notes([mpw.NOTE_PREFIX + "a test note"])   # no analysis shown: nothing to attach to
        assert w._warning_count == 0
        first.close()
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------ the window
def test_opens_with_the_defaults_and_a_bar_with_nothing_else(app, inputs):
    w = mpw.MPSPerimeterWindow(inputs, params=FAST)
    try:
        p = w.selected_params()
        assert p.z0_criterion == pdt.Z0Criterion.PEAK and p.contour_method == pdt.DEFAULT_CONTOUR_METHOD
        assert p.band_half_width_nm == pdt.BAND_HALF_WIDTH_NM and p.band_mode == "fixed"
        assert [w.combo_centre.itemText(i) for i in range(w.combo_centre.count())] == [
            "Peak of the z density curve", "Focus, nominal", "Focus, empirical"]
        assert w.combo_centre.currentIndex() == 0                   # the density curve's peak by default
        assert [w.combo_contour.itemText(i) for i in range(w.combo_contour.count())] == ["Automatic", "Drawn"]
        assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == list(mpw.TAB_NAMES)
        assert w.btn_run.isEnabled() and not w.btn_cancel.isEnabled() and not w.btn_save.isEnabled()
        assert not w.progress.isVisible() and w.result is None
        # one way to save: a single "Save..." button, no Browse / Export pair
        texts = [b.text() for b in w.findChildren(QtWidgets.QPushButton)]
        assert texts.count("Save...") == 1 and not [t for t in texts if t.lower().startswith(("browse", "export"))]
        assert w.font().pointSizeF() >= mpw.MIN_POINT_SIZE
    finally:
        w.close()


def test_run_fills_the_three_tabs(window):
    r = window.result
    assert r is not None and r.status == pdt.Status.OK, (r.status, r.failure)
    assert window.status.full().startswith("Done")
    assert not window.progress.isVisible()
    # Slices: the cloud (three slices, contour and two band edges), the z density curve, the tables
    assert len(window.plot_cloud.listDataItems()) >= 3
    assert len(window.plot_z.listDataItems()) == 1                      # the density curve, with its markers
    t = window.table_slices
    assert t.rowCount() == 3 and t.item(1, 0).text().startswith("z0") and t.item(1, 4).text() == "100"
    assert all(t.item(i, j).text() for i in range(3) for j in range(6))
    assert window.table_z0.item(0, 1).text() not in ("", "n/a")          # the peak
    assert "Lateral slices hold" in window.line_slices.full() and "Contour:" in window.line_contour.full()
    assert window.line_contour_state.full()                             # the contour's state is always said
    # Profile: three curves, A and C with numbers, readings
    assert len(window.plot_profile.listDataItems()) == 3
    pt = window.table_pairs
    assert [pt.item(i, 0).text() for i in range(3)] == ["z-1 vs z0", "z0 vs z+1", "z-1 vs z+1"]
    assert all(pt.item(i, j).text() for i in range(3) for j in range(1, 4))
    assert window.table_a.rowCount() == 3 and window.table_a.item(0, 3).text() not in ("", "n/a")
    assert window.line_a.full().startswith("A (central slice)") and "C:" in window.line_c.full()
    assert window.line_a_reading.full().startswith("A")
    assert window.line_c_reading.full().startswith("C")
    # Curvature: tortuosity and density per window, Spearman, z and p, a reading and its control
    assert len(window.plot_tau.listDataItems()) == 1 and len(window.plot_wdens.listDataItems()) == 1
    bt = window.table_b
    assert bt.item(0, 0).text() == "Spearman rho" and bt.item(0, 1).text() not in ("", "n/a")
    assert bt.item(1, 0).text() == "z" and bt.item(2, 0).text() == "p"
    assert window.line_b_reading.full().startswith("B")
    assert "Control: tortuosity vs radial offset" in window.line_b_control.full()
    assert window.btn_save.isEnabled()


def test_reading_warnings_are_visible_in_one_line_each(window):
    assert "exploratory" in window.line_a_warn.full() and "exclusion zone" in window.line_a_warn.full()
    assert "pooled" in window.line_a_pooled.full()
    assert "NEGATIVE" in window.line_b_sign.full() and "convex" in window.line_b_sign.full()
    assert "no comparison between animals" in window.line_one_axon.full()
    assert "clean reading" in window.line_c_clean.full() and "contour error" in window.line_c_contour.full()
    assert "Contour: knot" in window.line_contour.full() or "Contour: drawn" in window.line_contour.full()
    assert "of the central slice" in window.line_slices.full()
    assert "peak - focus" in window.line_z0.full()


def test_every_text_is_one_line_and_the_tooltips_are_short(window):
    for text in window.one_line_texts():
        assert "\n" not in text, text
    for w in (window.combo_centre, window.spin_band, window.combo_contour, window.btn_run, window.btn_cancel,
              window.btn_save):
        assert len(w.toolTip()) <= 100, w.toolTip()
    for table in (window.table_slices, window.table_z0, window.table_pairs, window.table_b, window.table_a):
        for j in range(table.columnCount()):
            header = table.horizontalHeaderItem(j).text()
            assert "\n" not in header
            # the header text fits its column: no cut headers
            assert table.columnWidth(j) >= table.fontMetrics().horizontalAdvance(header), header


def test_centre_selector_changes_z0(app, sim, inputs, window):
    r0 = window.result
    assert r0.z0.criterion == pdt.Z0Criterion.PEAK and r0.focus is not None
    assert r0.focus.empirical_nm is not None, "the simulated file must hold an empirical focus (sx != sy)"
    w = mpw.MPSPerimeterWindow(inputs, params=FAST, contour_fn=_true_contour(sim))
    try:
        i = w.combo_centre.findText("Focus, empirical")
        w.combo_centre.setCurrentIndex(i)
        assert w.selected_params().z0_criterion == pdt.Z0Criterion.FOCUS_EMPIRICAL
        assert w.start_run() and w.wait_run(120)
        r1 = w.result
        assert r1.status == pdt.Status.OK and r1.z0.criterion == pdt.Z0Criterion.FOCUS_EMPIRICAL
        assert r1.z0.z0_nm == pytest.approx(r1.focus.empirical_nm)
        assert abs(r1.z0.z0_nm - r0.z0.z0_nm) > 10.0               # the simulated focus is 43 nm from the peak
        assert w.table_z0.item(3, 1).text() == f"{r1.z0.z0_nm:.0f}"
    finally:
        w.close()


def test_a_focus_the_file_lacks_stops_with_a_status_and_a_note(app, sim, inputs):
    notes = []
    w = mpw.MPSPerimeterWindow(inputs, params=FAST, contour_fn=_true_contour(sim), notes_callback=notes.append)
    try:
        w.combo_centre.setCurrentIndex(w.combo_centre.findText("Focus, nominal"))      # no calibration record
        assert w.start_run() and w.wait_run(120)
        assert w.result.status == pdt.Status.NO_FOCUS            # never a silent fall back to the peak
        assert "no focus" in w.status.full()
        assert notes and any("nominal" in n for n in notes[-1]) and all(n.startswith(mpw.NOTE_PREFIX) for n in notes[-1])
    finally:
        w.close()


def test_band_and_drawn_choices_reach_the_parameters(app, inputs):
    w = mpw.MPSPerimeterWindow(inputs, params=FAST)
    try:
        w.spin_band.setValue(150.0)
        w.combo_contour.setCurrentIndex(w.combo_contour.findText("Drawn"))
        p = w.selected_params()
        assert p.band_half_width_nm == 150.0 and p.contour_method == "drawn"
        assert not w.start_run()                                  # no drawn trace in these inputs
        assert "No drawn trace" in w.status.full() and w.result is None
    finally:
        w.close()


def test_run_without_an_axon_says_so(app):
    w = mpw.MPSPerimeterWindow(None, params=FAST)
    try:
        assert not w.start_run()
        assert mpw.NO_INPUTS in w.status.full()
    finally:
        w.close()


def test_progress_bar_moves_and_cancel_really_stops_the_run(app, sim, inputs):
    """A contour that takes 1.5 s: the bar shows while the run goes, Cancel is enabled only then, and Cancel stops
    the analysis at its next step (the worker raises; nothing is shown)."""
    # live_runs() counts the workers of EVERY window in the process. Run after other test files, a worker an earlier
    # test left unjoined (its window closed while it ran) is still in the count and was read as this test's own.
    # The count that matters is the one this run adds: take the baseline before it starts.
    baseline = mpw.live_runs()
    w = mpw.MPSPerimeterWindow(inputs, params=FAST, contour_fn=_slow_contour(sim))
    w.show()
    try:
        assert not w.btn_cancel.isEnabled()
        assert w.start_run()
        assert w.btn_cancel.isEnabled() and not w.btn_run.isEnabled() and w.progress.isVisible()
        end = time.monotonic() + 5
        while time.monotonic() < end and w.progress.value() == 0:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.02)
        assert w.progress.value() > 0                              # the contour step was reached and shown
        assert "Running" in w.status.full() and "contour" in w.status.full()
        w.cancel()
        assert not w.btn_cancel.isEnabled()
        assert w.wait_run(30)
        assert "Cancelled" in w.status.full() and w.result is None
        assert w.btn_run.isEnabled() and not w.progress.isVisible()
        _settle()
        assert mpw.live_runs() <= baseline                          # this run was joined (a foreign one may remain)
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------ save
def test_save_uses_the_name_chosen_and_asks_before_replacing(window, tmp_path):
    target = tmp_path / "my_axon.csv"
    first = window.save_to(str(target))
    names = sorted(os.path.basename(p) for p in first)
    assert str(target) == first[0]                                 # the name chosen, as chosen
    assert "my_axon_perimeter_profiles.csv" in names and "my_axon_columns.csv" in names
    assert "my_axon_perimeter_contour_columns.csv" in names
    header = target.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:2] == ["axon_id", "source_path"] and "pd_status" in header
    before = {p: _sha(p) for p in first}
    asked = []
    assert window.save_to(str(target), confirm=lambda e: asked.append(e) or False) == []
    assert asked and {p: _sha(p) for p in first} == before          # refused: nothing touched
    again = window.save_to(str(target), confirm=lambda e: True)
    assert sorted(again) == sorted(first)                          # replaced in place, no _run2


def test_save_button_opens_a_dialog_and_respects_the_name(window, tmp_path, monkeypatch):
    chosen = str(tmp_path / "chosen_name.csv")
    seen = {}

    def fake(parent, title, start, flt):
        seen["start"] = start
        return chosen, flt
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(fake))
    window.btn_save.click()
    assert seen["start"].endswith("_perimeter.csv")
    assert os.path.exists(chosen) and os.path.exists(str(tmp_path / "chosen_name_perimeter_profiles.csv"))
    assert "Saved chosen_name.csv" in window.status.full()


def test_save_without_a_result_is_refused(app, inputs, tmp_path):
    w = mpw.MPSPerimeterWindow(inputs, params=FAST)
    try:
        with pytest.raises(ValueError):
            w.save_to(str(tmp_path / "x.csv"))
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------ pure texts
def test_notes_and_contour_state(window):
    r = window.result
    assert mpw.perimeter_notes(r) == mpw.perimeter_notes(r)       # deterministic, no repeated line
    assert len(set(mpw.perimeter_notes(r))) == len(mpw.perimeter_notes(r))
    truncated = replace(r, diagnostics=replace(r.diagnostics, z_truncated_slices=(-1,)))
    assert any("truncates slice(s) -1" in n for n in mpw.perimeter_notes(truncated))
    rejected = replace(r, status=pdt.Status.CONTOUR_FAILED, failure="contour failed: too few nodes",
                       contour=replace(r.contour, status=pdt.Status.CONTOUR_FAILED, path=None,
                                       warnings=("too few nodes",)))
    assert any("contour failed" in n for n in mpw.perimeter_notes(rejected))
    assert mpw.contour_state(rejected) == ("bad", "Contour rejected: too few nodes")
    prefix = mpw._low_quality_prefix()
    unreliable = replace(r, contour=replace(r.contour, warnings=(prefix + "frac100 0.41 < 0.55",)))
    assert mpw.contour_low_quality(unreliable)
    kind, text = mpw.contour_state(unreliable)
    assert kind == "bad" and "not reliable" in text and "frac100 0.41" in text
    assert any("contour not reliable (frac100 0.41 < 0.55)" in n for n in mpw.perimeter_notes(unreliable))
    good = replace(r, contour=replace(r.contour, warnings=()))
    assert mpw.contour_state(good)[0] == "good" and not mpw.contour_low_quality(good)
    assert mpw.contour_state(replace(r, contour=None))[0] == "dim"


def test_readings_follow_the_numbers(window):
    r = window.result
    t = r.b.test
    neg = replace(r, b=replace(r.b, test=replace(t, value=-0.4, p=0.01)))
    pos = replace(r, b=replace(r.b, test=replace(t, value=0.4, p=0.01)))
    none = replace(r, b=replace(r.b, test=replace(t, value=-0.4, p=0.5)))
    assert "more curved" in mpw.b_reading(neg) and "straighter" in mpw.b_reading(pos)
    assert "no link" in mpw.b_reading(none)
    c = r.a["contrast"]
    up = replace(r, a=dict(r.a, contrast=replace(c, z=3.0, p=0.001)))
    assert "more density near the maximum" in mpw.a_reading(up)


def test_z_density_curve_peaks_where_the_analysis_does(window):
    from tools import mps_perimeter_slices as sl
    z = window.last_inputs.z_nm
    zc, d = mpw.z_density_curve(z)
    assert d.max() == pytest.approx(1.0)
    assert float(zc[int(np.argmax(d))]) == pytest.approx(sl.z_peak_nm(z), abs=1.0)


def test_contour_kind_names():
    base = dict(path=None, length_nm=1.0, residual_sd_nm=0.0, n_self_crossings=0, nodes_nm=np.zeros((0, 2)),
                polar_distance_nm=float("nan"), star_shaped=True, status=pdt.Status.OK)
    knot = lambda m, k: pdt.AutoContour(method=m, knot_nm=k, **base)             # noqa: E731
    assert mpw.contour_kind_text(knot("density_nodes", 400.0)) == "knot 400 nm (auto)"
    assert mpw.contour_kind_text(knot("density_nodes", 600.0)) == "knot 600 nm (auto)"
    assert mpw.contour_kind_text(knot("polar", float("nan"))) == "polar (auto)"
    assert mpw.contour_kind_text(knot("drawn", float("nan"))) == "drawn"
    assert mpw.contour_kind_text(None) == "none"


def test_a_zero_is_never_written_with_a_sign():
    # the manual test showed "-0" for a nominal focus of a fraction of a nanometre below zero
    assert mpw._num(-0.3, "{:.0f}") == "0" and mpw._num(-0.0, "{:.0f}", " nm") == "0 nm"
    assert mpw._num(-0.0) == "0" and mpw._num(-0.004, "{:.2f}") == "0.00"
    assert mpw._signed(-0.2) == "0 nm" and mpw._signed(0.2, "") == "0"
    assert mpw._num(-1.0, "{:.0f}") == "-1" and mpw._signed(21.4) == "+21 nm" and mpw._signed(-22.0) == "-22 nm"
    assert mpw._num(-1e-5) == "-1e-05" and mpw._num(float("nan")) == "n/a"


def test_z0_line_and_table_say_zero_without_a_sign(window):
    r = window.result
    z = replace(r.z0, z0_nm=-0.3, peak_nm=-22.0, focus_nominal_nm=-0.3, focus_empirical_nm=-43.0,
                peak_minus_focus_nominal_nm=-21.7, peak_minus_focus_empirical_nm=21.0)
    line = mpw.z0_line(replace(r, z0=z))
    assert "z0 = 0 nm" in line and "-0" not in line
    rows = mpw.z0_table_rows(z)
    assert [row[0] for row in rows] == ["Peak (density curve)", "Focus, nominal", "Focus, empirical", "z0 used"]
    assert [row[1] for row in rows] == ["-22", "0", "-43", "0"]
    # "Peak - this": the reference mark, then peak minus each z (the z0 row computed here: -22 - (-0.3))
    assert [row[2] for row in rows] == [mpw.REFERENCE_MARK, "-22", "+21", "-22"]
    same = mpw.z0_table_rows(replace(z, z0_nm=-22.0))
    assert same[3][2] == "0"                                                   # z0 is the peak: no sign
    absent = mpw.z0_table_rows(replace(z, focus_nominal_nm=None, peak_minus_focus_nominal_nm=None))
    assert absent[1][1:] == [mpw.FOCUS_ABSENT, mpw.FOCUS_ABSENT]


def test_every_cell_of_the_z0_table_is_filled_and_its_header_reads(window):
    t = window.table_z0
    heads = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
    assert heads == ["Reference", "z [nm]", "Peak − this [nm]"]
    assert t.horizontalHeaderItem(2).toolTip()
    assert all(t.item(i, j).text() for i in range(t.rowCount()) for j in range(t.columnCount()))
    assert t.item(0, 2).text() == mpw.REFERENCE_MARK
    z = window.result.z0
    assert t.item(3, 2).text() == mpw._signed(z.peak_nm - z.z0_nm, "")


def test_z_coverage_is_written_and_explained(window):
    t = window.table_slices
    assert t.horizontalHeaderItem(5).text() == "z coverage" and t.horizontalHeaderItem(5).toolTip()
    cov = [float(t.item(i, 5).text()) for i in range(3)]
    assert cov == [pytest.approx(c, abs=0.005) for c in window.result.slices.z_coverage]
    assert all(0.0 <= c <= 1.0 for c in cov)


def test_tables_are_as_wide_as_their_columns_and_do_not_grow(window):
    # the stretched last column grew at every fill and pushed its right-aligned numbers out of view
    tables = (window.table_slices, window.table_z0, window.table_pairs, window.table_a, window.table_b)
    before = [[t.columnWidth(j) for j in range(t.columnCount())] for t in tables]
    for _ in range(3):
        window._fill_slices(window.result)
        window._fill_profile(window.result)
        window._fill_curvature(window.result)
        QtWidgets.QApplication.processEvents()
    after = [[t.columnWidth(j) for j in range(t.columnCount())] for t in tables]
    assert after == before
    for t in tables:
        assert not t.horizontalHeader().stretchLastSection()
        assert t.width() == mpw.table_content_width(t) == t.minimumWidth() == t.maximumWidth(), t.objectName()


def test_the_tour_of_the_window_exists_and_its_targets_resolve(window):
    from tools import mps_tour
    steps = mps_tour.steps_for(mpw.TOUR_ID)
    assert len(steps) >= 3
    for step in steps:
        if step.target:
            assert mps_tour.find_target(window, step.target) is not None, step.target
        for lang in ("es", "en"):
            assert step.text(lang, "title").strip()


def test_the_name_of_the_researcher_is_not_in_the_new_files():
    for name in ("tools/mps_perimeter_window.py", "test_perimeter_window_gui.py", "tools/mps_tour_steps.py"):
        text = (Path(__file__).resolve().parent / name).read_text(encoding="utf-8").lower()
        assert ("un" + "sain") not in text and ("un" + "saín") not in text


# ------------------------------------------------------------------------------------------------ the Batch box
def test_batch_has_one_perimeter_box_off_by_default(app):
    from tools.batch_window import BatchWindow
    from tools.mps_settings import MPSSettings
    w = BatchWindow(MPSSettings())
    try:
        boxes = [c for c in w.findChildren(QtWidgets.QCheckBox) if "perimeter" in c.text().lower()]
        assert [c.text() for c in boxes] == ["Perimeter density"]
        assert boxes[0] is w.check_perimeter and not boxes[0].isChecked()
        assert "\n" not in boxes[0].toolTip() and len(boxes[0].toolTip()) <= 100
    finally:
        w.close()


def test_batch_box_analyses_then_writes_under_the_export_name(app, sim, tmp_path):
    """The box's own step, on two simulated files: analysed in the worker (nothing written then), written at
    Export beside the table chosen there and named after it, never over an earlier export."""
    from tools.batch_window import BatchWindow
    from tools.mps_settings import MPSSettings
    for name, seed in (("axon1.csv", 1), ("axon2.csv", 2)):
        s = simulate_scenario("a_flat", n_locs=2500, seed=seed)
        with open(tmp_path / name, "w", encoding="utf-8") as fh:
            fh.write("id,frame,x [nm],y [nm],z [nm],uncertainty_xy [nm],intensity [photon]\n")
            for i in range(s.x_nm.size):
                fh.write(f"{i + 1},{i + 1},{s.x_nm[i] + 3000:.2f},{s.y_nm[i] + 3000:.2f},{s.z_nm[i]:.2f},8.0,1000\n")
    w = BatchWindow(MPSSettings())
    try:
        w.find_files(str(tmp_path))
        items = [w.plan.files[r] for r in range(len(w.plan.files))]
        assert len(items) == 2
        started = []
        line = w._run_perimeter_density(items, on_file=lambda i, p: started.append(i))
        assert line.startswith("Perimeter density:") and "failed" not in line, line
        assert started == [0, 1] and len(w.perimeter_results) == 2
        assert not list(tmp_path.glob("*_perimeter_*.csv"))          # nothing written by the analysis
        out = tmp_path / "out" / "prueba_batch.csv"
        stem = w.write_perimeter_tables(str(out))
        assert os.path.basename(stem) == "prueba_batch"
        tables = sorted(p.name for p in (tmp_path / "out").glob("prueba_batch_perimeter_*.csv"))
        assert "prueba_batch_perimeter_axons.csv" in tables and "prueba_batch_perimeter_log.csv" in tables
        before = {p: _sha(p) for p in (tmp_path / "out").glob("prueba_batch_perimeter_*.csv")}
        stem2 = w.write_perimeter_tables(str(out))
        assert stem2.endswith("prueba_batch_run2")
        assert {p: _sha(p) for p in before} == before                 # nothing overwritten
        axons = (tmp_path / "out" / "prueba_batch_perimeter_axons.csv").read_text(encoding="utf-8").splitlines()
        assert len(axons) == 3                                        # header + one row per ticked file
    finally:
        w.close()


def test_batch_without_the_box_writes_no_perimeter_tables(app, tmp_path):
    from tools.batch_window import BatchWindow
    from tools.mps_settings import MPSSettings
    w = BatchWindow(MPSSettings())
    try:
        assert w.write_perimeter_tables(str(tmp_path / "t.csv")) == ""
        assert not list(tmp_path.iterdir())
    finally:
        w.close()
