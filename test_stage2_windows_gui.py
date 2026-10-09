# -*- coding: utf-8 -*-
"""
Offscreen test of the stage-2 windows wired together (UI stage 2, design 8.4; IMPL-C).

IMPL-C moved the plots into the MPS analysis window (``tools.mps_results_window``): the axon map at the top, the
tabs Axial / Nearest neighbours / Cluster area / Scatter off the outline under it, the stale banner, the views the
Axoplasm panel and the Rings window ask for. ``test_axon_map_gui`` checks the widgets on their own; this test drives
the real main window on SIMULATED axon A (``golden_ui_stage2.make_inputs``) and checks the windows' contract:

   1. B10: the MPS analysis window opens before any analysis (empty table, banner with "Run", the ROI's z and the
      prefill fit on the Axial tab); "Run the MPS analysis" fills the same window;
   2. views and placement: "Show on the axon map" (Axoplasm view, discard radio), "Show the segments..." (Segments
      view of the map and of the axial view); ``beside_position`` and ``place_beside`` once per anchor;
   3. live redraw: the Axoplasm panel's margin reaches the map's caption with no other action;
   4. stale: an ROI drag with the Axoplasm panel open (banner, dropped comparison, disabled groups, stale cut,
      nothing recomputed), "Show the current selection" and back, "Save distances..." refused, Run;
   5. typed range (Q12): the cut is applied without re-running the analysis, and the banner names the new cut;
   6. export: every plot written, the information message, hidden tabs, the per-plot white offer and the dialog;
   7. SVG titles of the map (three views) and of the axial view (two views);
   8. every tab's title says which analysis it shows when a discard comparison exists;
   9. the main window's M4 / M5 / M2z widgets are hidden and point to the new window;
  10. 1366 x 728 with real fonts: the map's share of the window height.

R8: simulated data only; every file in a temporary folder (settings, review store, selection log included).

Run:  venv\\Scripts\\python.exe test_stage2_windows_gui.py      (offscreen; about 1-2 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Offscreen Qt has no fonts on Windows unless pointed at them (check 10 measures with the real ones).
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import html  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
T0 = time.perf_counter()
WORK = tempfile.mkdtemp(prefix="test_stage2_windows_gui_")
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-6:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def need(st: Dict[str, Any], key: str) -> Any:
    if key not in st:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return st[key]


def pump(app: Any, seconds: float = 0.1) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


def svg_text(plot_item: Any, name: str) -> str:
    from tools import figure_export
    path = os.path.join(WORK, name + ".svg")
    figure_export.write(plot_item, figure_export.FigureRequest(plot=name, path=path))
    with open(path, encoding="utf-8") as f:
        return html.unescape(f.read())


def title_text(plot_widget: Any) -> str:
    return str(plot_widget.getPlotItem().titleLabel.text)


def table_texts(table: Any, columns: Optional[List[int]] = None) -> List[Tuple[str, ...]]:
    cols = list(range(table.columnCount())) if columns is None else columns
    out = []
    for r in range(table.rowCount()):
        out.append(tuple("" if table.item(r, c) is None else table.item(r, c).text() for c in cols))
    return out


def main() -> int:
    print("=" * 100)
    print("STAGE-2 WINDOWS: MPS ANALYSIS WINDOW, AXOPLASM PANEL, RINGS, MAIN WINDOW (simulated axon A, offscreen)")
    print("=" * 100)
    from PyQt5 import QtCore, QtGui, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 9))
    st: Dict[str, Any] = {}
    msgs: List[Tuple[str, str, str]] = []

    # ------------------------------------------------------------------ 0. setup
    print("\n0. The main window on simulated axon A (settings, review store and selection log in a temp folder)")

    def setup() -> str:
        import golden_ui_stage2 as gold
        inputs = os.path.join(WORK, "inputs")
        described = gold.make_inputs(inputs)
        os.chdir(WORK)
        from tools import mps_settings
        settings_dir = os.path.join(WORK, "settings")
        os.makedirs(settings_dir, exist_ok=True)
        original = mps_settings.settings_path
        user_settings = original()
        st["user_settings"] = (str(user_settings), os.path.getmtime(user_settings)
                               if os.path.exists(user_settings) else None)
        mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)
        import MPS_explorer
        MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
        MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
        for kind in ("information", "warning", "critical", "question"):
            setattr(QtWidgets.QMessageBox, kind,
                    staticmethod(lambda *a, _k=kind, **k: (msgs.append(
                        (_k, a[1] if len(a) > 1 else "", a[2] if len(a) > 2 else "")),
                        QtWidgets.QMessageBox.Ok)[1]))
        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(WORK, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
        assert mw.load_channel2(os.path.join(inputs, "sim_axon_B.hdf5"), 0)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="stage2",
                                   roi_name="roi", axon_name="axonA")
        mw.identity_path = mw._identity_source()
        mw.scatterplot()
        d = described["A"]
        radius = d["ring_radius_nm"] + 450.0
        size = 2.0 * radius / MPS_explorer.ROI_DIAMETER_SCALE_FACTOR
        roi = mw.circular_roi
        roi.setSize((size, size), update=False, finish=False)
        roi.setPos((d["centre_nm"][0] - size / 2.0, d["centre_nm"][1] - size / 2.0), update=False, finish=False)
        mw.update_ROI()
        pump(app, 0.2)
        st.update(mw=mw, MPS_explorer=MPS_explorer, inputs=inputs, roi=roi)
        return f"{len(mw.zroi_unfiltered):,} localizations in the ROI before the cut; cut {mw._cut_now()}"

    check("setup", setup)

    # ------------------------------------------------------------------ 1. B10
    print("\n1. B10: the MPS analysis window before any analysis")

    def b10() -> str:
        mw = need(st, "mw")
        from tools.mps_results_window import NO_ANALYSIS
        w = mw.show_mps_window()
        pump(app, 0.2)
        assert w is mw.mps_window and w.analysis is None and mw.mps_analysis is None
        assert w.table.rowCount() == 1 and w.table.item(0, 0).text() == NO_ANALYSIS == \
            "No MPS analysis of this selection yet"
        assert w.banner.isVisible() and NO_ANALYSIS in w.banner.text(), w.banner.text()
        assert w.banner.btn_run.isVisibleTo(w.banner) and not w.banner.btn_show.isVisibleTo(w.banner), \
            "the empty state offers Run only"
        roi_out = w.axial.layer("roi_out")
        assert roi_out.label == "All" and roi_out.count == len(mw.zroi_unfiltered), (roi_out.label, roi_out.count)
        assert "roi_in" not in w.axial.layers.keys()
        fit = w.axial.inputs().fit
        assert fit is not None and fit is mw._prefill_fit, "the axial view draws the prefill's fit"
        assert w.axial.layers.is_group_enabled("mixture") and "mixture" in w.axial.layers.shown_keys()
        w.banner.btn_run.click()
        pump(app, 0.3)
        assert mw.mps_window is w, "Run fills the same window"
        assert w.analysis is not None and w.analysis is mw.mps_analysis
        assert not w.banner.isVisible(), w.banner.text()
        assert w.table.rowCount() > 1 and w.table.item(0, 0).text() != NO_ANALYSIS
        st["w"] = w
        return (f"banner '{w.banner.text() or '(hidden after Run)'}'; ROI z 'All' {roi_out.count:,}; prefill fit with "
                f"{len(fit.means_nm)} components; Run -> {w.analysis.n_clusters_kept} clusters in the same window")

    check("1. B10: empty state, Run fills the same window", b10)

    # ------------------------------------------------------------------ 9. the main window
    print("\n9. The main window: M4, M5, M2z hidden")

    def main_window() -> str:
        mw = need(st, "mw")
        ui = mw.ui
        names = ("verticalLayoutWidget_12", "verticalLayoutWidget_5", "groupBox_9", "lineEdit_Nneighbor",
                 "pushButton_Distances", "pushButton_savedistdata", "verticalLayoutWidget_3")
        mw.show()
        pump(app, 0.1)
        shown = [n for n in names if getattr(ui, n).isVisibleTo(mw)]
        assert not shown, f"still visible: {shown}"
        for label in ("label_moved_clusters", "label_moved_z"):
            lab = getattr(mw, label, None)
            assert isinstance(lab, QtWidgets.QLabel) and lab.objectName() == label and lab.text(), label
            assert lab.isVisibleTo(mw), f"{label} is not shown"
        gone = [n for n in ("KNdist_hist", "savedistdata", "latchange") if hasattr(mw, n)]
        assert not gone, f"still on the main window: {gone}"
        return f"{len(names)} widgets hidden; the two pointers shown; no KNdist_hist / savedistdata / latchange"

    check("9. main window: moved widgets hidden, pointers present, old methods gone", main_window)

    # ------------------------------------------------------------------ 10. 1366 x 728
    print("\n10. The MPS analysis window at 1366 x 728 with real fonts (Segoe UI 9)")

    def fit_1366() -> str:
        w = need(st, "w")
        assert not w.banner.isVisible()
        w.resize(1366, 728)
        w.show()
        pump(app, 0.4)
        share = w.axon_map.height() / w.height()
        min_w = w.minimumSizeHint().width()
        assert w.height() == 728, f"window height {w.height()} (minimum {w.minimumSizeHint().height()})"
        assert share >= 0.55, f"the map takes {share:.2f} of the window height"
        return (f"map {w.axon_map.height()} of {w.height()} px = {share:.2f} of the height; window {w.width()} px "
                f"wide, minimum width {min_w} px (not asserted: IMPL-D replaces the parameter row)")

    check("10. 1366 x 728: the axon map takes at least 55 % of the window height", fit_1366)

    # ------------------------------------------------------------------ 2a. the Axoplasm view
    print("\n2. Views and placement")

    def axoplasm_view() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        inputs = need(st, "inputs")
        mw.show_axoplasm_panel()
        aw = mw.axoplasm_window
        aw.load_tubulin(os.path.join(inputs, "wf_tubulin_A.tif"))
        aw.load_reference(os.path.join(inputs, "wf_spectrin_A.tif"))
        aw.flush()
        aw.measure()
        aw.wait(300)
        pump(app, 0.3)
        aw.flush()
        pump(app, 0.2)
        assert w.comparison is not None, "no discard comparison after the panel measured"
        assert w.axon_map.view() == "mps" and w.radio_measured.isChecked()
        assert aw.btn_show_map.isVisibleTo(aw) and aw.label_on_map.text()
        aw.btn_show_map.click()
        pump(app, 0.3)
        assert w.axon_map.view() == "axoplasm", w.axon_map.view()
        assert w.radio_discard.isChecked(), "the Axoplasm view shows the discard applied"
        assert w.axon_map.inputs().axoplasm is not None and w.axon_map.layers.is_group_enabled("images")
        key = aw.objectName() or type(aw).__name__
        assert key in w._placed_beside, "placed beside the panel the first time"
        st["aw"] = aw
        return (f"view 'axoplasm', radio 'discard applied' ({w.comparison.discard_applied.n_clusters_kept} of "
                f"{w.comparison.all_clusters.n_clusters_kept}); placed beside '{key}'")

    check("2. 'Show on the axon map': Axoplasm view, discard radio, placed beside", axoplasm_view)

    # ------------------------------------------------------------------ 3. live redraw
    print("\n3. Live redraw from the Axoplasm panel")

    def live_margin() -> str:
        w, aw = need(st, "w"), need(st, "aw")
        before = w.axon_map.layers.group_caption("images")
        old = aw.spin_margin.value()
        aw.spin_margin.setValue(old + 400)
        aw.flush()
        pump(app, 0.3)
        after = w.axon_map.layers.group_caption("images")
        new_words = f"margin {old + 400:,.0f} nm"
        assert before != after and new_words in after and f"margin {old:,.0f} nm" in before, (before, after)
        aw.spin_margin.setValue(old)
        aw.flush()
        pump(app, 0.3)
        back = w.axon_map.layers.group_caption("images")
        assert f"margin {old:,.0f} nm" in back, back
        return f"caption followed '{new_words}' and back to {old:g} nm with no other action"

    check("3. the margin reaches the map's caption live", live_margin)

    # ------------------------------------------------------------------ 8. titles name the analysis
    print("\n8. Titles say which analysis they show")

    def titles() -> str:
        w = need(st, "w")
        assert w.comparison is not None

        def read() -> Dict[str, str]:
            pump(app, 0.05)
            return {"area": title_text(w.plot_area), "nn": w.nn.nn_title(), "nn_plot": title_text(w.nn.plot_nn),
                    "cdf": w.nn.cdf_title(), "cdf_plot": title_text(w.nn.plot_cdf),
                    "scatter": title_text(w.plot_scatter)}

        w.radio_discard.setChecked(True)
        disc = read()
        bad = [k for k, t in disc.items() if "discard applied" not in t]
        assert not bad, {k: disc[k] for k in bad}
        w.radio_measured.setChecked(True)
        meas = read()
        bad = [k for k, t in meas.items() if "measured" not in t or "discard applied" in t]
        assert not bad, {k: meas[k] for k in bad}
        w.radio_discard.setChecked(True)
        pump(app, 0.05)
        return f"area: '{disc['area']}' / '{meas['area']}'"

    check("8. area, NN, CDF and scatter titles: 'discard applied' / 'measured'", titles)

    # ------------------------------------------------------------------ 2b. the segments view and placement
    def segments_view() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        w.btn_rings.click()
        pump(app, 0.3)
        rw = mw.rings_window
        assert rw is not None and rw.ms is not None and rw.lbl_params.text().startswith("Computed with eps")
        assert not rw.banner.isVisible(), rw.banner.text()
        ms, reason = w.links.rings()
        assert ms is rw.ms and reason == "", reason
        rw.btn_show_segments.click()
        pump(app, 0.3)
        assert w.axon_map.view() == "segments" and w.axial.view() == "segments"
        assert w.tabs.currentWidget() is w.axial
        rows = w.axon_map.layers.group_rows("localizations")
        assert rows and all(k.startswith("seg") for k in rows), rows
        assert set(w.axon_map.layers.shown_keys()) >= set(rows)
        assert any(k.startswith("segband") for k in w.axial.layers.shown_keys())
        st["rw"] = rw
        return f"map rows {rows}; axial view 'segments'; {len(ms.segments)} segments from the provider"

    check("2. 'Show the segments...': map and axial Segments views, seg* rows", segments_view)

    def placement() -> str:
        from tools import mps_results_window as R
        Rect, Size, Point = QtCore.QRect, QtCore.QSize, QtCore.QPoint
        avail = Rect(0, 0, 1920, 1040)
        # right when it fits
        assert R.beside_position(Rect(100, 100, 400, 300), Size(500, 400), avail) == Point(500, 100)
        # left otherwise
        assert R.beside_position(Rect(1500, 100, 400, 300), Size(500, 400), avail) == Point(1000, 100)
        # neither
        assert R.beside_position(Rect(200, 100, 1500, 300), Size(600, 400), avail) is None
        # taller than the screen
        assert R.beside_position(Rect(0, 0, 100, 100), Size(200, 1200), avail) is None
        # the top kept on the screen
        assert R.beside_position(Rect(100, 900, 400, 100), Size(500, 400), avail) == Point(500, 640)
        # place_beside moves only the first time per anchor
        w = need(st, "w")
        original = R.beside_position
        calls: List[Any] = []

        def fake(anchor: Any, size: Any, available: Any) -> Any:
            calls.append(anchor)
            return QtCore.QPoint(37, 41)

        a1, a2 = QtWidgets.QWidget(), QtWidgets.QWidget()
        a1.setObjectName("test_anchor_one")
        a2.setObjectName("test_anchor_two")
        a1.setGeometry(50, 50, 200, 200)
        a2.setGeometry(60, 60, 200, 200)
        R.beside_position = fake
        try:
            assert w.place_beside(a1) is True and (w.pos().x(), w.pos().y()) == (37, 41), w.pos()
            w.move(5, 5)
            assert w.place_beside(a1) is False and (w.pos().x(), w.pos().y()) == (5, 5), w.pos()
            assert len(calls) == 1, "asked the geometry again for the same anchor"
            assert w.place_beside(a2) is True and len(calls) == 2, "another anchor is placed once too"
        finally:
            R.beside_position = original
            a1.deleteLater()
            a2.deleteLater()
        return "right / left / None / clamped top; place_beside once per anchor"

    check("2. placement: beside_position (pure) and place_beside once per anchor", placement)

    # ------------------------------------------------------------------ 7. SVG titles
    print("\n7. SVG titles")

    def svg_titles() -> str:
        w = need(st, "w")
        m = w.axon_map
        counts: Dict[str, int] = {}
        for view in ("mps", "axoplasm", "segments"):
            m.set_view(view)
            pump(app, 0.05)
            svg = svg_text(m.plot_item(), f"map_{view}")
            for line in m.title_lines():
                for part in line.split("; "):
                    part = html.unescape(part)
                    assert part in svg, (view, part)
            counts[f"map {view}"] = len(m.title_lines())
        w.tabs.setCurrentWidget(w.axial)
        for view in ("slab", "segments"):
            w.axial.set_view(view)
            pump(app, 0.05)
            svg = svg_text(w.axial.plot_item(), f"axial_{view}")
            for line in w.axial.title_lines():
                for part in line.split("; "):
                    part = html.unescape(part)
                    assert part in svg, (view, part)
            counts[f"axial {view}"] = len(w.axial.title_lines())
        m.set_view("mps")
        w.axial.set_view("slab")
        return f"title lines found in every SVG: {counts}"

    check("7. every title line of the map (3 views) and the axial view (2 views) is in its SVG", svg_titles)

    # ------------------------------------------------------------------ 6. export
    print("\n6. Export image")

    def export_all() -> str:
        w = need(st, "w")
        from tools import figure_export, figure_export_ui
        from tools import mps_results_window as R
        names = [R.PLOT_MAP, R.PLOT_AXIAL, R.PLOT_AREA, R.PLOT_NN, R.PLOT_CDF, R.PLOT_SCATTER]
        assert list(w._plots()) == names
        original = figure_export_ui.ask
        offers: List[Any] = []
        written: List[str] = []
        try:
            for i, name in enumerate(names):
                ext = ".png" if name == R.PLOT_AREA else ".svg"
                path = os.path.join(WORK, "export_" + name.replace(" ", "_") + ext)

                def fake(plots: Any, suggested: str, parent: Any = None, offer_white: Any = True,
                         _n: str = name, _p: str = path) -> Any:
                    offers.append(offer_white)
                    return figure_export.FigureRequest(plot=_n, path=_p,
                                                       white=figure_export_ui.offers_white(offer_white, _n))

                figure_export_ui.ask = fake
                # a plot in a hidden tab: another tab is current before the export
                hidden_tab = w._tab_of(name)
                if hidden_tab is not None:
                    other = w.plot_scatter if hidden_tab is not w.plot_scatter else w.plot_area
                    w.tabs.setCurrentWidget(other)
                    pump(app, 0.05)
                    assert w.tabs.currentWidget() is not hidden_tab
                n = len(msgs)
                w._on_export_image()
                pump(app, 0.05)
                assert os.path.exists(path) and os.path.getsize(path) > 0, f"{name}: nothing written"
                if hidden_tab is not None:
                    assert w.tabs.currentWidget() is hidden_tab, f"{name}: its tab was not made current"
                new = msgs[n:]
                assert len(new) == 1 and new[0][0] == "information" and new[0][1] == "Image written", new
                text = new[0][2]
                assert "Title:" in text, (name, text)
                if name == R.PLOT_MAP:
                    assert "Visible layers:" in text and "hidden or not available" in text, text
                written.append(os.path.basename(path))
            # per-entry white: the map's image ticked -> no white for the map
            w.axon_map.set_view("axoplasm")
            pump(app, 0.05)
            assert "image" in w.axon_map.layers.shown_keys(), "the Axoplasm view draws the image"
            offers.clear()
            figure_export_ui.ask = lambda plots, suggested, parent=None, offer_white=True: (
                offers.append(offer_white), None)[1]
            w._on_export_image()
            with_image = offers[-1]
            assert isinstance(with_image, (set, frozenset)), with_image
            assert R.PLOT_MAP not in with_image and set(names) - {R.PLOT_MAP} == set(with_image), with_image
            w.axon_map.set_view("mps")
            pump(app, 0.05)
            w._on_export_image()
            assert set(offers[-1]) == set(names), offers[-1]
        finally:
            figure_export_ui.ask = original
        # the dialog with a set: chk_white only for the plots in it
        dlg = figure_export_ui.ExportFigureDialog(names, os.path.join(WORK, "dialog.svg"),
                                                  offer_white=set(names) - {R.PLOT_MAP})
        seen = {}
        for name in names + [R.PLOT_MAP]:
            dlg.combo_plot.setCurrentText(name)
            seen[name] = (dlg.chk_white.isVisibleTo(dlg), dlg.chk_white.isHidden())
            if name == R.PLOT_MAP:
                assert seen[name] == (False, True), (name, seen[name])
                assert dlg.request().white is False, "no white redraw of the map over the image"
            else:
                assert seen[name] == (True, False), (name, seen[name])
                assert dlg.request().white is True
        dlg.deleteLater()
        return (f"{len(written)} files ({', '.join(written)}); white offered for {len(with_image)} plots with the "
                f"map's image ticked (not 'Axon map'); dialog hides chk_white for the map")

    check("6. export: six plots, messages, hidden tabs, per-plot white, the dialog", export_all)

    # ------------------------------------------------------------------ 4. stale
    print("\n4. Stale: an ROI drag with the Axoplasm panel open")

    def stale() -> str:
        mw, w, aw, rw, roi = need(st, "mw"), need(st, "w"), need(st, "aw"), need(st, "rw"), need(st, "roi")
        from tools.mps_results_window import NO_ANALYSIS, _COL_DISCARD
        aw.show()
        pump(app, 0.1)
        w.axial.set_view("slab")
        w.axon_map.set_view("mps")
        assert w.comparison is not None and not w.banner.isVisible()
        analysis = w.analysis
        had_comparison = w.comparison is not None
        visible_cols = [c for c in range(w.table.columnCount()) if not w.table.isColumnHidden(c)]
        before_table = table_texts(w.table)
        before_prov = w.lbl_provenance.text()
        before_ctrl = (w.spin_eps.value(), w.spin_min.value(), w.spin_half.value(), w.spin_maha.value(),
                       [w.combo_peak.itemText(i) for i in range(w.combo_peak.count())],
                       w.combo_peak.currentIndex())
        roi.setPos((roi.pos().x() + 150.0, roi.pos().y()), update=False, finish=False)
        mw.update_ROI()
        pump(app, 0.3)
        sel = mw._selection_view()
        text = w.banner.text()
        assert w.banner.isVisible() and "previous selection" in text, text
        assert sel.analysed_words and sel.analysed_words in text, (sel.analysed_words, text)
        assert sel.current_words and sel.current_words in text and sel.current_words != sel.analysed_words
        if had_comparison:
            assert "comparison without the discarded clusters was dropped" in text, text
        assert w.banner.btn_show.isVisibleTo(w.banner) and w.banner.btn_run.isVisibleTo(w.banner)
        assert w.analysis is analysis, "the drag replaced the analysis"
        # nothing of the analysis changed
        after_table = table_texts(w.table)
        dropped_col = had_comparison and w.table.isColumnHidden(_COL_DISCARD)
        keep_cols = [c for c in visible_cols if c != _COL_DISCARD]
        assert table_texts(w.table, keep_cols) == [tuple(r[c] for c in keep_cols) for r in before_table], \
            "the table changed"
        assert w.lbl_provenance.text() == before_prov, (before_prov, w.lbl_provenance.text())
        after_ctrl = (w.spin_eps.value(), w.spin_min.value(), w.spin_half.value(), w.spin_maha.value(),
                      [w.combo_peak.itemText(i) for i in range(w.combo_peak.count())], w.combo_peak.currentIndex())
        assert after_ctrl == before_ctrl, (before_ctrl, after_ctrl)
        # the map: the Axoplasm groups (and the selection's localizations) disabled, "previous"
        m = w.axon_map
        assert not m.layers.is_group_enabled("images") and "previous" in m.layers.group_reason("images"), \
            m.layers.group_reason("images")
        m.set_view("axoplasm")
        pump(app, 0.05)
        for group in ("localizations", "centres", "images"):
            assert not m.layers.is_group_enabled(group), group
            assert "previous" in m.layers.group_reason(group), (group, m.layers.group_reason(group))
        assert m.title_lines()[0].endswith("(previous selection)"), m.title_lines()
        m.set_view("mps")
        # the axial view: the cut of the current selection is disabled
        box = w.axial.layers.checkbox("main_cut")
        assert not box.isEnabled(), "main_cut is enabled while the analysis is stale"
        # the rings
        assert rw.banner.isVisible() and "another selection" in rw.banner.text(), rw.banner.text()
        st["stale_info"] = (analysis, before_table, dropped_col, after_table)
        return (f"banner: '{text.splitlines()[0][:110]}...'; discard column "
                f"{'hidden (comparison dropped)' if dropped_col else 'kept'}; rings: '{rw.banner.text()[:60]}'")

    check("4. stale after an ROI drag: banner, groups disabled, analysis untouched, rings banner", stale)

    def stale_save() -> str:
        w = need(st, "w")
        out = os.path.join(WORK, "stale_distances.csv")
        asked: List[Any] = []
        original = QtWidgets.QFileDialog.getSaveFileName
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(  # type: ignore
            lambda *a, **k: (asked.append(a), (out, "CSV Files (*.csv)"))[1])
        try:
            n = len(msgs)
            result = w.nn.on_save()
        finally:
            QtWidgets.QFileDialog.getSaveFileName = original  # type: ignore
        new = msgs[n:]
        assert result is None and not asked and not os.path.exists(out), (result, asked)
        assert [(k, t) for k, t, _x in new] == [("warning", "No clusters"), ("warning", "No distance data")], new
        return "warnings 'No clusters', 'No distance data'; no dialog, nothing written"

    check("4. 'Save distances...' while stale: two warnings, nothing written", stale_save)

    def show_current() -> str:
        mw, w, aw = need(st, "mw"), need(st, "w"), need(st, "aw")
        from tools.mps_results_window import NO_ANALYSIS
        analysis = need(st, "stale_info")[0]
        w.banner.btn_show.click()
        pump(app, 0.3)
        assert w.table.rowCount() == 1 and w.table.item(0, 0).text() == NO_ANALYSIS
        assert w.analysis is analysis, "Export axon must still write the analysed selection"
        assert w.banner.btn_show.text() == "Show the analysed selection" and w.banner.isVisible()
        m = w.axon_map
        m.set_view("axoplasm")
        pump(app, 0.1)
        state = m.inputs().axoplasm
        assert m.inputs().analysis is None, "the current selection is drawn without an analysis"
        assert aw.inputs.selection_key is mw.roi_indices, "the panel follows the current selection"
        assert state is not None and state.has_mask, m.inputs().axoplasm_reason
        assert m.layers.is_group_enabled("images") and "image" in m.layers.shown_keys(), \
            m.layers.group_reason("images")
        assert w.axial.layer("roi_out").label == "All", "the axial view draws the current ROI's z"
        m.set_view("mps")
        w.banner.btn_show.click()
        pump(app, 0.3)
        assert w.table.rowCount() > 1 and w.analysis is analysis
        assert "previous selection" in w.banner.text() and w.banner.btn_show.text() == "Show the current selection"
        return "empty table + new ROI's mask and image; analysis kept; back to the analysed selection"

    check("4. 'Show the current selection' and back", show_current)

    def run_again() -> str:
        mw, w, rw = need(st, "mw"), need(st, "w"), need(st, "rw")
        analysis = need(st, "stale_info")[0]
        w.banner.btn_run.click()
        pump(app, 0.3)
        assert w.analysis is not analysis and w.analysis is mw.mps_analysis
        assert not w.banner.isVisible(), w.banner.text()
        assert w.table.rowCount() > 1
        assert w.axial.layers.checkbox("main_cut").isEnabled()
        return f"new analysis ({w.analysis.n_clusters_kept} clusters); banner hidden; rings banner " \
               f"{'still up (segments of the old selection)' if rw.banner.isVisible() else 'hidden'}"

    check("4. a new analysis (Run) returns to normal", run_again)

    # ------------------------------------------------------------------ 5. typed range
    print("\n5. Typed range (Q12)")

    def typed_range() -> str:
        mw, w, MPS_explorer = need(st, "mw"), need(st, "w"), need(st, "MPS_explorer")
        assert not w.banner.isVisible()
        original = MPS_explorer.analyze_axon
        calls = [0]

        def counting(*a: Any, **k: Any) -> Any:
            calls[0] += 1
            return original(*a, **k)

        MPS_explorer.analyze_axon = counting
        try:
            analysis = w.analysis
            zmin = float(mw.ui.lineEdit_zmin.text()) + 20
            zmax = float(mw.ui.lineEdit_zmax.text()) - 20
            for edit, v in ((mw.ui.lineEdit_zmin, zmin), (mw.ui.lineEdit_zmax, zmax)):
                edit.setText(f"{v:.1f}")
                edit.textEdited.emit(f"{v:.1f}")
            mw.ui.pushButton_zrange.click()
            pump(app, 0.3)
            assert calls[0] == 0, f"analyze_axon ran {calls[0]} time(s) for a typed range"
        finally:
            MPS_explorer.analyze_axon = original
        assert w.analysis is analysis
        assert (round(mw.zmin, 1), round(mw.zmax, 1)) == (round(zmin, 1), round(zmax, 1)), (mw.zmin, mw.zmax)
        cut_words = f"main-window cut {zmin:,.0f}..{zmax:,.0f} nm"
        assert w.banner.isVisible() and "previous selection" in w.banner.text(), w.banner.text()
        assert cut_words in w.banner.text(), (cut_words, w.banner.text())
        cut = w.axial.inputs().cut
        assert cut is not None and abs(cut[0] - zmin) < 0.06 and abs(cut[1] - zmax) < 0.06, cut
        return f"cut {zmin:.1f}..{zmax:.1f} applied, analyze_axon not called; banner names '{cut_words}'"

    check("5. typed range: cut applied without re-running; banner names the new cut", typed_range)

    # ------------------------------------------------------------------ R8
    def settings_untouched() -> str:
        path, mtime = need(st, "user_settings")
        now = os.path.getmtime(path) if os.path.exists(path) else None
        assert now == mtime, f"{path} was written"
        written = sorted(os.listdir(os.path.join(WORK, "settings")))
        return f"the program's own settings file untouched; temp settings: {written}"

    check("R8: settings only in the temporary folder", settings_untouched)

    for key in ("rw", "aw", "w", "mw"):
        try:
            need(st, key).close()
        except Exception:  # noqa: BLE001
            pass
    pump(app, 0.1)
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.0f} s; work folder {WORK}")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
