# -*- coding: utf-8 -*-
"""
Regression tests for manual test 2 of UI stage 2 (feat/ui-condense).

Each scenario drives the real windows offscreen on SIMULATED axon A (``golden_ui_stage2.make_inputs``) in its own
Python process, with every file (settings, review store, selection log, exports, figures) under its own temporary
folder, and the parent checks what it recorded:

  1  Export image: a path without a known extension gets ".png"; a read-only file, a missing folder and a file the
     exporter cannot write are reported, never "Image written" for a file that is not there (PNG and SVG);
  2  the typed Z range fields always show the current slab (component, automatic, half-width);
  3  text can be copied: tables (selected cells, tab-separated), the warnings, the labels;
  4  the MPS analysis window: the axon map has the larger share, every panel boundary is a splitter, sizes persist;
  5  "Warnings (N)..." writes a .txt and never overwrites one;
  6  one main-window button runs the analysis when none is current, otherwise shows the window;
  7  eps and min samples edited in the main window are the same stored value as the MPS window's strip;
  10 the "paper" badge reads "Gazal et al. (2026)", its tooltip keeps the full reference;
  11 loading a file draws the scatter, applies a circular ROI covering the whole axon, and selects DBSCAN;
  12 the animal defaults to "mouse";
  13 the Rings window carries the axial view and the cluster area per segment;
  14 a click on a cluster of the axon map shows its id, area and nearest-neighbour distances;
  15 the Axoplasm panel shows which widefield image is under the axon map.

R8: simulated data only.

Run:  venv\\Scripts\\python.exe test_mt2_gui.py      (offscreen; about 1-2 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SCENARIOS = ["figure_export", "typed_fields", "copy", "layout_warnings", "main_params"]

# Scenarios that measure geometry, with the platform's font.
FONT_SCENARIOS = {"layout_warnings", "main_layout"}


# ============================================================================ one scenario (child process)
def run_scenario(scenario: str, out_dir: str, inputs: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(out_dir, "selection_log")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    if scenario in FONT_SCENARIOS:
        # Geometry is measured: the platform's own font (as on the user's screen), not the offscreen fallback.
        os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
    from PyQt5 import QtGui, QtWidgets

    import golden_ui_stage2 as G

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["mt2"])
    if scenario in FONT_SCENARIOS:
        app.setFont(QtGui.QFont("Segoe UI", 9))
    cell = G.Cell("A", "S1", "E1", os.path.join(out_dir, "cell"), inputs)
    cell.patch()
    exc: List[str] = []
    sys.excepthook = lambda *e: exc.append("".join(traceback.format_exception(*e)))
    out: Dict[str, Any] = {"scenario": scenario}

    def pump(s: float = 0.05) -> None:
        cell.pump(app, s)

    def calls(name: str) -> List[Dict[str, Any]]:
        return [c for c in cell.calls if c.get("function") == name]

    with open(os.path.join(inputs, "inputs.json"), encoding="utf-8") as f:
        described = json.load(f)["axons"]["A"]

    def make_main(load: bool = True, roi: bool = True, cluster: bool = True) -> Any:
        import MPS_explorer
        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(out_dir, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        if not load:
            return mw
        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="mt2",
                                   roi_name="roi", axon_name="axonA")
        mw.identity_path = mw._identity_source()
        if roi:
            mw.scatterplot()
            cell.place_roi(mw, tuple(described["centre_nm"]), described["ring_radius_nm"] + 450.0)
        if cluster:
            mw.ui.pushButton_clusterch1.click()
            pump()
        return mw

    s = scenario
    if s == "figure_export":
        import pyqtgraph as pg

        from tools import figure_export as fe
        plot = pg.PlotWidget()
        plot.plot([0, 1, 2], [1, 3, 2])
        plot.resize(400, 300)
        plot.show()
        pump(0.1)
        item = plot.getPlotItem()
        res: Dict[str, Any] = {}

        def attempt(name: str, path: str) -> Dict[str, Any]:
            req = fe.FigureRequest(plot="p", path=path, width_mm=20.0, dpi=300)
            try:
                written = fe.write(item, req)
                return {"raised": "", "written": written,
                        "exists": os.path.isfile(written) and os.path.getsize(written) > 0}
            except Exception as error:  # noqa: BLE001 - recorded
                return {"raised": f"{type(error).__name__}: {error}", "written": "", "exists": False}

        folder = os.path.join(out_dir, "figs")
        os.makedirs(folder, exist_ok=True)
        res["no_extension"] = attempt("no_extension", os.path.join(folder, "fig_noext"))
        res["unknown_extension"] = attempt("unknown", os.path.join(folder, "fig.v2"))
        res["png"] = attempt("png", os.path.join(folder, "fig_ok.png"))
        res["svg"] = attempt("svg", os.path.join(folder, "fig_ok.svg"))
        res["no_folder"] = attempt("no_folder", os.path.join(folder, "missing", "fig.png"))
        import stat
        for ext in ("png", "svg"):
            ro = os.path.join(folder, f"read_only.{ext}")
            with open(ro, "wb") as fh:
                fh.write(b"x")
            os.chmod(ro, stat.S_IREAD)
            res[f"read_only_{ext}"] = attempt("ro", ro)
            res[f"read_only_{ext}"]["size_after"] = os.path.getsize(ro)
            os.chmod(ro, stat.S_IWRITE | stat.S_IREAD)
        # The window's own message: a file that is not written is "Export failed", never "Image written".
        mw = make_main()
        w = mw.mps_window
        from tools import figure_export_ui
        ro = os.path.join(folder, "window_read_only.png")
        with open(ro, "wb") as fh:
            fh.write(b"x")
        os.chmod(ro, stat.S_IREAD)
        figure_export_ui.ask = lambda plots, suggested, parent=None, offer_white=True: fe.FigureRequest(
            plot=list(plots)[1], path=ro, width_mm=20.0, dpi=300, white=False)
        n0 = len(cell.messages)
        w._on_export_image()
        pump()
        res["window_read_only"] = [m["title"] for m in cell.messages[n0:]]
        os.chmod(ro, stat.S_IWRITE | stat.S_IREAD)
        noext = os.path.join(folder, "window_noext")
        figure_export_ui.ask = lambda plots, suggested, parent=None, offer_white=True: fe.FigureRequest(
            plot=list(plots)[1], path=noext, width_mm=20.0, dpi=300, white=False)
        n0 = len(cell.messages)
        w._on_export_image()
        pump()
        res["window_noext"] = {"titles": [m["title"] for m in cell.messages[n0:]],
                               "texts": [m["text"] for m in cell.messages[n0:]],
                               "exists": os.path.isfile(noext + ".png")}
        out["figure_export"] = res
        mw.close()
    elif s == "typed_fields":
        mw = make_main()
        w = mw.mps_window

        def state() -> Dict[str, Any]:
            a = w.analysis
            return {"fields": [w.edit_typed_min.text(), w.edit_typed_max.text()],
                    "cut": [f"{float(mw.zmin):.1f}", f"{float(mw.zmax):.1f}"],
                    "slab": [f"{a.slab_zmin_nm:.1f}", f"{a.slab_zmax_nm:.1f}"]}
        out["start"] = state()
        cell.window_edit(app, w.spin_half, 70.0)
        pump()
        out["half_70"] = state()
        a = w.analysis
        centre = (a.slab_zmin_nm + a.slab_zmax_nm) / 2.0
        means = [float(m) for m in a.z_result.means_nm]
        far = max(range(len(means)), key=lambda i: abs(means[i] - centre))
        w.combo_peak.setCurrentIndex(far)
        pump()
        out["component"] = state()
        w.btn_slab_auto.click()
        pump()
        out["automatic"] = state()
        # Ticked, then the slab changes and "Apply": the bounds applied are the slab's, not the old ones.
        w.chk_typed.setChecked(True)
        out["ticked"] = state()
        mw.close()
    elif s == "copy":
        from PyQt5 import QtCore, QtTest
        mw = make_main()
        w = mw.mps_window
        w.show()
        pump(0.2)
        clip = QtWidgets.QApplication.clipboard()

        def press_copy(view: Any) -> str:
            clip.setText("<nothing>")
            view.setFocus()
            QtTest.QTest.keyClick(view, QtCore.Qt.Key.Key_C, QtCore.Qt.KeyboardModifier.ControlModifier)
            pump(0.02)
            return clip.text()

        t = w.table
        t.clearSelection()
        t.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(0, 0, 1, 1), True)
        want = "\n".join("\t".join(t.item(r, c).text() for c in (0, 1)) for r in (0, 1))
        out["table"] = {"got": press_copy(t), "want": want,
                        "menu": [a.text() for a in t.actions()]}
        menu_copy = [a for a in t.actions() if a.text() == "Copy"]
        clip.setText("<nothing>")
        menu_copy[0].trigger()
        out["table_menu"] = clip.text()
        # The rings window's warnings list (the MPS window's warnings are a .txt file since item 5).
        from PyQt5.QtCore import Qt
        flags = int(w.lbl_slab_mode.textInteractionFlags())
        out["label_selectable"] = bool(flags & int(Qt.TextInteractionFlag.TextSelectableByMouse))
        w.tabs.setCurrentWidget(w.area_tab)
        w.details_area.button.setChecked(True)
        pump(0.05)
        d = w.details_area.table
        d.clearSelection()
        d.selectRow(0)
        out["details"] = {"got": press_copy(d),
                          "want": "\t".join(d.item(0, c).text() for c in range(d.columnCount()))}
        w.btn_rings.click()
        pump(0.3)
        rw = mw.rings_window
        ts = rw.table_seg
        ts.clearSelection()
        ts.selectRow(0)
        out["rings"] = {"got": press_copy(ts),
                        "want": "\t".join(ts.item(0, c).text() if ts.item(0, c) else ""
                                          for c in range(ts.columnCount()))}
        lw = rw.list_warnings
        lw.clearSelection()
        lw.item(0).setSelected(True)
        out["warnings"] = {"got": press_copy(lw), "want": lw.item(0).text()}
        rf = int(rw.lbl_header.textInteractionFlags())
        out["rings_label_selectable"] = bool(rf & int(Qt.TextInteractionFlag.TextSelectableByKeyboard))
        mw.close()
    elif s == "layout_warnings":
        from tools import mps_ui_layout
        mw = make_main()
        w = mw.mps_window
        w.resize(1920, 1040)
        w.show()
        pump(0.6)
        out["at_1920"] = {"results_plots": w.splitter.sizes(), "map_tabs": w.plot_splitter.sizes(),
                          "window": [w.width(), w.height()]}
        w.resize(1366, 728)
        pump(0.6)
        sp = {"results_plots": w.splitter.sizes(), "map_tabs": w.plot_splitter.sizes(),
              "map_layers": w.axon_map.splitter.sizes()}
        out["first"] = {"sizes": sp, "window": [w.width(), w.height()],
                        "handles": [w.splitter.handleWidth(), w.plot_splitter.handleWidth(),
                                    w.axon_map.splitter.handleWidth()],
                        "file_before_drag": os.path.exists(mps_ui_layout.layout_path())}
        # The user drags the three dividers: the sizes are kept.
        for sp_w, sizes in ((w.splitter, [700, 666]), (w.plot_splitter, [380, 300]),
                            (w.axon_map.splitter, [300, 500])):
            total = sum(sp_w.sizes())
            sizes = [int(sizes[0] * total / sum(sizes)), total - int(sizes[0] * total / sum(sizes))]
            sp_w.setSizes(sizes)
            sp_w.splitterMoved.emit(sizes[0], 1)
        pump(0.8)
        with open(mps_ui_layout.layout_path(), encoding="utf-8") as fh:
            out["stored"] = json.load(fh)
        out["dragged"] = {"results_plots": w.splitter.sizes(), "map_tabs": w.plot_splitter.sizes(),
                          "map_layers": w.axon_map.splitter.sizes()}
        # Warnings: the button, its count and preview, the .txt file; never overwritten.
        out["warnings"] = {"text": w.btn_warnings.text(), "tip": w.btn_warnings.toolTip(),
                           "count": w._warning_count, "first": w.warning_lines()[0],
                           "has_list": hasattr(w, "list_warnings")}
        target = os.path.join(out_dir, "axonA_warnings.txt")
        cell.save_queue = [target]
        w.btn_warnings.click()
        pump()
        first = open(target, encoding="utf-8").read()
        cell.save_queue = [target]
        n0 = len(cell.messages)
        w.btn_warnings.click()
        pump()
        second = os.path.join(out_dir, "axonA_warnings_2.txt")
        out["warnings_file"] = {"first": first, "unchanged": open(target, encoding="utf-8").read() == first,
                                "second_exists": os.path.exists(second),
                                "messages": [m["text"] for m in cell.messages[n0:]]}
        mw.close()
        pump(0.1)
        # Another window, next session: the dividers where the user left them.
        mw2 = make_main()
        w2 = mw2.mps_window
        w2.resize(1366, 728)
        w2.show()
        pump(0.6)
        out["reopened"] = {"results_plots": w2.splitter.sizes(), "map_tabs": w2.plot_splitter.sizes(),
                           "map_layers": w2.axon_map.splitter.sizes()}
        mw2.close()
    elif s == "main_params":
        mw = make_main()
        w = mw.mps_window

        def main_edit(edit: Any, text: str) -> None:
            edit.setText(text)
            edit.editingFinished.emit()
            pump(0.02)

        out["editable"] = [not mw.ui.lineEdit_eps.isReadOnly(), not mw.ui.lineEdit_minsamples.isReadOnly(),
                           mw.ui.lineEdit_zmin.isReadOnly()]
        out["change_button"] = hasattr(mw, "button_change_params")
        n0 = len(calls("analyze_axon"))
        cell.window_edit(app, w.spin_eps, 31.0)                 # the strip
        strip_kw = calls("analyze_axon")[n0:]
        out["strip"] = {"field_eps": mw.ui.lineEdit_eps.text(), "kwargs": [sorted(c["kwargs"]) for c in strip_kw]}
        n1 = len(calls("analyze_axon"))
        main_edit(mw.ui.lineEdit_eps, "32")                     # the main window
        main_kw = calls("analyze_axon")[n1:]
        out["main"] = {"kwargs": [sorted(c["kwargs"]) for c in main_kw],
                       "eps_passed": [c["kwargs"].get("eps_nm") for c in main_kw],
                       "spin": float(w.spin_eps.value()), "settings": float(mw.mps_settings.eps_nm),
                       "analysis_eps": float(w.analysis.eps_nm)}
        n2 = len(calls("analyze_axon"))
        main_edit(mw.ui.lineEdit_minsamples, "8")
        out["main_min"] = {"runs": len(calls("analyze_axon")) - n2, "spin": int(w.spin_min.value()),
                           "settings": int(mw.mps_settings.min_samples)}
        n3, m3 = len(calls("analyze_axon")), len(cell.messages)
        main_edit(mw.ui.lineEdit_eps, "abc")
        main_edit(mw.ui.lineEdit_minsamples, "8.5")
        main_edit(mw.ui.lineEdit_eps, "32")                     # unchanged: nothing runs
        out["invalid"] = {"runs": len(calls("analyze_axon")) - n3,
                          "messages": [m["title"] for m in cell.messages[m3:]],
                          "field_eps": mw.ui.lineEdit_eps.text(), "field_min": mw.ui.lineEdit_minsamples.text(),
                          "settings": [float(mw.mps_settings.eps_nm), int(mw.mps_settings.min_samples)]}
        # The merged button: current -> shows the window; not current -> runs.
        w.hide()
        n4 = len(calls("analyze_axon"))
        mw.ui.pushButton_remove_bad_cluster.click()
        pump()
        out["button_current"] = {"runs": len(calls("analyze_axon")) - n4, "visible": w.isVisible()}
        cell.drag_roi(mw, 40.0)
        pump()
        n5 = len(calls("analyze_axon"))
        mw.ui.pushButton_remove_bad_cluster.click()
        pump()
        out["button_stale"] = {"runs": len(calls("analyze_axon")) - n5,
                               "current": mw._current_cluster_centroids() is not None}
        mw.close()
        mw0 = make_main(load=False)
        m6 = len(cell.messages)
        mw0.ui.pushButton_remove_bad_cluster.click()
        pump()
        out["button_no_file"] = {"window": mw0.mps_window is not None and mw0.mps_window.isVisible(),
                                 "messages": [m["title"] for m in cell.messages[m6:]],
                                 "show_button_visible": mw0.button_show_mps_window.isVisibleTo(mw0)}
        mw0.close()
    else:
        raise SystemExit(f"unknown scenario {s}")
    out["slot_exceptions"] = exc
    out["harness_errors"] = cell.errors
    return out


def child(argv: List[str]) -> int:
    scenario, out_dir, inputs = argv
    try:
        out = run_scenario(scenario, out_dir, inputs)
        out["ok"] = True
    except Exception:  # noqa: BLE001 - reported to the parent
        out = {"scenario": scenario, "ok": False, "error": traceback.format_exc()}
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    return 0


# ============================================================================ the checks (parent)
PASSED = 0
FAILED = 0


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-4:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 100)
    print("MANUAL TEST 2: the fixes and simplifications (simulated axon A, offscreen)")
    print("=" * 100)
    work = tempfile.mkdtemp(prefix="test_mt2_")
    import golden_ui_stage2 as G
    inputs = os.path.join(work, "inputs")
    described = G.make_inputs(inputs)
    with open(os.path.join(inputs, "inputs.json"), "w", encoding="utf-8") as f:
        json.dump({"axons": described}, f, default=str)
    workers = int(os.environ.get("MPS_TEST_WORKERS", "6"))
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    scenarios = [s for s in SCENARIOS if not only or s in only]

    def one(s: str) -> Dict[str, Any]:
        out_dir = os.path.join(work, s)
        os.makedirs(out_dir, exist_ok=True)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--scenario", s, out_dir, inputs],
                              capture_output=True, text=True, timeout=900)
        path = os.path.join(out_dir, "result.json")
        if not os.path.exists(path):
            return {"scenario": s, "ok": False, "error": (proc.stderr or proc.stdout)[-3000:]}
        with open(path, encoding="utf-8") as f:
            return dict(json.load(f))

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 12))) as pool:
        results = {r["scenario"]: r for r in pool.map(one, scenarios)}

    def res(s: str) -> Dict[str, Any]:
        r = results[s]
        if not r.get("ok"):
            raise RuntimeError(f"{s} did not run:\n{r.get('error')}")
        assert not r.get("slot_exceptions"), r["slot_exceptions"][-1][-800:]
        assert not r.get("harness_errors"), r["harness_errors"]
        return r

    def t1() -> str:
        r = res("figure_export")["figure_export"]
        for key in ("no_extension", "unknown_extension"):
            a = r[key]
            assert not a["raised"] and a["exists"] and a["written"].endswith(".png"), (key, a)
        for key in ("png", "svg"):
            assert not r[key]["raised"] and r[key]["exists"], (key, r[key])
        assert "no folder" in r["no_folder"]["raised"], r["no_folder"]
        for ext in ("png", "svg"):
            a = r[f"read_only_{ext}"]
            assert "read-only" in a["raised"] and a["size_after"] == 1, (ext, a)
        assert r["window_read_only"] == ["Export failed"], r["window_read_only"]
        wn = r["window_noext"]
        assert wn["titles"] == ["Image written"] and wn["exists"], wn
        assert "window_noext.png" in wn["texts"][0], wn["texts"]
        return "no / unknown extension -> .png; read-only (PNG, SVG), no folder -> the reason; window: Export failed"

    if "figure_export" in scenarios:
        check("1: Export image never says 'Image written' for a file that is not there", t1)

    def t2() -> str:
        r = res("typed_fields")
        seen = []
        for key in ("start", "half_70", "component", "automatic", "ticked"):
            st = r[key]
            assert st["fields"] == st["cut"] == st["slab"], (key, st)
            seen.append(f"{key} {st['fields'][0]}..{st['fields'][1]}")
        assert r["half_70"]["fields"] != r["start"]["fields"] and r["component"]["fields"] != r["half_70"]["fields"]
        return "; ".join(seen)

    if "typed_fields" in scenarios:
        check("2: the typed Z range fields always show the current slab", t2)

    def t3() -> str:
        r = res("copy")
        assert r["table"]["got"] == r["table"]["want"] and "\t" in r["table"]["got"], r["table"]
        assert "Copy" in r["table"]["menu"] and r["table_menu"] == r["table"]["want"], (r["table"], r["table_menu"])
        assert r["details"]["got"] == r["details"]["want"], r["details"]
        assert r["rings"]["got"] == r["rings"]["want"], r["rings"]
        assert r["warnings"]["got"] == r["warnings"]["want"], r["warnings"]
        assert r["label_selectable"] and r["rings_label_selectable"], r
        return "results table 2x2 -> " + repr(r["table"]["got"][:60])

    if "copy" in scenarios:
        check("3: tables, warnings, Details and labels copy their text", t3)

    def t4() -> str:
        r = res("layout_warnings")
        f = r["first"]
        big = r["at_1920"]
        rp, mt = big["results_plots"], big["map_tabs"]
        assert big["window"][0] == 1920 and f["window"][0] == 1366, (big, f)
        assert rp[0] / sum(rp) <= 0.27 and mt[0] / sum(mt) >= 0.75, big
        rp1, mt1 = f["sizes"]["results_plots"], f["sizes"]["map_tabs"]
        assert rp1[0] / sum(rp1) <= 0.34 and mt1[0] / sum(mt1) >= 0.6, f
        assert min(f["handles"]) >= 6 and not f["file_before_drag"], f
        st = r["stored"]
        assert set(st) == {"mps_window.results_plots", "mps_window.map_tabs", "mps_window.map_layers"}, st
        for key in ("results_plots", "map_tabs", "map_layers"):
            a, b = r["dragged"][key], r["reopened"][key]
            assert all(abs(x - y) <= 4 for x, y in zip(a, b)), (key, a, b)
        return (f"map {mt[0]}/{sum(mt)} px of the plot column, table {rp[0]}/{sum(rp)} px; "
                f"dragged sizes back next session: {r['reopened']}")

    def t5() -> str:
        r = res("layout_warnings")
        w = r["warnings"]
        assert w["count"] > 0 and w["text"] == f"Warnings ({w['count']})...", w
        assert w["first"][:40] in w["tip"] and ".txt" in w["tip"] and not w["has_list"], w
        f = r["warnings_file"]
        assert w["first"] in f["first"] and f["unchanged"] and f["second_exists"], f
        assert "already existed" in f["messages"][0], f["messages"]
        return f"{w['text']}; the second save went to _2, the first file unchanged"

    if "layout_warnings" in scenarios:
        check("4: the map has the larger share; the three dividers are splitters kept across sessions", t4)
        check("5: Warnings (N)... previews them and writes a .txt, never over an existing file", t5)

    def t6() -> str:
        r = res("main_params")
        assert r["button_current"] == {"runs": 0, "visible": True}, r["button_current"]
        assert r["button_stale"]["runs"] == 1 and r["button_stale"]["current"], r["button_stale"]
        b = r["button_no_file"]
        assert b["window"] and b["messages"] == [] and not b["show_button_visible"], b
        return "current: window shown, nothing run; ROI moved: one run; no file: the empty window"

    def t7() -> str:
        r = res("main_params")
        assert r["editable"] == [True, True, True] and not r["change_button"], r
        assert r["strip"]["field_eps"] == "31", r["strip"]
        m = r["main"]
        assert m["eps_passed"] == [{"float": "32.0"}] and m["spin"] == m["settings"] == m["analysis_eps"] == 32.0, m
        assert m["kwargs"] == r["strip"]["kwargs"], (m["kwargs"], r["strip"]["kwargs"])
        assert r["main_min"] == {"runs": 1, "spin": 8, "settings": 8}, r["main_min"]
        i = r["invalid"]
        assert i["runs"] == 0 and i["messages"] == ["DBSCAN parameters"] * 2, i
        assert i["field_eps"] == "32" and i["field_min"] == "8" and i["settings"] == [32.0, 8], i
        return "main box 32 -> one re-run with the strip's keyword arguments; strip 31 -> box '31'; junk refused"

    if "main_params" in scenarios:
        check("6: one 'MPS analysis' button: runs when no analysis is current, else shows the window", t6)
        check("7: eps / min samples edited in the main window = the strip's value and re-run (rule K)", t7)

    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - t0:.1f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scenario":
        sys.exit(child(sys.argv[2:]))
    sys.exit(main())
