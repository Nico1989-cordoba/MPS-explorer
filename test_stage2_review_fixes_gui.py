# -*- coding: utf-8 -*-
"""
Regression tests for the code findings of the stage-2 review (UI stage 2, feat/ui-condense; REVIEW 1).

Each scenario drives the real main window offscreen on SIMULATED axon A (``golden_ui_stage2.make_inputs``) through a
path the golden does not walk, and checks the behaviour the review asked for. Every scenario runs in its own Python
process (some replace a function to force a failure) and writes every file under its own temporary folder (settings,
review store, selection log, exports).

  B-1  typed range applied and analysed, then unticked, then any strip edit: the re-run uses the automatic slab, not
       the typed range's centre, and the analysis is never reported current on another slab than the cut's;
  B-2  a new channel-1 file, then the MPS analysis window reopened: it shows no analysis of the previous file and
       "Save distances..." refuses;
  M-1  "typed" ticked and unticked without Apply: the cut, the Axoplasm panel's hand-set threshold and the discard
       comparison are untouched;
  M-2  a typed range set in the strip while the Axoplasm panel is open: the banner does not promise a refusal that
       does not happen, and "Export axon" warns that the selection on screen is not the analysed one;
  M-3  Nearest neighbours, range "from 900 to 800" while typing: no exception, the title says why nothing is drawn;
  m-1  a failed re-run (eps) without a slab change: the banner says so; the Rings window's banner is filled at once;
  m-2  the window without an analysis: editing a value stores it and runs nothing; a typed range applied while the
       half-width box has the focus does not re-run the analysis;
  m-3  the axial view's channel-2 flag follows the row actually drawn (view switch, stale analysis);
  m-4  another localization source or centre colouring in the map's header ticks its rows;
  m-5  a failed discard comparison followed by a successful one: the failure is forgotten;
  m-6  "Show the current selection" after a typed range and an ROI move draws the mixture of the current ROI;
  m-7  a new channel-1 file drops the contour drawn for the previous one, and says so.

R8: simulated data only.

Run:  venv\\Scripts\\python.exe test_stage2_review_fixes_gui.py      (offscreen; about 1 min)

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

SCENARIOS = ["typed_untick_rerun", "reopen_after_new_file", "typed_toggle", "typed_export_strip",
             "nn_range_inverted", "rerun_failure_eps", "b10_no_file", "b10_no_cluster", "typed_focus_half",
             "ch2_view_switch", "ch2_stale", "source_switch", "discard_failed_sticky", "typed_drag_show_current",
             "guide_new_file"]


# ============================================================================ one scenario (child process)
def run_scenario(scenario: str, out_dir: str, inputs: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(out_dir, "selection_log")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from PyQt5 import QtWidgets

    import golden_ui_stage2 as G

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["review_fixes"])
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
        assert mw.load_channel2(os.path.join(inputs, "sim_axon_B.hdf5"), 0)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="review",
                                   roi_name="roi", axon_name="axonA")
        mw.identity_path = mw._identity_source()
        mw.scatterplot()
        if roi:
            cell.place_roi(mw, tuple(described["centre_nm"]), described["ring_radius_nm"] + 450.0)
        if cluster:
            mw.ui.pushButton_clusterch1.click()
            pump()
        return mw

    def banner(w: Any) -> Dict[str, Any]:
        return {"visible": w.banner.isVisibleTo(w), "text": w.banner.text(),
                "run": w.banner.btn_run.isVisibleTo(w)}

    s = scenario
    if s == "typed_untick_rerun":
        mw = make_main()
        w = mw.mps_window
        lo = float(mw.zmin) + 10.0
        hi = lo + 100.0                                  # an asymmetric typed range
        w.chk_typed.setChecked(True)
        w.edit_typed_min.setText(f"{lo:.1f}")
        w.edit_typed_max.setText(f"{hi:.1f}")
        w.btn_typed_apply.setFocus()
        w._on_typed_apply()
        pump()
        w.banner.btn_run.click()                         # "Run the MPS analysis"
        pump()
        out["typed_run"] = {"source": w.analysis.slab_source}
        w.chk_typed.setFocus()
        w.chk_typed.setChecked(False)                    # back to the automatic slab (no re-run)
        pump()
        out["unticked"] = {"stale": w._stale, "current": mw._current_cluster_centroids() is not None}
        n0 = len(calls("analyze_axon"))
        w.spin_eps.setFocus()
        cell.window_edit(app, w.spin_eps, 26.0)          # any strip edit re-runs
        pump()
        a2 = w.analysis
        kw = calls("analyze_axon")[n0:]
        out["after_eps"] = {"peak_kwargs": [c["kwargs"].get("main_peak_override_nm") for c in kw],
                            "slab_override": ["slab_override" in c["kwargs"] for c in kw],
                            "cut": [mw.zmin, mw.zmax], "analysis_slab": [a2.slab_zmin_nm, a2.slab_zmax_nm],
                            "source": a2.slab_source, "stale": w._stale}
    elif s == "reopen_after_new_file":
        mw = make_main()
        w = mw.mps_window
        a_old = w.analysis
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_B.hdf5"), 0)
        pump()
        mw.button_change_params.click()
        pump()
        w2 = mw.mps_window
        out["reopened"] = {"shows_old_analysis": w2.analysis is a_old, "analysis_none": w2.analysis is None}
        target = os.path.join(out_dir, "saved_distances.csv")
        cell.save_queue = [target]
        n_msg = len(cell.messages)
        res = w2.nn.on_save()
        cell.save_queue = []
        out["save"] = {"returned": res, "written": os.path.exists(target), "messages": cell.messages[n_msg:]}
        # The old window object may have been kept: whatever is shown, "Save distances..." on a window that
        # still holds the old analysis must refuse too.
        if w2.analysis is not None:
            out["save_old"] = True
    elif s == "typed_toggle":
        mw = make_main()
        cell.axoplasm_with_images(app, mw)
        aw = mw.axoplasm_window
        w = mw.mps_window
        thr = float(aw.spin_threshold.value())
        aw._threshold_typed(thr * 1.05)
        aw.flush()
        pump(0.1)
        before = {"manual_threshold": aw._manual_threshold, "note": aw.selection_note,
                  "roi_indices_id": id(mw.roi_indices), "cut": [mw.zmin, mw.zmax]}
        n_cd = len(calls("compare_discard"))
        w.chk_typed.setChecked(True)
        pump()
        w.chk_typed.setChecked(False)
        pump(0.1)
        aw.flush()
        pump(0.1)
        out["before"] = {k: v for k, v in before.items() if k != "roi_indices_id"}
        out["after"] = {"manual_threshold": aw._manual_threshold, "note": aw.selection_note,
                        "new_compare_discard_calls": len(calls("compare_discard")) - n_cd,
                        "roi_indices_replaced": id(mw.roi_indices) != before["roi_indices_id"],
                        "cut": [mw.zmin, mw.zmax], "typed": mw._z_range_user_edited, "stale": w._stale}
    elif s == "typed_export_strip":
        mw = make_main()
        cell.axoplasm_with_images(app, mw)
        w = mw.mps_window
        zmin, zmax = float(mw.zmin), float(mw.zmax)
        w.chk_typed.setChecked(True)
        w.edit_typed_min.setText(f"{zmin + 20:.1f}")
        w.edit_typed_max.setText(f"{zmax - 20:.1f}")
        w.btn_typed_apply.setFocus()
        w._on_typed_apply()
        pump(0.1)
        mw.axoplasm_window.flush()
        pump(0.1)
        out["state"] = {"stale": w._stale, "banner": banner(w)["text"]}
        exp = cell.export_axon(mw)
        out["export"] = {"returned": exp.get("returned"), "messages": exp.get("messages")}
    elif s == "nn_range_inverted":
        mw = make_main()
        w = mw.mps_window
        nn = w.nn
        w.tabs.setCurrentWidget(nn)
        pump()
        nn.radio_range_set.setChecked(True)          # "from 0 to 800 nm"
        pump()
        for v in (9.0, 90.0, 900.0):                 # the lower bound typed first, key by key
            nn.spin_range_min.setValue(v)
            pump(0.02)
        out["after_typing_900"] = {"range": nn.value_range(), "exceptions": len(exc),
                                   "last_exception": exc[-1].splitlines()[-1] if exc else None,
                                   "title": nn.nn_title()}
        nn.spin_range_max.setValue(1500.0)
        pump(0.02)
        out["after_1500"] = {"exceptions": len(exc), "title": nn.nn_title(),
                             "bars": "nn_bars" in nn.layers.keys()}
    elif s == "rerun_failure_eps":
        import MPS_explorer
        real = MPS_explorer.analyze_axon

        def failing(*a: Any, **k: Any) -> Any:
            if float(k.get("eps_nm", 0)) == 31.0:
                raise RuntimeError("forced failure at eps 31")
            return real(*a, **k)
        MPS_explorer.analyze_axon = failing
        mw = make_main()
        w = mw.mps_window
        cell.window_edit(app, w.spin_eps, 31.0)
        pump()
        out["after_failed_rerun"] = {"analysis_eps": float(w.analysis.eps_nm), "banner": banner(w)}
        w.btn_rings.click()
        pump(0.1)
        rw = mw.rings_window
        out["rings"] = None if rw is None else {"banner": rw.banner.text()}
    elif s in ("b10_no_file", "b10_no_cluster"):
        mw = make_main(load=(s == "b10_no_cluster"), cluster=False)
        w = mw.show_mps_window()
        pump()
        n_msg = len(cell.messages)
        n_calls = len(calls("analyze_axon"))
        w.spin_eps.setValue(30.0)
        w.spin_eps.editingFinished.emit()
        pump(0.1)
        w.spin_eps.editingFinished.emit()            # focus leaves the box again
        pump(0.1)
        out["after_eps_edit"] = {"messages": cell.messages[n_msg:],
                                 "analyze_axon_calls": len(calls("analyze_axon")) - n_calls,
                                 "window_analysis_none": w.analysis is None,
                                 "settings_eps": float(mw.mps_settings.eps_nm), "banner": banner(w)}
    elif s == "typed_focus_half":
        mw = make_main()
        w = mw.mps_window
        w.show()
        w.raise_()
        w.activateWindow()
        pump()
        zmin, zmax = float(mw.zmin), float(mw.zmax)
        n0 = len(calls("analyze_axon"))
        w.chk_typed.setChecked(True)
        w.edit_typed_min.setText(f"{zmin + 20:.1f}")
        w.edit_typed_max.setText(f"{zmax - 20:.1f}")
        w.spin_half.setFocus()
        pump()
        out["focus_before_apply"] = type(QtWidgets.QApplication.focusWidget()).__name__ \
            if QtWidgets.QApplication.focusWidget() is not None else None
        w._on_typed_apply()                           # what a click on Apply runs
        pump()
        out["after_apply"] = {"analyze_axon_calls": len(calls("analyze_axon")) - n0, "stale": w._stale,
                              "cut": [mw.zmin, mw.zmax]}
    elif s in ("ch2_view_switch", "ch2_stale"):
        mw = make_main()
        w = mw.mps_window
        ax = w.axial
        w.tabs.setCurrentWidget(ax)
        pump()
        ax.layers.set_visible("ch2_roi", True)
        pump()
        out["ch2_ticked"] = {"ch2_shown": ax.inputs().ch2_shown,
                             "components_enabled": ax.layer("components").enabled}
        if s == "ch2_view_switch":
            ax.combo_view.setCurrentIndex(ax.combo_view.findData("segments"))
            pump()
            ax.combo_view.setCurrentIndex(ax.combo_view.findData("slab"))
            pump()
        else:
            cell.drag_roi(mw, 150.0)
            pump()
        layer = ax.layer("components")
        out["after"] = {"ch2_shown_flag": ax.inputs().ch2_shown,
                        "ch2_row_shown": "ch2_roi" in ax.layers.shown_keys(),
                        "ch2_row_enabled": ax.layers.checkbox("ch2_roi").isEnabled(),
                        "components_enabled": layer.enabled, "components_reason": layer.reason,
                        "stale": w._stale}
    elif s == "source_switch":
        mw = make_main()
        cell.axoplasm_with_images(app, mw)
        w = mw.mps_window
        m = w.axon_map
        m.combo_source.setCurrentIndex(m.combo_source.findData("selection"))
        pump()
        out["after_source_selection"] = {"view": m.view(), "shown": m.layers.shown_keys(), "keys": m.layers.keys()}
        m.combo_colour.setCurrentIndex(m.combo_colour.findData("images"))
        pump()
        out["after_colour_images"] = {"view": m.view(), "shown": m.layers.shown_keys(), "keys": m.layers.keys()}
    elif s == "discard_failed_sticky":
        from tools import mps_analysis
        real_cd = mps_analysis.compare_discard
        state = {"n": 0}

        def flaky(*a: Any, **k: Any) -> Any:
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("forced failure")
            return real_cd(*a, **k)
        mps_analysis.compare_discard = flaky
        mw = make_main()
        cell.axoplasm_with_images(app, mw)
        w = mw.mps_window
        out["after_first"] = {"calls": state["n"], "discard_failed": mw._discard_failed[1]}
        aw = mw.axoplasm_window
        aw._use_otsu()
        aw.flush()
        pump(0.2)
        w.refresh()
        cap = w.axon_map.layers.group_caption("contour") if "contour" in w.axon_map.layers.groups() else None
        out["after_second"] = {"calls": state["n"], "discard_failed": mw._discard_failed[1],
                               "comparison": w.comparison is not None, "contour_caption": cap}
    elif s == "typed_drag_show_current":
        mw = make_main()
        w = mw.mps_window
        zmin, zmax = float(mw.zmin), float(mw.zmax)
        w.chk_typed.setChecked(True)
        w.edit_typed_min.setText(f"{zmin + 20:.1f}")
        w.edit_typed_max.setText(f"{zmax - 20:.1f}")
        w.btn_typed_apply.setFocus()
        w._on_typed_apply()
        pump()
        cell.drag_roi(mw, 150.0)
        pump()
        out["after_drag"] = {"stale": w._stale}
        w.banner.btn_show.click()
        pump()
        inp = w.axial.inputs()
        out["show_current"] = {"show_current": w._show_current, "fit_none": inp.fit is None,
                               "fit_reason": inp.fit_reason}
    elif s == "guide_new_file":
        mw = make_main()
        w = mw.mps_window
        w.btn_contour.click()                  # the harness's contour editor: Apply with a fixed path
        pump()
        out["after_draw"] = {"guide": mw._guide is not None}
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_B.hdf5"), 0)
        pump()
        out["after_load"] = {"guide": mw._guide is not None, "note": mw._guide_note}
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
    print("STAGE-2 REVIEW FIXES: the paths the golden does not walk (simulated axon A, offscreen)")
    print("=" * 100)
    work = tempfile.mkdtemp(prefix="test_stage2_review_fixes_")
    import golden_ui_stage2 as G
    inputs = os.path.join(work, "inputs")
    described = G.make_inputs(inputs)
    with open(os.path.join(inputs, "inputs.json"), "w", encoding="utf-8") as f:
        json.dump({"axons": described}, f, default=str)
    workers = int(os.environ.get("MPS_TEST_WORKERS", "6"))

    def one(s: str) -> Dict[str, Any]:
        out_dir = os.path.join(work, s)
        os.makedirs(out_dir, exist_ok=True)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--scenario", s, out_dir, inputs],
                              capture_output=True, text=True, timeout=600)
        path = os.path.join(out_dir, "result.json")
        if not os.path.exists(path):
            return {"scenario": s, "ok": False, "error": (proc.stderr or proc.stdout)[-3000:]}
        with open(path, encoding="utf-8") as f:
            return dict(json.load(f))

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 12))) as pool:
        results = {r["scenario"]: r for r in pool.map(one, SCENARIOS)}

    def res(s: str) -> Dict[str, Any]:
        r = results[s]
        if not r.get("ok"):
            raise RuntimeError(f"{s} did not run:\n{r.get('error')}")
        assert not r.get("slot_exceptions"), r["slot_exceptions"][-1][-800:]
        assert not r.get("harness_errors"), r["harness_errors"]
        return r

    def b1() -> str:
        r = res("typed_untick_rerun")
        assert r["typed_run"]["source"] == "range typed", r["typed_run"]
        assert r["unticked"]["stale"] and not r["unticked"]["current"], r["unticked"]
        a = r["after_eps"]
        assert a["peak_kwargs"] == [None], a
        assert a["slab_override"] == [False], a
        assert a["source"] == "automatic", a
        assert not a["stale"], a
        cut, slab = a["cut"], a["analysis_slab"]
        assert abs(cut[0] - slab[0]) < 0.1 and abs(cut[1] - slab[1]) < 0.1, (cut, slab)
        return f"re-run after unticking: automatic slab {slab[0]:.1f}..{slab[1]:.1f} = cut {cut}"

    def b2() -> str:
        r = res("reopen_after_new_file")
        assert not r["reopened"]["shows_old_analysis"] and r["reopened"]["analysis_none"], r["reopened"]
        assert r["save"]["returned"] is None and not r["save"]["written"], r["save"]
        return "the reopened window shows no analysis of the previous file; nothing saved"

    def m1() -> str:
        r = res("typed_toggle")
        b, a = r["before"], r["after"]
        assert b["manual_threshold"] is not None
        assert a["manual_threshold"] == b["manual_threshold"], (b, a)
        assert a["note"] == b["note"], (b["note"], a["note"])
        assert not a["roi_indices_replaced"] and a["new_compare_discard_calls"] == 0, a
        assert a["cut"] == b["cut"] and not a["typed"] and not a["stale"], a
        return f"threshold {a['manual_threshold']:.1f} kept; cut {a['cut']} untouched; no new comparison"

    def m2() -> str:
        r = res("typed_export_strip")
        st = r["state"]
        assert st["stale"], st
        assert "it refuses while" not in st["banner"], st["banner"]
        assert "run the analysis first" in st["banner"].lower(), st["banner"]
        texts = " ".join(m["text"] for m in r["export"]["messages"])
        assert "not the one on screen any more" in texts, texts
        return "banner: " + st["banner"].splitlines()[-1]

    def m3() -> str:
        r = res("nn_range_inverted")
        a = r["after_typing_900"]
        assert a["exceptions"] == 0, a
        assert "upper bound" in a["title"], a["title"]
        b = r["after_1500"]
        assert b["exceptions"] == 0 and b["bars"] and "upper bound" not in b["title"], b
        return f"900..800: '{a['title'][-70:]}'; 900..1500 drawn"

    def mi1() -> str:
        r = res("rerun_failure_eps")
        a = r["after_failed_rerun"]
        assert a["analysis_eps"] == 25.0, a
        assert a["banner"]["visible"] and "could not be re-run" in a["banner"]["text"], a["banner"]
        assert r["rings"] is not None and "eps 31" in r["rings"]["banner"], r["rings"]
        return "banner: " + a["banner"]["text"].splitlines()[0][:80]

    def mi2() -> str:
        out = []
        for s in ("b10_no_file", "b10_no_cluster"):
            a = res(s)["after_eps_edit"]
            assert a["messages"] == [], (s, a["messages"])
            assert a["analyze_axon_calls"] == 0 and a["window_analysis_none"], (s, a)
            out.append(f"{s}: 0 runs, 0 dialogs")
        a = res("b10_no_cluster")["after_eps_edit"]
        assert a["settings_eps"] == 30.0 and a["banner"]["run"], a
        r = res("typed_focus_half")
        assert r["focus_before_apply"] == "QDoubleSpinBox", r["focus_before_apply"]
        assert r["after_apply"]["analyze_axon_calls"] == 0 and r["after_apply"]["stale"], r["after_apply"]
        out.append("typed range with the half-width focused: no re-run, banner")
        return "; ".join(out)

    def mi3() -> str:
        for s in ("ch2_view_switch", "ch2_stale"):
            r = res(s)
            assert r["ch2_ticked"]["ch2_shown"] and not r["ch2_ticked"]["components_enabled"], (s, r["ch2_ticked"])
            a = r["after"]
            drawn = a["ch2_row_shown"] and a["ch2_row_enabled"]
            assert not drawn, (s, a)
            assert a["components_enabled"], (s, a)
        return "components available again once channel 2 is not drawn (view switch, stale analysis)"

    def mi4() -> str:
        r = res("source_switch")
        a = r["after_source_selection"]
        sel = [k for k in a["keys"] if k.startswith("sel_")]
        assert sel and all(k in a["shown"] for k in sel), a
        b = r["after_colour_images"]
        classes = [k for k in b["keys"] if k in ("c_both", "c_tubulin", "c_spectrin", "c_neither")]
        assert classes and all(k in b["shown"] for k in classes), b
        return f"source -> {sel} ticked; colouring -> {classes} ticked"

    def mi5() -> str:
        r = res("discard_failed_sticky")
        assert r["after_first"]["discard_failed"], r["after_first"]
        a = r["after_second"]
        assert a["comparison"] and a["discard_failed"] == "", a
        assert "No discard comparison" not in (a["contour_caption"] or ""), a["contour_caption"]
        return "the failure is forgotten once a comparison exists"

    def mi6() -> str:
        r = res("typed_drag_show_current")
        assert r["after_drag"]["stale"]
        a = r["show_current"]
        assert a["show_current"] and not a["fit_none"], a
        return "the current ROI's mixture is drawn"

    def mi7() -> str:
        r = res("guide_new_file")
        assert r["after_draw"]["guide"], r
        a = r["after_load"]
        assert not a["guide"] and "not applied" in a["note"], a
        return a["note"]

    check("B-1: unticking a typed range, then a strip edit, re-runs on the automatic slab", b1)
    check("B-2: a new file empties the MPS analysis window; Save distances refuses", b2)
    check("M-1: ticking and unticking 'typed' without Apply changes nothing", m1)
    check("M-2: a typed range in the strip: the banner is true and the export warns", m2)
    check("M-3: an inverted 1NN range raises nothing and says why nothing is drawn", m3)
    check("m-1: a failed re-run says so; the Rings banner is filled at first open", mi1)
    check("m-2: no analysis: an edit stores and runs nothing; Apply with the half-width focused", mi2)
    check("m-3: the axial channel-2 flag follows the row drawn", mi3)
    check("m-4: another source or colouring ticks its rows", mi4)
    check("m-5: a failed discard comparison is forgotten after a successful one", mi5)
    check("m-6: Show the current selection after a typed range and an ROI move fits the mixture", mi6)
    check("m-7: a new file drops the drawn contour and says so", mi7)

    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - t0:.1f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scenario":
        sys.exit(child(sys.argv[2:]))
    sys.exit(main())
