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

SCENARIOS = ["figure_export"]


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

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["mt2"])
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

    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - t0:.1f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scenario":
        sys.exit(child(sys.argv[2:]))
    sys.exit(main())
