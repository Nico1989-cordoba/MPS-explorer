# -*- coding: utf-8 -*-
"""
Offscreen test of the H5-D review window (``tools/mps_columns_window.py``; DECISIONES D-35(b), acceptance item
D-35(d)(5)): "marcar, seleccionar uno/todos, sacar a mano, devolver, correr de nuevo sin congelar la ventana,
procedencia y etiqueta de edición posterior". Written BEFORE the window (tests first): the window's names are
imported lazily, so a missing name fails its own checks and never crashes the run.

The axon is SIMULATED (D-35e: the interface is tested on simulated axons only): tools.mps_simulate_axon with the
synthetic harness's slot-shaped case (tools.mps_sim_harness: its nuisance configuration, background inside the pick,
the leak on) plus synthetic lumen clusters (tools.mps_sim_harness.simulate_and_build, objects "syn"), through
build_rings. Its classification uses an EMULATED widefield (tools.mps_lumen.widefield_flags_from_depths): the
clusters whose localizations are mostly lumen get +600 nm in both images (WF250 with a usable interior -> REMOVE), three
membrane clusters +100 nm (WF0 only -> DOUBTFUL), every other cluster NaN (off the images); isolation is computed on
the axon. So the window opens with auto-removed, doubtful and kept clusters. A second window opens on the same axon
without any widefield (isolation only), which must say so.

WINDOW CONTRACT (the names this test fixes; the window follows them)
--------------------------------------------------------------------
tools/mps_columns_window.py (PyQt5 + pyqtgraph, like the other panels):
  DISPLAY_CLASSES = ("kept", "auto_removed", "doubtful", "manual_removed", "restored")
  LUMEN_CLASS_STYLE: Dict[display class, (role of tools.mps_plot_style.ROLES, alpha 0-255, pyqtgraph symbol)] -- the
      brush of a cluster's marker is the role's colour, its symbol the symbol; the table follows the conventions of
      validate_plot_colours.py (every pair of classes apart by colour under every dichromacy, or by symbol; the same
      grey -> another symbol; each class stands off the dark plot background). The ring is carried by something
      else (the pen, e.g. tools.mps_plot_style.segment_colour(ring index), or the size): two markers of one class in
      two rings differ in pen colour or size.
  ColumnsWindow(res, classification, *, decisions=None, lpz_nm=None, columns_params=None, n_null=None,
                widefield_images=None, identity=None, source_name="", parent=None)
      decisions               LumenDecisions (from the classification when not given)
      plot_widget             the pyqtgraph PlotWidget with the clusters in the axon frame (x', y'); a click on a
                              marker (ScatterPlotItem.sigClicked(item, points, ev); ev may be None) toggles that cluster
                              (any cluster: kept <-> removed by hand, auto-removed <-> restored, doubtful <-> removed)
      doubtful_table          QTableWidget: one row per DOUBTFUL cluster; column 0 checkable (checked = removed), its
                              Qt.ItemDataRole.UserRole data = the stable key; headers naming ring, hull depth, hop, tubulin depth,
                              spectrin depth, n_locs, n_events and |z' - ring centre|
      select_all_doubtful_button, clear_all_doubtful_button, run_button (English text naming the re-run), export_button
                              QPushButtons with tooltips
      progress_bar            visible while an analysis runs
      leak_banner             QLabel "... not calibrated for axial leak (H5-E) ..." shown with the results
      edited_label            QLabel "edited after results were shown", visible iff decisions.results_shown_before_edit
      widefield_warning       QLabel visible when the classification has no widefield (says the rule is isolation only)
      display_class(stable_key) -> str (DISPLAY_CLASSES); results_text() -> str (the results area as plain text);
      is_running() -> bool; last_result -> CleanedAnalyses | None;
      analysis_finished       pyqtSignal(object): emitted with the CleanedAnalyses once shown
      export_tables(path) -> List[str]: the decision table at ``path`` (LumenDecisions.to_rows + identity: axon_id) and
                              the results at <stem>_results.csv (axon_id, arcc_z_A, arcc_p_A, leak_calibrated,
                              edited_after_results_shown, lumen_n_removed_auto, lumen_n_removed_final, lumen_n_manual,
                              lumen_n_doubtful), with tools.results_table's rules: a table of another analysis is
                              refused (ValueError, the file untouched), never overwritten
  The run button runs tools.mps_lumen.run_cleaned_analyses in a WORKER THREAD (clean_rings + the analyses; build_rings
  is not re-run): the click returns at once, is_running() is True, the run button is disabled and the progress bar
  shown until the results are; a QTimer keeps ticking meanwhile.
  Added by the Fix stage (the reviews of H5-D, 2026-09-29): ColumnsWindow(..., review_store=LumenReviewStore | None)
  restores the axon's saved review (decisions, log, "results shown", the edited flag) and saves every change;
  load_decisions(path) -> clusters matched, refusing (ValueError, nothing changed) a table of another axon or cluster
  set and using the last of appended exports; highlight_item (the selected row's ring) never takes a click;
  MPS_explorer.open_columns_review brings an open review of the same selection forward instead of rebuilding it,
  builds the rings without the ROI edge, and keeps the reviews in columns_review_store_dir.

Checks are numbered as they print; each states its expected value and its source (D-35(b) / D-35(d)(5) / the
specification's ADDENDUM v2 section B).

Run:  venv\\Scripts\\python.exe test_columns_window_gui.py      (offscreen; ~1-2 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import importlib  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MW = "tools.mps_columns_window"
ML = "tools.mps_lumen"
DISPLAY = ("kept", "auto_removed", "doubtful", "manual_removed", "restored")
RUN_TIMEOUT_S = 240.0
TICK_MS = 20
SEED = 0

PASSED = 0
FAILED = 0
T0 = time.perf_counter()
WORK = tempfile.mkdtemp(prefix="test_columns_window_")


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def require(module: str, *names: str) -> Any:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{module} not importable ({type(exc).__name__}: {exc}); H5-D not implemented")
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        raise RuntimeError(f"{module} has no {', '.join(missing)} (H5-D not implemented)")
    got = tuple(getattr(mod, n) for n in names)
    return got[0] if len(got) == 1 else got


def need(st: Dict[str, Any], key: str) -> Any:
    if key not in st:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return st[key]


# ============================================================ the simulated axon and its classification
def simulated_axon(st: Dict[str, Any]) -> str:
    import power_columns as pc
    from tools import mps_sim_harness as vl
    seed = pc.replicate_seed(vl.SEED_BASE + 1_000_000 * SEED + 10_000 * 10, 0)
    sim = vl.simulate_and_build(vl.cell_spec("slot"), seed, leak=True, objects="syn")
    st["sim"] = sim
    st["res"] = sim["res"]
    st["truth"] = vl.truth_of(sim)
    st["sig_before"] = vl.ring_signature(sim["res"])
    return f"slot + synthetic lumen, leak on, seed {seed}: K {[len(r.clusters) for r in sim['res'].rings]}"


def classification(st: Dict[str, Any]) -> str:
    require(ML, "classify_lumen", "widefield_flags_from_depths")
    L = importlib.import_module(ML)
    res = need(st, "res")
    tr = st["truth"]
    n = len(tr["keys"])
    dt = np.full(n, np.nan)
    lumen = np.array([t == "lumen" for t in tr["truth"]], bool)
    dt[lumen] = 600.0
    mem = np.flatnonzero(~lumen)
    dt[mem[[3, 17, 31]]] = 100.0
    wf = L.widefield_flags_from_depths(res, dt, dt.copy(), interior_usable=True, registration="emulated (test)")
    c = L.classify_lumen(res, widefield=wf)
    cls = list(c.auto_classes)
    assert cls.count("removed") >= 2 and cls.count("doubtful") >= 3, (cls.count("removed"), cls.count("doubtful"))
    st["cls"] = c
    st["cls_nowf"] = L.classify_lumen(res)
    return f"{cls.count('removed')} auto-removed, {cls.count('doubtful')} doubtful, {cls.count('kept')} kept (emulated widefield)"


def all_pairs_viable(res: Any, xyz_lab_nm: Any = None) -> Any:
    """Test stub (H6, D-41): every consecutive ring pair VIABLE. The checks of this file are about the lumen review
    (D-35b), not about the viability rule v2, whose own offscreen test is test_zquality_gui.py; the simulated axon
    here need not have a viable pair under rule v2 (its leak is on), and without the stub the window would then rightly
    compute nothing."""
    import batch_columns as bc
    M = importlib.import_module(MW)
    rings = sorted(int(r.index) for r in res.rings)
    return M.ReviewViability(rule="test stub: every pair viable",
                             pairs=tuple(bc.PairVerdict(a, b, "viable") for a, b in zip(rings, rings[1:])))


# ============================================================ helpers on the drawn items
def scatter_items(window: Any) -> List[Any]:
    import pyqtgraph as pg
    plot_item = window.plot_widget.getPlotItem() if hasattr(window.plot_widget, "getPlotItem") else window.plot_widget
    out = []
    for item in list(plot_item.items):
        if isinstance(item, pg.ScatterPlotItem):
            out.append(item)
        elif getattr(item, "scatter", None) is not None and isinstance(item.scatter, pg.ScatterPlotItem):
            out.append(item.scatter)
    return out


def spots_at(window: Any, xy: np.ndarray, tol: float = 0.5) -> List[Tuple[Any, Any]]:
    """(scatter item, spot) of every drawn marker within ``tol`` nm of ``xy``."""
    found = []
    for sc in scatter_items(window):
        for sp in sc.points():
            p = sp.pos()
            if abs(float(p.x()) - float(xy[0])) <= tol and abs(float(p.y()) - float(xy[1])) <= tol:
                found.append((sc, sp))
    return found


def spot_style(sp: Any) -> Dict[str, Any]:
    return dict(symbol=str(sp.symbol()), brush=sp.brush().color().name().lower(), pen=sp.pen().color().name().lower(),
                size=float(sp.size()))


def class_style(cls: str) -> Tuple[str, str]:
    from tools.mps_plot_style import ROLES
    style = require(MW, "LUMEN_CLASS_STYLE")
    role, _alpha, symbol = style[cls]
    return str(symbol), ROLES[role].lower()


def centroid_of(st: Dict[str, Any], key: str) -> np.ndarray:
    c = st["cls"]
    i = list(c.stable_keys).index(key)
    out: np.ndarray = np.asarray(c.centroids_nm, float)[i]
    return out


def marker_matches(window: Any, st: Dict[str, Any], key: str) -> bool:
    """The cluster's marker shows its display class's (symbol, brush), and no other class's."""
    want = class_style(window.display_class(key))
    styles = [spot_style(sp) for _sc, sp in spots_at(window, centroid_of(st, key))]
    others = {class_style(c) for c in DISPLAY if c != window.display_class(key)}
    shown = {(s["symbol"], s["brush"]) for s in styles}
    return want in shown and not (shown & others)


def click(window: Any, st: Dict[str, Any], key: str) -> None:
    hits = spots_at(window, centroid_of(st, key))
    if not hits:
        raise AssertionError(f"no marker drawn at the centroid of {key}")
    sc, sp = hits[0]
    sc.sigClicked.emit(sc, [sp], None)


def table_row(window: Any, key: str) -> int:
    from PyQt5 import QtCore
    t = window.doubtful_table
    for r in range(t.rowCount()):
        it = t.item(r, 0)
        if it is not None and it.data(QtCore.Qt.ItemDataRole.UserRole) == key:
            return r
    return -1


def pump(app: Any, seconds: float = 0.05) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


def viewport_point(window: Any, xy: np.ndarray) -> Any:
    """The plot viewport's pixel of a data point (x', y') in nm."""
    from PyQt5 import QtCore
    vb = window.plot_widget.getViewBox()
    sp = vb.mapViewToScene(QtCore.QPointF(float(xy[0]), float(xy[1])))
    return window.plot_widget.mapFromScene(sp)


def real_click(app: Any, window: Any, xy: np.ndarray) -> None:
    """A real mouse click (QTest) on the plot at (x', y'), the mouse first moved onto it as a user does (hover
    events), unlike ``click``, which emits the scatter item's signal directly."""
    from PyQt5 import QtCore, QtTest
    qtest: Any = QtTest.QTest  # static helpers; the stubs declare them as instance methods
    vp = window.plot_widget.viewport()
    p = viewport_point(window, xy)
    assert vp.rect().contains(p), f"{xy} maps to {p.x()},{p.y()}, outside the viewport"
    for dx in (-30, -15, -5, 0):
        qtest.mouseMove(vp, QtCore.QPoint(p.x() + dx, p.y()))
        pump(app, 0.02)
    qtest.mouseClick(vp, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, p)
    pump(app, 0.05)


def run_and_wait(app: Any, window: Any, timeout: float = RUN_TIMEOUT_S) -> Any:
    """Start the window's re-run and wait for its result (None when it did not finish)."""
    got: Dict[str, Any] = {}
    window.analysis_finished.connect(lambda r: got.__setitem__("r", r))
    assert window.start_run(), "a run was already running"
    t_end = time.perf_counter() + timeout
    while window.is_running() and time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.01)
    pump(app, 0.1)
    return got.get("r")


# ============================================================ main
def main() -> int:
    print("=" * 100)
    print("COLUMNS WINDOW (H5-D) OFFSCREEN TEST: markers, click / checkbox / select all, restore, re-run in a worker, provenance")
    print("=" * 100)
    st: Dict[str, Any] = {}
    from PyQt5 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    # H6: the lumen-review checks below run with every pair viable (all_pairs_viable); rule v2 is tested elsewhere
    setattr(importlib.import_module(MW), "DEFAULT_VIABILITY_FN", all_pairs_viable)

    print("\n1. The simulated axon and its classification")
    check("a SIMULATED axon (tools.mps_simulate_axon + build_rings) with synthetic lumen clusters", lambda: simulated_axon(st))
    check("classified with an emulated widefield: auto-removed and doubtful clusters present (D-35a)", lambda: classification(st))

    print("\n2. The window opens; differential markers (D-35(d)(5): 'marcar')")

    def opens() -> str:
        Win = require(MW, "ColumnsWindow")
        w = Win(need(st, "res"), need(st, "cls"), n_null=49, source_name="simulated slot")
        w.show()
        pump(app, 0.3)
        st["w"] = w
        for name in ("plot_widget", "doubtful_table", "select_all_doubtful_button", "clear_all_doubtful_button", "run_button",
                     "export_button", "progress_bar", "leak_banner", "edited_label", "widefield_warning", "decisions",
                     "analysis_finished"):
            assert hasattr(w, name), f"ColumnsWindow has no {name}"
        assert not w.progress_bar.isVisible() and not w.edited_label.isVisible(), "nothing runs and nothing was edited yet"
        assert not w.widefield_warning.isVisible(), "this classification has a widefield"
        return f"{type(w).__name__} built offscreen; {len(scatter_items(w))} scatter item(s)"

    def style_table() -> str:
        import pyqtgraph as pg
        import validate_plot_colours as vpc
        from tools.mps_plot_style import DARK_BG, ROLES
        style = require(MW, "LUMEN_CLASS_STYLE")
        disp = tuple(require(MW, "DISPLAY_CLASSES"))
        assert disp == DISPLAY and set(style) == set(DISPLAY), (disp, list(style))
        symbols = set(pg.graphicsItems.ScatterPlotItem.Symbols.keys())
        close = []
        for c, (role, alpha, sym) in style.items():
            assert role in ROLES, f"{c}: {role} is not a role of tools.mps_plot_style"
            assert sym in symbols, f"{c}: {sym} is not a pyqtgraph symbol"
            for kind in vpc.SIMULATIONS:
                d = vpc.distance(vpc.blend(ROLES[role], int(alpha), DARK_BG), DARK_BG, kind)
                assert d >= 20.0, f"{c} vanishes on the dark background ({kind}: {d:.0f})"
        items = list(style.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, (ra, aa, sa)), (b, (rb, ab, sb)) = items[i], items[j]
                ca, cb = vpc.blend(ROLES[ra], int(aa), DARK_BG), vpc.blend(ROLES[rb], int(ab), DARK_BG)
                assert (ra, sa) != (rb, sb), f"{a} and {b} are drawn alike"
                apart = all(vpc.distance(ca, cb, k) >= vpc.APART for k in vpc.SIMULATIONS)
                if not apart and sa == sb:
                    close.append(f"{a} vs {b}: colours close under a dichromacy and the same symbol")
                la, lb = vpc.to_lab(vpc.to_rgb(ca))[0], vpc.to_lab(vpc.to_rgb(cb))[0]
                if abs(la - lb) < 12 and sa == sb:
                    close.append(f"{a} vs {b}: the same grey and the same symbol")
        assert not close, close
        return "5 classes; roles of tools.mps_plot_style; every pair apart by colour (CVD, APART 25) or by symbol; printable"

    def markers() -> str:
        w = need(st, "w")
        c = st["cls"]
        seen: Dict[str, int] = {}
        bad = []
        for k in c.stable_keys:
            cls = w.display_class(k)
            seen[cls] = seen.get(cls, 0) + 1
            if not marker_matches(w, st, k):
                bad.append((k, cls, [spot_style(sp) for _s, sp in spots_at(w, centroid_of(st, k))]))
        assert not bad, bad[:3]
        for want in ("kept", "auto_removed", "doubtful"):
            assert seen.get(want, 0) > 0, seen
        auto = list(c.auto_classes)
        assert seen["auto_removed"] == auto.count("removed") and seen["doubtful"] == auto.count("doubtful"), (seen, auto.count("removed"))
        return f"every cluster drawn at its centroid with its class's marker: {seen}"

    def ring_cue() -> str:
        w = need(st, "w")
        c = st["cls"]
        by_ring: Dict[int, Tuple[str, float]] = {}
        for (ring, _pos), k in zip(c.keys, c.stable_keys):
            if w.display_class(k) != "kept" or int(ring) in by_ring:
                continue
            hits = spots_at(w, centroid_of(st, k))
            s = [spot_style(sp) for _sc, sp in hits if (spot_style(sp)["symbol"], spot_style(sp)["brush"]) == class_style("kept")][0]
            by_ring[int(ring)] = (s["pen"], s["size"])
        assert len(by_ring) >= 2 and len(set(by_ring.values())) == len(by_ring), by_ring
        return f"kept markers of {len(by_ring)} rings differ in pen / size: {by_ring}"

    check("the window opens offscreen with its widgets; nothing running, nothing edited", opens)
    check("LUMEN_CLASS_STYLE follows validate_plot_colours' conventions (roles, CVD, greyscale, background)", style_table)
    check("differential markers: each cluster shows its display class (kept / auto_removed / doubtful present)", markers)
    check("the ring is shown too: markers of one class in different rings differ in pen or size", ring_cue)

    print("\n3. Manual review: click any cluster, restore, checkboxes, select / clear all doubtful (D-35(b))")

    def click_kept() -> str:
        w = need(st, "w")
        k = next(k for k in st["cls"].stable_keys if w.display_class(k) == "kept")
        click(w, st, k)
        pump(app)
        assert w.display_class(k) == "manual_removed" and w.decisions.manual_action(k) == "removed" and w.decisions.is_removed(k)
        assert marker_matches(w, st, k), "the marker did not change"
        click(w, st, k)
        pump(app)
        assert w.display_class(k) == "kept" and w.decisions.manual_action(k) == "none"
        click(w, st, k)
        pump(app)
        st["manual_key"] = k
        return f"{k}: kept -> removed by hand -> kept -> removed by hand (marker follows)"

    def restore() -> str:
        w = need(st, "w")
        k = next(k for k in st["cls"].stable_keys if w.display_class(k) == "auto_removed")
        click(w, st, k)
        pump(app)
        assert w.display_class(k) == "restored" and w.decisions.manual_action(k) == "restored" and not w.decisions.is_removed(k)
        assert marker_matches(w, st, k)
        st["restored_key"] = k
        return f"{k}: auto-removed -> restored (kept by hand)"

    def table() -> str:
        w = need(st, "w")
        t = w.doubtful_table
        dbt = [k for k in st["cls"].stable_keys if st["cls"].auto_classes[list(st["cls"].stable_keys).index(k)] == "doubtful"]
        assert t.rowCount() == len(dbt), (t.rowCount(), len(dbt))
        heads = " | ".join(t.horizontalHeaderItem(i).text() for i in range(t.columnCount()) if t.horizontalHeaderItem(i) is not None).lower()
        for token in ("ring", "hull", "hop", "tubulin", "spectrin", "locs", "events", "z"):
            assert token in heads, f"the doubtful table has no column naming {token!r}: {heads}"
        rows = [table_row(w, k) for k in dbt]
        assert all(r >= 0 for r in rows)
        for k, r in zip(dbt, rows):
            assert bool(t.item(r, 0).flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            assert t.item(r, 0).checkState() == QtCore.Qt.CheckState.Unchecked, "doubtful clusters start kept"
        st["dbt"] = dbt
        return f"{len(dbt)} rows with a checkbox each (unchecked = kept); columns: {heads}"

    def checkbox() -> str:
        w = need(st, "w")
        k = need(st, "dbt")[0]
        r = table_row(w, k)
        w.doubtful_table.item(r, 0).setCheckState(QtCore.Qt.CheckState.Checked)
        pump(app)
        assert w.decisions.is_removed(k) and w.display_class(k) == "manual_removed" and marker_matches(w, st, k)
        w.doubtful_table.item(table_row(w, k), 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        pump(app)
        assert not w.decisions.is_removed(k) and w.display_class(k) == "doubtful"
        return f"{k}: checked -> removed (marker manual_removed), unchecked -> doubtful again"

    def click_doubtful_syncs_table() -> str:
        w = need(st, "w")
        k = need(st, "dbt")[1]
        click(w, st, k)
        pump(app)
        assert w.decisions.is_removed(k) and w.doubtful_table.item(table_row(w, k), 0).checkState() == QtCore.Qt.CheckState.Checked
        click(w, st, k)
        pump(app)
        assert w.doubtful_table.item(table_row(w, k), 0).checkState() == QtCore.Qt.CheckState.Unchecked
        return "a click on a doubtful marker checks / unchecks its row"

    def select_clear_all() -> str:
        w = need(st, "w")
        dbt = need(st, "dbt")
        w.select_all_doubtful_button.click()
        pump(app)
        assert all(w.decisions.is_removed(k) and w.display_class(k) == "manual_removed" for k in dbt)
        assert all(w.doubtful_table.item(table_row(w, k), 0).checkState() == QtCore.Qt.CheckState.Checked for k in dbt)
        w.clear_all_doubtful_button.click()
        pump(app)
        assert not any(w.decisions.is_removed(k) for k in dbt) and all(w.display_class(k) == "doubtful" for k in dbt)
        assert all(w.doubtful_table.item(table_row(w, k), 0).checkState() == QtCore.Qt.CheckState.Unchecked for k in dbt)
        w.doubtful_table.item(table_row(w, dbt[0]), 0).setCheckState(QtCore.Qt.CheckState.Checked)
        pump(app)
        n_log = len(w.decisions.edit_log)
        assert n_log >= 2 * len(dbt) + 4, n_log
        return f"select all: {len(dbt)} doubtful removed and checked; clear all: back; {n_log} edits logged in order"

    def english() -> str:
        w = need(st, "w")
        txt = w.run_button.text()
        assert txt.isascii() and re.search(r"run|analy", txt, re.I) and not re.search(r"correr|an[aá]lisis", txt, re.I), txt
        for b in (w.run_button, w.select_all_doubtful_button, w.clear_all_doubtful_button, w.export_button):
            assert b.text().strip() and b.toolTip().strip(), f"{b.text()!r} has no tooltip"
            assert b.text().isascii(), b.text()
        return f"run button {txt!r}; buttons with English text and tooltips"

    check("click a kept cluster: removed by hand, and back (any cluster can be removed by hand)", click_kept)
    check("click an auto-removed cluster: restored", restore)
    check("the doubtful table: one row per doubtful cluster, a checkbox each, the descriptors as columns", table)
    check("a row's checkbox removes / keeps that doubtful cluster (marker follows)", checkbox)
    check("clicking a doubtful marker keeps its checkbox in sync", click_doubtful_syncs_table)
    check("'Select all doubtful' / 'Clear all doubtful': every doubtful cluster, table and markers", select_clear_all)
    check("visible text in English (the run button need not say 'correr análisis'), tooltips present", english)

    print("\n4. Re-run in a worker thread; results with the leak banner (D-35(d)(5): 'correr de nuevo sin congelar la ventana')")

    def rerun() -> str:
        w = need(st, "w")
        ticks = [0]
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: ticks.__setitem__(0, ticks[0] + 1))
        timer.start(TICK_MS)
        got: Dict[str, Any] = {}
        loop = QtCore.QEventLoop()

        def on_done(result: Any) -> None:
            got["result"] = result
            got["t"] = time.perf_counter()
            loop.quit()

        w.analysis_finished.connect(on_done)
        t0 = time.perf_counter()
        w.run_button.click()
        t_click = time.perf_counter() - t0
        running = w.is_running()
        disabled = not w.run_button.isEnabled()
        app.processEvents()
        progress = w.progress_bar.isVisible()
        ticks_at_click = ticks[0]
        QtCore.QTimer.singleShot(int(RUN_TIMEOUT_S * 1000), loop.quit)
        if "result" not in got:
            loop.exec_()
        timer.stop()
        assert "result" in got, f"no analysis_finished within {RUN_TIMEOUT_S:.0f} s"
        run_s = got["t"] - t0
        n_ticks = ticks[0] - ticks_at_click
        assert running and disabled and progress, (running, disabled, progress)
        assert t_click < 1.0 and run_s > t_click, (t_click, run_s)
        assert n_ticks >= max(5, int(0.2 * run_s * 1000 / TICK_MS)), (n_ticks, run_s)
        pump(app, 0.2)
        assert not w.is_running() and w.run_button.isEnabled() and not w.progress_bar.isVisible()
        st["result"] = got["result"]
        return (f"the click returned in {t_click * 1000:.0f} ms, the run took {run_s:.1f} s in a worker; the UI ticked {n_ticks} "
                f"times meanwhile ({TICK_MS} ms timer); run button disabled and progress shown until the results")

    def results_shown() -> str:
        w = need(st, "w")
        r = need(st, "result")
        assert w.last_result is r
        assert r.arc_centroid is not None and r.arc_centroid.reference_curve == "centroid_membrane"
        assert r.leak_calibrated is False
        assert r.n_removed_final == w.decisions.n_removed_final and r.n_manual == w.decisions.n_manual
        text = w.results_text()
        low = text.lower()
        for token in ("z_a", "centroid membrane", "2d", "localization membrane", "not calibrated for axial leak", "removed"):
            assert token in low, f"the results area does not show {token!r}"
        assert re.search(r"conservative|bias", low), "the localization membrane's known bias is not printed beside it"
        numbers = [float(v) for v in re.findall(r"[-+]?\d+\.\d+", text)]
        assert any(abs(v - float(r.arc_centroid.z_A)) <= 0.01 for v in numbers), "z_A of the centroid membrane not shown"
        assert w.leak_banner.isVisible() and "not calibrated for axial leak" in w.leak_banner.text().lower() and "H5-E" in w.leak_banner.text()
        assert w.decisions.results_shown is True, "showing the results must mark them shown (provenance)"
        assert not w.edited_label.isVisible(), "no edit after the results yet"
        return f"z_A (centroid membrane) {r.arc_centroid.z_A:+.2f}, the 2D test, the localization membrane with its bias, counts; banner shown"

    def input_untouched() -> str:
        from tools import mps_sim_harness as vl
        diff = vl.signature_differences(vl.ring_signature(need(st, "res")), st["sig_before"])
        assert not diff, diff
        r = need(st, "result")
        diff = vl.signature_differences(vl.ring_signature(r.rings), vl.ring_signature(need(st, "w").decisions.apply(st["res"])))
        assert not diff, diff
        return "the re-run cleaned a COPY (clean_rings of the final set; build_rings not re-run); the window's rings untouched"

    check("the re-run runs in a worker: the click returns at once, the UI keeps ticking, button and progress follow", rerun)
    check("the results appear: centroid-membrane z_A, 2D sensitivity, localization membrane with its bias, the leak banner", results_shown)
    check("the rings the window holds are untouched; the result's rings = decisions.apply(res)", input_untouched)

    print("\n5. Provenance: 'edited after results were shown' and the export (D-35(b))")

    def edited_label() -> str:
        w = need(st, "w")
        need(st, "result")
        k = st["dbt"][-1]
        click(w, st, k)
        pump(app)
        assert w.decisions.results_shown_before_edit is True
        assert w.edited_label.isVisible() and "edited after results were shown" in w.edited_label.text().lower()
        return "an edit after the results were shown: the label appears (and the decisions carry the flag)"

    def export() -> str:
        w = need(st, "w")
        path = os.path.join(WORK, "axon_lumen_decisions.csv")
        written = w.export_tables(path)
        res_path = os.path.splitext(path)[0] + "_results.csv"
        assert os.path.isfile(path) and os.path.isfile(res_path) and set(map(os.path.abspath, written)) >= {os.path.abspath(path), os.path.abspath(res_path)}
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        L = importlib.import_module(ML)
        need_cols = set(L.LUMEN_DECISION_COLUMNS) | {"axon_id"}
        assert need_cols <= set(rows[0]), need_cols - set(rows[0])
        assert len(rows) == len(st["cls"].stable_keys)
        byk = {r["stable_key"]: r for r in rows}
        assert byk[st["restored_key"]]["manual_action"] == "restored" and byk[st["manual_key"]]["manual_action"] == "removed"
        assert all(str(r["results_shown_before_edit"]).lower() in ("true", "1") for r in rows)
        with open(res_path, "r", encoding="utf-8-sig", newline="") as fh:
            rr = list(csv.DictReader(fh))
        for col in ("axon_id", "arcc_z_A", "arcc_p_A", "leak_calibrated", "edited_after_results_shown", "lumen_n_removed_auto",
                    "lumen_n_removed_final", "lumen_n_manual", "lumen_n_doubtful"):
            assert col in rr[0], f"the results table has no {col}"
        assert str(rr[-1]["edited_after_results_shown"]).lower() in ("true", "1")
        assert str(rr[-1]["leak_calibrated"]).lower() in ("false", "0")
        # never overwrite a table of another analysis
        foreign = os.path.join(WORK, "other_analysis.csv")
        with open(foreign, "w", encoding="utf-8", newline="") as fh:
            fh.write("source,roi,segment_mode,guard_nm\nx.hdf5,roi,valley,0\n")
        before = open(foreign, "rb").read()
        try:
            w.export_tables(foreign)
        except ValueError:
            pass
        else:
            raise AssertionError("exporting onto another analysis' table must be refused")
        assert open(foreign, "rb").read() == before, "the other analysis' table was modified"
        return (f"decision table ({len(rows)} rows, the contract's columns + axon_id, the edit flag) and results table (arcc, "
                "leak_calibrated False, the edit flag); another analysis' table refused and left untouched")

    def no_widefield_window() -> str:
        Win = require(MW, "ColumnsWindow")
        w2 = Win(need(st, "res"), need(st, "cls_nowf"), n_null=49, source_name="simulated slot, no widefield")
        w2.show()
        pump(app, 0.2)
        assert w2.widefield_warning.isVisible() and re.search(r"widefield", w2.widefield_warning.text(), re.I)
        assert re.search(r"isolation", w2.widefield_warning.text(), re.I)
        w2.close()
        return "without widefield images the window says the rule is isolation only"

    def wired() -> str:
        text = ""
        for f in ("MPS_explorer.py", os.path.join("tools", "mps_rings_window.py")):
            with open(os.path.join(REPO_ROOT, f), "r", encoding="utf-8") as fh:
                text += fh.read()
        assert "ColumnsWindow" in text or "mps_columns_window" in text, "the window is launched from neither the Rings panel nor the main window"
        return "the Rings panel or the main window references the window (the launch itself: the user's manual test)"

    check("'edited after results were shown': the label appears after an edit that follows the results", edited_label)
    check("export: the decision table and the results with the provenance flag; another analysis' table refused", export)
    check("no widefield: the window warns that the rule is isolation only", no_widefield_window)
    check("wired: launched from the Rings panel or the main window (static check)", wired)

    print("\n6. After the reviews of H5-D (Fix stage): real clicks through the selected row's ring, the axon's provenance "
          "kept between windows, foreign and appended tables, the analyses' own warnings")

    def click_through_highlight() -> str:
        w = need(st, "w")
        t = w.doubtful_table
        c = st["cls"]
        kidx = {k: i for i, k in enumerate(c.stable_keys)}
        C = np.asarray(c.centroids_nm, float)
        reach = []
        for r in range(t.rowCount()):
            k = t.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            pts = w.markers_at(QtCore.QPointF(float(C[kidx[k]][0]), float(C[kidx[k]][1])))
            if len(pts) and pts[0].data() == k:
                reach.append((r, k))
        assert reach, "no doubtful marker is drawn on top"
        done = 0
        for r, k in reach[:4]:
            t.selectRow(r)
            pump(app, 0.05)
            assert len(w.highlight_item.points()) == 1, "the selected row's cluster is not ringed"
            before = w.decisions.is_removed(k)
            real_click(app, w, C[kidx[k]])
            assert w.decisions.is_removed(k) != before, f"{k}: a real click on the ringed marker of the selected row did nothing"
            real_click(app, w, C[kidx[k]])
            assert w.decisions.is_removed(k) == before
            t.clearSelection()
            pump(app, 0.05)
            done += 1
        return f"{done} doubtful markers toggled by a real mouse click while their row was selected (the ring passes clicks)"

    def provenance_between_windows() -> str:
        Win = require(MW, "ColumnsWindow")
        L = importlib.import_module(ML)
        store = L.LumenReviewStore(os.path.join(WORK, "review_store"))
        a = Win(need(st, "res"), need(st, "cls"), n_null=19, source_name="simulated slot (store)", review_store=store)
        a.show()
        pump(app, 0.2)
        kept = [k for k in st["cls"].stable_keys if a.display_class(k) == "kept"]
        click(a, st, kept[0])
        pump(app)
        assert run_and_wait(app, a) is not None and a.decisions.results_shown
        click(a, st, kept[1])
        pump(app)
        assert a.decisions.results_shown_before_edit
        final, n_log = sorted(a.decisions.final_removed_keys()), len(a.decisions.edit_log)
        a.close()
        pump(app, 0.1)
        b = Win(need(st, "res"), need(st, "cls"), n_null=19, source_name="simulated slot (store)", review_store=store)
        b.show()
        pump(app, 0.2)
        assert sorted(b.decisions.final_removed_keys()) == final and len(b.decisions.edit_log) == n_log, "not restored"
        assert b.decisions.results_shown and b.decisions.results_shown_before_edit and b.edited_label.isVisible()
        click(b, st, kept[2])
        pump(app)
        assert b.decisions.edit_log[-1].after_results_shown, "an edit after re-opening an axon whose results were seen"
        other = Win(need(st, "res"), need(st, "cls"), n_null=19, source_name="another simulated axon", review_store=store)
        assert not other.decisions.edit_log and not other.decisions.results_shown, "another axon inherited this one's review"
        for x in (b, other):
            x.close()
        pump(app, 0.1)
        return (f"closed and re-opened: {len(final)} removed, {n_log} logged edits and 'results shown' restored from the "
                "review store; the next edit is flagged; another axon starts clean")

    def review_of_another_build() -> str:
        # Final audit (2026-10-06): the same axon id opened on ANOTHER ring build (another cluster set: the parameters
        # edited, a re-exported file of the same name, another pixel size) and that window merely closed overwrote the
        # axon's saved hand review. Build B here = the same rings without one cluster (another fingerprint).
        Win = require(MW, "ColumnsWindow")
        L = importlib.import_module(ML)
        Mc = importlib.import_module("tools.mps_columns")
        res, cls = need(st, "res"), need(st, "cls")
        ring0 = sorted(res.rings, key=lambda r: int(r.index))[0]
        res_b = Mc.clean_rings(res, [(int(ring0.index), 0)], reason="test: another ring build of the axon")
        cls_b = L.classify_lumen(res_b)
        assert cls_b.cluster_set_sha != cls.cluster_set_sha
        store = L.LumenReviewStore(os.path.join(WORK, "review_store_builds"))

        def opened(r: Any, c: Any) -> Any:
            w = Win(r, c, n_null=19, source_name="simulated slot (two builds)", review_store=store)
            w.show()
            pump(app, 0.2)
            return w

        a = opened(res, cls)
        for k in [k for k in cls.stable_keys if a.display_class(k) == "kept"][:2]:
            a.decisions.toggle(k, via="click")
            a._after_edit()
        final, n_log = sorted(a.decisions.final_removed_keys()), len(a.decisions.edit_log)
        a.close()
        pump(app, 0.1)
        b = opened(res_b, cls_b)
        assert b.decisions.n_manual == 0 and any("not lost" in n for n in b.store_notes), b.store_notes
        b.close()  # no edit, no run: only build B's automatic state is saved
        pump(app, 0.1)
        assert os.path.isfile(store.kept_path_for(a.axon_id, cls.cluster_set_sha)), "the hand review of build A was not kept"
        c = opened(res, cls)
        assert sorted(c.decisions.final_removed_keys()) == final and len(c.decisions.edit_log) == n_log, c.store_notes
        c.close()
        pump(app, 0.1)
        d = opened(res_b, cls_b)
        assert d.decisions.n_manual == 0 and any(n.startswith("restored") for n in d.store_notes), d.store_notes
        d.close()
        pump(app, 0.1)
        return (f"{n_log} hand edits of build A survive build B ({cls_b.n} clusters) opened and closed on the same axon id, "
                "and come back with A; B's own review comes back with B")

    def foreign_and_appended_tables() -> str:
        Win = require(MW, "ColumnsWindow")
        res, cls = need(st, "res"), need(st, "cls")
        path = os.path.join(WORK, "axon_lumen_decisions.csv")
        assert os.path.isfile(path), "the export check did not write the decision table"
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

        def write(name: str, rows_: List[Dict[str, Any]]) -> str:
            p = os.path.join(WORK, name)
            with open(p, "w", encoding="utf-8", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows_[0].keys()))
                wr.writeheader()
                wr.writerows(rows_)
            return p

        fresh = Win(res, cls, n_null=19, source_name="simulated slot")
        state0 = {k: fresh.decisions.is_removed(k) for k in cls.stable_keys}
        bad_set = write("same_id_other_set.csv", [dict(r, cluster_set_sha="0123456789abcdef") for r in rows])
        bad_keys = write("other_keys.csv", [{c_: v for c_, v in dict(r, stable_key=r["stable_key"] + "9", cluster_set_sha="").items()
                                             if c_ != "axon_id"} for r in rows])
        for p, what in ((bad_set, "same axon_id, another cluster set"), (bad_keys, "no axon_id, none of these clusters")):
            try:
                fresh.load_decisions(p)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{what}: loaded instead of refused")
            assert {k: fresh.decisions.is_removed(k) for k in cls.stable_keys} == state0 and not fresh.decisions.edit_log
        # two exports of one axon appended to one table: the NEWER decisions load
        e = Win(res, cls, n_null=19, source_name="simulated slot appended")
        p = os.path.join(WORK, "appended.csv")
        e.export_tables(p, duplicates="replace")
        k = next(k for k in cls.stable_keys if e.display_class(k) == "doubtful")
        click(e, st, k)
        pump(app)
        e.export_tables(p, duplicates="append")
        with open(p, "r", encoding="utf-8-sig", newline="") as fh:
            n_rows = len(list(csv.DictReader(fh)))
        assert n_rows == 2 * len(cls.stable_keys), n_rows
        f = Win(res, cls, n_null=19, source_name="simulated slot appended")
        n = f.load_decisions(p)
        assert f.decisions.is_removed(k) and n == len(cls.stable_keys), (f.decisions.is_removed(k), n)
        for x in (fresh, e, f):
            x.close()
        pump(app, 0.1)
        return "a table of another cluster set and one of other clusters refused (nothing changed); appended exports -> the newer"

    def analyses_warnings_shown() -> str:
        w = need(st, "w")
        r = need(st, "result")
        n = 0
        for what, a in (("arc test (centroid membrane)", r.arc_centroid), ("2D test", r.columns_2d),
                        ("arc test (localization membrane, diagnostic)", r.arc_localization)):
            for x in (getattr(a, "warnings", None) or []):
                assert f"{what}: {x}" in r.warnings, f"{what}: {x!r} is not among the run's warnings"
                n += 1
        shown = [w.warnings_list.item(i).text() for i in range(w.warnings_list.count())]
        if w.last_result is r:
            missing = [x for x in r.warnings if not any(x in s for s in shown)]
            assert not missing, missing[:3]
        return f"{n} warning(s) of the three analyses carried into the run's warnings and listed in the window"

    def main_window_reopen() -> str:
        import functools
        from types import SimpleNamespace
        from tools import mps_settings
        import MPS_explorer
        tmp = os.path.join(WORK, "settings")
        os.makedirs(tmp, exist_ok=True)
        MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=tmp)
        MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=tmp)
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(WORK, "main_store")
        sim = need(st, "sim")
        n = int(np.asarray(sim["x"]).size)
        fake_movie: Any = SimpleNamespace(frame=np.asarray(sim["frame"]), lp_lateral_nm=np.asarray(sim["lp"]),
                                          lpz_nm=np.asarray(sim["lpz"]), n_frames=int(sim["cfg"].n_frames),
                                          pixel_size_nm=float(sim["cfg"].pixel_size_nm), pixel_size_source="override",
                                          path="simulated_A7c.hdf5")
        mw.locs1 = fake_movie
        mw.xroi_unfiltered, mw.yroi_unfiltered, mw.zroi_unfiltered = (np.asarray(sim[q], float).copy() for q in ("x", "y", "z"))
        mw.roi_indices_unfiltered = np.arange(n)
        mw.roi_indices = np.arange(n)
        mw.ui.lineEdit_filename.setText("simulated_A7c.hdf5")
        try:
            l1 = mw.open_columns_review()
            assert l1 is not None and l1.is_running()
            assert mw.open_columns_review() is l1, "pressing the button again while the rings build started a second build"
            l1.wait(600)
            w1 = l1.window
            assert w1 is not None, l1.error
            assert w1.res.rings and "without the ROI" in w1.header_label.text(), "the rings note is missing"
            w1.n_null = 19
            kept = [k for k in w1.decisions.stable_keys if w1.display_class(k) == "kept"]
            w1.decisions.toggle(kept[0], via="click")
            w1._after_edit()
            assert run_and_wait(app, w1) is not None
            w1.decisions.toggle(kept[1], via="click")
            w1._after_edit()
            final, n_log = sorted(w1.decisions.final_removed_keys()), len(w1.decisions.edit_log)
            assert w1.decisions.results_shown_before_edit
            l2 = mw.open_columns_review()
            assert l2 is l1 and l2.window is w1 and w1.isVisible(), "pressing the button again rebuilt the review"
            w1.close()
            pump(app, 0.1)
            l3 = mw.open_columns_review()
            assert l3 is l1 and w1.isVisible() and len(w1.decisions.edit_log) == n_log, "a closed review is not shown again as it was"
            mw._close_columns_review()          # what loading another file does
            pump(app, 0.1)
            l4 = mw.open_columns_review()
            l4.wait(600)
            w4 = l4.window
            assert w4 is not None and w4 is not w1, "a new review was expected after the old one was closed and deleted"
            assert sorted(w4.decisions.final_removed_keys()) == final and len(w4.decisions.edit_log) == n_log
            assert w4.decisions.results_shown and w4.decisions.results_shown_before_edit and w4.edited_label.isVisible()
            w4.decisions.toggle(kept[2], via="click")
            w4._after_edit()
            assert w4.decisions.edit_log[-1].after_results_shown
        finally:
            mw._close_columns_review()
            mw.hide()
            pump(app, 0.1)
        return (f"MPS_explorer.open_columns_review: pressed again while building -> the same build; pressed again -> the same "
                f"window; closed -> shown again as it was; rebuilt after the file's review was closed -> {len(final)} removed, "
                f"{n_log} edits and the flag restored")

    check("a real click on the marker of the SELECTED doubtful row toggles it (the row's ring passes clicks)",
          click_through_highlight)
    check("the axon's decisions and 'results shown' survive closing and re-opening the review (review store; D-35b)",
          provenance_between_windows)
    check("another ring build of the same axon never overwrites its saved hand review (final audit)",
          review_of_another_build)
    check("load decisions: a table of another axon or cluster set is refused; appended exports load the newer",
          foreign_and_appended_tables)
    check("the analyses' own warnings reach the run's warnings and the window's list", analyses_warnings_shown)
    check("the main window: the button again brings the review forward; after a rebuild the axon's review is restored",
          main_window_reopen)

    w = st.get("w")
    if w is not None:
        w.close()
    pump(app, 0.1)
    shutil.rmtree(WORK, ignore_errors=True)
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.0f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
