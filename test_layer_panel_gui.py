# -*- coding: utf-8 -*-
"""
Offscreen test of UI stage 0, items 0.1 and 0.2 (the user's manual test of 2026-10-05): every legend entry and every
layer checkbox controls the REAL layer it names, and the columns review map has no legend over the data.

What the user reported: in the columns review's map ("Ring clusters in the axon frame ...") clicking a legend entry
such as "centroid membrane, every cluster" crossed out the entry's eye while the curve stayed drawn, and the legend
sat over the plot. The Z quality view's legends had the same stand-in entries (bars, peak / valley, SiZer, ring
centres). What this test fixes:

  1. tools.mps_layer_panel on its own: a row hides and shows exactly its items (nothing else), "Show all" / "Hide all"
     skip a disabled row, the state survives ``bind`` and ``clear_layers``, counts and reasons; LegendLayers: a click
     on a legend entry (pyqtgraph's own ItemSample.mouseClickEvent) hides and shows every member of its group.
  2. The columns review on a SIMULATED axon (R8): no LegendItem in the plot, the layer panel beside it (not over it),
     one marker item per class (every spot of an item of its class), each class checkbox hides exactly its item,
     ``markers_at`` ignores a hidden class, a REAL mouse click on a hidden class changes no decision, the visible
     marker still toggles (logged "click", persisted in the review store), the hover tooltip follows what is shown,
     the selected row is ringed even when its class is hidden, the ring filter, an edit and a re-run keep what is
     hidden, the membranes and the widefield image underneath toggle the real items.
  3. The Z quality view on a SIMULATED axon: every entry of both legends is bound to real items (labels unchanged),
     a click hides all of them and nothing else, and a redraw (criteria, cluster set, report) keeps it hidden.
  4. The segments superimposed (UI stage 2): the rings window has no superimposed plot and its small multiples
     still draw every segment; the axon map's segment rows (built from the same segments) sit in the layer panel
     beside its plot (no legend inside it), a row hides that segment there only, and a redraw keeps it hidden.

Column statistics are computed only on simulated axons (tools.mps_zquality_window.DEMO_CASES through
batch_columns.write_simulated_input); every file goes to a temporary folder (MPS_SELECTION_LOG_DIR included).

Run:  venv\\Scripts\\python.exe test_layer_panel_gui.py      (offscreen; about 1-2 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
T0 = time.perf_counter()
WORK = tempfile.mkdtemp(prefix="test_layer_panel_gui_")
# the exploration log of this test, never the user's (tools.mps_selection.default_log_path)
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")
N_NULL = 19


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


def need(st: Dict[str, Any], key: str) -> Any:
    if key not in st:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return st[key]


def pump(app: Any, seconds: float = 0.05) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


class _LeftClick:
    """What pyqtgraph's ItemSample.mouseClickEvent reads from a mouse event."""

    def __init__(self) -> None:
        self.accepted = False

    def button(self) -> Any:
        from PyQt5 import QtCore
        return QtCore.Qt.MouseButton.LeftButton

    def accept(self) -> None:
        self.accepted = True


def click_sample(sample: Any) -> None:
    sample.mouseClickEvent(_LeftClick())


def viewport_point(window: Any, xy: Any) -> Any:
    from PyQt5 import QtCore
    vb = window.plot_widget.getViewBox()
    sp = vb.mapViewToScene(QtCore.QPointF(float(xy[0]), float(xy[1])))
    return window.plot_widget.mapFromScene(sp)


def real_click(app: Any, window: Any, xy: Any) -> None:
    """A real mouse click (QTest) at (x', y'), the mouse first moved onto it as a user does."""
    from PyQt5 import QtCore, QtTest
    qtest: Any = QtTest.QTest
    vp = window.plot_widget.viewport()
    p = viewport_point(window, xy)
    assert vp.rect().contains(p), f"{xy} maps to {p.x()},{p.y()}, outside the viewport"
    for dx in (-30, -15, -5, 0):
        qtest.mouseMove(vp, QtCore.QPoint(p.x() + dx, p.y()))
        pump(app, 0.02)
    qtest.mouseClick(vp, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, p)
    pump(app, 0.05)


def hover(app: Any, window: Any, xy: Any) -> str:
    from PyQt5 import QtCore, QtTest
    qtest: Any = QtTest.QTest
    vp = window.plot_widget.viewport()
    p = viewport_point(window, xy)
    for dx in (-40, -2, 0):
        qtest.mouseMove(vp, QtCore.QPoint(p.x() + dx, p.y()))
        pump(app, 0.02)
    return str(window.plot_widget.getViewBox().toolTip())


def visible_map(items: Dict[str, Any]) -> Dict[str, bool]:
    return {k: bool(v.isVisible()) for k, v in items.items()}


def main() -> int:
    print("=" * 100)
    print("LAYER PANEL AND LEGENDS (UI stage 0, items 0.1 and 0.2) OFFSCREEN TEST on SIMULATED axons")
    print("=" * 100)
    from PyQt5 import QtCore, QtWidgets
    import pyqtgraph as pg
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import batch_columns as bc
    from tools import mps_columns_window as mcw
    from tools import mps_layer_panel as lp
    from tools import mps_lumen as L
    from tools import mps_zquality_window as zw
    st: Dict[str, Any] = {}

    # ------------------------------------------------------------------ 1. the panel alone
    print("\n1. tools.mps_layer_panel on its own")

    def panel_rows() -> str:
        panel = lp.LayerPanel()
        panel.add_group("Group")
        a1, a2, b1 = pg.ScatterPlotItem(), pg.PlotDataItem(), pg.ScatterPlotItem()
        seen: List[Tuple[str, bool]] = []
        calls: List[bool] = []
        panel.toggled.connect(lambda k, on: seen.append((k, on)))
        box_a = panel.add_layer("a", "Layer A", lp.Swatch("symbol", "#56b4e9", symbol="d"), items=[a1, a2],
                                on_toggle=calls.append, count=3)
        panel.add_layer("b", "Layer B", lp.Swatch("line", "#e69f00", dash=True), items=[b1])
        assert box_a.text() == "Layer A (3)" and box_a.isChecked() and not box_a.icon().isNull()
        box_a.setChecked(False)
        assert not a1.isVisible() and not a2.isVisible() and b1.isVisible(), "a row must hide exactly its own items"
        assert seen == [("a", False)] and calls == [False] and panel.state() == {"a": False, "b": True}
        assert panel.hidden() == ["a"] and not panel.is_visible("a")
        box_a.setChecked(True)
        assert a1.isVisible() and a2.isVisible() and b1.isVisible() and calls == [False, True]
        panel.set_enabled("b", False, "nothing to draw")
        assert "nothing to draw" in panel.checkbox("b").toolTip()
        panel.hide_all()
        assert not a1.isVisible() and b1.isVisible(), "Hide all must skip a disabled row"
        panel.show_all()
        assert a1.isVisible()
        panel.set_count("a", None)
        assert box_a.text() == "Layer A"
        for name in ("layers_show_all", "layers_hide_all"):
            b = panel.findChild(QtWidgets.QPushButton, name)
            assert b is not None and b.toolTip(), name
        return "a row hides exactly its items, emits toggled and calls on_toggle; Hide all skips a disabled row"

    def panel_state() -> str:
        panel = lp.LayerPanel()
        a1 = pg.ScatterPlotItem()
        panel.add_layer("a", "A", items=[a1])
        panel.set_visible("a", False)
        a_new = pg.ScatterPlotItem()
        panel.bind("a", [a_new])
        assert not a_new.isVisible(), "bind must apply the remembered state to the new items"
        panel.clear_layers(keep_state=True)
        assert panel.keys() == []
        a_again = pg.ScatterPlotItem()
        box = panel.add_layer("a", "A", items=[a_again], visible=True)
        assert not box.isChecked() and not a_again.isVisible(), "clear_layers(keep_state) forgot the state"
        panel.clear_layers(keep_state=False)
        a_fresh = pg.ScatterPlotItem()
        panel.add_layer("a", "A", items=[a_fresh])
        assert a_fresh.isVisible()
        panel.restore({"a": False, "later": False})
        assert not a_fresh.isVisible() and panel.state()["later"] is False
        external = QtWidgets.QCheckBox("x")
        external.setObjectName("mine")
        external.setChecked(False)
        hits: List[bool] = []
        panel.add_layer("ext", "External", lp.Swatch("image"), checkbox=external, on_toggle=hits.append)
        assert external.objectName() == "mine" and not panel.is_visible("ext")
        external.setChecked(True)
        assert hits == [True] and panel.is_visible("ext")
        for kind in lp.SWATCH_KINDS:
            assert not lp.swatch_icon(lp.Swatch(kind, "#009e73")).isNull(), kind
        try:
            lp.swatch_icon(lp.Swatch("pie"))
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown swatch kind must be refused")
        return "state kept through bind, clear_layers(keep_state) and restore; an adopted checkbox keeps its name"

    def legend_layers() -> str:
        state: Dict[str, bool] = {}
        legend = pg.LegendItem()
        ll = lp.LegendLayers(state)
        m1, m2, other = pg.BarGraphItem(x=[0], height=[1], width=1), pg.InfiniteLine(pos=1.0), pg.InfiniteLine(pos=2.0)
        ll.entry(legend, "group", "a group", "line", [m1, m2], pen=pg.mkPen("w"))
        ll.entry(legend, "single", "one", "scatter", [other], symbol="s", size=10)
        sample = legend.items[0][0]
        click_sample(sample)
        assert not m1.isVisible() and not m2.isVisible() and other.isVisible(), "the click hid the wrong items"
        assert not sample.item.isVisible() and state == {"group": False, "single": True} and ll.hidden() == ["group"]
        click_sample(sample)
        assert m1.isVisible() and m2.isVisible() and state["group"] is True
        click_sample(sample)
        legend.clear()
        ll.clear()
        n1 = pg.InfiniteLine(pos=3.0)
        proxy = ll.entry(legend, "group", "a group", "line", [n1], pen=pg.mkPen("w"))
        assert not n1.isVisible() and not proxy.isVisible(), "a redrawn entry forgot it was hidden"
        assert [lab.text for _s, lab in legend.items] == ["a group"]
        return "a click on the entry hides and shows every member of its group, and the state survives a redraw"

    check("LayerPanel: rows bound to real items, toggled signal, Show all / Hide all, counts, reasons", panel_rows)
    check("LayerPanel: state through bind, clear_layers, restore; an adopted checkbox; swatches", panel_state)
    check("LegendLayers: a legend click drives a group of real items; state survives a redraw", legend_layers)

    # ------------------------------------------------------------------ 2. the columns review
    print("\n2. The columns review map (simulated axon)")

    def simulate() -> str:
        out = []
        for name in ("viable", "marginal"):
            contour, seed, s = zw.DEMO_CASES[name]
            p = bc.write_simulated_input(os.path.join(WORK, f"sim_{name}.npz"), contour, seed, s)
            assert bc.input_kind(p)[0] == "simulated"
            st[f"inp_{name}"] = zw.inputs_from_npz(p)
            out.append(f"{name}: {st[f'inp_{name}'].x_nm.size:,} locs")
        return "; ".join(out)

    check("simulated axons (DEMO_CASES through batch_columns.write_simulated_input)", simulate)

    def build_window(store: Any = None, with_images: bool = True) -> Any:
        inp = need(st, "inp_viable")
        if "res" not in st:
            res, _params, lpz, _w = mcw.build_review_rings(inp)
            st["res"], st["lpz"] = res, lpz
            n = sum(len(r.clusters) for r in res.rings)
            dt = np.full(n, np.nan)
            dt[[1, 7]] = 600.0                      # both images: removed by the rule (emulated widefield)
            dt[[3, 11, 19]] = 100.0                 # WF0 only: doubtful
            wf_flags = L.widefield_flags_from_depths(res, dt, dt.copy(), interior_usable=True,
                                                     registration="emulated (test)")
            st["cls"] = L.classify_lumen(res, widefield=wf_flags)
        res, cls = st["res"], st["cls"]
        wf = None
        if with_images:
            px = 130.0
            cols = int(np.ceil(float(np.max(inp.x_nm)) / px)) + 40
            rows = int(np.ceil(float(np.max(inp.y_nm)) / px)) + 40
            rng = np.random.default_rng(0)
            img = rng.random((rows, cols))
            place = mcw.ImagePlacement(from_localizations=True)
            wf = mcw.ReviewWidefield(pixel_size_nm=px, tubulin_mask=None, spectrin_interior=None,
                                     tubulin_image=img, spectrin_image=img[::-1].copy(), tubulin_placement=place,
                                     spectrin_placement=place, registration="test image")
        w = mcw.ColumnsWindow(res, cls, n_null=N_NULL, lpz_nm=st["lpz"], widefield_images=wf,
                              source_name="simulated viable (layer test)", review_store=store,
                              xyz_lab_nm=(inp.x_nm, inp.y_nm, inp.z_nm), review_inputs=inp)
        w.resize(1440, 900)
        w.show()
        pump(app, 0.4)
        return w

    def review_opens() -> str:
        store = L.LumenReviewStore(os.path.join(WORK, "review_store"))
        st["store"] = store
        w = build_window(store)
        st["w"] = w
        w.wait_viability(120)
        pi = w.plot_widget.getPlotItem()
        assert pi.legend is None, "a legend is drawn inside the map"
        assert not any(isinstance(it, pg.LegendItem) for it in w.plot_widget.scene().items()), "a LegendItem in the map"
        panel = w.layer_panel
        assert panel.objectName() == "layer_panel" and w.findChild(QtWidgets.QWidget, "layer_panel") is panel
        want = list(mcw.DISPLAY_CLASSES) + ["membrane_before", "membrane_after", "underlay", "matches_arc", "matches_2d"]
        assert panel.keys() == want, panel.keys()
        assert panel.checkbox("underlay") is w.underlay_check and w.underlay_check.objectName() == "underlay_check"
        assert panel.toolTip() and "cannot be clicked" in panel.toolTip()
        # beside the plot, never over it
        g_plot = w.plot_widget.geometry()
        g_panel = panel.geometry()
        assert panel.parent() is w.plot_widget.parent() and not g_plot.intersects(g_panel), (g_plot, g_panel)
        assert g_plot.width() >= 0.6 * (g_plot.width() + g_panel.width()), (g_plot.width(), g_panel.width())
        return (f"no legend in the plot; panel {g_panel.width()} px beside a {g_plot.width()} px plot; rows {want}")

    def per_class_items() -> str:
        w = need(st, "w")
        assert not hasattr(w, "marker_item"), "the single marker item is still there"
        assert list(w.marker_items) == list(mcw._DRAW_ORDER)
        zs = [w.marker_items[c].zValue() for c in mcw._DRAW_ORDER]
        assert zs == sorted(zs) and len(set(zs)) == len(zs), zs
        from tools.mps_plot_style import ROLES
        counts = {}
        for cls, item in w.marker_items.items():
            role, _alpha, sym = mcw.LUMEN_CLASS_STYLE[cls]
            pts = item.points()
            for sp in pts:
                key = str(sp.data())
                assert w.display_class(key) == cls, (key, w.display_class(key), cls)
                assert str(sp.symbol()) == sym and sp.brush().color().name().lower() == ROLES[role].lower()
            counts[cls] = len(pts)
            assert w.layer_panel.checkbox(cls).text().endswith(f"({len(pts)})"), w.layer_panel.checkbox(cls).text()
        total = sum(counts.values())
        assert total == len(st["cls"].stable_keys), (total, len(st["cls"].stable_keys))
        assert counts["auto_removed"] >= 1 and counts["doubtful"] >= 1 and counts["kept"] >= 1, counts
        return f"one item per class, z in draw order, every spot of its class; counts {counts}"

    def isolated_key(w: Any, cls: str) -> Tuple[str, Any]:
        """A cluster of class ``cls`` with no other marker within 25 px (a click there can only reach it)."""
        c = st["cls"]
        C = np.asarray(c.centroids_nm, float)
        vb = w.plot_widget.getViewBox()
        nm_per_px = float(vb.viewPixelSize()[0])
        for i, k in enumerate(c.stable_keys):
            if w.display_class(k) != cls:
                continue
            d = np.hypot(C[:, 0] - C[i, 0], C[:, 1] - C[i, 1])
            d[i] = np.inf
            if float(d.min()) > 25.0 * nm_per_px:
                return k, C[i]
        raise AssertionError(f"no isolated {cls} marker")

    def class_checkbox_hides_only_its_item() -> str:
        w = need(st, "w")
        others = {"membrane_before": w.membrane_before_item, "membrane_after": w.membrane_after_item,
                  "highlight": w.highlight_item, "underlay": w.underlay_item}
        base_others = visible_map(others)
        done = []
        for cls in mcw.DISPLAY_CLASSES:
            item = w.marker_items[cls]
            before = visible_map(w.marker_items)
            w.layer_panel.checkbox(cls).setChecked(False)
            pump(app, 0.02)
            after = visible_map(w.marker_items)
            assert after[cls] is False and {k: v for k, v in after.items() if k != cls} == \
                {k: v for k, v in before.items() if k != cls}, (cls, before, after)
            assert visible_map(others) == base_others, f"hiding {cls} changed another layer"
            for sp in item.points():
                found = [p.data() for p in w.markers_at(sp.pos())]
                assert sp.data() not in found, f"markers_at still returns a hidden {cls} marker"
                break
            w.layer_panel.checkbox(cls).setChecked(True)
            pump(app, 0.02)
            assert visible_map(w.marker_items) == before
            if len(item.points()):
                sp = item.points()[0]
                assert sp.data() in [p.data() for p in w.markers_at(sp.pos())]
            done.append(cls)
        return f"each of {len(done)} classes hides exactly its own item; markers_at follows"

    def hidden_class_not_clickable() -> str:
        w = need(st, "w")
        store = need(st, "store")
        k, xy = isolated_key(w, "kept")
        n_log = len(w.decisions.edit_log)
        w.layer_panel.checkbox("kept").setChecked(False)
        pump(app, 0.05)
        real_click(app, w, xy)
        assert w.display_class(k) == "kept" and len(w.decisions.edit_log) == n_log, "a hidden marker took the click"
        w.layer_panel.checkbox("kept").setChecked(True)
        pump(app, 0.05)
        real_click(app, w, xy)
        assert w.display_class(k) == "manual_removed", "the visible marker did not toggle"
        last = w.decisions.edit_log[-1]
        assert last.via == "click" and last.stable_key == k, (last.via, last.stable_key)
        saved = store.path_for(w.axon_id)
        assert os.path.isfile(saved), "the click was not persisted"
        w2 = build_window(store)
        try:
            assert w2.display_class(k) == "manual_removed", "the decision was not restored from the review store"
        finally:
            w2.close()
            pump(app, 0.1)
        real_click(app, w, xy)
        assert w.display_class(k) == "kept"
        st["iso_kept"] = (k, xy)
        return f"{k}: hidden -> a real click changes nothing; shown -> toggles (via click, saved, restored)"

    def hover_follows() -> str:
        w = need(st, "w")
        k, xy = need(st, "iso_kept")
        tip = hover(app, w, xy)
        assert k in tip, f"no tooltip of {k}: {tip!r}"
        w.layer_panel.checkbox("kept").setChecked(False)
        pump(app, 0.02)
        tip_hidden = hover(app, w, xy)
        assert k not in tip_hidden, tip_hidden
        w.layer_panel.checkbox("kept").setChecked(True)
        pump(app, 0.02)
        # re-ticked under the still mouse: the tooltip names it again at once (review fix)
        assert k in str(w.plot_widget.getViewBox().toolTip())
        # far from it, inside the plot (the empty middle of the ring; a point outside the viewport gets no event)
        centre = np.asarray(st["cls"].centroids_nm, float)[:, :2].mean(axis=0)
        far = hover(app, w, centre)
        assert k not in far and far == "", far[:80]
        # review fix: a click redraws the markers; the tooltip under the still mouse names the NEW class at once,
        # and no stale text is left over the empty middle of the ring
        hover(app, w, xy)
        real_click(app, w, xy)
        tip_after = str(w.plot_widget.getViewBox().toolTip())
        assert w.display_class(k) == "manual_removed" and k in tip_after and "removed by hand" in tip_after, tip_after
        empty = hover(app, w, centre)
        assert empty == "", f"stale tooltip over the empty centre: {empty[:80]!r}"
        real_click(app, w, xy)
        assert w.display_class(k) == "kept"
        return f"tooltip shows {k} over its marker, nothing once its class is hidden; follows a click, never stale"

    def highlight_and_filter() -> str:
        w = need(st, "w")
        t = w.doubtful_table
        assert t.rowCount() >= 1
        w.layer_panel.checkbox("doubtful").setChecked(False)
        t.selectRow(0)
        pump(app, 0.05)
        assert len(w.highlight_item.points()) == 1 and w.highlight_item.isVisible(), "the selected row is not ringed"
        t.clearSelection()
        rings = [w.ring_filter.itemData(i) for i in range(w.ring_filter.count())]
        ring = [r for r in rings if r is not None][0]
        w.ring_filter.setCurrentIndex(rings.index(ring))
        pump(app, 0.05)
        assert not w.marker_items["doubtful"].isVisible(), "the ring filter showed a hidden class again"
        ring_of = {str(sp.data()): w._ring_of[str(sp.data())] for it in w.marker_items.values() for sp in it.points()}
        assert ring_of and set(ring_of.values()) == {int(ring)}, set(ring_of.values())
        n_kept = len(w.marker_items["kept"].points())
        assert w.layer_panel.checkbox("kept").text().endswith(f"({n_kept})")
        w.ring_filter.setCurrentIndex(0)
        pump(app, 0.05)
        assert not w.marker_items["doubtful"].isVisible()
        return f"highlight drawn with its class hidden; ring {ring} alone keeps 'doubtful' hidden; counts follow"

    def survives_edit_and_rerun() -> str:
        w = need(st, "w")
        k, xy = need(st, "iso_kept")
        w.layer_panel.checkbox("membrane_before").setChecked(False)
        real_click(app, w, xy)                      # an edit redraws the markers
        assert w.display_class(k) == "manual_removed"
        assert not w.marker_items["doubtful"].isVisible() and not w.membrane_before_item.isVisible()
        got: Dict[str, Any] = {}
        w.analysis_finished.connect(lambda r: got.__setitem__("r", r))
        assert w.start_run(), "the re-run did not start"
        t_end = time.perf_counter() + 600
        while w.is_running() and time.perf_counter() < t_end:
            pump(app, 0.05)
        pump(app, 0.2)
        assert "r" in got, "no result of the re-run"
        assert not w.marker_items["doubtful"].isVisible() and not w.membrane_before_item.isVisible()
        assert w.marker_items["kept"].isVisible() and w.membrane_after_item.isVisible()
        w.layer_panel.checkbox("membrane_after").setChecked(False)
        assert not w.membrane_after_item.isVisible()
        w._draw_membranes()
        assert not w.membrane_after_item.isVisible()
        real_click(app, w, xy)
        assert w.display_class(k) == "kept"
        w.layer_panel.show_all()
        assert all(visible_map(w.marker_items).values()) and w.membrane_before_item.isVisible()
        return "after an edit and a re-run (simulated axon) the hidden layers stay hidden; Show all restores"

    def underlay_toggles() -> str:
        w = need(st, "w")
        assert w.underlay_check.isEnabled() and w.underlay_check.isChecked() and w.underlay_item.isVisible()
        w.layer_panel.checkbox("underlay").setChecked(False)
        assert not w.underlay_item.isVisible()
        w.underlay_combo.setCurrentIndex(1)
        pump(app, 0.02)
        assert not w.underlay_item.isVisible(), "changing the image showed it while unticked"
        w.layer_panel.checkbox("underlay").setChecked(True)
        assert w.underlay_item.isVisible()
        w.layer_panel.hide_all()
        assert not w.underlay_item.isVisible() and not any(visible_map(w.marker_items).values())
        w.layer_panel.show_all()
        assert w.underlay_item.isVisible() and all(visible_map(w.marker_items).values())
        w2 = build_window(None, with_images=False)
        try:
            assert not w2.underlay_check.isEnabled() and not w2.underlay_item.isVisible()
            assert "no widefield" in w2.underlay_check.toolTip().lower() or not w2.underlay_check.isEnabled()
            w2.layer_panel.show_all()
            assert not w2.underlay_item.isVisible(), "Show all drew an image that does not exist"
        finally:
            w2.close()
            pump(app, 0.1)
        return "the widefield image underneath toggles the real image item; disabled without images"

    check("review opens: no legend in the plot, the layer panel beside it with its rows", review_opens)
    check("one marker item per class, in draw order, counts in the panel", per_class_items)
    check("each class checkbox hides exactly its item (nothing else); markers_at ignores it", class_checkbox_hides_only_its_item)
    check("a hidden class is not clickable (real click); visible: toggles, logged, persisted", hidden_class_not_clickable)
    check("the hover tooltip shows the visible markers only", hover_follows)
    check("selected-row highlight with its class hidden; ring filter keeps hidden classes; counts", highlight_and_filter)
    check("hidden layers survive an edit and a re-run; membranes toggle the real curves", survives_edit_and_rerun)
    check("widefield image underneath: the real image item; disabled without images", underlay_toggles)

    # ------------------------------------------------------------------ 3. the Z quality view
    print("\n3. The Z quality legends (simulated axon)")

    def zq_open() -> str:
        w = zw.open_z_quality_for_inputs(need(st, "inp_marginal"))
        w.wait(300)
        assert w.error is None and w.report is not None, w.error
        w.show()
        pump(app, 0.3)
        st["zq"] = w
        assert not w.cluster_panel.isHidden(), "the cluster panel is not shown"
        return f"{len(w.report.v2.rings)} rings; {len(w.legend.items)} + {len(w.cluster_legend.items)} legend entries"

    def plot_items(plot: Any) -> List[Any]:
        return [it for it in plot.items if not isinstance(it, pg.LegendItem)]

    def legend_entries_drive(which: str) -> Callable[[], str]:
        def fn() -> str:
            w = need(st, "zq")
            legend = w.legend if which == "profile" else w.cluster_legend
            plots = (w.profile_plot, w.strip_plot) if which == "profile" else (w.cluster_plot, w.cluster_strip)
            every = [it for p in plots for it in plot_items(p)]
            claimed: Dict[int, str] = {}
            report = []
            for sample, label in legend.items:
                proxy: Any = sample.item
                assert isinstance(sample.item, lp._GroupMixin), f"{label.text!r} is not bound to real items"
                members = proxy.members()
                for m in members:
                    assert any(m is it for it in every), f"{label.text!r}: a member is not drawn in the plots"
                    assert id(m) not in claimed, f"{label.text!r} and {claimed.get(id(m))!r} share an item"
                    claimed[id(m)] = label.text
                before = {id(it): it.isVisible() for it in every}
                click_sample(sample)
                assert not proxy.isVisible(), "the entry's eye is not crossed"
                for it in every:
                    want = False if any(it is m for m in members) else before[id(it)]
                    assert it.isVisible() == want, f"{label.text!r}: {type(it).__name__} visible {it.isVisible()}"
                click_sample(sample)
                assert all(it.isVisible() == before[id(it)] for it in every)
                report.append(f"{label.text.split('<br>')[0][:22]}={len(members)}")
            # what no entry names is drawn always: the strip's zero line, the 'data cut' marks, and the cluster
            # panel's SiZer bands (no entry, as before: rows would overflow the panel's fixed height)
            free = [it for it in every if id(it) not in claimed]
            for it in free:
                if which == "clusters" and isinstance(it, pg.BarGraphItem):
                    assert any(it is x for x in w.cluster_strip.items), "an unnamed bar outside the SiZer strip"
                    continue
                assert isinstance(it, (pg.InfiniteLine, pg.TextItem)), f"an unnamed {type(it).__name__}"
                if isinstance(it, pg.TextItem):
                    assert "data cut" in it.toPlainText(), it.toPlainText()
            if which == "profile":
                n_r = len(w.report.v2.rings)
                rc = [s.item for s, lab in legend.items if lab.text.startswith("ring centre")][0]
                assert len(rc.members()) == 3 * n_r, (len(rc.members()), n_r)
            return "; ".join(report) + f"; {len(free)} always drawn"
        return fn

    def zq_survives_redraw() -> str:
        w = need(st, "zq")
        labels_before = [lab.text for _s, lab in w.legend.items]
        clabels_before = [lab.text for _s, lab in w.cluster_legend.items]
        for legend, start in ((w.legend, "ring centre"), (w.legend, "localizations"), (w.cluster_legend, "smoothed"),
                              (w.cluster_legend, "ring slab cut")):
            click_sample([s for s, lab in legend.items if lab.text.startswith(start)][0])
        hidden = sorted(w.hidden_layers())
        assert hidden == sorted(["ring_centre", "loc_bars", "cl_kde", "cl_cut"]), hidden
        w._draw(w.report)
        w.cluster_source_combo.setCurrentIndex((w.cluster_source_combo.currentIndex() + 1)
                                               % w.cluster_source_combo.count())
        pump(app, 0.1)
        assert [lab.text for _s, lab in w.legend.items] == labels_before, "a legend label changed"
        assert [lab.text for _s, lab in w.cluster_legend.items] == clabels_before
        for legend, start in ((w.legend, "ring centre"), (w.legend, "localizations"), (w.cluster_legend, "smoothed"),
                              (w.cluster_legend, "ring slab cut")):
            proxy = [s.item for s, lab in legend.items if lab.text.startswith(start)][0]
            assert not proxy.isVisible() and proxy.members(), start
            assert not any(m.isVisible() for m in proxy.members()), f"{start}: shown again after a redraw"
        folder = os.path.join(WORK, "zq_export")
        paths = w.export_report(folder)
        assert sum(p.endswith(".png") for p in paths) == 2
        for legend in (w.legend, w.cluster_legend):
            for s, _lab in legend.items:
                if not s.item.isVisible():
                    click_sample(s)
        assert not w.hidden_layers()
        return f"4 entries hidden through a redraw and a cluster-set change; labels unchanged; export writes {len(paths)}"

    def zq_tooltip() -> str:
        assert "Click a legend entry to hide or show it." in zw.ZQUALITY_WINDOW_TOOLTIPS["graphics"]
        assert "Click a legend entry to hide or show it." in need(st, "zq").graphics.toolTip()
        need(st, "zq").close()
        return "the profile's tooltip says how the legend works"

    check("Z quality view on a simulated axon with its cluster panel", zq_open)
    check("profile legend: every entry hides all its items and nothing else (groups included)",
          legend_entries_drive("profile"))
    check("cluster legend: every entry hides all its items and nothing else", legend_entries_drive("clusters"))
    check("hidden entries survive a redraw and a cluster-set change; labels identical", zq_survives_redraw)
    check("the profile tooltip mentions the legend clicks", zq_tooltip)

    # ------------------------------------------------------------------ 4. the rings window
    print("\n4. The segments superimposed: on the axon map, not in the rings window (simulated axon)")

    def rings_overlay() -> str:
        # Design (UI stage 2, IMPL-C): the rings window has no superimposed plot and its small multiples still
        # draw every segment; the axon map's segment rows (built from the same ms) drive only their own items,
        # keep their state through a redraw, sit beside the plot, and there is no legend inside it.
        from tools import mps_axon_map_layers as L
        from tools.mps_axon_map import AxonMap
        from tools.mps_multisegment import analyze_all_segments
        from tools.mps_rings_window import MPSRingsWindow
        inp = need(st, "inp_viable")
        ms = analyze_all_segments(inp.x_nm, inp.y_nm, inp.z_nm, source_name="simulated viable",
                                  pixel_size_nm=130.0, pixel_size_source="override")
        w = MPSRingsWindow(ms)
        w.show()
        tabs = w.findChild(QtWidgets.QTabWidget)
        tabs.setCurrentIndex(1)                     # "Each segment": laid out only when shown
        pump(app, 0.3)
        st["rings"] = w
        for gone in ("plot_overlay", "overlay_layers", "plot_z", "zhist_grid"):
            assert not hasattr(w, gone), f"the rings window still has {gone}"
        with_locs = [(k, seg, an) for k, (seg, an) in enumerate(zip(ms.segments, ms.analyses))
                     if an is not None and np.asarray(an.x_slab).size]
        assert with_locs, "no segment with localizations"
        plots = list(w.spatial_grid.ci.items)
        assert len(plots) == len(with_locs), (len(plots), len(with_locs))
        multiples = []
        for p, (_k, _seg, an) in zip(plots, with_locs):
            scat = [it for it in p.items if isinstance(it, pg.ScatterPlotItem)]
            assert len(scat) == 1 and len(scat[0].points()) == np.asarray(an.x_slab).size
            multiples.append(scat[0])

        amap = AxonMap(dark=True)
        an0 = next(a for a in ms.analyses if a is not None)
        amap.set_inputs(L.MapInputs(analysis=an0, rings=ms, source="segments"), view="segments")
        amap.resize(1000, 700)
        amap.show()
        pump(app, 0.3)
        assert amap.view() == "segments" and amap.inputs().source == "segments"
        assert amap.plot.getPlotItem().legend is None, "a legend inside the map's plot"
        panel = amap.layers
        keys = [k for k in panel.keys() if k.startswith("seg") and k[3:].isdigit()]
        assert keys == [f"seg{seg.index}" for _k, seg, _an in with_locs], keys
        all_scat = [it for it in amap.plot.getPlotItem().items if isinstance(it, pg.ScatterPlotItem)]
        seg_items = {k: panel.items(k) for k in keys}
        for k, (_kk, seg, an) in zip(keys, with_locs):
            items = seg_items[k]
            assert len(items) == 1 and items[0].isVisible(), k
            assert len(items[0].points()) == np.asarray(an.x_slab).size
            assert panel.checkbox(k).isChecked() and panel.checkbox(k).text().endswith(f"({len(items[0].points())})")
        k0 = keys[0]
        item = seg_items[k0][0]
        before = {id(it): it.isVisible() for it in all_scat}
        panel.checkbox(k0).setChecked(False)
        assert not item.isVisible()
        assert all(it.isVisible() == before[id(it)] for it in all_scat if it is not item), \
            "a segment row hid more than its own items"
        assert all(m.isVisible() for m in multiples), "the small multiples must keep every segment"
        # a redraw from the same inputs (set_inputs, no preset): the row keeps its state on the new items
        amap.set_inputs(L.MapInputs(analysis=an0, rings=ms, source="segments"))
        pump(app, 0.05)
        new_item = panel.items(k0)[0]
        assert new_item is not item and not new_item.isVisible() and not panel.checkbox(k0).isChecked()
        assert all(panel.items(k)[0].isVisible() for k in keys[1:])
        panel.checkbox(k0).setChecked(True)
        assert new_item.isVisible()
        g1, g2 = amap.plot.geometry(), panel.geometry()
        assert not g1.intersects(g2) and g2.left() >= g1.right(), (g1, g2)
        amap.close()
        w.close()
        return (f"rings window: no superimposed plot, {len(plots)} multiples with every segment; map: {len(keys)} "
                "segment rows beside the plot, a row hides only its segment, kept through a redraw, no legend")

    check("segments: rings window multiples only; the map's segment rows beside its plot", rings_overlay)

    # ------------------------------------------------------------------ 5. groups (UI stage 2, IMPL-B)
    print("\n5. LayerPanel groups, swatch size and casing, scroll, title hook (UI stage 2)")

    def groups_rows_and_toggle() -> str:
        panel = lp.LayerPanel()
        combo = QtWidgets.QComboBox()
        combo.addItems(["one source", "another"])
        title = panel.add_group("Localizations", key="locs", caption="analyze_axon on the ROI", header=combo,
                                collapsible=True)
        a1, a2, b1, c1 = (pg.ScatterPlotItem() for _ in range(4))
        panel.add_layer("a", "A", lp.Swatch("symbol", "#56b4e9", size=3), items=[a1], count=4)
        panel.add_layer("a2", "A2", items=[a2], visible=False)
        panel.add_group("Contour", key="contour")
        panel.add_layer("b", "B", lp.Swatch("line", "#e8e8e8", cased=True), items=[b1])
        panel.add_group("Plain title")                              # stage 0: no box, rows outside any group
        panel.add_layer("c", "C", items=[c1])
        assert title.text() == "Localizations" and panel.groups() == ["locs", "contour"]
        assert panel.group_rows("locs") == ["a", "a2"] and panel.group_of("b") == "contour"
        assert panel.group_of("c") is None and panel.group_header("locs") is combo
        assert panel.group_caption("locs") == "analyze_axon on the ROI"
        toggle = panel.findChild(QtWidgets.QCheckBox, "group_toggle_locs")
        assert toggle is not None and toggle.checkState() == QtCore.Qt.CheckState.PartiallyChecked
        toggle.click()                                    # partial -> every row shown
        assert a1.isVisible() and a2.isVisible() and toggle.checkState() == QtCore.Qt.CheckState.Checked
        toggle.click()                                    # all shown -> every row hidden
        assert not a1.isVisible() and not a2.isVisible() and b1.isVisible() and c1.isVisible()
        assert toggle.checkState() == QtCore.Qt.CheckState.Unchecked
        panel.set_group_shown("locs", True)
        assert a1.isVisible() and a2.isVisible()
        panel.checkbox("a2").setChecked(False)
        assert toggle.checkState() == QtCore.Qt.CheckState.PartiallyChecked
        panel.set_group_collapsed("locs", True)
        body = panel.findChild(QtWidgets.QWidget, "group_body_locs")
        assert body is not None and body.isHidden() and a1.isVisible(), "folding must not hide what the rows draw"
        panel.set_group_collapsed("locs", False)
        assert not body.isHidden()
        assert panel.shown() == ["A (4)", "B", "C"] and panel.shown_keys() == ["a", "b", "c"]
        return "rows belong to the last keyed group; its check is tristate and shows/hides them; folding keeps items"

    def groups_enabled_and_state() -> str:
        panel = lp.LayerPanel()
        combo = QtWidgets.QComboBox()
        panel.add_group("Rings", key="rings", caption="from ms", header=combo)
        r1 = pg.ScatterPlotItem()
        panel.add_layer("r", "Ring origin", items=[r1], tip="its tip")
        panel.set_group_enabled("rings", False, "the segments are of another selection")
        reason = panel.findChild(QtWidgets.QLabel, "group_reason_rings")
        assert reason is not None and "another selection" in reason.text() and not reason.isHidden()
        assert not panel.checkbox("r").isEnabled() and not combo.isEnabled()
        assert "another selection" in panel.checkbox("r").toolTip() and panel.shown() == []
        panel.hide_all()
        assert r1.isVisible(), "Hide all must skip the rows of a disabled group"
        panel.set_group_enabled("rings", True)
        assert panel.checkbox("r").isEnabled() and panel.checkbox("r").toolTip() == "its tip" and combo.isEnabled()
        assert reason.isHidden() and panel.shown() == ["Ring origin"]
        panel.set_group_collapsed("rings", True)
        panel.checkbox("r").setChecked(False)
        panel.clear_layers(keep_state=True)
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        pump(app, 0.02)
        assert panel.groups() == [] and panel.keys() == []
        assert combo.count() == 0 and combo.parent() is None, "the caller's header widget must survive a clear"
        panel.add_group("Rings", key="rings", collapsible=True)
        r2 = pg.ScatterPlotItem()
        panel.add_layer("r", "Ring origin", items=[r2])
        assert panel.is_group_collapsed("rings") and not r2.isVisible(), "group fold and row state survive a redraw"
        panel.set_group_enabled("rings", False, "why")
        panel.add_layer("late", "Added while the group is disabled", items=[])
        assert not panel.checkbox("late").isEnabled() and "why" in panel.checkbox("late").toolTip()
        panel.set_label("late", "Renamed")
        assert panel.checkbox("late").text() == "Renamed"
        panel.clear_layers(keep_state=False)
        panel.add_group("Rings", key="rings")
        assert not panel.is_group_collapsed("rings")
        try:
            panel.add_group("again", key="rings")
        except ValueError:
            pass
        else:
            raise AssertionError("a second group under the same key must be refused")
        return "a disabled group says why, disables its rows and header, and Hide all skips it; state kept by key"

    def swatch_size_casing_scroll_hook() -> str:
        def image(sw: Any) -> Any:
            return lp.swatch_icon(sw).pixmap(16, 16).toImage()
        plain, cased = image(lp.Swatch("line", "#0072b2")), image(lp.Swatch("line", "#0072b2", cased=True))
        small, big = image(lp.Swatch("symbol", "#56b4e9", size=3)), image(lp.Swatch("symbol", "#56b4e9"))
        assert plain != cased and small != big
        assert not lp.swatch_icon(lp.Swatch("symbol", "#009e73", symbol="s", size=10, cased=True)).isNull()
        assert lp.Swatch("bar") == lp.Swatch("bar", size=0.0, cased=False), "the stage-0 defaults must not change"
        panel = lp.LayerPanel(scroll=True)
        assert panel.scroll_area is not None and lp.LayerPanel().scroll_area is None
        panel.add_group("Many", key="many")
        for i in range(60):
            panel.add_layer(f"k{i}", f"Layer {i}", items=[])
        panel.resize(240, 300)
        panel.show()
        pump(app, 0.1)
        inner = panel.scroll_area.widget()
        assert inner.height() > panel.scroll_area.viewport().height(), "the rows must scroll, not squeeze"
        panel.close()
        hooked = lp.LayerPanel()
        t1 = hooked.add_group("Legacy title")
        t2 = hooked.add_group("Boxed title", key="g")
        assert t1.styleSheet() == lp.group_title_style(False) == "font-weight: bold;"
        assert lp.LayerPanel(dark=True).add_group("x").styleSheet() == f"font-weight: bold; color: {lp.TITLE_FG};"
        hooked.set_title_style(lambda dark: "font-weight: bold; color: #123456;")
        assert "#123456" in t1.styleSheet() and "#123456" in t2.styleSheet()
        return "size and casing change the icon; 60 rows scroll; the title colour goes through one hook"

    check("groups: rows, tristate group check, fold, shown() names", groups_rows_and_toggle)
    check("groups: disabled with a reason, Hide all skips it, state by key through clear_layers",
          groups_enabled_and_state)
    check("swatch size and casing, scroll area, title-style hook (stage-0 defaults unchanged)",
          swatch_size_casing_scroll_hook)

    for key in ("w",):
        try:
            st[key].close()
        except Exception:  # noqa: BLE001
            pass
    pump(app, 0.2)
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.0f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
