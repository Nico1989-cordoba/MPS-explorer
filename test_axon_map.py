# -*- coding: utf-8 -*-
"""
Layer parity of the stage-2 builders (UI stage 2, design 8.4; ``tools/mps_axon_map_layers.py``).

The axon map, the single axial view and the nearest-neighbours tab are drawn from pure builders. Before any old plot
is removed (IMPL-C), this test proves that each builder hands its widget the SAME arrays the old drawing code drew:

  1. the main window on a SIMULATED axon (the golden's axon A: ``batch_columns.write_simulated_input`` + three
     interior clusters + synthetic widefield images), "cluster Ch1", the Axoplasm panel with both images (shift
     measured) -> a discard comparison; the Rings window;
  2. map vs R1 (MPS analysis window, "Contour and centre") for every analysis the radio can show; vs M4 (the main
     window's "Clusters centers"); vs A1 (the Axoplasm panel's image, edges, localizations by class, the four
     cluster groups, its contours and centre: the same objects, compared by value); vs G1 (the Rings window's
     superimposed segments);
  3. axial view vs Rz (slab bars' data, components and their means, slab), Gz (bands, centre lines, valleys and
     their labels), Gh (each segment's own z);
  4. nearest neighbours vs Rn (bars, median, the reference value), M5 (pooled 1st..3rd distances), Rc (the two
     CDFs and the crossing); "Save distances..." writes the bytes "save dist data" wrote, N = 1 and 3; CSV nn1_nm ==
     a.nn.first_nn_nm (%.2f) for every analysis the radio can show;
  5. titles and captions read from the drawn objects (an editor showing another value never reaches them), for each
     radio state, source and colouring; rows disabled with their reasons (discard only, segments' colours, a
     panel that classified other centres, a stale analysis, a missing channel 2).

R8: simulated axons and synthetic images only; every file in a temporary folder (settings, logs, cache, review
store, selection log); the user's settings are never read or written.

Run:  venv\\Scripts\\python.exe test_axon_map.py      (offscreen; about 1 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import json  # noqa: E402
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
WORK = tempfile.mkdtemp(prefix="test_axon_map_")
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-5:]:
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


def same(a: Any, b: Any) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def xy(item: Any) -> Tuple[np.ndarray, np.ndarray]:
    x, y = item.getData()
    return (np.asarray([] if x is None else x, float), np.asarray([] if y is None else y, float))


def layer_of(groups: List[Any], key: str) -> Any:
    for g in groups:
        for layer in g.layers:
            if layer.key == key:
                return layer
    raise KeyError(key)


def group_of(groups: List[Any], key: str) -> Any:
    for g in groups:
        if g.key == key:
            return g
    raise KeyError(key)


def scatter_items(plot: Any) -> List[Any]:
    import pyqtgraph as pg
    return [it for it in plot.getPlotItem().items if isinstance(it, pg.ScatterPlotItem)]


def line_items(plot: Any) -> List[Any]:
    import pyqtgraph as pg
    return [it for it in plot.getPlotItem().items if isinstance(it, pg.PlotDataItem)]


def pen_width(item: Any) -> float:
    return float(item.opts["pen"].widthF())


def main() -> int:
    print("=" * 100)
    print("STAGE-2 BUILDERS: LAYER PARITY WITH THE OLD DRAWING CODE (simulated axon A, offscreen)")
    print("=" * 100)
    from PyQt5 import QtCore, QtWidgets
    import pyqtgraph as pg
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from tools import mps_axon_map_layers as L
    from tools import mps_param_registry as reg
    st: Dict[str, Any] = {}

    # ------------------------------------------------------------------ 1. the main window
    print("\n1. The main window on simulated axon A, cluster Ch1, the Axoplasm panel with both images, Rings")

    def setup() -> str:
        import golden_ui_stage2 as gold
        inputs = os.path.join(WORK, "inputs")
        described = gold.make_inputs(inputs)
        st["inputs"], st["described"] = inputs, described
        os.chdir(WORK)                                     # cache/ and logs/ of the main window land here
        from tools import mps_settings
        settings_dir = os.path.join(WORK, "settings")
        os.makedirs(settings_dir, exist_ok=True)
        original = mps_settings.settings_path
        mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)  # type: ignore
        import MPS_explorer
        MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
        MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
        st["messages"] = []

        def box(kind: str) -> Callable[..., Any]:
            def show(*args: Any, **_kw: Any) -> Any:
                st["messages"].append((kind, str(args[1]) if len(args) > 1 else ""))
                return QtWidgets.QMessageBox.Ok
            return show
        for kind in ("information", "warning", "critical", "question"):
            setattr(QtWidgets.QMessageBox, kind, staticmethod(box(kind)))
        st["save_queue"] = []

        def save_name(*_a: Any, **_k: Any) -> Tuple[str, str]:
            return (st["save_queue"].pop(0) if st["save_queue"] else ""), "CSV Files (*.csv)"
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(save_name)  # type: ignore
        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(WORK, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="test",
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
        mw.ui.pushButton_clusterch1.click()
        pump(app, 0.05)
        w = mw.mps_window
        assert w is not None and w.analysis is not None
        mw.show_axoplasm_panel()
        aw = mw.axoplasm_window
        aw.load_tubulin(os.path.join(inputs, "wf_tubulin_A.tif"))
        aw.load_reference(os.path.join(inputs, "wf_spectrin_A.tif"))
        aw.flush()
        aw.measure()
        aw.wait(300)
        pump(app, 0.2)
        aw.flush()
        pump(app, 0.1)
        w.refresh()
        assert w.comparison is not None, "no discard comparison: the interior clusters were not discarded"
        ms = mw.run_ring_analysis(show_window=True)
        pump(app, 0.1)
        assert ms is not None and mw.rings_window is not None
        st.update(mw=mw, w=w, aw=aw, ms=mw.rings_window.ms, measured=w.analysis, comp=w.comparison)
        c = w.comparison
        return (f"{w.analysis.n_clusters_kept} clusters measured; discard applied keeps "
                f"{c.discard_applied.n_clusters_kept} ({len(c.discard_applied.discarded_labels)} discarded); "
                f"{len(mw.rings_window.ms.segments)} segments")

    check("simulated axon A through cluster Ch1, the Axoplasm panel (both images) and Rings", setup)

    def radios() -> List[Tuple[str, Any]]:
        w = need(st, "w")
        out = [("measured", w.radio_measured)]
        if w.radio_every.isVisible() or w._every_column():
            out.append(("every", w.radio_every))
        out.append(("discard", w.radio_discard))
        return out

    def map_inputs(shown: str, **kw: Any) -> Any:
        return L.MapInputs(analysis=need(st, "measured"), comparison=need(st, "comp"), shown=shown, **kw)

    # ------------------------------------------------------------------ 2. the map
    print("\n2. The axon map against R1, M4, A1 and G1")

    def r1_parity() -> str:
        w = need(st, "w")
        done = []
        for name, radio in radios():
            radio.setChecked(True)
            pump(app, 0.02)
            a = w._shown()
            groups = L.build_map(map_inputs(name))
            scat = scatter_items(w.plot_contour)
            lines = line_items(w.plot_contour)

            def by(size: float, symbol: Optional[str] = None) -> List[Any]:
                return [s for s in scat if float(s.opts["size"]) == size
                        and (symbol is None or s.opts["symbol"] == symbol)]
            for key, size, symbol in (("slab_noise", 2.0, None), ("slab_curated", 5.0, "x"),
                                      ("slab_discarded", 4.0, "d"), ("slab_kept", 3.0, None)):
                layer = layer_of(groups, key)
                old = by(size, symbol)
                if old:
                    ox, oy = xy(old[0])
                    assert same(layer.data["x"], ox) and same(layer.data["y"], oy), (name, key)
                    assert layer.count == len(ox), (name, key)
                else:
                    assert layer.count == 0 or not layer.enabled, (name, key, layer.count)
            kept = layer_of(groups, "c_kept")
            ox, oy = xy(by(7.0)[0])
            assert same(kept.data["x"], ox) and same(kept.data["y"], oy), (name, "c_kept")
            cdis = layer_of(groups, "c_discarded")
            old_dis = by(9.0, "d")
            if a.discard_applied:
                ox, oy = xy(old_dis[0])
                assert cdis.enabled and same(cdis.data["x"], ox) and same(cdis.data["y"], oy), name
                every_line = [ln for ln in lines if pen_width(ln) == 1.0]
                ex, ey = xy(every_line[0])
                ca = layer_of(groups, "contour_all")
                assert ca.enabled and same(ca.data["x"], ex) and same(ca.data["y"], ey), name
                cx, cy = xy(by(14.0, "+")[0])
                ce = layer_of(groups, "centre_all")
                assert same(ce.data["x"], cx) and same(ce.data["y"], cy), name
            else:
                assert not old_dis and not cdis.enabled and "discard applied" in cdis.reason, name
                assert not layer_of(groups, "contour_all").enabled
            contour = [ln for ln in lines if pen_width(ln) == 1.5]
            cx, cy = xy(contour[0])
            lay = layer_of(groups, "contour")
            assert same(lay.data["x"], cx) and same(lay.data["y"], cy), (name, "contour")
            runs = [xy(ln) for ln in lines if pen_width(ln) == 4.0]
            occ = layer_of(groups, "occupied").data["runs"]
            assert len(runs) == len(occ) and all(same(r[:, 0], ox) and same(r[:, 1], oy)
                                                 for r, (ox, oy) in zip(occ, runs)), (name, "occupied")
            px, py = xy(by(18.0, "+")[0])
            ctr = layer_of(groups, "centre")
            assert same(ctr.data["x"], px) and same(ctr.data["y"], py), (name, "centre")
            done.append(f"{name}: {len(runs)} occupied runs")
        need(st, "w").radio_measured.setChecked(True)
        return "; ".join(done) + "; every R1 layer equal for each radio state"

    def contour_by_value() -> str:
        # With nothing discarded the discard-applied contour is every.perimeter: equal in value to the panel's
        # contour, not the same object (3.3) - compared by value here, by identity nowhere.
        aw, comp = need(st, "aw"), need(st, "comp")
        found = aw.anchored
        new = found.contour_anchored
        lay = layer_of(L.build_map(map_inputs("discard")), "contour")
        cc = L.closed(new.contour)
        assert same(lay.data["x"], cc[:, 0]) and same(lay.data["y"], cc[:, 1])
        every = found.contour_all
        ce = L.closed(every.contour)
        e2 = L.closed(comp.all_clusters.perimeter.contour)
        assert same(ce, e2), "the panel's contour of every cluster and the analysis' differ"
        return "the discard-applied contour equals A1's blue contour; every-cluster contour equals A1's orange one"

    def m4_parity() -> str:
        mw = need(st, "mw")
        layout = mw.ui.scatterlayout_goodclus
        widget = layout.itemAt(layout.count() - 1).widget()
        plot = widget.ci.items if hasattr(widget, "ci") else None
        items = [it for p in (plot or []) for it in getattr(p, "items", []) if isinstance(it, pg.ScatterPlotItem)]
        assert items, "M4 drew no centres"
        ox, oy = xy(items[0])
        kept = layer_of(L.build_map(map_inputs("measured")), "c_kept")
        assert same(kept.data["x"], ox) and same(kept.data["y"], oy)
        return f"'Cluster centres > Kept' (measured) is M4's {len(ox)} green centres"

    def axoplasm_state(aw: Any) -> Any:
        return L.axoplasm_map_state(
            tubulin_image=None if aw.tubulin is None else aw.tubulin.image, tubulin_offset=aw.tubulin_offset,
            reference_image=None if aw.reference is None else aw.reference.image,
            reference_offset=aw.reference_offset, mask=aw.axoplasm, interior=aw.spectrin_interior,
            shift_px=aw._shift_px(), pixel_nm=aw.pixel_nm, loc_x=aw.inputs.loc.x_nm, loc_y=aw.inputs.loc.y_nm,
            located=aw.located, anchored=aw.anchored, anchored_centroids=aw.anchored_centroids,
            clusters=aw.inputs.clusters(), threshold_source=aw.axoplasm.threshold_source,
            selection_note=aw.selection_note, margin_nm=float(aw.anchored.margin_nm) if aw.anchored else None,
            registration="measured", has_result=aw.result is not None)

    def a1_parity() -> str:
        from tools import mps_axoplasm_window as axw
        from tools.mps_axon_map import cased_edge_rgba
        aw = need(st, "aw")
        aw.check_reference.setChecked(False)
        aw._draw()
        state = axoplasm_state(aw)
        st["ax_state"] = state
        inp = map_inputs("discard", axoplasm=state, source="selection", colour_by="images")
        groups = L.build_map(inp)
        for key, item in (("sel_inside", aw.interior_item), ("sel_membrane", aw.membrane_item),
                          ("sel_free", aw.free_item), ("c_both", aw.discarded_item),
                          ("c_tubulin", aw.tubulin_only_item), ("c_spectrin", aw.spectrin_only_item),
                          ("c_neither", aw.cluster_item)):
            ox, oy = xy(item)
            lay = layer_of(groups, key)
            assert same(lay.data["x"], ox) and same(lay.data["y"], oy), key
        img = layer_of(groups, "image")
        assert same(img.data["image"].T, aw.image_item.image), "tubulin image region"
        r = aw.image_item.mapRectToParent(aw.image_item.boundingRect())
        assert np.allclose(img.data["rect"], (r.x(), r.y(), r.width(), r.height())), "tubulin image rect"
        assert np.allclose(img.data["levels"], aw.image_item.levels), "levels p1-p99.7"
        edge = layer_of(groups, "tubulin_edge")
        assert same(np.transpose(axw._cased_edge(edge.data["edge"], axw._OUTLINE), (1, 0, 2)),
                    aw.outline_item.image), "tubulin edge"
        assert same(cased_edge_rgba(edge.data["edge"], axw._OUTLINE), axw._cased_edge(edge.data["edge"],
                                                                                       axw._OUTLINE))
        sedge = layer_of(groups, "spectrin_edge")
        assert same(np.transpose(axw._cased_edge(sedge.data["edge"], axw._SPECTRIN_OUTLINE), (1, 0, 2)),
                    aw.spectrin_outline_item.image), "spectrin edge"
        r2 = aw.spectrin_outline_item.mapRectToParent(aw.spectrin_outline_item.boundingRect())
        assert np.allclose(sedge.data["rect"], (r2.x(), r2.y(), r2.width(), r2.height()))
        # A1's contours and centre are the discard comparison's own objects
        cx, cy = xy(aw.contour_item)
        lay = layer_of(groups, "contour")
        assert same(lay.data["x"], cx) and same(lay.data["y"], cy), "A1's blue contour"
        ax_, ay_ = xy(aw.contour_all_item)
        la = layer_of(groups, "contour_all")
        assert same(la.data["x"], ax_) and same(la.data["y"], ay_), "A1's orange dashed contour"
        px, py = xy(aw.centre_item)
        assert same(layer_of(groups, "centre").data["x"], px) and same(layer_of(groups, "centre").data["y"], py)
        # the spectrin image, as A1's check box drew it
        aw.check_reference.setChecked(True)
        aw._draw()
        sp = layer_of(L.build_map(map_inputs("discard", axoplasm=state, source="selection", colour_by="images",
                                             image="spectrin")), "image")
        assert sp.data["which"] == "spectrin" and same(sp.data["image"].T, aw.image_item.image)
        r3 = aw.image_item.mapRectToParent(aw.image_item.boundingRect())
        assert np.allclose(sp.data["rect"], (r3.x(), r3.y(), r3.width(), r3.height()))
        aw.check_reference.setChecked(False)
        aw._draw()
        n = {k: layer_of(groups, k).count for k in ("sel_inside", "sel_membrane", "sel_free", "c_both", "c_tubulin",
                                                     "c_spectrin", "c_neither")}
        assert L.MAX_DRAWN_PER_CLASS == axw.MAX_DRAWN
        return "image (both), edges, 3 localization classes, 4 cluster groups, contours and centre equal: " + \
            ", ".join(f"{k} {v}" for k, v in n.items())

    def g1_parity() -> str:
        rw = need(st, "mw").rings_window
        ms = rw.ms
        groups = L.build_map(map_inputs("measured", rings=ms, source="segments"))
        rows = [layer for layer in group_of(groups, "localizations").layers]
        keys = rw.overlay_layers.keys()
        assert [r.key for r in rows] == keys, (keys, [r.key for r in rows])
        for r in rows:
            ox, oy = xy(rw.overlay_layers.items(r.key)[0])
            assert same(r.data["x"], ox) and same(r.data["y"], oy), r.key
            assert rw.overlay_layers.checkbox(r.key).text() == f"{r.label} ({r.count})", r.key
        for key in L.SEGMENTS_DISABLE:
            for g in groups:
                for layer in g.layers:
                    if layer.key == key:
                        assert not layer.enabled and layer.reason == L.SEGMENT_COLOUR_REASON, key
        return f"{len(rows)} segment rows: same keys, labels, counts and points as the rings panel"

    check("map vs R1 (each radio state): every layer's arrays", r1_parity)
    check("map's contours by value with nothing to compare by identity", contour_by_value)
    check("map vs M4 (main window's cluster centres)", m4_parity)
    check("map vs A1 (image, edges, classes, groups, contours, centre)", a1_parity)
    check("map vs G1 (rings window's superimposed segments)", g1_parity)

    # ------------------------------------------------------------------ 3. the axial view
    print("\n3. The axial view against Rz, Gz and Gh")

    def rz_parity() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        out = []
        for name, radio in radios():
            radio.setChecked(True)
            pump(app, 0.02)
            a = w._shown()
            inp = L.AxialInputs(z_roi=mw.zroi_unfiltered, analysis=a, cut=(float(mw.zmin), float(mw.zmax)))
            groups = L.build_axial(inp)
            items = w.plot_z.getPlotItem().items
            bars = [it for it in items if isinstance(it, pg.BarGraphItem)][0]
            roi_in = layer_of(groups, "roi_in")
            assert same(np.sort(roi_in.data["z"]), np.sort(a.z_slab)), "the slab's own z"
            counts, edges = np.histogram(roi_in.data["z"], bins=60, density=True)
            assert np.allclose(bars.opts["height"], counts) and np.allclose(bars.opts["x"],
                                                                            (edges[:-1] + edges[1:]) / 2)
            curves = [it for it in items if isinstance(it, pg.PlotDataItem)]
            comp = layer_of(groups, "components")
            assert len(curves) == len(comp.data["curves"])
            for old, new in zip(curves, comp.data["curves"]):
                gx, gy = xy(old)
                assert same(gx, comp.data["grid"]) and same(gy, new)
            means = [float(it.value()) for it in items if isinstance(it, pg.InfiniteLine)]
            assert np.allclose(means, comp.data["means"])
            region = [it for it in items if isinstance(it, pg.LinearRegionItem)][0]
            slab = layer_of(groups, "slab")
            assert np.allclose(region.getRegion(), (slab.data["lo"], slab.data["hi"]))
            total = roi_in.count + layer_of(groups, "roi_out").count
            assert total == int(np.isfinite(mw.zroi_unfiltered).sum()) == a.n_locs_total
            # one scale: the bars integrate to 1 over the ROI, as the mixture does
            d = roi_in.data["height"] + layer_of(groups, "roi_out").data["height"]
            assert abs(float(np.sum(d) * roi_in.data["width"]) - 1.0) < 1e-9
            out.append(f"{name}: {len(curves)} components")
        w.radio_measured.setChecked(True)
        return "; ".join(out) + "; slab z, curves, means and band equal; bars and mixture on one scale"

    def gz_gh_parity() -> str:
        mw = need(st, "mw")
        rw = mw.rings_window
        ms = rw.ms
        inp = L.AxialInputs(z_roi=mw.zroi_unfiltered, analysis=None, rings=ms, view="segments",
                            cut=(float(mw.zmin), float(mw.zmax)))
        groups = L.build_axial(inp)
        items = rw.plot_z.getPlotItem().items
        regions = [it for it in items if isinstance(it, pg.LinearRegionItem)]
        lines = [it for it in items if isinstance(it, pg.InfiniteLine)]
        bands = [layer for layer in group_of(groups, "segments").layers if layer.key.startswith("segband")]
        assert len(regions) == len(bands)
        for reg_item, band in zip(regions, bands):
            assert np.allclose(reg_item.getRegion(), (band.data["lo"], band.data["hi"]))
        seg_lines = lines[:len(bands)]
        for ln, band in zip(seg_lines, bands):
            assert float(ln.value()) == band.data["line"] and ln.label.format == band.data["label"]
        valleys = layer_of(groups, "valleys")
        vlines = lines[len(bands):]
        assert np.allclose([float(v.value()) for v in vlines], valleys.data["positions"])
        assert [v.label.format for v in vlines] == valleys.data["labels"]
        styles = ["solid" if v.pen.style() == QtCore.Qt.PenStyle.SolidLine else "dash" for v in vlines]
        assert styles == valleys.data["pens"]
        mix = [it for it in items if isinstance(it, pg.PlotDataItem)][0]
        gx, gy = xy(mix)
        assert np.allclose(gy, ms.z_result.mixture_density(gx)), "the same mixture"
        assert not layer_of(groups, "components").enabled, "components disabled in Every segment"
        # Gh: each segment's own z (40-bin counts of the same array)
        plots = [p for p in rw.zhist_grid.ci.items]
        hists = [layer for layer in group_of(groups, "segments").layers if layer.key.startswith("seghist")]
        assert len(plots) == len(hists)
        for p, h in zip(plots, hists):
            bar = [it for it in p.items if isinstance(it, pg.BarGraphItem)][0]
            counts, _e = np.histogram(h.data["z"], bins=40)
            assert same(bar.opts["height"], counts), h.key
            inf = [it for it in p.items if isinstance(it, pg.InfiniteLine)]
            assert np.allclose([float(i.value()) for i in inf], h.data["bounds"]), h.key
        return (f"{len(bands)} bands and centre lines, {len(vlines)} boundaries with their labels and styles, the "
                f"mixture, {len(hists)} per-segment z equal")

    check("axial view vs Rz (each radio state)", rz_parity)
    check("axial view vs Gz and Gh", gz_gh_parity)

    # ------------------------------------------------------------------ 4. nearest neighbours
    print("\n4. Nearest neighbours against Rn, M5, Rc and 'save dist data'")

    def rn_rc_parity() -> str:
        w = need(st, "w")
        for name, radio in radios():
            radio.setChecked(True)
            pump(app, 0.02)
            a = w._shown()
            hist = L.nn_histogram(a, 1, 30)
            items = w.plot_nn.getPlotItem().items
            bars = [it for it in items if isinstance(it, pg.BarGraphItem)][0]
            nl = L.nn_layers(a, hist)
            nb = [x for x in nl if x.key == "nn_bars"][0]
            assert same(bars.opts["height"], nb.data["height"]) and same(bars.opts["x"], nb.data["x"]), name
            lines = [float(it.value()) for it in items if isinstance(it, pg.InfiniteLine)]
            med = [x for x in nl if x.key == "nn_median"][0]
            ref = [x for x in nl if x.key == "nn_reference"][0]
            assert lines == med.data["positions"] + ref.data["positions"], (name, lines)
            cdf_items = w.plot_cdf.getPlotItem().items
            curves = {it.opts["name"]: xy(it) for it in cdf_items if isinstance(it, pg.PlotDataItem)}
            cl = {x.key: x for x in L.cdf_layers(a)}
            for key, nm in (("cdf_observed", "observed"), ("cdf_randomized", "randomized")):
                assert same(cl[key].data["x"], curves[nm][0]) and same(cl[key].data["y"], curves[nm][1]), (name, key)
            cross = [float(it.value()) for it in cdf_items if isinstance(it, pg.InfiniteLine)]
            if "cdf_crossing" in cl:
                assert cross == [cl["cdf_crossing"].data["y"]], name
                assert L.style_of("cdf_crossing").role == "neutral" and "(heuristic)" in cl["cdf_crossing"].label
        w.radio_measured.setChecked(True)
        return "bars, median and reference value equal Rn; the two CDFs and the crossing equal Rc, for each radio"

    def m5_parity() -> str:
        mw = need(st, "mw")
        a = need(st, "measured")
        mw.ui.lineEdit_Nneighbor.setText("3")
        mw.KNdist_hist()
        pooled = L.knn_distances(a.centroids, 3)
        assert same(pooled, mw.distances), "the same KD-tree distances"
        h = L.nn_histogram(a, 3, int(mw.bins))
        assert same(np.sort(h["values"]), np.sort(np.asarray(mw.distances).ravel()))
        layout = mw.ui.zhistlayout_cmdist
        widget = layout.itemAt(layout.count() - 1).widget()
        bars = [it for p in widget.ci.items for it in getattr(p, "items", []) if isinstance(it, pg.BarGraphItem)][0]
        d = np.asarray(mw.distances)
        shown = d[(d > mw.lmin) & (d < mw.lmax)]
        counts, edges = np.histogram(shown, int(mw.bins))
        assert same(bars.opts["height"], counts)
        if shown.size == d.size:
            assert same(h["counts"], counts) and same(h["edges"], edges), "nothing outside: the same histogram"
        hr = L.nn_histogram(a, 3, int(mw.bins), (float(mw.lmin), float(mw.lmax)))
        assert hr["outside"] == int(np.sum((d < mw.lmin) | (d > mw.lmax)))
        assert L.nn_max_neighbours(a.n_clusters_kept) == min(10, a.n_clusters_kept - 1)
        return f"pooled 1st..3rd distances = M5's ({d.size}); the bars equal M5's; N capped at clusters - 1"

    def csv_bytes() -> str:
        mw = need(st, "mw")
        a = need(st, "measured")
        out = []
        for n in (1, 3):
            mw.ui.lineEdit_Nneighbor.setText(str(n))
            mw.KNdist_hist()
            old = os.path.join(WORK, f"old_{n}.csv")
            st["save_queue"].append(old)
            mw.savedistdata()
            new = os.path.join(WORK, f"new_{n}.csv")
            L.write_distances_csv(new, a, a, n)
            with open(old, "rb") as f1, open(new, "rb") as f2:
                b1, b2 = f1.read(), f2.read()
            assert b1 == b2, f"N={n}: the bytes differ"
            out.append(f"N={n}: {len(b1)} bytes")
        root = mw.get_root_filename()
        assert L.default_distances_name(root, 3) == f"{root}_3neighbor_distances.csv"
        return "; ".join(out) + " identical to 'save dist data'"

    def nn1_csv_parity() -> str:
        import csv
        w, comp, a = need(st, "w"), need(st, "comp"), need(st, "measured")
        out = []
        for name, radio in radios():
            radio.setChecked(True)
            pump(app, 0.02)
            shown = w._shown()
            path = os.path.join(WORK, f"nn1_{name}.csv")
            L.write_distances_csv(path, shown, a, 1)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            col = [r["nn1_nm"] for r in rows]
            want = [f"{v:.2f}" for v in shown.nn.first_nn_nm]
            assert col == want, (name, col[:3], want[:3])
            labels = [int(r["cluster_label"]) for r in rows]
            assert len(labels) == shown.n_clusters_kept and not set(labels) & set(shown.discarded_labels)
            out.append(f"{name}: {len(col)} rows")
            assert L.default_distances_name("x", 1, shown, a) == {
                "measured": "x_1neighbor_distances.csv", "every": "x_1neighbor_distances_all_clusters.csv",
                "discard": "x_1neighbor_distances_discard_applied.csv"}[name]
        w.radio_measured.setChecked(True)
        return "CSV nn1_nm == a.nn.first_nn_nm (%.2f) for " + "; ".join(out)

    check("nearest neighbours vs Rn and Rc (each radio state)", rn_rc_parity)
    check("pooled distances vs M5 (N = 3)", m5_parity)
    check("'Save distances...' writes the bytes 'save dist data' wrote (N = 1, 3)", csv_bytes)
    check("CSV nn1_nm == a.nn.first_nn_nm for every analysis the radio can show", nn1_csv_parity)

    # ------------------------------------------------------------------ 5. titles, captions, reasons
    print("\n5. Titles and captions from the drawn objects; disabled rows with their reasons")

    def titles() -> str:
        import dataclasses
        comp, a, ms = need(st, "comp"), need(st, "measured"), need(st, "ms")
        every_name = [n for n, _r in radios()]
        t_meas = L.map_title(map_inputs("measured"), L.VIEW_ON["mps"])
        assert t_meas.startswith(f"MPS analysis, measured: {a.n_clusters_kept} clusters<br>localizations: MPS "
                                 "analysis slab"), t_meas
        assert "centres by status in the analysis" in t_meas
        d = comp.discard_applied
        t_dis = L.map_title(map_inputs("discard"), L.VIEW_ON["mps"])
        assert t_dis.startswith(f"Discard applied: {d.n_clusters_kept} of {comp.all_clusters.n_clusters_kept} "
                                f"clusters, {len(d.discarded_labels)} discarded"), t_dis
        if "every" in every_name:
            assert L.map_title(map_inputs("every"), set()).startswith("All ")
        t_ax = L.map_title(map_inputs("discard", axoplasm=need(st, "ax_state"), source="selection",
                                      colour_by="images"), L.VIEW_ON["axoplasm"])
        assert "localizations: main-window cut, axoplasm classes" in t_ax and "image: betaIII-tubulin" in t_ax
        assert "centres by the two widefield images" in t_ax
        t_seg = L.map_title(map_inputs("measured", rings=ms, source="segments"), {f"seg{ms.segments[0].index}"})
        assert f"each segment's slab, mode {ms.mode}" in t_seg
        t_stale = L.map_title(map_inputs("measured", stale=True), set())
        assert t_stale == f"MPS analysis, measured: {a.n_clusters_kept} clusters (previous selection)"
        assert L.map_title(L.MapInputs(), set()) == "No MPS analysis of this selection yet"
        # axial titles: three lines, the last the z-calibration line
        mw = need(st, "mw")
        at = L.axial_title(L.AxialInputs(z_roi=mw.zroi_unfiltered, analysis=a, cut=(1.0, 2.0)))
        lines = at.split("<br>")
        assert len(lines) == 3 and lines[0].startswith("z of the ROI before the cut (") and \
            lines[1].startswith(f"MPS analysis slab {a.slab_zmin_nm:,.0f}..") and \
            lines[2] == "z as fitted; no z-calibration record read", at
        st_ = L.axial_title(L.AxialInputs(z_roi=mw.zroi_unfiltered, rings=ms, view="segments"))
        assert f"segments: mode {ms.mode}, guard" in st_ and "(as computed)" in st_
        # nearest neighbours: N and the analysis shown, only when a comparison exists
        h1 = L.nn_histogram(a, 1)
        assert L.nn_title(h1, a, None, a, None) == "1NN distance between cluster centres"
        assert L.nn_title(h1, a, comp, a, None) == "1NN distance between cluster centres (measured)"
        h3 = L.nn_histogram(d, 3, 30, (0.0, 100.0))
        t3 = L.nn_title(h3, d, comp, a, (0.0, 100.0))
        assert t3.startswith("Distances to the 1st..3rd nearest centre, pooled (N = 3; discard applied") and \
            "outside the range not shown" in t3, t3
        assert L.cdf_title(d, comp, a).startswith("1NN CDF: observed vs randomized (discard applied")
        # a caption reads the drawn object, never an editor
        w = need(st, "w")
        w.spin_eps.setValue(41.0)                           # an editor showing another value, never applied
        other = dataclasses.replace(a, eps_nm=30.0)
        cap = group_of(L.build_map(L.MapInputs(analysis=other)), "localizations").caption
        assert "eps 30 nm [user]" in cap and "41" not in cap, cap
        cap_a = group_of(L.build_map(L.MapInputs(analysis=a)), "localizations").caption
        assert f"eps {a.eps_nm:g} nm," in cap_a and "[user]" not in cap_a.split(";")[1], cap_a
        w.spin_eps.setValue(a.eps_nm)
        cc = group_of(L.build_map(map_inputs("discard")), "contour").caption
        assert f"for {d.occupancy.n_capped} of {len(d.occupancy.ellipses)} ellipses" in cc and \
            "2-opt from every start" in cc, cc
        seg_cap = group_of(L.build_map(map_inputs("measured", rings=ms, source="segments")), "localizations").caption
        an0 = next(x for x in ms.analyses if x is not None)
        assert f"eps {an0.eps_nm:g} nm" in seg_cap and f"Mahalanobis {an0.mahalanobis_threshold:g}" in seg_cap
        line2 = L.provenance_line2(a, roi_words="ROI circle", cut=(float(mw.zmin), float(mw.zmax)))
        assert line2.endswith("z as fitted, no z-calibration record read") and \
            f"{a.n_locs_total:,} localizations given to the analysis" in line2
        return "map (3 radio states, 3 sources, stale, none), axial (2 views), NN (N, range), CDF, captions"

    def reasons() -> str:
        import dataclasses
        a, comp, state = need(st, "measured"), need(st, "comp"), need(st, "ax_state")
        g = L.build_map(map_inputs("measured"))
        assert "discard applied" in layer_of(g, "slab_discarded").reason
        moved = dataclasses.replace(state, anchored_centroids=np.asarray(state.anchored_centroids) + 1.0)
        gi = group_of(L.build_map(map_inputs("discard", axoplasm=moved, colour_by="images")), "centres")
        assert not gi.enabled and "other clusters" in gi.reason
        gs = group_of(L.build_map(map_inputs("discard", axoplasm=state, source="selection", stale=True)),
                      "localizations")
        assert not gs.enabled and "previous" in gs.reason
        gn = group_of(L.build_map(map_inputs("measured", source="selection")), "localizations")
        assert not gn.enabled and gn.reason == L.MapInputs().axoplasm_reason
        gr = group_of(L.build_map(map_inputs("measured")), "rings")
        assert not gr.enabled and "Rings" in gr.reason
        ax = L.build_axial(L.AxialInputs(z_roi=need(st, "mw").zroi_unfiltered, analysis=a))
        ch2 = layer_of(ax, "ch2_roi")
        assert not ch2.enabled and "channel 2" in ch2.reason
        z2 = np.asarray(need(st, "mw").zroi_unfiltered) + 30.0
        ax2 = L.build_axial(L.AxialInputs(z_roi=need(st, "mw").zroi_unfiltered, analysis=a, ch2_z=z2,
                                          ch2_shown=True))
        assert layer_of(ax2, "ch2_roi").enabled and not layer_of(ax2, "components").enabled
        assert layer_of(ax2, "mixture").enabled
        stale = L.build_axial(L.AxialInputs(z_roi=need(st, "mw").zroi_unfiltered, analysis=a, cut=(1.0, 2.0),
                                            cut_stale=True))
        assert not layer_of(stale, "main_cut").enabled
        none = L.build_axial(L.AxialInputs(z_roi=need(st, "mw").zroi_unfiltered, fit=a.z_result, cut=(1.0, 2.0)))
        assert layer_of(none, "roi_out").label == "All" and not any(
            layer.key in ("roi_in", "slab") for grp in none for layer in grp.layers)
        h3 = L.nn_layers(a, L.nn_histogram(a, 3))
        assert all(not x.enabled and x.reason == L.NN_FIRST_ONLY for x in h3 if x.key != "nn_bars")
        assert "nn_reference" not in L.NN_ON, "the reference layer starts off (P-R25)"
        details = dict((r[0], r) for r in L.axial_details(L.AxialInputs(analysis=a, cut=(1.0, 2.0))))
        assert details["density peak"][3] == "derived" and "components chosen by BIC" in details
        cd = [r[0] for r in L.cdf_details(a)]
        assert not any("p" == x.split()[-1] or "p-value" in x or "KS p" in x for x in cd), cd
        assert reg.badge("randomization.band_half_width_nm", a.randomization.annulus_half_width_nm) == "paper"
        return ("discard-only rows, images colouring of other clusters, stale analysis, no panel, no rings, no "
                "channel 2, channel 2 hides the components, stale cut, empty state, N > 1; no KS p in the details")

    check("titles: map, axial, nearest neighbours, CDF; captions from the drawn objects", titles)
    check("disabled rows and groups say why", reasons)

    for key in ("aw", "w"):
        try:
            st[key].close()
        except Exception:  # noqa: BLE001
            pass
    try:
        need(st, "mw").rings_window.close()
        need(st, "mw").close()
    except Exception:  # noqa: BLE001
        pass
    pump(app, 0.2)
    os.chdir(REPO_ROOT)
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.0f} s; work folder {WORK}")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    _ = json
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
