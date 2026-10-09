# -*- coding: utf-8 -*-
"""
Layer parity of the stage-2 builders (UI stage 2, design 8.4; ``tools/mps_axon_map_layers.py``).

The axon map, the single axial view and the nearest-neighbours tab are drawn from pure builders. This test proves
that each builder hands its widget the SAME arrays the old drawing code drew. The old drawing code was removed from
the windows in IMPL-C (6c1ee21..9d1e86c); it is kept HERE as frozen test oracles, copied verbatim from 01e603f
(the ``old_*`` functions and ``OldAxoplasmImage`` below: R1, Rz, Rn, Rc of the MPS analysis window, A1 of the
Axoplasm panel, G1, Gz, Gh of the Rings window, M4 and M5 + "save dist data" of the main window), each drawing into
a fresh pyqtgraph widget from the same inputs the window used to draw from. Never edit an oracle to make a check
pass: a difference is a regression of a builder.

  1. the main window on a SIMULATED axon (the golden's axon A: ``batch_columns.write_simulated_input`` + three
     interior clusters + synthetic widefield images), "cluster Ch1", the Axoplasm panel with both images (shift
     measured) -> a discard comparison; the Rings window;
  2. map vs R1 (MPS analysis window, "Contour and centre") for every analysis the radio can show; vs M4 (the main
     window's "Clusters centers"); vs A1 (the Axoplasm panel's image, edges, localizations by class, the four
     cluster groups, its contours and centre: the same objects, compared by value; also through the panel's own
     ``map_state()``); vs G1 (the Rings window's superimposed segments); and the windows that replaced them really
     draw from the builders (the MPS window's map, axial view and nearest-neighbours tab);
  3. axial view vs Rz (slab bars' data, components and their means, slab), Gz (bands, centre lines, valleys and
     their labels), Gh (each segment's own z);
  4. nearest neighbours vs Rn (bars, median, the reference value), M5 (pooled 1st..3rd distances), Rc (the two
     CDFs and the crossing); "Save distances..." (the MPS window's tab and ``write_distances_csv``) writes the bytes
     "save dist data" wrote, N = 1 and 3; CSV nn1_nm == a.nn.first_nn_nm (%.2f) for every analysis the radio can
     show;
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


# ============================================================================================ frozen oracles
# The old drawing code, copied verbatim from 01e603f (``git show 01e603f:<file>``); ``self.<plot>`` became a fresh
# widget and ``self.<attr>`` a parameter. Only what decides the drawn arrays matters to the checks, but the bodies
# are kept whole (colours, sizes, symbols, pens) so the checks can still tell the items apart the way they did.

def old_r1(plot: Any, a: Any, analysis: Any, comparison: Any, dark: bool = True) -> None:
    """R1: tools/mps_results_window.py MPSResultsWindow._draw_contour at 01e603f (without its title).
    ``a`` = self._shown(), ``analysis`` = self.analysis (measured), ``comparison`` = self.comparison."""
    import pyqtgraph as pg
    from PyQt5 import QtCore
    from tools.cluster_quality import good_cluster_labels
    from tools.mps_plot_style import neutral, rgba, role

    def _neutral() -> str:
        return neutral(dark=dark)
    plot.clear()
    if a.x_slab.size == 0:
        return

    bad = a.bad_report.bad_labels
    noise = a.labels == -1
    if np.any(noise):
        plot.addItem(pg.ScatterPlotItem(
            a.x_slab[noise], a.y_slab[noise], size=2, pen=None,
            brush=pg.mkBrush(*rgba("noise", 90)), name="noise"))

    bad_mask = np.isin(a.labels, list(bad)) if bad else np.zeros_like(noise)
    if np.any(bad_mask):
        plot.addItem(pg.ScatterPlotItem(
            a.x_slab[bad_mask], a.y_slab[bad_mask], size=5, symbol="x",
            pen=pg.mkPen(*rgba("curated", 160)), brush=None))

    gone = (np.isin(a.labels, list(a.discarded_labels))
            if a.discarded_labels else np.zeros_like(noise))
    if np.any(gone):
        plot.addItem(pg.ScatterPlotItem(
            a.x_slab[gone], a.y_slab[gone], size=4, symbol="d",
            pen=None, brush=pg.mkBrush(role("discarded"))))

    good_mask = (~noise) & (~bad_mask) & (~gone)
    if np.any(good_mask):
        plot.addItem(pg.ScatterPlotItem(
            a.x_slab[good_mask], a.y_slab[good_mask], size=3, pen=None,
            brush=pg.mkBrush(*rgba("locs", 200))))

    if a.discard_applied and comparison is not None:
        every = comparison.all_clusters.perimeter
        if every is not None:
            closed = np.vstack([every.contour, every.contour[:1]])
            plot.addItem(pg.PlotDataItem(
                closed[:, 0], closed[:, 1],
                pen=pg.mkPen(_neutral(), width=1,
                             style=QtCore.Qt.PenStyle.DashLine)))
            if every.centre is not None:
                plot.addItem(pg.ScatterPlotItem(
                    [every.centre.x_nm], [every.centre.y_nm], size=14,
                    symbol="+", pen=pg.mkPen(_neutral()),
                    brush=pg.mkBrush(_neutral())))
        labels = good_cluster_labels(analysis.labels,
                                     analysis.bad_report.bad_labels)
        out = np.isin(labels, list(a.discarded_labels))
        if np.any(out):
            c = np.asarray(analysis.centroids)[out]
            plot.addItem(pg.ScatterPlotItem(
                c[:, 0], c[:, 1], size=9, symbol="d",
                pen=pg.mkPen(_neutral()),
                brush=pg.mkBrush(role("discarded"))))

    if a.perimeter is not None:
        c = a.perimeter.contour
        closed = np.vstack([c, c[:1]])
        plot.addItem(pg.PlotDataItem(
            closed[:, 0], closed[:, 1],
            pen=pg.mkPen(_neutral(), width=1.5)))

    if a.occupancy is not None:
        pts = a.occupancy.perimeter_points
        occ = a.occupancy.occupied_mask
        if np.any(occ):
            idx = np.flatnonzero(occ)
            breaks = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate([[0], breaks + 1])
            ends = np.concatenate([breaks, [len(idx) - 1]])
            for s, e in zip(starts, ends):
                run = pts[idx[s]:idx[e] + 1]
                if len(run) >= 2:
                    plot.addItem(pg.PlotDataItem(
                        run[:, 0], run[:, 1],
                        pen=pg.mkPen(role("occupied"), width=4)))

    if a.centroids.size:
        plot.addItem(pg.ScatterPlotItem(
            a.centroids[:, 0], a.centroids[:, 1], size=7,
            pen=pg.mkPen(_neutral()),
            brush=pg.mkBrush(role("centroid"))))

    if a.centre is not None:
        plot.addItem(pg.ScatterPlotItem(
            [a.centre.x_nm], [a.centre.y_nm], size=18, symbol="+",
            pen=pg.mkPen(role("centre"), width=2),
            brush=pg.mkBrush(role("centre"))))


def old_rz(plot: Any, a: Any) -> None:
    """Rz: MPSResultsWindow._draw_z at 01e603f; ``a`` = self._shown()."""
    import pyqtgraph as pg
    from PyQt5 import QtCore
    from tools.mps_plot_style import rgba, role
    plot.clear()
    zr = a.z_result
    if a.z_slab.size:
        counts, edges = np.histogram(a.z_slab, bins=60, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        width = float(np.mean(np.diff(edges)))
        plot.addItem(pg.BarGraphItem(
            x=centres, height=counts, width=width,
            brush=pg.mkBrush(*rgba("locs", 170)), pen=None))

    lo, hi = a.slab_zmin_nm, a.slab_zmax_nm
    span = max(hi - lo, 1.0)
    grid = np.linspace(lo - 2 * span, hi + 2 * span, 1024)
    for m, wgt, s in zip(zr.means_nm, zr.weights, zr.sigmas_nm):
        if s <= 0:
            continue
        dens = wgt * np.exp(-0.5 * ((grid - m) / s) ** 2) / (
            s * np.sqrt(2 * np.pi))
        plot.addItem(pg.PlotDataItem(
            grid, dens, pen=pg.mkPen(role("fit"), width=2,
                                     style=QtCore.Qt.DashLine)))
        plot.addItem(pg.InfiniteLine(
            pos=float(m), angle=90,
            pen=pg.mkPen(role("fit"), width=1,
                         style=QtCore.Qt.DotLine)))

    region = pg.LinearRegionItem(values=(lo, hi), movable=False)
    region.setBrush(pg.mkBrush(*rgba("slab", 60)))
    region.setZValue(-10)
    plot.addItem(region)


def old_rn(plot: Any, a: Any) -> None:
    """Rn: MPSResultsWindow._draw_nn at 01e603f; ``a`` = self._shown()."""
    import pyqtgraph as pg
    from PyQt5 import QtCore
    from tools.mps_plot_style import rgba, role
    plot.clear()
    if a.nn is None or a.nn.first_nn_nm.size == 0:
        return
    vals = a.nn.first_nn_nm
    counts, edges = np.histogram(vals, bins=30)
    centres = (edges[:-1] + edges[1:]) / 2
    width = float(np.mean(np.diff(edges)))
    plot.addItem(pg.BarGraphItem(
        x=centres, height=counts, width=width,
        brush=pg.mkBrush(*rgba("locs", 190)), pen=None))
    med = a.nn.median_1nn_nm
    if med is not None:
        plot.addItem(pg.InfiniteLine(
            pos=med, angle=90, pen=pg.mkPen(role("summary"), width=2),
            label=f"median {med:,.0f} nm",
            labelOpts={"position": 0.9, "color": role("summary")}))
    plot.addItem(pg.InfiniteLine(
        pos=260.0, angle=90,
        pen=pg.mkPen(role("paper"), width=2, style=QtCore.Qt.DashLine),
        label="paper 260 nm",
        labelOpts={"position": 0.75, "color": role("paper")}))


def old_rc(plot: Any, a: Any) -> None:
    """Rc: MPSResultsWindow._draw_cdf at 01e603f; ``a`` = self._shown()."""
    import pyqtgraph as pg
    from PyQt5 import QtCore
    from tools.mps_plot_style import role
    plot.clear()
    r = a.randomization
    if r is None:
        return

    def cdf(v: Any) -> Tuple[np.ndarray, np.ndarray]:
        v = np.sort(np.asarray(v, dtype=float))
        return v, np.arange(1, v.size + 1) / v.size

    xe, ye = cdf(r.experimental_1nn_nm)
    xr, yr = cdf(r.randomized_1nn_nm)
    plot.addItem(pg.PlotDataItem(
        xr, yr, pen=pg.mkPen(role("randomized"), width=2),
        name="randomized"))
    plot.addItem(pg.PlotDataItem(
        xe, ye, pen=pg.mkPen(role("observed"), width=2),
        name="observed"))

    if r.cdf_crossing is not None:
        line = pg.InfiniteLine(
            pos=r.cdf_crossing, angle=0,
            pen=pg.mkPen(role("paper"), width=1,
                         style=QtCore.Qt.DashLine),
            label=f"crossing {r.cdf_crossing:.2f}",
            labelOpts={"position": 0.05, "color": role("paper")})
        plot.addItem(line)


# A1: tools/mps_axoplasm_window.py at 01e603f (module constants, _cased_edge, _region_rect, _draw's image part).
OLD_MAX_DRAWN = 20000


def _old_outline_colours() -> Tuple[Any, Any]:
    from tools.mps_plot_style import rgba
    return rgba("image_tubulin", 255), rgba("image_spectrin", 255)   # _OUTLINE, _SPECTRIN_OUTLINE


def old_cased_edge(edge: "np.ndarray", colour: tuple) -> "np.ndarray":
    """_cased_edge at 01e603f."""
    from scipy import ndimage
    rim = ndimage.binary_dilation(edge) & ~edge
    rgba = np.zeros(edge.shape + (4,), dtype=np.ubyte)
    rgba[rim] = (0, 0, 0, 200)
    rgba[edge] = colour
    return rgba


class OldAxoplasmImage:
    """A1, the Axoplasm panel's image as AxoplasmWindow._draw drew it at 01e603f: the same items, in a fresh plot,
    drawn from the panel's model attributes (``aw``). ``show_reference`` stands for ``check_reference``."""

    def __init__(self) -> None:
        import pyqtgraph as pg
        from PyQt5 import QtCore
        from tools.mps_plot_style import neutral, role
        self.plot_image = pg.PlotWidget()
        self.image_item = pg.ImageItem()
        self.outline_item = pg.ImageItem()
        self.spectrin_outline_item = pg.ImageItem()
        self.membrane_item = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(role("locs")), size=2)
        self.interior_item = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(role("discarded")), size=2)
        self.contour_all_item = pg.PlotDataItem(pen=pg.mkPen(role("contour_all"), width=2,
                                                             style=QtCore.Qt.PenStyle.DashLine))
        self.contour_item = pg.PlotDataItem(pen=pg.mkPen(role("contour_kept"), width=2))
        self.centre_item = pg.ScatterPlotItem(pen=pg.mkPen("k", width=2), brush=pg.mkBrush(neutral(dark=True)),
                                              size=18, symbol="+")
        self.cluster_item = pg.ScatterPlotItem(pen=pg.mkPen(neutral(dark=True)), brush=None, size=9)
        self.tubulin_only_item = pg.ScatterPlotItem(pen=pg.mkPen(role("image_tubulin"), width=2), brush=None,
                                                    size=10)
        self.spectrin_only_item = pg.ScatterPlotItem(pen=pg.mkPen(role("image_spectrin"), width=2), brush=None,
                                                     size=10, symbol="s")
        self.discarded_item = pg.ScatterPlotItem(pen=pg.mkPen("k"), brush=pg.mkBrush(role("discarded")), size=10)
        self.free_item = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(neutral(dark=True)), size=2)
        for item in (self.image_item, self.outline_item, self.spectrin_outline_item, self.membrane_item,
                     self.interior_item, self.free_item, self.cluster_item, self.tubulin_only_item,
                     self.spectrin_only_item, self.discarded_item, self.contour_all_item, self.contour_item,
                     self.centre_item):
            self.plot_image.addItem(item)

    @staticmethod
    def _region_rect(aw: Any, rows: Tuple[int, int], cols: Tuple[int, int], offset: Tuple[float, float]) -> Any:
        from PyQt5 import QtCore
        sx, sy = aw._shift_px()
        px = aw.pixel_nm
        return QtCore.QRectF(
            (cols[0] - 0.5 - offset[0] - sx) * px,
            (rows[0] - 0.5 - offset[1] - sy) * px,
            (cols[1] - cols[0]) * px, (rows[1] - rows[0]) * px)

    def draw(self, aw: Any, show_reference: bool) -> None:
        from scipy import ndimage
        from tools import mps_axoplasm as ax
        _OUTLINE, _SPECTRIN_OUTLINE = _old_outline_colours()
        mask = aw.axoplasm
        for item in (self.membrane_item, self.interior_item, self.free_item,
                     self.cluster_item, self.tubulin_only_item,
                     self.spectrin_only_item, self.discarded_item,
                     self.contour_all_item, self.contour_item,
                     self.centre_item):
            item.setData([], [])
        self.spectrin_outline_item.clear()
        if mask is None or aw.tubulin is None:
            self.image_item.clear()
            self.outline_item.clear()
            return
        r0, r1, c0, c1 = mask.region
        image, offset = aw.tubulin.image, aw.tubulin_offset
        rows, cols = (r0, r1), (c0, c1)
        if show_reference and aw.reference is not None:
            dr = aw.reference_offset[1] - aw.tubulin_offset[1]
            dc = aw.reference_offset[0] - aw.tubulin_offset[0]
            height, width = aw.reference.shape
            rows = (int(max(0, r0 + dr)), int(min(height, r1 + dr)))
            cols = (int(max(0, c0 + dc)), int(min(width, c1 + dc)))
            image, offset = aw.reference.image, aw.reference_offset
        region = image[rows[0]:rows[1], cols[0]:cols[1]]
        if region.size:
            low, high = np.percentile(region, (1, 99.7))
            self.image_item.setImage(region.T, levels=(low, max(high, low + 1)))
            self.image_item.setRect(self._region_rect(aw, rows, cols, offset))
        else:
            self.image_item.clear()

        edge = mask.mask & ~ndimage.binary_erosion(mask.mask)
        self.outline_item.setImage(
            np.transpose(old_cased_edge(edge, _OUTLINE), (1, 0, 2)),
            levels=(0, 255))
        self.outline_item.setRect(self._region_rect(aw, (r0, r1), (c0, c1), aw.tubulin_offset))

        result = aw.result
        loc = aw.inputs.loc
        if result is not None:
            rng = np.random.default_rng(0)
            located = (aw.located if aw.located is not None
                       else np.full(loc.n, ax.LOC_NO_CLUSTER, dtype=object))
            for label, item in ((ax.LOC_NO_CLUSTER, self.free_item),
                                (ax.LOC_MEMBRANE, self.membrane_item),
                                (ax.LOC_INSIDE, self.interior_item)):
                idx = np.nonzero(located == label)[0]
                if idx.size > OLD_MAX_DRAWN:
                    idx = rng.choice(idx, OLD_MAX_DRAWN, replace=False)
                item.setData(loc.x_nm[idx], loc.y_nm[idx])
        interior = aw.spectrin_interior
        if interior is not None and interior.mask.any():
            ring = interior.mask & ~ndimage.binary_erosion(interior.mask)
            cased = old_cased_edge(ring, _SPECTRIN_OUTLINE)
            self.spectrin_outline_item.setImage(
                np.transpose(cased, (1, 0, 2)), levels=(0, 255))
            ir0, ir1, ic0, ic1 = interior.region
            self.spectrin_outline_item.setRect(self._region_rect(
                aw, (ir0, ir1), (ic0, ic1), aw.reference_offset))
        found = aw.anchored
        if found is not None and aw.anchored_centroids is not None:
            c = aw.anchored_centroids
            for item, group in ((self.discarded_item, found.discarded),
                                (self.tubulin_only_item, found.tubulin_only),
                                (self.spectrin_only_item,
                                 found.spectrin_only),
                                (self.cluster_item, found.inside_neither)):
                members = np.asarray(group, dtype=bool)
                item.setData(c[members, 0], c[members, 1])
            for item, drawn in ((self.contour_all_item, found.contour_all),
                                (self.contour_item, found.contour_anchored)):
                if drawn is not None:
                    closed = np.vstack([drawn.contour, drawn.contour[:1]])
                    item.setData(closed[:, 0], closed[:, 1])
            new = found.contour_anchored
            if new is not None and new.centre is not None:
                self.centre_item.setData([new.centre.x_nm],
                                         [new.centre.y_nm])
        else:
            centroids = aw.inputs.clusters()
            if centroids is not None and len(centroids):
                c = np.asarray(centroids, float)
                self.cluster_item.setData(c[:, 0], c[:, 1])


def _old_segments_with_locs(ms: Any, field: str) -> List[Any]:
    """MPSRingsWindow._segments_with_locs at 01e603f."""
    out = []
    for k, (seg, an) in enumerate(zip(ms.segments, ms.analyses)):
        if an is not None and getattr(an, field).size:
            out.append((k, seg, an))
    return out


def old_g1(ms: Any) -> Tuple[Any, Any]:
    """G1: the superimposed part of MPSRingsWindow._draw_spatial at 01e603f (the plot and its layer panel).
    Returns (plot_overlay, overlay_layers)."""
    import pyqtgraph as pg
    from tools.mps_layer_panel import LayerPanel, Swatch
    from tools.mps_plot_style import segment_colour, segment_symbol
    plot_overlay = pg.PlotWidget()
    overlay_layers = LayerPanel()
    plot_overlay.clear()
    overlay_layers.clear_layers(keep_state=True)
    overlay_layers.add_group("Segments")
    rows = _old_segments_with_locs(ms, "x_slab")
    for k, seg, an in rows:
        scatter = pg.ScatterPlotItem(
            an.x_slab, an.y_slab, pen=pg.mkPen(segment_colour(k), width=1),
            brush=None, size=5, symbol=segment_symbol(k))
        plot_overlay.addItem(scatter)
        overlay_layers.add_layer(
            f"seg{seg.index}", f"Segment {seg.index}",
            Swatch("symbol", segment_colour(k), symbol=segment_symbol(k),
                   hollow=True),
            items=[scatter], count=int(np.asarray(an.x_slab).size),
            tip=f"Draw segment {seg.index}'s localizations in the "
                "superimposed plot (the number is how many).")
    return plot_overlay, overlay_layers


def old_gz(plot: Any, ms: Any) -> None:
    """Gz: MPSRingsWindow._draw_z at 01e603f."""
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtGui
    from tools.mps_plot_style import neutral, rgba, segment_colour
    _C_NEUTRAL = neutral(dark=True)
    plot.clear()
    zr = ms.z_result

    zs = [a.z_slab for a in ms.analyses if a is not None and a.z_slab.size]
    if zs:
        allz = np.concatenate(zs)
        counts, edges = np.histogram(allz, bins=80, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        plot.addItem(pg.BarGraphItem(
            x=centres, height=counts, width=float(np.mean(np.diff(edges))),
            brush=pg.mkBrush(*rgba("dim", 80)), pen=None))

        grid = np.linspace(allz.min(), allz.max(), 1024)
        dens = zr.mixture_density(grid)
        if np.any(dens > 0):
            plot.addItem(pg.PlotDataItem(
                grid, dens, pen=pg.mkPen(_C_NEUTRAL, width=2)))

    for k, seg in enumerate(ms.segments):
        region = pg.LinearRegionItem(
            values=(seg.zmin_nm, seg.zmax_nm), movable=False)
        col = QtGui.QColor(segment_colour(k))
        col.setAlpha(55)
        region.setBrush(pg.mkBrush(col))
        region.setZValue(-10)
        plot.addItem(region)
        plot.addItem(pg.InfiniteLine(
            pos=seg.center_nm, angle=90,
            pen=pg.mkPen(segment_colour(k), width=2,
                         style=QtCore.Qt.DotLine),
            label=f"seg {seg.index}",
            labelOpts={"position": 0.95,
                       "color": segment_colour(k)}))

    if ms.valleys is not None:
        for pos, real, depth in zip(ms.valleys.positions_nm,
                                    ms.valleys.is_true_valley,
                                    ms.valleys.relative_depth):
            plot.addItem(pg.InfiniteLine(
                pos=float(pos), angle=90,
                pen=pg.mkPen(
                    _C_NEUTRAL, width=2,
                    style=QtCore.Qt.SolidLine if real
                    else QtCore.Qt.DashLine),
                label=(f"valley {depth:.2f}" if real else "no valley"),
                labelOpts={"position": 0.08, "color": _C_NEUTRAL}))


def old_gh(grid: Any, ms: Any) -> None:
    """Gh: MPSRingsWindow._draw_zhist at 01e603f, into ``grid`` (a GraphicsLayoutWidget)."""
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtGui
    from tools.mps_plot_style import segment_colour, set_title, style_dark
    grid.clear()
    rows = _old_segments_with_locs(ms, "z_slab")
    if not rows:
        return
    lo = min(seg.zmin_nm for _, seg, _ in rows)
    hi = max(seg.zmax_nm for _, seg, _ in rows)
    pad = 0.05 * max(hi - lo, 1.0)
    zr = (lo - pad, hi + pad)

    first: Optional[Any] = None
    for i, (k, seg, an) in enumerate(rows):
        colour = segment_colour(k)
        p = grid.addPlot(row=0, col=i)
        style_dark(p)
        set_title(p, f"segment {seg.index}")
        counts, edges = np.histogram(an.z_slab, bins=40)
        centres = (edges[:-1] + edges[1:]) / 2
        width = float(np.mean(np.diff(edges))) if edges.size > 1 else 1.0
        fill = QtGui.QColor(colour)
        fill.setAlpha(170)
        p.addItem(pg.BarGraphItem(
            x=centres, height=counts, width=width,
            brush=pg.mkBrush(fill), pen=None))
        for bound in (seg.zmin_nm, seg.zmax_nm):
            p.addItem(pg.InfiniteLine(
                pos=float(bound), angle=90,
                pen=pg.mkPen(colour, width=1, style=QtCore.Qt.DashLine)))
        p.setLabels(bottom="z [nm]", left="count" if i == 0 else "")
        p.setXRange(*zr, padding=0)
        if first is None:
            first = p
        else:
            p.setXLink(first)


def old_m4(centroids: Any, xroi: Any, point_size: float) -> Any:
    """M4: MPS_explorer._render_good_clusters_panel at 01e603f; returns the widget it put in the layout."""
    import pyqtgraph as pg
    from tools.mps_plot_style import role
    brush3 = pg.mkBrush(role("centroid"))
    good_clusters_widget = pg.GraphicsLayoutWidget()
    good_clusters_plot = good_clusters_widget.addPlot(
        title="Clusters centers and distances")
    good_clusters_plot.setAspectLocked(True)
    good_clusters_plot.setLabels(bottom='x [nm]', left='y [nm]')

    if len(centroids):
        good_clusters_plot.addItem(pg.ScatterPlotItem(
            centroids[:, 0], centroids[:, 1],
            size=point_size, brush=brush3))

    if xroi is not None and len(xroi):
        good_clusters_plot.setXRange(
            np.min(xroi), np.max(xroi), padding=0)
    return good_clusters_widget


def old_m5(good_cluster_centroids: Any, n_text: str, lmin: Optional[float], lmax: Optional[float],
           bins_: Optional[int]) -> Tuple[np.ndarray, Any]:
    """M5: MPS_explorer.KNdist_hist at 01e603f (the message boxes left out). Returns (self.distances, the widget
    it put in the layout, or None when nothing was in range)."""
    import pyqtgraph as pg
    from sklearn.neighbors import KDTree
    from tools.mps_plot_style import role
    brush3 = pg.mkBrush(role("centroid"))
    Nneighbor = int(float(n_text))
    tree = KDTree(good_cluster_centroids)
    distances, indexes = tree.query(good_cluster_centroids, Nneighbor + 1)
    distances = distances[:, 1:]

    histzWidget3 = pg.GraphicsLayoutWidget()
    histabcm = histzWidget3.addPlot(title="distances Histogram")
    distances_full = distances
    plot_distances = distances_full
    if lmin is not None and lmax is not None:
        in_range = ((distances_full > lmin)
                    & (distances_full < lmax))
        plot_distances = distances_full[in_range]
    bins = bins_ if bins_ is not None else 20
    if plot_distances.size == 0:
        return distances, None
    histcmdist, bin_edgescmdist = np.histogram(plot_distances, bins)
    widthcmdist = np.mean(np.diff(bin_edgescmdist))
    bincenterscmdist = np.mean(np.vstack([bin_edgescmdist[0:-1], bin_edgescmdist[1:]]), axis=0)
    bargraphcmdist = pg.BarGraphItem(x=bincenterscmdist, height=histcmdist,
                                     width=widthcmdist, brush=brush3, pen=None)
    histabcm.addItem(bargraphcmdist)
    histabcm.setXRange(lmin, lmax)
    return distances, histzWidget3


def old_savedistdata(filename: str, dist: np.ndarray, labels: Optional[np.ndarray]) -> None:
    """MPS_explorer.savedistdata at 01e603f (after the file dialog); ``labels`` = self._current_cluster_labels()."""
    import pandas as pd
    data = {}
    if labels is not None and len(labels) == len(dist):
        data["cluster_label"] = np.asarray(labels, dtype=int)
    for k in range(dist.shape[1]):
        data[f"nn{k + 1}_nm"] = dist[:, k]
    pd.DataFrame(data).to_csv(filename, index=False,
                              float_format="%.2f")


def old_current_cluster_labels(mw: Any) -> Optional[np.ndarray]:
    """MPS_explorer._current_cluster_labels at 01e603f (with _current_cluster_centroids inlined)."""
    from tools.cluster_quality import good_cluster_labels
    analysis = mw.mps_analysis
    current = mw._analysed_x is not None and (
        mw._analysed_x is mw.xroi_unfiltered
        or mw._analysed_x is mw.xroi)
    if analysis is None or not current:
        return None
    return good_cluster_labels(np.asarray(analysis.labels),
                               analysis.bad_report.bad_labels)


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
            r1 = pg.PlotWidget()                            # R1 as it drew, from the same objects
            old_r1(r1, a, w.analysis, w.comparison, dark=w.dark)
            scat = scatter_items(r1)
            lines = line_items(r1)

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
        import MPS_explorer
        mw = need(st, "mw")
        # run_mps_analysis drew M4 from analysis.centroids (it set good_cluster_centroids from the same array)
        assert same(mw.good_cluster_centroids, need(st, "measured").centroids)
        widget = old_m4(mw.mps_analysis.centroids, mw.xroi,
                        getattr(MPS_explorer, "CLUSTER_CENTROID_POINT_SIZE", 10))
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

    a1_keys = ("sel_inside", "sel_membrane", "sel_free", "c_both", "c_tubulin", "c_spectrin", "c_neither")

    def a1_parity() -> str:
        from tools import mps_axoplasm_window as axw
        from tools.mps_axon_map import cased_edge_rgba
        aw = need(st, "aw")
        _OUTLINE, _SPECTRIN_OUTLINE = _old_outline_colours()
        a1 = OldAxoplasmImage()                             # A1 as it drew, check box off (the tubulin image)
        a1.draw(aw, show_reference=False)
        state = axoplasm_state(aw)
        st["ax_state"] = state
        inp = map_inputs("discard", axoplasm=state, source="selection", colour_by="images")
        groups = L.build_map(inp)
        old_items = dict(zip(a1_keys, (a1.interior_item, a1.membrane_item, a1.free_item, a1.discarded_item,
                                       a1.tubulin_only_item, a1.spectrin_only_item, a1.cluster_item)))
        for key, item in old_items.items():
            ox, oy = xy(item)
            lay = layer_of(groups, key)
            assert same(lay.data["x"], ox) and same(lay.data["y"], oy), key
        img = layer_of(groups, "image")
        assert same(img.data["image"].T, a1.image_item.image), "tubulin image region"
        r = a1.image_item.mapRectToParent(a1.image_item.boundingRect())
        assert np.allclose(img.data["rect"], (r.x(), r.y(), r.width(), r.height())), "tubulin image rect"
        assert np.allclose(img.data["levels"], a1.image_item.levels), "levels p1-p99.7"
        edge = layer_of(groups, "tubulin_edge")
        assert same(np.transpose(old_cased_edge(edge.data["edge"], _OUTLINE), (1, 0, 2)),
                    a1.outline_item.image), "tubulin edge"
        r1_ = a1.outline_item.mapRectToParent(a1.outline_item.boundingRect())
        assert np.allclose(edge.data["rect"], (r1_.x(), r1_.y(), r1_.width(), r1_.height())), "tubulin edge rect"
        assert same(cased_edge_rgba(edge.data["edge"], _OUTLINE), old_cased_edge(edge.data["edge"], _OUTLINE))
        sedge = layer_of(groups, "spectrin_edge")
        assert same(np.transpose(old_cased_edge(sedge.data["edge"], _SPECTRIN_OUTLINE), (1, 0, 2)),
                    a1.spectrin_outline_item.image), "spectrin edge"
        r2 = a1.spectrin_outline_item.mapRectToParent(a1.spectrin_outline_item.boundingRect())
        assert np.allclose(sedge.data["rect"], (r2.x(), r2.y(), r2.width(), r2.height()))
        # A1's contours and centre are the discard comparison's own objects
        cx, cy = xy(a1.contour_item)
        lay = layer_of(groups, "contour")
        assert same(lay.data["x"], cx) and same(lay.data["y"], cy), "A1's blue contour"
        ax_, ay_ = xy(a1.contour_all_item)
        la = layer_of(groups, "contour_all")
        assert same(la.data["x"], ax_) and same(la.data["y"], ay_), "A1's orange dashed contour"
        px, py = xy(a1.centre_item)
        assert same(layer_of(groups, "centre").data["x"], px) and same(layer_of(groups, "centre").data["y"], py)
        # the spectrin image, as A1's check box drew it
        a1.draw(aw, show_reference=True)
        sp = layer_of(L.build_map(map_inputs("discard", axoplasm=state, source="selection", colour_by="images",
                                             image="spectrin")), "image")
        assert sp.data["which"] == "spectrin" and same(sp.data["image"].T, a1.image_item.image)
        r3 = a1.image_item.mapRectToParent(a1.image_item.boundingRect())
        assert np.allclose(sp.data["rect"], (r3.x(), r3.y(), r3.width(), r3.height()))
        # the panel's own map_state() (what the windows draw from) gives the same layers
        a1.draw(aw, show_reference=False)
        own = L.build_map(map_inputs("discard", axoplasm=aw.map_state(), source="selection", colour_by="images"))
        for key, item in old_items.items():
            ox, oy = xy(item)
            lay = layer_of(own, key)
            assert same(lay.data["x"], ox) and same(lay.data["y"], oy), ("map_state", key)
        for key, fields in (("image", ("image", "rect", "levels")), ("tubulin_edge", ("edge", "rect")),
                            ("spectrin_edge", ("edge", "rect"))):
            for field in fields:
                assert same(layer_of(own, key).data[field], layer_of(groups, key).data[field]), \
                    ("map_state", key, field)
        n = {k: layer_of(groups, k).count for k in a1_keys}
        assert L.MAX_DRAWN_PER_CLASS == axw.MAX_DRAWN == OLD_MAX_DRAWN
        return "image (both), edges, 3 localization classes, 4 cluster groups, contours and centre equal " \
            "(also via map_state()): " + ", ".join(f"{k} {v}" for k, v in n.items())

    def g1_parity() -> str:
        rw = need(st, "mw").rings_window
        ms = rw.ms
        groups = L.build_map(map_inputs("measured", rings=ms, source="segments"))
        rows = [layer for layer in group_of(groups, "localizations").layers]
        _plot_overlay, overlay_layers = old_g1(ms)          # G1 as it drew, from the same ms
        keys = overlay_layers.keys()
        assert [r.key for r in rows] == keys, (keys, [r.key for r in rows])
        for r in rows:
            ox, oy = xy(overlay_layers.items(r.key)[0])
            assert same(r.data["x"], ox) and same(r.data["y"], oy), r.key
            assert overlay_layers.checkbox(r.key).text() == f"{r.label} ({r.count})", r.key
        for key in L.SEGMENTS_DISABLE:
            for g in groups:
                for layer in g.layers:
                    if layer.key == key:
                        assert not layer.enabled and layer.reason == L.SEGMENT_COLOUR_REASON, key
        return f"{len(rows)} segment rows: same keys, labels, counts and points as the rings panel"

    def windows_use_builders() -> str:
        # Light sanity: the MPS window that replaced R1, Rz, Rn and Rc draws what the builders build, and its
        # drawn items carry the arrays the old code drew.
        w = need(st, "w")
        w.axon_map.set_view("mps")
        nn = w.nn
        bins0 = nn.spin_bins.value()
        nn.radio_range_auto.setChecked(True)
        nn.spin_neighbours.setValue(1)
        nn.spin_bins.setValue(30)
        out = []
        try:
            for name, radio in radios():
                radio.setChecked(True)
                pump(app, 0.02)
                a = w._shown()
                amap = w.axon_map
                assert amap.inputs().shown == name and amap.inputs().source == "slab", name
                built = L.build_map(map_inputs(name))
                for key in ("slab_kept", "c_kept", "contour", "centre"):
                    b = layer_of(built, key)
                    drawn = amap.layer(key)
                    assert same(drawn.data["x"], b.data["x"]) and same(drawn.data["y"], b.data["y"]), (name, key)
                    items = amap.items(key)
                    assert items, (name, key)
                    for it in items:
                        ix, iy = xy(it)
                        assert same(ix, b.data["x"]) and same(iy, b.data["y"]), (name, key, "item")
                # axial: the slab's own z and the components, as Rz drew them
                rz = pg.PlotWidget()
                old_rz(rz, a)
                old_means = [float(it.value()) for it in rz.getPlotItem().items if isinstance(it, pg.InfiniteLine)]
                comp_items = w.axial.items("components")
                means = [float(it.value()) for it in comp_items if isinstance(it, pg.InfiniteLine)]
                assert np.allclose(means, old_means), (name, means, old_means)
                old_curves = [xy(it) for it in rz.getPlotItem().items if isinstance(it, pg.PlotDataItem)]
                curves = [xy(it) for it in comp_items if isinstance(it, pg.PlotDataItem)]
                assert len(curves) == len(old_curves) and all(
                    same(c[0], o[0]) and same(c[1], o[1]) for c, o in zip(curves, old_curves)), name
                assert same(np.sort(w.axial.layer("roi_in").data["z"]), np.sort(a.z_slab)), name
                # nearest neighbours: the bars Rn drew, the CDFs Rc drew
                rn, rc = pg.PlotWidget(), pg.PlotWidget()
                old_rn(rn, a)
                old_rc(rc, a)
                ob = [it for it in rn.getPlotItem().items if isinstance(it, pg.BarGraphItem)][0]
                nb = [it for it in nn.items_of("nn_bars") if isinstance(it, pg.BarGraphItem)][0]
                assert same(nb.opts["height"], ob.opts["height"]) and same(nb.opts["x"], ob.opts["x"]), name
                old_c = {it.opts["name"]: xy(it) for it in rc.getPlotItem().items if isinstance(it, pg.PlotDataItem)}
                new_c = {it.opts["name"]: xy(it) for it in nn.plot_cdf.getPlotItem().items
                         if isinstance(it, pg.PlotDataItem)}
                for nm_ in ("observed", "randomized"):
                    key = [k for k in new_c if k and nm_ in k.lower()]
                    assert key, (name, nm_, list(new_c))
                    assert same(new_c[key[0]][0], old_c[nm_][0]) and same(new_c[key[0]][1], old_c[nm_][1]), name
                out.append(name)
        finally:
            nn.spin_bins.setValue(bins0)
            w.radio_measured.setChecked(True)
        return ("the MPS window's map items, axial components and slab z, NN bars and CDFs equal the builders' "
                "and the old code's, for " + ", ".join(out))

    check("map vs R1 (each radio state): every layer's arrays", r1_parity)
    check("the MPS window draws from the builders (map, axial, nearest neighbours)", windows_use_builders)
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
            rz = pg.PlotWidget()                            # Rz as it drew
            old_rz(rz, a)
            items = rz.getPlotItem().items
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
        gz = pg.PlotWidget()                                # Gz as it drew, from the same ms
        old_gz(gz, ms)
        items = gz.getPlotItem().items
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
        zhist_grid = pg.GraphicsLayoutWidget()              # Gh as it drew
        old_gh(zhist_grid, ms)
        plots = [p for p in zhist_grid.ci.items]
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
            plot_nn, plot_cdf = pg.PlotWidget(), pg.PlotWidget()   # Rn and Rc as they drew
            old_rn(plot_nn, a)
            old_rc(plot_cdf, a)
            items = plot_nn.getPlotItem().items
            bars = [it for it in items if isinstance(it, pg.BarGraphItem)][0]
            nl = L.nn_layers(a, hist)
            nb = [x for x in nl if x.key == "nn_bars"][0]
            assert same(bars.opts["height"], nb.data["height"]) and same(bars.opts["x"], nb.data["x"]), name
            lines = [float(it.value()) for it in items if isinstance(it, pg.InfiniteLine)]
            med = [x for x in nl if x.key == "nn_median"][0]
            ref = [x for x in nl if x.key == "nn_reference"][0]
            assert lines == med.data["positions"] + ref.data["positions"], (name, lines)
            cdf_items = plot_cdf.getPlotItem().items
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

    # M5's display state when nothing was typed in its three fields (MPS_explorer.__init__ at 01e603f)
    def m5_state() -> Tuple[float, float, int]:
        import MPS_explorer
        return 0.0, float(MPS_explorer.MAX_LATERAL_DISTANCE_NM), int(MPS_explorer.DEFAULT_KNN_BINS)

    def m5_parity() -> str:
        mw = need(st, "mw")
        a = need(st, "measured")
        lmin, lmax, bins = m5_state()
        distances, widget = old_m5(mw.good_cluster_centroids, "3", lmin, lmax, bins)   # M5 as it drew, N = 3
        pooled = L.knn_distances(a.centroids, 3)
        assert same(pooled, distances), "the same KD-tree distances"
        h = L.nn_histogram(a, 3, int(bins))
        assert same(np.sort(h["values"]), np.sort(np.asarray(distances).ravel()))
        assert widget is not None, "M5 drew nothing: every distance outside its range"
        bars = [it for p in widget.ci.items for it in getattr(p, "items", []) if isinstance(it, pg.BarGraphItem)][0]
        d = np.asarray(distances)
        shown = d[(d > lmin) & (d < lmax)]
        counts, edges = np.histogram(shown, int(bins))
        assert same(bars.opts["height"], counts)
        if shown.size == d.size:
            assert same(h["counts"], counts) and same(h["edges"], edges), "nothing outside: the same histogram"
        hr = L.nn_histogram(a, 3, int(bins), (lmin, lmax))
        assert hr["outside"] == int(np.sum((d < lmin) | (d > lmax)))
        assert L.nn_max_neighbours(a.n_clusters_kept) == min(10, a.n_clusters_kept - 1)
        # the MPS window's tab starts where M5 started: its bins and its range
        nn = need(st, "w").nn
        assert nn.spin_bins.value() == bins and (nn.spin_range_min.value(), nn.spin_range_max.value()) == \
            (lmin, lmax), (nn.spin_bins.value(), nn.spin_range_min.value(), nn.spin_range_max.value())
        return (f"pooled 1st..3rd distances = M5's ({d.size}); the bars equal M5's; N capped at clusters - 1; the "
                f"tab starts at M5's {bins} bins and {lmin:g}-{lmax:g} nm")

    def csv_bytes() -> str:
        mw, w = need(st, "mw"), need(st, "w")
        a = need(st, "measured")
        labels = old_current_cluster_labels(mw)
        assert labels is not None, "the analysis does not describe the current selection"
        w.radio_measured.setChecked(True)
        pump(app, 0.02)
        out = []
        for n in (1, 3):
            distances, _widget = old_m5(mw.good_cluster_centroids, str(n), *m5_state())
            old = os.path.join(WORK, f"old_{n}.csv")
            old_savedistdata(old, distances, labels)      # "save dist data" as it wrote
            new = os.path.join(WORK, f"new_{n}.csv")
            L.write_distances_csv(new, a, a, n)
            tab = os.path.join(WORK, f"tab_{n}.csv")
            w.nn.spin_neighbours.setValue(n)
            assert w.nn.n() == n and w.nn.shown is a
            w.nn.save_to(tab)                             # the MPS window's "Save distances..."
            with open(old, "rb") as f1, open(new, "rb") as f2, open(tab, "rb") as f3:
                b1, b2, b3 = f1.read(), f2.read(), f3.read()
            assert b1 == b2, f"N={n}: write_distances_csv's bytes differ"
            assert b1 == b3, f"N={n}: the tab's bytes differ"
            out.append(f"N={n}: {len(b1)} bytes")
        w.nn.spin_neighbours.setValue(1)
        root = mw.get_root_filename()
        assert L.default_distances_name(root, 3) == f"{root}_3neighbor_distances.csv"
        return "; ".join(out) + " identical to 'save dist data' (write_distances_csv and the tab's save_to)"

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
            lines[2] == "z as fitted; no z correction applied", at
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
        assert line2.endswith("z as fitted, no z correction applied") and \
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
