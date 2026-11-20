# -*- coding: utf-8 -*-
"""
The axon map: one lab-frame map of the axon with its layers (UI stage 2, design 3).

``AxonMap`` is a plain QWidget - a pyqtgraph plot (lab x, y in nm, aspect locked), a ``LayerPanel`` beside it with
one group per computation, a "View" combo, and two provenance lines above the plot - drawn entirely from the pure
builders of ``tools.mps_axon_map_layers``. It replaces, in IMPL-C, the MPS analysis window's "Contour and centre", the
Rings window's superimposed segments, the Axoplasm panel's image and the main window's cluster centres; until then
nothing uses it and every old plot is unchanged.

What it keeps (3.1-3.6):

* every row shows or hides exactly its own items, its state kept through redraws, radio changes and re-runs;
* three exclusive localization sources and two colourings of the one set of centres, chosen in the group headers;
* rows that cannot be drawn honestly are disabled with the reason (grey on the grey image; the segments' colours;
  another selection; discard only);
* the plot title is the provenance that travels with an exported figure (two lines, ``map_title``); the export
  message names the visible layers, counts the hidden ones and repeats the provenance lines.

``render_layer`` turns one row specification into pyqtgraph items; the axial view and the nearest-neighbours tab use
it too.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_axon_map_layers as L
from tools.plot_mouse import RIGHT_DRAG_HELP, install_right_drag_pan
from tools import mps_param_registry as reg
from tools.mps_layer_panel import LayerPanel, Swatch
from tools.mps_plot_style import (
    DENSITY_RAMP, PLOT_BG, fit_title_width, neutral, role, segment_colour, segment_symbol, set_title, style_dark,
    style_light, title_natural_width)

__all__ = ["AxonMap", "render_layer", "swatch_for", "cased_edge_rgba", "colour_of"]

DARK_RIM = "#000000"
PANEL_WIDTH = 280          # the side column's narrowest width
SIDE_MAX_WIDTH = 460       # and its widest: what the plot's shape leaves, up to this
CLUSTER_INFO_TIP = ("Click a cluster centre on the map: its label, area and its distances to its nearest centres "
                    "(the same query as the Nearest neighbours tab). Click elsewhere to clear.")
CLICK_TOLERANCE_PX = 12     # how near a click must fall to a centre, in screen pixels
SIDE_DRAG_MAX = 1200       # how wide the user may drag it, to read long layer names (manual test 2, item 4)
FIT_PAD_FRACTION = 0.06    # the view: the data's extent plus this fraction of its larger side ...
FIT_PAD_MIN_NM = 150.0     # ... and at least this much
VIEW_LABEL = "View"
SOURCE_TIP = ("Which localizations: three different sets, one at a time - the MPS analysis' own slab, the "
              "main-window selection placed by the Axoplasm panel, or each segment's slab from the Rings window.")
COLOUR_TIP = ("How the cluster centres are coloured: by their status in the MPS analysis, or by the two widefield "
              "images of the Axoplasm panel. The same clusters either way.")
IMAGE_TIP = "Which widefield image is drawn under the map."


def _hex_rgba(colour: str, alpha: int) -> Tuple[int, int, int, int]:
    value = colour.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), int(alpha))


def colour_of(layer: L.Layer, dark: bool = True) -> str:
    """The hex colour of a row: its role, the neutral of the background, or its segment's colour."""
    s = layer.style
    if s.role == "neutral":
        return neutral(dark)
    if s.role == "segment":
        return segment_colour(int(layer.segment or 0))
    if s.role == "image":
        return "#9a9a9a"
    if s.role == "ramp":
        # a quantity: the swatch shows a class above the threshold for the cells of a cluster, one below for the rest
        return DENSITY_RAMP[5] if layer.key.startswith("vor_in") else DENSITY_RAMP[2]
    return role(s.role)


def cased_edge_rgba(edge: NDArray[np.bool_], colour: Tuple[int, int, int, int]) -> NDArray[np.uint8]:
    """An edge with a one-pixel dark rim, as the Axoplasm panel drew its edges over a photograph
    (``mps_axoplasm_window._cased_edge``)."""
    from scipy import ndimage
    e = np.asarray(edge, dtype=bool)
    rim = ndimage.binary_dilation(e) & ~e
    out = np.zeros(e.shape + (4,), dtype=np.ubyte)
    out[rim] = (0, 0, 0, 200)
    out[e] = colour
    return out


def _pen_style(name: str) -> Any:
    return {"dash": QtCore.Qt.PenStyle.DashLine, "dot": QtCore.Qt.PenStyle.DotLine,
            "dashdot": QtCore.Qt.PenStyle.DashDotLine, "dashdotdot": QtCore.Qt.PenStyle.DashDotDotLine}.get(
        name, QtCore.Qt.PenStyle.SolidLine)


def swatch_for(layer: L.Layer, dark: bool = True) -> Swatch:
    """The row's icon, drawn like the plot draws it."""
    s = layer.style
    colour = colour_of(layer, dark)
    if layer.kind in ("scatter",):
        symbol = segment_symbol(int(layer.segment or 0)) if s.role == "segment" else s.symbol
        if symbol == "star":
            symbol = "star"
        pen = None if s.outline is None else (neutral(dark) if s.outline == "neutral" else DARK_RIM)
        return Swatch("symbol", _hex_rgba(colour, s.alpha) if s.alpha < 255 else colour, symbol=symbol,
                      pen=pen, hollow=s.hollow, size=min(12.0, max(3.0, s.size)), cased=s.cased)
    if layer.kind in ("bars",):
        return Swatch("bar", _hex_rgba(colour, max(s.alpha, 140)))
    if layer.kind == "image":
        return Swatch("image")
    if layer.kind == "band":
        return Swatch("bar", _hex_rgba(colour, max(s.alpha, 120)))
    if layer.kind == "voronoi":
        return Swatch("bar", _hex_rgba(colour, s.alpha))
    return Swatch("line", colour, dash=s.pen in ("dash", "dashdot", "dashdotdot"), dot=s.pen == "dot",
                  width=min(3.0, max(1.5, s.width)), cased=s.cased)


def _brush(layer: L.Layer, dark: bool) -> Any:
    s = layer.style
    return pg.mkBrush(*_hex_rgba(colour_of(layer, dark), s.alpha))


VORONOI_Z = {"vor_out": -18.0, "vor_in": -17.0}   # over the widefield image (-20), under its edges (-15)
VORONOI_OUTLINE_ALPHA = 150                       # the thin neutral outline of a cluster's cells [ad hoc]
# The localizations drawn inside each row's cells (z 0, over the cells at -17/-18).
VORONOI_POINTS = {"vor_in": "slab_kept", "vor_out": "slab_noise"}
SLAB_POINT_KEYS = frozenset({"slab_kept", "slab_noise", "slab_curated", "slab_discarded"})
# Over the cells the points get a dark rim and a little more size, so a sky-blue dot reads on a green or yellow cell
# (manual test of 2026-10-23: no point could be told inside the cells) [ad hoc: 1 px black rim, 5 px].
POINTS_OVER_CELLS_SIZE = 5.0
POINTS_OVER_CELLS_RIM = 1.0


def voronoi_items(layer: L.Layer, dark: bool = True) -> List[Any]:
    """One QGraphicsPathItem per density class (not one item per cell): the polygons of the class joined into one
    path (``pyqtgraph.arrayToQPath`` with a connect array), filled in the class colour. Empty when the row has nothing
    to draw or its cells cannot be rebuilt (``tools.voronoi_layer.cell_drawing`` says why)."""
    from tools import voronoi_layer as V
    d = layer.data
    if "diag" not in d:
        return []
    s = layer.style
    drawing, why = V.cell_drawing(d["diag"], d["x"], d["y"], d["labels"], d["scale"], bool(d["in_clusters"]))
    if why:
        return []
    if s.outline == "neutral":
        pen = pg.mkPen(*_hex_rgba(neutral(dark), VORONOI_OUTLINE_ALPHA), width=s.width)
        pen.setCosmetic(True)
    else:
        pen = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
    items: List[Any] = []
    for k in sorted(drawing.classes):
        x, y, connect = drawing.classes[k]
        path = pg.arrayToQPath(x, y, connect=connect)
        item = QtWidgets.QGraphicsPathItem(path)
        item.setBrush(pg.mkBrush(*_hex_rgba(d["scale"].colours[k], s.alpha)))
        item.setPen(pen)
        item.setZValue(VORONOI_Z.get(layer.key[:7] if layer.key.startswith("vor_out") else layer.key[:6], -18.0))
        items.append(item)
    return items


def legend_sample(layer: L.Layer, dark: bool = True) -> Any:
    """F6: a stand-in item drawn in the exported figure's legend like the row's own items (not added to the plot);
    None for a row with no mark of its own (the widefield image)."""
    s = layer.style
    colour = colour_of(layer, dark)
    if layer.kind == "image":
        return None
    if layer.kind == "scatter":
        symbol = segment_symbol(int(layer.segment or 0)) if s.role == "segment" else s.symbol
        if s.hollow or s.role == "segment" or symbol in ("x", "+"):
            return pg.ScatterPlotItem(symbol=symbol, size=min(12.0, max(5.0, s.size)),
                                      pen=pg.mkPen(colour, width=1.5), brush=None)
        pen = (None if s.outline is None else pg.mkPen(neutral(dark) if s.outline == "neutral" else DARK_RIM))
        return pg.ScatterPlotItem(symbol=symbol, size=min(12.0, max(5.0, s.size)), pen=pen,
                                  brush=pg.mkBrush(*_hex_rgba(colour, max(s.alpha, 160))))
    if layer.kind in ("bars", "band", "voronoi"):
        return pg.BarGraphItem(x=[0.0], height=[1.0], width=1.0, brush=pg.mkBrush(*_hex_rgba(colour, max(s.alpha, 140))),
                               pen=None)
    return pg.PlotDataItem(pen=pg.mkPen(colour, width=max(1.5, s.width), style=_pen_style(s.pen)))


def legend_entries(layers: Dict[str, L.Layer], panel: Any, dark: bool = True,
                   keep: Optional[Callable[[str], bool]] = None) -> List[Tuple[str, Any]]:
    """F6: (label, sample) for every row the plot draws now (ticked, enabled and with an item shown), in the
    panel's order; ``keep`` narrows the rows (the NN panel's two plots)."""
    out: List[Tuple[str, Any]] = []
    in_panel = set(panel.keys())
    for key, layer in layers.items():
        if (keep is not None and not keep(key)) or key not in in_panel:
            continue
        if not panel.is_visible(key) or not layer.enabled:
            continue
        items = panel.items(key)
        if not items or not any(getattr(it, "isVisible", lambda: True)() for it in items):
            continue
        sample = legend_sample(layer, dark)
        if sample is not None:
            label = layer.label + (f" (n = {layer.count:,})" if layer.count is not None and layer.kind in
                                   ("scatter", "outlines") else "")
            out.append((label, sample))
    return out


class DensityScaleBar(QtWidgets.QWidget):
    """The Voronoi layer's scale under its rows: the classes in their colours, equal widths, with a white tick at the
    threshold the method applied, and the densities at the ends and at the tick (localizations per um^2)."""

    BAR_PX = (216, 14)

    def __init__(self, scale: Any, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        from tools import voronoi_layer as V
        self.setObjectName("voronoi_scale")
        self.scale = scale
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(1)
        self.bar = QtWidgets.QLabel()
        self.bar.setObjectName("voronoi_scale_bar")
        self.bar.setPixmap(self.pixmap())
        lay.addWidget(self.bar)
        values = scale.edge_values_um2()
        if scale.threshold_edge is not None:
            words = (f"{V.density_words(values[0])}  |  threshold {V.density_words(scale.threshold * V.PER_UM2)}  |  "
                     f"{V.density_words(values[-1])} loc/um^2")
        else:
            words = f"{V.density_words(values[0])} ... {V.density_words(values[-1])} loc/um^2 ({scale.note})"
        self.text = QtWidgets.QLabel(words)
        self.text.setObjectName("voronoi_scale_text")
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.text)
        self.setToolTip("Density classes of the Voronoi cells, from sparse (left) to dense (right): equal steps of "
                        "log density below the threshold and above it, so the tick sits on a class edge. The two "
                        "ends are the 1 % and 99 % quantiles of the cells (denser or sparser cells take the end "
                        "colour). Colours: viridis, lightness rising with density.")

    def pixmap(self) -> QtGui.QPixmap:
        w, h = self.BAR_PX
        pm = QtGui.QPixmap(w, h + 6)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        p.fillRect(QtCore.QRectF(0, 0, w, h + 6), pg.mkColor(PLOT_BG))
        n = len(self.scale.colours)
        step = (w - 4) / float(n)
        for k, colour in enumerate(self.scale.colours):
            p.fillRect(QtCore.QRectF(2 + k * step, 3, step + 0.5, h), pg.mkColor(colour))
        if self.scale.threshold_edge is not None:
            x = 2 + self.scale.threshold_edge * step
            p.setPen(QtGui.QPen(QtGui.QColor("#000000"), 4))
            p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h + 6))
            p.setPen(QtGui.QPen(pg.mkColor(neutral(dark=True)), 2))
            p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h + 6))
        p.end()
        return pm


def render_layer(layer: L.Layer, dark: bool = True) -> List[Any]:
    """The pyqtgraph items of one row (empty when it has nothing to draw)."""
    s = layer.style
    d = layer.data
    colour = colour_of(layer, dark)
    items: List[Any] = []
    if layer.kind == "voronoi":
        return voronoi_items(layer, dark)
    if layer.kind == "outlines":
        # every outline of the row in one item: a connect array ends each one
        if "x" not in d or not np.asarray(d["x"]).size:
            return items
        return [pg.PlotDataItem(np.asarray(d["x"], float), np.asarray(d["y"], float),
                                connect=np.asarray(d["connect"]),
                                pen=pg.mkPen(colour, width=s.width, style=_pen_style(s.pen)))]
    if layer.kind == "scatter":
        x = np.asarray(d.get("x", ()), float)
        y = np.asarray(d.get("y", ()), float)
        if s.role == "segment":
            k = int(layer.segment or 0)
            items.append(pg.ScatterPlotItem(x, y, pen=pg.mkPen(segment_colour(k), width=1), brush=None,
                                            size=s.size, symbol=segment_symbol(k)))
            return items
        if s.cased:
            items.append(pg.ScatterPlotItem(x, y, size=s.size, symbol=s.symbol, brush=None,
                                            pen=pg.mkPen(DARK_RIM, width=max(3.0, s.width + 2.0))))
        if s.hollow:
            pen = pg.mkPen(colour, width=1.5 if s.cased else 1.0)
            items.append(pg.ScatterPlotItem(x, y, size=s.size, symbol=s.symbol, pen=pen, brush=None))
        elif s.symbol == "x":
            items.append(pg.ScatterPlotItem(x, y, size=s.size, symbol="x",
                                            pen=pg.mkPen(*_hex_rgba(colour, s.alpha)), brush=None))
        elif s.symbol == "+":
            items.append(pg.ScatterPlotItem(x, y, size=s.size, symbol="+", pen=pg.mkPen(colour, width=s.width),
                                            brush=pg.mkBrush(colour)))
        else:
            pen = None
            if s.outline == "neutral":
                pen = pg.mkPen(neutral(dark))
            elif s.outline == "dark":
                pen = pg.mkPen(DARK_RIM)
            items.append(pg.ScatterPlotItem(x, y, size=s.size, symbol=s.symbol, pen=pen, brush=_brush(layer, dark)))
        if "contour_x" in d:
            items.append(pg.PlotDataItem(np.asarray(d["contour_x"], float), np.asarray(d["contour_y"], float),
                                         pen=pg.mkPen(*_hex_rgba(colour, max(s.alpha, 90)), width=1)))
        return items
    if layer.kind in ("polyline", "polylines"):
        runs = d.get("runs") if layer.kind == "polylines" else [np.column_stack([d.get("x", ()), d.get("y", ())])]
        for run in runs or []:
            r = np.asarray(run, float).reshape(-1, 2)
            if s.cased:
                items.append(pg.PlotDataItem(r[:, 0], r[:, 1], pen=pg.mkPen(DARK_RIM, width=s.width + 2.0)))
            items.append(pg.PlotDataItem(r[:, 0], r[:, 1], pen=pg.mkPen(colour, width=s.width,
                                                                         style=_pen_style(s.pen))))
        return items
    if layer.kind == "image":
        if "image" not in d:     # the image chosen is not loaded: the row says why, nothing is drawn
            return items
        image = np.asarray(d["image"])
        item = pg.ImageItem(image.T, levels=d["levels"])
        item.setRect(QtCore.QRectF(*d["rect"]))
        item.setZValue(-20)
        return [item]
    if layer.kind == "edge":
        rgba = _hex_rgba(colour, 255)
        item = pg.ImageItem(np.transpose(cased_edge_rgba(d["edge"], rgba), (1, 0, 2)), levels=(0, 255))
        item.setRect(QtCore.QRectF(*d["rect"]))
        item.setZValue(-15)
        return [item]
    if layer.kind == "bars":
        return [pg.BarGraphItem(x=d["x"], height=d["height"], width=d["width"], y0=d.get("y0", 0.0),
                                brush=_brush(layer, dark), pen=None)]
    if layer.kind == "step":
        if "edges" not in d:
            return []
        return [pg.PlotDataItem(np.asarray(d["edges"], float), np.asarray(d["values"], float), stepMode="center",
                                pen=pg.mkPen(colour, width=s.width))]
    if layer.kind == "curves":
        if "grid" in d:
            for curve in d["curves"]:
                items.append(pg.PlotDataItem(d["grid"], curve, pen=pg.mkPen(colour, width=s.width,
                                                                            style=_pen_style(s.pen))))
            for m in d.get("means", []):
                items.append(pg.InfiniteLine(pos=float(m), angle=90,
                                             pen=pg.mkPen(colour, width=1, style=QtCore.Qt.PenStyle.DotLine)))
        else:
            items.append(pg.PlotDataItem(d["x"], d["y"], pen=pg.mkPen(colour, width=s.width), name=layer.label))
        return items
    if layer.kind == "vlines":
        pens = list(d.get("pens", [])) or ["solid"]
        for i, pos in enumerate(d.get("positions", [])):
            label = d.get("labels", [None] * (i + 1))[i] if d.get("labels") else None
            pen_name = pens[i] if i < len(pens) else pens[-1]
            kw: Dict[str, Any] = {}
            if label:
                kw = {"label": label, "labelOpts": {"position": 0.9 - 0.15 * (i % 3), "color": colour}}
            items.append(pg.InfiniteLine(pos=float(pos), angle=90,
                                         pen=pg.mkPen(colour, width=s.width, style=_pen_style(pen_name)), **kw))
        return items
    if layer.kind == "band":
        region = pg.LinearRegionItem(values=(d["lo"], d["hi"]), movable=False)
        region.setBrush(pg.mkBrush(*_hex_rgba(colour, s.alpha)))
        region.setZValue(-10)
        items.append(region)
        if "line" in d:
            items.append(pg.InfiniteLine(pos=float(d["line"]), angle=90,
                                         pen=pg.mkPen(colour, width=2, style=QtCore.Qt.PenStyle.DotLine),
                                         label=d.get("label"), labelOpts={"position": 0.95, "color": colour}))
        return items
    if layer.kind == "hline":
        return [pg.InfiniteLine(pos=float(d["y"]), angle=0, pen=pg.mkPen(colour, width=s.width,
                                                                          style=_pen_style(s.pen)),
                                # the label starts at the line's left end and runs right, under it: never cut by
                                # the left axis (centred on the end, half of it was), and clear of the legend's row
                                label=d.get("label"), labelOpts={"position": 0.05, "color": colour,
                                                                 "anchors": [(0, 0), (0, 0)]})]
    raise ValueError(f"unknown layer kind {layer.kind!r}")


class AxonMap(QtWidgets.QWidget):
    """The axon map (design 3). Feed it with ``set_inputs(MapInputs)``; choose a preset with ``set_view``."""

    view_changed = QtCore.pyqtSignal(str)
    redrawn = QtCore.pyqtSignal()
    # The widefield image chosen ("tubulin" | "spectrin"), each time it changes: the Axoplasm panel's own selector
    # follows it (the map's input is the one source of truth).
    image_changed = QtCore.pyqtSignal(str)
    # "Clear the overlay" pressed under the second method's rows (tools.method_overlay): the window drops it.
    overlay_clear_requested = QtCore.pyqtSignal()
    # F5: "Clear the methods" under the methods' rows; "Method diagnostics..." beside it.
    methods_clear_requested = QtCore.pyqtSignal()
    diagnostics_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, dark: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("axon_map")
        self.dark = bool(dark)
        self._inputs = L.MapInputs()
        self._view = "mps"
        self._base_view = "mps"
        self._applying = False
        self._groups: List[L.Group] = []
        self._layers: Dict[str, L.Layer] = {}
        # rows whose items are built only when ticked (the Voronoi cells): their keys, until then
        self._pending: Set[str] = set()
        self._fitted = False
        self._user_range = False
        self._user_split = False
        self._fit_key: Tuple[int, Optional[int]] = (0, None)
        self._data_extent: Optional[Tuple[float, float, float, float]] = None
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        # The plot takes the map's whole height (an axon is drawn as large as
        # the height allows, the aspect locked); the View combo, the two
        # provenance lines and the layers sit in a column beside it, which
        # takes the width the plot's shape leaves.
        self.side = QtWidgets.QWidget()
        self.side.setObjectName("map_side")
        side = QtWidgets.QVBoxLayout(self.side)
        side.setContentsMargins(4, 0, 0, 0)
        side.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel(VIEW_LABEL))
        self.combo_view = QtWidgets.QComboBox()
        self.combo_view.setObjectName("combo_view")
        for key in L.MAP_VIEWS:
            self.combo_view.addItem(L.MAP_VIEW_NAMES[key], key)
        self.combo_view.setToolTip(L.MAP_VIEW_TIP)
        self.combo_view.currentIndexChanged.connect(self._on_view_combo)
        top.addWidget(self.combo_view)
        top.addStretch(1)
        side.addLayout(top)
        self.lbl_line1 = QtWidgets.QLabel("")
        self.lbl_line1.setObjectName("map_provenance_1")
        self.lbl_line1.setWordWrap(True)
        self.lbl_line1.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_line2 = QtWidgets.QLabel("")
        self.lbl_line2.setObjectName("map_provenance_2")
        self.lbl_line2.setWordWrap(True)
        self.lbl_line2.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        # Manual test of 2026-10-23: the two provenance lines (four to eight lines once wrapped) sat above the
        # layers' scroll area and left it a strip of ~150 px at 1366 x 700, where a group's caption looked as if it
        # ran over them. They are the first thing in that scroll area now: shown as before, scrolled with the rows.
        self.provenance = QtWidgets.QWidget()
        self.provenance.setObjectName("map_provenance")
        prov = QtWidgets.QVBoxLayout(self.provenance)
        prov.setContentsMargins(0, 0, 0, 4)
        prov.setSpacing(2)
        prov.addWidget(self.lbl_line1)
        prov.addWidget(self.lbl_line2)
        self.lbl_mouse = QtWidgets.QLabel(RIGHT_DRAG_HELP)
        self.lbl_mouse.setObjectName("map_mouse_help")
        self.lbl_mouse.setWordWrap(True)
        self.lbl_mouse.setStyleSheet("color: gray;")
        prov.addWidget(self.lbl_mouse)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.plot = pg.PlotWidget()
        self.plot.setObjectName("map_plot")
        self.plot.setAspectLocked(True)
        self.plot.setLabels(bottom="x [nm]", left="y [nm]")
        # manual test of 2026-10-23: a right-button drag moves the map too (tools.plot_mouse)
        install_right_drag_pan(self.plot.getViewBox())
        self.layers = LayerPanel(scroll=True, max_width=SIDE_DRAG_MAX, hide_disabled=True, elide=True)
        self.layers.setObjectName("map_layers")
        self.layers.toggled.connect(self._on_toggled)
        side.addWidget(self.layers, 1)
        self.side.setMinimumWidth(PANEL_WIDTH)
        self.side.setMaximumWidth(SIDE_DRAG_MAX)
        self.splitter.addWidget(self.plot)
        self.splitter.addWidget(self.side)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, True)
        self.splitter.splitterMoved.connect(self._on_splitter_moved)
        self.splitter.splitterMoved.connect(lambda *_a: QtCore.QTimer.singleShot(0, self._fit_title))
        self.plot.getViewBox().sigRangeChangedManually.connect(self._on_range_by_hand)
        root.addWidget(self.splitter, 1)
        # Manual test 2, item 14: a click on a cluster centre says which cluster it is, its area and its distances
        # to its nearest centres, and rings it and its neighbours on the map (display only).
        self.lbl_cluster = QtWidgets.QLabel("")
        self.lbl_cluster.setObjectName("map_cluster_info")
        self.lbl_cluster.setWordWrap(True)
        self.lbl_cluster.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                                                 | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.lbl_cluster.setToolTip(CLUSTER_INFO_TIP)
        self.lbl_cluster.setVisible(False)
        # the clicked cluster's line stays in sight above the layers (it is hidden until a click)
        side.insertWidget(side.indexOf(self.layers), self.lbl_cluster)
        self.layers.set_preamble(self.provenance)
        self._cluster_items: List[Any] = []
        self._cluster_info: Optional[Dict[str, Any]] = None
        self.plot.scene().sigMouseClicked.connect(self._on_scene_click)
        # the group headers' selectors (owned here, shown in the panel)
        self.combo_source = QtWidgets.QComboBox()
        self.combo_source.setObjectName("combo_source")
        for key, name in L.SOURCES.items():
            self.combo_source.addItem(name, key)
        self.combo_source.setToolTip(SOURCE_TIP)
        self.combo_source.currentIndexChanged.connect(self._on_header)
        self.combo_colour = QtWidgets.QComboBox()
        self.combo_colour.setObjectName("combo_colour_by")
        for key, name in L.COLOURINGS.items():
            self.combo_colour.addItem(name, key)
        self.combo_colour.setToolTip(COLOUR_TIP)
        self.combo_colour.currentIndexChanged.connect(self._on_header)
        self.combo_image = QtWidgets.QComboBox()
        self.combo_image.setObjectName("combo_image")
        for key, name in L.IMAGES.items():
            self.combo_image.addItem(name, key)
        self.combo_image.setToolTip(IMAGE_TIP)
        self.combo_image.currentIndexChanged.connect(self._on_header)
        for combo in (self.combo_source, self.combo_colour, self.combo_image):
            # a narrow side column must not be held open by a combo's longest item (the full item is in its list)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(14)
        self._style()
        self.redraw()

    # ------------------------------------------------------------------ public
    def inputs(self) -> L.MapInputs:
        return self._inputs

    def set_inputs(self, inputs: L.MapInputs, view: Optional[str] = None) -> None:
        """Draw from new inputs. ``view`` applies a preset (its source and colouring included); without it the
        source, colouring and image follow the header selectors and every row keeps its state."""
        self._inputs = inputs
        key = (id(inputs.analysis), inputs.selection_n)
        if key != self._fit_key:
            # another analysis or selection: the view fits the new data, even after a zoom by hand
            self._fit_key = key
            self._user_range = False
        if view is not None:
            self.set_view(view)
            return
        note = self._fall_back_from_selection()
        self._sync_selectors()
        self.redraw()
        if note and "localizations" in self.layers.groups():
            self.layers.set_group_caption("localizations", note + " " + self.layers.group_caption("localizations"))

    def _fall_back_from_selection(self) -> str:
        """Manual test of 2026-10-23: a remembered 'Main-window selection (axoplasm classes)' with the Axoplasm
        panel closed left the Localizations group greyed out. New inputs that cannot place the selection show the
        MPS analysis slab instead, and the group's caption says why. Returns that note ('' when nothing fell
        back). A source chosen by hand in the selector is left as chosen (its group says what is missing)."""
        inp = self._inputs
        st = inp.axoplasm
        if inp.source != "selection" or (st is not None and st.loc_x is not None):
            return ""
        inp.source = "slab"
        return (f"{L.SOURCES['selection']} is not available ({inp.axoplasm_reason}): showing the "
                f"{L.SOURCES['slab']}.")

    def view(self) -> str:
        return self._view

    def set_image(self, key: str) -> None:
        """Choose the widefield image under the map, as its group's selector does (the Axoplasm panel's selector
        calls this)."""
        if key not in L.IMAGES:
            raise ValueError(f"unknown image {key!r}")
        index = self.combo_image.findData(key)
        if index != self.combo_image.currentIndex():
            self.combo_image.setCurrentIndex(index)

    def set_view(self, view: str) -> None:
        """A preset of the rows (3.2): its source, its colouring and the rows it ticks."""
        if view not in L.MAP_VIEWS:
            raise ValueError(f"unknown view {view!r}")
        self._view = view
        if view != "custom":
            self._base_view = view
            self._inputs.source = L.VIEW_SOURCE[view]
            self._inputs.colour_by = L.VIEW_COLOUR_BY[view]
        self._sync_selectors()
        self._applying = True
        try:
            self.redraw()
            if view != "custom":
                for key in self.layers.keys():
                    self.layers.set_visible(key, L.view_on(view, key))
        finally:
            self._applying = False
        self._apply_rules()
        self._retitle()
        self.view_changed.emit(view)

    def plot_item(self) -> Any:
        return self.plot.getPlotItem()

    def title(self) -> str:
        return L.map_title(self._inputs, set(self.layers.shown_keys()))

    def title_lines(self) -> List[str]:
        return self.title().split("<br>")

    def set_provenance(self, line1: str, line2: str) -> None:
        """The two provenance lines above the plot (3.1): line 1 is the window's provenance text, line 2 the
        builder's ``provenance_line2``."""
        self.lbl_line1.setText(line1)
        self.lbl_line2.setText(line2)

    def offers_white(self) -> bool:
        """A white redraw is offered unless the widefield image is drawn (a photograph, 3.6)."""
        return not ("image" in self.layers.keys() and "image" in self.layers.shown_keys())

    def export_message(self) -> str:
        """What the export dialog says: the visible layers, how many are hidden, both provenance lines and the
        origin of the values they name (3.6)."""
        shown = self.layers.shown()
        hidden = [k for k in self.layers.keys() if k not in self.layers.shown_keys()]
        lines = ["Visible layers: " + ("; ".join(shown) if shown else "none") + ".",
                 f"{len(hidden)} layer{'s' if len(hidden) != 1 else ''} hidden or not available."]
        for text in (self.lbl_line1.text(), self.lbl_line2.text()):
            stripped = _strip_tags(str(text).replace("&nbsp;", " "))
            if stripped:
                lines.append(stripped)
        a = self._inputs.shown_analysis()
        if a is not None:
            origins = [("dbscan.eps_nm", float(a.eps_nm)), ("dbscan.min_samples", int(a.min_samples)),
                       ("slab.half_width_nm", float(a.slab_half_width_nm)),
                       ("occupancy.mahalanobis", float(a.mahalanobis_threshold))]
            lines.append("Origins: " + ", ".join(f"{reg.info(k).label} {reg.format_value(k, v)} "
                                                 f"({reg.badge(k, v)})" for k, v in origins) + ".")
        lines.append("Title: " + " | ".join(self.title_lines()))
        return "\n".join(lines)

    def set_dark(self, dark: bool) -> None:
        """Dark on screen; white while a figure is written (the window's existing white path)."""
        self.dark = bool(dark)
        self._style()
        self.redraw()

    def items(self, key: str) -> List[Any]:
        return self.layers.items(key)

    def layer(self, key: str) -> L.Layer:
        return self._layers[key]

    # ------------------------------------------------------------------ drawing
    # ------------------------------------------------------------------ a click on a cluster (item 14)
    def _on_scene_click(self, event: Any) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        vb = self.plot.getViewBox()
        pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        p = vb.mapSceneToView(pos)
        self.select_at(float(p.x()), float(p.y()))

    def _tolerance_nm(self) -> float:
        vb = self.plot.getViewBox()
        (x0, x1), _y = vb.viewRange()
        width_px = max(1.0, float(vb.width()))
        return float(CLICK_TOLERANCE_PX * abs(float(x1) - float(x0)) / width_px)

    def select_at(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        """The cluster centre at (x, y) nm (within the click tolerance), selected; None clears the selection."""
        shown = self._inputs.shown_analysis()
        if shown is None or getattr(shown, "centroids", None) is None:
            self.clear_cluster()
            return None
        i = L.nearest_centre(shown.centroids, x, y, self._tolerance_nm())
        if i is None:
            self.clear_cluster()
            return None
        return self.select_cluster(i)

    def select_cluster(self, index: int) -> Dict[str, Any]:
        """Show centre ``index`` of the analysis drawn: the info line, and rings on it and its neighbours."""
        self.clear_cluster()
        shown = self._inputs.shown_analysis()
        info = L.cluster_info(shown, self._inputs.analysis, int(index))
        self._cluster_info = info
        self.lbl_cluster.setText(L.cluster_info_text(info))
        self.lbl_cluster.setVisible(True)
        c = np.asarray(shown.centroids, dtype=float).reshape(-1, 2)
        ink = neutral(dark=self.dark)
        cx, cy = info["centre"]
        for j in info["neighbours"]:
            line = pg.PlotDataItem([cx, c[j, 0]], [cy, c[j, 1]],
                                   pen=pg.mkPen(ink, width=1, style=QtCore.Qt.PenStyle.DashLine))
            self._cluster_items.append(line)
        if info["neighbours"]:
            nb = c[info["neighbours"]]
            self._cluster_items.append(pg.ScatterPlotItem(
                nb[:, 0], nb[:, 1], symbol="o", size=16, pen=pg.mkPen(ink, width=1.5), brush=None))
        self._cluster_items.append(pg.ScatterPlotItem(
            [cx], [cy], symbol="o", size=22, pen=pg.mkPen(ink, width=2.5), brush=None))
        for item in self._cluster_items:
            item.setZValue(1e6)
            self.plot.addItem(item)
        return info

    def clear_cluster(self) -> None:
        for item in self._cluster_items:
            self.plot.removeItem(item)
        self._cluster_items = []
        self._cluster_info = None
        self.lbl_cluster.setText("")
        self.lbl_cluster.setVisible(False)

    def cluster_info(self) -> Optional[Dict[str, Any]]:
        """What the last click selected (None: nothing)."""
        return self._cluster_info

    def redraw(self) -> None:
        # a new drawing: the clicked cluster's rings are gone with the plot's items
        self._cluster_items = []
        self._cluster_info = None
        if hasattr(self, "lbl_cluster"):
            self.lbl_cluster.setVisible(False)
        self.plot.clear()
        self.layers.clear_layers(keep_state=True)
        self._groups = L.build_map(self._inputs)
        if self._base_view == "axoplasm":
            # the Axoplasm view is about the images: their group (and its selector) first, above the fold
            self._groups.sort(key=lambda g: g.key != "images")
        self._sync_image_items()
        self._layers = {}
        headers = {"source": self.combo_source, "colour_by": self.combo_colour, "image": self.combo_image}
        was = self._applying
        self._applying = True
        try:
            self._pending = set()
            for g in self._groups:
                header = headers.get(g.header or "")
                self.layers.add_group(g.title, key=g.key, caption=g.caption, header=header, collapsible=True)
                for layer in g.layers:
                    self._layers[layer.key] = layer
                    if layer.kind == "voronoi" and not (layer.enabled and g.enabled and self._starts_on(layer.key)):
                        # thousands of cells: built when the row is ticked (_on_toggled), not at every redraw
                        items = []
                        self._pending.add(layer.key)
                    else:
                        items = render_layer(layer, self.dark)
                    for item in items:
                        self.plot.addItem(item)
                    self.layers.add_layer(layer.key, layer.label, swatch_for(layer, True), items=items,
                                          count=layer.count, tip=layer.tip,
                                          visible=L.view_on(self._base_view, layer.key))
                    if not layer.enabled:
                        self.layers.set_enabled(layer.key, False, layer.reason)
                scale = next((layer.data.get("scale") for layer in g.layers if layer.kind == "voronoi"
                              and layer.data.get("scale") is not None), None)
                if scale is not None and g.layers:
                    self.layers.add_widget(DensityScaleBar(scale), key=g.layers[-1].key)
                if g.key == "overlay" and g.layers:
                    clear = QtWidgets.QPushButton("Clear the overlay")
                    clear.setObjectName("btn_clear_overlay")
                    clear.setToolTip("Remove the second method's outlines from the map (the analysis shown does not "
                                     "change; 'Compare methods...' can draw another).")
                    clear.clicked.connect(self.overlay_clear_requested)
                    self.layers.add_widget(clear, key=g.layers[-1].key)
                if g.key == "methods" and g.layers:
                    row = QtWidgets.QWidget()
                    hl = QtWidgets.QHBoxLayout(row)
                    hl.setContentsMargins(0, 0, 0, 0)
                    diag = QtWidgets.QPushButton("Method diagnostics...")
                    diag.setObjectName("btn_method_diagnostics")
                    diag.setToolTip("Each method's own diagnostic plot, where one is already computed or cheap: "
                                    "the k-distance graph of DBSCAN, the density of the Voronoi cells, the "
                                    "persistence diagram of ToMATo. Each can be exported as a PNG.")
                    diag.clicked.connect(self.diagnostics_requested)
                    clear = QtWidgets.QPushButton("Clear the methods")
                    clear.setObjectName("btn_clear_methods")
                    clear.setToolTip("Remove the compared methods' rows from the map (the analysis shown does not "
                                     "change; 'Compare methods...' can run them again).")
                    clear.clicked.connect(self.methods_clear_requested)
                    hl.addWidget(diag)
                    hl.addWidget(clear)
                    self.layers.add_widget(row, key=g.layers[-1].key)
                if not g.enabled:
                    self.layers.set_group_enabled(g.key, False, g.reason)
        finally:
            self._applying = was
        self._apply_rules()
        self._mark_points_over_cells()
        self._retitle()
        self._data_extent = self._extent()
        if not self._user_range:
            self.fit_view()
        self.redrawn.emit()

    # ------------------------------------------------------------------ the view
    def _extent(self) -> Optional[Tuple[float, float, float, float]]:
        """The lab extent of what the rows draw (every row with points, shown or not, so a tick does not move the
        view; never the widefield image, whose field is far larger than the axon)."""
        lo_x = lo_y = np.inf
        hi_x = hi_y = -np.inf
        for layer in self._layers.values():
            if layer.kind == "image":
                continue
            d = layer.data
            if "x" not in d or "y" not in d:
                continue
            x = np.asarray(d["x"], dtype=float).ravel()
            y = np.asarray(d["y"], dtype=float).ravel()
            if x.size == 0 or x.size != y.size:
                continue
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            lo_x, hi_x = min(lo_x, float(x[ok].min())), max(hi_x, float(x[ok].max()))
            lo_y, hi_y = min(lo_y, float(y[ok].min())), max(hi_y, float(y[ok].max()))
        if not np.isfinite([lo_x, hi_x, lo_y, hi_y]).all():
            return None
        return lo_x, hi_x, lo_y, hi_y

    def fit_view(self) -> None:
        """Fit the view to the data's extent with a small padding, the aspect locked (what is drawn is unchanged);
        then give the side column the width the plot's shape leaves."""
        ext = self._data_extent
        if ext is None:
            self.plot.getViewBox().enableAutoRange()
            return
        x0, x1, y0, y1 = ext
        pad = max(FIT_PAD_FRACTION * max(x1 - x0, y1 - y0), FIT_PAD_MIN_NM)
        self._fitting = True
        try:
            self.plot.setRange(xRange=(x0 - pad, x1 + pad), yRange=(y0 - pad, y1 + pad), padding=0.0)
        finally:
            self._fitting = False
        self._fitted = True
        self._apply_split()

    def view_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """The data extent the view was fitted to (x0, x1, y0, y1), or None with nothing drawn."""
        return self._data_extent

    def _apply_split(self) -> None:
        """The plot as wide as the data's shape needs at the plot's height; the side column gets the rest, between
        its narrowest and widest. Until the user drags the divider."""
        if self._user_split or self._data_extent is None:
            return
        total = sum(self.splitter.sizes())
        vb = self.plot.getViewBox()
        vb_h = float(vb.height())
        if total <= 0 or vb_h <= 0:
            return
        x0, x1, y0, y1 = self._data_extent
        pad = max(FIT_PAD_FRACTION * max(x1 - x0, y1 - y0), FIT_PAD_MIN_NM)
        aspect = (x1 - x0 + 2 * pad) / max(y1 - y0 + 2 * pad, 1e-9)
        frame = max(0.0, float(self.plot.width()) - float(vb.width()))
        axis = float(self.plot.getPlotItem().getAxis("left").width())
        # never narrower than the title on its own lines, when the side column can give the room
        ideal_plot = int(max(vb_h * aspect + frame, title_natural_width(self.plot) + axis + 20.0))
        side = int(min(SIDE_MAX_WIDTH, max(PANEL_WIDTH, total - ideal_plot)))
        if abs(self.splitter.sizes()[1] - side) <= 2:
            return
        self._splitting = True
        try:
            self.splitter.setSizes([total - side, side])
        finally:
            self._splitting = False
        # the plot has another width: its title and its view follow once the layout has settled
        QtCore.QTimer.singleShot(0, self._after_resize)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._after_resize)

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._after_resize)

    def _after_resize(self) -> None:
        self._apply_split()
        self._fit_title()
        if not self._user_range and self._data_extent is not None:
            self.fit_view()

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        if not getattr(self, "_splitting", False):
            self._user_split = True

    def _on_range_by_hand(self, *_args: Any) -> None:
        if not getattr(self, "_fitting", False):
            self._user_range = True

    def _apply_rules(self) -> None:
        """Rows that cannot be read over what else is drawn (3.5): grey on the grey image."""
        image_on = "image" in self.layers.keys() and "image" in self.layers.shown_keys()
        for key in L.OVER_IMAGE_DISABLED:
            if key not in self._layers:
                continue
            layer = self._layers[key]
            if image_on:
                self.layers.set_enabled(key, False, L.OVER_IMAGE_REASON)
            elif layer.enabled and self._group_enabled(key):
                self.layers.set_enabled(key, True)

    def _group_enabled(self, key: str) -> bool:
        group = self.layers.group_of(key)
        return group is None or self.layers.is_group_enabled(group)

    def _retitle(self) -> None:
        set_title(self.plot, self.title(), dark=self.dark)
        self._fit_title()

    def _title_room(self) -> float:
        """The width a title may take: the plot's, less its left axis and a margin."""
        axis = self.plot.getPlotItem().getAxis("left")
        return float(self.plot.width()) - float(axis.width()) - 16.0

    def _fit_title(self) -> None:
        """Wrap the title only when the plot is narrower than it (the words do not change)."""
        if self.plot.width() > 50:
            fit_title_width(self.plot, self._title_room())

    def _style(self) -> None:
        (style_dark if self.dark else style_light)(self.plot)

    def _sync_image_items(self) -> None:
        """betaII-spectrin can be chosen only once the Axoplasm panel has it loaded; the item says why not."""
        st = self._inputs.axoplasm
        missing = st is not None and st.has_mask and st.spectrin_region is None
        item = self.combo_image.model().item(self.combo_image.findData("spectrin"))
        item.setEnabled(not missing)
        item.setToolTip("Not available: " + L.SPECTRIN_NOT_LOADED + "." if missing else "")

    def _sync_selectors(self) -> None:
        for combo, value in ((self.combo_source, self._inputs.source), (self.combo_colour, self._inputs.colour_by),
                             (self.combo_image, self._inputs.image)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            combo.blockSignals(False)
        self.combo_view.blockSignals(True)
        self.combo_view.setCurrentIndex(max(0, self.combo_view.findData(self._view)))
        self.combo_view.blockSignals(False)

    # ------------------------------------------------------------------ interaction
    def _on_view_combo(self, _index: int) -> None:
        view = str(self.combo_view.currentData())
        if view != self._view:
            self.set_view(view)

    def _on_header(self, _index: int) -> None:
        old_source, old_colour, old_image = self._inputs.source, self._inputs.colour_by, self._inputs.image
        self._inputs.source = str(self.combo_source.currentData())
        self._inputs.colour_by = str(self.combo_colour.currentData())
        self._inputs.image = str(self.combo_image.currentData())
        self._to_custom()
        self.redraw()
        if self._inputs.image != old_image:
            self.image_changed.emit(self._inputs.image)
        # The rows of a source or a colouring chosen here start as the view that uses it ticks them; left
        # unticked, the map would lose its localizations or centres without a word.
        picks: List[Tuple[str, List[str]]] = []
        if self._inputs.source != old_source:
            picks.append(("localizations", [v for v, src in L.VIEW_SOURCE.items() if src == self._inputs.source]))
        if self._inputs.colour_by != old_colour:
            picks.append(("centres", [v for v, c in L.VIEW_COLOUR_BY.items() if c == self._inputs.colour_by]))
        if not picks:
            return
        was = self._applying
        self._applying = True
        try:
            for group, views in picks:
                for key in self.layers.keys():
                    if self.layers.group_of(key) == group:
                        self.layers.set_visible(key, any(L.view_on(v, key) for v in views))
        finally:
            self._applying = was
        self._apply_rules()
        self._retitle()

    def _starts_on(self, key: str) -> bool:
        """Whether the row of ``key`` is ticked in the drawing being built: its remembered state, else the view's."""
        return bool(self.layers.state().get(key, L.view_on(self._base_view, key)))

    def build_pending(self, key: str) -> List[Any]:
        """Build the items of a row left empty at the redraw (the Voronoi cells), add them to the plot and bind them
        to the row. Returns them ([] when there is nothing to build)."""
        if key not in self._pending or key not in self._layers:
            return []
        self._pending.discard(key)
        items = render_layer(self._layers[key], self.dark)
        for item in items:
            self.plot.addItem(item)
        self.layers.bind(key, items)
        return items

    def _points_for_cells(self, key: str) -> Optional[str]:
        """The localization row drawn over the cells of ``key`` ("vor_in" -> "slab_kept", "vor_out" ->
        "slab_noise"; a compared method's cells have no points of their own on this map), or None."""
        return VORONOI_POINTS.get(key)

    def _tick_points_under_cells(self, key: str) -> None:
        """Plan E's picture is every localization inside its own cell (manual test of 2026-10-23: with the
        localization rows off the cells were drawn with no point in them, and read as "the cells are gone"). A
        Voronoi row ticked while none of the slab's localization rows is drawn ticks the matching one, once; it
        can be unticked again."""
        points = self._points_for_cells(key)
        if points is None or points not in self.layers.keys():
            return
        if set(self.layers.shown_keys()) & SLAB_POINT_KEYS:
            return
        self.layers.set_visible(points, True)

    def _mark_points_over_cells(self) -> None:
        """While some Voronoi cells of the analysis shown are drawn, the slab's points carry a dark rim and are a
        little larger; otherwise they are drawn as their style says. Display only."""
        shown = set(self.layers.shown_keys())
        cells = bool(shown & set(VORONOI_POINTS))
        for key in SLAB_POINT_KEYS:
            if key not in self._layers:
                continue
            for item in self.layers.items(key):
                if not isinstance(item, pg.ScatterPlotItem):
                    continue
                if not hasattr(item, "_style_pen"):
                    item._style_pen = item.opts.get("pen")
                    item._style_size = item.opts.get("size")
                if cells:
                    item.setPen(pg.mkPen(DARK_RIM, width=POINTS_OVER_CELLS_RIM))
                    item.setSize(max(float(item._style_size or 0.0), POINTS_OVER_CELLS_SIZE))
                else:
                    item.setPen(item._style_pen)
                    item.setSize(item._style_size)

    def _on_toggled(self, key: str, _on: bool) -> None:
        if _on and key in self._pending:
            self.build_pending(key)
        if _on and not self._applying:
            self._tick_points_under_cells(key)
        if key in VORONOI_POINTS or key in SLAB_POINT_KEYS:
            self._mark_points_over_cells()
        if key == "image":
            self._apply_rules()
        self._retitle()
        if not self._applying:
            self._to_custom()

    def _to_custom(self) -> None:
        if self._view != "custom":
            self._view = "custom"
            self.combo_view.blockSignals(True)
            self.combo_view.setCurrentIndex(self.combo_view.findData("custom"))
            self.combo_view.blockSignals(False)
            self.view_changed.emit("custom")


def _strip_tags(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def ticked(panel: LayerPanel) -> Set[str]:
    """The rows of a panel drawn now (ticked and available)."""
    return set(panel.shown_keys())
