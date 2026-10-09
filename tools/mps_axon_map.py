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

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt5 import QtCore, QtWidgets

from tools import mps_axon_map_layers as L
from tools import mps_param_registry as reg
from tools.mps_layer_panel import LayerPanel, Swatch
from tools.mps_plot_style import (
    fit_title_width, neutral, role, segment_colour, segment_symbol, set_title, style_dark, style_light,
    title_natural_width)

__all__ = ["AxonMap", "render_layer", "swatch_for", "cased_edge_rgba", "colour_of"]

DARK_RIM = "#000000"
PANEL_WIDTH = 280          # the side column's narrowest width
SIDE_MAX_WIDTH = 460       # and its widest: what the plot's shape leaves, up to this
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
    return {"dash": QtCore.Qt.PenStyle.DashLine, "dot": QtCore.Qt.PenStyle.DotLine}.get(
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
    return Swatch("line", colour, dash=s.pen == "dash", width=min(3.0, max(1.5, s.width)), cased=s.cased)


def _brush(layer: L.Layer, dark: bool) -> Any:
    s = layer.style
    return pg.mkBrush(*_hex_rgba(colour_of(layer, dark), s.alpha))


def render_layer(layer: L.Layer, dark: bool = True) -> List[Any]:
    """The pyqtgraph items of one row (empty when it has nothing to draw)."""
    s = layer.style
    d = layer.data
    colour = colour_of(layer, dark)
    items: List[Any] = []
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
        side.addWidget(self.lbl_line1)
        side.addWidget(self.lbl_line2)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.plot = pg.PlotWidget()
        self.plot.setObjectName("map_plot")
        self.plot.setAspectLocked(True)
        self.plot.setLabels(bottom="x [nm]", left="y [nm]")
        self.layers = LayerPanel(scroll=True, max_width=SIDE_MAX_WIDTH, hide_disabled=True, elide=True)
        self.layers.setObjectName("map_layers")
        self.layers.toggled.connect(self._on_toggled)
        side.addWidget(self.layers, 1)
        self.side.setMinimumWidth(PANEL_WIDTH)
        self.side.setMaximumWidth(SIDE_MAX_WIDTH)
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
        self._sync_selectors()
        self.redraw()

    def view(self) -> str:
        return self._view

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
    def redraw(self) -> None:
        self.plot.clear()
        self.layers.clear_layers(keep_state=True)
        self._groups = L.build_map(self._inputs)
        self._layers = {}
        headers = {"source": self.combo_source, "colour_by": self.combo_colour, "image": self.combo_image}
        was = self._applying
        self._applying = True
        try:
            for g in self._groups:
                header = headers.get(g.header or "")
                self.layers.add_group(g.title, key=g.key, caption=g.caption, header=header, collapsible=True)
                for layer in g.layers:
                    self._layers[layer.key] = layer
                    items = render_layer(layer, self.dark)
                    for item in items:
                        self.plot.addItem(item)
                    self.layers.add_layer(layer.key, layer.label, swatch_for(layer, True), items=items,
                                          count=layer.count, tip=layer.tip,
                                          visible=L.view_on(self._base_view, layer.key))
                    if not layer.enabled:
                        self.layers.set_enabled(layer.key, False, layer.reason)
                if not g.enabled:
                    self.layers.set_group_enabled(g.key, False, g.reason)
        finally:
            self._applying = was
        self._apply_rules()
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
        old_source, old_colour = self._inputs.source, self._inputs.colour_by
        self._inputs.source = str(self.combo_source.currentData())
        self._inputs.colour_by = str(self.combo_colour.currentData())
        self._inputs.image = str(self.combo_image.currentData())
        self._to_custom()
        self.redraw()
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

    def _on_toggled(self, key: str, _on: bool) -> None:
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
