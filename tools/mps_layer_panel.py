# -*- coding: utf-8 -*-
"""
A column of checkboxes beside a plot, one per thing the plot draws: the layer panel.

Why it exists (UI stage 0, the user's manual test of 2026-10-05): the legends drawn INSIDE the plots covered part of
the data, and clicking most of their entries crossed out the entry's eye while the curve it named stayed drawn,
because the entry was a stand-in that had never been added to the plot. This panel lives outside the plot, and every
row is bound to the real items it names: unticking it hides them, ticking it shows them again.

Two pieces, both free of any analysis import so that any window (and the axon map of stage 2) can use them:

``LayerPanel``
    A QWidget with groups of checkbox rows, each row with a swatch drawn like the plot draws the layer (the
    Axoplasm window's panel, ``tools/mps_axoplasm_window.py``, is the model), and "Show all" / "Hide all". Rows are
    addressed by a stable ASCII key; the text shown is a separate label (stage 4 translates labels, never keys).
    What a row shows survives a redraw: ``bind`` gives a row the items of the new drawing and applies the
    remembered state at once, and ``clear_layers`` forgets the rows but not their state.

``LegendLayers``
    For a pyqtgraph ``LegendItem`` that must stay where it is (the Z quality view keeps its legends in their own
    column, outside the plots): each entry gets a stand-in sample whose ``setVisible`` also hides or shows a GROUP of
    real items. pyqtgraph's ``ItemSample.mouseClickEvent`` toggles ``item.setVisible`` on the sample's item and paints
    the crossed eye from ``item.isVisible()``, so a click on the entry now drives the real items.

Display text here is short plain English; nothing in a key, hash or exported column depends on it.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from tools.mps_plot_style import PLOT_BG, TITLE_FG, neutral

__all__ = ["LegendLayers", "LayerPanel", "Swatch", "swatch_icon"]

SWATCH_KINDS = ("symbol", "line", "bar", "image")
PANEL_MAX_WIDTH = 240
SHOW_ALL_TIP = "Draw every layer listed here."
HIDE_ALL_TIP = "Hide every layer listed here, then tick the ones you want to see."


@dataclass(frozen=True)
class Swatch:
    """How a row's icon is drawn: like the plot draws the layer.

    kind     "symbol" (a marker: ``symbol`` filled with ``colour``, outlined with ``pen``), "line" (solid, or dashed
             when ``dash``), "bar" (a filled square) or "image" (a grey gradient square).
    colour   a colour pyqtgraph understands ("#rrggbb", a QColor, an (r, g, b[, a]) tuple...).
    pen      the marker's outline colour (None: the neutral line colour of a dark plot).
    hollow   a marker drawn as an outline only, in ``colour`` (the Rings window's segments).
    """
    kind: str
    colour: Any = "#9a9a9a"
    symbol: str = "o"
    pen: Any = None
    dash: bool = False
    hollow: bool = False
    width: float = 2.0


def swatch_icon(sw: Swatch, size: int = 16) -> QtGui.QIcon:
    """The swatch on the plots' own dark background, so a colour reads as it does in the plot whatever the window's
    theme."""
    if sw.kind not in SWATCH_KINDS:
        raise ValueError(f"unknown swatch kind {sw.kind!r}")
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtGui.QPen(QtGui.QColor("#555555"), 1))
    painter.setBrush(QtGui.QBrush(pg.mkColor(PLOT_BG)))
    painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, size - 1.0, size - 1.0), 2.0, 2.0)
    ink = pg.mkColor(sw.colour)
    if sw.kind == "symbol":
        from pyqtgraph.graphicsItems.ScatterPlotItem import drawSymbol
        painter.translate(size / 2.0, size / 2.0)
        if sw.hollow:
            pen = pg.mkPen(ink, width=1.5)
            brush = pg.mkBrush(None)
        else:
            pen = pg.mkPen(sw.pen if sw.pen is not None else neutral(dark=True), width=1)
            brush = pg.mkBrush(ink)
        drawSymbol(painter, sw.symbol, size - 5.0, pen, brush)
    elif sw.kind == "line":
        pen = QtGui.QPen(ink, sw.width)
        if sw.dash:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QtCore.QPointF(2.0, size / 2.0), QtCore.QPointF(size - 2.0, size / 2.0))
    elif sw.kind == "bar":
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(ink))
        painter.drawRect(QtCore.QRectF(3.0, 3.0, size - 6.0, size - 6.0))
    else:
        grad = QtGui.QLinearGradient(0.0, 0.0, float(size), float(size))
        grad.setColorAt(0.0, QtGui.QColor("#202020"))
        grad.setColorAt(1.0, QtGui.QColor("#c8c8c8"))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(grad))
        painter.drawRect(QtCore.QRectF(2.0, 2.0, size - 4.0, size - 4.0))
    painter.end()
    return QtGui.QIcon(pixmap)


@dataclass
class _Layer:
    key: str
    label: str
    box: QtWidgets.QCheckBox
    items: List[Any]
    on_toggle: Optional[Callable[[bool], Any]]
    tip: str
    count: Optional[int]
    owned: bool          # the panel made the checkbox (and removes it in clear_layers)
    extras: List[QtWidgets.QWidget]


class LayerPanel(QtWidgets.QWidget):
    """Checkbox rows beside a plot, each bound to the real items it names.

    ``toggled(key, visible)`` is emitted after a row's items were shown or hidden and its ``on_toggle`` called.
    ``dark`` styles the group titles for a dark window (the Axoplasm window); otherwise they use the window's own
    text colour.
    """

    toggled = QtCore.pyqtSignal(str, bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, dark: bool = False,
                 max_width: int = PANEL_MAX_WIDTH) -> None:
        super().__init__(parent)
        self._dark = bool(dark)
        self._layers: Dict[str, _Layer] = {}
        self._state: Dict[str, bool] = {}
        self._groups: List[QtWidgets.QWidget] = []
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 0, 0, 0)
        outer.setSpacing(2)
        self._rows = QtWidgets.QVBoxLayout()
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(2)
        outer.addLayout(self._rows)
        outer.addSpacing(6)
        footer = QtWidgets.QHBoxLayout()
        self.show_all_button = QtWidgets.QPushButton("Show all")
        self.show_all_button.setObjectName("layers_show_all")
        self.show_all_button.setToolTip(SHOW_ALL_TIP)
        self.show_all_button.clicked.connect(lambda: self.show_all())
        self.hide_all_button = QtWidgets.QPushButton("Hide all")
        self.hide_all_button.setObjectName("layers_hide_all")
        self.hide_all_button.setToolTip(HIDE_ALL_TIP)
        self.hide_all_button.clicked.connect(lambda: self.hide_all())
        footer.addWidget(self.show_all_button)
        footer.addWidget(self.hide_all_button)
        outer.addLayout(footer)
        outer.addStretch(1)
        self.setMaximumWidth(int(max_width))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)

    # ------------------------------------------------------------ building
    def add_group(self, title: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(title)
        style = "font-weight: bold;" + (f" color: {TITLE_FG};" if self._dark else "")
        label.setStyleSheet(style)
        if self._groups or self._layers:
            self._rows.addSpacing(6)
        self._rows.addWidget(label)
        self._groups.append(label)
        return label

    def add_layer(self, key: str, label: str, swatch: Optional[Swatch] = None, *, items: Iterable[Any] = (),
                  on_toggle: Optional[Callable[[bool], Any]] = None, tip: str = "", count: Optional[int] = None,
                  visible: bool = True, checkbox: Optional[QtWidgets.QCheckBox] = None) -> QtWidgets.QCheckBox:
        """A row for ``key``. The state remembered for the key (an earlier row of the same key, ``restore``) wins
        over ``visible``. ``checkbox`` adopts an existing box (its object name and checked state are kept); the
        items are shown or hidden at once, ``on_toggle`` is called only when the user (or the code) toggles."""
        if key in self._layers:
            raise ValueError(f"layer {key!r} is already in the panel")
        owned = checkbox is None
        box = checkbox if checkbox is not None else QtWidgets.QCheckBox()
        if owned:
            box.setObjectName(f"layer_{key}")
            on = bool(self._state.get(key, visible))
        else:
            on = bool(self._state.get(key, box.isChecked()))
        if swatch is not None:
            box.setIcon(swatch_icon(swatch))
        layer = _Layer(key=key, label=label, box=box, items=list(items), on_toggle=on_toggle, tip=tip, count=count,
                       owned=owned, extras=[])
        self._layers[key] = layer
        self._state[key] = on
        box.blockSignals(True)
        box.setChecked(on)
        box.blockSignals(False)
        self._set_text(layer)
        box.setToolTip(tip)
        box.toggled.connect(lambda checked, k=key: self._on_box(k, bool(checked)))
        self._rows.addWidget(box)
        self._apply(layer)
        return box

    def add_widget(self, widget: QtWidgets.QWidget, key: Optional[str] = None) -> None:
        """An extra control under the last row (e.g. which image to draw); removed with that row's layer."""
        holder = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(holder)
        hl.setContentsMargins(22, 0, 0, 0)
        hl.addWidget(widget)
        self._rows.addWidget(holder)
        target = key if key is not None else (list(self._layers)[-1] if self._layers else None)
        if target is not None:
            self._layers[target].extras.append(holder)
        else:
            self._groups.append(holder)

    def bind(self, key: str, items: Iterable[Any]) -> None:
        """The items of a new drawing for ``key``: the row's state applies to them at once."""
        layer = self._layers[key]
        layer.items = list(items)
        self._apply(layer)

    def clear_layers(self, keep_state: bool = True) -> None:
        """Remove every row and group title; with ``keep_state`` a row added again under the same key starts as it
        was left."""
        for layer in self._layers.values():
            for w in [layer.box, *layer.extras] if layer.owned else layer.extras:
                self._rows.removeWidget(w)
                w.setParent(None)
                w.deleteLater()
            if not layer.owned:      # the caller's box: out of the panel, not deleted
                self._rows.removeWidget(layer.box)
                layer.box.setParent(None)
        for w in self._groups:
            self._rows.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        # spacer items left by add_group
        for i in reversed(range(self._rows.count())):
            it = self._rows.itemAt(i)
            if it is not None and it.spacerItem() is not None:
                self._rows.takeAt(i)
        self._layers.clear()
        self._groups = []
        if not keep_state:
            self._state.clear()

    # ------------------------------------------------------------ state
    def keys(self) -> List[str]:
        return list(self._layers)

    def checkbox(self, key: str) -> QtWidgets.QCheckBox:
        return self._layers[key].box

    def items(self, key: str) -> List[Any]:
        return list(self._layers[key].items)

    def is_visible(self, key: str) -> bool:
        return bool(self._state.get(key, True))

    def set_visible(self, key: str, on: bool) -> None:
        if key in self._layers:
            self._layers[key].box.setChecked(bool(on))   # toggled -> _on_box
        else:
            self._state[key] = bool(on)

    def show_all(self) -> None:
        for layer in list(self._layers.values()):
            if layer.box.isEnabled():
                layer.box.setChecked(True)

    def hide_all(self) -> None:
        for layer in list(self._layers.values()):
            if layer.box.isEnabled():
                layer.box.setChecked(False)

    def state(self) -> Dict[str, bool]:
        return dict(self._state)

    def restore(self, state: Mapping[str, bool]) -> None:
        for key, on in state.items():
            self.set_visible(str(key), bool(on))

    def hidden(self) -> List[str]:
        """The keys of the rows now in the panel whose layer is hidden."""
        return [k for k in self._layers if not self._state.get(k, True)]

    def set_count(self, key: str, n: Optional[int]) -> None:
        layer = self._layers[key]
        layer.count = None if n is None else int(n)
        self._set_text(layer)

    def set_enabled(self, key: str, on: bool, reason: str = "") -> None:
        layer = self._layers[key]
        layer.box.setEnabled(bool(on))
        for w in layer.extras:
            w.setEnabled(bool(on))
        tip = layer.tip
        if not on and reason:
            tip = (tip + "\n\n" if tip else "") + f"Not available: {reason}."
        layer.box.setToolTip(tip)

    def set_tip(self, key: str, tip: str) -> None:
        layer = self._layers[key]
        layer.tip = tip
        layer.box.setToolTip(tip)

    # ------------------------------------------------------------ internals
    @staticmethod
    def _set_text(layer: _Layer) -> None:
        layer.box.setText(layer.label if layer.count is None else f"{layer.label} ({layer.count})")

    def _apply(self, layer: _Layer) -> None:
        on = self._state.get(layer.key, True)
        for item in layer.items:
            try:
                item.setVisible(on)
            except RuntimeError:      # an item of a drawing already deleted
                continue

    def _on_box(self, key: str, on: bool) -> None:
        layer = self._layers.get(key)
        if layer is None:
            return
        self._state[key] = on
        self._apply(layer)
        if layer.on_toggle is not None:
            layer.on_toggle(on)
        self.toggled.emit(key, on)


# ============================================================================
# Legend entries that drive groups of real items (pyqtgraph LegendItem)
# ============================================================================

class _GroupMixin:
    """``setVisible`` of the sample also shows or hides every member and records the state under the key."""

    _group_key: str = ""
    _group_members: Sequence[Any] = ()
    _group_state: Optional[MutableMapping[str, bool]] = None

    def _init_group(self, key: str, members: Sequence[Any], state: MutableMapping[str, bool]) -> None:
        self._group_key = key
        self._group_members = list(members)
        self._group_state = state
        self.setVisible(bool(state.get(key, True)))

    def setVisible(self, visible: bool) -> None:
        super().setVisible(visible)  # type: ignore[misc]
        for m in getattr(self, "_group_members", ()):
            try:
                m.setVisible(visible)
            except RuntimeError:
                continue
        state = getattr(self, "_group_state", None)
        if state is not None:
            state[self._group_key] = bool(visible)

    def members(self) -> List[Any]:
        return list(self._group_members)


class _GroupScatter(_GroupMixin, pg.ScatterPlotItem):
    pass


class _GroupLine(_GroupMixin, pg.PlotDataItem):
    pass


class LegendLayers:
    """Entries of a pyqtgraph ``LegendItem`` that hide and show groups of real items.

    ``state`` is the window's own dictionary (key -> visible); it outlives a redraw, so an entry built again under
    the same key starts hidden when the user had hidden it, and its new members with it.
    """

    def __init__(self, state: MutableMapping[str, bool]) -> None:
        self.state = state
        self.proxies: Dict[str, Any] = {}

    def entry(self, legend: Any, key: str, label: str, sample_kind: str, members: Sequence[Any],
              **style: Any) -> Any:
        """Add ``label`` to ``legend`` with a sample drawn from ``style`` ("scatter": ScatterPlotItem keywords;
        "line": PlotDataItem keywords, e.g. ``pen``), bound to ``members``."""
        proxy: Any
        if sample_kind == "scatter":
            proxy = _GroupScatter(**style)
        elif sample_kind == "line":
            proxy = _GroupLine(**style)
        else:
            raise ValueError(f"unknown sample kind {sample_kind!r}")
        proxy._init_group(key, members, self.state)
        legend.addItem(proxy, label)
        self.proxies[key] = proxy
        return proxy

    def clear(self) -> None:
        self.proxies = {}

    def hidden(self) -> List[str]:
        """The keys of the entries now listed whose items are hidden."""
        return [k for k in self.proxies if not self.state.get(k, True)]
