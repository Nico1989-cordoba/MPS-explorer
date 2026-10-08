# -*- coding: utf-8 -*-
"""
The single axial view: the ROI's z before the cut, the fitted mixture, the analysis slab, the main-window cut and the
segments, on one axis and one scale (UI stage 2, design 4).

``AxialView`` is a plain QWidget - a pyqtgraph plot, a ``LayerPanel`` beside it, a "View" combo ("MPS analysis slab"
/ "Every segment") and a read-only details panel under the plot - drawn from ``tools.mps_axon_map_layers``. In IMPL-C
it becomes the "Axial" tab of the MPS analysis window and replaces the results window's axial plot, the Rings
window's two z plots and the main window's ROI z histogram; until then nothing uses it.

Rules it keeps: the bars are density over the ROI, so the mixture sits on them (4.3); the left axis says so; channel 2
is drawn as loaded, never as registered, and drawing it disables the components (orange against vermillion is the
closest pair for a deuteranope); the cut lines are disabled while the analysis is of another selection (4.5); the
title has three lines and travels with the figure (4.4).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from tools import mps_axon_map_layers as L
from tools.mps_axon_map import render_layer, swatch_for
from tools.mps_layer_panel import LayerPanel
from tools.mps_origin_ui import DetailsPanel
from tools.mps_plot_style import set_title, style_dark, style_light

__all__ = ["AxialView"]

VIEW_TIP = ("MPS analysis slab: the ROI's z before the cut with the slab the analysis kept, the fitted components and "
            "the main-window cut. Every segment: the same z with each segment's slab and the boundaries between them "
            "(set by the Rings window's button).")


class AxialView(QtWidgets.QWidget):
    """The axial view (design 4). ``set_inputs(AxialInputs)``; ``set_view("slab" | "segments")``."""

    view_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, dark: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("axial_view")
        self.dark = bool(dark)
        self._inputs = L.AxialInputs()
        self._layers: Dict[str, L.Layer] = {}
        self._applying = False
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("View"))
        self.combo_view = QtWidgets.QComboBox()
        self.combo_view.setObjectName("combo_axial_view")
        for key in L.AXIAL_VIEWS:
            self.combo_view.addItem(L.AXIAL_VIEW_NAMES[key], key)
        self.combo_view.setToolTip(VIEW_TIP)
        self.combo_view.currentIndexChanged.connect(
            lambda _i: self.set_view(str(self.combo_view.currentData())))
        top.addWidget(self.combo_view)
        top.addStretch(1)
        root.addLayout(top)
        row = QtWidgets.QHBoxLayout()
        self.plot = pg.PlotWidget()
        self.plot.setObjectName("axial_plot")
        self.plot.setLabels(bottom="z [nm]", left=L.AXIAL_LEFT_LABEL)
        self.layers = LayerPanel(scroll=True, max_width=260, hide_disabled=True)
        self.layers.setObjectName("axial_layers")
        self.layers.toggled.connect(self._on_toggled)
        row.addWidget(self.plot, 1)
        row.addWidget(self.layers)
        root.addLayout(row, 1)
        self.details = DetailsPanel(object_name="details_axial", collapsed=True)
        root.addWidget(self.details)
        self._style()
        self.redraw()

    # ------------------------------------------------------------------ public
    def inputs(self) -> L.AxialInputs:
        return self._inputs

    def set_inputs(self, inputs: L.AxialInputs, view: Optional[str] = None) -> None:
        self._inputs = inputs
        if view is not None:
            self.set_view(view)
        else:
            self.redraw()

    def view(self) -> str:
        return self._inputs.view

    def set_view(self, view: str) -> None:
        if view not in L.AXIAL_VIEWS:
            raise ValueError(f"unknown view {view!r}")
        self._inputs.view = view
        self.combo_view.blockSignals(True)
        self.combo_view.setCurrentIndex(self.combo_view.findData(view))
        self.combo_view.blockSignals(False)
        self._applying = True
        try:
            self.redraw()
            for key in self.layers.keys():
                self.layers.set_visible(key, L.axial_on(view, key))
        finally:
            self._applying = False
        self._retitle()
        self.view_changed.emit(view)

    def title(self) -> str:
        return L.axial_title(self._inputs)

    def title_lines(self) -> List[str]:
        return self.title().split("<br>")

    def plot_item(self) -> Any:
        return self.plot.getPlotItem()

    def items(self, key: str) -> List[Any]:
        return self.layers.items(key)

    def layer(self, key: str) -> L.Layer:
        return self._layers[key]

    def set_dark(self, dark: bool) -> None:
        self.dark = bool(dark)
        self._style()
        self.redraw()

    def export_message(self) -> str:
        shown = self.layers.shown()
        hidden = [k for k in self.layers.keys() if k not in self.layers.shown_keys()]
        return "\n".join(["Visible layers: " + ("; ".join(shown) if shown else "none") + ".",
                          f"{len(hidden)} layer{'s' if len(hidden) != 1 else ''} hidden or not available.",
                          "Title: " + " | ".join(self.title_lines())])

    # ------------------------------------------------------------------ drawing
    def redraw(self) -> None:
        self.plot.clear()
        self.layers.clear_layers(keep_state=True)
        self._layers = {}
        was = self._applying
        self._applying = True
        try:
            for g in L.build_axial(self._inputs):
                self.layers.add_group(g.title, key=g.key, caption=g.caption, collapsible=True)
                for layer in g.layers:
                    self._layers[layer.key] = layer
                    items = render_layer(layer, self.dark)
                    for item in items:
                        self.plot.addItem(item)
                    self.layers.add_layer(layer.key, layer.label, swatch_for(layer, True), items=items,
                                          count=layer.count, tip=layer.tip,
                                          visible=L.axial_on(self._inputs.view, layer.key))
                    if not layer.enabled:
                        self.layers.set_enabled(layer.key, False, layer.reason)
                if not g.enabled:
                    self.layers.set_group_enabled(g.key, False, g.reason)
        finally:
            self._applying = was
        self.details.set_rows(L.axial_details(self._inputs))
        self._retitle()

    def _retitle(self) -> None:
        set_title(self.plot, self.title(), dark=self.dark)

    def _style(self) -> None:
        (style_dark if self.dark else style_light)(self.plot)
        self.plot.setLabels(left=L.AXIAL_LEFT_LABEL)

    def _on_toggled(self, key: str, on: bool) -> None:
        if key == "ch2_roi" and not self._applying and bool(on) != bool(self._inputs.ch2_shown):
            # channel 2 and the components are never drawn together (Appendix A: 18.3 under deuteranopia)
            self._inputs.ch2_shown = bool(on)
            self.redraw()
