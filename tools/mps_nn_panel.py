# -*- coding: utf-8 -*-
"""
Nearest neighbours in one place (UI stage 2, design 5): the "Nearest neighbours" tab of the MPS analysis window.

``NearestNeighboursPanel`` is a plain QWidget: the histogram of the distances between cluster centres of the analysis
the radio shows - "Neighbours: 1st to [N]", "Bins", "Range: automatic / from..to nm" - with its median and the
published reference as a layer that starts off; the observed-vs-randomized CDF with its legend in the bottom-right
corner and the crossing drawn as this program's heuristic, never in the paper's colour; a read-only details panel
under each; and "Save distances...", which writes the CSV the main window's "save dist data" wrote (the same bytes
for the measured analysis) for the analysis shown.

At N = 1 with the range automatic the histogram is the old "1NN distance between cluster centres" exactly; N > 1
pools the 1st..Nth distances as the main window's distances histogram did; a range is display only (the CSV keeps
every distance) and the title counts what falls outside it. N is capped at clusters - 1 (today N + 1 > clusters
raises inside sklearn).

It replaces the results window's old 1NN plots and the main window's distances controls (M5: "N neighbor",
"Distances", the lateral range, the binning and "save dist data"). While the analysis shown is not of the main
window's current selection, "Save distances..." refuses with the two messages the main window's "Distances" and
"save dist data" gave in that state, so nothing of another selection is written.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from tools import mps_axon_map_layers as L
from tools.mps_axon_map import render_layer, swatch_for
from tools.mps_layer_panel import LayerPanel
from tools.mps_origin_ui import DetailsPanel
from tools.mps_plot_style import AXIS_FG, AXIS_FG_LIGHT, set_title, style_dark, style_light

__all__ = ["NearestNeighboursPanel", "SAVE_TIP", "SAVE_TITLE"]

SAVE_TITLE = "Save Distance Data"
SAVE_TIP = ("KD-tree query, k = N, the centre itself excluded; the same numbers as the histogram. Writes every "
            "distance of the analysis shown (the range only limits what the histogram draws).")
N_TIP = "Pool the distances from each centre to its 1st..Nth nearest centre (at most clusters - 1, and 10)."
BINS_TIP = "Number of bins of the histogram (display only)."
# What the main window's "Distances" and "save dist data" said when the analysis was not of the current selection
# (its centres were cleared by a new ROI or axial cut): "Save distances..." says the same in that state.
NOT_CURRENT_MESSAGES = (
    ("No clusters", "Please run clustering and confirm cluster selection before computing distances."),
    ("No distance data", "Please compute the nearest-neighbour distances first."),
)
RANGE_TIP = ("Automatic: the bins span the distances. From..to: the bins span this range, and the title counts the "
             "distances outside it (display only: the CSV keeps every distance).")


class NearestNeighboursPanel(QtWidgets.QWidget):
    """The nearest-neighbours tab (design 5)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, dark: bool = True, bins: int = 30,
                 range_nm: Tuple[float, float] = (0.0, 800.0),
                 root_name: Optional[Callable[[], str]] = None,
                 current: Optional[Callable[[], bool]] = None) -> None:
        super().__init__(parent)
        self.setObjectName("nn_panel")
        self.dark = bool(dark)
        self.shown: Any = None
        self.measured: Any = None
        self.comparison: Any = None
        self.root_name = root_name or (lambda: "distances")
        # Whether the analysis shown describes the main window's current selection (a panel on its own: always).
        self.current: Callable[[], bool] = current or (lambda: True)
        self._layers: Dict[str, L.Layer] = {}
        self._hist: Dict[str, Any] = {}
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Neighbours: 1st to"))
        self.spin_neighbours = QtWidgets.QSpinBox()
        self.spin_neighbours.setObjectName("spin_neighbours")
        self.spin_neighbours.setRange(1, 1)
        self.spin_neighbours.setToolTip(N_TIP)
        controls.addWidget(self.spin_neighbours)
        controls.addWidget(QtWidgets.QLabel("Bins"))
        self.spin_bins = QtWidgets.QSpinBox()
        self.spin_bins.setObjectName("spin_bins")
        self.spin_bins.setRange(1, 1000)
        self.spin_bins.setValue(int(bins))
        self.spin_bins.setToolTip(BINS_TIP)
        controls.addWidget(self.spin_bins)
        controls.addWidget(QtWidgets.QLabel("Range:"))
        self.radio_range_auto = QtWidgets.QRadioButton("automatic")
        self.radio_range_auto.setObjectName("radio_range_auto")
        self.radio_range_set = QtWidgets.QRadioButton("from")
        self.radio_range_set.setObjectName("radio_range_set")
        self.radio_range_auto.setChecked(True)
        for w in (self.radio_range_auto, self.radio_range_set):
            w.setToolTip(RANGE_TIP)
            controls.addWidget(w)
        self.spin_range_min = QtWidgets.QDoubleSpinBox()
        self.spin_range_min.setObjectName("spin_range_min")
        self.spin_range_max = QtWidgets.QDoubleSpinBox()
        self.spin_range_max.setObjectName("spin_range_max")
        for spin, value in ((self.spin_range_min, range_nm[0]), (self.spin_range_max, range_nm[1])):
            spin.setRange(0.0, 1e6)
            spin.setDecimals(0)
            spin.setSuffix(" nm")
            spin.setValue(float(value))
            spin.setToolTip(RANGE_TIP)
        controls.addWidget(self.spin_range_min)
        controls.addWidget(QtWidgets.QLabel("to"))
        controls.addWidget(self.spin_range_max)
        controls.addStretch(1)
        self.btn_save = QtWidgets.QPushButton("Save distances...")
        self.btn_save.setObjectName("btn_save_distances")
        self.btn_save.setToolTip(SAVE_TIP)
        self.btn_save.clicked.connect(lambda: self.on_save())
        controls.addWidget(self.btn_save)
        root.addLayout(controls)
        self.lbl_note = QtWidgets.QLabel("")
        self.lbl_note.setObjectName("nn_note")
        self.lbl_note.setWordWrap(True)
        root.addWidget(self.lbl_note)
        self.lbl_note.setVisible(False)
        plots = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        hist_row = QtWidgets.QHBoxLayout()
        self.plot_nn = pg.PlotWidget()
        self.plot_nn.setObjectName("nn_plot")
        self.plot_nn.setLabels(bottom="distance [nm]", left="count")
        self.layers = LayerPanel(scroll=True, max_width=220, hide_disabled=True)
        self.layers.setObjectName("nn_layers")
        self.layers.toggled.connect(lambda _k, _on: None)
        hist_row.addWidget(self.plot_nn, 1)
        hist_row.addWidget(self.layers)
        left.addLayout(hist_row, 1)
        self.details_nn = DetailsPanel(object_name="details_nn", collapsed=True)
        left.addWidget(self.details_nn)
        right = QtWidgets.QVBoxLayout()
        self.plot_cdf = pg.PlotWidget()
        self.plot_cdf.setObjectName("cdf_plot")
        self.plot_cdf.setLabels(bottom="distance [nm]", left="cumulative")
        # inside the plot, so a figure carries it; bottom-right, the empty corner of a CDF
        self.legend = self.plot_cdf.addLegend(offset=(-10, -10), labelTextColor=AXIS_FG)
        right.addWidget(self.plot_cdf, 1)
        self.details_cdf = DetailsPanel(object_name="details_cdf", collapsed=True)
        right.addWidget(self.details_cdf)
        plots.addLayout(left, 1)
        plots.addLayout(right, 1)
        root.addLayout(plots, 1)
        for signal in (self.spin_neighbours.valueChanged, self.spin_bins.valueChanged,
                       self.spin_range_min.valueChanged, self.spin_range_max.valueChanged,
                       self.radio_range_auto.toggled):
            signal.connect(lambda *_a: self.redraw())
        self._style()
        self.redraw()

    # ------------------------------------------------------------------ public
    def set_analysis(self, shown: Any, measured: Any = None, comparison: Any = None) -> None:
        """The analysis the radio shows (and the measured one, and the comparison, for the titles and the CSV)."""
        self.shown = shown
        self.measured = measured if measured is not None else shown
        self.comparison = comparison
        k = 0 if shown is None else int(shown.n_clusters_kept)
        top = L.nn_max_neighbours(k)
        self.spin_neighbours.blockSignals(True)
        self.spin_neighbours.setRange(1, max(1, top))
        self.spin_neighbours.blockSignals(False)
        self.spin_neighbours.setEnabled(top >= 1)
        self.btn_save.setEnabled(top >= 1)
        self.lbl_note.setText("" if top >= 1 else "Fewer than two clusters: no distance between centres.")
        # Shown only when it says something: an empty line costs the map its room on a short screen.
        self.lbl_note.setVisible(top < 1)
        self.redraw()

    def n(self) -> int:
        return int(self.spin_neighbours.value())

    def value_range(self) -> Optional[Tuple[float, float]]:
        if self.radio_range_auto.isChecked():
            return None
        return (float(self.spin_range_min.value()), float(self.spin_range_max.value()))

    def nn_title(self) -> str:
        return L.nn_title(self._hist or {"n": self.n()}, self.shown, self.comparison, self.measured,
                          self.value_range())

    def cdf_title(self) -> str:
        return L.cdf_title(self.shown, self.comparison, self.measured)

    def plots(self) -> Dict[str, Any]:
        """The two plots by the name the export lists."""
        return {"Nearest neighbours": self.plot_nn, "1NN CDF": self.plot_cdf}

    def default_name(self) -> str:
        return L.default_distances_name(self.root_name(), self.n(), self.shown, self.measured)

    def save_to(self, path: str) -> int:
        """Write the distances CSV of the analysis shown at the current N (see ``write_distances_csv``)."""
        return L.write_distances_csv(path, self.shown, self.measured, self.n())

    def on_save(self) -> Optional[str]:
        """"Save distances...": ask where, then write (the dialog title is the main window's)."""
        if not self.current():
            for title, text in NOT_CURRENT_MESSAGES:
                QtWidgets.QMessageBox.warning(self, title, text)
            return None
        if self.shown is None or L.nn_max_neighbours(int(self.shown.n_clusters_kept)) < 1:
            return None
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, SAVE_TITLE, self.default_name(), "CSV Files (*.csv)")
        if not path:
            return None
        try:
            self.save_to(path)
        except Exception as error:                     # noqa: BLE001 - reported to the user
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save the distances:\n\n{error}")
            return None
        return str(path)

    def set_dark(self, dark: bool) -> None:
        self.dark = bool(dark)
        self._style()
        self.redraw()

    def layer(self, key: str) -> L.Layer:
        return self._layers[key]

    # ------------------------------------------------------------------ drawing
    def redraw(self) -> None:
        on_range = not self.radio_range_auto.isChecked()
        self.spin_range_min.setEnabled(on_range)
        self.spin_range_max.setEnabled(on_range)
        self.plot_nn.clear()
        self.layers.clear_layers(keep_state=True)
        self._layers = {}
        self._hist = L.nn_histogram(self.shown, self.n(), int(self.spin_bins.value()), self.value_range())
        for layer in L.nn_layers(self.shown, self._hist):
            self._layers[layer.key] = layer
            items = render_layer(layer, self.dark)
            for item in items:
                self.plot_nn.addItem(item)
            self.layers.add_layer(layer.key, layer.label, swatch_for(layer, True), items=items, count=layer.count,
                                  tip=layer.tip, visible=layer.key in L.NN_ON)
            if not layer.enabled:
                self.layers.set_enabled(layer.key, False, layer.reason)
        set_title(self.plot_nn, self.nn_title(), dark=self.dark)
        self.plot_cdf.clear()
        if self.legend is not None:
            self.legend.clear()
        for layer in L.cdf_layers(self.shown):
            self._layers[layer.key] = layer
            for item in render_layer(layer, self.dark):
                self.plot_cdf.addItem(item)
        set_title(self.plot_cdf, self.cdf_title(), dark=self.dark)
        self.details_nn.set_rows(L.nn_details(self.shown, self.n()))
        self.details_cdf.set_rows(L.cdf_details(self.shown))

    def _style(self) -> None:
        for plot in (self.plot_nn, self.plot_cdf):
            (style_dark if self.dark else style_light)(plot)
        if self.legend is not None:
            self.legend.setLabelTextColor(AXIS_FG if self.dark else AXIS_FG_LIGHT)

    def items_of(self, key: str) -> List[Any]:
        return self.layers.items(key) if key in self.layers.keys() else []
