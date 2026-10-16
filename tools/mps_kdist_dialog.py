# -*- coding: utf-8 -*-
"""
The k-distance graph dialog: what "eps = auto" opens (decision P-R58; docs/eps_kdistance.md).

Two views of the same numbers, side by side: the k-distance graph (each localization's distance to its k-th other
neighbour, sorted; k = min samples - 1) and the density of those distances on a log axis, where the valley between
the clustered localizations and the background is found (``tools.mps_kdistance.find_valley``). On both:

* the valley, when there is one, as a SUGGESTION: a dotted vermillion line with its bootstrap interval shaded, the
  eps it implies and the share of localizations DBSCAN leaves unclustered there (DBSCAN run, not estimated). When
  there is none, the dialog says so and suggests nothing;
* a dashed line at 25 nm, labelled "Gazal et al. 2026", with its unclustered share;
* the value a click sets (solid orange): a click anywhere on either plot sets eps there and runs DBSCAN for its
  unclustered share; moving the pointer shows, live, the share of localizations that are not core points at the
  pointer's eps (k-distance above it), which the curve reads directly.

Nothing is applied by itself: the person accepts the suggestion, the published value or the clicked value, and the
choice is returned as a record (``tools.mps_kdistance.choice_record``) that the caller keeps and exports.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from tools import mps_kdistance as kdm
from tools.mps_plot_style import OKABE_ITO, TEXT_DIM, set_title, style_dark

DOC = "docs/eps_kdistance.md"
EPS_LIMITS_NM = (0.1, 1000.0)
EPS_DECIMALS = 1

_SUGGESTION = OKABE_ITO["vermillion"]
_REFERENCE = OKABE_ITO["sky_blue"]
_CHOSEN = OKABE_ITO["orange"]
_CURVE = "#e8e8e8"


def _pct(value: Optional[float]) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{100.0 * value:.1f} %"


class KDistanceDialog(QtWidgets.QDialog):
    """Choose DBSCAN's eps on the k-distance graph of one channel's selection."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], x_nm: Any, y_nm: Any, min_samples: Optional[int], *,
                 channel: int = 1, file_name: str = "", current_eps_nm: Optional[float] = None,
                 min_samples_editable: bool = False, exploratory_note: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle(f"eps from the k-distance graph - channel {channel}")
        self.setObjectName("kdistance_dialog")
        self.x_nm = np.asarray(x_nm, dtype=np.float64).ravel()
        self.y_nm = np.asarray(y_nm, dtype=np.float64).ravel()
        self.channel = int(channel)
        self.file_name = str(file_name)
        self.current_eps = current_eps_nm
        self.min_samples: Optional[int] = None
        self.kdist: np.ndarray = np.zeros(0)
        self.valley: Optional[kdm.Valley] = None
        self.chosen_eps: Optional[float] = None
        self._unclustered: Dict[float, float] = {}
        self._record: Optional[Dict[str, Any]] = None
        self._items: list = []

        outer = QtWidgets.QVBoxLayout(self)
        self.header = QtWidgets.QLabel()
        self.header.setWordWrap(True)
        self.header.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.header)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Min Pts:"))
        self.edit_min = QtWidgets.QLineEdit()
        self.edit_min.setObjectName("kdist_min_samples")
        self.edit_min.setMaximumWidth(70)
        self.edit_min.setReadOnly(not min_samples_editable)
        self.edit_min.setToolTip(
            "DBSCAN's Min Pts: the graph is drawn for k = Min Pts - 1 (the distance to the (Min Pts - 1)-th other "
            "localization). " + ("Type it to draw the graph; it is never taken from channel 1." if min_samples_editable
                                 else "The analysis' own value: change it in the parameter strip."))
        self.button_draw = QtWidgets.QPushButton("Draw")
        self.button_draw.setVisible(bool(min_samples_editable))
        self.button_draw.clicked.connect(self._on_draw)
        self.edit_min.returnPressed.connect(self._on_draw)
        row.addWidget(self.edit_min)
        row.addWidget(self.button_draw)
        self.k_label = QtWidgets.QLabel()
        row.addWidget(self.k_label)
        row.addStretch(1)
        outer.addLayout(row)

        plots = QtWidgets.QHBoxLayout()
        self.plot_curve = pg.PlotWidget()
        self.plot_density = pg.PlotWidget()
        for p in (self.plot_curve, self.plot_density):
            style_dark(p)
            p.setMinimumSize(360, 300)
            p.showGrid(x=True, y=True, alpha=0.15)
            plots.addWidget(p)
        self.plot_curve.setLogMode(x=False, y=True)
        self.plot_curve.setLabel("bottom", "localizations, sorted by k-distance (share)")
        self.plot_curve.setLabel("left", "k-distance (nm)")
        self.plot_density.setLogMode(x=True, y=False)
        self.plot_density.setLabel("bottom", "k-distance (nm)")
        self.plot_density.setLabel("left", "density (per log10 nm)")
        set_title(self.plot_curve, "k-distance graph")
        set_title(self.plot_density, "Density of the k-distances")
        outer.addLayout(plots, 1)

        self.pointer = QtWidgets.QLabel(" ")
        self.pointer.setStyleSheet(f"color: {TEXT_DIM};")
        outer.addWidget(self.pointer)
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("kdist_summary")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.summary)
        note = QtWidgets.QLabel(
            (exploratory_note + " " if exploratory_note else "")
            + "The valley is a suggestion, never applied by itself; when there is none, nothing is suggested. In "
              "simulated rings it lay above the eps that best recovered the true clusters in 9 runs of 10 (median "
              "1.8 times; own simulation, docs/sims/kdist). Click on either plot to "
              "set eps there. How eps was chosen is kept and written in the export (eps_source). Method and its "
              f"validation: {DOC}")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {TEXT_DIM};")
        outer.addWidget(note)

        buttons = QtWidgets.QHBoxLayout()
        self.button_valley = QtWidgets.QPushButton("Use the suggestion")
        self.button_valley.setObjectName("kdist_use_valley")
        self.button_gazal = QtWidgets.QPushButton(f"Use {kdm.GAZAL_EPS_NM:g} nm (Gazal et al. 2026)")
        self.button_gazal.setObjectName("kdist_use_gazal")
        self.button_chosen = QtWidgets.QPushButton("Use the clicked value")
        self.button_chosen.setObjectName("kdist_use_click")
        cancel = QtWidgets.QPushButton("Cancel")
        self.button_valley.clicked.connect(lambda: self._accept("kdist_valley"))
        self.button_gazal.clicked.connect(lambda: self._accept("gazal_default"))
        self.button_chosen.clicked.connect(lambda: self._accept("kdist_click"))
        cancel.clicked.connect(self.reject)
        for b in (self.button_valley, self.button_gazal, self.button_chosen):
            b.setAutoDefault(False)
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        outer.addLayout(buttons)

        for p in (self.plot_curve, self.plot_density):
            p.scene().sigMouseClicked.connect(lambda ev, plot=p: self._on_scene_click(plot, ev))
            p.scene().sigMouseMoved.connect(lambda pos, plot=p: self._on_scene_move(plot, pos))

        self.resize(900, 620)
        if min_samples is not None:
            self.edit_min.setText(str(int(min_samples)))
            self.draw(int(min_samples))
        else:
            self._set_header()
            self.summary.setText("Type channel 2's Min Pts and press Draw: the graph depends on it (k = Min Pts - 1).")
            self._update_buttons()

    # ------------------------------------------------------------------ computing
    def draw(self, min_samples: int) -> None:
        """Compute the k-distances, the valley and the reference shares for this Min Pts, and draw them."""
        self.min_samples = int(min_samples)
        self.chosen_eps = None
        self._unclustered = {}
        try:
            self.kdist = kdm.k_distances(self.x_nm, self.y_nm, self.min_samples)
        except ValueError as error:
            self.kdist = np.zeros(0)
            self.valley = None
            self._set_header()
            self.summary.setText(str(error))
            self._clear_plots()
            self._update_buttons()
            return
        self.valley = kdm.find_valley(self.kdist)
        self._set_header()
        self._redraw()
        self._update_summary()
        self._update_buttons()

    def unclustered(self, eps_nm: float) -> float:
        """DBSCAN's unclustered share at this eps (cached per eps)."""
        key = round(float(eps_nm), 6)
        if key not in self._unclustered and self.min_samples is not None:
            self._unclustered[key] = kdm.unclustered_fraction(self.x_nm, self.y_nm, key, self.min_samples)
        return self._unclustered.get(key, float("nan"))

    def set_chosen(self, eps_nm: float) -> float:
        """Set eps as a click does: rounded, kept within limits, drawn, its unclustered share computed."""
        eps = round(min(max(float(eps_nm), EPS_LIMITS_NM[0]), EPS_LIMITS_NM[1]), EPS_DECIMALS)
        self.chosen_eps = eps
        self._redraw()
        self._update_summary()
        self._update_buttons()
        return eps

    # ------------------------------------------------------------------ events
    def _view_point(self, plot: Any, scene_pos: Any) -> Optional[QtCore.QPointF]:
        vb = plot.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        point: QtCore.QPointF = vb.mapSceneToView(scene_pos)
        return point

    def eps_at(self, plot: Any, view_x: float, view_y: float) -> float:
        """The eps a point of a plot stands for (both plots draw eps on a log10 axis)."""
        return float(10.0 ** (view_y if plot is self.plot_curve else view_x))

    def _on_scene_click(self, plot: Any, event: Any) -> None:
        if self.kdist.size == 0:
            return
        pt = self._view_point(plot, event.scenePos())
        if pt is None:
            return
        self.set_chosen(self.eps_at(plot, pt.x(), pt.y()))

    def _on_scene_move(self, plot: Any, scene_pos: Any) -> None:
        if self.kdist.size == 0:
            return
        pt = self._view_point(plot, scene_pos)
        if pt is None:
            return
        eps = self.eps_at(plot, pt.x(), pt.y())
        if not (EPS_LIMITS_NM[0] <= eps <= EPS_LIMITS_NM[1]):
            return
        self.pointer.setText(f"Pointer: eps {eps:.1f} nm - {_pct(kdm.noncore_fraction(self.kdist, eps))} of the "
                             f"localizations are not core points there (k-distance above it); click to set eps.")

    def _on_draw(self) -> None:
        try:
            value = float(self.edit_min.text().strip().replace(",", "."))
        except ValueError:
            value = float("nan")
        if not (math.isfinite(value) and value.is_integer() and 2 <= value <= 10000):
            self.summary.setText("Min Pts is a whole number from 2 to 10000 (k = Min Pts - 1 must be at least 1).")
            return
        self.draw(int(value))

    def _accept(self, source: str) -> None:
        if self.min_samples is None:
            return
        if source == "kdist_valley":
            if self.valley is None or not self.valley.found:
                return
            eps = round(float(self.valley.eps_nm), EPS_DECIMALS)
        elif source == "gazal_default":
            eps = float(kdm.GAZAL_EPS_NM)
        else:
            if self.chosen_eps is None:
                return
            eps = float(self.chosen_eps)
        self._record = kdm.choice_record(source, eps, self.min_samples, valley=self.valley,
                                         unclustered=self.unclustered(eps), file_name=self.file_name,
                                         n_locs=int(self.x_nm.size))
        self.accept()

    def record(self) -> Optional[Dict[str, Any]]:
        """The choice made (``choice_record``), or None when the dialog was cancelled."""
        return self._record

    # ------------------------------------------------------------------ drawing
    def _set_header(self) -> None:
        name = f" of {self.file_name}" if self.file_name else ""
        k = f"k = {self.min_samples - 1}" if self.min_samples else "k = Min Pts - 1"
        self.header.setText(
            f"Channel {self.channel}: {self.x_nm.size:,} localizations of the selection{name}. Each one's distance to "
            f"its k-th nearest other localization, {k} (scikit-learn's Min Pts counts the point itself; Ester et al. "
            f"1996, Schubert et al. 2017). A localization is a core point at eps when its k-distance is at most eps.")
        self.k_label.setText(f"-> {k}")

    def _clear_plots(self) -> None:
        for p in (self.plot_curve, self.plot_density):
            p.clear()
        self._items = []

    def _vline(self, plot: Any, eps: float, colour: str, style: Any, label: str, horizontal: bool,
               where: float = 0.9) -> None:
        """A line at eps (log10 axis) with its label at ``where`` along it (0-1), so close lines keep apart."""
        opts: Dict[str, Any] = {"color": colour, "position": where}
        if horizontal:
            opts["anchors"] = [(1, 1), (1, 1)]
        line = pg.InfiniteLine(pos=math.log10(eps), angle=0 if horizontal else 90,
                               pen=pg.mkPen(colour, width=2, style=style), label=label or None,
                               labelOpts=opts if label else None)
        plot.addItem(line)

    def _redraw(self) -> None:
        self._clear_plots()
        if self.kdist.size == 0:
            return
        curve = kdm.sorted_curve(self.kdist)
        positive = curve > 0
        share = (np.arange(curve.size) + 0.5) / curve.size
        self.plot_curve.plot(share[positive], curve[positive], pen=pg.mkPen(_CURVE, width=2))
        v = self.valley
        if v is not None and v.grid_nm.size:
            self.plot_density.plot(v.grid_nm, v.density, pen=pg.mkPen(_CURVE, width=2))
        dot, dash, solid = QtCore.Qt.PenStyle.DotLine, QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.SolidLine
        if v is not None and v.found:
            lo, hi = math.log10(v.low_nm), math.log10(v.high_nm)
            band = pg.LinearRegionItem(values=(lo, hi), orientation="vertical", movable=False,
                                       brush=pg.mkBrush(213, 94, 0, 40), pen=pg.mkPen(None))
            self.plot_density.addItem(band)
            band2 = pg.LinearRegionItem(values=(lo, hi), orientation="horizontal", movable=False,
                                        brush=pg.mkBrush(213, 94, 0, 40), pen=pg.mkPen(None))
            self.plot_curve.addItem(band2)
            self._vline(self.plot_curve, v.eps_nm, _SUGGESTION, dot, f"valley {v.eps_nm:.1f} nm", True, 0.95)
            self._vline(self.plot_density, v.eps_nm, _SUGGESTION, dot, f"valley {v.eps_nm:.1f} nm", False, 0.95)
        self._vline(self.plot_curve, kdm.GAZAL_EPS_NM, _REFERENCE, dash, "Gazal et al. 2026", True, 0.95)
        self._vline(self.plot_density, kdm.GAZAL_EPS_NM, _REFERENCE, dash, "Gazal et al. 2026", False, 0.75)
        if self.chosen_eps is not None:
            self._vline(self.plot_curve, self.chosen_eps, _CHOSEN, solid, f"chosen {self.chosen_eps:.1f} nm", True,
                        0.55)
            self._vline(self.plot_density, self.chosen_eps, _CHOSEN, solid, "chosen", False, 0.55)

    def _update_summary(self) -> None:
        if self.min_samples is None or self.kdist.size == 0:
            return
        v = self.valley
        lines = []
        if v is not None and v.found:
            lines.append(f"<b>Suggestion</b>: {kdm.valley_words(v)} At {v.eps_nm:.1f} nm DBSCAN leaves "
                         f"{_pct(self.unclustered(round(v.eps_nm, EPS_DECIMALS)))} of the localizations "
                         f"unclustered.")
        else:
            reason = v.reason if v is not None else "no density"
            lines.append(f"<b>No valley, so no suggestion</b>: {reason}. Choose eps by clicking on the graph, or "
                         f"keep the published value.")
        lines.append(f"<b>Gazal et al. 2026</b>: {kdm.GAZAL_EPS_NM:g} nm leaves "
                     f"{_pct(self.unclustered(kdm.GAZAL_EPS_NM))} unclustered.")
        if self.current_eps is not None and not math.isclose(float(self.current_eps), kdm.GAZAL_EPS_NM):
            lines.append(f"In use now: {float(self.current_eps):g} nm leaves "
                         f"{_pct(self.unclustered(float(self.current_eps)))} unclustered.")
        if self.chosen_eps is not None:
            lines.append(f"<b>Clicked</b>: {self.chosen_eps:.1f} nm leaves {_pct(self.unclustered(self.chosen_eps))}"
                         f" unclustered ({_pct(kdm.noncore_fraction(self.kdist, self.chosen_eps))} are not core "
                         f"points).")
        lines.append(f"Unclustered = labelled noise by DBSCAN(eps, Min Pts {self.min_samples}) on these "
                     f"{self.x_nm.size:,} localizations, before the analysis removes any cluster.")
        self.summary.setText("<br>".join(lines))

    def _update_buttons(self) -> None:
        ready = self.min_samples is not None and self.kdist.size > 0
        v = self.valley
        has_valley = bool(ready and v is not None and v.found)
        self.button_valley.setEnabled(has_valley)
        self.button_valley.setText(f"Use the suggestion ({v.eps_nm:.1f} nm)" if has_valley and v is not None
                                   else "No suggestion (no valley)")
        self.button_gazal.setEnabled(bool(ready))
        self.button_chosen.setEnabled(bool(ready and self.chosen_eps is not None))
        self.button_chosen.setText(f"Use the clicked value ({self.chosen_eps:.1f} nm)"
                                   if self.chosen_eps is not None else "Use the clicked value (click a plot)")


def choose_eps(parent: Optional[QtWidgets.QWidget], x_nm: Any, y_nm: Any, min_samples: Optional[int],
               **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Open the dialog and return the record of the choice, or None when cancelled."""
    dialog = KDistanceDialog(parent, x_nm, y_nm, min_samples, **kwargs)
    if dialog.exec_() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return dialog.record()
