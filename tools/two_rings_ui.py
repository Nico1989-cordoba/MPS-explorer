# -*- coding: utf-8 -*-
"""
The H-ECL window's "Two rings (unrolled)" view (user request D1, 2026-10-23): two consecutive rings of the run shown
laid flat along the membrane, a tolerance +/- tau set by hand, a window of 2 tau that can be dragged along the arc,
the clusters painted when they coincide within tau, the coincidences observed against those expected by chance, and
the same pair in the plane (the 2D sensitivity) beside it.

Rule D4: tau = tau_0 = 31 nm carries no mark; any other value shows EXPLORATORY here and in the export.
The numbers come from :mod:`tools.two_rings_view` (no Qt there); nothing here is a test.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from tools import two_rings_view as trv
from tools.mps_plot_style import neutral, rgba, role, set_title, style_dark, verdict

TITLE = "Two rings (unrolled)"
BUTTON_TEXT = "Two rings (unrolled)..."
BUTTON_TIP = ("Two consecutive rings laid flat, with a tau you set by hand (this view only, not the analysis). "
              "Details: docs/eclipse_tolerance.md")
BUTTON_TIP_MORE = ("The clusters that coincide within tau are painted, and their count is compared with what chance "
                   "gives (rotations of the second ring). tau = 31 nm (tau_0) is the pre-specified value; any other is "
                   "EXPLORATORY. A view to understand the tolerance, not a test.")
VIEW_TAU_LABEL = "tau of this view only [nm]:"
VIEW_TAU_TIP = ("Changes only this view and its CSV; the lumen review's 'Tolerance tau' (which re-runs the whole "
                "analysis) is another control.")
NOTE_TEXT = "Line = one-to-one match within tau; drag the shaded window (2 tau) along the arc. Not a test. Hover for more."
NOTE_TIP = ("Top row: the first ring's clusters; bottom row: the second's, at their arcs on the centroid membrane "
            "(the coordinate of the PRIMARY arc test). A line joins each pair that coincides within tau (one-to-one, "
            "as the test matches). The shaded window is 2 tau wide: drag it along the arc to see which clusters fall "
            "inside. 'Chance' = the mean count over 199 rotations of the second ring [ad hoc]. Right: the same rings "
            "in the plane (the 2D sensitivity). Not a test: no joint null, no viability, no p.")
CSV_TIP = "Exports this view only: one row per cluster of the two rings, with tau and its role. Never over a file."
CSV_TIP_MORE = ("One row per cluster of the two rings: arc, centroid, partner along the arc and in the plane, tau "
                "and its role (EXPLORATORY unless tau_0). Never over an existing file.")
SIZE_KEY = "two_rings_dialog.size"

__all__ = ["TwoRingsDialog", "TITLE", "BUTTON_TEXT", "BUTTON_TIP", "BUTTON_TIP_MORE"]


class TwoRingsDialog(QtWidgets.QDialog):
    """See the module docstring. ``analyses``: the window's finished run (``CleanedAnalyses``)."""

    def __init__(self, analyses: Any, parent: Optional[QtWidgets.QWidget] = None, source: str = "") -> None:
        super().__init__(parent)
        self.setObjectName("two_rings_dialog")
        self.setWindowTitle(TITLE)
        from tools.mps_ui_layout import open_sized
        open_sized(self, SIZE_KEY)
        self.analyses = analyses
        self.source = str(source)
        self.data: Optional[trv.PairData] = None
        self.reading: Optional[trv.PairReading] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(QtWidgets.QLabel("Rings:"))
        self.combo_pair = QtWidgets.QComboBox()
        self.combo_pair.setObjectName("combo_two_rings_pair")
        for a, b in trv.pairs_of(analyses):
            self.combo_pair.addItem(f"{a}-{b}", (a, b))
        self.combo_pair.currentIndexChanged.connect(lambda _i: self._load_pair())
        top.addWidget(self.combo_pair)
        top.addWidget(QtWidgets.QLabel(VIEW_TAU_LABEL))
        self.spin_tau = QtWidgets.QDoubleSpinBox()
        self.spin_tau.setObjectName("spin_two_rings_tau")
        self.spin_tau.setRange(1.0, 300.0)
        self.spin_tau.setDecimals(1)
        self.spin_tau.setSingleStep(1.0)
        self.spin_tau.setValue(trv.TAU0_NM)
        self.spin_tau.setToolTip(VIEW_TAU_TIP + "\n\nHow close (nm, along the membrane or in the plane) two clusters "
                                 "of the two rings must be to coincide. tau_0 = 31 nm [derivation: 2.77 x 11.3 nm, "
                                 "docs/eclipse_tolerance.md sec. 2]; any other value is EXPLORATORY (rule D4).")
        self.spin_tau.valueChanged.connect(lambda _v: self.update_reading())
        top.addWidget(self.spin_tau)
        self.btn_tau0 = QtWidgets.QPushButton("Back to tau_0")
        self.btn_tau0.setObjectName("btn_two_rings_tau0")
        self.btn_tau0.clicked.connect(lambda: self.spin_tau.setValue(trv.TAU0_NM))
        top.addWidget(self.btn_tau0)
        self.lbl_role = QtWidgets.QLabel("")
        self.lbl_role.setObjectName("lbl_two_rings_role")
        top.addWidget(self.lbl_role)
        top.addStretch(1)
        lay.addLayout(top)
        self.lbl_counts = QtWidgets.QLabel("")
        self.lbl_counts.setObjectName("lbl_two_rings_counts")
        self.lbl_counts.setWordWrap(True)
        self.lbl_counts.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_counts)
        self.plot_arc = pg.PlotWidget()
        self.plot_arc.setObjectName("plot_two_rings_arc")
        style_dark(self.plot_arc)
        self.plot_arc.setLabels(bottom="position along the membrane [um]", left="ring")
        self.plot_arc.setMinimumHeight(180)
        self.plot_plane = pg.PlotWidget()
        self.plot_plane.setObjectName("plot_two_rings_plane")
        style_dark(self.plot_plane)
        self.plot_plane.setLabels(bottom="x' [nm]", left="y' [nm]")
        self.plot_plane.setAspectLocked(True)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self.plot_arc)
        split.addWidget(self.plot_plane)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay.addWidget(split, 1)
        self.lbl_window = QtWidgets.QLabel("")
        self.lbl_window.setObjectName("lbl_two_rings_window")
        self.lbl_window.setWordWrap(True)
        lay.addWidget(self.lbl_window)
        note = QtWidgets.QLabel(NOTE_TEXT)
        note.setObjectName("lbl_two_rings_note")
        note.setToolTip(NOTE_TIP)
        note.setWordWrap(True)
        lay.addWidget(note)
        btns = QtWidgets.QHBoxLayout()
        self.btn_csv = QtWidgets.QPushButton("Export CSV...")
        self.btn_csv.setObjectName("btn_two_rings_csv")
        self.btn_csv.setToolTip(CSV_TIP + "\n\n" + CSV_TIP_MORE)
        self.btn_csv.clicked.connect(lambda: self.export_csv())
        btns.addWidget(self.btn_csv)
        self.btn_png = QtWidgets.QPushButton("Export image...")
        self.btn_png.setObjectName("btn_two_rings_png")
        self.btn_png.clicked.connect(lambda: self.export_image())
        btns.addWidget(self.btn_png)
        btns.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        btns.addWidget(close)
        lay.addLayout(btns)
        self.window_item: Optional[pg.LinearRegionItem] = None
        self._load_pair()

    # -- state -------------------------------------------------------------------------------------------------
    def tau(self) -> float:
        return float(self.spin_tau.value())

    def _load_pair(self) -> None:
        pair = self.combo_pair.currentData()
        self.data = None
        if pair is None:
            self.lbl_counts.setText("This run has no pair of rings in its arc test.")
            return
        try:
            self.data = trv.pair_data(self.analyses, pair)
        except ValueError as exc:
            self.lbl_counts.setText(f"Rings {pair[0]}-{pair[1]}: {exc}")
            return
        self.window_item = None
        self.update_reading()

    def update_reading(self) -> None:
        if self.data is None:
            return
        tau = self.tau()
        self.reading = trv.read_pair(self.data, tau)
        r = self.reading
        if r.exploratory:
            self.lbl_role.setText("EXPLORATORY")
            self.lbl_role.setStyleSheet(f"color: {verdict('warn', dark=False)}; font-weight: bold;")
        else:
            self.lbl_role.setText("tau_0 (pre-specified)")
            self.lbl_role.setStyleSheet("")
        self.lbl_counts.setText(r.sentence(self.data))
        self._draw()

    # -- drawing -----------------------------------------------------------------------------------------------
    def _draw(self) -> None:
        d, r = self.data, self.reading
        if d is None or r is None:
            return
        keep = None if self.window_item is None else self.window_item.getRegion()
        p = self.plot_arc
        p.clear()
        L = d.length_nm / 1000.0
        matched_a = set(int(i) for i in r.arc_a)
        matched_b = set(int(j) for j in r.arc_b)
        for i, j in zip(r.arc_a, r.arc_b):
            sa, sb = float(d.s_a[int(i)]) / 1000.0, float(d.s_b[int(j)]) / 1000.0
            if abs(sa - sb) > L / 2:                     # the pair straddles the start of the closed membrane
                sb = sb + L if sb < sa else sb - L
            p.plot([sa, sb], [1.0, 0.0], pen=pg.mkPen(role("summary"), width=1.5))
        for y, s, matched in ((1.0, d.s_a, matched_a), (0.0, d.s_b, matched_b)):
            x = np.asarray(s, dtype=float) / 1000.0
            colours = [pg.mkBrush(role("observed")) if k in matched else pg.mkBrush(rgba("locs", 160))
                       for k in range(x.size)]
            p.plot(x, np.full(x.size, y), pen=None, symbol="o", symbolSize=9, symbolBrush=colours,
                   symbolPen=pg.mkPen(neutral(True)))
        p.setXRange(0.0, L, padding=0.01)
        p.setYRange(-0.6, 1.6, padding=0.0)
        p.getAxis("left").setTicks([[(1.0, f"ring {d.ring_a}"), (0.0, f"ring {d.ring_b}")]])
        tau_um = self.tau() / 1000.0
        centre = 0.5 * L if keep is None else 0.5 * (keep[0] + keep[1])
        self.window_item = pg.LinearRegionItem((centre - tau_um, centre + tau_um), movable=True,
                                               brush=pg.mkBrush(rgba("randomized", 70)))
        self.window_item.setZValue(-10)
        self.window_item.sigRegionChanged.connect(self._on_window_moved)
        p.addItem(self.window_item)
        mark = "" if not r.exploratory else "  [EXPLORATORY]"
        set_title(p, f"Rings {d.ring_a}-{d.ring_b} along the membrane: {r.n_arc} coincide within +/-{self.tau():.1f} "
                     f"nm (chance {r.expected_arc:.1f}){mark}")
        p.legend_box = self._legend(p)
        q = self.plot_plane
        q.clear()
        for i, j in zip(r.plane_a, r.plane_b):
            q.plot([d.xy_a[int(i), 0], d.xy_b[int(j), 0]], [d.xy_a[int(i), 1], d.xy_b[int(j), 1]],
                   pen=pg.mkPen(role("summary"), width=1.5))
        pa = set(int(i) for i in r.plane_a)
        pb = set(int(j) for j in r.plane_b)
        for xy, matched, sym in ((d.xy_a, pa, "o"), (d.xy_b, pb, "t")):
            colours = [pg.mkBrush(role("observed")) if k in matched else pg.mkBrush(rgba("locs", 160))
                       for k in range(xy.shape[0])]
            q.plot(xy[:, 0], xy[:, 1], pen=None, symbol=sym, symbolSize=8, symbolBrush=colours,
                   symbolPen=pg.mkPen(neutral(True)))
        set_title(q, f"In the plane: {r.n_plane} within {self.tau():.1f} nm (chance {r.expected_plane:.1f}){mark}")
        self._on_window_moved()

    @staticmethod
    def legend_entries() -> List[Tuple[str, Any]]:
        return [("coincides within tau", pg.ScatterPlotItem(symbol="o", size=8, brush=pg.mkBrush(role("observed")))),
                ("no partner within tau", pg.ScatterPlotItem(symbol="o", size=8, brush=pg.mkBrush(rgba("locs", 160)))),
                ("matched pair (one-to-one)", pg.PlotDataItem(pen=pg.mkPen(role("summary"), width=1.5))),
                ("window of 2 tau (drag it)", pg.BarGraphItem(x=[0.0], height=[1.0], width=1.0,
                                                               brush=pg.mkBrush(rgba("randomized", 110)), pen=None))]

    def _legend(self, plot: Any) -> Any:
        from tools.arrangement_ui import corner_legend
        return corner_legend(plot, self.legend_entries(), True)

    def _on_window_moved(self) -> None:
        if self.window_item is None or self.data is None:
            return
        lo, hi = self.window_item.getRegion()
        L = self.data.length_nm / 1000.0
        ins = []
        for name, s in ((self.data.ring_a, self.data.s_a), (self.data.ring_b, self.data.s_b)):
            x = np.asarray(s, dtype=float) / 1000.0
            inside = (x >= lo) & (x <= hi) | (x + L >= lo) & (x + L <= hi) | (x - L >= lo) & (x - L <= hi)
            ins.append(f"ring {name}: {int(inside.sum())}")
        self.lbl_window.setText(f"Window {lo:.3f}-{hi:.3f} um ({(hi - lo) * 1000.0:.0f} nm): " + ", ".join(ins)
                                + " cluster(s) inside.")

    # -- exports -----------------------------------------------------------------------------------------------
    def export_rows(self) -> List[dict]:
        if self.data is None or self.reading is None:
            return []
        return trv.export_rows(self.data, self.reading, self.source)

    def export_csv(self, path: Optional[str] = None) -> Optional[str]:
        from tools import multimethod as mm
        from tools.results_table import free_path
        rows = self.export_rows()
        if not rows:
            return None
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the two rings", "two_rings.csv",
                                                            "CSV (*.csv)")
            if not path:
                return None
        target = str(free_path(path))
        return mm.write_rows_csv(target, rows, list(trv.EXPORT_COLUMNS))

    def export_image(self, path: Optional[str] = None) -> Optional[str]:
        from tools.results_table import free_path
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export image", "two_rings.png", "PNG (*.png)")
            if not path:
                return None
        target = str(free_path(path))
        self.grab().save(target, "PNG")
        return target
