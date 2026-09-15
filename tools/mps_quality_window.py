# -*- coding: utf-8 -*-
"""
The data-quality panel: can this dataset answer the question?

Shows the checks in ``tools.mps_quality`` for the file currently loaded.
Deliberately separate from the per-axon results panel, because these
numbers are properties of the ACQUISITION, not of the axon: they do not
change when the clustering parameters are edited, and mixing them in
would invite reading them as results.

The findings list is at the top rather than the bottom. A quality panel
whose warnings are below the fold is a quality panel nobody reads.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from tools.mps_plot_style import AXIS_FG, TITLE_FG, set_title, style_dark
from tools.mps_quality import QualityReport, quality_report

_OK = "#5fd75f"
_WARN = "#ffaf5f"
_BAD = "#ff6b6b"
_DIM = "#9a9a9a"


def _label(text: str, colour: str = TITLE_FG, bold: bool = False) -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    weight = "bold" if bold else "normal"
    widget.setStyleSheet(f"color: {colour}; font-weight: {weight};")
    widget.setWordWrap(True)
    return widget


class MPSQualityWindow(QtWidgets.QMainWindow):
    """Acquisition-level quality checks for one loaded file."""

    def __init__(
        self,
        report: QualityReport,
        loc: Any,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.report = report
        self.loc = loc

        self.setWindowTitle("Data quality - " + os.path.basename(report.source))
        self.resize(1180, 820)
        self.setStyleSheet("QMainWindow { background: #1a1a1a; }")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.addWidget(self._build_header())
        root.addWidget(self._build_findings())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid #333; }} "
            f"QTabBar::tab {{ background: #262626; color: {AXIS_FG}; "
            f"padding: 6px 14px; }} "
            f"QTabBar::tab:selected {{ background: #3a3a3a; color: #fff; }}"
        )
        root.addWidget(self.tabs, stretch=1)
        self._build_precision_tab()
        self._build_box_tab()
        self._build_axial_tab()
        self._build_drift_tab()

    # ------------------------------------------------------------ header
    def _build_header(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 0)
        report, loc = self.report, self.loc
        bits = [
            f"{report.n_locs:,} localizations",
            f"{len(loc.columns)} columns",
            f"pixel {loc.pixel_size_nm:g} nm ({loc.pixel_size_source})"
            if loc.pixel_size_nm else "no pixel size",
        ]
        if loc.n_frames:
            bits.append(f"{loc.n_frames:,} frames")
        if loc.fit_method:
            bits.append(str(loc.fit_method))
        lay.addWidget(_label("   |   ".join(bits), _DIM))
        lay.addStretch(1)
        return box

    # ---------------------------------------------------------- findings
    def _build_findings(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Findings")
        box.setStyleSheet(
            f"QGroupBox {{ color: {TITLE_FG}; border: 1px solid #383838; "
            f"margin-top: 8px; }} "
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; }}"
        )
        lay = QtWidgets.QVBoxLayout(box)
        warnings = self.report.warnings
        if not warnings:
            lay.addWidget(
                _label("Every check that could run, ran clean.", _OK)
            )
        for text in warnings:
            lay.addWidget(_label("!  " + text, _WARN))
        for text in self.report.missing:
            lay.addWidget(
                _label("-  not checked: " + text, _DIM)
            )
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(box)
        scroll.setMaximumHeight(180)
        scroll.setStyleSheet("background: #1a1a1a; border: none;")
        return scroll

    # --------------------------------------------------------- precision
    def _build_precision_tab(self) -> None:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        check = self.report.precision
        if check is None:
            lay.addWidget(_label("No 'frame' or precision columns.", _DIM))
            self.tabs.addTab(page, "Precision")
            return

        colour = _OK if check.ratio < 1.3 else (
            _WARN if check.ratio < 2.0 else _BAD
        )
        rows = [
            ("NeNA (experimental)", f"{check.nena_nm:.2f} nm"),
            ("reported lpx (median)", f"{check.reported_lpx_nm:.2f} nm"),
            ("reported lpy (median)", f"{check.reported_lpy_nm:.2f} nm"),
            ("reported lpz (median)",
             "-" if check.reported_lpz_nm is None
             else f"{check.reported_lpz_nm:.2f} nm"),
            ("NeNA / reported", f"{check.ratio:.2f}"),
            ("next-frame pairs", f"{check.n_pairs:,}"),
        ]
        grid = QtWidgets.QGridLayout()
        for row, (name, value) in enumerate(rows):
            grid.addWidget(_label(name, _DIM), row, 0)
            grid.addWidget(_label(value, TITLE_FG, bold=True), row, 1)
        grid.setColumnStretch(2, 1)
        lay.addLayout(grid)
        lay.addWidget(_label(check.verdict, colour, bold=True))
        lay.addWidget(_label(
            "NeNA measures the precision from the data itself: the same "
            "molecule seen in two consecutive frames appears twice, and the "
            "spread between those two positions is the error. It needs no "
            "photon calibration, which is what makes it a check on the "
            "precision the fit reported.", _DIM))

        plot = pg.PlotWidget()
        style_dark(plot)
        set_title(plot, "Next-frame neighbour distances")
        plot.setLabel("bottom", "distance [nm]", color=AXIS_FG)
        plot.setLabel("left", "count", color=AXIS_FG)
        # Recompute so the curve can be drawn; the summary kept only numbers.
        from tools.mps_quality import nena as _nena
        detail = _nena(
            self.loc.frame, self.loc.x_nm, self.loc.y_nm,
            reported_lp_nm=check.reported_lateral_nm,
        )
        if detail.histogram.size:
            plot.plot(detail.distances_nm, detail.histogram,
                      pen=pg.mkPen("#6fa8ff", width=1))
        if detail.best_fit.size:
            plot.plot(detail.distances_nm, detail.best_fit,
                      pen=pg.mkPen("#ff9f43", width=2))
        if np.isfinite(check.nena_nm):
            plot.addItem(pg.InfiniteLine(
                pos=check.nena_nm, angle=90,
                pen=pg.mkPen("#5fd75f", width=2, style=QtCore.Qt.DashLine),
                label=f"NeNA {check.nena_nm:.1f} nm",
                labelOpts={"color": "#5fd75f", "position": 0.9}))
        lay.addWidget(plot, stretch=1)
        self.tabs.addTab(page, "Precision")

    # --------------------------------------------------------- box size
    def _build_box_tab(self) -> None:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        check = self.report.box
        if check is None:
            lay.addWidget(_label("No 'sx'/'sy' columns.", _DIM))
            self.tabs.addTab(page, "Fitting box")
            return

        lay.addWidget(_label(
            f"Box Size: {check.box_size_px} px      "
            f"sx median {check.sx_median_px:.2f} px, p95 {check.sx_p95_px:.2f}      "
            f"sy median {check.sy_median_px:.2f} px, p95 {check.sy_p95_px:.2f}",
            TITLE_FG, bold=True))
        if check.box_size_px:
            lay.addWidget(_label(
                f"Spots wider than the box: "
                f"{100 * check.fraction_over_box_2sigma:.1f} % at +-2 sigma, "
                f"{100 * check.fraction_over_box_3sigma:.1f} % at +-3 sigma.",
                _WARN if check.fraction_over_box_2sigma > 0.05 else _OK))
        lay.addWidget(_label(
            "An astigmatic PSF is widest at the ends of the z range. A spot "
            "wider than its fitting box is fitted on a truncated image, so "
            "its sx and sy come out too small -- and z, lpz and the "
            "ellipticity cut are all derived from sx and sy.", _DIM))

        plot = pg.PlotWidget()
        style_dark(plot)
        set_title(plot, "PSF width against the fitting box")
        plot.setLabel("bottom", "PSF sigma [camera pixels]", color=AXIS_FG)
        plot.setLabel("left", "count", color=AXIS_FG)
        for name, colour in (("sx", "#6fa8ff"), ("sy", "#ff9f43")):
            values = self.loc.column(name)
            if values is None:
                continue
            counts, edges = np.histogram(
                np.asarray(values, dtype=float), bins=120
            )
            plot.plot(0.5 * (edges[:-1] + edges[1:]), counts,
                      pen=pg.mkPen(colour, width=1), name=name)
        if check.box_size_px:
            for divisor, style, text in (
                (4.0, QtCore.Qt.DashLine, "box / 4  (+-2 sigma limit)"),
                (6.0, QtCore.Qt.DotLine, "box / 6  (+-3 sigma limit)"),
            ):
                plot.addItem(pg.InfiniteLine(
                    pos=check.box_size_px / divisor, angle=90,
                    pen=pg.mkPen(_BAD, width=2, style=style),
                    label=text,
                    labelOpts={"color": _BAD, "position": 0.85}))
        lay.addWidget(plot, stretch=1)
        self.tabs.addTab(page, "Fitting box")

    # ------------------------------------------------------------ axial
    def _build_axial_tab(self) -> None:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        check = self.report.axial
        if check is None:
            lay.addWidget(_label(
                "Needs the Z periodicity fit and an 'lpz' column.", _DIM))
            self.tabs.addTab(page, "Axial resolvedness")
            return

        lay.addWidget(_label(
            "Is each axial component as narrow as the data can show, or is "
            "it carrying real width? A ring that were infinitely thin would "
            "still be lpz wide, because that is what the microscope does to "
            "a point.", _DIM))

        table = QtWidgets.QTableWidget(len(check.components), 6)
        table.setHorizontalHeaderLabels(
            ["mean z [nm]", "sigma [nm]", "lpz [nm]", "sigma/lpz",
             "structural [nm]", "verdict"])
        table.setStyleSheet(
            f"QTableWidget {{ background: #202020; color: {TITLE_FG}; "
            f"gridline-color: #383838; }} "
            f"QHeaderView::section {{ background: #2a2a2a; "
            f"color: {AXIS_FG}; border: 0; padding: 4px; }}")
        for row, component in enumerate(check.components):
            if component.ratio < 0.7:
                verdict, colour = "narrower than lpz - impossible", _BAD
            elif component.is_resolved:
                verdict, colour = "precision-limited", _OK
            elif component.ratio > 2.0:
                verdict, colour = "broad - may be two rings", _BAD
            else:
                verdict, colour = "slightly broad", _WARN
            values = [
                f"{component.mean_nm:.1f}",
                f"{component.sigma_nm:.1f}",
                f"{component.lpz_nm:.1f}",
                f"{component.ratio:.2f}",
                f"{component.structural_width_nm:.1f}",
                verdict,
            ]
            for column, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if column == 5:
                    item.setForeground(pg.mkColor(colour))
                table.setItem(row, column, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setMaximumHeight(46 + 28 * len(check.components))
        lay.addWidget(table)

        plot = pg.PlotWidget()
        style_dark(plot)
        set_title(plot, "Z distribution, with each component's width")
        plot.setLabel("bottom", "z [nm]", color=AXIS_FG)
        plot.setLabel("left", "count", color=AXIS_FG)
        counts, edges = np.histogram(self.loc.z_nm, bins=120)
        centres = 0.5 * (edges[:-1] + edges[1:])
        plot.plot(centres, counts, pen=pg.mkPen("#6fa8ff", width=1))
        peak = float(counts.max()) if counts.size else 1.0
        for component in check.components:
            colour = _OK if component.is_resolved else _WARN
            plot.addItem(pg.InfiniteLine(
                pos=component.mean_nm, angle=90,
                pen=pg.mkPen(colour, width=2, style=QtCore.Qt.DashLine)))
            # The bar spans +-sigma; the tick marks +-lpz for comparison.
            bar = pg.PlotDataItem(
                [component.mean_nm - component.sigma_nm,
                 component.mean_nm + component.sigma_nm],
                [peak * 0.94, peak * 0.94],
                pen=pg.mkPen(colour, width=6))
            plot.addItem(bar)
            plot.addItem(pg.PlotDataItem(
                [component.mean_nm - component.lpz_nm,
                 component.mean_nm + component.lpz_nm],
                [peak * 0.88, peak * 0.88],
                pen=pg.mkPen(_DIM, width=3)))
        lay.addWidget(plot, stretch=1)
        lay.addWidget(_label(
            "Thick bar: the fitted component width (+-sigma). Thin grey bar "
            "below it: the axial precision (+-lpz) of the localizations "
            "there. When the two are the same length, the component is as "
            "thin as this data can resolve.", _DIM))
        self.tabs.addTab(page, "Axial resolvedness")

    # ------------------------------------------------------------ drift
    def _build_drift_tab(self) -> None:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        check = self.report.drift
        if check is None or not check.axes:
            lay.addWidget(_label("No 'frame' column, or too few segments.",
                                 _DIM))
            self.tabs.addTab(page, "Residual drift")
            return

        for axis in check.axes:
            colour = _BAD if axis.is_coherent else _OK
            verdict = (
                "a smooth drift" if axis.is_coherent
                else "no coherent drift (consistent with counting noise)"
            )
            lay.addWidget(_label(
                f"{axis.axis}:  profile moves over a range of "
                f"{axis.shift_range_nm:.0f} nm;  lag-1 autocorrelation "
                f"{axis.lag1_autocorrelation:+.2f},  permutation p = "
                f"{axis.p_permutation:.3f}   ->   {verdict}", colour))
        lay.addWidget(_label(
            "Real drift is smooth: consecutive time segments move by similar "
            "amounts. Counting noise is not. Note that Picasso's RCC "
            "undrifting is 2D and never touches z; only AIM corrects the "
            "axial direction.", _DIM))
        lay.addStretch(1)
        self.tabs.addTab(page, "Residual drift")


def show_quality_window(
    loc: Any,
    *,
    means_nm: Optional[Union[Sequence[float], np.ndarray]] = None,
    sigmas_nm: Optional[Union[Sequence[float], np.ndarray]] = None,
    weights: Optional[Union[Sequence[float], np.ndarray]] = None,
    period_nm: Optional[float] = None,
    parent: Optional[QtWidgets.QWidget] = None,
) -> MPSQualityWindow:
    """Run every check and open the panel on the result."""
    report = quality_report(
        loc, means_nm=means_nm, sigmas_nm=sigmas_nm, weights=weights,
        period_nm=period_nm,
    )
    window = MPSQualityWindow(report, loc, parent=parent)
    window.show()
    window.raise_()
    window.activateWindow()
    return window
