# -*- coding: utf-8 -*-
"""
The DNA-PAINT panel, with qPAINT counting on its own tab.

Separate from everything else on purpose. DNA-PAINT changes what the
software's existing numbers MEAN -- a cluster's localization count stops
being a proxy for how much protein is there and becomes a measure of how
long you imaged -- so the DNA-PAINT path must never be something the user
can wander into without noticing.

qPAINT is separated again inside it, because it carries one more
requirement the rest does not: a calibration. Linking and kinetics work
on any DNA-PAINT movie; a docking-site COUNT needs a reference of known
valency imaged in the same sample at the same imager concentration.
Without it the numbers are a ranking, and the tab says so in the loudest
way a tab can.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from tools.mps_paint import (
    DEFAULT_MAX_DARK_TIME,
    LINK_RADIUS_IN_PRECISIONS,
    PaintReport,
    cumulative_exponential,
    dark_times,
    estimate_kinetic_rate,
    paint_report,
    qpaint,
)
from tools.mps_plot_style import AXIS_FG, TITLE_FG, set_title, style_dark

_OK = "#5fd75f"
_WARN = "#ffaf5f"
_BAD = "#ff6b6b"
_DIM = "#9a9a9a"


def _label(text: str, colour: str = TITLE_FG,
           bold: bool = False) -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    widget.setStyleSheet(
        f"color: {colour}; font-weight: {'bold' if bold else 'normal'};"
    )
    widget.setWordWrap(True)
    return widget


class MPSPaintWindow(QtWidgets.QMainWindow):
    """Binding events, sticking, kinetics and qPAINT for one loaded file."""

    def __init__(
        self,
        loc: Any,
        site_labels: Optional[np.ndarray] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        exposure_s: Optional[float] = None,
    ):
        super().__init__(parent)
        self.loc = loc
        self.site_labels = site_labels
        self.exposure_s = exposure_s
        self.report: Optional[PaintReport] = None

        self.setWindowTitle(
            "DNA-PAINT - " + os.path.basename(getattr(loc, "path", ""))
        )
        self.resize(1200, 860)
        self.setStyleSheet("QMainWindow { background: #1a1a1a; }")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.addWidget(self._build_controls())
        self.findings = QtWidgets.QVBoxLayout()
        holder = QtWidgets.QWidget()
        holder.setLayout(self.findings)
        root.addWidget(holder)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid #333; }} "
            f"QTabBar::tab {{ background: #262626; color: {AXIS_FG}; "
            f"padding: 6px 14px; }} "
            f"QTabBar::tab:selected {{ background: #3a3a3a; color: #fff; }}"
        )
        root.addWidget(self.tabs, stretch=1)
        self.page_events = QtWidgets.QWidget()
        self.page_kinetics = QtWidgets.QWidget()
        self.page_qpaint = QtWidgets.QWidget()
        # The layouts are kept as QVBoxLayout rather than reached through
        # QWidget.layout(), which returns the QLayout base class and loses
        # the `stretch` argument the plots need.
        self.layout_events = QtWidgets.QVBoxLayout(self.page_events)
        self.layout_kinetics = QtWidgets.QVBoxLayout(self.page_kinetics)
        self.layout_qpaint = QtWidgets.QVBoxLayout(self.page_qpaint)
        for page, name in (
            (self.page_events, "Binding events"),
            (self.page_kinetics, "Kinetics"),
            (self.page_qpaint, "qPAINT counting"),
        ):
            self.tabs.addTab(page, name)

        self.recompute()

    # ---------------------------------------------------------- controls
    def _build_controls(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(box)

        lp = self.loc.lp_lateral_nm
        suggested = (
            LINK_RADIUS_IN_PRECISIONS * float(np.nanmedian(lp))
            if lp is not None else 25.0
        )
        lay.addWidget(_label("Link radius [nm]:", _DIM))
        self.spin_radius = QtWidgets.QDoubleSpinBox()
        self.spin_radius.setRange(1.0, 1000.0)
        self.spin_radius.setDecimals(1)
        self.spin_radius.setValue(suggested)
        self.spin_radius.setToolTip(
            f"Maximum step between consecutive localizations of one binding "
            f"event. Suggested {LINK_RADIUS_IN_PRECISIONS:g} x the median "
            f"lateral precision of this file."
        )
        lay.addWidget(self.spin_radius)

        lay.addWidget(_label("Max dark frames:", _DIM))
        self.spin_dark = QtWidgets.QSpinBox()
        self.spin_dark.setRange(0, 20)
        self.spin_dark.setValue(DEFAULT_MAX_DARK_TIME)
        self.spin_dark.setToolTip(
            "Frames that may be missing inside one binding event. Raising "
            "it merges events and therefore lengthens tau_dark, which is "
            "the qPAINT measurement itself."
        )
        lay.addWidget(self.spin_dark)

        lay.addWidget(_label("Exposure [s]:", _DIM))
        self.spin_exposure = QtWidgets.QDoubleSpinBox()
        self.spin_exposure.setRange(0.0, 10.0)
        self.spin_exposure.setDecimals(3)
        self.spin_exposure.setValue(self.exposure_s or 0.0)
        self.spin_exposure.setToolTip(
            "Only used to report times in seconds as well as frames."
        )
        lay.addWidget(self.spin_exposure)

        self.btn_run = QtWidgets.QPushButton("Recompute")
        self.btn_run.clicked.connect(self.recompute)
        lay.addWidget(self.btn_run)
        lay.addStretch(1)
        return box

    # ----------------------------------------------------------- compute
    def recompute(self) -> None:
        self.report = paint_report(
            self.loc,
            site_labels=self.site_labels,
            radius_nm=float(self.spin_radius.value()),
            max_dark_time=int(self.spin_dark.value()),
            exposure_s=(
                float(self.spin_exposure.value())
                if self.spin_exposure.value() > 0 else None
            ),
            run_qpaint=False,   # the qPAINT tab drives its own calibration
        )
        self._refresh_findings()
        self._fill_events_tab()
        self._fill_kinetics_tab()
        self._fill_qpaint_tab()

    def _clear(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _refresh_findings(self) -> None:
        self._clear(self.findings)
        assert self.report is not None
        for text in self.report.missing:
            self.findings.addWidget(_label("-  " + text, _BAD))
        for text in self.report.warnings:
            self.findings.addWidget(_label("!  " + text, _WARN))

    # ------------------------------------------------------------ events
    def _fill_events_tab(self) -> None:
        lay = self.layout_events
        self._clear(lay)
        report = self.report
        assert report is not None
        if report.events is None:
            lay.addWidget(_label(
                "No binding events: this file has no 'frame' column.", _BAD))
            return

        events = report.events
        lay.addWidget(_label(
            f"{report.n_locs:,} localizations  ->  {events.n:,} binding "
            f"events     ({report.compression:.2f} localizations per event, "
            f"link radius {report.radius_nm:.1f} nm, "
            f"max dark {events.max_dark_time} frames)",
            TITLE_FG, bold=True))
        lay.addWidget(_label(
            "Everything downstream must be recomputed on EVENTS, not on "
            "localizations. In DNA-PAINT a docking site keeps emitting for "
            "the whole acquisition, so a cluster's localization count "
            "measures imaging time, not how much protein is there.", _DIM))

        if report.sticking is not None:
            sticking = report.sticking
            kept = sticking.labels.size - sticking.n_rejected
            colour = _WARN if sticking.n_rejected else _OK
            lay.addWidget(_label(
                f"Sticking filter: {kept} of {sticking.labels.size} clusters "
                f"kept, {sticking.n_rejected} rejected as non-specific "
                f"sticking (localizations confined to a short stretch of "
                f"the movie).", colour))
        else:
            lay.addWidget(_label(
                "Sticking filter not run: it needs cluster labels. Cluster "
                "the ROI first, then reopen this panel.", _DIM))

        row = QtWidgets.QWidget()
        row_lay = QtWidgets.QHBoxLayout(row)

        length_plot = pg.PlotWidget()
        style_dark(length_plot)
        set_title(length_plot, "Binding event length")
        length_plot.setLabel("bottom", "frames", color=AXIS_FG)
        length_plot.setLabel("left", "count", color=AXIS_FG)
        if events.n:
            top = int(np.percentile(events.length, 99)) + 2
            counts, edges = np.histogram(
                events.length, bins=np.arange(0.5, top + 1.5, 1.0))
            length_plot.plot(
                0.5 * (edges[:-1] + edges[1:]), counts,
                stepMode=False, pen=pg.mkPen("#6fa8ff", width=2))
        row_lay.addWidget(length_plot)

        time_plot = pg.PlotWidget()
        style_dark(time_plot)
        set_title(time_plot, "Binding events over the acquisition")
        time_plot.setLabel("bottom", "frame", color=AXIS_FG)
        time_plot.setLabel("left", "events", color=AXIS_FG)
        if events.n:
            counts, edges = np.histogram(events.first_frame, bins=60)
            time_plot.plot(0.5 * (edges[:-1] + edges[1:]), counts,
                           pen=pg.mkPen("#ff9f43", width=2))
        row_lay.addWidget(time_plot)
        lay.addWidget(row, stretch=1)
        lay.addWidget(_label(
            "A flat rate of events over the acquisition is what a real "
            "docking site looks like. A decaying one means bleaching or "
            "imager depletion; a spike means sticking.", _DIM))

    # ---------------------------------------------------------- kinetics
    def _fill_kinetics_tab(self) -> None:
        lay = self.layout_kinetics
        self._clear(lay)
        report = self.report
        assert report is not None
        result = report.kinetics_result
        if result is None or report.events is None:
            lay.addWidget(_label("No kinetics to show.", _DIM))
            return

        def as_time(frames: float, seconds: Optional[float]) -> str:
            if not np.isfinite(frames):
                return "-"
            return (f"{frames:.1f} frames"
                    + (f"  ({seconds:.2f} s)" if seconds else ""))

        lay.addWidget(_label(
            f"tau_bright = {as_time(result.tau_bright_frames, result.tau_bright_s)}"
            f"        tau_dark = {as_time(result.tau_dark_frames, result.tau_dark_s)}"
            f"        {result.n_events:,} events over {result.n_sites} site(s)",
            TITLE_FG, bold=True))
        lay.addWidget(_label(
            "Both are fitted to the cumulative distribution rather than "
            "averaged. Dark periods longer than the remaining acquisition "
            "are never observed, so the sample mean is biased low by an "
            "amount that depends on how long you imaged; the fitted time "
            "constant is not.", _DIM))

        row = QtWidgets.QWidget()
        row_lay = QtWidgets.QHBoxLayout(row)
        events = report.events
        labels = (
            events.group if events.group is not None
            else np.zeros(events.n, dtype=np.int64)
        )
        dark = dark_times(events.first_frame, events.last_frame, labels)
        for name, data, tau, colour in (
            ("bright (event length)", events.length.astype(float),
             result.tau_bright_frames, "#6fa8ff"),
            ("dark (between events)", dark[dark > 0].astype(float),
             result.tau_dark_frames, "#ff9f43"),
        ):
            plot = pg.PlotWidget()
            style_dark(plot)
            set_title(plot, f"Cumulative {name}")
            plot.setLabel("bottom", "frames", color=AXIS_FG)
            plot.setLabel("left", "cumulative count", color=AXIS_FG)
            if data.size > 2:
                ordered = np.sort(data)
                ranks = np.arange(1, ordered.size + 1, dtype=float)
                plot.plot(ordered, ranks, pen=pg.mkPen(colour, width=2))
                if np.isfinite(tau) and tau > 0:
                    fitted = cumulative_exponential(
                        ordered, float(ordered.size), float(tau), 0.0)
                    plot.plot(ordered, fitted,
                              pen=pg.mkPen("#ffffff", width=1,
                                           style=QtCore.Qt.DashLine))
            row_lay.addWidget(plot)
        lay.addWidget(row, stretch=1)

    # ------------------------------------------------------------ qPAINT
    def _fill_qpaint_tab(self) -> None:
        lay = self.layout_qpaint
        self._clear(lay)
        report = self.report
        assert report is not None
        if report.events is None or report.events.group is None:
            lay.addWidget(_label(
                "qPAINT counts docking sites per CLUSTER, so it needs "
                "cluster labels. Cluster the ROI first, then reopen this "
                "panel.", _BAD))
            return

        banner = QtWidgets.QWidget()
        banner_lay = QtWidgets.QHBoxLayout(banner)
        banner_lay.addWidget(_label("Calibration:", _DIM))
        self.combo_calibration = QtWidgets.QComboBox()
        self.combo_calibration.addItems([
            "none - relative units only",
            "single-site reference tau_dark [frames]",
            "influx rate [per frame]",
        ])
        banner_lay.addWidget(self.combo_calibration)
        self.spin_calibration = QtWidgets.QDoubleSpinBox()
        self.spin_calibration.setRange(0.0, 1e6)
        self.spin_calibration.setDecimals(4)
        self.spin_calibration.setValue(0.0)
        banner_lay.addWidget(self.spin_calibration)
        button = QtWidgets.QPushButton("Count")
        button.clicked.connect(self._run_qpaint)
        banner_lay.addWidget(button)
        banner_lay.addStretch(1)
        lay.addWidget(banner)

        lay.addWidget(_label(
            "qPAINT is a RELATIVE measurement. N docking sites capture "
            "imagers N times as fast, so tau_dark falls as 1/N -- but "
            "turning that into a count needs a reference of known valency "
            "imaged in the same sample at the same imager concentration "
            "(a single-docking-strand origami is the standard one). Without "
            "it, the numbers rank clusters and nothing more.", _DIM))

        self.qpaint_holder = QtWidgets.QVBoxLayout()
        holder = QtWidgets.QWidget()
        holder.setLayout(self.qpaint_holder)
        lay.addWidget(holder, stretch=1)
        self._run_qpaint()

    def _run_qpaint(self) -> None:
        self._clear(self.qpaint_holder)
        report = self.report
        if report is None or report.events is None or \
                report.events.group is None:
            return

        mode = self.combo_calibration.currentIndex()
        value = float(self.spin_calibration.value())
        kwargs: dict = {}
        if mode == 1 and value > 0:
            kwargs["tau_dark_reference_frames"] = value
        elif mode == 2 and value > 0:
            kwargs["influx_rate"] = value

        result = qpaint(report.events, report.events.group, **kwargs)

        if result.is_calibrated:
            self.qpaint_holder.addWidget(_label(
                f"Calibrated: {result.calibration_source}. The numbers below "
                f"are docking sites per cluster.", _OK, bold=True))
        else:
            self.qpaint_holder.addWidget(_label(
                "UNCALIBRATED - these are relative units scaled so the "
                "median cluster reads 1. They rank clusters; they are NOT "
                "docking-site counts and must not be reported as such.",
                _BAD, bold=True))
        for text in result.warnings:
            self.qpaint_holder.addWidget(_label("!  " + text, _WARN))

        usable = np.isfinite(result.n_units)
        if np.any(usable):
            units = result.n_units[usable]
            self.qpaint_holder.addWidget(_label(
                f"{int(usable.sum())} of {result.labels.size} clusters "
                f"counted:  median {np.median(units):.2f}, "
                f"range {units.min():.2f} - {units.max():.2f}"
                + ("  units" if result.is_calibrated else "  (relative)"),
                TITLE_FG, bold=True))

            plot = pg.PlotWidget()
            style_dark(plot)
            set_title(
                plot,
                "Docking sites per cluster" if result.is_calibrated
                else "Relative binding frequency per cluster",
            )
            plot.setLabel(
                "bottom",
                "units" if result.is_calibrated else "relative units",
                color=AXIS_FG)
            plot.setLabel("left", "clusters", color=AXIS_FG)
            counts, edges = np.histogram(units, bins=40)
            plot.plot(0.5 * (edges[:-1] + edges[1:]), counts,
                      pen=pg.mkPen("#6fa8ff", width=2))
            if result.is_calibrated:
                for integer in range(1, min(int(np.ceil(units.max())) + 1, 13)):
                    plot.addItem(pg.InfiniteLine(
                        pos=integer, angle=90,
                        pen=pg.mkPen(_DIM, width=1,
                                     style=QtCore.Qt.DotLine)))
            self.qpaint_holder.addWidget(plot, stretch=1)
            if result.is_calibrated:
                self.qpaint_holder.addWidget(_label(
                    "Dotted lines mark whole numbers of sites. A histogram "
                    "with peaks on them is the sign that the calibration is "
                    "right; a smear across them means it is not, or that the "
                    "clusters hold varying numbers of labelled molecules.",
                    _DIM))
        else:
            self.qpaint_holder.addWidget(_label(
                "No cluster has enough binding events for a dark-time "
                "distribution. Either the acquisition is too short, or the "
                "clusters are too small.", _BAD))


def show_paint_window(
    loc: Any,
    site_labels: Optional[np.ndarray] = None,
    parent: Optional[QtWidgets.QWidget] = None,
    exposure_s: Optional[float] = None,
) -> MPSPaintWindow:
    """Open the DNA-PAINT panel on a loaded file."""
    window = MPSPaintWindow(loc, site_labels, parent=parent,
                            exposure_s=exposure_s)
    window.show()
    window.raise_()
    window.activateWindow()
    return window
