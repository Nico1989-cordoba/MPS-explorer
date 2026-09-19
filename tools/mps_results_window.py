# -*- coding: utf-8 -*-
"""
Results window for the automatic per-axon MPS analysis.

Shows every parameter computed by ``tools.mps_analysis.analyze_axon`` next
to the corresponding value from Gazal et al. (2026), together with the
plots that make the numbers interpretable (reconstructed contour, cluster
area distribution, 1NN distribution, axial histogram with the GMM fit and
the selected slab).

Design principle requested by the user: the analysis runs automatically and
instantly, but *every* automatic choice stays editable, because reading the
histograms is where expert judgement enters. Editable here:

  * the axial peak the 180 nm slab is centred on (a selector listing every
    peak the GMM found -- this is what the "ambiguous main peak" warning
    refers to)
  * the slab half-width
  * DBSCAN eps and min_samples
  * the DBCV threshold used for automatic bad-cluster removal

Changing any of them re-runs the analysis on the same localizations and
redraws everything.

Once the axoplasm panel has found the clusters not anchored to the
membrane, another column appears: every parameter again with the discard
applied. Both it and the measured column build the contour by 2-opt from
every start, so they differ only by the clusters left out. (An analysis
refined from one start, as this program did before 2026-09-19, gets a
third column in between: all the clusters, every start.) The plots show any of them,
and each has its own export, so the user decides which numbers go to the
statistics. The contour plot marks the centre of each contour, its area
centroid, with a cross.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import os
from typing import Any, Callable, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from tools.cluster_quality import good_cluster_labels
from tools.mps_analysis import (
    ANALYSIS_COLUMNS, AxonAnalysis, DiscardComparison)
from tools.mps_plot_style import AXIS_FG, set_title, style_dark
from tools.results_table import append_rows, refuse_other_analysis


# Colour-blind-safe palette, matching the one already used in MPS_explorer.
_C_ORANGE = "#d55e00"
_C_GREEN = "#009e73"
_C_BLUE = "#0072b2"
_C_GREY = "#888888"
# Outline for filled markers. Was black, to make them crisp against the
# white background these plots used to have.
_C_DARK_OUTLINE = "#e0e0e0"
# The clusters the discard left out, as the axoplasm panel draws them.
_C_DISCARDED = "#ff4040"
# The centre of the contour shown (Okabe-Ito reddish purple, as in the
# axoplasm panel, whose yellow is already the spectrin interior).
_C_CENTRE = "#cc79a7"

# Columns of the parameter table.
(_COL_NAME, _COL_MEASURED, _COL_EVERY, _COL_DISCARD, _COL_PAPER,
 _COL_NOTE) = range(6)


class MPSResultsWindow(QtWidgets.QMainWindow):
    """Panel showing the per-axon parameters, plots and warnings."""

    def __init__(
        self,
        analysis: AxonAnalysis,
        rerun_callback: Optional[Callable[..., AxonAnalysis]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        rings_callback: Optional[Callable[[], Any]] = None,
        discard_callback: Optional[
            Callable[[AxonAnalysis], Optional[DiscardComparison]]] = None,
    ):
        """
        Parameters
        ----------
        analysis : the initial result to display.
        rerun_callback : called with keyword overrides
            (main_peak_override_nm, slab_half_width_nm, eps_nm,
            min_samples, dbcv_threshold) when the user edits a control;
            must return a fresh AxonAnalysis. If None, the controls are
            disabled and the window is read-only.
        rings_callback : called when the user presses "Rings...", to open
            the multi-segment panel. None hides that route.
        discard_callback : given the analysis shown, returns it with all
            the clusters and without the ones the axoplasm panel discarded
            (tools.mps_analysis.compare_discard), or None when there is
            none for it. Asked again at every refresh.
        """
        super().__init__(parent)
        self.analysis = analysis
        self.rerun_callback = rerun_callback
        self.rings_callback = rings_callback
        self.discard_callback = discard_callback
        self.comparison: Optional[DiscardComparison] = None

        self.setWindowTitle("MPS analysis - per-axon parameters")
        self.resize(1250, 860)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        root.addWidget(self._build_controls())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_left_column())
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, stretch=1)
        self.splitter = splitter
        # The table is widened once when its extra columns appear; after
        # that the divider stays where the user leaves it.
        self._widened = False

        self.refresh()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_controls(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Parameters (editable - the analysis re-runs on change)")
        lay = QtWidgets.QHBoxLayout(box)

        lay.addWidget(QtWidgets.QLabel("Axial peak:"))
        self.combo_peak = QtWidgets.QComboBox()
        self.combo_peak.setMinimumWidth(230)
        self.combo_peak.setToolTip(
            "Centre of the 180 nm axial slab that isolates one MPS segment.\n"
            "Defaults to the main peak of the fitted Gaussian mixture. When\n"
            "two peaks have comparable weight the choice is ambiguous and\n"
            "changes which segment is analysed - pick it yourself here."
        )
        lay.addWidget(self.combo_peak)

        lay.addSpacing(12)
        lay.addWidget(QtWidgets.QLabel("Slab half-width [nm]:"))
        self.spin_half = QtWidgets.QDoubleSpinBox()
        self.spin_half.setRange(1.0, 5000.0)
        self.spin_half.setDecimals(1)
        self.spin_half.setSingleStep(5.0)
        self.spin_half.setToolTip(
            "Half of the axial window. The paper uses 90 nm, i.e. a 180 nm slab."
        )
        lay.addWidget(self.spin_half)

        lay.addSpacing(12)
        lay.addWidget(QtWidgets.QLabel("eps [nm]:"))
        self.spin_eps = QtWidgets.QDoubleSpinBox()
        self.spin_eps.setRange(0.1, 1000.0)
        self.spin_eps.setDecimals(1)
        self.spin_eps.setToolTip(
            "DBSCAN search radius. The paper uses 25 nm, matching its ~20 nm\n"
            "lateral localization precision."
        )
        lay.addWidget(self.spin_eps)

        lay.addWidget(QtWidgets.QLabel("min samples:"))
        self.spin_min = QtWidgets.QSpinBox()
        self.spin_min.setRange(1, 10000)
        self.spin_min.setToolTip(
            "DBSCAN minimum points per cluster. The paper uses 10, matching\n"
            "the expected number of blinking cycles per fluorophore."
        )
        lay.addWidget(self.spin_min)

        lay.addWidget(QtWidgets.QLabel("DBCV thr.:"))
        self.spin_dbcv = QtWidgets.QDoubleSpinBox()
        self.spin_dbcv.setRange(-1.0, 1.0)
        self.spin_dbcv.setDecimals(2)
        self.spin_dbcv.setSingleStep(0.05)
        self.spin_dbcv.setToolTip(
            "Clusters scoring below this density-validation index are removed\n"
            "automatically. OFF by default (-1.0): DBCV score correlates with\n"
            "log(cluster area) at -0.78 to -0.79, so raising this threshold\n"
            "also removes large clusters -- exactly the ones Gazal et al.\n"
            "(2026) interpret as spectrin oligomers and keep. Raise only\n"
            "deliberately."
        )
        lay.addWidget(self.spin_dbcv)

        lay.addWidget(QtWidgets.QLabel("Mahalanobis:"))
        self.spin_maha = QtWidgets.QDoubleSpinBox()
        self.spin_maha.setRange(0.1, 10.0)
        self.spin_maha.setDecimals(1)
        self.spin_maha.setSingleStep(0.5)
        self.spin_maha.setToolTip(
            "Occupancy threshold: a perimeter point counts as occupied when\n"
            "it lies within this Mahalanobis distance of some cluster's\n"
            "constrained Gaussian. The paper uses 3."
        )
        lay.addWidget(self.spin_maha)

        self.chk_random = QtWidgets.QCheckBox("Randomization")
        self.chk_random.setChecked(True)
        self.chk_random.setToolTip(
            "Run the 1,000-iteration randomization control (parameter 8).\n"
            "It is the slowest step (a few seconds per axon), so switch it\n"
            "off while tuning the other parameters and back on for the\n"
            "final numbers."
        )
        lay.addWidget(self.chk_random)

        lay.addStretch(1)

        self.btn_rings = QtWidgets.QPushButton("Rings...")
        self.btn_rings.setToolTip(
            "Analyse EVERY axial segment of this axon, not just this slab,\n"
            "and compare the gap/patch pattern of consecutive segments."
        )
        self.btn_rings.clicked.connect(self._on_rings)
        self.btn_rings.setEnabled(self.rings_callback is not None)
        lay.addWidget(self.btn_rings)

        self.btn_reset = QtWidgets.QPushButton("Reset to paper defaults")
        self.btn_reset.clicked.connect(self._on_reset)
        lay.addWidget(self.btn_reset)

        self.btn_export = QtWidgets.QPushButton("Export CSV")
        self.btn_export.setToolTip(
            "The measured parameters (the first column), one row per axon.")
        self.btn_export.clicked.connect(self._on_export)
        lay.addWidget(self.btn_export)

        enabled = self.rerun_callback is not None
        for w in (self.combo_peak, self.spin_half, self.spin_eps,
                  self.spin_min, self.spin_dbcv, self.spin_maha,
                  self.chk_random, self.btn_reset):
            w.setEnabled(enabled)

        return box

    def _on_rings(self) -> None:
        if self.rings_callback is not None:
            self.rings_callback()

    def _build_left_column(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        self.lbl_provenance = QtWidgets.QLabel()
        self.lbl_provenance.setWordWrap(True)
        self.lbl_provenance.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.lbl_provenance)

        # Which analysis the plots show, once the discard is compared.
        self.box_shown = QtWidgets.QWidget()
        shown = QtWidgets.QHBoxLayout(self.box_shown)
        shown.setContentsMargins(0, 0, 0, 0)
        shown.addWidget(QtWidgets.QLabel("Plots show:"))
        self.radio_measured = QtWidgets.QRadioButton("measured")
        self.radio_every = QtWidgets.QRadioButton("all clusters")
        self.radio_discard = QtWidgets.QRadioButton("discard applied")
        self.radio_discard.setToolTip(
            "Without the clusters both widefield images place inside the\n"
            "axon (axoplasm panel, section 5); they are drawn in red.")
        self.radio_measured.setChecked(True)
        self._shown_group = QtWidgets.QButtonGroup(self)
        for radio in (self.radio_measured, self.radio_every,
                      self.radio_discard):
            self._shown_group.addButton(radio)
            shown.addWidget(radio)
            radio.toggled.connect(
                lambda on: self._draw_plots() if on else None)
        shown.addStretch(1)
        self.box_shown.setVisible(False)
        lay.addWidget(self.box_shown)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Parameter", "Measured", "All clusters", "Discard applied",
             "Gazal 2026", "Note"])
        for col in (_COL_NAME, _COL_MEASURED, _COL_EVERY, _COL_DISCARD):
            self.table.horizontalHeader().setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeaderItem(_COL_EVERY).setToolTip(
            "Every kept cluster, with the contour refined by 2-opt from\n"
            "every start (the shortest tour). Shown only when the measured\n"
            "analysis used one start, which can settle on a longer tour --\n"
            "by up to 7.1 % on the April axons -- and would blur the\n"
            "comparison with the next column.")
        for col in (_COL_EVERY, _COL_DISCARD):
            self.table.setColumnHidden(col, True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, stretch=3)

        # One table per set of numbers: the analyses of one axon must not
        # be pooled in one statistic.
        self.box_export = QtWidgets.QWidget()
        export = QtWidgets.QHBoxLayout(self.box_export)
        export.setContentsMargins(0, 0, 0, 0)
        export.addWidget(QtWidgets.QLabel("Export for the statistics:"))
        self.btn_export_every = QtWidgets.QPushButton("All clusters")
        self.btn_export_every.setToolTip(
            "The 'All clusters' column (2-opt from every start), to a table\n"
            "of its own.")
        self.btn_export_every.clicked.connect(self._on_export_every)
        self.btn_export_discard = QtWidgets.QPushButton("Discard applied")
        self.btn_export_discard.setToolTip(
            "The 'Discard applied' column, to a table of its own.")
        self.btn_export_discard.clicked.connect(self._on_export_discard)
        export.addWidget(self.btn_export_every)
        export.addWidget(self.btn_export_discard)
        export.addStretch(1)
        self.box_export.setVisible(False)
        lay.addWidget(self.box_export)

        lay.addWidget(QtWidgets.QLabel("Warnings"))
        self.list_warnings = QtWidgets.QListWidget()
        self.list_warnings.setWordWrap(True)
        lay.addWidget(self.list_warnings, stretch=2)

        return w

    def _build_plots(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        grid.setContentsMargins(0, 0, 0, 0)

        self.plot_contour = pg.PlotWidget()
        style_dark(self.plot_contour)
        set_title(self.plot_contour,
                  "Clusters, reconstructed perimeter and its centre (+)")
        self.plot_contour.setAspectLocked(True)
        self.plot_contour.setLabels(bottom="x [nm]", left="y [nm]")
        grid.addWidget(self.plot_contour, 0, 0, 2, 1)

        self.plot_z = pg.PlotWidget()
        style_dark(self.plot_z)
        set_title(self.plot_z, "Axial (z) distribution, GMM fit and slab")
        self.plot_z.setLabels(bottom="z [nm]", left="density")
        grid.addWidget(self.plot_z, 0, 1)

        self.plot_area = pg.PlotWidget()
        style_dark(self.plot_area)
        set_title(self.plot_area, "Cluster area")
        self.plot_area.setLabels(bottom="area [nm^2]", left="count")
        grid.addWidget(self.plot_area, 1, 1)

        self.plot_nn = pg.PlotWidget()
        style_dark(self.plot_nn)
        set_title(self.plot_nn, "1NN distance between cluster centres")
        self.plot_nn.setLabels(bottom="distance [nm]", left="count")
        grid.addWidget(self.plot_nn, 2, 0)

        self.plot_cdf = pg.PlotWidget()
        style_dark(self.plot_cdf)
        set_title(self.plot_cdf, "1NN CDF: observed vs randomized")
        self.plot_cdf.setLabels(bottom="distance [nm]", left="cumulative")
        # The legend takes its text colour from the application's global
        # foreground, which is black: on the dark background it would be an
        # empty box.
        self.plot_cdf.addLegend(offset=(-10, 10), labelTextColor=AXIS_FG)
        grid.addWidget(self.plot_cdf, 2, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 2)
        grid.setRowStretch(2, 2)
        return w

    # ------------------------------------------------------------------
    # refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Redraw everything from ``self.analysis``, and the analysis
        without the discarded clusters when there is one for it."""
        self.comparison = self._ask_discard()
        self._sync_controls()
        self._sync_discard_widgets()
        self._fill_provenance()
        self._fill_table()
        self._fill_warnings()
        self._draw_plots()

    def _draw_plots(self) -> None:
        self._draw_contour()
        self._draw_z()
        self._draw_area()
        self._draw_nn()
        self._draw_cdf()

    def _ask_discard(self) -> Optional[DiscardComparison]:
        if self.discard_callback is None:
            return None
        found = self.discard_callback(self.analysis)
        # Only analyses of these same clusters.
        if found is None or any(
                a.labels is not self.analysis.labels
                for a in (found.all_clusters, found.discard_applied)):
            return None
        return found

    def _shown(self) -> AxonAnalysis:
        """The analysis the plots show."""
        if self.comparison is not None:
            if self.radio_every.isChecked():
                return self.comparison.all_clusters
            if self.radio_discard.isChecked():
                return self.comparison.discard_applied
        return self.analysis

    def _every_column(self) -> bool:
        """Whether 'All clusters' says something the measured column does
        not: only when the measured contour came from one 2-opt start."""
        return (self.comparison is not None
                and self.comparison.all_clusters is not self.analysis)

    def _sync_discard_widgets(self) -> None:
        compared = self.comparison is not None
        every = self._every_column()
        self.box_shown.setVisible(compared)
        self.box_export.setVisible(compared)
        self.radio_every.setVisible(every)
        self.btn_export_every.setVisible(every)
        if not every and self.radio_every.isChecked():
            self.radio_measured.setChecked(True)
        self.table.setColumnHidden(_COL_EVERY, not every)
        self.table.setColumnHidden(_COL_DISCARD, not compared)
        one_start = self.analysis.contour_2opt == "one start"
        self.table.horizontalHeaderItem(_COL_MEASURED).setToolTip(
            "As the MPS analysis measures every axon: the contour refined\n"
            + ("by 2-opt from one start, as before 2026-09-19."
               if one_start else
               "by 2-opt from every start, keeping the shortest tour."))
        self.table.horizontalHeaderItem(_COL_DISCARD).setToolTip(
            "Every parameter again without the clusters the axoplasm panel\n"
            "discarded, the contour also from every start: it differs from\n"
            + ("'All clusters'" if every else "'Measured'")
            + " only by those clusters. Values in bold changed.")
        if compared and not self._widened:
            # Before the window is first laid out the splitter has no size
            # yet; its width will be the window's.
            total = max(sum(self.splitter.sizes()), self.width())
            self.splitter.setSizes([total // 2, total - total // 2])
            self._widened = True

    def _sync_controls(self) -> None:
        """Push the current analysis' parameters into the widgets without
        triggering a re-run (signals are blocked while setting)."""
        a = self.analysis
        for wdg in (self.combo_peak, self.spin_half, self.spin_eps,
                    self.spin_min, self.spin_dbcv, self.spin_maha):
            wdg.blockSignals(True)

        self.combo_peak.clear()
        means = a.z_result.means_nm
        weights = a.z_result.weights
        centre = (a.slab_zmin_nm + a.slab_zmax_nm) / 2.0
        best_i, best_d = 0, float("inf")
        for i, (m, wgt) in enumerate(zip(means, weights)):
            self.combo_peak.addItem(
                f"z = {m:8.1f} nm   (weight {wgt:.2f})", float(m))
            d = abs(float(m) - centre)
            if d < best_d:
                best_i, best_d = i, d
        # The automatic centre is the density peak, which need not coincide
        # with any component mean; expose it as its own entry so the user
        # can always get back to it.
        self.combo_peak.addItem(
            f"z = {centre:8.1f} nm   (current slab centre)", float(centre))
        self.combo_peak.setCurrentIndex(
            self.combo_peak.count() - 1 if best_d > 1e-6 else best_i)

        self.spin_half.setValue(a.slab_half_width_nm)
        self.spin_eps.setValue(a.eps_nm)
        self.spin_min.setValue(int(a.min_samples))
        # Read back the threshold that actually produced this analysis,
        # not Qt's own 0.0 spinbox default. DBCV is OFF by default (-1.0):
        # if this were left at whatever the widget happens to show, editing
        # ANY other control (eps, slab width, ...) would silently resend
        # 0.0 as dbcv_threshold and re-enable a criterion measured to
        # remove the largest, most biologically relevant clusters in an
        # axon (log-area vs. DBCV score correlation -0.78 to -0.79).
        self.spin_dbcv.setValue(a.dbcv_threshold)
        if a.occupancy is not None:
            self.spin_maha.setValue(a.occupancy.mahalanobis_threshold)

        for wdg in (self.combo_peak, self.spin_half, self.spin_eps,
                    self.spin_min, self.spin_dbcv, self.spin_maha):
            wdg.blockSignals(False)

        # (Re)connect after the initial population so the first fill does
        # not trigger a re-run.
        if self.rerun_callback is not None:
            try:
                self.combo_peak.currentIndexChanged.disconnect()
                self.spin_half.editingFinished.disconnect()
                self.spin_eps.editingFinished.disconnect()
                self.spin_min.editingFinished.disconnect()
                self.spin_dbcv.editingFinished.disconnect()
                self.spin_maha.editingFinished.disconnect()
            except TypeError:
                pass
            self.combo_peak.currentIndexChanged.connect(self._on_param_changed)
            self.spin_half.editingFinished.connect(self._on_param_changed)
            self.spin_eps.editingFinished.connect(self._on_param_changed)
            self.spin_min.editingFinished.connect(self._on_param_changed)
            self.spin_dbcv.editingFinished.connect(self._on_param_changed)
            self.spin_maha.editingFinished.connect(self._on_param_changed)
            try:
                self.chk_random.stateChanged.disconnect()
            except TypeError:
                pass
            self.chk_random.stateChanged.connect(self._on_param_changed)

    def _fill_provenance(self) -> None:
        a = self.analysis
        px = "unknown" if a.pixel_size_nm is None else f"{a.pixel_size_nm:g} nm"
        src = {"yaml": "from Picasso YAML",
               "hdf5": "from the metadata inside the HDF5",
               "yaml_scan": "from Picasso YAML",
               "override": "given explicitly",
               "manual": "entered manually",
               "unknown": "UNKNOWN"}.get(a.pixel_size_source, a.pixel_size_source)
        colour = ("#333333" if a.pixel_size_source in ("yaml", "hdf5", "yaml_scan")
                  else _C_ORANGE)
        self.lbl_provenance.setText(
            f"<b>{os.path.basename(a.source_name) or '(unnamed ROI)'}</b><br>"
            f"Pixel size: <span style='color:{colour}'><b>{px}</b> ({src})</span>"
            f" &nbsp;|&nbsp; eps {a.eps_nm:g} nm, min samples {a.min_samples}"
            f" &nbsp;|&nbsp; slab {a.slab_zmin_nm:,.0f} .. {a.slab_zmax_nm:,.0f} nm"
        )

    def _fill_table(self) -> None:
        rows = self.analysis.summary_rows()
        # The same rows, in the same order, for the other two analyses:
        # summary_rows does not depend on the values.
        c = self.comparison
        every = (c.all_clusters.summary_rows() if c is not None
                 else [None] * len(rows))
        applied = (c.discard_applied.summary_rows() if c is not None
                   else [None] * len(rows))
        self.table.setRowCount(len(rows))
        for r, ((name, measured, paper, note), ev, ap) in enumerate(
                zip(rows, every, applied)):
            cells = {_COL_NAME: name, _COL_MEASURED: measured,
                     _COL_EVERY: "" if ev is None else ev[1],
                     _COL_DISCARD: "" if ap is None else ap[1],
                     _COL_PAPER: paper, _COL_NOTE: note}
            for col, text in cells.items():
                item = QtWidgets.QTableWidgetItem(str(text))
                if col in (_COL_MEASURED, _COL_EVERY, _COL_DISCARD) and \
                        "not implemented" in str(text):
                    item.setForeground(QtGui.QBrush(QtGui.QColor(_C_GREY)))
                    item.setFont(self._italic())
                own = {_COL_EVERY: ev, _COL_DISCARD: ap}.get(col)
                if own is not None:
                    # Its own note, which can differ (the points occupancy
                    # was sampled on, the clusters left out).
                    item.setToolTip(str(own[3]) or str(own[1]))
                if col == _COL_DISCARD and ap is not None and ev is not None \
                        and ap[1] != ev[1]:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(r, col, item)

    @staticmethod
    def _italic() -> QtGui.QFont:
        f = QtGui.QFont()
        f.setItalic(True)
        return f

    def _fill_warnings(self) -> None:
        self.list_warnings.clear()
        messages: List[str] = list(self.analysis.warnings)
        if self.comparison is not None:
            # Their own warnings, once each: those shared with an analysis
            # listed before are not repeated.
            seen = set(self.analysis.warnings)
            for prefix, other in (
                    ("All clusters, every 2-opt start: ",
                     self.comparison.all_clusters),
                    ("With the discard: ", self.comparison.discard_applied)):
                messages += [prefix + w for w in other.warnings
                             if w not in seen]
                seen.update(other.warnings)
        for w in messages:
            item = QtWidgets.QListWidgetItem(w)
            item.setForeground(QtGui.QBrush(QtGui.QColor(_C_ORANGE)))
            item.setToolTip(w)
            self.list_warnings.addItem(item)
        if not messages:
            self.list_warnings.addItem(
                QtWidgets.QListWidgetItem("No warnings."))

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------

    def _draw_contour(self) -> None:
        a = self._shown()
        self.plot_contour.clear()
        if a.x_slab.size == 0:
            return

        bad = a.bad_report.bad_labels
        noise = a.labels == -1
        if np.any(noise):
            self.plot_contour.addItem(pg.ScatterPlotItem(
                a.x_slab[noise], a.y_slab[noise], size=2, pen=None,
                brush=pg.mkBrush(200, 200, 200, 120), name="noise"))

        bad_mask = np.isin(a.labels, list(bad)) if bad else np.zeros_like(noise)
        if np.any(bad_mask):
            self.plot_contour.addItem(pg.ScatterPlotItem(
                a.x_slab[bad_mask], a.y_slab[bad_mask], size=3, pen=None,
                brush=pg.mkBrush(213, 94, 0, 90)))

        gone = (np.isin(a.labels, list(a.discarded_labels))
                if a.discarded_labels else np.zeros_like(noise))
        if np.any(gone):
            self.plot_contour.addItem(pg.ScatterPlotItem(
                a.x_slab[gone], a.y_slab[gone], size=3, pen=None,
                brush=pg.mkBrush(_C_DISCARDED)))

        good_mask = (~noise) & (~bad_mask) & (~gone)
        if np.any(good_mask):
            self.plot_contour.addItem(pg.ScatterPlotItem(
                a.x_slab[good_mask], a.y_slab[good_mask], size=3, pen=None,
                brush=pg.mkBrush(0, 114, 178, 110)))

        if a.discard_applied and self.comparison is not None:
            # The contour with every cluster, and its centre, for what the
            # discard changed.
            every = self.comparison.all_clusters.perimeter
            if every is not None:
                closed = np.vstack([every.contour, every.contour[:1]])
                self.plot_contour.addItem(pg.PlotDataItem(
                    closed[:, 0], closed[:, 1],
                    pen=pg.mkPen(_C_GREY, width=1,
                                 style=QtCore.Qt.PenStyle.DashLine)))
                if every.centre is not None:
                    self.plot_contour.addItem(pg.ScatterPlotItem(
                        [every.centre.x_nm], [every.centre.y_nm], size=14,
                        symbol="+", pen=pg.mkPen(_C_GREY),
                        brush=pg.mkBrush(_C_GREY)))
            labels = good_cluster_labels(self.analysis.labels,
                                         self.analysis.bad_report.bad_labels)
            out = np.isin(labels, list(a.discarded_labels))
            if np.any(out):
                c = np.asarray(self.analysis.centroids)[out]
                self.plot_contour.addItem(pg.ScatterPlotItem(
                    c[:, 0], c[:, 1], size=9, pen=pg.mkPen(_C_DARK_OUTLINE),
                    brush=pg.mkBrush(_C_DISCARDED)))

        if a.perimeter is not None:
            c = a.perimeter.contour
            closed = np.vstack([c, c[:1]])
            self.plot_contour.addItem(pg.PlotDataItem(
                closed[:, 0], closed[:, 1],
                pen=pg.mkPen(_C_GREY, width=1)))

        # Occupied stretches drawn on top of the contour: this is what makes
        # the occupancy percentage legible -- the paper's central claim is
        # that most of the perimeter carries no spectrin, and that is only
        # convincing when you can see the gaps.
        if a.occupancy is not None:
            pts = a.occupancy.perimeter_points
            occ = a.occupancy.occupied_mask
            if np.any(occ):
                # Split into contiguous runs so each occupied stretch is one
                # polyline; a single scatter of 10,000 points would hide the
                # segment structure and is far slower to draw.
                idx = np.flatnonzero(occ)
                breaks = np.flatnonzero(np.diff(idx) > 1)
                starts = np.concatenate([[0], breaks + 1])
                ends = np.concatenate([breaks, [len(idx) - 1]])
                for s, e in zip(starts, ends):
                    run = pts[idx[s]:idx[e] + 1]
                    if len(run) >= 2:
                        self.plot_contour.addItem(pg.PlotDataItem(
                            run[:, 0], run[:, 1],
                            pen=pg.mkPen(_C_GREEN, width=4)))

        if a.centroids.size:
            self.plot_contour.addItem(pg.ScatterPlotItem(
                a.centroids[:, 0], a.centroids[:, 1], size=7,
                pen=pg.mkPen(_C_DARK_OUTLINE), brush=pg.mkBrush(_C_GREEN)))

        # The centre of the contour drawn: its area centroid.
        if a.centre is not None:
            self.plot_contour.addItem(pg.ScatterPlotItem(
                [a.centre.x_nm], [a.centre.y_nm], size=18, symbol="+",
                pen=pg.mkPen("#000000"), brush=pg.mkBrush(_C_CENTRE)))

    def _draw_z(self) -> None:
        a = self._shown()
        self.plot_z.clear()
        # Full ROI z-distribution is not carried in the analysis object;
        # the slab's own z values plus the fitted components still convey
        # where the slab sits relative to the peaks.
        zr = a.z_result
        if a.z_slab.size:
            counts, edges = np.histogram(a.z_slab, bins=60, density=True)
            centres = (edges[:-1] + edges[1:]) / 2
            width = float(np.mean(np.diff(edges)))
            self.plot_z.addItem(pg.BarGraphItem(
                x=centres, height=counts, width=width,
                brush=pg.mkBrush(0, 114, 178, 90), pen=None))

        lo, hi = a.slab_zmin_nm, a.slab_zmax_nm
        span = max(hi - lo, 1.0)
        grid = np.linspace(lo - 2 * span, hi + 2 * span, 1024)
        for m, wgt, s in zip(zr.means_nm, zr.weights, zr.sigmas_nm):
            if s <= 0:
                continue
            dens = wgt * np.exp(-0.5 * ((grid - m) / s) ** 2) / (
                s * np.sqrt(2 * np.pi))
            self.plot_z.addItem(pg.PlotDataItem(
                grid, dens, pen=pg.mkPen(_C_ORANGE, width=2,
                                         style=QtCore.Qt.DashLine)))
            self.plot_z.addItem(pg.InfiniteLine(
                pos=float(m), angle=90,
                pen=pg.mkPen(_C_ORANGE, width=1, style=QtCore.Qt.DotLine)))

        region = pg.LinearRegionItem(values=(lo, hi), movable=False)
        region.setBrush(pg.mkBrush(0, 158, 115, 45))
        region.setZValue(-10)
        self.plot_z.addItem(region)

    def _draw_area(self) -> None:
        a = self._shown()
        self.plot_area.clear()
        if a.areas is None or a.areas.areas_nm2.size == 0:
            return
        vals = a.areas.areas_nm2
        counts, edges = np.histogram(vals, bins=30)
        centres = (edges[:-1] + edges[1:]) / 2
        width = float(np.mean(np.diff(edges)))
        self.plot_area.addItem(pg.BarGraphItem(
            x=centres, height=counts, width=width,
            brush=pg.mkBrush(0, 114, 178, 160), pen=None))
        med = a.areas.median_area_nm2
        if med is not None:
            self.plot_area.addItem(pg.InfiniteLine(
                pos=med, angle=90,
                pen=pg.mkPen(_C_BLUE, width=2),
                label=f"median {med:,.0f}",
                labelOpts={"position": 0.9, "color": _C_BLUE}))
        self.plot_area.addItem(pg.InfiniteLine(
            pos=1965.0, angle=90,
            pen=pg.mkPen(_C_ORANGE, width=2, style=QtCore.Qt.DashLine),
            label="paper 1,965",
            labelOpts={"position": 0.75, "color": _C_ORANGE}))

    def _draw_nn(self) -> None:
        a = self._shown()
        self.plot_nn.clear()
        if a.nn is None or a.nn.first_nn_nm.size == 0:
            return
        vals = a.nn.first_nn_nm
        counts, edges = np.histogram(vals, bins=30)
        centres = (edges[:-1] + edges[1:]) / 2
        width = float(np.mean(np.diff(edges)))
        self.plot_nn.addItem(pg.BarGraphItem(
            x=centres, height=counts, width=width,
            brush=pg.mkBrush(0, 158, 115, 160), pen=None))
        med = a.nn.median_1nn_nm
        if med is not None:
            self.plot_nn.addItem(pg.InfiniteLine(
                pos=med, angle=90, pen=pg.mkPen(_C_GREEN, width=2),
                label=f"median {med:,.0f} nm",
                labelOpts={"position": 0.9, "color": _C_GREEN}))
        self.plot_nn.addItem(pg.InfiniteLine(
            pos=260.0, angle=90,
            pen=pg.mkPen(_C_ORANGE, width=2, style=QtCore.Qt.DashLine),
            label="paper 260 nm",
            labelOpts={"position": 0.75, "color": _C_ORANGE}))

    def _draw_cdf(self) -> None:
        """Observed vs randomized 1NN cumulative distributions (Fig. 4E)."""
        a = self._shown()
        self.plot_cdf.clear()
        r = a.randomization
        if r is None:
            return

        def cdf(v):
            v = np.sort(np.asarray(v, dtype=float))
            return v, np.arange(1, v.size + 1) / v.size

        xe, ye = cdf(r.experimental_1nn_nm)
        xr, yr = cdf(r.randomized_1nn_nm)
        self.plot_cdf.addItem(pg.PlotDataItem(
            xr, yr, pen=pg.mkPen(_C_GREY, width=2), name="randomized"))
        self.plot_cdf.addItem(pg.PlotDataItem(
            xe, ye, pen=pg.mkPen(_C_BLUE, width=2), name="observed"))

        if r.cdf_crossing is not None:
            line = pg.InfiniteLine(
                pos=r.cdf_crossing, angle=0,
                pen=pg.mkPen(_C_ORANGE, width=1, style=QtCore.Qt.DashLine),
                label=f"crossing {r.cdf_crossing:.2f}",
                labelOpts={"position": 0.05, "color": _C_ORANGE})
            self.plot_cdf.addItem(line)

    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------

    def _on_param_changed(self) -> None:
        if self.rerun_callback is None:
            return
        peak = self.combo_peak.currentData()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.analysis = self.rerun_callback(
                main_peak_override_nm=(None if peak is None else float(peak)),
                slab_half_width_nm=float(self.spin_half.value()),
                eps_nm=float(self.spin_eps.value()),
                min_samples=int(self.spin_min.value()),
                dbcv_threshold=float(self.spin_dbcv.value()),
                mahalanobis_threshold=float(self.spin_maha.value()),
                run_randomization=bool(self.chk_random.isChecked()),
            )
        except Exception as exc:                      # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, "Analysis failed",
                f"Could not re-run the analysis with these parameters:\n\n{exc}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.refresh()

    def _on_reset(self) -> None:
        from tools.mps_settings import (
            DEFAULT_DBCV_THRESHOLD, DEFAULT_EPS_NM, DEFAULT_MIN_SAMPLES,
            DEFAULT_SLAB_HALF_WIDTH_NM,
        )
        for w in (self.spin_half, self.spin_eps, self.spin_min,
                  self.spin_dbcv, self.spin_maha):
            w.blockSignals(True)
        self.spin_half.setValue(DEFAULT_SLAB_HALF_WIDTH_NM)
        self.spin_eps.setValue(DEFAULT_EPS_NM)
        self.spin_min.setValue(DEFAULT_MIN_SAMPLES)
        self.spin_dbcv.setValue(DEFAULT_DBCV_THRESHOLD)
        self.spin_maha.setValue(3.0)
        self.combo_peak.blockSignals(True)
        self.combo_peak.setCurrentIndex(max(0, self.combo_peak.count() - 1))
        self.combo_peak.blockSignals(False)
        for w in (self.spin_half, self.spin_eps, self.spin_min,
                  self.spin_dbcv, self.spin_maha):
            w.blockSignals(False)
        self._on_param_changed()

    def _on_export(self) -> None:
        """Append this axon's parameters to a CSV, one row per axon."""
        self._export(self.analysis, "")

    def _on_export_every(self) -> None:
        if self.comparison is not None:
            self._export(self.comparison.all_clusters, "_every_start")

    def _on_export_discard(self) -> None:
        if self.comparison is not None:
            self._export(self.comparison.discard_applied, "_discard")

    def _export(self, analysis: AxonAnalysis, suffix: str) -> None:
        base = os.path.splitext(os.path.basename(
            analysis.source_name))[0] or "axon"
        default = f"{base}_mps_parameters{suffix}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export MPS parameters", default, "CSV Files (*.csv)")
        if not path:
            return

        record = analysis.export_dict()
        try:
            # The measured analysis and the one with the discard each go to
            # a table of their own, and so does the one with every 2-opt
            # start when the measured analysis used one start.
            refuse_other_analysis(path, record, ANALYSIS_COLUMNS)
            # A batch of axons accumulates into one table. A file with other
            # columns is refused rather than overwritten.
            append = append_rows(path, [record])
        except (OSError, ValueError, csv.Error) as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not write {path}:\n\n{exc}")
            return

        QtWidgets.QMessageBox.information(
            self, "Exported",
            f"{'Appended to' if append else 'Wrote'} {path}")
