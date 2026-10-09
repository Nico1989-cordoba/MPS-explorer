# -*- coding: utf-8 -*-
"""
Results window for the automatic per-axon MPS analysis: the axon's page.

Shows every parameter computed by ``tools.mps_analysis.analyze_axon`` next
to the corresponding value from Gazal et al. (2026), together with the
plots that make the numbers interpretable. Since UI stage 2 (design 3-5)
the plots are:

  * the Axon map (``tools.mps_axon_map``), at the top: the analysis'
    localizations, cluster centres, contour, occupied stretches and centre,
    and - through its views - the Axoplasm panel's image and classes and
    the Rings window's segments, every layer in a group named after the
    computation it comes from;
  * under it, tabs: Axial (the single axial view, ``tools.mps_axial_view``),
    Nearest neighbours (``tools.mps_nn_panel``), Cluster area, Scatter off
    the outline.

Design principle requested by the user: the analysis runs automatically and
instantly, but *every* automatic choice stays editable, because reading the
histograms is where expert judgement enters. Editable here:

  * the axial peak the 180 nm slab is centred on (a selector listing every
    peak the GMM found -- this is what the "ambiguous main peak" warning
    refers to)
  * the slab half-width
  * DBSCAN eps and min_samples
  * the Mahalanobis threshold of the occupancy and the randomization

Changing any of them re-runs the analysis on the same localizations and
redraws everything.

Once the axoplasm panel has found the clusters not anchored to the
membrane, another column appears: every parameter again with the discard
applied. Both it and the measured column build the contour by 2-opt from
every start, so they differ only by the clusters left out. (An analysis
refined from one start, as this program did before 2026-09-19, gets a
third column in between: all the clusters, every start.) The plots show any of them,
and each has its own export, so the user decides which numbers go to the
statistics. Every exported figure carries in its own title which analysis
it shows.

The window can be opened before any analysis (it then draws the current
selection only), and it says when what it shows is the analysis of a
previous selection (the stale banner, ``tools.mps_stale_banner``).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_axon_map_layers as L
from tools import mps_param_registry as reg
from tools.mps_analysis import AxonAnalysis, DiscardComparison
from tools.mps_axial_view import AxialView
from tools.mps_axon_map import AxonMap
from tools.mps_nn_panel import NearestNeighboursPanel
from tools.mps_plot_style import (
    AXIS_FG_LIGHT, neutral, rgba, role, set_title, style_dark, style_light, verdict)
from tools.mps_settings import (
    DEFAULT_DBCV_THRESHOLD, DEFAULT_MAHALANOBIS_THRESHOLD)
from tools.mps_stale_banner import (
    StaleBanner, analysis_stale_lines, showing_current_lines)


# Every colour comes from tools.mps_plot_style, by the role it plays.
# Which roles may share a plot, and which pairs have to be told apart by
# a symbol as well, is checked by validate_plot_colours.py.
#
# The one colour that depends on the background -- a line, the outline of
# a marker -- is asked for through self._neutral(), because this window
# draws on black on screen and on white when it exports a figure for a
# journal.
# Text on the window's own white chrome (tables, the warning list), where
# the neutral for a dark plot would be invisible.
_C_TEXT_WARN = verdict("warn", dark=False)
# A cell the program cannot fill in: grey enough to recede on white,
# dark enough to be read. The panel's prose grey would vanish there,
# which is why the palette carries one of each.
_C_TEXT_DIM = verdict("dim", dark=False)

# The radial profile plot's title; gains the measured scatter when there is
# one.
SCATTER_TITLE = "Scatter of the centres off a smooth outline"
AREA_TITLE = "Cluster area"
NO_ANALYSIS = "No MPS analysis of this selection yet"

# Columns of the parameter table.
(_COL_NAME, _COL_MEASURED, _COL_EVERY, _COL_DISCARD, _COL_PAPER,
 _COL_NOTE) = range(6)

# The plots "Export image..." lists, in this order (design 3.6).
PLOT_MAP = "Axon map"
PLOT_AXIAL = "Axial distribution"
PLOT_AREA = "Cluster area"
PLOT_NN = "Nearest neighbours"
PLOT_CDF = "1NN CDF"
PLOT_SCATTER = "Scatter off the outline"

# Tooltips of the results table (rev 3, 12.8): what each reference value is,
# and the two cells whose number is this program's heuristic or has no
# valid reading. The texts of the cells themselves are unchanged.
PAPER_HEADER_TIP = (
    "Values of Gazal et al. 2026 (preprint v1): another dataset and another pipeline; agreeing with them is not a "
    "validation criterion (P-R25). SCI-3 renames this column.")
KS_P_TIP = (
    "This p treats the K distances as independent and the 1,000 randomizations as a second sample; neither holds, "
    "and it rejects true nulls several times more often than 5 % (P-R23). SCI-3 replaces it with Monte Carlo "
    "p-values.")
CROSSING_TIP = ("This program's heuristic; SCI-3 reports a crossing only where the curves leave a null band.")
# The reference cell of each row, by the row's name (summary_rows), and the registry entry that says what it is.
_REFERENCE_KEYS: Dict[str, str] = {
    "Clusters per um": "reference.clusters_per_um",
    "Cluster area (median)": "reference.cluster_area_nm2",
    "Effective radius (median)": "reference.r_eff_nm",
    "1NN spacing (median)": "reference.nn1_median_nm",
    "Perimeter occupancy": "reference.occupancy_percent",
    "Randomization KS test": "reference.ks",
    "  CDF crossing": "reference.cdf_crossing",
}
_KS_ROW = "Randomization KS test"
_CROSSING_ROW = "  CDF crossing"

# How much of the plot column the map takes when the window opens (the rest
# is the tabs under it): the map keeps at least 55 % of a 1366 x 728 window.
MAP_SHARE = 0.68


@dataclass
class SelectionView:
    """What the main window knows about the current selection and the analysis, for display only (Appendix B).
    Nothing here is ever passed back to an analysis."""
    z_roi: Optional[NDArray[np.float64]] = None      # the current ROI's z before the cut
    n_locs: Optional[int] = None
    roi_words: str = ""                              # "ROI circle ..." | "whole field of view, no ROI drawn"
    current_words: str = ""                          # the current selection, for the banner
    cut: Optional[Tuple[float, float]] = None        # the main-window cut (Z min, Z max)
    cut_from_histogram_mode: bool = False
    fit: Any = None                                  # the prefill's ZPeriodicityResult (or one on demand)
    fit_reason: str = "the mixture could not be fitted"
    ch2_z: Optional[NDArray[np.float64]] = None      # channel 2 in the ROI before the cut, as loaded
    ch2_reason: str = "no channel 2 is loaded"
    analysis: Any = None                             # the main window's current analysis
    analysis_current: bool = False                   # it describes the current selection
    analysed_z: Optional[NDArray[np.float64]] = None  # z of the selection the analysis was given
    analysed_after_cut: bool = False                 # it was given the cut selection (fallback)
    analysed_words: str = ""                         # the selection then, for the banner
    analysed_roi_words: str = ""
    analysed_cut: Optional[Tuple[float, float]] = None
    discard_dropped: bool = False                    # the comparison was dropped by the panel's move
    discard_failed: str = ""                         # why the discard comparison could not be made


def _no_selection() -> Optional[SelectionView]:
    return None


def _no_axoplasm() -> Tuple[Any, str]:
    return None, "the Axoplasm panel is not open on this selection"


def _no_rings() -> Tuple[Any, str]:
    return None, "press Rings... to compute the segments"


def _no_params() -> Optional[Tuple[float, int, float, float]]:
    return None


def _no_name() -> str:
    return "distances"


@dataclass
class WindowLinks:
    """The providers and callbacks the main window gives this window (Appendix B). Every default is the standalone
    window's: no selection, no panel, no rings."""
    selection: Callable[[], Optional[SelectionView]] = field(default=_no_selection)
    axoplasm: Callable[[], Tuple[Any, str]] = field(default=_no_axoplasm)
    rings: Callable[[], Tuple[Any, str]] = field(default=_no_rings)
    run_analysis: Optional[Callable[[], Any]] = None
    params: Callable[[], Optional[Tuple[float, int, float, float]]] = field(default=_no_params)
    root_name: Callable[[], str] = field(default=_no_name)
    nn_bins: int = L.NN_DEFAULT_BINS
    nn_range: Tuple[float, float] = (0.0, 800.0)


def beside_position(anchor: QtCore.QRect, size: QtCore.QSize,
                    available: QtCore.QRect) -> Optional[QtCore.QPoint]:
    """Where a window of frame ``size`` goes beside the frame ``anchor`` (3.1): to its right when it fits in the
    ``available`` geometry, else to its left, else None (only raised)."""
    top = max(available.top(), min(anchor.top(), available.bottom() - size.height() + 1))
    if size.height() > available.height():
        return None
    right = anchor.right() + 1
    if right + size.width() - 1 <= available.right():
        return QtCore.QPoint(right, top)
    left = anchor.left() - size.width()
    if left >= available.left():
        return QtCore.QPoint(left, top)
    return None


class MPSResultsWindow(QtWidgets.QMainWindow):
    """Panel showing the per-axon parameters, plots and warnings."""

    def __init__(
        self,
        analysis: Optional[AxonAnalysis],
        rerun_callback: Optional[Callable[..., AxonAnalysis]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        rings_callback: Optional[Callable[[], Any]] = None,
        discard_callback: Optional[
            Callable[[AxonAnalysis], Optional[DiscardComparison]]] = None,
        export_callback: Optional[Callable[[], Any]] = None,
        links: Optional[WindowLinks] = None,
    ):
        """
        Parameters
        ----------
        analysis : the initial result to display, or None to open the
            window on the current selection before any analysis.
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
        export_callback : writes this axon's tables. It is the main
            window's single export, which sees the analysis, the discard
            and the axoplasm panel at once; this window only asks for it.
            None disables the button.
        links : what the main window tells this window about the current
            selection, the Axoplasm panel and the rings (``WindowLinks``);
            None for a window on its own.
        """
        super().__init__(parent)
        self.analysis = analysis
        self.rerun_callback = rerun_callback
        self.rings_callback = rings_callback
        self.discard_callback = discard_callback
        self.export_callback = export_callback
        self.links = links if links is not None else WindowLinks()
        self.comparison: Optional[DiscardComparison] = None
        # Black on screen; white while a figure is being written.
        self.dark = True
        # The selection as the main window last described it, whether the
        # analysis is of a previous one, and whether the user asked to see
        # the current selection instead ("Show the current selection").
        self._sel: Optional[SelectionView] = None
        self._stale = False
        self._show_current = False
        self._last_analysis: Any = analysis
        # Windows this one was placed beside once already (3.1).
        self._placed_beside: set = set()

        self.setWindowTitle("MPS analysis - per-axon parameters")
        self.resize(1250, 860)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        root.addWidget(self._build_controls())
        self.banner = StaleBanner(object_name="mps_stale_banner")
        self.banner.show_current_toggled.connect(self._on_show_current)
        self.banner.run_requested.connect(self._on_run_requested)
        root.addWidget(self.banner)

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

        # A "DBCV thr." spin box stood here until 2026-09-20. Measured
        # on unpublished pilot axons, the score correlates strongly and
        # negatively with log10(cluster area), and in many axons the
        # LARGEST cluster is among the lowest-scoring ones: every
        # threshold above off removes
        # the big clusters first. There is no setting of it that curates
        # without doing that, so it is gone rather than defaulted. The
        # analysis still takes the parameter and the export still records
        # it, so older tables stay readable; what curates is the
        # edge-touching criterion.

        lay.addWidget(QtWidgets.QLabel("Mahalanobis:"))
        self.spin_maha = QtWidgets.QDoubleSpinBox()
        self.spin_maha.setRange(0.1, 10.0)
        self.spin_maha.setDecimals(1)
        self.spin_maha.setSingleStep(0.5)
        # Qt starts a spin box at its minimum, which here is 0.1: an
        # analysis re-run from this window while the box had never been
        # synced measured occupancy with that threshold (a many-fold
        # smaller occupancy), and no exported column said so.
        self.spin_maha.setValue(DEFAULT_MAHALANOBIS_THRESHOLD)
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
        self.btn_reset.setToolTip(
            "Put the parameters above back to the values Gazal et al. "
            "(2026) report, and run the analysis again.\n\n"
            "It touches the parameters only. The axial slab stays where "
            "it is and the randomization stays as it is set.")
        self.btn_reset.clicked.connect(self._on_reset)
        lay.addWidget(self.btn_reset)

        self.btn_export = QtWidgets.QPushButton("Export axon...")
        self.btn_export.setToolTip(
            "Write this axon: one row with the measured parameters and the\n"
            "ones with the discard applied, plus the per-cluster and\n"
            "per-localization tables if you ask for them. Everything is\n"
            "written from the state on screen, in one go.")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(self.export_callback is not None)
        lay.addWidget(self.btn_export)

        self.btn_image = QtWidgets.QPushButton("Export image...")
        self.btn_image.setToolTip(
            "Write one of these plots as a figure: at the width a journal\n"
            "asks for, at 300 to 1200 dpi or as an SVG of curves, and drawn\n"
            "on white rather than on the screen's black. Its title says\n"
            "which analysis, which localizations and which colouring it\n"
            "draws, and the message after writing it repeats the provenance.")
        self.btn_image.clicked.connect(self._on_export_image)
        lay.addWidget(self.btn_image)

        self.btn_contour = QtWidgets.QPushButton("Draw contour...")
        self.btn_contour.setToolTip(
            "Drag a path along the membrane and join the cluster centres in\n"
            "the order they fall along it, instead of the automatic 2-opt\n"
            "contour. For axons where the automatic contour runs through\n"
            "centres that are not on the membrane, or cuts a concavity.\n\n"
            "The path is kept for this axon: changing eps, min samples or\n"
            "the slab re-orders the new clusters along it, and the clusters\n"
            "the axoplasm panel keeps are joined along it too. It is written\n"
            "to the table, so the contour can be rebuilt from there.")
        self.btn_contour.clicked.connect(self._on_draw_contour)
        lay.addWidget(self.btn_contour)

        enabled = self.rerun_callback is not None
        for w in (self.combo_peak, self.spin_half, self.spin_eps,
                  self.spin_min, self.spin_maha,
                  self.chk_random, self.btn_reset, self.btn_contour):
            w.setEnabled(enabled)

        return box

    def _on_rings(self) -> None:
        if self.rings_callback is not None:
            self.rings_callback()

    def _build_left_column(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        # Which analysis the plots show, once the discard is compared.
        self.box_shown = QtWidgets.QWidget()
        shown = QtWidgets.QHBoxLayout(self.box_shown)
        shown.setContentsMargins(0, 0, 0, 0)
        self.label_shown = QtWidgets.QLabel("Draw the plots with:")
        font = self.label_shown.font()
        font.setBold(True)
        self.label_shown.setFont(font)
        shown.addWidget(self.label_shown)
        self.radio_measured = QtWidgets.QRadioButton("measured")
        self.radio_every = QtWidgets.QRadioButton("all clusters")
        self.radio_discard = QtWidgets.QRadioButton("discard applied")
        for radio in (self.radio_measured, self.radio_every,
                      self.radio_discard):
            radio.setToolTip(
                "Which clusters every plot is drawn from. The axon map, the "
                "areas, the nearest neighbours and the randomization all\n"
                "follow this, and each plot's own title says which one it "
                "is showing.\n\n"
                "'Discard applied' leaves out the clusters both widefield "
                "images place inside the axon (axoplasm panel,\n"
                "section 5); on the axon map they are vermillion diamonds, "
                "and the contour with every cluster stays as a dashed line.")
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
            "by a few percent on real axons -- and would blur the\n"
            "comparison with the next column.")
        self.table.horizontalHeaderItem(_COL_PAPER).setToolTip(PAPER_HEADER_TIP)
        for col in (_COL_EVERY, _COL_DISCARD):
            self.table.setColumnHidden(col, True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, stretch=3)

        # The two columns leave together, in one row. A table holding the
        # two analyses of one axon as two rows is how a statistic over it
        # counts that axon twice.
        self.box_export = QtWidgets.QWidget()
        export = QtWidgets.QHBoxLayout(self.box_export)
        export.setContentsMargins(0, 0, 0, 0)
        self.label_export = QtWidgets.QLabel(
            "Both columns go into one row, with and without the discard: "
            "'Export axon...' above.")
        self.label_export.setWordWrap(True)
        export.addWidget(self.label_export)
        export.addStretch(1)
        self.box_export.setVisible(False)
        lay.addWidget(self.box_export)

        lay.addWidget(QtWidgets.QLabel("Warnings"))
        self.list_warnings = QtWidgets.QListWidget()
        self.list_warnings.setWordWrap(True)
        lay.addWidget(self.list_warnings, stretch=2)

        return w

    def _build_plots(self) -> QtWidgets.QWidget:
        """The axon map above, the tabs under it (design 3.1, Q4)."""
        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.plot_splitter.setObjectName("plot_splitter")

        self.axon_map = AxonMap(dark=self.dark)
        # The map's first provenance line IS the provenance this window has
        # always shown (file, pixel size and its source, eps, min samples,
        # slab): one label, now above the map, where a figure is read.
        self.lbl_provenance = self.axon_map.lbl_line1
        self.axon_map.view_changed.connect(self._on_map_view)
        self.plot_splitter.addWidget(self.axon_map)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("plot_tabs")
        self.axial = AxialView(dark=self.dark)
        self.tabs.addTab(self.axial, "Axial")
        self.nn = NearestNeighboursPanel(
            dark=self.dark, bins=int(self.links.nn_bins), range_nm=self.links.nn_range,
            root_name=self.links.root_name, current=self._analysis_is_current)
        self.tabs.addTab(self.nn, "Nearest neighbours")

        self.plot_area = pg.PlotWidget()
        style_dark(self.plot_area)
        set_title(self.plot_area, AREA_TITLE)
        self.plot_area.setLabels(bottom="area [nm^2]", left="count")
        self.tabs.addTab(self.plot_area, "Cluster area")

        # Each centre's distance from the centres' mean against its
        # angle, with the smooth outline fitted to them and a band of the
        # scatter either side: how rough the ring is for its size, and
        # which centres sit far inside it.
        self.plot_scatter = pg.PlotWidget()
        style_dark(self.plot_scatter)
        set_title(self.plot_scatter, SCATTER_TITLE)
        self.plot_scatter.setLabels(
            bottom="angle about the centres' mean [deg]",
            left="distance from it [nm]")
        self.plot_scatter.setXRange(0.0, 360.0, padding=0.0)
        self.tabs.addTab(self.plot_scatter, "Scatter off the outline")
        self.plot_splitter.addWidget(self.tabs)
        self.plot_splitter.setStretchFactor(0, 2)
        self.plot_splitter.setStretchFactor(1, 1)
        total = 1000
        self.plot_splitter.setSizes([int(total * MAP_SHARE), total - int(total * MAP_SHARE)])
        return self.plot_splitter

    # ------------------------------------------------------------------
    # refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Redraw everything from ``self.analysis``, and the analysis
        without the discarded clusters when there is one for it."""
        if self.analysis is not self._last_analysis:
            # A new analysis is shown as itself.
            self._last_analysis = self.analysis
            self._show_current = False
        self.comparison = self._ask_discard()
        self._read_selection()
        self._sync_controls()
        self._sync_discard_widgets()
        self._fill_provenance()
        self._fill_table()
        self._fill_warnings()
        self._update_banner()
        self._draw_plots()

    def selection_changed(self) -> None:
        """The main window applied another selection (an ROI moved, an
        axial range was applied). Only what depends on the selection is
        redrawn: the banner, the map, the axial view. The analysis, the
        table and the comparison stay as they are."""
        self._read_selection()
        if self._show_current:
            self._fill_provenance()
            self._fill_table()
            self._fill_warnings()
        self._update_banner()
        self._draw_plots()

    def axoplasm_changed(self) -> None:
        """The Axoplasm panel changed its picture (threshold, smoothing,
        margin, shift, images): redraw the map."""
        self._draw_map()

    def rings_changed(self) -> None:
        """The rings window shows other segments: the map and the axial
        view redraw their segment groups."""
        self._draw_map()
        self._draw_axial()

    def show_view(self, view: str, beside: Optional[QtWidgets.QWidget] = None) -> None:
        """Bring the window up in a view of the map ("mps", "axoplasm",
        "segments"; the segments view also sets the axial view's "Every
        segment"), placed beside ``beside`` the first time (3.1)."""
        # Read the providers first: the segments or the panel that asked
        # for this view may be newer than the last redraw.
        self._read_selection()
        self._draw_map()
        self._draw_axial()
        self.axon_map.set_view(view)
        if view == "segments":
            self.axial.set_view("segments")
            self.tabs.setCurrentWidget(self.axial)
        self.show()
        if beside is not None:
            self.place_beside(beside)
        self.raise_()
        self.activateWindow()

    def place_beside(self, anchor: QtWidgets.QWidget) -> bool:
        """Move beside ``anchor`` the first time it is asked for that
        window, when the screen has room; False when it was not moved."""
        key = anchor.objectName() or type(anchor).__name__
        if key in self._placed_beside:
            return False
        self._placed_beside.add(key)
        screen = anchor.screen() if hasattr(anchor, "screen") else None
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return False
        frame = self.frameGeometry()
        where = beside_position(anchor.frameGeometry(), frame.size(), screen.availableGeometry())
        if where is None:
            return False
        self.move(where)
        return True

    def _draw_plots(self) -> None:
        self._draw_map()
        self._draw_axial()
        self._draw_area()
        self._draw_nn()
        self._draw_scatter()

    def _ask_discard(self) -> Optional[DiscardComparison]:
        if self.discard_callback is None or self.analysis is None:
            return None
        found = self.discard_callback(self.analysis)
        # Only analyses of these same clusters.
        if found is None or any(
                a.labels is not self.analysis.labels
                for a in (found.all_clusters, found.discard_applied)):
            return None
        return found

    def _read_selection(self) -> None:
        """What the main window says about the current selection, and
        whether the analysis shown is of a previous one."""
        sel = self.links.selection()
        self._sel = sel
        self._stale = (self.analysis is not None and sel is not None
                       and not (sel.analysis is self.analysis and sel.analysis_current))

    def _analysis_is_current(self) -> bool:
        """Whether the analysis describes the current selection (a window
        on its own: always)."""
        sel = self._sel
        if sel is None:
            return True
        return bool(sel.analysis is self.analysis and sel.analysis_current)

    def _display_analysis(self) -> Optional[AxonAnalysis]:
        """The measured analysis drawn, or None in the empty state."""
        if self.analysis is None or self._show_current:
            return None
        return self.analysis

    def _neutral(self) -> str:
        """The line and outline colour for the background in use."""
        return neutral(dark=self.dark)

    def _plots(self) -> "dict":
        """Every plot of this window, by the name the export shows."""
        return {PLOT_MAP: self.axon_map.plot,
                PLOT_AXIAL: self.axial.plot,
                PLOT_AREA: self.plot_area,
                PLOT_NN: self.nn.plot_nn,
                PLOT_CDF: self.nn.plot_cdf,
                PLOT_SCATTER: self.plot_scatter}

    def _tab_of(self, name: str) -> Optional[QtWidgets.QWidget]:
        """The tab a plot sits in (None: the map, always visible)."""
        return {PLOT_AXIAL: self.axial, PLOT_AREA: self.plot_area, PLOT_NN: self.nn,
                PLOT_CDF: self.nn, PLOT_SCATTER: self.plot_scatter}.get(name)

    def _apply_background(self) -> None:
        """Style every plot for the background in use, and redraw."""
        for plot in (self.plot_area, self.plot_scatter):
            (style_dark if self.dark else style_light)(plot)
        self.axon_map.set_dark(self.dark)
        self.axial.set_dark(self.dark)
        self.nn.set_dark(self.dark)
        self._draw_plots()

    def _export_message(self, name: str) -> str:
        """What the export says beside the file: the visible layers, the
        hidden count and the provenance lines (map, axial), or the title."""
        if name == PLOT_MAP:
            return self.axon_map.export_message()
        if name == PLOT_AXIAL:
            return self.axial.export_message()
        plot = self._plots().get(name)
        if plot is None:
            return ""
        title = plot.getPlotItem().titleLabel.text
        return "Title: " + " | ".join(_strip(str(title)).split("<br>"))

    def _on_export_image(self) -> None:
        """Write one plot as a figure."""
        from tools import figure_export
        from tools.figure_export_ui import ask

        plots = self._plots()
        suggested = figure_export.suggested_name(
            self.analysis.source_name if self.analysis is not None
            else self.links.root_name(), next(iter(plots)))
        # A white redraw is offered for every plot but the map while it
        # draws the widefield image: a photograph is not redrawn (3.6).
        white = {name for name in plots
                 if name != PLOT_MAP or self.axon_map.offers_white()}
        request = ask(list(plots), suggested, parent=self, offer_white=white)
        if request is None:
            return
        plot = plots.get(request.plot)
        if plot is None:
            return
        tab = self._tab_of(request.plot)
        if tab is not None and self.tabs.currentWidget() is not tab:
            # pyqtgraph exports a laid-out item: show its tab once.
            self.tabs.setCurrentWidget(tab)
            QtWidgets.QApplication.processEvents()
        was_dark = self.dark
        try:
            if request.white and was_dark:
                self.dark = False
                self._apply_background()
                QtWidgets.QApplication.processEvents()
            written = figure_export.write(plot.getPlotItem(), request)
            message = self._export_message(request.plot)
        except Exception as error:                        # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self, "Export failed",
                f"Could not write the image:\n\n{error}")
            return
        finally:
            if self.dark != was_dark:
                self.dark = was_dark
                self._apply_background()
        QtWidgets.QMessageBox.information(
            self, "Image written",
            f"{figure_export.describe(request)}\n\n{written}"
            + (f"\n\n{message}" if message else ""))

    def _shown(self) -> Optional[AxonAnalysis]:
        """The analysis the plots show (None in the empty state)."""
        a = self._display_analysis()
        if a is None:
            return None
        if self.comparison is not None:
            if self.radio_every.isChecked():
                return self.comparison.all_clusters
            if self.radio_discard.isChecked():
                return self.comparison.discard_applied
        return a

    def _shown_key(self) -> str:
        if self.comparison is not None:
            if self.radio_every.isChecked():
                return "every"
            if self.radio_discard.isChecked():
                return "discard"
        return "measured"

    def _shown_words(self, shown: Optional[AxonAnalysis]) -> str:
        """Which analysis a plot shows, as its title ends with it when a
        discard comparison exists (3.6)."""
        return L.shown_words(shown, self.comparison, self.analysis)

    def _every_column(self) -> bool:
        """Whether 'All clusters' says something the measured column does
        not: only when the measured contour came from one 2-opt start."""
        return (self.comparison is not None
                and self.comparison.all_clusters is not self.analysis)

    def _sync_discard_widgets(self) -> None:
        compared = self.comparison is not None
        every = self._every_column()
        self.box_shown.setVisible(compared)
        if compared and self.comparison is not None:
            # The counts on the buttons themselves: the row is one line
            # above a long table, and "89" beside "discard applied" is
            # what makes it read as a choice rather than a label.
            total = self.comparison.all_clusters.n_clusters_kept
            left = self.comparison.discard_applied.n_clusters_kept
            self.radio_measured.setText(f"measured ({total} clusters)")
            self.radio_every.setText(f"all clusters ({total})")
            self.radio_discard.setText(f"discard applied ({left})")
        self.box_export.setVisible(compared)
        self.radio_every.setVisible(every)
        if not every and self.radio_every.isChecked():
            self.radio_measured.setChecked(True)
        self.table.setColumnHidden(_COL_EVERY, not every)
        self.table.setColumnHidden(_COL_DISCARD, not compared)
        one_start = (self.analysis is not None
                     and self.analysis.contour_2opt == "one start")
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
                    self.spin_min, self.spin_maha):
            wdg.blockSignals(True)

        self.combo_peak.clear()
        if a is not None:
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
            # From the analysis, not from its occupancy: an analysis with too
            # few clusters to measure occupancy still ran with a threshold, and
            # reading it back from the widget is what let 0.1 through.
            self.spin_maha.setValue(a.mahalanobis_threshold)
        else:
            # No analysis yet: the values the next run will use.
            params = self.links.params()
            if params is not None:
                eps, minimum, half, maha = params
                self.spin_eps.setValue(float(eps))
                self.spin_min.setValue(int(minimum))
                self.spin_half.setValue(float(half))
                self.spin_maha.setValue(float(maha))

        for wdg in (self.combo_peak, self.spin_half, self.spin_eps,
                    self.spin_min, self.spin_maha):
            wdg.blockSignals(False)

        # (Re)connect after the initial population so the first fill does
        # not trigger a re-run.
        if self.rerun_callback is not None:
            try:
                self.combo_peak.currentIndexChanged.disconnect()
                self.spin_half.editingFinished.disconnect()
                self.spin_eps.editingFinished.disconnect()
                self.spin_min.editingFinished.disconnect()
                self.spin_maha.editingFinished.disconnect()
            except TypeError:
                pass
            self.combo_peak.currentIndexChanged.connect(self._on_param_changed)
            self.spin_half.editingFinished.connect(self._on_param_changed)
            self.spin_eps.editingFinished.connect(self._on_param_changed)
            self.spin_min.editingFinished.connect(self._on_param_changed)
            self.spin_maha.editingFinished.connect(self._on_param_changed)
            try:
                self.chk_random.stateChanged.disconnect()
            except TypeError:
                pass
            self.chk_random.stateChanged.connect(self._on_param_changed)

    def _fill_provenance(self) -> None:
        a = self._display_analysis()
        if a is None:
            name = (os.path.basename(self.analysis.source_name)
                    if self.analysis is not None else os.path.basename(self.links.root_name()))
            self.lbl_provenance.setText(
                f"<b>{name or '(unnamed ROI)'}</b><br>{NO_ANALYSIS}")
            return
        px = "unknown" if a.pixel_size_nm is None else f"{a.pixel_size_nm:g} nm"
        src = {"yaml": "from Picasso YAML",
               "hdf5": "from the metadata inside the HDF5",
               "yaml_scan": "from Picasso YAML",
               "override": "given explicitly",
               "manual": "entered manually",
               "neighbour": "from a file beside it, not this one",
               "remembered": "typed by hand for this folder earlier",
               "unknown": "UNKNOWN"}.get(a.pixel_size_source, a.pixel_size_source)
        colour = (AXIS_FG_LIGHT
                  if a.pixel_size_source in ("yaml", "hdf5", "yaml_scan")
                  else _C_TEXT_WARN)
        self.lbl_provenance.setText(
            f"<b>{os.path.basename(a.source_name) or '(unnamed ROI)'}</b><br>"
            f"Pixel size: <span style='color:{colour}'><b>{px}</b> ({src})</span>"
            f" &nbsp;|&nbsp; eps {a.eps_nm:g} nm, min samples {a.min_samples}"
            f" &nbsp;|&nbsp; slab {a.slab_zmin_nm:,.0f} .. {a.slab_zmax_nm:,.0f} nm"
        )

    def _fill_table(self) -> None:
        if self._display_analysis() is None:
            self.table.setRowCount(1)
            for col in range(self.table.columnCount()):
                self.table.setItem(0, col, QtWidgets.QTableWidgetItem(
                    NO_ANALYSIS if col == _COL_NAME else ""))
            return
        assert self.analysis is not None
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
                    item.setForeground(QtGui.QBrush(QtGui.QColor(_C_TEXT_DIM)))
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
                self._cell_tip(item, name, col, str(text))
                self.table.setItem(r, col, item)

    @staticmethod
    def _cell_tip(item: QtWidgets.QTableWidgetItem, name: str, col: int, text: str) -> None:
        """The tooltips of the reference cells, the KS p cells and the CDF
        crossing cells (rev 3, 12.8); their texts stay as they are."""
        key = _REFERENCE_KEYS.get(name)
        if col == _COL_PAPER and key is not None:
            item.setToolTip(f"{text}: {reg.format_value(key)}.\n{reg.tooltip(key)}")
        elif col in (_COL_MEASURED, _COL_EVERY, _COL_DISCARD):
            own = item.toolTip()
            if name == _KS_ROW:
                item.setToolTip(KS_P_TIP + (f"\n\n{own}" if own else ""))
            elif name == _CROSSING_ROW:
                item.setToolTip(CROSSING_TIP + "\n" + reg.tooltip("randomization.cdf_crossing_floor")
                                + (f"\n\n{own}" if own else ""))

    @staticmethod
    def _italic() -> QtGui.QFont:
        f = QtGui.QFont()
        f.setItalic(True)
        return f

    def _fill_warnings(self) -> None:
        self.list_warnings.clear()
        if self._display_analysis() is None:
            self.list_warnings.addItem(QtWidgets.QListWidgetItem(NO_ANALYSIS + "."))
            return
        assert self.analysis is not None
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
            item.setForeground(QtGui.QBrush(QtGui.QColor(_C_TEXT_WARN)))
            item.setToolTip(w)
            self.list_warnings.addItem(item)
        if not messages:
            self.list_warnings.addItem(
                QtWidgets.QListWidgetItem("No warnings."))

    # ------------------------------------------------------------------
    # the banner (3.4 rule 5)
    # ------------------------------------------------------------------

    def _update_banner(self) -> None:
        sel = self._sel
        run = self.links.run_analysis is not None
        if self.analysis is None:
            words = "" if sel is None else sel.current_words
            self.banner.set_state([NO_ANALYSIS + (f" ({words})." if words else ".")], offer_run=run)
            return
        if self._show_current:
            self.banner.set_state(showing_current_lines(current_words="" if sel is None else sel.current_words),
                                  offer_show=True, showing_current=True, offer_run=run)
            return
        if not self._stale or sel is None:
            self.banner.set_state([])
            return
        self.banner.set_state(analysis_stale_lines(analysed_words=sel.analysed_words or "as analysed",
                                                   current_words=sel.current_words or "another one",
                                                   discard_dropped=sel.discard_dropped),
                              offer_show=True, showing_current=False, offer_run=run)

    def _on_show_current(self, on: bool) -> None:
        """'Show the current selection' / 'Show the analysed selection'."""
        if self.analysis is None:
            return
        self._show_current = bool(on)
        self._fill_provenance()
        self._fill_table()
        self._fill_warnings()
        self._update_banner()
        self._draw_plots()

    def _on_run_requested(self) -> None:
        if self.links.run_analysis is not None:
            self.links.run_analysis()

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------

    def _map_inputs(self) -> L.MapInputs:
        prev = self.axon_map.inputs()
        a = self._display_analysis()
        sel = self._sel
        if a is not None and self._stale:
            # The Axoplasm panel follows the current selection: never
            # drawn against the clusters of the previous one (P2).
            state, state_reason = None, "this analysis is of the previous selection"
        else:
            state, state_reason = self.links.axoplasm()
        ms, ms_reason = self.links.rings()
        return L.MapInputs(
            analysis=a, comparison=self.comparison if a is not None else None,
            shown=self._shown_key(), stale=bool(self._stale and a is not None),
            given_after_cut=bool(sel is not None and a is not None and sel.analysed_after_cut),
            discard_failed=("" if sel is None or a is None else sel.discard_failed),
            axoplasm=state, axoplasm_reason=state_reason or prev.axoplasm_reason,
            rings=ms, rings_reason=ms_reason or prev.rings_reason,
            source=prev.source, colour_by=prev.colour_by, image=prev.image,
            selection_n=None if sel is None else sel.n_locs)

    def _draw_map(self) -> None:
        self.axon_map.set_inputs(self._map_inputs())
        a = self._display_analysis()
        sel = self._sel
        if a is not None:
            shown = self._shown()
            roi = "" if sel is None else (sel.analysed_roi_words or sel.roi_words)
            cut = None if sel is None else (sel.analysed_cut or sel.cut)
            line2 = L.provenance_line2(shown, roi_words=roi, cut=cut,
                                       given_after_cut=bool(sel is not None and sel.analysed_after_cut))
        else:
            line2 = L.provenance_line2(None, roi_words="" if sel is None else sel.roi_words,
                                       cut=None if sel is None else sel.cut)
        self.axon_map.set_provenance(self.lbl_provenance.text(), line2)

    def _draw_axial(self) -> None:
        prev = self.axial.inputs()
        sel = self._sel
        a = self._display_analysis()
        ms, ms_reason = self.links.rings()
        if a is not None:
            z = (sel.analysed_z if sel is not None and sel.analysis is self.analysis else None)
            ch2 = None if (sel is None or self._stale) else sel.ch2_z
            ch2_reason = ("the analysis is of the previous selection" if self._stale
                          else ("no channel 2 is loaded" if sel is None else sel.ch2_reason))
            inp = L.AxialInputs(
                z_roi=z, analysis=self._shown(), cut=None if sel is None else sel.cut,
                cut_from_histogram_mode=bool(sel is not None and sel.cut_from_histogram_mode),
                cut_stale=self._stale, given_after_cut=bool(sel is not None and sel.analysed_after_cut),
                ch2_z=ch2, ch2_reason=ch2_reason, ch2_shown=prev.ch2_shown,
                rings=ms, rings_reason=ms_reason or prev.rings_reason, view=prev.view)
        else:
            inp = L.AxialInputs(
                z_roi=None if sel is None else sel.z_roi, analysis=None,
                fit=None if sel is None else sel.fit,
                cut=None if sel is None else sel.cut,
                cut_from_histogram_mode=bool(sel is not None and sel.cut_from_histogram_mode),
                ch2_z=None if sel is None else sel.ch2_z,
                ch2_reason="no channel 2 is loaded" if sel is None else sel.ch2_reason,
                ch2_shown=prev.ch2_shown, rings=ms, rings_reason=ms_reason or prev.rings_reason,
                view=prev.view, fit_reason="the mixture could not be fitted" if sel is None else sel.fit_reason)
        self.axial.set_inputs(inp)

    def _draw_nn(self) -> None:
        self.nn.set_analysis(self._shown(), measured=self._display_analysis(), comparison=self.comparison)

    def _on_map_view(self, view: str) -> None:
        """The Axoplasm view shows the discard applied when there is one, as
        A1 drew both contours (3.2): the radio follows, and with it the
        table's emphasis and every tab."""
        if view == "axoplasm" and self.comparison is not None \
                and not self.radio_discard.isChecked():
            self.radio_discard.setChecked(True)

    def _draw_area(self) -> None:
        a = self._shown()
        self.plot_area.clear()
        words = self._shown_words(a)
        set_title(self.plot_area, AREA_TITLE + (f" ({words})" if words else ""), dark=self.dark)
        if a is None or a.areas is None or a.areas.areas_nm2.size == 0:
            return
        vals = a.areas.areas_nm2
        counts, edges = np.histogram(vals, bins=30)
        centres = (edges[:-1] + edges[1:]) / 2
        width = float(np.mean(np.diff(edges)))
        self.plot_area.addItem(pg.BarGraphItem(
            x=centres, height=counts, width=width,
            brush=pg.mkBrush(*rgba("locs", 190)), pen=None))
        med = a.areas.median_area_nm2
        if med is not None:
            self.plot_area.addItem(pg.InfiniteLine(
                pos=med, angle=90,
                pen=pg.mkPen(role("summary"), width=2),
                label=f"median {med:,.0f}",
                labelOpts={"position": 0.9, "color": role("summary")}))
        # The published value, in the paper role: the line itself is the
        # badge (P5), and its label names the publication.
        ref = float(reg.default("reference.cluster_area_nm2"))
        self.plot_area.addItem(pg.InfiniteLine(
            pos=ref, angle=90,
            pen=pg.mkPen(role("paper"), width=2, style=QtCore.Qt.DashLine),
            label=f"Gazal 2026 (preprint v1): {ref:,.0f} nm^2",
            labelOpts={"position": 0.75, "color": role("paper")}))

    def _draw_scatter(self) -> None:
        """Each centre against the smooth outline, with the scatter band."""
        from tools.mps_geometry import (
            DEEP_VERTEX_FRACTION, radial_profile, vertex_depths)

        a = self._shown()
        self.plot_scatter.clear()
        words = self._shown_words(a)
        title = SCATTER_TITLE + (f" ({words})" if words else "")
        if a is None:
            set_title(self.plot_scatter, title, dark=self.dark)
            return
        contour = None if a.perimeter is None else a.perimeter.contour
        profile = None if contour is None else radial_profile(contour)
        h = a.contour_health
        if profile is None or h is None:
            set_title(self.plot_scatter,
                      title + ": too few centres to fit one",
                      dark=self.dark)
            return
        deg = np.degrees(np.mod(profile.theta, 2 * np.pi))
        dense = np.linspace(0.0, 2 * np.pi, 361)
        line = profile.outline_at(dense)
        s = profile.scatter_nm
        band_hi = pg.PlotDataItem(np.degrees(dense), line + s,
                                  pen=pg.mkPen(None))
        band_lo = pg.PlotDataItem(np.degrees(dense), line - s,
                                  pen=pg.mkPen(None))
        self.plot_scatter.addItem(band_hi)
        self.plot_scatter.addItem(band_lo)
        self.plot_scatter.addItem(pg.FillBetweenItem(
            band_lo, band_hi, brush=pg.mkBrush(*rgba("occupied", 60))))
        self.plot_scatter.addItem(pg.PlotDataItem(
            np.degrees(dense), line, pen=pg.mkPen(self._neutral(), width=2)))

        depth, radius = vertex_depths(contour)
        deep = depth > DEEP_VERTEX_FRACTION * radius
        beyond = np.zeros_like(deep)
        if h.healthy_depth_fraction is not None:
            beyond = depth > max(DEEP_VERTEX_FRACTION,
                                 h.healthy_depth_fraction) * radius
        plain = ~deep
        self.plot_scatter.addItem(pg.ScatterPlotItem(
            deg[plain], profile.r[plain], size=7,
            brush=pg.mkBrush(role("centroid")), pen=pg.mkPen(None)))
        # Deep by the 0.40 count but within this axon's own scatter:
        # hollow, because the claim about them is the weak one.
        within = deep & ~beyond
        if within.any():
            self.plot_scatter.addItem(pg.ScatterPlotItem(
                deg[within], profile.r[within], size=11, symbol="s",
                brush=pg.mkBrush(None),
                pen=pg.mkPen(verdict("warn"), width=2)))
        # Deeper than a healthy ring with this scatter reaches: filled,
        # a different shape, the strong claim.
        if beyond.any():
            self.plot_scatter.addItem(pg.ScatterPlotItem(
                deg[beyond], profile.r[beyond], size=12, symbol="t",
                brush=pg.mkBrush(verdict("bad")), pen=pg.mkPen(None)))
        set_title(
            self.plot_scatter,
            f"{title}: {s:,.0f} nm = {h.scatter_percent:.1f} % of "
            f"the hull radius"
            + ("" if not deep.any() else
               f"; {int(beyond.sum())} deeper than that scatter explains "
               f"(filled), {int(within.sum())} within it (hollow)"),
            dark=self.dark)

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
                # Always off: see the note where the control used to be.
                dbcv_threshold=DEFAULT_DBCV_THRESHOLD,
                mahalanobis_threshold=float(self.spin_maha.value()),
                run_randomization=bool(self.chk_random.isChecked()),
                # A contour drawn by hand is kept across every re-run of
                # this axon. Without this line, moving ANY control -- the
                # Mahalanobis threshold included, which does not touch the
                # clusters -- would drop it and hand back the automatic
                # contour without a word: the 19.31 -> 8.96 um kind of jump
                # this program must not make on its own.
                contour_guide=(None if self.analysis is None
                               else self.analysis.contour_guide),
            )
        except Exception as exc:                      # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, "Analysis failed",
                f"Could not re-run the analysis with these parameters:\n\n{exc}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.refresh()

    def _on_draw_contour(self) -> None:
        """Open the editor; re-run the analysis along what it returns."""
        from tools.contour_editor import APPLY, AUTOMATIC, draw_contour

        a = self.analysis
        if self.rerun_callback is None:
            return
        if a is None or a.centroids is None or len(a.centroids) < 3:
            QtWidgets.QMessageBox.information(
                self, "Contour",
                "This axon has fewer than three clusters: there is no "
                "contour to draw.")
            return
        answer, path = draw_contour(
            self, a.centroids,
            current_contour=(None if a.perimeter is None
                             else a.perimeter.contour),
            current_guide=a.contour_guide)
        if answer not in (APPLY, AUTOMATIC):
            return
        peak = self.combo_peak.currentData()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.analysis = self.rerun_callback(
                main_peak_override_nm=(None if peak is None else float(peak)),
                slab_half_width_nm=float(self.spin_half.value()),
                eps_nm=float(self.spin_eps.value()),
                min_samples=int(self.spin_min.value()),
                dbcv_threshold=DEFAULT_DBCV_THRESHOLD,
                mahalanobis_threshold=float(self.spin_maha.value()),
                run_randomization=bool(self.chk_random.isChecked()),
                contour_guide=(path if answer == APPLY else None),
            )
        except Exception as exc:                      # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, "Contour",
                f"Could not re-run the analysis along that path:\n\n{exc}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.refresh()

    def _on_reset(self) -> None:
        from tools.mps_settings import (
            DEFAULT_EPS_NM, DEFAULT_MIN_SAMPLES, DEFAULT_SLAB_HALF_WIDTH_NM,
        )
        for w in (self.spin_half, self.spin_eps, self.spin_min,
                  self.spin_maha):
            w.blockSignals(True)
        self.spin_half.setValue(DEFAULT_SLAB_HALF_WIDTH_NM)
        self.spin_eps.setValue(DEFAULT_EPS_NM)
        self.spin_min.setValue(DEFAULT_MIN_SAMPLES)
        self.spin_maha.setValue(DEFAULT_MAHALANOBIS_THRESHOLD)
        self.combo_peak.blockSignals(True)
        self.combo_peak.setCurrentIndex(max(0, self.combo_peak.count() - 1))
        self.combo_peak.blockSignals(False)
        for w in (self.spin_half, self.spin_eps, self.spin_min,
                  self.spin_maha):
            w.blockSignals(False)
        self._on_param_changed()

    def _on_export(self) -> None:
        """
        Ask the main window to write this axon.

        The export is not done here on purpose: this window knows the two
        analyses, and the axoplasm panel knows the rest of what the row
        says. One place sees both, and that is where the tables are
        written -- from one state, in one act.
        """
        if self.export_callback is None:
            QtWidgets.QMessageBox.information(
                self, "Export axon",
                "This window was opened on its own, so it cannot write the "
                "tables. Use 'Export axon' in the main window.")
            return
        self.export_callback()


def _strip(text: str) -> str:
    import re
    return re.sub(r"<(?!br>)[^>]+>", "", text).replace("&nbsp;", " ")
