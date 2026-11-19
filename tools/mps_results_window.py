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
histograms is where expert judgement enters. The parameter strip at the top
is, since UI stage 2 (design 6), the ONE editor of each MPS-analysis
parameter in the program:

  * the axial slab: its centre (automatic, or a component of the fitted
    mixture -- what the "ambiguous main peak" warning refers to), its
    half-width, or a typed range that decides it
  * DBSCAN eps and min_samples
  * the Mahalanobis threshold of the occupancy and the randomization
  * the contour drawn by hand, and "Reset to defaults"

What the strip sets is written into the main window's one settings object
(``WindowLinks.store``), which every consumer reads: "cluster Ch1", "MPS
analysis", Rings, Batch, Two channels, the main-window cut and the next
session. The main window's boxes only show it. Changing a value re-runs the
analysis on the same localizations and redraws everything; a typed range
only applies the cut, as the main window's "Apply ROI" does, and the banner
offers to run the analysis. Every value carries its origin badge and its
documented range (``tools.mps_param_registry``).

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
from tools.copyable import copyable_table, selectable_labels, write_text_file
from tools.mps_ui_layout import PersistentSplitter, style_splitter
from tools.mps_nn_panel import NearestNeighboursPanel
from tools.mps_origin_ui import DetailsPanel, MoreLine, OriginBadge, ParamField
from tools.mps_params_panel import FlowLayout
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
RERUN_FAILED = ("The analysis could not be re-run with these values; what is shown was computed with the previous "
                "ones.")

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
# F6: the plots with a layer panel, whose exported figure can carry a legend of the rows drawn (the 1NN CDF draws
# its own legend inside the plot).
LEGEND_PLOTS = (PLOT_MAP, PLOT_AXIAL, PLOT_NN)

# Tooltips of the results table (rev 3, 12.8): what each reference value is,
# and the two cells whose number is this program's heuristic or has no
# valid reading. The texts of the cells themselves are unchanged.
PAPER_HEADER = "Reference (Gazal et al. 2026, preprint v1, pooled)"
PAPER_HEADER_TIP = (
    "Values of Gazal et al. 2026 (preprint v1), read on data pooled over its axons: another dataset and another "
    "pipeline; agreeing with them is not a validation criterion. The 1NN scales with the clusters detected per um, "
    "so compare that and the regularity index r first.")
KS_TIP = (
    "D is descriptive: its size under the null falls with the number of clusters K, so it is not comparable between "
    "axons or with the paper's pooled value. z_D is D in standard deviations of its own null (the D each "
    "randomization reaches against the pool): comparable between axons. The two-sample KS p is not reported: these "
    "samples break its assumptions, and under a true null it rejected up to about four times its nominal 5 % in "
    "this project's simulations.")
MC_P_TIP = (
    "Monte Carlo rank p-values, (1 + b) / (B + 1): how often a random placement with the same cluster number, band "
    "and minimum separation gives a median 1NN (two-sided) or a D at least as extreme. Approximate: the minimum "
    "separation of the null is estimated from the same data.")
CROSSING_TIP = ("Reported only where the observed CDF leaves the band the randomizations themselves give; called "
                "significant when the crossing is stronger than the randomizations themselves reach (Monte Carlo).")
# The public methods page of the randomization rows (tools.mps_randomization.METHODS_DOC).
METHODS_TIP = "Methods: docs/randomization_and_ks.md"
REGULARITY_TIP = (
    "r = median 1NN x K / P: 1 for evenly spaced centres. What random placement gives is not a fixed number: it "
    "depends on the minimum separation and the band, so it is computed for this axon from its own randomizations "
    "(mean and central 95 % of their r). Below that interval: an excess of short distances; above it: more "
    "regular than random. Its two-sided test is the Monte Carlo p of the median. "
    "Methods: docs/randomization_and_ks.md, section 6")
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
_MC_ROW = "  Monte Carlo p"
# The other rows the randomization methods page documents.
_RAND_ROWS = ("  centres outside the band",)
_REGULARITY_ROW = "  regularity r"
# The slab's expected foreign fraction (SCI-6) and the Delta-Z row: what they are, and where it is documented.
_FOREIGN_ROW = "  foreign fraction of the slab"
_DELTA_Z_ROW = "Delta-Z (axial periodicity)"
FOREIGN_TIP = ("The share of the slab's localizations the fitted axial mixture expects from components other than "
               "the slab's own ring (the normal tails of the neighbours inside the slab). This program warns above "
               "5 %, where a neighbouring ring's clusters start to reach min samples inside the slab.\n\n"
               "Methods: docs/axial_mixture_and_rings.md, section 1.")
DELTA_Z_TIP = ("Mean spacing between consecutive axial components, with its standard error and its signed distance "
               "from the published axial period in units of the published spread: a description, never a verdict. "
               "Every reference is in the Axial tab's details.\n\nMethods: docs/axial_mixture_and_rings.md, "
               "section 2.")
_CROSSING_ROW = "  CDF crossing"

# How much of the plot column the map takes when the window opens (the rest
# is the tabs under it): the map keeps at least 55 % of a 1366 x 728 window.
MAP_SHARE = 0.80
# On a short screen the tabs under the map keep at least this height (a plot
# of ~30 px cannot be read), as long as the map keeps MAP_MIN_OF_WINDOW of the
# window's height.
TABS_MIN_PX = 215
# The hard floor of the tabs' height: below the TABS_MIN_PX wish, above an unreadable strip.
TABS_FLOOR_PX = 150
# The least height of the table + plots area (a 1366 x 640 window with the Emitters box open keeps it).
SPLITTER_MIN_PX = 320
MAP_MIN_OF_WINDOW = 0.56
# Manual test of 2026-10-23 (1920 x 1020): the map ended at ~200 px under ~600 px of tabs, and with its aspect
# locked a short map shows x = -20000..20000 nm around a dot of an axon. Until the user drags the divider, a map
# below this share of the plot column is given MAP_SHARE back [ad hoc].
MAP_FLOOR_SHARE = 0.5
# The table column's width when the window opens: wide enough for the name,
# "Measured" and the reference column (1366 px screens; the user's divider
# wins afterwards). On a wider screen it takes LEFT_SHARE of the window, never
# more than a third: the axon map is the window's main element (manual test 2,
# item 4).
LEFT_MIN_PX = 430
# The limits of the DBSCAN editors, shared with the main window's boxes (manual test 2, item 7).
EPS_RANGE_NM = (0.1, 1000.0)
EPS_DECIMALS = 1
MIN_SAMPLES_RANGE = (1, 10000)
LEFT_SHARE = 0.25
# F5d: the parameter strip scrolls (tools.mps_window_scroll) once it would take more than this share of the window's
# height [ad hoc: half, so the map and the table keep the other half; the strip of a DBSCAN analysis, two lines at
# 1366 px, never reaches it].
STRIP_MAX_SHARE = 0.5
# How long the line "Cancelled ... the analysis shown is the previous one" stays after a cancelled run [ad hoc].
CANCELLED_NOTE_MS = 15000
# F5: how many methods' rows start ticked after 'Compare methods...' [ad hoc: two or three loops per cluster are
# still readable; the rest are one tick away].
METHODS_TICKED = 3
# F4: where the window remembers that the clustering method's panel was collapsed (mps_ui_layout.json).
METHOD_COLLAPSED_KEY = "mps_window.method_panel_collapsed"
# How many warnings the button's tooltip previews, and how long each line.
WARNINGS_PREVIEW = 5
WARNINGS_PREVIEW_CHARS = 180


class _Row:
    """One line of a strip item: widgets and layouts left to right (the grid positions the strip's code gives are
    ignored: an item is one line, and the flow wraps whole items)."""

    def __init__(self, frame: QtWidgets.QFrame) -> None:
        self.lay = QtWidgets.QHBoxLayout(frame)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(4)

    def addWidget(self, widget: QtWidgets.QWidget, *_grid: int) -> None:  # noqa: N802 - Qt's name
        self.lay.addWidget(widget)

    def addLayout(self, layout: QtWidgets.QLayout, *_grid: int) -> None:  # noqa: N802
        self.lay.addLayout(layout)


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
    rerun_failed: bool = False                       # a slab change's re-run failed (design 6.3)
    guide_note: str = ""                             # a drawn contour was dropped by a new ROI or file (B11)


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


def _no_z_calibration() -> Tuple[str, Optional[float]]:
    return "", None


def _no_slab_state() -> Tuple[str, Optional[float], Optional[Tuple[float, float]]]:
    return "automatic", None, None


def _no_cut_state() -> Tuple[bool, str]:
    return False, ""


def _no_nena() -> Tuple[Optional[float], str]:
    return None, "no NeNA measured for this file"


def _no_rule_inputs() -> Optional[Any]:
    return None


def _no_c_label() -> Optional[float]:
    return None


def _no_rule_inputs_at(_radius: Optional[float]) -> Optional[Any]:
    return None


def _no_lpc_depth(_analysis: Any, _radius: Optional[float] = None) -> Tuple[Optional[Any], str]:
    return None, "no file loaded in this window"


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
    # The measurement's z calibration (SCI-1): the words of the provenance line and the axial titles ("" keeps
    # today's "z as fitted") and m. Display only.
    z_calibration: Callable[[], Tuple[str, Optional[float]]] = field(default=_no_z_calibration)
    # The one value of each parameter (design 6.2): the strip's setters write it through ``store`` (eps_nm,
    # min_samples, slab_half_width_nm, mahalanobis_threshold, randomization, slab = ("automatic", None) |
    # ("component", centre), reset); ``slab_state`` says how the slab is chosen now (mode, component centre, the
    # typed range or None); ``apply_typed`` applies a typed range, or clears it with None (the cut only, no re-run).
    # None / the defaults: a window on its own, whose controls only re-run its own analysis.
    store: Optional[Callable[[Dict[str, Any]], Any]] = None
    slab_state: Callable[[], Tuple[str, Optional[float], Optional[Tuple[float, float]]]] = field(
        default=_no_slab_state)
    # Phase D (D0): the clustering method in use (name, its parameters, the 'Experimental' box, the methods
    # compared before choosing); the strip writes them through ``store`` (cluster_method, cluster_params,
    # experimental_methods_enabled, cluster_methods_compared). None: the window's own panel keeps them.
    method_state: Optional[Callable[[], Tuple[str, Dict[str, Any], bool, List[str]]]] = None
    apply_typed: Optional[Callable[[Optional[Tuple[float, float]]], Any]] = None
    # The clusters cut by the Picasso pick (P-R59): the switch "Exclude cut clusters" and what the next analysis
    # will do before there is one (``cut_state``), the switch's setter (it re-runs) and the Pick boundary dialog.
    cut_state: Callable[[], Tuple[bool, str]] = field(default=_no_cut_state)
    set_exclude_cut: Optional[Callable[[bool], Any]] = None
    pick_boundary: Optional[Callable[[], Any]] = None
    nn_bins: int = L.NN_DEFAULT_BINS
    nn_range: Tuple[float, float] = (0.0, 800.0)
    # The clustering integration (2026-10-22): the file's NeNA and where it came from (the working scale, the event
    # layer, the DBSCAN rule, the null), and the localizations per cycle measured on the whole file (open question
    # B3; a tools.mps_events.LocsPerCycle, or None for the rule's labelled fallback).
    nena: Callable[[], Tuple[Optional[float], str]] = field(default=_no_nena)
    rule_inputs: Callable[[], Optional[Any]] = field(default=_no_rule_inputs)
    # The c_label kept in the settings (2026-10-29): the Working scale panel starts at it; None keeps the panel's 14.
    c_label: Callable[[], Optional[float]] = field(default=_no_c_label)
    # The interface batch (2026-11-07). B1: the whole file's localizations per cycle at the radius typed in 'Working
    # scale' (None = the rule's). B3 (c): the localizations per cycle by depth of the loaded file for an analysis'
    # slab (tools.lpc_depth), (profile or None, why empty).
    rule_inputs_at: Callable[[Optional[float]], Optional[Any]] = field(default=_no_rule_inputs_at)
    lpc_depth: Callable[..., Tuple[Optional[Any], str]] = field(default=_no_lpc_depth)
    # The perimeter density window (P12): the whole pick of the current ROI as a
    # tools.mps_perimeter_window.PerimeterInputs, or None. None here: the button stays disabled.
    perimeter_inputs: Optional[Callable[[], Optional[Any]]] = None
    # The unanchored clusters window (P15): opens it on the analysis shown. None here: the button stays disabled.
    open_unanchored: Optional[Callable[[], Any]] = None
    # Cancels the automatic unanchored run of the analysis shown (the second progress line's Cancel).
    cancel_unanchored: Optional[Callable[[], Any]] = None


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
        background: Optional[Callable[..., Any]] = None,
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
        background : F5d. Called in the GUI thread with the same keywords
            as ``rerun_callback`` when the method is slow
            (``tools.cluster_method_ui.is_slow``); returns a
            ``tools.mps_background_run.PreparedRun`` whose computation runs
            in a worker thread while this window shows the time and a Cancel
            button, or raises. None: every re-run is synchronous, as before.
        """
        super().__init__(parent)
        self.analysis = analysis
        self.rerun_callback = rerun_callback
        self.rings_callback = rings_callback
        self.discard_callback = discard_callback
        self.export_callback = export_callback
        self.links = links if links is not None else WindowLinks()
        self.background = background
        # The main window's answer to "is the selection large enough to run in the background whatever the
        # method" (tools.mps_size_guard); None: only the slow methods do.
        self.big_run: Optional[Callable[[], bool]] = None
        # The perimeter density window (P12), its notes and the axon they are about.
        self._perimeter_window: Any = None
        self._perimeter_notes: List[str] = []
        self._perimeter_notes_key: Any = None
        # F5d: the run in the background, if any, and how to give the controls back their states after it.
        self._run: Any = None
        self._restore_controls: Optional[Callable[[], None]] = None
        self.comparison: Optional[DiscardComparison] = None
        # A second method drawn over this analysis on the map ('Compare methods' -> 'Overlay on the map';
        # tools.method_overlay.MethodOverlay), dropped when another analysis is shown.
        self._overlay: Any = None
        # The clustering integration: the last 'Compare methods...' of the analysis shown
        # (tools.clustering_integration.ComparisonResult), the event layer's probable fusions on the map (cached per
        # analysis) and the c_label of the Working scale panel.
        self.method_comparison: Any = None
        self.method_comparison_files: Dict[str, str] = {}   # the comparison's tables, when 'Export tables...' ran
        self._fusion_cache: Tuple[Any, Any] = (None, None)
        # F5: every method 'Compare methods...' ran on this analysis, each a row of the map
        # (tools.method_overlay.MethodLayers), dropped with the analysis as the overlay is.
        self._methods: Any = None
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

        # F5d: the strip in a scroll area of its own: a group that grows (the Emitters box, a "More" line) scrolls
        # inside it instead of pushing the window past the bottom of the screen (manual test of 2026-10-19).
        from tools.mps_window_scroll import CappedScrollArea
        self.strip_scroll = CappedScrollArea(self._build_controls(), object_name="param_strip_scroll")
        root.addWidget(self.strip_scroll)
        self.method_panel.combo.activated.connect(lambda _i: self._reveal_method_box())
        self.banner = StaleBanner(object_name="mps_stale_banner")
        self.banner.show_current_toggled.connect(self._on_show_current)
        self.banner.run_requested.connect(self._on_run_requested)
        root.addWidget(self.banner)
        from tools.mps_background_run import RunProgress
        self.run_progress = RunProgress()
        self.run_progress.cancel_requested.connect(self.cancel_background)
        self._progress_timer = QtCore.QTimer(self)
        self._progress_timer.setSingleShot(True)
        self._progress_timer.timeout.connect(self._hide_progress)
        root.addWidget(self.run_progress)
        # P15: the automatic unanchored run (the rule, then the analysis without the excluded clusters) has its own
        # line, so it never takes the re-run's Cancel.
        self.side_progress = RunProgress()
        self.side_progress.setObjectName("mps_unanchored_progress")
        self.side_progress.label.setWordWrap(False)          # one line: it must not take the map's height
        self.side_progress.label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.side_progress.btn_cancel.setToolTip("Stop the unanchored-cluster run; the analysis shown stays as it is.")
        self.side_progress.cancel_requested.connect(self._cancel_unanchored)
        root.addWidget(self.side_progress)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_left_column())
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        # The window never gives the table and the plots less than this, whatever the strip above has open
        # (it was the tabs' own ~205 px minimum that guaranteed it before TABS_FLOOR_PX).
        splitter.setMinimumHeight(SPLITTER_MIN_PX)
        root.addWidget(splitter, stretch=1)
        self.splitter = splitter
        # The table is widened once when its extra columns appear; after
        # that the divider stays where the user leaves it.
        self._widened = False
        self._left_done = False
        splitter.splitterMoved.connect(self._on_left_moved)
        # Every boundary is a splitter the user can drag, and the sizes are kept for the next session (manual
        # test 2, item 4): results | plots, map | tabs, map | its layers.
        self._layout_keys = {"mps_window.results_plots": splitter,
                             "mps_window.map_tabs": self.plot_splitter,
                             "mps_window.map_layers": self.axon_map.splitter}
        self._persistent: List[PersistentSplitter] = []
        for key, sp in self._layout_keys.items():
            style_splitter(sp)
            self._persistent.append(PersistentSplitter(sp, key))
        self._restored = False
        self._make_text_copyable()
        from tools.mps_ui_layout import load_flag
        self.set_method_collapsed(load_flag(METHOD_COLLAPSED_KEY), save=False)
        from tools import ui_search
        ui_search.install(self)                       # F3: Ctrl+F

        self.refresh()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_controls(self) -> QtWidgets.QWidget:
        """The parameter strip (design 6.1, 12.3-12.5): the one editor of
        each MPS-analysis parameter. Two rows of groups that wrap (a flow
        layout) so nothing is clipped on a 1366-px screen: the parameters
        (Axial slab, DBSCAN, Occupancy, Randomization, Contour) and the
        actions. Every value carries its origin badge and its documented
        range; each group has one collapsed "More" line with the read-only
        research values. The editors keep their names and limits (G4,
        12.4): nothing is clamped, nothing new is editable."""
        box = QtWidgets.QWidget()
        box.setObjectName("param_strip")
        outer = QtWidgets.QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        params_row = QtWidgets.QWidget()
        params_row.setObjectName("strip_params")
        # One flow of one-line items, the parameters and "Reset to
        # defaults": it wraps onto a second line at 1366 px, so the map
        # keeps its share of the height (a "More" line opened widens its
        # item, which then wraps whole).
        self.strip_flow = FlowLayout(params_row, spacing=8)
        self.actions_flow = self.strip_flow
        outer.addWidget(params_row)

        def group(title: str, name: str) -> Tuple[QtWidgets.QFrame, "_Row"]:
            g = QtWidgets.QFrame()
            g.setObjectName(name)
            row = _Row(g)
            if title:
                head = QtWidgets.QLabel(f"<b>{title}</b>")
                head.setObjectName(name + "_title")
                row.addWidget(head)
            self.strip_flow.addWidget(g)
            return g, row

        # --- Axial slab: centre (automatic / a component), half-width, or a
        # typed range that decides it.
        g_slab, ls = group("Axial slab", "group_slab")
        self.combo_peak = QtWidgets.QComboBox()
        self.combo_peak.setMinimumWidth(205)
        # A fixed width: the items' length must not widen the strip's line.
        self.combo_peak.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_peak.setMinimumContentsLength(24)
        self.combo_peak.setToolTip(
            "Centre of the 180 nm axial slab that isolates one MPS segment.\n"
            "Defaults to the main peak of the fitted Gaussian mixture. When\n"
            "two peaks have comparable weight the choice is ambiguous and\n"
            "changes which segment is analysed - pick it yourself here."
        )
        centre_row = QtWidgets.QHBoxLayout()
        centre_row.setSpacing(4)
        centre_row.addWidget(self.combo_peak)
        self.badge_centre = OriginBadge("slab.centre", None)
        centre_row.addWidget(self.badge_centre)
        self.btn_slab_auto = QtWidgets.QPushButton("Automatic")
        self.btn_slab_auto.setObjectName("btn_slab_auto")
        self.btn_slab_auto.setToolTip(
            "Centre the slab on the density peak of the fitted mixture again\n"
            "(the automatic choice), and run the analysis.")
        self.btn_slab_auto.clicked.connect(self._on_slab_automatic)
        centre_row.addWidget(self.btn_slab_auto)
        centre_row.addStretch(1)
        ls.addLayout(centre_row, 0, 1)
        self.lbl_slab_mode = QtWidgets.QLabel("")
        self.lbl_slab_mode.setObjectName("lbl_slab_mode")
        self.lbl_slab_mode.setStyleSheet(f"color: {_C_TEXT_DIM};")
        ls.addWidget(self.lbl_slab_mode, 1, 1)

        g_half, ls = group("", "group_slab_half")
        ls.addWidget(QtWidgets.QLabel("half-width [nm]:"), 2, 0)
        self.spin_half = QtWidgets.QDoubleSpinBox()
        self.spin_half.setRange(1.0, 5000.0)
        self.spin_half.setDecimals(1)
        self.spin_half.setSingleStep(5.0)
        self.spin_half.setToolTip(
            "Half of the axial window. The paper uses 90 nm, i.e. a 180 nm slab."
        )
        self.field_half = ParamField("slab.half_width_nm", self.spin_half)
        ls.addWidget(self.field_half, 2, 1)

        self.chk_typed = QtWidgets.QCheckBox("typed [nm]:")
        self.chk_typed.setObjectName("chk_typed")
        self.chk_typed.setToolTip(
            "Type the slab's bounds yourself. Applying them sets the main\n"
            "window's cut, as its 'Apply ROI' does, and does not re-run the\n"
            "analysis: the banner says the analysis is of the previous slab\n"
            "and offers to run it. While a range is typed, the centre and\n"
            "the half-width are not used. Untick to go back to them.")
        g_typed, ls = group("", "group_slab_typed")
        ls.addWidget(self.chk_typed, 3, 0)
        typed_row = QtWidgets.QHBoxLayout()
        typed_row.setSpacing(4)
        self.edit_typed_min = QtWidgets.QLineEdit()
        self.edit_typed_min.setObjectName("edit_typed_min")
        self.edit_typed_max = QtWidgets.QLineEdit()
        self.edit_typed_max.setObjectName("edit_typed_max")
        for edit in (self.edit_typed_min, self.edit_typed_max):
            edit.setMaximumWidth(62)
            edit.setPlaceholderText("nm")
            typed_row.addWidget(edit)
        self.btn_typed_apply = QtWidgets.QPushButton("Apply")
        self.btn_typed_apply.setObjectName("btn_typed_apply")
        self.btn_typed_apply.setToolTip(
            "Apply this axial range to the main window's cut (the analysis\n"
            "is not re-run: the banner offers it).")
        self.btn_typed_apply.clicked.connect(self._on_typed_apply)
        typed_row.addWidget(self.btn_typed_apply)
        self.badge_typed = OriginBadge("slab.typed_range", None)
        typed_row.addWidget(self.badge_typed)
        typed_row.addStretch(1)
        ls.addLayout(typed_row, 3, 1)
        self.chk_typed.toggled.connect(self._on_typed_toggled)

        # --- DBSCAN (MPS analysis): eps, min samples; the curation and the
        # contour in its "More" line (the TODO-B6 hook, filled by B6).
        g_db, ld = group("DBSCAN", "group_dbscan")
        g_db.setToolTip("DBSCAN of the MPS analysis (not the main window's window clustering).")
        ld.addWidget(QtWidgets.QLabel("eps [nm]:"), 0, 0)
        self.spin_eps = QtWidgets.QDoubleSpinBox()
        self.spin_eps.setRange(*EPS_RANGE_NM)
        self.spin_eps.setDecimals(EPS_DECIMALS)
        self.spin_eps.setToolTip(
            "DBSCAN search radius. The paper uses 25 nm, matching its ~20 nm\n"
            "lateral localization precision."
        )
        self.field_eps = ParamField("dbscan.eps_nm", self.spin_eps)
        ld.addWidget(self.field_eps, 0, 1)
        ld.addWidget(QtWidgets.QLabel("min samples:"), 1, 0)
        self.spin_min = QtWidgets.QSpinBox()
        self.spin_min.setRange(*MIN_SAMPLES_RANGE)
        self.spin_min.setToolTip(
            "DBSCAN minimum points per cluster. The paper uses 10, matching\n"
            "the expected number of blinking cycles per fluorophore."
        )
        self.field_min = ParamField("dbscan.min_samples", self.spin_min)
        ld.addWidget(self.field_min, 1, 1)
        self.more_dbscan = MoreLine("dbscan")
        ld.addWidget(self.more_dbscan, 2, 0, 1, 2)
        # Phase D (D0): which method makes the clusters. DBSCAN (above) is
        # the default and the compatibility path; the rest sit behind the
        # 'Experimental' box (D10). Nothing here runs on its own (D12).
        # Manual test of 2026-10-23: the method's controls are an item of the strip of their own (they sat inside
        # the DBSCAN line, which made that line as tall as the method's parameters, with empty space on both
        # sides); the flow puts them on a line of their own when the window is narrow.
        from tools.cluster_method_ui import MethodPanel
        g_method, lm = group("", "group_method")
        self.method_panel = MethodPanel()
        lm.addWidget(self.method_panel)
        self.method_panel.method_changed.connect(self._on_method_changed)
        self.method_panel.layer_run_requested.connect(self._on_layer_run_requested)
        self.method_panel.compat_requested.connect(self._on_compat)
        self.method_panel.compare_requested.connect(self._on_compare)
        # F4 (manual test of 2026-10-22): the panel collapses to one line and back; a run is never touched.
        from tools.cluster_method_ui import MethodCollapsedBar
        self.method_bar = MethodCollapsedBar()
        self.method_bar.setVisible(False)
        lm.addWidget(self.method_bar)
        self.method_panel.collapse_requested.connect(lambda: self.set_method_collapsed(True))
        self.method_bar.expand_requested.connect(lambda: self.set_method_collapsed(False))
        self.method_bar.cancel_requested.connect(self.cancel_background)
        self.method_panel.combo.currentTextChanged.connect(lambda name: self.method_bar.set_method(name or "DBSCAN"))

        # A "DBCV thr." spin box stood here until 2026-09-20. On real axons
        # the per-cluster score falls with the cluster's extent, and in
        # many axons the LARGEST cluster is among the lowest-scoring ones:
        # every threshold above off removes the big clusters first. There
        # is no setting of it that curates without doing that, so it is
        # gone rather than defaulted. The analysis still takes the
        # parameter and the export still records it, so older tables stay
        # readable; what curates is the edge-touching criterion.

        # --- Occupancy: the Mahalanobis threshold; the sigma cap in "More".
        g_occ, lo = group("", "group_occupancy")
        lo.addWidget(QtWidgets.QLabel("<b>Occupancy</b> Mahalanobis:"), 0, 0)
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
        self.field_maha = ParamField("occupancy.mahalanobis", self.spin_maha)
        lo.addWidget(self.field_maha, 0, 1)
        self.more_occupancy = MoreLine("occupancy", editable=True)
        self.more_occupancy.edit_button.setToolTip("Edit the sigma cap of the occupancy ellipses, then re-run.")
        self.more_occupancy.edit_button.clicked.connect(self._on_edit_occupancy)
        lo.addWidget(self.more_occupancy, 1, 0, 1, 2)

        # --- Randomization: the switch (no badge: a switch carries none,
        # 12.1); its parameters, read-only in stage 2, in "More".
        g_rand, lr = group("", "group_randomization")
        self.chk_random = QtWidgets.QCheckBox("Randomization")
        self.chk_random.setChecked(True)
        self.chk_random.setToolTip(
            "Run the randomization control (parameter 8; 999 iterations by default).\n"
            "It is the slowest step (a few seconds per axon), so switch it\n"
            "off while tuning the other parameters and back on for the\n"
            "final numbers."
        )
        lr.addWidget(self.chk_random, 0, 0)
        self.more_randomization = MoreLine("randomization", editable=True)
        self.more_randomization.edit_button.setToolTip(
            "Edit the randomization control (band, iterations, seed, smoothing, incomplete iterations, attempts, "
            "grid, CDF band, hard core), then re-run.")
        self.more_randomization.edit_button.clicked.connect(self._on_edit_randomization)
        lr.addWidget(self.more_randomization, 1, 0)

        # --- Contour
        g_con, lc = group("", "group_contour")
        self.btn_contour = QtWidgets.QPushButton("Draw contour...")
        self.btn_contour.setToolTip(
            "Drag a path along the membrane and join the cluster centres in\n"
            "the order they fall along it, instead of the automatic 2-opt\n"
            "contour. For axons where the automatic contour runs through\n"
            "centres that are not on the membrane, or cuts a concavity.\n\n"
            "The path is kept for this axon: changing eps, min samples or\n"
            "the slab re-orders the new clusters along it, and the clusters\n"
            "the axoplasm panel keeps are joined along it too. 'cluster Ch1'\n"
            "and 'MPS analysis' keep it while the file and the ROI are the\n"
            "same; a new ROI or file drops it, and says so. It is written\n"
            "to the table, so the contour can be rebuilt from there.")
        self.btn_contour.clicked.connect(self._on_draw_contour)
        lc.addWidget(self.btn_contour, 0, 0)

        for more in (self.more_dbscan, self.more_occupancy, self.more_randomization):
            more.button.toggled.connect(lambda _on: self._strip_relayout())

        # --- the actions
        self.btn_reset = QtWidgets.QPushButton("Reset to defaults")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setToolTip(reg.reset_tooltip())
        self.btn_reset.clicked.connect(self._on_reset)
        # placed under the results table, beside 'Analysis notes (N)...' (_build_left_column): with the badges'
        # publication name the strip keeps two lines at 1366 px only without it (manual test 2)

        # The axon's actions sit at the right of the plot tabs' bar, a line
        # that is otherwise empty: the strip keeps two lines at 1366 px and
        # the map its share of the height.
        self.actions_box = QtWidgets.QWidget()
        self.actions_box.setObjectName("strip_actions")
        actions = QtWidgets.QHBoxLayout(self.actions_box)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)

        self.btn_rings = QtWidgets.QPushButton("Rings...")
        self.btn_rings.setToolTip(
            "Analyse EVERY axial segment of this axon, not just this slab,\n"
            "and compare the gap/patch pattern of consecutive segments.\n"
            "Pressed again, it keeps the rings window's mode and guard."
        )
        self.btn_rings.clicked.connect(self._on_rings)
        self.btn_rings.setEnabled(self.rings_callback is not None)
        actions.addWidget(self.btn_rings)
        # The perimeter density analysis (P12): the ONE way in; its notes join 'Analysis notes'.
        self.btn_perimeter = QtWidgets.QPushButton("Perimeter...")
        self.btn_perimeter.setObjectName("btn_perimeter")
        self.btn_perimeter.setToolTip("Perimeter density: slices, density profiles and tortuosity of the membrane.")
        self.btn_perimeter.clicked.connect(self._on_perimeter)
        self.btn_perimeter.setEnabled(self.links.perimeter_inputs is not None)
        actions.addWidget(self.btn_perimeter)
        # The unanchored clusters (P15): preview of the automatic discard and the decisions by hand.
        self.btn_unanchored = QtWidgets.QPushButton("Unanchored...")
        self.btn_unanchored.setObjectName("btn_unanchored")
        self.btn_unanchored.setToolTip("Clusters under the membrane: preview, exclude or restore by hand, Apply.")
        self.btn_unanchored.clicked.connect(self._on_unanchored)
        self.btn_unanchored.setEnabled(self.links.open_unanchored is not None)
        actions.addWidget(self.btn_unanchored)

        self.btn_export = QtWidgets.QPushButton("Save axon...")
        self.btn_export.setToolTip(
            "Write this axon: one row with the measured parameters and the\n"
            "ones with the discard applied, plus the per-cluster and\n"
            "per-localization tables if you ask for them. Everything is\n"
            "written from the state on screen, in one go.")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(self.export_callback is not None)
        actions.addWidget(self.btn_export)

        self.btn_image = QtWidgets.QPushButton("Export image...")
        self.btn_image.setToolTip(
            "Write one of these plots as a figure: at the width a journal\n"
            "asks for, at 300 to 1200 dpi or as an SVG of curves, and drawn\n"
            "on white rather than on the screen's black. Its title says\n"
            "which analysis, which localizations and which colouring it\n"
            "draws, and the message after writing it repeats the provenance.")
        self.btn_image.clicked.connect(self._on_export_image)
        actions.addWidget(self.btn_image)

        enabled = self.rerun_callback is not None
        for w in (self.combo_peak, self.spin_half, self.spin_eps,
                  self.spin_min, self.spin_maha,
                  self.chk_random, self.btn_reset, self.btn_contour,
                  self.more_randomization.edit_button, self.more_occupancy.edit_button,
                  self.btn_slab_auto, self.chk_typed, self.edit_typed_min,
                  self.edit_typed_max, self.btn_typed_apply, self.method_panel):
            w.setEnabled(enabled)
        # A typed range needs the main window's cut: a window on its own
        # has none.
        if self.links.apply_typed is None:
            for w in (self.chk_typed, self.edit_typed_min, self.edit_typed_max,
                      self.btn_typed_apply):
                w.setEnabled(False)
        for spin in (self.spin_half, self.spin_eps, self.spin_min, self.spin_maha):
            spin.valueChanged.connect(lambda *_a: self._sync_reset())
        # The method's controls last in the flow: the one-line items pack the lines above them, and the taller
        # method item takes the last line (one line less at 1366 px; manual test of 2026-10-23).
        self.strip_flow.removeWidget(g_method)
        self.strip_flow.addWidget(g_method)

        return box

    def _make_text_copyable(self) -> None:
        """Manual test 2, item 3: the tables (selected cells, tab-separated), the warnings and the labels that
        carry values can be selected and copied (Ctrl+C or the context menu)."""
        copyable_table(self.table)
        copyable_table(self.details_area.table)
        # manual test of 2026-10-23: a plot's title is drawn, not a label: 'Copy title' in its context menu
        from tools.copyable import copy_title_action
        for plot in self._plots().values():
            copy_title_action(plot)
        selectable_labels([self.lbl_slab_mode, self.label_export, self.banner.label, self.axon_map.lbl_line1,
                           self.axon_map.lbl_line2, self.nn.lbl_note, self.label_shown])

    def _strip_relayout(self) -> None:
        """A "More" line opened or closed: the flow re-places the groups."""
        self.strip_flow.invalidate()
        self.centralWidget().updateGeometry()
        scroll = getattr(self, "strip_scroll", None)
        if scroll is not None:
            scroll.schedule_fit()

    def _fit_strip(self) -> None:
        """The strip's cap: STRIP_MAX_SHARE of the window's height (its minimum stays a constant, so the cap
        never raises the window's minimum height)."""
        scroll = getattr(self, "strip_scroll", None)
        central = self.centralWidget()
        if scroll is None or central is None:
            return
        scroll.set_cap(int(STRIP_MAX_SHARE * central.height()))
        scroll.fit()

    def method_collapsed(self) -> bool:
        """F4: the clustering method's panel is collapsed to its one-line bar."""
        return not self.method_bar.isHidden()

    def set_method_collapsed(self, on: bool, save: bool = True) -> None:
        """F4: collapse the clustering method's panel to one line (its name, a run's state and Cancel), or expand
        it. Only widgets are hidden or shown: a run in the background goes on, nothing is re-run, and the state is
        kept for the next session (mps_ui_layout.json, beside the settings; not an analysis setting)."""
        on = bool(on)
        self.method_bar.set_method(self.method_panel.current_method())
        self.method_panel.setVisible(not on)
        self.method_bar.setVisible(on)
        if save:
            from tools.mps_ui_layout import save_flag
            save_flag(METHOD_COLLAPSED_KEY, on)
        self._strip_relayout()
        if not on:
            self._reveal_method_box()

    def _reveal_method_box(self) -> None:
        """The user chose a method: its slow-method note and its Emitters box are scrolled into sight."""
        p = self.method_panel
        if not p.layer_box.isHidden():
            # the note first; the box's title and first rows (the imaging buffer to choose) win on a short screen
            self.strip_scroll.reveal(p.lbl_slow, p.layer_box)
        elif not p.lbl_hint.isHidden():
            # F2 (manual test of 2026-10-22): the hint of a method with its own picture (Voronoi: where the cells
            # are drawn) sits under the parameters, out of a 1366 x 700 viewport; the slow note first, the hint wins
            self.strip_scroll.reveal(p.lbl_slow if not p.lbl_slow.isHidden() else p.lbl_hint, p.lbl_hint)
        elif not p.lbl_slow.isHidden():
            self.strip_scroll.reveal(p.lbl_slow)

    # ------------------------------------------------------------------
    # the strip and the one value (design 6.2, 6.3, 12.5)
    # ------------------------------------------------------------------

    def _store_values(self, reset: bool = False) -> None:
        """Write what the strip shows into the one settings object (the
        main window's setters): the strip is the only editor, and nothing
        reads a mirror back. A window on its own has no store."""
        if self.links.store is None:
            return
        values: Dict[str, Any] = dict(
            eps_nm=float(self.spin_eps.value()),
            min_samples=int(self.spin_min.value()),
            slab_half_width_nm=float(self.spin_half.value()),
            mahalanobis_threshold=float(self.spin_maha.value()),
            randomization=bool(self.chk_random.isChecked()),
            cluster_method=self.method_panel.current_method(),
            cluster_params=self.method_panel.current_params(),
            experimental_methods_enabled=self.method_panel.experimental(),
            cluster_methods_compared=self.method_panel.compared())
        _mode, _centre, typed = self.links.slab_state()
        i, n = self.combo_peak.currentIndex(), self.combo_peak.count()
        if typed is None and 0 <= i < n - 1:
            # A component of the mixture (the last item is "the current
            # slab centre": choosing it keeps the slab as it is).
            values["slab"] = ("component", float(self.combo_peak.itemData(i)))
        if reset:
            values["reset"] = True
        self.links.store(values)

    def _sync_slab_widgets(self) -> None:
        """The slab's mode in words, its badges, and the typed range."""
        mode, centre, typed = self.links.slab_state()
        from tools.mps_params_panel import SlabChoice, mode_words
        choice = SlabChoice("component", centre) if (mode == "component" and centre is not None) \
            else SlabChoice()
        self.lbl_slab_mode.setText(mode_words(choice, typed))
        self.badge_centre.set_value(
            None if (mode == "automatic" or typed is not None) else centre)
        self.badge_typed.set_value(typed)
        self.chk_typed.blockSignals(True)
        self.chk_typed.setChecked(typed is not None)
        self.chk_typed.blockSignals(False)
        # The fields always show the slab in force (manual test 2, item 2): the typed range when there is one,
        # otherwise the current cut - refreshed whenever the slab changes (a component, "Automatic", the
        # half-width) - so "Apply" never applies the bounds of a slab that is no longer there.
        bounds = typed if typed is not None else self._current_slab_bounds()
        for edit, value in zip((self.edit_typed_min, self.edit_typed_max),
                               (None, None) if bounds is None else bounds):
            edit.setText("" if value is None else f"{float(value):.1f}")
        enabled = self.rerun_callback is not None
        typed_on = typed is not None
        reason = ("A typed range decides the slab: untick 'typed range' to use the "
                  "centre and the half-width again.")
        for w in (self.combo_peak, self.btn_slab_auto):
            w.setEnabled(enabled and not typed_on)
        # Disabling a box that has the focus moves the focus out of it, and
        # its editingFinished would re-run the analysis on the typed range
        # just applied without a re-run (Q12): not a user's edit.
        was = self.spin_half.blockSignals(True)
        try:
            self.spin_half.setEnabled(enabled and not typed_on)
        finally:
            self.spin_half.blockSignals(was)
        self.btn_slab_auto.setEnabled(enabled and not typed_on and mode == "component")
        self.lbl_slab_mode.setToolTip(reason if typed_on else "")

    def _current_slab_bounds(self) -> Optional[Tuple[float, float]]:
        """The slab in force now: the main window's cut, else the analysis' own slab, else None."""
        sel = self._sel
        if sel is not None and sel.cut is not None:
            return (float(sel.cut[0]), float(sel.cut[1]))
        a = self.analysis
        if a is not None and a.slab_zmin_nm is not None and a.slab_zmax_nm is not None:
            return (float(a.slab_zmin_nm), float(a.slab_zmax_nm))
        return None

    def _sync_reset(self) -> None:
        """'Reset to defaults' is highlighted while any of its four values
        departs from its default (display only). The badges follow the
        values too (a refresh sets them with the signals blocked)."""
        for f in (self.field_half, self.field_eps, self.field_min, self.field_maha):
            f.refresh()
        departs = any(reg.badge(key, value) == "user" for key, value in (
            ("dbscan.eps_nm", float(self.spin_eps.value())),
            ("dbscan.min_samples", int(self.spin_min.value())),
            ("slab.half_width_nm", float(self.spin_half.value())),
            ("occupancy.mahalanobis", float(self.spin_maha.value()))))
        self.btn_reset.setStyleSheet("font-weight: bold;" if departs else "")

    # ---------------- Phase D: the clustering method (D10-D12) ----------

    def _method_kwargs(self) -> Dict[str, Any]:
        """What the re-run passes for the method: nothing for DBSCAN (the
        compatibility path: analyze_axon's defaults, bit for bit)."""
        name = self.method_panel.current_method()
        if name == "DBSCAN":
            compared = self.method_panel.compared()
            return {"cluster_methods_compared": compared} if compared else {}
        return dict(cluster_method=name, cluster_params=self.method_panel.current_params(),
                    cluster_methods_compared=self.method_panel.compared())

    def _on_method_changed(self) -> None:
        """Another method chosen in the strip: saved, and the analysis re-run
        with it (the user asked for it by choosing; nothing runs on load)."""
        if self.rerun_callback is None:
            return
        self._on_param_changed()

    def _on_layer_run_requested(self) -> None:
        """'Run with these values' pressed. method_changed, emitted just before, has already stored the strip's
        values (and re-run when there is an analysis). With no analysis yet that stops at the banner (B10), but
        this button is an explicit run request (manual test step 14): it starts the run as the banner's 'Run the
        MPS analysis' does. Choosing a method or editing a value with no analysis still runs nothing (D12)."""
        if self.rerun_callback is None or self.analysis is not None:
            return
        self._on_run_requested()

    def _on_compat(self) -> None:
        """'Compatibility 25/10': DBSCAN, eps 25 nm, min samples 10 (the
        published pipeline, Gazal et al. 2026), saved and re-run."""
        from tools.mps_settings import DEFAULT_EPS_NM, DEFAULT_MIN_SAMPLES
        for w in (self.spin_eps, self.spin_min):
            w.blockSignals(True)
        self.spin_eps.setValue(DEFAULT_EPS_NM)
        self.spin_min.setValue(DEFAULT_MIN_SAMPLES)
        for w in (self.spin_eps, self.spin_min):
            w.blockSignals(False)
        self.method_panel.set_compat()
        self._on_param_changed()

    def _compare_run(self, method: str, params: Dict[str, Any]) -> Any:
        """One method on the slab of the analysis shown, in the worker
        thread (``tools.multimethod.analyze_like``, the clustering
        integration): the same slab (slab_override), ROI, curation,
        occupancy threshold, contour (2-opt from every start and the same
        contour guide), pick-boundary rule, and the slab's precisions and
        frames; DBSCAN takes eps / min samples from ``params`` (the
        analysis' own when absent), so the DBSCAN row is the analysis shown
        and DBSCAN 25/10 is the reference; randomization off for speed
        [ad hoc]."""
        from tools.multimethod import analyze_like
        a = self.analysis
        assert a is not None
        return analyze_like(a, a.x_slab, a.y_slab, a.z_slab, method, params)

    def _on_compare(self) -> None:
        """'Compare methods...': the dialog; only what it ticks runs, only
        when it says Run (D12). 'Use this method' saves the choice and
        re-runs the analysis with the first chosen method."""
        if self.analysis is None or self.analysis.x_slab is None:
            QtWidgets.QMessageBox.information(self, "Compare methods", "Run an analysis first.")
            return
        from tools.clustering_strategies import strategy_registry
        from tools.compare_analyses_ui import CompareAnalysesDialog
        panel = self.method_panel
        a = self.analysis

        def params_for(name: str) -> Dict[str, Any]:
            if name == "DBSCAN":
                # the analysis' own eps / min samples (DBSCAN 25/10, the reference, is added by tools.multimethod)
                return {"eps": float(a.eps_nm), "min_samples": int(a.min_samples)}
            if name == panel.current_method():
                return panel.current_params()
            info = strategy_registry().get(name)
            return {} if info is None else {k: info.defaults[k]["value"] for k in info.numeric_params()}

        nena, _words = self.links.nena()
        ws = self.working_scale
        dlg = CompareAnalysesDialog(self._compare_run, int(np.asarray(a.x_slab).size),
                                    panel.experimental(), panel.current_method(), params_for, parent=self,
                                    analysis=a, nena_nm=nena, c_label=ws.c_label(), rule=ws.rule, rule26=ws.rule26)
        dlg.events_radius_in_nena = ws.events_radius()           # B1: None at the default 3 NeNA
        self._compare_dialog = dlg
        accepted = dlg.exec_() == QtWidgets.QDialog.Accepted
        if dlg.comparison_result is not None and self.analysis is a:
            # every method compared joins the Perimeter arrangement tab, the double table and the export
            self.method_comparison = dlg.comparison_result
            self.method_comparison_files = dict(getattr(dlg, "written_tables", {}) or {})
            self._sync_integration()
        self._remember_compared(dlg)
        # F5: every run of the comparison as a row of the map, by the name the dialog shows (the reference 'DBSCAN
        # 25/10' is left out when it IS the analysis shown, which is drawn as itself)
        from tools import clustering_integration as _ci
        shown_is_reference = _ci.run_of(a).is_reference
        ran = {k: v for k, v in dict(getattr(dlg, "analyses", {}) or {}).items()
               if not (shown_is_reference and k == _ci.mm.REFERENCE_KEY)}
        if not accepted:
            # F5: closed without a choice: every method that ran stays on the map, each a row to tick
            self.show_methods(ran)
            return
        if dlg.overlay_method:
            # 'Overlay on the map': drawn over the analysis shown; the method in use does not change. The
            # methods' rows come too, unticked (the overlay draws the pair).
            self.show_methods(ran, tick=False)
            self.show_overlay(dlg.overlay_method, dlg.analyses.get(dlg.overlay_method))
            return
        if not dlg.chosen:
            self.show_methods(ran)
            return
        chosen = dlg.chosen[0]
        run = dlg.run_for(chosen)
        method = chosen if run is None else run.method
        compared = sorted({r.method for r in dlg.comparison_result.runs} if dlg.comparison_result is not None
                          else set(dlg.ran))
        if method == "DBSCAN" and run is not None:
            # a DBSCAN row (the reference, the analysis' own or the rule): its eps / min samples go to the strip
            for w in (self.spin_eps, self.spin_min):
                w.blockSignals(True)
            self.spin_eps.setValue(float(run.params.get("eps", a.eps_nm)))
            self.spin_min.setValue(int(run.params.get("min_samples", a.min_samples)))
            for w in (self.spin_eps, self.spin_min):
                w.blockSignals(False)
            panel.set_state("DBSCAN", {}, panel.experimental(), compared)
        else:
            panel.set_state(method, params_for(method) if run is None else dict(run.params),
                            panel.experimental() or method != "DBSCAN", compared)
        self._on_param_changed()

    def _store_c_label(self, value: float) -> None:
        """The Working scale panel's c_label into the one settings object (a window on its own has no store)."""
        if self.links.store is not None:
            self.links.store({"c_label": float(value)})

    def _remember_compared(self, dlg: Any) -> None:
        """Manual test of 2026-10-23: the methods that ran in 'Compare methods...' are saved as the compared list
        (settings.cluster_methods_compared, which the batch's 'Compare methods (slow)' box offers) whenever the
        dialog closes after a run -- closed, Overlay or Use this method -- not only with Use this method. The
        method shown is not changed and nothing is re-run."""
        res = getattr(dlg, "comparison_result", None)
        ran = sorted({str(r.method) for r in res.runs}) if res is not None else []
        if not ran:
            return
        panel = self.method_panel
        panel.set_state(panel.current_method(), panel.current_params(), panel.experimental(), ran)
        if self.links.store is not None:
            self.links.store({"cluster_methods_compared": ran})

    def _on_null_shortcut(self) -> None:
        """'Null (per axon)...' in the strip: the Perimeter arrangement tab, with its estimate; it runs from there
        (the estimate is read first: a null of a slow method takes hours)."""
        self.tabs.setCurrentWidget(self.arrangement_scroll)
        self._sync_integration()

    def _sync_integration(self) -> None:
        """The Working scale panel and the Perimeter arrangement tab follow the analysis shown: its working scale,
        the DBSCAN rule with the panel's c_label, the double table, and the runs a null can be run for (the
        analysis shown, then the methods of its last comparison)."""
        from tools import clustering_integration as ci
        a = self.analysis
        nena, words = self.links.nena()
        ws = getattr(self, "working_scale", None)
        if ws is None:
            return
        ws.show_scale(a, nena, words, self._rule_inputs())
        comp = self.method_comparison
        if comp is not None and (a is None or not comp.runs or comp.runs[0].analysis is None
                                 or not np.array_equal(np.asarray(comp.runs[0].analysis.x_slab),
                                                       np.asarray(a.x_slab))):
            comp = self.method_comparison = None
            self.method_comparison_files = {}
        ws.show_double_table(comp)
        runs = [] if a is None else [ci.run_of(a, axon_id=str(getattr(a, "source_name", "")))]
        if comp is not None:
            import dataclasses
            # the rule's runs under the name the user knows them by ("DBSCAN rule (c_label 14)")
            # every run, the failed / refused / cancelled ones too: the tab names them as not included (manual
            # test of 2026-10-23: Emitters with no imaging buffer vanished from the estimate without a word)
            runs += [dataclasses.replace(r, key=comp.labels.get(r.key, r.key)) for r in comp.runs]
        self.arrangement.set_context(a, runs, nena, ws.c_label())

    def _rule_inputs(self) -> Optional[Any]:
        """The localizations per cycle of the rule at the panel's radius (B1; the default radius: the cached one)."""
        ws = getattr(self, "working_scale", None)
        r = None if ws is None else ws.lpc_radius()
        return self.links.rule_inputs() if r is None else self.links.rule_inputs_at(r)

    def compute_lpc_depth(self) -> Tuple[Optional[Any], str]:
        """B3 (c): the localizations per cycle by depth of the analysis shown (the Working scale panel's button),
        shown in the panel and carried by the export."""
        a = self.analysis
        if a is None:
            return None, "no analysis"
        ws = self.working_scale
        prof, why = self.links.lpc_depth(a, ws.lpc_radius())
        self._lpc_depth = (a, ws.lpc_radius(), prof, why)
        ws.show_lpc_depth(prof, why)
        return prof, why

    def _lpc_depth_of_shown(self) -> Tuple[Optional[Any], str]:
        st = getattr(self, "_lpc_depth", None)
        ws = self.working_scale
        if st is None or st[0] is not self.analysis or st[1] != ws.lpc_radius():
            from tools.clustering_integration import LPC_DEPTH_NOT_HERE
            return None, ("not computed: press 'Localizations per cycle by depth' in Working scale (or run the "
                          "Batch, which computes it per file)" if self.analysis is not None else LPC_DEPTH_NOT_HERE)
        return st[2], st[3]

    def integration_columns(self) -> Optional[Dict[str, Any]]:
        """The v13 block of the axon table for the analysis shown (tools.clustering_integration.axon_columns): the
        file's NeNA, the panel's c_label, the localizations per cycle, the null of this analysis when it ran and
        the last comparison. None without an analysis."""
        from tools import clustering_integration as ci
        a = self.analysis
        if a is None:
            return None
        nena, words = self.links.nena()
        prof, why = self._lpc_depth_of_shown()
        ws = self.working_scale
        return ci.axon_columns(a, nena_nm=nena, nena_source=words, c_label=ws.c_label(),
                               locs_per_cycle=self._rule_inputs(), null=self.arrangement.outcome_for(a),
                               comparison=self.method_comparison,
                               comparison_files=self.method_comparison_files if self.method_comparison else None,
                               lpc_profile=prof, lpc_profile_note=why, lpc_radius_in_nena=ws.lpc_radius(),
                               events_radius_in_nena=ws.events_radius(),
                               membrane_deformation=self.arrangement.deformation())

    def show_overlay(self, method: str, other: Any) -> bool:
        """Draw ``method``'s run (``other``, an AxonAnalysis of the same slab from 'Compare methods') over the
        analysis shown, on the axon map (tools.method_overlay). False, with a message, when it cannot be drawn."""
        from tools.method_overlay import build_overlay
        if self.analysis is None or other is None:
            return False
        ov = build_overlay(self.analysis, other, method)
        if not ov.ok:
            QtWidgets.QMessageBox.information(self, "Overlay on the map", f"No overlay: {ov.note}.")
            return False
        self._overlay = ov
        for key in ("ov_a", "ov_b"):
            self.axon_map.layers.set_visible(key, True)
        self._draw_map()
        return True

    def show_methods(self, analyses: Dict[str, Any], tick: bool = True) -> bool:
        """F5: every method in ``analyses`` (method -> AxonAnalysis of this slab, from 'Compare methods...') and the
        analysis shown, each a row of the map's 'Methods on the map' group. ``tick``: the first METHODS_TICKED rows
        start ticked. False when there is nothing to add (no analysis, or no method but the one shown ran)."""
        from tools.method_overlay import build_method_layers
        if self.analysis is None or getattr(self.analysis, "x_slab", None) is None:
            return False
        others = {m: a for m, a in dict(analyses or {}).items()
                  if a is not None and m != getattr(self.analysis, "cluster_method", "DBSCAN")}
        if not others:
            return False
        ml = build_method_layers(self.analysis, analyses)
        self._methods = ml
        compared = [run for run in ml.runs if not run.is_shown]
        for run in ml.runs:
            # Manual test of 2026-10-23: the analysis shown is already on the map as its localizations and centres;
            # its own outline starts off (a row to tick), so the group adds only the COMPARED methods' outlines.
            on = tick and not run.note and not run.is_shown and compared.index(run) < METHODS_TICKED
            self.axon_map.layers.set_visible(run.key, bool(on))
        self._draw_map()
        return True

    def clear_methods(self) -> None:
        """'Clear the methods': the compared methods' rows leave the map."""
        self._methods = None
        self._draw_map()

    def _on_method_diagnostics(self) -> None:
        """'Method diagnostics...': each method's own diagnostic plot (tools.method_diagnostics)."""
        from tools.method_diagnostics import MethodDiagnosticsDialog
        ml = self._methods
        if ml is None or ml.base is not self.analysis:
            return
        dlg = MethodDiagnosticsDialog([(r.method, r.analysis) for r in ml.runs if not r.note], parent=self)
        self._diagnostics_dialog = dlg
        dlg.show()

    def clear_overlay(self) -> None:
        """'Clear the overlay': the second method's outlines leave the map."""
        self._overlay = None
        self._draw_map()

    def _on_slab_automatic(self) -> None:
        """Back to the automatic slab centre, and run the analysis."""
        if self.links.store is None or self.rerun_callback is None:
            return
        self.links.store({"slab": ("automatic", None)})
        self._rerun(main_peak_override_nm=None)

    def _on_typed_toggled(self, on: bool) -> None:
        if self.links.apply_typed is None:
            return
        if on:
            # The bounds are applied with "Apply"; they start from the slab in force (the fields already show it).
            bounds = self._current_slab_bounds()
            if bounds is not None and self.links.slab_state()[2] is None:
                self.edit_typed_min.setText(f"{bounds[0]:.1f}")
                self.edit_typed_max.setText(f"{bounds[1]:.1f}")
            return
        # Unticked: only a range that was applied is cleared. Ticking and
        # unticking without "Apply" changed nothing, so nothing is re-cut
        # (a re-cut would hand the Axoplasm panel a "new" selection and
        # drop its hand-set threshold).
        if self.links.slab_state()[2] is not None:
            self.links.apply_typed(None)
        self._sync_slab_widgets()

    def _on_typed_apply(self) -> None:
        if self.links.apply_typed is None:
            return
        try:
            a = float(self.edit_typed_min.text().strip())
            b = float(self.edit_typed_max.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Typed range", "Type both bounds as numbers, in nm.")
            return
        if not a < b:
            QtWidgets.QMessageBox.warning(
                self, "Typed range", "The lower bound must be below the upper one.")
            return
        self.links.apply_typed((a, b))
        self._sync_slab_widgets()

    def _on_rings(self) -> None:
        if self.rings_callback is not None:
            self.rings_callback()

    def _on_perimeter(self) -> None:
        """'Perimeter...': the perimeter density window, kept (one per results window) and raised when pressed
        again; its notes come back through ``set_perimeter_notes``."""
        if self.links.perimeter_inputs is None:
            return
        from tools import mps_perimeter_window as mpw
        w = self._perimeter_window
        if w is None:
            w = mpw.MPSPerimeterWindow(inputs_provider=self.links.perimeter_inputs,
                                       notes_callback=self.set_perimeter_notes, parent=self)
            self._perimeter_window = w
            self.register_tour(w, mpw.TOUR_ID)
        w.show()
        w.raise_()
        w.activateWindow()

    def _on_unanchored(self) -> None:
        if self.links.open_unanchored is not None:
            self.links.open_unanchored()

    def _cancel_unanchored(self) -> None:
        if self.links.cancel_unanchored is not None:
            self.links.cancel_unanchored()

    def show_side_progress(self, line: str, value: int, running: bool) -> None:
        """The automatic unanchored run's line: ``value`` 0..1000 (-1: no fraction), hidden a moment after it
        ends."""
        sp = self.side_progress
        self._side_active = bool(running)
        if running:
            if value >= 0:
                sp.bar.setRange(0, 1000)
                sp.bar.setValue(int(value))
            else:
                sp.bar.setRange(0, 0)
            sp.btn_cancel.setVisible(True)
            sp.show_line(line, cancelling=line.find("Cancelling") >= 0)
        else:
            sp.btn_cancel.setVisible(False)
            if not any(word in line for word in ("Cancelled", "Failed", "Not applied", "Not evaluated")):
                sp.setVisible(False)           # done: the discard column says it; no note takes the map's room
                return
            sp.show_line(line, cancelling=True)
            QtCore.QTimer.singleShot(CANCELLED_NOTE_MS, lambda: (None if self._side_running() else sp.setVisible(False)))

    def _side_running(self) -> bool:
        return bool(getattr(self, "_side_active", False))

    def register_tour(self, window: QtWidgets.QWidget, tour_id: str) -> None:
        """Give a window opened from here its guided tour (tools.mps_tour), when the program that owns this window
        has one: the long explanations live in the tour, not in fixed paragraphs."""
        owner: Any = self.parent()
        while owner is not None and getattr(owner, "tour", None) is None:
            owner = owner.parent() if hasattr(owner, "parent") else None
        tour = None if owner is None else getattr(owner, "tour", None)
        if tour is not None:
            try:
                tour.register_window(window, tour_id)
            except Exception:                          # noqa: BLE001 - the tour is a convenience
                pass

    def _perimeter_key(self) -> Any:
        a = self.analysis
        return None if a is None else (getattr(a, "source_name", ""), repr(getattr(a, "roi", None)))

    def set_perimeter_notes(self, lines: List[str]) -> None:
        """The notes of the perimeter density run, for the axon shown now: they join 'Analysis notes' until
        another axon is shown."""
        self._perimeter_notes = list(lines)
        self._perimeter_notes_key = self._perimeter_key()
        self._fill_warnings()

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
             PAPER_HEADER, "Note"])
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
        # Every row in sight without scrolling (manual test of SCI-3: the last rows, "CDF crossing" and "centres
        # outside the band", sat below the fold at 1366 x 768 and a new user did not know to scroll): one text line
        # per row, compact, instead of the style's ~31 px. Every cell is one line of text.
        vh = self.table.verticalHeader()
        row_px = self.table.fontMetrics().height() + 4
        vh.setMinimumSectionSize(row_px)
        vh.setDefaultSectionSize(row_px)
        vh.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        lay.addWidget(self.table, stretch=3)
        # If the user still makes the table shorter than its rows (a dragged divider, a smaller screen), it says
        # so in words: a scrollbar alone is easy to miss.
        self.lbl_more_rows = QtWidgets.QLabel("More rows below: scroll the table.")
        self.lbl_more_rows.setObjectName("lbl_more_rows")
        self.lbl_more_rows.setStyleSheet(f"color: {_C_TEXT_WARN}; font-weight: bold;")
        self.lbl_more_rows.setVisible(False)
        lay.addWidget(self.lbl_more_rows)
        bar = self.table.verticalScrollBar()
        bar.rangeChanged.connect(lambda *_a: self._update_more_rows())
        bar.valueChanged.connect(lambda *_a: self._update_more_rows())

        # The two columns leave together, in one row. A table holding the
        # two analyses of one axon as two rows is how a statistic over it
        # counts that axon twice.
        self.box_export = QtWidgets.QWidget()
        export = QtWidgets.QHBoxLayout(self.box_export)
        export.setContentsMargins(0, 0, 0, 0)
        self.label_export = QtWidgets.QLabel(
            "Both columns go into one row, with and without the discard: "
            "'Save axon...' above.")
        self.label_export.setWordWrap(True)
        export.addWidget(self.label_export)
        export.addStretch(1)
        self.box_export.setVisible(False)
        lay.addWidget(self.box_export)

        # The clusters cut by the Picasso pick (P-R59, docs/cut_clusters.md): their count, the switch that leaves
        # them out and the Pick boundary dialog, under the table that counts them (the strip keeps its two lines
        # at 1366 px, and the map its share of the height).
        from tools.mps_cut_ui import CutStrip

        self.cut_strip = CutStrip()
        self.cut_strip.exclude_toggled.connect(self._on_exclude_cut)
        self.cut_strip.pick_requested.connect(self._on_pick_boundary)
        self.cut_strip.chk_exclude.setEnabled(self.links.set_exclude_cut is not None)
        self.cut_strip.btn_pick.setEnabled(self.links.pick_boundary is not None)
        lay.addWidget(self.cut_strip)


        # The warnings leave the window (manual test 2, item 5): a button with their count, a preview in its
        # tooltip, and the whole list written to a .txt file on a click (never over an existing file). The
        # user reads them as "Analysis notes" (manual test of SCI-3: "Warnings" sounded like danger, a new
        # user would hesitate to press it); the code keeps calling them warnings.
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.btn_warnings = QtWidgets.QPushButton("Analysis notes (0)...")
        self.btn_warnings.setObjectName("btn_warnings")
        self.btn_warnings.clicked.connect(self._on_save_warnings)
        row.addWidget(self.btn_warnings)
        # F5c: check T4 fired after an Emitters run (precision under-reported): the suggested se_adjust_nm in
        # one click, then the analysis re-runs (hidden otherwise).
        self.btn_apply_precision = QtWidgets.QPushButton("Apply suggested correction and re-run")
        self.btn_apply_precision.setObjectName("btn_apply_precision")
        self.btn_apply_precision.setVisible(False)
        self.btn_apply_precision.clicked.connect(self._on_apply_precision)
        row.addWidget(self.btn_apply_precision)
        row.addWidget(self.btn_reset)
        row.addStretch(1)
        lay.addLayout(row)
        self._warning_lines: List[str] = []
        self._warning_count = 0

        return w

    def _update_more_rows(self) -> None:
        """The 'more rows below' line: shown while rows of the table lie below its visible part."""
        bar = self.table.verticalScrollBar()
        self.lbl_more_rows.setVisible(bar.maximum() > 0 and bar.value() < bar.maximum())

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
        self.axon_map.overlay_clear_requested.connect(self.clear_overlay)
        self.axon_map.methods_clear_requested.connect(self.clear_methods)
        self.axon_map.diagnostics_requested.connect(self._on_method_diagnostics)
        self.plot_splitter.addWidget(self.axon_map)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("plot_tabs")
        self.tabs.setCornerWidget(self.actions_box, QtCore.Qt.Corner.TopRightCorner)
        self.axial = AxialView(dark=self.dark)
        self.tabs.addTab(self.axial, "Axial")
        self.nn = NearestNeighboursPanel(
            dark=self.dark, bins=int(self.links.nn_bins), range_nm=self.links.nn_range,
            root_name=self.links.root_name, current=lambda: self._analysis_is_current(fresh=True))
        self.tabs.addTab(self.nn, "Nearest neighbours")

        self.plot_area = pg.PlotWidget()
        style_dark(self.plot_area)
        set_title(self.plot_area, AREA_TITLE)
        self.plot_area.setLabels(bottom="area [nm^2]", left="count")
        # The plot with its details panel under it (12.8): the median area
        # and effective radius with the reference values, read-only.
        self.area_tab = QtWidgets.QWidget()
        self.area_tab.setObjectName("area_tab")
        area_lay = QtWidgets.QVBoxLayout(self.area_tab)
        area_lay.setContentsMargins(0, 0, 0, 0)
        area_lay.setSpacing(2)
        area_lay.addWidget(self.plot_area, 1)
        self.details_area = DetailsPanel(object_name="details_area", collapsed=True)
        area_lay.addWidget(self.details_area)
        self.tabs.addTab(self.area_tab, "Cluster area")

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
        # The clustering integration: the per-axon null of every method compared, H1 / H2 and the regularity reading
        # (tools.arrangement_ui).
        from tools.arrangement_ui import TAB_TITLE, ArrangementPanel, WorkingScalePanel
        self.arrangement = ArrangementPanel(dark=self.dark)
        # The working scale of the analysis shown (A9 sec. 4.1; A5 inside it), the editable c_label and the DBSCAN
        # rule beside 25/10 (D2), at the top of the tab whose tests start at its marks (the left column keeps every
        # row of the results table in sight at 1366 x 728).
        self.working_scale = WorkingScalePanel()
        # the c_label kept in the settings (2026-10-29): the panel starts at it, and an edit is written back through
        # the store, so the next session and the Batch use it
        stored_c = self.links.c_label()
        if stored_c is not None:
            self.working_scale.spin_c_label.setValue(float(stored_c))
        self.working_scale.c_label_changed.connect(lambda _v: self._sync_integration())
        self.working_scale.c_label_changed.connect(self._store_c_label)
        # B1 / B3 (2026-11-07): the radii re-read the rule (and the arrangement's context); the lpc-by-depth button
        self.working_scale.radii_changed.connect(lambda: self._sync_integration())
        self.working_scale.lpc_depth_requested.connect(self.compute_lpc_depth)
        self.arrangement.set_header(self.working_scale)
        # in a scroll area: the tab's controls must not raise the tabs' minimum height and take the map's share
        self.arrangement_scroll = QtWidgets.QScrollArea()
        self.arrangement_scroll.setObjectName("arrangement_scroll")
        self.arrangement_scroll.setWidgetResizable(True)
        self.arrangement_scroll.setWidget(self.arrangement)
        self.arrangement_scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                              QtWidgets.QSizePolicy.Policy.Ignored)
        self.tabs.addTab(self.arrangement_scroll, TAB_TITLE)
        # The tabs' own minimum height (the tallest tab's, ~205 px) was larger than what is left under the map
        # on a 1366 x 728 window, so the splitter refused _apply_map_share and the map got 0.49 of the window
        # instead of MAP_MIN_OF_WINDOW. A floor that fits keeps the design: the map first, the tabs scroll.
        self.tabs.setMinimumHeight(TABS_FLOOR_PX)
        self.plot_splitter.addWidget(self.tabs)
        # The map keeps MAP_SHARE of the plot column through every show and
        # resize, until the user drags the divider (then it stays there).
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 1)
        self._split_done = False
        self.plot_splitter.splitterMoved.connect(self._on_split_moved)
        self.tabs.currentChanged.connect(self._guard_map_share)
        return self.plot_splitter

    def _on_split_moved(self, _pos: int, _index: int) -> None:
        if not getattr(self, "_applying_share", False):
            self._split_done = True

    def _on_left_moved(self, _pos: int, _index: int) -> None:
        if not getattr(self, "_applying_share", False):
            self._left_done = True

    def _apply_left_width(self) -> None:
        """The table column LEFT_SHARE of the window, at least LEFT_MIN_PX
        wide and never more than a third of it, until the user drags the
        divider: the map takes the rest."""
        if self._left_done or self._widened:
            return
        sizes = self.splitter.sizes()
        total = sum(sizes)
        if total <= 0 or len(sizes) != 2:
            return
        left = int(min(max(LEFT_MIN_PX, LEFT_SHARE * total), total / 3.0))
        if abs(sizes[0] - left) <= 2:
            return
        self._applying_share = True
        try:
            self.splitter.setSizes([left, total - left])
        finally:
            self._applying_share = False

    def _apply_map_share(self) -> None:
        self._apply_left_width()
        total = self.plot_splitter.height()
        if total <= 0:
            return
        top = int(total * MAP_SHARE)
        if total - top < TABS_MIN_PX:
            # a short screen: the tabs get their minimum, the map keeps its share of the window
            top = max(total - TABS_MIN_PX, int(MAP_MIN_OF_WINDOW * self.height()))
        self._applying_share = True
        try:
            self.plot_splitter.setSizes([top, total - top])
        finally:
            self._applying_share = False

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        # manual test of 2026-10-23: every informative text selectable (labels built since, too)
        from tools.copyable import make_labels_selectable
        make_labels_selectable(self)
        self._fit_strip()
        if not self._split_done:
            self._apply_map_share()
        if not self._restored:
            self._restored = True
            # Once the first layout has settled: sizes set before it are redistributed by the stretch factors.
            QtCore.QTimer.singleShot(0, self._restore_layout)

    def _restore_layout(self) -> None:
        """The dividers where the user left them last session (once, at the first show); a restored divider is
        the user's, and the automatic shares leave it alone."""
        self._applying_share = True
        try:
            for keeper in self._persistent:
                if not keeper.restore():
                    continue
                if keeper.splitter is self.splitter:
                    self._left_done = True
                elif keeper.splitter is self.plot_splitter:
                    self._split_done = True
                elif keeper.splitter is self.axon_map.splitter:
                    self.axon_map._user_split = True
        finally:
            self._applying_share = False

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._fit_strip()
        if not self._split_done:
            QtCore.QTimer.singleShot(0, self._apply_map_share)

    def _guard_map_share(self, *_args: Any) -> None:
        """The divider was not dragged by the user and the map has less than MAP_FLOOR_SHARE of the column (the
        tabs grew under it): give the map its share back."""
        if self._split_done or getattr(self, "_applying_share", False):
            return
        sizes = self.plot_splitter.sizes()
        total = sum(sizes)
        if total > 0 and sizes[0] < MAP_FLOOR_SHARE * total:
            QtCore.QTimer.singleShot(0, self._apply_map_share)

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
        if self._overlay is not None and self._overlay.base is not self.analysis:
            # the second method was run on the previous analysis' slab
            self._overlay = None
        if self._methods is not None and self._methods.base is not self.analysis:
            self._methods = None                       # F5: the same rule for every compared method
        self.comparison = self._ask_discard()
        self._read_selection()
        self._sync_controls()
        self._sync_more()
        self._sync_slab_widgets()
        self._sync_reset()
        self._sync_discard_widgets()
        self._sync_cut()
        self._fill_provenance()
        self._fill_table()
        self._fill_warnings()
        self._update_banner()
        self._sync_integration()
        self._draw_plots()

    def _sync_cut(self) -> None:
        """The Pick item of the strip: the analysis' count, or what the
        next analysis will do."""
        exclude, words = self.links.cut_state()
        shown = self._display_analysis()
        self.cut_strip.sync(None if shown is None else getattr(shown, "cut", None), exclude,
                            "" if shown is not None else words)

    def _on_exclude_cut(self, on: bool) -> None:
        if self.links.set_exclude_cut is not None:
            self.links.set_exclude_cut(bool(on))

    def _on_pick_boundary(self) -> None:
        if self.links.pick_boundary is not None:
            self.links.pick_boundary()

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
        self._sync_slab_widgets()
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

    def _analysis_is_current(self, fresh: bool = False) -> bool:
        """Whether the analysis describes the current selection (a window
        on its own: always). ``fresh`` asks the main window again instead of
        trusting the last redraw (a save must never write an analysis of
        another file or selection)."""
        sel = self.links.selection() if fresh else self._sel
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
        out = {PLOT_MAP: self.axon_map.plot,
               PLOT_AXIAL: self.axial.plot,
               PLOT_AREA: self.plot_area,
               PLOT_NN: self.nn.plot_nn,
               PLOT_CDF: self.nn.plot_cdf,
               PLOT_SCATTER: self.plot_scatter}
        # the clustering integration: the null's H(r) and the regularity plots (F6's export, as every plot)
        out.update(self.arrangement.figure_plots())
        return out

    def _tab_of(self, name: str) -> Optional[QtWidgets.QWidget]:
        """The tab a plot sits in (None: the map, always visible)."""
        if name in self.arrangement.figure_plots():
            self.arrangement.show_plot(name)
            return self.arrangement_scroll
        return {PLOT_AXIAL: self.axial, PLOT_AREA: self.area_tab, PLOT_NN: self.nn,
                PLOT_CDF: self.nn, PLOT_SCATTER: self.plot_scatter}.get(name)

    def _apply_background(self) -> None:
        """Style every plot for the background in use, and redraw."""
        for plot in (self.plot_area, self.plot_scatter):
            (style_dark if self.dark else style_light)(plot)
        self.axon_map.set_dark(self.dark)
        self.axial.set_dark(self.dark)
        self.nn.set_dark(self.dark)
        self.arrangement.set_dark(self.dark)
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
        # F6: each plot's title, editable in the dialog; a legend of the rows drawn for the plots with a layer panel
        titles = {name: figure_export.plain_title(plot.getPlotItem()) for name, plot in plots.items()}
        request = ask(list(plots), suggested, parent=self, offer_white=white, titles=titles,
                      legend=(set(LEGEND_PLOTS) | set(self.arrangement.figure_plots())) & set(plots))
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
            legend = self.legend_entries(request.plot) if request.legend else None
            # the tab's own corner legends step aside: the figure gets Export image's legend only
            with self.arrangement.legends_hidden():
                written = figure_export.write(plot.getPlotItem(), request, legend=legend)
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

    def legend_entries(self, name: str) -> List[Any]:
        """F6: the legend of plot ``name`` as drawn now: (label, sample) of every row ticked in its layer panel."""
        from tools.mps_axon_map import legend_entries
        if name in self.arrangement.figure_plots():
            return self.arrangement.legend_entries(name)
        if name == PLOT_MAP:
            return legend_entries(self.axon_map._layers, self.axon_map.layers, self.dark)
        if name == PLOT_AXIAL:
            return legend_entries(self.axial._layers, self.axial.layers, self.dark)
        if name == PLOT_NN:
            return legend_entries(self.nn._layers, self.nn.layers, self.dark, keep=lambda k: k.startswith("nn_"))
        return []

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

            params = self.links.params() if self.links.store is not None else None
            if params is not None:
                # The one value of each parameter (design 6.2): what the strip
                # set, which every consumer reads - the analysis shown is of
                # those values unless the banner says otherwise.
                eps, minimum, half, maha = params
                self.spin_half.setValue(float(half))
                self.spin_eps.setValue(float(eps))
                self.spin_min.setValue(int(minimum))
                self.spin_maha.setValue(float(maha))
            else:
                self.spin_half.setValue(a.slab_half_width_nm)
                self.spin_eps.setValue(a.eps_nm)
                self.spin_min.setValue(int(a.min_samples))
                # From the analysis, not from its occupancy: an analysis with
                # too few clusters to measure occupancy still ran with a
                # threshold, and reading it back from the widget is what let
                # 0.1 through.
                self.spin_maha.setValue(a.mahalanobis_threshold)
            if self.links.method_state is not None:
                m_name, m_params, m_exp, m_cmp = self.links.method_state()
                self.method_panel.set_state(m_name, m_params, m_exp, m_cmp)
            else:
                self.method_panel.set_state(a.cluster_method, dict(a.cluster_params), False,
                                            list(a.cluster_methods_compared))
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
        # One dictionary of source words, shared with the main window's
        # Measurement panel (design 12.7): the text is today's, byte for byte.
        src = reg.pixel_source_words(a.pixel_size_source)
        colour = (AXIS_FG_LIGHT
                  if a.pixel_size_source in reg.PIXEL_SOURCES_FROM_FILE
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
                item.setToolTip(KS_TIP + "\n" + METHODS_TIP + (f"\n\n{own}" if own else ""))
            elif name == _MC_ROW:
                item.setToolTip(MC_P_TIP + "\n" + METHODS_TIP + (f"\n\n{own}" if own else ""))
            elif name == _REGULARITY_ROW:
                item.setToolTip(REGULARITY_TIP + (f"\n\n{own}" if own else ""))
            elif name in _RAND_ROWS:
                item.setToolTip(METHODS_TIP + (f"\n\n{own}" if own else ""))
            elif name == _FOREIGN_ROW:
                item.setToolTip(FOREIGN_TIP + (f"\n\n{own}" if own else ""))
            elif name == _DELTA_Z_ROW:
                item.setToolTip(DELTA_Z_TIP + (f"\n\n{own}" if own else ""))
            elif name == _CROSSING_ROW:
                item.setToolTip(CROSSING_TIP + "\n" + reg.tooltip("randomization.cdf_band_alpha")
                                + (f"\n\n{own}" if own else ""))

    @staticmethod
    def _italic() -> QtGui.QFont:
        f = QtGui.QFont()
        f.setItalic(True)
        return f

    def _fill_warnings(self) -> None:
        """The warnings of the analysis shown (and of the other two columns): counted on the button, previewed in
        its tooltip, written by it (manual test 2, item 5)."""
        messages: List[str] = []
        if self._display_analysis() is None:
            lines = [NO_ANALYSIS + "."]
        else:
            assert self.analysis is not None
            messages = list(self.analysis.warnings)
            if self._perimeter_notes and self._perimeter_notes_key == self._perimeter_key():
                messages += [n for n in self._perimeter_notes if n not in messages]
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
            lines = list(messages) if messages else ["No analysis notes."]
        self._warning_lines = lines
        self._warning_count = len(messages)
        n = len(messages)
        self.btn_warnings.setText(f"Analysis notes ({n})...")
        self.btn_warnings.setEnabled(n > 0)
        self.btn_warnings.setStyleSheet(f"color: {_C_TEXT_WARN}; font-weight: bold;" if n else "")
        if n:
            preview = [("- " + w if len(w) <= WARNINGS_PREVIEW_CHARS
                        else "- " + w[:WARNINGS_PREVIEW_CHARS].rstrip() + " ...")
                       for w in messages[:WARNINGS_PREVIEW]]
            more = n - len(preview)
            tip = "\n".join(preview) + (f"\n... and {more} more." if more > 0 else "")
            self.btn_warnings.setToolTip(
                tip + "\n\nClick to save every note, in full, as a .txt file (an existing file is never "
                "overwritten: a new name is chosen).")
        else:
            self.btn_warnings.setToolTip(lines[0] if lines else "")
        self._sync_precision_button()

    def suggested_se_adjust_nm(self) -> Optional[float]:
        """The total se_adjust_nm check T4 suggests for the analysis shown (Emitters, precision under-reported),
        else None."""
        a = self._display_analysis()
        em = None if a is None else getattr(a, "emitter_readouts", None)
        return None if em is None else em.suggested_se_adjust_nm

    def _sync_precision_button(self) -> None:
        value = self.suggested_se_adjust_nm()
        show = value is not None and self.rerun_callback is not None
        self.btn_apply_precision.setVisible(show)
        if show:
            self.btn_apply_precision.setToolTip(
                f"Check T4: NeNA / median reported precision > 1.2, the precisions look under-reported.\n"
                f"Sets se_adjust_nm = {value:.1f} nm (NeNA - median precision) in the Emitters parameters\n"
                f"and re-runs the analysis (slow). docs/emitters.md")

    def _on_apply_precision(self) -> None:
        """'Apply suggested correction and re-run': se_adjust_nm = NeNA - median precision, then the re-run."""
        value = self.suggested_se_adjust_nm()
        if value is None or self.rerun_callback is None:
            return
        self.method_panel.set_param_value("se_adjust_nm", float(value))
        self._on_method_changed()

    def warning_lines(self) -> List[str]:
        """What the warnings list used to show, line by line ("No warnings." / the empty state included)."""
        return list(self._warning_lines)

    def _warnings_text(self) -> str:
        a = self._display_analysis()
        head = [f"MPS analysis notes: {os.path.basename(a.source_name) if a is not None else ''}",
                _strip(self.lbl_provenance.text()).replace("<br>", " | "),
                f"{self._warning_count} note(s)", ""]
        body = [f"{i}. {w}" for i, w in enumerate(self._warning_lines, 1)]
        return "\n".join(head) + "\n" + "\n\n".join(body) + "\n"

    def _on_save_warnings(self) -> None:
        """Write the warnings as a .txt file; never over an existing one."""
        if not self._warning_count:
            return
        a = self._display_analysis()
        source = a.source_name if a is not None else self.links.root_name()
        stem = os.path.splitext(os.path.basename(str(source or "axon")))[0] or "axon"
        parent = os.path.dirname(str(source)) if source else ""
        folder = parent if parent and os.path.isdir(parent) else os.getcwd()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the analysis notes", os.path.join(folder, f"{stem}_analysis_notes.txt"), "Text (*.txt)")
        if not path:
            return
        if os.path.splitext(path)[1].lower() != ".txt":
            path += ".txt"
        try:
            written = write_text_file(path, self._warnings_text())
        except OSError as error:
            QtWidgets.QMessageBox.critical(self, "Analysis notes not saved",
                                           f"Could not write the analysis notes:\n\n{error}")
            return
        renamed = ("" if os.path.normcase(written) == os.path.normcase(path)
                   else f"\n\n{os.path.basename(path)} already existed and was left as it is.")
        QtWidgets.QMessageBox.information(
            self, "Analysis notes saved", f"{self._warning_count} note(s) written to\n{written}{renamed}")

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
        note = [] if sel is None or not sel.guide_note else [sel.guide_note]
        if not self._stale or sel is None:
            if sel is not None and sel.rerun_failed:
                # A failed re-run with the same selection (eps, min samples,
                # Mahalanobis): the values shown are not the analysis' (6.3).
                self.banner.set_state([RERUN_FAILED] + note, offer_run=run)
                return
            self.banner.set_state(note)
            return
        lines = analysis_stale_lines(analysed_words=sel.analysed_words or "as analysed",
                                     current_words=sel.current_words or "another one",
                                     discard_dropped=sel.discard_dropped,
                                     cut_only=bool(sel.analysed_roi_words == sel.roi_words
                                                   and sel.analysed_cut != sel.cut))
        if sel.rerun_failed:
            lines.insert(0, RERUN_FAILED)
        self.banner.set_state(lines + note, offer_show=True, showing_current=False, offer_run=run)

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
        if self.busy():
            return
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
            selection_n=None if sel is None else sel.n_locs,
            overlay=(self._overlay if (a is not None and self._overlay is not None and self._overlay.base is a)
                     else None),
            methods=(self._methods if (a is not None and self._methods is not None and self._methods.base is a)
                     else None),
            fusion=self._fusion_state(a))

    def _fusion_state(self, a: Any) -> Any:
        """The event layer's 'probable fusion' marks of the analysis drawn (tools.mps_axon_map_layers.fusion_state),
        computed once per analysis and c_label."""
        if a is None:
            return None
        nena, _words = self.links.nena()
        c = self.working_scale.c_label() if hasattr(self, "working_scale") else 14.0
        key = (id(a), nena, c)
        if self._fusion_cache[0] == key:
            return self._fusion_cache[1]
        state = L.fusion_state(a, nena, c)
        self._fusion_cache = (key, state)
        return state

    def _draw_map(self) -> None:
        self.axon_map.set_inputs(self._map_inputs())
        # the layer panel's group titles are rebuilt at each drawing: selectable again (tools.copyable)
        from tools.copyable import make_labels_selectable
        make_labels_selectable(self.axon_map)
        a = self._display_analysis()
        sel = self._sel
        if a is not None:
            shown = self._shown()
            roi = "" if sel is None else (sel.analysed_roi_words or sel.roi_words)
            cut = None if sel is None else (sel.analysed_cut or sel.cut)
            line2 = L.provenance_line2(shown, roi_words=roi, cut=cut,
                                       given_after_cut=bool(sel is not None and sel.analysed_after_cut),
                                       z_words=self._z_words()[0] or None)
        else:
            line2 = L.provenance_line2(None, roi_words="" if sel is None else sel.roi_words,
                                       cut=None if sel is None else sel.cut, z_words=self._z_words()[0] or None)
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
                rings=ms, rings_reason=ms_reason or prev.rings_reason, view=prev.view,
                z_words=self._z_words()[0], z_magnification=self._z_words()[1])
        else:
            inp = L.AxialInputs(
                z_roi=None if sel is None else sel.z_roi, analysis=None,
                fit=None if sel is None else sel.fit,
                cut=None if sel is None else sel.cut,
                cut_from_histogram_mode=bool(sel is not None and sel.cut_from_histogram_mode),
                ch2_z=None if sel is None else sel.ch2_z,
                ch2_reason="no channel 2 is loaded" if sel is None else sel.ch2_reason,
                ch2_shown=prev.ch2_shown, rings=ms, rings_reason=ms_reason or prev.rings_reason,
                view=prev.view, fit_reason="the mixture could not be fitted" if sel is None else sel.fit_reason,
                z_words=self._z_words()[0], z_magnification=self._z_words()[1])
        self.axial.set_inputs(inp)

    def _z_words(self) -> Tuple[str, Optional[float]]:
        """The measurement's z calibration words and m (WindowLinks.z_calibration); ("", None) when unknown."""
        try:
            words, m = self.links.z_calibration()
        except Exception:  # noqa: BLE001 - a display: no words is today's text
            return "", None
        return str(words or ""), (None if m is None else float(m))

    def refresh_z_calibration(self) -> None:
        """Redraw what names the z calibration (the provenance line 2, the axial title and details) after the
        Measurement panel's inputs changed; nothing is re-run."""
        self._draw_map()
        self._draw_axial()

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
        self.details_area.set_rows(L.area_details(a))
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
                label=f"median {med:,.0f} nm^2",
                labelOpts={"position": 0.9, "color": role("summary"), "anchors": [(1, 0.5), (1, 0.5)]}))
        # The published value, in the paper role: the line itself is the
        # badge (P5), and its label names the publication.
        ref = float(reg.default("reference.cluster_area_nm2"))
        self.plot_area.addItem(pg.InfiniteLine(
            pos=ref, angle=90,
            pen=pg.mkPen(role("paper"), width=2, style=QtCore.Qt.DashLine),
            label=f"Gazal 2026 (preprint v1): {ref:,.0f} nm^2",
            labelOpts={"position": 0.75, "color": role("paper"), "anchors": [(0, 0.5), (0, 0.5)]}))

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
        """A strip value changed: write it into the one settings object,
        then re-run with today's keyword arguments (rule K, design 6.3: this
        is the window's re-run of before stage 2, moved, not rewritten)."""
        if self.rerun_callback is None:
            return
        self._store_values(reset=getattr(self, "_resetting", False))
        if self.analysis is None:
            # No analysis yet (B10): the value is stored, and the banner
            # offers "Run the MPS analysis"; a focus-out runs nothing.
            self._update_banner()
            return
        peak = self.combo_peak.currentData()
        mode, centre, typed = self.links.slab_state()
        if (self.links.store is not None and typed is None
                and getattr(self.analysis, "slab_source", "") == "range typed"):
            # The analysis shown ran on a typed range that is no longer
            # applied: its "current slab centre" is the typed range's, not
            # the slab's now. The re-run takes the slab the mode gives.
            peak = centre if mode == "component" else None
        self._rerun(main_peak_override_nm=(None if peak is None else float(peak)))

    def _sync_more(self) -> None:
        """The randomization and occupancy "More" lines say the values the analysis shown ran with (SCI-3)."""
        from tools.mps_randomization_ui import analysis_values, more_values

        a = self.analysis
        self.more_randomization.set_values(more_values(analysis_values(a)))
        self.more_occupancy.set_values(
            {} if a is None else {"occupancy.sigma_cap_fraction": float(a.sigma_cap_fraction)})

    def _rerun_with(self, extra: Dict[str, Any]) -> None:
        """Re-run with the strip's values and ``extra`` keywords (the randomization's or the sigma cap's)."""
        if self.analysis is None:
            self._update_banner()
            return
        peak = self.combo_peak.currentData()
        self._rerun(main_peak_override_nm=(None if peak is None else float(peak)), **extra)

    def _on_edit_randomization(self) -> None:
        """The randomization dialog: its values go into the one store and the analysis runs again with them."""
        from tools.mps_randomization_ui import (
            RandomizationDialog, analysis_values, analyze_kwargs, values_from_store,
        )

        if self.rerun_callback is None:
            return
        dialog = RandomizationDialog(analysis_values(self.analysis), self)
        if dialog.exec_() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        if self.links.store is not None:
            self.links.store({"randomization_params": values})
        kw = analyze_kwargs(values)
        # every keyword, so the run is these values whatever the store's defaults
        full = values_from_store(values)
        from tools.mps_randomization import RandomizationParams

        kw.setdefault("n_randomizations", int(full["n_iterations"]))
        kw.setdefault("random_seed", int(full["seed"]))
        kw.setdefault("randomization_params", RandomizationParams.from_dict(full))
        self._rerun_with(kw)

    def _on_edit_occupancy(self) -> None:
        """The sigma-cap dialog: the value goes into the one store and the analysis runs again with it."""
        from tools.mps_randomization_ui import SigmaCapDialog

        if self.rerun_callback is None:
            return
        cap = None if self.analysis is None else float(self.analysis.sigma_cap_fraction)
        dialog = SigmaCapDialog(cap, self)
        if dialog.exec_() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        value = float(dialog.values()["occupancy_sigma_cap_fraction"])
        if self.links.store is not None:
            self.links.store({"occupancy_sigma_cap_fraction": value})
        self._rerun_with({"sigma_cap_fraction": value})

    def _rerun(self, main_peak_override_nm: Optional[float], **extra: Any) -> None:
        if self.rerun_callback is None:
            return
        self._run_analysis(
            dict(
                **extra,
                main_peak_override_nm=main_peak_override_nm,
                slab_half_width_nm=float(self.spin_half.value()),
                eps_nm=float(self.spin_eps.value()),
                min_samples=int(self.spin_min.value()),
                **self._method_kwargs(),
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
            ),
            "Analysis failed", "Could not re-run the analysis with these parameters")

    # ---------------- F5d: a slow method runs in the background ----------

    def busy(self) -> bool:
        """True while an analysis runs in the background for this window."""
        return self._run is not None

    def _run_analysis(self, kw: Dict[str, Any], fail_title: str, fail_text: str) -> None:
        """Re-run with ``kw``: in the background when the method is slow and the main window gave the hook,
        else here and now (DBSCAN and the fast methods: the path of before F5d, unchanged)."""
        if self.rerun_callback is None or self.busy():
            return
        from tools.cluster_method_ui import is_slow
        if self.background is not None and (is_slow(str(kw.get("cluster_method", "DBSCAN")))
                                            or (self.big_run is not None and self.big_run())):
            self.start_background(kw, fail_title, fail_text)
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.analysis = self.rerun_callback(**kw)
        except Exception as exc:                      # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, fail_title, f"{fail_text}:\n\n{exc}")
            # The banner says what is shown was computed with the previous
            # values (design 6.3).
            self.refresh()
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.refresh()

    def _run_controls(self) -> List[QtWidgets.QWidget]:
        """Everything that would start another analysis (or write a table of the one about to change)."""
        # F4: the strip, except the clustering method's collapse / expand controls (usable during a run)
        from tools.mps_background_run import controls_except
        keep = [self.method_panel.btn_collapse, self.method_bar]
        return [*controls_except(self.strip_scroll.widget(), keep), self.banner, self.btn_rings, self.btn_export,
                self.btn_apply_precision, self.btn_reset, self.cut_strip]

    def start_background(self, kw: Dict[str, Any], fail_title: str = "Analysis failed",
                         fail_text: str = "Could not re-run the analysis with these parameters") -> bool:
        """Prepare (GUI thread: the dialogs before a run show here), then run the analysis in a worker thread.
        False when nothing started (a run is already going, no hook, or preparing failed and said why)."""
        if self.background is None or self.busy():
            return False
        from tools.mps_background_run import BackgroundRun, disabled_while
        try:
            prepared = self.background(**kw)
        except Exception as exc:                      # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, fail_title, f"{fail_text}:\n\n{exc}")
            self.refresh()
            return False
        if prepared is None:
            # nothing can run (no ROI, no imaging buffer, ...): the main window said why; the banner says the
            # analysis shown is the previous one
            self.refresh()
            return False
        run = BackgroundRun(prepared, parent=self)
        run.tick.connect(lambda line: self.run_progress.show_line(line, cancelling=run.cancelled()))
        run.progress.connect(self.run_progress.set_fraction)
        run.tick.connect(lambda line: self.method_bar.set_status(line, True, cancelling=run.cancelled()))
        run.ended.connect(lambda status, value, tb: self._on_run_ended(run, status, value, tb, fail_title,
                                                                       fail_text))
        self._run = run
        self._progress_timer.stop()
        self.run_progress.btn_cancel.setVisible(True)
        self._restore_controls = disabled_while(self._run_controls())
        run.start()
        return True

    def cancel_background(self) -> None:
        """Cancel: the run stops at its next check; whatever it returns is not used."""
        if self._run is not None:
            self._run.cancel()

    def _hide_progress(self) -> None:
        if not self.busy():
            self.run_progress.setVisible(False)

    def _on_run_ended(self, run: Any, status: str, value: Any, tb: str, fail_title: str, fail_text: str) -> None:
        """GUI thread: the worker has finished. Store the result, or say why there is none."""
        from tools import emitters_layer as el
        from tools import mps_cancel
        from tools import mps_perimeter_cost as cost
        from tools.mps_background_run import elapsed_words
        if run is not self._run:
            return
        self._run = None
        self.method_bar.set_status("", False)
        run.deleteLater()        # its worker is not its child: the module keeps that one until it is joined
        restore, self._restore_controls = self._restore_controls, None
        if restore is not None:
            restore()
        took = elapsed_words(run.elapsed_s())
        if status == "done":
            self.run_progress.setVisible(False)
            try:
                self.analysis = run.prepared.finish(value)
            except Exception as exc:                  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, fail_title, f"{fail_text}:\n\n{exc}")
            self.refresh()
            return
        err = value if isinstance(value, BaseException) else mps_cancel.AnalysisCancelled()
        try:
            run.prepared.fail(err)
        except Exception:                             # noqa: BLE001 - the window must stay usable
            pass
        user_stop = isinstance(err, (mps_cancel.AnalysisCancelled, cost.PerimeterCancelled, el.EmittersCancelled))
        if status == "cancelled" or user_stop:
            words = "Cancelled." if isinstance(err, mps_cancel.AnalysisCancelled) else str(err)
            self.run_progress.show_line(f"{words} After {took}; nothing it computed is used: the analysis shown is "
                                        f"the previous one.", cancelling=True)
            self.run_progress.btn_cancel.setVisible(False)
            self._progress_timer.start(CANCELLED_NOTE_MS)
        else:
            self.run_progress.setVisible(False)
            QtWidgets.QMessageBox.critical(self, fail_title, f"{fail_text}:\n\n{err}")
        self.refresh()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        """Closing while a run is in the background cancels it: the window only hides (the main window keeps it),
        the worker is held until it ends (tools.mps_background_run, review A1) and its result is not used."""
        self.cancel_background()
        super().closeEvent(event)

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
        self._store_values()
        peak = self.combo_peak.currentData()
        self._run_analysis(
            dict(
                main_peak_override_nm=(None if peak is None else float(peak)),
                slab_half_width_nm=float(self.spin_half.value()),
                eps_nm=float(self.spin_eps.value()),
                min_samples=int(self.spin_min.value()),
                **self._method_kwargs(),
                dbcv_threshold=DEFAULT_DBCV_THRESHOLD,
                mahalanobis_threshold=float(self.spin_maha.value()),
                run_randomization=bool(self.chk_random.isChecked()),
                contour_guide=(path if answer == APPLY else None),
            ),
            "Contour", "Could not re-run the analysis along that path")

    def _on_reset(self) -> None:
        """Reset to defaults (design 12.5): the same four values as before
        stage 2, read from the same constants (which the registry reads
        too); the slab centre, a typed range, the randomization switch and
        the drawn contour are kept. The values are saved (B12)."""
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
        self._resetting = True
        try:
            self._on_param_changed()
        finally:
            self._resetting = False

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
