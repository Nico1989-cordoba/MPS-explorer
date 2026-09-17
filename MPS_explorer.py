# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:03:00 2021

@author: Lucia Lopez

GUI for MPS x,y,z data exploration and quick analysis

conda command for converting QtDesigner file to .py:
pyuic5 -x data_explorer.ui -o data_explorer.py
    
"""

import os
import sys
from typing import Optional, Tuple, List, Dict, Union, Any
from pathlib import Path
import logging
import traceback

cdir = os.getcwd()
os.chdir(cdir)

import ctypes
import h5py as h5
import pandas as pd
from tkinter import Tk, filedialog
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import tools.utils as utils
import tools.clustering as clustering
from tools.clustering_strategies import create_clustering_strategy, AutoClusteringStrategy
from tools.parallel_clustering import create_parallel_clustering_manager
from tools.parameter_cache import create_parameter_cache
import hdbscan

# --- Gazal et al. (2026) per-axon MPS analysis -------------------------------
# Automatic replication of the paper's per-axon parameters. These modules are
# Qt-free so the whole pipeline can be run and validated headlessly.
from tools.cluster_quality import (
    CircularROI,
    PolygonROI,
    SquareROI,
    points_in_polygon,
    points_in_roi,
)
from tools.mps_analysis import analyze_axon
from tools.mps_periodicity import fit_z_periodicity
from tools import mps_io
from tools.mps_picasso_tools import PicassoTools
from tools.mps_settings import load_settings, save_settings

# Import logging configuration
from logging_config import setup_logging, get_logger, get_log_filename

# Import configuration loader
from config_loader import load_config, get_roi_config, get_histogram_config, get_visualization_config


import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow, QApplication
import data_explorer


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================
# All application constants are now loaded from config.yaml (or config.json)
# This enables easy tuning of parameters without modifying code.
# Environment variable overrides supported via MPS_* prefix (e.g., MPS_ROI_DIAMETER_SCALE_FACTOR=1.5)

_config = load_config()
_roi_config = _config.get('roi', {})
_hist_config = _config.get('histogram', {})
_viz_config = _config.get('visualization', {})

# --- ROI Rendering Configuration ---
ROI_DIAMETER_SCALE_FACTOR = _roi_config.get('diameter_scale_factor', 1.3)
ROI_EXTENT_DIVISOR = _roi_config.get('extent_divisor', 10)
CIRCULAR_ROI_COLOR = tuple(_roi_config.get('color_rgb', [255, 0, 0]))
ROI_ZORDER = _roi_config.get('z_order', 10)

# --- Histogram Configuration ---
DEFAULT_KNN_BINS = _hist_config.get('default_knn_bins', 30)
MAX_LATERAL_DISTANCE_NM = _hist_config.get('max_lateral_distance_nm', 800)
HISTOGRAM_2D_BINS = _hist_config.get('bins_2d', 400)
HISTOGRAM_Z_BINS = _hist_config.get('bins_z', 500)

# --- Visualization Configuration ---
NOISE_POINT_SIZE = _viz_config.get('point_size_noise', 3)
GOOD_CLUSTER_POINT_SIZE = _viz_config.get('point_size_good_cluster', 5)
CLUSTER_CENTROID_POINT_SIZE = _viz_config.get('point_size_centroid', 10)


# Windows-only: set an explicit AppUserModelID so the taskbar icon
# matches the window icon instead of the generic Python icon.
# Wrapped in a platform check so the application can be launched on
# Linux or macOS without an AttributeError on ctypes.windll.
# See https://stackoverflow.com/questions/1551605 for background.
if sys.platform == "win32":
    myappid = "MPS-Explorer.1.0"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class MPS_explorer(QtWidgets.QMainWindow):
    
    def __init__(self, *args, **kwargs):
        """
        Initialize MPS Explorer GUI application.

        Sets up the PyQt5 user interface, connects UI elements to their corresponding
        methods, initializes color schemes for visualization, and defensively initializes
        all instance attributes to None to prevent AttributeErrors when methods are
        called out of order (e.g., clicking "cluster" before loading a file).

        The application supports two-channel SMLM data analysis with ROI selection,
        DBSCAN clustering, and k-nearest neighbor distance analysis.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to the parent QMainWindow.__init__().
        """
        super().__init__(*args, **kwargs)

        # ===== LOGGING INITIALIZATION =====
        # Configure logging with file output in logs/ directory
        log_file = Path("logs") / get_log_filename("mps_explorer")
        self.logger = setup_logging(log_level="INFO", log_file=str(log_file))
        self.logger.info("=" * 80)
        self.logger.info("MPS Explorer Application Started")
        self.logger.info("=" * 80)

        # ===== PHASE 4: PARAMETER CACHING INITIALIZATION =====
        # Initialize parameter cache for Phase 1 optimization
        cache_dir = Path.cwd() / "cache"
        self.param_cache = create_parameter_cache(
            cache_dir=str(cache_dir),
            max_entries=100,
            similarity_threshold=0.95,
            logger=self.logger
        )
        self.logger.debug("Parameter cache initialized (Phase 4)")

        self.ui = data_explorer.Ui_MainWindow()
        self.ui.setupUi(self)
        self.logger.debug("UI setup complete")
        self._make_content_scrollable()
        self._build_analysis_toolbar()

        # Define initial directory
        self.initialDir = "Desktop"  # You can set the initial directory here
        self.logger.debug(f"Initial directory: {self.initialDir}")
        
        # File Formats
        fileformat_list = ["Picasso hdf5", "ThunderStorm csv", "custom csv"]
        self.fileformat = self.ui.comboBox_fileformat
        self.fileformat.addItems(fileformat_list)
        self.fileformat_2 = self.ui.comboBox_fileformat_2
        self.fileformat_2.addItems(fileformat_list)  # Reuse the same list for second channel

        # Algorithm Selection (Manual control over DBSCAN vs HDBSCAN)
        self.algorithm_selector = self.ui.comboBox_algorithm
        self.algorithm_selector.currentTextChanged.connect(self.on_algorithm_changed)
        self.logger.debug("Algorithm selector initialized (Auto/DBSCAN/HDBSCAN)")

        # Connect Buttons to Methods
        self.ui.pushButton_browsefile.clicked.connect(lambda:self.select_file(1))
        self.ui.pushButton_browsefile_2.clicked.connect(lambda:self.select_file(2))
        self.ui.pushButton_scatter.clicked.connect(self.scatterplot)
        self.ui.pushButton_zrange.clicked.connect(self.update_ROI)
        self.ui.pushButton_savexyzROI.clicked.connect(lambda: self.savexyzROI(1))
        self.ui.pushButton_savexyzROI_2.clicked.connect(lambda: self.savexyzROI(2))
        self.ui.pushButton_savedistdata.clicked.connect(self.savedistdata)
        self.ui.pushButton_clusterch1.clicked.connect(lambda:self.cluster(1))
        self.ui.pushButton_clusterch2.clicked.connect(lambda:self.cluster(2))
        # Bad clusters are now removed automatically (edge-touching + DBCV,
        # see tools.cluster_quality), so this button no longer performs a
        # manual removal step. It is repurposed as the entry point to the
        # per-axon MPS analysis panel; the .ui file is left untouched so the
        # window layout the user built is preserved.
        self.ui.pushButton_remove_bad_cluster.setText("MPS analysis")
        self.ui.pushButton_remove_bad_cluster.setToolTip(
            "Open the per-axon MPS parameters panel (Gazal et al. 2026).\n"
            "Bad clusters are removed automatically; every parameter in the\n"
            "panel stays editable."
        )
        self.ui.pushButton_remove_bad_cluster.clicked.connect(
            lambda: self.run_mps_analysis(show_window=True))
        self.ui.pushButton_savecluscenters.clicked.connect(self.save_clus_CM)
        self.ui.pushButton_Distances.clicked.connect(self.KNdist_hist)
        self.ui.pushButton_saveAllClusterData.clicked.connect(lambda: self.save_all_clustered_data(1))
        self.ui.pushButton_saveAllClusterDataThunderStorm.clicked.connect(lambda: self.save_all_clustered_data_thunderstorm(1))
        
        # Z range: remember when the user overrides the automatic value.
        # textEdited (unlike textChanged) fires only on real keystrokes, so
        # the pre-fill's own setText calls do not mark the field as edited.
        self.ui.lineEdit_zmin.textEdited.connect(self._on_z_range_edited)
        self.ui.lineEdit_zmax.textEdited.connect(self._on_z_range_edited)

        # Fine Tuning Parameters
        self.ui.lineEdit_latmin.textChanged.connect(self.latchange)
        self.ui.lineEdit_latmax.textChanged.connect(self.latchange)
        self.ui.lineEdit_bin.textChanged.connect(self.latchange)

        self.lmin = 0
        self.lmax = MAX_LATERAL_DISTANCE_NM
        self.bins = DEFAULT_KNN_BINS
        
        # Colors
        self.brush1 = pg.mkBrush("#d55e00")
        self.brush2 = pg.mkBrush("#009e73")
        self.brush3 = pg.mkBrush("#0072b2")
        
        self.pen1 = pg.mkPen("#d55e00")
        self.pen2 = pg.mkPen("#009e73")
        self.pen3 = pg.mkPen("#0072b2")
        
        # ROI Shape Radio Buttons
        self.radioButton_circROI = self.ui.radioButton_circROI
        self.radioButton_squareROI = self.ui.radioButton_squareROI
        self.radioButton_polygonROI = self.ui.radioButton_polygonROI
        self.radioButton_circROI.clicked.connect(self.scatterplot)
        self.radioButton_squareROI.clicked.connect(self.scatterplot)
        self.radioButton_polygonROI.clicked.connect(self.scatterplot)

        # ------------------------------------------------------------------
        # Hallazgo 14 — Defensive attribute initialisation
        #
        # Several instance attributes are created as side-effects of user
        # interactions (load file → scatter → ROI → cluster → distances).
        # When a button is pressed out of order the attribute does not yet
        # exist and Python raises an AttributeError that propagates silently
        # or crashes the GUI.  Initialising every attribute here to None (or
        # an appropriate empty sentinel) makes the broken-workflow case
        # explicit and allows the guard-clauses in each method to give the
        # user a meaningful error message instead of a traceback.
        # ------------------------------------------------------------------

        # --- raw data loaded from file (set by select_file / import_file) ---
        self.pxsize: Optional[float] = None          # effective pixel size in nm (from YAML or user)
        # Where pxsize came from: "yaml" | "hdf5" | "yaml_scan" | "manual" |
        # "not_applicable" | "unknown". A pixel size that did not come from
        # the Picasso metadata silently rescales every lateral distance
        # (and, squared, every cluster area), so its provenance travels with
        # the analysis and into the exported CSV.
        self.pxsize_source: str = "unknown"

        # The complete loaded file per channel: every column of the source
        # table plus Picasso's processing chain. xdata/ydata/zdata below are
        # the three coordinate arrays taken out of these, kept as separate
        # attributes because the whole application already reads them that
        # way. Anything that needs `frame` (binding events, dark times,
        # sticking rejection) or `lpx/lpy/lpz` (is this structure wide, or
        # is this measurement imprecise?) reads it from here instead.
        self.locs1: Optional[mps_io.Localizations] = None
        self.locs2: Optional[mps_io.Localizations] = None
        # Indices into the full table of the localizations that survived the
        # ROI and the axial cut; set by _apply_z_range. The cluster labels
        # index the same subset.
        self.roi_indices: Optional[NDArray[np.int64]] = None
        self.quality_window: Optional[Any] = None
        self.paint_window: Optional[Any] = None
        self.two_channel_window: Optional[Any] = None
        # The ROI shape and axial range the current channel-1 selection was
        # made with. The ROI widget can move without being applied (a
        # redraw puts a default one up), so this, not the widget, describes
        # xroi.
        self._applied_roi_shape: Optional[Any] = None
        self._applied_slab: Optional[Tuple[float, float]] = None
        self.picasso_tools = PicassoTools(self)

        # Channel 1 raw coordinates (pixel→nm converted)
        self.xdata: Optional[NDArray[np.float64]] = None
        self.ydata: Optional[NDArray[np.float64]] = None
        self.zdata: Optional[NDArray[np.float64]] = None
        self.fileformat1: int = 0        # default: Picasso hdf5

        # Channel 2 raw coordinates
        self.xdata2: Optional[NDArray[np.float64]] = None
        self.ydata2: Optional[NDArray[np.float64]] = None
        self.zdata2: Optional[NDArray[np.float64]] = None
        self.fileformat2: int = 0

        # --- scatter / overview (set by scatterplot) ---
        self.x: Optional[NDArray[np.float64]] = None               # Ch1 x coordinates currently displayed
        self.y: Optional[NDArray[np.float64]] = None
        self.z: Optional[NDArray[np.float64]] = None
        self.data_points: Optional[NDArray[np.float64]] = None     # (N, 2) array used by ROI filter

        self.x2: Optional[NDArray[np.float64]] = None              # Ch2 coordinates
        self.y2: Optional[NDArray[np.float64]] = None
        self.z2: Optional[NDArray[np.float64]] = None
        self.data_points2: Optional[NDArray[np.float64]] = None
        # The overview plot and channel 2's layer on it, so a newly loaded
        # channel-2 file can replace that layer without a redraw.
        self._overview_plot: Optional[Any] = None
        self._ch2_overlay: Optional[Any] = None

        # --- ROI selection (set by update_ROI) ---
        self.xroi: Optional[NDArray[np.float64]] = None            # Ch1 localizations inside the ROI
        self.yroi: Optional[NDArray[np.float64]] = None
        self.zroi: Optional[NDArray[np.float64]] = None
        self.zmin: Optional[float] = None            # z slab lower bound (nm)
        self.zmax: Optional[float] = None            # z slab upper bound (nm)

        # The same ROI selection BEFORE the axial range is applied. The MPS
        # analysis fits its Gaussian mixture to the full axial distribution
        # to obtain Delta-Z and locate the main peak; handing it the already
        # sliced 180 nm slab would refit the mixture inside that slab and
        # make the periodicity meaningless.
        self.xroi_unfiltered: Optional[NDArray[np.float64]] = None
        self.yroi_unfiltered: Optional[NDArray[np.float64]] = None
        self.zroi_unfiltered: Optional[NDArray[np.float64]] = None

        # True once the user types into Z min / Z max, after which the
        # automatic pre-fill stops overwriting their choice.
        self._z_range_user_edited: bool = False

        self.xroi2: Optional[NDArray[np.float64]] = None           # Ch2 localizations inside the ROI
        self.yroi2: Optional[NDArray[np.float64]] = None
        self.zroi2: Optional[NDArray[np.float64]] = None

        # --- clustering (set by cluster) ---
        self.cluster_labels: Optional[NDArray[np.int64]] = None  # DBSCAN labels array for Ch1 (-1 = noise)
        self.cluster_labels2: Optional[NDArray[np.int64]] = None # DBSCAN labels array for Ch2
        self.original_points: Optional[NDArray[np.float64]] = None # (N, 2) XY used during clustering (Ch1)
        self.original_z: Optional[NDArray[np.float64]] = None      # z array used during clustering (Ch1)
        self.cluster_centroids: Optional[NDArray[np.float64]] = None             # (K, 2) cluster centroids
        self.good_cluster_centroids: Optional[NDArray[np.float64]] = None            # (M, 2) centroids after removing bad clusters
        self.bad_cluster_indices: List[int] = []  # indices into cluster_centroids marked as bad by the user
        self.cluster_centroids2: Optional[NDArray[np.float64]] = None  # (K, 2) Ch2 cluster centroids
        # The ROI selection each channel's labels were computed on. A new
        # selection is a new array, and the labels no longer line up with it.
        self._clustered_x: Dict[int, Optional[NDArray[np.float64]]] = {
            1: None, 2: None}

        self.eps: Optional[float] = None             # DBSCAN epsilon parameter
        self.minsamples: Optional[int] = None      # DBSCAN min_samples parameter

        # --- polygon drawing mode (interactive drawing) ---
        self.polygon_drawing_mode: bool = False     # True when actively drawing polygon
        self.polygon_points_temp: List[List[float]] = []  # Accumulate clicked points during drawing
        self.polygon_drawing_visual: Optional[pg.PolyLineROI] = None  # Visual feedback (polyline + points)
        self.polygon_drawing_label: Optional[QtWidgets.QLabel] = None  # Status label for user guidance
        self.mouse_click_handler: Optional[Any] = None  # Connection handle for mouse clicks
        self.current_plot: Optional[Any] = None  # Reference to plotxy during drawing
        self.current_viewbox: Optional[Any] = None  # Reference to ViewBox during drawing
        self.current_roi_pen: Optional[Any] = None  # ROI pen color during drawing

        # --- nearest-neighbour distances (set by KNdist_hist) ---
        self.distances: Optional[NDArray[np.float64]] = None       # distance array to the K nearest centroids
        self.Nneighbor: Optional[int] = None       # number of neighbours requested

        # --- Gazal 2026 per-axon analysis (set by run_mps_analysis) ---
        self.mps_analysis: Optional[Any] = None      # last AxonAnalysis
        # The selection array the last analysis was run on. A new selection
        # is a new array, and the analysis no longer describes it.
        self._analysed_x: Optional[NDArray[np.float64]] = None
        self.mps_window: Optional[Any] = None        # results window (kept alive)
        self.rings_window: Optional[Any] = None      # multi-segment panel
        self.mps_settings = load_settings()          # persisted across sessions
        self._apply_mps_settings()

        # Connect the close event to your method
        self.closeEvent = self.onCloseEvent

    # ========================================================================
    # Gazal et al. (2026) per-axon MPS analysis
    # ========================================================================

    def _on_z_range_edited(self, _text: str = "") -> None:
        """Mark the Z range as user-controlled so the pre-fill stops touching it."""
        if not self._z_range_user_edited:
            self._z_range_user_edited = True
            self.logger.debug(
                "Z range edited manually; automatic pre-fill disabled for "
                "this dataset."
            )

    def _compute_z_main_peak(self, z_values: NDArray[np.float64]) -> Optional[float]:
        """
        Locate the dominant peak of an axial distribution, in nm.

        Uses the same Gaussian-mixture fit the MPS analysis uses
        (tools.mps_periodicity), so the range shown in Z min / Z max is
        exactly the slab the analysis will work on rather than a second,
        slightly different estimate. Falls back to the tallest histogram
        bin if the mixture cannot be fitted (too few localizations).
        """
        z = np.asarray(z_values, dtype=float).ravel()
        z = z[np.isfinite(z)]
        if z.size < 2:
            return None
        try:
            return float(fit_z_periodicity(z).main_peak_nm)
        except Exception as exc:                          # noqa: BLE001
            self.logger.debug(
                f"GMM peak estimation failed ({exc}); using histogram mode.")
            counts, edges = np.histogram(z, bins="auto")
            if counts.size == 0:
                return None
            i = int(np.argmax(counts))
            return float((edges[i] + edges[i + 1]) / 2.0)

    def _prefill_z_range(self, z_values: NDArray[np.float64]) -> None:
        """
        Pre-fill Z min / Z max with the +/-90 nm window around the axial peak.

        This is the 180 nm slab that isolates a single MPS segment in Gazal
        et al. (2026). Showing it in the fields means the user can see, and
        change, the axial window the analysis will use instead of it being
        applied invisibly. A range the user typed themselves is never
        overwritten.
        """
        if self._z_range_user_edited:
            return
        peak = self._compute_z_main_peak(z_values)
        if peak is None:
            return
        half = float(self.mps_settings.slab_half_width_nm)
        self.ui.lineEdit_zmin.setText(f"{peak - half:.1f}")
        self.ui.lineEdit_zmax.setText(f"{peak + half:.1f}")
        self.logger.info(
            f"Z range pre-filled from the axial peak at {peak:.1f} nm: "
            f"{peak - half:.1f} .. {peak + half:.1f} nm (+/-{half:g} nm)"
        )

    def _apply_z_range(
        self, ind_inside_roi: Optional[NDArray[np.intp]] = None
    ) -> None:
        """
        Apply the Z range to the current channel-1 ROI selection.

        Replaces the four copies of this logic that used to live inline in
        each ROI-shape branch of update_ROI. Also stores the selection
        BEFORE axial filtering, which the MPS analysis needs (see
        self.zroi_unfiltered), and pre-fills the range on first use.

        Parameters
        ----------
        ind_inside_roi : indices into self.z of the spatially selected
            localizations, or None when the whole field of view is used.
        """
        z_all = self.z if ind_inside_roi is None else self.z[ind_inside_roi]

        self.xroi_unfiltered = np.asarray(self.xroi).copy()
        self.yroi_unfiltered = np.asarray(self.yroi).copy()
        self.zroi_unfiltered = np.asarray(z_all).copy()

        self._prefill_z_range(z_all)

        zmin_text = self.ui.lineEdit_zmin.text().strip()
        zmax_text = self.ui.lineEdit_zmax.text().strip()
        try:
            self.zmin = float(zmin_text) if zmin_text else None
            self.zmax = float(zmax_text) if zmax_text else None
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Z range",
                "Z min and Z max must be numeric. Ignoring the axial filter "
                "for this selection."
            )
            self.zmin = self.zmax = None

        # Indices into the FULL localization table of whatever survives the
        # ROI and the axial cut, in the same order as xroi/yroi/zroi. The
        # cluster labels index into that subset, so without this there is
        # no way to line a label up with the frame, photon count or
        # precision of the localization it belongs to -- which is
        # everything the DNA-PAINT and quality panels need.
        base = (
            np.arange(np.asarray(self.z).size)
            if ind_inside_roi is None
            else np.asarray(ind_inside_roi)
        )

        # Both bounds are required: filtering on one alone silently kept the
        # other side unbounded, and comparing against None raised a TypeError.
        if self.zmin is None or self.zmax is None:
            self.zroi = z_all
            self.roi_indices = base
        else:
            keep = (z_all > self.zmin) & (z_all < self.zmax)
            self.zroi = z_all[keep]
            self.xroi = self.xroi[keep]
            self.yroi = self.yroi[keep]
            self.roi_indices = base[keep]

        self._applied_roi_shape = (
            None if ind_inside_roi is None else self._current_roi_shape())
        self._applied_slab = (
            (float(self.zmin), float(self.zmax))
            if (self._z_range_user_edited and self.zmin is not None
                and self.zmax is not None) else None)
        # The centroids and the distances between them belong to the
        # previous selection's analysis: saving them now would write
        # another selection's results. Clustering again recomputes them.
        self.good_cluster_centroids = None
        self.distances = None
        self.Nneighbor = None
        if self.two_channel_window is not None and \
                self.two_channel_window.isVisible():
            self.two_channel_window.update_selection(
                self._applied_roi_shape, self._applied_slab)

    # ------------------------------------------------------------------
    #  Acquisition-level panels: data quality and DNA-PAINT
    # ------------------------------------------------------------------
    def _make_content_scrollable(self) -> None:
        """
        Put the window's content in a scroll area.

        The .ui places every widget at a fixed position, down to y = 701, so
        on a short screen (1280x800, 1366x768) the bottom row of buttons --
        Distances and the exports -- used to be cut off with no way to reach
        it.
        """
        # takeCentralWidget, not a plain setCentralWidget: the latter
        # deletes the widget it replaces.
        content = self.takeCentralWidget()
        content.setMinimumSize(1381, 701)
        scroll = QtWidgets.QScrollArea()
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.setCentralWidget(scroll)

    def _build_analysis_toolbar(self) -> None:
        """
        Add a toolbar for the tools that are about the ACQUISITION rather
        than the axon: data quality, DNA-PAINT, and the Picasso tools.

        Kept out of the results window because none of them changes when
        the clustering parameters are edited, and putting them there would
        invite reading them as results. DNA-PAINT is separate from data
        quality in turn because it does not merely describe the data, it
        changes what the software's other numbers mean.

        A toolbar rather than a menu: a menu bar holding a single entry
        reads as part of the title bar, and users did not find it. The
        .ui's menu bar is hidden, since nothing else lives there.
        """
        self.menuBar().setVisible(False)

        self.action_quality = QtWidgets.QAction("Data quality", self)
        self.action_quality.setToolTip(
            "NeNA precision, fitting-box check, axial resolvedness and "
            "residual drift for the loaded file."
        )
        self.action_quality.triggered.connect(self.show_quality_panel)

        self.action_paint = QtWidgets.QAction("DNA-PAINT", self)
        self.action_paint.setToolTip(
            "Link localizations into binding events, reject non-specific "
            "sticking, and measure binding kinetics. qPAINT counting is a "
            "tab inside it."
        )
        self.action_paint.triggered.connect(self.show_paint_panel)

        self.action_two_channels = QtWidgets.QAction("Two channels", self)
        self.action_two_channels.setToolTip(
            "Register channel 2 onto channel 1 (markers or a calibration), "
            "then compare them: axial phase and transverse arrangement."
        )
        self.action_two_channels.triggered.connect(
            self.show_two_channel_panel)

        # Tools that call the Picasso program. They are wired lazily through
        # self.picasso_tools, which is created later in __init__.
        picasso_menu = QtWidgets.QMenu("Picasso tools", self)
        picasso_menu.setToolTipsVisible(True)
        actions = (
            ("Undrift with AIM...", "undrift",
             "Drift correction in x, y and z (AIM). Runs on the whole "
             "loaded file and writes *_aim.hdf5 next to it."),
            ("SMLM clustering...", "cluster",
             "Picasso's SMLM clusterer, with separate lateral and axial "
             "radii for 3D data."),
            ("Molecular mapping (G5M)...", "map_molecules",
             "Split clusters into molecules whose widths are bounded by the "
             "localization precision. Needs clustered localizations."),
        )
        for label, method, tip in actions:
            action = picasso_menu.addAction(label)
            action.setToolTip(tip)
            action.triggered.connect(
                lambda _checked=False, m=method: getattr(self.picasso_tools, m)())
        picasso_menu.addSeparator()
        where = picasso_menu.addAction("Picasso location...")
        where.setToolTip("Choose the Picasso executable, or check which one "
                         "is used.")
        where.triggered.connect(lambda: self.picasso_tools.locate())

        toolbar = self.addToolBar("Analysis")
        toolbar.setObjectName("analysisToolBar")
        toolbar.setMovable(False)
        # With no menu bar left, a toolbar hidden from the window's
        # right-click menu could not be brought back.
        toolbar.toggleViewAction().setVisible(False)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        # The native toolbar draws buttons flat, as plain text until hovered;
        # framed like the window's own push buttons, they read as buttons.
        toolbar.setStyleSheet(
            "QToolBar { spacing: 6px; padding: 2px 4px; }"
            "QToolButton { border: 1px solid #adadad; border-radius: 3px;"
            " background: #fdfdfd; color: #000000; padding: 2px 10px; }"
            "QToolButton:hover { border-color: #0078d7; background: #e5f1fb; }"
            "QToolButton:pressed { background: #cce4f7; }"
            "QToolButton[popupMode=\"2\"] { padding-right: 18px; }"
        )
        toolbar.addAction(self.action_quality)
        toolbar.addAction(self.action_paint)
        toolbar.addAction(self.action_two_channels)
        picasso_button = QtWidgets.QToolButton(toolbar)
        picasso_button.setText("Picasso tools")
        picasso_button.setToolTip(
            "AIM drift correction, SMLM clustering and G5M molecular "
            "mapping, run by the Picasso program.")
        picasso_button.setMenu(picasso_menu)
        picasso_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        toolbar.addWidget(picasso_button)
        self.analysis_toolbar = toolbar

    def _roi_localizations(self) -> Optional[Any]:
        """
        The loaded file restricted to the current ROI and axial slab, or
        the whole file when nothing has been selected yet.

        Returns None when no file is loaded.
        """
        if self.locs1 is None:
            QtWidgets.QMessageBox.information(
                self, "No file loaded",
                "Load a localization file first (channel 1)."
            )
            return None
        if self.roi_indices is None:
            return self.locs1
        return self.locs1.subset(self.roi_indices)

    def show_quality_panel(self) -> None:
        """Open the data-quality panel for the loaded file."""
        loc = self._roi_localizations()
        if loc is None:
            return
        try:
            from tools.mps_quality_window import show_quality_window

            means = sigmas = weights = None
            period = None
            if loc.is_3d:
                try:
                    fit = fit_z_periodicity(loc.z_nm)
                    means, sigmas = fit.means_nm, fit.sigmas_nm
                    weights = getattr(fit, "weights", None)
                    period = getattr(fit, "delta_z_nm", None)
                    if period is not None and np.ndim(period) > 0:
                        period = float(np.median(np.asarray(period)))
                except Exception:      # noqa: BLE001 - the panel still runs
                    self.logger.debug(
                        "Z periodicity fit failed; the axial tab will say so",
                        exc_info=True)
            self.quality_window = show_quality_window(
                loc, means_nm=means, sigmas_nm=sigmas, weights=weights,
                period_nm=period, parent=self,
            )
            self.logger.info(
                f"Quality panel opened for {os.path.basename(loc.path)} "
                f"({loc.n:,} localizations)")
        except Exception as error:     # noqa: BLE001 - surfaced to the user
            self.logger.error(f"Quality panel failed: {error}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Data quality", f"Could not build the panel:\n{error}")

    def show_paint_panel(self) -> None:
        """Open the DNA-PAINT panel for the loaded file."""
        loc = self._roi_localizations()
        if loc is None:
            return
        if not loc.has("frame"):
            QtWidgets.QMessageBox.warning(
                self, "DNA-PAINT",
                "This file has no 'frame' column. Binding events, dark "
                "times and qPAINT counting are all functions of the frame "
                "number, so none of them can be computed. A ThunderSTORM "
                "CSV export usually carries one; a three-column CSV does "
                "not."
            )
            return

        labels = self.cluster_labels
        if labels is not None and self.roi_indices is not None and \
                len(labels) != loc.n:
            self.logger.warning(
                f"Cluster labels ({len(labels)}) do not match the ROI "
                f"selection ({loc.n}); the sticking filter and qPAINT will "
                f"be offered without them. Re-cluster to re-enable them.")
            labels = None

        try:
            from tools.mps_paint_window import show_paint_window

            self.paint_window = show_paint_window(
                loc, site_labels=labels, parent=self)
            self.logger.info(
                f"DNA-PAINT panel opened for {os.path.basename(loc.path)} "
                f"({loc.n:,} localizations, "
                f"{'with' if labels is not None else 'without'} cluster "
                f"labels)")
        except Exception as error:     # noqa: BLE001 - surfaced to the user
            self.logger.error(f"DNA-PAINT panel failed: {error}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "DNA-PAINT", f"Could not build the panel:\n{error}")

    def _field_parameter(self, edit: Any, fallback: float) -> float:
        try:
            value = float(edit.text())
        except (TypeError, ValueError):
            return fallback
        return value if value > 0 else fallback

    def _clustering_parameters(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        eps and min samples typed for each channel. "auto" or a blank
        field falls back to the stored settings.
        """
        s = self.mps_settings

        def read(eps: Any, minimum: Any) -> Dict[str, Any]:
            return dict(
                eps_nm=self._field_parameter(eps, float(s.eps_nm)),
                min_samples=int(self._field_parameter(
                    minimum, float(s.min_samples))))

        return (read(self.ui.lineEdit_eps, self.ui.lineEdit_minsamples),
                read(self.ui.lineEdit_eps_2, self.ui.lineEdit_minsamples_2))

    def show_two_channel_panel(self) -> None:
        """Open the two-channel panel on the two loaded files."""
        if self.locs1 is None or self.locs2 is None:
            QtWidgets.QMessageBox.information(
                self, "Two channels",
                "Load a file in channel 1 (betaII-spectrin) and one in "
                "channel 2 (the partner protein) first.")
            return
        s = self.mps_settings
        roi = self._applied_roi_shape if self.xroi is not None else None
        slab = self._applied_slab
        shared = dict(dbcv_threshold=float(s.dbcv_threshold), roi=roi)
        params_a, params_b = self._clustering_parameters()
        kwargs_a = dict(
            shared, **params_a,
            pixel_size_nm=self.locs1.pixel_size_nm,
            pixel_size_source=self.locs1.pixel_size_source,
            source_name=self.ui.lineEdit_filename.text(),
        )
        kwargs_b = dict(
            **params_b,
            pixel_size_nm=self.locs2.pixel_size_nm,
            pixel_size_source=self.locs2.pixel_size_source,
            source_name=self.ui.lineEdit_filename_2.text(),
        )
        try:
            from tools.mps_twochannel_window import (
                TwoChannelInputs,
                show_two_channel_window,
            )

            if self.two_channel_window is not None:
                self.two_channel_window.close()
            self.two_channel_window = show_two_channel_window(
                TwoChannelInputs(
                    loc_a=self.locs1, loc_b=self.locs2, roi=roi, slab=slab,
                    slab_half_width_nm=float(s.slab_half_width_nm),
                    kwargs_a=kwargs_a, kwargs_b=kwargs_b),
                parent=self, parameters=self._clustering_parameters)
            self.logger.info(
                f"Two-channel panel opened for "
                f"{os.path.basename(self.locs1.path)} and "
                f"{os.path.basename(self.locs2.path)}"
                + ("" if roi is not None else " (no ROI)"))
        except Exception as error:     # noqa: BLE001 - surfaced to the user
            self.logger.error(f"Two-channel panel failed: {error}",
                              exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Two channels", f"Could not build the panel:\n{error}")

    def _apply_mps_settings(self) -> None:
        """
        Restore the DBSCAN parameters persisted from the previous session.

        The user loads axons in series from one acquisition, so retyping
        these every launch is how a stale value slips into a batch. The
        .ui ships eps = 10, which is not the paper's value (25 nm); the
        stored settings -- seeded with the paper defaults on first run --
        take precedence, and what was restored is logged.
        """
        s = self.mps_settings
        for widget in (self.ui.lineEdit_eps, self.ui.lineEdit_eps_2):
            widget.setText(f"{s.eps_nm:g}")
        for widget in (self.ui.lineEdit_minsamples, self.ui.lineEdit_minsamples_2):
            widget.setText(f"{int(s.min_samples)}")
        self.logger.info(
            f"MPS settings restored: eps={s.eps_nm:g} nm, "
            f"min_samples={int(s.min_samples)}, "
            f"slab half-width={s.slab_half_width_nm:g} nm"
        )

    def _persist_mps_settings(self) -> None:
        """Store the parameters currently in the UI for the next session."""
        try:
            self.mps_settings.eps_nm = float(self.ui.lineEdit_eps.text())
            self.mps_settings.min_samples = int(
                float(self.ui.lineEdit_minsamples.text()))
        except (ValueError, AttributeError):
            # "auto" or a malformed entry: keep whatever was stored before
            # rather than writing a value the analysis never actually used.
            pass
        self.mps_settings.validate()
        save_settings(self.mps_settings)

    def _channel1_pixel_size(self) -> Tuple[Optional[float], str]:
        """
        Pixel size of the channel-1 file and where it came from.

        ``self.pxsize`` belongs to whichever file was loaded last, which is
        channel 2's if that one was loaded after.
        """
        if self.locs1 is not None:
            return self.locs1.pixel_size_nm, self.locs1.pixel_size_source
        return self.pxsize, self.pxsize_source

    def _check_channel_pixel_sizes(self) -> None:
        """
        Warn when the two loaded channels disagree on the pixel size.

        Both channels go through the same optics, so one of the two files
        is wrong. The error scales one channel about the camera origin
        rather than shifting it, which no registration can undo, and it is
        easy to miss: each panel simply uses its own file's value. The 2023
        sciatic-nerve data has it -- in one acquisition, adducin was
        localized with 133 nm and spectrin with 135 nm.
        """
        if self.locs1 is None or self.locs2 is None:
            return
        from tools.mps_registration import pixel_size_disagreement, reach_nm

        note = pixel_size_disagreement(
            f"Channel 1 ({os.path.basename(str(self.locs1.path))})",
            self.locs1.pixel_size_nm,
            f"channel 2 ({os.path.basename(str(self.locs2.path))})",
            self.locs2.pixel_size_nm,
            reach_nm(self.locs1, self.locs2))
        if note is None:
            return
        self.logger.warning(note)
        QtWidgets.QMessageBox.warning(
            self, "The channels disagree on the pixel size",
            note + "\n\nLocalize both channels with the same pixel size, or "
                   "correct the metadata of the one that is wrong.")

    def _polygon_vertices(self) -> NDArray[np.float64]:
        """The polygon ROI's vertices in data coordinates.

        ``getState()['points']`` gives them relative to the ROI's own
        position, so a dragged polygon would still select where it was.
        """
        return np.array([
            [p.x(), p.y()] for p in (
                self.polygon_roi.mapToParent(h.pos())
                for h in self.polygon_roi.getHandles())
        ], dtype=float)

    def _rotated_square_corners(self) -> Optional[NDArray[np.float64]]:
        """The square ROI's corners in data coordinates, if it is rotated."""
        if abs(float(self.square_roi.angle())) < 1e-9:
            return None
        w, h = self.square_roi.size()
        return np.array([
            [p.x(), p.y()] for p in (
                self.square_roi.mapToParent(QtCore.QPointF(cx, cy))
                for cx, cy in ((0, 0), (w, 0), (w, h), (0, h)))
        ], dtype=float)

    def _current_roi_shape(self) -> Optional[Any]:
        """
        Describe the active ROI for the edge-touching bad-cluster criterion.

        Returns None when no ROI widget is active, in which case the
        analysis falls back to the convex hull of the localizations.
        """
        try:
            if self.ui.radioButton_circROI.isChecked():
                pos = self.circular_roi.pos()
                size = self.circular_roi.size()
                if hasattr(size, "x"):
                    s = float(size.x())
                elif hasattr(size, "width"):
                    s = float(size.width())
                else:
                    s = float(size)
                return CircularROI(
                    center_x=float(pos.x()) + s / 2,
                    center_y=float(pos.y()) + s / 2,
                    radius=(ROI_DIAMETER_SCALE_FACTOR * s) / 2,
                )
            if self.ui.radioButton_squareROI.isChecked():
                corners = self._rotated_square_corners()
                if corners is not None:
                    return PolygonROI(vertices=corners)
                xmin, ymin = self.square_roi.pos()
                xmax, ymax = self.square_roi.pos() + self.square_roi.size()
                return SquareROI(xmin=float(xmin), ymin=float(ymin),
                                 xmax=float(xmax), ymax=float(ymax))
            if self.ui.radioButton_polygonROI.isChecked():
                return PolygonROI(vertices=self._polygon_vertices())
        except (AttributeError, KeyError, TypeError):
            return None
        return None

    def _render_good_clusters_panel(
        self, centroids: NDArray[np.float64]
    ) -> None:
        """
        Render curated cluster centroids into the main window's own "good
        clusters" panel (blue, brush3).

        This panel used to populate only after manual curation (clicking
        each bad centroid via rx(), then dist_cm_good_clus()). Once bad-
        cluster removal became automatic, nothing called it any more and
        it went permanently blank -- not broken, just orphaned. Shared here
        by run_mps_analysis (automatic path) and dist_cm_good_clus (manual
        fallback, see its docstring) so both draw it the same way.

        Safe to call with zero centroids (e.g. curation removed every
        cluster): the panel is cleared rather than raising on an empty
        scatter.
        """
        good_clusters_widget = pg.GraphicsLayoutWidget()
        good_clusters_plot = good_clusters_widget.addPlot(
            title="Clusters centers and distances")
        good_clusters_plot.setAspectLocked(True)
        good_clusters_plot.setLabels(bottom='x [nm]', left='y [nm]')

        if len(centroids):
            self.good_clusters_scatter_plot = pg.ScatterPlotItem(
                centroids[:, 0], centroids[:, 1],
                size=CLUSTER_CENTROID_POINT_SIZE, brush=self.brush3)
            good_clusters_plot.addItem(self.good_clusters_scatter_plot)

        if self.xroi is not None and len(self.xroi):
            good_clusters_plot.setXRange(
                np.min(self.xroi), np.max(self.xroi), padding=0)

        self.empty_layout(self.ui.scatterlayout_goodclus)
        self.ui.scatterlayout_goodclus.addWidget(good_clusters_widget)

    def run_mps_analysis(self, show_window: bool = True, **overrides) -> Optional[Any]:
        """
        Run the full Gazal-2026 per-axon pipeline on the current ROI.

        Called automatically at the end of ``cluster(1)`` and re-callable
        with overrides from the results window. betaII-spectrin is always
        channel 1, so the analysis always uses the channel-1 ROI.

        Parameters
        ----------
        show_window : open (or raise) the results window.
        **overrides : forwarded to ``analyze_axon`` -- main_peak_override_nm,
            slab_half_width_nm, eps_nm, min_samples, dbcv_threshold,
            custom_contour_order.
        """
        if self.xroi is None or self.zroi is None or len(self.xroi) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No ROI selected",
                "Load a file, draw the scatter plot and select an ROI before "
                "running the MPS analysis."
            )
            return None

        s = self.mps_settings
        params = dict(
            source_name=self.ui.lineEdit_filename.text(),
            pixel_size_nm=self._channel1_pixel_size()[0],
            pixel_size_source=self._channel1_pixel_size()[1],
            eps_nm=float(overrides.pop("eps_nm", s.eps_nm)),
            min_samples=int(overrides.pop("min_samples", s.min_samples)),
            slab_half_width_nm=float(
                overrides.pop("slab_half_width_nm", s.slab_half_width_nm)),
            dbcv_threshold=float(
                overrides.pop("dbcv_threshold", s.dbcv_threshold)),
            # The ROI the selection was made with: the widget may since
            # have been redrawn at its default place.
            roi=self._applied_roi_shape,
        )
        params.update(overrides)

        # Feed the analysis the ROI selection BEFORE the axial range was
        # applied. Its Gaussian mixture needs the full axial distribution to
        # measure Delta-Z and locate the main peak; handing it the already
        # sliced 180 nm slab would refit the mixture inside that slab and
        # report a meaningless periodicity.
        if self.zroi_unfiltered is not None and len(self.zroi_unfiltered):
            x_in, y_in, z_in = (self.xroi_unfiltered,
                                self.yroi_unfiltered,
                                self.zroi_unfiltered)
        else:
            x_in, y_in, z_in = self.xroi, self.yroi, self.zroi

        # When the user set the range by hand, that choice wins over the
        # automatic slab. Left automatic, the analysis recomputes the same
        # window the fields are showing, so no override is needed.
        if (self._z_range_user_edited and self.zmin is not None
                and self.zmax is not None):
            params["slab_override"] = (self.zmin, self.zmax)

        self.logger.info(
            f"MPS analysis: {len(x_in):,} ROI localizations "
            f"(axial range {'manual' if 'slab_override' in params else 'automatic'}), "
            f"eps={params['eps_nm']:g}, min_samples={params['min_samples']}"
        )
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            analysis = analyze_axon(x_in, y_in, z_in, **params)
        except Exception as exc:                          # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            self.logger.error(f"MPS analysis failed: {exc}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "MPS analysis failed", f"{exc}")
            return None
        QtWidgets.QApplication.restoreOverrideCursor()

        self.mps_analysis = analysis
        self._analysed_x = x_in
        self.logger.info(
            f"MPS analysis: {analysis.n_clusters_kept}/{analysis.n_clusters_raw} "
            f"clusters kept, perimeter="
            f"{'n/a' if analysis.perimeter_um is None else f'{analysis.perimeter_um:.2f} um'}, "
            f"1NN median="
            f"{'n/a' if analysis.median_1nn_nm is None else f'{analysis.median_1nn_nm:.0f} nm'}"
        )
        for w in analysis.warnings:
            self.logger.warning(f"MPS analysis: {w}")

        # Keep the legacy attributes consistent so the existing centroid
        # plots and exports reflect the same automatically curated set.
        self.bad_cluster_indices = sorted(analysis.bad_report.bad_labels)
        self.good_cluster_centroids = analysis.centroids
        self._render_good_clusters_panel(analysis.centroids)

        if show_window:
            self._show_mps_window(analysis)
        return analysis

    def _show_mps_window(self, analysis: Any) -> None:
        """Open, or refresh in place, the MPS results window."""
        from tools.mps_results_window import MPSResultsWindow

        def rerun(**kw):
            # show_window=False: the window refreshes itself with the
            # returned analysis, so reopening it here would recurse.
            result = self.run_mps_analysis(show_window=False, **kw)
            if result is None:
                raise RuntimeError(
                    "The analysis could not be re-run with those parameters.")
            return result

        if self.mps_window is None:
            self.mps_window = MPSResultsWindow(
                analysis, rerun_callback=rerun, parent=self,
                rings_callback=self.run_ring_analysis)
        else:
            self.mps_window.analysis = analysis
            self.mps_window.refresh()
        self.mps_window.show()
        self.mps_window.raise_()
        self.mps_window.activateWindow()

    def run_ring_analysis(
        self, show_window: bool = True, mode: str = "valley",
        guard_nm: float = 0.0,
    ) -> Optional[Any]:
        """
        Analyse every axial segment of the current ROI and compare
        consecutive ones (thesis objective (e)).

        Unlike ``run_mps_analysis``, which measures the single 180 nm slab
        the paper defines, this runs that same per-segment pipeline on every
        dominant axial component and correlates the gap/patch pattern of
        neighbours.

        Parameters
        ----------
        show_window : open (or refresh) the rings panel.
        mode : slab boundaries -- "valley", "paper" or "partition".
        guard_nm : dead zone at each boundary, as a bleed-through control.
        """
        from tools.mps_multisegment import analyze_all_segments

        if self.xroi is None or self.zroi is None or len(self.xroi) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No ROI selected",
                "Load a file, draw the scatter plot and select an ROI before "
                "running the ring analysis."
            )
            return None

        # Same reason as in run_mps_analysis: the mixture must see the full
        # axial distribution, not an already sliced slab.
        if self.zroi_unfiltered is not None and len(self.zroi_unfiltered):
            x_in, y_in, z_in = (self.xroi_unfiltered,
                                self.yroi_unfiltered,
                                self.zroi_unfiltered)
        else:
            x_in, y_in, z_in = self.xroi, self.yroi, self.zroi

        s = self.mps_settings
        self.logger.info(
            f"Ring analysis: {len(x_in):,} ROI localizations, mode={mode}, "
            f"guard={guard_nm:g} nm, eps={s.eps_nm:g}, "
            f"min_samples={s.min_samples}"
        )
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            ms = analyze_all_segments(
                x_in, y_in, z_in,
                source_name=self.ui.lineEdit_filename.text(),
                mode=mode, guard_nm=guard_nm,
                half_width_nm=s.slab_half_width_nm,
                eps_nm=s.eps_nm, min_samples=s.min_samples,
                dbcv_threshold=s.dbcv_threshold,
                pixel_size_nm=self._channel1_pixel_size()[0],
                pixel_size_source=self._channel1_pixel_size()[1],
                roi=self._applied_roi_shape,
            )
        except Exception as exc:                          # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            self.logger.error(f"Ring analysis failed: {exc}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Ring analysis failed", f"{exc}")
            return None
        QtWidgets.QApplication.restoreOverrideCursor()

        self.logger.info(
            f"Ring analysis: {ms.n_analyzed}/{ms.n_segments} segments "
            f"analysed, {len(ms.pairs)} consecutive pair(s)"
        )
        for w in ms.warnings:
            self.logger.warning(f"Ring analysis: {w}")

        if show_window:
            self._show_rings_window(ms, guard_nm)
        return ms

    def _show_rings_window(self, ms: Any, guard_nm: float = 0.0) -> None:
        """Open, or refresh in place, the rings panel."""
        from tools.mps_gaps import analyze_rings
        from tools.mps_rings_window import MPSRingsWindow

        def rerun(mode: str = "valley", guard_nm: float = 0.0):
            result = self.run_ring_analysis(
                show_window=False, mode=mode, guard_nm=guard_nm)
            if result is None:
                raise RuntimeError(
                    "The ring analysis could not be re-run with those "
                    "parameters.")
            return result

        if self.rings_window is None:
            self.rings_window = MPSRingsWindow(
                ms, rerun_callback=rerun, parent=self)
        else:
            self.rings_window.ms = ms
            self.rings_window.rings = analyze_rings(ms)
            self.rings_window.refresh()
        self.rings_window._guard_nm = guard_nm
        self.rings_window.show()
        self.rings_window.raise_()
        self.rings_window.activateWindow()

    def select_file(self, channel: int) -> None:
        """
        Open file dialog to select an HDF5 or CSV file for channel 1 or 2.
        Returns early if user cancels the dialog (filename is empty).

        Parameters
        ----------
        channel : int
            Channel number (1 or 2) to load data into.
        """
        try:
            self.logger.debug(f"File dialog opened for channel {channel}")
            root = Tk()
            root.withdraw()
            if channel == 1:
                root.filenamedata = filedialog.askopenfilename(initialdir=self.initialDir,
                                                               title='Select file')
                if root.filenamedata != '':
                    self.logger.info(f"Channel 1 file selected: {root.filenamedata}")
                    self.load_channel1(root.filenamedata,
                                       int(self.fileformat.currentIndex()))
                else:
                    self.logger.debug("File dialog cancelled for channel 1")
                    return
            elif channel == 2:
                root.filenamedata2 = filedialog.askopenfilename(initialdir=self.initialDir,
                                                                title='Select file')
                if root.filenamedata2 != '':
                    self.logger.info(f"Channel 2 file selected: {root.filenamedata2}")
                    self.load_channel2(root.filenamedata2,
                                       int(self.fileformat_2.currentIndex()))
                else:
                    self.logger.debug("File dialog cancelled for channel 2")
                    return
        except OSError as e:
            self.logger.error(f"Error loading file: {e}", exc_info=True)
            pass

   
    
    def load_channel1(self, path: str, fileformat: int = 0) -> bool:
        """Load a file into channel 1 without the file dialog."""
        try:
            x, y, z = self.import_file(path, fileformat, channel=1)
        except ValueError as error:
            # Nothing has changed yet: the previous file stays loaded.
            self.logger.error(f"Could not load {path}: {error}")
            QtWidgets.QMessageBox.warning(
                self, "Could not load the file",
                f"{os.path.basename(path)} was not loaded:\n{error}")
            return False
        self.xdata, self.ydata, self.zdata = x, y, z
        self.ui.lineEdit_filename.setText(path)
        self.fileformat.setCurrentIndex(fileformat)
        self.fileformat1 = fileformat
        self.logger.debug(f"File format: {['Picasso HDF5', 'ThunderStorm CSV', 'Custom CSV'][self.fileformat1]}")
        # New data means a new axial distribution, so the Z range goes back
        # to being derived automatically.
        self._z_range_user_edited = False
        # Everything derived from the previous file refers to ITS rows. Left
        # in place, the ROI button would cut the old coordinates, the panels
        # would index the new file with old row numbers, and the exports
        # would write the old results under the new file's name. Clearing
        # the scatter makes ROI ask for Scatter first.
        self.x = self.y = self.z = self.data_points = None
        self.xroi = self.yroi = self.zroi = None
        self.xroi_unfiltered = self.yroi_unfiltered = None
        self.zroi_unfiltered = None
        self.roi_indices = None
        self._applied_roi_shape = None
        self._applied_slab = None
        self.cluster_labels = None
        self.original_points = self.original_z = None
        self.cluster_centroids = self.good_cluster_centroids = None
        self.bad_cluster_indices = []
        self._clustered_x[1] = None
        self.distances = None
        self.Nneighbor = None
        self.mps_analysis = None
        self._analysed_x = None
        # Channel 2's selection was cut with channel 1's ROI, which is gone.
        self.xroi2 = self.yroi2 = self.zroi2 = None
        self.cluster_labels2 = self.cluster_centroids2 = None
        self._clustered_x[2] = None
        self._overview_plot = self._ch2_overlay = None
        for window in (self.mps_window, self.rings_window,
                       self.two_channel_window):
            if window is not None:
                window.close()
        # The plots still show the previous file until they are redrawn;
        # channel 2's histogram of its whole file is left alone.
        if self.polygon_drawing_mode:
            self._cleanup_drawing_mode()
        for layout in (
            self.ui.scatterlayout, self.ui.zhistlayoutch1,
            self.ui.scatterlayout_3, self.ui.zhistlayout_2,
            self.ui.scatterlayout_clusterch1, self.ui.scatterlayout_goodclus,
            self.ui.zhistlayout_cmdist, self.ui.scatterlayout_clusterch2,
        ):
            self.empty_layout(layout)
        self.logger.info(f"Channel 1 loaded: {len(x):,} localizations")
        self._check_channel_pixel_sizes()
        return True

    def load_channel2(self, path: str, fileformat: int = 0) -> bool:
        """Load a file into channel 2 without the file dialog."""
        try:
            x, y, z = self.import_file(path, fileformat, channel=2)
        except ValueError as error:
            # Nothing has changed yet: the previous file stays loaded.
            self.logger.error(f"Could not load {path}: {error}")
            QtWidgets.QMessageBox.warning(
                self, "Could not load the file",
                f"{os.path.basename(path)} was not loaded:\n{error}")
            return False
        self.xdata2, self.ydata2, self.zdata2 = x, y, z
        self.ui.lineEdit_filename_2.setText(path)
        self.fileformat_2.setCurrentIndex(fileformat)
        self.fileformat2 = fileformat
        self.logger.debug(f"File format: {['Picasso HDF5', 'ThunderStorm CSV', 'Custom CSV'][self.fileformat2]}")
        # Everything derived from the previous channel-2 file refers to its
        # rows: the selection, the clustering and what was drawn from them.
        self.x2 = self.y2 = self.z2 = self.data_points2 = None
        self.xroi2 = self.yroi2 = self.zroi2 = None
        self.cluster_labels2 = self.cluster_centroids2 = None
        self._clustered_x[2] = None
        if self.two_channel_window is not None:
            self.two_channel_window.close()
        for layout in (self.ui.zhistlayoutch2,
                       self.ui.scatterlayout_clusterch2):
            self.empty_layout(layout)
        if self._ch2_overlay is not None:
            plot, item = self._ch2_overlay
            plot.removeItem(item)
            self._ch2_overlay = None
        self.logger.info(f"Channel 2 loaded: {len(x):,} localizations")
        # Channel 1's overview and selection still stand. Show the new file
        # on them: a redraw would also put the ROI back at its default.
        if self.x is not None and self._overview_plot is not None:
            self._render_channel_2(self._overview_plot)
        if self.xroi is not None and len(self.xroi):
            self._select_channel2()
            self._draw_roi_panels()
        self._check_channel_pixel_sizes()
        return True

    def _get_pixel_size_from_yaml(self, hdf5_filename: str) -> Optional[float]:
        """
        Reads the pixel size recorded for a Picasso HDF5 file.

        Kept under its original name because callers use it, but it no
        longer looks only at the YAML sidecar: since Picasso v0.11 the
        metadata is also embedded in the HDF5 itself (dataset
        '/metadata') and writing the sidecar can be switched off, so a
        perfectly valid current file may have no .yaml at all.

        Parameters
        ----------
        hdf5_filename : str
            Path to the HDF5 file.

        Returns
        -------
        Optional[float]
            Pixel size in nm if found, otherwise None. The caller is
            responsible for handling the None case (e.g., by asking the user).
        """
        return mps_io.read_pixel_size(hdf5_filename)

    def _ask_user_for_pixel_size(self) -> float:
        """
        Prompts the user for the pixel size when it cannot be read from the YAML.
        Defaults to 113 nm (Hamamatsu ORCA-Flash 4.0 with 2x2 binning, the most
        common configuration) and falls back to 133 nm if the user cancels,
        preserving original behavior with a visible warning.

        Returns
        -------
        float
            Pixel size in nm.
        """
        from PyQt5.QtWidgets import QInputDialog
        value, ok = QInputDialog.getDouble(
            self,
            "Pixel size required",
            "Could not read pixel size from YAML.\n"
            "Please enter the effective pixel size in nm:",
            value=113.0, min=1.0, max=1000.0, decimals=2
        )
        if ok:
            return value
        else:
            QtWidgets.QMessageBox.warning(
                self, "Using default pixel size",
                "No pixel size provided. Falling back to 133 nm.\n"
                "WARNING: this is the original hardcoded value and may not\n"
                "match your optical system. All distances will be miscalibrated\n"
                "if the true pixel size differs from 133 nm."
            )
            return 133.0

    def import_file(
        self,
        filename: str,
        fileformat: int,
        channel: int = 1,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Import localization data from HDF5 or CSV files.

        Reads the WHOLE table, not just the coordinates. The extra columns
        are kept on ``self.locs1`` / ``self.locs2``; this method still
        returns only (x, y, z) because that is what the rest of the
        application consumes.

        The loading itself lives in ``tools.mps_io`` so that the GUI and
        the batch runner cannot drift apart on the one decision that
        silently corrupts results -- the pixel size. They differ in
        exactly one place, deliberately: a batch run refuses a file whose
        pixel size is unknown, while here there is a user to ask.

        Parameters
        ----------
        filename : str
            Path to the data file.
        fileformat : int
            File format code: 0=Picasso HDF5, 1=ThunderSTORM CSV, 2=Custom CSV.
        channel : int
            Which channel this file belongs to (1 or 2); decides where the
            full ``Localizations`` is stored.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
            Tuple of (x, y, z) coordinate arrays in nanometers.
        """
        try:
            loc = mps_io.load_localizations(filename, fileformat)
        except ValueError as error:
            if "pixel size" not in str(error).lower():
                raise
            # Neither a YAML sidecar nor an embedded '/metadata'. Whatever
            # the user types (or the fallback) is a guess. Record that,
            # because a wrong pixel size rescales every lateral distance
            # and squares into the cluster areas without ever raising.
            guessed = self._ask_user_for_pixel_size()
            loc = mps_io.load_localizations(
                filename, fileformat, pixel_size_nm=guessed
            )
            loc.pixel_size_source = "manual"
            self.logger.warning(
                f"No Picasso metadata for {os.path.basename(filename)} "
                f"(no YAML sidecar and no embedded '/metadata'); pixel size "
                f"{guessed} nm was supplied manually. All lateral distances "
                f"scale with it and cluster areas scale with its square."
            )

        self.pxsize = loc.pixel_size_nm
        self.pxsize_source = loc.pixel_size_source
        if channel == 2:
            self.locs2 = loc
        else:
            self.locs1 = loc

        if loc.pixel_size_nm is not None:
            self.logger.info(
                f"Using pixel size = {loc.pixel_size_nm} nm "
                f"(source: {loc.pixel_size_source}) for "
                f"{os.path.basename(filename)}")
        self.logger.info(
            f"Read {loc.n:,} localizations, {len(loc.columns)} columns "
            f"({', '.join(sorted(loc.columns))}); metadata from "
            f"{loc.metadata_source}"
            + (f"; steps: {' -> '.join(loc.processing_steps)}"
               if loc.processing_steps else "")
        )
        return loc.x_nm, loc.y_nm, loc.z_nm


    def get_root_filename(self) -> str:
        """
        Returns the base filename without extension from the main filename field.

        Returns
        -------
        str
            Base filename without extension, or "data" if no filename is set.
        """
        filename = self.ui.lineEdit_filename.text()
        if filename:
            return os.path.splitext(os.path.basename(filename))[0]
        return "data"  # Default if no filename is set

    # ========================================================================
    # Hallazgo 05 — Helper methods for scatterplot refactoring
    # ========================================================================

    def _render_scatter_xy_heatmap(self, x_data: NDArray[np.float64], y_data: NDArray[np.float64], plotxy: Any, brush_color: str) -> None:
        """Render X,Y scatter as a 2D density heatmap instead of individual points.

        Hallazgo 13 — Scalable rendering for millions of points.

        PyQtGraph's ScatterPlotItem is slow with >500k points because each
        point is a separate graphics object. This method renders the data as
        a 2D histogram (density map) displayed as a single raster image,
        which is orders of magnitude faster.

        Trade-off: You see density (color intensity), not individual points.
        But the ROI panel still shows individual points (fast, since ROI
        filters to fewer points).

        Parameters
        ----------
        x_data : NDArray[np.float64]
            1D array of x-coordinates
        y_data : NDArray[np.float64]
            1D array of y-coordinates
        plotxy : pyqtgraph.PlotItem
            PlotItem to add the heatmap to
        brush_color : str
            Color string (unused, heatmap has its own colormap)
        """
        # Compute 2D histogram (density map)
        # 400x400 bins balances detail vs. speed (160k pixels, still 6x faster than 969k points)
        xmin, xmax = np.min(x_data), np.max(x_data)
        ymin, ymax = np.min(y_data), np.max(y_data)

        hist2d, xedges, yedges = np.histogram2d(x_data, y_data, bins=HISTOGRAM_2D_BINS)
        hist2d = hist2d.T  # Transpose for correct orientation

        # Log-scale for better visibility of density variations
        # (linear scale would make high-density regions wash out to white)
        hist2d_log = np.log1p(hist2d)

        # Create image item with log-scaled data
        # Let PyQtGraph handle the normalization/display via levels parameter
        img = pg.ImageItem(image=hist2d_log)
        img.setRect(pg.QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))

        # Apply a heatmap colormap (viridis: good for perceptual uniformity)
        cmap = pg.colormap.get('viridis')
        img.setColorMap(cmap)

        # Set display levels to use full colormap range
        img.setLevels([hist2d_log.min(), hist2d_log.max()])

        plotxy.addItem(img)
        plotxy.setXRange(xmin, xmax, padding=0)
        plotxy.setYRange(ymin, ymax, padding=0)

    def _render_scatter_xy_ch1(self, scatterWidgetxy: Any, plotxy: Any) -> None:
        """Render the X,Y scatter plot for channel 1.

        Parameters
        ----------
        scatterWidgetxy : pyqtgraph.GraphicsLayoutWidget
            PyQtGraph GraphicsLayoutWidget to hold the plot
        plotxy : pyqtgraph.PlotItem
            PyQtGraph PlotItem to draw on
        """
        # Use already-loaded channel 1 data (set in select_file)
        self.x = self.xdata
        self.y = self.ydata
        self.z = self.zdata
        self.data_points = np.column_stack((self.x, self.y))

        # Compute axis ranges
        xmin, xmax = np.min(self.x), np.max(self.x)
        ymin, ymax = np.min(self.y), np.max(self.y)

        # Draw scatter points (individual points for maximum visual quality)
        xy = pg.ScatterPlotItem(self.x, self.y, pen=None,
                                brush=self.brush1, size=1)
        plotxy.addItem(xy)
        plotxy.setXRange(xmin, xmax, padding=0)
        plotxy.setYRange(ymin, ymax, padding=0)

        # Update layout
        self.empty_layout(self.ui.scatterlayout)
        self.ui.scatterlayout.addWidget(scatterWidgetxy)

    def _render_z_histogram(self, z_data: NDArray[np.float64], channel: int, brush: Any, pen: Any, layout_widget: Any) -> None:
        """Render a z-axis histogram for a single channel.

        Parameters
        ----------
        z_data : NDArray[np.float64]
            1D array of z-coordinates
        channel : int
            Channel number (1 or 2), for labeling
        brush : pyqtgraph.Brush
            PyQtGraph brush for bar color
        pen : pyqtgraph.Pen
            PyQtGraph pen for bar outline
        layout_widget : pyqtgraph.GraphicsLayoutWidget
            UI layout to place the histogram
        """
        histzWidget = pg.GraphicsLayoutWidget()
        histabsz = histzWidget.addPlot(title=f"z Histogram Ch {channel}")

        histz, bin_edgesz = np.histogram(z_data, bins=HISTOGRAM_Z_BINS)
        widthzabs = np.mean(np.diff(bin_edgesz))
        bincentersz = np.mean(np.vstack([bin_edgesz[0:-1], bin_edgesz[1:]]),
                              axis=0)
        bargraphz = pg.BarGraphItem(x=bincentersz, height=histz,
                                    width=widthzabs, brush=brush, pen=pen)
        histabsz.addItem(bargraphz)

        self.empty_layout(layout_widget)
        layout_widget.addWidget(histzWidget)

    def _setup_roi_widget(self, scatterWidgetxy: Any, plotxy: Any) -> None:
        """Create and configure the interactive ROI (circle or square).

        Parameters
        ----------
        scatterWidgetxy : pyqtgraph.GraphicsLayoutWidget
            GraphicsLayoutWidget containing the scatter
        plotxy : pyqtgraph.PlotItem
            PlotItem to add the ROI to
        """
        npixels = np.size(self.x)
        ROIpos = (int(min(self.x)), int(min(self.y)))
        ROIextent = int(npixels / ROI_EXTENT_DIVISOR)
        ROIpen = pg.mkPen(color='r')

        if self.ui.radioButton_circROI.isChecked():
            # Create circular ROI
            self.circular_roi = pg.CircleROI(ROIpos, ROIextent, movable=True,
                                             pen=ROIpen)
            self.circular_roi.handleColor = CIRCULAR_ROI_COLOR
            self.circular_roi.addScaleHandle([1, 1], [0, 0])
            self.circular_roi.setZValue(ROI_ZORDER)
            plotxy.addItem(self.circular_roi)
            self.circular_roi.sigRegionChangeFinished.connect(self.update_ROI)

        elif self.ui.radioButton_squareROI.isChecked():
            # Create square ROI
            self.square_roi = pg.ROI(ROIpos, ROIextent, pen=ROIpen)
            self.square_roi.setZValue(ROI_ZORDER)
            self.square_roi.addScaleHandle([1, 1], [0, 0])
            self.square_roi.addRotateHandle([0, 0], [1, 1])
            plotxy.addItem(self.square_roi)
            self.square_roi.sigRegionChangeFinished.connect(self.update_ROI)

        elif self.ui.radioButton_polygonROI.isChecked():
            # Create intelligent polygon ROI using ConvexHull of densest region
            self._create_polygon_roi(plotxy, ROIpen)

    def _create_polygon_roi(self, plotxy: Any, roi_pen: Any) -> None:
        """Create a simple circular polygonal ROI for easy manual editing.

        Parameters
        ----------
        plotxy : pyqtgraph.PlotItem
            Plot to add ROI to
        roi_pen : pyqtgraph pen
            Pen for ROI outline
        """
        # Create a simple circle centered at the data center
        # 12 vertices = simple circle, easy to edit
        center_x = np.mean(self.x)
        center_y = np.mean(self.y)

        # Use a reasonable radius based on data spread
        # Use the smaller of the x or y range to avoid too-large polygon
        x_range = np.max(self.x) - np.min(self.x)
        y_range = np.max(self.y) - np.min(self.y)
        radius = min(x_range, y_range) / 6  # 1/6 of the smaller dimension

        # Create simple 4-vertex square (simplest to edit)
        initial_points = self._create_circle_polygon(center_x, center_y, radius, 4)

        self.logger.debug(f"Polygon ROI: Simple square created with 4 vertices at center=({center_x:.1f}, {center_y:.1f}), radius={radius:.1f}")

        # Create PyQtGraph PolyLineROI (note: capital L)
        try:
            self.polygon_roi = pg.PolyLineROI(
                initial_points,
                closed=True,
                movable=True,
                pen=roi_pen
            )
            self.polygon_roi.setZValue(ROI_ZORDER)
            plotxy.addItem(self.polygon_roi)
            self.polygon_roi.sigRegionChangeFinished.connect(self.update_ROI)

            self.logger.debug(f"Polygon ROI created with {len(initial_points)} initial vertices")
            self.logger.info(
                "Polygon ROI ready! Interactive controls:\n"
                "  • Drag vertices to adjust polygon shape\n"
                "  • Click+drag center to move entire polygon\n"
                "  • Right-click to edit polygon (add/remove vertices)"
            )
        except Exception as e:
            self.logger.error(f"Error creating polygon ROI: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Polygon ROI Error",
                f"Failed to create polygon ROI:\n{str(e)}\n\n"
                "Try selecting a different ROI type (Circle or Square)"
            )

    @staticmethod
    def _create_circle_polygon(center_x: float, center_y: float,
                               radius: float, n_vertices: int) -> NDArray[np.float64]:
        """Create polygon vertices approximating a circle.

        Parameters
        ----------
        center_x, center_y : float
            Circle center coordinates
        radius : float
            Circle radius
        n_vertices : int
            Number of vertices to create

        Returns
        -------
        NDArray[np.float64]
            (n_vertices, 2) array of polygon vertices
        """
        theta = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        return np.column_stack([x, y])

    def _render_channel_2(self, plotxy: Any) -> None:
        """Overlay channel 2 on the overview and draw its z histogram.

        If no channel 2 file is loaded, this method does nothing.

        Parameters
        ----------
        plotxy : pyqtgraph.PlotItem
            The overview plot, already placed in its layout.
        """
        if self.xdata2 is None:
            return

        # Use already-loaded channel 2 data
        self.x2 = self.xdata2
        self.y2 = self.ydata2
        self.z2 = self.zdata2
        self.data_points2 = np.column_stack((self.x2, self.y2))

        # Add scatter overlay (individual points for quality)
        xy2 = pg.ScatterPlotItem(self.x2, self.y2, pen=None,
                                 brush=self.brush2, size=1)
        plotxy.addItem(xy2)
        self._ch2_overlay = (plotxy, xy2)

        # Render z-histogram for channel 2
        self._render_z_histogram(self.z2, 2, self.brush2, self.pen2,
                                 self.ui.zhistlayoutch2)

    def scatterplot(self) -> None:
        """Render the overview scatter plot with channel 1 and (optionally) channel 2.

        This method orchestrates helper methods to:
        1. Render the X,Y scatter plot for channel 1
        2. Render the z-histogram for channel 1
        3. Setup the interactive ROI widget (circle or square)
        4. If loaded, overlay channel 2 and its z-histogram

        Hallazgo 05 — Refactored from a 130-line monolith into a dispatcher
        that calls focused helper methods. Each helper does one thing well
        and can be tested/reused independently.
        """
        self.logger.info("Scatterplot: Generating overview plot...")

        # Guard: refuse to run if the user hasn't loaded channel 1 yet.
        if self.xdata is None:
            self.logger.warning("Scatterplot: No data loaded for channel 1")
            QtWidgets.QMessageBox.warning(
                self, "No data loaded",
                "Please load a channel-1 file before drawing the scatter plot."
            )
            return

        # Create the scatter widget and plot
        scatterWidgetxy = pg.GraphicsLayoutWidget()
        plotxy = scatterWidgetxy.addPlot(title="Scatter plot (x,y) both channels")
        plotxy.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotxy.setAspectLocked(True)
        self.logger.debug(f"Scatterplot: Rendering {len(self.xdata):,} points from channel 1")

        # Render channel 1 scatter (sets self.x, self.y, self.z, self.data_points)
        self._render_scatter_xy_ch1(scatterWidgetxy, plotxy)

        # Render channel 1 z-histogram
        self._render_z_histogram(self.z, 1, self.brush1, self.pen1,
                                 self.ui.zhistlayoutch1)

        # Setup ROI (creates self.circular_roi or self.square_roi)
        self._setup_roi_widget(scatterWidgetxy, plotxy)

        # Overlay channel 2 if present
        self._overview_plot = plotxy
        self._ch2_overlay = None
        self._render_channel_2(plotxy)
     
              
    
    def update_ROI(self) -> None:
        """Filter localizations to the current ROI and update the ROI panel."""
        # Guard: scatter plot must have been drawn first so that self.x/y/z
        # and the ROI widget objects exist.
        if self.x is None:
            QtWidgets.QMessageBox.warning(
                self, "No scatter plot",
                "Please draw the scatter plot first (click 'Scatter')."
            )
            return

        if self.ui.radioButton_circROI.isChecked():

            # Get circular ROI position and size
            pos = self.circular_roi.pos()
            size = self.circular_roi.size()

            # Extract scalar from size (PyQtGraph returns Point or QSizeF)
            # Point object has .x() method; QSizeF has .width() method
            if hasattr(size, 'x'):
                size_scalar = float(size.x())  # Point object
            elif hasattr(size, 'width'):
                size_scalar = float(size.width())  # QSizeF-like
            else:
                size_scalar = float(size)  # Fallback

            diameter = ROI_DIAMETER_SCALE_FACTOR * size_scalar
            radius = diameter / 2

            # Calculate the center coordinates
            center_x = float(pos.x()) + size_scalar / 2
            center_y = float(pos.y()) + size_scalar / 2
            
            # Vectorized point-in-circle test. The original implementation
            # iterated in Python over every localization and called
            # np.linalg.norm one point at a time, which becomes prohibitively
            # slow for datasets with millions of points. NumPy can compute all
            # distances in a single vectorized call evaluated in compiled C,
            # typically 100-1000x faster than the Python loop.
            #
            # Note: center is shape (1, 2), self.data_points is shape (N, 2).
            # Broadcasting subtracts the center from every row, then axis=1
            # collapses the x,y components into a single Euclidean distance
            # per point, yielding a 1D array of length N.
            import time
            t_start = time.perf_counter()
            center_flat = np.array([center_x, center_y]).flatten()
            distances = np.linalg.norm(self.data_points - center_flat, axis=1)
            mask = distances <= float(radius)
            ind_inside_roi = np.where(mask)[0]
            points_inside_roi = self.data_points[mask]
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000
            n_points = len(self.data_points)
            n_selected = len(points_inside_roi)
            self.logger.debug(f"ROI Filter Ch1: Vectorized filter over {n_points:,} points "
                             f"-> {n_selected:,} selected in {elapsed_ms:.2f} ms")
            
            # Guard against empty selections that would crash the slicing below.
            if len(points_inside_roi) == 0:
                QtWidgets.QMessageBox.warning(
                    self, "Empty ROI",
                    "The selected ROI contains no localizations. "
                    "Please draw a larger ROI or move it over a denser region."
                )
                return
            
            self.xroi = points_inside_roi[:,0]
            self.yroi = points_inside_roi[:,1]

            self._apply_z_range(ind_inside_roi)

        elif self.ui.radioButton_squareROI.isChecked():
            # Hallazgo 07 — Square ROI with proper guard for empty selection
            #
            # The original logic used np.in1d(indx, indy) where indx/indy
            # are tuples from np.where(), which is incorrect. Here we use
            # vectorized boolean masking like the circular ROI case for
            # consistency and correctness.
            #
            # Get square ROI position and size
            xmin, ymin = self.square_roi.pos()
            xmax, ymax = self.square_roi.pos() + self.square_roi.size()

            # Vectorized boolean mask: True where point is inside the square.
            # A rotated square is tested as the polygon it is drawn as.
            corners = self._rotated_square_corners()
            if corners is not None:
                mask = points_in_polygon(self.data_points, corners)
            else:
                mask = ((self.x > xmin) & (self.x < xmax)
                        & (self.y > ymin) & (self.y < ymax))
            points_inside_roi = self.data_points[mask]

            # Guard: bail out if ROI contains no localizations
            if len(points_inside_roi) == 0:
                QtWidgets.QMessageBox.warning(
                    self, "Empty ROI",
                    "The selected ROI contains no localizations. "
                    "Please draw a larger ROI or move it over a denser region."
                )
                return

            self.xroi = points_inside_roi[:, 0]
            self.yroi = points_inside_roi[:, 1]

            # Get original indices of points inside ROI for z-filtering
            ind_inside_roi = np.where(mask)[0]

            self._apply_z_range(ind_inside_roi)

        elif self.ui.radioButton_polygonROI.isChecked():
            # Polygon ROI filtering using ray-casting algorithm
            vertices = self._polygon_vertices()

            import time
            t_start = time.perf_counter()

            # Vectorized ray-casting for point-in-polygon test
            mask = self._point_in_polygon(self.data_points, vertices)
            points_inside_roi = self.data_points[mask]
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000

            # Guard: empty selection check
            if len(points_inside_roi) == 0:
                QtWidgets.QMessageBox.warning(
                    self, "Empty ROI",
                    "The selected polygon contains no localizations. "
                    "Adjust vertices or move polygon to denser region."
                )
                return

            self.xroi = points_inside_roi[:, 0]
            self.yroi = points_inside_roi[:, 1]

            # Get indices for z-filtering
            ind_inside_roi = np.where(mask)[0]

            self._apply_z_range(ind_inside_roi)

            n_points = len(self.data_points)
            n_selected = len(points_inside_roi)
            self.logger.debug(f"ROI Filter Ch1: Polygon filter over {n_points:,} points "
                             f"-> {n_selected:,} selected in {elapsed_ms:.2f} ms")

        else:

            self.xroi = self.x
            self.yroi = self.y

            self._apply_z_range(None)
            
            
        # Final guard: after all ROI and z-filtering, check if anything remains.
        # This is a belt-and-suspenders check; the earlier guards should prevent
        # this, but aggressive z-filtering can theoretically remove all points.
        if len(self.xroi) == 0:
            # Channel 2's previous selection does not match this one either.
            self.xroi2 = self.yroi2 = self.zroi2 = None
            QtWidgets.QMessageBox.warning(
                self, "No data in ROI after filtering",
                "All localizations were filtered out by the z-range. "
                "Adjust zmin/zmax and try again."
            )
            return

        self._select_channel2()
        self._draw_roi_panels()

    def _select_channel2(self) -> None:
        """
        Select channel 2 with the ROI and axial range applied to channel 1.

        The shape is the one ``_apply_z_range`` recorded, tested by the
        same rules as channel 1's, and the range is the one it read and
        checked: a range it ignored for channel 1 is ignored here too.

        The range is channel 1's z, so it applies only when both channels
        have one. Channel 2 may be another dye imaged in 2D: its z is all
        zeros, and a slab away from z = 0 would remove all of it.
        """
        self.xroi2 = self.yroi2 = self.zroi2 = None
        if self.xdata2 is None:
            return
        x2 = np.asarray(self.xdata2, dtype=float)
        y2 = np.asarray(self.ydata2, dtype=float)
        z2 = np.asarray(self.zdata2, dtype=float)
        shape = self._applied_roi_shape
        # No shape means channel 1 was taken whole.
        keep = (np.ones(x2.size, dtype=bool) if shape is None
                else points_in_roi(x2, y2, shape))
        z_range = ((self.zmin, self.zmax)
                   if self.zmin is not None and self.zmax is not None
                   else None)
        if z_range is not None and not all(
                loc is None or loc.is_3d for loc in (self.locs1, self.locs2)):
            z_range = None
            self.logger.info(
                "Channel 1 or channel 2 has no z: the axial range is not "
                "applied to channel 2.")
        if z_range is not None:
            keep &= (z2 > z_range[0]) & (z2 < z_range[1])
        self.logger.debug(
            f"ROI Filter Ch2: {int(keep.sum()):,} of {x2.size:,} "
            f"localizations selected")
        if not np.any(keep):
            in_range = " inside the Z range" if z_range is not None else ""
            QtWidgets.QMessageBox.warning(
                self, "Empty ROI in channel 2",
                f"The selected ROI contains no localizations in channel 2"
                f"{in_range}."
            )
            return
        self.xroi2, self.yroi2, self.zroi2 = x2[keep], y2[keep], z2[keep]

    def _add_roi_histogram(self, plot: Any, z: NDArray[np.float64],
                           brush: Any, pen: Any) -> None:
        """Add one channel's axial histogram to the ROI histogram plot."""
        histz2, bin_edgesz2 = np.histogram(z, bins='auto')
        widthzabs2 = np.mean(np.diff(bin_edgesz2))
        bincentersz2 = np.mean(np.vstack([bin_edgesz2[0:-1],bin_edgesz2[1:]]), axis=0)
        bargraphz2 = pg.BarGraphItem(x = bincentersz2, height = histz2,
                                    width = widthzabs2, brush = brush, pen = pen)
        bargraphz2.setOpacity(0.5)
        plot.addItem(bargraphz2)

    def _draw_roi_panels(self) -> None:
        """Draw both channels' selections in the ROI scatter and histogram."""
        scatterWidgetROI = pg.GraphicsLayoutWidget()
        plotROI = scatterWidgetROI.addPlot(title="Scatter plot ROI selected")
        plotROI.setAspectLocked(True)
        plotROI.setLabels(bottom=('x [nm]'), left=('y [nm]'))

        histzWidget2 = pg.GraphicsLayoutWidget()
        histabsz2 = histzWidget2.addPlot(title="z ROI Histogram")

        shown = [self.xroi]
        self.selected = pg.ScatterPlotItem(self.xroi, self.yroi, pen = self.pen1,
                                           brush = None, size = GOOD_CLUSTER_POINT_SIZE)
        plotROI.addItem(self.selected)
        self._add_roi_histogram(histabsz2, self.zroi, self.brush1, self.pen1)

        if self.xroi2 is not None:
            self.selected2 = pg.ScatterPlotItem(self.xroi2, self.yroi2, pen = self.pen2,
                                                brush = None, size = GOOD_CLUSTER_POINT_SIZE)
            plotROI.addItem(self.selected2)
            shown.append(self.xroi2)
            self._add_roi_histogram(histabsz2, self.zroi2, self.brush2, self.pen2)

        x_shown = np.concatenate(shown)
        plotROI.setXRange(np.min(x_shown), np.max(x_shown), padding=0)

        self.empty_layout(self.ui.scatterlayout_3)
        self.ui.scatterlayout_3.addWidget(scatterWidgetROI)
        self.empty_layout(self.ui.zhistlayout_2)
        self.ui.zhistlayout_2.addWidget(histzWidget2)

    # ========================================================================
    # PHASE 1-10: INTERACTIVE POLYGON DRAWING MODE IMPLEMENTATION
    # ========================================================================

    def _start_polygon_drawing_mode(self, plotxy: Any, roi_pen: Any) -> None:
        """Enter interactive polygon drawing mode.

        User clicks to place vertices, presses Enter/Escape to finish drawing.

        Parameters
        ----------
        plotxy : pyqtgraph.PlotItem
            Plot to draw on
        roi_pen : pyqtgraph pen
            Pen for drawing visualization
        """
        self.polygon_drawing_mode = True
        self.polygon_points_temp = []
        self.current_plot = plotxy
        self.current_roi_pen = roi_pen
        self.current_viewbox = plotxy.getViewBox()

        # Connect click handler DIRECTLY to the ViewBox's mouse clicked signal
        # This is more direct than connecting to scene()
        self.mouse_click_handler = self.current_viewbox.scene().sigMouseClicked.connect(
            self._on_polygon_click
        )

        # Create status label with instructions
        self._show_drawing_status("Click to place vertices. Press ENTER to finish. ESC to cancel.")

        self.logger.info("Polygon drawing mode activated - waiting for user clicks")

    def _on_polygon_click(self, event: Any) -> None:
        """Handle mouse clicks during polygon drawing mode.

        Each click adds a vertex to the polygon being drawn.

        Parameters
        ----------
        event : pyqtgraph MouseClickEvent
            Mouse click event from the plot
        """
        # Guard: only process clicks if in drawing mode
        if not self.polygon_drawing_mode:
            return

        # Ignore double-clicks
        if event.double():
            return

        # Get the ViewBox (should be set during mode activation)
        if self.current_viewbox is None:
            self.logger.error("current_viewbox is None!")
            return

        try:
            # Get mouse position in scene coordinates
            scene_pos = event.scenePos()

            # Get the ViewBox's viewport rectangle in scene coordinates
            vb_rect = self.current_viewbox.sceneBoundingRect()

            # Check if click is within the plot area
            if not vb_rect.contains(scene_pos):
                self.logger.debug(f"Click outside ViewBox: scene_pos=({scene_pos.x():.1f}, {scene_pos.y():.1f}), vb_rect={vb_rect}")
                return

            # Transform: scene coordinates → view (data) coordinates
            # Use the ViewBox's transformation matrix directly
            click_point = self.current_viewbox.mapSceneToView(scene_pos)

            x = float(click_point.x())
            y = float(click_point.y())

            self.logger.info(f"Click registered: x={x:.2f}, y={y:.2f}")
            self.logger.info(f"Data range: x=[{self.x.min():.2f}, {self.x.max():.2f}], y=[{self.y.min():.2f}, {self.y.max():.2f}]")

            self.polygon_points_temp.append([x, y])

            # Update visualization
            self._update_polygon_drawing_visual()

            # Show status with vertex count
            self._show_drawing_status(
                f"Vertices: {len(self.polygon_points_temp)} | "
                f"ENTER to finish | ESC to cancel"
            )
        except Exception as e:
            self.logger.error(f"Error in _on_polygon_click: {e}", exc_info=True)

    def _update_polygon_drawing_visual(self) -> None:
        """Update polyline visualization during drawing.

        Shows a live polyline connecting the clicked points as the user draws.
        """
        # Need at least 2 points to draw a line
        if len(self.polygon_points_temp) < 2:
            return

        # Remove old visual if exists
        if self.polygon_drawing_visual is not None:
            self.current_plot.removeItem(self.polygon_drawing_visual)

        # Create temporary polyline connecting clicked points (not closed yet)
        points = np.array(self.polygon_points_temp)
        try:
            self.polygon_drawing_visual = pg.PolyLineROI(
                points,
                closed=False,  # Not closed yet, only shows clicked points
                movable=False,  # Can't move while drawing
                pen=self.current_roi_pen
            )
            self.current_plot.addItem(self.polygon_drawing_visual)
            self.logger.debug(f"Updated drawing visual with {len(points)} points")
        except Exception as e:
            self.logger.error(f"Error updating drawing visual: {e}", exc_info=True)

    def _finish_polygon_drawing(self) -> None:
        """Complete polygon drawing and create PolyLineROI.

        Called when user presses ENTER. Creates the final closed polygon
        and integrates it with the clustering system.
        """
        # Validate minimum vertices
        if len(self.polygon_points_temp) < 3:
            QtWidgets.QMessageBox.warning(
                self, "Too Few Vertices",
                "Polygon must have at least 3 vertices. You have "
                f"{len(self.polygon_points_temp)}."
            )
            return

        try:
            # Create the final PolyLineROI (closed this time)
            points = np.array(self.polygon_points_temp)
            self.polygon_roi = pg.PolyLineROI(
                points,
                closed=True,
                movable=True,
                pen=self.current_roi_pen
            )

            self.polygon_roi.setZValue(ROI_ZORDER)
            self.current_plot.addItem(self.polygon_roi)
            self.polygon_roi.sigRegionChangeFinished.connect(self.update_ROI)

            # Clean up drawing mode
            self._cleanup_drawing_mode()

            self.logger.info(
                f"Polygon drawing complete: {len(points)} vertices. "
                "Polygon ready for clustering (vertices can still be edited)."
            )

            # Trigger ROI update to show filtered points
            self.update_ROI()

        except Exception as e:
            self.logger.error(f"Error finishing polygon drawing: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Polygon Creation Error",
                f"Failed to create polygon:\n{str(e)}"
            )
            self._cleanup_drawing_mode()

    def _cancel_polygon_drawing(self) -> None:
        """Cancel polygon drawing and return to normal state.

        Called when user presses ESC. Resets to circular ROI.
        """
        self._cleanup_drawing_mode()
        self.logger.info("Polygon drawing cancelled by user")

        # Reset to circular ROI
        self.ui.radioButton_circROI.setChecked(True)
        self.scatterplot()

    def _cleanup_drawing_mode(self) -> None:
        """Clean up drawing mode state and visual elements.

        Called after drawing is finished or cancelled.
        """
        # Remove status label
        if self.polygon_drawing_label is not None:
            self.ui.scatterlayout.removeWidget(self.polygon_drawing_label)
            self.polygon_drawing_label.deleteLater()
            self.polygon_drawing_label = None

        # Remove temporary visual
        if self.polygon_drawing_visual is not None:
            self.current_plot.removeItem(self.polygon_drawing_visual)
            self.polygon_drawing_visual = None

        # Disconnect mouse handler
        if self.mouse_click_handler is not None and self.current_plot is not None:
            try:
                self.current_plot.getViewBox().scene().sigMouseClicked.disconnect(
                    self.mouse_click_handler
                )
            except Exception as e:
                self.logger.debug(f"Error disconnecting mouse handler: {e}")
            self.mouse_click_handler = None

        # Reset state
        self.polygon_drawing_mode = False
        self.polygon_points_temp = []
        self.current_plot = None
        self.current_viewbox = None
        self.current_roi_pen = None

    def _show_drawing_status(self, message: str) -> None:
        """Show status message to user during drawing.

        Displays a colored label with instructions/feedback.

        Parameters
        ----------
        message : str
            Status message to display
        """
        # Remove old label if exists
        if self.polygon_drawing_label is not None:
            self.ui.scatterlayout.removeWidget(self.polygon_drawing_label)
            self.polygon_drawing_label.deleteLater()

        # Create new label with styling
        self.polygon_drawing_label = QtWidgets.QLabel(message)
        self.polygon_drawing_label.setStyleSheet(
            "QLabel { background-color: #ffffcc; padding: 10px; "
            "border: 2px solid #ffcc00; border-radius: 4px; "
            "font-weight: bold; font-size: 11px; }"
        )
        self.polygon_drawing_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.scatterlayout.addWidget(self.polygon_drawing_label)
        self.logger.debug(f"Status: {message}")

    def keyPressEvent(self, event: Any) -> None:
        """Handle keyboard input during polygon drawing.

        ENTER: Finish drawing and create polygon
        ESC: Cancel drawing and return to circular ROI

        Parameters
        ----------
        event : QKeyEvent
            Keyboard event
        """
        # Check if we're in polygon drawing mode
        if not self.polygon_drawing_mode:
            super().keyPressEvent(event)
            return

        # Handle ENTER key to finish drawing
        if event.key() == QtCore.Qt.Key_Return:
            self._finish_polygon_drawing()
            event.accept()
        # Handle ESC key to cancel drawing
        elif event.key() == QtCore.Qt.Key_Escape:
            self._cancel_polygon_drawing()
            event.accept()
        else:
            # Pass other keys to parent
            super().keyPressEvent(event)

    def _point_in_polygon(self, points: NDArray[np.float64],
                          polygon: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Boolean mask of the (N, 2) ``points`` inside ``polygon``."""
        return points_in_polygon(points, polygon)

    def _apply_polygon_smoothing(self, polygon: NDArray[np.float64],
                                 smoothness: int = 5) -> NDArray[np.float64]:
        """Apply spline smoothing to polygon vertices.

        Optional: Creates smooth curves through vertices for realistic axon outlines

        Parameters
        ----------
        polygon : NDArray[np.float64]
            (M, 2) array of polygon vertices
        smoothness : int
            Number of interpolated points between vertices

        Returns
        -------
        NDArray[np.float64]
            (M*smoothness, 2) smoothed polygon vertices
        """
        try:
            from scipy.interpolate import CubicSpline
        except ImportError:
            self.logger.warning("scipy.interpolate not available, returning original polygon")
            return polygon

        # Handle small polygons
        if len(polygon) < 4:
            return polygon

        # Create closed spline by adding first point at end
        x = np.concatenate([polygon[:, 0], [polygon[0, 0]]])
        y = np.concatenate([polygon[:, 1], [polygon[0, 1]]])

        # Parameter t goes 0 to len(polygon)
        t = np.arange(len(x))

        # Create cubic splines for x and y with periodic boundary conditions
        cs_x = CubicSpline(t, x, bc_type='periodic')
        cs_y = CubicSpline(t, y, bc_type='periodic')

        # Evaluate at higher resolution
        t_smooth = np.linspace(0, len(polygon), len(polygon) * smoothness, endpoint=False)
        x_smooth = cs_x(t_smooth)
        y_smooth = cs_y(t_smooth)

        return np.column_stack([x_smooth, y_smooth])

    def savexyzROI(self, channel: int) -> None:
        """
        Save ROI-selected localizations to a CSV file in ThunderSTORM format.

        Exports the (x, y, z) coordinates of all localizations within the selected
        Region of Interest (ROI) to a CSV file compatible with ThunderSTORM and other
        analysis software. The output includes columns for x [nm], y [nm], z [nm],
        and optionally cluster labels if clustering has been performed.

        Parameters
        ----------
        channel : int
            Channel to save (1 or 2).

        Raises
        ------
        ValueError
            If channel is not 1 or 2.

        Notes
        -----
        Output filename: `{original_filename}_ch{channel}_roi.csv`
        Format: ThunderSTORM-compatible CSV with headers.
        """
        # Get root filename
        root_name = self.get_root_filename()
        
        # Get the data based on channel
        if channel == 1:
            x_roi = self.xroi
            y_roi = self.yroi
            z_roi = self.zroi
            labels = self.dblabels if hasattr(self, 'dblabels') else None
            suffix = f"_ch{channel}_roi"
            if x_roi is None:
                QtWidgets.QMessageBox.warning(
                    self, "No ROI", "Select a ROI in channel 1 first.")
                return
        elif channel == 2:
            x_roi = self.xroi2
            y_roi = self.yroi2
            z_roi = self.zroi2
            labels = self.dblabels2 if hasattr(self, 'dblabels2') else None
            suffix = f"_ch{channel}_roi"
            if x_roi is None:
                QtWidgets.QMessageBox.warning(
                    self, "No ROI",
                    "Load a file in channel 2 and select a ROI that holds "
                    "channel-2 localizations first.")
                return
        else:
            raise ValueError("Invalid channel number")
    
        # Create ThunderSTORM-compatible DataFrame
        data = {
            'x [nm]': x_roi,
            'y [nm]': y_roi,
            'z [nm]': z_roi,
        }
        
        # Add cluster labels if available
        if labels is not None:
            data['cluster_id'] = labels
        
        df = pd.DataFrame(data)
        
        # Open file dialog with suggested filename
        file_dialog = QFileDialog()
        default_filename = f"{root_name}{suffix}.csv"
        filename, _ = file_dialog.getSaveFileName(
            caption="Save ROI Data with Clusters",
            directory=default_filename,  # Suggest the default filename
            filter="CSV Files (*.csv)"
        )
        
        if filename:
            try:
                df.to_csv(filename, index=False, float_format='%.2f')
                self.logger.info(f"Saved ROI data to: {filename} ({len(df)} rows)")
            except Exception as e:
                self.logger.error(f"Error saving ROI data to {filename}: {e}", exc_info=True)
                QtWidgets.QMessageBox.critical(
                    self, "Save Error",
                    f"Failed to save file: {str(e)}"
                )
        else:
            self.logger.debug("Save operation cancelled by user")

            
    
    def savedistdata(self) -> None:
        """
        Save k-nearest neighbor distances to a CSV file.

        Exports the distances from each good cluster centroid to its k nearest neighbors
        to a CSV file. This data is useful for analyzing cluster spacing and detecting
        potential artifacts or incomplete clustering.

        The output contains columns for each neighbor (1st NN, 2nd NN, etc.) with
        distances in nanometers.

        Raises
        ------
        UserWarning
            If distances have not been computed yet (user must run KNdist_hist first).

        Notes
        -----
        Output filename: `{original_filename}_distances.csv`
        """
        # Guard: distances must have been computed first.
        if self.distances is None or self.Nneighbor is None:
            QtWidgets.QMessageBox.warning(
                self, "No distance data",
                "Please compute the nearest-neighbour distances first."
            )
            return
        Nneighbor = int(self.Nneighbor)
        dist = self.distances
        
        # Get root filename
        root_name = self.get_root_filename()
        default_filename = f"{root_name}_{Nneighbor}neighbor_distances.csv"
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Distance Data",
            default_filename,
            "CSV Files (*.csv)"
        )
        
        if filename:
            np.savetxt(filename, dist, delimiter=",", fmt="%.2f")

    def on_algorithm_changed(self, algorithm: str) -> None:
        """Handle algorithm selection change to show/hide algorithm-specific parameters.

        Parameters
        ----------
        algorithm : str
            Selected algorithm: "Auto", "DBSCAN", or "HDBSCAN"
        """
        # Show/hide parameters based on algorithm selection
        if algorithm == "DBSCAN":
            # DBSCAN requires epsilon; HDBSCAN doesn't use it
            self.ui.label_eps.show()
            self.ui.lineEdit_eps.show()
            self.ui.label_eps_2.show()
            self.ui.lineEdit_eps_2.show()

            # HDBSCAN-specific parameter (min_cluster_size) is not used in DBSCAN
            self.ui.label_minclustersize.hide()
            self.ui.lineEdit_minclustersize.hide()

            self.logger.debug("Algorithm changed to DBSCAN - showing epsilon, hiding min_cluster_size")

        elif algorithm == "HDBSCAN":
            # HDBSCAN doesn't use epsilon but uses min_cluster_size
            self.ui.label_eps.hide()
            self.ui.lineEdit_eps.hide()
            self.ui.label_eps_2.hide()
            self.ui.lineEdit_eps_2.hide()

            # HDBSCAN-specific parameter
            self.ui.label_minclustersize.show()
            self.ui.lineEdit_minclustersize.show()

            self.logger.debug("Algorithm changed to HDBSCAN - hiding epsilon, showing min_cluster_size")

        elif algorithm == "Auto":
            # Auto mode uses all parameters (user can set any of them)
            self.ui.label_eps.show()
            self.ui.lineEdit_eps.show()
            self.ui.label_eps_2.show()
            self.ui.lineEdit_eps_2.show()
            self.ui.label_minclustersize.show()
            self.ui.lineEdit_minclustersize.show()

            self.logger.debug("Algorithm changed to Auto - showing all parameters")


    def cluster(self, channel: int) -> None:
        """Perform DBSCAN clustering on the selected ROI data and visualize results.

        Parameters
        ----------
        channel : int
            1 for primary channel, 2 for secondary channel.
        """
        self.logger.info(f"Clustering: Starting DBSCAN on channel {channel}")

        # Guard: ROI must have been selected first.
        if channel == 1 and self.xroi is None:
            self.logger.warning("Clustering Ch1: No ROI selected")
            QtWidgets.QMessageBox.warning(
                self, "No ROI selected",
                "Please select an ROI before clustering channel 1."
            )
            return
        if channel == 2 and self.xroi2 is None:
            self.logger.warning("Clustering Ch2: No ROI selected")
            QtWidgets.QMessageBox.warning(
                self, "No ROI selected",
                "Please load a channel-2 file and select an ROI that holds "
                "channel-2 localizations before clustering channel 2."
            )
            return

        # Reset bad-cluster list for each new clustering run so stale
        # selections from a previous run don't carry over. The list holds
        # channel 1's clusters (run_mps_analysis fills it), so clustering
        # channel 2 leaves it alone.
        if channel == 1:
            self.bad_cluster_indices = []
            # (Legacy attribute kept for any code that may still reference it)
            self.indbc = []

        # Channel-specific data setup
        if channel == 1:
            # Channel 1 data and parameters
            x_roi = self.xroi
            y_roi = self.yroi
            z_roi = self.zroi
            roi_brush = self.brush1  # Channel 1 color
            roi_pen = self.pen1      # Channel 1 border color
            scatter_layout_cluster = self.ui.scatterlayout_clusterch1  # Target UI layout
            eps_input = self.ui.lineEdit_eps.text()
            minsamples_input = self.ui.lineEdit_minsamples.text()

        elif channel == 2:
            # Channel 2 data and parameters
            x_roi = self.xroi2
            y_roi = self.yroi2
            z_roi = self.zroi2
            roi_brush = self.brush2  # Channel 2 color
            roi_pen = self.pen2      # Channel 2 border color
            scatter_layout_cluster = self.ui.scatterlayout_clusterch2  # Target UI layout
            eps_input = self.ui.lineEdit_eps_2.text()
            minsamples_input = self.ui.lineEdit_minsamples_2.text()
        else:
            return  # Invalid channel

        # Read algorithm selection (Auto/DBSCAN/HDBSCAN)
        algorithm_selection = self.algorithm_selector.currentText()
        minclustersize_input = self.ui.lineEdit_minclustersize.text().strip()

        # Prepare XY coordinate array
        roi_points = np.column_stack((x_roi, y_roi))
        n_roi_points = len(roi_points)
        self.logger.debug(f"Clustering Ch{channel}: Processing {n_roi_points:,} ROI points")
        self.logger.debug(f"Clustering Ch{channel}: Algorithm={algorithm_selection}, MinClusterSize={minclustersize_input}")

        # Parameter handling: Support "auto" or numeric input
        # "auto" triggers adaptive parameter estimation (with Phase 4 caching)
        # Algorithm-specific parameter validation
        try:
            # Determine strategy type based on algorithm selection
            if algorithm_selection == "Auto":
                strategy_type = "auto"
            elif algorithm_selection == "DBSCAN":
                strategy_type = "dbscan"
            elif algorithm_selection == "HDBSCAN":
                strategy_type = "hdbscan"
            else:
                strategy_type = "auto"

            # DBSCAN and Auto modes require epsilon
            if strategy_type in ["dbscan", "auto"]:
                if eps_input.lower().strip() == "auto":
                    # Phase 4: Check cache first for similar dataset
                    cached_params = self.param_cache.get_cached_parameters(roi_points)

                    if cached_params:
                        # Use cached parameters (scientific quality UNCHANGED)
                        self.eps = cached_params.eps
                        self.minsamples = cached_params.min_samples
                        use_auto_eps = True
                        use_auto_ms = True
                        source = "cache"
                    else:
                        # Estimate fresh parameters and cache them
                        import time
                        start_time = time.time()
                        self.eps = clustering.estimate_optimal_eps(roi_points, k=5, percentile=90)
                        estimation_time = (time.time() - start_time) * 1000  # Convert to ms

                        # Also estimate min_samples in auto mode
                        self.minsamples = clustering.estimate_min_samples(n_roi_points, dimensionality=2)

                        # Cache the estimated parameters
                        self.param_cache.cache_parameters(
                            roi_points,
                            self.eps,
                            int(self.minsamples),
                            estimation_time_ms=estimation_time,
                            source="estimated"
                        )
                        use_auto_eps = True
                        use_auto_ms = True
                        source = "estimated"
                else:
                    self.eps = float(eps_input)
                    use_auto_eps = False
                    source = "manual"
            else:
                # HDBSCAN mode: epsilon is not used
                self.eps = None
                use_auto_eps = False
                source = "manual"

            # Min samples handling (common to DBSCAN and HDBSCAN)
            if minsamples_input.lower().strip() == "auto":
                # If eps was auto and came from cache, min_samples already set
                if source != "cache":
                    self.minsamples = clustering.estimate_min_samples(n_roi_points, dimensionality=2)
                use_auto_ms = True
            else:
                self.minsamples = int(float(minsamples_input))
                use_auto_ms = False

            # HDBSCAN-specific: min_cluster_size parameter
            if strategy_type == "hdbscan":
                if minclustersize_input.lower() == "auto":
                    # Auto estimate for HDBSCAN min_cluster_size
                    self.min_cluster_size = max(5, int(np.sqrt(n_roi_points)))
                    use_auto_mcs = True
                else:
                    try:
                        self.min_cluster_size = int(float(minclustersize_input))
                        if self.min_cluster_size < 1:
                            raise ValueError("Min Cluster Size must be >= 1")
                        use_auto_mcs = False
                    except (ValueError, AttributeError) as e:
                        self.logger.error(f"Clustering Ch{channel}: Invalid min_cluster_size: {e}")
                        QtWidgets.QMessageBox.warning(
                            self, "Invalid Input",
                            "HDBSCAN Min Cluster Size must be numeric or 'auto'.\n"
                            "Examples: 5, 10, auto"
                        )
                        return
            else:
                self.min_cluster_size = None
                use_auto_mcs = False

            # Log parameters based on algorithm
            if strategy_type == "hdbscan":
                self.logger.info(
                    f"Clustering Ch{channel}: Algorithm=HDBSCAN, "
                    f"min_samples={int(self.minsamples)} "
                    f"({'auto-detected' if use_auto_ms else 'manual'}), "
                    f"min_cluster_size={int(self.min_cluster_size)} "
                    f"({'auto-detected' if use_auto_mcs else 'manual'})"
                )
            else:
                self.logger.info(
                    f"Clustering Ch{channel}: Algorithm={algorithm_selection}, "
                    f"eps={self.eps:.3f} "
                    f"({'cache' if source == 'cache' else 'auto-detected' if use_auto_eps else 'manual'}), "
                    f"min_samples={int(self.minsamples)} "
                    f"({'auto-detected' if use_auto_ms else 'manual'})"
                )

        except (ValueError, AttributeError) as e:
            self.logger.error(f"Clustering Ch{channel}: Invalid parameters: {e}")
            QtWidgets.QMessageBox.warning(
                self, "Invalid Input",
                f"Invalid clustering parameters: {str(e)}\n"
                f"Algorithm: {algorithm_selection}\n"
                "Examples: eps=1.0, min_samples=10, min_cluster_size=5"
            )
            return

        # Perform clustering using strategy pattern (with manual algorithm control)
        try:
            # Create clustering strategy: uses user-selected algorithm or auto-selection
            # strategy_type can be "auto" (adaptive), "dbscan", or "hdbscan"
            # HDBSCAN-specific parameters are passed if needed
            if strategy_type == "hdbscan":
                strategy = create_clustering_strategy(
                    strategy_type="hdbscan",
                    min_samples=int(self.minsamples),
                    min_cluster_size=int(self.min_cluster_size),
                    metric="euclidean",
                    logger=self.logger
                )
            else:
                # DBSCAN or Auto modes use epsilon
                strategy = create_clustering_strategy(
                    strategy_type=strategy_type,
                    eps=self.eps,
                    min_samples=int(self.minsamples),
                    metric="euclidean",
                    logger=self.logger
                )

            # Perform clustering with selected strategy
            cluster_assignments = strategy.fit(roi_points)  # Get cluster labels (-1 for noise)
            strategy_name = strategy.get_strategy_name()

            self.logger.info(f"Clustering Ch{channel}: Using {strategy_name}")

        except ImportError as e:
            self.logger.error(f"Clustering: Required library not available: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Clustering Error",
                f"Clustering library not available: {str(e)}\n"
                f"Please install required dependencies with: pip install hdbscan"
            )
            return
        except Exception as e:
            self.logger.error(f"Clustering: Strategy-based clustering failed: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Clustering Error",
                f"Clustering failed: {str(e)}"
            )
            return

        # Calculate cluster centers (centroids)
        unique_labels = np.unique(cluster_assignments)
        centroids_list = []
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
            cluster_points = roi_points[cluster_assignments == label]
            centroids_list.append(np.mean(cluster_points, axis=0))  # Calculate centroid

        # Rounded cluster centers; (0, 2) when nothing clustered, so the
        # plot below still gets two columns
        centroids = np.around(np.array(centroids_list), decimals=2).reshape(-1, 2)

        # Store clustering results, each channel in its own attributes
        if channel == 1:
            self.cluster_labels = cluster_assignments  # Array assigning each point to a cluster (or -1 for noise)
            self.original_points = roi_points          # Store original coordinates for reference
            self.original_z = z_roi                    # Store original z-values
            self.cluster_centroids = centroids
        else:
            self.cluster_labels2 = cluster_assignments
            self.cluster_centroids2 = centroids
        self._clustered_x[channel] = x_roi

        # Log clustering statistics
        n_clusters = len(centroids)
        n_noise = np.sum(cluster_assignments == -1)
        self.logger.info(f"Clustering Ch{channel}: Found {n_clusters} clusters, {n_noise:,} noise points")

        # Analyze clustering quality and suggest parameter adjustments if needed
        quality_stats = clustering.analyze_clustering_quality(cluster_assignments, n_roi_points)
        self.logger.info(
            f"Clustering Quality Ch{channel}: {quality_stats['quality_assessment']} "
            f"(noise: {quality_stats['noise_percentage']:.1f}%)"
        )

        # Suggest parameter adjustments if results are suboptimal
        suggested_eps, _ = clustering.suggest_parameter_adjustment(
            current_eps=self.eps,
            current_min_samples=int(self.minsamples),
            labels=cluster_assignments,
            n_points=n_roi_points,
            logger=self.logger
        )

        # If suggestions are available and clustering quality is poor, show user
        if suggested_eps is not None and quality_stats["n_clusters"] == 0:
            self.logger.warning(
                f"Clustering Ch{channel}: No clusters found. Consider adjusting parameters. "
                f"Current eps={self.eps:.2f}, suggested eps={suggested_eps:.2f}"
            )
            # Show suggestion as a message to the user
            QtWidgets.QMessageBox.information(
                self, "Parameter Adjustment Suggestion",
                f"No clusters were found with current parameters.\n\n"
                f"Current: eps={self.eps:.2f}, min_samples={int(self.minsamples)}\n"
                f"Suggested: eps={suggested_eps:.2f}\n\n"
                f"Try adjusting parameters or enter 'auto' for automatic detection."
            )

        # Create cluster visualization
        scatterWidgetcluster = pg.GraphicsLayoutWidget()
        plotclusters = scatterWidgetcluster.addPlot(title="Clustered data")
        plotclusters.setAspectLocked(True)  # Maintain aspect ratio
        plotclusters.setLabels(bottom='x [nm]', left='y [nm]')
        
        # Plot points for each cluster
        for label in unique_labels:
            if label == -1:  # Noise points
                noise_points = roi_points[cluster_assignments == -1]
                noise_plot = pg.ScatterPlotItem(
                    noise_points[:, 0], noise_points[:, 1],
                    pen=roi_pen, brush=None, size=NOISE_POINT_SIZE, symbol='x'  # Cross symbol for noise
                )
                plotclusters.addItem(noise_plot)
            else:  # Cluster points
                cluster_points = roi_points[cluster_assignments == label]
                cluster_plot = pg.ScatterPlotItem(
                    cluster_points[:, 0], cluster_points[:, 1],
                    pen=roi_pen, brush=None, size=CLUSTER_CENTROID_POINT_SIZE  # Hollow circles for cluster members
                )
                plotclusters.addItem(cluster_plot)
        
        # Plot cluster centers
        self.selectedcluscm = pg.ScatterPlotItem(
            centroids[:, 0], centroids[:, 1],
            size=CLUSTER_CENTROID_POINT_SIZE, pen=pg.mkPen('k'), brush=roi_brush  # Filled circles for centers
        )
        plotclusters.addItem(self.selectedcluscm)

        # Bad clusters used to be marked by clicking each centroid (rx).
        # They are now identified automatically and objectively -- clusters
        # touching the ROI boundary, whose centroid and area are biased by
        # the arbitrary cut, plus clusters failing the DBCV density-validity
        # index. The click handler is therefore no longer connected.

        # Update UI with new plot
        self.empty_layout(scatter_layout_cluster)
        scatter_layout_cluster.addWidget(scatterWidgetcluster)

        # Automatic curation + full per-axon analysis, on channel 1 only:
        # betaII-spectrin is always loaded there.
        if channel == 1:
            self._persist_mps_settings()
            self.run_mps_analysis(
                show_window=self.mps_settings.auto_analyze_on_cluster)

    def cluster_both_channels(self) -> None:
        """
        Cluster both channels in parallel for improved performance.

        Phase 3: Parallel Processing
        Executes clustering for Ch1 and Ch2 simultaneously using ThreadPoolExecutor.
        Expected 1.8-2.0x speedup compared to sequential clustering.

        Sequential: ~200ms (100ms Ch1 + 100ms Ch2)
        Parallel:   ~110ms (max(100ms Ch1, 100ms Ch2))

        Returns
        -------
        None
            Updates UI with clustering results for both channels
        """
        self.logger.info("Starting parallel clustering for both channels...")

        # Create task dictionary for parallel execution
        clustering_tasks = {
            1: lambda: self.cluster(channel=1),
            2: lambda: self.cluster(channel=2)
        }

        channel_names = {
            1: "Channel 1",
            2: "Channel 2"
        }

        # Execute clustering in parallel
        try:
            with create_parallel_clustering_manager(
                max_workers=2,
                logger=self.logger
            ) as manager:
                # Define progress callback to show status
                def on_channel_progress(channel_id: int, status: str) -> None:
                    """Progress callback for clustering tasks."""
                    channel_name = channel_names[channel_id]
                    self.logger.info(f"{channel_name}: {status}")

                # Execute both channels in parallel
                results = manager.cluster_with_progress(
                    clustering_tasks,
                    progress_callback=on_channel_progress,
                    channel_names=channel_names
                )

                self.logger.info(
                    f"Parallel clustering completed: "
                    f"Ch1={'OK' if results[1] is None else 'FAILED'}, "
                    f"Ch2={'OK' if results[2] is None else 'FAILED'}"
                )

        except Exception as e:
            self.logger.error(f"Parallel clustering failed: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Parallel Clustering Error",
                f"Parallel clustering failed: {str(e)}"
            )

    def cluster_both_channels_sequential(self) -> None:
        """
        Cluster both channels sequentially (for comparison/debugging).

        Uses sequential execution instead of parallel. Useful for:
        - Debugging clustering issues
        - Reducing memory usage for large datasets
        - Performance comparison with parallel mode

        Returns
        -------
        None
            Updates UI with clustering results for both channels
        """
        self.logger.info("Starting sequential clustering for both channels...")

        # Create task dictionary
        clustering_tasks = {
            1: lambda: self.cluster(channel=1),
            2: lambda: self.cluster(channel=2)
        }

        channel_names = {
            1: "Channel 1",
            2: "Channel 2"
        }

        try:
            with create_parallel_clustering_manager(
                max_workers=1,  # Sequential: 1 worker
                logger=self.logger
            ) as manager:
                results = manager.cluster_sequential(
                    clustering_tasks,
                    channel_names=channel_names
                )

                self.logger.info(
                    f"Sequential clustering completed: "
                    f"Ch1={'OK' if results[1] is None else 'FAILED'}, "
                    f"Ch2={'OK' if results[2] is None else 'FAILED'}"
                )

        except Exception as e:
            self.logger.error(f"Sequential clustering failed: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Sequential Clustering Error",
                f"Sequential clustering failed: {str(e)}"
            )


    def rx(self, obj: Any, points: Any) -> None:
        """Handle clicking on cluster centers to mark them as bad.

        SUPERSEDED -- no longer connected to any plot. Bad clusters are now
        identified automatically and objectively by
        ``tools.cluster_quality.identify_bad_clusters`` (edge-touching plus
        DBCV validity), which runs as part of ``run_mps_analysis``. Kept so
        the manual workflow can be restored by reconnecting
        ``sigClicked`` if a dataset ever needs hand curation.

        Parameters
        ----------
        obj : Any
            The scatter plot item (unused).
        points : Any
            List of clicked points containing position information.
        """
        try:
            clicked_pos = np.array([points[0].pos().x(), points[0].pos().y()])
            distances = np.linalg.norm(self.cluster_centroids - clicked_pos, axis=1)
            bad_cluster_idx = np.argmin(distances)  # Índice del CM más cercano
            
            if not hasattr(self, 'bad_cluster_indices'):
                self.bad_cluster_indices = []
            
            # Toggle cluster status (add if not present, remove if present)
            if bad_cluster_idx in self.bad_cluster_indices:
                self.bad_cluster_indices.remove(bad_cluster_idx)
            else:
                self.bad_cluster_indices.append(bad_cluster_idx)
            
            # Update display
            self.update_display_after_cluster_removal()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error en rx: {str(e)}")
        
        
    def update_display_after_cluster_removal(self) -> None:
        """Update the display after cluster removal."""
        try:
            if not hasattr(self, 'bad_cluster_indices'):
                return
            
            # Fiter good clusters
            mask = ~np.isin(self.cluster_labels, self.bad_cluster_indices)
            
            # Update good clusters CM 
            good_cluster_indices = [i for i in range(len(self.cluster_centroids)) 
                              if i not in self.bad_cluster_indices]
            self.good_cluster_centroids = self.cluster_centroids[good_cluster_indices] if good_cluster_indices else np.array([])
            
            # Update display
            good_clusters_widget = pg.GraphicsLayoutWidget()
            good_clusters_plot = good_clusters_widget.addPlot(title="Selected Clusters")
            good_clusters_plot.setAspectLocked(True)
            
            # Scatter plot good CMs
            # good_points = self.original_points[mask]
            # good_plot = pg.ScatterPlotItem(
            #     good_points[:, 0], good_points[:, 1], 
            #     pen=None, brush=self.brush3, size=5
            # )
            # good_clusters_plot.addItem(good_plot)
            

            if len(self.good_cluster_centroids) > 0:
                cm_plot = pg.ScatterPlotItem(
                    self.good_cluster_centroids[:, 0], self.good_cluster_centroids[:, 1], 
                    size=CLUSTER_CENTROID_POINT_SIZE, pen=pg.mkPen('k'), brush=self.brush3
                )
                good_clusters_plot.addItem(cm_plot)
            
            self.empty_layout(self.ui.scatterlayout_goodclus)
            self.ui.scatterlayout_goodclus.addWidget(good_clusters_widget)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error al filtrar: {str(e)}")
        

        
    
    def dist_cm_good_clus(self) -> None:
        """
        Display the centroids of good clusters (after removing bad clusters).

        SUPERSEDED -- the button that used to call this now opens the MPS
        analysis panel, which draws the curated centroids together with the
        reconstructed perimeter. Kept for the manual fallback described in
        ``rx``.

        Renders a scatter plot of cluster centroid coordinates (X, Y) for all clusters
        that were not marked as bad by the user. If no clusters have been explicitly
        marked as good yet, displays all cluster centroids.

        This is typically called after the user has clicked on cluster centers to mark
        them as "bad" (artifacts, false positives). The remaining clusters are considered
        "good" and are visualized here.

        Notes
        -----
        - Uses pyqtgraph for interactive visualization
        - Cluster centers are displayed with size=10 pixels in blue color (brush3)
        - X/Y range is set to match the ROI boundaries
        """
        if len(self.good_cluster_centroids) == 0:
            self.good_cluster_centroids = self.cluster_centroids
        self._render_good_clusters_panel(self.good_cluster_centroids)


    def save_clus_CM(self) -> None:
        """
        Save good cluster centroids to a CSV file.

        Exports the (X, Y) coordinates of all cluster centers that were not marked
        as bad. Useful for downstream analysis, comparing with other clustering methods,
        or creating publication-quality visualizations.

        Output format is a simple two-column CSV file with X and Y coordinates in
        nanometers.

        Raises
        ------
        UserWarning
            If no good clusters are available (user must run clustering first).

        Notes
        -----
        Output filename: `{original_filename}_cluster_centers.csv`
        Columns: X [nm], Y [nm]
        """
        # Guard: good cluster centroids must exist.
        if self.good_cluster_centroids is None or len(self.good_cluster_centroids) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No cluster centers",
                "Please run clustering and confirm cluster selection first."
            )
            return
        cluster_centers_xy = np.array([self.good_cluster_centroids[:,0],self.good_cluster_centroids[:,1]])
        cluster_centers_xy = np.transpose(cluster_centers_xy)
        
        # Get root filename
        root_name = self.get_root_filename()
        default_filename = f"{root_name}_cluster_centers.csv"
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cluster Centers",
            default_filename,
            "CSV Files (*.csv)"
        )
        
        if filename:
            np.savetxt(filename, cluster_centers_xy, delimiter=",", fmt="%.2f", comments="")
    
    
    
        
    def _clustered_selection(
        self, channel: int
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64],
                        NDArray[np.float64], NDArray[np.int64]]]:
        """
        A channel's ROI selection with its cluster labels.

        None, after telling the user why, when the channel has not been
        clustered or its selection changed since: the labels index the
        selection they were computed on, row by row.
        """
        if channel == 1:
            x, y, z = self.xroi, self.yroi, self.zroi
            labels = self.cluster_labels
        elif channel == 2:
            x, y, z = self.xroi2, self.yroi2, self.zroi2
            labels = self.cluster_labels2
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid channel selected")
            return None
        if x is None or y is None or z is None or labels is None:
            QtWidgets.QMessageBox.warning(
                self, "Error",
                f"No clustering data available for channel {channel}")
            return None
        if self._clustered_x[channel] is not x:
            QtWidgets.QMessageBox.warning(
                self, "Selection changed",
                f"The ROI selection of channel {channel} changed after it "
                f"was clustered. Cluster channel {channel} again before "
                f"saving.")
            return None
        return x, y, z, labels

    def _clusters_to_export(
        self, channel: int
    ) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64],
                        NDArray[np.float64], NDArray[np.int64], List[int],
                        str]]:
        """
        What a clustered-data export writes: the localizations, their
        cluster labels, the labels to leave out, and a line saying where
        the clusters come from. None, after telling the user why, when
        nothing current can be written.

        Channel 1's rejected clusters come from the MPS analysis, which
        clusters its own axial slab with its own parameters. They are
        labels of that clustering. In the window's clustering -- made with
        "auto" eps, another algorithm, or before the analysis was re-run
        with other parameters -- the same numbers name other clusters. So
        while the analysis describes the current selection, channel 1
        exports the analysis's localizations and labels. Otherwise, and on
        channel 2, the window's clustering is written whole.
        """
        analysis = self.mps_analysis
        current = self._analysed_x is not None and (
            self._analysed_x is self.xroi_unfiltered
            or self._analysed_x is self.xroi)
        if channel == 1 and analysis is not None and current:
            labels = np.asarray(analysis.labels)
            if labels.size == 0:
                QtWidgets.QMessageBox.warning(
                    self, "Error",
                    "The MPS analysis found too few localizations in its "
                    "axial slab to cluster them: there is nothing to save.")
                return None
            rejected = sorted(int(label)
                              for label in analysis.bad_report.bad_labels)
            source = (
                f"Clusters of the MPS analysis: DBSCAN eps "
                f"{analysis.eps_nm:g} nm, min samples "
                f"{analysis.min_samples}, on its axial slab "
                f"{analysis.slab_zmin_nm:.1f} .. {analysis.slab_zmax_nm:.1f} "
                f"nm. {len(rejected)} rejected cluster(s) left out.")
            return (np.asarray(analysis.x_slab), np.asarray(analysis.y_slab),
                    np.asarray(analysis.z_slab), labels, rejected, source)
        selection = self._clustered_selection(channel)
        if selection is None:
            return None
        x, y, z, labels = selection
        source = ("Clusters of the window's clustering"
                  + ("; there is no MPS analysis of this selection, so no "
                     "cluster was left out." if channel == 1 else "."))
        return x, y, z, labels, [], source

    def save_all_clustered_data(self, channel: int) -> None:
        """
        Save all ROI localizations with cluster assignments to CSV (default format).

        Exports the complete set of localizations within the selected ROI along with
        their DBSCAN cluster assignments. On channel 1 these are the MPS
        analysis's clusters, without the ones it rejected (see
        _clusters_to_export). Noise points (cluster = -1) are preserved.

        Output columns: x [nm], y [nm], z [nm], cluster_id

        This export format is suitable for custom post-processing pipelines or when
        standard software packages are not available.

        Parameters
        ----------
        channel : int
            Channel to save (1 or 2).

        Raises
        ------
        UserWarning
            If clustering has not been performed yet.
        ValueError
            If channel is not 1 or 2.

        Notes
        -----
        Output filename: `{original_filename}_ch{channel}_all_clusters.csv`
        Format: Standard CSV with headers
        """
        try:
            exported = self._clusters_to_export(channel)
            if exported is None:
                return
            x_data, y_data, z_data, labels, rejected, source = exported

            # Get root filename
            root_name = self.get_root_filename()

            # Exclude the rejected clusters (but keep noise points)
            mask = ~np.isin(labels, rejected)
            x_data = x_data[mask]
            y_data = y_data[mask]
            z_data = z_data[mask]
            labels = labels[mask]

            # Prepare data for saving
            data = {
                'x [nm]': x_data,
                'y [nm]': y_data,
                'z [nm]': z_data,
                'cluster_id': labels
            }
            
            # Set default filename
            default_filename = f"{root_name}_ch{channel}_all_clusters.csv"
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save All Cluster Data (including noise)",
                default_filename,
                "CSV Files (*.csv)"
            )
    
            if filename:
                pd.DataFrame(data).to_csv(filename, index=False, float_format='%.2f')
                QtWidgets.QMessageBox.information(
                    self,
                    "Success",
                    f"All clustered data (including noise) saved to "
                    f"{filename}\n\n{source}"
                )
    
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to save data: {str(e)}"
            )
    
    def save_all_clustered_data_thunderstorm(self, channel: int) -> None:
        """Save filtered clustered data (excluding noise and bad clusters) in ThunderSTORM format.

        Parameters
        ----------
        channel : int
            Channel to save (1 or 2).
        """
        try:
            exported = self._clusters_to_export(channel)
            if exported is None:
                return
            x_data, y_data, z_data, labels, rejected, source = exported

            # Get root filename
            root_name = self.get_root_filename()
            suffix = f"_ch{channel}_filtered_clusters_thunderstorm"

            # Exclude noise (-1) and the rejected clusters
            mask = (labels != -1) & ~np.isin(labels, rejected)

            # Prepare ThunderSTORM compatible data (without cluster IDs)
            data = {
                'x [nm]': x_data[mask],
                'y [nm]': y_data[mask],
                'z [nm]': z_data[mask]
            }
            
            # Set default filename
            default_filename = f"{root_name}{suffix}.csv"
            
            # Open file dialog with suggested filename
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Filtered Cluster Data (ThunderSTORM)",
                default_filename,
                "CSV Files (*.csv)"
            )
    
            if filename:
                pd.DataFrame(data).to_csv(filename, index=False, float_format='%.2f')
                QtWidgets.QMessageBox.information(
                    self,
                    "Success",
                    f"Filtered clustered data saved in ThunderSTORM format "
                    f"to {filename}\n\n{source}"
                )
    
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to save data: {str(e)}"
            )
    
    def latchange(self) -> None:
        """
        Update histogram parameters (bins, lmin, lmax) from UI inputs with validation.
        Hallazgo: Input validation for histogram configuration parameters.
        """
        try:
            self.bins = int(self.nbins.text())
            self.lmin = float(self.latmin.text())
            self.lmax = float(self.latmax.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Input",
                "Histogram parameters must be numeric values.\n"
                "Bins must be an integer, and Min/Max must be valid numbers."
            )
            return

        self.KNdist_hist()
        
    def KNdist_hist(self) -> None:
        """Compute K-nearest-neighbour distances between cluster centroids and
        display the distribution as a bar histogram.

        The centroids in ``self.good_cluster_centroids`` (good clusters, after optional removal
        of bad ones) are queried against themselves with a KDTree.  The
        self-distance (distance = 0, always the first column returned by
        ``tree.query``) is excluded.
        """
        # Guard: clustering must have been run first.
        if self.good_cluster_centroids is None or len(self.good_cluster_centroids) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No clusters",
                "Please run clustering and confirm cluster selection before "
                "computing distances."
            )
            return

        # Hallazgo: Input validation for K-nearest neighbors parameter
        try:
            self.Nneighbor = float(self.ui.lineEdit_Nneighbor.text())
            Nneighbor = int(self.Nneighbor)
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Input",
                "Number of neighbors must be a numeric value."
            )
            return
        
        tree = KDTree(self.good_cluster_centroids)
        distances, indexes = tree.query(self.good_cluster_centroids, Nneighbor+1)
        self.distances = distances[:,1:] # exclude distance to the same molecule; distances has N rows (#clusters) and M columns (# neighbors)

        self.logger.debug(f"Computed distances for {len(self.distances)} cluster centers")
        indexes = indexes[:,1:]    
                
        histzWidget3 = pg.GraphicsLayoutWidget()
        histabcm = histzWidget3.addPlot(title="distances Histogram")

        # The display range must not destroy data. The previous version
        # reassigned self.distances to the filtered subset, so (a) the CSV
        # written by savedistdata() silently lost every distance outside
        # (lmin, lmax) -- with the default 0-800 nm window that quietly
        # dropped real neighbours, e.g. an 836 nm 1NN in one test axon --
        # and (b) calling this method twice filtered the already-filtered
        # array, shrinking the data further on each redraw. Keep the full
        # array and derive a separate view for plotting.
        distances_full = self.distances
        plot_distances = distances_full
        if self.lmin is not None and self.lmax is not None:
            in_range = ((distances_full > self.lmin)
                        & (distances_full < self.lmax))
            plot_distances = distances_full[in_range]
            n_excluded = int(distances_full.size - plot_distances.size)
            if n_excluded:
                self.logger.info(
                    f"KNdist_hist: {n_excluded} distance(s) outside "
                    f"({self.lmin}, {self.lmax}) nm are hidden from the "
                    f"histogram but kept in the exported data."
                )

        bins = self.bins if self.bins is not None else 20

        if plot_distances.size == 0:
            QtWidgets.QMessageBox.warning(
                self, "No distances in range",
                f"All {distances_full.size} neighbour distances fall outside "
                f"the display range ({self.lmin}, {self.lmax}) nm.\n\n"
                f"Widen the range to see the histogram. The underlying data "
                f"is unchanged."
            )
            return

        histcmdist, bin_edgescmdist = np.histogram(plot_distances, bins)
        widthcmdist = np.mean(np.diff(bin_edgescmdist))
        bincenterscmdist = np.mean(np.vstack([bin_edgescmdist[0:-1],bin_edgescmdist[1:]]), axis=0)
        bargraphcmdist = pg.BarGraphItem(x = bincenterscmdist, height = histcmdist, 
                                    width = widthcmdist, brush = self.brush3, pen = None)
        histabcm.addItem(bargraphcmdist)
        histabcm.setXRange(self.lmin, self.lmax)
                
        self.empty_layout(self.ui.zhistlayout_cmdist)
        self.ui.zhistlayout_cmdist.addWidget(histzWidget3)
    
        
        
    def empty_layout(self, layout: Any) -> None:
        """
        Remove all widgets from a PyQt5 layout.

        Clears all child widgets from a given layout container, preparing it to
        receive new content. This is used before adding new visualization plots
        to ensure old graphics are properly garbage collected.

        Parameters
        ----------
        layout : PyQt5.QtWidgets.QLayout
            The layout widget to empty (e.g., self.ui.scatterlayout_clusterch1).

        Notes
        -----
        Iterates in reverse order to safely remove items without indexing issues.
        """
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
            
            
    def onCloseEvent(self, event: Any) -> None:
        """
        Handle the application close event.

        Called when the user attempts to close the main window. Terminates the
        Qt application cleanly, ensuring proper resource cleanup.

        Parameters
        ----------
        event : PyQt5.QtGui.QCloseEvent
            The close event triggered by the window manager or user action.
        """
        # Persist the analysis parameters so the next session starts where
        # this one left off, rather than reverting to the .ui defaults.
        try:
            self._persist_mps_settings()
        except Exception as exc:                          # noqa: BLE001
            self.logger.warning(f"Could not persist MPS settings: {exc}")
        if self.mps_window is not None:
            self.mps_window.close()
        if self.rings_window is not None:
            self.rings_window.close()
        if self.quality_window is not None:
            self.quality_window.close()
        if self.paint_window is not None:
            self.paint_window.close()
        if self.two_channel_window is not None:
            self.two_channel_window.close()
        self.picasso_tools.shutdown()

        self.logger.info("=" * 80)
        self.logger.info("MPS Explorer Application Closed")
        self.logger.info("=" * 80)

        # Stop the entire process and close the application
        QApplication.quit()
    
if __name__ == '__main__':
    
    app = QtWidgets.QApplication([])
    # app = QtGui.QApplication([])
    win = MPS_explorer()
    win.show()
    app.exec_()
    

    

