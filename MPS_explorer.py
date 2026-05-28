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
import hdbscan

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

        self.ui = data_explorer.Ui_MainWindow()
        self.ui.setupUi(self)
        self.logger.debug("UI setup complete")

        # Define initial directory
        self.initialDir = "Desktop"  # You can set the initial directory here
        self.logger.debug(f"Initial directory: {self.initialDir}")
        
        # File Formats
        fileformat_list = ["Picasso hdf5", "ThunderStorm csv", "custom csv"]
        self.fileformat = self.ui.comboBox_fileformat
        self.fileformat.addItems(fileformat_list)
        self.fileformat_2 = self.ui.comboBox_fileformat_2
        self.fileformat_2.addItems(fileformat_list)  # Reuse the same list for second channel
        
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
        self.ui.pushButton_remove_bad_cluster.clicked.connect(self.dist_cm_good_clus)
        self.ui.pushButton_savecluscenters.clicked.connect(self.save_clus_CM)
        self.ui.pushButton_Distances.clicked.connect(self.KNdist_hist)
        self.ui.pushButton_saveAllClusterData.clicked.connect(lambda: self.save_all_clustered_data(1))
        self.ui.pushButton_saveAllClusterDataThunderStorm.clicked.connect(lambda: self.save_all_clustered_data_thunderstorm(1))
        
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
        self.radioButton_circROI.clicked.connect(self.scatterplot)
        self.radioButton_squareROI.clicked.connect(self.scatterplot)

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

        # --- ROI selection (set by update_ROI) ---
        self.xroi: Optional[NDArray[np.float64]] = None            # Ch1 localizations inside the ROI
        self.yroi: Optional[NDArray[np.float64]] = None
        self.zroi: Optional[NDArray[np.float64]] = None
        self.zmin: Optional[float] = None            # z slab lower bound (nm)
        self.zmax: Optional[float] = None            # z slab upper bound (nm)

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

        self.eps: Optional[float] = None             # DBSCAN epsilon parameter
        self.minsamples: Optional[int] = None      # DBSCAN min_samples parameter

        # --- nearest-neighbour distances (set by KNdist_hist) ---
        self.distances: Optional[NDArray[np.float64]] = None       # distance array to the K nearest centroids
        self.Nneighbor: Optional[int] = None       # number of neighbours requested

        # Connect the close event to your method
        self.closeEvent = self.onCloseEvent

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
                    self.ui.lineEdit_filename.setText(root.filenamedata)
                    self.fileformat1 = int(self.fileformat.currentIndex())
                    self.logger.debug(f"File format: {['Picasso HDF5', 'ThunderStorm CSV', 'Custom CSV'][self.fileformat1]}")
                    self.xdata, self.ydata, self.zdata = self.import_file(root.filenamedata, self.fileformat1)
                    self.logger.info(f"Channel 1 loaded: {len(self.xdata):,} localizations")
                else:
                    self.logger.debug("File dialog cancelled for channel 1")
                    return
            elif channel == 2:
                root.filenamedata2 = filedialog.askopenfilename(initialdir=self.initialDir,
                                                                title='Select file')
                if root.filenamedata2 != '':
                    self.logger.info(f"Channel 2 file selected: {root.filenamedata2}")
                    self.ui.lineEdit_filename_2.setText(root.filenamedata2)
                    self.fileformat2 = int(self.fileformat_2.currentIndex())
                    self.logger.debug(f"File format: {['Picasso HDF5', 'ThunderStorm CSV', 'Custom CSV'][self.fileformat2]}")
                    self.xdata2, self.ydata2, self.zdata2 = self.import_file(root.filenamedata2, self.fileformat2)
                    self.logger.info(f"Channel 2 loaded: {len(self.xdata2):,} localizations")
                else:
                    self.logger.debug("File dialog cancelled for channel 2")
                    return
        except OSError as e:
            self.logger.error(f"Error loading file: {e}", exc_info=True)
            pass

   
    
    def _get_pixel_size_from_yaml(self, hdf5_filename: str) -> Optional[float]:
        """
        Reads the pixel size from the YAML file paired with a Picasso HDF5.
        Picasso always writes a .yaml file alongside the .hdf5 file containing
        acquisition metadata, including the line 'Pixelsize: <value_in_nm>'.

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
        yaml_filename = os.path.splitext(hdf5_filename)[0] + ".yaml"
        if not os.path.exists(yaml_filename):
            return None
        try:
            with open(yaml_filename, "r", encoding="utf-8") as f:
                # The YAML file may contain multiple documents separated by '---'.
                # We do a simple line-based parse to avoid adding a pyyaml dependency.
                for line in f:
                    line = line.strip()
                    if line.startswith("Pixelsize:"):
                        # Format example: "Pixelsize: 113"
                        value_str = line.split(":", 1)[1].strip()
                        return float(value_str)
        except (OSError, ValueError):
            return None
        return None

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

    def import_file(self, filename: str, fileformat: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Import localization data from HDF5 or CSV files.

        Parameters
        ----------
        filename : str
            Path to the data file.
        fileformat : int
            File format code: 0=Picasso HDF5, 1=ThunderSTORM CSV, 2=Custom CSV.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
            Tuple of (x, y, z) coordinate arrays in nanometers.
        """
        if fileformat == 0: # Importation procedure for Picasso hdf5 files.
            f = h5.File(filename, "r")
            dataset = f['locs']
            xdata = dataset['x']
            ydata = dataset['y']
            zdata = dataset['z']
            # Read pixel size from the YAML companion file instead of hardcoding.
            # This fixes the original behavior where coordinates were always
            # scaled by 133 nm regardless of the actual optical configuration.
            pxsize = self._get_pixel_size_from_yaml(filename)
            if pxsize is None:
                pxsize = self._ask_user_for_pixel_size()
            self.pxsize = pxsize
            self.logger.info(f"Using pixel size = {self.pxsize} nm for {os.path.basename(filename)}")
            xdata = xdata * self.pxsize
            ydata = ydata * self.pxsize
        elif fileformat == 1: # Importation procedure for ThunderSTORM csv files.
            dataset = pd.read_csv(filename)
            headers = dataset.columns.values
            xdata = dataset[headers[np.where(headers=='x [nm]')]].values.flatten()
            ydata = dataset[headers[np.where(headers=='y [nm]')]].values.flatten()
            zdata = dataset[headers[np.where(headers=='z [nm]')]].values.flatten()
        else: # Importation procedure for custom csv files.
            dataset = pd.read_csv(filename)
            data = pd.DataFrame(dataset)
            dataxyz = data.values
            dataxyz = dataxyz.astype(float)
            xdata = dataxyz[:,0]
            ydata = dataxyz[:,1]
            zdata = dataxyz[:,2]
        return xdata, ydata, zdata


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

    def _render_channel_2(self, scatterWidgetxy: Any, plotxy: Any) -> None:
        """Add channel 2 scatter and histogram overlay to the overview.

        If no channel 2 file is loaded, this method does nothing.

        Parameters
        ----------
        scatterWidgetxy : pyqtgraph.GraphicsLayoutWidget
            GraphicsLayoutWidget with the scatter plot
        plotxy : pyqtgraph.PlotItem
            PlotItem to add channel 2 scatter to
        """
        filename2 = self.ui.lineEdit_filename_2.text()
        if filename2 == '':
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
        self.empty_layout(self.ui.scatterlayout)
        self.ui.scatterlayout.addWidget(scatterWidgetxy)

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
        self._render_channel_2(scatterWidgetxy, plotxy)
     
              
    
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

        scatterWidgetROI = pg.GraphicsLayoutWidget()
        plotROI = scatterWidgetROI.addPlot(title="Scatter plot ROI selected")
        plotROI.setAspectLocked(True)
        
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
                     
            
            # Define zmin and zmax
            zmin = self.ui.lineEdit_zmin.text()
            zmax = self.ui.lineEdit_zmax.text()
            
            # Hallazgo H04: Convert zmin and zmax to float (not int).
            # Z-coordinates are stored as float (from dataxyz.astype(float)),
            # so boundaries must also be float to preserve precision and avoid
            # truncation errors when filtering z-slices.
            self.zmin = float(zmin) if zmin else None
            self.zmax = float(zmax) if zmax else None
            
            if self.zmax is None:
                self.zroi = self.z[ind_inside_roi]
            else:
                zroi = self.z[ind_inside_roi]
                indz = np.where((zroi > self.zmin) & (zroi < self.zmax))
                self.zroi = zroi[indz]
                self.xroi = self.xroi[indz]
                self.yroi = self.yroi[indz]
            
            
  
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

            # Vectorized boolean mask: True where point is inside the square
            mask = (self.x > xmin) & (self.x < xmax) & (self.y > ymin) & (self.y < ymax)
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

            zmin = self.ui.lineEdit_zmin.text()
            zmax = self.ui.lineEdit_zmax.text()

            self.zmin = float(zmin) if zmin else None
            self.zmax = float(zmax) if zmax else None

            if self.zmax is None:
                self.zroi = self.z[ind_inside_roi]
            else:
                zroi = self.z[ind_inside_roi]
                indz = np.where((zroi > self.zmin) & (zroi < self.zmax))
                self.zroi = zroi[indz]
                self.xroi = self.xroi[indz]
                self.yroi = self.yroi[indz]
        
        else:
            
            self.xroi = self.x
            self.yroi = self.y
            
            zmin = self.ui.lineEdit_zmin.text()
            zmax = self.ui.lineEdit_zmax.text()
        
            self.zmin = float(zmin) if zmin else None
            self.zmax = float(zmax) if zmax else None
        
            if self.zmax is None:
                self.zroi = self.z
            else:
                zroi = self.z
                indz = np.where((zroi > self.zmin) & (zroi < self.zmax))
                self.zroi = zroi[indz]
                self.xroi = self.xroi[indz]
                self.yroi = self.yroi[indz]
            
            
        # Final guard: after all ROI and z-filtering, check if anything remains.
        # This is a belt-and-suspenders check; the earlier guards should prevent
        # this, but aggressive z-filtering can theoretically remove all points.
        if len(self.xroi) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No data in ROI after filtering",
                "All localizations were filtered out by the z-range. "
                "Adjust zmin/zmax and try again."
            )
            return

        self.selected = pg.ScatterPlotItem(self.xroi, self.yroi, pen = self.pen1,
                                           brush = None, size = GOOD_CLUSTER_POINT_SIZE)
        plotROI.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotROI.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        plotROI.addItem(self.selected)
        
        
        self.empty_layout(self.ui.scatterlayout_3)
        self.ui.scatterlayout_3.addWidget(scatterWidgetROI)    
        
        
        histzWidget2 = pg.GraphicsLayoutWidget()
        histabsz2 = histzWidget2.addPlot(title="z ROI Histogram")
        
        histz2, bin_edgesz2 = np.histogram(self.zroi, bins='auto')
        widthzabs2 = np.mean(np.diff(bin_edgesz2))
        bincentersz2 = np.mean(np.vstack([bin_edgesz2[0:-1],bin_edgesz2[1:]]), axis=0)
        bargraphz2 = pg.BarGraphItem(x = bincentersz2, height = histz2, 
                                    width = widthzabs2, brush = self.brush1, pen = self.pen1)
        bargraphz2.setOpacity(0.5) 
        histabsz2.addItem(bargraphz2)
        
        filename2 = self.ui.lineEdit_filename_2.text()
        
        if filename2 == '':
            
            
            pass
        
        else:
            
            if self.ui.radioButton_circROI.isChecked():

                # Get circular ROI position and size
                pos = self.circular_roi.pos()
                size = self.circular_roi.size()

                # Extract scalar from size (PyQtGraph returns Point or QSizeF)
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
                
                # Vectorized point-in-circle test for channel 2 (same logic
                # as channel 1, see comments above). Replaces the original
                # Python loop that did not scale with dataset size.
                import time
                t_start = time.perf_counter()
                center_flat = np.array([center_x, center_y]).flatten()
                distances2 = np.linalg.norm(self.data_points2 - center_flat, axis=1)
                mask2 = distances2 <= float(radius)
                ind_inside_roi2 = np.where(mask2)[0]
                points_inside_roi2 = self.data_points2[mask2]
                t_end = time.perf_counter()
                elapsed_ms = (t_end - t_start) * 1000
                n_points = len(self.data_points2)
                n_selected = len(points_inside_roi2)
                self.logger.debug(f"ROI Filter Ch2: Vectorized filter over {n_points:,} points "
                                 f"-> {n_selected:,} selected in {elapsed_ms:.2f} ms")
                
                if len(points_inside_roi2) == 0:
                    QtWidgets.QMessageBox.warning(
                        self, "Empty ROI in channel 2",
                        "The selected ROI contains no localizations in channel 2."
                    )
                    return
                
                self.xroi2 = points_inside_roi2[:,0]
                self.yroi2 = points_inside_roi2[:,1]
                         
                
                # Define zmin and zmax
                zmin = self.ui.lineEdit_zmin.text()
                zmax = self.ui.lineEdit_zmax.text()

                # Hallazgo H04: Convert zmin and zmax to float (not int).
                # Z-coordinates are stored as float, so boundaries must also be float
                # to preserve precision and avoid truncation errors when filtering z-slices.
                self.zmin = float(zmin) if zmin else None
                self.zmax = float(zmax) if zmax else None
                
                if self.zmax is None:
                    self.zroi2 = self.z2[ind_inside_roi2]
                else:
                    zroi = self.z2[ind_inside_roi2]
                    indz = np.where((zroi > self.zmin) & (zroi < self.zmax))
                    self.zroi2 = zroi[indz]
                    self.xroi2 = self.xroi2[indz]
                    self.yroi2 = self.yroi2[indz]
                
                
      
            elif self.ui.radioButton_squareROI.isChecked():
                # Hallazgo 07 (ch2 square ROI — fixed)
                # Get square ROI position and size
                xmin, ymin = self.square_roi.pos()
                xmax, ymax = self.square_roi.pos() + self.square_roi.size()

                # Vectorized boolean mask for channel 2
                mask2 = (self.x2 > xmin) & (self.x2 < xmax) & (self.y2 > ymin) & (self.y2 < ymax)
                points_inside_roi2 = self.data_points2[mask2]

                # Guard: bail out if ROI contains no localizations in ch2
                if len(points_inside_roi2) == 0:
                    QtWidgets.QMessageBox.warning(
                        self, "Empty ROI in channel 2",
                        "The selected ROI contains no localizations in channel 2."
                    )
                    return

                self.xroi2 = points_inside_roi2[:, 0]
                self.yroi2 = points_inside_roi2[:, 1]

                # Get original indices of points inside ROI for z-filtering
                ind_inside_roi2 = np.where(mask2)[0]

                zmin = self.ui.lineEdit_zmin.text()
                zmax = self.ui.lineEdit_zmax.text()

                # Hallazgo H04: Convert zmin and zmax to float (not int) for Ch2 square ROI.
                # Z-coordinates are stored as float, so boundaries must also be float.
                self.zmin = float(zmin) if zmin else None
                self.zmax = float(zmax) if zmax else None

                if self.zmax is None:
                    self.zroi2 = self.z2[ind_inside_roi2]
                else:
                    # Use self.z2, not self.z (bug fix)
                    zroi = self.z2[ind_inside_roi2]
                    indz = np.where((zroi > self.zmin) & (zroi < self.zmax))
                    self.zroi2 = zroi[indz]
                    self.xroi2 = self.xroi2[indz]
                    self.yroi2 = self.yroi2[indz]
              
                    
            self.selected2 = pg.ScatterPlotItem(self.xroi2, self.yroi2, pen = self.pen2,
                                               brush = None, size = GOOD_CLUSTER_POINT_SIZE)
            # NOTE: Original code had size=3 here (smaller than Ch1's size=5).
            # Changed to GOOD_CLUSTER_POINT_SIZE for consistency. If Ch2 should
            # be visually smaller, this is a design choice that should be documented.  
            plotROI.setLabels(bottom=('x [nm]'), left=('y [nm]'))
            plotROI.setXRange(np.min(self.xroi2), np.max(self.xroi2), padding=0)
            plotROI.addItem(self.selected2)
            
            
            self.empty_layout(self.ui.scatterlayout_3)
            self.ui.scatterlayout_3.addWidget(scatterWidgetROI)    

            
            histz2, bin_edgesz2 = np.histogram(self.zroi2, bins='auto')
            widthzabs2 = np.mean(np.diff(bin_edgesz2))
            bincentersz2 = np.mean(np.vstack([bin_edgesz2[0:-1],bin_edgesz2[1:]]), axis=0)
            bargraphz22 = pg.BarGraphItem(x = bincentersz2, height = histz2, 
                                        width = widthzabs2, brush = self.brush2, pen = self.pen2)
            bargraphz22.setOpacity(0.5) 
            histabsz2.addItem(bargraphz22)
        
                
        self.empty_layout(self.ui.zhistlayout_2)
        self.ui.zhistlayout_2.addWidget(histzWidget2)
          

     

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
        elif channel == 2:
            x_roi = self.xroi2
            y_roi = self.yroi2
            z_roi = self.zroi2
            labels = self.dblabels2 if hasattr(self, 'dblabels2') else None
            suffix = f"_ch{channel}_roi"
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
                "Please select an ROI before clustering channel 2."
            )
            return

        # Reset bad-cluster list for each new clustering run so stale
        # selections from a previous run don't carry over.
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

            # Hallazgo: Input validation for DBSCAN parameters (Ch1)
            try:
                self.minsamples = float(self.ui.lineEdit_minsamples.text())
                self.eps = float(self.ui.lineEdit_eps.text())
                self.logger.debug(f"Clustering Ch1 parameters: eps={self.eps}, min_samples={self.minsamples}")
            except ValueError as e:
                self.logger.error(f"Clustering Ch1: Invalid DBSCAN parameters: {e}")
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Input",
                    "DBSCAN parameters must be numeric values.\n"
                    "Min Samples and Epsilon must be valid numbers."
                )
                return

        elif channel == 2:
            # Channel 2 data and parameters
            x_roi = self.xroi2
            y_roi = self.yroi2
            z_roi = self.zroi2
            roi_brush = self.brush2  # Channel 2 color
            roi_pen = self.pen2      # Channel 2 border color
            scatter_layout_cluster = self.ui.scatterlayout_clusterch2  # Target UI layout

            # Hallazgo: Input validation for DBSCAN parameters (Ch2)
            try:
                self.minsamples = float(self.ui.lineEdit_minsamples_2.text())
                self.eps = float(self.ui.lineEdit_eps_2.text())
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Input",
                    "DBSCAN parameters must be numeric values.\n"
                    "Min Samples and Epsilon must be valid numbers."
                )
                return
        else:
            return  # Invalid channel
    
        # Prepare XY coordinate array
        roi_points = np.column_stack((x_roi, y_roi))
        self.logger.debug(f"Clustering: Clustering {len(roi_points):,} points in ROI")

        # Perform DBSCAN clustering
        try:
            dbscan_result = DBSCAN(eps=self.eps, min_samples=int(self.minsamples)).fit(roi_points)
            cluster_assignments = dbscan_result.labels_  # Get cluster labels (-1 for noise)
        except Exception as e:
            self.logger.error(f"Clustering: DBSCAN failed: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self, "Clustering Error",
                f"DBSCAN clustering failed: {str(e)}"
            )
            return

        # Store clustering results
        self.cluster_labels = cluster_assignments  # Array assigning each point to a cluster (or -1 for noise)
        self.original_points = roi_points          # Store original coordinates for reference
        self.original_z = z_roi                    # Store original z-values

        # Calculate cluster centers (centroids)
        unique_labels = np.unique(cluster_assignments)
        centroids_list = []
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points
            cluster_points = roi_points[cluster_assignments == label]
            centroids_list.append(np.mean(cluster_points, axis=0))  # Calculate centroid

        # Store rounded cluster centers
        self.cluster_centroids = np.around(np.array(centroids_list), decimals=2)

        # Log clustering statistics
        n_clusters = len(self.cluster_centroids)
        n_noise = np.sum(cluster_assignments == -1)
        self.logger.info(f"Clustering Ch{channel}: Found {n_clusters} clusters, {n_noise:,} noise points")
        
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
            self.cluster_centroids[:, 0], self.cluster_centroids[:, 1], 
            size=CLUSTER_CENTROID_POINT_SIZE, pen=pg.mkPen('k'), brush=roi_brush  # Filled circles for centers
        )
        plotclusters.addItem(self.selectedcluscm)
        
        # Connect click event for center selection
        self.selectedcluscm.sigClicked.connect(self.rx)  # rx handles center clicks
        
        # Update UI with new plot
        self.empty_layout(scatter_layout_cluster)
        scatter_layout_cluster.addWidget(scatterWidgetcluster)
        
        
    def rx(self, obj: Any, points: Any) -> None:
        """Handle clicking on cluster centers to mark them as bad.

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
        good_clusters_widget = pg.GraphicsLayoutWidget()
        good_clusters_plot = good_clusters_widget.addPlot(title="Clusters centers and distances")
        good_clusters_plot.setAspectLocked(True)
        
        
        if len(self.good_cluster_centroids) == 0:
            self.good_cluster_centroids = self.cluster_centroids
        else:
            pass

        self.good_clusters_scatter_plot = pg.ScatterPlotItem(self.good_cluster_centroids[:,0], self.good_cluster_centroids[:,1], size=CLUSTER_CENTROID_POINT_SIZE, brush = self.brush3)  
        good_clusters_plot.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        good_clusters_plot.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        
        good_clusters_plot.addItem(self.good_clusters_scatter_plot)
        
        self.empty_layout(self.ui.scatterlayout_goodclus)
        self.ui.scatterlayout_goodclus.addWidget(good_clusters_widget)
        

        
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
    
    
    
        
    def save_all_clustered_data(self, channel: int) -> None:
        """
        Save all ROI localizations with cluster assignments to CSV (default format).

        Exports the complete set of localizations within the selected ROI along with
        their DBSCAN cluster assignments. Bad clusters (marked by the user) are excluded
        from the output. Noise points (cluster = -1) are preserved.

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
            if not hasattr(self, 'cluster_labels'):
                QtWidgets.QMessageBox.warning(self, "Error", "No clustering data available")
                return
    
            # Get root filename
            root_name = self.get_root_filename()
            
            # Get data for specified channel
            if channel == 1:
                x_data = self.xroi
                y_data = self.yroi
                z_data = self.zroi
                labels = self.cluster_labels
            elif channel == 2:
                x_data = self.xroi2
                y_data = self.yroi2
                z_data = self.zroi2
                labels = self.cluster_labels2
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Invalid channel selected")
                return
    
            # Create mask to exclude bad clusters (but keep noise points)
            if hasattr(self, 'bad_cluster_indices'):
                mask = ~np.isin(labels, self.bad_cluster_indices)
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
                    f"All clustered data (including noise) saved to {filename}"
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
            if not hasattr(self, 'cluster_labels'):
                QtWidgets.QMessageBox.warning(self, "Error", "No clustering data available")
                return
    
            # Get root filename
            root_name = self.get_root_filename()
            
            # Get data for specified channel
            if channel == 1:
                x_data = self.xroi
                y_data = self.yroi
                z_data = self.zroi
                labels = self.cluster_labels
                suffix = f"_ch{channel}_filtered_clusters_thunderstorm"
            elif channel == 2:
                x_data = self.xroi2
                y_data = self.yroi2
                z_data = self.zroi2
                labels = self.cluster_labels2
                suffix = f"_ch{channel}_filtered_clusters_thunderstorm"
            else:
                QtWidgets.QMessageBox.warning(self, "Error", "Invalid channel selected")
                return
    
            # Create mask to exclude noise (-1) and bad clusters
            noise_mask = (labels != -1)  # Exclude noise points
            if hasattr(self, 'bad_cluster_indices'):
                bad_cluster_mask = ~np.isin(labels, self.bad_cluster_indices)
                mask = noise_mask & bad_cluster_mask
            else:
                mask = noise_mask
    
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
                    f"Filtered clustered data saved in ThunderSTORM format to {filename}"
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
        
        if self.lmin != None:
            
            self.distances = self.distances[(self.distances>self.lmin) & (self.distances<self.lmax)]
            
        else:
            pass
        
        if self.lmax != None:
            
            self.distances = self.distances[(self.distances>self.lmin) & (self.distances<self.lmax)]
            
        else:
            pass
        
        if self.bins != None:
            
            bins = self.bins
        else:
            bins = 20
        
        
        histcmdist, bin_edgescmdist = np.histogram(self.distances, bins)
        widthcmdist = np.mean(np.diff(bin_edgescmdist))
        bincenterscmdist = np.mean(np.vstack([bin_edgescmdist[0:-1],bin_edgescmdist[1:]]), axis=0)
        bargraphcmdist = pg.BarGraphItem(x = bincenterscmdist, height = histcmdist, 
                                    width = widthcmdist, brush = self.brush3, pen = None)
        histabcm.addItem(bargraphcmdist)
        histabcm.setXRange(self.lmin, self.lmax)
                
        self.empty_layout(self.ui.zhistlayout_cmdist)
        self.ui.zhistlayout_cmdist.addWidget(histzWidget3)
    
        
        
    def dist_cmDBSCAN(self) -> None:
        """
        Display all DBSCAN cluster centroids with interactive removal capability.

        Renders an interactive scatter plot of all cluster centroid coordinates (X, Y)
        obtained from DBSCAN clustering. Users can click on any cluster center to mark it
        as "bad" (artifact, false positive, etc.) via the rx() callback handler.

        This visualization helps users visually inspect clustering results and manually
        remove spurious or unwanted clusters before downstream analysis.

        Notes
        -----
        - Click on any cluster center to mark it as bad
        - The rx() method handles cluster removal
        - All clusters (good and bad) are shown; use dist_cm_good_clus() to view filtered results
        - Cluster centers are displayed with size=10 pixels in blue color (brush3)
        """
        scatterWidgetDBSCAN_cmdist = pg.GraphicsLayoutWidget()
        plotdistcmd = scatterWidgetDBSCAN_cmdist.addPlot(title="Clusters centers and distances")
        plotdistcmd.setAspectLocked(True)

        self.selectedcluscmd = pg.ScatterPlotItem(self.cluster_centroids[:,0], self.cluster_centroids[:,1], size=CLUSTER_CENTROID_POINT_SIZE, brush = self.brush3)  
        plotdistcmd.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        plotdistcmd.setXRange(np.min(self.xroi), np.max(self.xroi), padding=0)
        self.selectedcluscmd.sigClicked.connect(self.rx)
        
        plotdistcmd.addItem(self.selectedcluscmd)
        
        self.empty_layout(self.ui.scatterlayout_histcmdist)
        self.ui.scatterlayout_histcmdist.addWidget(scatterWidgetDBSCAN_cmdist) 
        
                
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
    

    

