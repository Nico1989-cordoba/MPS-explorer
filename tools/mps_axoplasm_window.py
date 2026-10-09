# -*- coding: utf-8 -*-
"""
The axoplasm panel: is each betaII-spectrin localization at the membrane
or inside the axon, according to a widefield betaIII-tubulin image?

Top to bottom:
  1. the images. The tubulin one, and optionally a widefield image of the
     super-resolved protein together with the localizations of the whole
     movie, to measure the shift between the acquisitions. Section 5 also
     needs that spectrin image: it finds the ring's dark inside in it;
  2. the alignment, measured and adjustable by hand;
  3. the mask, with Otsu's threshold on the axon's region, adjustable;
  4. the margin the user sets, where the localizations are -- inside only
     through a cluster both images put inside -- and the export;
  5. the MPS analysis' clusters that are not anchored to the membrane --
     the ones both widefield images put inside the axon -- and the contour
     rebuilt without them. The main window is told, and repeats the whole
     MPS analysis without them beside the original.

Dual view (2026-09-22). Either image can instead be a localization file
of that protein acquired AT THE SAME TIME as the movie, on the other half
of one camera -- the 2023 sciatic-nerve stainings of betaII-spectrin with
betaIII-tubulin are like that. Its localizations are counted on the
movie's own camera grid, so it is placed by construction: no widefield
image, no shift, and section 5 runs without one; the table records
"same acquisition". For the spectrin, that can be the movie's own file.
Measured on a few picked axons of unpublished data, the tubulin mask
drawn from localizations and the one drawn from the widefield image of
the same axons overlap only partly, and the widefield one is larger --
its edge a few hundred nm further out in effective radius -- so the same
margin discards less against a mask drawn from localizations. Which to use is the user's call; the table says
which it was.

The calculations are in ``tools.mps_axoplasm``.

UI stage 2 (design 3.1): the image, its two edges, the localizations by
class and the clusters in four groups are drawn on the axon map of the MPS
analysis window (its Axoplasm view), the program's one map; "Show on the
axon map" raises it beside this panel, and every change made here redraws
it live (``AxoplasmInputs.map_changed``, ``map_state``). This panel keeps
its controls, its findings and the histogram of the distances to the
tubulin mask's edge.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from tools import mps_axoplasm as ax
from tools import mps_axon_map_layers as map_layers
from tools import mps_file_drop
from tools.mps_io import load_localizations
from tools.mps_plot_style import (
    AXIS_FG, PANEL_BG, TEXT_DIM, TITLE_FG, marked, neutral, role,
    set_title, style_dark)
from tools.cluster_quality import describe_roi

# Every colour is a role from tools.mps_plot_style. The image, its edges,
# the localizations by class and the clusters are drawn on the axon map
# (tools.mps_axon_map), in the roles validate_plot_colours.py checks there.
# A finding is text, not a mark on the image: the verdict roles, the
# same three as in every other panel, each with the mark for its kind.
_TEXT_OK = role("good")
_WARN = role("warn")
_DIM = TEXT_DIM
# The margin on the distance histogram: the discard's vermillion.
_INTERIOR = role("discarded")
# Where the image went, said where it used to be drawn.
ON_THE_MAP = ("The image, the mask's and the spectrin interior's edges, the "
              "localizations by class and the clusters in their four groups are "
              "drawn on the axon map: MPS analysis window, Axoplasm view. It "
              "follows every change made here.")
SHOW_ON_MAP = "Show on the axon map"
SHOW_ON_MAP_TIP = ("Open or raise the MPS analysis window with the axon map in "
                   "its Axoplasm view (the image, the edges, the classes and the "
                   "clusters), beside this panel the first time, and keep it "
                   "following the threshold, smoothing, margin and shift.")


# Contours rebuilt with every 2-opt start, kept per set of cluster centres.
CONTOUR_CACHE_SIZE = 32

# How long the panel waits, after the last step of a drag, before it
# recomputes the mask and the discard. Measured on a pilot axon (tens of
# clusters, ~10^4 localizations): one control event costs ~0.1 s of panel
# work even when nothing is discarded, and seconds when the discarded set
# changes, most of it the 1000-iteration randomization. A slider
# emits about 60 of those a second, so without this the window is not
# slow -- it is frozen.
RECOMPUTE_DELAY_MS = 250

# Points drawn per class on the map; the classification itself uses all of
# them (tools.mps_axon_map_layers.MAX_DRAWN_PER_CLASS).
MAX_DRAWN = map_layers.MAX_DRAWN_PER_CLASS
# Histogram of the distances to the edge.
HIST_RANGE_NM = 1500.0
HIST_BIN_NM = 25.0


def _label(text: str, colour: str = TITLE_FG,
           bold: bool = False) -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    widget.setStyleSheet(
        f"color: {colour}; font-weight: {'bold' if bold else 'normal'};")
    widget.setWordWrap(True)
    widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    return widget


def _spin(low: float, high: float, step: float, decimals: int = 1,
          suffix: str = "") -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setSuffix(suffix)
    # Only a finished edit recomputes, not every keystroke.
    spin.setKeyboardTracking(False)
    return spin


@dataclass
class AxoplasmInputs:
    """What the panel needs from the main window."""

    loc: Any                    # the channel-1 selection (Localizations)
    movie: Any                  # the whole channel-1 file (Localizations)
    roi: Optional[Any] = None   # tools.cluster_quality ROI shape
    # Centroids (nm) of the clusters the MPS analysis kept for this
    # selection, or None when it has not been analysed.
    clusters: Callable[[], Optional[np.ndarray]] = field(
        default=lambda: None)
    # Identifies the main window's selection, to tell whether a panel
    # reopened later still shows it.
    selection_key: Any = None
    # The contour the MPS analysis built from those clusters
    # (tools.mps_geometry.PerimeterResult), or None.
    contour: Callable[[], Optional[Any]] = field(default=lambda: None)
    # Told the clusters not anchored to the membrane each time they are
    # found again -- the AnchoredClusters and the centres (nm) they were
    # found for -- or (None, None) when there are none to tell.
    anchored_changed: Callable[[Optional[ax.AnchoredClusters],
                                Optional[np.ndarray]], None] = field(
        default=lambda found, centroids: None)
    # For localizations of the selection (x, y, z in nm), the index of the
    # kept cluster each belongs to, in the order of ``clusters`` (-1: none),
    # or None when the selection has not been analysed.
    cluster_of: Callable[[np.ndarray, np.ndarray, np.ndarray],
                         Optional[np.ndarray]] = field(
        default=lambda x, y, z: None)
    # The axial cut the main window applied to this selection, and where
    # it came from ("typed", "axial peak" or "none"). Every count in this
    # panel is a count of the localizations that survived it: clearing the
    # Z fields on a pilot axon more than doubled the selection and more
    # than halved fraction_inside, with nothing in the table to say why.
    z_range: Optional[Tuple[float, float]] = None
    z_range_source: str = "none"
    # The DBSCAN label of each kept cluster, in the order of ``clusters``,
    # or None when the selection has not been analysed. Every table names
    # a cluster by that label, the main window's own export included.
    cluster_labels: Callable[[], Optional[np.ndarray]] = field(
        default=lambda: None)
    # Told every time the panel's picture changes (threshold, smoothing,
    # margin, shift, images, a new selection or new clusters), so the axon
    # map that draws it can redraw: called where the panel drew its image.
    map_changed: Callable[[], None] = field(default=lambda: None)


@dataclass
class _Measurement:
    registration: ax.ImageRegistration
    path: str           # the localizations it was measured with
    reference: str      # the widefield image it was measured against
    notes: List[str]


class _Relay(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, int)


class AxoplasmWindow(QtWidgets.QMainWindow):
    """Membrane or interior, from tubulin and spectrin images -- widefield,
    or drawn from localizations acquired with the movie."""

    def __init__(self, inputs: AxoplasmInputs,
                 parent: Optional[QtWidgets.QWidget] = None,
                 export_callback: Optional[Callable[[], Any]] = None,
                 map_callback: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(parent)
        self.setObjectName("axoplasm_window")
        self.inputs = inputs
        # "Show on the axon map": the main window raises the MPS analysis
        # window in its Axoplasm view. None hides the button.
        self.map_callback = map_callback
        # Writes this axon's tables. The main window's single export: it
        # sees this panel, the analysis and the discard at once, which is
        # what keeps the rows of one axon describing one state.
        self.export_callback = export_callback
        self.tubulin: Optional[ax.WidefieldImage] = None
        self.reference: Optional[ax.WidefieldImage] = None
        self.tubulin_offset: Tuple[float, float] = (0.0, 0.0)
        self.reference_offset: Tuple[float, float] = (0.0, 0.0)
        self.tubulin_notes: List[str] = []
        self.reference_notes: List[str] = []
        self.region_notes: List[str] = []
        self.measured: Optional[_Measurement] = None
        self.axoplasm: Optional[ax.AxoplasmMask] = None
        self.result: Optional[ax.Classification] = None
        # Where each localization is through its cluster
        # (ax.localization_labels), once both images placed the clusters.
        self.located: Optional[np.ndarray] = None
        self.loc_cluster: Optional[np.ndarray] = None
        self.spectrin_interior: Optional[ax.AxoplasmMask] = None
        self.anchored: Optional[ax.AnchoredClusters] = None
        # The cluster centres (nm) the anchored result was computed for.
        self.anchored_centroids: Optional[np.ndarray] = None
        self.anchored_notes: List[str] = []
        self._contour_cache: Dict[bytes, Any] = {}
        self.selection_note: Optional[str] = None
        self._manual_threshold: Optional[float] = None
        self._threshold_range: Tuple[float, float] = (0.0, 1.0)
        self._updating = False
        self._thread: Optional[threading.Thread] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None
        # Bumped when the panel closes; a measurement started before is
        # discarded when it ends.
        self._generation = 0
        self._relay = _Relay()
        self._relay.finished.connect(self._measured)

        # What a drag has asked for and not yet been given. The timer is
        # parented to the window so it dies with it on the paths that
        # delete the panel rather than keep it.
        self._pending: Optional[str] = None
        self._recompute = QtCore.QTimer(self)
        self._recompute.setSingleShot(True)
        self._recompute.setInterval(RECOMPUTE_DELAY_MS)
        self._recompute.timeout.connect(self._run_pending)

        self.setWindowTitle(
            "Axoplasm - " + os.path.basename(str(inputs.movie.path)))
        self.resize(1320, 900)
        self.setStyleSheet(
            f"QMainWindow {{ background: {PANEL_BG}; }} "
            f"QLabel, QCheckBox {{ color: {TITLE_FG}; }} "
            f"QGroupBox {{ color: {TITLE_FG}; }}")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self.label_header = _label("")
        root.addWidget(self.label_header)
        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, stretch=1)

        # The controls, the results and the warnings scroll together: with
        # five sections they no longer fit a 900-pixel window.
        left = QtWidgets.QVBoxLayout()
        left_box = QtWidgets.QWidget()
        left_box.setObjectName("controls")
        left_box.setStyleSheet(f"#controls {{ background: {PANEL_BG}; }}")
        left_box.setLayout(left)
        left_box.setFixedWidth(450)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet(
            f"QScrollArea {{ background: {PANEL_BG}; }}")
        left_scroll.setWidget(left_box)
        left_scroll.setFixedWidth(
            450 + left_scroll.verticalScrollBar().sizeHint().width() + 4)
        body.addWidget(left_scroll)
        left.addWidget(self._build_images())
        left.addWidget(self._build_alignment())
        left.addWidget(self._build_mask())
        left.addWidget(self._build_classification())
        left.addWidget(self._build_anchored())
        self.findings = QtWidgets.QVBoxLayout()
        holder = QtWidgets.QWidget()
        holder.setObjectName("findings")
        holder.setStyleSheet(f"#findings {{ background: {PANEL_BG}; }}")
        holder.setLayout(self.findings)
        left.addWidget(holder)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, stretch=1)
        # The image moved to the axon map (UI stage 2): one line says where,
        # and one button brings it up beside this panel.
        on_map = QtWidgets.QHBoxLayout()
        self.label_on_map = _label(ON_THE_MAP, _DIM)
        self.label_on_map.setObjectName("label_on_map")
        on_map.addWidget(self.label_on_map, stretch=1)
        self.btn_show_map = QtWidgets.QPushButton(SHOW_ON_MAP)
        self.btn_show_map.setObjectName("btn_show_on_map")
        self.btn_show_map.setToolTip(SHOW_ON_MAP_TIP)
        self.btn_show_map.clicked.connect(self._on_show_map)
        self.btn_show_map.setVisible(self.map_callback is not None)
        on_map.addWidget(self.btn_show_map)
        right.addLayout(on_map)
        self.plot_hist = pg.PlotWidget()
        style_dark(self.plot_hist)
        set_title(self.plot_hist,
                  "Distance to the tubulin mask's edge (positive inside)")
        self.plot_hist.setLabel("bottom", "distance [nm]", color=AXIS_FG)
        self.plot_hist.setLabel("left", "localizations", color=AXIS_FG)
        right.addWidget(self.plot_hist, stretch=1)

        self._show_header()
        self._refresh()

    # ------------------------------------------------------------ layout
    def _build_images(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("1. Images")
        lay = QtWidgets.QGridLayout(box)
        self.edit_tubulin = QtWidgets.QLineEdit()
        self.edit_reference = QtWidgets.QLineEdit()
        self.edit_movie = QtWidgets.QLineEdit(str(self.inputs.movie.path))
        images = mps_file_drop.IMAGE_SUFFIXES
        # A widefield image, or -- dual view -- the localizations of the
        # same protein acquired together with the movie, which need no
        # alignment at all.
        either = (tuple(mps_file_drop.IMAGE_SUFFIXES)
                  + tuple(mps_file_drop.LOCALIZATION_SUFFIXES))
        pattern = ("Widefield image or localizations "
                   "(*.tif *.tiff *.hdf5 *.h5 *.csv)")
        rows = (
            ("betaIII-tubulin, widefield image or localizations:",
             self.edit_tubulin, "betaIII-tubulin", pattern,
             self._load_tubulin, either),
            ("betaII-spectrin, widefield image or localizations (to align; "
             "section 5 needs it):",
             self.edit_reference, "betaII-spectrin", pattern,
             self._load_reference, either),
            ("Localizations of the whole movie (to align):", self.edit_movie,
             "Localizations of the whole movie",
             "Localizations (*.hdf5 *.h5 *.csv)", None,
             mps_file_drop.LOCALIZATION_SUFFIXES),
        )
        for i, (text, edit, title, pattern, loader, suffixes) in enumerate(rows):
            lay.addWidget(_label(text, _DIM), 2 * i, 0, 1, 2)
            button = QtWidgets.QPushButton("Browse...")
            button.setToolTip(f"Choose the file for: {text}\n\n"
                              f"It can also be dropped onto the box beside "
                              f"this button.")
            button.clicked.connect(
                lambda _=False, e=edit, t=title, p=pattern, f=loader:
                self._browse(e, t, p, f))
            if loader is not None:
                edit.editingFinished.connect(loader)
            mps_file_drop.accept_files(
                edit, suffixes, self._dropped(edit, loader),
                hint="Drop the file here, or press Browse")
            lay.addWidget(edit, 2 * i + 1, 0)
            lay.addWidget(button, 2 * i + 1, 1)
        self.edit_movie.setToolTip(
            "The alignment needs the whole field of the movie the selection "
            "was picked from: one picked axon alone rarely aligns.")
        for edit in (self.edit_tubulin, self.edit_reference):
            edit.setToolTip(
                "A widefield image taken with the movie, placed on the "
                "localizations by the shift of section 2.\n\n"
                "Or, for dual-view data, a localization file of the same "
                "protein acquired AT THE SAME TIME as the movie, on the "
                "other half of the camera: its localizations are counted on "
                "the movie's own camera grid, so it needs no shift and no "
                "widefield image. For the spectrin, that can be the movie's "
                "own file.")
        return box

    def _build_alignment(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("2. Alignment")
        lay = QtWidgets.QGridLayout(box)
        self.btn_measure = QtWidgets.QPushButton("Measure the shift")
        self.btn_measure.setToolTip(
            "Cross-correlates the spectrin widefield image with the "
            "localizations. The tubulin image is taken to share the "
            "widefield image's position, since they were acquired together.")
        self.btn_measure.clicked.connect(self.measure)
        self.btn_back = QtWidgets.QPushButton("Back to the measured shift")
        self.btn_back.setToolTip(
            "Undo a shift typed by hand and go back to the one the "
            "correlation measured.\n\n"
            "The exported row records which of the two it was, so a "
            "shift moved by hand is never reported as a measured one.")
        self.btn_back.clicked.connect(self._back_to_measured)
        lay.addWidget(self.btn_measure, 0, 0, 1, 2)
        lay.addWidget(self.btn_back, 0, 2, 1, 2)
        self.spin_dx = _spin(-20000, 20000, 10, suffix=" nm")
        self.spin_dy = _spin(-20000, 20000, 10, suffix=" nm")
        for spin in (self.spin_dx, self.spin_dy):
            spin.setToolTip("Added to the localizations to land them on the "
                            "widefield images.")
            spin.valueChanged.connect(self._shift_edited)
        lay.addWidget(_label("Shift x:", _DIM), 1, 0)
        lay.addWidget(self.spin_dx, 1, 1)
        lay.addWidget(_label("y:", _DIM), 1, 2)
        lay.addWidget(self.spin_dy, 1, 3)
        self.label_alignment = _label("")
        lay.addWidget(self.label_alignment, 2, 0, 1, 4)
        return box

    def _build_mask(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("3. Axoplasm mask")
        lay = QtWidgets.QGridLayout(box)
        self.slider_threshold = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_threshold.setRange(0, 1000)
        self.slider_threshold.valueChanged.connect(self._slider_moved)
        self.spin_threshold = _spin(-1e9, 1e9, 1)
        self.spin_threshold.setToolTip(
            "The grey level that separates the inside of the axon from "
            "the background in the tubulin image.\n\n"
            "Everything brighter is axoplasm. The slider beside it does "
            "the same thing; Otsu picks a value from the image itself.")
        self.spin_threshold.valueChanged.connect(self._threshold_typed)
        self.btn_otsu = QtWidgets.QPushButton("Otsu")
        self.btn_otsu.setToolTip(
            "Otsu's threshold (1979) for the region around the axon.")
        self.btn_otsu.clicked.connect(self._use_otsu)
        lay.addWidget(_label("Threshold:", _DIM), 0, 0)
        lay.addWidget(self.slider_threshold, 0, 1)
        lay.addWidget(self.spin_threshold, 0, 2)
        lay.addWidget(self.btn_otsu, 0, 3)
        self.spin_smooth = _spin(0, 2000, 10, suffix=" nm")
        self.spin_smooth.setValue(ax.DEFAULT_SMOOTH_SIGMA_PX * self.pixel_nm)
        self.spin_smooth.setToolTip(
            "Gaussian smoothing of the tubulin image before thresholding "
            "(sigma).")
        self.spin_smooth.valueChanged.connect(
            lambda _v: self._schedule("rebuild"))
        lay.addWidget(_label("Smoothing:", _DIM), 1, 0)
        lay.addWidget(self.spin_smooth, 1, 1)
        self.label_mask = _label("")
        lay.addWidget(self.label_mask, 2, 0, 1, 4)
        return box

    def _build_classification(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("4. Membrane or interior")
        lay = QtWidgets.QGridLayout(box)
        self.spin_margin = _spin(0, 5000, 10, suffix=" nm")
        self.spin_margin.setValue(ax.DEFAULT_MARGIN_NM)
        self.spin_margin.setToolTip(
            "How far inside a widefield edge a cluster must be before that "
            "image puts it inside the axon. The edge is blurred by the "
            "diffraction limit, so a spectrin ring on the membrane falls "
            "half inside without a margin. A cluster is inside only when "
            "both images put it inside (section 5), and a localization only "
            "when its cluster is.")
        self.spin_margin.valueChanged.connect(
            lambda _v: self._schedule("reclassify"))
        lay.addWidget(_label("Margin:", _DIM), 0, 0)
        lay.addWidget(self.spin_margin, 0, 1)
        self.btn_export = QtWidgets.QPushButton("Export axon...")
        self.btn_export.setToolTip(
            "Write this axon to the tables: the analysis, this panel and "
            "the discard in one row, and the per-cluster and "
            "per-localization tables if you ask for them. The same button "
            "as in the main window, so nothing is written twice.")
        self.btn_export.clicked.connect(self._export_clicked)
        lay.addWidget(self.btn_export, 0, 2)
        self.btn_image = QtWidgets.QPushButton("Export image...")
        self.btn_image.setToolTip(
            "Write the distance histogram as a figure: at the width a journal "
            "asks for, at 300 to 1200 dpi, or as an SVG. The image with the "
            "masks and the clusters is exported from the axon map (MPS "
            "analysis window, Export image..., Axon map).")
        self.btn_image.clicked.connect(self._on_export_image)
        lay.addWidget(self.btn_image, 0, 3)
        self.label_result = _label("")
        lay.addWidget(self.label_result, 1, 0, 1, 3)
        return box

    def _build_anchored(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("5. Clusters anchored to the membrane")
        box.setToolTip(
            "A cluster of the MPS analysis is discarded only when BOTH "
            "widefield images put it inside the axon by more than the "
            "margin: inside the tubulin mask, and inside the dark area the "
            "betaII-spectrin ring encloses. The contour is then rebuilt "
            "from the rest with 2-opt from every starting point, as the "
            "MPS analysis builds its own, and the MPS analysis window "
            "repeats every parameter with and without them. The cross is "
            "the centre of that contour, its area centroid.")
        lay = QtWidgets.QVBoxLayout(box)
        self.label_anchored = _label("")
        lay.addWidget(self.label_anchored)
        return box

    # ------------------------------------------------------------ state
    @property
    def pixel_nm(self) -> float:
        return float(self.inputs.loc.pixel_size_nm)

    def _show_header(self) -> None:
        loc = self.inputs.loc
        self.label_header.setText(
            f"Channel 1: {os.path.basename(str(self.inputs.movie.path))}. "
            f"Selection: {loc.n:,} localizations, ROI "
            f"{describe_roi(self.inputs.roi)}; pixel {self.pixel_nm:g} nm.")

    def update_selection(self, inputs: AxoplasmInputs) -> None:
        """The main window applied another selection."""
        self.inputs = inputs
        if self._manual_threshold is not None:
            # A threshold chosen for one axon is not carried to another. The
            # note stays until the threshold is touched again: dragging an
            # ROI can apply several selections in a row.
            self._manual_threshold = None
            self.selection_note = (
                "The selection changed in the main window: the threshold "
                "set by hand was replaced by Otsu's for the new selection.")
        self._show_header()
        self._rebuild()

    def refresh_clusters(self) -> None:
        """The main window analysed the selection's clusters again."""
        self._reclassify()

    def closeEvent(self, event: Any) -> None:
        self._generation += 1
        # A timer armed just before the close would otherwise fire on a
        # hidden window and push a new discard into the main window. The
        # panel is kept alive and reopened on one of the three close
        # paths, so it cannot be left to deletion.
        self._recompute.stop()
        self._pending = None
        self._close_progress()
        super().closeEvent(event)

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.hide()
            self._progress.deleteLater()
            self._progress = None

    def _dropped(self, edit: QtWidgets.QLineEdit,
                 loader: Optional[Callable[[], None]]
                 ) -> Callable[[List[str]], None]:
        """Take the first file of a drag into one of the fields."""
        def handler(paths: List[str]) -> None:
            edit.setText(paths[0])
            if loader is not None:
                loader()
        return handler

    def _browse(self, edit: QtWidgets.QLineEdit, title: str, pattern: str,
                loader: Optional[Callable[[], None]]) -> None:
        start = os.path.dirname(edit.text() or str(self.inputs.movie.path))
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, title, start, pattern)
        if chosen:
            edit.setText(chosen)
            if loader is not None:
                loader()

    def _load_image(self, edit: QtWidgets.QLineEdit, what: str
                    ) -> Tuple[Optional[ax.WidefieldImage], Tuple[float, float],
                               List[str]]:
        path = edit.text().strip()
        if not path:
            return None, (0.0, 0.0), []
        try:
            if path.lower().endswith(tuple(
                    mps_file_drop.LOCALIZATION_SUFFIXES)):
                movie = self.inputs.movie
                extent = (0.0, 0.0)
                if movie is not None and len(movie.x_nm):
                    extent = (float(np.max(movie.x_nm)) / self.pixel_nm,
                              float(np.max(movie.y_nm)) / self.pixel_nm)
                image = ax.image_from_localizations(
                    path, self.pixel_nm, min_extent_px=extent)
                offset = (0.0, 0.0)
                notes: List[str] = []
            else:
                image = ax.load_widefield(path)
                offset, notes = ax.camera_offset(
                    image, self.inputs.loc.info, self.pixel_nm)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.critical(
                self, "Axoplasm", f"The {what} image was not loaded:\n{error}")
            edit.setText("")
            return None, (0.0, 0.0), []
        return image, offset, list(image.notes) + notes

    def load_tubulin(self, path: str) -> None:
        self.edit_tubulin.setText(path)
        self._load_tubulin()

    def load_reference(self, path: str) -> None:
        self.edit_reference.setText(path)
        self._load_reference()

    def _load_tubulin(self) -> None:
        current = self.tubulin.path if self.tubulin else ""
        if self.edit_tubulin.text().strip() == current:
            return
        (self.tubulin, self.tubulin_offset,
         self.tubulin_notes) = self._load_image(self.edit_tubulin, "tubulin")
        self._manual_threshold = None
        self.selection_note = None
        self._rebuild()

    def _load_reference(self) -> None:
        current = self.reference.path if self.reference else ""
        if self.edit_reference.text().strip() == current:
            return
        (self.reference, self.reference_offset,
         self.reference_notes) = self._load_image(self.edit_reference,
                                                  "spectrin widefield")
        # The spectrin image gives section 5 its ring interior: finding the
        # discarded clusters again, with this image or without one, also
        # tells the main window.
        self._reclassify()

    # -------------------------------------------------------- alignment
    def _shift_px(self) -> Tuple[float, float]:
        return (self.spin_dx.value() / self.pixel_nm,
                self.spin_dy.value() / self.pixel_nm)

    def registration(self) -> ax.ImageRegistration:
        """The shift in use, with the measurement it came from."""
        if self._placed_by_construction():
            # Every image is the movie's own kind of data, acquired with it:
            # nothing was measured and nothing needs to be.
            return ax.ImageRegistration(shift_px=(0.0, 0.0),
                                        source="same acquisition")
        shift = self._shift_px()
        measured = self.measured.registration if self.measured else None
        if measured is None:
            source = "none" if shift == (0.0, 0.0) else "manual"
            return ax.ImageRegistration(shift_px=shift, source=source)
        same = (abs(shift[0] - measured.shift_px[0]) * self.pixel_nm < 0.05
                and abs(shift[1] - measured.shift_px[1]) * self.pixel_nm < 0.05)
        return ax.ImageRegistration(
            shift_px=shift, source="measured" if same else "adjusted",
            peak=measured.peak, zero=measured.zero,
            runner_up=measured.runner_up, score=measured.score,
            n_localizations=measured.n_localizations,
            warnings=list(measured.warnings))

    def _set_shift(self, shift_px: Tuple[float, float]) -> None:
        self._updating = True
        try:
            self.spin_dx.setValue(shift_px[0] * self.pixel_nm)
            self.spin_dy.setValue(shift_px[1] * self.pixel_nm)
        finally:
            self._updating = False

    def _shift_edited(self, _value: float) -> None:
        # The shift spins step by 10 nm and are held down the same way
        # the margin is.
        if not self._updating:
            self._schedule("rebuild")

    def _back_to_measured(self) -> None:
        if self.measured is not None:
            self._set_shift(self.measured.registration.shift_px)
            self._rebuild()

    def measure(self) -> None:
        """Measure the shift in a worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self.reference is None:
            QtWidgets.QMessageBox.warning(
                self, "Axoplasm",
                "Choose the widefield betaII-spectrin image first.")
            return
        path = self.edit_movie.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(
                self, "Axoplasm",
                "Choose the localization file of the whole movie.")
            return
        reference = self.reference
        movie = self.inputs.movie
        selection = self.inputs.loc
        generation = self._generation

        def work() -> _Measurement:
            notes: List[str] = []
            if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                    os.path.abspath(str(movie.path))):
                loc = movie
            else:
                loc = load_localizations(path)
            pixel = loc.pixel_size_nm
            if not pixel or abs(pixel - selection.pixel_size_nm) > 1e-6:
                raise ValueError(
                    f"{os.path.basename(path)} gives a pixel size of {pixel} "
                    f"nm and the selection {selection.pixel_size_nm:g} nm.")
            source = ax.movie_geometry(loc.info)[3]
            own = ax.movie_geometry(selection.info)[3]
            if source is None or own is None:
                notes.append(
                    "Picasso's metadata does not name the movie of one of "
                    "the files: check that the alignment localizations come "
                    "from the same movie as the selection.")
            elif os.path.basename(source) != os.path.basename(own):
                notes.append(
                    f"The alignment localizations come from "
                    f"{os.path.basename(source)} and the selection from "
                    f"{os.path.basename(own)}: the shift may not apply.")
            offset, more = ax.camera_offset(reference, loc.info, pixel)
            notes.extend(more)
            registration = ax.measure_shift(
                reference.image, loc.x_nm / pixel + offset[0],
                loc.y_nm / pixel + offset[1])
            return _Measurement(registration=registration, path=path,
                                reference=reference.path, notes=notes)

        def body() -> None:
            try:
                result, error = work(), None
            except Exception as exc:  # noqa: BLE001 - shown to the user
                result, error = None, exc
            self._relay.finished.emit(result, error, generation)

        self._progress = QtWidgets.QProgressDialog(
            "Measuring the shift...", None, 0, 0, self)
        self._progress.setWindowTitle("Axoplasm")
        self._progress.setWindowModality(QtCore.Qt.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.show()
        self.btn_measure.setEnabled(False)
        self._thread = threading.Thread(target=body, daemon=True)
        self._thread.start()

    def wait(self, timeout: float = 600.0) -> None:
        """Block until a measurement finishes (for scripts and tests)."""
        self.flush()
        if self._thread is not None:
            self._thread.join(timeout)
        QtWidgets.QApplication.processEvents()

    def _measured(self, result: Optional[_Measurement],
                  error: Optional[BaseException], generation: int) -> None:
        # The worker is done once its result arrives, even if its thread
        # has not quite exited: the Measure button must come back.
        self._thread = None
        if generation != self._generation:
            self._refresh()
            return
        self._close_progress()
        if error is not None or result is None:
            self._refresh()
            QtWidgets.QMessageBox.critical(
                self, "Axoplasm", f"The shift could not be measured:\n{error}")
            return
        self.measured = result
        self._set_shift(result.registration.shift_px)
        self._rebuild()

    # ------------------------------------------------------ computation
    def _coordinates(self, x_nm: np.ndarray, y_nm: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Tubulin-image pixel coordinates of positions in nm."""
        return self._coordinates_for(self.tubulin, self.tubulin_offset,
                                     x_nm, y_nm)

    def _coordinates_for(self, image: Optional[ax.WidefieldImage],
                         offset: Tuple[float, float], x_nm: np.ndarray,
                         y_nm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates in ``image`` of positions in nm.

        An image drawn from localizations acquired with the movie sits on
        the movie's own camera grid: no camera offset and no shift, which
        belong to widefield images taken at another moment.
        """
        if image is not None and image.from_localizations:
            return (np.asarray(x_nm, float) / self.pixel_nm,
                    np.asarray(y_nm, float) / self.pixel_nm)
        return self._coordinates_in(offset, x_nm, y_nm)

    def _placed_by_construction(self) -> bool:
        """Every image loaded was drawn from localizations of the movie."""
        loaded = [i for i in (self.tubulin, self.reference) if i is not None]
        return bool(loaded) and all(i.from_localizations for i in loaded)

    def _needs_shift(self) -> bool:
        """Some image in use is a widefield image placed by a shift."""
        return any(i is not None and not i.from_localizations
                   for i in (self.tubulin, self.reference))

    def _coordinates_in(self, offset: Tuple[float, float], x_nm: np.ndarray,
                        y_nm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates, in an image at ``offset``, of positions in nm."""
        sx, sy = self._shift_px()
        return (np.asarray(x_nm, float) / self.pixel_nm + offset[0] + sx,
                np.asarray(y_nm, float) / self.pixel_nm + offset[1] + sy)

    def _schedule(self, what: str) -> None:
        """Recompute once the drag stops, not on every step of it.

        A rebuild subsumes a reclassify -- it ends in one -- so a pending
        rebuild is never downgraded.
        """
        self._pending = ("rebuild" if (what == "rebuild"
                                       or self._pending == "rebuild")
                         else "reclassify")
        self._recompute.start()

    def _run_pending(self) -> None:
        """What the timer fires. Clears the request before dispatching,
        because the work below re-enters the event loop."""
        what, self._pending = self._pending, None
        if what == "rebuild":
            self._rebuild()
        elif what == "reclassify":
            self._reclassify()

    def flush(self) -> None:
        """Do now whatever a drag has left pending.

        Anything that READS what the panel computed has to call this
        first: what is exported, drawn into a figure or compared against
        the analysis must be what the controls currently say, not what
        they said 250 ms ago.
        """
        if self._pending is not None:
            self._recompute.stop()
            self._run_pending()

    def _rebuild(self) -> None:
        """Mask and classification from the current controls."""
        self.axoplasm = self.result = self.located = self.loc_cluster = None
        self.region_notes = []
        if self.tubulin is not None and self.inputs.loc.n:
            col, row = self._coordinates(self.inputs.loc.x_nm,
                                         self.inputs.loc.y_nm)
            centre, radius, reach = ax.axon_centre(col, row)
            try:
                self.axoplasm = ax.build_mask(
                    self.tubulin.image, centre, radius, self.pixel_nm,
                    reach_px=reach, threshold=self._manual_threshold,
                    smooth_sigma_px=self.spin_smooth.value() / self.pixel_nm)
            except ValueError as error:
                self.region_notes = [str(error)]
            else:
                self._sync_threshold()
        self._reclassify()

    def _reclassify(self) -> None:
        # Seven callers reach here directly -- loading an image, pressing
        # Otsu, going back to the measured shift, a new selection. Any of
        # them can land inside the pause a drag opened, and without this
        # the timer would fire afterwards and run the whole chain a
        # second time, including the 2.6 s discard comparison.
        self._recompute.stop()
        self._pending = None
        self.result = self.located = self.loc_cluster = None
        self.spectrin_interior = self.anchored = None
        self.anchored_centroids = None
        self.anchored_notes = []
        if self.axoplasm is not None:
            margin = self.spin_margin.value()
            loc = self.inputs.loc
            col, row = self._coordinates(loc.x_nm, loc.y_nm)
            self.result = ax.classify(self.axoplasm, col, row, margin)
            centroids = self.inputs.clusters()
            if centroids is not None:
                c = np.asarray(centroids, float)
                if c.size == 0:
                    c = np.empty((0, 2))
                self._find_anchored(c[:, :2], margin)
            if self.anchored is not None:
                cluster_of = self.inputs.cluster_of(loc.x_nm, loc.y_nm,
                                                    loc.z_nm)
                if cluster_of is not None and len(cluster_of) == loc.n:
                    self.loc_cluster = np.asarray(cluster_of, dtype=np.intp)
                    self.located = ax.localization_labels(
                        self.loc_cluster, self.anchored.discarded)
        self._refresh()
        # After drawing: the main window may take seconds to repeat the MPS
        # analysis without the discarded clusters.
        self.inputs.anchored_changed(self.anchored, self.anchored_centroids)

    def _registration_state(self) -> str:
        """How the widefield images are placed right now, in one phrase."""
        reg = self.registration()
        words = {"none": "no shift", "manual": "set by hand",
                 "measured": "measured",
                 "adjusted": "measured, then adjusted by hand",
                 "same acquisition": "placed by construction: every image "
                                     "drawn from localizations acquired "
                                     "with the movie"}
        state = words.get(reg.source, reg.source)
        if reg.score is not None:
            state += f", score {reg.score:.1f}"
            if reg.score < ax.MIN_REGISTRATION_SCORE:
                state += " (low)"
        return state

    def _find_anchored(self, centroids: np.ndarray, margin: float) -> None:
        """The clusters both images put inside, and the contour without."""
        if self.reference is None or self.axoplasm is None \
                or len(centroids) < 3:
            return
        # Which clusters are inside depends entirely on where the images
        # sit: on a pilot axon the discarded set and the contour length
        # changed once the shift was measured.
        # Sorting them before anything is placed would hand the results
        # window, and the tables, a discard measured on an unplaced image.
        # Only a widefield image needs placing. Localizations acquired with
        # the movie are placed by construction, and asking for a shift
        # there would block the one case that needs none.
        if self._needs_shift() and self.registration().source == "none":
            self.anchored_notes = [
                "The widefield images are not placed on the localizations "
                "yet: measure the shift in section 2 (or set one by hand) "
                "before the clusters can be sorted."]
            return
        offset = self.reference_offset
        col, row = self._coordinates_for(self.reference, offset,
                                         self.inputs.loc.x_nm,
                                         self.inputs.loc.y_nm)
        centre, radius, reach = ax.axon_centre(col, row)
        scol, srow = self._coordinates_for(self.reference, offset,
                                           centroids[:, 0], centroids[:, 1])
        try:
            interior = ax.build_ring_interior(
                self.reference.image, centre, radius, self.pixel_nm,
                ring_col=scol, ring_row=srow, reach_px=reach)
        except ValueError as error:
            self.anchored_notes = [str(error)]
            return
        if len(self._contour_cache) > CONTOUR_CACHE_SIZE:
            self._contour_cache.clear()
        tcol, trow = self._coordinates(centroids[:, 0], centroids[:, 1])
        self.spectrin_interior = interior
        self.anchored_centroids = centroids
        self.anchored = ax.anchored_clusters(
            centroids, self.axoplasm, tcol, trow, interior, scol, srow,
            margin, contour_all=self.inputs.contour(),
            contour_cache=self._contour_cache,
            registration=self._registration_state())

    def _sync_threshold(self) -> None:
        mask = self.axoplasm
        if mask is None:
            return
        low = float(np.min(mask.smoothed))
        high = float(np.max(mask.smoothed))
        self._updating = True
        try:
            self._threshold_range = (low, high)
            self.spin_threshold.setValue(mask.threshold)
            span = high - low
            position = 0 if span <= 0 else int(round(
                1000 * (mask.threshold - low) / span))
            self.slider_threshold.setValue(max(0, min(1000, position)))
        finally:
            self._updating = False

    def _slider_moved(self, position: int) -> None:
        if self._updating or self.axoplasm is None:
            return
        low, high = self._threshold_range
        # The threshold itself follows the slider at once; only the mask
        # built from it waits for the drag to stop.
        self._manual_threshold = low + (high - low) * position / 1000.0
        self.selection_note = None
        self._schedule("rebuild")

    def _threshold_typed(self, value: float) -> None:
        if self._updating or self.axoplasm is None:
            return
        self._manual_threshold = float(value)
        self.selection_note = None
        self._schedule("rebuild")

    def _use_otsu(self) -> None:
        self._manual_threshold = None
        self.selection_note = None
        self._rebuild()

    # ---------------------------------------------------------- display
    def _clear(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def warnings(self) -> List[str]:
        out = self.tubulin_notes + self.reference_notes + self.region_notes
        if self.selection_note:
            out.append(self.selection_note)
        if self.measured is not None:
            out.extend(self.measured.notes)
            out.extend(self.measured.registration.warnings)
        if self.axoplasm is not None:
            out.extend(self.axoplasm.warnings)
        out.extend(self.anchored_notes)
        if self.spectrin_interior is not None:
            out.extend(self.spectrin_interior.warnings)
        if self.anchored is not None:
            out.extend(self.anchored.warnings)
        return out

    def _refresh(self) -> None:
        self.btn_measure.setEnabled(
            self.reference is not None
            and not self.reference.from_localizations
            and not (self._thread is not None and self._thread.is_alive()))
        for widget in (self.spin_dx, self.spin_dy):
            widget.setEnabled(self._needs_shift())
        self.btn_back.setEnabled(self.measured is not None)
        self.btn_export.setEnabled(self.result is not None)
        for widget in (self.slider_threshold, self.spin_threshold,
                       self.btn_otsu):
            widget.setEnabled(self.axoplasm is not None)
        self._clear(self.findings)
        messages = self.warnings()
        if self.tubulin is None:
            self.findings.addWidget(_label(marked(
                "dim",
                "Choose the betaIII-tubulin widefield image, or its "
                "localizations when they were acquired with the movie."),
                _DIM))
        elif not messages:
            self.findings.addWidget(_label(
                marked("good", "No warnings."), _TEXT_OK))
        for text in messages:
            self.findings.addWidget(_label(marked("warn", text), _WARN))
        self.findings.addStretch(1)
        self._write_labels()
        self._draw()

    def _anchored_text(self) -> str:
        if self.tubulin is None:
            return ""
        if self.reference is None:
            return ("Load the betaII-spectrin widefield image (section 1): "
                    "a cluster is discarded only when it and the tubulin "
                    "both put it inside the axon.")
        found = self.anchored
        if found is None:
            # Why there is nothing to show, when there is a reason: an
            # empty section 5 left "measure the shift first" unsaid.
            if self.anchored_notes:
                return self.anchored_notes[0]
            if self.axoplasm is None:
                return ""
            if self.inputs.clusters() is None:
                return ("Run 'cluster Ch1' on this selection: the clusters "
                        "are the ones the MPS analysis kept.")
            return "Fewer than three clusters: there is no ring to look inside."
        lines = []
        interior = self.spectrin_interior
        if interior is not None and interior.mask.any():
            levels = interior.levels
            how = ("half way between the axon's dark inside and its ring"
                   if interior.threshold_source == "half maximum" else
                   "short of half way, where the ring opens")
            lines.append(
                f"Spectrin interior (purple): {interior.area_um2:.2f} µm², "
                f"cut at {interior.threshold:.0f} counts, {how} (inside "
                f"{levels['interior']:.0f}, ring {levels['ring']:.0f}).")
        else:
            lines.append("No spectrin interior was found, so no cluster is "
                         "discarded.")
        lines.append(
            f"Discarded (vermillion discs on the axon map): {found.n_discarded} of {found.n} "
            f"clusters, "
            f"inside both images by more than {found.margin_nm:.0f} nm. "
            f"Kept although one image alone puts them inside: "
            f"{found.n_tubulin_only} by the tubulin (green), "
            f"{found.n_spectrin_only} by the spectrin (purple).")
        parts = []
        same = found.contour_all_starts is found.contour_all
        if found.contour_all is not None:
            parts.append(f"{found.contour_all.perimeter_um:.2f} µm with all "
                         f"clusters, as the MPS analysis connects them"
                         + (" (2-opt from every start; dashed on the axon map)"
                            if same else " (dashed on the axon map)"))
        if found.contour_all_starts is not None and not same:
            parts.append(f"{found.contour_all_starts.perimeter_um:.2f} µm with "
                         f"all clusters and 2-opt from every start")
        new = found.contour_anchored
        if new is not None:
            spread = new.start_spread_um
            parts.append(
                f"{new.perimeter_um:.2f} µm without the discarded ones "
                f"(solid on the axon map; 2-opt from all {new.n_starts} starts"
                + (f", which a single start could have missed by up to "
                   f"{spread:.2f} µm" if spread else "") + ")")
        if parts:
            lines.append("Perimeter: " + "; ".join(parts) + ".")
        if new is not None and new.centre is not None:
            c = new.centre
            lines.append(
                f"Centre of the contour (+, its area centroid): "
                f"x {c.x_nm:.0f}, y {c.y_nm:.0f} nm"
                + ("" if c.max_shift_nm is None else
                   f"; it moves at most {c.max_shift_nm:.0f} nm when one "
                   f"cluster is left out") + ".")
        if new is not None and found.contour_all_starts is not None:
            lines.append("The MPS analysis window repeats every parameter "
                         "without the discarded clusters, next to the "
                         "measured ones.")
        return "\n".join(lines)

    def _write_labels(self) -> None:
        self.label_anchored.setText(self._anchored_text())
        reg = self.registration()
        sx, sy = reg.shift_nm(self.pixel_nm)
        words = {"none": "no shift", "manual": "set by hand",
                 "measured": "as measured", "adjusted": "measured, then "
                 "adjusted by hand",
                 "same acquisition": "none needed: the images are drawn "
                                     "from localizations acquired with the "
                                     "movie"}.get(reg.source, reg.source)
        lines = [f"Shift in use: x {sx:+.0f} nm, y {sy:+.0f} nm ({words})."]
        if self.measured is not None:
            m = self.measured.registration
            msx, msy = m.shift_nm(self.pixel_nm)
            lines.append(
                f"Measured against "
                f"{os.path.basename(self.measured.reference)}: "
                f"({msx:+.0f}, {msy:+.0f}) nm on "
                f"{m.n_localizations:,} localizations; correlation "
                f"{m.peak:.2f} (unshifted {m.zero:.2f}, next peak "
                f"{'-' if m.runner_up is None else f'{m.runner_up:.2f}'}); "
                f"score {'-' if m.score is None else f'{m.score:.0f}'}.")
        times = []
        if self.tubulin is not None and self.tubulin.acquired:
            times.append(f"tubulin at {self.tubulin.acquired}")
        if self.reference is not None and self.reference.acquired:
            times.append(f"spectrin at {self.reference.acquired}")
        movie_time = ax.movie_geometry(self.inputs.loc.info)[2]
        if movie_time:
            times.append(f"the movie started at {movie_time}")
        if times:
            lines.append("Acquired: " + "; ".join(times) + ".")
        self.label_alignment.setText("\n".join(lines))

        mask = self.axoplasm
        if mask is None:
            self.label_mask.setText("")
        else:
            ratio = mask.area_ratio
            if mask.threshold_source == "otsu":
                source = "Otsu's"
            else:
                source = f"set by hand; Otsu's is {mask.otsu:.1f}"
            self.label_mask.setText(
                f"Threshold {mask.threshold:.1f} ({source}). "
                f"Mask {mask.area_um2:.2f} µm²"
                + ("" if ratio is None else
                   f", {ratio:.2f} times the area inside the spectrin ring")
                + ".")

        self.label_result.setText(self._located_text())

    def _located_text(self) -> str:
        """
        Section 4: where the localizations are. A localization is counted
        inside only through its cluster, when both images put that cluster
        inside; the tubulin mask alone decides nothing here.
        """
        result = self.result
        if result is None:
            return ""
        text = f"{result.n:,} localizations"
        outside = result.count(ax.LABEL_OUTSIDE)
        if outside:
            text += f", {outside:,} of them outside the analysed region"
        text += "; their distances to the tubulin mask's edge are below. "
        located = self.located
        found = self.anchored
        centroids = self.inputs.clusters()
        if located is not None and found is not None:
            inside = int(np.sum(located == ax.LOC_INSIDE))
            membrane = int(np.sum(located == ax.LOC_MEMBRANE))
            free = int(np.sum(located == ax.LOC_NO_CLUSTER))
            share = inside / result.n if result.n else 0.0
            text += (
                f"Inside the axon: {inside:,} ({share:.1%}), the "
                f"localizations of the {found.n_discarded} clusters both "
                f"images put inside. At the membrane: {membrane:,}, those "
                f"of the {found.n - found.n_discarded} kept clusters. In no "
                f"cluster: {free:,}.")
        elif centroids is None:
            text += ("Run 'cluster Ch1' on this selection: a localization "
                     "is counted inside only through its cluster.")
        elif len(centroids) == 0:
            text += "The MPS analysis kept none of this selection's clusters."
        elif self.reference is None:
            text += ("Load the betaII-spectrin widefield image: a "
                     "localization is counted inside only when both images "
                     "put its cluster inside (section 5).")
        else:
            text += "Section 5 has not placed the clusters yet."
        return text

    def _draw(self) -> None:
        """Draw what this panel draws itself - the distance histogram - and
        tell the axon map that the picture changed (it reads ``map_state``)."""
        self._draw_hist()
        self.inputs.map_changed()

    def _draw_hist(self) -> None:
        result = self.result
        self.plot_hist.clear()
        if self.axoplasm is None or self.tubulin is None or result is None:
            return
        d = result.distance_nm[np.isfinite(result.distance_nm)]
        if d.size:
            edges = np.arange(-HIST_RANGE_NM, HIST_RANGE_NM + HIST_BIN_NM,
                              HIST_BIN_NM)
            counts, _ = np.histogram(np.clip(d, edges[0], edges[-1]),
                                     bins=edges)
            self.plot_hist.plot(edges, counts, stepMode="center",
                                pen=pg.mkPen(TITLE_FG, width=1.5))
        self.plot_hist.addItem(pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(neutral(dark=True), style=QtCore.Qt.DashLine)))
        self.plot_hist.addItem(pg.InfiniteLine(
            pos=result.margin_nm, angle=90, pen=pg.mkPen(_INTERIOR)))

    def map_state(self) -> map_layers.AxoplasmMapState:
        """What this panel draws on the axon map, computed as its image used
        to be drawn: the tubulin region of the mask (and the same area in the
        spectrin image), the two cased edges, the localizations of the
        selection by class, the anchored classification, and the words the
        map's captions need (threshold source, the selection note, margin,
        registration). Whatever a drag left pending is done first."""
        self.flush()
        mask = self.axoplasm
        found = self.anchored if self.anchored_centroids is not None else None
        loc = self.inputs.loc
        return map_layers.axoplasm_map_state(
            tubulin_image=None if self.tubulin is None else self.tubulin.image,
            tubulin_offset=self.tubulin_offset,
            reference_image=None if self.reference is None else self.reference.image,
            reference_offset=self.reference_offset,
            mask=mask, interior=self.spectrin_interior,
            shift_px=self._shift_px(), pixel_nm=self.pixel_nm,
            loc_x=np.asarray(loc.x_nm, dtype=float), loc_y=np.asarray(loc.y_nm, dtype=float),
            located=self.located, anchored=found,
            anchored_centroids=None if found is None else self.anchored_centroids,
            clusters=self.inputs.clusters(),
            threshold_source=("otsu" if mask is None or mask.threshold_source == "otsu" else "manual"),
            selection_note=self.selection_note, margin_nm=float(self.spin_margin.value()),
            registration=self._registration_state(), has_result=self.result is not None)

    def _on_show_map(self) -> None:
        """"Show on the axon map" (3.1)."""
        self.flush()
        if self.map_callback is not None:
            self.map_callback()

    # ------------------------------------------------------------ export
    def summary_row(self) -> Dict[str, Any]:
        assert self.axoplasm is not None and self.result is not None
        return ax.summary_row(
            localizations=str(self.inputs.movie.path),
            tubulin=self.tubulin.path if self.tubulin else "",
            # The image the shift was measured against, not whichever is
            # loaded now; without a measurement, the one shown to align by
            # hand.
            reference=(self.measured.reference if self.measured
                       else self.reference.path if self.reference else ""),
            registration_file=self.measured.path if self.measured else "",
            roi=describe_roi(self.inputs.roi),
            pixel_size_nm=self.pixel_nm,
            offset_px=self.tubulin_offset,
            registration=self.registration(),
            mask=self.axoplasm, result=self.result,
            spectrin=self.spectrin_interior, anchored=self.anchored,
            tubulin_source=(self.tubulin.source if self.tubulin
                            else "widefield image"),
            spectrin_source=(self.reference.source
                             if self.reference is not None
                             and self.spectrin_interior is not None
                             else ""),
            spectrin_image=self._interior_image(), located=self.located,
            n_clusters=self._n_clusters(),
            z_range=self.inputs.z_range,
            z_range_source=self.inputs.z_range_source,
            warnings=self.warnings())

    def _n_clusters(self) -> Optional[int]:
        """How many clusters the MPS analysis kept here; None before."""
        centroids = self.inputs.clusters()
        return None if centroids is None else int(len(centroids))

    def _interior_image(self) -> str:
        """The spectrin image the ring interior was found in: the one
        loaded, since loading one finds the interior again."""
        if self.spectrin_interior is None or self.reference is None:
            return ""
        return self.reference.path

    def cluster_rows(self) -> List[Dict[str, Any]]:
        assert self.anchored is not None and self.anchored_centroids is not None
        return ax.cluster_rows(
            localizations=str(self.inputs.movie.path),
            roi=describe_roi(self.inputs.roi),
            centroids_nm=self.anchored_centroids, anchored=self.anchored,
            spectrin_image=self._interior_image(),
            labels=self.inputs.cluster_labels())

    def localization_rows(self) -> List[Dict[str, Any]]:
        assert self.result is not None
        loc = self.inputs.loc
        source = str(self.inputs.movie.path)
        roi = describe_roi(self.inputs.roi)
        # The cluster each localization is in (numbered as in the clusters
        # table) and where it is through that cluster; empty until both
        # images have placed the clusters.
        placed = self.located is not None and self.loc_cluster is not None
        cluster = (self.loc_cluster if placed
                   else np.full(loc.n, -1, dtype=np.intp))
        # The label the rest of the program calls this cluster by, not its
        # position among the kept ones: the two differ as soon as the
        # automatic curation removes a cluster.
        names = self.inputs.cluster_labels()
        where = (self.located if placed
                 else np.full(loc.n, "", dtype=object))
        return [
            {"source_localizations": source, "roi": roi,
             "x_nm": round(float(x), 2), "y_nm": round(float(y), 2),
             "z_nm": round(float(z), 2),
             # Empty where there is no distance: off the analysed region,
             # or no mask at all. Written as "nan" and "-inf", both were
             # read back as numbers.
             "distance_to_tubulin_edge_nm": (
                 round(float(d), 1) if np.isfinite(d) else None),
             "cluster_label": ("" if c < 0 else int(c) if names is None
                               else int(names[c])),
             "label": str(label)}
            for x, y, z, d, c, label in zip(loc.x_nm, loc.y_nm, loc.z_nm,
                                            self.result.distance_nm,
                                            cluster, where)
        ]

    def _on_export_image(self) -> None:
        """
        Write the distance histogram as a figure.

        The image with the masks and the clusters is the axon map's: its
        export is the MPS analysis window's "Export image...", "Axon map".
        """
        # A figure written within the pause a drag opened would show the
        # mask from before the drag.
        self.flush()
        from tools import figure_export
        from tools.figure_export_ui import ask

        plots = {"Distance to the tubulin mask's edge": self.plot_hist}
        suggested = figure_export.suggested_name(
            str(self.inputs.movie.path), "axoplasm")
        request = ask(list(plots), suggested, parent=self, offer_white=False)
        if request is None:
            return
        plot = plots.get(request.plot)
        if plot is None:
            return
        try:
            written = figure_export.write(plot.getPlotItem(), request)
        except Exception as error:                        # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self, "Export failed",
                f"Could not write the image:\n\n{error}")
            return
        QtWidgets.QMessageBox.information(
            self, "Image written",
            f"{figure_export.describe(request)}\n\n{written}")

    def _export_clicked(self) -> None:
        """
        Ask the main window to write this axon.

        This panel used to write three tables of its own, from its own
        state, while the results window wrote another from its own -- which
        is how one axon came to be described twice, differently. What this
        panel measures is now part of the axon's single row, and the
        export that writes it is the one place that sees both.
        """
        if self.export_callback is None:
            QtWidgets.QMessageBox.information(
                self, "Export axon",
                "This panel was opened on its own, so it cannot write the "
                "tables. Use 'Export axon' in the main window.")
            return
        self.export_callback()


def show_axoplasm_window(
    inputs: AxoplasmInputs,
    parent: Optional[QtWidgets.QWidget] = None,
    export_callback: Optional[Callable[[], Any]] = None,
    map_callback: Optional[Callable[[], Any]] = None,
) -> AxoplasmWindow:
    """Open the axoplasm panel."""
    window = AxoplasmWindow(inputs, parent=parent,
                            export_callback=export_callback,
                            map_callback=map_callback)
    window.show()
    window.raise_()
    window.activateWindow()
    return window
