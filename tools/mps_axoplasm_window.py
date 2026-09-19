# -*- coding: utf-8 -*-
"""
The axoplasm panel: is each betaII-spectrin localization at the membrane
or inside the axon, according to a widefield betaIII-tubulin image?

Top to bottom:
  1. the images. The tubulin one, and optionally a widefield image of the
     super-resolved protein together with the localizations of the whole
     movie, to measure the shift between the acquisitions;
  2. the alignment, measured and adjustable by hand;
  3. the mask, with Otsu's threshold on the axon's region, adjustable;
  4. the classification with the margin the user sets, and the export;
  5. the MPS analysis' clusters that are not anchored to the membrane --
     the ones both widefield images put inside the axon -- and the contour
     rebuilt without them. The main window is told, and repeats the whole
     MPS analysis without them beside the original.

The calculations are in ``tools.mps_axoplasm``.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from scipy import ndimage

from tools import mps_axoplasm as ax
from tools import mps_file_drop
from tools.mps_io import load_localizations
from tools.mps_plot_style import AXIS_FG, TITLE_FG, set_title, style_dark
from tools.mps_twochannel_window import describe_roi
from tools.results_table import append_rows

_OK = "#5fd75f"
_WARN = "#ffaf5f"
_DIM = "#9a9a9a"
_INTERIOR = "#6fa8ff"
_MEMBRANE = "#ff9f43"
_OUTLINE = (0, 230, 230, 255)
_SPECTRIN_OUTLINE = (255, 230, 0, 255)
_DISCARDED = "#ff4040"
_ALL_CONTOUR = "#8a8a8a"
# The centre of the green contour, as the MPS analysis window draws it.
_CENTRE = "#cc79a7"
# Contours rebuilt with every 2-opt start, kept per set of cluster centres.
CONTOUR_CACHE_SIZE = 32

# Points drawn per class; the classification itself uses all of them.
MAX_DRAWN = 20000
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


@dataclass
class _Measurement:
    registration: ax.ImageRegistration
    path: str           # the localizations it was measured with
    reference: str      # the widefield image it was measured against
    notes: List[str]


class _Relay(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, int)


class AxoplasmWindow(QtWidgets.QMainWindow):
    """Membrane or interior, from a widefield tubulin image."""

    def __init__(self, inputs: AxoplasmInputs,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.inputs = inputs
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
        self.cluster_result: Optional[ax.Classification] = None
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

        self.setWindowTitle(
            "Axoplasm - " + os.path.basename(str(inputs.movie.path)))
        self.resize(1320, 900)
        self.setStyleSheet(
            "QMainWindow { background: #1a1a1a; } "
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
        left_box.setStyleSheet("#controls { background: #1a1a1a; }")
        left_box.setLayout(left)
        left_box.setFixedWidth(450)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea { background: #1a1a1a; }")
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
        holder.setStyleSheet("#findings { background: #1a1a1a; }")
        holder.setLayout(self.findings)
        left.addWidget(holder)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, stretch=1)
        self.check_reference = QtWidgets.QCheckBox(
            "Show the betaII-spectrin widefield image instead")
        self.check_reference.setToolTip(
            "The spectrin rings of the localizations should lie on the "
            "widefield membranes when the alignment is right.")
        self.check_reference.toggled.connect(lambda _on: self._draw())
        right.addWidget(self.check_reference)
        self.plot_image = pg.PlotWidget()
        style_dark(self.plot_image)
        self.plot_image.setAspectLocked(True)
        self.plot_image.setLabel("bottom", "x [nm]", color=AXIS_FG)
        self.plot_image.setLabel("left", "y [nm]", color=AXIS_FG)
        self.image_item = pg.ImageItem()
        self.outline_item = pg.ImageItem()
        self.spectrin_outline_item = pg.ImageItem()
        self.membrane_item = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(_MEMBRANE), size=2)
        self.interior_item = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(_INTERIOR), size=2)
        self.contour_all_item = pg.PlotDataItem(
            pen=pg.mkPen(_ALL_CONTOUR, width=1,
                         style=QtCore.Qt.PenStyle.DashLine))
        self.contour_item = pg.PlotDataItem(pen=pg.mkPen(_OK, width=2))
        self.cluster_item = pg.ScatterPlotItem(
            pen=pg.mkPen("w"), brush=None, size=9)
        self.discarded_item = pg.ScatterPlotItem(
            pen=pg.mkPen("k"), brush=pg.mkBrush(_DISCARDED), size=10)
        self.centre_item = pg.ScatterPlotItem(
            pen=pg.mkPen("k"), brush=pg.mkBrush(_CENTRE), size=18,
            symbol="+")
        for item in (self.image_item, self.outline_item,
                     self.spectrin_outline_item, self.membrane_item,
                     self.interior_item, self.contour_all_item,
                     self.contour_item, self.cluster_item,
                     self.discarded_item, self.centre_item):
            self.plot_image.addItem(item)
        right.addWidget(self.plot_image, stretch=3)
        self.plot_hist = pg.PlotWidget()
        style_dark(self.plot_hist)
        set_title(self.plot_hist,
                  "Distance to the axoplasm's edge (positive inside)")
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
        rows = (
            ("betaIII-tubulin, widefield:", self.edit_tubulin,
             "Widefield betaIII-tubulin", "TIFF (*.tif *.tiff)",
             self._load_tubulin, images),
            ("betaII-spectrin, widefield (to align):", self.edit_reference,
             "Widefield betaII-spectrin", "TIFF (*.tif *.tiff)",
             self._load_reference, images),
            ("Localizations of the whole movie (to align):", self.edit_movie,
             "Localizations of the whole movie",
             "Localizations (*.hdf5 *.h5 *.csv)", None,
             mps_file_drop.LOCALIZATION_SUFFIXES),
        )
        for i, (text, edit, title, pattern, loader, suffixes) in enumerate(rows):
            lay.addWidget(_label(text, _DIM), 2 * i, 0, 1, 2)
            button = QtWidgets.QPushButton("Browse...")
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
        self.spin_smooth.valueChanged.connect(lambda _v: self._rebuild())
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
            "A localization is interior when it lies deeper inside the mask "
            "than this; membrane otherwise. The widefield edge is blurred "
            "by the diffraction limit, so a spectrin ring on the membrane "
            "falls half inside a mask without a margin.")
        self.spin_margin.valueChanged.connect(lambda _v: self._reclassify())
        lay.addWidget(_label("Margin:", _DIM), 0, 0)
        lay.addWidget(self.spin_margin, 0, 1)
        self.btn_export = QtWidgets.QPushButton("Export CSV...")
        self.btn_export.clicked.connect(self._export_clicked)
        lay.addWidget(self.btn_export, 0, 2)
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
            image = ax.load_widefield(path)
            offset, notes = ax.camera_offset(image, self.inputs.loc.info,
                                             self.pixel_nm)
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
        self._refresh()

    # -------------------------------------------------------- alignment
    def _shift_px(self) -> Tuple[float, float]:
        return (self.spin_dx.value() / self.pixel_nm,
                self.spin_dy.value() / self.pixel_nm)

    def registration(self) -> ax.ImageRegistration:
        """The shift in use, with the measurement it came from."""
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
        if not self._updating:
            self._rebuild()

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
        return self._coordinates_in(self.tubulin_offset, x_nm, y_nm)

    def _coordinates_in(self, offset: Tuple[float, float], x_nm: np.ndarray,
                        y_nm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates, in an image at ``offset``, of positions in nm."""
        sx, sy = self._shift_px()
        return (np.asarray(x_nm, float) / self.pixel_nm + offset[0] + sx,
                np.asarray(y_nm, float) / self.pixel_nm + offset[1] + sy)

    def _rebuild(self) -> None:
        """Mask and classification from the current controls."""
        self.axoplasm = self.result = self.cluster_result = None
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
        self.result = self.cluster_result = None
        self.spectrin_interior = self.anchored = None
        self.anchored_centroids = None
        self.anchored_notes = []
        if self.axoplasm is not None:
            margin = self.spin_margin.value()
            col, row = self._coordinates(self.inputs.loc.x_nm,
                                         self.inputs.loc.y_nm)
            self.result = ax.classify(self.axoplasm, col, row, margin)
            centroids = self.inputs.clusters()
            if centroids is not None:
                c = np.asarray(centroids, float)
                if c.size == 0:
                    c = np.empty((0, 2))
                ccol, crow = self._coordinates(c[:, 0], c[:, 1])
                self.cluster_result = ax.classify(self.axoplasm, ccol, crow,
                                                  margin)
                self._find_anchored(c, margin)
        self._refresh()
        # After drawing: the main window may take seconds to repeat the MPS
        # analysis without the discarded clusters.
        self.inputs.anchored_changed(self.anchored, self.anchored_centroids)

    def _find_anchored(self, centroids: np.ndarray, margin: float) -> None:
        """The clusters both images put inside, and the contour without."""
        if self.reference is None or self.axoplasm is None \
                or len(centroids) < 3:
            return
        offset = self.reference_offset
        col, row = self._coordinates_in(offset, self.inputs.loc.x_nm,
                                        self.inputs.loc.y_nm)
        centre, radius, reach = ax.axon_centre(col, row)
        scol, srow = self._coordinates_in(offset, centroids[:, 0],
                                          centroids[:, 1])
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
            contour_cache=self._contour_cache)

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
        self._manual_threshold = low + (high - low) * position / 1000.0
        self.selection_note = None
        self._rebuild()

    def _threshold_typed(self, value: float) -> None:
        if self._updating or self.axoplasm is None:
            return
        self._manual_threshold = float(value)
        self.selection_note = None
        self._rebuild()

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
            and not (self._thread is not None and self._thread.is_alive()))
        self.btn_back.setEnabled(self.measured is not None)
        self.btn_export.setEnabled(self.result is not None)
        for widget in (self.slider_threshold, self.spin_threshold,
                       self.btn_otsu):
            widget.setEnabled(self.axoplasm is not None)
        self._clear(self.findings)
        messages = self.warnings()
        if self.tubulin is None:
            self.findings.addWidget(_label(
                "Choose the widefield betaIII-tubulin image.", _DIM))
        elif not messages:
            self.findings.addWidget(_label("No warnings.", _OK))
        for text in messages:
            self.findings.addWidget(_label("!  " + text, _WARN))
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
            if self.anchored_notes or self.axoplasm is None:
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
                f"Spectrin interior (yellow): {interior.area_um2:.2f} µm², "
                f"cut at {interior.threshold:.0f} counts, {how} (inside "
                f"{levels['interior']:.0f}, ring {levels['ring']:.0f}).")
        else:
            lines.append("No spectrin interior was found, so no cluster is "
                         "discarded.")
        lines.append(
            f"Discarded (red): {found.n_discarded} of {found.n} clusters, "
            f"inside both images by more than {found.margin_nm:.0f} nm. "
            f"Kept although one image alone puts them inside: "
            f"{found.n_tubulin_only} by the tubulin, {found.n_spectrin_only} "
            f"by the spectrin.")
        parts = []
        same = found.contour_all_starts is found.contour_all
        if found.contour_all is not None:
            parts.append(f"{found.contour_all.perimeter_um:.2f} µm with all "
                         f"clusters, as the MPS analysis connects them"
                         + (" (2-opt from every start; dashed)" if same
                            else " (dashed)"))
        if found.contour_all_starts is not None and not same:
            parts.append(f"{found.contour_all_starts.perimeter_um:.2f} µm with "
                         f"all clusters and 2-opt from every start")
        new = found.contour_anchored
        if new is not None:
            spread = new.start_spread_um
            parts.append(
                f"{new.perimeter_um:.2f} µm without the discarded ones "
                f"(green; 2-opt from all {new.n_starts} starts"
                + (f", which a single start could have missed by up to "
                   f"{spread:.2f} µm" if spread else "") + ")")
        if parts:
            lines.append("Perimeter: " + "; ".join(parts) + ".")
        if new is not None and new.centre is not None:
            c = new.centre
            lines.append(
                f"Centre of the green contour (+, its area centroid): "
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
                 "adjusted by hand"}[reg.source]
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

        result = self.result
        if result is None:
            self.label_result.setText("")
            return
        fraction = result.fraction_interior
        text = (f"{result.count(ax.LABEL_INTERIOR):,} interior and "
                f"{result.count(ax.LABEL_MEMBRANE):,} membrane of "
                f"{result.n:,} localizations"
                + ("" if fraction is None else f" ({fraction:.0%} interior)"))
        outside = result.count(ax.LABEL_OUTSIDE)
        if outside:
            text += f"; {outside:,} outside the analysed region"
        text += "."
        clusters = self.cluster_result
        if clusters is None:
            text += (" Clusters: run 'cluster Ch1' on this selection to "
                     "classify them.")
        elif clusters.n == 0:
            text += " Clusters: the MPS analysis kept none."
        else:
            text += (f" Clusters: {clusters.count(ax.LABEL_INTERIOR)} "
                     f"interior, {clusters.count(ax.LABEL_MEMBRANE)} "
                     f"membrane")
            outside = clusters.count(ax.LABEL_OUTSIDE)
            text += (f", {outside} outside the analysed region."
                     if outside else ".")
        self.label_result.setText(text)

    def _region_rect(self, rows: Tuple[int, int], cols: Tuple[int, int],
                     offset: Tuple[float, float]) -> QtCore.QRectF:
        """Where image pixels [rows) x [cols) lie, in localization nm."""
        sx, sy = self._shift_px()
        px = self.pixel_nm
        return QtCore.QRectF(
            (cols[0] - 0.5 - offset[0] - sx) * px,
            (rows[0] - 0.5 - offset[1] - sy) * px,
            (cols[1] - cols[0]) * px, (rows[1] - rows[0]) * px)

    def _draw(self) -> None:
        mask = self.axoplasm
        for item in (self.membrane_item, self.interior_item,
                     self.cluster_item, self.discarded_item,
                     self.contour_all_item, self.contour_item,
                     self.centre_item):
            item.setData([], [])
        self.spectrin_outline_item.clear()
        if mask is None or self.tubulin is None:
            self.image_item.clear()
            self.outline_item.clear()
            self.plot_hist.clear()
            set_title(self.plot_image, "betaIII-tubulin around the axon")
            return
        r0, r1, c0, c1 = mask.region
        image, offset, title = (
            self.tubulin.image, self.tubulin_offset,
            "betaIII-tubulin around the axon")
        rows, cols = (r0, r1), (c0, c1)
        if self.check_reference.isChecked() and self.reference is not None:
            # The same area, in the reference image's pixels.
            dr = self.reference_offset[1] - self.tubulin_offset[1]
            dc = self.reference_offset[0] - self.tubulin_offset[0]
            height, width = self.reference.shape
            rows = (int(max(0, r0 + dr)), int(min(height, r1 + dr)))
            cols = (int(max(0, c0 + dc)), int(min(width, c1 + dc)))
            image, offset = self.reference.image, self.reference_offset
            title = "betaII-spectrin widefield around the axon"
        region = image[rows[0]:rows[1], cols[0]:cols[1]]
        set_title(self.plot_image, title)
        if region.size:
            low, high = np.percentile(region, (1, 99.7))
            self.image_item.setImage(region.T, levels=(low, max(high, low + 1)))
            self.image_item.setRect(self._region_rect(rows, cols, offset))
        else:
            self.image_item.clear()

        edge = mask.mask & ~ndimage.binary_erosion(mask.mask)
        rgba = np.zeros(edge.shape + (4,), dtype=np.ubyte)
        rgba[edge] = _OUTLINE
        self.outline_item.setImage(np.transpose(rgba, (1, 0, 2)),
                                   levels=(0, 255))
        self.outline_item.setRect(self._region_rect((r0, r1), (c0, c1),
                                                    self.tubulin_offset))

        result = self.result
        loc = self.inputs.loc
        if result is not None:
            rng = np.random.default_rng(0)
            for label, item in ((ax.LABEL_MEMBRANE, self.membrane_item),
                                (ax.LABEL_INTERIOR, self.interior_item)):
                idx = np.nonzero(result.labels == label)[0]
                if idx.size > MAX_DRAWN:
                    idx = rng.choice(idx, MAX_DRAWN, replace=False)
                item.setData(loc.x_nm[idx], loc.y_nm[idx])
        interior = self.spectrin_interior
        if interior is not None and interior.mask.any():
            ring = interior.mask & ~ndimage.binary_erosion(interior.mask)
            yellow = np.zeros(ring.shape + (4,), dtype=np.ubyte)
            yellow[ring] = _SPECTRIN_OUTLINE
            self.spectrin_outline_item.setImage(
                np.transpose(yellow, (1, 0, 2)), levels=(0, 255))
            ir0, ir1, ic0, ic1 = interior.region
            self.spectrin_outline_item.setRect(self._region_rect(
                (ir0, ir1), (ic0, ic1), self.reference_offset))
        found = self.anchored
        if found is not None and self.anchored_centroids is not None:
            c = self.anchored_centroids
            gone = found.discarded
            self.cluster_item.setData(c[~gone, 0], c[~gone, 1])
            self.discarded_item.setData(c[gone, 0], c[gone, 1])
            for item, drawn in ((self.contour_all_item, found.contour_all),
                                (self.contour_item, found.contour_anchored)):
                if drawn is not None:
                    closed = np.vstack([drawn.contour, drawn.contour[:1]])
                    item.setData(closed[:, 0], closed[:, 1])
            new = found.contour_anchored
            if new is not None and new.centre is not None:
                self.centre_item.setData([new.centre.x_nm],
                                         [new.centre.y_nm])
        else:
            centroids = self.inputs.clusters()
            if centroids is not None and len(centroids):
                c = np.asarray(centroids, float)
                self.cluster_item.setData(c[:, 0], c[:, 1])

        self.plot_hist.clear()
        if result is not None:
            d = result.distance_nm[np.isfinite(result.distance_nm)]
            if d.size:
                edges = np.arange(-HIST_RANGE_NM, HIST_RANGE_NM + HIST_BIN_NM,
                                  HIST_BIN_NM)
                counts, _ = np.histogram(np.clip(d, edges[0], edges[-1]),
                                         bins=edges)
                self.plot_hist.plot(edges, counts, stepMode="center",
                                    pen=pg.mkPen(TITLE_FG, width=1.5))
            self.plot_hist.addItem(pg.InfiniteLine(
                pos=0, angle=90, pen=pg.mkPen(_DIM, style=QtCore.Qt.DashLine)))
            self.plot_hist.addItem(pg.InfiniteLine(
                pos=result.margin_nm, angle=90, pen=pg.mkPen(_INTERIOR)))

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
            cluster_result=self.cluster_result,
            spectrin=self.spectrin_interior, anchored=self.anchored)

    def cluster_rows(self) -> List[Dict[str, Any]]:
        assert self.anchored is not None and self.anchored_centroids is not None
        return ax.cluster_rows(
            localizations=str(self.inputs.movie.path),
            roi=describe_roi(self.inputs.roi),
            centroids_nm=self.anchored_centroids, anchored=self.anchored)

    def localization_rows(self) -> List[Dict[str, Any]]:
        assert self.result is not None
        loc = self.inputs.loc
        source = str(self.inputs.movie.path)
        roi = describe_roi(self.inputs.roi)
        return [
            {"source_localizations": source, "roi": roi,
             "x_nm": round(float(x), 2), "y_nm": round(float(y), 2),
             "z_nm": round(float(z), 2),
             "distance_to_edge_nm": (round(float(d), 1) if np.isfinite(d)
                                     else float(d)),
             "label": str(label)}
            for x, y, z, d, label in zip(loc.x_nm, loc.y_nm, loc.z_nm,
                                         self.result.distance_nm,
                                         self.result.labels)
        ]

    def _export_clicked(self) -> None:
        # Not export_csv directly: clicked passes a bool, taken for a path.
        self.export_csv()

    def export_csv(self, path: Optional[str] = None) -> Optional[str]:
        """
        Append this axon to a per-axon table, and its localizations to a
        second table beside it (``<name>_localizations.csv``). When the
        clusters were checked against both images, they go to a third
        (``<name>_clusters.csv``), with which ones were discarded.
        """
        if self.result is None:
            return None
        if path is None:
            stem = os.path.splitext(os.path.basename(
                str(self.inputs.movie.path)))[0]
            suggested = os.path.join(
                os.path.dirname(str(self.inputs.movie.path)),
                f"{stem}_axoplasm.csv")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export the classification", suggested, "CSV (*.csv)")
            if not path:
                return None
        base, ext = os.path.splitext(path)
        detail = f"{base}_localizations{ext or '.csv'}"
        clusters = f"{base}_clusters{ext or '.csv'}"
        lines = []
        try:
            for target, rows in ((path, [self.summary_row()]),
                                 (detail, self.localization_rows())):
                appended = append_rows(target, rows)
                lines.append(f"{'Appended to' if appended else 'Wrote'} "
                             f"{target}")
            if self.anchored is not None:
                appended = append_rows(clusters, self.cluster_rows())
                lines.append(f"{'Appended to' if appended else 'Wrote'} "
                             f"{clusters}")
        except (OSError, ValueError, csv.Error) as error:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not write:\n\n{error}")
            return None
        QtWidgets.QMessageBox.information(self, "Exported", "\n".join(lines))
        return path


def show_axoplasm_window(
    inputs: AxoplasmInputs,
    parent: Optional[QtWidgets.QWidget] = None,
) -> AxoplasmWindow:
    """Open the axoplasm panel."""
    window = AxoplasmWindow(inputs, parent=parent)
    window.show()
    window.raise_()
    window.activateWindow()
    return window
