# -*- coding: utf-8 -*-
"""
The two-channel panel: register channel B onto channel A, then compare.

For Exchange-PAINT, channel A is betaII-spectrin and channel B the partner
protein (adducin or 4.1B), imaged in a later round. The panel

  1. measures the registration -- from markers present in both rounds,
     from the chromatic calibration file, or both -- and shows it;
  2. moves channel B by the measured shift and only then selects its
     localizations with the channel-1 ROI, so the selection is made in the
     registered frame;
  3. runs the axial phase and the transverse comparison of
     ``tools.mps_crosschannel`` with the registration's errors attached.

Markers are usually outside a picked axon, so they can be taken from other
files -- the full fields of view of the two rounds -- as long as those share
the picked files' coordinates, which Picasso's picks keep.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import dataclasses
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from tools import mps_io
from tools.cluster_quality import (
    CircularROI,
    PolygonROI,
    SquareROI,
    points_in_roi,
)
from tools.mps_analysis import DEFAULT_EPS_NM, DEFAULT_MIN_SAMPLES
from tools.mps_crosschannel import (
    AxialPhaseResult,
    CrossChannelResult,
    axial_phase,
    cross_channel_transverse,
    export_cross_channel,
)
from tools.mps_plot_style import AXIS_FG, TITLE_FG, set_title, style_dark
from tools.mps_registration import (
    NO_REGISTRATION,
    Registration,
    combine,
    read_calibration,
    register_localizations,
)
from tools.results_table import append_rows

_OK = "#5fd75f"
_WARN = "#ffaf5f"
_BAD = "#ff6b6b"
_DIM = "#9a9a9a"
_COLOUR_A = "#6fa8ff"
_COLOUR_B = "#ff9f43"

# Below this many localizations in the ROI a channel is not analysed.
MIN_ROI_LOCALIZATIONS = 50

MARKERS_NONE = "Do not use markers"
MARKERS_LOADED = "Markers in the loaded files"
MARKERS_OTHER = "Markers in other files (full fields of view)"


def _label(text: str, colour: str = TITLE_FG,
           bold: bool = False) -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    widget.setStyleSheet(
        f"color: {colour}; font-weight: {'bold' if bold else 'normal'};")
    widget.setWordWrap(True)
    widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    return widget


def describe_roi(roi: Optional[Any]) -> str:
    """One line naming an ROI shape, for the header and the export."""
    if isinstance(roi, CircularROI):
        return (f"circle centred at ({roi.center_x:.0f}, {roi.center_y:.0f}) "
                f"nm, radius {roi.radius:.0f} nm")
    if isinstance(roi, SquareROI):
        return (f"square x {roi.xmin:.0f}..{roi.xmax:.0f}, "
                f"y {roi.ymin:.0f}..{roi.ymax:.0f} nm")
    if isinstance(roi, PolygonROI):
        return f"polygon of {len(roi.vertices)} vertices"
    return "none"


def clustering_parameters(inputs: "TwoChannelInputs"
                          ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """eps and min samples each channel is clustered with."""
    a = {"eps_nm": inputs.kwargs_a.get("eps_nm", DEFAULT_EPS_NM),
         "min_samples": inputs.kwargs_a.get("min_samples",
                                            DEFAULT_MIN_SAMPLES)}
    # Channel B takes channel A's value for anything it does not set.
    b = {key: inputs.kwargs_b.get(key, value) for key, value in a.items()}
    return a, b


def describe_parameters(inputs: "TwoChannelInputs") -> str:
    """One line with both channels' clustering parameters."""
    a, b = clustering_parameters(inputs)
    return "; ".join(
        f"channel {n}: eps {p['eps_nm']:g} nm, min samples {p['min_samples']}"
        for n, p in (("1", a), ("2", b)))


def _same_pixel(a: Optional[float], b: Optional[float]) -> bool:
    return not (a and b) or abs(float(a) - float(b)) < 1e-6


def _marker_file(path: str, channel: Any, round_name: str
                 ) -> Tuple[Any, Optional[str]]:
    """
    Load a marker file for a round, in that round's pixel scale.

    A shift measured in one scale and applied in another is wrong by the
    ratio times the distance from the camera origin -- 93 nm in a test at
    122 vs 130 nm -- so a disagreement is refused, and a file with no
    pixel size takes the round's.
    """
    name = os.path.basename(path)
    note = None
    try:
        loc = mps_io.load_localizations(path)
    except ValueError as error:
        if "pixel size" not in str(error).lower() or \
                not channel.pixel_size_nm:
            raise
        loc = mps_io.load_localizations(
            path, pixel_size_nm=channel.pixel_size_nm)
        note = (f"{name} records no pixel size; the {round_name} file's "
                f"({channel.pixel_size_nm:g} nm) was used.")
    if not _same_pixel(loc.pixel_size_nm, channel.pixel_size_nm):
        raise ValueError(
            f"The marker file {name} gives a pixel size of "
            f"{loc.pixel_size_nm:g} nm and the {round_name} file "
            f"{channel.pixel_size_nm:g} nm. A shift measured in one scale "
            f"cannot be applied in the other; correct the metadata first.")
    return loc, note


@dataclass
class TwoChannelInputs:
    """Everything the panel needs from the main window."""

    loc_a: Any                       # tools.mps_io.Localizations, channel 1
    loc_b: Any                       # channel 2
    roi: Optional[Any]               # tools.cluster_quality ROI shape
    slab: Optional[Tuple[float, float]]
    slab_half_width_nm: float
    kwargs_a: Dict[str, Any] = field(default_factory=dict)
    kwargs_b: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TwoChannelOutcome:
    registration: Registration
    n_a: int = 0
    n_b: int = 0
    z_a: Optional[np.ndarray] = None
    z_b: Optional[np.ndarray] = None
    axial: Optional[AxialPhaseResult] = None
    transverse: Optional[CrossChannelResult] = None
    notes: List[str] = field(default_factory=list)
    # eps and min samples the channels were clustered with.
    parameters_a: Dict[str, Any] = field(default_factory=dict)
    parameters_b: Dict[str, Any] = field(default_factory=dict)


def measure_registration(
    inputs: TwoChannelInputs,
    markers: str,
    marker_paths: Tuple[str, str],
    calibration_path: str,
) -> Registration:
    """The registration the panel's controls ask for."""
    fiducial: Optional[Registration] = None
    if markers == MARKERS_LOADED:
        fiducial = register_localizations(inputs.loc_a, inputs.loc_b)
    elif markers == MARKERS_OTHER:
        path_a, path_b = marker_paths
        if not path_a or not path_b:
            raise ValueError("Choose the two files that hold the markers.")
        marker_a, note_a = _marker_file(path_a, inputs.loc_a, "channel-1")
        marker_b, note_b = _marker_file(path_b, inputs.loc_b, "channel-2")
        fiducial = register_localizations(marker_a, marker_b)
        fiducial.warnings.extend(n for n in (note_a, note_b) if n)
    calibration: Optional[Registration] = None
    if calibration_path:
        calibration = read_calibration(
            calibration_path, pixel_size_nm=inputs.loc_b.pixel_size_nm)
    if fiducial is not None and calibration is not None:
        return combine(fiducial, calibration)
    if fiducial is not None:
        return fiducial
    if calibration is not None:
        return calibration
    return NO_REGISTRATION


def run_two_channels(
    inputs: TwoChannelInputs,
    registration: Registration,
    progress: Callable[[str], None] = lambda text: None,
) -> TwoChannelOutcome:
    """Register channel B, select both channels with the ROI, compare."""
    out = TwoChannelOutcome(registration=registration)
    out.parameters_a, out.parameters_b = clustering_parameters(inputs)
    a, b = inputs.loc_a, inputs.loc_b
    xa, ya, za = a.x_nm, a.y_nm, a.z_nm
    xb, yb, zb = registration.apply(b.x_nm, b.y_nm, b.z_nm)
    pixels_differ = not _same_pixel(a.pixel_size_nm, b.pixel_size_nm)
    if pixels_differ:
        out.notes.append(
            f"The two files give different pixel sizes ({a.pixel_size_nm:g} "
            f"and {b.pixel_size_nm:g} nm). Exchange-PAINT rounds share the "
            f"camera, so one is wrong, and it scales channel 2 about the "
            f"camera origin: no shift can correct that.")
    if inputs.roi is None:
        out.notes.append(
            "No ROI is selected on channel 1: only the registration was "
            "measured. Draw the scatter plot and select an axon to compare "
            "the channels.")
        return out
    sel_a = points_in_roi(xa, ya, inputs.roi)
    sel_b = points_in_roi(xb, yb, inputs.roi)
    out.n_a, out.n_b = int(sel_a.sum()), int(sel_b.sum())
    out.z_a, out.z_b = za[sel_a], zb[sel_b]
    for name, n in (("channel 1", out.n_a), ("channel 2", out.n_b)):
        if n < MIN_ROI_LOCALIZATIONS:
            hint = ""
            if name == "channel 2" and pixels_differ:
                hint = (" The pixel-size disagreement above moves channel 2 "
                        "as a whole.")
            elif name == "channel 2":
                hint = (" Check the registration: a wrong shift moves "
                        "channel 2 out of the ROI."
                        if registration.shifts_channel_b else
                        " Channel 2 is used as loaded: if the rounds are "
                        "offset, it lies elsewhere. Register them.")
            out.notes.append(
                f"The ROI holds {n} localizations of {name} (at least "
                f"{MIN_ROI_LOCALIZATIONS} are needed): the channels were not "
                f"compared.{hint}")
    if out.n_a < MIN_ROI_LOCALIZATIONS or out.n_b < MIN_ROI_LOCALIZATIONS:
        return out
    slab = inputs.slab
    if a.is_3d and b.is_3d:
        progress("Fitting the axial periodicity of both channels")
        out.axial = axial_phase(out.z_a, out.z_b, registration=registration)
    else:
        # A slab found on one channel would cut the other's placeholder
        # zeros arbitrarily; both are compared whole instead.
        both = np.concatenate([out.z_a, out.z_b])
        slab = (float(both.min()) - 1.0, float(both.max()) + 1.0)
        out.notes.append(
            f"Channel {'2' if a.is_3d else '1'} has no z: the axial phase "
            f"was skipped and the channels were compared as 2D projections, "
            f"without an axial slab.")
    progress("Clustering and comparing the channels")
    out.transverse = cross_channel_transverse(
        xa[sel_a], ya[sel_a], za[sel_a], xb[sel_b], yb[sel_b], zb[sel_b],
        slab=slab, slab_half_width_nm=inputs.slab_half_width_nm,
        registration=registration, analyze_kwargs_b=inputs.kwargs_b,
        **inputs.kwargs_a)
    return out


class _Relay(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, int)
    step = QtCore.pyqtSignal(str, int)


class TwoChannelWindow(QtWidgets.QMainWindow):
    """Registration, axial phase and transverse comparison of two channels."""

    def __init__(self, inputs: TwoChannelInputs,
                 parent: Optional[QtWidgets.QWidget] = None,
                 parameters: Optional[Callable[
                     [], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
                 ) -> None:
        super().__init__(parent)
        self.inputs = inputs
        # Reads the clustering parameters typed in the main window, so a
        # run uses the values shown there when it starts.
        self._parameters = parameters
        self.outcome: Optional[TwoChannelOutcome] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_generation = -1
        self._relay = _Relay()
        self._relay.finished.connect(self._finished)
        self._relay.step.connect(self._set_step)
        self._progress: Optional[QtWidgets.QProgressDialog] = None
        # Bumped whenever the inputs change or the panel closes; a run
        # started under an older value is discarded when it ends.
        self._generation = 0

        self.setWindowTitle(
            "Two channels - " + os.path.basename(str(inputs.loc_a.path))
            + " + " + os.path.basename(str(inputs.loc_b.path)))
        self.resize(1200, 880)
        self.setStyleSheet(
            "QMainWindow { background: #1a1a1a; } "
            f"QLabel, QCheckBox {{ color: {TITLE_FG}; }}")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.addWidget(self._build_header())
        root.addWidget(self._build_controls())

        self.findings = QtWidgets.QVBoxLayout()
        holder = QtWidgets.QWidget()
        holder.setObjectName("findings")
        holder.setStyleSheet("#findings { background: #1a1a1a; }")
        holder.setLayout(self.findings)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(holder)
        scroll.setMaximumHeight(170)
        root.addWidget(scroll)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid #333; }} "
            f"QTabBar::tab {{ background: #262626; color: {AXIS_FG}; "
            f"padding: 6px 14px; }} "
            f"QTabBar::tab:selected {{ background: #3a3a3a; color: #fff; }}")
        root.addWidget(self.tabs, stretch=1)
        self.layout_registration = self._add_tab("Registration")
        self.layout_axial = self._add_tab("Axial phase")
        self.layout_transverse = self._add_tab("Transverse")

        self._show_placeholder()

    # ------------------------------------------------------------ layout
    def _add_tab(self, name: str) -> QtWidgets.QVBoxLayout:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.tabs.addTab(page, name)
        return layout

    def _build_header(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        a, b = self.inputs.loc_a, self.inputs.loc_b
        for tag, loc, colour in (("Channel 1 (A)", a, _COLOUR_A),
                                 ("Channel 2 (B)", b, _COLOUR_B)):
            lay.addWidget(_label(
                f"{tag}: {os.path.basename(str(loc.path))} -- "
                f"{loc.n:,} localizations, pixel {loc.pixel_size_nm} nm "
                f"({loc.pixel_size_source}), "
                f"{'3D' if loc.is_3d else '2D'}", colour))
        self.label_selection = _label("")
        lay.addWidget(self.label_selection)
        self._show_selection()
        return box

    def _show_selection(self) -> None:
        roi, slab = self.inputs.roi, self.inputs.slab
        if roi is None:
            self.label_selection.setText(
                "No ROI applied on channel 1: the panel can measure the "
                "registration, but not compare the channels.")
            self.label_selection.setStyleSheet(f"color: {_WARN};")
            return
        axial = ("automatic, around channel 1's main peak" if slab is None
                 else f"{slab[0]:.0f} .. {slab[1]:.0f} nm, set by hand")
        self.label_selection.setText(
            f"ROI: {describe_roi(roi)}. Axial slab: {axial}. Clustering: "
            f"{describe_parameters(self.inputs)}.")
        self.label_selection.setStyleSheet(f"color: {TITLE_FG};")

    def update_selection(self, roi: Optional[Any],
                         slab: Optional[Tuple[float, float]]) -> None:
        """The main window applied another ROI or axial range."""
        kwargs_a = dict(self.inputs.kwargs_a, roi=roi)
        kwargs_b = dict(self.inputs.kwargs_b)
        if "roi" in kwargs_b:
            kwargs_b["roi"] = roi
        self.inputs = dataclasses.replace(
            self.inputs, roi=roi, slab=slab, kwargs_a=kwargs_a,
            kwargs_b=kwargs_b)
        self._generation += 1
        self._close_progress()
        self.btn_run.setEnabled(True)
        self.outcome = None
        self.btn_export.setEnabled(False)
        self._show_selection()
        self._clear(self.findings)
        self.findings.addWidget(_label(
            "The selection changed in the main window: press Run again.",
            _WARN))
        self._show_placeholder()

    def closeEvent(self, event: Any) -> None:
        self._generation += 1
        self._close_progress()
        super().closeEvent(event)

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.hide()
            self._progress.deleteLater()
            self._progress = None

    def _build_controls(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Registration of channel 2 onto channel 1")
        box.setStyleSheet(f"QGroupBox {{ color: {TITLE_FG}; }}")
        grid = QtWidgets.QGridLayout(box)

        self.combo_markers = QtWidgets.QComboBox()
        self.combo_markers.addItems([MARKERS_LOADED, MARKERS_OTHER,
                                     MARKERS_NONE])
        self.combo_markers.setToolTip(
            "Markers (gold nanoparticles, or any spot localized in at least "
            "80% of the frames) seen in both rounds give the shift between "
            "the rounds in x, y and z, and its error. They are usually "
            "outside the picked axon: then use the full fields of view.")
        self.combo_markers.currentTextChanged.connect(self._update_enabled)
        grid.addWidget(_label("Markers:", _DIM), 0, 0)
        grid.addWidget(self.combo_markers, 0, 1, 1, 3)

        self.edit_marker_a = QtWidgets.QLineEdit()
        self.edit_marker_b = QtWidgets.QLineEdit()
        self.btn_marker_a = QtWidgets.QPushButton("Browse...")
        self.btn_marker_b = QtWidgets.QPushButton("Browse...")
        self.btn_marker_a.clicked.connect(
            lambda: self._browse(self.edit_marker_a, "Round 1 field of view",
                                 "Picasso HDF5 (*.hdf5)"))
        self.btn_marker_b.clicked.connect(
            lambda: self._browse(self.edit_marker_b, "Round 2 field of view",
                                 "Picasso HDF5 (*.hdf5)"))
        grid.addWidget(_label("Round 1 file:", _DIM), 1, 0)
        grid.addWidget(self.edit_marker_a, 1, 1)
        grid.addWidget(self.btn_marker_a, 1, 2)
        grid.addWidget(_label("Round 2 file:", _DIM), 2, 0)
        grid.addWidget(self.edit_marker_b, 2, 1)
        grid.addWidget(self.btn_marker_b, 2, 2)

        self.check_calibration = QtWidgets.QCheckBox(
            "Add a chromatic calibration (matriz-transformacion)")
        self.check_calibration.setToolTip(
            "For two colours imaged at once. Exchange-PAINT uses one dye, "
            "so a chromatic calibration does not describe the offset "
            "between its rounds. The matrix is assumed already applied; "
            "only its error is used, added to the markers' in quadrature.")
        self.check_calibration.toggled.connect(self._update_enabled)
        self.edit_calibration = QtWidgets.QLineEdit()
        self.btn_calibration = QtWidgets.QPushButton("Browse...")
        self.btn_calibration.clicked.connect(
            lambda: self._browse(
                self.edit_calibration, "Calibration output",
                "Calibration (*.json *.yaml *.yml)"))
        grid.addWidget(self.check_calibration, 3, 0, 1, 2)
        grid.addWidget(self.edit_calibration, 4, 1)
        grid.addWidget(self.btn_calibration, 4, 2)

        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setToolTip(
            "Measure the registration, move channel 2 by it, select both "
            "channels with the ROI and compare them.")
        self.btn_run.clicked.connect(self.run)
        self.btn_export = QtWidgets.QPushButton("Export CSV...")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_clicked)
        grid.addWidget(self.btn_run, 0, 4)
        grid.addWidget(self.btn_export, 1, 4)
        grid.setColumnStretch(1, 1)
        self._update_enabled()
        return box

    def _update_enabled(self, *_args: Any) -> None:
        other = self.combo_markers.currentText() == MARKERS_OTHER
        for widget in (self.edit_marker_a, self.edit_marker_b,
                       self.btn_marker_a, self.btn_marker_b):
            widget.setEnabled(other)
        calibrated = self.check_calibration.isChecked()
        self.edit_calibration.setEnabled(calibrated)
        self.btn_calibration.setEnabled(calibrated)

    def _browse(self, edit: QtWidgets.QLineEdit, title: str,
                pattern: str) -> None:
        start = os.path.dirname(edit.text() or str(self.inputs.loc_a.path))
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, title, start, pattern)
        if chosen:
            edit.setText(chosen)

    def _clear(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _show_placeholder(self) -> None:
        for layout in (self.layout_registration, self.layout_axial,
                       self.layout_transverse):
            self._clear(layout)
            layout.addWidget(_label(
                "Choose how to register channel 2 and press Run.", _DIM))
            layout.addStretch(1)

    # ------------------------------------------------------------- run
    def run(self) -> None:
        # A run the inputs have since overtaken is discarded when it ends;
        # it does not hold up a new one.
        if (self._thread is not None and self._thread.is_alive()
                and self._thread_generation == self._generation):
            return
        markers = self.combo_markers.currentText()
        paths = (self.edit_marker_a.text().strip(),
                 self.edit_marker_b.text().strip())
        calibration = (self.edit_calibration.text().strip()
                       if self.check_calibration.isChecked() else "")
        if self.check_calibration.isChecked() and not calibration:
            QtWidgets.QMessageBox.warning(
                self, "Two channels", "Choose the calibration file.")
            return
        if markers == MARKERS_OTHER and not all(paths):
            QtWidgets.QMessageBox.warning(
                self, "Two channels",
                "Choose the two files that hold the markers.")
            return
        if self._parameters is not None:
            params_a, params_b = self._parameters()
            self.inputs = dataclasses.replace(
                self.inputs,
                kwargs_a=dict(self.inputs.kwargs_a, **params_a),
                kwargs_b=dict(self.inputs.kwargs_b, **params_b))
            self._show_selection()
        inputs = self.inputs
        generation = self._generation

        def step(text: str) -> None:
            self._relay.step.emit(text, generation)

        def work() -> TwoChannelOutcome:
            step("Measuring the registration")
            registration = measure_registration(inputs, markers, paths,
                                                calibration)
            return run_two_channels(inputs, registration, progress=step)

        def body() -> None:
            try:
                result, error = work(), None
            except Exception as exc:  # noqa: BLE001 - shown to the user
                result, error = None, exc
            self._relay.finished.emit(result, error, generation)

        self._progress = QtWidgets.QProgressDialog("", None, 0, 0, self)
        self._progress.setWindowTitle("Two channels")
        self._progress.setWindowModality(QtCore.Qt.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.setLabelText("Starting")
        self._progress.show()
        self.btn_run.setEnabled(False)
        self._thread = threading.Thread(target=body, daemon=True)
        self._thread_generation = generation
        self._thread.start()

    def _set_step(self, text: str, generation: int) -> None:
        if self._progress is not None and generation == self._generation:
            self._progress.setLabelText(text + "...")

    def _finished(self, outcome: Optional[TwoChannelOutcome],
                  error: Optional[BaseException], generation: int) -> None:
        if generation != self._generation:
            return          # started before the inputs changed or the close
        self._close_progress()
        self.btn_run.setEnabled(True)
        if error is not None or outcome is None:
            QtWidgets.QMessageBox.critical(
                self, "Two channels", f"The comparison failed:\n{error}")
            return
        self.outcome = outcome
        self.btn_export.setEnabled(
            outcome.axial is not None or outcome.transverse is not None)
        self._refresh()

    def wait(self, timeout: float = 600.0) -> None:
        """Block until a run finishes (for scripts and tests)."""
        if self._thread is not None:
            self._thread.join(timeout)
        QtWidgets.QApplication.processEvents()

    # ---------------------------------------------------------- results
    def _refresh(self) -> None:
        outcome = self.outcome
        assert outcome is not None
        self._clear(self.findings)
        messages: List[Tuple[str, str]] = []
        messages += [(n, _WARN) for n in outcome.notes]
        messages += [(w, _WARN) for w in outcome.registration.warnings]
        if outcome.axial is not None:
            messages += [(w, _WARN) for w in outcome.axial.warnings]
        if outcome.transverse is not None:
            messages += [(w, _WARN) for w in outcome.transverse.warnings]
        if not messages:
            messages = [("No warnings.", _OK)]
        for text, colour in messages:
            self.findings.addWidget(_label("!  " + text, colour))
        self._fill_registration()
        self._fill_axial()
        self._fill_transverse()

    def _fill_registration(self) -> None:
        lay = self.layout_registration
        self._clear(lay)
        reg = self.outcome.registration if self.outcome else NO_REGISTRATION
        colour = _BAD if reg.source == "none" else TITLE_FG
        lay.addWidget(_label(reg.description, colour, bold=True))
        dx, dy, dz = reg.shift_nm
        lines = [
            f"Shift applied to channel 2: x {dx:+.1f} nm, y {dy:+.1f} nm, "
            f"z {dz:+.1f} nm",
            "Lateral error: " + (f"{reg.lateral_rms_nm:.1f} nm RMS"
                                 if reg.lateral_rms_nm is not None
                                 else "unknown"),
            "Axial error: " + (f"{reg.axial_rms_nm:.1f} nm RMS"
                               if reg.axial_rms_nm is not None
                               else "unknown"),
        ]
        for text in lines:
            lay.addWidget(_label(text))
        for path in reg.paths:
            lay.addWidget(_label(path, _DIM))
        if reg.pair_residuals_nm:
            lay.addWidget(_label(
                "Each marker's offset from the shift fitted without it "
                "(leave-one-out). The error above is the RMS of these.",
                _DIM))
            table = QtWidgets.QTableWidget(len(reg.pair_residuals_nm), 5)
            table.setHorizontalHeaderLabels(
                ["x [nm]", "y [nm]", "dx [nm]", "dy [nm]", "dz [nm]"])
            for row, ((px, py), (rx, ry, rz)) in enumerate(
                    zip(reg.pair_positions_nm, reg.pair_residuals_nm)):
                for col, value in enumerate((px, py, rx, ry, rz)):
                    item = QtWidgets.QTableWidgetItem(f"{value:,.1f}")
                    item.setTextAlignment(QtCore.Qt.AlignRight
                                          | QtCore.Qt.AlignVCenter)
                    table.setItem(row, col, item)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.Stretch)
            lay.addWidget(table, stretch=1)
        else:
            lay.addStretch(1)

    def _fill_axial(self) -> None:
        lay = self.layout_axial
        self._clear(lay)
        outcome = self.outcome
        axial = outcome.axial if outcome else None
        if axial is None:
            lay.addWidget(_label(
                "The axial phase was not computed; see the notes above.",
                _DIM))
            lay.addStretch(1)
            return
        phase = ("undetermined" if axial.phase_fraction is None
                 else f"{axial.phase_fraction:.2f}")
        if axial.phase_fraction is not None and \
                axial.phase_uncertainty is not None:
            phase += f" +/- {axial.phase_uncertainty:.2f} (registration)"
        lay.addWidget(_label(
            f"Phase {phase}  ({axial.interpretation_hint}; 0 = in phase, "
            f"0.5 = antiphase)", TITLE_FG, bold=True))
        period = ("unknown" if axial.period_used_nm is None
                  else f"{axial.period_used_nm:.0f} nm")
        lay.addWidget(_label(
            f"Main peaks: channel 1 at {axial.peak_a_nm:.1f} nm, channel 2 "
            f"at {axial.peak_b_nm:.1f} nm (offset {axial.offset_nm:+.1f} nm); "
            f"period {period}."))
        plot = pg.PlotWidget()
        style_dark(plot)
        set_title(plot, "Axial distributions in the ROI (channel 2 registered)")
        plot.setLabel("bottom", "z [nm]", color=AXIS_FG)
        plot.setLabel("left", "fraction of localizations", color=AXIS_FG)
        zs = [z for z in (outcome.z_a, outcome.z_b) if z is not None and z.size]
        if zs:
            lo = float(min(np.min(z) for z in zs))
            hi = float(max(np.max(z) for z in zs))
            edges = np.arange(lo, hi + 10.0, 10.0)
            for z, colour, peak in ((outcome.z_a, _COLOUR_A, axial.peak_a_nm),
                                    (outcome.z_b, _COLOUR_B, axial.peak_b_nm)):
                if z is None or not z.size or edges.size < 2:
                    continue
                counts, _ = np.histogram(z, bins=edges)
                plot.plot(edges, counts / max(z.size, 1), stepMode=True,
                          pen=pg.mkPen(colour, width=2))
                plot.addItem(pg.InfiniteLine(
                    pos=peak, angle=90,
                    pen=pg.mkPen(colour, width=1, style=QtCore.Qt.DashLine)))
        lay.addWidget(plot, stretch=1)

    def _fill_transverse(self) -> None:
        lay = self.layout_transverse
        self._clear(lay)
        t = self.outcome.transverse if self.outcome else None
        if t is None:
            lay.addWidget(_label(
                "The channels were not compared; see the notes above.", _DIM))
            lay.addStretch(1)
            return

        def nm(value: Optional[float]) -> str:
            return "-" if value is None else f"{value:.0f} nm"

        rows = [
            f"Slab {t.slab_nm[0]:.0f} .. {t.slab_nm[1]:.0f} nm; clusters: "
            f"{t.n_clusters_a} in channel 1, {t.n_clusters_b} in channel 2.",
            f"Nearest channel-2 cluster from each channel-1 cluster: median "
            f"{nm(t.median_hetero_nn_a_to_b)}; the other way round "
            f"{nm(t.median_hetero_nn_b_to_a)}.",
            f"Channel 2 placed at random in the same annulus: median "
            f"{nm(t.null_median_a_to_b)}"
            + ("" if t.fraction_null_below_measured is None else
               f"; as close or closer in "
               f"{100 * t.fraction_null_below_measured:.0f}% of draws") + ".",
            "Angular: median offset to the nearest other-channel cluster "
            + ("-" if t.angular_nn_median_deg is None
               else f"{t.angular_nn_median_deg:.1f} deg")
            + ("" if t.rotation_fraction_of_spacing is None else
               f"; best rotation {t.best_rotation_deg:.1f} deg "
               f"({t.rotation_fraction_of_spacing:.2f} of the spacing, "
               f"correlation {t.max_correlation:.2f})") + ".",
            f"Median radius: {nm(t.median_radius_a_nm)} (channel 1), "
            f"{nm(t.median_radius_b_nm)} (channel 2).",
            "Lateral registration error: "
            + ("unknown" if t.registration_rms_nm is None
               else f"{t.registration_rms_nm:.1f} nm") + ".",
        ]
        for text in rows:
            lay.addWidget(_label(text))

        plot = pg.PlotWidget()
        style_dark(plot)
        set_title(plot, "Cluster centroids in the slab")
        plot.setAspectLocked(True)
        plot.setLabel("bottom", "x [nm]", color=AXIS_FG)
        plot.setLabel("left", "y [nm]", color=AXIS_FG)
        for analysis, colour in ((t.analysis_a, _COLOUR_A),
                                 (t.analysis_b, _COLOUR_B)):
            if analysis is None or not len(analysis.centroids):
                continue
            c = np.asarray(analysis.centroids)
            plot.addItem(pg.ScatterPlotItem(
                c[:, 0], c[:, 1], size=10, brush=pg.mkBrush(colour),
                pen=pg.mkPen(None)))
        lay.addWidget(plot, stretch=1)

    # ------------------------------------------------------------ export
    def export_row(self) -> Dict[str, Any]:
        assert self.outcome is not None
        row = export_cross_channel(
            self.outcome.axial, self.outcome.transverse,
            source_a=str(self.inputs.loc_a.path),
            source_b=str(self.inputs.loc_b.path))
        row["roi"] = describe_roi(self.inputs.roi)
        row["slab_mode"] = "manual" if self.inputs.slab else "automatic"
        row["n_locs_a"] = self.outcome.n_a
        row["n_locs_b"] = self.outcome.n_b
        for tag, params in (("a", self.outcome.parameters_a),
                            ("b", self.outcome.parameters_b)):
            row[f"eps_{tag}_nm"] = params.get("eps_nm")
            row[f"min_samples_{tag}"] = params.get("min_samples")
        return row

    def _export_clicked(self) -> None:
        # Not export_csv directly: clicked passes a bool, taken for a path.
        self.export_csv()

    def export_csv(self, path: Optional[str] = None) -> Optional[str]:
        if self.outcome is None:
            return None
        if path is None:
            stem_a = os.path.splitext(
                os.path.basename(str(self.inputs.loc_a.path)))[0]
            suggested = os.path.join(
                os.path.dirname(str(self.inputs.loc_a.path)),
                f"{stem_a}_two_channels.csv")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export two-channel results", suggested,
                "CSV (*.csv)")
            if not path:
                return None
        row = self.export_row()
        try:
            # Axons accumulate into one table, as in the other exports.
            append = append_rows(path, [row])
        except (OSError, ValueError, csv.Error) as error:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not write {path}:\n\n{error}")
            return None
        QtWidgets.QMessageBox.information(
            self, "Exported",
            f"{'Appended to' if append else 'Wrote'} {path}")
        return path


def show_two_channel_window(
    inputs: TwoChannelInputs,
    parent: Optional[QtWidgets.QWidget] = None,
    parameters: Optional[Callable[
        [], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
) -> TwoChannelWindow:
    """Open the two-channel panel."""
    window = TwoChannelWindow(inputs, parent=parent, parameters=parameters)
    window.show()
    window.raise_()
    window.activateWindow()
    return window
