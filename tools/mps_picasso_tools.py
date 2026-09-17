# -*- coding: utf-8 -*-
"""
Menu actions that run Picasso on the loaded file.

The dialogs speak nanometres, like the rest of MPS Explorer, and
``tools.picasso_cli`` converts. Every run happens off the GUI thread
behind a progress dialog with a Stop button: AIM takes seconds, but G5M
took about a minute on a 19,000-localization sample, and a frozen window
for that long reads as a crash.

Picasso works on FILES, so these actions always use the whole loaded
file, not the ROI -- the dialogs say so.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from tools import mps_io, mps_metadata, picasso_cli
from tools.mps_quality import REPEAT_GAP_FRAMES, repeat_neighbour_fraction
from tools.mps_settings import save_settings

PICASSO_RELEASES = "https://github.com/jungmannlab/picasso/releases"

# On the 15.07.26 sample (5.6 localizations per frame), segments holding 560
# or 1,100 localizations made AIM invent 0.1-1.3 um of drift and scramble
# the data. From about 2,800 on, the result stopped degrading but still
# scattered with the segment length -- by tens of nm in the drift, and by
# 96-100% in the sharpness kept -- and no run was sharper than the
# uncorrected file. Picasso's own default of 100 frames assumes much denser
# data. AIM also needs at least four segments; with two it crashes.
AIM_LOCS_PER_SEGMENT = 3000
# Share of the original's repeat-neighbour fraction a corrected file keeps.
# Measured on that sample: 17% and 79% for the scrambling runs, 94% for
# the mildly damaged ones, 96-100% for the rest.
UNDRIFT_SCRAMBLED_RATIO = 0.90
UNDRIFT_KEEP_RATIO = 0.98
UNDRIFT_SHARPER_RATIO = 1.02


def default_segmentation(n_locs: int, n_frames: Optional[int]) -> int:
    """Frames per AIM segment giving about AIM_LOCS_PER_SEGMENT each."""
    if not n_frames:
        return 100
    per_frame = max(n_locs / n_frames, 1e-9)
    frames = max(100, math.ceil(AIM_LOCS_PER_SEGMENT / per_frame / 50) * 50)
    # AIM needs at least four time points.
    return int(min(frames, max(n_frames // 4, 10)))


@dataclass
class AimOutcome:
    run: picasso_cli.PicassoRun
    # Repeat-neighbour fraction before and after, or None if unchecked:
    # laterally, and in 3D for a 3D file.
    lateral: Optional[Tuple[float, float]]
    axial: Optional[Tuple[float, float]] = None


def _kept(pair: Optional[Tuple[float, float]]) -> Optional[float]:
    """after / before, or None when the pair cannot be compared."""
    if pair is None:
        return None
    before, after = pair
    if not (np.isfinite(before) and np.isfinite(after) and before > 0):
        return None
    return after / before


def _twice_median(values: Optional[np.ndarray]) -> Optional[float]:
    """2x the median of the plausible values, or None."""
    if values is None:
        return None
    good = mps_io.plausible_precision(values)
    if not good.any():
        return None
    return 2.0 * float(np.median(np.asarray(values, dtype=float)[good]))


# Steps after which Picasso's "group" column holds cluster labels, and the
# step after which it holds pick numbers instead.
_CLUSTERING_STEPS = ("SMLM clusterer", "DBSCAN", "HDBSCAN")


def groups_are_clusters(steps: Sequence[str]) -> bool:
    """Whether the latest step that wrote "group" was a clustering."""
    for step in reversed(list(steps)):
        if any(name in step for name in _CLUSTERING_STEPS):
            return True
        if "Pick" in step:
            return False
    return False


# ===================================================================
#  A small parameter form
# ===================================================================
@dataclass
class Field:
    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    decimals: int = 0
    suffix: str = ""
    tooltip: str = ""


@dataclass
class Check:
    key: str
    label: str
    default: bool
    tooltip: str = ""


class ParamsDialog(QtWidgets.QDialog):
    """A form of numbers and switches, plus an optional file."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        title: str,
        intro: str,
        fields: Sequence[Field],
        checks: Sequence[Check] = (),
        file_label: Optional[str] = None,
        file_filter: str = "All files (*)",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        layout = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QLabel(intro)
        text.setWordWrap(True)
        layout.addWidget(text)

        form = QtWidgets.QFormLayout()
        self._numbers: Dict[str, QtWidgets.QAbstractSpinBox] = {}
        for spec in fields:
            box: QtWidgets.QAbstractSpinBox
            if spec.decimals:
                box = QtWidgets.QDoubleSpinBox()
                box.setDecimals(spec.decimals)
                box.setRange(spec.minimum, spec.maximum)
                box.setValue(spec.default)
            else:
                box = QtWidgets.QSpinBox()
                box.setRange(int(spec.minimum), int(spec.maximum))
                box.setValue(int(spec.default))
            box.setSuffix(spec.suffix)
            box.setToolTip(spec.tooltip)
            form.addRow(spec.label, box)
            self._numbers[spec.key] = box

        self._checks: Dict[str, QtWidgets.QCheckBox] = {}
        for check in checks:
            tick = QtWidgets.QCheckBox(check.label)
            tick.setChecked(check.default)
            tick.setToolTip(check.tooltip)
            form.addRow("", tick)
            self._checks[check.key] = tick

        self._file: Optional[QtWidgets.QLineEdit] = None
        if file_label is not None:
            row = QtWidgets.QHBoxLayout()
            self._file = QtWidgets.QLineEdit()
            browse = QtWidgets.QPushButton("Browse...")
            file_edit = self._file

            def pick() -> None:
                chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, file_label, "", file_filter)
                if chosen:
                    file_edit.setText(chosen)

            browse.clicked.connect(pick)
            row.addWidget(self._file)
            row.addWidget(browse)
            form.addRow(file_label, row)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Run")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        if self._file is not None and not self._file.text().strip():
            QtWidgets.QMessageBox.warning(
                self, self.windowTitle(), "A file is required here.")
            return
        self.accept()

    def values(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, box in self._numbers.items():
            if isinstance(box, QtWidgets.QDoubleSpinBox):
                out[key] = float(box.value())
            elif isinstance(box, QtWidgets.QSpinBox):
                out[key] = int(box.value())
        for key, tick in self._checks.items():
            out[key] = tick.isChecked()
        return out

    def file(self) -> Optional[str]:
        return self._file.text().strip() if self._file is not None else None


# ===================================================================
#  One run on a worker thread
# ===================================================================
class _Relay(QtCore.QObject):
    # Emitted from the worker thread; Qt queues it to the GUI thread.
    finished = QtCore.pyqtSignal(object, object)


class PicassoJob:
    """Run ``work`` off the GUI thread behind a progress dialog."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        title: str,
        work: Callable[["PicassoJob"], Any],
        on_done: Callable[[Any, Optional[BaseException]], None],
    ) -> None:
        self._work = work
        self.stop_requested = threading.Event()
        # Set by the worker; read by the timer in the GUI thread.
        self.phase = "Picasso is running"
        self._on_done = on_done
        self._title = title
        self._started = 0.0
        self.relay = _Relay()
        self.relay.finished.connect(self._finish)

        self.dialog = QtWidgets.QProgressDialog(
            "", "Stop", 0, 0, parent)
        self.dialog.setWindowTitle(title)
        self.dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.dialog.setMinimumDuration(0)
        self.dialog.setAutoClose(False)
        self.dialog.setAutoReset(False)
        self.dialog.canceled.connect(self._stop)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        self.thread = threading.Thread(target=self._run, daemon=True)

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def start(self) -> None:
        self._started = time.time()
        self._tick()
        self.dialog.show()
        self.timer.start()
        self.thread.start()

    def _tick(self) -> None:
        if self.stop_requested.is_set():
            # Stop may have been pressed before the process existed.
            picasso_cli.cancel_running()
        seconds = int(time.time() - self._started)
        self.dialog.setLabelText(
            f"{self._title}: {self.phase} ({seconds // 60}:"
            f"{seconds % 60:02d}).\nThis can take minutes; Stop ends it.")

    def _stop(self) -> None:
        self.stop_requested.set()
        picasso_cli.cancel_running()

    def _run(self) -> None:
        try:
            result, error = self._work(self), None
        except BaseException as exc:  # noqa: BLE001 - handed to the GUI
            result, error = None, exc
        self.relay.finished.emit(result, error)

    def _finish(self, result: Any, error: Optional[BaseException]) -> None:
        self.timer.stop()
        self.dialog.hide()
        self.dialog.deleteLater()
        self._on_done(result, error)


# ===================================================================
#  The actions
# ===================================================================
class PicassoTools:
    """
    The Analysis > Picasso tools menu.

    ``window`` is the main MPS Explorer window. It is expected to provide
    ``locs1`` (the loaded ``Localizations`` or None), ``mps_settings``,
    ``logger`` and ``load_channel1(path)``.
    """

    def __init__(self, window: Any) -> None:
        self.window = window
        self.job: Optional[PicassoJob] = None
        self._install: Optional[picasso_cli.PicassoInstall] = None

    # ------------------------------------------------------------ plumbing
    def _info(self, title: str, text: str) -> None:
        QtWidgets.QMessageBox.information(self.window, title, text)

    def _not_found_text(self, configured: Optional[str]) -> str:
        if configured:
            head = (f"The configured Picasso executable did not answer as "
                    f"Picasso:\n{configured}")
        else:
            head = ("No Picasso executable was found on the PATH or in the "
                    "usual install folders.")
        return (f"{head}\n\nPoint at picasso.exe with Analysis > Picasso "
                f"tools > Picasso location..., or install Picasso from "
                f"{PICASSO_RELEASES}. Everything else in MPS Explorer works "
                f"without it.")

    def _run(
        self,
        title: str,
        call: Callable[[picasso_cli.PicassoInstall, PicassoJob], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        if self.job is not None and self.job.running:
            self._info(title, "A Picasso run is already in progress.")
            return
        configured = self.window.mps_settings.picasso_path or None
        known = self._install

        def work(job: PicassoJob) -> Any:
            install = known or picasso_cli.find_picasso(explicit=configured)
            if install is None:
                raise picasso_cli.PicassoNotFound(
                    self._not_found_text(configured))
            if job.stop_requested.is_set():
                raise picasso_cli.PicassoCancelled(
                    f"{title} was stopped before Picasso started.")
            return install, call(install, job)

        def done(result: Any, error: Optional[BaseException]) -> None:
            self.job = None
            logger = self.window.logger
            if error is None:
                self._install, outcome = result
                run = (outcome if isinstance(outcome, picasso_cli.PicassoRun)
                       else getattr(outcome, "run", None))
                if run is not None:
                    logger.info(
                        f"picasso {run.command} finished in "
                        f"{run.seconds:.0f} s: {run.outputs}")
                    logger.debug(f"picasso stdout:\n{run.stdout[-4000:]}")
                on_success(outcome)
            elif isinstance(error, picasso_cli.PicassoCancelled):
                logger.info(str(error))
                self._info(title, str(error))
            elif isinstance(error, InterruptedError):
                text = ("Stopped while checking Picasso's result. The "
                        "output was left next to the input, unchecked.")
                logger.info(f"{title}: {text}")
                self._info(title, text)
            elif isinstance(error, picasso_cli.PicassoNotFound):
                logger.warning(str(error))
                QtWidgets.QMessageBox.warning(self.window, title, str(error))
            elif isinstance(error, (picasso_cli.PicassoFailed, ValueError)):
                logger.error(f"{title}: {error}")
                QtWidgets.QMessageBox.critical(self.window, title, str(error))
            else:
                logger.error(f"{title} failed", exc_info=error)
                QtWidgets.QMessageBox.critical(
                    self.window, title, f"Unexpected error:\n{error!r}")

        self.job = PicassoJob(self.window, title, work, done)
        self.job.start()

    def _loaded_hdf5(self, title: str) -> Optional[Any]:
        loc = self.window.locs1
        if loc is None:
            self._info(title, "Load a Picasso HDF5 file into channel 1 first.")
            return None
        if not str(loc.path).lower().endswith(".hdf5"):
            self._info(
                title,
                "Picasso works on its own HDF5 localization files, and the "
                f"loaded file is not one:\n{loc.path}")
            return None
        return loc

    def _offer_load(self, title: str, text: str, path: str) -> None:
        answer = QtWidgets.QMessageBox.question(
            self.window, title,
            f"{text}\n\nLoad {os.path.basename(path)} into channel 1 now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if answer == QtWidgets.QMessageBox.Yes:
            self.window.load_channel1(path)

    def shutdown(self) -> None:
        """Stop a run in progress; called when the main window closes."""
        if self.job is not None and self.job.running:
            self.job.stop_requested.set()
            picasso_cli.cancel_running()
            self.job.thread.join(timeout=10)

    # ------------------------------------------------------------ AIM
    def undrift(self) -> None:
        title = "Undrift with AIM"
        loc = self._loaded_hdf5(title)
        if loc is None:
            return
        n_frames = loc.n_frames
        per_frame = loc.n / n_frames if n_frames else float("nan")
        dialog = ParamsDialog(
            self.window, title,
            f"Runs Picasso's AIM on the whole of {os.path.basename(loc.path)}"
            f", not just the ROI. AIM corrects z as well as x and y, which "
            f"RCC never does. The result is written next to the input as "
            f"*_aim.hdf5, with the drift trace as *_aimdrift.txt, and is "
            f"checked against the original before it is offered.",
            [
                Field("segmentation", "Frames per segment",
                      default_segmentation(loc.n, n_frames), 10, 1000000,
                      suffix=" frames",
                      tooltip=f"Frames pooled into one time point, chosen "
                      f"so each holds about {AIM_LOCS_PER_SEGMENT:,} "
                      f"localizations (this file has {per_frame:.1f} per "
                      f"frame). With too few, AIM invents drift and "
                      f"scrambles the data; Picasso's default of 100 frames "
                      f"assumes much denser data."),
                Field("intersect_distance_nm", "Intersection distance", 20.0,
                      1.0, 1000.0, 1, " nm",
                      "How close localizations in consecutive segments must "
                      "be to count as the same site. Picasso's default."),
                Field("max_drift_nm", "Largest drift between segments", 60.0,
                      1.0, 10000.0, 1, " nm",
                      "Search radius for the shift from one segment to the "
                      "next. Picasso's default."),
            ])
        if not dialog.exec_():
            return
        values = dialog.values()
        if n_frames and n_frames / values["segmentation"] < 4:
            self._info(title, (
                f"{n_frames} frames in segments of {values['segmentation']} "
                f"give fewer than 4 time points; AIM cannot follow drift with "
                f"that. Use shorter segments."))
            return
        path, pixel = loc.path, loc.pixel_size_nm
        per_segment = per_frame * values["segmentation"]
        radius = _twice_median(loc.lp_lateral_nm)
        radius_z = _twice_median(loc.lpz_nm) if loc.is_3d else None
        z_unchecked = loc.is_3d and radius_z is None
        frame, x, y, z = loc.frame, loc.x_nm, loc.y_nm, loc.z_nm

        def call(inst: picasso_cli.PicassoInstall,
                 job: PicassoJob) -> AimOutcome:
            run = picasso_cli.undrift_aim(
                path, pixel_size_nm=pixel, install=inst, **values)
            if radius is None or frame is None:
                return AimOutcome(run, None)
            job.phase = "checking the corrected file"
            stop = job.stop_requested.is_set
            fixed = mps_io.load_localizations(
                run.outputs["locs"], pixel_size_nm=pixel)
            lateral = (
                repeat_neighbour_fraction(frame, x, y, radius, stop=stop),
                repeat_neighbour_fraction(fixed.frame, fixed.x_nm,
                                          fixed.y_nm, radius, stop=stop),
            )
            axial = None
            if radius_z is not None and fixed.is_3d:
                scale = radius / radius_z
                axial = (
                    repeat_neighbour_fraction(frame, x, y, radius,
                                              z_scaled=z * scale, stop=stop),
                    repeat_neighbour_fraction(
                        fixed.frame, fixed.x_nm, fixed.y_nm, radius,
                        z_scaled=fixed.z_nm * scale, stop=stop),
                )
            return AimOutcome(run, lateral, axial)

        def after(outcome: AimOutcome) -> None:
            run = outcome.run
            drift = np.loadtxt(run.outputs["drift"], ndmin=2)
            span = np.ptp(drift, axis=0)
            text = (f"Drift reported over the acquisition: "
                    f"{span[0] * pixel:.1f} nm in x, "
                    f"{span[1] * pixel:.1f} nm in y")
            if drift.shape[1] > 2:
                text += f", {span[2]:.1f} nm in z"   # Picasso writes z in nm
            text += f" (peak to peak), in {run.seconds:.0f} s."

            ratios: List[float] = []
            for label, pair in (
                (f"within {radius or 0:.0f} nm", outcome.lateral),
                (f"within {radius or 0:.0f} nm laterally and "
                 f"{radius_z or 0:.0f} nm axially", outcome.axial),
            ):
                kept = _kept(pair)
                if pair is None or kept is None:
                    continue
                ratios.append(kept)
                text += (f"\n\nLocalizations revisited {label} at least "
                         f"{REPEAT_GAP_FRAMES} frames later: {pair[0]:.1%} "
                         f"before, {pair[1]:.1%} after.")
                self.window.logger.info(
                    f"AIM on {path}: repeat-neighbour fraction {label}: "
                    f"{pair[0]:.3f} -> {pair[1]:.3f}")
            if z_unchecked:
                text += ("\n\nz could not be checked (no usable axial "
                         "precision), so an axial scramble would go unseen.")

            if not ratios:
                text += ("\n\nThe result could not be checked: the file has "
                         "no usable precision or frame columns.")
            else:
                worst = min(ratios)
                if worst < UNDRIFT_SCRAMBLED_RATIO:
                    QtWidgets.QMessageBox.warning(self.window, title, (
                        f"{text}\n\nThe correction scrambled the data "
                        f"instead of sharpening it. AIM most likely had too "
                        f"few localizations per segment ({per_segment:,.0f} "
                        f"here): run it again with longer segments. Do not "
                        f"analyse {os.path.basename(run.outputs['locs'])}; "
                        f"it was left next to the input."))
                    return
                if worst < UNDRIFT_KEEP_RATIO:
                    text += ("\n\nThe corrected file is somewhat LESS sharp "
                             "than the original: the drift estimate added "
                             "more error than it removed. On data this "
                             "sparse the original may be the better file.")
                elif worst > UNDRIFT_SHARPER_RATIO:
                    text += "\n\nThe corrected file is sharper."
                else:
                    text += ("\n\nNo change in sharpness: either there was "
                             "little drift to remove, or it was removed "
                             "without harm.")
            text += (f"\n\nWritten next to the input:\n"
                     f"{os.path.basename(run.outputs['locs'])}\n"
                     f"{os.path.basename(run.outputs['drift'])}")
            self._offer_load(title, text, run.outputs["locs"])

        self._run(title, call, after)

    # ------------------------------------------------------------ SMLM
    def cluster(self) -> None:
        title = "SMLM clustering"
        loc = self._loaded_hdf5(title)
        if loc is None:
            return
        lp = loc.lp_lateral_nm
        lp_xy = float(np.nanmedian(lp)) if lp is not None else float("nan")
        lpz = loc.lpz_nm
        lp_z = float(np.nanmedian(lpz)) if lpz is not None else float("nan")
        xy_default = round(2 * lp_xy) if np.isfinite(lp_xy) else 20
        fields = [
            Field("radius_nm", "Lateral radius", xy_default, 1.0, 1000.0, 1,
                  " nm", "Twice the median lateral precision by default."),
        ]
        if loc.is_3d:
            z_default = round(2 * lp_z) if np.isfinite(lp_z) else 100
            fields.append(Field(
                "radius_z_nm", "Axial radius", z_default, 1.0, 5000.0, 1,
                " nm", "Twice the median axial precision by default. It is "
                "larger than the lateral one because z is measured worse."))
        fields.append(Field("min_locs", "Minimum localizations", 10, 2,
                            100000, tooltip="Smaller groups are noise."))
        precision = (f"median precision {lp_xy:.1f} nm laterally"
                     if np.isfinite(lp_xy) else "no precision columns")
        if loc.is_3d and np.isfinite(lp_z):
            precision += f", {lp_z:.1f} nm axially"
        dialog = ParamsDialog(
            self.window, title,
            f"Runs Picasso's SMLM clusterer on the whole of "
            f"{os.path.basename(loc.path)} ({precision}). "
            f"{'Lateral and axial radii are separate.' if loc.is_3d else ''} "
            f"The result is written next to the input as *_clusters.hdf5 "
            f"and *_cluster_centers.hdf5.",
            fields,
            [Check("frame_analysis", "Remove sticking (Picasso's basic frame "
                   "analysis)", False,
                   "Drops clusters whose localizations are confined to a "
                   "short stretch of the movie -- the same rule as the "
                   "DNA-PAINT panel's sticking filter.")])
        if not dialog.exec_():
            return
        values = dialog.values()
        path, pixel, n_total = loc.path, loc.pixel_size_nm, loc.n

        def after(run: picasso_cli.PicassoRun) -> None:
            import h5py

            with h5py.File(run.outputs["centers"], "r") as handle:
                n_clusters = len(handle["locs"])
            with h5py.File(run.outputs["locs"], "r") as handle:
                n_in = len(handle["locs"])
            text = (f"{n_clusters:,} clusters holding {n_in:,} of "
                    f"{n_total:,} localizations "
                    f"({100 * n_in / max(n_total, 1):.0f} %), in "
                    f"{run.seconds:.0f} s.")
            box = QtWidgets.QMessageBox(self.window)
            box.setWindowTitle(title)
            box.setText(text)
            g5m = box.addButton("Map molecules (G5M)...",
                                QtWidgets.QMessageBox.AcceptRole)
            load = box.addButton("Load into channel 1",
                                 QtWidgets.QMessageBox.ActionRole)
            box.addButton(QtWidgets.QMessageBox.Close)
            box.exec_()
            if box.clickedButton() is g5m:
                self.map_molecules(run.outputs["locs"])
            elif box.clickedButton() is load:
                self.window.load_channel1(run.outputs["locs"])

        self._run(title, lambda inst, job: picasso_cli.smlm_cluster(
            path, pixel_size_nm=pixel, install=inst, **values), after)

    # ------------------------------------------------------------ G5M
    def _clustered_input(self, title: str) -> Optional[str]:
        loc = self.window.locs1
        if loc is not None and str(loc.path).lower().endswith(".hdf5"):
            if loc.has("group") and groups_are_clusters(loc.processing_steps):
                return str(loc.path)
            beside = str(os.path.splitext(loc.path)[0]) + "_clusters.hdf5"
            if os.path.exists(beside):
                answer = QtWidgets.QMessageBox.question(
                    self.window, title,
                    f"G5M needs clustered localizations, and the loaded file "
                    f"is not clustered (any groups it has are picks). Use the "
                    f"clustering saved next to it?\n\n"
                    f"{os.path.basename(beside)}",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes)
                if answer == QtWidgets.QMessageBox.Yes:
                    return beside
        start = os.path.dirname(loc.path) if loc is not None else ""
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Clustered localizations for G5M", start,
            "Picasso HDF5 (*.hdf5)")
        return str(chosen) if chosen else None

    def map_molecules(self, path: Optional[str] = None) -> None:
        title = "Molecular mapping (G5M)"
        if path is None:
            path = self._clustered_input(title)
            if path is None:
                return
        columns = picasso_cli.hdf5_columns(path)
        if "group" not in columns:
            self._info(title, (
                f"{os.path.basename(path)} has no cluster labels. Cluster it "
                f"first (Picasso tools > SMLM clustering)."))
            return
        info, _ = mps_metadata.load_metadata(path)
        steps = mps_metadata.steps(info)
        if not groups_are_clusters(steps):
            origin = steps[-1] if steps else "an unrecorded step"
            answer = QtWidgets.QMessageBox.question(
                self.window, title,
                f"The groups in {os.path.basename(path)} do not come from a "
                f"clustering (last step: {origin}). G5M fits each group as "
                f"one cluster, so a picked axon would be fitted whole.\n\n"
                f"Run it anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        is_3d = "z" in columns
        dialog = ParamsDialog(
            self.window, title,
            f"Fits each cluster of {os.path.basename(path)} as a mixture of "
            f"molecules whose widths are bounded by the local localization "
            f"precision, so a cloud too wide for one emitter must be split. "
            f"(Kowalewski, Reinhardt et al., Nat Commun 2026.) The result is "
            f"written next to the input as *_molmap.hdf5. This is the slow "
            f"one: about a minute for 160 clusters.",
            [
                Field("min_locs", "Minimum localizations per molecule", 10, 2,
                      100000),
                Field("min_sigma", "Narrowest molecule", 0.8, 0.1, 10.0, 2,
                      " x precision", "Picasso's default."),
                Field("max_sigma", "Widest molecule", 1.5, 0.1, 10.0, 2,
                      " x precision", "Picasso's default."),
            ],
            [Check("postprocess", "Apply Picasso's filters", True,
                   "Keeps molecules with more than 3 binding events, p value "
                   "above 0.015, and frames spread over more than 10% of the "
                   "acquisition. The last is a sticking test that can remove "
                   "almost everything from a short acquisition; switch the "
                   "filters off to see how many molecules each one "
                   "removes.")],
            file_label="Astigmatism calibration" if is_3d else None,
            file_filter="Calibration (*.yaml)")
        if not dialog.exec_():
            return
        values = dialog.values()
        if values["min_sigma"] > values["max_sigma"]:
            self._info(title, "The narrowest molecule cannot be wider than "
                       "the widest.")
            return
        calibration = dialog.file()
        source = path

        def after(run: picasso_cli.PicassoRun) -> None:
            summary = picasso_cli.molmap_summary(run.outputs["molmap"])
            text = "\n".join(summary.lines())
            text += f"\n\nWritten in {run.seconds:.0f} s next to the input."
            self._offer_load(title, text, run.outputs["molmap"])

        self._run(title, lambda inst, job: picasso_cli.molecular_map(
            source, calibration_path=calibration, install=inst, **values),
            after)

    # ------------------------------------------------------------ location
    def locate(self) -> None:
        title = "Picasso location"
        settings = self.window.mps_settings
        current = settings.picasso_path
        if self._install is not None:
            status = picasso_cli.describe(self._install)
        elif current:
            status = f"Configured: {current} (not checked yet)"
        else:
            status = ("Searched automatically on the PATH and in the usual "
                      "install folders (not checked yet).")
        box = QtWidgets.QMessageBox(self.window)
        box.setWindowTitle(title)
        box.setText(status)
        choose = box.addButton("Choose picasso.exe...",
                               QtWidgets.QMessageBox.ActionRole)
        auto = box.addButton("Search automatically",
                             QtWidgets.QMessageBox.ActionRole)
        check = box.addButton("Check now", QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Close)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is choose:
            start = os.path.dirname(current) if current else ""
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.window, "Picasso executable", start,
                "Programs (*.exe);;All files (*)")
            if not chosen:
                return
            settings.picasso_path = chosen
        elif clicked is auto:
            settings.picasso_path = ""
        elif clicked is not check:
            return
        if clicked is not check:
            save_settings(settings)
        self._install = None
        self._run(title, lambda inst, job: inst, lambda inst: self._info(
            title, picasso_cli.describe(inst)))

