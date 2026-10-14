# -*- coding: utf-8 -*-
"""
The Measurement panel of the main window (UI stage 2, design 12.7; SCI-1): what each loaded file says about the
measurement.

One row per loaded channel with its pixel size and the source of that number, in the program's own pixel-size words
(``tools.mps_param_registry.pixel_source_words``, the same dictionary as the MPS analysis window's provenance line);
then the per-measurement z calibration (``tools.mps_zcalibration``): which record describes the z of each 3D file
(its own Picasso record, or a calibration file chosen for its folder or for the file), the magnification factor m
with its source, the calibrated range and the record's checks; whether the file's own record is the same calibration
as a chosen one; and a localized bead stack's z_zero, usable range and lateral shift with depth. With no 3D file
loaded the z rows are hidden.

What it guarantees: it is a display with two inputs. "Calibration record..." chooses a calibration file or types m
(only used when the record carries none); "Bead stack..." chooses a localized bead stack. Those choices are written
to the one store (``MPSSettings.z_calibration_by_folder``) by the main window, which redraws the panel from it; the
panel itself writes nothing. Nothing here changes a z: Picasso already multiplied every z by the record's m.

``measurement_lines`` is pure (no Qt) so the texts are tested without a window.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from tools import mps_param_registry as reg
from tools import mps_zcalibration as zcal
from tools.mps_origin_ui import OriginBadge
from tools.mps_plot_style import verdict

__all__ = ["ChannelFile", "MeasurementPanel", "measurement_lines", "z_record_line", "z_records_line", "TITLE",
           "CALIBRATION_BUTTON", "BEAD_BUTTON", "CALIBRATION_TIP", "BEAD_TIP", "z_line", "z_detail_line",
           "z_detail_lines", "z_match_line",
           "bead_line"]

TITLE = "Measurement"
NO_FILE = "No file loaded."
Z_LINE = "z calibration: - (none read)"
Z_NOTE = "z as fitted by the localization software"
BEAD_LINE = "bead stack: - (not given)"
CALIBRATION_BUTTON = "Calibration record..."
BEAD_BUTTON = "Bead stack..."
CALIBRATION_TIP = (
    "The astigmatism calibration that defines this measurement's z. By default the record the file carries in its "
    "Picasso metadata ('Z Calibration' block or 'Fit 3D' step) is read. Here you can choose a calibration file "
    "(Picasso *_3d_calib.yaml, a localization YAML that carries one, or a ThunderSTORM calibration) for the whole "
    "folder or for this file only, type the magnification factor m when the record has none, or forget the choice.\n\n"
    "m is never assumed: it depends on the microscope, objective, medium and depth (published values range from 0.57 "
    "to 0.85), so it is read from the record; typed, it is labelled 'user' in every export; missing, it is blank.\n\n"
    "Nothing changes a z: Picasso already multiplied every z and lpz by the record's m when it fitted them. The "
    "record gives the nominal calibrated range (Picasso does not store the z_zero it centred on) and the checks of "
    "the record.")
BEAD_TIP = (
    "A localized bead z stack of the calibration (the Picasso HDF5 with frame, x, y, sx, sy; a raw .tif is not "
    "localized here). With the record's step and width curves it gives the real z_zero (so the real calibrated "
    "range), the usable range (steps whose refitted z lies within two axial sigmas of the true z, off the fit bounds, "
    "in order) and the lateral shift with depth (a cubic B-spline of the beads' x and y; Carlini et al. 2015). Shown "
    "and exported; nothing is subtracted from any analysis in this version.")
PANEL_TIP = ("What each loaded file says about the measurement. The pixel size is given at load (from the file's "
             "metadata, or typed once per folder); the z calibration is read from the file's record or chosen here. "
             "Nothing here changes a coordinate.")
PIXEL_TIP = ("The pixel size of this file and where the number comes from. Every lateral distance scales with it and "
             "every area with its square. It is read at load (Picasso YAML or the HDF5 metadata) or typed once for "
             "the folder; this panel only shows it.")
RECORDS_TIP = ("Read from each file's own metadata (Picasso's 'Z Calibration' block or 'Fit 3D' step), and compared "
               "with a chosen calibration file when there is one: the same width curves, step, frames and m, or not.")


@dataclass
class ChannelFile:
    """What the panel shows of one loaded channel: read from its ``Localizations``, never changed."""

    channel: int
    path: str
    pixel_size_nm: Optional[float]
    pixel_source: str
    is_3d: bool
    has_z_record: bool

    @classmethod
    def from_localizations(cls, channel: int, path: str, loc: Any) -> "ChannelFile":
        try:
            record = zcal.record_from_info(getattr(loc, "info", None) or []) is not None \
                or loc.z_calibration is not None
        except Exception:  # noqa: BLE001 - a display: an unreadable record reads as absent
            record = False
        try:
            three_d = bool(loc.is_3d)
        except Exception:  # noqa: BLE001
            three_d = False
        return cls(channel=int(channel), path=str(path or ""),
                   pixel_size_nm=None if loc.pixel_size_nm is None else float(loc.pixel_size_nm),
                   pixel_source=str(getattr(loc, "pixel_size_source", "unknown") or "unknown"),
                   is_3d=three_d, has_z_record=record)


def pixel_line(f: ChannelFile) -> str:
    px = "unknown" if f.pixel_size_nm is None else f"{f.pixel_size_nm:g} nm"
    return f"Channel {f.channel}: pixel size {px} ({reg.pixel_source_words(f.pixel_source)})"


def _empty(state: Optional[zcal.ZCalibrationState]) -> bool:
    return state is None or (state.is_empty and not state.chosen_error and not state.bead_error)


def z_line(state: Optional[zcal.ZCalibrationState], channel: Optional[int] = None) -> str:
    """The z-calibration row: the record in use and m with its source ("[word]" for a blank or typed value)."""
    head = "" if channel is None else f"Channel {channel}: "
    if _empty(state):
        return f"{head}{Z_LINE} [{reg.badge('measurement.z_calibration', None)}] - {Z_NOTE}"
    assert state is not None
    rec = state.record
    if rec is None:
        name = "- (none read)"
    else:
        name = rec.file_name or (f"the file's '{rec.layout}' record" if rec.origin == zcal.ORIGIN_FILE
                                 else rec.layout)
    m = state.magnification
    origin = state.magnification_origin
    words = zcal.SOURCE_WORDS[origin]
    if m is None:
        m_text = f"m = - ({words}) [{reg.badge('measurement.z_magnification', None)}]"
    elif origin == zcal.ORIGIN_USER:
        m_text = f"m = {m:g} ({words}) [{reg.badge('measurement.z_magnification', m)}]"
    else:
        m_text = f"m = {m:g} ({words})"
    return f"{head}z calibration: {name}, {m_text} - {Z_NOTE}, no z correction applied"


def z_match_line(state: Optional[zcal.ZCalibrationState]) -> str:
    """When a calibration file was chosen: whether it is the same calibration as the file's own record, and what
    differs; "" when nothing was chosen."""
    if state is None or state.chosen_record is None:
        return ""
    if state.file_record is None:
        return "Record: the file carries none of its own to compare with"
    diff = zcal.record_differences(state.file_record, state.chosen_record)
    if diff is None:
        return "Record: cannot be compared with the file's (one of them has no width curves)"
    if not diff:
        return "Record: same as the file's"
    return "Record: DIFFERENT from the file's (" + ", ".join(diff) + ")"


def z_detail_lines(state: Optional[zcal.ZCalibrationState]) -> List[str]:
    """What the record in use gives (steps, calibrated range, fit bounds), whether a chosen file matches the file's
    own record, and each of its check warnings on a line of its own; [] when nothing is read."""
    if _empty(state):
        return []
    assert state is not None
    out: List[str] = []
    if state.chosen_error:
        out.append(f"The chosen calibration file could not be read ({state.chosen_error})")
    rec = state.record
    if rec is not None:
        what = [f"Record in use: {rec.origin} ({rec.layout})"]
        if rec.n_steps is not None and rec.step_nm is not None:
            what.append(f"{rec.n_steps} steps of {rec.step_nm:g} nm")
        cols = zcal.export_columns(state)
        lo, hi = cols["z_calibrated_min_nm"], cols["z_calibrated_max_nm"]
        if lo is not None and hi is not None:
            how = "from the bead stack" if cols["z_calibrated_range_origin"] == "bead stack" else \
                "nominal: Picasso does not store the z_zero it centred on"
            what.append(f"calibrated range {lo:+.0f} to {hi:+.0f} nm ({how})")
        else:
            what.append("calibrated range: - (not recorded)")
        m = state.magnification
        if m is not None:
            what.append(f"fit bounds +/-{zcal.FIT_BOUND_NM * m:.0f} nm")
        out.append("; ".join(what))
    match = z_match_line(state)
    if match:
        out.append(match)
    if rec is not None:
        problems = zcal.validate_record(rec)
        if problems:
            out.extend(f"Check: {p}" for p in problems)
        else:
            out.append("Checks: passed")
    return out


def z_detail_line(state: Optional[zcal.ZCalibrationState]) -> str:
    """``z_detail_lines`` joined by newlines; "" when nothing is read."""
    return "\n".join(z_detail_lines(state))


def _is_warning(line: str) -> bool:
    return line.startswith(("Check: ", "Record: DIFFERENT", "The chosen calibration file could not"))


def z_record_words(f: ChannelFile, state: Optional[zcal.ZCalibrationState] = None) -> str:
    """Whether the file carries its own z-calibration record (and whether a chosen one matches it), or is 2D."""
    if not f.is_3d:
        return "2D file: no z"
    if f.has_z_record:
        own = state.file_record if state is not None else None
        name = "" if own is None or not own.file_name else f", {own.file_name}"
        layout = "Z Calibration" if own is None else own.layout
        text = (f"the file carries a Picasso '{layout}' record{name} (read: m and the calibrated range; z is not "
                f"corrected with it)")
        match = None if state is None else state.match
        if match is True:
            text += "; the same calibration as the chosen file"
        elif match is False:
            text += "; NOT the same calibration as the chosen file"
        return text
    return "the file carries no z calibration record"


def z_record_line(f: ChannelFile, state: Optional[zcal.ZCalibrationState] = None) -> str:
    return f"Channel {f.channel}: {z_record_words(f, state)}"


def z_records_line(files: Sequence[ChannelFile],
                   states: Optional[Mapping[int, zcal.ZCalibrationState]] = None) -> str:
    """Every channel's record on one line (the panel's row)."""
    return "; ".join(z_record_line(f, (states or {}).get(f.channel)) for f in files)


def bead_line(state: Optional[zcal.ZCalibrationState], channel: Optional[int] = None) -> str:
    head = "" if channel is None else f"Channel {channel}: "
    if state is None or (state.bead is None and not state.bead_error):
        return f"{head}{BEAD_LINE} [{reg.badge('measurement.bead_stack', None)}]"
    if state.bead is None:
        return (f"{head}bead stack: {state.bead_name}: not read - {state.bead_error} "
                f"[{reg.badge('measurement.bead_stack', None)}]")
    b = state.bead
    parts = [f"{b.n_beads} bead(s) over {b.n_steps_used} steps"]
    if b.z_zero_nm is not None:
        parts.append(f"z_zero {b.z_zero_nm:+.0f} nm")
    if b.calibrated_min_nm is not None and b.calibrated_max_nm is not None:
        parts.append(f"calibrated {b.calibrated_min_nm:+.0f} to {b.calibrated_max_nm:+.0f} nm")
    if b.usable_min_nm is not None and b.usable_max_nm is not None:
        parts.append(f"usable {b.usable_min_nm:+.0f} to {b.usable_max_nm:+.0f} nm")
    if b.shift_x_nm is not None and b.shift_y_nm is not None:
        parts.append(f"lateral shift over the usable range: x {b.shift_x_nm:.0f} nm, y {b.shift_y_nm:.0f} nm "
                     f"(not subtracted)")
    parts.extend(b.notes)
    return f"{head}bead stack: {b.file_name}: " + "; ".join(parts)


def measurement_lines(files: Sequence[ChannelFile],
                      states: Optional[Mapping[int, zcal.ZCalibrationState]] = None) -> List[str]:
    """Every line of the panel, in order (badges as "[word]"): what a copy of the panel gives."""
    if not files:
        return [NO_FILE]
    lines = [pixel_line(f) for f in files]
    three_d = [f for f in files if f.is_3d]
    if not three_d:
        return lines                     # 2D files: no z, so no z calibration to speak of
    states = states or {}
    several = len(three_d) > 1 and any(not _empty(states.get(f.channel)) for f in three_d)
    for f in (three_d if several else three_d[:1]):
        st = states.get(f.channel)
        lines.append(z_line(st, f.channel if several else None))
        lines.extend(z_detail_lines(st))
    lines.append(z_records_line(files, states))
    for f in (three_d if several else three_d[:1]):
        lines.append(bead_line(states.get(f.channel), f.channel if several else None))
    return lines


def _label(text: str = "", tip: str = "", wrap: bool = False) -> QtWidgets.QLabel:
    w = QtWidgets.QLabel(text)
    w.setWordWrap(wrap)
    w.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    if tip:
        w.setToolTip(tip)
    return w


class MeasurementPanel(QtWidgets.QGroupBox):
    """The Measurement panel (12.7). ``set_files([ChannelFile, ...], states)`` redraws it. Its two buttons emit
    ``calibration_requested(action)`` ("choose", "choose_file", "type_m", "forget") and ``bead_requested(action)``
    ("choose", "forget"); the main window acts on them and redraws. The panel itself writes nothing."""

    calibration_requested = QtCore.pyqtSignal(str)
    bead_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(TITLE, parent)
        self.setObjectName("measurement_panel")
        self.setToolTip(PANEL_TIP)
        self._files: List[ChannelFile] = []
        self._states: Dict[int, zcal.ZCalibrationState] = {}
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        self.lbl_pixels = _label(NO_FILE, PIXEL_TIP, wrap=True)
        self.lbl_pixels.setObjectName("measurement_pixel_sizes")
        lay.addWidget(self.lbl_pixels)
        zrow = QtWidgets.QHBoxLayout()
        zrow.setSpacing(4)
        self.lbl_z = _label(Z_LINE, reg.tooltip("measurement.z_calibration", None), wrap=True)
        self.lbl_z.setObjectName("measurement_z_calibration")
        zrow.addWidget(self.lbl_z, 1)
        self.badge_z = OriginBadge("measurement.z_calibration", None)
        zrow.addWidget(self.badge_z)
        self.badge_m = OriginBadge("measurement.z_magnification", None)
        zrow.addWidget(self.badge_m)
        self.lbl_z_note = _label(f"- {Z_NOTE}")
        zrow.addWidget(self.lbl_z_note)
        self.z_row = QtWidgets.QWidget()
        self.z_row.setLayout(zrow)
        zrow.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.z_row)
        self.lbl_detail = _label("", CALIBRATION_TIP, wrap=True)
        self.lbl_detail.setObjectName("measurement_z_detail")
        lay.addWidget(self.lbl_detail)
        self.lbl_records = _label("", wrap=True, tip=RECORDS_TIP)
        self.lbl_records.setObjectName("measurement_z_records")
        lay.addWidget(self.lbl_records)
        brow = QtWidgets.QHBoxLayout()
        brow.setSpacing(4)
        self.btn_calibration = QtWidgets.QPushButton(CALIBRATION_BUTTON)
        self.btn_calibration.setObjectName("button_z_calibration_record")
        self.btn_calibration.setToolTip(CALIBRATION_TIP)
        self.btn_bead = QtWidgets.QPushButton(BEAD_BUTTON)
        self.btn_bead.setObjectName("button_bead_stack")
        self.btn_bead.setToolTip(BEAD_TIP)
        self.calibration_menu = QtWidgets.QMenu(self.btn_calibration)
        self.calibration_actions: Dict[str, QtWidgets.QAction] = {}
        for action, text in (("choose", "Choose a calibration file for this folder..."),
                             ("choose_file", "Choose a calibration file for this file only..."),
                             ("type_m", "Type the magnification factor m (when the record has none)..."),
                             ("forget", "Forget the chosen file and the typed m: read the file's own record")):
            act = self.calibration_menu.addAction(text)
            act.triggered.connect(lambda _=False, a=action: self.calibration_requested.emit(a))
            self.calibration_actions[action] = act
        self.btn_calibration.setMenu(self.calibration_menu)
        self.bead_menu = QtWidgets.QMenu(self.btn_bead)
        self.bead_actions: Dict[str, QtWidgets.QAction] = {}
        for action, text in (("choose", "Choose a localized bead stack (Picasso HDF5)..."),
                             ("forget", "Forget the bead stack")):
            act = self.bead_menu.addAction(text)
            act.triggered.connect(lambda _=False, a=action: self.bead_requested.emit(a))
            self.bead_actions[action] = act
        self.btn_bead.setMenu(self.bead_menu)
        for b in (self.btn_calibration, self.btn_bead):
            b.setEnabled(False)
            brow.addWidget(b)
        self.lbl_bead = _label(BEAD_LINE, reg.tooltip("measurement.bead_stack", None), wrap=True)
        self.lbl_bead.setObjectName("measurement_bead_stack")
        brow.addWidget(self.lbl_bead, 1)
        self.badge_bead = OriginBadge("measurement.bead_stack", None)
        brow.addWidget(self.badge_bead)
        self.bead_row = QtWidgets.QWidget()
        self.bead_row.setLayout(brow)
        brow.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.bead_row)
        self.set_files([])

    # ------------------------------------------------------------------ public
    def files(self) -> List[ChannelFile]:
        return list(self._files)

    def states(self) -> Dict[int, zcal.ZCalibrationState]:
        return dict(self._states)

    def keys_shown(self) -> List[str]:
        """The registry keys this panel shows (the coverage test of 12.3)."""
        return [e.key for e in reg.by_home("measurement")]

    def set_files(self, files: Sequence[ChannelFile],
                  states: Optional[Mapping[int, zcal.ZCalibrationState]] = None) -> None:
        self._files = list(files)
        self._states = dict(states or {})
        # The z rows only when a loaded file has z (m14): a 2D file has no z calibration to speak of.
        three_d = [f for f in self._files if f.is_3d]
        self.z_row.setVisible(bool(three_d) or not self._files)
        self.bead_row.setVisible(bool(three_d) or not self._files)
        # The inputs act on a loaded 3D file: before one, there is no measurement to describe.
        for b in (self.btn_calibration, self.btn_bead):
            b.setEnabled(bool(three_d))
        if not self._files:
            self.lbl_pixels.setText(NO_FILE)
            self.lbl_pixels.setStyleSheet("")
            self.lbl_records.setText("")
            self.lbl_records.setVisible(False)
            self.lbl_detail.setText("")
            self.lbl_detail.setVisible(False)
            self._show_z(None)
            return
        warn = verdict("warn", dark=False)
        parts = []
        for f in self._files:
            text = pixel_line(f)
            if f.pixel_source not in reg.PIXEL_SOURCES_FROM_FILE:
                text = f"<span style='color:{warn}'>{text}</span>"
            parts.append(text)
        self.lbl_pixels.setText("; ".join(parts))
        self.lbl_pixels.setToolTip(PIXEL_TIP + "\n\n" + "\n".join(
            f"Channel {f.channel}: {os.path.basename(f.path) or '(unnamed)'}" for f in self._files))
        self.lbl_records.setText(z_records_line(self._files, self._states))
        self.lbl_records.setVisible(bool(three_d))
        lines = measurement_lines(self._files, self._states)
        z_lines = [ln for ln in lines if "z calibration:" in ln and "carries" not in ln]
        details = [ln for f in three_d for ln in z_detail_lines(self._states.get(f.channel))]
        bead_lines = [ln for ln in lines if "bead stack:" in ln]
        # One line each, a warning in the warning colour: the match verdict and the checks are what the user must
        # see, so none of them sits at the end of a long line (manual test 3).
        self.lbl_detail.setText("<br>".join(
            f"<span style='color:{warn}'>{html.escape(ln, quote=False)}</span>" if _is_warning(ln)
            else html.escape(ln, quote=False) for ln in details))
        self.lbl_detail.setVisible(bool(details))
        self.lbl_detail.setToolTip(CALIBRATION_TIP + ("\n\n" + "\n".join(details) if details else ""))
        self.lbl_z.setToolTip(reg.tooltip("measurement.z_calibration", None)
                              + ("\n\n" + "\n".join(details) if details else ""))
        first = self._states.get(three_d[0].channel) if three_d else None
        self._show_z(first, z_lines, bead_lines)
        self._fit_height()

    def _show_z(self, state: Optional[zcal.ZCalibrationState], z_lines: Sequence[str] = (),
                bead_lines: Sequence[str] = ()) -> None:
        """The z row's words and badges: "blank" when nothing is read, "user" for a typed m; a value read from a
        record shows its source in words, not a badge."""
        if _empty(state) and len(z_lines) <= 1:
            self.lbl_z.setText(Z_LINE)
            self.lbl_z_note.setVisible(True)
            self.badge_z.setVisible(True)
            self.badge_z.set_value(None)
            self.badge_m.setVisible(False)
        else:
            self.lbl_z.setText("\n".join(ln.split(" [")[0] if " [" in ln else ln for ln in z_lines))
            self.lbl_z_note.setVisible(False)
            self.badge_z.setVisible(False)
            origin = None if state is None else state.magnification_origin
            if origin == zcal.ORIGIN_USER and state is not None:
                self.badge_m.set_value(state.magnification)
                self.badge_m.setVisible(True)
            elif origin == zcal.ORIGIN_BLANK:
                self.badge_m.set_value(None)
                self.badge_m.setVisible(True)
            else:
                self.badge_m.setVisible(False)
        has_bead = state is not None and (state.bead is not None or bool(state.bead_error))
        self.lbl_bead.setText("\n".join(bead_lines) if (bead_lines and has_bead) else BEAD_LINE)
        self.badge_bead.setVisible(not (state is not None and state.bead is not None))

    def _fit_height(self) -> None:
        """Never shorter than its own lines at its width: a layout that squeezed the panel cut off the record's
        match line and the check warnings (manual test 3)."""
        lay = self.layout()
        if lay is None:
            return
        lay.activate()
        width = max(1, self.width())
        need = lay.totalHeightForWidth(width) if lay.hasHeightForWidth() else lay.totalSizeHint().height()
        need = max(int(need), lay.totalMinimumSize().height())
        if need != self.minimumHeight():
            self.setMinimumHeight(need)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._fit_height()

    def text(self) -> str:
        return "\n".join(measurement_lines(self._files, self._states))
