# -*- coding: utf-8 -*-
"""
The Measurement panel of the main window (UI stage 2, design 12.7): what each loaded file says about the measurement.

One row per loaded channel with its pixel size and the source of that number, in the program's own pixel-size words
(``tools.mps_param_registry.pixel_source_words``, the same dictionary as the MPS analysis window's provenance line);
then the per-measurement z calibration: "none read", badge "blank", z as fitted by the localization software; whether
each file carries a Picasso "Z Calibration" record of its own (``Localizations.z_calibration``) or is a 2D file; and
two inputs, "Calibration record..." and "Bead stack...", present and disabled until SCI-1 (Q11).

What it guarantees: it is a display. It stores nothing (no settings field, no file written), passes nothing to any
analysis or export, and is not a second editor of the pixel size, whose one input stays the per-folder prompt at load.
The rows come from the registry entries whose home is ``measurement``; SCI-1 enables the z-calibration rows.

``measurement_lines`` is pure (no Qt) so the texts are tested without a window.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from tools import mps_param_registry as reg
from tools.mps_origin_ui import OriginBadge
from tools.mps_plot_style import verdict

__all__ = ["ChannelFile", "MeasurementPanel", "measurement_lines", "z_record_line", "z_records_line", "TITLE",
           "CALIBRATION_BUTTON", "BEAD_BUTTON", "DISABLED_TIP"]

TITLE = "Measurement"
NO_FILE = "No file loaded."
Z_LINE = "z calibration: - (none read)"
Z_NOTE = "z as fitted by the localization software"
BEAD_LINE = "bead stack: - (not given)"
CALIBRATION_BUTTON = "Calibration record..."
BEAD_BUTTON = "Bead stack..."
DISABLED_TIP = ("Reading a calibration record (Picasso YAML, or the 'Fit 3D' / 'Z Calibration' block of the file) and "
                "a localized bead stack comes with SCI-1: the magnification factor m with its source, the calibrated "
                "range, the lateral shift with depth. Until then nothing is read and z is used as the localization "
                "software fitted it.")
PANEL_TIP = ("What each loaded file says about the measurement. A display: nothing here is stored or passed to an "
             "analysis. The pixel size is given at load (from the file's metadata, or typed once per folder); the "
             "z calibration is not read yet.")
PIXEL_TIP = ("The pixel size of this file and where the number comes from. Every lateral distance scales with it and "
             "every area with its square. It is read at load (Picasso YAML or the HDF5 metadata) or typed once for "
             "the folder; this panel only shows it.")


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
            record = loc.z_calibration is not None
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


def z_record_words(f: ChannelFile) -> str:
    """Whether the file carries its own z-calibration record, or is a 2D file."""
    if not f.is_3d:
        return "2D file: no z"
    if f.has_z_record:
        return "the file carries a Picasso 'Z Calibration' record (not read by this version)"
    return "the file carries no 'Z Calibration' record"


def z_record_line(f: ChannelFile) -> str:
    return f"Channel {f.channel}: {z_record_words(f)}"


def z_records_line(files: Sequence[ChannelFile]) -> str:
    """Every channel's record on one line (the panel's row)."""
    return "; ".join(z_record_line(f) for f in files)


def measurement_lines(files: Sequence[ChannelFile]) -> List[str]:
    """Every line of the panel, in order (badges as "[word]"): what a copy of the panel gives."""
    if not files:
        return [NO_FILE]
    lines = [pixel_line(f) for f in files]
    lines.append(f"{Z_LINE} [{reg.badge('measurement.z_calibration', None)}] - {Z_NOTE}")
    lines.append(z_records_line(files))
    lines.append(f"{BEAD_LINE} [{reg.badge('measurement.bead_stack', None)}]")
    return lines


def _label(text: str = "", tip: str = "", wrap: bool = False) -> QtWidgets.QLabel:
    w = QtWidgets.QLabel(text)
    w.setWordWrap(wrap)
    w.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    if tip:
        w.setToolTip(tip)
    return w


class MeasurementPanel(QtWidgets.QGroupBox):
    """The Measurement panel (12.7). ``set_files([ChannelFile, ...])`` redraws it; it never writes anything."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(TITLE, parent)
        self.setObjectName("measurement_panel")
        self.setToolTip(PANEL_TIP)
        self._files: List[ChannelFile] = []
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        self.lbl_pixels = _label(NO_FILE, PIXEL_TIP, wrap=True)
        self.lbl_pixels.setObjectName("measurement_pixel_sizes")
        lay.addWidget(self.lbl_pixels)
        zrow = QtWidgets.QHBoxLayout()
        zrow.setSpacing(4)
        self.lbl_z = _label(Z_LINE, reg.tooltip("measurement.z_calibration", None))
        self.lbl_z.setObjectName("measurement_z_calibration")
        zrow.addWidget(self.lbl_z)
        self.badge_z = OriginBadge("measurement.z_calibration", None)
        zrow.addWidget(self.badge_z)
        self.lbl_z_note = _label(f"- {Z_NOTE}")
        zrow.addWidget(self.lbl_z_note, 1)
        lay.addLayout(zrow)
        self.lbl_records = _label("", wrap=True, tip="Read from each file's own metadata (Picasso's 'Z Calibration' block). It is "
                                      "shown, not used: reading it comes with SCI-1.")
        self.lbl_records.setObjectName("measurement_z_records")
        lay.addWidget(self.lbl_records)
        brow = QtWidgets.QHBoxLayout()
        brow.setSpacing(4)
        self.btn_calibration = QtWidgets.QPushButton(CALIBRATION_BUTTON)
        self.btn_calibration.setObjectName("button_z_calibration_record")
        self.btn_bead = QtWidgets.QPushButton(BEAD_BUTTON)
        self.btn_bead.setObjectName("button_bead_stack")
        for b in (self.btn_calibration, self.btn_bead):
            b.setEnabled(False)
            b.setToolTip(DISABLED_TIP)
            brow.addWidget(b)
        self.lbl_bead = _label(BEAD_LINE, reg.tooltip("measurement.bead_stack", None))
        self.lbl_bead.setObjectName("measurement_bead_stack")
        brow.addWidget(self.lbl_bead)
        self.badge_bead = OriginBadge("measurement.bead_stack", None)
        brow.addWidget(self.badge_bead)
        brow.addStretch(1)
        lay.addLayout(brow)
        self.set_files([])

    # ------------------------------------------------------------------ public
    def files(self) -> List[ChannelFile]:
        return list(self._files)

    def keys_shown(self) -> List[str]:
        """The registry keys this panel shows (the coverage test of 12.3)."""
        return [e.key for e in reg.by_home("measurement")]

    def set_files(self, files: Sequence[ChannelFile]) -> None:
        self._files = list(files)
        if not self._files:
            self.lbl_pixels.setText(NO_FILE)
            self.lbl_pixels.setStyleSheet("")
            self.lbl_records.setText("")
            self.lbl_records.setVisible(False)
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
        self.lbl_records.setText(z_records_line(self._files))
        self.lbl_records.setVisible(True)

    def text(self) -> str:
        return "\n".join(measurement_lines(self._files))
