# -*- coding: utf-8 -*-
"""
Channel 2's own clustering parameters, once per acquisition: derived, pre-filled, editable.

The two-channel comparison never clusters the partner protein with betaII-spectrin's eps and min_samples: DBSCAN's
radius and minimum count are density parameters of one dataset (Ester et al. 1996). This box is where channel 2's own
values come from. Following the decision of 2026-10-14 (``docs/channel2_parameters.md``) it

  - measures each channel's precision by NeNA (Endesfelder et al. 2014) on the channel's WHOLE-FIELD, UNLINKED
    localization file, found by name from the loaded file (``tools.mps_file_match``, as the Axoplasm panel does) or
    chosen here;
  - asks for channel 2's dye and buffer (the switching-cycle table of Dempsey et al. 2011);
  - pre-fills eps_2 = eps_1 x NeNA_2 / NeNA_1 and min_samples_2 = round(1 + (n_loc,mol - 1)(1 - exp(-eps_2^2 /
    4 NeNA_2^2))), each with its origin written under it. An edited value becomes "user".

When channel 2's precision cannot be measured (too few next-frame pairs, a linked file, no whole-field file) the
fields stay EMPTY with the reason: nothing is invented and nothing is copied from channel 1. How densely each channel
is sampled -- the localizations within its own eps of a localization, the count DBSCAN tests -- is shown as
information, and a ratio of three or more as a warning; it never sets min_samples.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_channel_params as cp
from tools import mps_plot_style as plot_style
from tools.mps_crosschannel import OVERSAMPLING_WARN, locs_per_cell

Selection = Optional[Tuple[Any, Any]]

HELP_TEXT = (
    "How these values are derived\n\n"
    "DBSCAN's eps and Min Pts belong to one dataset, so channel 2 gets its own, computed from its own "
    "measurements:\n"
    "  eps_2 = eps_1 x NeNA_2 / NeNA_1\n"
    "  Min Pts_2 = round(1 + (n_loc,mol - 1) x (1 - exp(-eps_2^2 / (4 NeNA_2^2))))\n\n"
    "NeNA is the precision measured on each channel's whole-field, unlinked file (a picked axon has too few "
    "next-frame pairs; a linked file has none). n_loc,mol is the dye's mean switching cycles in its buffer "
    "(Dempsey et al. 2011, Table 1), a lower bound. With eps 25 nm, NeNA 12 nm and Alexa Fluor 647 in MEA this "
    "gives Gazal et al.'s Min Pts of 10.\n\n"
    "The full derivation, the dye table and the decision record: " + cp.DOC_PATH)


def _dim(widget: QtWidgets.QLabel) -> QtWidgets.QLabel:
    widget.setWordWrap(True)
    widget.setStyleSheet(f"color: {plot_style.TEXT_DIM_LIGHT};")
    widget.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


class Channel2ParametersDialog(QtWidgets.QDialog):
    """Channel 2's own eps and Min Pts, derived from its NeNA and dye, pre-filled and editable."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *,
                 precision_a: Optional[cp.ChannelPrecision] = None,
                 precision_b: Optional[cp.ChannelPrecision] = None,
                 eps1_nm: float = cp.GAZAL_EPS_NM,
                 dye: str = "", buffer: str = "",
                 selection_a: Selection = None, selection_b: Selection = None,
                 measure: Optional[Callable[[str, str], cp.ChannelPrecision]] = None,
                 file_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Channel 2's clustering")
        self.precision = {"a": precision_a or cp.ChannelPrecision(reason="no whole-field file: choose one"),
                          "b": precision_b or cp.ChannelPrecision(reason="no whole-field file: choose one")}
        self.eps1_nm = float(eps1_nm)
        self._measure = measure or (lambda path, how: cp.precision_from_file(path, found_by=how))
        self._selection = {"a": selection_a, "b": selection_b}
        self.derivation = cp.Channel2Derivation()
        self._user = {"eps": False, "min": False}
        lay = QtWidgets.QVBoxLayout(self)

        head = QtWidgets.QLabel(
            "Channel 2 is clustered with its own Epsilon and Min Pts, derived from its own precision and dye -- "
            "never channel 1's." + (f"\nChannel 2: {file_name}" if file_name else ""))
        head.setWordWrap(True)
        font = head.font()
        font.setBold(True)
        head.setFont(font)
        lay.addWidget(head)

        # --- precision of each channel
        box = QtWidgets.QGroupBox("Precision: NeNA on each channel's whole-field, unlinked file")
        grid = QtWidgets.QGridLayout(box)
        self.label_precision: Dict[str, QtWidgets.QLabel] = {}
        self.label_found: Dict[str, QtWidgets.QLabel] = {}
        self.btn_choose: Dict[str, QtWidgets.QPushButton] = {}
        for row, (tag, name) in enumerate((("a", "Channel 1"), ("b", "Channel 2"))):
            grid.addWidget(QtWidgets.QLabel(f"{name}:"), 2 * row, 0)
            self.label_precision[tag] = QtWidgets.QLabel()
            self.label_precision[tag].setWordWrap(True)
            grid.addWidget(self.label_precision[tag], 2 * row, 1)
            button = QtWidgets.QPushButton("Choose whole-field file...")
            button.setToolTip("The channel's whole-field, unlinked localization file (not the picked axon): NeNA "
                              "needs many next-frame pairs, which a linked file has merged.")
            button.clicked.connect(lambda _=False, t=tag: self._choose_file(t))
            self.btn_choose[tag] = button
            grid.addWidget(button, 2 * row, 2)
            self.label_found[tag] = _dim(QtWidgets.QLabel())
            grid.addWidget(self.label_found[tag], 2 * row + 1, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        lay.addWidget(box)

        # --- channel 2's dye
        dye_box = QtWidgets.QGroupBox("Channel 2's dye and buffer (localizations per molecule)")
        dye_lay = QtWidgets.QGridLayout(dye_box)
        self.combo_dye = QtWidgets.QComboBox()
        self.combo_dye.addItem("(choose the dye)", "")
        for name in cp.dyes():
            self.combo_dye.addItem(name, name)
        self.combo_buffer = QtWidgets.QComboBox()
        self.combo_buffer.addItem("(buffer)", "")
        for code in cp.BUFFERS:
            self.combo_buffer.addItem(cp.BUFFER_WORDS[code], code)
        for combo, value in ((self.combo_dye, dye), (self.combo_buffer, buffer)):
            index = combo.findData(value) if value else 0
            combo.setCurrentIndex(max(index, 0))
        tip = f"Mean switching cycles per molecule: {cp.DYE_TABLE_SOURCE}. A lower bound of the localizations " \
              f"per labelled molecule until the label is calibrated."
        self.combo_dye.setToolTip(tip)
        self.combo_buffer.setToolTip(tip)
        dye_lay.addWidget(self.combo_dye, 0, 0)
        dye_lay.addWidget(self.combo_buffer, 0, 1)
        self.label_cycles = _dim(QtWidgets.QLabel())
        dye_lay.addWidget(self.label_cycles, 1, 0, 1, 2)
        lay.addWidget(dye_box)

        # --- the two values
        form = QtWidgets.QFormLayout()
        self.edit_eps = QtWidgets.QLineEdit()
        self.edit_eps.setPlaceholderText("nanometres")
        self.edit_min = QtWidgets.QLineEdit()
        self.edit_min.setPlaceholderText("localizations")
        field_tip = ("Pre-filled from channel 2's own NeNA and dye; edit it if you have a reason (it is then "
                     f"recorded as yours). The derivation: {cp.DOC_PATH}")
        self.edit_eps.setToolTip(field_tip)
        self.edit_min.setToolTip(field_tip)
        self.label_eps_origin = _dim(QtWidgets.QLabel())
        self.label_min_origin = _dim(QtWidgets.QLabel())
        form.addRow("Epsilon (nm):", self.edit_eps)
        form.addRow("", self.label_eps_origin)
        form.addRow("Min Pts:", self.edit_min)
        form.addRow("", self.label_min_origin)
        lay.addLayout(form)

        self.label_notes = _dim(QtWidgets.QLabel())
        lay.addWidget(self.label_notes)
        self.label_sampling = _dim(QtWidgets.QLabel())
        lay.addWidget(self.label_sampling)

        self.label_help = QtWidgets.QLabel(
            f'<a href="{cp.DOC_URL}">How these values are derived</a> ({cp.DOC_PATH})')
        self.label_help.setOpenExternalLinks(True)
        self.label_help.setToolTip(HELP_TEXT)
        lay.addWidget(self.label_help)

        self.label_error = QtWidgets.QLabel("")
        self.label_error.setWordWrap(True)
        self.label_error.setStyleSheet(f"color: {plot_style.verdict('bad', dark=False)};")
        self.label_error.hide()
        lay.addWidget(self.label_error)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.btn_derived = buttons.addButton("Use derived values", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_derived.setToolTip("Put the derived values back in both fields (an edit is undone).")
        self.btn_derived.clicked.connect(self._use_derived)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self.edit_eps.textEdited.connect(lambda _t: self._edited("eps"))
        self.edit_min.textEdited.connect(lambda _t: self._edited("min"))
        self.edit_eps.textChanged.connect(lambda _t: self._show_sampling())
        self.combo_dye.currentIndexChanged.connect(lambda _i: self._rederive())
        self.combo_buffer.currentIndexChanged.connect(lambda _i: self._rederive())
        self._rederive()

    # ------------------------------------------------------------ state
    def dye(self) -> str:
        return str(self.combo_dye.currentData() or "")

    def buffer(self) -> str:
        return str(self.combo_buffer.currentData() or "")

    def _rederive(self) -> None:
        """Derive again and refill every field the user has not edited."""
        self.derivation = cp.derive_channel2(self.precision["a"], self.precision["b"], self.dye(), self.buffer(),
                                             eps1_nm=self.eps1_nm)
        for tag in ("a", "b"):
            p = self.precision[tag]
            self.label_precision[tag].setText(p.describe())
            self.label_found[tag].setText(p.found_by or "")
            self.label_found[tag].setVisible(bool(p.found_by))
        d = self.derivation
        if d.n_loc_mol is not None:
            self.label_cycles.setText(f"n_loc,mol = {d.n_loc_mol:g} switching cycles per molecule "
                                      f"({cp.DYE_TABLE_SOURCE}; a lower bound).")
        else:
            self.label_cycles.setText("Choose the dye and buffer of channel 2 to derive Min Pts.")
        if not self._user["eps"]:
            self.edit_eps.setText("" if d.eps_nm is None else f"{d.eps_nm:.1f}")
        if not self._user["min"]:
            self.edit_min.setText("" if d.min_samples is None else f"{d.min_samples}")
        self.label_notes.setText("\n".join(d.notes))
        self.label_notes.setVisible(bool(d.notes))
        self._show_origins()
        self._show_sampling()

    def _show_origins(self) -> None:
        d = self.derivation
        for key, label, value, origin, reason in (
                ("eps", self.label_eps_origin, d.eps_nm, d.eps_origin, d.eps_reason),
                ("min", self.label_min_origin, d.min_samples, d.min_samples_origin, d.min_samples_reason)):
            if self._user[key]:
                shown = "" if value is None else (f"{value:.1f} nm" if key == "eps" else f"{value}")
                label.setText("Origin: user (typed)." + (f" The derived value is {shown}." if shown else ""))
            elif value is not None:
                label.setText(f"Origin: {origin}.")
            else:
                label.setText(f"Empty: {reason}.")

    def _edited(self, key: str) -> None:
        self._user[key] = True
        self._show_origins()

    def _use_derived(self) -> None:
        self._user = {"eps": False, "min": False}
        self._rederive()

    def _choose_file(self, tag: str) -> None:
        start = self.precision[tag].source_path
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Whole-field localization file of channel {1 if tag == 'a' else 2}", start,
            "Localizations (*.hdf5 *.h5)")
        if path:
            self.set_whole_field(tag, path)

    def set_whole_field(self, tag: str, path: str) -> None:
        """Measure the NeNA of ``path`` for channel ``tag`` ("a" or "b") and derive again."""
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        try:
            self.precision[tag] = self._measure(path, "chosen")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._rederive()

    def _show_sampling(self) -> None:
        sel_a, sel_b = self._selection["a"], self._selection["b"]
        if sel_a is None or sel_b is None:
            self.label_sampling.setText("")
            self.label_sampling.hide()
            return
        eps_b = self._number(self.edit_eps.text())
        count_a = locs_per_cell(np.asarray(sel_a[0], float), np.asarray(sel_a[1], float), self.eps1_nm)
        if count_a is None:
            text = ""
        elif eps_b is None:
            text = (f"Within channel 1's eps ({self.eps1_nm:g} nm) of a localization of this selection: "
                    f"{count_a:.0f} of channel 1's.")
        else:
            count_b = locs_per_cell(np.asarray(sel_b[0], float), np.asarray(sel_b[1], float), eps_b)
            text = (f"Within each channel's own eps of a localization of this selection (the count Min Pts is "
                    f"compared with): {count_a:.0f} of channel 1's ({self.eps1_nm:g} nm), "
                    + ("none" if count_b is None else f"{count_b:.0f}") + f" of channel 2's ({eps_b:g} nm).")
            if count_a and count_b:
                ratio = count_b / count_a
                if ratio >= OVERSAMPLING_WARN or ratio <= 1.0 / OVERSAMPLING_WARN:
                    text += (f" A factor of {max(ratio, 1 / ratio):.1f}: the channels are sampled differently. "
                             f"A warning only -- the ratio also holds the partner's abundance, so it does not "
                             f"set Min Pts.")
        self.label_sampling.setText(text)
        self.label_sampling.setVisible(bool(text))

    # --------------------------------------------------------- behaviour
    @staticmethod
    def _number(text: str) -> Optional[float]:
        try:
            value = float(text.strip().replace(",", "."))
        except ValueError:
            return None
        return value if np.isfinite(value) and value > 0 else None

    def _accept_if_valid(self) -> None:
        """Keep the box open on a bad value rather than losing both."""
        if self.values() is None:
            return
        self.accept()

    def values(self) -> Optional[Dict[str, Any]]:
        """The pair in the fields with its provenance, or None with the reason shown in the box."""
        def number(edit: QtWidgets.QLineEdit, name: str, lo: float, hi: float, reason: str) -> Optional[float]:
            text = edit.text().strip().replace(",", ".")
            if not text:
                self._complain(f"{name} is empty" + (f": {reason}" if reason else "") + ". Type a value or "
                               "choose the whole-field file / the dye.", edit)
                return None
            try:
                value = float(text)
            except ValueError:
                self._complain(f"{name}: {text!r} is not a number.", edit)
                return None
            if not (lo <= value <= hi):
                self._complain(f"{name}: {value:g} is outside {lo:g} to {hi:g}.", edit)
                return None
            return value

        eps = number(self.edit_eps, "Epsilon", *cp.EPS_LIMITS_NM, self.derivation.eps_reason)
        if eps is None:
            return None
        minimum = number(self.edit_min, "Min Pts", *cp.MIN_SAMPLES_LIMITS, self.derivation.min_samples_reason)
        if minimum is None:
            return None
        self.label_error.hide()
        record = cp.provenance(self.derivation, self.precision["a"], self.precision["b"], float(eps), int(minimum))
        if self._user["eps"]:
            record["eps_b_origin"] = cp.ORIGIN_USER
        if self._user["min"]:
            record["min_samples_b_origin"] = cp.ORIGIN_USER
        return record

    def _complain(self, text: str, edit: QtWidgets.QLineEdit) -> None:
        self.label_error.setText(text)
        self.label_error.show()
        edit.setFocus()
        edit.selectAll()


def measure_channels(path_a: str, path_b: str) -> Tuple[cp.ChannelPrecision, cp.ChannelPrecision]:
    """Find each loaded channel's whole-field file by name and measure its NeNA."""
    out = []
    for path in (path_a, path_b):
        whole, how = cp.find_whole_field(path)
        if whole:
            out.append(cp.precision_from_file(whole, found_by=how))
        else:
            out.append(cp.ChannelPrecision(reason=how, found_by=""))
    return out[0], out[1]


def ask_channel2_parameters(
        parent: Optional[QtWidgets.QWidget], *,
        x_a: Any = None, y_a: Any = None, x_b: Any = None, y_b: Any = None,
        path_a: str = "", path_b: str = "",
        eps1_nm: float = cp.GAZAL_EPS_NM,
        remembered: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Ask for channel 2's own eps and Min Pts, pre-filled from its NeNA and dye.

    Returns the pair with its provenance (``tools.mps_channel_params.provenance``), or None when the person
    declined -- in which case the caller must refuse the comparison, which is what ``cross_channel_transverse``
    does on its own anyway. The densities shown are measured on the selections passed in (the ROI the comparison
    will use, before the axial slab).
    """
    QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
    try:
        precision_a, precision_b = measure_channels(path_a, path_b)
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
    previous = remembered or {}
    dialog = Channel2ParametersDialog(
        parent, precision_a=precision_a, precision_b=precision_b, eps1_nm=eps1_nm,
        dye=str(previous.get("dye_b", "") or ""), buffer=str(previous.get("buffer_b", "") or ""),
        selection_a=None if x_a is None else (x_a, y_a),
        selection_b=None if x_b is None else (x_b, y_b),
        file_name=os.path.basename(path_b))
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.values()
