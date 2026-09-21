# -*- coding: utf-8 -*-
"""
Asking for channel 2's own clustering parameters, once per acquisition.

The two-channel comparison used to cluster the partner protein with
betaII-spectrin's eps and min_samples whenever nothing set its own. That
is an assumption about the partner's density rather than a measurement
of it, and measured over five real adducin files the two channels are 1
to 138 times apart in localizations per resolution cell. min_samples
counts localizations, so the same number does not mean the same thing in
the two channels, and on one axon the difference was 13 clusters against
35 and a median heterotypic distance of 280 nm against 223.

So the comparison refuses, and this is the box that unblocks it.

The hard part is not refusing, it is that a person asked for a number
they do not have will type anything to make the box go away. So the box
carries what they need in order to answer:

  - the measured density of BOTH channels, in the unit min_samples
    counts, which is the whole reason the question exists;
  - what the program's own estimator makes of channel 2's points, and --
    just as important -- how far that estimator is off on channel 1,
    where the right answer is known.

Nothing is filled in. The estimate is one deliberate click away, never
an Enter away: measured on this project's own spectrin, where 25 nm is
established, the same k-distance estimator returns 58 to 109 nm, so it
is a starting point and not an answer.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from tools import mps_plot_style as plot_style
from tools.mps_crosschannel import RESOLUTION_CELL_NM, locs_per_cell

# What the k-distance estimator returned on this project's 18 spectrin
# axons, where the established value is 25 nm. Quoted so the estimate is
# never read as an answer.
ESTIMATOR_ON_KNOWN_NM = (58.0, 109.0)
ESTABLISHED_CHANNEL1_NM = 25.0


def estimate_for(x: Any, y: Any) -> Tuple[Optional[float], Optional[int]]:
    """What the program's own estimators make of one channel's points.

    The same two functions the "auto" setting of the cluster button uses,
    run on channel 2's own localizations -- so the number owes nothing to
    channel 1. Returns (None, None) when there are too few points.
    """
    try:
        from tools import clustering
    except Exception:                                     # noqa: BLE001
        return None, None
    pts = np.column_stack([np.asarray(x, float).ravel(),
                           np.asarray(y, float).ravel()])
    if len(pts) < 10:
        return None, None
    try:
        eps = float(clustering.estimate_optimal_eps(pts, k=5, percentile=90))
        minimum = int(clustering.estimate_min_samples(len(pts), 2))
    except Exception:                                     # noqa: BLE001
        return None, None
    if not np.isfinite(eps) or eps <= 0:
        return None, minimum
    return eps, minimum


class Channel2ParametersDialog(QtWidgets.QDialog):
    """Channel 2's own eps and min samples, with the evidence to choose."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *,
                 cell_a: Optional[float] = None,
                 cell_b: Optional[float] = None,
                 estimate: Tuple[Optional[float], Optional[int]] = (None, None),
                 channel1: Optional[Dict[str, Any]] = None,
                 file_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Channel 2's clustering")
        self._estimate = estimate
        lay = QtWidgets.QVBoxLayout(self)

        head = QtWidgets.QLabel(
            "The two channels are not clustered with the same numbers."
            + (f"\nChannel 2: {file_name}" if file_name else ""))
        head.setWordWrap(True)
        font = head.font()
        font.setBold(True)
        head.setFont(font)
        lay.addWidget(head)

        lay.addWidget(self._measured(cell_a, cell_b, channel1))

        eps_hint, min_hint = estimate
        self._shown_estimate = QtWidgets.QLabel(self._estimate_text())
        self._shown_estimate.setWordWrap(True)
        self._shown_estimate.setStyleSheet(
            f"color: {plot_style.TEXT_DIM_LIGHT};")
        lay.addWidget(self._shown_estimate)

        form = QtWidgets.QFormLayout()
        self.edit_eps = QtWidgets.QLineEdit()
        self.edit_eps.setPlaceholderText("nanometres")
        self.edit_min = QtWidgets.QLineEdit()
        self.edit_min.setPlaceholderText("localizations")
        # Nothing is filled in. A number already in the box is an answer
        # one Enter accepts, and the estimate above is measured to be off
        # by a factor of 2 to 4 where the answer is known.
        form.addRow("Epsilon (nm):", self.edit_eps)
        form.addRow("Min Pts:", self.edit_min)
        lay.addLayout(form)

        self.label_error = QtWidgets.QLabel("")
        self.label_error.setWordWrap(True)
        self.label_error.setStyleSheet(
            f"color: {plot_style.verdict('bad', dark=False)};")
        self.label_error.hide()
        lay.addWidget(self.label_error)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        if eps_hint is not None or min_hint is not None:
            self.btn_estimate = buttons.addButton(
                "Use the estimate", QtWidgets.QDialogButtonBox.ActionRole)
            self.btn_estimate.clicked.connect(self._fill_with_estimate)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    # ------------------------------------------------------------ text
    def _measured(self, cell_a: Optional[float], cell_b: Optional[float],
                  channel1: Optional[Dict[str, Any]]) -> QtWidgets.QLabel:
        lines = []
        if cell_a and cell_b:
            ratio = max(cell_a, cell_b) / min(cell_a, cell_b)
            lines.append(
                f"Within {RESOLUTION_CELL_NM:g} nm of a localization there "
                f"are {cell_a:.0f} of channel 1's and {cell_b:.0f} of "
                f"channel 2's, a factor of {ratio:.0f}. Min Pts counts "
                f"localizations, so channel 1's value does not mean the "
                f"same thing here.")
        else:
            lines.append(
                "Min Pts counts localizations, and the two channels are "
                "rarely sampled alike, so channel 1's value does not mean "
                "the same thing in channel 2.")
        if channel1 and "eps_nm" in channel1:
            lines.append(
                f"Channel 1 is on eps {float(channel1['eps_nm']):g} nm, "
                f"Min Pts {int(channel1['min_samples'])}. Type those here "
                f"too if that is what you want for channel 2 -- what this "
                f"box will not do is assume it.")
        label = QtWidgets.QLabel("\n\n".join(lines))
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _estimate_text(self) -> str:
        eps, minimum = self._estimate
        if eps is None and minimum is None:
            return ("There are too few localizations in channel 2 for the "
                    "program to estimate anything from them.")
        parts = []
        if eps is not None:
            parts.append(f"eps {eps:.0f} nm")
        if minimum is not None:
            parts.append(f"Min Pts {minimum}")
        return (
            f"Estimated from channel 2's own points: {', '.join(parts)}. "
            f"A starting point, not an answer: on channel 1, where "
            f"{ESTABLISHED_CHANNEL1_NM:g} nm is the established value, the "
            f"same estimator returns {ESTIMATOR_ON_KNOWN_NM[0]:.0f} to "
            f"{ESTIMATOR_ON_KNOWN_NM[1]:.0f} nm.")

    # --------------------------------------------------------- behaviour
    def _fill_with_estimate(self) -> None:
        eps, minimum = self._estimate
        if eps is not None:
            self.edit_eps.setText(f"{eps:.0f}")
        if minimum is not None:
            self.edit_min.setText(f"{int(minimum)}")

    def _accept_if_valid(self) -> None:
        """Keep the box open on a bad value rather than losing both.

        Cancelling is how a person refuses; a mistyped character is not a
        refusal, and throwing away the field they got right would cost
        them the whole sequence a second time.
        """
        values = self.values()
        if values is None:
            return
        self.accept()

    def values(self) -> Optional[Dict[str, float]]:
        """The pair typed, or None with the reason shown in the box."""
        def number(edit: QtWidgets.QLineEdit, name: str,
                   lo: float, hi: float) -> Optional[float]:
            text = edit.text().strip().replace(",", ".")
            if not text:
                self._complain(f"{name} is empty.", edit)
                return None
            try:
                value = float(text)
            except ValueError:
                self._complain(f"{name}: {text!r} is not a number.", edit)
                return None
            if not (lo <= value <= hi):
                self._complain(
                    f"{name}: {value:g} is outside {lo:g} to {hi:g}.", edit)
                return None
            return value

        eps = number(self.edit_eps, "Epsilon", 1e-9, 1000.0)
        if eps is None:
            return None
        minimum = number(self.edit_min, "Min Pts", 1, 10000)
        if minimum is None:
            return None
        self.label_error.hide()
        return {"eps_nm": float(eps), "min_samples": float(int(minimum))}

    def _complain(self, text: str, edit: QtWidgets.QLineEdit) -> None:
        self.label_error.setText(text)
        self.label_error.show()
        edit.setFocus()
        edit.selectAll()


def ask_channel2_parameters(
        parent: Optional[QtWidgets.QWidget], *,
        x_a: Any = None, y_a: Any = None, x_b: Any = None, y_b: Any = None,
        channel1: Optional[Dict[str, Any]] = None,
        file_name: str = "") -> Optional[Dict[str, float]]:
    """
    Ask for channel 2's own eps and min samples.

    Returns the pair, or None when the person declined -- in which case
    the caller must refuse the comparison, which is what
    ``cross_channel_transverse`` does on its own anyway.

    The densities and the estimate are measured on whatever selection is
    passed in, which is the ROI the comparison will use, before the axial
    slab. Say so where they are shown.
    """
    cell_a = None if x_a is None else locs_per_cell(x_a, y_a)
    cell_b = None if x_b is None else locs_per_cell(x_b, y_b)
    estimate = (None, None) if x_b is None else estimate_for(x_b, y_b)
    dialog = Channel2ParametersDialog(
        parent, cell_a=cell_a, cell_b=cell_b, estimate=estimate,
        channel1=channel1, file_name=file_name)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.values()
