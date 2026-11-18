# -*- coding: utf-8 -*-
"""
The question asked before an Emitters run whose precisions look under-reported (check T4, Qt side; F5c).

``make_gate(parent)`` returns the callable ``tools.emitters_layer.asking_precision`` installs around the
interactive MPS analysis. As the perimeter's question (``tools.perimeter_cost_ui``), it shows the dialog only in
an interactive session on the GUI thread; offscreen, from a worker thread, in the batch and in the tests it answers
"run as is" without showing anything, and the analysis notes say what was found.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt5 import QtCore, QtWidgets

from tools import emitters_layer as el
from tools.emitters_prior import NENA_RATIO_MAX
from tools.perimeter_cost_ui import interactive


def message(pre: el.PrecisionPrecheck) -> str:
    """The dialog's text: the measurement, the suggestion and where the numbers come from."""
    value = pre.suggested_se_adjust_nm
    return (
        f"Before the Emitters run: {pre.words()}.\n\n"
        f"The localization precisions of this slab look under-reported. The model then splits emitters "
        f"(+53 % emitters with precisions 20 % under-reported, simulation F4).\n\n"
        f"Suggested correction: se_adjust_nm = {value:.1f} nm (NeNA - median precision), added to every "
        f"precision of this run and recorded with its parameters.\n\n"
        f"Threshold {NENA_RATIO_MAX:g} [ad hoc, the 20 % case of F4]; NeNA from >= 500 next-frame pairs of the "
        f"slab [simulation, F4]. docs/emitters.md")


def build_dialog(parent: Optional[Any], pre: el.PrecisionPrecheck) -> Any:
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Warning)
    box.setWindowTitle("Emitters: precision under-reported?")
    box.setText(message(pre))
    apply = box.addButton("Apply and run", QtWidgets.QMessageBox.AcceptRole)
    as_is = box.addButton("Run as is", QtWidgets.QMessageBox.AcceptRole)
    cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
    apply.setProperty("precision_answer", el.PRECISION_APPLY)
    as_is.setProperty("precision_answer", el.PRECISION_AS_IS)
    cancel.setProperty("precision_answer", el.PRECISION_CANCEL)
    box.setDefaultButton(apply)
    box.setEscapeButton(cancel)
    return box


def ask(parent: Optional[Any], pre: el.PrecisionPrecheck) -> str:
    """Show the dialog; its answer (Escape or closing it is Cancel)."""
    box = build_dialog(parent, pre)
    box.exec_()
    clicked = box.clickedButton()
    answer = None if clicked is None else clicked.property("precision_answer")
    return str(answer) if answer in el.PRECISION_ANSWERS else el.PRECISION_CANCEL


def make_gate(parent: Optional[Any]) -> Callable[[el.PrecisionPrecheck], str]:
    """The gate for ``el.asking_precision``: the dialog when interactive, else "run as is" without a word."""
    def gate(pre: el.PrecisionPrecheck) -> str:
        if not interactive():
            return el.PRECISION_AS_IS
        lifted = 0
        while QtWidgets.QApplication.overrideCursor() is not None and lifted < 16:
            QtWidgets.QApplication.restoreOverrideCursor()
            lifted += 1
        try:
            return ask(parent, pre)
        finally:
            for _ in range(lifted):
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
    return gate
