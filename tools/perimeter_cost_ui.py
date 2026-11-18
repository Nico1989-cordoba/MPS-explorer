# -*- coding: utf-8 -*-
"""
The question asked before a perimeter expected to take long (Qt side).

``make_gate(parent)`` returns the callable ``tools.mps_perimeter_cost.asking``
installs around the interactive MPS analysis. It shows the dialog only in an
interactive session, on the GUI thread; offscreen (the golden reference, the
tests) or from a worker thread it answers "every start" without showing
anything, so no non-interactive path can be blocked by it.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt5 import QtCore, QtWidgets

from tools import mps_perimeter_cost as cost

# Qt platforms with no screen and no one to answer.
HEADLESS_PLATFORMS = frozenset({"offscreen", "minimal"})


def interactive() -> bool:
    """A QApplication on a real screen, and this is its thread."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return False
    if str(QtWidgets.QApplication.platformName()).lower() in HEADLESS_PLATFORMS:
        return False
    return bool(QtCore.QThread.currentThread() is app.thread())


def message(n_clusters: int) -> str:
    """The dialog's text: the count, both estimates and where they come from."""
    n = int(n_clusters)
    every = cost.format_minutes(cost.estimate_perimeter_seconds(n))
    one = cost.format_minutes(cost.estimate_one_start_seconds(n))
    return (
        f"{n:,} clusters were kept. Building the perimeter by 2-opt from every "
        f"start is expected to take {every}; meanwhile the program does not respond, "
        f"or, with a slow method running in the background, waits with its Cancel button.\n\n"
        f"From one start (as before 2026-09-19): {one}; the tour can come out a "
        f"few percent longer, and the export records 'one start'.\n\n"
        f"Estimates from synthetic axons on one computer (docs/perimeter_time.md); "
        f"a messier contour takes longer (1.4 x on the simulated Emitters axon of "
        f"docs/emitters.md). Many small clusters usually mean eps, Min Pts or the "
        f"method's scale are worth a look first.")


def build_dialog(parent: Optional[Any], n_clusters: int) -> Any:
    """The QMessageBox, with its three buttons named by their answers."""
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Question)
    box.setWindowTitle("Perimeter with many clusters")
    box.setText(message(n_clusters))
    every = box.addButton("Continue (every start)", QtWidgets.QMessageBox.AcceptRole)
    one = box.addButton("One start (faster)", QtWidgets.QMessageBox.AcceptRole)
    cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
    every.setProperty("perimeter_answer", cost.EVERY_START)
    one.setProperty("perimeter_answer", cost.ONE_START)
    cancel.setProperty("perimeter_answer", cost.CANCEL)
    box.setDefaultButton(every)
    box.setEscapeButton(cancel)
    return box


def ask(parent: Optional[Any], n_clusters: int) -> str:
    """Show the dialog; its answer (Escape or closing it is Cancel)."""
    box = build_dialog(parent, n_clusters)
    box.exec_()
    clicked = box.clickedButton()
    answer = None if clicked is None else clicked.property("perimeter_answer")
    return str(answer) if answer in cost.ANSWERS else cost.CANCEL


def make_gate(parent: Optional[Any]) -> Callable[[int], str]:
    """The gate for ``cost.asking``: the dialog when interactive, else
    "every start" without a word (the analysis' note says the rest).

    The wait cursors the callers stacked (the results window and the main
    window each set one) are lifted while the dialog is shown and put back
    after it, as many as there were.
    """
    def gate(n_clusters: int) -> str:
        if not interactive():
            return cost.EVERY_START
        lifted = 0
        while QtWidgets.QApplication.overrideCursor() is not None and lifted < 16:
            QtWidgets.QApplication.restoreOverrideCursor()
            lifted += 1
        try:
            return ask(parent, n_clusters)
        finally:
            for _ in range(lifted):
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
    return gate
