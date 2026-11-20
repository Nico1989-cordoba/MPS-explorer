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

from typing import Any, Callable, Optional, Sequence, Tuple

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


# How many files the batch question lists by name before "and N more" [ad hoc: fits a 700-px-high screen].
BATCH_LISTED_FILES = 8


def batch_message(over: Sequence[Tuple[str, int]], n_files: int, compared: bool = False,
                  scanned: bool = True) -> str:
    """The batch question's text (F1, 2026-10-22).

    ``over``: (file name, kept clusters) of each file whose perimeter reaches the warning threshold. ``scanned``
    False: the method is too slow to cluster every file twice, so no file was counted and the question is asked
    for whatever file ends up over the threshold.
    """
    threshold = cost.PERIMETER_WARN_CLUSTERS
    tail = (f"\n\nThe answer applies to the whole batch: every file with {threshold:,} kept clusters or more"
            + (", in the method comparison too" if compared else "") + ". "
            "Estimates from synthetic axons on one computer (docs/perimeter_time.md); a messier contour takes "
            "longer (1.4 x on the simulated Emitters axon). From one start (as before 2026-09-19) the tour can "
            "come out a few percent longer, and the export records 'one start'.")
    if not scanned:
        return (f"The clusters of this method are not counted before the batch: clustering every file twice "
                f"would take too long. A file that ends with {threshold:,} kept clusters or more would build its "
                f"perimeter by 2-opt from every start for {cost.format_minutes(cost.estimate_perimeter_seconds(threshold))} "
                f"or more, while the batch's bar shows the stage it is in.\n\n"
                f"What should the batch do with such a file?" + tail)
    every = sum(cost.estimate_perimeter_seconds(n) for _name, n in over)
    one = sum(cost.estimate_one_start_seconds(n) for _name, n in over)
    lines = [f"  {name}: {n:,} clusters, {cost.format_minutes(cost.estimate_perimeter_seconds(n))} "
             f"(one start: {cost.format_minutes(cost.estimate_one_start_seconds(n))})"
             for name, n in list(over)[:BATCH_LISTED_FILES]]
    if len(over) > BATCH_LISTED_FILES:
        lines.append(f"  and {len(over) - BATCH_LISTED_FILES} more")
    return (f"{len(over)} of {n_files} file(s) keep {threshold:,} clusters or more. Their perimeters, built by "
            f"2-opt from every start, are expected to take {cost.format_minutes(every)} in all "
            f"(from one start: {cost.format_minutes(one)}):\n\n" + "\n".join(lines) + tail)


def build_batch_dialog(parent: Optional[Any], over: Sequence[Tuple[str, int]], n_files: int,
                       compared: bool = False, scanned: bool = True) -> Any:
    """The batch's QMessageBox: the same three answers as the axon window's, for the whole batch."""
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Question)
    box.setWindowTitle("Batch: perimeters with many clusters")
    box.setText(batch_message(over, n_files, compared, scanned))
    every = box.addButton("Every start (as always)", QtWidgets.QMessageBox.AcceptRole)
    one = box.addButton("One start (faster)", QtWidgets.QMessageBox.AcceptRole)
    cancel = box.addButton("Cancel the batch", QtWidgets.QMessageBox.RejectRole)
    every.setToolTip("Build each long perimeter by 2-opt from every start and keep the shortest tour:\n"
                     "the analysis the batch always ran, and the same numbers.")
    one.setToolTip("Build each long perimeter from one start, as this program did before 2026-09-19:\n"
                   "much faster; the tour can be a few percent longer, and the export says 'one start'.")
    cancel.setToolTip("Start nothing. Fewer, larger clusters (eps, Min Pts, the method's scale) make the\n"
                      "perimeter fast again.")
    every.setProperty("perimeter_answer", cost.EVERY_START)
    one.setProperty("perimeter_answer", cost.ONE_START)
    cancel.setProperty("perimeter_answer", cost.CANCEL)
    box.setDefaultButton(every)
    box.setEscapeButton(cancel)
    return box


def ask_batch(parent: Optional[Any], over: Sequence[Tuple[str, int]], n_files: int,
              compared: bool = False, scanned: bool = True) -> str:
    """Show the batch question; its answer (Escape or closing it is Cancel)."""
    box = build_batch_dialog(parent, over, n_files, compared, scanned)
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
