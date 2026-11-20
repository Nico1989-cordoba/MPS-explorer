# -*- coding: utf-8 -*-
"""
The confirmation shown every time a slow clustering method is about to run (user decision A2 (b), 2026-10-23).

A2 (a): the Batch and the per-axon null refuse the slow methods (Emitters, OPTICS, voronoi_csr:
``tools.method_costs.is_slow``) unless 'Allow slow methods' is ticked. A2 (b): ADDITIONALLY, each time a slow method
is going to run -- 'Compare methods...' with one ticked, the Batch or the null with the box ticked -- a dialog says
which methods, the time estimate, and asks to go on. Nothing runs when the answer is no.

The estimate is the per-method cost table's (``tools.method_costs``; a lower bound for the slow methods, said in the
dialog). Tests and scripts answer through :data:`ANSWER` (a callable taking the dialog's text and returning True to
go on); :data:`ASKED` keeps every text shown.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from tools.method_costs import method_label, words_seconds

TITLE = "Slow method: confirm before it runs"
# Tests and scripts: answer without a dialog (None: ask the user).
ANSWER: Optional[Callable[[str], bool]] = None
ASKED: List[str] = []

__all__ = ["TITLE", "ANSWER", "ASKED", "confirm_text", "confirm_slow"]


def confirm_text(where: str, items: Sequence[Tuple[str, float]], total_s: Optional[float] = None,
                 note: str = "") -> str:
    """The dialog's words: what will run in ``where``, each slow method with its estimate (seconds), the total."""
    lines = [f"{where} is about to run slow method(s) (decision A2):", ""]
    for name, sec in items:
        lines.append(f"  - {method_label(name)}: about {words_seconds(sec)}")
    tot = float(sum(s for _n, s in items)) if total_s is None else float(total_s)
    lines += ["", f"Estimated time in all: about {words_seconds(tot)} -- a lower bound (the slow methods grow faster "
              "than the table says with many localizations or clusters)."]
    if note:
        lines += ["", note]
    lines += ["", "Run it now? (Cancel runs nothing.)"]
    return "\n".join(lines)


def confirm_slow(parent: Any, where: str, items: Sequence[Tuple[str, float]], total_s: Optional[float] = None,
                 note: str = "") -> bool:
    """Ask before running the slow methods ``items`` ((method, estimated seconds)); True to go on. With no item,
    True without asking."""
    if not items:
        return True
    text = confirm_text(where, items, total_s, note)
    ASKED.append(text)
    if ANSWER is not None:
        return bool(ANSWER(text))
    from PyQt5 import QtWidgets
    box = QtWidgets.QMessageBox(parent)
    box.setObjectName("slow_confirm_dialog")
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle(TITLE)
    box.setText(text)
    run = box.addButton("Run", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
    box.exec_()
    return box.clickedButton() is run
