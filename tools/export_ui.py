# -*- coding: utf-8 -*-
"""
What an export asks before it adds an axon a table already holds.

Every panel's export appends, which is what lets a folder's worth of
axons accumulate into one table. Appending the same axon twice is silent
and weights it twice in every statistic afterwards, and it happens: the
table of axon 7 exported on 2026-09-17 holds that axon twice, with
identical rows. The panels therefore look before they write, and let the
user replace the rows instead.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PyQt5 import QtWidgets

from tools.results_table import duplicate_rows

# (path, rows, key columns that say two rows describe the same thing)
ExportTarget = Tuple[str, Sequence[Dict[str, Any]], Sequence[str]]

WRITE = "write"
REPLACE = "replace"
CANCEL = "cancel"


def _sentence(path: str, numbers: List[int]) -> str:
    name = os.path.basename(path)
    where = ", ".join(str(n) for n in numbers[:6])
    if len(numbers) > 6:
        where += f" ... and {len(numbers) - 6} more"
    rows = "row" if len(numbers) == 1 else "rows"
    return f"{name}: {rows} {where}"


def resolve_duplicates(
    parent: Optional[QtWidgets.QWidget],
    targets: Sequence[ExportTarget],
) -> str:
    """
    WRITE, REPLACE or CANCEL for this export.

    WRITE when no table holds these rows yet, so nothing is asked. When
    one does, the user chooses: replacing rewrites those rows in place
    (what a re-export after fixing something means), adding keeps both
    copies, and cancelling writes nothing at all.
    """
    found = []
    for path, rows, keys in targets:
        if not rows:
            continue
        numbers = duplicate_rows(path, rows, keys)
        if numbers:
            found.append(_sentence(path, numbers))
    if not found:
        return WRITE

    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Warning)
    box.setWindowTitle("This axon is already in the table")
    box.setText("These rows describe what is about to be written:\n\n"
                + "\n".join(found))
    box.setInformativeText(
        "Replace them to write this export in their place. Adding leaves "
        "both copies, and every statistic over the table will then count "
        "this axon twice.")
    replace = box.addButton("Replace", QtWidgets.QMessageBox.AcceptRole)
    add = box.addButton("Add anyway", QtWidgets.QMessageBox.DestructiveRole)
    box.addButton(QtWidgets.QMessageBox.Cancel)
    box.setDefaultButton(replace)
    box.exec_()
    clicked = box.clickedButton()
    if clicked is replace:
        return REPLACE
    if clicked is add:
        return WRITE
    return CANCEL
