# -*- coding: utf-8 -*-
"""
The one export: where this axon's tables go, and which of them to write.

Before this, three buttons in two windows each wrote a file from its own
state, and it was on the user to press all of them, in the right order,
after re-running whatever had changed. This asks once, writes everything
from one state, and says what it wrote.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from PyQt5 import QtWidgets

from tools.axon_export import table_paths
from tools.mps_identity import AxonIdentity

HINT_STYLE = "color: #666666; font-size: 11px;"
WARN_STYLE = "color: #a04000;"


class ExportAxonDialog(QtWidgets.QDialog):
    """Which axon, where, and which of the three tables."""

    def __init__(
        self,
        path: str,
        identity: Optional[AxonIdentity],
        *,
        state: str = "",
        warnings: Optional[List[str]] = None,
        n_clusters: int = 0,
        n_localizations: int = 0,
        on_edit_identity: Optional[Callable[[], Optional[AxonIdentity]]] = None,
        with_clusters: bool = False,
        with_localizations: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export this axon")
        self._identity = identity
        self._on_edit_identity = on_edit_identity

        layout = QtWidgets.QVBoxLayout(self)

        # --- who ---------------------------------------------------------
        who = QtWidgets.QHBoxLayout()
        self.label_identity = QtWidgets.QLabel("")
        self.label_identity.setWordWrap(True)
        who.addWidget(self.label_identity, 1)
        self.btn_identity = QtWidgets.QPushButton("Change...")
        self.btn_identity.setToolTip(
            "Which genotype, protein, slide, ROI and axon these rows say "
            "they are.")
        self.btn_identity.clicked.connect(self._edit_identity)
        self.btn_identity.setEnabled(on_edit_identity is not None)
        who.addWidget(self.btn_identity)
        layout.addLayout(who)

        if state:
            label_state = QtWidgets.QLabel(state)
            label_state.setWordWrap(True)
            label_state.setStyleSheet(HINT_STYLE)
            layout.addWidget(label_state)

        # --- where -------------------------------------------------------
        layout.addWidget(QtWidgets.QLabel("Table to add this axon to:"))
        where = QtWidgets.QHBoxLayout()
        self.edit_path = QtWidgets.QLineEdit(path)
        self.edit_path.setToolTip(
            "One table for a whole folder of axons: each export adds a row. "
            "A table of another analysis is refused, not overwritten.")
        self.edit_path.textChanged.connect(self._refresh)
        where.addWidget(self.edit_path, 1)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        where.addWidget(browse)
        layout.addLayout(where)

        # --- which tables -------------------------------------------------
        self.chk_clusters = QtWidgets.QCheckBox(
            f"Also one row per cluster ({n_clusters} rows)")
        self.chk_clusters.setToolTip(
            "Area, effective radius, 1NN and which neighbour, place on the "
            "contour, and what the axoplasm panel decided, for each "
            "cluster.")
        self.chk_clusters.setChecked(with_clusters)
        self.chk_clusters.setEnabled(n_clusters > 0)
        self.chk_clusters.toggled.connect(self._refresh)
        layout.addWidget(self.chk_clusters)

        self.chk_localizations = QtWidgets.QCheckBox(
            f"Also one row per localization ({n_localizations:,} rows)")
        self.chk_localizations.setToolTip(
            "Every localization of the selection: where it is, whether the "
            "axial slab holds it, and which cluster it belongs to. It is by "
            "far the largest of the three.")
        self.chk_localizations.setChecked(with_localizations)
        self.chk_localizations.setEnabled(n_localizations > 0)
        self.chk_localizations.toggled.connect(self._refresh)
        layout.addWidget(self.chk_localizations)

        self.label_files = QtWidgets.QLabel("")
        self.label_files.setStyleSheet(HINT_STYLE)
        self.label_files.setWordWrap(True)
        layout.addWidget(self.label_files)

        self.label_warnings = QtWidgets.QLabel("")
        self.label_warnings.setStyleSheet(WARN_STYLE)
        self.label_warnings.setWordWrap(True)
        layout.addWidget(self.label_warnings)
        self._warnings = list(warnings or [])

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self)
        self._ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self._ok.setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh()

    # -- state -------------------------------------------------------------

    def path(self) -> str:
        return self.edit_path.text().strip()

    def identity(self) -> Optional[AxonIdentity]:
        return self._identity

    def wants_clusters(self) -> bool:
        return self.chk_clusters.isChecked()

    def wants_localizations(self) -> bool:
        return self.chk_localizations.isChecked()

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Table to add this axon to", self.path(),
            "CSV Files (*.csv)")
        if path:
            self.edit_path.setText(path)

    def _edit_identity(self) -> None:
        if self._on_edit_identity is None:
            return
        chosen = self._on_edit_identity()
        if chosen is not None:
            self._identity = chosen
            self._refresh()

    def _refresh(self) -> None:
        identity = self._identity
        if identity is None or not identity.describe() or \
                identity.describe() == "nothing filled in":
            self.label_identity.setText(
                "<b>This axon has no identity yet.</b>")
        else:
            self.label_identity.setText(f"<b>{identity.describe()}</b>")

        path = self.path()
        if path:
            paths = table_paths(path)
            names = [os.path.basename(paths["axon"])]
            if self.wants_clusters():
                names.append(os.path.basename(paths["clusters"]))
            if self.wants_localizations():
                names.append(os.path.basename(paths["localizations"]))
            self.label_files.setText("Writes: " + ", ".join(names))
        else:
            self.label_files.setText("")

        lines = list(self._warnings)
        if identity is not None:
            lines = identity.warnings() + lines
        self.label_warnings.setText("\n".join(lines))
        if self._ok is not None:
            self._ok.setEnabled(bool(path))


def suggested_path(source: str, folder: str = "") -> str:
    """Where to propose writing, given the axon's own file.

    The folder of the last export when there is one: a table is meant to
    gather a folder's worth of axons, and proposing the axon's own folder
    every time is how one table per axon happens instead.
    """
    stem = os.path.splitext(os.path.basename(str(source or "axon")))[0]
    if folder and os.path.isdir(folder):
        return os.path.join(folder, "mps_axons.csv")
    return os.path.join(os.path.dirname(str(source)) or os.getcwd(),
                        f"{stem}_mps_axons.csv")


def describe_state(analysis, discard, panel_open: bool) -> str:
    """One line saying what state this export describes."""
    parts = []
    if analysis is not None:
        parts.append(f"{analysis.n_clusters_kept} clusters kept")
    if discard is not None:
        parts.append(f"{len(discard.discarded_labels)} discarded "
                     f"({discard.discard_registration or 'not registered'})")
    else:
        parts.append("no discard applied")
    parts.append("with the axoplasm panel" if panel_open
                 else "without the axoplasm panel")
    return "; ".join(parts) + "."


def written(paths: Dict[str, str], counts: Dict[str, int],
            replaced: bool) -> str:
    """The message that says what was written."""
    verb = "Replaced" if replaced else "Added"
    lines = []
    for key in ("axon", "clusters", "localizations"):
        if key in counts:
            lines.append(f"{verb} {counts[key]} row(s) in {paths[key]}")
    return "\n".join(lines)
