# -*- coding: utf-8 -*-
"""
The panel that shows which axon is about to be exported, and lets the
user correct it.

``tools.mps_identity`` proposes the genotype, the protein, the slide, the
ROI and the axon from the path and leaves empty whatever the patterns do
not match. This is where that proposal is shown -- each field beside the
reason it holds what it holds -- before anything is written. Typing here
is the only way a field gets a value the path does not contain.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from PyQt5 import QtWidgets

from tools.mps_plot_style import marked, verdict
from tools.mps_identity import (
    DEFAULT_PATTERNS, FIELD_HINTS, FIELD_LABELS, FIELDS, AxonIdentity,
    Proposal, check_patterns, propose,
)

# Read on the application's white, so the darker set of the verdict
# colours, and each message carries the mark for its kind: darkening
# brings a warning and a failure to 2 apart for a deuteranope.
HINT_STYLE = f"color: {verdict('dim', dark=False)}; font-size: 11px;"
WARN_STYLE = f"color: {verdict('warn', dark=False)};"
ERROR_STYLE = f"color: {verdict('bad', dark=False)};"


class PatternsDialog(QtWidgets.QDialog):
    """Where the five patterns are written, with what they find here."""

    def __init__(self, patterns: Dict[str, str], path: str,
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("How the fields are read off the path")
        self._path = path
        self.edits: Dict[str, QtWidgets.QLineEdit] = {}
        self.previews: Dict[str, QtWidgets.QLabel] = {}

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Each field is a regular expression searched in the whole path, "
            "with '/' as the separator and upper or lower case alike. The "
            "first capturing group is the value.\n"
            "The slide is left empty on purpose: with no pattern it is taken "
            "as the folder above the ROI folder.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        for row, name in enumerate(FIELDS):
            label = QtWidgets.QLabel(FIELD_LABELS[name])
            label.setToolTip(FIELD_HINTS[name])
            edit = QtWidgets.QLineEdit(
                patterns.get(name, DEFAULT_PATTERNS[name]))
            edit.setToolTip(FIELD_HINTS[name])
            edit.textChanged.connect(self._refresh)
            preview = QtWidgets.QLabel("")
            preview.setStyleSheet(HINT_STYLE)
            preview.setWordWrap(True)
            grid.addWidget(label, row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(preview, row, 2)
            self.edits[name] = edit
            self.previews[name] = preview
        layout.addLayout(grid)

        self.where = QtWidgets.QLabel(f"Tried on: {path}" if path else "")
        self.where.setStyleSheet(HINT_STYLE)
        self.where.setWordWrap(True)
        layout.addWidget(self.where)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save
            | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.RestoreDefaults, self)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        buttons.button(
            QtWidgets.QDialogButtonBox.RestoreDefaults
        ).clicked.connect(self._restore)
        layout.addWidget(buttons)
        self._buttons = buttons
        self._refresh()

    # -- state ------------------------------------------------------------

    def patterns(self) -> Dict[str, str]:
        return {n: self.edits[n].text().strip() for n in FIELDS}

    def _restore(self) -> None:
        for name in FIELDS:
            self.edits[name].setText(DEFAULT_PATTERNS[name])

    def _refresh(self) -> None:
        pats = self.patterns()
        broken = check_patterns(pats)
        result = propose(self._path, patterns=pats) if self._path else None
        for name in FIELDS:
            if name in broken:
                self.previews[name].setStyleSheet(ERROR_STYLE)
                self.previews[name].setText(marked(
                    "bad", f"not a valid pattern: {broken[name]}"))
                continue
            self.previews[name].setStyleSheet(HINT_STYLE)
            if result is None:
                self.previews[name].setText("")
                continue
            value = getattr(result.identity, name)
            self.previews[name].setText(
                f"finds '{value}'" if value else f"finds nothing "
                f"({result.explain(name)})")
        save = self._buttons.button(QtWidgets.QDialogButtonBox.Save)
        if save is not None:
            save.setEnabled(not broken)

    def _on_save(self) -> None:
        broken = check_patterns(self.patterns())
        if broken:
            QtWidgets.QMessageBox.warning(
                self, "That pattern cannot be used",
                "\n".join(f"{FIELD_LABELS[n]}: {why}"
                          for n, why in broken.items()))
            return
        self.accept()


class IdentityDialog(QtWidgets.QDialog):
    """Which axon this is: proposed from the path, corrected here."""

    def __init__(self, path: str, patterns: Optional[Dict[str, str]] = None,
                 remembered: Optional[AxonIdentity] = None,
                 remembered_folder: str = "",
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Which axon is this?")
        self._path = path
        self._patterns = dict(patterns or {})
        self._remembered = remembered
        self._remembered_folder = remembered_folder
        self.edits: Dict[str, QtWidgets.QLineEdit] = {}
        self.hints: Dict[str, QtWidgets.QLabel] = {}

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "These five columns go into every table this axon is exported "
            "to. They are read off the path; what the path does not say is "
            "left empty, and only you can fill it in.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        where = QtWidgets.QLabel(path)
        where.setStyleSheet(HINT_STYLE)
        where.setWordWrap(True)
        layout.addWidget(where)

        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1, 1)
        for row, name in enumerate(FIELDS):
            label = QtWidgets.QLabel(FIELD_LABELS[name])
            label.setToolTip(FIELD_HINTS[name])
            edit = QtWidgets.QLineEdit()
            edit.setToolTip(FIELD_HINTS[name])
            edit.textChanged.connect(self._refresh_warning)
            hint = QtWidgets.QLabel("")
            hint.setStyleSheet(HINT_STYLE)
            hint.setWordWrap(True)
            grid.addWidget(label, 2 * row, 0)
            grid.addWidget(edit, 2 * row, 1)
            grid.addWidget(hint, 2 * row + 1, 1)
            self.edits[name] = edit
            self.hints[name] = hint
        layout.addLayout(grid)

        self.warning = QtWidgets.QLabel("")
        self.warning.setStyleSheet(WARN_STYLE)
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)

        bottom = QtWidgets.QHBoxLayout()
        self.btn_patterns = QtWidgets.QPushButton("Patterns...")
        self.btn_patterns.setToolTip(
            "Change how these fields are read off a path. Written once for "
            "your own folders, and kept for the next session.")
        self.btn_patterns.clicked.connect(self._edit_patterns)
        bottom.addWidget(self.btn_patterns)
        bottom.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self)
        ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        ok.setText("Use these")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        bottom.addWidget(buttons)
        layout.addLayout(bottom)

        self._apply(self._propose())

    # -- state ------------------------------------------------------------

    def _propose(self) -> Proposal:
        return propose(self._path, patterns=self._patterns,
                       remembered=self._remembered,
                       remembered_folder=self._remembered_folder)

    def _apply(self, proposal: Proposal) -> None:
        for name in FIELDS:
            self.edits[name].setText(getattr(proposal.identity, name))
            self.hints[name].setText(proposal.explain(name))
        # What the proposal put there, so a value the user types instead is
        # recognisable later and not overwritten by a new pattern.
        self._applied = {n: self.edits[n].text() for n in FIELDS}
        self._refresh_warning()

    def typed_by_hand(self) -> Tuple[str, ...]:
        """The fields whose value the user wrote rather than accepted."""
        applied = getattr(self, "_applied", {})
        return tuple(n for n in FIELDS
                     if self.edits[n].text().strip() != applied.get(n, ""))

    def identity(self) -> AxonIdentity:
        """What the fields say now."""
        return AxonIdentity(**{n: self.edits[n].text() for n in FIELDS})

    def patterns(self) -> Dict[str, str]:
        """The patterns, changed or not, to be stored by the caller."""
        return dict(self._patterns)

    def _refresh_warning(self) -> None:
        missing = self.identity().missing
        if not missing:
            self.warning.setText("")
            return
        names = ", ".join(FIELD_LABELS[n].lower() for n in missing)
        self.warning.setText(marked(
            "warn",
            f"Left empty: {names}. The rows are written all the same, with "
            f"those cells empty, and a comparison between groups cannot use "
            f"them."))

    def _edit_patterns(self) -> None:
        dialog = PatternsDialog(self._patterns, self._path, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        # Only what differs from the default is stored, so a later change
        # of the defaults reaches the users who never wrote their own.
        chosen = dialog.patterns()
        self._patterns = {n: v for n, v in chosen.items()
                          if v != DEFAULT_PATTERNS[n]}
        by_hand = self.typed_by_hand()
        typed = {n: self.edits[n].text().strip() for n in FIELDS}
        proposal = self._propose()
        self._apply(proposal)
        # What the user wrote is not thrown away by a new pattern: it is put
        # back, and says so.
        for name in by_hand:
            if typed[name]:
                self.edits[name].setText(typed[name])
                self.hints[name].setText("typed here")
        self._refresh_warning()


def ask_identity(
    parent: Optional[QtWidgets.QWidget],
    path: str,
    patterns: Optional[Dict[str, str]] = None,
    remembered: Optional[AxonIdentity] = None,
    remembered_folder: str = "",
) -> Optional[Tuple[AxonIdentity, Dict[str, str]]]:
    """
    Show the panel and return (identity, patterns), or None if cancelled.

    The caller stores both: the patterns so they are not written twice, and
    the identity so the next axon of the same slide starts from it.
    """
    dialog = IdentityDialog(path, patterns=patterns, remembered=remembered,
                            remembered_folder=remembered_folder,
                            parent=parent)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.identity(), dialog.patterns()


def summarise(identity: Optional[AxonIdentity], path: str = "") -> str:
    """One line for a status bar or a message box."""
    if identity is None or not any(getattr(identity, n) for n in FIELDS):
        return f"no identity yet for {os.path.basename(path)}" if path \
            else "no identity yet"
    return identity.describe()
