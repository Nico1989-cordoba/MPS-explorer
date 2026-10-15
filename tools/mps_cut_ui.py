# -*- coding: utf-8 -*-
"""
The widgets of the cut-cluster rule (decision P-R59, docs/cut_clusters.md):
the "Pick boundary" item of the MPS window's parameter strip and the
"Pick boundary" dialog.

The strip item says how many of the kept clusters the Picasso pick's edge
cut ("k of K (x %)") or why the rule was not applied, holds the switch
"Exclude cut clusters" (off by default) and opens the dialog. The dialog
is where the pick is chosen when more than one fits, a pick file is
pointed to by hand, and the NeNA is measured on the whole-field file.
Everything they show comes from the analysis (``AxonAnalysis.cut``) or
from the main window's ``CutSession``; the decisions are made in
``tools.mps_cut_clusters`` and ``tools.mps_picks``, which have no Qt.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from PyQt5 import QtCore, QtWidgets

from tools import mps_cut_clusters as cut_rule
from tools.mps_plot_style import verdict

__all__ = ["CutStrip", "PickBoundaryDialog", "STRIP_TIP", "EXCLUDE_TIP", "DOC"]

DOC = "docs/cut_clusters.md"
STRIP_TIP = (
    "Clusters cut by the Picasso pick.\n\n"
    "The per-axon file is what a pick drawn in Picasso Render kept of the\n"
    "whole field. Where the pick's edge runs through a cluster, part of\n"
    "its localizations are missing from the file, and the cluster is\n"
    "measured smaller than it is. A kept cluster is counted as cut when\n"
    "any of its localizations lies within 1 NeNA (the localization\n"
    "precision) of the pick's edge.\n\n"
    "'k of K (x %)': k cut among the K clusters the curation kept.\n"
    "Without the pick's edge or a NeNA the rule is not applied, and the\n"
    "table says so: nothing is guessed. Why 1 NeNA, and how well it\n"
    f"finds cut clusters on simulated axons: {DOC}.")
EXCLUDE_TIP = (
    "Leave the clusters cut by the pick out of the analysis (perimeter,\n"
    "areas, 1NN, occupancy, randomization), as the curation leaves out\n"
    "its own. Off by default: the cut clusters are then flagged, counted\n"
    "and drawn, and measured as they are in the file. The axon table says\n"
    f"which one was done (cut_excluded). {DOC}.")


class CutStrip(QtWidgets.QFrame):
    """The strip item: count, switch, dialog button."""

    exclude_toggled = QtCore.pyqtSignal(bool)
    pick_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("group_cut")
        # Two short lines, so the table's column keeps its width (the map keeps its share of the window).
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(4)
        head = QtWidgets.QLabel("<b>Pick</b>")
        head.setObjectName("group_cut_title")
        head.setToolTip(STRIP_TIP)
        top.addWidget(head)
        self.label = QtWidgets.QLabel("")
        self.label.setObjectName("lbl_cut")
        self.label.setToolTip(STRIP_TIP)
        self.label.setWordWrap(True)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        top.addWidget(self.label, 1)
        outer.addLayout(top)
        lay = QtWidgets.QHBoxLayout()
        lay.setSpacing(4)
        self.chk_exclude = QtWidgets.QCheckBox("Exclude cut clusters")
        self.chk_exclude.setObjectName("chk_exclude_cut")
        self.chk_exclude.setToolTip(EXCLUDE_TIP)
        self.chk_exclude.toggled.connect(self._on_toggled)
        lay.addWidget(self.chk_exclude)
        self.btn_pick = QtWidgets.QPushButton("Pick boundary...")
        self.btn_pick.setObjectName("btn_pick_boundary")
        self.btn_pick.setToolTip(
            "Which Picasso pick this file was cut with, and the NeNA the rule\n"
            "uses: choose the pick when more than one fits, point to the\n"
            "pick file by hand, or measure the NeNA on the whole-field file.")
        self.btn_pick.clicked.connect(self.pick_requested.emit)
        lay.addWidget(self.btn_pick)
        lay.addStretch(1)
        outer.addLayout(lay)

    def _on_toggled(self, on: bool) -> None:
        self.exclude_toggled.emit(bool(on))

    def sync(self, report: Optional[cut_rule.CutReport], exclude: bool, before_words: str = "") -> None:
        """Show ``report`` (the analysis'), or ``before_words`` (the main
        window's session) when there is no analysis yet."""
        self.chk_exclude.blockSignals(True)
        self.chk_exclude.setChecked(bool(exclude))
        self.chk_exclude.blockSignals(False)
        if report is None and before_words:
            text, kind, tip = before_words, "dim", before_words
        elif report is None:
            text, kind = cut_rule.STATUS_NO_BOUNDARY, "dim"
            tip = "no Picasso pick boundary was given for this file"
        elif report.evaluated:
            text = report.count_words() + (" cut, excluded" if report.excluded else " cut")
            kind = "warn" if report.cut_labels else "good"
            tip = report.note_words()
        elif "ambiguous" in report.reason:
            # more than one pick fits: the user is asked, here and in the table, never answered for
            text, kind, tip = "more than one pick fits: choose one (Pick boundary...)", "warn", report.reason
        else:
            text, kind, tip = report.status, "dim", report.reason
        self.label.setText(text)
        self.label.setStyleSheet(f"color: {verdict(kind, dark=False)};")
        self.label.setToolTip(f"{tip}\n\n{STRIP_TIP}")


class PickBoundaryDialog(QtWidgets.QDialog):
    """Choose the pick, point to a pick file, measure the NeNA.

    ``host`` is the main window; it must provide ``cut_session_now()``,
    ``locs1``, ``choose_pick_file(path)``, ``choose_pick_candidate(i)``,
    ``set_pick_aside()``, ``measure_whole_field_nena(path)``,
    ``set_exclude_cut(on)`` and ``exclude_cut_clusters``. Each action
    re-runs the analysis there and refreshes this dialog.
    """

    def __init__(self, host: Any, parent: Optional[QtWidgets.QWidget] = None,
                 ask_file: Optional[Callable[[str, str, str], str]] = None) -> None:
        super().__init__(parent)
        self.host = host
        self.ask_file = ask_file or self._ask_file
        self.setWindowTitle("Pick boundary - clusters cut by the Picasso pick")
        self.resize(640, 460)
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "A kept cluster is counted as <b>cut</b> when any of its localizations lies within "
            "<b>1 NeNA</b> of the edge of the Picasso pick this file was cut with. The edge is read from the "
            "file Picasso's <i>Save picks</i> writes; the NeNA is measured on the whole-field, unlinked file. "
            f"Without either the rule is not applied, and the table says so. See {DOC}.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        box = QtWidgets.QGroupBox("1. The pick")
        bl = QtWidgets.QVBoxLayout(box)
        self.lbl_pick = QtWidgets.QLabel("")
        self.lbl_pick.setObjectName("lbl_pick_status")
        self.lbl_pick.setWordWrap(True)
        self.lbl_pick.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        bl.addWidget(self.lbl_pick)
        self.list_candidates = QtWidgets.QListWidget()
        self.list_candidates.setObjectName("list_pick_candidates")
        self.list_candidates.setMaximumHeight(110)
        bl.addWidget(self.list_candidates)
        row = QtWidgets.QHBoxLayout()
        self.btn_use = QtWidgets.QPushButton("Use the selected pick")
        self.btn_use.setObjectName("btn_use_pick")
        self.btn_use.clicked.connect(self._on_use)
        row.addWidget(self.btn_use)
        self.btn_file = QtWidgets.QPushButton("Choose pick file...")
        self.btn_file.setObjectName("btn_choose_pick_file")
        self.btn_file.setToolTip("Point to the .yaml file Picasso Render's 'Save picks' wrote for this field.")
        self.btn_file.clicked.connect(self._on_file)
        row.addWidget(self.btn_file)
        self.btn_aside = QtWidgets.QPushButton("Do not use a boundary")
        self.btn_aside.setObjectName("btn_pick_aside")
        self.btn_aside.setToolTip("Apply no rule to this file: the table then says the boundary was set aside.")
        self.btn_aside.clicked.connect(self._on_aside)
        row.addWidget(self.btn_aside)
        row.addStretch(1)
        bl.addLayout(row)
        lay.addWidget(box)

        box2 = QtWidgets.QGroupBox("2. The NeNA of channel 1")
        nl = QtWidgets.QVBoxLayout(box2)
        self.lbl_nena = QtWidgets.QLabel("")
        self.lbl_nena.setObjectName("lbl_nena_status")
        self.lbl_nena.setWordWrap(True)
        self.lbl_nena.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        nl.addWidget(self.lbl_nena)
        self.btn_nena = QtWidgets.QPushButton("Measure on the whole-field file...")
        self.btn_nena.setObjectName("btn_measure_nena")
        self.btn_nena.setToolTip(
            "Choose the whole-field, UNLINKED localization file this axon was picked from. Its NeNA is measured\n"
            "and kept for this folder, so the other axons of the acquisition use the same value.")
        self.btn_nena.clicked.connect(self._on_nena)
        nl.addWidget(self.btn_nena, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(box2)

        box3 = QtWidgets.QGroupBox("3. What is done with the cut clusters")
        el = QtWidgets.QVBoxLayout(box3)
        self.chk_exclude = QtWidgets.QCheckBox("Exclude cut clusters from the analysis")
        self.chk_exclude.setObjectName("chk_dialog_exclude_cut")
        self.chk_exclude.setToolTip(EXCLUDE_TIP)
        self.chk_exclude.toggled.connect(self._on_exclude)
        el.addWidget(self.chk_exclude)
        lay.addWidget(box3)

        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        lay.addWidget(close)
        self.refresh()

    # ------------------------------------------------------------------ state
    def refresh(self) -> None:
        session = self.host.cut_session_now()
        self.list_candidates.clear()
        if session is None:
            self.lbl_pick.setText("No Picasso localization file is loaded in channel 1: the rule needs one "
                                  "(its x and y in camera pixels).")
            for w in (self.btn_use, self.btn_file, self.btn_aside, self.btn_nena, self.list_candidates):
                w.setEnabled(False)
            self.lbl_nena.setText("")
        else:
            m = session.match
            if session.set_aside:
                text = "Set aside by hand: no rule for this file."
            elif m.status == "matched":
                text = f"<b>Found</b>: {m.message}" if not m.by_hand else f"<b>Chosen</b>: {m.message}"
            elif m.status == "ambiguous":
                text = f"<b>Choose one</b>: {m.message}."
            else:
                text = f"<b>Not found</b>: {m.message}. Point to the pick file by hand."
            if m.files:
                text += "<br>Pick files read: " + ", ".join(os.path.basename(f) for f in m.files[:6]) + (
                    " ..." if len(m.files) > 6 else "")
            self.lbl_pick.setText(text)
            for cand in m.candidates:
                self.list_candidates.addItem(cand.describe())
            self.list_candidates.setVisible(bool(m.candidates))
            self.btn_use.setVisible(bool(m.candidates))
            for w in (self.btn_use, self.btn_file, self.btn_aside, self.btn_nena, self.list_candidates):
                w.setEnabled(True)
            nena = session.nena
            if session.boundary is None:
                self.lbl_nena.setText("Measured once a pick boundary is known.")
            elif nena is not None and nena.measured:
                self.lbl_nena.setText(f"<b>{nena.nena_nm:.2f} nm</b> - {nena.words}")
            elif nena is not None:
                self.lbl_nena.setText(f"<b>Unknown</b>: {nena.reason}. Measure it on the whole-field file.")
            else:
                self.lbl_nena.setText("Not measured yet: it is measured when the analysis runs.")
        self.chk_exclude.blockSignals(True)
        self.chk_exclude.setChecked(bool(getattr(self.host, "exclude_cut_clusters", False)))
        self.chk_exclude.blockSignals(False)

    # ------------------------------------------------------------------ actions
    def _ask_file(self, title: str, start: str, pattern: str) -> str:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, start, pattern)
        return str(path or "")

    def _start_dir(self) -> str:
        session = self.host.cut_session_now()
        return os.path.dirname(session.path) if session is not None else ""

    def _on_use(self) -> None:
        row = self.list_candidates.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "Pick boundary", "Select one of the picks first.")
            return
        self.host.choose_pick_candidate(row)
        self.refresh()

    def _on_file(self) -> None:
        path = self.ask_file("Picasso pick file (Save picks)", self._start_dir(), "Picasso picks (*.yaml *.yml)")
        if path:
            self.host.choose_pick_file(path)
            self.refresh()

    def _on_aside(self) -> None:
        self.host.set_pick_aside()
        self.refresh()

    def _on_nena(self) -> None:
        start = os.path.dirname(self._start_dir()) or self._start_dir()
        path = self.ask_file("Whole-field, unlinked localization file", start, "Picasso localizations (*.hdf5)")
        if not path:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            source = self.host.measure_whole_field_nena(path)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if source is not None and not source.measured:
            QtWidgets.QMessageBox.warning(self, "NeNA", f"No NeNA from {os.path.basename(path)}: {source.reason}.")
        self.refresh()

    def _on_exclude(self, on: bool) -> None:
        self.host.set_exclude_cut(bool(on))
        self.refresh()
