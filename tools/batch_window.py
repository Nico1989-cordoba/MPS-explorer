# -*- coding: utf-8 -*-
"""
The batch, from the interface: a folder of picked axons analysed with the
main window's settings, written to the same table the axon window
writes, and summarised by group.

Two tabs. "Analyse a folder" finds the files, shows who each one is --
read off its path, and correctable here for many files at once -- runs
them in the background and writes their rows. "Compare groups" reads
those rows, or any exported table, and shows one column per group at two
levels, per axon and per nesting unit, with how much the axons of one
unit resemble each other. It computes no p-value: see tools/mps_batch.py.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_batch as mb
from tools.mps_identity import (
    DEFAULT_PATTERNS, FIELD_HINTS, FIELD_LABELS, FIELDS, AxonIdentity,
    propose,
)
from tools.mps_plot_style import marked, verdict

# Columns of the table of files.
COL_USE = 0
COL_FILE = 1
COL_FIRST_FIELD = 2
COL_PIXEL = COL_FIRST_FIELD + len(FIELDS)
COL_RESULT = COL_PIXEL + 1

HINT_STYLE = f"color: {verdict('dim', dark=False)};"
WARN_STYLE = f"color: {verdict('warn', dark=False)};"

# How the origin of each proposed field reads in its cell's tooltip.
_ORIGIN_TEXT = {
    "pattern": "read off the path by its pattern",
    "folder above the ROI": "the folder above the ROI folder",
    "not found": "the pattern found nothing in this path",
    "no pattern": "no pattern is set for this field: type it",
    "bad pattern": "the pattern is not a valid regular expression",
}

# Nothing grouped: every axon in one group.
NO_GROUPS = "(all axons together)"


class _Relay(QtCore.QObject):
    """Carries the worker's news to the window's thread."""

    done = QtCore.pyqtSignal(int, object)
    finished = QtCore.pyqtSignal()


def _item(text: str, editable: bool = False,
          tip: str = "") -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    flags = QtCore.Qt.ItemFlags(QtCore.Qt.ItemFlag.ItemIsSelectable)
    flags |= QtCore.Qt.ItemFlag.ItemIsEnabled
    if editable:
        flags |= QtCore.Qt.ItemFlag.ItemIsEditable
    item.setFlags(flags)
    if tip:
        item.setToolTip(tip)
    return item


def ask_pixel_size(parent: Optional[QtWidgets.QWidget], folder: str,
                   candidates: Sequence[Any] = (),
                   notes: Sequence[str] = ()) -> Optional[float]:
    """
    Ask for the pixel size of the files of one folder that lost theirs.

    Nothing is filled in: a number from another microscope would rescale
    every distance this program measures, and one Enter would accept it.
    """
    from tools import mps_pixel_size

    found = (mps_pixel_size.describe(candidates, notes) + "\n\n"
             "Type the one that belongs to these files, in nanometres.\n\n"
             if candidates else
             "These files carry no pixel size, and nothing in the folders\n"
             "around them records one.\n\n"
             "Type the effective pixel size in nanometres -- the camera\n"
             "pixel divided by the magnification. Nothing is filled in on\n"
             "purpose.\n\n")
    text, ok = QtWidgets.QInputDialog.getText(
        parent, "Pixel size required",
        f"{folder}\n\n{found}It is remembered for this folder.\n"
        f"Leave it empty to leave these files out of the batch.")
    if not ok or not text.strip():
        return None
    try:
        value = float(text.strip().replace(",", "."))
    except ValueError:
        return None
    return value if 1.0 <= value <= 1000.0 else None


class SetFieldDialog(QtWidgets.QDialog):
    """One identity field, one value, for every selected file."""

    def __init__(self, n_files: int, parent: Optional[QtWidgets.QWidget]):
        super().__init__(parent)
        self.setWindowTitle("Set a field for the selected files")
        layout = QtWidgets.QFormLayout(self)
        self.combo = QtWidgets.QComboBox()
        for name in FIELDS:
            self.combo.addItem(FIELD_LABELS[name], name)
        self.combo.setToolTip("Which column of the identity to fill in.")
        self.edit = QtWidgets.QLineEdit()
        self.edit.setToolTip(
            "The value, the same for every selected file. Leave it empty "
            "to clear the field.")
        self.combo.currentIndexChanged.connect(self._hint)
        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(HINT_STYLE)
        layout.addRow(QtWidgets.QLabel(
            f"For the {n_files} selected file(s):"))
        layout.addRow("Field", self.combo)
        layout.addRow("Value", self.edit)
        layout.addRow(self.hint)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self._hint()

    def _hint(self) -> None:
        self.hint.setText(FIELD_HINTS[self.field()])

    def field(self) -> str:
        return str(self.combo.currentData())

    def value(self) -> str:
        return self.edit.text().strip()


class BatchWindow(QtWidgets.QMainWindow):
    """Analyse a folder of axons at once, and compare groups."""

    def __init__(self, settings: Any,
                 save: Optional[Callable[..., Any]] = None,
                 sync: Optional[Callable[[], None]] = None,
                 parent: Optional[QtWidgets.QWidget] = None):
        """
        ``settings`` is the main window's own MPSSettings object, read when
        a batch starts; ``save`` stores it. ``sync`` brings it up to date
        with what the main window's boxes say -- a value typed there and
        not yet clustered with is not in the settings until then.
        """
        super().__init__(parent)
        self.setWindowTitle("Batch: a folder of axons")
        self.resize(1250, 780)
        self.settings = settings
        self._save = save
        self._sync = sync
        self.plan: Optional[mb.BatchPlan] = None
        # What the run produced, by row of the table of files.
        self.records: Dict[int, mb.AxonRecord] = {}
        # Why a file was not analysed, for the log.
        self.not_analysed: Dict[str, str] = {}
        # Cells the user typed, kept when the patterns change.
        self._typed: Set[Tuple[int, str]] = set()
        # The rows of an exported table opened in the second tab.
        self.table_records: List[mb.AxonRecord] = []
        self.table_notes: List[str] = []
        self.comparison: Optional[mb.GroupComparison] = None

        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._close_when_done = False
        self._relay = _Relay(self)
        self._relay.done.connect(self._on_done)
        self._relay.finished.connect(self._on_finished)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_run_tab(), "Analyse a folder")
        tabs.addTab(self._build_compare_tab(), "Compare groups")
        self.tabs = tabs
        # An identity typed in the first tab changes the groups of the
        # second: it is recomputed whenever it is looked at.
        tabs.currentChanged.connect(self._refresh_comparison)
        self.setCentralWidget(tabs)
        self._refresh_buttons()

    # ------------------------------------------------------------------
    # Tab 1: the folder
    # ------------------------------------------------------------------

    def _build_run_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "Every picked-axon file under a folder is analysed the way the "
            "axon window analyses one, and written as one row per axon to "
            "the same table 'Export axon' writes. Check who each file is "
            "before running: the columns are read off the path, and what "
            "the path does not say is left empty.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Folder:"))
        self.edit_folder = QtWidgets.QLineEdit()
        self.edit_folder.setToolTip(
            "The folder to look in. Its subfolders are searched too, so "
            "the folder of a whole experiment can be given at once.")
        row.addWidget(self.edit_folder, 1)
        browse = QtWidgets.QPushButton("Browse...")
        browse.setToolTip("Choose the folder with the picked axons.")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        row.addWidget(QtWidgets.QLabel("File names containing:"))
        self.edit_pattern = QtWidgets.QLineEdit("axon")
        self.edit_pattern.setMaximumWidth(120)
        self.edit_pattern.setToolTip(
            "Only files whose name contains this text are taken, upper or "
            "lower case alike. 'axon' takes the picked axons and leaves the "
            "whole-field files out. Empty takes every localization file.")
        row.addWidget(self.edit_pattern)
        self.btn_find = QtWidgets.QPushButton("Find files")
        self.btn_find.setToolTip(
            "List the localization files found. Nothing is analysed yet.")
        self.btn_find.clicked.connect(self._on_find)
        row.addWidget(self.btn_find)
        layout.addLayout(row)

        self.label_plan = QtWidgets.QLabel("")
        self.label_plan.setWordWrap(True)
        layout.addWidget(self.label_plan)

        headers = (["Use", "File"] + [FIELD_LABELS[n] for n in FIELDS]
                   + ["Pixel size", "Result"])
        self.table = QtWidgets.QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.AnyKeyPressed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_FILE, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(COL_RESULT, QtWidgets.QHeaderView.Stretch)
        self.table.setColumnWidth(COL_USE, 40)
        self.table.setColumnWidth(COL_FILE, 260)
        tips = {COL_USE: "Untick a file to leave it out of the batch.",
                COL_FILE: "The file; its whole path is in the tooltip.",
                COL_PIXEL: "The file's own pixel size, or 'needs one' for "
                           "a Picasso file that lost its metadata: asked "
                           "once per folder when the batch starts.",
                COL_RESULT: "What became of the file."}
        for i, name in enumerate(FIELDS):
            tips[COL_FIRST_FIELD + i] = (FIELD_HINTS[name] + "\n\nDouble-"
                                         "click to type it for one file, or "
                                         "select several and use 'Set a "
                                         "field...'.")
        for column, tip in tips.items():
            self.table.horizontalHeaderItem(column).setToolTip(tip)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        self.btn_set = QtWidgets.QPushButton("Set a field for the selected "
                                             "files...")
        self.btn_set.setToolTip(
            "Type one value -- a genotype, an animal, a slide -- for every "
            "selected file at once. Select rows with Ctrl or Shift.")
        self.btn_set.clicked.connect(self._on_set_field)
        row.addWidget(self.btn_set)
        self.btn_patterns = QtWidgets.QPushButton("Patterns...")
        self.btn_patterns.setToolTip(
            "Change how the fields are read off a path, for your own "
            "folder names. The same patterns the axon window uses; what "
            "you typed by hand is kept.")
        self.btn_patterns.clicked.connect(self._on_patterns)
        row.addWidget(self.btn_patterns)
        row.addStretch(1)
        layout.addLayout(row)

        self.label_settings = QtWidgets.QLabel("")
        self.label_settings.setWordWrap(True)
        self.label_settings.setStyleSheet(HINT_STYLE)
        layout.addWidget(self.label_settings)

        row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setToolTip(
            "Analyse the ticked files, one after the other, in the "
            "background. A few seconds per axon; the window stays usable.")
        self.btn_run.clicked.connect(self.run)
        row.addWidget(self.btn_run)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setToolTip(
            "Stop after the axon being analysed. What was analysed is "
            "kept and can be written.")
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_cancel)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        row.addWidget(self.progress, 1)
        self.btn_export = QtWidgets.QPushButton("Write to a table...")
        self.btn_export.setToolTip(
            "Add one row per analysed axon to a table: a new one, or the "
            "one you export the axon window's axons to. A log of what "
            "became of every file is written beside it.")
        self.btn_export.clicked.connect(self._on_export)
        row.addWidget(self.btn_export)
        layout.addLayout(row)

        self.label_status = QtWidgets.QLabel("")
        self.label_status.setWordWrap(True)
        self.label_status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.label_status)
        self._show_settings()
        return page

    def _show_settings(self) -> None:
        try:
            described = mb.BatchSettings.from_settings(self.settings).describe()
        except (AttributeError, TypeError, ValueError):
            described = "the main window's settings"
        self.label_settings.setText(
            "Analysed with the main window's settings, read when Run is "
            f"pressed: {described}.\n"
            "A batch draws no ROI: clusters touching the edge are judged "
            "against the convex hull of each file's localizations, and the "
            "rows say so (edge_reference). An axon exported from the axon "
            "window with an ROI is the same axon measured another way, so "
            "a table that already holds it keeps that row, not this one.")

    # -- finding the files ----------------------------------------------

    def _browse(self) -> None:
        start = self.edit_folder.text().strip() or getattr(
            self.settings, "last_open_dir", "") or os.getcwd()
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "The folder with the picked axons", start)
        if folder:
            self.edit_folder.setText(folder)
            self._on_find()

    def _patterns(self) -> Dict[str, str]:
        return dict(getattr(self.settings, "identity_patterns", {}) or {})

    def _on_find(self) -> None:
        folder = self.edit_folder.text().strip()
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.information(
                self, "Batch", "Choose a folder that exists first.")
            return
        self.find_files(folder)

    def find_files(self, folder: str) -> mb.BatchPlan:
        """List the files under ``folder`` (also for scripts and tests)."""
        if self.running:
            raise RuntimeError("A batch is running.")
        self.edit_folder.setText(folder)
        plan = mb.plan_batch(folder, pattern=self.edit_pattern.text().strip(),
                             patterns=self._patterns())
        self.plan = plan
        self.records = {}
        self.not_analysed = {}
        self._typed = set()
        self._fill_table()
        text = plan.describe()
        if plan.duplicates:
            later = sum(len(paths) - 1 for paths in plan.duplicates.values())
            text += (f" {later} file(s) look like an acquisition already "
                     f"listed under another name and were unticked: counted "
                     f"twice, one axon would weigh twice in every "
                     f"statistic.")
        self.label_plan.setText(text)
        self.label_status.setText("")
        self._refresh_buttons()
        self._refresh_comparison()
        return plan

    def _fill_table(self) -> None:
        plan = self.plan
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if plan is None:
            self.table.blockSignals(False)
            return
        repeated: Dict[str, str] = {}
        for paths in plan.duplicates.values():
            for later in paths[1:]:
                repeated[later] = paths[0]
        self.table.setRowCount(len(plan.files))
        for row, item in enumerate(plan.files):
            use = _item("", tip="Untick to leave this file out.")
            use.setFlags(use.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(QtCore.Qt.CheckState.Unchecked if item.path in repeated
                              else QtCore.Qt.CheckState.Checked)
            self.table.setItem(row, COL_USE, use)
            rel = os.path.relpath(item.path, plan.root)
            self.table.setItem(row, COL_FILE, _item(
                os.path.basename(item.path), tip=rel))
            self._fill_identity(row)
            if item.needs_pixel_size:
                pixel = _item("needs one", tip=(
                    "This Picasso file lost its metadata. The folders "
                    "around it are searched when the batch starts, and if "
                    "they do not say, you are asked once for this folder."))
                pixel.setForeground(QtGui.QColor(verdict("warn", dark=False)))
            else:
                own = item.own_pixel_size_nm
                pixel = _item("in nm already" if own is None
                              else f"{own:g} nm", tip=(
                                  "A CSV in nanometres: no pixel size is "
                                  "needed." if own is None else
                                  "Read from the file's own metadata."))
            self.table.setItem(row, COL_PIXEL, pixel)
            result = ""
            if item.path in repeated:
                result = marked("warn", "unticked: the same acquisition as "
                                + os.path.basename(repeated[item.path]))
            self.table.setItem(row, COL_RESULT, _item(result))
        self.table.blockSignals(False)

    def _fill_identity(self, row: int) -> None:
        assert self.plan is not None
        item = self.plan.files[row]
        for i, name in enumerate(FIELDS):
            origin = item.origin.get(name, "")
            if (row, name) in self._typed:
                origin_text = "typed here"
            else:
                origin_text = _ORIGIN_TEXT.get(origin, origin)
            cell = _item(getattr(item.identity, name), editable=True,
                         tip=f"{FIELD_HINTS[name]}\n\nThis value: "
                             f"{origin_text or 'typed here'}.")
            self.table.setItem(row, COL_FIRST_FIELD + i, cell)

    def _on_item_changed(self, cell: QtWidgets.QTableWidgetItem) -> None:
        column = cell.column()
        if self.plan is None:
            return
        if not COL_FIRST_FIELD <= column < COL_FIRST_FIELD + len(FIELDS):
            return
        name = FIELDS[column - COL_FIRST_FIELD]
        self._set_field(cell.row(), name, cell.text())

    def _set_field(self, row: int, name: str, value: str) -> None:
        """One identity field of one file, here and in its row if any."""
        assert self.plan is not None
        item = self.plan.files[row]
        value = value.strip()
        setattr(item.identity, name, value)
        item.origin[name] = "typed here"
        self._typed.add((row, name))
        record = self.records.get(row)
        if record is not None:
            # The identity is not part of the analysis: a row analysed
            # before it was typed takes it as it is, and analysis_id and
            # axon_id stay what they were.
            record.identity = item.identity
            if record.row:
                record.row[name] = value or None
        self.table.blockSignals(True)
        cell = self.table.item(row, COL_FIRST_FIELD + FIELDS.index(name))
        if cell is not None and cell.text() != value:
            cell.setText(value)
        if cell is not None:
            cell.setToolTip(f"{FIELD_HINTS[name]}\n\nThis value: typed here.")
        self.table.blockSignals(False)

    def selected_rows(self) -> List[int]:
        return sorted({index.row() for index in
                       self.table.selectionModel().selectedRows()})

    def set_field(self, rows: Sequence[int], name: str, value: str) -> None:
        """Fill ``name`` in for ``rows`` (also for scripts and tests)."""
        if name not in FIELDS:
            raise ValueError(f"{name!r} is not an identity field.")
        for row in rows:
            self._set_field(row, name, value)

    def _on_set_field(self) -> None:
        rows = self.selected_rows()
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Batch", "Select the files first: click one row, then "
                "Ctrl-click or Shift-click the others.")
            return
        dialog = SetFieldDialog(len(rows), self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self.set_field(rows, dialog.field(), dialog.value())

    def _on_patterns(self) -> None:
        from tools.identity_ui import PatternsDialog

        if self.plan is None or not self.plan.files:
            QtWidgets.QMessageBox.information(
                self, "Batch", "Find the files first: the patterns are "
                "tried on one of them.")
            return
        rows = self.selected_rows()
        path = self.plan.files[rows[0] if rows else 0].path
        dialog = PatternsDialog(self._patterns(), path, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        chosen = {n: v for n, v in dialog.patterns().items()
                  if v != DEFAULT_PATTERNS[n]}
        self.settings.identity_patterns = chosen
        self._persist()
        self.apply_patterns(chosen)

    def apply_patterns(self, patterns: Dict[str, str]) -> None:
        """Read the fields again with ``patterns``, keeping what was typed."""
        if self.plan is None:
            return
        for row, item in enumerate(self.plan.files):
            proposal = propose(item.path, patterns=patterns)
            for name in FIELDS:
                if (row, name) in self._typed:
                    continue
                setattr(item.identity, name,
                        getattr(proposal.identity, name))
                item.origin[name] = proposal.origin.get(name, "")
            record = self.records.get(row)
            if record is not None and record.row:
                record.row.update(item.identity.columns())
        self.table.blockSignals(True)
        for row in range(len(self.plan.files)):
            self._fill_identity(row)
        self.table.blockSignals(False)

    def _persist(self) -> None:
        if self._save is not None:
            try:
                self._save(self.settings)
            except OSError:
                pass

    # -- running ----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def ticked_rows(self) -> List[int]:
        rows = []
        for row in range(self.table.rowCount()):
            use = self.table.item(row, COL_USE)
            if use is not None and use.checkState() == QtCore.Qt.CheckState.Checked:
                rows.append(row)
        return rows

    def set_ticked(self, row: int, ticked: bool) -> None:
        use = self.table.item(row, COL_USE)
        if use is not None:
            use.setCheckState(QtCore.Qt.CheckState.Checked if ticked
                              else QtCore.Qt.CheckState.Unchecked)

    def _settle_pixel_sizes(self, rows: Sequence[int],
                            ask: Callable[..., Optional[float]]) -> None:
        """Settle, once per folder, the pixel size of files that lost it."""
        from tools import mps_pixel_size

        assert self.plan is not None
        remembered = getattr(self.settings, "pixel_size_by_folder", None)
        if remembered is None:
            remembered = {}
        settled: Dict[str, Any] = {}
        for row in rows:
            item = self.plan.files[row]
            if not item.needs_pixel_size or item.pixel_size_nm is not None:
                continue
            key = mps_pixel_size.folder_key(item.path)
            if key not in settled:
                folder = os.path.dirname(item.path)
                resolution, _notes = mps_pixel_size.resolve(
                    item.path, remembered,
                    lambda candidates, notes: ask(self, folder, candidates,
                                                  notes))
                settled[key] = resolution
                if resolution is not None and resolution.token == "manual":
                    self._persist()
            resolution = settled[key]
            if resolution is None:
                continue
            item.pixel_size_nm = float(resolution.nm)
            item.pixel_size_source = resolution.token
            cell = self.table.item(row, COL_PIXEL)
            if cell is not None:
                cell.setText(f"{resolution.nm:g} nm ({resolution.token})")
                cell.setToolTip(f"Not the file's own: {resolution.source}. "
                                f"Every lateral distance scales with it.")

    def run(self, ask: Optional[Callable[..., Optional[float]]] = None) -> None:
        """Analyse the ticked files in a worker thread."""
        if self.running or self.plan is None:
            return
        rows = self.ticked_rows()
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Batch", "Tick at least one file.")
            return
        self._settle_pixel_sizes(rows, ask or ask_pixel_size)
        if self._sync is not None:
            self._sync()
        settings = mb.BatchSettings.from_settings(self.settings)
        self._show_settings()
        items = [self.plan.files[row] for row in rows]
        self._rows = list(rows)
        self.not_analysed = {
            self.plan.files[row].path: "unticked"
            for row in range(len(self.plan.files)) if row not in set(rows)}
        for row in rows:
            self.records.pop(row, None)
            self.table.item(row, COL_RESULT).setText("waiting")
        self.progress.setRange(0, len(rows))
        self.progress.setValue(0)
        self._cancel.clear()
        self.label_status.setText(f"Analysing {len(rows)} file(s)...")

        def body() -> None:
            try:
                mb.run_batch(items, settings,
                             on_done=lambda i, r: self._relay.done.emit(i, r),
                             cancelled=self._cancel.is_set)
            finally:
                self._relay.finished.emit()

        self._thread = threading.Thread(target=body, daemon=True)
        self._thread.start()
        self._refresh_buttons()

    def cancel(self) -> None:
        if self.running:
            self._cancel.set()
            self.label_status.setText(
                "Stopping after the axon being analysed...")

    def wait(self, timeout: float = 3600.0) -> None:
        """Block until the batch finishes (for scripts and tests)."""
        if self._thread is not None:
            self._thread.join(timeout)
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.processEvents()

    def _on_done(self, index: int, record: mb.AxonRecord) -> None:
        row = self._rows[index]
        self.records[row] = record
        cell = self.table.item(row, COL_RESULT)
        if record.ok:
            r = record.row
            perimeter = r.get("perimeter_um")
            text = (f"{r.get('n_clusters_kept')} clusters"
                    + ("" if perimeter is None
                       else f", perimeter {perimeter:g} um")
                    + f", {r.get('n_warnings') or 0} warning(s)")
            cell.setText(marked("good", text))
            cell.setToolTip(str(r.get("warnings") or "No warnings."))
        else:
            cell.setText(marked("bad", record.error or "failed"))
            cell.setToolTip(record.error or "")
        self.progress.setValue(index + 1)

    def _on_finished(self) -> None:
        self._thread = None
        done = set(self.records)
        assert self.plan is not None
        for row in getattr(self, "_rows", []):
            if row not in done:
                self.not_analysed[self.plan.files[row].path] = (
                    "the batch was cancelled before it")
                self.table.item(row, COL_RESULT).setText(
                    marked("dim", "not analysed: cancelled"))
        records = [self.records[r] for r in sorted(self.records)]
        self.label_status.setText(mb.batch_summary(records))
        self._refresh_buttons()
        self._refresh_comparison()
        if self._close_when_done:
            self.close()

    def batch_records(self) -> List[mb.AxonRecord]:
        return [self.records[r] for r in sorted(self.records)]

    # -- writing ----------------------------------------------------------

    def _on_export(self) -> None:
        records = [r for r in self.batch_records() if r.ok]
        if not records:
            QtWidgets.QMessageBox.information(
                self, "Batch", "Run the batch first: nothing is analysed.")
            return
        start = (getattr(self.settings, "last_export_dir", "")
                 or (self.plan.root if self.plan else ""))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "The table to add the axons to",
            os.path.join(start, "mps_axons.csv"), "CSV (*.csv)",
            options=QtWidgets.QFileDialog.DontConfirmOverwrite)
        if path:
            self.export(path)

    def export(self, path: str,
               decide: Optional[Callable[..., str]] = None) -> Optional[str]:
        """
        Write the analysed axons to the table at ``path``: the rows, the
        dictionary of columns and the log. Returns the message shown, or
        None when nothing was written.
        """
        from tools import export_ui
        from tools.axon_export import AXON_KEY_COLUMNS
        from tools.results_table import (
            TableMismatch, append_rows, replace_rows,
        )

        records = self.batch_records()
        try:
            check = mb.check_export(records, path)
        except (TableMismatch, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Not written", str(error))
            return None
        rows = check.rows_to_write()
        left_out = {r.source: "the table already holds this file, exported "
                              "from the axon window with its ROI"
                    for r in records if r.ok and os.path.basename(
                        r.source).lower() in check.in_table_otherwise}
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Nothing to write",
                "Every analysed file is already in that table from the "
                "axon window.")
            return None
        decision = (decide or export_ui.resolve_duplicates)(
            self, [(path, rows, AXON_KEY_COLUMNS)])
        if decision == export_ui.CANCEL:
            return None
        try:
            if decision == export_ui.REPLACE:
                replaced = replace_rows(path, rows, AXON_KEY_COLUMNS)
                done = (f"{len(rows)} row(s) written to {path}, "
                        f"{replaced} of them in place of the same axons.")
            else:
                append_rows(path, rows)
                done = f"{len(rows)} row(s) written to {path}."
        except (OSError, ValueError, csv.Error) as error:
            QtWidgets.QMessageBox.critical(self, "Not written", str(error))
            return None

        note = done
        if left_out:
            note += (f"\n\n{len(left_out)} file(s) left out: the table "
                     f"already holds them, exported from the axon window "
                     f"with an ROI -- the same axons, measured another way.")
        try:
            from tools.axon_export import table_paths
            from tools.column_dictionary import write_dictionary
            from tools.results_table import read_header

            # The dictionary is rewritten at every export. The clusters and
            # localizations tables beside this one -- written from the axon
            # window -- keep their entries.
            listing = {"axon": list(rows[0])}
            for key in ("clusters", "localizations"):
                header = read_header(table_paths(path)[key])
                if header:
                    listing[key] = header
            note += ("\n\nWhat the columns mean: "
                     + write_dictionary(path, listing))
        except (OSError, ValueError):
            pass
        try:
            note += ("\nWhat became of every file: "
                     + mb.write_log(records, path, left_out=left_out,
                                    skipped=self.not_analysed))
        except OSError:
            pass
        failed = [r for r in records if not r.ok]
        if failed:
            note += (f"\n\n{len(failed)} file(s) failed and are not in the "
                     f"table; the log says why.")
        self.settings.last_export_dir = os.path.dirname(path)
        self._persist()
        self.label_status.setText(note)
        QtWidgets.QMessageBox.information(self, "Written", note)
        return note

    # ------------------------------------------------------------------
    # Tab 2: comparing groups
    # ------------------------------------------------------------------

    def _build_compare_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        intro = QtWidgets.QLabel(
            "One column, per group, at two levels: every axon, and one "
            "value per nesting unit -- the mean of that unit's axons. For "
            "a knockout the unit is the animal, since the knockout is done "
            "to an animal. How much the axons of one unit resemble each "
            "other (the ICC) says how far apart the two levels can be. No "
            "p-value is computed: which level to test, or whether to fit a "
            "mixed model, is a decision for you and your supervisor.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QtWidgets.QHBoxLayout()
        self.radio_batch = QtWidgets.QRadioButton("This batch")
        self.radio_batch.setToolTip("The axons analysed in the first tab.")
        self.radio_table = QtWidgets.QRadioButton("A table:")
        self.radio_table.setToolTip(
            "An exported table of axons -- from the axon window, a batch, "
            "or both.")
        self.radio_batch.setChecked(True)
        self.radio_batch.toggled.connect(self._refresh_comparison)
        row.addWidget(self.radio_batch)
        row.addWidget(self.radio_table)
        self.label_table = QtWidgets.QLabel("none opened")
        self.label_table.setStyleSheet(HINT_STYLE)
        row.addWidget(self.label_table, 1)
        open_table = QtWidgets.QPushButton("Open a table...")
        open_table.setToolTip("Read an axon table exported earlier.")
        open_table.clicked.connect(self._on_open_table)
        row.addWidget(open_table)
        layout.addLayout(row)

        form = QtWidgets.QHBoxLayout()
        form.addWidget(QtWidgets.QLabel("Column:"))
        self.combo_column = QtWidgets.QComboBox()
        for name, text in mb.COMPARABLE_COLUMNS.items():
            self.combo_column.addItem(f"{text}  [{name}]", name)
        self.combo_column.setToolTip(
            "What to compare. The KS p-value is not offered: it is a "
            "within-axon p-value, not a quantity a genotype can change.")
        form.addWidget(self.combo_column, 2)
        self.check_discard = QtWidgets.QCheckBox(
            "without the discarded clusters")
        self.check_discard.setToolTip(
            "The same column measured without the clusters the axoplasm "
            "panel discarded (the _discard columns). Axons exported "
            "without the discard have no value there and are left out.")
        form.addWidget(self.check_discard)
        form.addWidget(QtWidgets.QLabel("Groups:"))
        self.combo_group = QtWidgets.QComboBox()
        self.combo_group.addItem(NO_GROUPS, None)
        for name in FIELDS:
            self.combo_group.addItem(FIELD_LABELS[name], name)
        self.combo_group.setCurrentIndex(1 + FIELDS.index("genotype"))
        self.combo_group.setToolTip("What the groups are: normally the "
                                    "genotype.")
        form.addWidget(self.combo_group, 1)
        form.addWidget(QtWidgets.QLabel("Nesting unit:"))
        self.combo_nest = QtWidgets.QComboBox()
        for name in FIELDS:
            self.combo_nest.addItem(FIELD_LABELS[name], name)
        self.combo_nest.setCurrentIndex(FIELDS.index("animal"))
        self.combo_nest.setToolTip(
            "The unit the axons are nested in: the animal for a knockout. "
            "Without it, the slide or the ROI can show how alike the axons "
            "of one measurement are -- but several slides of one animal "
            "are still not independent.")
        form.addWidget(self.combo_nest, 1)
        layout.addLayout(form)
        for combo in (self.combo_column, self.combo_group, self.combo_nest):
            combo.currentIndexChanged.connect(self._refresh_comparison)
        self.check_discard.toggled.connect(self._refresh_comparison)

        self.result = QtWidgets.QTableWidget(0, 7)
        self.result.setHorizontalHeaderLabels(
            ["Level", "Group", "n", "Mean", "SD", "Median",
             "Nesting (ICC)"])
        self.result.horizontalHeader().setSectionResizeMode(
            6, QtWidgets.QHeaderView.Stretch)
        self.result.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.result.horizontalHeaderItem(6).setToolTip(
            "Within each group: the share of the variance between nesting "
            "units. Beside it, what this design reaches by chance alone "
            "(95th percentile), and how many independent axons the group's "
            "axons are worth.")
        layout.addWidget(self.result, 1)
        self.label_warnings = QtWidgets.QLabel("")
        self.label_warnings.setWordWrap(True)
        self.label_warnings.setStyleSheet(WARN_STYLE)
        self.label_warnings.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.label_warnings)
        return page

    def _on_open_table(self) -> None:
        start = getattr(self.settings, "last_export_dir", "") or os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "An exported table of axons", start, "CSV (*.csv)")
        if path:
            self.open_table(path)

    def open_table(self, path: str) -> bool:
        """Read an exported axon table into the comparison."""
        try:
            records, notes = mb.read_axon_table(path)
        except (OSError, ValueError, csv.Error) as error:
            QtWidgets.QMessageBox.warning(self, "Not read", str(error))
            return False
        self.table_records, self.table_notes = records, notes
        self.label_table.setText(f"{os.path.basename(path)}: "
                                 f"{len(records)} axon(s)")
        self.radio_table.setChecked(True)
        self._refresh_comparison()
        return True

    def comparison_records(self) -> List[mb.AxonRecord]:
        if self.radio_table.isChecked():
            return list(self.table_records)
        return self.batch_records()

    def _refresh_comparison(self) -> None:
        records = self.comparison_records()
        self.result.setRowCount(0)
        self.comparison = None
        if not records:
            self.label_warnings.setText(
                "Nothing to compare yet: run the batch, or open a table.")
            return
        column = mb.column_for(str(self.combo_column.currentData()),
                               self.check_discard.isChecked())
        group_by = self.combo_group.currentData()
        nest_by = str(self.combo_nest.currentData())
        try:
            result = mb.compare_groups(records, column, group_by=group_by,
                                       nest_by=nest_by)
        except ValueError as error:
            self.label_warnings.setText(marked("warn", str(error)))
            return
        self.comparison = result
        unit = mb.unit_noun(nest_by)
        rows: List[List[str]] = []

        def fmt(value: Optional[float]) -> str:
            return "" if value is None else f"{value:.4g}"

        by_unit = {s.group: s for s in result.per_unit}
        for summary in result.per_axon:
            nesting = result.nesting.get(summary.group)
            icc = ""
            if nesting is not None and nesting.icc is not None:
                icc = (f"{nesting.icc:.2f} ({nesting.severity}; chance "
                       f"reaches {nesting.null_p95:.2f}); worth about "
                       f"{nesting.effective_n:.1f} independent axons"
                       if nesting.null_p95 is not None
                       and nesting.effective_n is not None
                       else f"{nesting.icc:.2f}")
            rows.append(["per axon", summary.group, str(summary.n),
                         fmt(summary.mean), fmt(summary.sd),
                         fmt(summary.median), icc])
            per = by_unit.get(summary.group)
            if per is not None:
                rows.append([f"per {unit}", summary.group, str(per.n),
                             fmt(per.mean), fmt(per.sd), fmt(per.median),
                             ""])
        self.result.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, text in enumerate(values):
                self.result.setItem(r, c, _item(text))
        notes = list(self.table_notes if self.radio_table.isChecked()
                     else []) + list(result.warnings)
        self.label_warnings.setText("\n".join(marked("warn", n)
                                              for n in notes))

    # ------------------------------------------------------------------

    def _refresh_buttons(self) -> None:
        running = self.running
        have_files = self.plan is not None and bool(self.plan.files)
        self.btn_find.setEnabled(not running)
        self.btn_run.setEnabled(have_files and not running)
        self.btn_cancel.setEnabled(running)
        self.btn_set.setEnabled(have_files)
        self.btn_patterns.setEnabled(have_files and not running)
        self.btn_export.setEnabled(not running and any(
            r.ok for r in self.records.values()))

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        # The main window's settings may have changed since this window
        # was last shown.
        if self._sync is not None and not self.running:
            self._sync()
        self._show_settings()
        super().showEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self.running:
            # The axon being analysed is finished first; the window closes
            # when it is.
            self._close_when_done = True
            self.cancel()
            event.ignore()
            return
        super().closeEvent(event)


def identity_of(window: BatchWindow, row: int) -> AxonIdentity:
    """The identity a row of the table of files holds (for tests)."""
    assert window.plan is not None
    return window.plan.files[row].identity
