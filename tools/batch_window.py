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
import time
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
    # F1: where the worker is (unit of the bar, check point, done, total), and the scan's estimates.
    stage = QtCore.pyqtSignal(int, str, object, object)
    scanned = QtCore.pyqtSignal(object)


# F1: the bar counts each file (and each method run of the comparison) as this many steps, so it moves inside a
# file too (tools.mps_progress.file_fraction).
UNIT_STEPS = 1000
# The worker tells the window where it is at most this often, unless the stage changes [ad hoc: the bar is read
# by a person; 4 updates a second look continuous].
STAGE_EVERY_S = 0.25
# The last stage of the perimeter density box: the pooled A of the groups, whose null has no count of its own.
POOLING_WORDS = "pooled A of the groups (its Monte Carlo null)"


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


def ask_emitters_buffer(parent: Optional[QtWidgets.QWidget]) -> Optional[str]:
    """The imaging buffer of a batch that runs Emitters, asked once for the run (F5c): no default, OK stays
    disabled until one is chosen; None when cancelled (the batch then does not start)."""
    from tools import emitters_layer as el
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("Emitters: imaging buffer")
    lay = QtWidgets.QVBoxLayout(dlg)
    text = QtWidgets.QLabel(
        "This batch runs Emitters (BaGoL-like reimplementation). Choose the imaging buffer of these files:\n"
        "it sets the literature xi (Dempsey et al. 2011, Table 1) and every row records it.")
    text.setWordWrap(True)
    lay.addWidget(text)
    combo = QtWidgets.QComboBox()
    combo.setObjectName("combo_batch_buffer")
    combo.addItem("(choose)", None)
    for buf in el.known_buffers():
        combo.addItem(el.buffer_words(buf), buf)
    lay.addWidget(combo)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
    ok.setEnabled(False)
    combo.currentIndexChanged.connect(lambda _i: ok.setEnabled(combo.currentData() is not None))
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    lay.addWidget(buttons)
    if dlg.exec_() != QtWidgets.QDialog.Accepted or combo.currentData() is None:
        return None
    return str(combo.currentData())


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
        # str(): where PyQt5 has no type stubs -- the CI -- text() is Any.
        return str(self.edit.text()).strip()


# The batch's per-axon null (the clustering integration): B replicates [user decision 2026-10-22, A9 / P-R78: 199;
# 999 on demand in the axon window], and the slab size the estimate before Run assumes [ad hoc: the order of a
# sciatic axon's slab; a lower bound for denser files].
NULL_REPLICATES = 199
NULL_TYPICAL_SLAB_LOCS = 5000


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
        self._relay.stage.connect(self._on_stage)
        self._relay.scanned.connect(self._on_scanned)
        # F1: what the run in preparation or in progress needs between its phases (scan -> question -> batch).
        self._pending: Dict[str, Any] = {}
        self._phase = ""

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_run_tab(), "Analyse a folder")
        tabs.addTab(self._build_compare_tab(), "Compare groups")
        self.tabs = tabs
        # An identity typed in the first tab changes the groups of the
        # second: it is recomputed whenever it is looked at.
        tabs.currentChanged.connect(self._refresh_comparison)
        self.setCentralWidget(tabs)
        self._refresh_buttons()
        from tools import ui_search
        ui_search.install(self)                       # F3: Ctrl+F

    # ------------------------------------------------------------------
    # Tab 1: the folder
    # ------------------------------------------------------------------

    def _build_run_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "Every picked-axon file under a folder is analysed the way the "
            "axon window analyses one, and written as one row per axon to "
            "the same table 'Save axon' writes. Check who each file is "
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
        # Phase D (D12, review of D0 M4): off by default; ticked, every
        # ticked file is also clustered, after the batch, by exactly the
        # methods compared in the axon window's 'Compare methods...'
        # (mb.methods_to_compare, DBSCAN added as the reference), into
        # batch_methods_comparison.csv beside the exported table. Disabled
        # while none has been compared. _show_settings keeps it current.
        self.check_compare_methods = QtWidgets.QCheckBox("Compare methods (slow)")
        self.check_compare_methods.setObjectName("check_compare_methods")
        self.check_compare_methods.setChecked(False)
        row.addWidget(self.check_compare_methods)
        # BaGoL F5: the Emitters layer takes minutes per axon, so the batch
        # never runs it unless this box is ticked (off by default and not
        # remembered); without it a file saved with that method is reported
        # as not analysed, and Compare methods skips it.
        # A2 (user decision 2026-10-23): one box for every slow method (Emitters, OPTICS, voronoi_csr), and a
        # confirmation with the time estimate each time one is about to run (tools.slow_confirm)
        self.check_slow_methods = QtWidgets.QCheckBox("Allow slow methods")
        self.check_slow_methods.setObjectName("check_slow_methods")
        self.check_slow_methods.setChecked(False)
        self.check_slow_methods.setToolTip(
            "Let the batch run the slow methods -- Emitters (BaGoL-like reimplementation), OPTICS, voronoi_csr:\n"
            "minutes per axon, hours with the null. Off: a file saved with one of them is reported as not\n"
            "analysed, with the reason; 'Compare methods (slow)' and the null leave them out. On: Run asks\n"
            "once more, with the time estimate, before anything starts (decision A2). docs/emitters.md")
        row.addWidget(self.check_slow_methods)
        # The clustering integration (A9 sec. 5, regularity PR 4): the per-axon pipeline null of every file's own
        # method, with B replicates [user decision 2026-10-22: 199], and the regularity reading on it (stage 2 at
        # the pipeline level). Off by default and not remembered: it multiplies the batch's time by about B.
        # A slow method's null is refused unless 'Slow methods' is ticked too (open question A2).
        self.check_null = QtWidgets.QCheckBox(f"Null per axon (B = {NULL_REPLICATES})")
        self.check_null.setObjectName("check_null_per_axon")
        self.check_null.setChecked(False)
        self.check_null.setToolTip(
            "Also run the per-axon pipeline null of every file: molecules at random on its membrane, with this\n"
            "file's blinking and precision, analysed by the SAME method, curation and contour (B simulated\n"
            "axons), and the two-stage regularity reading on that null. About B times the time of the batch\n"
            "(the estimate is shown when ticked). A slow method's null is refused unless 'Allow slow methods'\n"
            "is ticked. Conclusions on the arrangement are read per animal (z), not per axon.\n"
            "Methods: docs/null_and_scale.md, docs/regularity.md")
        self.check_null.toggled.connect(self._show_null_estimate)
        row.addWidget(self.check_null)
        # C3 / C4 (user decision 2026-10-23): the closure of the Batch's null, kept in the settings
        # (batch_null_closure; N_loc by default). 'both' writes <table>_null_by_closure.csv beside the table.
        self.combo_null_closure = QtWidgets.QComboBox()
        self.combo_null_closure.setObjectName("combo_batch_null_closure")
        for text, value in (("closure N_loc", "n_locs"), ("closure K", "k"), ("both closures (x2)", "both")):
            self.combo_null_closure.addItem(text, value)
        current = str(getattr(self.settings, "batch_null_closure", "n_locs") or "n_locs")
        idx = self.combo_null_closure.findData(current)
        self.combo_null_closure.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_null_closure.setToolTip(
            "What each simulated axon of the null is matched to: N_loc (the slab's localizations, default) or K (the\n"
            "clusters found); 'both' runs the two nulls (about twice the time) and writes one row per axon and\n"
            "closure in <table>_null_by_closure.csv. Kept in the settings. docs/null_and_scale.md")
        self.combo_null_closure.currentIndexChanged.connect(self._on_null_closure)
        row.addWidget(self.combo_null_closure)
        # The perimeter density analysis (P12): ticked, the ticked files are also analysed by it after the batch
        # (batch_perimeter.analyse_files, in the worker, on the bar); Export writes its tables beside the table
        # chosen there, as <name>_perimeter_*.csv, never overwriting.
        self.check_perimeter = QtWidgets.QCheckBox("Perimeter density")
        self.check_perimeter.setObjectName("check_perimeter_density")
        self.check_perimeter.setChecked(False)
        self.check_perimeter.setToolTip(
            "Also run the perimeter density analysis; Export writes its tables as <name>_perimeter_*.csv.")
        row.addWidget(self.check_perimeter)
        self.perimeter_summary = ""
        self.perimeter_results: List[Any] = []
        self.perimeter_pooled: Optional[Dict[str, Any]] = None
        self.comparison_rows: List[Dict[str, Any]] = []
        self.comparison_results: List[Any] = []
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setToolTip(
            "Stop after the axon being analysed. What was analysed is "
            "kept and can be written. While the clusters are counted\n"
            "before the batch, stop that count: nothing is analysed.")
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_cancel)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setToolTip(
            "Moves inside each file too: loading, clustering, the perimeter (start by start\n"
            "when it is built from every start), the occupancy and the randomization\n"
            "(iteration by iteration). The text says which file and which stage.")
        row.addWidget(self.progress, 1)
        self.btn_export = QtWidgets.QPushButton("Save table...")
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

    def _on_null_closure(self, _i: int = 0) -> None:
        value = str(self.combo_null_closure.currentData() or "n_locs")
        if hasattr(self.settings, "batch_null_closure"):
            self.settings.batch_null_closure = value
            if self._save is not None:
                try:
                    self._save(self.settings)
                except Exception:  # noqa: BLE001 - kept in memory; saved with the next settings save
                    pass
        self._show_null_estimate(self.check_null.isChecked())

    def _show_null_estimate(self, on: bool = False) -> None:
        """The null's cost, said before Run: B simulated axons per file, at the table's per-method estimate for a
        typical slab (a lower bound: an axon with many more localizations costs more)."""
        if not on:
            self.check_null.setToolTip(self.check_null.toolTip().split("\n\nEstimate:")[0])
            return
        from tools.method_costs import estimate_seconds, words_seconds
        try:
            method = mb.BatchSettings.from_settings(self.settings).cluster_method
        except (AttributeError, TypeError, ValueError):
            method = "DBSCAN"
        # manual test of 2026-10-23: it said "for 1 file(s)" with two files in the table (computed before the
        # table was filled, and floored at 1); the ticked files of the table, updated as the table changes
        n_files = len(self.ticked_rows()) if self.plan is not None else 0
        try:
            closure = mb.BatchSettings.from_settings(self.settings).null_closure
        except (AttributeError, TypeError, ValueError):
            closure = "n_locs"
        # C3 / C4 and the design review (2026-10-29): the closure(s) and the regularity's stage 2 at the pipeline
        # level are counted; before, B + 8 replicates at the table's figure
        per_axon = mb.null_cost_per_file(method, NULL_TYPICAL_SLAB_LOCS, NULL_REPLICATES, closure)
        if per_axon <= 0:
            per_axon = estimate_seconds([method], NULL_TYPICAL_SLAB_LOCS) * (NULL_REPLICATES + 8)
        total = (f"{words_seconds(per_axon * n_files)} for the {n_files} ticked file(s)" if n_files
                 else "no file ticked in the table yet")
        text = (f"Estimate: about {words_seconds(per_axon)} per file for {method} on a slab of "
                f"{NULL_TYPICAL_SLAB_LOCS:,} localizations (closure: {closure}), {total} -- a lower bound.")
        self.check_null.setToolTip(self.check_null.toolTip().split("\n\nEstimate:")[0] + "\n\n" + text)
        self.label_status.setText(text)

    def _compare_methods_list(self) -> List[str]:
        """What 'Compare methods (slow)' would run with the settings in force (mb.methods_to_compare)."""
        try:
            return mb.methods_to_compare(mb.BatchSettings.from_settings(self.settings))
        except (AttributeError, TypeError, ValueError):
            return []

    def _show_compare_methods(self) -> None:
        methods = self._compare_methods_list()
        box = self.check_compare_methods
        box.setEnabled(bool(methods) and not self.running)
        if not methods:
            box.setChecked(False)
            box.setToolTip("Run 'Compare methods...' in the axon's MPS analysis window first: this box runs\n"
                           "only the methods compared there (closing that dialog after a run is enough).\n"
                           "Corré «Compare methods…» en la ventana MPS analysis del axón primero: esta casilla\n"
                           "corre sólo los métodos comparados ahí (alcanza con cerrar el diálogo tras correrlos).")
            return
        box.setToolTip(
            "After the batch, cluster every ticked file with the methods you compared in\n"
            f"'Compare methods...': {', '.join(methods)} (DBSCAN is the reference).\n"
            "Saved parameters for the saved method, registry defaults for the others;\n"
            "the file is loaded again per method. Writes batch_methods_comparison.csv\n"
            "next to the exported table; the axon table itself uses only the saved method.")

    def _show_settings(self) -> None:
        self._show_compare_methods()
        try:
            described = mb.BatchSettings.from_settings(self.settings).describe()
        except (AttributeError, TypeError, ValueError):
            described = "the main window's settings"
        self.label_settings.setText(
            "Analysed with the main window's settings, read when Run is "
            f"pressed: {described}.\n"
            f"{self._origins()}"
            "A batch draws no ROI: clusters touching the edge are judged "
            "against the convex hull of each file's localizations, and the "
            "rows say so (edge_reference). An axon exported from the axon "
            "window with an ROI is the same axon measured another way, so "
            "a table that already holds it keeps that row, not this one.")

    def _origins(self) -> str:
        """Where each value comes from (UI stage 2, design 12.3): 'eps 25 nm
        (paper)', or '(user; default 25 nm, paper)' for a value set in the
        MPS analysis window's parameter strip. Display only."""
        try:
            from tools.mps_params_panel import origin_words
            st = self.settings
            parts = [origin_words("dbscan.eps_nm", float(st.eps_nm)),
                     origin_words("dbscan.min_samples", int(st.min_samples)),
                     origin_words("slab.half_width_nm",
                                  float(st.slab_half_width_nm)),
                     origin_words("occupancy.mahalanobis",
                                  float(st.mahalanobis_threshold))]
        except (AttributeError, TypeError, ValueError, KeyError):
            return ""
        return ("Origin of each value: " + "; ".join(parts)
                + ". They are set in the MPS analysis window.\n")

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
        self._show_null_estimate(self.check_null.isChecked())

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
        if column == COL_USE:
            self._show_null_estimate(self.check_null.isChecked())
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

    def run(self, ask: Optional[Callable[..., Optional[float]]] = None,
            ask_buffer: Optional[Callable[..., Optional[str]]] = None,
            ask_perimeter: Optional[Callable[..., str]] = None) -> None:
        """Analyse the ticked files in a worker thread. A run that will run Emitters asks the imaging buffer
        once first (``ask_buffer``, the dialog by default; F5c): without it nothing starts.

        F1: before the batch, the files are clustered and their kept clusters counted; when any file reaches the
        perimeter warning, ``ask_perimeter(parent, over, n_files, compared, scanned)`` is asked once for the whole
        batch (tools.perimeter_cost_ui.ask_batch by default, only in an interactive session) and answers
        "every start", "one start" or "cancel"."""
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
        allow_slow = bool(self.check_slow_methods.isChecked())
        compare_list = mb.methods_to_compare(settings) if self.check_compare_methods.isChecked() else []
        null_b = NULL_REPLICATES if self.check_null.isChecked() else 0
        if allow_slow and not self._confirm_slow(settings, compare_list, null_b, len(rows)):
            self.label_status.setText("Not started: the slow method(s) were not confirmed.")
            return
        emitters_buffer: Optional[str] = None
        if mb.needs_emitters_buffer(settings, allow_slow, compare_list):
            emitters_buffer = (ask_buffer or ask_emitters_buffer)(self)
            if not emitters_buffer:
                self.label_status.setText(
                    "Not started: choose the imaging buffer (MEA or \u03b2ME) for the Emitters run.")
                return
            settings = mb.with_emitters_buffer(settings, emitters_buffer)
        z_store = dict(getattr(self.settings, "z_calibration_by_folder",
                               None) or {})
        self._show_settings()
        items = [self.plan.files[row] for row in rows]
        compare_kw: Dict[str, Any] = {} if emitters_buffer is None else {"emitters_buffer": emitters_buffer}
        self._pending = dict(rows=list(rows), items=items, settings=settings, z_store=z_store,
                             allow_slow=allow_slow, compare_list=list(compare_list), compare_kw=compare_kw,
                             null_b=null_b, perimeter=bool(self.check_perimeter.isChecked()))
        self._cancel.clear()
        # F1: a file with many clusters builds its perimeter for minutes. Before the batch, each file is clustered
        # and its kept clusters counted (mb.scan_perimeter_costs); when any reaches the threshold of the axon
        # window's question, one question is asked for the whole batch. With no one to ask (offscreen, the
        # tests) nothing is scanned or asked and the batch runs from every start, as it always did.
        from tools import perimeter_cost_ui
        asker = ask_perimeter or (perimeter_cost_ui.ask_batch if perimeter_cost_ui.interactive() else None)
        self._pending["asker"] = asker
        if asker is None or not self._method_runs(settings, allow_slow):
            self._start_analysis(None)
        elif self._too_slow_to_scan(settings):
            self._decide_perimeter(asker, [], scanned=False)
        else:
            self._start_scan()

    def _confirm_slow(self, settings: mb.BatchSettings, compare_list: List[str], null_b: int, n_files: int) -> bool:
        """A2 (b): with 'Allow slow methods' ticked, the slow methods this run would start -- the saved method, the
        compared ones, the saved method's null -- with the estimate for the ticked files, asked before Run."""
        from tools import slow_confirm
        from tools.method_costs import estimate_seconds, is_slow
        n = max(int(n_files), 1)
        items: List[Tuple[str, float]] = []
        if is_slow(settings.cluster_method):
            items.append((settings.cluster_method,
                          estimate_seconds([settings.cluster_method], NULL_TYPICAL_SLAB_LOCS) * n))
            if null_b:
                items.append((f"{settings.cluster_method} (its null, B = {null_b})",
                              mb.null_cost_per_file(settings.cluster_method, NULL_TYPICAL_SLAB_LOCS, null_b,
                                                    settings.null_closure) * n))
        for m in compare_list:
            if is_slow(m) and m != settings.cluster_method:
                items.append((f"{m} (Compare methods)", estimate_seconds([m], NULL_TYPICAL_SLAB_LOCS) * n))
        return slow_confirm.confirm_slow(self, "The Batch", items, note=(
            f"For the {n} ticked file(s), on a slab of {NULL_TYPICAL_SLAB_LOCS:,} localizations each."))

    @staticmethod
    def _method_runs(settings: mb.BatchSettings, allow_slow: bool) -> bool:
        """The saved method will cluster the files (Emitters without 'Slow methods' is refused file by file)."""
        from tools.method_costs import is_slow
        return allow_slow or (settings.cluster_method not in mb.BATCH_SLOW_METHODS
                              and not is_slow(settings.cluster_method))

    @staticmethod
    def _too_slow_to_scan(settings: mb.BatchSettings) -> bool:
        """A method the axon window runs in the background (tools.cluster_method_ui.is_slow): clustering every
        file twice would cost minutes per file, so its clusters are not counted first."""
        from tools.cluster_method_ui import is_slow
        return bool(is_slow(settings.cluster_method))

    # -- F1: the scan, the question, the batch -----------------------------------------------------------------

    def _stage_relay(self, unit: Callable[[], int]) -> Callable[[str, Optional[int], Optional[int]], None]:
        """An observer for the worker (tools.mps_progress): posts where it is, at most every STAGE_EVERY_S unless
        the stage or the file changes."""
        last: Dict[str, Any] = {"key": None, "t": 0.0}

        def observe(where: str, done: Optional[int], total: Optional[int]) -> None:
            key = (unit(), where)
            now = time.monotonic()
            if key != last["key"] or now - last["t"] >= STAGE_EVERY_S:
                last["key"], last["t"] = key, now
                self._relay.stage.emit(key[0], where, done, total)
        return observe

    def _set_bar(self, units: int, phase: str) -> None:
        self._phase = phase
        self._t_phase = time.monotonic()
        self._bar_head, self._bar_words = "", ""
        timer = getattr(self, "_bar_timer", None)
        if timer is None:
            timer = self._bar_timer = QtCore.QTimer(self)
            timer.setInterval(1000)
            timer.timeout.connect(self._redraw_bar_text)
        timer.start()
        self._units = max(1, int(units))
        self.progress.setRange(0, self._units * UNIT_STEPS)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")

    def _start_scan(self) -> None:
        pend = self._pending
        items = pend["items"]
        self._set_bar(len(items), "scan")
        self.label_status.setText(
            f"Counting the clusters of {len(items)} file(s) before the batch, to estimate the perimeter's time "
            f"(one reading and one clustering per file)...")
        current = [0]

        def done(i: int, _e: Any) -> None:
            current[0] = i + 1

        def body() -> None:
            estimates: List[mb.PerimeterEstimate] = []
            try:
                estimates = mb.scan_perimeter_costs(
                    items, pend["settings"], z_store=pend["z_store"], allow_slow=pend["allow_slow"],
                    cancelled=self._cancel.is_set, on_done=done, observer=self._stage_relay(lambda: current[0]))
            finally:
                self._relay.scanned.emit(estimates)

        self._thread = threading.Thread(target=body, daemon=True)
        self._thread.start()
        self._refresh_buttons()

    def _on_scanned(self, estimates: List[mb.PerimeterEstimate]) -> None:
        self._thread = None
        pend = self._pending
        if self._cancel.is_set() or len(estimates) < len(pend.get("items", [])):
            self.progress.setValue(0)
            self.progress.setFormat("%p%")
            self.label_status.setText("Cancelled before the batch started: nothing was analysed.")
            self._refresh_buttons()
            if self._close_when_done:
                self.close()
            return
        over = [(os.path.basename(e.source), int(e.n_clusters)) for e in estimates if e.n_clusters is not None]
        if not over:
            self._start_analysis(None)
            return
        self._decide_perimeter(pend["asker"], over, scanned=True)

    def _decide_perimeter(self, asker: Callable[..., str], over: List[Tuple[str, int]], scanned: bool) -> None:
        from tools import mps_perimeter_cost as cost
        pend = self._pending
        answer = asker(self, over, len(pend["items"]), bool(pend["compare_list"]), scanned)
        if answer not in (cost.EVERY_START, cost.ONE_START):
            self.progress.setValue(0)
            self.progress.setFormat("%p%")
            self.label_status.setText("Not started: the batch was cancelled at the perimeter question.")
            self._refresh_buttons()
            return
        self._start_analysis(answer)

    def _start_analysis(self, perimeter_choice: Optional[str]) -> None:
        """The batch itself, in the worker thread. ``perimeter_choice``: the answer for the whole batch (installed
        as the gate of tools.mps_perimeter_cost), or None: no gate, every start, as before F1."""
        assert self.plan is not None
        pend = self._pending
        rows, items, settings = pend["rows"], pend["items"], pend["settings"]
        z_store, allow_slow = pend["z_store"], pend["allow_slow"]
        compare_list, compare_kw = pend["compare_list"], pend["compare_kw"]
        null_b = int(pend.get("null_b", 0))
        self._rows = list(rows)
        self.not_analysed = {
            self.plan.files[row].path: "unticked"
            for row in range(len(self.plan.files)) if row not in set(rows)}
        for row in rows:
            self.records.pop(row, None)
            self.table.item(row, COL_RESULT).setText("waiting")
        self._set_bar(len(rows) * (1 + len(compare_list)) + (len(rows) + 1 if self.check_perimeter.isChecked() else 0),
                      "batch")        # the perimeter's extra unit: the pooled A of the groups (its Monte Carlo null)
        self._n_files = len(rows)
        self._with_null = bool(null_b)
        self._compare_names = list(compare_list)
        self.label_status.setText(f"Analysing {len(rows)} file(s)..."
                                  + ("" if perimeter_choice is None
                                     else f" Long perimeters: {perimeter_choice}."))

        self.comparison_rows = []
        self.comparison_results = []
        batch_kw: Dict[str, Any] = {} if not null_b else {"null_replicates": null_b}
        current = [0]
        n_files = len(items)
        n_methods = max(1, len(compare_list))
        perimeter_items = list(items) if pend.get("perimeter") else []
        self.perimeter_summary = ""
        self.perimeter_results = []
        self.perimeter_pooled = None
        perimeter_base = n_files * (1 + len(compare_list))

        def file_done(i: int, r: mb.AxonRecord) -> None:
            current[0] = i + 1
            self._relay.done.emit(i, r)

        def run_done(i: int, j: int) -> None:
            current[0] = n_files + i * n_methods + j + 1

        def body() -> None:
            from tools import mps_perimeter_cost as cost
            from tools import mps_cancel, mps_progress
            observer = self._stage_relay(lambda: current[0])
            try:
                # Cancel reaches INSIDE the axon being analysed (a start of the perimeter, a replicate of the null):
                # before, it was heard only between two files and a null of 40 minutes could not be stopped.
                with cost.asking(None if perimeter_choice is None else mb.perimeter_policy_gate(perimeter_choice)), \
                        mps_progress.observing(observer), mps_cancel.cancellable(self._cancel.is_set):
                    mb.run_batch(items, settings,
                                 on_done=file_done,
                                 cancelled=self._cancel.is_set, z_store=z_store, allow_slow=allow_slow, **batch_kw)
                    if compare_list and not self._cancel.is_set():
                        self.comparison_rows = mb.compare_methods(
                            items, settings, compare_list, cancelled=self._cancel.is_set, z_store=z_store,
                            on_done=run_done, allow_slow=allow_slow, results_out=self.comparison_results,
                            **compare_kw)
                    if perimeter_items and not self._cancel.is_set():
                        def perimeter_file(i: int, _path: str) -> None:
                            current[0] = perimeter_base + i
                        self.perimeter_summary = self._run_perimeter_density(perimeter_items, on_file=perimeter_file)
            except mps_cancel.AnalysisCancelled:
                pass                                   # the files finished before it keep their rows
            finally:
                self._relay.finished.emit()

        self._thread = threading.Thread(target=body, daemon=True)
        self._thread.start()
        self._refresh_buttons()

    def _run_perimeter_density(self, items: Sequence[Any],
                               on_file: Optional[Callable[[int, str], None]] = None) -> str:
        """Worker thread: the perimeter density analysis of the batch's files (batch_perimeter.analyse_files, the
        command-line batch's own loop), with the identities and pixel sizes the table holds, and the pooled A of
        the groups (its null is the slow part of the writing). Nothing is written here: Export writes the tables
        beside the table chosen there. Returns the line the status shows; never raises."""
        try:
            import batch_perimeter as bp
            results = bp.analyse_files(
                [it.path for it in items], identities={it.path: it.identity for it in items},
                pixel_sizes={it.path: it.pixel_size_nm for it in items if it.pixel_size_nm is not None},
                cancelled=self._cancel.is_set, on_file=on_file)
            if self._cancel.is_set():
                self.perimeter_results, self.perimeter_pooled = [], None
                return "Perimeter density: stopped; nothing kept."
            n_f = int(getattr(self, "_n_files", len(results)))
            self._relay.stage.emit(n_f * (2 + len(getattr(self, "_compare_names", []) or [])),
                                   POOLING_WORDS, None, None)
            pooled = bp.pooled_of(results)
            self.perimeter_results, self.perimeter_pooled = list(results), pooled
            n_ok = sum(1 for r in results if r.status.value == "ok")
            return f"Perimeter density: {n_ok} of {len(results)} ok; Export writes its tables."
        except Exception as exc:                       # noqa: BLE001 - said in the status, never a crash
            self.perimeter_results, self.perimeter_pooled = [], None
            return f"Perimeter density failed: {type(exc).__name__}: {exc}"

    def write_perimeter_tables(self, table_path: str) -> str:
        """Export: the perimeter density tables of this batch beside the table at ``table_path``, at the stem
        ``<table without .csv>`` (``<stem>_perimeter_*.csv``; ``_run2``... when taken: nothing is overwritten).
        Returns the stem used, or "" when the box did not run."""
        if not self.perimeter_results:
            return ""
        import batch_perimeter as bp
        stem = os.path.splitext(table_path)[0]
        return bp.write_tables(self.perimeter_results, stem, pooled=self.perimeter_pooled)

    def _on_stage(self, unit: int, where: str, done: Any, total: Any) -> None:
        """GUI thread: move the bar inside the file being analysed and say which stage it is in."""
        from tools import mps_progress
        # With the per-axon null ticked, a file's bar has its own shares (analysis, then the null replicate by
        # replicate): before, the bar stood at ~99 % for the whole null.
        with_null = bool(getattr(self, "_with_null", False)) and self._phase == "batch"             and unit < getattr(self, "_n_files", 0)
        frac, words = mps_progress.file_fraction(where, done, total, with_null=with_null)
        if frac is not None and unit < getattr(self, "_units", 1):
            self.progress.setValue(max(self.progress.value(), unit * UNIT_STEPS + int(UNIT_STEPS * frac)))
        if self._phase == "scan":
            n = len(self._pending.get("items", []))
            head = f"Counting clusters, file {min(unit + 1, n)} of {n}"
        else:
            n = getattr(self, "_n_files", 0)
            if unit < n:
                head = f"File {unit + 1} of {n}"
            elif unit >= n * (1 + len(getattr(self, "_compare_names", []) or [])):
                k = unit - n * (1 + len(getattr(self, "_compare_names", []) or []))
                head = f"Perimeter density, file {min(k + 1, n)} of {n}"
                if k >= n:
                    head = "Perimeter density, pooling the groups"
                    self.progress.setValue(max(self.progress.value(), unit * UNIT_STEPS))
                if frac is None and done is not None and total:
                    frac = min(1.0, float(done) / float(total))
                    if unit < getattr(self, "_units", 1):
                        self.progress.setValue(max(self.progress.value(), unit * UNIT_STEPS + int(UNIT_STEPS * frac)))
            else:
                names = getattr(self, "_compare_names", []) or ["?"]
                k = unit - n
                i, j = divmod(k, len(names))
                head = f"Comparing methods, file {min(i + 1, n)} of {n}, {names[min(j, len(names) - 1)]}"
        self._bar_head, self._bar_words = head, words
        self._redraw_bar_text()

    def _redraw_bar_text(self) -> None:
        """The bar's text: file, stage, percentage, time spent and an estimate of what is left (a straight line from
        the share done: the stages are not equally long, so it is said to be about)."""
        if not getattr(self, "_bar_head", "") or not self.running:
            return
        from tools import mps_progress
        span = max(1, self.progress.maximum())
        f = self.progress.value() / span
        spent = time.monotonic() - getattr(self, "_t_phase", time.monotonic())
        eta = mps_progress.eta_words(f, spent)
        tail = f" - {mps_progress.elapsed_words(spent)} elapsed" + (f", {eta}" if eta else "")
        self.progress.setFormat(f"{self._bar_head}: {self._bar_words} - %p%{tail}")

    def cancel(self) -> None:
        if self.running:
            self._cancel.set()
            self.label_status.setText(
                "Stopping at the next check of the axon being analysed (a start of its perimeter, a replicate of its "
                "null); the axons finished before stay in the table...")

    def wait(self, timeout: float = 3600.0) -> None:
        """Block until the batch finishes (for scripts and tests), through its phases (F1: the scan, then the
        batch the question starts)."""
        deadline = time.monotonic() + float(timeout)
        while True:
            thread = self._thread
            if thread is not None:
                thread.join(max(0.0, deadline - time.monotonic()))
            QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.processEvents()
            # done, or out of time; a new thread (the batch after the scan) is waited for in turn
            if self._thread is None or time.monotonic() >= deadline:
                break

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
                    + f", {r.get('n_warnings') or 0} analysis note(s)")
            cell.setText(marked("good", text))
            cell.setToolTip(str(r.get("warnings") or "No analysis notes."))
        else:
            cell.setText(marked("bad", record.error or "failed"))
            cell.setToolTip(record.error or "")
        self.progress.setValue(max(self.progress.value(), (index + 1) * UNIT_STEPS))

    def _on_finished(self) -> None:
        self._thread = None
        self._bar_head = ""
        if getattr(self, "_bar_timer", None) is not None:
            self._bar_timer.stop()
        self.progress.setFormat("%p%")
        if not self._cancel.is_set():
            self.progress.setValue(self.progress.maximum())
        done = set(self.records)
        assert self.plan is not None
        for row in getattr(self, "_rows", []):
            if row not in done:
                self.not_analysed[self.plan.files[row].path] = (
                    "the batch was cancelled before it")
                self.table.item(row, COL_RESULT).setText(
                    marked("dim", "not analysed: cancelled"))
        records = [self.records[r] for r in sorted(self.records)]
        self.label_status.setText(mb.batch_summary(records)
                                  + (f"\n{self.perimeter_summary}" if self.perimeter_summary else ""))
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
            from tools.results_table import with_csv_extension
            self.export(with_csv_extension(path))

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
            TableMismatch, append_rows, replace_rows, saved_as_notes, with_csv_extension,
        )

        path = with_csv_extension(path)
        table_existed = os.path.exists(path)
        records = self.batch_records()
        try:
            check = mb.check_export(records, path)
        except (TableMismatch, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Not written", str(error))
            return None
        rows = check.rows_to_write()
        if self.comparison_rows:
            long_file = ""
            try:
                mb.write_methods_comparison(self.comparison_rows, mb.comparison_path(path))
                # the clustering integration (A4): the long table, the agreement and the summary beside it
                long_file = mb.write_methods_comparison_long(self.comparison_results, path).get("long", "")
            except OSError as error:
                QtWidgets.QMessageBox.warning(self, "Comparison not written", str(error))
            # finding (n): the rows' methods block from this comparison, as the axon window fills it
            mb.fill_methods_columns(rows, self.comparison_results, long_file)
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
            if table_existed:
                done += ("\nThe table already existed: nothing was overwritten, the rows were "
                         + ("put in place of the same axons." if decision == export_ui.REPLACE else "added to it."))
        except (OSError, ValueError, csv.Error) as error:
            QtWidgets.QMessageBox.critical(self, "Not written", str(error))
            return None

        regularity_written: Dict[str, str] = {}
        regularity_rows = [dict(r.row, animal=r.identity.animal, genotype=r.identity.genotype)
                           for r in records if r.ok and r.row.get("regularity_reading")]
        if regularity_rows:
            # regularity PR 4: per animal z with Holm, the P-R83 lattice criterion per genotype, the calibre model
            from tools import clustering_integration as ci
            try:
                summary = ci.regularity_batch_summary(regularity_rows)
                written = ci.write_regularity_summary(path, summary)
                regularity_written = dict(written)
                done += "\n\nRegularity: " + written.get("lines", "") + "\n" + "\n".join(summary["lines"][:6])
            except (OSError, ValueError) as error:
                done += f"\n\nRegularity summary not written: {error}"
        # the size readouts across axons and animals (tools.mps_size_stats; it existed and nothing called it)
        size_written: Dict[str, str] = {}
        size_rows = [dict(r.row, animal=r.identity.animal, genotype=r.identity.genotype) for r in records if r.ok]
        if size_rows:
            from tools import clustering_integration as ci_size
            try:
                size_summary = ci_size.size_batch_summary(size_rows)
                size_written = ci_size.write_size_summary(path, size_summary)
                if size_written:
                    done += ("\n\nSize across axons: " + size_written.get("size_stats", "") + "\n"
                             + "\n".join(size_summary["lines"][:6]))
            except (OSError, ValueError) as error:
                done += f"\n\nSize summary not written: {error}"
        # P12: the perimeter density tables of this batch, beside the table and named after it
        try:
            pd_stem = self.write_perimeter_tables(path)
            if pd_stem:
                done += f"\n\nPerimeter density: tables at {os.path.basename(pd_stem)}_perimeter_*.csv."
        except (OSError, ValueError) as error:
            done += f"\n\nPerimeter density tables not written: {error}"
        # C3 / C4: the per-axon null, one row per axon x closure (written before the dictionary, which lists it)
        by_closure = ""
        try:
            by_closure = mb.write_null_by_closure(records, path)
            if by_closure:
                done += "\n\nThe per-axon null, one row per closure: " + by_closure
        except OSError as error:
            done += f"\n\nThe per-closure null table was not written: {error}"
        base = os.path.splitext(path)[0]
        expected = {"size": base + "_size_stats.csv", "closure": base + "_null_by_closure.csv"}
        got = {"size": size_written.get("size_stats", ""), "closure": by_closure}
        for key, suffix in (("by_animal", "_regularity_by_animal.csv"), ("readings", "_regularity_readings.csv"),
                            ("lines", "_regularity.txt")):
            if regularity_written.get(key):
                expected[key], got[key] = base + suffix, regularity_written[key]
        renamed = saved_as_notes(expected, got)
        if renamed:
            done += "\n\n" + "\n".join(renamed)
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
            if size_written:
                from tools.clustering_integration import SIZE_STATS_COLUMNS
                listing["size_stats"] = list(SIZE_STATS_COLUMNS)
            if by_closure:
                from tools.clustering_integration import null_by_closure_columns
                listing["null_by_closure"] = list(null_by_closure_columns())
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
                       f"reaches {nesting.null_p95:.2f})"
                       if nesting.null_p95 is not None
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
        self.check_compare_methods.setEnabled(not running and bool(self._compare_methods_list()))
        self.check_slow_methods.setEnabled(not running)
        self.check_perimeter.setEnabled(not running)
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
