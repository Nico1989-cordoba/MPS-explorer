# -*- coding: utf-8 -*-
"""
The viability explorer (H6 toggles, D-43; the user's request of 2026-10-02): on many picked axons at once, how many
consecutive ring pairs -- and how many axons -- each selection of the criteria passes, "with all four criteria no pair,
without the leak criterion one appears, without the valley three", for rule v2 (localizations) and rule v2c (clusters,
variants A / B / C), stricter or more permissive, changing in real time.

What it shows
-------------
* The program's criteria switches (``tools.mps_selection_ui``): every window of the program shows the same selection.
* Live counts of the current selection: "pairs passing: V (V+M) of N; axons with >= 1 passing pair: a of A".
* One row per ring pair of every axon under the current selection: its tier, the checked criteria that fail, the
  unchecked ones that would fail, and every criterion of rule v2 and of the current v2c variant.
* The 16 x 4 table: each of the 16 subsets of the four criteria (strictest first) under v2 / v2c-A / v2c-B / v2c-C,
  "V (V+M) | axons" per cell, with the current selection's cell outlined; a click sets the switches to that cell.
* "Export CSV...": the 16 x 4 table and the pairs of the current selection, both with the selection's label and hash.

Geometry only (R8): the per-pair criteria come from ``tools.mps_selection.geometry_for_file`` (the batch's loader,
the pre-registered rings, z_quality, rule v2, rule v2c) and nothing here imports or runs a column analysis, so it may
run on real axons. The files are computed once (in worker processes, at most 12) and cached as JSON
(``tools.mps_viability_jobs``): toggling never recomputes anything.

Run on its own:
    venv\\Scripts\\python.exe -m tools.mps_viability_explorer --demo
    venv\\Scripts\\python.exe -m tools.mps_viability_explorer --inputs <folder> [<folder> ...] --workers 4

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import math
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # run as a script: make "tools" importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_selection as msel
from tools.mps_plot_style import marked, verdict
from tools.mps_selection_ui import SelectionWidget, app_log_path
from tools.mps_tooltips import apply_tooltips
from tools.mps_viability_jobs import cache_dir_for, geometry_job, init_worker, load_cached

__all__ = ["COMBINATION_HEADERS", "EXPLORER_TOOLTIPS", "PAIR_HEADERS", "ViabilityExplorer", "main", "subset_text"]

MAX_WORKERS = 12
DEFAULT_WORKERS = 4
OUTLINE_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 7
COMBINATION_HEADERS: Dict[str, str] = {"v2": "v2\n(locs)", "v2c-A": "v2c-A\n(ring slabs)",
                                       "v2c-B": "v2c-B\n(3D clusters)", "v2c-C": "v2c-C\n(corrected)"}
COMBINATION_HEADER_TIPS: Dict[str, str] = {
    "v2": "Rule v2 (D-41): the criteria read on the LOCALIZATIONS; leak estimate of the switches' v2 row.",
    "v2c-A": "Rule v2c, variant A: ring-slab cluster medians (cannot pass (2c) for similar rings: slab cut at the "
             "valley).",
    "v2c-B": "Rule v2c, variant B: slab-free 3D clusters (its selection depends on whether columns exist).",
    "v2c-C": "Rule v2c, variant C: ring slabs with the centre corrected for the slab cut (cannot pass (2c) for similar "
             "rings)."}
PAIR_HEADERS: Tuple[str, ...] = (
    "Axon", "Pair", "Selection", "Fails (checked)", "Would fail (not checked)", "v2 (1) peaks", "v2 (2) valley",
    "v2 (3) % of central (min)", "(4) leak D-39", "(4c) leak tnfix", "v2c (1c) peaks", "v2c (2c) valley",
    "v2c (3c) K / K central", "v2 verdict", "v2c verdict")
GEOMETRY_NOTE = ("Geometry only: which ring pairs each selection would read. No column statistic is computed here "
                 "(no matching, no z_A, no p; R8), so real axons are allowed.")

EXPLORER_TOOLTIPS: Dict[str, str] = {
    "inputs_list": "The picked axons (files) and folders (every localization file and simulated .npz in them).",
    "pixel_spin": "The camera pixel in nm for files without one in their metadata. 0 = read it from the metadata "
                  "(Picasso's YAML, the NPZ), as the batch does.",
    "workers_spin": "Processes computing the criteria at once (at most 12). 1 = in this program's background thread. "
                    "A file already computed (same file, pixel size, rules and column parameters) is read from the "
                    "cache in a fraction of a second.",
    "compute_button": "Compute the per-pair criteria of both rules for every axon listed (geometry only: SiZer, "
                      "counts, leak estimates; about 12 s per real axon the first time, then cached).",
    "cancel_button": "Stop after the files already being computed; the finished ones stay (and are cached).",
    "counts_label": "The current selection on every axon listed: pairs VIABLE (VIABLE + MARGINAL) of all pairs, and "
                    "axons with at least one VIABLE pair.",
    "pair_table": "One row per consecutive ring pair of every axon under the current selection: its tier, the "
                  "CHECKED criteria that fail, the UNCHECKED ones that would fail, and every criterion of rule v2 and "
                  "of the v2c variant chosen (the one of the switches; B under rule v2).",
    "combination_table": "Each of the 16 subsets of the four criteria (rows, strictest first) under each rule "
                         "(columns): 'V (V+M) | axons' = pairs VIABLE, pairs VIABLE + MARGINAL, axons with at least "
                         "one VIABLE pair. The leak estimate is the one of the switches (v2: its own; v2c: its "
                         "own).\n\n"
                         "The outlined cell is the current selection. Click a cell to set the switches to it.",
    "export_button": "Write the 16 x 4 table and the pairs of the current selection as CSV, both with the selection's "
                     "label and hash.",
    "status_label": "What the computation is doing; files that failed are listed with their error.",
}


def subset_text(criteria: Sequence[str]) -> str:
    """A row of the 16 x 4 table: the criteria checked, or 'none (every pair)'."""
    return ", ".join(criteria) if criteria else "none (every pair)"


def axon_name(path: str) -> str:
    """A short name of a picked axon for the table: "ROI<n>_axon<m>" when the path has a ROI folder and an axon
    number (as power_columns.axon_label names picked axons), else the folder and the file's stem, shortened."""
    import re
    base = os.path.basename(path)
    m = re.search(r"axon\s*(\d+)", base, re.IGNORECASE)
    parts = os.path.normpath(os.path.dirname(path)).split(os.sep)
    roi = next((re.sub(r"[^A-Za-z0-9]+", "", p) for p in reversed(parts) if p.lower().startswith("roi")), "")
    if m and roi:
        return f"{roi}_axon{m.group(1)}"
    name = f"{os.path.basename(os.path.dirname(path))}/{os.path.splitext(base)[0]}"
    return name if len(name) <= 46 else name[:18] + "..." + name[-25:]


def _pct(v: Any) -> str:
    f = msel._f(v)
    return f"{100 * f:.1f} %" if math.isfinite(f) else "n/a"


def _yes(b: Any) -> str:
    return "yes" if bool(b) else "NO"


class _Relay(QtCore.QObject):
    one = QtCore.pyqtSignal(int, object)     # generation, geometry_job's dict
    done = QtCore.pyqtSignal(int, object)    # generation, {"cancelled", "seconds", "error"}


class _OutlineDelegate(QtWidgets.QStyledItemDelegate):
    """Draws a frame around the cell of the current selection."""

    def paint(self, painter: Any, option: Any, index: Any) -> None:
        super().paint(painter, option, index)
        if index.data(OUTLINE_ROLE):
            painter.save()
            pen = QtGui.QPen(QtGui.QColor(verdict("bad", dark=False)))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(option.rect.adjusted(1, 1, -2, -2))
            painter.restore()


class ViabilityExplorer(QtWidgets.QMainWindow):
    """
    The viability explorer (module docstring). ``inputs``: files or folders to list; ``compute()`` starts the
    background computation, ``wait()`` blocks until it ends (scripts and tests). ``criteria`` maps each file to its
    ``AxonCriteria``; ``errors`` each failed file to its error. ``state``: a ``SelectionState`` (the program's when
    None); ``log_path``: the exploration log whose folder's ``../viability_cache`` holds the cache.
    """

    computed = QtCore.pyqtSignal()

    def __init__(self, inputs: Optional[Sequence[str]] = None, parent: Optional[QtWidgets.QWidget] = None, *,
                 pixel_size_nm: Optional[float] = None, workers: int = DEFAULT_WORKERS, state: Any = None,
                 log_path: Optional[str] = None) -> None:
        super().__init__(parent)
        self.criteria: Dict[str, msel.AxonCriteria] = {}
        self.errors: Dict[str, str] = {}
        self.order: List[str] = []
        self.n_cached = 0
        self.n_computed = 0
        self.seconds = 0.0
        self.cache_dir = cache_dir_for(log_path or app_log_path())
        self._generation = 0
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._relay = _Relay()
        self._relay.one.connect(self._on_one)
        self._relay.done.connect(self._on_done)
        self.setWindowTitle("Viability explorer (geometry only): ring pairs under every selection of the criteria")
        self.resize(1500, 920)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        note = QtWidgets.QLabel(marked("dim", GEOMETRY_NOTE))
        note.setWordWrap(True)
        root.addWidget(note)

        top = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        self.inputs_list = QtWidgets.QListWidget()
        self.inputs_list.setObjectName("inputs_list")
        self.inputs_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.inputs_list.setMaximumHeight(110)
        left.addWidget(self.inputs_list)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (("Add files...", self._on_add_files), ("Add folder...", self._on_add_folder),
                           ("Remove selected", self._on_remove), ("Clear", self._on_clear)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        left.addLayout(row)
        top.addLayout(left, 3)
        form = QtWidgets.QFormLayout()
        self.pixel_spin = QtWidgets.QDoubleSpinBox()
        self.pixel_spin.setObjectName("pixel_spin")
        self.pixel_spin.setRange(0.0, 1000.0)
        self.pixel_spin.setDecimals(1)
        self.pixel_spin.setSpecialValueText("from the metadata")
        self.pixel_spin.setSuffix(" nm")
        self.pixel_spin.setValue(float(pixel_size_nm or 0.0))
        form.addRow("Pixel size:", self.pixel_spin)
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setObjectName("workers_spin")
        self.workers_spin.setRange(1, MAX_WORKERS)
        self.workers_spin.setValue(max(1, min(MAX_WORKERS, int(workers))))
        form.addRow("Worker processes:", self.workers_spin)
        for spin in (self.pixel_spin, self.workers_spin):
            spin.setMaximumWidth(180)
        buttons = QtWidgets.QHBoxLayout()
        self.compute_button = QtWidgets.QPushButton("Compute")
        self.compute_button.setObjectName("compute_button")
        self.compute_button.clicked.connect(self._on_compute)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setObjectName("cancel_button")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        self.export_button = QtWidgets.QPushButton("Export CSV...")
        self.export_button.setObjectName("export_button")
        self.export_button.clicked.connect(self._on_export)
        for b in (self.compute_button, self.cancel_button, self.export_button):
            buttons.addWidget(b)
        form.addRow(buttons)
        top.addLayout(form, 2)
        root.addLayout(top)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)
        self.status_label = QtWidgets.QLabel("Add axons and press Compute.")
        self.status_label.setObjectName("status_label")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.selection_widget = SelectionWidget(state, parent=central)
        self.selection_widget.state.changed.connect(self._on_selection)
        root.addWidget(self.selection_widget)
        self.counts_label = QtWidgets.QLabel()
        self.counts_label.setObjectName("counts_label")
        self.counts_label.setWordWrap(True)
        f = self.counts_label.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.counts_label.setFont(f)
        root.addWidget(self.counts_label)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.pair_table = QtWidgets.QTableWidget(0, len(PAIR_HEADERS))
        self.pair_table.setObjectName("pair_table")
        self.pair_table.setHorizontalHeaderLabels(list(PAIR_HEADERS))
        self.pair_table.verticalHeader().setVisible(False)
        self.pair_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.pair_table.setAlternatingRowColors(True)
        self.pair_table.setWordWrap(False)
        split.addWidget(self.pair_table)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QtWidgets.QLabel("<b>Every combination of the four criteria</b> &mdash; V (V+M) | axons; "
                                      "click a cell to select it"))
        self.combination_table = QtWidgets.QTableWidget(len(msel.criteria_subsets()), len(msel.COMBINATION_COLUMNS))
        self.combination_table.setObjectName("combination_table")
        self.combination_table.setHorizontalHeaderLabels([COMBINATION_HEADERS[c] for c in msel.COMBINATION_COLUMNS])
        for j, c in enumerate(msel.COMBINATION_COLUMNS):
            head = self.combination_table.horizontalHeaderItem(j)
            if head is not None:
                head.setToolTip(COMBINATION_HEADER_TIPS[c])
        self.combination_table.setVerticalHeaderLabels([subset_text(s) for s in msel.criteria_subsets()])
        self.combination_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.combination_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.combination_table.setItemDelegate(_OutlineDelegate(self.combination_table))
        self.combination_table.cellClicked.connect(self._on_cell_clicked)
        self.combination_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.combination_table.verticalHeader().setDefaultSectionSize(
            self.combination_table.fontMetrics().height() + 6)
        rl.addWidget(self.combination_table, 1)
        split.addWidget(right)
        split.setSizes([int(self.width() * 0.56), int(self.width() * 0.44)])
        root.addWidget(split, 1)
        self.missing_tooltips: List[str] = (apply_tooltips(self, EXPLORER_TOOLTIPS)
                                            + list(self.selection_widget.missing_tooltips))
        for p in inputs or []:
            self.add_input(p)
        self.refresh()

    # ------------------------------------------------------------ inputs
    def add_input(self, path: str) -> None:
        path = os.path.abspath(path)
        if path not in self.inputs():
            self.inputs_list.addItem(path)

    def inputs(self) -> List[str]:
        return [self.inputs_list.item(i).text() for i in range(self.inputs_list.count())]

    def files(self) -> List[str]:
        """The inputs as files (folders expanded as the batch expands them)."""
        import batch_columns as bc
        return bc.collect_inputs(self.inputs())

    def _on_add_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Axons for the viability explorer", "", "Localization files (*.hdf5 *.h5 *.csv *.npz);;All files (*)")
        for f in files:
            self.add_input(f)

    def _on_add_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder of axons for the viability explorer")
        if d:
            self.add_input(d)

    def _on_remove(self) -> None:
        for it in self.inputs_list.selectedItems():
            self.inputs_list.takeItem(self.inputs_list.row(it))

    def _on_clear(self) -> None:
        self.inputs_list.clear()

    # ------------------------------------------------------------ the computation (background)
    def is_running(self) -> bool:
        return self._thread is not None

    def compute(self, workers: Optional[int] = None) -> bool:
        """Compute (or read from the cache) every file's criteria in the background; False when running or empty."""
        if self._thread is not None:
            return False
        try:
            files = self.files()
        except (OSError, ValueError) as exc:
            self.status_label.setText(marked("bad", str(exc)))
            return False
        if not files:
            self.status_label.setText(marked("bad", "No localization file in the inputs."))
            return False
        n_workers = int(workers if workers is not None else self.workers_spin.value())
        n_workers = max(1, min(MAX_WORKERS, n_workers))
        px = float(self.pixel_spin.value()) or None
        self._generation += 1
        gen, relay, cancel, cache = self._generation, self._relay, self._cancel, self.cache_dir
        cancel.clear()
        self.criteria.clear()
        self.errors.clear()
        self.order = list(files)
        self.n_cached = self.n_computed = 0
        t0 = time.perf_counter()

        def body() -> None:
            info: Dict[str, Any] = {"cancelled": False, "error": ""}
            try:
                todo = []
                for p in files:
                    if cancel.is_set():
                        break
                    hit = None
                    try:
                        hit = load_cached(p, px, cache)
                    except (OSError, ValueError):
                        hit = None
                    if hit is not None:
                        relay.one.emit(gen, hit)
                    else:
                        todo.append(p)
                if n_workers <= 1 or len(todo) <= 1:
                    for p in todo:
                        if cancel.is_set():
                            break
                        relay.one.emit(gen, geometry_job(p, px, cache))
                elif todo:
                    _run_pool(todo, px, cache, n_workers, cancel, lambda d: relay.one.emit(gen, d))
                info["cancelled"] = cancel.is_set()
            except Exception as exc:  # noqa: BLE001 - shown
                info["error"] = f"{type(exc).__name__}: {exc}"
            info["seconds"] = time.perf_counter() - t0
            relay.done.emit(gen, info)

        self.progress_bar.setRange(0, len(files))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v of %m axon(s)")
        self.progress_bar.setVisible(True)
        self.compute_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText(f"Computing the criteria of {len(files)} axon(s) with {n_workers} worker(s) "
                                  "(cached files are read at once)...")
        self.refresh()
        self._thread = threading.Thread(target=body, daemon=True, name="viability-explorer")
        self._thread.start()
        return True

    def _on_compute(self) -> None:
        self.compute()

    def cancel(self) -> None:
        if self._thread is not None:
            self._cancel.set()
            self.status_label.setText(marked("warn", "Cancelling: the files being computed finish first..."))

    def wait(self, timeout: float = 1800.0) -> None:
        """Block until the computation ends (scripts and tests)."""
        t_end = time.perf_counter() + float(timeout)
        while self._thread is not None and time.perf_counter() < t_end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
        QtWidgets.QApplication.processEvents()

    def _on_one(self, gen: int, d: Any) -> None:
        if gen != self._generation:
            return
        path = str(d.get("path", ""))
        if d.get("ok") and d.get("criteria") is not None:
            try:
                crit = msel.AxonCriteria.from_json(d["criteria"])
                self.criteria[path] = crit
                if d.get("cached"):
                    self.n_cached += 1
                else:
                    self.n_computed += 1
            except Exception as exc:  # noqa: BLE001 - a broken entry is an error of that file
                self.errors[path] = f"{type(exc).__name__}: {exc}"
        else:
            self.errors[path] = str(d.get("error") or "unknown error")
        self.progress_bar.setValue(len(self.criteria) + len(self.errors))
        self.refresh()

    def _on_done(self, gen: int, info: Any) -> None:
        if gen != self._generation:
            return
        self._thread = None
        self.seconds = float(info.get("seconds", 0.0))
        self.progress_bar.setVisible(False)
        self.compute_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        parts = [f"{len(self.criteria)} of {len(self.order)} axon(s) ready ({self.n_cached} from the cache, "
                 f"{self.n_computed} computed) in {self.seconds:.1f} s"]
        if info.get("cancelled"):
            parts.append("cancelled")
        if info.get("error"):
            parts.append(f"error: {info['error']}")
        if self.errors:
            parts.append(f"{len(self.errors)} failed: " + "; ".join(
                f"{os.path.basename(p)}: {e}" for p, e in list(self.errors.items())[:4]))
        kind = "bad" if (self.errors or info.get("error")) else ("warn" if info.get("cancelled") else "good")
        self.status_label.setText(marked(kind, ". ".join(parts) + "."))
        self.refresh()
        self.computed.emit()

    # ------------------------------------------------------------ the selection, live
    def crits(self) -> List[msel.AxonCriteria]:
        """The axons computed so far, in the order of the inputs."""
        return [self.criteria[p] for p in self.order if p in self.criteria]

    def counts(self, spec: Optional[msel.SelectionSpec] = None) -> Dict[str, int]:
        """The current selection (or ``spec``) on every axon: pairs VIABLE / MARGINAL / all, axons with >= 1 VIABLE."""
        spec = spec if spec is not None else self.selection_widget.spec
        v = m = n = a = 0
        crits = self.crits()
        for c in crits:
            r = msel.evaluate_selection(c, spec)
            v += len(r.viable)
            m += len(r.marginal)
            n += len(r.pairs)
            a += int(bool(r.viable))
        return {"viable": v, "marginal": m, "pairs": n, "axons_viable": a, "axons": len(crits)}

    def current_cell(self) -> Optional[Tuple[Tuple[str, ...], str]]:
        """(criteria subset, column) of the current selection in the 16 x 4 table; None for AND / OR."""
        spec = self.selection_widget.spec
        if spec.rule == "v2":
            return spec.v2.enabled, "v2"
        if spec.rule == "v2c":
            return spec.v2c.enabled, f"v2c-{spec.variant}"
        return None

    def _on_selection(self, _spec: Any = None) -> None:
        try:
            self.refresh()
        except RuntimeError:        # the window is being destroyed
            return

    def refresh(self) -> None:
        """Counts, pair table and the 16 x 4 table under the current selection (pure evaluation, milliseconds)."""
        spec = self.selection_widget.spec
        c = self.counts(spec)
        if c["axons"]:
            text = (f"pairs passing: {c['viable']} ({c['viable'] + c['marginal']}) of {c['pairs']}; axons with >= 1 "
                    f"passing pair: {c['axons_viable']} of {c['axons']}")
            self.selection_widget.set_summary(f"{c['viable']} ({c['viable'] + c['marginal']}) of {c['pairs']} pairs, "
                                              f"{c['axons_viable']} of {c['axons']} axons")
        else:
            text = "No axon computed yet."
            self.selection_widget.set_summary("")
        if spec.exploratory and c["axons"]:
            text = f"EXPLORATORY SELECTION {spec.label} #{spec.hash}: " + text
        self.counts_label.setText(text)
        self._fill_pairs(spec)
        self._fill_combinations(spec)

    def _fill_pairs(self, spec: msel.SelectionSpec) -> None:
        t = self.pair_table
        t.setUpdatesEnabled(False)
        try:
            t.setRowCount(0)
            letter = spec.variant
            for path in self.order:
                crit = self.criteria.get(path)
                if crit is None:
                    continue
                res = msel.evaluate_selection(crit, spec)
                by = {(p.ring_a, p.ring_b): p for p in res.pairs}
                name = axon_name(path)
                for pc in crit.pairs:
                    ps = by[(pc.ring_a, pc.ring_b)]
                    s2 = pc.v2
                    sc = pc.side(letter)
                    tier, kind = {"viable": ("VIABLE", "good"), "marginal": ("MARGINAL", "warn")}.get(
                        str(ps.verdict), ("not selected", "bad"))
                    cells = [
                        (name, path, None), (f"{pc.ring_a}-{pc.ring_b}", "", None),
                        (tier, "; ".join(ps.reasons), kind),
                        (", ".join(ps.failing) or "-", "", None),
                        (", ".join(ps.unchecked_failing) or "-", "", None),
                        (f"{_yes(s2.peak_a)} / {_yes(s2.peak_b)}", "", None),
                        (_yes(s2.valley), "", None),
                        (f"{_pct(s2.f_a)} / {_pct(s2.f_b)} (min {_pct(s2.f_min)})", "", None),
                        (_pct(pc.d39), "D-39 expected leak copies (per ring)", None),
                        (_pct(pc.tnfix), "cluster-level expected leak copies (tnfix)", None),
                        ("n/a" if sc is None else f"{_yes(sc.peak_a)} / {_yes(sc.peak_b)}", f"v2c-{letter}", None),
                        ("n/a" if sc is None else (_yes(sc.valley) + (
                            " (in slab gap)" if sc.valley_raw and not sc.valley else "")), f"v2c-{letter}", None),
                        ("n/a" if sc is None else f"{_pct(sc.k_ratio_a)} / {_pct(sc.k_ratio_b)}", f"v2c-{letter}",
                         None),
                        (s2.verdict, "; ".join(s2.reasons), None),
                        ("n/a" if sc is None else sc.verdict, pc.v2c_error if sc is None else "; ".join(sc.reasons),
                         None),
                    ]
                    i = t.rowCount()
                    t.insertRow(i)
                    for j, (txt, tip, k) in enumerate(cells):
                        it = QtWidgets.QTableWidgetItem(str(txt))
                        if tip:
                            it.setToolTip(str(tip))
                        if k is not None:
                            it.setForeground(QtGui.QBrush(QtGui.QColor(verdict(k, dark=False))))
                            f = it.font()
                            f.setBold(True)
                            it.setFont(f)
                        t.setItem(i, j, it)
            t.resizeColumnsToContents()
        finally:
            t.setUpdatesEnabled(True)

    def combination(self, spec: Optional[msel.SelectionSpec] = None) -> List[msel.CombinationCell]:
        """The 16 x 4 cells with the leak estimates of ``spec`` (the current selection)."""
        spec = spec if spec is not None else self.selection_widget.spec
        return msel.combination_table(self.crits(), spec.v2.leak_estimate, spec.v2c.leak_estimate)

    def _fill_combinations(self, spec: msel.SelectionSpec) -> None:
        t = self.combination_table
        subsets = msel.criteria_subsets()
        cols = list(msel.COMBINATION_COLUMNS)
        cur = self.current_cell()
        cells = self.combination(spec) if self.crits() else []
        by = {(c.criteria, c.column): c for c in cells}
        for i, sub in enumerate(subsets):
            for j, col in enumerate(cols):
                cell = by.get((tuple(sub), col))
                it = QtWidgets.QTableWidgetItem("" if cell is None else cell.text)
                it.setTextAlignment(int(QtCore.Qt.AlignmentFlag.AlignCenter))
                if cell is not None:
                    it.setToolTip(f"{col}, criteria: {subset_text(sub)} (leak estimate "
                                  f"{spec.v2.leak_estimate if col == 'v2' else spec.v2c.leak_estimate}).\n"
                                  f"{cell.n_viable} pairs VIABLE, {cell.n_viable_marginal} VIABLE + MARGINAL, of "
                                  f"{cell.n_pairs}; {cell.n_axons_viable} of {cell.n_axons} axons with >= 1 VIABLE "
                                  "pair.\nClick to select this combination.")
                    if cell.n_viable_marginal == 0:
                        it.setForeground(QtGui.QBrush(QtGui.QColor(verdict("dim", dark=False))))
                is_cur = cur is not None and tuple(sub) == tuple(cur[0]) and col == cur[1]
                it.setData(OUTLINE_ROLE, bool(is_cur))
                if is_cur:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                t.setItem(i, j, it)
        t.viewport().update()
        if cur is not None:
            # the outlined cell in view (a permissive selection sits at the bottom of the 16 rows)
            for i, sub in enumerate(subsets):
                if tuple(sub) == tuple(cur[0]):
                    item = t.item(i, cols.index(cur[1]))
                    if item is not None:
                        t.scrollToItem(item)
                    break

    def _on_cell_clicked(self, row: int, col: int) -> None:
        subsets = msel.criteria_subsets()
        if not (0 <= row < len(subsets) and 0 <= col < len(msel.COMBINATION_COLUMNS)):
            return
        spec = self.selection_widget.spec
        self.selection_widget.state.set_spec(msel.spec_of_cell(
            subsets[row], msel.COMBINATION_COLUMNS[col], spec.v2.leak_estimate, spec.v2c.leak_estimate))

    # ------------------------------------------------------------ export
    def export_csv(self, path: str) -> List[str]:
        """``path``: the 16 x 4 table (one row per cell); ``<stem>_pairs.csv``: every pair under the current
        selection (``tools.mps_selection.selection_pair_rows``). Both carry the selection's label and hash. Never
        overwrites: an existing file gets ``_2``, ``_3``..."""
        spec = self.selection_widget.spec
        stem, ext = os.path.splitext(os.path.abspath(path))
        ext = ext or ".csv"
        k = 1
        while True:
            cand = stem if k == 1 else f"{stem}_{k}"
            if not (os.path.exists(cand + ext) or os.path.exists(f"{cand}_pairs{ext}")):
                break
            k += 1
        table_path, pairs_path = cand + ext, f"{cand}_pairs{ext}"
        os.makedirs(os.path.dirname(table_path) or ".", exist_ok=True)
        when = datetime.now().isoformat(timespec="seconds")
        head = {"selection_label": spec.label, "selection_hash": spec.hash, "selection_exploratory": spec.exploratory,
                "leak_v2": spec.v2.leak_estimate, "leak_v2c": spec.v2c.leak_estimate, "n_axons": len(self.crits()),
                "exported_at": when, "program": "MPS Explorer (tools/mps_viability_explorer.py)",
                "note": "geometry only (R8): no column statistic"}
        cols = ["criteria", "column", "n_viable", "n_viable_marginal", "n_axons_viable", "n_pairs", "n_axons",
                "is_current_selection"] + list(head)
        cur = self.current_cell()
        with open(table_path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for c in self.combination(spec):
                w.writerow(dict(head, criteria=" ".join(c.criteria) or "none", column=c.column, n_viable=c.n_viable,
                                n_viable_marginal=c.n_viable_marginal, n_axons_viable=c.n_axons_viable,
                                n_pairs=c.n_pairs, n_axons=c.n_axons,
                                is_current_selection=bool(cur is not None and tuple(cur[0]) == tuple(c.criteria)
                                                          and cur[1] == c.column)))
        rows: List[Dict[str, Any]] = []
        for p in self.order:
            crit = self.criteria.get(p)
            if crit is None:
                continue
            res = msel.evaluate_selection(crit, spec)
            for r in msel.selection_pair_rows(crit, res, input_key=p):
                rows.append(dict(r, source=p))
        pcols: List[str] = []
        for r in rows:
            for key in r:
                if key not in pcols:
                    pcols.append(key)
        with open(pairs_path, "w", encoding="utf-8", newline="") as fh:
            w2 = csv.writer(fh)
            w2.writerow(pcols or ["source"])
            for r in rows:
                w2.writerow(["" if r.get(c) is None or (isinstance(r.get(c), float) and not math.isfinite(r[c]))
                             else r.get(c) for c in pcols])
        return [table_path, pairs_path]

    def _on_export(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the viability explorer",
                                                        "viability_explorer.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            written = self.export_csv(path)
        except Exception as exc:  # noqa: BLE001 - shown
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Nothing was written:\n\n{exc}")
            return
        QtWidgets.QMessageBox.information(self, "Exported", "Written:\n" + "\n".join(written))

    def closeEvent(self, event: Any) -> None:
        self._cancel.set()
        self._generation += 1        # a late result is dropped
        self._thread = None
        super().closeEvent(event)


def _run_pool(todo: Sequence[str], px: Optional[float], cache: str, n_workers: int, cancel: threading.Event,
              emit: Any) -> None:
    """``geometry_job`` of every file in worker processes (one numerical thread each); stops submitting on cancel."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    keep = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    for v in keep:
        os.environ[v] = "1"
    try:
        ex = ProcessPoolExecutor(max_workers=min(int(n_workers), len(todo)), initializer=init_worker)
        try:
            pending = {ex.submit(geometry_job, p, px, cache) for p in todo}
            while pending:
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for f in done:
                    if not f.cancelled():
                        emit(f.result())
                if cancel.is_set():
                    for f in pending:
                        f.cancel()
                    pending = {f for f in pending if not f.cancelled()}
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
    finally:
        for v, old in keep.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old


# ============================================================================ stand-alone
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="The viability explorer (geometry only, R8) on picked axons.")
    ap.add_argument("--inputs", nargs="*", default=[], help="files or folders")
    ap.add_argument("--list", default=None, help="a text file with one file or folder per line")
    ap.add_argument("--demo", action="store_true", help="three SIMULATED axons (the Z quality demo cases)")
    ap.add_argument("--pixel-size", type=float, default=None, help="nm, for files without it in their metadata")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    paths = list(args.inputs)
    if args.list:
        with open(args.list, encoding="utf-8") as fh:
            paths += [ln.strip() for ln in fh if ln.strip()]
    # The application first (final audit, 2026-10-06): the demo's simulator once imported a validation harness that
    # sets QT_QPA_PLATFORM=offscreen, and an application created after that was never on screen.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if args.demo:
        import tempfile
        import batch_columns as bc
        from tools.mps_zquality_window import DEMO_CASES
        folder = tempfile.mkdtemp(prefix="mps_viability_demo_")
        for name, (contour, seed, s) in DEMO_CASES.items():
            paths.append(bc.write_simulated_input(os.path.join(folder, f"simulated_{name}.npz"), contour, seed, s))
    w = ViabilityExplorer(inputs=paths, pixel_size_nm=args.pixel_size, workers=args.workers)
    w.show()
    if paths:
        w.compute()
    return int(app.exec_())


if __name__ == "__main__":
    sys.exit(main())
