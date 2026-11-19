# -*- coding: utf-8 -*-
"""
The unanchored clusters window (package P15): the automatic discard of clusters that lie under the membrane, its
preview, and the decisions by hand.

Opened by the single button "Unanchored..." of the MPS analysis window; the main window also keeps one hidden
instance that runs the rule on every analysis shown, so that the analysis is ALWAYS reported with and without the
discard (decision D4) and the exports carry the v18 block instead of "not run". The rule and every number come from
``tools.mps_unanchored`` (decisions D1-D11); this module computes nothing of its own beyond what to draw.

What the window holds
---------------------
* The bar: method (automatic | widefield, when the axoplasm panel has flags | off = by hand only), reference ring
  (polar ring of the localizations, the default | smooth ring of the centres), "Exclude beyond (nm)" (400) and
  "Doubtful from (nm)" (250), Apply, Cancel, Load... and Save... (one save button: a dialog whose name is used).
* A progress bar and a status line (one line).
* The map: the slab's localizations, the reference ring (dashed), the cluster centres -- excluded with a cross,
  doubtful with an open circle, restored by hand with an open square, the rest as dots. A click on a centre
  excludes it (a reason is asked) or restores it; a click on a cluster that has a decision takes the decision
  back (the automatic verdict again).
* One line with what the discard would do ("would exclude N of K; M doubtful") and the rule's warnings.
* The record of the decisions by hand: cluster, position, depth, automatic state, decision, reason, date; Undo and
  "Back to automatic" for the selected row.

Nothing in the analysis changes until Apply (the preview is only drawn); Apply runs the analysis without the
excluded clusters in the background (bar and Cancel) and hands the pair to the main window, which shows both
columns in the results window and exports both. Decisions are kept by POSITION in ``<data>_unanchored.json``
beside the data file (D11), read when the window opens on that file.

English only, as the rest of the interface (the redesign's stage 4 makes it bilingual); the guided tour
("unanchored", ``tools.mps_tour_steps``) is bilingual and holds the long explanations.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_unanchored as mu
from tools.mps_plot_style import marked, neutral, role, set_title, style_dark, verdict

# ----------------------------------------------------------------------------------------------------
# Texts (one line each) and constants
# ----------------------------------------------------------------------------------------------------
WINDOW_TITLE = "Unanchored clusters"
TOUR_ID = "unanchored"
NO_INPUTS = "No analysis to look at: cluster an axon first."
METHOD_ITEMS: Tuple[Tuple[str, str], ...] = (
    ("Automatic (own ring)", mu.METHOD_AUTOMATIC),
    ("Widefield images", mu.METHOD_WIDEFIELD),
    ("Off (by hand only)", mu.METHOD_OFF),
)
REFERENCE_ITEMS: Tuple[Tuple[str, str], ...] = (
    ("Polar ring of the localizations", mu.REFERENCE_POLAR),
    ("Smooth ring of the centres", mu.REFERENCE_SMOOTH),
)
LOG_HEADERS: Tuple[str, ...] = ("Cluster", "x [nm]", "y [nm]", "Depth [nm]", "Automatic", "Decision", "Reason",
                                "Date")
MARGIN = 16
SPACING = 12
MIN_POINT_SIZE = 10.0
PROGRESS_STEPS = 1000
MAX_DRAWN_LOCS = 30000
# How close (screen pixels) a click must fall to a centre to pick it [ad hoc: a finger's width on the map]
CLICK_TOLERANCE_PX = 12.0
SAVE_FILTER = "Decisions (*.json)"
# The automatic run on every analysis shown can be switched off for scripted runs (golden references)
ENV_AUTO = "MPS_UNANCHORED_AUTO"


def auto_enabled() -> bool:
    """Whether the main window runs the rule (and the analysis with the discard) on every analysis shown;
    ``MPS_UNANCHORED_AUTO=0`` turns it off (scripted runs that must reproduce an older reference)."""
    return os.environ.get(ENV_AUTO, "").strip() != "0"


# ----------------------------------------------------------------------------------------------------
# Inputs and what the window says (no Qt)
# ----------------------------------------------------------------------------------------------------
@dataclass
class UnanchoredInputs:
    """What the main window hands over: the analysis shown (``AxonAnalysis``: slab localizations, kept centres,
    labels), the data file it came from (where the decisions file lives) and, when the axoplasm panel has them,
    its widefield flags for these same centres."""

    analysis: Any
    data_path: str = ""
    widefield: Optional[NDArray[np.bool_]] = None


def kept_labels(analysis: Any) -> NDArray[np.int64]:
    """The labels of the analysis' kept clusters, in the order of its centroids (0..K-1 when unknown)."""
    cent = np.asarray(analysis.centroids, dtype=np.float64).reshape(-1, 2)
    try:
        from tools.mps_analysis import good_cluster_labels
        lab = np.asarray(good_cluster_labels(analysis.labels, analysis.bad_report.bad_labels), dtype=np.int64)
        if lab.size == cent.shape[0]:
            return lab
    except Exception:                                  # noqa: BLE001 - labels are a convenience for the reader
        pass
    return np.arange(cent.shape[0], dtype=np.int64)


def _fmt(v: Any, fmt: str = "{:,.0f}") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return fmt.format(f) if math.isfinite(f) else ""


def log_rows(flags: mu.FinalFlags) -> List[Tuple[Optional[int], List[str]]]:
    """The record of the decisions by hand: (cluster row or None for an orphan, the cells), in the order they were
    made."""
    matched, orphans = mu.match_decisions(flags.centroids, flags.decisions)
    by_decision = {id(d): row for row, d in matched.items()}
    res = flags.result
    out: List[Tuple[Optional[int], List[str]]] = []
    for d in flags.decisions.decisions:
        row = by_decision.get(id(d))
        if row is None:
            out.append((None, ["-", _fmt(d.x_nm, "{:,.1f}"), _fmt(d.y_nm, "{:,.1f}"), "", "no cluster within "
                               f"{flags.decisions.tolerance_nm:g} nm", d.action, d.reason, d.date[:10]]))
            continue
        depth = "" if res is None else _fmt(res.depth_nm[row])
        state = "" if res is None else str(res.state[row])
        out.append((row, [str(int(flags.labels[row])), _fmt(d.x_nm, "{:,.1f}"), _fmt(d.y_nm, "{:,.1f}"), depth,
                          state, d.action, d.reason, d.date[:10]]))
    return out


def preview_kind(pv: mu.Preview) -> str:
    """The verdict kind of the preview line: 'bad' when the rule could not be evaluated, 'warn' with warnings."""
    res = pv.flags.result
    if pv.flags.method == mu.METHOD_AUTOMATIC and res is not None and not res.evaluated:
        return "bad"
    return "warn" if pv.warnings else ""


def same_flags(a: Optional[mu.FinalFlags], b: Optional[mu.FinalFlags]) -> bool:
    """True when two sets of flags exclude the same clusters for the same reasons of record (the analysis with the
    discard computed for one serves the other)."""
    if a is None or b is None:
        return False
    return (a.excluded.shape == b.excluded.shape and bool(np.array_equal(a.excluded, b.excluded))
            and a.record() == b.record() and a.centroids.shape == b.centroids.shape
            and bool(np.allclose(a.centroids, b.centroids)))


# ----------------------------------------------------------------------------------------------------
# Small widgets
# ----------------------------------------------------------------------------------------------------
class _Line(QtWidgets.QLabel):
    """One line: elided when too wide, whole in the tooltip; ``kind`` colours it and prefixes its mark."""

    def __init__(self, object_name: str = "") -> None:
        super().__init__("")
        if object_name:
            self.setObjectName(object_name)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._full = ""

    def set_full(self, text: str, kind: str = "") -> None:
        clean = " ".join(str(text).split())
        self._full = marked(kind, clean) if (kind and clean) else clean
        self.setStyleSheet(f"color: {verdict(kind, dark=False)};" if kind else "")
        self.setToolTip(self._full)
        self.setVisible(bool(self._full))          # an empty line takes no room
        self._elide()

    def full(self) -> str:
        return self._full

    def _elide(self) -> None:
        super().setText(self.fontMetrics().elidedText(self._full, QtCore.Qt.TextElideMode.ElideRight,
                                                      max(40, self.width())))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._elide()


class ReasonDialog(QtWidgets.QDialog):
    """Asks why a cluster is excluded (required) or restored (optional): the suggested reasons, or free text."""

    def __init__(self, action: str, label: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exclude a cluster" if action == mu.ACTION_EXCLUDE else "Restore a cluster")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        lay.setSpacing(SPACING)
        what = "exclude" if action == mu.ACTION_EXCLUDE else "restore"
        lay.addWidget(QtWidgets.QLabel(f"Why {what} cluster {label}?"))
        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(True)
        if action != mu.ACTION_EXCLUDE:
            self.combo.addItem("")
        self.combo.addItems(list(mu.SUGGESTED_REASONS))
        self.combo.setToolTip("Pick a reason or type your own.")
        lay.addWidget(self.combo)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def reason(self) -> str:
        return str(self.combo.currentText()).strip()


def ask_reason_dialog(parent: QtWidgets.QWidget, action: str, label: str) -> Optional[str]:
    """The reason typed in ``ReasonDialog``; None when cancelled (or empty for an exclusion, which needs one)."""
    dlg = ReasonDialog(action, label, parent)
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        return None
    text = dlg.reason()
    if action == mu.ACTION_EXCLUDE and not text:
        return None
    return text


def _table(name: str, headers: Sequence[str]) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setObjectName(name)
    t.setHorizontalHeaderLabels(list(headers))
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    t.setWordWrap(False)
    t.horizontalHeader().setStretchLastSection(True)
    try:
        from tools.copyable import copyable_table
        copyable_table(t)
    except Exception:                                  # noqa: BLE001 - copying is a convenience
        pass
    return t


# ----------------------------------------------------------------------------------------------------
# The window
# ----------------------------------------------------------------------------------------------------
class UnanchoredWindow(QtWidgets.QMainWindow):
    """The automatic discard of unanchored clusters: preview, decisions by hand, Apply."""

    # (analysis, flags): the rule's flags for the analysis shown (before any Apply)
    flags_ready = QtCore.pyqtSignal(object, object)
    # (analysis, flags, DiscardComparison): the analysis with and without the discard, after Apply
    applied = QtCore.pyqtSignal(object, object, object)
    # (one line, progress 0..PROGRESS_STEPS or -1, running): for a progress line elsewhere (the results window)
    progress_changed = QtCore.pyqtSignal(str, int, bool)

    def __init__(self, inputs_provider: Optional[Callable[[], Optional[UnanchoredInputs]]] = None, *,
                 inputs: Optional[UnanchoredInputs] = None,
                 ask_reason: Optional[Callable[[str, str], Optional[str]]] = None,
                 noise_replicates: int = mu.NOISE_REPLICATES,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        """``inputs_provider``: asked at every refresh (the analysis the results window shows now); ``inputs``: a
        fixed one (tests). ``ask_reason(action, cluster label)``: replaces the reason dialog (tests).
        ``noise_replicates``: of the rule's noise check (tests use fewer)."""
        super().__init__(parent)
        self.setObjectName("unanchored_window")
        self._provider = inputs_provider
        self._fixed = inputs
        self._ask_reason = ask_reason
        self._noise_replicates = int(noise_replicates)
        self.inputs: Optional[UnanchoredInputs] = None
        self.result: Optional[mu.UnanchoredResult] = None
        self.flags: Optional[mu.FinalFlags] = None
        self.applied_flags: Optional[mu.FinalFlags] = None
        self.comparison: Any = None
        self.decisions = mu.ManualDecisions()
        self.dirty = False
        self._unsaved: Dict[str, mu.ManualDecisions] = {}
        self._history: List[mu.ManualDecisions] = []
        self._run: Any = None
        self._run_kind = ""
        self._run_inputs: Optional[UnanchoredInputs] = None
        self._apply_flags: Optional[mu.FinalFlags] = None
        self._auto_apply = False
        self._rule_pending = False
        self._where: Dict[str, Any] = {"words": "", "frac": 0.0}
        self._click_items: List[pg.ScatterPlotItem] = []
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 820)
        f = self.font()
        if f.pointSizeF() < MIN_POINT_SIZE:
            f.setPointSizeF(MIN_POINT_SIZE)
            self.setFont(f)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        root.setSpacing(SPACING)
        root.addLayout(self._build_bar())
        prog = QtWidgets.QHBoxLayout()
        prog.setSpacing(SPACING)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("unanchored_progress")
        self.progress.setRange(0, PROGRESS_STEPS)
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)
        prog.addWidget(self.progress)
        self.status = _Line("unanchored_status")
        prog.addWidget(self.status, 1)
        root.addLayout(prog)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.plot = pg.PlotWidget()
        self.plot.setObjectName("unanchored_map")
        style_dark(self.plot)
        set_title(self.plot, "Slab localizations, reference ring and cluster centres")
        self.plot.setLabels(bottom="x [nm]", left="y [nm]")
        self.plot.setAspectLocked(True)
        self.plot.setMinimumHeight(420)
        self.plot.addLegend(offset=(10, 10), labelTextColor=neutral(True))
        split.addWidget(self.plot)
        side = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACING // 2)
        self.line_preview = _Line("unanchored_preview")
        fb = self.line_preview.font()
        fb.setBold(True)
        self.line_preview.setFont(fb)
        lay.addWidget(self.line_preview)
        self.line_warnings = [_Line(f"unanchored_warning_{i}") for i in range(3)]
        for ln in self.line_warnings:
            lay.addWidget(ln)
        self.line_file = _Line("unanchored_file")
        lay.addWidget(self.line_file)
        lay.addSpacing(SPACING)
        head = QtWidgets.QLabel("Decisions by hand")
        fh = head.font()
        fh.setBold(True)
        fh.setPointSizeF(fh.pointSizeF() + 1.0)
        head.setFont(fh)
        lay.addWidget(head)
        self.table = _table("unanchored_log", LOG_HEADERS)
        lay.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(SPACING)
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_undo.setObjectName("unanchored_undo")
        self.btn_undo.setToolTip("Take back the last decision by hand.")
        self.btn_undo.clicked.connect(self.undo)
        row.addWidget(self.btn_undo)
        self.btn_back = QtWidgets.QPushButton("Back to automatic")
        self.btn_back.setObjectName("unanchored_back")
        self.btn_back.setToolTip("Remove the selected decision: that cluster follows the rule again.")
        self.btn_back.clicked.connect(self._on_back)
        row.addWidget(self.btn_back)
        row.addStretch(1)
        lay.addLayout(row)
        side.setMinimumWidth(520)                  # [ad hoc] the record table whole at 10 pt
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._show_progress)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(400)
        self._debounce.timeout.connect(self.recompute)
        self._sync_buttons()
        try:
            from tools import ui_search
            ui_search.install(self)
        except Exception:                              # noqa: BLE001 - the search is a convenience
            pass

    # ---------------------------------------------------------------- bar
    def _build_bar(self) -> QtWidgets.QHBoxLayout:
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(SPACING)
        lab = QtWidgets.QLabel("Method")
        bar.addWidget(lab)
        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.setObjectName("unanchored_method")
        for text, key in METHOD_ITEMS:
            self.combo_method.addItem(text, key)
        self.combo_method.setToolTip("Where the discard comes from: the spectrin's own ring, widefield images, or none.")
        self.combo_method.currentIndexChanged.connect(self._on_method)
        lab.setBuddy(self.combo_method)
        bar.addWidget(self.combo_method)
        bar.addSpacing(SPACING)
        lab = QtWidgets.QLabel("Ring")
        bar.addWidget(lab)
        self.combo_reference = QtWidgets.QComboBox()
        self.combo_reference.setObjectName("unanchored_reference")
        for text, key in REFERENCE_ITEMS:
            self.combo_reference.addItem(text, key)
        self.combo_reference.setToolTip("The ring the depth of every centre is measured from.")
        self.combo_reference.currentIndexChanged.connect(lambda _i: self._debounce.start())
        lab.setBuddy(self.combo_reference)
        bar.addWidget(self.combo_reference)
        bar.addSpacing(SPACING)
        self.spin_exclude = self._spin("unanchored_exclude", mu.DEFAULT_EXCLUDE_NM,
                                       "Centres deeper than this under the ring are excluded (nm).")
        self.spin_doubt = self._spin("unanchored_doubt", mu.DEFAULT_DOUBT_NM,
                                     "Centres deeper than this are marked doubtful and kept (nm).")
        for text, spin in (("Exclude beyond (nm)", self.spin_exclude), ("Doubtful from (nm)", self.spin_doubt)):
            lab = QtWidgets.QLabel(text)
            lab.setBuddy(spin)
            bar.addWidget(lab)
            bar.addWidget(spin)
            bar.addSpacing(SPACING // 2)
        bar.addStretch(1)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.setObjectName("unanchored_apply")
        self.btn_apply.setToolTip("Analyse the axon again without the excluded clusters; both columns are kept.")
        self.btn_apply.clicked.connect(self.apply)
        bar.addWidget(self.btn_apply)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setObjectName("unanchored_cancel")
        self.btn_cancel.setToolTip("Stop the computation at its next step; nothing it computed is used.")
        self.btn_cancel.clicked.connect(self.cancel)
        bar.addWidget(self.btn_cancel)
        self.btn_load = QtWidgets.QPushButton("Load...")
        self.btn_load.setObjectName("unanchored_load")
        self.btn_load.setToolTip("Read decisions by hand saved under another name.")
        self.btn_load.clicked.connect(self._on_load)
        bar.addWidget(self.btn_load)
        self.btn_save = QtWidgets.QPushButton("Save...")
        self.btn_save.setObjectName("unanchored_save")
        self.btn_save.setToolTip("Save the decisions by hand; beside the data file they load by themselves.")
        self.btn_save.clicked.connect(self._on_save)
        bar.addWidget(self.btn_save)
        return bar

    def _spin(self, name: str, value: float, tip: str) -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setObjectName(name)
        s.setDecimals(0)
        s.setRange(0.0, 3000.0)
        s.setSingleStep(25.0)
        s.setValue(float(value))
        s.setToolTip(tip)
        s.valueChanged.connect(lambda _v: self._debounce.start())
        return s

    # ---------------------------------------------------------------- state
    def method(self) -> str:
        return str(self.combo_method.currentData())

    def reference(self) -> str:
        return str(self.combo_reference.currentData())

    def running(self) -> bool:
        return self._run is not None

    def one_line_texts(self) -> List[str]:
        """Every text of the window that must be a single line (a test reads them)."""
        texts = [self.status.full(), self.line_preview.full(), self.line_file.full()]
        texts += [ln.full() for ln in self.line_warnings]
        for w in (self.combo_method, self.combo_reference, self.spin_exclude, self.spin_doubt, self.btn_apply,
                  self.btn_cancel, self.btn_load, self.btn_save, self.btn_undo, self.btn_back):
            texts.append(w.toolTip())
        return texts

    def _sync_buttons(self) -> None:
        run = self.running()
        have = self.inputs is not None
        self.btn_cancel.setEnabled(run and not self._run.cancelled())
        self.btn_apply.setEnabled(not run and self.flags is not None)
        self.btn_save.setEnabled(not run and have)
        self.btn_load.setEnabled(not run and have)
        self.btn_undo.setEnabled(not run and bool(self._history))
        self.btn_back.setEnabled(not run and len(self.decisions) > 0)
        wf = have and self.inputs is not None and self.inputs.widefield is not None
        model: Any = self.combo_method.model()
        item = model.item(1) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(bool(wf))
        for w in (self.combo_method, self.combo_reference, self.spin_exclude, self.spin_doubt):
            w.setEnabled(not run)
        auto = self.method() == mu.METHOD_AUTOMATIC
        for w in (self.combo_reference, self.spin_exclude, self.spin_doubt):
            w.setEnabled(not run and auto)

    def _get_inputs(self) -> Optional[UnanchoredInputs]:
        if self._provider is not None:
            try:
                got = self._provider()
            except Exception:                          # noqa: BLE001 - a provider failure is "no analysis"
                got = None
            return got
        return self._fixed

    # ---------------------------------------------------------------- entry points
    def analysis_changed(self, auto_apply: bool = False) -> bool:
        """Read the analysis shown now, its decisions file, and run the rule (in the background). ``auto_apply``:
        when the rule is done, compute the analysis with the discard as well (the main window's automatic run).
        False when there is no analysis."""
        inputs = self._get_inputs()
        old_path = None if self.inputs is None else self.inputs.data_path
        if self.inputs is not None and self.dirty and old_path:
            self._unsaved[old_path] = self.decisions
        self.inputs = inputs
        self.result = None
        self.flags = None
        self.applied_flags = None
        self.comparison = None
        if inputs is None or inputs.analysis is None:
            self.status.set_full(NO_INPUTS, "warn")
            self._draw()
            self._sync_buttons()
            return False
        if inputs.data_path != old_path:
            self._history = []
            self._load_default_decisions()
        if self.method() == mu.METHOD_WIDEFIELD and inputs.widefield is None:
            self.combo_method.setCurrentIndex(0)
        self._auto_apply = bool(auto_apply)
        self.recompute()
        return True

    def _load_default_decisions(self) -> None:
        assert self.inputs is not None
        path = self.inputs.data_path
        if path in self._unsaved:
            self.decisions = self._unsaved.pop(path)
            self.dirty = True
            self.line_file.set_full("Decisions by hand not saved yet (kept from before).", "warn")
            return
        default = mu.decisions_path(path) if path else ""
        self.dirty = False
        if not default:
            self.decisions = mu.ManualDecisions()
            self.line_file.set_full("No data file: the decisions can be saved only by name.", "dim")
            return
        try:
            self.decisions = mu.load_decisions(default)
        except mu.ManualDecisionsError as exc:
            self.decisions = mu.ManualDecisions(path=default)
            self.line_file.set_full(f"Decisions file not read: {exc}", "bad")
            return
        n = len(self.decisions)
        self.line_file.set_full(f"{n} decision(s) by hand read from {os.path.basename(default)}." if n else
                                f"No decisions by hand yet ({os.path.basename(default)}).", "dim")

    def recompute(self) -> None:
        """Run the rule again with the bar's values (in the background), or only re-apply the decisions when the
        method is not automatic."""
        self._debounce.stop()
        if self.inputs is None or self.inputs.analysis is None:
            return
        if self.running():
            self._rule_pending = True
            return
        if self.spin_doubt.value() >= self.spin_exclude.value():
            self.status.set_full("'Doubtful from' must be below 'Exclude beyond'.", "warn")
            return
        if self.method() != mu.METHOD_AUTOMATIC:
            self.result = None
            self._refresh_flags()
            if self._auto_apply:
                self._auto_apply = False
                self.apply()
            return
        analysis = self.inputs.analysis
        x = np.asarray(analysis.x_slab, dtype=np.float64)
        y = np.asarray(analysis.y_slab, dtype=np.float64)
        cent = np.asarray(analysis.centroids, dtype=np.float64).reshape(-1, 2)
        labels = kept_labels(analysis)
        reference, exclude, doubt = self.reference(), float(self.spin_exclude.value()), float(self.spin_doubt.value())
        reps = self._noise_replicates

        def compute() -> mu.UnanchoredResult:
            return mu.auto_unanchored(x, y, cent, labels, reference=reference, exclude_nm=exclude, doubt_nm=doubt,
                                      noise_replicates=reps)

        self._start("rule", compute, "the unanchored-cluster rule", mu.PROGRESS_PREFIX)

    def apply(self) -> bool:
        """The analysis without the excluded clusters, in the background; the pair goes to the main window
        (``applied``). False when there is nothing to apply or a run is going."""
        if self.running() or self.flags is None or self.inputs is None:
            return False
        flags = self.flags
        analysis = self.inputs.analysis
        if same_flags(flags, self.applied_flags) and self.comparison is not None:
            self.status.set_full("Already applied: the results window shows both columns.", "good")
            return False

        def compute() -> Any:
            return mu.compare_unanchored(analysis, flags)

        self._apply_flags = flags
        self._start("apply", compute, "the analysis without the excluded clusters", "")
        return True

    # ---------------------------------------------------------------- background runs
    def _start(self, kind: str, compute: Callable[[], Any], words: str, prefix: str) -> None:
        from tools import mps_progress
        from tools.mps_background_run import BackgroundRun, PreparedRun
        where: Dict[str, Any] = {"words": "starting", "frac": 0.0}
        self._where = where

        def observe(text: str, done: Optional[int], total: Optional[int]) -> None:
            if prefix and text.startswith(prefix):
                where["words"] = text[len(prefix):]
                if done is not None and total:
                    where["frac"] = max(float(where["frac"]), float(done) / float(total))
                return
            frac, said = mps_progress.file_fraction(text, done, total)
            where["words"] = said
            if frac is not None:
                # the analysis' own stages start at the perimeter: stretch them over the whole bar
                lo = 0.30
                where["frac"] = max(float(where["frac"]), min(1.0, max(0.0, (frac - lo) / (1.0 - lo))))

        def run_it() -> Any:
            with mps_progress.observing(observe):
                return compute()

        run = BackgroundRun(PreparedRun(compute=run_it, finish=lambda v: v, words=words), parent=self)
        run.ended.connect(lambda status, value, tb: self._on_ended(run, status, value, tb))
        self._run = run
        self._run_kind = kind
        self._run_inputs = self.inputs
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._sync_buttons()
        run.start()
        self._timer.start()
        self._show_progress()

    def _show_progress(self) -> None:
        run = self._run
        if run is None:
            return
        value = int(PROGRESS_STEPS * min(1.0, float(self._where.get("frac", 0.0))))
        self.progress.setValue(value)
        what = "Rule" if self._run_kind == "rule" else "Analysis without the excluded clusters"
        if run.cancelled():
            line, kind = f"Cancelling ({run.elapsed_s():.0f} s): it stops at the next step.", "warn"
        else:
            line, kind = f"{what} ({run.elapsed_s():.0f} s): {self._where.get('words', '')}...", ""
        self.status.set_full(line, kind)
        self.progress_changed.emit(f"Unanchored clusters - {line}", value, True)

    def cancel(self) -> None:
        if self._run is not None:
            self._run.cancel()
            self._auto_apply = False
            self._show_progress()
            self._sync_buttons()

    def wait_run(self, timeout_s: float = 120.0) -> bool:
        """Pump events until no run is going (tests and scripts)."""
        import time
        end = time.monotonic() + float(timeout_s)
        while time.monotonic() < end:
            QtWidgets.QApplication.processEvents()
            if self._run is None and not self._rule_pending:
                return True
            time.sleep(0.01)
        return self._run is None

    def _on_ended(self, run: Any, status: str, value: Any, tb: str) -> None:
        if run is not self._run:
            return
        kind = self._run_kind
        self._run = None
        self._timer.stop()
        took = run.elapsed_s()
        run.deleteLater()
        self.progress.setVisible(False)
        if status == "done" and self._run_inputs is not self.inputs:
            # another analysis came meanwhile: this result is about the old one (the pending rule run follows)
            self._rule_pending = self.inputs is not None
            status = "stale"
        if status == "stale":
            pass
        elif status == "done":
            if kind == "rule":
                self.result = value
                self._refresh_flags()
                self.status.set_full(f"Rule done in {took:.1f} s; nothing changes in the analysis until Apply."
                                     if not self._auto_apply else "Rule done; analysing without the excluded "
                                     "clusters...")
            else:
                self.comparison = value
                self.applied_flags = self._apply_flags
                n = self._apply_flags.n_excluded
                self.status.set_full(f"Applied in {took:.0f} s: {n} cluster(s) left out of the 'discard applied' "
                                     "column.", "good")
                if self.inputs is not None:
                    self.applied.emit(self.inputs.analysis, self._apply_flags, value)
        elif status == "cancelled":
            self._auto_apply = False
            self._rule_pending = False
            self.status.set_full(f"Cancelled after {took:.0f} s: nothing it computed is used.", "warn")
        else:
            self._auto_apply = False
            last = [ln for ln in str(tb).strip().splitlines() if ln.strip()]
            self.status.set_full("Failed: " + (last[-1] if last else str(value)), "bad")
        self.progress_changed.emit(self.status.full(), -1, False)
        self._sync_buttons()
        if self._rule_pending:
            self._rule_pending = False
            self.recompute()
        elif status == "done" and kind == "rule" and self._auto_apply:
            self._auto_apply = False
            analysis = None if self.inputs is None else self.inputs.analysis
            if getattr(analysis, "contour_2opt", None) == "one start":
                # 'one start' was chosen for a long perimeter: the automatic run does not undo that choice
                self.status.set_full("Not applied by itself: this perimeter was built from one start; press Apply "
                                     "to compute the discard column anyway.", "warn")
                self.progress_changed.emit(self.status.full(), -1, False)
            elif not self.apply():
                self.progress_changed.emit(self.status.full(), -1, False)

    # ---------------------------------------------------------------- flags and decisions
    def _refresh_flags(self) -> None:
        if self.inputs is None:
            return
        analysis = self.inputs.analysis
        method = self.method()
        try:
            self.flags = mu.final_flags(self.result if method == mu.METHOD_AUTOMATIC else None, self.decisions,
                                        method=method, widefield=self.inputs.widefield,
                                        centroids=np.asarray(analysis.centroids, dtype=np.float64).reshape(-1, 2),
                                        labels=list(kept_labels(analysis)),
                                        data_source=os.path.basename(self.inputs.data_path))
        except ValueError as exc:
            self.flags = None
            self.status.set_full(f"Not evaluated: {exc}", "bad")
        if self.flags is not None:
            self.flags_ready.emit(analysis, self.flags)
        self._draw()
        self._fill_log()
        self._sync_buttons()

    def _on_method(self, _index: int) -> None:
        if self.inputs is None:
            self._sync_buttons()
            return
        self.recompute()

    def decision_at(self, row: int) -> Optional[mu.ManualDecision]:
        if self.flags is None:
            return None
        matched, _orph = mu.match_decisions(self.flags.centroids, self.decisions)
        return matched.get(int(row))

    def toggle_cluster(self, row: int) -> bool:
        """A click on cluster ``row``: a decision on it is taken back (the rule again); otherwise an excluded cluster
        is restored and a kept one excluded, with the reason asked. False when nothing changed."""
        if self.flags is None or self.running() or not (0 <= int(row) < self.flags.centroids.shape[0]):
            return False
        row = int(row)
        cx, cy = (float(v) for v in self.flags.centroids[row])
        label = str(int(self.flags.labels[row]))
        before = self.decisions
        if self.decision_at(row) is not None:
            new = mu.remove_decision(self.decisions, cx, cy)
        else:
            action = mu.ACTION_RESTORE if bool(self.flags.excluded[row]) else mu.ACTION_EXCLUDE
            asker = self._ask_reason or (lambda a, lab: ask_reason_dialog(self, a, lab))
            reason = asker(action, label)
            if reason is None:
                return False
            try:
                new = mu.add_decision(self.decisions, cx, cy, action, reason)
            except mu.ManualDecisionsError as exc:
                self.status.set_full(str(exc), "warn")
                return False
        self._history.append(before)
        self.decisions = new
        self.dirty = True
        self.line_file.set_full("Decisions by hand changed: Save... keeps them; Apply uses them.", "warn")
        self._refresh_flags()
        return True

    def undo(self) -> None:
        if not self._history or self.running():
            return
        self.decisions = self._history.pop()
        self.dirty = True
        self._refresh_flags()

    def _on_back(self) -> None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            self.status.set_full("Select a decision in the table first.", "warn")
            return
        self.back_to_automatic(rows[0].row())

    def back_to_automatic(self, table_row: int) -> bool:
        """Remove the decision shown in row ``table_row`` of the record."""
        if not (0 <= table_row < len(self.decisions)) or self.running():
            return False
        d = self.decisions.decisions[table_row]
        self._history.append(self.decisions)
        self.decisions = replace(self.decisions, decisions=tuple(x for x in self.decisions.decisions if x is not d))
        self.dirty = True
        self._refresh_flags()
        return True

    # ---------------------------------------------------------------- save / load
    def default_decisions_path(self) -> str:
        if self.inputs is None or not self.inputs.data_path:
            return "decisions" + mu.DECISIONS_SUFFIX
        return mu.decisions_path(self.inputs.data_path)

    def save_to(self, path: str) -> str:
        """Write the decisions by hand at ``path`` (the name chosen). Refuses to replace a file that is not a
        decisions file. Returns the path written."""
        src = "" if self.inputs is None else os.path.basename(self.inputs.data_path)
        written = mu.save_decisions(replace(self.decisions, source=src or self.decisions.source, path=path), path)
        self.dirty = False
        default = self.default_decisions_path()
        if os.path.normcase(os.path.abspath(written)) == os.path.normcase(os.path.abspath(default)):
            self.line_file.set_full(f"Saved {os.path.basename(written)}: it loads by itself with this file.", "good")
        else:
            self.line_file.set_full(f"Saved {os.path.basename(written)}; only {os.path.basename(default)} loads by "
                                    "itself.", "warn")
        return written

    def load_from(self, path: str) -> int:
        """Read decisions by hand from ``path`` (replacing the ones in memory); returns how many."""
        got = mu.load_decisions(path)
        self._history.append(self.decisions)
        self.decisions = got
        self.dirty = True
        self.line_file.set_full(f"{len(got)} decision(s) read from {os.path.basename(path)}.", "dim")
        self._refresh_flags()
        return len(got)

    def _on_save(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the decisions by hand",
                                                        self.default_decisions_path(), SAVE_FILTER)
        if not path:
            return
        try:
            self.save_to(path)
        except (OSError, ValueError) as exc:
            self.line_file.set_full(f"Not saved: {exc}", "bad")

    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Read decisions by hand",
                                                        os.path.dirname(self.default_decisions_path()), SAVE_FILTER)
        if not path:
            return
        try:
            self.load_from(path)
        except (OSError, ValueError) as exc:
            self.line_file.set_full(f"Not read: {exc}", "bad")

    # ---------------------------------------------------------------- drawing
    def _draw(self) -> None:
        self.plot.clear()
        self._click_items = []
        for ln in self.line_warnings:
            ln.set_full("")
        if self.inputs is None or self.inputs.analysis is None:
            self.line_preview.set_full("")
            return
        a = self.inputs.analysis
        x = np.asarray(a.x_slab, dtype=np.float64).reshape(-1)
        y = np.asarray(a.y_slab, dtype=np.float64).reshape(-1)
        if x.size:
            step = max(1, int(math.ceil(x.size / MAX_DRAWN_LOCS)))
            self.plot.addItem(pg.ScatterPlotItem(x=x[::step], y=y[::step], size=2, pen=pg.mkPen(None),
                                                 brush=pg.mkBrush(role("noise")), name="slab localizations"))
        flags = self.flags
        if flags is None:
            self.line_preview.set_full("Computing the rule..." if self.running() else "")
            return
        pv = mu.preview(flags)
        if pv.ring is not None and len(pv.ring) >= 3:
            ring = np.vstack([pv.ring, pv.ring[:1]])
            self.plot.plot(ring[:, 0], ring[:, 1], pen=pg.mkPen(neutral(True), width=2,
                                                                style=QtCore.Qt.PenStyle.DashLine),
                           name="reference ring")
        c = flags.centroids
        res = flags.result
        doubtful = (res.doubtful & ~flags.excluded) if (res is not None and flags.method == mu.METHOD_AUTOMATIC) \
            else np.zeros(c.shape[0], dtype=bool)
        restored = np.asarray([s == mu.SOURCE_RESTORED for s in flags.source], dtype=bool)
        excluded = flags.excluded
        kept = ~excluded & ~doubtful & ~restored
        exc_text = f"excluded (> {res.exclude_nm:g} nm or by hand)" if res is not None else "excluded"
        doubt_text = (f"doubtful ({res.doubt_nm:g}-{res.exclude_nm:g} nm, kept)" if res is not None
                      else "doubtful (kept)")
        classes = (
            (kept, "centres", "o", 7, pg.mkBrush(role("centroid")), pg.mkPen(None)),
            (doubtful, doubt_text, "o", 14, pg.mkBrush(None), pg.mkPen(role("warn"), width=2)),
            (restored, "restored by hand", "s", 13, pg.mkBrush(None), pg.mkPen(neutral(True), width=2)),
            (excluded, exc_text, "x", 15, pg.mkBrush(role("discarded")), pg.mkPen(role("discarded"), width=1)),
        )
        for mask, name, symbol, size, brush, pen in classes:
            idx = np.flatnonzero(mask)
            item = pg.ScatterPlotItem(x=c[idx, 0], y=c[idx, 1], data=list(idx), symbol=symbol, size=size,
                                      brush=brush, pen=pen, name=f"{name} ({idx.size})")
            item.sigClicked.connect(self._on_points_clicked)
            self.plot.addItem(item)
            self._click_items.append(item)
        self.line_preview.set_full(pv.text[0].upper() + pv.text[1:] + ".", preview_kind(pv))
        for ln, w in zip(self.line_warnings, pv.warnings):
            ln.set_full(w, "warn")
        if len(pv.warnings) > len(self.line_warnings):
            self.line_warnings[-1].set_full(f"{len(pv.warnings) - len(self.line_warnings) + 1} more warning(s): "
                                            + " | ".join(pv.warnings[len(self.line_warnings) - 1:]), "warn")
        self.plot.autoRange()

    def _on_points_clicked(self, _item: Any, points: Any, *_rest: Any) -> None:
        try:
            pts = list(points)
        except TypeError:
            return
        if not pts:
            return
        data = pts[0].data()
        if data is None:
            return
        self.toggle_cluster(int(data))

    def _fill_log(self) -> None:
        rows = [] if self.flags is None else log_rows(self.flags)
        self.table.setRowCount(len(rows))
        for i, (_row, cells) in enumerate(rows):
            for j, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if j in (1, 2, 3):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()
        pad = 2 * self.table.fontMetrics().horizontalAdvance("M")
        for j in range(self.table.columnCount()):
            head = self.table.horizontalHeaderItem(j)
            need = (self.table.fontMetrics().horizontalAdvance(head.text()) if head is not None else 0) + pad
            self.table.setColumnWidth(j, max(self.table.columnWidth(j), need))
        self.table.horizontalHeader().setStretchLastSection(True)

    # ---------------------------------------------------------------- closing
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Closing only hides the window (the main window keeps it for the automatic run); a run in progress goes
        on, its progress is still shown in the results window."""
        super().closeEvent(event)


__all__ = ["UnanchoredWindow", "UnanchoredInputs", "log_rows", "preview_kind", "same_flags", "kept_labels",
           "auto_enabled", "ReasonDialog", "TOUR_ID", "ENV_AUTO"]
