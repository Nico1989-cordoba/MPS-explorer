# -*- coding: utf-8 -*-
"""
"Simulated-null p (not calibrated)...": the optional per-axon simulated
null of the column review window (Q-30), restricted by rule v2 (D-41).

It runs ``power_columns.py simnull`` on the review's selection in a
separate process: the axon is written to an NPZ (x, y, z, frame, lp, lpz,
n_frames), the measured configuration is fitted to it, and R simulated
axons without columns are analysed; with ``--zq-select v2-viable`` the
observed axon and every simulated one keep only the pairs rule v2 calls
VIABLE (``v2-viable+marginal``: also the marginal ones, a sensitivity),
and a simulated axon counts only when its selection matches the
observed one. The p is that of the arc test on the centroid membrane
over the selected pairs (``arcc_z_A_sel``) against the accepted
simulated axons, (#{sim >= obs} + 1) / (R + 1).

The p is NOT calibrated: the calibration of this simulated null (H5-E)
was closed as not accepted (D-41, Q-33: the null may under-simulate the
axial leak). The window says so wherever the number appears.

The lumen cleaning of the observed axon is the automatic rule plus the
user's manual edits (``--lumen-edits``); the simulated axons are cleaned
by the automatic rule only (D-35b), and without the widefield masks
(isolation only: there is no writer of the widefield JSON yet).

H6 toggles (D-43): the dialog holds the program's criteria switches. Under
an exploratory selection the run gets ``--zq-selection <label>``: simnull
applies the SAME selection to the observed axon and to every simulated one
(its rows carry ``zq_mode`` "v2-viable@sel:<hash>" and a sidecar
``<name>_selection.json``; a folder of another selection is refused on a
restart). The selection is fixed when the run starts; the result names it
and shows how many selections were tried on this axon in this session
(``--exploration-log``: every run is logged).

UI stage 0 (the user's manual test of 2026-10-05): the dialog says what the
test is in plain words (``SIMNULL_EXPLANATION``); it warns under the spin
when fewer than 19 accepted simulated axons are asked for (the p can never
reach 0.05) and the Start button asks first; the progress line reads
power_columns' whole-run tag "[attempt a of m; accepted k of n]"; and the
result puts "NOT INTERPRETABLE" with its reasons (too few accepted, the
fidelity checks of the report not met) and the fidelity summary ABOVE the p.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import math
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_selection as msel
from tools.mps_background_process import REPO_ROOT, BackgroundRun, is_inside, open_folder, python_executable, tail
from tools.mps_plot_style import marked, verdict
from tools.mps_selection_ui import SelectionWidget, app_log_path, counter_line, variants_tried
from tools.mps_tooltips import apply_tooltips

__all__ = ["SIMNULL_EXPLANATION", "SIMNULL_RESULT_LABEL", "SIMNULL_UI_TOOLTIPS", "SimnullDialog", "SimnullResult",
           "parse_progress_tag", "read_simnull_result", "small_n_text", "write_simnull_npz"]

SIMNULL_RESULT_LABEL = "Simulated-null p (NOT calibrated: H5-E closed as not accepted, Q-33)"
DEFAULT_REPLICATES = 99
DEFAULT_N_NULL = 199
MAX_WORKERS = 12
# Below this many accepted simulated axons the p can never reach 0.05 (its floor is 1 / (R + 1), D-41)
MIN_REPLICATES_FOR_005 = 19
# power_columns simnull draws at most this many attempts per accepted axon asked for (its --zq-max-attempts default)
MAX_ATTEMPTS_PER_REPLICATE = 3
MODES: Tuple[Tuple[str, str], ...] = (("v2-viable", "VIABLE pairs (primary)"),
                                      ("v2-viable+marginal", "VIABLE + MARGINAL pairs (sensitivity)"))
_REPLICATE_LINE = re.compile(r"^\s+replicate \d+ seed")
# The tag power_columns.simnull_progress_tag prints at the end of every replicate line (UI stage 0)
_PROGRESS_TAG = re.compile(r"\[attempt (\d+) of (\d+); accepted (\d+) of (\d+)\]")

# What the test is, in plain words (UI stage 0, manual test of 2026-10-05; D-28, D-30, D-41). At most 90 words.
SIMNULL_EXPLANATION = (
    "The usual column p compares this axon with copies of itself, one ring turned at random. Those copies lack the "
    "axial leak (clusters repeated in the next ring, in register, that mimic columns), so on real axons that p is "
    "too optimistic. This test builds simulated axons like this one (contour, density, z precision, leak) without "
    "columns, analyses each identically, and counts how often one looks as columnar as this axon. Experimental: its "
    "calibration was not accepted. Trust it only if the simulations resemble this axon and many were accepted.")

SIMNULL_UI_TOOLTIPS: Dict[str, str] = {
    "replicates_spin": "How many simulated axons must be ACCEPTED (their rule-v2 selection equal to the observed "
                       "one's). More gives a finer p (the smallest possible p is 1 / (R + 1)) and takes longer.",
    "workers_spin": "Processes running simulated axons at once (at most 12 on this laptop).",
    "n_null_spin": "The rotation-null size inside every analysis (observed and simulated). 199 is the simnull "
                   "default.",
    "mode_combo": "Which pairs the p is computed on: the VIABLE pairs (the primary analysis) or VIABLE + MARGINAL "
                  "(a sensitivity check; only when the axon has marginal pairs).",
    "out_edit": "Where the run writes its files (observed row, simulated rows, configuration, report, log). Outside "
                "the program's folder. Starting again with the same folder resumes the run.",
    "start_button": "Write the axon to the output folder and start power_columns.py simnull in a separate process.",
    "cancel_button": "Stop the run and every process it started. The rows already written stay; starting again with "
                     "the same folder resumes.",
    "open_button": "Show the output folder.",
    "result_label": "The p of this axon against the simulated null. NOT calibrated (H5-E closed as not accepted, "
                    "D-41): read it as exploratory. Under an exploratory selection its first line names it.",
    "counter_label": "How many different selections of the criteria switches were run or shown on this axon in this "
                     "session (selection_exploration_log.csv). The p is NOT corrected for trying several selections.",
    "explain_label": "What the simulated null is for. The fidelity checks (in the report, and summed up under the "
                     "result) say whether the simulated axons resemble this one.",
    "small_n_label": "The p counts how many simulated axons are at least as columnar as this one, plus one, over the "
                     "accepted ones plus one: with R accepted it cannot go below 1 / (R + 1).",
    "result_warning_label": "Why this p cannot be read: too few accepted simulated axons, or simulated axons that do "
                            "not resemble this one (the fidelity checks of the report).",
    "fidelity_label": "The fidelity checks of the report (D-29c): the simulated axons against this one in clusters "
                      "per ring, localizations, rings detected and radial scatter, plus the traces of the axial leak. "
                      "A check met does not prove the null is right.",
}


def small_n_text(r: int) -> Tuple[str, str]:
    """(kind, text) of the note under the replicates spin: "bad" below 19 accepted simulated axons (the p can never
    reach 0.05), "dim" up to 98, ("", "") from 99 on."""
    r = int(r)
    floor = 1.0 / (r + 1)
    if r < MIN_REPLICATES_FOR_005:
        return "bad", (f"With {r} accepted simulated axons the smallest possible p is 1/({r}+1) = {floor:.3f}: it can "
                       f"never reach 0.05. Use 99 (p down to 0.01) unless you are only trying the dialog.")
    if r < DEFAULT_REPLICATES:
        return "dim", f"Smallest possible p: 1/({r}+1) = {floor:.3f}."
    return "", ""


def parse_progress_tag(lines: List[str]) -> Optional[Tuple[int, int, int, int]]:
    """(attempt, max attempts, accepted, requested) of the LAST progress tag in ``lines``; None without one."""
    for ln in reversed(lines):
        m = _PROGRESS_TAG.search(ln)
        if m:
            a, mx, k, n = (int(g) for g in m.groups())
            return a, mx, k, n
    return None


def write_simnull_npz(path: str, inputs: Any) -> str:
    """The selection as the NPZ ``power_columns.load_simnull_input`` reads (x, y, z, frame, lp, lpz in nm,
    n_frames). ValueError when frames or precisions are missing."""
    for what, v in (("frames", inputs.frame), ("lateral precisions", inputs.lp_lateral_nm),
                    ("axial precisions", inputs.lpz_nm)):
        if v is None:
            raise ValueError(f"the selection has no {what}: the simulated null needs them")
    frame = np.asarray(inputs.frame, dtype=np.int64).reshape(-1)
    n_frames = int(inputs.n_frames) if inputs.n_frames is not None else int(frame.max()) + 1
    np.savez(path, x=np.asarray(inputs.x_nm, float).reshape(-1), y=np.asarray(inputs.y_nm, float).reshape(-1),
             z=np.asarray(inputs.z_nm, float).reshape(-1), frame=frame,
             lp=np.asarray(inputs.lp_lateral_nm, float).reshape(-1), lpz=np.asarray(inputs.lpz_nm, float).reshape(-1),
             n_frames=np.array(n_frames))
    return path if path.lower().endswith(".npz") else path + ".npz"


class SimnullResult:
    """What a finished run gives: the observed selection key and statistic, the accepted replicates, the p (NaN when
    not computable) and a sentence saying what happened."""

    def __init__(self, *, mode: str, key: str, observed: float, n_accepted: int, n_rows: int, p: float,
                 message: str, selection_hash: str = "", selection_label: str = "", n_requested: int = 0,
                 fidelity: Optional[List[Dict[str, Any]]] = None, fidelity_error: str = "") -> None:
        self.mode, self.key, self.observed = mode, key, observed
        self.n_accepted, self.n_rows, self.p, self.message = n_accepted, n_rows, p, message
        # H6 toggles: the exploratory selection of the run ("" under the pre-specified rule)
        self.selection_hash, self.selection_label = selection_hash, selection_label
        # UI stage 0: how many accepted simulated axons were asked for, and power_columns.simnull_fidelity on the
        # accepted rows (the table of the report; [] when it could not be computed, the reason in fidelity_error)
        self.n_requested = int(n_requested)
        self.fidelity: List[Dict[str, Any]] = list(fidelity or [])
        self.fidelity_error = str(fidelity_error or "")

    @property
    def base_mode(self) -> str:
        """``mode`` without its "@sel:<hash>" suffix."""
        return self.mode.partition("@sel:")[0]

    def fidelity_counts(self) -> Dict[str, int]:
        """The fidelity entries by kind: D-29c checks ("checks") and leak traces ("leak", kind "central"), each
        with how many were met, not met, and undefined."""
        out = {"checks": 0, "checks_met": 0, "checks_not": 0, "leak": 0, "leak_met": 0, "leak_not": 0,
               "undefined": 0}
        for e in self.fidelity:
            group = "leak" if e.get("kind") == "central" else "checks"
            met = e.get("met")
            if met is None:
                out["undefined"] += 1
                continue
            out[group] += 1
            out[group + ("_met" if met else "_not")] += 1
        return out

    def fidelity_summary(self) -> str:
        """One plain line on the fidelity checks (empty when there is nothing to say)."""
        if self.fidelity_error:
            return f"Fidelity could not be checked ({self.fidelity_error})."
        if not self.fidelity:
            return ""
        c = self.fidelity_counts()
        text = (f"Fidelity: {c['checks_met']} of {c['checks']} checks met (clusters, localizations, rings, scatter); "
                f"{c['leak_met']} of {c['leak']} leak traces met")
        if c["undefined"]:
            text += f"; {c['undefined']} undefined"
        n = c["checks"] + c["leak"]
        if n and not (c["checks_not"] + c["leak_not"]):
            text += f". All {n} fidelity checks met (this does not prove the null is right)"
        return text + ". The full table is in the report."

    def warnings(self) -> List[str]:
        """Why the p cannot be read (empty: nothing found against it). Plain sentences, in order of weight."""
        out: List[str] = []
        n_req, k = int(self.n_requested), int(self.n_accepted)
        if n_req and k < n_req:
            out.append(f"Only {k} of the {n_req} simulated axons asked for were accepted.")
        if k < MIN_REPLICATES_FOR_005:
            out.append(f"With {k} accepted simulated axons the p cannot go below 1/({k}+1) = {1.0 / (k + 1):.3f}.")
        c = self.fidelity_counts()
        n_def = c["checks"] + c["leak"]
        n_not = c["checks_not"] + c["leak_not"]
        if n_not:
            out.append(f"The simulated axons do not resemble this axon ({n_not} of {n_def} fidelity checks not "
                       "met): read the p as not interpretable.")
        elif not n_def:
            out.append("Fidelity could not be checked." if not self.fidelity_error
                       else f"Fidelity could not be checked ({self.fidelity_error}).")
        return out

    @property
    def interpretable(self) -> bool:
        return not self.warnings()


def read_simnull_result(out_dir: str, name: str, n_rep: int) -> SimnullResult:
    """The p of a finished simnull run in ``out_dir`` (files named after ``name``), with power_columns' own
    ``accepted_rows`` and ``calibrated_p`` (the same p its report prints)."""
    import power_columns as pc
    paths = pc.simnull_paths(out_dir, name)

    def rows(path: str) -> List[Dict[str, str]]:
        if not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    obs_rows = rows(paths["observed"])
    if not obs_rows:
        return SimnullResult(mode="", key="", observed=float("nan"), n_accepted=0, n_rows=0, p=float("nan"),
                             message="no observed row was written (see the log)")
    obs = obs_rows[0]
    sims = rows(paths["simnull"])
    acc = pc.accepted_rows(sims, n_rep)
    o = pc.parse_float(obs.get("arcc_z_A_sel", ""))
    vals = np.array([pc.parse_float(r.get("arcc_z_A_sel", "")) for r in acc], dtype=float)
    p = float(pc.calibrated_p(vals, o)) if vals.size else float("nan")
    key = str(obs.get("zq_key", ""))
    if not math.isfinite(o):
        msg = ("the observed axon has no selected pair under rule v2 in simnull's own build: no replicate was run"
               if key.endswith("|") or not key else "the observed statistic over the selected pairs is undefined")
    elif not vals.size:
        msg = "no simulated axon was accepted (none matched the observed selection)"
    elif len(acc) < n_rep:
        msg = f"only {len(acc)} of the {n_rep} simulated axons asked for were accepted: the p uses those"
    else:
        msg = f"{len(acc)} accepted simulated axons"
    # H6 toggles (D-43): "v2-viable@sel:<hash>" under an exploratory selection; its label is in the sidecar
    mode_text = str(obs.get("zq_mode", ""))
    _base, sel_hash = pc.zq_mode_parse(mode_text)
    sel_label = ""
    if sel_hash:
        side = pc.read_selection_sidecar(os.path.join(out_dir, f"{name}{pc.SELECTION_FILE_SUFFIX}"))
        sel_label = str(side.get("label", "") or "")
        side_hash = str(side.get("hash", "") or "")
        if not side:
            msg += f"; its selection sidecar {name}{pc.SELECTION_FILE_SUFFIX} is MISSING"
        elif side_hash and side_hash != sel_hash:
            msg += f"; its selection sidecar has another hash ({side_hash})"
    # UI stage 0: the fidelity of the report's table -- the SAME function on the SAME accepted rows
    # (write_simnull_report gets accepted_rows(...) under a z-selection); never fatal
    fidelity: List[Dict[str, Any]] = []
    fidelity_error = ""
    if not acc:
        fidelity_error = "no accepted simulated axon"
    elif "arcl_z_A" not in obs:
        fidelity_error = "the rows of this run have no fidelity columns"
    else:
        try:
            fidelity = list(pc.simnull_fidelity(obs, acc))
        except Exception as exc:  # noqa: BLE001 - shown, never fatal
            fidelity_error = f"{type(exc).__name__}: {exc}"
    return SimnullResult(mode=mode_text, key=key, observed=o, n_accepted=len(acc),
                         n_rows=len(sims), p=p, message=msg, selection_hash=sel_hash, selection_label=sel_label,
                         n_requested=n_rep, fidelity=fidelity, fidelity_error=fidelity_error)


class _Relay(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object)


def _default_root() -> str:
    docs = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation) or os.path.expanduser("~")
    return os.path.join(docs, "MPS_explorer_runs", "simnull")


def _stem(source: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", os.path.splitext(os.path.basename(source))[0]).strip("_.-")
    return (s[:60] or "axon")


class SimnullDialog(QtWidgets.QDialog):
    """
    The simulated null of one axon (module docstring). ``inputs``: the
    review's ``ColumnsReviewInputs``; ``decisions``: its ``LumenDecisions``
    (manual edits are passed as ``--lumen-edits``); ``viability``: its
    ``ReviewViability`` (the dialog refuses to start without a VIABLE pair).
    """

    def __init__(self, inputs: Any, decisions: Any, viability: Any, parent: Optional[QtWidgets.QWidget] = None, *,
                 source_name: str = "", criteria: Any = None, base_viability: Any = None,
                 log_path: Optional[str] = None, axon_ids: Optional[List[str]] = None) -> None:
        super().__init__(parent)
        self.inputs = inputs
        self.decisions = decisions
        self.viability = viability
        # H6 toggles (D-43): the per-pair criteria of the review (None: the selection cannot be re-evaluated here, so
        # only the pre-specified rule can start), rule v2's own viability, the exploration log, the axon's names
        self.criteria = criteria
        self.base_viability = base_viability if base_viability is not None else (
            viability if getattr(viability, "spec", None) is None else None)
        self.log_path = str(log_path or app_log_path())
        self.axon_ids = [str(a) for a in (axon_ids or [])]
        self.run_spec: msel.SelectionSpec = msel.DEFAULT_SELECTION
        self.source_name = str(source_name or getattr(inputs, "source_name", "") or "axon")
        self.run: Optional[BackgroundRun] = None
        self.outcome: Optional[SimnullResult] = None
        self.out_dir = ""
        self.name = ""
        self.started_at = 0.0
        self._relay = _Relay()
        self._relay.finished.connect(self._on_result)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.poll)
        self.setWindowTitle(f"Simulated-null p (not calibrated) - {os.path.basename(self.source_name)}")
        # UI stage 0: as tall as the screen allows (a 1366 x 768 laptop leaves about 700 px), at most 900 px
        screen = QtWidgets.QApplication.primaryScreen()
        avail_h = screen.availableGeometry().height() if screen is not None else 760
        self.resize(880, max(560, min(900, avail_h - 60)))
        lay = QtWidgets.QVBoxLayout(self)
        n_v = len(viability.viable) if viability is not None else 0
        warn = QtWidgets.QLabel(marked("warn", "NOT CALIBRATED. The calibration of this simulated null (H5-E) was "
                                               "closed as not accepted (D-41, Q-33): the null may simulate too little "
                                               "axial leak, so the p can be too small. Read it as exploratory."))
        warn.setWordWrap(True)
        warn.setStyleSheet(f"color: {verdict('warn', dark=False)}; font-weight: bold;")
        lay.addWidget(warn)
        # UI stage 0: what the test is and its settings scroll on their own, so the buttons, the result and its
        # warnings always stay in view (on a 1366 x 728 screen the whole dialog does not fit)
        settings = QtWidgets.QWidget()
        slay = QtWidgets.QVBoxLayout(settings)
        slay.setContentsMargins(0, 0, 6, 0)
        self.settings_scroll = QtWidgets.QScrollArea()
        self.settings_scroll.setObjectName("settings_scroll")
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.settings_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.settings_scroll.setWidget(settings)
        # UI stage 0: what the test is for, in plain words, right under the warning
        self.explain_label = QtWidgets.QLabel(f"<b>What this is.</b> {SIMNULL_EXPLANATION}")
        self.explain_label.setObjectName("explain_label")
        self.explain_label.setWordWrap(True)
        slay.addWidget(self.explain_label)
        # tests replace it; the Start button asks it before a run with fewer than 19 accepted simulated axons
        self.confirm_small: Callable[[int], bool] = self._ask_small
        info = QtWidgets.QLabel(self._info_text(viability))
        info.setWordWrap(True)
        self.info_label = info
        slay.addWidget(info)
        # H6 toggles: the program's criteria switches; the run takes the selection current when it starts
        self.selection_widget = SelectionWidget(parent=self)
        if criteria is None:
            self.selection_widget.set_available(False, "this dialog was opened without the criteria of both rules")
        elif not getattr(criteria, "has_v2c", True):
            self.selection_widget.enable_v2c(False, f"rule v2c was not computed for this axon "
                                                    f"({getattr(criteria, 'v2c_error', '')})")
        self.selection_widget.state.changed.connect(self._on_selection)
        slay.addWidget(self.selection_widget)
        form = QtWidgets.QFormLayout()
        self.replicates_spin = QtWidgets.QSpinBox()
        self.replicates_spin.setObjectName("replicates_spin")
        self.replicates_spin.setRange(3, 999)
        self.replicates_spin.setValue(DEFAULT_REPLICATES)
        # UI stage 0: the spin with, under it, what that count allows (the smallest possible p)
        rep_box = QtWidgets.QWidget()
        rep_lay = QtWidgets.QVBoxLayout(rep_box)
        rep_lay.setContentsMargins(0, 0, 0, 0)
        rep_lay.setSpacing(2)
        rep_lay.addWidget(self.replicates_spin)
        self.small_n_label = QtWidgets.QLabel()
        self.small_n_label.setObjectName("small_n_label")
        self.small_n_label.setWordWrap(True)
        rep_lay.addWidget(self.small_n_label)
        form.addRow("Simulated axons (accepted):", rep_box)
        self.replicates_spin.valueChanged.connect(lambda _v: self._update_small_n())
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setObjectName("workers_spin")
        self.workers_spin.setRange(1, MAX_WORKERS)
        self.workers_spin.setValue(max(1, min(8, (os.cpu_count() or 4) - 2)))
        form.addRow("Worker processes:", self.workers_spin)
        self.n_null_spin = QtWidgets.QSpinBox()
        self.n_null_spin.setObjectName("n_null_spin")
        self.n_null_spin.setRange(19, 1999)
        self.n_null_spin.setValue(DEFAULT_N_NULL)
        form.addRow("Rotation null per analysis:", self.n_null_spin)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setObjectName("mode_combo")
        for mode, text in MODES:
            self.mode_combo.addItem(text, mode)
        if viability is None or not viability.marginal:
            item = self.mode_combo.model().item(1)  # type: ignore[attr-defined,unused-ignore]
            if item is not None:
                item.setEnabled(False)
        form.addRow("Pairs:", self.mode_combo)
        out_row = QtWidgets.QHBoxLayout()
        self.out_edit = QtWidgets.QLineEdit(os.path.join(
            _default_root(), f"{_stem(self.source_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
        self.out_edit.setObjectName("out_edit")
        out_row.addWidget(self.out_edit, stretch=1)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._on_browse)
        out_row.addWidget(browse)
        form.addRow("Output folder:", out_row)
        slay.addLayout(form)
        self.estimate_label = QtWidgets.QLabel()
        self.estimate_label.setWordWrap(True)
        slay.addWidget(self.estimate_label)
        slay.addStretch(1)
        lay.addWidget(self.settings_scroll, stretch=3)
        for w in (self.replicates_spin, self.workers_spin):
            w.valueChanged.connect(lambda _v: self._update_estimate())
        self._update_estimate()
        self._update_small_n()
        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setObjectName("start_button")
        self.start_button.clicked.connect(self._on_start)
        self.cancel_button = QtWidgets.QPushButton("Cancel run")
        self.cancel_button.setObjectName("cancel_button")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        self.open_button = QtWidgets.QPushButton("Open output folder")
        self.open_button.setObjectName("open_button")
        self.open_button.clicked.connect(self._on_open)
        self.open_button.setEnabled(False)
        for b in (self.start_button, self.cancel_button, self.open_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self._on_close)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)
        self.status_label = QtWidgets.QLabel("Not started.")
        self.status_label.setWordWrap(True)
        lay.addWidget(self.status_label)
        # UI stage 0: why the p cannot be read, ABOVE the p, and the fidelity checks in one line
        self.result_warning_label = QtWidgets.QLabel()
        self.result_warning_label.setObjectName("result_warning_label")
        self.result_warning_label.setWordWrap(True)
        self.result_warning_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        fw = self.result_warning_label.font()
        fw.setPointSize(fw.pointSize() + 2)
        fw.setBold(True)
        self.result_warning_label.setFont(fw)
        self.result_warning_label.setStyleSheet(f"color: {verdict('bad', dark=False)};")
        self.result_warning_label.setVisible(False)
        lay.addWidget(self.result_warning_label)
        self.fidelity_label = QtWidgets.QLabel()
        self.fidelity_label.setObjectName("fidelity_label")
        self.fidelity_label.setWordWrap(True)
        self.fidelity_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.fidelity_label.setVisible(False)
        lay.addWidget(self.fidelity_label)
        self.result_label = QtWidgets.QLabel(f"{SIMNULL_RESULT_LABEL}: not computed yet.")
        self.result_label.setObjectName("result_label")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        f = self.result_label.font()
        f.setPointSize(f.pointSize() + 2)
        self.result_label.setFont(f)
        lay.addWidget(self.result_label)
        # H6 toggles: next to the p, the selections tried on this axon in this session
        self.counter_label = QtWidgets.QLabel()
        self.counter_label.setObjectName("counter_label")
        self.counter_label.setWordWrap(True)
        self.counter_label.setStyleSheet(f"color: {verdict('warn', dark=False)};")
        self.counter_label.setVisible(False)
        lay.addWidget(self.counter_label)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.log_view.setMinimumHeight(70)
        self.log_view.setVisible(False)       # UI stage 0: empty until a run starts; the settings get the room
        lay.addWidget(self.log_view, stretch=2)
        self.missing_tooltips = apply_tooltips(self, SIMNULL_UI_TOOLTIPS) + list(self.selection_widget.missing_tooltips)
        self._refresh_selection()

    # ------------------------------------------------------------ the selection (H6 toggles, D-43)
    @staticmethod
    def _info_text(viability: Any) -> str:
        n_v = len(viability.viable) if viability is not None else 0
        pairs = ", ".join(f"{a}-{b}" for a, b in viability.viable) if n_v else "none"
        marg = ", ".join(f"{a}-{b}" for a, b in viability.marginal) if viability is not None else ""
        spec = getattr(viability, "spec", None) if viability is not None else None
        who = (f"Exploratory selection <tt>{spec.label}</tt> #{spec.hash}" if spec is not None and not spec.is_default
               else "Rule v2")
        return (f"{who} pairs here: VIABLE {pairs}" + (f"; MARGINAL {marg}" if marg else "") + ".<br>"
                "power_columns.py simnull measures this axon (2-4 min), then simulates axons WITHOUT columns and keeps "
                "those whose selection (the same rule and criteria) equals this axon's; the p is the arc test on the "
                "centroid membrane over the selected pairs. Manual lumen edits apply to this axon only; the simulated "
                "axons get the automatic rule, isolation only (no widefield masks).")

    def _on_selection(self, _spec: Any = None) -> None:
        try:
            self._refresh_selection()
        except RuntimeError:        # the dialog is being destroyed
            return

    def _refresh_selection(self) -> None:
        """The pairs of the current selection: the info, the Pairs combo and whether Start is possible."""
        spec = self.selection_widget.spec
        crit, base = self.criteria, self.base_viability
        if crit is not None and base is not None:
            res = msel.evaluate_selection(crit, spec)
            self.viability = base if spec.is_default else msel.selection_viability(base, res)
            self.selection_widget.set_summary("this axon: " + res.summary())
        v = self.viability
        self.info_label.setText(f"Axon: <b>{os.path.basename(self.source_name)}</b>. " + self._info_text(v))
        item = self.mode_combo.model().item(1)  # type: ignore[attr-defined,unused-ignore]
        if item is not None:
            item.setEnabled(bool(v is not None and v.marginal))
        if (v is None or not v.marginal) and self.mode_combo.currentIndex() == 1:
            self.mode_combo.setCurrentIndex(0)
        running = self.run is not None and self.run.poll() is None
        if crit is None and spec.exploratory:
            self.start_button.setEnabled(False)
            self.status_label.setText(marked("bad", "An exploratory selection needs the criteria of the review; this "
                                                    "dialog only runs the pre-specified rule v2."))
        elif v is None or not v.viable:
            self.start_button.setEnabled(False)
            who = "rule v2" if spec.is_default else f"the exploratory selection {spec.label}"
            self.status_label.setText(marked("bad", f"No VIABLE pair under {who}: nothing to compare."))
        else:
            self.start_button.setEnabled(not running)
            if not running and self.outcome is None and "No VIABLE pair" in self.status_label.text():
                self.status_label.setText("Not started.")

    # ------------------------------------------------------------ the run
    @property
    def mode(self) -> str:
        return str(self.mode_combo.currentData())

    def _update_estimate(self) -> None:
        r, w = self.replicates_spin.value(), self.workers_spin.value()
        lo = 2 + r * 7.5 / w / 60
        hi = 4 + r * 11 * 1.5 / w / 60      # some simulated axons are rejected and drawn again
        m = MAX_ATTEMPTS_PER_REPLICATE * r  # power_columns' default --zq-max-attempts
        worst = 4 + m * 11 / w / 60
        self.estimate_label.setText(f"Rough time: {lo:.0f}-{hi:.0f} min; up to {worst:.0f} min if most simulated axons "
                                    f"are rejected (at most {m} attempts). Measurement 2-4 min, then about 8-11 s per "
                                    "simulated axon per worker.")

    def _update_small_n(self) -> None:
        kind, text = small_n_text(self.replicates_spin.value())
        self.small_n_label.setText(marked(kind, text) if kind else "")
        if kind == "bad":
            self.small_n_label.setStyleSheet(f"color: {verdict('bad', dark=False)}; font-weight: bold;")
        else:
            self.small_n_label.setStyleSheet(f"color: {verdict('dim', dark=False)};")
        self.small_n_label.setVisible(bool(kind))

    def _ask_small(self, r: int) -> bool:
        ans = QtWidgets.QMessageBox.question(
            self, "Simulated null",
            f"Only {r} accepted simulated axons: the p can never go below 1/({r}+1) = {1.0 / (r + 1):.3f}, so it "
            "cannot be read as evidence. Run anyway (for trying the dialog)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        return bool(ans == QtWidgets.QMessageBox.Yes)

    def _on_browse(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder of the simulated null", _default_root())
        if d:
            self.out_edit.setText(d)

    def _on_start(self) -> None:
        """The Start button: asks first when fewer than 19 accepted simulated axons are asked for (``start()``
        itself never asks: scripts and tests call it)."""
        r = int(self.replicates_spin.value())
        if r < MIN_REPLICATES_FOR_005 and not self.confirm_small(r):
            self.status_label.setText(f"Not started: {r} accepted simulated axons is too few for a readable p.")
            return
        self.start()

    def _on_open(self) -> None:
        if os.path.isdir(self.out_dir):
            open_folder(self.out_dir)

    def command(self, npz: str, edits: Optional[str]) -> List[str]:
        args = [python_executable(), "-u", os.path.join(REPO_ROOT, "power_columns.py"), "simnull", "--from-arrays", npz,
                "--out", self.out_dir, "--replicates", str(self.replicates_spin.value()), "--workers",
                str(self.workers_spin.value()), "--n-null", str(self.n_null_spin.value()), "--table-version", "v5",
                "--zq-select", self.mode, "--lumen-clean"]
        if edits:
            args += ["--lumen-edits", edits]
        # H6 toggles (D-43): the SAME selection for the observed axon and every simulated one; every run is logged
        spec = self.selection_widget.spec
        if spec.exploratory:
            args += ["--zq-selection", spec.label]
        args += ["--exploration-log", self.log_path]
        return args

    def start(self) -> bool:
        """Write the inputs and start the run; False (with the reason on screen) when it cannot start."""
        if self.run is not None and self.run.poll() is None:
            return False
        if self.viability is None or not self.viability.viable:
            return False
        spec = self.selection_widget.spec
        if self.criteria is None and spec.exploratory:
            return False
        out = os.path.abspath(self.out_edit.text().strip())
        if not self.out_edit.text().strip() or is_inside(out, REPO_ROOT):
            self.status_label.setText(marked("bad", "Choose an output folder OUTSIDE the program's folder."))
            return False
        try:
            os.makedirs(out, exist_ok=True)
            self.out_dir = out
            self.name = _stem(self.source_name)
            npz = write_simnull_npz(os.path.join(out, f"{self.name}.npz"), self.inputs)
            edits = None
            if self.decisions is not None and int(getattr(self.decisions, "n_manual", 0)) > 0:
                edits = os.path.join(out, f"{self.name}_lumen_edits.csv")
                rows = self.decisions.to_rows()
                with open(edits, "w", encoding="utf-8", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            self.run = BackgroundRun(self.command(npz, edits), os.path.join(out, "simnull_gui.log"))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.status_label.setText(marked("bad", f"Could not start: {type(exc).__name__}: {exc}"))
            return False
        self.outcome = None
        self.run_spec = spec
        self.counter_label.setVisible(False)
        self.result_warning_label.setVisible(False)
        self.fidelity_label.setVisible(False)
        self.progress_bar.setFormat("%p%")
        self.started_at = time.perf_counter()
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.log_view.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Started: measuring the axon...")
        self.result_label.setText(f"{SIMNULL_RESULT_LABEL}: running...")
        self.timer.start()
        return True

    def accepted_so_far(self) -> int:
        path = os.path.join(self.out_dir, f"{self.name}_simnull.csv")
        if not os.path.isfile(path):
            return 0
        try:
            with open(path, encoding="utf-8", newline="") as fh:
                return sum(1 for r in csv.DictReader(fh) if str(r.get("zq_accepted", "")).strip() in ("1", "1.0", "True"))
        except (OSError, csv.Error):
            return 0

    def poll(self) -> None:
        if self.run is None:
            return
        lines = tail(self.run.log_path, 200)
        self.log_view.setPlainText("\n".join(lines))
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())
        n_rep = self.replicates_spin.value()
        acc = self.accepted_so_far()
        tried = sum(1 for ln in lines if _REPLICATE_LINE.match(ln))
        el = time.perf_counter() - self.started_at
        # UI stage 0: the runner's own count over the whole run (every round of redraws); the CSV as a fallback
        tag = parse_progress_tag(lines)
        if tag is not None:
            attempt, max_attempts, acc_tag, n_tag = tag
            acc = max(acc_tag, min(acc, n_tag))
            n_rep = n_tag
            progress = (f"attempt {attempt} of up to {max_attempts}, {min(acc, n_rep)} of {n_rep} accepted")
        else:
            progress = (f"{acc} of {n_rep} accepted"
                        + (f" ({tried} finished in the last lines of the log)" if tried else ""))
        self.progress_bar.setValue(int(round(100 * min(1.0, acc / max(1, n_rep)))))
        self.progress_bar.setFormat(f"{min(acc, n_rep)} of {n_rep} accepted")
        phase = ("measuring the axon" if not any(("configuration measured" in ln or "configuration:" in ln)
                                                 for ln in lines) else "simulated axons")
        self.status_label.setText(f"Running ({el / 60:.1f} min): {phase}; {progress}.")
        code = self.run.poll()
        if code is None:
            if self.run.cancelled:
                self.status_label.setText(marked("warn", "Cancelling: stopping the run and its workers..."))
            return
        self.timer.stop()
        self.cancel_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.result_warning_label.setVisible(False)
        self.fidelity_label.setVisible(False)
        if self.run.cancelled:
            self.status_label.setText(marked("warn", "Cancelled. Start again with the same output folder to resume."))
            self.result_label.setText(f"{SIMNULL_RESULT_LABEL}: cancelled.")
            return
        if code != 0:
            # the folder may hold rows of an earlier run (another mode, another selection): never show their p
            self.status_label.setText(marked("bad", f"The run failed (exit {code}, {el / 60:.1f} min): see the "
                                                     "log below and simnull_gui.log in the output folder."))
            self.result_label.setText(marked("bad", f"{SIMNULL_RESULT_LABEL}: failed (exit {code})."))
            return
        self.status_label.setText(f"Finished (exit {code}, {el / 60:.1f} min); reading the result...")
        out, name = self.out_dir, self.name
        relay = self._relay

        def body() -> None:
            try:
                res = read_simnull_result(out, name, n_rep)
            except Exception as exc:  # noqa: BLE001 - shown
                relay.finished.emit(None, exc)
                return
            relay.finished.emit(res, None)

        threading.Thread(target=body, daemon=True, name="simnull-result").start()

    def _on_result(self, res: Any, error: Any) -> None:
        if error is not None or res is None:
            self.result_warning_label.setVisible(False)
            self.fidelity_label.setVisible(False)
            self.result_label.setText(marked("bad", f"{SIMNULL_RESULT_LABEL}: could not be read ({error}); see the log."))
            return
        self.outcome = res
        p = f"{res.p:.4f}" if math.isfinite(res.p) else "not computable"
        obs = f"{res.observed:+.2f}" if math.isfinite(res.observed) else "n/a"
        # H6 toggles (D-43): the first line names an exploratory selection (from the run's own rows and sidecar)
        banner = ""
        if res.selection_hash:
            spec = self.run_spec
            label = res.selection_label or (spec.label if spec.hash == res.selection_hash else "?")
            banner = (f"<span style='color:{verdict('bad', dark=False)}'><b>EXPLORATORY SELECTION {label} "
                      f"#{res.selection_hash}</b>: not the pre-specified rule; p values are not corrected for trying "
                      "several selections.</span><br>")
        # UI stage 0: first why the p cannot be read, then the fidelity checks, then the p itself
        warnings = res.warnings() if hasattr(res, "warnings") else []
        if warnings:
            self.result_warning_label.setText(
                marked("bad", "NOT INTERPRETABLE") + "<br>" + "<br>".join(f"&bull; {w}" for w in warnings))
        self.result_warning_label.setVisible(bool(warnings))
        fid = res.fidelity_summary() if hasattr(res, "fidelity_summary") else ""
        self.fidelity_label.setText(fid)
        self.fidelity_label.setVisible(bool(fid))
        n_acc = f" ({res.n_accepted} accepted)"
        self.result_label.setText(
            f"{banner}<b>{SIMNULL_RESULT_LABEL}: p = {p}{n_acc}</b><br>observed z_A over the selected pairs {obs}; "
            f"selection {res.key or 'n/a'} ({res.mode}); {res.message}. NOT calibrated: exploratory only (D-41).")
        self.status_label.setText(f"Done: report {self.name}_simnull.md in {self.out_dir}.")
        try:
            n = variants_tried(msel.ExplorationLog(self.log_path), self.axon_ids + [self.name])
            self.counter_label.setText(counter_line(max(n, 1)) + f" (log: {self.log_path})")
        except Exception as exc:  # noqa: BLE001 - shown; never fatal
            self.counter_label.setText(marked("bad", f"The exploration log could not be read ({exc}): the selections "
                                                     "tried are NOT counted; p values are not corrected for trying "
                                                     "several selections."))
        self.counter_label.setVisible(True)

    def wait(self, timeout: float = 3600.0) -> None:
        """Block until the run ends and its result is shown (scripts and tests)."""
        t_end = time.perf_counter() + float(timeout)
        while time.perf_counter() < t_end:
            QtWidgets.QApplication.processEvents()
            if self.run is None or (self.run.poll() is not None and not self.timer.isActive()
                                    and (self.outcome is not None or self.run.cancelled
                                         or "could not" in self.result_label.text()
                                         or "failed (exit" in self.result_label.text())):
                break
            time.sleep(0.05)

    def cancel(self) -> None:
        if self.run is not None and self.run.poll() is None:
            self.run.cancel()
            self.poll()

    def _on_close(self) -> None:
        self.close()

    def closeEvent(self, event: Any) -> None:
        if self.run is not None and self.run.poll() is None:
            ans = QtWidgets.QMessageBox.question(
                self, "Simulated null", "The simulated null is still running. Stop it? (Rows already written stay; "
                "starting again with the same folder resumes.)")
            if ans != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.run.cancel()
        self.timer.stop()
        super().closeEvent(event)

