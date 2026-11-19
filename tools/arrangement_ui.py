# -*- coding: utf-8 -*-
"""
The analysis window's "Working scale" panel, its "Perimeter arrangement" tab and the "Show synthetic axon" view
(the clustering integration, 2026-10-22; plan A9 sec. 4, regularity PR 3).

* :class:`WorkingScalePanel` -- the distances below which this file's pipeline cannot tell two objects apart
  (``tools.mps_working_scale``: eps, min samples, NeNA, fusion, d50, slab), each with its origin; the editable
  c_label (14 [lit: Dempsey et al. 2011, MEA, per fluorophore on glass; provisional]); the DBSCAN rule of this file beside 25/10 (D2), and the
  double table "25/10 vs rule" once 'Compare methods...' ran them.
* :class:`ArrangementPanel` -- "Null (per axon)": the per-axon pipeline null of every method compared (DBSCAN 25/10
  always, the method shown, and the methods of the last comparison), each re-running ITS method, in a worker thread
  with the time and a Cancel button; B = 199 by default, 999 on demand; slow methods refused unless ticked. Per
  method: H1 along the arc and H2 in the band against the null (observed, null median, the GLOBAL (MAD) envelope that
  decides and the pointwise envelope only as a shade; r*, h_max; NOT INTERPRETABLE above everything when it is), and
  the regularity reading (gap histogram, g(s), S(m) with the null's pipeline-level bands, the merged-cluster mark,
  the edge note). A summary table puts the methods side by side.
* :class:`SyntheticAxonDialog` -- the observed axon and one saved replicate of its null side by side, same contour,
  scale and colours; drawn only when asked (A9 sec. 4.3), with "Export image...".

Nothing here computes: ``tools.clustering_integration`` does, in the worker.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import contextlib
import math
import re
import time
import traceback
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from tools import clustering_integration as ci
from tools.method_costs import method_label, words_seconds
from tools.mps_plot_style import neutral, rgba, role, set_title, style_dark, verdict

# B: 199 by default, 999 on demand [user decision 2026-10-22, A9 / P-R78].
B_CHOICES: Tuple[int, ...] = (199, 999)
WORKING_SCALE_TITLE = "Working scale"
# The names 'Export image...' lists these plots by.
PLOT_H1 = "H1(r) along the arc (null per axon)"
PLOT_H2 = "H2(r) in the band (null per axon)"
PLOT_GAPS = "Gaps between consecutive centres"
PLOT_G = "g(s) along the arc"
PLOT_S = "S(m) periodogram of the centres"
TAB_TITLE = "Perimeter arrangement"
NULL_BUTTON = "Null (per axon)"
SYNTH_BUTTON = "Show synthetic axon"
C5_NOTE = ("Power per axon is low with 35-50 clusters: conclude on the arrangement from the batch, per animal (the "
           "z of each measure), not from one axon's p (open question C5).")
NULL_TIP = ("Simulate this axon without structure: molecules at random on its own membrane, with this file's labels, "
            "blinking cycles and precision, analysed exactly like it -- the SAME clustering method, parameters, "
            "curation and contour -- B times. Every method compared gets its own null. The p values compare each "
            "readout with what the pipeline alone produces; NOT INTERPRETABLE when the simulated axons do not "
            "resemble this one. Methods: docs/null_and_scale.md, docs/regularity.md.")
SYNTH_TIP = ("The observed axon beside one simulated axon of its null (the replicate kept), with the same contour, "
             "scale and colours. Drawn only when you press this; nothing of it stays on the axon map.")
# Manual test (2026-11-19): H1 / H2 stayed empty axes until the null ran, and nothing said so; the Randomization
# box (KS test) of the results window does not compute it.
NULL_NEEDED = ("Empty until you press 'Null (per axon)' above; the Randomization box (KS test) does not compute "
               "this null.")
GLOBAL_TIP = ("Decision: the global envelope (maximum studentised deviation over r >= the working scale's d50) and "
              "its p. The grey shade is the pointwise 2.5-97.5 % envelope, for the eye only: crossing it at some r "
              "is not a test. " + NULL_NEEDED)


# C3 / C4 (user decision 2026-10-23): the null's closure in the panel; N_loc stays the default.
CLOSURE_ITEMS: Tuple[Tuple[str, str], ...] = (("N_loc (default)", "n_locs"), ("K", "k"),
                                              ("both side by side (x2 time)", "both"))
CLOSURE_TIP = ("What each simulated axon is matched to: N_loc = as many localizations in the slab as this axon "
               "(default, user decision A9 / C4); K = as many clusters as this axon found (its pilots run the whole "
               "pipeline: slower). 'both' runs the two nulls with the same seed -- about twice the time, shown in "
               "the estimate -- and 'Compare closures...' puts their p values and envelopes side by side. Methods: "
               "docs/null_and_scale.md.")
# C2 (user decision 2026-10-23): the membrane of every method's null.
MEMBRANE_ITEMS: Tuple[Tuple[str, str], ...] = (("DBSCAN 25/10 centroids (default)", "dbscan_centroids"),
                                               ("localizations", "localizations"))
MEMBRANE_TIP = ("Where the null puts its molecules: by default on the membrane through the centroids of DBSCAN "
                "25/10, the SAME curve for every method (decision C2); 'localizations' fits it to the slab's "
                "localizations instead, to compare where the centroids deform what the localizations show. The line "
                "below says how far apart the two curves are (a warning per 600 nm stretch beyond 50 nm [ad hoc: "
                "the H2 band's half-width]). tools/null_membrane.py")


def _fmt(v: Any, spec: str = ".3g") -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "-" if not math.isfinite(f) else format(f, spec)


# =====================================================================================================================
# Working scale
# =====================================================================================================================

class WorkingScalePanel(QtWidgets.QGroupBox):
    """The working scale of the analysis shown, the editable c_label and the DBSCAN rule beside 25/10."""

    c_label_changed = QtCore.pyqtSignal(float)
    radii_changed = QtCore.pyqtSignal()
    lpc_depth_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(WORKING_SCALE_TITLE, parent)
        self.setObjectName("working_scale_panel")
        self.setCheckable(True)
        self.setChecked(True)
        self.setToolTip("The distances below which this file's pipeline cannot tell two objects apart, each with "
                        "its origin [literature / measured / own simulation / ad hoc]. Untick to fold it. "
                        "Methods: docs/null_and_scale.md section 2, docs/dbscan_rule.md.")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        self.body = QtWidgets.QWidget()
        outer.addWidget(self.body)
        lay = QtWidgets.QVBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setObjectName("working_scale_table")
        self.table.setHorizontalHeaderLabels(["Mark", "Value", "Origin"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # at least 170 px [ad hoc: the marks' rows at 1366 x 728], and more when the window grows (manual test of
        # 2026-10-23: "compressed in a window that could grow"); the tab scrolls
        self.table.setMinimumHeight(170)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        lay.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("c_label (cycles per label):"))
        self.spin_c_label = QtWidgets.QDoubleSpinBox()
        self.spin_c_label.setObjectName("spin_c_label")
        self.spin_c_label.setRange(1.0, 200.0)
        self.spin_c_label.setDecimals(1)
        from tools.dbscan_rule import C_LABEL_DEFAULT
        self.spin_c_label.setValue(float(C_LABEL_DEFAULT))
        self.spin_c_label.setToolTip(
            "Switching cycles per label: 14 for Alexa Fluor 647 in MEA (Dempsey et al. 2011, Table 1; provisional, "
            "not measured in this laboratory's buffer), 26 in beta-ME as the sensitivity column. It sets the DBSCAN "
            "rule's min samples, the 'probable fusion' mark, the footprint and the null's labels.")
        row.addWidget(self.spin_c_label)
        from tools.mps_origin_ui import OriginBadge
        self.badge_c_label = OriginBadge("dbscan_rule.c_label", float(C_LABEL_DEFAULT))
        row.addWidget(self.badge_c_label)
        row.addStretch(1)
        lay.addLayout(row)
        # B1 (user decision 2026-10-23): the two event-linking radii, editable to explore (equal or different); any
        # value but the default is EXPLORATORY here and in the export (radii_exploratory, lpc_rule_radius_nena,
        # link_radius_nena). Not saved: every session and the Batch start at the defaults.
        radii = QtWidgets.QHBoxLayout()
        radii.addWidget(QtWidgets.QLabel("Linking radius [NeNA]: rule's lpc"))
        self.spin_lpc_radius = QtWidgets.QDoubleSpinBox()
        self.spin_lpc_radius.setObjectName("spin_lpc_radius")
        self.spin_lpc_radius.setRange(0.5, 10.0)
        self.spin_lpc_radius.setDecimals(3)
        self.spin_lpc_radius.setSingleStep(0.25)
        self.spin_lpc_radius.setValue(float(ci.LPC_RULE_RADIUS_IN_NENA))
        self.spin_lpc_radius.setToolTip(
            "Radius of the event linking that measures the localizations per cycle the DBSCAN rule uses, in NeNA. "
            "Default 3 sqrt(2) = 4.243 [own simulation, docs/dbscan_rule.md sec. 4: 3 NeNA under-reads it by "
            "9-26 %, 3 sqrt(2) NeNA by 0.4-3.4 %]. Any other value is EXPLORATORY (decision B1).")
        radii.addWidget(self.spin_lpc_radius)
        radii.addWidget(QtWidgets.QLabel("events / fusion"))
        self.spin_events_radius = QtWidgets.QDoubleSpinBox()
        self.spin_events_radius.setObjectName("spin_events_radius")
        self.spin_events_radius.setRange(0.5, 10.0)
        self.spin_events_radius.setDecimals(3)
        self.spin_events_radius.setSingleStep(0.25)
        self.spin_events_radius.setValue(float(ci.EVENTS_RADIUS_IN_NENA))
        self.spin_events_radius.setToolTip(
            "Radius of the event linking of the events layer, the 'probable fusion' mark and the footprint, in NeNA. "
            "Default 3 [decision D5; own simulation, docs/dbscan_rule.md]. Any other value is EXPLORATORY "
            "(decision B1). It reaches 'Compare methods...' and the export of this axon; the Batch keeps 3.")
        radii.addWidget(self.spin_events_radius)
        self.btn_radii_default = QtWidgets.QPushButton("Defaults")
        self.btn_radii_default.setObjectName("btn_radii_default")
        self.btn_radii_default.setToolTip("Both radii back to 3 sqrt(2) and 3 NeNA (no EXPLORATORY mark).")
        self.btn_radii_default.clicked.connect(self.reset_radii)
        radii.addWidget(self.btn_radii_default)
        self.lbl_radii = QtWidgets.QLabel("")
        self.lbl_radii.setObjectName("lbl_radii_exploratory")
        self.lbl_radii.setWordWrap(True)
        radii.addWidget(self.lbl_radii, 1)
        lay.addLayout(radii)
        self.spin_lpc_radius.valueChanged.connect(self._on_radii)
        self.spin_events_radius.valueChanged.connect(self._on_radii)
        # B3 (c): the localizations per cycle by depth (tools.lpc_depth), computed when asked (it links the whole
        # file and bootstraps 999 times: seconds on a large file)
        depth = QtWidgets.QHBoxLayout()
        self.btn_lpc_depth = QtWidgets.QPushButton("Localizations per cycle by depth")
        self.btn_lpc_depth.setObjectName("btn_lpc_depth")
        self.btn_lpc_depth.setToolTip(
            "Do the localizations per event change with z? The whole file's events, the slab's and z bands as wide "
            "as the slab, each with its bootstrap 95 % interval (999 draws [ad hoc]), and a warning when the slab "
            "differs from the rest of the file beyond chance. A diagnostic: the DBSCAN rule keeps the whole file's "
            "value (B3). The export carries the lpc_ columns once computed. Methods: tools/lpc_depth.py.")
        self.btn_lpc_depth.clicked.connect(lambda: self.lpc_depth_requested.emit())
        depth.addWidget(self.btn_lpc_depth)
        self.lbl_lpc_depth = QtWidgets.QLabel("Not computed (press the button; the Batch computes it per file).")
        self.lbl_lpc_depth.setObjectName("lbl_lpc_depth")
        self.lbl_lpc_depth.setWordWrap(True)
        self.lbl_lpc_depth.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        depth.addWidget(self.lbl_lpc_depth, 1)
        lay.addLayout(depth)
        self.plot_lpc = pg.PlotWidget()
        self.plot_lpc.setObjectName("plot_lpc_depth")
        style_dark(self.plot_lpc)
        set_title(self.plot_lpc, "Localizations per event against z (slab band filled)")
        self.plot_lpc.setLabels(bottom="z band centre [nm]", left="localizations per event")
        self.plot_lpc.setMinimumHeight(150)
        self.plot_lpc.setVisible(False)
        lay.addWidget(self.plot_lpc)
        self.lbl_rule = QtWidgets.QLabel("")
        self.lbl_rule.setObjectName("lbl_dbscan_rule")
        self.lbl_rule.setWordWrap(True)
        self.lbl_rule.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_rule)
        self.double = QtWidgets.QTableWidget(0, 7)
        self.double.setObjectName("double_table")
        self.double.setHorizontalHeaderLabels(["Run", "eps [nm]", "min samples", "K/um", "median 1NN [nm]",
                                               "occupancy [%]", "probable fusions (c14)"])
        self.double.setToolTip("Decision D2: DBSCAN 25/10 (the reference, Gazal et al. 2026) and the DBSCAN rule of "
                               "this file, side by side. Filled by 'Compare methods...' with 'DBSCAN rule' ticked.")
        self.double.verticalHeader().setVisible(False)
        self.double.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.double.setMinimumHeight(110)
        self.double.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        self.double.setVisible(False)
        lay.addWidget(self.double)
        self.toggled.connect(self.body.setVisible)
        self.spin_c_label.valueChanged.connect(self._on_c_label)
        self.rule: Any = None
        self.rule26: Any = None

    def c_label(self) -> float:
        return float(self.spin_c_label.value())

    def lpc_radius(self) -> Optional[float]:
        """The rule's lpc radius in NeNA, None at the default (3 sqrt(2))."""
        v = float(self.spin_lpc_radius.value())
        return None if abs(v - ci.LPC_RULE_RADIUS_IN_NENA) < 5e-4 else v

    def events_radius(self) -> Optional[float]:
        """The events / fusion radius in NeNA, None at the default (3)."""
        v = float(self.spin_events_radius.value())
        return None if abs(v - ci.EVENTS_RADIUS_IN_NENA) < 5e-4 else v

    def radii_words(self) -> str:
        return ci.radii_exploratory(self.lpc_radius(), self.events_radius())

    def reset_radii(self) -> None:
        self.spin_lpc_radius.setValue(float(ci.LPC_RULE_RADIUS_IN_NENA))
        self.spin_events_radius.setValue(float(ci.EVENTS_RADIUS_IN_NENA))

    def _on_radii(self, _v: float = 0.0) -> None:
        words = self.radii_words()
        if words == "no":
            self.lbl_radii.setText("defaults (pre-specified)")
            self.lbl_radii.setStyleSheet("")
        else:
            self.lbl_radii.setText(words)
            self.lbl_radii.setStyleSheet(f"color: {verdict('warn', dark=False)}; font-weight: bold;")
        self.radii_changed.emit()

    def show_lpc_depth(self, profile: Any, why: str = "") -> None:
        """B3 (c): the file's lpc, the slab's with its interval, the difference and its warning, and the profile
        against z (error bars = bootstrap 95 % intervals; the slab's band filled)."""
        self.plot_lpc.clear()
        if profile is None:
            self.lbl_lpc_depth.setText(why or "not computed")
            self.lbl_lpc_depth.setStyleSheet("")
            self.plot_lpc.setVisible(False)
            return

        def f(v: Any) -> str:
            return _fmt(v, ".3f")

        text = (f"lpc of the whole file {f(profile.lpc_file)} ({profile.n_file:,} events); slab {f(profile.lpc_slab)} "
                f"[{f(profile.slab_ci[0])}, {f(profile.slab_ci[1])}] ({profile.n_slab:,} events); slab - rest "
                f"{_fmt(profile.diff, '+.3f')} [{_fmt(profile.diff_ci[0], '+.3f')}, {_fmt(profile.diff_ci[1], '+.3f')}]. "
                f"Events: {profile.source}.")
        if profile.differs:
            text = ("WARNING: the slab's localizations per event differ from the rest of the file beyond chance "
                    "(the rule keeps the whole file's value, B3). " + text)
            self.lbl_lpc_depth.setStyleSheet(f"color: {verdict('warn', dark=False)};")
        else:
            self.lbl_lpc_depth.setStyleSheet("")
        self.lbl_lpc_depth.setText(text)
        xs, ys, lo, hi, slab = [], [], [], [], []
        for b in profile.bands:
            if b.mean is None:
                continue
            xs.append(0.5 * (b.z_lo_nm + b.z_hi_nm))
            ys.append(float(b.mean))
            lo.append(float(b.mean) - float(b.ci_low))
            hi.append(float(b.ci_high) - float(b.mean))
            slab.append(bool(b.is_slab))
        if xs:
            x, y = np.asarray(xs), np.asarray(ys)
            self.plot_lpc.addItem(pg.ErrorBarItem(x=x, y=y, top=np.asarray(hi), bottom=np.asarray(lo), beam=5.0,
                                                  pen=pg.mkPen(neutral(True), width=1)))
            brushes = [pg.mkBrush(role("observed")) if s else pg.mkBrush(0, 0, 0, 0) for s in slab]
            self.plot_lpc.plot(x, y, pen=pg.mkPen(role("observed"), width=1), symbol="o", symbolSize=8,
                               symbolBrush=brushes, symbolPen=pg.mkPen(role("observed")))
            if profile.lpc_file is not None:
                self.plot_lpc.addItem(pg.InfiniteLine(pos=float(profile.lpc_file), angle=0,
                                                      pen=pg.mkPen(neutral(True), style=QtCore.Qt.PenStyle.DashLine)))
        self.plot_lpc.setVisible(True)

    def _on_c_label(self, value: float) -> None:
        self.badge_c_label.set_value(float(value))
        self.c_label_changed.emit(float(value))

    def show_scale(self, analysis: Any, nena_nm: Optional[float], nena_words: str,
                   locs_per_cycle: Optional[Any] = None) -> None:
        """The marks of ``analysis`` (None: empty) with this file's NeNA, and the rule with the current c_label."""
        from tools.dbscan_rule import C_LABEL_BME
        self.table.setRowCount(0)
        self.rule = self.rule26 = None
        if analysis is None:
            self.lbl_rule.setText("No analysis yet.")
            return
        from tools.mps_working_scale import working_scale_of_analysis
        ws = working_scale_of_analysis(analysis, nena_nm=nena_nm)
        for m in ws.marks():
            r = self.table.rowCount()
            self.table.insertRow(r)
            unit = f" {m.unit}" if m.unit and m.value is not None else ""
            origin = m.origin + (f" ({nena_words})" if m.key == "nena_nm" and m.origin == "measured" and nena_words
                                 else "")
            for c, text in enumerate((m.label, _fmt(m.value, ".4g") + unit, origin)):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(m.why)
                self.table.setItem(r, c, item)
        for note in ws.notes:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem("note"))
            note_item = QtWidgets.QTableWidgetItem(note)
            note_item.setToolTip(note)
            self.table.setItem(r, 1, note_item)
        self.table.resizeColumnsToContents()
        self.rule = ci.rule_for(nena_nm, c_label=self.c_label(), locs_per_cycle=locs_per_cycle)
        self.rule26 = ci.rule_for(nena_nm, c_label=C_LABEL_BME, locs_per_cycle=locs_per_cycle)
        # the parenthesis shows the OTHER reported c_label (manual test of 2026-10-23: with 26 it repeated 26)
        from tools.dbscan_rule import C_LABEL_MEA
        other_c = C_LABEL_MEA if math.isclose(self.c_label(), C_LABEL_BME) else C_LABEL_BME
        other_rule = ci.rule_for(nena_nm, c_label=other_c, locs_per_cycle=locs_per_cycle)
        mode = ci.dbscan_mode(analysis, self.rule)
        if self.rule is None:
            self.lbl_rule.setText(f"DBSCAN parameters: {mode}. The rule of docs/dbscan_rule.md needs this file's "
                                  "NeNA (none measured): 25/10 stays the default (decision D2).")
            return
        lpc = self.rule.inputs["loc_per_cycle"]
        self._on_radii_label()
        self.lbl_rule.setText(
            f"DBSCAN parameters: {mode}. The rule for this file (shown, never applied: decision D2): eps "
            f"{self.rule.eps_nm:.1f} nm, min samples {self.rule.min_samples} with c_label {self.c_label():g} "
            f"({other_rule.min_samples if other_rule is not None else '-'} with {other_c:g}); localizations per "
            f"cycle "
            f"{float(lpc.value):.2f} {lpc.origin}. Tick 'DBSCAN rule' in 'Compare methods...' to see both.")
        self.lbl_rule.setToolTip(self.rule.describe())

    def _on_radii_label(self) -> None:
        words = self.radii_words()
        self.lbl_radii.setText("defaults (pre-specified)" if words == "no" else words)

    def show_double_table(self, comparison: Optional[ci.ComparisonResult]) -> None:
        """D2's double table from the last comparison (hidden when it has no rule run)."""
        rows = [] if comparison is None else comparison.double_table()
        self.double.setRowCount(0)
        self.double.setVisible(len(rows) > 1)
        for d in rows:
            r = self.double.rowCount()
            self.double.insertRow(r)
            run = comparison.run(str(d.get("method_key"))) if comparison is not None else None
            params = {} if run is None else run.params
            cells = (d.get("label"), params.get("eps"), params.get("min_samples"), d.get("clusters_per_um"),
                     d.get("median_1nn_nm"), d.get("occupancy_percent"), d.get("events_n_fusion_c14"))
            for c, v in enumerate(cells):
                self.double.setItem(r, c, QtWidgets.QTableWidgetItem(v if isinstance(v, str) else _fmt(v)))
        self.double.resizeColumnsToContents()


# =====================================================================================================================
# The null, in a worker
# =====================================================================================================================

# Every null worker that started and was not joined yet (as tools.cluster_method_ui._LIVE_WORKERS: a QThread
# destroyed while it runs aborts the program).
_LIVE: set = set()


class NullWorker(QtCore.QThread):
    """Runs ``tools.clustering_integration.run_nulls`` on the given runs; touches no widget."""

    progress = QtCore.pyqtSignal(str, int, int)
    finished_outcomes = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, runs: Sequence[Any], n_replicates: int, nena_nm: Optional[float], c_label: float,
                 allow_slow: bool, seed: int = 0, closure: str = ci.NULL_CLOSURE, membrane: str = ci.NULL_MEMBRANE,
                 membrane_curve: Optional[Any] = None) -> None:
        super().__init__()
        self.closure = str(closure)
        self.membrane = str(membrane)
        self.membrane_curve = membrane_curve
        self.runs = list(runs)
        self.n_replicates = int(n_replicates)
        self.nena_nm = nena_nm
        self.c_label = float(c_label)
        self.allow_slow = bool(allow_slow)
        self.seed = int(seed)
        self.cancelled = False

    def run(self) -> None:
        try:
            out = ci.run_nulls(self.runs, n_replicates=self.n_replicates, nena_nm=self.nena_nm, seed=self.seed,
                               c_label=self.c_label, allow_slow=self.allow_slow,
                               cancelled=lambda: self.cancelled,
                               progress=lambda k, d, t: self.progress.emit(str(k), int(d), int(t)),
                               closure=self.closure, membrane=self.membrane, membrane_curve=self.membrane_curve)
            self.finished_outcomes.emit(out)
        except BaseException:                              # noqa: BLE001 - reported, never a crash
            self.failed.emit(traceback.format_exc())


def _release(worker: QtCore.QThread) -> None:
    worker.wait()
    _LIVE.discard(worker)


# =====================================================================================================================
# The tab
# =====================================================================================================================

SUMMARY_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("method", "Method"), ("status", "Null"), ("B", "B"), ("interpretable", "Interpretable"),
    ("h1_p", "H1 global p"), ("h1_rstar", "H1 r* [nm]"), ("h1_hmax", "H1 h_max [nm]"), ("h2_p", "H2 global p"),
    ("h2_rstar", "H2 r* [nm]"), ("h2_hmax", "H2 h_max [nm]"), ("p_k", "p K/um"), ("p_nn", "p 1NN"),
    ("p_reg", "p r"), ("reading", "Regularity reading"), ("elongated", "Merged-looking (elongated)"),
    ("seconds", "time"), ("null_closure", "Closure"), ("null_membrane", "Membrane"),
)


def excluded_words(run: Any) -> str:
    """Why a method of the comparison has no null: 'no result in the comparison (<status>: <why>) -- not
    included'."""
    status = str(getattr(run, "status", "") or "no result")
    why = str(getattr(run, "error", "") or "").strip().splitlines()
    detail = f"{status}: {why[-1]}" if why else status
    return f"no result in the comparison ({detail}) -- not included"


def summary_row(key: str, o: ci.NullOutcome) -> Dict[str, Any]:
    """One method's row of the summary table (plain values)."""
    row: Dict[str, Any] = {"method": key, "status": o.status, "seconds": words_seconds(o.seconds) if o.ok else "",
                           "null_closure": getattr(o, "closure", ""), "null_membrane": getattr(o, "membrane", "")}
    if not o.ok:
        row["reading"] = o.message
        return row
    r = o.result
    row.update(B=r.n_accepted, interpretable=r.interpretable,
               h1_p=None if r.h1 is None else r.h1.p_global, h1_rstar=None if r.h1 is None else r.h1.r_star,
               h1_hmax=None if r.h1 is None else r.h1.h_max, h2_p=r.h2.p_global, h2_rstar=r.h2.r_star,
               h2_hmax=r.h2.h_max, p_k=r.p_values.get("k_per_um"), p_nn=r.p_values.get("nn_median"),
               p_reg=r.p_values.get("regularity"))
    if o.gap is not None:
        row["reading"] = o.gap.reading.level
        el = o.gap.elongation
        row["elongated"] = None if el is None else el.n_elongated
    return row


# The words of the plots' legends (manual test of 2026-10-23: "every element explained, in simple words").
LEG_OBSERVED = "This axon"
LEG_MEDIAN = "Null median (random, same pipeline)"
LEG_POINTWISE = "Pointwise 95 % envelope (shade, guide only)"
LEG_GLOBAL = "Global envelope (decides)"
LEG_FUSION = "Fusion distance"
LEG_D50 = "d50"
LEG_RSTAR = "r* (largest departure)"
LEG_BAND = "Null 95 % band (shade, guide only)"
# G2 (user decision 2026-11-07): which null a regularity band comes from, said on screen and in the exported image.
BAND_PIPELINE = "pipeline null, B = {n}"
BAND_CENTRES = "approx., centres"
READING_TIP = ("Descriptive only: a simulated full lattice reads 'lattice' in 4 of 12 axons, 'periodic' in 8. "
               "Group type: Batch z. docs/regularity.md")
CENTRES_READING_TIP = ("K centres at random with hard core 2 x median r_eff, until the pipeline null runs; then the "
                       "bands come from it (G2).")


def band_label(source: str) -> str:
    """The legend's band entry with its source ('pipeline null, B = ...' or 'approx., centres')."""
    return f"Null 95 % band ({source}; shade, guide only)" if source else LEG_BAND
H_HELP = ("Above the null = more neighbours than random (clustering); below = fewer (spacing / regularity). Only "
          "the global envelope decides; its p is in the title. Left of the grey zone's edge nothing is tested. "
          + NULL_NEEDED)
HINT_WIDTH = 46                                          # characters per line of the 'No null yet' notice


def null_hint_words(b: Optional[int], estimate: Optional[str]) -> str:
    """The notice centred on H1 / H2 while no null of the method shown is drawn."""
    if b is None:
        return f"No null yet: run the MPS analysis, then press '{NULL_BUTTON}' above."
    return f"No null yet: press '{NULL_BUTTON}' above (B = {b}" + (f", {estimate}" if estimate else "") + ")."


def wrap_words(text: str, width: int = HINT_WIDTH) -> str:
    """``text`` in lines of at most ``width`` characters (a longer word keeps its own line)."""
    lines: List[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return "\n".join(lines)
REGULARITY_HELP = ("Gaps: distances between consecutive centres along the ring. g(s): neighbours at arc distance s "
                   "(1 = random). S(m): periodogram of the centres (a peak = a regular spacing). The curves are for "
                   "the eye; the reading above is the test.")


def _legend_line(colour: Any, width: float, style: Any = None) -> Any:
    return pg.PlotDataItem(pen=pg.mkPen(colour, width=width, style=style or QtCore.Qt.PenStyle.SolidLine))


def _legend_bar(brush: Any) -> Any:
    return pg.BarGraphItem(x=[0.0], height=[1.0], width=1.0, brush=brush, pen=None)


def h_legend_entries(res: Any, env: Any, dark: bool = True) -> List[Tuple[str, Any]]:
    """(label, sample) of what an H1 / H2 plot draws, in drawing order (the on-screen legend and Export image)."""
    line = neutral(dark)
    out: List[Tuple[str, Any]] = [
        (LEG_OBSERVED, _legend_line(role("observed"), 2.0)),
        (LEG_MEDIAN, _legend_line(line, 1.0, QtCore.Qt.PenStyle.DashLine)),
        (LEG_POINTWISE, _legend_bar(pg.mkBrush(rgba("randomized", 110)))),
        (LEG_GLOBAL, _legend_line(role("summary"), 1.5, QtCore.Qt.PenStyle.DotLine)),
    ]
    r_min = float(getattr(res, "global_r_min_nm", float("nan")))
    if math.isfinite(r_min):
        d50 = getattr(res.scale, "d50_nm", None)
        word = "d50" if d50 is not None and math.isclose(r_min, float(d50)) else "the fusion distance"
        out.append((f"Not testable: below {word}", _legend_bar(pg.mkBrush(rgba("slab", 90)))))
    out.append((LEG_FUSION, _legend_line(line, 1.0, QtCore.Qt.PenStyle.DotLine)))
    out.append((LEG_D50, _legend_line(line, 1.0)))
    if env is not None and env.r_star is not None and math.isfinite(float(env.r_star)):
        out.append((LEG_RSTAR, _legend_line(role("summary"), 1.0, QtCore.Qt.PenStyle.DashLine)))
    return out


def regularity_legend_entries(dark: bool = True, source: str = "") -> List[Tuple[str, Any]]:
    """(label, sample) of what a regularity plot draws; ``source`` names the null the band comes from (G2)."""
    return [(LEG_OBSERVED, _legend_line(role("observed"), 2.0)),
            (LEG_MEDIAN, _legend_line(neutral(dark), 1.0, QtCore.Qt.PenStyle.DashLine)),
            (band_label(source), _legend_bar(pg.mkBrush(rgba("randomized", 110))))]


# More rows than this and the corner legend takes two columns (half as tall over a short plot) [ad hoc].
CORNER_LEGEND_ONE_COLUMN = 5


# Axis text and padding of the panel's plots (manual test: small type, empty margins) [ad hoc].
PLOT_TICK_PT = 9
PLOT_LABEL_PT = "10pt"


def tidy_plot(plot: Any) -> None:
    """Bigger axis type and tighter margins for one PlotWidget: nothing drawn, no number changes. Call after
    ``setLabels``."""
    item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
    font = QtGui.QFont()
    font.setPointSize(PLOT_TICK_PT)
    for side in ("left", "bottom"):
        axis = item.getAxis(side)
        axis.setStyle(tickFont=font)
        if getattr(axis, "labelText", ""):
            axis.setLabel(axis.labelText, axis.labelUnits, **{"font-size": PLOT_LABEL_PT})
    try:
        item.layout.setContentsMargins(2, 2, 6, 2)
        item.getViewBox().setDefaultPadding(0.02)
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


# Share of the plot the legend box is taken to cover when the freest corner is chosen [ad hoc].
LEGEND_BOX_WIDTH_SHARE = 0.45
LEGEND_ROW_HEIGHT_SHARE = 0.075
_LEGEND_CORNERS: Tuple[Tuple[str, Tuple[int, int]], ...] = (
    ("top-right", (-4, 4)), ("top-left", (4, 4)), ("bottom-right", (-4, -4)), ("bottom-left", (4, -4)))


def free_legend_corner(plot: Any, n_rows: int) -> Tuple[int, int]:
    """The offset of the plot's corner where a legend of ``n_rows`` rows covers the fewest drawn points (the top
    right on a tie, where it always was). The data are read off the plot's own curves, so nothing is drawn."""
    xs: List[Any] = []
    ys: List[Any] = []
    try:
        items = list(plot.getPlotItem().listDataItems())
    except Exception:  # noqa: BLE001 - no curves to avoid: the default corner
        items = []
    for it in items:
        try:
            x, y = it.getData()
        except Exception:  # noqa: BLE001
            continue
        if x is None or y is None:
            continue
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        k = np.isfinite(x) & np.isfinite(y)
        xs.append(x[k])
        ys.append(y[k])
    if not xs:
        return _LEGEND_CORNERS[0][1]
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if x.size == 0 or not (np.ptp(x) > 0 and np.ptp(y) > 0):
        return _LEGEND_CORNERS[0][1]
    u = (x - x.min()) / np.ptp(x)
    v = (y - y.min()) / np.ptp(y)
    w = LEGEND_BOX_WIDTH_SHARE
    h = min(0.9, 0.04 + LEGEND_ROW_HEIGHT_SHARE * max(int(n_rows), 1))
    best, best_n = _LEGEND_CORNERS[0][1], None
    for _name, off in _LEGEND_CORNERS:
        in_x = (u >= 1.0 - w) if off[0] < 0 else (u <= w)
        in_y = (v >= 1.0 - h) if off[1] > 0 else (v <= h)
        n = int(np.count_nonzero(in_x & in_y))
        if best_n is None or n < best_n:
            best, best_n = off, n
    return best


def corner_legend(plot: Any, entries: Sequence[Tuple[str, Any]], dark: bool = True) -> Any:
    """A compact legend in the plot's freest corner (the top right unless the curves run there: it never sits on
    them) on a semi-transparent box (the data stay readable under it), in the theme's colours. Returns the
    LegendItem."""
    text = "#ffffff" if dark else "#000000"
    fill = (0, 0, 0, 150) if dark else (255, 255, 255, 190)
    rows = len(entries)
    n_cols = 2 if rows > CORNER_LEGEND_ONE_COLUMN else 1
    leg = pg.LegendItem(offset=free_legend_corner(plot, (rows + n_cols - 1) // n_cols), labelTextColor=text,
                        brush=pg.mkBrush(*fill), pen=pg.mkPen(text, width=0.5),
                        labelTextSize="9pt", verSpacing=-6, colCount=n_cols)
    leg.setParentItem(plot.getPlotItem().vb)
    for label, sample in entries:
        leg.addItem(sample, label)
    leg.setZValue(1e6)
    return leg


# Where the tab's divider (controls and tables | plots) is kept (tools.mps_ui_layout), and the most of the tab the
# upper pane takes before the user moves it [ad hoc: the plots keep at least half].
ARRANGEMENT_SPLIT_KEY = "mps_window.arrangement_split"
ARRANGEMENT_UPPER_MAX_SHARE = 0.5
# How far the global envelope may reach beyond the observed / median / pointwise span before it stops setting the
# view [ad hoc: half the span on each side]; beyond it the curve leaves the plot and the title counts it.
ENVELOPE_VIEW_SLACK = 0.5


def envelope_y_range(main: Sequence[Any], extra: Sequence[Any] = (),
                     slack: float = ENVELOPE_VIEW_SLACK) -> Tuple[Optional[Tuple[float, float]], int]:
    """(y_min, y_max) of the finite values of the ``main`` series, widened to the ``extra`` series' values that lie
    within ``slack`` x that span of it; and how many finite values of all the series fall outside the range. None
    when nothing is finite."""
    vals = [np.asarray(v, dtype=float) for v in main if v is not None]
    finite = np.concatenate([v[np.isfinite(v)] for v in vals]) if vals else np.zeros(0)
    if finite.size == 0:
        return None, 0
    lo, hi = float(finite.min()), float(finite.max())
    span = max(hi - lo, 1.0)
    allowed = (lo - slack * span, hi + slack * span)
    for v in extra:
        if v is None:
            continue
        e = np.asarray(v, dtype=float)
        e = e[np.isfinite(e) & (e >= allowed[0]) & (e <= allowed[1])]
        if e.size:
            lo, hi = min(lo, float(e.min())), max(hi, float(e.max()))
    off = 0
    for v in list(main) + list(extra):
        if v is None:
            continue
        a = np.asarray(v, dtype=float)
        off += int(np.sum(np.isfinite(a) & ((a < lo) | (a > hi))))
    return (lo, hi), off


class ArrangementPanel(QtWidgets.QWidget):
    """The "Perimeter arrangement" tab (see the module docstring)."""

    nulls_changed = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, dark: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("arrangement_panel")
        self.dark = dark
        self.analysis: Any = None
        self.runs: List[Any] = []                      # MethodRun: the shown analysis + the last comparison's
        self.outcomes: Dict[str, ci.NullOutcome] = {}
        self._saved_as_notes: List[str] = []
        self.nena_nm: Optional[float] = None
        self.c_label: float = 14.0
        self._worker: Optional[NullWorker] = None
        self._t0 = 0.0
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        top = QtWidgets.QHBoxLayout()
        self.btn_null = QtWidgets.QPushButton(NULL_BUTTON)
        self.btn_null.setObjectName("btn_null_per_axon")
        self.btn_null.setToolTip(NULL_TIP)
        self.btn_null.clicked.connect(self._on_null_clicked)
        top.addWidget(self.btn_null)
        top.addWidget(QtWidgets.QLabel("B:"))
        self.combo_b = QtWidgets.QComboBox()
        self.combo_b.setObjectName("combo_null_b")
        for b in B_CHOICES:
            self.combo_b.addItem(str(b), b)
        self.combo_b.setToolTip("Simulated axons per method: 199 by default (p down to 0.005), 999 on demand (five "
                                "times the time) [user decision 2026-10-22].")
        self.combo_b.currentIndexChanged.connect(lambda _i: self._update_estimate())
        top.addWidget(self.combo_b)
        self.chk_slow = QtWidgets.QCheckBox("Allow slow methods")
        self.chk_slow.setObjectName("chk_null_slow")
        self.chk_slow.setToolTip("Emitters, OPTICS and voronoi_csr take minutes to hours per null: refused unless "
                                 "this is ticked (open question A2), with the estimate shown here first.")
        self.chk_slow.toggled.connect(lambda _on: self._update_estimate())
        top.addWidget(self.chk_slow)
        # C3 / C4 (user decision 2026-10-23): which closure the null is matched to; both side by side doubles the time
        top.addWidget(QtWidgets.QLabel("Closure:"))
        self.combo_closure = QtWidgets.QComboBox()
        self.combo_closure.setObjectName("combo_null_closure")
        for text, value in CLOSURE_ITEMS:
            self.combo_closure.addItem(text, value)
        self.combo_closure.setToolTip(CLOSURE_TIP)
        self.combo_closure.currentIndexChanged.connect(lambda _i: self._update_estimate())
        top.addWidget(self.combo_closure)
        # C2 (user decision 2026-10-23): the membrane of every method's null
        top.addWidget(QtWidgets.QLabel("Membrane:"))
        self.combo_membrane = QtWidgets.QComboBox()
        self.combo_membrane.setObjectName("combo_null_membrane")
        for text, value in MEMBRANE_ITEMS:
            self.combo_membrane.addItem(text, value)
        self.combo_membrane.setToolTip(MEMBRANE_TIP)
        self.combo_membrane.currentIndexChanged.connect(lambda _i: self._update_membrane())
        top.addWidget(self.combo_membrane)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setObjectName("btn_null_cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Stop at the next simulated axon; the methods already done are kept.")
        self.btn_cancel.clicked.connect(self.cancel)
        top.addWidget(self.btn_cancel)
        self.btn_synth = QtWidgets.QPushButton(SYNTH_BUTTON)
        self.btn_synth.setObjectName("btn_show_synthetic_axon")
        self.btn_synth.setToolTip(SYNTH_TIP)
        self.btn_synth.setEnabled(False)
        self.btn_synth.clicked.connect(self._on_synth_clicked)
        top.addWidget(self.btn_synth)
        self.btn_export = QtWidgets.QPushButton("Save curves...")
        self.btn_export.setObjectName("btn_null_export")
        self.btn_export.setToolTip("Write the H1 / H2 curves (observed, null median, pointwise envelope) and the "
                                   "summary table of the methods to CSV files (never over an existing file).")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export_clicked)
        top.addWidget(self.btn_export)
        self.chk_legend = QtWidgets.QCheckBox("Legends on the plots")
        self.chk_legend.setObjectName("chk_arrangement_legends")
        self.chk_legend.setChecked(True)
        self.chk_legend.setToolTip("Show or hide the legend box in a corner of each plot (Export image... has its "
                                   "own 'legend' box).")
        self.chk_legend.toggled.connect(self.set_legends_visible)
        top.addWidget(self.chk_legend)
        self.btn_compare_closures = QtWidgets.QPushButton("Compare closures...")
        self.btn_compare_closures.setObjectName("btn_compare_closures")
        self.btn_compare_closures.setToolTip("C4: the p values and the envelopes of the N_loc and the K closure side "
                                             "by side, per method. Run the null with 'Closure: both' first.")
        self.btn_compare_closures.setEnabled(False)
        self.btn_compare_closures.clicked.connect(self._on_compare_closures)
        top.addWidget(self.btn_compare_closures)
        top.addStretch(1)
        lay.addLayout(top)
        self.lbl_membrane = QtWidgets.QLabel("")
        self.lbl_membrane.setObjectName("lbl_membrane_deformation")
        self.lbl_membrane.setWordWrap(True)
        self.lbl_membrane.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_membrane)
        self._membrane: Any = None                     # tools.null_membrane.NullMembrane of the analysis shown
        self._membrane_key: Any = None
        self.band_source = ""
        self._legends: Dict[str, Any] = {}
        self._shown_res: Any = None
        self._drawn_key: Optional[str] = None
        self._base_titles: Dict[Any, str] = {}           # plot -> its title before any null is drawn
        self._hints: Dict[Any, Any] = {}                 # plot -> the 'No null yet' notice centred on H1 / H2
        self._hint_fixed: Optional[str] = None           # the notice's text when it is not 'No null yet'
        self.excluded: List[Tuple[str, str]] = []
        self.lbl_estimate = QtWidgets.QLabel("")
        self.lbl_estimate.setObjectName("lbl_null_estimate")
        self.lbl_estimate.setWordWrap(True)
        lay.addWidget(self.lbl_estimate)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.banner = QtWidgets.QLabel("")
        self.banner.setObjectName("lbl_not_interpretable")
        self.banner.setWordWrap(True)
        self.banner.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.banner)
        self.table = QtWidgets.QTableWidget(0, len(SUMMARY_COLUMNS))
        self.table.setObjectName("null_summary_table")
        self.table.setHorizontalHeaderLabels([c[1] for c in SUMMARY_COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setMinimumHeight(120)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        self.table.setToolTip("One row per method: its own null (the same method re-run on B simulated axons). "
                              "Select a row to draw it. " + C5_NOTE)
        self.table.itemSelectionChanged.connect(self._on_select)
        lay.addWidget(self.table)
        self.plots = QtWidgets.QTabWidget()
        self.plots.setObjectName("arrangement_plots")
        h = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(h)
        hl.setContentsMargins(0, 0, 0, 0)
        self.plot_h1 = pg.PlotWidget()
        self.plot_h2 = pg.PlotWidget()
        for p, title in ((self.plot_h1, "H1(r) along the arc"), (self.plot_h2, "H2(r) in the band (+/-50 nm)")):
            style_dark(p)
            set_title(p, title)
            p.setLabels(bottom="r [nm]", left="H(r) [nm]")
            tidy_plot(p)
            p.setToolTip(GLOBAL_TIP)
            hl.addWidget(p)
        hw = QtWidgets.QWidget()
        hv = QtWidgets.QVBoxLayout(hw)
        hv.setContentsMargins(0, 0, 0, 0)
        hv.addWidget(h, 1)
        self.lbl_h_help = QtWidgets.QLabel(H_HELP)
        self.lbl_h_help.setObjectName("lbl_h_help")
        self.lbl_h_help.setWordWrap(True)
        hv.addWidget(self.lbl_h_help)
        self.plots.addTab(hw, "H(r): arc and band")
        g = QtWidgets.QWidget()
        gl = QtWidgets.QVBoxLayout(g)
        gl.setContentsMargins(0, 0, 0, 0)
        self.lbl_reading = QtWidgets.QLabel("")
        self.lbl_reading.setObjectName("lbl_regularity_reading")
        self.lbl_reading.setWordWrap(True)
        self.lbl_reading.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        gl.addWidget(self.lbl_reading)
        row = QtWidgets.QHBoxLayout()
        self.plot_gaps = pg.PlotWidget()
        self.plot_g = pg.PlotWidget()
        self.plot_s = pg.PlotWidget()
        for p, title, bottom in ((self.plot_gaps, "Gaps between consecutive centres", "gap [nm]"),
                                 (self.plot_g, "g(s) along the arc", "s [nm]"),
                                 (self.plot_s, "S(m): periodogram of the centres", "mode m")):
            style_dark(p)
            set_title(p, title)
            p.setLabels(bottom=bottom)
            tidy_plot(p)
            p.setToolTip("Observed (line) against the pipeline null's pointwise 2.5-97.5 % band (shade) and median "
                         "(dashed): for the eye only; the reading above is the test (docs/regularity.md).")
            row.addWidget(p)
        gl.addLayout(row, 1)
        self.lbl_regularity_help = QtWidgets.QLabel(REGULARITY_HELP)
        self.lbl_regularity_help.setObjectName("lbl_regularity_help")
        self.lbl_regularity_help.setWordWrap(True)
        gl.addWidget(self.lbl_regularity_help)
        self.plots.addTab(g, "Regularity")
        for p in (self.plot_h1, self.plot_h2, self.plot_gaps, self.plot_g, self.plot_s):
            p.setMinimumHeight(180)
            self._base_titles[p] = re.sub(r"<[^>]+>", "", str(p.getPlotItem().titleLabel.text or ""))
        # Manual test of 2026-10-23: the controls, the Working scale panel (set_header) and the nulls' table sat in
        # one column with the plots, which took every spare pixel and left the panel a sliver with a scroll bar.
        # They are now the upper pane of a divider (kept in mps_ui_layout.json): at their natural height, scrolling
        # inside it when the tab is short; the plots get the rest.
        from tools.mps_ui_layout import PersistentSplitter, style_splitter
        self.upper = QtWidgets.QWidget()
        self.upper.setObjectName("arrangement_upper")
        up = QtWidgets.QVBoxLayout(self.upper)
        up.setContentsMargins(0, 0, 0, 0)
        while lay.count():
            item = lay.takeAt(0)
            if item is None:
                break
            if item.widget() is not None:
                up.addWidget(item.widget())
            elif item.layout() is not None:
                sub = item.layout()
                sub.setParent(None)
                up.addLayout(sub)
        self.upper_scroll = QtWidgets.QScrollArea()
        self.upper_scroll.setObjectName("arrangement_upper_scroll")
        self.upper_scroll.setWidgetResizable(True)
        self.upper_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.upper_scroll.setWidget(self.upper)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter.setObjectName("arrangement_splitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.upper_scroll)
        self.splitter.addWidget(self.plots)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        style_splitter(self.splitter)
        self._split_keeper = PersistentSplitter(self.splitter, ARRANGEMENT_SPLIT_KEY)
        self._split_user = False
        self.splitter.splitterMoved.connect(lambda *_a: setattr(self, "_split_user", True))
        self._split_placed = False
        lay.addWidget(self.splitter, 1)
        self._update_estimate()
        self._draw(None)

    def set_header(self, widget: QtWidgets.QWidget) -> None:
        """A panel in the upper pane (the window's Working scale), below the null's controls, estimate, progress and
        banner and above the nulls' table: the 'Null (per axon)' row stays in sight when the upper pane is short
        (manual test of 2026-11-19: at the top, the panel pushed that row out of view)."""
        lay = self.upper.layout()
        assert isinstance(lay, QtWidgets.QVBoxLayout)
        idx = lay.indexOf(self.banner)
        lay.insertWidget(idx + 1 if idx >= 0 else 0, widget)

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        super().showEvent(event)
        self._update_membrane(True)
        if not self._split_placed:
            self._split_placed = True
            QtCore.QTimer.singleShot(0, self.place_split)

    def place_split(self) -> None:
        """The divider where the user left it, else the upper pane at its natural height up to
        ARRANGEMENT_UPPER_MAX_SHARE of the tab."""
        if self._split_user or self._split_keeper.restore():
            self._split_user = True
            return
        total = sum(self.splitter.sizes())
        if total <= 0:
            return
        upper = min(self.upper.sizeHint().height() + 4, int(total * ARRANGEMENT_UPPER_MAX_SHARE))
        self.splitter.setSizes([upper, max(total - upper, 1)])

    def _on_null_clicked(self) -> None:
        self.start()

    def _on_synth_clicked(self) -> None:
        self.show_synthetic()

    def _on_export_clicked(self) -> None:
        out = self.export_curves()
        if out:
            self.banner.setText("Written: " + ", ".join(out.values())
                                + ("\n" + "\n".join(self._saved_as_notes) if self._saved_as_notes else ""))

    def figure_plots(self) -> Dict[str, Any]:
        """The plots 'Export image...' of the analysis window offers (F6: title, legend, white, never overwritten)."""
        return {PLOT_H1: self.plot_h1, PLOT_H2: self.plot_h2, PLOT_GAPS: self.plot_gaps, PLOT_G: self.plot_g,
                PLOT_S: self.plot_s}

    def show_plot(self, name: str) -> None:
        """Bring the sub-tab of plot ``name`` to the front (pyqtgraph exports a laid-out item)."""
        self.plots.setCurrentIndex(0 if name in (PLOT_H1, PLOT_H2) else 1)

    def set_dark(self, dark: bool) -> None:
        """Restyle and redraw for a black screen or a white figure."""
        from tools.mps_plot_style import style_light
        self.dark = bool(dark)
        for p in self.figure_plots().values():
            (style_dark if self.dark else style_light)(p)
        # Manual test of 2026-10-23 (Export image of H1 on white: axes only, no curve, the title in the screen's
        # pale grey): the redraw followed the table's selection, and with none (or a row with no result) the white
        # figure kept no curve and the dark title. It redraws the null last drawn; the plots' own titles follow
        # the background in every case.
        key = self._drawn_key if self._drawn_key in self.outcomes else self.selected_key()
        self._draw(key)

    # -- state ---------------------------------------------------------------------------------------------------
    def set_context(self, analysis: Any, runs: Sequence[Any], nena_nm: Optional[float], c_label: float) -> None:
        """The analysis shown and the runs a null can be run for (the shown analysis first, then the comparison's
        runs on the same slab). The nulls of another analysis are dropped."""
        if analysis is not self.analysis:
            self.outcomes = {}
        self.analysis = analysis
        self.nena_nm = nena_nm
        self.c_label = float(c_label)
        seen: Dict[str, Any] = {}
        self.excluded = []                                  # (key, why): runs with no result, named, not run
        for r in runs:
            if r is None or r.key in seen or any(k == r.key for k, _w in self.excluded):
                continue
            if getattr(r, "ok", False):
                seen[r.key] = r
            else:
                self.excluded.append((r.key, excluded_words(r)))
        self.runs = list(seen.values())
        self.btn_null.setEnabled(analysis is not None and self._worker is None)
        self._update_membrane()
        self._update_estimate()
        self._fill_table()

    # -- C2: the null's membrane and the deformation indicator ---------------------------------------------------
    def membrane_choice(self) -> str:
        return str(self.combo_membrane.currentData() or ci.NULL_MEMBRANE)

    def closure_choice(self) -> str:
        return str(self.combo_closure.currentData() or ci.NULL_CLOSURE)

    def _update_membrane(self, compute: Optional[bool] = None) -> None:
        """The DBSCAN 25/10 centroid membrane and the localization membrane of the analysis shown, once per
        analysis, and the deformation between them (a warning per stretch). Computed when the tab is shown, the null
        starts or the export asks (``compute``; default: when visible), not on every analysis."""
        from tools import null_membrane as nmb
        a = self.analysis
        if a is None:
            self._membrane = self._membrane_key = None
            self.lbl_membrane.setText("")
            return
        if compute is None:
            compute = self.isVisible()
        if self._membrane_key is not a and not compute:
            self.lbl_membrane.setText("Membrane of the null: " + nmb.MEMBRANE_WORDS[self.membrane_choice()] + ".")
            self.lbl_membrane.setStyleSheet("")
            return
        if self._membrane_key is not a:
            try:
                self._membrane = nmb.null_membrane_for(a, self.runs, nmb.MEMBRANE_DBSCAN, with_deformation=True)
            except Exception as exc:  # noqa: BLE001 - the indicator says it could not be computed
                self._membrane = nmb.NullMembrane(nmb.MEMBRANE_DBSCAN, None, "", None, f"{type(exc).__name__}: {exc}")
            self._membrane_key = a
        m = self._membrane
        d = None if m is None else m.deformation
        if d is None:
            self.lbl_membrane.setText("Membrane of the null: " + nmb.MEMBRANE_WORDS[self.membrane_choice()] +
                                      ". Deformation indicator not computed" + (f" ({m.note})." if m and m.note else "."))
            self.lbl_membrane.setStyleSheet("")
            return
        self.lbl_membrane.setText("Membrane of the null: " + nmb.MEMBRANE_WORDS[self.membrane_choice()] + ". "
                                  + d.sentence())
        level = d.level()
        self.lbl_membrane.setStyleSheet("" if level == "ok" else f"color: {verdict('warn', dark=False)};"
                                        + (" font-weight: bold;" if level == "warn" else ""))

    def deformation(self) -> Optional[Any]:
        """The deformation indicator of the analysis shown (for the export), or None."""
        if self.analysis is not None and self._membrane_key is not self.analysis:
            self._update_membrane(True)
        m = self._membrane if self._membrane_key is self.analysis else None
        return None if m is None else m.deformation

    def _membrane_curve(self) -> Optional[Any]:
        """The curve every method's null uses under the chosen membrane (None: each run's own reading)."""
        from tools import null_membrane as nmb
        if self.analysis is None:
            return None
        self._update_membrane(True)
        choice = self.membrane_choice()
        m = self._membrane
        if choice == nmb.MEMBRANE_DBSCAN:
            return None if m is None else m.curve_nm
        try:
            return nmb.null_membrane_for(self.analysis, self.runs, choice, with_deformation=False).curve_nm
        except Exception:  # noqa: BLE001 - each null falls back to its own reading, said in its row
            return None

    def _estimate_parts(self) -> Tuple[int, float, List[str], List[str]]:
        """B, the estimated seconds of the methods that would run, their keys and the slow ones refused."""
        b = int(self.combo_b.currentData() or B_CHOICES[0])
        parts: List[str] = []
        total = 0.0
        refused: List[str] = []
        closure = self.closure_choice()
        for r in self.runs:
            s = ci.null_estimate_seconds(r, b, closure)
            if ci.is_slow(r.method) and not self.chk_slow.isChecked():
                refused.append(f"{method_label(r.method)} (about {words_seconds(s)})")
                continue
            total += s
            parts.append(r.key)
        return b, total, parts, refused

    def null_hint_text(self) -> str:
        """The 'No null yet' notice of H1 / H2, with B and the time the null would take."""
        if not self.runs:
            return null_hint_words(None, None)
        b, total, parts, _refused = self._estimate_parts()
        return null_hint_words(b, ci.words_estimate(total) if parts else None)

    def _update_estimate(self) -> None:
        self._refresh_hints()
        if not self.runs:
            self.lbl_estimate.setText("Run the MPS analysis first: the null simulates the axon shown.")
            return
        closure = self.closure_choice()
        b, total, parts, refused = self._estimate_parts()
        both = " (closure: both side by side, about twice the time)" if closure == "both" else f" (closure: {closure})"
        text = (f"{NULL_BUTTON} for {len(parts)} method(s) ({', '.join(parts)}), B = {b}{both}: "
                f"{ci.words_estimate(total)} (a first estimate from the per-method cost table; a lower bound for "
                "the slow methods).")
        if refused:
            text += " Refused as slow unless 'Allow slow methods' is ticked: " + ", ".join(refused) + "."
        if self.excluded:
            text += " " + "; ".join(f"{key}: {why}" for key, why in self.excluded) + "."
        self.lbl_estimate.setText(text + " " + C5_NOTE)

    def running(self) -> bool:
        return self._worker is not None

    # -- run -----------------------------------------------------------------------------------------------------
    def start(self) -> bool:
        if self._worker is not None or not self.runs:
            return False
        allow = self.chk_slow.isChecked()
        runs = [r for r in self.runs if allow or not ci.is_slow(r.method)]
        for r in self.runs:
            if r not in runs:
                self.outcomes[r.key] = ci.NullOutcome(key=r.key, method=r.method, params=dict(r.params),
                                                      status="refused", message=ci.mm.slow_refusal(r.method, 200),
                                                      estimate_s=ci.null_estimate_seconds(r, 199))
        if not runs:
            self._fill_table()
            return False
        b = int(self.combo_b.currentData() or B_CHOICES[0])
        closure = self.closure_choice()
        # A2 (b): a slow method's null asks once more, with its estimate, before it runs
        from tools import slow_confirm
        slow = [(r.method, ci.null_estimate_seconds(r, b, closure)) for r in runs if ci.is_slow(r.method)]
        if slow and not slow_confirm.confirm_slow(self, NULL_BUTTON, slow, note=f"B = {b}, closure: {closure}."):
            self.banner.setText("Not run: the slow method(s) were not confirmed.")
            return False
        w = NullWorker(runs, b, self.nena_nm, self.c_label, allow, closure=closure, membrane=self.membrane_choice(),
                       membrane_curve=self._membrane_curve())
        w.progress.connect(self._on_progress)
        w.finished_outcomes.connect(self._on_outcomes)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_finished)
        w.finished.connect(lambda w=w: QtCore.QTimer.singleShot(0, lambda: _release(w)))
        _LIVE.add(w)
        self._worker = w
        self._t0 = time.perf_counter()
        self.progress.setRange(0, b * len(runs) * len(ci.null_closures(closure)))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.btn_null.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._done_keys: List[str] = []
        self.banner.setText(f"Running the null of {len(runs)} method(s), B = {b}...")
        w.start()
        return True

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancelled = True
            self.btn_cancel.setEnabled(False)
            self.banner.setText("Cancelling at the next simulated axon; the methods already done are kept.")

    def wait(self, timeout_ms: int = 600000) -> None:
        """Block until the worker ends and its signals were delivered (scripts and tests)."""
        w = self._worker
        if w is not None:
            w.wait(timeout_ms)
        for _ in range(5):
            QtWidgets.QApplication.processEvents()

    def _on_progress(self, key: str, done: int, total: int) -> None:
        if key not in self._done_keys:
            self._done_keys.append(key)
        base = (len(self._done_keys) - 1) * int(total)
        self.progress.setValue(min(self.progress.maximum(), base + int(done)))
        el = time.perf_counter() - self._t0
        self.progress.setFormat(f"{key}: {done} of {total}  ({words_seconds(el)} elapsed)")

    def _on_outcomes(self, out: Dict[str, ci.NullOutcome]) -> None:
        self.outcomes.update(out)

    def _on_failed(self, tb: str) -> None:
        self.banner.setText("The null failed: " + (tb.strip().splitlines()[-1] if tb.strip() else "unknown"))
        self.banner.setToolTip(tb)

    def _on_finished(self) -> None:
        self._worker = None
        self.progress.setVisible(False)
        self.btn_null.setEnabled(self.analysis is not None)
        self.btn_cancel.setEnabled(False)
        self._fill_table()
        self.nulls_changed.emit()

    # -- display -------------------------------------------------------------------------------------------------
    def outcome_for(self, analysis: Any) -> Optional[ci.NullOutcome]:
        """The null of ``analysis`` itself (the shown analysis' run), when it ran."""
        for o in self.outcomes.values():
            if o.ok and o.result is not None and analysis is not None and getattr(self.runs[0], "analysis", None) \
                    is analysis and o.key == self.runs[0].key:
                return o
        return None

    def _fill_table(self) -> None:
        self.table.setRowCount(0)
        keys = []
        for r in self.runs:
            for k in (r.key, r.key + ci.K_CLOSURE_SUFFIX):
                if k in self.outcomes:
                    keys.append(k)
        keys += [k for k in self.outcomes if k not in keys]
        for key in keys:
            row = summary_row(key, self.outcomes[key])
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, (name, _label) in enumerate(SUMMARY_COLUMNS):
                v = row.get(name)
                item = QtWidgets.QTableWidgetItem(v if isinstance(v, str) else _fmt(v))
                if name == "reading" and isinstance(v, str):
                    if v in ci.READING_LABELS:
                        item.setText(ci.reading_label(v, self._language()))
                    item.setToolTip((self.outcomes[key].gap.reading.sentence if self.outcomes[key].gap is not None
                                     else v) + f" [{v}]")
                self.table.setItem(r, c, item)
        for key, why in getattr(self, "excluded", []):
            if key in self.outcomes:
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, (name, _label) in enumerate(SUMMARY_COLUMNS):
                text = {"method": key, "status": "not included", "reading": why}.get(name, "")
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        ok = [k for k in keys if self.outcomes[k].ok]
        self.btn_export.setEnabled(bool(ok))
        self.btn_compare_closures.setEnabled(bool(ci.closure_pairs(self.outcomes)))
        if ok:
            self.table.selectRow(keys.index(ok[0]))
            self._draw(ok[0])
        else:
            self._draw(None)

    def _on_select(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return
        item = self.table.item(rows[0], 0)
        if item is not None:
            self._draw(item.text())

    def selected_key(self) -> Optional[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        item = self.table.item(rows[0], 0) if rows else None
        return None if item is None else str(item.text())

    def _draw(self, key: Optional[str]) -> None:
        for p in (self.plot_h1, self.plot_h2, self.plot_gaps, self.plot_g, self.plot_s):
            p.clear()
            # a range left by an earlier drawing (or a zoom) never survives a new one (manual test of 2026-10-23:
            # H1 kept +/-2e8 until the user pressed pyqtgraph's 'A')
            p.getViewBox().enableAutoRange()
        self._clear_legends()
        self._hints = {}                                 # p.clear() above took them off the plots
        self._shown_res = None
        self._drawn_key = None
        for p, title in self._base_titles.items():
            set_title(p, title, dark=self.dark)        # a title of the background in use, never one left over
        o = None if key is None else self.outcomes.get(key)
        self.btn_synth.setEnabled(bool(o is not None and o.ok and o.result.example is not None))
        self.band_source = ""
        if o is None or not o.ok:
            self.banner.setText("" if o is None else f"{key}: {o.status} -- {o.message}")
            self.banner.setStyleSheet(f"color: {verdict('warn', dark=False)};" if o is not None else "")
            self.lbl_reading.setText("")
            self._show_hints(None if o is None else f"No null drawn for {key}: {o.status} (see the message under "
                                                     "the estimate).")
            # G2: before the null, the regularity plots show the centres-level band, labelled so
            self._draw_centres_preview()
            for name, plot in self.figure_plots().items():
                entries = self.legend_entries(name)
                if entries:
                    self._legends[name] = corner_legend(plot, entries, self.dark)
            self.set_legends_visible(self.chk_legend.isChecked())
            return
        res = o.result
        self._drawn_key = key
        warn = verdict("bad", dark=False)
        if res.interpretable:
            self.banner.setStyleSheet("")
            self.banner.setText(f"{key}: {res.headline()}")
        else:
            self.banner.setStyleSheet(f"color: {warn}; font-weight: bold;")
            self.banner.setText(f"{key}: {res.headline()}")
        for plot, env, tag in ((self.plot_h1, res.h1, "H1"), (self.plot_h2, res.h2, "H2")):
            self._draw_env(plot, env, res, tag)
        self._draw_regularity(o)
        for p in (self.plot_gaps, self.plot_g, self.plot_s):
            p.getViewBox().enableAutoRange()
            p.autoRange()
        self._shown_res = res
        for name, plot in self.figure_plots().items():
            entries = self.legend_entries(name)
            if entries:
                self._legends[name] = corner_legend(plot, entries, self.dark)
        self.set_legends_visible(self.chk_legend.isChecked())

    def legend_entries(self, name: str) -> List[Tuple[str, Any]]:
        """F6 and the on-screen legend: (label, sample) of plot ``name`` as drawn now ([] before a drawing)."""
        res = self._shown_res
        if res is None:
            if name in (PLOT_GAPS, PLOT_G, PLOT_S) and self.band_source and self.plot_has_curves(name):
                return regularity_legend_entries(self.dark, self.band_source)
            return []
        if name == PLOT_H1:
            return h_legend_entries(res, res.h1, self.dark) if res.h1 is not None else []
        if name == PLOT_H2:
            return h_legend_entries(res, res.h2, self.dark) if res.h2 is not None else []
        if name in (PLOT_GAPS, PLOT_G, PLOT_S):
            return regularity_legend_entries(self.dark, self.band_source) if self.plot_has_curves(name) else []
        return []

    def _show_hints(self, text: Optional[str] = None) -> None:
        """H1 / H2 with no null drawn: a notice centred on each (``text``; default: 'No null yet', with B and the
        estimate), in a 0-1 view; the next drawing clears it. Presentation only."""
        colour = neutral(self.dark)
        for plot in (self.plot_h1, self.plot_h2):
            item = pg.TextItem(anchor=(0.5, 0.5), color=colour)
            plot.addItem(item, ignoreBounds=True)
            item.setPos(0.5, 0.5)
            plot.getViewBox().setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0.0)
            self._hints[plot] = item
        self._hint_fixed = text
        self._refresh_hints()

    def _refresh_hints(self) -> None:
        """The notice's text follows B, the closure and 'Allow slow methods' (its estimate)."""
        if not getattr(self, "_hints", None):
            return
        text = wrap_words(self._hint_fixed or self.null_hint_text())
        for item in self._hints.values():
            item.setText(text)

    def null_hint_shown(self) -> str:
        """The notice on H1 now, in one line ('' when a null is drawn)."""
        item = self._hints.get(self.plot_h1)
        if item is None or item.scene() is None:
            return ""
        return " ".join(str(item.toPlainText()).split())

    def plot_has_curves(self, name: str) -> bool:
        plot = self.figure_plots().get(name)
        return plot is not None and any(isinstance(it, pg.PlotDataItem) for it in plot.getPlotItem().items)

    def _clear_legends(self) -> None:
        for leg in self._legends.values():
            scene = leg.scene()
            if scene is not None:
                scene.removeItem(leg)
        self._legends = {}

    def set_legends_visible(self, on: bool) -> None:
        for leg in self._legends.values():
            leg.setVisible(bool(on))

    @contextlib.contextmanager
    def legends_hidden(self) -> Iterator[None]:
        """While Export image... writes a plot (it draws its own legend from legend_entries); the 'No null yet'
        notice steps aside too (it speaks to the screen, not to a figure)."""
        was = {k: leg.isVisible() for k, leg in self._legends.items()}
        hints = list(self._hints.values())
        try:
            self.set_legends_visible(False)
            for item in hints:
                item.setVisible(False)
            yield
        finally:
            for k, leg in self._legends.items():
                leg.setVisible(was.get(k, True))
            for item in hints:
                item.setVisible(True)

    def _draw_env(self, plot: Any, env: Any, res: Any, tag: str) -> None:
        if env is None:
            set_title(plot, f"{tag}: not computed (no ring)", dark=self.dark)
            return
        r = np.asarray(env.r, dtype=float)
        line = neutral(self.dark)

        def series(values: Any) -> Any:
            """The series as floats, NaN where it is not finite (drawn with connect='finite': a gap, never a jump)."""
            if values is None:
                return None
            y = np.asarray(values, dtype=float).copy()
            if y.shape != r.shape:
                return None
            y[~np.isfinite(y)] = np.nan
            return y

        obs, med = series(env.observed), series(env.median)
        p_lo, p_hi = series(env.pointwise_lo), series(env.pointwise_hi)
        g_lo, g_hi = series(env.global_lo), series(env.global_hi)
        if p_lo is not None and p_hi is not None:
            both = np.isfinite(p_lo) & np.isfinite(p_hi) & np.isfinite(r)
            if both.sum() >= 2:
                lo = pg.PlotDataItem(r[both], p_lo[both])
                hi = pg.PlotDataItem(r[both], p_hi[both])
                plot.addItem(pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(rgba("randomized", 70))))
        if med is not None:
            plot.plot(r, med, pen=pg.mkPen(line, width=1, style=QtCore.Qt.PenStyle.DashLine), connect="finite")
        mask = r >= float(res.global_r_min_nm) if math.isfinite(res.global_r_min_nm) else np.ones(r.size, bool)
        for y in (g_lo, g_hi):
            if y is None:
                continue
            y = y.copy()
            y[~mask] = np.nan
            plot.plot(r, y, pen=pg.mkPen(role("summary"), width=1.5, style=QtCore.Qt.PenStyle.DotLine), connect="finite")
        if obs is not None:
            plot.plot(r, obs, pen=pg.mkPen(role("observed"), width=2), connect="finite")
        if math.isfinite(res.global_r_min_nm):
            # below the global test's start: what the algorithm merges, not tested (grey)
            region = pg.LinearRegionItem((0.0, float(res.global_r_min_nm)), movable=False,
                                         brush=pg.mkBrush(rgba("slab", 40)))
            plot.addItem(region, ignoreBounds=True)
        for x, style in ((res.scale.fusion_nm, QtCore.Qt.PenStyle.DotLine), (res.scale.d50_nm, QtCore.Qt.PenStyle.SolidLine)):
            if x is not None and math.isfinite(float(x)):
                plot.addItem(pg.InfiniteLine(pos=float(x), angle=90, pen=pg.mkPen(line, width=1, style=style)),
                             ignoreBounds=True)
        if env.r_star is not None and math.isfinite(float(env.r_star)):
            plot.addItem(pg.InfiniteLine(pos=float(env.r_star), angle=90,
                                         pen=pg.mkPen(role("summary"), width=1, style=QtCore.Qt.PenStyle.DashLine)),
                         ignoreBounds=True)
        # Manual test of 2026-10-23: the view is set here, from the data, never left to the auto-range (a value
        # far off the others pushed H1's axis to +/-2e8 and every curve flattened). x: 0 .. the largest r; y: the
        # observed curve, the median and the pointwise band (the global envelope is drawn but does not set it).
        y_range, off = envelope_y_range([obs, med, p_lo, p_hi], [g_lo, g_hi])
        r_ok = r[np.isfinite(r)]
        vb = plot.getViewBox()
        vb.disableAutoRange()
        if r_ok.size:
            plot.setXRange(0.0, float(r_ok.max()), padding=0.02)
        if y_range is not None:
            plot.setYRange(y_range[0], y_range[1], padding=0.05)
        note = f"; {off} value(s) off the scale" if off else ""
        set_title(plot, dark=self.dark, text=f"{tag}: global p = {_fmt(env.p_global)} (B = {env.n_sim}); r* = {_fmt(env.r_star, '.0f')} nm, "
                        f"h_max = {_fmt(env.h_max, '+.3g')} nm{note}")

    def _language(self) -> str:
        """The programme's ES / EN switch (the tour controller of the main window); Spanish when there is none."""
        w: Any = self
        while w is not None:
            tour = getattr(w, "tour", None)
            if tour is not None and hasattr(tour, "language"):
                try:
                    return str(tour.language())
                except Exception:  # noqa: BLE001
                    break
            w = w.parent() if hasattr(w, "parent") else None
        return "es"

    def _edge_warning_text(self, an: Any) -> str:
        """The coloured warning for the two ways the ROI edge biases the reading: clusters removed at the edge, or
        the edge criterion switched off by the guard (ROI hugging the whole ring). Empty when neither."""
        if an is None:
            return ""
        parts = []
        note = ci.edge_note(an)
        if note:
            parts.append(f" WARNING: {ci.EDGE_WARNING} ({note}).")
        why = ci.edge_disabled_note(an)
        if why:
            lang = self._language()
            parts.append(" " + ci.EDGE_DISABLED_WARNING[lang].format(why=ci.EDGE_DISABLED_SHORT[lang]))
        return "".join(parts)

    def _draw_regularity(self, o: ci.NullOutcome) -> None:
        from tools import mps_gap_measures as gm
        an = self.analysis if (self.runs and o.key == self.runs[0].key) else next(
            (r.analysis for r in self.runs if r.key == o.key), None)
        gap = o.gap
        note = "" if an is None else ci.edge_note(an)
        warning = self._edge_warning_text(an)
        if gap is None:
            self.lbl_reading.setText("Regularity: not computed. " + note + warning)
            self.lbl_reading.setStyleSheet(f"color: {verdict('warn', dark=False)};" if warning else "")
            return
        el = gap.elongation
        mark = ("" if el is None else f" Merged-looking (elongated) clusters: {el.n_elongated} "
                f"(null: {_fmt(el.expected_under_null)}); a mark, never split.")
        # G4 / G5 (user decisions 2026-11-07): the per-axon label is descriptive; a visible warning when the edge
        # curation removed clusters
        edge = warning
        self.lbl_reading.setText(gap.reading.sentence + mark + " (descriptive only: hover for why)." + edge)
        self.lbl_reading.setToolTip(READING_TIP)
        self.lbl_reading.setStyleSheet(f"color: {verdict('warn', dark=False)};" if warning else "")
        if an is None or getattr(an, "perimeter", None) is None:
            return
        s, length, _m = gm.axon_arcs(an)
        if s is None:
            return
        bands = gm.pipeline_null_bands(o.result, int(s.size), float(length))
        if int(bands["n"][0]) > 0:
            self.band_source = BAND_PIPELINE.format(n=int(bands["n"][0]))
        else:
            # the null kept no arcs (an older null): the centres-level band, labelled so (G2)
            self._draw_centres_preview(s, length)
            return
        self._draw_bands(s, length, bands, ci.reading_label(gap.reading.level, self._language()))

    def _draw_centres_preview(self, s: Any = None, length: Optional[float] = None) -> None:
        """G2: the regularity plots of the analysis shown against the CENTRES-level null (sequential inhibition of
        its K centres with the contact hard core 2 x median r_eff on its membrane; seconds), labelled 'approx.,
        centres' -- drawn before the pipeline null ran."""
        from tools import mps_gap_measures as gm
        an = self.analysis
        if s is None:
            if an is None or getattr(an, "perimeter", None) is None:
                return
            try:
                s, length, _m = gm.axon_arcs(an)
            except Exception:  # noqa: BLE001 - nothing drawn, the reading says why
                return
            if s is None or np.asarray(s).size < 3:
                return
        r_eff = getattr(an, "median_r_eff_nm", None) if an is not None else None
        d = 2.0 * float(r_eff) if r_eff is not None and math.isfinite(float(r_eff)) else 0.0
        try:
            bands = gm.centres_null_bands(int(np.asarray(s).size), float(length), d, n_replicates=199, seed=0)
        except Exception:  # noqa: BLE001 - e.g. the hard core does not fit: nothing drawn
            return
        self.band_source = BAND_CENTRES
        self._draw_bands(np.asarray(s, dtype=float), float(length), bands, "before the null")
        if not self.lbl_reading.text():
            warning = self._edge_warning_text(an)
            self.lbl_reading.setToolTip(CENTRES_READING_TIP)
            self.lbl_reading.setText("Regularity bands: approx., centres (until the pipeline null runs)." + warning)
            self.lbl_reading.setStyleSheet(f"color: {verdict('warn', dark=False)};" if warning else "")

    def _draw_bands(self, s: Any, length: float, bands: Dict[str, Any], what: str) -> None:
        from tools import mps_gap_measures as gm
        obs_col = role("observed")
        shade = pg.mkBrush(rgba("randomized", 70))
        median_pen = pg.mkPen(neutral(self.dark), width=1, style=QtCore.Qt.PenStyle.DashLine)
        edges = np.asarray(bands["gap_edges_nm"], dtype=float)
        centres = 0.5 * (edges[1:] + edges[:-1])
        hist = np.histogram(gm.arc_gaps(s, length), edges)[0].astype(float)
        self._band(self.plot_gaps, centres, bands["gap_hist_band"], shade, median_pen)
        self.plot_gaps.plot(centres, hist, pen=pg.mkPen(obs_col, width=2), stepMode=None)
        gx, gy = gm.pair_correlation_arc(s, length)
        n = min(gx.size, np.asarray(bands["g_s_nm"]).size)
        self._band(self.plot_g, np.asarray(bands["g_s_nm"])[:n], np.asarray(bands["g_band"])[:, :n], shade,
                   median_pen)
        self.plot_g.plot(gx, gy, pen=pg.mkPen(obs_col, width=2))
        modes = np.asarray(bands["modes"], dtype=float)
        self._band(self.plot_s, modes, bands["s_band"], shade, median_pen)
        self.plot_s.plot(modes, gm.structure_factor(s, length, modes), pen=pg.mkPen(obs_col, width=2))
        set_title(self.plot_s, dark=self.dark, text=f"S(m): {what} (bands: {self.band_source})")
        for p, base in ((self.plot_gaps, PLOT_GAPS), (self.plot_g, PLOT_G)):
            set_title(p, dark=self.dark, text=f"{base} (bands: {self.band_source})")

    @staticmethod
    def _band(plot: Any, x: Any, band: Any, brush: Any, median_pen: Any) -> None:
        b = np.asarray(band, dtype=float)
        x = np.asarray(x, dtype=float)
        if b.ndim != 2 or b.shape[0] != 3 or b.shape[1] == 0 or b.shape[1] != x.size:
            return
        lo = pg.PlotDataItem(x, b[0])
        hi = pg.PlotDataItem(x, b[2])
        plot.addItem(pg.FillBetweenItem(lo, hi, brush=brush))
        plot.plot(x, b[1], pen=median_pen)

    # -- the synthetic axon and the export -----------------------------------------------------------------------
    def show_synthetic(self) -> Optional["SyntheticAxonDialog"]:
        key = self.selected_key()
        o = None if key is None else self.outcomes.get(key)
        if o is None or not o.ok or o.result.example is None:
            return None
        an = next((r.analysis for r in self.runs if r.key == key), self.analysis)
        dlg = SyntheticAxonDialog(an, o.result, key, parent=self)
        dlg.show()
        self._synthetic = dlg
        return dlg

    def export_curves(self, path: Optional[str] = None) -> Dict[str, str]:
        """``<stem>_ripley.csv`` of the method selected and ``<stem>_nulls.csv`` (the summary table)."""
        from tools import multimethod as mm
        key = self.selected_key()
        o = None if key is None else self.outcomes.get(key)
        if o is None or not o.ok:
            return {}
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Write the null's curves", "axon_null.csv",
                                                            "CSV (*.csv)")
            if not path:
                return {}
        from tools.results_table import saved_as_notes
        stem = path[:-4] if path.lower().endswith(".csv") else path     # every file takes the name chosen
        rows = o.result.ripley_rows()
        cols: List[str] = []
        for r in rows:
            cols += [c for c in r if c not in cols]
        out = {"ripley": mm.write_rows_csv(stem + "_ripley.csv", rows, cols)}
        summ = [summary_row(k, v) for k, v in self.outcomes.items()]
        out["nulls"] = mm.write_rows_csv(stem + "_nulls.csv", summ, [c[0] for c in SUMMARY_COLUMNS])
        # C3 / C4: the axon's nulls one row per method x closure (null_closure per row), as the Batch's
        # <table>_null_by_closure.csv
        closure_rows: List[Dict[str, Any]] = []
        for run in self.runs:
            found = {c: self.outcomes[k] for c, k in (("n_locs", run.key),
                                                      ("k_clusters", run.key + ci.K_CLOSURE_SUFFIX))
                     if k in self.outcomes}
            outs = {getattr(oc, "closure", c) or c: oc for c, oc in found.items()}
            if outs:
                head = {"axon_id": str(getattr(self.analysis, "source_name", "")), "source":
                        str(getattr(self.analysis, "source_name", "")), "cluster_method": run.method}
                for row in ci.null_closure_rows(head, outs):
                    row["axon_id"] = f"{head['axon_id']} | {run.key}"
                    closure_rows.append(row)
        if closure_rows:
            out["by_closure"] = mm.write_rows_csv(stem + "_null_by_closure.csv", closure_rows,
                                                  list(ci.null_by_closure_columns()))
        cmp_rows = ci.closure_comparison_rows(self.outcomes)
        if cmp_rows:
            out["closures"] = mm.write_rows_csv(stem + "_closures_compared.csv", cmp_rows, list(cmp_rows[0]))
        expected = {"ripley": stem + "_ripley.csv", "nulls": stem + "_nulls.csv",
                    "by_closure": stem + "_null_by_closure.csv", "closures": stem + "_closures_compared.csv"}
        self._saved_as_notes = saved_as_notes(expected, out)             # a taken name is said, not changed in silence
        return out

    def _on_compare_closures(self) -> None:
        self.show_closure_comparison()

    def show_closure_comparison(self) -> Optional["ClosureComparisonDialog"]:
        """C4: the p values of the two closures side by side and their H1 / H2 envelopes on one plot."""
        if not ci.closure_pairs(self.outcomes):
            return None
        dlg = ClosureComparisonDialog(self.outcomes, parent=self)
        dlg.show()
        self._closure_dialog = dlg
        return dlg


# =====================================================================================================================
# C4: the two closures side by side
# =====================================================================================================================

CLOSURE_TABLE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("method", "Method"), ("test", "Test"), ("p_n_locs", "p (N_loc)"), ("p_k_clusters", "p (K)"),
    ("same_verdict_at_0_05", "Same verdict at 0.05"), ("h_max_n_locs", "h_max N_loc [nm]"),
    ("h_max_k_clusters", "h_max K [nm]"))


class ClosureComparisonDialog(QtWidgets.QDialog):
    """C4: per method and test the p of the N_loc and the K closure, and the H1 / H2 curves of both nulls (median
    and global envelope of each, the observed once) for the method chosen."""

    def __init__(self, outcomes: Dict[str, ci.NullOutcome], parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("closure_comparison_dialog")
        self.setWindowTitle("Null per axon: the two closures side by side")
        from tools.mps_ui_layout import open_sized
        open_sized(self, "closure_comparison_dialog.size")
        self.outcomes = outcomes
        self.pairs = ci.closure_pairs(outcomes)
        self.rows = ci.closure_comparison_rows(outcomes)
        lay = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel("Same seed, same method, same membrane: the two nulls differ only by what each "
                                "simulated axon is matched to (N_loc = the slab's localizations, default; K = the "
                                "clusters found). A test whose verdict changes with the closure depends on that "
                                "choice; read it with caution (C3: K's pilots can leave the closure off).")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.table = QtWidgets.QTableWidget(0, len(CLOSURE_TABLE_COLUMNS))
        self.table.setObjectName("closure_table")
        self.table.setHorizontalHeaderLabels([c[1] for c in CLOSURE_TABLE_COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for row in self.rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, (name, _l) in enumerate(CLOSURE_TABLE_COLUMNS):
                v = row.get(name)
                item = QtWidgets.QTableWidgetItem(v if isinstance(v, str) else _fmt(v))
                if name == "same_verdict_at_0_05" and v is False:
                    item.setForeground(pg.mkColor(verdict("warn", dark=False)))
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        lay.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Envelopes of:"))
        self.combo = QtWidgets.QComboBox()
        self.combo.setObjectName("combo_closure_method")
        for key, _a, _b in self.pairs:
            self.combo.addItem(key)
        self.combo.currentIndexChanged.connect(lambda _i: self.draw())
        row.addWidget(self.combo)
        row.addStretch(1)
        lay.addLayout(row)
        plots = QtWidgets.QHBoxLayout()
        self.plot_h1 = pg.PlotWidget()
        self.plot_h2 = pg.PlotWidget()
        for p in (self.plot_h1, self.plot_h2):
            style_dark(p)
            p.setLabels(bottom="r [nm]", left="H(r) [nm]")
            tidy_plot(p)
            p.setMinimumHeight(200)
            plots.addWidget(p)
        lay.addLayout(plots, 2)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        lay.addWidget(close)
        self.draw()

    # One stroke per curve (colour from the verified palette, shape and thickness apart so they read in grey too).
    @staticmethod
    def closure_pen(word: str, what: str) -> Any:
        """The pen of the N_loc / K closure's ``median`` or ``envelope`` curve."""
        colour = role("summary") if word == "N_loc" else role("randomized")
        qt = QtCore.Qt.PenStyle
        style, width = {("N_loc", "median"): (qt.DashLine, 3.0), ("N_loc", "envelope"): (qt.DotLine, 3.0),
                        ("K", "median"): (qt.DashDotLine, 1.5), ("K", "envelope"): (qt.DashDotDotLine, 1.5)}[
            (word, what)]
        return pg.mkPen(colour, width=width, style=style)

    def legend_entries(self) -> List[Tuple[str, Any]]:
        return [(LEG_OBSERVED, _legend_line(role("observed"), 2.5)),
                ("N_loc closure: median (thick dashes)", pg.PlotDataItem(pen=self.closure_pen("N_loc", "median"))),
                ("N_loc closure: global envelope (thick dots)",
                 pg.PlotDataItem(pen=self.closure_pen("N_loc", "envelope"))),
                ("K closure: median (thin dash-dot)", pg.PlotDataItem(pen=self.closure_pen("K", "median"))),
                ("K closure: global envelope (thin dash-dot-dot)",
                 pg.PlotDataItem(pen=self.closure_pen("K", "envelope")))]

    def draw(self) -> None:
        i = self.combo.currentIndex()
        if i < 0 or i >= len(self.pairs):
            return
        key, a, b = self.pairs[i]
        for plot, tag in ((self.plot_h1, "h1"), (self.plot_h2, "h2")):
            plot.clear()
            titles = []
            obs_done = False
            for o, word in ((a, "N_loc"), (b, "K")):
                env = getattr(o.result, tag, None) if o.ok else None
                if env is None:
                    titles.append(f"{word}: no curve")
                    continue
                r = np.asarray(env.r, dtype=float)
                if not obs_done and env.observed is not None:
                    plot.plot(r, np.asarray(env.observed, dtype=float), pen=pg.mkPen(role("observed"), width=2.5),
                              connect="finite")
                    obs_done = True
                if env.median is not None:
                    plot.plot(r, np.asarray(env.median, dtype=float), pen=self.closure_pen(word, "median"),
                              connect="finite")
                for y in (env.global_lo, env.global_hi):
                    if y is not None:
                        plot.plot(r, np.asarray(y, dtype=float), pen=self.closure_pen(word, "envelope"),
                                  connect="finite")
                titles.append(f"{word} p = {_fmt(env.p_global)}")
            set_title(plot, f"{key}: {tag.upper()} -- " + "; ".join(titles))
            plot.legend_box = corner_legend(plot, self.legend_entries(), True)


# =====================================================================================================================
# Show synthetic axon
# =====================================================================================================================

SYNTHETIC_SIZE_KEY = "synthetic_axon_dialog.size"


class SyntheticAxonDialog(QtWidgets.QDialog):
    """The observed axon and the null's saved replicate side by side: same contour, scale and colours."""

    def __init__(self, analysis: Any, result: Any, key: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("synthetic_axon_dialog")
        self.setWindowTitle(f"Synthetic axon of the null: {key}")
        # maximize / resize, most of the screen, the size the user left it (tools.mps_ui_layout, 2026-10-23)
        from tools.mps_ui_layout import open_sized
        open_sized(self, SYNTHETIC_SIZE_KEY)
        lay = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        ex = result.example
        self.plot_obs = pg.PlotWidget()
        self.plot_syn = pg.PlotWidget()
        contour = None if getattr(analysis, "perimeter", None) is None else np.asarray(analysis.perimeter.contour)
        self._draw(self.plot_obs, np.asarray(analysis.x_slab), np.asarray(analysis.y_slab),
                   np.asarray(analysis.centroids).reshape(-1, 2), contour,
                   f"observed ({method_label(str(analysis.cluster_method))}), K = {int(analysis.n_clusters_kept)}")
        syn_contour = None if ex.get("contour") is None else np.asarray(ex["contour"])
        self._draw(self.plot_syn, np.asarray(ex["x_slab"]), np.asarray(ex["y_slab"]),
                   np.asarray(ex["centroids"]).reshape(-1, 2), syn_contour,
                   f"synthetic null, replicate 1 of {result.n_accepted} (seed {result.seed}), "
                   f"K = {np.asarray(ex['centroids']).reshape(-1, 2).shape[0]}")
        # the same scale on both
        xs = np.concatenate([np.asarray(analysis.x_slab), np.asarray(ex["x_slab"])])
        ys = np.concatenate([np.asarray(analysis.y_slab), np.asarray(ex["y_slab"])])
        if xs.size:
            for p in (self.plot_obs, self.plot_syn):
                p.setXRange(float(np.min(xs)), float(np.max(xs)), padding=0.02)
                p.setYRange(float(np.min(ys)), float(np.max(ys)), padding=0.02)
                p.setAspectLocked(True)
        row.addWidget(self.plot_obs)
        row.addWidget(self.plot_syn)
        lay.addLayout(row, 1)
        note = QtWidgets.QLabel(
            "Same contour drawing, scale and colours. The synthetic axon has no structure by construction: what "
            "looks like clusters there is what blinking and this clustering method make of molecules at random.")
        note.setWordWrap(True)
        lay.addWidget(note)
        btns = QtWidgets.QHBoxLayout()
        self.btn_image = QtWidgets.QPushButton("Export image...")
        self.btn_image.setObjectName("btn_synthetic_export_image")
        self.btn_image.setToolTip("Write both views as one PNG (never over an existing file).")
        self.btn_image.clicked.connect(self._on_export_image)
        btns.addWidget(self.btn_image)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self._on_close)
        btns.addStretch(1)
        btns.addWidget(close)
        lay.addLayout(btns)

    def _on_export_image(self) -> None:
        self.export_image()

    def _on_close(self) -> None:
        self.close()

    @staticmethod
    def legend_entries() -> List[Tuple[str, Any]]:
        """(label, sample) of both views (manual test of 2026-10-23: every new plot carries a legend)."""
        return [("Localizations of the slab", pg.ScatterPlotItem(symbol="o", size=5, brush=pg.mkBrush(role("locs")),
                                                                 pen=None)),
                ("Cluster centres (this method)", pg.ScatterPlotItem(symbol="o", size=8,
                                                                     brush=pg.mkBrush(role("centroid")),
                                                                     pen=pg.mkPen(neutral(True)))),
                ("Contour (same drawing on both)", _legend_line(neutral(True), 1.5))]

    @staticmethod
    def _draw(plot: Any, x: Any, y: Any, centres: Any, contour: Optional[Any], title: str) -> None:
        style_dark(plot)
        set_title(plot, title)
        plot.legend_box = corner_legend(plot, SyntheticAxonDialog.legend_entries(), True)
        plot.plot(x, y, pen=None, symbol="o", symbolSize=2, symbolBrush=pg.mkBrush(role("locs")),
                  symbolPen=None)
        if centres.size:
            plot.plot(centres[:, 0], centres[:, 1], pen=None, symbol="o", symbolSize=7,
                      symbolBrush=pg.mkBrush(role("centroid")), symbolPen=pg.mkPen(neutral(True)))
        if contour is not None and np.asarray(contour).size:
            c = np.vstack([contour, contour[:1]])
            plot.plot(c[:, 0], c[:, 1], pen=pg.mkPen(neutral(True), width=1.5))
        plot.setLabels(bottom="x [nm]", left="y [nm]")

    def export_image(self, path: Optional[str] = None) -> Optional[str]:
        from tools.results_table import free_path
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export image", "synthetic_axon.png", "PNG (*.png)")
            if not path:
                return None
        target = str(free_path(path))
        self.grab().save(target, "PNG")
        return target


__all__ = ["WorkingScalePanel", "ArrangementPanel", "NullWorker", "SyntheticAxonDialog", "ClosureComparisonDialog",
           "summary_row", "band_label", "BAND_PIPELINE", "BAND_CENTRES", "CLOSURE_ITEMS", "MEMBRANE_ITEMS",
           "B_CHOICES", "TAB_TITLE", "NULL_BUTTON", "SYNTH_BUTTON", "WORKING_SCALE_TITLE"]
