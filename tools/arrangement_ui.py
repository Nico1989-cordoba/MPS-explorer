# -*- coding: utf-8 -*-
"""
The analysis window's "Working scale" panel, its "Perimeter arrangement" tab and the "Show synthetic axon" view
(the clustering integration, 2026-10-22; plan A9 sec. 4, regularity PR 3).

* :class:`WorkingScalePanel` -- the distances below which this file's pipeline cannot tell two objects apart
  (``tools.mps_working_scale``: eps, min samples, NeNA, fusion, d50, slab), each with its origin; the editable
  c_label (14 [lit: Dempsey et al. 2011, MEA; provisional]); the DBSCAN rule of this file beside 25/10 (D2), and the
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
from PyQt5 import QtCore, QtWidgets

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
GLOBAL_TIP = ("Decision: the global envelope (maximum studentised deviation over r >= the working scale's d50) and "
              "its p. The grey shade is the pointwise 2.5-97.5 % envelope, for the eye only: crossing it at some r "
              "is not a test.")


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
        self.lbl_rule.setText(
            f"DBSCAN parameters: {mode}. The rule for this file (shown, never applied: decision D2): eps "
            f"{self.rule.eps_nm:.1f} nm, min samples {self.rule.min_samples} with c_label {self.c_label():g} "
            f"({other_rule.min_samples if other_rule is not None else '-'} with {other_c:g}); localizations per "
            f"cycle "
            f"{float(lpc.value):.2f} {lpc.origin}. Tick 'DBSCAN rule' in 'Compare methods...' to see both.")
        self.lbl_rule.setToolTip(self.rule.describe())

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
                 allow_slow: bool, seed: int = 0) -> None:
        super().__init__()
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
                               progress=lambda k, d, t: self.progress.emit(str(k), int(d), int(t)))
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
    ("seconds", "time"),
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
    row: Dict[str, Any] = {"method": key, "status": o.status, "seconds": words_seconds(o.seconds) if o.ok else ""}
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
H_HELP = ("Above the null = more neighbours than random (clustering); below = fewer (spacing / regularity). Only "
          "the global envelope decides; its p is in the title. Left of the grey zone's edge nothing is tested.")
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


def regularity_legend_entries(dark: bool = True) -> List[Tuple[str, Any]]:
    """(label, sample) of what a regularity plot draws."""
    return [(LEG_OBSERVED, _legend_line(role("observed"), 2.0)),
            (LEG_MEDIAN, _legend_line(neutral(dark), 1.0, QtCore.Qt.PenStyle.DashLine)),
            (LEG_BAND, _legend_bar(pg.mkBrush(rgba("randomized", 110))))]


# More rows than this and the corner legend takes two columns (half as tall over a short plot) [ad hoc].
CORNER_LEGEND_ONE_COLUMN = 5


def corner_legend(plot: Any, entries: Sequence[Tuple[str, Any]], dark: bool = True) -> Any:
    """A compact legend in the plot's top-right corner on a semi-transparent box (the data stay readable under
    it), in the theme's colours. Returns the LegendItem."""
    text = "#ffffff" if dark else "#000000"
    fill = (0, 0, 0, 150) if dark else (255, 255, 255, 190)
    leg = pg.LegendItem(offset=(-4, 4), labelTextColor=text, brush=pg.mkBrush(*fill), pen=pg.mkPen(text, width=0.5),
                        labelTextSize="7pt", verSpacing=-6,
                        colCount=2 if len(entries) > CORNER_LEGEND_ONE_COLUMN else 1)
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
        self.btn_export = QtWidgets.QPushButton("Export curves...")
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
        top.addStretch(1)
        lay.addLayout(top)
        self._legends: Dict[str, Any] = {}
        self._shown_res: Any = None
        self._drawn_key: Optional[str] = None
        self._base_titles: Dict[Any, str] = {}           # plot -> its title before any null is drawn
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

    def set_header(self, widget: QtWidgets.QWidget) -> None:
        """A panel at the top of the upper pane (the window's Working scale)."""
        lay = self.upper.layout()
        assert isinstance(lay, QtWidgets.QVBoxLayout)
        lay.insertWidget(0, widget)

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        super().showEvent(event)
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
            self.banner.setText("Written: " + ", ".join(out.values()))

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
        self._update_estimate()
        self._fill_table()

    def _update_estimate(self) -> None:
        if not self.runs:
            self.lbl_estimate.setText("Run the MPS analysis first: the null simulates the axon shown.")
            return
        b = int(self.combo_b.currentData() or B_CHOICES[0])
        parts = []
        total = 0.0
        refused = []
        for r in self.runs:
            s = ci.null_estimate_seconds(r, b)
            if ci.is_slow(r.method) and not self.chk_slow.isChecked():
                refused.append(f"{method_label(r.method)} (about {words_seconds(s)})")
                continue
            total += s
            parts.append(r.key)
        text = (f"{NULL_BUTTON} for {len(parts)} method(s) ({', '.join(parts)}), B = {b}: "
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
        w = NullWorker(runs, b, self.nena_nm, self.c_label, allow)
        w.progress.connect(self._on_progress)
        w.finished_outcomes.connect(self._on_outcomes)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_finished)
        w.finished.connect(lambda w=w: QtCore.QTimer.singleShot(0, lambda: _release(w)))
        _LIVE.add(w)
        self._worker = w
        self._t0 = time.perf_counter()
        self.progress.setRange(0, b * len(runs))
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
        keys = [r.key for r in self.runs if r.key in self.outcomes] + [k for k in self.outcomes
                                                                      if k not in {r.key for r in self.runs}]
        for key in keys:
            row = summary_row(key, self.outcomes[key])
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, (name, _label) in enumerate(SUMMARY_COLUMNS):
                v = row.get(name)
                item = QtWidgets.QTableWidgetItem(v if isinstance(v, str) else _fmt(v))
                if name == "reading" and isinstance(v, str):
                    item.setToolTip(self.outcomes[key].gap.reading.sentence if self.outcomes[key].gap is not None
                                    else v)
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
        self._shown_res = None
        self._drawn_key = None
        for p, title in self._base_titles.items():
            set_title(p, title, dark=self.dark)        # a title of the background in use, never one left over
        o = None if key is None else self.outcomes.get(key)
        self.btn_synth.setEnabled(bool(o is not None and o.ok and o.result.example is not None))
        if o is None or not o.ok:
            self.banner.setText("" if o is None else f"{key}: {o.status} -- {o.message}")
            self.banner.setStyleSheet(f"color: {verdict('warn', dark=False)};" if o is not None else "")
            self.lbl_reading.setText("")
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
            return []
        if name == PLOT_H1:
            return h_legend_entries(res, res.h1, self.dark) if res.h1 is not None else []
        if name == PLOT_H2:
            return h_legend_entries(res, res.h2, self.dark) if res.h2 is not None else []
        if name in (PLOT_GAPS, PLOT_G, PLOT_S):
            return regularity_legend_entries(self.dark) if self.plot_has_curves(name) else []
        return []

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
        """While Export image... writes a plot (it draws its own legend from legend_entries)."""
        was = {k: leg.isVisible() for k, leg in self._legends.items()}
        try:
            self.set_legends_visible(False)
            yield
        finally:
            for k, leg in self._legends.items():
                leg.setVisible(was.get(k, True))

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

    def _draw_regularity(self, o: ci.NullOutcome) -> None:
        from tools import mps_gap_measures as gm
        an = self.analysis if (self.runs and o.key == self.runs[0].key) else next(
            (r.analysis for r in self.runs if r.key == o.key), None)
        gap = o.gap
        note = "" if an is None else ci.edge_note(an)
        if gap is None:
            self.lbl_reading.setText("Regularity: not computed. " + note)
            return
        el = gap.elongation
        mark = ("" if el is None else f" Merged-looking clusters (elongated beyond the null's 95th percentile): "
                f"{el.n_elongated}, the null itself gives {_fmt(el.expected_under_null)} -- a mark, never split.")
        self.lbl_reading.setText(gap.reading.sentence + mark + (f" Note: {note}." if note else ""))
        if an is None or getattr(an, "perimeter", None) is None:
            return
        s, length, _m = gm.axon_arcs(an)
        if s is None:
            return
        bands = gm.pipeline_null_bands(o.result, int(s.size), float(length))
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
        set_title(self.plot_s, dark=self.dark, text=f"S(m): reading {gap.reading.level} (bands: pipeline null, B = {int(bands['n'][0])})")

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
        stem = path[:-4] if path.lower().endswith(".csv") else path
        rows = o.result.ripley_rows()
        cols: List[str] = []
        for r in rows:
            cols += [c for c in r if c not in cols]
        out = {"ripley": mm.write_rows_csv(stem + "_ripley.csv", rows, cols)}
        summ = [summary_row(k, v) for k, v in self.outcomes.items()]
        out["nulls"] = mm.write_rows_csv(stem + "_nulls.csv", summ, [c[0] for c in SUMMARY_COLUMNS])
        return out


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


__all__ = ["WorkingScalePanel", "ArrangementPanel", "NullWorker", "SyntheticAxonDialog", "summary_row",
           "B_CHOICES", "TAB_TITLE", "NULL_BUTTON", "SYNTH_BUTTON", "WORKING_SCALE_TITLE"]
