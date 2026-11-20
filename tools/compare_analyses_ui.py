# -*- coding: utf-8 -*-
"""
'Compare methods...' with every analysis, side by side (the clustering integration, 2026-10-22).

The dialog of Phase D (``tools.cluster_method_ui.CompareMethodsDialog``) ran the ticked methods and showed four
numbers per method. This subclass keeps its layout, buttons and behaviour (Run, Cancel, Use this method, Overlay on
the map) and adds what the user asked for -- every analysis comparable across methods:

* the analyses to run on each method's result (``tools.clustering_integration.ANALYSIS_CHOICES``: per-axon readouts,
  events and probable fusions, eclipse tolerance and apparent size on by default; the sweep and the size at eps
  20/25/30 off, with their cost in the estimate -- open questions B5, E4);
* the DBSCAN rule of this file (c_label 14, and 26 as the sensitivity) as two more runs, so the double table
  "25/10 vs rule" (decision D2) comes out of the same comparison;
* DBSCAN 25/10 always first, as the reference (``tools.multimethod``); a slow method ticked here is the permission
  to run it (open question A2);
* after the run: the readouts by method, "does the conclusion depend on the method?", the agreement between methods
  (ARI, one-to-one / split / merged) and the double table, in tabs; "Export tables..." writes
  ``<stem>_methods_long.csv`` with the agreement, the summary and the wide table beside it (A4).

The work runs in a worker thread (``ComparisonWorker``) that touches no widget.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from tools import clustering_integration as ci
from tools import multimethod as mm
from tools.cluster_method_ui import (_LIVE_WORKERS, CANCEL_NOTE, CompareMethodsDialog, RunFn, _release,
                                     _Worker, readouts_of)
from tools.method_costs import method_label, words_seconds

# Where the divider between the methods' table and the comparison's tabs is kept (tools.mps_ui_layout).
COMPARE_SPLIT_KEY = "compare_dialog.table_tabs"

INTRO = (
    "Nothing has run yet. Tick the methods and the analyses, press Run. DBSCAN 25/10 (the reference pipeline,\n"
    "Gazal et al. 2026) always runs first; every method clusters the axon shown on the same slab, ROI, curation\n"
    "and contour (2-opt from every start), with the pick-boundary rule of the analysis shown; randomization off for\n"
    "speed. Each analysis then runs on each method's result, so every readout can be read across methods.\n"
    "The verdict columns come from the synthetic benchmark only (D11), never from this axon; '(provisional)' means\n"
    "the benchmark judged the methods at initial defaults pending verification (docs/clustering_benchmark.md).\n"
    "A slow method ticked here is your permission to run it. " + CANCEL_NOTE + "\n"
    "To see two methods on the map: select another method's row and press 'Overlay on the map'.")

WIDE_TIP = "Every readout of every analysis, one row per method (the long table's wide view)."
SUMMARY_TIP = ("Per readout: the smallest and largest value over the methods, the range over the reference's value "
               "(flagged above 10 %: a flag for the eye, not a test -- open question A3), and for p-values and "
               "verdicts whether a method concludes differently from DBSCAN 25/10.")
AGREEMENT_TIP = ("Each pair of methods (the reference first): adjusted Rand index on the localizations both put in a "
                 "cluster, and the clusters matched one to one, split, merged or without a partner (the benchmark's "
                 "rules, ad hoc thresholds).")


class ComparisonWorker(_Worker):
    """``tools.clustering_integration.run_comparison`` in a thread; emits each method as it finishes (the base
    worker's ``one_done``, for the dialog's table) and the whole result at the end (``comparison``). Never touches a
    widget. A ``_Worker`` so the dialog's join-and-release bookkeeping holds it like its own."""

    comparison = QtCore.pyqtSignal(object)

    def __init__(self, run_fn: RunFn, specs: Sequence[Any], plan: mm.RunPlan, n_locs: int,
                 analyses: Sequence[str], extras: Mapping[str, Any], labels: Mapping[str, str]) -> None:
        super().__init__(run_fn, [])
        self.specs = list(specs)
        self.plan = plan
        self.n_locs = int(n_locs)
        self.analyses = list(analyses)
        self.extras = dict(extras)
        self.labels = dict(labels)
        self.cancelled = False

    def _done(self, _i: int, _n: int, run: mm.MethodRun) -> None:
        if run.ok:
            row = readouts_of(run.analysis)
            row["error"] = ""
            row["analysis"] = run.analysis
        else:
            row = {"error": run.error or run.status, "traceback": run.traceback}
        row["seconds"] = float(run.seconds)
        self.one_done.emit(self.labels.get(run.key, run.key), row)

    def run(self) -> None:
        try:
            res = ci.run_comparison(self.run_fn, self.specs, n_locs=self.n_locs, plan=self.plan, allow_slow=True,
                                    analyses=self.analyses, extras=self.extras, labels=self.labels,
                                    cancelled=lambda: self.cancelled, on_method_done=self._done)
            self.comparison.emit(res)
        except BaseException:                              # noqa: BLE001 - a cancellation or a bug: reported
            self.failed.emit(traceback.format_exc())


def _table(columns: Sequence[str], tip: str, name: str) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(columns))
    t.setObjectName(name)
    t.setHorizontalHeaderLabels(list(columns))
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setToolTip(tip)
    return t


def _cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def fill(table: QtWidgets.QTableWidget, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    table.setRowCount(0)
    for r in rows:
        i = table.rowCount()
        table.insertRow(i)
        for c, name in enumerate(columns):
            item = QtWidgets.QTableWidgetItem(_cell(r.get(name)))
            d = ci.describe_column(name)
            if d:
                item.setToolTip(d)
            table.setItem(i, c, item)
    from tools.mps_ui_layout import fit_columns
    fit_columns(table)


class CompareAnalysesDialog(CompareMethodsDialog):
    """'Compare methods...' with the analyses, the DBSCAN rule and the comparison's tables (module docstring)."""

    def __init__(self, run_fn: RunFn, n_locs: int, experimental: bool, current: str,
                 params_for: Callable[[str], Dict[str, Any]], parent: Optional[QtWidgets.QWidget] = None,
                 verdicts: Optional[Dict[str, Any]] = None, *, analysis: Any = None,
                 nena_nm: Optional[float] = None, c_label: Optional[float] = None,
                 rule: Any = None, rule26: Any = None) -> None:
        self._ready = False
        super().__init__(run_fn, n_locs, experimental, current, params_for, parent=parent, verdicts=verdicts)
        ci.register_all_plugins()
        self.base_analysis = analysis
        self.nena_nm = nena_nm
        self.c_label = c_label
        self.rule = rule
        self.rule26 = rule26
        self.comparison_result: Optional[ci.ComparisonResult] = None
        self.labels: Dict[str, str] = {}
        lay = self.layout()
        assert isinstance(lay, QtWidgets.QVBoxLayout)
        intro = self.findChild(QtWidgets.QLabel)
        if intro is not None:
            intro.setText(INTRO)
        box = QtWidgets.QGroupBox("Analyses on each method's result")
        box.setObjectName("compare_analyses_box")
        g = QtWidgets.QGridLayout(box)
        self.analysis_checks: Dict[str, QtWidgets.QCheckBox] = {}
        for i, ch in enumerate(ci.ANALYSIS_CHOICES):
            cb = QtWidgets.QCheckBox(ch.label)
            cb.setObjectName(f"chk_analysis_{ch.name}")
            cb.setChecked(ch.default_on)
            cb.setToolTip(ch.tip)
            if ch.name == "per_axon":
                cb.setEnabled(False)                      # always: the table's own readouts
            cb.toggled.connect(self._update_estimate)
            self.analysis_checks[ch.name] = cb
            g.addWidget(cb, i // 3, i % 3)
        self.chk_rule = QtWidgets.QCheckBox("DBSCAN rule (c_label {:g})".format(14.0 if c_label is None else c_label))
        self.chk_rule.setObjectName("chk_dbscan_rule")
        self.chk_rule26 = QtWidgets.QCheckBox("DBSCAN rule (c_label 26, sensitivity)")
        self.chk_rule26.setObjectName("chk_dbscan_rule26")
        why = "" if rule is not None else " Not available: this file has no NeNA measured."
        for cb, r in ((self.chk_rule, rule), (self.chk_rule26, rule26)):
            cb.setEnabled(r is not None)
            cb.setChecked(False)
            cb.setToolTip(("DBSCAN with this file's rule (docs/dbscan_rule.md): "
                           + ("" if r is None else f"eps {r.eps_nm:.1f} nm, min samples {r.min_samples}. ")
                           + "Runs beside 25/10 so the double table (decision D2) fills.") + why)
            cb.toggled.connect(self._update_estimate)
        n = len(ci.ANALYSIS_CHOICES)
        g.addWidget(self.chk_rule, n // 3 + 1, 0)
        g.addWidget(self.chk_rule26, n // 3 + 1, 1)
        lay.insertWidget(lay.indexOf(self.table), box)
        self.comparison_result_tabs = QtWidgets.QTabWidget()
        self.comparison_result_tabs.setObjectName("compare_result_tabs")
        self.table_wide = _table([], WIDE_TIP, "compare_wide_table")
        self.table_summary = _table([], SUMMARY_TIP, "compare_summary_table")
        self.table_agreement = _table([], AGREEMENT_TIP, "compare_agreement_table")
        self.table_double = _table([], "Decision D2: DBSCAN 25/10 and the DBSCAN rule of this file, side by side.",
                                   "compare_double_table")
        self.comparison_result_tabs.addTab(self.table_wide, "Readouts by method")
        self.comparison_result_tabs.addTab(self.table_summary, "Does the conclusion depend on the method?")
        self.comparison_result_tabs.addTab(self.table_agreement, "Agreement between methods")
        self.comparison_result_tabs.addTab(self.table_double, "25/10 vs rule")
        self.comparison_result_tabs.setVisible(False)
        # Manual test of 2026-10-23: the methods' table and the comparison's tabs share a divider the user can drag
        # (kept in mps_ui_layout.json); both grow with the dialog.
        from tools.mps_ui_layout import PersistentSplitter, style_splitter
        at = lay.indexOf(self.table)
        lay.removeWidget(self.table)
        # the choices above the table (the intro, the methods, the estimate, the analyses) scroll in a pane of
        # their own, so the results can take the height once they are in
        self.choices = QtWidgets.QWidget()
        self.choices.setObjectName("compare_choices")
        cl = QtWidgets.QVBoxLayout(self.choices)
        cl.setContentsMargins(0, 0, 0, 0)
        for _i in range(at):
            item = lay.takeAt(0)
            if item is None:
                continue
            if item.widget() is not None:
                cl.addWidget(item.widget())
            elif item.layout() is not None:
                sub = item.layout()
                sub.setParent(None)
                cl.addLayout(sub)
        self.choices_scroll = QtWidgets.QScrollArea()
        self.choices_scroll.setObjectName("compare_choices_scroll")
        self.choices_scroll.setWidgetResizable(True)
        self.choices_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.choices_scroll.setWidget(self.choices)
        self.results_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.results_splitter.setObjectName("compare_results_splitter")
        self.results_splitter.setChildrenCollapsible(False)
        self.results_splitter.addWidget(self.choices_scroll)
        self.results_splitter.addWidget(self.table)
        self.results_splitter.addWidget(self.comparison_result_tabs)
        self.results_splitter.setStretchFactor(0, 0)
        self.results_splitter.setStretchFactor(1, 1)
        self.results_splitter.setStretchFactor(2, 2)
        at = 0
        style_splitter(self.results_splitter)
        self._results_split = PersistentSplitter(self.results_splitter, COMPARE_SPLIT_KEY)
        self._user_split = False
        self.results_splitter.splitterMoved.connect(lambda *_a: setattr(self, "_user_split", True))
        lay.insertWidget(at, self.results_splitter, 1)
        self.btn_tables = QtWidgets.QPushButton("Export tables...")
        self.btn_tables.setObjectName("btn_compare_export")
        self.btn_tables.setToolTip("Write <name>_methods_long.csv (every readout, one row per method x readout), "
                                   "with _methods_agreement, _methods_summary and _methods_wide beside it. Never "
                                   "over an existing file.")
        self.btn_tables.setEnabled(False)
        self.btn_tables.clicked.connect(self._on_export_tables)
        btns = next((lay.itemAt(i).layout() for i in range(lay.count())
                     if lay.itemAt(i).layout() is not None and lay.itemAt(i).layout().indexOf(self.btn_run) >= 0),
                     None)
        if isinstance(btns, QtWidgets.QBoxLayout):
            btns.insertWidget(btns.indexOf(self.btn_close), self.btn_tables)
        self._ready = True
        self._update_estimate()

    # -- what runs -------------------------------------------------------------------------------------------------
    def analyses_ticked(self) -> List[str]:
        return [n for n, cb in getattr(self, "analysis_checks", {}).items() if cb.isChecked()]

    def specs(self) -> List[Any]:
        out: List[Any] = list(self.ticked())
        rule = self.rule if getattr(self, "chk_rule", None) is not None and self.chk_rule.isChecked() else None
        rule26 = self.rule26 if getattr(self, "chk_rule26", None) is not None and self.chk_rule26.isChecked() \
            else None
        out += ci.rule_method_specs(rule, rule26)
        return out

    def _update_estimate(self) -> None:
        if not getattr(self, "_ready", False):
            super()._update_estimate()
            return
        plan = mm.plan_runs(self.specs(), self.n_locs, params_for=self.params_for, allow_slow=True)
        methods = [j[1] for j in plan.jobs]
        extra = ci.estimate_analyses_seconds(self.analyses_ticked(), methods, self.n_locs)
        slow = [method_label(m) for m in methods if ci.is_slow(m)]
        self.lbl_estimate.setText(
            f"{len(plan.jobs)} run(s) (DBSCAN 25/10 first) on {self.n_locs:,} slab localizations: about "
            f"{words_seconds(plan.seconds + extra)} with the analyses (rough, per-method constants in the code; a "
            "lower bound with many localizations)" + (f"; slow: {', '.join(slow)}" if slow else "") + ".")
        if self._worker is None:
            self.btn_run.setEnabled(bool(self.ticked()) or bool(ci.rule_method_specs(self.rule, None)))

    def start(self) -> None:
        if self._worker is not None or self._pending_done is not None:
            return
        specs = self.specs()
        plan = mm.plan_runs(specs, self.n_locs, params_for=self.params_for, allow_slow=True)
        self.labels = ci.rule_labels(plan, self.rule if self.chk_rule.isChecked() else None,
                                     self.rule26 if self.chk_rule26.isChecked() else None)
        self.ran = [self.labels.get(j[0], j[0]) for j in plan.jobs]
        self.errors = []
        self.table.setRowCount(0)
        self.rows = {}
        self.analyses = {}
        self.comparison_result = None
        self.comparison_result_tabs.setVisible(False)
        self.btn_tables.setEnabled(False)
        self.progress.setRange(0, len(plan.jobs))
        self.progress.setValue(0)
        self.lbl_status.setText(f"Running {len(plan.jobs)} method(s), then the analyses... {CANCEL_NOTE}")
        extras = ci.comparison_extras(nena_nm=self.nena_nm, c_label=self.c_label, analysis=self.base_analysis)
        w = ComparisonWorker(self.run_fn, specs, plan, self.n_locs, self.analyses_ticked(), extras, self.labels)
        w.one_done.connect(self._on_one)
        w.comparison.connect(self._on_comparison)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_finished)
        w.finished.connect(lambda w=w: QtCore.QTimer.singleShot(0, lambda: _release(w)))
        _LIVE_WORKERS.add(w)
        self._worker = w
        self._set_running(True)
        self._t0 = time.perf_counter()
        w.start()

    # -- the result ------------------------------------------------------------------------------------------------
    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if not getattr(self, "_split_restored", False):
            self._split_restored = True
            QtCore.QTimer.singleShot(0, self._restore_split)

    def _restore_split(self) -> None:
        self._user_split = self._results_split.restore()
        if not self._user_split:
            total = sum(self.results_splitter.sizes())
            choices = min(self.choices.sizeHint().height() + 4, int(total * 0.6))
            self.results_splitter.setSizes([choices, max(total - choices, 60), 0])

    def _share_results(self) -> None:
        """The results are in: unless the user placed the dividers (this run or a session before), the methods'
        table gets its rows' height, the tabs the rest, the choices what is left down to a few lines."""
        if getattr(self, "_user_split", False):
            return
        total = sum(self.results_splitter.sizes())
        if total <= 0:
            return
        t = self.table
        rows = t.horizontalHeader().height() + 2 * t.frameWidth() + 4 + sum(
            t.rowHeight(r) for r in range(t.rowCount())) + (t.horizontalScrollBar().sizeHint().height()
                                                           if t.horizontalScrollBar().isVisible() else 0)
        table = min(rows, int(total * 0.35))
        choices = min(self.choices.sizeHint().height() + 4, int(total * 0.25))
        self.results_splitter.setSizes([choices, table, max(total - choices - table, 100)])

    def _on_comparison(self, res: ci.ComparisonResult) -> None:
        self.comparison_result = res
        heads = list(mm.WIDE_HEAD_COLUMNS)
        wide = [dict(r, method_key=self.labels.get(str(r.get("method_key")), r.get("method_key")))
                for r in res.wide_rows]
        cols = ["method_key"] + [c for c in (wide[0] if wide else {}) if c not in heads]
        fill(self.table_wide, wide, cols + ["run_status", "error"])
        fill(self.table_summary, res.summary_rows,
             ["readout", "kind", "reference_value", "min_value", "min_method", "max_value", "max_method",
              "range_rel_reference", "large_spread", "reference_verdict", "depends_on_method",
              "methods_disagreeing"])
        fill(self.table_agreement, res.agreement_rows,
             ["method_a", "method_b", "n_a", "n_b", "n_one_to_one", "n_a_split", "n_b_merged", "n_a_unmatched",
              "n_b_unmatched", "ari", "note"])
        double = res.double_table()
        fill(self.table_double, double, ["label"] + [c for c in ("clusters_per_um", "median_1nn_nm",
                                                                 "median_r_eff_nm", "occupancy_percent",
                                                                 "n_clusters_kept", "events_n_fusion_c14",
                                                                 "size_n_locs_median") if double])
        self.comparison_result_tabs.setTabEnabled(3, len(double) > 1)
        self.comparison_result_tabs.setVisible(True)
        QtCore.QTimer.singleShot(0, self._share_results)
        self.btn_tables.setEnabled(True)
        dep = res.depends_on_method()
        self.lbl_status.setToolTip("Readouts whose verdict changes with the method: " + (", ".join(dep) or "none"))

    def _on_finished(self) -> None:
        super()._on_finished()
        if self.comparison_result is not None:
            dep = self.comparison_result.depends_on_method()
            self.lbl_status.setText(self.lbl_status.text() + (
                f" Verdicts that change with the method: {', '.join(dep)}." if dep else
                " No p-value or verdict changes with the method."))

    def _on_export_tables(self) -> None:
        self.export_tables()

    def run_for(self, label: str) -> Optional[mm.MethodRun]:
        """The run a row of the table stands for (its label: the method, or 'DBSCAN rule (c_label 14)')."""
        if self.comparison_result is None:
            return None
        inv = {v: k for k, v in self.labels.items()}
        return self.comparison_result.run(inv.get(label, label))

    def export_tables(self, stem: Optional[str] = None) -> Dict[str, str]:
        if self.comparison_result is None:
            return {}
        if stem is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Write the comparison's tables",
                                                            "axon_methods.csv", "CSV (*.csv)")
            if not path:
                return {}
            stem = path[:-4] if path.lower().endswith(".csv") else path
            for suffix in ("_methods_long", "_methods"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
        out = ci.write_comparison(stem, self.comparison_result)
        self.written_tables = dict(out)               # the axon table's methods_long_file names it
        self.lbl_status.setText(self.lbl_status.text() + " Written: " + ", ".join(out.values()))
        return out


__all__ = ["CompareAnalysesDialog", "ComparisonWorker", "INTRO", "fill"]
