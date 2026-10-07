# -*- coding: utf-8 -*-
"""
Offscreen test of the criteria switches on screen (H6 toggles, D-43; the user's request of 2026-10-02: "que se elijan
los anillos clickeando y desclickeando los criterios ... con sus consecuentes cambios en la interfaz ... cambiando en
tiempo real el análisis"): ``tools/mps_selection_ui.py`` (the switches and the program's one selection), the Z quality
view's live tint, the column review's live re-evaluation and background re-run, the exploration log and its counter,
and the simulated-null and batch dialogs' command lines.

R8: every axon here is SIMULATED (tools.mps_zquality_window.DEMO_CASES through batch_columns.write_simulated_input):
column statistics are computed on simulated axons only.

Checks are numbered as they print; each states what it expects and why (D-43, SPEC_TOGGLES sections 3-5).

Run:  venv\\Scripts\\python.exe test_selection_ui_gui.py      (offscreen; ~3-5 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
WORK = tempfile.mkdtemp(prefix="test_selection_ui_")
# the exploration log of this test, never the user's (tools.mps_selection.default_log_path)
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")

import csv  # noqa: E402
import filecmp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
N_NULL = 19
T0 = time.perf_counter()


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-4:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def need(st: Dict[str, Any], key: str) -> Any:
    if key not in st:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return st[key]


def main() -> int:
    from PyQt5 import QtCore, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import batch_columns as bc
    from tools import mps_columns_window as mcw
    from tools import mps_selection as ms
    from tools import mps_selection_ui as su
    from tools import mps_zquality_window as zw

    st: Dict[str, Any] = {}
    state = su.selection_state()

    def pump(seconds: float = 0.05) -> None:
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            app.processEvents()
            time.sleep(0.005)

    def wait_idle(w: Any, timeout: float = 300.0) -> None:
        """Until no run is in flight and no live run is waiting for its debounce."""
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            pump(0.05)
            if not w.is_running() and not w._debounce.isActive():
                pump(0.1)
                if not w.is_running() and not w._debounce.isActive():
                    return
        raise AssertionError("the window did not become idle")

    print("=" * 100)
    print("CRITERIA SWITCHES (H6 toggles, D-43): widget, Z view, review window, dialogs -- SIMULATED axons only (R8)")
    print("=" * 100)

    # ------------------------------------------------------------------ 0. simulated axons
    def simulate() -> str:
        for name in ("marginal", "none"):
            contour, seed, s = zw.DEMO_CASES[name]
            st[f"npz_{name}"] = bc.write_simulated_input(os.path.join(WORK, f"sim_{name}.npz"), contour, seed, s)
        return ", ".join(os.path.basename(st[k]) for k in ("npz_marginal", "npz_none"))

    check("0. two SIMULATED demo axons (marginal, none)", simulate)

    # ------------------------------------------------------------------ 1. the widget and the state
    print("\n1. The switches and the program's one selection")

    def widget_basics() -> str:
        s1 = su.SelectionState()
        w1, w2 = su.SelectionWidget(s1), su.SelectionWidget(s1)
        assert not w1.missing_tooltips, w1.missing_tooltips
        for name in su.SELECTION_UI_TOOLTIPS:
            obj = w1.findChild(QtWidgets.QWidget, name)
            assert obj is not None and obj.toolTip().strip(), name
        assert s1.spec.is_default and "pre-specified rule" in w1.label.text() and "EXPLORATORY" not in w1.label.text()
        w1.v2_row.checks["leak"].setChecked(False)
        assert s1.spec.label == "v2[peak,valley,count]", s1.spec.label
        assert not w2.v2_row.checks["leak"].isChecked() and "EXPLORATORY" in w2.label.text()
        assert s1.spec.hash in w2.label.text()
        assert not w1.v2_row.leak_combo.isEnabled(), "the leak estimate of an unchecked leak is still editable"
        w1.reset_button.click()
        assert s1.spec.is_default and w2.v2_row.checks["leak"].isChecked()
        # with only rule v2 the switches are one row of controls: the v2c row and the variant combo are hidden
        assert w1.v2c_row.isHidden() and w1.variant_combo.isHidden() and not w1.v2_row.isHidden()
        return "two widgets on one state stay in step; reset; tooltips on every control"

    check("1.1 widget <-> state <-> label: a checkbox sets the state, a second widget follows, reset restores",
          widget_basics)

    def warnings_per_spec() -> str:
        s1 = su.SelectionState()
        w1 = su.SelectionWidget(s1)
        seen = []
        for label in ("v2[peak,valley,count,leak:D-39]", "v2c-A[peak,valley,count,leak:tnfix]", "v2c-A[peak,count]",
                      "v2c-B[peak,valley,count,leak:tnfix]", "v2[peak,valley,count,leak:max]",
                      "v2[peak] AND v2c-C[valley,leak:D-39]"):
            s1.set_label(label)
            text = w1.warnings_label.text()
            want = [t for _lv, t in ms.selector_warnings(s1.spec)]
            for t in want:
                assert su._esc(t) in text, (label, t)
            got_ac = ms.WARNING_AC_VALLEY in " ".join(want)
            assert got_ac == (s1.spec.uses_v2c and s1.spec.variant in ("A", "C") and s1.spec.v2c.valley), label
            assert (ms.WARNING_EXPLORATORY in want) == s1.spec.exploratory, label
            seen.append(len(want))
        return f"warnings per selection {seen} (the texts of tools.mps_selection.selector_warnings)"

    check("1.2 the plain warnings follow the selection (A/C valley, B, no leak, D-39, tnfix, exploratory)",
          warnings_per_spec)

    def rule_and_variant() -> str:
        s1 = su.SelectionState()
        w1 = su.SelectionWidget(s1)
        w1.rule_combo.setCurrentIndex([d for _t, d in su.RULE_ENTRIES].index(("both", None)))
        assert s1.spec.rule == "both" and not w1.variant_combo.isHidden() and not w1.v2c_row.isHidden()
        w1.variant_combo.setCurrentIndex(ms.VARIANTS.index("C"))
        assert s1.spec.label == "v2[peak,valley,count,leak:D-39] AND v2c-C[peak,valley,count,leak:tnfix]", s1.spec.label
        w1.v2c_row.leak_combo.setCurrentIndex(ms.LEAK_ESTIMATES.index("max"))
        w1.v2c_row.checks["leak"].setChecked(False)
        w1.v2c_row.checks["leak"].setChecked(True)
        assert s1.spec.v2c.leak_estimate == "max", "re-checking the leak lost its estimate"
        w1.enable_v2c(False, "no rule v2c here")
        model = w1.rule_combo.model()
        greyed = [i for i, (_t, (r, _l)) in enumerate(su.RULE_ENTRIES)
                  if not model.item(i).isEnabled()]  # type: ignore[attr-defined,unused-ignore]
        assert greyed == [1, 2, 3, 4, 5], greyed
        return s1.spec.label

    check("1.3 rule combo, AND / OR with the variant combo, the leak estimate kept, v2c entries greyed",
          rule_and_variant)

    # ------------------------------------------------------------------ 2. the Z quality view
    print("\n2. The Z quality view: live tint, default exports unchanged")

    def zq_default() -> str:
        state.reset()
        w = zw.open_z_quality_for_inputs(zw.inputs_from_npz(need(st, "npz_marginal")))
        w.wait(300)
        assert w.report is not None, w.error
        st["zq"] = w
        assert w.criteria is not None, w.criteria_error
        assert not w.missing_tooltips, w.missing_tooltips
        for t in (w.pair_table, w.v2c_table):
            for i in range(t.rowCount()):
                for j in range(t.columnCount()):
                    it = t.item(i, j)
                    assert it is None or it.background().style() == QtCore.Qt.BrushStyle.NoBrush, (t.objectName(), i, j)
        assert w.pairs_title.text() == zw.PAIRS_TITLE
        assert "never selects pairs" in w.v2c_title.text() and "D-43" in w.v2c_title.text()
        st["zq_texts"] = [[w.pair_table.item(i, j).text() for j in range(w.pair_table.columnCount())]
                          for i in range(w.pair_table.rowCount())]
        a, b = os.path.join(WORK, "zq_default"), os.path.join(WORK, "zq_direct")
        files = w.export_report(a)
        first = os.path.basename(files[0])
        stem = first[:-len("_zquality_rings.csv")]
        files += w.export_cluster_report(a, stem=stem) + w.export_v2c_report(a, stem=stem)
        files += w.export_selection_report(a, stem=stem)
        assert not any(f.endswith("_zquality_selection.csv") for f in files), files
        assert len(files) == 7 + 2 + 1, [os.path.basename(f) for f in files]
        direct = (zw.write_report_files(w.report, b, stem=stem) + zw.write_cluster_report_files(w.report, b, stem=stem)
                  + zw.write_v2c_report_files(w.report, b, stem=stem))
        for f in direct:
            twin = os.path.join(a, os.path.basename(f))
            assert filecmp.cmp(f, twin, shallow=False), os.path.basename(f)
        st["zq_stem"] = stem
        return f"10 files (7 + 2 + 1), the {len(direct)} data files byte-identical to the report's own writers"

    check("2.1 default: no tint, today's titles, the export is today's 7 + 2 + 1 files (data byte-identical)",
          zq_default)

    def zq_tint() -> str:
        w = need(st, "zq")
        t0 = time.perf_counter()
        state.set_label("v2[peak,valley,count]")
        pump(0.02)
        dt = time.perf_counter() - t0
        res = w.selection_result
        assert res is not None and res.spec.label == "v2[peak,valley,count]"
        assert sorted(res.viable) == [(0, 1), (1, 2)] and not res.marginal, (res.viable, res.marginal)
        good = QtCore.Qt.BrushStyle.SolidPattern
        for i in range(w.pair_table.rowCount()):
            assert w.pair_table.item(i, 0).background().style() == good, i
            assert "Exploratory selection" in w.pair_table.item(i, 0).toolTip()
        assert all(w.ring_table.item(i, 0).font().bold() for i in range(w.ring_table.rowCount()))
        assert "tinted" in w.selection_widget.label.text()
        # the verdict cells stay rule v2's own
        assert [[w.pair_table.item(i, j).text() for j in range(w.pair_table.columnCount())]
                for i in range(w.pair_table.rowCount())] == need(st, "zq_texts")
        out = os.path.join(WORK, "zq_explor")
        files = w.export_selection_report(out, stem=need(st, "zq_stem"))
        assert len(files) == 1 and files[0].endswith("_zquality_selection.csv")
        with open(files[0], encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert {r["selection_label"] for r in rows} == {"v2[peak,valley,count]"}
        assert {r["selection_hash"] for r in rows} == {res.hash} and len(rows) == 2
        assert dt < 0.5, f"a toggle took {dt:.3f} s"
        return f"both pairs tinted VIABLE (the marginal one: leak unchecked), rings bold; toggle {1000 * dt:.1f} ms"

    check("2.2 exploratory v2[peak,valley,count]: rows tinted, rings bold, verdicts unchanged, + selection CSV",
          zq_tint)

    def zq_v2c() -> str:
        w = need(st, "zq")
        state.set_label("v2c-B[peak,valley,count]")
        pump(0.02)
        res = w.selection_result
        rep = w.report
        for i in range(w.pair_table.rowCount()):
            assert w.pair_table.item(i, 0).background().style() == QtCore.Qt.BrushStyle.NoBrush, "v2 rows tinted"
        tinted = []
        by = {(p.ring_a, p.ring_b): p for p in res.pairs}
        for i, p in enumerate(rep.v2c.pairs):
            bg = w.v2c_table.item(i, 0).background().style()
            if p.variant != "groups_3d":
                assert bg == QtCore.Qt.BrushStyle.NoBrush, (i, p.variant)
            elif by[(p.ring_a, p.ring_b)].selected:
                assert bg == QtCore.Qt.BrushStyle.SolidPattern, i
                tinted.append(f"{p.ring_a}-{p.ring_b}")
        state.reset()
        pump(0.02)
        assert [[w.pair_table.item(i, j).text() for j in range(w.pair_table.columnCount())]
                for i in range(w.pair_table.rowCount())] == need(st, "zq_texts")
        assert all(w.pair_table.item(i, 0).background().style() == QtCore.Qt.BrushStyle.NoBrush
                   for i in range(w.pair_table.rowCount()))
        assert not any(w.ring_table.item(i, 0).font().bold() for i in range(w.ring_table.rowCount()))
        w.close()
        return f"v2c-B rows tinted: {tinted or 'none'}; back to the pre-specified rule: today's tables"

    check("2.3 v2c-B selection tints the v2c table's B rows only; reset gives today's tables back", zq_v2c)

    # ------------------------------------------------------------------ 3. the review window
    print("\n3. The column review: live re-evaluation, background run, log and counter")
    calls: List[float] = []
    orig_run = mcw.run_cleaned_analyses

    def counting(*a: Any, **k: Any) -> Any:
        calls.append(time.perf_counter())
        return orig_run(*a, **k)

    mcw.run_cleaned_analyses = counting

    def open_review(name: str) -> Any:
        launcher = mcw.start_columns_review(zw.inputs_from_npz(need(st, f"npz_{name}")), n_null=N_NULL)
        launcher.wait(600)
        w = launcher.window
        assert w is not None, launcher.error
        w.wait_viability(120)
        pump(0.1)
        assert w.viability is not None and w.criteria is not None, (w.viability_error, w.criteria_error)
        st[f"launcher_{name}"] = launcher
        return w

    def review_none_default() -> str:
        state.reset()
        w = open_review("none")
        st["rv_none"] = w
        assert w.viability is w.base_viability, "the default selection replaced rule v2's own viability"
        assert w.selection_widget.isEnabled()
        assert w.start_run()
        wait_idle(w)
        text = w.results_text()
        assert mcw.NO_VIABLE_PAIR_TEXT in text and "EXPLORATORY" not in text and not w.last_run.ran
        assert not w.selection_counter_label.isVisible() and len(calls) == 0
        return "rule v2's own viability object; nothing computed; no counter (no p shown)"

    check("3.1 'none' under the pre-specified rule: rule v2's viability itself, nothing computed", review_none_default)

    def review_live() -> str:
        w = need(st, "rv_none")
        n0 = len(calls)
        state.set_label("v2[]")
        pump(0.02)
        assert w.viability.viable == [(0, 1), (1, 2)] and w._debounce.isActive(), "no live run was scheduled"
        wait_idle(w)
        assert len(calls) == n0 + 1 and w.last_run.ran and w.n_live_runs == 1
        text = w.results_text()
        assert text.splitlines()[0].startswith(f"EXPLORATORY SELECTION v2[] #{state.spec.hash}"), text.splitlines()[0]
        primaries = []
        for label in ("v2[peak]", "v2c-B[]", "v2[count]", "v2[] OR v2c-C[count]", "v2[] AND v2c-A[peak]",
                      "v2c-B[valley]", "v2[]"):
            t0 = time.perf_counter()
            state.set_label(label)
            pump(0.02)
            dt = time.perf_counter() - t0
            assert not w.is_running() and not w._debounce.isActive(), f"{label} started a run"
            v = w.viability
            if v.viable or v.marginal:
                assert w.last_run.ran and w.last_run.primary is not None
                arc = w.last_run.analyses.arc_centroid
                want = bc.restricted_joint(arc, v.viable)
                assert w.last_run.primary["p_excess_uncalibrated"] == want["p_excess_uncalibrated"], label
                assert w.results_text().splitlines()[0].startswith("EXPLORATORY SELECTION " + state.spec.label)
                primaries.append(f"{label}: {w.last_run.primary['pairs']}")
            else:
                assert not w.last_run.ran and "passes the exploratory selection" in w.results_text(), label
                primaries.append(f"{label}: none")
            assert dt < 1.0, f"{label} took {dt:.2f} s"
        assert len(calls) == n0 + 1, f"{len(calls) - n0} analyses for 8 toggles"
        return f"ONE run_cleaned_analyses for 8 toggles; {'; '.join(primaries)}"

    check("3.2 unchecking every criterion selects the pairs: ONE background run, then every toggle re-drawn at once",
          review_live)

    def review_log() -> str:
        w = need(st, "rv_none")
        log = ms.ExplorationLog(ms.default_log_path())
        rows = [r for r in log.rows() if r["source"] == "review" and r["axon_id"] == w.axon_id
                and r["session_id"] == ms.SESSION_ID]
        labels = sorted({r["selection_label"] for r in rows})
        assert len(rows) == len(labels), "a selection was logged twice for one cluster set"
        assert set(labels) == {"v2[]", "v2c-B[]", "v2[] OR v2c-C[count]", "v2[] AND v2c-A[peak]", "v2c-B[valley]"} or \
            len(labels) >= 4, labels
        n = len(labels)
        text = w.selection_counter_label.text()
        assert f"{n} selection variants tried this session" in text, text
        assert "EXPLORATORY SELECTION" in text and "not corrected for trying several selections" in text
        assert all(r["primary_p_excess"] for r in rows) and all(r["pair_p"] for r in rows)
        return f"{n} rows (one per selection shown with p): {labels}; counter: {n} variants"

    check("3.3 every selection a p was shown under is logged once; the counter next to the p says how many",
          review_log)

    def review_export() -> str:
        w = need(st, "rv_none")
        state.set_label("v2[]")
        pump(0.05)
        path = os.path.join(WORK, "review_export", "decisions.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        written = w.export_tables(path, duplicates="append")
        with open(written[1], encoding="utf-8", newline="") as fh:
            row = list(csv.DictReader(fh))[0]
        rule = row["viability_rule"]
        assert rule.startswith(f"exploratory selection v2[] #{state.spec.hash} | "), rule
        assert row["viable_pairs"] == "0-1 1-2"
        state.set_label("v2c-B[]")
        pump(0.05)
        refused = ""
        try:
            w.export_tables(path, duplicates="append")
        except ValueError as exc:
            refused = str(exc)
        assert refused, "rows of another selection were appended to the same results table"
        state.reset()
        pump(0.05)
        assert w.viability is w.base_viability
        return f"rule text {row['viability_rule'][:46]}...; another selection into the same table refused"

    check("3.4 export: the selection's label and hash in viability_rule; another selection's rows are refused",
          review_export)

    def review_stale() -> str:
        state.reset()
        w = open_review("none")
        st["rv_none2"] = w
        assert w.start_run()
        wait_idle(w)
        real = mcw.run_cleaned_analyses

        def slow(*a: Any, progress: Any = None, **k: Any) -> Any:
            calls.append(time.perf_counter())
            if progress is not None:
                progress("slow start (test)", 0.05)
            time.sleep(1.5)
            if progress is not None:
                progress("slow middle (test)", 0.1)     # raises when the run was superseded
            return orig_run(*a, progress=progress, **k)

        mcw.run_cleaned_analyses = slow
        try:
            n0 = len(calls)
            state.set_label("v2[]")
            t_end = time.perf_counter() + 10
            while not w.is_running() and time.perf_counter() < t_end:
                pump(0.02)
            assert w.is_running() and w._run_live, "the live run did not start"
            state.set_label("v2[peak]")         # no pair selected: nobody needs that run any more
            pump(0.02)
            assert w._cancel_seq == w._run_seq, "the unneeded run was not superseded"
            wait_idle(w)
            assert not w._analyses, "a superseded run's analyses were kept"
            assert not w.last_run.ran and "passes the exploratory selection" in w.results_text()
            assert "stopped" in w.selection_note.text()
            state.set_label("v2[]")
            wait_idle(w)
            assert w.last_run.ran and len(calls) == n0 + 2
        finally:
            mcw.run_cleaned_analyses = real
            state.reset()
        return "the live run was stopped between two steps, nothing kept; the next request ran and showed its results"

    check("3.5 a live run nobody needs any more is stopped (stale run discarded); the next one runs", review_stale)

    def review_live_off() -> str:
        w = need(st, "rv_none2")
        state.reset()
        pump(0.05)
        w._analyses.clear()
        w.last_run = mcw.ReviewRun(w.viability, None, fp=ms.lumen_fp(w.decisions))
        w.live_update_check.setChecked(False)
        n0 = len(calls)
        state.set_label("v2[]")
        pump(0.5)
        assert len(calls) == n0 and not w.is_running() and not w._debounce.isActive()
        assert "press 'Apply and re-run analyses'" in w.selection_note.text()
        w.live_update_check.setChecked(True)
        wait_idle(w)
        assert len(calls) == n0 + 1 and w.last_run.ran
        state.reset()
        return "Live update off: 'press Apply and re-run'; on again: the run starts"

    check("3.6 'Live update' off: no run, the window says to press the button", review_live_off)

    def review_marginal() -> str:
        state.reset()
        w = open_review("marginal")
        st["rv_marg"] = w
        assert w.viability is w.base_viability and w.viability.viable and w.viability.marginal
        assert w.start_run()
        wait_idle(w)
        run = w.last_run
        assert run.ran and run.sensitivity is not None and "SENSITIVITY" in w.results_text()
        assert "EXPLORATORY" not in w.results_text() and "1 selection variant tried" in w.selection_counter_label.text()
        n0 = len(calls)
        state.set_label("v2[peak,valley,count]")
        pump(0.05)
        run2 = w.last_run
        both = sorted(w.base_viability.viable + w.base_viability.marginal)
        assert run2.ran and sorted(run2.viability.viable) == both and not run2.viability.marginal
        assert run2.sensitivity is None and "SENSITIVITY: viable + marginal" not in w.results_text()
        want = bc.restricted_joint(run2.analyses.arc_centroid, both)
        assert run2.primary["p_excess_uncalibrated"] == want["p_excess_uncalibrated"] and len(calls) == n0
        state.reset()
        pump(0.05)
        assert w.last_run.sensitivity is not None and "SENSITIVITY" in w.results_text()
        return (f"default: primary {run.primary['pairs']} + sensitivity; leak unchecked: no MARGINAL tier, primary "
                f"{run2.primary['pairs']} (restricted_joint), no new run")

    check("3.7 'marginal': unchecking the leak removes the MARGINAL tier (both pairs VIABLE), no new analysis",
          review_marginal)

    def review_toggle_during_run() -> str:
        # final review F1: a toggle while a BUTTON run is in flight re-draws the analyses already on screen under the
        # new selection at once (they do not depend on it): no p of the previous selection stays on screen
        w = need(st, "rv_marg")
        state.reset()
        pump(0.05)
        wait_idle(w)
        assert w.last_run is not None and w.last_run.ran
        real = mcw.run_cleaned_analyses

        def slow(*a: Any, progress: Any = None, **k: Any) -> Any:
            calls.append(time.perf_counter())
            time.sleep(2.0)
            return orig_run(*a, progress=progress, **k)

        mcw.run_cleaned_analyses = slow
        try:
            assert w.start_run()
            pump(0.1)
            state.set_label("v2[peak,valley,count]")
            pump(0.1)
            assert w.is_running(), "the run ended before the check"
            first = w.results_text().splitlines()[0]
            assert first.startswith(f"EXPLORATORY SELECTION {state.spec.label} #{state.spec.hash}"), first
            want = bc.restricted_joint(w.last_run.analyses.arc_centroid, w.viability.viable)
            assert w.last_run.primary["p_excess_uncalibrated"] == want["p_excess_uncalibrated"]
            assert f"#{state.spec.hash}" in w.selection_counter_label.text()
            state.set_label("v2[peak,valley,count,leak:D-39] AND v2c-A[peak,valley,count,leak:tnfix]")
            pump(0.1)
            assert w.is_running(), "the run ended before the check"
            assert not w.viability.viable and not w.viability.marginal
            assert not w.last_run.ran and "passes the exploratory selection" in w.results_text()
            assert "p_excess" not in w.results_text()
            wait_idle(w)
            # the run, when it ends, is shown under the selection current then (no pair: nothing shown)
            assert not w.last_run.ran and "passes the exploratory selection" in w.results_text()
            state.reset()
            pump(0.05)
            assert w.last_run.ran and "EXPLORATORY" not in w.results_text()
        finally:
            mcw.run_cleaned_analyses = real
            state.reset()
            pump(0.05)
        return "re-drawn under the new selection during the run; a selection without pairs clears the p at once"

    check("3.8 a toggle while a button run is in flight: the results follow the selection at once (final review F1)",
          review_toggle_during_run)

    # ------------------------------------------------------------------ 4. the dialogs
    print("\n4. The simulated-null and batch dialogs")

    def simnull_command() -> str:
        from tools import mps_simnull_ui as snu
        w = need(st, "rv_marg")
        state.reset()
        d = snu.SimnullDialog(w.review_inputs, w.decisions, w.viability, source_name=w.source_name,
                              criteria=w.criteria, base_viability=w.base_viability,
                              log_path=os.path.join(WORK, "l.csv"),
                              axon_ids=[w.axon_id])
        assert not d.missing_tooltips, d.missing_tooltips
        d.out_dir = os.path.join(WORK, "simnull_cmd")
        cmd = d.command("x.npz", None)
        assert "--zq-selection" not in cmd and cmd[cmd.index("--exploration-log") + 1].endswith("l.csv")
        item1 = d.mode_combo.model().item(1)  # type: ignore[attr-defined,unused-ignore]
        assert item1.isEnabled(), "the marginal mode is greyed with a marginal pair"
        state.set_label("v2[peak,valley,count]")
        pump(0.02)
        cmd2 = d.command("x.npz", None)
        assert cmd2[cmd2.index("--zq-selection") + 1] == "v2[peak,valley,count]"
        assert not item1.isEnabled(), "no MARGINAL tier without the leak criterion"
        assert d.start_button.isEnabled() and "v2[peak,valley,count]" in d.info_label.text()
        state.set_label("v2[peak,valley]")
        pump(0.02)
        state.set_label("v2c-A[peak,valley,count,leak:tnfix]")
        pump(0.02)
        assert not d.start_button.isEnabled() and "No VIABLE pair" in d.status_label.text()
        state.reset()
        pump(0.02)
        assert d.start_button.isEnabled()
        d.close()
        return f"default: {' '.join(cmd[-2:])[:40]}...; exploratory adds --zq-selection; no VIABLE pair: Start off"

    check("4.1 simnull dialog: --zq-selection only under an exploratory selection, --exploration-log always",
          simnull_command)

    def simnull_result() -> str:
        import power_columns as pc
        from tools import mps_simnull_ui as snu
        out = os.path.join(WORK, "simnull_read")
        os.makedirs(out, exist_ok=True)
        spec = ms.SelectionSpec.from_label("v2[peak,count]")
        paths = pc.simnull_paths(out, "ax")
        with open(paths["observed"], "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["arcc_z_A_sel", "zq_key", "zq_mode"])
            wr.writeheader()
            wr.writerow({"arcc_z_A_sel": "1.5", "zq_key": "3|0-1", "zq_mode": f"v2-viable@sel:{spec.hash}"})
        r1 = snu.read_simnull_result(out, "ax", 3)
        assert r1.selection_hash == spec.hash and r1.base_mode == "v2-viable" and "MISSING" in r1.message
        pc.write_selection_sidecar(os.path.join(out, f"ax{pc.SELECTION_FILE_SUFFIX}"), spec, None,
                                   f"v2-viable@sel:{spec.hash}")
        r2 = snu.read_simnull_result(out, "ax", 3)
        assert r2.selection_label == spec.label and "MISSING" not in r2.message
        return f"@sel:{spec.hash} read; label {r2.selection_label} from the sidecar; a missing sidecar is said"

    check("4.2 simnull result: the '@sel:<hash>' mode and the sidecar's label are read and shown", simnull_result)

    def batch_command() -> str:
        from tools.mps_columns_batch_ui import ColumnsBatchDialog
        state.reset()
        d = ColumnsBatchDialog()
        d.open_when_done = False
        d.out_edit.setText(os.path.join(WORK, "batch_cmd"))
        cmd = d.command("list.txt")
        assert "--selection" not in cmd and "--exploration-log" in cmd
        state.set_label("v2[peak,valley,count,leak:D-39] OR v2c-B[peak,valley,count]")
        cmd2 = d.command("list.txt")
        assert cmd2[cmd2.index("--selection") + 1] == state.spec.label, cmd2
        d.add_input(need(st, "npz_marginal"))
        ex = d.open_explorer()
        assert ex is not None and ex.inputs() == [os.path.abspath(need(st, "npz_marginal"))]
        ex.close()
        state.reset()
        d.close()
        return "--selection <label> only when exploratory; the explorer opens with the dialog's inputs"

    check("4.3 batch dialog: --selection only under an exploratory selection, --exploration-log always, explorer",
          batch_command)

    # ------------------------------------------------------------------ the end
    state.reset()
    for k in ("rv_none", "rv_none2", "rv_marg"):
        if k in st:
            st[k].close()
    mcw.run_cleaned_analyses = orig_run
    print(f"\n{PASSED} passed, {FAILED} failed ({time.perf_counter() - T0:.0f} s)")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
