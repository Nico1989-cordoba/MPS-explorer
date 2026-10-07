# -*- coding: utf-8 -*-
"""
Offscreen test of the tolerance control of the columns review (UI stage 1, D-44; the plan approved on 2026-10-05: a
tau control -- the pre-registered tau_0, the 2D test's two sensitivity values, a free value -- and the E(tau) curve in
the review, with the EXPLORATORY label and the counter): ``tools/mps_tau_ui.py`` and its wiring in
``tools/mps_columns_window.py``.

R8: every axon here is SIMULATED (tools.mps_zquality_window.DEMO_CASES through batch_columns.write_simulated_input);
column statistics are computed on simulated axons only. Every file goes to a temporary folder (the exploration log
through MPS_SELECTION_LOG_DIR, the review stores, the exports).

What it checks (G1-G15 of the specification; I1-I5 are the invariants the checks share):
  G1  golden parity: fresh windows at tau_0 on the three demos give, after the golden's normalization, exactly what
      the program wrote before this stage (testdata/tau_control/review_tau0_golden.json) -- texts, exports, log rows,
      the simulated-null button -- and pass the window's own parameters (None) to the analyses as before (I5).
  G2  the control as it opens: the entries come from the loaded parameters (and follow modified ones), the free value
      is off, "Back to tau_0" off, no badge, the note shown, every tooltip present, right above the run row.
  G3  tau -> the larger preset after a default run: in the same event-loop turn no p of tau_0 stays on screen
      (pending text, leak banner hidden, matches empty, counter = the analysis' head only), the curve is unchanged
      with its marker moved, the simulated null is off; then exactly one run at that tau, the UI ticking meanwhile
      (I3); the first line, the 2D line, the arc line, the run's tau, the counter and the log row (I1).
  G4  the export at that tau: tau0_nm and the analysis tag; into the tau_0 table it is refused and nothing changes;
      the decisions do not depend on tau (I2).
  G5  Free and three quick steps: one run, at the last value.
  G6  stale runs: a live run at a tau the control has left is stopped and never shown (the results view is polled
      meanwhile), the new tau is computed after it; a BUTTON run likewise when the new tau was computed before (I4).
  G7  "Back to tau_0": no badge, the simulated null back with its own tooltip, the results text and the export row of
      the golden, no new log row, the counter keeps the tau wording (history).
  G8  an exploratory selection AND another tau: one label, a first line naming both; back to both defaults -> the
      pre-registered texts.
  G9  before any run a change of tau runs nothing; the button then runs at it.
  G10 the demo without a viable pair: a change of tau runs nothing and changes no text.
  G11 the E(tau) curve: the selected pairs only (VIABLE first, MARGINAL tagged), its global p "not calibrated", the
      note always visible, its arrays exactly the 2D test's curve, layer rows that drive real items and survive
      redraws, beside the plot (no legend inside).
  G12 the matches on the map: two rows after the stage-0 rows, counts = what each test matched at the current tau for
      the curve's pair, following the pair combo; no mouse button on them.
  G13 a new window starts at tau_0; the review store's JSON has the keys of a default session (tau is never saved).
  G14 'Live update' off: the window says to press the button and runs nothing; the export writes the decisions only;
      ticked again, the run follows.
  G15 a custom viability function (the criteria switches off): tau still re-runs.
Found by the review of the stage (each failed before its fix):
  G16 the warnings list follows the analyses on screen: back at tau_0 from the cache it is tau_0's list, and while
      the results at a new tau are pending no note of another tau's run stays in it.
  G17 analyses kept from a run that ended at a tau the control had left, then shown from the cache, mark the axon:
      the next edit is flagged (D-35b) and saved.
  G18 a request to stop a run never outlives that run (a run that ended normally leaves no stop reason behind).
  G19 without the criteria (switches off) a p at another tau is logged and counted, under rule v2's own label, with
      the tau_0 result shown before it; at tau_0 alone nothing is logged, as before.

Run:  venv\\Scripts\\python.exe test_tau_control_gui.py      (offscreen; about 2-3 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
WORK = tempfile.mkdtemp(prefix="test_tau_control_gui_")
# G1's exploration log (the golden's windows were the only ones of their session); the rest use another one
LOG_G1 = os.path.join(WORK, "selection_log_g1")
LOG_MAIN = os.path.join(WORK, "selection_log")
os.environ["MPS_SELECTION_LOG_DIR"] = LOG_G1

import csv  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GOLDEN = os.path.join(REPO_ROOT, "testdata", "tau_control", "review_tau0_golden.json")
N_NULL = 19
CASES = ("viable", "marginal", "none")
TICK_MS = 20
PASSED = 0
FAILED = 0
T0 = time.perf_counter()

# ============================================================ the golden's normalization (OUT/golden/make_review_golden.py)
DROP_EXPORT = ("exported_at", "run_seconds")
DROP_LOG = ("row_id", "time_iso", "session_id", "git_head")
_RUN_FINISHED = re.compile(r"\(run finished \d\d:\d\d:\d\d, \d+(?:\.\d+)? s,")
_CHECKED = re.compile(r"\(checked \d\d:\d\d:\d\d, \d+(?:\.\d+)? s;")


def normalize_text(text: Any, work: str) -> str:
    """One string with the clock times of a run and the work folder normalized."""
    s = "" if text is None else str(text)
    s = _RUN_FINISHED.sub("(run finished HH:MM:SS, S s,", s)
    s = _CHECKED.sub("(checked HH:MM:SS, S s;", s)
    for form in {work, work.replace("\\", "/"), work.replace("/", "\\")}:
        if form:
            s = s.replace(form, "<WORK>")
    return s


def normalize_row(row: Dict[str, Any], work: str) -> Dict[str, str]:
    """An exported row (decisions or results): exported_at and run_seconds dropped, source -> basename."""
    out: Dict[str, str] = {}
    for k, v in row.items():
        if k in DROP_EXPORT:
            continue
        if k == "source":
            v = os.path.basename(str(v or "").replace("\\", "/"))
        out[k] = normalize_text(v, work)
    return out


def normalize_log_row(row: Dict[str, Any], work: str) -> Dict[str, str]:
    return {k: normalize_text(v, work) for k, v in row.items() if k not in DROP_LOG}


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def rows_sha256(rows: List[Dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()


def capture(w: Any, name: str, work: str) -> Dict[str, Any]:
    """What the golden records of one window after its default run (the window idle), exactly as it records it."""
    from tools import mps_selection as ms
    out: Dict[str, Any] = {}
    out["header"] = normalize_text(w.header_label.text(), work)
    out["viability_label"] = normalize_text(w.viability_label.text(), work)
    out["results_text"] = normalize_text(w.results_text(), work)
    out["counter_text"] = normalize_text(w.selection_counter_label.text(), work)
    out["counter_visible"] = bool(w.selection_counter_label.isVisible())
    out["leak_banner_visible"] = bool(w.leak_banner.isVisible())
    out["warnings"] = [normalize_text(w.warnings_list.item(i).text(), work) for i in range(w.warnings_list.count())]
    out["simnull_enabled"] = bool(w.simnull_button.isEnabled())
    out["simnull_tooltip"] = normalize_text(w.simnull_button.toolTip(), work)
    path = os.path.join(work, f"export_{name}", "decisions.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    written = w.export_tables(path, duplicates="append")
    out["export_files"] = [os.path.basename(p) for p in written]
    dec = [normalize_row(r, work) for r in read_csv(written[0])]
    out["decisions_n_rows"] = len(dec)
    out["decisions_header"] = read_header(written[0])
    out["decisions_sha256"] = rows_sha256(dec)
    if len(written) > 1:
        rows = read_csv(written[1])
        assert len(rows) == 1, rows
        out["results_header"] = read_header(written[1])
        out["results_row"] = normalize_row(rows[0], work)
    else:
        out["results_header"] = None
        out["results_row"] = None
    log = ms.ExplorationLog(ms.default_log_path())
    out["log_rows"] = [normalize_log_row(r, work) for r in log.rows() if r.get("axon_id") == w.axon_id]
    out["axon_id"] = w.axon_id
    return out


# ============================================================ the harness
def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-5:]:
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
    from tools import mps_lumen as L
    from tools import mps_selection as ms
    from tools import mps_selection_ui as su
    from tools import mps_tau as mt
    from tools import mps_tau_ui as mtui
    from tools import mps_zquality_window as zw

    st: Dict[str, Any] = {}
    state = su.selection_state()
    cp = L.default_columns_params()
    TAU0 = float(cp.tau0_nm)
    choices = mt.tau_choices(cp)
    presets = [float(c.tau_nm) for c in choices if c.key == "preset" and c.tau_nm is not None]
    with open(GOLDEN, encoding="utf-8") as fh:
        golden: Dict[str, Any] = json.load(fh)

    # a spy of the analyses the window starts: which tau each run was given (None: the window's own parameters,
    # i.e. the pre-registered file, exactly as before this stage)
    calls: List[Optional[float]] = []
    orig_run = mcw.run_cleaned_analyses

    def spy(*a: Any, **k: Any) -> Any:
        params = k.get("columns_params")
        calls.append(None if params is None else float(params.tau0_nm))
        return orig_run(*a, **k)

    mcw.run_cleaned_analyses = spy
    orig_viability_fn = mcw.DEFAULT_VIABILITY_FN

    def pump(seconds: float = 0.05) -> None:
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            app.processEvents()
            time.sleep(0.005)

    def wait_idle(w: Any, timeout: float = 300.0, poll: Optional[Callable[[], None]] = None) -> None:
        """Until no run is in flight and no live run waits for its debounce (``poll`` called at every step)."""
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            pump(0.03)
            if poll is not None:
                poll()
            if not w.is_running() and not w._debounce.isActive():
                pump(0.15)
                if poll is not None:
                    poll()
                if not w.is_running() and not w._debounce.isActive():
                    return
        raise AssertionError("the window did not become idle")

    def open_review(name: str, store: Any = None) -> Tuple[Any, Any]:
        launcher = mcw.start_columns_review(zw.inputs_from_npz(need(st, f"npz_{name}")), n_null=N_NULL,
                                            review_store=store)
        launcher.wait(900)
        w = launcher.window
        assert w is not None, launcher.error
        w.wait_viability(300)
        pump(0.2)
        assert w.viability is not None, w.viability_error
        return launcher, w

    def run_default(w: Any) -> None:
        assert w.start_run(), "start_run refused"
        pump(0.05)
        wait_idle(w)
        pump(0.2)

    def choose(w: Any, key: str, tau: Optional[float] = None) -> None:
        ctl = w.tau_control
        for i, c in enumerate(ctl.choices()):
            if c.key == key and (tau is None or (c.tau_nm is not None and abs(float(c.tau_nm) - tau) < 1e-9)):
                ctl.tau_combo.setCurrentIndex(i)
                return
        raise AssertionError(f"the control has no {key} {tau}")

    def choose_free(w: Any, value: int) -> None:
        choose(w, "free")
        w.tau_control.tau_spin.setValue(int(value))

    def variant(spec: Any, tau: float) -> Any:
        return mt.AnalysisVariant(spec, tau, TAU0)

    def first_line(w: Any) -> str:
        lines = w.results_text().splitlines()
        return lines[0] if lines else ""

    def log_rows(w: Any) -> List[Dict[str, str]]:
        return [r for r in ms.ExplorationLog(ms.default_log_path()).rows() if r.get("axon_id") == w.axon_id]

    def counts_now(w: Any) -> Tuple[int, int]:
        log = ms.ExplorationLog(ms.default_log_path())
        return mt.variant_counts(log, [w.axon_id, os.path.splitext(os.path.basename(w.source_name))[0]])

    def no_p(text: str) -> bool:
        return not any(tok in text for tok in ("p_excess", "z_A", "two-sided p", "matched of K"))

    print("=" * 100)
    print("TAU CONTROL AND E(tau) CURVE (UI stage 1, D-44) OFFSCREEN TEST -- SIMULATED axons only (R8)")
    print("=" * 100)

    # ------------------------------------------------------------------ 0. simulated axons
    def simulate() -> str:
        for name in CASES:
            contour, seed, s = zw.DEMO_CASES[name]
            p = bc.write_simulated_input(os.path.join(WORK, f"sim_{name}.npz"), contour, seed, s)
            assert bc.input_kind(p)[0] == "simulated"
            st[f"npz_{name}"] = p
        assert presets and all(not mt.is_tau0(t, TAU0) for t in presets), presets
        return f"{', '.join(CASES)}; tau_0 and {len(presets)} preset(s) read from the parameters file"

    check("0. three SIMULATED demo axons (DEMO_CASES through batch_columns.write_simulated_input)", simulate)

    # ------------------------------------------------------------------ G1. golden parity at tau_0
    print("\nG1. The review at tau_0 is the program before this stage (golden of three simulated demos)")

    def g1_golden() -> str:
        assert golden["n_null"] == N_NULL and sorted(golden["cases"]) == sorted(CASES)
        out = []
        for name in CASES:
            assert list(golden["demo_cases"][name]) == [zw.DEMO_CASES[name][0], int(zw.DEMO_CASES[name][1]),
                                                        float(zw.DEMO_CASES[name][2])], name
            state.reset()
            n0 = len(calls)
            launcher, w = open_review(name)
            assert w.criteria is not None, w.criteria_error
            assert w.selection_widget.spec.is_default and w.tau_control.is_tau0()
            run_default(w)
            got = capture(w, name, WORK)
            want = golden["cases"][name]
            diff = [k for k in want if got.get(k) != want[k]]
            assert not diff, f"{name}: {diff}: " + "; ".join(f"{k}: {got.get(k)!r} != {want[k]!r}"[:400]
                                                             for k in diff[:2])
            assert set(got) == set(want), set(got) ^ set(want)
            # the window's own parameters (None: the pre-registered file) reach the analyses, as before
            assert all(c is None for c in calls[n0:]), calls[n0:]
            out.append(f"{name}: {len(want)} fields equal, {len(calls) - n0} run(s)")
            w.close()
            launcher.close()
            pump(0.2)
        return "; ".join(out)

    check("G1 fresh windows at tau_0 on the 3 demos == the golden (texts, exports, log rows, simnull) after "
          "normalization", g1_golden)
    os.environ["MPS_SELECTION_LOG_DIR"] = LOG_MAIN
    state.reset()

    # ------------------------------------------------------------------ G2. the control as it opens
    print("\nG2. The control as it opens")

    def g2_initial() -> str:
        launcher, w = open_review("marginal")
        st["launcher_m"], st["m"] = launcher, w
        ctl = w.tau_control
        items = [ctl.tau_combo.itemText(i) for i in range(ctl.tau_combo.count())]
        want = ([mtui.tau0_item_text(TAU0)] + [mtui.preset_item_text(t) for t in presets] + [mtui.FREE_TEXT])
        assert items == want, items
        assert ctl.tau_combo.currentIndex() == 0 and ctl.value() == TAU0 and w.tau_nm == TAU0
        assert not ctl.tau_spin.isEnabled() and ctl.tau_spin.isHidden() and not ctl.tau_reset_button.isEnabled()
        assert ctl.tau_spin.minimum() == int(mt.TAU_FREE_MIN_NM) and ctl.tau_spin.maximum() == int(mt.TAU_FREE_MAX_NM)
        assert ctl.tau_spin.singleStep() == int(mt.TAU_FREE_STEP_NM) and ctl.tau_spin.suffix() == " nm"
        assert not ctl.tau_spin.keyboardTracking()
        assert not ctl.tau_badge.isVisible() and ctl.tau_note.isVisible() and ctl.tau_note.text() == mtui.TAU_NOTE_TEXT
        assert ctl.isVisible() and ctl.objectName() == "tau_control" and not w.advanced_box.isVisible()
        assert not w.tau_missing_tooltips and not ctl.missing_tooltips, (w.tau_missing_tooltips, ctl.missing_tooltips)
        for name, text in mtui.TAU_UI_TOOLTIPS.items():
            obj = w.findChild(QtCore.QObject, name)
            assert obj is not None and obj.toolTip() == text, name
        # directly above the run row, in the side column
        lay = w.tau_control.parentWidget().layout()
        i = lay.indexOf(w.tau_control)
        nxt = lay.itemAt(i + 1).layout()
        assert nxt is not None and nxt.indexOf(w.run_button) >= 0, "the control is not right above the run row"
        # the simulated null as before, and the curve says it has no data yet; its note is always shown
        assert w.curve_note.isVisible() and w.curve_note.text() == mtui.CURVE_NOTE_TEXT
        assert w.curve_p_label.text() == mtui.CURVE_NO_RUN_TEXT, w.curve_p_label.text()
        # the presets follow the parameters file (never typed in): a modified copy offers other ones
        other = dataclasses.replace(cp, tau_sensitivity_nm=[presets[0] + 15.0, TAU0, presets[-1] + 20.0])
        c2 = mtui.TauControl(other)
        items2 = [c2.tau_combo.itemText(i) for i in range(c2.tau_combo.count())]
        assert items2 == [mtui.tau0_item_text(TAU0), mtui.preset_item_text(presets[0] + 15.0),
                          mtui.preset_item_text(presets[-1] + 20.0), mtui.FREE_TEXT], items2
        seen: List[float] = []
        c2.changed.connect(seen.append)
        c2.tau_combo.setCurrentIndex(c2.tau_combo.count() - 1)          # Free: the current tau, rounded
        assert seen == [float(round(TAU0))] and c2.tau_spin.isEnabled() and not c2.tau_spin.isHidden(), seen
        assert not c2.tau_badge.isHidden() and c2.tau_spin.value() == int(round(TAU0))
        c2.tau_spin.setValue(int(round(TAU0)))                          # the same value: no signal
        assert seen == [float(round(TAU0))]
        c2.reset()
        assert seen[-1] == TAU0 and c2.is_tau0() and c2.tau_badge.isHidden() and not c2.tau_spin.isEnabled()
        assert c2.tau_spin.isHidden()
        c3 = mtui.TauControl(None, unavailable="test")
        assert c3.value() is None and not c3.tau_combo.isEnabled() and "test" in c3.tau_note.text()
        for c in (c2, c3):
            c.deleteLater()
        return f"items {items}; spin {ctl.tau_spin.minimum()}-{ctl.tau_spin.maximum()} nm off; reset off; no badge"

    check("G2 initial state: entries from the parameters (and from modified ones), spin/reset off, no badge, note, "
          "tooltips, above the run row", g2_initial)

    # ------------------------------------------------------------------ G3. tau -> the larger preset
    print("\nG3-G8. One window (demo marginal): the larger preset, export, Free, stale runs, back, a selection too")

    def g3_change() -> str:
        w = need(st, "m")
        run_default(w)
        assert w.last_run is not None and w.last_run.ran and w.last_run.tau_nm == TAU0
        st["m_text0"] = normalize_text(w.results_text(), WORK)
        assert st["m_text0"] == golden["cases"]["marginal"]["results_text"]
        p0 = os.path.join(WORK, "g4", "tau0", "decisions.csv")
        os.makedirs(os.path.dirname(p0), exist_ok=True)
        st["p0_written"] = w.export_tables(p0, duplicates="append")
        cv0 = w.curve_shown
        assert cv0 is not None
        x0 = np.array(w.curve_observed_item.getData()[1], dtype=float)
        n_log0 = len(log_rows(w))
        n0 = len(calls)
        ticks = [0]
        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: ticks.__setitem__(0, ticks[0] + 1))
        timer.start(TICK_MS)
        hi = presets[-1]
        t_change = time.perf_counter()
        choose(w, "preset", hi)
        slot_ms = (time.perf_counter() - t_change) * 1000.0
        assert slot_ms < 100.0, f"the change of tau took {slot_ms:.0f} ms in the GUI thread"
        # the same event-loop turn: nothing of tau_0 with a p stays on screen
        text = w.results_text()
        assert text == mtui.pending_live_text(hi), text
        assert no_p(text) and not w.leak_banner.isVisible() and w.is_tau_pending()
        assert w.last_result is None and w.export_note() == "\n\n" + mt.pending_export_message(hi)
        assert w.match_counts() == {"matches_arc": 0, "matches_2d": 0}
        assert not w.layer_panel.checkbox("matches_arc").isEnabled()
        assert mtui.NO_RESULTS_AT_TAU in w.layer_panel.checkbox("matches_arc").toolTip()
        counter = w.selection_counter_label.text()
        assert "variant" not in counter and f"| tau = {mt.tau_text(hi)} nm" in counter, counter
        y_now = np.array(w.curve_observed_item.getData()[1], dtype=float)
        assert np.array_equal(y_now, x0), "the curve changed with tau"
        assert abs(w.curve_tau_line.value() - hi) < 1e-9 and w.curve_tau_line.isVisible()
        assert not w.simnull_button.isEnabled() and w.simnull_button.toolTip() == mtui.SIMNULL_TAU_TIP
        assert w.tau_control.tau_badge.isVisible() and w.tau_control.tau_reset_button.isEnabled()
        assert w.tau_control.tau_badge.text() == mtui.badge_text(hi)
        wait_idle(w)
        run_s = time.perf_counter() - t_change
        timer.stop()
        assert calls[n0:] == [hi], calls[n0:]
        assert ticks[0] >= max(5, int(0.2 * run_s * 1000 / TICK_MS)), (ticks[0], run_s)
        v = variant(ms.DEFAULT_SELECTION, hi)
        lines = w.results_text().splitlines()
        assert lines[0] == mt.tau_banner(v), lines[0]
        assert mt.tau_2d_line(hi, TAU0) in lines, "the 2D line"
        arc = [ln for ln in lines if ln.startswith("   membrane length")]
        assert len(arc) == 1 and f"tau {hi:.1f} nm" in arc[0], arc
        assert w.last_run.tau_nm == hi and w.last_result is not None and not w.is_tau_pending()
        a = w.last_result
        assert a.arc_centroid.tau_nm == hi and a.columns_2d.tau0_nm == hi and a.arc_localization.tau_nm == hi
        n, n_tau = counts_now(w)
        assert (n, n_tau) == (2, 1), (n, n_tau)
        want_counter = mt.variant_counter_line(n, n_tau)
        assert w.selection_counter_label.text().endswith(" &mdash; " + want_counter), w.selection_counter_label.text()
        assert f"EXPLORATORY ANALYSIS {v.label} #{v.hash}" in w.selection_counter_label.text()
        rows = log_rows(w)
        assert len(rows) == n_log0 + 1, (len(rows), n_log0)
        last = rows[-1]
        assert last["selection_label"] == v.label and last["selection_hash"] == v.hash and last["is_default"] == "False"
        assert w.leak_banner.isVisible() and w.match_counts()["matches_arc"] > 0
        st["v_hi"] = v
        return (f"pending at once ({slot_ms:.0f} ms: no p, banner hidden, matches empty, curve kept, marker at "
                f"{hi:g}); one run at {hi:g} nm in {run_s:.1f} s, {ticks[0]} ticks; log row '{v.label}' #{v.hash}")

    check("G3 tau -> the larger preset after a default run: pending at once, one run at it, its texts, counter and "
          "log row", g3_change)

    def g4_export() -> str:
        w = need(st, "m")
        v = need(st, "v_hi")
        p0_written = need(st, "p0_written")
        assert len(p0_written) == 2
        p1 = os.path.join(WORK, "g4", "hi", "decisions.csv")
        os.makedirs(os.path.dirname(p1), exist_ok=True)
        written = w.export_tables(p1, duplicates="append")
        assert len(written) == 2, written
        row = read_csv(written[1])[0]
        assert float(row["tau0_nm"]) == v.tau_nm, row["tau0_nm"]
        assert row["analysis"] == f"{mcw._ANALYSIS_NAME} | {mt.tau_tag(v)}", row["analysis"]
        assert read_header(written[1]) == list(mcw.RESULTS_TABLE_COLUMNS), "a column was added"
        before = {p: open(p, "rb").read() for p in p0_written}
        refused = ""
        try:
            w.export_tables(p0_written[0], duplicates="append")
        except ValueError as exc:
            refused = str(exc)
        assert refused and "tau0_nm" in refused, refused
        assert {p: open(p, "rb").read() for p in p0_written} == before, "the tau_0 tables were changed"
        dec0 = rows_sha256([normalize_row(r, WORK) for r in read_csv(p0_written[0])])
        dec1 = rows_sha256([normalize_row(r, WORK) for r in read_csv(written[0])])
        assert dec0 == dec1 == golden["cases"]["marginal"]["decisions_sha256"], (dec0, dec1)
        return f"tau0_nm {row['tau0_nm']}, analysis '...{row['analysis'][-48:]}'; the tau_0 table refused, untouched"

    check("G4 export at that tau: tau0_nm and the tag; into the tau_0 table refused (untouched); decisions unchanged",
          g4_export)

    def g5_free() -> str:
        w = need(st, "m")
        hi = presets[-1]
        n0 = len(calls)
        choose(w, "free")
        assert w.tau_control.tau_spin.isEnabled() and w.tau_control.tau_spin.value() == int(round(hi))
        assert w.tau_nm == hi and not w._debounce.isActive(), "choosing Free at a whole tau must not change it"
        last = int(round(hi))
        for _ in range(3):
            w.tau_control.tau_spin.stepUp()
            last += 1
            pump(0.02)
        assert w.tau_nm == float(last) and w.results_text() == mtui.pending_live_text(last)
        wait_idle(w)
        assert calls[n0:] == [float(last)], calls[n0:]
        assert first_line(w) == mt.tau_banner(variant(ms.DEFAULT_SELECTION, float(last)))
        st["free_last"] = last
        return f"Free starts at {int(round(hi))}; three quick steps -> ONE run, at {last} nm"

    check("G5 Free + three quick spin steps -> one run, at the last value", g5_free)

    def g6_stale() -> str:
        w = need(st, "m")
        lo, hi = presets[0], presets[-1]
        slow_calls: List[Optional[float]] = []

        def slow(*a: Any, progress: Any = None, **k: Any) -> Any:
            params = k.get("columns_params")
            slow_calls.append(None if params is None else float(params.tau0_nm))
            if progress is not None:
                progress("slow start (test)", 0.05)
            time.sleep(1.5)
            if progress is not None:
                progress("slow middle (test)", 0.1)     # raises when the run was stopped
            return spy(*a, progress=progress, **k)

        mcw.run_cleaned_analyses = slow
        try:
            # (a) a LIVE run at the smaller preset, stopped by a free value before it ends
            n0 = len(calls)
            choose(w, "preset", lo)
            t_end = time.perf_counter() + 10
            while not (w.is_running() and w._run_tau == lo) and time.perf_counter() < t_end:
                pump(0.02)
            assert w.is_running() and w._run_tau == lo and w._run_live, "the live run did not start"
            free = 55
            choose_free(w, free)
            assert w._cancel_seq == w._run_seq and w._cancel_reason == "tau", "the stale run was not stopped"
            assert w.results_text() == mtui.pending_live_text(free), w.results_text()
            stale = (f"tau = {mt.tau_text(lo)} nm", f"tau {lo:.1f} nm")
            seen = {"stale": False, "stopped": False}

            def poll() -> None:
                text = w.results_text()
                if any(s in text for s in stale):
                    seen["stale"] = True
                if w.selection_note.isVisible() and w.selection_note.text() == mtui.TAU_STOPPED_NOTE:
                    seen["stopped"] = True

            wait_idle(w, 120, poll)
            assert not seen["stale"], "a result of the stopped run was shown"
            assert seen["stopped"], "the window did not say the run was stopped"
            assert slow_calls[-2:] == [lo, float(free)], slow_calls
            assert calls[n0:] == [float(free)], calls[n0:]            # the stopped run never reached the analyses
            assert first_line(w) == mt.tau_banner(variant(ms.DEFAULT_SELECTION, float(free)))
            assert w.last_run.tau_nm == float(free)
            assert all(k[1] != lo for k in w._analyses), "a stopped run's analyses were kept"
            # (b) a BUTTON run at the free value, stopped by a change to a tau computed before (shown at once)
            n1 = len(calls)
            assert w.start_run(), "start_run refused"
            pump(0.1)
            assert w.is_running() and not w._run_live
            choose(w, "preset", hi)
            assert first_line(w) == mt.tau_banner(variant(ms.DEFAULT_SELECTION, hi)), first_line(w)
            assert w._cancel_seq == w._run_seq
            stale_b = (f"tau = {free} nm", f"tau {free:.1f} nm")
            seen_b = {"stale": False}

            def poll_b() -> None:
                if any(s in w.results_text() for s in stale_b):
                    seen_b["stale"] = True

            wait_idle(w, 60, poll_b)
            assert not seen_b["stale"], "the stopped button run was shown"
            assert calls[n1:] == [] and slow_calls[-1] == float(free), (calls[n1:], slow_calls)
            assert first_line(w) == mt.tau_banner(variant(ms.DEFAULT_SELECTION, hi))
            assert w.selection_note.isVisible() and w.selection_note.text() == mtui.TAU_STOPPED_NOTE
        finally:
            mcw.run_cleaned_analyses = spy
        return (f"live run at {lo:g} stopped by {free} (never shown; {free} computed after); a button run at {free} "
                f"stopped by {hi:g} (shown at once from the cache), never shown")

    check("G6 stale runs: a live run and a button run at a tau the control left are stopped and never shown", g6_stale)

    def g7_back() -> str:
        w = need(st, "m")
        n_log = len(log_rows(w))
        n0 = len(calls)
        t_click = time.perf_counter()
        w.tau_control.tau_reset_button.click()
        slot_ms = (time.perf_counter() - t_click) * 1000.0
        assert slot_ms < 100.0, f"re-drawing tau_0 from the cache took {slot_ms:.0f} ms in the GUI thread"
        assert w.tau_nm == TAU0 and w.tau_control.tau_combo.currentIndex() == 0
        assert not w.tau_control.tau_badge.isVisible() and not w.tau_control.tau_reset_button.isEnabled()
        assert not w.is_running() and not w._debounce.isActive(), "tau_0 was computed before: no run"
        pump(0.3)
        g = golden["cases"]["marginal"]
        assert normalize_text(w.results_text(), WORK) == g["results_text"]
        assert w.simnull_button.isEnabled() == g["simnull_enabled"] and w.simnull_button.toolTip() == g["simnull_tooltip"]
        assert not w.curve_tau_line.isVisible()
        p2 = os.path.join(WORK, "g7", "decisions.csv")
        os.makedirs(os.path.dirname(p2), exist_ok=True)
        written = w.export_tables(p2, duplicates="append")
        assert read_header(written[1]) == g["results_header"]
        assert normalize_row(read_csv(written[1])[0], WORK) == g["results_row"]
        assert len(log_rows(w)) == n_log, "going back to tau_0 logged a row"
        assert calls[n0:] == []
        n, n_tau = counts_now(w)
        assert n_tau >= 1 and w.selection_counter_label.text() == mt.variant_counter_line(n, n_tau)
        assert "analysis variants (selection and tau)" in w.selection_counter_label.text()
        return (f"re-drawn in {slot_ms:.0f} ms: the golden's text and export row; no run, no new log row; counter: "
                f"{n} variants ({n_tau} at a tau)")

    check("G7 'Back to tau_0': badge off, simnull back, the golden's text and export row, no new log row, counter "
          "keeps the history", g7_back)

    def g8_selection() -> str:
        w = need(st, "m")
        hi = presets[-1]
        state.set_label("v2[]")
        pump(0.1)
        spec = state.spec
        assert spec.label == "v2[]" and not spec.is_default
        assert first_line(w) == su.exploratory_banner(spec), first_line(w)
        choose(w, "preset", hi)
        pump(0.1)
        assert not w.is_running() and not w._debounce.isActive(), "both were computed before: no run"
        v = variant(spec, hi)
        assert v.label == f"v2[]{mt.TAU_LABEL_MARK}{mt.tau_text(hi)} nm"
        line = first_line(w)
        assert line == mt.tau_banner(v) and f"the selection is not the pre-specified rule {ms.DEFAULT_SELECTION.label}" \
            in line, line
        assert f"EXPLORATORY ANALYSIS {v.label} #{v.hash}" in w.selection_counter_label.text()
        assert any(r["selection_label"] == v.label and r["selection_hash"] == v.hash for r in log_rows(w))
        state.reset()
        w.tau_control.reset()
        pump(0.1)
        assert normalize_text(w.results_text(), WORK) == golden["cases"]["marginal"]["results_text"]
        assert "EXPLORATORY" not in w.selection_counter_label.text()
        return f"label '{v.label}' #{v.hash}; the first line names the tau and the selection; both reset: the golden"

    check("G8 selection v2[] AND another tau: one label, a first line naming both; both reset -> the pre-registered "
          "texts", g8_selection)

    # ------------------------------------------------------------------ G11, G12 on the same window
    print("\nG11-G12. The E(tau) curve and the matches on the map")

    def g11_curve() -> str:
        w = need(st, "m")
        v = w.viability
        want = mtui.curve_pairs(v.viable, v.marginal)
        combo = w.curve_pair_combo
        got = [(combo.itemText(i), tuple(combo.itemData(i))) for i in range(combo.count())]
        assert got == [(t, p) for t, p in want] and v.viable and v.marginal, got
        assert combo.currentIndex() == 0 and tuple(combo.itemData(0)) == tuple(v.viable[0])
        a = w.last_result
        pair = (int(v.viable[0][0]), int(v.viable[0][1]))
        cv = mt.curve_view(a, pair)
        assert cv is not None
        assert w.curve_p_label.text() == mtui.curve_p_text(cv)
        assert "not calibrated" in w.curve_p_label.text() and "global p of the curve" in w.curve_p_label.text()
        assert w.curve_counter_label.isVisible() and w.curve_counter_label.text() == \
            mt.variant_counter_line(*counts_now(w))
        assert w.curve_note.isVisible()
        xo, yo = w.curve_observed_item.getData()
        assert np.array_equal(np.asarray(xo, float), cv.tau_grid_nm) and np.array_equal(np.asarray(yo, float),
                                                                                        cv.E_dir_obs)
        assert np.array_equal(np.asarray(w.curve_mean_item.getData()[1], float), cv.E_star)
        assert np.array_equal(np.asarray(w.curve_band_lo.getData()[1], float), cv.envelope_lo)
        assert np.array_equal(np.asarray(w.curve_band_hi.getData()[1], float), cv.envelope_hi)
        # layer rows drive the real items and survive a redraw (another pair)
        layers = w.curve_layers
        assert layers.keys() == list(mtui.CURVE_LAYER_LABELS), layers.keys()
        for key, item in (("observed", w.curve_observed_item), ("null_band", w.curve_band_item),
                          ("null_mean", w.curve_mean_item), ("tau0", w.curve_tau0_line)):
            layers.checkbox(key).setChecked(False)
            assert not item.isVisible(), key
        combo.setCurrentIndex(1)
        pump(0.05)
        other = (int(v.marginal[0][0]), int(v.marginal[0][1]))
        cv2 = mt.curve_view(a, other)
        assert cv2 is not None and "(MARGINAL: sensitivity only)" in w.curve_p_label.text()
        assert np.array_equal(np.asarray(w.curve_observed_item.getData()[1], float), cv2.E_dir_obs)
        assert not w.curve_observed_item.isVisible() and not w.curve_band_item.isVisible(), "a redraw showed a row"
        layers.show_all()
        assert all(it.isVisible() for it in (w.curve_observed_item, w.curve_band_item, w.curve_mean_item,
                                             w.curve_tau0_line))
        # the chosen tau's line: hidden at tau_0 whatever its row says; drawn at another tau, hidden by its row
        assert layers.checkbox("tau").isChecked() and not w.curve_tau_line.isVisible()
        hi = presets[-1]
        choose(w, "preset", hi)
        pump(0.05)
        assert w.curve_tau_line.isVisible() and abs(w.curve_tau_line.value() - hi) < 1e-9
        assert np.array_equal(np.asarray(w.curve_observed_item.getData()[1], float), cv2.E_dir_obs), "tau moved it"
        layers.checkbox("tau").setChecked(False)
        assert not w.curve_tau_line.isVisible()
        layers.checkbox("tau").setChecked(True)
        assert w.curve_tau_line.isVisible()
        xr = w.curve_plot.getViewBox().viewRange()[0]
        grid = [float(t) for t in cp.tau_grid_nm]
        assert xr[0] <= min(grid) and xr[1] >= max(grid) and xr[0] <= TAU0 <= xr[1], xr
        w.tau_control.reset()
        combo.setCurrentIndex(0)
        pump(0.05)
        # beside the plot, never over it; no legend inside
        assert w.curve_plot.getPlotItem().legend is None
        assert layers.parent() is w.curve_plot.parent()
        assert not w.curve_plot.geometry().intersects(layers.geometry())
        return (f"pairs {[t for t, _p in got]}; p label from the 2D test's curve; rows drive real items through a "
                "redraw; marker hidden at tau_0")

    check("G11 E(tau) curve: the selected pairs, its p 'not calibrated', the note, the arrays of the 2D test, rows "
          "that drive real items", g11_curve)

    def g12_matches() -> str:
        w = need(st, "m")
        panel = w.layer_panel
        keys = list(mcw.DISPLAY_CLASSES) + ["membrane_before", "membrane_after", "underlay", "matches_arc", "matches_2d"]
        assert panel.keys() == keys, panel.keys()
        assert panel.checkbox("matches_arc").isChecked() and not panel.checkbox("matches_2d").isChecked()
        assert w.match_items["matches_arc"].isVisible() and not w.match_items["matches_2d"].isVisible()
        for item in w.match_items.values():
            assert int(item.acceptedMouseButtons()) == 0 and int(item.curve.acceptedMouseButtons()) == 0
            assert -4 < item.zValue() < min(m.zValue() for m in w.marker_items.values())

        def expect(tau_label: str) -> Dict[str, int]:
            a = w.last_result
            pair = w._curve_pair()
            out = {}
            for key, test, res in (("matches_arc", "arc", a.arc_centroid), ("matches_2d", "2d", a.columns_2d)):
                m = next(q for q in res.adjacent if (int(q.ring_a), int(q.ring_b)) == pair)
                n = int(m.n_matched)
                seg = mt.match_segments(a, pair, test)
                x, y = w.match_items[key].getData()
                xy = np.zeros((0, 2)) if x is None else np.column_stack([np.asarray(x, float), np.asarray(y, float)])
                assert np.array_equal(xy, seg.reshape(-1, 2)), (tau_label, key)
                assert panel.checkbox(key).text().endswith(f"({n})") and panel.checkbox(key).isEnabled(), \
                    panel.checkbox(key).text()
                out[key] = n
            assert w.match_counts() == out
            return out

        at0 = expect("tau_0")
        w.curve_pair_combo.setCurrentIndex(1)
        pump(0.05)
        at0_other = expect("tau_0, the other pair")
        hi = presets[-1]
        choose(w, "preset", hi)
        pump(0.05)
        at_hi = expect("the larger preset")
        assert all(at_hi[k] >= at0_other[k] for k in at_hi), (at0_other, at_hi)
        w.tau_control.reset()
        w.curve_pair_combo.setCurrentIndex(0)
        pump(0.05)
        assert expect("tau_0 again") == at0
        return f"rows after the stage-0 ones; counts {at0} / other pair {at0_other} / at {hi:g} nm {at_hi}"

    check("G12 map matches: two rows after the stage-0 rows, counts = n matched at the current tau, follow the pair "
          "combo, never clickable", g12_matches)

    def g14_live_off() -> str:
        w = need(st, "m")
        w.live_update_check.setChecked(False)
        pump(0.05)
        n0 = len(calls)
        free = 57
        choose_free(w, free)
        pump(0.8)
        assert len(calls) == n0 and not w.is_running() and not w._debounce.isActive(), "a run started"
        assert w.results_text() == mtui.pending_press_text(free), w.results_text()
        assert w.is_tau_pending() and not w.leak_banner.isVisible() and w.match_counts() == {"matches_arc": 0,
                                                                                             "matches_2d": 0}
        assert w.export_note() == "\n\n" + mt.pending_export_message(free)
        p = os.path.join(WORK, "g14", "decisions.csv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        written = w.export_tables(p, duplicates="append")
        assert written == [os.path.abspath(p)], written
        w.live_update_check.setChecked(True)
        assert w.results_text() == mtui.pending_live_text(free)
        wait_idle(w)
        assert calls[n0:] == [float(free)], calls[n0:]
        assert first_line(w) == mt.tau_banner(variant(ms.DEFAULT_SELECTION, float(free)))
        w.tau_control.reset()
        pump(0.05)
        return f"'press the button' at {free} nm, no run, decisions only; Live update on again: one run"

    check("G14 'Live update' off: the window says to press the button, runs nothing, exports the decisions only",
          g14_live_off)

    # ------------------------------------------------------------------ G9, G10, G13, G15: other windows
    print("\nG9-G15. Before any run, no viable pair, a new window, a custom viability function")

    def g9_before_run() -> str:
        launcher, w = open_review("viable")
        st["launcher_v"], st["v"] = launcher, w
        lo = presets[0]
        n0 = len(calls)
        choose(w, "preset", lo)
        pump(0.8)
        assert len(calls) == n0 and not w.is_running() and not w._debounce.isActive(), "a change of tau ran"
        assert w.results_text() == "" and not w.is_tau_pending()
        assert w.tau_control.tau_badge.isVisible() and not w.simnull_button.isEnabled()
        assert w.simnull_button.toolTip() == mtui.SIMNULL_TAU_TIP
        assert w.curve_p_label.text() == mtui.CURVE_NO_RUN_TEXT
        run_default(w)
        assert calls[n0:] == [lo], calls[n0:]
        assert first_line(w) == mt.tau_banner(variant(ms.DEFAULT_SELECTION, lo))
        assert w.last_run.tau_nm == lo
        # back to tau_0, never computed in this window: one run, with the window's own parameters, and the golden's
        # results text (the pre-registered analysis after an excursion)
        w.tau_control.reset()
        assert w.is_tau_pending() and w.results_text() == mtui.pending_live_text(TAU0)
        wait_idle(w)
        assert calls[n0 + 1:] == [None], calls[n0:]
        assert normalize_text(w.results_text(), WORK) == golden["cases"]["viable"]["results_text"]
        return f"no run before the first one; the button then ran at {lo:g} nm; back at tau_0: the golden's text"

    check("G9 before any run a change of tau runs nothing; the button then runs at it", g9_before_run)

    def g10_none() -> str:
        launcher, w = open_review("none")
        run_default(w)
        g = golden["cases"]["none"]
        before = w.results_text()
        assert normalize_text(before, WORK) == g["results_text"]
        n0 = len(calls)
        choose(w, "preset", presets[-1])
        pump(0.8)
        assert len(calls) == n0 and not w.is_running() and not w._debounce.isActive()
        assert w.results_text() == before and not w.selection_counter_label.isVisible()
        assert w.curve_p_label.text() == mtui.CURVE_NO_PAIR_TEXT
        assert w.match_counts() == {"matches_arc": 0, "matches_2d": 0}
        w.tau_control.reset()
        pump(0.05)
        assert w.simnull_button.toolTip() == g["simnull_tooltip"] and not w.simnull_button.isEnabled()
        w.close()
        launcher.close()
        pump(0.1)
        return "no pair: a change of tau computes nothing and changes no text; back at tau_0 the simnull tip is the old"

    check("G10 demo 'none': a change of tau runs nothing, the text stays the default one", g10_none)

    def g13_new_window() -> str:
        store = L.LumenReviewStore(os.path.join(WORK, "store_tau"))
        launcher, w = open_review("marginal", store)
        run_default(w)
        choose(w, "preset", presets[-1])
        wait_idle(w)
        assert w.last_run.tau_nm == presets[-1]
        path = store.path_for(w.axon_id)
        w.close()
        launcher.close()
        pump(0.2)
        launcher2, w2 = open_review("marginal", store)
        assert w2.tau_nm == TAU0 and w2.tau_control.is_tau0() and w2.tau_control.tau_combo.currentIndex() == 0
        assert not w2.tau_control.tau_badge.isVisible()
        w2.close()
        launcher2.close()
        store_d = L.LumenReviewStore(os.path.join(WORK, "store_default"))
        launcher3, w3 = open_review("marginal", store_d)
        run_default(w3)
        path_d = store_d.path_for(w3.axon_id)
        w3.close()
        launcher3.close()
        pump(0.2)

        def keys_of(obj: Any, prefix: str = "") -> List[str]:
            out: List[str] = []
            if isinstance(obj, dict):
                for k, val in obj.items():
                    out.append(prefix + str(k))
                    out.extend(keys_of(val, prefix + str(k) + "."))
            elif isinstance(obj, list):
                for val in obj[:1]:
                    out.extend(keys_of(val, prefix + "[]."))
            return out

        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        with open(path_d, encoding="utf-8") as fh:
            text_d = fh.read()
        k1, k2 = sorted(set(keys_of(json.loads(text)))), sorted(set(keys_of(json.loads(text_d))))
        assert k1 == k2, set(k1) ^ set(k2)
        assert mt.TAU_LABEL_MARK not in text and "tau_nm" not in text
        return f"a new window of the same axon starts at tau_0; the store JSON has the default's {len(k1)} keys"

    check("G13 a new window starts at tau_0; the review store's JSON key set == a default session's", g13_new_window)

    def g15_custom() -> str:
        import batch_columns as bc2

        def all_pairs_viable(res: Any, xyz_lab_nm: Any = None) -> Any:
            rings = sorted(int(r.index) for r in res.rings)
            return mcw.ReviewViability(rule="test stub: every pair viable",
                                       pairs=tuple(bc2.PairVerdict(a, b, "viable") for a, b in zip(rings, rings[1:])))

        mcw.DEFAULT_VIABILITY_FN = all_pairs_viable
        try:
            launcher, w = open_review("marginal")
            assert w.criteria is None and not w.selection_widget.isEnabled(), "the switches should be off"
            run_default(w)
            assert w.last_run.ran and not first_line(w).startswith("EXPLORATORY")
            n0 = len(calls)
            hi = presets[-1]
            choose(w, "preset", hi)
            assert w.is_tau_pending() and w.results_text() == mtui.pending_live_text(hi)
            wait_idle(w)
            assert calls[n0:] == [hi], calls[n0:]
            assert first_line(w) == mt.tau_banner(variant(ms.DEFAULT_SELECTION, hi))
            w.tau_control.reset()
            pump(0.1)
            assert not first_line(w).startswith("EXPLORATORY") and not w.is_running()
            w.close()
            launcher.close()
        finally:
            mcw.DEFAULT_VIABILITY_FN = orig_viability_fn
        return "switches off (custom viability function); tau re-ran and came back"

    check("G15 a custom viability function (switches off): tau still re-runs", g15_custom)

    # ------------------------------------------------------------------ G16-G19: found by the review of stage 1
    print("\nG16-G19. Review of stage 1: the warnings list, the D-35b flag, the stop reason, no criteria")

    def listed(w: Any) -> List[str]:
        return [w.warnings_list.item(i).text() for i in range(w.warnings_list.count())]

    def g16_warnings() -> str:
        launcher, w = open_review("viable")
        try:
            run_default(w)
            at0 = listed(w)
            assert any("run: " in x for x in at0), at0
            lo = int(mt.TAU_FREE_MIN_NM)
            choose_free(w, lo)
            wait_idle(w)
            at_lo = listed(w)
            # this demo's notes at the smallest free tau are not tau_0's: otherwise the check below proves nothing
            assert at_lo != at0, "the notes do not depend on tau here"
            # back to tau_0 from the cache: the list is tau_0's again, item for item
            w.tau_control.reset()
            pump(0.1)
            assert not w.is_running() and not w.is_tau_pending() and w.last_run.tau_nm == TAU0
            assert listed(w) == at0, [x[:90] for x in listed(w) if x not in at0][:2]
            # a tau not computed yet, Live update off: no note of another tau's run stays while its results are pending
            w.live_update_check.setChecked(False)
            choose(w, "preset", presets[0])
            pump(0.1)
            assert w.is_tau_pending() and not w.is_running()
            stale = [x for x in listed(w) if "run: " in x]
            assert not stale, [x[:90] for x in stale][:2]
            w.live_update_check.setChecked(True)
            wait_idle(w)
            run_notes = [x for x in listed(w) if "run: " in x]
            assert [x.split("run: ", 1)[1] for x in run_notes] == list(w.last_result.warnings), run_notes[:2]
            w.tau_control.reset()
            pump(0.1)
            assert listed(w) == at0
        finally:
            w.close()
            launcher.close()
            pump(0.1)
        return (f"{len(at0)} notes at tau_0, {len(at_lo)} at {lo} nm; back from the cache: tau_0's list; pending: no "
                "'run:' note of another tau")

    check("G16 the warnings list follows the analyses on screen: back to tau_0 from the cache, and while pending",
          g16_warnings)

    def gated_run(w: Any, change: Callable[[], None]) -> None:
        """One button run held after its last step (it can no longer be stopped) while ``change`` runs, then
        released."""
        import threading
        reached, gate = threading.Event(), threading.Event()
        inner = mcw.run_cleaned_analyses

        def gated(*a: Any, **k: Any) -> Any:
            result = inner(*a, **k)
            reached.set()
            gate.wait(120)
            return result

        mcw.run_cleaned_analyses = gated
        try:
            assert w.start_run(), "start_run refused"
            t_end = time.perf_counter() + 300
            while not reached.is_set() and time.perf_counter() < t_end:
                pump(0.02)
            assert reached.is_set(), "the run never reached its end"
            change()
            gate.set()
            wait_idle(w)
        finally:
            gate.set()
            mcw.run_cleaned_analyses = inner

    def g17_flag() -> str:
        store = L.LumenReviewStore(os.path.join(WORK, "store_g17"))
        launcher, w = open_review("marginal", store)
        st["launcher_g17"], st["g17"] = launcher, w
        w.live_update_check.setChecked(False)
        assert not w.decisions.results_shown
        hi = presets[-1]
        gated_run(w, lambda: choose(w, "preset", hi))
        # the run ended at tau_0, which the control had left: kept, never shown
        assert w.is_tau_pending() and w.last_result is None and not w.decisions.results_shown
        w.tau_control.reset()                      # tau_0 from the cache: a column result is on screen now
        pump(0.1)
        assert not w.is_running() and not w.is_tau_pending() and w.last_result is not None
        assert normalize_text(w.results_text(), WORK) == golden["cases"]["marginal"]["results_text"]
        assert w.decisions.results_shown, "a column result is on screen but the axon is not marked (D-35b)"
        with open(store.path_for(w.axon_id), encoding="utf-8") as fh:
            assert json.load(fh)["results_shown"] is True
        key = w.decisions.stable_keys[0]
        w.decisions.toggle(key, via="click")
        w._after_edit()
        assert w.decisions.results_shown_before_edit and w.edited_label.isVisible()
        with open(store.path_for(w.axon_id), encoding="utf-8") as fh:
            assert json.load(fh)["results_shown_before_edit"] is True
        return "a run kept for tau_0 and shown from the cache marks the axon; the next edit is flagged and saved"

    check("G17 analyses shown from the cache (never shown when their run ended) mark the axon (D-35b)", g17_flag)

    def g18_reason() -> str:
        w = need(st, "g17")
        hi = presets[-1]

        def there_and_back() -> None:
            choose(w, "preset", hi)                # asks the run to stop (too late: it is past its last step)
            assert w._cancel_reason == "tau"
            w.tau_control.reset()                  # the run's tau is the current one again

        gated_run(w, there_and_back)
        assert w.last_run.tau_nm == TAU0 and w.last_result is not None and not w.is_tau_pending()
        assert w._cancel_reason == "", f"the stop request outlived its run: {w._cancel_reason!r}"
        assert w.selection_note.text() != mtui.TAU_STOPPED_NOTE or not w.selection_note.isVisible()
        w.close()
        st["launcher_g17"].close()
        pump(0.1)
        return "a run asked to stop that ended normally leaves no stop reason for the next run"

    check("G18 a stop request never outlives its run (the next stopped run is reported for its own reason)",
          g18_reason)

    def g19_no_criteria() -> str:
        import batch_columns as bc3

        def all_pairs_viable(res: Any, xyz_lab_nm: Any = None) -> Any:
            rings = sorted(int(r.index) for r in res.rings)
            return mcw.ReviewViability(rule="test stub: every pair viable",
                                       pairs=tuple(bc3.PairVerdict(a, b, "viable") for a, b in zip(rings, rings[1:])))

        log_dir = os.environ["MPS_SELECTION_LOG_DIR"]
        os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log_g19")
        mcw.DEFAULT_VIABILITY_FN = all_pairs_viable
        state.set_label("v2[]")          # the program's selection is exploratory; this window cannot apply it
        try:
            launcher, w = open_review("marginal")
            assert w.criteria is None, "the switches should be off"
            run_default(w)
            # as before this stage: without the criteria nothing is logged or counted at tau_0
            assert log_rows(w) == [] and not w.selection_counter_label.isVisible()
            hi = presets[-1]
            choose(w, "preset", hi)
            wait_idle(w)
            v = variant(ms.DEFAULT_SELECTION, hi)
            assert first_line(w) == mt.tau_banner(v), first_line(w)
            rows = log_rows(w)
            assert [(r["selection_label"], r["selection_hash"], r["is_default"]) for r in rows] == [
                (ms.DEFAULT_SELECTION.label, ms.DEFAULT_SELECTION.hash, "True"), (v.label, v.hash, "False")], rows
            assert counts_now(w) == (2, 1)
            want = mt.variant_counter_line(2, 1)
            assert w.selection_counter_label.isVisible() and w.selection_counter_label.text().endswith(want), \
                w.selection_counter_label.text()
            assert w.curve_counter_label.text() == want
            w.tau_control.reset()
            pump(0.1)
            assert not first_line(w).startswith("EXPLORATORY") and len(log_rows(w)) == 2
            assert w.selection_counter_label.text() == want
            w.close()
            launcher.close()
        finally:
            mcw.DEFAULT_VIABILITY_FN = orig_viability_fn
            os.environ["MPS_SELECTION_LOG_DIR"] = log_dir
            state.reset()
        return "switches off: the p shown at another tau is logged and counted, with the tau_0 result before it"

    check("G19 no criteria (switches off): a p at another tau is logged and counted, under rule v2's own label",
          g19_no_criteria)

    # ------------------------------------------------------------------ the end
    state.reset()
    for key in ("m", "v"):
        if key in st:
            try:
                st[key].close()
            except Exception:  # noqa: BLE001
                pass
    mcw.run_cleaned_analyses = orig_run
    mcw.DEFAULT_VIABILITY_FN = orig_viability_fn
    pump(0.2)
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.0f} s; work folder {WORK}")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
