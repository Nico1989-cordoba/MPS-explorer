# -*- coding: utf-8 -*-
"""
Offscreen test of UI stage 0, items 0.3 and 0.4 (the user's manual test of 2026-10-05):

What the user reported: he did not understand what "Simulated-null p (not calibrated)..." is for; he asked for 4
accepted simulated axons, the run printed progress tags [1/4] ... [5/3] ... [12/2] (counted per round of redraws), 2
of the 4 were accepted, most fidelity checks were NOT met, and the dialog showed a p from 2 simulated axons as if it
could be read. And two Z quality views, two batch dialogs and two viability explorers could be open for one axon.

  B. The simulated-null dialog: the plain explanation (<= 90 words) under the NOT CALIBRATED warning; the small-n
     warning under the spin (bold below 19, neutral below 99, hidden from 99) and the confirmation the Start button
     asks for (start() itself never asks); the worst-case time; the progress tag of power_columns
     ("[attempt a of m; accepted k of n]", over the whole run) and the dialog's reading of it (status line and
     progress bar); SimnullResult.warnings / fidelity summary and the "NOT INTERPRETABLE" block ABOVE the p; a failed
     run hides it; a tiny REAL run on a SIMULATED axon whose selection rejects simulated axons ends with "Only k of
     the n" and the fidelity line on top; the button inside a collapsed "Advanced (experimental)" section.
  C. One window each: the review's "Z quality..." and "Columns batch...", the batch dialog's "Viability explorer..."
     and the main window's own openers give ONE window (brought forward); a Z quality view of another axon replaces
     the old one; closing and re-opening works; windows run on their own still open their own copies.

R8: column statistics only on SIMULATED axons (tools.mps_zquality_window.DEMO_CASES through
batch_columns.write_simulated_input); every file goes to a temporary folder (MPS_SELECTION_LOG_DIR, the review store
and the settings included).

Run:  venv\\Scripts\\python.exe test_simnull_ux_gui.py      (offscreen; about 3-4 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
T0 = time.perf_counter()
WORK = tempfile.mkdtemp(prefix="test_simnull_ux_gui_")
# the exploration log of this test, never the user's (tools.mps_selection.default_log_path)
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")
N_NULL = 19


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


def pump(app: Any, seconds: float = 0.05) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


def fidelity_entries(n_checks: int, n_leak: int, not_met: int) -> List[Dict[str, Any]]:
    """Synthetic simnull_fidelity entries: ``not_met`` of the D-29c checks NOT met, the rest met."""
    out: List[Dict[str, Any]] = []
    for i in range(n_checks):
        out.append({"name": f"check {i}", "observed": 1.0, "simulated": 1.0, "met": i >= not_met})
    for i in range(n_leak):
        out.append({"name": f"leak {i}", "observed": 0.0, "simulated": 0.0, "kind": "central", "met": True})
    return out


def main() -> int:
    print("=" * 100)
    print("SIMULATED NULL EXPLAINED AND GUARDED, ONE WINDOW PER AXON (UI stage 0, 0.3 + 0.4) on SIMULATED axons")
    print("=" * 100)
    from PyQt5 import QtCore, QtTest, QtWidgets
    qtest: Any = QtTest.QTest
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import batch_columns as bc
    import power_columns as pc
    from tools import mps_columns_window as mcw
    from tools import mps_simnull_ui as su
    from tools import mps_zquality_window as zw
    st: Dict[str, Any] = {}

    print("\n0. The simulated axons")

    def simulate() -> str:
        from tools import mps_sim_harness as vl
        from tools.mps_simulate_axon import scale_axial_leak, write_sim_config
        out = []
        for name in ("viable", "marginal"):
            contour, seed, s = zw.DEMO_CASES[name]
            p = bc.write_simulated_input(os.path.join(WORK, f"sim_{name}.npz"), contour, seed, s)
            assert bc.input_kind(p)[0] == "simulated"
            st[f"npz_{name}"] = p
            st[f"inp_{name}"] = zw.inputs_from_npz(p)
            spec = vl.case_spec(contour)
            cfg = os.path.join(WORK, f"truth_{name}.yaml")
            write_sim_config(cfg, scale_axial_leak(vl.sim_config(spec["contour"], spec["overrides"], spec["n_rings"],
                                                                 True, spec["box"]), s))
            st[f"cfg_{name}"] = cfg
            out.append(f"{name}: {st[f'inp_{name}'].x_nm.size:,} locs")
        return "; ".join(out)

    check("two SIMULATED axons (DEMO_CASES viable, marginal) and their truth configurations", simulate)

    # ------------------------------------------------------------------ B5 the advanced section
    print("\n1. The review window: 'Advanced (experimental)'")

    def advanced() -> str:
        launcher = mcw.start_columns_review(need(st, "inp_viable"), n_null=N_NULL)
        launcher.wait(600)
        w = launcher.window
        assert w is not None, launcher.error
        w.wait_viability(120)
        pump(app, 0.2)
        st["review_viable"] = w
        assert w.zquality_callback is None and w.batch_callback is None, "a window run on its own has no callbacks"
        assert not w.advanced_toggle.isChecked() and w.advanced_box.isHidden(), "the section is open by default"
        assert w.advanced_box.isAncestorOf(w.simnull_button) and not w.simnull_button.isVisible()
        assert w.advanced_box.isAncestorOf(w.advanced_note) and "Experimental" in w.advanced_note.text()
        assert w.advanced_toggle.text() == "Advanced (experimental)" and w.advanced_toggle.toolTip()
        assert w.advanced_toggle.arrowType() == QtCore.Qt.ArrowType.RightArrow
        assert w.simnull_button.isEnabled(), w.simnull_button.toolTip()
        assert "not calibrated" in w.simnull_button.text() and w.simnull_button.toolTip() == mcw.SIMNULL_BUTTON_TIP
        assert not [x for x in w.extra_warnings if 'no widget for the tooltips' in x], w.extra_warnings
        qtest.mouseClick(w.advanced_toggle, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.1)
        assert w.advanced_toggle.isChecked() and w.simnull_button.isVisible()
        assert w.advanced_toggle.arrowType() == QtCore.Qt.ArrowType.DownArrow
        qtest.mouseClick(w.simnull_button, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.2)
        d = w.simnull_dialog
        assert isinstance(d, su.SimnullDialog) and d.isVisible()
        st["dialog"] = d
        qtest.mouseClick(w.simnull_button, QtCore.Qt.MouseButton.LeftButton)
        assert w.simnull_dialog is d, "a second click opened a second dialog"
        qtest.mouseClick(w.advanced_toggle, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.1)
        assert not w.simnull_button.isVisible() and w.advanced_box.isHidden()
        return "collapsed by default; opening it shows the button, which opens the same dialog"

    check("the simnull button sits in a collapsed 'Advanced (experimental)' section and opens the dialog", advanced)

    # ------------------------------------------------------------------ B1 B2 the dialog before a run
    print("\n2. The dialog before a run")

    def explanation() -> str:
        d = need(st, "dialog")
        words = len(su.SIMNULL_EXPLANATION.split())
        assert words <= 90, words
        assert su.SIMNULL_EXPLANATION in d.explain_label.text() and d.explain_label.isVisible()
        for must in ("axial leak", "too optimistic", "without columns", "not accepted", "resemble"):
            assert must in su.SIMNULL_EXPLANATION, must
        lay = d.layout()
        inner = d.settings_scroll.widget().layout()
        idx = [inner.indexOf(wd) for wd in (d.explain_label, d.info_label)]
        assert idx == [0, 1] and lay.indexOf(d.settings_scroll) == 1, (idx, lay.indexOf(d.settings_scroll))
        # right under the NOT CALIBRATED warning (the first widget), at the top of the settings that scroll
        assert "NOT CALIBRATED" in lay.itemAt(0).widget().text()
        assert d.log_view.isHidden(), "an empty log takes the room before a run"
        assert not d.missing_tooltips, d.missing_tooltips
        return f"{words} words, right under the warning"

    check("B1: the explanation is shown in the dialog (<= 90 words)", explanation)

    def small_n() -> str:
        d = need(st, "dialog")
        assert d.replicates_spin.value() == su.DEFAULT_REPLICATES == 99
        assert d.small_n_label.isHidden(), "a note at 99"
        d.replicates_spin.setValue(4)
        pump(app, 0.05)
        t4 = d.small_n_label.text()
        assert not d.small_n_label.isHidden() and "never reach 0.05" in t4 and "1/(4+1) = 0.200" in t4, t4
        assert "font-weight: bold" in d.small_n_label.styleSheet()
        est = d.estimate_label.text()
        assert "up to" in est and "12 attempts" in est and "rejected" in est, est
        d.replicates_spin.setValue(50)
        pump(app, 0.05)
        t50 = d.small_n_label.text()
        assert not d.small_n_label.isHidden() and "1/(50+1) = 0.020" in t50 and "never" not in t50, t50
        assert "bold" not in d.small_n_label.styleSheet()
        d.replicates_spin.setValue(99)
        pump(app, 0.05)
        assert d.small_n_label.isHidden()
        return f"at 4: {t4[:60]}...; at 50: {t50}; at 99 hidden"

    check("B2: the small-n note follows the spin (bold < 19, neutral < 99, hidden at 99), worst-case time", small_n)

    def confirm() -> str:
        d = need(st, "dialog")
        asked: List[int] = []
        started: List[bool] = []
        real_start = d.start
        answer = {"yes": False}

        def fake_start() -> bool:
            started.append(True)
            return False

        def fake_confirm(r: int) -> bool:
            asked.append(r)
            return answer["yes"]

        setattr(d, "start", fake_start)
        try:
            d.confirm_small = fake_confirm
            d.replicates_spin.setValue(4)
            qtest.mouseClick(d.start_button, QtCore.Qt.MouseButton.LeftButton)
            assert asked == [4] and not started and d.run is None, (asked, started)
            assert "Not started" in d.status_label.text()
            answer["yes"] = True
            qtest.mouseClick(d.start_button, QtCore.Qt.MouseButton.LeftButton)
            assert asked == [4, 4] and started == [True]
            d.replicates_spin.setValue(19)
            qtest.mouseClick(d.start_button, QtCore.Qt.MouseButton.LeftButton)
            assert asked == [4, 4] and started == [True, True], "asked at 19"
        finally:
            setattr(d, "start", real_start)
            d.replicates_spin.setValue(99)
        return "R=4: asked, No -> nothing starts, Yes -> starts; R=19: not asked"

    check("B2: Start asks before a run of fewer than 19 accepted axons; start() itself never asks", confirm)

    # ------------------------------------------------------------------ B3 the progress tag
    print("\n3. The progress tag")

    def tags() -> str:
        for a, m, k, n in ((1, 12, 0, 4), (5, 12, 2, 4), (12, 12, 2, 4), (297, 297, 99, 99)):
            t = pc.simnull_progress_tag(a, m, k, n)
            assert t == f"[attempt {a} of {m}; accepted {k} of {n}]", t
            assert su.parse_progress_tag([f"  replicate 3 seed 1: 9 s ..., spurious 2  {t}"]) == (a, m, k, n)
        assert pc.simnull_progress_tag(7, 12, 9, 4) == "[attempt 7 of 12; accepted 4 of 4]", "k shown above n"
        assert pc.simnull_progress_tag(3, None, 0, 5) == "[3 of 5]"
        assert su.parse_progress_tag(["  replicate 0 seed 1: ... [5/3]", "[3 of 5]"]) is None
        lines = [f"  replicate {i} seed 1: x  {pc.simnull_progress_tag(i + 1, 8, i // 2, 4)}" for i in range(6)]
        assert su.parse_progress_tag(lines) == (6, 8, 2, 4), "not the last tag"
        assert su._REPLICATE_LINE.match(lines[0])
        return "round-trip through _PROGRESS_TAG; the last tag wins; old tags are not read"

    check("B3: the tag formatter round-trips through the dialog's reader", tags)

    def poll_reads_tag() -> str:
        from tools.mps_background_process import BackgroundRun
        d = need(st, "dialog")
        folder = os.path.join(WORK, "poll_tag")
        os.makedirs(folder, exist_ok=True)
        log = os.path.join(folder, "fake.log")
        script = os.path.join(folder, "fake_runner.py")
        lines = ["power_columns simnull: fake", "  configuration: truth.yaml (--config-from)"] + [
            f"  replicate {i} seed 1: 9 s, spurious 2  {pc.simnull_progress_tag(i + 1, 12, min(2, i // 2), 4)}"
            for i in range(7)]
        with open(script, "w", encoding="utf-8") as fh:
            fh.write("import sys, time\n" + "".join(f"print({ln!r}, flush=True)\n" for ln in lines)
                     + "time.sleep(30)\n")
        d.out_dir, d.name = folder, "fake"
        d.run = BackgroundRun([sys.executable, "-u", script], log)
        d.started_at = time.perf_counter()
        d.replicates_spin.setValue(4)
        try:
            t_end = time.perf_counter() + 20
            while time.perf_counter() < t_end:
                pump(app, 0.2)
                d.poll()
                if "attempt 7" in d.status_label.text():
                    break
            text = d.status_label.text()
            assert "attempt 7 of up to 12, 2 of 4 accepted" in text, text
            assert d.progress_bar.value() == 50 and d.progress_bar.format() == "2 of 4 accepted", (
                d.progress_bar.value(), d.progress_bar.format())
            assert "simulated axons" in text
        finally:
            d.run.cancel()
            t_end = time.perf_counter() + 15
            while d.run.poll() is None and time.perf_counter() < t_end:
                pump(app, 0.1)
            d.timer.stop()
            d.run = None
            d.replicates_spin.setValue(99)
            d.cancel_button.setEnabled(False)
            d.start_button.setEnabled(True)
        return str(text)

    check("B3: the dialog reads the last tag: 'attempt a of up to m, k of n accepted', the bar k/n", poll_reads_tag)

    # ------------------------------------------------------------------ B4 the result
    print("\n4. The result: NOT INTERPRETABLE above the p")

    def warnings_pure() -> str:
        bad = su.SimnullResult(mode="v2-viable", key="3|0-1", observed=1.0, n_accepted=2, n_rows=12, p=0.333,
                               message="m", n_requested=4, fidelity=fidelity_entries(5, 4, 3))
        w = bad.warnings()
        assert not bad.interpretable and len(w) == 3, w
        assert "Only 2 of the 4" in w[0] and "1/(2+1) = 0.333" in w[1] and "do not resemble" in w[2], w
        assert "3 of 9 fidelity checks not met" in w[2], w[2]
        good = su.SimnullResult(mode="v2-viable", key="3|0-1", observed=1.0, n_accepted=99, n_rows=120, p=0.2,
                                message="m", n_requested=99, fidelity=fidelity_entries(5, 4, 0))
        assert good.interpretable and good.warnings() == []
        assert "All 9 fidelity checks met (this does not prove the null is right)" in good.fidelity_summary()
        none = su.SimnullResult(mode="v2-viable", key="", observed=float("nan"), n_accepted=99, n_rows=99,
                                p=float("nan"), message="m", n_requested=99, fidelity=[])
        assert none.warnings() == ["Fidelity could not be checked."], none.warnings()
        und = su.SimnullResult(mode="v2-viable", key="", observed=1.0, n_accepted=99, n_rows=99, p=0.5, message="m",
                               n_requested=99, fidelity=[{"name": "x", "met": None}])
        assert und.warnings() == ["Fidelity could not be checked."], und.warnings()
        c = bad.fidelity_counts()
        assert (c["checks"], c["checks_met"], c["leak"], c["leak_met"]) == (5, 2, 4, 4), c
        return "; ".join(w)

    check("B4: SimnullResult.warnings / interpretable / fidelity summary (pure)", warnings_pure)

    def warning_label() -> str:
        d = need(st, "dialog")
        bad = su.SimnullResult(mode="v2-viable", key="3|0-1", observed=1.0, n_accepted=2, n_rows=12, p=0.333,
                               message="only 2 of the 4 simulated axons asked for were accepted: the p uses those",
                               n_requested=4, fidelity=fidelity_entries(5, 4, 3))
        d._on_result(bad, None)
        pump(app, 0.1)
        lab = d.result_warning_label
        assert lab.isVisible(), "the warning is hidden"
        text = lab.text()
        for must in ("NOT INTERPRETABLE", "2 of the 4", "do not resemble"):
            assert must in text, (must, text)
        assert d.fidelity_label.isVisible() and "2 of 5 checks met" in d.fidelity_label.text(), d.fidelity_label.text()
        lay = d.layout()
        order = [lay.indexOf(x) for x in (d.result_warning_label, d.fidelity_label, d.result_label)]
        assert order[0] < order[1] < order[2] and lab.y() < d.result_label.y(), (order, lab.y(), d.result_label.y())
        assert "p = 0.3330 (2 accepted)" in d.result_label.text() and "NOT calibrated" in d.result_label.text()
        assert lab.font().bold() and lab.font().pointSize() > d.status_label.font().pointSize()
        good = su.SimnullResult(mode="v2-viable", key="3|0-1", observed=1.0, n_accepted=99, n_rows=110, p=0.2,
                                message="99 accepted simulated axons", n_requested=99, fidelity=fidelity_entries(5, 4, 0))
        d._on_result(good, None)
        pump(app, 0.05)
        assert not d.result_warning_label.isVisible() and "All 9 fidelity checks met" in d.fidelity_label.text()
        d._on_result(bad, None)
        # a failed run hides it (POLISH N7 kept: no p)
        d.out_edit.setText(os.path.join(WORK, "simnull_failed"))
        real_command = d.command
        setattr(d, "command", lambda npz, edits: [sys.executable, "-c", "import sys; sys.exit(3)"])
        try:
            assert d.start(), d.status_label.text()
            assert not d.result_warning_label.isVisible(), "an old warning stays while running"
            d.wait(60)
        finally:
            setattr(d, "command", real_command)
        assert "failed (exit 3)" in d.result_label.text() and "p =" not in d.result_label.text()
        assert not d.result_warning_label.isVisible() and not d.fidelity_label.isVisible()
        return str(text).replace("<br>", " | ")[:200]

    check("B4: 'NOT INTERPRETABLE' above the p (2 of 4, fidelity not met); hidden when fine and on a failed run",
          warning_label)

    def tiny_run() -> str:
        """A REAL tiny simnull on the SIMULATED 'marginal' axon (R8): 3 accepted asked, at most 6 attempts, its
        truth configuration (--config-from) to skip the measurement; under v2-viable most simulated axons are
        rejected (their pair set is not the observed one)."""
        launcher = mcw.start_columns_review(need(st, "inp_marginal"), n_null=N_NULL)
        launcher.wait(600)
        w = launcher.window
        assert w is not None, launcher.error
        w.wait_viability(120)
        st["review_marginal"] = w
        assert w.viability is not None and w.viability.viable, "the marginal demo has no VIABLE pair"
        w.advanced_toggle.setChecked(True)
        w._on_simnull()
        d = w.simnull_dialog
        assert d is not None
        d.out_edit.setText(os.path.join(WORK, "simnull_tiny"))
        d.replicates_spin.setValue(3)
        d.workers_spin.setValue(2)
        d.n_null_spin.setValue(N_NULL)
        extra = ["--config-from", need(st, "cfg_marginal"), "--seed", "7", "--zq-max-attempts", "6"]
        real_command = d.command
        setattr(d, "command", lambda npz, edits: real_command(npz, edits) + extra)
        seen: List[str] = []
        bars: List[int] = []
        timer = QtCore.QTimer()
        def sample() -> None:
            seen.append(d.status_label.text())
            bars.append(d.progress_bar.value())

        timer.timeout.connect(sample)
        timer.start(250)
        t0 = time.perf_counter()
        try:
            assert d.start(), d.status_label.text()
            d.wait(1200)
        finally:
            timer.stop()
            setattr(d, "command", real_command)
        res = d.outcome
        assert res is not None, d.result_label.text()
        log = open(d.run.log_path, encoding="utf-8", errors="replace").read()
        tags = [tuple(int(g) for g in m.groups()) for m in su._PROGRESS_TAG.finditer(log)]
        assert tags and not re.search(r"\[\d+/\d+\]", log), "old per-round tags in the log"
        attempts = [t[0] for t in tags]
        assert attempts == list(range(1, len(tags) + 1)), attempts           # counted over the whole run
        assert all(t[1] == 6 and t[3] == 3 for t in tags), tags
        accepted = [t[2] for t in tags]
        assert accepted == sorted(accepted), accepted
        rows = list(csv.DictReader(open(os.path.join(d.out_dir, f"{d.name}_simnull.csv"), encoding="utf-8")))
        n_acc_csv = sum(1 for r in rows if pc.is_accepted_row(r))
        assert len(tags) == len(rows) and accepted[-1] == min(3, n_acc_csv), (len(tags), len(rows), accepted, n_acc_csv)
        assert res.n_requested == 3 and res.n_accepted == min(3, n_acc_csv)
        assert res.n_accepted < 3, f"every simulated axon was accepted ({res.n_accepted}): the case rejects none"
        assert d.result_warning_label.isVisible(), "no warning"
        wt = d.result_warning_label.text()
        assert "NOT INTERPRETABLE" in wt and f"Only {res.n_accepted} of the 3" in wt, wt
        assert d.fidelity_label.isVisible() and "Fidelity" in d.fidelity_label.text(), d.fidelity_label.text()
        lay = d.layout()
        assert lay.indexOf(d.result_warning_label) < lay.indexOf(d.fidelity_label) < lay.indexOf(d.result_label)
        # the fidelity is the report's: the same function on the same accepted rows
        obs = list(csv.DictReader(open(os.path.join(d.out_dir, f"{d.name}_observed.csv"), encoding="utf-8")))[0]
        again = pc.simnull_fidelity(obs, pc.accepted_rows(rows, 3))
        assert [(e["name"], e["met"]) for e in again] == [(e["name"], e["met"]) for e in res.fidelity]
        md = open(os.path.join(d.out_dir, f"{d.name}_simnull.md"), encoding="utf-8").read()
        assert "Fidelity of the simulated null" in md
        status_attempts = [s for s in seen if "attempt" in s and "of up to 6" in s]
        assert status_attempts, seen[-5:]
        assert bars and bars == sorted(bars), bars                         # the bar never goes back
        assert d.progress_bar.value() == int(round(100 * accepted[-1] / 3)), (d.progress_bar.value(), accepted)
        assert d.progress_bar.format().endswith("of 3 accepted"), d.progress_bar.format()
        st["tiny_dialog"] = d
        return (f"{len(rows)} attempts, {res.n_accepted} of 3 accepted, tags {tags[0]} .. {tags[-1]}; "
                f"{res.fidelity_summary()[:80]}...; {time.perf_counter() - t0:.0f} s")

    check("B3/B4: a tiny run on a SIMULATED axon that rejects simulated axons: whole-run tags, 'Only k of the 3', "
          "fidelity on top", tiny_run)

    # ------------------------------------------------------------------ C one window each
    print("\n5. One window each (the main window)")

    def make_main() -> Any:
        import functools
        from types import SimpleNamespace
        from tools import mps_settings
        import MPS_explorer
        tmp = os.path.join(WORK, "settings")
        os.makedirs(tmp, exist_ok=True)
        MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=tmp)
        MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=tmp)
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(WORK, "main_store")
        inp = need(st, "inp_viable")
        n = int(inp.x_nm.size)
        fake: Any = SimpleNamespace(frame=inp.frame, lp_lateral_nm=inp.lp_lateral_nm, lpz_nm=inp.lpz_nm,
                                   n_frames=inp.n_frames, pixel_size_nm=130.0, pixel_size_source="override",
                                   path="simulated_viable.hdf5")
        mw.locs1 = fake
        mw.xroi_unfiltered, mw.yroi_unfiltered, mw.zroi_unfiltered = inp.x_nm.copy(), inp.y_nm.copy(), inp.z_nm.copy()
        mw.roi_indices_unfiltered = np.arange(n)
        mw.roi_indices = np.arange(n)
        mw.ui.lineEdit_filename.setText("simulated_viable.hdf5")
        return mw

    def one_zquality() -> str:
        mw = make_main()
        st["mw"] = mw
        launcher = mw.open_columns_review()
        assert launcher is not None
        launcher.wait(600)
        w = launcher.window
        assert w is not None, launcher.error
        w.wait_viability(120)
        st["mw_review"] = w
        assert w.zquality_callback == mw.open_z_quality and w.batch_callback == mw.open_columns_batch
        v_rings = mw.open_z_quality()                      # the Rings panel's button
        assert v_rings is not None
        qtest.mouseClick(w.zquality_button, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.2)
        assert w.zquality_window is v_rings and mw.zquality_window is v_rings, "a second z-quality view"
        assert v_rings.isVisible()
        v_rings.wait(300)
        assert v_rings.report is not None and [p.verdict for p in v_rings.report.v2.pairs] == ["viable", "viable"]
        tops = [x for x in QtWidgets.QApplication.topLevelWidgets()
                if isinstance(x, zw.ZQualityWindow) and x.isVisible()]
        assert tops == [v_rings], tops
        # close and re-open: the same view shown again
        v_rings.close()
        pump(app, 0.1)
        qtest.mouseClick(w.zquality_button, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.1)
        assert w.zquality_window is v_rings and v_rings.isVisible(), "closing and re-opening"
        # another axon replaces it (the other-axon rule)
        v_other = mw.open_z_quality(need(st, "inp_marginal"))
        pump(app, 0.2)
        assert v_other is not None and v_other is not v_rings and mw.zquality_window is v_other
        assert not v_rings.isVisible(), "the view of the other axon stays open"
        v_other.wait(300)
        assert v_other.report is not None, v_other.error
        other_verdicts = [p.verdict for p in v_other.report.v2.pairs]
        assert v_other.signature == mcw.review_signature(need(st, "inp_marginal")) != v_rings.signature
        rm = need(st, "review_marginal")              # the same axon's own review (check B3/B4)
        assert other_verdicts == [str(p.verdict) for p in rm.viability.pairs], (other_verdicts, rm.viability.pairs)
        # the review asks again for ITS axon: the marginal view is replaced by a view of the review's axon
        qtest.mouseClick(w.zquality_button, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.2)
        v3 = mw.zquality_window
        assert v3 is not v_other and w.zquality_window is v3 and not v_other.isVisible() and v3.isVisible()
        v3.wait(300)
        assert [p.verdict for p in v3.report.v2.pairs] == ["viable", "viable"]
        assert mw.open_z_quality() is v3, "the Rings panel's button opened another view of the same axon"
        tops = [x for x in QtWidgets.QApplication.topLevelWidgets()
                if isinstance(x, zw.ZQualityWindow) and x.isVisible()]
        assert tops == [v3], tops
        return "review button == Rings button (same view); close/re-open; another axon replaces it"

    check("C1: one Z quality view: the review's button and the main window's give the same view", one_zquality)

    def one_batch_and_explorer() -> str:
        from tools.mps_columns_batch_ui import ColumnsBatchDialog
        from tools.mps_viability_explorer import ViabilityExplorer
        mw = need(st, "mw")
        w = need(st, "mw_review")
        qtest.mouseClick(w.columns_batch_button, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.1)
        dlg = mw.columns_batch_dialog
        assert dlg is not None and w.batch_dialog is dlg and dlg.isVisible()
        assert mw.open_columns_batch() is dlg, "the Rings panel's button opened a second batch dialog"
        qtest.mouseClick(w.columns_batch_button, QtCore.Qt.MouseButton.LeftButton)
        assert mw.columns_batch_dialog is dlg
        assert dlg.explorer_callback == mw.open_viability_explorer
        dlg.add_input(need(st, "npz_viable"))
        ex = dlg.open_explorer()
        assert ex is not None and ex is mw.viability_explorer and ex.isVisible()
        assert os.path.abspath(need(st, "npz_viable")) in ex.inputs()
        assert mw.open_viability_explorer() is ex, "the Rings panel's button opened a second explorer"
        dlg.add_input(need(st, "npz_marginal"))
        qtest.mouseClick(dlg.explorer_button, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.1)
        assert mw.viability_explorer is ex and dlg.explorer is ex
        assert sorted(ex.inputs()) == sorted(os.path.abspath(need(st, k)) for k in ("npz_viable", "npz_marginal"))
        n_ex = [x for x in QtWidgets.QApplication.topLevelWidgets() if isinstance(x, ViabilityExplorer) and x.isVisible()]
        n_bd = [x for x in QtWidgets.QApplication.topLevelWidgets() if isinstance(x, ColumnsBatchDialog)
                and x.isVisible()]
        assert n_ex == [ex] and n_bd == [dlg], (n_ex, n_bd)
        # close and re-open
        ex.close()
        dlg.close()
        pump(app, 0.1)
        dlg2 = mw.open_columns_batch()
        assert dlg2 is not None and dlg2.isVisible() and mw.columns_batch_dialog is dlg2
        ex2 = dlg2.open_explorer()
        assert ex2 is not None and ex2.isVisible() and mw.viability_explorer is ex2
        dlg2.close()
        ex2.close()
        return "review 'Columns batch...' == main's dialog; its 'Viability explorer...' == main's explorer (inputs added)"

    check("C2/C3: one Columns batch dialog and one Viability explorer for the whole program", one_batch_and_explorer)

    def standalone() -> str:
        from tools.mps_columns_batch_ui import ColumnsBatchDialog
        from tools.mps_viability_explorer import ViabilityExplorer
        mw = need(st, "mw")
        w = need(st, "review_viable")                     # a review run on its own (no callbacks)
        qtest.mouseClick(w.zquality_button, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.1)
        own = w.zquality_window
        assert own is not None and own is not mw.zquality_window and own.isVisible()
        own.wait(300)
        assert own.report is not None and own.report.lab_source == "lab coordinates of the selection"
        qtest.mouseClick(w.zquality_button, QtCore.Qt.MouseButton.LeftButton)
        assert w.zquality_window is own
        qtest.mouseClick(w.columns_batch_button, QtCore.Qt.MouseButton.LeftButton)
        pump(app, 0.1)
        bd = w.batch_dialog
        assert isinstance(bd, ColumnsBatchDialog) and bd is not mw.columns_batch_dialog and bd.explorer_callback is None
        bd.add_input(need(st, "npz_viable"))
        ex = bd.open_explorer()
        assert isinstance(ex, ViabilityExplorer) and ex is not mw.viability_explorer
        assert bd.open_explorer() is ex
        for x in (ex, bd, own):
            x.close()
        return "a review / batch dialog run on its own still opens its own view, dialog and explorer"

    check("C: standalone fallbacks (tests, demos) keep their own windows", standalone)

    for key in list(st):
        obj = st[key]
        if hasattr(obj, "close") and hasattr(obj, "isVisible"):
            try:
                obj.close()
            except Exception:  # noqa: BLE001
                pass
    pump(app, 0.2)
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.0f} s; work folder {WORK}")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 1 if FAILED else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
