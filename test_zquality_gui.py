# -*- coding: utf-8 -*-
"""
Offscreen test of the H6 z-quality GUI (D-41: the ring-pair viability rule v2 in MPS Explorer), on SIMULATED axons
only (R8: no column statistic on a real axon; tools.mps_simulate_axon through batch_columns.write_simulated_input):

  "viable"   tools.mps_sim_harness's circle contour, seed 39, axial leak x0.3 -> both pairs VIABLE under rule v2
  "none"     circle, seed 30, leak x0.7                                     -> no viable pair (wide rings, much leak)
  "marginal" circle, seed 22, leak x0.6                                     -> one VIABLE, one MARGINAL pair

What is checked (the numbers print with each check):
  1. the Z quality view: computed in the background (the call returns at once), the verdicts of the pair table equal
     viability_v2 on the lab coordinates, the central ring marked, the SiZer legend always present, the plain summary,
     the export (2 PNG + 4 CSV + text, never overwritten);
  2. the columns review window: the viability table, the column analysis ONLY on viable pairs (marginal = sensitivity),
     nothing computed and the reasons shown when no pair is viable, every p labelled "not calibrated", the primary p
     equal to the batch runner's for the same axon and rule, the export columns;
  3. the simulated-null button: enabled only with a viable pair, its command (v5, --zq-select v2-viable, the lumen
     cleaning), refusal of an output folder inside the repo, start in a background process and Cancel (the process
     tree is gone); with --full-simnull a complete tiny run (3 replicates) and its NOT-calibrated p;
  4. the columns batch dialog: 2 simulated axons through batch_columns.py in a subprocess with live progress, exit 0,
     the tables; a real input is refused unless confirmed, and --allow-real is passed only after the confirmation;
  5. the main window: Rings panel callbacks, open_z_quality on the current ROI (and the same view brought forward).

Run:  venv\\Scripts\\python.exe test_zquality_gui.py [--full-simnull]      (offscreen; ~3-5 min, +5 min full)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import inspect  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CASES = {"viable": ("circle", 39, 0.3), "none": ("circle", 30, 0.7), "marginal": ("circle", 22, 0.6)}
N_NULL = 19
PASSED = 0
FAILED = 0
T0 = time.perf_counter()
WORK = tempfile.mkdtemp(prefix="test_zquality_gui_")
# the exploration log of this test, never the user's (tools.mps_selection.default_log_path)
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")


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


def rows_of(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def table_texts(t: Any, col: int) -> List[str]:
    return [t.item(r, col).text() if t.item(r, col) is not None else "" for r in range(t.rowCount())]


def run_window(app: Any, w: Any, timeout: float = 600.0) -> Any:
    got: Dict[str, Any] = {}
    w.run_finished.connect(lambda r: got.__setitem__("r", r))
    assert w.start_run(), "the run did not start"
    assert w.is_running() and not w.run_button.isEnabled(), "the run button stays enabled while running"
    t_end = time.perf_counter() + timeout
    while "r" not in got and time.perf_counter() < t_end:
        pump(app, 0.02)
        if not w.is_running() and "r" not in got:
            break
    pump(app, 0.05)
    assert "r" in got, f"no run result (error: {getattr(w, '_run_error', None)})"
    return got["r"]


def main() -> int:
    full_simnull = "--full-simnull" in sys.argv[1:]
    print("=" * 100)
    print("Z-QUALITY GUI (H6, D-41) OFFSCREEN TEST on SIMULATED axons")
    print("=" * 100)
    from PyQt5 import QtCore, QtTest, QtWidgets
    qtest: Any = QtTest.QTest  # static helpers; the stubs declare them as instance methods
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import batch_columns as bc
    from tools import mps_columns_window as mcw
    from tools import mps_zquality_window as zw
    from tools.mps_axial_precision import viability_v2, z_quality
    st: Dict[str, Any] = {}

    print("\n0. The simulated axons")

    def simulate() -> str:
        out = []
        for name, (contour, seed, s) in CASES.items():
            p = bc.write_simulated_input(os.path.join(WORK, f"sim_{name}.npz"), contour, seed, s)
            st[f"npz_{name}"] = p
            st[f"inp_{name}"] = zw.inputs_from_npz(p)
            assert bc.input_kind(p)[0] == "simulated"
            out.append(f"{name}: {st[f'inp_{name}'].x_nm.size:,} locs")
        return "; ".join(out)

    check("three SIMULATED axons written as marked-simulated NPZ (R8-safe)", simulate)

    # ------------------------------------------------------------------ 1. the view
    print("\n1. The Z quality view")

    def view_case(name: str, expect: List[str]) -> Callable[[], str]:
        def fn() -> str:
            inp = need(st, f"inp_{name}")
            t0 = time.perf_counter()
            w = zw.open_z_quality_for_inputs(inp)
            dt = time.perf_counter() - t0
            assert w.is_running() and dt < 1.0, f"the view blocked for {dt:.2f} s"
            ticks = {"n": 0}
            timer = QtCore.QTimer()
            timer.timeout.connect(lambda: ticks.__setitem__("n", ticks["n"] + 1))
            timer.start(20)
            w.wait(300)
            timer.stop()
            assert w.error is None and w.report is not None, w.error
            assert ticks["n"] >= 3, f"the event loop did not run while computing ({ticks['n']} ticks)"
            st[f"view_{name}"] = w
            rep = w.report
            # parity with viability_v2 on the lab coordinates of the same rings
            from tools.mps_columns_window import build_review_rings
            res, _p, _l, _w = build_review_rings(inp)
            direct = viability_v2(res, z_quality(res), xyz_lab_nm=(inp.x_nm, inp.y_nm, inp.z_nm))
            got = [p.verdict for p in rep.v2.pairs]
            assert got == [p.verdict for p in direct.pairs] == expect, (got, [p.verdict for p in direct.pairs], expect)
            texts = table_texts(w.pair_table, zw.PAIR_COLUMNS.index("Verdict (rule v2)"))
            want = {"viable": "VIABLE", "marginal": "MARGINAL", "not viable": "NOT VIABLE"}
            assert all(want[v] in t for v, t in zip(expect, texts)) and len(texts) == len(expect), texts
            roles = table_texts(w.ring_table, 1)
            assert sum("central" in r for r in roles) == 1, roles
            assert w.ring_table.rowCount() == len(rep.v2.rings)
            labels = [lab.text for _s, lab in w.legend.items]
            for s in (1, -1, 0):
                assert zw.SIZER_LABELS[s] in labels, (zw.SIZER_LABELS[s], labels)
            assert not w.missing_tooltips, w.missing_tooltips
            summary = w.summary_view.toPlainText()
            assert "WHAT LIMITS Z QUALITY HERE" in summary and "NOT calibrated" in summary
            if "viable" not in expect and "marginal" not in expect:
                assert "No pair passes the rule" in summary and "WHAT WOULD HELP" in summary
            # POLISH N2: no advice the data ruled out; the measured levers, every advice line with its evidence
            assert "drift correction" not in summary and "fewer rings" not in summary, "advice without evidence"
            assert "covers +-3 sigma" in summary and "widen the z filter when exporting from Picasso" in summary
            help_ = summary.split("WHAT WOULD HELP")[1].split("HOW THE VERDICT IS MADE")[0]
            bullets = [ln for ln in help_.splitlines() if ln.startswith("   * ")]
            assert bullets and all(ln.rstrip().endswith("]") or "Nothing limits" in ln for ln in bullets), bullets
            # POLISH N3: three decimals, judged on the value shown, no hard-threshold wording
            ax = [f for f in rep.v2.limiting_factors if f.key == "axial_resolution"][0]
            shown = float(str(ax.value).split()[0])
            assert len(str(ax.value).split()[0].split(".")[1]) == 3, ax.value
            assert (ax.status == "limiting") == (shown > 0.30), (ax.value, ax.status)
            assert "no hard threshold" in ax.needed and "becomes detectable" not in summary
            return (f"{len(rep.v2.rings)} rings, verdicts {got}, central ring {rep.v2.central_index}, x {rep.v2.x:.1f}, "
                    f"{rep.seconds:.1f} s, {ticks['n']} timer ticks meanwhile")
        return fn

    check("viable axon: both pairs VIABLE in the view, equal to viability_v2 (lab coordinates)",
          view_case("viable", ["viable", "viable"]))
    check("axon with no viable pair: both NOT VIABLE, the summary says why and what would help",
          view_case("none", ["not viable", "not viable"]))

    def lab_rebuild() -> str:
        inp = need(st, "inp_viable")
        res, _p, _l, _w = mcw.build_review_rings(inp)
        x, y, z = zw.lab_xyz_of(res)
        err = max(float(np.max(np.abs(x - inp.x_nm))), float(np.max(np.abs(y - inp.y_nm))),
                  float(np.max(np.abs(z - inp.z_nm))))
        assert err < 1e-6, err
        return f"max |lab - rebuilt| = {err:.1e} nm"

    check("the lab coordinates rebuilt from the axon frame equal the selection's", lab_rebuild)

    def export() -> str:
        w = need(st, "view_none")
        folder = os.path.join(WORK, "zq_export")
        a = w.export_report(folder)
        b = w.export_report(folder)
        assert len(a) == 7 and len(b) == 7, (len(a), len(b))
        assert not set(a) & set(b), "the second export overwrote the first"
        pngs = [p for p in a if p.endswith(".png")]
        assert len(pngs) == 2 and all(os.path.getsize(p) > 5000 for p in pngs), [os.path.getsize(p) for p in pngs]
        pairs = rows_of([p for p in a if p.endswith("_pairs.csv")][0])
        assert [r["verdict"] for r in pairs] == ["not viable", "not viable"] and all(r["reasons"] for r in pairs)
        rings = rows_of([p for p in a if p.endswith("_rings.csv")][0])
        assert sum(r["is_central"] == "True" for r in rings) == 1
        txt = open([p for p in a if p.endswith("_summary.txt")][0], encoding="utf-8").read()
        assert "WHAT LIMITS Z QUALITY HERE" in txt
        prof = rows_of([p for p in a if p.endswith("_profile.csv")][0])
        assert prof and {int(r["sizer_sign"]) for r in prof} <= {-1, 0, 1}
        return f"{[os.path.basename(p) for p in a][:3]}... and a second set with suffix _2"

    check("Export z-quality report: 2 PNG + 4 CSV + summary, never overwritten", export)

    # ------------------------------------------------------------------ 2. the review window
    print("\n2. The columns review window: viable pairs only")

    def open_review(name: str) -> Any:
        launcher = mcw.start_columns_review(need(st, f"inp_{name}"), n_null=N_NULL)
        launcher.wait(600)
        w = launcher.window
        assert w is not None, launcher.error
        w.wait_viability(120)
        assert w.viability is not None, w.viability_error
        st[f"launcher_{name}"] = launcher
        return w

    def review_viable() -> str:
        w = open_review("viable")
        st["review_viable"] = w
        assert w.viability_table.rowCount() == 2 and all("VIABLE" in t and "NOT" not in t
                                                         for t in table_texts(w.viability_table, 1))
        assert w.simnull_button.isEnabled(), w.simnull_button.toolTip()
        assert w.zquality_button.isEnabled() and w.columns_batch_button.isEnabled()
        missing = [n for n in ("viability_table", "zquality_button", "simnull_button", "columns_batch_button")
                   if not w.findChild(QtCore.QObject, n)]
        assert not missing, missing
        run = run_window(app, w)
        st["run_viable"] = run
        assert run.ran and run.primary is not None and run.primary["n_pairs"] == 2 and run.sensitivity is None
        text = w.results_text()
        assert "PRIMARY: viable pairs" in text and "SENSITIVITY: viable + marginal" not in text
        assert text.count("not calibrated") >= 3, text.count("not calibrated")
        for line in text.splitlines():
            if "p_excess" in line or "p_A" in line:
                assert "not calibrated" in line, line
        assert w.decisions.results_shown_before_edit is False and w.leak_banner.isVisible()
        return (f"primary pairs {run.primary['pairs']}: z_A {run.primary['z_A']:+.2f}, p_excess "
                f"{run.primary['p_excess_uncalibrated']:.3f} (not calibrated), {run.seconds:.0f} s")

    check("viable axon: viability table, buttons enabled, the run restricted to the VIABLE pairs, every p labelled",
          review_viable)

    def parity_batch() -> str:
        run = need(st, "run_viable")
        out = bc.process_axon(need(st, "npz_viable"), bc.ColumnBatchSettings(n_null=N_NULL))
        ax = out["axon"]
        assert ax["analysis_status"] == "ok", ax["analysis_status"]
        for f in ("n_pairs", "T_obs", "p_excess_uncalibrated", "z_A"):
            a, b = ax[f"primary_{f}"], run.primary[f]
            assert (a == b) or (isinstance(a, float) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)), (f, a, b)
        assert ax["viability_rule"] == run.viability.rule
        return f"batch primary p {ax['primary_p_excess_uncalibrated']:.4f} == window's; rule text equal"

    check("parity: the window's primary p equals batch_columns.process_axon's (same axon, rule v2, n_null)",
          parity_batch)

    def export_review() -> str:
        w = need(st, "review_viable")
        path = os.path.join(WORK, "review_decisions.csv")
        written = w.export_tables(path, duplicates="append")
        res = rows_of(written[1])[0]
        assert "rule v2" in res["viability_rule"] and res["viable_pairs"] == "0-1 1-2", res["viable_pairs"]
        assert res["primary_n_pairs"] == "2" and res["primary_p_excess_uncalibrated"]
        assert "NOT calibrated" in res["calibration_note"]
        assert set(mcw.RESULTS_TABLE_COLUMNS) >= {f"primary_{f}" for f in bc.TIER_STAT_FIELDS}
        assert mcw.TIER_STAT_FIELDS == bc.TIER_STAT_FIELDS
        return f"results row: viable_pairs {res['viable_pairs']}, primary p {res['primary_p_excess_uncalibrated']}"

    check("the review's export carries the rule, the pairs and the restricted (primary) statistics", export_review)

    def export_restricted() -> str:
        """Review of H6: the results row carries no statistic of a pair rule v2 did not select, and a test's joint over
        all of its pairs only when every one of them is VIABLE (else the restricted primary_* / sensitivity_*)."""
        w, run = need(st, "review_viable"), need(st, "run_viable")
        a, PV, head = run.analyses, bc.PairVerdict, {"axon_id": "probe"}
        joints = ("arcc_z_A", "arcc_p_A", "arcc_T_A", "cols2d_z_A", "cols2d_p_A", "arcl_z_A", "arcl_p_A")
        one = mcw.ReviewRun(mcw.ReviewViability("stub", (PV(0, 1, "viable"), PV(1, 2, "not viable"))), a,
                            primary=bc.restricted_joint(a.arc_centroid, [(0, 1)]))
        row = mcw.results_row(a, w.decisions, head, res_before=w.res, run=one)
        for c in joints + ("arcc_zeta_pair1", "arcc_p_excess_pair1", "cols2d_zeta_pair1", "cols2d_p_excess_pair1"):
            assert row[c] is None, (c, row[c])
        assert row["arcc_zeta_pair0"] is not None and "1-2" not in row["arcc_pairs"] and row["primary_pairs"] == "0-1"
        marg = mcw.ReviewRun(mcw.ReviewViability("stub", (PV(0, 1, "marginal"), PV(1, 2, "viable"))), a)
        row = mcw.results_row(a, w.decisions, head, res_before=w.res, run=marg)
        assert all(row[c] is None for c in joints) and "(marginal: sensitivity only)" in row["arcc_pairs"]
        full = mcw.results_row(a, w.decisions, head, res_before=w.res, run=run)
        assert math.isclose(full["arcc_z_A"], run.primary["z_A"], rel_tol=1e-12), (full["arcc_z_A"], run.primary["z_A"])
        return "a NOT VIABLE pair has no cells; a joint over a non-viable pair is blank; all viable: joint = primary"

    check("the review's export never carries a statistic of an unselected pair or an unrestricted joint",
          export_restricted)

    def window_joints() -> str:
        """POLISH N8: the window shows the 2D-test and localization-membrane joints only when every pair is VIABLE,
        labelled not calibrated, with the numbers the export writes (results_row)."""
        w, run = need(st, "review_viable"), need(st, "run_viable")
        text = w.results_text()
        joints = [ln for ln in text.splitlines() if "JOINT: every pair VIABLE" in ln]
        assert len(joints) == 2 and all(ln.count("(not calibrated)") == 2 for ln in joints), joints
        row = mcw.results_row(run.analyses, w.decisions, {"axon_id": "probe"}, res_before=w.res, run=run)
        for ln, (z, p) in zip(joints, (("cols2d_z_A", "cols2d_p_A"), ("arcl_z_A", "arcl_p_A"))):
            assert f"z_A = {row[z]:+.3f}" in ln and f"two-sided p = {row[p]:.4f}" in ln, (ln, row[z], row[p])
        return f"2D joint z_A {row['cols2d_z_A']:+.3f}, arc-of-localizations joint z_A {row['arcl_z_A']:+.3f} = export"

    check("the window's 2D and localization-membrane joints (all pairs VIABLE) equal the export's", window_joints)

    def review_none() -> str:
        w = open_review("none")
        st["review_none"] = w
        assert all("NOT VIABLE" in t for t in table_texts(w.viability_table, 1))
        assert not w.simnull_button.isEnabled() and "no ring pair" in w.simnull_button.toolTip().lower()
        run = run_window(app, w)
        assert not run.ran and w.last_result is None
        text = w.results_text()
        assert mcw.NO_VIABLE_PAIR_TEXT in text and "z_A" not in text and "p_excess" not in text
        assert "Z quality" in text and "no significant valley" in text
        assert not w.decisions.results_shown_before_edit, "a run that computed nothing marked results as shown"
        assert not w.leak_banner.isVisible()
        return f"nothing computed in {run.seconds:.2f} s; the reasons listed; simnull disabled"

    check("no viable pair: nothing computed, the reasons shown instead, simnull button disabled with its reason",
          review_none)

    def review_marginal() -> str:
        w = open_review("marginal")
        st["review_marginal"] = w
        v = w.viability
        assert v.viable and v.marginal, (v.viable, v.marginal)
        run = run_window(app, w)
        assert run.ran and run.primary["n_pairs"] == len(v.viable)
        assert run.sensitivity is not None and run.sensitivity["n_pairs"] == len(v.viable) + len(v.marginal)
        text = w.results_text()
        assert "SENSITIVITY: viable + marginal" in text and "[marginal: sensitivity only]" in text
        assert "JOINT: every pair VIABLE" not in text, "a joint over a MARGINAL pair is shown"     # POLISH N8
        return f"viable {v.viable}, marginal {v.marginal}: primary {run.primary['pairs']}, sensitivity {run.sensitivity['pairs']}"

    check("viable + marginal axon: primary on the viable pair, the marginal one only as sensitivity", review_marginal)

    def zq_from_review() -> str:
        w = need(st, "review_viable")
        qtest.mouseClick(w.zquality_button, QtCore.Qt.MouseButton.LeftButton)
        view = w.zquality_window
        assert view is not None
        view.wait(120)
        assert view.report is not None and [p.verdict for p in view.report.v2.pairs] == ["viable", "viable"]
        assert view.report.lab_source == "lab coordinates of the selection"
        return "the review's 'Z quality...' opens the view on its own rings"

    check("'Z quality...' in the review window", zq_from_review)

    # ------------------------------------------------------------------ 3. simulated-null button
    print("\n3. Simulated-null p (not calibrated)")

    def simnull_cancel() -> str:
        from tools import mps_simnull_ui as su
        w = need(st, "review_viable")
        qtest.mouseClick(w.simnull_button, QtCore.Qt.MouseButton.LeftButton)
        d = w.simnull_dialog
        assert d is not None and isinstance(d, su.SimnullDialog)
        assert "NOT calibrated" in su.SIMNULL_RESULT_LABEL and "not calibrated" in w.simnull_button.text()
        d.out_edit.setText(os.path.join(REPO_ROOT, "should_not_exist_simnull"))
        assert not d.start() and not os.path.exists(os.path.join(REPO_ROOT, "should_not_exist_simnull"))
        d.out_edit.setText(os.path.join(WORK, "simnull_cancel"))
        d.replicates_spin.setValue(3)
        d.workers_spin.setValue(2)
        d.n_null_spin.setValue(N_NULL)
        assert d.start(), d.status_label.text()
        assert d.run is not None
        cmd = d.run.args
        for tok in ("simnull", "--table-version", "v5", "--zq-select", "v2-viable", "--lumen-clean", "--from-arrays"):
            assert tok in cmd, (tok, cmd)
        assert os.path.isfile(cmd[cmd.index("--from-arrays") + 1])
        pump(app, 3.0)
        assert d.run.poll() is None, "the run ended by itself within 3 s: " + "\n".join(
            open(d.run.log_path, encoding="utf-8", errors="replace").read().splitlines()[-5:])
        pid = d.run.pid
        t0 = time.perf_counter()
        d.cancel()
        dt_cancel = time.perf_counter() - t0
        assert dt_cancel < 1.0, f"Cancel blocked the GUI thread for {dt_cancel:.2f} s"     # POLISH N11
        t_end = time.perf_counter() + 15.0
        while time.perf_counter() < t_end and "Cancelled" not in d.status_label.text():
            pump(app, 0.1)
        assert d.run.poll() is not None and d.run.cancelled and "Cancelled" in d.status_label.text()
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        assert str(pid) not in out, out
        st["simnull_dialog"] = d
        return f"command {' '.join(cmd[3:12])} ...; cancel returned in {dt_cancel:.3f} s, pid {pid} gone"

    check("the simnull dialog: refuses a folder in the repo, starts simnull in a subprocess, Cancel kills it",
          simnull_cancel)

    def simnull_failed() -> str:
        """POLISH N7: a run that exits != 0 shows 'failed (exit N)' and never a p read from the folder's old rows."""
        d = need(st, "simnull_dialog")
        d.out_edit.setText(os.path.join(WORK, "simnull_failed"))
        real_command = d.command
        setattr(d, "command", lambda npz, edits: [sys.executable, "-c", "import sys; sys.exit(3)"])
        try:
            assert d.start(), d.status_label.text()
            d.wait(60)
        finally:
            setattr(d, "command", real_command)
        assert d.outcome is None and "failed (exit 3)" in d.result_label.text(), d.result_label.text()
        assert "p =" not in d.result_label.text() and "exit 3" in d.status_label.text()
        return str(d.result_label.text())

    check("the simnull dialog: a failed run says 'failed (exit N)' and shows no p", simnull_failed)

    if full_simnull:
        def simnull_full() -> str:
            d = need(st, "simnull_dialog")
            d.out_edit.setText(os.path.join(WORK, "simnull_full"))
            assert d.start()
            d.wait(1800)
            assert d.outcome is not None, d.result_label.text()
            assert "NOT calibrated" in d.result_label.text()
            return f"p {d.outcome.p:.3f} ({d.outcome.n_accepted} accepted, key {d.outcome.key}); {d.outcome.message}"

        check("FULL tiny simnull (3 replicates, n_null 19): a NOT-calibrated p is shown", simnull_full)

    # ------------------------------------------------------------------ 4. batch dialog
    print("\n4. Columns batch dialog")

    def batch() -> str:
        from tools.mps_columns_batch_ui import ColumnsBatchDialog
        d = ColumnsBatchDialog()
        d.open_when_done = False
        asked: List[Any] = []
        d.confirm_real = lambda n, t: asked.append((n, t)) or False  # type: ignore[func-returns-value]
        # a real input (any non-simulated file) is refused unless confirmed
        fake = os.path.join(WORK, "fake_real_axon.hdf5")
        open(fake, "wb").close()
        d.add_input(fake)
        d.out_edit.setText(os.path.join(WORK, "batch_refused"))
        assert not d.start() and asked == [(1, 1)] and d.run is None
        d.allow_real = True
        assert "--allow-real" in d.command("x.txt")
        d.allow_real = False
        d.inputs_list.clear()
        d.add_input(need(st, "npz_viable"))
        d.add_input(need(st, "npz_none"))
        d.out_edit.setText(os.path.join(REPO_ROOT, "should_not_exist_batch"))
        assert not d.start()
        d.out_edit.setText(os.path.join(WORK, "batch"))
        d.workers_spin.setValue(2)
        d.n_null_spin.setValue(N_NULL)
        t0 = time.perf_counter()
        assert d.start(), d.status_label.text()
        assert d.run is not None
        assert "--allow-real" not in d.run.args and "--n-null" in d.run.args
        assert time.perf_counter() - t0 < 2.0
        d.wait(900)
        assert d.exit_code == 0, (d.exit_code, d.log_view.toPlainText()[-800:])
        assert d.progress_bar.value() == 2 and d.progress_bar.maximum() == 2
        axons = rows_of(os.path.join(WORK, "batch", "column_axons.csv"))
        assert len(axons) == 2 and all("rule v2" in r["viability_rule"] for r in axons)
        status = sorted(r["analysis_status"] for r in axons)
        assert status == ["no analysable pair", "ok"], status
        assert "finished" in d.status_label.text()
        return f"exit 0 in {time.perf_counter() - t0:.0f} s; statuses {status}"

    check("batch dialog: 2 simulated axons through batch_columns.py with progress; real input needs confirmation",
          batch)

    # ------------------------------------------------------------------ 5. main window
    print("\n5. The main window and the Rings panel")

    def rings_panel() -> str:
        from tools.mps_rings_window import MPSRingsWindow
        sig = inspect.signature(MPSRingsWindow.__init__).parameters
        assert "zquality_callback" in sig and "batch_callback" in sig
        src = inspect.getsource(sys.modules["MPS_explorer"].MPS_explorer._show_rings_window) if "MPS_explorer" in sys.modules else ""
        return "MPSRingsWindow takes zquality_callback and batch_callback" + (" (wired)" if "open_z_quality" in src else "")

    def main_window() -> str:
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
        v1 = mw.open_z_quality()
        assert v1 is not None
        v1.wait(300)
        assert v1.report is not None and [p.verdict for p in v1.report.v2.pairs] == ["viable", "viable"], v1.error
        assert mw.open_z_quality() is v1, "pressing it again rebuilt the view"
        src = inspect.getsource(MPS_explorer.MPS_explorer._show_rings_window)
        assert "zquality_callback=self.open_z_quality" in src and "batch_callback=self.open_columns_batch" in src
        dlg = mw.open_columns_batch()
        assert dlg is not None and dlg.inputs_list.count() == 0
        dlg.close()
        v1.close()
        mw.close()
        return "open_z_quality on the ROI: VIABLE x2; the same view brought forward; the batch dialog opens empty"

    check("Rings panel accepts the two new callbacks", rings_panel)
    check("main window: Z quality on the current ROI, wired into the Rings panel", main_window)

    def demo_application_first() -> str:
        # Final audit (2026-10-06): the guide's demo (-m tools.mps_zquality_window --demo ...) created its application
        # AFTER the simulator had imported a validation harness that sets QT_QPA_PLATFORM=offscreen: the window was never on
        # screen. main() must create the application before it simulates anything (checked with stand-ins: no window).
        import batch_columns as bc
        import tools.mps_zquality_window as zw
        events: List[str] = []

        class Stop(Exception):
            pass

        class RecordingApp:
            @staticmethod
            def instance() -> None:
                return None

            def __init__(self, *_a: Any) -> None:
                events.append("application")
                raise Stop()

        class Proxy:
            QApplication = RecordingApp

            def __getattr__(self, name: str) -> Any:
                return getattr(QtWidgets, name)

        def fake_write(path: str, *_a: Any, **_k: Any) -> str:
            events.append("simulation")
            return path

        real_write = bc.write_simulated_input
        setattr(zw, "QtWidgets", Proxy())
        setattr(bc, "write_simulated_input", fake_write)
        try:
            zw.main(["--demo", "viable"])
        except Stop:
            pass
        finally:
            setattr(zw, "QtWidgets", QtWidgets)
            setattr(bc, "write_simulated_input", real_write)
        assert events[:1] == ["application"], events
        return "main(['--demo', ...]) creates the QApplication before the simulator runs (the demo window is on screen)"

    check("the guide's demo command creates its application before simulating (its window is not offscreen)",
          demo_application_first)

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
    _code = main()
    # Leave without Python's teardown of the Qt objects (main window, plots, dialogs): their destruction order at
    # interpreter exit can crash (exit 139) after every check has passed. The result is already printed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
