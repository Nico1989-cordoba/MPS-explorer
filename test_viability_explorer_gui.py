# -*- coding: utf-8 -*-
"""
Offscreen test of the viability explorer (``tools/mps_viability_explorer.py``; H6 toggles, D-43): on many axons at
once, how many ring pairs and axons every selection of the criteria passes, live, with the 16 x 4 table of every
combination of the four criteria under v2 / v2c-A / v2c-B / v2c-C.

It is GEOMETRY ONLY (R8): the computation runs here with every column analysis replaced by a function that raises
(a call would fail the file), on SIMULATED axons (tools.mps_zquality_window.DEMO_CASES plus one more seed). Checks:
the counts and the 16 x 4 table against a brute-force evaluation, the outlined cell, a click on a cell, the live
counts, the cache (a second computation reads every file from it), the worker processes (the same criteria as in
this process), the export.

Run:  venv\\Scripts\\python.exe test_viability_explorer_gui.py      (offscreen; ~1-2 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
WORK = tempfile.mkdtemp(prefix="test_viability_explorer_")
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "state", "selection_log")

import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
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


def _forbidden(name: str) -> Callable[..., Any]:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError(f"the viability explorer called the column analysis {name} (R8: geometry only)")
    return boom


def main() -> int:
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import batch_columns as bc
    from tools import mps_selection as ms
    from tools import mps_selection_ui as su
    from tools import mps_viability_explorer as ve
    from tools import mps_zquality_window as zw
    from tools.mps_viability_jobs import cache_dir_for

    st: Dict[str, Any] = {}
    state = su.selection_state()
    state.reset()

    def pump(seconds: float = 0.05) -> None:
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            app.processEvents()
            time.sleep(0.005)

    print("=" * 100)
    print("VIABILITY EXPLORER (H6 toggles, D-43): geometry only, SIMULATED axons")
    print("=" * 100)

    def simulate() -> str:
        files = []
        cases = list(zw.DEMO_CASES.items()) + [("extra", ("circle", 7, 0.45))]
        for name, (contour, seed, s) in cases:
            files.append(bc.write_simulated_input(os.path.join(WORK, f"sim_{name}.npz"), contour, seed, s))
        st["files"] = files
        return ", ".join(os.path.basename(f) for f in files)

    check("0. four SIMULATED axons (the three demo cases and one more seed)", simulate)

    def compute_geometry_only() -> str:
        # every entry point of a column statistic raises: the explorer must never reach one (R8)
        import tools.mps_lumen as ml
        import tools.mps_matching as mm
        import tools.mps_unroll as mu
        saved: List[Any] = []
        for mod, names in ((mu, ("analyze_arc_columns",)), (mm, ("analyze_columns", "match_rings", "match_arcs")),
                           (ml, ("run_cleaned_analyses",)), (bc, ("restricted_joint", "analyze_axon"))):
            for n in names:
                if hasattr(mod, n):
                    saved.append((mod, n, getattr(mod, n)))
                    setattr(mod, n, _forbidden(n))
        try:
            w = ve.ViabilityExplorer(inputs=need(st, "files"), workers=1)
            st["w"] = w
            assert not w.missing_tooltips, w.missing_tooltips
            w.show()
            t0 = time.perf_counter()
            assert w.compute()
            w.wait(900)
            dt = time.perf_counter() - t0
        finally:
            for mod, n, f in saved:
                setattr(mod, n, f)
        assert not w.errors, w.errors
        assert len(w.criteria) == 4 and w.n_computed == 4 and w.n_cached == 0, (len(w.criteria), w.n_computed)
        st["first_json"] = {p: c.to_json() for p, c in w.criteria.items()}
        return f"4 axons, {sum(len(c.pairs) for c in w.crits())} pairs, in {dt:.1f} s (in this process)"

    check("1. compute in the background with every column analysis forbidden (geometry only, R8)",
          compute_geometry_only)

    def counts_brute() -> str:
        w = need(st, "w")
        out = []
        for label in ("v2[peak,valley,count,leak:D-39]", "v2[]", "v2[peak,valley,count]", "v2c-A[peak,count]",
                      "v2c-B[valley,leak:tnfix]", "v2[peak] AND v2c-C[count]", "v2[count] OR v2c-B[peak,valley]"):
            state.set_label(label)
            pump(0.02)
            v = m = n = a = 0
            for c in w.crits():
                r = ms.evaluate_selection(c, state.spec)
                v += len(r.viable)
                m += len(r.marginal)
                n += len(r.pairs)
                a += int(bool(r.viable))
            got = w.counts()
            assert (got["viable"], got["marginal"], got["pairs"], got["axons_viable"]) == (v, m, n, a), (label, got)
            text = w.counts_label.text()
            assert f"pairs passing: {v} ({v + m}) of {n}" in text and f"{a} of 4" in text, text
            assert ("EXPLORATORY SELECTION" in text) == state.spec.exploratory
            assert w.pair_table.rowCount() == n
            out.append(f"{label}: {v} ({v + m})/{n}")
        state.reset()
        return "; ".join(out)

    check("2. live counts and the pair table follow every selection (brute force over evaluate_selection)",
          counts_brute)

    def combination() -> str:
        w = need(st, "w")
        subsets = ms.criteria_subsets()
        cols = list(ms.COMBINATION_COLUMNS)
        for label in ("v2[peak,valley,count,leak:D-39]", "v2[peak,valley,count,leak:max]",
                      "v2c-B[valley,count,leak:D-39]"):
            state.set_label(label)
            pump(0.02)
            spec = state.spec
            want = ms.combination_table(w.crits(), spec.v2.leak_estimate, spec.v2c.leak_estimate)
            assert len(want) == 64
            for c in want:
                i, j = subsets.index(tuple(c.criteria)), cols.index(c.column)
                assert w.combination_table.item(i, j).text() == c.text, (label, c.criteria, c.column)
                # brute force of one cell
                s2 = ms.spec_of_cell(c.criteria, c.column, spec.v2.leak_estimate, spec.v2c.leak_estimate)
                nv = sum(len(ms.evaluate_selection(x, s2).viable) for x in w.crits())
                assert nv == c.n_viable, (c.criteria, c.column, nv, c.n_viable)
            outlined = [(i, j) for i in range(16) for j in range(4)
                        if w.combination_table.item(i, j).data(ve.OUTLINE_ROLE)]
            cur = w.current_cell()
            assert cur is not None and outlined == [(subsets.index(tuple(cur[0])), cols.index(cur[1]))], outlined
        state.set_label("v2[peak] AND v2c-C[count]")
        pump(0.02)
        assert not any(w.combination_table.item(i, j).data(ve.OUTLINE_ROLE) for i in range(16) for j in range(4))
        state.reset()
        return "64 cells equal combination_table and a per-cell brute force; one outlined cell (none under AND)"

    check("3. the 16 x 4 table: every cell, the leak estimates of the switches, the current cell outlined",
          combination)

    def click_cell() -> str:
        w = need(st, "w")
        state.set_label("v2[leak:tnfix]")
        state.set_label("v2[peak,valley,count,leak:max]")
        subsets = ms.criteria_subsets()
        row = subsets.index(("peak", "count"))
        w._on_cell_clicked(row, list(ms.COMBINATION_COLUMNS).index("v2c-A"))
        pump(0.02)
        assert state.spec == ms.spec_of_cell(("peak", "count"), "v2c-A", "max", "tnfix"), state.spec.label
        assert w.combination_table.item(row, 1).data(ve.OUTLINE_ROLE)
        label = state.spec.label
        state.reset()
        return f"click (peak, count | v2c-A) -> {label}"

    check("4. a click on a cell sets the switches to that combination", click_cell)

    def cache_hit() -> str:
        w = need(st, "w")
        cache = cache_dir_for(ms.default_log_path())
        files = [f for f in os.listdir(cache) if f.endswith(".json")]
        assert len(files) == 4, files
        t0 = time.perf_counter()
        assert w.compute()
        w.wait(120)
        dt = time.perf_counter() - t0
        assert w.n_cached == 4 and w.n_computed == 0, (w.n_cached, w.n_computed)
        assert {p: c.to_json() for p, c in w.criteria.items()} == need(st, "first_json")
        assert dt < 5.0, f"{dt:.1f} s from the cache"
        return f"4 of 4 from {cache} in {dt:.2f} s; the same criteria"

    check("5. a second computation reads every file from the cache (same criteria)", cache_hit)

    def pool() -> str:
        other = os.path.join(WORK, "other_state", "selection_log", ms.LOG_FILE)
        w2 = ve.ViabilityExplorer(inputs=need(st, "files")[:2], workers=2, log_path=other)
        t0 = time.perf_counter()
        assert w2.compute()
        w2.wait(900)
        dt = time.perf_counter() - t0
        assert not w2.errors and w2.n_computed == 2, (w2.errors, w2.n_computed)
        first = need(st, "first_json")
        for p, c in w2.criteria.items():
            assert c.to_json() == first[p], os.path.basename(p)
        w2.close()
        return f"2 axons in 2 worker processes ({dt:.1f} s): the same criteria as in this process"

    check("6. worker processes (a fresh cache) give the same criteria", pool)

    def export() -> str:
        w = need(st, "w")
        state.set_label("v2c-B[peak,valley,count]")
        pump(0.02)
        paths = w.export_csv(os.path.join(WORK, "export", "explorer.csv"))
        with open(paths[0], encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 64 and {r["selection_hash"] for r in rows} == {state.spec.hash}
        assert sum(r["is_current_selection"] == "True" for r in rows) == 1
        with open(paths[1], encoding="utf-8", newline="") as fh:
            prs = list(csv.DictReader(fh))
        assert len(prs) == sum(len(c.pairs) for c in w.crits())
        assert {r["selection_label"] for r in prs} == {state.spec.label}
        again = w.export_csv(os.path.join(WORK, "export", "explorer.csv"))
        assert again[0] != paths[0], "an export overwrote the previous one"
        state.reset()
        json.dumps(rows[0])
        return (f"{os.path.basename(paths[0])}: 64 cells; {os.path.basename(paths[1])}: {len(prs)} pairs; never "
                "overwritten")

    check("7. export: the 16 x 4 table and the pairs, with the selection's label and hash", export)

    def demo_application_first() -> str:
        # Final audit (2026-10-06): `-m tools.mps_viability_explorer --demo` created its application AFTER the simulator
        # had imported a validation harness that sets QT_QPA_PLATFORM=offscreen: the window was never on screen. main() must
        # create the application before it simulates anything (checked with stand-ins: no window, nothing simulated).
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
        setattr(ve, "QtWidgets", Proxy())
        setattr(bc, "write_simulated_input", fake_write)
        try:
            ve.main(["--demo"])
        except Stop:
            pass
        finally:
            setattr(ve, "QtWidgets", QtWidgets)
            setattr(bc, "write_simulated_input", real_write)
        assert events[:1] == ["application"], events
        return "main(['--demo']) creates the QApplication before the simulator runs (the demo window is on screen)"

    check("8. the --demo command creates its application before simulating (its window is not offscreen)",
          demo_application_first)

    state.reset()
    if "w" in st:
        st["w"].close()
    print(f"\n{PASSED} passed, {FAILED} failed ({time.perf_counter() - T0:.0f} s)")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
