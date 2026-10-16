# -*- coding: utf-8 -*-
"""
Parity of UI stage 2 with the golden reference (design 8.1 G1-G6, 8.4; IMPL-D).

The branch drives the real application through the golden's 140 cells (2 simulated axons x 14 scenarios x 5 entry
points: ``golden_ui_stage2.py``) and must reproduce ``testdata/ui_stage2/parity_golden.json`` with EXACTLY the
differences listed in ``testdata/ui_stage2/expected_differences.json`` - each listed difference present, nothing
else different (``golden_ui_stage2_manifest.py``). The manifest itself is checked against the design:

  1. every entry names one of B1-B13 (G6: no rev-3 item may change what the golden records), or B14, the one
     behaviour change of manual test 2 the golden records (item 6: 'MPS analysis' only shows the window when the
     analysis is current, so E3 runs nothing there);
  2. none names B5, B7, B8, B9 or B10, which change no recorded field;
  3. scenarios S1, S5, S9, S11a and S11b differ only by B13, the column dictionary's line (and B14 at E3);
  4. B13 is in every cell (one line of the column dictionary written beside every "Export axon");
  5. B1 only in S6 through Rings (E4); B6 only in S7 and B4 only in S2, S3, S4, S6, through "cluster Ch1" (E2) and
     "MPS analysis" (E3); B11 only in S8, S12a, S12b through E2 / E3; B3 only where the slab was changed in the
     window (S3, S4, S10); B2 only where a value was set in the window (S2, S4, S6, S10).
  6. B17 (the species column) and B18 (SCI-1, the z calibration) in every cell: the species column and v4; the
     axon table's 15 z-calibration columns and v5, the dictionary's entries, the settings' new store; nothing else.
  7. B19 (SCI-3) in every cell: the settings' randomization store and sigma cap; the axon table's columns (v6) and the
     dictionary fall under the paths B17 and B18 already list.
  8. B20 (SCI-5) in every cell: the settings file's new (empty) store of the viability parameters; nothing else.
  9. B21 (SCI-6, the axial mixture, the ring boundaries and the ring correlation) in every cell: the axon table's
     five axial-mixture columns and v7, the dictionary's entries, the results table's foreign-fraction row, the
     warnings, the rings window and its CSVs.

R8: simulated axons only (the harness's own); every cell runs in its own temporary folder.

Run:  venv\\Scripts\\python.exe test_ui_stage2_parity.py [--golden BRANCH_GOLDEN.json] [--workers 12]
      Without --golden the 140 cells are run first (about 5-8 minutes with 12 workers).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(REPO_ROOT, "testdata", "ui_stage2", "parity_golden.json")
MANIFEST = os.path.join(REPO_ROOT, "testdata", "ui_stage2", "expected_differences.json")

ONLY_B13 = ("S1", "S5", "S9", "S11a", "S11b")
WHERE = {   # B-change -> (scenarios, entry points) it may appear in (design 6.4, 8.4)
    "B1": ({"S6"}, {"E4"}),
    "B2": ({"S2", "S4", "S6", "S10"}, {"E1", "E2", "E3", "E4", "E5"}),
    "B3": ({"S3", "S4", "S10"}, {"E1", "E2", "E3", "E4", "E5"}),
    "B4": ({"S2", "S3", "S4", "S6"}, {"E2", "E3"}),
    "B6": ({"S7"}, {"E2", "E3"}),
    "B11": ({"S8", "S12a", "S12b"}, {"E2", "E3"}),
    "B12": ({"S10"}, {"E1", "E2", "E3", "E4", "E5"}),
    # manual test 2, item 6: only through "MPS analysis" (E3), wherever the analysis was current
    "B14": ({"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11a", "S11b", "S12a", "S12b"},
            {"E3"}),
    # the species column (every export, table v4)
    "B17": ({"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11a", "S11b", "S12a", "S12b"},
            {"E1", "E2", "E3", "E4", "E5"}),
    # SCI-1: every export (15 z-calibration columns, table v5, the dictionary) and the settings' new store
    "B18": ({"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11a", "S11b", "S12a", "S12b"},
            {"E1", "E2", "E3", "E4", "E5"}),
    # SCI-3: the settings' randomization store and sigma cap in every cell (the table's v6 columns fall under the
    # B17 / B18 paths)
    "B19": ({"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11a", "S11b", "S12a", "S12b"},
            {"E1", "E2", "E3", "E4", "E5"}),
    # SCI-5: the settings file's new store of the viability parameters (empty), in every cell
    "B20": ({"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11a", "S11b", "S12a", "S12b"},
            {"E1", "E2", "E3", "E4", "E5"}),
    # SCI-6: every export (five axial-mixture columns, table v7, the dictionary), the results table's foreign
    # fraction row and warnings, the rings window and its CSVs
    "B21": ({"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11a", "S11b", "S12a", "S12b"},
            {"E1", "E2", "E3", "E4", "E5"}),
}
# Changes that reach every cell (their columns are written beside every export, or the settings file every cell
# leaves behind).
EVERY_CELL = {"B17", "B18", "B19", "B20", "B21"}


def check_manifest(manifest: List[Dict[str, Any]], cells: Set[str]) -> List[str]:
    problems: List[str] = []
    by_cell: Dict[str, Set[str]] = defaultdict(set)
    for e in manifest:
        b = str(e.get("b_change"))
        by_cell[e["cell"]].add(b)
        if b not in {f"B{i}" for i in range(1, 15)} | EVERY_CELL:
            problems.append(f"{e['cell']}: {b} is not one of B1-B13 (G6), B14 (manual test 2), B17 (species), "
                            f"B18 (SCI-1), B19 (SCI-3), B20 (SCI-5) or B21 (SCI-6)")
        if b in ("B5", "B7", "B8", "B9", "B10"):
            problems.append(f"{e['cell']}: {b} changes no recorded field and may not be listed")
        if not e.get("reason"):
            problems.append(f"{e['cell']} / {e.get('path')}: no reason")
        _axon, scenario, entry = e["cell"].split(":")
        if b in WHERE:
            scen, entries = WHERE[b]
            if scenario not in scen or entry not in entries:
                problems.append(f"{e['cell']} / {e['path']}: {b} outside the design's outline")
    for cell in sorted(cells):
        if "B13" not in by_cell.get(cell, set()):
            problems.append(f"{cell}: no B13 entry (the dictionary line is written beside every export)")
        scenario = cell.split(":")[1]
        if scenario in ONLY_B13 and by_cell.get(cell, set()) - {"B13", "B14"} - EVERY_CELL:
            problems.append(f"{cell}: {sorted(by_cell[cell] - {'B13', 'B14'} - EVERY_CELL)} in a scenario that "
                            "differs only by B13 (and B14 at E3, B17, B18, B19, B20 and B21 everywhere)")
        for b in sorted(EVERY_CELL):
            if b not in by_cell.get(cell, set()):
                problems.append(f"{cell}: no {b} entry (its columns are written beside every export)")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="UI stage 2 parity: the golden with exactly the manifest's differences")
    ap.add_argument("--golden", default="", help="a branch golden already run (skips the 140 cells)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    print("=" * 100)
    print("UI STAGE 2 PARITY (design 8.1 G1-G6): the reference golden with exactly the expected differences")
    print("=" * 100)
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(REFERENCE, encoding="utf-8") as f:
        cells = set(json.load(f)["cells"])
    problems = check_manifest(manifest, cells)
    counts: Dict[str, int] = defaultdict(int)
    for e in manifest:
        counts[e["b_change"]] += 1
    print(f"manifest: {len(manifest)} entries, " + ", ".join(f"{b} {n}" for b, n in sorted(
        counts.items(), key=lambda kv: int(kv[0][1:]))))
    for p in problems[:40]:
        print("  FAIL  " + p)
    golden = args.golden
    if not golden:
        out = tempfile.mkdtemp(prefix="ui_stage2_parity_")
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        rc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "golden_ui_stage2.py"), "--out", out,
                             "--workers", str(args.workers)], cwd=REPO_ROOT, env=env).returncode
        if rc != 0:
            print(f"  FAIL  the harness exited with {rc}")
            return 1
        golden = os.path.join(out, "golden.json")
    rc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "golden_ui_stage2_manifest.py"), REFERENCE, golden,
                         MANIFEST], cwd=REPO_ROOT).returncode
    passed = int(not problems) + int(rc == 0)
    failed = 2 - passed
    print("=" * 100)
    print(f"total {time.perf_counter() - t0:.0f} s")
    print(f"{passed} passed, {failed} failed")
    print("=" * 100)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
