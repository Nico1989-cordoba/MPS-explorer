"""
The expected-differences check of UI stage 2 (design 8.1 G2): a branch's golden against the reference golden, where
the ONLY allowed differences are the entries of a manifest - each must be present, and nothing else may differ.

    venv/Scripts/python.exe golden_ui_stage2_manifest.py REFERENCE.json BRANCH.json MANIFEST.json

MANIFEST.json is a list of entries ``{"cell": "A:S2:E2", "snapshot": "entry", "path": "export_axon / tables / ...",
"b_change": "B2", "reason": "..."}``. ``path`` is a prefix of the difference's path inside the snapshot, as
``golden_ui_stage2_compare.py`` prints it after "<cell> / snapshots / <snapshot> / "; an entry may cover several
differences under that prefix, and must cover at least one; a difference belongs to the longest prefix that matches
it. ``b_change`` must be one of B1-B13 (G6: no rev-3 item may appear). An empty list (``[]``) means "byte-identical": the expectation of every step before IMPL-D (IMPL-0 to
IMPL-C), since no B-change is implemented before it.

Exit status 0 when the differences are exactly the manifest's, 1 otherwise. Nothing is normalised here (the golden
was normalised when written).
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from golden_ui_stage2_compare import _diff, _load

B_CHANGES = {f"B{i}" for i in range(1, 14)}
# The path ends where the compare tool's ": <first> != <second>" (or ": only in ...", ": n items vs m") begins: at
# the first colon followed by a space. A key may hold a colon of its own ("other:_columns.csv").
_PREFIX = re.compile(r"^(?P<cell>[^ ]+) / snapshots / (?P<snap>[^ /]+) / (?P<path>.*?)(?:: .*)?$")


def differences(ref: Dict[str, Any], branch: Dict[str, Any]) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Every difference as (cell, snapshot, path), and the ones outside any snapshot (whole-cell, inputs)."""
    found: List[Tuple[str, str, str]] = []
    other: List[str] = []
    ca, cb = ref.get("cells", {}), branch.get("cells", {})
    for name in sorted(set(ca) | set(cb)):
        if name not in ca or name not in cb:
            other.append(f"{name}: only in one golden")
            continue
        out: List[str] = []
        _diff(ca[name], cb[name], name, ref, branch, out, 10 ** 9)
        for line in out:
            m = _PREFIX.match(line)
            if m:
                found.append((m.group("cell"), m.group("snap"), m.group("path").strip()))
            else:
                other.append(line)
    inputs: List[str] = []
    _diff(ref.get("axons"), branch.get("axons"), "inputs", ref, branch, inputs, 10 ** 9)
    other.extend(inputs)
    return found, other


def check(ref_path: str, branch_path: str, manifest_path: str) -> int:
    ref, branch = _load(ref_path), _load(branch_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest: List[Dict[str, Any]] = json.load(f)
    problems: List[str] = []
    for i, e in enumerate(manifest):
        if e.get("b_change") not in B_CHANGES:
            problems.append(f"manifest entry {i} names {e.get('b_change')!r}, not one of B1-B13 (G6)")
    found, other = differences(ref, branch)
    problems.extend(f"not a snapshot difference: {o}" for o in other)
    used: Set[int] = set()
    for cell, snap, path in found:
        # The most specific entry wins: the column dictionary's one line (B13) inside a table that another
        # B-change rewrites whole is that B13 entry's difference.
        hit: Optional[int] = None
        for i, e in enumerate(manifest):
            prefix = str(e.get("path", ""))
            if e.get("cell") == cell and e.get("snapshot") == snap and path.startswith(prefix) and (
                    hit is None or len(prefix) > len(str(manifest[hit].get("path", "")))):
                hit = i
        if hit is None:
            problems.append(f"unlisted difference: {cell} / {snap} / {path}")
        else:
            used.add(hit)
    for i, e in enumerate(manifest):
        if i not in used:
            problems.append(f"listed but not different: {e.get('cell')} / {e.get('snapshot')} / {e.get('path')} "
                            f"({e.get('b_change')})")
    failed = [n for n, c in branch.get("cells", {}).items() if not c.get("ok")]
    problems.extend(f"cell failed to run: {n}" for n in failed)
    print(f"differences found: {len(found)}; manifest entries: {len(manifest)}; matched: {len(used)}")
    for p in problems[:60]:
        print("  " + p)
    ok = not problems
    print("AS EXPECTED" if ok else "NOT AS EXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(check(sys.argv[1], sys.argv[2], sys.argv[3]))
