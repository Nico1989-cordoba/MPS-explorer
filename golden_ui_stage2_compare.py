"""
Compare two UI stage-2 goldens (``golden_ui_stage2.py``), cell by cell.

    venv/Scripts/python.exe golden_ui_stage2_compare.py GOLDEN_1.json GOLDEN_2.json [--max 40]
                                      [--cells A:S1:E1,...]

Each argument is a golden.json written by ``golden_ui_stage2.py --out`` or
the committed ``testdata/ui_stage2/parity_golden.json``. Parts stored as
blobs are expanded before they are compared, so a difference is reported
with its full path inside the cell, for example

    A:S2:E2 / entry / export_axon / tables / axon / rows / 0 / 57

Exit status: 0 when every compared cell is identical (and the inputs are
the same), 1 otherwise. Nothing is normalised here: the normalisation is
done once, when the golden is written (see the docstring of
golden_ui_stage2.py), so two goldens are compared exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Dict, List, Optional


def _load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _expand(golden: Dict[str, Any], value: Any) -> Any:
    blobs = golden.get("blobs", {})
    if isinstance(value, dict):
        if set(value) == {"$blob"}:
            return _expand(golden, blobs[value["$blob"]])
        return {k: _expand(golden, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(golden, v) for v in value]
    return value


def _diff(a: Any, b: Any, path: str, ga: Dict[str, Any], gb: Dict[str, Any],
          out: List[str], limit: int) -> None:
    if len(out) >= limit:
        return
    # Equal blob references are equal subtrees: no need to expand them.
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == {"$blob"} == set(b) \
            and a["$blob"] == b["$blob"]:
        return
    a = _expand(ga, a) if isinstance(a, dict) and set(a) == {"$blob"} else a
    b = _expand(gb, b) if isinstance(b, dict) and set(b) == {"$blob"} else b
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path} / {k}: only in the second")
            elif k not in b:
                out.append(f"{path} / {k}: only in the first")
            else:
                _diff(a[k], b[k], f"{path} / {k}", ga, gb, out, limit)
            if len(out) >= limit:
                return
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path} / {i}", ga, gb, out, limit)
            if len(out) >= limit:
                return
        return
    if a != b:
        sa, sb = json.dumps(a, ensure_ascii=False), json.dumps(b, ensure_ascii=False)
        out.append(f"{path}: {sa[:160]}  !=  {sb[:160]}")


def compare(path_a: str, path_b: str, limit: int = 40,
            cells: Optional[List[str]] = None) -> int:
    ga, gb = _load(path_a), _load(path_b)
    problems: List[str] = []
    if ga.get("golden_version") != gb.get("golden_version"):
        problems.append(f"golden_version {ga.get('golden_version')} vs {gb.get('golden_version')}")
    _diff(ga.get("axons"), gb.get("axons"), "inputs", ga, gb, problems, limit)
    va = (ga.get("environment") or {}).get("versions")
    vb = (gb.get("environment") or {}).get("versions")
    if va != vb:
        # Not a difference of the program: the last digits of the results
        # depend on these libraries. Said first, so it is not mistaken for one.
        print(f"WARNING: the goldens were made with other library versions: {va} vs {vb}")
        problems.append("environment / versions differ (see the warning above)")
    ca, cb = ga.get("cells", {}), gb.get("cells", {})
    names = sorted(set(ca) | set(cb))
    if cells:
        names = [n for n in names if n in cells]
    identical = 0
    differing: List[str] = []
    for name in names:
        if name not in ca or name not in cb:
            problems.append(f"{name}: only in the {'second' if name not in ca else 'first'}")
            differing.append(name)
            continue
        found: List[str] = []
        _diff(ca[name], cb[name], name, ga, gb, found, limit)
        if found:
            differing.append(name)
            problems.extend(found[: max(1, limit - len(problems))])
        else:
            identical += 1
    with open(path_a, "rb") as f:
        ha = hashlib.sha256(f.read()).hexdigest()
    with open(path_b, "rb") as f:
        hb = hashlib.sha256(f.read()).hexdigest()
    print(f"cells compared: {len(names)}; identical: {identical}; differing: {len(differing)}")
    failed_a = [n for n in names if n in ca and not ca[n].get("ok")]
    failed_b = [n for n in names if n in cb and not cb[n].get("ok")]
    if failed_a or failed_b:
        print(f"cells that failed to run: first {failed_a}, second {failed_b}")
    print(f"files byte-identical: {'yes' if ha == hb else 'no'} "
          f"(sha256 {ha[:16]} / {hb[:16]})")
    if differing:
        print("differing cells: " + ", ".join(differing))
    for p in problems[:limit]:
        print("  " + p)
    ok = not problems and not failed_a and not failed_b
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two UI stage-2 goldens.")
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--cells", default="")
    args = ap.parse_args(argv)
    return compare(args.first, args.second, args.max,
                   [c for c in args.cells.split(",") if c] or None)


if __name__ == "__main__":
    sys.exit(main())
