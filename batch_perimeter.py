"""
Command-line batch of the perimeter density analysis over a folder (package P10).

One file at a time, one process, no pools. Nothing existing is ever overwritten: when any table of ``out_stem``
already exists, the stem gets ``_run2``, ``_run3``, ... until none does, and the stem used is returned and
printed. The input files are only read.

Tables written (``<stem>_perimeter_<name>.csv``): ``axons`` (one row per axon, the v17 column names), ``profiles``,
``pooled`` (pooled A per group), ``by_animal``, ``contour``, ``sensitivity`` (band and axial guard),
``peak_focus`` (decision Q1, per ROI and over all axons), ``gate`` (whether each axon enters the pooled analyses
and why not) and ``gate_summary`` (how many each pooled set holds), ``genotypes`` (the animal-level comparison,
descriptive when it cannot be tested) and ``log`` (one row per file: status, reason, warnings; duplicate trace
labels). The pooled tables (pooled, by_animal, genotypes) only take axons whose contour passed the quality gate;
the slice misfit does not keep an axon out (decision D12), and every pooled table is given with and without the
axons it marked (column ``scope``), and per kind of curve.
P9 owns the final export (v17 block of the per-axon table); these are the batch's own files.

Usage::

    python batch_perimeter.py FOLDER --out STEM [--pattern axon] [--z0 peak|focus_nominal|focus_empirical]
        [--band 100] [--contour density_nodes|polar] [--single-animal | --animal LABEL] [--year 26]
        [--n-null 1999] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools import mps_perimeter_pool as pool
from tools import mps_perimeter_types as pdt
from tools import perimeter_export as pe
from tools.mps_perimeter_analysis import ContourFn, analyze_file
from tools.mps_trace_id import duplicate_labels

TABLES: Tuple[str, ...] = ("axons", "profiles", "pooled", "by_animal", "contour", "sensitivity", "peak_focus",
                           "gate", "gate_summary", "genotypes", "log")
OUTPUT_MARK: str = "_perimeter_"      # files this batch writes; never taken as input


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "" if not math.isfinite(v) else repr(v)
    return str(v)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    # "x": exclusive creation, the last guard against overwriting
    with open(path, "x", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in cols])


def free_stem(out_stem: str) -> str:
    """``out_stem`` itself when none of its tables exists, else the first ``<stem>_runN`` (N >= 2) that is free."""
    def taken(stem: str) -> bool:
        return any(os.path.exists(f"{stem}{OUTPUT_MARK}{t}.csv") for t in TABLES)

    stem, n = out_stem, 1
    while taken(stem):
        n += 1
        stem = f"{out_stem}_run{n}"
    return stem


def _genotype_rows(cmp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m, entry in cmp.get("measures", {}).items():
        row: Dict[str, Any] = {"measure": m, "descriptive_only": cmp.get("descriptive_only"),
                               "n_animals_total": cmp.get("n_animals_total"), "notes": " | ".join(cmp.get("notes", []))}
        row.update(entry)
        rows.append(row)
    if not rows:
        rows.append({"measure": None, "descriptive_only": True, "n_animals_total": cmp.get("n_animals_total"),
                     "notes": " | ".join(cmp.get("notes", [])) or "no axon with an animal and a genotype"})
    for row in rows:
        row["scope"] = cmp.get("scope")
    return rows


def process_folder(folder: str, out_stem: str, params: pdt.PerimeterParams = pdt.PerimeterParams(), *,
                   pattern: str = "axon", animal: str = "", year: str = "",
                   contour_fn: Optional[ContourFn] = None) -> Sequence[pdt.PerimeterResult]:
    """Analyse every localization file of ``folder`` and write the tables at a free stem derived from ``out_stem``.

    ``pattern``: substring filter on file names (``mps_io.find_localization_files``; "" takes every file). The
    program's own outputs and this batch's tables are skipped. ``animal``: written into every identity (e.g.
    ``mps_perimeter_pool.PROVISIONAL_ANIMAL`` for a folder known to be one animal). Returns the results, in file
    order; the stem actually used is on ``process_folder.last_stem``.
    """
    from tools.mps_io import find_localization_files

    files, _skipped = find_localization_files(folder, pattern=pattern)
    files = [f for f in files if OUTPUT_MARK not in os.path.basename(f).lower()]
    process_folder.last_stem = None  # type: ignore[attr-defined]
    results: List[pdt.PerimeterResult] = []
    if not files:
        return results                                   # nothing to write
    for path in files:                                   # one file at a time
        results.append(analyze_file(path, params, animal=animal, year=year, contour_fn=contour_fn))

    stem = free_stem(out_stem)
    parent = os.path.dirname(os.path.abspath(stem))
    os.makedirs(parent, exist_ok=True)
    rows = pool.perimeter_rows(results)
    dups = duplicate_labels([r.trace for r in results])
    log: List[Dict[str, Any]] = []
    for r in results:
        dup = [results[i].source_path for i in dups.get(r.trace.label, []) if results[i] is not r]
        log.append({"file": r.source_path, "axon_id": r.axon_id, "trace_label": r.trace.label,
                    "status": r.status.value, "failure": r.failure or None,
                    "duplicate_label_with": " | ".join(dup) or None,
                    "warnings": " | ".join(list(r.trace_warnings) + list(r.warnings)) or None})
    scoped = pool.pooled_a_scopes(results)
    genotypes: List[Dict[str, Any]] = []
    for scope in pool.row_scopes(rows):
        genotypes.extend(_genotype_rows(pool.compare_genotypes(rows, scope)))
    tables: Dict[str, List[Dict[str, Any]]] = {"axons": rows, "genotypes": genotypes, "log": log}
    # the long tables and their dictionaries: tools.perimeter_export (P9), never overwriting
    pe.write_perimeter_tables(results, stem, pooled_scopes=scoped, rows=rows)
    for name in TABLES:
        if name in pe.LONG_TABLES:
            continue
        path = f"{stem}{OUTPUT_MARK}{name}.csv"
        if name in ("axons", "log"):
            pe.write_flat_table(path, name, tables[name])      # with its _columns.csv
        else:
            _write_csv(path, tables[name])
    process_folder.last_stem = stem  # type: ignore[attr-defined]
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; returns the exit code (0 ran, 2 bad arguments or no file)."""
    ap = argparse.ArgumentParser(description="Perimeter density analysis over a folder of axon files.")
    ap.add_argument("folder")
    ap.add_argument("--out", required=True, help="output stem; tables are <stem>_perimeter_<name>.csv")
    ap.add_argument("--pattern", default="axon", help="substring of the file names to take ('' = all)")
    ap.add_argument("--z0", default=pdt.DEFAULT_Z0_CRITERION.value, choices=[c.value for c in pdt.Z0Criterion])
    ap.add_argument("--band", type=float, default=pdt.BAND_HALF_WIDTH_NM, help="band half-width, nm")
    ap.add_argument("--contour", default=pdt.DEFAULT_CONTOUR_METHOD, choices=["density_nodes", "polar"])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--single-animal", action="store_true",
                       help=f"the folder is ONE animal of unknown label: use '{pool.PROVISIONAL_ANIMAL}'")
    group.add_argument("--animal", default="", help="animal label written into every identity")
    ap.add_argument("--year", default="", help="acquisition year typed for the folder (wins over file names)")
    ap.add_argument("--n-null", type=int, default=pdt.N_NULL_DEFAULT)
    ap.add_argument("--seed", type=int, default=pdt.SEED_DEFAULT)
    ap.add_argument("--trim-interior", type=float, default=None, metavar="NM",
                    help="OFF by default (decision D8): leave out of the contour fit the localizations deeper than NM "
                         "below the polar ring (250 when used); trims nothing when that ring is not valid")
    try:
        args = ap.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    if not os.path.isdir(args.folder):
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2
    params = replace(pdt.PerimeterParams(), z0_criterion=pdt.Z0Criterion(args.z0), band_half_width_nm=args.band,
                     contour_method=args.contour, n_null=args.n_null, seed=args.seed,
                     contour_trim_interior_nm=args.trim_interior)
    animal = pool.PROVISIONAL_ANIMAL if args.single_animal else args.animal
    results = process_folder(args.folder, args.out, params, pattern=args.pattern, animal=animal, year=args.year)
    if not results:
        print("no localization file found", file=sys.stderr)
        return 2
    stem = getattr(process_folder, "last_stem", args.out)
    n_ok = sum(1 for r in results if r.status == pdt.Status.OK)
    print(f"{len(results)} files, {n_ok} ok; tables at {stem}{OUTPUT_MARK}*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
