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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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


def analyse_files(files: Sequence[str], params: pdt.PerimeterParams = pdt.PerimeterParams(), *,
                  animal: str = "", year: str = "", contour_fn: Optional[ContourFn] = None,
                  identities: Optional[Mapping[str, Any]] = None,
                  pixel_sizes: Optional[Mapping[str, float]] = None,
                  cancelled: Optional[Callable[[], bool]] = None,
                  on_file: Optional[Callable[[int, str], None]] = None) -> List[pdt.PerimeterResult]:
    """Analyse ``files`` in order, one at a time; write nothing. ``identities``: an ``AxonIdentity`` typed for a
    path (by path), used instead of the one proposed from it. ``pixel_sizes``: the pixel size settled for a path
    whose file lost its own. ``cancelled``: asked before each file; True stops the loop (the results so far are
    returned). ``on_file(index, path)``: told before each file starts (a progress bar)."""
    typed = dict(identities) if identities else {}
    pixels = dict(pixel_sizes) if pixel_sizes else {}
    results: List[pdt.PerimeterResult] = []
    for i, path in enumerate(files):                     # one file at a time
        if cancelled is not None and cancelled():
            break
        if on_file is not None:
            on_file(i, path)
        results.append(analyze_file(path, params, identity=typed.get(path), animal=animal, year=year,
                                    contour_fn=contour_fn, pixel_size_nm=pixels.get(path)))
    return results


def pooled_of(results: Sequence[pdt.PerimeterResult]) -> Dict[str, Any]:
    """The pooled A of every group (the slow part of the writing: its Monte Carlo null). A caller with a worker
    thread computes it there and hands it to ``write_tables``."""
    return dict(pool.pooled_a_scopes(list(results)))


def write_tables(results: Sequence[pdt.PerimeterResult], out_stem: str, *,
                 pooled: Optional[Mapping[str, Any]] = None) -> str:
    """Write every table of ``results`` at a free stem derived from ``out_stem`` (``_run2``... when taken; nothing
    is overwritten) and return the stem used. ``pooled``: ``pooled_of(results)`` when already computed."""
    stem = free_stem(out_stem)
    parent = os.path.dirname(os.path.abspath(stem))
    os.makedirs(parent, exist_ok=True)
    rows = pool.perimeter_rows(list(results))
    dups = duplicate_labels([r.trace for r in results])
    log: List[Dict[str, Any]] = []
    for r in results:
        dup = [results[i].source_path for i in dups.get(r.trace.label, []) if results[i] is not r]
        log.append({"file": r.source_path, "axon_id": r.axon_id, "trace_label": r.trace.label,
                    "status": r.status.value, "failure": r.failure or None,
                    "duplicate_label_with": " | ".join(dup) or None,
                    "warnings": " | ".join(list(r.trace_warnings) + list(r.warnings)) or None})
    scoped = dict(pooled) if pooled is not None else pooled_of(results)
    genotypes: List[Dict[str, Any]] = []
    for scope in pool.row_scopes(rows):
        genotypes.extend(_genotype_rows(pool.compare_genotypes(rows, scope)))
    tables: Dict[str, List[Dict[str, Any]]] = {"axons": rows, "genotypes": genotypes, "log": log}
    # the long tables and their dictionaries: tools.perimeter_export (P9), never overwriting
    pe.write_perimeter_tables(list(results), stem, pooled_scopes=scoped, rows=rows)
    for name in TABLES:
        if name in pe.LONG_TABLES:
            continue
        path = f"{stem}{OUTPUT_MARK}{name}.csv"
        if name in ("axons", "log"):
            pe.write_flat_table(path, name, tables[name])      # with its _columns.csv
        else:
            _write_csv(path, tables[name])
    return stem


def process_folder(folder: str, out_stem: str, params: pdt.PerimeterParams = pdt.PerimeterParams(), *,
                   pattern: str = "axon", animal: str = "", year: str = "",
                   contour_fn: Optional[ContourFn] = None, files: Optional[Sequence[str]] = None,
                   identities: Optional[Mapping[str, Any]] = None,
                   pixel_sizes: Optional[Mapping[str, float]] = None,
                   cancelled: Optional[Callable[[], bool]] = None) -> Sequence[pdt.PerimeterResult]:
    """Analyse every localization file of ``folder`` and write the tables at a free stem derived from ``out_stem``.

    ``pattern``: substring filter on file names (``mps_io.find_localization_files``; "" takes every file). The
    program's own outputs and this batch's tables are skipped. ``animal``: written into every identity (e.g.
    ``mps_perimeter_pool.PROVISIONAL_ANIMAL`` for a folder known to be one animal). Returns the results, in file
    order; the stem actually used is on ``process_folder.last_stem``.

    ``files``: analyse exactly these paths, in this order, instead of listing the folder (``pattern`` is then not
    used). ``identities`` / ``pixel_sizes`` / ``cancelled``: see ``analyse_files``; a cancelled run writes NOTHING
    (the partial results are returned).
    """
    from tools.mps_io import find_localization_files

    if files is None:
        found, _skipped = find_localization_files(folder, pattern=pattern)
    else:
        found = list(files)
    paths = [f for f in found if OUTPUT_MARK not in os.path.basename(f).lower()]
    process_folder.last_stem = None  # type: ignore[attr-defined]
    if not paths:
        return []                                        # nothing to write
    results = analyse_files(paths, params, animal=animal, year=year, contour_fn=contour_fn, identities=identities,
                            pixel_sizes=pixel_sizes, cancelled=cancelled)
    if cancelled is not None and cancelled():
        return results
    process_folder.last_stem = write_tables(results, out_stem)  # type: ignore[attr-defined]
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
