# -*- coding: utf-8 -*-
"""
Run the every-segment (ring) analysis over a folder of axons and write two
CSVs: one row per segment, one row per consecutive segment pair.

This is the path meant for a real dataset, where clicking through the GUI
one axon at a time stops being practical.

    python batch_rings.py DATA_ROOT [-o OUT_DIR] [--mode valley|paper|partition]
                          [--guard 0] [--eps 25] [--min-samples 10]
                          [--pattern axon] [--pixel-size 113]
                          [--animal-pattern REGEX] [--genotype-pattern REGEX]

Metadata for the statistics (animal, genotype) is NOT guessed: pass the
regexes that match your own folder naming, or the exported rows carry
empty fields and the summary says so. Nothing downstream can recover an
animal identity that was never supplied, and a comparison that ignores
which animal an axon came from is pseudoreplication -- see the module
docstring of tools/mps_batch.py.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_batch import (  # noqa: E402
    RingRecord,
    export_ring_csvs,
    parse_metadata,
    ring_batch_summary,
    segment_nesting,
)
from tools.mps_gaps import analyze_rings  # noqa: E402
from tools.mps_io import (  # noqa: E402
    duplicate_sources,
    find_localization_files,
    load_localizations,
)
from tools.mps_multisegment import analyze_all_segments  # noqa: E402

# Per-segment parameters worth a nesting diagnostic: one geometric, one
# about the gaps themselves, one a plain count.
NESTED_PARAMETERS = ("occupancy_percent", "median_gap_nm", "n_patches")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch ring/gap analysis over a folder of axons.")
    p.add_argument("data_root", help="folder to walk (recursively)")
    p.add_argument("-o", "--out-dir", default=".",
                   help="where to write the CSVs (default: current folder)")
    p.add_argument("--mode", default="valley",
                   choices=("valley", "paper", "partition"),
                   help="how to bound each segment's axial slab")
    p.add_argument("--guard", type=float, default=0.0,
                   help="dead zone at each boundary, in nm (bleed-through "
                        "control)")
    p.add_argument("--eps", type=float, default=25.0,
                   help="DBSCAN eps in nm (paper: 25)")
    p.add_argument("--min-samples", type=int, default=10,
                   help="DBSCAN min_samples (paper: 10)")
    p.add_argument("--half-width", type=float, default=90.0,
                   help="slab half-width in nm (paper: 90)")
    p.add_argument("--pattern", default="axon",
                   help="only files whose name contains this (default: axon)")
    p.add_argument("--include-derived", action="store_true",
                   help="also analyse the CSVs MPS Explorer writes itself "
                        "(cluster centres, filtered clusters). They are "
                        "already processed, and some carry the same column "
                        "headers as raw exports, so this is almost never "
                        "what you want.")
    p.add_argument("--pixel-size", type=float, default=None,
                   help="nm per camera pixel, for Picasso files with no YAML "
                        "sidecar. Applies to every such file, so use it only "
                        "when they share one acquisition.")
    p.add_argument("--animal-pattern", default=None,
                   help=r"regex with one group capturing the animal id, e.g. "
                        r"'(M\d+)_'")
    p.add_argument("--genotype-pattern", default=None,
                   help=r"regex with one group capturing the genotype, e.g. "
                        r"'_(WT|KO)'")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if not os.path.isdir(args.data_root):
        print(f"Not a folder: {args.data_root}")
        return 1
    files, skipped = find_localization_files(
        args.data_root, pattern=args.pattern,
        exclude_derived=not args.include_derived)
    if not files:
        print(f"No localization files matching {args.pattern!r} under "
              f"{args.data_root}")
        return 1
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"{len(files)} file(s) under {args.data_root}")
    if skipped:
        print(f"{len(skipped)} skipped as derived output (cluster centres, "
              f"ROI exports, neighbour distances, filtered clusters, and the "
              f"Picasso tools' cluster, molecule-map and link files); "
              f"--include-derived to analyse them anyway")

    # A derived suffix nobody has added to DERIVED_SUFFIXES yet shows up
    # here as two files claiming the same acquisition. Analysing both
    # counts one axon twice, which inflates every n and makes the nesting
    # statistics below describe the duplication rather than the biology.
    collisions = duplicate_sources(files)
    if collisions:
        print(f"\n! {len(collisions)} acquisition(s) appear more than once "
              f"among the files to analyse. Each will be counted as a "
              f"separate axon, so every n below is inflated:")
        for stem, paths in sorted(collisions.items()):
            print(f"    {stem}")
            for path in paths:
                print(f"      {os.path.basename(path)}")
        print("  Use --pattern to narrow the selection, or move the extra "
              "copies out of the folder.\n")
    print(f"mode={args.mode} guard={args.guard:g} nm eps={args.eps:g} "
          f"min_samples={args.min_samples} half_width={args.half_width:g}\n")

    records: List[RingRecord] = []
    started = time.time()
    for i, path in enumerate(files, 1):
        meta = parse_metadata(
            path,
            animal_pattern=args.animal_pattern,
            genotype_pattern=args.genotype_pattern,
        )
        name = os.path.basename(path)
        print(f"[{i}/{len(files)}] {name[:58]:<58}", end="", flush=True)

        try:
            loc = load_localizations(path, pixel_size_nm=args.pixel_size)
        except Exception as exc:                          # noqa: BLE001
            print(f"  LOAD FAILED: {exc}")
            records.append(RingRecord(metadata=meta, error=f"load: {exc}"))
            continue

        try:
            ms = analyze_all_segments(
                loc.x_nm, loc.y_nm, loc.z_nm,
                source_name=path, mode=args.mode, guard_nm=args.guard,
                half_width_nm=args.half_width,
                eps_nm=args.eps, min_samples=args.min_samples,
                pixel_size_nm=loc.pixel_size_nm,
                pixel_size_source=loc.pixel_size_source,
            )
            rings = analyze_rings(ms)
        except Exception as exc:                          # noqa: BLE001
            print(f"  ANALYSIS FAILED: {exc}")
            records.append(RingRecord(metadata=meta, error=f"analysis: {exc}"))
            continue

        records.append(RingRecord(metadata=meta, ms=ms, rings=rings))
        print(f"  {loc.n:>8,} locs  {ms.n_analyzed}/{ms.n_segments} seg  "
              f"{len(rings.pairs)} pair(s)")

    seg_path = os.path.join(args.out_dir, "mps_rings_segments.csv")
    pair_path = os.path.join(args.out_dir, "mps_rings_pairs.csv")
    n_seg, n_pair = export_ring_csvs(records, seg_path, pair_path)

    print(f"\n{'=' * 78}")
    print(ring_batch_summary(records))
    print(f"\nwrote {n_seg} segment row(s) -> {seg_path}")
    print(f"wrote {n_pair} pair row(s)    -> {pair_path}")
    print(f"elapsed {time.time() - started:.0f} s")

    print(f"\n{'=' * 78}")
    print("NESTING: do the segments of one axon behave like replicates?")
    print("Segments from the same axon share an acquisition, so a comparison")
    print("made over segments can claim more evidence than the data holds.")
    print(f"{'=' * 78}")
    for parameter in NESTED_PARAMETERS:
        diag = segment_nesting(records, parameter)
        print(f"\n{parameter}")
        if diag.icc is None:
            print(f"  no ICC: {'; '.join(diag.warnings) or 'not computable'}")
        else:
            print(f"  ICC = {diag.icc:.2f} ({diag.severity}) over "
                  f"{diag.n_observations} segments in {diag.n_groups} axons "
                  f"(about {diag.mean_group_size:.1f} each)")
            if diag.design_effect is not None and diag.effective_n is not None:
                print(f"  design effect {diag.design_effect:.1f} -> worth "
                      f"about {diag.effective_n:.1f} independent segments")
            if diag.null_p95 is not None:
                print(f"  this design reaches ICC {diag.null_p95:.2f} by "
                      f"chance (95th percentile)")
        for w in diag.warnings:
            print(f"  ! {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
