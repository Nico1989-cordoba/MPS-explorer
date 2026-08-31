# -*- coding: utf-8 -*-
"""
Validation of tools/mps_multisegment.py on the 18 real axons.

Runs the per-segment analysis on EVERY MPS segment of each axon and
compares consecutive segments -- item (e) of the thesis plan's Figure 3.

Reports both segmentation modes side by side, because the choice between
them changes which localizations each segment owns.

Run:  python validate_multisegment.py
"""

from __future__ import annotations

import glob
import os
import sys

import h5py as h5
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_multisegment import analyze_all_segments  # noqa: E402

DATA_ROOT = os.environ.get(
    "MPS_VALIDATION_DATA",
    r"C:\Users\nicol\OneDrive\Doctorado\1°Reunión de avances de tesis\Abril",
)


def read_pixelsize(path: str) -> float:
    y = os.path.splitext(path)[0] + ".yaml"
    for line in open(y, encoding="utf-8", errors="ignore"):
        if line.strip().startswith("Pixelsize:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"no Pixelsize for {path}")


def find_axons():
    out = []
    for pat in ["ROI 1/*.hdf5", "ROI 1/*/*.hdf5", "ROI 2/*.hdf5"]:
        for f in glob.glob(os.path.join(DATA_ROOT, pat)):
            if "axon" in os.path.basename(f).lower():
                out.append(f)
    return sorted(set(out))


def label(path: str) -> str:
    roi = "ROI1" if "ROI 1" in path else "ROI2"
    return roi + "/" + os.path.basename(path).split("icked")[-1][:10]


def run_mode(files, mode):
    print(f"\n{'=' * 100}")
    print(f"MODE = {mode}")
    print(f"{'=' * 100}")
    print(f"{'axon':<20} {'seg':>3} {'z_c':>7} {'locs':>7} {'N':>4} "
          f"{'perim_um':>9} {'occ%':>6} {'1NN':>6}")
    print("-" * 100)

    all_pairs = []
    n_seg_total = 0
    for f in files:
        px = read_pixelsize(f)
        with h5.File(f, "r") as h:
            ds = h["locs"]
            x = np.asarray(ds["x"], float) * px
            y = np.asarray(ds["y"], float) * px
            z = np.asarray(ds["z"], float)

        ms = analyze_all_segments(
            x, y, z, source_name=f, mode=mode,
            pixel_size_nm=px, pixel_size_source="yaml",
        )
        n_seg_total += ms.n_analyzed

        for seg, an in zip(ms.segments, ms.analyses):
            if an is None:
                print(f"{label(f):<20} {seg.index:>3} {seg.center_nm:>7.0f} "
                      f"{seg.n_locs:>7,}  (analysis failed)")
                continue
            fmt = lambda v, n=1: "n/a" if v is None else f"{v:,.{n}f}"
            print(f"{label(f):<20} {seg.index:>3} {seg.center_nm:>7.0f} "
                  f"{seg.n_locs:>7,} {an.n_clusters_kept:>4} "
                  f"{fmt(an.perimeter_um, 2):>9} "
                  f"{fmt(an.occupancy_percent, 1):>6} "
                  f"{fmt(an.median_1nn_nm, 0):>6}")
        all_pairs.extend(ms.pair_rows())

    print(f"\n  segments analysed: {n_seg_total} across {len(files)} axons")
    print(f"  consecutive pairs : {len(all_pairs)}")

    # ---- consecutive-segment relationship ----
    ang = np.array([p["angular_nn_median_deg"] for p in all_pairs
                    if p["angular_nn_median_deg"] is not None], float)
    frac = np.array([p["rotation_fraction_of_spacing"] for p in all_pairs
                     if p["rotation_fraction_of_spacing"] is not None], float)
    mx = np.array([p["max_correlation"] for p in all_pairs
                   if p["max_correlation"] is not None], float)
    c0 = np.array([p["correlation_at_zero"] for p in all_pairs
                   if p["correlation_at_zero"] is not None], float)
    spa = np.array([p["mean_angular_spacing_deg"] for p in all_pairs
                    if p["mean_angular_spacing_deg"] is not None], float)

    print(f"\n  {'-' * 70}")
    print(f"  CONSECUTIVE-SEGMENT RELATIONSHIP (n = {ang.size} pairs)")
    print(f"  {'-' * 70}")
    if ang.size:
        print(f"  mean angular spacing within a segment : "
              f"{np.median(spa):.1f} deg (median)")
        print(f"  angular NN offset A->B               : "
              f"median {np.median(ang):.1f} deg, IQR "
              f"{np.percentile(ang,25):.1f}-{np.percentile(ang,75):.1f}")
        print(f"     as a fraction of the spacing      : "
              f"{np.median(ang)/np.median(spa):.3f}")
        print(f"  rotation, folded [0=aligned, 0.5=staggered]:")
        print(f"     median {np.median(frac):.3f}   IQR "
              f"{np.percentile(frac,25):.3f}-{np.percentile(frac,75):.3f}")
        print(f"  cross-correlation max                : median {np.median(mx):.3f}")
        print(f"  cross-correlation at 0 rotation      : median {np.median(c0):.3f}")

        # Reference: what would a random angular arrangement give?
        rng = np.random.default_rng(0)
        sim = []
        for p in all_pairs:
            na, nb = p["n_clusters_a"], p["n_clusters_b"]
            if na < 3 or nb < 3:
                continue
            a = rng.uniform(0, 2 * np.pi, na)
            b = rng.uniform(0, 2 * np.pi, nb)
            from tools.mps_multisegment import angular_nn_offsets_deg
            sim.append(np.median(angular_nn_offsets_deg(a, b)))
        sim = np.array(sim, float)
        if sim.size:
            print(f"\n  reference, angles drawn uniformly at random:")
            print(f"     angular NN offset median          : {np.median(sim):.1f} deg")
            print(f"     (measured {np.median(ang):.1f} deg vs random "
                  f"{np.median(sim):.1f} deg)")
    return all_pairs


if __name__ == "__main__":
    files = find_axons()
    print(f"Found {len(files)} axon files")
    for mode in ("paper", "partition"):
        run_mode(files, mode)
