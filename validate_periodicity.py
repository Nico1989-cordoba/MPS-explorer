# -*- coding: utf-8 -*-
"""
Validation harness for tools/mps_periodicity.py against real axon data.

Checks parameter 7 (axial periodicity Delta-Z) against the paper's
reported 170 +/- 15 nm, and reports the automatically selected 180 nm
slab that replaces the manual zmin/zmax fields.

Run:  python validate_periodicity.py
"""

from __future__ import annotations

import os
import sys

import h5py as h5
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_periodicity import (  # noqa: E402
    fit_z_periodicity,
    select_mps_slab,
    slab_bounds,
)

PXSIZE_NM = 133.0  # confirmed by the user for this example data
PAPER_DELTA_Z = (170.0, 15.0)

EXAMPLE_DIR = "example data"
SPECTRIN_FILES = [
    "ROI6_B2spectrin_unified_locs_rcc_apicked3_filter.hdf5",
    "ROI6_B2spectrin_unified_locs_rcc_apicked6_filter.hdf5",
    "ROI6_B2spectrin_unified_locs_rcc_apicked_4_filter.hdf5",
    "ROI7_B2spectrin_unified_locs_rcc_apicked1_filter.hdf5",
    "ROI7_B2spectrin_unified_locs_rcc_apicked3_filter.hdf5",
]


def load_z(path: str) -> np.ndarray:
    with h5.File(path, "r") as f:
        return np.asarray(f["locs"]["z"], dtype=float)


def run_one(filename: str) -> list:
    path = os.path.join(EXAMPLE_DIR, filename)
    z = load_z(path)
    res = fit_z_periodicity(z)

    print(f"\n{'=' * 78}\n{filename}\n{'=' * 78}")
    print(f"  localizations        : {len(z):,}")
    print(f"  Z range              : {z.min():8.1f} .. {z.max():8.1f} nm "
          f"({z.max() - z.min():.0f} nm span)")
    print(f"  BIC by n_components  : "
          + ", ".join(f"n={n}: {b:,.0f}" for n, b in sorted(res.bic_by_n.items())))
    print(f"  chosen n_components  : {res.n_components}"
          f"   (converged: {res.converged})")
    if res.n_discarded_components:
        print(f"  components dropped   : {res.n_discarded_components} (weight <= 5%)")

    print(f"  retained components  :")
    for m, w, s in zip(res.means_nm, res.weights, res.sigmas_nm):
        print(f"      mu = {m:8.1f} nm   weight = {w:.3f}   sigma = {s:6.1f} nm")

    if res.delta_z_nm.size:
        print(f"  Delta-Z              : "
              + ", ".join(f"{d:.1f}" for d in res.delta_z_nm) + " nm")
        mean_dz = res.mean_delta_z_nm
        lo, hi = PAPER_DELTA_Z[0] - PAPER_DELTA_Z[1], PAPER_DELTA_Z[0] + PAPER_DELTA_Z[1]
        verdict = "WITHIN" if lo <= mean_dz <= hi else "OUTSIDE"
        print(f"  mean Delta-Z         : {mean_dz:.1f} nm   "
              f"[{verdict} paper's 170 +/- 15 nm]")
    else:
        print(f"  Delta-Z              : (none - single dominant component)")

    zmin, zmax = slab_bounds(res.main_peak_nm)
    mask = select_mps_slab(z, res.main_peak_nm)
    print(f"  main axial peak      : {res.main_peak_nm:.1f} nm")
    print(f"  auto slab (180 nm)   : zmin = {zmin:.1f}, zmax = {zmax:.1f}")
    print(f"  locs in slab         : {mask.sum():,} / {len(z):,} "
          f"({100 * mask.sum() / len(z):.1f}%)")

    for w in res.warnings:
        print(f"  ! WARNING: {w}")

    return list(res.delta_z_nm)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    all_dz = []
    for fn in SPECTRIN_FILES:
        if os.path.exists(os.path.join(EXAMPLE_DIR, fn)):
            all_dz.extend(run_one(fn))
        else:
            print(f"missing: {fn}")

    if all_dz:
        a = np.array(all_dz)
        print(f"\n{'=' * 78}\nPOOLED Delta-Z across all example ROIs\n{'=' * 78}")
        print(f"  n values : {a.size}")
        print(f"  mean     : {a.mean():.1f} nm")
        print(f"  median   : {np.median(a):.1f} nm")
        print(f"  std      : {a.std(ddof=1) if a.size > 1 else 0:.1f} nm")
        print(f"  range    : {a.min():.1f} .. {a.max():.1f} nm")
        print(f"  paper    : 170 +/- 15 nm")
