# -*- coding: utf-8 -*-
"""
Validation harness for tools/mps_periodicity.py against real axon data.

A consistency check of parameter 7 (axial periodicity Delta-Z) on the five
example regions shipped with the repository, not a calibration: each
region's mean Delta-Z is printed with its own standard error and its signed
distance, in spread units, from every published period of the reference
table (``tools.mps_periodicity.REFERENCE_PERIODS``; Gazal et al. 2026, 170
+/- 15 nm along z in mouse sciatic nerve, the nature of the +/- not
stated). A spread of pooled spacings is not a tolerance -- about a third of
healthy spacings lie outside one SD -- so no WITHIN / OUTSIDE label is
attached. It also reports the automatically selected 180 nm slab that
replaces the manual zmin/zmax fields.

z does not depend on the pixel size in Picasso files (z is in nm), so the
pixel size affects nothing here; it is read from the file's metadata and
the assumption is printed when there is none.

Run:  python validate_periodicity.py
"""

from __future__ import annotations

import os
import sys

import h5py as h5
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_io import read_pixel_size  # noqa: E402
from tools.mps_periodicity import (  # noqa: E402
    delta_z_summary,
    fit_z_periodicity,
    select_mps_slab,
    slab_bounds,
)

PXSIZE_NM = 133.0  # assumed for this example data when the file carries none

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
    try:
        px = read_pixel_size(path)
    except Exception:  # noqa: BLE001 - a missing sidecar is the case printed
        px = None

    print(f"\n{'=' * 78}\n{filename}\n{'=' * 78}")
    print(f"  pixel size           : "
          + (f"{px:g} nm (from the file's metadata)" if px else f"assumed {PXSIZE_NM:g} nm (no metadata)")
          + "; z is in nm and does not depend on it")
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
        summary = delta_z_summary(res.delta_z_nm)
        se = "" if summary.se_nm is None else f" +/- {summary.se_nm:.1f} nm (SE)"
        print(f"  mean Delta-Z         : {summary.mean_nm:.1f} nm{se}")
        for line in summary.lines():
            print(f"      {line}")
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
        pooled = delta_z_summary(a)
        print(f"  SE       : {pooled.se_nm:.1f} nm" if pooled.se_nm is not None else "  SE       : -")
        for line in pooled.lines():
            print(f"  {line}")
        print("  (a consistency check on the shipped example regions, not a calibration)")
