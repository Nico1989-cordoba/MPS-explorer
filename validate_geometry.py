# -*- coding: utf-8 -*-
"""
Validation harness for tools/mps_geometry.py against real axon data.

Chains the full pipeline built so far:
  Z-GMM slab (step 2) -> DBSCAN -> automatic bad-cluster removal (step 1)
  -> perimeter / cluster area / effective radius (step 3)

and compares against the paper's reference values.

Run:  python validate_geometry.py
"""

from __future__ import annotations

import os
import sys

import h5py as h5
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.cluster_quality import (  # noqa: E402
    PolygonROI, identify_bad_clusters, good_cluster_centroids,
)
from tools.mps_periodicity import fit_z_periodicity, select_mps_slab  # noqa: E402
from tools.mps_geometry import (  # noqa: E402
    reconstruct_perimeter, compute_cluster_areas,
    PAPER_MEDIAN_CLUSTER_AREA_NM2, PAPER_MEDIAN_R_EFF_NM,
)

PXSIZE_NM = 133.0
EPS_NM, MIN_SAMPLES = 25.0, 10
EXAMPLE_DIR = "example data"
FILES = [
    "ROI6_B2spectrin_unified_locs_rcc_apicked3_filter.hdf5",
    "ROI6_B2spectrin_unified_locs_rcc_apicked6_filter.hdf5",
    "ROI6_B2spectrin_unified_locs_rcc_apicked_4_filter.hdf5",
    "ROI7_B2spectrin_unified_locs_rcc_apicked1_filter.hdf5",
    "ROI7_B2spectrin_unified_locs_rcc_apicked3_filter.hdf5",
]


def run_one(filename: str):
    with h5.File(os.path.join(EXAMPLE_DIR, filename), "r") as f:
        ds = f["locs"]
        x = np.asarray(ds["x"], float) * PXSIZE_NM
        y = np.asarray(ds["y"], float) * PXSIZE_NM
        z = np.asarray(ds["z"], float)

    print(f"\n{'=' * 78}\n{filename}\n{'=' * 78}")

    per = fit_z_periodicity(z)
    slab = select_mps_slab(z, per.main_peak_nm)
    xs, ys = x[slab], y[slab]

    labels = DBSCAN(eps=EPS_NM, min_samples=MIN_SAMPLES).fit(
        np.column_stack([xs, ys])).labels_
    n_raw = len(set(labels.tolist()) - {-1})
    if n_raw < 3:
        print(f"  only {n_raw} clusters - skipping (cannot close a contour)")
        return None

    hull = ConvexHull(np.column_stack([xs, ys]))
    roi = PolygonROI(vertices=np.column_stack([xs, ys])[hull.vertices])
    rep = identify_bad_clusters(xs, ys, labels, roi, edge_margin_nm=EPS_NM)
    cms = good_cluster_centroids(xs, ys, labels, rep.bad_labels)

    print(f"  clusters raw / kept  : {n_raw} / {len(cms)}")

    # --- perimeter -----------------------------------------------------
    pr = reconstruct_perimeter(cms)
    print(f"  perimeter            : {pr.perimeter_nm:,.0f} nm "
          f"({pr.perimeter_um:.2f} um)")
    print(f"  clusters per um      : {pr.clusters_per_um:.2f}"
          f"   [paper slope: 4.08]")
    exp_n = pr.expected_n_clusters_from_paper()
    print(f"  N predicted by paper : {exp_n:.1f}   vs measured {pr.n_clusters}"
          f"   (diff {pr.n_clusters - exp_n:+.1f})")
    print(f"  2-opt improvements   : {pr.n_2opt_improvements}"
          f"   self-intersections {pr.self_intersections_before} -> "
          f"{pr.self_intersections_after}")

    # --- areas ---------------------------------------------------------
    ar = compute_cluster_areas(xs, ys, labels, exclude_labels=rep.bad_labels)
    a, r = ar.areas_nm2, ar.r_eff_nm
    print(f"  cluster area   nm^2  : median {ar.median_area_nm2:,.0f}"
          f"   [paper: {PAPER_MEDIAN_CLUSTER_AREA_NM2:,.0f}]")
    print(f"      quartiles        : Q1 {np.percentile(a,25):,.0f} | "
          f"Q3 {np.percentile(a,75):,.0f} | max {a.max():,.0f}")
    print(f"  r_eff          nm    : median {ar.median_r_eff_nm:.1f}"
          f"   [paper: {PAPER_MEDIAN_R_EFF_NM:.0f}]")
    print(f"      quartiles        : Q1 {np.percentile(r,25):.1f} | "
          f"Q3 {np.percentile(r,75):.1f} | max {r.max():.1f}")
    print(f"  locs per cluster     : median {np.median(ar.n_locs):.0f} "
          f"(min {ar.n_locs.min()}, max {ar.n_locs.max()})")

    for w in pr.warnings + ar.warnings:
        print(f"  ! {w}")

    return dict(areas=a, r_eff=r, perim_um=pr.perimeter_um, n=pr.n_clusters)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    all_areas, all_r, pts = [], [], []
    for fn in FILES:
        if os.path.exists(os.path.join(EXAMPLE_DIR, fn)):
            res = run_one(fn)
            if res:
                all_areas.append(res["areas"])
                all_r.append(res["r_eff"])
                pts.append((res["perim_um"], res["n"]))

    if all_areas:
        A = np.concatenate(all_areas)
        R = np.concatenate(all_r)
        print(f"\n{'=' * 78}\nPOOLED across example axons\n{'=' * 78}")
        print(f"  clusters pooled      : {A.size}")
        print(f"  area median          : {np.median(A):,.0f} nm^2"
              f"   [paper 1,965]")
        print(f"  r_eff median         : {np.median(R):.1f} nm   [paper ~25]")

        if len(pts) >= 3:
            P = np.array(pts)
            slope, intercept = np.polyfit(P[:, 0], P[:, 1], 1)
            rr = np.corrcoef(P[:, 0], P[:, 1])[0, 1]
            print(f"  N vs perimeter fit   : y = {slope:.2f}x {intercept:+.2f}"
                  f"   r = {rr:.2f}")
            print(f"    paper              : y = 4.08x -12.62   r = 0.81")
            print(f"    (only {len(pts)} axons here -- the paper used 34)")
