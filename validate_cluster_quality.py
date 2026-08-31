# -*- coding: utf-8 -*-
"""
Validation harness for tools/cluster_quality.py against real axon data.

Runs the Gazal et al. (2026) front half of the pipeline on the example
βII-spectrin ROIs shipped in this repo:
  - load Picasso HDF5, convert px -> nm
  - restrict to a 180 nm axial window centred on the main Z peak
  - DBSCAN with eps=25 nm, min_samples=10
  - apply the automatic bad-cluster criteria (edge-touching + DBCV)

Prints how many clusters each criterion removes so the thresholds can be
calibrated on real data rather than guessed.

Run:  python validate_cluster_quality.py
"""

from __future__ import annotations

import os
import sys

import h5py as h5
import numpy as np
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.cluster_quality import (  # noqa: E402
    CircularROI,
    PolygonROI,
    identify_bad_clusters,
    good_cluster_centroids,
)

# Paper parameters (Gazal et al. 2026)
EPS_NM = 25.0
MIN_SAMPLES = 10
Z_WINDOW_HALF_NM = 90.0  # ±90 nm around the main Z peak = 180 nm window

EXAMPLE_DIR = "example data"
SPECTRIN_FILES = [
    "ROI6_B2spectrin_unified_locs_rcc_apicked3_filter.hdf5",
    "ROI6_B2spectrin_unified_locs_rcc_apicked6_filter.hdf5",
    "ROI6_B2spectrin_unified_locs_rcc_apicked_4_filter.hdf5",
    "ROI7_B2spectrin_unified_locs_rcc_apicked1_filter.hdf5",
    "ROI7_B2spectrin_unified_locs_rcc_apicked3_filter.hdf5",
]

# The example HDF5s ship without their Picasso .yaml sidecar, so the pixel
# size cannot be read automatically here. 133 nm is the value hardcoded in
# the original MPS Explorer and used in this repo's own axon analysis
# script (tools/test_DBCV_data_axons.py), so it is the right value for
# THIS example data specifically. In the GUI the real pixel size comes
# from the YAML sidecar (_get_pixel_size_from_yaml).
PXSIZE_NM = 133.0


def load_spectrin(path: str):
    with h5.File(path, "r") as f:
        ds = f["locs"]
        x = np.asarray(ds["x"], dtype=float) * PXSIZE_NM
        y = np.asarray(ds["y"], dtype=float) * PXSIZE_NM
        z = np.asarray(ds["z"], dtype=float)  # Picasso z is already in nm
    return x, y, z


def main_z_peak(z: np.ndarray) -> float:
    """Crude main-peak estimate via histogram mode (the real pipeline will
    use the GMM; this is only to get a realistic slab for THIS test)."""
    hist, edges = np.histogram(z, bins=100)
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def run_one(filename: str) -> None:
    path = os.path.join(EXAMPLE_DIR, filename)
    x, y, z = load_spectrin(path)

    peak = main_z_peak(z)
    slab = (z >= peak - Z_WINDOW_HALF_NM) & (z <= peak + Z_WINDOW_HALF_NM)
    xs, ys = x[slab], y[slab]

    print(f"\n{'=' * 78}\n{filename}\n{'=' * 78}")
    print(f"  total locs           : {len(x):,}")
    print(f"  main Z peak          : {peak:8.1f} nm")
    print(f"  locs in 180nm slab   : {len(xs):,}")
    if len(xs) < MIN_SAMPLES:
        print("  !! too few localizations in slab, skipping")
        return

    labels = DBSCAN(eps=EPS_NM, min_samples=MIN_SAMPLES).fit(
        np.column_stack([xs, ys])
    ).labels_
    n_clusters = len({int(l) for l in labels if l != -1})
    n_noise = int(np.sum(labels == -1))
    print(f"  DBSCAN eps={EPS_NM:g} minPts={MIN_SAMPLES}")
    print(f"    clusters found     : {n_clusters}")
    print(f"    noise points       : {n_noise:,} ({100*n_noise/len(xs):.1f}%)")

    if n_clusters == 0:
        print("  !! no clusters, skipping quality analysis")
        return

    # Build an ROI description. The example files are already cropped
    # ("apicked") axon sub-ROIs rather than a GUI-drawn shape, so use a
    # convex-hull polygon of the slab points as a stand-in boundary --
    # this is the most faithful proxy for "the edge of the loaded data".
    from scipy.spatial import ConvexHull

    hull = ConvexHull(np.column_stack([xs, ys]))
    roi = PolygonROI(vertices=np.column_stack([xs, ys])[hull.vertices])

    report = identify_bad_clusters(xs, ys, labels, roi,
                                   edge_margin_nm=EPS_NM,
                                   dbcv_threshold=0.0)

    print(f"  Automatic bad-cluster removal:")
    print(f"    edge-touching      : {len(report.edge_touching):3d}")
    print(f"    low DBCV (<0)      : {len(report.low_dbcv):3d}")
    print(f"    union (removed)    : {len(report.bad_labels):3d}")
    print(f"    surviving clusters : {n_clusters - len(report.bad_labels):3d}"
          f"  ({100*(n_clusters-len(report.bad_labels))/n_clusters:.0f}% kept)")

    if report.dbcv_scores:
        vals = np.array(list(report.dbcv_scores.values()))
        print(f"    DBCV score range   : [{vals.min():+.3f}, {vals.max():+.3f}]"
              f"  median {np.median(vals):+.3f}")
    else:
        print(f"    DBCV scores        : unavailable (see compute_dbcv_per_cluster)")

    cms = good_cluster_centroids(xs, ys, labels, report.bad_labels)
    print(f"    good centroids     : {len(cms)}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for fn in SPECTRIN_FILES:
        if os.path.exists(os.path.join(EXAMPLE_DIR, fn)):
            run_one(fn)
        else:
            print(f"missing: {fn}")
