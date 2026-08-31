# -*- coding: utf-8 -*-
"""
Full-pipeline validation over the 18 real axons of varied diameter in
ROI 1 / ROI 2, using the pixel size read from each file's Picasso YAML.

Runs steps 1-4 end to end and checks every reference value the paper
gives that is reachable so far:
    Delta-Z            170 +/- 15 nm
    cluster area       median 1965 nm^2
    r_eff              median ~25 nm
    1NN pooled peak    ~200 nm
    1NN median of medians  260 nm
    N vs perimeter     y = 4.08x - 12.62, r = 0.81   <-- needs varied calibres

Run:  python validate_full_18axons.py
"""

from __future__ import annotations

import glob
import os
import sys

import h5py as h5
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.cluster_quality import (  # noqa: E402
    PolygonROI, identify_bad_clusters, good_cluster_centroids,
)
from tools.mps_periodicity import fit_z_periodicity, select_mps_slab  # noqa: E402
from tools.mps_geometry import reconstruct_perimeter, compute_cluster_areas  # noqa: E402
from tools.mps_spatial import compute_nn_distances, pool_nn_across_axons  # noqa: E402

# Root of the validation dataset (18 axon cross-sections of varied calibre).
# Override without editing this file:
#     set MPS_VALIDATION_DATA=D:\path\to\axons     (Windows)
#     export MPS_VALIDATION_DATA=/path/to/axons    (POSIX)
DATA_ROOT = os.environ.get(
    "MPS_VALIDATION_DATA",
    r"C:\Users\nicol\OneDrive\Doctorado\1°Reunión de avances de tesis\Abril",
)
EPS_NM, MIN_SAMPLES = 25.0, 10


def read_pixelsize(hdf5_path: str) -> float:
    """Pixel size in nm from the Picasso YAML sidecar (same logic as the GUI)."""
    y = os.path.splitext(hdf5_path)[0] + ".yaml"
    if os.path.exists(y):
        for line in open(y, encoding="utf-8", errors="ignore"):
            if line.strip().startswith("Pixelsize:"):
                return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"No Pixelsize in YAML for {hdf5_path}")


def find_axons():
    out = []
    for pat in ["ROI 1/*.hdf5", "ROI 1/*/*.hdf5", "ROI 2/*.hdf5"]:
        for f in glob.glob(os.path.join(DATA_ROOT, pat)):
            if "axon" in os.path.basename(f).lower():
                out.append(f)
    return sorted(set(out))


def label_of(path: str) -> str:
    d = os.path.basename(os.path.dirname(path))
    b = os.path.basename(path)
    import re
    m = re.search(r"[Aa]xon\s*_?(\d+)", b)
    n = m.group(1) if m else "?"
    roi = "ROI1" if "ROI 1" in path else "ROI2"
    return f"{roi}/axon{n}"


def run(path: str):
    px = read_pixelsize(path)
    with h5.File(path, "r") as f:
        ds = f["locs"]
        x = np.asarray(ds["x"], float) * px
        y = np.asarray(ds["y"], float) * px
        z = np.asarray(ds["z"], float)

    per = fit_z_periodicity(z)
    slab = select_mps_slab(z, per.main_peak_nm)
    xs, ys = x[slab], y[slab]
    if xs.size < MIN_SAMPLES:
        return None

    labels = DBSCAN(eps=EPS_NM, min_samples=MIN_SAMPLES).fit(
        np.column_stack([xs, ys])).labels_
    n_raw = len(set(labels.tolist()) - {-1})
    if n_raw < 3:
        return None

    pts = np.column_stack([xs, ys])
    hull = ConvexHull(pts)
    roi = PolygonROI(vertices=pts[hull.vertices])
    rep = identify_bad_clusters(xs, ys, labels, roi, edge_margin_nm=EPS_NM)
    cms = good_cluster_centroids(xs, ys, labels, rep.bad_labels)
    if len(cms) < 3:
        return None

    pr = reconstruct_perimeter(cms)
    ar = compute_cluster_areas(xs, ys, labels, exclude_labels=rep.bad_labels)
    nn = compute_nn_distances(cms, k=1)

    return dict(
        label=label_of(path), px=px, n_locs=len(x), n_slab=int(slab.sum()),
        dz=list(per.delta_z_nm), peak=per.main_peak_nm,
        n_raw=n_raw, n_kept=len(cms),
        perim_um=pr.perimeter_um, xi=pr.self_intersections_after,
        areas=ar.areas_nm2, r_eff=ar.r_eff_nm,
        nn1=nn.first_nn_nm, nn_med=nn.median_1nn_nm,
        guard=rep.edge_criterion_disabled is not None,
        warn=per.warnings,
    )


if __name__ == "__main__":
    axons = find_axons()
    print(f"Found {len(axons)} axon files\n")

    rows = []
    for p in axons:
        try:
            r = run(p)
        except Exception as e:
            print(f"  ERROR {label_of(p)}: {e}")
            continue
        if r is None:
            print(f"  skipped {label_of(p)} (too few clusters)")
            continue
        rows.append(r)

    print(f"\n{'axon':<12} {'px':>4} {'locs':>7} {'slab':>6} {'Nraw':>5} "
          f"{'Nkeep':>6} {'perim_um':>9} {'1NNmed':>7} {'area_med':>9} {'reff':>6} {'dZ':>7}")
    print("-" * 92)
    for r in rows:
        dz = np.mean(r["dz"]) if len(r["dz"]) else float("nan")
        print(f"{r['label']:<12} {r['px']:>4.0f} {r['n_locs']:>7,} {r['n_slab']:>6,} "
              f"{r['n_raw']:>5} {r['n_kept']:>6} {r['perim_um']:>9.2f} "
              f"{r['nn_med']:>7.0f} {np.median(r['areas']):>9,.0f} "
              f"{np.median(r['r_eff']):>6.1f} {dz:>7.0f}")

    if not rows:
        sys.exit("no usable axons")

    # ---------------- pooled statistics ----------------
    print(f"\n{'=' * 92}\nPOOLED over {len(rows)} axons\n{'=' * 92}")

    A = np.concatenate([r["areas"] for r in rows])
    R = np.concatenate([r["r_eff"] for r in rows])
    print(f"  cluster area median   : {np.median(A):>9,.0f} nm^2   [paper 1,965]")
    print(f"  r_eff median          : {np.median(R):>9.1f} nm     [paper ~25]")

    pooled = pool_nn_across_axons([r["nn1"] for r in rows], peak_method="kde")
    pooled_h = pool_nn_across_axons([r["nn1"] for r in rows], peak_method="histogram")
    print(f"  1NN pooled peak (KDE) : {pooled.peak_nm:>9.0f} nm     [paper ~200]")
    print(f"  1NN pooled peak (hist): {pooled_h.peak_nm:>9.0f} nm     [paper ~200]")
    print(f"  1NN median of medians : {pooled.median_of_medians_nm:>9.0f} nm     [paper 260]")
    print(f"  per-axon 1NN medians  : {np.min(pooled.per_axon_medians_nm):.0f} .. "
          f"{np.max(pooled.per_axon_medians_nm):.0f} nm")

    alldz = np.concatenate([r["dz"] for r in rows if len(r["dz"])])
    print(f"  Delta-Z pooled        : median {np.median(alldz):.0f} nm, "
          f"mean {alldz.mean():.0f} +/- {alldz.std(ddof=1):.0f}   [paper 170 +/- 15]")

    # ---------------- N vs perimeter regression ----------------
    P = np.array([[r["perim_um"], r["n_kept"]] for r in rows], float)
    print(f"\n  perimeter range       : {P[:,0].min():.2f} .. {P[:,0].max():.2f} um"
          f"   (spread {P[:,0].max()-P[:,0].min():.2f} um)")
    if len(P) >= 3 and np.ptp(P[:, 0]) > 0:
        slope, intercept = np.polyfit(P[:, 0], P[:, 1], 1)
        rr = np.corrcoef(P[:, 0], P[:, 1])[0, 1]
        print(f"  N vs perimeter        : y = {slope:.2f}x {intercept:+.2f}   r = {rr:.2f}")
        print(f"  paper                 : y = 4.08x -12.62   r = 0.81")

    # ---------------- occupancy-independent constancy check ----------------
    med = pooled.per_axon_medians_nm
    rr2 = np.corrcoef(P[:, 0], med)[0, 1] if len(P) > 2 else float("nan")
    print(f"\n  1NN median vs perimeter correlation: r = {rr2:+.2f}")
    print(f"    (paper's claim: per-axon 1NN median is ~constant regardless of"
          f" perimeter, i.e. r near 0)")

    ng = sum(1 for r in rows if r["guard"])
    if ng:
        print(f"\n  ! edge-criterion guard fired on {ng}/{len(rows)} axons")
