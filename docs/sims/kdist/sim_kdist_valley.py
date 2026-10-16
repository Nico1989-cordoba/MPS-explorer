# -*- coding: utf-8 -*-
"""
Synthetic validation of the k-distance valley (decision P-R58; docs/eps_kdistance.md).

Question
--------
On simulated transverse rings with a known truth per localization: how often does the valley of the log k-distance
density exist, how close does it land to the eps that best recovers the true clusters, and when does it not exist?
The same runs score two other automatic readings for comparison: the knee of the sorted curve (Kneedle, Satopaa et
al. 2011, in its basic offline form) and the reading the program used before P-R58 (90th percentile of the 4-th
other neighbour's distance).

Data
----
One ring of ``tools.mps_simulate_axon`` (the project's simulator: harness configuration of tools.mps_sim_harness,
ellipse 1500 x 1000 nm, 5 clusters per um along the membrane, hard core 50 nm, radial offset sd 30 nm, epitope disc
10 nm, linkage 12 nm per axis, geometric blinking 2.5 localizations per fluorophore; ring 0 of a 2-ring axon kept),
plus a uniform background in the ring's bounding box widened by 300 nm, its size set by the target unclustered share
f = background / all. Grid: lateral precision 6 / 10 / 15 / 20 nm x mean localizations per cluster 10 / 25 / 60 / 150
(fluorophores per cluster 4 / 10 / 24 / 60) x f = 0 / 0.1 / 0.3 / 0.5 / 0.7, 3 seeds each; and two negative controls:
background only (no clusters; f = 1) at three densities. min samples 10 (the analysis'), so k = 9.

Truth and score
---------------
DBSCAN(eps, min_samples = 10) on an eps grid of 5-150 nm (log spaced, 60 values); score = adjusted Rand index (ARI)
between its labels and the truth (background and noise both as label -1). eps* = the eps of the best ARI. A method's
eps is scored by ARI(eps) / ARI(eps*) ("relative ARI"; 1 = as good as the best eps) and by eps / eps*. A second
optimum, eps_sep*, maximises only the separation of clustered from background localizations (balanced accuracy of
"labelled noise" against "is background", whichever cluster a localization joins): what a valley between the two
populations can know about.

Seeds: run seed s uses simulator seed 1000 + s and background seed 2000 + s (numpy default_rng); the valley's
bootstrap uses its fixed seed (tools.mps_kdistance.BOOTSTRAP_SEED). Output: sim_kdist_valley_results.json beside this
file. Run from the repository root:

    venv\\Scripts\\python.exe docs\\sims\\kdist\\sim_kdist_valley.py [--workers 4]

Synthetic data only (R8): no real file is read.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

MIN_SAMPLES = 10
SIGMAS = (6.0, 10.0, 15.0, 20.0)
LOCS_PER_CLUSTER = (10, 25, 60, 150)
FRACTIONS = (0.0, 0.1, 0.3, 0.5, 0.7)
SEEDS = (0, 1, 2)
LOCS_PER_FLUOR = 2.5
BOX_MARGIN_NM = 300.0
EPS_GRID = np.unique(np.round(np.geomspace(5.0, 150.0, 60), 2))
CONTROL_DENSITIES_PER_UM2 = (200.0, 1000.0, 5000.0)


def ring(sigma: float, locs: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from tools.mps_sim_harness import base_config
    from tools.mps_simulate_axon import simulate_axon

    base = base_config()
    cfg = dataclasses.replace(
        base, n_rings=2, background_per_um3=0.0, clusters_per_um=5.0, d_min_nm=50.0, radial_offset_sd_nm=30.0,
        lp_lateral_samples_nm=np.array([float(sigma)]), n_fluor_mean=float(locs) / LOCS_PER_FLUOR,
        locs_per_fluor_mean=LOCS_PER_FLUOR, sigma_struct_nm=0.0, sigma_period_nm=0.0,
        axial_scale_by_bin=np.full(np.asarray(base.axial_scale_by_bin).size, 0.05))
    ax = simulate_axon(cfg, 1000 + int(seed))
    keep = np.asarray(ax.ring_true) == 0
    return (np.asarray(ax.x_nm)[keep], np.asarray(ax.y_nm)[keep], np.asarray(ax.cluster_true)[keep])


def add_background(x: np.ndarray, y: np.ndarray, lab: np.ndarray, f: float, seed: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if f <= 0:
        return x, y, lab
    n_bg = int(round(f / (1.0 - f) * x.size))
    rng = np.random.default_rng(2000 + int(seed))
    x0, x1 = x.min() - BOX_MARGIN_NM, x.max() + BOX_MARGIN_NM
    y0, y1 = y.min() - BOX_MARGIN_NM, y.max() + BOX_MARGIN_NM
    bx, by = rng.uniform(x0, x1, n_bg), rng.uniform(y0, y1, n_bg)
    return (np.concatenate([x, bx]), np.concatenate([y, by]), np.concatenate([lab, np.full(n_bg, -1)]))


def kneedle(kd_sorted: np.ndarray) -> float:
    """Basic offline Kneedle (Satopaa et al. 2011) on an increasing convex curve: the point of the normalised curve
    farthest below the chord, i.e. argmax(x_n - y_n)."""
    y = np.asarray(kd_sorted, dtype=np.float64)
    if y.size < 3 or y[-1] <= y[0]:
        return float("nan")
    xn = np.linspace(0.0, 1.0, y.size)
    yn = (y - y[0]) / (y[-1] - y[0])
    return float(y[int(np.argmax(xn - yn))])


def percentile90_old(x: np.ndarray, y: np.ndarray) -> float:
    """What 'auto' did before P-R58: 90th percentile of the distance to the 4-th other neighbour."""
    from tools.clustering import estimate_optimal_eps
    return float(estimate_optimal_eps(np.column_stack([x, y]), k=5, percentile=90))


def ari_curve(x: np.ndarray, y: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ARI against the true clusters, and the balanced accuracy of clustered-vs-unclustered (background) alone."""
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score

    pts = np.column_stack([x, y])
    out = np.empty(EPS_GRID.size)
    sep = np.full(EPS_GRID.size, np.nan)
    is_bg = truth == -1
    for i, eps in enumerate(EPS_GRID):
        lab = DBSCAN(eps=float(eps), min_samples=MIN_SAMPLES).fit_predict(pts)
        out[i] = adjusted_rand_score(truth, lab)
        if is_bg.any() and (~is_bg).any():
            noise = lab == -1
            sep[i] = 0.5 * (np.mean(noise[is_bg]) + np.mean(~noise[~is_bg]))
    return out, sep


def ari_at(x: np.ndarray, y: np.ndarray, truth: np.ndarray, eps: float) -> float:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score

    if not (eps > 0 and math.isfinite(eps)):
        return float("nan")
    lab = DBSCAN(eps=float(eps), min_samples=MIN_SAMPLES).fit_predict(np.column_stack([x, y]))
    return float(adjusted_rand_score(truth, lab))


def run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from tools import mps_kdistance as kdm

    t0 = time.time()
    if case["kind"] == "control":
        rng = np.random.default_rng(3000 + int(case["seed"]))
        area_um2 = 3.6 * 2.6        # the ring's box (3.0 x 2.0 um) widened by 300 nm
        n = int(round(case["density"] * area_um2))
        x, y = rng.uniform(0, 3600.0, n), rng.uniform(0, 2600.0, n)
        truth = np.full(n, -1)
    else:
        x, y, truth = ring(case["sigma"], case["locs"], case["seed"])
        x, y, truth = add_background(x, y, truth, case["f"], case["seed"])
    kd = kdm.k_distances(x, y, MIN_SAMPLES)
    valley = kdm.find_valley(kd)
    out: Dict[str, Any] = dict(case)
    out.update(n_locs=int(x.size), f_true=float(np.mean(truth == -1)),
               valley_found=bool(valley.found), valley_nm=valley.eps_nm if valley.found else None,
               valley_low_nm=valley.low_nm if valley.found else None,
               valley_high_nm=valley.high_nm if valley.found else None,
               valley_presence=valley.presence, valley_n_modes=valley.n_modes, valley_dip_z=valley.dip_z,
               valley_reason=valley.reason)
    knee = kneedle(kdm.sorted_curve(kd))
    p90 = percentile90_old(x, y)
    out.update(kneedle_nm=knee, p90_old_nm=p90)
    if case["kind"] == "control":
        out["seconds"] = time.time() - t0
        return out
    curve, sep = ari_curve(x, y, truth)
    if np.isfinite(sep).any():
        out["eps_sep_star_nm"] = float(EPS_GRID[int(np.nanargmax(sep))])
        out["sep_best"] = float(np.nanmax(sep))
    best = int(np.argmax(curve))
    ari_best = float(curve[best])
    out.update(eps_star_nm=float(EPS_GRID[best]), ari_best=ari_best,
               eps_good_low_nm=float(EPS_GRID[curve >= 0.9 * ari_best].min()),
               eps_good_high_nm=float(EPS_GRID[curve >= 0.9 * ari_best].max()))
    for name, eps in (("valley", valley.eps_nm if valley.found else float("nan")), ("kneedle", knee),
                      ("p90_old", p90), ("gazal25", 25.0)):
        a = ari_at(x, y, truth, eps)
        out[f"{name}_rel_ari"] = (a / ari_best) if (ari_best > 0 and math.isfinite(a)) else None
        out[f"{name}_ratio"] = (eps / out["eps_star_nm"]) if math.isfinite(eps) else None
    if valley.found and out.get("eps_sep_star_nm"):
        out["valley_ratio_to_sep_star"] = valley.eps_nm / out["eps_sep_star_nm"]
    if valley.found:
        out["valley_unclustered"] = kdm.unclustered_fraction(x, y, valley.eps_nm, MIN_SAMPLES)
    out["seconds"] = time.time() - t0
    return out


def cases() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in SIGMAS:
        for n in LOCS_PER_CLUSTER:
            for f in FRACTIONS:
                for seed in SEEDS:
                    out.append(dict(kind="ring", sigma=s, locs=n, f=f, seed=seed))
    for d in CONTROL_DENSITIES_PER_UM2:
        for seed in SEEDS:
            out.append(dict(kind="control", density=d, seed=seed))
    return out


def _summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def share(vals: List[bool]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    ring_rows = [r for r in rows if r["kind"] == "ring"]
    summ: Dict[str, Any] = {}
    by_f: Dict[str, Any] = {}
    for f in FRACTIONS:
        rr = [r for r in ring_rows if r["f"] == f]
        entry: Dict[str, Any] = {"n_runs": len(rr), "valley_found": share([r["valley_found"] for r in rr])}
        for m in ("valley", "kneedle", "p90_old", "gazal25"):
            rel = [r[f"{m}_rel_ari"] for r in rr if r.get(f"{m}_rel_ari") is not None]
            ratio = [r[f"{m}_ratio"] for r in rr if r.get(f"{m}_ratio") is not None]
            entry[m] = {"n": len(rel),
                        "median_rel_ari": float(np.median(rel)) if rel else None,
                        "share_rel_ari_ge_0.9": share([v >= 0.9 for v in rel]),
                        "median_ratio": float(np.median(ratio)) if ratio else None,
                        "ratio_q10_q90": ([float(q) for q in np.percentile(ratio, [10, 90])] if ratio else None)}
        by_f[f"{f:g}"] = entry
    summ["by_background_fraction"] = by_f
    by_locs: Dict[str, Any] = {}
    for n in LOCS_PER_CLUSTER:
        rr = [r for r in ring_rows if r["locs"] == n]
        rel = [r["valley_rel_ari"] for r in rr if r.get("valley_rel_ari") is not None]
        by_locs[str(n)] = {"valley_found": share([r["valley_found"] for r in rr]),
                           "valley_median_rel_ari": float(np.median(rel)) if rel else None,
                           "valley_median_ratio": float(np.median([r["valley_ratio"] for r in rr
                                                                   if r.get("valley_ratio") is not None]))
                           if rel else None}
    summ["by_locs_per_cluster"] = by_locs
    by_sigma: Dict[str, Any] = {}
    for s in SIGMAS:
        rr = [r for r in ring_rows if r["sigma"] == s]
        rel = [r["valley_rel_ari"] for r in rr if r.get("valley_rel_ari") is not None]
        by_sigma[f"{s:g}"] = {"valley_found": share([r["valley_found"] for r in rr]),
                              "valley_median_rel_ari": float(np.median(rel)) if rel else None}
    summ["by_sigma"] = by_sigma
    cells: Dict[str, Any] = {}
    for f in FRACTIONS:
        for n in LOCS_PER_CLUSTER:
            rr = [r for r in ring_rows if r["f"] == f and r["locs"] == n]
            ratio = [r["valley_ratio"] for r in rr if r.get("valley_ratio") is not None]
            rel = [r["valley_rel_ari"] for r in rr if r.get("valley_rel_ari") is not None]
            sep_ratio = [r["valley_ratio_to_sep_star"] for r in rr if r.get("valley_ratio_to_sep_star") is not None]
            cells[f"f={f:g} locs={n}"] = {
                "valley_found": f"{sum(r['valley_found'] for r in rr)}/{len(rr)}",
                "median_ratio_to_eps_star": float(np.median(ratio)) if ratio else None,
                "median_ratio_to_sep_star": float(np.median(sep_ratio)) if sep_ratio else None,
                "median_rel_ari": float(np.median(rel)) if rel else None,
                "median_eps_star_nm": float(np.median([r["eps_star_nm"] for r in rr])),
                "median_valley_nm": float(np.median([r["valley_nm"] for r in rr if r["valley_found"]]))
                if ratio else None}
    summ["by_cell"] = cells
    sep_all = [r["valley_ratio_to_sep_star"] for r in ring_rows if r.get("valley_ratio_to_sep_star") is not None]
    summ["valley_ratio_to_sep_star_median_q10_q90"] = (
        [float(np.median(sep_all))] + [float(q) for q in np.percentile(sep_all, [10, 90])]) if sep_all else None
    kn_ctrl = [r["kneedle_nm"] for r in rows if r["kind"] == "control"]
    summ["controls_kneedle_nm"] = kn_ctrl
    found = [r for r in ring_rows if r["valley_found"]]
    summ["valley_interval_covers_eps_star"] = share(
        [r["valley_low_nm"] <= r["eps_star_nm"] <= r["valley_high_nm"] for r in found])
    summ["valley_inside_good_eps_range"] = share(
        [r["eps_good_low_nm"] <= r["valley_nm"] <= r["eps_good_high_nm"] for r in found])
    ctrl = [r for r in rows if r["kind"] == "control"]
    summ["controls_background_only"] = {"n_runs": len(ctrl),
                                        "valley_found": share([r["valley_found"] for r in ctrl]),
                                        "reasons": sorted({r["valley_reason"] for r in ctrl})}
    return summ


def _clean(o: Any) -> Any:
    """NaN and infinities as null, numpy scalars as Python numbers: strict JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    todo = cases()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(4, int(args.workers))) as pool:
        rows = list(pool.map(run_case, todo, chunksize=4))
    result = {"script": "docs/sims/kdist/sim_kdist_valley.py", "min_samples": MIN_SAMPLES,
              "grid": {"sigma_nm": SIGMAS, "locs_per_cluster": LOCS_PER_CLUSTER, "background_fraction": FRACTIONS,
                       "seeds": SEEDS, "control_densities_per_um2": CONTROL_DENSITIES_PER_UM2},
              "eps_grid_nm": [float(e) for e in EPS_GRID], "seconds": time.time() - t0,
              "summary": _summary(rows), "runs": rows}
    path = os.path.join(HERE, "sim_kdist_valley_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean(result), fh, indent=1)
    print(json.dumps(result["summary"], indent=1, default=str))
    print(f"{len(rows)} runs in {result['seconds']:.0f} s -> docs/sims/kdist/{os.path.basename(path)}")


if __name__ == "__main__":
    main()
