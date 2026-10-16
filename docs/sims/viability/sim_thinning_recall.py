# -*- coding: utf-8 -*-
"""Support simulation of docs/viability_and_reference_period.md (synthetic only): DBSCAN cluster recall under Bernoulli thinning, the quantity behind the
count criterion f_min(x) and the cluster-count criterion k_min.

A "ring" is a strip of K clusters along a line (mean spacing 200 nm); cluster i has n_i ~ lognormal(mean m, cv 0.7)
localizations, 2D Gaussian spread sd 20 nm around its centre, plus uniform noise.  DBSCAN (eps 25 nm, min_samples 10,
Ester et al. 1996) at thinning fraction f (each localization kept with probability f) gives K(f) detected clusters;
recall(f) = K(f) / K(1).  The f at which recall = 0.90 is f_90.  Three things are checked:
 (1) f_90 depends on the cluster-size distribution (x = m / min_samples), so no single number can be first principles;
 (2) the K ratio IS the recall (so a cluster-count threshold k_min equals the recall tolerance by definition);
 (3) a per-axon estimate -- thin the DETECTED clusters' localizations in silico and re-run DBSCAN -- recovers the
     generator's f_90 (the calibration the program can run on any axon without pilot data).
Writes sim_thinning_recall_results.json.  Single process, < 3 min.
"""
import json
import math
import os
import time

import numpy as np
from sklearn.cluster import DBSCAN

HERE = os.path.dirname(os.path.abspath(__file__))
EPS, MS = 25.0, 10
FS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def make_ring(rng, K, mean_n, cv=0.7, spacing=200.0, sd_xy=20.0, noise_density=2e-4):
    sig = math.sqrt(math.log(1 + cv * cv))
    mu = math.log(mean_n) - 0.5 * sig * sig
    sizes = np.maximum(1, rng.lognormal(mu, sig, K).astype(int))
    centres = np.arange(K) * spacing + rng.normal(0, 30.0, K)
    pts, lab = [], []
    for i, (c, n) in enumerate(zip(centres, sizes)):
        pts.append(np.column_stack([rng.normal(c, sd_xy, n), rng.normal(0.0, sd_xy, n)]))
        lab.append(np.full(n, i))
    L = K * spacing
    n_noise = rng.poisson(noise_density * L * 400.0)
    pts.append(np.column_stack([rng.uniform(-100, L + 100, n_noise), rng.uniform(-200, 200, n_noise)]))
    lab.append(np.full(n_noise, -1))
    return np.vstack(pts), np.concatenate(lab), sizes


def n_detected(xy):
    if xy.shape[0] < MS:
        return 0
    lab = DBSCAN(eps=EPS, min_samples=MS).fit_predict(xy)
    return int(lab.max() + 1) if lab.size else 0


def recall_curve(rng, xy):
    k1 = n_detected(xy)
    out = {}
    for f in FS:
        keep = rng.random(xy.shape[0]) < f
        out[f] = n_detected(xy[keep]) / k1 if k1 else float("nan")
    return out, k1


def f_at(curve, level=0.9):
    fs = sorted(curve)
    for lo, hi in zip(fs[:-1], fs[1:]):
        a, b = curve[lo], curve[hi]
        if a <= level <= b:
            return lo + (level - a) / (b - a) * (hi - lo) if b > a else hi
    return float("nan")


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(11)
    K, reps = 60, 25
    results = []
    for mean_n in (40, 80, 160):
        x = mean_n / MS
        f90_true, f90_insilico, kratio_vs_recall = [], [], []
        for _ in range(reps):
            xy, lab, sizes = make_ring(rng, K, mean_n)
            curve, k1 = recall_curve(rng, xy)
            f90_true.append(f_at(curve))
            # per-axon in-silico estimate: keep only the localizations DBSCAN assigned to clusters at f = 1 (what the
            # program has), thin them and re-run DBSCAN
            lab1 = DBSCAN(eps=EPS, min_samples=MS).fit_predict(xy)
            det = xy[lab1 >= 0]
            curve2, _ = recall_curve(rng, det)
            f90_insilico.append(f_at(curve2))
            kratio_vs_recall.append(curve[0.7])
        fl = 1.46 - 0.336 * math.log(x)
        results.append(dict(mean_locs_per_cluster=mean_n, x=x, f90_true_mean=float(np.nanmean(f90_true)),
                            f90_true_sd=float(np.nanstd(f90_true)), f90_insilico_mean=float(np.nanmean(f90_insilico)),
                            f90_insilico_sd=float(np.nanstd(f90_insilico)), reps=reps,
                            f_min_formula_of_code=fl, recall_at_f_0p7_mean=float(np.nanmean(kratio_vs_recall))))
    out = dict(eps=EPS, min_samples=MS, K=K, results=results, seconds=time.perf_counter() - t0,
               note="generic generator (lognormal sizes cv 0.7, sd_xy 20 nm, spacing 200 nm); not the project's simulator")
    with open(os.path.join(HERE, "sim_thinning_recall_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{out['seconds']:.1f} s")
    print("mean n   x     f90 true (sd)      f90 in-silico (sd)   code formula   recall at f=0.7")
    for r in results:
        print(f"{r['mean_locs_per_cluster']:5d}  {r['x']:5.1f}  {r['f90_true_mean']:.3f} ({r['f90_true_sd']:.3f})   "
              f"{r['f90_insilico_mean']:.3f} ({r['f90_insilico_sd']:.3f})      {r['f_min_formula_of_code']:.3f}        "
              f"{r['recall_at_f_0p7_mean']:.3f}")


if __name__ == "__main__":
    main()
