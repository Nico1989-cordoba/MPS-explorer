# -*- coding: utf-8 -*-
"""
sim1: behaviour of the KS statistic D and of candidate p-values under a TRUE
null, for the randomization control of MPS Explorer (parameter 8).

Synthetic only. K cluster centres on a ring (radius R, band +/-50 nm) placed
by simple sequential inhibition (hard core d_min, as the program does). The
"observed" pattern is one more draw of the same null, so every valid test
must reject at the nominal 5 %.

Measured per replicate:
  * D_obs = sup|F_obs - F_pool| (scipy.stats.ks_2samp, as the program)
  * p_scipy (asymptotic two-sample p with m = 1000 K)
  * p_MC(median): rank of the observed median among per-iteration medians
  * p_MC(D): rank of D_obs among D_i = sup|F_i - F_pool| of each iteration
  * z_D = (D_obs - mean D_i) / sd D_i (calibrated effect size)
Also: E[sqrt(n) D_n] for one-sample uniform data (Kolmogorov mean, 0.8687).
"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1)


def ssi_ring(K, R, dmin, band, rng, max_attempts_factor=20):
    """K points on a ring: arc angles by sequential inhibition (hard core
    dmin measured as Euclidean distance), radial offset uniform in +/-band."""
    pts = []
    attempts = 0
    while len(pts) < K and attempts < max_attempts_factor * K:
        attempts += 1
        th = rng.uniform(0, 2 * np.pi)
        r = R + rng.uniform(-band, band)
        p = np.array([r * np.cos(th), r * np.sin(th)])
        if pts:
            d2 = np.sum((np.array(pts) - p) ** 2, axis=1)
            if np.min(d2) < dmin * dmin:
                continue
        pts.append(p)
    return np.array(pts)


def nn1(pts):
    t = cKDTree(pts)
    d, _ = t.query(pts, k=2)
    return d[:, 1]


def sup_diff(a, b_sorted, nb):
    """sup |F_a - F_b| with b given sorted (b is the pooled null)."""
    a = np.sort(a)
    grid = np.concatenate([a, b_sorted])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b_sorted, grid, side="right") / nb
    return float(np.max(np.abs(fa - fb)))


def one_replicate(K, R, band, n_iter, rng):
    # "observed" = a null draw with its own hard core d0; the program then
    # sets the null's d_min = min observed 1NN (data-conditioned null).
    d0 = 60.0
    obs = ssi_ring(K, R, d0, band, rng)
    nn_obs = nn1(obs)
    dmin = float(np.min(nn_obs))
    pooled = []
    medians = np.empty(n_iter)
    per_iter = []
    for it in range(n_iter):
        c = ssi_ring(K, R, dmin, band, rng)
        nn = nn1(c)
        pooled.append(nn)
        medians[it] = np.median(nn)
        per_iter.append(nn)
    pool = np.sort(np.concatenate(pooled))
    nb = pool.size
    ks = ks_2samp(nn_obs, pool)
    D_obs = float(ks.statistic)
    D_i = np.array([sup_diff(x, pool, nb) for x in per_iter])
    med_obs = float(np.median(nn_obs))
    # Monte Carlo rank p-values (Besag & Diggle 1977; Phipson & Smyth 2010)
    p_med_lower = (1 + np.sum(medians <= med_obs)) / (n_iter + 1)
    p_med_upper = (1 + np.sum(medians >= med_obs)) / (n_iter + 1)
    p_med_two = min(1.0, 2 * min(p_med_lower, p_med_upper))
    p_D = (1 + np.sum(D_i >= D_obs)) / (n_iter + 1)
    z_D = (D_obs - D_i.mean()) / D_i.std(ddof=1)
    return dict(K=K, D_obs=D_obs, p_scipy=float(ks.pvalue), p_med_two=p_med_two,
                p_D=p_D, z_D=float(z_D), mean_D_i=float(D_i.mean()),
                q95_D_i=float(np.quantile(D_i, 0.95)))


def main():
    t0 = time.perf_counter()
    R, band = 2000.0, 50.0
    n_iter = 1000
    results = {}
    for K, n_rep in ((20, 120), (50, 120), (100, 80)):
        rows = [one_replicate(K, R, band, n_iter, rng) for _ in range(n_rep)]
        D = np.array([r["D_obs"] for r in rows])
        ps = np.array([r["p_scipy"] for r in rows])
        pm = np.array([r["p_med_two"] for r in rows])
        pD = np.array([r["p_D"] for r in rows])
        results[K] = dict(
            n_rep=n_rep,
            mean_D_obs=float(D.mean()), q95_D_obs=float(np.quantile(D, 0.95)),
            dkw_expect_0p87_over_sqrtK=0.8687 / np.sqrt(K),
            dkw_95_band=float(np.sqrt(np.log(2 / 0.05) / (2 * K))),
            mean_D_i=float(np.mean([r["mean_D_i"] for r in rows])),
            reject_scipy_005=float(np.mean(ps < 0.05)),
            reject_MC_median_005=float(np.mean(pm < 0.05)),
            reject_MC_D_005=float(np.mean(pD < 0.05)),
            mean_zD=float(np.mean([r["z_D"] for r in rows])),
            sd_zD=float(np.std([r["z_D"] for r in rows], ddof=1)),
        )
        print(f"K={K}: E[D_obs]={D.mean():.3f} (0.87/sqrt K={0.8687/np.sqrt(K):.3f}), "
              f"q95 D={np.quantile(D,0.95):.3f}, DKW95={np.sqrt(np.log(40)/(2*K)):.3f}; "
              f"reject@0.05: scipy={np.mean(ps<0.05):.3f} MC-median={np.mean(pm<0.05):.3f} "
              f"MC-D={np.mean(pD<0.05):.3f}; zD mean {np.mean([r['z_D'] for r in rows]):+.2f}",
              flush=True)

    # Kolmogorov mean: one-sample D_n for uniform data, n = 50 and 200
    kol = {}
    for n in (50, 200):
        vals = []
        for _ in range(4000):
            x = np.sort(rng.uniform(size=n))
            i = np.arange(1, n + 1)
            d = max(np.max(i / n - x), np.max(x - (i - 1) / n))
            vals.append(np.sqrt(n) * d)
        kol[n] = float(np.mean(vals))
    print("E[sqrt(n) D_n] one-sample uniform:", kol, "(theory 0.8687)")
    results["kolmogorov_mean_sqrt_n_D"] = kol
    results["elapsed_s"] = time.perf_counter() - t0
    with open(os.path.join(OUT, "sim1_ks_mc.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"done in {results['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()
