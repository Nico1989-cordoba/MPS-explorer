# -*- coding: utf-8 -*-
"""
sim1b: isolates the effect of the DATA-CONDITIONED hard core.

Variant X (exchangeable): the null is simulated with the TRUE hard core d0
of the generating process -> the observed pattern is one more draw of the
null and the Monte Carlo rank p must be uniform (5 % at alpha = 0.05).
Variant P (paper's rule): the null uses d_min = min observed 1NN, which is
always >= d0 (the minimum of K distances that are all >= d0), so the null
is MORE regular than the process that produced the data. Also records the
fraction of observed 1NN distances that fall below the null's smallest
distance (a built-in excess of short distances, F_obs > F_null).
Same generator as sim1; fewer iterations (B = 199) so it runs fast.
"""
import json
import os
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11)


def ssi_ring(K, R, dmin, band, rng, max_attempts_factor=20):
    pts = []
    attempts = 0
    while len(pts) < K and attempts < max_attempts_factor * K:
        attempts += 1
        th = rng.uniform(0, 2 * np.pi)
        r = R + rng.uniform(-band, band)
        p = np.array([r * np.cos(th), r * np.sin(th)])
        if pts:
            if np.min(np.sum((np.array(pts) - p) ** 2, axis=1)) < dmin * dmin:
                continue
        pts.append(p)
    return np.array(pts)


def nn1(pts):
    d, _ = cKDTree(pts).query(pts, k=2)
    return d[:, 1]


def run(K, variant, n_rep, B=199, R=2000.0, band=50.0, d0=60.0):
    rej_med = rej_scipy = 0
    below = []
    for _ in range(n_rep):
        obs = ssi_ring(K, R, d0, band, rng)
        nn_obs = nn1(obs)
        dmin = d0 if variant == "X" else float(np.min(nn_obs))
        meds = np.empty(B)
        pool = []
        for b in range(B):
            nn = nn1(ssi_ring(K, R, dmin, band, rng))
            meds[b] = np.median(nn)
            pool.append(nn)
        pool = np.concatenate(pool)
        m = np.median(nn_obs)
        p_lo = (1 + np.sum(meds <= m)) / (B + 1)
        p_hi = (1 + np.sum(meds >= m)) / (B + 1)
        p_two = min(1.0, 2 * min(p_lo, p_hi))
        rej_med += p_two < 0.05
        rej_scipy += ks_2samp(nn_obs, pool).pvalue < 0.05
        below.append(np.mean(nn_obs < pool.min()))
    return dict(K=K, variant=variant, n_rep=n_rep, B=B,
                reject_MC_median=rej_med / n_rep, reject_scipy=rej_scipy / n_rep,
                mean_frac_obs_below_null_min=float(np.mean(below)))


def main():
    t0 = time.perf_counter()
    out = []
    for K in (20, 50):
        for v in ("X", "P"):
            r = run(K, v, n_rep=150)
            out.append(r)
            print(r, flush=True)
    with open(os.path.join(OUT, "sim1b_exchangeable.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=out, elapsed_s=time.perf_counter() - t0), f, indent=1)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
