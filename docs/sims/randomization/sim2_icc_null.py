# -*- coding: utf-8 -*-
"""
sim2: the chance level of the clamped one-way ICC estimator (true ICC 0)
by Monte Carlo (as tools/mps_batch.icc_null_for_sizes does, 400 draws)
against the closed form from the F distribution:

   ICC_hat = (F - 1) / (F + n0 - 1),  F = MS_between / MS_within,
   under H0 (balanced): F ~ F(k - 1, N - k)   =>
   q95(ICC_hat) = (F_0.95(k-1, N-k) - 1) / (F_0.95 + n0 - 1)

(ANOVA estimator of ICC(1) and its F-based inference: McGraw & Wong 1996,
Psychological Methods 1:30-46; Donner 1986, Int Stat Rev 54:67-82).
For unbalanced designs n0 = (N - sum n_i^2 / N) / (k - 1) (the same n0 the
program uses) and the F reference is approximate.
"""
import json
import os

import numpy as np
from scipy.stats import f as fdist

OUT = os.path.dirname(os.path.abspath(__file__))


def icc_core(groups):
    k = len(groups)
    sizes = np.array([g.size for g in groups], float)
    n = sizes.sum()
    allv = np.concatenate(groups)
    means = np.array([g.mean() for g in groups])
    ms_b = np.sum(sizes * (means - allv.mean()) ** 2) / (k - 1)
    ms_w = sum(((g - m) ** 2).sum() for g, m in zip(groups, means)) / (n - k)
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    var_b = (ms_b - ms_w) / n0
    return max(0.0, var_b / (var_b + ms_w))


def mc_q95(sizes, n_sim, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty(n_sim)
    for i in range(n_sim):
        out[i] = icc_core([rng.normal(size=s) for s in sizes])
    return float(np.percentile(out, 95))


def analytic_q95(sizes, alpha=0.05):
    sizes = np.asarray(sizes, float)
    k, n = sizes.size, sizes.sum()
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    fq = fdist.ppf(1 - alpha, k - 1, n - k)
    return float((fq - 1) / (fq + n0 - 1))


def main():
    designs = {
        "2x9": [9, 9], "6x9": [9] * 6, "6x50": [50] * 6, "3x6": [6] * 3,
        "4x15": [15] * 4, "ragged 2,3,12,20": [2, 3, 12, 20],
        "ragged 3,4,5": [3, 4, 5], "5x8": [8] * 5,
    }
    rows = {}
    for name, sizes in designs.items():
        a = analytic_q95(sizes)
        m400 = mc_q95(sizes, 400)
        m20k = mc_q95(sizes, 20000, seed=1)
        rows[name] = dict(sizes=sizes, analytic_q95=a, mc400_q95=m400, mc20000_q95=m20k)
        print(f"{name:>18}: analytic {a:.3f}   MC400 {m400:.3f}   MC20000 {m20k:.3f}")
    # Monte Carlo sd of a 95th percentile from 400 draws (bootstrap over seeds)
    sds = {}
    for name in ("2x9", "6x9"):
        vals = [mc_q95(designs[name], 400, seed=s) for s in range(30)]
        sds[name] = float(np.std(vals, ddof=1))
        print(f"sd of the 400-draw q95 over seeds, {name}: {sds[name]:.3f}")
    with open(os.path.join(OUT, "sim2_icc_null.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, mc400_sd_over_seeds=sds), f, indent=1)


if __name__ == "__main__":
    main()
