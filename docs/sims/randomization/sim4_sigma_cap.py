# -*- coding: utf-8 -*-
"""
sim4: when does the occupancy cap sigma_max <= d_max/3 (Gazal et al. 2026)
bind, for synthetic clusters of n localizations?

Isotropic 2D Gaussian cluster (sd sigma): r^2/sigma^2 ~ chi2_2 = 2 Exp(1),
so r_max ~ sigma sqrt(2 (ln n + gamma)) asymptotically (max of n Exp(1)
~ ln n + 0.5772). Cap binds when sigma_hat > r_max/3, i.e. when
sqrt(2 (ln n + 0.5772)) < 3  <=>  ln n < 3.92  <=>  n < ~50.
Uniform disc cluster (radius a): sigma = a/2 per axis, d_max -> a, so the
cap (a/3) ALWAYS binds for large n (a/2 > a/3).
Also records the occupied half-width the ellipse claims (3 sigma_eff).
"""
import json
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(4)


def sigma_major(pts):
    cov = np.cov(pts, rowvar=False, ddof=1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return float(np.sqrt(ev[-1]))


res = {}
for n in (5, 8, 10, 15, 20, 30, 50, 80, 100, 200, 500):
    capped_g, capped_u, hw_g, hw_u = [], [], [], []
    for _ in range(2000):
        g = rng.normal(0, 20.0, (n, 2))
        dmax = np.max(np.linalg.norm(g - g.mean(axis=0), axis=1))
        s = sigma_major(g)
        capped_g.append(s > dmax / 3)
        hw_g.append(3 * min(s, dmax / 3))
        th = rng.uniform(0, 2 * np.pi, n)
        r = 40.0 * np.sqrt(rng.uniform(size=n))
        u = np.column_stack([r * np.cos(th), r * np.sin(th)])
        dmax_u = np.max(np.linalg.norm(u - u.mean(axis=0), axis=1))
        su = sigma_major(u)
        capped_u.append(su > dmax_u / 3)
        hw_u.append(3 * min(su, dmax_u / 3))
    res[n] = dict(gauss_sd20_fraction_capped=float(np.mean(capped_g)),
                  gauss_claimed_halfwidth_nm=float(np.mean(hw_g)),
                  uniform_disc_r40_fraction_capped=float(np.mean(capped_u)),
                  uniform_claimed_halfwidth_nm=float(np.mean(hw_u)))
    print(f"n={n:>4}: Gaussian capped {np.mean(capped_g):.2f} (half-width {np.mean(hw_g):.0f} nm; "
          f"3 sigma = 60), uniform disc capped {np.mean(capped_u):.2f} (half-width {np.mean(hw_u):.0f} nm; radius 40)")
with open(os.path.join(OUT, "sim4_sigma_cap.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
