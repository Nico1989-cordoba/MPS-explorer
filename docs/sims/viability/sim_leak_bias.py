# -*- coding: utf-8 -*-
"""Support simulation of docs/viability_and_reference_period.md (synthetic only): how much a fraction c of axial-leak COPIES biases the column excess zeta.

Model: two rings, K clusters each, centres uniform on a closed perimeter of length L (1D circle); a cluster of ring a
is "matched" when some cluster of ring b lies within tau (arc distance); E_dir = n_matched / min(K_a, K_b); the null is
the toroidal (arc) shift of ring b (Lotwick & Silverman 1982), B shifts, E* and sd*; zeta = (E_dir - E*) / sd*.
Copies: a fraction c of all clusters (copies / (K_a + K_b), the program's exp_spur_frac) are leak copies of ring-a
clusters placed in ring b at the SAME arc position (in register), as the leak mechanism makes them.
Prediction (first principles): each copy adds one match unless its parent was already matched (probability p0 = E*),
so  E[delta n_matched] = (K_a + K_b) c (1 - p0)  and  E[delta zeta] = (K_a + K_b) c (1 - p0) / (sd* min K)
= 2 c (1 - p0) / sd*  for K_a = K_b.  The copy fraction tolerated for a bias of delta sd is
c_max = delta sd* / (2 (1 - p0)).  With the binomial approximation sd* ~ sqrt(p0 (1 - p0) / K) this is
c_max ~ delta sqrt(p0 / ((1 - p0) K)) / 2.
Writes sim_leak_bias_results.json.  Single process, < 2 min.
"""
import json
import math
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def n_matched(a, b, L, tau):
    """Clusters of a with a cluster of b within arc distance tau (closed perimeter L)."""
    d = np.abs(a[:, None] - b[None, :])
    d = np.minimum(d, L - d)
    return int(np.count_nonzero(d.min(axis=1) <= tau))


def zeta_of(a, b, L, tau, rng, B=199):
    k = min(a.size, b.size)
    obs = n_matched(a, b, L, tau) / k
    shifts = rng.uniform(0.0, L, size=B)
    null = np.array([n_matched(a, (b + s) % L, L, tau) / k for s in shifts])
    e_star, sd_star = null.mean(), null.std(ddof=1)
    return obs, e_star, sd_star, (obs - e_star) / sd_star if sd_star > 0 else float("nan")


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(7)
    tau = 80.0                              # tolerance 80 nm (generic, not a pilot value)
    reps = 300
    all_out = {}
    for K in (20, 40, 80):
        L = K * 200.0                       # mean spacing 200 nm
        # copy fractions that give whole copies for 2K clusters
        cs = [k / (2.0 * K) for k in (0, 1, 2, 3, 4, 6, 8)]
        rows = []
        for c in cs:
            zs, e_stars, sd_stars = [], [], []
            for _ in range(reps):
                a = rng.uniform(0.0, L, K)
                b = rng.uniform(0.0, L, K)
                n_copies = int(round(c * (2 * K)))
                if n_copies:
                    parents = rng.choice(K, n_copies, replace=False)
                    b = np.concatenate([b, a[parents]])      # copies in register with their parents
                obs, e_star, sd_star, z = zeta_of(a, b, L, tau, rng)
                zs.append(z)
                e_stars.append(e_star)
                sd_stars.append(sd_star)
            rows.append(dict(c=c, n_copies=int(round(c * 2 * K)), zeta_mean=float(np.mean(zs)),
                             zeta_se=float(np.std(zs) / math.sqrt(reps)), e_star_mean=float(np.mean(e_stars)),
                             sd_star_mean=float(np.mean(sd_stars)), reps=reps))
        base = rows[0]
        p0, sd0 = base["e_star_mean"], base["sd_star_mean"]
        pred = {str(round(r["c"], 4)): 2.0 * r["c"] * (1.0 - p0) / sd0 for r in rows}
        slope_meas = np.polyfit([r["c"] for r in rows], [r["zeta_mean"] - base["zeta_mean"] for r in rows], 1)[0]
        slope_pred = 2.0 * (1.0 - p0) / sd0
        c_max = {str(d): d * sd0 / (2.0 * (1.0 - p0)) for d in (0.25, 0.5)}
        c_max_meas = {str(d): d / slope_meas for d in (0.25, 0.5)}
        binom_sd = math.sqrt(p0 * (1.0 - p0) / K)
        all_out[str(K)] = dict(L=L, K=K, tau=tau, rows=rows, p0=p0, sd_star=sd0, sd_star_binomial_approx=binom_sd,
                               zeta_slope_per_unit_c_measured=float(slope_meas),
                               zeta_slope_per_unit_c_predicted=float(slope_pred), delta_zeta_predicted=pred,
                               c_max_for_delta_sd_predicted=c_max, c_max_for_delta_sd_measured_slope=c_max_meas)
        print(f"K = {K}: p0 = {p0:.3f}, sd* = {sd0:.4f} (binomial approx {binom_sd:.4f})")
        for r in rows:
            print(f"  c = {100 * r['c']:5.2f} % ({r['n_copies']:2d} copies): zeta {r['zeta_mean']:+.3f} +/- {r['zeta_se']:.3f}"
                  f"   predicted shift {pred[str(round(r['c'], 4))]:+.3f}")
        print(f"  slope dzeta/dc: measured {slope_meas:.2f}, predicted {slope_pred:.2f}  (per 1 % copies: "
              f"{slope_meas / 100:.3f} / {slope_pred / 100:.3f} sd)")
        print("  c_max predicted (0.25 / 0.5 sd):", {k: f"{100 * v:.2f} %" for k, v in c_max.items()},
              " measured-slope:", {k: f"{100 * v:.2f} %" for k, v in c_max_meas.items()})
    all_out["seconds"] = time.perf_counter() - t0
    with open(os.path.join(HERE, "sim_leak_bias_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=1)
    print(f"{all_out['seconds']:.1f} s")
    return
    for c in []:
        zs, e_stars, sd_stars, dn = [], [], [], []
        for _ in range(reps):
            a = rng.uniform(0.0, L, K)
            b = rng.uniform(0.0, L, K)
            n_copies = int(round(c * (2 * K)))
            if n_copies:
                parents = rng.choice(K, n_copies, replace=False)
                b = np.concatenate([b, a[parents]])      # copies in register with their parents
            obs, e_star, sd_star, z = zeta_of(a, b, L, tau, rng)
            zs.append(z)
            e_stars.append(e_star)
            sd_stars.append(sd_star)
        rows.append(dict(c=c, n_copies=int(round(c * 2 * K)), zeta_mean=float(np.mean(zs)), zeta_se=float(np.std(zs) / math.sqrt(reps)),
                         e_star_mean=float(np.mean(e_stars)), sd_star_mean=float(np.mean(sd_stars)), reps=reps))
    base = rows[0]
    p0, sd0 = base["e_star_mean"], base["sd_star_mean"]
    pred = {str(r["c"]): 2.0 * r["c"] * (1.0 - p0) / sd0 for r in rows}
    slope_meas = np.polyfit([r["c"] for r in rows], [r["zeta_mean"] - base["zeta_mean"] for r in rows], 1)[0]
    slope_pred = 2.0 * (1.0 - p0) / sd0
    c_max = {str(d): d * sd0 / (2.0 * (1.0 - p0)) for d in (0.25, 0.5)}
    binom_sd = math.sqrt(p0 * (1.0 - p0) / K)
    out = dict(L=L, K=K, tau=tau, rows=rows, p0=p0, sd_star=sd0, sd_star_binomial_approx=binom_sd,
               zeta_slope_per_unit_c_measured=float(slope_meas), zeta_slope_per_unit_c_predicted=float(slope_pred),
               delta_zeta_predicted=pred, c_max_for_delta_sd=c_max, seconds=time.perf_counter() - t0)
    with open(os.path.join(HERE, "sim_leak_bias_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"p0 = {p0:.3f}, sd* = {sd0:.4f} (binomial approx {binom_sd:.4f}); {out['seconds']:.1f} s")
    for r in rows:
        print(f"c = {r['c']:.2f} ({r['n_copies']:2d} copies): zeta {r['zeta_mean']:+.3f} +/- {r['zeta_se']:.3f}   "
              f"predicted shift {pred[str(r['c'])]:+.3f}")
    print(f"slope dzeta/dc: measured {slope_meas:.2f}, predicted {slope_pred:.2f}  (per 1 % copies: "
          f"{slope_meas / 100:.3f} / {slope_pred / 100:.3f} sd)")
    print("c_max for 0.25 sd / 0.5 sd:", {k: f"{100 * v:.2f} %" for k, v in c_max.items()})


if __name__ == "__main__":
    main()
