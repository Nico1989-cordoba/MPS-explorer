# -*- coding: utf-8 -*-
"""Support simulation of docs/viability_and_reference_period.md (synthetic only): detectability of the SiZer valley and peaks of a two-ring axial profile.

Two equal rings at z = 0 and z = P, each N(centre, sigma), sigma/P on a grid; n points (iid) or clustered points
(cluster centres jittered inside the ring, cluster-robust variance as in tools.mps_sizer); SiZer at bandwidth h = c P
with the simultaneous quantile of Chaudhuri & Marron (1999) and the independent-blocks device (ESS >= 5).
Outputs: P[valley between the centres], P[peak within P/3 of each centre], sd of the detected peak position, and the
analytic peak-location sd  sqrt( (sqrt2/4) (s^2+h^2)^(5/2) / (n h^3) ).  Units: P = 1.
Writes sim_sizer_valley_results.json next to this file.  Single process, well under 5 min.
"""
import json
import math
import os
import time

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
SQ2PI = math.sqrt(2.0 * math.pi)
MIN_ESS = 5.0


def sizer(z, h, groups=None, alpha=0.05):
    step = max(h / 4.0, 0.002)
    lo, hi = np.percentile(z, [0.2, 99.8])
    grid = np.arange(lo - 2 * h, hi + 2 * h + step, step)
    n = z.size
    u = (grid[:, None] - z[None, :]) / h
    ph = np.exp(-0.5 * u * u)
    D = -u * ph / (SQ2PI * h * h)
    fd = D.mean(axis=1)
    fh = ph.sum(axis=1) / (SQ2PI * h * n)
    ess = ph.sum(axis=1)
    if groups is None:
        var = ((D * D).mean(axis=1) - fd ** 2) / n
    else:
        g = np.unique(groups, return_inverse=True)[1]
        G = g.max() + 1
        S = np.zeros((G, grid.size))
        np.add.at(S, g, D.T)
        m_g = np.bincount(g, minlength=G).astype(float)
        r = S - m_g[:, None] * fd[None, :]
        var = (r * r).sum(axis=0) / (n * n)
    ok = ess >= MIN_ESS
    l_blocks = max(n / max(float(np.mean(ess[ok])), 1e-9), 1.0) if ok.any() else 1.0
    q = norm.ppf((1.0 + (1.0 - alpha) ** (1.0 / l_blocks)) / 2.0)
    t = fd / np.sqrt(np.maximum(var, 1e-300))
    sign = np.where(t > q, 1, np.where(t < -q, -1, 0))
    sign[~ok] = 0
    return grid, sign, fh


def peaks_valleys(sign, fhat, grid):
    idx = np.flatnonzero(sign != 0)
    peaks, valleys = [], []
    if idx.size == 0:
        return peaks, valleys
    s = sign[idx]
    brk = np.flatnonzero(np.diff(s) != 0)
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, s.size - 1]
    for a in range(starts.size - 1):
        s1, s2 = s[starts[a]], s[starts[a + 1]]
        i0, i1 = idx[ends[a]], idx[starts[a + 1]]
        seg = slice(i0, i1 + 1)
        if s1 > 0 and s2 < 0:
            peaks.append(float(grid[i0 + int(np.argmax(fhat[seg]))]))
        elif s1 < 0 and s2 > 0:
            valleys.append(float(grid[i0 + int(np.argmin(fhat[seg]))]))
    return peaks, valleys


def valley_between(grid, sign, a, b):
    m = (grid > a) & (grid < b)
    s = sign[m]
    neg = np.flatnonzero(s < 0)
    pos = np.flatnonzero(s > 0)
    return bool(neg.size and pos.size and neg.min() < pos.max())


def analytic_peak_sd(s, h, n):
    return math.sqrt((math.sqrt(2.0) / 4.0) * (s * s + h * h) ** 2.5 / (n * h ** 3))


def run_cell(rng, sig, n, c, reps, clustered, win=1.0 / 3.0, mean_cluster=50, jitter_frac=0.6):
    h = c
    hits_v = hits_p = 0
    pk_err = []
    for _ in range(reps):
        if not clustered:
            z = np.concatenate([rng.normal(0.0, sig, n // 2), rng.normal(1.0, sig, n - n // 2)])
            groups = None
        else:
            # cluster centres carry jitter_frac of the ring variance, points the rest (between + within = sigma^2)
            s_c = sig * math.sqrt(jitter_frac)
            s_w = sig * math.sqrt(1.0 - jitter_frac)
            parts, gl, gid = [], [], 0
            for centre in (0.0, 1.0):
                n_half = n // 2
                sizes = rng.poisson(mean_cluster, size=max(1, n_half // mean_cluster))
                sizes = np.maximum(sizes, 1)
                for sz in sizes:
                    cz = rng.normal(centre, s_c)
                    parts.append(rng.normal(cz, s_w, sz))
                    gl.append(np.full(sz, gid))
                    gid += 1
            z = np.concatenate(parts)
            groups = np.concatenate(gl)
        grid, sign, fh = sizer(z, h, groups)
        pk, vl = peaks_valleys(sign, fh, grid)
        hits_v += valley_between(grid, sign, 0.0, 1.0)
        near0 = [p for p in pk if abs(p - 0.0) <= win]
        near1 = [p for p in pk if abs(p - 1.0) <= win]
        hits_p += bool(near0 and near1)
        for cen, near in ((0.0, near0), (1.0, near1)):
            if near:
                pk_err.append(min(near, key=lambda p: abs(p - cen)) - cen)
    sd_pk = float(np.std(pk_err)) if len(pk_err) > 5 else float("nan")
    return hits_v / reps, hits_p / reps, sd_pk


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(20261007)
    sigmas = [0.25, 0.30, 0.35, 0.40, 0.45]
    ns = [400, 1000, 4000, 16000]
    cs = [0.05, 0.10, 0.15, 0.20]
    rows = []
    for sig in sigmas:
        for n in ns:
            reps = 60 if n >= 16000 else 100
            for c in cs:
                pv, pp, sdp = run_cell(rng, sig, n, c, reps, clustered=False)
                rows.append(dict(sigma_over_p=sig, n=n, h_over_p=c, clustered=False, p_valley=pv, p_both_peaks=pp,
                                 peak_sd_meas=sdp, peak_sd_analytic=analytic_peak_sd(sig, c, n // 2), reps=reps))
    # clustered variant at n = 4000 (mean 50 localizations per cluster -> 80 clusters), h = 0.10 and 0.20
    for sig in sigmas:
        for c in (0.10, 0.20):
            pv, pp, sdp = run_cell(rng, sig, 4000, c, 100, clustered=True)
            rows.append(dict(sigma_over_p=sig, n=4000, h_over_p=c, clustered=True, p_valley=pv, p_both_peaks=pp,
                             peak_sd_meas=sdp, peak_sd_analytic=analytic_peak_sd(sig, c, 40), reps=100))
    # analytic valley depth of the equal-weight mixture at the midpoint, relative to the density at a ring centre
    depth = {}
    for sig in [0.20, 0.25, 0.30, 0.32, 0.35, 0.36, 0.40, 0.45, 0.50]:
        f_mid = norm.pdf(0.5 / sig)
        f_cen = 0.5 * (norm.pdf(0.0) + norm.pdf(1.0 / sig))
        depth[str(sig)] = 1.0 - f_mid / f_cen
    out = dict(rows=rows, valley_depth_equal_rings=depth, seconds=time.perf_counter() - t0,
               note="P = 1 units; two equal rings; iid or clustered (cluster-robust variance); alpha 0.05 simultaneous")
    with open(os.path.join(HERE, "sim_sizer_valley_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{len(rows)} cells in {out['seconds']:.1f} s")
    print("sigma/P  n      h/P   clustered  P(valley)  P(both peaks)  peak sd meas / analytic")
    for r in rows:
        print(f"{r['sigma_over_p']:.2f}  {r['n']:6d}  {r['h_over_p']:.2f}  {str(r['clustered']):5s}  "
              f"{r['p_valley']:.2f}       {r['p_both_peaks']:.2f}        {r['peak_sd_meas']:.4f} / {r['peak_sd_analytic']:.4f}")
    print("valley depth (equal rings, gap P):", {k: round(v, 3) for k, v in depth.items()})


if __name__ == "__main__":
    main()
