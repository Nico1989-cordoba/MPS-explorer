# -*- coding: utf-8 -*-
"""
sim3: how the median Euclidean 1NN between cluster centres on a ring
responds to the factors that differ between two measurements of the same
structure (U3). Synthetic rings only; nothing from real data.

Reference ring: K_true = 70 centres on a circle of perimeter P = 25 um
(mean spacing s = P/K = 357 nm), near-regular (arc jitter sd = j).
Factors:
  A. regularity: arc jitter j in {0, 30, 60, 100} nm and a fully random
     hard-core arrangement; metric r = median 1NN / (P/K).
  B. detection efficiency q (Bernoulli thinning of true clusters).
  C. over-splitting: a fraction f of clusters split into two centres 40 nm
     apart (DBSCAN fragmenting one cluster).
  D. foreign clusters: a fraction f of K extra centres from a neighbouring
     ring at independent arc positions (axial slab spanning two rings).
  E. non-anchored clusters: a fraction f of K centres inside the ring
     (radius 0.5-0.9 R).
  F. chord vs arc: a contour with a deep fold brings two membrane stretches
     within 100 nm; Euclidean 1NN vs arc-length 1NN.
Also: median angular NN offset for N uniform points on a circle vs the
closed forms 180(1 - 2^(-1/N)) (median) and 180/(N+1) (mean).
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)
P = 25000.0
K = 70
R = P / (2 * np.pi)


def ring_points(arcs, radius=R):
    th = 2 * np.pi * arcs / P
    return np.column_stack([radius * np.cos(th), radius * np.sin(th)])


def nn1(pts):
    d, _ = cKDTree(pts).query(pts, k=2)
    return d[:, 1]


def regular_arcs(jitter, k=K):
    base = (np.arange(k) + 0.5) * P / k
    return np.mod(base + rng.normal(0, jitter, k), P)


def random_hardcore_arcs(k=K, dmin=60.0):
    pts = []
    tries = 0
    while len(pts) < k and tries < 50 * k:
        tries += 1
        a = rng.uniform(0, P)
        if pts:
            d = np.abs(np.array(pts) - a)
            d = np.minimum(d, P - d)
            if d.min() < dmin:
                continue
        pts.append(a)
    return np.array(pts)


def med_ratio(arcs, extra=None):
    pts = ring_points(arcs)
    if extra is not None:
        pts = np.vstack([pts, extra])
    return float(np.median(nn1(pts))) / (P / K)


def rep(fn, n=200):
    v = np.array([fn() for _ in range(n)])
    return float(v.mean()), float(v.std(ddof=1))


res = {}
# A regularity
A = {}
for j in (0.0, 30.0, 60.0, 100.0):
    A[f"jitter_{j:.0f}"] = rep(lambda: med_ratio(regular_arcs(j)))
A["random_hardcore_60nm"] = rep(lambda: med_ratio(random_hardcore_arcs()))
res["A_regularity_median1NN_over_P_by_K"] = A
print("A regularity r = median1NN/(P/K):", {k: round(v[0], 3) for k, v in A.items()})

# B thinning (detection efficiency), near-regular ring with jitter 30
B = {}
for q in (1.0, 0.8, 0.6, 0.5, 0.4):
    def f(q=q):
        a = regular_arcs(30.0)
        keep = rng.uniform(size=a.size) < q
        if keep.sum() < 3:
            return np.nan
        pts = ring_points(a[keep])
        return float(np.median(nn1(pts)))
    m, s = rep(f)
    B[f"q_{q}"] = dict(median_1nn_nm=m, sd=s, ratio_to_q1=m / (P / K))
print("B thinning median 1NN (nm):", {k: round(v["median_1nn_nm"]) for k, v in B.items()})
res["B_detection_efficiency"] = B

# C over-splitting
C = {}
for fs in (0.0, 0.1, 0.2, 0.3, 0.5):
    def f(fs=fs):
        a = regular_arcs(30.0)
        pts = ring_points(a)
        n_split = int(round(fs * K))
        idx = rng.choice(K, n_split, replace=False)
        extra = pts[idx] + rng.normal(0, 1, (n_split, 2)) * 0  # placeholder
        ang = rng.uniform(0, 2 * np.pi, n_split)
        extra = pts[idx] + 40.0 * np.column_stack([np.cos(ang), np.sin(ang)])
        return float(np.median(nn1(np.vstack([pts, extra]))))
    m, s = rep(f)
    C[f"split_{fs}"] = dict(median_1nn_nm=m, sd=s)
print("C over-splitting median 1NN (nm):", {k: round(v["median_1nn_nm"]) for k, v in C.items()})
res["C_oversplitting_40nm"] = C

# D foreign ring clusters at independent arcs
D = {}
for ff in (0.0, 0.1, 0.25, 0.5, 1.0):
    def f(ff=ff):
        a = regular_arcs(30.0)
        pts = ring_points(a)
        n_f = int(round(ff * K))
        extra = ring_points(regular_arcs(30.0)[:n_f] + rng.uniform(0, P))
        return float(np.median(nn1(np.vstack([pts, extra]))))
    m, s = rep(f)
    D[f"foreign_{ff}"] = dict(median_1nn_nm=m, sd=s)
print("D foreign-ring clusters median 1NN (nm):", {k: round(v["median_1nn_nm"]) for k, v in D.items()})
res["D_foreign_ring_fraction"] = D

# E interior clusters
E = {}
for fi in (0.0, 0.05, 0.1, 0.2):
    def f(fi=fi):
        a = regular_arcs(30.0)
        pts = ring_points(a)
        n_i = int(round(fi * K))
        th = rng.uniform(0, 2 * np.pi, n_i)
        r = R * rng.uniform(0.5, 0.9, n_i)
        extra = np.column_stack([r * np.cos(th), r * np.sin(th)])
        return float(np.median(nn1(np.vstack([pts, extra]))))
    m, s = rep(f)
    E[f"interior_{fi}"] = dict(median_1nn_nm=m, sd=s)
print("E interior clusters median 1NN (nm):", {k: round(v["median_1nn_nm"]) for k, v in E.items()})
res["E_interior_fraction"] = E

# F chord vs arc on a folded contour: a circle with one deep inward fold
# (two membrane stretches 100 nm apart over 2 um of arc).
def folded_contour(n=4000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # fold: a Gaussian dent of depth 0.6 R and angular width 0.25 rad
    depth = 0.6 * R * np.exp(-0.5 * ((t - np.pi) / 0.12) ** 2)
    r = R - depth
    return np.column_stack([r * np.cos(t), r * np.sin(t)])

cont = folded_contour()
seg = np.linalg.norm(np.diff(np.vstack([cont, cont[:1]]), axis=0), axis=1)
cum = np.concatenate([[0], np.cumsum(seg)])
L = cum[-1]
F = {}
vals_e, vals_a = [], []
for _ in range(200):
    arcs = np.sort(np.mod((np.arange(K) + 0.5) * L / K + rng.normal(0, 30, K), L))
    idx = np.searchsorted(cum, arcs) - 1
    pts = cont[np.clip(idx, 0, len(cont) - 1)]
    vals_e.append(np.median(nn1(pts)))
    da = np.diff(np.concatenate([arcs, [arcs[0] + L]]))
    arc_nn = np.minimum(da, np.roll(da, 1))
    vals_a.append(np.median(arc_nn))
F = dict(contour_length_nm=float(L), median_euclid_1nn=float(np.mean(vals_e)),
         median_arc_1nn=float(np.mean(vals_a)),
         fraction_of_points_whose_euclid_nn_is_not_arc_neighbour=None)
print("F folded contour: Euclidean median 1NN", round(F["median_euclid_1nn"]),
      "arc median 1NN", round(F["median_arc_1nn"]), "L", round(L))
res["F_chord_vs_arc_folded_contour"] = F

# angular NN offset closed forms
G = {}
for N in (20, 50, 100):
    meds, means = [], []
    for _ in range(500):
        a = np.sort(rng.uniform(0, 360, N))
        b = np.sort(rng.uniform(0, 360, N))
        d = np.abs(a[:, None] - b[None, :])
        d = np.minimum(d, 360 - d).min(axis=1)
        meds.append(np.median(d)); means.append(d.mean())
    G[N] = dict(sim_median=float(np.mean(meds)), closed_median=180 * (1 - 2 ** (-1 / N)),
                sim_mean=float(np.mean(means)), closed_mean=180 / (N + 1), old_90_over_N=90 / N)
print("G angular NN:", {k: (round(v["sim_median"], 2), round(v["closed_median"], 2)) for k, v in G.items()})
res["G_angular_nn_offset"] = G

with open(os.path.join(OUT, "sim3_nn_factors.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
print("done")
