# -*- coding: utf-8 -*-
"""
Why Picasso's least-squares NeNA fit came out low, and what each part of the new estimator fixes
(docs/nena.md, section 2). Synthetic data only.

Files: the channel-2 tests' generator (600 emitters over 25 um, 4000 frames, 1 + Poisson(3) events of
Geometric(0.4) frames, sigma = 12 nm per axis, float32 pixel round trip at 130 nm), seeds 0-19 -- the 20 files
where 5 low values were first seen -- and 20 files of docs/sims/nena/nena_sim.py at sigma 12 nm, ~2000 pairs,
no contamination (seeds 900000-900019).

Estimators, cumulative:
  A  Picasso least squares on the 0.5 nm histogram, 3 starts (the previous estimator; unseeded, as the channel-2
     code calls it)
  B  the same least squares from 36 starts, the smallest residual kept (is it a local-minimum problem?)
  C  maximum likelihood of the same model on the distances, 24 starts, nearby width free (>= 0.05 s)
  D  C with the nearby width >= sqrt(2) s
  E  D with the nearby term kept only when BIC supports it (the new estimator, tools.mps_quality.nena)
and the oracle sqrt(<d^2>/4) of the pairs the nearest-neighbour rule matched to the same molecule.

Also: for seeds 3 and 7 the least-squares residual profiled over s, which shows the two failure modes.

Usage: python docs/sims/nena/nena_diagnosis.py  ->  nena_diagnosis.json (+ nena_diagnosis.png if matplotlib)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from nena_sim import next_frame_pairs, rayleigh_rms_scale, simulate  # noqa: E402
from tools import mps_quality as q  # noqa: E402

D_MAX = 120.0
SIGMA = 12.0


def channel2_field(seed: int):
    rng = np.random.default_rng(seed)
    n_emitters, frames, field_nm = 600, 4000, 25000.0
    cx = rng.uniform(0, field_nm, n_emitters)
    cy = rng.uniform(0, field_nm, n_emitters)
    fr, xs, ys, mol = [], [], [], []
    for i in range(n_emitters):
        for _ in range(rng.poisson(3) + 1):
            start = int(rng.integers(0, frames - 10))
            length = int(rng.geometric(0.4))
            for k in range(length):
                fr.append(start + k)
                xs.append(cx[i] + rng.normal(0, SIGMA))
                ys.append(cy[i] + rng.normal(0, SIGMA))
                mol.append(i)
    order = np.argsort(fr, kind="stable")
    x = (np.asarray(xs)[order] / 130.0).astype(np.float32).astype(float) * 130.0
    y = (np.asarray(ys)[order] / 130.0).astype(np.float32).astype(float) * 130.0
    return np.asarray(fr)[order].astype(np.int64), x, y, np.asarray(mol)[order]


def hist(d):
    edges = np.arange(0.0, D_MAX + 0.5, 0.5)
    counts, edges = np.histogram(d, bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), counts.astype(float)


def lsq_many(d) -> float:
    c, counts = hist(d)
    area = float(np.trapezoid(counts, c))
    med = float(np.median(d)) / 1.665
    best, rss = float("nan"), np.inf
    for s0 in (0.7 * med, med, 1.4 * med):
        for frac in (0.0, 0.1, 0.3):
            for dc0, sc0 in ((2 * s0, s0), (4 * s0, 2 * s0), (0.5 * D_MAX, 0.25 * D_MAX), (D_MAX, D_MAX / 3)):
                try:
                    p, _ = curve_fit(q._picasso_model, c, counts,
                                     p0=[(1 - 2 * frac) * area + 1e-9, s0, frac * area + 1e-9, dc0, sc0],
                                     bounds=([0] * 5, [np.inf] * 5), maxfev=5000)
                except (RuntimeError, ValueError):
                    continue
                r = float(np.sum((q._picasso_model(c, *p) - counts) ** 2))
                if r < rss:
                    rss, best = r, float(p[1])
    return best


def ml_two(d, min_sd_factor: float) -> float:
    saved = q.NENA_NEARBY_MIN_SD_FACTOR
    q.NENA_NEARBY_MIN_SD_FACTOR = min_sd_factor
    try:
        lik = q._NeNALikelihood(d, D_MAX)
        _, best = q._nena_fit_two(lik, q._nena_starts(lik, None))
    finally:
        q.NENA_NEARBY_MIN_SD_FACTOR = saved
    return float(best[0]) if best is not None else float("nan")


def lsq_profile(d) -> List[List[float]]:
    """Smallest least-squares residual at each fixed s, the other four parameters free (many starts)."""
    c, counts = hist(d)
    area = float(np.trapezoid(counts, c))
    out = []
    for s in np.arange(8.0, 14.01, 0.25):
        best = np.inf
        for frac in (1e-6, 0.02, 0.1, 0.3):
            for dc0, sc0 in ((5, 3), (10, 5), (20, 10), (30, 10), (40, 15), (60, 30), (120, 40), (300, 50)):
                def f(dd, da, ac, dc, sc):
                    return q._picasso_model(dd, da, s, ac, dc, sc)
                try:
                    p, _ = curve_fit(f, c, counts, p0=[(1 - 2 * frac) * area, frac * area, dc0, sc0],
                                     bounds=([0] * 4, [np.inf] * 4), maxfev=5000)
                except (RuntimeError, ValueError):
                    continue
                best = min(best, float(np.sum((f(c, *p) - counts) ** 2)))
        out.append([float(s), best])
    return out


def evaluate(frame, x, y, mol) -> Dict[str, Any]:
    d, same = next_frame_pairs(frame, x, y, mol, D_MAX)
    r = q.nena(frame, x, y)
    return dict(n_pairs=int(d.size), A_old=float(r.lsq_nm), B_lsq_36_starts=lsq_many(d),
                C_ml_free_width=ml_two(d, 0.05), D_ml_width_ge_sqrt2s=ml_two(d, float(np.sqrt(2.0))),
                E_new=float(r.precision_nm), oracle=rayleigh_rms_scale(d[same]),
                ci=[float(r.ci_low_nm), float(r.ci_high_nm)], nearby_term_used=bool(r.nearby_term_used),
                flags=list(r.flags))


def table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for key in ("A_old", "B_lsq_36_starts", "C_ml_free_width", "D_ml_width_ge_sqrt2s", "E_new", "oracle"):
        v = np.array([r[key] for r in rows]) / SIGMA
        out[key] = dict(min=float(v.min()), max=float(v.max()), median=float(np.median(v)), sd=float(v.std()),
                        off_more_than_5pct=int(np.sum(np.abs(v - 1) > 0.05)))
    return out


def main() -> None:
    ch2 = []
    for seed in range(20):
        frame, x, y, mol = channel2_field(seed)
        row = evaluate(frame, x, y, mol)
        row["seed"] = seed
        ch2.append(row)
        print("channel-2 file", seed, {k: (round(v / SIGMA, 3) if isinstance(v, float) else v) for k, v in row.items()
                                       if k not in ("ci", "flags")}, flush=True)
    grid = []
    for seed in range(900000, 900020):
        frame, x, y, mol = simulate(SIGMA, n_mol=333, seed=seed)
        row = evaluate(frame, x, y, mol)
        row["seed"] = seed
        grid.append(row)
    profiles = {}
    for seed in (3, 7):
        frame, x, y, mol = channel2_field(seed)
        d, _ = next_frame_pairs(frame, x, y, mol, D_MAX)
        profiles[str(seed)] = lsq_profile(d)
    result = dict(description=__doc__.strip().splitlines()[0], sigma_nm=SIGMA, d_max_nm=D_MAX,
                  channel2_files=dict(summary=table(ch2), files=ch2),
                  sim_files=dict(summary=table(grid), files=grid), lsq_residual_profiles=profiles)
    with open(os.path.join(HERE, "nena_diagnosis.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    for name, part in (("channel-2 files", ch2), ("nena_sim files", grid)):
        print(name)
        for key, s in table(part).items():
            print(f"   {key:22s} {s}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    frame, x, y, mol = channel2_field(3)
    r = q.nena(frame, x, y)
    c, counts = hist(q._next_frame_distances_nm(frame, x, y, D_MAX))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.bar(c, counts, width=0.5, color="#bbbbbb", label="next-frame distances (file 3)")
    ax.plot(c, r.best_fit, color="#0072B2", lw=2, label=f"new: {r.precision_nm:.1f} nm")
    s_old = r.lsq_nm
    ray = counts.sum() * 0.5 * (c / (2 * s_old ** 2)) * np.exp(-c ** 2 / (4 * s_old ** 2))
    ax.plot(c, ray, color="#D55E00", lw=2, ls="--", label=f"old same-molecule term: {s_old:.1f} nm")
    ax.axvline(np.sqrt(2) * SIGMA, color="k", lw=1, ls=":")
    ax.set_xlim(0, 70)
    ax.set_xlabel("distance [nm]")
    ax.set_ylabel("count per 0.5 nm")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for seed, colour in (("3", "#D55E00"), ("7", "#009E73")):
        prof = np.array(profiles[seed])
        ax.plot(prof[:, 0], prof[:, 1] - prof[:, 1].min(), marker="o", ms=3, color=colour, label=f"file {seed}")
    ax.axvline(SIGMA, color="k", lw=1, ls=":")
    ax.set_xlabel("s held fixed [nm] (true 12)")
    ax.set_ylabel("least-squares residual - its minimum")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "nena_diagnosis.png"), dpi=110)


if __name__ == "__main__":
    main()
