# -*- coding: utf-8 -*-
"""
Probes behind the numbers of docs/apparent_size.md that come from THIS repository's code (synthetic data only, R8).
The feasibility numbers quoted as "SIZE_SIM" come from the apparent-size simulation of 2026-10-15, whose generator
is ported in tools/mps_size_synth.py; the probes here re-measure what the shipped estimators do.

Run from the repository root:  py -3 docs/sims/apparent_size/size_probes.py
Writes size_probes.json beside this file (the committed JSON is the run quoted in the docs). ~1-2 minutes, 1 process.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from tools import mps_apparent_size as asz  # noqa: E402
from tools import mps_curvature as curv  # noqa: E402
from tools import mps_size_synth as syn  # noqa: E402

SEED = 20261022


def _r(v: Any, nd: int = 3) -> Any:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


def probe_same_cloud(rng: np.random.Generator) -> Dict[str, Any]:
    """The same Gaussian cloud (sigma 10 nm, one label, no linkage) measured with 20 and with 200 localizations."""
    out: Dict[str, Any] = {}
    for n in (20, 200):
        sd, a15, r15, hull = [], [], [], []
        for i in range(500):
            p = rng.normal(0.0, 10.0, (n, 2))
            cs = asz.cluster_sizes(p[:, 0], p[:, 1], np.zeros(n, dtype=np.int64), seed=i)
            sd.append(cs.cloud_sd[0]); a15.append(cs.hull_area_n15[0]); r15.append(cs.r_rare[0])
            hull.append(asz.hull_area(p))
        out[str(n)] = {"cloud_sd_nm": _r(np.median(sd), 2), "hull_area_n15_nm2": _r(np.median(a15), 1),
                       "r_rare_nm": _r(np.median(r15), 2), "hull_area_all_nm2": _r(np.median(hull), 1),
                       "r_hull_all_nm": _r(math.sqrt(float(np.median(hull)) / math.pi), 2)}
    out["hull_ratio_200_over_20"] = _r(out["200"]["hull_area_all_nm2"] / out["20"]["hull_area_all_nm2"], 2)
    return out


def probe_population(rng: np.random.Generator) -> Dict[str, Any]:
    """One-label and three-label populations of the P-R61 model (rho 25, sigma_loc 10 with 25 % spread, linkage 10):
    Spearman of each estimator with N_loc, and the cloud SD definitions."""
    out: Dict[str, Any] = {}
    for labels in (1, 3):
        xs, ys, labs = [], [], []
        g, t = [], []
        for c in range(2000):
            d = syn.draw_cluster(rng, 25.0, labels)
            if d.t.size < 3:
                continue
            xs.append(d.t + 1000.0 * c); ys.append(d.n); labs.append(np.full(d.t.size, c))
            ev = np.linalg.eigvalsh(np.cov(np.column_stack([d.t, d.n]).T, ddof=1))
            g.append(math.sqrt(math.sqrt(max(ev[0], 0) * ev[1])))
            t.append(math.sqrt(np.std(d.t, ddof=1) * np.std(d.n, ddof=1)))
        x, y, lab = np.concatenate(xs), np.concatenate(ys), np.concatenate(labs)
        cs = asz.cluster_sizes(x, y, lab, rarefy_reps=5)
        nl = cs.n_locs.astype(float)
        hull = np.array([asz.hull_area(np.column_stack([x[lab == c], y[lab == c]])) for c in cs.labels])
        out[f"labels_{labels}"] = {
            "n_clusters": int(cs.n_clusters),
            "spearman_with_n_locs": {
                "hull_area_all (legacy Cluster area)": _r(asz.spearman(hull, nl)[0]),
                "cloud_sd (sqrt(trace/2), shipped)": _r(asz.spearman(cs.cloud_sd, nl)[0]),
                "cloud_sd geometric mean of principal SDs": _r(asz.spearman(np.array(g), nl)[0]),
                "r_disp sqrt(s_t s_n) fixed axes (simulation)": _r(asz.spearman(np.array(t), nl)[0]),
                "radius_gyration": _r(asz.spearman(cs.radius_gyration, nl)[0]),
                "r_rare (n = 15)": _r(asz.spearman(cs.r_rare, nl)[0]),
            },
            "fraction_r_rare_undefined": _r(float(cs.r_rare_undefined.mean())),
            "median_cloud_sd_nm": _r(float(np.median(cs.cloud_sd)), 2),
        }
    return out


def _dbscan(x: np.ndarray, y: np.ndarray, eps: float) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    return np.asarray(DBSCAN(eps=eps, min_samples=10).fit(np.column_stack([x, y])).labels_, dtype=np.int64)


def probe_permutation(rng: np.random.Generator, n_axons: int = 20) -> Dict[str, Any]:
    """Question (a) under a TRUE null (sizes independent of position; 3 labels; one ring of R 2000 nm per axon; 5 %
    background): mark-permutation z of Spearman(N_loc, d_nn) with DBSCAN eps 20/25/30 and with the true clusters."""
    res: Dict[str, List[float]] = {"oracle": [], "eps20": [], "eps25": [], "eps30": []}
    rhos: Dict[str, List[float]] = {k: [] for k in res}
    for _ in range(n_axons):
        c = syn.make_contour("circle", 2000.0, rng)
        ring = syn.generate_ring(rng, c, labels_mean=3.0, background_fraction=0.05)
        cases = {"oracle": ring.truth, "eps20": _dbscan(ring.x, ring.y, 20.0), "eps25": _dbscan(ring.x, ring.y, 25.0),
                 "eps30": _dbscan(ring.x, ring.y, 30.0)}
        for k, lab in cases.items():
            cs = asz.cluster_sizes(ring.x, ring.y, lab, rarefy_reps=1)
            mp = asz.mark_permutation_diagnostic(cs.n_locs.astype(float), cs.d_nn, rng)
            res[k].append(float(mp.z)); rhos[k].append(float(mp.rho))
    out: Dict[str, Any] = {"n_axons": n_axons}
    for k, z in res.items():
        za = np.array(z)
        out[k] = {"mean_spearman": _r(np.mean(rhos[k])), "mean_z": _r(za.mean(), 2),
                  "stouffer_z": _r(za.sum() / math.sqrt(za.size), 2),
                  "fraction_axons_abs_z_gt_1.96": _r(float(np.mean(np.abs(za) > 1.96)), 2)}
    return out


def probe_curvature(n_seeds: int = 5) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kind, R, asp, notch in (("ellipse", 1400.0, 1.6, 0), ("notched", 1400.0, 1.4, 3), ("ellipse", 500.0, 1.6, 0),
                                ("notched", 500.0, 1.4, 3)):
        rs, rm, hit = [], [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng([SEED, seed])
            c = syn.make_contour(kind, R, rng, aspect=asp, n_notch=notch)
            ring = syn.generate_ring(rng, c, labels_mean=3.0)
            cs = asz.cluster_sizes(ring.x, ring.y, ring.truth, rarefy_reps=1)
            m = ring.truth >= 0
            mem = curv.fit_curvature_membrane(ring.x[m], ring.y[m], cs.centroid)
            _, _, kr = mem.at(cs.centroid)
            true = c.kappa_at_points(cs.centroid) * c.R
            rs.append(float(asz.spearman(kr, true)[0])); rm.append(float(np.sqrt(np.mean((kr - true) ** 2))))
            if np.any(true < -0.3):
                hit.append(float(np.mean(kr[true < -0.3] < 0)))
        out[f"{kind} R{int(R)}"] = {"spearman_mean": _r(np.mean(rs)), "spearman_min": _r(min(rs)),
                                    "rmse_kappa_r_mean": _r(np.mean(rm)),
                                    "concave_found_mean": _r(np.mean(hit)) if hit else None}
    return out


def probe_pool(rng: np.random.Generator, n_axons: int = 12) -> Dict[str, Any]:
    """The footprint pool of one ring per axon (circle R 1400 nm, K = 60) with the true clusters ("oracle") and with
    DBSCAN eps 25 / 10, leaving out no cluster, elongation > 1.8 (the flag) or > 2.5 (the pool's cut); and the
    effect of a NeNA off by 2 nm."""
    out: Dict[str, Any] = {}
    cases = [(2.0, 0.0, 0.0), (2.0, 25.0, 0.0), (3.0, 0.0, 0.0), (3.0, 25.0, 0.0), (3.0, 25.0, -2.0),
             (3.0, 25.0, 2.0), (1.0, 25.0, 0.0)]
    for labels, rho, nena_err in cases:
        acc: Dict[str, List[float]] = {}
        verdicts: List[str] = []
        for _ in range(n_axons):
            c = syn.make_contour("circle", 1400.0, rng)
            ring = syn.generate_ring(rng, c, labels_mean=labels, rho=rho, K=60)
            for det, lab in (("oracle", ring.truth), ("dbscan25", _dbscan(ring.x, ring.y, 25.0))):
                cs = asz.cluster_sizes(ring.x, ring.y, lab, event_id=ring.event, curve=c.xy, rarefy_reps=1)
                for cut in (None, 1.8, 2.5):
                    ex = None if cut is None else np.where(np.isfinite(cs.elongation), cs.elongation > cut, False)
                    p = asz.pooled_footprint(cs.sd_tangential, cs.n_locs, cs.n_events, syn.SIGMA_LOC_NM + nena_err,
                                             exclude=ex)
                    if det == "dbscan25" and cut == asz.POOL_EXCLUDE_ELONGATION:
                        verdicts.append(p.verdict)
                    if p.rho_nm is not None:
                        acc.setdefault(f"{det}, cut {cut}", []).append(p.rho_nm)
        out[f"labels {labels:g}, rho {rho:g} nm, NeNA error {nena_err:+g} nm"] = {
            "rho_mean_sd_nm": {k: [_r(np.mean(v), 1), _r(np.std(v, ddof=1), 1) if len(v) > 1 else None]
                               for k, v in acc.items()},
            "verdicts_dbscan_cut_2.5": {v: verdicts.count(v) for v in sorted(set(verdicts))}}
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    result = {"seed": SEED, "same_cloud": probe_same_cloud(rng), "population": probe_population(rng),
              "permutation_diagnostic": probe_permutation(rng), "curvature_400nm": probe_curvature(),
              "footprint_pool": probe_pool(rng)}
    path = os.path.join(HERE, "size_probes.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
