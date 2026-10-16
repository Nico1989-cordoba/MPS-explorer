# -*- coding: utf-8 -*-
"""
Own simulation behind docs/cut_clusters.md (decision P-R59): how well the rule "a cluster is cut when any of its
localizations lies within m NeNA of the pick boundary" finds the clusters a hand-drawn pick really cut, at m = 0.5,
1 and 2, against the localization precision; and how it compares with "centroid selection" (cluster the whole field,
keep a cluster when its centroid is inside the pick).

One simulated field = one axon of the project's simulator (``tools.mps_simulate_axon.simulate_axon``: an ellipse
membrane, clusters on it, fluorophores per cluster, epitope disc, linkage, consecutive-frame blinking, a lateral
error of sd sigma), the first of its two rings (the second is the next MPS ring, 190 nm away in z; the analysis keeps
one ring's slab). A pick is drawn around it like a hand-drawn polygon: 12 corners at jittered angles, each at the
membrane's radius plus a margin uniform in [-150, +250] nm, so the edge runs through the ring in places.

Per field:
  * NeNA measured on the whole field, unlinked (``tools.mps_quality.nena``), as the program does;
  * the per-axon file = the localizations inside the pick (Picasso's even-odd rule);
  * DBSCAN(eps 25 nm, min samples 10; Gazal et al. 2026) on it; each cluster is named by the true cluster most of its
    localizations come from (clusters mostly of no true cluster are left out);
  * truth: a true cluster is CUT when some of its localizations are inside the pick and some outside;
    MATERIALLY cut when it lost at least 10 % of them;
  * the rule at 0.5, 1 and 2 x NeNA (``tools.mps_cut_clusters.find_cut_clusters``);
  * centroid selection: DBSCAN on the whole field, a cluster kept when its centroid is inside the pick.

Reported (per precision): sensitivity (cut clusters flagged / cut clusters), the same for materially cut ones,
specificity (intact clusters not flagged / intact clusters), positive predictive value; and per method of building
the analysed set (no handling; the rule with exclusion; centroid selection): how many clusters it analyses, the share
of them that miss at least 10 % of their localizations, and the median of (localizations analysed / localizations of
the same cluster found by DBSCAN on the whole field, so that DBSCAN's own losses are not counted as cuts).

Also: a Monte Carlo check of the derivation (a localization at distance d inside a straight edge; the chance that a
second localization of the same molecule falls beyond it is Phi(-d / (sqrt(2) sigma))).

Run (4 worker processes, a few minutes):
    venv\\Scripts\\python.exe docs/sims/cut_clusters/cut_sim.py --out docs/sims/cut_clusters/cut_sim.json --workers 4

Seeds: field k of precision sigma uses seed 59000 + 1000 * index(sigma) + k (the simulator and the pick); the
derivation check uses seed 59. Synthetic data only.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SIGMAS_NM = (5.0, 8.0, 12.0, 16.0)
MULTIPLES = (0.5, 1.0, 2.0)
N_FIELDS = 200
EPS_NM, MIN_SAMPLES = 25.0, 10
MATERIAL = 0.10
MARGIN_RANGE_NM = (-150.0, 250.0)
N_CORNERS = 12


def _polar_radius(contour: np.ndarray, centre: np.ndarray, angles: np.ndarray) -> np.ndarray:
    d = contour - centre
    th = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])
    order = np.argsort(th)
    th, r = th[order], r[order]
    th = np.concatenate([th - 2 * np.pi, th, th + 2 * np.pi])
    r = np.concatenate([r, r, r])
    return np.asarray(np.interp(angles, th, r), dtype=float)


def one_field(args: Any) -> Dict[str, Any]:
    sigma, k, seed = args
    from sklearn.cluster import DBSCAN

    from tools.mps_cut_clusters import find_cut_clusters
    from tools.mps_picks import PickBoundary, polygon_contains
    from tools.mps_quality import nena
    from tools.mps_simulate_axon import SimConfig, simulate_axon

    rng = np.random.default_rng(seed)
    a = float(rng.uniform(600.0, 1500.0))
    b = float(a * rng.uniform(0.55, 1.0))
    cfg = SimConfig(ellipse_semi_axes_nm=(a, b), n_rings=2, lp_lateral_samples_nm=np.array([sigma]),
                    n_fluor_mean=15.0, locs_per_fluor_mean=3.0, background_per_um3=0.0)
    ax = simulate_axon(cfg, seed)
    field = nena(ax.frame, ax.x_nm, ax.y_nm)
    ring0 = ax.ring_true == 0
    x, y, truth = ax.x_nm[ring0], ax.y_nm[ring0], ax.cluster_true[ring0]
    centre = np.array([float(np.mean(ax.contour_used_nm[:, 0])), float(np.mean(ax.contour_used_nm[:, 1]))])
    angles = np.sort((np.arange(N_CORNERS) + rng.uniform(-0.3, 0.3, N_CORNERS)) * 2 * np.pi / N_CORNERS - np.pi)
    radius = _polar_radius(ax.contour_used_nm, centre, angles) + rng.uniform(*MARGIN_RANGE_NM, N_CORNERS)
    poly = np.column_stack([centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)])
    inside = polygon_contains(x, y, poly)
    boundary = PickBoundary(kind="polygon", shape="Polygon", source="simulation", vertices_nm=poly)

    ids = np.unique(truth[truth >= 0])
    n_total = {int(c): int(np.sum(truth == c)) for c in ids}
    n_in = {int(c): int(np.sum((truth == c) & inside)) for c in ids}
    cut_true = {c for c in n_total if 0 < n_in[c] < n_total[c]}
    material = {c for c in cut_true if 1.0 - n_in[c] / n_total[c] >= MATERIAL}

    # the reference: DBSCAN on the whole field (this ring); each true cluster's size is that of the largest
    # whole-field cluster it is the majority of, so DBSCAN's own losses are not counted as cuts
    lab_all = DBSCAN(eps=EPS_NM, min_samples=MIN_SAMPLES).fit(np.column_stack([x, y])).labels_
    ref: Dict[int, int] = {}
    whole: List[Any] = []
    for lab in np.unique(lab_all[lab_all >= 0]):
        sel = lab_all == lab
        values, counts = np.unique(truth[sel], return_counts=True)
        top = int(values[np.argmax(counts)])
        if top < 0:
            continue
        ref[top] = max(ref.get(top, 0), int(sel.sum()))
        whole.append((sel, top))

    xi, yi, ti = x[inside], y[inside], truth[inside]
    labels = DBSCAN(eps=EPS_NM, min_samples=MIN_SAMPLES).fit(np.column_stack([xi, yi])).labels_
    named: Dict[int, int] = {}
    n_locs: Dict[int, int] = {}
    for lab in np.unique(labels[labels >= 0]):
        members = ti[labels == lab]
        values, counts = np.unique(members, return_counts=True)
        top = int(values[np.argmax(counts)])
        if top < 0:
            continue
        named[int(lab)] = top
        n_locs[int(lab)] = int(np.sum(labels == lab))
    nena_nm = float(field.precision_nm)
    out: Dict[str, Any] = dict(sigma=sigma, field=k, seed=seed, nena_nm=nena_nm, nena_pairs=int(field.n_pairs),
                               n_true_clusters=len(ids), n_true_cut=len(cut_true), n_true_material=len(material),
                               n_detected=len(named), rule={}, methods={})
    for m in MULTIPLES:
        flagged, _ = find_cut_clusters(xi, yi, labels, list(named), boundary, m * nena_nm)
        tp = sum(1 for lab in named if lab in flagged and named[lab] in cut_true)
        fn = sum(1 for lab in named if lab not in flagged and named[lab] in cut_true)
        fp = sum(1 for lab in named if lab in flagged and named[lab] not in cut_true)
        tn = sum(1 for lab in named if lab not in flagged and named[lab] not in cut_true)
        tpm = sum(1 for lab in named if lab in flagged and named[lab] in material)
        fnm = sum(1 for lab in named if lab not in flagged and named[lab] in material)
        out["rule"][str(m)] = dict(tp=tp, fn=fn, fp=fp, tn=tn, tp_material=tpm, fn_material=fnm)
        if m == 1.0:
            kept = [lab for lab in named if lab not in flagged]
            out["methods"]["rule_excluded"] = _set_metrics([(n_locs[lab], named[lab]) for lab in kept], ref)
    out["methods"]["no_handling"] = _set_metrics([(n_locs[lab], named[lab]) for lab in named], ref)

    # centroid selection: the whole-field clusters, kept when the centroid is inside the pick
    chosen = []
    straddling_in = straddling_out = 0
    for sel, top in whole:
        cx_, cy_ = float(np.mean(x[sel])), float(np.mean(y[sel]))
        keep = bool(polygon_contains(np.array([cx_]), np.array([cy_]), poly)[0])
        if top in cut_true:
            straddling_in += int(keep)
            straddling_out += int(not keep)
        if keep:
            chosen.append((int(sel.sum()), top))
    out["methods"]["centroid_selection"] = _set_metrics(chosen, ref)
    # the true clusters with localizations in the pick that a method leaves out entirely
    reached = {c for c in n_in if n_in[c] > 0 and c in ref}
    out["lost_no_handling"] = len(reached - set(named.values()))
    out["lost_centroid_selection"] = len(reached - {t for _n, t in chosen})
    out["centroid_straddling_kept"] = straddling_in
    out["centroid_straddling_dropped"] = straddling_out
    return out


def _set_metrics(items: List[Any], n_total: Dict[int, int]) -> Dict[str, Any]:
    ratios = [n / n_total[t] for n, t in items if t in n_total and n_total[t] > 0]
    damaged = sum(1 for r in ratios if r < 1.0 - MATERIAL)
    return dict(n=len(items), damaged=damaged, ratios=[round(r, 4) for r in ratios])


def derivation_check(seed: int = 59, n: int = 400_000) -> Dict[str, Any]:
    """A straight edge at 0; molecules at true signed distance u, two localizations of each with sd sigma along the
    normal. Among the localizations observed inside at distance d (+-0.05 sigma), the share whose sibling falls
    beyond the edge. "both_sides": molecules uniform on both sides of the edge (the pick cuts through structure; the
    flat-prior derivation, Phi(-d / (sqrt 2 sigma))); "inside_only": every molecule inside (no structure beyond the
    edge, where only the localization error can carry a sibling out), which gives a lower share."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {}
    sigma = 10.0
    for name, lo in (("both_sides", -200.0), ("inside_only", 0.0)):
        u = rng.uniform(lo, 200.0, n)
        a = u + rng.normal(0.0, sigma, n)
        b = u + rng.normal(0.0, sigma, n)
        rows = {}
        for d in (0.5, 1.0, 2.0):
            sel = np.abs(a - d * sigma) < 0.05 * sigma
            rows[str(d)] = dict(monte_carlo=round(float(np.mean(b[sel] < 0.0)), 4), n=int(sel.sum()))
        out[name] = rows
    out["theory_both_sides"] = {str(d): round(0.5 * math.erfc(d / 2.0), 4) for d in (0.5, 1.0, 2.0)}
    return out


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for sigma in SIGMAS_NM:
        rs = [r for r in rows if r["sigma"] == sigma]
        s: Dict[str, Any] = dict(n_fields=len(rs), nena_over_sigma_median=float(np.median([r["nena_nm"] / sigma
                                                                                            for r in rs])),
                                 detected_per_field=float(np.mean([r["n_detected"] for r in rs])),
                                 cut_true_per_field=float(np.mean([r["n_true_cut"] for r in rs])))
        for m in MULTIPLES:
            t = {key: sum(r["rule"][str(m)][key] for r in rs) for key in rs[0]["rule"][str(m)]}
            s[f"rule_{m:g}"] = dict(
                sensitivity=t["tp"] / max(1, t["tp"] + t["fn"]),
                sensitivity_material=t["tp_material"] / max(1, t["tp_material"] + t["fn_material"]),
                specificity=t["tn"] / max(1, t["tn"] + t["fp"]),
                ppv=t["tp"] / max(1, t["tp"] + t["fp"]), **t)
        for method in ("no_handling", "rule_excluded", "centroid_selection"):
            n = sum(r["methods"][method]["n"] for r in rs)
            damaged = sum(r["methods"][method]["damaged"] for r in rs)
            ratios = [v for r in rs for v in r["methods"][method]["ratios"]]
            s[method] = dict(clusters_per_field=n / max(1, len(rs)), damaged_share=damaged / max(1, n),
                             median_locs_ratio=float(np.median(ratios)) if ratios else float("nan"))
        s["lost_no_handling_per_field"] = float(np.mean([r["lost_no_handling"] for r in rs]))
        s["lost_centroid_selection_per_field"] = float(np.mean([r["lost_centroid_selection"] for r in rs]))
        s["centroid_straddling_kept"] = sum(r["centroid_straddling_kept"] for r in rs)
        s["centroid_straddling_dropped"] = sum(r["centroid_straddling_dropped"] for r in rs)
        summary[f"{sigma:g}"] = s
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "cut_sim.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fields", type=int, default=N_FIELDS)
    args = ap.parse_args()
    jobs = [(s, k, 59000 + 1000 * i + k) for i, s in enumerate(SIGMAS_NM) for k in range(args.fields)]
    t0 = time.time()
    with Pool(max(1, min(4, args.workers))) as pool:
        rows = pool.map(one_field, jobs, chunksize=4)
    result = dict(script="docs/sims/cut_clusters/cut_sim.py", seeds="59000 + 1000 * index(sigma) + field",
                  sigmas_nm=list(SIGMAS_NM), multiples=list(MULTIPLES), fields_per_sigma=args.fields,
                  eps_nm=EPS_NM, min_samples=MIN_SAMPLES, material_loss=MATERIAL,
                  pick_margin_nm=list(MARGIN_RANGE_NM), pick_corners=N_CORNERS,
                  derivation=derivation_check(), summary=summarise(rows), fields=rows,
                  seconds=round(time.time() - t0, 1))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps(dict(derivation=result["derivation"], summary=result["summary"]), indent=1))
    print(f"{len(rows)} fields in {result['seconds']} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
