# -*- coding: utf-8 -*-
"""
Synthetic checks of the eclipse tolerance (P-R61; ``tools/mps_eclipse_tolerance.py``). SYNTHETIC DATA ONLY (R8).

The generator is the P-R61 research's (its ``sims/common.py`` and ``sims/sim2_eclipse_rules.py``, 2026-10-15),
copied here so that the repository reproduces the numbers of ``docs/eclipse_tolerance.md``:

* a cluster of beta-II-spectrin in the (tangent, normal) frame of the membrane: footprint = a disc of radius rho
  in the membrane plane (tangent sd rho / 2), N_label antibody stacks each at its own epitope with a FIXED linkage
  offset N(0, sigma_link) per axis, a geometric number of events per label (mean c_label = 14 [lit: Dempsey et al.
  2011]), a geometric number of localizations per event (mean 2 [ad hoc: the project simulator]) and a per-
  localization precision N(0, sigma_loc), sigma_loc log-normal around 10 nm with CV 0.25 [lit: Huang et al. 2008];
* two rings on a circular membrane of radius 1000 nm, K ~ Poisson(31) clusters (mean spacing 200 nm [lit: Gazal et
  al. 2026, ~200 nm]) with an 80 nm hard core; ring B copies a fraction f of ring A's arc positions (columns) with a
  structural jitter sigma_col, the rest fresh; a rigid N(0, 2 nm) registration per ring and 5 % background.

Two experiments (``python eclipse_sim.py --out eclipse_sim.json``; ~2-4 min, one process):

1. ``merged``: the merged-cluster criterion (decision 4a). Rings measured, clustered with DBSCAN eps 25 nm /
   min_samples 10 [lit: Gazal et al. 2026] (scikit-learn), each detected cluster's truth by majority and its
   "merged" label when two true clusters each give >= 25 % of their localizations to it (the research's
   definition); the criterion of ``merged_cluster_flags`` against that truth, and the pooled footprint with and
   without the exclusion against the true root-mean-square footprint.
2. ``rules``: false-positive rate under the null (f = 0) and power with columns (f = 0.3) of three rules, all
   through the program's own ``arc_eclipse_test`` with the same rotation draws: the scalar tau = 31 nm, the
   per-pair rule (t_ij from ``axon_tolerances``) and a WIDE scalar tau of 80 nm [ad hoc: the order of magnitude
   of the contact-type tolerances the research found weak, to reproduce its power ordering].

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools import mps_eclipse_tolerance as et  # noqa: E402

R_NM = 1000.0
L_NM = 2.0 * math.pi * R_NM
P_BAR_NM = 200.0
D_MIN_NM = 80.0
EPS_NM, MIN_SAMPLES = 25.0, 10
BACKGROUND_FRACTION = 0.05
SEED = 20261022

# Heterogeneity scenarios of the research (rho mixture, label prior)
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "H0_rho25_ztp3": dict(rho_mix=[(25.0, 1.0)], label_mode="ztp", lam=3.0),
    "H1_mixed_ztp3": dict(rho_mix=[(0.0, 0.5), (25.0, 0.3), (40.0, 0.2)], label_mode="ztp", lam=3.0),
    "H2_rho0_one_label": dict(rho_mix=[(0.0, 1.0)], label_mode="fixed", lam=1),
    "H4_rho40_ztp3": dict(rho_mix=[(40.0, 1.0)], label_mode="ztp", lam=3.0),
}


# --------------------------------------------------------------------------------------------- generator
def draw_labels(rng: np.random.Generator, n: int, mode: str, lam: float) -> NDArray[np.int64]:
    if mode == "fixed":
        return np.full(n, int(round(lam)), dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    todo = np.ones(n, dtype=bool)
    while todo.any():
        idx = np.flatnonzero(todo)
        k = rng.poisson(lam, idx.size)
        out[idx] = k
        todo[idx[k > 0]] = False
    return out


def draw_cluster(rng: np.random.Generator, rho: float, n_label: int, *, c_label: float = 14.0,
                 locs_per_event: float = 2.0, sigma_loc: float = 10.0, sigma_loc_cv: float = 0.25,
                 sigma_link: float = 10.0) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """(t, n, event) of one cluster's localizations, centred on its true centre (research ``draw_cluster``)."""
    r = rho * np.sqrt(rng.uniform(size=n_label))
    phi = rng.uniform(0.0, 2.0 * np.pi, n_label)
    ep_t = r * np.cos(phi)
    link = rng.normal(0.0, sigma_link, (n_label, 2))
    lab_t, lab_n = ep_t + link[:, 0], link[:, 1]
    n_ev = rng.geometric(1.0 / c_label, n_label).astype(np.int64)
    ev_label = np.repeat(np.arange(n_label), n_ev)
    n_loc = rng.geometric(1.0 / locs_per_event, int(n_ev.sum())).astype(np.int64)
    loc_event = np.repeat(np.arange(int(n_ev.sum())), n_loc)
    loc_label = ev_label[loc_event]
    total = int(n_loc.sum())
    v = math.log1p(sigma_loc_cv ** 2)
    s = sigma_loc * np.exp(rng.normal(-0.5 * v, math.sqrt(v), total))
    t = lab_t[loc_label] + rng.normal(size=total) * s
    n = lab_n[loc_label] + rng.normal(size=total) * s
    return t, n, loc_event


def cyc(d: NDArray[np.float64]) -> NDArray[np.float64]:
    d = np.mod(d, L_NM)
    return np.asarray(np.minimum(d, L_NM - d), dtype=np.float64)


def hard_core_arcs(rng: np.random.Generator, n: int, fixed: Sequence[float] = ()) -> NDArray[np.float64]:
    placed = np.asarray(list(fixed), dtype=float)
    out = np.empty(n)
    for k in range(n):
        for _ in range(20000):
            s = float(rng.uniform(0.0, L_NM))
            if placed.size == 0 or np.all(cyc(placed - s) >= D_MIN_NM):
                break
        else:
            raise RuntimeError("hard core jammed")
        placed = np.append(placed, s)
        out[k] = s
    return out


def make_rings(rng: np.random.Generator, f: float, sigma_col: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    K = max(5, int(rng.poisson(L_NM / P_BAR_NM)))
    sA = hard_core_arcs(rng, K)
    n_copy = int(round(f * K))
    parents = np.sort(rng.choice(K, n_copy, replace=False)) if n_copy else np.zeros(0, dtype=int)
    s_copy = np.mod(sA[parents] + rng.normal(0.0, sigma_col, n_copy), L_NM)
    s_fresh = hard_core_arcs(rng, K - n_copy, fixed=s_copy.tolist())
    return sA, np.concatenate([s_copy, s_fresh])


def measure_ring(rng: np.random.Generator, s_true: NDArray[np.float64], scen: Dict[str, Any], ev_offset: int
                 ) -> Dict[str, Any]:
    """The localizations of one ring (x, y, event id, true cluster id, -1 background) and each true cluster's rho."""
    K = s_true.size
    vals, probs = zip(*scen["rho_mix"])
    rho = rng.choice(np.asarray(vals, dtype=float), K, p=np.asarray(probs, dtype=float))
    labels = draw_labels(rng, K, scen["label_mode"], scen["lam"])
    th = s_true / R_NM
    reg = rng.normal(0.0, 2.0, 2)
    xs, ys, evs, tru = [], [], [], []
    for i in range(K):
        t, n, ev = draw_cluster(rng, float(rho[i]), int(labels[i]))
        tx, ty, nx, ny = -math.sin(th[i]), math.cos(th[i]), math.cos(th[i]), math.sin(th[i])
        xs.append(R_NM * math.cos(th[i]) + t * tx + n * nx + reg[0])
        ys.append(R_NM * math.sin(th[i]) + t * ty + n * ny + reg[1])
        evs.append(ev + ev_offset)
        ev_offset += int(ev.max()) + 1 if ev.size else 0
        tru.append(np.full(t.size, i))
    x, y, event, truth = (np.concatenate(v) for v in (xs, ys, evs, tru))
    n_bg = int(rng.poisson(BACKGROUND_FRACTION * x.size))
    rb = np.sqrt(rng.uniform((R_NM - 150) ** 2, (R_NM + 150) ** 2, n_bg))
    tb = rng.uniform(0, 2 * np.pi, n_bg)
    x = np.concatenate([x, rb * np.cos(tb)])
    y = np.concatenate([y, rb * np.sin(tb)])
    event = np.concatenate([event, np.arange(n_bg) + ev_offset])
    truth = np.concatenate([truth, np.full(n_bg, -1)])
    return dict(x=x, y=y, event=event.astype(np.int64), truth=truth.astype(np.int64), rho=rho, K=K,
                ev_offset=ev_offset + n_bg)


def dbscan_labels(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.int64]:
    from sklearn.cluster import DBSCAN
    return np.asarray(DBSCAN(eps=EPS_NM, min_samples=MIN_SAMPLES).fit(np.column_stack([x, y])).labels_, dtype=np.int64)


def truth_merged(labels: NDArray[np.int64], truth: NDArray[np.int64], K: int) -> Tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.int64]]:
    """For each detected cluster (ascending label): its majority true cluster (-1 none), whether it is MERGED (two
    true clusters each give >= 25 % of their localizations to it; the research's definition)."""
    ids = np.unique(labels[labels >= 0])
    true_locs = np.bincount(truth[truth >= 0], minlength=K)
    major = np.full(ids.size, -1, dtype=np.int64)
    merged = np.zeros(ids.size, dtype=bool)
    for j, lid in enumerate(ids.tolist()):
        tr = truth[labels == lid]
        tr = tr[tr >= 0]
        if tr.size:
            c = np.bincount(tr, minlength=K)
            major[j] = int(np.argmax(c))
            merged[j] = int(np.sum(c / np.maximum(true_locs, 1) >= 0.25)) >= 2
    return ids, merged, major


# --------------------------------------------------------------------------------------------- experiment 1
def merged_experiment(n_axons: int, seed: int = SEED, scenarios: Sequence[str] = tuple(SCENARIOS)) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    inp = et.ToleranceInputs()
    for si, name in enumerate(scenarios):
        scen = SCENARIOS[name]
        fp_single = n_single = tp = n_merged = 0
        fp_oracle = n_oracle = 0
        pool_all: List[float] = []
        pool_excl: List[float] = []
        pool_oracle: List[float] = []
        pool_oracle_excl: List[float] = []
        rms_true: List[float] = []
        for a in range(n_axons):
            rng = np.random.default_rng(np.random.SeedSequence([seed, si, a]))
            sA, sB = make_rings(rng, 0.0, 0.0)
            A = measure_ring(rng, sA, scen, 0)
            B = measure_ring(rng, sB, scen, A["ev_offset"])
            x = np.concatenate([A["x"], B["x"]])
            y = np.concatenate([A["y"], B["y"]])
            ev = np.concatenate([A["event"], B["event"]])
            nA = A["x"].size
            groups: Dict[Any, List[NDArray[np.intp]]] = {}
            ogroups: Dict[Any, List[NDArray[np.intp]]] = {}
            merged_truth: List[NDArray[np.bool_]] = []
            for key, R, off in ((0, A, 0), (1, B, nA)):
                lab = dbscan_labels(R["x"], R["y"])
                ids, mg, _ = truth_merged(lab, R["truth"], R["K"])
                groups[key] = [np.flatnonzero(lab == v) + off for v in ids.tolist()]
                merged_truth.append(mg)
                ogroups[key] = [np.flatnonzero(R["truth"] == i) + off for i in range(R["K"])]
            centres = {0: [0.0, 0.0], 1: [0.0, 0.0]}
            tol = et.axon_tolerances(x, y, groups, event_id=ev, centres_nm=centres, inputs=inp, seed=a)
            otol = et.axon_tolerances(x, y, ogroups, event_id=ev, centres_nm=centres, inputs=inp, seed=a)
            flags = np.concatenate([tol.groups[0].merged_suspect, tol.groups[1].merged_suspect])
            mt = np.concatenate(merged_truth)
            fp_single += int(np.count_nonzero(flags & ~mt))
            n_single += int(np.count_nonzero(~mt))
            tp += int(np.count_nonzero(flags & mt))
            n_merged += int(np.count_nonzero(mt))
            of = np.concatenate([otol.groups[0].merged_suspect, otol.groups[1].merged_suspect])
            fp_oracle += int(np.count_nonzero(of))
            n_oracle += int(of.size)
            rms_true.append(float(np.sqrt(np.mean(np.concatenate([A["rho"], B["rho"]]) ** 2))))
            if tol.pool.identifiable:
                pool_all.append(float(tol.pool.rho_pooled_nm))
                if tol.pool.rho_pooled_excluding_merged_nm is not None:
                    pool_excl.append(float(tol.pool.rho_pooled_excluding_merged_nm))
            if otol.pool.identifiable:
                pool_oracle.append(float(otol.pool.rho_pooled_nm))
                if otol.pool.rho_pooled_excluding_merged_nm is not None:
                    pool_oracle_excl.append(float(otol.pool.rho_pooled_excluding_merged_nm))

        def ms(v: List[float]) -> Optional[List[float]]:
            return [float(np.mean(v)), float(np.std(v))] if v else None

        out[name] = dict(
            n_axons=n_axons, threshold_nm=et.merged_threshold_nm(inp),
            flag_rate_oracle_singles=fp_oracle / max(n_oracle, 1), n_oracle_clusters=n_oracle,
            flag_rate_dbscan_singles=fp_single / max(n_single, 1), n_dbscan_singles=n_single,
            detection_rate_dbscan_merged=tp / max(n_merged, 1), n_dbscan_merged=n_merged,
            rho_true_rms=ms(rms_true), rho_pool_oracle=ms(pool_oracle), rho_pool_oracle_excluded=ms(pool_oracle_excl),
            rho_pool_dbscan_all=ms(pool_all),
            rho_pool_dbscan_excluded=ms(pool_excl), n_identifiable_dbscan=len(pool_all))
    return out


def merged_z_sweep(n_axons: int, z_values: Sequence[float] = (2.576, 3.29, 3.89),
                   scenarios: Sequence[str] = ("H0_rho25_ztp3", "H1_mixed_ztp3", "H4_rho40_ztp3")) -> Dict[str, Any]:
    """The merged experiment at several MERGED_Z (the evidence behind the chosen 3.29): the module constant is set
    for the call and restored."""
    keep = et.MERGED_Z
    out: Dict[str, Any] = {}
    try:
        for z in z_values:
            et.MERGED_Z = float(z)
            out[f"{z:g}"] = merged_experiment(n_axons, scenarios=scenarios)
    finally:
        et.MERGED_Z = keep
    return out


# --------------------------------------------------------------------------------------------- experiment 2
RULES = ("scalar31", "per_pair", "scalar80_wide")


def rules_one_axon(rng: np.random.Generator, scen: Dict[str, Any], f: float, n_null: int, seed: int,
                   use_dbscan: bool = False) -> Dict[str, Tuple[int, float]]:
    """(matched, p_excess) of each rule on one synthetic two-ring axon, through tools.mps_unroll.arc_eclipse_test
    with the same draws for every rule. Arcs: the clusters' centroid angles on the circle."""
    from tools.mps_unroll import arc_eclipse_test
    sA, sB = make_rings(rng, f, 0.0)
    A = measure_ring(rng, sA, scen, 0)
    B = measure_ring(rng, sB, scen, A["ev_offset"])
    x = np.concatenate([A["x"], B["x"]])
    y = np.concatenate([A["y"], B["y"]])
    ev = np.concatenate([A["event"], B["event"]])
    nA = A["x"].size
    groups: Dict[Any, List[NDArray[np.intp]]] = {}
    for key, R, off in ((0, A, 0), (1, B, nA)):
        if use_dbscan:
            lab = dbscan_labels(R["x"], R["y"])
            ids = np.unique(lab[lab >= 0])
            groups[key] = [np.flatnonzero(lab == v) + off for v in ids.tolist()]
        else:
            groups[key] = [np.flatnonzero(R["truth"] == i) + off for i in range(R["K"])]
    tol = et.axon_tolerances(x, y, groups, event_id=ev, centres_nm={0: [0.0, 0.0], 1: [0.0, 0.0]}, seed=seed)
    arcs = {k: np.mod(np.arctan2(g.centroid_nm[:, 1], g.centroid_nm[:, 0]), 2 * np.pi) * R_NM
            for k, g in tol.groups.items()}
    out: Dict[str, Tuple[int, float]] = {}
    for rule in RULES:
        kw: Dict[str, Any] = {}
        tau = 31.0
        if rule == "per_pair":
            kw = dict(sigma_a_nm=tol.groups[0].sigma_c_nm, sigma_b_nm=tol.groups[1].sigma_c_nm)
        elif rule == "scalar80_wide":
            tau = 80.0
        m = arc_eclipse_test(arcs[0], arcs[1], L_NM, tau, n_null=n_null, random_seed=seed, **kw)
        out[rule] = (int(m.n_matched), float(m.p_excess))
    return out


def rules_experiment(n_axons: int, n_null: int, seed: int = SEED,
                     cells: Sequence[Tuple[str, float]] = (("H1_mixed_ztp3", 0.0), ("H1_mixed_ztp3", 0.3),
                                                           ("H2_rho0_one_label", 0.0), ("H4_rho40_ztp3", 0.0)),
                     use_dbscan: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for ci, (name, f) in enumerate(cells):
        rej = {r: 0 for r in RULES}
        for a in range(n_axons):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 7, ci, a]))
            res = rules_one_axon(rng, SCENARIOS[name], f, n_null, seed=a, use_dbscan=use_dbscan)
            for r in RULES:
                rej[r] += int(res[r][1] <= 0.05)
        out[f"{name}|f={f}"] = {r: rej[r] / n_axons for r in RULES} | {"n_axons": n_axons, "n_null": n_null}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "eclipse_sim.json"))
    ap.add_argument("--merged-axons", type=int, default=40)
    ap.add_argument("--rules-axons", type=int, default=100)
    ap.add_argument("--null", type=int, default=199)
    args = ap.parse_args()
    t0 = time.time()
    res: Dict[str, Any] = {"seed": SEED, "generator": "P-R61 research sims/common.py + sim2 geometry (copied here)"}
    res["merged"] = merged_experiment(args.merged_axons)
    print(json.dumps(res["merged"], indent=1), flush=True)
    res["merged_z_sweep"] = merged_z_sweep(args.merged_axons)
    print(json.dumps(res["merged_z_sweep"], indent=1), flush=True)
    res["rules"] = rules_experiment(args.rules_axons, args.null)
    print(json.dumps(res["rules"], indent=1), flush=True)
    res["rules_dbscan"] = rules_experiment(args.rules_axons, args.null, cells=(("H1_mixed_ztp3", 0.0), ("H1_mixed_ztp3", 0.3)),
                                           use_dbscan=True)
    print(json.dumps(res["rules_dbscan"], indent=1), flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"elapsed {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
