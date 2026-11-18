# -*- coding: utf-8 -*-
"""
Decision D2 (2026-10-23): what the eclipse test should do with a cluster flagged as MERGED -- keep its one middle
centre, SPLIT it into the two halves of its two-means split, or EXCLUDE it (``tools/mps_eclipse_merged.py``).
SYNTHETIC DATA ONLY (R8).

Generator: the P-R61 research's (``eclipse_sim.py`` beside this file: two rings on a circle of radius 1000 nm,
K ~ Poisson(31) clusters with an 80 nm hard core, ring B copying a fraction f of ring A's arcs = the columns; mixed
footprints 0 / 25 / 40 nm, zero-truncated Poisson labels of mean ~3, c_label 14 events per label, 2 localizations
per event, 5 % background). On top of it, PLANTED FUSIONS: in each ring, independently, a share of the reported
clusters are two true clusters linked into one -- a true cluster plus a partner placed 50 to 100 nm away along the
ring [ad hoc: the centre separations at which DBSCAN 25 / 10 links two clusters of this generator; the merged
criterion detects about half of them, in line with the 56-62 % it detected of the DBSCAN merges of
docs/eclipse_tolerance.md sec. 6]. The partner is an extra cluster, never a column (a fusion does not create a
column; it hides the one its parent may have). Fusion rate r = merged clusters / reported clusters: m =
round(r K / (1 + r)) fusions per ring. Clusters are the TRUE clusters with the planted fusions as labels (no
clustering noise: the comparison is about the handling, not about DBSCAN).

For each synthetic axon, the program's own ``axon_tolerances`` flags the merged-looking clusters (z 3.29) and
``centres_with_handling`` gives the centres under the three handlings, plus "truth" (every true cluster with its
own centre: the best any handling can do). Each enters ``tools.mps_unroll.arc_eclipse_test`` with the same
rotation draws (seed), under the scalar tau = 31 nm (PRIMARY) and the per-pair rule (exploratory). Reported per
cell (f, r): the share of axons with p_excess <= 0.05 (false "eclipsed" under f = 0, power under f = 0.3) with its
Wilson 95 % interval, the share of planted fusions flagged and of single clusters flagged.

Run (two processes at most, resumable; ~15-25 min per shard for 400 axons per cell):
    py -3 merged_handling_sim.py --shard 0/2
    py -3 merged_handling_sim.py --shard 1/2
    py -3 merged_handling_sim.py --summarise
The per-axon rows go to merged_handling_sim.jsonl (NOT committed: regenerate), the summary to
merged_handling_sim.json (committed).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eclipse_sim as es  # noqa: E402
from tools import mps_eclipse_merged as mg  # noqa: E402
from tools import mps_eclipse_tolerance as et  # noqa: E402
from tools.mps_unroll import arc_eclipse_test  # noqa: E402

SEED = 20261029
SCENARIO = "H1_mixed_ztp3"
COLUMN_FRACTIONS: Tuple[float, ...] = (0.0, 0.3)       # f = 0: independent rings; 0.3 as eclipse_sim's power cell
FUSION_RATES: Tuple[float, ...] = (0.0, 0.10, 0.25)    # decision D2: 0, 10, 25 %
PARTNER_NM: Tuple[float, float] = (50.0, 100.0)        # [ad hoc: see the module docstring]
TAU_NM = 31.0                                          # the pre-specified scalar tau [simulation: P-R61]
HANDLINGS: Tuple[str, ...] = ("truth",) + mg.MERGED_HANDLINGS
RULES: Tuple[str, ...] = ("scalar31", "per_pair")
JSONL = os.path.join(HERE, "merged_handling_sim.jsonl")
JSON_OUT = os.path.join(HERE, "merged_handling_sim.json")


def ring_with_fusions(rng: np.random.Generator, s_true: NDArray[np.float64], rate: float, ev_offset: int
                      ) -> Dict[str, Any]:
    """One ring measured with planted fusions: the localizations, the reported clusters (members), the TRUE clusters
    (truth_members) and which reported clusters are planted fusions."""
    K = int(s_true.size)
    m = int(round(rate * K / (1.0 + rate)))
    parents = np.sort(rng.choice(K, m, replace=False)) if m else np.zeros(0, dtype=np.int64)
    delta = rng.uniform(PARTNER_NM[0], PARTNER_NM[1], m) * rng.choice([-1.0, 1.0], m)
    s_all = np.concatenate([s_true, np.mod(s_true[parents] + delta, es.L_NM)])
    R = es.measure_ring(rng, s_all, es.SCENARIOS[SCENARIO], ev_offset)
    truth = np.asarray(R["truth"], dtype=np.int64)
    label = truth.copy()
    for j, par in enumerate(parents.tolist()):
        label[truth == K + j] = par
    planted = np.zeros(K, dtype=bool)
    planted[parents] = True
    return dict(x=R["x"], y=R["y"], event=R["event"], ev_offset=R["ev_offset"],
                members=[np.flatnonzero(label == i) for i in range(K)],
                truth_members=[np.flatnonzero(truth == i) for i in range(K + m)], planted=planted)


def _arcs(c: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.mod(np.arctan2(c[:, 1], c[:, 0]), 2.0 * np.pi) * es.R_NM, dtype=np.float64)


def one_axon(f: float, rate: float, a: int, n_null: int) -> Dict[str, Any]:
    ci = COLUMN_FRACTIONS.index(f) * 10 + FUSION_RATES.index(rate)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, ci, a]))
    sA, sB = es.make_rings(rng, f, 0.0)
    A = ring_with_fusions(rng, sA, rate, 0)
    B = ring_with_fusions(rng, sB, rate, int(A["ev_offset"]))
    nA = A["x"].size
    x = np.concatenate([A["x"], B["x"]])
    y = np.concatenate([A["y"], B["y"]])
    ev = np.concatenate([A["event"], B["event"]])
    members = {0: list(A["members"]), 1: [m + nA for m in B["members"]]}
    truth_members = {0: list(A["truth_members"]), 1: [m + nA for m in B["truth_members"]]}
    tol = et.axon_tolerances(x, y, members, event_id=ev, centres_nm={0: [0.0, 0.0], 1: [0.0, 0.0]}, seed=a)
    flags = {k: np.asarray(tol.groups[k].merged_suspect, dtype=bool) for k in (0, 1)}
    planted = {0: A["planted"], 1: B["planted"]}
    row: Dict[str, Any] = {"f": f, "rate": rate, "axon": a,
                           "n_clusters": [len(members[0]), len(members[1])],
                           "n_planted": int(planted[0].sum() + planted[1].sum()),
                           "n_planted_flagged": int(sum(int((flags[k] & planted[k]).sum()) for k in (0, 1))),
                           "n_single": int(sum(int((~planted[k]).sum()) for k in (0, 1))),
                           "n_single_flagged": int(sum(int((flags[k] & ~planted[k]).sum()) for k in (0, 1)))}
    for h in HANDLINGS:
        cents: Dict[int, NDArray[np.float64]] = {}
        usable: Dict[int, NDArray[np.bool_]] = {}
        sig: Dict[int, NDArray[np.float64]] = {}
        for k in (0, 1):
            if h == "truth":
                cw = mg.centres_with_handling(x, y, truth_members[k], np.zeros(len(truth_members[k]), dtype=bool),
                                              "middle", event=ev)
                s = mg.half_sigma_c_nm(cw.n_locs, cw.n_events, tol)
            else:
                cw = mg.centres_with_handling(x, y, members[k], flags[k], h, event=ev)
                s = np.asarray(tol.groups[k].sigma_c_nm, dtype=np.float64)[cw.owner]
                halves = cw.half > 0
                if halves.any():
                    s = s.copy()
                    s[halves] = mg.half_sigma_c_nm(cw.n_locs[halves], cw.n_events[halves], tol)
            cents[k], usable[k], sig[k] = cw.centres_nm, cw.usable, s
        for rule in RULES:
            kw: Dict[str, Any] = {}
            if rule == "per_pair":
                kw = dict(sigma_a_nm=sig[0], sigma_b_nm=sig[1])
            m = arc_eclipse_test(_arcs(cents[0]), _arcs(cents[1]), es.L_NM, TAU_NM, n_null=n_null, random_seed=a,
                                 usable_a=usable[0], usable_b=usable[1], **kw)
            row[f"{h}|{rule}"] = [int(m.n_matched), round(float(m.p_excess), 5)]
    return row


def wilson(k: int, n: int, z: float = 1.959964) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def paired(cell: Sequence[Dict[str, Any]], a: str, b: str) -> Dict[str, Any]:
    """The paired comparison of two handlings on the SAME axons (same draws): n_a_only / n_b_only axons rejected by
    one and not the other, the difference of rejection rates with a 95 % normal interval for paired proportions
    [lit: Agresti 2013, sec. 11.1] and the exact two-sided McNemar p (binomial on the discordant pairs)."""
    from scipy.stats import binomtest
    n = len(cell)
    ra = np.array([w[a][1] <= 0.05 for w in cell])
    rb = np.array([w[b][1] <= 0.05 for w in cell])
    n10, n01 = int(np.sum(ra & ~rb)), int(np.sum(~ra & rb))
    diff = (n10 - n01) / n
    se = math.sqrt(max(n10 + n01 - (n10 - n01) ** 2 / n, 0.0)) / n
    p = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {"n_a_only": n10, "n_b_only": n01, "difference": round(diff, 4),
            "ci95": [round(diff - 1.959964 * se, 4), round(diff + 1.959964 * se, 4)], "mcnemar_p": round(p, 5)}


def summarise(path: str = JSONL) -> Dict[str, Any]:
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    out: Dict[str, Any] = {"seed": SEED, "scenario": SCENARIO, "tau_nm": TAU_NM, "partner_nm": list(PARTNER_NM),
                           "origin": "[own sim] docs/sims/eclipse_tolerance/merged_handling_sim.py", "cells": {}}
    for f in COLUMN_FRACTIONS:
        for r in FUSION_RATES:
            cell = [w for w in rows if w["f"] == f and w["rate"] == r]
            if not cell:
                continue
            n = len(cell)
            d: Dict[str, Any] = {"n_axons": n, "n_null": cell[0].get("n_null")}
            npl = sum(w["n_planted"] for w in cell)
            d["planted_fusions"] = npl
            d["planted_flagged_share"] = round(sum(w["n_planted_flagged"] for w in cell) / npl, 4) if npl else None
            d["single_flagged_share"] = round(sum(w["n_single_flagged"] for w in cell)
                                              / max(1, sum(w["n_single"] for w in cell)), 4)
            for h in HANDLINGS:
                for rule in RULES:
                    k = sum(1 for w in cell if w[f"{h}|{rule}"][1] <= 0.05)
                    d[f"{h}|{rule}"] = {"rejections": k, "rate": round(k / n, 4), "wilson95": wilson(k, n),
                                        "mean_matched": round(float(np.mean([w[f"{h}|{rule}"][0] for w in cell])), 2)}
            for rule in RULES:
                for other in ("middle", "exclude"):
                    d[f"split_vs_{other}|{rule}"] = paired(cell, f"split|{rule}", f"{other}|{rule}")
            out["cells"][f"f={f}|rate={r}"] = d
    with open(JSON_OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    lines = ["| f | fusions | handling | scalar 31 nm: rate [Wilson 95 %] | per pair: rate [Wilson 95 %] |",
             "|---|---|---|---|---|"]
    for key, d in out["cells"].items():
        f, r = key.split("|")
        for h in HANDLINGS:
            a, b = d[f"{h}|scalar31"], d[f"{h}|per_pair"]
            lines.append(f"| {f[2:]} | {float(r[5:]):.0%} | {h} | {a['rate']:.3f} {a['wilson95']} | "
                         f"{b['rate']:.3f} {b['wilson95']} |")
    print("\n".join(lines))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-axons", type=int, default=400)
    ap.add_argument("--null", type=int, default=199)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        summarise()
        return 0
    k, m = (int(v) for v in a.shard.split("/"))
    done = set()
    if os.path.exists(JSONL):
        with open(JSONL, encoding="utf-8") as fh:
            done = {(w["f"], w["rate"], w["axon"]) for w in (json.loads(ln) for ln in fh if ln.strip())}
    t0 = time.time()
    j = 0
    for f in COLUMN_FRACTIONS:
        for r in FUSION_RATES:
            for ax in range(a.n_axons):
                j += 1
                if j % m != k or (f, r, ax) in done:
                    continue
                row = one_axon(f, r, ax, a.null)
                row["n_null"] = a.null
                with open(JSONL, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                if ax % 50 == 0:
                    print(f"f={f} rate={r} axon {ax}: {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
