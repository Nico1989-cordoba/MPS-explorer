# -*- coding: utf-8 -*-
"""
Own simulation behind docs/dbscan_rule.md (clustering plan A2, A3, A4; decisions D3 and D5): synthetic data only.

Part 1 - localizations per cycle on isolated emitters. 600 emitters spread over a 20 um field (no two within reach
of each other), geometric switching cycles of mean 14 [lit: Dempsey et al. 2011, Alexa Fluor 647 in MEA], each cycle
a run of consecutive frames of geometric length with mean 1, 1.5, 2, 3 or 5 frames, one localization per frame with
N(0, 12 nm) per axis, a 20,000-frame movie. ``tools.mps_events.locs_per_cycle`` with NeNA 12 nm, at 3 NeNA (decision
D5, what the research simulated) and 3 sqrt(2) NeNA (the research's recommendation): bias of the measured value,
and of the 3 NeNA value corrected for the expected split of events (``split_corrected``).

Part 2 - linking on the clustering benchmark's generator (``tools.cluster_benchmark.generate``, the model of the
clustering research: molecules on an axon contour, two epitopes, 1-3 frames per cycle, a log-normal spread of the
per-localization precision, background) with its true event ids: the share of true events split, the share of linked
events that join two true events, and the localizations per cycle, at both radii. Background localizations (no true
event) are left out.

Part 3 - the rule against the compatibility mode on the same generator: DBSCAN 25/10 and DBSCAN with the rule
(eps = min(2.5 x 12, 25) nm; min_samples from c_label 14 and the localizations per cycle measured on the sample at
3 NeNA), molecule-level metrics of ``tools.cluster_benchmark.molecule_metrics`` (missed molecules, merged and false
clusters), 5 replicates per scenario, and the "probable fusion" mark (events / 14 > 2) against the truth "the cluster
holds two or more molecules".

Run (one process, about a minute):
    py -3 docs/sims/events/events_rule_sim.py --out docs/sims/events/events_rule_sim.json

Seeds: part 1 seed 11; parts 2-3 ``tools.cluster_benchmark.scenario_seed(name) + replicate``.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sklearn.cluster import DBSCAN  # noqa: E402

from tools import cluster_benchmark as cb  # noqa: E402
from tools import dbscan_rule as rule  # noqa: E402
from tools import mps_events as ev  # noqa: E402

SIGMA_NM = 12.0                       # [lit: ~NeNA of Gazal et al. 2026]
N_FRAMES = 20000                      # [ad hoc] as the research's simulation
RADII = (3.0, 3.0 * math.sqrt(2.0))   # D5 (a) as simulated; (b) as recommended
LENGTH_MEANS = (1.0, 1.5, 2.0, 3.0, 5.0)
SCENARIOS = ("sparse_lab1", "medium_lab1", "sparse_lab2", "medium_lab2")
REPLICATES = 5


def isolated_movie(seed: int, length_mean: float) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 20000.0, (600, 2))
    n_cyc = rng.geometric(1.0 / 14.0, 600)
    cyc_emitter = np.repeat(np.arange(600), n_cyc)
    m = cyc_emitter.size
    start = rng.integers(1, N_FRAMES - 60, m)
    length = rng.geometric(1.0 / length_mean, m)
    loc_cyc = np.repeat(np.arange(m), length)
    frame = start[loc_cyc] + (np.arange(loc_cyc.size) - np.repeat(np.cumsum(length) - length, length))
    xy = pos[cyc_emitter[loc_cyc]] + rng.normal(size=(loc_cyc.size, 2)) * SIGMA_NM
    order = np.argsort(frame, kind="stable")
    return {"frame": frame[order], "xy": xy[order], "truth": float(length.mean())}


def part1() -> List[Dict[str, Any]]:
    rows = []
    for lm in LENGTH_MEANS:
        mv = isolated_movie(11, lm)
        row: Dict[str, Any] = {"length_mean_frames": lm, "truth": round(mv["truth"], 4)}
        for k in RADII:
            r = ev.locs_per_cycle(mv["frame"], mv["xy"][:, 0], mv["xy"][:, 1], nena_nm=SIGMA_NM, radius_in_nena=k,
                                  n_frames=N_FRAMES)
            tag = f"k{k:.2f}"
            row[f"measured_{tag}"] = round(r.value, 4)
            row[f"bias_pct_{tag}"] = round(100.0 * (r.value / mv["truth"] - 1.0), 2)
            if k == RADII[0] and r.split_corrected is not None:
                row["corrected_k3"] = round(r.split_corrected, 4)
                row["corrected_bias_pct_k3"] = round(100.0 * (r.split_corrected / mv["truth"] - 1.0), 2)
        rows.append(row)
    return rows


def part2_3() -> List[Dict[str, Any]]:
    rows = []
    for name in SCENARIOS:
        sc = cb.PRESETS[name]
        for rep in range(REPLICATES):
            s = cb.generate(sc, np.random.default_rng([cb.scenario_seed(name), rep]))
            real = s.event_id >= 0
            row: Dict[str, Any] = {"scenario": name, "replicate": rep, "n_locs": int(s.xy.shape[0])}
            eid = s.event_id[real]
            row["lpc_truth"] = round(float(np.bincount(np.unique(eid, return_inverse=True)[1]).mean()), 4)
            lpc3 = None
            for k in RADII:
                layer = ev.link_events(s.frame[real], s.xy[real, 0], s.xy[real, 1], nena_nm=SIGMA_NM,
                                       radius_in_nena=k, n_frames=cb.N_FRAMES)
                pairs = np.unique(np.column_stack([layer.link_group, eid]), axis=0)
                tag = f"k{k:.2f}"
                row[f"split_pct_{tag}"] = round(100.0 * float(np.mean(np.unique(pairs[:, 1], return_counts=True)[1]
                                                                      > 1)), 2)
                row[f"joined_pct_{tag}"] = round(100.0 * float(np.mean(np.bincount(pairs[:, 0]) > 1)), 3)
                # the program measures on the whole file, background included
                lpc = ev.locs_per_cycle(s.frame, s.xy[:, 0], s.xy[:, 1], nena_nm=SIGMA_NM, radius_in_nena=k,
                                        n_frames=cb.N_FRAMES)
                row[f"lpc_{tag}"] = round(lpc.value, 4)
                if k == RADII[0]:
                    lpc3 = lpc
            r = rule.derive_rule(SIGMA_NM, c_label=14.0, locs_per_cycle=lpc3)
            row["rule_eps_nm"], row["rule_min_samples"] = r.eps_nm, r.min_samples
            for mode, eps, ms in (("compat", 25.0, 10), ("rule", r.eps_nm, r.min_samples)):
                labels = DBSCAN(eps=eps, min_samples=int(ms)).fit_predict(s.xy).astype(np.int64)
                m = cb.molecule_metrics(labels, s)
                for key in ("missed", "merge", "false", "exact"):
                    row[f"{mode}_{key}_pct"] = round(100.0 * float(m.get(key, float("nan"))), 2)
                if mode == "compat":
                    ce = ev.cluster_events(s.frame, s.xy[:, 0], s.xy[:, 1], labels, nena_nm=SIGMA_NM)
                    # merged as the benchmark counts it: >= 2 molecules each with >= PRESENT_SHARE of ITS
                    # localizations in the cluster, and the cluster not mostly background
                    mol_total = np.bincount(s.molecule_id[s.molecule_id >= 0])
                    fused = np.zeros(ce.cluster_ids.size, dtype=bool)
                    for i, c in enumerate(ce.cluster_ids):
                        inside = labels == c
                        mol = s.molecule_id[inside & (s.molecule_id >= 0)]
                        if mol.size <= 0.5 * np.count_nonzero(inside):
                            continue
                        counts = np.bincount(mol, minlength=mol_total.size)
                        fused[i] = int(np.sum(counts >= cb.PRESENT_SHARE * np.maximum(mol_total, 1))) >= 2
                    _, flag = ev.fusion_flags(ce.n_events, 14.0)
                    row["fusion_true_n"] = int(fused.sum())
                    row["fusion_flag_n"] = int(flag.sum())
                    row["fusion_recall_pct"] = (round(100.0 * float(flag[fused].mean()), 1)
                                                if fused.any() else None)
                    row["fusion_precision_pct"] = (round(100.0 * float(fused[flag].mean()), 1)
                                                   if flag.any() else None)
            rows.append(row)
    return rows


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name in SCENARIOS:
        sub = [r for r in rows if r["scenario"] == name]
        keys = [k for k in sub[0] if k not in ("scenario", "replicate")]
        out[name] = {k: round(float(np.nanmean([np.nan if r[k] is None else float(r[k]) for r in sub])), 3)
                     for k in keys}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "events_rule_sim.json"))
    args = ap.parse_args()
    t0 = time.time()
    p1 = part1()
    p23 = part2_3()
    result = {"script": "docs/sims/events/events_rule_sim.py", "sigma_nm": SIGMA_NM, "radii_in_nena": list(RADII),
              "part1_isolated": p1, "part2_3_summary": summarise(p23), "part2_3_rows": p23,
              "seconds": round(time.time() - t0, 1)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(json.dumps({"part1": p1, "summary": result["part2_3_summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
