# -*- coding: utf-8 -*-
"""
Decision E1 (2026-10-23): does the footprint pool's elongation cut (``POOL_EXCLUDE_ELONGATION`` = 2.5, calibrated
with c_label = 14 events per label in ``size_probes.py``) still land within 1-2 nm of the true footprint when the
dye blinks more or fewer times per label? SYNTHETIC DATA ONLY (R8).

c_label values: 8 and 20 (the trend), 14 (the MEA preset the cut was calibrated with) and 26 [lit: Dempsey et al.
2011, Alexa Fluor 647 in a beta-mercaptoethanol (betaME) buffer: 26 is the value for ANOTHER imaging buffer, not a
sensitivity value of the same buffer]. In every cell the analysis is told the true c_label (the user picks the
buffer's preset); one extra cell analyses a c_label 26 file with the 14 preset (the wrong buffer chosen).

The generator is ``tools.mps_size_synth`` (the same as ``size_probes.py``): one ring per axon, circle R 1400 nm,
K = 60 clusters with an 80 nm hard core, footprint rho (disc on the membrane), zero-truncated Poisson labels of mean
2 or 3, linkage 10 nm, precision 10 nm (25 % spread), 2 localizations per event. Each ring is clustered with DBSCAN
eps 25 nm / min_samples 10 [lit: Gazal et al. 2026] and with the true clusters ("oracle"); the pool
(``tools.mps_apparent_size.pooled_footprint``) is computed with no cut, the 1.8 flag and the 2.5 cut.

Run from the repository root:  py -3 docs/sims/apparent_size/elongation_cut_clabel.py [--n-axons 200]
Writes elongation_cut_clabel.json beside this file (committed). ~5 min, one process.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from tools import mps_apparent_size as asz  # noqa: E402
from tools import mps_size_synth as syn  # noqa: E402

SEED = 20261029
C_LABELS: Tuple[float, ...] = (8.0, 14.0, 20.0, 26.0)
# (labels per cluster, true footprint rho nm)
CASES: Tuple[Tuple[float, float], ...] = ((2.0, 25.0), (3.0, 25.0), (3.0, 0.0))
CUTS: Tuple[Optional[float], ...] = (None, 1.8, 2.0, 2.25, asz.POOL_EXCLUDE_ELONGATION, 2.75, 3.0, 3.5)
# the cuts besides 1.8 / 2.5 only serve to find, per c_label, the cut that would remove the bias (the proposal of
# docs/apparent_size.md sec. 12; NOT a default)
MISMATCH = (26.0, 14.0, 3.0, 25.0)        # (true c_label, c_label told to the analysis, labels, rho)


def _dbscan(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    return np.asarray(DBSCAN(eps=25.0, min_samples=10).fit(np.column_stack([x, y])).labels_, dtype=np.int64)


def _r(v: Any, nd: int = 2) -> Any:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), nd)


def cell(rng: np.random.Generator, c_true: float, c_told: float, labels: float, rho: float, n_axons: int
         ) -> Dict[str, Any]:
    acc: Dict[str, List[float]] = {}
    excluded: Dict[str, List[float]] = {}
    verdicts: List[str] = []
    for _ in range(n_axons):
        c = syn.make_contour("circle", 1400.0, rng)
        ring = syn.generate_ring(rng, c, labels_mean=labels, rho=rho, K=60, c_label=c_true)
        for det, lab in (("oracle", ring.truth), ("dbscan25", _dbscan(ring.x, ring.y))):
            cs = asz.cluster_sizes(ring.x, ring.y, lab, event_id=ring.event, curve=c.xy, rarefy_reps=1)
            for cut in CUTS:
                ex = None if cut is None else np.where(np.isfinite(cs.elongation), cs.elongation > cut, False)
                p = asz.pooled_footprint(cs.sd_tangential, cs.n_locs.astype(np.float64), cs.n_events, syn.SIGMA_LOC_NM, exclude=ex,
                                         c_label=c_told)
                key = f"{det}, cut {cut}"
                if det == "dbscan25" and cut == asz.POOL_EXCLUDE_ELONGATION:
                    verdicts.append(p.verdict)
                if ex is not None:
                    excluded.setdefault(key, []).append(float(np.mean(ex)) if ex.size else 0.0)
                if p.rho_nm is not None:
                    acc.setdefault(key, []).append(float(p.rho_nm))
    out: Dict[str, Any] = {"n_axons": n_axons, "c_label_true": c_true, "c_label_analysis": c_told,
                           "labels_mean": labels, "rho_true_nm": rho, "pool": {}}
    for k, v in acc.items():
        a = np.asarray(v)
        se = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else float("nan")
        out["pool"][k] = {"n": int(a.size), "mean_nm": _r(a.mean()), "sd_nm": _r(a.std(ddof=1) if a.size > 1 else None),
                          "se_nm": _r(se), "bias_vs_true_nm": _r(a.mean() - rho),
                          "share_excluded": _r(np.mean(excluded[k]), 3) if k in excluded else None}
    o = out["pool"].get(f"oracle, cut {asz.POOL_EXCLUDE_ELONGATION}")
    d = out["pool"].get(f"dbscan25, cut {asz.POOL_EXCLUDE_ELONGATION}")
    if o and d:
        out["dbscan_cut_minus_oracle_cut_nm"] = _r(d["mean_nm"] - o["mean_nm"])
    out["verdicts_dbscan_cut"] = {v: verdicts.count(v) for v in sorted(set(verdicts))}
    best = [(abs(v["bias_vs_true_nm"]), float(k.split("cut ")[1])) for k, v in out["pool"].items()
            if k.startswith("dbscan25, cut ") and not k.endswith("None")]
    out["dbscan_cut_with_least_bias"] = min(best)[1] if best else None
    return out


def best_cut_per_c_label(cells: Dict[str, Any]) -> Dict[str, Any]:
    """Per c_label, the cut whose mean |bias| over the rho 25 nm cells (labels 2 and 3) is smallest -- the evidence
    for a cut as a function of c_label (a PROPOSAL, not a default)."""
    out: Dict[str, Any] = {}
    for c_label in C_LABELS:
        rows = [c for c in cells.values() if c["c_label_true"] == c_label == c["c_label_analysis"]
                and c["rho_true_nm"] == 25.0]
        score: Dict[float, float] = {}
        for cut in CUTS:
            if cut is None or cut == 1.8:
                continue
            b = [abs(r["pool"][f"dbscan25, cut {cut}"]["bias_vs_true_nm"]) for r in rows
                 if f"dbscan25, cut {cut}" in r["pool"]]
            if b:
                score[float(cut)] = float(np.mean(b))
        if score:
            k = min(score, key=lambda v: score[v])
            out[f"{c_label:g}"] = {"best_cut": k, "mean_abs_bias_nm": _r(score[k]),
                                   "mean_abs_bias_at_2.5_nm": _r(score.get(2.5))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-axons", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    res: Dict[str, Any] = {"seed": SEED, "cut": asz.POOL_EXCLUDE_ELONGATION,
                           "origin": "[own sim] docs/sims/apparent_size/elongation_cut_clabel.py", "cells": {}}
    for c_label in C_LABELS:
        for labels, rho in CASES:
            key = f"c_label {c_label:g}, labels {labels:g}, rho {rho:g} nm"
            res["cells"][key] = cell(rng, c_label, c_label, labels, rho, a.n_axons)
            print(key, json.dumps(res["cells"][key]["pool"]), f"{time.time() - t0:.0f} s", flush=True)
    ct, cl, labels, rho = MISMATCH
    key = f"c_label {ct:g} analysed as {cl:g}, labels {labels:g}, rho {rho:g} nm"
    res["cells"][key] = cell(rng, ct, cl, labels, rho, a.n_axons)
    print(key, json.dumps(res["cells"][key]["pool"]), flush=True)
    res["best_cut_per_c_label"] = best_cut_per_c_label(res["cells"])
    print(json.dumps(res["best_cut_per_c_label"]), flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(HERE, "elongation_cut_clabel.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
