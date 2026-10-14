# -*- coding: utf-8 -*-
"""
Does a longer next-frame window fix the large-sigma failures of the grid? (docs/nena.md, section 6.) Synthetic only.

The same files as nena_grid.py's cells "uniform 10 %" and "uniform 30 %" at sigma 20 and 30 nm, 2000 target pairs
(seeds 1000 x cell index + replicate, 10 replicates), measured with max_distance_nm = 120 (the default) and 240 nm.

Usage: python docs/sims/nena/nena_window.py [--workers 4]  ->  nena_window.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from typing import Any, Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import nena_grid as g  # noqa: E402
from nena_sim import simulate  # noqa: E402
from tools.mps_quality import nena  # noqa: E402

WINDOWS = (120.0, 240.0)


def run(task: Dict[str, Any]) -> Dict[str, Any]:
    frame, x, y, _ = simulate(task["sigma"], seed=task["seed"], **g.cell_kwargs(task["contamination"], 2000))
    out = dict(task)
    for window in WINDOWS:
        r = nena(frame, x, y, max_distance_nm=window)
        out[f"new_{window:g}"] = r.precision_nm / task["sigma"]
        out[f"old_{window:g}"] = r.lsq_nm / task["sigma"]
        out[f"flags_{window:g}"] = r.flags
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    tasks = []
    for ci, cont in enumerate(g.CONTAMINATIONS):
        if cont not in ("bg10", "bg30"):
            continue
        for si, sigma in enumerate(g.SIGMAS):
            if sigma not in (20.0, 30.0):
                continue
            ni = g.PAIRS.index(2000)
            cell = (ci * len(g.SIGMAS) + si) * len(g.PAIRS) + ni
            tasks += [dict(contamination=cont, sigma=sigma, seed=1000 * cell + rep) for rep in range(10)]
    with Pool(args.workers) as pool:
        rows = pool.map(run, tasks)
    summary = []
    for cont in ("bg10", "bg30"):
        for sigma in (20.0, 30.0):
            grp = [r for r in rows if r["contamination"] == cont and r["sigma"] == sigma]
            entry: Dict[str, Any] = dict(contamination=cont, sigma=sigma)
            for w in WINDOWS:
                for est in ("old", "new"):
                    v = np.array([r[f"{est}_{w:g}"] for r in grp])
                    entry[f"{est}_{w:g}"] = dict(median_bias=float(np.median(v) - 1), fail=float(np.mean(np.abs(v - 1) > .05)))
                entry[f"flagged_{w:g}"] = float(np.mean([bool(set(r[f'flags_{w:g}']) - {'window_short'}) for r in grp]))
            summary.append(entry)
            print(entry)
    with open(os.path.join(HERE, "nena_window.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(description=__doc__.strip().splitlines()[0], summary=summary, files=rows), fh, indent=1)


if __name__ == "__main__":
    main()
