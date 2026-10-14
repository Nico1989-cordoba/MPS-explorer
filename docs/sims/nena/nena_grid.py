# -*- coding: utf-8 -*-
"""
Validation grid of the NeNA estimator (docs/nena.md, section 5). Synthetic data only.

Grid: sigma in {6, 10, 12, 20, 30} nm x contamination {none, uniform background reaching ~10 / 30 / 50 % of the
pairs, clustered molecules at moderate / heavy crowding} x target same-molecule pairs {500, 2000, 10000, 50000} x
20 files per cell; plus linked files (none and clustered-heavy, 2000 and 10000 target pairs).

For every file: the previous estimator (Picasso's least squares on the 0.5 nm histogram, unseeded as the channel-2
code calls it, and seeded with the true sigma as Data quality would with honest lpx), the new estimate with its
95 % profile interval, flags and diagnostics, and the oracle -- sqrt(<d^2>/4) of the pairs the nearest-neighbour
rule matched to the SAME molecule, i.e. what a perfect fit could recover from those pairs. Errors are relative to
the true sigma; "fail" means more than 5 % off.

Usage:  python docs/sims/nena/nena_grid.py [--workers 11] [--reps 20] [--out nena_grid.json]
Seeds:  file seed = 1000 x cell index + replicate (fixed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from nena_sim import (PAIRS_PER_MOLECULE, background_for, next_frame_pairs,  # noqa: E402
                      rayleigh_rms_scale, simulate)
from tools.mps_quality import _picasso_lsq, nena  # noqa: E402

D_MAX = 120.0
SIGMAS = (6.0, 10.0, 12.0, 20.0, 30.0)
PAIRS = (500, 2000, 10000, 50000)
CONTAMINATIONS = ("none", "bg10", "bg30", "bg50", "clu_mod", "clu_heavy")
LINKED = ("linked_none", "linked_clu_heavy")
CLUSTER_SIZE = 10
# clustered: frames chosen so that (cluster_size - 1) x mean on-frames per molecule / frames = 0.1 or 0.4
MEAN_ON_FRAMES = 4.0 * 2.5


def cell_kwargs(contamination: str, n_target: int) -> Dict[str, Any]:
    n_mol = max(10, int(round(n_target / PAIRS_PER_MOLECULE)))
    kw: Dict[str, Any] = dict(n_mol=n_mol)
    base = contamination.replace("linked_", "")
    if base.startswith("bg"):
        kw["bg_per_frame"] = background_for(int(base[2:]) / 100.0, n_mol, max_distance_nm=D_MAX)
    elif base.startswith("clu"):
        mates_on = 0.1 if base == "clu_mod" else 0.4
        kw.update(cluster_size=CLUSTER_SIZE, frames=int(round((CLUSTER_SIZE - 1) * MEAN_ON_FRAMES / mates_on)))
    if contamination.startswith("linked_"):
        kw["linked"] = True
    return kw


def run_one(task: Dict[str, Any]) -> Dict[str, Any]:
    sigma, contamination, n_target, seed = task["sigma"], task["contamination"], task["n_target"], task["seed"]
    frame, x, y, mol = simulate(sigma, seed=seed, **cell_kwargs(contamination, n_target))
    d, same = next_frame_pairs(frame, x, y, mol, D_MAX)
    t0 = time.time()
    r = nena(frame, x, y, max_distance_nm=D_MAX)
    elapsed = time.time() - t0
    seeded = float("nan")
    if d.size >= 500:
        edges = np.arange(0.0, D_MAX + 0.5, 0.5)
        counts, edges = np.histogram(d, bins=edges)
        seeded = _picasso_lsq(0.5 * (edges[:-1] + edges[1:]), counts.astype(float), D_MAX, sigma)
    return dict(
        sigma=sigma, contamination=contamination, n_target=n_target, seed=seed, n_pairs=int(r.n_pairs),
        true_contamination=float(1.0 - same.mean()) if same.size else float("nan"),
        oracle=rayleigh_rms_scale(d[same]) if same.any() else float("nan"),
        old=float(r.lsq_nm), old_seeded=float(seeded), new=float(r.precision_nm),
        ci_low=float(r.ci_low_nm), ci_high=float(r.ci_high_nm), random_term=float(r.random_term_nm),
        w=float(r.same_molecule_fraction), crowding=float(r.crowding_fraction), converged=bool(r.converged),
        well_identified=bool(r.well_identified), flags=list(r.flags), seconds=round(elapsed, 2),
    )


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cells: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["contamination"], row["sigma"], row["n_target"]), []).append(row)
    out = []
    for (cont, sigma, n), group in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        def rel(key: str) -> np.ndarray:
            return np.array([g[key] for g in group], dtype=float) / sigma

        entry: Dict[str, Any] = dict(contamination=cont, sigma=sigma, n_target=n, files=len(group),
                                     median_pairs=float(np.median([g["n_pairs"] for g in group])),
                                     true_contamination=float(np.nanmean([g["true_contamination"] for g in group])))
        for key in ("old", "old_seeded", "new", "random_term", "oracle"):
            v = rel(key)
            ok = np.isfinite(v)
            entry[key] = dict(
                reported=int(ok.sum()),
                bias=float(np.median(v[ok]) - 1.0) if ok.any() else float("nan"),
                spread=float(np.std(v[ok])) if ok.sum() > 1 else float("nan"),
                fail=float(np.mean(np.abs(v[ok] - 1.0) > 0.05)) if ok.any() else float("nan"),
            )
        lo, hi = rel("ci_low"), rel("ci_high")
        ok = np.isfinite(lo) & np.isfinite(hi)
        entry["ci_covers_sigma"] = float(np.mean((lo[ok] <= 1.0) & (hi[ok] >= 1.0))) if ok.any() else float("nan")
        orc = rel("oracle")
        ok2 = ok & np.isfinite(orc)
        entry["ci_covers_oracle"] = (float(np.mean((lo[ok2] <= orc[ok2]) & (hi[ok2] >= orc[ok2])))
                                     if ok2.any() else float("nan"))
        entry["flagged"] = float(np.mean([bool(set(g["flags"]) - {"window_short"}) for g in group]))
        entry["not_well_identified"] = float(np.mean([not g["well_identified"] for g in group]))
        new = rel("new")
        bad = np.abs(new - 1.0) > 0.05
        flagged = np.array([bool(set(g["flags"]) - {"window_short"}) for g in group])
        entry["new_fail_unflagged"] = float(np.mean(bad & ~flagged & np.isfinite(new)))
        entry["flag_counts"] = {f: int(sum(f in g["flags"] for g in group))
                                for f in sorted({f for g in group for f in g["flags"]})}
        entry["median_seconds"] = float(np.median([g["seconds"] for g in group]))
        out.append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "nena_grid.json"))
    ap.add_argument("--quick", action="store_true", help="sigma 12 only, 2000 pairs, 5 files per cell")
    args = ap.parse_args()
    tasks = []
    contaminations = CONTAMINATIONS + LINKED
    for ci, cont in enumerate(contaminations):
        for si, sigma in enumerate(SIGMAS):
            for ni, n in enumerate(PAIRS):
                if cont.startswith("linked_") and n not in (2000, 10000):
                    continue
                if args.quick and (sigma != 12.0 or n != 2000):
                    continue
                cell_index = (ci * len(SIGMAS) + si) * len(PAIRS) + ni
                for rep in range(args.reps if not args.quick else 5):
                    tasks.append(dict(sigma=sigma, contamination=cont, n_target=n, seed=1000 * cell_index + rep))
    # big files first so the pool ends evenly
    tasks.sort(key=lambda t: -t["n_target"])
    # Resumable: every finished file is appended to <out>.partial.jsonl; a re-run skips those.
    partial = args.out + ".partial.jsonl"
    rows = []
    if os.path.exists(partial):
        with open(partial, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    done = {(r["contamination"], r["sigma"], r["n_target"], r["seed"]) for r in rows}
    tasks = [t for t in tasks if (t["contamination"], t["sigma"], t["n_target"], t["seed"]) not in done]
    print(f"{len(rows)} files already done, {len(tasks)} to run", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool, open(partial, "a", encoding="utf-8") as sink:
        for i, row in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tasks)} files, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["contamination"], r["sigma"], r["n_target"], r["seed"]))
    result = dict(
        description=__doc__.strip().splitlines()[0], d_max_nm=D_MAX, sigmas_nm=SIGMAS, pairs=PAIRS,
        contaminations=contaminations, reps=args.reps, seconds_total=round(time.time() - t0, 1),
        numpy=np.__version__, summary=summarize(rows), files=rows,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"done: {len(rows)} files in {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
