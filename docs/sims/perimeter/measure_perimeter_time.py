"""
Timings of the perimeter's 2-opt on SYNTHETIC cluster centres (R8: no real data).

What it measures: ``tools.mps_geometry.reconstruct_perimeter`` from every start
(``all_starts=True``, what the MPS analysis runs) and from one start
(``all_starts=False``, the program before 2026-09-19), wall-clock seconds, one
process. The power laws of ``tools/mps_perimeter_cost.py`` are fitted to its
output (``--fit``).

Synthetic axon: an ellipse of semi-axes 3000 x 2200 nm (about 16.4 um round),
n cluster centres at uniform random angles, each moved radially by a normal
scatter of ``--scatter`` nm. The same axon with more clusters is what a smaller
xi (Emitters) or a large axon (DBSCAN) produces; the scatter sets how often the
polar-angle order crosses itself, and so how much work 2-opt has.

Usage (from the repository root, one thread):
    set OMP_NUM_THREADS=1
    py -3 docs/sims/perimeter/measure_perimeter_time.py --sizes 100,300,600,1000 \
        --scatter 25 --out docs/sims/perimeter/perimeter_time.jsonl
    py -3 docs/sims/perimeter/measure_perimeter_time.py --fit docs/sims/perimeter/perimeter_time.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEMI_AXES_NM = (3000.0, 2200.0)


def axon(n: int, seed: int, scatter_nm: float) -> np.ndarray:
    """n synthetic cluster centres around the ellipse, in nm."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 2 * np.pi, n)
    r = rng.normal(0.0, scatter_nm, n)
    a, b = SEMI_AXES_NM
    return np.column_stack([(a + r) * np.cos(t), (b + r) * np.sin(t)])


def measure(sizes: List[int], seed: int, scatter_nm: float, modes: List[str], out: str) -> None:
    from tools.mps_geometry import reconstruct_perimeter
    for n in sizes:
        pts = axon(n, seed, scatter_nm)
        for mode in modes:
            t0 = time.perf_counter()
            res = reconstruct_perimeter(pts, all_starts=(mode == "every"))
            row = dict(n=n, seed=seed, scatter_nm=scatter_nm, mode=mode,
                       seconds=round(time.perf_counter() - t0, 3),
                       perimeter_um=round(res.perimeter_um, 4), n_starts=res.n_starts,
                       improvements=res.n_2opt_improvements)
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(row, flush=True)


def fit(path: str) -> Dict[str, Tuple[float, float]]:
    """log-log least squares, per mode, through the slowest timing at each n."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    result: Dict[str, Tuple[float, float]] = {}
    for mode in ("every", "one"):
        slowest: Dict[int, float] = {}
        for r in rows:
            if r["mode"] == mode and r["n"] >= 100:
                slowest[r["n"]] = max(slowest.get(r["n"], 0.0), float(r["seconds"]))
        if len(slowest) < 2:
            continue
        n = np.array(sorted(slowest), dtype=float)
        s = np.array([slowest[int(k)] for k in n])
        b, log_a = np.polyfit(np.log(n), np.log(s), 1)
        result[mode] = (float(np.exp(log_a)), float(b))
        pred = np.exp(log_a) * n ** b
        print(f"{mode}: seconds = {np.exp(log_a):.4g} * n ** {b:.4f}")
        for k, obs, p in zip(n, s, pred):
            print(f"   n={int(k):5d}  slowest {obs:9.2f} s   fit {p:9.2f} s   ratio {obs / p:5.2f}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="100,300,600,1000")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scatter", type=float, default=25.0)
    ap.add_argument("--modes", default="every,one")
    ap.add_argument("--out", default=os.path.join(HERE, "perimeter_time.jsonl"))
    ap.add_argument("--fit", default="")
    args = ap.parse_args()
    if args.fit:
        fit(args.fit)
        return 0
    measure([int(s) for s in args.sizes.split(",")], args.seed, args.scatter,
            args.modes.split(","), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
