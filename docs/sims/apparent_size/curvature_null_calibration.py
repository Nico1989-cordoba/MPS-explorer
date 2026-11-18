# -*- coding: utf-8 -*-
"""
Decision E5a: is the pipeline-null p of the size-vs-curvature statistic (question b,
``tools.size_curvature_null``) calibrated under a TRUE null? SYNTHETIC DATA ONLY (R8).

Each "observed" axon is drawn from ``tools.mps_pipeline_null.synthetic_model()`` (ellipse 2000 x 1400 nm -- kappa R
varies along it, every stretch convex --, molecules at random on the membrane with a 50 nm contact hard core, two
labels per molecule, c_label 14, 2 loc/cycle, NeNA 12 nm, three rings): the localizations per cluster are
independent of the curvature by construction, so the null of question (b) is TRUE. The axon is clustered with
DBSCAN 25 / 10 [lit: Gazal et al. 2026] and ``run_pipeline_null`` re-reads everything from it and tests the
registered readouts on B replicates -- the SAME replicates for both size statistics (decision G3: one null per
axon; question (a)'s ``size_nloc_nn_spearman`` is recorded too, as a reference).

B: with the two-sided rank p 2 min(lo, hi) of ``tools.mps_randomization.monte_carlo_p``, B = 39 makes
p <= 0.05 exactly the event "the observed value is the most extreme of the 40 on one side", whose probability
under exchangeability is 2 / 40 = 0.05: the nominal level is attained exactly [derivation]. A larger B (99) only
attains 0.04 (p takes the values 0.04, 0.06, ...), so B = 39 tests the level more sharply and costs 2.5 x less.

Run (two processes, resumable):
    py -3 curvature_null_calibration.py --n-axons 300 --B 39 --shard 0/2
    py -3 curvature_null_calibration.py --n-axons 300 --B 39 --shard 1/2
    py -3 curvature_null_calibration.py --summarise
Per-axon rows: curvature_null_calibration.jsonl (not committed); summary: curvature_null_calibration.json.
~32 s per axon with B = 39 (0.68 s per replicate) [own sim: this machine, one process].

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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools import mps_pipeline_null as pn  # noqa: E402
from tools import size_curvature_null as scn  # noqa: E402
from tools import size_plugins as sp  # noqa: E402

BASE_SEED = 20261029
READOUTS = [scn.NULL_STATISTIC, "size_nloc_nn_spearman"]
JSONL = os.path.join(HERE, "curvature_null_calibration.jsonl")
JSON_OUT = os.path.join(HERE, "curvature_null_calibration.json")


def one_axon(i: int, n_rep: int) -> Dict[str, Any]:
    seed = BASE_SEED + i
    truth = pn.synthetic_model()
    spec = pn.PipelineSpec.for_method("DBSCAN")
    t0 = time.perf_counter()
    an, _ax = pn.observe_synthetic(truth, seed, spec)
    row: Dict[str, Any] = {"axon": i, "seed": seed, "k_obs": int(an.n_clusters_kept), "B": n_rep}
    try:
        res = pn.run_pipeline_null(an, spec=spec, nena_nm=12.0, n_replicates=n_rep, seed=seed, readouts=READOUTS)
    except pn.NullRefused as exc:
        row["refused"] = str(exc)
        return row
    row.update({"p": {k: res.p_values.get(k) for k in READOUTS}, "observed": {k: res.observed.get(k) for k in READOUTS},
                "null_median": {k: res.null_medians.get(k) for k in READOUTS},
                "n_null_values": {k: int(sum(1 for r in res.replicate_readouts if r.get(k) is not None))
                                  for k in READOUTS},
                "n_accepted": res.n_accepted, "interpretable": res.interpretable,
                "seconds_axon": round(time.perf_counter() - t0, 1)})
    return row


def wilson(k: int, n: int, z: float = 1.959964) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def summarise() -> Dict[str, Any]:
    rows = [json.loads(ln) for ln in open(JSONL, encoding="utf-8") if ln.strip()]
    rows.sort(key=lambda r: r["axon"])
    done = [r for r in rows if "refused" not in r]
    out: Dict[str, Any] = {"origin": "[own sim] docs/sims/apparent_size/curvature_null_calibration.py",
                           "seed": BASE_SEED, "B": sorted({r["B"] for r in done}), "n_axons": len(done),
                           "n_refused": len(rows) - len(done), "readouts": {}}
    for k in READOUTS:
        ps = [r["p"][k] for r in done if r["p"].get(k) is not None]
        rej = int(sum(1 for p in ps if p <= 0.05))
        n = len(ps)
        # under the exact null the rank of the observed value among B + 1 is uniform: a check of the whole p law
        hist = np.histogram(ps, bins=[0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0001])[0].tolist() if ps else []
        out["readouts"][k] = {"tested": n, "rejections_at_0.05": rej, "rate": round(rej / n, 4) if n else None,
                              "wilson95": wilson(rej, n), "five_percent_inside_wilson": bool(n) and
                              wilson(rej, n)[0] <= 0.05 <= wilson(rej, n)[1],
                              "p_histogram_bins_0_.05_.2_.4_.6_.8_1": hist,
                              "mean_observed": round(float(np.mean([r["observed"][k] for r in done
                                                                    if r["observed"].get(k) is not None])), 4)}
    out["seconds_axon_median"] = float(np.median([r["seconds_axon"] for r in done])) if done else None
    out["k_obs_median"] = float(np.median([r["k_obs"] for r in done])) if done else None
    with open(JSON_OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-axons", type=int, default=300)
    ap.add_argument("--B", type=int, default=39)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    sp.register_with_pipeline_null(replace=True)
    if a.summarise:
        summarise()
        return 0
    k, m = (int(v) for v in a.shard.split("/"))
    done = set()
    if os.path.exists(JSONL):
        with open(JSONL, encoding="utf-8") as fh:
            done = {json.loads(ln)["axon"] for ln in fh if ln.strip()}
    for i in range(a.n_axons):
        if i % m != k or i in done:
            continue
        row = one_axon(i, a.B)
        with open(JSONL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        print(f"axon {i}: k={row.get('k_obs')} p={row.get('p')} {row.get('seconds_axon')} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
