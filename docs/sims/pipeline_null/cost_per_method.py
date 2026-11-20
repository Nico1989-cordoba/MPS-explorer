# -*- coding: utf-8 -*-
"""
Seconds per replicate of the per-axon pipeline null, per clustering method, on ONE synthetic axon (rule R8: no
real data). For each method the synthetic "observed" axon is analysed with that method, then
``tools.mps_pipeline_null.measure_cost`` times ``--probes`` replicates of its null (generate + the full
analyze_axon with that method + Ripley curves), with allow_slow=True so the slow methods are measured too.

Axon: ``synthetic_model()`` defaults (ellipse 2000 x 1400 nm, perimeter ~10.8 um, 4 molecules per um, contact
50 nm, NeNA 12 nm, c_label 14) and, with ``--dense``, c_label 56 (4x the localizations per molecule, a slab of
roughly 4x the localizations).

Run (one process; Emitters forced to 1 worker):
    py -3 docs/sims/pipeline_null/cost_per_method.py [--dense] [--probes 2] [--methods DBSCAN,HDBSCAN,...]
Writes cost_per_method[_dense].json next to this script.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools import mps_pipeline_null as pn  # noqa: E402

DEFAULT_METHODS = ["DBSCAN", "HDBSCAN", "HDBSCAN (scikit-learn)", "FOCAL", "ToMATo", "GraphCommunity",
                   "voronoi_factor", "OPTICS", "voronoi_csr", "emitters_bagol_like"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--probes", type=int, default=2)
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", default=None, help="output JSON name (default cost_per_method[_dense].json)")
    a = ap.parse_args()
    model = pn.synthetic_model(c_label=56.0 if a.dense else pn.DEFAULT_C_LABEL)
    rows = []
    for method in [m.strip() for m in a.methods.split(",") if m.strip()]:
        # Emitters: one worker (the machine rule) and the buffer the layer requires per file (F5c); MEA is the
        # buffer of c_label 14 [lit: Dempsey et al. 2011]
        params = {"processes": 1, "buffer": "MEA"} if method == "emitters_bagol_like" else {}
        spec = pn.PipelineSpec.for_method(method, params)
        t0 = time.perf_counter()
        try:
            an, ax = pn.observe_synthetic(model, a.seed, spec)
            obs_s = time.perf_counter() - t0
            cost = pn.measure_cost(an, spec=spec, n_probe=a.probes, seed=a.seed, nena_nm=12.0, allow_slow=True)
            est = pn.estimated_seconds_per_replicate(method, int(an.n_locs_slab))
            row = {"method": method, "slow": pn.method_is_slow(method), "n_locs_total": ax.n_locs,
                   "n_locs_slab_observed": int(an.n_locs_slab), "k_observed": int(an.n_clusters_kept),
                   "observed_analysis_s": round(obs_s, 2),
                   "seconds_per_replicate": round(cost["seconds_per_replicate"], 3),
                   "null_n_locs_slab": round(cost["n_locs_slab"], 0),
                   "table_estimate_s": round(est, 3),
                   "minutes_for_199": round(cost["seconds_per_replicate"] * 199 / 60.0, 1),
                   "minutes_for_999": round(cost["seconds_per_replicate"] * 999 / 60.0, 1)}
        except Exception as exc:  # noqa: BLE001
            row = {"method": method, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(row), flush=True)
        rows.append(row)
    out = {"what": "seconds per replicate of the pipeline null, one synthetic axon, one process",
           "origin": "simulation (synthetic only)", "dense": bool(a.dense), "probes": a.probes, "seed": a.seed,
           "machine": platform.processor() or platform.machine(), "python": platform.python_version(),
           "date": time.strftime("%Y-%m-%d"), "rows": rows}
    path = os.path.join(HERE, a.out or ("cost_per_method_dense.json" if a.dense else "cost_per_method.json"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
