# -*- coding: utf-8 -*-
"""
Calibration of the per-axon pipeline null (A9 sec. 6, tests 3-5) on SYNTHETIC axons only (rule R8).

Scenarios (each "observed" axon is drawn from ``tools.mps_pipeline_null.synthetic_model()``: ellipse 2000 x 1400 nm,
perimeter ~10.8 um, NeNA 12 nm, sigma_z 40 nm, period 190 nm, background 2 per um^2, c_label 14, 2 loc/cycle,
two labels per molecule, linkage 12 nm, epitope disc 25 nm, three rings):

  null     molecules at random with a contact hard core of 50 nm, 4 per um: the null is TRUE (regularity plan
           T0). The fraction of axons with p <= 0.05 is the rejection rate under the null (A9 test 3: global 1D
           rejections <= 7 of 60, P(fail | level 5 %) = 0.03), plus the rate of the POINTWISE envelope (the
           observed curve leaving the 2.5-97.5 % band at some r above the fusion mark), which A9 asked to measure
           before writing anything about it.
  pairs    molecules in pairs 150 nm apart (pair centres by sequential inhibition at 200 nm): power, and r* within
           +/-30 nm of 150 nm (A9 test 4).
  lattice  molecules on a 190 nm lattice with N(0, 30 nm) jitter (A9 test 4).

Each axon: ``run_pipeline_null`` with B replicates (closure on the slab's localizations, the observed method),
the null re-reading everything from the observed analysis (membrane, hard core, loc/cycle, background, axial
width), so the rate measures the whole procedure, estimation included (the p are approximate: Dao & Genton 2014).

Sharding keeps the machine rule (<= 2 Python processes): run two shards in two processes, then summarise:
    py -3 calibration_null.py --scenario null --n-axons 60 --B 99 --shard 0/2
    py -3 calibration_null.py --scenario null --n-axons 60 --B 99 --shard 1/2
    py -3 calibration_null.py --summarise null
Each shard appends one JSON line per axon to calibration_<scenario>[_<method>].jsonl (a re-run skips the axons
already there: resumable). The summary writes calibration_<scenario>[_<method>].json (the committed table) and
prints a Markdown table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools import mps_pipeline_null as pn  # noqa: E402

P_KEYS = ["ripley1d_p_global", "ripley2d_p_global", "p_mc_k_per_um", "p_mc_nn_median", "p_mc_reff_median",
          "p_mc_occupancy", "p_mc_regularity"]
BASE_SEED = 20261022
PLANTED_SCALE = {"pairs": 150.0, "lattice": 190.0}


def _stem(scenario: str, method: str, closure: str = "n_locs", hard_core: Optional[float] = None) -> str:
    return (f"calibration_{scenario}" + ("" if method == "DBSCAN" else "_" + method.replace(" ", "_"))
            + ("" if closure == "n_locs" else "_closure_" + closure)
            + ("" if hard_core is None else f"_hardcore{hard_core:g}"))


def _pointwise_exceeds(env: Optional[Any], r_min: float) -> Optional[bool]:
    if env is None or env.observed is None:
        return None
    test = env.r >= r_min
    obs = env.observed[test]
    lo, hi = env.pointwise_lo[test], env.pointwise_hi[test]
    ok = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    return bool(np.any((obs[ok] < lo[ok]) | (obs[ok] > hi[ok])))


def one_axon(i: int, scenario: str, method: str, n_rep: int, closure: str = "n_locs",
             hard_core: Optional[float] = None) -> Dict[str, Any]:
    seed = BASE_SEED + i
    truth = pn.synthetic_model()
    spec = pn.PipelineSpec.for_method(method)
    arcs = None
    if scenario in PLANTED_SCALE:
        rng = np.random.default_rng([seed, 77])
        length = pn.smooth_closed_path(truth.membrane_nm).length_nm
        n_mol = int(round(truth.n_molecules_per_ring))
        arcs = [pn.planted_arcs(scenario, length, n_mol, rng) for _ in range(truth.n_rings)]
    t0 = time.perf_counter()
    an, _ax = pn.observe_synthetic(truth, seed, spec, molecule_arcs=arcs)
    row: Dict[str, Any] = {"axon": i, "seed": seed, "scenario": scenario, "method": method,
                           "k_obs": int(an.n_clusters_kept), "n_locs_slab": int(an.n_locs_slab)}
    try:
        res = pn.run_pipeline_null(an, spec=spec, nena_nm=12.0, n_replicates=n_rep, seed=seed, closure=closure,
                                   hard_core_nm=hard_core)
    except pn.NullRefused as exc:
        row["refused"] = str(exc)
        return row
    d = res.export_dict()
    row.update({k: d.get(k) for k in P_KEYS})
    row.update({"ripley1d_rstar_nm": d.get("ripley1d_rstar_nm"), "ripley2d_rstar_nm": d.get("ripley2d_rstar_nm"),
                "ripley1d_hmax": d.get("ripley1d_hmax"), "ripley2d_hmax": d.get("ripley2d_hmax"),
                "pointwise1d_exceeds": _pointwise_exceeds(res.h1, res.scale.fusion_nm),
                "pointwise2d_exceeds": _pointwise_exceeds(res.h2, res.scale.fusion_nm),
                "interpretable": res.interpretable, "reasons": res.not_interpretable_reasons(),
                "closure_ratio": res.closure_ratio, "n_accepted": res.n_accepted,
                "fidelity": [[c.name, c.observed, c.null_median, c.met] for c in res.fidelity],
                "model": {k: (round(float(getattr(res.model, k)), 3)) for k in (
                    "n_molecules_per_ring", "hard_core_nm", "radial_sd_nm", "sigma_z_nm", "locs_per_cycle",
                    "background_per_um2", "period_nm")},
                "observed_values": res.observed, "null_medians": res.null_medians,
                "seconds_per_replicate": round(res.seconds_per_replicate, 3),
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


def summarise(scenario: str, method: str, out_dir: str = HERE, closure: str = "n_locs",
              hard_core: Optional[float] = None) -> Dict[str, Any]:
    stem = _stem(scenario, method, closure, hard_core)
    rows = []
    with open(os.path.join(out_dir, stem + ".jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                rows.append(json.loads(ln))
    rows.sort(key=lambda r: r["axon"])
    done = [r for r in rows if "refused" not in r]
    n = len(done)
    rej = {k: int(sum(1 for r in done if r.get(k) is not None and r[k] <= 0.05)) for k in P_KEYS}
    tested = {k: int(sum(1 for r in done if r.get(k) is not None)) for k in P_KEYS}
    pw = {k: int(sum(1 for r in done if r.get(k))) for k in ("pointwise1d_exceeds", "pointwise2d_exceeds")}
    out: Dict[str, Any] = {
        "scenario": scenario, "method": method, "closure": closure, "hard_core_nm": hard_core, "origin": "simulation (synthetic only), docs/sims/pipeline_null/"
        "calibration_null.py", "B": done[0]["n_accepted"] if done else None, "n_axons": n,
        "n_refused": len(rows) - n, "rejections_at_0.05": rej, "tested": tested,
        "rejection_rate": {k: (round(rej[k] / tested[k], 4) if tested[k] else None) for k in P_KEYS},
        "wilson95": {k: wilson(rej[k], tested[k]) for k in P_KEYS},
        "pointwise_exceedances": pw,
        "pointwise_rate": {k: round(v / n, 4) if n else None for k, v in pw.items()},
        "pointwise_wilson95": {k: wilson(v, n) for k, v in pw.items()},
        "interpretable": int(sum(1 for r in done if r.get("interpretable"))),
        "closure_ratio_median": round(float(np.median([r["closure_ratio"] for r in done])), 4) if done else None,
        "closure_ratio_range": [round(float(np.min([r["closure_ratio"] for r in done])), 4),
                                round(float(np.max([r["closure_ratio"] for r in done])), 4)] if done else None,
        "seconds_per_replicate_median": round(float(np.median([r["seconds_per_replicate"] for r in done])), 3)
        if done else None,
        "k_obs_median": float(np.median([r["k_obs"] for r in done])) if done else None,
        "not_interpretable_reasons": sorted({s.split(":")[0][:80] for r in done for s in r.get("reasons", [])}),
    }
    if scenario in PLANTED_SCALE:
        target = PLANTED_SCALE[scenario]
        for tag in ("ripley1d", "ripley2d"):
            hit = [r for r in done if r.get(f"{tag}_p_global") is not None and r[f"{tag}_p_global"] <= 0.05]
            near = [r for r in hit if r.get(f"{tag}_rstar_nm") is not None
                    and abs(r[f"{tag}_rstar_nm"] - target) <= 30.0]
            out[f"{tag}_rstar_within_30nm_of_{int(target)}"] = f"{len(near)} of {len(hit)} rejections"
            out[f"{tag}_rstar_values"] = [r.get(f"{tag}_rstar_nm") for r in hit]
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    lines = [f"| statistic | rejections at 0.05 | rate | Wilson 95 % |", "|---|---|---|---|"]
    for k in P_KEYS:
        lines.append(f"| {k} | {rej[k]} / {tested[k]} | {out['rejection_rate'][k]} | {out['wilson95'][k]} |")
    for k, v in pw.items():
        lines.append(f"| {k} (pointwise band) | {v} / {n} | {out['pointwise_rate'][k]} | {out['pointwise_wilson95'][k]} |")
    print("\n".join(lines))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["null", "pairs", "lattice"], default="null")
    ap.add_argument("--method", default="DBSCAN")
    ap.add_argument("--n-axons", type=int, default=60)
    ap.add_argument("--B", type=int, default=99)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summarise", choices=["null", "pairs", "lattice"], default=None)
    ap.add_argument("--out-dir", default=HERE, help="where the .jsonl / .json go (default: next to this script)")
    ap.add_argument("--closure", choices=["n_locs", "k_clusters"], default="n_locs")
    ap.add_argument("--hard-core", type=float, default=None, help="null hard core in nm (default: read, 2 r_eff)")
    a = ap.parse_args()
    if a.summarise:
        summarise(a.summarise, a.method, a.out_dir, a.closure, a.hard_core)
        return 0
    k, m = (int(v) for v in a.shard.split("/"))
    path = os.path.join(a.out_dir, _stem(a.scenario, a.method, a.closure, a.hard_core) + ".jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            done = {json.loads(ln)["axon"] for ln in fh if ln.strip()}
    for i in range(a.n_axons):
        if i % m != k or i in done:
            continue
        row = one_axon(i, a.scenario, a.method, a.B, a.closure, a.hard_core)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        print(f"axon {i}: k={row.get('k_obs')} p1d={row.get('ripley1d_p_global')} p2d={row.get('ripley2d_p_global')} "
              f"{row.get('seconds_axon')} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
