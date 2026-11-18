# -*- coding: utf-8 -*-
"""
The two-stage regularity reading THROUGH THE PIPELINE (the acceptance test of regularity PR 2): synthetic axons with
a known arrangement of molecules go through the localization simulator of the pipeline null
(``tools.mps_pipeline_null.observe_synthetic``: labels, blinking, precision, background, two neighbouring rings), the
reference clustering (DBSCAN 25/10), curation and contour; then ``tools.mps_gap_measures.analyze_gap_regularity``
with stage 1 against the pipeline null and stage 2 against the matched-exclusion null, both at the pipeline level
(exact) and at the centres level (approx.). SYNTHETIC DATA ONLY (rule R8).

Truths (per ring, three independent rings; ellipse 2.0 x 1.4 um semi-axes, membrane ~10.8 um):
  contact     random molecules with the contact hard core 50 nm (the null is true: the LEVEL)
  excl150     random with hard core 150 nm (a strong exclusion; the audit's blocking case)
  lattice     full lattice, spacing ~250 nm, jitter 30 nm
  vacancies   lattice with 15 % empty sites (spacing ~211 nm), jitter 30 nm
  m8          iid Gamma gaps, sigma/a = 0.25 (model M8, P-R92: generator only, no label)
Molecules per ring: 4 per um [lit order: Gazal et al. 2026] -> 43 per ring.

Plan acceptance (regularity plan, PR 2): excl150 labelled lattice <= 5 % and periodic_component <= 15 %; lattice
labelled lattice >= 80 %; vacancies periodic_component + lattice >= 90 %; contact random_with_contact >= 85 %.

Costs (this laptop, one process): ~2 min per axon for stage 1 (B = 199), ~100 s more when stage 2 runs at the
pipeline level (5 steps x 16 + 49). Run in shards (one process each), resumable (``*.jsonl`` checkpoint):

    py -3 docs/sims/gap_measures/pipeline_confusion.py --shard 0 --nshards 2
    py -3 docs/sims/gap_measures/pipeline_confusion.py --shard 1 --nshards 2
    py -3 docs/sims/gap_measures/pipeline_confusion.py --summarise
    (add --roi to every call for the run with a drawn ROI: pipeline_confusion_roi.*)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from tools import mps_gap_measures as gm  # noqa: E402
from tools import mps_pipeline_null as pn  # noqa: E402
from tools.mps_matching import smooth_closed_path  # noqa: E402

SEED = 2810                                     # [ad hoc] fixed
N_AXONS = 12                                    # per truth
B_STAGE1 = 199                                  # the default (P-R78); two-sided p_min = 2/200 = 0.01
# (a first run at B = 49 was stopped after 3 axons: two-sided p_min 0.04 leaves no room, 1 of 49 null CVs below
#  the observed already gives p = 0.08; that run's rows were discarded)
B_MATCHED = 49
PIPE_STEPS, PIPE_N = 5, 16
SEMI_AXES = (2000.0, 1400.0)
MOLECULES_PER_UM = 4.0
TRUTHS: List[Dict[str, Any]] = [
    {"name": "contact", "kind": None, "kw": {}},
    {"name": "excl150", "kind": "hard_core", "kw": {"d_nm": 150.0}},
    {"name": "lattice", "kind": "lattice", "kw": {"jitter_nm": 30.0}},
    {"name": "vacancies", "kind": "lattice", "kw": {"jitter_nm": 30.0, "occupancy": 0.85}},
    {"name": "m8", "kind": "m8_gamma", "kw": {"sigma_rel": 0.25}},
]
OUT_JSONL = os.path.join(HERE, "pipeline_confusion.jsonl")
OUT_JSON = os.path.join(HERE, "pipeline_confusion.json")
# --roi: the observed axon (and therefore its null) is curated against a drawn ROI = the membrane's box + 500 nm,
# as a user draws it; without it the curation's edge reference is the slab's own outline and removes runs of
# clusters on the outer rim (see the summary).
ROI_MODE = False


def out_paths() -> tuple:
    if ROI_MODE:
        return (os.path.join(HERE, "pipeline_confusion_roi.jsonl"), os.path.join(HERE, "pipeline_confusion_roi.json"))
    return OUT_JSONL, OUT_JSON


def wilson(k: int, n: int, z: float = 1.959964) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def one_axon(ti: int, i: int) -> Dict[str, Any]:
    t = TRUTHS[ti]
    model = pn.synthetic_model(SEMI_AXES, molecules_per_um=MOLECULES_PER_UM)
    length = float(smooth_closed_path(np.asarray(model.membrane_nm)).length_nm)
    n = int(round(MOLECULES_PER_UM * length / 1000.0))
    obs_seed = SEED + 1000 * ti + i
    arcs = None
    if t["kind"] is not None:
        rng = np.random.default_rng([SEED, ti, i])
        arcs = [gm.planted_arrangement(t["kind"], rng, n, length, **t["kw"]) for _ in range(model.n_rings)]
    t0 = time.perf_counter()
    spec = None
    if ROI_MODE:
        from tools.cluster_quality import SquareROI
        r = np.asarray(model.region_nm)
        spec = pn.PipelineSpec(roi=SquareROI(float(r[:, 0].min()), float(r[:, 1].min()), float(r[:, 0].max()),
                                             float(r[:, 1].max())))
    an, _ax = pn.observe_synthetic(model, obs_seed, spec, molecule_arcs=arcs)
    row: Dict[str, Any] = {"truth": t["name"], "axon": i, "seed": obs_seed, "k": int(an.n_clusters_kept)}
    try:
        res = gm.analyze_gap_regularity(an, n_replicates=B_STAGE1, n_matched=B_MATCHED, stage2="both",
                                        seed=obs_seed, nena_nm=12.0,
                                        matched_kw={"steps": PIPE_STEPS, "n_bisect": PIPE_N})
    except pn.NullRefused as exc:
        row.update({"refused": str(exc), "seconds": time.perf_counter() - t0})
        return row
    gm_ = res.measures
    s1 = res.stage1
    row.update({
        "cv": gm_.cv, "rho1": gm_.rho1, "excess_cv": gm_.excess_cv, "bragg_s": gm_.bragg_s,
        "bragg_spacing_nm": gm_.bragg_spacing_nm, "d_contact_nm": res.d_contact_nm, "min_nn_nm": res.min_nn_nm,
        "p_cv": s1["gap_cv"].p, "cv_null_mean": s1["gap_cv"].mean, "cv_null_sd": s1["gap_cv"].sd,
        "p_rho1_contact": gm.summarise("r", gm_.rho1, [r.get("gap_rho1") for r in res.null_result.replicate_readouts],
                                       "lower").p,
        "p_bragg_contact": s1["bragg_s"].p, "p_excess_cv": s1["gap_excess_cv"].p,
        "null_interpretable": bool(res.null_result.interpretable),
        "closure_ratio": res.null_result.closure_ratio, "k_null_mean": float(np.mean(res.null_result.k_null)),
        "n_elongated": res.elongation.n_elongated if res.elongation else None,
        "elongation_expected": res.elongation.expected_under_null if res.elongation else None,
        "reading_pipeline": res.reading.level,
        "n_edge_removed": len(getattr(an.bad_report, "edge_touching", ()) or ()),
    })
    for tag, mt in (("centres", res.matched_centres), ("pipeline", res.matched_pipeline)):
        if mt is None:
            continue
        rd = gm.regularity_reading(k=res.k, cv=s1["gap_cv"], rho1_matched=mt.rho1, bragg_matched=mt.bragg_s,
                                   matched=mt, excess_cv=gm_.excess_cv, bragg_m=gm_.bragg_m,
                                   bragg_spacing_nm=gm_.bragg_spacing_nm)
        row.update({f"d_m_{tag}": mt.d_m_nm, f"capped_{tag}": mt.capped, f"p_rho1_{tag}": mt.rho1.p,
                    f"p_bragg_{tag}": mt.bragg_s.p, f"reading_{tag}": rd.level,
                    f"cv_null_at_dm_{tag}": mt.cv_null_at_d_m, f"notes_{tag}": mt.notes})
    if res.matched_centres is None:
        row["reading_centres"] = res.reading.level
    row["seconds"] = time.perf_counter() - t0
    return row


def done_keys() -> set:
    keys = set()
    if os.path.exists(out_paths()[0]):
        with open(out_paths()[0], encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                keys.add((r["truth"], r["axon"]))
    return keys


def run_shard(shard: int, nshards: int) -> None:
    done = done_keys()
    jobs = [(ti, i) for i in range(N_AXONS) for ti in range(len(TRUTHS))]
    for j, (ti, i) in enumerate(jobs):
        if j % nshards != shard or (TRUTHS[ti]["name"], i) in done:
            continue
        try:
            row = one_axon(ti, i)
        except RuntimeError as exc:          # recorded with its reason, never silently dropped
            row = {"truth": TRUTHS[ti]["name"], "axon": i, "k": -1, "refused": f"RuntimeError: {exc}", "seconds": 0.0}
        with open(out_paths()[0], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        print(f"{row['truth']:10s} {row['axon']:2d} K {row['k']:3d} -> {row.get('reading_pipeline', row.get('refused'))} / "
              f"{row.get('reading_centres')} ({row['seconds']:.0f} s)", flush=True)


def summarise() -> None:
    rows = []
    with open(out_paths()[0], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    cells = []
    for t in TRUTHS:
        rr = [r for r in rows if r["truth"] == t["name"] and "refused" not in r]
        n = len(rr)
        cell: Dict[str, Any] = {"truth": t["name"], "n": n,
                                "n_refused": sum(1 for r in rows if r["truth"] == t["name"] and "refused" in r)}
        for tag in ("pipeline", "centres"):
            counts = {lv: sum(1 for r in rr if r.get(f"reading_{tag}") == lv) for lv in gm.READING_LEVELS}
            cell[f"labels_{tag}"] = counts
            cell[f"rates_{tag}"] = {lv: [c / n if n else None] + wilson(c, n) for lv, c in counts.items() if c}
        cell["stage1_reject_cv"] = sum(1 for r in rr if (r.get("p_cv") or 1) <= 0.05) / max(n, 1)
        cell["bragg_vs_contact_reject"] = sum(1 for r in rr if (r.get("p_bragg_contact") or 1) <= 0.05) / max(n, 1)
        for key in ("n_edge_removed", "k", "cv", "rho1", "excess_cv", "bragg_s", "d_m_pipeline", "d_m_centres", "d_contact_nm",
                    "k_null_mean", "closure_ratio", "seconds"):
            v = [r[key] for r in rr if r.get(key) is not None]
            cell[f"mean_{key}"] = float(np.mean(v)) if v else None
        cell["n_capped_pipeline"] = sum(1 for r in rr if r.get("capped_pipeline"))
        cell["n_null_not_interpretable"] = sum(1 for r in rr if not r.get("null_interpretable", True))
        cells.append(cell)
    out = {"script": os.path.basename(__file__), "seed": SEED, "n_axons_per_truth": N_AXONS, "B_stage1": B_STAGE1,
           "B_matched": B_MATCHED, "pipeline_bisection": [PIPE_STEPS, PIPE_N], "semi_axes_nm": SEMI_AXES,
           "molecules_per_um": MOLECULES_PER_UM, "method": "DBSCAN 25/10", "roi": "membrane box + 500 nm" if ROI_MODE
           else "none (edge reference = the slab's outline)", "cells": cells}
    with open(out_paths()[1], "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    for c in cells:
        print(c["truth"], c["n"], "pipeline:", {k: v for k, v in c["labels_pipeline"].items() if v},
              "centres:", {k: v for k, v in c["labels_centres"].items() if v})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--roi", action="store_true")
    a = ap.parse_args()
    global ROI_MODE
    ROI_MODE = bool(a.roi)
    if a.summarise:
        summarise()
    else:
        run_shard(a.shard, a.nshards)


if __name__ == "__main__":
    main()
