# -*- coding: utf-8 -*-
"""
Centre-level calibration of the gap measures and of the two-stage reading (``tools.mps_gap_measures``), with the
module's own code. SYNTHETIC DATA ONLY (rule R8): arc positions on a circle, no localizations, no clustering.

Part 1 -- the catalogue and its identities (the regularity research's sim1 reproduced with the repository code):
K = 20 / 50 / 100, mean spacing a = 250 nm, 400 patterns per cell. Checks: Poisson CV^2 = (K - 1)/(K + 1)
[Moran 1947]; jittered lattice rho1 = -1/2; renewal (hard core, M8) rho1 ~ 0; Bragg height of a lattice
K e^-q + 1 - e^-q, q = (2 pi sigma / a)^2; S_low ~ CV^2 for renewal processes; arc r of Poisson = ln2/2.

Part 2 -- the two-stage rule at the CENTRES level (the regularity audit's matched_exclusion_null.py, seed 1011,
re-done with this code): K = 50, a = 250 nm; stage 1 = gap CV two-sided against sequential inhibition with the
contact hard core 50 nm (B = 199; a centre-level proxy of the pipeline null), stage 2 = the matched-exclusion null
(``matched_null_centres``, B = 199); 150 patterns per truth. The pipeline-level version (the acceptance test of PR 2)
is ``pipeline_confusion.py``.

Run:  py -3 docs/sims/gap_measures/centres_calibration.py      (about 2 minutes, one process)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from tools import mps_gap_measures as gm  # noqa: E402

SEED = 1                     # part 1 (as research sim1)
SEED2 = 1011                 # part 2 (as the audit's matched-null check)
A_NM = 250.0
N_REP1 = 400
N_REP2 = 150
B = 199
D_CONTACT = 50.0
OUT = os.path.join(HERE, "centres_calibration.json")

CATALOGUE = [("poisson", {}), ("hard_core", {"d_nm": 80.0}), ("hard_core", {"d_nm": 120.0}),
             ("hard_core", {"d_nm": 160.0}), ("m8_gamma", {"sigma_rel": 0.25}),
             ("lattice", {"jitter_nm": 30.0}), ("lattice", {"jitter_nm": 60.0}),
             ("lattice", {"jitter_nm": 30.0, "occupancy": 0.85})]
TRUTHS = [("contact_d50", "hard_core", {"d_nm": 50.0}), ("excl_d120", "hard_core", {"d_nm": 120.0}),
          ("excl_d150", "hard_core", {"d_nm": 150.0}), ("excl_d170", "hard_core", {"d_nm": 170.0}),
          ("lattice_s30", "lattice", {"jitter_nm": 30.0}), ("lattice_s60", "lattice", {"jitter_nm": 60.0}),
          ("lattice_s30_vac15", "lattice", {"jitter_nm": 30.0, "occupancy": 0.85}),
          ("lattice_s30_vac30", "lattice", {"jitter_nm": 30.0, "occupancy": 0.70}),
          ("m8_gamma_0.25", "m8_gamma", {"sigma_rel": 0.25}), ("poisson", "poisson", {})]


def wilson(k: int, n: int, z: float = 1.959964) -> List[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def part1() -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    cells = {}
    for k in (20, 50, 100):
        length = k * A_NM
        for kind, kw in CATALOGUE:
            rows = [gm.gap_measures(gm.planted_arrangement(kind, rng, k, length, **kw), length) for _ in range(N_REP1)]
            cell = {}
            for f in ("cv", "rho1", "excess_cv", "bragg_s", "s_low", "r_arc"):
                v = np.array([getattr(r, f) for r in rows if getattr(r, f) is not None], dtype=float)
                cell[f] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
            cv2 = np.array([r.cv ** 2 for r in rows if r.cv is not None])
            cell["cv2_mean"] = float(np.mean(cv2))
            cells[f"{kind}{kw}|K={k}"] = cell
    checks = {}
    for k in (20, 50, 100):
        c = cells[f"poisson{{}}|K={k}"]
        checks[f"poisson_cv2_K{k}"] = {"sim": c["cv2_mean"], "theory": (k - 1) / (k + 1)}
        checks[f"poisson_r_arc_K{k}"] = {"sim": c["r_arc"]["mean"], "theory": math.log(2) / 2}
        lat = cells[f"lattice{{'jitter_nm': 30.0}}|K={k}"]
        q = (2 * math.pi * 30.0 / A_NM) ** 2
        checks[f"lattice_rho1_K{k}"] = {"sim": lat["rho1"]["mean"], "theory": -0.5}
        checks[f"lattice_bragg_K{k}"] = {"sim": lat["bragg_s"]["mean"], "theory": k * math.exp(-q) + 1 - math.exp(-q)}
        for name in ("hard_core{'d_nm': 120.0}", "m8_gamma{'sigma_rel': 0.25}"):
            cc = cells[f"{name}|K={k}"]
            checks[f"renewal_rho1_{name}_K{k}"] = {"sim": cc["rho1"]["mean"], "theory": 0.0}
            checks[f"renewal_slow_{name}_K{k}"] = {"sim": cc["s_low"]["mean"], "theory_cv2": cc["cv2_mean"]}
    return {"cells": cells, "checks": checks}


def part2() -> Dict[str, Any]:
    k = 50
    length = k * A_NM
    cells = []
    for ti, (name, kind, kw) in enumerate(TRUTHS):
        rng = np.random.default_rng([SEED2, ti])
        labels: Dict[str, int] = {lv: 0 for lv in gm.READING_LEVELS}
        dms: List[float] = []
        rej = {"cv_contact": 0, "bragg_contact": 0, "bragg_matched": 0, "rho1_matched": 0}
        n_stage2 = 0
        means: Dict[str, List[float]] = {"cv": [], "rho1": [], "excess_cv": [], "bragg_s": []}
        for i in range(N_REP2):
            s = gm.planted_arrangement(kind, rng, k, length, **kw)
            obs = gm.gap_measures(s, length)
            for f in means:
                v = getattr(obs, f)
                if v is not None:
                    means[f].append(float(v))
            pts, jam = gm.hard_core_arcs(np.random.default_rng([SEED2, ti, i, 1]), k, length, D_CONTACT, B)
            nb = gm._batch_measures(pts[~jam], length)
            cv1 = gm.summarise("gap_cv", obs.cv, list(nb["cv"]), "two-sided")
            b1 = gm.summarise("bragg_s", obs.bragg_s, list(nb["bragg_s"]), "upper")
            rej["cv_contact"] += int((cv1.p or 1) <= 0.05)
            rej["bragg_contact"] += int((b1.p or 1) <= 0.05)
            mt: Optional[gm.MatchedNullResult] = None
            if cv1.p is not None and cv1.p <= 0.05 and obs.cv is not None and cv1.mean is not None and obs.cv < cv1.mean:
                mt = gm.matched_null_centres(s, length, n_replicates=B, seed=SEED2 * 1000 + 100 * ti + i)
                n_stage2 += 1
                dms.append(mt.d_m_nm)
                rej["bragg_matched"] += int((mt.bragg_s.p or 1) <= 0.05)
                rej["rho1_matched"] += int((mt.rho1.p or 1) <= 0.05)
            rd = gm.regularity_reading(k=k, cv=cv1, rho1_matched=None if mt is None else mt.rho1,
                                       bragg_matched=None if mt is None else mt.bragg_s, matched=mt,
                                       excess_cv=obs.excess_cv, bragg_m=obs.bragg_m,
                                       bragg_spacing_nm=obs.bragg_spacing_nm)
            labels[rd.level] += 1
        cells.append({
            "truth": name, "kind": kind, "params": kw, "n": N_REP2, "n_stage2": n_stage2,
            "mean": {f: float(np.mean(v)) for f, v in means.items()},
            "d_m_mean": float(np.mean(dms)) if dms else None,
            "reject_rate": {k2: v / N_REP2 for k2, v in rej.items()},
            "reject_rate_given_stage2": {k2: rej[k2] / n_stage2 for k2 in ("bragg_matched", "rho1_matched")}
            if n_stage2 else None,
            "labels": labels,
            "label_rates": {lv: [c / N_REP2] + wilson(c, N_REP2) for lv, c in labels.items() if c},
        })
        print(f"{name:20s} stage2 {n_stage2:3d} d_m {cells[-1]['d_m_mean']} labels "
              f"{ {lv: c for lv, c in labels.items() if c} }", flush=True)
    return {"K": k, "a_nm": A_NM, "B": B, "d_contact_nm": D_CONTACT, "cells": cells}


def main() -> None:
    t0 = time.perf_counter()
    out = {"script": os.path.basename(__file__), "seed_part1": SEED, "seed_part2": SEED2, "n_rep_part1": N_REP1,
           "n_rep_part2": N_REP2, "part1": part1()}
    for name, c in out["part1"]["checks"].items():
        print(name, {k: round(v, 3) for k, v in c.items()})
    out["part2"] = part2()
    out["seconds"] = time.perf_counter() - t0
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT, f"{out['seconds']:.0f} s")


if __name__ == "__main__":
    main()
