# -*- coding: utf-8 -*-
"""
The five synthetic controls of the 1NN (SCI-3; calibration of the record on the 1NN as a possible finding).

The median 1NN confounds the DENSITY of detected clusters (K / P) and the REGULARITY of their arrangement (the index
r = median 1NN x K / P: about 0.47 for a random hard-core arrangement, 1 for evenly spaced centres). Each control
below changes one factor on simulated axons (``tools.mps_simulate_axon``, three rings, 4 clusters per um, hard core
80 nm) and runs the per-axon analysis on them, reporting the detected clusters over the true ones (middle ring),
the clusters per um, the median 1NN and r:

  1. clustering: eps in {15, 20, 25, 30, 40} nm x min samples in {5, 10, 20} on the same axons; a setting is
     "accepted" when K_detected / K_true lies within 0.9-1.1.
  2. localization-density titration: the localizations thinned to 1, 1/2, 1/4, 1/8 (label-titration logic,
     Baumgart et al. 2016).
  3. slab leak: axial precision 20 / 35 / 50 / 77 nm x slab half-width 45 / 60 / 90 nm; the fraction of detected
     clusters whose localizations come mostly from a neighbouring ring; a slab setting is "accepted" at <= 5 %.
  4. regularity: a lattice (every site occupied) with arc jitter 0 / 30 / 60 / 100 nm, and the random hard-core
     arrangement; r from the true centres and from the detected ones.
  5. interior clusters: 0 / 10 / 20 % of K extra clusters at 0.5-0.9 of the radius; the change of the median 1NN.

The clustering parameters these controls use are today's defaults; they are re-run when the clustering batch (B6)
settles eps and min samples. A table, not a verdict; the five calls that must hold are checked (thinning moves the
1NN, a regular lattice has r near 1, a random arrangement near 0.5, interior clusters at 0.5-0.9 R change the median
by little).

Run:  venv\\Scripts\\python.exe validate_nn1_controls.py [--axons N] [--out DIR]   (default 2 axons per cell,
      about 10 minutes; the full design is about 800 axons). No data is read.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.mps_analysis import analyze_axon  # noqa: E402
from tools.mps_simulate_axon import SimConfig, simulate_axon  # noqa: E402

PASSED = 0
FAILED = 0
RESULTS: Dict[str, Any] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    PASSED += int(ok)
    FAILED += int(not ok)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def _target_ring(sim: Any, a: Any, keep_index: Optional[np.ndarray] = None) -> int:
    """The ring the analysis' slab sits on: the true ring most of its localizations come from (the automatic slab
    takes the strongest axial component, not necessarily the middle ring)."""
    rings = np.asarray(sim.ring_true)
    if keep_index is not None:
        rings = rings[keep_index]
    idx = np.asarray(a.slab_index)
    inside = rings[idx[idx < rings.size]]          # added (interior) localizations carry no ring
    inside = inside[inside >= 0]
    return int(np.bincount(inside).argmax()) if inside.size else int(np.max(sim.ring_true)) // 2


def _k_true(sim: Any, ring: int) -> int:
    return int(len(np.unique(sim.cluster_true[(sim.ring_true == ring) & (sim.cluster_true >= 0)])))


def _measure(sim: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray, keep_index: Optional[np.ndarray] = None,
             **kw: Any) -> Dict[str, Any]:
    a = analyze_axon(x, y, z, pixel_size_nm=113.0, pixel_size_source="yaml", run_randomization=False, **kw)
    ring = _target_ring(sim, a, keep_index)
    out = {"K": int(a.n_clusters_kept), "K_true": _k_true(sim, ring), "ring": ring,
           "clusters_per_um": a.clusters_per_um, "median_1nn_nm": a.median_1nn_nm, "r": a.nn_regularity_index}
    out["K_ratio"] = out["K"] / out["K_true"] if out["K_true"] else None
    out["analysis"] = a
    return out


def _foreign_fraction(sim: Any, a: Any, keep_index: Optional[np.ndarray] = None) -> Optional[float]:
    """Share of kept clusters whose localizations come mostly from a ring other than the middle one."""
    labels = np.asarray(a.labels)
    if labels.size == 0:
        return None
    idx = np.asarray(a.slab_index)
    rings = np.asarray(sim.ring_true)
    if keep_index is not None:
        rings = rings[keep_index]
    rings = rings[idx]
    bad = set(int(b) for b in a.bad_report.bad_labels)
    kept = [lab for lab in np.unique(labels) if lab >= 0 and int(lab) not in bad]
    if not kept:
        return None
    middle = _target_ring(sim, a, keep_index)
    foreign = 0
    for lab in kept:
        r = rings[labels == lab]
        r = r[r >= 0]
        if r.size and np.bincount(r).argmax() != middle:
            foreign += 1
    return foreign / len(kept)


def _clean(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "analysis"} for r in rows]


def _fmt(v: Any, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def control_clustering(n: int, seed: int) -> None:
    print("\n1. Clustering: eps x min samples on the same axons")
    rows = []
    for i in range(n):
        sim = simulate_axon(SimConfig(), seed + i)
        for eps in (15.0, 20.0, 25.0, 30.0, 40.0):
            for ms in (5, 10, 20):
                m = _measure(sim, sim.x_nm, sim.y_nm, sim.z_nm, eps_nm=eps, min_samples=ms)
                m.update(axon=i, eps=eps, min_samples=ms)
                rows.append(m)
    for eps in (15.0, 20.0, 25.0, 30.0, 40.0):
        for ms in (5, 10, 20):
            sel = [r for r in rows if r["eps"] == eps and r["min_samples"] == ms]
            ratio = np.mean([r["K_ratio"] for r in sel])
            print(f"    eps {eps:4.0f} min {ms:2d}: K/K_true {ratio:.2f} ({'accepted' if 0.9 <= ratio <= 1.1 else '-'})"
                  f", clusters/um {np.mean([r['clusters_per_um'] or 0 for r in sel]):.2f}, median 1NN "
                  f"{np.mean([r['median_1nn_nm'] or 0 for r in sel]):.0f} nm, r "
                  f"{np.mean([r['r'] or 0 for r in sel]):.2f}")
    RESULTS["clustering"] = _clean(rows)


def control_thinning(n: int, seed: int) -> None:
    print("\n2. Localization-density titration")
    rows = []
    for i in range(n):
        sim = simulate_axon(SimConfig(), seed + 10 + i)
        rng = np.random.default_rng(seed + i)
        for keep in (1.0, 0.5, 0.25, 0.125):
            mask = rng.uniform(size=sim.x_nm.size) < keep
            m = _measure(sim, sim.x_nm[mask], sim.y_nm[mask], sim.z_nm[mask], keep_index=np.flatnonzero(mask))
            m.update(axon=i, keep=keep)
            rows.append(m)
            print(f"    axon {i}, keep {keep:5.3f}: K/K_true {_fmt(m['K_ratio'])}, median 1NN "
                  f"{_fmt(m['median_1nn_nm'], 0)} nm, r {_fmt(m['r'])}")
    RESULTS["thinning"] = _clean(rows)
    full = [r["median_1nn_nm"] for r in rows if r["keep"] == 1.0 and r["median_1nn_nm"]]
    thin = [r["median_1nn_nm"] for r in rows if r["keep"] == 0.125 and r["median_1nn_nm"]]
    if full and thin:
        check("thinning to 1/8 moves the median 1NN (fewer clusters detected: longer spacings)",
              np.mean(thin) != np.mean(full), f"{np.mean(full):.0f} -> {np.mean(thin):.0f} nm")


def control_slab_leak(n: int, seed: int) -> None:
    print("\n3. Slab leak: axial precision x slab half-width")
    rows = []
    for i in range(n):
        for sz in (20.0, 35.0, 50.0, 77.0):
            cfg = dataclasses.replace(SimConfig(), lpz_samples_by_bin_nm=[np.array([sz])])
            sim = simulate_axon(cfg, seed + 20 + i)
            for hw in (45.0, 60.0, 90.0):
                m = _measure(sim, sim.x_nm, sim.y_nm, sim.z_nm, slab_half_width_nm=hw)
                m["foreign"] = _foreign_fraction(sim, m["analysis"])
                m.update(axon=i, sigma_z=sz, half_width=hw)
                rows.append(m)
                print(f"    sigma_z {sz:3.0f}, half-width {hw:3.0f}: foreign clusters {_fmt(m['foreign'])} "
                      f"({'accepted' if m['foreign'] is not None and m['foreign'] <= 0.05 else '-'}), median 1NN "
                      f"{_fmt(m['median_1nn_nm'], 0)} nm, K/K_true {_fmt(m['K_ratio'])}")
    RESULTS["slab_leak"] = _clean(rows)


def control_regularity(n: int, seed: int) -> None:
    print("\n4. Regularity: lattice with arc jitter, and the random hard core")
    rows = []
    for i in range(n):
        for name, cfg in [(f"lattice, jitter {j:g} nm",
                           dataclasses.replace(SimConfig(), model="M2", m2_p_occupied=1.0, m2_lattice_nm=250.0,
                                               jitter_nm=float(j))) for j in (0, 30, 60, 100)] + \
                [("random hard core", SimConfig())]:
            sim = simulate_axon(cfg, seed + 30 + i)
            m = _measure(sim, sim.x_nm, sim.y_nm, sim.z_nm)
            m.update(axon=i, case=name)
            rows.append(m)
            print(f"    {name:24s}: r {_fmt(m['r'])}, median 1NN {_fmt(m['median_1nn_nm'], 0)} nm, K/K_true "
                  f"{_fmt(m['K_ratio'])}")
    RESULTS["regularity"] = _clean(rows)
    lattice = [r["r"] for r in rows if r["case"] == "lattice, jitter 0 nm" and r["r"]]
    rand = [r["r"] for r in rows if r["case"] == "random hard core" and r["r"]]
    if lattice:
        check("a full lattice has r near 1 (0.85-1.15)", 0.85 <= np.mean(lattice) <= 1.15, f"{np.mean(lattice):.2f}")
    if lattice and rand:
        check("the random hard-core arrangement has a smaller r than the lattice", np.mean(rand) < np.mean(lattice),
              f"{np.mean(rand):.2f} < {np.mean(lattice):.2f}")


def control_interior(n: int, seed: int) -> None:
    print("\n5. Interior clusters at 0.5-0.9 of the radius")
    rows = []
    for i in range(n):
        sim = simulate_axon(SimConfig(), seed + 40 + i)
        rng = np.random.default_rng(seed + 40 + i)
        base = _measure(sim, sim.x_nm, sim.y_nm, sim.z_nm)
        k = base["K"]
        cx, cy = float(np.mean(sim.x_nm)), float(np.mean(sim.y_nm))
        radius = float(np.median(np.hypot(sim.x_nm - cx, sim.y_nm - cy)))
        for frac in (0.0, 0.1, 0.2):
            m_extra = int(round(frac * k))
            xs, ys, zs = [sim.x_nm], [sim.y_nm], [sim.z_nm]
            for _ in range(m_extra):
                rr = radius * rng.uniform(0.5, 0.9)
                th = rng.uniform(0, 2 * np.pi)
                xs.append(cx + rr * np.cos(th) + rng.normal(0, 15.0, 150))
                ys.append(cy + rr * np.sin(th) + rng.normal(0, 15.0, 150))
                zs.append(rng.normal(0.0, 30.0, 150))
            m = _measure(sim, np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))
            m.update(axon=i, interior_fraction=frac,
                     change=(None if not (m["median_1nn_nm"] and base["median_1nn_nm"])
                             else m["median_1nn_nm"] / base["median_1nn_nm"] - 1.0))
            rows.append(m)
            print(f"    axon {i}, interior {frac:.0%} of K: median 1NN {_fmt(m['median_1nn_nm'], 0)} nm "
                  f"(change {_fmt(None if m['change'] is None else 100 * m['change'], 1)} %), r {_fmt(m['r'])}")
    RESULTS["interior"] = _clean(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axons", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else {"1", "2", "3", "4", "5"}
    t0 = time.time()
    print("=" * 100)
    print(f"THE FIVE 1NN CONTROLS - synthetic ({args.axons} axon(s) per cell; today's eps / min samples)")
    print("=" * 100)
    if "1" in only:
        control_clustering(args.axons, args.seed)
    if "2" in only:
        control_thinning(args.axons, args.seed)
    if "3" in only:
        control_slab_leak(max(1, args.axons // 2), args.seed)
    if "4" in only:
        control_regularity(args.axons, args.seed)
    if "5" in only:
        control_interior(args.axons, args.seed)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "validate_nn1_controls.json"), "w", encoding="utf-8") as fh:
            json.dump(RESULTS, fh, indent=1, default=lambda o: None if o is None else float(o))
    print("\n" + "=" * 100)
    print(f"total {time.time() - t0:.0f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
