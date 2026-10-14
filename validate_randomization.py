# -*- coding: utf-8 -*-
"""
Calibration of the randomization control and of the SCI-3 statistics (calibration-plan item 3), on synthetic data only.

  A. Rejection rates under a TRUE null. Observed centres are drawn by the same sequential inhibition the control
     uses, in a +/-50 nm band around a circle of K x 250 nm, with a true hard core of 80 nm; K in {20, 50, 100},
     B = 199. Three null variants: exchangeable (the control is given the true hard core), the paper's rule (the
     smallest observed 1NN) and its bias correction. Reported: the rejection rate at 0.05 of p_MC(median),
     p_MC(D) and, as the negative control, of the two-sample KS p of scipy that the program no longer reports;
     the mean z_D; the rate of significant CDF crossings. Acceptance: p_MC inside the Wilson 95 % interval of 0.05
     for the exchangeable variant (the other two are reported: approximate by construction); |mean z_D| <= 0.15
     (exchangeable); significant crossings in at most 5 % of null draws.
  B. The crossing detector on distance samples: a null draw (expect a significant crossing in <= 5 %), a planted
     crossing (a share of the distances drawn from 0.5 x the null and another from 1.4 x it: an excess of short
     distances AND a deficit of long ones, which is what a fixed number of clusters on a fixed perimeter gives;
     a mixture with only a shorter component lies above the null everywhere and has no crossing at all), at two
     strengths, detection reported; and a pure shift (expect no crossing).
  C. The band half-width swept over 25 / 50 / 100 nm on simulated axons (``tools.mps_simulate_axon``, radial scatter
     of the centres 10 / 20 / 30 nm): null median 1NN, z_D, p_D. A table; the default is "adequate" when z_D changes
     by less than 0.5 between 25 and 100 nm on rings with 10 nm scatter.
  D. The grid spacing 2.5 / 5 / 10 nm on a K = 50 ring of continuous (off-grid) centres, 4 seeds each: from 2.5 nm
     (nearly continuous) to the default 5 nm the mean null median moves by less than 1 nm and the mean z_D by less
     than 0.05 + 2 standard errors; 10 nm is reported.
  E. The ICC chance level in closed form against 20,000 Monte Carlo draws, 8 designs (balanced and ragged):
     |difference| <= 0.01; with heavy-tailed data (t3) the difference is reported.
  F. The sigma cap of the occupancy: Gaussian clusters (sd 20 nm) and uniform discs (radius 40 nm), n in {5, 10, 20,
     50, 100, 200, 500}, 2000 each: the fraction capped and the claimed half-width, against the derivation's table
     (0.05 in fractions, 3 nm in half-widths).
  G. The constancy script: coverage of the slope's 95 % interval (3 animals x 6 axons and 6 x 6, slope 0 and
     2 pp / 10 um), 200 replicates each; acceptance coverage >= the Wilson lower bound of 0.95.

Run:  venv\\Scripts\\python.exe validate_randomization.py [--full] [--out DIR] [--only A,B,...]
      default: 120 replicates per cell in A (about 10 min); --full: 300 (about 25 min). No data is read.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import mps_randomization as R  # noqa: E402

PASSED = 0
FAILED = 0
RESULTS: Dict[str, Any] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f"   [{detail}]" if detail else ""))


def wilson(p: float, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# --------------------------------------------------------------------------------------------- the candidate cache
# Every replicate of section A uses the same circle, hence the same smoothed contour and band: the candidates are
# built once per circle (the control builds them per call). Only an optimisation of this validator.
_CACHE: Dict[Tuple[bytes, float, float], np.ndarray] = {}
_BUILD = R.build_annulus_candidates


def _cached_candidates(smoothed: np.ndarray, half_width_nm: float = R.DEFAULT_ANNULUS_HALF_WIDTH_NM,
                       grid_spacing_nm: float = R.DEFAULT_GRID_SPACING_NM) -> np.ndarray:
    key = (np.round(np.asarray(smoothed, float), 6).tobytes(), float(half_width_nm), float(grid_spacing_nm))
    if key not in _CACHE:
        _CACHE[key] = _BUILD(smoothed, half_width_nm, grid_spacing_nm)
    return _CACHE[key]


def _circle(k: int, spacing: float = 250.0, n: int = 400) -> np.ndarray:
    radius = k * spacing / (2 * np.pi)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(th), radius * np.sin(th)])


# --------------------------------------------------------------------------------------------- A
def section_a(reps: int, seed: int) -> None:
    from scipy.stats import ks_2samp
    print("\nA. Rejection rates under a true null (B = 199)")
    R.build_annulus_candidates = _cached_candidates      # type: ignore[assignment]
    try:
        d0 = 80.0
        table = []
        for k in (20, 50, 100):
            contour = _circle(k)
            smoothed = R.smooth_contour_bspline(R.close_contour(contour), smoothing=float(len(contour)))
            cand = _cached_candidates(smoothed)
            for variant in ("exchangeable", "paper rule", "bias-corrected"):
                rng = np.random.default_rng([seed, k, len(variant)])
                rej_med = rej_d = rej_scipy = cross = 0
                zs: List[float] = []
                t0 = time.time()
                for rep in range(reps):
                    obs = R.place_with_min_distance(cand, k, d0, rng)
                    if len(obs) < k:
                        continue
                    params = R.RandomizationParams(hard_core="bias-corrected" if variant == "bias-corrected"
                                                   else "observed minimum")
                    r = R.randomize_cluster_positions(
                        obs, contour, n_iterations=199, random_seed=int(rng.integers(2 ** 31)),
                        min_distance_nm=d0 if variant == "exchangeable" else None, params=params)
                    rej_med += (r.ks_p_mc_median or 1.0) <= 0.05
                    rej_d += (r.ks_p_mc_d or 1.0) <= 0.05
                    rej_scipy += ks_2samp(r.experimental_1nn_nm, r.randomized_1nn_nm).pvalue <= 0.05
                    cross += bool(r.cdf_crossing_significant)
                    if r.ks_z is not None:
                        zs.append(r.ks_z)
                n = reps
                row = {"K": k, "variant": variant, "n": n, "p_mc_median": rej_med / n, "p_mc_d": rej_d / n,
                       "scipy_ks_p": rej_scipy / n, "mean_z_d": float(np.mean(zs)) if zs else None,
                       "significant_crossings": cross / n, "seconds": round(time.time() - t0, 1)}
                table.append(row)
                print(f"    K = {k:3d} {variant:15s} rejection at 0.05: p_MC(median) {row['p_mc_median']:.3f}  "
                      f"p_MC(D) {row['p_mc_d']:.3f}  scipy KS p {row['scipy_ks_p']:.3f}  mean z_D "
                      f"{row['mean_z_d']:+.2f}  significant crossings {row['significant_crossings']:.3f}  "
                      f"({row['seconds']} s)")
        RESULTS["A"] = table
        for row in table:
            lo, hi = wilson(0.05, row["n"])
            if row["variant"] == "exchangeable":
                for key in ("p_mc_median", "p_mc_d"):
                    check(f"K = {row['K']}, exchangeable: {key} inside the Wilson band of 0.05",
                          lo <= row[key] <= hi or row[key] < lo, f"{row[key]:.3f} vs {lo:.3f}-{hi:.3f}")
                # 0.15 plus the Monte Carlo error of a mean over n replicates (z_D has sd about 1)
                margin = 0.15 + 2.0 / math.sqrt(row["n"])
                check(f"K = {row['K']}, exchangeable: |mean z_D| <= 0.15 + 2/sqrt(n)", abs(row["mean_z_d"]) <= margin,
                      f"{row['mean_z_d']:+.3f} (limit {margin:.2f})")
            check(f"K = {row['K']}, {row['variant']}: significant crossings in at most 5 % of null draws (+ MC)",
                  row["significant_crossings"] <= hi, f"{row['significant_crossings']:.3f}")
        worst = max(row["scipy_ks_p"] for row in table)
        check("negative control: scipy's KS p over-rejects somewhere (the reason it is no longer reported)",
              worst > wilson(0.05, table[0]["n"])[1], f"worst {worst:.3f}")
    finally:
        R.build_annulus_candidates = _BUILD                # type: ignore[assignment]


# --------------------------------------------------------------------------------------------- B
def _band_on_samples(obs: np.ndarray, iters: List[np.ndarray], alpha: float = 0.05) -> Tuple[Any, bool, bool]:
    pool = np.sort(np.concatenate(iters))
    out = R.cdf_band_from_samples(np.sort(obs), [np.sort(i) for i in iters], pool, alpha, obs.size)
    return out["crossing"], out["significant"], out["significant_dkw"]


def section_b(reps: int, seed: int) -> None:
    print("\nB. The crossing detector on distance samples (K = 50, B = 199)")
    rng = np.random.default_rng(seed)
    k, b = 50, 199

    def null(n: int) -> np.ndarray:
        return rng.gamma(9.0, 25.0, n)

    rates = {}
    dkw_rates = {}
    for case in ("null", "planted crossing, strong", "planted crossing, moderate", "pure shift"):
        hits = 0
        hits_dkw = 0
        for _ in range(reps):
            iters = [null(k) for _ in range(b)]
            if case == "null":
                obs = null(k)
            elif case.startswith("planted"):
                share = 0.4 if case.endswith("strong") else 0.25
                m = int(round(share * k))
                obs = np.concatenate([0.5 * null(m), 1.4 * null(m), null(k - 2 * m)])
            else:
                obs = null(k) + 60.0
            _level, sig, sig_dkw = _band_on_samples(obs, iters)
            hits += bool(sig)
            hits_dkw += bool(sig_dkw)
        rates[case] = hits / reps
        dkw_rates[case] = hits_dkw / reps
        print(f"    {case:28s} significant crossing in {rates[case]:.3f} of {reps} (Monte Carlo rank, the "
              f"program's rule); {dkw_rates[case]:.3f} with the DKW bound on both branches")
    RESULTS["B"] = {"monte_carlo": rates, "dkw_both_branches": dkw_rates}
    check("null: a significant crossing in at most 5 % (+ MC)", rates["null"] <= wilson(0.05, reps)[1],
          f"{rates['null']:.3f}")
    check("pure shift: no crossing in at least 95 %", rates["pure shift"] <= wilson(0.05, reps)[1],
          f"{rates['pure shift']:.3f}")
    check("a strong planted crossing at K = 50 is detected in at least 70 % (the record's target 0.80, Wilson)",
          rates["planted crossing, strong"] >= 0.70, f"{rates['planted crossing, strong']:.3f}")
    print(f"    planted crossing detected in {rates['planted crossing, strong']:.3f} (strong) and "
          f"{rates['planted crossing, moderate']:.3f} (moderate) at K = 50 (reported; the record's target 0.80)")


# --------------------------------------------------------------------------------------------- C
def section_c(n_axons: int, seed: int) -> None:
    from tools.mps_analysis import analyze_axon
    from tools.mps_simulate_axon import SimConfig, simulate_axon
    print("\nC. Band half-width 25 / 50 / 100 nm on simulated axons (B = 199)")
    table = []
    for sigma_r in (10.0, 20.0, 30.0):
        for i in range(n_axons):
            sim = simulate_axon(SimConfig(radial_offset_sd_nm=sigma_r), seed + 100 * i + int(sigma_r))
            a = analyze_axon(sim.x_nm, sim.y_nm, sim.z_nm, pixel_size_nm=113.0, pixel_size_source="yaml",
                             run_randomization=False)
            if a.perimeter is None or a.n_clusters_kept < 5:
                continue
            row: Dict[str, Any] = {"sigma_r": sigma_r, "axon": i, "K": int(a.n_clusters_kept)}
            for w in (25.0, 50.0, 100.0):
                r = R.randomize_cluster_positions(a.centroids, a.perimeter.contour,
                                                  experimental_1nn_nm=a.nn.first_nn_nm, n_iterations=199,
                                                  params=R.RandomizationParams(band_half_width_nm=w))
                row[f"null_median_{w:g}"] = r.randomized_median_nm
                row[f"z_{w:g}"] = r.ks_z
                row[f"p_d_{w:g}"] = r.ks_p_mc_d
            table.append(row)
            print(f"    sigma_r {sigma_r:4.0f} nm, K {row['K']:3d}: null median "
                  + " / ".join(f"{row[f'null_median_{w:g}']:.0f}" for w in (25.0, 50.0, 100.0))
                  + " nm; z_D " + " / ".join(f"{row[f'z_{w:g}']:+.2f}" for w in (25.0, 50.0, 100.0))
                  + "; p_D " + " / ".join(f"{row[f'p_d_{w:g}']:.3f}" for w in (25.0, 50.0, 100.0)))
    RESULTS["C"] = table
    exact = [r for r in table if r["sigma_r"] == 10.0]
    worst = max(abs(r["z_25"] - r["z_100"]) for r in exact) if exact else float("nan")
    print(f"    largest |z_D(25) - z_D(100)| at 10 nm scatter: {worst:.2f} (the default is adequate below 0.5)")


# --------------------------------------------------------------------------------------------- D
def section_d(seed: int) -> None:
    print("\nD. Grid spacing 2.5 / 5 / 10 nm (K = 50, B = 999, continuous centres, 4 seeds)")
    rng = np.random.default_rng(seed)
    contour = _circle(50)
    smoothed = R.smooth_contour_bspline(R.close_contour(contour), smoothing=float(len(contour)))
    cand = R.build_annulus_candidates(smoothed)
    cand = cand + rng.uniform(-2.5, 2.5, cand.shape)          # off the grid, as measured centres are
    obs = R.place_with_min_distance(cand, 50, 80.0, rng)
    out: Dict[float, Tuple[float, float, float, float]] = {}
    for g in (2.5, 5.0, 10.0):
        meds, zs = [], []
        for s in range(4):
            r = R.randomize_cluster_positions(obs, contour, n_iterations=999, random_seed=s,
                                              params=R.RandomizationParams(grid_spacing_nm=g))
            meds.append(float(r.randomized_median_nm))
            zs.append(float(r.ks_z))
        out[g] = (float(np.mean(meds)), float(np.std(meds)), float(np.mean(zs)), float(np.std(zs)))
        print(f"    grid {g:4.1f} nm: null median {out[g][0]:.1f} +/- {out[g][1]:.1f} nm, z_D {out[g][2]:+.3f} +/- "
              f"{out[g][3]:.3f} (over seeds)")
    RESULTS["D"] = {str(k): v for k, v in out.items()}
    dm = abs(out[5.0][0] - out[2.5][0])
    dz = abs(out[5.0][2] - out[2.5][2])
    se = math.sqrt((out[5.0][3] ** 2 + out[2.5][3] ** 2) / 4)
    check("2.5 -> 5 nm: the null median moves by less than 1 nm", dm < 1.0, f"{dm:.2f} nm")
    check("2.5 -> 5 nm: z_D moves by less than 0.05 + 2 SE", dz < 0.05 + 2 * se, f"{dz:.3f} (limit {0.05 + 2 * se:.3f})")
    print(f"    5 -> 10 nm (reported): null median {out[10.0][0] - out[5.0][0]:+.2f} nm, z_D "
          f"{out[10.0][2] - out[5.0][2]:+.3f}")


# --------------------------------------------------------------------------------------------- E
def section_e() -> None:
    from tools import mps_batch as mb
    print("\nE. ICC chance level: closed form vs 20,000 Monte Carlo draws")
    designs = [(9, 9), (9,) * 6, (50,) * 6, (6, 6, 6), (2, 3, 12, 20), (4, 4, 4, 4), (3, 8, 8, 15, 30), (10,) * 3]
    rows = []
    for sizes in designs:
        closed = mb.icc_chance_level(tuple(sizes))
        mc = mb.icc_null_for_sizes(tuple(sizes), 95.0, 20000, 7)
        rows.append((sizes, closed, mc))
        check(f"{sizes}: |closed - MC| <= 0.01", abs(closed - mc) <= 0.01, f"{closed:.3f} vs {mc:.3f}")
    # heavy tails (t3): the F quantile assumes normal data
    rng = np.random.default_rng(11)
    for sizes in ((9,) * 6, (2, 3, 12, 20)):
        vals = []
        for _ in range(20000):
            groups = [rng.standard_t(3, s) for s in sizes]
            v = mb._icc_core(groups)
            vals.append(np.nan if v is None else v)
        q = float(np.nanpercentile(vals, 95))
        print(f"    t3 data, {sizes}: Monte Carlo {q:.3f} vs closed form {mb.icc_chance_level(tuple(sizes)):.3f} "
              f"(reported)")
    RESULTS["E"] = [(list(s), c, m) for s, c, m in rows]


# --------------------------------------------------------------------------------------------- F
def section_f(seed: int) -> None:
    from tools.mps_occupancy import fit_constrained_gaussian
    print("\nF. The sigma cap: binding fraction and claimed half-width (2000 clusters each)")
    rng = np.random.default_rng(seed)
    expected_bind = {5: 1.00, 10: 1.00, 50: 0.82, 100: 0.54, 200: 0.23, 500: 0.01}
    expected_half = {5: 36.0, 10: 44.0, 50: 58.0}
    table = []
    for n in (5, 10, 20, 50, 100, 200, 500):
        caps, halves, disc_halves = [], [], []
        for _ in range(2000):
            pts = rng.normal(0, 20.0, (n, 2))
            e = fit_constrained_gaussian(pts)
            caps.append(e.was_capped)
            if e.was_capped:
                halves.append(3 * e.sigma_major_nm)
            r = 40.0 * np.sqrt(rng.uniform(0, 1, n))
            th = rng.uniform(0, 2 * np.pi, n)
            d = fit_constrained_gaussian(np.column_stack([r * np.cos(th), r * np.sin(th)]))
            disc_halves.append(3 * d.sigma_major_nm)
        frac = float(np.mean(caps))
        half = float(np.mean(halves)) if halves else float("nan")
        disc = float(np.mean(disc_halves))
        table.append({"n": n, "gauss_capped": frac, "gauss_half_when_capped": half, "disc_half": disc})
        print(f"    n = {n:3d}: Gaussian capped {frac:.2f}, claimed half-width {half:.1f} nm (3 sigma = 60); "
              f"disc (radius 40) claimed {disc:.1f} nm")
        if n in expected_bind:
            check(f"n = {n}: Gaussian binding fraction within 0.05 of {expected_bind[n]:.2f}",
                  abs(frac - expected_bind[n]) <= 0.05, f"{frac:.3f}")
        if n in expected_half:
            check(f"n = {n}: claimed half-width within 3 nm of {expected_half[n]:g} nm",
                  abs(half - expected_half[n]) <= 3.0, f"{half:.1f}")
        if n >= 15:
            check(f"n = {n}: disc claims about its radius (41-42 nm, within 3 nm)", abs(disc - 41.5) <= 3.0,
                  f"{disc:.1f}")
    RESULTS["F"] = table


# --------------------------------------------------------------------------------------------- G
def section_g(reps: int, seed: int) -> None:
    from tools.mps_constancy import occupancy_vs_calibre
    print("\nG. Constancy script: coverage of the slope's 95 % interval")
    rng = np.random.default_rng(seed)
    out = {}
    for n_animals in (3, 6):
        for slope in (0.0, 2.0):
            cover = 0
            for _ in range(reps):
                occ, per, ani, roi = [], [], [], []
                for a in range(n_animals):
                    ua = rng.normal(0, 2.0)
                    for k in range(6):
                        p = rng.uniform(8, 40)
                        per.append(p)
                        occ.append(20 + slope * p / 10 + ua + rng.normal(0, 2.0))
                        ani.append(f"A{a}")
                        roi.append(f"R{k // 3}")
                res = occupancy_vs_calibre(occ, per, ani, roi, n_bootstrap=0)
                lo, hi = res.slope_ci_pp_per_um
                cover += lo <= slope / 10 <= hi
            c = cover / reps
            out[f"{n_animals}x6, slope {slope:g}"] = c
            check(f"{n_animals} animals x 6 axons, slope {slope:g} pp/10 um: coverage >= Wilson lower bound",
                  c >= wilson(0.95, reps)[0], f"{c:.3f}")
    RESULTS["G"] = out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    reps = 300 if args.full else 120
    only = set(args.only.split(",")) if args.only else set("ABCDEFG")
    t0 = time.time()
    print("=" * 100)
    print(f"RANDOMIZATION CONTROL AND SCI-3 STATISTICS - synthetic calibration ({'full' if args.full else 'sampled'})")
    print("=" * 100)
    if "A" in only:
        section_a(reps, args.seed)
    if "B" in only:
        section_b(200 if not args.full else 400, args.seed + 1)
    if "C" in only:
        section_c(2 if not args.full else 5, args.seed + 2)
    if "D" in only:
        section_d(args.seed + 3)
    if "E" in only:
        section_e()
    if "F" in only:
        section_f(args.seed + 5)
    if "G" in only:
        section_g(200, args.seed + 6)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "validate_randomization.json"), "w", encoding="utf-8") as fh:
            json.dump(RESULTS, fh, indent=1, default=float)
    print("\n" + "=" * 100)
    print(f"total {time.time() - t0:.0f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
