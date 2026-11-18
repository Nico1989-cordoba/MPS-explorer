"""
V2 of the perimeter density analysis (package P11): question C, correlation between slices.

    python validate_perimeter_c.py run --case null|identical|rho0.2|rho0.3|rho0.5|leak --n N --out DIR
    python validate_perimeter_c.py report --out DIR

Each simulated axon is a circle whose perimeter holds 80-250 nominal bins (the range of the real axons
[real data, pilot: 6.7-32 um]), ``random`` density (P1) with ~33 localizations per bin and slice, blinking, and the
analysis pipeline from the projection on the TRUE contour onwards: P2 slices (z0 = 0, contiguous 170 nm), P4
band +-100 nm and profiles, P5 ``slice_correlations`` with 1999 rotations (every shift: n_bins - 1 < 1999).
The fitted-contour negatives asked by review R1 are in ``validate_perimeter_b.py fitted`` (one simulated
axon serves both B and C there); this report reads their CSV when it exists.

Criteria [plan sec. 4.4 P5 and sec. 5 V2]: false positives 0.03-0.07 for every pair under the null and p uniform
(KS); power >= 0.8 on the identical pattern; leakage only: neighbour r ~ 0.27 +- 0.05 and significant, r(-1, +1)
not significant (false positives <= 0.07), the reading says "leakage only"; with a 42.5 nm guard the neighbour r
< 0.10 (plan) -- the count says ~0.12 (P1 and P5 agree), reported as FAIL if so.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import validate_perimeter_cross as X  # noqa: E402
from tools import mps_perimeter_bc as BC  # noqa: E402
from tools import mps_perimeter_sim as S  # noqa: E402

PAIRS = (((-1, 0), "m1_0"), ((0, 1), "0_p1"), ((-1, 1), "m1_p1"))
N_BINS_CHOICES = (80, 120, 160, 200, 250)     # [real data, pilot] perimeters 8-25 um -> 80-250 bins
LOCS_PER_BIN = 100                            # [ad hoc] ~33 per bin and slice (3 slices)
CASES: Dict[str, Dict[str, Any]] = {
    "null": {"pattern": "independent", "lpz_nm": 0.0},
    "identical": {"pattern": "identical", "lpz_nm": 47.0},
    "rho0.2": {"pattern": "correlated", "rho": 0.2, "lpz_nm": 47.0},
    "rho0.3": {"pattern": "correlated", "rho": 0.3, "lpz_nm": 47.0},
    "rho0.5": {"pattern": "correlated", "rho": 0.5, "lpz_nm": 47.0},
    "leak": {"pattern": "independent", "lpz_nm": 47.0},
}
CASE_SEED = {name: 100000 * (i + 1) for i, name in enumerate(CASES)}


def one_axon(case: str, i: int, n_null: int = 1999) -> Dict[str, Any]:
    cfg = CASES[case]
    seed = CASE_SEED[case] + i
    rng = np.random.default_rng(seed)
    n_bins = int(rng.choice(N_BINS_CHOICES))
    slices: Dict[str, Any] = {"pattern": cfg["pattern"]}
    if "rho" in cfg:
        slices["rho"] = cfg["rho"]
    sim = S.simulate_perimeter("circle", {"kind": "random"}, n_locs=LOCS_PER_BIN * n_bins, radial_sd_nm=20.0,
                               slices=slices, lpz_nm=cfg["lpz_nm"], background_frac=0.0, interior_frac=0.0,
                               blink=True, seed=seed, options={"radius_nm": X.radius_for_bins(n_bins)})
    row: Dict[str, Any] = {"case": case, "axon": i, "seed": seed, "n_bins_nominal": n_bins}
    for guard in ((0.0, 42.5) if case == "leak" else (0.0,)):
        sp, _, _, _ = X.profiles_of(sim, guard_nm=guard)
        c = BC.slice_correlations(sp, n_null=n_null, seed=i)
        tag = "" if guard == 0.0 else "_g42"
        for pair, name in PAIRS:
            row[f"r_{name}{tag}"] = c[pair].value
            row[f"p_{name}{tag}"] = c[pair].p
        if guard == 0.0:
            row["n_bins"] = sp.by_slice[0].n_bins
            row["reading"] = BC.slice_reading(c, BC.leak_expected_r(cfg["lpz_nm"] or 0.0, 170.0, 0.0))
            row["true_r_m1_p1"] = float(sim.truth["true_slice_lambda_r"][0, 2])
    return row


def run_case(case: str, n: int, start: int = 0) -> List[Dict[str, Any]]:
    return [one_axon(case, i) for i in range(start, start + n)]


# ----------------------------------------------------------------------------------------------------
# Review R1's unmeasured risk: a contour error COMMON to the three slices
# ----------------------------------------------------------------------------------------------------
CERR_AMPS_NM: Sequence[float] = (0.0, 50.0, 75.0, 100.0, 150.0, 200.0)   # [ad hoc] up to the real persistent
#                                  offsets of the pilot curves (98-309 nm, P3 v2 report sec. 7)
CERR_SIGMA_NM: float = 300.0       # [ad hoc] bump sd: FWHM ~700 nm, the stretch length the quality gate reads
CERR_SPACING_NM: float = 3000.0    # [ad hoc] one bump per 3 um of perimeter, random sign
CERR_SEED = 900000


def perturbed_path(path: Any, centres: np.ndarray, signs: np.ndarray, amp_nm: float,
                   sigma_nm: float = CERR_SIGMA_NM) -> tuple:
    """The curve moved along its normal by a sum of Gaussian bumps (amplitude ``amp_nm``, sd ``sigma_nm``, at the
    arcs ``centres`` with ``signs``). Returns (SmoothPath, rms displacement, max |displacement|)."""
    from tools.mps_matching import SmoothPath
    pts = np.asarray(path.points_nm, dtype=np.float64)
    cum = np.asarray(path.cum_nm, dtype=np.float64)[: pts.shape[0]]
    L = float(path.length_nm)
    dd = np.abs(cum[:, None] - centres[None, :])
    dd = np.minimum(dd, L - dd)
    disp = amp_nm * np.sum(signs[None, :] * np.exp(-0.5 * (dd / sigma_nm) ** 2), axis=1)
    tan = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tn = np.hypot(tan[:, 0], tan[:, 1])
    tn[tn == 0] = 1.0
    normal = np.column_stack([tan[:, 1] / tn, -tan[:, 0] / tn])
    new = pts + disp[:, None] * normal
    nxt = np.roll(new, -1, axis=0)
    edges = np.hypot(nxt[:, 0] - new[:, 0], nxt[:, 1] - new[:, 1]).astype(np.float64)
    cn = np.concatenate([[0.0], np.cumsum(edges)]).astype(np.float64)
    sp = SmoothPath(points_nm=new, vertex_row=np.asarray(path.vertex_row), edges_nm=edges, cum_nm=cn,
                    length_nm=float(cn[-1]))
    return sp, float(np.sqrt(np.mean(disp ** 2))), float(np.max(np.abs(disp)))


def cerr_axon(sd: float, i: int, n_null: int = 1999) -> List[Dict[str, Any]]:
    """One wavy axon, ``random`` density with INDEPENDENT slice patterns and NO axial leak (a negative for C),
    read on the true curve deformed by a common error of growing amplitude. Every positive C is false; the
    quality gate of P3 is evaluated on each deformed curve (as if it were the fitted one)."""
    from tools import mps_auto_contour as AC
    seed = CERR_SEED + int(sd) * 1000 + i
    rng = np.random.default_rng(seed)
    radius = float(rng.uniform(1300.0, 3500.0))
    opt = {"radius_nm": radius, "wave_phases": tuple(float(v) for v in rng.uniform(0.0, 2 * np.pi, 4)),
           "rotation_deg": float(rng.uniform(0.0, 360.0))}
    wide = sd > 50
    sim = S.simulate_perimeter("wavy", {"kind": "random"}, n_locs=int(LOCS_PER_BIN * 2 * np.pi * radius / 100.0),
                               radial_sd_nm=sd, slices={"pattern": "independent"}, lpz_nm=0.0,
                               background_frac=0.15 if wide else 0.0, interior_frac=0.10 if wide else 0.0,
                               blink=True, seed=seed, options=opt)
    L = float(sim.true_path.length_nm)
    k = max(1, int(round(L / CERR_SPACING_NM)))
    centres = rng.uniform(0.0, L, k)
    signs = rng.choice([-1.0, 1.0], size=k)
    rows: List[Dict[str, Any]] = []
    for amp in CERR_AMPS_NM:
        path, rms, mx = perturbed_path(sim.true_path, centres, signs, amp)
        sp, _s, _r, lab = X.profiles_of(sim, path)
        c = BC.slice_correlations(sp, n_null=n_null, seed=i)
        in3 = lab != -9
        q = AC.contour_quality(path, sim.x_nm[in3], sim.y_nm[in3])
        row: Dict[str, Any] = {"case": "cerr", "radial_sd_nm": sd, "axon": i, "seed": seed, "amp_nm": amp,
                               "rms_disp_nm": rms, "max_disp_nm": mx, "n_bumps": k, "length_nm": L,
                               "gate_reliable": not q.low_quality, "persistent_offset_nm": q.persistent_offset_nm,
                               "frac100": q.frac_within[100.0], "robust_sd_nm": q.robust_sd_nm,
                               "off_band_fraction": q.off_band_fraction}
        for pair, name in PAIRS:
            row[f"r_{name}"] = c[pair].value
            row[f"p_{name}"] = c[pair].p
        rows.append(row)
    return rows


def cpos_fitted_verdicts(path: str) -> List[Dict[str, Any]]:
    """C positive (identical pattern, leak 47 nm) read on the true and on the fitted contour."""
    if not os.path.exists(path):
        return []
    rows = X.read_csv(path)
    out: List[Dict[str, Any]] = []
    for sd in sorted({r["radial_sd_nm"] for r in rows}, key=X.fnum):
        for kind in ("true",) + X.CURVE_KINDS:
            sel = [r for r in rows if r["radial_sd_nm"] == sd and r["asked"] == kind]
            if not sel:
                continue
            k, n = _rate(sel, "c_p_m1_p1")
            rr = np.array([X.fnum(x.get("c_r_m1_p1")) for x in sel])
            out.append(X.verdict(f"C positive (identical), contour '{kind}', radial sd {sd} nm, pair -1/+1",
                                 "power >= 0.8", f"power {X.rate_text(k, len(sel))} (accepted {n}); mean r "
                                 f"{np.nanmean(rr) if np.isfinite(rr).any() else float('nan'):+.3f}",
                                 bool(sel and k / len(sel) >= X.POWER_MIN)))
        rel = [r for r in rows if r["radial_sd_nm"] == sd and r["asked"] == "auto" and str(r.get("reliable")) == "True"]
        k, n = _rate(rel, "c_p_m1_p1")
        out.append(X.verdict(f"C positive, auto contour, only gate-reliable, radial sd {sd} nm", "power >= 0.8",
                             X.rate_text(k, n), bool(n and k / n >= X.POWER_MIN)))
    return out


def cerr_verdicts(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = X.read_csv(path)
    out: List[Dict[str, Any]] = []
    for sd in sorted({r["radial_sd_nm"] for r in rows}, key=X.fnum):
        for amp in sorted({r["amp_nm"] for r in rows}, key=X.fnum):
            sel = [r for r in rows if r["radial_sd_nm"] == sd and r["amp_nm"] == amp]
            k, n = _rate(sel, "p_m1_p1")
            kn = sum(int(X.fnum(r["p_m1_0"]) < X.ALPHA or X.fnum(r["p_0_p1"]) < X.ALPHA) for r in sel)
            rr = np.array([X.fnum(r["r_m1_p1"]) for r in sel])
            rel = [r for r in sel if str(r["gate_reliable"]) == "True"]
            kr, nr = _rate(rel, "p_m1_p1")
            rms = np.nanmedian([X.fnum(r["rms_disp_nm"]) for r in sel])
            pers = np.nanmedian([X.fnum(r["persistent_offset_nm"]) for r in sel])
            out.append(X.verdict(f"C common contour error, radial sd {sd} nm, bump {X.fnum(amp):g} nm",
                                 "false positives (-1,+1) <= 0.07",
                                 f"(-1,+1) {X.rate_text(k, n)}; mean r {np.nanmean(rr):+.3f}; a neighbour pair "
                                 f"significant in {kn}/{len(sel)}; rms displacement {rms:.0f} nm; persistent offset "
                                 f"{pers:.0f} nm; gate reliable {len(rel)}/{len(sel)}; FP among reliable "
                                 f"{X.rate_text(kr, nr)}", bool(n and k / n <= 0.07)))
    return out


def _rate(rows: Sequence[Dict[str, Any]], key: str) -> tuple:
    p = np.array([X.fnum(r.get(key)) for r in rows])
    fin = np.isfinite(p)
    k = int(np.count_nonzero(p[fin] < X.ALPHA))
    return k, int(fin.sum())


def verdicts(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by = {c: [r for r in rows if r["case"] == c] for c in CASES}
    lo, hi = X.FPR_BAND
    null = by["null"]
    if null:
        for _pair, name in PAIRS:
            k, n = _rate(null, f"p_{name}")
            d, pks = X.ks_uniform([X.fnum(r[f"p_{name}"]) for r in null])
            out.append(verdict_fpr(f"V2 null (independent, no leak), pair {name}", k, n))
            out.append(X.verdict(f"V2 null p uniform, pair {name}", "KS p > 0.05", f"D {d:.3f}, KS p {pks:.3f}",
                                 pks > 0.05))
    ident = by["identical"]
    if ident:
        k, n = _rate(ident, "p_m1_p1")
        r = np.array([X.fnum(x["r_m1_p1"]) for x in ident])
        out.append(X.verdict("V2 positive (identical pattern), pair -1/+1", "power >= 0.8 [plan V2]",
                             f"power {X.rate_text(k, n)}; median r {np.nanmedian(r):.3f}",
                             n > 0 and k / n >= X.POWER_MIN,
                             "r is capped by Poisson + blinking noise at ~33 locs per bin and slice; P5's r > 0.8 "
                             "needs denser profiles"))
    for rc in ("rho0.2", "rho0.3", "rho0.5"):
        if by[rc]:
            k, n = _rate(by[rc], "p_m1_p1")
            r = np.array([X.fnum(x["r_m1_p1"]) for x in by[rc]])
            out.append(X.verdict(f"V2 dose-response {rc}, pair -1/+1", "informative (no plan criterion)",
                                 f"power {X.rate_text(k, n)}; median r {np.nanmedian(r):.3f}", None))
    leak = by["leak"]
    if leak:
        nb = np.array([0.5 * (X.fnum(r["r_m1_0"]) + X.fnum(r["r_0_p1"])) for r in leak])
        expected = BC.leak_expected_r(47.0, 170.0, 0.0)
        k1, n1 = _rate(leak, "p_m1_0")
        k2, n2 = _rate(leak, "p_0_p1")
        out.append(X.verdict("V2 leakage only: neighbour r", "0.27 +- 0.05 and significant (power >= 0.8)",
                             f"mean r {np.nanmean(nb):.3f} (P5 expects {expected:.3f}, a no-blink bound); "
                             f"significant {X.rate_text(k1, n1)} / {X.rate_text(k2, n2)}",
                             abs(float(np.nanmean(nb)) - 0.27) <= 0.05 and min(k1 / n1, k2 / n2) >= X.POWER_MIN))
        k, n = _rate(leak, "p_m1_p1")
        out.append(verdict_fpr("V2 leakage only: pair -1/+1 must not be significant", k, n, upper_only=True))
        lab = [str(r["reading"]) for r in leak]
        kk = sum(x == "leakage only" for x in lab)
        reg = sum(x == "registry between slices" for x in lab)
        out.append(X.verdict("V2 leakage recognised by slice_reading", "'leakage only' in most axons (>= 0.8); "
                             "'registry' <= 0.07", f"leakage only {X.rate_text(kk, len(lab))}; registry "
                             f"{X.rate_text(reg, len(lab))}; others: "
                             f"{sorted({x for x in lab if x not in ('leakage only', 'registry between slices')})}",
                             kk / len(lab) >= 0.8 and reg / len(lab) <= 0.07))
        g = np.array([0.5 * (X.fnum(r["r_m1_0_g42"]) + X.fnum(r["r_0_p1_g42"])) for r in leak])
        out.append(X.verdict("V2 guard 42.5 nm: neighbour r", "< 0.10 [plan]; the count says 0.12",
                             f"mean r {np.nanmean(g):.3f} (P5 expects {BC.leak_expected_r(47.0, 170.0, 42.5):.3f})",
                             float(np.nanmean(g)) < 0.10, "plan deviation if FAIL: documented, not relaxed"))
    return out


def verdict_fpr(check: str, k: int, n: int, upper_only: bool = False) -> Dict[str, Any]:
    lo, hi = X.FPR_BAND
    rate = k / n if n else float("nan")
    ok = (rate <= hi) if upper_only else (lo <= rate <= hi)
    crit = f"false positives <= {hi}" if upper_only else f"false positives {lo}-{hi}"
    if not n:
        return X.verdict(check, crit, "n = 0", None, "not measurable (no accepted contour)")
    return X.verdict(check, crit, X.rate_text(k, n), bool(ok))


def fitted_verdicts(path: str) -> List[Dict[str, Any]]:
    """C false positives with the FITTED contour (review R1), from ``validate_perimeter_b.py fitted``."""
    if not os.path.exists(path):
        return []
    rows = X.read_csv(path)
    out: List[Dict[str, Any]] = []
    for wide in sorted({r["radial_sd_nm"] for r in rows}):
        for kind in ("true",) + X.CURVE_KINDS:
            sel = [r for r in rows if r["radial_sd_nm"] == wide and r["asked"] == kind]
            if not sel:
                continue
            for _pair, name in PAIRS:
                k, n = _rate(sel, f"c_p_{name}")
                r = np.array([X.fnum(x.get(f"c_r_{name}")) for x in sel])
                v = verdict_fpr(f"C null with contour '{kind}', radial sd {wide} nm, pair {name}", k, n,
                                upper_only=True)
                v["measured"] += f"; mean r {np.nanmean(r):+.3f}" if np.isfinite(r).any() else ""
                out.append(v)
        for used in sorted({r["used"] for r in rows if r["radial_sd_nm"] == wide and r["asked"] != "true"
                            and r["used"] != "failed"}):
            sel = [r for r in rows if r["radial_sd_nm"] == wide and r["asked"] != "true" and r["used"] == used]
            k, n = _rate(sel, "c_p_m1_p1")
            out.append(X.verdict(f"C null pair -1/+1 by curve USED '{used}', radial sd {wide} nm",
                                 "false positives <= 0.07", X.rate_text(k, n), bool(n and k / n <= 0.07)))
        for relv in ("True", "False"):
            sel = [r for r in rows if r["radial_sd_nm"] == wide and r["asked"] == "auto"
                   and str(r.get("reliable", "")) == relv and str(r.get("pattern", "independent")) == "independent"]
            if not sel:
                continue
            k, n = _rate(sel, "c_p_m1_p1")
            rr = np.array([X.fnum(x.get("c_r_m1_p1")) for x in sel])
            out.append(X.verdict(f"C null pair -1/+1, auto contour, gate reliable={relv}, radial sd {wide} nm",
                                 "false positives <= 0.07", X.rate_text(k, n) + f"; mean r {np.nanmean(rr):+.3f}",
                                 bool(n and k / n <= 0.07)))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P11 V2 (question C)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--case", required=True, choices=sorted(CASES))
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--start", type=int, default=0)
    a.add_argument("--out", required=True)
    e = sub.add_parser("cerr")
    e.add_argument("--sd", type=float, required=True)
    e.add_argument("--n", type=int, required=True)
    e.add_argument("--start", type=int, default=0)
    e.add_argument("--out", required=True)
    b = sub.add_parser("report")
    b.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = X.ensure_out(args.out)
    t0 = time.time()
    if args.cmd == "run":
        rows = run_case(args.case, args.n, args.start)
        X.write_csv(os.path.join(out, "v2_c.csv"), rows, append=True)
    elif args.cmd == "cerr":
        rows_e: List[Dict[str, Any]] = []
        for i in range(args.start, args.start + args.n):
            rows_e += cerr_axon(args.sd, i)
        X.write_csv(os.path.join(out, "v2_cerr.csv"), rows_e, append=True)
    else:
        p_c = os.path.join(out, "v2_c.csv")
        rows_all: List[Dict[str, Any]] = [dict(r) for r in X.read_csv(p_c)] if os.path.exists(p_c) else []
        v = (verdicts(rows_all) + fitted_verdicts(os.path.join(out, "v3_fitted.csv"))
             + cpos_fitted_verdicts(os.path.join(out, "v3_fitted_cpos.csv"))
             + cerr_verdicts(os.path.join(out, "v2_cerr.csv")))
        sizes = {c: sum(r["case"] == c for r in rows_all) for c in CASES}
        X.write_verdicts(os.path.join(out, "VERDICT_v2_c.md"), "V2 question C (correlation between slices)", v,
                         [f"Axons per case: {sizes}. True contour; circle with 80-250 bins; {LOCS_PER_BIN} locs per "
                          "bin (3 slices); blinking; 1999 rotations (all shifts). Fitted-contour rows: v3_fitted.csv."])
        for x in v:
            print(x["verdict"], x["check"], "|", x["measured"], flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
