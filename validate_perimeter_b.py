"""
V3 of the perimeter density analysis (package P11): question B, density against curvature (tortuosity), and
the fitted-contour negatives of B and C asked by review R1.

    python validate_perimeter_b.py true --case null|pos_wavy|pos_ellipse --n N [--start K] --out DIR
    python validate_perimeter_b.py fitted --sd 20|100 --n N [--start K] --out DIR
    python validate_perimeter_b.py fitted-pos --n N [--start K] --out DIR
    python validate_perimeter_b.py report --out DIR

``true``: the TRUE contour of the simulator (P1), so only the statistic is under test.
  null        wavy shape (random harmonic phases, radius 1.3-3.5 um), lambda ~ 1 + beta kappa R with beta = 0.
  pos_wavy    the plan's positive: wavy, R = 1.5 um, beta = 0.5, 10 000 localizations.
  pos_ellipse the same on a 2-fold symmetric ellipse (P6's documented limit: power 0.60).
``fitted``: one simulated axon serves B and C. Wavy shape, ``random`` density with INDEPENDENT slice patterns
  and no axial leak (so density is independent of curvature and the slices are independent: a negative for both
  questions), read on four contours of the same cloud: the true one, P3's automatic curve (defaults and
  fallbacks), and the two fallbacks FORCED (600 nm knot, polar). Review R1 derived that a local contour error
  lowers the same bins in the three slices and makes a common positive correlation; this measures it.
  ``--sd 100`` makes the cloud as wide as the real one (radial sd 100 nm, 10 % interior, 15 % background).
``fitted-pos``: the plan's positive with P3's automatic contour.

B is read as the orchestration does (``mps_perimeter_analysis._read_abc``): central slice, band +-50 / 100 / 150
nm recomputed each time, ``axon_b_readout`` with the radial-offset control, 1999 rotations, two-sided.
Criteria [plan sec. 4.4 P6 and sec. 5 V3]: false positives 0.03-0.07 with the true contour; with the fitted
contour a cell above 0.08 is documented and B is conditioned on that band; power >= 0.8 for the positive.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import validate_perimeter_cross as X  # noqa: E402
from tools import mps_perimeter_bc as BC  # noqa: E402
from tools import mps_perimeter_density as D  # noqa: E402
from tools import mps_perimeter_sim as S  # noqa: E402
from tools.mps_matching import SmoothPath  # noqa: E402
from tools.mps_perimeter_slices import cut_slices  # noqa: E402

BANDS: Tuple[float, ...] = (50.0, 100.0, 150.0)   # [plan] main band and the mandatory sensitivities
FP_HARD_MAX: float = 0.08                         # [plan sec. 4.4 P6] hard negative threshold
N_NULL: int = 1999                                # [lit] as the analysis
PAIRS = (((-1, 0), "m1_0"), ((0, 1), "0_p1"), ((-1, 1), "m1_p1"))
SEED_BASE = {"null": 300000, "pos_wavy": 310000, "pos_ellipse": 320000, "fitted20": 330000, "fitted100": 340000,
             "fitted_pos": 350000, "fitted_cpos": 360000}


def read_bc(sim: S.SimPerimeter, path: SmoothPath, *, n_null: int = N_NULL, seed: int = 0,
            bands: Sequence[float] = BANDS, with_c: bool = True) -> Dict[str, Any]:
    """B at every band and C at 100 nm on ``path`` (the orchestration's reading, z0 = 0)."""
    s, r = D.project(path, sim.x_nm, sim.y_nm)
    label = cut_slices(sim.z_nm, 0.0).label
    L = float(path.length_nm)
    out: Dict[str, Any] = {}
    for hw in bands:
        sp = D.slice_profiles(s, r, label, L, half_width_nm=hw)
        p0 = sp.by_slice[0]
        sel0 = (label == 0) & np.isfinite(r) & (np.abs(r) <= hw)
        wrm = BC.window_radial_median(s[sel0], r[sel0], length_nm=L, n_bins=p0.n_bins,
                                      origin_arc_nm=p0.origin_arc_nm)
        b = BC.axon_b_readout(path, p0.counts, p0.measured, origin_arc_nm=p0.origin_arc_nm, radial_median_nm=wrm,
                              n_null=n_null, seed=seed)
        tag = f"{int(hw)}"
        out[f"b_r_{tag}"] = b.test.value
        out[f"b_p_{tag}"] = b.test.p
        out[f"b_z_{tag}"] = b.test.z
        out[f"tau_vs_offset_{tag}"] = b.tau_vs_offset
        if hw == 100.0:
            out["tau_min"] = b.tau_min
            out["resolved"] = b.resolved
            out["n_bins"] = p0.n_bins
            if with_c:
                c = BC.slice_correlations(sp, n_null=n_null, seed=seed)
                for pair, name in PAIRS:
                    out[f"c_r_{name}"] = c[pair].value
                    out[f"c_p_{name}"] = c[pair].p
    bw = D.band_width(r[label != -9])
    out["frac_within_100"] = bw.frac_within[100.0]
    out["radial_sd_robust_nm"] = bw.robust_sd_nm
    return out


def _wavy_options(rng: np.random.Generator, radius: Optional[float] = None) -> Dict[str, Any]:
    return {"radius_nm": float(rng.uniform(1300.0, 3500.0)) if radius is None else radius,
            "wave_phases": tuple(float(v) for v in rng.uniform(0.0, 2 * math.pi, 4)),
            "rotation_deg": float(rng.uniform(0.0, 360.0))}


def true_axon(case: str, i: int) -> Dict[str, Any]:
    seed = SEED_BASE[case] + i
    rng = np.random.default_rng(seed)
    if case == "null":
        opt = _wavy_options(rng)
        L = 2 * math.pi * opt["radius_nm"]
        sim = S.simulate_perimeter("wavy", {"kind": "curvature", "beta": 0.0}, n_locs=int(70 * L / 100.0),
                                   radial_sd_nm=20.0, slices={}, lpz_nm=47.0, background_frac=0.0,
                                   interior_frac=0.0, blink=True, seed=seed, options=opt)
    elif case == "pos_wavy":
        sim = S.simulate_perimeter("wavy", {"kind": "curvature", "beta": 0.5}, n_locs=10000, radial_sd_nm=20.0,
                                   slices={}, lpz_nm=47.0, background_frac=0.0, interior_frac=0.0, blink=True,
                                   seed=seed, options=_wavy_options(rng, 1500.0))
    else:
        sim = S.simulate_perimeter("ellipse", {"kind": "curvature", "beta": 0.5}, n_locs=10000, radial_sd_nm=20.0,
                                   slices={}, lpz_nm=47.0, background_frac=0.0, interior_frac=0.0, blink=True,
                                   seed=seed, options={"radius_nm": 1500.0,
                                                       "rotation_deg": float(rng.uniform(0, 360))})
    row: Dict[str, Any] = {"case": case, "axon": i, "seed": seed, "asked": "true", "used": "true",
                           "length_nm": float(sim.true_path.length_nm),
                           "min_radius_curv_nm": sim.truth["min_radius_of_curvature_nm"]}
    row.update(read_bc(sim, sim.true_path, seed=i, bands=(100.0,), with_c=False))
    return row


def assessment_cols(det: Any) -> Dict[str, Any]:
    """P3's quality gate on the fitted curve: reliable, curve kind and the gate's numbers."""
    a = det.assessment
    out: Dict[str, Any] = {"reliable": bool(a.reliable) if a is not None else False,
                           "kind": a.kind if a is not None else "none"}
    if a is not None and a.quality is not None:
        q = a.quality
        out.update(gate_frac100=q.frac_within[100.0], gate_sd_nm=q.robust_sd_nm,
                   gate_persistent_nm=q.persistent_offset_nm, gate_off_band=q.off_band_fraction,
                   gate_reasons=" | ".join(q.reasons)[:300])
    return out


def fitted_axon(sd: float, i: int, kinds: Sequence[str] = X.CURVE_KINDS, pattern: str = "independent"
                ) -> List[Dict[str, Any]]:
    """``pattern`` 'independent' (the negative of B and C) or 'identical' (the positive of C: the same random
    pattern in the three slices, axial leak of 47 nm; B stays a negative)."""
    key = ("fitted20" if sd <= 50 else "fitted100") if pattern == "independent" else "fitted_cpos"
    seed = SEED_BASE[key] + i + (0 if pattern == "independent" else int(sd) * 1000)
    rng = np.random.default_rng(seed)
    opt = _wavy_options(rng)
    L = 2 * math.pi * opt["radius_nm"]
    wide = sd > 50
    sim = S.simulate_perimeter("wavy", {"kind": "random"}, n_locs=int(100 * L / 100.0), radial_sd_nm=sd,
                               slices={"pattern": pattern}, lpz_nm=0.0 if pattern == "independent" else 47.0,
                               background_frac=0.15 if wide else 0.0, interior_frac=0.10 if wide else 0.0,
                               blink=True, seed=seed, options=opt)
    rows: List[Dict[str, Any]] = []
    base = {"axon": i, "seed": seed, "radial_sd_nm": sd, "length_true_nm": L, "pattern": pattern}
    rows.append({**base, "asked": "true", "used": "true", **read_bc(sim, sim.true_path, seed=i)})
    for kind in kinds:
        t0 = time.time()
        det = X.fit_curve(sim.x_nm, sim.y_nm, kind)
        c = det.contour
        row: Dict[str, Any] = {**base, "asked": kind, "used": X.curve_type(c), "fit_seconds": round(time.time() - t0, 2),
                               **assessment_cols(det)}
        if c.path is not None:
            row.update(X.curve_error(sim, c.path))
            row.update(read_bc(sim, c.path, seed=i))
        rows.append(row)
    return rows


def fitted_pos_axon(i: int) -> Dict[str, Any]:
    seed = SEED_BASE["fitted_pos"] + i
    rng = np.random.default_rng(seed)
    sim = S.simulate_perimeter("wavy", {"kind": "curvature", "beta": 0.5}, n_locs=10000, radial_sd_nm=20.0,
                               slices={}, lpz_nm=47.0, background_frac=0.0, interior_frac=0.0, blink=True,
                               seed=seed, options=_wavy_options(rng, 1500.0))
    det = X.fit_curve(sim.x_nm, sim.y_nm, "auto")
    c = det.contour
    row: Dict[str, Any] = {"case": "fitted_pos", "axon": i, "seed": seed, "asked": "auto", "used": X.curve_type(c),
                           **assessment_cols(det)}
    if c.path is not None:
        row.update(X.curve_error(sim, c.path))
        row.update(read_bc(sim, c.path, seed=i, with_c=False))
    return row


# ----------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------
def _rate(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[int, int]:
    p = np.array([X.fnum(r.get(key)) for r in rows])
    fin = np.isfinite(p)
    return int(np.count_nonzero(p[fin] < X.ALPHA)), int(fin.sum())


def _med(rows: Sequence[Dict[str, Any]], key: str) -> float:
    v = np.array([X.fnum(r.get(key)) for r in rows])
    return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")


def verdicts(true_rows: Sequence[Dict[str, Any]], fitted_rows: Sequence[Dict[str, Any]],
             pos_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    lo, hi = X.FPR_BAND
    null = [r for r in true_rows if r["case"] == "null"]
    if null:
        k, n = _rate(null, "b_p_100")
        d, pks = X.ks_uniform([X.fnum(r["b_p_100"]) for r in null])
        out.append(X.verdict("V3 null, true contour (wavy, beta 0)", f"false positives {lo}-{hi}",
                             f"{X.rate_text(k, n)}; KS p {pks:.3f}; median tau_vs_offset "
                             f"{_med(null, 'tau_vs_offset_100'):+.3f}", bool(n and lo <= k / n <= hi)))
    for case, label, plan_shape in (("pos_wavy", "V3 positive, true contour, wavy beta 0.5", True),
                                    ("pos_ellipse", "V3 positive, true contour, symmetric ellipse beta 0.5", False)):
        sel = [r for r in true_rows if r["case"] == case]
        if sel:
            k, n = _rate(sel, "b_p_100")
            out.append(X.verdict(label, "power >= 0.8 (10 000 locs, R >= 600 nm)" if plan_shape else
                                 "power >= 0.8 (plan); P6 measured 0.60 on this shape",
                                 f"power {X.rate_text(k, n)}; median Spearman {_med(sel, 'b_r_100'):+.3f}",
                                 bool(n and k / n >= X.POWER_MIN)))
    if pos_rows:
        accepted = [r for r in pos_rows if r.get("b_p_100") not in (None, "")]
        k, n = _rate(accepted, "b_p_100")
        out.append(X.verdict("V3 positive, P3 automatic contour, wavy beta 0.5", "power >= 0.8",
                             f"power {X.rate_text(k, len(pos_rows))} (contour failed in {len(pos_rows) - n})",
                             bool(pos_rows and k / len(pos_rows) >= X.POWER_MIN)))
        rel = [r for r in accepted if str(r.get("reliable")) == "True"]
        kr, nr = _rate(rel, "b_p_100")
        out.append(X.verdict("V3 positive, P3 automatic contour, only gate-reliable curves", "power >= 0.8",
                             f"power {X.rate_text(kr, nr)}; reliable {nr}/{len(pos_rows)}",
                             bool(nr and kr / nr >= X.POWER_MIN)))
        for kind in sorted({str(r.get("kind")) for r in accepted}):
            sel = [r for r in accepted if str(r.get("kind")) == kind]
            kk, nn = _rate(sel, "b_p_100")
            out.append(X.verdict(f"V3 positive, auto contour, curve kind '{kind}'", "informative",
                                 X.rate_text(kk, nn), None))
    for sd in sorted({r["radial_sd_nm"] for r in fitted_rows}, key=X.fnum):
        for kind in ("true",) + X.CURVE_KINDS:
            sel = [r for r in fitted_rows if r["radial_sd_nm"] == sd and r["asked"] == kind]
            if not sel:
                continue
            nfail = sum(r.get("b_p_100") in (None, "") for r in sel)
            parts = []
            rates = []
            for hw in BANDS:
                k, n = _rate(sel, f"b_p_{int(hw)}")
                if n:
                    rates.append(k / n)
                parts.append(f"{int(hw)} nm: {X.rate_text(k, n)}")
            crit_s = f"false positives {lo}-{hi}" if kind == "true" else f"false positives <= {FP_HARD_MAX} per band"
            ok_v: Optional[bool] = None
            if rates:
                ok_v = (max(rates) <= hi) if kind == "true" else (max(rates) <= FP_HARD_MAX)
            out.append(X.verdict(f"V3 null (random density) contour '{kind}', radial sd {sd} nm", crit_s,
                                 "; ".join(parts) + f"; contour failed {nfail}/{len(sel)}; median tau_vs_offset(100) "
                                 f"{_med(sel, 'tau_vs_offset_100'):+.3f}; median contour err "
                                 f"{_med(sel, 'err_mean'):.0f} nm", ok_v,
                                 "" if rates else "no accepted contour: not measurable"))
        for used in sorted({r["used"] for r in fitted_rows if r["radial_sd_nm"] == sd and r["asked"] != "true"
                            and r["used"] != "failed"}):
            sel = [r for r in fitted_rows if r["radial_sd_nm"] == sd and r["asked"] != "true" and r["used"] == used]
            k, n = _rate(sel, "b_p_100")
            out.append(X.verdict(f"V3 null by curve USED '{used}', radial sd {sd} nm, band 100",
                                 f"false positives <= {FP_HARD_MAX}", X.rate_text(k, n),
                                 bool(n == 0 or k / n <= FP_HARD_MAX)))
        for relv in ("True", "False"):
            sel = [r for r in fitted_rows if r["radial_sd_nm"] == sd and r["asked"] == "auto"
                   and str(r.get("reliable", "")) == relv]
            if not sel:
                continue
            parts = []
            rates = []
            for hw in BANDS:
                k, n = _rate(sel, f"b_p_{int(hw)}")
                if n:
                    rates.append(k / n)
                parts.append(f"{int(hw)} nm: {X.rate_text(k, n)}")
            out.append(X.verdict(f"V3 null, auto contour, gate reliable={relv}, radial sd {sd} nm",
                                 f"false positives <= {FP_HARD_MAX} per band", "; ".join(parts),
                                 bool(rates) and max(rates) <= FP_HARD_MAX))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P11 V3 (question B) and fitted-contour negatives of B and C")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("true")
    a.add_argument("--case", required=True, choices=["null", "pos_wavy", "pos_ellipse"])
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--start", type=int, default=0)
    a.add_argument("--out", required=True)
    f = sub.add_parser("fitted")
    f.add_argument("--sd", type=float, required=True)
    f.add_argument("--n", type=int, required=True)
    f.add_argument("--start", type=int, default=0)
    f.add_argument("--pattern", default="independent", choices=["independent", "identical"])
    f.add_argument("--kinds", default=",".join(X.CURVE_KINDS))
    f.add_argument("--out", required=True)
    p = sub.add_parser("fitted-pos")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = X.ensure_out(args.out)
    t0 = time.time()
    if args.cmd == "true":
        X.write_csv(os.path.join(out, "v3_true.csv"),
                    [true_axon(args.case, i) for i in range(args.start, args.start + args.n)], append=True)
    elif args.cmd == "fitted":
        rows: List[Dict[str, Any]] = []
        kinds = tuple(k for k in args.kinds.split(",") if k)
        for i in range(args.start, args.start + args.n):
            rows += fitted_axon(args.sd, i, kinds=kinds, pattern=args.pattern)
            if (i - args.start) % 10 == 9:
                print(f"fitted sd {args.sd:g}: {i - args.start + 1} axons, {time.time() - t0:.0f} s", flush=True)
        name = "v3_fitted.csv" if args.pattern == "independent" else "v3_fitted_cpos.csv"
        X.write_csv(os.path.join(out, name), rows, append=True)
    elif args.cmd == "fitted-pos":
        X.write_csv(os.path.join(out, "v3_fitted_pos.csv"),
                    [fitted_pos_axon(i) for i in range(args.start, args.start + args.n)], append=True)
    else:
        def load(name: str) -> List[Dict[str, Any]]:
            pth = os.path.join(out, name)
            return [dict(x) for x in X.read_csv(pth)] if os.path.exists(pth) else []
        tr, fr, pr = load("v3_true.csv"), load("v3_fitted.csv"), load("v3_fitted_pos.csv")
        v = verdicts(tr, fr, pr)
        sizes = {c: sum(r["case"] == c for r in tr) for c in ("null", "pos_wavy", "pos_ellipse")}
        nf = {sd: len({r["axon"] for r in fr if r["radial_sd_nm"] == sd}) for sd in {r["radial_sd_nm"] for r in fr}}
        X.write_verdicts(os.path.join(out, "VERDICT_v3_b.md"), "V3 question B (density vs tortuosity)", v,
                         [f"True-contour axons: {sizes}; fitted-contour axons by radial sd: {nf}; fitted positives: "
                          f"{len(pr)}. 1999 rotations, two-sided, window 2 bins."])
        for x in v:
            print(x["verdict"], x["check"], "|", x["measured"], flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
