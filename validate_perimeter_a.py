"""
V4 of the perimeter density analysis (package P11): question A, the density around its maximum, pooled over
sets of 18 axons (the size of the training set).

    python validate_perimeter_a.py run --cond COND --n N [--start K] --out DIR
    python validate_perimeter_a.py report --out DIR

Each axon: a closed perimeter of 8-25 um (80-250 bins, the real range), ~30 localizations per bin in the central
slice (as review R2's checks), blinking (1 + geometric localizations per molecule), arcs taken from P1's truth
(``true_arc_nm``: question A is tested, not the contour), profile from P4 ``slice_profiles`` (bins centred, origin
at the maximum), pooled with P7 ``pooled_details`` (default "baseline" zone, widened x2 / x3 for robustness,
Holm between the two statistics).

Conditions (COND) and the answer the reading must give:

========== ============================================================== =======================================
cond       density                                                        expected reading
========== ============================================================== =======================================
flat       P1 ``flat``                                                    no structure (the pooled raw profile
                                                                          still peaks at 0: the artefact)
clumps     P1 ``clumps``: big cluster (sd 40 nm) + 20 clumps at random    no structure
g100/150/  P1 ``cluster``: resolved Gaussian cluster, sd 100/150/300 nm,  no structure (review R2, B1)
g300       25 % of the localizations
t100/150/  heavy-tailed cluster (Student t, 3 df, scale 100/150/300 nm),  no structure per review R2; P7 documents
t300       25 % of the localizations (added here; P1 has no t cluster)    that a heavy tail IS a halo by any
                                                                          measurable property
attractor  P1 ``attractor``: big cluster + Gaussian excess sd 600 nm,     attractor (robust)
           amplitude 0.5
halo       the same excess around a RESOLVED cluster (sd 150 nm, 25 %)    attractor (robust): a real halo is still
                                                                          found beyond a wide cluster (review R2)
inhibition P1 ``inhibition``: empty zone of 800 nm on each side           inhibition zone
regular    P1 ``regular``: modes every ~800 nm in phase with the cluster  regular, period within +-1 bin
========== ============================================================== =======================================

Criteria [plan sec. 4.4 P7, sec. 5 V4; review R2 B1]: flat / clumps: false positives of each statistic 0.03-0.07
(Holm p); every null: emitted "attractor" and "regular" <= 0.07; positives: power >= 0.8. Pooled null:
``N_NULL_POOLED`` replicates (a compute choice, reported). Readings that carry "not robust to the exclusion
zone" are counted separately.
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
from tools import mps_perimeter_a as A  # noqa: E402
from tools import mps_perimeter_density as D  # noqa: E402
from tools import mps_perimeter_sim as S  # noqa: E402
from tools.mps_perimeter_types import DensityProfile  # noqa: E402

N_AXONS: int = 18                   # [real data, pilot] the training set
N_NULL_POOLED: int = 499            # [ad hoc, compute] pooled replicates per set (p floor 0.002)
N_NULL_AXON: int = 999              # [ad hoc, compute] per-axon replicates (flat / clumps only)
LOCS_PER_BIN: int = 30              # [ad hoc, = review R2's checks]
RESOLVED_FRAC: float = 0.25         # [ad hoc, = review R2's checks: 1500 of ~5700 localizations]
PERIOD_TOL_NM: float = 100.0        # [plan] period recovered within +-1 bin
NULL_CONDS: Tuple[str, ...] = ("flat", "clumps", "g100", "g150", "g300", "t100", "t150", "t300")
POS_CONDS: Tuple[str, ...] = ("attractor", "halo", "inhibition", "regular")
CONDS: Tuple[str, ...] = NULL_CONDS + POS_CONDS


def axon_profile(cond: str, rng: np.random.Generator) -> DensityProfile:
    """One synthetic central-slice profile of condition ``cond``."""
    L_target = float(rng.uniform(8000.0, 25000.0))
    n_bins = int(round(L_target / 100.0))
    n = LOCS_PER_BIN * n_bins
    kind: Dict[str, Any]
    n_sim, t_scale = n, 0.0
    if cond == "flat":
        kind = {"kind": "flat"}
    elif cond == "clumps":
        kind = {"kind": "clumps"}
    elif cond.startswith("g"):
        kind = {"kind": "cluster", "cluster_sd_nm": float(cond[1:]), "cluster_frac": RESOLVED_FRAC}
    elif cond.startswith("t"):
        kind, n_sim, t_scale = {"kind": "flat"}, int(round((1 - RESOLVED_FRAC) * n)), float(cond[1:])
    elif cond == "attractor":
        kind = {"kind": "attractor"}
    elif cond == "halo":
        kind = {"kind": "attractor", "cluster_sd_nm": 150.0, "cluster_frac": RESOLVED_FRAC}
    elif cond == "inhibition":
        kind = {"kind": "inhibition"}
    elif cond == "regular":
        kind = {"kind": "regular"}
    else:
        raise ValueError(cond)
    sim = S.simulate_perimeter("circle", kind, n_locs=n_sim, radial_sd_nm=0.0, slices={"fracs": (0.0, 1.0, 0.0)},
                               lpz_nm=0.0, background_frac=0.0, interior_frac=0.0, blink=True,
                               seed=int(rng.integers(0, 2 ** 31)),
                               options={"radius_nm": L_target / (2 * math.pi), "precision_nm": 0.0})
    L = float(sim.true_path.length_nm)
    s = np.asarray(sim.true_arc_nm, dtype=np.float64)
    if t_scale > 0:
        centre = float(rng.uniform(0.0, L))
        s = np.concatenate([s, np.mod(centre + t_scale * rng.standard_t(3, size=n - n_sim), L)])
    sp = D.slice_profiles(np.mod(s, L), np.zeros(s.size), np.zeros(s.size, dtype=np.int8), L, half_width_nm=100.0)
    return sp.by_slice[0]


FIT_CONDS: Tuple[str, ...] = ("flat_fit", "attractor_fit")
FIT_RADIAL_SD_NM: float = 40.0      # [ad hoc] a membrane band the contour can follow (P3 v2: real median ~73 nm)


def axon_profile_fit(cond: str, rng: np.random.Generator) -> Tuple[Optional[DensityProfile], bool, str]:
    """The same axon on a wavy shape with a radial spread, read on P3's AUTOMATIC contour (defaults and
    fallbacks). Returns (profile or None when the contour fails, gate reliable, curve kind)."""
    L_target = float(rng.uniform(8000.0, 25000.0))
    n = LOCS_PER_BIN * int(round(L_target / 100.0))
    kind = {"kind": "flat"} if cond == "flat_fit" else {"kind": "attractor"}
    opt = {"radius_nm": L_target / (2 * math.pi),
           "wave_phases": tuple(float(v) for v in rng.uniform(0.0, 2 * math.pi, 4)),
           "rotation_deg": float(rng.uniform(0.0, 360.0))}
    sim = S.simulate_perimeter("wavy", kind, n_locs=n, radial_sd_nm=FIT_RADIAL_SD_NM,
                               slices={"fracs": (0.0, 1.0, 0.0)}, lpz_nm=0.0, background_frac=0.0,
                               interior_frac=0.0, blink=True, seed=int(rng.integers(0, 2 ** 31)), options=opt)
    det = X.fit_curve(sim.x_nm, sim.y_nm, "auto")
    a = det.assessment
    rel = bool(a.reliable) if a is not None else False
    knd = a.kind if a is not None else "none"
    c = det.contour
    if c.path is None:
        return None, False, knd
    s, r = D.project(c.path, sim.x_nm, sim.y_nm)
    sp = D.slice_profiles(s, r, np.zeros(s.size, dtype=np.int8), float(c.path.length_nm), half_width_nm=100.0)
    return sp.by_slice[0], rel, knd


def one_set_fit(cond: str, k: int) -> Dict[str, Any]:
    seed = 2_000_000 * (FIT_CONDS.index(cond) + 1) + k
    rng = np.random.default_rng(seed)
    got = [axon_profile_fit(cond, rng) for _ in range(N_AXONS)]
    acc = [p for p, _r, _k in got if p is not None]
    rel = [p for p, r, _k in got if p is not None and r]
    row: Dict[str, Any] = {"cond": cond, "set": k, "seed": seed, "n_accepted": len(acc), "n_reliable": len(rel),
                           "kinds": ";".join(kk for _p, _r, kk in got)}
    for tag, profs in (("acc", acc), ("rel", rel)):
        if len(profs) < 2:
            row[f"{tag}_label"] = "too few axons"
            continue
        pdt = A.pooled_details(profs, n_null=N_NULL_POOLED, seed=k)
        toks = _tokens(pdt.label)
        row.update({f"{tag}_label": pdt.label, f"{tag}_attractor_robust": A.READING_ATTRACTOR in toks,
                    f"{tag}_regular_robust": A.READING_REGULAR in toks,
                    f"{tag}_inhibition": A.READING_INHIBITION in toks,
                    f"{tag}_contrast_p_holm": pdt.contrast_p_holm, f"{tag}_period_p_holm": pdt.period_p_holm})
    return row


def verdicts_fit(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cond in FIT_CONDS:
        sel = [r for r in rows if r["cond"] == cond]
        if not sel:
            continue
        na = np.mean([X.fnum(r["n_accepted"]) for r in sel])
        nr = np.mean([X.fnum(r["n_reliable"]) for r in sel])
        for tag, lab in (("acc", "all accepted"), ("rel", "only gate-reliable")):
            ok = [r for r in sel if str(r.get(f"{tag}_label")) not in ("", "too few axons")]
            att = sum(_true(r.get(f"{tag}_attractor_robust")) for r in ok)
            reg = sum(_true(r.get(f"{tag}_regular_robust")) for r in ok)
            inh = sum(_true(r.get(f"{tag}_inhibition")) for r in ok)
            if cond == "flat_fit":
                out.append(X.verdict(f"V4 null on the AUTO contour ({lab}), flat", "'attractor' / 'regular' / "
                                     "'inhibition' <= 0.07 each", f"attractor {X.rate_text(att, len(ok))}; regular "
                                     f"{X.rate_text(reg, len(ok))}; inhibition {X.rate_text(inh, len(ok))}; mean "
                                     f"axons per set: accepted {na:.1f}, reliable {nr:.1f}",
                                     bool(ok) and max(att, reg, inh) / len(ok) <= X.FPR_BAND[1]))
            else:
                out.append(X.verdict(f"V4 positive on the AUTO contour ({lab}), attractor", "power >= 0.8",
                                     f"robust attractor {X.rate_text(att, len(ok))}; mean axons per set: accepted "
                                     f"{na:.1f}, reliable {nr:.1f}", bool(ok) and att / len(ok) >= X.POWER_MIN))
    return out


def _tokens(label: str) -> List[str]:
    return [t.strip() for t in label.split(A.READING_JOIN)] if label else []


def one_set(cond: str, k: int, n_axons: int = N_AXONS) -> Dict[str, Any]:
    seed = 1_000_000 * (CONDS.index(cond) + 1) + k + (0 if n_axons == N_AXONS else 100_000 * n_axons)
    rng = np.random.default_rng(seed)
    profs = [axon_profile(cond, rng) for _ in range(n_axons)]
    pdt = A.pooled_details(profs, n_null=N_NULL_POOLED, seed=k)
    toks = _tokens(pdt.label)
    prof = pdt.result.profile
    lag = prof.lag_nm
    far = lag >= 2000.0
    row: Dict[str, Any] = {
        "cond": cond, "set": k, "seed": seed, "label": pdt.label, "n_axons": pdt.result.n_axons,
        "attractor_robust": A.READING_ATTRACTOR in toks, "attractor_not_robust":
            (A.READING_ATTRACTOR + A.READING_NOT_ROBUST) in toks,
        "inhibition": A.READING_INHIBITION in toks, "regular_robust": A.READING_REGULAR in toks,
        "regular_not_robust": (A.READING_REGULAR + A.READING_NOT_ROBUST) in toks,
        "contrast_p_raw": pdt.result.contrast.p, "period_p_raw": pdt.result.period.p,
        "contrast_p_holm": pdt.contrast_p_holm, "period_p_holm": pdt.period_p_holm,
        "contrast_minus_null": pdt.contrast_minus_null, "contrast_z": pdt.result.contrast.z,
        "period_nm": pdt.period_nm, "period_at_edge": pdt.period_at_edge,
        "zone_nm_median": pdt.zone_nm_median, "zone_nm_max": pdt.zone_nm_max,
        # the alignment artefact: the raw pooled profile at lag 0 / 1 bin against far lags and the null band
        "raw_lag0": float(prof.mean_relative[0]) if lag.size else float("nan"),
        "raw_lag1": float(prof.mean_relative[1]) if lag.size > 1 else float("nan"),
        "raw_far_median": float(np.nanmedian(prof.mean_relative[far])) if far.any() else float("nan"),
        "lag1_inside_null_band": bool(prof.null_lo[1] <= prof.mean_relative[1] <= prof.null_hi[1])
        if lag.size > 1 and np.isfinite(prof.null_lo[1]) else None,
    }
    if cond in ("flat", "clumps"):
        pc = pp = nn = 0
        for j, p in enumerate(profs):
            d = A.axon_a_details(p, n_null=N_NULL_AXON, seed=j)
            if d.usable:
                nn += 1
                pc += int(d.contrast.p < X.ALPHA)
                pp += int(d.period.p < X.ALPHA)
        row.update(axon_n=nn, axon_contrast_sig=pc, axon_period_sig=pp)
    return row


def _true(v: Any) -> bool:
    return str(v) in ("True", "true", "1")


def verdicts(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    lo, hi = X.FPR_BAND
    for cond in CONDS:
        sel = [r for r in rows if r["cond"] == cond]
        if not sel:
            continue
        n = len(sel)
        att = sum(_true(r["attractor_robust"]) for r in sel)
        att_nr = sum(_true(r["attractor_not_robust"]) for r in sel)
        reg = sum(_true(r["regular_robust"]) for r in sel)
        reg_nr = sum(_true(r["regular_not_robust"]) for r in sel)
        inh = sum(_true(r["inhibition"]) for r in sel)
        not_rob = f"not robust: attractor {att_nr}, regular {reg_nr}"
        if cond in NULL_CONDS:
            out.append(X.verdict(f"V4 null '{cond}': emitted 'attractor'", "<= 0.07 [review R2]",
                                 X.rate_text(att, n) + f"; {not_rob}", att / n <= hi))
            out.append(X.verdict(f"V4 null '{cond}': emitted 'regular'", "<= 0.07 [review R2]",
                                 X.rate_text(reg, n), reg / n <= hi))
            out.append(X.verdict(f"V4 null '{cond}': 'inhibition zone'", "<= 0.07", X.rate_text(inh, n),
                                 inh / n <= hi))
            if cond in ("flat", "clumps"):
                for stat in ("contrast", "period"):
                    k = sum(X.fnum(r[f"{stat}_p_holm"]) <= X.ALPHA for r in sel)
                    out.append(X.verdict(f"V4 null '{cond}': pooled {stat} (Holm p <= 0.05)",
                                         f"false positives {lo}-{hi} [plan]", X.rate_text(k, n),
                                         lo <= k / n <= hi,
                                         "a rate under 0.03 is conservative, reported as FAIL of the band"))
                    ka = sum(int(X.fnum(r[f"axon_{stat}_sig"])) for r in sel)
                    na = sum(int(X.fnum(r["axon_n"])) for r in sel)
                    out.append(X.verdict(f"V4 null '{cond}': per-axon {stat} p < 0.05", "informative (descriptive p)",
                                         X.rate_text(ka, na), None))
                l0 = np.nanmean([X.fnum(r["raw_lag0"]) for r in sel])
                l1 = np.nanmean([X.fnum(r["raw_lag1"]) for r in sel])
                lf = np.nanmean([X.fnum(r["raw_far_median"]) for r in sel])
                inside = sum(_true(r["lag1_inside_null_band"]) for r in sel)
                out.append(X.verdict(f"V4 alignment artefact '{cond}'", "documented: raw pooled peak at 0 with no "
                                     "structure, and inside its own null band",
                                     f"mean raw profile lag 0 = {l0:.2f}, lag 100 nm = {l1:.2f}, far = {lf:.2f}; "
                                     f"lag 100 nm inside the null band in {inside}/{n}", bool(l0 > lf)))
        else:
            if cond in ("attractor", "halo"):
                k, crit = att, "robust 'attractor'"
            elif cond == "inhibition":
                k, crit = inh, "'inhibition zone'"
            else:
                k = sum(_true(r["regular_robust"]) and abs(X.fnum(r["period_nm"]) - 800.0) <= PERIOD_TOL_NM
                        for r in sel)
                crit = "robust 'regular' with period 800 +- 100 nm"
            out.append(X.verdict(f"V4 positive '{cond}'", f"power >= 0.8 ({crit})",
                                 X.rate_text(k, n) + f"; labels: {_label_counts(sel)}", k / n >= X.POWER_MIN))
    allr = len(rows)
    nr = sum(_true(r["attractor_not_robust"]) or _true(r["regular_not_robust"]) for r in rows)
    out.append(X.verdict("V4 readings carrying 'not robust to the exclusion zone'", "informative",
                         f"{nr}/{allr} sets over all conditions", None))
    return out


def _label_counts(sel: Sequence[Dict[str, Any]]) -> str:
    c: Dict[str, int] = {}
    for r in sel:
        c[str(r["label"])] = c.get(str(r["label"]), 0) + 1
    return "; ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])[:4])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P11 V4 (question A)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--cond", required=True, choices=CONDS + FIT_CONDS)
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--start", type=int, default=0)
    a.add_argument("--n-axons", type=int, default=N_AXONS)
    a.add_argument("--out", required=True)
    b = sub.add_parser("report")
    b.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = X.ensure_out(args.out)
    t0 = time.time()
    if args.cmd == "run":
        rows = []
        fit = args.cond in FIT_CONDS
        for k in range(args.start, args.start + args.n):
            rows.append(one_set_fit(args.cond, k) if fit else one_set(args.cond, k, args.n_axons))
        name = "v4_a_fit.csv" if fit else ("v4_a.csv" if args.n_axons == N_AXONS else f"v4_a_n{args.n_axons}.csv")
        X.write_csv(os.path.join(out, name), rows, append=True)
        print(f"{args.cond}: {args.n} sets", flush=True)
    else:
        pa = os.path.join(out, "v4_a.csv")
        rows_all = [dict(r) for r in X.read_csv(pa)] if os.path.exists(pa) else []
        v = verdicts(rows_all)
        pf = os.path.join(out, "v4_a_fit.csv")
        if os.path.exists(pf):
            v += verdicts_fit([dict(r) for r in X.read_csv(pf)])
        for f in sorted(os.listdir(out)):
            if f.startswith("v4_a_n") and f.endswith(".csv"):
                nax = f[len("v4_a_n"):-4]
                for row in verdicts([dict(r) for r in X.read_csv(os.path.join(out, f))]):
                    row["check"] = f"[{nax} axons per set] " + str(row["check"])
                    v.append(row)
        sizes = {c: sum(r["cond"] == c for r in rows_all) for c in CONDS}
        X.write_verdicts(os.path.join(out, "VERDICT_v4_a.md"), "V4 question A (pooled, 18 axons per set)", v,
                         [f"Sets per condition: {sizes}. {N_AXONS} axons per set, 8-25 um, {LOCS_PER_BIN} locs per "
                          f"bin, blinking; pooled null {N_NULL_POOLED} replicates; per-axon null {N_NULL_AXON}; "
                          "zone rule 'baseline' (default), robustness x2 / x3."])
        for x in v:
            print(x["verdict"], x["check"], "|", x["measured"], flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
