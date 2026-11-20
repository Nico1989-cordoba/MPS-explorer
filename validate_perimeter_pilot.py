"""
V5 of the perimeter density analysis (package P11): the 18 axons of the training set (the pilot dataset).

    python validate_perimeter_pilot.py list
    python validate_perimeter_pilot.py one --file PATH --variant auto|knot600|polar --out DIR   (one file per process)
    python validate_perimeter_pilot.py pool --out DIR

EXPLORATORY. Training set, ONE experiment, ONE animal of unknown label (decision Q7): the axons get
the provisional label of ``mps_perimeter_pool.PROVISIONAL_ANIMAL``; n_animals = 1 rules out any comparison
between animals, and nothing here is a test between groups. Parameters chosen while looking at these axons are
to be frozen with a date before new data (statistical plan).

Real data are READ ONLY, one file per process, never copied (the folder comes from the MPS_DATA_DIR environment variable): ``one`` loads one file (``mps_io``), runs the whole
orchestration (``analyze_perimeter``, default parameters: z0 = peak, contiguous 170 nm slices, band +-100 nm with
50 / 150 nm sensitivity, 1999 rotations) with P3's automatic contour (``auto``) or a fallback FORCED (``knot600``,
``polar``: review R1), and writes outside the repository: one CSV row of readouts, the program's own per-axon
row (``mps_perimeter_pool.perimeter_row``) and a pickle of the result (derived numbers only: profiles, contour,
readouts; no localization coordinates beyond the per-localization slice label). ``pool`` builds the batch
tables from the pickles (the same functions ``batch_perimeter.py`` uses) and the summaries:

* time per axon with the real automatic contour (criterion < 30 s [plan sec. 4.4 P10]);
* curve used per axon (knot 400 / knot 600 / polar) and whether A, B, C change with the curve;
* the three slice percentages (lateral / central);
* peak - focus per axon against both foci, and its summary by ROI and overall;
* P2's KDE peak against ``fit_z_periodicity`` (the peak the rest of the programme exports);
* band sensitivity 50 / 150 nm;
* A, B, C stratified by curve type and by contour quality (fraction within +-100 nm, radial robust sd) -- review R1;
* the pooled question A over the 18 (exploratory, within-sample).
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import validate_perimeter_cross as X  # noqa: E402
from tools import mps_perimeter_pool as pool  # noqa: E402
from tools import mps_perimeter_types as pdt  # noqa: E402

LABEL = "EXPLORATORY: training set, one experiment, one animal (provisional label), no between-animal test"
TIME_MAX_S: float = 30.0            # [plan sec. 4.4 P10] one axon end to end
VARIANTS = ("auto", "knot600", "polar", "auto_trim")
PAIRS = (((-1, 0), "m1_0"), ((0, 1), "0_p1"), ((-1, 1), "m1_p1"))


def real_files() -> List[str]:
    from validate_auto_contour import real_files as rf
    return rf()


def real_tag(path: str) -> str:
    from validate_auto_contour import real_tag as rt
    return rt(path)


def _timed_contour(variant: str, box: Dict[str, Any]) -> Any:
    def fn(x: Any, y: Any, **_kw: Any) -> Any:
        t0 = time.time()
        det = X.fit_curve(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), variant)
        box["contour_s"] = time.time() - t0
        box["primary_reason"] = det.candidates[0].reason if det.candidates else ""
        box["n_candidates"] = len(det.candidates)
        box["details"] = det
        return det.contour
    return fn


def gate_columns(det: Any, x_fit: Any, y_fit: Any, label_fit: Any) -> Dict[str, Any]:
    """P3's quality gate as the analysis will read it after P10b: ``assessment_row`` of the curve the analysis
    used, plus the slice misfit (D7) measured on the same curve with the analysis's slice labels. Two
    reliabilities: ``reliable_d7`` (the misfit excludes, P3 v2 default) and ``reliable_d12`` (decision D12: the
    misfit only informs)."""
    from tools import mps_auto_contour as AC
    row: Dict[str, Any] = dict(AC.assessment_row(det))
    rel_no_slices = False
    slice_low = False
    if det.contour.path is not None:
        # measured on EVERY localization of the three slices (with the trim on, the fit saw fewer)
        q = AC.contour_quality(det.contour.path, x_fit, y_fit, slice_label=label_fit)
        sm = q.slices
        other = [r for r in q.reasons if sm is None or r != sm.reason]
        rel_no_slices = not other
        row.update(contour_frac100=q.frac_within[100.0], contour_robust_sd_nm=q.robust_sd_nm,
                   contour_persistent_offset_nm=q.persistent_offset_nm, contour_off_band_fraction=q.off_band_fraction,
                   contour_quality_reasons=" | ".join(q.reasons))
        if sm is not None:
            slice_low = bool(sm.low)
            row.update(contour_slice_rms_minus_nm=sm.rms_displacement_nm[0],
                       contour_slice_rms_plus_nm=sm.rms_displacement_nm[1],
                       contour_slice_radius_change_nm=sm.radius_change_per_slice_nm,
                       contour_slice_frac100_central=sm.frac100_central, contour_slice_frac100_all=sm.frac100_all,
                       contour_slice_reason=sm.reason)
    row["reliable_d12"] = rel_no_slices
    row["reliable_d7"] = rel_no_slices and not slice_low
    row["slice_misfit_low"] = slice_low
    return row


def pdt_outside() -> int:
    return int(pdt.OUTSIDE_LABEL)


def _sig(t: Optional[pdt.TestReadout]) -> Optional[bool]:
    if t is None or not math.isfinite(t.p):
        return None
    return bool(t.p < X.ALPHA)


def readout_row(res: pdt.PerimeterResult, z_all: Any, box: Dict[str, Any], variant: str, tag: str,
                seconds: float, load_s: float) -> Dict[str, Any]:
    from tools.mps_periodicity import fit_z_periodicity
    row: Dict[str, Any] = {"label": LABEL, "axon": tag, "trace_label": res.trace.label, "variant": variant,
                           "status": res.status.value, "failure": res.failure, "seconds_total": round(seconds, 2),
                           "seconds_load": round(load_s, 2), "seconds_contour": round(box.get("contour_s", float("nan")), 2),
                           "primary_reason": box.get("primary_reason", "")}
    z0 = res.z0
    try:
        zp = float(fit_z_periodicity(np.asarray(z_all, dtype=np.float64)).main_peak_nm)
    except Exception as exc:  # noqa: BLE001 - reported
        zp = float("nan")
        row["fit_z_error"] = str(exc)
    row.update(z_peak_kde_nm=z0.peak_nm, z_peak_fit_z_periodicity_nm=zp,
               kde_minus_fit_nm=None if z0.peak_nm is None or not math.isfinite(zp) else z0.peak_nm - zp,
               focus_nominal_nm=z0.focus_nominal_nm, focus_empirical_nm=z0.focus_empirical_nm,
               peak_minus_focus_nominal_nm=z0.peak_minus_focus_nominal_nm,
               peak_minus_focus_empirical_nm=z0.peak_minus_focus_empirical_nm, z0_nm=z0.z0_nm)
    if res.slices is not None:
        sl = res.slices
        row.update(n_m1=sl.n_locs[0], n_0=sl.n_locs[1], n_p1=sl.n_locs[2], pct_m1=sl.pct_of_central[0],
                   pct_p1=sl.pct_of_central[1], cov_m1=sl.z_coverage[0], cov_p1=sl.z_coverage[2],
                   frac_in_three=sl.frac_in_three)
    c = res.contour
    if c is not None:
        row.update(curve_used=X.curve_type(c), contour_length_um=c.length_nm / 1000.0 if c.path is not None else None,
                   polar_distance_nm=c.polar_distance_nm, star_shaped=c.star_shaped)
    if res.band is not None:
        b = res.band
        row.update(frac_within_50=b.frac_within[50.0], frac_within_100=b.frac_within[100.0],
                   frac_within_150=b.frac_within[150.0], radial_sd_robust_nm=b.robust_sd_nm, radial_fwhm_nm=b.fwhm_nm)
    if res.profiles is not None:
        p0 = res.profiles.by_slice[0]
        row.update(n_bins=p0.n_bins, bin_eff_nm=p0.bin_nm_effective, max_tie=p0.max_tie)
    d = res.diagnostics
    if d is not None:
        row.update(unsupported_fraction=d.unsupported_fraction, radial_shift_m1_nm=d.radial_shift_nm[0],
                   radial_shift_p1_nm=d.radial_shift_nm[1], a_exclusion_nm=d.a_exclusion_nm, a_usable=d.a_usable,
                   c_reading=d.c_reading, leak_expected_r=d.leak_expected_r, lpz_nm=d.lpz_nm,
                   band_flag=d.band_sensitivity_flag)
    for name, t in res.a.items():
        row.update({f"a_{name}": t.value, f"a_{name}_z": t.z, f"a_{name}_p": t.p})
    if res.b is not None:
        row.update(b_spearman=res.b.test.value, b_z=res.b.test.z, b_p=res.b.test.p, b_tau_min=res.b.tau_min,
                   b_resolved=res.b.resolved, b_tau_vs_offset=res.b.tau_vs_offset)
    for pair, name in PAIRS:
        t = res.c.get(pair)
        if t is not None:
            row.update({f"c_r_{name}": t.value, f"c_z_{name}": t.z, f"c_p_{name}": t.p})
    for g, gs in res.guard_sensitivity.items():
        for pair, name in PAIRS:
            t = gs.c.get(pair)
            if t is not None:
                row[f"c_r_{name}_guard{g:g}"] = t.value
                row[f"c_p_{name}_guard{g:g}"] = t.p
    for hw, s in res.sensitivity.items():
        tag_hw = f"{hw:g}"
        row[f"sens{tag_hw}_a_contrast_p"] = s.a["contrast"].p if "contrast" in s.a else None
        row[f"sens{tag_hw}_a_period_p"] = s.a["period"].p if "period" in s.a else None
        row[f"sens{tag_hw}_b_p"] = None if s.b is None else s.b.test.p
        row[f"sens{tag_hw}_b_r"] = None if s.b is None else s.b.test.value
        cc = s.c.get((-1, 1))
        row[f"sens{tag_hw}_c_r_m1_p1"] = None if cc is None else cc.value
        row[f"sens{tag_hw}_c_p_m1_p1"] = None if cc is None else cc.p
    row["warnings"] = " | ".join(res.warnings)[:800]
    return row


def run_one(path: str, variant: str, out: str) -> Dict[str, Any]:
    from dataclasses import replace
    from tools.mps_identity import propose
    from tools.mps_io import load_localizations
    from tools.mps_perimeter_analysis import analyze_perimeter
    tag = real_tag(path)
    box: Dict[str, Any] = {}
    t0 = time.time()
    loc = load_localizations(path)
    load_s = time.time() - t0
    ident = replace(propose(path).identity, animal=pool.PROVISIONAL_ANIMAL)
    params = pdt.PerimeterParams()
    curve = variant
    if variant == "auto_trim":      # P14 / D8 trim of interior localizations before the fit (off by default)
        params = replace(params, contour_trim_interior_nm=250.0)
        curve = "auto"
    res = analyze_perimeter(loc.x_nm, loc.y_nm, loc.z_nm, columns=loc.columns, info=loc.info, identity=ident,
                            source_path=path, params=params, contour_fn=_timed_contour(curve, box))
    seconds = time.time() - t0
    row = readout_row(res, loc.z_nm, box, variant, tag, seconds, load_s)
    if "details" in box and res.slices is not None and res.slices.label.size == loc.x_nm.size:
        lab = np.asarray(res.slices.label)
        in3 = lab != pdt_outside()
        row.update(gate_columns(box["details"], np.asarray(loc.x_nm)[in3], np.asarray(loc.y_nm)[in3], lab[in3]))
    else:
        row.update(reliable_d7=False, reliable_d12=False, contour_kind="none")
    del loc
    safe = tag.replace("/", "_")
    X.write_csv(os.path.join(out, f"pilot_readouts_{variant}.csv"), [row], append=True)
    X.write_csv(os.path.join(out, f"pilot_rows_{variant}.csv"), [pool.perimeter_row(res)], append=True)
    os.makedirs(os.path.join(out, "pilot_pickles"), exist_ok=True)
    slim = replace(res, slices=None if res.slices is None else replace(res.slices, label=np.zeros(0, dtype=np.int8)))
    with open(os.path.join(out, "pilot_pickles", f"{variant}_{safe}.pkl"), "wb") as fh:
        pickle.dump(slim, fh)
    print({k: row.get(k) for k in ("axon", "status", "curve_used", "seconds_total", "seconds_contour", "b_p",
                                    "c_p_m1_p1", "a_contrast_p")}, flush=True)
    return row


# ----------------------------------------------------------------------------------------------------
# Pool
# ----------------------------------------------------------------------------------------------------
def _f(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([X.fnum(r.get(key)) for r in rows], dtype=float)


def _summ(v: np.ndarray, unit: str = "") -> str:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n = 0"
    return (f"median {np.median(v):.1f}{unit} (range {v.min():.1f} to {v.max():.1f}), mean {v.mean():.1f} +- "
            f"{v.std(ddof=1) if v.size > 1 else 0.0:.1f}, n = {v.size}")


def _decisions(r: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    def s(key: str) -> Optional[bool]:
        p = X.fnum(r.get(key))
        return None if not math.isfinite(p) else bool(p < X.ALPHA)
    return {"A contrast": s("a_contrast_p"), "A period": s("a_period_p"), "B": s("b_p"), "C(-1,+1)": s("c_p_m1_p1")}


def pool_report(out: str) -> List[Dict[str, Any]]:
    from batch_perimeter import TABLES
    from tools import mps_perimeter_a as A
    lines = [f"**{LABEL}**", ""]
    v: List[Dict[str, Any]] = []
    rows = {var: [dict(r) for r in X.read_csv(os.path.join(out, f"pilot_readouts_{var}.csv"))]
            for var in VARIANTS if os.path.exists(os.path.join(out, f"pilot_readouts_{var}.csv"))}
    main = rows.get("auto", [])
    n = len(main)
    st: Dict[str, int] = {}
    for r in main:
        st[r["status"]] = st.get(r["status"], 0) + 1
    v.append(X.verdict("V5 runs on the 18 axons", "18 files, no exception (no 'internal error' / 'invalid input')",
                       f"{n} files; status {st}", n == 18 and not any(s in st for s in ("internal error", "invalid input"))))
    # batch tables from the pickles (the functions batch_perimeter uses)
    results: List[pdt.PerimeterResult] = []
    pk = os.path.join(out, "pilot_pickles")
    for f in sorted(os.listdir(pk)) if os.path.isdir(pk) else []:
        if f.startswith("auto_") and not f.startswith("auto_trim_"):   # the main variant only
            with open(os.path.join(pk, f), "rb") as fh:
                results.append(pickle.load(fh))
    results.sort(key=lambda r: r.axon_id)
    tables: Dict[str, List[Dict[str, Any]]] = {}
    if results:
        prow = pool.perimeter_rows(results)
        pooled = pool.pooled_a(results)
        tables = {"axons": prow, "profiles": pool.profile_rows(results), "pooled": pool.pooled_rows(pooled),
                  "by_animal": pool.summarise_by_animal(prow), "contour": pool.contour_rows(results),
                  "sensitivity": pool.sensitivity_rows(results), "peak_focus": pool.summarise_focus_by_roi(prow)}
        for name, t in tables.items():
            X.write_csv(os.path.join(out, f"pilot_table_{name}.csv"), t)
        cols = [set(r.keys()) for r in prow]
        same = all(c == cols[0] for c in cols)
        union = set().union(*cols) if cols else set()
        missing = sorted({k for c in cols for k in union - c})
        empty = [k for k, t in tables.items() if not t]
        v.append(X.verdict("V5 tables complete", "every batch table written and non-empty; one column set for all "
                           "axon rows", f"{len(tables)} of {len(TABLES)} batch tables built (genotypes/log are batch "
                           f"bookkeeping); empty: {empty}; same columns in all rows: {same}; keys missing in some rows "
                           f"({len(missing)}): {', '.join(missing[:12])}", same and not empty))
        for g, pr in pooled.items():
            det = A.pooled_details([r.profiles.by_slice[0] for r in results if r.profiles is not None],
                                   n_null=1999, seed=0,
                                   animals=[pool.PROVISIONAL_ANIMAL] * sum(r.profiles is not None for r in results))
            lines.append(f"Pooled A (group {g}): `{pr.reading}`")
            lines.append(f"Zone sensitivity: " + "; ".join(
                f"x{z.widen:g}: contrast-null {z.contrast_minus_null:+.3f} p {z.contrast.p:.3g}, period p "
                f"{z.period.p:.3g} ({z.period_nm:.0f} nm)" for z in det.sensitivity))
            break
    ok = [r for r in main if r["status"] == "ok"]
    t = _f(main, "seconds_total")
    tc = _f(main, "seconds_contour")
    v.append(X.verdict("V5 time per axon (real automatic contour)", f"< {TIME_MAX_S:g} s each [plan P10]",
                       f"total {_summ(t, ' s')}; contour {_summ(tc, ' s')}; over 30 s: "
                       f"{int(np.sum(t >= TIME_MAX_S))}", bool(np.all(t[np.isfinite(t)] < TIME_MAX_S))))
    used: Dict[str, int] = {}
    for r in main:
        used[str(r.get("curve_used"))] = used.get(str(r.get("curve_used")), 0) + 1
    v.append(X.verdict("V5 curve used (auto)", "informative", str(used), None))
    for key, lab in (("pct_m1", "slice -1 / central (%)"), ("pct_p1", "slice +1 / central (%)")):
        v.append(X.verdict(f"V5 {lab}", "informative", _summ(_f(main, key), " %"), None))
    for key, lab in (("peak_minus_focus_nominal_nm", "peak - nominal focus"),
                     ("peak_minus_focus_empirical_nm", "peak - empirical focus")):
        vals = _f(main, key)
        txt = f"signed {_summ(vals, ' nm')}; |.| {_summ(np.abs(vals), ' nm')}"
        for roi in ("R1", "R2"):
            sel = [r for r in main if str(r["axon"]).startswith(roi + "/")]
            txt += f"; {roi}: {_summ(_f(sel, key), ' nm')}"
        v.append(X.verdict(f"V5 {lab}", "informative (axons of one ROI share the focus)", txt, None))
    d = _f(main, "kde_minus_fit_nm")
    v.append(X.verdict("V5 P2 KDE peak vs fit_z_periodicity", "informative; the plan expected the same peak",
                       f"KDE - fit: {_summ(d, ' nm')}; |.| {_summ(np.abs(d), ' nm')}", None))
    # A, B, C per axon
    for name, key in (("A contrast", "a_contrast_p"), ("A period", "a_period_p"), ("B", "b_p"),
                      ("C(-1,0)", "c_p_m1_0"), ("C(0,+1)", "c_p_0_p1"), ("C(-1,+1)", "c_p_m1_p1"),
                      ("C(-1,0) guard 42.5", "c_p_m1_0_guard42.5"), ("C(-1,+1) guard 42.5", "c_p_m1_p1_guard42.5")):
        p = _f(ok, key)
        v.append(X.verdict(f"V5 {name}: axons with p < 0.05 (descriptive)", "informative",
                           f"{int(np.sum(p < X.ALPHA))}/{int(np.isfinite(p).sum())}", None))
    for key in ("c_r_m1_0", "c_r_0_p1", "c_r_m1_p1", "b_spearman", "b_tau_vs_offset", "a_contrast_z", "b_z",
                "c_z_m1_p1"):
        v.append(X.verdict(f"V5 distribution of {key}", "informative", _summ(_f(ok, key)), None))
    readings: Dict[str, int] = {}
    for r in ok:
        readings[str(r.get("c_reading"))] = readings.get(str(r.get("c_reading")), 0) + 1
    v.append(X.verdict("V5 C reading per axon", "informative", str(readings), None))
    flags: Dict[str, int] = {}
    for r in ok:
        f = str(r.get("band_flag"))
        f = "consistent" if f == "consistent" else "differs"
        flags[f] = flags.get(f, 0) + 1
    v.append(X.verdict("V5 band sensitivity 50 / 150 nm", "informative (axons whose decisions change)",
                       f"{flags}; " + "; ".join(sorted({str(r.get('band_flag')) for r in ok if r.get('band_flag') != 'consistent'}))[:600],
                       None))
    # variants: does the curve change A, B, C?
    for var in ("knot600", "polar", "auto_trim"):
        if var not in rows:
            continue
        v.append(X.verdict(f"V5 variant '{var}': contours", "informative",
                           f"accepted {sum(r['status'] == 'ok' for r in rows[var])}/{len(rows[var])}; reliable d7 "
                           f"{sum(str(r.get('reliable_d7')) == 'True' for r in rows[var])}, d12 "
                           f"{sum(str(r.get('reliable_d12')) == 'True' for r in rows[var])}: "
                           + ", ".join(str(r['axon']) for r in rows[var] if str(r.get('reliable_d12')) == 'True'),
                           None))
        other = {r["axon"]: r for r in rows[var]}
        changed: Dict[str, int] = {}
        both = 0
        for r in ok:
            o = other.get(r["axon"])
            if o is None or o["status"] != "ok":
                continue
            both += 1
            da, db = _decisions(r), _decisions(o)
            for k in da:
                if da[k] != db[k]:
                    changed[k] = changed.get(k, 0) + 1
        fails = sum(1 for r in rows[var] if r["status"] != "ok")
        v.append(X.verdict(f"V5 auto vs forced '{var}'", "informative: decisions at 0.05 that change",
                           f"{both} axons ok in both ({fails} failed with '{var}'); changed: {changed or 'none'}", None))
    # stratification by curve type and quality (review R1)
    for used_t in sorted(used):
        sel = [r for r in ok if str(r.get("curve_used")) == used_t]
        if not sel:
            continue
        txt = "; ".join(f"{k} {int(np.sum(_f(sel, key) < X.ALPHA))}/{len(sel)}" for k, key in
                        (("A contrast", "a_contrast_p"), ("B", "b_p"), ("C(-1,+1)", "c_p_m1_p1")))
        txt += f"; frac +-100 {_summ(_f(sel, 'frac_within_100'))}; radial sd {_summ(_f(sel, 'radial_sd_robust_nm'))}"
        v.append(X.verdict(f"V5 stratum curve '{used_t}'", "informative", txt, None))
    q = _f(ok, "frac_within_100")
    if np.isfinite(q).sum() >= 4:
        med = float(np.nanmedian(q))
        for name, sel in (("low", [r for r in ok if X.fnum(r.get("frac_within_100")) < med]),
                          ("high", [r for r in ok if X.fnum(r.get("frac_within_100")) >= med])):
            txt = "; ".join(f"{k} {int(np.sum(_f(sel, key) < X.ALPHA))}/{len(sel)}" for k, key in
                            (("A contrast", "a_contrast_p"), ("B", "b_p"), ("C(-1,+1)", "c_p_m1_p1")))
            txt += f"; median C r(-1,+1) {np.nanmedian(_f(sel, 'c_r_m1_p1')):+.3f}; median B r {np.nanmedian(_f(sel, 'b_spearman')):+.3f}"
            v.append(X.verdict(f"V5 stratum quality {name} (frac within +-100 nm vs median {med:.3f})",
                               "informative", txt, None))
    v += gate_strata(ok, results, lines)
    X.write_verdicts(os.path.join(out, "VERDICT_v5_pilot.md"), "V5 training set (pilot), EXPLORATORY", v, lines)
    return v


SUBSETS = (("reliable_d7", "reliable (gate with slice misfit, P3 v2 default)"),
           ("reliable_d12", "reliable (D12: slice misfit only informs)"),
           ("accepted", "all accepted contours"))


def gate_strata(ok: Sequence[Dict[str, Any]], results: Sequence[pdt.PerimeterResult],
                lines: List[str]) -> List[Dict[str, Any]]:
    """A, B, C per axon and pooled A on three subsets (reliable D7 / reliable D12 / all accepted) and by curve
    kind (primary / stiff fallback / polar fallback). Descriptive: one animal, no p between groups."""
    from tools import mps_perimeter_a as A
    v: List[Dict[str, Any]] = []
    by_axon = {r.trace.label: r for r in results}
    lines.append("")
    lines.append("Per-axon gate: " + "; ".join(
        f"{r['axon']} {r.get('contour_kind')} d7={r.get('reliable_d7')} d12={r.get('reliable_d12')}" for r in ok))
    keys = (("A contrast", "a_contrast_p"), ("A period", "a_period_p"), ("B", "b_p"), ("C(-1,0)", "c_p_m1_0"),
            ("C(0,+1)", "c_p_0_p1"), ("C(-1,+1)", "c_p_m1_p1"))
    for sub, lab in SUBSETS:
        sel = list(ok) if sub == "accepted" else [r for r in ok if str(r.get(sub)) == "True"]
        txt = f"n = {len(sel)} ({', '.join(str(r['axon']) for r in sel)}); "
        txt += "; ".join(f"{k} p<0.05 {int(np.sum(_f(sel, key) < X.ALPHA))}/{int(np.isfinite(_f(sel, key)).sum())}"
                         for k, key in keys)
        for key in ("c_r_m1_0", "c_r_0_p1", "c_r_m1_p1", "b_spearman", "b_tau_vs_offset", "a_contrast_z"):
            vals = _f(sel, key)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                txt += f"; {key} " + ", ".join(f"{x:+.2f}" for x in vals) if vals.size <= 4 else \
                    f"; {key} median {np.median(vals):+.3f} (range {vals.min():+.2f} to {vals.max():+.2f})"
        v.append(X.verdict(f"V5 subset: {lab}", "descriptive (one animal)", txt, None))
        profs = []
        for r in sel:
            res = by_axon.get(str(r.get("trace_label")))
            if res is not None and res.profiles is not None:
                profs.append(res.profiles.by_slice[0])
        if len(profs) >= 2:
            det = A.pooled_details(profs, n_null=1999, seed=0)
            z = "; ".join(f"x{s.widen:g}: contrast-null {s.contrast_minus_null:+.3f} p {s.contrast.p:.3g}, period p "
                          f"{s.period.p:.3g} ({s.period_nm:.0f} nm)" for s in det.sensitivity)
            v.append(X.verdict(f"V5 pooled A over the subset '{sub}' (within-sample, exploratory)", "descriptive",
                               f"{len(profs)} axons: reading '{det.label}'; contrast Holm p {det.contrast_p_holm:.3g}, "
                               f"period Holm p {det.period_p_holm:.3g} ({det.period_nm:.0f} nm); zone {z}", None))
        else:
            v.append(X.verdict(f"V5 pooled A over the subset '{sub}'", "descriptive",
                               f"{len(profs)} axon(s): not poolable", None))
    for kind in ("primary", "stiff fallback", "polar fallback"):
        sel = [r for r in ok if str(r.get("contour_kind")) == kind]
        txt = f"n = {len(sel)}; " + "; ".join(
            f"{k} {int(np.sum(_f(sel, key) < X.ALPHA))}/{int(np.isfinite(_f(sel, key)).sum())}" for k, key in keys)
        txt += f"; reliable d7 {sum(str(r.get('reliable_d7')) == 'True' for r in sel)}, d12 " \
               f"{sum(str(r.get('reliable_d12')) == 'True' for r in sel)}"
        for kk in ("contour_frac100", "contour_robust_sd_nm", "contour_persistent_offset_nm"):
            txt += f"; {kk} {_summ(_f(sel, kk))}"
        v.append(X.verdict(f"V5 stratum curve kind '{kind}'", "descriptive", txt, None))
    return v


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P11 V5 (training set, exploratory)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    a = sub.add_parser("one")
    a.add_argument("--file", required=True)
    a.add_argument("--variant", default="auto", choices=VARIANTS)
    a.add_argument("--out", required=True)
    c = sub.add_parser("chunk")      # several files in one process, sequentially, skipping those already done
    c.add_argument("--variant", default="auto", choices=VARIANTS)
    c.add_argument("--max-seconds", type=float, default=480.0)
    c.add_argument("--out", required=True)
    b = sub.add_parser("pool")
    b.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "list":
        for f in real_files():
            print(f)
        return 0
    out = X.ensure_out(args.out)
    if args.cmd == "one":
        run_one(args.file, args.variant, out)
    elif args.cmd == "chunk":
        t0 = time.time()
        done_csv = os.path.join(out, f"pilot_readouts_{args.variant}.csv")
        done = {r["axon"] for r in X.read_csv(done_csv)} if os.path.exists(done_csv) else set()
        todo = [f for f in sorted(real_files()) if real_tag(f) not in done]
        for f in todo:
            if time.time() - t0 > args.max_seconds:
                break
            run_one(f, args.variant, out)
        left = len([f for f in todo if real_tag(f) not in
                    {r["axon"] for r in X.read_csv(done_csv)}]) if os.path.exists(done_csv) else len(todo)
        print(f"chunk {args.variant}: {left} files left", flush=True)
        return 0 if left == 0 else 10
    else:
        for x in pool_report(out):
            print(x["verdict"], x["check"], "|", x["measured"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
