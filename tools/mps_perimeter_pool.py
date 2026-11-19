"""
Pooling the perimeter results across axons and animals (package P10).

The unit of inference between groups is the ANIMAL (``tools.mps_gap_batch``: ``aggregate_by_animal`` averages the
signed z readouts per animal, ``compare_groups`` refuses a p with fewer than 3 animals per group).

* Rows without an animal go to a group labelled ``NO_ANIMAL_GROUP`` ("SIN ANIMAL", the label of the statistical
  plan, imported from ``mps_perimeter_a``): descriptive numbers per ROI, never a p between genotypes.
* A folder known to be ONE animal whose label is unknown (the 18-axon training set) is given
  ``PROVISIONAL_ANIMAL`` by the batch when asked; it is a placeholder, not an animal's name. With
  n_animals = 1 no comparison between animals is possible and every output says so.
* The readouts that are aggregated are the type-B signed z of each axon against its own null (``Z_READOUTS``).

Besides the four functions of the contract this module builds the rows of the long tables (profiles, contour,
pooled A, band sensitivity, peak - focus per ROI). They are plain dictionaries: P9 decides the final files.
Pure functions, numpy only (scipy through ``mps_gap_batch``), no Qt, no I/O.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools import mps_gap_batch as gb
from tools import mps_perimeter_a as pa
from tools import mps_perimeter_bc as bc
from tools import mps_perimeter_types as pdt
from tools.mps_perimeter_density import centred_lags_nm, centred_order

NO_ANIMAL_GROUP: str = pa.NO_ANIMAL_GROUP
# [ad hoc, decision Q7] placeholder for a folder known to be a single animal whose label is unknown. Explicitly
# provisional; never an animal's real name.
PROVISIONAL_ANIMAL: str = "provisional-single-animal"
# The signed z readouts aggregated per animal (plan sec. 3.1, type B of the statistical plan).
Z_READOUTS: Tuple[str, ...] = ("pd_a_contrast_z", "pd_a_period_z", "pd_b_z", "pd_c_z_m1_p1", "pd_c_z_m1_0",
                               "pd_c_z_0_p1")
# The readouts whose per-animal t test is computed (and Holm-adjusted among themselves). Left out, always with an
# empty p: the neighbouring slice pairs (axial leakage alone makes their z positive, so a p would measure the leak)
# and the period z (its per-axon null is skewed; doc sec. 6). They are still averaged (descriptive).
TESTED_Z_READOUTS: Tuple[str, ...] = ("pd_a_contrast_z", "pd_b_z", "pd_c_z_m1_p1")
# v20 (P-R3-1): C is tested only in its own scope (SCOPE_C_MAIN, axons whose slices coincide); A and B only outside it.
C_TESTED: Tuple[str, ...] = ("pd_c_z_m1_p1",)
AB_TESTED: Tuple[str, ...] = ("pd_a_contrast_z", "pd_b_z")
UNTESTED_NOTE: str = ("no p for pd_c_z_m1_0 and pd_c_z_0_p1 (axial leakage alone makes the neighbouring pairs "
                      "positive) nor for pd_a_period_z (its per-axon null is skewed): descriptive only")
C_ELSEWHERE_NOTE: str = ("pd_c_* descriptive here: the main reading of C is the scope 'reliable contours with "
                         "coinciding slices (main reading of C)'")
AB_ELSEWHERE_NOTE: str = ("pd_a_* and pd_b_* descriptive here: A and B are read in the other scopes (the slice "
                          "criterion is for C only)")
JOINT_HOLM_NOTE: str = ("p_holm_* is Holm over the family of three (A contrast and B from the main scope, C(-1, +1) "
                        "from the main scope of C)")
ONE_ANIMAL_NOTE: str = "one animal: no p"
PROVISIONAL_NOTE: str = "provisional animal: no p"
# Suffixes of the per-axon reading of C (pd_c_reading) when it cannot be read as it stands. The numbers do not change.
C_SUFFIX_NOT_RELIABLE: str = " (contour not reliable: not interpretable)"
C_SUFFIX_NOT_ASSESSED: str = " (contour not assessed: not interpretable)"
C_SUFFIX_MISFIT: str = " (slice misfit: the correlation can be produced by a lateral shift)"   # before v20
# v20 (P-R3-1): the calibrated slice criterion (mps_perimeter_bc.slices_coincide_for_c) replaces the misfit mark
# for C. A reliable contour whose slices do not coincide is read apart, descriptively.
C_SUFFIX_SLICES: str = " (slices do not coincide: outside the main reading of C, descriptive)"
# r(-1, +1) of a "registry" reading compared with the leak expected for the pair with the axon's own fractions.
# v21 (user's decision 2026-11-17, NeNA in z): main width 47 nm, measured range 36-51 nm, 77 nm only an upper bound.
C_SUFFIX_LEAK_NOT_ABOVE: str = " (r(-1, +1) not above the axial leak expected even with sigma_z 36 nm)"
C_SUFFIX_LEAK_LIMIT: str = (" (r(-1, +1) at the limit: above the axial leak expected with sigma_z 36 nm, not with "
                            "47 nm)")
C_SUFFIX_LEAK_RANGE: str = (" (r(-1, +1) above the axial leak expected with sigma_z 47 nm but within its measured "
                            "range 36-51 nm)")
# Smallest pooled set of A that gets a reading (v20 set it to 11 from sizes 4, 11 and 18 only; fixed at 10 on
# 2026-11-17 after simulating every size from 5 to 10). [own sim; Bradley 1978 robustness band 7.5 %]: with
# ``validate_perimeter_a.py`` (P1's simulator, the program's pooled A; 8-25 um axons, 30 localizations per bin,
# blinking, true arc), 300 sets per null (flat, clumps, resolved cluster sd 150 nm, t-tailed cluster 300 nm) and 500
# per effect (attractor, halo, inhibition zone, regular) at each size, the minimum is the smallest size that meets,
# together with every larger simulated size:
#   S  every null emits each reading ('attractor', 'regular', 'inhibition zone') with an upper 95 % Wilson bound
#      <= 0.075 (the ceiling of Bradley's liberal band, the same as the slice criterion of C): met at every size;
#   I  every effect the reading can recognise (attractor, inhibition zone, regular) appears with a lower 95 % Wilson
#      bound > 0.075, i.e. more often than a calibrated null may produce it, so "no structure detectable" carries
#      information against it. The halo is left out: ~0.01 at every size up to 18 (the robustness rule takes it for
#      a cluster flank), no number of axons fixes it.
# The inhibition zone rules: 0.092 (0.070-0.121) with 9 axons, 0.104 (0.080-0.134) with 10, 0.125 (0.086-0.178)
# with 11; attractor 0.39 with 10. The margin at 10 is thin (lower bound 0.080 against 0.075). Power is far below the
# conventional 0.8 at every size up to 18 (attractor 0.62, inhibition 0.28 with 18): the minimum makes a reading
# informative, not powerful. Below it the label says "n too small" and the numbers stay, not interpreted.
# 2026-11-19 [own sim, 2000 sets per scenario at 9, 10 and 11 axons, new seeds, same criterion fixed beforehand]:
# S and I are met at 9, 10 and 11; inhibition zone 0.106 (0.094-0.121) with 9, 0.112 (0.099-0.127) with 10,
# 0.119 (0.106-0.134) with 11; worst null 0.016-0.025 (upper bound <= 0.032). With 8 the 500-set estimate of
# the inhibition zone (0.074) is already below the ceiling. Minimum lowered to 9 [user decision 2026-11-19].
MIN_POOLED_A_AXONS: int = 9
READING_N_TOO_SMALL: str = "n too small"
# Distances summarised per ROI for decision Q1.
PEAK_FOCUS_COLUMNS: Tuple[str, ...] = ("pd_peak_minus_focus_nominal_nm", "pd_abs_peak_minus_focus_nominal_nm",
                                       "pd_peak_minus_focus_empirical_nm", "pd_abs_peak_minus_focus_empirical_nm")
_PAIRS: Tuple[Tuple[Tuple[int, int], str], ...] = (((-1, 0), "m1_0"), ((0, 1), "0_p1"), ((-1, 1), "m1_p1"))


# ----------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------
def _f(v: Optional[float]) -> Optional[float]:
    """A finite float, or None (an empty cell)."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _abs(v: Optional[float]) -> Optional[float]:
    x = _f(v)
    return None if x is None else abs(x)


def _z0_mode(c: pdt.Z0Criterion) -> Tuple[str, str]:
    if c == pdt.Z0Criterion.PEAK:
        return "peak", ""
    return "focus", "nominal" if c == pdt.Z0Criterion.FOCUS_NOMINAL else "empirical"


def _ident(r: pdt.PerimeterResult, name: str) -> str:
    return "" if r.identity is None else str(getattr(r.identity, name, "") or "")


# ----------------------------------------------------------------------------------------------------
# The quality gate of the pooled analyses (P10b; decisions D9 and D12)
# ----------------------------------------------------------------------------------------------------
# Which pooled set a pooled row belongs to. The main set holds every axon whose analysis finished and whose contour
# passed the quality gate; the slice misfit (D7) does NOT keep an axon out (D12), so the same pooling is also given
# WITHOUT the axons the misfit marked, and stratified by the kind of curve (the three automatic curves and the
# drawn trace are different estimators: report v2 of the automatic contour, sec. 2.2).
SCOPE_MAIN: str = "reliable contours (slice misfit included)"
SCOPE_NO_MISFIT: str = "reliable contours without slice misfit"
SCOPE_CURVE_PREFIX: str = "reliable contours, curve: "
# v20 (P-R3-1): the main reading of C: the main set restricted to axons whose slices coincide (C only).
SCOPE_C_MAIN: str = "reliable contours with coinciding slices (main reading of C)"
# [code] the automatic contour's CURVE_KINDS folded into the four strata of the report; anything else (a curve a
# caller-supplied contour function fitted, "none") is its own stratum "external" / "none".
CURVE_STRATA: Dict[str, str] = {"primary": "primary", "stiff fallback": "stiff", "polar fallback": "polar",
                                "polar": "polar", "drawn": "drawn"}
STRATA_ORDER: Tuple[str, ...] = ("primary", "stiff", "polar", "drawn", "external")


def curve_stratum(kind: Optional[str]) -> str:
    """The stratum of a curve kind: primary / stiff / polar / drawn, else the kind itself ("none" when empty)."""
    k = str(kind or "").strip()
    return CURVE_STRATA.get(k, k or "none")


def pool_gate(r: pdt.PerimeterResult) -> Dict[str, Any]:
    """Whether one axon enters the pooled analyses, and why not.

    Keys: ``pd_pool_included`` (main set), ``pd_pool_included_without_misfit`` (the set without the slice misfit),
    ``pd_contour_slice_misfit`` (the mark of D7, reported only), ``pd_curve_stratum`` and
    ``pd_pool_exclusion_reason`` (one sentence; empty when included). An axon is left out when its analysis did not
    finish (its status and failure) or when its contour failed the quality gate for any reason other than the slice
    misfit (fraction within +-100 nm, radial spread, persistent band offset, arc off the band). A finished analysis
    whose contour carries no assessment is left out too (the gate cannot vouch for it).

    v20 (P-R3-1): ``pd_c_slices_coincide`` (``mps_perimeter_bc.slices_coincide_for_c`` on the two slice misfits;
    False when either is not measurable, None without an assessment) and ``pd_c_main_reading`` (included AND the
    slices coincide: the axon is in the main reading of C).
    """
    a = (r.extras or {}).get("contour_assessment")
    coincide: Optional[bool] = None
    if a:
        coincide = bc.slices_coincide_for_c(a.get("contour_slice_rms_minus_nm"), a.get("contour_slice_rms_plus_nm"))
    misfit = bool(a.get("contour_slice_misfit")) if a else False
    stratum = curve_stratum(a.get("contour_kind") if a else None)
    reason = ""
    if r.status != pdt.Status.OK:
        reason = f"analysis {r.status.value}" + (f": {r.failure}" if r.failure else "")
    elif not a:
        reason = "contour not assessed by the quality gate"
    elif str(a.get("contour_failure_reason") or ""):
        reason = f"contour rejected: {a.get('contour_failure_reason')}"
    elif str(a.get("contour_gate_reasons") or ""):
        reason = f"contour not reliable: {a.get('contour_gate_reasons')}"
    included = reason == ""
    return {"pd_pool_included": included, "pd_pool_included_without_misfit": bool(included and not misfit),
            "pd_contour_slice_misfit": misfit if a else None, "pd_curve_stratum": stratum,
            "pd_pool_exclusion_reason": reason or None,
            "pd_c_slices_coincide": coincide, "pd_c_main_reading": bool(included and coincide)}


def _in_scope_row(row: Dict[str, Any], scope: str) -> bool:
    """Whether a flat row belongs to ``scope``. A row without the gate keys (built by hand, older code) counts as
    included in every scope but the curve strata and the main scope of C (which needs ``pd_c_main_reading``)."""
    if scope == SCOPE_C_MAIN:
        return bool(row.get("pd_c_main_reading"))
    if "pd_pool_included" not in row:
        return not scope.startswith(SCOPE_CURVE_PREFIX)
    if scope == SCOPE_NO_MISFIT:
        return bool(row.get("pd_pool_included_without_misfit"))
    if scope.startswith(SCOPE_CURVE_PREFIX):
        return bool(row.get("pd_pool_included")) and row.get("pd_curve_stratum") == scope[len(SCOPE_CURVE_PREFIX):]
    if scope != SCOPE_MAIN:
        raise ValueError(f"unknown pooling scope {scope!r}")
    return bool(row.get("pd_pool_included"))


def _in_scope(r: pdt.PerimeterResult, scope: str) -> bool:
    return _in_scope_row(pool_gate(r), scope)


def scopes_for(strata: Sequence[str]) -> List[str]:
    """The scopes reported: the main set, the set without slice misfit, the main scope of C (v20), then one per
    curve stratum present."""
    present = [s for s in STRATA_ORDER if s in strata] + sorted(set(strata) - set(STRATA_ORDER) - {"none"})
    return [SCOPE_MAIN, SCOPE_NO_MISFIT, SCOPE_C_MAIN] + [SCOPE_CURVE_PREFIX + s for s in present]


def tested_for_scope(scope: str) -> Tuple[str, ...]:
    """The readouts that get a p in ``scope``: C(-1, +1) alone in the main scope of C, A contrast and B elsewhere."""
    return C_TESTED if scope == SCOPE_C_MAIN else AB_TESTED


def gate_rows(results: Sequence[pdt.PerimeterResult]) -> List[Dict[str, Any]]:
    """One row per axon: whether it enters the pooled analyses (with and without the slice misfit) and why not."""
    out: List[Dict[str, Any]] = []
    for r in results:
        a = (r.extras or {}).get("contour_assessment") or {}
        row: Dict[str, Any] = {"axon_id": r.axon_id, "trace_label": r.trace.label,
                               "trace_roi": r.trace.roi_number or None, "pd_status": r.status.value,
                               "pd_contour_kind": a.get("contour_kind")}
        row.update(pool_gate(r))
        row["pd_contour_reliable"] = a.get("contour_reliable") if a else None
        row["pd_contour_quality_reasons"] = a.get("contour_quality_reasons") or None
        out.append(row)
    return out


def gate_summary_rows(results: Sequence[pdt.PerimeterResult]) -> List[Dict[str, Any]]:
    """How many axons each pooled set holds and why the others are out: one row per scope."""
    gates = [pool_gate(r) for r in results]
    strata = [g["pd_curve_stratum"] for g in gates if g["pd_pool_included"]]
    out: List[Dict[str, Any]] = []
    for scope in scopes_for(strata):
        inside = [g for g in gates if _in_scope_row(g, scope)]
        failed = sum(1 for r in results if r.status != pdt.Status.OK)
        unreliable = sum(1 for r, g in zip(results, gates) if r.status == pdt.Status.OK and not g["pd_pool_included"])
        misfit_out = sum(1 for g in gates if g["pd_pool_included"] and not _in_scope_row(g, scope)
                         and scope == SCOPE_NO_MISFIT)
        slices_out = sum(1 for g in gates if g["pd_pool_included"] and not g["pd_c_slices_coincide"]
                         and scope == SCOPE_C_MAIN)
        out.append({"scope": scope, "n_axons_total": len(results), "n_included": len(inside),
                    "n_excluded": len(results) - len(inside), "n_excluded_analysis_failed": failed,
                    "n_excluded_contour_not_reliable": unreliable, "n_excluded_slice_misfit": misfit_out,
                    "n_included_with_slice_misfit": sum(1 for g in inside if g["pd_contour_slice_misfit"]),
                    "included_axons": " | ".join(r.trace.label for r, g in zip(results, gates)
                                                 if _in_scope_row(g, scope)) or None,
                    "n_excluded_slices_not_coinciding": slices_out})
    return out


def axon_c_reading(reading: Optional[str], gate: Dict[str, Any], vs_leak: Optional[str] = None) -> Optional[str]:
    """The per-axon reading of C (``mps_perimeter_bc.slice_reading_real_fractions``) with the reason it cannot be
    read as it stands, from the pooling gate of the same axon (``pool_gate``). The rule and the numbers do not change.

    * contour not reliable (the gate left the axon out for a reason other than the misfit) or not assessed: every
      computable reading gets a suffix saying it is not interpretable (a contour error common to the three slices
      creates a common correlation, r(-1, +1) included, and blurs a real one; doc sec. 8);
    * v20 (P-R3-1): contour reliable but the slices do not coincide (``pd_c_slices_coincide`` false, which includes
      "not measurable"): every computable reading gets ``C_SUFFIX_SLICES``: the axon is outside the main reading of
      C and is reported apart, descriptively. Before v20 the suffix followed the uncalibrated misfit mark
      (``C_SUFFIX_MISFIT``); the calibrated criterion replaces it for C (one shifted lateral, which the mark
      flags, does not inflate C);
    * v20: a "registry between slices" reading whose r(-1, +1) is not above the leak expected for the pair
      (``vs_leak``, ``mps_perimeter_bc.m1_p1_vs_leak``) says so (v21: main width 47 nm, measured range 36-51 nm):
      ``C_SUFFIX_LEAK_NOT_ABOVE`` (not even above the 36 nm leak), ``C_SUFFIX_LEAK_LIMIT`` (above 36, not above
      47: reported as a limit case) or ``C_SUFFIX_LEAK_RANGE`` (above 47, not above 51). The 77 nm upper bound adds
      no suffix (it holds the rings' physical thickness).
    "not computable" and an empty reading are unchanged.
    """
    if not reading or reading == "not computable":
        return reading or None
    reason = str(gate.get("pd_pool_exclusion_reason") or "")
    out = reading
    if reason.startswith("contour not assessed"):
        out += C_SUFFIX_NOT_ASSESSED
    elif reason.startswith("contour"):
        out += C_SUFFIX_NOT_RELIABLE
    elif gate.get("pd_pool_included") and not gate.get("pd_c_slices_coincide"):
        out += C_SUFFIX_SLICES
    if reading == "registry between slices":
        out += {"not_above": C_SUFFIX_LEAK_NOT_ABOVE, "limit": C_SUFFIX_LEAK_LIMIT,
                "range": C_SUFFIX_LEAK_RANGE}.get(bc.vs_leak_kind(vs_leak), "")
    return out


def axon_a_reading(r: pdt.PerimeterResult) -> Optional[str]:
    """The reading of question A for ONE axon, with the pooled reading's vocabulary (``mps_perimeter_a``):
    "attractor" / "inhibition zone" (contrast, Holm-adjusted with the period, above / below its null),
    "regular" (period peak significant and not at the edge), a combination with " + ", "no structure detectable",
    or "not computable". No exclusion-zone sensitivity is run per axon, so the "not robust" suffix of the pooled
    reading is never added here. None when the analysis did not finish."""
    if r.status != pdt.Status.OK:
        return None
    dg = r.diagnostics
    c, per = r.a.get("contrast"), r.a.get("period")
    if dg is None or not dg.a_usable or c is None or per is None:
        return pa.READING_NOT_COMPUTABLE
    p_c, p_p = pa.holm_two(float(c.p), float(per.p))
    found: List[str] = []
    if math.isfinite(p_c) and p_c <= pa.ALPHA:
        found.append(pa.READING_ATTRACTOR if c.value - c.null_mean > 0 else pa.READING_INHIBITION)
    if math.isfinite(p_p) and p_p <= pa.ALPHA and not dg.a_period_at_edge:
        found.append(pa.READING_REGULAR)
    return pa.READING_JOIN.join(found) if found else pa.READING_NONE


def a_reading_label(result: pdt.PooledResult) -> str:
    """The bare reading label of a pooled A ("attractor", "regular", "inhibition zone + regular", "no structure
    detectable", "not computable"), without the Holm p, the period and the scope that ``reading`` appends, and
    without the exploratory / within-sample prefix of P7's R2 corrections (``mps_perimeter_a.reading_label``)."""
    return pa.reading_label(result)


# ----------------------------------------------------------------------------------------------------
# Per-axon rows
# ----------------------------------------------------------------------------------------------------
def perimeter_row(r: pdt.PerimeterResult) -> Dict[str, Any]:
    """The flat row of one axon: every ``tools.perimeter_export.PERIMETER_COLUMNS`` name plus the identity, both
    peak - focus distances and the readings. Empty cells are None; a failed axon keeps what was computed."""
    p, z0, sl, ct, bd, pr, dg = r.params, r.z0, r.slices, r.contour, r.band, r.profiles, r.diagnostics
    mode, kind = _z0_mode(p.z0_criterion)
    t = r.trace
    row: Dict[str, Any] = {
        "axon_id": r.axon_id, "source_path": r.source_path,
        "genotype": _ident(r, "genotype"), "animal": _ident(r, "animal"), "protein": _ident(r, "protein"),
        "roi_name": _ident(r, "roi_name"), "axon_name": _ident(r, "axon_name"),
        "trace_label": t.label, "trace_year": t.year, "trace_complete": bool(t.complete),
        "acquisition_date": ".".join(t.acquisition_date) if t.acquisition_date else None,
        "trace_warnings": " | ".join(r.trace_warnings) or None,
        "pd_status": r.status.value, "pd_failure": r.failure or None, "pd_version": pdt.PD_VERSION,
        "pd_z0_mode": mode, "pd_focus_kind": kind, "pd_z0_criterion": p.z0_criterion.value,
        "pd_z0_nm": _f(z0.z0_nm), "pd_z_peak_nm": _f(z0.peak_nm),
        "pd_z_focus_nominal_nm": _f(z0.focus_nominal_nm), "pd_z_focus_empirical_nm": _f(z0.focus_empirical_nm),
        "pd_focus_empirical_n_bins": None if r.focus is None else int(r.focus.empirical_n_bins),
        "pd_peak_minus_focus_nm": _f(z0.peak_minus_focus_nm), "pd_abs_peak_minus_focus_nm": _abs(z0.peak_minus_focus_nm),
        "pd_peak_minus_focus_nominal_nm": _f(z0.peak_minus_focus_nominal_nm),
        "pd_abs_peak_minus_focus_nominal_nm": _abs(z0.peak_minus_focus_nominal_nm),
        "pd_peak_minus_focus_empirical_nm": _f(z0.peak_minus_focus_empirical_nm),
        "pd_abs_peak_minus_focus_empirical_nm": _abs(z0.peak_minus_focus_empirical_nm),
        "pd_slice_thickness_nm": float(p.slice_thickness_nm), "pd_slice_guard_nm": float(p.slice_guard_nm),
        "pd_bin_nm": float(p.bin_nm), "pd_b_window_nm": float(p.window_nm), "pd_n_null": int(p.n_null),
        "pd_seed": int(p.seed), "pd_warnings": " | ".join(r.warnings) or None,
    }
    if sl is not None:
        row.update(pd_n_locs_m1=sl.n_locs[0], pd_n_locs_0=sl.n_locs[1], pd_n_locs_p1=sl.n_locs[2],
                   pd_pct_m1_of_0=_f(sl.pct_of_central[0]), pd_pct_p1_of_0=_f(sl.pct_of_central[1]),
                   pd_frac_locs_in_slices=_f(sl.frac_in_three), pd_z_coverage_m1=_f(sl.z_coverage[0]),
                   pd_z_coverage_0=_f(sl.z_coverage[1]), pd_z_coverage_p1=_f(sl.z_coverage[2]))
    if ct is not None:
        row.update(pd_contour_method=ct.method, pd_contour_status=ct.status.value, pd_contour_knot_nm=_f(ct.knot_nm),
                   pd_contour_length_nm=_f(ct.length_nm if ct.path is None else ct.path.length_nm),
                   pd_contour_residual_sd_nm=_f(ct.residual_sd_nm), pd_contour_self_crossings=int(ct.n_self_crossings),
                   pd_contour_vs_polar_nm=_f(ct.polar_distance_nm), pd_contour_star_shaped=bool(ct.star_shaped),
                   pd_contour_n_nodes=int(np.asarray(ct.nodes_nm).reshape(-1, 2).shape[0]))
    if bd is not None:
        row.update(pd_band_mode=bd.mode, pd_band_half_width_nm=_f(bd.half_width_nm), pd_radial_fwhm_nm=_f(bd.fwhm_nm),
                   pd_radial_sd_nm=_f(bd.robust_sd_nm), pd_frac_within_50nm=_f(bd.frac_within.get(50.0)),
                   pd_frac_within_100nm=_f(bd.frac_within.get(100.0)),
                   pd_frac_within_150nm=_f(bd.frac_within.get(150.0)))
    if pr is not None:
        p0 = pr.by_slice[0]
        row.update(pd_bin_nm_effective=_f(p0.bin_nm_effective), pd_n_bins=int(p0.n_bins),
                   pd_max_count_0=int(p0.counts.max()) if p0.counts.size else None, pd_max_tie=bool(p0.max_tie))
    if dg is not None:
        row.update(pd_contour_unsupported_fraction=_f(dg.unsupported_fraction),
                   pd_radial_shift_m1_nm=_f(dg.radial_shift_nm[0]), pd_radial_shift_p1_nm=_f(dg.radial_shift_nm[1]),
                   pd_contour_reversed=bool(dg.contour_reversed), pd_drawn_vs_auto_nm=_f(dg.drawn_vs_auto_nm),
                   pd_lpz_nm=_f(dg.lpz_nm), pd_lpz_source=dg.lpz_source or None,
                   pd_c_leak_expected_r=_f(dg.leak_expected_r), pd_c_reading=dg.c_reading or None,
                   pd_a_exclusion_nm=_f(dg.a_exclusion_nm), pd_a_period_nm=_f(dg.a_period_nm),
                   pd_a_period_at_edge=bool(dg.a_period_at_edge) if dg.a_usable else None,
                   pd_band_sensitivity=dg.band_sensitivity_flag or None,
                   pd_z_truncated_slices=",".join(f"{k:+d}" for k in dg.z_truncated_slices) or None)
    for key, name in (("contrast", "contrast"), ("period", "period")):
        tr = r.a.get(key)
        if tr is not None:
            row[f"pd_a_{name}"] = _f(tr.value)
            row[f"pd_a_{name}_z"] = _f(tr.z)
            row[f"pd_a_{name}_p"] = _f(tr.p)
    if r.b is not None:
        row.update(pd_b_spearman=_f(r.b.test.value), pd_b_z=_f(r.b.test.z), pd_b_p=_f(r.b.test.p),
                   pd_b_tau_min=_f(r.b.tau_min), pd_b_curvature_resolved=bool(r.b.resolved),
                   pd_b_tau_vs_offset=_f(r.b.tau_vs_offset))
    for pair, suffix in _PAIRS:
        tr = r.c.get(pair)
        if tr is not None:
            row[f"pd_c_r_{suffix}"] = _f(tr.value)
            row[f"pd_c_z_{suffix}"] = _f(tr.z)
            row[f"pd_c_p_{suffix}"] = _f(tr.p)
    # ---- P10b (v19): the contour's assessment, the pooling gate, the per-axon reading of A ----
    ex = r.extras or {}
    row["trace_roi"] = t.roi_number or None
    a = ex.get("contour_assessment")
    if a:
        for key, value in a.items():
            if key in ("contour_slice_misfit",):
                continue                                   # written by pool_gate (None when not assessed)
            name = ASSESSMENT_RENAMES.get(key, "pd_" + key)
            row[name] = (value or None) if isinstance(value, str) else (
                _f(value) if isinstance(value, float) else value)
    gate = pool_gate(r)
    row.update(gate)
    # ---- v20 / v21: the axial leak of every pair with this axon's own slice fractions: 47 nm (main) and 77 nm (upper
    # bound) in v20's columns, the measured range 36 and 51 nm in v21's ----
    widths = result_widths(r)
    leaks: Dict[float, Dict[Tuple[int, int], float]] = {}
    if sl is not None:
        for sz in tuple(bc.LEAK_SIGMA_Z_ALL_NM) + (widths.lo, widths.main, widths.hi):
            leaks[sz] = bc.leak_expected_r_pairs(sl.n_locs, sz, float(p.slice_thickness_nm), float(p.slice_guard_nm))
    for sz in bc.LEAK_SIGMA_Z_ALL_NM:
        for pair, suffix in _PAIRS:
            row[leak_column(suffix, sz)] = _f(leaks[sz][pair]) if sz in leaks else None
    row.update(sigma_z_columns(r, widths, leaks))
    vs_leak = bc.m1_p1_vs_leak(row.get("pd_c_r_m1_p1"), {sz: leaks[sz][(-1, 1)] for sz in leaks}, widths)
    row["pd_c_m1_p1_vs_leak"] = vs_leak
    if "pd_c_reading" in row:                              # R3: the reading says when it cannot be read as it is
        row["pd_c_reading"] = axon_c_reading(row["pd_c_reading"], gate, vs_leak)
    row["pd_contour_trim_status"] = ex.get("contour_trim_status")
    row["pd_a_reading"] = axon_a_reading(r)
    # P11 sec. 9.1: one grid of columns for every axon. A failed axon (contour failed, ...) carries every column of
    # the layout, empty where nothing was computed (pd_a_*, pd_b_*, pd_c_* included), instead of omitting them.
    from tools.perimeter_export import ALL_PERIMETER_COLUMNS
    for name in ALL_PERIMETER_COLUMNS:
        row.setdefault(name, None)
    return row


def result_widths(r: pdt.PerimeterResult) -> bc.LeakWidths:
    """The axial widths the leak of this axon was read against (``PerimeterDiagnostics.sigma_z_*``): the file's own
    when its sigma_z was measured reliably, the pilot's 47 nm (36-51 nm) otherwise."""
    dg = r.diagnostics
    if dg is not None and dg.sigma_z_source == bc.SIGMA_Z_SOURCE_MEASURED and all(
            math.isfinite(v) for v in dg.sigma_z_widths):
        return bc.measured_widths(*dg.sigma_z_widths)
    return bc.pilot_widths()


def sigma_z_columns(r: pdt.PerimeterResult, widths: bc.LeakWidths,
                    leaks: Dict[float, Dict[Tuple[int, int], float]]) -> Dict[str, Any]:
    """The v22 columns: where the width of the leak came from, what was measured, and the leak at the width used."""
    dg = r.diagnostics
    rep = None if dg is None else dg.sigma_z_report
    out: Dict[str, Any] = {}
    if dg is not None and dg.sigma_z_source:
        out["pd_sz_source"] = dg.sigma_z_source
        out["pd_sz_main_nm"], out["pd_sz_lo_nm"], out["pd_sz_hi_nm"] = _f(widths.main), _f(widths.lo), _f(widths.hi)
        for pair, suffix in _PAIRS:
            for tag, sz in (("", widths.main), ("_lo", widths.lo), ("_hi", widths.hi)):
                out[f"pd_c_leak_r_{suffix}_szused{tag}"] = _f(leaks[sz][pair]) if sz in leaks else None
    if rep is not None:
        out.update(pd_sz_status=rep.status, pd_sz_frame_nm=_f(rep.frame_nm), pd_sz_frame_lo_nm=_f(rep.frame_lo_nm),
                   pd_sz_frame_hi_nm=_f(rep.frame_hi_nm), pd_sz_slow_nm=_f(rep.slow_nm),
                   pd_sz_slow_lo_nm=_f(rep.slow_lo_nm), pd_sz_slow_hi_nm=_f(rep.slow_hi_nm),
                   pd_sz_total_nm=_f(rep.total_nm), pd_sz_total_lo_nm=_f(rep.total_lo_nm),
                   pd_sz_total_hi_nm=_f(rep.total_hi_nm), pd_sz_n_pairs_frame=rep.n_pairs_frame,
                   pd_sz_n_pairs_lag1=rep.n_pairs_lag1, pd_sz_n_pairs_long=rep.n_pairs_long,
                   pd_sz_contamination=_f(rep.contamination), pd_sz_same_frame_fraction=_f(rep.same_frame_fraction),
                   pd_sz_note=rep.note() or None)
    return out


def leak_column(pair_suffix: str, sigma_z_nm: float) -> str:
    """Name of the v20 / v21 column of the expected leak of one pair at one axial width, e.g.
    pd_c_leak_r_m1_p1_sz47nm."""
    return f"pd_c_leak_r_{pair_suffix}_sz{sigma_z_nm:g}nm"


# P3's assessment_row names that would clash with a v17 column of another meaning: v17's
# pd_contour_unsupported_fraction is the share of central-slice bins not measured; P3's is the share of the curve's
# length with no localization density under it.
ASSESSMENT_RENAMES: Dict[str, str] = {"contour_unsupported_fraction": "pd_contour_support_unsupported_fraction"}


def perimeter_rows(results: Sequence[pdt.PerimeterResult]) -> List[Dict[str, Any]]:
    """One flat row per axon (``perimeter_row``), with the signed z readouts used downstream."""
    return [perimeter_row(r) for r in results]


# ----------------------------------------------------------------------------------------------------
# Per animal and between genotypes
# ----------------------------------------------------------------------------------------------------
def _ok_rows(rows: Sequence[Dict[str, Any]], scope: str = SCOPE_MAIN) -> List[Dict[str, Any]]:
    """The rows with status ok that belong to ``scope`` (the quality gate: ``pool_gate``)."""
    return [r for r in rows if r.get("pd_status") == pdt.Status.OK.value and _in_scope_row(r, scope)]


def summarise_by_animal(rows: Sequence[Dict[str, Any]], scope: str = SCOPE_MAIN) -> List[Dict[str, Any]]:
    """Per-animal means of the signed z readouts (``mps_gap_batch.aggregate_by_animal``), only axons with status ok
    whose contour passed the quality gate (``scope``: the main set by default; ``SCOPE_NO_MISFIT`` without the axons
    the slice misfit marked; ``SCOPE_CURVE_PREFIX`` + stratum for one kind of curve). Every row says its ``scope``.

    Rows without an animal: one row per ROI in the group ``NO_ANIMAL_GROUP`` with means, SDs and counts, and no p.
    Every row carries ``n_animals_total`` and a ``note``; with n_animals = 1 the note says no comparison between
    animals is possible, and a provisional label is flagged as such.

    The per-animal t test (``p_*``, ``p_holm_*``) is computed only for ``TESTED_Z_READOUTS`` and only with two or more
    animals; with one animal or the provisional label every p of the row is empty ("one animal: no p").
    v20 (P-R3-1): within one scope only ``tested_for_scope(scope)`` gets a p: C(-1, +1) in ``SCOPE_C_MAIN`` (axons
    whose slices coincide), A contrast and B in the other scopes; the rest of the row is descriptive and the note
    says where it is read. Holm here runs over the scope's own tested readouts; ``by_animal_rows_scoped`` re-does it
    over the family of three across the two scopes.
    """
    ok = _ok_rows(rows, scope)
    with_animal = [r for r in ok if str(r.get("animal") or "").strip()]
    without = [r for r in ok if not str(r.get("animal") or "").strip()]
    n_animals = len({str(r["animal"]).strip() for r in with_animal})
    out: List[Dict[str, Any]] = []
    tested = tested_for_scope(scope)
    for a in gb.aggregate_by_animal(with_animal, Z_READOUTS, animal_key="animal", tested=tested):
        genotypes = sorted({str(r.get("genotype") or "") for r in with_animal if str(r["animal"]).strip() == a["animal"]})
        a["genotype"] = genotypes[0] if len(genotypes) == 1 else ("mixed: " + ", ".join(genotypes))
        a["roi_name"] = None
        a["provisional"] = a["animal"] == PROVISIONAL_ANIMAL
        a["n_animals_total"] = n_animals
        notes: List[str] = []
        no_p = a["provisional"] or n_animals == 1
        if n_animals == 1:
            notes.append(ONE_ANIMAL_NOTE + " (n_animals = 1: no comparison between animals is possible)")
        if a["provisional"]:
            notes.append(PROVISIONAL_NOTE + " (provisional animal label: the real label is unknown)")
        if no_p:
            for m in Z_READOUTS:                            # the descriptive numbers stay
                a[f"p_{m}"] = None
                a[f"p_holm_{m}"] = None
        else:
            notes.append("p_* (" + ", ".join(tested) + ") is a t test of the axons' z against 0 within "
                         "this animal, treating its axons as replicates: not a comparison between animals, and it "
                         "does not generalise to other animals")
        notes.append(UNTESTED_NOTE)
        notes.append(AB_ELSEWHERE_NOTE if scope == SCOPE_C_MAIN else C_ELSEWHERE_NOTE)
        a["note"] = "; ".join(notes)
        out.append(a)
    by_roi: Dict[str, List[Dict[str, Any]]] = {}
    for r in without:
        by_roi.setdefault(str(r.get("roi_name") or ""), []).append(r)
    for roi, rr in sorted(by_roi.items()):
        row: Dict[str, Any] = {"animal": NO_ANIMAL_GROUP, "n_axons": len(rr), "genotype": None,
                               "roi_name": roi or None, "provisional": False, "n_animals_total": n_animals}
        for m in Z_READOUTS:
            v = np.array([x for x in (_f(q.get(m)) for q in rr) if x is not None], dtype=np.float64)
            row[f"n_{m}"] = int(v.size)
            row[f"mean_{m}"] = float(np.mean(v)) if v.size else None
            row[f"sd_{m}"] = float(np.std(v, ddof=1)) if v.size > 1 else None
            row[f"p_{m}"] = None
            row[f"p_holm_{m}"] = None
        row["note"] = f"{NO_ANIMAL_GROUP}: descriptive only, per ROI; no p and no comparison between genotypes"
        out.append(row)
    for row in out:
        row["scope"] = scope                                # last key: appended after the summarised readouts
    return out


def compare_genotypes(rows: Sequence[Dict[str, Any]], scope: str = SCOPE_MAIN) -> Dict[str, Any]:
    """Genotype comparison with the animal as the unit (``mps_gap_batch.compare_groups``), axons with status ok in
    ``scope`` (see ``summarise_by_animal``); the result says its ``scope``.

    Axons without an animal never enter (they are ``NO_ANIMAL_GROUP``); axons without a genotype neither. With
    fewer than 3 animals per group, or a single animal in total, the result is descriptive only (no p).
    """
    ok = _ok_rows(rows, scope)
    usable = [r for r in ok if str(r.get("animal") or "").strip() and str(r.get("genotype") or "").strip()]
    res = gb.compare_groups(usable, Z_READOUTS, group_key="genotype", animal_key="animal")
    # v20: p and Holm only for the readouts of this scope (tested_for_scope); the others stay descriptive
    tested = tested_for_scope(scope)
    res_t = gb.compare_groups(usable, tested, group_key="genotype", animal_key="animal")
    for m, entry in res.get("measures", {}).items():
        src = res_t.get("measures", {}).get(m, {}) if m in tested else {}
        for key in ("p", "p_holm", "reject"):
            entry[key] = src.get(key)
    notes = list(res.get("notes", []))
    notes.append(AB_ELSEWHERE_NOTE if scope == SCOPE_C_MAIN else C_ELSEWHERE_NOTE)
    n_without = sum(1 for r in ok if not str(r.get("animal") or "").strip())
    if n_without:
        notes.append(f"{n_without} axons without an animal ({NO_ANIMAL_GROUP}) left out: no p between genotypes")
    n_no_geno = sum(1 for r in ok if str(r.get("animal") or "").strip() and not str(r.get("genotype") or "").strip())
    if n_no_geno:
        notes.append(f"{n_no_geno} axons without a genotype left out")
    n_animals = len({str(r["animal"]).strip() for r in usable})
    if n_animals <= 1:
        res["descriptive_only"] = True
        notes.append(f"n_animals = {n_animals}: no comparison between animals is possible")
        for entry in res.get("measures", {}).values():
            entry["p"] = None
            entry["p_holm"] = None
            entry["reject"] = None
    if any(str(r["animal"]).strip() == PROVISIONAL_ANIMAL for r in usable):
        notes.append("the animal label is provisional")
    n_gate = sum(1 for r in rows if r.get("pd_status") == pdt.Status.OK.value and not _in_scope_row(r, scope))
    if n_gate:
        notes.append(f"{n_gate} finished axons left out by the quality gate of the contour ({scope})")
    res["notes"] = notes
    res["n_animals_total"] = n_animals
    res["scope"] = scope
    return res


def pooled_a(results: Sequence[pdt.PerimeterResult], *, group_by: Tuple[str, ...] = ("genotype",),
             axis: str = "nm", weights: str = "equal", n_null: Optional[int] = None,
             seed: Optional[int] = None, scope: str = SCOPE_MAIN,
             min_axons: int = MIN_POOLED_A_AXONS) -> Dict[str, pdt.PooledResult]:
    """Pooled question A per group (``mps_perimeter_a.pooled_test`` on the central slice of each ok axon whose
    contour passed the quality gate; ``scope`` as in ``summarise_by_animal``, the main set by default).

    v20 (R3 M3): a group with fewer than ``min_axons`` pooled axons (``MIN_POOLED_A_AXONS`` = 9, see there) gets
    the label ``n too small (n = k axons < 9): not interpreted; computed label '<label>'``; the numbers are kept
    and exported, and must not be read. ``min_axons=1`` gives the bare labels (validation code).

    Groups are the values of the identity fields in ``group_by`` joined by "/" (blank -> ""); the dictionary key
    is the group label ``pooled_test`` writes (``NO_ANIMAL_GROUP`` when nobody has an animal). Axons are sorted by
    ``axon_id`` first: the null draws one generator in profile order, so the order must not depend on the folder.
    ``n_null`` / ``seed`` default to those of the first axon's parameters.
    """
    ok = sorted((r for r in results if r.status == pdt.Status.OK and r.profiles is not None and _in_scope(r, scope)),
                key=lambda r: r.axon_id)
    groups: Dict[str, List[pdt.PerimeterResult]] = {}
    for r in ok:
        key = "/".join(_ident(r, g) for g in group_by) if any(_ident(r, g) for g in group_by) else ""
        groups.setdefault(key, []).append(r)
    out: Dict[str, pdt.PooledResult] = {}
    for key, rr in sorted(groups.items()):
        p = rr[0].params
        res = pa.pooled_test([r.profiles.by_slice[0] for r in rr if r.profiles is not None],
                             n_null=p.n_null if n_null is None else int(n_null),
                             seed=p.seed if seed is None else int(seed), axis=axis, weights=weights,
                             animals=[_ident(r, "animal") for r in rr], group=key,
                             rule=p.a_exclusion_rule, min_nm=p.a_exclusion_min_nm)
        out[res.profile.group or key] = n_too_small(res, min_axons)
    return out


def n_too_small(res: pdt.PooledResult, min_axons: int = MIN_POOLED_A_AXONS) -> pdt.PooledResult:
    """The pooled result with its label replaced by ``READING_N_TOO_SMALL`` when fewer than ``min_axons`` axons were
    pooled (the reading keeps its prefix, the computed label in quotes and every number after it)."""
    from dataclasses import replace
    n = int(res.n_axons)
    if n >= int(min_axons):
        return res
    text = str(res.reading)
    prefix = ""
    for pre in (pa.PREFIX_EXPLORATORY, pa.PREFIX_WITHIN_SAMPLE):
        if text.startswith(pre):
            prefix, text = pre, text[len(pre):]
            break
    label, _, rest = text.partition(pa.READING_SEPARATOR)
    new = (f"{prefix}{READING_N_TOO_SMALL} (n = {n} axons < {int(min_axons)}): not interpreted; computed label "
           f"'{label}'")
    return replace(res, reading=new + (pa.READING_SEPARATOR + rest if rest else ""))


def pooled_a_scopes(results: Sequence[pdt.PerimeterResult], **kw: Any) -> Dict[str, Dict[str, pdt.PooledResult]]:
    """``pooled_a`` for every reported scope (``scopes_for``): the main set, the set without slice misfit and one per
    curve stratum present among the included axons. A scope with no axon is left out, and so is the main scope of C
    (the slice criterion is for C only). ``kw`` as ``pooled_a``."""
    strata = [g["pd_curve_stratum"] for g in (pool_gate(r) for r in results) if g["pd_pool_included"]]
    out: Dict[str, Dict[str, pdt.PooledResult]] = {}
    for scope in scopes_for(strata):
        if scope == SCOPE_C_MAIN:
            continue
        res = pooled_a(results, scope=scope, **kw)
        if res:
            out[scope] = res
    return out


# ----------------------------------------------------------------------------------------------------
# Long tables (rows only)
# ----------------------------------------------------------------------------------------------------
def summarise_focus_by_roi(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Decision Q1: peak - focus (both foci, signed and absolute) per ROI and over all axons.

    Uses every axon whose peak was found (also failed analyses: the distance does not need the contour). Per
    group: n, mean, median, SD, min, max of each distance; None where no axon has it.
    """
    usable = [r for r in rows if _f(r.get("pd_z_peak_nm")) is not None]
    groups: Dict[str, List[Dict[str, Any]]] = {"(all axons)": usable}
    for r in usable:
        groups.setdefault(f"ROI {r.get('trace_label', 'NA').split('.')[1] if r.get('trace_label') else 'NA'}",
                          []).append(r)
    out: List[Dict[str, Any]] = []
    for name, rr in groups.items():
        row: Dict[str, Any] = {"group": name, "n_axons": len(rr)}
        for m in PEAK_FOCUS_COLUMNS:
            v = np.array([x for x in (_f(q.get(m)) for q in rr) if x is not None], dtype=np.float64)
            row[f"n_{m}"] = int(v.size)
            row[f"mean_{m}"] = float(v.mean()) if v.size else None
            row[f"median_{m}"] = float(np.median(v)) if v.size else None
            row[f"sd_{m}"] = float(v.std(ddof=1)) if v.size > 1 else None
            row[f"min_{m}"] = float(v.min()) if v.size else None
            row[f"max_{m}"] = float(v.max()) if v.size else None
        out.append(row)
    return out


def profile_rows(results: Sequence[pdt.PerimeterResult]) -> List[Dict[str, Any]]:
    """One row per axon x slice x bin: ``lag_nm`` on the straight axis centred on the central slice's maximum,
    counts, relative density, the measured flag, and (window starting at that bin) tortuosity, signed mean
    curvature and radial median of the central slice; ``radial_median_nm`` is that slice's own per bin."""
    out: List[Dict[str, Any]] = []
    for r in results:
        if r.profiles is None:
            continue
        dg = r.diagnostics
        for k in pdt.SLICES:
            p = r.profiles.by_slice[k]
            order = centred_order(p.n_bins)
            lags = centred_lags_nm(p.n_bins, p.bin_nm_effective)
            for j, lag in zip(order, lags):
                j = int(j)

                def at(arr: Optional[np.ndarray]) -> Optional[float]:
                    return None if arr is None or arr.size <= j else _f(arr[j])

                out.append({
                    "axon_id": r.axon_id, "trace_label": r.trace.label, "slice": k, "bin": j, "lag_nm": float(lag),
                    "count": int(p.counts[j]), "relative": float(p.relative[j]), "measured": bool(p.measured[j]),
                    "tau_window": None if dg is None else at(dg.tau_window),
                    "kappa_mean_per_nm": None if dg is None else at(dg.kappa_window_per_nm),
                    "window_radial_median_nm": None if dg is None else at(dg.window_radial_median_nm),
                    "radial_median_nm": None if dg is None else at(dg.radial_median_by_bin_nm.get(k)),
                })
    return out


def contour_rows(results: Sequence[pdt.PerimeterResult]) -> List[Dict[str, Any]]:
    """The contour used (counter-clockwise), x, y and arc per table row, per axon."""
    out: List[Dict[str, Any]] = []
    for r in results:
        if r.contour is None or r.contour.path is None:
            continue
        path = r.contour.path
        pts = np.asarray(path.points_nm)
        for i in range(pts.shape[0]):
            out.append({"axon_id": r.axon_id, "trace_label": r.trace.label, "row": i, "x_nm": float(pts[i, 0]),
                        "y_nm": float(pts[i, 1]), "arc_nm": float(path.cum_nm[i])})
    return out


def pooled_rows(pooled: Dict[str, pdt.PooledResult], scope: str = SCOPE_MAIN) -> List[Dict[str, Any]]:
    """The pooled A profile per group, plus the group's statistics and clean reading on every row, and the
    ``scope`` (which pooled set) last."""
    out: List[Dict[str, Any]] = []
    for name, res in pooled.items():
        pp = res.profile
        for i in range(pp.lag_nm.size):
            out.append({"group": name, "axis": pp.axis, "lag_nm": float(pp.lag_nm[i]),
                        "mean_relative": _f(pp.mean_relative[i]), "n_axons": int(pp.n_axons[i]),
                        "n_animals": int(pp.n_animals[i]), "null_lo": _f(pp.null_lo[i]), "null_hi": _f(pp.null_hi[i]),
                        "global_lo": _f(pp.global_lo[i]), "global_hi": _f(pp.global_hi[i]),
                        "pd_a_reading": a_reading_label(res), "reading_full": res.reading,
                        "contrast": _f(res.contrast.value), "contrast_z": _f(res.contrast.z),
                        "contrast_p": _f(res.contrast.p), "period": _f(res.period.value), "period_z": _f(res.period.z),
                        "period_p": _f(res.period.p), "n_axons_pooled": int(res.n_axons), "scope": scope})
    return out


def pooled_rows_scoped(scoped: Dict[str, Dict[str, pdt.PooledResult]]) -> List[Dict[str, Any]]:
    """``pooled_rows`` of every scope of ``pooled_a_scopes``, one after the other."""
    out: List[Dict[str, Any]] = []
    for scope, pooled in scoped.items():
        out.extend(pooled_rows(pooled, scope))
    return out


def by_animal_rows_scoped(rows: Sequence[Dict[str, Any]], scopes: Sequence[str]) -> List[Dict[str, Any]]:
    """``summarise_by_animal`` of every scope, one after the other.

    v20: when both the main scope and the main scope of C are listed, the Holm adjustment of each animal is re-done
    over the family of three (``pd_a_contrast_z`` and ``pd_b_z`` of the main scope, ``pd_c_z_m1_p1`` of the main scope
    of C) and written back into both rows, so that splitting C off does not loosen the family-wise control."""
    out: List[Dict[str, Any]] = []
    for scope in scopes:
        out.extend(summarise_by_animal(rows, scope))
    main = {r["animal"]: r for r in out if r.get("scope") == SCOPE_MAIN and r.get("animal") != NO_ANIMAL_GROUP}
    c_main = {r["animal"]: r for r in out if r.get("scope") == SCOPE_C_MAIN and r.get("animal") != NO_ANIMAL_GROUP}
    for animal, rm in main.items():
        rc = c_main.get(animal)
        family = [(rm, m) for m in AB_TESTED] + ([(rc, m) for m in C_TESTED] if rc is not None else [])
        ps = [row.get(f"p_{m}") for row, m in family]
        if all(p is None for p in ps):
            continue
        for (row, m), ph in zip(family, gb.holm(ps)):
            row[f"p_holm_{m}"] = ph
        for row in (rm, rc):
            if row is not None:
                row["note"] = (str(row.get("note") or "") + "; " + JOINT_HOLM_NOTE).lstrip("; ")
    return out


def row_scopes(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The scopes reported for a set of flat rows (``scopes_for`` of the strata of the included ones)."""
    return scopes_for([str(r.get("pd_curve_stratum") or "none") for r in rows
                       if r.get("pd_pool_included") and r.get("pd_status") == pdt.Status.OK.value])


def sensitivity_rows(results: Sequence[pdt.PerimeterResult]) -> List[Dict[str, Any]]:
    """Long table of the sensitivities: one row per axon x setting (main band, other bands, axial guards)."""
    out: List[Dict[str, Any]] = []

    def put(r: pdt.PerimeterResult, setting: str, value: float, a: Dict[str, pdt.TestReadout],
            b: Optional[pdt.BReadout], c: Dict[Tuple[int, int], pdt.TestReadout], leak: Optional[float]) -> None:
        row: Dict[str, Any] = {"axon_id": r.axon_id, "trace_label": r.trace.label, "setting": setting,
                               "value_nm": value, "c_leak_expected_r": _f(leak)}
        for key in ("contrast", "period"):
            t = a.get(key)
            row[f"a_{key}"], row[f"a_{key}_z"], row[f"a_{key}_p"] = ((None, None, None) if t is None
                                                                      else (_f(t.value), _f(t.z), _f(t.p)))
        row["b_spearman"], row["b_z"], row["b_p"] = ((None, None, None) if b is None
                                                    else (_f(b.test.value), _f(b.test.z), _f(b.test.p)))
        for pair, suffix in _PAIRS:
            t = c.get(pair)
            row[f"c_r_{suffix}"], row[f"c_z_{suffix}"], row[f"c_p_{suffix}"] = ((None, None, None) if t is None
                                                                               else (_f(t.value), _f(t.z), _f(t.p)))
        out.append(row)

    for r in results:
        if r.status != pdt.Status.OK or r.band is None:
            continue
        leak = None if r.diagnostics is None else r.diagnostics.leak_expected_r
        put(r, "band (main)", float(r.band.half_width_nm), r.a, r.b, r.c, leak)
        for hw, s in sorted(r.sensitivity.items()):
            put(r, "band", hw, s.a, s.b, s.c, leak)
        for g, gs in sorted(r.guard_sensitivity.items()):
            put(r, "axial guard", g, {}, None, gs.c, gs.leak_expected_r)
    return out


__all__ = ["perimeter_rows", "perimeter_row", "summarise_by_animal", "compare_genotypes", "pooled_a",
           "pooled_a_scopes", "pooled_rows_scoped", "by_animal_rows_scoped", "row_scopes", "pool_gate", "gate_rows",
           "gate_summary_rows", "axon_a_reading", "curve_stratum", "scopes_for", "SCOPE_MAIN", "SCOPE_NO_MISFIT",
           "SCOPE_CURVE_PREFIX",
           "a_reading_label", "summarise_focus_by_roi", "profile_rows", "contour_rows", "pooled_rows",
           "sensitivity_rows", "NO_ANIMAL_GROUP", "PROVISIONAL_ANIMAL", "Z_READOUTS",
           "TESTED_Z_READOUTS", "axon_c_reading", "C_SUFFIX_NOT_RELIABLE", "C_SUFFIX_NOT_ASSESSED", "C_SUFFIX_MISFIT",
           "C_SUFFIX_SLICES", "C_SUFFIX_LEAK_NOT_ABOVE", "C_SUFFIX_LEAK_LIMIT", "C_SUFFIX_LEAK_RANGE",
           "SCOPE_C_MAIN", "C_TESTED", "AB_TESTED",
           "tested_for_scope", "leak_column", "MIN_POOLED_A_AXONS", "READING_N_TOO_SMALL", "n_too_small"]
