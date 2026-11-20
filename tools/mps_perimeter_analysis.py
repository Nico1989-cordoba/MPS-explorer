"""
Orchestration of the perimeter density analysis for one axon, without Qt (package P10).

Receives the localizations of the whole pick: no cut in z, no clustering. Does not call ``analyze_axon``.

What one call does, in order
----------------------------
1. Identity and readable label: the ``AxonIdentity`` given, or the one ``mps_identity.propose`` reads off the path;
   ``axon_id`` is the key of the per-axon table for a file with no ROI; ``mps_trace_id.trace_label`` gives the
   ``year.ROI.axon`` label and its warnings.
2. z0 (``mps_perimeter_slices``): both foci are always estimated (nominal from the file's calibration record,
   empirical from sx = sy) and both peak - focus distances are always reported; z0 is the PEAK by default, a focus
   when asked. A focus criterion without that focus stops here with ``no focus`` (never falls back to the peak).
3. Three contiguous slices around z0; each lateral slice reports its percentage of the central one; slices whose
   interval leaves the file's z range are named in the warnings (the analysis goes on).
4. The AUTOMATIC contour (``mps_auto_contour.auto_contour`` by default; the function is a parameter so a test or a
   comparison can inject another one; ``contour_method="drawn"`` uses a drawn trace instead). ONE contour per
   axon, fitted on the localizations of the three slices (label != OUTSIDE_LABEL). A rejected contour stops here with ``contour failed``. The contour is normalised to
   COUNTER-CLOCKWISE seen from +z (the convention of the profiles and of question B).
   The automatic contour is called through ``contour_details`` with the slice of every localization, so its quality
   gate includes the misfit between the slices (D7); its assessment (curve kind, reliability and reasons, slice
   misfit, support, trim) travels in ``extras["contour_assessment"]`` (``assessment_extras``). A drawn trace or a
   ``contour_fn`` curve goes through the same gate. The pooling reads that assessment: a contour the gate marks
   not reliable is left out of the pooled analyses with its reason, EXCEPT for the slice misfit, which is only
   reported (decision D12). Optional (off by default, D8): ``params.contour_trim_interior_nm`` leaves out of the fit
   the localizations deeper than that below P14's polar ring (``mps_unanchored.trim_interior``, which trims nothing
   unless the polar curve passes the STRICT rules of the automatic contour, user's decision 2026-11-17); the
   quality is still measured on all of them.
5. Projection, band and the three profiles on one grid (``mps_perimeter_density``, frame ``z0_max``).
6. Question A on the central slice (``mps_perimeter_a.axon_a_details``), question B on the central slice
   (``mps_perimeter_bc.axon_b_readout`` with the radial-offset control), question C on the three slices
   (``slice_correlations`` with the expected axial leak).
7. Sensitivity: the profiles are RECOMPUTED at every band half-width of ``params.band_sensitivity_nm`` (the
   not-measured mask depends on the band, so it is never reused) and A, B, C are read again; C is repeated with the
   axial guards of ``params.guard_sensitivity_nm`` (decision Q3).
8. Diagnostics: unsupported fraction of the contour (bins of the central slice not measured), radial median per slice and per bin, radial shift of the
   lateral slices, tortuosity, curvature and radial median per window.

Failure states: every failure returns a complete ``PerimeterResult`` with ``status`` and a one-sentence
``failure``; the fields computed before the failure are kept. No exception leaves ``analyze_perimeter`` or
``analyze_file``: an unexpected one is caught and reported as ``internal error`` with its text.

Pure functions apart from ``analyze_file`` (which reads one file). Numpy only. Origin tags as in the contract.
"""
from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_perimeter_a as pa
from tools import mps_perimeter_bc as bc
from tools import mps_perimeter_density as dens
from tools import mps_perimeter_slices as sl
from tools.mps_matching import SmoothPath
from tools.mps_perimeter_types import (OUTSIDE_LABEL, PD_VERSION, SLICES, AutoContour, BandSensitivity, BReadout,
                                       FocusEstimate, GuardSensitivity, PerimeterDiagnostics, PerimeterParams,
                                       PerimeterResult, SliceProfiles, SliceSet, Status, TestReadout, TraceLabel,
                                       Z0Choice, Z0Criterion)
from tools.mps_trace_id import trace_label, trace_warnings

if TYPE_CHECKING:
    from tools.mps_identity import AxonIdentity

# ----------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------
# [ad hoc] fewest finite localizations analysed. The contour needs >= 8 nodes and the profile >= 8 bins; below
# ~100 localizations an 8-bin profile holds ~12 counts per bin, nothing any of A, B or C can test.
MIN_LOCALIZATIONS: int = 100
# [contract, mps_identity] the ``roi`` text of a whole pick file: ``describe_roi(None)`` in
# ``tools.cluster_quality``. ``axon_id(source, "none")`` is the key the per-axon table gives such a file.
NO_ROI_TEXT: str = "none"
# [lit] the conventional level used to compare decisions across band half-widths (the flag only; no new p).
ALPHA: float = 0.05
# Column names of sx / sy: Picasso names, then ThunderSTORM's (``mps_io._THUNDERSTORM_ALIASES``). Only the zero
# crossing of sx - sy is used, so the unit (pixels or nm) does not matter as long as both share it.
_SX_NAMES: Tuple[str, ...] = ("sx", "sigma1 [nm]")
_SY_NAMES: Tuple[str, ...] = ("sy", "sigma2 [nm]")
_LPZ_NAMES: Tuple[str, ...] = ("lpz", "uncertainty_z [nm]")

ContourFn = Callable[..., AutoContour]


# ----------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------
def _column(columns: Mapping[str, NDArray[Any]], names: Sequence[str], n: int) -> Optional[NDArray[np.float64]]:
    lowered = {str(k).strip().lower(): k for k in columns}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            arr = np.asarray(columns[key], dtype=np.float64).reshape(-1)
            if arr.size == n:
                return arr
    return None


def _signed_area(points: NDArray[np.float64]) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def counter_clockwise(path: SmoothPath) -> Tuple[SmoothPath, bool]:
    """The same closed curve, counter-clockwise seen from +z; (path, True) when it had to be reversed.

    Row 0 stays row 0 (the arc origin does not move); the other rows are taken in reverse order, so arc s of the
    reversed path is arc L - s of the original. ``vertex_row`` is remapped.
    """
    pts = np.asarray(path.points_nm, dtype=np.float64)
    if pts.shape[0] < 3 or _signed_area(pts) >= 0.0:
        return path, False
    n = pts.shape[0]
    order = np.mod(-np.arange(n), n)
    new = pts[order]
    nxt = np.roll(new, -1, axis=0)
    edges = np.hypot(nxt[:, 0] - new[:, 0], nxt[:, 1] - new[:, 1]).astype(np.float64)
    cum = np.concatenate([[0.0], np.cumsum(edges)]).astype(np.float64)
    rows = np.mod(-np.asarray(path.vertex_row, dtype=np.intp), n).astype(np.intp)
    return SmoothPath(points_nm=new, vertex_row=rows, edges_nm=edges, cum_nm=cum, length_nm=float(cum[-1])), True


def _empty_z0(criterion: Z0Criterion, status: Status) -> Z0Choice:
    return Z0Choice(z0_nm=None, criterion=criterion, peak_nm=None, focus_nominal_nm=None, focus_empirical_nm=None,
                    peak_minus_focus_nm=None, peak_minus_focus_nominal_nm=None, peak_minus_focus_empirical_nm=None,
                    status=status)


def _identity_for(identity: Optional["AxonIdentity"], source_path: str) -> Tuple["AxonIdentity", List[str]]:
    from tools import mps_identity
    if identity is not None:
        return identity, []
    proposal = mps_identity.propose(source_path)
    return proposal.identity, [f"identity proposed from the path: {w}" for w in proposal.warnings]


def _median(values: NDArray[np.float64]) -> float:
    v = values[np.isfinite(values)]
    return float(np.median(v)) if v.size else float("nan")


def _radial_median_by_bin(s: NDArray[np.float64], r: NDArray[np.float64], length: float, n_bins: int,
                          origin: float) -> NDArray[np.float64]:
    out = np.full(n_bins, np.nan, dtype=np.float64)
    if s.size == 0:
        return out
    j = bc.arc_bin_index(s, length_nm=length, n_bins=n_bins, origin_arc_nm=origin)
    order = np.argsort(j, kind="stable")
    js, rs = j[order], r[order]
    cuts = np.flatnonzero(np.diff(js)) + 1
    for chunk_j, chunk_r in zip(np.split(js, cuts), np.split(rs, cuts)):
        out[int(chunk_j[0])] = float(np.median(chunk_r))
    return out


def _lpz(columns: Mapping[str, NDArray[Any]], n: int) -> Tuple[float, str]:
    lpz = _column(columns, _LPZ_NAMES, n)
    if lpz is not None:
        v = lpz[np.isfinite(lpz) & (lpz > 0)]
        if v.size:
            return float(np.median(v)), "median lpz of the file"
    return float(bc.LEAK_DEFAULT_LPZ_NM), "default (pilot lpz, no lpz column)"


def _window_bins(params: PerimeterParams) -> int:
    # [director] window 200 nm on 100 nm bins = 2 bins; the window is a whole number of bins of the grid
    return max(1, int(round(float(params.window_nm) / float(params.bin_nm))))


def _read_abc(path: SmoothPath, s: NDArray[np.float64], r: NDArray[np.float64], label: NDArray[np.int8],
              profiles: SliceProfiles, half: float, params: PerimeterParams, warnings: List[str],
              tag: str) -> Tuple[Dict[str, Any], Optional[BReadout], Dict[Tuple[int, int], TestReadout],
                                 NDArray[np.float64]]:
    """A details, B readout and C correlations of one set of profiles (one band half-width)."""
    p0 = profiles.by_slice[0]
    a_obj = pa.axon_a_details(p0, n_null=params.n_null, seed=params.seed, rule=params.a_exclusion_rule,
                              min_nm=params.a_exclusion_min_nm)
    # P7 returns the AxonADetails dataclass (R2 corrections); this module keeps working with the dict of fields
    a: Dict[str, Any] = {"contrast": a_obj.contrast, "period": a_obj.period, "zone": a_obj.zone,
                         "exclusion_nm": a_obj.exclusion_nm, "period_nm": a_obj.period_nm,
                         "period_at_edge": a_obj.period_at_edge, "usable": a_obj.usable}
    wb = _window_bins(params)
    sel0 = (label == 0) & np.isfinite(r) & (np.abs(r) <= half)
    wrm: NDArray[np.float64] = np.full(p0.n_bins, np.nan)
    b: Optional[BReadout] = None
    if p0.n_bins >= wb:
        wrm = bc.window_radial_median(s[sel0], r[sel0], length_nm=profiles.length_nm, n_bins=p0.n_bins,
                                      origin_arc_nm=p0.origin_arc_nm, window_bins=wb)
        b = bc.axon_b_readout(path, p0.counts, p0.measured, origin_arc_nm=p0.origin_arc_nm, window_bins=wb,
                              radial_median_nm=wrm, n_null=params.n_null, seed=params.seed)
    else:
        warnings.append(f"{tag}question B not computed: {p0.n_bins} bins for a {wb}-bin window")
    c = bc.slice_correlations(profiles, n_null=params.n_null, seed=params.seed)
    return a, b, c, wrm


def _significant(t: Optional[TestReadout]) -> Optional[bool]:
    if t is None or not math.isfinite(t.p):
        return None
    return bool(t.p < ALPHA)


def _sensitivity_flag(main_a: Dict[str, TestReadout], main_b: Optional[BReadout],
                      main_c: Dict[Tuple[int, int], TestReadout], sens: Dict[float, BandSensitivity]) -> str:
    """"consistent" when every decision at alpha (A contrast, A period, B, C(-1, +1)) is the same at every
    half-width; otherwise which ones change and where."""
    def decisions(a: Dict[str, TestReadout], b: Optional[BReadout],
                  c: Dict[Tuple[int, int], TestReadout]) -> Dict[str, Optional[bool]]:
        return {"A contrast": _significant(a.get("contrast")), "A period": _significant(a.get("period")),
                "B": _significant(None if b is None else b.test), "C(-1,+1)": _significant(c.get((-1, 1)))}

    base = decisions(main_a, main_b, main_c)
    changes: List[str] = []
    for hw in sorted(sens):
        other = decisions(sens[hw].a, sens[hw].b, sens[hw].c)
        for name, value in base.items():
            if other[name] != value:
                changes.append(f"{name} at {hw:g} nm")
    return "consistent" if not changes else "differs: " + ", ".join(changes)


# ----------------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------------
def analyze_perimeter(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], z_nm: NDArray[np.float64], *,
                      columns: Mapping[str, NDArray[Any]], info: Sequence[Mapping[str, Any]],
                      identity: Optional["AxonIdentity"], source_path: str,
                      params: PerimeterParams = PerimeterParams(),
                      drawn_contour: Optional[NDArray[np.float64]] = None,
                      contour_fn: Optional[ContourFn] = None, year: str = "") -> PerimeterResult:
    """One axon end to end; every failure state returns a complete result with the reason.

    ``columns`` / ``info``: the file's columns (sx, sy, lpz are read when present) and Picasso metadata chain (the
    calibration record of the nominal focus). ``identity`` None: proposed from ``source_path``. ``drawn_contour``
    (N, 2) nm: used as the contour when ``params.contour_method == "drawn"``, otherwise only compared with the
    automatic one (``diagnostics.drawn_vs_auto_nm``). ``contour_fn``: replaces
    ``tools.mps_auto_contour.auto_contour`` (same keyword signature); for tests and comparisons. ``year``: a year
    typed for the folder (wins over the file name in the trace label).
    """
    from tools import mps_identity

    criterion = params.z0_criterion
    warnings: List[str] = []
    try:
        ident, id_warn = _identity_for(identity, source_path)
    except Exception as exc:  # noqa: BLE001 - never a raw exception out of the orchestration
        ident, id_warn = mps_identity.AxonIdentity(), [f"identity could not be proposed: {exc}"]
    warnings.extend(id_warn)
    aid = mps_identity.axon_id(source_path, NO_ROI_TEXT)
    trace = trace_label(ident, source_path, year=year)
    t_warn = tuple(trace_warnings(trace))

    def result(status: Status, failure: str, **kw: Any) -> PerimeterResult:
        base: Dict[str, Any] = dict(axon_id=aid, trace=trace, params=params, z0=_empty_z0(criterion, status),
                                    slices=None, contour=None, band=None, profiles=None)
        base.update(kw)
        if status != Status.OK and "z0" in kw:
            base["z0"] = kw["z0"]
        return PerimeterResult(status=status, failure=failure, warnings=tuple(warnings), trace_warnings=t_warn,
                               source_path=str(source_path), identity=ident, **base)

    try:
        return _analyze(x_nm, y_nm, z_nm, columns=columns, info=info, params=params, drawn_contour=drawn_contour,
                        contour_fn=contour_fn, warnings=warnings, result=result)
    except Exception as exc:  # noqa: BLE001 - reported as a status, with its text
        return result(Status.ERROR, f"unexpected error: {type(exc).__name__}: {exc}")


def _analyze(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], z_nm: NDArray[np.float64], *,
             columns: Mapping[str, NDArray[Any]], info: Sequence[Mapping[str, Any]], params: PerimeterParams,
             drawn_contour: Optional[NDArray[np.float64]], contour_fn: Optional[ContourFn], warnings: List[str],
             result: Callable[..., PerimeterResult]) -> PerimeterResult:
    criterion = params.z0_criterion
    x_all = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y_all = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    z_all = np.asarray(z_nm, dtype=np.float64).reshape(-1)
    n_given = int(x_all.size)
    if not (x_all.size == y_all.size == z_all.size):
        return result(Status.INVALID_INPUT, f"x, y and z differ in length ({x_all.size}, {y_all.size}, {z_all.size})")
    finite = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(z_all)
    n_finite = int(finite.sum())
    diag: Dict[str, Any] = {"n_locs_given": n_given, "n_locs_finite": n_finite}
    if n_finite < n_given:
        warnings.append(f"{n_given - n_finite} localizations with a non-finite coordinate were left out")
    x, y, z = x_all[finite], y_all[finite], z_all[finite]
    cols = {k: np.asarray(v).reshape(-1)[finite] for k, v in columns.items()
            if np.asarray(v).reshape(-1).size == n_given}

    # ---- z0: both foci, both distances ----
    sx = _column(cols, _SX_NAMES, n_finite)
    sy = _column(cols, _SY_NAMES, n_finite)
    record = None
    try:
        from tools.mps_zcalibration import record_from_info
        record = record_from_info(info)
    except Exception as exc:  # noqa: BLE001 - a broken metadata chain only loses the nominal focus
        warnings.append(f"calibration record unreadable: {exc}")
    focus: FocusEstimate = sl.estimate_focus(z, sx, sy, record)
    warnings.extend(focus.warnings)
    z0 = sl.choose_z0(z, focus, criterion=criterion)
    if n_finite < MIN_LOCALIZATIONS:
        z0 = replace(z0, status=Status.TOO_FEW_LOCS)
        return result(Status.TOO_FEW_LOCS, f"{n_finite} finite localizations (fewer than {MIN_LOCALIZATIONS})",
                      z0=z0, focus=focus, diagnostics=PerimeterDiagnostics(**diag))
    if z0.status != Status.OK or z0.z0_nm is None:
        why = ("the z0 criterion asks for a focus this file does not have"
               if z0.status == Status.NO_FOCUS else "no z peak could be found")
        return result(z0.status if z0.status != Status.OK else Status.TOO_FEW_LOCS, why, z0=z0, focus=focus,
                      diagnostics=PerimeterDiagnostics(**diag))

    # ---- slices ----
    slices: SliceSet = sl.cut_slices(z, float(z0.z0_nm), thickness_nm=params.slice_thickness_nm,
                                     guard_nm=params.slice_guard_nm)
    truncated = tuple(k for k, cov in zip(SLICES, slices.z_coverage) if cov < 1.0)
    diag["z_truncated_slices"] = truncated
    for k, cov in zip(SLICES, slices.z_coverage):
        if cov < 1.0:
            warnings.append(f"z range truncates slice {k:+d} (coverage {cov:.2f}); its pairs are marked")
    common = dict(z0=z0, focus=focus, slices=slices)
    if slices.n_locs[1] == 0:
        return result(Status.EMPTY_PROFILE, "the central slice holds no localization", **common,
                      diagnostics=PerimeterDiagnostics(**diag))

    # ---- contour ----
    in_three = slices.label != OUTSIDE_LABEL
    x_fit, y_fit = x[in_three], y[in_three]
    slice_fit = np.asarray(slices.label[in_three])
    trim_extras: Dict[str, Any] = {}
    exclude: Optional[NDArray[np.bool_]] = None
    if params.contour_trim_interior_nm is not None and params.contour_method != "drawn":
        # P14 / D8: the trim of interior localizations, off by default; the fit only. The depth below the polar
        # ring comes from P14 (``trim_interior``, which trims nothing unless the polar curve passes the strict rules
        # of the automatic contour: folded membrane, notch, cloud leaving it, too few localizations); P3 receives it as ``exclude`` and still measures the quality on ALL of them.
        from tools.mps_unanchored import trim_interior
        trim = trim_interior(x_fit, y_fit, depth_nm=float(params.contour_trim_interior_nm))
        trim_extras = {"contour_trim_nm": float(params.contour_trim_interior_nm),
                       "contour_n_locs_trimmed": trim.n_trimmed, "contour_trim_status": trim.status}
        warnings.append(f"contour fitted after the trim of interior localizations: {trim.status}, "
                        f"{trim.n_trimmed} of {x_fit.size} left out of the fit")
        exclude = ~np.asarray(trim.keep, dtype=bool)
    contour, assessment = _fit_contour(x_fit, y_fit, params, drawn_contour, contour_fn, warnings,
                                       slice_label=slice_fit, exclude=exclude)
    trim_extras["contour_assessment"] = assessment
    common["contour"] = contour
    if contour.status != Status.OK or contour.path is None:
        why = "; ".join(contour.warnings) if contour.warnings else "the contour was rejected"
        return result(Status.CONTOUR_FAILED, f"contour failed: {why}", **common,
                      diagnostics=PerimeterDiagnostics(**diag), extras=dict(trim_extras))
    path, reversed_ = counter_clockwise(contour.path)
    if reversed_:
        warnings.append("the contour came clockwise and was reversed to counter-clockwise")
        contour = replace(contour, path=path)
        common["contour"] = contour
    diag["contour_reversed"] = reversed_
    length = float(path.length_nm)
    if drawn_contour is not None and params.contour_method != "drawn":
        try:
            d = np.asarray(drawn_contour, dtype=np.float64).reshape(-1, 2)
            _, rd = dens.project(path, d[:, 0], d[:, 1])
            diag["drawn_vs_auto_nm"] = float(np.mean(np.abs(rd)))
        except Exception as exc:  # noqa: BLE001 - the comparison is optional
            warnings.append(f"drawn trace not compared: {exc}")

    # ---- projection, band, profiles ----
    s, r = dens.project(path, x, y)
    label = slices.label
    band = dens.band_width(r[in_three], mode=params.band_mode, fixed_nm=params.band_half_width_nm)
    half = float(band.half_width_nm)
    common["band"] = band
    profiles = dens.slice_profiles(s, r, label, length, half_width_nm=half, bin_nm=params.bin_nm)
    common["profiles"] = profiles
    p0 = profiles.by_slice[0]
    in_band = np.isfinite(r) & (np.abs(r) <= half)
    diag["unsupported_fraction"] = float(1.0 - np.mean(p0.measured))
    med = tuple(_median(r[in_band & (label == k)]) for k in SLICES)
    diag["radial_median_by_slice_nm"] = med
    diag["radial_shift_nm"] = (med[0] - med[1], med[2] - med[1])
    diag["radial_median_by_bin_nm"] = {
        k: _radial_median_by_bin(s[in_band & (label == k)], r[in_band & (label == k)], length, p0.n_bins,
                                 p0.origin_arc_nm) for k in SLICES}
    lpz, lpz_src = _lpz(cols, n_finite)
    if lpz_src.startswith("default"):
        warnings.append(f"no lpz column: the expected axial leak uses the pilot lpz of {lpz:g} nm")
    leak = bc.leak_expected_r(lpz, params.slice_thickness_nm, params.slice_guard_nm)
    diag.update(lpz_nm=lpz, lpz_source=lpz_src, leak_expected_r=leak)
    if p0.n_locs == 0:
        return result(Status.EMPTY_PROFILE, f"no localization of the central slice within +-{half:g} nm of the "
                      "contour", **common, diagnostics=PerimeterDiagnostics(**diag))

    # ---- A, B, C at the main half-width ----
    a_det, b, c, wrm = _read_abc(path, s, r, label, profiles, half, params, warnings, "")
    a = {"contrast": a_det["contrast"], "period": a_det["period"]}
    wb = _window_bins(params)
    # The reading of C compares the neighbouring pairs with the leak of the axon's OWN slice fractions at 47 nm, with
    # the measured range 36-51 nm (user's decisions 2026-11-17), not with ``leak`` (three equal slices), which is still
    # reported.
    leaks_real = {sz: bc.leak_expected_r_pairs(slices.n_locs, sz, params.slice_thickness_nm, params.slice_guard_nm)
                  for sz in bc.LEAK_SIGMA_Z_ALL_NM}
    diag.update(a_exclusion_nm=float(a_det["exclusion_nm"]), a_period_nm=float(a_det["period_nm"]),
                a_period_at_edge=bool(a_det["period_at_edge"]), a_zone=tuple(int(v) for v in a_det["zone"]),
                a_usable=bool(a_det["usable"]), c_reading=bc.slice_reading_real_fractions(c, leaks_real),
                window_bins=wb,
                window_radial_median_nm=wrm)
    if not a_det["usable"]:
        warnings.append("question A not computable: the remainder outside the exclusion zone is too short")
    if p0.n_bins >= wb:
        diag["tau_window"] = bc.tortuosity_windows(path, n_bins=p0.n_bins, origin_arc_nm=p0.origin_arc_nm,
                                                   window_bins=wb)
        diag["kappa_window_per_nm"] = bc.window_curvature(path, n_bins=p0.n_bins, origin_arc_nm=p0.origin_arc_nm,
                                                          window_bins=wb)
    if b is not None and not b.resolved:
        warnings.append("question B: the axon is thinner than the resolved curvature radius (L / 2 pi < 600 nm)")

    # ---- band sensitivity: profiles recomputed, never the main mask ----
    sens: Dict[float, BandSensitivity] = {}
    for hw in params.band_sensitivity_nm:
        prof_hw = dens.slice_profiles(s, r, label, length, half_width_nm=float(hw), bin_nm=params.bin_nm)
        a_hw, b_hw, c_hw, _ = _read_abc(path, s, r, label, prof_hw, float(hw), params, warnings,
                                        f"band {float(hw):g} nm: ")
        sens[float(hw)] = BandSensitivity(half_width_nm=float(hw),
                                          a={"contrast": a_hw["contrast"], "period": a_hw["period"]}, b=b_hw, c=c_hw)
    diag["band_sensitivity_flag"] = _sensitivity_flag(a, b, c, sens)

    # ---- axial guard sensitivity of C (decision Q3) ----
    guards: Dict[float, GuardSensitivity] = {}
    for g in params.guard_sensitivity_nm:
        try:
            sl_g = sl.cut_slices(z, float(z0.z0_nm), thickness_nm=params.slice_thickness_nm, guard_nm=float(g))
            prof_g = dens.slice_profiles(s, r, sl_g.label, length, half_width_nm=half, bin_nm=params.bin_nm)
            guards[float(g)] = GuardSensitivity(
                guard_nm=float(g), n_locs=sl_g.n_locs,
                c=bc.slice_correlations(prof_g, n_null=params.n_null, seed=params.seed),
                leak_expected_r=bc.leak_expected_r(lpz, params.slice_thickness_nm, float(g)))
        except ValueError as exc:
            warnings.append(f"guard {float(g):g} nm not computed: {exc}")

    return result(Status.OK, "", **common, a=a, b=b, c=c, sensitivity=sens, guard_sensitivity=guards,
                  diagnostics=PerimeterDiagnostics(**diag),
                  extras={"leak_expected_r": leak, "tau_vs_offset": None if b is None else b.tau_vs_offset,
                          "pd_version": PD_VERSION, **trim_extras})


def _contour_failed(method: str, knot: float, why: str) -> AutoContour:
    nan = float("nan")
    return AutoContour(path=None, length_nm=nan, method=method, knot_nm=knot, residual_sd_nm=nan,
                       n_self_crossings=0, nodes_nm=np.zeros((0, 2)), polar_distance_nm=nan, star_shaped=False,
                       status=Status.CONTOUR_FAILED, warnings=(why,))


# The curve kind written for a contour that a caller-supplied ``contour_fn`` fitted (tests, comparisons): not one of
# the automatic contour's ``CURVE_KINDS``, so that it is never pooled as if the automatic contour had made it.
EXTERNAL_KIND: str = "external"
SLICE_MISFIT_PREFIX: str = ("SLICE MISFIT (reported only; the axon stays in the pooled analyses, decision D12): ")


def assessment_extras(details: Any) -> Dict[str, Any]:
    """The flat assessment of one contour (``mps_auto_contour.assessment_row``, keys ``contour_*`` without the
    ``pd_`` prefix) plus the two fields the pooling gate reads (decision D12):

    * ``contour_slice_misfit``: the misfit between the z slices (D7) flagged the contour. It is REPORTED only: it
      never keeps the axon out of the pooled analyses.
    * ``contour_gate_reasons``: every OTHER reason of the quality gate (fraction within +-100 nm, radial spread,
      persistent band offset, arc off the band), joined by " | "; any of them keeps the axon out of the pooled
      analyses. Empty when the contour passed them all.
    """
    from tools.mps_auto_contour import assessment_row
    row: Dict[str, Any] = dict(assessment_row(details))
    a = getattr(details, "assessment", None)
    q = None if a is None else a.quality
    slice_reason = q.slices.reason if (q is not None and q.slices is not None and q.slices.low) else ""
    row["contour_slice_misfit"] = bool(slice_reason)
    row["contour_gate_reasons"] = ("" if q is None else " | ".join(w for w in q.reasons if w != slice_reason))
    return row


def _assess_path(contour: AutoContour, x: NDArray[np.float64], y: NDArray[np.float64],
                 slice_label: Optional[NDArray[Any]], kind: str, n_excluded: int) -> Dict[str, Any]:
    """The same quality gate for a curve the automatic contour did not choose (drawn trace, ``contour_fn``)."""
    from tools import mps_auto_contour as ac
    if contour.status != Status.OK or contour.path is None:
        det = ac.ContourDetails(contour=contour, candidates=(), polar_path=None, n_nodes_initial=0,
                                assessment=ac.ContourAssessment(kind="none", failure_reason=ac.failure_reason(contour),
                                                                quality=None, support=None))
        return assessment_extras(det)
    q = ac.contour_quality(contour.path, x, y, slice_label=slice_label)
    sup = ac.contour_support(contour.path, x, y)
    det = ac.ContourDetails(contour=contour, candidates=(), polar_path=None, n_nodes_initial=0,
                            assessment=ac.ContourAssessment(kind=kind, failure_reason="", quality=q, support=sup,
                                                            reliable=not q.low_quality, n_excluded=int(n_excluded)))
    return assessment_extras(det)


def _gate_warning(assessment: Dict[str, Any]) -> Optional[str]:
    """The warning of the quality gate as the pooling applies it (D12), or None when the contour passed."""
    from tools.mps_auto_contour import LOW_QUALITY_PREFIX
    gate = str(assessment.get("contour_gate_reasons") or "")
    if gate:
        return LOW_QUALITY_PREFIX + "; ".join(gate.split(" | "))
    if assessment.get("contour_slice_misfit"):
        reasons = [w for w in str(assessment.get("contour_quality_reasons") or "").split(" | ") if w]
        return SLICE_MISFIT_PREFIX + "; ".join(reasons)
    return None


def _fit_contour(x: NDArray[np.float64], y: NDArray[np.float64], params: PerimeterParams,
                 drawn: Optional[NDArray[np.float64]], contour_fn: Optional[ContourFn],
                 warnings: List[str], *, slice_label: Optional[NDArray[Any]] = None,
                 exclude: Optional[NDArray[np.bool_]] = None) -> Tuple[AutoContour, Dict[str, Any]]:
    """The contour and its assessment (``assessment_extras``). ``x``, ``y``: the localizations of the three slices,
    ``slice_label`` their slice (-1 / 0 / +1, for the slice misfit of D7), ``exclude`` the optional interior trim
    (D8; only the automatic contour and a ``contour_fn`` honour it: a drawn trace is never trimmed)."""
    from tools.mps_auto_contour import LOW_QUALITY_PREFIX
    method, knot = params.contour_method, float(params.contour_knot_nm)
    n_ex = 0 if exclude is None else int(np.count_nonzero(exclude))
    if method == "drawn":
        if drawn is None:
            out = _contour_failed(method, knot, "contour method 'drawn' without a drawn trace")
            return out, _assess_path(out, x, y, slice_label, "drawn", 0)
        try:
            from tools.mps_matching import smooth_closed_path
            pts: NDArray[np.float64] = np.asarray(drawn, dtype=np.float64).reshape(-1, 2)
            if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            path = smooth_closed_path(pts)
        except Exception as exc:  # noqa: BLE001 - a bad trace is a failed contour
            out = _contour_failed(method, knot, f"the drawn trace could not be smoothed: {exc}")
            return out, _assess_path(out, x, y, slice_label, "drawn", 0)
        out = AutoContour(path=path, length_nm=float(path.length_nm), method="drawn", knot_nm=float("nan"),
                          residual_sd_nm=float("nan"), n_self_crossings=0, nodes_nm=pts,
                          polar_distance_nm=float("nan"), star_shaped=False, status=Status.OK,
                          warnings=("drawn trace: comparison option, not the automatic contour",))
        assessment = _assess_path(out, x, y, slice_label, "drawn", 0)
        gw = _gate_warning(assessment)
        if gw:
            warnings.append(f"contour: {gw}")
        return out, assessment
    if contour_fn is None:
        from tools.mps_auto_contour import contour_details
        try:
            det = contour_details(x, y, method=method, knot_nm=knot, drawn=drawn, exclude=exclude,
                                  slice_label=slice_label)
        except Exception as exc:  # noqa: BLE001 - any failure of the fit is a failed contour, with its text
            out = _contour_failed(method, knot, f"{type(exc).__name__}: {exc}")
            return out, _assess_path(out, x, y, slice_label, "none", n_ex)
        out, assessment = det.contour, assessment_extras(det)
    else:
        xk, yk = (x, y) if exclude is None else (x[~exclude], y[~exclude])
        try:
            out = contour_fn(xk, yk, method=method, knot_nm=knot, drawn=drawn)
        except NotImplementedError:
            out = _contour_failed(method, knot, "the automatic contour is not available in this build")
        except Exception as exc:  # noqa: BLE001 - any failure of the fit is a failed contour, with its text
            out = _contour_failed(method, knot, f"{type(exc).__name__}: {exc}")
        if not isinstance(out, AutoContour):
            out = _contour_failed(method, knot, "the contour function returned no AutoContour")
        assessment = _assess_path(out, x, y, slice_label, EXTERNAL_KIND, n_ex)
    if out.status == Status.OK:
        gw = _gate_warning(assessment)
        own = [w for w in out.warnings if not w.startswith(LOW_QUALITY_PREFIX)]
        warnings.extend(f"contour: {w}" for w in ([gw] if gw else []) + own)
    return out, assessment


def analyze_file(path: str, params: PerimeterParams = PerimeterParams(), *,
                 identity: Optional["AxonIdentity"] = None, animal: str = "", year: str = "",
                 drawn_contour: Optional[NDArray[np.float64]] = None,
                 contour_fn: Optional[ContourFn] = None,
                 pixel_size_nm: Optional[float] = None) -> PerimeterResult:
    """Load a localization file (``mps_io.load_localizations``) and analyse it.

    ``animal``: written into the identity when given (a provisional label for a folder known to be one animal).
    A file that cannot be read gives ``invalid input`` with the reason, never an exception.
    """
    from tools import mps_identity
    from tools.mps_io import load_localizations

    ident = identity
    try:
        if ident is None:
            ident = mps_identity.propose(path).identity
        if animal:
            ident = replace(ident, animal=str(animal))
    except Exception:  # noqa: BLE001 - analyze_perimeter proposes again and reports
        ident = None
    try:
        locs = load_localizations(path, pixel_size_nm=pixel_size_nm)
    except Exception as exc:  # noqa: BLE001 - an unreadable file is a status
        empty = np.zeros(0)
        res = analyze_perimeter(empty, empty, empty, columns={}, info=[], identity=ident, source_path=path,
                                params=params, contour_fn=contour_fn, year=year)
        return replace(res, status=Status.INVALID_INPUT, z0=replace(res.z0, status=Status.INVALID_INPUT),
                       failure=f"could not read {os.path.basename(path)}: {exc}")
    return analyze_perimeter(locs.x_nm, locs.y_nm, locs.z_nm, columns=locs.columns, info=locs.info,
                             identity=ident, source_path=path, params=params, drawn_contour=drawn_contour,
                             contour_fn=contour_fn, year=year)


__all__ = ["analyze_perimeter", "analyze_file", "counter_clockwise", "MIN_LOCALIZATIONS", "NO_ROI_TEXT",
           "assessment_extras", "EXTERNAL_KIND", "SLICE_MISFIT_PREFIX",
           "ContourFn", "TraceLabel"]
