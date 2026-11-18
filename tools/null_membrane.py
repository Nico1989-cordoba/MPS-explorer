# -*- coding: utf-8 -*-
"""
The membrane the per-axon pipeline null puts its molecules on (user decision C2, 2026-10-23) -- without Qt.

The question
------------
The pipeline null (``tools.mps_pipeline_null``) simulates molecules at random on the axon's membrane. Which membrane?
Before this module every method's null used a P-spline through ITS OWN centroids, so a method that leaves gaps
(HDBSCAN) or merges clusters drew a different, sometimes cut-corner membrane, and the nulls of two methods differed
by more than the method. The user decided (C2, hybrid (a) + (b)):

* default: the membrane through the centroids of DBSCAN 25/10 (the reference of every comparison, Gazal et al.
  2026), the SAME curve for every method's null (``MEMBRANE_DBSCAN``);
* option: the membrane fitted to the slab's LOCALIZATIONS (``MEMBRANE_LOCALIZATIONS``;
  ``tools.mps_membrane.fit_membrane_from_localizations``, started from the centroid curve), to compare, above all
  where the centroids deform what the localizations show;
* an automatic deformation indicator between the two curves (:func:`membrane_deformation`): mean and largest
  distance, and a warning per stretch of the contour.

For the DBSCAN 25/10 run itself the default membrane is exactly the curve the null used before (the P-spline through
its own centroids), so its null is unchanged.

Numbers and their origin
------------------------
* curve: 600 nm P-spline through the centroids in tour order (``tools.mps_membrane.radial_scatter_nm``,
  ``MEMBRANE_KNOT_SPACING_NM``) [regularity plan P-R80 (a)], as the null always drew it.
* stretch length of the indicator: 600 nm, the knot spacing [derivation: the curve cannot bend on a shorter scale,
  so a stretch is the smallest piece that can be wrong on its own].
* warning threshold: 50 nm = the half-width of the H2 band (``tools.mps_pipeline_null.BAND_HALF_WIDTH_NM``)
  [ad hoc: a membrane displaced by more than the band's half-width moves the simulated molecules out of the band
  that H2 counts them in]; 25 nm as the "worth a look" level [ad hoc: half of it].

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

MEMBRANE_DBSCAN = "dbscan_centroids"
MEMBRANE_LOCALIZATIONS = "localizations"
MEMBRANE_CHOICES: Tuple[str, ...] = (MEMBRANE_DBSCAN, MEMBRANE_LOCALIZATIONS)
MEMBRANE_DEFAULT = MEMBRANE_DBSCAN                      # [user decision C2, 2026-10-23]
MEMBRANE_WORDS = {MEMBRANE_DBSCAN: "centroids of DBSCAN 25/10 (common to every method)",
                  MEMBRANE_LOCALIZATIONS: "localizations of the slab"}

STRETCH_NM = 600.0             # [derivation: the 600 nm knot spacing, the smallest piece the curve bends on]
WARN_NM = 50.0                 # [ad hoc: the H2 band's half-width, tools.mps_pipeline_null.BAND_HALF_WIDTH_NM]
LOOK_NM = 25.0                 # [ad hoc: half of WARN_NM]

__all__ = ["MEMBRANE_DBSCAN", "MEMBRANE_LOCALIZATIONS", "MEMBRANE_CHOICES", "MEMBRANE_DEFAULT", "MEMBRANE_WORDS",
           "STRETCH_NM", "WARN_NM", "LOOK_NM", "check_choice", "centroid_curve", "localization_curve",
           "reference_analysis", "Deformation", "membrane_deformation", "NullMembrane", "null_membrane_for"]


def check_choice(choice: Optional[str]) -> str:
    c = MEMBRANE_DEFAULT if choice is None else str(choice)
    if c not in MEMBRANE_CHOICES:
        raise ValueError(f"null membrane must be one of {MEMBRANE_CHOICES}, got {choice!r}")
    return c


def centroid_curve(analysis: Any) -> Optional[FloatArray]:
    """The 600 nm P-spline through ``analysis``'s centroids in tour order (``perimeter.contour``), as the null has
    always drawn it; None without a contour of >= 3 clusters."""
    from tools.mps_membrane import MEMBRANE_KNOT_SPACING_NM, radial_scatter_nm
    per = getattr(analysis, "perimeter", None)
    if per is None or getattr(per, "contour", None) is None or len(per.contour) < 3:
        return None
    fit = radial_scatter_nm(np.asarray(per.contour, dtype=np.float64), knot_spacing_nm=MEMBRANE_KNOT_SPACING_NM)
    curve: FloatArray = np.asarray(fit.curve_nm, dtype=np.float64)
    return curve


def localization_curve(analysis: Any, initial_curve: FloatArray, n_samples: int = 400) -> FloatArray:
    """The membrane fitted to the slab's localizations (``fit_membrane_from_localizations``, robust periodic
    P-spline, 600 nm knots) started from ``initial_curve``; resampled at ``n_samples`` rows. Raises ValueError when
    it cannot be fitted."""
    from tools.mps_matching import smooth_closed_path
    from tools.mps_membrane import fit_membrane_from_localizations
    init = smooth_closed_path(np.asarray(initial_curve, dtype=np.float64))
    fit = fit_membrane_from_localizations(np.asarray(analysis.x_slab, dtype=np.float64),
                                          np.asarray(analysis.y_slab, dtype=np.float64), initial_path=init)
    pts: FloatArray = np.asarray(fit.path.points_nm, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("the localization membrane has fewer than 3 points")
    step = max(1, int(pts.shape[0] // max(int(n_samples), 3)))
    out: FloatArray = np.array(pts[::step], dtype=np.float64)
    return out


def reference_analysis(analysis: Any, runs: Sequence[Any] = ()) -> Optional[Any]:
    """The DBSCAN 25/10 analysis of this slab: ``analysis`` itself when it is one, else a run of ``runs`` that is
    the reference, else re-clustered here (``tools.multimethod.analyze_like``, a fraction of a second). None when it
    cannot be built."""
    from tools import multimethod as mm
    if analysis is None:
        return None
    method = str(getattr(analysis, "cluster_method", mm.REFERENCE_METHOD))
    if method == mm.REFERENCE_METHOD and float(getattr(analysis, "eps_nm", 0)) == float(mm.REFERENCE_PARAMS["eps"]) \
            and int(getattr(analysis, "min_samples", 0)) == int(mm.REFERENCE_PARAMS["min_samples"]):
        return analysis
    for r in runs:
        if getattr(r, "is_reference", False) and getattr(r, "analysis", None) is not None:
            return r.analysis
    try:
        return mm.analyze_like(analysis, analysis.x_slab, analysis.y_slab, analysis.z_slab, mm.REFERENCE_METHOD,
                               dict(mm.REFERENCE_PARAMS))
    except Exception:  # noqa: BLE001 - no reference: the caller falls back to the run's own centroids, said so
        return None


@dataclass
class Deformation:
    """How far the centroid membrane is from the localization membrane: per stretch of ``stretch_nm`` along the
    centroid curve, the largest distance; the mean and largest over the whole curve; the stretches beyond
    ``warn_nm`` (their start and end arcs, nm)."""
    mean_nm: float
    max_nm: float
    length_nm: float
    stretch_nm: float
    warn_nm: float
    stretch_max_nm: List[float] = field(default_factory=list)
    flagged: List[Tuple[float, float, float]] = field(default_factory=list)   # (s_start, s_end, max distance)

    @property
    def n_flagged(self) -> int:
        return len(self.flagged)

    def level(self) -> str:
        """"ok", "look" (some stretch beyond LOOK_NM) or "warn" (some stretch beyond warn_nm)."""
        if self.flagged:
            return "warn"
        return "look" if self.max_nm > LOOK_NM else "ok"

    def sentence(self) -> str:
        head = (f"Centroid vs localization membrane: mean {self.mean_nm:.0f} nm, largest {self.max_nm:.0f} nm over "
                f"{self.length_nm / 1000.0:.1f} um")
        if not self.flagged:
            return head + f"; no {self.stretch_nm:.0f} nm stretch beyond {self.warn_nm:.0f} nm."
        where = ", ".join(f"{a / 1000.0:.1f}-{b / 1000.0:.1f} um ({d:.0f} nm)" for a, b, d in self.flagged[:6])
        more = f" and {len(self.flagged) - 6} more" if len(self.flagged) > 6 else ""
        return (head + f"; WARNING: {len(self.flagged)} stretch(es) of {self.stretch_nm:.0f} nm beyond "
                f"{self.warn_nm:.0f} nm (the H2 band's half-width): {where}{more}. There the centroids deform the "
                "membrane the localizations show: compare the null on the localization membrane.")


def _distances_to_polyline(pts: FloatArray, curve: FloatArray) -> FloatArray:
    """Distance of every point to the closed polyline ``curve`` (exact, segment by segment)."""
    a = np.asarray(curve, dtype=np.float64)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    out = np.empty(pts.shape[0], dtype=np.float64)
    for i0 in range(0, pts.shape[0], 512):
        p = pts[i0:i0 + 512]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("ijk,jk->ij", ap, ab) / ab2[None, :], 0.0, 1.0)
        proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        out[i0:i0 + 512] = np.sqrt(np.min(np.sum((p[:, None, :] - proj) ** 2, axis=2), axis=1))
    return out


def membrane_deformation(centroid_nm: FloatArray, localization_nm: FloatArray, *, stretch_nm: float = STRETCH_NM,
                         warn_nm: float = WARN_NM, step_nm: float = 10.0) -> Deformation:
    """The distance from the centroid curve to the localization curve, sampled every ``step_nm`` along the centroid
    curve [ad hoc: 10 nm, far below the stretch], in stretches of ``stretch_nm``. Symmetric enough for a warning:
    the localization curve's own samples are checked too and the larger of the two directions is kept per stretch."""
    c = np.asarray(centroid_nm, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(localization_nm, dtype=np.float64).reshape(-1, 2)
    if c.shape[0] < 3 or m.shape[0] < 3:
        raise ValueError("membrane_deformation needs two closed curves of >= 3 points")
    seg = np.roll(c, -1, axis=0) - c
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    length = float(seg_len.sum())
    n = max(int(math.ceil(length / float(step_nm))), 3)
    s_samples = np.linspace(0.0, length, n, endpoint=False)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    idx = np.clip(np.searchsorted(cum, s_samples, side="right") - 1, 0, c.shape[0] - 1)
    frac = (s_samples - cum[idx]) / np.maximum(seg_len[idx], 1e-12)
    pts = c[idx] + frac[:, None] * seg[idx]
    d_cm = _distances_to_polyline(pts, m)
    # the other direction: each localization-curve point's distance to the centroid curve, assigned to the stretch
    # of its nearest centroid-curve sample
    d_mc = _distances_to_polyline(m, c)
    near = np.argmin(np.sum((m[:, None, :] - pts[None, :, :]) ** 2, axis=2), axis=1)
    n_str = max(int(math.ceil(length / float(stretch_nm))), 1)
    edges = np.linspace(0.0, length, n_str + 1)
    which = np.clip(np.searchsorted(edges, s_samples, side="right") - 1, 0, n_str - 1)
    smax = []
    flagged: List[Tuple[float, float, float]] = []
    for k in range(n_str):
        v = d_cm[which == k]
        w = d_mc[which[near] == k]
        top = float(max(v.max() if v.size else 0.0, w.max() if w.size else 0.0))
        smax.append(top)
        if top > float(warn_nm):
            flagged.append((float(edges[k]), float(edges[k + 1]), top))
    return Deformation(mean_nm=float(np.mean(d_cm)), max_nm=float(max(d_cm.max(), d_mc.max())), length_nm=length,
                       stretch_nm=float(stretch_nm), warn_nm=float(warn_nm), stretch_max_nm=smax, flagged=flagged)


@dataclass
class NullMembrane:
    """The curve the null of a run will use and what it is: ``curve_nm`` (None: the run's own centroids, the
    null's own default reading), ``choice``, ``source`` (the words of the null's origins), the deformation between
    the centroid and the localization curves when both could be built, and why something fell back."""
    choice: str
    curve_nm: Optional[FloatArray]
    source: str
    deformation: Optional[Deformation] = None
    note: str = ""


def null_membrane_for(analysis: Any, runs: Sequence[Any] = (), choice: Optional[str] = None,
                      with_deformation: bool = True) -> NullMembrane:
    """The membrane of the nulls of ``analysis``'s slab under ``choice`` (C2): the DBSCAN 25/10 centroid curve
    (default), or the localization curve started from it. With ``with_deformation`` the indicator between the two
    is computed as well (the localization fit costs a fraction of a second). Never raises: a curve that cannot be
    built falls back to the run's own centroids (``curve_nm`` None) with the reason in ``note``."""
    c = check_choice(choice)
    ref = reference_analysis(analysis, runs)
    cen = None if ref is None else centroid_curve(ref)
    note = ""
    if cen is None:
        note = "no DBSCAN 25/10 contour on this slab: each null uses its own method's centroids"
        return NullMembrane(c, None, "read: 600 nm P-spline through the observed centroids (P-R80 a)", None, note)
    loc: Optional[FloatArray] = None
    loc_note = ""
    if c == MEMBRANE_LOCALIZATIONS or with_deformation:
        try:
            loc = localization_curve(analysis, cen)
        except Exception as exc:  # noqa: BLE001 - the indicator / option says it could not be fitted
            loc_note = f"the localization membrane could not be fitted ({type(exc).__name__}: {exc})"
    deform = None
    if loc is not None:
        try:
            deform = membrane_deformation(cen, loc)
        except ValueError as exc:
            loc_note = f"deformation not computed: {exc}"
    if c == MEMBRANE_LOCALIZATIONS:
        if loc is None:
            return NullMembrane(c, cen, "read: 600 nm P-spline through the centroids of DBSCAN 25/10 (C2; the "
                                "localization membrane failed)", deform, loc_note)
        return NullMembrane(c, loc, "read: robust P-spline fitted to the slab's localizations (C2 option)", deform,
                            loc_note)
    return NullMembrane(c, cen, "read: 600 nm P-spline through the centroids of DBSCAN 25/10, common to every "
                        "method (C2)", deform, loc_note)
