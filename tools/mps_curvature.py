# -*- coding: utf-8 -*-
"""
Local curvature of the membrane at each cluster, normalised by the calibre (kappa R), for the question "does a
cluster's size depend on the local curvature of the perimeter?" (apparent-size simulation, scenario C;
``docs/apparent_size.md``).

The membrane is the periodic cubic P-spline of ``tools.mps_membrane.fit_membrane_from_localizations`` fitted to the
CLUSTERED localizations, with knots every ``CURVATURE_KNOT_SPACING_NM`` = 400 nm (decision 4a), started from the
P-spline through the cluster centroids at the same spacing (``tools.mps_membrane.fit_membrane``). The membrane the
pipeline null walks keeps its own 600 nm: the two answer different questions (the null wants the scatter below the
knot scale counted as scatter; the curvature wants every bend it can resolve).

Why 400 nm: on synthetic axons the curvature at 400 / 600 / 800 nm knots correlates with the true one 0.88 / 0.78 /
0.70 (Spearman) on thin axons (R 400-600 nm) and 0.90 / 0.95 / 0.96 on thick ones (R 1200-1600 nm); with concave
notches 0.75 / 0.66 / 0.60 on thin ones [simulation: SIZE_SIM sec. 6.1]. 400 nm is the best compromise; thin axons
stay poorly resolved, hence the label below.

Why kappa R and never raw kappa: thin axons have a large kappa AND may differ in everything else (NeNA, labelling,
fusions); with raw kappa and clusters as units 95-100 % false positives, with kappa R and the axon as unit 2-8 %
[simulation: SIZE_SIM sec. 6.2]. R = L / (2 pi) with L the length of the fitted membrane (kappa R = 1 on a circle).

Labels:
* ``R < CURVATURE_RESOLVED_MIN_RADIUS_NM`` = 600 nm: "curvature not resolved" (notches shorter than ~500 nm are lost
  in thin axons) [simulation: SIZE_SIM sec. 6.1, 7.3; decision 4a]. The values are still computed and exported;
  the label travels with them.
* concave stretches (kappa R < 0) are EXCLUDED from the primary analysis of question (b) and reported apart
  (decision 3a): a sectioning artefact changes density and shape there without changing size (15-77 % false
  positives in the simulation when not treated) [simulation: SIZE_SIM sec. 6.2-6.3].

Sign convention: + convex (centre of curvature inside the axon), whatever the orientation of the curve. No Qt.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from tools.mps_apparent_size import fisher_z, partial_spearman, spearman

__all__ = [
    "CURVATURE_KNOT_SPACING_NM", "CURVATURE_RESOLVED_MIN_RADIUS_NM", "CURVATURE_STEP_NM", "CURVATURE_STRIDE",
    "RESOLUTION_LEVELS", "CurvatureMembrane", "curvature_of_closed_curve", "fit_curvature_membrane",
    "membrane_from_curve", "SizeCurvature", "size_vs_curvature",
]

# Knots of the P-spline the curvature is read from [simulation: SIZE_SIM sec. 6.1; decision 4a].
CURVATURE_KNOT_SPACING_NM = 400.0
# Below this calibre (R = L / 2 pi) the curvature is labelled "not resolved" [simulation: SIZE_SIM sec. 6.1, 7.3;
# decision 4a].
CURVATURE_RESOLVED_MIN_RADIUS_NM = 600.0
# Resampling step (arc length) of the fitted curve before differentiation, and the stride of the central
# differences (baseline 2 x 5 = 10 nm) [derivation: the P-spline is smooth on the 400 nm knot scale, so a 10 nm
# baseline differentiates it with a truncation error ~ (10 nm)^2 kappa''' / 6, far below 1 % of kappa; the 2 nm
# table rows of the fit would amplify rounding instead].
CURVATURE_STEP_NM = 5.0
CURVATURE_STRIDE = 2

RESOLUTION_LEVELS: Tuple[str, ...] = ("resolved", "curvature not resolved", "not evaluated")


def curvature_of_closed_curve(points_nm: NDArray[np.float64], *, step_nm: float = CURVATURE_STEP_NM,
                              stride: int = CURVATURE_STRIDE
                              ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Signed curvature (1/nm, + convex) of a closed polyline: the curve is resampled uniformly in arc length every
    ``step_nm`` and differentiated by periodic central differences ``stride`` samples apart. Returns (resampled
    points (M, 2), their arc (M,), their curvature (M,), the closed length)."""
    p = np.asarray(points_nm, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        raise ValueError("curvature_of_closed_curve: a closed curve needs at least three points")
    seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    length = float(seg.sum())
    if not (length > 0):
        raise ValueError("curvature_of_closed_curve: zero-length curve")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    m = max(16, int(round(length / float(step_nm))))
    s = np.arange(m) * (length / m)
    closed = np.vstack([p, p[:1]])
    q = np.column_stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])])
    h = (length / m) * int(stride)
    fwd = np.roll(q, -int(stride), axis=0)
    bwd = np.roll(q, int(stride), axis=0)
    d1 = (fwd - bwd) / (2.0 * h)
    d2 = (fwd - 2.0 * q + bwd) / (h * h)
    speed = np.linalg.norm(d1, axis=1)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed, 1e-12) ** 3
    area = 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    if area == 0.0:
        raise ValueError("curvature_of_closed_curve: the curve encloses no area")
    return (np.asarray(q, dtype=np.float64), np.asarray(s, dtype=np.float64),
            np.asarray(kappa * math.copysign(1.0, area), dtype=np.float64), float(length))


@dataclass
class CurvatureMembrane:
    """The membrane the curvature is read from: ``points_nm`` (M, 2) resampled every ``CURVATURE_STEP_NM``, their
    arc ``arc_nm`` and signed curvature ``kappa_per_nm``; ``length_nm`` L; ``radius_nm`` R = L / (2 pi);
    ``source`` "localizations" (the P-spline fitted to the clustered localizations) or "centroids" (the P-spline
    through the centroids, the fallback when the first fit failed: a warning says why) or "given"."""
    points_nm: NDArray[np.float64]
    arc_nm: NDArray[np.float64]
    kappa_per_nm: NDArray[np.float64]
    length_nm: float
    knot_spacing_nm: float
    source: str
    warnings: List[str] = field(default_factory=list)

    @property
    def radius_nm(self) -> float:
        return self.length_nm / (2.0 * math.pi)

    @property
    def resolved(self) -> bool:
        return self.radius_nm >= CURVATURE_RESOLVED_MIN_RADIUS_NM

    @property
    def resolution(self) -> str:
        return RESOLUTION_LEVELS[0] if self.resolved else RESOLUTION_LEVELS[1]

    def at(self, xy: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """(arc, curvature 1/nm, kappa R) at the curve sample nearest each of ``xy`` (n, 2) (the projection; the
        5 nm sampling moves the arc by < 2.5 nm)."""
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] == 0:
            e = np.zeros(0)
            return e, e, e
        _, idx = cKDTree(self.points_nm).query(pts)
        kap = self.kappa_per_nm[idx]
        return self.arc_nm[idx], kap, kap * self.radius_nm

    def tangent_curve(self) -> NDArray[np.float64]:
        """The curve as a polyline, for the (tangent, normal) frame of ``tools.mps_apparent_size.cluster_sizes``."""
        return self.points_nm


def membrane_from_curve(points_nm: NDArray[np.float64], *, source: str = "given",
                        knot_spacing_nm: float = float("nan")) -> CurvatureMembrane:
    """A ``CurvatureMembrane`` from any closed curve (e.g. a true synthetic contour, or a curve fitted elsewhere)."""
    q, s, k, length = curvature_of_closed_curve(points_nm)
    return CurvatureMembrane(points_nm=q, arc_nm=s, kappa_per_nm=k, length_nm=length,
                             knot_spacing_nm=float(knot_spacing_nm), source=source)


def fit_curvature_membrane(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64],
                           centroids_nm: NDArray[np.float64], *,
                           knot_spacing_nm: float = CURVATURE_KNOT_SPACING_NM) -> CurvatureMembrane:
    """The 400 nm P-spline membrane of one ring and its curvature (module docstring).

    ``x_nm``, ``y_nm``: the CLUSTERED localizations (kept clusters only: the background must not pull the curve);
    ``centroids_nm``: their clusters' centroids (any order), for the initial curve. Raises ValueError with fewer
    than three centroids; when the localization fit fails, falls back to the centroid P-spline with a warning."""
    cen = np.asarray(centroids_nm, dtype=np.float64).reshape(-1, 2)
    if cen.shape[0] < 3:
        raise ValueError("fit_curvature_membrane: at least three cluster centroids are needed")
    from tools.mps_membrane import fit_membrane, fit_membrane_from_localizations
    initial = fit_membrane(cen, knot_spacing_nm=float(knot_spacing_nm))
    if initial.path is None:
        raise ValueError("fit_curvature_membrane: the centroid P-spline has no path")
    warnings: List[str] = []
    try:
        lm = fit_membrane_from_localizations(np.asarray(x_nm, dtype=np.float64), np.asarray(y_nm, dtype=np.float64),
                                             initial_path=initial.path, knot_spacing_nm=float(knot_spacing_nm))
        curve = np.asarray(lm.path.points_nm, dtype=np.float64)
        source = "localizations"
        warnings.extend(lm.warnings)
        if lm.n_self_crossings:
            raise ValueError("the localization membrane crosses itself")
    except ValueError as exc:
        curve = np.asarray(initial.path.points_nm, dtype=np.float64)
        source = "centroids"
        warnings.append(f"the localization P-spline failed ({exc}); curvature read from the centroid P-spline")
    q, s, k, length = curvature_of_closed_curve(curve)
    out = CurvatureMembrane(points_nm=q, arc_nm=s, kappa_per_nm=k, length_nm=length,
                            knot_spacing_nm=float(knot_spacing_nm), source=source, warnings=warnings)
    if not out.resolved:
        out.warnings.append(f"R = {out.radius_nm:.0f} nm < {CURVATURE_RESOLVED_MIN_RADIUS_NM:.0f} nm: curvature not "
                            "resolved (notches shorter than ~500 nm are lost at this calibre)")
    return out


@dataclass
class SizeCurvature:
    """Question (b) on ONE axon: Spearman of the content with kappa R on the CONVEX clusters (kappa R >= 0; the
    primary readout) and its Fisher z for the across-axon combination; the same on the concave clusters, reported
    apart; the partial Spearman of an apparent estimator (cloud SD) with kappa R given log N_loc (convex)."""
    content_rho: Optional[float]
    content_z: Optional[float]
    n_convex: int
    n_concave: int
    content_rho_concave: Optional[float]
    apparent_partial_rho: Optional[float]
    resolution: str


def size_vs_curvature(content: NDArray[np.float64], kappa_r: NDArray[np.float64], *,
                      apparent: Optional[NDArray[np.float64]] = None,
                      n_locs: Optional[NDArray[np.float64]] = None,
                      resolution: str = RESOLUTION_LEVELS[0]) -> SizeCurvature:
    """Question (b) within an axon (decisions 3a, 5a, 6a): primary = Spearman(content, kappa R) over the convex
    clusters; the concave ones apart; the apparent estimator partialled on log N_loc."""
    kr = np.asarray(kappa_r, dtype=np.float64).reshape(-1)
    c = np.asarray(content, dtype=np.float64).reshape(-1)
    convex = np.isfinite(kr) & (kr >= 0.0)
    concave = np.isfinite(kr) & (kr < 0.0)
    rho, n = spearman(c[convex], kr[convex])
    rho_cc, _ = spearman(c[concave], kr[concave])
    part: Optional[float] = None
    if apparent is not None and n_locs is not None:
        a = np.asarray(apparent, dtype=np.float64).reshape(-1)
        lg = np.log(np.maximum(np.asarray(n_locs, dtype=np.float64).reshape(-1), 1.0))
        part, _ = partial_spearman(a[convex], kr[convex], lg[convex])
    return SizeCurvature(content_rho=rho, content_z=fisher_z(rho, n), n_convex=int(convex.sum()),
                         n_concave=int(concave.sum()), content_rho_concave=rho_cc, apparent_partial_rho=part,
                         resolution=resolution)


def describe() -> Dict[str, str]:
    """The defaults of this module with their origin (for the docs and the parameter registry)."""
    return {
        "knot_spacing_nm": f"{CURVATURE_KNOT_SPACING_NM:g} [simulation: SIZE_SIM sec. 6.1; decision 4a]",
        "resolved_min_radius_nm": f"{CURVATURE_RESOLVED_MIN_RADIUS_NM:g} [simulation: SIZE_SIM sec. 6.1, 7.3; decision 4a]",
        "step_nm": f"{CURVATURE_STEP_NM:g} x stride {CURVATURE_STRIDE} [derivation]",
    }
