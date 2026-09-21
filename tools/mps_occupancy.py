# -*- coding: utf-8 -*-
"""
Perimeter occupancy by betaII-spectrin clusters.

Implements parameter 6 of Gazal et al. (2026) -- the paper's most
consequential result, since it is what contradicts the classical picture of
a continuous spectrin ring:

  "the axonal perimeter was discretized into 10,000 equally spaced points at
   distances < 1 nm. Each perimeter point was then evaluated to determine
   whether it should be considered occupied by any of the detected clusters.
   [...] we computed the mean and covariance matrix of the (x, y) coordinates
   and fitted a Gaussian distribution whose maximum standard deviation was
   limited to one third of the largest distance from the cluster center to
   any of its points. [...] Points within a Mahalanobis distance of 3 sigma_max
   were considered to be 'occupied' by that cluster."

Reference result: ~20 % of the perimeter occupied, essentially constant
across axon calibres. The remaining ~80 % carries no betaII-spectrin.

Why the constraint is what it is
--------------------------------
Capping the largest standard deviation at d_max/3 makes the 3-sigma
occupancy ellipse coincide, at most, with the cluster's own measured
extent: 3 * (d_max / 3) = d_max. Without it a cluster with few
localizations and one outlier would claim a stretch of perimeter far
larger than the molecules it actually contains, inflating occupancy.

How far that protection reaches was measured rather than assumed, and it
is narrower than it sounds. For m - 1 localizations at one spot plus a
single outlier at distance R, sigma_max is R / sqrt(m) while the cap is
(m - 1)R / 3m, so the cap binds only for m < 10.9 -- confirmed
numerically at m = 10 (capped) and m = 11 (not). A lone outlier in any
cluster of eleven or more localizations passes through untouched, and
the claimed half-width peaks at 271 nm for a 300 nm outlier exactly
where the cap stops biting. The constraint keeps the headline ~20 % from
being an artefact of the SPARSEST clusters; what it does not do is
police a badly clustered large one. In the real April data, one cluster
of 4,500 localizations spanning 847 nm claims 14.0 % of its axon's
perimeter on its own -- a clustering problem, which no setting of this
module fixes.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

# Paper constants
PAPER_OCCUPANCY_PERCENT = 20.0
DEFAULT_N_PERIMETER_POINTS = 10_000
DEFAULT_MAHALANOBIS_THRESHOLD = 3.0
SIGMA_CAP_FRACTION = 1.0 / 3.0      # sigma_max <= d_max / 3

# Numerical floor for a standard deviation, in nm. A cluster whose
# localizations are collinear (or identical) yields a singular covariance
# that cannot be inverted; rather than dropping the cluster, its ellipse is
# widened to this minimum so it still occupies a small, honest stretch of
# perimeter. 1 nm is far below the ~20 nm localization precision, so this
# can only ever under-state occupancy, never inflate it.
_MIN_SIGMA_NM = 1.0


@dataclass
class ClusterEllipse:
    """The constrained 2D Gaussian standing in for one cluster's footprint."""

    label: int
    mean: NDArray[np.float64]           # (2,) centre in nm
    cov: NDArray[np.float64]            # (2, 2) constrained covariance
    inv_cov: NDArray[np.float64]        # (2, 2) precomputed inverse
    sigma_major_nm: float               # after the cap
    sigma_minor_nm: float
    sigma_major_raw_nm: float           # before the cap (for inspection)
    d_max_nm: float                     # largest centre-to-point distance
    was_capped: bool
    was_regularized: bool               # singular covariance widened


@dataclass
class OccupancyResult:
    """Perimeter occupancy for one axon."""

    occupancy_percent: float
    occupied_length_nm: float
    perimeter_length_nm: float
    n_points: int
    point_spacing_nm: float
    occupied_mask: NDArray[np.bool_]        # (n_points,) per sampled point
    perimeter_points: NDArray[np.float64]   # (n_points, 2)
    ellipses: List[ClusterEllipse]
    n_capped: int
    n_regularized: int
    mahalanobis_threshold: float
    warnings: List[str] = field(default_factory=list)

    @property
    def n_occupied_points(self) -> int:
        return int(np.count_nonzero(self.occupied_mask))


def fit_constrained_gaussian(
    points: NDArray[np.float64],
    label: int = -1,
    sigma_cap_fraction: float = SIGMA_CAP_FRACTION,
    mode: str = "clip",
) -> ClusterEllipse:
    """
    Fit the paper's constrained 2D Gaussian to one cluster's localizations.

    The empirical mean and covariance are computed, then the standard
    deviations are limited so that the largest does not exceed
    ``sigma_cap_fraction`` times the largest centre-to-point distance.

    Parameters
    ----------
    points : (n, 2) localizations of one cluster, in nm.
    label : cluster label, carried through for reporting.
    sigma_cap_fraction : 1/3 per the paper.
    mode : how to apply the cap when the empirical sigma exceeds it.
        - "clip" (default): clamp each standard deviation to the cap
          individually. This is the literal reading of "whose maximum
          standard deviation was limited to ...", and makes an
          over-extended ellipse rounder.
        - "scale": divide both axes by the same factor, preserving the
          cluster's measured anisotropy (elongation and orientation).

        The paper does not disambiguate, and the choice is not small:
        measured over the 18 real April axons, occupancy differs by a
        median 3.57 percentage points (2.36 to 6.61; every axon moves by
        more than one), which is 9.6 times the measurement's own
        bootstrap noise of 0.37 pp. It changes nothing else -- areas,
        r_eff, 1NN, perimeter and cluster counts are bit-identical -- and
        it does not change the ranking of axons (Spearman 0.998).

        "clip" stays the default on evidence, not on wording alone. Both
        modes set the major axis to exactly the cap; they differ only in
        the minor axis, which "scale" always shrinks (median 0.754 times
        "clip"'s) although the paper's constraint says nothing about it.
        Optimising the Gaussian log-likelihood over every covariance
        whose sigmas are all at most the cap reproduced "clip"'s sigmas
        in every case tested, at anisotropies from 1.2 to 8; "scale"
        matched only when the cap did not bind. "clip" is the
        constrained maximum-likelihood fit; "scale" is an ad-hoc shrink.

        The two are nearly degenerate with ``mahalanobis_threshold``:
        calibrated so each mode's claimed arc matches the arc its own
        localizations span, "clip" lands at 2.4 and "scale" at 2.8, and
        the two then agree to 0.39 pp with Spearman 1.000. So this is a
        second knob on one degree of freedom, which is why it is a
        keyword argument with a justified default and not a control --
        the threshold is the one to turn, and it is persisted, exported
        and documented.

    Returns
    -------
    ClusterEllipse
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (n, 2), got {pts.shape}")
    if len(pts) == 0:
        raise ValueError("Cannot fit a Gaussian to an empty cluster.")

    mean = pts.mean(axis=0)
    offsets = pts - mean
    d_max = float(np.max(np.linalg.norm(offsets, axis=1))) if len(pts) else 0.0

    # Empirical covariance. ddof=1 matches the usual sample covariance;
    # a single-point cluster has none, handled by the regularization below.
    if len(pts) >= 2:
        cov = np.cov(pts, rowvar=False, ddof=1)
    else:
        cov = np.zeros((2, 2))
    cov = np.asarray(cov, dtype=float).reshape(2, 2)

    # Symmetric eigendecomposition: eigenvalues are variances along the
    # principal axes, eigenvectors give the orientation.
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)          # kill negative round-off
    sigmas = np.sqrt(evals)                    # ascending (eigh convention)
    sigma_major_raw = float(sigmas[-1])

    cap = sigma_cap_fraction * d_max
    was_capped = False
    if cap > 0 and sigma_major_raw > cap:
        was_capped = True
        if mode == "clip":
            sigmas = np.minimum(sigmas, cap)
        elif mode == "scale":
            sigmas = sigmas * (cap / sigma_major_raw)
        else:
            raise ValueError(f"mode must be 'clip' or 'scale', got {mode!r}")

    # Regularize anything degenerate so the covariance stays invertible.
    was_regularized = bool(np.any(sigmas < _MIN_SIGMA_NM))
    sigmas = np.maximum(sigmas, _MIN_SIGMA_NM)

    cov_c = (evecs * (sigmas ** 2)) @ evecs.T
    inv_cov = np.linalg.inv(cov_c)

    return ClusterEllipse(
        label=int(label),
        mean=mean,
        cov=cov_c,
        inv_cov=inv_cov,
        sigma_major_nm=float(sigmas[-1]),
        sigma_minor_nm=float(sigmas[0]),
        sigma_major_raw_nm=sigma_major_raw,
        d_max_nm=d_max,
        was_capped=was_capped,
        was_regularized=was_regularized,
    )


def discretize_perimeter(
    contour: NDArray[np.float64],
    n_points: int = DEFAULT_N_PERIMETER_POINTS,
) -> Tuple[NDArray[np.float64], float, float]:
    """
    Sample a closed polygonal contour at equally spaced points.

    Parameters
    ----------
    contour : (K, 2) vertices in contour order (open; the closing edge back
        to the first vertex is added here).
    n_points : how many samples to place along the closed path.

    Returns
    -------
    (points, total_length_nm, spacing_nm) -- ``points`` has shape
    (n_points, 2), equally spaced by arc length.
    """
    contour = np.asarray(contour, dtype=float)
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(f"contour must be (K, 2), got {contour.shape}")
    if len(contour) < 3:
        raise ValueError(
            f"Need at least 3 vertices to close a contour, got {len(contour)}.")

    closed = np.vstack([contour, contour[:1]])
    seg = np.diff(closed, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 0:
        raise ValueError("Contour has zero length.")

    # endpoint=False: the closing point coincides with the start, so
    # including it would double-count one sample.
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(seg_len) - 1)

    seg_start = cum[idx]
    frac = np.where(seg_len[idx] > 0,
                    (targets - seg_start) / np.where(seg_len[idx] > 0,
                                                     seg_len[idx], 1.0),
                    0.0)
    pts = closed[idx] + frac[:, None] * seg[idx]
    return pts, total, total / n_points


def compute_occupancy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    labels: NDArray[np.int64],
    contour: NDArray[np.float64],
    exclude_labels: Optional[Set[int]] = None,
    n_points: int = DEFAULT_N_PERIMETER_POINTS,
    mahalanobis_threshold: float = DEFAULT_MAHALANOBIS_THRESHOLD,
    sigma_cap_fraction: float = SIGMA_CAP_FRACTION,
    ellipse_mode: str = "clip",
    ensure_subnm_spacing: bool = True,
) -> OccupancyResult:
    """
    Percentage of the reconstructed axonal perimeter covered by clusters.

    A sampled perimeter point counts as occupied when it lies within
    ``mahalanobis_threshold`` of ANY cluster's constrained Gaussian -- a
    union over clusters, so overlapping ellipses never double-count length.

    Parameters
    ----------
    x, y : localizations of the analysed axial slab, in nm.
    labels : DBSCAN labels for those localizations (-1 = noise, excluded).
    contour : (K, 2) reconstructed perimeter vertices, in contour order
        (``mps_geometry.reconstruct_perimeter(...).contour``).
    exclude_labels : clusters removed by the automatic curation.
    n_points : perimeter samples; the paper uses 10,000.
    mahalanobis_threshold : 3, per the paper.
    sigma_cap_fraction : 1/3, per the paper.
    ellipse_mode : "clip" or "scale" (see ``fit_constrained_gaussian``).
    ensure_subnm_spacing : the paper specifies BOTH 10,000 points AND a
        spacing below 1 nm, which only agree for perimeters up to 10 um.
        For longer perimeters, raise the sample count so the stated sub-nm
        spacing still holds (the occupied fraction is insensitive to this,
        but the sampling then matches the stated procedure). The adjustment
        is recorded in ``warnings``.

    Returns
    -------
    OccupancyResult
    """
    exclude = exclude_labels or set()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    warnings_: List[str] = []

    # ---------------- ellipses ------------------------------------------
    ellipses: List[ClusterEllipse] = []
    for label in np.unique(labels):
        if label == -1 or int(label) in exclude:
            continue
        mask = labels == label
        pts = np.column_stack([x[mask], y[mask]])
        ellipses.append(fit_constrained_gaussian(
            pts, label=int(label),
            sigma_cap_fraction=sigma_cap_fraction, mode=ellipse_mode))

    if not ellipses:
        raise ValueError(
            "No clusters left to compute occupancy (all excluded or noise).")

    # ---------------- perimeter sampling --------------------------------
    probe_pts, total_len, spacing = discretize_perimeter(contour, n_points)
    if ensure_subnm_spacing and spacing > 1.0:
        n_needed = int(np.ceil(total_len))          # 1 nm spacing
        warnings_.append(
            f"Perimeter is {total_len / 1000:.1f} um, so {n_points:,} samples "
            f"would sit {spacing:.2f} nm apart. Raised to {n_needed:,} samples "
            f"to keep the sub-nanometre spacing the method specifies."
        )
        probe_pts, total_len, spacing = discretize_perimeter(contour, n_needed)
        n_points = n_needed

    # ---------------- occupancy test ------------------------------------
    # Union over clusters: a point is occupied as soon as one ellipse
    # claims it. Evaluated cluster by cluster so memory stays O(n_points)
    # rather than O(n_points * n_clusters).
    occupied = np.zeros(len(probe_pts), dtype=bool)
    thresh_sq = float(mahalanobis_threshold) ** 2
    for e in ellipses:
        d = probe_pts - e.mean
        # Mahalanobis squared, vectorized: sum((d @ inv_cov) * d, axis=1)
        m2 = np.einsum("ij,jk,ik->i", d, e.inv_cov, d)
        occupied |= m2 <= thresh_sq

    occupied_len = float(np.count_nonzero(occupied)) * spacing
    occupancy_pct = 100.0 * float(np.count_nonzero(occupied)) / len(probe_pts)

    n_capped = sum(1 for e in ellipses if e.was_capped)
    n_reg = sum(1 for e in ellipses if e.was_regularized)
    if n_reg:
        warnings_.append(
            f"{n_reg} cluster(s) had a degenerate (collinear) covariance and "
            f"were widened to a {_MIN_SIGMA_NM:g} nm minimum standard "
            f"deviation so occupancy could still be evaluated."
        )

    return OccupancyResult(
        occupancy_percent=occupancy_pct,
        occupied_length_nm=occupied_len,
        perimeter_length_nm=total_len,
        n_points=len(probe_pts),
        point_spacing_nm=spacing,
        occupied_mask=occupied,
        perimeter_points=probe_pts,
        ellipses=ellipses,
        n_capped=n_capped,
        n_regularized=n_reg,
        mahalanobis_threshold=float(mahalanobis_threshold),
        warnings=warnings_,
    )
