# -*- coding: utf-8 -*-
"""
One axis for one axon: the direction along which its rings stack, the
tilt of that direction from the optical axis, and the rigid rotation
that puts z' along it.

Why this exists
---------------
The ring analysis cuts an axon into axial slabs -- one ring per dominant
component of the z distribution, membership by a hard mask on z. That is
only right if the rings are stacked along z. In a transverse section the
axon's own axis is a few degrees off the optical axis, and a tilt beta
does two very different things to those slabs (03_plan section 3.1,
02_investigacion section 2.2 B1):

  * consecutive rings shift laterally by P * tan(delta_beta), with P the
    period (~190 nm) and delta_beta the disagreement between ring
    planes. Under 10 nm for any plausible delta_beta: irrelevant;
  * within ONE ring, the far edge of the axon sits D * tan(beta) higher
    in z than the near edge, with D the diameter. For an 8 um axon at
    3 deg that is ~420 nm, more than two periods: a hard z mask then
    hands one side of the ring to the neighbour above and the other side
    to the neighbour below.

So the z-segmentation has to happen in a frame where z' runs along the
axon, which means the axis has to be estimated first, from the rings
themselves.

How the axis is estimated
-------------------------
Each ring with enough localizations gets its own plane z = a x + b y + c.
An ordinary least-squares plane comes first, then a robust refit
(``scipy.optimize.least_squares`` with a soft-L1 loss by default).
Background localizations in z do not know about the plane, so they are
NOT symmetric about it at the edges of a tilted ring: they pull a plain
least-squares slope towards zero. A robust loss bounds their influence.
Its scale is the MAD of the least-squares residuals -- a noise-only
estimate that the background barely moves -- floored at 10 nm so that
an unusually clean ring does not make the loss go L1 on its own noise.

The common axis is the localization-count-weighted mean of the ring
normals: the slope error of a plane fitted on n points falls as
1/sqrt(n), so n is the natural weight, and it also keeps a sparse ring
from steering the axis. delta_beta, the weighted RMS angle between each
ring normal and the axis, says how much the rings disagree with a single
axis; it is what P * tan(delta_beta) needs.

A per-ring fit needs ring labels, and the labels come from a
z-segmentation that is only right once the tilt is gone: for a wide,
tilted axon the two are circular. ``fit_axis_by_profile_sharpness``
breaks the circle without labels: it searches the tilt at which the
axial profile z' = z - a x - b y is sharpest (the collision probability
of its histogram, ``profile_sharpness``). Unlike a single plane through
all localizations, it is not fooled by rings whose lateral footprints
differ (a missing arc on one side of one ring and on the other side of
the next reads, to a pooled plane, as a tilt). It is a starting
estimate, accurate to a few tenths of a degree; the per-ring fit in that
frame then does the rest.

The frame
---------
``to_axon_frame`` applies p' = R (p - origin) + origin, with R the
Rodrigues rotation taking the axis onto z and origin the mean of the
localizations that entered the fits (a point on the axis). Rotating
about a point on the axis rather than about the lab origin keeps every
coordinate in the range it had, so nothing downstream sees numbers of a
different magnitude, and adding the origin back keeps z' comparable
with z. ``from_axon_frame`` is the exact inverse.

Everything here is deterministic: no random numbers anywhere, so two
runs on the same input are bit-identical.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

__all__ = [
    "AxonFrame",
    "DEFAULT_LOSS",
    "DEFAULT_MIN_LOCS",
    "DELTA_BETA_WARNING_DEG",
    "TILT_WARNING_DEG",
    "fit_axis_by_profile_sharpness",
    "fit_axon_frame",
    "from_axon_frame",
    "identity_axon_frame",
    "lateral_shift_per_period_nm",
    "profile_sharpness",
    "rodrigues_rotation",
    "to_axon_frame",
]

Z_HAT: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])

# A ring needs this many localizations to get a plane of its own. The
# slope error of a plane on n points at lateral spread s and axial noise
# sigma is ~ sigma / (sqrt(n) * s): for sigma = 50 nm and s = 900 nm,
# n = 200 gives ~0.2 deg, about the smallest tilt worth correcting.
DEFAULT_MIN_LOCS = 200
DEFAULT_LOSS = "soft_l1"
# The losses scipy.optimize.least_squares understands.
ROBUST_LOSSES: Tuple[str, ...] = ("linear", "soft_l1", "huber", "cauchy", "arctan")
# Robust scale: 1.4826 * MAD estimates sigma for Gaussian residuals; the
# floor keeps the loss quadratic over at least +-10 nm, so that a ring
# with tiny axial noise is not treated as if every point were an outlier.
MAD_TO_SIGMA = 1.4826
MIN_F_SCALE_NM = 10.0
# Warnings: a section cut more than 5 deg off the axon is unusual, and
# rings disagreeing by more than 1 deg means "one common axis" is a
# poor description of that axon.
TILT_WARNING_DEG = 5.0
DELTA_BETA_WARNING_DEG = 1.0
# Rotation algebra tolerances.
IDENTITY_TOLERANCE = 1e-9        # |a - b| below this: the rotation is the identity
ANTIPARALLEL_TOLERANCE = 1e-3    # |a + b| below this: go through -a (see rodrigues_rotation)
# A plane needs three non-collinear points; the lateral spread along the
# thinnest direction must be a non-negligible fraction of the widest.
MIN_LOCS_FOR_PLANE = 3
DEGENERATE_SINGULAR_RATIO = 1e-6

METHOD_PER_RING = "plane_per_ring"
METHOD_ALL = "plane_all"
METHOD_IDENTITY = "identity"
METHOD_SHARPNESS = "profile_sharpness"

# Axis from the sharpness of the axial profile (``fit_axis_by_profile_sharpness``).
# The statistic is the collision probability of the z' histogram, sum of
# p_i^2 over bins of SHARPNESS_BIN_NM: for a Gaussian ring of width sigma
# it is bin / (2 sigma sqrt(pi)), so the frame that maximises it is the
# one in which the rings are thinnest. Bins of 5 nm sit well below any
# axial precision (the discretisation bias is (bin / sigma)^2 / 12 < 0.2 %
# for sigma >= 35 nm) and still hold tens of localizations per bin for
# the smallest axon the harness uses.
SHARPNESS_BIN_NM = 5.0
# A residual tilt delta smears each ring uniformly over D * tan(delta),
# which halves the sharpness once D * tan(delta) ~ 3.5 sigma_z (210 nm
# for sigma_z 35 nm): the peak is > 2 deg wide (FWHM) for a 12 um axon,
# so a coarse step of 0.5 deg cannot straddle it; the fine step of 0.1 deg
# then locates it to a few tenths of a degree, which is inside the basin
# of the per-ring refinement (P / (2 D) = 0.45 deg at 12 um). Sections are
# cut within a few degrees of transverse (5 deg already warns), so the
# search stops at +-8 deg per lateral axis and warns when it ends there.
SHARPNESS_MAX_TILT_DEG = 8.0
SHARPNESS_COARSE_STEP_DEG = 0.5
SHARPNESS_FINE_STEP_DEG = 0.1
SHARPNESS_FINE_HALF_WIDTH_DEG = 0.6
# The search evaluates ~1300 candidate frames; 20000 localizations (every
# k-th one, deterministic) keep that under a second while the sampling
# noise of the statistic stays far below its variation across the grid.
SHARPNESS_MAX_LOCS = 20000


@dataclass
class AxonFrame:
    """
    The common axis of one axon and the rotation into its frame.

    Rows of ``ring_normals``, ``ring_n_locs`` and ``ring_used`` are
    indexed by ring LABEL (row k is label k), so a label absent from
    ``ring_index`` leaves a row of zero localizations, a NaN normal and
    ``ring_used`` False, and labels below zero appear nowhere. The arrays
    therefore have max(label) + 1 rows: labels are meant to be small
    consecutive integers (segment positions), not arbitrary ids.
    """

    u: NDArray[np.float64]              # (3,) unit axis vector, u[2] > 0
    origin_nm: NDArray[np.float64]      # (3,) point on the axis: mean of the localizations used
    beta_deg: float                     # angle between u and the z axis (tilt)
    delta_beta_deg: float               # weighted RMS angle ring normal vs u; NaN with < 2 rings
    ring_normals: NDArray[np.float64]   # (N, 3) per-ring unit normals, z > 0; NaN rows for rings not fitted
    ring_n_locs: NDArray[np.int64]      # (N,) localizations per ring label
    ring_used: NDArray[np.bool_]        # (N,) rings whose own plane entered the fit
    rotation: NDArray[np.float64]       # (3, 3) rotation @ u == [0, 0, 1]; identity if u == z
    method: str                         # "plane_per_ring" | "plane_all" | "profile_sharpness" | "identity"
    min_locs: int
    loss: str
    warnings: List[str] = field(default_factory=list)
    ring_f_scale_nm: NDArray[np.float64] = field(       # (N,) robust scale used per ring; NaN if not fitted
        default_factory=lambda: np.array([])
    )
    n_locs_used: int = 0                                # localizations that entered the fit(s)
    # Collision probability of the z' histogram (``profile_sharpness``) of
    # the localizations used, in this frame; NaN unless the frame came
    # from ``fit_axis_by_profile_sharpness`` or a caller that computed it.
    profile_sharpness: float = float("nan")

    @property
    def n_rings_used(self) -> int:
        """Rings whose own plane entered the common-axis fit."""
        return int(np.count_nonzero(self.ring_used))

    @property
    def is_identity(self) -> bool:
        """True when the frame leaves every coordinate exactly as it was."""
        return bool(np.array_equal(self.rotation, np.eye(3)))


# --------------------------------------------------------------- helpers
def _as_float_1d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64).ravel()


def _unit(v: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Unit vector along ``v``; a zero vector has no direction."""
    v = _as_float_1d(v)
    if v.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {v.shape}")
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must be a finite non-zero vector, got {v}")
    return np.asarray(v / norm, dtype=np.float64)


def _angle_deg(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """
    Angle between two vectors, in degrees.

    The atan2 form is used instead of arccos of the dot product because
    arccos loses half its digits near 0 deg -- exactly where a tilt of a
    fraction of a degree lives. Mathematically both are the same angle.
    """
    cross = np.cross(a, b)
    return math.degrees(math.atan2(float(np.linalg.norm(cross)), float(a @ b)))


def _half_turn(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Rotation by 180 deg about an axis perpendicular to unit ``a``.

    Any such rotation maps a onto -a; the axis is a x e for the
    coordinate axis e least aligned with a, which is never parallel to
    a and so always gives a well-defined perpendicular.
    """
    e = np.zeros(3)
    e[int(np.argmin(np.abs(a)))] = 1.0
    p = np.cross(a, e)
    p /= np.linalg.norm(p)
    return np.asarray(2.0 * np.outer(p, p) - np.eye(3), dtype=np.float64)


def rodrigues_rotation(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Proper rotation matrix taking unit vector ``a`` onto unit vector ``b``.

    Rodrigues' formula for the rotation about a x b by the angle between
    them: R = I + K + K^2 / (1 + a.b), K the cross-product matrix of
    a x b. Its rotation angle is exactly the angle between a and b, which
    is what makes the tilt of the frame equal to ``beta_deg``.

    Two cases the bare formula does not cover:

    * a within ``IDENTITY_TOLERANCE`` of b: the identity is returned
      exactly, so that an axis already along z produces a frame that
      changes nothing (not a rotation by 1e-10 deg).
    * a nearly antiparallel to b: 1 + a.b is then swamped by rounding.
      For exact antiparallel vectors a half-turn about any axis
      perpendicular to a is returned; otherwise the rotation is built as
      "half-turn taking a onto -a, then the (well-conditioned) rotation
      taking -a onto b". Neither can arise from ``fit_axon_frame``, whose
      axis always has u[2] > 0; they exist so that the function is a
      rotation for every pair of directions.

    Parameters
    ----------
    a, b : (3,) array_like
        Directions; they are normalised here.

    Returns
    -------
    (3, 3) float64 array
        R with R @ a == b, R.T @ R == I and det(R) == +1.
    """
    a_u = _unit(a, "a")
    b_u = _unit(b, "b")
    if float(np.linalg.norm(a_u - b_u)) < IDENTITY_TOLERANCE:
        return np.eye(3)
    a_plus_b = a_u + b_u
    sum_norm = float(np.linalg.norm(a_plus_b))
    if sum_norm < IDENTITY_TOLERANCE:
        return _half_turn(a_u)
    if sum_norm < ANTIPARALLEL_TOLERANCE:
        half = _half_turn(a_u)
        return np.asarray(rodrigues_rotation(-a_u, b_u) @ half, dtype=np.float64)
    v = np.cross(a_u, b_u)
    k = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    # 1 + a.b == |a + b|^2 / 2 for unit vectors; the right-hand side has
    # no cancellation, the left-hand side does when a.b is near -1.
    one_plus_c = 0.5 * float(a_plus_b @ a_plus_b)
    rot = np.eye(3) + k + (k @ k) / one_plus_c
    return np.asarray(rot, dtype=np.float64)


# ------------------------------------------------------------ plane fits
@dataclass
class _PlaneFit:
    """One plane z = a x + b y + c on centred coordinates."""

    normal: NDArray[np.float64]     # (3,) unit, z-component > 0
    a: float
    b: float
    c_nm: float
    f_scale_nm: float
    status: int                     # scipy least_squares status; > 0 means converged
    converged: bool


def _fit_plane_robust(
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: NDArray[np.float64],
    loss: str,
) -> Optional[_PlaneFit]:
    """
    Plane z = a x + b y + c: ordinary least squares, then a robust refit.

    Coordinates are centred on their means first. The plane's normal is
    translation-invariant, so this changes nothing in the answer, but it
    decouples the intercept from the slopes and keeps the design matrix
    well conditioned when x and y sit at tens of thousands of nm.

    Returns None when the design is degenerate: all points (laterally)
    on one line, or at one point, in which case a plane through them is
    undetermined and no normal is meaningful.
    """
    xc = x_nm - x_nm.mean()
    yc = y_nm - y_nm.mean()
    zc = z_nm - z_nm.mean()
    lateral = np.column_stack([xc, yc])
    singular = np.linalg.svd(lateral, compute_uv=False)
    if singular[0] <= 0.0 or singular[-1] <= DEGENERATE_SINGULAR_RATIO * singular[0]:
        return None
    design = np.column_stack([xc, yc, np.ones_like(xc)])
    coef_ols, _, rank, _ = np.linalg.lstsq(design, zc, rcond=None)
    if int(rank) < 3:
        return None
    coef_ols = np.asarray(coef_ols, dtype=np.float64)
    residual_ols = zc - design @ coef_ols
    mad = float(np.median(np.abs(residual_ols - np.median(residual_ols))))
    f_scale = max(MAD_TO_SIGMA * mad, MIN_F_SCALE_NM)

    def residual(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(design @ p - zc, dtype=np.float64)

    def jacobian(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return design

    result = least_squares(
        residual, coef_ols, jac=jacobian, loss=loss, f_scale=f_scale,
        method="trf", x_scale="jac",
    )
    status = int(result.status)
    converged = status > 0
    coef = np.asarray(result.x, dtype=np.float64) if converged else coef_ols
    a = float(coef[0])
    b = float(coef[1])
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    return _PlaneFit(normal=normal, a=a, b=b, c_nm=float(coef[2]) + float(z_nm.mean()),
                     f_scale_nm=f_scale, status=status, converged=converged)


def _identity_frame(
    ring_n_locs: NDArray[np.int64],
    origin_nm: NDArray[np.float64],
    *,
    min_locs: int,
    loss: str,
    warnings: List[str],
) -> AxonFrame:
    n_rings = int(ring_n_locs.shape[0])
    return AxonFrame(
        u=Z_HAT.copy(),
        origin_nm=np.asarray(origin_nm, dtype=np.float64),
        beta_deg=0.0,
        delta_beta_deg=float("nan"),
        ring_normals=np.full((n_rings, 3), np.nan),
        ring_n_locs=ring_n_locs,
        ring_used=np.zeros(n_rings, dtype=bool),
        rotation=np.eye(3),
        method=METHOD_IDENTITY,
        min_locs=min_locs,
        loss=loss,
        warnings=warnings,
        ring_f_scale_nm=np.full(n_rings, np.nan),
        n_locs_used=0,
    )


def identity_axon_frame(
    ring_index: Optional[NDArray[np.int64]] = None,
    origin_nm: Optional[NDArray[np.float64]] = None,
    *,
    min_locs: int = DEFAULT_MIN_LOCS,
    loss: str = DEFAULT_LOSS,
    warnings: Optional[List[str]] = None,
) -> AxonFrame:
    """
    A frame that changes nothing: u = z, identity rotation, beta = 0.

    For callers that skip the tilt correction on purpose and still want
    a frame to carry through the pipeline (``build_rings`` with
    ``correct_tilt=False``).

    Parameters
    ----------
    ring_index : int array, optional
        Ring label per localization; only used to fill ``ring_n_locs``.
    origin_nm : (3,) array, optional
        Point recorded as the origin; zeros if omitted. The identity
        transform does not use it.
    """
    if ring_index is None:
        counts = np.zeros(0, dtype=np.int64)
    else:
        labels = np.asarray(ring_index).astype(np.int64).ravel()
        labelled = labels[labels >= 0]
        n_rings = int(labelled.max()) + 1 if labelled.size else 0
        counts = np.bincount(labelled, minlength=n_rings).astype(np.int64)
    origin = np.zeros(3) if origin_nm is None else _as_float_1d(origin_nm)
    return _identity_frame(counts, origin, min_locs=min_locs, loss=loss,
                           warnings=list(warnings or []))


# ------------------------------------------------------------ the fit
def fit_axon_frame(
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: NDArray[np.float64],
    ring_index: NDArray[np.int64],
    *,
    min_locs: int = DEFAULT_MIN_LOCS,
    loss: str = DEFAULT_LOSS,
) -> AxonFrame:
    """
    Common axis of an axon from its rings, and the rotation onto it.

    Parameters
    ----------
    x_nm, y_nm, z_nm : (M,) arrays
        Localizations in nm.
    ring_index : (M,) int array
        Ring label per localization; labels < 0 mean "in no ring" and
        never enter any fit. Rows of the per-ring outputs are indexed by
        label value.
    min_locs : int
        Localizations a ring needs to get a plane of its own. Fewer and
        the ring is skipped with a warning.
    loss : str
        Loss of the robust refit (``scipy.optimize.least_squares``);
        "linear" makes it plain least squares.

    Returns
    -------
    AxonFrame
        See the class. ``method`` says which path was taken:

        * "plane_per_ring": at least one ring reached ``min_locs`` and
          gave a non-degenerate plane; u is the count-weighted mean of
          those normals, origin the mean of their localizations.
        * "plane_all": no ring did, but the ring localizations together
          number at least 3 and are not collinear: one plane on all of
          them. ``delta_beta_deg`` is NaN (nothing to disagree).
        * "identity": fewer than 3 ring localizations, or a degenerate
          design. u = z, rotation = I, beta = 0.

    Notes
    -----
    Warnings are strings in ``AxonFrame.warnings``, never exceptions:
    a tilt above ``TILT_WARNING_DEG``, a ``delta_beta_deg`` above
    ``DELTA_BETA_WARNING_DEG``, every ring skipped, every fallback, and
    localizations with non-finite coordinates (ignored). Exceptions are
    reserved for programming errors (mismatched lengths, unknown loss).
    """
    if loss not in ROBUST_LOSSES:
        raise ValueError(f"loss must be one of {ROBUST_LOSSES}, got {loss!r}")
    if min_locs < MIN_LOCS_FOR_PLANE:
        raise ValueError(f"min_locs must be >= {MIN_LOCS_FOR_PLANE}, got {min_locs}")
    x = _as_float_1d(x_nm)
    y = _as_float_1d(y_nm)
    z = _as_float_1d(z_nm)
    labels = np.asarray(ring_index).astype(np.int64).ravel()
    if not (x.shape == y.shape == z.shape == labels.shape):
        raise ValueError(
            "x_nm, y_nm, z_nm and ring_index must have the same length, got "
            f"{x.shape}, {y.shape}, {z.shape}, {labels.shape}"
        )
    warnings: List[str] = []

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n_bad = int(np.count_nonzero(~finite))
    if n_bad:
        warnings.append(f"{n_bad} localizations with non-finite coordinates ignored")
    in_ring = finite & (labels >= 0)
    labelled = labels[in_ring]
    n_rings = int(labelled.max()) + 1 if labelled.size else 0
    ring_n_locs = np.bincount(labelled, minlength=n_rings).astype(np.int64)
    ring_normals = np.full((n_rings, 3), np.nan)
    ring_used = np.zeros(n_rings, dtype=bool)
    ring_f_scale = np.full(n_rings, np.nan)

    # One plane per ring that has enough localizations. Only the labels
    # present are visited (ascending), so the cost follows the number of
    # rings, not the value of the largest label.
    for k in np.unique(labelled).tolist():
        n_k = int(ring_n_locs[k])
        if n_k < min_locs:
            warnings.append(
                f"ring {k} skipped: {n_k} localizations < min_locs={min_locs}"
            )
            continue
        m = in_ring & (labels == k)
        fit = _fit_plane_robust(x[m], y[m], z[m], loss)
        if fit is None:
            warnings.append(
                f"ring {k} skipped: degenerate design (localizations collinear), "
                "no plane can be fitted"
            )
            continue
        if not fit.converged:
            warnings.append(
                f"ring {k}: robust refit (loss={loss!r}) stopped with status "
                f"{fit.status}; least-squares plane kept"
            )
        ring_normals[k] = fit.normal
        ring_used[k] = True
        ring_f_scale[k] = fit.f_scale_nm

    if ring_used.any():
        used_labels = np.flatnonzero(ring_used)
        weights = ring_n_locs[used_labels].astype(np.float64)
        normals = ring_normals[used_labels]
        u = np.asarray((weights[:, None] * normals).sum(axis=0), dtype=np.float64)
        u /= np.linalg.norm(u)
        if u[2] < 0.0:                  # cannot happen (every normal has z > 0); kept as the stated convention
            u = -u
        used_mask = in_ring & np.isin(labels, used_labels)
        origin = np.array([x[used_mask].mean(), y[used_mask].mean(), z[used_mask].mean()])
        if used_labels.size >= 2:
            theta = np.array([_angle_deg(normals[i], u) for i in range(used_labels.size)])
            delta_beta = math.sqrt(float(np.sum(weights * theta ** 2) / np.sum(weights)))
        else:
            delta_beta = float("nan")
        method = METHOD_PER_RING
        n_used = int(np.count_nonzero(used_mask))
    else:
        n_labelled = int(np.count_nonzero(in_ring))
        origin = (np.array([x[in_ring].mean(), y[in_ring].mean(), z[in_ring].mean()])
                  if n_labelled else np.zeros(3))
        if n_labelled < MIN_LOCS_FOR_PLANE:
            warnings.append(
                f"only {n_labelled} localizations in rings (< {MIN_LOCS_FOR_PLANE}): "
                "identity frame, no tilt correction"
            )
            return _identity_frame(ring_n_locs, origin, min_locs=min_locs, loss=loss,
                                   warnings=warnings)
        fit = _fit_plane_robust(x[in_ring], y[in_ring], z[in_ring], loss)
        if fit is None:
            warnings.append(
                "degenerate design (all ring localizations collinear): identity frame, "
                "no tilt correction"
            )
            return _identity_frame(ring_n_locs, origin, min_locs=min_locs, loss=loss,
                                   warnings=warnings)
        warnings.append(
            f"no ring reaches min_locs={min_locs}: one plane fitted on all "
            f"{n_labelled} ring localizations (method '{METHOD_ALL}'); "
            "delta_beta undefined"
        )
        if not fit.converged:
            warnings.append(
                f"robust refit (loss={loss!r}) stopped with status {fit.status}; "
                "least-squares plane kept"
            )
        u = fit.normal
        delta_beta = float("nan")
        method = METHOD_ALL
        n_used = n_labelled

    beta = _angle_deg(u, Z_HAT)
    rotation = rodrigues_rotation(u, Z_HAT)
    if beta > TILT_WARNING_DEG:
        warnings.append(
            f"axis tilt beta = {beta:.2f} deg exceeds {TILT_WARNING_DEG:g} deg: "
            "check the section and the ring labels before trusting the frame"
        )
    if np.isfinite(delta_beta) and delta_beta > DELTA_BETA_WARNING_DEG:
        warnings.append(
            f"rings disagree on the axis: delta_beta = {delta_beta:.2f} deg exceeds "
            f"{DELTA_BETA_WARNING_DEG:g} deg (one common axis describes this axon poorly)"
        )
    return AxonFrame(
        u=u,
        origin_nm=np.asarray(origin, dtype=np.float64),
        beta_deg=beta,
        delta_beta_deg=delta_beta,
        ring_normals=ring_normals,
        ring_n_locs=ring_n_locs,
        ring_used=ring_used,
        rotation=rotation,
        method=method,
        min_locs=min_locs,
        loss=loss,
        warnings=warnings,
        ring_f_scale_nm=ring_f_scale,
        n_locs_used=n_used,
    )


# ------------------------------------------------ axis from the profile
def profile_sharpness(
    z_nm: NDArray[np.float64], bin_nm: float = SHARPNESS_BIN_NM
) -> float:
    """
    Sharpness of an axial profile: the probability that two localizations
    drawn at random fall in the same bin of a histogram of ``z_nm``.

    For a single Gaussian ring of width sigma this is bin / (2 sigma
    sqrt(pi)); for several rings it is the count-weighted sum of such
    terms. It grows as the rings get thinner and is invariant to where
    the rings are, how many there are or how far apart, so it can rank
    frames of the same axon without ring labels: the frame in which the
    rings are planar is the frame in which they are thinnest.

    Parameters
    ----------
    z_nm : (M,) array
        Axial coordinate per localization; non-finite values are ignored.
    bin_nm : float
        Histogram bin width. The statistic scales with it, so only values
        computed with the same bin are comparable.

    Returns
    -------
    float
        sum_i p_i^2 over the bins; 0.0 with no finite value.
    """
    z = _as_float_1d(z_nm)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0
    counts = np.bincount(np.floor((z - z.min()) / float(bin_nm)).astype(np.int64))
    p = counts / float(z.size)
    return float(np.dot(p, p))


def _sharpness_rows(
    z: NDArray[np.float64], bin_nm: float
) -> NDArray[np.float64]:
    """``profile_sharpness`` of every row of a (R, M) array of z' values."""
    lo = z.min(axis=1, keepdims=True)
    idx = np.floor((z - lo) / bin_nm).astype(np.int64)
    out = np.empty(z.shape[0], dtype=np.float64)
    n = float(z.shape[1])
    for r in range(z.shape[0]):
        p = np.bincount(idx[r]) / n
        out[r] = float(np.dot(p, p))
    return out


def fit_axis_by_profile_sharpness(
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: NDArray[np.float64],
    *,
    max_tilt_deg: float = SHARPNESS_MAX_TILT_DEG,
    coarse_step_deg: float = SHARPNESS_COARSE_STEP_DEG,
    fine_step_deg: float = SHARPNESS_FINE_STEP_DEG,
    fine_half_width_deg: float = SHARPNESS_FINE_HALF_WIDTH_DEG,
    bin_nm: float = SHARPNESS_BIN_NM,
    max_locs: int = SHARPNESS_MAX_LOCS,
) -> AxonFrame:
    """
    The axis along which the axial profile is sharpest, without ring labels.

    A plane through ALL localizations reads any difference in lateral
    footprint between rings as a tilt: two consecutive rings with
    incomplete arcs on opposite sides shift the cloud's centre with z,
    and the pooled regression turns that shift into a slope (1.7 deg on
    an 8 um axon that is not tilted at all). A per-ring plane needs ring
    labels, and the labels need the tilt removed first once D * tan(beta)
    exceeds half a period. This estimator needs neither: for each trial
    tilt (a, b) it forms z' = z - a x - b y and measures how sharp the
    axial profile is (``profile_sharpness``); a partial ring is still a
    thin peak at its own z', so a missing arc changes nothing. The search
    is a coarse grid over +-``max_tilt_deg`` per lateral axis followed by
    a fine grid around the coarse optimum; the shear z - a x - b y differs
    from the rigid rotation by O(beta^2) laterally, which is irrelevant
    for a starting estimate that a per-ring fit then refines.

    Parameters
    ----------
    x_nm, y_nm, z_nm : (M,) arrays
        Localizations in nm; non-finite ones are ignored with a warning.
    max_tilt_deg, coarse_step_deg, fine_step_deg, fine_half_width_deg : float
        Search range and steps per lateral slope axis, in degrees.
    bin_nm : float
        Histogram bin of ``profile_sharpness``.
    max_locs : int
        Every k-th localization is used, k the smallest stride that keeps
        at most this many (deterministic); the returned
        ``profile_sharpness`` is then recomputed on every localization.

    Returns
    -------
    AxonFrame
        ``method`` "profile_sharpness", u the normal of the best (a, b),
        origin the mean of the finite localizations, no per-ring rows
        (``ring_normals`` has shape (0, 3)), ``delta_beta_deg`` NaN, and
        ``profile_sharpness`` the statistic at the returned frame.
        Ties on the grid go to the smaller tilt, so a profile that no
        tilt can sharpen (all localizations at one lateral point) gives
        beta = 0. Fewer than 3 finite localizations, or a laterally
        collinear design, give the identity frame with a warning; a best
        tilt on the border of the coarse grid is warned about, because
        the true tilt may lie beyond the search range.
    """
    x = _as_float_1d(x_nm)
    y = _as_float_1d(y_nm)
    z = _as_float_1d(z_nm)
    if not (x.shape == y.shape == z.shape):
        raise ValueError(
            f"x_nm, y_nm and z_nm must have the same length, got {x.shape}, "
            f"{y.shape}, {z.shape}"
        )
    warnings: List[str] = []
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n_bad = int(np.count_nonzero(~finite))
    if n_bad:
        warnings.append(f"{n_bad} localizations with non-finite coordinates ignored")
    x, y, z = x[finite], y[finite], z[finite]
    n = int(x.size)
    no_rings = np.zeros(0, dtype=np.int64)
    origin = (np.array([x.mean(), y.mean(), z.mean()], dtype=np.float64)
              if n else np.zeros(3))
    loss = "none"
    if n < MIN_LOCS_FOR_PLANE:
        warnings.append(
            f"only {n} finite localizations (< {MIN_LOCS_FOR_PLANE}): identity "
            "frame, no tilt correction"
        )
        return _identity_frame(no_rings, origin, min_locs=MIN_LOCS_FOR_PLANE,
                               loss=loss, warnings=warnings)
    stride = max(1, int(math.ceil(n / float(max_locs))))
    xs, ys, zs = x[::stride], y[::stride], z[::stride]
    xc = xs - xs.mean()
    yc = ys - ys.mean()
    singular = np.linalg.svd(np.column_stack([xc, yc]), compute_uv=False)
    if singular[0] <= 0.0 or singular[-1] <= DEGENERATE_SINGULAR_RATIO * singular[0]:
        warnings.append(
            "degenerate design (localizations laterally collinear): identity frame, "
            "no tilt correction"
        )
        return _identity_frame(no_rings, origin, min_locs=MIN_LOCS_FOR_PLANE,
                               loss=loss, warnings=warnings)

    def search(a0: float, b0: float, half_deg: float, step_deg: float
               ) -> Tuple[float, float, float, bool]:
        """Best (sharpness, a, b) on the grid a0 + i s, b0 + j s, |i|, |j| <= k;
        ties go to the point nearest (a0, b0); also whether the best sits
        on the border of the grid."""
        k = int(round(half_deg / step_deg))
        offsets = np.arange(-k, k + 1, dtype=np.float64) * math.tan(math.radians(step_deg))
        a_values = a0 + offsets
        b_values = b0 + offsets
        best_s = -1.0
        best_a, best_b = a0, b0
        best_rank = math.inf          # |i| + |j| of the best, for the tie rule
        for i, a in enumerate(a_values.tolist()):
            z_rows = zs[None, :] - a * xc[None, :] - b_values[:, None] * yc[None, :]
            s = _sharpness_rows(z_rows, bin_nm)
            top = float(s.max())
            for j in np.flatnonzero(s == top).tolist():
                rank = abs(i - k) + abs(j - k)
                if top > best_s or (top == best_s and rank < best_rank):
                    best_s, best_a, best_b, best_rank = top, float(a), float(b_values[j]), rank
        on_border = (abs(best_a - a0) >= abs(offsets[-1]) - 1e-15
                     or abs(best_b - b0) >= abs(offsets[-1]) - 1e-15)
        return best_s, best_a, best_b, on_border

    _, a, b, on_border = search(0.0, 0.0, max_tilt_deg, coarse_step_deg)
    if on_border:
        warnings.append(
            f"profile sharpness: best tilt on the border of the +-{max_tilt_deg:g} deg "
            "search range; the true tilt may lie beyond it (frame not trusted)"
        )
    _, a, b, _ = search(a, b, fine_half_width_deg, fine_step_deg)
    u = np.array([-a, -b, 1.0], dtype=np.float64)
    u /= np.linalg.norm(u)
    beta = _angle_deg(u, Z_HAT)
    rotation = rodrigues_rotation(u, Z_HAT)
    z_p = ((np.column_stack([x, y, z]) - origin) @ rotation.T)[:, 2]
    if beta > TILT_WARNING_DEG:
        warnings.append(
            f"axis tilt beta = {beta:.2f} deg exceeds {TILT_WARNING_DEG:g} deg: "
            "check the section and the ring labels before trusting the frame"
        )
    return AxonFrame(
        u=u,
        origin_nm=origin,
        beta_deg=beta,
        delta_beta_deg=float("nan"),
        ring_normals=np.full((0, 3), np.nan),
        ring_n_locs=no_rings,
        ring_used=np.zeros(0, dtype=bool),
        rotation=rotation,
        method=METHOD_SHARPNESS,
        min_locs=MIN_LOCS_FOR_PLANE,
        loss=loss,
        warnings=warnings,
        ring_f_scale_nm=np.full(0, np.nan),
        n_locs_used=n,
        profile_sharpness=profile_sharpness(z_p, bin_nm),
    )


# ------------------------------------------------------------ transforms
def to_axon_frame(
    frame: AxonFrame,
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Rotate localizations into the axon frame: p' = R (p - origin) + origin.

    An identity frame returns copies of the inputs untouched: in exact
    arithmetic the formula reduces to p, but in floating point
    (p - origin) + origin can differ from p in the last bit, and a frame
    that "changes nothing" must change nothing.

    Parameters
    ----------
    frame : AxonFrame
    x_nm, y_nm, z_nm : (M,) arrays
        Lab coordinates in nm.

    Returns
    -------
    (x', y', z') : three (M,) float64 arrays
        Axon-frame coordinates in nm; z' runs along the axis.
    """
    x = _as_float_1d(x_nm)
    y = _as_float_1d(y_nm)
    z = _as_float_1d(z_nm)
    if frame.is_identity:
        return x.copy(), y.copy(), z.copy()
    origin = _as_float_1d(frame.origin_nm)
    pts = np.column_stack([x, y, z]) - origin
    rotated = pts @ frame.rotation.T + origin
    return (np.ascontiguousarray(rotated[:, 0]),
            np.ascontiguousarray(rotated[:, 1]),
            np.ascontiguousarray(rotated[:, 2]))


def from_axon_frame(
    frame: AxonFrame,
    x_p: NDArray[np.float64],
    y_p: NDArray[np.float64],
    z_p: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Exact inverse of ``to_axon_frame``: p = R^T (p' - origin) + origin.

    Parameters
    ----------
    frame : AxonFrame
    x_p, y_p, z_p : (M,) arrays
        Axon-frame coordinates in nm.

    Returns
    -------
    (x, y, z) : three (M,) float64 arrays
        Lab coordinates in nm.
    """
    x = _as_float_1d(x_p)
    y = _as_float_1d(y_p)
    z = _as_float_1d(z_p)
    if frame.is_identity:
        return x.copy(), y.copy(), z.copy()
    origin = _as_float_1d(frame.origin_nm)
    pts = np.column_stack([x, y, z]) - origin
    back = pts @ frame.rotation + origin
    return (np.ascontiguousarray(back[:, 0]),
            np.ascontiguousarray(back[:, 1]),
            np.ascontiguousarray(back[:, 2]))


def lateral_shift_per_period_nm(frame: AxonFrame, period_nm: float) -> float:
    """
    Lateral displacement between consecutive rings that the disagreement
    of their planes implies: period * tan(delta_beta).

    This is the quantity whose smallness (under 10 nm) justifies pairing
    clusters ring-to-ring without any per-ring lateral correction. NaN
    when ``delta_beta_deg`` is NaN (fewer than two rings entered the
    fit), because then nothing is known about the disagreement.

    Parameters
    ----------
    frame : AxonFrame
    period_nm : float
        Ring period along the axis, nm.
    """
    return float(period_nm) * math.tan(math.radians(float(frame.delta_beta_deg)))
