# -*- coding: utf-8 -*-
"""
Checks for tools/mps_axis.py against rings of KNOWN tilt.

The module answers one question: along which direction do the rings of
an axon stack, and how far is that direction from the optical axis. A
tilt beta mixes consecutive rings in z by D * tan(beta) -- hundreds of
nanometres for a large axon -- so the axial segmentation has to happen
in a rotated frame, and the rotation has to be right. Every number here
is checked against synthetic rings built with the answer known in
advance: tilt, azimuth, disagreement between rings and axial noise are
INPUTS of the generator, never estimated twice.

Tolerances are derived, not tuned (H1 spec, "validate_axis.py"):

  * three rings of 5000 localizations on an elliptical annulus with
    semi-axes 1.5 and 1.0 um and axial noise sigma_z = 50 nm give a
    slope error per plane of about sigma_z / (sqrt(N) * sd(x))
    ~ 50 / (sqrt(5000) * 900) rad ~ 0.045 deg; the common axis averages
    three such planes. Accept |beta_hat - beta| < 0.15 deg (~3 sd of one
    plane, > 4 sd of the average);
  * the azimuth of the axis is its lateral component, sin(beta), seen
    with the same ~0.0005 rad error: for beta >= 1.5 deg that is
    0.0005 / 0.026 ~ 1.1 deg, so accept 3 deg;
  * after rotation the residual tilt of each ring is that same per-plane
    error: accept < 0.15 deg; the spread of z' inside a ring is the
    axial noise, whose sd is estimated to ~1 % with N = 5000: accept
    10 %;
  * delta_beta on rings deliberately tilted 2, 3 and 4 deg is compared
    with the weighted RMS of their true angles to the true axis: the
    estimate carries the three per-plane errors (~0.05 deg each) in
    quadrature: accept 0.3 deg.

The module does not exist until the implementation phase; this harness
is written first and fails at import until then.

Run:  venv\\Scripts\\python.exe validate_axis.py

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_axis import (  # noqa: E402
    AxonFrame,
    fit_axis_by_profile_sharpness,
    fit_axon_frame,
    from_axon_frame,
    lateral_shift_per_period_nm,
    profile_sharpness,
    rodrigues_rotation,
    to_axon_frame,
)

PASSED = 0
FAILED = 0

SEED = 0
Z_HAT = np.array([0.0, 0.0, 1.0])

# Geometry of the synthetic axon (spec): three rings of 5000 points on an
# elliptical annulus with semi-axes 1.5 and 1.0 um, ring planes 190 nm
# apart along the common axis, axial noise 50 nm, tilt azimuth 40 deg.
N_PER_RING = 5000
SEMI_AXES_NM = (1500.0, 1000.0)
SPACING_NM = 190.0
SIGMA_Z_NM = 50.0
AZIMUTH_DEG = 40.0
# A non-zero origin, so that a rotation about the wrong point shows up.
ORIGIN_NM = np.array([12_000.0, 9_000.0, 250.0])


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


# ------------------------------------------------------------ geometry
def unit_axis(beta_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit vector at polar angle beta from z, azimuth measured from x."""
    b = math.radians(beta_deg)
    p = math.radians(azimuth_deg)
    return np.array([math.sin(b) * math.cos(p), math.sin(b) * math.sin(p),
                     math.cos(b)])


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors, in degrees."""
    c = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def azimuth_deg_of(u: np.ndarray) -> float:
    return math.degrees(math.atan2(u[1], u[0]))


def wrap_deg(d: float) -> float:
    """Difference of two azimuths reduced to (-180, 180]."""
    return ((d + 180.0) % 360.0) - 180.0


def plane_basis(normal: np.ndarray, azimuth_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two orthonormal in-plane vectors for a plane of the given normal.

    e1 is horizontal and perpendicular to the tilt azimuth (it lies in
    the plane whatever the tilt, because the tilt happens along the
    azimuth), e2 = n x e1 is the downhill direction. Keeping e1 fixed
    across tilts keeps the ellipse orientation the same for every beta,
    so the datasets differ only in what is being estimated.
    """
    p = math.radians(azimuth_deg)
    e1 = np.array([-math.sin(p), math.cos(p), 0.0])
    e2 = np.cross(normal, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def fit_plane_ols(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """Plain least-squares plane z = a x + b y + c (the harness's own)."""
    design = np.column_stack([x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def tilt_deg_of_plane(a: float, b: float) -> float:
    """Angle between the plane normal (-a, -b, 1) and z."""
    return math.degrees(math.atan(math.hypot(a, b)))


@dataclass
class SyntheticRings:
    """Localizations of stacked rings with the truth kept alongside."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    ring_index: np.ndarray
    normals: np.ndarray        # (N, 3) true unit normal of each ring
    u_true: np.ndarray         # (3,) count-weighted mean normal, unit
    centres: np.ndarray        # (N, 3) ring centres, nm
    counts: np.ndarray         # (N,) localizations per ring
    tilts_deg: np.ndarray
    azimuth_deg: float
    sigma_z_nm: float
    spacing_nm: float

    @property
    def beta_true_deg(self) -> float:
        return angle_deg(self.u_true, Z_HAT)

    @property
    def delta_beta_true_deg(self) -> float:
        """Count-weighted RMS angle between each ring normal and u_true."""
        theta = np.array([angle_deg(n, self.u_true) for n in self.normals])
        w = self.counts.astype(float)
        return float(math.sqrt(np.sum(w * theta ** 2) / np.sum(w)))


def make_rings(
    rng: np.random.Generator,
    tilts_deg: Sequence[float],
    azimuth_deg: float = AZIMUTH_DEG,
    *,
    counts: Sequence[int] | int = N_PER_RING,
    sigma_z_nm: float = SIGMA_Z_NM,
    spacing_nm: float = SPACING_NM,
    semi_axes_nm: Tuple[float, float] = SEMI_AXES_NM,
    origin_nm: np.ndarray = ORIGIN_NM,
) -> SyntheticRings:
    """
    Rings on tilted planes, one per entry of ``tilts_deg``.

    Ring k lies in the plane with normal at polar angle tilts_deg[k] and
    the common azimuth; the centres are spaced ``spacing_nm`` apart along
    the count-weighted mean normal (the truth of ``u``). Points sit on an
    elliptical annulus (radial scale uniform in [0.93, 1.07]) and receive
    Gaussian noise along the LAB z axis, which is what a microscope's
    axial localization error is.
    """
    tilts = np.atleast_1d(np.asarray(tilts_deg, dtype=float))
    n_rings = tilts.size
    n_per = (np.full(n_rings, int(counts)) if np.isscalar(counts)
             else np.asarray(counts, dtype=int))
    normals = np.array([unit_axis(t, azimuth_deg) for t in tilts])
    u_true = (normals * n_per[:, None]).sum(axis=0)
    u_true /= np.linalg.norm(u_true)
    offsets = (np.arange(n_rings) - (n_rings - 1) / 2.0) * spacing_nm
    centres = origin_nm[None, :] + offsets[:, None] * u_true[None, :]
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for k in range(n_rings):
        e1, e2 = plane_basis(normals[k], azimuth_deg)
        theta = rng.uniform(0.0, 2.0 * math.pi, n_per[k])
        scale = rng.uniform(0.93, 1.07, n_per[k])
        along_e1 = scale * semi_axes_nm[0] * np.cos(theta)
        along_e2 = scale * semi_axes_nm[1] * np.sin(theta)
        pts = (centres[k][None, :] + along_e1[:, None] * e1[None, :]
               + along_e2[:, None] * e2[None, :])
        pts[:, 2] += rng.normal(0.0, sigma_z_nm, n_per[k])
        xs.append(pts[:, 0])
        ys.append(pts[:, 1])
        zs.append(pts[:, 2])
        labels.append(np.full(n_per[k], k, dtype=np.int64))
    return SyntheticRings(
        x=np.concatenate(xs), y=np.concatenate(ys), z=np.concatenate(zs),
        ring_index=np.concatenate(labels), normals=normals, u_true=u_true,
        centres=centres, counts=n_per, tilts_deg=tilts,
        azimuth_deg=azimuth_deg, sigma_z_nm=sigma_z_nm, spacing_nm=spacing_nm,
    )


def assert_frame_basics(frame: AxonFrame, n_rings: int) -> None:
    """Shapes, dtypes and the sign convention every fitted frame obeys."""
    assert frame.u.shape == (3,), frame.u.shape
    assert abs(np.linalg.norm(frame.u) - 1.0) < 1e-12, np.linalg.norm(frame.u)
    assert frame.u[2] > 0, frame.u
    assert frame.origin_nm.shape == (3,), frame.origin_nm.shape
    assert frame.rotation.shape == (3, 3), frame.rotation.shape
    assert frame.ring_normals.shape == (n_rings, 3), frame.ring_normals.shape
    assert frame.ring_n_locs.shape == (n_rings,), frame.ring_n_locs.shape
    assert frame.ring_used.shape == (n_rings,), frame.ring_used.shape
    assert np.issubdtype(frame.ring_n_locs.dtype, np.integer), frame.ring_n_locs.dtype
    assert frame.ring_used.dtype == np.bool_, frame.ring_used.dtype
    assert isinstance(frame.warnings, list)
    for k in range(n_rings):
        row = frame.ring_normals[k]
        if frame.ring_used[k]:
            assert np.all(np.isfinite(row)), (k, row)
            assert abs(np.linalg.norm(row) - 1.0) < 1e-12, (k, row)
            assert row[2] > 0, (k, row)
        else:
            assert np.all(np.isnan(row)), (k, row)


# ============================================== 1. tilt of the common axis
def test_tilt_recovery() -> None:
    print("\n1. TILT AND AZIMUTH OF THE COMMON AXIS (three parallel rings)")
    rng = np.random.default_rng(SEED)

    for beta in (0.0, 1.5, 3.0, 5.0):
        truth = make_rings(rng, [beta] * 3)

        def recovered(beta=beta, truth=truth):
            frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
            assert_frame_basics(frame, 3)
            assert frame.method == "plane_per_ring", frame.method
            assert frame.ring_used.all(), frame.ring_used
            assert np.array_equal(frame.ring_n_locs, truth.counts), frame.ring_n_locs
            err_beta = frame.beta_deg - beta
            assert abs(err_beta) < 0.15, f"beta_hat={frame.beta_deg:.4f} vs {beta}"
            # The axis itself, not only its polar angle.
            err_axis = angle_deg(frame.u, truth.u_true)
            assert err_axis < 0.15, f"axis off by {err_axis:.4f} deg"
            per_ring = max(angle_deg(frame.ring_normals[k], truth.normals[k])
                           for k in range(3))
            detail = (f"beta_hat={frame.beta_deg:.3f} (err {err_beta:+.3f} deg), "
                      f"axis off {err_axis:.3f} deg, worst ring normal off "
                      f"{per_ring:.3f} deg, delta_beta={frame.delta_beta_deg:.3f}")
            if beta >= 1.5:
                err_az = wrap_deg(azimuth_deg_of(frame.u) - truth.azimuth_deg)
                assert abs(err_az) < 3.0, f"azimuth off by {err_az:.3f} deg"
                detail += f", azimuth err {err_az:+.2f} deg"
            if beta <= 3.0:
                # Clean parallel rings at a modest tilt: none of the four
                # warnings of the spec applies (tilt <= 5, delta_beta < 1,
                # no ring skipped, no fallback).
                assert frame.warnings == [], frame.warnings
            return detail

        check(f"beta = {beta:g} deg recovered (|beta_hat - beta| < 0.15 deg"
              + ("; azimuth within 3 deg)" if beta >= 1.5 else ")"), recovered)

        def rotated_rings_are_flat(truth=truth):
            frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
            x_p, y_p, z_p = to_axon_frame(frame, truth.x, truth.y, truth.z)
            tilts = []
            spreads = []
            means = []
            for k in range(3):
                m = truth.ring_index == k
                a, b, _ = fit_plane_ols(x_p[m], y_p[m], z_p[m])
                tilts.append(tilt_deg_of_plane(a, b))
                spreads.append(float(np.std(z_p[m])))
                means.append(float(np.mean(z_p[m])))
            assert max(tilts) < 0.15, f"residual tilts {tilts}"
            for s in spreads:
                assert abs(s - truth.sigma_z_nm) < 0.10 * truth.sigma_z_nm, spreads
            # Consecutive rings stay spacing_nm apart along z' (sd of a
            # ring mean is 50/sqrt(5000) = 0.7 nm; 3 nm is > 3 sd of a
            # difference).
            gaps = np.diff(means)
            assert np.all(np.abs(gaps - truth.spacing_nm) < 3.0), gaps
            return (f"residual tilt max {max(tilts):.3f} deg, sd(z') = "
                    f"{min(spreads):.1f}..{max(spreads):.1f} nm (truth "
                    f"{truth.sigma_z_nm:g}), z' gaps {gaps[0]:.1f}, {gaps[1]:.1f} nm")

        check(f"beta = {beta:g} deg: rotated rings flat (tilt < 0.15 deg), "
              f"sd(z') = sigma_z within 10 %, spacing kept within 3 nm",
              rotated_rings_are_flat)


# ================================================= 2. rotation algebra
def test_rotation_algebra() -> None:
    print("\n2. ROTATION ALGEBRA (beta = 5 deg, the strongest rotation)")
    rng = np.random.default_rng(SEED)
    truth = make_rings(rng, [5.0] * 3)
    frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)

    def round_trip():
        x_p, y_p, z_p = to_axon_frame(frame, truth.x, truth.y, truth.z)
        x_b, y_b, z_b = from_axon_frame(frame, x_p, y_p, z_p)
        worst = max(np.max(np.abs(x_b - truth.x)), np.max(np.abs(y_b - truth.y)),
                    np.max(np.abs(z_b - truth.z)))
        assert worst < 1e-9, worst
        # And the transform really moved the points (not a no-op).
        moved = float(np.max(np.abs(z_p - truth.z)))
        assert moved > 10.0, moved
        return f"max |from(to(p)) - p| = {worst:.2e} nm; max |z' - z| = {moved:.0f} nm"

    def forward_formula():
        x_p, y_p, z_p = to_axon_frame(frame, truth.x, truth.y, truth.z)
        pts = np.column_stack([truth.x, truth.y, truth.z])
        expect = (frame.rotation @ (pts - frame.origin_nm).T).T + frame.origin_nm
        worst = float(np.max(np.abs(np.column_stack([x_p, y_p, z_p]) - expect)))
        assert worst < 1e-9, worst
        return f"p' = R (p - origin) + origin to {worst:.2e} nm"

    def rotation_properties():
        r = frame.rotation
        rz = r @ frame.u
        assert np.max(np.abs(rz - Z_HAT)) < 1e-12, rz
        assert frame.u[2] > 0, frame.u
        det = float(np.linalg.det(r))
        assert abs(det - 1.0) < 1e-12, det
        assert np.max(np.abs(r.T @ r - np.eye(3))) < 1e-12
        # The rotation angle is exactly beta.
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1.0) / 2.0))))
        assert abs(angle - frame.beta_deg) < 1e-9, (angle, frame.beta_deg)
        return (f"|R u - z| = {np.max(np.abs(rz - Z_HAT)):.1e}, det = {det:.15f}, "
                f"rotation angle {angle:.4f} deg = beta_hat")

    def origin_is_mean_of_used_locs():
        expect = np.array([truth.x.mean(), truth.y.mean(), truth.z.mean()])
        worst = float(np.max(np.abs(frame.origin_nm - expect)))
        assert worst < 1e-6, (frame.origin_nm, expect)
        return f"origin = mean of the 15000 localizations to {worst:.1e} nm"

    check("from_axon_frame(to_axon_frame(p)) == p (1e-9 nm)", round_trip)
    check("to_axon_frame is R (p - origin) + origin (1e-9 nm)", forward_formula)
    check("rotation @ u == z (1e-12), u[2] > 0, det == 1 (1e-12), orthogonal",
          rotation_properties)
    check("origin_nm is the mean of the localizations used (1e-6 nm)",
          origin_is_mean_of_used_locs)


# ==================================================== 3. rodrigues_rotation
def test_rodrigues() -> None:
    print("\n3. rodrigues_rotation")
    rng = np.random.default_rng(SEED)

    def same_vector_is_identity():
        r = rodrigues_rotation(Z_HAT, Z_HAT)
        assert np.max(np.abs(r - np.eye(3))) < 1e-12, r
        a = unit_axis(37.0, -110.0)
        r2 = rodrigues_rotation(a, a)
        assert np.max(np.abs(r2 - np.eye(3))) < 1e-12, r2
        return "z -> z and a -> a give the identity"

    def random_pairs():
        worst_map = 0.0
        worst_det = 0.0
        worst_orth = 0.0
        for _ in range(20):
            a = rng.normal(size=3)
            a /= np.linalg.norm(a)
            b = rng.normal(size=3)
            b /= np.linalg.norm(b)
            r = rodrigues_rotation(a, b)
            worst_map = max(worst_map, float(np.max(np.abs(r @ a - b))))
            worst_det = max(worst_det, abs(float(np.linalg.det(r)) - 1.0))
            worst_orth = max(worst_orth, float(np.max(np.abs(r.T @ r - np.eye(3)))))
        assert worst_map < 1e-12, worst_map
        assert worst_det < 1e-12, worst_det
        assert worst_orth < 1e-12, worst_orth
        return (f"20 random pairs: |R a - b| <= {worst_map:.1e}, "
                f"|det - 1| <= {worst_det:.1e}, |R'R - I| <= {worst_orth:.1e}")

    def small_angle_pair():
        # The case the module actually meets: u a few degrees from z.
        a = unit_axis(2.0, AZIMUTH_DEG)
        r = rodrigues_rotation(a, Z_HAT)
        assert np.max(np.abs(r @ a - Z_HAT)) < 1e-12
        angle = math.degrees(math.acos((np.trace(r) - 1.0) / 2.0))
        assert abs(angle - 2.0) < 1e-9, angle
        return f"rotation angle {angle:.9f} deg for a 2 deg tilt"

    def antiparallel_pair():
        # Beyond the spec list: the naive Rodrigues formula divides by
        # 1 + a.b, which is 0 here. A rotation "taking a onto b" must
        # still be a proper rotation in that case.
        r = rodrigues_rotation(Z_HAT, -Z_HAT)
        assert np.all(np.isfinite(r)), r
        assert np.max(np.abs(r @ Z_HAT + Z_HAT)) < 1e-12, r
        assert abs(float(np.linalg.det(r)) - 1.0) < 1e-12
        assert np.max(np.abs(r.T @ r - np.eye(3))) < 1e-12
        return "z -> -z is finite, proper (det +1) and maps z onto -z"

    check("rodrigues: a == b gives the identity (1e-12)", same_vector_is_identity)
    check("rodrigues: R a == b, det 1, orthonormal on 20 random pairs (1e-12)",
          random_pairs)
    check("rodrigues: 2 deg tilt rotated by exactly 2 deg (1e-9 deg)",
          small_angle_pair)
    check("rodrigues: antiparallel a = -b still a proper rotation (beyond spec)",
          antiparallel_pair)


# ================================================== 4. delta_beta
def test_delta_beta() -> None:
    print("\n4. DISAGREEMENT BETWEEN RINGS (delta_beta)")
    rng = np.random.default_rng(SEED)

    def parallel_rings():
        truth = make_rings(rng, [3.0] * 3)
        frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        assert np.isfinite(frame.delta_beta_deg)
        assert frame.delta_beta_deg < 0.15, frame.delta_beta_deg
        return f"delta_beta = {frame.delta_beta_deg:.3f} deg (truth 0)"

    def tilted_2_3_4():
        truth = make_rings(rng, [2.0, 3.0, 4.0])
        frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        assert_frame_basics(frame, 3)
        expect = truth.delta_beta_true_deg
        err = frame.delta_beta_deg - expect
        assert abs(err) < 0.3, f"delta_beta {frame.delta_beta_deg:.3f} vs truth {expect:.3f}"
        err_beta = frame.beta_deg - truth.beta_true_deg
        assert abs(err_beta) < 0.15, f"beta_hat {frame.beta_deg:.3f} vs {truth.beta_true_deg:.3f}"
        # Each ring's own normal, against its own truth.
        per_ring = [angle_deg(frame.ring_normals[k], truth.normals[k]) for k in range(3)]
        assert max(per_ring) < 0.15, per_ring
        # delta_beta < 1 deg here (~0.82), so no warning is due.
        assert frame.warnings == [], frame.warnings
        return (f"delta_beta = {frame.delta_beta_deg:.3f} deg (truth {expect:.3f}, "
                f"err {err:+.3f}); beta_hat = {frame.beta_deg:.3f} "
                f"(truth {truth.beta_true_deg:.3f}); ring normals off "
                f"{max(per_ring):.3f} deg at most")

    def tilted_0_3_6_warns():
        truth = make_rings(rng, [0.0, 3.0, 6.0])
        frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        expect = truth.delta_beta_true_deg          # ~2.45 deg
        assert expect > 1.0, expect
        assert abs(frame.delta_beta_deg - expect) < 0.3, (frame.delta_beta_deg, expect)
        # beta_hat ~ 3 deg (no tilt warning), all rings used (no skip
        # warning), no fallback: any warning present is the delta_beta one.
        assert frame.beta_deg < 5.0, frame.beta_deg
        assert frame.ring_used.all()
        assert frame.method == "plane_per_ring"
        assert len(frame.warnings) >= 1, "delta_beta > 1 deg must warn"
        return (f"delta_beta = {frame.delta_beta_deg:.3f} deg (truth {expect:.3f}); "
                f"warnings: {frame.warnings}")

    check("parallel rings: delta_beta < 0.15 deg", parallel_rings)
    check("rings tilted 2/3/4 deg, same azimuth: delta_beta within 0.3 deg of the "
          "truth RMS; beta within 0.15 deg", tilted_2_3_4)
    check("rings tilted 0/3/6 deg: delta_beta > 1 deg produces a warning",
          tilted_0_3_6_warns)


# ============================================== 5. robustness to background
def test_robustness() -> None:
    print("\n5. ROBUSTNESS: 15 % UNIFORM BACKGROUND IN z ON ONE RING")
    rng = np.random.default_rng(SEED)

    def background_on_one_ring():
        beta = 3.0
        truth = make_rings(rng, [beta] * 3)
        clean = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index,
                               loss="soft_l1")
        # Background: 15 % extra localizations on ring 1, laterally on the
        # same annulus, z uniform over +-600 nm about the ring's centre z
        # (background does not know about the plane, so it is NOT
        # symmetric about the plane at the edges of a tilted ring: it
        # pulls a plain least-squares slope towards zero).
        k = 1
        n_bg = int(round(0.15 * truth.counts[k]))
        e1, e2 = plane_basis(truth.normals[k], truth.azimuth_deg)
        theta = rng.uniform(0.0, 2.0 * math.pi, n_bg)
        scale = rng.uniform(0.93, 1.07, n_bg)
        pts = (truth.centres[k][None, :]
               + (scale * SEMI_AXES_NM[0] * np.cos(theta))[:, None] * e1[None, :]
               + (scale * SEMI_AXES_NM[1] * np.sin(theta))[:, None] * e2[None, :])
        z_bg = truth.centres[k][2] + rng.uniform(-600.0, 600.0, n_bg)
        x = np.r_[truth.x, pts[:, 0]]
        y = np.r_[truth.y, pts[:, 1]]
        z = np.r_[truth.z, z_bg]
        lab = np.r_[truth.ring_index, np.full(n_bg, k, dtype=np.int64)]
        dirty = fit_axon_frame(x, y, z, lab, loss="soft_l1")
        assert dirty.ring_n_locs[k] == truth.counts[k] + n_bg, dirty.ring_n_locs
        shift = dirty.beta_deg - clean.beta_deg
        assert abs(shift) < 0.2, f"beta shifted by {shift:.3f} deg"
        err_truth = dirty.beta_deg - beta
        assert abs(err_truth) < 0.2, f"beta_hat {dirty.beta_deg:.3f} vs truth {beta}"
        axis_shift = angle_deg(dirty.u, clean.u)
        # For the reader: what a plain least-squares plane makes of the
        # contaminated ring on its own.
        m = lab == k
        a_ols, b_ols, _ = fit_plane_ols(x[m], y[m], z[m])
        ring_ols = tilt_deg_of_plane(a_ols, b_ols)
        ring_robust = angle_deg(dirty.ring_normals[k], Z_HAT)
        return (f"{n_bg} background locs on ring {k}: beta {clean.beta_deg:.3f} -> "
                f"{dirty.beta_deg:.3f} (shift {shift:+.3f} deg, truth err "
                f"{err_truth:+.3f}), axis moved {axis_shift:.3f} deg; ring {k} "
                f"alone: robust tilt {ring_robust:.3f} vs plain OLS {ring_ols:.3f} "
                f"(truth {beta})")

    check("15 % uniform z background (+-600 nm) on one ring: beta shift < 0.2 deg "
          "with loss='soft_l1'", background_on_one_ring)


# ============================================ 6. warnings and fallbacks
def test_fallbacks_and_warnings() -> None:
    print("\n6. WARNINGS, FALLBACKS AND DETERMINISM")
    rng = np.random.default_rng(SEED)

    def tilt_warning_at_6_deg():
        truth = make_rings(rng, [6.0] * 3)
        frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        assert abs(frame.beta_deg - 6.0) < 0.15, frame.beta_deg
        hits = [w for w in frame.warnings if "tilt" in w.lower()]
        assert hits, f"no warning mentions the tilt: {frame.warnings}"
        return f"beta_hat = {frame.beta_deg:.3f} deg; warning: {hits[0]!r}"

    def small_ring_is_skipped():
        truth = make_rings(rng, [3.0] * 3, counts=[N_PER_RING, 50, N_PER_RING])
        frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        assert_frame_basics(frame, 3)
        assert frame.method == "plane_per_ring", frame.method
        assert np.array_equal(frame.ring_used, [True, False, True]), frame.ring_used
        assert np.array_equal(frame.ring_n_locs, [N_PER_RING, 50, N_PER_RING])
        assert np.all(np.isnan(frame.ring_normals[1])), frame.ring_normals[1]
        assert abs(frame.beta_deg - 3.0) < 0.15, frame.beta_deg
        # delta_beta from the two rings that entered (parallel -> small).
        assert np.isfinite(frame.delta_beta_deg) and frame.delta_beta_deg < 0.15
        # Origin = mean of the localizations that entered, not of all.
        used = truth.ring_index != 1
        expect = np.array([truth.x[used].mean(), truth.y[used].mean(),
                           truth.z[used].mean()])
        assert np.max(np.abs(frame.origin_nm - expect)) < 1e-6, (frame.origin_nm, expect)
        keywords = ("skip", "few", "min_locs", "ignored")
        hits = [w for w in frame.warnings
                if any(k in w.lower() for k in keywords)]
        assert hits, f"skipped ring not reported: {frame.warnings}"
        return (f"ring 1 (50 locs) skipped, beta_hat = {frame.beta_deg:.3f} deg from "
                f"rings 0 and 2; warning: {hits[0]!r}")

    def identity_when_fewer_than_3_points():
        details = []
        for n in (2, 1, 0):
            x = np.array([100.0, 300.0][:n])
            y = np.array([50.0, -20.0][:n])
            z = np.array([10.0, 40.0][:n])
            lab = np.zeros(n, dtype=np.int64)
            frame = fit_axon_frame(x, y, z, lab)
            assert frame.method == "identity", (n, frame.method)
            assert np.array_equal(frame.rotation, np.eye(3)), frame.rotation
            assert np.array_equal(frame.u, Z_HAT), frame.u
            assert frame.beta_deg == 0.0, frame.beta_deg
            assert np.isnan(frame.delta_beta_deg), frame.delta_beta_deg
            assert not frame.ring_used.any()
            assert frame.ring_normals.shape[0] == frame.ring_n_locs.shape[0] \
                == frame.ring_used.shape[0]
            assert frame.warnings, f"no warning with {n} points"
            x_p, y_p, z_p = to_axon_frame(frame, x, y, z)
            assert np.array_equal(x_p, x) and np.array_equal(y_p, y) \
                and np.array_equal(z_p, z)
            details.append(f"n={n}: {frame.warnings}")
        return "; ".join(details)

    def identity_when_collinear():
        t = np.linspace(-1000.0, 1000.0, 500)
        x = 200.0 + t
        y = -50.0 + 2.0 * t
        z = 30.0 + 0.1 * t
        lab = np.zeros(500, dtype=np.int64)
        frame = fit_axon_frame(x, y, z, lab)
        assert frame.method == "identity", frame.method
        assert np.array_equal(frame.rotation, np.eye(3)), frame.rotation
        assert np.array_equal(frame.u, Z_HAT), frame.u
        assert np.all(np.isfinite(frame.origin_nm)), frame.origin_nm
        assert frame.warnings, "collinear design must warn"
        return f"500 collinear points: {frame.warnings}"

    def plane_all_when_no_ring_reaches_min_locs():
        # One labelled ring of 100 points (< min_locs = 200) with a tiny
        # axial noise (5 nm), so the single plane on all points is
        # precise: slope error ~ 5 / (sqrt(100) * 700) rad ~ 0.04 deg.
        truth = make_rings(rng, [3.0], counts=100, sigma_z_nm=5.0)
        frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        assert frame.method == "plane_all", frame.method
        assert frame.warnings, "plane_all fallback must warn"
        assert frame.u[2] > 0
        assert abs(frame.beta_deg - 3.0) < 0.3, frame.beta_deg
        assert np.isnan(frame.delta_beta_deg), frame.delta_beta_deg
        assert np.max(np.abs(frame.rotation @ frame.u - Z_HAT)) < 1e-12
        return (f"100 locs < min_locs=200: one plane, beta_hat = {frame.beta_deg:.3f} "
                f"deg (truth 3); warnings: {frame.warnings}")

    def negative_labels_are_ignored():
        truth = make_rings(rng, [3.0] * 3)
        base = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        n_junk = 1000
        x = np.r_[truth.x, rng.uniform(truth.x.min(), truth.x.max(), n_junk)]
        y = np.r_[truth.y, rng.uniform(truth.y.min(), truth.y.max(), n_junk)]
        z = np.r_[truth.z, ORIGIN_NM[2] + rng.uniform(-2000.0, 2000.0, n_junk)]
        lab = np.r_[truth.ring_index, np.full(n_junk, -1, dtype=np.int64)]
        frame = fit_axon_frame(x, y, z, lab)
        assert np.array_equal(frame.ring_n_locs, base.ring_n_locs)
        assert np.max(np.abs(frame.u - base.u)) < 1e-9, (frame.u, base.u)
        assert np.max(np.abs(frame.origin_nm - base.origin_nm)) < 1e-6
        assert np.max(np.abs(frame.rotation - base.rotation)) < 1e-9
        assert abs(frame.beta_deg - base.beta_deg) < 1e-9
        assert frame.ring_normals.shape == base.ring_normals.shape
        return f"{n_junk} locs labelled -1 change u by {np.max(np.abs(frame.u - base.u)):.1e}"

    def deterministic():
        truth = make_rings(rng, [2.0, 3.0, 4.0])
        a = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        b = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        for name in ("u", "origin_nm", "ring_normals", "rotation"):
            assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
        assert np.array_equal(a.ring_n_locs, b.ring_n_locs)
        assert np.array_equal(a.ring_used, b.ring_used)
        assert a.beta_deg == b.beta_deg
        assert (a.delta_beta_deg == b.delta_beta_deg
                or (math.isnan(a.delta_beta_deg) and math.isnan(b.delta_beta_deg)))
        assert a.method == b.method and a.min_locs == b.min_locs and a.loss == b.loss
        assert a.warnings == b.warnings
        return "two runs: every field bit-identical"

    check("beta = 6 deg: a warning mentions the tilt", tilt_warning_at_6_deg)
    check("ring with 50 points is skipped (ring_used False, NaN normal) and reported",
          small_ring_is_skipped)
    check("fewer than 3 points (2, 1, 0): identity frame with a warning",
          identity_when_fewer_than_3_points)
    check("collinear points: identity frame with a warning", identity_when_collinear)
    check("no ring reaches min_locs: one plane on all points ('plane_all') with a "
          "warning, beta within 0.3 deg", plane_all_when_no_ring_reaches_min_locs)
    check("labels < 0 are ignored (u, origin, rotation unchanged to 1e-9)",
          negative_labels_are_ignored)
    check("deterministic: two runs give identical fields", deterministic)


# ======================================= 7. lateral shift per period
def test_lateral_shift() -> None:
    print("\n7. lateral_shift_per_period_nm")
    rng = np.random.default_rng(SEED)

    def equals_period_tan_delta_beta():
        truth = make_rings(rng, [2.0, 3.0, 4.0])
        frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        period = 190.0
        got = lateral_shift_per_period_nm(frame, period)
        expect = period * math.tan(math.radians(frame.delta_beta_deg))
        assert abs(got - expect) < 1e-9, (got, expect)
        from_truth = period * math.tan(math.radians(truth.delta_beta_true_deg))
        return (f"{got:.3f} nm per {period:g} nm period (delta_beta "
                f"{frame.delta_beta_deg:.3f} deg; truth would give {from_truth:.3f} nm)")

    def nan_with_one_ring():
        truth = make_rings(rng, [3.0])
        frame = fit_axon_frame(truth.x, truth.y, truth.z, truth.ring_index)
        assert frame.method == "plane_per_ring", frame.method
        assert frame.ring_used.sum() == 1
        assert np.isnan(frame.delta_beta_deg), frame.delta_beta_deg
        got = lateral_shift_per_period_nm(frame, 190.0)
        assert isinstance(got, float) and math.isnan(got), got
        assert abs(frame.beta_deg - 3.0) < 0.15, frame.beta_deg
        return f"one ring: delta_beta NaN, shift NaN (beta_hat {frame.beta_deg:.3f} deg)"

    check("lateral_shift_per_period_nm == period * tan(delta_beta) (1e-9 nm)",
          equals_period_tan_delta_beta)
    check("lateral_shift_per_period_nm is NaN with one ring", nan_with_one_ring)


# ================================= 8. axis from the profile sharpness
def test_profile_sharpness() -> None:
    print("\n8. AXIS FROM THE SHARPNESS OF THE AXIAL PROFILE (no ring labels)")
    rng = np.random.default_rng(SEED)

    def gaussian_value():
        # For one Gaussian of width sigma the collision probability of a
        # histogram with bin h is h / (2 sigma sqrt(pi)); with N = 200000
        # the sampling noise is ~0.3 % and the discretisation bias
        # (h / sigma)^2 / 12 = 0.08 % at h = 5 nm, sigma = 50 nm: accept 2 %.
        z = rng.normal(0.0, 50.0, 200_000)
        got = profile_sharpness(z)
        expected = 5.0 / (2.0 * 50.0 * math.sqrt(math.pi))
        assert abs(got - expected) / expected < 0.02, (got, expected)
        assert profile_sharpness(np.array([np.nan, np.inf])) == 0.0
        assert profile_sharpness(np.array([])) == 0.0
        return f"sum p^2 = {got:.5f} vs h / (2 sigma sqrt(pi)) = {expected:.5f}; empty -> 0"

    def tilt_from_sharpness():
        # The search ends on a 0.1 deg grid, and on this 1.5 x 1.0 um axon
        # the sharpness peak is shallow (a 0.3 deg residual smears each ring
        # by 16 nm against sigma_z = 50 nm, 0.4 % of the statistic, about
        # its sampling noise at N = 15000), so the grid can wander a few
        # steps: accept 0.3 deg (0.06-0.10 deg measured), azimuth 5 deg
        # for beta >= 1.5 (0.3 / 1.5 rad ~ 11 deg would be the worst case
        # of a 0.3 deg error in the wrong direction; 2 deg measured).
        out = []
        for beta in (0.0, 1.5, 3.0, 5.0):
            rings = make_rings(rng, [beta, beta, beta])
            frame = fit_axis_by_profile_sharpness(rings.x, rings.y, rings.z)
            assert frame.method == "profile_sharpness", frame.method
            assert frame.ring_normals.shape == (0, 3) and math.isnan(frame.delta_beta_deg)
            assert frame.u[2] > 0 and abs(np.linalg.norm(frame.u) - 1.0) < 1e-12
            assert np.allclose(frame.rotation @ frame.u, Z_HAT, atol=1e-12)
            assert np.isfinite(frame.profile_sharpness) and frame.profile_sharpness > 0
            assert abs(frame.beta_deg - beta) < 0.3, (beta, frame.beta_deg)
            if beta >= 1.5:
                az = wrap_deg(azimuth_deg_of(frame.u) - AZIMUTH_DEG)
                assert abs(az) < 5.0, (beta, az)
            assert np.allclose(frame.origin_nm, [rings.x.mean(), rings.y.mean(), rings.z.mean()])
            out.append(f"{beta}->{frame.beta_deg:.2f}")
        return "beta truth->hat (deg): " + ", ".join(out)

    def partial_rings(arc_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Untilted rings; ring 0 loses the arc of ``arc_deg`` about 0 deg
        of parametric angle, ring 2 the same arc about 180 deg."""
        rings = make_rings(rng, [0.0, 0.0, 0.0])
        dx = rings.x - rings.centres[rings.ring_index, 0]
        dy = rings.y - rings.centres[rings.ring_index, 1]
        ang = np.degrees(np.arctan2(dy / SEMI_AXES_NM[1], dx / SEMI_AXES_NM[0]))
        keep = np.ones(rings.x.size, dtype=bool)
        keep[(rings.ring_index == 0) & (np.abs(ang) < arc_deg / 2)] = False
        keep[(rings.ring_index == 2) & (np.abs(np.abs(ang) - 180.0) < arc_deg / 2)] = False
        return rings.x[keep], rings.y[keep], rings.z[keep]

    def immune_to_opposite_arcs():
        # Ring 0 misses a 120 deg arc on one side, ring 2 the same arc on
        # the other, no tilt. A single plane through all localizations
        # reads the shift of the cloud's centre with z as a tilt (4.2-4.4
        # deg measured), while the profile sharpness does not care where
        # a ring's localizations are laterally: within 0.5 deg (0.20-0.22
        # measured; the 0.3 deg of the complete rings widened because the
        # partial rings hold fewer localizations).
        x, y, z = partial_rings(120.0)
        pooled = fit_axon_frame(x, y, z, np.zeros(x.size, dtype=np.int64), min_locs=3)
        sharp = fit_axis_by_profile_sharpness(x, y, z)
        assert pooled.beta_deg > 1.0, pooled.beta_deg      # the failure being guarded
        assert sharp.beta_deg < 0.5, sharp.beta_deg
        return (f"pooled plane reads {pooled.beta_deg:.2f} deg of tilt into untilted "
                f"rings; profile sharpness {sharp.beta_deg:.2f} deg")

    def stacked_by_symmetric_halves():
        # The known limit of the statistic, encoded so that a change is
        # noticed: with ring 0 the x < 0 half and ring 2 the x > 0 half, a
        # shear of about P / semi-axis = atan(190 / 1500) = 7 deg lifts one
        # half onto the middle ring and lowers the other onto it, and the
        # collision probability rewards that stacking more than the true
        # frame's three thin rings (measured 5.2-5.6 deg, sharpness 10 %
        # above the true frame's). build_rings therefore never trusts the
        # start alone: it refines both starts and chooses on the depth of
        # the density valleys (validate_rings_h1 section 7c).
        x, y, z = partial_rings(180.0)
        sharp = fit_axis_by_profile_sharpness(x, y, z)
        assert sharp.beta_deg > 3.0, sharp.beta_deg
        assert sharp.profile_sharpness > profile_sharpness(z), (sharp.profile_sharpness, profile_sharpness(z))
        return (f"sharpness start {sharp.beta_deg:.2f} deg on untilted symmetric half rings "
                f"(statistic {sharp.profile_sharpness:.4f} vs {profile_sharpness(z):.4f} at the truth)")

    def degenerate_and_limits():
        n = 400
        line = np.linspace(-1000.0, 1000.0, n)
        col = fit_axis_by_profile_sharpness(line, 0.5 * line, rng.normal(0.0, 40.0, n))
        assert col.method == "identity" and any("collinear" in w for w in col.warnings)
        few = fit_axis_by_profile_sharpness(line[:2], line[:2], line[:2])
        assert few.method == "identity" and any("identity" in w for w in few.warnings)
        rings = make_rings(rng, [12.0, 12.0, 12.0])
        far = fit_axis_by_profile_sharpness(rings.x, rings.y, rings.z)
        assert any("search range" in w for w in far.warnings), far.warnings
        assert any("tilt" in w for w in far.warnings)
        x = rings.x.copy()
        x[5] = np.nan
        nonfinite = fit_axis_by_profile_sharpness(x, rings.y, rings.z)
        assert any("non-finite" in w for w in nonfinite.warnings), nonfinite.warnings
        assert nonfinite.n_locs_used == rings.x.size - 1
        return (f"collinear -> identity; 2 points -> identity; 12 deg -> border warning "
                f"(hat {far.beta_deg:.1f} deg); one NaN ignored and reported")

    def deterministic_and_stray_label():
        rings = make_rings(rng, [2.0, 2.0, 2.0])
        a = fit_axis_by_profile_sharpness(rings.x, rings.y, rings.z)
        b = fit_axis_by_profile_sharpness(rings.x, rings.y, rings.z)
        assert np.array_equal(a.u, b.u) and a.profile_sharpness == b.profile_sharpness
        # fit_axon_frame visits only the labels present: a stray large
        # label costs a row per label value (documented contract) but not
        # a loop iteration per value, and changes nothing else.
        labels = rings.ring_index.copy()
        labels[0] = 5_000_000
        stray = fit_axon_frame(rings.x, rings.y, rings.z, labels)
        compact = fit_axon_frame(rings.x[1:], rings.y[1:], rings.z[1:], rings.ring_index[1:])
        assert stray.ring_normals.shape == (5_000_001, 3)
        assert np.array_equal(stray.u, compact.u) and stray.beta_deg == compact.beta_deg
        assert any("5000000 skipped" in w for w in stray.warnings), stray.warnings
        return (f"two runs identical (beta {a.beta_deg:.3f}); stray label 5e6: same axis "
                f"as compact labels, skipped and reported")

    check("profile_sharpness of a Gaussian equals h / (2 sigma sqrt(pi)) within 2 %; empty -> 0",
          gaussian_value)
    check("fit_axis_by_profile_sharpness: tilt within 0.3 deg for beta 0/1.5/3/5, azimuth within 5 deg",
          tilt_from_sharpness)
    check("opposite 120 deg arcs missing, no tilt: pooled plane > 1 deg, profile sharpness < 0.5 deg",
          immune_to_opposite_arcs)
    check("known limit: symmetric half rings stack under a shear, sharpness start > 3 deg (guarded in build_rings)",
          stacked_by_symmetric_halves)
    check("collinear / < 3 points -> identity with warning; 12 deg -> border warning; NaN ignored",
          degenerate_and_limits)
    check("deterministic; fit_axon_frame with a stray label 5e6 gives the compact result",
          deterministic_and_stray_label)


def main() -> int:
    print("=" * 72)
    print("AXON AXIS CHECKS (tools/mps_axis.py) -- synthetic truth, seed 0")
    print("=" * 72)
    test_tilt_recovery()
    test_rotation_algebra()
    test_rodrigues()
    test_delta_beta()
    test_robustness()
    test_fallbacks_and_warnings()
    test_lateral_shift()
    test_profile_sharpness()
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
