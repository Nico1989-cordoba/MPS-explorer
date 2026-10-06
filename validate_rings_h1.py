# -*- coding: utf-8 -*-
"""
Checks for tools/mps_columns.py (H1 part: ``build_rings`` and its data
model) and for the ``roi_indices_unfiltered`` hook in
``MPS_explorer._apply_z_range``, against synthetic axons of KNOWN truth.

Every localization of the synthetic axons carries its true ring, cluster
and fluorophore, its frame number and its true (noise-free) position, so
each quantity ``build_rings`` reports can be compared with what generated
it rather than with what "looks fine". Where a tolerance appears, the
argument for it is next to the check.

Points where this harness departs from the letter of the H1
specification, each with the number that forced it (measured with the
existing pipeline on the same synthetic axon before the module existed):

* Cluster centroids: ``tools.cluster_quality.good_cluster_centroids``
  rounds to 2 decimals, so the mean of a cluster's localizations equals
  its ``an.centroids`` row to 0.005 nm, never to the 1e-6 nm the spec
  asks for. ``Cluster.centroid_nm`` is the unrounded mean and is checked
  to 1e-6 nm; the ``an.centroids`` row to 0.005 nm + 1e-6.
* Events per ring: with lateral precision 8 nm the step between two
  localizations of one fluorophore has sd sqrt(2)*8 = 11.3 nm, so a link
  radius of factor * 8 nm loses exp(-factor^2 / 4) of the steps: 21 % at
  the spec's factor 2.5 (1.66 events per fluorophore measured), 0.2 % at
  factor 5 (1.00 measured). The module's default is therefore 5, the
  +/-5 % check runs at the DEFAULT, and a run at the spec's 2.5 is kept
  as a sensitivity check of the recorded lost-step fraction.
* Tilt: the spec's own case (D = 8 um, 3 deg, D*tan(beta) = 420 nm) is
  encoded as specified. A moderate case (D = 4 um, 3 deg, 210 nm, still
  > P/2) is added beyond the spec so the report separates "no tilt
  correction at all" from "cannot handle the extreme regime". The spec's
  single lab-frame per-ring fit gives 0.045 deg and 30 % ring agreement
  on the spec's own case (measured), so the module refines two starts
  (that fit, and the tilt at which the axial profile is sharpest) and
  keeps the one whose segmentation is better resolved (deeper density
  valleys); section 7b checks the case that breaks a pooled plane, rings
  with incomplete arcs on opposite sides, and section 7c the symmetric
  half rings that fool the sharpness start alone.

GUI hook: no GUI harness existed in the repository, so the main window is
instantiated offscreen (``QT_QPA_PLATFORM=offscreen``,
``MPS_explorer.MPS_explorer()`` after a ``QApplication``) and
``_apply_z_range`` is called on it with synthetic arrays. Constructing
the window loads and re-saves the user's persisted settings file
(``mps_analysis_settings.json``), so the harness points the window's
``load_settings``/``save_settings`` at a temporary directory and checks
that the real file is byte-identical afterwards; the window's log file
(``logs/mps_explorer_<date>.log``) is still appended to. "Bit-identical
before and after the hook" is encoded as rule-based expectations that the
pre-hook code satisfies (the rule is the code's own: ``keep = (z > zmin)
& (z < zmax)``), so the same checks pass before and after the edit and
only the ``roi_indices_unfiltered`` checks change state.

Run:  venv\\Scripts\\python.exe validate_rings_h1.py   (from the repo root)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import contextlib
import functools
import io
import os
import shutil
import sys
import tempfile
import traceback
import warnings as warnings_module
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

# Must be set before any Qt import, for the GUI-hook section.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.cluster_quality import good_cluster_labels  # noqa: E402
from tools.mps_multisegment import (  # noqa: E402
    AxialSegment,
    find_axial_segments,
)
from tools.mps_periodicity import (  # noqa: E402
    ZPeriodicityResult,
    fit_z_periodicity,
)

# The module under test does not exist until the implementation lands; the
# harness must still run to the end (the GUI-hook section is independent)
# and report every module check as failed rather than crash at import.
COLUMNS_IMPORT_ERROR: Optional[str] = None
try:
    from tools.mps_columns import (  # noqa: E402
        AXIS_START_LAB_RINGS,
        AXIS_START_SHARPNESS,
        RingsParams,
        build_rings,
        posterior_ambiguous,
        ring_index_from_segments,
    )
except Exception as exc:  # noqa: BLE001
    COLUMNS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

PASSED = 0
FAILED = 0


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
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


def require_module() -> None:
    """Fail with the import error, not with a NameError further down."""
    if COLUMNS_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"tools.mps_columns not importable ({COLUMNS_IMPORT_ERROR})")


def need(state: Dict[str, Any], key: str) -> Any:
    """The result an earlier check of the section produced, or a clear error."""
    require_module()
    if key not in state:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return state[key]


# ============================================================ synthetic truth
@dataclass
class SyntheticAxon:
    """One synthetic axon in the laboratory frame, with its truth."""

    name: str
    x_nm: NDArray[np.float64]           # noisy, laboratory frame
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    true_x_nm: NDArray[np.float64]      # noise-free, laboratory frame
    true_y_nm: NDArray[np.float64]
    true_z_nm: NDArray[np.float64]
    frame: NDArray[np.int64]
    ring: NDArray[np.int64]             # 0..n_rings-1, ascending z
    cluster: NDArray[np.int64]          # global cluster id
    fluorophore: NDArray[np.int64]      # global fluorophore id
    lp_lateral_nm: NDArray[np.float64]
    lpz_nm: NDArray[np.float64]
    axis: NDArray[np.float64]           # (3,) true unit axis, z-component > 0
    tilt_deg: float
    semi_axes_nm: Tuple[float, float]
    ring_z_nm: Tuple[float, ...]
    n_clusters_per_ring: int
    n_fluor_per_cluster: int
    n_frames_per_fluor: int
    lateral_noise_nm: float
    axial_noise_nm: float

    @property
    def n_locs(self) -> int:
        return int(self.x_nm.size)


def rotation_z_to(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rodrigues rotation taking the z axis onto unit vector ``u``.

    Harness-local on purpose: the module's own ``rodrigues_rotation`` is
    what is being tested, so the truth must not be built with it.
    """
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    k = np.cross(z, u)
    s = float(np.linalg.norm(k))
    c = float(np.dot(z, u))
    if s < 1e-12:
        return np.eye(3)
    k = k / s
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    rot: NDArray[np.float64] = np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)
    return rot


def make_synthetic_axon(
    seed: int,
    *,
    name: str,
    a_nm: float,
    b_nm: float,
    n_clusters: int,
    tilt_deg: float,
    azimuth_deg: float,
    ring_z_nm: Tuple[float, ...] = (-190.0, 0.0, 190.0),
    n_fluor: int = 4,
    n_per: int = 5,
    total_frames: int = 60000,
    lateral_noise_nm: float = 8.0,
    axial_noise_nm: float = 35.0,
    fluor_spread_nm: float = 8.0,
    missing_arc_deg: Optional[Tuple[Optional[Tuple[float, float]], ...]] = None,
) -> SyntheticAxon:
    """
    Rings of clusters on an ellipse, tilted rigidly, in the laboratory frame.

    ``missing_arc_deg``, one entry per ring, removes the clusters whose
    parametric angle lies within (centre_deg +/- width_deg / 2) of that
    ring (None keeps the ring complete): rings with incomplete arcs, as
    in real sections. With the arcs absent the random draws are the same
    as without the argument.

    In the axon frame each ring is a plane z' = ring_z with ``n_clusters``
    cluster centres at known parametric angles on the ellipse (a, b). A
    cluster is ``n_fluor`` fluorophores at Gaussian offsets of
    ``fluor_spread_nm`` from its centre (8 nm: with 4 x 5 localizations
    and min_samples = 10 DBSCAN then keeps every localization but one or
    two, so the centroid check below is not at the mercy of border
    losses); each fluorophore emits ``n_per`` consecutive frames from a
    random start in [0, total_frames). Localization noise is Gaussian,
    ``lateral_noise_nm`` per lateral axis and ``axial_noise_nm`` in z'.
    The whole set is rotated so that the axon axis points along
    (sin b cos phi, sin b sin phi, cos b).
    """
    rng = np.random.default_rng(seed)
    rows: List[Tuple[float, ...]] = []
    fl = 0
    for r, zr in enumerate(ring_z_nm):
        offset = rng.uniform(0.0, 2.0 * np.pi / n_clusters)
        t = offset + 2.0 * np.pi * np.arange(n_clusters) / n_clusters
        cx, cy = a_nm * np.cos(t), b_nm * np.sin(t)
        keep = np.ones(n_clusters, dtype=bool)
        arc = None if missing_arc_deg is None else missing_arc_deg[r]
        if arc is not None:
            centre, width = arc
            away = (np.degrees(t) - centre + 180.0) % 360.0 - 180.0
            keep = np.abs(away) > width / 2.0
        for j in np.flatnonzero(keep).tolist():
            for _k in range(n_fluor):
                px = cx[j] + rng.normal(0.0, fluor_spread_nm)
                py = cy[j] + rng.normal(0.0, fluor_spread_nm)
                start = int(rng.integers(0, total_frames - n_per))
                for m in range(n_per):
                    rows.append((r, r * n_clusters + j, fl, start + m, px, py, zr))
                fl += 1
    table = np.array(rows, dtype=float)
    ring = table[:, 0].astype(np.int64)
    cluster = table[:, 1].astype(np.int64)
    fluor = table[:, 2].astype(np.int64)
    frame = table[:, 3].astype(np.int64)
    tx, ty, tz = table[:, 4], table[:, 5], table[:, 6]
    n = tx.size
    x = tx + rng.normal(0.0, lateral_noise_nm, n)
    y = ty + rng.normal(0.0, lateral_noise_nm, n)
    z = tz + rng.normal(0.0, axial_noise_nm, n)

    b = np.radians(tilt_deg)
    ph = np.radians(azimuth_deg)
    u = np.array([np.sin(b) * np.cos(ph), np.sin(b) * np.sin(ph), np.cos(b)])
    rot = rotation_z_to(u)
    lab = rot @ np.vstack([x, y, z])
    lab_true = rot @ np.vstack([tx, ty, tz])
    return SyntheticAxon(
        name=name,
        x_nm=lab[0], y_nm=lab[1], z_nm=lab[2],
        true_x_nm=lab_true[0], true_y_nm=lab_true[1], true_z_nm=lab_true[2],
        frame=frame, ring=ring, cluster=cluster, fluorophore=fluor,
        lp_lateral_nm=np.full(n, lateral_noise_nm),
        lpz_nm=np.full(n, axial_noise_nm),
        axis=u, tilt_deg=tilt_deg, semi_axes_nm=(a_nm, b_nm),
        ring_z_nm=tuple(ring_z_nm), n_clusters_per_ring=n_clusters,
        n_fluor_per_cluster=n_fluor, n_frames_per_fluor=n_per,
        lateral_noise_nm=lateral_noise_nm, axial_noise_nm=axial_noise_nm,
    )


def temporal_overlap_fraction(
    x: NDArray[np.float64], y: NDArray[np.float64], frame: NDArray[np.int64],
    fluor: NDArray[np.int64], radius_nm: float, max_frame_gap: int,
) -> float:
    """Fraction of localizations with another fluorophore's localization
    within ``radius_nm`` laterally and ``max_frame_gap`` frames in time --
    the situations in which chain linking can merge two fluorophores."""
    tree = cKDTree(np.column_stack([x, y]))
    pairs = np.array(sorted(tree.query_pairs(radius_nm)), dtype=int).reshape(-1, 2)
    if pairs.size == 0:
        return 0.0
    i, j = pairs[:, 0], pairs[:, 1]
    bad = (np.abs(frame[i] - frame[j]) <= max_frame_gap) & (fluor[i] != fluor[j])
    hit = np.zeros(x.size, dtype=bool)
    hit[i[bad]] = True
    hit[j[bad]] = True
    return float(hit.mean())


def ramanujan_perimeter_nm(a: float, b: float) -> float:
    return float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def into_frame(frame: Any, x: NDArray[np.float64], y: NDArray[np.float64],
               z: NDArray[np.float64]) -> Tuple[NDArray[np.float64], ...]:
    """p' = R (p - origin) + origin with the module's own frame, so the
    truth is compared in exactly the coordinates the module reports."""
    rot = np.asarray(frame.rotation, dtype=float)
    o = np.asarray(frame.origin_nm, dtype=float).reshape(3, 1)
    p = rot @ (np.vstack([x, y, z]) - o) + o
    return p[0], p[1], p[2]


def ring_assignment(res: Any, n: int) -> NDArray[np.int64]:
    """Ring position per input localization from ``Ring.loc_index``; -1 outside."""
    out = np.full(n, -1, dtype=np.int64)
    for k, ring in enumerate(res.rings):
        out[np.asarray(ring.loc_index, dtype=np.intp)] = k
    return out


def ring_agreement(
    assigned: NDArray[np.int64], truth: NDArray[np.int64]
) -> Tuple[float, Dict[int, int]]:
    """Best one-to-one match of found rings to true rings; fraction of
    localizations whose ring is right under that match."""
    k = int(assigned.max()) + 1 if assigned.size else 0
    n_true = int(truth.max()) + 1
    if k <= 0:
        return 0.0, {}
    conf = np.zeros((k, n_true))
    for r in range(k):
        for t in range(n_true):
            conf[r, t] = np.count_nonzero((assigned == r) & (truth == t))
    ri, ci = linear_sum_assignment(-conf)
    return float(conf[ri, ci].sum() / truth.size), dict(zip(ri.tolist(), ci.tolist()))


def segment_of(res: Any, ring: Any) -> AxialSegment:
    for seg in res.ms.segments:
        if seg.index == ring.index:
            found: AxialSegment = seg
            return found
    raise AssertionError(f"ring index {ring.index} has no segment in ms.segments")


def direct_ambiguous(
    z: NDArray[np.float64], zr: ZPeriodicityResult, posterior_min: float
) -> NDArray[np.bool_]:
    """Max component posterior < posterior_min, dominant components only,
    weights renormalised -- computed here from first principles."""
    m = np.asarray(zr.means_nm, dtype=float)
    s = np.asarray(zr.sigmas_nm, dtype=float)
    w = np.asarray(zr.weights, dtype=float)
    w = w / w.sum()
    logp = (np.log(w)[None, :] - np.log(s)[None, :]
            - 0.5 * ((z[:, None] - m[None, :]) / s[None, :]) ** 2)
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p /= p.sum(axis=1, keepdims=True)
    flags: NDArray[np.bool_] = p.max(axis=1) < posterior_min
    return flags


def default_params(**kw: Any) -> Any:
    return RingsParams(**kw)


# ============================================================ 0. the truth
SMALL = make_synthetic_axon(
    0, name="small", a_nm=800.0, b_nm=550.0, n_clusters=24,
    tilt_deg=2.0, azimuth_deg=30.0)


def test_synthetic_truth() -> None:
    print("\n0. SYNTHETIC AXON (seed 0): 3 rings x 24 clusters x 4 fluorophores x 5 frames")

    def overlap_rare():
        # Linking looks max_dark_time + 1 = 2 frames ahead; the default link
        # radius is 20 nm and the +/-5 % check uses 40 nm, so both radii
        # (and the spec's 30 nm) must see < 1 % of localizations exposed to
        # a merge with another fluorophore.
        fr = {r: temporal_overlap_fraction(SMALL.x_nm, SMALL.y_nm, SMALL.frame,
                                           SMALL.fluorophore, r, 2)
              for r in (30.0, 40.0)}
        assert all(v < 0.01 for v in fr.values()), fr
        return (f"N={SMALL.n_locs}, overlap within 30 nm {100*fr[30.0]:.2f} %, "
                f"within 40 nm {100*fr[40.0]:.2f} %")

    def three_resolvable_rings_in_raw_z():
        # The design premise, with the existing segmentation only: 35 nm
        # axial noise against 190 nm spacing (boundary at 2.7 sigma) must
        # give three dominant components in valley mode even before the
        # 2 deg tilt is removed (it mixes at most 1600*tan(2 deg) = 56 nm).
        segs, zres, valleys, _ = find_axial_segments(
            SMALL.z_nm, half_width_nm=90.0, mode="valley", min_locs=10)
        assert len(segs) == 3, [(s.center_nm, s.n_locs) for s in segs]
        centres = np.array([s.center_nm for s in segs])
        assert np.all(np.abs(centres - np.array(SMALL.ring_z_nm)) < 20.0), centres
        return (f"centres {np.round(centres, 1).tolist()} nm, sigmas "
                f"{np.round(zres.sigmas_nm, 1).tolist()} nm, valley depths "
                f"{np.round(valleys.relative_depth, 3).tolist()}")

    check("synthetic axon: temporal overlap of different fluorophores < 1 %",
          overlap_rare)
    check("synthetic axon: raw z has three resolvable rings (existing segmentation)",
          three_resolvable_rings_in_raw_z)


# ============================================================ 1. ring index
def test_ring_index_from_segments() -> None:
    print("\n1. ring_index_from_segments")

    def hard_mask_minus_one_outside():
        require_module()
        segs, *_ = find_axial_segments(
            SMALL.z_nm, half_width_nm=90.0, mode="valley", min_locs=10)
        idx = np.asarray(ring_index_from_segments(SMALL.z_nm, segs))
        expected = np.full(SMALL.n_locs, -1, dtype=np.int64)
        for k, s in enumerate(segs):
            expected[(SMALL.z_nm >= s.zmin_nm) & (SMALL.z_nm <= s.zmax_nm)] = k
        assert idx.shape == expected.shape and np.array_equal(idx, expected), \
            np.flatnonzero(idx != expected)[:10]
        return (f"{int((idx >= 0).sum())} in a ring, "
                f"{int((idx < 0).sum())} outside (-1)")

    def overlapping_slabs_go_to_nearer_centre():
        require_module()
        # Two hand-built overlapping slabs: A = [-100, 60] about -50,
        # B = [0, 150] about 100. No z below sits at equal distance from
        # both centres, so "nearer centre" is unambiguous.
        a = AxialSegment(index=0, center_nm=-50.0, zmin_nm=-100.0, zmax_nm=60.0,
                         weight=0.5, sigma_nm=40.0, n_locs=0, component_index=0)
        b = AxialSegment(index=1, center_nm=100.0, zmin_nm=0.0, zmax_nm=150.0,
                         weight=0.5, sigma_nm=40.0, n_locs=0, component_index=1)
        z = np.array([-120.0, -50.0, 10.0, 20.0, 30.0, 55.0, 120.0, 200.0])
        expected = np.array([-1, 0, 0, 0, 1, 1, 1, -1])
        idx = np.asarray(ring_index_from_segments(z, [a, b]))
        assert np.array_equal(idx, expected), (idx, expected)
        return f"z {z.tolist()} -> {idx.tolist()}"

    check("ring_index_from_segments: hard mask per segment, -1 outside",
          hard_mask_minus_one_outside)
    check("ring_index_from_segments: overlapping slabs -> nearer centre",
          overlapping_slabs_go_to_nearer_centre)


# ============================================================ 2. build_rings
def run_build_rings(axon: SyntheticAxon, **kw: Any) -> Any:
    params = kw.pop("params", None)
    if params is None:
        params = default_params()
    return build_rings(
        axon.x_nm.copy(), axon.y_nm.copy(), axon.z_nm.copy(),
        frame=kw.pop("frame", axon.frame.copy()),
        lp_lateral_nm=kw.pop("lp_lateral_nm", axon.lp_lateral_nm.copy()),
        lpz_nm=kw.pop("lpz_nm", axon.lpz_nm.copy()),
        params=params, source_name=axon.name, pixel_size_nm=None, **kw)


def test_build_rings_primary() -> None:
    print("\n2. build_rings, primary path (defaults: valley mode, tilt corrected)")
    st: Dict[str, Any] = {}
    axon = SMALL

    def runs():
        require_module()
        res = run_build_rings(axon)
        st["res"] = res
        st["assigned"] = ring_assignment(res, axon.n_locs)
        st["truth_p"] = into_frame(res.frame, axon.true_x_nm, axon.true_y_nm,
                                   axon.true_z_nm)
        return (f"{len(res.rings)} rings, {len(res.warnings)} warnings, "
                f"link radius {res.link_radius_nm} nm, n_events {res.n_events}, "
                f"axis start '{res.axis_start}'")

    def pixel_size_source_forwarded():
        # The default "unknown" makes analyze_axon warn on every ring (provenance
        # lost); a caller that knows the source passes it and the warning goes.
        res = need(st, "res")
        assert all(r.analysis.pixel_size_source == "unknown" for r in res.rings)
        assert all(any("did not come from" in w for w in r.warnings) for r in res.rings), (
            "the provenance warning of analyze_axon lives in each ring's warnings")
        known = build_rings(
            axon.x_nm.copy(), axon.y_nm.copy(), axon.z_nm.copy(),
            frame=axon.frame.copy(), lp_lateral_nm=axon.lp_lateral_nm.copy(),
            lpz_nm=axon.lpz_nm.copy(), params=default_params(), source_name=axon.name,
            pixel_size_nm=113.0, pixel_size_source="yaml")
        assert len(known.rings) == len(res.rings)
        assert all(r.analysis.pixel_size_source == "yaml" for r in known.rings)
        assert all(r.analysis.pixel_size_nm == 113.0 for r in known.rings)
        assert not any("did not come from" in w for r in known.rings for w in r.warnings)
        assert not any("did not come from" in w for w in known.warnings), known.warnings[:5]
        same = all(np.array_equal(a.loc_index, b.loc_index) for a, b in zip(res.rings, known.rings))
        assert same, "provenance must not change the rings"
        return (f"default: source 'unknown' + warning; with pixel_size_source='yaml': "
                f"no warning, {len(known.rings)} identical rings")

    def three_rings():
        res = need(st, "res")
        assert len(res.rings) == 3, [(r.z_lo_nm, r.z_hi_nm) for r in res.rings]
        assert len(res.ms.segments) == 3 and all(a is not None for a in res.ms.analyses)
        return "3 rings, 3 segments analysed"

    def ordered_disjoint():
        res = need(st, "res")
        lo = np.array([r.z_lo_nm for r in res.rings])
        hi = np.array([r.z_hi_nm for r in res.rings])
        assert np.all(np.diff(lo) > 0) and np.all(hi[:-1] < lo[1:]), (lo, hi)
        assert np.all(hi > lo)
        return " | ".join(f"[{a:.0f}, {b:.0f}]" for a, b in zip(lo, hi)) + " nm"

    def bounds_from_segments():
        res = need(st, "res")
        for ring in res.rings:
            seg = segment_of(res, ring)
            assert ring.z_lo_nm == seg.zmin_nm and ring.z_hi_nm == seg.zmax_nm
            assert ring.centre_z_nm == seg.center_nm, (ring.centre_z_nm, seg.center_nm)
            assert ring.sigma_z_nm == seg.sigma_nm
            assert ring.component_index == seg.component_index
            assert ring.guard_nm == res.params.guard_nm == 0.0
            assert ring.analysis is res.ms.analyses[res.ms.segments.index(seg)]
        return "z_lo/z_hi/centre/sigma/component_index/analysis identical to ms"

    def tilt_recovered():
        # Slope error of one plane ~ sigma_z / (sqrt(N) * sd(x)) =
        # 35 / (sqrt(480) * 566) rad = 0.16 deg; three rings ~ 0.09 deg.
        # 0.5 deg is > 5 sd (0.14 deg measured with the same fit outside
        # the module).
        res = need(st, "res")
        beta = float(res.frame.beta_deg)
        u = np.asarray(res.frame.u, dtype=float)
        assert abs(beta - axon.tilt_deg) < 0.5, beta
        assert u[2] > 0 and res.frame.method == "plane_per_ring", res.frame.method
        ang = np.degrees(np.arccos(np.clip(np.dot(u, axon.axis), -1, 1)))
        assert ang < 0.5, ang
        return (f"beta {beta:.3f} deg (truth {axon.tilt_deg}), angle to true axis "
                f"{ang:.3f} deg, delta_beta {res.frame.delta_beta_deg:.3f} deg")

    def frame_coordinates_consistent():
        res = need(st, "res")
        xp, yp, zp = into_frame(res.frame, axon.x_nm, axon.y_nm, axon.z_nm)
        d = max(np.abs(res.x_p - xp).max(), np.abs(res.y_p - yp).max(),
                np.abs(res.z_p - zp).max())
        assert res.x_p.shape == (axon.n_locs,) and d < 1e-6, d
        return f"max |p' - (R(p - o) + o)| = {d:.2e} nm"

    def axis_starts_agree():
        # On complete rings both starts must converge to the same axis:
        # the spec's start is then kept, both refined tilts are within the
        # 0.5 deg of the tilt check, and no disagreement is reported.
        res = need(st, "res")
        betas = np.asarray(res.axis_candidate_beta_deg, dtype=float)
        assert betas.shape == (2,) and np.all(np.isfinite(betas)), betas
        assert np.all(np.abs(betas - axon.tilt_deg) < 0.5), betas
        assert res.axis_start == AXIS_START_LAB_RINGS, res.axis_start
        assert not any("axis starts converged" in w for w in res.warnings), res.warnings
        assert np.isfinite(res.frame.profile_sharpness) and res.frame.profile_sharpness > 0
        stages = np.asarray(res.axis_stages_beta_deg)
        assert stages.size >= 2 and stages[-1] < 0.05, stages
        return (f"refined tilts {np.round(betas, 3).tolist()} deg, start "
                f"'{res.axis_start}', stages {np.round(stages, 3).tolist()}, "
                f"sharpness {res.frame.profile_sharpness:.4f}")

    def origin_is_mean_of_used_rings():
        # origin_nm = mean of the localizations of the rings that entered
        # the last axis stage. That stage's segmentation differs from the
        # final rings by a few boundary localizations; each moves the
        # mean of ~1400 localizations spread over +-800 nm by < 0.6 nm,
        # so the mean over the final rings agrees to a few nm: accept 5.
        res = need(st, "res")
        used = np.unique(np.concatenate([np.asarray(r.loc_index) for r in res.rings]))
        assert used.size > 0.9 * axon.n_locs
        mean = np.array([axon.x_nm[used].mean(), axon.y_nm[used].mean(),
                         axon.z_nm[used].mean()])
        d = float(np.abs(np.asarray(res.frame.origin_nm) - mean).max())
        assert d < 5.0, d
        assert res.frame.n_locs_used > 0.9 * axon.n_locs, res.frame.n_locs_used
        return f"|origin - mean(used rings)| = {d:.2f} nm, n_locs_used {res.frame.n_locs_used}"

    def loc_index_is_hard_mask():
        res = need(st, "res")
        for ring in res.rings:
            li = np.asarray(ring.loc_index, dtype=np.intp)
            mask = (res.z_p >= ring.z_lo_nm) & (res.z_p <= ring.z_hi_nm)
            assert np.array_equal(np.sort(li), np.flatnonzero(mask)), ring.index
            assert np.array_equal(li, np.asarray(ring.analysis.slab_index)), ring.index
            assert ring.n_locs == li.size
        return "loc_index == flatnonzero(mask) == analysis.slab_index; n_locs matches"

    def ring_labels_agree_with_truth():
        # 35 nm axial noise against 90-95 nm to the nearest boundary:
        # expected loss 1 - Phi(90/35) = 0.5 % per side, ~1 % overall.
        res = need(st, "res")
        frac, mapping = ring_agreement(st["assigned"], axon.ring)
        assert frac >= 0.97, frac
        assert all(mapping.get(k) == k for k in range(len(res.rings))), mapping
        return f"{100*frac:.2f} % of {axon.n_locs} localizations, rings in truth order"

    def centroid_equals_loc_mean():
        # Cluster.centroid_nm is the unrounded mean: the spec's 1e-6 nm
        # holds there. good_cluster_centroids rounds to 2 decimals:
        # 0.005 nm is the largest deviation rounding alone can produce
        # between that mean and the an.centroids row.
        res = need(st, "res")
        worst_exact = 0.0
        worst_row = 0.0
        n = 0
        for ring in res.rings:
            an = ring.analysis
            for i, cl in enumerate(ring.clusters):
                li = np.asarray(cl.loc_index, dtype=np.intp)
                mean = np.array([res.x_p[li].mean(), res.y_p[li].mean()])
                worst_exact = max(worst_exact, np.abs(mean - np.asarray(cl.centroid_nm)).max())
                worst_row = max(worst_row, np.abs(mean - np.asarray(an.centroids[i])).max())
                n += 1
        assert n > 0 and worst_exact <= 1e-6, worst_exact
        assert worst_row <= 0.005 + 1e-6, worst_row
        return (f"{n} clusters, max |mean(loc) - centroid_nm| = {worst_exact:.1e} nm, "
                f"max |mean(loc) - an.centroids[i]| = {worst_row:.4f} nm")

    def cluster_bookkeeping():
        res = need(st, "res")
        for ring in res.rings:
            an = ring.analysis
            labels = good_cluster_labels(an.labels, an.bad_report.bad_labels)
            assert len(ring.clusters) == len(labels) == an.n_clusters_kept
            ring_li = np.asarray(ring.loc_index, dtype=np.intp)
            for i, cl in enumerate(ring.clusters):
                assert cl.ring == ring.index and cl.label == int(labels[i]), (cl.label, labels[i])
                expected = ring_li[an.labels == labels[i]]
                li = np.asarray(cl.loc_index, dtype=np.intp)
                assert np.array_equal(np.sort(li), np.sort(expected)), cl.label
                assert cl.n_locs == li.size
                assert np.array_equal(np.asarray(cl.z_values_nm), res.z_p[li])
        return "label order == good_cluster_labels; loc_index/n_locs/z_values match"

    def cluster_areas_line_up():
        res = need(st, "res")
        n = 0
        for ring in res.rings:
            areas = ring.analysis.areas
            assert areas is not None
            for cl in ring.clusters:
                k = np.flatnonzero(np.asarray(areas.labels) == cl.label)
                assert k.size == 1, cl.label
                assert cl.area_nm2 == float(areas.areas_nm2[k[0]]), cl.label
                assert abs(cl.rho_nm - np.sqrt(cl.area_nm2 / np.pi)) < 1e-9
                n += 1
        return f"{n} clusters: area_nm2 == an.areas by label, rho == sqrt(area/pi)"

    def clusters_found_near_truth():
        # Centroid of 20 localizations with 8 nm noise: sd 1.8 nm per axis;
        # 5.7 nm was the largest miss measured with the existing pipeline;
        # 15 nm leaves room for one or two border localizations lost.
        res = need(st, "res")
        txp, typ, _ = st["truth_p"]
        worst = 0.0
        kept = []
        for k, ring in enumerate(res.rings):
            cents = np.array([cl.centroid_nm for cl in ring.clusters], dtype=float).reshape(-1, 2)
            kept.append(len(ring.clusters))
            assert len(ring.clusters) >= 22, len(ring.clusters)
            for c in np.unique(axon.cluster[axon.ring == k]):
                mm = np.flatnonzero(axon.cluster == c)
                tc = np.array([txp[mm].mean(), typ[mm].mean()])
                d = float(np.min(np.hypot(cents[:, 0] - tc[0], cents[:, 1] - tc[1])))
                worst = max(worst, d)
        assert worst < 15.0, worst
        return f"kept per ring {kept}, farthest true centre from a kept centroid {worst:.1f} nm"

    def event_id_complete():
        res = need(st, "res")
        ev = np.asarray(res.event_id)
        assert ev.shape == (axon.n_locs,) and not np.any(ev < 0)
        assert res.n_events == np.unique(ev).size
        assert res.link_radius_nm == default_params().link_radius_factor * 8.0
        return f"{res.n_events} events over {axon.n_locs} localizations, radius {res.link_radius_nm} nm"

    def ring_events_bookkeeping():
        res = need(st, "res")
        ev = np.asarray(res.event_id)
        out = []
        for ring in res.rings:
            li = np.asarray(ring.loc_index, dtype=np.intp)
            assert np.array_equal(np.asarray(ring.event_id), ev[li]), ring.index
            ids = np.asarray(ring.event_id)
            assert ring.n_events == np.unique(ids[ids >= 0]).size, ring.index
            out.append(ring.n_events)
        return f"n_events per ring {out}"

    def cluster_events_bookkeeping():
        res = need(st, "res")
        ev = np.asarray(res.event_id)
        n = 0
        for ring in res.rings:
            for cl in ring.clusters:
                ids = ev[np.asarray(cl.loc_index, dtype=np.intp)]
                assert cl.n_events == np.unique(ids[ids >= 0]).size >= 1, cl.label
                n += 1
        return f"{n} clusters, n_events == distinct event ids of their localizations"

    def events_within_five_percent():
        # Default radius 5 * 8 = 40 nm = 3.5 step sd: exp(-5^2 / 4) = 0.2 %
        # of steps break, 4 steps per fluorophore -> < 1 % extra events;
        # temporal overlap within 40 nm is < 1 % (section 0). Both are
        # well inside the +/-5 % of the spec, checked here at the DEFAULT.
        res = need(st, "res")
        assert res.link_lost_step_fraction is not None
        assert abs(res.link_lost_step_fraction - np.exp(-25.0 / 4.0)) < 1e-12
        out = []
        for ring in res.rings:
            li = np.asarray(ring.loc_index, dtype=np.intp)
            n_true = np.unique(axon.fluorophore[li]).size
            assert abs(ring.n_events - n_true) <= 0.05 * n_true, (ring.n_events, n_true)
            out.append(f"{ring.n_events}/{n_true}")
        return ("events/fluorophores per ring " + ", ".join(out)
                + f"; lost-step fraction recorded {res.link_lost_step_fraction:.4f}")

    def lpz_median():
        res = need(st, "res")
        for ring in res.rings:
            assert abs(ring.lpz_median_nm - 35.0) < 1.0, ring.lpz_median_nm
            for cl in ring.clusters:
                assert abs(cl.lpz_median_nm - 35.0) < 1.0, cl.lpz_median_nm
        return "rings and clusters: lpz_median_nm = 35 nm"

    def relative_depths():
        # Two rings 190 nm apart with sigma 35-41 nm: the mixture density
        # at the valley is ~2 exp(-95^2 / (2 sigma^2)) of a peak, so the
        # relative depth is > 0.9 (0.94-0.95 measured); > 0.5 is asked.
        res = need(st, "res")
        r0, r1, r2 = res.rings
        assert r0.relative_depth_lo is None and r2.relative_depth_hi is None
        inner = [r0.relative_depth_hi, r1.relative_depth_lo, r1.relative_depth_hi,
                 r2.relative_depth_lo]
        assert all(v is not None and 0.5 < v <= 1.0 for v in inner), inner
        depths = np.asarray(res.ms.valleys.relative_depth)
        assert r0.relative_depth_hi == r1.relative_depth_lo == depths[0]
        assert r1.relative_depth_hi == r2.relative_depth_lo == depths[1]
        return f"inner boundaries {np.round(depths, 3).tolist()}, outer None"

    def contour_length_and_centre():
        # A 24-gon inscribed in the ellipse is shorter than it by
        # ~(pi/24)^2/6 = 0.3 %; the 15 % of the spec is generous.
        res = need(st, "res")
        expected = ramanujan_perimeter_nm(*axon.semi_axes_nm)
        gen_rot = rotation_z_to(axon.axis)
        lengths, offsets = [], []
        for k, ring in enumerate(res.rings):
            c_lab = gen_rot @ np.array([0.0, 0.0, axon.ring_z_nm[k]])
            cx, cy, _ = into_frame(res.frame, c_lab[:1], c_lab[1:2], c_lab[2:])
            per = ring.analysis.perimeter
            assert per is not None and ring.contour_nm is not None
            assert np.array_equal(np.asarray(ring.contour_nm), per.contour)
            assert ring.length_nm == per.perimeter_nm
            assert abs(ring.length_nm - expected) / expected < 0.15, ring.length_nm
            assert per.centre is not None and ring.centroid_nm is not None
            assert np.allclose(ring.centroid_nm, [per.centre.x_nm, per.centre.y_nm])
            # Area centroid of the inscribed polygon: a few nm of cluster
            # scatter (5.7 nm was the largest centroid miss); 25 nm.
            off = float(np.hypot(ring.centroid_nm[0] - cx[0], ring.centroid_nm[1] - cy[0]))
            assert off < 25.0, off
            lengths.append(ring.length_nm)
            offsets.append(off)
        return (f"lengths {np.round(lengths).astype(int).tolist()} nm vs Ramanujan "
                f"{expected:.0f} nm; centroid offsets {np.round(offsets, 1).tolist()} nm")

    def primary_has_no_ambiguity():
        res = need(st, "res")
        assert res.ambiguous is None and res.n_ambiguous == 0
        assert res.source_name == axon.name and res.params.posterior_min is None
        return "ambiguous None, n_ambiguous 0"

    check("build_rings runs on the synthetic axon", runs)
    check("build_rings returns 3 rings", three_rings)
    check("pixel_size_source is forwarded to every ring analysis (default 'unknown' warns; 'yaml' does not; rings unchanged)",
          pixel_size_source_forwarded)
    check("ring z bounds ordered and disjoint (valley mode)", ordered_disjoint)
    check("ring bounds, centre, sigma, component_index and analysis come from ms",
          bounds_from_segments)
    check("frame: tilt recovered within 0.5 deg of 2 deg (method plane_per_ring)",
          tilt_recovered)
    check("x_p/y_p/z_p equal R(p - origin) + origin of the returned frame",
          frame_coordinates_consistent)
    check("axis: both starts refine to the same axis (spec's start kept, no disagreement)",
          axis_starts_agree)
    check("frame.origin_nm is the mean of the localizations of the rings used (5 nm)",
          origin_is_mean_of_used_rings)
    check("loc_index equals the hard mask (z_p >= z_lo) & (z_p <= z_hi) and slab_index",
          loc_index_is_hard_mask)
    check(">= 97 % of localizations get the true ring label (rings in truth order)",
          ring_labels_agree_with_truth)
    check("cluster centroid_nm == mean(loc_index) to 1e-6 nm; an.centroids row to 0.005 nm (2-decimal rounding)",
          centroid_equals_loc_mean)
    check("cluster label/loc_index/n_locs/z_values follow good_cluster_labels order",
          cluster_bookkeeping)
    check("cluster area_nm2 and rho_nm line up with an.areas by label",
          cluster_areas_line_up)
    check(">= 22 kept clusters per ring; every true cluster centre within 15 nm",
          clusters_found_near_truth)
    check("event_id has no -1, covers every input localization; radius = factor * median lp",
          event_id_complete)
    check("ring event_id and n_events are the bookkeeping of event_id[loc_index]",
          ring_events_bookkeeping)
    check("cluster n_events is the bookkeeping of event_id[cluster.loc_index]",
          cluster_events_bookkeeping)
    check("n_events per ring within 5 % of the true fluorophores emitting in it (DEFAULT radius 40 nm)",
          events_within_five_percent)
    check("lpz_median_nm == 35 nm within 1 nm (rings and clusters)", lpz_median)
    check("relative depths at the two inner boundaries (> 0.5), None at the outer ones",
          relative_depths)
    check("contour length within 15 % of the Ramanujan perimeter; contour/centroid from an.perimeter",
          contour_length_and_centre)
    check("primary path: ambiguous is None and n_ambiguous == 0", primary_has_no_ambiguity)


def test_events_vs_fluorophores() -> None:
    print("\n3. Link radius sensitivity: the spec's link_radius_factor = 2.5")
    st: Dict[str, Any] = {}
    axon = SMALL

    def runs():
        require_module()
        assert default_params().link_radius_factor == 5.0
        res = run_build_rings(axon, params=default_params(link_radius_factor=2.5))
        st["res"] = res
        assert res.link_radius_nm == 20.0, res.link_radius_nm
        return f"link radius {res.link_radius_nm} nm, {res.n_events} events"

    def overcount_as_predicted():
        # Radius 20 nm vs step sd 11.3 nm: exp(-2.5^2 / 4) = 21 % of steps
        # break the chain (max_dark_time = 1 heals part of it), so the
        # ratio of events to fluorophores is ~1.7 (1.66 measured): never
        # the 5 of a per-localization count, never below 1, and above the
        # 1.05 the +/-5 % acceptance allows -- the reason 2.5 is not the
        # default. The recorded lost-step fraction must say so.
        res = need(st, "res")
        assert res.link_lost_step_fraction is not None
        assert abs(res.link_lost_step_fraction - np.exp(-2.5 ** 2 / 4.0)) < 1e-12
        ratios = []
        for ring in res.rings:
            li = np.asarray(ring.loc_index, dtype=np.intp)
            ratios.append(ring.n_events / np.unique(axon.fluorophore[li]).size)
        assert all(1.05 < r <= 2.0 for r in ratios), ratios
        return ("events / true fluorophores per ring " + str([round(r, 3) for r in ratios])
                + f"; lost-step fraction recorded {res.link_lost_step_fraction:.3f}")

    check("build_rings runs with link_radius_factor = 2.5 (radius 20 nm; default is 5)", runs)
    check("factor 2.5 overcounts events (ratio in (1.05, 2]) and records lost-step fraction 0.21",
          overcount_as_predicted)


# ============================================================ 4. posterior
def test_posterior() -> None:
    print("\n4. posterior_min: ambiguous localizations")
    st: Dict[str, Any] = {}
    axon = SMALL

    def direct_two_component():
        require_module()
        # A hand-built mixture: two dominant components plus one discarded
        # low-weight component (present in all_* only), so "dominant
        # components only, weights renormalised" is exercised.
        rng = np.random.default_rng(0)
        z = np.concatenate([rng.normal(0.0, 40.0, 2200), rng.normal(190.0, 40.0, 1800)])
        zr = ZPeriodicityResult(
            n_components=3, means_nm=np.array([0.0, 190.0]),
            weights=np.array([0.55, 0.42]), sigmas_nm=np.array([40.0, 40.0]),
            delta_z_nm=np.array([190.0]), main_peak_nm=0.0, bic_by_n={},
            n_discarded_components=1, converged=True, warnings=[],
            all_means_nm=np.array([0.0, 190.0, 400.0]),
            all_weights=np.array([0.55, 0.42, 0.03]),
            all_sigmas_nm=np.array([40.0, 40.0, 60.0]))
        got = np.asarray(posterior_ambiguous(z, zr, 0.85), dtype=bool)
        exp = direct_ambiguous(z, zr, 0.85)
        assert got.shape == exp.shape and got.dtype == bool
        assert np.array_equal(got, exp), int((got != exp).sum())
        assert 0 < exp.sum() < z.size
        return f"{int(exp.sum())} of {z.size} flagged, 0 disagreements"

    def runs():
        require_module()
        res = run_build_rings(axon, params=default_params(posterior_min=0.85))
        st["res"] = res
        return f"{len(res.rings)} rings, n_ambiguous {res.n_ambiguous}"

    def ambiguous_reported():
        # ~1 % expected: |z - boundary| < ln(0.85/0.15) sigma^2 / 190 =
        # 11 nm around boundaries 2.7 sigma from the ring centres.
        res = need(st, "res")
        amb = np.asarray(res.ambiguous)
        assert amb is not None and amb.dtype == bool and amb.shape == (axon.n_locs,)
        assert res.n_ambiguous > 0 and int(amb.sum()) == res.n_ambiguous
        assert len(res.rings) == 3, len(res.rings)
        frac = res.n_ambiguous / axon.n_locs
        hits = [w for w in res.warnings
                if "ambiguous" in w.lower()
                and ("%" in w or str(res.n_ambiguous) in w)]
        assert hits, res.warnings
        direct = direct_ambiguous(res.z_p, fit_z_periodicity(res.z_p), 0.85)
        assert np.array_equal(amb, direct), int((amb != direct).sum())
        return (f"n_ambiguous {res.n_ambiguous} ({100*frac:.2f} %), equals the direct "
                f"computation on fit_z_periodicity(z_p); warning: {hits[0][:70]}...")

    def rings_exclude_ambiguous():
        res = need(st, "res")
        amb = np.asarray(res.ambiguous)
        n_in = 0
        for ring in res.rings:
            li = np.asarray(ring.loc_index, dtype=np.intp)
            assert li.size == ring.n_locs and np.all(li >= 0) and np.all(li < axon.n_locs)
            assert not np.any(amb[li]), ring.index
            mask = (res.z_p >= ring.z_lo_nm) & (res.z_p <= ring.z_hi_nm) & ~amb
            assert np.array_equal(np.sort(li), np.flatnonzero(mask)), ring.index
            for cl in ring.clusters:
                assert np.all(np.isin(np.asarray(cl.loc_index), li)), cl.label
            n_in += li.size
        frac, _ = ring_agreement(ring_assignment(res, axon.n_locs), axon.ring)
        assert frac >= 0.96, frac
        return f"{n_in} localizations in rings, none ambiguous; ring agreement {100*frac:.2f} %"

    def non_finite_not_flagged():
        require_module()
        rng = np.random.default_rng(1)
        z = np.concatenate([rng.normal(0.0, 40.0, 500), rng.normal(190.0, 40.0, 500)])
        zr = fit_z_periodicity(z)
        zz = np.concatenate([z, [np.nan, np.inf, -np.inf, 1e6]])
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("error")      # any numpy warning fails the check
            got = np.asarray(posterior_ambiguous(zz, zr, 0.85), dtype=bool)
        assert got.shape == zz.shape and not np.any(got[-4:-1]), got[-4:]
        assert np.array_equal(got[:-4], np.asarray(posterior_ambiguous(z, zr, 0.85), dtype=bool))
        return "nan/inf/-inf -> False without a RuntimeWarning; finite flags unchanged"

    check("posterior_ambiguous flags exactly max posterior < 0.85 (direct numpy)",
          direct_two_component)
    check("posterior_ambiguous: non-finite z are not flagged and raise no numpy warning",
          non_finite_not_flagged)
    check("build_rings runs with posterior_min = 0.85", runs)
    check("posterior_min=0.85: n_ambiguous > 0, ambiguous.sum() == n_ambiguous, 3 rings, warning with the fraction",
          ambiguous_reported)
    check("posterior_min=0.85: ring loc_index excludes ambiguous localizations and maps to input indices",
          rings_exclude_ambiguous)


# ============================================================ 5. absent inputs
def test_absent_inputs() -> None:
    print("\n5. Without frame numbers / without lateral precision")
    axon = SMALL

    def without_frame():
        require_module()
        res = run_build_rings(axon, frame=None)
        assert res.n_events is None and res.event_id is None
        for ring in res.rings:
            assert ring.n_events is None and ring.event_id is None, ring.index
            for cl in ring.clusters:
                assert cl.n_events is None, cl.label
        hits = [w for w in res.warnings if "frame" in w.lower() and "event" in w.lower()]
        assert hits, res.warnings
        assert len(res.rings) == 3
        return (f"{len(res.rings)} rings, link_radius_nm {res.link_radius_nm}; "
                f"warning: {hits[0][:70]}")

    def without_lateral_precision():
        require_module()
        res = run_build_rings(axon, lp_lateral_nm=None)
        assert res.link_radius_nm == default_params().link_radius_default_nm == 25.0, res.link_radius_nm
        hits = [w for w in res.warnings
                if "radius" in w.lower() or "precision" in w.lower()]
        assert hits, res.warnings
        assert res.event_id is not None and res.n_events is not None
        return f"link radius {res.link_radius_nm} nm, {res.n_events} events; warning: {hits[0][:70]}"

    def empty_inputs():
        require_module()
        out = []
        for n in (0, 1):
            xs = np.arange(n, dtype=float)
            res = build_rings(xs, xs, xs, frame=np.arange(n))
            assert res.rings == [] and res.n_rings == 0 and res.ms.n_segments == 0
            assert res.frame.method == "identity" and res.frame.is_identity
            assert res.event_id is not None and res.n_events == n
            assert any("empty result" in w for w in res.warnings), res.warnings
            res2 = build_rings(xs, xs, xs, params=default_params(posterior_min=0.85))
            assert res2.n_events is None and res2.n_ambiguous == 0
            assert res2.ambiguous is not None and res2.ambiguous.shape == (n,)
            out.append(f"n={n}: 0 rings, {len(res.warnings)} warnings")
        return "; ".join(out)

    def frame_dtypes():
        require_module()
        nan_frame = axon.frame.astype(float)
        nan_frame[0] = np.nan
        for bad, what in (((axon.frame % 2).astype(bool), "bool"),
                          (axon.frame + 0.7, "fractional float"),
                          (nan_frame, "non-finite")):
            try:
                run_build_rings(axon, frame=bad)
            except ValueError as exc:
                assert "frame" in str(exc).lower(), exc
            else:
                raise AssertionError(f"{what} frames accepted")
        as_float = run_build_rings(axon, frame=axon.frame.astype(float))
        as_int = run_build_rings(axon)
        assert np.array_equal(np.asarray(as_float.event_id), np.asarray(as_int.event_id))
        return "bool, fractional and NaN frames raise ValueError; whole-number floats link identically"

    check("without frame: n_events and event_id are None (result, rings, clusters) and a warning says so",
          without_frame)
    check("without lp_lateral_nm: default link radius 25 nm used and a warning says so",
          without_lateral_precision)
    check("0 or 1 localizations: empty result with a warning, no exception", empty_inputs)
    check("frame numbers: booleans, fractional and non-finite values are rejected", frame_dtypes)


# ============================================================ 6. determinism
def test_determinism() -> None:
    print("\n6. Determinism")
    axon = SMALL

    def two_calls_identical():
        require_module()
        a = run_build_rings(axon)
        b = run_build_rings(axon)
        assert len(a.rings) == len(b.rings)
        assert a.warnings == b.warnings, (a.warnings, b.warnings)
        assert np.array_equal(a.frame.rotation, b.frame.rotation)
        assert np.array_equal(np.asarray(a.event_id), np.asarray(b.event_id))
        for ra, rb in zip(a.rings, b.rings):
            assert np.array_equal(np.asarray(ra.loc_index), np.asarray(rb.loc_index))
            assert ra.warnings == rb.warnings
            assert len(ra.clusters) == len(rb.clusters)
            for ca, cb in zip(ra.clusters, rb.clusters):
                assert np.array_equal(np.asarray(ca.centroid_nm), np.asarray(cb.centroid_nm))
                assert np.array_equal(np.asarray(ca.loc_index), np.asarray(cb.loc_index))
        return f"{len(a.rings)} rings, {sum(len(r.clusters) for r in a.rings)} clusters, {len(a.warnings)} warnings identical"

    check("determinism: two calls give identical loc_index, centroids and warnings",
          two_calls_identical)


# ============================================================ 7. tilt matters
def tilt_case(axon: SyntheticAxon, label: str) -> None:
    st: Dict[str, Any] = {}

    def corrected():
        require_module()
        res = run_build_rings(axon, params=default_params(correct_tilt=True))
        frac, _ = ring_agreement(ring_assignment(res, axon.n_locs), axon.ring)
        st["corrected"] = frac
        st["beta"] = float(res.frame.beta_deg)
        st["n_rings_c"] = len(res.rings)
        assert frac >= 0.95, f"agreement {frac:.3f}, beta_hat {res.frame.beta_deg:.3f} deg"
        return (f"agreement {100*frac:.1f} %, beta_hat {res.frame.beta_deg:.3f} deg "
                f"(truth {axon.tilt_deg}), {len(res.rings)} rings")

    def uncorrected():
        require_module()
        res = run_build_rings(axon, params=default_params(correct_tilt=False))
        frac, _ = ring_agreement(ring_assignment(res, axon.n_locs), axon.ring)
        st["uncorrected"] = frac
        assert res.frame.method == "identity"
        assert np.array_equal(res.x_p, axon.x_nm) and np.array_equal(res.z_p, axon.z_nm)
        assert np.allclose(res.frame.rotation, np.eye(3))
        return f"agreement {100*frac:.1f} %, {len(res.rings)} rings, identity frame"

    def difference():
        require_module()
        c, u = st.get("corrected"), st.get("uncorrected")
        assert c is not None and u is not None, "one of the two runs did not complete"
        assert c - u >= 0.20, f"corrected {c:.3f} vs uncorrected {u:.3f}"
        return f"corrected {100*c:.1f} % vs uncorrected {100*u:.1f} % ({100*(c-u):.1f} pp)"

    check(f"tilt matters ({label}): correct_tilt=True agrees with truth >= 95 %", corrected)
    check(f"tilt matters ({label}): correct_tilt=False gives the identity frame and untouched coordinates",
          uncorrected)
    check(f"tilt matters ({label}): correct_tilt=False is worse by >= 20 pp", difference)


def test_opposite_arcs() -> None:
    print("\n7b. Rings with incomplete arcs on opposite sides (footprints differ between rings)")
    print("   ring 0 missing an arc at 0 deg, ring 2 the same arc at 180 deg; truth beta 0 and 2 deg")
    # The failure this guards against: a plane through ALL localizations
    # reads the z-dependent shift of the cloud's centre as a tilt (1.7 deg
    # on the 8 um axon at truth 0, measured), rotates the rings into each
    # other and the per-ring stages cannot recover (45-56 % agreement).
    # Criterion: agreement >= 95 % as in section 7 and |beta_hat - beta|
    # < 0.5 deg (> 2.5 sd of one plane's slope error on a partial ring
    # of ~350 localizations: 35 / (sqrt(350) * 500) rad = 0.2 deg).
    for a, b, ncl, seed, tag in ((800.0, 550.0, 24, 10, "1.6 x 1.1 um"),
                                 (4000.0, 4000.0, 60, 11, "8 um")):
        for tilt in (0.0, 2.0):
            for arc in (90.0, 120.0, 180.0):
                axon = make_synthetic_axon(
                    seed, name=f"arcs-{tag}-{tilt}-{arc}", a_nm=a, b_nm=b,
                    n_clusters=ncl, tilt_deg=tilt, azimuth_deg=30.0,
                    missing_arc_deg=((0.0, arc), None, (180.0, arc)))

                def case(axon: SyntheticAxon = axon, tag: str = tag, tilt: float = tilt,
                         arc: float = arc, a: float = a) -> str:
                    require_module()
                    res = run_build_rings(axon)
                    frac, _ = ring_agreement(ring_assignment(res, axon.n_locs), axon.ring)
                    beta = float(res.frame.beta_deg)
                    assert len(res.rings) == 3, len(res.rings)
                    assert frac >= 0.95, f"agreement {frac:.3f}"
                    assert abs(beta - tilt) < 0.5, f"beta_hat {beta:.3f} (truth {tilt})"
                    disagreed = any("axis starts converged" in w for w in res.warnings)
                    if a >= 4000.0 and tilt > 0.0:
                        # D * tan(2 deg) = 280 nm > P/2: the spec's lab-frame
                        # start cannot recover here (0.2-0.3 deg measured),
                        # so the sharpness start must win and say so.
                        assert res.axis_start == AXIS_START_SHARPNESS, res.axis_start
                        assert disagreed, res.warnings
                    return (f"beta_hat {beta:.3f}, agreement {100*frac:.1f} %, start "
                            f"'{res.axis_start}', candidates "
                            f"{np.round(res.axis_candidate_beta_deg, 2).tolist()}"
                            + (", disagreement warned" if disagreed else ""))

                check(f"opposite arcs {arc:.0f} deg, {tag}, tilt {tilt} deg: agreement >= 95 %, |beta_hat - beta| < 0.5",
                      case)


def test_half_rings() -> None:
    print("\n7c. Symmetric half rings (ring 0 the x < 0 half, ring 2 the x > 0 half, ring 1 complete)")
    # A shear that lifts ring 0 by one period and lowers ring 2 by one
    # period stacks all three on the middle ring: the global profile
    # sharpness rewards that (its start lands at 5-8 deg for truth 0),
    # so the choice between the starts must catch it. With sigma_z = 35
    # nm both starts refine to the same axis; with sigma_z = 50 nm the
    # sharpness start stays stacked (2 rings, shallow valleys) and the
    # specification's start must win on valley depth. Tilt tolerance from
    # the per-plane slope error of a half ring of 480 localizations,
    # sigma_z / (sqrt(480) * 250 nm): 0.36 deg at 35 nm, 0.52 deg at 50
    # nm, halved in the count-weighted axis: accept 0.5 deg (~3 sd) and
    # 1.0 deg (~4 sd). Agreement: 95 % at 35 nm as elsewhere; at 50 nm
    # the hard mask itself misplaces 2 (1 - Phi(95 / 50)) = 5.7 % of a
    # ring, so 90 %.
    for sigma, tol_deg, min_agree in ((35.0, 0.5, 0.95), (50.0, 1.0, 0.90)):
        for tilt in (0.0, 2.0):
            axon = make_synthetic_axon(
                5, name=f"half-{sigma:.0f}-{tilt}", a_nm=800.0, b_nm=550.0,
                n_clusters=24, tilt_deg=tilt, azimuth_deg=30.0, n_fluor=8,
                axial_noise_nm=sigma,
                missing_arc_deg=((0.0, 180.0), None, (180.0, 180.0)))

            def case(axon: SyntheticAxon = axon, sigma: float = sigma, tilt: float = tilt,
                     tol_deg: float = tol_deg, min_agree: float = min_agree) -> str:
                require_module()
                res = run_build_rings(axon)
                frac, _ = ring_agreement(ring_assignment(res, axon.n_locs), axon.ring)
                beta = float(res.frame.beta_deg)
                cands = np.asarray(res.axis_candidate_beta_deg, dtype=float)
                depth = np.asarray(res.axis_candidate_valley_depth, dtype=float)
                assert len(res.rings) == 3, len(res.rings)
                assert frac >= min_agree, f"agreement {frac:.3f}"
                assert abs(beta - tilt) < tol_deg, f"beta_hat {beta:.3f} (truth {tilt})"
                disagreed = any("axis starts converged" in w for w in res.warnings)
                if sigma >= 50.0:
                    assert abs(cands[1] - tilt) > 3.0, cands      # the sharpness start stacked the halves
                    assert res.axis_start == AXIS_START_LAB_RINGS and disagreed, (res.axis_start, res.warnings)
                    assert depth[0] > depth[1], depth
                return (f"beta_hat {beta:.3f}, agreement {100*frac:.1f} %, start "
                        f"'{res.axis_start}', candidates {np.round(cands, 2).tolist()}, "
                        f"valley depths {np.round(depth, 2).tolist()}"
                        + (", disagreement warned" if disagreed else ""))

            check(f"half rings, sigma_z {sigma:.0f} nm, tilt {tilt} deg: |beta_hat - beta| < {tol_deg}, "
                  f"agreement >= {100*min_agree:.0f} %"
                  + (", stacked sharpness start rejected on valley depth" if sigma >= 50.0 else ""),
                  case)


def test_tilt_matters() -> None:
    print("\n7. Tilt matters: large axon, D * tan(beta) > P/2")
    # Spec case: 8 um diameter, 3 deg -> 8000 * tan(3 deg) = 419 nm.
    big = make_synthetic_axon(
        1, name="big-8um-3deg", a_nm=4000.0, b_nm=4000.0, n_clusters=60,
        tilt_deg=3.0, azimuth_deg=30.0)
    print(f"   spec case: D = 8 um, beta = 3 deg, D*tan(beta) = "
          f"{8000*np.tan(np.radians(3)):.0f} nm, N = {big.n_locs}")
    tilt_case(big, "spec case D=8 um, 3 deg")
    # Beyond the spec: still D*tan(beta) = 210 nm > P/2 = 95 nm, but within
    # reach of a fit that iterates segmentation and plane fit.
    mid = make_synthetic_axon(
        2, name="big-4um-3deg", a_nm=2000.0, b_nm=2000.0, n_clusters=40,
        tilt_deg=3.0, azimuth_deg=30.0)
    print(f"   moderate case (beyond spec): D = 4 um, beta = 3 deg, D*tan(beta) = "
          f"{4000*np.tan(np.radians(3)):.0f} nm, N = {mid.n_locs}")
    tilt_case(mid, "moderate case beyond spec, D=4 um, 3 deg")


# ============================================================ 8. GUI hook
def test_gui_hook() -> None:
    print("\n8. GUI HOOK: MPS_explorer._apply_z_range keeps the spatial selection's indices")
    try:
        from PyQt5 import QtWidgets
    except Exception as exc:  # noqa: BLE001
        print(f"  note  PyQt5 not importable ({exc}); GUI hook checks skipped")
        return
    print("   approach: main window instantiated offscreen "
          "(MPS_explorer.MPS_explorer() after QApplication); no GUI harness "
          "existed in the repository")

    st: Dict[str, Any] = {}
    rng = np.random.default_rng(0)
    n = 600
    x = rng.uniform(0.0, 2000.0, n)
    y = rng.uniform(0.0, 2000.0, n)
    z = np.where(rng.uniform(size=n) < 0.55,
                 rng.normal(0.0, 40.0, n), rng.normal(190.0, 40.0, n))
    ind = np.flatnonzero(np.hypot(x - 1000.0, y - 1000.0) < 700.0)
    zmin, zmax = -60.0, 60.0

    def need_window() -> Any:
        if "window" not in st:
            raise RuntimeError("window unavailable: construction failed")
        return st["window"]

    repo_root = os.path.dirname(os.path.abspath(__file__))
    settings_file = os.path.join(repo_root, "mps_analysis_settings.json")
    settings_before: Optional[Tuple[bytes, int]] = None
    if os.path.exists(settings_file):
        with open(settings_file, "rb") as fh:
            settings_before = (fh.read(), os.stat(settings_file).st_mtime_ns)
    tmp_dir = tempfile.mkdtemp(prefix="mps_h1_settings_")

    def window_builds():
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        st["app"] = app
        # The window's own logging attaches a console handler at
        # construction; redirecting stderr during construction keeps it
        # out of the harness output (the log file is written regardless).
        # Its settings are read from and written to a temporary directory
        # so that a validation run never rewrites the user's file.
        with contextlib.redirect_stderr(io.StringIO()):
            import MPS_explorer  # noqa: E402
            from tools import mps_settings
            MPS_explorer.load_settings = functools.partial(
                mps_settings.load_settings, directory=tmp_dir)
            MPS_explorer.save_settings = functools.partial(
                mps_settings.save_settings, directory=tmp_dir)
            window = MPS_explorer.MPS_explorer()
        st["window"] = window
        return (f"{type(window).__name__} built; ind_inside_roi has {ind.size} of {n}; "
                f"settings redirected to {tmp_dir}")

    def user_settings_untouched():
        # Run last: every save the window performs during the section
        # (construction, _apply_z_range, close) must have gone to tmp_dir.
        need_window()
        redirected = os.path.exists(os.path.join(tmp_dir, "mps_analysis_settings.json"))
        if settings_before is None:
            assert not os.path.exists(settings_file), "settings file created in the repo"
            return "no user settings file existed; none was created"
        with open(settings_file, "rb") as fh:
            after = fh.read()
        assert after == settings_before[0], "the user's mps_analysis_settings.json changed"
        assert os.stat(settings_file).st_mtime_ns == settings_before[1], \
            "the user's mps_analysis_settings.json was rewritten (mtime changed)"
        return (f"mps_analysis_settings.json untouched ({len(after)} bytes, same mtime); "
                f"window saved to the temporary directory: {redirected}")

    def fresh_window_default():
        w = need_window()
        assert hasattr(w, "roi_indices_unfiltered"), "attribute not initialised in __init__"
        assert w.roi_indices_unfiltered is None and w.roi_indices is None
        return "roi_indices_unfiltered is None on a fresh window"

    def prepare(w: Any, indices: Optional[NDArray[np.intp]], user_edited: bool,
                fields: Tuple[str, str]) -> None:
        w.x, w.y, w.z = x.copy(), y.copy(), z.copy()
        if indices is None:
            w.xroi, w.yroi = w.x, w.y
        else:
            w.xroi, w.yroi = w.x[indices], w.y[indices]
        w._z_range_user_edited = user_edited
        w.ui.lineEdit_zmin.setText(fields[0])
        w.ui.lineEdit_zmax.setText(fields[1])

    def with_roi_indices():
        w = need_window()
        ind_arg = ind.copy()
        prepare(w, ind_arg, True, (f"{zmin:.1f}", f"{zmax:.1f}"))
        w._apply_z_range(ind_arg)
        st["roi_done"] = True
        got = w.roi_indices_unfiltered
        assert got is not None, "roi_indices_unfiltered not set"
        got = np.asarray(got)
        assert np.array_equal(got, ind), got[:10]
        assert not np.shares_memory(got, ind_arg), "must be a copy of base"
        ind_arg[0] = -1                                   # the copy must not follow
        assert np.asarray(w.roi_indices_unfiltered)[0] == ind[0]
        assert np.array_equal(z[np.asarray(w.roi_indices_unfiltered)], w.zroi_unfiltered)
        return f"roi_indices_unfiltered == ind_inside_roi ({ind.size} indices), a copy"

    def unchanged_filtering():
        w = need_window()
        if not st.get("roi_done"):
            # The previous check raised before or during the call; re-run the
            # call so this check reports on the pre-existing behaviour.
            prepare(w, ind.copy(), True, (f"{zmin:.1f}", f"{zmax:.1f}"))
            w._apply_z_range(ind.copy())
        z_all = z[ind]
        keep = (z_all > zmin) & (z_all < zmax)
        assert w.zmin == zmin and w.zmax == zmax
        assert np.array_equal(w.roi_indices, ind[keep])
        assert np.array_equal(w.xroi, x[ind][keep])
        assert np.array_equal(w.yroi, y[ind][keep])
        assert np.array_equal(w.zroi, z_all[keep])
        assert w._applied_slab == (zmin, zmax)
        return f"roi_indices/xroi/yroi/zroi: {int(keep.sum())} of {ind.size} kept by ({zmin}, {zmax})"

    def unchanged_unfiltered():
        w = need_window()
        assert np.array_equal(w.xroi_unfiltered, x[ind])
        assert np.array_equal(w.yroi_unfiltered, y[ind])
        assert np.array_equal(w.zroi_unfiltered, z[ind])
        assert not np.shares_memory(w.xroi_unfiltered, w.xroi)
        return "xroi/yroi/zroi_unfiltered == spatial selection, copies"

    def whole_field():
        w = need_window()
        prepare(w, None, True, (f"{zmin:.1f}", f"{zmax:.1f}"))
        w._apply_z_range(None)
        keep = (z > zmin) & (z < zmax)
        assert np.array_equal(w.roi_indices, np.flatnonzero(keep))
        assert np.array_equal(w.zroi, z[keep]) and np.array_equal(w.xroi, x[keep])
        assert np.array_equal(w.zroi_unfiltered, z)
        got = w.roi_indices_unfiltered
        assert got is not None and np.array_equal(np.asarray(got), np.arange(n)), got
        return f"ind_inside_roi=None: roi_indices_unfiltered == arange({n}), {int(keep.sum())} kept"

    def prefill_path():
        w = need_window()
        prepare(w, ind.copy(), False, ("", ""))
        w._apply_z_range(ind.copy())
        assert w.ui.lineEdit_zmin.text() and w.ui.lineEdit_zmax.text(), "pre-fill did not run"
        assert w.zmin is not None and w.zmax is not None
        z_all = z[ind]
        keep = (z_all > w.zmin) & (z_all < w.zmax)
        assert np.array_equal(w.roi_indices, ind[keep])
        assert np.array_equal(w.zroi, z_all[keep])
        assert w._applied_slab is None
        got = w.roi_indices_unfiltered
        assert got is not None and np.array_equal(np.asarray(got), ind), got
        return (f"pre-filled ({w.zmin:.1f}, {w.zmax:.1f}) nm from the axial peak; "
                f"{int(keep.sum())} of {ind.size} kept; roi_indices_unfiltered == ind")

    check("GUI hook: main window builds offscreen", window_builds)
    check("GUI hook: fresh window initialises roi_indices_unfiltered to None",
          fresh_window_default)
    check("GUI hook: _apply_z_range(ind) -> roi_indices_unfiltered == ind_inside_roi (a copy)",
          with_roi_indices)
    check("GUI hook: roi_indices == ind[(z > zmin) & (z < zmax)], xroi/yroi/zroi filtered (unchanged)",
          unchanged_filtering)
    check("GUI hook: *_unfiltered arrays equal the spatial selection (unchanged)",
          unchanged_unfiltered)
    check("GUI hook: ind_inside_roi=None -> whole field, roi_indices_unfiltered == arange(N)",
          whole_field)
    def reload_resets():
        # Loading another file clears everything derived from the previous
        # one (load_channel1); the new attribute has to go with roi_indices,
        # or a consumer could index the new file with the old selection.
        # import_file is replaced so that no file and no dialog is needed.
        w = need_window()
        prepare(w, ind.copy(), True, (f"{zmin:.1f}", f"{zmax:.1f}"))
        w._apply_z_range(ind.copy())
        assert w.roi_indices_unfiltered is not None
        w.import_file = lambda path, fileformat, channel=1: (x.copy(), y.copy(), z.copy())
        assert w.load_channel1("synthetic-h1", 0) is True
        assert w.roi_indices_unfiltered is None, "stale roi_indices_unfiltered after reload"
        assert w.roi_indices is None and w.xroi_unfiltered is None and w.zroi_unfiltered is None
        return "load_channel1 clears roi_indices_unfiltered together with roi_indices"

    check("GUI hook: automatic pre-fill path (fields empty, not user-edited) stays consistent",
          prefill_path)
    check("GUI hook: loading another file resets roi_indices_unfiltered (with roi_indices, *_unfiltered)",
          reload_resets)

    w = st.get("window")
    if w is not None:
        try:
            w.close()
        except Exception:  # noqa: BLE001
            pass
    check("GUI hook: the user's settings file is not touched (content and mtime; window saves go to a temporary directory)",
          user_settings_untouched)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    print("=" * 72)
    print("RINGS H1 CHECKS: tools/mps_columns.py (build_rings) and the GUI hook")
    print("=" * 72)
    if COLUMNS_IMPORT_ERROR is not None:
        print(f"note: tools.mps_columns not importable -> every module check "
              f"fails ({COLUMNS_IMPORT_ERROR})")
    test_synthetic_truth()
    test_ring_index_from_segments()
    test_build_rings_primary()
    test_events_vs_fluorophores()
    test_posterior()
    test_absent_inputs()
    test_determinism()
    test_tilt_matters()
    test_opposite_arcs()
    test_half_rings()
    test_gui_hook()
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
