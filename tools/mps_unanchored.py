"""
Automatic discard of betaII-spectrin clusters that are not anchored to the membrane, without tubulin (package P14).

Why
---
The old discard (``tools.mps_axoplasm.anchored_clusters``) needs two widefield images, the tubulin mask and the
spectrin ring, and is a floor rather than a truth: on the pilot data it left a chain of clusters more than 400 nm
inside the ring because the spectrin interior did not close. New measurements no longer image tubulin. What
separated the clusters inside the axon from the membrane ones on the pilot data was their POSITION, not their size,
shape or z: how deep the cluster centre lies below the ring that the spectrin itself draws.

The rule
--------
1. A reference ring is drawn from the spectrin itself, one of two (decision D3, both selectable):

   * ``REFERENCE_POLAR`` (default): the polar contour of the slab's localizations
     (``tools.mps_auto_contour.polar_contour``: from the cloud's centre, the mode of the radial histogram in each of
     72 sectors). It uses thousands of localizations, not tens of centres, and does not depend on the final contour.
   * ``REFERENCE_SMOOTH``: a smooth closed curve through the cluster CENTRES, r(theta) with harmonics up to order 4
     around the area centroid of their convex hull, fitted with Tukey's bisquare so that interior centres get no
     weight.

2. The depth of each cluster centre is its signed distance to that ring, positive inside (both rings: the same
   definition, so a threshold means the same thing whichever ring is used).

3. Three states per cluster (decisions D1 and D2):

   * ``STATE_UNANCHORED`` -- deeper than ``exclude_nm`` (400 nm): excluded by the automatic rule;
   * ``STATE_DOUBTFUL`` -- deeper than ``doubt_nm`` (250 nm) and not deeper than ``exclude_nm``: only MARKED, never
     excluded automatically (the person decides by hand);
   * ``STATE_ANCHORED`` -- the rest;
   * ``STATE_NOT_EVALUATED`` -- when the ring could not be drawn (the reason is given): nothing is excluded in
     silence.

4. Decisions by hand (D11) overlay the automatic ones: exclude any cluster or restore one the rule excluded, with a
   reason and a date, kept in a JSON file beside the data (``decisions_path``) by POSITION (the cluster centre in
   nm, with a tolerance), so that they survive a re-analysis that renumbers the clusters.

5. The analysis is always reported with and without the discard (D4): ``compare_unanchored`` runs
   ``tools.mps_analysis.compare_discard`` with the final flags.

The old widefield method stays as an option (D5): ``final_flags(method="widefield", widefield=...)`` takes its
flags, and the export says where every discard came from.

Origin of every number (honest label): [pilot] measured on one pilot experiment (18 axons, one animal, unpublished);
[own sim] swept in this package's simulation (``test_unanchored.py``); [code] an existing constant of the programme;
[ad hoc] chosen here without further support. None comes from the literature; no published work found separates
peripheral from interior spectrin automatically.

Pure and deterministic: no Qt; the only random draws (the noise of the reference) use a fixed seed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_auto_contour as ac
from tools.unanchored_export import (
    CLUSTER_UNANCHORED_COLUMNS,
    UNANCHORED_COLUMNS,
    empty_cluster_row,
    not_run_block,
)

# ----------------------------------------------------------------------------------------------------
# Parameters, with their origin
# ----------------------------------------------------------------------------------------------------
DEFAULT_EXCLUDE_NM: float = 400.0
"""[pilot + ad hoc] (decision D1, PROVISIONAL until the 18 pilot axons are labelled by hand). Beyond 400 nm below the
polar ring, the pilot's tubulin (one ROI) put 71 % of the flagged clusters inside the axon against a 24 % background
for membrane clusters; 5 of 51 flags changed when 30 % of the localizations were left out; the 8 deep clusters the
widefield method found were all >= 737 nm deep. One experiment: not a validated value."""

DEFAULT_DOUBT_NM: float = 250.0
"""[code + ad hoc] (decision D2) Start of the doubtful band: the margin the widefield method already uses
(``tools.mps_axoplasm.DEFAULT_MARGIN_NM``). Between 250 and 400 nm the pilot's tubulin did not decide (36 % inside,
32 % at the edge) and most of the instability sat there; a cluster in a real bay of the membrane can be ~260 nm
deep. Only marked, never excluded automatically."""

REFERENCE_POLAR: str = "polar_localizations"
REFERENCE_SMOOTH: str = "smooth_centres"
REFERENCES: Tuple[str, ...] = (REFERENCE_POLAR, REFERENCE_SMOOTH)
DEFAULT_REFERENCE: str = REFERENCE_POLAR
"""(decision D3) The polar ring is the default: on the pilot data its depth moved by 21 nm (median; p90 54 nm) when
30 % of the localizations were left out, while the smooth curve of the centres changed 18 of 108 flags when 10 % of
the centres were left out."""

MIN_LOCALIZATIONS: int = 200
"""[code] Fewer slab localizations: no polar ring (``tools.mps_auto_contour.MIN_LOCALIZATIONS``)."""

SMOOTH_ORDER: int = 4
"""[ad hoc, pilot] Highest harmonic of the smooth curve of the centres (9 coefficients): enough for the oval and
lobed pilot sections, too stiff to follow one interior centre."""

SMOOTH_TUKEY_C: float = 4.685
"""Tukey's bisquare constant in units of the robust (MAD) residual scale: the usual choice, 95 % efficiency for
normal residuals (Beaton & Tukey 1974). [ad hoc] here: not tuned on these data."""

SMOOTH_MIN_CENTRES: int = 12
"""[ad hoc] Fewer centres: no smooth curve (9 coefficients need clearly more than 9 points)."""

SMOOTH_ITERATIONS: int = 30
"""[ad hoc] Iterations of the reweighted fit at most (it converged in fewer on every pilot axon)."""

SMOOTH_SAMPLES: int = 360
"""[ad hoc] Vertices of the smooth ring's polygon (1 degree)."""

NOISE_REPLICATES: int = 20
"""[ad hoc] Subsamples for the noise of the reference."""

NOISE_FRACTION_POLAR: float = 0.7
"""[ad hoc] Fraction of the localizations each polar subsample keeps (the pilot's stability test)."""

NOISE_FRACTION_SMOOTH: float = 0.9
"""[ad hoc] Fraction of the centres each smooth subsample keeps (the pilot's stability test)."""

NOISE_SEED: int = 0
"""[ad hoc] Seed of the subsampling: the same data give the same noise."""

BORDERLINE_NOISE_MULTIPLE: float = 2.0
"""[ad hoc] A cluster closer than this many of its own depth SDs to ``exclude_nm`` is marked borderline."""

MAX_FLAGGED_FRACTION: float = 0.20
"""[ad hoc, pilot] Above this fraction of the clusters excluded automatically a warning is raised: the problem is
then more likely the axon or the reference than the clusters (the idea of rejecting a whole structure with too much
inside, Thevathasan et al. 2019). With 400 nm the pilot's maximum was 19 %."""

INNER_BAND_MAX_ARC_NM: float = 1500.0
"""[own sim + ad hoc] Star-shape test of the cloud. Along 72 rays from the ring's centre, density peaks deeper than
``exclude_nm`` are linked from ray to ray when their radii differ by less than ``INNER_BAND_LINK_NM``; the longest
linked arc is the inner band. A membrane folded inward (a "C", two lobes) makes a long inner band, an interior lump
or a chain of lumps across the centre a short one. Above this arc the reference is REJECTED (the ring cannot
represent that membrane and its depths would exclude membrane). Simulation (test_unanchored.py, 2026-11-17): rings
with interior lumps of 20 localizations and a diametral chain of 13 lumps gave 0 nm; a thick "C" 3992 nm (polar) /
4386 nm (smooth); two separate discs 200 nm apart 1837 nm (the closest to the limit). Not checked on real axons."""

INNER_BAND_LINK_NM: float = 150.0
"""[ad hoc] Two inner peaks on neighbouring rays belong to one band when their radii differ by less than this
(~ the radial FWHM of the pilot's ring, 148 nm)."""

INNER_BAND_GAP_NM: float = 400.0
"""[ad hoc] Arc a band may skip between two rays that see it: the rays see the band only near its clusters, which
sit every ~250 nm on a membrane (pilot 1NN median 127 nm); 400 nm bridges one missing cluster."""

INNER_BAND_LEVEL_FRAC: float = 0.25
"""[ad hoc] An inner peak counts when the smoothed density exceeds this fraction of the reference level of the
density nodes (``tools.mps_auto_contour.reference_level``: a membrane cluster's level); background stays below
0.01 of it at 50 nm smoothing (``tools.mps_auto_contour.BYPASS_LEVEL_FRAC``)."""

INNER_BAND_N_RAYS: int = 72
"""[ad hoc] Rays of the star-shape test (5 degrees, the polar contour's sectors)."""

INNER_BAND_STEP_NM: float = 10.0
"""[ad hoc] Radial sampling step along each ray."""

MANUAL_TOLERANCE_NM: float = 50.0
"""[ad hoc] A decision by hand applies to the cluster whose centre is within this distance of the stored position
(about two cluster radii, r_eff ~21-25 nm on the pilot data): the same cluster found again after eps, min samples
or the slab change moves its centre by much less, and two clusters are rarely this close (1NN median 127 nm)."""

TRIM_DEPTH_NM: float = 250.0
"""[pilot + ad hoc] (decision D8, OFF by default) Depth below the polar ring beyond which localizations are left out
before the perimeter contour is fitted. On the pilot data, trimming every localization deeper than 250 nm made the
contour accepted on 17 of 18 axons (base 12). Not yet validated on the V1 simulation of the contour package: that is
why it stays off."""

# States and sources (the exported words)
STATE_ANCHORED: str = "anchored"
STATE_DOUBTFUL: str = "doubtful"
STATE_UNANCHORED: str = "unanchored"
STATE_NOT_EVALUATED: str = "not evaluated"
STATES: Tuple[str, ...] = (STATE_ANCHORED, STATE_DOUBTFUL, STATE_UNANCHORED, STATE_NOT_EVALUATED)

METHOD_OFF: str = "off"
METHOD_AUTOMATIC: str = "automatic"
METHOD_WIDEFIELD: str = "widefield"
METHODS: Tuple[str, ...] = (METHOD_OFF, METHOD_AUTOMATIC, METHOD_WIDEFIELD)

SOURCE_AUTOMATIC: str = "automatic"
SOURCE_WIDEFIELD: str = "widefield"
SOURCE_MANUAL: str = "manual"
SOURCE_RESTORED: str = "restored by hand"

ACTION_EXCLUDE: str = "exclude"
ACTION_RESTORE: str = "restore"
ACTIONS: Tuple[str, ...] = (ACTION_EXCLUDE, ACTION_RESTORE)

SUGGESTED_REASONS: Tuple[str, ...] = ("interior", "neighbouring axon", "glia", "artefact", "other")
"""Reasons the interface offers; any non-empty text is accepted."""

# Failure states of a reference ring
STATUS_OK: str = "ok"
FAIL_TOO_FEW_LOCS: str = "too few localizations"
FAIL_TOO_FEW_CENTRES: str = "too few cluster centres"
FAIL_NO_POLAR: str = "no polar contour"
FAIL_SMOOTH_FIT: str = "smooth fit failed"
FAIL_NOT_STAR: str = "cloud not star-shaped"
FAIL_UNKNOWN_REFERENCE: str = "unknown reference"
FAILURE_REASONS: Tuple[str, ...] = (FAIL_TOO_FEW_LOCS, FAIL_TOO_FEW_CENTRES, FAIL_NO_POLAR, FAIL_SMOOTH_FIT,
                                    FAIL_NOT_STAR, FAIL_UNKNOWN_REFERENCE)

DECISIONS_FORMAT: str = "MPS Explorer unanchored decisions v1"
DECISIONS_SUFFIX: str = "_unanchored.json"


class ManualDecisionsError(ValueError):
    """The decisions file cannot be read as a decisions file (its text says why)."""


# ----------------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------------
def _as_points(points: Any) -> NDArray[np.float64]:
    p = np.asarray(points, dtype=np.float64)
    if p.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    p = p.reshape(-1, 2)
    return p


def _as_xy(x_nm: Any, y_nm: Any) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y have different lengths ({x.size} and {y.size})")
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def points_in_polygon(points: Any, polygon: Any) -> NDArray[np.bool_]:
    """Even-odd rule: whether each point lies inside the closed polygon (vertices not repeated)."""
    p = _as_points(points)
    poly = _as_points(polygon)
    inside = np.zeros(p.shape[0], dtype=bool)
    if poly.shape[0] < 3 or p.shape[0] == 0:
        return inside
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    px, py = p[:, 0], p[:, 1]
    for k in range(poly.shape[0]):
        cond = (y1[k] > py) != (y2[k] > py)
        if not cond.any():
            continue
        dy = y2[k] - y1[k]
        xint = x1[k] + (py - y1[k]) * (x2[k] - x1[k]) / (dy if dy != 0.0 else 1e-300)
        inside ^= cond & (px < xint)
    return inside


def distance_to_polygon(points: Any, polygon: Any) -> NDArray[np.float64]:
    """Shortest distance (nm) from each point to the closed polygon's edges."""
    p = _as_points(points)
    a = _as_points(polygon)
    out = np.full(p.shape[0], np.inf)
    if a.shape[0] < 2 or p.shape[0] == 0:
        return out
    b = np.roll(a, -1, axis=0)
    ab = b - a
    den = np.einsum("ij,ij->i", ab, ab)
    den = np.where(den > 0.0, den, 1e-12)
    for i in range(a.shape[0]):
        t = np.clip(((p - a[i]) @ ab[i]) / den[i], 0.0, 1.0)
        q = a[i] + t[:, None] * ab[i]
        out = np.minimum(out, np.hypot(p[:, 0] - q[:, 0], p[:, 1] - q[:, 1]))
    return out


def signed_depth(points: Any, polygon: Any) -> NDArray[np.float64]:
    """Depth (nm) of each point below the closed polygon: distance to it, positive inside, negative outside."""
    d = distance_to_polygon(points, polygon)
    return np.where(points_in_polygon(points, polygon), d, -d)


def _area_centroid(poly: NDArray[np.float64]) -> Tuple[float, float, float]:
    """(cx, cy, signed area) of a closed polygon; the vertex mean when the area is ~0."""
    nxt = np.roll(poly, -1, axis=0)
    cross = poly[:, 0] * nxt[:, 1] - nxt[:, 0] * poly[:, 1]
    area = 0.5 * float(np.sum(cross))
    if abs(area) < 1e-9:
        return float(poly[:, 0].mean()), float(poly[:, 1].mean()), 0.0
    cx = float(np.sum((poly[:, 0] + nxt[:, 0]) * cross)) / (6.0 * area)
    cy = float(np.sum((poly[:, 1] + nxt[:, 1]) * cross)) / (6.0 * area)
    return cx, cy, area


def _counter_clockwise(poly: NDArray[np.float64]) -> NDArray[np.float64]:
    return poly if _area_centroid(poly)[2] >= 0.0 else poly[::-1].copy()


def _ring_radius_along(poly: NDArray[np.float64], centre: Tuple[float, float],
                       theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Radius of the ring at each angle, interpolated over its vertices sorted by angle around ``centre``
    (exact for the rings of this module, which are radial functions of the angle around a centre near it)."""
    dx, dy = poly[:, 0] - centre[0], poly[:, 1] - centre[1]
    th = np.arctan2(dy, dx)
    r = np.hypot(dx, dy)
    order = np.argsort(th)
    th, r = th[order], r[order]
    th_ext = np.concatenate([th - 2.0 * np.pi, th, th + 2.0 * np.pi])
    r_ext = np.tile(r, 3)
    return np.asarray(np.interp(theta, th_ext, r_ext), dtype=np.float64)


# ----------------------------------------------------------------------------------------------------
# Smooth curve of the centres
# ----------------------------------------------------------------------------------------------------
def _harmonic_design(theta: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    cols = [np.ones_like(theta)]
    for k in range(1, order + 1):
        cols += [np.cos(k * theta), np.sin(k * theta)]
    return np.column_stack(cols)


def _mad_sd(v: NDArray[np.float64]) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def robust_harmonic_fit(theta: NDArray[np.float64], r: NDArray[np.float64], *, order: int = SMOOTH_ORDER,
                        c: float = SMOOTH_TUKEY_C, iterations: int = SMOOTH_ITERATIONS) -> NDArray[np.float64]:
    """Coefficients of r(theta) = a0 + sum_k (a_k cos k theta + b_k sin k theta), k <= ``order``, by iteratively
    reweighted least squares with Tukey's bisquare (residual scale: MAD of the residuals)."""
    A = _harmonic_design(theta, order)
    beta = np.linalg.lstsq(A, r, rcond=None)[0]
    for _ in range(iterations):
        res = r - A @ beta
        s = _mad_sd(res)
        if not math.isfinite(s) or s <= 0.0:
            break
        u = res / (c * s)
        w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if int(np.count_nonzero(w)) <= A.shape[1]:
            break
        sw = np.sqrt(w)
        new = np.linalg.lstsq(A * sw[:, None], r * sw, rcond=None)[0]
        converged = bool(np.allclose(new, beta, rtol=1e-6, atol=1e-6))
        beta = new
        if converged:
            break
    return np.asarray(beta, dtype=np.float64)


def _hull_centroid(points: NDArray[np.float64]) -> Tuple[float, float]:
    from scipy.spatial import ConvexHull, QhullError
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return float(points[:, 0].mean()), float(points[:, 1].mean())
    cx, cy, _a = _area_centroid(points[hull.vertices])
    return cx, cy


def smooth_ring_polygon(centroids: Any, *, order: int = SMOOTH_ORDER,
                        samples: int = SMOOTH_SAMPLES) -> Tuple[Optional[NDArray[np.float64]], Tuple[float, float], str]:
    """(polygon or None, centre, reason when None) of the smooth curve through the cluster centres."""
    c = _as_points(centroids)
    c = c[np.isfinite(c).all(axis=1)]
    if c.shape[0] < SMOOTH_MIN_CENTRES:
        return None, (float("nan"), float("nan")), (
            f"{FAIL_TOO_FEW_CENTRES}: {c.shape[0]} (at least {SMOOTH_MIN_CENTRES})")
    centre = _hull_centroid(c)
    dx, dy = c[:, 0] - centre[0], c[:, 1] - centre[1]
    theta, r = np.arctan2(dy, dx), np.hypot(dx, dy)
    beta = robust_harmonic_fit(theta, r, order=order)
    t = -np.pi + (np.arange(samples) + 0.5) * 2.0 * np.pi / samples
    R = _harmonic_design(t, order) @ beta
    if not np.isfinite(R).all() or float(R.min()) <= 0.0:
        return None, centre, f"{FAIL_SMOOTH_FIT}: the fitted radius is not positive all round"
    poly = np.column_stack([centre[0] + R * np.cos(t), centre[1] + R * np.sin(t)]).astype(np.float64)
    return poly, centre, ""


# ----------------------------------------------------------------------------------------------------
# Star-shape test of the cloud around a ring
# ----------------------------------------------------------------------------------------------------
def inner_band_arc_nm(x_nm: Any, y_nm: Any, polygon: Any, centre: Tuple[float, float], *,
                      exclude_nm: float = DEFAULT_EXCLUDE_NM, n_rays: int = INNER_BAND_N_RAYS,
                      step_nm: float = INNER_BAND_STEP_NM, link_nm: float = INNER_BAND_LINK_NM,
                      gap_nm: float = INNER_BAND_GAP_NM,
                      level_frac: float = INNER_BAND_LEVEL_FRAC) -> float:
    """
    Arc length (nm) of the longest band of density deeper than ``exclude_nm`` below the ring that runs from ray to
    ray around ``centre``. 0 when nothing dense lies that deep. See ``INNER_BAND_MAX_ARC_NM``.
    """
    x, y = _as_xy(x_nm, y_nm)
    poly = _as_points(polygon)
    if x.size == 0 or poly.shape[0] < 3:
        return 0.0
    dm = ac.density_map(x, y, pixel_nm=ac.NODE_PIXEL_NM, sigma_nm=ac.NODE_SIGMA_NM)
    _nodes, weights = ac.density_nodes(x, y)
    level = float(level_frac) * ac.reference_level(weights)
    if not level > 0.0:
        return 0.0
    theta = -np.pi + (np.arange(n_rays) + 0.5) * 2.0 * np.pi / n_rays
    R = _ring_radius_along(poly, centre, theta)
    dtheta = 2.0 * np.pi / n_rays
    peaks: List[NDArray[np.float64]] = []
    for k in range(n_rays):
        r_max = float(R[k]) - float(exclude_nm)
        if r_max <= step_nm:
            peaks.append(np.zeros(0))
            continue
        rr = np.arange(0.0, r_max + 0.5 * step_nm, step_nm)
        pts = np.column_stack([centre[0] + rr * np.cos(theta[k]), centre[1] + rr * np.sin(theta[k])])
        v = dm.sample(pts)
        if v.size < 3:
            peaks.append(np.zeros(0))
            continue
        is_peak = np.zeros(v.size, dtype=bool)
        is_peak[1:-1] = (v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:])
        is_peak[-1] = v[-1] > v[-2]          # the band may reach past the exclusion depth
        is_peak &= v >= level
        peaks.append(rr[is_peak])
    if not any(p.size for p in peaks):
        return 0.0
    # Longest chain: walk twice round so a band crossing the angle origin is followed; a chain cannot exceed one
    # turn. A ray may miss the band between two clusters (they sit every ~250 nm and a 5 degree ray only sees a
    # cluster it passes within ~60 nm of), so a peak links to one up to ``gap_nm`` of arc back, not only to the
    # previous ray; the bridged arc counts.
    best = 0.0
    max_back = n_rays // 4
    hist: List[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []   # (r, length, rays)
    for step in range(2 * n_rays):
        k = step % n_rays
        cur_r = peaks[k]
        cur_len = cur_r * dtheta
        cur_rays = np.ones(cur_r.size)
        for i in range(cur_r.size):
            for back in range(1, min(max_back, len(hist)) + 1):
                if cur_r[i] * dtheta * (back - 1) > gap_nm:
                    break
                pr, pl, pn = hist[-back]
                if pr.size == 0:
                    continue
                ok = (np.abs(pr - cur_r[i]) <= link_nm) & (pn + back <= n_rays)
                if ok.any():
                    js = np.flatnonzero(ok)
                    j = js[int(np.argmax(pl[js]))]
                    cand = pl[j] + cur_r[i] * dtheta * back
                    if cand > cur_len[i]:
                        cur_len[i] = cand
                        cur_rays[i] = pn[j] + back
        if cur_len.size:
            best = max(best, float(cur_len.max()))
        hist.append((cur_r, cur_len, cur_rays))
        if len(hist) > max_back:
            hist.pop(0)
    return best


# ----------------------------------------------------------------------------------------------------
# Reference ring
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ReferenceRing:
    """One reference ring and whether it can be used. ``polygon`` is None when ``status`` is not "ok"."""

    kind: str
    status: str                                   # STATUS_OK or one of FAILURE_REASONS
    detail: str                                   # one sentence; "" when ok
    polygon: Optional[NDArray[np.float64]]        # (n, 2) nm, counter-clockwise
    centre: Tuple[float, float]
    inner_band_arc_nm: float

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.polygon is not None


def _failed_ring(kind: str, status: str, detail: str,
                 centre: Tuple[float, float] = (float("nan"), float("nan")),
                 arc: float = float("nan")) -> ReferenceRing:
    return ReferenceRing(kind=kind, status=status, detail=detail, polygon=None, centre=centre,
                         inner_band_arc_nm=arc)


def _raw_ring(kind: str, x: NDArray[np.float64], y: NDArray[np.float64],
              centroids: NDArray[np.float64]) -> Tuple[Optional[NDArray[np.float64]], Tuple[float, float], str, str]:
    """(polygon, centre, status, detail) before the star-shape test."""
    nan2 = (float("nan"), float("nan"))
    if kind == REFERENCE_POLAR:
        if x.size < MIN_LOCALIZATIONS:
            return None, nan2, FAIL_TOO_FEW_LOCS, f"{x.size} slab localizations (at least {MIN_LOCALIZATIONS})"
        poly = ac.polar_contour(x, y)
        if poly is None:
            return None, nan2, FAIL_NO_POLAR, "fewer than a third of the polar sectors hold enough localizations"
        poly = _counter_clockwise(np.asarray(poly, dtype=np.float64))
        cx, cy, _a = _area_centroid(poly)
        return poly, (cx, cy), STATUS_OK, ""
    if kind == REFERENCE_SMOOTH:
        poly_s, centre, why = smooth_ring_polygon(centroids)
        if poly_s is None:
            status = FAIL_TOO_FEW_CENTRES if why.startswith(FAIL_TOO_FEW_CENTRES) else FAIL_SMOOTH_FIT
            return None, centre, status, why
        return _counter_clockwise(poly_s), centre, STATUS_OK, ""
    return None, nan2, FAIL_UNKNOWN_REFERENCE, f"{kind!r} is not one of {', '.join(REFERENCES)}"


def reference_ring(kind: str, x_nm: Any, y_nm: Any, centroids: Any, *,
                   exclude_nm: float = DEFAULT_EXCLUDE_NM, check_star: bool = True) -> ReferenceRing:
    """The reference ring ``kind`` of this axon, checked (``x_nm``/``y_nm``: the slab's localizations; ``centroids``:
    the kept cluster centres, nm). A ring that cannot be drawn, or around which the cloud is not star-shaped, comes
    back with its failure status and no polygon."""
    x, y = _as_xy(x_nm, y_nm)
    c = _as_points(centroids)
    poly, centre, status, detail = _raw_ring(kind, x, y, c)
    if poly is None:
        return _failed_ring(kind, status, detail, centre)
    arc = float("nan")
    if check_star and x.size:
        arc = inner_band_arc_nm(x, y, poly, centre, exclude_nm=exclude_nm)
        if arc > INNER_BAND_MAX_ARC_NM:
            return _failed_ring(kind, FAIL_NOT_STAR,
                                f"a dense band runs {arc:,.0f} nm along the inside of the ring, deeper than "
                                f"{exclude_nm:,.0f} nm (limit {INNER_BAND_MAX_ARC_NM:,.0f} nm): the membrane folds "
                                f"inward and this ring cannot represent it", centre, arc)
    return ReferenceRing(kind=kind, status=STATUS_OK, detail="", polygon=poly, centre=centre, inner_band_arc_nm=arc)


def depth_noise(kind: str, x_nm: Any, y_nm: Any, centroids: Any, *, replicates: int = NOISE_REPLICATES,
                seed: int = NOISE_SEED) -> NDArray[np.float64]:
    """Per-cluster SD (nm) of the depth over ``replicates`` subsamples of the reference's own input (localizations
    for the polar ring, centres for the smooth one). nan where fewer than two replicates gave a ring."""
    x, y = _as_xy(x_nm, y_nm)
    c = _as_points(centroids)
    k = c.shape[0]
    if replicates < 2 or k == 0:
        return np.full(k, np.nan)
    rng = np.random.default_rng(int(seed))
    dep = np.full((replicates, k), np.nan)
    for i in range(replicates):
        if kind == REFERENCE_POLAR:
            m = np.asarray(rng.random(x.size) < NOISE_FRACTION_POLAR, dtype=bool)
            poly, _c, status, _d = _raw_ring(kind, x[m], y[m], c)
        else:
            m = np.asarray(rng.random(k) < NOISE_FRACTION_SMOOTH, dtype=bool)
            poly, _c, status, _d = _raw_ring(kind, x, y, c[m])
        if poly is not None:
            dep[i] = signed_depth(c, poly)
    good = np.isfinite(dep).sum(axis=0)
    sd = np.full(k, np.nan)
    ok = good >= 2
    if ok.any():
        sd[ok] = np.nanstd(dep[:, ok], axis=0, ddof=1)
    return sd


# ----------------------------------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------------------------------
def _check_thresholds(exclude_nm: float, doubt_nm: float) -> None:
    if not (math.isfinite(exclude_nm) and math.isfinite(doubt_nm)):
        raise ValueError("the thresholds must be finite numbers")
    if doubt_nm < 0.0 or exclude_nm <= 0.0:
        raise ValueError("the thresholds must be positive")
    if doubt_nm > exclude_nm:
        raise ValueError(f"the doubtful band starts ({doubt_nm:g} nm) deeper than the exclusion ({exclude_nm:g} nm)")


def classify(depth_nm: Any, *, exclude_nm: float = DEFAULT_EXCLUDE_NM,
             doubt_nm: float = DEFAULT_DOUBT_NM) -> NDArray[np.str_]:
    """One state per depth: unanchored (> exclude_nm), doubtful (> doubt_nm), anchored, or not evaluated (nan)."""
    _check_thresholds(float(exclude_nm), float(doubt_nm))
    d = np.asarray(depth_nm, dtype=np.float64).reshape(-1)
    out = np.full(d.size, STATE_ANCHORED, dtype=object)
    with np.errstate(invalid="ignore"):
        out[d > doubt_nm] = STATE_DOUBTFUL
        out[d > exclude_nm] = STATE_UNANCHORED
    out[~np.isfinite(d)] = STATE_NOT_EVALUATED
    return np.asarray(out.astype(str))


@dataclass(frozen=True, eq=False)
class RingVerdict:
    """The depths and states of the clusters under one reference ring."""

    ring: ReferenceRing
    depth_nm: NDArray[np.float64]                 # per cluster, nan when not evaluated
    state: NDArray[np.str_]
    depth_sd_nm: NDArray[np.float64]              # per cluster noise, nan when not computed
    borderline: NDArray[np.bool_]

    @property
    def n_unanchored(self) -> int:
        return int(np.count_nonzero(self.state == STATE_UNANCHORED))

    @property
    def n_doubtful(self) -> int:
        return int(np.count_nonzero(self.state == STATE_DOUBTFUL))

    @property
    def noise_nm(self) -> float:
        v = self.depth_sd_nm[np.isfinite(self.depth_sd_nm)]
        return float(np.median(v)) if v.size else float("nan")


def _verdict(ring: ReferenceRing, centroids: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64],
             *, exclude_nm: float, doubt_nm: float, noise_replicates: int) -> RingVerdict:
    k = centroids.shape[0]
    if not ring.ok or ring.polygon is None:
        depth: NDArray[np.float64] = np.full(k, np.nan)
        sd: NDArray[np.float64] = np.full(k, np.nan)
    else:
        depth = signed_depth(centroids, ring.polygon)
        sd = (depth_noise(ring.kind, x, y, centroids, replicates=noise_replicates)
              if noise_replicates >= 2 else np.full(k, np.nan))
    state = classify(depth, exclude_nm=exclude_nm, doubt_nm=doubt_nm)
    with np.errstate(invalid="ignore"):
        border = np.isfinite(sd) & (np.abs(depth - exclude_nm) < BORDERLINE_NOISE_MULTIPLE * sd)
    return RingVerdict(ring=ring, depth_nm=depth, state=state, depth_sd_nm=sd, borderline=np.asarray(border))


@dataclass(frozen=True, eq=False)
class UnanchoredResult:
    """What the automatic rule says about the clusters of one axon (``auto_unanchored``)."""

    labels: NDArray[np.int64]                     # cluster labels, one per centroid row
    centroids: NDArray[np.float64]                # (K, 2) nm
    reference: str                                # the ring used
    exclude_nm: float
    doubt_nm: float
    used: RingVerdict
    other: Optional[RingVerdict]                  # the other ring, when it was compared (D3)
    warnings: Tuple[str, ...] = ()

    @property
    def evaluated(self) -> bool:
        return self.used.ring.ok

    @property
    def state(self) -> NDArray[np.str_]:
        return self.used.state

    @property
    def depth_nm(self) -> NDArray[np.float64]:
        return self.used.depth_nm

    @property
    def unanchored(self) -> NDArray[np.bool_]:
        return np.asarray(self.used.state == STATE_UNANCHORED)

    @property
    def doubtful(self) -> NDArray[np.bool_]:
        return np.asarray(self.used.state == STATE_DOUBTFUL)

    @property
    def n_changed_by_reference(self) -> Optional[int]:
        """Clusters whose state differs between the two rings (None when either ring could not be used)."""
        if self.other is None or not self.other.ring.ok or not self.used.ring.ok:
            return None
        return int(np.count_nonzero(self.other.state != self.used.state))

    @property
    def status(self) -> str:
        return STATUS_OK if self.evaluated else self.used.ring.status

    @property
    def failure(self) -> str:
        if self.evaluated:
            return ""
        return f"{self.used.ring.status}: {self.used.ring.detail}" if self.used.ring.detail else self.used.ring.status


def auto_unanchored(x_nm: Any, y_nm: Any, centroids: Any, labels: Optional[Any] = None, *,
                    reference: str = DEFAULT_REFERENCE, exclude_nm: float = DEFAULT_EXCLUDE_NM,
                    doubt_nm: float = DEFAULT_DOUBT_NM, compare_references: bool = True,
                    noise_replicates: int = NOISE_REPLICATES) -> UnanchoredResult:
    """
    The automatic rule on one axon. ``x_nm``/``y_nm``: the slab's localizations (nm); ``centroids``: the kept
    cluster centres (K, 2) nm; ``labels``: their cluster labels (default 0..K-1). Never excludes in silence: when the
    ring cannot be used every cluster is "not evaluated" and ``failure`` says why. ``compare_references`` also
    evaluates the other ring, for the count of clusters that change verdict with the ring (D3).
    """
    if reference not in REFERENCES:
        raise ValueError(f"unknown reference {reference!r}; one of {', '.join(REFERENCES)}")
    _check_thresholds(float(exclude_nm), float(doubt_nm))
    x, y = _as_xy(x_nm, y_nm)
    c = _as_points(centroids)
    if not np.isfinite(c).all():
        raise ValueError("a cluster centre is not a finite coordinate")
    lab = (np.arange(c.shape[0], dtype=np.int64) if labels is None
           else np.asarray(list(labels), dtype=np.int64).reshape(-1))
    if lab.size != c.shape[0]:
        raise ValueError(f"{lab.size} labels for {c.shape[0]} cluster centres")
    warnings: List[str] = []
    kinds = [reference] + ([k for k in REFERENCES if k != reference] if compare_references else [])
    verdicts: List[RingVerdict] = []
    for kind in kinds:
        ring = reference_ring(kind, x, y, c, exclude_nm=float(exclude_nm))
        verdicts.append(_verdict(ring, c, x, y, exclude_nm=float(exclude_nm), doubt_nm=float(doubt_nm),
                                 noise_replicates=int(noise_replicates)))
    used = verdicts[0]
    other = verdicts[1] if len(verdicts) > 1 else None
    if not used.ring.ok:
        warnings.append(f"automatic discard not evaluated ({used.ring.kind}): {used.ring.status}"
                        + (f": {used.ring.detail}" if used.ring.detail else "")
                        + "; nothing was excluded automatically")
    elif c.shape[0] and used.n_unanchored / c.shape[0] > MAX_FLAGGED_FRACTION:
        warnings.append(f"{used.n_unanchored} of {c.shape[0]} clusters ({100.0 * used.n_unanchored / c.shape[0]:.0f} %) "
                        f"are more than {exclude_nm:g} nm below the ring, above {100 * MAX_FLAGGED_FRACTION:.0f} %: "
                        f"check the axon and the ring before trusting the discard")
    if other is not None and not other.ring.ok:
        warnings.append(f"the other ring ({other.ring.kind}) could not be used: {other.ring.status}; the count of "
                        f"clusters that change verdict with the ring is not available")
    if used.borderline.any():
        warnings.append(f"{int(used.borderline.sum())} cluster(s) within {BORDERLINE_NOISE_MULTIPLE:g} noise SD of "
                        f"{exclude_nm:g} nm: their verdict depends on the subsample")
    return UnanchoredResult(labels=lab, centroids=c.copy(), reference=reference, exclude_nm=float(exclude_nm),
                            doubt_nm=float(doubt_nm), used=used, other=other, warnings=tuple(warnings))


# ----------------------------------------------------------------------------------------------------
# Decisions by hand (D11)
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ManualDecision:
    """One decision by hand on the cluster centred at (x_nm, y_nm)."""

    x_nm: float
    y_nm: float
    action: str                                   # ACTION_EXCLUDE or ACTION_RESTORE
    reason: str
    date: str                                     # ISO 8601, UTC
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"x_nm": round(float(self.x_nm), 1), "y_nm": round(float(self.y_nm), 1), "action": self.action,
                "reason": self.reason, "date": self.date, "note": self.note}


@dataclass(frozen=True)
class ManualDecisions:
    """The decisions by hand of one data file, as stored beside it."""

    source: str = ""                              # basename of the data file they were made on
    tolerance_nm: float = MANUAL_TOLERANCE_NM
    decisions: Tuple[ManualDecision, ...] = ()
    path: str = ""                                # where they were read from / will be written

    def __len__(self) -> int:
        return len(self.decisions)

    def to_json(self) -> Dict[str, Any]:
        return {"format": DECISIONS_FORMAT, "source": self.source, "tolerance_nm": float(self.tolerance_nm),
                "decisions": [d.to_json() for d in self.decisions]}

    def fingerprint(self) -> str:
        """Short hash of the decisions (not of the path): enters the record of the analysis they were applied to."""
        if not self.decisions:
            return ""
        text = json.dumps(self.to_json()["decisions"], sort_keys=True)
        return hashlib.sha1(f"{self.tolerance_nm!r}|{text}".encode("utf-8")).hexdigest()[:10]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def decisions_path(data_path: str) -> str:
    """Where the decisions of ``data_path`` live: ``<data without extension>_unanchored.json`` beside it."""
    stem, _ext = os.path.splitext(str(data_path))
    return stem + DECISIONS_SUFFIX


def _decision_from(entry: Any, where: str) -> ManualDecision:
    if not isinstance(entry, Mapping):
        raise ManualDecisionsError(f"{where}: a decision is not an object")
    try:
        x, y = float(entry["x_nm"]), float(entry["y_nm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManualDecisionsError(f"{where}: a decision has no usable x_nm / y_nm ({exc})") from exc
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ManualDecisionsError(f"{where}: a decision's position is not finite")
    x, y = round(x, 1), round(y, 1)          # stored to 0.1 nm: a decision reads back exactly as it was made
    action = str(entry.get("action", ""))
    if action not in ACTIONS:
        raise ManualDecisionsError(f"{where}: action {action!r} is not one of {', '.join(ACTIONS)}")
    reason = str(entry.get("reason", "")).strip()
    if action == ACTION_EXCLUDE and not reason:
        raise ManualDecisionsError(f"{where}: an exclusion without a reason")
    return ManualDecision(x_nm=x, y_nm=y, action=action, reason=reason, date=str(entry.get("date", "")),
                          note=str(entry.get("note", "")))


def load_decisions(path: str) -> ManualDecisions:
    """The decisions stored at ``path``; none (and no error) when the file does not exist. A file that exists and is
    not a decisions file raises ``ManualDecisionsError`` -- it is never ignored or overwritten in silence."""
    if not os.path.exists(path):
        return ManualDecisions(path=str(path))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ManualDecisionsError(f"{path}: not readable as JSON ({exc})") from exc
    if not isinstance(data, Mapping) or data.get("format") != DECISIONS_FORMAT:
        raise ManualDecisionsError(f"{path}: not a decisions file (format must be {DECISIONS_FORMAT!r})")
    entries = data.get("decisions", [])
    if not isinstance(entries, list):
        raise ManualDecisionsError(f"{path}: 'decisions' is not a list")
    try:
        tol = float(data.get("tolerance_nm", MANUAL_TOLERANCE_NM))
    except (TypeError, ValueError) as exc:
        raise ManualDecisionsError(f"{path}: tolerance_nm is not a number") from exc
    if not (math.isfinite(tol) and tol > 0.0):
        raise ManualDecisionsError(f"{path}: tolerance_nm must be a positive number")
    decisions = tuple(_decision_from(e, f"{path} [{i}]") for i, e in enumerate(entries))
    return ManualDecisions(source=str(data.get("source", "")), tolerance_nm=tol, decisions=decisions, path=str(path))


def save_decisions(decisions: ManualDecisions, path: Optional[str] = None) -> str:
    """Write the decisions (atomically: a temporary file in the same folder, then a rename). Refuses to replace a
    file that is not a decisions file. Returns the path written."""
    target = str(path or decisions.path)
    if not target:
        raise ValueError("no path to write the decisions to")
    if os.path.exists(target):
        load_decisions(target)          # raises when the file there is something else
    folder = os.path.dirname(os.path.abspath(target))
    fd, tmp = tempfile.mkstemp(prefix=".unanchored_", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(decisions.to_json(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return target


def add_decision(decisions: ManualDecisions, x_nm: float, y_nm: float, action: str, reason: str = "", *,
                 note: str = "", date: Optional[str] = None) -> ManualDecisions:
    """The decisions with one more; an earlier decision at the same position (within the tolerance) is replaced,
    so the latest decision on a cluster is the one that counts."""
    entry = _decision_from({"x_nm": x_nm, "y_nm": y_nm, "action": action, "reason": reason, "note": note,
                            "date": date or _now()}, "new decision")
    kept = tuple(d for d in decisions.decisions
                 if math.hypot(d.x_nm - entry.x_nm, d.y_nm - entry.y_nm) > decisions.tolerance_nm)
    return replace(decisions, decisions=kept + (entry,))


def remove_decision(decisions: ManualDecisions, x_nm: float, y_nm: float) -> ManualDecisions:
    """The decisions without the ones at this position (within the tolerance): back to the automatic verdict."""
    kept = tuple(d for d in decisions.decisions
                 if math.hypot(d.x_nm - float(x_nm), d.y_nm - float(y_nm)) > decisions.tolerance_nm)
    return replace(decisions, decisions=kept)


def match_decisions(centroids: Any, decisions: ManualDecisions) -> Tuple[Dict[int, ManualDecision], List[int]]:
    """({cluster row: its decision}, [indices of decisions that match no cluster]). Each decision goes to the
    nearest centre within the tolerance; when two decisions land on one cluster the later one in the file wins."""
    c = _as_points(centroids)
    matched: Dict[int, ManualDecision] = {}
    orphans: List[int] = []
    for i, d in enumerate(decisions.decisions):
        if c.shape[0] == 0:
            orphans.append(i)
            continue
        dist = np.hypot(c[:, 0] - d.x_nm, c[:, 1] - d.y_nm)
        j = int(np.argmin(dist))
        if float(dist[j]) <= decisions.tolerance_nm:
            matched[j] = d
        else:
            orphans.append(i)
    return matched, orphans


# ----------------------------------------------------------------------------------------------------
# Final flags and preview
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FinalFlags:
    """Which clusters leave the 'discard applied' analysis, and why (automatic rule + method + decisions by hand)."""

    method: str
    result: Optional[UnanchoredResult]
    labels: NDArray[np.int64]
    centroids: NDArray[np.float64]
    base: NDArray[np.bool_]                       # excluded by the method alone (automatic or widefield)
    excluded: NDArray[np.bool_]                   # final
    source: Tuple[str, ...]                       # per cluster: automatic / widefield / manual / restored by hand / ""
    reason: Tuple[str, ...]                       # per cluster, one sentence or ""
    n_excluded_manual: int
    n_restored_manual: int
    orphans: Tuple[ManualDecision, ...]
    decisions: ManualDecisions
    n_widefield: Optional[int]
    warnings: Tuple[str, ...] = ()

    @property
    def n_excluded(self) -> int:
        return int(self.excluded.sum())

    @property
    def discard_source(self) -> str:
        """'automatic', 'widefield', 'manual', their combination joined by '+', or 'none'."""
        found = [s for s in (SOURCE_AUTOMATIC, SOURCE_WIDEFIELD, SOURCE_MANUAL)
                 if any(src == s for src, ex in zip(self.source, self.excluded) if ex)]
        return "+".join(found) if found else "none"

    def record(self) -> str:
        """One line for the analysis' record (and its id): method, ring, thresholds and decisions."""
        parts = [f"method {self.method}"]
        if self.method == METHOD_AUTOMATIC and self.result is not None:
            parts.append(f"{self.result.reference}, exclude > {self.result.exclude_nm:g} nm")
        if self.decisions.decisions:
            parts.append(f"{len(self.decisions)} decision(s) by hand {self.decisions.fingerprint()}")
        return "; ".join(parts)


def final_flags(result: Optional[UnanchoredResult], decisions: Optional[ManualDecisions] = None, *,
                method: str = METHOD_AUTOMATIC, widefield: Optional[Any] = None,
                centroids: Optional[Any] = None, labels: Optional[Sequence[int]] = None,
                data_source: str = "") -> FinalFlags:
    """
    The final discard. ``method``: "automatic" (the rule's unanchored clusters; doubtful ones are kept, D2),
    "widefield" (``widefield``: the flags of ``tools.mps_axoplasm.anchored_clusters``, D5) or "off" (nothing but the
    decisions by hand). ``decisions`` overlay any method. ``centroids``/``labels`` are needed only when ``result``
    is None. ``data_source``: the data file's basename, checked against the decisions'.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {', '.join(METHODS)}")
    if result is not None:
        c, lab = result.centroids, result.labels
    else:
        if centroids is None:
            raise ValueError("without an automatic result the cluster centres must be given")
        c = _as_points(centroids)
        lab = (np.arange(c.shape[0], dtype=np.int64) if labels is None
               else np.asarray(list(labels), dtype=np.int64).reshape(-1))
    k = c.shape[0]
    warnings: List[str] = list(result.warnings) if (result is not None and method == METHOD_AUTOMATIC) else []
    reason = [""] * k
    source = [""] * k
    n_wf: Optional[int] = None
    wf: Optional[NDArray[np.bool_]] = None
    if widefield is not None:
        wf = np.asarray(widefield, dtype=bool).reshape(-1)
        if wf.size != k:
            raise ValueError(f"{wf.size} widefield flags for {k} clusters")
        n_wf = int(wf.sum())
    if method == METHOD_AUTOMATIC:
        if result is None:
            raise ValueError("method 'automatic' needs the automatic result")
        base = result.unanchored.copy()
        for i in np.flatnonzero(base):
            source[i] = SOURCE_AUTOMATIC
            reason[i] = (f"{result.depth_nm[i]:,.0f} nm below the {result.reference} ring "
                         f"(more than {result.exclude_nm:g} nm)")
        for i in np.flatnonzero(result.doubtful):
            reason[i] = (f"doubtful: {result.depth_nm[i]:,.0f} nm below the ring "
                         f"({result.doubt_nm:g}-{result.exclude_nm:g} nm), kept")
    elif method == METHOD_WIDEFIELD:
        if wf is None:
            raise ValueError("method 'widefield' needs the widefield flags")
        base = wf.copy()
        for i in np.flatnonzero(base):
            source[i] = SOURCE_WIDEFIELD
            reason[i] = "inside both widefield images by more than the margin"
    else:
        base = np.zeros(k, dtype=bool)
    excluded = base.copy()
    dec = decisions if decisions is not None else ManualDecisions()
    if dec.source and data_source and os.path.basename(dec.source) != os.path.basename(data_source):
        warnings.append(f"the decisions by hand were made on {dec.source}, not on {data_source}")
    matched, orphan_idx = match_decisions(c, dec)
    n_ex = n_re = 0
    for row, d in matched.items():
        if d.action == ACTION_EXCLUDE:
            n_ex += int(not base[row])
            excluded[row] = True
            if not base[row]:
                source[row] = SOURCE_MANUAL
            reason[row] = f"by hand ({d.date[:10]}): {d.reason}" + (f" -- {d.note}" if d.note else "")
        else:
            if base[row]:
                n_re += 1
                source[row] = SOURCE_RESTORED
            excluded[row] = False
            reason[row] = f"restored by hand ({d.date[:10]})" + (f": {d.reason}" if d.reason else "")
    orphans = tuple(dec.decisions[i] for i in orphan_idx)
    if orphans:
        warnings.append(f"{len(orphans)} decision(s) by hand match no cluster within {dec.tolerance_nm:g} nm "
                        f"(kept in the file, not applied)")
    return FinalFlags(method=method, result=result, labels=lab,
                      centroids=c, base=np.asarray(base, dtype=bool), excluded=np.asarray(excluded, dtype=bool),
                      source=tuple(source), reason=tuple(reason), n_excluded_manual=n_ex, n_restored_manual=n_re,
                      orphans=orphans, decisions=dec, n_widefield=n_wf, warnings=tuple(warnings))


@dataclass(frozen=True, eq=False)
class Preview:
    """What the interface draws before 'Apply': nothing in the analysis changes until the flags are applied."""

    ring: Optional[NDArray[np.float64]]           # the reference ring (dashed)
    excluded_xy: NDArray[np.float64]              # red crosses
    doubtful_xy: NDArray[np.float64]              # amber rings (kept)
    restored_xy: NDArray[np.float64]              # clusters the rule excluded and a person restored
    borderline_xy: NDArray[np.float64]
    text: str                                     # "would exclude N of K; M doubtful"
    warnings: Tuple[str, ...]
    flags: FinalFlags


def preview(flags: FinalFlags) -> Preview:
    """The preview of a set of final flags (``final_flags``)."""
    c = flags.centroids
    res = flags.result
    doubtful = (res.doubtful & ~flags.excluded) if (res is not None and flags.method == METHOD_AUTOMATIC) \
        else np.zeros(c.shape[0], dtype=bool)
    border = (res.used.borderline if (res is not None and flags.method == METHOD_AUTOMATIC)
              else np.zeros(c.shape[0], dtype=bool))
    restored = np.asarray([s == SOURCE_RESTORED for s in flags.source], dtype=bool)
    ring = res.used.ring.polygon if (res is not None and flags.method == METHOD_AUTOMATIC) else None
    k = c.shape[0]
    text = f"would exclude {flags.n_excluded} of {k}"
    if flags.method == METHOD_AUTOMATIC:
        text += f"; {int(doubtful.sum())} doubtful"
        if res is not None and not res.evaluated:
            text = f"automatic discard not evaluated ({res.failure}); " + text
    if flags.n_excluded_manual or flags.n_restored_manual:
        text += f"; by hand: {flags.n_excluded_manual} excluded, {flags.n_restored_manual} restored"
    return Preview(ring=ring, excluded_xy=c[flags.excluded], doubtful_xy=c[doubtful], restored_xy=c[restored],
                   borderline_xy=c[border], text=text, warnings=flags.warnings, flags=flags)


# ----------------------------------------------------------------------------------------------------
# The analysis with and without the discard
# ----------------------------------------------------------------------------------------------------
def compare_unanchored(analysis: Any, flags: FinalFlags, **kwargs: Any) -> Any:
    """``tools.mps_analysis.compare_discard`` of ``analysis`` (all kept clusters) with the final flags: the pair
    'all clusters' / 'discard applied' (D4), the second recording where its discard came from. The flags must
    describe the analysis' own centroids, in its order."""
    from tools import mps_analysis
    cent = np.asarray(analysis.centroids, dtype=np.float64).reshape(-1, 2)
    if cent.shape != flags.centroids.shape or not np.allclose(cent, flags.centroids, atol=1e-6):
        raise ValueError("the flags were computed on other cluster centres than this analysis'")
    margin = (flags.result.exclude_nm if (flags.result is not None and flags.method == METHOD_AUTOMATIC)
              else kwargs.pop("margin_nm", DEFAULT_DOUBT_NM))
    kwargs.pop("margin_nm", None)
    return mps_analysis.compare_discard(analysis, flags.excluded, margin_nm=float(margin),
                                        source=flags.record(), **kwargs)


def auto_for_analysis(analysis: Any, **kwargs: Any) -> UnanchoredResult:
    """``auto_unanchored`` on an ``AxonAnalysis``: its slab localizations, kept centroids and their labels."""
    from tools.mps_analysis import good_cluster_labels
    labels = good_cluster_labels(analysis.labels, analysis.bad_report.bad_labels)
    cent = np.asarray(analysis.centroids, dtype=np.float64).reshape(-1, 2)
    lab = labels if np.asarray(labels).size == cent.shape[0] else None
    return auto_unanchored(analysis.x_slab, analysis.y_slab, cent, lab, **kwargs)


# ----------------------------------------------------------------------------------------------------
# Trim of interior localizations before the perimeter contour (D8, off by default)
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Trim:
    """Which localizations the contour is fitted on after the trim."""

    keep: NDArray[np.bool_]
    depth_nm: float
    n_trimmed: int
    status: str                                   # "trimmed" or "not trimmed: <reason>"


TRIM_STRICT_PREFIX: str = "not trimmed: the polar reference fails the strict rules of the automatic contour"


def strict_polar_reference(x_nm: Any, y_nm: Any) -> Tuple[bool, str]:
    """Whether the polar curve of these localizations passes the STRICT rules of the automatic contour (P3 v2), the
    condition for the interior trim (decision D8, user's decision 2026-11-17). The polar curve is the P-spline fitted
    on the polar contour (``mps_auto_contour.contour_details(method="polar", fallbacks=False)``) and it must come out
    with no rejection reason: no self-crossing, length <= 1.5 x its convex hull, no inward bend tighter than 150 nm,
    the band never more than 75 nm to one side of it along a stretch, and a lumen. Returns (valid, reason).

    Why the strict rules and not only "the cloud is star-shaped" (``reference_ring``'s own check): in the contour
    validation (report P3 v2, sec. 5) a trim against ANY polar curve that exists removed 5-22 % of the true membrane
    on clouds that are not star-shaped ("C"), and the curve moved up to 1.2 um; against a polar curve that passes
    these rules it removed <= 0.5 % (p95). On the pilot data the weak rule trims 16 of 18 axons, the strict one 4."""
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    det = ac.contour_details(x[finite], y[finite], method="polar", fallbacks=False)
    cand = det.candidates[0]
    if cand.path is None:
        return False, f"no polar curve ({cand.reason or 'not fitted'})"
    if cand.reason:
        return False, f"{cand.reason}: {cand.detail}" if cand.detail else cand.reason
    return True, ""


def trim_interior(x_nm: Any, y_nm: Any, *, depth_nm: float = TRIM_DEPTH_NM, require_strict: bool = True) -> Trim:
    """Localizations deeper than ``depth_nm`` below the polar ring of these same localizations are left out. When the
    polar ring cannot be used (too few localizations, no polar contour, a folded membrane) nothing is trimmed and the
    status says why: the trim never removes membrane on a shape the ring cannot represent.

    ``require_strict`` (default True, user's decision 2026-11-17): the trim is applied only when the polar curve also
    passes the strict rules of the automatic contour (``strict_polar_reference``); otherwise nothing is trimmed and
    the status starts with ``TRIM_STRICT_PREFIX``. False keeps only the weak rule (the cloud is star-shaped), which
    on non-star-shaped clouds removes 5-22 % of the membrane in simulation: for validation only."""
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    keep = np.ones(x.size, dtype=bool)
    if not (math.isfinite(depth_nm) and depth_nm > 0.0):
        raise ValueError("the trim depth must be a positive number")
    finite = np.isfinite(x) & np.isfinite(y)
    if require_strict:
        valid, why = strict_polar_reference(x[finite], y[finite])
        if not valid:
            return Trim(keep=keep, depth_nm=float(depth_nm), n_trimmed=0, status=f"{TRIM_STRICT_PREFIX} ({why})")
    ring = reference_ring(REFERENCE_POLAR, x[finite], y[finite], np.zeros((0, 2)), exclude_nm=float(depth_nm))
    if not ring.ok or ring.polygon is None:
        return Trim(keep=keep, depth_nm=float(depth_nm), n_trimmed=0, status=f"not trimmed: {ring.status}")
    pts = np.column_stack([x, y])
    deep = np.zeros(x.size, dtype=bool)
    deep[finite] = signed_depth(pts[finite], ring.polygon) > float(depth_nm)
    keep = ~deep
    return Trim(keep=keep, depth_nm=float(depth_nm), n_trimmed=int(deep.sum()), status="trimmed")


# ----------------------------------------------------------------------------------------------------
# Export (axon tables v18)
# ----------------------------------------------------------------------------------------------------
def _labels_text(labels: NDArray[np.int64], mask: NDArray[np.bool_]) -> Optional[str]:
    picked = [str(int(v)) for v in labels[np.asarray(mask, dtype=bool)]]
    return " ".join(picked) if picked else None


def _finite_or_none(v: Optional[float], digits: int = 1) -> Optional[float]:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), digits)


def unanchored_block(flags: Optional[FinalFlags], *, perimeter: Any = None) -> Dict[str, Any]:
    """The UNANCHORED_COLUMNS of one axon. ``flags`` None: not run. ``perimeter``: the perimeter density result
    (``PerimeterResult``), only read for the trim columns (its ``extras``)."""
    block = not_run_block()
    if perimeter is not None:
        extras = getattr(perimeter, "extras", None) or {}
        block["pd_contour_trim_nm"] = extras.get("contour_trim_nm")
        block["pd_contour_n_locs_trimmed"] = extras.get("contour_n_locs_trimmed")
    if flags is None:
        return block
    res = flags.result
    k = int(flags.centroids.shape[0])
    block.update({
        "unanchored_method": flags.method,
        "unanchored_status": STATUS_OK,
        "unanchored_failure": None,
        "unanchored_n_clusters": k,
        "unanchored_n_widefield": flags.n_widefield,
        "unanchored_n_excluded_manual": flags.n_excluded_manual,
        "unanchored_n_restored_manual": flags.n_restored_manual,
        "unanchored_n_final": flags.n_excluded,
        "unanchored_fraction": round(flags.n_excluded / k, 4) if k else None,
        "unanchored_discard_source": flags.discard_source,
        "unanchored_labels_excluded_manual": _labels_text(
            flags.labels, np.asarray([s == SOURCE_MANUAL for s in flags.source], dtype=bool)),
        "unanchored_labels_restored_manual": _labels_text(
            flags.labels, np.asarray([s == SOURCE_RESTORED for s in flags.source], dtype=bool)),
        "unanchored_manual_file": os.path.basename(flags.decisions.path) if flags.decisions.path else None,
        "unanchored_manual_fingerprint": flags.decisions.fingerprint() or None,
        "unanchored_manual_n_orphan": len(flags.orphans),
        "unanchored_warnings": " | ".join(flags.warnings) or None,
    })
    manual_rows = [i for i, s in enumerate(flags.source) if s in (SOURCE_MANUAL, SOURCE_RESTORED)]
    if manual_rows:
        block["unanchored_manual_reasons"] = " | ".join(
            f"{int(flags.labels[i])}: {flags.reason[i]}" for i in manual_rows)
    if res is not None:
        used = res.used
        block.update({
            "unanchored_reference": res.reference,
            "unanchored_exclude_nm": res.exclude_nm,
            "unanchored_doubt_nm": res.doubt_nm,
            "unanchored_reference_noise_nm": _finite_or_none(used.noise_nm),
            "unanchored_inner_band_arc_nm": _finite_or_none(used.ring.inner_band_arc_nm),
            "unanchored_n_auto": used.n_unanchored if res.evaluated else None,
            "unanchored_n_doubtful": used.n_doubtful if res.evaluated else None,
            "unanchored_n_borderline": int(used.borderline.sum()) if res.evaluated else None,
            "unanchored_max_depth_nm": (_finite_or_none(float(np.nanmax(used.depth_nm)))
                                        if res.evaluated and k else None),
            "unanchored_labels_auto": _labels_text(res.labels, res.unanchored) if res.evaluated else None,
            "unanchored_labels_doubtful": _labels_text(res.labels, res.doubtful) if res.evaluated else None,
            "unanchored_gate_exceeded": (bool(k and used.n_unanchored / k > MAX_FLAGGED_FRACTION)
                                         if res.evaluated else None),
            "unanchored_n_changed_by_reference": res.n_changed_by_reference,
        })
        if not res.evaluated:
            block["unanchored_status"] = res.status
            block["unanchored_failure"] = res.failure
        if res.other is not None:
            block.update({
                "unanchored_other_reference": res.other.ring.kind,
                "unanchored_other_status": res.other.ring.status,
                "unanchored_n_auto_other": res.other.n_unanchored if res.other.ring.ok else None,
            })
    return {name: block.get(name) for name in UNANCHORED_COLUMNS}


def cluster_rows(flags: Optional[FinalFlags]) -> Dict[int, Dict[str, Any]]:
    """{cluster label: CLUSTER_UNANCHORED_COLUMNS} for the per-cluster table (empty dict: not run)."""
    if flags is None:
        return {}
    res = flags.result
    out: Dict[int, Dict[str, Any]] = {}
    for i, label in enumerate(flags.labels):
        row = empty_cluster_row()
        if res is not None:
            row["depth_below_ring_nm"] = _finite_or_none(float(res.depth_nm[i]))
            row["anchor_state"] = str(res.state[i])
            row["anchor_borderline"] = bool(res.used.borderline[i]) if res.evaluated else None
        row["anchor_source"] = flags.source[i] or None
        row["anchor_reason"] = flags.reason[i] or None
        out[int(label)] = {name: row.get(name) for name in CLUSTER_UNANCHORED_COLUMNS}
    return out


__all__ = [
    "DEFAULT_EXCLUDE_NM", "DEFAULT_DOUBT_NM", "REFERENCE_POLAR", "REFERENCE_SMOOTH", "REFERENCES", "DEFAULT_REFERENCE",
    "STATES", "STATE_ANCHORED", "STATE_DOUBTFUL", "STATE_UNANCHORED", "STATE_NOT_EVALUATED", "METHODS",
    "ManualDecisionsError", "ReferenceRing", "RingVerdict", "UnanchoredResult", "ManualDecision", "ManualDecisions",
    "FinalFlags", "Preview", "Trim", "points_in_polygon", "distance_to_polygon", "signed_depth",
    "smooth_ring_polygon", "inner_band_arc_nm", "reference_ring", "depth_noise", "classify", "auto_unanchored",
    "auto_for_analysis", "decisions_path", "load_decisions", "save_decisions", "add_decision", "remove_decision",
    "match_decisions", "final_flags", "preview", "compare_unanchored", "trim_interior", "strict_polar_reference",
    "TRIM_STRICT_PREFIX", "unanchored_block",
    "cluster_rows", "UNANCHORED_COLUMNS", "CLUSTER_UNANCHORED_COLUMNS",
]
