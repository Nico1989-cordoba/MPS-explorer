# -*- coding: utf-8 -*-
"""
Non-assignative spatial statistics for point patterns: Ripley's K / L / H in
2D (polygonal window) and in 1D (closed or open arc), pair-correlation g(r),
and Monte Carlo envelopes under complete spatial randomness (CSR).

Pure functions, no interface. Written for PR A9 ("per-axon null and working
scale": H1 on the perimeter arc and H2 in the slab, computed on the curated
cluster centres) and for the multi-algorithm clustering benchmark. Every
formula carries its source; every default says where it comes from:
[literature] (citation), [derivation], [simulation] (project) or [ad hoc].
References checked against CrossRef on 2026-10-16 (CITATIONS_D0); a passage
that could not be checked in the text itself says so.

Estimators
----------
2D, window W of area A (shoelace formula), n points, ordered pairs i != j:

    K(r) = A / (n (n - 1)) * sum_{i != j} w_ij 1[d_ij <= r]      (Ripley, 1977)
    L(r) = sqrt(K(r) / pi);  H(r) = L(r) - r     (L: Besag, 1977, in the discussion of
                                                  Ripley, 1977; H: Kiskowski et al., 2009)

  with A / (n (n - 1)) the unbiased replacement of 1 / (lambda^2 A), lambda =
  n / A, and w_ij an edge-correction weight:
    'none'        w_ij = 1 (biased downward near the boundary; the Monte
                  Carlo test in the SAME window cancels that bias, which is
                  A9's default for the thin +/-50 nm band: Besag & Diggle,
                  1977; Baddeley, Rubak & Turner, 2015, book; the exact
                  passage was not checked).
    'ripley'      w_ij = 1 / f_ij, f_ij the fraction of the circumference of
                  the circle of radius d_ij centred on x_i that lies inside W
                  (Ripley, 1977, isotropic correction). Exact (weight 1) when
                  the circle does not reach the boundary; otherwise f_ij is
                  measured by sampling ``n_angles`` equally spaced directions
                  on the circle and testing them against the polygon
                  (even-odd rule), an approximation with absolute error of
                  order 1 / n_angles in f. Weights are capped at
                  ``max_weight`` (default 10 [ad hoc], reported in the
                  result; spatstat -- Baddeley & Turner, 2005 -- caps nothing;
                  keeping r below a quarter of the window's shorter side is
                  a rule of thumb whose source was not verified).
    'translation' w_ij = A / |W intersect (W + (x_j - x_i))| for an
                  axis-aligned rectangle a x b:  A / ((a - |dx|) (b - |dy|))
                  (Ohser, 1983, reference verified, formula not checked in
                  the text; Baddeley, Rubak & Turner, 2015). Exact, cheap,
                  rectangles only.

1D, closed curve of length L (arc coordinate s in [0, L)), cyclic distance
d(i, j) = min(|s_i - s_j|, L - |s_i - s_j|):

    K1(r) = L / (n (n - 1)) * sum_{i != j} w_ij 1[d(i, j) <= r]
    under CSR on the closed curve  E[K1(r)] = 2 r  (r <= L / 2)
    H1(r) = K1(r) - 2 r                               ([derivation]: E[K1] = 2 r)

  On a closed curve there is no boundary and w_ij = 1. On an open arc
  (``closed=False``) the distance is |s_i - s_j| and the 1D analogue of
  Ripley's isotropic correction is used: w_ij = 2 / #{s_i - d, s_i + d
  inside [0, L]}, the inverse of the fraction of the 1D "circle" of radius
  d (the two points s_i +/- d) that falls inside the arc; unbiased under
  CSR by the same argument as in 2D ([derivation]: Ripley, 1977, applied in
  one dimension; no 1D reference was checked).

Pair correlation (Sengupta et al., 2011; Veatch et al., 2012), annular bins
[r_k, r_{k+1}):

    g(r_k) = A / (n (n - 1)) * sum_{i != j} w_ij 1[r_k <= d_ij < r_{k+1}]
             / (pi (r_{k+1}^2 - r_k^2))                      -> 1 under CSR
    g1(r_k) = L / (n (n - 1)) * sum_{i != j} w_ij 1[bin] / (2 (r_{k+1} - r_k))

Monte Carlo under CSR (``csr_envelopes``): B simulations with the same n in
the same window or on the same arc, the statistic on each, and
  * pointwise envelope: the alpha/2 and 1 - alpha/2 quantiles at each r;
  * global envelope and p by maximum studentised deviation
        T_b = max_{r >= r_min} |H_b(r) - median_b H(r)| / sd_b H(r)
        p_global = (1 + #{T_b >= T_obs}) / (B + 1)
    (Besag & Diggle, 1977; Schneider & Schuetz, 2022; the maximum-absolute-
    deviation (MAD) test of Baddeley et al., 2014); the envelope is
    median +/- q_{1-alpha}(T) sd;
  * global envelope by extreme rank (Myllymaki et al., 2017);
  * r*, the r of the maximum studentised deviation of the observed curve,
    and h_max, the signed deviation there ([ad hoc]: the project's working
    scale, plan A9).
Everything is deterministic given ``seed``.

Defaults: n_sim = 199 [derivation: the smallest Monte Carlo p is
1 / (B + 1) = 0.005, ten times below alpha], alpha = 0.05 [ad hoc:
conventional level], n_angles = 360 [derivation: f is measured in steps of
1 / 360 ~ 0.3 % of the circle], max_weight = 10 [ad hoc].

References (APA 7)
------------------
Baddeley, A., Diggle, P. J., Hardegen, A., Lawrence, T., Milne, R. K., & Nair, G. (2014). On tests of spatial
    pattern based on simulation envelopes. Ecological Monographs, 84(3), 477-489. https://doi.org/10.1890/13-2042.1
Baddeley, A., Rubak, E., & Turner, R. (2015). Spatial point patterns: Methodology and applications with R. CRC
    Press. https://doi.org/10.1201/b19708
Baddeley, A., & Turner, R. (2005). spatstat: An R package for analyzing spatial point patterns. Journal of
    Statistical Software, 12(6). https://doi.org/10.18637/jss.v012.i06
Besag, J. (1977). Comments on Ripley's paper. Journal of the Royal Statistical Society: Series B (Methodological),
    39(2), 193-195 [pages to confirm]. https://doi.org/10.1111/j.2517-6161.1977.tb01616.x
Besag, J., & Diggle, P. J. (1977). Simple Monte Carlo tests for spatial pattern. Applied Statistics, 26(3), 327-333.
    https://doi.org/10.2307/2346974
Myllymaki, M., Mrkvicka, T., Grabarnik, P., Seijo, H., & Hahn, U. (2017). Global envelope tests for spatial
    processes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 79(2), 381-404.
    https://doi.org/10.1111/rssb.12172
Ohser, J. (1983). On estimators for the reduced second moment measure of point processes. Mathematische
    Operationsforschung und Statistik, Series Statistics, 14(1), 63-71. https://doi.org/10.1080/02331888308801687
Ripley, B. D. (1977). Modelling spatial patterns. Journal of the Royal Statistical Society: Series B
    (Methodological), 39(2), 172-192. https://doi.org/10.1111/j.2517-6161.1977.tb01615.x
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

FloatArray = NDArray[np.float64]

DEFAULT_N_SIM = 199            # [derivation] p_min = 1 / (B + 1) = 0.005
DEFAULT_ALPHA = 0.05           # [ad hoc] conventional level
DEFAULT_N_ANGLES = 360         # [derivation] f measured in steps of 1/360 (~0.3 %)
DEFAULT_MAX_WEIGHT = 10.0      # ad hoc cap on Ripley's weights (reported in the result)
_EPS = 1e-12


# ============================================================================
# Geometry utilities
# ============================================================================

def polygon_area(polygon: FloatArray) -> float:
    """Unsigned area of a simple polygon by the shoelace formula (last vertex joined to the first)."""
    p = np.asarray(polygon, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
        raise ValueError(f"polygon_area: expected a (K >= 3, 2) polygon, got shape {p.shape}")
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_signed_area(polygon: FloatArray) -> float:
    """Signed shoelace area: positive for counter-clockwise vertex order."""
    p = np.asarray(polygon, dtype=np.float64)
    return float(0.5 * (np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def _inside_polygon_local(x: FloatArray, y: FloatArray, polygon: FloatArray) -> NDArray[np.bool_]:
    """Even-odd point-in-polygon (same algorithm as ``_inside_polygon`` in tools/mps_simulate_axon.py)."""
    px = np.asarray(x, dtype=np.float64)
    py = np.asarray(y, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    inside = np.zeros(px.shape, dtype=bool)
    xj, yj = float(poly[-1, 0]), float(poly[-1, 1])
    for xi, yi in poly.tolist():
        crosses = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_edge = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= crosses & (px < x_edge)
        xj, yj = xi, yi
    return inside


def point_in_polygon(x: FloatArray, y: FloatArray, polygon: FloatArray) -> NDArray[np.bool_]:
    """True for the points (x, y) inside the closed polygon (even-odd rule; convex or concave alike).

    Reuses ``_inside_polygon`` from tools/mps_simulate_axon.py when that module imports cleanly (it has
    no import-time side effects, but pulls scipy.special and the columns/matching modules), otherwise a
    verbatim local copy of the same algorithm."""
    try:
        from tools.mps_simulate_axon import _inside_polygon
    except Exception:            # pragma: no cover - the copy above is the same algorithm
        return _inside_polygon_local(x, y, polygon)
    return _inside_polygon(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(polygon, dtype=np.float64))


def is_axis_aligned_rectangle(polygon: FloatArray, tol: float = 1e-9) -> bool:
    """True when the polygon is an axis-aligned rectangle given by its 4 corners (any order of traversal)."""
    p = np.asarray(polygon, dtype=np.float64)
    if p.shape != (4, 2):
        return False
    xs = np.unique(np.round(p[:, 0], 9))
    ys = np.unique(np.round(p[:, 1], 9))
    if xs.size != 2 or ys.size != 2:
        return False
    # consecutive vertices must share x or y
    q = np.vstack([p, p[:1]])
    for a, b in zip(q[:-1], q[1:]):
        if abs(a[0] - b[0]) > tol and abs(a[1] - b[1]) > tol:
            return False
    return True


def distance_to_boundary(xy: FloatArray, polygon: FloatArray, chunk: int = 4096) -> FloatArray:
    """Euclidean distance from each point to the nearest edge of the polygon (unsigned)."""
    pts = np.asarray(xy, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ab2 = np.maximum(np.sum(ab * ab, axis=1), _EPS)
    out = np.empty(pts.shape[0], dtype=np.float64)
    for start in range(0, pts.shape[0], chunk):
        p = pts[start:start + chunk]
        ap = p[:, None, :] - a[None, :, :]                       # (m, K, 2)
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab2[None, :], 0.0, 1.0)
        foot = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d2 = np.sum((p[:, None, :] - foot) ** 2, axis=2)
        out[start:start + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def arc_coordinates(xy: FloatArray, contour: FloatArray, chunk: int = 4096) -> Tuple[FloatArray, FloatArray, float]:
    """Project points onto a closed polyline contour.

    Returns ``(s, d_signed, length)``: the arc coordinate s in [0, length) of the nearest point of the
    contour (measured from the first vertex along the vertex order), the signed normal distance (negative
    inside the contour, positive outside, whatever the vertex orientation) and the contour length. The
    nearest point is found segment by segment (no smoothing: pass the B-spline contour from
    ``tools.mps_randomization.smooth_contour_bspline`` for a smooth arc). This is the 2D -> 1D step of
    A9 (sec. 2.2: "positions of arc of the centres projected on the contour")."""
    pts = np.asarray(xy, dtype=np.float64)
    c = np.asarray(contour, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 3:
        raise ValueError(f"arc_coordinates: expected a (K >= 3, 2) contour, got shape {c.shape}")
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    a = c
    b = np.roll(c, -1, axis=0)
    ab = b - a
    seg_len = np.sqrt(np.sum(ab * ab, axis=1))
    ab2 = np.maximum(seg_len ** 2, _EPS)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    length = float(cum[-1])
    orient = 1.0 if polygon_signed_area(c) > 0 else -1.0       # ccw: inside is to the left of each edge
    s_out = np.empty(pts.shape[0], dtype=np.float64)
    d_out = np.empty(pts.shape[0], dtype=np.float64)
    for start in range(0, pts.shape[0], chunk):
        p = pts[start:start + chunk]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / ab2[None, :], 0.0, 1.0)
        foot = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        d2 = np.sum((p[:, None, :] - foot) ** 2, axis=2)
        k = np.argmin(d2, axis=1)
        rows = np.arange(p.shape[0])
        s = cum[k] + t[rows, k] * seg_len[k]
        s_out[start:start + chunk] = np.mod(s, length) if length > 0 else s
        # sign: cross(edge direction, p - foot) > 0 means p is left of the edge
        cross = ab[k, 0] * (p[:, 1] - foot[rows, k, 1]) - ab[k, 1] * (p[:, 0] - foot[rows, k, 0])
        sign = np.where(cross * orient > 0, -1.0, 1.0)
        d_out[start:start + chunk] = sign * np.sqrt(d2[rows, k])
    return s_out, d_out, length


# ============================================================================
# CSR simulation
# ============================================================================

def simulate_csr_polygon(n: int, polygon: FloatArray, rng: np.random.Generator, max_rounds: int = 1000) -> FloatArray:
    """n points uniform in the polygon by rejection from its bounding box (deterministic given ``rng``)."""
    poly = np.asarray(polygon, dtype=np.float64)
    lo = poly.min(axis=0)
    hi = poly.max(axis=0)
    area = polygon_area(poly)
    box = float(np.prod(hi - lo))
    if area <= 0 or box <= 0:
        raise ValueError("simulate_csr_polygon: degenerate polygon")
    accept = max(area / box, 1e-3)
    out: List[FloatArray] = []
    got = 0
    for _ in range(max_rounds):
        m = int(math.ceil((n - got) / accept * 1.2)) + 16
        cand = lo + rng.random((m, 2)) * (hi - lo)
        keep = point_in_polygon(cand[:, 0], cand[:, 1], poly)
        sel = cand[keep]
        out.append(sel)
        got += sel.shape[0]
        if got >= n:
            break
    pts = np.vstack(out)[:n]
    if pts.shape[0] < n:        # pragma: no cover
        raise RuntimeError("simulate_csr_polygon: rejection sampling did not converge")
    return pts


def simulate_csr_arc(n: int, length: float, rng: np.random.Generator) -> FloatArray:
    """n arc coordinates uniform on [0, length)."""
    return rng.random(n) * float(length)


# ============================================================================
# Edge-correction weights
# ============================================================================

def _ripley_weights_2d(xy: FloatArray, i: NDArray[np.int64], d: FloatArray, polygon: FloatArray,
                       n_angles: int, max_weight: float) -> Tuple[FloatArray, int]:
    """Ripley's isotropic weights 1 / f for ordered pairs (i -> j) at distance d, circle centred on x_i.

    f = 1 exactly when d <= distance(x_i, boundary); otherwise f is the fraction of ``n_angles`` equally
    spaced points of the circle that lie inside the polygon. Returns the weights and the number of pairs
    that needed sampling."""
    w = np.ones(d.shape[0], dtype=np.float64)
    if d.shape[0] == 0:
        return w, 0
    dist_b = distance_to_boundary(xy, polygon)
    need = d > dist_b[i]
    idx = np.flatnonzero(need)
    if idx.size == 0:
        return w, 0
    theta = np.arange(n_angles, dtype=np.float64) * (2.0 * math.pi / n_angles)
    ct, st = np.cos(theta), np.sin(theta)
    chunk = max(1, int(2_000_000 // n_angles))
    for start in range(0, idx.size, chunk):
        sel = idx[start:start + chunk]
        cx = xy[i[sel], 0][:, None] + d[sel][:, None] * ct[None, :]
        cy = xy[i[sel], 1][:, None] + d[sel][:, None] * st[None, :]
        inside = point_in_polygon(cx.ravel(), cy.ravel(), polygon).reshape(sel.size, n_angles)
        frac = np.maximum(inside.mean(axis=1), 1.0 / n_angles)
        w[sel] = np.minimum(1.0 / frac, max_weight)
    return w, int(idx.size)


def _translation_weights_rect(dxy: FloatArray, polygon: FloatArray, max_weight: float) -> FloatArray:
    """Translation weights A / |W intersect (W + t)| for an axis-aligned rectangle (exact)."""
    poly = np.asarray(polygon, dtype=np.float64)
    a = float(poly[:, 0].max() - poly[:, 0].min())
    b = float(poly[:, 1].max() - poly[:, 1].min())
    overlap = np.maximum(a - np.abs(dxy[:, 0]), _EPS) * np.maximum(b - np.abs(dxy[:, 1]), _EPS)
    return np.asarray(np.minimum(a * b / overlap, max_weight), dtype=np.float64)


def _pairs_2d(xy: FloatArray, r_max: float) -> Tuple[NDArray[np.int64], NDArray[np.int64], FloatArray]:
    """All ORDERED pairs (i, j), i != j, with d_ij <= r_max, via cKDTree (O(n * neighbours))."""
    tree = cKDTree(xy)
    pr = tree.query_pairs(r_max, output_type="ndarray")
    if pr.shape[0] == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0, dtype=np.float64)
    i = np.concatenate([pr[:, 0], pr[:, 1]]).astype(np.int64)
    j = np.concatenate([pr[:, 1], pr[:, 0]]).astype(np.int64)
    d = np.sqrt(np.sum((xy[i] - xy[j]) ** 2, axis=1))
    return i, j, d


def _weights_2d(xy: FloatArray, i: NDArray[np.int64], j: NDArray[np.int64], d: FloatArray, polygon: FloatArray,
                correction: str, n_angles: int, max_weight: float) -> Tuple[FloatArray, int]:
    if correction == "none":
        return np.ones(d.shape[0], dtype=np.float64), 0
    if correction == "ripley":
        return _ripley_weights_2d(xy, i, d, polygon, n_angles, max_weight)
    if correction == "translation":
        if not is_axis_aligned_rectangle(polygon):
            raise ValueError("correction='translation' is implemented for axis-aligned rectangles only")
        return _translation_weights_rect(xy[j] - xy[i], polygon, max_weight), 0
    raise ValueError(f"unknown correction {correction!r}: use 'ripley', 'translation' or 'none'")


# ============================================================================
# Ripley's K in 2D
# ============================================================================

@dataclass
class RipleyResult:
    """K, L and H on ``r``; ``expected_k`` is the CSR expectation (pi r^2 in 2D, 2 r on a closed arc)."""
    r: FloatArray
    k: FloatArray
    h: FloatArray
    expected_k: FloatArray
    n: int
    measure: float                      # window area (2D) or arc length (1D)
    intensity: float
    correction: str
    l: Optional[FloatArray] = None      # 2D only: sqrt(K / pi)
    n_pairs: int = 0
    n_pairs_sampled: int = 0            # pairs whose Ripley weight needed angular sampling
    max_weight_hits: int = 0
    notes: List[str] = field(default_factory=list)


def ripley_k_2d(xy: FloatArray, window_polygon: FloatArray, r_grid: FloatArray, correction: str = "ripley",
                n_angles: int = DEFAULT_N_ANGLES, max_weight: float = DEFAULT_MAX_WEIGHT) -> RipleyResult:
    """Ripley's K(r), L(r) = sqrt(K / pi) and H(r) = L(r) - r for points in a polygonal window.

    K(r) = A / (n (n - 1)) sum_{i != j} w_ij 1[d_ij <= r] (Ripley, 1977); the module docstring gives the
    weights of each ``correction``. Pairs are enumerated once with cKDTree up to max(r_grid) and K is read
    on the grid from the cumulative sum of the sorted weighted distances."""
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"ripley_k_2d: expected (n, 2) points, got shape {pts.shape}")
    poly = np.asarray(window_polygon, dtype=np.float64)
    r = np.asarray(r_grid, dtype=np.float64)
    if r.ndim != 1 or r.size == 0 or np.any(r < 0) or np.any(np.diff(r) < 0):
        raise ValueError("ripley_k_2d: r_grid must be a non-decreasing 1D array of non-negative radii")
    area = polygon_area(poly)
    n = int(pts.shape[0])
    expected = math.pi * r ** 2
    notes: List[str] = []
    if n < 2 or area <= 0:
        k = np.full(r.shape, np.nan)
        return RipleyResult(r=r, k=k, h=k.copy(), expected_k=expected, n=n, measure=area,
                            intensity=(n / area if area > 0 else float("nan")), correction=correction, l=k.copy(),
                            notes=["fewer than 2 points or empty window"])
    i, j, d = _pairs_2d(pts, float(r[-1]))
    w, n_sampled = _weights_2d(pts, i, j, d, poly, correction, n_angles, max_weight)
    hits = int(np.count_nonzero(w >= max_weight - _EPS)) if correction != "none" else 0
    order = np.argsort(d, kind="stable")
    d_sorted = d[order]
    cum = np.concatenate([[0.0], np.cumsum(w[order])])
    counts = cum[np.searchsorted(d_sorted, r, side="right")]
    k = area / (n * (n - 1.0)) * counts
    l = np.sqrt(np.maximum(k, 0.0) / math.pi)
    if hits:
        notes.append(f"{hits} pair weights capped at max_weight = {max_weight:g}")
    return RipleyResult(r=r, k=k, h=l - r, expected_k=expected, n=n, measure=area, intensity=n / area,
                        correction=correction, l=l, n_pairs=int(d.size), n_pairs_sampled=n_sampled,
                        max_weight_hits=hits, notes=notes)


def ripley_k_2d_area(xy: FloatArray, area_nm2: float, r_grid: FloatArray) -> RipleyResult:
    """Uncorrected Ripley's K(r), L and H = L - r for points in a window known only by its AREA.

    For windows that are not simple polygons -- the +/-50 nm band around an axon's contour, an annulus -- with
    correction 'none' (w_ij = 1), which is all the A9 band test uses: K is then A / (n (n - 1)) times the count of
    ordered pairs at d <= r and the window enters through A alone. Biased low near the boundary like any
    uncorrected K; a Monte Carlo test in the same window cancels it (module docstring, 'none')."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(r_grid, dtype=np.float64)
    if r.ndim != 1 or r.size == 0 or np.any(r < 0) or np.any(np.diff(r) < 0):
        raise ValueError("ripley_k_2d_area: r_grid must be a non-decreasing 1D array of non-negative radii")
    area = float(area_nm2)
    n = int(pts.shape[0])
    expected = math.pi * r ** 2
    if n < 2 or not area > 0:
        k = np.full(r.shape, np.nan)
        return RipleyResult(r=r, k=k, h=k.copy(), expected_k=expected, n=n, measure=area,
                            intensity=(n / area if area > 0 else float("nan")), correction="none", l=k.copy(),
                            notes=["fewer than 2 points or empty window"])
    _i, _j, d = _pairs_2d(pts, float(r[-1]))
    d_sorted = np.sort(d)
    counts = np.searchsorted(d_sorted, r, side="right").astype(np.float64)
    k = area / (n * (n - 1.0)) * counts
    l = np.sqrt(np.maximum(k, 0.0) / math.pi)
    return RipleyResult(r=r, k=k, h=l - r, expected_k=expected, n=n, measure=area, intensity=n / area,
                        correction="none", l=l, n_pairs=int(d.size))


# ============================================================================
# Ripley's K in 1D on an arc
# ============================================================================

def _pair_bins_1d(s: FloatArray, length: float, closed: bool, edges: FloatArray, chunk: int = 2048) -> FloatArray:
    """Weighted count of ordered pairs i != j with distance in each cell of ``edges`` (np.histogram
    convention, last cell closed). O(n^2) in row chunks; weights are 1 on a closed curve, the 1D Ripley
    weight 2 / #{s_i - d, s_i + d inside [0, L]} on an open arc."""
    n = s.shape[0]
    tot = np.zeros(edges.size - 1, dtype=np.float64)
    for start in range(0, n, chunk):
        si = s[start:start + chunk]
        diff = np.abs(si[:, None] - s[None, :])
        if closed:
            diff = np.minimum(diff, length - diff)
        rows = np.arange(si.shape[0])
        diff[rows, rows + start] = np.inf            # exclude i == j
        if closed:
            wts = np.ones_like(diff)
        else:
            # the 1D "circle" of radius d about s_i is the pair {s_i - d, s_i + d}: weight 2 / (how many
            # of the two fall inside [0, L]); unbiased under CSR (same argument as Ripley, 1977, in 1D)
            n_in = (si[:, None] - diff >= 0.0).astype(np.float64) + (si[:, None] + diff <= length).astype(np.float64)
            wts = 2.0 / np.maximum(n_in, 1.0)
        finite = np.isfinite(diff)
        h, _ = np.histogram(diff[finite], bins=edges, weights=wts[finite])
        tot += h
    return tot


def ripley_k_1d_arc(s: FloatArray, length_nm: float, r_grid: FloatArray, closed: bool = True) -> RipleyResult:
    """Ripley's K1(r) = L / (n (n - 1)) sum_{i != j} w_ij 1[d(i, j) <= r] on an arc and H1(r) = K1(r) - 2 r.

    ``closed=True``: cyclic distance, no edge effect, E[K1] = 2 r under CSR for r <= L / 2 (A9 spec, sec. 2.2).
    ``closed=False``: linear distance on [0, L] with the 1D Ripley weight (module docstring), so that
    E[K1] = 2 r still holds under CSR up to the cap of the weight (none applied in 1D: the maximum weight is
    2 at an end point)."""
    sv: FloatArray = np.asarray(s, dtype=np.float64).ravel()
    r = np.asarray(r_grid, dtype=np.float64)
    length = float(length_nm)
    if r.ndim != 1 or r.size == 0 or np.any(r < 0) or np.any(np.diff(r) < 0):
        raise ValueError("ripley_k_1d_arc: r_grid must be a non-decreasing 1D array of non-negative radii")
    if length <= 0:
        raise ValueError("ripley_k_1d_arc: length_nm must be positive")
    n = int(sv.size)
    expected = 2.0 * r
    notes: List[str] = []
    if closed and r[-1] > length / 2:
        notes.append("r beyond L / 2 on a closed curve: K1 saturates, 2 r is no longer the CSR expectation")
    if n < 2:
        k = np.full(r.shape, np.nan)
        return RipleyResult(r=r, k=k, h=k.copy(), expected_k=expected, n=n, measure=length, intensity=n / length,
                            correction="none" if closed else "ripley1d", notes=["fewer than 2 points"])
    if closed:
        sv = np.mod(sv, length)
    # cumulative count: cells (-inf, r_0], (r_0, r_1], ... via edges [0, r_0, r_1, ...] with a tiny shift so
    # that d == r counts in 1[d <= r]
    edges = np.concatenate([[-np.inf], r + 1e-9 * max(1.0, length)])
    counts = np.cumsum(_pair_bins_1d(sv, length, closed, edges))
    k = length / (n * (n - 1.0)) * counts
    return RipleyResult(r=r, k=k, h=k - expected, expected_k=expected, n=n, measure=length, intensity=n / length,
                        correction="none" if closed else "ripley1d", n_pairs=n * (n - 1), notes=notes)


# ============================================================================
# Pair correlation
# ============================================================================

@dataclass
class PairCorrelationResult:
    r_edges: FloatArray
    r_mid: FloatArray
    g: FloatArray
    n: int
    measure: float
    correction: str
    counts: FloatArray                  # weighted pair counts per bin


def pair_correlation_2d(xy: FloatArray, window_polygon: FloatArray, r_edges: FloatArray, correction: str = "ripley",
                        n_angles: int = DEFAULT_N_ANGLES, max_weight: float = DEFAULT_MAX_WEIGHT) -> PairCorrelationResult:
    """g(r) by the histogram of pair distances in annular bins, edge-corrected and normalised by the
    unbiased intensity squared and the annulus area (Sengupta et al., 2011; Veatch et al., 2012):
    g_k = A / (n (n - 1)) sum_{i != j} w_ij 1[r_k <= d_ij < r_{k+1}] / (pi (r_{k+1}^2 - r_k^2)). CSR -> 1."""
    pts = np.asarray(xy, dtype=np.float64)
    poly = np.asarray(window_polygon, dtype=np.float64)
    edges = np.asarray(r_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValueError("pair_correlation_2d: r_edges must be strictly increasing and non-negative")
    area = polygon_area(poly)
    n = int(pts.shape[0])
    mid = 0.5 * (edges[:-1] + edges[1:])
    ring = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    if n < 2 or area <= 0:
        nan = np.full(mid.shape, np.nan)
        return PairCorrelationResult(edges, mid, nan, n, area, correction, nan.copy())
    i, j, d = _pairs_2d(pts, float(edges[-1]))
    w, _ = _weights_2d(pts, i, j, d, poly, correction, n_angles, max_weight)
    counts, _ = np.histogram(d, bins=edges, weights=w)
    g = area / (n * (n - 1.0)) * counts / ring
    return PairCorrelationResult(edges, mid, g, n, area, correction, counts)


def pair_correlation_1d_arc(s: FloatArray, length_nm: float, r_edges: FloatArray, closed: bool = True) -> PairCorrelationResult:
    """g1(r) on an arc: g_k = L / (n (n - 1)) sum_{i != j} w_ij 1[r_k <= d < r_{k+1}] / (2 (r_{k+1} - r_k)).
    The 1D "annulus" of radii [r_k, r_{k+1}) has length 2 (r_{k+1} - r_k) (both sides). CSR -> 1."""
    sv: FloatArray = np.asarray(s, dtype=np.float64).ravel()
    edges = np.asarray(r_edges, dtype=np.float64)
    length = float(length_nm)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValueError("pair_correlation_1d_arc: r_edges must be strictly increasing and non-negative")
    n = int(sv.size)
    mid = 0.5 * (edges[:-1] + edges[1:])
    if n < 2 or length <= 0:
        nan = np.full(mid.shape, np.nan)
        return PairCorrelationResult(edges, mid, nan, n, length, "none", nan.copy())
    if closed:
        sv = np.mod(sv, length)
    counts = _pair_bins_1d(sv, length, closed, edges)
    g = length / (n * (n - 1.0)) * counts / (2.0 * np.diff(edges))
    return PairCorrelationResult(edges, mid, g, n, length, "none" if closed else "ripley1d", counts)


# ============================================================================
# Monte Carlo envelopes under CSR
# ============================================================================

@dataclass
class CsrEnvelope:
    """Pointwise and global envelopes of a curve statistic under CSR, and the global Monte Carlo test."""
    r: FloatArray
    sims: FloatArray                    # (n_sim, len(r)) statistic of each CSR replicate
    median: FloatArray
    sd: FloatArray
    pointwise_lo: FloatArray
    pointwise_hi: FloatArray
    global_lo: FloatArray               # median - t_crit * sd (maximum studentised deviation)
    global_hi: FloatArray
    t_crit: float                       # (1 - alpha) quantile of T_b over the replicates
    t_sims: FloatArray
    rank_lo: FloatArray                 # extreme-rank global envelope (Myllymaki et al., 2017)
    rank_hi: FloatArray
    alpha: float
    n_sim: int
    seed: int
    r_min: float
    observed: Optional[FloatArray] = None
    t_obs: Optional[float] = None
    p_global: Optional[float] = None    # (1 + #{T_b >= T_obs}) / (B + 1)
    p_rank: Optional[float] = None      # extreme-rank p (Myllymaki et al., 2017), ties counted conservatively
    r_star: Optional[float] = None      # r of the maximum studentised deviation of the observed curve
    h_max: Optional[float] = None       # observed - median at r_star (signed: + aggregation, - inhibition)
    notes: List[str] = field(default_factory=list)


def _extreme_ranks(curves: FloatArray) -> NDArray[np.int64]:
    """Extreme rank of each curve among all (rows): the smallest, over r, of its rank from below and
    from above (ties share the lowest rank). Myllymaki et al. (2017)."""
    m = curves.shape[0]
    ranks = np.full(m, m, dtype=np.int64)
    for col in range(curves.shape[1]):
        v = curves[:, col]
        if not np.all(np.isfinite(v)):
            continue
        order = np.argsort(v, kind="stable")
        sv = v[order]
        lo = np.searchsorted(sv, v, side="left") + 1           # rank from below, ties -> lowest
        hi = m - np.searchsorted(sv, v, side="right") + 1      # rank from above
        ranks = np.minimum(ranks, np.minimum(lo, hi))
    return ranks


def csr_envelopes(stat_fn: Callable[[FloatArray], FloatArray], n: int, window: Union[FloatArray, float],
                  r_grid: FloatArray, n_sim: int = DEFAULT_N_SIM, seed: int = 0, alpha: float = DEFAULT_ALPHA,
                  observed: Optional[FloatArray] = None, r_min: float = 0.0) -> CsrEnvelope:
    """Simulate CSR ``n_sim`` times with ``n`` points (uniform in the polygon ``window`` when it is an
    (K, 2) array, uniform on [0, window) when it is a length), evaluate ``stat_fn(points) -> curve on r_grid``
    on each, and return the pointwise envelope (alpha / 2, 1 - alpha / 2 quantiles), the global envelope by
    maximum studentised deviation over r >= r_min (median +/- q_{1 - alpha}(T) sd), the extreme-rank global
    envelope (to verify) and, when ``observed`` is given, T_obs, p_global = (1 + #{T_b >= T_obs}) / (B + 1)
    (Besag & Diggle, 1977; Schneider & Schuetz, 2022), r* and h_max (A9 spec, sec. 2.5). When ``observed``
    is given, the median and sd used to studentise are those of the B + 1 curves (observed included), so
    that T_obs and the T_b are exchangeable under the null; the pointwise quantiles use the replicates only.

    ``r_min`` is A9's d_fusion: radii below it are drawn but excluded from the test. Deterministic: the
    replicates come from numpy's default_rng(seed) in a fixed order."""
    r = np.asarray(r_grid, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    is_poly = isinstance(window, np.ndarray) or (not np.isscalar(window) and np.asarray(window).ndim == 2)
    sims = np.empty((int(n_sim), r.size), dtype=np.float64)
    for b in range(int(n_sim)):
        pts = simulate_csr_polygon(n, np.asarray(window, dtype=np.float64), rng) if is_poly else \
            simulate_csr_arc(n, float(window), rng)
        sims[b] = np.asarray(stat_fn(pts), dtype=np.float64)
    return envelope_from_curves(sims, r, observed=observed, alpha=alpha, r_min=r_min, seed=seed)


def envelope_from_curves(sims: FloatArray, r_grid: FloatArray, observed: Optional[FloatArray] = None,
                         alpha: float = DEFAULT_ALPHA, r_min: float = 0.0, seed: int = 0) -> CsrEnvelope:
    """The envelopes and the global test of ``csr_envelopes`` for curves simulated by ANY null.

    ``sims`` is (B, len(r_grid)): one curve per replicate of the null (CSR here, or the per-axon pipeline null of
    ``tools.mps_pipeline_null``, whose replicates are whole simulated axons run through the clustering). Same
    pointwise envelope, maximum-studentised-deviation global envelope and p_global (Besag & Diggle, 1977; the MAD
    test of Baddeley et al., 2014), extreme-rank envelope (Myllymaki et al., 2017), r* and h_max as
    ``csr_envelopes``; ``seed`` is only recorded. A replicate curve with NaN at some r (too few points) is kept: the
    nan-aware medians and quantiles skip it there."""
    r = np.asarray(r_grid, dtype=np.float64)
    sims = np.asarray(sims, dtype=np.float64).reshape(-1, r.size)
    n_sim = int(sims.shape[0])
    # the studentisation pools the observed curve with the replicates when it is given, so that T_obs and
    # every T_b are computed in exactly the same way and the B + 1 values are exchangeable under the null
    # (the Monte Carlo rank p is then exact: Besag & Diggle, 1977)
    pool = sims if observed is None else np.vstack([sims, np.asarray(observed, dtype=np.float64)[None, :]])
    med = np.nanmedian(pool, axis=0)
    sd = np.nanstd(pool, axis=0, ddof=1) if pool.shape[0] > 1 else np.full(r.size, np.nan)
    sd_floor = np.where(sd > 0, sd, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        dev_sims = np.abs(sims - med[None, :]) / sd_floor[None, :]
    test = r >= float(r_min)
    notes: List[str] = []
    if not np.any(test):
        notes.append("no radius at or above r_min: the global test is empty")
    dev_sims[:, ~test] = np.nan
    t_sims = np.nanmax(dev_sims, axis=1) if np.any(test) else np.full(n_sim, np.nan)
    t_crit = float(np.nanquantile(t_sims, 1.0 - alpha)) if np.any(np.isfinite(t_sims)) else float("nan")
    p_lo = np.nanquantile(sims, alpha / 2.0, axis=0)
    p_hi = np.nanquantile(sims, 1.0 - alpha / 2.0, axis=0)
    env = CsrEnvelope(r=r, sims=sims, median=med, sd=sd, pointwise_lo=p_lo, pointwise_hi=p_hi,
                      global_lo=med - t_crit * sd, global_hi=med + t_crit * sd, t_crit=t_crit, t_sims=t_sims,
                      rank_lo=p_lo.copy(), rank_hi=p_hi.copy(), alpha=float(alpha), n_sim=int(n_sim),
                      seed=int(seed), r_min=float(r_min), notes=notes)
    # extreme-rank envelope over the replicates alone (observed added below when given)
    all_curves = sims if observed is None else np.vstack([sims, np.asarray(observed, dtype=np.float64)[None, :]])
    sub = all_curves[:, test] if np.any(test) else all_curves
    ranks = _extreme_ranks(sub)
    m = all_curves.shape[0]
    # k_alpha: the largest k with #{R <= k} / m <= alpha; the envelope is the hull of the curves with R > k
    k_alpha = 0
    for k in range(1, m + 1):
        if np.count_nonzero(ranks <= k) / m <= alpha:
            k_alpha = k
        else:
            break
    keep = ranks > k_alpha
    if np.any(keep):
        env.rank_lo = np.min(all_curves[keep], axis=0)
        env.rank_hi = np.max(all_curves[keep], axis=0)
    if observed is not None:
        obs = np.asarray(observed, dtype=np.float64)
        env.observed = obs
        with np.errstate(invalid="ignore", divide="ignore"):
            dev_obs = np.abs(obs - med) / sd_floor
        dev_obs[~test] = np.nan
        if np.any(np.isfinite(dev_obs)):
            k_star = int(np.nanargmax(dev_obs))
            env.t_obs = float(dev_obs[k_star])
            env.r_star = float(r[k_star])
            env.h_max = float(obs[k_star] - med[k_star])
            tol = 1e-12 * max(1.0, abs(env.t_obs))
            env.p_global = float((1.0 + np.count_nonzero(t_sims >= env.t_obs - tol)) / (n_sim + 1.0))
            r_obs = int(ranks[-1])
            env.p_rank = float(np.count_nonzero(ranks <= r_obs) / m)
        else:
            notes.append("observed curve has no finite studentised deviation on the test range")
    return env
