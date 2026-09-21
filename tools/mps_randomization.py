# -*- coding: utf-8 -*-
"""
Randomization control for the spatial arrangement of betaII-spectrin clusters.

Implements parameter 8 of Gazal et al. (2026):

  "the axonal contour was first smoothed using B-spline interpolation, and
   offset curves positioned +/-50 nm from this contour delineated the inner
   (red) and outer (green) boundaries of the region. A dense grid of
   candidate positions was generated, retaining only points between the
   boundaries. For each axon, cluster centers were randomly assigned within
   this region under a minimum inter-cluster distance constraint equal to
   the minimum first nearest-neighbor (1NN) distance measured
   experimentally for that axon. Then, the experimentally detected clusters
   were reassigned to one of the randomly generated centers via a
   rigid-body translation, thus preserving the internal cluster structure
   of each cluster. The randomization procedure was repeated 1000 times per
   axon. In each iteration, the 1NN distances between cluster centers were
   recalculated and stored."

Reference result: Kolmogorov-Smirnov D = 0.054, p < 1e-3, with the
experimental and randomized CDFs crossing near 0.8 -- more short distances
than chance below that, fewer long ones above it.

Two deliberate implementation choices, both documented for the report to
the authors (the paper does not specify either):

* The annulus is defined implicitly as "within 50 nm of the smoothed
  contour" rather than by constructing two explicit offset curves. The
  region is identical, but explicit offsets self-intersect on concave
  contours -- and sciatic-nerve axons are markedly concave -- which would
  silently corrupt the candidate region.
* Placement is sequential rejection sampling with a spatial hash. The paper
  states the minimum-distance constraint but not how to satisfy it, nor
  what to do when it cannot be satisfied; here that is counted and
  reported rather than hidden.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp

# Paper constants
PAPER_KS_D = 0.054
PAPER_CDF_CROSSING = 0.8
DEFAULT_ANNULUS_HALF_WIDTH_NM = 50.0     # +/-50 nm -> 100 nm wide
DEFAULT_N_RANDOMIZATIONS = 1000
DEFAULT_GRID_SPACING_NM = 5.0
DEFAULT_SPLINE_SAMPLES = 2000


@dataclass
class RandomizationResult:
    """Outcome of the randomization control for one axon."""

    experimental_1nn_nm: NDArray[np.float64]
    randomized_1nn_nm: NDArray[np.float64]      # pooled over all iterations
    per_iteration_medians_nm: NDArray[np.float64]

    ks_statistic: float
    ks_pvalue: float
    cdf_crossing: Optional[float]               # CDF level where curves cross

    n_iterations: int
    n_clusters: int
    min_distance_nm: float                      # the constraint actually used
    n_candidates: int
    annulus_half_width_nm: float

    n_incomplete_iterations: int                # could not place every cluster
    mean_placed_fraction: float

    smoothed_contour: NDArray[np.float64]
    candidate_points: NDArray[np.float64]
    example_random_centers: Optional[NDArray[np.float64]] = None

    warnings: List[str] = field(default_factory=list)

    @property
    def experimental_median_nm(self) -> Optional[float]:
        d = self.experimental_1nn_nm
        return float(np.median(d)) if d.size else None

    @property
    def randomized_median_nm(self) -> Optional[float]:
        d = self.randomized_1nn_nm
        return float(np.median(d)) if d.size else None


def smooth_contour_bspline(
    contour: NDArray[np.float64],
    n_samples: int = DEFAULT_SPLINE_SAMPLES,
    smoothing: Optional[float] = None,
) -> NDArray[np.float64]:
    """
    Periodic B-spline smoothing of a closed contour.

    Parameters
    ----------
    contour : (K, 2) vertices in contour order (open; treated as closed).
    n_samples : points to evaluate along the smoothed curve.
    smoothing : scipy's ``s``. None selects ``s = K``, scipy's own
        documented rule of thumb for K points with unit weights -- the
        paper does not state a smoothing factor, so a principled default
        is used rather than an arbitrary constant.

    Returns
    -------
    (n_samples, 2) smoothed contour.

    Notes
    -----
    Falls back to the raw contour if the spline fit fails (too few points,
    duplicate vertices), so a difficult axon degrades to "no smoothing"
    instead of raising.
    """
    contour = np.asarray(contour, dtype=float)
    k_pts = len(contour)
    if k_pts < 4:
        return contour.copy()

    if smoothing is None:
        smoothing = float(k_pts)

    # splprep(per=1) writes into its input: FITPACK requires a periodic
    # curve to close exactly, so scipy overwrites the last point with the
    # first. Passing a view of the caller's centroids would therefore
    # corrupt them -- silently turning two real clusters into coincident
    # ones, which collapses the minimum-distance constraint used by the
    # randomization and changes results between successive calls. Always
    # hand splprep its own copy.
    pts = np.array(contour.T, dtype=float, copy=True)
    try:
        tck, _u = splprep(pts, s=smoothing, per=1, k=3)
        u = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        out = np.array(splev(u, tck)).T
        if not np.all(np.isfinite(out)):
            return contour.copy()
        return out
    except Exception:
        return contour.copy()


# The name this helper had while it was private. The shared perimeter
# occupancy of tools.mps_crosschannel needs it too, so it is public now.


def _place_with_min_distance(*args, **kwargs):          # pragma: no cover
    """Deprecated alias of ``place_with_min_distance``."""
    return place_with_min_distance(*args, **kwargs)


def build_annulus_candidates(
    smoothed: NDArray[np.float64],
    half_width_nm: float = DEFAULT_ANNULUS_HALF_WIDTH_NM,
    grid_spacing_nm: float = DEFAULT_GRID_SPACING_NM,
) -> NDArray[np.float64]:
    """
    Dense grid of candidate positions inside the annulus around the contour.

    The annulus is defined implicitly: a grid point qualifies when its
    distance to the smoothed contour is <= ``half_width_nm``. This is the
    same region the paper's two offset curves delimit, without the
    self-intersection failure mode explicit offsets have on concave
    contours.

    Returns
    -------
    (M, 2) candidate positions.
    """
    smoothed = np.asarray(smoothed, dtype=float)
    lo = smoothed.min(axis=0) - half_width_nm - grid_spacing_nm
    hi = smoothed.max(axis=0) + half_width_nm + grid_spacing_nm

    xs = np.arange(lo[0], hi[0] + grid_spacing_nm, grid_spacing_nm)
    ys = np.arange(lo[1], hi[1] + grid_spacing_nm, grid_spacing_nm)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    # Distance from every grid point to the nearest contour sample. With a
    # densely sampled contour this approximates distance-to-curve closely.
    tree = cKDTree(smoothed)
    dist, _ = tree.query(grid, k=1)
    return grid[dist <= half_width_nm]


def place_with_min_distance(
    candidates: NDArray[np.float64],
    n_wanted: int,
    min_distance: float,
    rng: np.random.Generator,
    max_attempts_factor: int = 20,
) -> NDArray[np.float64]:
    """
    Sequentially place ``n_wanted`` centres drawn from ``candidates`` such
    that no two are closer than ``min_distance``.

    Uses a spatial hash keyed on cells of side ``min_distance`` so each
    acceptance test only inspects the 3x3 neighbouring cells -- O(1) per
    attempt instead of O(placed).

    Returns whatever it managed to place (possibly fewer than requested);
    the caller reports incomplete placements rather than silently
    accepting a thinner random pattern.
    """
    if n_wanted <= 0 or len(candidates) == 0:
        return np.empty((0, 2))

    if min_distance <= 0:
        idx = rng.choice(len(candidates), size=min(n_wanted, len(candidates)),
                         replace=False)
        return candidates[idx]

    cell = float(min_distance)
    buckets: Dict[Tuple[int, int], List[NDArray[np.float64]]] = {}
    placed: List[NDArray[np.float64]] = []

    order = rng.permutation(len(candidates))
    max_attempts = max_attempts_factor * n_wanted
    attempts = 0
    min_d_sq = min_distance * min_distance

    for ci in order:
        if len(placed) >= n_wanted or attempts >= max_attempts:
            break
        attempts += 1
        p = candidates[ci]
        cx, cy = int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell))

        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in buckets.get((cx + dx, cy + dy), ()):
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_d_sq:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            placed.append(p)
            buckets.setdefault((cx, cy), []).append(p)

    return np.array(placed) if placed else np.empty((0, 2))


def randomize_cluster_positions(
    centroids: NDArray[np.float64],
    contour: NDArray[np.float64],
    experimental_1nn_nm: Optional[NDArray[np.float64]] = None,
    n_iterations: int = DEFAULT_N_RANDOMIZATIONS,
    annulus_half_width_nm: float = DEFAULT_ANNULUS_HALF_WIDTH_NM,
    grid_spacing_nm: float = DEFAULT_GRID_SPACING_NM,
    random_seed: int = 0,
    min_distance_nm: Optional[float] = None,
) -> RandomizationResult:
    """
    Compare the observed cluster arrangement against a constrained random one.

    Parameters
    ----------
    centroids : (K, 2) observed cluster centres of mass, in nm.
    contour : (K, 2) reconstructed perimeter, in contour order.
    experimental_1nn_nm : the observed 1NN distances. Computed from
        ``centroids`` when omitted.
    n_iterations : randomizations per axon; the paper uses 1000.
    annulus_half_width_nm : +/-50 nm per the paper.
    grid_spacing_nm : candidate grid resolution. Not specified by the
        paper; 5 nm is far below the ~100+ nm inter-cluster distances, so
        the placement is effectively continuous.
    random_seed : pinned so the same axon always yields the same control.
        A randomization whose p-value moved between runs would not be a
        reportable number.
    min_distance_nm : override for the constraint. Defaults to the minimum
        experimental 1NN of this axon, as the paper specifies.

    Returns
    -------
    RandomizationResult
    """
    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"centroids must be (K, 2), got {centroids.shape}")
    n_clusters = len(centroids)
    warnings_: List[str] = []

    if n_clusters < 2:
        raise ValueError(
            f"Need at least 2 clusters to randomize, got {n_clusters}.")

    if experimental_1nn_nm is None:
        experimental_1nn_nm = _self_excluded_1nn(centroids)
    experimental_1nn_nm = np.asarray(experimental_1nn_nm, dtype=float).ravel()

    if min_distance_nm is None:
        min_distance_nm = float(np.min(experimental_1nn_nm))
        # A minimum of zero (coincident centroids) would silently switch the
        # placement off: with no separation to enforce, the "constrained"
        # random model becomes a completely unconstrained one and the
        # comparison loses its meaning. Fall back to the smallest non-zero
        # separation, which preserves the paper's intent, and say so.
        if min_distance_nm <= 0:
            nonzero = experimental_1nn_nm[experimental_1nn_nm > 0]
            if nonzero.size:
                min_distance_nm = float(nonzero.min())
                warnings_.append(
                    f"The smallest experimental 1NN is 0 nm (coincident "
                    f"cluster centroids), which would disable the minimum "
                    f"inter-cluster distance constraint entirely. Using the "
                    f"smallest non-zero separation ({min_distance_nm:.1f} nm) "
                    f"instead."
                )
            else:
                warnings_.append(
                    "All experimental 1NN distances are 0 nm; the "
                    "randomization runs without a separation constraint."
                )

    smoothed = smooth_contour_bspline(contour)
    candidates = build_annulus_candidates(
        smoothed, annulus_half_width_nm, grid_spacing_nm)

    if len(candidates) < n_clusters:
        raise ValueError(
            f"Only {len(candidates)} candidate positions in the annulus for "
            f"{n_clusters} clusters; increase the annulus width or reduce the "
            f"grid spacing."
        )

    rng = np.random.default_rng(random_seed)
    pooled: List[NDArray[np.float64]] = []
    medians = np.empty(n_iterations)
    placed_fracs = np.empty(n_iterations)
    n_incomplete = 0
    example: Optional[NDArray[np.float64]] = None

    for it in range(n_iterations):
        centres = place_with_min_distance(
            candidates, n_clusters, min_distance_nm, rng)
        placed_fracs[it] = len(centres) / n_clusters
        if len(centres) < n_clusters:
            n_incomplete += 1

        if len(centres) < 2:
            medians[it] = np.nan
            continue

        t = cKDTree(centres)
        d, _ = t.query(centres, k=2)
        nn = d[:, 1]
        pooled.append(nn)
        medians[it] = float(np.median(nn))
        if example is None:
            example = centres

    if not pooled:
        raise RuntimeError(
            "No randomization iteration produced a usable arrangement.")

    randomized = np.concatenate(pooled)

    if n_incomplete:
        warnings_.append(
            f"{n_incomplete}/{n_iterations} randomizations could not place all "
            f"{n_clusters} clusters at least {min_distance_nm:.0f} nm apart "
            f"inside the {2 * annulus_half_width_nm:.0f} nm annulus "
            f"(mean placed {100 * placed_fracs.mean():.1f}%). The random model "
            f"is therefore slightly sparser than the observed pattern, which "
            f"biases the comparison toward longer random distances."
        )

    ks = ks_2samp(experimental_1nn_nm, randomized)
    crossing = _cdf_crossing(experimental_1nn_nm, randomized)

    # The pooled random sample is n_iterations times larger than the
    # experimental one, so the KS p-value is driven largely by that size.
    # Flag it: the informative quantity is D and the shape of the two CDFs,
    # not the p-value.
    warnings_.append(
        f"KS p-value compares {experimental_1nn_nm.size} experimental "
        f"distances against {randomized.size:,} pooled randomized ones; with "
        f"such an unbalanced sample size almost any difference reaches "
        f"significance. Interpret D = {ks.statistic:.3f} and the CDF shapes, "
        f"not the p-value alone."
    )

    return RandomizationResult(
        experimental_1nn_nm=experimental_1nn_nm,
        randomized_1nn_nm=randomized,
        per_iteration_medians_nm=medians,
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        cdf_crossing=crossing,
        n_iterations=n_iterations,
        n_clusters=n_clusters,
        min_distance_nm=float(min_distance_nm),
        n_candidates=len(candidates),
        annulus_half_width_nm=annulus_half_width_nm,
        n_incomplete_iterations=n_incomplete,
        mean_placed_fraction=float(placed_fracs.mean()),
        smoothed_contour=smoothed,
        candidate_points=candidates,
        example_random_centers=example,
        warnings=warnings_,
    )


def _self_excluded_1nn(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Nearest-neighbour distance for each point, excluding the point itself
    BY INDEX rather than by assuming the self-match is returned first.

    With coincident points a k=2 query may return the duplicate ahead of
    the self-match, so taking column 1 blindly is both wrong and unstable
    between runs.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=min(n, 2))
    if n < 2:
        return np.array([])
    out = np.empty(n)
    for i in range(n):
        row_d, row_i = np.atleast_1d(dist[i]), np.atleast_1d(idx[i])
        keep = row_i != i
        out[i] = row_d[keep][0] if np.any(keep) else 0.0
    return out


def _cdf_crossing(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> Optional[float]:
    """
    CDF level at which the two empirical distributions cross.

    The paper reports a crossing near 0.8: below it the experimental data
    has more short distances than chance, above it fewer long ones. Returns
    the last crossing level, or None if the CDFs never cross.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return None

    grid = np.linspace(min(a[0], b[0]), max(a[-1], b[-1]), 4096)
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    diff = ca - cb

    # Compare consecutive NON-ZERO signs. Requiring both neighbours of a
    # transition to be non-zero misses every crossing that passes through a
    # flat stretch where the two CDFs coincide exactly -- which is the
    # common case, since both curves are step functions that share long
    # equal-valued runs.
    # Only count a sign change as a crossing when BOTH branches around it
    # are substantial. Two CDFs that never really cross still wobble by a
    # few samples in the tails, and reporting that as "the curves cross at
    # 0.99" would invent the paper's qualitative finding where the data
    # does not show it.
    max_abs = float(np.max(np.abs(diff)))
    if max_abs == 0:
        return None
    floor = 0.15 * max_abs

    strong = np.flatnonzero(np.abs(diff) >= floor)
    if strong.size < 2:
        return None
    signs = np.sign(diff[strong])
    changes = np.flatnonzero(signs[:-1] != signs[1:])
    if changes.size == 0:
        return None

    # Report the CDF level midway across the transition.
    j = int(changes[-1])
    i_lo, i_hi = int(strong[j]), int(strong[j + 1])
    return float((ca[i_lo] + cb[i_lo] + ca[i_hi] + cb[i_hi]) / 4.0)
