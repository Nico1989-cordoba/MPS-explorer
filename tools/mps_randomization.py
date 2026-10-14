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
than chance below that, fewer long ones above it. The aggregation level of
that test is not stated (presumably pooled over the paper's 34 axons), so a
per-axon D is not comparable with it.

What is reported, and why (SCI-3)
---------------------------------
* D, the largest gap between the observed 1NN CDF and the CDF pooled over
  the iterations, is kept as a descriptive distance WITH ITS OWN NULL: the
  same distance of each iteration against the pool, D_i, gives the null
  mean of D for this axon's number of clusters (about 0.87/sqrt(K) for
  independent data, larger for nearest-neighbour distances, which are not
  independent; Diggle 2013), a standardised z_D and a Monte Carlo rank
  p-value p_D = (1 + #{D_i >= D}) / (B + 1) (Besag & Diggle 1977; Phipson &
  Smyth 2010). The per-iteration medians give a two-sided p of the median.
  Both are labelled approximate: the hard core of the null is estimated
  from the same data (Dao & Genton 2014).
* The two-sample KS p-value is NOT reported. Its assumptions fail here:
  the K distances are dependent, the randomized "sample" is B replicates of
  one null rather than an independent sample, the hard core is estimated
  from the data (Lilliefors 1967) and the positions are discrete (Conover
  1972); under a true null it rejected up to about four times its nominal
  5 % in this project's simulations.
* B = 999 so that alpha (B + 1) is an integer at alpha = 0.05 and 0.01: the
  Monte Carlo p is then exact on its grid.
* The CDF crossing is reported only where the observed difference leaves
  the band the iterations themselves give (2.5-97.5 % of F_i - F_pool on a
  grid of 256 distances); it is called significant when, in addition, both
  branches exceed the Dvoretzky-Kiefer-Wolfowitz bound sqrt(ln(2/alpha) /
  (2K)) (Massart 1990). A single axon with tens of clusters often cannot
  resolve the crossing, and the program then says so.

Implementation choices the paper does not specify
-------------------------------------------------
* The annulus is defined implicitly as "within the half-width of the
  smoothed contour" rather than by two explicit offset curves. The region
  is identical wherever the offsets do not self-intersect, and stays well
  defined on concave contours, where they do.
* The contour is CLOSED before the periodic spline is fitted:
  ``splprep(per=1)`` overwrites the last point with the first, so an open
  contour leaves its last cluster out of the fit. The smoothing is given as
  an RMS residual per vertex (s = K x rms^2 with unit weights; Dierckx
  1993): 1 nm is near-interpolation, the paper's "B-spline interpolation".
* Placement is simple sequential inhibition (Diggle, Besag & Gleaves 1976;
  random sequential adsorption, Evans 1993) on a 5 nm grid with a spatial
  hash. An iteration that cannot place every cluster within the attempt
  cap is drawn again (up to 10 times) and, if it still cannot, counted and
  left out of the null: pooling sparser patterns would bias the null
  toward longer distances. The packing fraction of the constraint is
  reported with it.
* The observed centres farther than the half-width from the smoothed
  contour are counted: a Monte Carlo comparison only means something when
  the null could have produced the observed positions.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

# Paper constants (Gazal et al. 2026; another dataset and pipeline)
PAPER_KS_D = 0.054
PAPER_CDF_CROSSING = 0.8
DEFAULT_ANNULUS_HALF_WIDTH_NM = 50.0     # +/-50 nm -> 100 nm wide
# alpha (B + 1) an integer at alpha = 0.05 and 0.01 (C10): the Monte Carlo
# p-value is exact on its grid. The paper used 1000.
DEFAULT_N_RANDOMIZATIONS = 999
DEFAULT_GRID_SPACING_NM = 5.0
DEFAULT_SPLINE_SAMPLES = 2000
# RMS residual per contour vertex of the periodic spline (s = K x rms^2
# under unit weights): 1 nm is near-interpolation.
DEFAULT_SMOOTHING_RMS_NM = 1.0
# At most this many candidates inspected per centre before an iteration
# gives up (it only binds near the jamming limit of sequential placement).
DEFAULT_MAX_ATTEMPTS_FACTOR = 20
# An incomplete iteration is drawn again up to this many times, and left
# out of the null if it still cannot place every cluster.
DEFAULT_EXCLUDE_INCOMPLETE = True
DEFAULT_MAX_REDRAWS = 10
# Level of the CDF null band and of the DKW floor.
DEFAULT_CDF_BAND_ALPHA = 0.05
# Distances the CDF band is evaluated at.
CDF_GRID_POINTS = 256
# How the hard core (minimum separation) of the null is set: the paper's
# rule, or its first-order bias correction for a 1-D hard core.
HARD_CORE_RULES: Tuple[str, ...] = ("observed minimum", "bias-corrected")
DEFAULT_HARD_CORE = "observed minimum"

# The documented ranges of the editable values (never used to clamp).
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "band_half_width_nm": (20.0, 150.0),
    "smoothing_rms_nm": (0.0, 50.0),
    "max_attempts_factor": (5.0, 100.0),
    "grid_spacing_nm": (1.0, 20.0),
    "cdf_band_alpha": (0.01, 0.10),
}
ALLOWED_ITERATIONS: Tuple[int, ...] = (199, 399, 999, 1999)


@dataclass(frozen=True)
class RandomizationParams:
    """How the randomization control is run, besides B and the seed (which
    ``analyze_axon`` takes as ``n_randomizations`` and ``random_seed``).

    One store (the settings' ``randomization_params``); a value outside its
    documented range is kept and flagged, never clamped."""

    band_half_width_nm: float = DEFAULT_ANNULUS_HALF_WIDTH_NM
    smoothing_rms_nm: float = DEFAULT_SMOOTHING_RMS_NM
    exclude_incomplete: bool = DEFAULT_EXCLUDE_INCOMPLETE
    max_redraws: int = DEFAULT_MAX_REDRAWS
    max_attempts_factor: int = DEFAULT_MAX_ATTEMPTS_FACTOR
    grid_spacing_nm: float = DEFAULT_GRID_SPACING_NM
    cdf_band_alpha: float = DEFAULT_CDF_BAND_ALPHA
    hard_core: str = DEFAULT_HARD_CORE

    def __post_init__(self) -> None:
        if not (math.isfinite(self.band_half_width_nm) and self.band_half_width_nm > 0):
            raise ValueError(f"band half-width must be > 0 nm, got {self.band_half_width_nm!r}")
        if not (math.isfinite(self.smoothing_rms_nm) and self.smoothing_rms_nm >= 0):
            raise ValueError(f"smoothing RMS must be >= 0 nm, got {self.smoothing_rms_nm!r}")
        if int(self.max_redraws) < 0:
            raise ValueError(f"redraws must be >= 0, got {self.max_redraws!r}")
        if int(self.max_attempts_factor) < 1:
            raise ValueError(f"attempts factor must be >= 1, got {self.max_attempts_factor!r}")
        if not (math.isfinite(self.grid_spacing_nm) and self.grid_spacing_nm > 0):
            raise ValueError(f"grid spacing must be > 0 nm, got {self.grid_spacing_nm!r}")
        if not (0.0 < self.cdf_band_alpha < 0.5):
            raise ValueError(f"band alpha must be in (0, 0.5), got {self.cdf_band_alpha!r}")
        if self.hard_core not in HARD_CORE_RULES:
            raise ValueError(f"hard core must be one of {HARD_CORE_RULES}, got {self.hard_core!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "RandomizationParams":
        """The parameters a stored dict holds; unknown keys and values of the wrong kind are ignored (the
        default stands), values outside the documented range are kept."""
        kw: Dict[str, Any] = {}
        default = cls()
        for f in fields(cls):
            if not data or f.name not in data:
                continue
            raw = data[f.name]
            try:
                if isinstance(getattr(default, f.name), bool):
                    if not isinstance(raw, bool):
                        continue
                    value: Any = raw
                elif isinstance(getattr(default, f.name), int):
                    value = int(raw)
                    if value != float(raw):
                        continue
                elif isinstance(getattr(default, f.name), float):
                    value = float(raw)
                else:
                    value = str(raw)
            except (TypeError, ValueError):
                continue
            kw[f.name] = value
        try:
            return cls(**kw)
        except ValueError:
            # one impossible value: keep the others
            good: Dict[str, Any] = {}
            for name, value in kw.items():
                try:
                    cls(**{**good, name: value})
                    good[name] = value
                except ValueError:
                    pass
            return cls(**good)

    def outside_range(self) -> List[str]:
        """The names whose value lies outside the documented range."""
        out = []
        for name, (lo, hi) in PARAM_RANGES.items():
            v = float(getattr(self, name))
            if v < lo or v > hi:
                out.append(name)
        return out

    @property
    def smoothing_factor_per_vertex(self) -> float:
        """FITPACK's s per contour vertex (unit weights, nm^2)."""
        return float(self.smoothing_rms_nm) ** 2


@dataclass
class RandomizationResult:
    """Outcome of the randomization control for one axon."""

    experimental_1nn_nm: NDArray[np.float64]
    randomized_1nn_nm: NDArray[np.float64]      # pooled over the iterations used
    per_iteration_medians_nm: NDArray[np.float64]

    ks_statistic: float                         # D, descriptive
    cdf_crossing: Optional[float]               # F level where the band-exceeding excess ends

    n_iterations: int                           # requested (B)
    n_clusters: int
    min_distance_nm: float                      # the constraint actually used
    n_candidates: int
    annulus_half_width_nm: float

    n_incomplete_iterations: int                # could not place every cluster (after the redraws)
    mean_placed_fraction: float

    smoothed_contour: NDArray[np.float64]
    candidate_points: NDArray[np.float64]
    example_random_centers: Optional[NDArray[np.float64]] = None

    # --- D's own null and the Monte Carlo p-values (approximate) ----------
    per_iteration_d: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    ks_null_mean: Optional[float] = None
    ks_null_sd: Optional[float] = None
    ks_z: Optional[float] = None
    ks_p_mc_d: Optional[float] = None
    ks_p_mc_median: Optional[float] = None
    n_iterations_used: int = 0                  # complete iterations in the null (B_eff)

    # --- the CDF band ------------------------------------------------------
    cdf_grid_nm: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    cdf_null: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))       # F_pool on the grid
    cdf_band_lo: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))    # F_pool + lower band
    cdf_band_hi: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    cdf_dkw_floor: Optional[float] = None
    cdf_crossing_significant: bool = False

    # --- how it ran -----------------------------------------------------------
    params: RandomizationParams = field(default_factory=RandomizationParams)
    n_redrawn: int = 0                          # redraws spent on incomplete iterations
    packing_fraction: Optional[float] = None    # K pi d^2 / (4 x band area)
    n_outside_band: int = 0                     # observed centres beyond the half-width
    observed_min_1nn_nm: Optional[float] = None
    hard_core: str = DEFAULT_HARD_CORE

    warnings: List[str] = field(default_factory=list)

    @property
    def experimental_median_nm(self) -> Optional[float]:
        d = self.experimental_1nn_nm
        return float(np.median(d)) if d.size else None

    @property
    def randomized_median_nm(self) -> Optional[float]:
        d = self.randomized_1nn_nm
        return float(np.median(d)) if d.size else None

    @property
    def fraction_outside_band(self) -> Optional[float]:
        return None if self.n_clusters <= 0 else self.n_outside_band / self.n_clusters

    def crossing_words(self) -> str:
        """The CDF crossing as the results table writes it."""
        if self.cdf_crossing is None:
            return "not detectable at this K"
        if self.cdf_crossing_significant:
            return f"short-distance excess up to F = {self.cdf_crossing:.2f}"
        return (f"excess up to F = {self.cdf_crossing:.2f} inside the DKW bound "
                f"(not significant at this K)")


def smooth_contour_bspline(
    contour: NDArray[np.float64],
    n_samples: int = DEFAULT_SPLINE_SAMPLES,
    smoothing: Optional[float] = None,
) -> NDArray[np.float64]:
    """
    Periodic cubic B-spline through a closed contour (scipy splprep, FITPACK).

    Parameters
    ----------
    contour : (K, 2) vertices in contour order. FITPACK needs a periodic
        curve to close exactly, so pass it CLOSED (last vertex = first):
        given open, splprep overwrites the last vertex with the first and
        that vertex is left out of the fit.
    n_samples : points to evaluate along the smoothed curve.
    smoothing : FITPACK's ``s``, the bound on the summed squared residuals
        under unit weights (Dierckx 1993). None selects ``s = K`` (K the
        number of vertices given): an RMS residual of at most 1 nm per
        vertex, i.e. near-interpolation. This is a convention of this
        program; scipy's suggested range m +/- sqrt(2m) applies to weights
        1/sigma, not to unit weights.

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


def close_contour(contour: NDArray[np.float64]) -> NDArray[np.float64]:
    """The contour with its first vertex appended, unless it is closed already."""
    c = np.asarray(contour, dtype=float).reshape(-1, 2)
    if len(c) >= 2 and not np.array_equal(c[0], c[-1]):
        c = np.vstack([c, c[:1]])
    return c


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
    region the paper's two offset curves delimit wherever they do not
    self-intersect, and stays well defined on concave contours, where they
    do.

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
    max_attempts_factor: int = DEFAULT_MAX_ATTEMPTS_FACTOR,
) -> NDArray[np.float64]:
    """
    Sequentially place ``n_wanted`` centres drawn from ``candidates`` such
    that no two are closer than ``min_distance`` (simple sequential
    inhibition, Diggle, Besag & Gleaves 1976).

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


def corrected_hard_core(d_min_nm: float, n_clusters: int, perimeter_nm: float) -> float:
    """First-order bias correction of a 1-D hard core estimated by the smallest observed spacing:
    E[d_min] = d0 + (P - K d0) / K^2, hence d0_hat = (K^2 d_min - P) / (K (K - 1)). May be <= 0."""
    k = int(n_clusters)
    if k < 2:
        return float(d_min_nm)
    return float((k * k * d_min_nm - perimeter_nm) / (k * (k - 1)))


def dkw_floor(n: int, alpha: float = DEFAULT_CDF_BAND_ALPHA) -> float:
    """The Dvoretzky-Kiefer-Wolfowitz bound (Massart 1990) of an empirical CDF of n points at level alpha."""
    return float(math.sqrt(math.log(2.0 / alpha) / (2.0 * max(int(n), 1))))


def ks_distance(a_sorted: NDArray[np.float64], b_sorted: NDArray[np.float64],
                cdf_b_at_b: Optional[NDArray[np.float64]] = None) -> float:
    """sup |F_a - F_b| of two sorted samples; the same number as ``scipy.stats.ks_2samp(a, b).statistic``.
    ``cdf_b_at_b`` (F_b at b's own points) may be passed when b is reused."""
    a = np.asarray(a_sorted, dtype=float)
    b = np.asarray(b_sorted, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if cdf_b_at_b is None:
        cdf_b_at_b = np.searchsorted(b, b, side="right") / b.size
    fa_b = np.searchsorted(a, b, side="right") / a.size
    fa_a = np.searchsorted(a, a, side="right") / a.size
    fb_a = np.searchsorted(b, a, side="right") / b.size
    return float(max(np.max(np.abs(fa_b - cdf_b_at_b)), np.max(np.abs(fa_a - fb_a))))


def monte_carlo_p(observed: float, null: NDArray[np.float64], two_sided: bool = False) -> Optional[float]:
    """Rank p-value (1 + b) / (B + 1) (Phipson & Smyth 2010). One-sided: b = #{null >= observed}; two-sided:
    twice the smaller tail, capped at 1."""
    v = np.asarray(null, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or not math.isfinite(observed):
        return None
    tol = 1e-12 * max(1.0, abs(observed))
    hi = (1.0 + float(np.count_nonzero(v >= observed - tol))) / (v.size + 1.0)
    if not two_sided:
        return hi
    lo = (1.0 + float(np.count_nonzero(v <= observed + tol))) / (v.size + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi)))


def cdf_band_crossing(
    grid: NDArray[np.float64], f_obs: NDArray[np.float64], f_null: NDArray[np.float64],
    band_lo: NDArray[np.float64], band_hi: NDArray[np.float64], floor: float,
) -> Tuple[Optional[float], bool]:
    """Where the band-exceeding excess of short distances ends.

    ``band_lo`` / ``band_hi`` are the null band of F_obs - F_null (differences, not levels). A crossing is
    reported when the difference rises above the band at some distance and falls below it at a longer one; its
    level is F_null where the difference first turns non-positive after the last excess point. It is
    significant when both branches also exceed ``floor`` (the DKW bound). Returns (level or None,
    significant)."""
    diff = np.asarray(f_obs, float) - np.asarray(f_null, float)
    up = diff > np.asarray(band_hi, float)
    down = diff < np.asarray(band_lo, float)
    if not up.any() or not down.any():
        return None, False
    first_up = int(np.argmax(up))
    later_down = np.flatnonzero(down[first_up + 1:])
    if later_down.size == 0:
        return None, False
    j = first_up + 1 + int(later_down[0])
    i_last = int(np.flatnonzero(up[:j])[-1])
    k = i_last + 1 + int(np.argmax(diff[i_last + 1:j + 1] <= 0.0))
    level = float(f_null[k])
    sig_up = bool(np.any(up[:j] & (diff[:j] > floor)))
    sig_down = bool(np.any(down[j:] & (diff[j:] < -floor)))
    return level, bool(sig_up and sig_down)


def randomize_cluster_positions(
    centroids: NDArray[np.float64],
    contour: NDArray[np.float64],
    experimental_1nn_nm: Optional[NDArray[np.float64]] = None,
    n_iterations: int = DEFAULT_N_RANDOMIZATIONS,
    annulus_half_width_nm: Optional[float] = None,
    grid_spacing_nm: Optional[float] = None,
    random_seed: int = 0,
    min_distance_nm: Optional[float] = None,
    params: Optional[RandomizationParams] = None,
) -> RandomizationResult:
    """
    Compare the observed cluster arrangement against a constrained random one.

    Parameters
    ----------
    centroids : (K, 2) observed cluster centres of mass, in nm.
    contour : (K, 2) reconstructed perimeter, in contour order (open; it is
        closed here before the spline is fitted).
    experimental_1nn_nm : the observed 1NN distances. Computed from
        ``centroids`` when omitted.
    n_iterations : randomizations per axon (B); 999 by default (the paper
        used 1000).
    annulus_half_width_nm, grid_spacing_nm : override ``params``' values.
    random_seed : pinned so the same axon always yields the same control.
        The Monte Carlo error is a property of B: SE(p) = sqrt(p (1 - p) /
        B), about 0.007 at p = 0.05 (Marriott 1979).
    min_distance_nm : override for the constraint. Defaults to the minimum
        experimental 1NN of this axon, as the paper specifies (or its bias
        correction, ``params.hard_core``).
    params : the rest of how the control runs (``RandomizationParams``).

    Raises
    ------
    ValueError when every experimental 1NN is 0 nm (a hard-core model has
    no coincident points: the control is refused rather than run without
    its constraint), or when the band cannot hold the clusters.
    RuntimeError when no iteration could place every cluster and
    incomplete iterations are excluded.
    """
    p = params if params is not None else RandomizationParams()
    half_width = float(p.band_half_width_nm if annulus_half_width_nm is None else annulus_half_width_nm)
    grid_step = float(p.grid_spacing_nm if grid_spacing_nm is None else grid_spacing_nm)

    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"centroids must be (K, 2), got {centroids.shape}")
    n_clusters = len(centroids)
    warnings_: List[str] = []

    if n_clusters < 2:
        raise ValueError(
            f"Need at least 2 clusters to randomize, got {n_clusters}.")
    n_iterations = int(n_iterations)
    if n_iterations < 1:
        raise ValueError(f"Need at least one iteration, got {n_iterations}.")

    if experimental_1nn_nm is None:
        experimental_1nn_nm = _self_excluded_1nn(centroids)
    experimental_1nn_nm = np.asarray(experimental_1nn_nm, dtype=float).ravel()

    closed = close_contour(contour)
    smoothed = smooth_contour_bspline(
        closed, smoothing=n_clusters * p.smoothing_factor_per_vertex)
    ring = np.vstack([smoothed, smoothed[:1]])
    smooth_length = float(np.sum(np.linalg.norm(np.diff(ring, axis=0), axis=1)))

    observed_min = float(np.min(experimental_1nn_nm))
    hard_core = p.hard_core
    if min_distance_nm is None:
        min_distance_nm = observed_min
        # A minimum of zero (coincident centroids) would silently switch the
        # placement off: with no separation to enforce, the "constrained"
        # random model becomes a completely unconstrained one and the
        # comparison loses its meaning. Fall back to the smallest non-zero
        # separation, which preserves the paper's intent, and say so.
        if min_distance_nm <= 0:
            nonzero = experimental_1nn_nm[experimental_1nn_nm > 0]
            if not nonzero.size:
                raise ValueError(
                    "Every experimental 1NN distance is 0 nm (coincident cluster centroids): a hard-core null "
                    "has no coincident points, so the randomization control is refused. Check the clustering.")
            min_distance_nm = float(nonzero.min())
            warnings_.append(
                f"The smallest experimental 1NN is 0 nm (coincident "
                f"cluster centroids), which would disable the minimum "
                f"inter-cluster distance constraint entirely. Using the "
                f"smallest non-zero separation ({min_distance_nm:.1f} nm) "
                f"instead."
            )
        if hard_core == "bias-corrected":
            observed_rule = float(min_distance_nm)
            min_distance_nm = corrected_hard_core(observed_rule, n_clusters, smooth_length)
            if min_distance_nm <= 0:
                warnings_.append(
                    f"The bias-corrected hard core is {min_distance_nm:.1f} nm (observed minimum "
                    f"{observed_rule:.1f} nm, K = {n_clusters}): no evidence of a hard core, so the null places "
                    f"the centres without a minimum separation (approximate).")
                min_distance_nm = 0.0
    else:
        hard_core = "given"

    candidates = build_annulus_candidates(smoothed, half_width, grid_step)

    if len(candidates) < n_clusters:
        raise ValueError(
            f"Only {len(candidates)} candidate positions in the annulus for "
            f"{n_clusters} clusters; increase the annulus width or reduce the "
            f"grid spacing."
        )

    # Could the null have produced the observed positions at all?
    d_obs, _ = cKDTree(smoothed).query(centroids, k=1)
    n_outside = int(np.count_nonzero(d_obs > half_width))
    if n_outside:
        warnings_.append(
            f"{n_outside} of {n_clusters} observed cluster centres lie farther than {half_width:g} nm from the "
            f"smoothed contour, where the randomization never places one: the comparison then partly measures "
            f"the band, not the arrangement.")

    band_area = 2.0 * half_width * smooth_length
    phi = (float(n_clusters * np.pi * min_distance_nm ** 2 / (4.0 * band_area))
           if band_area > 0 else None)

    rng = np.random.default_rng(random_seed)
    pooled: List[NDArray[np.float64]] = []
    medians: List[float] = []
    placed_fracs = np.empty(n_iterations)
    n_incomplete = 0
    n_redrawn = 0
    example: Optional[NDArray[np.float64]] = None
    tries = 1 + (int(p.max_redraws) if p.exclude_incomplete else 0)

    for it in range(n_iterations):
        centres = np.empty((0, 2))
        for attempt in range(tries):
            if attempt:
                n_redrawn += 1
            centres = place_with_min_distance(
                candidates, n_clusters, min_distance_nm, rng,
                max_attempts_factor=int(p.max_attempts_factor))
            if len(centres) >= n_clusters:
                break
        placed_fracs[it] = len(centres) / n_clusters
        if len(centres) < n_clusters:
            n_incomplete += 1
            if p.exclude_incomplete:
                continue

        if len(centres) < 2:
            continue

        t = cKDTree(centres)
        d, _ = t.query(centres, k=2)
        nn = d[:, 1]
        pooled.append(nn)
        medians.append(float(np.median(nn)))
        if example is None:
            example = centres

    phi_text = "" if phi is None else f"; packing fraction phi = {phi:.2f}"
    if not pooled:
        raise RuntimeError(
            f"No randomization could place all {n_clusters} clusters at least {min_distance_nm:.0f} nm apart "
            f"inside the {2 * half_width:.0f} nm band ({tries} draw(s) per iteration{phi_text}); the control is "
            f"refused rather than compared with a sparser null.")

    if n_incomplete:
        if p.exclude_incomplete:
            warnings_.append(
                f"{n_incomplete}/{n_iterations} randomizations could not place all {n_clusters} clusters at least "
                f"{min_distance_nm:.0f} nm apart inside the {2 * half_width:.0f} nm band, even after "
                f"{int(p.max_redraws)} redraws each{phi_text}; they are left out of the null, which uses the "
                f"{len(pooled)} complete ones.")
        else:
            warnings_.append(
                f"{n_incomplete}/{n_iterations} randomizations could not place all "
                f"{n_clusters} clusters at least {min_distance_nm:.0f} nm apart "
                f"inside the {2 * half_width:.0f} nm band "
                f"(mean placed {100 * placed_fracs.mean():.1f}%{phi_text}). They are pooled, so the random "
                f"model is sparser than the observed pattern, which biases the comparison toward longer "
                f"random distances.")

    # ---------------- D, its own null and the Monte Carlo p-values -------
    randomized = np.concatenate(pooled)
    pool = np.sort(randomized)
    cdf_pool_at_pool = np.searchsorted(pool, pool, side="right") / pool.size
    exp_sorted = np.sort(experimental_1nn_nm)
    d_obs_ks = ks_distance(exp_sorted, pool, cdf_pool_at_pool)
    sorted_iters = [np.sort(nn) for nn in pooled]
    d_null = np.array([ks_distance(nn, pool, cdf_pool_at_pool) for nn in sorted_iters])
    b_used = len(pooled)
    null_mean = float(np.mean(d_null))
    null_sd = float(np.std(d_null, ddof=1)) if b_used > 1 else None
    z = (None if null_sd is None or null_sd <= 0 else float((d_obs_ks - null_mean) / null_sd))
    p_d = monte_carlo_p(d_obs_ks, d_null)
    med_arr = np.asarray(medians, dtype=float)
    p_med = monte_carlo_p(float(np.median(experimental_1nn_nm)), med_arr, two_sided=True)

    # ---------------- the CDF band and the crossing -----------------------
    lo_x = float(min(pool[0], exp_sorted[0]))
    hi_x = float(max(pool[-1], exp_sorted[-1]))
    grid = np.linspace(lo_x, hi_x, CDF_GRID_POINTS)
    f_null = np.searchsorted(pool, grid, side="right") / pool.size
    f_obs = np.searchsorted(exp_sorted, grid, side="right") / exp_sorted.size
    diffs = np.empty((b_used, grid.size))
    for i, nn in enumerate(sorted_iters):
        diffs[i] = np.searchsorted(nn, grid, side="right") / nn.size - f_null
    alpha = float(p.cdf_band_alpha)
    band_lo = np.percentile(diffs, 100.0 * alpha / 2.0, axis=0)
    band_hi = np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0), axis=0)
    floor = dkw_floor(n_clusters, alpha)
    crossing, significant = cdf_band_crossing(grid, f_obs, f_null, band_lo, band_hi, floor)

    return RandomizationResult(
        experimental_1nn_nm=experimental_1nn_nm,
        randomized_1nn_nm=randomized,
        per_iteration_medians_nm=med_arr,
        ks_statistic=float(d_obs_ks),
        cdf_crossing=crossing,
        n_iterations=n_iterations,
        n_clusters=n_clusters,
        min_distance_nm=float(min_distance_nm),
        n_candidates=len(candidates),
        annulus_half_width_nm=half_width,
        n_incomplete_iterations=n_incomplete,
        mean_placed_fraction=float(placed_fracs.mean()),
        smoothed_contour=smoothed,
        candidate_points=candidates,
        example_random_centers=example,
        per_iteration_d=d_null,
        ks_null_mean=null_mean,
        ks_null_sd=null_sd,
        ks_z=z,
        ks_p_mc_d=p_d,
        ks_p_mc_median=p_med,
        n_iterations_used=b_used,
        cdf_grid_nm=grid,
        cdf_null=f_null,
        cdf_band_lo=np.clip(f_null + band_lo, 0.0, 1.0),
        cdf_band_hi=np.clip(f_null + band_hi, 0.0, 1.0),
        cdf_dkw_floor=floor,
        cdf_crossing_significant=bool(significant),
        params=RandomizationParams(**{**p.to_dict(), "band_half_width_nm": half_width,
                                      "grid_spacing_nm": grid_step}),
        n_redrawn=n_redrawn,
        packing_fraction=phi,
        n_outside_band=n_outside,
        observed_min_1nn_nm=observed_min,
        hard_core=hard_core,
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
