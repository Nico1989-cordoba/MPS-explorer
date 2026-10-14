# -*- coding: utf-8 -*-
"""
Nearest-neighbour spacing between betaII-spectrin cluster centres.

Implements parameter 5 of Gazal et al. (2026):

  "we analyzed the nearest-neighbor (NN) distances between the centers of
   mass of betaII-spectrin clusters displayed along the axonal perimeter of
   individual MPS segments."

The paper reports this quantity in TWO different ways, which do not give
the same number and must both be produced:

  * Fig. 4A -- the POOLED distribution of every 1NN distance from every
    axon, whose prominent peak (a mode, not a median) sits at ~200 nm.
  * Fig. 4B -- one boxplot PER AXON, each axon's median highlighted; the
    median of those 34 per-axon medians is 260 nm.

Comparing with the published values
-----------------------------------
The median 1NN depends on how many clusters a pipeline detects per
micrometre of perimeter -- the mean 1NN cannot exceed P/K, and the median
equals P/K for evenly spaced centres -- as well as on the regularity of
the arrangement. SMLM cluster detection depends on localization and label
density, blinking and the clustering parameters (Baumgart et al. 2016;
Burgert et al. 2015; Nieves et al. 2023), and an axial slab can collect
clusters of a neighbouring ring, so two measurements of one structure can
legitimately report different 1NN values: agreement with Gazal et al.
(2026) is not a validation criterion. The analysis therefore reports the
clusters per micrometre and the density-normalised regularity index
r = median 1NN x K / P beside the 1NN (``AxonAnalysis.nn_regularity_index``;
about 0.47 for a random hard-core arrangement and 1 for evenly spaced
centres in this project's simulation), and ``validate_nn1_controls.py``
gives a synthetic control for each factor.

Distance metric
---------------
Euclidean, straight-line, between centres of mass -- not arc length along
the reconstructed contour. Chosen to match the paper: it describes the
measurement as "center-to-center", and its randomization control
(parameter 8) relocates cluster centres inside a 2D annulus and recomputes
1NN there, which has no stable geodesic equivalent. ``geodesic=True`` is
offered for exploration but is NOT the paper-faithful default.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde
from sklearn.neighbors import KDTree

# Paper reference values
PAPER_POOLED_PEAK_NM = 200.0
PAPER_MEDIAN_OF_MEDIANS_NM = 260.0


@dataclass
class NNResult:
    """Nearest-neighbour distances for the clusters of a single axon."""

    distances_nm: NDArray[np.float64]   # (K, k) distances to the k nearest
    k: int
    n_clusters: int
    warnings: List[str] = field(default_factory=list)
    # (K, k) rows of the centroid array those distances are to, so a table
    # can name the neighbour and not only say how far it is.
    neighbour_index: Optional[NDArray[np.intp]] = None

    @property
    def first_nn_nm(self) -> NDArray[np.float64]:
        """The 1NN distance for each cluster (first column)."""
        if self.distances_nm.size == 0:
            return np.array([])
        return self.distances_nm[:, 0]

    @property
    def median_1nn_nm(self) -> Optional[float]:
        """This axon's median 1NN -- the value plotted per axon in Fig. 4B."""
        d = self.first_nn_nm
        return float(np.median(d)) if d.size else None

    @property
    def mean_1nn_nm(self) -> Optional[float]:
        d = self.first_nn_nm
        return float(np.mean(d)) if d.size else None


def compute_nn_distances(
    centroids: NDArray[np.float64],
    k: int = 1,
) -> NNResult:
    """
    Euclidean k-nearest-neighbour distances between cluster centres of mass.

    Parameters
    ----------
    centroids : (K, 2) cluster centres in nm (the good/retained clusters).
    k : how many neighbours to return per cluster. Default 1 -- the paper's
        1NN. Larger k is available for exploration.

    Returns
    -------
    NNResult with a (K, k) distance array; the self-distance (always 0) is
    excluded.

    Notes
    -----
    Mirrors the KDTree approach already used by ``MPS_explorer.KNdist_hist``
    so results stay comparable with the software's existing output, but
    returns the raw values rather than only a histogram, and adds the
    per-axon median the paper reports.
    """
    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"centroids must be (K, 2), got {centroids.shape}")

    n = len(centroids)
    warnings_: List[str] = []

    if n < 2:
        warnings_.append(
            f"Only {n} cluster(s): nearest-neighbour distances are undefined."
        )
        return NNResult(np.empty((0, k)), k, n, warnings_,
                        neighbour_index=np.empty((0, k), dtype=np.intp))

    k_eff = min(k, n - 1)
    if k_eff < k:
        warnings_.append(
            f"Requested k={k} neighbours but only {n} clusters exist; "
            f"using k={k_eff}."
        )

    tree = KDTree(centroids)
    dist, idx = tree.query(centroids, k_eff + 1)

    # Drop each point's match to ITSELF by index rather than assuming it is
    # column 0. When two centroids coincide exactly, the query may return
    # the duplicate before the self-match, and slicing [:, 1:] would then
    # silently discard a real zero-distance neighbour and keep the self
    # match -- a result that also varies between runs.
    rows = np.arange(n)[:, None]
    is_self = idx == rows
    # Keep only the first self-match per row, in case a point is its own
    # duplicate: everything else is a genuine neighbour.
    first_self = np.zeros_like(is_self)
    has_self = is_self.any(axis=1)
    first_self[has_self, np.argmax(is_self[has_self], axis=1)] = True

    kept = np.empty((n, k_eff), dtype=float)
    # Which cluster each of those distances is to. The per-cluster table
    # exports it beside the distance: a 1NN of 200 nm says nothing about
    # which neighbour, and a reader cannot recover it from the centroids
    # without repeating this query.
    kept_index = np.empty((n, k_eff), dtype=np.intp)
    for i in range(n):
        keep = ~first_self[i] if has_self[i] else np.ones(len(dist[i]), bool)
        kept[i] = dist[i][keep][:k_eff]
        kept_index[i] = idx[i][keep][:k_eff]

    n_dup = int(np.count_nonzero(kept[:, 0] == 0.0))
    if n_dup:
        warnings_.append(
            f"{n_dup} cluster centre(s) have a nearest neighbour at exactly "
            f"0 nm, i.e. coincident centroids. Check the clustering: this "
            f"collapses the minimum inter-cluster distance used by the "
            f"randomization control."
        )

    return NNResult(kept, k_eff, n, warnings_, neighbour_index=kept_index)


def compute_nn_geodesic(
    contour: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Arc-length spacing between consecutive centres along the reconstructed
    contour. Provided for comparison only -- the paper-faithful metric is
    the Euclidean one in ``compute_nn_distances``.

    Parameters
    ----------
    contour : (K, 2) centroids already in contour order (from
        ``mps_geometry.reconstruct_perimeter().contour``).

    Returns
    -------
    (K,) array of distances from each centre to the next along the closed
    contour.
    """
    contour = np.asarray(contour, dtype=float)
    if len(contour) < 2:
        return np.array([])
    closed = np.vstack([contour, contour[:1]])
    return np.linalg.norm(np.diff(closed, axis=0), axis=1)


# ============================================================================
# Pooled distribution and its peak (Fig. 4A)
# ============================================================================

@dataclass
class PooledNNResult:
    """Pooled 1NN distribution across many axons.

    The median and quartiles of the pooled distances are the primary,
    smoothing-free summary. The peak (mode) is a smoothing-dependent
    quantity: it is given by two estimators, each with its smoothing
    parameter and a bootstrap 95 % interval obtained by resampling AXONS
    (pooling weights axons by their cluster count, so resampling distances
    would ignore that nesting)."""

    all_distances_nm: NDArray[np.float64]
    per_axon_medians_nm: NDArray[np.float64]
    peak_nm: float                 # mode of the pooled distribution (Fig. 4A), by peak_method
    peak_method: str
    median_of_medians_nm: float    # Fig. 4B reference line
    n_axons: int
    warnings: List[str] = field(default_factory=list)
    pooled_median_nm: Optional[float] = None
    pooled_quartiles_nm: Optional[Tuple[float, float]] = None
    # The two estimators of the mode, each with its smoothing and interval.
    kde_peak_nm: Optional[float] = None
    kde_bandwidth_nm: Optional[float] = None
    kde_peak_ci_nm: Optional[Tuple[float, float]] = None
    hist_peak_nm: Optional[float] = None
    hist_bin_width_nm: Optional[float] = None
    hist_bins: str = "fd"
    hist_peak_ci_nm: Optional[Tuple[float, float]] = None
    n_bootstrap: int = 0

    def summary_lines(self) -> List[str]:
        """The helper's printout: median and quartiles first, then each mode with its smoothing and interval."""
        def ci(c: Optional[Tuple[float, float]]) -> str:
            return "" if c is None else f", 95 % CI {c[0]:.0f}-{c[1]:.0f} nm over axons"
        out = [f"{self.n_axons} axons, {self.all_distances_nm.size} distances"]
        if self.pooled_median_nm is not None and self.pooled_quartiles_nm is not None:
            out.append(f"pooled median {self.pooled_median_nm:.1f} nm (quartiles "
                       f"{self.pooled_quartiles_nm[0]:.1f}-{self.pooled_quartiles_nm[1]:.1f} nm)")
        out.append(f"median of per-axon medians {self.median_of_medians_nm:.1f} nm")
        if self.kde_peak_nm is not None:
            out.append(f"KDE mode {self.kde_peak_nm:.1f} nm (Scott bandwidth {self.kde_bandwidth_nm:.1f} nm"
                       f"{ci(self.kde_peak_ci_nm)})")
        if self.hist_peak_nm is not None:
            out.append(f"histogram mode {self.hist_peak_nm:.1f} nm ({self.hist_bins} bins, width "
                       f"{self.hist_bin_width_nm:.1f} nm{ci(self.hist_peak_ci_nm)})")
        return out


# The literal-reading option of the histogram mode: the bin count of a
# reading of the published figure (Gazal et al. 2026 state none).
LITERAL_HISTOGRAM_BINS = 50
HISTOGRAM_BINS_RANGE = (20, 100)
KDE_BANDWIDTH_FACTOR_RANGE = (0.5, 2.0)
DEFAULT_MODE_BOOTSTRAP = 500


def fd_bin_width(values: NDArray[np.float64]) -> float:
    """The Freedman-Diaconis bin width 2 IQR n^(-1/3) (Freedman & Diaconis 1981)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    q1, q3 = np.percentile(v, [25.0, 75.0])
    return float(2.0 * (q3 - q1) * v.size ** (-1.0 / 3.0))


def estimate_mode(
    values: NDArray[np.float64],
    method: str = "kde",
    bins: Any = "fd",
    bw_method: Optional[float] = None,
) -> float:
    """
    Estimate the peak (mode) of a distribution.

    The mode of a pooled sample is a smoothing-dependent functional: with a
    bandwidth h the KDE mode has a sampling sd of order n^(-1/5) (Parzen
    1962; Silverman 1986), far larger than that of a median. Gazal et al.
    (2026) read a ~200 nm peak off a pooled distribution without stating a
    bin width or a smoothing, so no estimator can claim to reproduce their
    procedure; the mode is reported with its smoothing parameter and an
    interval, beside the median and quartiles. Two estimators:

    * "kde" (default): argmax of a Gaussian kernel density on a fine grid,
      Scott's rule bandwidth (scipy's default; Scott 1992), times
      ``bw_method`` when a factor is given (documented range 0.5-2).
    * "histogram": centre of the tallest bin. ``bins`` "fd" (default) is
      the Freedman-Diaconis width 2 IQR n^(-1/3), robust to extremes; an
      integer is a bin count over the data range (50 is the literal-reading
      option; documented range 20-100), which ties the width to the
      sample's maximum.

    Report both when the number matters.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate the mode of an empty array.")
    if values.size == 1:
        return float(values[0])

    if method == "histogram":
        if isinstance(bins, str):
            if bins != "fd":
                raise ValueError(f"bins must be 'fd' or a count, got {bins!r}")
            width = fd_bin_width(values)
            span = float(values.max() - values.min())
            n_bins = 1 if width <= 0 or span <= 0 else max(1, int(np.ceil(span / width)))
        else:
            n_bins = int(bins)
        counts, edges = np.histogram(values, bins=n_bins)
        i = int(np.argmax(counts))
        return float((edges[i] + edges[i + 1]) / 2)

    if method == "kde":
        if float(np.ptp(values)) == 0.0:
            return float(values[0])
        if bw_method is None:
            kde = gaussian_kde(values)
        else:
            scott = float(values.size ** (-1.0 / 5.0))
            kde = gaussian_kde(values, bw_method=scott * float(bw_method))
        grid = np.linspace(values.min(), values.max(), 4096)
        return float(grid[int(np.argmax(kde(grid)))])

    raise ValueError(f"method must be 'kde' or 'histogram', got {method!r}")


def pool_nn_across_axons(
    per_axon_distances: Sequence[NDArray[np.float64]],
    peak_method: str = "kde",
    bins: Any = "fd",
    n_bootstrap: int = DEFAULT_MODE_BOOTSTRAP,
    random_seed: int = 0,
) -> PooledNNResult:
    """
    Combine per-axon 1NN distances into the summaries of the validation
    helper: the pooled median and quartiles (primary, smoothing-free), the
    median of per-axon medians (Fig. 4B's quantity), and the pooled mode by
    the KDE and by the histogram, each with an axon-bootstrap 95 % interval.

    Parameters
    ----------
    per_axon_distances : one 1-D array of 1NN distances per axon.
    peak_method : which mode fills ``peak_nm``: "kde" or "histogram".
    bins : the histogram's bins: "fd" (Freedman-Diaconis) or a count (50 is
        the literal-reading option).
    n_bootstrap : resamples of the AXONS for the intervals (0: none).

    Returns
    -------
    PooledNNResult. The paper's values (~200 nm peak, 260 nm median of
    medians) come from another dataset and another pipeline and are not a
    validation criterion.
    """
    arrays = [np.asarray(a, dtype=float).ravel() for a in per_axon_distances]
    arrays = [a[np.isfinite(a)] for a in arrays]
    arrays = [a for a in arrays if a.size > 0]

    warnings_: List[str] = []
    if not arrays:
        raise ValueError("No usable 1NN distances supplied.")

    medians = np.array([float(np.median(a)) for a in arrays])
    pooled = np.concatenate(arrays)

    if len(arrays) < 5:
        warnings_.append(
            f"Only {len(arrays)} axons pooled; the paper's reference values "
            f"come from 34 axons, so agreement here is weak evidence."
        )

    kde_peak = estimate_mode(pooled, method="kde")
    hist_peak = estimate_mode(pooled, method="histogram", bins=bins)
    peak = kde_peak if peak_method == "kde" else estimate_mode(pooled, method=peak_method, bins=bins)
    if isinstance(bins, str):
        width = fd_bin_width(pooled)
    else:
        width = float(np.ptp(pooled)) / max(int(bins), 1)
    bandwidth = (float(np.std(pooled, ddof=1)) * pooled.size ** (-1.0 / 5.0)
                 if pooled.size > 1 else 0.0)

    kde_ci = hist_ci = None
    if n_bootstrap > 0 and len(arrays) >= 2:
        rng = np.random.default_rng(random_seed)
        kde_b = np.empty(int(n_bootstrap))
        hist_b = np.empty(int(n_bootstrap))
        for b in range(int(n_bootstrap)):
            pick = rng.integers(0, len(arrays), len(arrays))
            sample = np.concatenate([arrays[i] for i in pick])
            kde_b[b] = estimate_mode(sample, method="kde")
            hist_b[b] = estimate_mode(sample, method="histogram", bins=bins)
        kde_ci = (float(np.percentile(kde_b, 2.5)), float(np.percentile(kde_b, 97.5)))
        hist_ci = (float(np.percentile(hist_b, 2.5)), float(np.percentile(hist_b, 97.5)))

    return PooledNNResult(
        all_distances_nm=pooled,
        per_axon_medians_nm=medians,
        peak_nm=peak,
        peak_method=peak_method,
        median_of_medians_nm=float(np.median(medians)),
        n_axons=len(arrays),
        warnings=warnings_,
        pooled_median_nm=float(np.median(pooled)),
        pooled_quartiles_nm=(float(np.percentile(pooled, 25.0)), float(np.percentile(pooled, 75.0))),
        kde_peak_nm=kde_peak,
        kde_bandwidth_nm=bandwidth,
        kde_peak_ci_nm=kde_ci,
        hist_peak_nm=hist_peak,
        hist_bin_width_nm=width,
        hist_bins="fd" if isinstance(bins, str) else str(int(bins)),
        hist_peak_ci_nm=hist_ci,
        n_bootstrap=int(n_bootstrap) if len(arrays) >= 2 else 0,
    )
