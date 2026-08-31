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

The biological point is that each axon's median is remarkably constant
regardless of perimeter: larger axons add more clusters rather than
spacing them further apart.

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
from typing import List, Optional, Sequence, Tuple

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
        return NNResult(np.empty((0, k)), k, n, warnings_)

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
    for i in range(n):
        row = dist[i][~first_self[i]] if has_self[i] else dist[i][:k_eff]
        kept[i] = row[:k_eff]

    n_dup = int(np.count_nonzero(kept[:, 0] == 0.0))
    if n_dup:
        warnings_.append(
            f"{n_dup} cluster centre(s) have a nearest neighbour at exactly "
            f"0 nm, i.e. coincident centroids. Check the clustering: this "
            f"collapses the minimum inter-cluster distance used by the "
            f"randomization control."
        )

    return NNResult(kept, k_eff, n, warnings_)


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
    """Pooled 1NN distribution across many axons."""

    all_distances_nm: NDArray[np.float64]
    per_axon_medians_nm: NDArray[np.float64]
    peak_nm: float                 # mode of the pooled distribution (Fig. 4A)
    peak_method: str
    median_of_medians_nm: float    # Fig. 4B reference line
    n_axons: int
    warnings: List[str] = field(default_factory=list)


def estimate_mode(
    values: NDArray[np.float64],
    method: str = "kde",
    bins: int = 50,
    bw_method: Optional[float] = None,
) -> float:
    """
    Estimate the peak (mode) of a distribution.

    The paper reads the ~200 nm figure off the peak of a histogram, so the
    value depends on the smoothing choice. Two estimators are offered:

    * "kde" (default): Gaussian kernel density estimate, peak taken as the
      argmax of the density on a fine grid. Smooth and bin-independent,
      but sensitive to the bandwidth.
    * "histogram": centre of the tallest bin. Reproduces the paper's own
      procedure most literally, but depends on ``bins``.

    Report both when the number matters.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate the mode of an empty array.")
    if values.size == 1:
        return float(values[0])

    if method == "histogram":
        counts, edges = np.histogram(values, bins=bins)
        i = int(np.argmax(counts))
        return float((edges[i] + edges[i + 1]) / 2)

    if method == "kde":
        kde = gaussian_kde(values, bw_method=bw_method)
        grid = np.linspace(values.min(), values.max(), 4096)
        return float(grid[int(np.argmax(kde(grid)))])

    raise ValueError(f"method must be 'kde' or 'histogram', got {method!r}")


def pool_nn_across_axons(
    per_axon_distances: Sequence[NDArray[np.float64]],
    peak_method: str = "kde",
    bins: int = 50,
) -> PooledNNResult:
    """
    Combine per-axon 1NN distances into the paper's two summary statistics.

    Parameters
    ----------
    per_axon_distances : one 1-D array of 1NN distances per axon.
    peak_method : "kde" or "histogram" (see ``estimate_mode``).
    bins : histogram bins if peak_method == "histogram".

    Returns
    -------
    PooledNNResult with the pooled peak (Fig. 4A, paper ~200 nm) and the
    median of per-axon medians (Fig. 4B, paper 260 nm).
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

    peak = estimate_mode(pooled, method=peak_method, bins=bins)

    return PooledNNResult(
        all_distances_nm=pooled,
        per_axon_medians_nm=medians,
        peak_nm=peak,
        peak_method=peak_method,
        median_of_medians_nm=float(np.median(medians)),
        n_axons=len(arrays),
        warnings=warnings_,
    )
