# -*- coding: utf-8 -*-
"""
The pipeline null for the apparent-size question (b), "does a cluster's size depend on the local curvature of the
membrane?" (user's decision E5a, 2026-10-23; branch ``feat/validations-d2-e1-e5``; ``docs/apparent_size.md``
sec. 13).

Why a null at all: the clustering, the curation and the 400 nm membrane fit can make the localizations per cluster
correlate with the curvature on their own -- the same reason question (a) is read only against the pipeline null
(decision 2a). The null here is NOT a second null: it is the per-axon pipeline null of ``tools.mps_pipeline_null``
(decision G3, one null per axon), with one more readout registered in its registry (``register_readout``): every
simulated axon -- same method, parameters, curation and contour, molecules at random on the membrane, so size and
curvature are independent by construction -- is read with the same statistic as the observed axon, and the p is
the null's two-sided Monte Carlo rank p [lit: Besag & Diggle 1977; Phipson & Smyth 2010].

The statistic, :func:`nloc_kappa_r_statistic`, a pure function of an ``AxonAnalysis`` from ANY method: Spearman of
LOCALIZATIONS per kept cluster with kappa R at its centre, over the convex clusters (kappa R >= 0; the concave ones
are reported apart and never tested, decision 3a), with kappa R read from the 400 nm P-spline membrane fitted to
the clustered localizations (``tools.mps_curvature``, decision 4a). Localizations and not events, because the
replicates carry no NeNA-linked events -- the same choice as question (a) (``docs/apparent_size.md``, deviation 3);
the events version stays descriptive (``size_content_kappa_r_spearman``).

Calibration (``docs/sims/apparent_size/curvature_null_calibration.py``): under a TRUE null the share of axons with
p <= 0.05 is within the Wilson 95 % interval of 5 % (see the doc).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import importlib
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_apparent_size as asz
from tools import mps_curvature as curv

__all__ = [
    "NULL_STATISTIC", "NULL_P_READOUT", "READOUT_ROWS", "nloc_kappa_r_statistic", "nloc_kappa_r_from_arrays",
    "view_statistic", "register_with_pipeline_null",
]

NULL_STATISTIC = "size_nloc_kappa_r_spearman"
NULL_P_READOUT = "size_nloc_kappa_r_null_p"

# The per-axon readouts this module adds to tools.size_plugins.SIZE_READOUTS (same row layout: name, kind,
# description, origin, levels).
READOUT_ROWS: Tuple[Tuple[str, str, str, str, Tuple[str, ...]], ...] = (
    (NULL_STATISTIC, "value", "Question (b) as the pipeline null tests it: Spearman of LOCALIZATIONS per cluster "
     "with kappa R on the convex clusters (the replicates have no events). Read only against the pipeline null.",
     "decisions 3a, 4a, E5a", ()),
    (NULL_P_READOUT, "p_value", "Pipeline-null Monte Carlo p of size_nloc_kappa_r_spearman (the same null replicates "
     "as every other readout of the axon: same method, curation and contour). Empty when the null did not run.",
     "decision E5a, G3; [lit: Besag & Diggle 1977]; calibration [own sim: docs/sims/apparent_size/"
     "curvature_null_calibration.py]", ()),
)


def _excluded(analysis: Any) -> List[int]:
    """The labels the curation left out (bad + discarded), as ``tools.size_plugins`` reads them."""
    bad = getattr(getattr(analysis, "bad_report", None), "bad_labels", None) or set()
    dropped = getattr(analysis, "discarded_labels", None) or frozenset()
    return sorted({int(v) for v in bad} | {int(v) for v in dropped})


def nloc_kappa_r_from_arrays(n_locs: Any, kappa_r: Any) -> Optional[float]:
    """Spearman of localizations with kappa R over the convex clusters (kappa R >= 0); None below 4 clusters."""
    n = np.asarray(n_locs, dtype=np.float64).reshape(-1)
    kr = np.asarray(kappa_r, dtype=np.float64).reshape(-1)
    convex = np.isfinite(kr) & (kr >= 0.0)
    rho, _ = asz.spearman(n[convex], kr[convex])
    return rho


def _kept_counts_centroids(x: NDArray[np.float64], y: NDArray[np.float64], labels: NDArray[np.int64],
                           exclude: Sequence[int]) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    keep = asz.kept_cluster_labels(labels, exclude)
    if keep.size == 0:
        return keep, np.zeros(0), np.zeros((0, 2))
    m = np.isin(labels, keep)
    idx = np.searchsorted(keep, labels[m])
    n = np.bincount(idx, minlength=keep.size).astype(np.float64)
    cx = np.bincount(idx, weights=x[m], minlength=keep.size) / n
    cy = np.bincount(idx, weights=y[m], minlength=keep.size) / n
    return keep, n, np.column_stack([cx, cy])


def nloc_kappa_r_statistic(analysis: Any) -> Optional[float]:
    """THE statistic of question (b) the pipeline null tests (module docstring): a pure function of an analysis,
    observed or simulated, any method. None with fewer than 3 kept clusters, when the membrane cannot be fitted, or
    with fewer than 4 convex clusters."""
    x = np.asarray(analysis.x_slab, dtype=np.float64)
    y = np.asarray(analysis.y_slab, dtype=np.float64)
    labels = np.asarray(analysis.labels, dtype=np.int64)
    keep, n, cen = _kept_counts_centroids(x, y, labels, _excluded(analysis))
    if keep.size < 3:
        return None
    m = np.isin(labels, keep)
    try:
        membrane = curv.fit_curvature_membrane(x[m], y[m], cen)
    except ValueError:
        return None
    _, _, kr = membrane.at(cen)
    return nloc_kappa_r_from_arrays(n, kr)


def view_statistic(view: Any) -> Optional[float]:
    """``tools.mps_pipeline_null`` readout function (module level: picklable for its worker processes)."""
    return nloc_kappa_r_statistic(view.analysis)


def register_with_pipeline_null(replace: bool = False) -> List[str]:
    """Add the question-(b) statistic to ``tools.mps_pipeline_null``'s readout registry, so that the axon's ONE null
    (decision G3) carries ``p_values["size_nloc_kappa_r_spearman"]`` (export column ``p_mc_size_nloc_kappa_r_spearman``).
    It needs the membrane, not the null's ring: ``needs_ring=False``."""
    pn = importlib.import_module("tools.mps_pipeline_null")
    pn.register_readout(pn.ReadoutSpec(
        NULL_STATISTIC, view_statistic, "", needs_ring=False, two_sided=True,
        description="Spearman of localizations per cluster with kappa R on the convex clusters (apparent-size "
                    "question b; docs/apparent_size.md)"), replace=replace)
    return [NULL_STATISTIC]
