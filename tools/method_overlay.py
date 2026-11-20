# -*- coding: utf-8 -*-
"""
A second clustering method over the first on the axon map: what it draws and how the two partitions correspond.

The MPS analysis window shows one analysis (method A, the one in use). "Compare methods..." runs other methods on the
same slab; choosing one of its rows and "Overlay on the map" draws that method (B) over A: A's kept clusters as thin
solid outlines in the colour of A's localizations, B's as dotted outlines in their own colour, each outline the convex
hull of the cluster's localizations - where A and B part ways (one A cluster cut in two by B, two A clusters joined by
B, a cluster one of them does not have) is read off the map.

The correspondence summary does not invent a metric: it is ``tools.cluster_benchmark.molecule_metrics`` - the
benchmark's majority-matching rules, written for clusters against simulated molecules - with A's clusters in the role
of the molecules and A's unclustered localizations in the role of the background:

* one-to-one   B clusters whose majority A cluster holds >= ``TAU_EXACT`` (0.8) of the B cluster and the B cluster
               >= 0.8 of that A cluster;
* split        A clusters present (>= ``PRESENT_SHARE``, 20 % of their localizations) in two or more B clusters;
* merged       B clusters in which two or more A clusters are present;
* A unmatched  A clusters with less than ``MISSED_SHARE`` (50 %) of their localizations in any B cluster;
* B unmatched  B clusters more than ``FALSE_BG_SHARE`` (50 %) of whose localizations are in no A cluster;
* ARI          adjusted Rand index of the two labellings on the localizations both put in a cluster.

The thresholds are the benchmark's, all ad hoc (``tools/cluster_benchmark.py``). The clusters are each analysis'
KEPT clusters (after its edge curation; a curated cluster counts as unclustered), on the same localizations: the
overlay refuses two analyses of different localizations. The roles are not symmetric: "split" and "merged" are from
A's side. Nothing is computed for the analysis itself: A and B keep every number they had.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["Correspondence", "MethodOverlay", "kept_labels", "cluster_hulls", "correspondence", "build_overlay",
           "joined_outlines"]


@dataclass
class Correspondence:
    """How B's kept clusters correspond to A's (counts; see the module docstring for each rule)."""
    n_a: int
    n_b: int
    n_one_to_one: int
    n_a_split: int
    n_b_merged: int
    n_a_unmatched: int
    n_b_unmatched: int
    ari: float

    def words(self, method_a: str, method_b: str) -> str:
        ari = "-" if not np.isfinite(self.ari) else f"{self.ari:.2f}"
        return (f"{method_b} ({self.n_b} clusters) against {method_a} ({self.n_a}): {self.n_one_to_one} one-to-one; "
                f"{self.n_a_split} of {method_a}'s split by {method_b}; {self.n_b_merged} of {method_b}'s merge two or "
                f"more of {method_a}'s; {self.n_a_unmatched} of {method_a}'s without a partner; {self.n_b_unmatched} "
                f"of {method_b}'s mostly on {method_a}'s unclustered localizations; ARI {ari}")


@dataclass
class MethodOverlay:
    """Method B over the analysis shown (A). ``base`` is A itself: the overlay belongs to that analysis only."""
    method_a: str
    method_b: str
    base: Any
    other: Any
    hulls_a: List[NDArray[np.float64]] = field(default_factory=list)
    hulls_b: List[NDArray[np.float64]] = field(default_factory=list)
    summary: Optional[Correspondence] = None
    note: str = ""                       # why there is no overlay, when there is none

    @property
    def ok(self) -> bool:
        return self.summary is not None and not self.note


def kept_labels(analysis: Any) -> NDArray[np.int64]:
    """The analysis' labels with the clusters its curation removed set to -1 (its kept clusters only)."""
    labels: NDArray[np.int64] = np.array(analysis.labels, dtype=np.int64)
    bad = getattr(getattr(analysis, "bad_report", None), "bad_labels", None) or set()
    if bad:
        labels[np.isin(labels, list(bad))] = -1
    return labels


def cluster_hulls(x: NDArray[np.float64], y: NDArray[np.float64],
                  labels: NDArray[np.int64]) -> Tuple[List[NDArray[np.float64]], int]:
    """The convex hull of each cluster's localizations, closed (first vertex repeated); clusters with fewer than three
    distinct non-collinear localizations have none. Returns (hulls, how many clusters have none)."""
    from scipy.spatial import ConvexHull, QhullError
    xy = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    lab = np.asarray(labels)
    hulls: List[NDArray[np.float64]] = []
    none = 0
    for c in np.unique(lab[lab >= 0]):
        pts = np.unique(xy[lab == c], axis=0)
        if pts.shape[0] < 3:
            none += 1
            continue
        try:
            h = ConvexHull(pts)
        except (QhullError, ValueError):
            none += 1
            continue
        v = pts[h.vertices]
        hulls.append(np.vstack([v, v[:1]]))
    return hulls, none


def joined_outlines(hulls: List[NDArray[np.float64]]) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                                                                 NDArray[np.int32]]:
    """Every outline as one polyline: x, y and a connect array (0 ends an outline), for one plot item."""
    if not hulls:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int32)
    pts = np.concatenate(hulls, axis=0)
    connect = np.ones(pts.shape[0], dtype=np.int32)
    connect[np.cumsum([h.shape[0] for h in hulls]) - 1] = 0
    return pts[:, 0].copy(), pts[:, 1].copy(), connect


def correspondence(labels_a: NDArray[np.int64], labels_b: NDArray[np.int64]) -> Correspondence:
    """B's clusters against A's, by ``cluster_benchmark.molecule_metrics`` with A's clusters as the molecules."""
    from tools.cluster_benchmark import molecule_metrics
    la = np.asarray(labels_a, dtype=np.int64)
    lb = np.asarray(labels_b, dtype=np.int64)
    if la.shape != lb.shape:
        raise ValueError("the two labellings are not of the same localizations")
    ids_a = np.unique(la[la >= 0])
    if ids_a.size == 0:
        # no A cluster to match: every B cluster lies on A's unclustered localizations
        n_b0 = int(np.unique(lb[lb >= 0]).size)
        return Correspondence(n_a=0, n_b=n_b0, n_one_to_one=0, n_a_split=0, n_b_merged=0, n_a_unmatched=0,
                              n_b_unmatched=n_b0, ari=float("nan"))
    mol = np.full(la.shape, -1, dtype=np.int64)
    if ids_a.size:
        mol[la >= 0] = np.searchsorted(ids_a, la[la >= 0])
    sample = SimpleNamespace(molecule_id=mol, mol_xy=np.zeros((int(ids_a.size), 2)))
    r = molecule_metrics(lb, sample)          # type: ignore[arg-type]
    n_b = int(r["n_clusters"])
    n_a = int(r["n_mol_labelled"])

    def count(frac: Any, total: int) -> int:
        f = float(frac)
        return int(round(f * total)) if np.isfinite(f) else 0

    return Correspondence(
        n_a=n_a, n_b=n_b, n_one_to_one=count(r["exact"], n_b), n_a_split=count(r["split"], n_a),
        n_b_merged=int(r["n_merged"]), n_a_unmatched=count(r["missed"], n_a), n_b_unmatched=int(r["n_false"]),
        ari=float(r["ari"]) if n_b and n_a else float("nan"))


def build_overlay(base: Any, other: Any, method_b: str) -> MethodOverlay:
    """The overlay of ``other`` (method B) on ``base`` (the analysis shown), or one whose ``note`` says why not."""
    method_a = str(getattr(base, "cluster_method", "A"))
    ov = MethodOverlay(method_a=method_a, method_b=str(method_b), base=base, other=other)
    for name in ("x_slab", "y_slab", "labels"):
        if getattr(other, name, None) is None or getattr(base, name, None) is None:
            ov.note = f"the {method_b} run has no {name}"
            return ov
    xa, ya = np.asarray(base.x_slab, dtype=np.float64), np.asarray(base.y_slab, dtype=np.float64)
    xb, yb = np.asarray(other.x_slab, dtype=np.float64), np.asarray(other.y_slab, dtype=np.float64)
    if xa.shape != xb.shape or not (np.array_equal(xa, xb) and np.array_equal(ya, yb)):
        ov.note = "the two runs are not of the same localizations"
        return ov
    la, lb = kept_labels(base), kept_labels(other)
    ov.hulls_a, _ = cluster_hulls(xa, ya, la)
    ov.hulls_b, _ = cluster_hulls(xb, yb, lb)
    ov.summary = correspondence(la, lb)
    return ov
