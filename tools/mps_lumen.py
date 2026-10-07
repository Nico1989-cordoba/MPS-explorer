# -*- coding: utf-8 -*-
"""
The lumen clusters of the H-ECL rings of one axon: which to take out of the
rings before any curve is fitted, which the user should look at, and the
user's own decisions (milestone H5-D; DECISIONES D-31 to D-35).

Why this exists
---------------
Some betaII-spectrin clusters of a transverse section can sit inside the axon,
hundreds of nm from its membrane (the joint widefield rule of D-32a finds
them). Every curve fitted through the rings --
each ring's contour, the pooled membranes the arc test projects on -- is
pulled into the lumen by them, and the arc test of the column hypothesis
(H-ECL) then reads their bays as columns, by several sd on simulated axons
(D-32d). Excluding them from the matching is not enough; they must leave the
curves, i.e. the rings (``tools.mps_columns.clean_rings``), before anything
is fitted.

The rule (D-35(a), frozen 2026-09-28; ``LumenParams`` holds its numbers)
------------------------------------------------------------------------
Features on the POOLED centroids of every ring (the axon frame, x', y'):

* hd = depth inside the convex hull of the pooled centroids (+ inside);
* hop_X = the bottleneck (minimax) hop along the minimum spanning tree of
  the pooled centroids from the cluster to the outline set {hd < X nm}:
  the longest step a walk from the cluster to the outline must take, so a
  cluster in a crowd of membrane clusters (a concave membrane) hops short
  and a cluster alone in the lumen hops long, however deep either is;
* ISO_STRICT = hop_300 > 400 and hd > 400; ISO_MID = hop_250 > 250 and
  hd > 400; ISO_SENS = hop_250 > 200 and hd > 400;
* WF250 / WF0 = the joint widefield rule of the Axoplasm panel
  (``tools.mps_axoplasm``: deeper than 250 / 0 nm inside BOTH the tubulin
  mask and the dark interior of the spectrin ring); "interior usable" =
  the spectrin interior was cut at its half maximum (it closes).

REMOVE (automatic) = ISO_STRICT or (WF0 and ISO_MID) or (WF250 and a
usable interior). Veto: in an axon with a usable interior, a cluster
removed by isolation alone that the interior puts more than 250 nm OUTSIDE
it becomes DOUBTFUL. DOUBTFUL = flagged by WF0, WF250 or ISO_SENS and not
removed (the vetoed included). Without widefield images REMOVE = ISO_STRICT,
and completeness is not certified (a warning says so).

Why these and not the union the user leaned to (D-34c): widefield at margin
0 can mark membrane clusters, and the lax isolation rules remove membrane
clusters at the same arc of several rings, which opens shared gaps and
biases the arc test UP on the true curve -- the same direction as a lumen
cluster left in. So the automatic rule removes only with certainty ("sacar
solo con seguridad", Q-18) and everything else it doubts is shown to the
user. ISO_STRICT, measured on simulated axons (D-34c): false positives on
well under 1 % of the membrane clusters, more often with leak; sensitivity
low for lumen clusters shallower than ~600 nm and high beyond ~800 nm;
the arc test level unchanged on lumen-free axons.

The user's decisions (D-35(b), Q-19)
------------------------------------
``LumenDecisions`` holds, per cluster, the automatic class (kept / removed
/ doubtful, with the reasons), the manual action (none / removed /
restored) and the final state, an ordered edit log and the
``results_shown_before_edit`` flag (an edit made after a column result of
the axon was shown: the result is then labelled "edited after results were
shown" everywhere, R9/R10). Clusters are named by ``stable_cluster_key``
(ring, DBSCAN label and rounded centroid), which survives a cleaning and a
fresh ``build_rings`` of the same localizations, so saved decisions can be
re-applied (``to_rows`` / ``from_rows``). A doubtful cluster is KEPT unless
the user removes it. ``run_cleaned_analyses`` is the job of the review
window's re-run button: ``clean_rings`` with the final set (``build_rings``
is NOT re-run) and the analyses of the rings -- the arc test on the
centroid membrane (primary, D-35c), the 2D test (sensitivity) and the arc
test on the localization membrane (a diagnostic with a known conservative
bias) -- none calibrated for the axial leak yet (H5-E).

Saved decisions belong to ONE axon's ONE ring build: every row carries the
fingerprint of the axon's cluster set (``cluster_set_fingerprint``, the
sha256 of its sorted stable keys), and ``from_rows`` re-attached to a
classification refuses a table of another axon or ring build (another
fingerprint, or no saved cluster among these) instead of silently falling
back to the automatic state; when a table holds several exports of the
axon, the LAST row of each cluster is the one used. ``LumenReviewStore``
keeps each axon's decisions between sessions of the review window (keyed by
the axon's id and the rule version), so the "edited after results were
shown" flag belongs to the axon, not to one window (D-35b).

No Qt here: the window (``tools/mps_columns_window.py``) and the simulated
null (``power_columns.py simnull --lumen-clean``) use this model.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json
import math
import os
import re
import time
from datetime import datetime
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist

from tools import mps_axoplasm
from tools.mps_axoplasm import DEFAULT_SMOOTH_SIGMA_PX, AxoplasmMask, WidefieldImage
from tools.mps_columns import Cluster, ColumnsParams, RingsResult, clean_rings, load_columns_params

if TYPE_CHECKING:  # the analyses are imported at the call (run_cleaned_analyses): this module stays light
    from tools.mps_matching import AxonColumnsResult
    from tools.mps_unroll import ArcColumnsResult

__all__ = [
    "AUTO_CLASSES",
    "AUTO_DOUBTFUL",
    "AUTO_KEPT",
    "AUTO_REMOVED",
    "DEFAULT_COLUMNS_PARAMS_PATH",
    "LEAK_NOT_CALIBRATED_NOTE",
    "LUMEN_DECISION_COLUMNS",
    "LUMEN_REVIEW_SESSION_FORMAT",
    "LUMEN_RULE_VERSION",
    "MANUAL_ACTIONS",
    "MANUAL_NONE",
    "MANUAL_REMOVED",
    "MANUAL_RESTORED",
    "NO_WIDEFIELD_WARNING",
    "REASON_ISO_SENS",
    "REASON_ISO_STRICT",
    "REASON_VETO",
    "REASON_WF0",
    "REASON_WF0_ISO_MID",
    "REASON_WF250",
    "REASON_WF250_USABLE",
    "TUBULIN_SHIFT_WARNING",
    "CleanedAnalyses",
    "IsolationFlags",
    "LumenClassification",
    "LumenDecisions",
    "LumenEdit",
    "LumenParams",
    "LumenReviewStore",
    "LumenRuleResult",
    "LumenWidefieldMasks",
    "WidefieldLumenFlags",
    "apply_lumen_rule",
    "build_lumen_widefield_masks",
    "classify_lumen",
    "cluster_lab_centres",
    "cluster_set_fingerprint",
    "default_columns_params",
    "isolation_flags",
    "lumen_cluster_keys",
    "run_cleaned_analyses",
    "stable_cluster_key",
    "widefield_flags_from_depths",
    "widefield_lumen_flags",
]

# ============================================================================
# Constants
# ============================================================================

# The rule this module implements; written into every classification, every
# decision row and every result, so that a table says which rule made it.
LUMEN_RULE_VERSION = "lumen rule D-35(a) (2026-09-28)"

# Automatic classes and manual actions (D-35b). "restored" = a cluster the
# rule removed that the user kept; "removed" = a kept or doubtful cluster the
# user removed; "none" = the automatic state.
AUTO_KEPT = "kept"
AUTO_REMOVED = "removed"
AUTO_DOUBTFUL = "doubtful"
AUTO_CLASSES: Tuple[str, ...] = (AUTO_KEPT, AUTO_REMOVED, AUTO_DOUBTFUL)
MANUAL_NONE = "none"
MANUAL_REMOVED = "removed"
MANUAL_RESTORED = "restored"
MANUAL_ACTIONS: Tuple[str, ...] = (MANUAL_NONE, MANUAL_REMOVED, MANUAL_RESTORED)

# The reasons a cluster carries (``LumenRuleResult.reasons``): the removal
# terms of D-35(a) that hold for a removed cluster; the veto plus its flags
# for a vetoed one; the flags that made a doubtful one doubtful.
REASON_ISO_STRICT = "ISO_STRICT"
REASON_WF0_ISO_MID = "WF0+ISO_MID"
REASON_WF250_USABLE = "WF250+usable_interior"
REASON_VETO = "vetoed_by_interior"
REASON_WF0 = "WF0"
REASON_WF250 = "WF250"
REASON_ISO_SENS = "ISO_SENS"

# Two MST edges between identical centroids would read as "no edge" in scipy
# (a 0 in the dense matrix): they get this length instead (never met in
# practice; the research's guard, kept so that the tree stays connected).
DUPLICATE_CENTROID_DISTANCE_NM = 1e-6

NO_WIDEFIELD_WARNING = (
    "No widefield images: the lumen rule is isolation only (REMOVE = ISO_STRICT, D-35a); its completeness is not "
    "certified -- a lumen cluster shallower than ~600 nm or close to other clusters can stay (low sensitivity at "
    "400-600 nm, D-34c). Review the doubtful clusters.")
TUBULIN_SHIFT_WARNING = (
    "The tubulin image is placed with the registration shift measured on the spectrin image, as the Axoplasm panel "
    "places it: the tubulin channel's own shift was never measured (D-32a; it can sit a few pixels off), "
    "so its depths can be off by 100-250 nm.")
LEAK_NOT_CALIBRATED_NOTE = "not calibrated for axial leak (H5-E)"

# The pre-registered parameters of the column analysis (D-22), what
# ``run_cleaned_analyses`` runs with when given none.
DEFAULT_COLUMNS_PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           "config", "columns_params.yaml")

# The columns of ``LumenDecisions.to_rows`` (one row per cluster), in order.
# ``edit_orders`` lists the orders of the log entries that changed the
# cluster (";"-separated), ``edit_details`` each of them as
# order|final state|after results shown (0/1)|via, so that ``from_rows``
# rebuilds the log exactly.
LUMEN_DECISION_COLUMNS: Tuple[str, ...] = (
    "stable_key", "ring", "position", "label", "x_nm", "y_nm", "auto_class", "auto_reasons", "manual_action",
    "final_removed", "edit_orders", "edit_details", "results_shown", "results_shown_before_edit", "rule_version",
    "has_widefield", "interior_usable", "registration", "hull_depth_nm", "hop_strict_nm", "hop_mid_nm",
    "depth_tubulin_nm", "depth_spectrin_nm", "n_locs", "n_events", "abs_dz_ring_nm",
    # Which cluster set (one axon, one ring build) the row belongs to: cluster_set_fingerprint of every stable key.
    "cluster_set_sha",
)
_DECISION_REQUIRED_ON_LOAD: Tuple[str, ...] = ("stable_key", "auto_class", "final_removed")
_STABLE_KEY_RE = re.compile(r"^r(-?\d+)_l(-?\d+)_x(-?\d+)_y(-?\d+)$")
# The file format of LumenReviewStore (one JSON per axon and rule version).
LUMEN_REVIEW_SESSION_FORMAT = "mps lumen review session v1"
_FINGERPRINT_HEX = 16


def cluster_set_fingerprint(stable_keys: Iterable[str]) -> str:
    """
    The fingerprint of one axon's cluster set: the first 16 hex digits of
    the sha256 of its stable keys, sorted and joined by newlines. Two ring
    builds of the same localizations with the same parameters give the same
    fingerprint; another axon, or the same axon built otherwise (another
    ROI edge, other parameters), gives another one. Saved decisions carry it
    (``LUMEN_DECISION_COLUMNS``: ``cluster_set_sha``) so that a table is
    never applied to clusters it was not made on.
    """
    text = "\n".join(sorted(str(k) for k in stable_keys))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_FINGERPRINT_HEX]


# ============================================================================
# Parameters
# ============================================================================

@dataclass(frozen=True)
class LumenParams:
    """
    The numbers of the frozen lumen rule, D-35(a); the defaults ARE the rule
    (changing one is a deviation to be recorded in DECISIONES).

    ``iso_*_outline_nm`` / ``iso_*_hop_nm``: ISO_STRICT = hop to {hd < 300}
    > 400 nm, ISO_MID = hop to {hd < 250} > 250 nm, ISO_SENS = hop to
    {hd < 250} > 200 nm, each also deeper than ``iso_min_hull_depth_nm``
    (400 nm) inside the pooled hull. ``wf_margin_nm`` (250) and
    ``wf_low_margin_nm`` (0): the widefield joint rule's margins of WF250
    and WF0 (strictly deeper, in both images). ``veto_outside_nm`` (250): a
    cluster removed by isolation alone that a usable spectrin interior puts
    more than this OUTSIDE it becomes doubtful.
    """

    iso_strict_outline_nm: float = 300.0
    iso_strict_hop_nm: float = 400.0
    iso_mid_outline_nm: float = 250.0
    iso_mid_hop_nm: float = 250.0
    iso_sens_outline_nm: float = 250.0
    iso_sens_hop_nm: float = 200.0
    iso_min_hull_depth_nm: float = 400.0
    wf_margin_nm: float = 250.0
    wf_low_margin_nm: float = 0.0
    veto_outside_nm: float = 250.0

    def __post_init__(self) -> None:
        """Every value finite; the isolation thresholds and the veto distance non-negative."""
        for f in _field_names(self):
            v = float(getattr(self, f))
            if not math.isfinite(v):
                raise ValueError(f"LumenParams.{f} must be finite, got {getattr(self, f)!r}")
            if f.startswith("iso_") or f == "veto_outside_nm":
                if v < 0.0:
                    raise ValueError(f"LumenParams.{f} must be >= 0, got {v}")
            object.__setattr__(self, f, v)

    def as_dict(self) -> Dict[str, float]:
        """The values, for provenance."""
        return {f: float(getattr(self, f)) for f in _field_names(self)}


def _field_names(obj: Any) -> List[str]:
    """The field names of a dataclass instance or class, in order."""
    return [f.name for f in dataclasses.fields(obj)]


# ============================================================================
# Which cluster is which
# ============================================================================

def lumen_cluster_keys(res: RingsResult) -> List[Tuple[int, int]]:
    """
    (Ring.index, position in Ring.clusters) of every cluster of ``res``: the
    rings by index, then each ring's clusters in order. Every per-cluster
    array of this module is in this order.
    """
    return [(int(r.index), j) for r in sorted(res.rings, key=lambda q: int(q.index)) for j in range(len(r.clusters))]


def stable_cluster_key(ring_index: int, cluster: Cluster) -> str:
    """
    A name of a cluster that survives a re-run: ``r<ring>_l<label>_x<x'>_y<y'>``
    (ring index, DBSCAN label within the ring, the centroid's x', y' in the
    axon frame rounded to the nm). A cleaning keeps every surviving cluster's
    label and centroid, and ``build_rings`` is deterministic, so a cleaning or
    a fresh build of the same localizations gives the same keys; the
    centroid tells the clusters of two different axons apart.
    """
    c = np.asarray(cluster.centroid_nm, dtype=np.float64).reshape(2)
    return f"r{int(ring_index)}_l{int(cluster.label)}_x{float(c[0]):.0f}_y{float(c[1]):.0f}"


def _clusters_in_order(res: RingsResult) -> List[Tuple[int, int, Cluster]]:
    return [(int(r.index), j, cl) for r in sorted(res.rings, key=lambda q: int(q.index)) for j, cl in enumerate(r.clusters)]


def _centroids(res: RingsResult) -> NDArray[np.float64]:
    rows = [np.asarray(cl.centroid_nm, dtype=np.float64).reshape(2) for _k, _j, cl in _clusters_in_order(res)]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


# ============================================================================
# Isolation (the super-resolved data alone)
# ============================================================================

@dataclass
class IsolationFlags:
    """
    The isolation features and flags of every cluster (``lumen_cluster_keys``
    order): ``centroids_nm`` the pooled centroids (x', y'), ``hull_depth_nm``
    their depth inside the pooled convex hull (+ inside), ``hop_strict_nm`` /
    ``hop_mid_nm`` / ``hop_sens_nm`` the bottleneck hop to the outline sets of
    ISO_STRICT / ISO_MID / ISO_SENS, and the three flags (D-35a). ``params``
    are the numbers used.
    """

    keys: List[Tuple[int, int]]
    centroids_nm: NDArray[np.float64]
    hull_depth_nm: NDArray[np.float64]
    hop_strict_nm: NDArray[np.float64]
    hop_mid_nm: NDArray[np.float64]
    hop_sens_nm: NDArray[np.float64]
    iso_strict: NDArray[np.bool_]
    iso_mid: NDArray[np.bool_]
    iso_sens: NDArray[np.bool_]
    warnings: List[str] = field(default_factory=list)
    params: LumenParams = field(default_factory=LumenParams)


def _hull_depth_nm(points: NDArray[np.float64], query: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
    """Depth of ``query`` inside the convex hull of ``points`` (+ inside, the largest facet equation's value negated);
    None when there is no hull (fewer than three points, or collinear ones)."""
    if points.shape[0] < 3:
        return None
    try:
        eq = ConvexHull(points).equations
    except (QhullError, ValueError):
        return None
    return np.asarray(-(query @ eq[:, :2].T + eq[:, 2]).max(axis=1), dtype=np.float64)


def _pairwise_nm(P: NDArray[np.float64]) -> NDArray[np.float64]:
    """Euclidean distances between the pooled centroids; identical ones at DUPLICATE_CENTROID_DISTANCE_NM."""
    D = np.asarray(cdist(P, P), dtype=np.float64)
    iu = np.triu_indices(D.shape[0], 1)
    z = D[iu] <= 0.0
    if z.any():
        D[iu[0][z], iu[1][z]] = DUPLICATE_CENTROID_DISTANCE_NM
        D[iu[1][z], iu[0][z]] = DUPLICATE_CENTROID_DISTANCE_NM
    return D


def _mst_dense(D: NDArray[np.float64]) -> NDArray[np.float64]:
    """The minimum spanning tree of the complete graph ``D`` as a symmetric dense matrix (0 = no edge)."""
    m = np.asarray(minimum_spanning_tree(D).toarray(), dtype=np.float64)
    return np.asarray(np.maximum(m, m.T), dtype=np.float64)


def _minimax_to_set(mst: NDArray[np.float64], sources: NDArray[np.bool_]) -> NDArray[np.float64]:
    """The bottleneck distance from every node to the ``sources`` along the tree (the minimax path of the complete
    graph): a Dijkstra on max instead of sum; inf where no source is reachable."""
    n = mst.shape[0]
    best = np.full(n, np.inf)
    adj = [np.flatnonzero(mst[i] > 0) for i in range(n)]
    heap: List[Tuple[float, int]] = []
    for p in np.flatnonzero(sources):
        best[p] = 0.0
        heapq.heappush(heap, (0.0, int(p)))
    while heap:
        b, u = heapq.heappop(heap)
        if b > best[u]:
            continue
        for v in adj[u]:
            nb = max(b, float(mst[u, v]))
            if nb < best[v]:
                best[v] = nb
                heapq.heappush(heap, (nb, int(v)))
    return best


def isolation_flags(res: RingsResult, params: LumenParams = LumenParams()) -> IsolationFlags:
    """
    The isolation features of every cluster of ``res`` and the ISO_* flags of
    D-35(a) (the port of the research's isolation-feature script
    and its frozen rule terms):

    * hd = depth inside the convex hull of the POOLED centroids of every
      cluster of every ring (+ inside; 0 without a hull);
    * hop to X = the bottleneck (minimax) hop along the minimum spanning tree
      of the pooled centroids (Euclidean; identical centroids at
      ``DUPLICATE_CENTROID_DISTANCE_NM``) to the outline set {hd < X}: 0 for
      an outline cluster;
    * ISO_STRICT / ISO_MID / ISO_SENS = hop > H and hd >
      ``params.iso_min_hull_depth_nm``, with (X, H) = (300, 400), (250, 250),
      (250, 200) nm by default.

    With fewer than three clusters (or collinear ones) there is no hull: hd
    and the hops are 0, no flag, and a warning says so. Deterministic.
    """
    keys = lumen_cluster_keys(res)
    P = _centroids(res)
    n = int(P.shape[0])
    warnings_: List[str] = []
    if not np.isfinite(P).all():
        raise ValueError("isolation_flags: a cluster centroid is not finite")
    zeros = np.zeros(n, dtype=np.float64)
    no = np.zeros(n, dtype=bool)
    if n < 3:
        warnings_.append(f"isolation: {n} cluster(s) in the rings: no pooled hull, no isolation flag")
        return IsolationFlags(keys=keys, centroids_nm=P, hull_depth_nm=zeros, hop_strict_nm=zeros.copy(),
                              hop_mid_nm=zeros.copy(), hop_sens_nm=zeros.copy(), iso_strict=no, iso_mid=no.copy(),
                              iso_sens=no.copy(), warnings=warnings_, params=params)
    hd = _hull_depth_nm(P, P)
    if hd is None:
        warnings_.append("isolation: the pooled centroids are collinear (no hull): hull depths 0, no isolation flag")
        hd = zeros.copy()
    mst = _mst_dense(_pairwise_nm(P))
    hops: Dict[float, NDArray[np.float64]] = {}

    def hop_to(outline_nm: float) -> NDArray[np.float64]:
        x = float(outline_nm)
        if x not in hops:
            hops[x] = _minimax_to_set(mst, hd < x)
        return np.array(hops[x], dtype=np.float64)

    hop_strict = hop_to(params.iso_strict_outline_nm)
    hop_mid = hop_to(params.iso_mid_outline_nm)
    hop_sens = hop_to(params.iso_sens_outline_nm)
    deep = hd > params.iso_min_hull_depth_nm
    return IsolationFlags(
        keys=keys, centroids_nm=P, hull_depth_nm=np.asarray(hd, dtype=np.float64), hop_strict_nm=hop_strict,
        hop_mid_nm=hop_mid, hop_sens_nm=hop_sens,
        iso_strict=np.asarray((hop_strict > params.iso_strict_hop_nm) & deep, dtype=bool),
        iso_mid=np.asarray((hop_mid > params.iso_mid_hop_nm) & deep, dtype=bool),
        iso_sens=np.asarray((hop_sens > params.iso_sens_hop_nm) & deep, dtype=bool),
        warnings=warnings_, params=params)


# ============================================================================
# Widefield (the Axoplasm panel's masks)
# ============================================================================

@dataclass
class WidefieldLumenFlags:
    """
    Every cluster's place in the two widefield images (``lumen_cluster_keys``
    order): its lab centre (``centroids_lab_nm``, the mean laboratory x, y of
    its localizations; NaN when the depths were given without them), its
    signed depth in the tubulin mask and in the spectrin ring's interior (+
    inside, NaN off an analysed region), WF250 / WF0 (deeper than
    ``params.wf_margin_nm`` / ``params.wf_low_margin_nm`` in BOTH), whether
    the interior is usable (cut at its half maximum; ``interior_source`` is
    the cut's source), the registration used, and the warnings. ``masks`` are
    the masks themselves when they were built here.
    """

    keys: List[Tuple[int, int]]
    centroids_lab_nm: NDArray[np.float64]
    depth_tubulin_nm: NDArray[np.float64]
    depth_spectrin_nm: NDArray[np.float64]
    wf250: NDArray[np.bool_]
    wf0: NDArray[np.bool_]
    interior_usable: bool
    interior_source: str
    registration: str
    warnings: List[str] = field(default_factory=list)
    params: LumenParams = field(default_factory=LumenParams)
    masks: Optional["LumenWidefieldMasks"] = None


def _joint_flags(dt: NDArray[np.float64], ds: NDArray[np.float64],
                 params: LumenParams) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(WF250, WF0): strictly deeper than the margin in both images; NaN (off a region) is never inside."""
    with np.errstate(invalid="ignore"):
        wf250 = np.asarray((dt > params.wf_margin_nm) & (ds > params.wf_margin_nm), dtype=bool)
        wf0 = np.asarray((dt > params.wf_low_margin_nm) & (ds > params.wf_low_margin_nm), dtype=bool)
    return wf250, wf0


def _depth_array(values: Any, n: int, what: str) -> NDArray[np.float64]:
    if values is None:
        raise ValueError(f"{what}: no depths given")
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    if out.size != n:
        raise ValueError(f"{what}: {out.size} values for the {n} clusters of the rings")
    return out


def widefield_flags_from_depths(
    res: RingsResult,
    depth_tubulin_nm: Any,
    depth_spectrin_nm: Any,
    *,
    interior_usable: bool,
    params: LumenParams = LumenParams(),
    centroids_lab_nm: Optional[Any] = None,
    registration: str = "",
    interior_source: str = "given",
) -> WidefieldLumenFlags:
    """
    The widefield flags from depths already measured (one per cluster, in
    ``lumen_cluster_keys`` order; NaN = off the images): the depth-to-flag
    layer alone -- how a simulated axon, which lives in the observed axon's
    frame, is mapped onto the observed axon's masks, and how a harness
    emulates a known label. ValueError when a depth vector or the lab
    centres have the wrong length.
    """
    keys = lumen_cluster_keys(res)
    n = len(keys)
    dt = _depth_array(depth_tubulin_nm, n, "widefield_flags_from_depths: depth_tubulin_nm")
    ds = _depth_array(depth_spectrin_nm, n, "widefield_flags_from_depths: depth_spectrin_nm")
    if centroids_lab_nm is None:
        C = np.full((n, 2), np.nan)
    else:
        C = np.asarray(centroids_lab_nm, dtype=np.float64).reshape(-1, 2)
        if C.shape[0] != n:
            raise ValueError(f"widefield_flags_from_depths: {C.shape[0]} lab centres for the {n} clusters of the rings")
    wf250, wf0 = _joint_flags(dt, ds, params)
    warnings_: List[str] = []
    n_off = int(np.count_nonzero(~(np.isfinite(dt) & np.isfinite(ds))))
    if n_off:
        warnings_.append(f"widefield: {n_off} of {n} cluster(s) have no depth in one of the images (off its region): no "
                         "widefield flag for them")
    return WidefieldLumenFlags(keys=keys, centroids_lab_nm=C, depth_tubulin_nm=dt, depth_spectrin_nm=ds, wf250=wf250, wf0=wf0,
                               interior_usable=bool(interior_usable), interior_source=str(interior_source),
                               registration=str(registration), warnings=warnings_, params=params)


@dataclass
class LumenWidefieldMasks:
    """
    The two masks of one axon as the Axoplasm panel builds them, and how the
    laboratory frame maps onto each image (column = x / pixel + camera offset
    + shift; the SAME shift for both images, the spectrin one's):
    ``tubulin_mask`` (Otsu on the smoothed region unless a threshold was
    given) and ``spectrin_interior`` (``build_ring_interior``), both
    ``tools.mps_axoplasm.AxoplasmMask``.
    """

    tubulin_mask: AxoplasmMask
    spectrin_interior: AxoplasmMask
    pixel_size_nm: float
    shift_px: Tuple[float, float]
    tubulin_offset_px: Tuple[float, float]
    spectrin_offset_px: Tuple[float, float]
    registration: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def interior_source(self) -> str:
        return str(self.spectrin_interior.threshold_source)

    @property
    def interior_usable(self) -> bool:
        """The interior closes: cut at its half maximum, not at a spill point (D-35a)."""
        return self.interior_source == "half maximum"

    def to_image(self, which: str, x_lab_nm: Any, y_lab_nm: Any) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(column, row) in the tubulin (``which`` "tubulin") or spectrin ("spectrin") image of laboratory x, y."""
        off = self.tubulin_offset_px if which == "tubulin" else self.spectrin_offset_px
        if which not in ("tubulin", "spectrin"):
            raise ValueError(f"to_image: which must be 'tubulin' or 'spectrin', got {which!r}")
        return _to_image(x_lab_nm, y_lab_nm, self.pixel_size_nm, off, self.shift_px)

    def depths_at(self, x_lab_nm: Any, y_lab_nm: Any) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Signed depths (nm, + inside, NaN off a region) of laboratory positions in the tubulin mask and in the
        spectrin interior."""
        tcol, trow = self.to_image("tubulin", x_lab_nm, y_lab_nm)
        scol, srow = self.to_image("spectrin", x_lab_nm, y_lab_nm)
        return (np.asarray(self.tubulin_mask.distance_at(tcol, trow), dtype=np.float64),
                np.asarray(self.spectrin_interior.distance_at(scol, srow), dtype=np.float64))


def _to_image(x_nm: Any, y_nm: Any, pixel_nm: float, off: Tuple[float, float],
              shift: Tuple[float, float]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The GUI's mapping (``tools.mps_axoplasm_window``'s ``_coordinates_in``): x / pixel + camera offset + shift."""
    sx, sy = float(shift[0]), float(shift[1])
    return (np.asarray(x_nm, dtype=np.float64) / pixel_nm + off[0] + sx,
            np.asarray(y_nm, dtype=np.float64) / pixel_nm + off[1] + sy)


def _image_array(img: Union[WidefieldImage, NDArray[np.float64]], what: str) -> NDArray[np.float64]:
    arr = np.asarray(getattr(img, "image", img), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{what}: expected a 2D image, got shape {arr.shape}")
    return arr


def _pair(values: Sequence[float], what: str) -> Tuple[float, float]:
    v = [float(a) for a in values]
    if len(v) != 2 or not all(math.isfinite(a) for a in v):
        raise ValueError(f"{what}: expected two finite numbers, got {values!r}")
    return v[0], v[1]


def build_lumen_widefield_masks(
    x_sel_lab_nm: Any,
    y_sel_lab_nm: Any,
    tubulin: Union[WidefieldImage, NDArray[np.float64]],
    spectrin: Union[WidefieldImage, NDArray[np.float64]],
    *,
    ring_level_lab_nm: Any,
    pixel_size_nm: float,
    shift_px: Sequence[float],
    tubulin_offset_px: Sequence[float] = (0.0, 0.0),
    spectrin_offset_px: Sequence[float] = (0.0, 0.0),
    tubulin_smooth_sigma_px: float = DEFAULT_SMOOTH_SIGMA_PX,
    tubulin_threshold: Optional[float] = None,
    registration: str = "",
) -> LumenWidefieldMasks:
    """
    The tubulin mask and the spectrin ring's interior of one axon, exactly as
    the Axoplasm panel builds them (``_rebuild`` / ``_find_anchored``; the
    research's bit-identical copy ``axon_classification``): each image's camera offset plus the measured
    shift -- the SPECTRIN image's shift for both, with a warning
    (``TUBULIN_SHIFT_WARNING``); ``axon_centre`` of the selection's
    localizations (``x_sel_lab_nm``, ``y_sel_lab_nm``: what the panel sees)
    in each image; ``build_mask`` (Otsu on the region smoothed by
    ``tubulin_smooth_sigma_px`` unless ``tubulin_threshold``); and
    ``build_ring_interior`` with the ring level read at ``ring_level_lab_nm``
    ((m, 2) laboratory centres: the panel uses the 2D analysis' kept cluster
    centres). The masks' own warnings are kept, prefixed.
    """
    px = float(pixel_size_nm)
    if not (math.isfinite(px) and px > 0.0):
        raise ValueError(f"build_lumen_widefield_masks: pixel_size_nm must be positive and finite, got {pixel_size_nm}")
    shift = _pair(shift_px, "shift_px")
    off_t = _pair(tubulin_offset_px, "tubulin_offset_px")
    off_s = _pair(spectrin_offset_px, "spectrin_offset_px")
    xs = np.asarray(x_sel_lab_nm, dtype=np.float64).reshape(-1)
    ys = np.asarray(y_sel_lab_nm, dtype=np.float64).reshape(-1)
    if xs.size != ys.size or xs.size == 0:
        raise ValueError(f"build_lumen_widefield_masks: {xs.size} x and {ys.size} y values for the selection")
    tub_img = _image_array(tubulin, "tubulin")
    spec_img = _image_array(spectrin, "spectrin")
    warnings_: List[str] = [TUBULIN_SHIFT_WARNING]
    for img, name in ((tubulin, "tubulin"), (spectrin, "spectrin")):
        img_px = getattr(img, "pixel_size_nm", None)
        if img_px is not None and math.isfinite(float(img_px)) and abs(float(img_px) - px) > 0.01 * px:
            warnings_.append(f"the {name} image records a pixel of {float(img_px):g} nm against the localizations' {px:g} nm; "
                             "it is placed with the localizations' (as the panel places it)")
    col_t, row_t = _to_image(xs, ys, px, off_t, shift)
    ct, rt, et = mps_axoplasm.axon_centre(col_t, row_t)
    tmask = mps_axoplasm.build_mask(tub_img, ct, rt, px, reach_px=et, threshold=tubulin_threshold,
                                    smooth_sigma_px=float(tubulin_smooth_sigma_px))
    col_s, row_s = _to_image(xs, ys, px, off_s, shift)
    cs, rs, es = mps_axoplasm.axon_centre(col_s, row_s)
    Ci = np.asarray(ring_level_lab_nm, dtype=np.float64).reshape(-1, 2)
    icol, irow = _to_image(Ci[:, 0], Ci[:, 1], px, off_s, shift)
    interior = mps_axoplasm.build_ring_interior(spec_img, cs, rs, px, ring_col=icol, ring_row=irow, reach_px=es)
    warnings_.extend(f"tubulin mask: {w}" for w in tmask.warnings)
    warnings_.extend(f"spectrin interior: {w}" for w in interior.warnings)
    return LumenWidefieldMasks(tubulin_mask=tmask, spectrin_interior=interior, pixel_size_nm=px, shift_px=shift,
                               tubulin_offset_px=off_t, spectrin_offset_px=off_s, registration=str(registration),
                               warnings=warnings_)


def cluster_lab_centres(res: RingsResult, x_lab_nm: Any, y_lab_nm: Any) -> NDArray[np.float64]:
    """
    The laboratory centre of every cluster (``lumen_cluster_keys`` order):
    the mean of the INPUT x, y (the arrays ``build_rings`` received, in the
    laboratory frame the widefield images are registered in) of its
    localizations (``Cluster.loc_index``). ValueError when the arrays do not
    cover every localization ``build_rings`` received.
    """
    x = np.asarray(x_lab_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_lab_nm, dtype=np.float64).reshape(-1)
    n_input = int(np.asarray(res.x_p).size)
    if x.size != n_input or y.size != n_input:
        raise ValueError(f"cluster_lab_centres: {x.size} x and {y.size} y values for the {n_input} localizations "
                         "build_rings received")
    rows = []
    for _k, _j, cl in _clusters_in_order(res):
        idx = np.asarray(cl.loc_index, dtype=np.intp)
        rows.append([float(np.mean(x[idx])), float(np.mean(y[idx]))])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def widefield_lumen_flags(
    res: RingsResult,
    x_lab_nm: Any,
    y_lab_nm: Any,
    tubulin: Union[WidefieldImage, NDArray[np.float64]],
    spectrin: Union[WidefieldImage, NDArray[np.float64]],
    *,
    pixel_size_nm: float,
    shift_px: Sequence[float],
    tubulin_offset_px: Sequence[float] = (0.0, 0.0),
    spectrin_offset_px: Sequence[float] = (0.0, 0.0),
    interior_centroids_lab_nm: Optional[Any] = None,
    selection: Optional[Any] = None,
    tubulin_smooth_sigma_px: float = DEFAULT_SMOOTH_SIGMA_PX,
    tubulin_threshold: Optional[float] = None,
    params: LumenParams = LumenParams(),
    registration: str = "",
) -> WidefieldLumenFlags:
    """
    The widefield flags of every ring cluster, with the Axoplasm panel's own
    masks (the port of the research's ``axon_classification``,
    which reproduces the panel bit for bit, D-32a).

    ``x_lab_nm``, ``y_lab_nm``: the laboratory x, y of EVERY localization
    ``build_rings`` received. ``tubulin``, ``spectrin``: the widefield images
    (``tools.mps_axoplasm.WidefieldImage`` or 2D arrays), placed by the panel's
    mapping (column = x / pixel + camera offset + ``shift_px``) with each
    image's camera offset and the SAME shift -- the one measured on the
    spectrin image; the tubulin image's own was never measured, and a warning
    says so. ``selection`` (a boolean mask or indices of those
    localizations; default all): the localizations the panel sees, whose
    median centre finds the axon in each image. The ring level of the
    spectrin interior is read at ``interior_centroids_lab_nm`` -- the panel
    reads it at the 2D analysis' kept cluster centres; None uses the ring
    clusters' own lab centres, with a warning naming the 2D analysis. A
    cluster's lab centre is the mean lab x, y of its localizations
    (``cluster_lab_centres``); its depths are read there.
    """
    keys = lumen_cluster_keys(res)
    C = cluster_lab_centres(res, x_lab_nm, y_lab_nm)
    x = np.asarray(x_lab_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_lab_nm, dtype=np.float64).reshape(-1)
    xs, ys = (x, y) if selection is None else (x[np.asarray(selection)], y[np.asarray(selection)])
    warnings_: List[str] = []
    if interior_centroids_lab_nm is None:
        ring_level = C
        warnings_.append("The spectrin interior's ring level was read at the ring clusters' own lab centres; the Axoplasm "
                         "panel reads it at the kept cluster centres of the 2D analysis (pass interior_centroids_lab_nm "
                         "to reproduce the panel).")
    else:
        ring_level = np.asarray(interior_centroids_lab_nm, dtype=np.float64).reshape(-1, 2)
    masks = build_lumen_widefield_masks(xs, ys, tubulin, spectrin, ring_level_lab_nm=ring_level, pixel_size_nm=pixel_size_nm,
                                        shift_px=shift_px, tubulin_offset_px=tubulin_offset_px,
                                        spectrin_offset_px=spectrin_offset_px, tubulin_smooth_sigma_px=tubulin_smooth_sigma_px,
                                        tubulin_threshold=tubulin_threshold, registration=registration)
    dt, ds = masks.depths_at(C[:, 0], C[:, 1])
    out = widefield_flags_from_depths(res, dt, ds, interior_usable=masks.interior_usable, params=params, centroids_lab_nm=C,
                                      registration=registration, interior_source=masks.interior_source)
    if not masks.interior_usable:
        warnings_.append(f"The spectrin interior is cut at its {masks.interior_source}, not at its half maximum: it is not "
                         "usable, so the WF250 removal term and the veto are off for this axon (D-35a).")
    out.warnings = list(masks.warnings) + warnings_ + list(out.warnings)
    out.masks = masks
    assert [tuple(k) for k in out.keys] == keys
    return out


# ============================================================================
# The rule
# ============================================================================

@dataclass
class LumenRuleResult:
    """D-35(a) applied to per-cluster flags and depths: ``remove`` (automatic
    removal), ``doubtful``, ``vetoed`` (removed by isolation alone, put
    outside by a usable interior: doubtful instead), the widefield flags it
    read, and the reasons of every cluster (tuples of the REASON_* strings)."""

    remove: NDArray[np.bool_]
    doubtful: NDArray[np.bool_]
    vetoed: NDArray[np.bool_]
    wf250: NDArray[np.bool_]
    wf0: NDArray[np.bool_]
    reasons: List[Tuple[str, ...]]


def _bool_vector(values: Any, n: Optional[int], what: str) -> NDArray[np.bool_]:
    out = np.asarray(values, dtype=bool).reshape(-1)
    if n is not None and out.size != n:
        raise ValueError(f"apply_lumen_rule: {what} has {out.size} values, expected {n}")
    return out


def apply_lumen_rule(
    iso_strict: Any,
    iso_mid: Any,
    iso_sens: Any,
    depth_tubulin_nm: Any,
    depth_spectrin_nm: Any,
    *,
    interior_usable: bool,
    has_widefield: bool,
    params: LumenParams = LumenParams(),
) -> LumenRuleResult:
    """
    The frozen rule of D-35(a), literally:

    * WF250 / WF0 = deeper than ``params.wf_margin_nm`` / ``wf_low_margin_nm``
      in BOTH images (NaN = off a region = not inside);
    * REMOVE = ISO_STRICT or (WF0 and ISO_MID) or (WF250 and a usable
      interior);
    * veto: with a usable interior, a cluster removed by ISO_STRICT ALONE
      (neither widefield term holds) whose spectrin depth is below
      -``params.veto_outside_nm`` (more than 250 nm outside the interior) is
      not removed but DOUBTFUL;
    * DOUBTFUL = (WF0 or WF250 or ISO_SENS) and not removed -- the vetoed
      included ("incluye los vetados"; they are ISO_SENS anyway under the
      D-35 numbers, because hop_250 >= hop_300);
    * without widefield (``has_widefield`` False; the depths are ignored):
      REMOVE = ISO_STRICT, DOUBTFUL = ISO_SENS and not removed.

    Reasons: a removed cluster carries the removal terms that hold
    (``REASON_ISO_STRICT``, ``REASON_WF0_ISO_MID``, ``REASON_WF250_USABLE``);
    a vetoed one ``REASON_VETO``, ``REASON_ISO_STRICT`` and the flags that
    hold; a doubtful one the flags that hold (``REASON_WF0``,
    ``REASON_WF250``, ``REASON_ISO_SENS``); a kept one none. ValueError on
    vectors of different lengths.
    """
    s = _bool_vector(iso_strict, None, "iso_strict")
    n = int(s.size)
    m = _bool_vector(iso_mid, n, "iso_mid")
    se = _bool_vector(iso_sens, n, "iso_sens")
    usable = bool(interior_usable) and bool(has_widefield)
    if has_widefield:
        dt = _depth_array(depth_tubulin_nm, n, "apply_lumen_rule: depth_tubulin_nm")
        ds = _depth_array(depth_spectrin_nm, n, "apply_lumen_rule: depth_spectrin_nm")
        wf250, wf0 = _joint_flags(dt, ds, params)
        with np.errstate(invalid="ignore"):
            outside = np.asarray(ds < -params.veto_outside_nm, dtype=bool)
    else:
        wf250 = np.zeros(n, dtype=bool)
        wf0 = np.zeros(n, dtype=bool)
        outside = np.zeros(n, dtype=bool)
    term_mid = wf0 & m
    term_wf250 = wf250 & usable
    wf_part = term_mid | term_wf250
    vetoed = np.asarray(usable & s & ~wf_part & outside, dtype=bool)
    remove = np.asarray((s | wf_part) & ~vetoed, dtype=bool)
    doubtful = np.asarray((wf0 | wf250 | se | vetoed) & ~remove, dtype=bool)
    reasons: List[Tuple[str, ...]] = []
    for i in range(n):
        flags = tuple(name for name, on in ((REASON_WF0, wf0[i]), (REASON_WF250, wf250[i]), (REASON_ISO_SENS, se[i])) if on)
        if remove[i]:
            reasons.append(tuple(name for name, on in ((REASON_ISO_STRICT, s[i]), (REASON_WF0_ISO_MID, term_mid[i]),
                                                       (REASON_WF250_USABLE, term_wf250[i])) if on))
        elif vetoed[i]:
            reasons.append((REASON_VETO, REASON_ISO_STRICT) + flags)
        elif doubtful[i]:
            reasons.append(flags)
        else:
            reasons.append(())
    return LumenRuleResult(remove=remove, doubtful=doubtful, vetoed=vetoed, wf250=np.asarray(wf250, dtype=bool),
                           wf0=np.asarray(wf0, dtype=bool), reasons=reasons)


# ============================================================================
# The classification of one axon
# ============================================================================

@dataclass
class LumenClassification:
    """
    The lumen classification of every cluster of one axon's rings
    (``classify_lumen``; ``lumen_cluster_keys`` order): the keys and stable
    keys, the centroids (axon frame and laboratory), the isolation features
    and flags, the widefield depths and flags (NaN / False without images),
    the automatic decision (``remove``, ``doubtful``, ``vetoed``, the
    ``reasons``), the descriptors the review window lists (``n_locs``,
    ``n_events`` as float with NaN for None, ``abs_dz_ring_nm`` = |mean z' of
    the cluster - its ring's centre|), the registration the images were
    placed with, the rule's version and numbers, and the warnings. ``ring``
    and ``labels`` are each cluster's ring index and DBSCAN label.
    """

    keys: List[Tuple[int, int]]
    stable_keys: List[str]
    centroids_nm: NDArray[np.float64]
    centroids_lab_nm: NDArray[np.float64]
    hull_depth_nm: NDArray[np.float64]
    hop_strict_nm: NDArray[np.float64]
    hop_mid_nm: NDArray[np.float64]
    hop_sens_nm: NDArray[np.float64]
    iso_strict: NDArray[np.bool_]
    iso_mid: NDArray[np.bool_]
    iso_sens: NDArray[np.bool_]
    depth_tubulin_nm: NDArray[np.float64]
    depth_spectrin_nm: NDArray[np.float64]
    wf250: NDArray[np.bool_]
    wf0: NDArray[np.bool_]
    interior_usable: bool
    has_widefield: bool
    vetoed: NDArray[np.bool_]
    remove: NDArray[np.bool_]
    doubtful: NDArray[np.bool_]
    reasons: List[Tuple[str, ...]]
    n_locs: NDArray[np.int64]
    n_events: NDArray[np.float64]
    abs_dz_ring_nm: NDArray[np.float64]
    registration: str
    rule_version: str
    params: LumenParams
    warnings: List[str] = field(default_factory=list)
    ring: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    labels: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    interior_source: str = ""
    source_name: str = ""

    @property
    def auto_classes(self) -> List[str]:
        """The automatic class of every cluster: AUTO_REMOVED, AUTO_DOUBTFUL or AUTO_KEPT."""
        return [AUTO_REMOVED if bool(r) else (AUTO_DOUBTFUL if bool(d) else AUTO_KEPT) for r, d in zip(self.remove, self.doubtful)]

    @property
    def n(self) -> int:
        return len(self.keys)

    @property
    def n_remove(self) -> int:
        return int(np.count_nonzero(self.remove))

    @property
    def n_doubtful(self) -> int:
        return int(np.count_nonzero(self.doubtful))

    @property
    def n_vetoed(self) -> int:
        return int(np.count_nonzero(self.vetoed))

    @property
    def cluster_set_sha(self) -> str:
        """``cluster_set_fingerprint`` of these clusters: which axon and ring build saved decisions belong to."""
        return cluster_set_fingerprint(self.stable_keys)

    def index_of(self, stable_key: str) -> int:
        """The row of the cluster named ``stable_key`` (KeyError when none)."""
        try:
            return self.stable_keys.index(str(stable_key))
        except ValueError:
            raise KeyError(f"no cluster {stable_key!r} in this classification") from None

    def rule_description(self) -> str:
        """The rule applied, in words (for reports and the window)."""
        if self.has_widefield:
            return (f"{self.rule_version}: isolation + widefield (interior "
                    f"{'usable' if self.interior_usable else 'not usable: ' + (self.interior_source or 'no cut')})")
        return f"{self.rule_version}: isolation only (no widefield images; completeness not certified)"


def classify_lumen(
    res: RingsResult,
    *,
    widefield: Optional[WidefieldLumenFlags] = None,
    isolation: Optional[IsolationFlags] = None,
    params: LumenParams = LumenParams(),
) -> LumenClassification:
    """
    The automatic lumen classification of the rings of one axon (D-35a).

    ``isolation``: the ``IsolationFlags`` to read (None: ``isolation_flags(res,
    params)``); the rule reads its BOOLEAN flags as given. ``widefield``: the
    ``WidefieldLumenFlags`` of the same rings (``widefield_lumen_flags`` or
    ``widefield_flags_from_depths``); None = no images: the rule is isolation
    only (REMOVE = ISO_STRICT), the depths are NaN, the widefield flags False,
    and ``NO_WIDEFIELD_WARNING`` says that completeness is not certified. The
    widefield flags are recomputed from the depths with ``params`` (a
    warning when they disagree with the ones given). ValueError when either
    input describes other clusters than ``res``.
    """
    keys = lumen_cluster_keys(res)
    iso = isolation_flags(res, params) if isolation is None else isolation
    if [(int(k[0]), int(k[1])) for k in iso.keys] != keys:
        raise ValueError("classify_lumen: the isolation flags describe other clusters than these rings")
    n = len(keys)
    warnings_: List[str] = list(iso.warnings)
    has_wf = widefield is not None
    if widefield is not None:
        if [(int(k[0]), int(k[1])) for k in widefield.keys] != keys:
            raise ValueError("classify_lumen: the widefield flags describe other clusters than these rings")
        dt = _depth_array(widefield.depth_tubulin_nm, n, "classify_lumen: depth_tubulin_nm")
        ds = _depth_array(widefield.depth_spectrin_nm, n, "classify_lumen: depth_spectrin_nm")
        usable = bool(widefield.interior_usable)
        C_lab = np.asarray(widefield.centroids_lab_nm, dtype=np.float64).reshape(-1, 2)
        registration = str(widefield.registration)
        source = str(widefield.interior_source)
        warnings_.extend(widefield.warnings)
    else:
        dt = np.full(n, np.nan)
        ds = np.full(n, np.nan)
        usable = False
        C_lab = np.full((n, 2), np.nan)
        registration = ""
        source = ""
        warnings_.append(NO_WIDEFIELD_WARNING)
    rr = apply_lumen_rule(iso.iso_strict, iso.iso_mid, iso.iso_sens, dt, ds, interior_usable=usable, has_widefield=has_wf,
                          params=params)
    if widefield is not None and not (np.array_equal(np.asarray(widefield.wf250, bool), rr.wf250)
                                      and np.array_equal(np.asarray(widefield.wf0, bool), rr.wf0)):
        warnings_.append("the widefield flags given differ from the ones their depths give with these parameters; the "
                         "rule used the depths")
    clusters = _clusters_in_order(res)
    by_ring = {int(r.index): r for r in res.rings}
    stable = [stable_cluster_key(k, cl) for k, _j, cl in clusters]
    if len(set(stable)) != len(stable):
        warnings_.append("two clusters share a stable key (same ring, label and rounded centroid): decisions on them are "
                         "ambiguous")
    n_locs = np.array([int(cl.n_locs) for _k, _j, cl in clusters], dtype=np.int64)
    n_events = np.array([float("nan") if cl.n_events is None else float(cl.n_events) for _k, _j, cl in clusters],
                        dtype=np.float64)
    dz = []
    for k, _j, cl in clusters:
        zv = np.asarray(cl.z_values_nm, dtype=np.float64)
        dz.append(abs(float(np.mean(zv)) - float(by_ring[k].centre_z_nm)) if zv.size else float("nan"))
    n_rm, n_db = int(np.count_nonzero(rr.remove)), int(np.count_nonzero(rr.doubtful))
    if n_rm or n_db:
        warnings_.append(f"lumen rule: {n_rm} cluster(s) removed automatically, {n_db} doubtful (kept unless removed by "
                         f"hand), {int(np.count_nonzero(rr.vetoed))} vetoed by the interior")
    return LumenClassification(
        keys=keys, stable_keys=stable, centroids_nm=np.asarray(iso.centroids_nm, dtype=np.float64).reshape(-1, 2),
        centroids_lab_nm=C_lab, hull_depth_nm=np.asarray(iso.hull_depth_nm, dtype=np.float64),
        hop_strict_nm=np.asarray(iso.hop_strict_nm, dtype=np.float64), hop_mid_nm=np.asarray(iso.hop_mid_nm, dtype=np.float64),
        hop_sens_nm=np.asarray(iso.hop_sens_nm, dtype=np.float64), iso_strict=np.asarray(iso.iso_strict, dtype=bool),
        iso_mid=np.asarray(iso.iso_mid, dtype=bool), iso_sens=np.asarray(iso.iso_sens, dtype=bool), depth_tubulin_nm=dt,
        depth_spectrin_nm=ds, wf250=rr.wf250, wf0=rr.wf0, interior_usable=usable, has_widefield=has_wf, vetoed=rr.vetoed,
        remove=rr.remove, doubtful=rr.doubtful, reasons=list(rr.reasons), n_locs=n_locs, n_events=n_events,
        abs_dz_ring_nm=np.asarray(dz, dtype=np.float64), registration=registration, rule_version=LUMEN_RULE_VERSION,
        params=params, warnings=warnings_, ring=np.array([k for k, _j, _c in clusters], dtype=np.int64),
        labels=np.array([int(cl.label) for _k, _j, cl in clusters], dtype=np.int64), interior_source=source,
        source_name=str(res.source_name))


# ============================================================================
# The user's decisions (D-35b)
# ============================================================================

@dataclass
class LumenEdit:
    """One change of a cluster's final state: its ``order`` in the log (1, 2,
    ...), the cluster, the state AFTER the change (``removed``) and the manual
    action it leaves, whether a column result of the axon had been shown
    before it (``after_results_shown``), and what made it (``via``: "click",
    "checkbox", "select all doubtful", ...)."""

    order: int
    stable_key: str
    removed: bool
    manual_action: str
    after_results_shown: bool
    via: str


def _parse_bool(value: Any, what: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
        return bool(float(value))
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "t"):
        return True
    if text in ("false", "0", "no", "n", "f", ""):
        return False
    raise ValueError(f"{what}: not a boolean: {value!r}")


def _parse_int(value: Any, what: str) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "none"):
        return None
    try:
        f = float(text)
    except ValueError as exc:
        raise ValueError(f"{what}: not an integer: {value!r}") from exc
    if not (math.isfinite(f) and f.is_integer()):
        raise ValueError(f"{what}: not an integer: {value!r}")
    return int(f)


def _parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(str(value).strip())
    except ValueError:
        return float("nan")


def _clean_text(value: Any) -> str:
    """A free text safe inside ``edit_details`` (no ';' nor '|')."""
    return str(value).replace(";", ",").replace("|", "/").strip()


def _key_parts(stable_key: str) -> Optional[Tuple[int, int, int, int]]:
    m = _STABLE_KEY_RE.match(str(stable_key))
    return None if m is None else (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def _clean_sha(value: Any) -> str:
    """A saved ``cluster_set_sha`` cell ("" when missing or empty)."""
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in ("", "nan", "none") else text


class LumenDecisions:
    """
    The automatic classification of one axon's clusters and the user's
    changes to it (D-35b): per cluster (named by ``stable_cluster_key``) the
    automatic class (``auto_class``: kept / removed / doubtful, with the
    rule's reasons), the manual action (``manual_action``: "none" = the
    automatic state; "removed" = a kept or doubtful cluster removed by hand;
    "restored" = an automatically removed cluster kept by hand) and the final
    state (``is_removed``). Doubtful clusters are KEPT unless removed.

    Every change of a final state is one ``LumenEdit`` in ``edit_log`` (in
    order; a call that changes nothing adds nothing). ``mark_results_shown``
    records that a column result of this axon was shown to the user; an edit
    after that sets ``results_shown_before_edit``, which is never reset --
    reverting an edit does not un-see a result -- and every result made from
    these decisions carries it ("edited after results were shown", R9/R10).

    ``to_rows`` / ``from_rows`` write and read one row per cluster
    (``LUMEN_DECISION_COLUMNS``; strings from a CSV are accepted), the log,
    the flag and the cluster set's fingerprint included; ``apply`` removes
    the final set from the rings (``clean_rings``), locating each cluster by
    its stable key, so saved decisions apply to a fresh ``build_rings`` of
    the same localizations. ``carry_results_shown`` / ``adopt`` bring the
    provenance of an earlier review of the same axon into these decisions
    (the review window re-opened), so the flag belongs to the axon.
    Pure Python, no Qt: the review window drives it from the GUI thread.
    """

    def __init__(
        self,
        stable_keys: Sequence[str],
        auto_classes: Sequence[str],
        auto_reasons: Optional[Sequence[Sequence[str]]] = None,
        *,
        classification: Optional[LumenClassification] = None,
        info: Optional[Mapping[str, Mapping[str, Any]]] = None,
        rule_version: str = LUMEN_RULE_VERSION,
    ) -> None:
        keys = [str(k) for k in stable_keys]
        if len(set(keys)) != len(keys):
            raise ValueError("LumenDecisions: repeated stable key")
        classes = [str(c) for c in auto_classes]
        if len(classes) != len(keys):
            raise ValueError(f"LumenDecisions: {len(classes)} automatic classes for {len(keys)} clusters")
        bad = sorted({c for c in classes if c not in AUTO_CLASSES})
        if bad:
            raise ValueError(f"LumenDecisions: unknown automatic class(es) {bad} (expected {list(AUTO_CLASSES)})")
        reasons = [tuple()] * len(keys) if auto_reasons is None else [tuple(str(r) for r in rs) for rs in auto_reasons]
        if len(reasons) != len(keys):
            raise ValueError(f"LumenDecisions: {len(reasons)} reason tuples for {len(keys)} clusters")
        self.classification: Optional[LumenClassification] = classification
        self.rule_version = str(rule_version)
        self._keys: List[str] = keys
        self._auto: Dict[str, str] = dict(zip(keys, classes))
        self._reasons: Dict[str, Tuple[str, ...]] = dict(zip(keys, reasons))
        self._removed: Dict[str, bool] = {k: c == AUTO_REMOVED for k, c in zip(keys, classes)}
        self._info: Dict[str, Dict[str, Any]] = {k: dict(info.get(k, {})) if info is not None else {} for k in keys}
        self.edit_log: List[LumenEdit] = []
        self.results_shown: bool = False
        self._results_shown_before_edit: bool = False
        self.warnings: List[str] = []
        # Set by from_rows: how many saved clusters the rows named, and how many of them are clusters of these
        # decisions (re-attached to a classification; = rows_read when detached).
        self.rows_read: int = 0
        self.rows_matched: int = 0

    # ---- construction -------------------------------------------------
    @classmethod
    def from_classification(cls, classification: LumenClassification) -> "LumenDecisions":
        """The automatic state of ``classification``: removed clusters removed, the rest kept, no edit."""
        c = classification
        return cls(c.stable_keys, c.auto_classes, c.reasons, classification=c, info=_classification_info(c),
                   rule_version=c.rule_version)

    # ---- queries ------------------------------------------------------
    @property
    def stable_keys(self) -> List[str]:
        return list(self._keys)

    def _check(self, key: str) -> str:
        k = str(key)
        if k not in self._auto:
            raise KeyError(f"no cluster {k!r} in these decisions")
        return k

    def auto_class(self, key: str) -> str:
        return self._auto[self._check(key)]

    def auto_reasons(self, key: str) -> Tuple[str, ...]:
        return self._reasons[self._check(key)]

    def is_removed(self, key: str) -> bool:
        return self._removed[self._check(key)]

    def manual_action(self, key: str) -> str:
        k = self._check(key)
        return _manual_action(self._auto[k], self._removed[k])

    def doubtful_keys(self) -> List[str]:
        return [k for k in self._keys if self._auto[k] == AUTO_DOUBTFUL]

    def final_removed_keys(self) -> List[str]:
        return [k for k in self._keys if self._removed[k]]

    def automatic_removed_keys(self) -> List[str]:
        return [k for k in self._keys if self._auto[k] == AUTO_REMOVED]

    def info(self, key: str) -> Dict[str, Any]:
        """What is known of the cluster (ring, position, label, x_nm, y_nm and the descriptors)."""
        return dict(self._info[self._check(key)])

    @property
    def n_removed_auto(self) -> int:
        return sum(1 for k in self._keys if self._auto[k] == AUTO_REMOVED)

    @property
    def n_removed_final(self) -> int:
        return sum(1 for k in self._keys if self._removed[k])

    @property
    def n_manual(self) -> int:
        """Clusters whose final state differs from the automatic one."""
        return sum(1 for k in self._keys if _manual_action(self._auto[k], self._removed[k]) != MANUAL_NONE)

    @property
    def n_doubtful(self) -> int:
        return sum(1 for k in self._keys if self._auto[k] == AUTO_DOUBTFUL)

    @property
    def results_shown_before_edit(self) -> bool:
        """True once a final state changed after ``mark_results_shown``; never reset."""
        return self._results_shown_before_edit

    @property
    def cluster_set_sha(self) -> str:
        """``cluster_set_fingerprint`` of these decisions' clusters."""
        return cluster_set_fingerprint(self._keys)

    # ---- changes ------------------------------------------------------
    def set_removed(self, key: str, removed: bool, via: str = "api") -> bool:
        """Set the final state of one cluster; True when it changed (then one log entry)."""
        k = self._check(key)
        want = bool(removed)
        if self._removed[k] == want:
            return False
        self._removed[k] = want
        after = bool(self.results_shown)
        if after:
            self._results_shown_before_edit = True
        self.edit_log.append(LumenEdit(order=len(self.edit_log) + 1, stable_key=k, removed=want,
                                       manual_action=_manual_action(self._auto[k], want), after_results_shown=after,
                                       via=str(via)))
        return True

    def toggle(self, key: str, via: str = "click") -> bool:
        """Flip one cluster (removed <-> kept); returns the NEW state (True = removed)."""
        k = self._check(key)
        self.set_removed(k, not self._removed[k], via=via)
        return self._removed[k]

    def restore(self, key: str, via: str = "restore") -> bool:
        """Keep one cluster (an automatically removed one becomes "restored"); True when it changed."""
        return self.set_removed(key, False, via=via)

    def reset(self, key: str, via: str = "reset") -> bool:
        """Return one cluster to its automatic state; True when it changed."""
        k = self._check(key)
        return self.set_removed(k, self._auto[k] == AUTO_REMOVED, via=via)

    def select_all_doubtful(self, via: str = "select all doubtful") -> int:
        """Remove every doubtful cluster; returns how many changed."""
        return sum(1 for k in self.doubtful_keys() if self.set_removed(k, True, via=via))

    def clear_all_doubtful(self, via: str = "clear all doubtful") -> int:
        """Keep every doubtful cluster (their automatic state); returns how many changed."""
        return sum(1 for k in self.doubtful_keys() if self.set_removed(k, False, via=via))

    def mark_results_shown(self) -> None:
        """A column result of this axon was shown: every later edit sets ``results_shown_before_edit``."""
        self.results_shown = True

    def carry_results_shown(self, results_shown: bool, results_shown_before_edit: bool = False) -> None:
        """
        Carry the provenance of an earlier review of the SAME axon into these
        decisions (D-35b: the flag is the axon's, not one window's): if
        results of the axon had been shown there, they count as shown here
        (every later edit is flagged); if its set had been edited after
        results were shown, this flag is set too. Never clears anything.
        """
        if results_shown or results_shown_before_edit:
            self.results_shown = True
        if results_shown_before_edit:
            self._results_shown_before_edit = True

    def adopt(self, other: "LumenDecisions", *, via: str = "loaded") -> int:
        """
        Take ``other``'s final state for every cluster both decisions hold
        (e.g. a saved table loaded into a window that already has edits or
        results): each change is an edit of THESE decisions, logged with
        ``via`` (and flagged if results were already shown here), then
        ``other``'s provenance is carried (``carry_results_shown``). Clusters
        ``other`` lacks keep their state. Returns how many changed.
        """
        n = 0
        for k in self._keys:
            if k in other._removed and self.set_removed(k, bool(other._removed[k]), via=via):
                n += 1
        self.carry_results_shown(bool(other.results_shown), bool(other.results_shown_before_edit))
        return n

    def snapshot(self) -> "LumenDecisions":
        """An independent copy of the current decisions (the classification shared: it is never modified), for a
        worker thread to read while the user keeps editing these (``run_cleaned_analyses`` takes one first)."""
        other = type(self).__new__(type(self))
        other.classification = self.classification
        other.rule_version = self.rule_version
        other._keys = list(self._keys)
        other._auto = dict(self._auto)
        other._reasons = dict(self._reasons)
        other._removed = dict(self._removed)
        other._info = {k: dict(v) for k, v in self._info.items()}
        other.edit_log = list(self.edit_log)
        other.results_shown = bool(self.results_shown)
        other._results_shown_before_edit = bool(self._results_shown_before_edit)
        other.warnings = list(self.warnings)
        other.rows_read = int(self.rows_read)
        other.rows_matched = int(self.rows_matched)
        return other

    # ---- export / import ------------------------------------------------
    def to_rows(self) -> List[Dict[str, Any]]:
        """One row per cluster, ``LUMEN_DECISION_COLUMNS`` (plain str / int / float / bool / None values)."""
        orders: Dict[str, List[LumenEdit]] = {k: [] for k in self._keys}
        for e in self.edit_log:
            orders.setdefault(e.stable_key, []).append(e)
        sha = self.cluster_set_sha
        rows: List[Dict[str, Any]] = []
        for k in self._keys:
            inf = self._info.get(k, {})
            parts = _key_parts(k)
            edits = orders.get(k, [])
            row: Dict[str, Any] = {
                "stable_key": k,
                "ring": inf.get("ring", parts[0] if parts else None),
                "position": inf.get("position"),
                "label": inf.get("label", parts[1] if parts else None),
                "x_nm": inf.get("x_nm", float(parts[2]) if parts else float("nan")),
                "y_nm": inf.get("y_nm", float(parts[3]) if parts else float("nan")),
                "auto_class": self._auto[k],
                "auto_reasons": ";".join(self._reasons.get(k, ())),
                "manual_action": _manual_action(self._auto[k], self._removed[k]),
                "final_removed": bool(self._removed[k]),
                "edit_orders": ";".join(str(e.order) for e in edits),
                "edit_details": ";".join(f"{e.order}|{'removed' if e.removed else 'kept'}|{int(bool(e.after_results_shown))}|"
                                         f"{_clean_text(e.via)}" for e in edits),
                "results_shown": bool(self.results_shown),
                "results_shown_before_edit": bool(self._results_shown_before_edit),
                "rule_version": self.rule_version,
                "cluster_set_sha": sha,
            }
            for col in LUMEN_DECISION_COLUMNS:
                if col not in row:
                    row[col] = inf.get(col)
            rows.append({col: row[col] for col in LUMEN_DECISION_COLUMNS})
        return rows

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]],
                  classification: Optional[LumenClassification] = None, *,
                  strict_manual: bool = False) -> "LumenDecisions":
        """
        Decisions from ``to_rows`` rows (as read back from a CSV: every value a
        string is fine). Detached (no ``classification``): the clusters, their
        automatic classes and reasons, final states, the edit log and the flags
        exactly as saved. Re-attached to a ``classification`` (the same axon
        classified again, e.g. the window re-opened): the clusters and the
        automatic classes are the classification's; each saved final state is
        applied to the cluster of the same stable key. A saved cluster the
        classification lacks and a cluster without a saved row are reported in
        ``warnings``; a cluster whose automatic class changed keeps its saved
        MANUAL action (its saved final state) but, when it had none, takes the
        new automatic state (a warning either way). The edit log is rebuilt
        from ``edit_details`` (else from ``edit_orders``, the states
        alternating from the automatic one).

        A table of ANOTHER axon is refused, never silently read as the
        automatic state (ValueError): rows carrying a ``cluster_set_sha`` none
        of which is this classification's fingerprint (``cluster_set_
        fingerprint``: another axon, or this axon's rings built otherwise), or
        rows none of whose stable keys is a cluster here. Rows of other cluster
        sets in a shared table are ignored (a warning). When a cluster appears
        in several rows (a table holding several exports of the axon, appended
        one after the other), its LAST row is used. ``strict_manual``: a saved
        cluster with a manual action that is not a cluster here is an error
        too (the set the user chose could not be reproduced). ``rows_read`` /
        ``rows_matched`` of the result say how many saved clusters were read
        and matched. ValueError on a missing column, an unreadable value or
        (re-attached) no row at all.
        """
        table = [dict(r) for r in rows]
        warnings_: List[str] = []
        for i, r in enumerate(table):
            missing = [c for c in _DECISION_REQUIRED_ON_LOAD if c not in r]
            if missing:
                raise ValueError(f"LumenDecisions.from_rows: row {i + 1} lacks the column(s) {missing}")
        if classification is not None and not table:
            raise ValueError("LumenDecisions.from_rows: no decision row to apply")
        # Which cluster set the rows were saved on.
        shas = [_clean_sha(r.get("cluster_set_sha")) for r in table]
        distinct = sorted({s for s in shas if s})
        if classification is not None:
            mine = classification.cluster_set_sha
            if distinct and mine not in distinct:
                raise ValueError(f"LumenDecisions.from_rows: the decisions were saved on another cluster set (fingerprint "
                                 f"{', '.join(distinct)}; this axon's clusters: {mine}): another axon, or this axon's rings "
                                 "built otherwise (another ROI edge or other parameters). They are not applied.")
            n_other = sum(1 for s in shas if s != mine) if distinct else 0
            if n_other:
                table = [r for r, s in zip(table, shas) if s == mine]
                warnings_.append(f"{n_other} row(s) of other cluster sets (other axons or ring builds in the same table) "
                                 "ignored")
        elif len(distinct) > 1:
            warnings_.append(f"the rows come from {len(distinct)} cluster sets (several axons or ring builds), read together")
        saved: Dict[str, Dict[str, Any]] = {}
        order_keys: List[str] = []
        repeated: List[str] = []
        for i, r in enumerate(table):
            k = str(r["stable_key"]).strip()
            auto = str(r["auto_class"]).strip()
            if auto not in AUTO_CLASSES:
                raise ValueError(f"LumenDecisions.from_rows: row {i + 1}: unknown automatic class {auto!r}")
            if k in saved:
                repeated.append(k)
            else:
                order_keys.append(k)
            # The LAST row of a cluster wins: an export appended to a table that already holds this axon's rows comes
            # after them, and it is the newer state.
            saved[k] = dict(r, stable_key=k, auto_class=auto,
                            final_removed=_parse_bool(r["final_removed"], f"row {i + 1} final_removed"))
        if repeated:
            uniq = sorted(set(repeated))
            warnings_.append(f"{len(uniq)} cluster(s) appear in more than one row (the table holds several exports of this "
                             f"axon): the LAST row of each is used ({', '.join(uniq[:3])}{', ...' if len(uniq) > 3 else ''})")
        shown_before = any(_parse_bool(r.get("results_shown_before_edit", False), "results_shown_before_edit")
                           for r in saved.values())
        shown = shown_before or any(_parse_bool(r.get("results_shown", False), "results_shown") for r in saved.values())
        versions = sorted({str(r.get("rule_version", "")).strip() for r in saved.values()} - {""})
        keep_auto: set[str] = set()
        if classification is None:
            info = {k: _row_info(saved[k]) for k in order_keys}
            reasons = [tuple(x for x in str(saved[k].get("auto_reasons", "") or "").split(";") if x) for k in order_keys]
            out = cls(order_keys, [saved[k]["auto_class"] for k in order_keys], reasons, info=info,
                      rule_version=versions[0] if len(versions) == 1 else LUMEN_RULE_VERSION)
            if len(versions) > 1:
                warnings_.append(f"the rows carry several rule versions {versions}")
            matched = list(order_keys)
        else:
            out = cls.from_classification(classification)
            matched = [k for k in order_keys if k in out._auto]
            if not matched:
                raise ValueError(f"LumenDecisions.from_rows: none of the {len(order_keys)} saved cluster(s) is a cluster of "
                                 f"this axon's {len(out._keys)}: the table belongs to another axon, or to rings built with "
                                 "other parameters. Nothing was applied.")
            unknown = [k for k in order_keys if k not in out._auto]
            if strict_manual:
                lost = [k for k in unknown if _manual_action(saved[k]["auto_class"], saved[k]["final_removed"]) != MANUAL_NONE]
                if lost:
                    raise ValueError(f"LumenDecisions.from_rows: {len(lost)} saved manual edit(s) name clusters these rings "
                                     f"do not have ({', '.join(lost[:5])}{', ...' if len(lost) > 5 else ''}): the set the "
                                     "user chose cannot be reproduced here")
            for k in unknown:
                warnings_.append(f"saved cluster {k} is not in this classification (another axon, or rings built with other "
                                 "parameters): its decision is ignored")
            unsaved = [k for k in out._keys if k not in saved]
            if unsaved:
                warnings_.append(f"{len(unsaved)} cluster(s) of the classification have no saved decision: automatic state "
                                 f"({', '.join(unsaved[:5])}{', ...' if len(unsaved) > 5 else ''})")
            for k in out._keys:
                if k in saved and saved[k]["auto_class"] != out._auto[k]:
                    if _manual_action(saved[k]["auto_class"], saved[k]["final_removed"]) == MANUAL_NONE:
                        keep_auto.add(k)
                        warnings_.append(f"cluster {k}: saved automatic class {saved[k]['auto_class']!r}, now "
                                         f"{out._auto[k]!r}; it had no manual action, so it takes the new automatic state")
                    else:
                        warnings_.append(f"cluster {k}: saved automatic class {saved[k]['auto_class']!r}, now "
                                         f"{out._auto[k]!r}; its saved manual action is kept (final state "
                                         f"{'removed' if saved[k]['final_removed'] else 'kept'})")
            if versions and versions != [out.rule_version]:
                warnings_.append(f"the saved decisions were made under {versions}, this classification under "
                                 f"{out.rule_version!r}")
        # final states
        for k in out._keys:
            if k in saved and k not in keep_auto:
                out._removed[k] = bool(saved[k]["final_removed"])
        # the edit log
        entries: List[LumenEdit] = []
        for k in out._keys:
            if k not in saved:
                continue
            entries.extend(_edits_of_row(saved[k], out._auto[k], warnings_))
        entries.sort(key=lambda e: e.order)
        seen: Dict[int, str] = {}
        for e in entries:
            if e.order in seen:
                warnings_.append(f"edit order {e.order} appears for {seen[e.order]} and {e.stable_key}")
            seen[e.order] = e.stable_key
        out.edit_log = entries
        last: Dict[str, bool] = {}
        for e in entries:
            last[e.stable_key] = e.removed
        for k, state in last.items():
            if state != out._removed[k]:
                warnings_.append(f"cluster {k}: its last logged edit left it {'removed' if state else 'kept'}, its saved final "
                                 f"state is {'removed' if out._removed[k] else 'kept'}")
        out.results_shown = bool(shown)
        out._results_shown_before_edit = bool(shown_before)
        out.warnings = warnings_
        out.rows_read = len(order_keys)
        out.rows_matched = len(matched)
        return out

    # ---- applying them ------------------------------------------------
    def locate(self, res: RingsResult, keys: Optional[Sequence[str]] = None) -> Tuple[List[Tuple[int, int]], List[str], List[str]]:
        """((Ring.index, position) of each of ``keys`` -- default the final removed set -- in ``res``, the keys not
        found, notes on keys found only within 1 nm of their rounded centroid)."""
        wanted = self.final_removed_keys() if keys is None else [str(k) for k in keys]
        exact: Dict[str, Tuple[int, int]] = {}
        by_ring: Dict[int, List[Tuple[int, Cluster]]] = {}
        for k, j, cl in _clusters_in_order(res):
            exact.setdefault(stable_cluster_key(k, cl), (k, j))
            by_ring.setdefault(k, []).append((j, cl))
        found: List[Tuple[int, int]] = []
        missing: List[str] = []
        notes: List[str] = []
        for key in wanted:
            if key in exact:
                found.append(exact[key])
                continue
            parts = _key_parts(key)
            hit: Optional[Tuple[int, int]] = None
            if parts is not None:
                ring, label, x, y = parts
                for j, cl in by_ring.get(ring, []):
                    c = np.asarray(cl.centroid_nm, dtype=np.float64).reshape(2)
                    if int(cl.label) == label and abs(float(c[0]) - x) <= 1.0 and abs(float(c[1]) - y) <= 1.0:
                        hit = (ring, j)
                        break
            if hit is None:
                missing.append(key)
            else:
                found.append(hit)
                notes.append(f"cluster {key} found as {stable_cluster_key(hit[0], by_ring[hit[0]][hit[1]][1])} (same ring and "
                             "label, centroid within 1 nm of the saved one)")
        return found, missing, notes

    def removal_reasons(self) -> Dict[str, str]:
        """The reason recorded with each cluster of the final removed set: the rule's terms, or the manual removal."""
        out: Dict[str, str] = {}
        for k in self.final_removed_keys():
            if self._auto[k] == AUTO_REMOVED:
                out[k] = f"automatic ({'; '.join(self._reasons.get(k, ())) or LUMEN_RULE_VERSION})"
            else:
                out[k] = f"manual (automatic class {self._auto[k]})"
        return out

    def apply(self, res: RingsResult, *, lpz_nm: Optional[NDArray[np.float64]] = None) -> RingsResult:
        """
        ``clean_rings(res, the final removed clusters)`` -- each located in
        ``res`` by its stable key (a cluster whose rounded centroid moved by one
        nm on a rebuild is still found by its ring and label, with a note in
        ``warnings``), with the reason of each (``removal_reasons``). ``res``
        must be the rings of ``build_rings`` (not an already cleaned result:
        ``clean_rings`` rebuilds a ring from the analysis of all its clusters).
        ValueError when a cluster to remove is not in ``res``, or when none of
        these decisions' clusters is (the decisions of another axon, even
        with nothing to remove, are never applied silently).
        """
        if self._keys:
            anywhere, _missing_all, _notes_all = self.locate(res, self._keys)
            if not anywhere:
                raise ValueError(f"LumenDecisions.apply: none of these decisions' {len(self._keys)} cluster(s) is in these "
                                 "rings: they were made on another axon or on rings built with other parameters")
        keys = self.final_removed_keys()
        found, missing, notes = self.locate(res, keys)
        if missing:
            raise ValueError(f"LumenDecisions.apply: {len(missing)} cluster(s) to remove are not in these rings "
                             f"({', '.join(missing[:5])}{', ...' if len(missing) > 5 else ''}): the decisions were made on "
                             "another axon or on rings built with other parameters")
        for n in notes:
            if n not in self.warnings:
                self.warnings.append(n)
        reasons = self.removal_reasons()
        per = {pos: reasons[k] for pos, k in zip(found, keys)}
        return clean_rings(res, found, reason=per, lpz_nm=lpz_nm)


def _manual_action(auto: str, removed: bool) -> str:
    default = auto == AUTO_REMOVED
    if bool(removed) == default:
        return MANUAL_NONE
    return MANUAL_REMOVED if removed else MANUAL_RESTORED


def _classification_info(c: LumenClassification) -> Dict[str, Dict[str, Any]]:
    """What ``to_rows`` writes about each cluster of a classification."""
    info: Dict[str, Dict[str, Any]] = {}
    for i, k in enumerate(c.stable_keys):
        ring, pos = c.keys[i]
        info[k] = {
            "ring": int(ring), "position": int(pos),
            "label": int(c.labels[i]) if len(c.labels) == len(c.keys) else None,
            "x_nm": float(c.centroids_nm[i, 0]), "y_nm": float(c.centroids_nm[i, 1]),
            "has_widefield": bool(c.has_widefield), "interior_usable": bool(c.interior_usable),
            "registration": str(c.registration),
            "hull_depth_nm": float(c.hull_depth_nm[i]), "hop_strict_nm": float(c.hop_strict_nm[i]),
            "hop_mid_nm": float(c.hop_mid_nm[i]), "depth_tubulin_nm": float(c.depth_tubulin_nm[i]),
            "depth_spectrin_nm": float(c.depth_spectrin_nm[i]), "n_locs": int(c.n_locs[i]),
            "n_events": None if not math.isfinite(float(c.n_events[i])) else int(c.n_events[i]),
            "abs_dz_ring_nm": float(c.abs_dz_ring_nm[i]),
        }
    return info


def _row_info(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The cluster descriptors of a saved row, parsed back to numbers."""
    info: Dict[str, Any] = {}
    for col in ("ring", "position", "label", "n_locs", "n_events"):
        if col in row:
            try:
                info[col] = _parse_int(row[col], col)
            except ValueError:
                info[col] = None
    for col in ("x_nm", "y_nm", "hull_depth_nm", "hop_strict_nm", "hop_mid_nm", "depth_tubulin_nm", "depth_spectrin_nm",
                "abs_dz_ring_nm"):
        if col in row:
            info[col] = _parse_float(row[col])
    for col in ("has_widefield", "interior_usable"):
        if col in row:
            try:
                info[col] = _parse_bool(row[col], col)
            except ValueError:
                info[col] = None
    if "registration" in row:
        info["registration"] = "" if row["registration"] is None else str(row["registration"])
    return info


def _edits_of_row(row: Mapping[str, Any], auto: str, warnings_: List[str]) -> List[LumenEdit]:
    """The log entries of one saved row (``edit_details``, else ``edit_orders`` with alternating states)."""
    key = str(row["stable_key"])
    details = str(row.get("edit_details", "") or "").strip()
    out: List[LumenEdit] = []
    if details and details.lower() != "nan":
        for part in details.split(";"):
            fields_ = part.split("|")
            if len(fields_) < 3:
                warnings_.append(f"cluster {key}: unreadable edit entry {part!r} skipped")
                continue
            order = _parse_int(fields_[0], "edit order")
            if order is None:
                warnings_.append(f"cluster {key}: edit entry {part!r} without an order skipped")
                continue
            state = fields_[1].strip().lower() == "removed"
            after = _parse_bool(fields_[2], "edit after_results_shown")
            via = fields_[3].strip() if len(fields_) > 3 else "loaded"
            out.append(LumenEdit(order=order, stable_key=key, removed=state, manual_action=_manual_action(auto, state),
                                 after_results_shown=after, via=via))
        return out
    orders_text = str(row.get("edit_orders", "") or "").strip()
    if not orders_text or orders_text.lower() == "nan":
        return out
    state = auto == AUTO_REMOVED
    for part in orders_text.split(";"):
        order = _parse_int(part, "edit order")
        if order is None:
            continue
        state = not state
        out.append(LumenEdit(order=order, stable_key=key, removed=state, manual_action=_manual_action(auto, state),
                             after_results_shown=False, via="loaded"))
    return out


# ============================================================================
# The re-run (the review window's job)
# ============================================================================

@dataclass
class CleanedAnalyses:
    """
    The analyses of one axon's rings after the lumen cleaning
    (``run_cleaned_analyses``): the cleaned ``rings``; ``arc_centroid`` the arc
    test on the centroid membrane (PRIMARY, D-35c), ``columns_2d`` the 2D
    test (sensitivity, D-29a), ``arc_localization`` the arc test on the
    localization membrane (a DIAGNOSTIC with a known conservative bias,
    ``arc_localization_note``) -- each None when it failed, with a warning;
    the cleaning counts (automatic removals, doubtful, final removals,
    manual changes) and the removed sets at the moment the run started;
    ``edited_after_results_shown`` (the decisions' flag); ``leak_calibrated``
    False until H5-E (every statistic here reads against an uncalibrated
    null: ``leak_note``); the rule; the null size; the warnings; the time.
    """

    rings: RingsResult
    arc_centroid: Optional["ArcColumnsResult"]
    arc_localization: Optional["ArcColumnsResult"]
    columns_2d: Optional["AxonColumnsResult"]
    n_removed_auto: int
    n_doubtful: int
    n_removed_final: int
    n_manual: int
    edited_after_results_shown: bool
    leak_calibrated: bool = False
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0
    rule_version: str = LUMEN_RULE_VERSION
    has_widefield: bool = False
    removed_keys: List[str] = field(default_factory=list)
    automatic_removed_keys: List[str] = field(default_factory=list)
    n_null: int = 0
    leak_note: str = LEAK_NOT_CALIBRATED_NOTE
    arc_localization_note: str = ""


def _cleaning_notes(before: RingsResult, after: RingsResult) -> List[str]:
    """What ``clean_rings`` added to the rings' warnings (the result's and each ring's), prefixed."""
    old = set(str(w) for w in before.warnings)
    out = [f"cleaned rings: {w}" for w in after.warnings if str(w) not in old]
    by_index = {int(r.index): r for r in before.rings}
    for ring in sorted(after.rings, key=lambda q: int(q.index)):
        prev = by_index.get(int(ring.index))
        seen = set(str(w) for w in (prev.warnings if prev is not None else []))
        out.extend(f"cleaned ring {int(ring.index)}: {w}" for w in ring.warnings if str(w) not in seen)
    return out


def _analysis_notes(what: str, result: Any) -> List[str]:
    """An analysis' own warnings, prefixed with its name, and a note when it gave no z_A."""
    out = [f"{what}: {w}" for w in (getattr(result, "warnings", None) or [])]
    z = getattr(result, "z_A", float("nan"))
    try:
        finite = math.isfinite(float(z))
    except (TypeError, ValueError):
        finite = False
    if not finite:
        out.append(f"{what}: no z_A (not defined on these rings; the notes above say why)")
    return out


def default_columns_params(path: str = DEFAULT_COLUMNS_PARAMS_PATH) -> ColumnsParams:
    """The pre-registered column parameters (D-22, ``config/columns_params.yaml``): tau_0, the seed,
    the null size. FileNotFoundError when the file is not there."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"the pre-registered column parameters are not at {path}; pass columns_params")
    return load_columns_params(path)


def run_cleaned_analyses(
    res: RingsResult,
    decisions: LumenDecisions,
    *,
    columns_params: Optional[ColumnsParams] = None,
    n_null: Optional[int] = None,
    lpz_nm: Optional[NDArray[np.float64]] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> CleanedAnalyses:
    """
    The ring analyses of one axon on the clusters the decisions keep -- the
    job of the review window's re-run button, meant for a worker thread (it
    touches no Qt object and reads ``decisions`` once, at the start):

    1. ``rings = decisions.apply(res, lpz_nm=lpz_nm)`` (``clean_rings``;
       ``build_rings`` is NOT re-run);
    2. ``arc_centroid`` = ``analyze_arc_columns(reference_curve=
       "centroid_membrane")``, the primary test (D-35c);
    3. ``columns_2d`` = ``analyze_columns``, the 2D test (sensitivity);
    4. ``arc_localization`` = ``analyze_arc_columns(reference_curve=
       "localization_membrane")``, the diagnostic.

    ``columns_params``: the column parameters (None: the pre-registered file,
    ``default_columns_params``); ``n_null`` the null size of EVERY test (None:
    ``columns_params.n_null``; checked before anything runs). An analysis
    that fails is None with a warning; the cleaning itself raises
    (ValueError from ``apply``). ``warnings`` carries, besides this
    function's own notes, what the cleaning and every analysis had to say,
    each prefixed with where it comes from ("cleaned rings: ...", "arc test
    (centroid membrane): ...", "2D test: ...", "arc test (localization
    membrane, diagnostic): ..."): why a ring was left out, why z_A is not
    defined, a self-crossing membrane fit, a recipe that is not the frozen
    one. ``progress(text, fraction)`` is called before each step and at the
    end. It does not call ``decisions.mark_results_shown``: the caller
    does, when it shows them.
    """
    from tools.mps_matching import analyze_columns
    from tools.mps_unroll import LOCALIZATION_MEMBRANE_DIAGNOSTIC_NOTE, UnrollParams, analyze_arc_columns

    t0 = time.perf_counter()

    def say(text: str, fraction: float) -> None:
        if progress is not None:
            progress(str(text), float(fraction))

    # One snapshot of the decisions, taken before anything slow: the results describe exactly this set, whatever the
    # user edits while the analyses run (the window may pass a snapshot taken in its own thread).
    snap = decisions.snapshot()
    removed_keys = snap.final_removed_keys()
    auto_keys = snap.automatic_removed_keys()
    counts = (snap.n_removed_auto, snap.n_doubtful, snap.n_removed_final, snap.n_manual)
    edited = bool(snap.results_shown_before_edit)
    c = snap.classification
    has_wf = bool(c.has_widefield) if c is not None else False
    warnings_: List[str] = []
    # The parameters first: a bad null size must not cost a cleaning.
    cp = default_columns_params() if columns_params is None else columns_params
    if n_null is not None and (isinstance(n_null, bool) or not float(n_null).is_integer() or int(n_null) < 1):
        raise ValueError(f"run_cleaned_analyses: n_null must be an integer >= 1, got {n_null}")
    n_rep = int(cp.n_null if n_null is None else n_null)
    if n_rep < 1:
        raise ValueError(f"run_cleaned_analyses: n_null must be an integer >= 1, got {n_rep}")
    say(f"Removing {len(removed_keys)} cluster(s) and rebuilding the rings' contours", 0.02)
    n_notes = len(snap.warnings)
    rings = snap.apply(res, lpz_nm=lpz_nm)
    warnings_.extend(snap.warnings[n_notes:])
    warnings_.extend(_cleaning_notes(res, rings))
    cols_stub: Any = SimpleNamespace(include_suspect=True, tau0_nm=float(cp.tau0_nm), random_seed=int(cp.random_seed))
    uparams = UnrollParams(n_null=n_rep)
    arc_c: Optional[ArcColumnsResult] = None
    arc_l: Optional[ArcColumnsResult] = None
    cols: Optional[AxonColumnsResult] = None
    say("Arc test on the centroid membrane (primary)", 0.10)
    try:
        arc_c = analyze_arc_columns(rings, cols_stub, params=uparams, reference_curve="centroid_membrane")
    except Exception as exc:  # noqa: BLE001 - a worker job: every failure becomes a warning, never a crash
        warnings_.append(f"arc test on the centroid membrane failed: {type(exc).__name__}: {exc}")
    else:
        warnings_.extend(_analysis_notes("arc test (centroid membrane)", arc_c))
    say("2D test (sensitivity)", 0.35)
    try:
        cols = analyze_columns(rings, cp, n_null=n_rep)
    except Exception as exc:  # noqa: BLE001
        warnings_.append(f"2D test failed: {type(exc).__name__}: {exc}")
    else:
        warnings_.extend(_analysis_notes("2D test", cols))
    say("Arc test on the localization membrane (diagnostic)", 0.75)
    try:
        arc_l = analyze_arc_columns(rings, cols if cols is not None else cols_stub, params=uparams,
                                    reference_curve="localization_membrane")
    except Exception as exc:  # noqa: BLE001
        warnings_.append(f"arc test on the localization membrane failed: {type(exc).__name__}: {exc}")
    else:
        warnings_.extend(_analysis_notes("arc test (localization membrane, diagnostic)", arc_l))
    if edited:
        warnings_.append("the cluster set was edited after column results of this axon were shown (D-35b)")
    if not has_wf:
        warnings_.append(NO_WIDEFIELD_WARNING)
    warnings_.append(f"every statistic here is {LEAK_NOT_CALIBRATED_NOTE}")
    warnings_ = list(dict.fromkeys(warnings_))
    say("Done", 1.0)
    return CleanedAnalyses(
        rings=rings, arc_centroid=arc_c, arc_localization=arc_l, columns_2d=cols, n_removed_auto=int(counts[0]),
        n_doubtful=int(counts[1]), n_removed_final=int(counts[2]), n_manual=int(counts[3]), edited_after_results_shown=edited,
        leak_calibrated=False, warnings=warnings_, seconds=float(time.perf_counter() - t0), rule_version=snap.rule_version,
        has_widefield=has_wf, removed_keys=list(removed_keys), automatic_removed_keys=list(auto_keys), n_null=n_rep,
        arc_localization_note=LOCALIZATION_MEMBRANE_DIAGNOSTIC_NOTE)


# ============================================================================
# The review between sessions (the window's memory of each axon, D-35b)
# ============================================================================

def _json_default(obj: Any) -> Any:
    """numpy scalars in a saved session, as plain Python values."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


class LumenReviewStore:
    """
    The review window's memory of each axon: one JSON file per axon and
    rule version in ``directory`` holding that axon's decisions
    (``LumenDecisions.to_rows``: final states, the edit log, the flags, the
    cluster set's fingerprint), rewritten after every change. It makes the
    D-35(b) provenance the AXON's, not one window's: when the review of an
    axon is opened again -- the window closed and re-opened, the file
    reloaded, the application restarted -- its decisions come back together
    with the fact that its results were shown, so a later edit is still
    labelled "edited after results were shown". Files are named by a hash of
    the axon id and the rule version (an axon id holds a path); both are
    written inside and checked on reading. Pure Python, no Qt.

    One axon can be reviewed on several ring builds (another cluster set,
    with the same axon id: the parameters edited, the file re-exported under
    the same name, another pixel size). The axon's file holds the review of
    the build saved last; ``save`` first moves a review of ANOTHER cluster
    set to its own file (``kept_path_for``) instead of overwriting it, and
    ``restore`` brings it back when the axon's rings are built as they were
    then (final audit, 2026-10-06: saving one build -- even its automatic
    state, when its window was merely closed -- destroyed the hand-made
    review of the other).
    """

    def __init__(self, directory: str) -> None:
        self.directory = os.path.abspath(str(directory))

    def path_for(self, axon_id: str, rule_version: str = LUMEN_RULE_VERSION) -> str:
        """The file of ``axon_id``'s review under ``rule_version``."""
        digest = hashlib.sha256(f"{axon_id}\n{rule_version}".encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.directory, f"lumen_review_{digest}.json")

    def kept_path_for(self, axon_id: str, cluster_set_sha: str, rule_version: str = LUMEN_RULE_VERSION) -> str:
        """The file keeping ``axon_id``'s review of the cluster set ``cluster_set_sha`` while another ring build of the
        axon is the one in ``path_for``: that file's name plus the fingerprint."""
        sha = re.sub(r"[^0-9A-Za-z]", "", str(cluster_set_sha))[:64] or "unknown"
        root, ext = os.path.splitext(self.path_for(axon_id, rule_version))
        return f"{root}_{sha}{ext}"

    def save(self, axon_id: str, decisions: "LumenDecisions") -> str:
        """Write the decisions of ``axon_id`` (atomically: a temporary file, then a replace); returns the path. A review
        of another cluster set in that file is moved to ``kept_path_for`` first, never overwritten. OSError when the
        folder cannot be written."""
        os.makedirs(self.directory, exist_ok=True)
        path = self.path_for(axon_id, decisions.rule_version)
        self._keep_other_build(path, axon_id, decisions)
        data = {
            "format": LUMEN_REVIEW_SESSION_FORMAT, "axon_id": str(axon_id), "rule_version": str(decisions.rule_version),
            "cluster_set_sha": decisions.cluster_set_sha, "saved_at": datetime.now().isoformat(timespec="seconds"),
            "results_shown": bool(decisions.results_shown),
            "results_shown_before_edit": bool(decisions.results_shown_before_edit),
            "n_edits": len(decisions.edit_log), "rows": decisions.to_rows(),
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, default=_json_default)
        os.replace(tmp, path)
        return path

    def _keep_other_build(self, path: str, axon_id: str, decisions: "LumenDecisions") -> None:
        """Before ``path`` is rewritten with ``decisions``: a review saved there on ANOTHER cluster set is moved to its
        ``kept_path_for`` (an unreadable file is replaced as before: nothing in it could be restored)."""
        try:
            data = self._read(path, axon_id, decisions.rule_version)
        except ValueError:
            return
        old = _clean_sha(data.get("cluster_set_sha")) if data is not None else ""
        if old and old != decisions.cluster_set_sha:
            os.replace(path, self.kept_path_for(axon_id, old, decisions.rule_version))

    def load(self, axon_id: str, rule_version: str = LUMEN_RULE_VERSION) -> Optional[Dict[str, Any]]:
        """The saved review of ``axon_id`` (None when there is none). ValueError when the file cannot be read, is not a
        review session, or holds another axon or rule version."""
        return self._read(self.path_for(axon_id, rule_version), axon_id, rule_version)

    def _read(self, path: str, axon_id: str, rule_version: str) -> Optional[Dict[str, Any]]:
        """``load`` of the file ``path`` (the axon's file or one of its ``kept_path_for``)."""
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ValueError(f"the saved review {path} could not be read ({type(exc).__name__}: {exc})") from exc
        if not isinstance(data, dict) or data.get("format") != LUMEN_REVIEW_SESSION_FORMAT:
            raise ValueError(f"{path} is not a saved lumen review ({LUMEN_REVIEW_SESSION_FORMAT})")
        if str(data.get("axon_id")) != str(axon_id) or str(data.get("rule_version")) != str(rule_version):
            raise ValueError(f"{path} holds the review of {data.get('axon_id')!r} under {data.get('rule_version')!r}, not "
                             f"of {axon_id!r} under {rule_version!r}")
        if not isinstance(data.get("rows"), list):
            raise ValueError(f"{path} holds no decision rows")
        return data

    def restore(self, axon_id: str,
                classification: LumenClassification) -> Tuple[Optional["LumenDecisions"], List[str]]:
        """
        The decisions saved for ``axon_id`` re-attached to ``classification``
        (``LumenDecisions.from_rows``), with notes for the user; (None, [])
        when nothing was saved. When the saved decisions cannot be applied to
        these clusters (another ring build of the axon: another cluster-set
        fingerprint), the review of THESE clusters that ``save`` kept aside
        (``kept_path_for``) is restored when there is one; otherwise the
        automatic state is returned. Either way the saved provenance is
        carried (``carry_results_shown``: if results had been shown, every
        edit here is flagged), and a note says what was restored and that
        the other build's review is kept. An unreadable file gives (None,
        [its note]).
        """
        try:
            data = self.load(axon_id, classification.rule_version)
        except ValueError as exc:
            return None, [f"the saved review of this axon was not restored: {exc}"]
        if data is None:
            return None, []
        saved_at = str(data.get("saved_at", "?"))
        shown = bool(data.get("results_shown", False))
        before = bool(data.get("results_shown_before_edit", False))
        n_edits = int(data.get("n_edits", 0) or 0)
        try:
            d = LumenDecisions.from_rows(list(data["rows"]), classification)
        except ValueError as exc:
            other = _clean_sha(data.get("cluster_set_sha"))
            kept_note = ("; that review is not lost: it stays saved for its own clusters and comes back when this axon's "
                         "rings are built as they were then" if other and other != classification.cluster_set_sha else "")
            kept = self._restore_kept(axon_id, classification, shown, before)
            if kept is not None:
                return kept[0], [kept[1] + f" (the review saved {saved_at} on another ring build of this axon was not "
                                           f"applied here{kept_note})"]
            d = LumenDecisions.from_classification(classification)
            d.carry_results_shown(shown, before)
            return d, [f"the review of this axon saved {saved_at} ({n_edits} edit(s)) could not be applied to these "
                       f"clusters ({exc}); the automatic state is shown"
                       + ("; results of this axon had been shown, so every edit here is flagged 'edited after results "
                          "were shown' (D-35b)" if (shown or before) else "") + kept_note]
        note = (f"restored the review of this axon saved {saved_at}: {len(d.edit_log)} logged edit(s), "
                f"{d.n_manual} cluster(s) differing from the rule, results {'already shown' if d.results_shown else 'not shown yet'}"
                + ("; the set was edited after results were shown (D-35b)" if d.results_shown_before_edit else ""))
        return d, [note]

    def _restore_kept(self, axon_id: str, classification: LumenClassification, shown: bool,
                      before: bool) -> Optional[Tuple["LumenDecisions", str]]:
        """The review of THESE clusters that ``save`` kept aside when another ring build of the axon was saved after it,
        with the axon's provenance (``shown``, ``before``: the later review's) carried, D-35b; and its note. None when
        there is none or it cannot be applied."""
        path = self.kept_path_for(axon_id, classification.cluster_set_sha, classification.rule_version)
        try:
            data = self._read(path, axon_id, classification.rule_version)
            if data is None:
                return None
            d = LumenDecisions.from_rows(list(data["rows"]), classification)
        except ValueError:
            return None
        d.carry_results_shown(shown, before)
        note = (f"restored the review of this axon's clusters saved {data.get('saved_at', '?')} (kept when another ring "
                f"build of this axon was reviewed): {len(d.edit_log)} logged edit(s), {d.n_manual} cluster(s) differing "
                f"from the rule, results {'already shown' if d.results_shown else 'not shown yet'}"
                + ("; the set was edited after results were shown (D-35b)" if d.results_shown_before_edit else ""))
        return d, note
