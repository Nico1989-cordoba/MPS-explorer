# -*- coding: utf-8 -*-
"""
What the eclipse test does with a cluster flagged as MERGED (user's decision D2, 2026-10-23; branch
``feat/validations-d2-e1-e5``; ``docs/eclipse_tolerance.md`` sec. 10).

The merged-cluster criterion of ``tools.mps_eclipse_tolerance`` (``merged_cluster_flags``: the two event modes of a
cluster along the membrane are farther apart than one cluster's labels can be) flags clusters that the clustering
probably built from TWO real clusters. The eclipse question is "are the clusters of consecutive rings in the same
column?"; a merged cluster's single centre sits BETWEEN the two real ones, so it can miss both columns. Three
handlings, chosen with ``merged_handling``:

* ``"middle"``: the merged cluster enters the test with its one centre (the mean of all its localizations), as the
  program did before D2;
* ``"split"`` (decision D2a): the merged cluster enters as TWO clusters, the two halves of the same exact 1D
  two-means split along the membrane that measured its mode separation (``two_mode_split``: events, or
  localizations without events, grouped contiguously by position) -- each half with its own centre and its own
  centre error;
* ``"exclude"``: the merged cluster does not enter the matching (it stays in the polygon: the membrane is geometry).

Unflagged clusters are never touched, and the FOOTPRINT POOL is not affected by this choice (it keeps its own rule,
``ToleranceInputs.exclude_merged_from_pool``). The default, :data:`MERGED_HANDLING_DEFAULT`, comes from the
simulation in ``docs/sims/eclipse_tolerance/merged_handling_sim.py`` (sec. 10 of the doc).

Entry points: :func:`apply_merged_handling` on the rings of ``tools.mps_columns.build_rings`` (returns rings with
the halves inserted in tour order, the exclusions for ``analyze_arc_columns(exclude_clusters=...)`` and the per-pair
centre errors in the new cluster order), and :func:`centres_with_handling` for arrays (simulations, tests). Pure
functions: no file, no clock, no global random state.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_eclipse_tolerance as et

__all__ = [
    "MERGED_HANDLINGS", "MERGED_HANDLING_DEFAULT", "MERGED_HANDLING_ORIGIN", "check_handling", "two_mode_split",
    "split_cluster", "half_sigma_c_nm", "CentresWithHandling", "centres_with_handling", "MergedHandlingResult",
    "apply_merged_handling",
]

MERGED_HANDLINGS: Tuple[str, ...] = ("middle", "split", "exclude")
# The handling the program uses unless the caller asks for another one. The rule of decision D2 was "split becomes
# the default only if the simulation shows it keeps the 5 % level and gains power"; it does [simulation:
# docs/sims/eclipse_tolerance/merged_handling_sim.py, 1000 axons per cell, scalar tau 31 nm; docs/eclipse_tolerance.md
# sec. 10]: false "eclipsed" 2.4-4.4 % under independent rings for every handling; power with 30 % columns and 25 %
# fusions 0.390 split vs 0.347 middle (paired difference +0.043, 95 % CI +0.014 to +0.072) and 0.353 exclude; with
# 10 % fusions 0.517 vs 0.489 (+0.028, +0.001 to +0.055); with no fusion the three are the same. Before D2 the
# program used the middle centre.
MERGED_HANDLING_DEFAULT = "split"
MERGED_HANDLING_ORIGIN = ("[simulation: docs/sims/eclipse_tolerance/merged_handling_sim.py; user decision D2a] "
                          "merged clusters split in two for the eclipse test")


def check_handling(handling: Optional[str]) -> str:
    """The handling, validated (None -> :data:`MERGED_HANDLING_DEFAULT`); ValueError for anything else."""
    h = MERGED_HANDLING_DEFAULT if handling is None else str(handling).strip().lower()
    if h not in MERGED_HANDLINGS:
        raise ValueError(f"merged_handling must be one of {MERGED_HANDLINGS}, got {handling!r}")
    return h


# ===================================================================================================== the split
def _unit_ids(n: int, event: Optional[NDArray[np.int64]]) -> NDArray[np.int64]:
    """The unit of each localization: its event (an unlinked localization, -1, is its own event, as
    ``mps_eclipse_tolerance._raw_group``), or itself without events."""
    if event is None:
        return np.arange(n, dtype=np.int64)
    ev = np.asarray(event).astype(np.int64).reshape(-1)
    return np.where(ev >= 0, ev, -1 - np.arange(ev.size, dtype=np.int64))


def two_mode_split(t: NDArray[np.float64], event: Optional[NDArray[np.int64]] = None,
                   min_units_per_side: Optional[int] = None) -> Optional[NDArray[np.bool_]]:
    """
    The two halves of a cluster along the membrane: True for the localizations of the upper half. The SAME split as
    ``mps_eclipse_tolerance.two_mode_separation`` (exact 1D two-means of the units -- events, or localizations
    without events -- sorted by position, weighted by their localizations, each side with at least
    ``min_units_per_side`` units; default as ``_raw_group``: ``MERGED_MIN_UNITS_PER_SIDE`` with events, twice that
    without), so the halves are the two modes whose separation flagged the cluster. All localizations of one event
    fall on the same side. None when there are too few units.
    """
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    unit = _unit_ids(tt.size, None if event is None else np.asarray(event))
    k_min = int(min_units_per_side if min_units_per_side is not None else
                (et.MERGED_MIN_UNITS_PER_SIDE if event is not None else 2 * et.MERGED_MIN_UNITS_PER_SIDE))
    _, inv = np.unique(unit, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    counts = np.bincount(inv).astype(np.float64)
    means = np.bincount(inv, weights=tt) / counts
    order = np.argsort(means, kind="stable")
    m, w = means[order], counts[order]
    n = m.size
    if n < 2 * k_min:
        return None
    cw, cs = np.cumsum(w), np.cumsum(w * m)
    tot_w, tot_s = float(cw[-1]), float(cs[-1])
    k = np.arange(k_min, n - k_min + 1)
    w1, s1 = cw[k - 1], cs[k - 1]
    w2, s2 = tot_w - w1, tot_s - s1
    between = w1 * w2 / tot_w * (s2 / w2 - s1 / w1) ** 2
    cut = int(k[int(np.argmax(between))])         # units order[:cut] are the lower half
    upper_unit = np.zeros(n, dtype=bool)
    upper_unit[order[cut:]] = True
    return np.asarray(upper_unit[inv], dtype=bool)


def split_cluster(x: NDArray[np.float64], y: NDArray[np.float64], members: NDArray[np.intp],
                  event: Optional[NDArray[np.int64]], centre_nm: Sequence[float]
                  ) -> Optional[Tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """The two halves (indices into ``x`` / ``y``) of the cluster ``members``, split along the membrane tangent taken
    from the axon centre ``centre_nm`` (``mps_eclipse_tolerance.tangent_coordinates``); None when it cannot be
    split (too few units)."""
    mem = np.asarray(members, dtype=np.intp).reshape(-1)
    if mem.size < 2:
        return None
    t, _ = et.tangent_coordinates(np.asarray(x)[mem], np.asarray(y)[mem], [float(v) for v in centre_nm])
    ev = None if event is None else np.asarray(event)[mem]
    upper = two_mode_split(t, ev)
    if upper is None or not upper.any() or upper.all():
        return None
    return mem[~upper], mem[upper]


def half_sigma_c_nm(n_locs: Any, n_events: Any, tol: et.AxonTolerances
                    ) -> NDArray[np.float64]:
    """The centre error of each half the way ``axon_tolerances`` computes it for a whole cluster: the half's own
    localizations and events, the axon's label prior (``lambda_hat``) and the axon's footprint (``pool.rho_used_nm``,
    unchanged by the split)."""
    inp = tol.inputs
    nl = np.asarray(n_locs, dtype=np.float64).reshape(-1)
    ne = np.asarray(n_events, dtype=np.float64).reshape(-1)
    if tol.lambda_hat is None or not np.isfinite(ne).any():
        sw2: NDArray[np.float64] = np.ones(nl.size, dtype=np.float64)
    else:
        sw2 = et.sum_w2_given_events(ne, inp.c_label, float(tol.lambda_hat), inp.locs_per_event)
    return et.sigma_c_nm(nl, sw2, tol.pool.rho_used_nm, inp)


# ===================================================================================================== arrays
@dataclass
class CentresWithHandling:
    """The centres one group of clusters enters the eclipse test with: ``centres_nm`` (M, 2), ``owner`` (M,) the
    cluster each centre comes from (index into the caller's list), ``half`` (M,) 0 for a whole cluster, 1 / 2 for
    the halves of a split one, ``usable`` (M,) False for an excluded merged cluster (it keeps its centre so the
    caller's geometry is unchanged), ``n_locs`` / ``n_events`` per centre, and the counts."""
    centres_nm: NDArray[np.float64]
    owner: NDArray[np.int64]
    half: NDArray[np.int64]
    usable: NDArray[np.bool_]
    n_locs: NDArray[np.float64]
    n_events: NDArray[np.float64]
    handling: str
    n_merged: int
    n_split: int
    n_excluded: int
    n_split_failed: int


def _n_events(event: Optional[NDArray[np.int64]], idx: NDArray[np.intp]) -> float:
    if event is None:
        return float("nan")
    return float(np.unique(_unit_ids(idx.size, np.asarray(event)[idx])).size)


def centres_with_handling(x: Any, y: Any, members: Sequence[NDArray[np.intp]], merged: Any, handling: str, *,
                          event: Optional[Any] = None, centre_nm: Sequence[float] = (0.0, 0.0)
                          ) -> CentresWithHandling:
    """The centres of one group of clusters (``members``: the localization indices of each) under ``handling``,
    with ``merged`` the flags of ``mps_eclipse_tolerance.merged_cluster_flags`` (one per cluster). A flagged
    cluster that cannot be split (too few units) keeps its middle centre (counted in ``n_split_failed``)."""
    h = check_handling(handling)
    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    ev = None if event is None else np.asarray(event).astype(np.int64).reshape(-1)
    flags = np.asarray(merged, dtype=bool).reshape(-1)
    if flags.size != len(members):
        raise ValueError(f"centres_with_handling: {flags.size} merged flags for {len(members)} clusters")
    cen: List[List[float]] = []
    owner: List[int] = []
    half: List[int] = []
    usable: List[bool] = []
    nl: List[float] = []
    ne: List[float] = []
    n_split = n_excl = n_fail = 0
    for i, m in enumerate(members):
        mem = np.asarray(m, dtype=np.intp).reshape(-1)
        parts: Optional[Tuple[NDArray[np.intp], NDArray[np.intp]]] = None
        if flags[i] and h == "split":
            parts = split_cluster(xa, ya, mem, ev, centre_nm)
            if parts is None:
                n_fail += 1
            else:
                n_split += 1
        if parts is not None:
            for j, p in enumerate(parts):
                cen.append([float(np.mean(xa[p])), float(np.mean(ya[p]))])
                owner.append(i)
                half.append(j + 1)
                usable.append(True)
                nl.append(float(p.size))
                ne.append(_n_events(ev, p))
            continue
        excluded = bool(flags[i]) and h == "exclude"
        n_excl += int(excluded)
        cen.append([float(np.mean(xa[mem])), float(np.mean(ya[mem]))] if mem.size else [math.nan, math.nan])
        owner.append(i)
        half.append(0)
        usable.append(not excluded)
        nl.append(float(mem.size))
        ne.append(_n_events(ev, mem))
    return CentresWithHandling(
        centres_nm=np.asarray(cen, dtype=np.float64).reshape(-1, 2), owner=np.asarray(owner, dtype=np.int64),
        half=np.asarray(half, dtype=np.int64), usable=np.asarray(usable, dtype=bool),
        n_locs=np.asarray(nl, dtype=np.float64), n_events=np.asarray(ne, dtype=np.float64), handling=h,
        n_merged=int(flags.sum()), n_split=n_split, n_excluded=n_excl, n_split_failed=n_fail)


# ===================================================================================================== rings
@dataclass
class MergedHandlingResult:
    """The rings the eclipse test runs on under ``handling``: ``res`` (the input RingsResult itself for "middle" and
    "exclude"; a copy with every split cluster replaced by its two halves, inserted in tour order, for "split"),
    ``exclude_clusters`` (ring, cluster index) for ``analyze_arc_columns(exclude_clusters=...)``, ``pair_sigma_nm``
    (ring -> centre error of each cluster in the returned ``Ring.clusters`` order, for the per-pair rule), and the
    counts per axon."""
    handling: str
    res: Any
    exclude_clusters: List[Tuple[int, int]]
    pair_sigma_nm: Dict[int, NDArray[np.float64]]
    n_merged: int
    n_split: int
    n_excluded: int
    n_split_failed: int
    warnings: List[str] = field(default_factory=list)


def _split_ring(ring: Any, res: Any, tol_group: et.ClusterTolerances, tol: et.AxonTolerances
                ) -> Tuple[Any, NDArray[np.float64], int, int]:
    """One ring with its flagged clusters split: halves replace the cluster (first half at its index, second
    appended) and are inserted next to each other in the tour, in the order that keeps the tour short; the ring's
    contour and its analysis' perimeter order follow, so ``mps_matching.ring_geometry`` accepts them."""
    centre = getattr(ring, "centroid_nm", None)
    clusters = list(ring.clusters)
    k0 = len(clusters)
    sig: List[float] = [float(v) for v in np.asarray(tol_group.sigma_c_nm, dtype=np.float64).reshape(-1)]
    ev = None if res.event_id is None else np.asarray(res.event_id)
    if centre is None:
        good = [np.asarray(c.centroid_nm, dtype=np.float64) for c in clusters]
        centre = np.mean(good, axis=0) if good else np.zeros(2)
    second_of: Dict[int, int] = {}
    n_split = n_fail = 0
    for i, cl in enumerate(list(clusters)):
        if not bool(tol_group.merged_suspect[i]):
            continue
        idx = np.asarray(cl.loc_index, dtype=np.intp)
        parts = split_cluster(res.x_p, res.y_p, idx, ev, [float(v) for v in np.asarray(centre).ravel()])
        if parts is None:
            n_fail += 1
            continue
        n_split += 1
        halves = []
        zv = np.asarray(cl.z_values_nm, dtype=np.float64).reshape(-1)
        for p in parts:
            in_half = np.isin(idx, p)            # z_values_nm is in loc_index order, and so is each half
            z_half = zv[in_half] if zv.size == idx.size else zv
            n_ev = None if ev is None else int(_n_events(ev, p))
            halves.append(dataclasses.replace(
                cl, centroid_nm=np.array([float(np.mean(res.x_p[p])), float(np.mean(res.y_p[p]))]), n_locs=int(p.size),
                n_events=n_ev, loc_index=np.asarray(p, dtype=np.intp), z_values_nm=z_half, area_nm2=float("nan"),
                rho_nm=float("nan"), sigma_centroid_nm=float("nan"), sigma_centroid_x_nm=float("nan"),
                sigma_centroid_y_nm=float("nan"), n_bootstrap=0))
        hs = half_sigma_c_nm([h.n_locs for h in halves],
                             [np.nan if h.n_events is None else h.n_events for h in halves], tol)
        clusters[i] = halves[0]
        sig[i] = float(hs[0])
        clusters.append(halves[1])
        sig.append(float(hs[1]))
        second_of[i] = len(clusters) - 1
    if not second_of:
        return ring, np.asarray(sig, dtype=np.float64), 0, n_fail
    new_contour: Optional[NDArray[np.float64]] = None
    new_analysis = ring.analysis
    per = getattr(ring.analysis, "perimeter", None)
    if ring.contour_nm is not None and per is not None and np.asarray(per.order).size == k0:
        order = [int(v) for v in np.asarray(per.order).ravel()]
        cents = np.array([np.asarray(c.centroid_nm, dtype=np.float64).reshape(2) for c in clusters])
        new_order: List[int] = []
        for pos, i in enumerate(order):
            if i not in second_of:
                new_order.append(i)
                continue
            prev = cents[order[pos - 1]]
            a, b = i, second_of[i]
            # the half nearer the previous vertex comes first
            if np.hypot(*(cents[b] - prev)) < np.hypot(*(cents[a] - prev)):
                a, b = b, a
            new_order.extend([a, b])
        new_contour = cents[np.asarray(new_order, dtype=np.intp)]
        old_pc = np.asarray(per.contour, dtype=np.float64).reshape(-1, 2)
        old_cents = np.array([np.asarray(c.centroid_nm, dtype=np.float64).reshape(2) for c in ring.clusters])
        # the perimeter's own contour may be in the ring analysis' frame: move each vertex by the half's offset
        offset = old_pc[np.argsort(order)] - old_cents if old_pc.shape == (k0, 2) else np.zeros((k0, 2))
        owner = list(range(k0)) + [i for i in sorted(second_of, key=lambda v: second_of[v])]
        pc = np.array([cents[j] + offset[owner[j]] for j in new_order], dtype=np.float64)
        new_per = dataclasses.replace(per, order=np.asarray(new_order, dtype=np.intp), contour=pc,
                                      n_clusters=len(clusters))
        new_analysis = copy.copy(ring.analysis)
        new_analysis.perimeter = new_per
    return (dataclasses.replace(ring, clusters=clusters, contour_nm=new_contour, analysis=new_analysis),
            np.asarray(sig, dtype=np.float64), n_split, n_fail)


def apply_merged_handling(res: Any, tol: et.AxonTolerances, handling: Optional[str] = None
                          ) -> MergedHandlingResult:
    """The rings of ``res`` (``build_rings`` / ``clean_rings``) for the eclipse test under ``handling`` (None: the
    default), given ``tol`` = ``tolerances_from_rings(res, ...)`` (its ``merged_suspect`` flags, centre errors,
    label prior and footprint). The footprint pool is NOT recomputed: the split changes which centres enter the
    test, not the axon's footprint."""
    h = check_handling(handling)
    sig: Dict[int, NDArray[np.float64]] = {int(str(k)): np.asarray(g.sigma_c_nm, dtype=np.float64)
                                           for k, g in tol.groups.items()}
    n_merged = int(sum(int(np.count_nonzero(g.merged_suspect)) for g in tol.groups.values()))
    warnings: List[str] = []
    if h == "middle" or n_merged == 0:
        return MergedHandlingResult(h, res, [], sig, n_merged, 0, 0, 0, warnings)
    if h == "exclude":
        excl = [(int(str(k)), int(i)) for k, g in tol.groups.items() for i in np.flatnonzero(g.merged_suspect)]
        return MergedHandlingResult(h, res, excl, sig, n_merged, 0, len(excl), 0, warnings)
    rings = []
    n_split = n_fail = 0
    for ring in res.rings:
        g = tol.groups.get(int(ring.index))
        if g is None or not np.any(g.merged_suspect):
            rings.append(ring)
            continue
        if g.n_clusters != len(ring.clusters):
            raise ValueError(f"apply_merged_handling: ring {ring.index} has {len(ring.clusters)} clusters, the "
                             f"tolerances {g.n_clusters}: they must come from the same rings")
        new_ring, s, ns, nf = _split_ring(ring, res, g, tol)
        rings.append(new_ring)
        sig[int(ring.index)] = s
        n_split += ns
        n_fail += nf
    if n_fail:
        warnings.append(f"{n_fail} merged-looking cluster(s) had too few events to split: kept with one centre")
    return MergedHandlingResult(h, dataclasses.replace(res, rings=rings), [], sig, n_merged, n_split, 0, n_fail,
                                warnings)
