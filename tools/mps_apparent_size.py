# -*- coding: utf-8 -*-
"""
Apparent size of the clusters of ONE clustered result, and the per-axon footprint (clustering plan A6 and the
recommendations of the apparent-size simulation; ``docs/apparent_size.md``).

What "size" means here (the user's decision 1a of 2026-10-22)
-------------------------------------------------------------
A cluster in a dSTORM image is a cloud of localizations seen through four layers: the footprint (the nm of membrane
its epitopes occupy), the labels (1-5 antibody stacks landing at random on it), the linkage (a fixed ~10 nm offset
per label) and the precision (each localization ~8-14 nm off; the file's NeNA). So every per-cluster number is an
APPARENT size = footprint (+) linkage (+) precision, and with one label per cluster the footprint is invisible.

* PRIMARY: the CONTENT of the cluster -- its events (blinks linked in time) when the movie's frames and a NeNA are
  available, else its localizations -- labelled so. It is what is really measured per cluster.
* BESIDE IT, per axon: the footprint rho_pool (method of moments over the axon's clusters, deconvolving the
  precision and the linkage), only when identifiable (enough labels per cluster).
* The convex-hull area ("Cluster area", ``tools.mps_geometry.compute_cluster_areas``) and its r_eff stay exactly as
  they are, as the legacy columns; this module never reports r_eff as a size. Decision D6: two measures that do not
  grow with the number of localizations are ADDED beside them:

  - ``cloud_sd``: the SD of the localization cloud, the root mean square of its SDs along two perpendicular axes,
    sqrt(trace(S) / 2) with S the sample covariance (ddof 1): rotation-invariant, needs no membrane frame, = sigma
    for an isotropic cloud, and its variance part is unbiased at every n. A standard deviation does not grow with
    the number of points it is estimated from (E[s] / sigma = 0.94 at n = 5, 0.97 at 10, 0.99 at 20: the c4 bias
    [lit: Gurland & Tripathi 1971]). Not the geometric mean sqrt(sd_major sd_minor) of the principal axes (the
    simulation's r_disp in the membrane frame): det(S) subtracts the chance correlation of a small sample, and that
    version correlates with N_loc 0.38 against 0.18 for this one in a one-label population [simulation:
    docs/sims/apparent_size/size_probes.py, probe "cloud_sd_definition"]. The principal SDs are exported too.
  - the hull at FIXED N (rarefaction): the mean convex-hull area of ``RAREFY_N`` = 15 localizations drawn without
    replacement, over ``RAREFY_REPS`` = 20 draws, and the radius of that area (``r_rare``). The hull of n points
    grows with n [lit: Efron 1965; Renyi & Sulanke 1963 for the number of vertices]; at fixed n it does not.
    Undefined (flagged) below 15
    localizations.

Every estimator is a pure function of the labels of whatever clustering method produced them (``-1`` = noise), the
localizations and, when given, a closed curve along the membrane (for the tangent / normal frame). No Qt.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.stats import rankdata

__all__ = [
    "RAREFY_N", "RAREFY_REPS", "ELONGATION_FLAG", "CLOSE_FLAG_NM", "FEW_LOCS_FLAG", "SIGMA_LINK_NM", "C_LABEL",
    "IDENTIFIABLE_LABELS", "POOL_MIN_CLUSTERS", "POOL_EXCLUDE_ELONGATION", "N_PERMUTATIONS", "REGIME_LEVELS", "CONTENT_LEVELS",
    "ClusterSizes", "cluster_sizes", "hull_area", "rarefied_hull", "kept_cluster_labels", "events_per_cluster",
    "FootprintPool", "pooled_footprint", "sum_w2_given_events", "spearman", "fisher_z", "partial_spearman",
    "MarkPermutation", "mark_permutation_diagnostic", "summary_median", "summary_fraction", "as_float",
]

# ---------------------------------------------------------------------------------------------------- defaults
# Localizations of the hull at fixed N [simulation: SIZE_SIM report sec. 9.8, approved answer 8 of 2026-10-22:
# n = 15; the plan's first proposal of 20 was ad hoc]. Fifteen keeps the 1-label clusters that have >= 15
# localizations (41 % of the clusters with one label have fewer: they are flagged "r_rare undefined").
RAREFY_N = 15
# Draws averaged per cluster [simulation: SIZE_SIM sec. 3, 20 draws; the spread of the mean is ~1/sqrt(20) of one
# draw's].
RAREFY_REPS = 20
# Tangential elongation sd_t / sd_n above which a cluster is flagged "elongated" (candidate fusion) [simulation:
# SIZE_SIM sec. 5.1: 92 % of DBSCAN's fused clusters, 5 % false alarms with 3 labels].
ELONGATION_FLAG = 1.8
# Distance to the nearest cluster below which a cluster is flagged "close" (candidate split) [simulation: SIZE_SIM
# sec. 5, flag d_nn < 50 nm].
CLOSE_FLAG_NM = 50.0
# Localizations below which a cluster is flagged "few localizations" (detection margin) [simulation: SIZE_SIM
# sec. 5, flag N_loc >= 20].
FEW_LOCS_FLAG = 20
# Linkage error per axis [ad hoc, of the order of Frueh et al. 2021 (mean displacement 10.0 nm, radial sigma ~7.5 nm,
# maximum ~19 nm; not a per-axis value of that paper); adopted, not measured].
SIGMA_LINK_NM = 10.0
# Blinks per label [lit: Dempsey et al. 2011, MEA, per fluorophore on glass; other measurements 4-17 per label,
# docs/dbscan_rule.md 2.1; provisional, decision D1; 26 (beta-ME) as sensitivity].
C_LABEL = 14.0
# Mean labels per cluster (events / c_label) from which the footprint pool is identifiable [simulation: SIZE_SIM
# sec. 8.2 "extra" and the pool code: below 1.5 the footprint is not measurable, only the content]. The pool itself
# is tools.footprint_pool's since 2026-10-29 (one definition with the eclipse tolerance, finding q).
IDENTIFIABLE_LABELS = 1.5
# Elongation above which a cluster is left out of the footprint pool (a probable fusion) [simulation:
# docs/sims/apparent_size/size_probes.py, probe "footprint_pool": with DBSCAN eps 25 / 10 and 2-3 labels, keeping
# every cluster inflates the pool by 8-18 nm (fusions), leaving out the "elongated" flag's 1.8 truncates the
# tangential spread and biases it 8-11 nm low; 2.5 lands within ~1-4 nm of the pool of the TRUE clusters, which is
# what the simulation computed (it excluded the true fusions, which the program cannot know)]. Separate from
# ELONGATION_FLAG on purpose: a flag is a mark for the reader, this is a cut that changes a number.
POOL_EXCLUDE_ELONGATION = 2.5
# Fewest clusters the pool is computed from [ad hoc: tools.footprint_pool.MIN_POOL_CLUSTERS, 10 -- the stricter of
# the two modules' ad hoc values (this one's was 5, the simulation's code), finding q, 2026-10-29].
POOL_MIN_CLUSTERS = 10
# Mark permutations of the diagnostic [lit: Davison & Hinkley 1997 / the user's decision of 199 replicates (A9,
# P-R78)].
N_PERMUTATIONS = 199

REGIME_LEVELS: Tuple[str, ...] = ("footprint measurable", "content only", "not evaluated")
CONTENT_LEVELS: Tuple[str, ...] = ("events", "localizations")


# ---------------------------------------------------------------------------------------------------- helpers
def kept_cluster_labels(labels: NDArray[np.int64], exclude: Optional[Sequence[int]] = None) -> NDArray[np.int64]:
    """The cluster labels (sorted, >= 0) not in ``exclude``: the clusters the size table describes. For an
    ``AxonAnalysis`` pass ``exclude`` = its bad labels and its discarded labels, the clusters "Cluster area"
    describes (``tools.axon_export._by_label``)."""
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    out = np.unique(lab[lab >= 0])
    if exclude:
        out = out[~np.isin(out, np.asarray(sorted(int(v) for v in exclude), dtype=np.int64))]
    return out.astype(np.int64)


def hull_area(pts: NDArray[np.float64]) -> float:
    """Area of the convex hull of (n, 2) points by Andrew's monotone chain and the shoelace formula: the polygon of
    ``scipy.spatial.ConvexHull`` without its per-call cost (thousands of 15-point hulls per axon). 0 for fewer than
    three points or collinear points (as ``compute_cluster_areas`` gives a degenerate cluster)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    order = np.lexsort((p[:, 1], p[:, 0]))
    seq = p[order].tolist()

    def half(points: List[List[float]]) -> List[List[float]]:
        h: List[List[float]] = []
        for x, y in points:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (y - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (x - h[-2][0])) <= 0:
                h.pop()
            h.append([x, y])
        return h

    lower = half(seq)
    upper = half(seq[::-1])
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    H = np.asarray(hull, dtype=np.float64)
    return 0.5 * abs(float(np.dot(H[:, 0], np.roll(H[:, 1], -1)) - np.dot(H[:, 1], np.roll(H[:, 0], -1))))


def rarefied_hull(pts: NDArray[np.float64], rng: np.random.Generator, n0: int = RAREFY_N,
                  reps: int = RAREFY_REPS) -> Tuple[float, float]:
    """(mean hull area, mean hull radius sqrt(area / pi)) of ``n0`` points drawn without replacement, over ``reps``
    draws; (NaN, NaN) when the cloud has fewer than ``n0`` points. The radius is averaged per draw (the
    simulation's r_rare), not taken from the mean area."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < n0:
        return float("nan"), float("nan")
    a_sum = 0.0
    r_sum = 0.0
    for _ in range(int(reps)):
        a = hull_area(p[rng.choice(p.shape[0], int(n0), replace=False)])
        a_sum += a
        r_sum += math.sqrt(a / math.pi)
    return a_sum / reps, r_sum / reps


def _curve_tangents(curve: NDArray[np.float64]) -> Tuple[NDArray[np.float64], cKDTree]:
    c = np.asarray(curve, dtype=np.float64).reshape(-1, 2)
    d = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.where(nrm > 0, nrm, 1.0), cKDTree(c)


def events_per_cluster(frame: NDArray[np.int64], x: NDArray[np.float64], y: NDArray[np.float64],
                       labels: NDArray[np.int64], nena_nm: float,
                       radius_in_nena: Optional[float] = None) -> Dict[int, int]:
    """Events (blinks linked in time: radius 3 NeNA, 1 dark frame, decision D5) of every cluster of ``labels``, by
    ``tools.mps_events.cluster_events`` (the event layer A3). Labels already set to -1 are not counted."""
    from tools import mps_events
    ce = mps_events.cluster_events(np.asarray(frame), np.asarray(x, dtype=np.float64),
                                   np.asarray(y, dtype=np.float64), np.asarray(labels, dtype=np.int64),
                                   nena_nm=float(nena_nm),
                                   radius_in_nena=(mps_events.LINK_RADIUS_IN_NENA if radius_in_nena is None
                                                   else float(radius_in_nena)))
    return {int(c): int(n) for c, n in zip(ce.cluster_ids, ce.n_events)}


# ---------------------------------------------------------------------------------------------------- per cluster
@dataclass
class ClusterSizes:
    """Per-cluster apparent sizes of one clustered result, one entry per kept label (``labels``, sorted). Lengths
    in nm, areas in nm^2; NaN where a value is not defined (and the reason is a flag or ``frame``).

    ``frame`` is "membrane" when a closed curve gave the (tangent, normal) frame (``sd_tangential``,
    ``sd_normal``, ``elongation`` defined) and "none" otherwise."""
    labels: NDArray[np.int64]
    n_locs: NDArray[np.int64]
    n_events: NDArray[np.float64]
    centroid: NDArray[np.float64]
    sd_major: NDArray[np.float64]
    sd_minor: NDArray[np.float64]
    cloud_sd: NDArray[np.float64]
    radius_gyration: NDArray[np.float64]
    sd_tangential: NDArray[np.float64]
    sd_normal: NDArray[np.float64]
    elongation: NDArray[np.float64]
    hull_area_n15: NDArray[np.float64]
    r_rare: NDArray[np.float64]
    r_rare_undefined: NDArray[np.bool_]
    d_nn: NDArray[np.float64]
    flag_elongated: NDArray[np.bool_]
    flag_close: NDArray[np.bool_]
    flag_few_locs: NDArray[np.bool_]
    frame: str = "none"
    rarefy_n: int = RAREFY_N
    warnings: List[str] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return int(self.labels.size)

    @property
    def has_events(self) -> bool:
        return bool(self.n_events.size) and bool(np.isfinite(self.n_events).all())

    @property
    def content(self) -> NDArray[np.float64]:
        """The PRIMARY size: events when every cluster has them, else localizations (``content_unit`` says)."""
        return self.n_events if self.has_events else self.n_locs.astype(np.float64)

    @property
    def content_unit(self) -> str:
        return CONTENT_LEVELS[0] if self.has_events else CONTENT_LEVELS[1]

    @property
    def flagged_any(self) -> NDArray[np.bool_]:
        return self.flag_elongated | self.flag_close | self.flag_few_locs


def cluster_sizes(x: NDArray[np.float64], y: NDArray[np.float64], labels: NDArray[np.int64], *,
                  exclude: Optional[Sequence[int]] = None, n_events: Optional[Mapping[int, int]] = None,
                  event_id: Optional[NDArray[np.int64]] = None, curve: Optional[NDArray[np.float64]] = None,
                  rarefy_n: int = RAREFY_N, rarefy_reps: int = RAREFY_REPS, seed: int = 0,
                  elongation_flag: float = ELONGATION_FLAG, close_flag_nm: float = CLOSE_FLAG_NM,
                  few_locs_flag: int = FEW_LOCS_FLAG) -> ClusterSizes:
    """Every apparent-size estimator of every kept cluster (the module docstring says which and why).

    ``labels`` from any clustering method (-1 noise); ``exclude`` the labels the curation removed. Content: events
    from ``n_events`` (label -> events, e.g. :func:`events_per_cluster`) or from ``event_id`` (one id per
    localization: the distinct ids in a cluster), else NaN (the content is then the localizations). ``curve``: a
    closed (M, 2) polyline along the membrane (the 400 nm P-spline of ``tools.mps_curvature``, or the contour
    through the centroids); the tangent at the vertex nearest each centroid sets the frame of ``sd_tangential`` /
    ``sd_normal`` / ``elongation``. The rarefied hull draws from ``default_rng([seed, label])``, so a cluster's
    value does not depend on the other clusters (nor on which other methods ran). ``d_nn`` is the 2D distance
    between kept centroids (on a ring, the chord: within 1 % of the arc for neighbours < 200 nm apart on a ring of
    radius >= 600 nm)."""
    xs = np.asarray(x, dtype=np.float64).reshape(-1)
    ys = np.asarray(y, dtype=np.float64).reshape(-1)
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not (xs.size == ys.size == lab.size):
        raise ValueError("cluster_sizes: x, y and labels must have the same length")
    keep = kept_cluster_labels(lab, exclude)
    k = int(keep.size)
    nan = np.full(k, np.nan)
    out = ClusterSizes(
        labels=keep, n_locs=np.zeros(k, dtype=np.int64), n_events=nan.copy(), centroid=np.full((k, 2), np.nan),
        sd_major=nan.copy(), sd_minor=nan.copy(), cloud_sd=nan.copy(), radius_gyration=nan.copy(),
        sd_tangential=nan.copy(), sd_normal=nan.copy(), elongation=nan.copy(), hull_area_n15=nan.copy(),
        r_rare=nan.copy(), r_rare_undefined=np.zeros(k, dtype=bool), d_nn=nan.copy(),
        flag_elongated=np.zeros(k, dtype=bool), flag_close=np.zeros(k, dtype=bool),
        flag_few_locs=np.zeros(k, dtype=bool), frame="none", rarefy_n=int(rarefy_n))
    if k == 0:
        return out
    tang: Optional[NDArray[np.float64]] = None
    tree: Optional[cKDTree] = None
    if curve is not None and np.asarray(curve).reshape(-1, 2).shape[0] >= 3:
        tang, tree = _curve_tangents(np.asarray(curve, dtype=np.float64))
        out.frame = "membrane"
    ev_ids = None if event_id is None else np.asarray(event_id).reshape(-1)
    if ev_ids is not None and ev_ids.size != lab.size:
        raise ValueError("cluster_sizes: event_id must have one entry per localization")
    order = np.argsort(lab, kind="stable")
    sorted_lab = lab[order]
    starts = np.searchsorted(sorted_lab, keep, side="left")
    ends = np.searchsorted(sorted_lab, keep, side="right")
    for i, c in enumerate(keep.tolist()):
        idx = order[starts[i]:ends[i]]
        pts = np.column_stack([xs[idx], ys[idx]])
        n = int(idx.size)
        out.n_locs[i] = n
        cen = pts.mean(axis=0)
        out.centroid[i] = cen
        if n_events is not None and int(c) in n_events:
            out.n_events[i] = float(n_events[int(c)])
        elif ev_ids is not None:
            out.n_events[i] = float(np.unique(ev_ids[idx]).size)
        if n >= 2:
            cov = np.cov(pts.T, ddof=1)
            ev = np.clip(np.linalg.eigvalsh(cov), 0.0, None)        # ascending
            out.sd_minor[i] = math.sqrt(float(ev[0]))
            out.sd_major[i] = math.sqrt(float(ev[1]))
            out.cloud_sd[i] = math.sqrt(float(ev[0] + ev[1]) / 2.0)
            out.radius_gyration[i] = math.sqrt(float(ev[0] + ev[1]))
            if tang is not None and tree is not None:
                _, j = tree.query(cen)
                tg = tang[int(j)]
                nm = np.array([tg[1], -tg[0]])
                d = pts - cen
                st = float(np.std(d @ tg, ddof=1))
                sn = float(np.std(d @ nm, ddof=1))
                out.sd_tangential[i] = st
                out.sd_normal[i] = sn
                out.elongation[i] = st / sn if sn > 0 else float("nan")
        a, r = rarefied_hull(pts, np.random.default_rng([int(seed), int(c) & 0x7FFFFFFF]), int(rarefy_n),
                             int(rarefy_reps))
        out.hull_area_n15[i] = a
        out.r_rare[i] = r
        out.r_rare_undefined[i] = n < int(rarefy_n)
    if k >= 2:
        dist, _ = cKDTree(out.centroid).query(out.centroid, k=2)
        out.d_nn = np.asarray(dist[:, 1], dtype=np.float64)
    out.flag_elongated = np.where(np.isfinite(out.elongation), out.elongation > float(elongation_flag), False)
    out.flag_close = np.where(np.isfinite(out.d_nn), out.d_nn < float(close_flag_nm), False)
    out.flag_few_locs = out.n_locs < int(few_locs_flag)
    if out.frame == "none":
        out.warnings.append("no membrane curve given: tangential / normal SDs, elongation and its flag not computed")
    n_undef = int(out.r_rare_undefined.sum())
    if n_undef:
        out.warnings.append(f"{n_undef} of {k} clusters have fewer than {int(rarefy_n)} localizations: r_rare "
                            "undefined for them")
    return out


# ---------------------------------------------------------------------------------------------------- the pool
def sum_w2_given_events(n_events: NDArray[np.float64], c_label: float = C_LABEL,
                        locs_per_event: float = 2.0) -> NDArray[np.float64]:
    """Plug-in E[sum w^2 | events] of each cluster, the label prior's rate taken from the mean events per cluster
    (P-R61 sec. 3.2), from the shared table of ``tools.footprint_pool`` (2026-10-29; this module's own copy, with a
    lambda grid from 0, is retired)."""
    from tools import footprint_pool as fp
    ev = np.asarray(n_events, dtype=np.float64).reshape(-1)
    mean_ev = float(np.mean(ev)) if ev.size else c_label
    lam = fp.lambda_from_mean_labels(max(1.0, mean_ev / c_label))
    return fp.sum_w2_given_events(ev, c_label, lam, locs_per_event)


@dataclass
class FootprintPool:
    """The per-axon footprint rho_pool (nm) and its jackknife standard error; ``labels_per_cluster`` the mean
    labels per cluster estimated as mean events / c_label; ``verdict`` one of ``REGIME_LEVELS``; ``reason`` why it
    is not a number when it is not."""
    rho_nm: Optional[float]
    se_nm: Optional[float]
    labels_per_cluster: Optional[float]
    n_used: int
    verdict: str
    reason: str = ""
    sigma_loc_nm: Optional[float] = None
    sigma_link_nm: float = SIGMA_LINK_NM
    c_label: float = C_LABEL


def pooled_footprint(sd_tangential: NDArray[np.float64], n_locs: NDArray[np.float64],
                     n_events: Optional[NDArray[np.float64]], nena_nm: Optional[float], *,
                     sigma_link_nm: float = SIGMA_LINK_NM, c_label: float = C_LABEL, locs_per_event: float = 2.0,
                     exclude: Optional[NDArray[np.bool_]] = None, min_clusters: int = POOL_MIN_CLUSTERS,
                     identifiable_labels: float = IDENTIFIABLE_LABELS,
                     exclude_event_fusion: bool = True) -> FootprintPool:
    """The axon's footprint by the method of moments (P-R61 sec. 3.2), by the ONE definition the eclipse tolerance
    uses too (``tools.footprint_pool.footprint_pool``; finding q of the design review, 2026-10-29):

        sigma_L^2 = sum_i (s_t,i^2 - sigma_loc^2) (N_i - 1) / N_i  /  sum_i (1 - E[sum w^2 | events_i])
        rho^2     = 4 (sigma_L^2 - sigma_link^2)

    with s_t the SD of each cluster ALONG the membrane, sigma_loc the file's NeNA and sigma_link the linkage. Not
    computed (a reason instead) without events, without a NeNA, without the tangential frame, with fewer than
    ``min_clusters`` usable clusters, or when mean events / c_label of the pooled clusters < ``identifiable_labels``
    ("content only"). ``exclude``: the caller's fusion proxy (the plug-in: elongation > ``POOL_EXCLUDE_ELONGATION``);
    ``exclude_event_fusion``: also leave out the event layer's probable fusions (user decision E2, on). Standard
    error: delete-one jackknife [lit: Efron & Tibshirani 1993, ch. 11]. Ignores NeNA as covariate on purpose: the
    pool already deconvolves it (decision 6a)."""
    from tools import footprint_pool as fp
    p = fp.footprint_pool(sd_tangential, n_locs, n_events, nena_nm, sigma_link_nm=sigma_link_nm, c_label=c_label,
                          locs_per_event=locs_per_event, exclude=exclude, exclude_event_fusion=exclude_event_fusion,
                          min_clusters=min_clusters, identifiable_labels=identifiable_labels)
    return FootprintPool(p.rho_nm, p.se_nm, p.labels_per_cluster, int(p.n_used), p.verdict, p.reason,
                         p.sigma_loc_nm, float(sigma_link_nm), float(c_label))


# ---------------------------------------------------------------------------------------------------- statistics
def spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> Tuple[Optional[float], int]:
    """Spearman's rank correlation over the pairs where both are finite, and that n; None below 4 pairs or with a
    constant variable."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 4:
        return None, n
    rx = rankdata(x[ok])
    ry = rankdata(y[ok])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return (float((rx * ry).sum() / d) if d > 0 else None), n


def fisher_z(r: Optional[float], n: int) -> Optional[float]:
    """atanh(r) sqrt(n - 3): the per-axon z combined across axons (axon as the unit; decision 5a) [lit: Fisher
    1915]. None without r or with n < 4."""
    if r is None or n < 4:
        return None
    return float(math.atanh(max(-0.999, min(0.999, float(r)))) * math.sqrt(n - 3))


def partial_spearman(a: NDArray[np.float64], b: NDArray[np.float64], control: NDArray[np.float64]
                     ) -> Tuple[Optional[float], int]:
    """Spearman of ``a`` and ``b`` with ``control`` partialled out (Pearson of the rank residuals after a linear fit
    on the rank of ``control``). Used for the APPARENT estimators, which must be read beside log N_loc (decision
    6a): within an axon the NeNA is one number, so N_loc is the covariate left."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    z = np.asarray(control, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = int(ok.sum())
    if n < 5:
        return None, n
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    Z = np.column_stack([np.ones(n), rz])

    def resid(v: NDArray[np.float64]) -> NDArray[np.float64]:
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return np.asarray(v - Z @ beta, dtype=np.float64)

    ex, ey = resid(rx), resid(ry)
    d = math.sqrt(float((ex ** 2).sum() * (ey ** 2).sum()))
    return (float((ex * ey).sum() / d) if d > 0 else None), n


@dataclass
class MarkPermutation:
    """The mark-permutation DIAGNOSTIC: observed Spearman, its z against ``n_permutations`` reassignments of the
    sizes among the observed positions, the two-sided permutation p [lit: Phipson & Smyth 2010], n."""
    rho: Optional[float]
    z: Optional[float]
    p: Optional[float]
    n: int
    n_permutations: int
    note: str = ("DIAGNOSTIC, not a test of the biology: reassigning sizes among the detected positions does not "
                 "control the clustering, which changes size AND position together (fusions, rescued small "
                 "clusters). Under a true null with DBSCAN it rejects far above 5 % (apparent-size simulation, "
                 "scenario B: 100 % with 18 axons). The test is the pipeline null.")


def mark_permutation_diagnostic(size: NDArray[np.float64], covariate: NDArray[np.float64], rng: np.random.Generator,
                                n_permutations: int = N_PERMUTATIONS) -> MarkPermutation:
    """Spearman(size, covariate) against ``n_permutations`` random reassignments of the sizes (the null the user
    first proposed). Kept only as a labelled diagnostic (decision 2a): see ``MarkPermutation.note``."""
    x = np.asarray(covariate, dtype=np.float64).reshape(-1)
    y = np.asarray(size, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    B = int(n_permutations)
    if n < 6:
        return MarkPermutation(None, None, None, n, B)
    rx = rankdata(x[ok])
    ry = rankdata(y[ok])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    if d == 0:
        return MarkPermutation(None, None, None, n, B)
    obs = float((rx * ry).sum() / d)
    perm = np.array([float((rx * rng.permutation(ry)).sum() / d) for _ in range(B)])
    sd = float(perm.std(ddof=1))
    z = (obs - float(perm.mean())) / sd if sd > 0 else None
    p = float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (B + 1))
    return MarkPermutation(obs, z, p, n, B)


def summary_median(values: NDArray[np.float64]) -> Optional[float]:
    """Median of the finite values; None when there are none."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def summary_fraction(flags: NDArray[np.bool_]) -> Optional[float]:
    f = np.asarray(flags, dtype=bool).reshape(-1)
    return float(f.mean()) if f.size else None


def as_float(v: Any) -> Optional[float]:
    """A finite float or None (never NaN as a stand-in: the multi-method contract)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
