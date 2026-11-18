# -*- coding: utf-8 -*-
"""
The eclipse tolerance from each cluster's own centre error (P-R61; branch ``feat/eclipse-tolerance``).

Two clusters of consecutive MPS rings are "eclipsed" (in the same column) when their centres coincide along the
membrane within the error with which each centre is known. The program has two rules for "within the error"; both
are always computed and shown side by side:

* **scalar (PRIMARY today)**: |Δs| <= tau, one tolerance for every pair, the pre-specified ``tau0_nm`` of
  ``config/columns_params.yaml`` = 31 nm = 2.77 x 11.3 nm [simulation: P-R61 derivation; 2.77 = 1.96 sqrt(2), 11.3
  nm the per-axis centre error of a 3-label cluster on a 25 nm footprint with geometric event counts].
* **per-pair (EXPLORATORY today)**: |Δs_ij| <= t_ij = 1.96 sqrt(sigma_c,i^2 + sigma_c,j^2), with sigma_c of EACH
  cluster from its own localizations and events (option D of P-R61). It becomes the primary rule only when a
  calibration of c_label (events per antibody stack, P-R75) exists, with its date recorded
  (:meth:`ToleranceInputs.primary_rule`); until then it is labelled exploratory.

The per-cluster centre error (P-R61 S2.4; verified to 0.3 nm in 29 simulated scenarios)::

    sigma_c^2 = sigma_loc^2 / N_loc + (sigma_link^2 + rho^2 / 4) * E[sum w^2 | events] + sigma_reg^2

where N_loc are the cluster's localizations, sum w^2 the sum of squared shares of its localizations per label
(1 for one label), estimated from its number of emission EVENTS through a simulated table that needs c_label
(:func:`sum_w2_table`), and rho the footprint of the cluster on the membrane. **The footprint of one cluster cannot
be measured** for this purpose (1 to 5 labels per cluster: invisible with one label, +-100 % with three; P-R61 S3,
the user's decision 1a). It is measured once per AXON by pooling every cluster (method of moments,
:func:`pool_footprint`), and only when the clusters carry two or more labels on average; otherwise 25 nm, an upper
bound [lit: Gazal et al. 2026], is used as primary and 0 nm as sensitivity -- a DECISION, not a measurement
(decision 3a). Merged clusters (two real clusters linked into one by the clustering) inflate the pool; a criterion
that excludes them (:func:`merged_cluster_flags`) is implemented and validated on synthetic data, and until the
user accepts it as a measurement the pool is labelled an UPPER BOUND (decision 4).

Method-agnostic: everything here takes a cluster as the indices of its localizations (from ANY clustering method's
labels: :func:`clusters_from_labels`), the localizations' coordinates and, when frames exist, their event ids.
``tools.mps_columns.build_rings(cluster_method=..., cluster_params=...)`` builds the rings with any method; the
plug-in :func:`eclipse_analysis` (registered with ``tools.multimethod`` as ``"eclipse"``, readouts ``eclipse_*``)
gives the per-method readouts the integration compares.

Nothing here reads a file, a clock or global random state; the only randomness (the pool's bootstrap and the
simulated sum-w^2 table) comes from fixed seeds or the caller's generator.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "K_ARC", "K_PLANE", "K_SCALAR", "SIGMA_C_REFERENCE_NM", "TAU_SCALAR_DERIVED_NM", "SIGMA_LOC_DEFAULT_NM",
    "SIGMA_LINK_DEFAULT_NM", "SIGMA_REG_NM", "C_LABEL_PRESETS", "C_LABEL_DEFAULT", "LOCS_PER_EVENT_DEFAULT",
    "RHO_UPPER_NM", "RHO_SENSITIVITY_NM", "MIN_MEAN_LABELS_FOR_POOL", "MIN_POOL_CLUSTERS", "POOL_BOOTSTRAP",
    "MERGED_Z", "MERGED_MIN_UNITS_PER_SIDE", "SUM_W2_SEED", "SUM_W2_DRAWS", "LAMBDA_GRID", "EVENT_EDGES",
    "RHO_SOURCES", "SIGMA_C_SOURCES", "RULE_SCALAR", "RULE_PER_PAIR",
    "ToleranceInputs", "ClusterTolerances", "FootprintPool", "AxonTolerances",
    "clusters_from_labels", "tangent_coordinates", "two_mode_separation", "merged_threshold_nm",
    "merged_cluster_flags", "sum_w2_table", "lambda_from_mean_labels", "sum_w2_given_events", "sigma_c_nm",
    "pool_footprint", "axon_tolerances", "tolerances_from_rings", "pair_tolerance_matrix", "scalar_tau_from_sigma",
    "per_pair_arc_columns", "CLUSTER_COLUMNS", "PAIR_COLUMNS", "AXON_COLUMNS", "COLUMN_DESCRIPTIONS", "describe",
    "cluster_rows", "pair_rows", "axon_row", "summary_lines", "ECLIPSE_ANALYSIS", "ECLIPSE_READOUTS",
    "eclipse_analysis", "register",
]

# ===================================================================================================== constants
# The k of the per-pair arc rule: 95 % of the difference of two independent normal centre errors along the arc
# [derivation: the two-sided 95 % quantile of the standard normal]. The same constant as tools.mps_unroll.PAIR_K_ARC.
K_ARC = 1.96
# The 95 % radius of a 2D centre error with per-axis sigma: sqrt(-2 ln 0.05) = 2.448 [derivation: Rayleigh
# quantile]. ``centroid_r95_nm`` = K_PLANE * sigma_c (the circle that holds the true centre 95 % of the time), and
# the k of the 2D per-pair sensitivity rule.
K_PLANE = math.sqrt(-2.0 * math.log(0.05))
# The scalar rule's k on ONE typical centre error: 1.96 sqrt(2) = 2.77 [derivation: two equal centre errors].
K_SCALAR = 1.96 * math.sqrt(2.0)
# The per-axis centre error of the typical cluster the scalar tau is derived from [simulation: P-R61 / methods draft
# S2.3: sigma_loc 10 nm, N_loc 60, sigma_link 10 nm, rho 25 nm, three labels with geometric event counts of mean 10
# events of two localizations each -> 11.3 nm].
SIGMA_C_REFERENCE_NM = 11.3
# The pre-specified scalar tolerance: K_SCALAR * 11.3 = 31.3 -> 31 nm [simulation, as above; written in
# config/columns_params.yaml as tau0_nm].
TAU_SCALAR_DERIVED_NM = 31.0
# Localization precision when the file's NeNA is not given [lit: Huang et al. 2008, 8-20 nm in 3D-STORM; 10 nm
# the research's default]. The integration passes the file's NeNA (tools.mps_nena) through ToleranceInputs.
SIGMA_LOC_DEFAULT_NM = 10.0
# Linkage error per axis of a primary + secondary IgG stack [lit: Früh et al. 2021, offsets up to ~19 nm -> l /
# sqrt(3) per axis]; an ASSUMPTION until single-molecule clusters on glass calibrate it (Phase B).
SIGMA_LINK_DEFAULT_NM = 10.0
# Residual registration / drift error per ring [lit: Wang et al. 2014, RCC ~1.6 nm; rounded up].
SIGMA_REG_NM = 2.0
# Switching cycles (events) per Alexa Fluor 647 emitter [lit: Dempsey et al. 2011: 14 in MEA, 26 in betaME].
# PRESETS, labelled "preset (not calibrated)": the calibration of c_label on the user's own antibody stack (sparse
# primary on glass with the same secondary, P-R75) is Phase B.
C_LABEL_PRESETS: Mapping[str, float] = {"AF647 MEA": 14.0, "AF647 betaME": 26.0}
C_LABEL_DEFAULT = 14.0
# Localizations per event [ad hoc: the project simulator's default locs_per_fluor_mean]; only enters the simulated
# sum-w^2 table.
LOCS_PER_EVENT_DEFAULT = 2.0
# The footprint used when it cannot be measured: 25 nm as an UPPER bound (the median r_eff of the reference paper,
# which is inflated by the precision) [lit: Gazal et al. 2026; user's decision 3a]; 0 nm in the sensitivity row.
RHO_UPPER_NM = 25.0
RHO_SENSITIVITY_NM = 0.0
# The pool is used only when the clusters carry two or more labels on average [user's decision 1a; simulation P-R61
# S3.2: with one label the footprint is not identifiable, 29 +- 27 nm on a truth of 0].
MIN_MEAN_LABELS_FOR_POOL = 2.0
# Fewer usable clusters than this and the pool is not reported [ad hoc: the research's smallest axon had K = 30
# clusters, +-4-8 nm; below 10 the moment estimate is too noisy to call a measurement].
MIN_POOL_CLUSTERS = 10
# Bootstrap over clusters for the pool's standard error [ad hoc: 200 draws, a standard error to +-5 %].
POOL_BOOTSTRAP = 200
# Merged-cluster criterion (decision 4a): a cluster whose two event modes along the membrane are farther apart than
# MERGED_Z * sqrt(2) * sqrt(sigma_link^2 + rho_ref^2 / 4) -- what two labels of ONE cluster with the upper-bound
# footprint exceed 0.1 % of the time -- is flagged as merged [derivation: two-sided 99.9 % normal quantile 3.29;
# chosen by simulation among 2.576 / 3.29 / 3.89 (docs/eclipse_tolerance.md S6, docs/sims/eclipse_tolerance): 2.576
# over-corrects the pool by -1.2 to -7.3 nm, 3.29 leaves -3.8 to +1.6 nm, 3.89 leaves +1.0 to +6.4 nm of the merges].
MERGED_Z = 3.29
# Each mode must hold at least this many events (localizations when there are no events) [ad hoc: one event can be
# one blink of one label, not a cluster].
MERGED_MIN_UNITS_PER_SIDE = 2
# The simulated E[sum w^2 | events] table [simulation: P-R61 sims/common.py sum_w2_given_events_table, seed
# 20261015; 100 000 simulated clusters per label prior, as the research's sim2].
SUM_W2_SEED = 20261015
SUM_W2_DRAWS = 100_000
LAMBDA_GRID: Tuple[float, ...] = (0.05, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EVENT_EDGES: Tuple[int, ...] = (1, 4, 7, 10, 14, 18, 22, 27, 33, 40, 50, 60, 75, 90, 110, 140, 180, 250, 10 ** 9)

RULE_SCALAR = "scalar"
RULE_PER_PAIR = "per_pair"
RHO_POOL = "pool of every cluster (upper bound: merged clusters not excluded until the criterion is accepted)"
RHO_POOL_EXCLUDED = "pool without the merged-looking clusters (criterion accepted)"
RHO_DEFAULT_NOT_IDENTIFIABLE = "default 25 nm upper bound (not identifiable: fewer than 2 labels per cluster)"
RHO_DEFAULT_NO_EVENTS = "default 25 nm upper bound (no events: labels per cluster unknown)"
RHO_DEFAULT_FEW_CLUSTERS = "default 25 nm upper bound (too few clusters to pool)"
RHO_SOURCES: Tuple[str, ...] = (RHO_POOL, RHO_POOL_EXCLUDED, RHO_DEFAULT_NOT_IDENTIFIABLE, RHO_DEFAULT_NO_EVENTS,
                                RHO_DEFAULT_FEW_CLUSTERS)
SIGMA_C_CALIBRATED = "calibrated"
SIGMA_C_PRESET = "preset"
SIGMA_C_NO_EVENTS = "no events (one label assumed)"
SIGMA_C_SOURCES: Tuple[str, ...] = (SIGMA_C_CALIBRATED, SIGMA_C_PRESET, SIGMA_C_NO_EVENTS)


# ===================================================================================================== inputs
@dataclass(frozen=True)
class ToleranceInputs:
    """
    Everything the centre error is computed from, each with where it came from (the hooks of the Phase-B
    calibrations, decision 6): ``sigma_loc_nm`` (the file's NeNA when the caller has it), ``sigma_link_nm``,
    ``c_label`` (events per antibody stack: ``c_label_source`` "preset" until the P-R75 calibration exists, then
    "calibrated" with ``c_label_date``), ``locs_per_event``, ``sigma_reg_nm``, the footprint bounds, the k's and the
    pre-specified scalar tau. Frozen: change one with :func:`dataclasses.replace` and record why.
    """
    sigma_loc_nm: float = SIGMA_LOC_DEFAULT_NM
    sigma_loc_source: str = "default 10 nm [lit: Huang et al. 2008] (no NeNA given)"
    sigma_link_nm: float = SIGMA_LINK_DEFAULT_NM
    sigma_link_source: str = "assumed 10 nm [lit: Fruh et al. 2021], not calibrated"
    c_label: float = C_LABEL_DEFAULT
    c_label_source: str = SIGMA_C_PRESET
    c_label_date: str = ""
    locs_per_event: float = LOCS_PER_EVENT_DEFAULT
    sigma_reg_nm: float = SIGMA_REG_NM
    rho_upper_nm: float = RHO_UPPER_NM
    rho_sensitivity_nm: float = RHO_SENSITIVITY_NM
    k_arc: float = K_ARC
    k_plane: float = K_PLANE
    tau_scalar_nm: float = TAU_SCALAR_DERIVED_NM
    # Decision 4: the pool is used as an UPPER bound (every cluster, merged ones included) until the user accepts the
    # merged-cluster criterion as a measurement; True then switches sigma_c to the pool without them (recorded).
    exclude_merged_from_pool: bool = False

    def __post_init__(self) -> None:
        for name in ("sigma_loc_nm", "sigma_link_nm", "c_label", "locs_per_event", "rho_upper_nm", "k_arc", "k_plane",
                     "tau_scalar_nm"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"ToleranceInputs.{name} must be a positive finite number, got {v}")
        for name in ("sigma_reg_nm", "rho_sensitivity_nm"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"ToleranceInputs.{name} must be a non-negative finite number, got {v}")
        if self.c_label_source not in (SIGMA_C_PRESET, SIGMA_C_CALIBRATED):
            raise ValueError(f"ToleranceInputs.c_label_source must be {SIGMA_C_PRESET!r} or {SIGMA_C_CALIBRATED!r}")
        if self.c_label_source == SIGMA_C_CALIBRATED and not str(self.c_label_date).strip():
            raise ValueError("a calibrated c_label needs the date of its calibration (c_label_date)")

    @property
    def calibrated(self) -> bool:
        """True once c_label comes from the user's own calibration, with its date."""
        return self.c_label_source == SIGMA_C_CALIBRATED and bool(str(self.c_label_date).strip())

    def primary_rule(self) -> str:
        """The rule called PRIMARY (decision 2): the scalar tau until c_label is calibrated, then the per-pair rule
        (the change is recorded with the calibration's date)."""
        return RULE_PER_PAIR if self.calibrated else RULE_SCALAR

    def with_nena(self, nena_nm: Optional[float], source: str = "NeNA of the file") -> "ToleranceInputs":
        """These inputs with sigma_loc from the file's NeNA (unchanged when it is None or not a positive number)."""
        if nena_nm is None:
            return self
        try:
            v = float(nena_nm)
        except (TypeError, ValueError):
            return self
        if not (math.isfinite(v) and v > 0.0):
            return self
        return dataclasses.replace(self, sigma_loc_nm=v, sigma_loc_source=source)


# ===================================================================================================== clusters
def clusters_from_labels(labels: Any) -> Tuple[NDArray[np.int64], List[NDArray[np.intp]]]:
    """The clusters of ANY method's labels (-1 or below = noise): the labels in ascending order and the indices of
    each one's localizations."""
    lab = np.asarray(labels).astype(np.int64).reshape(-1)
    ids = np.unique(lab[lab >= 0])
    return ids, [np.flatnonzero(lab == v).astype(np.intp) for v in ids.tolist()]


def tangent_coordinates(x: NDArray[np.float64], y: NDArray[np.float64], centre_nm: Sequence[float]
                        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    One cluster's localizations along the membrane: (t, centroid). t is the signed offset of each localization from
    the cluster's centroid along the TANGENT at the centroid, taken perpendicular to the line from the axon's centre
    [ad hoc approximation of the membrane's local tangent: exact on a circle, which is where the research measured
    the estimators; a cluster's extent (~25 nm) is short against the curvature radii of these contours].
    """
    cx, cy = float(np.mean(x)), float(np.mean(y))
    rx, ry = cx - float(centre_nm[0]), cy - float(centre_nm[1])
    r = math.hypot(rx, ry)
    if r == 0.0:
        tx, ty = 1.0, 0.0
    else:
        tx, ty = -ry / r, rx / r
    t = (np.asarray(x, dtype=np.float64) - cx) * tx + (np.asarray(y, dtype=np.float64) - cy) * ty
    return np.asarray(t, dtype=np.float64), np.array([cx, cy], dtype=np.float64)


def _units(t: NDArray[np.float64], event: Optional[NDArray[np.int64]]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The units the mode split works on: the mean t of each event and its localization count (each localization
    its own unit without events). Sorted by position."""
    if event is None:
        order = np.argsort(t, kind="stable")
        return t[order], np.ones(t.size, dtype=np.float64)
    ev = np.asarray(event).astype(np.int64)
    _, inv = np.unique(ev, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    means = np.bincount(inv, weights=t) / counts
    order = np.argsort(means, kind="stable")
    return means[order], counts[order]


def two_mode_separation(t: NDArray[np.float64], event: Optional[NDArray[np.int64]] = None,
                        min_units_per_side: int = MERGED_MIN_UNITS_PER_SIDE) -> float:
    """
    The distance (nm) between the two modes of a cluster along the membrane: the units (events, or localizations
    without events) sorted by position are split in two contiguous groups so that the between-group sum of squares
    is largest (the exact 1D two-means optimum), each group holding at least ``min_units_per_side`` units, weighted by
    their localizations; the result is the difference of the two groups' weighted means. NaN when there are too few
    units.
    """
    m, w = _units(np.asarray(t, dtype=np.float64), event)
    n = m.size
    k_min = int(min_units_per_side)
    if n < 2 * k_min:
        return float("nan")
    cw = np.cumsum(w)
    cs = np.cumsum(w * m)
    tot_w, tot_s = float(cw[-1]), float(cs[-1])
    k = np.arange(k_min, n - k_min + 1)            # left group = units [0, k)
    w1 = cw[k - 1]
    s1 = cs[k - 1]
    w2 = tot_w - w1
    s2 = tot_s - s1
    mu1, mu2 = s1 / w1, s2 / w2
    between = w1 * w2 / tot_w * (mu2 - mu1) ** 2
    best = int(np.argmax(between))
    return float(abs(mu2[best] - mu1[best]))


def merged_threshold_nm(inputs: ToleranceInputs) -> float:
    """The mode separation above which a cluster is flagged as merged: MERGED_Z sqrt(2) sqrt(sigma_link^2 +
    rho_upper^2 / 4), the separation two labels of ONE cluster with the upper-bound footprint exceed 1 % of the time
    (58.3 nm with the defaults)."""
    return float(MERGED_Z * math.sqrt(2.0) * math.sqrt(inputs.sigma_link_nm ** 2 + inputs.rho_upper_nm ** 2 / 4.0))


def merged_cluster_flags(separations_nm: NDArray[np.float64], inputs: ToleranceInputs) -> NDArray[np.bool_]:
    """True where a cluster's two-mode separation exceeds :func:`merged_threshold_nm` (NaN separations: False)."""
    s = np.asarray(separations_nm, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(np.isfinite(s) & (s > merged_threshold_nm(inputs)), dtype=bool)


# ===================================================================================================== sum w^2
def _draw_labels_ztp(rng: np.random.Generator, n: int, lam: float) -> NDArray[np.int64]:
    out = np.zeros(n, dtype=np.int64)
    todo = np.ones(n, dtype=bool)
    while todo.any():
        idx = np.flatnonzero(todo)
        k = rng.poisson(lam, idx.size)
        out[idx] = k
        todo[idx[k > 0]] = False
    return out


@functools.lru_cache(maxsize=64)
def sum_w2_table(c_label: float, lam: float, locs_per_event: float = LOCS_PER_EVENT_DEFAULT,
                 draws: int = SUM_W2_DRAWS) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    """
    E[sum w^2 | events in bin] by simulation [simulation: P-R61 sims/common.py ``sum_w2_given_events_table``]: labels
    per cluster ~ zero-truncated Poisson(lam), each label a geometric number of events (mean ``c_label``), each event
    a geometric number of localizations (mean ``locs_per_event``); w_m the share of the cluster's localizations from
    label m. Returns (bin edges ``EVENT_EDGES``, mean sum w^2 per bin; NaN for an empty bin). Deterministic: seeded
    from ``SUM_W2_SEED`` and the arguments; cached.
    """
    seed = np.random.SeedSequence([SUM_W2_SEED, int(round(1000 * c_label)), int(round(1000 * lam)),
                                   int(round(1000 * locs_per_event)), int(draws)])
    rng = np.random.default_rng(seed)
    n_lab = _draw_labels_ztp(rng, int(draws), float(lam))
    tot_ev = np.zeros(int(draws))
    tot_loc = np.zeros(int(draws))
    sq = np.zeros(int(draws))
    for j in range(int(n_lab.max())):
        has = n_lab > j
        ev = rng.geometric(1.0 / float(c_label), int(has.sum()))
        locs = ev + rng.negative_binomial(ev, 1.0 / float(locs_per_event))
        tot_ev[has] += ev
        tot_loc[has] += locs
        sq[has] += locs.astype(np.float64) ** 2
    sw2 = sq / tot_loc ** 2
    edges = np.asarray(EVENT_EDGES, dtype=np.int64)
    b = np.searchsorted(edges, tot_ev, side="right") - 1
    table = tuple(float(sw2[b == i].mean()) if np.any(b == i) else float("nan") for i in range(edges.size - 1))
    return tuple(int(e) for e in EVENT_EDGES), table


def lambda_from_mean_labels(mean_labels: float) -> float:
    """The label prior's lam (zero-truncated Poisson) whose mean lam / (1 - exp(-lam)) is nearest ``mean_labels``,
    on ``LAMBDA_GRID`` [simulation: the research's grid]. The smallest lam when the mean is 1 or below."""
    m = max(1.0, float(mean_labels)) if math.isfinite(float(mean_labels)) else 1.0
    return min(LAMBDA_GRID, key=lambda lam: abs(lam / (1.0 - math.exp(-lam)) - m))


def sum_w2_given_events(n_events: Any, c_label: float, lam: float,
                        locs_per_event: float = LOCS_PER_EVENT_DEFAULT) -> NDArray[np.float64]:
    """E[sum w^2 | events] for each cluster's event count from :func:`sum_w2_table` (the nearest finite bin when a
    bin is empty); 1.0 (one label, the largest value) where the count is missing."""
    edges, table = sum_w2_table(float(c_label), float(lam), float(locs_per_event))
    tab = np.asarray(table, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(tab))
    ev = np.asarray(n_events, dtype=np.float64).reshape(-1)
    out = np.ones(ev.size, dtype=np.float64)
    for i, e in enumerate(ev.tolist()):
        if not math.isfinite(e) or e < 1:
            continue
        b = int(np.searchsorted(np.asarray(edges), int(e), side="right") - 1)
        b = min(max(b, 0), tab.size - 1)
        v = tab[b]
        if not math.isfinite(v):
            v = tab[finite[np.argmin(np.abs(finite - b))]]
        out[i] = float(v)
    return out


def sigma_c_nm(n_locs: Any, sum_w2: Any, rho_nm: float, inputs: ToleranceInputs) -> NDArray[np.float64]:
    """The per-axis centre error of each cluster (P-R61 S2.4): sqrt(sigma_loc^2 / N_loc + (sigma_link^2 + rho^2 /
    4) sum w^2 + sigma_reg^2)."""
    n = np.maximum(np.asarray(n_locs, dtype=np.float64).reshape(-1), 1.0)
    w = np.asarray(sum_w2, dtype=np.float64).reshape(-1)
    v = inputs.sigma_loc_nm ** 2 / n + (inputs.sigma_link_nm ** 2 + float(rho_nm) ** 2 / 4.0) * w + inputs.sigma_reg_nm ** 2
    return np.asarray(np.sqrt(v), dtype=np.float64)


# ===================================================================================================== footprint
@dataclass
class FootprintPool:
    """
    The footprint of the axon's clusters, pooled (method of moments, P-R61 S3.2):
    sigma_L^2 = sum_i (s_i^2 - sigma_loc^2) (N_i - 1) / N_i / sum_i (1 - sum w_i^2), rho^2 = 4 (sigma_L^2 -
    sigma_link^2), over the clusters with two or more localizations. In a population of mixed footprints it returns
    the ROOT-MEAN-SQUARE footprint, which is what sigma_c needs.

    ``rho_pooled_nm`` / ``rho_pooled_se_nm`` (bootstrap over clusters): the pool of EVERY usable cluster, an upper
    bound when the clustering merged some (decision 4b; +10 to +12 nm under DBSCAN 25/10 in the simulation);
    ``rho_pooled_excluding_merged_nm``: the same without the clusters :func:`merged_cluster_flags` flags (the
    validated candidate measurement, -3.8 to +1.6 nm in the simulation; used only with
    ``ToleranceInputs.exclude_merged_from_pool``). All None when the pool is not identifiable (``identifiable``
    False, ``reason`` says why). ``rho_used_nm`` is what sigma_c uses and ``rho_source`` says which
    (``RHO_SOURCES``); ``rho_sensitivity_nm`` the other bound (0 nm) when the pool is not identifiable, else the
    25 nm upper bound.
    """
    rho_pooled_nm: Optional[float]
    rho_pooled_se_nm: Optional[float]
    rho_pooled_excluding_merged_nm: Optional[float]
    n_used: int
    n_excluded_merged: int
    n_label_mean_hat: Optional[float]
    identifiable: bool
    reason: str
    rho_used_nm: float
    rho_source: str
    rho_sensitivity_nm: float


def _moments_rho(s2: NDArray[np.float64], n: NDArray[np.float64], sw2: NDArray[np.float64],
                 inputs: ToleranceInputs) -> Optional[float]:
    den = float(np.sum(1.0 - sw2))
    if not den > 0.0:
        return None
    sl2 = float(np.sum((s2 - inputs.sigma_loc_nm ** 2) * (n - 1.0) / n)) / den
    r2 = 4.0 * (sl2 - inputs.sigma_link_nm ** 2)
    return float(math.sqrt(r2)) if r2 > 0.0 else 0.0


def pool_footprint(s_t_nm: Any, n_locs: Any, sum_w2: Any, merged: Any, n_label_mean_hat: Optional[float],
                   inputs: ToleranceInputs, *, have_events: bool = True,
                   rng: Optional[np.random.Generator] = None, n_boot: int = POOL_BOOTSTRAP) -> FootprintPool:
    """The axon's pooled footprint (see :class:`FootprintPool`). ``rng`` drives the bootstrap (default seeded 0)."""
    s = np.asarray(s_t_nm, dtype=np.float64).reshape(-1)
    n = np.asarray(n_locs, dtype=np.float64).reshape(-1)
    w = np.asarray(sum_w2, dtype=np.float64).reshape(-1)
    mg = np.asarray(merged, dtype=bool).reshape(-1)
    ok = np.isfinite(s) & (n > 1)
    ok_excl = ok & ~mg
    n_excl = int(np.count_nonzero(ok & mg))

    def not_identifiable(reason: str, source: str) -> FootprintPool:
        return FootprintPool(rho_pooled_nm=None, rho_pooled_se_nm=None, rho_pooled_excluding_merged_nm=None,
                             n_used=int(ok.sum()),
                             n_excluded_merged=n_excl, n_label_mean_hat=n_label_mean_hat, identifiable=False,
                             reason=reason, rho_used_nm=inputs.rho_upper_nm, rho_source=source,
                             rho_sensitivity_nm=inputs.rho_sensitivity_nm)

    if not have_events or n_label_mean_hat is None:
        return not_identifiable("no emission events: the labels per cluster cannot be estimated", RHO_DEFAULT_NO_EVENTS)
    if n_label_mean_hat < MIN_MEAN_LABELS_FOR_POOL:
        return not_identifiable(f"{n_label_mean_hat:.2f} labels per cluster on average (< {MIN_MEAN_LABELS_FOR_POOL:g}): "
                                "the footprint is not identifiable", RHO_DEFAULT_NOT_IDENTIFIABLE)
    if int(ok.sum()) < MIN_POOL_CLUSTERS:
        return not_identifiable(f"{int(ok.sum())} usable clusters (< {MIN_POOL_CLUSTERS})", RHO_DEFAULT_FEW_CLUSTERS)
    s2 = s ** 2
    rho = _moments_rho(s2[ok], n[ok], w[ok], inputs)
    if rho is None:
        return not_identifiable("no label variance to pool (every cluster looks like one label)",
                                RHO_DEFAULT_NOT_IDENTIFIABLE)
    rho_excl = (_moments_rho(s2[ok_excl], n[ok_excl], w[ok_excl], inputs)
                if int(ok_excl.sum()) >= MIN_POOL_CLUSTERS else None)
    g = rng if rng is not None else np.random.default_rng(0)
    use_excl = bool(inputs.exclude_merged_from_pool) and rho_excl is not None
    idx = np.flatnonzero(ok_excl if use_excl else ok)
    boots: List[float] = []
    for _ in range(int(n_boot)):
        pick = idx[g.integers(0, idx.size, idx.size)]
        r = _moments_rho(s2[pick], n[pick], w[pick], inputs)
        if r is not None:
            boots.append(r)
    se = float(np.std(boots, ddof=1)) if len(boots) > 1 else None
    used = float(rho_excl) if use_excl and rho_excl is not None else rho
    return FootprintPool(rho_pooled_nm=rho, rho_pooled_se_nm=se, rho_pooled_excluding_merged_nm=rho_excl,
                         n_used=int((ok_excl if use_excl else ok).sum()), n_excluded_merged=n_excl,
                         n_label_mean_hat=n_label_mean_hat, identifiable=True, reason="", rho_used_nm=used,
                         rho_source=RHO_POOL_EXCLUDED if use_excl else RHO_POOL,
                         rho_sensitivity_nm=inputs.rho_upper_nm)


# ===================================================================================================== per axon
@dataclass
class ClusterTolerances:
    """
    The clusters of one group (a ring, or one method's slab), in the order the caller gave them. Per cluster:
    ``label`` (the method's label, or the cluster's index in ``Ring.clusters``), ``n_locs``, ``n_events`` (NaN
    without events), ``centroid_nm`` (K, 2), ``s_t_nm`` (sd of its localizations along the membrane; NaN below two),
    ``mode_separation_nm`` and ``merged_suspect`` (the merged-cluster criterion), ``sum_w2_hat``, ``n_label_hat``
    (events / c_label, at least 1; NaN without events), ``sigma_c_nm`` (with the axon's ``rho_used_nm``),
    ``sigma_c_sensitivity_nm`` (with the other footprint bound), ``centroid_r95_nm`` = K_PLANE sigma_c and
    ``r_eff_nm`` (sqrt(hull area / pi) when the caller has it: it GROWS WITH N_loc, is not the footprint and enters
    no tolerance; exported for the record only).
    """
    label: NDArray[np.int64]
    n_locs: NDArray[np.int64]
    n_events: NDArray[np.float64]
    centroid_nm: NDArray[np.float64]
    s_t_nm: NDArray[np.float64]
    mode_separation_nm: NDArray[np.float64]
    merged_suspect: NDArray[np.bool_]
    sum_w2_hat: NDArray[np.float64]
    n_label_hat: NDArray[np.float64]
    sigma_c_nm: NDArray[np.float64]
    sigma_c_sensitivity_nm: NDArray[np.float64]
    centroid_r95_nm: NDArray[np.float64]
    r_eff_nm: NDArray[np.float64]
    sigma_c_source: str

    @property
    def n_clusters(self) -> int:
        return int(self.label.size)


@dataclass
class AxonTolerances:
    """The centre errors of every cluster of one axon (``groups``: key -> :class:`ClusterTolerances`), the pooled
    footprint, the label prior used for sum w^2 (``lambda_hat``), the inputs, the scalar tau derived from THIS axon's
    centre errors (``tau_derived_nm`` = K_SCALAR x median sigma_c, the P-R4 comparator) and notes."""
    inputs: ToleranceInputs
    groups: Dict[Hashable, ClusterTolerances]
    pool: FootprintPool
    lambda_hat: Optional[float]
    tau_derived_nm: Optional[float]
    warnings: List[str] = field(default_factory=list)

    def sigma_of(self, key: Hashable) -> NDArray[np.float64]:
        return self.groups[key].sigma_c_nm

    def all_sigma_c(self) -> NDArray[np.float64]:
        parts = [g.sigma_c_nm for g in self.groups.values()]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


@dataclass
class _Raw:
    label: NDArray[np.int64]
    n_locs: NDArray[np.int64]
    n_events: NDArray[np.float64]
    centroid: NDArray[np.float64]
    s_t: NDArray[np.float64]
    sep: NDArray[np.float64]
    r_eff: NDArray[np.float64]


def _raw_group(x: NDArray[np.float64], y: NDArray[np.float64], event: Optional[NDArray[np.int64]],
               members: Sequence[NDArray[np.intp]], labels: Sequence[int], centre: Optional[Sequence[float]],
               r_eff: Optional[Sequence[float]]) -> _Raw:
    k = len(members)
    cents = np.array([[float(np.mean(x[m])), float(np.mean(y[m]))] if m.size else [np.nan, np.nan] for m in members],
                     dtype=np.float64).reshape(-1, 2)
    if centre is None:
        good = np.isfinite(cents).all(axis=1)
        c = cents[good].mean(axis=0) if good.any() else np.zeros(2)
    else:
        c = np.asarray(centre, dtype=np.float64)
    n_locs = np.array([m.size for m in members], dtype=np.int64)
    n_ev = np.full(k, np.nan)
    s_t = np.full(k, np.nan)
    sep = np.full(k, np.nan)
    for i, m in enumerate(members):
        if m.size == 0:
            continue
        t, _ = tangent_coordinates(x[m], y[m], [float(v) for v in np.asarray(c, dtype=np.float64).ravel()])
        ev = None
        if event is not None:
            ev_m = np.asarray(event[m]).astype(np.int64)
            linked = ev_m >= 0
            # an unlinked localization (-1) is its own event
            ev = np.where(linked, ev_m, -1 - np.arange(ev_m.size))
            n_ev[i] = float(np.unique(ev).size)
        if m.size > 1:
            s_t[i] = float(np.std(t, ddof=1))
        sep[i] = two_mode_separation(t, ev, MERGED_MIN_UNITS_PER_SIDE if ev is not None else 2 * MERGED_MIN_UNITS_PER_SIDE)
    reff = (np.full(k, np.nan) if r_eff is None else np.asarray(r_eff, dtype=np.float64).reshape(-1))
    return _Raw(label=np.asarray(labels, dtype=np.int64), n_locs=n_locs, n_events=n_ev, centroid=cents, s_t=s_t,
                sep=sep, r_eff=reff)


def axon_tolerances(
    x_nm: Any,
    y_nm: Any,
    groups: Mapping[Any, Sequence[NDArray[np.intp]]],
    *,
    event_id: Optional[Any] = None,
    labels: Optional[Mapping[Any, Sequence[int]]] = None,
    centres_nm: Optional[Mapping[Any, Optional[Sequence[float]]]] = None,
    r_eff_nm: Optional[Mapping[Any, Sequence[float]]] = None,
    inputs: ToleranceInputs = ToleranceInputs(),
    seed: int = 0,
) -> AxonTolerances:
    """
    The centre error of every cluster of one axon, from ANY clustering. ``groups``: key (a ring index, or anything)
    -> the clusters of that group, each the indices of its localizations into ``x_nm`` / ``y_nm`` / ``event_id``.
    ``event_id`` (per localization, -1 = not linked; None without frames) gives the events. ``centres_nm`` per group
    sets the axon centre the tangent is taken from (default: the mean of the group's cluster centroids). The
    footprint is pooled over EVERY group (it is a property of the axon), with the merged clusters excluded.
    Deterministic given ``seed`` (the pool's bootstrap).
    """
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("axon_tolerances: x and y differ in length")
    ev = None if event_id is None else np.asarray(event_id).astype(np.int64).reshape(-1)
    if ev is not None and ev.size != x.size:
        raise ValueError("axon_tolerances: event_id and x differ in length")
    warnings_: List[str] = []
    raws: Dict[Hashable, _Raw] = {}
    for key, members in groups.items():
        mem = [np.asarray(m, dtype=np.intp).reshape(-1) for m in members]
        labs = list(labels[key]) if labels is not None and key in labels else list(range(len(mem)))
        cen = None if centres_nm is None else centres_nm.get(key)
        reff = None if r_eff_nm is None else r_eff_nm.get(key)
        raws[key] = _raw_group(x, y, ev, mem, labs, cen, reff)
    all_ev = np.concatenate([r.n_events for r in raws.values()]) if raws else np.zeros(0)
    have_events = ev is not None and bool(np.isfinite(all_ev).any())
    inp = inputs
    if have_events:
        mean_ev = float(np.nanmean(all_ev))
        n_label_mean = mean_ev / inp.c_label
        lam: Optional[float] = lambda_from_mean_labels(n_label_mean)
    else:
        n_label_mean, lam = float("nan"), None
        warnings_.append("no emission events: sum w^2 = 1 for every cluster (one label assumed, the largest centre "
                         "error) and the footprint cannot be pooled")
    sw2 = {k: (sum_w2_given_events(r.n_events, inp.c_label, float(lam), inp.locs_per_event) if lam is not None
               else np.ones(r.n_locs.size)) for k, r in raws.items()}
    seps = np.concatenate([r.sep for r in raws.values()]) if raws else np.zeros(0)
    merged_all = merged_cluster_flags(seps, inp)
    pool = pool_footprint(
        np.concatenate([r.s_t for r in raws.values()]) if raws else np.zeros(0),
        np.concatenate([r.n_locs for r in raws.values()]) if raws else np.zeros(0),
        np.concatenate([sw2[k] for k in raws]) if raws else np.zeros(0),
        merged_all, None if not have_events else n_label_mean, inp, have_events=have_events,
        rng=np.random.default_rng(np.random.SeedSequence([int(seed), 61])))
    if pool.identifiable and pool.rho_pooled_excluding_merged_nm is not None and pool.n_excluded_merged:
        warnings_.append(f"footprint pool: {pool.n_excluded_merged} merged-looking cluster(s); "
                         f"{pool.rho_pooled_nm:.1f} nm with them (upper bound), "
                         f"{pool.rho_pooled_excluding_merged_nm:.1f} nm without")
    if not pool.identifiable:
        warnings_.append(f"footprint not measured ({pool.reason}): {pool.rho_used_nm:g} nm used as an upper bound, "
                         f"{pool.rho_sensitivity_nm:g} nm in the sensitivity row (a decision, not a measurement)")
    source = (SIGMA_C_NO_EVENTS if not have_events else
              (SIGMA_C_CALIBRATED if inp.calibrated else SIGMA_C_PRESET))
    out: Dict[Hashable, ClusterTolerances] = {}
    offset = 0
    for k, r in raws.items():
        m = r.n_locs.size
        sig = sigma_c_nm(r.n_locs, sw2[k], pool.rho_used_nm, inp)
        sig_s = sigma_c_nm(r.n_locs, sw2[k], pool.rho_sensitivity_nm, inp)
        nl = (np.maximum(1.0, r.n_events / inp.c_label) if have_events else np.full(m, np.nan))
        out[k] = ClusterTolerances(
            label=r.label, n_locs=r.n_locs, n_events=r.n_events, centroid_nm=r.centroid, s_t_nm=r.s_t,
            mode_separation_nm=r.sep, merged_suspect=merged_all[offset:offset + m], sum_w2_hat=sw2[k],
            n_label_hat=nl, sigma_c_nm=sig, sigma_c_sensitivity_nm=sig_s, centroid_r95_nm=inp.k_plane * sig,
            r_eff_nm=r.r_eff, sigma_c_source=source)
        offset += m
    allsig = np.concatenate([g.sigma_c_nm for g in out.values()]) if out else np.zeros(0)
    tau_d = scalar_tau_from_sigma(allsig) if allsig.size else None
    if not inp.calibrated:
        warnings_.append(f"c_label not calibrated: sigma_c from the preset c_label = {inp.c_label:g} events per label")
    return AxonTolerances(inputs=inp, groups=out, pool=pool, lambda_hat=lam, tau_derived_nm=tau_d, warnings=warnings_)


def tolerances_from_rings(res: Any, inputs: ToleranceInputs = ToleranceInputs(), *, seed: int = 0) -> AxonTolerances:
    """:func:`axon_tolerances` on the rings of ``tools.mps_columns.build_rings`` (or ``clean_rings``): groups keyed by
    ring index, clusters in ``Ring.clusters`` order (the order the arc test indexes), localizations ``res.x_p`` /
    ``res.y_p`` (axon frame), events ``res.event_id``, each ring's contour centroid as the centre, the cluster's
    r_eff (``Cluster.rho_nm``) recorded."""
    groups: Dict[Hashable, List[NDArray[np.intp]]] = {}
    labels: Dict[Hashable, List[int]] = {}
    centres: Dict[Hashable, Optional[Sequence[float]]] = {}
    reff: Dict[Hashable, List[float]] = {}
    for ring in res.rings:
        k = int(ring.index)
        groups[k] = [np.asarray(cl.loc_index, dtype=np.intp) for cl in ring.clusters]
        labels[k] = [int(cl.label) for cl in ring.clusters]
        c = getattr(ring, "centroid_nm", None)
        centres[k] = None if c is None else [float(c[0]), float(c[1])]
        reff[k] = [float(cl.rho_nm) for cl in ring.clusters]
    return axon_tolerances(res.x_p, res.y_p, groups, event_id=res.event_id, labels=labels, centres_nm=centres,
                           r_eff_nm=reff, inputs=inputs, seed=seed)


def pair_tolerance_matrix(sigma_a_nm: Any, sigma_b_nm: Any, k: float = K_ARC) -> NDArray[np.float64]:
    """t_ij = k sqrt(sigma_a,i^2 + sigma_b,j^2), (K_a, K_b). k = K_ARC on the arc, K_PLANE in the plane."""
    a = np.asarray(sigma_a_nm, dtype=np.float64).reshape(-1)
    b = np.asarray(sigma_b_nm, dtype=np.float64).reshape(-1)
    return np.asarray(float(k) * np.sqrt(a[:, None] ** 2 + b[None, :] ** 2), dtype=np.float64)


def scalar_tau_from_sigma(sigma_c: Any) -> Optional[float]:
    """The scalar tolerance derived from measured centre errors: K_SCALAR x median sigma_c (P-R4, the comparator)."""
    s = np.asarray(sigma_c, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    return float(K_SCALAR * np.median(s)) if s.size else None


def per_pair_arc_columns(res: Any, cols: Any, tolerances: AxonTolerances, **analyze_kwargs: Any) -> Any:
    """The arc test of ``tools.mps_unroll.analyze_arc_columns`` on the same rings, curve, seed and null as the scalar
    run, with the per-pair tolerance (``pair_sigma_nm`` = each ring's sigma_c, ``pair_k`` = inputs.k_arc). A ring
    the tolerances do not cover is an error (the two must come from the same rings)."""
    from tools.mps_unroll import analyze_arc_columns
    sig: Dict[int, NDArray[np.float64]] = {int(str(k)): np.asarray(g.sigma_c_nm, dtype=np.float64)
                                           for k, g in tolerances.groups.items()}
    return analyze_arc_columns(res, cols, pair_sigma_nm=sig, pair_k=tolerances.inputs.k_arc, **analyze_kwargs)


# ===================================================================================================== export
CLUSTER_COLUMNS: Tuple[str, ...] = (
    "ring", "cluster", "label", "n_locs", "n_events", "sum_w2_hat", "n_label_hat", "sigma_c_nm",
    "sigma_c_sensitivity_nm", "centroid_r95_nm", "r_eff_nm", "sigma_c_source", "s_t_nm", "mode_separation_nm",
    "merged_suspect")
PAIR_COLUMNS: Tuple[str, ...] = ("ring_a", "ring_b", "rule", "role", "cluster_a", "cluster_b", "d_nm", "t_ij_nm")
AXON_COLUMNS: Tuple[str, ...] = (
    "primary_rule", "rho_pooled_nm", "rho_pooled_se_nm", "rho_pooled_excluding_merged_nm", "rho_source",
    "rho_used_nm",
    "rho_sensitivity_nm", "n_pool_clusters", "n_merged_excluded", "n_label_mean_hat", "lambda_hat", "c_label_used",
    "c_label_source", "c_label_date", "sigma_loc_nena_nm", "sigma_loc_source", "sigma_link_nm", "sigma_link_source",
    "sigma_reg_nm", "tau_scalar_nm", "tau_derived_axon_nm", "sigma_c_median_nm",
    "scalar_pairs", "scalar_T_obs", "scalar_E_dir", "scalar_E_null", "scalar_excess", "scalar_z_A",
    "scalar_p_excess_uncalibrated",
    "per_pair_pairs", "per_pair_T_obs", "per_pair_E_dir", "per_pair_E_null", "per_pair_excess", "per_pair_z_A",
    "per_pair_p_excess_uncalibrated", "per_pair_mean_t_nm")

COLUMN_DESCRIPTIONS: Mapping[str, str] = {
    "ring": "Ring index (build_rings).",
    "cluster": "Index of the cluster in its ring (Ring.clusters order; the arc test's index).",
    "label": "The clustering method's label of the cluster.",
    "n_locs": "Localizations of the cluster.",
    "n_events": "Emission events of the cluster (linked localizations); empty without frames.",
    "sum_w2_hat": "E[sum of squared label shares | events], from the simulated table (1 = one label); 1 without events.",
    "n_label_hat": "Labels (antibody stacks) estimated as events / c_label, at least 1; empty without events.",
    "sigma_c_nm": "Per-axis centre error of the cluster: sqrt(sigma_loc^2/N_loc + (sigma_link^2 + rho^2/4) sum_w2 + "
                  "sigma_reg^2), rho = the axon's rho_used_nm.",
    "sigma_c_sensitivity_nm": "The same with the other footprint bound (rho_sensitivity_nm).",
    "centroid_r95_nm": "Radius of the 95 % uncertainty circle of the centre: 2.448 x sigma_c (Rayleigh quantile).",
    "r_eff_nm": "sqrt(convex hull area / pi). Grows with N_loc; NOT the footprint and NOT used by any tolerance.",
    "sigma_c_source": "Where c_label came from: 'calibrated', 'preset' or 'no events (one label assumed)'.",
    "s_t_nm": "Standard deviation of the cluster's localizations along the membrane (tangent), nm.",
    "mode_separation_nm": "Distance between the two event modes of the cluster along the membrane (merged criterion).",
    "merged_suspect": "True when the mode separation exceeds the merged threshold: excluded from the footprint pool.",
    "ring_a": "First ring of the pair.", "ring_b": "Second ring of the pair.",
    "rule": "'scalar' (|ds| <= tau) or 'per_pair' (|ds| <= t_ij = 1.96 sqrt(sigma_i^2 + sigma_j^2)).",
    "role": "'primary' or 'exploratory' (the per-pair rule is exploratory until c_label is calibrated).",
    "cluster_a": "Cluster index in ring_a.", "cluster_b": "Cluster index in ring_b.",
    "d_nm": "Arc distance of the matched pair on the reference curve, nm.",
    "t_ij_nm": "The tolerance the pair was admitted with (tau under the scalar rule), nm.",
    "primary_rule": "The rule called primary: 'scalar' until c_label is calibrated (dated), then 'per_pair'.",
    "rho_pooled_nm": "Footprint of the axon's clusters, pooled over every cluster (root-mean-square): an UPPER bound "
                     "when the clustering merged clusters; empty when not identifiable.",
    "rho_pooled_se_nm": "Bootstrap standard error of rho_pooled_nm (over clusters).",
    "rho_pooled_excluding_merged_nm": "The pooled footprint without the merged-looking clusters (validated "
                                      "candidate; used only once the criterion is accepted).",
    "rho_source": "Which footprint sigma_c used and why (pool as an upper bound, or the 25 nm default).",
    "rho_used_nm": "The footprint sigma_c used, nm.",
    "rho_sensitivity_nm": "The other footprint bound (0 nm when the pool is not identifiable; else 25 nm).",
    "n_pool_clusters": "Clusters that entered the pool used.",
    "n_merged_excluded": "Clusters flagged as merged-looking (left out of rho_pooled_excluding_merged_nm).",
    "n_label_mean_hat": "Mean events per cluster / c_label: labels per cluster on average.",
    "lambda_hat": "Label prior (zero-truncated Poisson lambda) of the sum_w2 table.",
    "c_label_used": "Events per label used.", "c_label_source": "'preset' or 'calibrated'.",
    "c_label_date": "Date of the c_label calibration (empty for a preset).",
    "sigma_loc_nena_nm": "Localization precision used (the file's NeNA when given).",
    "sigma_loc_source": "Where sigma_loc came from.",
    "sigma_link_nm": "Linkage error per axis used.", "sigma_link_source": "Where sigma_link came from.",
    "sigma_reg_nm": "Registration error per ring used.",
    "tau_scalar_nm": "The pre-specified scalar tolerance of the run (31 nm by default).",
    "tau_derived_axon_nm": "2.77 x median sigma_c of this axon: the scalar tolerance its own centre errors give.",
    "sigma_c_median_nm": "Median centre error of the axon's clusters.",
}
for _rule in ("scalar", "per_pair"):
    _r = "scalar rule" if _rule == "scalar" else "per-pair rule (EXPLORATORY until c_label is calibrated)"
    COLUMN_DESCRIPTIONS = {**COLUMN_DESCRIPTIONS,
                           f"{_rule}_pairs": f"Ring pairs of the joint arc test, {_r}.",
                           f"{_rule}_T_obs": f"Matched pairs summed over the ring pairs, {_r}.",
                           f"{_rule}_E_dir": f"Mean over ring pairs of matched / min(K_a, K_b), {_r}.",
                           f"{_rule}_E_null": f"The same under the rotation null (the chance rate), {_r}.",
                           f"{_rule}_excess": f"E_dir - E_null, {_r}: the evidence (never E alone).",
                           f"{_rule}_z_A": f"Standardised joint count, {_r}.",
                           f"{_rule}_p_excess_uncalibrated": f"Monte Carlo p of the excess, {_r} (not calibrated)."}
COLUMN_DESCRIPTIONS = {**COLUMN_DESCRIPTIONS,
                       "per_pair_mean_t_nm": "Mean tolerance of the matched pairs under the per-pair rule."}


def describe(column: str) -> Optional[str]:
    """The description of an export column of this module (None for an unknown name)."""
    return COLUMN_DESCRIPTIONS.get(column)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def cluster_rows(tol: AxonTolerances) -> List[Dict[str, Any]]:
    """One row per cluster (``CLUSTER_COLUMNS``), groups in key order."""
    rows: List[Dict[str, Any]] = []
    for key in sorted(tol.groups, key=lambda k: (0, k, "") if isinstance(k, int) else (1, 0, str(k))):
        g = tol.groups[key]
        for i in range(g.n_clusters):
            rows.append({
                "ring": key, "cluster": i, "label": int(g.label[i]), "n_locs": int(g.n_locs[i]),
                "n_events": None if not math.isfinite(g.n_events[i]) else int(g.n_events[i]),
                "sum_w2_hat": float(g.sum_w2_hat[i]), "n_label_hat": _num(g.n_label_hat[i]),
                "sigma_c_nm": float(g.sigma_c_nm[i]), "sigma_c_sensitivity_nm": float(g.sigma_c_sensitivity_nm[i]),
                "centroid_r95_nm": float(g.centroid_r95_nm[i]), "r_eff_nm": _num(g.r_eff_nm[i]),
                "sigma_c_source": g.sigma_c_source, "s_t_nm": _num(g.s_t_nm[i]),
                "mode_separation_nm": _num(g.mode_separation_nm[i]), "merged_suspect": bool(g.merged_suspect[i])})
    return rows


def _role(rule: str, inputs: ToleranceInputs) -> str:
    return "primary" if inputs.primary_rule() == rule else "exploratory"


def pair_rows(scalar_arc: Any, per_pair_arc: Any, inputs: ToleranceInputs) -> List[Dict[str, Any]]:
    """One row per MATCHED pair of clusters of every adjacent ring pair, under each rule (``PAIR_COLUMNS``):
    the arc distance and the tolerance it was admitted with (t_ij; tau under the scalar rule)."""
    rows: List[Dict[str, Any]] = []
    for rule, arc in ((RULE_SCALAR, scalar_arc), (RULE_PER_PAIR, per_pair_arc)):
        if arc is None:
            continue
        for m in arc.adjacent:
            t = np.asarray(getattr(m, "t_nm", np.zeros(0)), dtype=np.float64)
            for q in range(int(np.asarray(m.i_a).size)):
                rows.append({"ring_a": int(m.ring_a), "ring_b": int(m.ring_b), "rule": rule,
                             "role": _role(rule, inputs), "cluster_a": int(m.i_a[q]), "cluster_b": int(m.j_b[q]),
                             "d_nm": float(m.d_nm[q]),
                             "t_ij_nm": float(t[q]) if q < t.size else float(m.tau_nm)})
    return rows


def _joint_summary(arc: Any, pairs: Optional[Sequence[Tuple[int, int]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"pairs": "", "T_obs": None, "E_dir": None, "E_null": None, "excess": None, "z_A": None,
                           "p_excess_uncalibrated": None}
    joint = None if arc is None else getattr(arc, "joint", None)
    if joint is None or not len(joint.pairs):
        return out
    have = [(int(a), int(b)) for a, b in joint.pairs]
    sel = list(range(len(have))) if pairs is None else [q for q, p in enumerate(have)
                                                       if p in {(int(a), int(b)) for a, b in pairs}]
    if not sel:
        return out
    from tools.mps_matching import _phipson_smyth
    obs = np.asarray(joint.n_matched_obs)[sel]
    null = np.asarray(joint.null_n_matched)[:, sel]
    mk = np.asarray(joint.min_K, dtype=np.float64)[sel]
    t_obs, t_null = int(obs.sum()), null.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_dir = float(np.nanmean(np.where(mk > 0, obs / np.maximum(mk, 1), np.nan)))
        e_null = float(np.nanmean(np.where(mk > 0, null.mean(axis=0) / np.maximum(mk, 1), np.nan)))
    sd = float(np.std(t_null, ddof=1)) if t_null.size > 1 else float("nan")
    p_ex, _, _ = _phipson_smyth(t_null, t_obs)
    out.update(pairs=" ".join(f"{have[q][0]}-{have[q][1]}" for q in sel), T_obs=t_obs, E_dir=_num(e_dir),
               E_null=_num(e_null), excess=_num(e_dir - e_null),
               z_A=_num((t_obs - float(t_null.mean())) / sd) if sd > 0 else None, p_excess_uncalibrated=float(p_ex))
    return out


def axon_row(tol: AxonTolerances, scalar_arc: Any = None, per_pair_arc: Any = None,
             pairs: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """The axon's row (``AXON_COLUMNS``): footprint, inputs with their sources, both taus, and the joint arc test
    under each rule restricted to ``pairs`` (every adjacent pair when None; the review passes the VIABLE ones)."""
    inp, pool = tol.inputs, tol.pool
    allsig = tol.all_sigma_c()
    row: Dict[str, Any] = {
        "primary_rule": inp.primary_rule(), "rho_pooled_nm": pool.rho_pooled_nm,
        "rho_pooled_se_nm": pool.rho_pooled_se_nm,
        "rho_pooled_excluding_merged_nm": pool.rho_pooled_excluding_merged_nm,
        "rho_source": pool.rho_source, "rho_used_nm": pool.rho_used_nm, "rho_sensitivity_nm": pool.rho_sensitivity_nm,
        "n_pool_clusters": pool.n_used, "n_merged_excluded": pool.n_excluded_merged,
        "n_label_mean_hat": _num(pool.n_label_mean_hat), "lambda_hat": tol.lambda_hat, "c_label_used": inp.c_label,
        "c_label_source": inp.c_label_source, "c_label_date": inp.c_label_date, "sigma_loc_nena_nm": inp.sigma_loc_nm,
        "sigma_loc_source": inp.sigma_loc_source, "sigma_link_nm": inp.sigma_link_nm,
        "sigma_link_source": inp.sigma_link_source, "sigma_reg_nm": inp.sigma_reg_nm,
        "tau_scalar_nm": float(scalar_arc.tau_nm) if scalar_arc is not None else inp.tau_scalar_nm,
        "tau_derived_axon_nm": tol.tau_derived_nm,
        "sigma_c_median_nm": float(np.median(allsig)) if allsig.size else None,
    }
    for rule, arc in (("scalar", scalar_arc), ("per_pair", per_pair_arc)):
        for k, v in _joint_summary(arc, pairs).items():
            row[f"{rule}_{k}"] = v
    ts = [float(t) for m in (per_pair_arc.adjacent if per_pair_arc is not None else [])
          for t in np.asarray(getattr(m, "t_nm", np.zeros(0))).tolist()
          if pairs is None or (int(m.ring_a), int(m.ring_b)) in {(int(a), int(b)) for a, b in pairs}]
    row["per_pair_mean_t_nm"] = float(np.mean(ts)) if ts else None
    return {c: row.get(c) for c in AXON_COLUMNS}


def summary_lines(tol: AxonTolerances, scalar_arc: Any = None, per_pair_arc: Any = None,
                  pairs: Optional[Sequence[Tuple[int, int]]] = None) -> List[str]:
    """The lines the column review can show under its arc test (the integration places them): the footprint line,
    the c_label warning and the two rules side by side, each with its role."""
    inp, pool = tol.inputs, tol.pool
    out: List[str] = []
    if pool.identifiable and pool.rho_pooled_nm is not None:
        se = "" if pool.rho_pooled_se_nm is None else f" +- {pool.rho_pooled_se_nm:.0f}"
        ex = ("" if pool.rho_pooled_excluding_merged_nm is None
              else f"; {pool.rho_pooled_excluding_merged_nm:.0f} nm without the {pool.n_excluded_merged} merged-looking")
        role = ("used" if pool.rho_source == RHO_POOL_EXCLUDED else
                "an UPPER BOUND until the merged-cluster criterion is accepted")
        out.append(f"   footprint of the axon: rho = {pool.rho_used_nm:.0f}{se} nm (pool of {pool.n_used} clusters{ex}) "
                   f"-- {role}")
    else:
        out.append(f"   footprint not identifiable ({pool.reason}): {pool.rho_used_nm:g} nm used (upper bound), "
                   f"{pool.rho_sensitivity_nm:g} nm in the sensitivity -- a decision, not a measurement")
    if not inp.calibrated:
        out.append(f"   c_label not calibrated: sigma_c from the preset {inp.c_label:g} events per label")
    row = axon_row(tol, scalar_arc, per_pair_arc, pairs)
    for rule, name in (("scalar", f"tau = {row['tau_scalar_nm']:.1f} nm"),
                       ("per_pair", "t_ij = 1.96 sqrt(sigma_i^2 + sigma_j^2)")):
        if row[f"{rule}_T_obs"] is None:
            continue
        role = "PRIMARY" if inp.primary_rule() == (RULE_SCALAR if rule == "scalar" else RULE_PER_PAIR) else "EXPLORATORY"
        z = row[f"{rule}_z_A"]
        out.append(f"   {role} {name}: T = {row[f'{rule}_T_obs']} matched, E = {row[f'{rule}_E_dir']:.3f} vs "
                   f"chance {row[f'{rule}_E_null']:.3f} (excess {row[f'{rule}_excess']:+.3f}), "
                   f"z_A = {'nan' if z is None else f'{z:+.3f}'}, p_excess = {row[f'{rule}_p_excess_uncalibrated']:.4f} "
                   "(not calibrated)")
    if tol.tau_derived_nm is not None:
        out.append(f"   scalar tau from this axon's own centre errors (2.77 x median sigma_c): {tol.tau_derived_nm:.1f} nm")
    return out


# ===================================================================================================== plug-in
ECLIPSE_ANALYSIS = "eclipse"


def _readouts() -> Tuple[Any, ...]:
    from tools.multimethod import ReadoutSpec
    rule_levels = (RULE_SCALAR, RULE_PER_PAIR)
    return (
        ReadoutSpec("eclipse_n_clusters", kind="count", description="Clusters whose centre error was computed."),
        ReadoutSpec("eclipse_sigma_c_median_nm", description="Median per-axis centre error sigma_c of the clusters.",
                    origin="sigma_link 10 nm [lit: Fruh 2021, assumed]; c_label 14 [lit: Dempsey 2011, preset]; "
                           "sigma_reg 2 nm [lit: Wang 2014]; sigma_loc = NeNA or 10 nm [lit: Huang 2008]"),
        ReadoutSpec("eclipse_centroid_r95_median_nm", description="Median radius of the 95 % centre circle "
                                                                  "(2.448 sigma_c)."),
        ReadoutSpec("eclipse_tau_derived_nm", description="2.77 x median sigma_c: the scalar tolerance this method's "
                                                          "clusters give (P-R4 comparator).",
                    origin="2.77 = 1.96 sqrt(2) [derivation]"),
        ReadoutSpec("eclipse_rho_pooled_nm", description="Pooled footprint of the axon (upper bound); None when not "
                                                         "identifiable.", origin="method of moments [simulation: P-R61]"),
        ReadoutSpec("eclipse_rho_used_nm", description="Footprint sigma_c used (the pool, or the 25 nm upper bound).",
                    origin="25 nm [lit: Gazal 2026; user's decision 3a]"),
        ReadoutSpec("eclipse_rho_source", kind="verdict", levels=RHO_SOURCES,
                    description="Where the footprint came from."),
        ReadoutSpec("eclipse_n_label_mean_hat", description="Mean events per cluster / c_label."),
        ReadoutSpec("eclipse_n_merged_suspect", kind="count",
                    description="Clusters flagged as merged (two event modes too far apart for one cluster)."),
        ReadoutSpec("eclipse_sigma_c_source", kind="verdict", levels=SIGMA_C_SOURCES,
                    description="Whether c_label was calibrated, a preset, or no events existed."),
        ReadoutSpec("eclipse_n_ring_pairs", kind="count",
                    description="Adjacent ring pairs of the arc test (only with the rings input; else None)."),
        ReadoutSpec("eclipse_scalar_excess", description="E - chance under the scalar tau (all adjacent pairs, "
                                                         "uncleaned rings)."),
        ReadoutSpec("eclipse_scalar_z_a", description="Joint z_A under the scalar tau."),
        ReadoutSpec("eclipse_scalar_p", kind="p_value", tests="eclipse_scalar_excess",
                    description="Monte Carlo p of the excess under the scalar tau (not calibrated).",
                    origin="alpha 0.05 [ad hoc: conventional]"),
        ReadoutSpec("eclipse_per_pair_excess", description="E - chance under the per-pair tolerance (EXPLORATORY)."),
        ReadoutSpec("eclipse_per_pair_z_a", description="Joint z_A under the per-pair tolerance (EXPLORATORY)."),
        ReadoutSpec("eclipse_per_pair_p", kind="p_value", tests="eclipse_per_pair_excess",
                    description="Monte Carlo p of the excess under the per-pair tolerance (EXPLORATORY, not "
                                "calibrated).", origin="alpha 0.05 [ad hoc: conventional]"),
        ReadoutSpec("eclipse_primary_rule", kind="verdict", levels=rule_levels,
                    description="The rule called primary for this run."),
    )


ECLIPSE_READOUTS: Tuple[str, ...] = (
    "eclipse_n_clusters", "eclipse_sigma_c_median_nm", "eclipse_centroid_r95_median_nm", "eclipse_tau_derived_nm",
    "eclipse_rho_pooled_nm", "eclipse_rho_used_nm", "eclipse_rho_source", "eclipse_n_label_mean_hat",
    "eclipse_n_merged_suspect", "eclipse_sigma_c_source", "eclipse_n_ring_pairs", "eclipse_scalar_excess",
    "eclipse_scalar_z_a", "eclipse_scalar_p", "eclipse_per_pair_excess", "eclipse_per_pair_z_a", "eclipse_per_pair_p",
    "eclipse_primary_rule")


def _inputs_from_context(context: Any) -> ToleranceInputs:
    extras = getattr(context, "extras", {}) or {}
    base = extras.get("eclipse_inputs")
    inp = base if isinstance(base, ToleranceInputs) else ToleranceInputs()
    return inp.with_nena(extras.get("nena_nm"))


def _rings_readouts(run: Any, context: Any, inp: ToleranceInputs, rings_input: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the rings with THIS run's method and parameters and run the arc test under both rules (every adjacent
    pair, rings uncleaned: a cross-method comparison, not the review's primary test)."""
    from tools.multimethod import REFERENCE_METHOD, SlowMethodRefused, slow_refusal, _is_slow
    from tools.mps_columns import build_rings, rings_params_from
    from tools.mps_lumen import default_columns_params
    from tools.mps_unroll import UnrollParams, analyze_arc_columns
    if _is_slow(run.method) and not getattr(context, "allow_slow", False):
        raise SlowMethodRefused(slow_refusal(run.method, 6))
    cp = rings_input.get("columns_params") or default_columns_params()
    rp = rings_params_from(cp)
    kw: Dict[str, Any] = {}
    if run.method == REFERENCE_METHOD:
        rp = dataclasses.replace(rp, eps_nm=float(run.params.get("eps", rp.eps_nm)),
                                 min_samples=int(run.params.get("min_samples", rp.min_samples)))
    else:
        kw = dict(cluster_method=run.method, cluster_params=dict(run.params))
    context.check_cancelled("eclipse: rings")
    res = build_rings(np.asarray(rings_input["x"], dtype=np.float64), np.asarray(rings_input["y"], dtype=np.float64),
                      np.asarray(rings_input["z"], dtype=np.float64), frame=rings_input.get("frame"),
                      lp_lateral_nm=rings_input.get("lp_lateral_nm"), lpz_nm=rings_input.get("lpz_nm"), params=rp,
                      source_name=str(rings_input.get("source_name", "")),
                      pixel_size_nm=rings_input.get("pixel_size_nm"),
                      pixel_size_source=str(rings_input.get("pixel_size_source", "unknown")),
                      n_frames=rings_input.get("n_frames"), **kw)
    if len(res.rings) < 2:
        return {"eclipse_n_ring_pairs": 0}
    tol = tolerances_from_rings(res, inp, seed=int(getattr(context, "seed", 0)))
    stub: Any = SimpleNamespace(include_suspect=True, tau0_nm=inp.tau_scalar_nm, random_seed=int(getattr(context, "seed", 0)))
    up = UnrollParams(n_null=int(getattr(context, "n_replicates", 199)))
    context.check_cancelled("eclipse: arc test")
    scalar = analyze_arc_columns(res, stub, params=up, reference_curve="centroid_membrane")
    context.check_cancelled("eclipse: per-pair arc test")
    per_pair = per_pair_arc_columns(res, stub, tol, params=up, reference_curve="centroid_membrane")
    row = axon_row(tol, scalar, per_pair, rings_input.get("pairs"))
    n_pairs = 0 if scalar.joint is None else len(scalar.joint.pairs)
    return {"eclipse_n_ring_pairs": n_pairs, "eclipse_scalar_excess": row["scalar_excess"],
            "eclipse_scalar_z_a": row["scalar_z_A"], "eclipse_scalar_p": row["scalar_p_excess_uncalibrated"],
            "eclipse_per_pair_excess": row["per_pair_excess"], "eclipse_per_pair_z_a": row["per_pair_z_A"],
            "eclipse_per_pair_p": row["per_pair_p_excess_uncalibrated"]}


def eclipse_analysis(run: Any, context: Any) -> Dict[str, Any]:
    """
    The ``"eclipse"`` plug-in of ``tools.multimethod``: the centre errors of THIS method's clusters on its slab
    (``run.analysis``: x_slab / y_slab and the kept labels), with events when ``context.extras["event_id"]`` holds
    the event id of every localization the analysis received (sliced with ``analysis.slab_index``), sigma_loc from
    ``extras["nena_nm"]`` and the inputs from ``extras["eclipse_inputs"]`` (a :class:`ToleranceInputs`, for the
    calibrations). With ``extras["eclipse_rings_input"]`` (a mapping: x, y, z of the whole ROI in nm; optional
    frame, lp_lateral_nm, lpz_nm, n_frames, pixel_size_nm, columns_params, pairs) the rings are built with this
    method and the arc test runs under both rules; slow methods are refused unless ``allow_slow``.
    """
    from tools.method_overlay import kept_labels
    an = run.analysis
    inp = _inputs_from_context(context)
    labels = kept_labels(an)
    ids, members = clusters_from_labels(labels)
    extras = getattr(context, "extras", {}) or {}
    ev_all = extras.get("event_id")
    ev = None
    if ev_all is not None:
        ev_all = np.asarray(ev_all)
        idx = np.asarray(getattr(an, "slab_index", np.zeros(0)), dtype=np.intp)
        if idx.size == np.asarray(an.x_slab).size and (idx.size == 0 or int(idx.max()) < ev_all.size):
            ev = ev_all[idx]
    tol = axon_tolerances(an.x_slab, an.y_slab, {0: members}, event_id=ev, labels={0: ids.tolist()}, inputs=inp,
                          seed=int(getattr(context, "seed", 0)))
    g = tol.groups[0]
    out: Dict[str, Any] = {
        "eclipse_n_clusters": int(g.n_clusters),
        "eclipse_sigma_c_median_nm": float(np.median(g.sigma_c_nm)) if g.n_clusters else None,
        "eclipse_centroid_r95_median_nm": float(np.median(g.centroid_r95_nm)) if g.n_clusters else None,
        "eclipse_tau_derived_nm": tol.tau_derived_nm,
        "eclipse_rho_pooled_nm": tol.pool.rho_pooled_nm,
        "eclipse_rho_used_nm": float(tol.pool.rho_used_nm),
        "eclipse_rho_source": tol.pool.rho_source,
        "eclipse_n_label_mean_hat": _num(tol.pool.n_label_mean_hat),
        "eclipse_n_merged_suspect": int(np.count_nonzero(g.merged_suspect)),
        "eclipse_sigma_c_source": g.sigma_c_source,
        "eclipse_primary_rule": inp.primary_rule(),
    }
    rings_input = extras.get("eclipse_rings_input")
    if rings_input is not None:
        out.update(_rings_readouts(run, context, inp, rings_input))
    return out


def _cost(run: Any) -> float:
    """A rough cost of the slab readouts [ad hoc: 5 % of the run's own clustering time; they are cheap]. The rings
    path (``extras["eclipse_rings_input"]``) is NOT included: it re-clusters every ring a few times (about 12 x the
    run's clustering time [ad hoc]) plus two arc tests; the integration adds it when it enables that path."""
    return float(getattr(run, "seconds", 0.0) or 0.0) * 0.05


def register(replace: bool = True) -> None:
    """Register the ``"eclipse"`` plug-in with ``tools.multimethod`` (idempotent with ``replace``)."""
    from tools.multimethod import register_analysis
    register_analysis(ECLIPSE_ANALYSIS, eclipse_analysis, list(_readouts()), replace=replace,
                      description="P-R61: per-cluster centre errors, the axon's footprint, and the arc test under the "
                                  "scalar tau (primary) and the per-pair tolerance (exploratory).",
                      cost_seconds=_cost)


register()
