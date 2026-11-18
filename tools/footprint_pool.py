# -*- coding: utf-8 -*-
"""
The footprint pool of an axon's clusters -- ONE definition for the apparent size (``tools.mps_apparent_size``, the
"size" plug-in, the Batch) and the eclipse tolerance (``tools.mps_eclipse_tolerance``, the H-ECL review).

Why one function
----------------
Before 2026-10-29 the two modules pooled the footprint with two different recipes, and on the same axon they could
disagree about whether the footprint is measurable at all: 1.56 labels per cluster in the size table (measurable,
>= 1.5) against 1.49 in the eclipse review (not identifiable, < 2), finding (q) of the design review. They differed
in which clusters the labels per cluster were averaged over (the pool's clusters vs every cluster), in the label
threshold (1.5 vs 2), in the fewest clusters (5 vs 10), in the standard error (jackknife vs bootstrap) and in the
simulated E[sum w^2 | events] table (two Monte Carlo implementations). Both now call :func:`footprint_pool`.

What it computes (P-R61 sec. 3.2, method of moments)
----------------------------------------------------
    sigma_L^2 = sum_i (s_t,i^2 - sigma_loc^2) (N_i - 1) / N_i  /  sum_i (1 - E[sum w^2 | events_i])
    rho^2     = 4 (sigma_L^2 - sigma_link^2)

s_t the SD of each cluster's localizations ALONG the membrane, N its localizations, sigma_loc the precision (the
file's NeNA), sigma_link the linkage error, w the share of the cluster's localizations each label gave. It is the
root-mean-square footprint of the pooled clusters.

Which definition won, and why
-----------------------------
* The clusters pooled: every cluster with >= 2 localizations, a tangential SD and an event count, LESS the clusters
  the event layer marks as probable fusions (events / c_label > 2, ``tools.mps_events.fusion_flags``) -- user
  decision E2 (2026-10-23) -- LESS the caller's own geometric proxy of a fusion (``exclude``: the size's elongation
  > 2.5, decision E1; the eclipse's two-mode separation, decision D2 of the integration).
* The labels per cluster: mean events / c_label over THOSE clusters, at least 1 -- the size's definition. The
  identifiability of a pool is a property of the clusters it is computed on; averaging over clusters the pool
  leaves out (the eclipse's old definition) let the fused clusters' events lift the mean of a one-label axon above
  the threshold, which is exactly what E2 guards against.
* The threshold: 1.5 labels per cluster [simulation: SIZE_SIM sec. 8.2 and the pool code; user decision E2 names
  "the 1.5-labels filter"]. The eclipse module used 2 [user's decision 1a of P-R61, whose report itself is
  inconsistent: its sec. 7.3 says 2, its code and sec. 8.2 say 1.5]; the user's later decision E2 is taken as the
  one in force. Open for the user to confirm.
* The fewest clusters: 10 [ad hoc: the eclipse research's -- below 10 the moment estimate is too noisy to call a
  measurement; the size branch's 5 was also ad hoc]. The stricter of two ad hoc values.
* The standard error: delete-one jackknife over the pooled clusters [lit: Efron & Tibshirani 1993, ch. 11] --
  deterministic, no seed (the eclipse's 200-draw bootstrap needed one).
* E[sum w^2 | events]: the eclipse module's table (``sum_w2_table``: zero-truncated Poisson labels, geometric
  events and localizations, seeded from its arguments; the research's sims/common.py), its lambda grid starting at
  0.05 (one label). The size module's copy (grid from 0) is retired.

What stays the callers' own: the event counts (both link at 3 NeNA, 1 dark frame: ``tools.mps_events``), the
tangential frame (the size: the 400 nm membrane; the eclipse: the radial tangent about the ring's centre) and
sigma_loc (the size requires the file's NeNA; the eclipse falls back to its documented 10 nm).

No Qt.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "IDENTIFIABLE_LABELS", "MIN_POOL_CLUSTERS", "LOCS_PER_EVENT_DEFAULT", "LAMBDA_GRID", "EVENT_EDGES", "SUM_W2_SEED",
    "SUM_W2_DRAWS", "REGIME_LEVELS", "POOL_CODES", "PoolEstimate", "sum_w2_table", "lambda_from_mean_labels",
    "sum_w2_given_events", "labels_per_cluster", "moments_rho", "footprint_pool",
]

# Mean labels per pooled cluster from which the footprint is identifiable [simulation: SIZE_SIM sec. 8.2 and the
# pool code; user decision E2, 2026-10-23, "the 1.5-labels filter"]. Below it: "content only".
IDENTIFIABLE_LABELS = 1.5
# Fewest pooled clusters [ad hoc: the eclipse research's -- the smallest axon had K = 30, +-4-8 nm; below 10 the
# moment estimate is too noisy to call a measurement].
MIN_POOL_CLUSTERS = 10
# Localizations per event in the simulated sum-w^2 table [ad hoc: the project simulator's locs_per_fluor_mean].
LOCS_PER_EVENT_DEFAULT = 2.0
# The simulated E[sum w^2 | events] table [simulation: P-R61 sims/common.py sum_w2_given_events_table, seed
# 20261015; 100 000 simulated clusters per label prior, as the research's sim2].
SUM_W2_SEED = 20261015
SUM_W2_DRAWS = 100_000
LAMBDA_GRID: Tuple[float, ...] = (0.05, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
EVENT_EDGES: Tuple[int, ...] = (1, 4, 7, 10, 14, 18, 22, 27, 33, 40, 50, 60, 75, 90, 110, 140, 180, 250, 10 ** 9)

REGIME_LEVELS: Tuple[str, ...] = ("footprint measurable", "content only", "not evaluated")
# Why a pool is what it is (for the callers that word it their own way).
POOL_CODES: Tuple[str, ...] = ("ok", "no_events", "no_precision", "no_frame", "few_clusters", "not_identifiable",
                               "degenerate")


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


def labels_per_cluster(n_events: Any, c_label: float) -> Optional[float]:
    """Mean labels per cluster: mean events / c_label over the clusters given, at least 1 (a cluster holds one label
    or more); None without a finite event count."""
    ev = np.asarray(n_events, dtype=np.float64).reshape(-1)
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        return None
    return max(1.0, float(ev.mean()) / float(c_label))


def moments_rho(s_t: NDArray[np.float64], n_locs: NDArray[np.float64], sw2: NDArray[np.float64], sigma_loc_nm: float,
                sigma_link_nm: float) -> Optional[float]:
    """The method-of-moments footprint (module docstring); 0 when the moment falls below the linkage; None when no
    label variance is left to pool (sum of 1 - sum w^2 <= 0)."""
    den = float(np.sum(1.0 - sw2))
    if not den > 0.0:
        return None
    num = float(np.sum((s_t ** 2 - sigma_loc_nm ** 2) * (n_locs - 1.0) / n_locs))
    r2 = 4.0 * (num / den - sigma_link_nm ** 2)
    return float(math.sqrt(r2)) if r2 > 0.0 else 0.0


@dataclass
class PoolEstimate:
    """One footprint pool. ``rho_nm`` / ``se_nm`` (jackknife) the footprint and its standard error, None unless
    ``verdict`` is "footprint measurable"; ``labels_per_cluster`` the mean events / c_label of the pooled clusters
    (the ONE number both modules report and gate on); ``n_used`` the clusters pooled; ``n_excluded_fusion`` the
    clusters left out as event-layer fusions (E2) and ``n_excluded_other`` those the caller's ``exclude`` left out;
    ``lam`` the label prior used for sum w^2; ``code`` one of ``POOL_CODES`` and ``reason`` the words."""
    rho_nm: Optional[float]
    se_nm: Optional[float]
    labels_per_cluster: Optional[float]
    n_used: int
    n_excluded_fusion: int
    n_excluded_other: int
    verdict: str
    code: str
    reason: str
    lam: Optional[float]
    sigma_loc_nm: Optional[float]
    sigma_link_nm: float
    c_label: float

    @property
    def measurable(self) -> bool:
        return self.verdict == REGIME_LEVELS[0] and self.rho_nm is not None


def footprint_pool(sd_tangential: Any, n_locs: Any, n_events: Optional[Any], sigma_loc_nm: Optional[float], *,
                   sigma_link_nm: float, c_label: float, locs_per_event: float = LOCS_PER_EVENT_DEFAULT,
                   exclude: Optional[Any] = None, exclude_event_fusion: bool = True,
                   min_clusters: int = MIN_POOL_CLUSTERS,
                   identifiable_labels: float = IDENTIFIABLE_LABELS) -> PoolEstimate:
    """The axon's footprint pool (module docstring: the shared definition). ``n_events`` None: no events (frames or
    NeNA missing) -> "not evaluated". ``exclude`` (bool per cluster): the caller's own fusion proxy.
    ``exclude_event_fusion``: leave out the event layer's probable fusions (E2, on unless a caller studies the
    estimator itself)."""
    from tools.mps_events import FUSION_LABELS_THRESHOLD

    s_t = np.asarray(sd_tangential, dtype=np.float64).reshape(-1)
    nl = np.asarray(n_locs, dtype=np.float64).reshape(-1)
    c = float(c_label)
    link = float(sigma_link_nm)
    sig = None if sigma_loc_nm is None else float(sigma_loc_nm)

    def out(verdict: int, code: str, reason: str, *, rho: Optional[float] = None, se: Optional[float] = None,
            labels: Optional[float] = None, n_used: int = 0, n_fused: int = 0, n_other: int = 0,
            lam: Optional[float] = None) -> PoolEstimate:
        return PoolEstimate(rho, se, labels, int(n_used), int(n_fused), int(n_other), REGIME_LEVELS[verdict], code,
                            reason, lam, sig, link, c)

    if n_events is None:
        return out(2, "no_events", "no events (frames) or no NeNA: the footprint needs both")
    ev = np.asarray(n_events, dtype=np.float64).reshape(-1)
    if sig is None or not (math.isfinite(sig) and sig > 0):
        return out(2, "no_precision", "no NeNA (localization precision): the footprint needs it")
    if not np.isfinite(ev).any():
        return out(2, "no_events", "no emission events: the labels per cluster cannot be estimated")
    if not np.isfinite(s_t).any():
        return out(2, "no_frame", "no tangential SD (no membrane curve)")
    ok = np.isfinite(s_t) & (nl > 1) & np.isfinite(ev)
    n_fused = 0
    if exclude_event_fusion:
        with np.errstate(invalid="ignore"):
            fused = np.isfinite(ev) & (ev / c > float(FUSION_LABELS_THRESHOLD))
        n_fused = int(np.count_nonzero(ok & fused))
        ok &= ~fused
    n_other = 0
    if exclude is not None:
        ex = np.asarray(exclude, dtype=bool).reshape(-1)
        n_other = int(np.count_nonzero(ok & ex))
        ok &= ~ex
    n_ok = int(ok.sum())
    labels = labels_per_cluster(ev[ok], c) if n_ok else None
    if n_ok < int(min_clusters):
        return out(2, "few_clusters", f"{n_ok} usable clusters (fewer than {int(min_clusters)})", labels=labels,
                   n_used=n_ok, n_fused=n_fused, n_other=n_other)
    assert labels is not None
    if labels < float(identifiable_labels):
        return out(1, "not_identifiable",
                   f"{labels:.2f} labels per cluster (events / c_label) < {float(identifiable_labels):g}: the "
                   "footprint is not identifiable, only the content", labels=labels, n_used=n_ok, n_fused=n_fused,
                   n_other=n_other)
    lam = lambda_from_mean_labels(labels)
    sw2 = sum_w2_given_events(ev[ok], c, lam, float(locs_per_event))
    s_ok, n_ok_arr = s_t[ok], nl[ok]
    rho = moments_rho(s_ok, n_ok_arr, sw2, float(sig), link)
    if rho is None:
        return out(1, "degenerate", "no label variance to pool (every cluster looks like one label)", labels=labels,
                   n_used=n_ok, n_fused=n_fused, n_other=n_other, lam=lam)
    jk = []
    for i in range(n_ok):
        keep = np.arange(n_ok) != i
        r = moments_rho(s_ok[keep], n_ok_arr[keep], sw2[keep], float(sig), link)
        if r is not None:
            jk.append(r)
    jka = np.asarray(jk, dtype=np.float64)
    se = (math.sqrt((jka.size - 1) / jka.size * float(np.sum((jka - jka.mean()) ** 2))) if jka.size >= 2 else None)
    return out(0, "ok", "", rho=float(rho), se=se, labels=labels, n_used=n_ok, n_fused=n_fused, n_other=n_other,
               lam=lam)
