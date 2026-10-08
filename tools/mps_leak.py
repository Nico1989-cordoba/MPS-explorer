# -*- coding: utf-8 -*-
"""
Axial-leak diagnostics of the column analysis, the axial profile of the
columns and the guard sensitivity: milestone H4, module A, of
the research plan 03_plan.md (S3.1 ``AxialProfileFit``,
``LeakDiagnostics``; S3.2 ``leak_diagnostics``, ``axial_profile_test``;
S3.4 the diagnostics as validity checks without correction) on the rings
of ``tools.mps_columns.build_rings`` and the matches of
``tools.mps_matching.analyze_columns`` (DECISIONES D-10 the five
diagnostics, D-17 link radius, D-19 lpz, D-22 the seed, D-27 this
module's conventions).

Why this exists
---------------
Every ring boundary is a cut in z' and every cut leaks: a localization
of ring k measured with axial error beyond its distance to the boundary
lands in ring k +- 1, where DBSCAN can promote the leaked localizations
of one cluster to a cluster of their own -- a spurious CHILD sitting at
the parent's x', y', matched to it within tau_0 and counted as a column
(01_formalizacion E6 (a)). With P / 2 = 85 nm and sigma_z = 77 nm the
tail beyond a boundary is 1 - Phi(85 / 77) = 13.5 % of a cluster: 27 of
200 localizations, enough for min_samples = 10. The module puts the
five pre-specified diagnostics (01 S1.6 point 4) on every matched pair
and every ring pair, adds the principled per-pair form of (ii) (D-27:
the tail-versus-centred likelihood ratio of the child's z'), fits the
axial profile of every column (the mixture at the period against one
Gaussian, with a parametric bootstrap and the two sample-size bounds),
and rebuilds the rings at the guards of ``LeakParams.guards_nm`` so the
excess can be read against the dead zone alone. Nothing is corrected:
``leak_explained`` marks pairs, ``zeta_clean`` says what the excess is
without them, and the rules dict reports each diagnostic's verdict.

What is here
------------
``LeakParams``; ``PairLeak``, ``RingPairLeak``, ``AxialProfileFit``,
``GuardRun``, ``LeakDiagnostics``; ``child_tail_llr``,
``boundary_fraction``, ``boundary_fraction_expected``,
``shared_events``, ``pair_leak``, ``ring_pair_leak``,
``axon_period_nm``, ``fixed_offset_mixture_fit``, ``axial_profile_fit``,
``column_axial_profile``, ``guard_sensitivity``, ``analyze_leak``,
``analyze_columns_and_leak``. No Qt, no matplotlib; every function is
deterministic given the seed, which is ``ColumnsParams.random_seed``
carried by the ``AxonColumnsResult`` (D-22), never a field here. The
inputs are never modified (``analyze_columns_and_leak`` fills ``.leak``
of the result it creates).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom, norm, rankdata, spearmanr

from tools.mps_axial_precision import (
    leak_fraction,
    spurious_cluster_probability,
    structural_width_nm,
)
from tools.mps_columns import (
    Cluster,
    ColumnsParams,
    Ring,
    RingsResult,
    build_rings,
    rings_params_from,
)
from tools.mps_matching import (
    AxonColumnsResult,
    Column,
    RingGeometry,
    RingPairMatch,
    analyze_columns,
    eclipse_test,
    ring_geometry,
)

__all__ = [
    "LPZ_SPAWN_KEY",
    "PROFILE_SPAWN_KEY",
    "SHARED_SPAWN_KEY",
    "AxialProfileFit",
    "GuardRun",
    "LeakDiagnostics",
    "LeakParams",
    "PairLeak",
    "RingPairLeak",
    "analyze_columns_and_leak",
    "analyze_leak",
    "axial_profile_fit",
    "axon_period_nm",
    "boundary_fraction",
    "boundary_fraction_expected",
    "child_tail_llr",
    "child_tail_llr_two_parents",
    "column_axial_profile",
    "fixed_offset_mixture_fit",
    "guard_sensitivity",
    "pair_leak",
    "ring_pair_leak",
    "shared_events",
]

# Spawn keys of the seed sequences of this module, distinct from every
# (ring_a, ring_b) key of ``eclipse_test`` and from
# ``mps_matching.JOINT_SPAWN_KEY``: ``numpy.random.SeedSequence`` takes
# non-negative integers only, so each word is the integer its four ASCII
# bytes spell (big-endian), as H3 did for "join". The reassignment null
# of (i) and the permutation null of (ii) of one ring pair use
# ``SeedSequence(seed, spawn_key=(ring_a, ring_b, KEY))``; the profile of
# one column ``SeedSequence(seed, spawn_key=(PROFILE_SPAWN_KEY, start
# ring, cluster index))``.
SHARED_SPAWN_KEY = int.from_bytes(b"shar", "big")
LPZ_SPAWN_KEY = int.from_bytes(b"lpzp", "big")
PROFILE_SPAWN_KEY = int.from_bytes(b"prof", "big")

# Floor of a truncated-normal mass before its log (the specification's
# guard against -inf): 1e-300 is above the smallest normal double and
# its log, -690, keeps every ratio finite.
_MASS_FLOOR = 1e-300
# Floor of a fitted sigma (nm): a mixture component narrower than a
# thousandth of a nanometre is a degenerate fit, not a ring.
_MIN_SIGMA_NM = 1e-3
# Quadrature of the Fisher information in pi (``_fisher_information_pi``):
# 40 001 trapezoid points over the mixture's +-8 sigma range put the
# spacing at ~0.04 nm for sigma = 77 nm, where the trapezoid error on a
# smooth integrand is far below the 1e-3 relative the harness compares
# against its own 400 001-point rule.
_INFO_POINTS = 40001
_INFO_SIGMAS = 8.0
# Fewest values an axial profile is fitted on: two components and a
# shared sigma need at least three points to be more than a tautology.
_MIN_PROFILE_VALUES = 3
# EM stopping rule of the bootstrap replicates of the axial profile
# (``axial_profile_fit``): relative increment of the log-likelihood per
# SQUAREM cycle, and the cap on EM steps. The observed fit keeps the
# 1e-9 / 500 defaults of ``fixed_offset_mixture_fit``; a replicate's
# llr_b only has to be placed against llr_obs on a p-value grid of
# 1 / (B + 1), and under the one-Gaussian null the vanishing component's
# weight decays sublinearly (E_f1[f2 / f1] = 1, so w_k ~ 1 / k and the
# increments ~ 1 / k^2), which is why 1e-9 relative made most replicates
# run to the cap for a gain below 1e-3 in llr_b. 1e-7 relative (5e-5
# absolute at |loglik| ~ 500) leaves llr_b within ~1e-3 of its limit:
# measured against 1e-9 on 100 one-Gaussian samples of N = 100 and 20
# mixture samples of N = 500 (B = 99), not one p-value differs and the
# 120 fits take 14.2 s instead of 15.8 s -- the SQUAREM acceleration
# already does most of the work, so the gain is modest and the
# statistic is untouched.
_BOOT_TOL = 1e-7
_BOOT_MAX_ITER = 500
# Elements (rows x values x components) per block of ``_em_step_rows``:
# the batch of bootstrap rows is stepped in row blocks of about this
# many elements so that each block's work arrays (ten temporaries of
# 8 bytes per element) stay in cache. Bit-identical to one full-batch
# step (the rows are independent); measured 45 -> 17 ms per step on
# 600 rows x 600 values x 3 components and 18 -> 7 ms on 400 x 500 x 2,
# no change at 400 x 98 x 2.
_EM_CHUNK_ELEMENTS = 40_000
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


# ============================================================================
# Parameters
# ============================================================================

@dataclass
class LeakParams:
    """
    The parameters of H4 (its own dataclass: the frozen
    ``columns_params.yaml`` is not touched, D-27). Every value is
    recorded in the results.

    ``guards_nm`` are the ``RingsParams.guard_nm`` values the guard
    sensitivity rebuilds the rings with (the TOTAL width of a dead zone
    centred on each boundary, half per side; ``tools.mps_multisegment``);
    rule (iii) reads the entry at ``guard_rule_nm``. ``shared_threshold``
    marks a child whose shared-event fraction reaches it as
    ``shared_high`` (informative; H5 calibrates). ``n_null_shared`` is
    the number of replicates of the reassignment null of (i) and of the
    permutation p of (ii). ``boundary_band_sigma`` is the pre-specified
    band of (ii): within that many sigma_z of the boundary.
    ``lpz_scale`` multiplies every lpz before the leak fraction (D-19:
    the reported axial precision can be optimistic away from focus;
    the calibration is H5's, the factor is recorded).
    ``size_alpha``: ``p_size_leak >= size_alpha`` says the child's size
    is consistent with pure leak. ``profile_n_bootstrap`` replicates of
    the parametric bootstrap of the axial profile; a column enters rule
    (iv) when every cluster has ``profile_min_locs_per_cluster``
    localizations; ``profile_alpha`` its level. ``n_min_z`` is the z of
    the pre-specified n_min (two-sided 5 %) and ``n_min_power`` the
    target power of the Fisher-information sample size.

    The random seed is NOT a field: it is ``AxonColumnsResult.random_seed``
    (``ColumnsParams.random_seed``, D-22).
    """

    guards_nm: Tuple[float, ...] = (0.0, 30.0, 60.0, 90.0)
    guard_rule_nm: float = 60.0
    shared_threshold: float = 0.25
    n_null_shared: int = 1999
    boundary_band_sigma: float = 1.0
    lpz_scale: float = 1.0
    size_alpha: float = 0.01
    profile_n_bootstrap: int = 199
    profile_min_locs_per_cluster: int = 100
    profile_alpha: float = 0.05
    n_min_z: float = 1.959964
    n_min_power: float = 0.8

    def __post_init__(self) -> None:
        """Ranges: guards >= 0 and strictly increasing, fractions in
        [0, 1] (the power strictly inside), counts >= 1, scales > 0."""
        guards = tuple(float(g) for g in self.guards_nm)
        if not guards:
            raise ValueError("guards_nm must hold at least one guard")
        if any(not math.isfinite(g) or g < 0.0 for g in guards):
            raise ValueError(f"guards_nm must be finite and >= 0, got {self.guards_nm!r}")
        if any(b <= a for a, b in zip(guards[:-1], guards[1:])):
            raise ValueError(f"guards_nm must be strictly increasing, got {self.guards_nm!r}")
        self.guards_nm = guards
        if not (math.isfinite(self.guard_rule_nm) and self.guard_rule_nm >= 0.0):
            raise ValueError(f"guard_rule_nm must be >= 0, got {self.guard_rule_nm!r}")
        for name in ("shared_threshold", "size_alpha", "profile_alpha"):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
        if not (0.0 < float(self.n_min_power) < 1.0):
            raise ValueError(f"n_min_power must lie in (0, 1), got {self.n_min_power!r}")
        for name in ("n_null_shared", "profile_n_bootstrap"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)!r}")
        if int(self.profile_min_locs_per_cluster) < 0:
            raise ValueError(f"profile_min_locs_per_cluster must be >= 0, got "
                             f"{self.profile_min_locs_per_cluster!r}")
        for name in ("boundary_band_sigma", "lpz_scale", "n_min_z"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")


# ============================================================================
# Data model
# ============================================================================

@dataclass
class PairLeak:
    """
    The leak diagnostics of one matched pair (i in ring a, j in ring b).

    The CHILD is the cluster with fewer localizations (tie: the cluster
    of ring b), the PARENT the other; ``boundary_z_nm`` is the edge of
    the child's ring window facing the parent. ``boundary_fraction_child``
    is the pre-specified (ii): the fraction of the child's
    localizations within ``boundary_band_sigma`` x sigma_z of the child
    ring of that boundary; ``boundary_fraction_expected`` the same for
    N(centre, sigma_z) of the child ring truncated to the window and
    ``z_asymmetry`` their difference (> 0: loaded toward the boundary).
    ``llr_tail_child`` is the principled form (D-27): the log-likelihood
    ratio, summed over the child's localizations, of the TAIL model (the
    parent's Gaussian, per-localization width, truncated to the child's
    window) against the CENTRED model (the child ring's own component
    truncated to the same window); see ``child_tail_llr``. ``F_parent``
    is ``leak_fraction`` of the parent ring's centre against the
    boundary with the parent's localizations' lpz (x ``lpz_scale``) and
    the parent ring's structural width.

    The size rule is evaluated on the parent's ORIGINAL size, not on
    its detected count (D-27, amending the H4 specification's
    ``Bin(n_locs_parent, F)``): the detected parent has already lost its
    leaked localizations to BOTH of its window edges, so under the tail
    model n_locs_parent ~ n0 x P[z' in the parent's window] with n0
    the cluster's true size, and ``n_locs_parent_original`` = round(
    n_locs_parent / ``parent_retained_fraction``), where the retained
    fraction is 1 minus the Gaussian mass beyond each finite edge of
    the parent ring's window (``F_parent_far`` is the mass beyond the
    edge away from the child). With the detected count the binomial's
    mean sat 25-35 % below the truth and its sd shrank with n, so
    ``size_ok`` failed for most children of parents above ~150
    localizations (measured: 94 of 131 simulated children flagged with
    the detected count, against the 95 % acceptance; with the true n0
    the rejection rate is the nominal 1 %). ``expected_leaked_locs`` =
    n0 F, ``p_size_leak`` = P[Bin(n0, F) >= n_child] (is the child small
    enough to be pure leak), ``p_spurious`` = P[Bin(n0, F) >=
    min_samples] (could leak alone have made a cluster at all).
    ``shared_events`` = |events(parent) & events(child)| ignoring -1,
    ``shared_fraction`` = that over the child's events (NaN without
    events or with none in the child). ``z_tail`` = llr > 0, ``size_ok``
    = p_size_leak >= size_alpha, ``leak_explained`` = both; a diagnostic
    that cannot be computed (NaN) votes False, with a warning: it does
    not vote for leak.

    Limitation of the single-parent fields (stated, not corrected in
    them): the tail model has ONE parent. A child fed by both
    neighbouring rings at a chance-coincident arc (its z' pile at both
    window edges) or a real cluster that absorbed a leak is not that
    model, and the centred model can win on it (measured on the
    synthetic axons: 11-15 % of the majority-labelled children are
    bilateral and have llr < 0); such a pair is reported with
    ``z_tail`` False and is not counted in ``leak_explained``.

    The two-parent tail model (H5, D-27b) addresses exactly that case
    for a child that is the CHILD of both adjacent pairs -- the same
    (ring, cluster) is the child in a pair (k-1, k) and in a pair (k,
    k+1); nothing else qualifies. ``llr_tail_two_parents`` is the
    log-likelihood ratio of the MIXTURE of the two parents' truncated
    Gaussians (each with the widths of its own pair's tail model,
    ``child_tail_llr_two_parents``) with weights proportional to n0 x
    F_parent of each side (the expected leaked count of each parent:
    the mixture the two tails make in the child's window under pure
    leak) against the same centred model; ``z_tail_two_parents`` =
    llr > 0; ``size_ok_two_parents`` tests n_child against Bin(n0_below
    + n0_above, F_pooled) with F_pooled the weighted mean of the two
    leak fractions (so the mean is n0_below F_below + n0_above F_above),
    ``p_size_leak_two_parents`` that tail probability and
    ``weight_below_two_parents`` the lower parent's weight;
    ``leak_explained_any`` = ``leak_explained`` or (``z_tail_two_parents``
    and ``size_ok_two_parents``). For a child matched in ONE pair only
    the two-parent fields are NaN / None and ``leak_explained_any`` ==
    ``leak_explained``: the single-parent verdicts never change value.
    The two-parent fields are filled by ``analyze_leak`` (which sees both
    pairs), not by ``pair_leak``; the same values are written into the
    child's ``PairLeak`` of both pairs. What the model can separate: the
    mixture of two tails and the centred Gaussian differ by ~0.15 nats
    per localization (sd 0.55) at P = 170, sigma_z = 77, so the sign
    of llr is right in ~89 % of children of 20 localizations and ~98 %
    at 60 (validate_simulate_axon section 6); a bilateral child that
    is also a real cluster's absorbed leak is still not that model.

    What the detection figure means (re-review of 2026-09-24): the
    >= 95 % of 03 S3.6 is met on the HARNESS regime -- flat clusters of
    ~200 localizations, lpz 77 nm honest, no depth factor -- and on its
    unilateral population (113 of 116 in ``validate_leak.py``, 68 of 70
    on fresh seeds); counting the bilateral children the overall
    detection is 86-87 %. On a realistic
    generator (cluster sizes as dispersed as a real ring's, a range of
    reported lpz, depth factors on the true axial sd as in D-19) the
    same rule flags well under the harness figure of the matched
    children, more with a larger ``lpz_scale`` and more of the
    unilateral ones, with few true pairs flagged:
    the misses are children of purity 0.5-0.79 (bilateral, or merged
    with a real cluster: llr -19 to +54 but p_size 0) and a few pure
    children whose size exceeds Bin(n0, F) at the reported lpz. The
    acceptance is therefore regime-specific, not a property of the
    diagnostic: on data the size rule is limited by the lpz calibration
    (H5) and by the single-parent model, and the report says so.
    """

    ring_a: int
    ring_b: int
    i_a: int
    j_b: int
    d_nm: float
    parent_ring: int
    parent_index: int
    child_ring: int
    child_index: int
    n_locs_parent: int
    n_locs_child: int
    n_events_parent: Optional[int]
    n_events_child: Optional[int]
    shared_events: Optional[int]
    shared_fraction: float
    boundary_z_nm: float
    boundary_fraction_child: float
    boundary_fraction_expected: float
    z_asymmetry: float
    llr_tail_child: float
    F_parent: float
    expected_leaked_locs: float
    p_size_leak: float
    p_spurious: float
    F_parent_far: float
    parent_retained_fraction: float
    n_locs_parent_original: int
    size_ratio: float
    lpz_pair_median_nm: float
    shared_high: Optional[bool]
    z_tail: bool
    size_ok: bool
    leak_explained: bool
    warnings: List[str] = field(default_factory=list)
    llr_tail_two_parents: float = float("nan")
    z_tail_two_parents: Optional[bool] = None
    size_ok_two_parents: Optional[bool] = None
    p_size_leak_two_parents: float = float("nan")
    weight_below_two_parents: float = float("nan")
    leak_explained_any: bool = False


@dataclass
class RingPairLeak:
    """
    The diagnostics of one adjacent ring pair: its ``PairLeak`` list and
    the pair-level quantities of D-10.

    ``n_matched_clean`` = n_matched - n_leak_explained, ``E_dir_clean``
    that over min(K_a, K_b) and ``zeta_clean`` its standardisation
    against the UNCLEANED H3 null of the same ``RingPairMatch`` (the
    pre-specified form). That number is NOT conservative but biased
    low: the null replicates still contain the flagged children (a
    shifted child is matched to whatever it lands near), so the count
    removed from the observed side stays in E*, and zeta_clean sits
    about n_leak_explained / (sd* min K) below its true value
    (measured -4.8 to -9.6 on independent rings with leak, where the
    true excess is 0). It is kept because it is the specified field;
    read it as a lower bound of unknown depth. ``match_clean`` is the
    principled version (D-27): the eclipse test rerun with the same
    seed, tau and null settings on geometries whose flagged children
    are marked unusable on BOTH sides, so its E*, sd* and zeta
    (``zeta_clean_null``, ``p_excess_clean``) compare the cleaned count
    with a cleaned null; it is the run's own match when no pair was
    flagged (nothing to remove), None when the rerun was not asked for
    or a ring lost its contour. What the cleaned null still carries
    (measured on leak-only axons, re-review of 2026-09-24): the
    bilateral children are not flagged (single-parent model) and stay
    in on both sides, 0-7 per pair, so ``zeta_clean_null`` keeps a
    small positive residual -- mean +0.35 (se 0.19) over 26 pairs of
    independent rings with leak, ``p_excess_clean`` <= 0.05 in 3 of 26
    (not significant: P[>= 3 | Bin(26, 0.05)] = 0.14), against -4.3 to
    -9.2 for the specified ``zeta_clean`` on the same pairs. Read
    ``p_excess_clean`` knowing that the unflagged children are in it.
    ``shared_fraction_paired_mean`` is the mean over the matched pairs,
    ``shared_fraction_unpaired_mean`` the mean over every UNMATCHED cross
    pair of usable clusters (shared / min of the two event counts).
    ``p_shared`` is the pre-specified (i), a reassignment null (see
    ``ring_pair_leak``); ``spearman_matched_vs_lpz_rho`` / ``_p`` the
    pre-specified (ii) over the usable clusters of both rings pooled
    (one-sided, rho > 0 is what leak predicts). ``lpz_pair_median_nm``
    and ``E_excess`` = E_dir - E* are for cross-axon reporting.
    ``n_leak_explained_any`` counts the pairs with
    ``PairLeak.leak_explained_any`` (single-parent OR two-parent rule,
    H5) and ``fraction_leak_explained_any`` is that over ``n_matched``:
    equal to ``n_leak_explained`` / its fraction as ``ring_pair_leak``
    returns them, raised by ``analyze_leak`` once the two-parent pass
    has seen both adjacent pairs. The cleaned null (``match_clean``)
    removes the single-parent flags only (the pre-specified form).
    """

    ring_a: int
    ring_b: int
    tau_nm: float
    n_matched: int
    K_a: int
    K_b: int
    pairs: List[PairLeak]
    n_shared_high: Optional[int]
    n_z_tail: int
    n_size_ok: int
    n_leak_explained: int
    n_matched_clean: int
    E_dir_clean: float
    zeta_clean: float
    shared_fraction_paired_mean: float
    shared_fraction_unpaired_mean: float
    p_shared: float
    spearman_matched_vs_lpz_rho: float
    spearman_matched_vs_lpz_p: float
    lpz_pair_median_nm: float
    E_excess: float
    size_ratio_median: float
    fraction_size_ratio_below_half: float
    match_clean: Optional[RingPairMatch]
    zeta_clean_null: float
    p_excess_clean: float
    warnings: List[str] = field(default_factory=list)
    n_leak_explained_any: int = 0
    fraction_leak_explained_any: float = float("nan")


@dataclass
class AxialProfileFit:
    """
    The axial profile of one column of length >= 2: one z' value per
    event (the mean z' over ALL localizations of the event in the axon,
    no window cut: a localization of the event in the gap or in another
    ring counts), fitted by ``length`` Gaussians at mu + m x period (m =
    0..length-1) with a shared sigma and free weights against one
    Gaussian. ``llr`` = 2 (loglik2 - loglik1) >= 0 (the one-Gaussian
    solution is among the mixture's starts), ``p_bootstrap`` the
    parametric bootstrap under the one-Gaussian fit, ``pi_hat`` = 1 -
    max(weights): the mass outside the dominant component.

    ``n_min_events`` is the PRE-SPECIFIED form (03 S3.2): z^2 / (pi_hat^2
    (exp((period / sigma)^2) - 1)), the Cramer-Rao bound at pi -> 0 with
    mu, sigma and the period known -- a lower bound the fitted mixture
    (three free parameters more) cannot reach, and NOT the sample size
    of the bootstrap test. ``n_min_80_events`` is the Fisher-information
    sample size for power ``n_min_power`` of a Wald test of pi = 0 at
    two-sided 5 % with the numerical information I(pi) (mu, sigma
    known; ``_fisher_information_pi``). Both are inf when pi_hat = 0.
    Without events (``unit_is_events`` False) the values are per
    localization, with a warning. ``random_seed`` is the integer the
    bootstrap generator was made from (``default_rng(random_seed)``), so
    the draws can be rebuilt from this record alone.
    """

    members: List[Tuple[int, int]]
    length: int
    n_locs: int
    n_events: Optional[int]
    z_events_nm: NDArray[np.float64]
    unit_is_events: bool
    period_nm: float
    mu_nm: float
    sigma_nm: float
    weights: NDArray[np.float64]
    mu1_nm: float
    sigma1_nm: float
    loglik1: float
    loglik2: float
    llr: float
    p_bootstrap: float
    pi_hat: float
    n_min_events: float
    n_min_80_events: float
    meets_min_locs: bool
    n_bootstrap: int
    random_seed: int
    warnings: List[str] = field(default_factory=list)
    n_min_z: float = 1.959964
    n_min_power: float = 0.8


@dataclass
class GuardRun:
    """One rebuild of the rings at ``guard_nm`` (``guard_sensitivity``):
    the rings that came out, their cluster and localization counts, and
    the adjacent eclipse tests at the run's tau_0 with the run's null
    settings. No rings are kept (they are large); the counts and the
    matches are what rule (iii) reads."""

    guard_nm: float
    n_rings: int
    ring_indices: List[int]
    K_per_ring: List[int]
    n_locs_per_ring: List[int]
    matches: List[RingPairMatch]
    seconds: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class LeakDiagnostics:
    """
    Everything H4-A says about one axon. ``pairs`` holds one
    ``RingPairLeak`` per adjacent ``RingPairMatch`` of the columns
    result, ``k2`` the (k, k+2) matches of H3 (the persistence control,
    02 P7; referenced, not recomputed), ``guard`` the guard runs by
    guard (empty without the rebuild or its inputs, with a warning),
    ``profiles`` one ``AxialProfileFit`` per column of length >= 2 in
    ``cols.columns`` order. ``rules`` holds the verdict of each
    pre-specified diagnostic: "i_shared" (every p_shared > 0.05; None
    without events), "ii_lpz" (every pair has rho <= 0 or p > 0.05; None
    without lpz OR when rho is undefined on every pair, e.g. a constant
    lpz over the clusters -- an undefined correlation cannot vote, D-27),
    "iii_guard" (every zeta > 0 at ``guard_rule_nm``; None without that
    guard run), "iv_profile" (None: it needs H5's simulated-leak null;
    ``fraction_profiles_bimodal`` reports the fraction), "v_size_ratio"
    (the median size ratio over every matched pair >= 0.5). These are
    validity checks, not a correction (03 S3.4).

    What the rules can and cannot say (measured on synthetic axons
    with known truth, D-27 and the re-review of 2026-09-24):

    * (i) is small whenever ANY matched pair shares an event, and a
      true partner absorbs the leaked events of its neighbour, so with
      any leak at all (i) is False for real columns as much as for
      leak.
    * (ii) is a conjunction over the adjacent pairs of a per-pair
      one-sided test at the 5 % level, so its false-alarm rate on a
      leak-free axon is 1 - 0.95^n_pairs (9.8 % with two pairs;
      measured: a perfect-column axon read False on rho 0.189, p
      0.027). A pair whose rho is undefined does not vote: an
      indicator constant over the usable clusters (every cluster
      matched, or none) or a constant lpz leaves NaN, warned per pair,
      and the rule then rests on the remaining pairs (None when none
      is defined).
    * (iii) does NOT separate leak from columns at these sigma_z /
      period values: beyond the pre-specified dead zone the tail is
      still 1 - Phi((85 + 30) / 77) = 6.8 % of a cluster, 13.5 of 200
      localizations, above min_samples 10 in ~85 % of the clusters, so
      most children persist at g = 60 (measured 44 -> 32 on the
      harness axon) and the excess they carry stays positive: zeta at
      g = 60 was 1.6 / 1.2, 2.5 / 2.3 and 2.4 / 1.2 on three
      independent-ring axons with leak (2.3-3.3 / 0.5-2.0 at g = 90),
      and ``rules["iii_guard"]`` read True on every one of them. It
      holds for real columns (M5) and is reported because it is
      pre-specified; a True is not evidence against leak.
    * (v) assumes clusters of comparable size and reads False whenever
      the sizes are as dispersed as a real ring's (median size ratio
      0.33-0.43 on perfect columns of distinct molecules).

    On data none of them is a verdict; the per-pair ``leak_explained``
    and the cleaned matches are what the report reads.
    ``n_leak_explained_any`` / ``fraction_leak_explained_any`` are the
    axon-wide counts of ``PairLeak.leak_explained_any`` (single-parent
    OR two-parent tail model, H5), summed over the adjacent pairs like
    ``n_leak_explained``; NaN under the same condition as
    ``fraction_leak_explained``.
    """

    source_name: str
    tau0_nm: float
    period_nm: float
    pairs: List[RingPairLeak]
    k2: List[RingPairMatch]
    guard: Dict[float, GuardRun]
    profiles: List[AxialProfileFit]
    rules: Dict[str, Optional[bool]]
    fraction_profiles_bimodal: float
    n_leak_explained: int
    n_matched: int
    fraction_leak_explained: float
    params: LeakParams
    lpz_per_localization: bool
    events_available: bool
    seconds: float
    warnings: List[str] = field(default_factory=list)
    n_leak_explained_any: int = 0
    fraction_leak_explained_any: float = float("nan")


# ============================================================================
# Truncated-normal helpers and the per-pair statistics
# ============================================================================

def _log_truncated_mass(mu: Any, sigma: Any, lo: float, hi: float) -> NDArray[np.float64]:
    """
    log of the mass of N(mu, sigma) on [lo, hi] (mu, sigma broadcast),
    floored at log(``_MASS_FLOOR``).

    Computed in log space: when mu lies below the window's midpoint the
    window is in the upper tail and the mass is sf(lo) - sf(hi), whose
    two terms are small numbers that ``norm.logsf`` gives without the
    cancellation of two cdf values near 1; when mu lies above it, the
    mass is cdf(hi) - cdf(lo) from ``norm.logcdf``. Either way the log
    of the difference is log a + log1p(-exp(log b - log a)) with a the
    larger term, exact down to masses of 1e-300 (a window 37 sigma
    away). The first version of this helper picked the two branches
    the other way round, so a window more than ~8 sigma from mu (a tail
    width below ~10 nm at the 85 nm boundary distance) hit the floor at
    -690.8 instead of e.g. -39.2 and inflated the tail log-likelihood
    of such a localization by ~650 nats; the harness's identity check
    now covers widths down to 5 nm.
    """
    mu_a = np.asarray(mu, dtype=np.float64)
    s_a = np.asarray(sigma, dtype=np.float64)
    upper_tail = mu_a <= 0.5 * (lo + hi)
    log_big = np.where(upper_tail, norm.logsf(lo, mu_a, s_a), norm.logcdf(hi, mu_a, s_a))
    log_small = np.where(upper_tail, norm.logsf(hi, mu_a, s_a), norm.logcdf(lo, mu_a, s_a))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.exp(np.minimum(np.asarray(log_small - log_big, dtype=np.float64), 0.0))
        log_mass = np.asarray(log_big + np.log1p(-ratio), dtype=np.float64)
    log_mass = np.where(np.isfinite(log_mass), log_mass, -np.inf)
    return np.asarray(np.maximum(log_mass, math.log(_MASS_FLOOR)), dtype=np.float64)


def _truncated_logpdf(z: NDArray[np.float64], mu: Any, sigma: Any, lo: float, hi: float) -> NDArray[np.float64]:
    """log density at ``z`` of N(mu, sigma) truncated to [lo, hi]."""
    return np.asarray(norm.logpdf(z, mu, sigma) - _log_truncated_mass(mu, sigma, lo, hi), dtype=np.float64)


def child_tail_llr(
    z_child_nm: NDArray[np.float64],
    *,
    win_lo_nm: float,
    win_hi_nm: float,
    mu_centred_nm: float,
    sigma_centred_nm: float,
    mu_parent_nm: float,
    sigma_parent_nm: NDArray[np.float64],
) -> float:
    """
    Log-likelihood ratio of the TAIL model against the CENTRED model,
    summed over a child's localizations (D-27, the principled form of
    diagnostic (ii)).

    Both are densities on the child's ring window [win_lo, win_hi], so
    the ratio has no free parameter: the centred model is N(mu_centred,
    sigma_centred) truncated to the window (the child ring's own fitted
    component, whose width already contains the precision); the tail
    model is N(mu_parent, sigma_parent_i) truncated to the window, one
    width per localization (the parent ring's structural width in
    quadrature with the localization's own lpz), its truncation
    normaliser computed per localization. Under leak the child's z'
    pile against the boundary facing the parent and the tail model
    wins (llr > 0); under a real cluster they sit around mu_centred and
    the centred model wins. Guarded against -inf: log densities from
    ``norm.logpdf`` and the log of a truncated mass floored at 1e-300.

    Parameters
    ----------
    z_child_nm : (n,) z' of the child's localizations.
    win_lo_nm, win_hi_nm : the child's ring window.
    mu_centred_nm, sigma_centred_nm : centre and width of the child ring.
    mu_parent_nm : centre of the parent ring.
    sigma_parent_nm : (n,) or scalar, the tail width per localization.

    Returns
    -------
    float
        sum_i [log f_tail(z_i) - log f_centred(z_i)]; 0.0 with no
        localization; NaN when a width is not finite or not positive.
    """
    z = np.asarray(z_child_nm, dtype=np.float64).ravel()
    if z.size == 0:
        return 0.0
    s_p = np.broadcast_to(np.asarray(sigma_parent_nm, dtype=np.float64), z.shape)
    s_c = float(sigma_centred_nm)
    if not (math.isfinite(s_c) and s_c > 0.0) or not np.all(np.isfinite(s_p) & (s_p > 0.0)):
        return float("nan")
    lo, hi = float(win_lo_nm), float(win_hi_nm)
    tail = _truncated_logpdf(z, float(mu_parent_nm), s_p, lo, hi)
    centred = _truncated_logpdf(z, float(mu_centred_nm), s_c, lo, hi)
    return float(np.sum(tail - centred))


def child_tail_llr_two_parents(
    z_child_nm: NDArray[np.float64],
    *,
    win_lo_nm: float,
    win_hi_nm: float,
    mu_centred_nm: float,
    sigma_centred_nm: float,
    mu_below_nm: float,
    sigma_below_nm: Any,
    mu_above_nm: float,
    sigma_above_nm: Any,
    weight_below: float,
) -> float:
    """
    Log-likelihood ratio of the TWO-PARENT tail model against the
    CENTRED model, summed over a child's localizations (H5, D-27b: the
    case ``child_tail_llr`` states as its limitation).

    The tail model is the mixture w f_below + (1 - w) f_above of the two
    neighbouring rings' Gaussians, each truncated to the child's window
    with its own per-localization width (the parent ring's structural
    width in quadrature with the localization's lpz, as in the
    single-parent model) -- the density a child fed by both neighbours
    has in its window under pure leak, with w = ``weight_below`` the
    share of the lower parent (the caller sets it to n0_below F_below /
    (n0_below F_below + n0_above F_above), the expected leaked counts).
    The centred model is the child ring's own component truncated to
    the window, as in ``child_tail_llr``. The per-localization ratio is
    logaddexp(log w + log f_below, log(1 - w) + log f_above) - log
    f_centred, every log density from ``_truncated_logpdf`` (masses
    floored at 1e-300, so nothing is -inf); with w = 1 or 0 the mixture
    is the single parent and the value equals ``child_tail_llr`` of it.

    Parameters
    ----------
    z_child_nm : (n,) z' of the child's localizations.
    win_lo_nm, win_hi_nm : the child's ring window.
    mu_centred_nm, sigma_centred_nm : centre and width of the child ring.
    mu_below_nm, sigma_below_nm : centre of the ring below and the tail
        width per localization ((n,) or scalar).
    mu_above_nm, sigma_above_nm : the same for the ring above.
    weight_below : w in [0, 1].

    Returns
    -------
    float
        sum_i [log f_mix(z_i) - log f_centred(z_i)]; 0.0 with no
        localization; NaN when a width is not finite or not positive
        or the weight is outside [0, 1].
    """
    z = np.asarray(z_child_nm, dtype=np.float64).ravel()
    if z.size == 0:
        return 0.0
    w = float(weight_below)
    s_b = np.broadcast_to(np.asarray(sigma_below_nm, dtype=np.float64), z.shape)
    s_a = np.broadcast_to(np.asarray(sigma_above_nm, dtype=np.float64), z.shape)
    s_c = float(sigma_centred_nm)
    if (not (math.isfinite(s_c) and s_c > 0.0) or not np.all(np.isfinite(s_b) & (s_b > 0.0))
            or not np.all(np.isfinite(s_a) & (s_a > 0.0)) or not (math.isfinite(w) and 0.0 <= w <= 1.0)):
        return float("nan")
    lo, hi = float(win_lo_nm), float(win_hi_nm)
    below = _truncated_logpdf(z, float(mu_below_nm), s_b, lo, hi)
    above = _truncated_logpdf(z, float(mu_above_nm), s_a, lo, hi)
    centred = _truncated_logpdf(z, float(mu_centred_nm), s_c, lo, hi)
    with np.errstate(divide="ignore"):
        log_w, log_1w = np.log(w), np.log(1.0 - w)
    mixture = np.logaddexp(log_w + below, log_1w + above)
    return float(np.sum(mixture - centred))


def boundary_fraction(z_nm: NDArray[np.float64], boundary_z_nm: float, band_nm: float) -> float:
    """Fraction of the values within ``band_nm`` (inclusive) of the
    boundary; NaN on an empty array (no fraction exists, and 0 would
    read as 'nothing near the boundary')."""
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    if z.size == 0:
        return float("nan")
    return float(np.mean(np.abs(z - float(boundary_z_nm)) <= float(band_nm)))


def boundary_fraction_expected(
    *,
    win_lo_nm: float,
    win_hi_nm: float,
    mu_nm: float,
    sigma_nm: float,
    boundary_z_nm: float,
    band_nm: float,
) -> float:
    """
    The fraction ``boundary_fraction`` would give for N(mu, sigma)
    truncated to [win_lo, win_hi]: the Gaussian mass of the band
    [boundary - band, boundary + band] clipped to the window, over the
    Gaussian mass of the window (closed form with ``norm.cdf``). 0 when
    the clipped band is empty; NaN when the window has no mass or sigma
    is not positive.
    """
    lo, hi = float(win_lo_nm), float(win_hi_nm)
    mu, s = float(mu_nm), float(sigma_nm)
    if not (math.isfinite(s) and s > 0.0) or not (hi > lo):
        return float("nan")
    a = max(float(boundary_z_nm) - float(band_nm), lo)
    c = min(float(boundary_z_nm) + float(band_nm), hi)
    if c <= a:
        return 0.0
    window = float(norm.cdf(hi, mu, s) - norm.cdf(lo, mu, s))
    if not window > 0.0:
        return float("nan")
    return float((norm.cdf(c, mu, s) - norm.cdf(a, mu, s)) / window)


def _event_set(event_id: NDArray[np.int64], loc_index: NDArray[np.intp]) -> NDArray[np.int64]:
    """The distinct event ids >= 0 of the localizations ``loc_index``."""
    ids = np.asarray(event_id, dtype=np.int64)[np.asarray(loc_index, dtype=np.intp)]
    return np.unique(ids[ids >= 0])


def shared_events(
    event_id: NDArray[np.int64],
    loc_index_a: NDArray[np.intp],
    loc_index_b: NDArray[np.intp],
) -> Tuple[int, int, int]:
    """
    (|Ea & Eb|, |Ea|, |Eb|): the distinct event ids of the localizations
    ``loc_index_a`` and ``loc_index_b`` (ids -1 = no event, ignored) and
    how many they share. An event is shared when at least one of its
    localizations is in each cluster -- the signature of a leaked
    molecule, seen on both sides of the boundary within one blink.
    """
    ea = _event_set(event_id, loc_index_a)
    eb = _event_set(event_id, loc_index_b)
    return int(np.intersect1d(ea, eb, assume_unique=True).size), int(ea.size), int(eb.size)


def _ring_by_index(res: RingsResult, index: int) -> Ring:
    """The ring of ``res`` with ``Ring.index == index``."""
    for ring in res.rings:
        if int(ring.index) == int(index):
            return ring
    raise ValueError(f"no ring with index {index} in the rings result "
                     f"(indices {[r.index for r in res.rings]})")


def _lpz_of(cluster: Cluster, lpz_nm: Optional[NDArray[np.float64]], scale: float) -> Optional[NDArray[np.float64]]:
    """The per-localization lpz of a cluster (x scale), or None."""
    if lpz_nm is None:
        return None
    return np.asarray(lpz_nm, dtype=np.float64)[np.asarray(cluster.loc_index, dtype=np.intp)] * float(scale)


def _median_of_two(a: float, b: float) -> float:
    """Median of two values ignoring NaN; NaN when both are NaN."""
    vals = [float(v) for v in (a, b) if math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def _tail_width_inputs(
    parent_ring: Ring,
    parent: Cluster,
    child: Cluster,
    n_child: int,
    lpz_nm: Optional[NDArray[np.float64]],
    scale: float,
) -> Tuple[float, Optional[NDArray[np.float64]], Optional[NDArray[np.float64]], List[str]]:
    """
    The widths of one parent's tail model for one child (the block of
    ``pair_leak`` that ``PairLeak`` describes, in one place so that the
    two-parent model of H5 uses the same rule per side): (s_struct,
    lpz_child, lpz_parent, warnings) with s_struct the parent ring's
    structural width (``structural_width_nm`` of its component width
    against its median lpz x scale; 0 when it has no lpz), lpz_child the
    child's per-localization lpz x scale (``lpz_nm``), else the child
    cluster's median, else the parent ring's median (each fallback
    warned), lpz_parent the parent's per-localization lpz x scale, else
    its cluster median (Jensen lower bound, warned); and, with no lpz
    anywhere, the parent ring's fitted sigma_z as both widths with
    s_struct 0 (D-19, warned). None where nothing could be found. Same
    arithmetic and the same warnings, in the same order, as the H4
    version of ``pair_leak``.
    """
    warnings_: List[str] = []
    parent_lpz_median = float(parent_ring.lpz_median_nm) * scale
    s_struct = structural_width_nm(float(parent_ring.sigma_z_nm), parent_lpz_median)
    sigma_z_parent = float(parent_ring.sigma_z_nm)
    lpz_child = _lpz_of(child, lpz_nm, scale)
    if lpz_child is None:
        if math.isfinite(float(child.lpz_median_nm)):
            lpz_child = np.full(n_child, float(child.lpz_median_nm) * scale)
            warnings_.append("no per-localization lpz: the tail model uses the child cluster's median lpz")
        elif math.isfinite(float(parent_ring.lpz_median_nm)):
            lpz_child = np.full(n_child, parent_lpz_median)
            warnings_.append("no per-localization lpz and no child lpz: the tail model uses the parent ring's median lpz")
    if not math.isfinite(s_struct):
        # No lpz on the parent ring at all: the structural width cannot be
        # separated from the precision; the component width is used whole.
        s_struct = 0.0
        warnings_.append("parent ring without lpz: structural width taken as 0 (the tail width is the lpz alone)")
    lpz_parent = _lpz_of(parent, lpz_nm, scale)
    if lpz_parent is None and math.isfinite(float(parent.lpz_median_nm)):
        lpz_parent = np.array([float(parent.lpz_median_nm) * scale])
        warnings_.append("no per-localization lpz: F_parent from the parent cluster's median lpz (Jensen lower bound)")
    if (lpz_child is None or lpz_parent is None) and math.isfinite(sigma_z_parent) and sigma_z_parent > 0.0:
        # D-19: with no axial precision anywhere, the parent ring's fitted
        # component width IS the leak model's width (it contains the
        # precision and the structure, so the structural term is 0 here);
        # a diagnostic that stays NaN would read as "no leak" downstream.
        if lpz_child is None:
            lpz_child = np.full(n_child, sigma_z_parent)
        if lpz_parent is None:
            lpz_parent = np.array([sigma_z_parent])
        s_struct = 0.0
        warnings_.append("no lpz anywhere: the tail model and F_parent use the parent ring's fitted sigma_z as the "
                         "width (D-19), structural width 0")
    return s_struct, lpz_child, lpz_parent, warnings_


def pair_leak(
    res: RingsResult,
    match: RingPairMatch,
    i_a: int,
    j_b: int,
    *,
    params: LeakParams,
    lpz_nm: Optional[NDArray[np.float64]] = None,
    min_samples: int,
) -> PairLeak:
    """
    The ``PairLeak`` of the matched pair (cluster ``i_a`` of ring
    ``match.ring_a``, cluster ``j_b`` of ring ``match.ring_b``).

    Conventions (see ``PairLeak``): the child is the cluster with fewer
    localizations (tie: ring b's), the window is the child's ring
    window and the boundary its edge facing the parent (``z_lo`` when
    the parent ring's centre is below the child ring's, ``z_hi`` when
    above). The tail model's widths are sqrt(sigma_struct,p^2 + (lpz_i x
    lpz_scale)^2) with sigma_struct,p = ``structural_width_nm`` of the
    parent ring (its component width against its median lpz x
    lpz_scale) and lpz_i the child localization's own lpz when
    ``lpz_nm`` is given, else the child cluster's median, else the
    parent ring's median (each fallback warned). ``F_parent`` averages
    the tail over the parent's localizations' lpz when given (the exact
    mean, ``mps_axial_precision.leak_fraction``), else uses the parent
    cluster's median lpz as a scalar (the Jensen lower bound, warned),
    with the parent RING's centre z' as the cluster's axial centre (the
    tail model's convention: a leaked child is the parent ring's
    Gaussian seen beyond the boundary). Without any lpz (no array, no
    cluster median) the parent ring's fitted ``sigma_z_nm`` is the
    width of both the tail model and F (D-19), with a warning; the
    sizes are tested on the parent's original size (``PairLeak``).
    ``min_samples`` is the DBSCAN minimum of the run
    (``ColumnsParams.min_samples``) for ``p_spurious``. ``d_nm`` is the
    matched distance recorded in ``match`` (the centroid distance when
    the pair is not in it).
    """
    warnings_: List[str] = []
    ring_a = _ring_by_index(res, match.ring_a)
    ring_b = _ring_by_index(res, match.ring_b)
    ia, jb = int(i_a), int(j_b)
    a, b = ring_a.clusters[ia], ring_b.clusters[jb]
    hit = np.flatnonzero((np.asarray(match.i_a) == ia) & (np.asarray(match.j_b) == jb))
    if hit.size:
        d_nm = float(np.asarray(match.d_nm, dtype=np.float64)[hit[0]])
    else:
        d_nm = float(np.hypot(*(np.asarray(b.centroid_nm, dtype=np.float64) -
                                np.asarray(a.centroid_nm, dtype=np.float64))))
    if int(b.n_locs) <= int(a.n_locs):
        child, child_ring, child_index = b, ring_b, jb
        parent, parent_ring, parent_index = a, ring_a, ia
    else:
        child, child_ring, child_index = a, ring_a, ia
        parent, parent_ring, parent_index = b, ring_b, jb
    n_parent, n_child = int(parent.n_locs), int(child.n_locs)
    scale = float(params.lpz_scale)
    win_lo, win_hi = float(child_ring.z_lo_nm), float(child_ring.z_hi_nm)
    below = float(parent_ring.centre_z_nm) < float(child_ring.centre_z_nm)
    boundary = win_lo if below else win_hi

    # --- events -----------------------------------------------------------
    n_ev_parent: Optional[int] = None
    n_ev_child: Optional[int] = None
    shared: Optional[int] = None
    shared_frac = float("nan")
    shared_high: Optional[bool] = None
    if res.event_id is not None:
        shared, n_ev_parent, n_ev_child = shared_events(res.event_id, parent.loc_index, child.loc_index)
        shared_frac = shared / n_ev_child if n_ev_child > 0 else float("nan")
        shared_high = bool(shared_frac >= float(params.shared_threshold)) if math.isfinite(shared_frac) else None
        if n_ev_child == 0:
            warnings_.append("child cluster with no event (every id -1): shared_fraction NaN")

    # --- (ii) pre-specified band and the tail-versus-centred LLR ---------
    sigma_c = float(child_ring.sigma_z_nm)
    band = float(params.boundary_band_sigma) * sigma_c
    z_child = np.asarray(child.z_values_nm, dtype=np.float64)
    frac_child = boundary_fraction(z_child, boundary, band)
    frac_expected = boundary_fraction_expected(win_lo_nm=win_lo, win_hi_nm=win_hi, mu_nm=float(child_ring.centre_z_nm),
                                               sigma_nm=sigma_c, boundary_z_nm=boundary, band_nm=band)
    s_struct, lpz_child, lpz_parent, width_warnings = _tail_width_inputs(parent_ring, parent, child, int(z_child.size),
                                                                          lpz_nm, scale)
    warnings_.extend(width_warnings)
    if lpz_child is None:
        llr = float("nan")
        warnings_.append("no lpz and no finite parent ring width: llr_tail_child NaN, z_tail False")
    else:
        llr = child_tail_llr(z_child, win_lo_nm=win_lo, win_hi_nm=win_hi, mu_centred_nm=float(child_ring.centre_z_nm),
                             sigma_centred_nm=sigma_c, mu_parent_nm=float(parent_ring.centre_z_nm),
                             sigma_parent_nm=np.sqrt(s_struct ** 2 + lpz_child ** 2))
        if not math.isfinite(llr):
            warnings_.append("llr_tail_child not finite (child ring width or lpz not positive): z_tail False")

    # --- (iii)/(v) F and sizes on the parent's original size -------------
    centre_p = float(parent_ring.centre_z_nm)
    f_far = retained = float("nan")
    n0 = n_parent
    if lpz_parent is None:
        f_parent = float("nan")
    else:
        f_parent = leak_fraction(centre_p, boundary, lpz_parent, s_struct)
        p_lo, p_hi = float(parent_ring.z_lo_nm), float(parent_ring.z_hi_nm)
        f_lo = leak_fraction(centre_p, p_lo, lpz_parent, s_struct) if math.isfinite(p_lo) else 0.0
        f_hi = leak_fraction(centre_p, p_hi, lpz_parent, s_struct) if math.isfinite(p_hi) else 0.0
        f_far = f_lo if below else f_hi
        retained = 1.0 - f_lo - f_hi
        if math.isfinite(retained) and retained > 0.0:
            n0 = max(n_parent, int(round(n_parent / retained)))
        else:
            retained = float("nan")
            warnings_.append("parent window retains no mass under the tail model: the detected count is used as the "
                             "parent's original size")
    if math.isfinite(f_parent):
        expected = n0 * f_parent
        p_size = float(binom.sf(n_child - 1, n0, f_parent))
        p_spur = spurious_cluster_probability(n0, f_parent, int(min_samples))
    else:
        expected = p_size = p_spur = float("nan")
        warnings_.append("F_parent NaN (no lpz and no finite parent ring width): p_size_leak and p_spurious NaN, "
                         "size_ok False")
    z_tail = bool(math.isfinite(llr) and llr > 0.0)
    size_ok = bool(math.isfinite(p_size) and p_size >= float(params.size_alpha))
    return PairLeak(
        ring_a=int(match.ring_a), ring_b=int(match.ring_b), i_a=ia, j_b=jb, d_nm=d_nm,
        parent_ring=int(parent_ring.index), parent_index=parent_index,
        child_ring=int(child_ring.index), child_index=child_index,
        n_locs_parent=n_parent, n_locs_child=n_child,
        n_events_parent=n_ev_parent, n_events_child=n_ev_child,
        shared_events=shared, shared_fraction=float(shared_frac), boundary_z_nm=boundary,
        boundary_fraction_child=float(frac_child), boundary_fraction_expected=float(frac_expected),
        z_asymmetry=float(frac_child - frac_expected), llr_tail_child=float(llr),
        F_parent=float(f_parent), expected_leaked_locs=float(expected), p_size_leak=float(p_size),
        p_spurious=float(p_spur), F_parent_far=float(f_far), parent_retained_fraction=float(retained),
        n_locs_parent_original=int(n0), size_ratio=n_child / n_parent if n_parent else float("nan"),
        lpz_pair_median_nm=_median_of_two(a.lpz_median_nm, b.lpz_median_nm),
        shared_high=shared_high, z_tail=z_tail, size_ok=size_ok, leak_explained=bool(z_tail and size_ok),
        warnings=warnings_, leak_explained_any=bool(z_tail and size_ok),
    )


# ============================================================================
# Two-parent tail model (H5, D-27b)
# ============================================================================

def _two_parent_fields(
    res: RingsResult,
    below: PairLeak,
    above: PairLeak,
    *,
    params: LeakParams,
    lpz_nm: Optional[NDArray[np.float64]],
) -> Dict[str, Any]:
    """
    The two-parent fields of a child that is the child of both adjacent
    pairs: ``below`` its ``PairLeak`` with the parent in the ring below
    (ring k-1), ``above`` the one with the parent above (ring k+1). The
    child's z', window and centred model are those of its ring (the
    same in both pairs); each side's tail widths come from
    ``_tail_width_inputs`` with that side's parent (the very rule of
    its single-parent llr) and its weight is n0 x F_parent of that
    side (``PairLeak.n_locs_parent_original`` x ``F_parent``: the
    expected leaked count). Returns the dict of ``PairLeak`` field
    values; NaN / None when a weight or a width is unavailable (F NaN
    on a side, no lpz at all) -- the child then keeps its single-parent
    verdict alone.
    """
    child_ring = _ring_by_index(res, below.child_ring)
    child = child_ring.clusters[int(below.child_index)]
    z_child = np.asarray(child.z_values_nm, dtype=np.float64)
    n_child = int(child.n_locs)
    scale = float(params.lpz_scale)
    win_lo, win_hi = float(child_ring.z_lo_nm), float(child_ring.z_hi_nm)
    sigma_c = float(child_ring.sigma_z_nm)
    mu_c = float(child_ring.centre_z_nm)
    nothing: Dict[str, Any] = dict(llr_tail_two_parents=float("nan"), z_tail_two_parents=None,
                                   size_ok_two_parents=None, p_size_leak_two_parents=float("nan"),
                                   weight_below_two_parents=float("nan"))
    widths: List[NDArray[np.float64]] = []
    centres: List[float] = []
    for side in (below, above):
        parent_ring = _ring_by_index(res, side.parent_ring)
        parent = parent_ring.clusters[int(side.parent_index)]
        s_struct, lpz_child, _lpz_parent, _w = _tail_width_inputs(parent_ring, parent, child, int(z_child.size),
                                                                  lpz_nm, scale)
        if lpz_child is None:
            return nothing
        widths.append(np.sqrt(s_struct ** 2 + lpz_child ** 2))
        centres.append(float(parent_ring.centre_z_nm))
    w_b = float(below.n_locs_parent_original) * float(below.F_parent)
    w_a = float(above.n_locs_parent_original) * float(above.F_parent)
    if not (math.isfinite(w_b) and math.isfinite(w_a)) or not (w_b + w_a > 0.0):
        return nothing
    weight_below = w_b / (w_b + w_a)
    llr = child_tail_llr_two_parents(z_child, win_lo_nm=win_lo, win_hi_nm=win_hi, mu_centred_nm=mu_c,
                                     sigma_centred_nm=sigma_c, mu_below_nm=centres[0], sigma_below_nm=widths[0],
                                     mu_above_nm=centres[1], sigma_above_nm=widths[1], weight_below=weight_below)
    n0_total = int(below.n_locs_parent_original) + int(above.n_locs_parent_original)
    f_pooled = (w_b + w_a) / n0_total if n0_total > 0 else float("nan")
    p_size = float(binom.sf(n_child - 1, n0_total, f_pooled)) if math.isfinite(f_pooled) else float("nan")
    return dict(llr_tail_two_parents=float(llr), z_tail_two_parents=bool(math.isfinite(llr) and llr > 0.0),
                size_ok_two_parents=bool(math.isfinite(p_size) and p_size >= float(params.size_alpha)),
                p_size_leak_two_parents=p_size, weight_below_two_parents=float(weight_below))


def _apply_two_parent_model(
    res: RingsResult,
    pairs: Sequence[RingPairLeak],
    *,
    params: LeakParams,
    lpz_nm: Optional[NDArray[np.float64]],
) -> List[RingPairLeak]:
    """
    The two-parent pass over an axon's ``RingPairLeak`` list (H5): every
    (ring, cluster) that is the CHILD in exactly two adjacent pairs --
    one with its parent below, one with its parent above -- gets the
    ``_two_parent_fields`` written into its ``PairLeak`` of BOTH pairs
    and ``leak_explained_any`` raised accordingly; every other pair
    keeps ``leak_explained_any`` == ``leak_explained``. Each
    ``RingPairLeak`` then carries ``n_leak_explained_any`` and
    ``fraction_leak_explained_any`` over its own pairs. New objects
    (``dataclasses.replace``); the inputs are not modified.
    """
    by_child: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for q, rp in enumerate(pairs):
        for i, p in enumerate(rp.pairs):
            by_child.setdefault((int(p.child_ring), int(p.child_index)), []).append((q, i))
    updates: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for key, where in by_child.items():
        if len(where) != 2:
            continue
        p0, p1 = pairs[where[0][0]].pairs[where[0][1]], pairs[where[1][0]].pairs[where[1][1]]
        if p0.parent_ring == p1.parent_ring or {p0.parent_ring, p1.parent_ring} != {key[0] - 1, key[0] + 1}:
            continue
        below, above = (p0, p1) if p0.parent_ring < p1.parent_ring else (p1, p0)
        fields = _two_parent_fields(res, below, above, params=params, lpz_nm=lpz_nm)
        for loc in where:
            updates[loc] = fields
    out: List[RingPairLeak] = []
    for q, rp in enumerate(pairs):
        new_pairs: List[PairLeak] = []
        for i, p in enumerate(rp.pairs):
            fields = updates.get((q, i))
            if fields is None:
                new_pairs.append(dataclasses.replace(p, leak_explained_any=bool(p.leak_explained)))
            else:
                any_ = bool(p.leak_explained or (fields["z_tail_two_parents"] and fields["size_ok_two_parents"]))
                new_pairs.append(dataclasses.replace(p, leak_explained_any=any_, **fields))
        n_any = sum(1 for p in new_pairs if p.leak_explained_any)
        out.append(dataclasses.replace(rp, pairs=new_pairs, n_leak_explained_any=int(n_any),
                                       fraction_leak_explained_any=(n_any / rp.n_matched if rp.n_matched else float("nan"))))
    return out


# ============================================================================
# Ring pair: reassignment null, Spearman, clean excess
# ============================================================================

def _usable_mask(mask: Optional[NDArray[np.bool_]], k: int) -> NDArray[np.bool_]:
    """A match's usable mask, or all True when it carries none."""
    if mask is None:
        return np.ones(k, dtype=bool)
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size != k:
        raise ValueError(f"usable mask of {m.size} entries for {k} clusters")
    return m


def _event_matrix(
    event_id: NDArray[np.int64],
    clusters_a: Sequence[Cluster],
    clusters_b: Sequence[Cluster],
) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """
    (S, n_a, n_b): S[i, j] = events shared by cluster i of ring a and
    cluster j of ring b, n_a[i] / n_b[j] their event counts (ids -1
    ignored). Built through two dense event-by-cluster incidence
    matrices over the union of both rings' events, so every cross pair
    comes out of one matrix product instead of K_a x K_b set
    intersections.
    """
    ev = np.asarray(event_id, dtype=np.int64)
    sets_a = [_event_set(ev, cl.loc_index) for cl in clusters_a]
    sets_b = [_event_set(ev, cl.loc_index) for cl in clusters_b]
    union = np.unique(np.concatenate(sets_a + sets_b + [np.zeros(0, dtype=np.int64)]))
    a_inc = np.zeros((union.size, len(sets_a)), dtype=np.int64)
    b_inc = np.zeros((union.size, len(sets_b)), dtype=np.int64)
    for i, s in enumerate(sets_a):
        a_inc[np.searchsorted(union, s), i] = 1
    for j, s in enumerate(sets_b):
        b_inc[np.searchsorted(union, s), j] = 1
    shared = np.asarray(a_inc.T @ b_inc, dtype=np.int64)
    return shared, a_inc.sum(axis=0).astype(np.int64), b_inc.sum(axis=0).astype(np.int64)


def _spearman_with_permutation(
    indicator: NDArray[np.float64],
    lpz: NDArray[np.float64],
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    (rho, p): Spearman's rho between the has-a-partner indicator and
    the clusters' median lpz (``scipy.stats.spearmanr``, average ranks
    on ties) and its one-sided permutation p (rho > 0 is what leak
    predicts: a cluster with a worse axial precision leaks more and
    therefore has a partner more often): ``n_perm`` permutations of the
    indicator, p = (#{rho_perm >= rho} + 1) / (n_perm + 1) (Phipson and
    Smyth). Only clusters with a finite lpz enter; NaN, NaN with fewer
    than 3 of them or when either variable is constant (rho undefined).
    The permuted rhos are Pearson correlations of the standardised
    ranks, which is Spearman's rho to rounding; the count uses the
    observed value computed the same way, with 1e-12 slack.
    """
    ind = np.asarray(indicator, dtype=np.float64).ravel()
    val = np.asarray(lpz, dtype=np.float64).ravel()
    keep = np.isfinite(val)
    ind, val = ind[keep], val[keep]
    n = int(ind.size)
    if n < 3 or np.all(ind == ind[0]) or np.all(val == val[0]):
        return float("nan"), float("nan")
    rho = float(spearmanr(ind, val).correlation)
    if not math.isfinite(rho):
        return float("nan"), float("nan")
    r_ind = np.asarray(rankdata(ind), dtype=np.float64)
    r_val = np.asarray(rankdata(val), dtype=np.float64)
    z_ind = (r_ind - r_ind.mean()) / r_ind.std()
    z_val = (r_val - r_val.mean()) / r_val.std()
    rho_rank = float(np.mean(z_ind * z_val))
    perms = rng.permuted(np.tile(z_ind, (int(n_perm), 1)), axis=1)
    rho_perm = perms @ z_val / n
    p = (int(np.count_nonzero(rho_perm >= rho_rank - 1e-12)) + 1) / (int(n_perm) + 1)
    return rho, float(p)


def _nanmean_or_nan(values: NDArray[np.float64]) -> float:
    """Mean of the finite values; NaN with none."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _aggregate_warnings(pairs: Sequence[PairLeak]) -> List[str]:
    """The distinct per-pair warnings, each once with the number of
    pairs that raised it (fifty identical lines say less than one)."""
    counts: Dict[str, int] = {}
    for p in pairs:
        for w in p.warnings:
            counts[w] = counts.get(w, 0) + 1
    return [f"{n} pair(s): {w}" for w, n in counts.items()]


def ring_pair_leak(
    res: RingsResult,
    match: RingPairMatch,
    *,
    params: LeakParams,
    lpz_nm: Optional[NDArray[np.float64]] = None,
    min_samples: int,
    random_seed: int,
    min_shift_fraction: float = 0.5,
    clean_null: bool = True,
) -> RingPairLeak:
    """
    The ``RingPairLeak`` of one adjacent match: ``pair_leak`` on every
    matched pair, the counts and the clean excess, then the two
    pre-specified pair-level diagnostics.

    ``clean_null`` (D-27): when at least one pair is leak-explained,
    the eclipse test is rerun (``match_clean``) with the flagged
    children marked unusable in their rings' geometries -- the run's
    masks (``match.usable_a`` / ``usable_b``) minus the children, the
    suspect marks cleared so that the flag passed (the one the masks
    imply) removes nothing else -- at ``match.tau_nm`` with
    ``match.n_null``, ``match.random_seed`` and ``min_shift_fraction``
    (the run's ``ColumnsParams.min_shift_fraction``): same draws, so
    the only difference from the run is the removed clusters, on the
    observed and on the null side alike. With nothing flagged
    ``match_clean`` is ``match`` itself.

    (i) ``p_shared``, the reassignment null: B = ``params.n_null_shared``
    random one-to-one pairings of n_matched usable clusters of ring a
    with n_matched usable clusters of ring b (sampled without
    replacement on each side, paired in random order), the mean shared
    fraction over the pairing each time with the child rule (the
    cluster with fewer localizations is the child, tie: ring b's; its
    event count is the denominator; pairs without a child event are
    left out of the mean), p = (#{null mean >= observed mean} + 1) /
    (B + 1). The linking is local (radius 5 x lp ~ 50 nm), so random
    pairings almost never share an event and this p is small whenever
    ANY matched pair shares one; it is reported because it is
    pre-specified (D-10), the decision-relevant quantity is the
    per-pair level (``leak_explained``), and H5 calibrates.

    (ii) Spearman between the indicator "has a partner" and
    ``lpz_median_nm`` over the usable clusters of BOTH rings pooled
    (ring a's clusters in order, then ring b's), with a one-sided
    permutation p of ``params.n_null_shared`` permutations
    (``_spearman_with_permutation``). Rule (ii) reads this pair at the
    5 % level and conjoins the pairs (``LeakDiagnostics``); when rho is
    undefined (every usable cluster matched, none matched, a constant
    lpz, or fewer than 3 clusters with lpz) the pair is warned and
    does not vote.

    ``zeta_clean`` = (n_matched_clean - E* min K) / (sd* min K) with E*
    and sd* of ``match`` (the null count is not cleaned: conservative).
    The generators are ``SeedSequence(random_seed, spawn_key=(ring_a,
    ring_b, SHARED_SPAWN_KEY))`` and ``(..., LPZ_SPAWN_KEY)``.
    """
    ring_a = _ring_by_index(res, match.ring_a)
    ring_b = _ring_by_index(res, match.ring_b)
    i_arr = np.asarray(match.i_a, dtype=np.intp).ravel()
    j_arr = np.asarray(match.j_b, dtype=np.intp).ravel()
    pairs = [pair_leak(res, match, int(i), int(j), params=params, lpz_nm=lpz_nm, min_samples=min_samples)
             for i, j in zip(i_arr.tolist(), j_arr.tolist())]
    warnings_ = _aggregate_warnings(pairs)
    n_matched = int(match.n_matched)
    events = res.event_id is not None
    n_shared_high: Optional[int] = (sum(1 for p in pairs if p.shared_high) if events else None)
    n_z_tail = sum(1 for p in pairs if p.z_tail)
    n_size_ok = sum(1 for p in pairs if p.size_ok)
    n_expl = sum(1 for p in pairs if p.leak_explained)
    n_clean = n_matched - n_expl
    min_k = min(int(match.K_a), int(match.K_b))
    e_dir_clean = n_clean / min_k if min_k else float("nan")
    e_star, sd_star = float(match.E_star), float(match.sd_star)
    if min_k and math.isfinite(sd_star) and sd_star > 0.0 and math.isfinite(e_star):
        zeta_clean = (n_clean - e_star * min_k) / (sd_star * min_k)
    else:
        zeta_clean = float("nan")
        warnings_.append("zeta_clean undefined: the null of the match has no spread (or no usable cluster)")

    usable_a = _usable_mask(match.usable_a, len(ring_a.clusters))
    usable_b = _usable_mask(match.usable_b, len(ring_b.clusters))
    n_locs_a = np.array([int(cl.n_locs) for cl in ring_a.clusters], dtype=np.int64)
    n_locs_b = np.array([int(cl.n_locs) for cl in ring_b.clusters], dtype=np.int64)

    # --- the cleaned null: the eclipse test without the flagged children --
    match_clean: Optional[RingPairMatch] = None
    if clean_null:
        if n_expl == 0:
            match_clean = match
        else:
            child_a = np.zeros(len(ring_a.clusters), dtype=bool)
            child_b = np.zeros(len(ring_b.clusters), dtype=bool)
            for p in pairs:
                if p.leak_explained:
                    (child_a if p.child_ring == int(match.ring_a) else child_b)[p.child_index] = True
            try:
                ga = ring_geometry(ring_a, include_suspect=True)
                gb = ring_geometry(ring_b, include_suspect=True)
                keep_a, keep_b = usable_a & ~child_a, usable_b & ~child_b
                ga = dataclasses.replace(ga, usable=keep_a, suspect=None)
                gb = dataclasses.replace(gb, usable=keep_b, suspect=None)
                match_clean = eclipse_test(ga, gb, float(match.tau_nm), n_null=int(match.n_null),
                                           random_seed=int(match.random_seed),
                                           min_shift_fraction=float(min_shift_fraction),
                                           include_suspect=bool(np.all(keep_a) and np.all(keep_b)))
            except ValueError as exc:
                warnings_.append(f"cleaned null not computed: {exc}")
    zeta_clean_null = float(match_clean.zeta) if match_clean is not None else float("nan")
    p_excess_clean = float(match_clean.p_excess) if match_clean is not None else float("nan")

    # --- (i) shared fractions and the reassignment null -------------------
    paired_mean = unpaired_mean = p_shared = float("nan")
    if events:
        paired_mean = _nanmean_or_nan(np.array([p.shared_fraction for p in pairs], dtype=np.float64))
        shared, n_ev_a, n_ev_b = _event_matrix(res.event_id, ring_a.clusters, ring_b.clusters)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Child rule per cross pair: the cluster with fewer localizations (tie: b).
            child_is_b = n_locs_b[None, :] <= n_locs_a[:, None]
            denom_child = np.where(child_is_b, n_ev_b[None, :], n_ev_a[:, None]).astype(np.float64)
            frac_child = np.where(denom_child > 0, shared / np.where(denom_child > 0, denom_child, 1.0), np.nan)
            denom_min = np.minimum(n_ev_a[:, None], n_ev_b[None, :]).astype(np.float64)
            frac_min = np.where(denom_min > 0, shared / np.where(denom_min > 0, denom_min, 1.0), np.nan)
        cross = usable_a[:, None] & usable_b[None, :]
        matched = np.zeros_like(cross)
        matched[i_arr, j_arr] = True
        unpaired = frac_min[cross & ~matched]
        unpaired_mean = _nanmean_or_nan(unpaired) if unpaired.size else float("nan")
        ua, ub = np.flatnonzero(usable_a), np.flatnonzero(usable_b)
        if n_matched > 0 and math.isfinite(paired_mean) and ua.size >= n_matched and ub.size >= n_matched:
            rng = np.random.default_rng(np.random.SeedSequence(int(random_seed), spawn_key=(
                int(match.ring_a), int(match.ring_b), SHARED_SPAWN_KEY)))
            n_rep = int(params.n_null_shared)
            # One random one-to-one pairing per replicate: a fresh permutation of the usable clusters on
            # each side, its first n_matched entries paired in order (sampling without replacement on each
            # side, paired in random order), all replicates in one batch.
            sel_a = rng.permuted(np.tile(ua, (n_rep, 1)), axis=1)[:, :n_matched]
            sel_b = rng.permuted(np.tile(ub, (n_rep, 1)), axis=1)[:, :n_matched]
            with np.errstate(invalid="ignore"):
                null_means = np.nanmean(frac_child[sel_a, sel_b], axis=1)
            hits = int(np.count_nonzero(null_means[np.isfinite(null_means)] >= paired_mean))
            p_shared = (hits + 1) / (n_rep + 1)
        elif n_matched == 0:
            warnings_.append("no matched pair: shared fractions and p_shared NaN")
        else:
            warnings_.append("p_shared NaN: no matched pair with a child event, or fewer usable clusters than pairs")
    else:
        warnings_.append("no events (event_id None): shared_events, shared fractions, p_shared and n_shared_high "
                         "unavailable")

    # --- (ii) Spearman matched-vs-lpz over the pooled usable clusters -----
    ind: List[float] = []
    lpz_pool: List[float] = []
    for clusters, usable, matched_idx in ((ring_a.clusters, usable_a, set(i_arr.tolist())),
                                          (ring_b.clusters, usable_b, set(j_arr.tolist()))):
        for i, cl in enumerate(clusters):
            if usable[i]:
                ind.append(1.0 if i in matched_idx else 0.0)
                lpz_pool.append(float(cl.lpz_median_nm))
    lpz_arr = np.asarray(lpz_pool, dtype=np.float64)
    rng_perm = np.random.default_rng(np.random.SeedSequence(int(random_seed), spawn_key=(
        int(match.ring_a), int(match.ring_b), LPZ_SPAWN_KEY)))
    rho, p_rho = _spearman_with_permutation(np.asarray(ind, dtype=np.float64), lpz_arr,
                                            int(params.n_null_shared), rng_perm)
    if not math.isfinite(rho):
        # Name the cause: an undefined rho leaves this pair out of rule (ii) (it does not vote), and the
        # report must know whether that is a data gap (no lpz) or a full match (every usable cluster paired).
        ind_finite = np.asarray(ind, dtype=np.float64)[np.isfinite(lpz_arr)]
        if ind_finite.size < 3:
            cause = "fewer than 3 usable clusters with a finite lpz"
        elif np.all(ind_finite == ind_finite[0]):
            cause = ("the has-a-partner indicator is constant (every usable cluster matched)" if ind_finite[0] == 1.0
                     else "the has-a-partner indicator is constant (no usable cluster matched)")
        else:
            cause = "the clusters' median lpz is constant"
        warnings_.append(f"spearman_matched_vs_lpz undefined: {cause}; this pair does not vote in rule (ii)")
    finite_lpz = lpz_arr[np.isfinite(lpz_arr)]
    lpz_median = float(np.median(finite_lpz)) if finite_lpz.size else float("nan")

    ratios = np.array([p.size_ratio for p in pairs], dtype=np.float64)
    return RingPairLeak(
        ring_a=int(match.ring_a), ring_b=int(match.ring_b), tau_nm=float(match.tau_nm), n_matched=n_matched,
        K_a=int(match.K_a), K_b=int(match.K_b), pairs=pairs, n_shared_high=n_shared_high, n_z_tail=n_z_tail,
        n_size_ok=n_size_ok, n_leak_explained=n_expl, n_matched_clean=n_clean, E_dir_clean=float(e_dir_clean),
        zeta_clean=float(zeta_clean), shared_fraction_paired_mean=float(paired_mean),
        shared_fraction_unpaired_mean=float(unpaired_mean), p_shared=float(p_shared),
        spearman_matched_vs_lpz_rho=float(rho), spearman_matched_vs_lpz_p=float(p_rho),
        lpz_pair_median_nm=lpz_median, E_excess=float(match.E_dir) - float(match.E_star),
        size_ratio_median=float(np.median(ratios)) if ratios.size else float("nan"),
        fraction_size_ratio_below_half=float(np.mean(ratios < 0.5)) if ratios.size else float("nan"),
        match_clean=match_clean, zeta_clean_null=zeta_clean_null, p_excess_clean=p_excess_clean,
        warnings=warnings_, n_leak_explained_any=n_expl,
        fraction_leak_explained_any=(n_expl / n_matched if n_matched else float("nan")),
    )


# ============================================================================
# Axial profile: fixed-offset mixture, bootstrap, sample sizes
# ============================================================================

def axon_period_nm(rings: Sequence[Ring]) -> float:
    """The axon's period: the median over consecutive listed rings
    (sorted by index) of centre spacing / index step, the rule of
    ``tau_components`` (D-20); NaN with fewer than 2 rings."""
    ordered = sorted(rings, key=lambda r: int(r.index))
    if len(ordered) < 2:
        return float("nan")
    centres = np.array([float(r.centre_z_nm) for r in ordered])
    idx = np.array([float(r.index) for r in ordered])
    return float(np.median(np.diff(centres) / np.diff(idx)))


def _gaussian_mle_rows(z: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per row of ``z`` (R, N): the one-Gaussian MLE (mean, sd with
    ddof 0 floored at ``_MIN_SIGMA_NM``) and the log-likelihood AT the
    returned parameters, -N (log sigma + log sqrt(2 pi)) - sum (z -
    mu)^2 / (2 sigma^2), which equals -N (log sigma + log sqrt(2 pi) +
    1/2) whenever the floor is inactive and is larger by up to N / 2
    when it is: the mixture EM evaluates its likelihood at the same
    floored sigma, so the two must be on the same footing (a no-spread
    sample otherwise showed llr = N from nothing)."""
    n = z.shape[1]
    mu = z.mean(axis=1)
    var = z.var(axis=1)
    sigma = np.maximum(np.sqrt(var), _MIN_SIGMA_NM)
    ll = -n * (np.log(sigma) + _LOG_SQRT_2PI) - n * var / (2.0 * sigma * sigma)
    return mu, sigma, ll


def _em_step(
    za: NDArray[np.float64],
    theta: NDArray[np.float64],
    offsets: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    One EM step of the fixed-offset mixture on every row of ``za`` (m,
    N): ``theta`` is (m, 2 + n_comp) = [mu, sigma, w_0..w_{n-1}] per
    row. Returns the log-likelihood AT ``theta`` (the E-step's) and the
    M-step's new theta: weights = mean responsibility, mu = the
    responsibility-weighted mean of z - offset, sigma = the pooled
    residual sd about the new means (floored at ``_MIN_SIGMA_NM``).

    The work arrays are component-major, (n_comp, m, N), so that the
    reductions over the components (the log-sum-exp, the responsibility
    normaliser) run over the leading axis of contiguous (m, N) planes
    and the per-component sums are plain row reductions: 2-4x faster
    than the (m, N, n_comp) layout, whose reductions over a trailing
    axis of length 2 or 3 were strided (measured 8.0 -> 1.8 ms per step
    on 400 rows x 98 values, 122 -> 69 ms on 600 x 600 x 3; identical
    results to 1e-13).
    """
    n_rows, n = za.shape
    n_comp = int(offsets.size)
    mu, s, w = theta[:, 0], theta[:, 1], theta[:, 2:]
    inv_s = 1.0 / s
    const = np.log(np.maximum(w, _MASS_FLOOR)) - (np.log(s) + _LOG_SQRT_2PI)[:, None]     # (m, n_comp)
    lp = np.empty((n_comp, n_rows, n), dtype=np.float64)
    for c in range(n_comp):
        d = (za - (mu + offsets[c])[:, None]) * inv_s[:, None]
        np.multiply(d, d, out=d)
        lp[c] = const[:, c][:, None] - 0.5 * d
    mx = lp.max(axis=0)
    lp -= mx[None, :, :]
    np.exp(lp, out=lp)
    se = lp.sum(axis=0)
    ll = (mx + np.log(se)).sum(axis=1)
    lp /= se[None, :, :]                                              # responsibilities
    w_new = lp.mean(axis=2).T
    mu_new = np.zeros(n_rows, dtype=np.float64)
    for c in range(n_comp):
        mu_new += (lp[c] * (za - offsets[c])).sum(axis=1)
    mu_new /= n
    ss = np.zeros(n_rows, dtype=np.float64)
    for c in range(n_comp):
        d = za - (mu_new + offsets[c])[:, None]
        np.multiply(d, d, out=d)
        ss += (lp[c] * d).sum(axis=1)
    s_new = np.maximum(np.sqrt(ss / n), _MIN_SIGMA_NM)
    return ll, np.column_stack([mu_new, s_new, w_new])


def _em_step_rows(
    za: NDArray[np.float64],
    theta: NDArray[np.float64],
    offsets: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``_em_step`` on row blocks of about ``_EM_CHUNK_ELEMENTS``
    elements, concatenated: the same numbers as one call on the whole
    batch (rows are independent), at the cache-resident speed."""
    m, n = za.shape
    rows = max(1, _EM_CHUNK_ELEMENTS // max(1, n * int(offsets.size)))
    if m <= rows:
        return _em_step(za, theta, offsets)
    ll = np.empty(m, dtype=np.float64)
    out = np.empty_like(theta)
    for start in range(0, m, rows):
        stop = min(m, start + rows)
        ll[start:stop], out[start:stop] = _em_step(za[start:stop], theta[start:stop], offsets)
    return ll, out


def _project(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Back to the feasible set after an extrapolation: sigma >=
    ``_MIN_SIGMA_NM``, weights clipped to [1e-12, 1] and renormalised."""
    out = np.array(theta, dtype=np.float64, copy=True)
    out[:, 1] = np.maximum(out[:, 1], _MIN_SIGMA_NM)
    w = np.clip(out[:, 2:], 1e-12, 1.0)
    out[:, 2:] = w / w.sum(axis=1, keepdims=True)
    return np.asarray(out, dtype=np.float64)


def _em_fixed_offset_rows(
    z: NDArray[np.float64],
    period: float,
    n_comp: int,
    max_iter: int,
    tol: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Accelerated EM of the fixed-offset mixture on every row of ``z``
    (R, N) at once: components at mu + m period (m = 0..n_comp-1), one
    shared sigma, free weights. Per row 2 n_comp starts are run in the
    same batch -- mu = mean - k period / 2 with sigma = sd, and mu =
    mean - k period with sigma = sd / 2, k = 0..n_comp-1 -- and the best
    log-likelihood is kept; the one-Gaussian solution is NOT among them
    (the caller compares against its closed form).

    Why accelerated: plain EM crawls whenever a weight heads for 0 --
    the case of every bootstrap replicate drawn under the one-Gaussian
    null, where the log-likelihood surface is flat in the vanishing
    component and the increments decay sublinearly, so a 1e-9 relative
    tolerance costs hundreds of iterations per replicate (measured: 23
    s for 99 replicates of 500 events). Each cycle here is SQUAREM
    (Varadhan and Roland 2008, step length alpha = -|r| / |v| capped
    at -1): two EM steps theta0 -> theta1 -> theta2, the extrapolation
    theta' = theta0 - 2 alpha r + alpha^2 v with r = theta1 - theta0 and
    v = theta2 - 2 theta1 + theta0, projected back to the feasible set
    (``_project``), then one EM step from theta' whose E-step gives
    L(theta'); the extrapolation is kept only when L(theta') >= L(theta1)
    (else the cycle ends at theta2, so the tracked log-likelihood never
    decreases). A row-start stops when the tracked log-likelihood rises
    by less than ``tol`` x max(1, |loglik|) over a cycle or after
    ``max_iter`` EM steps, and leaves the batch, so the cost shrinks as
    rows converge; each step runs on cache-sized row blocks
    (``_em_step_rows``). The parameters returned are the ones whose
    log-likelihood was evaluated (an E-step's), never a half-updated
    set.

    Returns (mu (R,), sigma (R,), weights (R, n_comp), loglik (R,)).
    """
    n_rows = z.shape[0]
    mean = z.mean(axis=1)
    sd = np.maximum(z.std(axis=1), _MIN_SIGMA_NM)
    mu_starts = [mean - k * period * 0.5 for k in range(n_comp)] + [mean - k * period for k in range(n_comp)]
    s_starts = [sd for _ in range(n_comp)] + [np.maximum(sd / 2.0, _MIN_SIGMA_NM) for _ in range(n_comp)]
    n_starts = 2 * n_comp
    m_rows = n_rows * n_starts
    theta = np.empty((m_rows, 2 + n_comp), dtype=np.float64)
    theta[:, 0] = np.stack(mu_starts, axis=1).reshape(m_rows)
    theta[:, 1] = np.stack(s_starts, axis=1).reshape(m_rows)
    theta[:, 2:] = 1.0 / n_comp
    zz = np.repeat(z, n_starts, axis=0)
    offsets = period * np.arange(n_comp, dtype=np.float64)
    out_theta = theta.copy()
    out_ll = np.full(m_rows, -np.inf)
    ll_track = np.full(m_rows, -np.inf)
    active = np.arange(m_rows)
    steps = 0

    def record(rows: NDArray[np.intp], th: NDArray[np.float64], ll: NDArray[np.float64]) -> None:
        """Keep, per row, the evaluated parameters with the best log-likelihood."""
        better = ll > out_ll[rows]
        out_theta[rows[better]] = th[better]
        out_ll[rows[better]] = ll[better]

    while active.size and steps < int(max_iter):
        za = zz[active]
        th0 = theta[active]
        ll0, th1 = _em_step_rows(za, th0, offsets)
        record(active, th0, ll0)
        ll1, th2 = _em_step_rows(za, th1, offsets)
        record(active, th1, ll1)
        r = th1 - th0
        v = th2 - th1 - r
        norm_r = np.sqrt((r * r).sum(axis=1))
        norm_v = np.sqrt((v * v).sum(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.where(norm_v > 0.0, -norm_r / np.where(norm_v > 0.0, norm_v, 1.0), -1.0)
        alpha = np.minimum(alpha, -1.0)
        th_x = _project(th0 - 2.0 * alpha[:, None] * r + (alpha * alpha)[:, None] * v)
        ll_x, th_xx = _em_step_rows(za, th_x, offsets)
        accept = np.isfinite(ll_x) & (ll_x >= ll1)
        record(active[accept], th_x[accept], ll_x[accept])
        new_ll = np.where(accept, ll_x, ll1)
        theta[active] = np.where(accept[:, None], th_xx, th2)
        done = (new_ll - ll_track[active]) < tol * np.maximum(1.0, np.abs(new_ll))
        ll_track[active] = new_ll
        active = active[~done]
        steps += 3
    best = out_ll.reshape(n_rows, n_starts).argmax(axis=1) + n_starts * np.arange(n_rows)
    th = out_theta[best]
    return th[:, 0], th[:, 1], th[:, 2:], out_ll[best]


def fixed_offset_mixture_fit(
    z_nm: NDArray[np.float64],
    period_nm: float,
    n_components: int,
    *,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> Tuple[float, float, NDArray[np.float64], float]:
    """
    Maximum-likelihood fit of ``n_components`` Gaussians at mu + m x
    period (m = 0..n_components-1) with one shared sigma and free
    weights, by EM from several starts (``_em_fixed_offset_rows``) plus
    the one-Gaussian closed form with weights (1, 0, ...), which is a
    fixed point of the EM and guarantees loglik2 >= loglik1 whatever
    the data. With one component the closed-form MLE (mean, sd with
    ddof 0) is returned directly.

    Parameters
    ----------
    z_nm : (N,) the values (z' per event of a column).
    period_nm : the axon period (the fixed offset).
    n_components : the column length.
    max_iter, tol : EM stopping rule -- the log-likelihood increment
        below ``tol`` x max(1, |loglik|) (relative, so that a profile of
        thousands of events and one of twenty stop at the same
        precision of the fit), or ``max_iter`` iterations.

    Returns
    -------
    (mu_nm, sigma_nm, weights (n_components,), loglik)
        The best of the starts.
    """
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    n_comp = int(n_components)
    if n_comp < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components!r}")
    if z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("z_nm must be a non-empty finite array")
    if not (math.isfinite(float(period_nm)) and float(period_nm) > 0.0):
        raise ValueError(f"period_nm must be a positive finite number, got {period_nm!r}")
    mu1, s1, ll1 = _gaussian_mle_rows(z[None, :])
    one_hot = np.zeros(n_comp, dtype=np.float64)
    one_hot[0] = 1.0
    if n_comp == 1:
        return float(mu1[0]), float(s1[0]), one_hot, float(ll1[0])
    mu, sigma, w, ll = _em_fixed_offset_rows(z[None, :], float(period_nm), n_comp, int(max_iter), float(tol))
    if not (ll[0] > ll1[0]):
        return float(mu1[0]), float(s1[0]), one_hot, float(ll1[0])
    return float(mu[0]), float(sigma[0]), np.asarray(w[0], dtype=np.float64), float(ll[0])


def _fisher_information_pi(pi: float, mu: float, sigma: float, period: float, weights: NDArray[np.float64]) -> float:
    """
    I(pi) = integral (f2 - f1)^2 / (pi f2 + (1 - pi) f1) dz, the Fisher
    information for the mass pi outside the dominant component with mu,
    sigma and the period known: f1 is the dominant component N(mu + m*
    period, sigma), f2 the rest of the fitted mixture normalised by
    1 - w_max (for two components, the other Gaussian; then I(pi) =
    I(1 - pi) by mirror symmetry, so which component is called
    dominant does not matter). Trapezoid rule on ``_INFO_POINTS``
    points over the mixture's +-``_INFO_SIGMAS`` sigma range.
    """
    w = np.asarray(weights, dtype=np.float64)
    n_comp = int(w.size)
    m_star = int(w.argmax())
    rest = np.delete(w, m_star)
    if rest.sum() <= 0.0 or n_comp < 2:
        return float("nan")
    rest = rest / rest.sum()
    others = [m for m in range(n_comp) if m != m_star]
    grid = np.linspace(mu - _INFO_SIGMAS * sigma, mu + (n_comp - 1) * period + _INFO_SIGMAS * sigma, _INFO_POINTS)
    f1 = np.asarray(norm.pdf(grid, mu + m_star * period, sigma), dtype=np.float64)
    f2 = np.zeros_like(grid)
    for wt, m in zip(rest.tolist(), others):
        f2 += wt * np.asarray(norm.pdf(grid, mu + m * period, sigma), dtype=np.float64)
    mix = pi * f2 + (1.0 - pi) * f1
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(mix > 0.0, (f2 - f1) ** 2 / np.where(mix > 0.0, mix, 1.0), 0.0)
    return float(np.trapezoid(integrand, grid))


def _n_min_prereg(pi: float, period: float, sigma: float, z: float) -> float:
    """The pre-specified n_min: z^2 / (pi^2 (exp((P / sigma)^2) - 1));
    inf at pi = 0, 0 when the exponential overflows (sigma << P: the
    bound is then below one event anyway)."""
    if not (pi > 0.0):
        return float("inf")
    ratio2 = (period / sigma) ** 2
    if ratio2 > 700.0:
        return 0.0
    return float(z * z / (pi * pi * (math.exp(ratio2) - 1.0)))


def _n_min_wald(pi: float, info: float, power: float, z: float) -> float:
    """(z_{alpha/2} + z_power)^2 / (pi^2 I(pi)); inf at pi = 0 or I = 0."""
    if not (pi > 0.0) or not (math.isfinite(info) and info > 0.0):
        return float("inf")
    return float((z + float(norm.ppf(power))) ** 2 / (pi * pi * info))


def axial_profile_fit(
    z_nm: NDArray[np.float64],
    period_nm: float,
    n_components: int,
    *,
    n_bootstrap: int,
    random_seed: int,
    params: LeakParams,
) -> AxialProfileFit:
    """
    The axial profile test of one column on its values ``z_nm`` (one per
    event; 03 S3.2 ``axial_profile_test``): the one-Gaussian MLE, the
    ``n_components`` fixed-offset mixture (``fixed_offset_mixture_fit``),
    llr = 2 (loglik2 - loglik1) and its parametric bootstrap under the
    one-Gaussian fit -- B = ``n_bootstrap`` samples of the same size
    drawn from N(mu1, sigma1) with ``default_rng(random_seed)``, each
    fitted by the same two procedures (the B mixtures in one batch,
    EM stopping rule ``_BOOT_TOL`` / ``_BOOT_MAX_ITER``), p = (#{llr_b
    >= llr} + 1) / (B + 1). What that p tests: the null is ONE Gaussian,
    so any departure from Gaussianity that the period-offset mixture
    can absorb is rejected -- a heavy one-sided tail (a leak child's
    single-localization events beyond the boundary) as much as a second
    mode at the period; two real clusters one period apart are bimodal
    at the period whatever produced them (measured: chance pairs on
    independent rings rejected in 93-99 %; the length-2 parent+child
    columns of the harness regime keep p > 0.05 in 69 % pooled over
    three axons, 22 of 32, with a spread of 43 / 71 / 82 % per axon --
    while every
    length-3 column of a leak-only axon is "bimodal"). The p-value
    therefore says "not one Gaussian", not
    "column rather than leak", which is why rule (iv) stays None until
    H5 supplies the simulated-leak null. Then ``pi_hat`` and the two
    sample sizes (``AxialProfileFit``). ``members``, ``length``,
    ``n_locs``, ``n_events``, ``unit_is_events`` and ``meets_min_locs``
    are the caller's (``column_axial_profile``); here ``members`` is
    empty, ``n_locs`` = ``n_events`` = the number of values and the
    flags True.

    Fewer than ``_MIN_PROFILE_VALUES`` values give NaN fits and a
    warning: nothing can be tested. The random seed recorded is the
    integer given.
    """
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    n_comp = int(n_components)
    n_boot = int(n_bootstrap)
    if n_boot < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap!r}")
    warnings_: List[str] = []
    period = float(period_nm)
    z_alpha, power = float(params.n_min_z), float(params.n_min_power)
    nan_w = np.full(max(n_comp, 1), np.nan)
    if z.size < _MIN_PROFILE_VALUES or not np.all(np.isfinite(z)) or not (math.isfinite(period) and period > 0.0):
        warnings_.append(f"{z.size} value(s) (need {_MIN_PROFILE_VALUES}) or a non-finite value/period: "
                         "no axial profile fit (NaN)")
        return AxialProfileFit(
            members=[], length=n_comp, n_locs=int(z.size), n_events=int(z.size), z_events_nm=z, unit_is_events=True,
            period_nm=period, mu_nm=float("nan"), sigma_nm=float("nan"), weights=nan_w, mu1_nm=float("nan"),
            sigma1_nm=float("nan"), loglik1=float("nan"), loglik2=float("nan"), llr=float("nan"),
            p_bootstrap=float("nan"), pi_hat=float("nan"), n_min_events=float("nan"), n_min_80_events=float("nan"),
            meets_min_locs=True, n_bootstrap=n_boot, random_seed=int(random_seed), warnings=warnings_,
            n_min_z=z_alpha, n_min_power=power)
    mu1_r, s1_r, ll1_r = _gaussian_mle_rows(z[None, :])
    mu1, s1, ll1 = float(mu1_r[0]), float(s1_r[0]), float(ll1_r[0])
    if float(z.std()) < _MIN_SIGMA_NM:
        warnings_.append("the values have no spread: sigma floored at 1e-3 nm")
    mu, sigma, w, ll2 = fixed_offset_mixture_fit(z, period, n_comp)
    llr = max(0.0, 2.0 * (ll2 - ll1))
    rng = np.random.default_rng(int(random_seed))
    boot = rng.normal(mu1, s1, size=(n_boot, z.size))
    _, _, ll1_b = _gaussian_mle_rows(boot)
    if n_comp >= 2:
        _, _, _, ll2_b = _em_fixed_offset_rows(boot, period, n_comp, _BOOT_MAX_ITER, _BOOT_TOL)
        llr_b = np.maximum(0.0, 2.0 * (np.maximum(ll2_b, ll1_b) - ll1_b))
    else:
        llr_b = np.zeros(n_boot)
    p_boot = (int(np.count_nonzero(llr_b >= llr)) + 1) / (n_boot + 1)
    pi_hat = float(1.0 - float(np.max(w)))
    n_min = _n_min_prereg(pi_hat, period, float(sigma), z_alpha)
    info = _fisher_information_pi(pi_hat, float(mu), float(sigma), period, w) if pi_hat > 0.0 else float("nan")
    n_min_80 = _n_min_wald(pi_hat, info, power, z_alpha)
    return AxialProfileFit(
        members=[], length=n_comp, n_locs=int(z.size), n_events=int(z.size), z_events_nm=z, unit_is_events=True,
        period_nm=period, mu_nm=float(mu), sigma_nm=float(sigma), weights=np.asarray(w, dtype=np.float64),
        mu1_nm=mu1, sigma1_nm=s1, loglik1=ll1, loglik2=float(ll2), llr=float(llr), p_bootstrap=float(p_boot),
        pi_hat=pi_hat, n_min_events=float(n_min), n_min_80_events=float(n_min_80), meets_min_locs=True,
        n_bootstrap=n_boot, random_seed=int(random_seed), warnings=warnings_, n_min_z=z_alpha, n_min_power=power)


def _column_seed(random_seed: int, column: Column) -> int:
    """The bootstrap seed of one column: one 32-bit word of
    ``SeedSequence(random_seed, spawn_key=(PROFILE_SPAWN_KEY, start
    ring, cluster index))``, so that every column of an axon draws its
    own replicates and another axon seed changes all of them."""
    start_ring, start_cluster = int(column.members[0][0]), int(column.members[0][1])
    seq = np.random.SeedSequence(int(random_seed), spawn_key=(PROFILE_SPAWN_KEY, start_ring, start_cluster))
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def column_axial_profile(
    res: RingsResult,
    column: Column,
    period_nm: float,
    *,
    params: LeakParams,
    random_seed: int,
) -> AxialProfileFit:
    """
    ``axial_profile_fit`` of one column with ``column.length`` components
    on one value per EVENT: the mean z' over ALL localizations of the
    event in the axon (``res.event_id`` / ``res.z_p``; no window cut, so
    an event's localizations in the gap or in the neighbouring ring
    count, and an event shared by a parent and its child is ONE value).
    Without events (``res.event_id`` None, or no cluster localization
    with an id >= 0) one value per localization is used with a warning
    and ``unit_is_events`` False; a localization with id -1 among
    events enters the profile as a single-localization event of its
    own (its z' is the value, exactly as a genuine one-localization
    event's is), counted in ``n_events`` and named in a warning: the
    specification's rule that -1 ids give "one value per localization"
    is honoured per localization without changing the unit of the rest
    of the column (D-27; a first version left them out, which shrank
    the sample silently on files with many unlinked localizations).
    The seed is ``_column_seed``; ``meets_min_locs`` says whether every
    cluster has ``params.profile_min_locs_per_cluster`` localizations
    (rule (iv)).
    """
    warnings_: List[str] = []
    clusters = [_ring_by_index(res, k).clusters[int(i)] for k, i in column.members]
    n_locs = int(sum(int(cl.n_locs) for cl in clusters))
    loc = np.concatenate([np.asarray(cl.loc_index, dtype=np.intp) for cl in clusters])
    unit_events = False
    n_events: Optional[int] = None
    if res.event_id is None:
        values = np.asarray(res.z_p, dtype=np.float64)[loc]
        warnings_.append("no events (event_id None): the axial profile is one value per localization")
    else:
        ev_all = np.asarray(res.event_id, dtype=np.int64)
        ids = ev_all[loc]
        valid = ids >= 0
        if not np.any(valid):
            values = np.asarray(res.z_p, dtype=np.float64)[loc]
            warnings_.append("no localization of the column has an event id >= 0: one value per localization")
        else:
            uniq = np.unique(ids[valid])
            mask = np.isin(ev_all, uniq)
            _, inv = np.unique(ev_all[mask], return_inverse=True)
            z_all = np.asarray(res.z_p, dtype=np.float64)
            values = np.bincount(inv, weights=z_all[mask]) / np.bincount(inv)
            n_single = int(np.count_nonzero(~valid))
            if n_single:
                values = np.concatenate([values, z_all[loc[~valid]]])
                warnings_.append(f"{n_single} localization(s) of the column with event id -1 entered the profile "
                                 "as single-localization events")
            n_events = int(uniq.size) + n_single
            unit_events = True
    seed = _column_seed(random_seed, column)
    fit = axial_profile_fit(values, float(period_nm), int(column.length), n_bootstrap=int(params.profile_n_bootstrap),
                            random_seed=seed, params=params)
    meets = all(int(cl.n_locs) >= int(params.profile_min_locs_per_cluster) for cl in clusters)
    return dataclasses.replace(
        fit, members=[(int(k), int(i)) for k, i in column.members], length=int(column.length), n_locs=n_locs,
        n_events=n_events, unit_is_events=unit_events, meets_min_locs=bool(meets),
        warnings=warnings_ + list(fit.warnings))


# ============================================================================
# Guard sensitivity
# ============================================================================

def guard_sensitivity(
    res: RingsResult,
    cols: AxonColumnsResult,
    *,
    params: LeakParams,
    frame: Optional[NDArray[np.int64]] = None,
    lp_lateral_nm: Optional[NDArray[np.float64]] = None,
    lpz_nm: Optional[NDArray[np.float64]] = None,
    n_frames: Optional[int] = None,
    roi: Any = None,
    pixel_size_nm: Optional[float] = None,
    pixel_size_source: str = "unknown",
) -> Dict[float, GuardRun]:
    """
    The rings rebuilt at every guard of ``params.guards_nm`` and the
    adjacent eclipse tests on each rebuild (rule (iii), 02 P3).

    Each rebuild is ``build_rings(res.x_p, res.y_p, res.z_p, ...)`` with
    ``rings_params_from(cols.params)`` at ``guard_nm = g`` and
    ``correct_tilt = False``: the identity frame on the already rotated
    coordinates reproduces the run's rings at g = 0 exactly
    (``to_axon_frame`` with the identity returns copies; same z', same
    seed), so the differences between guards are the guard's alone and
    not a re-fitted axis. Then ``eclipse_test`` on every adjacent pair
    of rebuilt rings (by ``Ring.index``, geometries with
    ``cols.include_suspect``) at ``cols.tau0_nm`` with ``cols.n_null``,
    ``cols.random_seed`` and ``cols.params.min_shift_fraction`` -- the
    settings of ``analyze_columns``, so at g = 0 the matches equal
    ``cols.adjacent`` bit for bit WHEN the run itself used the YAML
    parameters at guard 0. When ``res.params`` differs from
    ``rings_params_from(cols.params)`` in any field other than
    ``correct_tilt`` and ``n_bootstrap`` (e.g. a run at ``guard_nm`` 40,
    or another eps),
    the g = 0 row is another segmentation than the run's and rule
    (iii) is read on it: every ``GuardRun`` then carries a warning
    naming the fields that differ (measured: a base run at guard 40
    gave K [36, 39, 28] against [39, 40, 29] at g = 0). Without
    ``frame`` the events and the suspect marks of the rebuilt rings are
    missing (warning; the eclipse test does not need them). A rebuilt
    ring without a contour (fewer than 3 clusters) is left out of the
    matching with a warning.
    """
    base = rings_params_from(cols.params)
    n = int(np.asarray(res.x_p).size)
    if lpz_nm is not None and np.asarray(lpz_nm).size != n:
        raise ValueError(f"lpz_nm has {np.asarray(lpz_nm).size} values for {n} localizations")
    run_params = getattr(res, "params", None)
    differing: List[str] = []
    if run_params is not None:
        for f in dataclasses.fields(base):
            # correct_tilt differs by design (identity frame here); n_bootstrap
            # only sets the H2 centroid bootstrap, not the segmentation.
            if f.name in ("correct_tilt", "n_bootstrap"):
                continue
            if getattr(run_params, f.name, None) != getattr(base, f.name):
                differing.append(f"{f.name} ({getattr(run_params, f.name, None)!r} in the run, "
                                 f"{getattr(base, f.name)!r} here)")
    out: Dict[float, GuardRun] = {}
    for g in params.guards_nm:
        t0 = time.perf_counter()
        warnings_: List[str] = []
        if differing:
            warnings_.append("the run's RingsParams differ from the rebuild's rings_params_from(cols.params): "
                             + "; ".join(differing) + " -- the g = 0 row is not the run's segmentation")
        if frame is None:
            warnings_.append("no frame numbers: the rebuilt rings carry no events and no suspect marks")
        rp = dataclasses.replace(base, guard_nm=float(g), correct_tilt=False)
        rebuilt = build_rings(
            np.array(res.x_p, dtype=np.float64), np.array(res.y_p, dtype=np.float64),
            np.array(res.z_p, dtype=np.float64),
            frame=None if frame is None else np.array(frame),
            lp_lateral_nm=None if lp_lateral_nm is None else np.array(lp_lateral_nm, dtype=np.float64),
            lpz_nm=None if lpz_nm is None else np.array(lpz_nm, dtype=np.float64),
            params=rp, source_name=res.source_name, pixel_size_nm=pixel_size_nm,
            pixel_size_source=pixel_size_source, n_frames=n_frames, roi=roi)
        warnings_.extend(f"build_rings: {w}" for w in rebuilt.warnings)
        rings = sorted(rebuilt.rings, key=lambda r: int(r.index))
        geoms: List[RingGeometry] = []
        for ring in rings:
            try:
                geoms.append(ring_geometry(ring, include_suspect=bool(cols.include_suspect)))
            except ValueError as exc:
                warnings_.append(f"ring {ring.index} left out of the matching: {exc}")
        by_index = {gm.index: gm for gm in geoms}
        matches: List[RingPairMatch] = []
        for gm in geoms:
            nxt = by_index.get(gm.index + 1)
            if nxt is None:
                continue
            m = eclipse_test(gm, nxt, float(cols.tau0_nm), n_null=int(cols.n_null), random_seed=int(cols.random_seed),
                             min_shift_fraction=float(cols.params.min_shift_fraction),
                             include_suspect=bool(cols.include_suspect))
            warnings_.extend(f"adjacent {gm.index}-{nxt.index}: {w}" for w in m.warnings)
            matches.append(m)
        out[float(g)] = GuardRun(
            guard_nm=float(g), n_rings=len(rings), ring_indices=[int(r.index) for r in rings],
            K_per_ring=[len(r.clusters) for r in rings], n_locs_per_ring=[int(r.n_locs) for r in rings],
            matches=matches, seconds=time.perf_counter() - t0, warnings=warnings_)
    return out


# ============================================================================
# Drivers
# ============================================================================

def _all_or_none(values: Sequence[float], predicate: Any) -> Optional[bool]:
    """``all(predicate(v))`` over the finite values; None with none finite."""
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    return bool(all(predicate(v) for v in finite))


def _empty_diagnostics(cols: AxonColumnsResult, params: LeakParams, period: float, lpz: bool, events: bool,
                       t0: float, warnings_: List[str]) -> LeakDiagnostics:
    """The result of an axon with nothing to diagnose."""
    return LeakDiagnostics(
        source_name=str(cols.source_name), tau0_nm=float(cols.tau0_nm), period_nm=period, pairs=[], k2=cols.k2,
        guard={}, profiles=[], rules={"i_shared": None, "ii_lpz": None, "iii_guard": None, "iv_profile": None,
                                       "v_size_ratio": None},
        fraction_profiles_bimodal=float("nan"), n_leak_explained=0, n_matched=0,
        fraction_leak_explained=float("nan"), params=params, lpz_per_localization=lpz, events_available=events,
        seconds=time.perf_counter() - t0, warnings=warnings_)


def analyze_leak(
    res: RingsResult,
    cols: AxonColumnsResult,
    *,
    params: LeakParams = LeakParams(),
    lpz_nm: Optional[NDArray[np.float64]] = None,
    frame: Optional[NDArray[np.int64]] = None,
    lp_lateral_nm: Optional[NDArray[np.float64]] = None,
    n_frames: Optional[int] = None,
    roi: Any = None,
    pixel_size_nm: Optional[float] = None,
    pixel_size_source: str = "unknown",
    guard: bool = True,
) -> LeakDiagnostics:
    """
    The H4-A diagnostics of one axon from its rings and its H3 columns
    result: ``ring_pair_leak`` on every adjacent match (seed
    ``cols.random_seed``, ``min_samples`` from ``cols.params``), the
    guard sensitivity when ``guard`` is on (``frame``, ``lp_lateral_nm``,
    ``lpz_nm``, ``n_frames``, ``roi`` and the pixel size are the
    rebuild's inputs), ``column_axial_profile`` of every column of
    length >= 2 in ``cols.columns`` order at the axon period
    (``axon_period_nm`` of ``res.rings``), and the rules dict
    (``LeakDiagnostics``). ``lpz_nm`` is the per-localization axial
    precision over EVERY input localization of ``res`` (input order);
    without it every lpz is a cluster median (Jensen lower bound,
    warned). With ``cols.adjacent`` empty the result is empty, with a
    warning. The inputs are not modified.
    """
    t0 = time.perf_counter()
    warnings_: List[str] = []
    n = int(np.asarray(res.x_p).size)
    if lpz_nm is not None and np.asarray(lpz_nm).size != n:
        raise ValueError(f"lpz_nm has {np.asarray(lpz_nm).size} values for {n} localizations")
    period = axon_period_nm(res.rings)
    lpz_given = lpz_nm is not None
    events = res.event_id is not None
    if not events:
        warnings_.append("no events (res.event_id None; build_rings without frame numbers): shared-event "
                         "quantities are NaN/None, rule (i) is None and the profiles are per localization")
    if not lpz_given:
        warnings_.append("no per-localization lpz: leak fractions and tail widths use cluster medians "
                         "(Jensen lower bound)")
    if not cols.adjacent:
        warnings_.append("no adjacent match in the columns result: nothing to diagnose")
        return _empty_diagnostics(cols, params, period, lpz_given, events, t0, warnings_)
    min_samples = int(cols.params.min_samples)
    pairs = [ring_pair_leak(res, m, params=params, lpz_nm=lpz_nm, min_samples=min_samples,
                            random_seed=int(cols.random_seed),
                            min_shift_fraction=float(cols.params.min_shift_fraction)) for m in cols.adjacent]
    # H5: the two-parent tail model needs both adjacent pairs of a child, so it runs here, after every pair.
    pairs = _apply_two_parent_model(res, pairs, params=params, lpz_nm=lpz_nm)
    for rp in pairs:
        warnings_.extend(f"rings {rp.ring_a}-{rp.ring_b}: {w}" for w in rp.warnings)

    guard_runs: Dict[float, GuardRun] = {}
    if guard:
        guard_runs = guard_sensitivity(res, cols, params=params, frame=frame, lp_lateral_nm=lp_lateral_nm,
                                       lpz_nm=lpz_nm, n_frames=n_frames, roi=roi, pixel_size_nm=pixel_size_nm,
                                       pixel_size_source=pixel_size_source)
        for g, run in guard_runs.items():
            warnings_.extend(f"guard {g:g} nm: {w}" for w in run.warnings)
    else:
        warnings_.append("guard sensitivity not run (guard=False): rule (iii) is None")

    profiles: List[AxialProfileFit] = []
    if not math.isfinite(period):
        warnings_.append("axon period undefined (fewer than 2 rings): no axial profile")
    else:
        for col in cols.columns:
            if int(col.length) >= 2:
                profiles.append(column_axial_profile(res, col, period, params=params,
                                                     random_seed=int(cols.random_seed)))
    for prof in profiles:
        warnings_.extend(f"column {prof.members}: {w}" for w in prof.warnings)

    # --- rules -------------------------------------------------------------
    rules: Dict[str, Optional[bool]] = {}
    rules["i_shared"] = _all_or_none([rp.p_shared for rp in pairs], lambda p: p > 0.05) if events else None
    rho_ok: List[float] = []
    for rp in pairs:
        if math.isfinite(rp.spearman_matched_vs_lpz_rho):
            rho_ok.append(1.0 if (rp.spearman_matched_vs_lpz_rho <= 0.0 or rp.spearman_matched_vs_lpz_p > 0.05) else 0.0)
    rules["ii_lpz"] = bool(all(v > 0.5 for v in rho_ok)) if rho_ok else None
    run_rule = guard_runs.get(float(params.guard_rule_nm))
    if run_rule is None:
        rules["iii_guard"] = None
        if guard_runs:
            warnings_.append(f"guard_rule_nm {params.guard_rule_nm:g} nm is not among the guards run "
                             f"{list(guard_runs)}: rule (iii) is None")
    else:
        rules["iii_guard"] = _all_or_none([m.zeta for m in run_rule.matches], lambda z: z > 0.0)
    rules["iv_profile"] = None
    ratios = [p.size_ratio for rp in pairs for p in rp.pairs]
    rules["v_size_ratio"] = bool(float(np.median(ratios)) >= 0.5) if ratios else None

    eligible = [p.p_bootstrap for p in profiles if p.meets_min_locs and math.isfinite(p.p_bootstrap)]
    frac_bimodal = (float(np.mean([p <= float(params.profile_alpha) for p in eligible])) if eligible
                    else float("nan"))
    n_matched = int(sum(int(m.n_matched) for m in cols.adjacent))
    n_expl = int(sum(rp.n_leak_explained for rp in pairs))
    n_any = int(sum(rp.n_leak_explained_any for rp in pairs))
    # A pair whose F and llr are both NaN (no lpz and no finite parent ring width) could not be assessed;
    # when NO pair could, the fraction is NaN rather than 0.0, which would read as "no leak".
    n_assessable = sum(1 for rp in pairs for p in rp.pairs
                       if math.isfinite(p.p_size_leak) or math.isfinite(p.llr_tail_child))
    if n_matched and n_assessable == 0:
        frac_expl = float("nan")
        frac_any = float("nan")
        warnings_.append("no matched pair could be assessed (F_parent and llr_tail_child NaN everywhere): "
                         "fraction_leak_explained NaN")
    else:
        frac_expl = n_expl / n_matched if n_matched else float("nan")
        frac_any = n_any / n_matched if n_matched else float("nan")
    return LeakDiagnostics(
        source_name=str(cols.source_name), tau0_nm=float(cols.tau0_nm), period_nm=float(period), pairs=pairs,
        k2=cols.k2, guard=guard_runs, profiles=profiles, rules=rules, fraction_profiles_bimodal=frac_bimodal,
        n_leak_explained=n_expl, n_matched=n_matched,
        fraction_leak_explained=frac_expl, params=params,
        lpz_per_localization=lpz_given, events_available=events, seconds=time.perf_counter() - t0,
        warnings=warnings_, n_leak_explained_any=n_any, fraction_leak_explained_any=frac_any)


def analyze_columns_and_leak(
    res: RingsResult,
    columns_params: ColumnsParams,
    *,
    leak_params: LeakParams = LeakParams(),
    include_suspect: bool = True,
    n_null: Optional[int] = None,
    source_name: str = "",
    lpz_nm: Optional[NDArray[np.float64]] = None,
    frame: Optional[NDArray[np.int64]] = None,
    lp_lateral_nm: Optional[NDArray[np.float64]] = None,
    n_frames: Optional[int] = None,
    roi: Any = None,
    pixel_size_nm: Optional[float] = None,
    pixel_size_source: str = "unknown",
    guard: bool = True,
) -> AxonColumnsResult:
    """
    ``analyze_columns`` (unchanged, H3) followed by ``analyze_leak`` on
    its result, stored in the returned result's ``leak`` slot. The
    columns result is the object this call created; ``res`` is not
    modified.
    """
    cols = analyze_columns(res, columns_params, include_suspect=include_suspect, n_null=n_null,
                           source_name=source_name)
    cols.leak = analyze_leak(res, cols, params=leak_params, lpz_nm=lpz_nm, frame=frame, lp_lateral_nm=lp_lateral_nm,
                             n_frames=n_frames, roi=roi, pixel_size_nm=pixel_size_nm,
                             pixel_size_source=pixel_size_source, guard=guard)
    return cols
