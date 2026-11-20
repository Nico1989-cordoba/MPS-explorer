# -*- coding: utf-8 -*-
"""
Regularity of the cluster arrangement around the perimeter: the gap measures and the two-stage reading
(regularity plan PR 2, branch ``feat/gap-measures``; public record: ``docs/regularity.md``).

The question
------------
"Is this axon's arrangement of detected clusters more regular than random with contact, and if so, is there a
periodic component beyond an exclusion of the same strength?" One index cannot answer it: exclusion (clusters never
closer than d, otherwise random), a lattice (clusters near the sites of a grid that repeats around the whole ring) and
the detection itself (a clustering method merges clusters whose clouds touch) all raise the regularity index r.

What it measures (all on ARC positions of the centroids along the 600 nm P-spline membrane through them, the same
curve the pipeline null places its molecules on; never on polar angles)
-----------------------------------------------------------------------------------------------------------------
* gap CV         sd / mean of the K cyclic gaps g_i (population sd). Random: CV^2 = (K - 1)/(K + 1)
                 [derivation: Moran 1947]; "how much regularity", not which kind.
* rho1           circular lag-1 correlation of consecutive gaps. A lattice with independent jitter has
                 g_i = a + e_(i+1) - e_i, so rho1 = -1/2 [derivation]; any renewal process (exclusion, iid gaps) 0.
* excess-gap CV  sd / mean of g - min g: ~1 for "minimum + exponential tail" (exclusion), 0.4-0.6 for bell-shaped
                 gaps [simulation: regularity research sim1, seed 1]. Descriptive pointer to the kind.
* S(m), S*       the periodogram |sum_j exp(i 2 pi m s_j / P)|^2 / K (Bartlett 1963); S* its largest value for m in
                 [K/2, 3K/2], m* that mode, a* = P / m* the spacing it implies. A lattice of n sites with jitter
                 sigma has E S(n) = K e^-q + 1 - e^-q, q = (2 pi sigma / a)^2 [derivation, audited]. A dense
                 hard-rod fluid ALSO has a peak near m ~ K (Salsburg, Zwanzig & Kirkwood 1953): S* is read only
                 against an exclusion of the same strength (stage 2).
* S_low          mean S(m) over m = 1..3: the renewal plateau ~CV^2, 0 for a lattice (descriptive).
* g(s)           pair correlation along the arc, edge-free on a closed curve (drawn, not tested).
* elongation     per cluster, sd of its localizations along the membrane tangent / sd along the normal; merged
                 pairs are elongated (median 3.2-3.7 against 1.3 for single clusters [simulation: regularity
                 research sim2, seed 2]). Flagged above the 95th percentile of the null's (P-R84 a: mark and count,
                 never split).

The two-stage reading (regularity plan sec. 5; P-R81 b, decided 2026-10-22)
--------------------------------------------------------------------------
Stage 1, against the PIPELINE null (``tools.mps_pipeline_null``: molecules at random on this membrane with a contact
hard core d_c = 2 x median r_eff, the same clustering method, curation and contour): the gap CV, TWO-SIDED (P-R91 a).
Stage 2, only when stage 1 says "more regular": against the MATCHED-EXCLUSION null, sequential inhibition whose
hard core d_m is chosen by bisection so that the null's mean gap CV equals the observed one; rho1 (lower tail) and
S* (upper tail) are tested. Where stage 2 runs (P-R87 c): at the pipeline level (exact; the batch) or at the
centres level on the detected centroids (seconds; "approx., centres"; the GUI preview); both are available.

    0. K < 8                                   -> not_testable
       min 1NN < d_c                           -> flag "minimum below contact" shown before any verdict
    1. CV p <= alpha and CV < null mean        -> stage 2
       CV p <= alpha and CV > null mean        -> less_regular_than_random
       otherwise                               -> random_with_contact
    2. rho1 rejects (p <= alpha AND rho1 <= -0.25) AND S* rejects (p <= alpha)   -> lattice
       only rho1 rejects                       -> lattice_rho1_only
       only S* rejects                         -> periodic_component
       neither                                 -> more_regular_kind_not_identified

No "paracrystal" label (never validated through the pipeline; P-R92: the M8 iid-gamma model exists in the
generator only) and no "vacancies" in any label (a lattice with empty sites, undetected clusters and the residual
structure of a strong exclusion are indistinguishable at K ~ 50). alpha = 0.05 per test without Bonferroni per axon:
the rule is hierarchical and pre-written (stage 1 is the gate); the batch applies Holm across measures per animal
(``tools.mps_gap_batch``).

Method-agnostic: everything works from the kept centroids (and, for the elongation, the kept clusters'
localizations) of whatever method clustered the axon; the pipeline null re-runs that same method.

Origins of every default (labels: [lit], [derivation], [simulation], [user decision], [ad hoc]):
  alpha 0.05 [ad hoc: conventional; regularity plan P-R81].
  rho1 lattice threshold -0.25 [regularity audit, must-fix: written threshold, half of the lattice's -1/2].
  Bragg window m in [K/2, 3K/2] [regularity plan, glossary].  S_low over m = 1..3 [ad hoc, research sim1].
  K < 8 not testable [regularity plan sec. 5 rule 1].
  B = 199 for both nulls [user decision P-R78, 2026-10-22].
  matched-null bisection: 8 steps x 40 patterns at the centres level [regularity plan sec. 5], 6 steps x 16
    replicates at the pipeline level [ad hoc: cost; the bisection uses common random numbers so the mean CV is a
    smooth function of d]. d_m capped at 0.65 x P / n [regularity plan sec. 5: the null jams near packing].
  g(s): bins of 25 nm up to 2 um [regularity plan sec. 6.3].
  elongation flag: 95th percentile of the null's [regularity plan sec. 6.1; P-R84 a]; clusters with >= 10
    localizations [ad hoc: as tools.mps_pipeline_null's tangential sd].

No Qt. Deterministic for a fixed seed.

References (APA 7)
------------------
Bartlett, M. S. (1963). The spectral analysis of point processes. Journal of the Royal Statistical Society B, 25(2),
    264-296. https://doi.org/10.1111/j.2517-6161.1963.tb00508.x
Besag, J., & Diggle, P. J. (1977). Simple Monte Carlo tests for spatial pattern. Applied Statistics, 26(3), 327-333.
    https://doi.org/10.2307/2346974
Diggle, P. J. (1979). On parameter estimation and goodness-of-fit testing for spatial point patterns. Biometrics,
    35(1), 87-101. https://doi.org/10.2307/2529938
Diggle, P. J., Besag, J., & Gleaves, J. T. (1976). Statistical analysis of spatial point patterns by means of
    distance methods. Biometrics, 32(3), 659-667. https://doi.org/10.2307/2529754
Moran, P. A. P. (1947). The random division of an interval. Journal of the Royal Statistical Society B, 9(1), 92-98.
    https://doi.org/10.2307/2983572
Salsburg, Z. W., Zwanzig, R. W., & Kirkwood, J. G. (1953). Molecular distribution functions in a one-dimensional
    fluid. Journal of Chemical Physics, 21(6), 1098-1107. https://doi.org/10.1063/1.1699116

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from tools import mps_pipeline_null as pn
from tools.mps_randomization import monte_carlo_p
from tools.spatial_stats import arc_coordinates

FloatArray = NDArray[np.float64]

# ============================================================================
# Defaults (origin in brackets; the module docstring gives the argument)
# ============================================================================

ALPHA = 0.05                          # [ad hoc: conventional; P-R81] per test, hierarchical gate
RHO1_LATTICE_MAX = -0.25              # [regularity audit, must-fix] rho1 must also be <= this to count as rejecting
BRAGG_WINDOW = 0.5                    # [regularity plan glossary] S* over m in [K (1 - w), K (1 + w)]
S_LOW_MODES = 3                       # [ad hoc, research sim1] S_low = mean S(m), m = 1..3
MIN_CLUSTERS = pn.MIN_CLUSTERS        # [regularity plan sec. 5, rule 1] K < 8: not testable
DEFAULT_N_REPLICATES = pn.DEFAULT_N_REPLICATES   # [user decision P-R78] 199
CENTRES_BISECTION_STEPS = 8           # [regularity plan sec. 5] ~8 steps
CENTRES_BISECTION_N = 40              # [regularity plan sec. 5] of 40 simulations each
PIPELINE_BISECTION_STEPS = 6          # [ad hoc: cost] pipeline-level bisection
PIPELINE_BISECTION_N = 16             # [ad hoc: cost] replicates per step (common random numbers across steps)
MATCHED_CAP_FRACTION = 0.65           # [regularity plan sec. 5] d_m <= 0.65 x P / n (the null jams near packing)
# Failed replicates a pipeline null may lose and still be read [user decision G1 (c), 2026-10-23; the rule lives in
# tools.mps_pipeline_null (TOLERATED_FAILED_FRACTION, floor(0.05 B)) and applies to stage 1 AND stage 2 alike: the
# failures are counted as at least as extreme as the observed in every p (conservative), never dropped. Evidence:
# [simulation: 22 of 60 synthetic axons lost 1 of 49 replicates to a sampler jammed near packing]].
MATCHED_TOLERATED_FAILED_FRACTION = pn.TOLERATED_FAILED_FRACTION
FINAL_BACKOFF_FACTOR = 0.95           # [ad hoc] the final pipeline run lowers d_m by 5 % when it cannot be packed
FINAL_BACKOFF_TRIES = 5               # [ad hoc] ... at most this many times (then stage 2 is reported as not run)
G_BIN_NM = 25.0                       # [regularity plan sec. 6.3]
G_MAX_NM = 2000.0                     # [regularity plan sec. 6.3]
ELONGATION_MIN_LOCS = 10              # [ad hoc: as tools.mps_pipeline_null._median_tangential_sd]
ELONGATION_QUANTILE = 0.95            # [regularity plan sec. 6.1; P-R84 a]
TANGENT_STEP_NM = 5.0                 # [ad hoc] half-step of the finite-difference tangent on the membrane

# The reading levels (fixed list; the multimethod ``verdict`` readout and the export column ``regularity_reading``).
NOT_TESTABLE = "not_testable"
RANDOM_WITH_CONTACT = "random_with_contact"
LESS_REGULAR = "less_regular_than_random"
KIND_NOT_IDENTIFIED = "more_regular_kind_not_identified"
PERIODIC_COMPONENT = "periodic_component"
LATTICE_RHO1_ONLY = "lattice_rho1_only"
LATTICE = "lattice"
READING_LEVELS: Tuple[str, ...] = (RANDOM_WITH_CONTACT, LESS_REGULAR, KIND_NOT_IDENTIFIED, PERIODIC_COMPONENT,
                                   LATTICE_RHO1_ONLY, LATTICE, NOT_TESTABLE)
STAGE2_KINDS: Tuple[str, ...] = ("pipeline", "centres_approx", "not_run")
STAGE2_MODES: Tuple[str, ...] = ("centres", "pipeline", "both", "none")
MIN_FLAG_LEVELS: Tuple[str, ...] = ("min_below_contact", "min_at_or_above_contact")

APPROX_NOTE = ("p approximate: the null reads K, size, precision and contour off this same axon; the clustering "
               "method's merging is in the null, the biology of the labelling (epitope density) is not.")


# ============================================================================
# Measures on arc positions (pure)
# ============================================================================

def arc_gaps(s_nm: ArrayLike, length_nm: float) -> FloatArray:
    """The K cyclic gaps between consecutive arc positions on a closed curve of length P (they sum to P).
    Independent of the origin and of the direction of travel (the multiset of gaps; their cyclic ORDER reverses)."""
    a = np.sort(np.mod(np.asarray(s_nm, dtype=np.float64).reshape(-1), float(length_nm)))
    if a.size == 0:
        return np.zeros(0)
    return np.diff(np.concatenate([a, [a[0] + float(length_nm)]]))


def gap_cv(g: FloatArray) -> Optional[float]:
    """sd / mean of the gaps (population sd). K uniform points: CV^2 = (K - 1)/(K + 1) [derivation: Moran 1947]."""
    g = np.asarray(g, dtype=np.float64)
    if g.size < 2 or not float(np.mean(g)) > 0:
        return None
    return float(np.std(g) / np.mean(g))


def gap_rho1(g: FloatArray) -> Optional[float]:
    """Circular lag-1 correlation of consecutive gaps: -1/2 for a jittered lattice, 0 for a renewal process
    [derivation]. None when the gaps do not vary (a perfect lattice)."""
    g = np.asarray(g, dtype=np.float64)
    if g.size < 3:
        return None
    x = g - np.mean(g)
    den = float(np.sum(x * x))
    if not den > 1e-12 * max(1.0, float(np.mean(g)) ** 2 * g.size):
        return None
    return float(np.sum(x * np.roll(x, -1)) / den)


def excess_gap_cv(g: FloatArray) -> Optional[float]:
    """sd / mean of (g - min g): ~1 for a shifted exponential (exclusion), < 1 for bell-shaped gaps."""
    g = np.asarray(g, dtype=np.float64)
    if g.size < 3:
        return None
    e = g - float(np.min(g))
    m = float(np.mean(e))
    return float(np.std(e) / m) if m > 0 else None


def structure_factor(s_nm: ArrayLike, length_nm: float, modes: ArrayLike) -> FloatArray:
    """Bartlett's periodogram S(m) = |sum_j exp(i 2 pi m s_j / P)|^2 / K at the integer modes ``modes``."""
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    m = np.asarray(modes, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return np.full(m.size, np.nan)
    ph = 2.0 * np.pi * np.outer(m, s) / float(length_nm)
    return np.asarray((np.sum(np.cos(ph), axis=1) ** 2 + np.sum(np.sin(ph), axis=1) ** 2) / s.size, dtype=np.float64)


def bragg_modes(k: int, window: float = BRAGG_WINDOW) -> NDArray[np.int64]:
    """The modes searched for the Bragg peak: m in [floor(K (1 - w)), ceil(K (1 + w))], m >= 1."""
    lo = max(1, int(math.floor(k * (1.0 - window))))
    hi = max(lo, int(math.ceil(k * (1.0 + window))))
    return np.arange(lo, hi + 1, dtype=np.int64)


def bragg_peak(s_nm: ArrayLike, length_nm: float, window: float = BRAGG_WINDOW
               ) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """(m*, S*, a* = P / m*): the largest S(m) in the Bragg window and the spacing it implies."""
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    if s.size < 2:
        return None, None, None
    m = bragg_modes(s.size, window)
    sm = structure_factor(s, length_nm, m)
    i = int(np.argmax(sm))
    return int(m[i]), float(sm[i]), float(length_nm) / float(m[i])


def s_low(s_nm: ArrayLike, length_nm: float, n_modes: int = S_LOW_MODES) -> Optional[float]:
    """Mean S(m) over the lowest ``n_modes`` modes: the renewal plateau ~CV^2, ~0 for a lattice (descriptive)."""
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    if s.size < 2:
        return None
    return float(np.mean(structure_factor(s, length_nm, np.arange(1, int(n_modes) + 1))))


def pair_correlation_arc(s_nm: ArrayLike, length_nm: float, bin_nm: float = G_BIN_NM,
                         max_nm: float = G_MAX_NM) -> Tuple[FloatArray, FloatArray]:
    """g(s) along the arc: pairs per cyclic arc-distance bin over the uniform expectation K(K-1)/2 x 2 bin / P.
    Edge-free on a closed curve [derivation]. Returns (bin centres, g); bins beyond P/2 are dropped."""
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    length = float(length_nm)
    top = min(float(max_nm), 0.5 * length)
    edges = np.arange(0.0, top + 1e-9, float(bin_nm))
    if edges.size < 2:
        return np.zeros(0), np.zeros(0)
    centres = 0.5 * (edges[1:] + edges[:-1])
    k = s.size
    if k < 2:
        return centres, np.full(centres.size, np.nan)
    d = np.abs(s[:, None] - s[None, :])
    d = np.minimum(d, length - d)[np.triu_indices(k, 1)]
    h, _ = np.histogram(d, edges)
    expect = k * (k - 1) / 2.0 * (2.0 * float(bin_nm) / length)
    return centres, h / expect


def regularity_r_arc(s_nm: ArrayLike, length_nm: float) -> Optional[float]:
    """median arc-1NN x K / P (the project's r along the arc; Poisson ln2/2 [derivation]). Descriptive."""
    g = arc_gaps(s_nm, length_nm)
    if g.size < 2:
        return None
    return float(np.median(np.minimum(g, np.roll(g, 1))) * g.size / float(length_nm))


@dataclass(frozen=True)
class GapMeasures:
    """Every scalar gap measure of one pattern of K arc positions on a closed curve of length P."""
    k: int
    length_nm: float
    cv: Optional[float]
    rho1: Optional[float]
    excess_cv: Optional[float]
    bragg_s: Optional[float]
    bragg_m: Optional[int]
    bragg_spacing_nm: Optional[float]
    s_low: Optional[float]
    min_gap_nm: Optional[float]
    r_arc: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def gap_measures(s_nm: ArrayLike, length_nm: float) -> GapMeasures:
    """The whole catalogue on one pattern (``s_nm`` in any order; values for K < 3 are None)."""
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    length = float(length_nm)
    g = arc_gaps(s, length)
    m, sb, a = bragg_peak(s, length) if s.size >= 3 else (None, None, None)
    return GapMeasures(k=int(s.size), length_nm=length, cv=gap_cv(g) if s.size >= 3 else None, rho1=gap_rho1(g),
                       excess_cv=excess_gap_cv(g), bragg_s=sb, bragg_m=m, bragg_spacing_nm=a,
                       s_low=s_low(s, length) if s.size >= 3 else None,
                       min_gap_nm=float(np.min(g)) if g.size else None,
                       r_arc=regularity_r_arc(s, length) if s.size >= 3 else None)


def polygon_arcs(contour_nm: FloatArray) -> Tuple[FloatArray, float]:
    """Arc positions of the vertices of a closed polygon in tour order, and its length: the measures 'on the
    polygon' (descriptive only, without p: regularity audit; the tested ones use the P-spline membrane)."""
    c: FloatArray = np.asarray(contour_nm, dtype=np.float64).reshape(-1, 2)
    if c.shape[0] >= 2 and np.allclose(c[0], c[-1]):
        c = c[:-1]
    seg = np.hypot(*(np.roll(c, -1, axis=0) - c).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum[:-1], float(cum[-1])


# ============================================================================
# Generators on the circle (synthetic tests, calibration, and the centres-level matched null)
# ============================================================================

def _ssi_batch(length_nm: float, k: int, d_nm: float, u_first: FloatArray, u_rest: FloatArray
               ) -> Tuple[FloatArray, NDArray[np.bool_]]:
    """Simple sequential inhibition (Diggle, Besag & Gleaves 1976) for n patterns at once: the first point
    uniform, every next one UNIFORM ON THE FREE SET (the arcs >= d from every placed point). That is exactly the
    law of the rejection sampler ("redraw until >= d from all", ``tools.mps_simulate_axon._hard_core_arcs``) --
    rejection sampling until acceptance is uniform on the acceptance region -- without its rejections, and it
    lets the same uniforms (``u_first`` (n,), ``u_rest`` (n, k - 1)) drive every d: common random numbers make
    the mean CV a smooth function of d for the bisection. Returns (sorted positions (n, k), jammed (n,)); a jammed
    pattern (no free arc left) has NaN rows."""
    n = int(u_first.size)
    length = float(length_nm)
    d = max(float(d_nm), 0.0)
    pts = (np.asarray(u_first, dtype=np.float64) * length).reshape(n, 1)
    jammed = np.zeros(n, dtype=bool)
    for i in range(1, int(k)):
        srt = np.sort(pts, axis=1)
        nxt = np.concatenate([srt[:, 1:], srt[:, :1] + length], axis=1)
        free = np.maximum(nxt - srt - 2.0 * d, 0.0)
        cum = np.cumsum(free, axis=1)
        total = cum[:, -1]
        jammed |= ~(total > 1e-9)
        u = u_rest[:, i - 1] * np.where(total > 0, total, 1.0)
        idx = np.minimum(np.sum(cum <= u[:, None], axis=1), srt.shape[1] - 1)
        before = np.where(idx > 0, np.take_along_axis(cum, np.maximum(idx - 1, 0)[:, None], axis=1)[:, 0], 0.0)
        new = np.mod(np.take_along_axis(srt, idx[:, None], axis=1)[:, 0] + d + (u - before), length)
        pts = np.concatenate([srt, new[:, None]], axis=1)
    out = np.sort(pts, axis=1)
    out[jammed] = np.nan
    return out, jammed


def hard_core_arcs(rng: np.random.Generator, k: int, length_nm: float, d_nm: float, n: int = 1
                   ) -> Tuple[FloatArray, NDArray[np.bool_]]:
    """``n`` patterns of K points by sequential inhibition with cyclic hard core d (see ``_ssi_batch``)."""
    u0 = rng.uniform(size=int(n))
    ur = rng.uniform(size=(int(n), max(int(k) - 1, 0)))
    return _ssi_batch(length_nm, int(k), d_nm, u0, ur)


def poisson_arcs(rng: np.random.Generator, k: int, length_nm: float) -> FloatArray:
    """K uniform positions (the binomial process on the circle)."""
    return np.sort(rng.uniform(0.0, float(length_nm), int(k)))


def lattice_arcs(rng: np.random.Generator, k: int, length_nm: float, jitter_nm: float, occupancy: float = 1.0
                 ) -> FloatArray:
    """A jittered lattice: n_sites = round(K / occupancy) sites a = P / n_sites apart (random phase), exactly K of
    them occupied at random, each displaced by N(0, jitter) along the arc ("disorder of the first kind")."""
    n_sites = max(int(k), int(round(int(k) / float(occupancy))))
    a = float(length_nm) / n_sites
    sites = rng.uniform(0.0, a) + a * np.arange(n_sites)
    occ = np.sort(rng.choice(n_sites, int(k), replace=False))
    return np.sort(np.mod(sites[occ] + rng.normal(0.0, float(jitter_nm), int(k)), float(length_nm)))


def m8_gamma_arcs(rng: np.random.Generator, k: int, length_nm: float, sigma_rel: float) -> FloatArray:
    """Model M8 (P-R92 a): K iid Gamma gaps of relative sd ``sigma_rel`` (shape 1 / sigma_rel^2), rescaled to sum to
    P, random origin -- Hosemann's ideal paracrystal, "disorder of the second kind". In the generator only: no
    reading label is issued for it (the "short-range order" label waits for its pipeline validation)."""
    shape = 1.0 / float(sigma_rel) ** 2
    g = rng.gamma(shape, 1.0 / shape, int(k))
    g = g / np.sum(g) * float(length_nm)
    s = np.cumsum(g) - g[0] + rng.uniform(0.0, float(length_nm))
    return np.sort(np.mod(s, float(length_nm)))


ARRANGEMENTS = ("poisson", "hard_core", "lattice", "m8_gamma")


def planted_arrangement(kind: str, rng: np.random.Generator, k: int, length_nm: float, **kw: float) -> FloatArray:
    """Arc positions of a known arrangement, for synthetic tests and for planting through the pipeline
    (``tools.mps_pipeline_null.observe_synthetic(model, seed, molecule_arcs=[...])``): "poisson", "hard_core"
    (``d_nm``), "lattice" (``jitter_nm``, ``occupancy``), "m8_gamma" (``sigma_rel``)."""
    if kind == "poisson":
        return poisson_arcs(rng, k, length_nm)
    if kind == "hard_core":
        pts, jam = hard_core_arcs(rng, k, length_nm, float(kw.get("d_nm", 80.0)))
        if bool(jam[0]):
            raise RuntimeError(f"hard core {kw.get('d_nm')} nm jammed with K = {k} on {length_nm:.0f} nm")
        return np.asarray(pts[0], dtype=np.float64)
    if kind == "lattice":
        return lattice_arcs(rng, k, length_nm, float(kw.get("jitter_nm", 30.0)), float(kw.get("occupancy", 1.0)))
    if kind == "m8_gamma":
        return m8_gamma_arcs(rng, k, length_nm, float(kw.get("sigma_rel", 0.25)))
    raise ValueError(f"kind must be one of {ARRANGEMENTS}, got {kind!r}")


# ============================================================================
# Batched measures (the centres-level nulls draw hundreds of patterns at once)
# ============================================================================

def _batch_measures(pts: FloatArray, length_nm: float) -> Dict[str, FloatArray]:
    """cv, rho1, excess_cv, bragg_s of n patterns (rows of sorted positions; NaN rows stay NaN)."""
    length = float(length_nm)
    k = pts.shape[1]
    g = np.diff(np.concatenate([pts, pts[:, :1] + length], axis=1), axis=1)
    mean = np.mean(g, axis=1)
    cv = np.std(g, axis=1) / mean
    x = g - mean[:, None]
    den = np.sum(x * x, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho1 = np.where(den > 0, np.sum(x * np.roll(x, -1, axis=1), axis=1) / den, np.nan)
        e = g - np.min(g, axis=1)[:, None]
        em = np.mean(e, axis=1)
        exc = np.where(em > 0, np.std(e, axis=1) / em, np.nan)
    modes = bragg_modes(k).astype(np.float64)
    ph = 2.0 * np.pi * modes[None, :, None] * pts[:, None, :] / length
    sm = (np.sum(np.cos(ph), axis=2) ** 2 + np.sum(np.sin(ph), axis=2) ** 2) / k
    return {"cv": cv, "rho1": rho1, "excess_cv": exc, "bragg_s": np.max(sm, axis=1)}


# ============================================================================
# Null summaries
# ============================================================================

@dataclass(frozen=True)
class NullSummary:
    """One measure against one null: observed, the null's mean / sd / central 95 % (pointwise), Monte Carlo p
    (Besag & Diggle 1977; (1 + b)/(B + 1), never 0) and z = (observed - mean) / sd (the batch's currency)."""
    name: str
    observed: Optional[float]
    mean: Optional[float]
    sd: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    p: Optional[float]
    n: int
    tail: str                       # "two-sided" | "lower" | "upper"
    # G1 (c): failed replicates of the null counted in ``p`` in the worst case; ``p_ignoring_failed`` the p over the
    # analysed replicates only (descriptive, exported beside it)
    n_failed: int = 0
    p_ignoring_failed: Optional[float] = None

    @property
    def z(self) -> Optional[float]:
        if self.observed is None or self.mean is None or self.sd is None or not self.sd > 0:
            return None
        return (self.observed - self.mean) / self.sd


def summarise(name: str, observed: Optional[float], null_values: Sequence[Optional[float]], tail: str,
              n_failed: int = 0) -> NullSummary:
    """A ``NullSummary`` from the null's values (None / NaN dropped). ``tail``: "two-sided", "lower", "upper".
    ``n_failed``: replicates of the null that failed (G1 c) -- counted in ``p`` as at least as extreme as the observed
    in the tail(s) tested (``tools.mps_pipeline_null.monte_carlo_p_worst_case``); ``p_ignoring_failed`` leaves them
    out (descriptive). The mean, sd and interval are over the analysed replicates."""
    if tail not in ("two-sided", "upper", "lower"):
        raise ValueError(f"tail must be two-sided, lower or upper, got {tail!r}")
    v = np.array([float(x) for x in null_values if x is not None and math.isfinite(float(x))], dtype=np.float64)
    f = max(0, int(n_failed))
    if v.size == 0:
        return NullSummary(name, observed, None, None, None, None, None, 0, tail, f, None)
    p: Optional[float] = None
    p0: Optional[float] = None
    if observed is not None and math.isfinite(float(observed)):
        o, vv = (-float(observed), -v) if tail == "lower" else (float(observed), v)
        two = tail == "two-sided"
        p = pn.monte_carlo_p_worst_case(o, vv, two_sided=two, n_failed=f)
        p0 = monte_carlo_p(o, vv, two_sided=two)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return NullSummary(name, observed, float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else None,
                       float(lo), float(hi), p, int(v.size), tail, f, p0)


# ============================================================================
# Readouts of the pipeline null (one null, D7: the same replicates give every p)
# ============================================================================

def _view_measures(view: pn.ReplicateView) -> Optional[GapMeasures]:
    if view.arc_s_nm is None or not view.membrane_length_nm > 0 or view.arc_s_nm.size < 3:
        return None
    return gap_measures(view.arc_s_nm, view.membrane_length_nm)


def _ro_gap_cv(view: pn.ReplicateView) -> Optional[float]:
    m = _view_measures(view)
    return None if m is None else m.cv


def _ro_gap_rho1(view: pn.ReplicateView) -> Optional[float]:
    m = _view_measures(view)
    return None if m is None else m.rho1


def _ro_gap_excess_cv(view: pn.ReplicateView) -> Optional[float]:
    m = _view_measures(view)
    return None if m is None else m.excess_cv


def _ro_bragg_s(view: pn.ReplicateView) -> Optional[float]:
    m = _view_measures(view)
    return None if m is None else m.bragg_s


def _ro_elongation_p95(view: pn.ReplicateView) -> Optional[float]:
    e = cluster_elongations(view.analysis, view.membrane_nm, centroids=view.centroids)
    return float(np.quantile(e, ELONGATION_QUANTILE)) if e.size else None


# name -> (function, two-sided, description). The p of each is in PipelineNullResult.p_values[name].
NULL_READOUTS: Dict[str, Tuple[Callable[[pn.ReplicateView], Optional[float]], bool, str]] = {
    "gap_cv": (_ro_gap_cv, True, "gap CV on the P-spline membrane (stage 1 of the regularity reading; two-sided, P-R91)"),
    "gap_rho1": (_ro_gap_rho1, True, "lag-1 correlation of consecutive gaps (descriptive against this null)"),
    "gap_excess_cv": (_ro_gap_excess_cv, True, "CV of the gaps in excess of the smallest gap"),
    "bragg_s": (_ro_bragg_s, False, "Bragg peak S* of the periodogram, m in [K/2, 3K/2] (descriptive against this null)"),
    "elongation_p95": (_ro_elongation_p95, False, "95th percentile of the clusters' tangential/normal sd ratio "
                       "(the merged-cluster flag's threshold)"),
}
TAILS_NULL1 = {"gap_cv": "two-sided", "gap_rho1": "two-sided", "gap_excess_cv": "two-sided", "bragg_s": "upper"}


def register_null_readouts(replace: bool = True) -> None:
    """Register the gap readouts with the pipeline null (``tools.mps_pipeline_null.register_readout``), so their
    Monte Carlo p come out of the same replicates as every other readout (decision D7). Idempotent."""
    for name, (fn, two, desc) in NULL_READOUTS.items():
        if name in pn.READOUTS and not replace:
            continue
        pn.register_readout(pn.ReadoutSpec(name, fn, "", True, two, desc), replace=True)


register_null_readouts()


# ============================================================================
# Arc positions and elongation of an analysed axon
# ============================================================================

def axon_arcs(analysis: Any) -> Tuple[Optional[FloatArray], float, Optional[FloatArray]]:
    """(arc positions of the kept centroids on the 600 nm P-spline membrane, its length, the membrane) -- the same
    computation the pipeline null applies to every replicate (``tools.mps_pipeline_null._view``)."""
    v = pn._view(analysis, True, observed=True)
    return v.arc_s_nm, float(v.membrane_length_nm), v.membrane_nm


def _tangents(membrane: FloatArray, s: FloatArray) -> FloatArray:
    """Unit tangents of the closed polyline ``membrane`` at arc positions s (central difference of +/- 5 nm)."""
    c = np.asarray(membrane, dtype=np.float64)
    if np.allclose(c[0], c[-1]):
        c = c[:-1]
    seg = np.hypot(*(np.roll(c, -1, axis=0) - c).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(cum[-1])
    cc = np.vstack([c, c[:1]])

    def at(q: FloatArray) -> FloatArray:
        q = np.mod(q, length)
        return np.column_stack([np.interp(q, cum, cc[:, 0]), np.interp(q, cum, cc[:, 1])])

    t = at(s + TANGENT_STEP_NM) - at(s - TANGENT_STEP_NM)
    nrm = np.hypot(t[:, 0], t[:, 1])
    nrm[nrm == 0] = 1.0
    return np.asarray(t / nrm[:, None], dtype=np.float64)


def cluster_elongations(analysis: Any, membrane_nm: Optional[FloatArray], *, centroids: Optional[FloatArray] = None,
                        min_locs: int = ELONGATION_MIN_LOCS) -> FloatArray:
    """Per kept cluster with >= ``min_locs`` localizations: sd along the membrane tangent at its centroid / sd along
    the normal. Merged pairs are elongated along the contour (P-R84 a: they are marked and counted, never split).
    NaN-free; empty when there is no membrane."""
    if membrane_nm is None:
        return np.zeros(0)
    cen = np.asarray(analysis.centroids if centroids is None else centroids, dtype=np.float64).reshape(-1, 2)
    if cen.shape[0] == 0:
        return np.zeros(0)
    s, _d, _len = arc_coordinates(cen, np.asarray(membrane_nm, dtype=np.float64))
    tan = _tangents(np.asarray(membrane_nm, dtype=np.float64), s)
    labels = np.asarray(analysis.labels)
    kept = pn._kept_labels(analysis)
    out: List[float] = []
    for i, lab in enumerate(kept[: cen.shape[0]]):
        sel = labels == lab
        if int(np.count_nonzero(sel)) < int(min_locs):
            continue
        dx = np.asarray(analysis.x_slab)[sel] - cen[i, 0]
        dy = np.asarray(analysis.y_slab)[sel] - cen[i, 1]
        st = float(np.std(dx * tan[i, 0] + dy * tan[i, 1], ddof=1))
        sn = float(np.std(-dx * tan[i, 1] + dy * tan[i, 0], ddof=1))
        if sn > 0 and math.isfinite(st):
            out.append(st / sn)
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class ElongationFlag:
    """P-R84 (a): clusters more elongated along the contour than the null's 95th percentile ("probable merged")."""
    n_measured: int
    n_elongated: int
    threshold: Optional[float]
    expected_under_null: Optional[float]
    elongations: Tuple[float, ...] = ()


def elongation_flag(analysis: Any, membrane_nm: Optional[FloatArray], null_p95: Sequence[Optional[float]]
                    ) -> ElongationFlag:
    """Count the kept clusters whose elongation exceeds the threshold = median over the null replicates of their
    95th percentile [ad hoc: the median of per-replicate quantiles approximates the pooled one]; under the null about
    5 % of the measured clusters exceed it by construction ("expected")."""
    e = cluster_elongations(analysis, membrane_nm)
    v = [float(x) for x in null_p95 if x is not None and math.isfinite(float(x))]
    thr = float(np.median(v)) if v else None
    n_el = int(np.count_nonzero(e > thr)) if thr is not None else 0
    exp = (1.0 - ELONGATION_QUANTILE) * e.size if thr is not None else None
    return ElongationFlag(int(e.size), n_el, thr, exp, tuple(float(x) for x in e))


def min_1nn_nm(centroids: FloatArray) -> Optional[float]:
    """Smallest Euclidean 1NN among the kept centroids (the 'minimum below contact' safeguard)."""
    c = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    if c.shape[0] < 2:
        return None
    d, _ = cKDTree(c).query(c, k=2)
    return float(np.min(d[:, 1]))


# ============================================================================
# Stage 2: the matched-exclusion null
# ============================================================================

@dataclass
class MatchedNullResult:
    """The matched-exclusion null of one axon: the hard core d_m whose mean gap CV equals the observed one
    (bisection; Diggle 1979's minimum contrast -- an "effective exclusion" estimate), and rho1 (lower tail) and S*
    (upper tail) against it. ``kind``: "pipeline" (exact) or "centres_approx" (on the detected centroids)."""
    kind: str
    d_m_nm: float
    capped: bool
    cap_nm: float
    cv_target: float
    cv_null_at_d_m: Optional[float]
    rho1: NullSummary
    bragg_s: NullSummary
    cv: NullSummary
    excess_cv: NullSummary
    n_replicates: int
    seed: int
    bisection: List[Tuple[float, float]] = field(default_factory=list)   # (d, mean CV) per step
    notes: List[str] = field(default_factory=list)
    null_result: Optional[Any] = None      # the PipelineNullResult of the final pipeline-level run
    interpretable: bool = True              # False: stage 2 cannot be read (the reason is in not_interpretable)
    not_interpretable: str = ""

    @property
    def label(self) -> str:
        return "pipeline" if self.kind == "pipeline" else "approx., centres"


def _seed_int(*key: int) -> int:
    return int(np.random.SeedSequence([int(k) & 0xFFFFFFFF for k in key]).generate_state(1)[0])


def matched_null_centres(s_nm: ArrayLike, length_nm: float, *, n_replicates: int = DEFAULT_N_REPLICATES,
                         seed: int = 0, steps: int = CENTRES_BISECTION_STEPS, n_bisect: int = CENTRES_BISECTION_N,
                         cap_fraction: float = MATCHED_CAP_FRACTION) -> MatchedNullResult:
    """Stage 2 at the CENTRES level (P-R87 b, "approx., centres"): sequential inhibition of the observed K on a circle
    of the membrane's length, d_m by bisection on [0, cap] with common random numbers, then B fresh patterns. The
    merging the clustering method does is NOT in this null (it moves rho1 from -0.49 to -0.38 and raises S* by ~0.5
    [simulation: regularity research sim7, seed 7]): a preview, labelled so."""
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    k, length = int(s.size), float(length_nm)
    if k < MIN_CLUSTERS:
        raise ValueError(f"K = {k} < {MIN_CLUSTERS}: not testable")
    obs = gap_measures(s, length)
    assert obs.cv is not None
    cap = float(cap_fraction) * length / k
    rng_b = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(21,)))
    u0, ur = rng_b.uniform(size=int(n_bisect)), rng_b.uniform(size=(int(n_bisect), k - 1))

    def mean_cv(d: float) -> float:
        pts, jam = _ssi_batch(length, k, d, u0, ur)
        v = _batch_measures(pts[~jam], length)["cv"] if np.any(~jam) else np.zeros(0)
        return float(np.mean(v)) if v.size else float("nan")

    traj: List[Tuple[float, float]] = []
    c_cap = mean_cv(cap)
    traj.append((cap, c_cap))
    capped = False
    if not (c_cap < obs.cv):              # even the densest allowed null is less regular: cap and say so
        d_m, capped = cap, True
    else:
        lo, hi = 0.0, cap
        for _ in range(int(steps)):
            mid = 0.5 * (lo + hi)
            c = mean_cv(mid)
            traj.append((mid, c))
            if c > obs.cv:
                lo = mid
            else:
                hi = mid
        d_m = 0.5 * (lo + hi)
    rng_n = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(22,)))
    pts, jam = hard_core_arcs(rng_n, k, length, d_m, int(n_replicates))
    good = pts[~jam]
    notes: List[str] = []
    if np.any(jam):
        notes.append(f"{int(np.count_nonzero(jam))} matched-null patterns jammed and were left out.")
    if capped:
        notes.append(f"d_m capped at {cap_fraction:.2f} x P/K = {cap:.0f} nm: no exclusion this side of packing is "
                     "as regular as the observed arrangement.")
    bm = _batch_measures(good, length) if good.shape[0] else {n: np.zeros(0) for n in ("cv", "rho1", "excess_cv",
                                                                                         "bragg_s")}
    return MatchedNullResult(
        kind="centres_approx", d_m_nm=float(d_m), capped=capped, cap_nm=cap, cv_target=float(obs.cv),
        cv_null_at_d_m=float(np.mean(bm["cv"])) if bm["cv"].size else None,
        rho1=summarise("gap_rho1", obs.rho1, list(bm["rho1"]), "lower"),
        bragg_s=summarise("bragg_s", obs.bragg_s, list(bm["bragg_s"]), "upper"),
        cv=summarise("gap_cv", obs.cv, list(bm["cv"]), "two-sided"),
        excess_cv=summarise("gap_excess_cv", obs.excess_cv, list(bm["excess_cv"]), "two-sided"),
        n_replicates=int(good.shape[0]), seed=int(seed), bisection=traj, notes=notes)


def matched_null_pipeline(analysis: Any, stage1: Any, *, spec: Optional[pn.PipelineSpec] = None,
                          n_replicates: int = DEFAULT_N_REPLICATES, seed: int = 0,
                          steps: int = PIPELINE_BISECTION_STEPS, n_bisect: int = PIPELINE_BISECTION_N,
                          cap_fraction: float = MATCHED_CAP_FRACTION, nena_nm: Optional[float] = None,
                          allow_slow: bool = False, n_jobs: int = 1,
                          progress: Optional[Callable[[str, int, int], None]] = None) -> MatchedNullResult:
    """Stage 2 at the PIPELINE level (P-R87 a, exact): the pipeline null of stage 1 (``stage1``, a
    ``PipelineNullResult``: its model, closure and method) re-run with the molecules' hard core raised to d_m, found
    by bisection on the null's mean detected gap CV (``n_bisect`` replicates per step, the same seed at every step:
    common random numbers), then ``n_replicates`` fresh replicates (another seed). A hard core the ring cannot pack
    counts as "too regular" for the bisection (its note says so). Costs (steps x n_bisect + B) analyses of the
    observed method: refused for slow methods unless ``allow_slow``."""
    s_obs, length, _mem = axon_arcs(analysis)
    if s_obs is None or s_obs.size < MIN_CLUSTERS:
        raise ValueError("not testable: fewer than 8 centroids on a ring")
    obs = gap_measures(s_obs, length)
    assert obs.cv is not None
    model0 = stage1.model
    n_mol = float(model0.n_molecules_per_ring)
    mem_len = float(model0.membrane_length_nm()) if model0.membrane_nm is not None else length
    cap = float(cap_fraction) * mem_len / max(n_mol, 1.0)
    closure = str(stage1.closure)
    base_seed = _seed_int(int(seed), 31)
    notes: List[str] = []
    traj: List[Tuple[float, float]] = []

    def run(d: float, b: int, sd: int, names: Sequence[str]) -> Optional[Any]:
        model = dataclasses.replace(model0, hard_core_nm=float(d), condition_slab_locs=None,
                                    origins=dict(model0.origins))
        model.origins["hard_core_nm"] = "matched-exclusion null (stage 2), bisection on the mean gap CV"
        try:
            return pn.run_pipeline_null(analysis, spec=spec, model=model, nena_nm=nena_nm, n_replicates=int(b),
                                        seed=int(sd), closure=closure, allow_slow=allow_slow, n_jobs=n_jobs,
                                        readouts=list(names), keep_example=False)
        except pn.SlowMethodRefused:
            raise
        except pn.NullRefused as exc:
            notes.append(f"hard core {d:.0f} nm refused: {exc}")
            return None
        except RuntimeError as exc:
            # the closure's pilots can ask for more molecules than stage 1 had and jam the sequential sampler
            # (tools.mps_simulate_axon._hard_core_arcs): the same meaning as a refused packing
            if "jammed" not in str(exc):
                raise
            notes.append(f"hard core {d:.0f} nm jammed while closing the null: {exc}")
            return None

    def mean_cv(d: float, i: int) -> float:
        if progress is not None:
            progress("matched null: bisection", i, int(steps) + 1)
        res = run(d, n_bisect, base_seed, ["gap_cv"])
        if res is None:
            return 0.0                      # cannot pack: treated as more regular than any observed CV
        v = [r.get("gap_cv") for r in res.replicate_readouts]
        vv = [float(x) for x in v if x is not None]
        return float(np.mean(vv)) if vv else float("nan")

    c_cap = mean_cv(cap, 0)
    traj.append((cap, c_cap))
    capped = False
    if not (c_cap < obs.cv):
        d_m, capped = cap, True
    else:
        lo = 0.0
        hi = cap
        for i in range(int(steps)):
            mid = 0.5 * (lo + hi)
            c = mean_cv(mid, i + 1)
            traj.append((mid, c))
            if c > obs.cv:
                lo = mid
            else:
                hi = mid
        d_m = 0.5 * (lo + hi)
    if progress is not None:
        progress("matched null: final replicates", int(steps) + 1, int(steps) + 1)
    final = None
    for attempt in range(FINAL_BACKOFF_TRIES):
        final = run(d_m, n_replicates, _seed_int(int(seed), 32), ["gap_cv", "gap_rho1", "gap_excess_cv", "bragg_s"])
        if final is not None:
            break
        # the bisection's d_m cannot be packed by the final run's closure: back off and say so
        d_m *= FINAL_BACKOFF_FACTOR
        capped = True
        notes.append(f"d_m lowered to {d_m:.0f} nm (attempt {attempt + 2}): the previous value could not be packed.")
    if capped:
        notes.append(f"d_m capped at {cap_fraction:.2f} x P / molecules = {cap:.0f} nm.")
    rows: List[Mapping[str, Optional[float]]] = [] if final is None else list(final.replicate_readouts)

    def col(name: str) -> List[Optional[float]]:
        return [r.get(name) for r in rows]

    interpretable, why = True, ""
    failed = 0 if final is None else int(final.n_failed)
    if final is not None and failed:
        # G1 (c): the null's own rule (floor(0.05 B) tolerated, each failure counted in the worst case); the
        # failures are no longer dropped from the p
        notes.append(f"{failed} of {final.n_requested} simulated axons of the matched null failed: "
                     + (f"tolerated (at most {final.max_failed}, 5 %), each counted as at least as extreme as the "
                        "observed in the stage-2 p values (conservative)." if failed <= final.max_failed else
                        f"more than the {final.max_failed} (5 %) tolerated."))
    if final is not None and not final.interpretable:
        why = " ".join(final.not_interpretable_reasons())
        interpretable = False
        notes.append("matched pipeline null NOT INTERPRETABLE: " + why)
    cvs = [float(x) for x in col("gap_cv") if x is not None]
    return MatchedNullResult(
        kind="pipeline", d_m_nm=float(d_m), capped=capped, cap_nm=cap, cv_target=float(obs.cv),
        cv_null_at_d_m=float(np.mean(cvs)) if cvs else None,
        rho1=summarise("gap_rho1", obs.rho1, col("gap_rho1"), "lower", failed),
        bragg_s=summarise("bragg_s", obs.bragg_s, col("bragg_s"), "upper", failed),
        cv=summarise("gap_cv", obs.cv, col("gap_cv"), "two-sided", failed),
        excess_cv=summarise("gap_excess_cv", obs.excess_cv, col("gap_excess_cv"), "two-sided", failed),
        n_replicates=len(rows), seed=int(seed), bisection=traj, notes=notes, null_result=final,
        interpretable=interpretable, not_interpretable=why)


# ============================================================================
# The reading rule
# ============================================================================

@dataclass(frozen=True)
class Reading:
    """The verdict of the two-stage rule (one of ``READING_LEVELS``), the minimum-below-contact flag, which stage 2
    ran, and the sentence (always "at the level of this axon", with the p)."""
    level: str
    flag_min_below_contact: bool
    stage2_kind: str
    sentence: str
    reason: str = ""


def _f(v: Optional[float], fmt: str = ".2f") -> str:
    return "n/a" if v is None else format(v, fmt)


def regularity_reading(*, k: int, cv: NullSummary, rho1_matched: Optional[NullSummary] = None,
                       bragg_matched: Optional[NullSummary] = None, matched: Optional[MatchedNullResult] = None,
                       excess_cv: Optional[float] = None, bragg_m: Optional[int] = None,
                       bragg_spacing_nm: Optional[float] = None, min_nn_nm: Optional[float] = None,
                       d_contact_nm: Optional[float] = None, alpha: float = ALPHA,
                       rho1_max: float = RHO1_LATTICE_MAX, not_testable_reason: str = "") -> Reading:
    """The pre-written rule of the regularity plan sec. 5 (module docstring). ``cv`` is the gap CV against the
    pipeline null (two-sided); ``rho1_matched`` / ``bragg_matched`` against the matched-exclusion null (lower /
    upper tail), None when stage 2 did not run. Pure: the tests drive it with hand-made summaries."""
    flag = (min_nn_nm is not None and d_contact_nm is not None and float(min_nn_nm) < float(d_contact_nm))
    pre = (f"Minimum 1NN ({min_nn_nm:.0f} nm) below the contact distance d_c = {d_contact_nm:.0f} nm: split "
           "clusters or clusters of another ring. " if flag and min_nn_nm is not None and d_contact_nm is not None
           else "")
    kind = STAGE2_KINDS[2] if matched is None else matched.kind
    if k < MIN_CLUSTERS or not_testable_reason or cv.p is None or cv.observed is None or cv.mean is None:
        why = not_testable_reason or (f"K = {k} < {MIN_CLUSTERS} clusters" if k < MIN_CLUSTERS
                                      else "the gap CV or its null could not be computed")
        return Reading(NOT_TESTABLE, flag, STAGE2_KINDS[2], pre + f"Not testable at the level of this axon: {why}.",
                       why)
    s1 = (f"gap CV {cv.observed:.2f} against {cv.mean:.2f} [{_f(cv.lo)}-{_f(cv.hi)}] of the pipeline null, "
          f"p = {cv.p:.3f} approx., B = {cv.n}")
    if not cv.p <= alpha:
        return Reading(RANDOM_WITH_CONTACT, flag, STAGE2_KINDS[2],
                       pre + f"At the level of this axon the K = {k} clusters are not distinguishable from random "
                       f"placement with contact once detection is accounted for ({s1}). " + APPROX_NOTE)
    if cv.observed > cv.mean:
        return Reading(LESS_REGULAR, flag, STAGE2_KINDS[2],
                       pre + f"At the level of this axon the K = {k} clusters are LESS regularly spaced than random "
                       f"placement with contact ({s1}): missed clusters (thinning) or clusters grouped together. "
                       + APPROX_NOTE)
    head = (pre + f"At the level of this axon the K = {k} clusters are more regularly spaced than random placement "
            f"with contact, once detection is accounted for ({s1}).")
    if matched is None or rho1_matched is None or bragg_matched is None or rho1_matched.p is None \
            or bragg_matched.p is None:
        return Reading(KIND_NOT_IDENTIFIED, flag, kind,
                       head + " Stage 2 (an exclusion of the same strength) did not run: the kind of regularity is "
                       f"not identified (excess-gap CV {_f(excess_cv)}, descriptive). " + APPROX_NOTE)
    if not getattr(matched, "interpretable", True):
        return Reading(KIND_NOT_IDENTIFIED, flag, kind,
                       head + f" Stage 2 is NOT INTERPRETABLE ({getattr(matched, 'not_interpretable', '')}): the kind of regularity is "
                       f"not identified (excess-gap CV {_f(excess_cv)}, descriptive). " + APPROX_NOTE)
    rho_obs = rho1_matched.observed
    rho_rej = rho1_matched.p <= alpha and rho_obs is not None and rho_obs <= rho1_max
    bragg_rej = bragg_matched.p <= alpha
    where = "pipeline" if matched.kind == "pipeline" else "approx., centres"
    cmp = (f"Against an exclusion of the same strength (matched hard core d_m = {matched.d_m_nm:.0f} nm"
           f"{', capped' if matched.capped else ''}; {where}, B = {matched.n_replicates})")
    rho_txt = f"rho1 = {_f(rho_obs, '+.2f')}, p = {rho1_matched.p:.3f}"
    peak_txt = (f"peak at m* = {bragg_m} (a* = {_f(bragg_spacing_nm, '.0f')} nm, S* = {_f(bragg_matched.observed, '.1f')} "
                f"against {_f(bragg_matched.mean, '.1f')} [{_f(bragg_matched.lo, '.1f')}-{_f(bragg_matched.hi, '.1f')}], "
                f"p = {bragg_matched.p:.3f})")
    if rho_rej and bragg_rej:
        return Reading(LATTICE, flag, kind, head + f" {cmp}, consecutive gaps are anticorrelated ({rho_txt}) and there "
                       f"is a {peak_txt}: lattice signature at the level of this axon (spacing a* = P/m*). "
                       + APPROX_NOTE)
    if rho_rej:
        return Reading(LATTICE_RHO1_ONLY, flag, kind, head + f" {cmp}, consecutive gaps are anticorrelated "
                       f"({rho_txt}) without a resolvable peak (Bragg p = {bragg_matched.p:.3f}): a lattice with large "
                       "jitter, at the level of this axon. " + APPROX_NOTE)
    if bragg_rej:
        return Reading(PERIODIC_COMPONENT, flag, kind, head + f" {cmp}, there is a periodic component: {peak_txt}; "
                       f"{rho_txt}. A lattice with empty sites, undetected clusters or residual structure of a strong "
                       "exclusion: indistinguishable from the arrangement alone. " + APPROX_NOTE)
    return Reading(KIND_NOT_IDENTIFIED, flag, kind, head + f" {cmp}, there is no evidence of a periodic component "
                   f"(Bragg p = {bragg_matched.p:.3f}; {rho_txt}): the kind of regularity is not identified with "
                   f"K = {k}; excess-gap CV {_f(excess_cv)} (~1 suggests exclusion, < 1 bell-shaped gaps; "
                   "descriptive). " + APPROX_NOTE)


# ============================================================================
# One axon, end to end
# ============================================================================

@dataclass
class GapRegularityResult:
    """Everything the gap analysis of one axon gives (``export_columns`` = the regularity plan sec. 6.2 columns)."""
    k: int
    measures: GapMeasures
    measures_polygon: Optional[GapMeasures]
    stage1: Dict[str, NullSummary]
    matched_centres: Optional[MatchedNullResult]
    matched_pipeline: Optional[MatchedNullResult]
    reading: Reading
    min_nn_nm: Optional[float]
    d_contact_nm: Optional[float]
    elongation: Optional[ElongationFlag]
    null_result: Optional[Any] = None              # the stage-1 PipelineNullResult
    warnings: List[str] = field(default_factory=list)
    # The level stage 2 was SET to run at ("centres" in the window, "pipeline" in the Batch: P-R87 c), whether or not
    # stage 1 asked for it; exported as regularity_stage2_level so a window row and a Batch row of one axon are never
    # compared blindly (design review 2026-10-29: 4 of 60 synthetic axons read differently at the two levels).
    stage2_level: str = ""

    @property
    def matched(self) -> Optional[MatchedNullResult]:
        """The stage-2 null the reading used: the pipeline one when it ran, else the centres one."""
        return self.matched_pipeline if self.matched_pipeline is not None else self.matched_centres

    def export_columns(self) -> Dict[str, Any]:
        """The regularity plan sec. 6.2 columns (P-R86: ``_matched`` for the stage-2 p values), plus the z scores
        the batch compares (``*_z``, ``*_z_matched``) and ``matched_null_kind``."""
        d: Dict[str, Any] = {}
        m = self.measures

        def put(prefix: str, value: Optional[float], s: Optional[NullSummary], p_name: Optional[str]) -> None:
            d[prefix] = value
            d[f"{prefix}_null_mean"] = None if s is None else s.mean
            d[f"{prefix}_null_lo"] = None if s is None else s.lo
            d[f"{prefix}_null_hi"] = None if s is None else s.hi
            if p_name is not None:
                d[p_name] = None if s is None else s.p
            d[f"{prefix}_z"] = None if s is None else s.z

        put("gap_cv", m.cv, self.stage1.get("gap_cv"), "gap_cv_p_mc")
        put("gap_rho1", m.rho1, self.stage1.get("gap_rho1"), None)
        put("gap_excess_cv", m.excess_cv, self.stage1.get("gap_excess_cv"), "gap_excess_cv_p_mc")
        put("bragg_s", m.bragg_s, self.stage1.get("bragg_s"), None)
        mt = self.matched
        d["gap_rho1_p_mc_matched"] = None if mt is None else mt.rho1.p
        d["gap_rho1_z_matched"] = None if mt is None else mt.rho1.z
        d["bragg_p_mc_matched"] = None if mt is None else mt.bragg_s.p
        # G1 (c): the same p without the failed replicates (descriptive), and how many failed
        for col_name, summ in (("gap_cv_p_mc_ignoring_failed", self.stage1.get("gap_cv")),
                               ("gap_excess_cv_p_mc_ignoring_failed", self.stage1.get("gap_excess_cv")),
                               ("gap_rho1_p_mc_matched_ignoring_failed", None if mt is None else mt.rho1),
                               ("bragg_p_mc_matched_ignoring_failed", None if mt is None else mt.bragg_s)):
            d[col_name] = None if summ is None else summ.p_ignoring_failed
        d["matched_null_failed_replicates"] = (None if mt is None or mt.null_result is None
                                               else int(getattr(mt.null_result, "n_failed", 0) or 0))
        d["bragg_s_z_matched"] = None if mt is None else mt.bragg_s.z
        d["bragg_m"] = m.bragg_m
        d["bragg_spacing_nm"] = m.bragg_spacing_nm
        d["s_low"] = m.s_low
        mp = self.measures_polygon
        for name in ("cv", "rho1", "excess_cv"):
            d[f"gap_{name}_polygon"] = None if mp is None else getattr(mp, name)
        d["bragg_s_polygon"] = None if mp is None else mp.bragg_s
        d["regularity_reading"] = self.reading.level
        d["regularity_flag_min_below_contact"] = bool(self.reading.flag_min_below_contact)
        d["matched_null_kind"] = self.reading.stage2_kind
        d["matched_null_d_nm"] = None if mt is None else round(mt.d_m_nm, 1)
        d["matched_null_B"] = None if mt is None else mt.n_replicates
        d["matched_null_capped"] = None if mt is None else bool(mt.capped)
        nr = self.null_result
        if nr is not None:
            ks = list(getattr(nr, "k_null", []) or [])
            d.update({
                "pipeline_null_B": nr.n_accepted, "pipeline_null_K_sim": round(float(nr.model.n_molecules_per_ring), 2),
                "pipeline_null_d_contact_nm": round(float(nr.model.hard_core_nm), 1),
                "pipeline_null_epitope_R_nm": round(float(nr.model.epitope_radius_nm), 1),
                "pipeline_null_lp_nm": round(float(nr.model.sigma_xy_nm), 2),
                "pipeline_null_K_det_mean": round(float(np.mean(ks)), 2) if ks else None,
                "pipeline_null_closure_ratio": None if nr.closure_ratio is None else round(float(nr.closure_ratio), 3),
                "pipeline_null_seed": nr.seed,
            })
        el = self.elongation
        d["n_clusters_elongated"] = None if el is None else el.n_elongated
        d["elongation_threshold"] = None if el is None or el.threshold is None else round(el.threshold, 3)
        d["n_clusters_elongated_null_expected"] = (None if el is None or el.expected_under_null is None
                                                   else round(el.expected_under_null, 2))
        d["nn_regularity_null_kind"] = "pipeline" if nr is not None else None
        d["regularity_stage2_level"] = self.stage2_level or None
        return d


def analyze_gap_regularity(analysis: Any, *, null_result: Optional[Any] = None, stage2: str = "centres",
                           seed: int = 0, n_replicates: int = DEFAULT_N_REPLICATES,
                           n_matched: int = DEFAULT_N_REPLICATES, nena_nm: Optional[float] = None,
                           alpha: float = ALPHA, allow_slow: bool = False, spec: Optional[pn.PipelineSpec] = None,
                           n_jobs: int = 1, progress: Optional[Callable[[str, int, int], None]] = None,
                           null_kw: Optional[Dict[str, Any]] = None,
                           matched_kw: Optional[Dict[str, Any]] = None) -> GapRegularityResult:
    """The gap analysis of one analysed axon (any clustering method).

    ``null_result``: a stage-1 ``PipelineNullResult`` already computed for this analysis (D7: one null for every
    readout; it must have been run with the gap readouts registered, which importing this module does); when None,
    the pipeline null is run here (``n_replicates``, ``seed``, ``nena_nm``, ``null_kw`` passed on; ``matched_kw``
    goes to the pipeline-level matched null, e.g. ``steps`` / ``n_bisect``). ``stage2``:
    "centres" (default; the GUI preview, "approx., centres"), "pipeline" (exact; the batch), "both", or "none".
    Stage 2 runs only when stage 1 says "more regular". K < 8 or no ring: ``not_testable`` without running a null.
    Raises ``tools.mps_pipeline_null.SlowMethodRefused`` for a slow method without ``allow_slow``."""
    if stage2 not in STAGE2_MODES:
        raise ValueError(f"stage2 must be one of {STAGE2_MODES}, got {stage2!r}")
    k = int(getattr(analysis, "n_clusters_kept", 0) or 0)
    cen = np.asarray(analysis.centroids, dtype=np.float64).reshape(-1, 2)
    mn = min_1nn_nm(cen)
    warnings: List[str] = []
    s_obs: Optional[FloatArray] = None
    length = float("nan")
    membrane: Optional[FloatArray] = None
    if k >= 3 and getattr(analysis, "perimeter", None) is not None:
        s_obs, length, membrane = axon_arcs(analysis)
    measures = gap_measures(s_obs if s_obs is not None else np.zeros(0), length if length > 0 else 1.0)
    poly: Optional[GapMeasures] = None
    if getattr(analysis, "perimeter", None) is not None and k >= 3:
        ps, pl = polygon_arcs(np.asarray(analysis.perimeter.contour, dtype=np.float64))
        poly = gap_measures(ps, pl)
    empty = summarise("gap_cv", measures.cv, [], "two-sided")
    if k < MIN_CLUSTERS or s_obs is None:
        why = f"K = {k} < {MIN_CLUSTERS} clusters" if k < MIN_CLUSTERS else "no contour (ring) for this axon"
        rd = regularity_reading(k=k, cv=empty, not_testable_reason=why)
        return GapRegularityResult(k, measures, poly, {}, None, None, rd, mn, None, None, None, warnings,
                                   stage2_level=str(stage2))
    if null_result is None:
        null_result = pn.run_pipeline_null(analysis, spec=spec, nena_nm=nena_nm, n_replicates=int(n_replicates),
                                           seed=int(seed), allow_slow=allow_slow, n_jobs=n_jobs,
                                           progress=None if progress is None else
                                           (lambda done, total: progress("pipeline null", done, total)),
                                           **(null_kw or {}))
    missing = [n for n in TAILS_NULL1 if n not in null_result.observed]
    if missing:
        warnings.append(f"the pipeline null was run without the gap readouts {missing}: import tools.mps_gap_measures "
                        "before running it.")
    stage1: Dict[str, NullSummary] = {}
    for name, tail in TAILS_NULL1.items():
        stage1[name] = summarise(name, getattr(measures, {"gap_cv": "cv", "gap_rho1": "rho1",
                                                          "gap_excess_cv": "excess_cv", "bragg_s": "bragg_s"}[name]),
                                 [r.get(name) for r in null_result.replicate_readouts], tail,
                                 int(getattr(null_result, "n_failed", 0) or 0))
    d_c = float(null_result.model.hard_core_nm)
    el = elongation_flag(analysis, membrane, [r.get("elongation_p95") for r in null_result.replicate_readouts])
    reason = "" if null_result.interpretable else ("pipeline null NOT INTERPRETABLE: "
                                                   + " ".join(null_result.not_interpretable_reasons()))
    cv1 = stage1["gap_cv"]
    more_regular = (not reason and cv1.p is not None and cv1.p <= alpha and cv1.observed is not None
                    and cv1.mean is not None and cv1.observed < cv1.mean)
    mc: Optional[MatchedNullResult] = None
    mp: Optional[MatchedNullResult] = None
    if more_regular and stage2 in ("centres", "both"):
        mc = matched_null_centres(s_obs, length, n_replicates=int(n_matched), seed=int(seed))
    if more_regular and stage2 in ("pipeline", "both"):
        mp = matched_null_pipeline(analysis, null_result, spec=spec, n_replicates=int(n_matched), seed=int(seed),
                                   nena_nm=nena_nm, allow_slow=allow_slow, n_jobs=n_jobs, progress=progress,
                                   **(matched_kw or {}))
    mt = mp if mp is not None else mc
    rd = regularity_reading(k=k, cv=cv1, rho1_matched=None if mt is None else mt.rho1,
                            bragg_matched=None if mt is None else mt.bragg_s, matched=mt,
                            excess_cv=measures.excess_cv, bragg_m=measures.bragg_m,
                            bragg_spacing_nm=measures.bragg_spacing_nm, min_nn_nm=mn, d_contact_nm=d_c, alpha=alpha,
                            not_testable_reason=reason)
    return GapRegularityResult(k, measures, poly, stage1, mc, mp, rd, mn, d_c, el, null_result,
                               warnings + list(null_result.warnings), stage2_level=str(stage2))


# ============================================================================
# Plot bands at the centres level (for the "Perimeter arrangement" tab; pointwise, labelled so)
# ============================================================================

def centres_null_bands(k: int, length_nm: float, d_nm: float, *, n_replicates: int = DEFAULT_N_REPLICATES,
                       seed: int = 0, gap_bins_nm: Optional[Sequence[float]] = None, bin_nm: float = G_BIN_NM,
                       max_nm: float = G_MAX_NM) -> Dict[str, FloatArray]:
    """Pointwise 2.5 / 50 / 97.5 % bands of the gap histogram, g(s) and S(m) (m = 1..2K) under sequential
    inhibition with hard core ``d_nm`` (d_c for the contact null, d_m for the matched one) at the CENTRES level.
    Pointwise bands are for drawing only (Loosmore & Ford 2006); the pipeline-level bands need the replicates'
    arc positions (open question for the integration)."""
    rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(23,)))
    pts, jam = hard_core_arcs(rng, int(k), float(length_nm), float(d_nm), int(n_replicates))
    good = pts[~jam]
    return _bands_from_arcs([(row, float(length_nm)) for row in good], int(k), float(length_nm),
                            gap_bins_nm=gap_bins_nm, bin_nm=bin_nm, max_nm=max_nm)


def pipeline_null_bands(null_result: Any, k: int, length_nm: float, *, gap_bins_nm: Optional[Sequence[float]] = None,
                        bin_nm: float = G_BIN_NM, max_nm: float = G_MAX_NM) -> Dict[str, FloatArray]:
    """The same pointwise bands as :func:`centres_null_bands`, at the PIPELINE level: from the arc positions every
    replicate of a ``PipelineNullResult`` run with ``keep_arcs=True`` kept (each on its own membrane; open question
    G2). Empty bands (``n`` = 0) when the null kept no arcs. For drawing only, as every pointwise band."""
    rows = [(np.asarray(s, dtype=np.float64), float(L)) for s, L in getattr(null_result, "replicate_arcs", []) or []
            if s is not None and np.asarray(s).size >= 2 and float(L) > 0]
    return _bands_from_arcs(rows, int(k), float(length_nm), gap_bins_nm=gap_bins_nm, bin_nm=bin_nm, max_nm=max_nm)


def _bands_from_arcs(rows: Sequence[Tuple[FloatArray, float]], k: int, length_nm: float, *,
                     gap_bins_nm: Optional[Sequence[float]] = None, bin_nm: float = G_BIN_NM,
                     max_nm: float = G_MAX_NM) -> Dict[str, FloatArray]:
    edges: FloatArray = np.asarray(gap_bins_nm if gap_bins_nm is not None
                                   else np.linspace(0.0, 3.0 * float(length_nm) / max(int(k), 1), 31),
                                   dtype=np.float64)
    hists, gs, ss = [], [], []
    modes = np.arange(1, 2 * int(k) + 1)
    centres: FloatArray = np.zeros(0)
    # g(s) stops at P/2 of each row's own membrane: the bands use the bins every row has (the shortest membrane's)
    top = min([float(max_nm)] + [0.5 * float(L) for _r, L in rows])
    for row, length in rows:
        hists.append(np.histogram(arc_gaps(row, length), edges)[0].astype(np.float64))
        centres, g = pair_correlation_arc(row, length, bin_nm, top)
        gs.append(g)
        ss.append(structure_factor(row, length, modes))

    def band(rows: List[FloatArray]) -> FloatArray:
        if not rows:
            return np.full((3, 0), np.nan)
        return np.asarray(np.percentile(np.vstack(rows), [2.5, 50.0, 97.5], axis=0), dtype=np.float64)

    return {"gap_edges_nm": edges, "gap_hist_band": band(hists), "g_s_nm": centres, "g_band": band(gs),
            "modes": modes.astype(np.float64), "s_band": band(ss), "n": np.array([float(len(rows))])}


# ============================================================================
# The multi-method plug-in (prefix gap_)
# ============================================================================

def _mm() -> Any:
    import importlib
    return importlib.import_module("tools.multimethod")


MULTIMETHOD_ANALYSIS = "gap_measures"
_NULL_CACHE: Dict[Tuple[Any, ...], Any] = {}
_NULL_CACHE_MAX = 4                     # [ad hoc] a few axons x methods: a null is ~1 MB


def _cache_key(run: Any, context: Any) -> Tuple[Any, ...]:
    an = run.analysis
    return (str(getattr(context, "axon_id", "")), str(run.key), id(an), int(getattr(an, "n_locs_slab", 0) or 0),
            int(getattr(an, "n_clusters_kept", 0) or 0), int(getattr(context, "seed", 0)),
            int(getattr(context, "n_replicates", DEFAULT_N_REPLICATES)))


def null_for_run(run: Any, context: Any, *, readouts: Optional[Sequence[str]] = None) -> Any:
    """The stage-1 pipeline null of one method's run, re-running THAT method (same params, curation, contour).
    Reused when ``context.extras["pipeline_null_results"][run.key]`` holds it or when it was computed in this process
    for the same (axon, run, seed, B); so the integration can make ``pipeline_null`` and ``gap_measures`` share ONE
    null (D7) by passing the first one's result in the extras. Extras read: ``nena_nm``, ``null_closure`` ("n_locs"),
    ``null_c_label`` (14), ``null_r_max_nm`` (400)."""
    extras = dict(getattr(context, "extras", {}) or {})
    given = extras.get("pipeline_null_results")
    if isinstance(given, Mapping) and run.key in given:
        return given[run.key]
    key = _cache_key(run, context)
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    an = run.analysis
    base = pn.PipelineSpec.from_analysis(an)
    spec = pn.PipelineSpec.for_method(str(run.method), dict(run.params or {}), **{
        k: getattr(base, k) for k in ("slab_half_width_nm", "dbcv_threshold", "roi", "mahalanobis_threshold",
                                      "ellipse_mode", "sigma_cap_fraction", "all_starts", "main_peak_mode")})
    n_rep = int(getattr(context, "n_replicates", DEFAULT_N_REPLICATES))
    res = pn.run_pipeline_null(
        an, spec=spec, nena_nm=extras.get("nena_nm"), n_replicates=n_rep, seed=int(getattr(context, "seed", 0)),
        closure=str(extras.get("null_closure", "n_locs")), c_label=float(extras.get("null_c_label", pn.DEFAULT_C_LABEL)),
        r_max_nm=float(extras.get("null_r_max_nm", pn.R_MAX_NM)), allow_slow=bool(getattr(context, "allow_slow", False)),
        readouts=readouts,
        progress=lambda done, total: context.check_cancelled(f"gap measures: pipeline null, replicate {done} of {total}"))
    if len(_NULL_CACHE) >= _NULL_CACHE_MAX:
        _NULL_CACHE.pop(next(iter(_NULL_CACHE)))
    _NULL_CACHE[key] = res
    return res


def gap_measures_analysis(run: Any, context: Any) -> Dict[str, Any]:
    """The ``tools.multimethod`` plug-in: the gap measures and the two-stage reading of ONE method's run.

    Extras read (all optional): ``gap_null`` ("pipeline" default; "none" = the values only, no null, seconds),
    ``gap_stage2`` ("centres" default = "approx., centres"; "pipeline" for the batch, P-R87 c; "both"; "none"),
    ``pipeline_null_results`` (run key -> stage-1 result to reuse), and the null's ``nena_nm``, ``null_closure``,
    ``null_c_label``, ``null_r_max_nm``. A slow method without allow_slow raises the runner's SlowMethodRefused."""
    mm = _mm()
    an = run.analysis
    extras = dict(getattr(context, "extras", {}) or {})
    mode = str(extras.get("gap_stage2", "centres"))
    out: Dict[str, Any] = {}
    if str(extras.get("gap_null", "pipeline")) == "none":
        s, length, _m = axon_arcs(an) if an.perimeter is not None and an.n_clusters_kept >= 3 else (None, 0.0, None)
        gm = gap_measures(s if s is not None else np.zeros(0), length if length > 0 else 1.0)
        return {"n_gap_clusters": int(an.n_clusters_kept), "gap_cv": gm.cv, "gap_rho1": gm.rho1,
                "gap_excess_cv": gm.excess_cv, "gap_bragg_s": gm.bragg_s, "gap_bragg_m": gm.bragg_m,
                "gap_bragg_spacing_nm": gm.bragg_spacing_nm, "gap_s_low": gm.s_low}
    n_rep = int(getattr(context, "n_replicates", DEFAULT_N_REPLICATES))
    try:
        if an.n_clusters_kept >= MIN_CLUSTERS and an.perimeter is not None:
            nr = null_for_run(run, context)
        else:
            nr = None
        res = analyze_gap_regularity(an, null_result=nr, stage2=mode, seed=int(getattr(context, "seed", 0)),
                                     n_replicates=n_rep, n_matched=n_rep, nena_nm=extras.get("nena_nm"),
                                     allow_slow=bool(getattr(context, "allow_slow", False)),
                                     spec=None if nr is None else _spec_of_run(run),
                                     progress=lambda what, i, n: context.check_cancelled(f"gap measures: {what}"))
    except pn.SlowMethodRefused as exc:
        raise mm.SlowMethodRefused(mm.slow_refusal(str(run.method), 1 + n_rep) + f" ({exc})") from exc
    except pn.NullRefused:
        return {}
    d = res.export_columns()
    m = res.measures
    out.update({
        "n_gap_clusters": res.k, "gap_cv": m.cv, "gap_rho1": m.rho1, "gap_excess_cv": m.excess_cv,
        "gap_bragg_s": m.bragg_s, "gap_bragg_m": m.bragg_m, "gap_bragg_spacing_nm": m.bragg_spacing_nm,
        "gap_s_low": m.s_low, "gap_cv_null_mean": d.get("gap_cv_null_mean"), "gap_cv_z": d.get("gap_cv_z"),
        "gap_cv_p": d.get("gap_cv_p_mc"), "gap_excess_cv_p": d.get("gap_excess_cv_p_mc"),
        "gap_rho1_matched_p": d.get("gap_rho1_p_mc_matched"), "gap_bragg_matched_p": d.get("bragg_p_mc_matched"),
        "gap_rho1_z_matched": d.get("gap_rho1_z_matched"), "gap_bragg_z_matched": d.get("bragg_s_z_matched"),
        "gap_matched_d_nm": d.get("matched_null_d_nm"), "gap_contact_d_nm": res.d_contact_nm,
        "n_gap_clusters_elongated": d.get("n_clusters_elongated"), "gap_elongation_threshold": d.get("elongation_threshold"),
        "gap_regularity_verdict": res.reading.level,
        "gap_min_contact_verdict": MIN_FLAG_LEVELS[0] if res.reading.flag_min_below_contact else MIN_FLAG_LEVELS[1],
        "gap_matched_null_verdict": res.reading.stage2_kind,
    })
    return out


def _spec_of_run(run: Any) -> pn.PipelineSpec:
    base = pn.PipelineSpec.from_analysis(run.analysis)
    return pn.PipelineSpec.for_method(str(run.method), dict(run.params or {}), **{
        k: getattr(base, k) for k in ("slab_half_width_nm", "dbcv_threshold", "roi", "mahalanobis_threshold",
                                      "ellipse_mode", "sigma_cap_fraction", "all_starts", "main_peak_mode")})


def multimethod_readouts() -> List[Any]:
    """The ``tools.multimethod.ReadoutSpec`` list of the plug-in (names: prefix gap_, p in _p, verdicts in _verdict)."""
    R = _mm().ReadoutSpec
    org = "regularity plan PR 2 (two-stage rule, P-R81 b); pipeline null A9; p: Besag & Diggle 1977"
    return [
        R("n_gap_clusters", kind="count", description="K: kept clusters whose gaps are measured", origin=org),
        R("gap_cv", description="sd/mean of the cyclic arc gaps on the P-spline membrane", origin=org),
        R("gap_rho1", description="lag-1 correlation of consecutive gaps (-1/2 lattice, 0 renewal)", origin=org),
        R("gap_excess_cv", description="CV of the gaps minus the smallest gap (~1 exclusion)", origin=org),
        R("gap_bragg_s", description="Bragg peak S* of the periodogram, m in [K/2, 3K/2]", origin=org),
        R("gap_bragg_m", kind="count", description="mode m* of the Bragg peak", origin=org),
        R("gap_bragg_spacing_nm", description="a* = P / m*, the spacing the peak implies", origin=org),
        R("gap_s_low", description="mean S(m), m = 1..3 (renewal plateau ~CV^2; ~0 lattice)", origin=org),
        R("gap_cv_null_mean", description="mean gap CV of the pipeline null", origin=org),
        R("gap_cv_z", description="(gap CV - null mean) / null sd (the batch's currency)", origin=org),
        R("gap_cv_p", kind="p_value", tests="gap_cv", description="stage 1: two-sided Monte Carlo p of the gap CV "
          "against the pipeline null (P-R91 a)", origin=org),
        R("gap_excess_cv_p", kind="p_value", tests="gap_excess_cv", description="two-sided p of the excess-gap CV "
          "against the pipeline null (descriptive of the kind)", origin=org),
        R("gap_rho1_matched_p", kind="p_value", tests="gap_rho1", description="stage 2: lower-tail p of rho1 against "
          "the matched-exclusion null (empty when stage 2 did not run)", origin=org),
        R("gap_bragg_matched_p", kind="p_value", tests="gap_bragg_s", description="stage 2: upper-tail p of S* "
          "against the matched-exclusion null", origin=org),
        R("gap_rho1_z_matched", description="z of rho1 against the matched null (P-R83: animal mean < -2)", origin=org),
        R("gap_bragg_z_matched", description="z of S* against the matched null (P-R83: animal mean > 2)", origin=org),
        R("gap_matched_d_nm", description="d_m: hard core of the matched-exclusion null (effective exclusion; "
          "Diggle 1979)", origin=org),
        R("gap_contact_d_nm", description="d_c = 2 x median r_eff, the pipeline null's contact hard core", origin=org),
        R("n_gap_clusters_elongated", kind="count", description="clusters above the null's 95th percentile of "
          "tangential elongation (probable merged; P-R84 a)", origin=org),
        R("gap_elongation_threshold", description="that threshold (tangential / normal sd)", origin=org),
        R("gap_regularity_verdict", kind="verdict", levels=READING_LEVELS, description="the two-stage reading",
          origin=org),
        R("gap_min_contact_verdict", kind="verdict", levels=MIN_FLAG_LEVELS, description="minimum 1NN below the "
          "contact distance: split clusters or clusters of another ring", origin=org),
        R("gap_matched_null_verdict", kind="verdict", levels=STAGE2_KINDS, description="where stage 2 ran "
          "(pipeline = exact; centres_approx = approx., centres; not_run)", origin=org),
    ]


def register_multimethod(replace: bool = True) -> None:
    """Register ``gap_measures_analysis`` with ``tools.multimethod`` (done at import; idempotent)."""
    mm = _mm()
    mm.register_analysis(
        MULTIMETHOD_ANALYSIS, gap_measures_analysis, multimethod_readouts(), replace=replace,
        description="Gap measures and the two-stage regularity reading (regularity PR 2), per method.",
        cost_seconds=lambda run: (1 + int(DEFAULT_N_REPLICATES)) * pn.estimated_seconds_per_replicate(
            str(run.method), int(getattr(run.analysis, "n_locs_slab", 0) or 0)))


register_multimethod()
