# -*- coding: utf-8 -*-
"""
Axial (Z) periodicity of the membrane-associated periodic skeleton, and
automatic selection of the single-MPS-segment axial window.

Implements parameter 7 of Gazal et al. (2026) plus the axial slab
selection that every downstream parameter depends on:

  "For each ROI, the z-distribution was modeled using a Gaussian mixture
   model (GMM), with the optimal number of components (2-3) selected based
   on the Bayesian information criterion. Only dominant Gaussian
   components, defined by a minimum mixture weight threshold (>5% of the
   z-values), were retained and sorted by their axial position. The mean
   positions (mu) of these Gaussians were interpreted as the most probable
   axial locations of betaII-spectrin. When multiple components were
   detected within a ROI, the axial distances between consecutive Gaussian
   means (Delta-Z) were computed."

  "[...] pulling localizations from a 180 nm axial range centered around
   the main peak of the axial localization distribution."

Published in sciatic nerve: Delta-Z = 170 +/- 15 nm (pooled; the nature
of the +/- is not stated), consistent with the spectrin tetramer spacing
between actin rings. A spread of pooled spacings is not a tolerance, so a
region's Delta-Z is compared with it as a signed distance, never judged
(``delta_z_summary``).

Design notes
------------
* The whole loaded ROI is used, as-is. Per the user's workflow, the ROI
  that is loaded IS the ROI that is analysed -- no interactive
  sub-selection of regions inside the axon (unlike Fig. 2A of the paper,
  where sub-ROIs were picked by hand).
* GMM fitting is randomly initialised, so ``random_state`` is pinned and
  ``n_init`` raised: the same ROI must always yield the same Delta-Z, or
  the reported periodicity would drift between runs of the same data.
* SCI-6 (the axial mixture and the ring boundaries): the candidate
  component counts, the restarts and the seed are parameters; both
  readings of the main peak are kept; the single-ring window reports its
  expected foreign fraction F (``foreign_fraction``) instead of a test on
  near-equal weights; Delta-Z is compared with published periods as a
  signed distance in spread units (``delta_z_summary``); the valley is the
  antimode between the two adjacent modes and its depth is referred to
  those peaks; the Bayes boundary, the separation index and the shifts are
  reported per boundary; the posterior is that of the complete mixture.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr
from sklearn.mixture import GaussianMixture


# Paper constants
DEFAULT_COMPONENT_RANGE: Tuple[int, ...] = (2, 3)
# The component counts the program offers (z-gmm-model). BIC approximates
# the density rather than counting rings (Baudry et al. 2010), so the
# candidate set is a constraint, not an inference: 1 makes a single-ring
# ROI visible, 4 shows when the profile is not two or three Gaussians. 2 and
# 3, the published method's (Gazal et al. 2026), are the ones pre-selected
# (``DEFAULT_COMPONENT_RANGE``).
OFFERED_COMPONENT_COUNTS: Tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_MIN_WEIGHT = 0.05          # ">5% of the z-values"
DEFAULT_SLAB_HALF_WIDTH_NM = 90.0  # 180 nm window, +/-90 nm around the peak

# GMM determinism / stability (z-gmm-determinism). EM reaches a local
# maximum that depends on its start (k-means in scikit-learn): a pinned
# seed makes the result reproducible, and of k restarts all miss the best
# optimum with probability (1 - p)^k, below 3 % for k = 10 when each start
# finds it with probability p >= 0.3 (Dempster et al. 1977). A convention.
DEFAULT_RANDOM_STATE = 0
DEFAULT_N_INIT = 10
_RANDOM_STATE = DEFAULT_RANDOM_STATE
_N_INIT = DEFAULT_N_INIT

# How the centre of the single-ring window is read (main-peak-density,
# main-peak-weight): the mode of the fitted density, or the mean of the
# heaviest component.
MAIN_PEAK_MODES: Tuple[str, ...] = ("density", "weight")

# The expected share of the window's localizations that belong to
# neighbouring components (ambiguous-main-peak-warning): warned above this.
# A suggestion: a neighbouring ring's clusters start to reach min_samples
# localizations of their own inside the window once F x n_cluster >=
# min_samples, i.e. F >= 3-10 % for min_samples = 10 and clusters of
# 100-300 localizations. Editable 1-20 %.
FOREIGN_FRACTION_WARN = 0.05

# One level for the depth of a valley (C8): the analysis' "barely separated"
# warning and the rings window's colour. For two equal Gaussian rings a
# depth of 0.10 is a spacing of about 2.45 widths; any dip at all needs
# more than 2 widths (Behboodian 1970; Schilling et al. 2002).
VALLEY_DEPTH_LEVEL = 0.10
# Sampling of the valley search: 1-2 orders of magnitude below any axial
# precision, so it never limits a position.
VALLEY_GRID_STEP_NM = 0.5
# What a valley's depth is referred to (valley-relative-depth): the two
# peaks (modes) adjacent to it, or (as before SCI-6) the density at the two
# component means.
DEPTH_REFERENCES: Tuple[str, ...] = ("peaks", "means")
# Which mixture the posterior membership is computed on
# (valley-uses-all-components): the complete fitted mixture, as the valleys
# are, or only the dominant components with renormalised weights.
POSTERIOR_MODELS: Tuple[str, ...] = ("full", "dominant")
# Retention bound of a guard band (C5): with guards at both boundaries a
# ring keeps at least half of its localizations when g/2 <= h - 0.67 s_eff
# (Phi(0.67) - Phi(-0.67) = 0.50).
GUARD_RETENTION_Z = 0.67


@dataclass(frozen=True)
class ReferencePeriod:
    """One published MPS period the axial period is compared with
    (delta-z-paper-check): a descriptive distance, never a verdict."""

    label: str
    tissue: str
    axis: str
    period_nm: float
    spread_nm: Optional[float]     # None: no spread published
    spread_kind: str               # "SD", "SEM", "unstated" or "" (no spread)
    citation: str


# The editable reference table. Values as published; what the +/- is, when
# the publication says.
REFERENCE_PERIODS: Tuple[ReferencePeriod, ...] = (
    ReferencePeriod("Gazal 2026", "mouse sciatic nerve", "axial (z)", 170.0, 15.0, "unstated",
                    "Gazal et al. 2026"),
    ReferencePeriod("D'Este 2016", "mouse sciatic nerve", "lateral (STED, actin)", 186.0, 23.0, "SD",
                    "D'Este et al. 2016"),
    ReferencePeriod("Xu 2013", "cultured hippocampal neurons", "lateral (actin)", 182.0, 16.0, "SD",
                    "Xu et al. 2013"),
    ReferencePeriod("Unsain 2018", "cultured neurons", "lateral (spectrin)", 191.0, 28.0, "unstated",
                    "Unsain et al. 2018"),
    ReferencePeriod("Zhong 2014", "cultured neurons", "lateral (autocorrelation)", 190.0, None, "",
                    "Zhong et al. 2014"),
)


@dataclass
class ZPeriodicityResult:
    """Outcome of the GMM fit to the axial (Z) coordinate of one ROI."""

    n_components: int                      # components chosen by BIC
    means_nm: NDArray[np.float64]          # retained component means, sorted ascending
    weights: NDArray[np.float64]           # matching mixture weights
    sigmas_nm: NDArray[np.float64]         # matching component std deviations
    delta_z_nm: NDArray[np.float64]        # distances between consecutive means
    main_peak_nm: float                    # centre of the 180 nm analysis slab
    bic_by_n: dict                         # {n_components: BIC} for inspection
    n_discarded_components: int            # dropped for weight <= min_weight
    converged: bool
    warnings: List[str] = field(default_factory=list)

    # The complete fitted mixture, BEFORE the weight filter. Kept so the
    # model's own account of the axial profile can be re-evaluated at any z
    # without refitting -- which is what locating the valley between two
    # segments needs. The discarded low-weight components are part of that
    # density: where one of them sits between two rings, the boundary
    # between them is genuinely less well defined, and dropping it would
    # hide that.
    all_means_nm: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=float))
    all_weights: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=float))
    all_sigmas_nm: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=float))
    # Which of the complete mixture's components are dominant (same order as
    # ``all_means_nm``); empty when the complete mixture is not known.
    all_dominant: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=bool))
    # SCI-6: both readings of "the main peak" and the one used; the
    # candidate counts BIC chose among (``bic_by_n`` may hold more, reported
    # only); the fit's restarts and seed.
    peak_by_density_nm: float = float("nan")
    peak_by_weight_nm: float = float("nan")
    main_peak_mode: str = "density"
    candidate_counts: Tuple[int, ...] = DEFAULT_COMPONENT_RANGE
    n_init: int = DEFAULT_N_INIT
    random_state: int = DEFAULT_RANDOM_STATE

    @property
    def mean_delta_z_nm(self) -> Optional[float]:
        """Mean Delta-Z for this ROI, or None if fewer than 2 components
        survived the weight filter (a single peak yields no spacing)."""
        if self.delta_z_nm.size == 0:
            return None
        return float(np.mean(self.delta_z_nm))

    @property
    def delta_z_se_nm(self) -> Optional[float]:
        """Standard error of the mean Delta-Z: sd of the spacings / sqrt(m),
        with m >= 2 spacings; None otherwise (one spacing has no spread)."""
        d = np.asarray(self.delta_z_nm, dtype=float)
        if d.size < 2:
            return None
        return float(np.std(d, ddof=1) / np.sqrt(d.size))

    def mixture_density(
        self, z: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Probability density of the fitted mixture at ``z``, in nm^-1.

        Uses every fitted component, dominant or not (see ``all_means_nm``).
        Returns zeros when the mixture parameters are unavailable.
        """
        zz = np.asarray(z, dtype=float).ravel()
        m = np.asarray(self.all_means_nm, dtype=float).ravel()
        w = np.asarray(self.all_weights, dtype=float).ravel()
        s = np.asarray(self.all_sigmas_nm, dtype=float).ravel()
        if m.size == 0 or m.size != w.size or m.size != s.size:
            return np.zeros_like(zz)
        s = np.maximum(s, 1e-9)
        u = (zz[:, None] - m[None, :]) / s[None, :]
        density: NDArray[np.float64] = np.sum(
            (w / (s * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * u ** 2), axis=1)
        return density


def _fit_gmm(Z: NDArray[np.float64], n: int, n_init: int, random_state: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n,
        covariance_type="full",
        random_state=random_state,
        n_init=n_init,
    )
    gmm.fit(Z)
    return gmm


def fit_z_periodicity(
    z: NDArray[np.float64],
    component_range: Sequence[int] = DEFAULT_COMPONENT_RANGE,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    main_peak_mode: str = "density",
    *,
    half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM,
    n_init: int = DEFAULT_N_INIT,
    random_state: int = DEFAULT_RANDOM_STATE,
    report_counts: Sequence[int] = (),
) -> ZPeriodicityResult:
    """
    Fit a Gaussian mixture model to the Z coordinates of one ROI and
    extract the axial periodicity (Delta-Z) and the main axial peak.

    Model (z-gmm-model): each ring is one Gaussian in z whose variance is the
    ring's axial thickness plus the axial localization precision plus the
    smear of any residual tilt; the Gaussian shape is an assumption. Two
    components are two resolved rings only when their means are more than
    about two widths apart (Behboodian 1970; Amendola et al. 2020), so a
    Delta-Z is read together with the valley between its components
    (``density_valleys``). Delta-Z is in the length units of the
    measurement's z calibration.

    Parameters
    ----------
    z : 1-D array of Z coordinates in nm (all localizations of the ROI).
    component_range : candidate component counts BIC chooses among.
        Default (2, 3) per the paper; any of ``OFFERED_COMPONENT_COUNTS``.
    min_weight : mixture-weight threshold; components at or below this are
        discarded as non-dominant. Default 0.05 (">5%").
    main_peak_mode : how to locate the centre of the 180 nm slab.
        - "density" (default): the global maximum of the fitted mixture
          density, i.e. literally "the main peak of the axial localization
          distribution". Robust when two components overlap into one
          visible peak.
        - "weight": the mean of the highest-weight retained component.
        Both are computed and kept; this only selects which one is returned
        as ``main_peak_nm``.
    half_width_nm : the slab half-width in use. The two readings of the main
        peak are reported as disagreeing when they are more than this apart:
        windows centred on them would then share less than half their extent
        (main-peak-disagreement-warning).
    n_init, random_state : EM restarts and seed (z-gmm-determinism).
    report_counts : component counts whose BIC is also computed and reported
        in ``bic_by_n`` without being candidates (costs one fit each).

    Returns
    -------
    ZPeriodicityResult

    Raises
    ------
    ValueError if fewer than 2 finite Z values are supplied.
    """
    z = np.asarray(z, dtype=float).ravel()
    z = z[np.isfinite(z)]
    if z.size < 2:
        raise ValueError(
            f"Need at least 2 finite Z values to fit a GMM, got {z.size}."
        )
    if main_peak_mode not in MAIN_PEAK_MODES:
        raise ValueError(
            f"main_peak_mode must be 'density' or 'weight', got {main_peak_mode!r}"
        )

    warnings_: List[str] = []
    Z = z.reshape(-1, 1)

    # --- BIC model selection over the candidate component counts --------
    # Guard against asking for more components than we have samples for;
    # sklearn raises in that case, and a ROI that sparse cannot support a
    # multi-component axial model anyway.
    usable = [int(n) for n in component_range if n <= z.size]
    if not usable:
        usable = [1]
        warnings_.append(
            f"Only {z.size} localizations: falling back to a 1-component fit."
        )

    bic_by_n = {}
    best_gmm = None
    best_bic = np.inf
    best_n = usable[0]

    for n in usable:
        gmm = _fit_gmm(Z, n, int(n_init), int(random_state))
        bic = float(gmm.bic(Z))
        bic_by_n[int(n)] = bic
        if bic < best_bic:
            best_bic, best_gmm, best_n = bic, gmm, int(n)
    for n in report_counts:
        n = int(n)
        if n in bic_by_n or n > z.size or n < 1:
            continue
        bic_by_n[n] = float(_fit_gmm(Z, n, int(n_init), int(random_state)).bic(Z))

    assert best_gmm is not None
    if not best_gmm.converged_:
        warnings_.append(
            f"GMM with {best_n} components did not converge; "
            f"Delta-Z values may be unreliable."
        )

    # --- Retain dominant components, sorted by axial position -----------
    means_all = best_gmm.means_.ravel()
    weights_all = best_gmm.weights_.ravel()
    sigmas_all = np.sqrt(best_gmm.covariances_.reshape(len(means_all)))

    dominant = weights_all > min_weight
    n_discarded = int(np.sum(~dominant))

    if not np.any(dominant):
        # Pathological: every component below threshold (only possible with
        # many components). Keep the single strongest so downstream code
        # still has a peak to centre the slab on.
        dominant = weights_all == weights_all.max()
        warnings_.append(
            "No component exceeded the weight threshold; kept the strongest one."
        )

    means = means_all[dominant]
    weights = weights_all[dominant]
    sigmas = sigmas_all[dominant]

    order = np.argsort(means)
    means, weights, sigmas = means[order], weights[order], sigmas[order]

    # --- Delta-Z between consecutive retained means ---------------------
    delta_z = np.diff(means) if means.size >= 2 else np.array([])
    if delta_z.size == 0:
        warnings_.append(
            "Only one dominant axial component: no Delta-Z could be computed "
            "for this ROI."
        )

    # --- Main peak: centre of the 180 nm analysis slab ------------------
    peak_by_weight = float(means[int(np.argmax(weights))])

    grid = np.linspace(z.min(), z.max(), 4096).reshape(-1, 1)
    density = np.exp(best_gmm.score_samples(grid))
    peak_by_density = float(grid[int(np.argmax(density)), 0])

    main_peak = peak_by_density if main_peak_mode == "density" else peak_by_weight

    # If the two readings are further apart than the half-width in use,
    # windows centred on them share less than half their extent: the choice
    # decides which MPS segment is analysed, so surface it rather than hide
    # it, with the shared fraction (2h - d) / (2h) a reader can judge.
    h = float(half_width_nm)
    d = abs(peak_by_density - peak_by_weight)
    if h > 0 and d > h:
        share = max(0.0, (2.0 * h - d) / (2.0 * h))
        warnings_.append(
            f"Main-peak estimates disagree by {d:.0f} nm "
            f"(density {peak_by_density:.0f} nm vs weight {peak_by_weight:.0f} nm), "
            f"more than the slab half-width of {h:g} nm: windows centred on the "
            f"two would share {share:.0%} of their extent, so the axial "
            f"distribution has more than one comparable peak and the choice "
            f"decides which MPS segment is analysed."
        )

    return ZPeriodicityResult(
        n_components=best_n,
        means_nm=means,
        weights=weights,
        sigmas_nm=sigmas,
        delta_z_nm=delta_z,
        main_peak_nm=main_peak,
        bic_by_n=bic_by_n,
        n_discarded_components=n_discarded,
        converged=bool(best_gmm.converged_),
        warnings=warnings_,
        all_means_nm=means_all,
        all_weights=weights_all,
        all_sigmas_nm=sigmas_all,
        all_dominant=np.asarray(dominant, dtype=bool),
        peak_by_density_nm=peak_by_density,
        peak_by_weight_nm=peak_by_weight,
        main_peak_mode=main_peak_mode,
        candidate_counts=tuple(int(n) for n in usable),
        n_init=int(n_init),
        random_state=int(random_state),
    )


# ============================================================================
# The single-ring window: its foreign fraction
# ============================================================================

def _components(result: ZPeriodicityResult) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                                                     NDArray[np.float64], NDArray[np.bool_]]:
    """(means, sigmas, weights, dominant) of the complete mixture when it is
    known, else of the dominant components (all dominant)."""
    m = np.asarray(result.all_means_nm, dtype=float).ravel()
    s = np.asarray(result.all_sigmas_nm, dtype=float).ravel()
    w = np.asarray(result.all_weights, dtype=float).ravel()
    if m.size and m.size == s.size == w.size:
        dom = np.asarray(result.all_dominant, dtype=bool).ravel()
        if dom.size != m.size:
            dom = w > DEFAULT_MIN_WEIGHT
        return m, s, w, dom
    m = np.asarray(result.means_nm, dtype=float).ravel()
    return (m, np.asarray(result.sigmas_nm, dtype=float).ravel(),
            np.asarray(result.weights, dtype=float).ravel(), np.ones(m.size, dtype=bool))


@dataclass
class ForeignFraction:
    """The expected share of a window's localizations contributed by
    components other than the window's own ring."""

    fraction: Optional[float]      # None when the mixture puts no mass in the window
    ring_mean_nm: Optional[float]  # mean of the component the window holds most of
    n_dominant: int                # dominant components of the ROI (the "comparable rings" note)
    lo_nm: float
    hi_nm: float


def foreign_fraction(result: ZPeriodicityResult, lo_nm: float, hi_nm: float) -> ForeignFraction:
    """
    F = sum over components k other than the window's ring of
    w_k [Phi((hi - mu_k)/sigma_k) - Phi((lo - mu_k)/sigma_k)], divided by the
    same sum over every component of the fitted mixture
    (ambiguous-main-peak-warning). The window's ring is the component that
    puts the most mass inside [lo, hi].

    Replaces the test on near-equal weights: equal weights are the expected
    state of two equally labelled rings and say nothing about overlap; what
    matters downstream is how much of the window comes from neighbouring
    rings.
    """
    m, s, w, dom = _components(result)
    n_dom = int(np.asarray(result.means_nm).size)
    lo, hi = float(min(lo_nm, hi_nm)), float(max(lo_nm, hi_nm))
    if m.size == 0:
        return ForeignFraction(None, None, n_dom, lo, hi)
    s = np.maximum(s, 1e-9)
    mass = w * (ndtr((hi - m) / s) - ndtr((lo - m) / s))
    total = float(np.sum(mass))
    if not np.isfinite(total) or total <= 0.0:
        return ForeignFraction(None, None, n_dom, lo, hi)
    k = int(np.argmax(mass))
    return ForeignFraction(float((total - float(mass[k])) / total), float(m[k]), n_dom, lo, hi)


# ============================================================================
# Delta-Z against published periods
# ============================================================================

@dataclass
class ReferenceDistance:
    reference: ReferencePeriod
    difference_nm: float               # mean Delta-Z - P_ref
    spread_units: Optional[float]      # difference / published spread; None without a spread

    def words(self) -> str:
        r = self.reference
        what = f"{r.label} ({r.tissue}, {r.axis}"
        what += (f"; +/- {r.spread_kind}" if r.spread_kind in ("SD", "SEM")
                 else "; nature of the +/- not stated" if r.spread_kind == "unstated" else "")
        what += ")"
        if self.spread_units is None:
            return f"{self.difference_nm:+.1f} nm from {r.period_nm:g} nm of {what}"
        return (f"{self.spread_units:+.2f} spread units from {r.period_nm:g} +/- {r.spread_nm:g} nm of "
                f"{what}")


@dataclass
class DeltaZSummary:
    """A region's mean Delta-Z with its own standard error and its signed
    distance from each published period (delta-z-paper-check). Descriptive:
    a published spread of pooled spacings is not a tolerance, so no
    pass/fail label is attached."""

    mean_nm: Optional[float]
    se_nm: Optional[float]
    n_spacings: int
    distances: List[ReferenceDistance] = field(default_factory=list)

    def lines(self) -> List[str]:
        return [d.words() for d in self.distances]


def delta_z_summary(
    delta_z_nm: Sequence[float],
    references: Sequence[ReferencePeriod] = REFERENCE_PERIODS,
) -> DeltaZSummary:
    """Mean Delta-Z, its SE (sd / sqrt(m), m >= 2 spacings) and, for every
    reference, (mean - P_ref) / s_ref (or the plain difference when the
    reference publishes no spread)."""
    d = np.asarray(list(delta_z_nm), dtype=float).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return DeltaZSummary(None, None, 0, [])
    mean = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(d.size)) if d.size >= 2 else None
    out: List[ReferenceDistance] = []
    for r in references:
        diff = mean - float(r.period_nm)
        units = (None if r.spread_nm is None or not (float(r.spread_nm) > 0)
                 else diff / float(r.spread_nm))
        out.append(ReferenceDistance(r, diff, units))
    return DeltaZSummary(mean, se, int(d.size), out)


# ============================================================================
# Boundaries between consecutive components
# ============================================================================

@dataclass
class ValleyResult:
    """Boundaries between consecutive MPS segments, and how well the axial
    density actually separates them."""

    positions_nm: NDArray[np.float64]   # the CUT of each consecutive pair of means (valley mode)
    is_true_valley: NDArray[np.bool_]   # True where the two components are two modes (an antimode exists)
    relative_depth: NDArray[np.float64]  # 0 = no dip at all, 1 = density -> 0
    midpoints_nm: NDArray[np.float64]
    warnings: List[str] = field(default_factory=list)
    # SCI-6. The Bayes boundary of each pair (where the two neighbouring
    # components are equally probable; the midpoint where it has no root
    # between the means: ``bayes_found`` False), the separation index
    # d = |mu_b - mu_a| / sqrt(sigma_a^2 + sigma_b^2), the two modes the
    # depth is referred to (NaN without a valley), and +1 where the upper
    # component of the pair is the lighter one (-1 otherwise): the side a
    # shift "toward the weaker ring" points to.
    bayes_positions_nm: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=float))
    bayes_found: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=bool))
    separation_d: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=float))
    mode_lo_nm: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=float))
    mode_hi_nm: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=float))
    weaker_side: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=float))
    depth_reference: str = "peaks"
    # The antimode itself (NaN where there is none). It is the cut unless the
    # dip lies above the density at one of the two component means: there the
    # density is nearly flat between the means, the antimode is ill-determined
    # (it moved by 35 nm between simulated replicates of one axon) and the cut
    # stays at the midpoint (``cut_at_antimode`` False).
    antimode_nm: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=float))
    cut_at_antimode: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([], dtype=bool))

    @property
    def n_boundaries(self) -> int:
        return int(self.positions_nm.size)

    @property
    def shift_from_midpoint_nm(self) -> NDArray[np.float64]:
        return np.abs(self.positions_nm - self.midpoints_nm)

    def _valley_position(self) -> NDArray[np.float64]:
        """The antimode where there is one, else the cut (the midpoint)."""
        am = np.asarray(self.antimode_nm, dtype=float)
        if am.size != self.positions_nm.size:
            return np.asarray(self.positions_nm, dtype=float)
        return np.where(np.isfinite(am), am, self.positions_nm)

    def _toward_weaker(self, other: NDArray[np.float64]) -> NDArray[np.float64]:
        side = np.asarray(self.weaker_side, dtype=float)
        if side.size != self.positions_nm.size:
            side = np.ones(self.positions_nm.size)
        return np.asarray((self._valley_position() - np.asarray(other, dtype=float)) * side, dtype=float)

    @property
    def shift_toward_weaker_from_midpoint_nm(self) -> NDArray[np.float64]:
        """The valley (antimode) minus the midpoint, positive toward the
        lighter ring (valley-shift-warning: reported, never judged)."""
        return self._toward_weaker(self.midpoints_nm)

    @property
    def shift_toward_weaker_from_bayes_nm(self) -> NDArray[np.float64]:
        """The valley (antimode) minus the Bayes boundary, positive toward the
        lighter ring (NaN where the Bayes boundary was not found)."""
        if np.asarray(self.bayes_positions_nm).size != self.positions_nm.size:
            return np.full(self.positions_nm.size, np.nan)
        out = self._toward_weaker(self.bayes_positions_nm)
        found = np.asarray(self.bayes_found, dtype=bool)
        if found.size == out.size:
            out = np.where(found, out, np.nan)
        return out


def bayes_boundary(mu_a: float, sigma_a: float, w_a: float,
                   mu_b: float, sigma_b: float, w_b: float) -> Optional[float]:
    """
    The z between mu_a < mu_b where w_a f_a(z) = w_b f_b(z): equal posterior
    membership of the two components, the minimum-misassignment cut of a
    fitted mixture (Kittler & Illingworth 1986; McLachlan & Peel 2000).
    For equal widths z_B = m + sigma^2 ln(w_a / w_b) / (mu_b - mu_a). None
    when no root lies between the two means (one component outweighs the
    other everywhere between them).
    """
    sa, sb = max(float(sigma_a), 1e-9), max(float(sigma_b), 1e-9)
    if not (w_a > 0 and w_b > 0 and mu_b > mu_a):
        return None
    ca = math.log(w_a) - math.log(sa)
    cb = math.log(w_b) - math.log(sb)
    # g(z) = ca - (z - mu_a)^2 / (2 sa^2) - cb + (z - mu_b)^2 / (2 sb^2) = A z^2 + B z + C
    A = 1.0 / (2.0 * sb * sb) - 1.0 / (2.0 * sa * sa)
    B = mu_a / (sa * sa) - mu_b / (sb * sb)
    C = mu_b * mu_b / (2.0 * sb * sb) - mu_a * mu_a / (2.0 * sa * sa) + ca - cb
    roots: List[float] = []
    if abs(A) < 1e-15 * max(abs(B), 1e-300):
        if B != 0:
            roots = [-C / B]
    else:
        disc = B * B - 4.0 * A * C
        if disc >= 0:
            q = -0.5 * (B + math.copysign(math.sqrt(disc), B))
            roots = [q / A] + ([C / q] if q != 0 else [])
    inside = [r for r in roots if mu_a <= r <= mu_b and math.isfinite(r)]
    if not inside:
        return None
    mid = 0.5 * (mu_a + mu_b)
    return float(min(inside, key=lambda r: abs(r - mid)))


def density_valleys(
    result: ZPeriodicityResult,
    grid_step_nm: float = VALLEY_GRID_STEP_NM,
    *,
    depth_reference: str = "peaks",
    shallow_level: float = VALLEY_DEPTH_LEVEL,
    shift_warning_nm: Optional[float] = None,
) -> ValleyResult:
    """
    Locate the axial density minimum between each pair of consecutive
    dominant components -- the natural boundary between two MPS segments.

    Splitting two overlapping slabs at the midpoint between their centres
    assumes the density is symmetric between them. It usually is not: the
    two rings differ in weight and width, so the lowest point of the axial
    profile sits off-centre, and a midpoint cut assigns localizations of
    the stronger ring to the weaker one. The minimum of the fitted mixture
    density is where the two rings genuinely separate.

    The valley is the ANTIMODE (density-valleys): a local minimum of the
    fitted density, found by the sign change of its slope on the grid,
    between the two modes adjacent to the boundary. A two-component normal
    mixture has at most three critical points (Behboodian 1970; Amendola et
    al. 2020), so when it is bimodal there is exactly one antimode between
    its modes; a minimum searched as the lowest point between the MEANS
    lands on an endpoint near the resolution limit although the density is
    bimodal (equal components are bimodal above 2 sigma, but their density
    at the means drops below the dip only above ~2.2 sigma). With the
    antimode, "no valley" means exactly "the two components are not two
    modes"; the boundary then falls back to the midpoint with depth 0.
    Several local minima (a low-weight component between two rings) give
    the lowest one.

    The depth (valley-relative-depth) is 1 - f(antimode) / min(f(mode_a),
    f(mode_b)), referred to the two PEAKS adjacent to the valley: 0 = no
    dip, 1 = the density vanishes between the rings. ``depth_reference =
    "means"`` refers it to the density at the two component means instead,
    which under-reads shallow valleys (the program's index before SCI-6).
    Separation itself is ``separation_d`` = |mu_b - mu_a| / sqrt(sigma_a^2
    + sigma_b^2).

    Read ``relative_depth`` before treating two segments as distinct rings:
    with components broad against their spacing it is 0, and the two
    "segments" are two halves of one axial distribution.

    Where the cut goes (``positions_nm``): at the antimode, unless the dip
    lies above the density at one of the two component means. There the two
    components are two modes only just, the density is nearly flat across
    tens of nanometres and the lowest point wanders with the sampling (by
    35 nm between simulated replicates of one axon, in either direction), so
    the cut stays at the midpoint and a warning says so; the antimode is
    still reported (``antimode_nm``) with its small positive depth. This is
    exactly where the program's earlier search (the lowest point between the
    means) cut, so no cut moves; only the depths and the resolved / not
    resolved reading change.

    The density is the complete fitted mixture, low-weight components
    included (valley-uses-all-components): the fitted density is their sum,
    and a minor component between two rings makes their boundary shallower
    rather than being hidden.

    Parameters
    ----------
    result : a fit from ``fit_z_periodicity``.
    grid_step_nm : sampling of the search. 0.5 nm is one to two orders of
        magnitude below any axial localization precision and the
        periodicity (~170 nm), so it cannot limit the result.
    depth_reference : "peaks" (default) or "means".
    shallow_level : boundaries with a valley shallower than this are
        flagged as barely separated (``VALLEY_DEPTH_LEVEL``, 0.10: one level
        with the rings window, C8).
    shift_warning_nm : an optional threshold on the shift of the valley
        from the midpoint; None (default) applies none. A shift is the
        normal state of rings that differ in weight or width, so it is
        reported (``shift_toward_weaker_from_midpoint_nm``), not judged.

    Returns
    -------
    ValleyResult -- one entry per consecutive pair of dominant means, in
    ascending axial order, so entry ``i`` separates component ``i`` from
    component ``i + 1``. Empty arrays if fewer than two dominant
    components.
    """
    if depth_reference not in DEPTH_REFERENCES:
        raise ValueError(f"depth_reference must be one of {DEPTH_REFERENCES}, got {depth_reference!r}")
    means = np.asarray(result.means_nm, dtype=float).ravel()
    sig = np.asarray(result.sigmas_nm, dtype=float).ravel()
    wts = np.asarray(result.weights, dtype=float).ravel()
    if means.size < 2:
        empty_f = np.array([], dtype=float)
        return ValleyResult(
            positions_nm=empty_f, is_true_valley=np.array([], dtype=bool),
            relative_depth=empty_f, midpoints_nm=empty_f,
            depth_reference=depth_reference)

    midpoints = (means[:-1] + means[1:]) / 2.0
    n_b = midpoints.size
    if sig.size == means.size and wts.size == means.size:
        sep = np.abs(np.diff(means)) / np.sqrt(sig[:-1] ** 2 + sig[1:] ** 2)
        side = np.where(wts[1:] < wts[:-1], 1.0, -1.0)
        bayes_list = [bayes_boundary(means[i], sig[i], wts[i], means[i + 1], sig[i + 1], wts[i + 1])
                      for i in range(n_b)]
    else:
        sep = np.full(n_b, np.nan)
        side = np.ones(n_b)
        bayes_list = [None] * n_b
    bayes_found = np.array([b is not None for b in bayes_list], dtype=bool)
    bayes = np.array([midpoints[i] if b is None else b for i, b in enumerate(bayes_list)], dtype=float)

    if np.asarray(result.all_means_nm).size == 0:
        return ValleyResult(
            positions_nm=midpoints,
            is_true_valley=np.zeros(n_b, dtype=bool),
            relative_depth=np.zeros(n_b, dtype=float),
            midpoints_nm=midpoints,
            warnings=["Mixture parameters unavailable, so segment boundaries "
                      "fell back to the midpoints between component means."],
            bayes_positions_nm=bayes, bayes_found=bayes_found, separation_d=sep,
            mode_lo_nm=np.full(n_b, np.nan), mode_hi_nm=np.full(n_b, np.nan),
            weaker_side=side, depth_reference=depth_reference)

    positions: List[float] = []
    true_valley: List[bool] = []
    depths: List[float] = []
    modes_lo: List[float] = []
    modes_hi: List[float] = []
    antimodes: List[float] = []
    at_antimode: List[bool] = []

    for i, (lo, hi) in enumerate(zip(means[:-1], means[1:])):
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            positions.append(float(midpoints[i]))
            true_valley.append(False)
            depths.append(0.0)
            modes_lo.append(float("nan"))
            modes_hi.append(float("nan"))
            antimodes.append(float("nan"))
            at_antimode.append(False)
            continue

        n = int(np.clip(np.ceil(span / max(grid_step_nm, 1e-6)), 64, 20_000))
        # The grid includes the two means as its ends, so that the slope at
        # the first and last interior points is known; the valley is only
        # ever an interior point.
        grid = np.linspace(lo, hi, n + 2)
        dens = result.mixture_density(grid)
        inner = np.arange(1, grid.size - 1)
        is_min = (dens[inner] < dens[inner - 1]) & (dens[inner] <= dens[inner + 1])
        minima = inner[is_min]
        if minima.size == 0:
            # No local minimum between the means: the density is monotone or
            # unimodal across the interval, the two components are not two
            # modes, and the midpoint is the honest fallback.
            positions.append(float(midpoints[i]))
            true_valley.append(False)
            depths.append(0.0)
            modes_lo.append(float("nan"))
            modes_hi.append(float("nan"))
            antimodes.append(float("nan"))
            at_antimode.append(False)
            continue
        k = int(minima[int(np.argmin(dens[minima]))])
        v = float(grid[k])
        f_v = float(dens[k])
        # The two peaks adjacent to the valley: the highest density on each
        # side, searched one component width beyond the means (a mode is
        # displaced from its mean, inward for two components and possibly
        # outward when a third component pulls it).
        s_lo = float(sig[i]) if sig.size == means.size else 0.0
        s_hi = float(sig[i + 1]) if sig.size == means.size else 0.0
        left = np.linspace(lo - s_lo, v, max(n // 2, 64))
        right = np.linspace(v, hi + s_hi, max(n // 2, 64))
        d_left = result.mixture_density(left)
        d_right = result.mixture_density(right)
        j_l = int(np.argmax(d_left))
        j_r = int(np.argmax(d_right))
        modes_lo.append(float(left[j_l]))
        modes_hi.append(float(right[j_r]))
        if depth_reference == "peaks":
            ref = float(min(d_left[j_l], d_right[j_r]))
        else:
            ref = float(min(dens[0], dens[-1]))
        depth = 0.0 if ref <= 0 else float(np.clip(1.0 - f_v / ref, 0.0, 1.0))
        # The cut: the antimode where the dip lies below the density at both
        # means (a well-determined minimum), the midpoint otherwise.
        below_means = f_v < float(min(dens[0], dens[-1]))
        positions.append(v if below_means else float(midpoints[i]))
        true_valley.append(True)
        depths.append(depth)
        antimodes.append(v)
        at_antimode.append(bool(below_means))

    pos = np.asarray(positions, dtype=float)
    tv = np.asarray(true_valley, dtype=bool)
    dp = np.asarray(depths, dtype=float)
    am = np.asarray(antimodes, dtype=float)
    cut_am = np.asarray(at_antimode, dtype=bool)
    warnings_: List[str] = []

    n_flat = int(np.count_nonzero(~tv))
    if n_flat:
        warnings_.append(
            f"{n_flat} of {pos.size} segment boundary(ies) had no interior "
            f"density minimum: those components are too broad relative to "
            f"their spacing to be resolved as separate rings, so the boundary "
            f"fell back to the midpoint between means. Segments either side of "
            f"such a boundary are two halves of one unresolved axial "
            f"distribution, not two rings."
        )

    n_flat_cut = int(np.count_nonzero(tv & ~cut_am))
    if n_flat_cut:
        warnings_.append(
            f"{n_flat_cut} boundary(ies) are two modes only just: the dip lies above the density at a "
            f"component mean, so its position is ill-determined and the cut stays at the midpoint between "
            f"the means."
        )

    shallow = int(np.count_nonzero(tv & (dp < float(shallow_level))))
    if shallow:
        warnings_.append(
            f"{shallow} boundary(ies) dip less than {float(shallow_level):.0%} below the lower "
            f"adjacent peak: the two segments are barely separated axially "
            f"(for two equal Gaussian rings a depth of 0.10 is a spacing of "
            f"about 2.45 widths)."
        )

    result_v = ValleyResult(
        positions_nm=pos, is_true_valley=tv, relative_depth=dp,
        midpoints_nm=midpoints, warnings=warnings_,
        bayes_positions_nm=bayes, bayes_found=bayes_found, separation_d=sep,
        mode_lo_nm=np.asarray(modes_lo, dtype=float), mode_hi_nm=np.asarray(modes_hi, dtype=float),
        weaker_side=side, depth_reference=depth_reference, antimode_nm=am, cut_at_antimode=cut_am)

    if shift_warning_nm is not None:
        shift = np.abs(result_v.shift_toward_weaker_from_midpoint_nm)
        if shift.size and float(np.max(shift)) > float(shift_warning_nm):
            warnings_.append(
                f"Valley boundaries sit up to {np.max(shift):.0f} nm away from the "
                f"midpoints between means (threshold {float(shift_warning_nm):g} nm, set by you)."
            )
    return result_v


# ============================================================================
# Posterior membership, and the closed-form trade-offs of a cut
# ============================================================================

def component_posterior(
    z_nm: NDArray[np.float64],
    result: ZPeriodicityResult,
    model: str = "full",
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    The maximum posterior membership of each localization and the dominant
    component (ring, in ascending axial order) it belongs to most probably.

    ``model = "full"`` (default, valley-uses-all-components) computes it on
    the complete fitted mixture, as the valleys are: a localization whose
    most probable component is a discarded low-weight one belongs to no
    ring (index -1). ``"dominant"`` uses only the dominant components with
    renormalised weights (the program's posterior before SCI-6). A
    non-finite z gets posterior NaN and index -1. Computed in log space,
    row-centred, so a localization far from every component does not
    underflow to 0/0.
    """
    if model not in POSTERIOR_MODELS:
        raise ValueError(f"model must be one of {POSTERIOR_MODELS}, got {model!r}")
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    post = np.full(z.shape, np.nan)
    index = np.full(z.shape, -1, dtype=np.int64)
    dom_means = np.asarray(result.means_nm, dtype=float).ravel()
    if model == "full":
        m, s, w, dom = _components(result)
    else:
        m = dom_means
        s = np.asarray(result.sigmas_nm, dtype=float).ravel()
        w = np.asarray(result.weights, dtype=float).ravel()
        dom = np.ones(m.size, dtype=bool)
    if m.size == 0 or m.size != s.size or m.size != w.size:
        return post, index
    s = np.maximum(s, 1e-9)
    w = w / w.sum()
    finite = np.isfinite(z)
    if not finite.any():
        return post, index
    zf = z[finite]
    with np.errstate(divide="ignore"):
        logw = np.log(w)
    logp = logw[None, :] - np.log(s)[None, :] - 0.5 * ((zf[:, None] - m[None, :]) / s[None, :]) ** 2
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p /= p.sum(axis=1, keepdims=True)
    j = np.argmax(p, axis=1)
    post[finite] = p[np.arange(zf.size), j]
    # The dominant component's rank along z (the ring index of the sorted
    # dominant means), or -1 for a discarded component.
    rank = np.full(m.size, -1, dtype=np.int64)
    dm = m[dom]
    order = np.argsort(dm)
    ranks_of_dom = np.empty(dm.size, dtype=np.int64)
    ranks_of_dom[order] = np.arange(dm.size)
    rank[np.flatnonzero(dom)] = ranks_of_dom
    index[finite] = rank[j]
    return post, index


@dataclass(frozen=True)
class PosteriorTradeoff:
    """What a posterior threshold p costs and buys at one boundary between
    two equal rings of width sigma at spacing P (posterior-min; Chow 1970)."""

    half_width_nm: float          # (sigma^2 / P) ln(p / (1 - p)) on each side of the boundary
    rejected_fraction: float      # of a two-ring stack (one boundary)
    error_among_accepted: float   # misassigned share of the accepted localizations


def posterior_tradeoff(sigma_nm: float, period_nm: float, p: float) -> PosteriorTradeoff:
    """Closed form for two equal Gaussian rings: the posterior of the nearer
    ring at distance x from the boundary is 1 / (1 + exp(-P x / sigma^2)), so
    the rejected zone has half-width x* = (sigma^2 / P) ln(p / (1 - p))."""
    s, P, p = float(sigma_nm), float(period_nm), float(p)
    if not (s > 0 and P > 0 and 0.0 < p < 1.0):
        raise ValueError(f"posterior_tradeoff needs sigma > 0, P > 0 and 0 < p < 1, got {s}, {P}, {p}")
    x = max((s * s / P) * math.log(p / (1.0 - p)), 0.0)
    h = 0.5 * P
    rejected = float(ndtr((h + x) / s) - ndtr((h - x) / s))
    wrong = float(1.0 - ndtr((h + x) / s))
    accepted = 1.0 - rejected
    return PosteriorTradeoff(x, rejected, wrong / accepted if accepted > 0 else float("nan"))


@dataclass(frozen=True)
class GuardTradeoff:
    """What a guard band of total width g costs and buys (guard-nm-0, C5)."""

    guard_nm: float
    lost_fraction: float          # of each ring, at one boundary
    leak_fraction: float          # from the neighbour across that boundary
    within_retention_bound: bool  # g/2 <= h - 0.67 s_eff


def guard_tradeoff(s_eff_nm: float, half_spacing_nm: float, guard_nm: float) -> GuardTradeoff:
    """Normal-tail arithmetic at a boundary h from the ring centre for rings
    of effective axial width s_eff: lost = Phi((h + g/2)/s) - Phi((h - g/2)/s),
    leak = 1 - Phi((h + g/2)/s)."""
    s, h, g = float(s_eff_nm), float(half_spacing_nm), max(float(guard_nm), 0.0)
    if not (s > 0 and h > 0):
        raise ValueError(f"guard_tradeoff needs s_eff > 0 and h > 0, got {s}, {h}")
    lost = float(ndtr((h + g / 2.0) / s) - ndtr((h - g / 2.0) / s))
    leak = float(1.0 - ndtr((h + g / 2.0) / s))
    return GuardTradeoff(g, lost, leak, g / 2.0 <= h - GUARD_RETENTION_Z * s + 1e-12)


def guard_retention_cap_nm(s_eff_nm: float, half_spacing_nm: float) -> float:
    """The widest total guard that keeps at least half of a ring guarded at
    both boundaries: g* = 2 (h - 0.67 s_eff), 0 when h <= 0.67 s_eff."""
    return float(max(2.0 * (float(half_spacing_nm) - GUARD_RETENTION_Z * float(s_eff_nm)), 0.0))


def effective_width_nm(result: ZPeriodicityResult) -> Optional[float]:
    """The rings' effective axial width s_eff as the fitted mixture measures
    it: the median width of the dominant components (each is the ring's
    thickness, the axial precision and any residual tilt in quadrature).
    None without a dominant component."""
    s = np.asarray(result.sigmas_nm, dtype=float).ravel()
    s = s[np.isfinite(s) & (s > 0)]
    return float(np.median(s)) if s.size else None


def select_mps_slab(
    z: NDArray[np.float64],
    main_peak_nm: float,
    half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM,
) -> NDArray[np.bool_]:
    """
    Boolean mask selecting the localizations of a single MPS segment: the
    180 nm axial window centred on the main peak.

    Replaces the manual ``zmin``/``zmax`` text fields of the GUI.

    Parameters
    ----------
    z : Z coordinates in nm.
    main_peak_nm : slab centre, from ``fit_z_periodicity``.
    half_width_nm : default 90 nm (so a 180 nm total window).

    Returns
    -------
    Boolean mask, same length as z. Bounds are inclusive, matching the
    paper's "180 nm axial range centered around the main peak".
    """
    z = np.asarray(z, dtype=float).ravel()
    lo = main_peak_nm - half_width_nm
    hi = main_peak_nm + half_width_nm
    return (z >= lo) & (z <= hi)


def slab_bounds(
    main_peak_nm: float, half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM
) -> Tuple[float, float]:
    """(zmin, zmax) of the analysis slab -- convenient for writing the
    values back into the GUI's zmin/zmax fields so the user can see what
    was chosen automatically."""
    return (main_peak_nm - half_width_nm, main_peak_nm + half_width_nm)
