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

Expected result in sciatic nerve: Delta-Z = 170 +/- 15 nm (pooled),
consistent with the spectrin tetramer spacing between actin rings.

Design notes
------------
* The whole loaded ROI is used, as-is. Per the user's workflow, the ROI
  that is loaded IS the ROI that is analysed -- no interactive
  sub-selection of regions inside the axon (unlike Fig. 2A of the paper,
  where sub-ROIs were picked by hand).
* GMM fitting is randomly initialised, so ``random_state`` is pinned and
  ``n_init`` raised: the same ROI must always yield the same Delta-Z, or
  the reported periodicity would drift between runs of the same data.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture


# Paper constants
DEFAULT_COMPONENT_RANGE: Tuple[int, ...] = (2, 3)
DEFAULT_MIN_WEIGHT = 0.05          # ">5% of the z-values"
DEFAULT_SLAB_HALF_WIDTH_NM = 90.0  # 180 nm window, +/-90 nm around the peak

# GMM determinism / stability
_RANDOM_STATE = 0
_N_INIT = 10


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

    @property
    def mean_delta_z_nm(self) -> Optional[float]:
        """Mean Delta-Z for this ROI, or None if fewer than 2 components
        survived the weight filter (a single peak yields no spacing)."""
        if self.delta_z_nm.size == 0:
            return None
        return float(np.mean(self.delta_z_nm))

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


def fit_z_periodicity(
    z: NDArray[np.float64],
    component_range: Sequence[int] = DEFAULT_COMPONENT_RANGE,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    main_peak_mode: str = "density",
) -> ZPeriodicityResult:
    """
    Fit a Gaussian mixture model to the Z coordinates of one ROI and
    extract the axial periodicity (Delta-Z) and the main axial peak.

    Parameters
    ----------
    z : 1-D array of Z coordinates in nm (all localizations of the ROI).
    component_range : candidate component counts to compare by BIC.
        Default (2, 3) per the paper.
    min_weight : mixture-weight threshold; components at or below this are
        discarded as non-dominant. Default 0.05 (">5%").
    main_peak_mode : how to locate the centre of the 180 nm slab.
        - "density" (default): the global maximum of the fitted mixture
          density, i.e. literally "the main peak of the axial localization
          distribution". Robust when two components overlap into one
          visible peak.
        - "weight": the mean of the highest-weight retained component.
        Both are computed; this only selects which one is returned as
        ``main_peak_nm``.

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

    warnings_: List[str] = []
    Z = z.reshape(-1, 1)

    # --- BIC model selection over the candidate component counts --------
    # Guard against asking for more components than we have samples for;
    # sklearn raises in that case, and a ROI that sparse cannot support a
    # multi-component axial model anyway.
    usable = [n for n in component_range if n <= z.size]
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
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            random_state=_RANDOM_STATE,
            n_init=_N_INIT,
        )
        gmm.fit(Z)
        bic = float(gmm.bic(Z))
        bic_by_n[int(n)] = bic
        if bic < best_bic:
            best_bic, best_gmm, best_n = bic, gmm, int(n)

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

    if main_peak_mode == "density":
        main_peak = peak_by_density
    elif main_peak_mode == "weight":
        main_peak = peak_by_weight
    else:
        raise ValueError(
            f"main_peak_mode must be 'density' or 'weight', got {main_peak_mode!r}"
        )

    # If the two definitions disagree substantially the axial profile is
    # ambiguous (e.g. two comparable peaks); the slab choice then changes
    # which MPS segment is analysed, so surface it rather than hide it.
    if abs(peak_by_density - peak_by_weight) > DEFAULT_SLAB_HALF_WIDTH_NM:
        warnings_.append(
            f"Main-peak estimates disagree by "
            f"{abs(peak_by_density - peak_by_weight):.0f} nm "
            f"(density {peak_by_density:.0f} nm vs weight {peak_by_weight:.0f} nm): "
            f"the axial distribution has more than one comparable peak, so the "
            f"180 nm slab may not isolate the intended MPS segment."
        )

    # Even when both estimates agree, near-equal component weights mean the
    # "main" peak is only nominally dominant: a small change in the data
    # would move the slab onto a different MPS segment entirely, changing
    # every downstream parameter. Flag it so the user can decide whether
    # to trust the automatic slab for this ROI.
    if weights.size >= 2:
        ranked = np.sort(weights)[::-1]
        if ranked[0] - ranked[1] < 0.10:
            warnings_.append(
                f"Ambiguous main peak: the two strongest axial components have "
                f"nearly equal weights ({ranked[0]:.2f} vs {ranked[1]:.2f}). "
                f"The 180 nm slab was centred on z = {main_peak:.0f} nm, but a "
                f"different MPS segment is almost equally supported by the data."
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
    )


@dataclass
class ValleyResult:
    """Boundaries between consecutive MPS segments, and how well the axial
    density actually separates them."""

    positions_nm: NDArray[np.float64]   # one per consecutive pair of means
    is_true_valley: NDArray[np.bool_]   # False where the midpoint was used
    relative_depth: NDArray[np.float64]  # 0 = no dip at all, 1 = density -> 0
    midpoints_nm: NDArray[np.float64]
    warnings: List[str] = field(default_factory=list)

    @property
    def n_boundaries(self) -> int:
        return int(self.positions_nm.size)

    @property
    def shift_from_midpoint_nm(self) -> NDArray[np.float64]:
        return np.abs(self.positions_nm - self.midpoints_nm)


def density_valleys(
    result: ZPeriodicityResult,
    grid_step_nm: float = 0.5,
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

    Whether they separate at all is the point of ``relative_depth``. Two
    components broad enough relative to their spacing sum to a density with
    no interior minimum between them: there is then no valley to cut at,
    and no evidence in the axial profile that these are two resolved rings
    rather than one broad distribution the mixture happened to split.
    Measured on the 18-axon dataset, that is the case for more than half of
    the boundaries (component sigmas of 75-90 nm against a ~190 nm
    spacing), so this is the normal case, not a corner case. Read
    ``relative_depth`` before treating two segments as distinct rings.

    Parameters
    ----------
    result : a fit from ``fit_z_periodicity``.
    grid_step_nm : sampling of the search. 0.5 nm is far below both the
        localization precision (~20 nm) and the periodicity (~170 nm), so
        it cannot limit the result.

    Returns
    -------
    ValleyResult -- one entry per consecutive pair of dominant means, in
    ascending axial order, so entry ``i`` separates component ``i`` from
    component ``i + 1``. Empty arrays if fewer than two dominant
    components.
    """
    means = np.asarray(result.means_nm, dtype=float).ravel()
    if means.size < 2:
        empty_f = np.array([], dtype=float)
        return ValleyResult(
            positions_nm=empty_f, is_true_valley=np.array([], dtype=bool),
            relative_depth=empty_f, midpoints_nm=empty_f)

    midpoints = (means[:-1] + means[1:]) / 2.0
    if np.asarray(result.all_means_nm).size == 0:
        return ValleyResult(
            positions_nm=midpoints,
            is_true_valley=np.zeros(midpoints.size, dtype=bool),
            relative_depth=np.zeros(midpoints.size, dtype=float),
            midpoints_nm=midpoints,
            warnings=["Mixture parameters unavailable, so segment boundaries "
                      "fell back to the midpoints between component means."])

    at_means = result.mixture_density(means)

    positions: List[float] = []
    true_valley: List[bool] = []
    depths: List[float] = []

    for i, (lo, hi) in enumerate(zip(means[:-1], means[1:])):
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            positions.append(float(midpoints[i]))
            true_valley.append(False)
            depths.append(0.0)
            continue

        n = int(np.clip(np.ceil(span / max(grid_step_nm, 1e-6)), 64, 20_000))
        # Strictly between the two means: the means themselves are where the
        # density peaks, never where it bottoms out.
        grid = np.linspace(lo, hi, n + 2)[1:-1]
        dens = result.mixture_density(grid)
        k = int(np.argmin(dens))

        # An interior minimum is what makes a valley a valley. When the
        # density falls monotonically across the whole interval -- the
        # components too broad to resolve, or one overwhelming the other --
        # the minimum pins to an endpoint, i.e. on top of a component mean,
        # which would leave that segment with no slab at all. The midpoint
        # is the honest fallback, and the depth of 0 records that there was
        # nothing to find.
        interior = 0 < k < grid.size - 1
        # Referenced to the shallower of the two peaks, so a weak component
        # next to a strong one cannot inflate the apparent separation.
        ref = float(np.min(at_means[i:i + 2]))
        depth = (0.0 if ref <= 0
                 else float(np.clip(1.0 - dens[k] / ref, 0.0, 1.0)))

        positions.append(float(grid[k]) if interior else float(midpoints[i]))
        true_valley.append(bool(interior))
        depths.append(depth if interior else 0.0)

    pos = np.asarray(positions, dtype=float)
    tv = np.asarray(true_valley, dtype=bool)
    dp = np.asarray(depths, dtype=float)
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

    shallow = int(np.count_nonzero(tv & (dp < 0.05)))
    if shallow:
        warnings_.append(
            f"{shallow} boundary(ies) dip less than 5% below the shallower "
            f"adjacent peak: the two segments are barely separated axially."
        )

    shift = np.abs(pos - midpoints)
    if shift.size and np.max(shift) > 1.0:
        warnings_.append(
            f"Valley boundaries sit up to {np.max(shift):.0f} nm away from the "
            f"midpoints between means (median {np.median(shift):.0f} nm)."
        )

    return ValleyResult(
        positions_nm=pos, is_true_valley=tv, relative_depth=dp,
        midpoints_nm=midpoints, warnings=warnings_)


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
