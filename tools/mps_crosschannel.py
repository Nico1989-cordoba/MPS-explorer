# -*- coding: utf-8 -*-
"""
Two-channel analysis: betaII-spectrin against an associated protein.

Built for objectives 2 and 3 of the thesis, which image betaII-spectrin
together with alpha-adducin or protein 4.1B in sciatic-nerve cross
sections and ask whether losing the partner protein changes how spectrin
is arranged.

Two questions, answered with different data:

  AXIAL (the "in phase or antiphase" question, objective 1 item c).
    Xu, Zhong & Zhuang (2013, Science 339:452) established that along the
    axon, adducin colocalizes with actin and ALTERNATES with the
    betaII-spectrin C-terminus, while their evidence for it is the
    qualitative alternation of peaks in 1D projected histograms. With 3D
    data from cross sections the same question becomes quantitative and
    does not need longitudinal sections at all: fit each channel's axial
    distribution separately and measure the offset between their peaks as
    a fraction of the period. 0 is in phase, 0.5 is antiphase.

  TRANSVERSE (objectives 2 and 3).
    Within one MPS segment, how are the two channels' clusters arranged
    relative to each other around the perimeter -- heterotypic nearest
    neighbours, angular co-distribution, and shared perimeter occupancy.

No published method exists for the transverse part: Gazal et al. (2026)
analyse a single channel, and Barabas et al. (2017) compares an image
against a synthetic reference rather than against a second channel. As in
the multi-segment module, the statistics here are offered with their
assumptions stated and no claim about which one answers the biology.

Channel registration is the limiting factor
-------------------------------------------
Every cross-channel number inherits the error of putting the two channels
in one frame. Xu et al. (2013) report ~7 nm residual for two-colour STORM;
in Exchange-PAINT the error is the drift between imaging rounds, in x, y
AND z. Both functions here take a ``tools.mps_registration.Registration``
and expect channel B ALREADY moved by its shift (``Registration.apply``).
The registration's errors then qualify the results: a cross-channel
distance comparable to the lateral error is flagged, because a 10 nm
"co-localization" measured with a 7 nm alignment error is not a
measurement; and the axial error bounds the phase, which a focus
difference between rounds shifts one to one.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from tools.mps_analysis import AxonAnalysis, analyze_axon
from tools.mps_multisegment import (
    MIN_CLUSTERS_FOR_ANGLE,
    angular_nn_offsets_deg,
    circular_cross_correlation,
    cluster_angles,
    fold_rotation,
)
from tools.mps_periodicity import (
    DEFAULT_SLAB_HALF_WIDTH_NM,
    ZPeriodicityResult,
    fit_z_periodicity,
)
from tools.mps_registration import (
    NO_REGISTRATION,
    Registration,
    export_registration,
)

# An axial registration error above this fraction of the period is called
# out: a tenth of a period is a fifth of the whole in-phase-to-antiphase
# range.
PHASE_UNCERTAINTY_WARN = 0.1


# ============================================================================
# Axial phase relationship
# ============================================================================

@dataclass
class AxialPhaseResult:
    """Axial relationship between the two channels' periodic distributions."""

    peak_a_nm: float
    peak_b_nm: float
    offset_nm: float                 # signed, b - a
    period_a_nm: Optional[float]     # mean Delta-Z of channel A
    period_b_nm: Optional[float]
    period_used_nm: Optional[float]
    phase_fraction: Optional[float]  # |offset| / period, folded to [0, 0.5]
    z_result_a: ZPeriodicityResult
    z_result_b: ZPeriodicityResult
    registration: Registration = field(default_factory=lambda: NO_REGISTRATION)
    # The axial registration error as a fraction of the period, or None
    # when either is unknown.
    phase_uncertainty: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def interpretation_hint(self) -> str:
        """Descriptive label only -- never a biological conclusion."""
        f = self.phase_fraction
        if f is None:
            return "undetermined"
        if f < 0.15:
            return "near in-phase"
        if f > 0.35:
            return "near antiphase"
        return "intermediate"


def axial_phase(
    z_a: NDArray[np.float64],
    z_b: NDArray[np.float64],
    period_nm: Optional[float] = None,
    registration: Optional[Registration] = None,
) -> AxialPhaseResult:
    """
    Offset between two channels' axial periodic distributions.

    Parameters
    ----------
    z_a, z_b : axial coordinates of each channel, in nm. Channel A is the
        reference (betaII-spectrin); channel B already registered onto it.
    period_nm : the MPS period to express the offset against. None uses the
        mean Delta-Z measured on channel A, falling back to channel B.
    registration : how channel B was registered. Its axial error is
        reported as ``phase_uncertainty``; when it is unknown, the result
        says that a focus difference between rounds is inside the offset.

    Returns
    -------
    AxialPhaseResult. ``phase_fraction`` is folded into [0, 0.5] because an
    offset of one full period is indistinguishable from no offset: the
    peaks are only ever located modulo the period.
    """
    warnings_: List[str] = []
    registration = registration or NO_REGISTRATION
    za = np.asarray(z_a, dtype=float).ravel()
    zb = np.asarray(z_b, dtype=float).ravel()

    ra = fit_z_periodicity(za)
    rb = fit_z_periodicity(zb)
    warnings_.extend(f"[ch A] {w}" for w in ra.warnings)
    warnings_.extend(f"[ch B] {w}" for w in rb.warnings)

    pa = ra.mean_delta_z_nm
    pb = rb.mean_delta_z_nm
    period = period_nm if period_nm is not None else (pa if pa else pb)

    offset = float(rb.main_peak_nm - ra.main_peak_nm)

    frac: Optional[float] = None
    if period and period > 0:
        frac = fold_rotation(abs(offset), period)
        if pa and pb and abs(pa - pb) > 0.25 * max(pa, pb):
            warnings_.append(
                f"The two channels give different periods ({pa:.0f} vs "
                f"{pb:.0f} nm). Expressing the offset as a phase assumes a "
                f"common period, so this fraction is unreliable."
            )
    else:
        warnings_.append(
            "No period could be measured in either channel; the axial "
            "offset cannot be expressed as a phase."
        )

    uncertainty: Optional[float] = None
    axial_error = registration.axial_rms_nm
    if axial_error is None:
        warnings_.append(
            "The axial registration of channel B is unknown: any focus "
            "difference between the two rounds is inside the measured "
            "offset, one to one. Register the channels on markers "
            "localized in both rounds.")
    elif period and period > 0:
        uncertainty = axial_error / period
        if uncertainty >= PHASE_UNCERTAINTY_WARN:
            warnings_.append(
                f"The axial registration error alone ({axial_error:.0f} nm) "
                f"is {uncertainty:.2f} of the period, so the phase is known "
                f"only to about +/-{uncertainty:.2f}.")

    return AxialPhaseResult(
        peak_a_nm=float(ra.main_peak_nm), peak_b_nm=float(rb.main_peak_nm),
        offset_nm=offset, period_a_nm=pa, period_b_nm=pb,
        period_used_nm=period, phase_fraction=frac,
        z_result_a=ra, z_result_b=rb, registration=registration,
        phase_uncertainty=uncertainty, warnings=warnings_,
    )


# ============================================================================
# Transverse cross-channel relationship
# ============================================================================

@dataclass
class CrossChannelResult:
    """Relationship between two channels' clusters in one MPS segment."""

    analysis_a: Optional[AxonAnalysis]
    analysis_b: Optional[AxonAnalysis]
    slab_nm: Tuple[float, float]

    n_clusters_a: int
    n_clusters_b: int

    # Heterotypic nearest neighbours (not symmetric: a->b differs from b->a)
    hetero_nn_a_to_b_nm: NDArray[np.float64]
    hetero_nn_b_to_a_nm: NDArray[np.float64]
    median_hetero_nn_a_to_b: Optional[float]
    median_hetero_nn_b_to_a: Optional[float]

    # Null: channel B relocated at random within the same annulus
    null_median_a_to_b: Optional[float]
    null_iqr_a_to_b: Optional[Tuple[float, float]]
    fraction_null_below_measured: Optional[float]

    # Angular co-distribution
    angular_nn_median_deg: Optional[float]
    best_rotation_deg: Optional[float]
    max_correlation: Optional[float]
    rotation_fraction_of_spacing: Optional[float]

    # Radial comparison
    median_radius_a_nm: Optional[float]
    median_radius_b_nm: Optional[float]

    # The lateral registration error, from ``registration``.
    registration_rms_nm: Optional[float]
    registration: Registration = field(default_factory=lambda: NO_REGISTRATION)
    warnings: List[str] = field(default_factory=list)


def _radii(centroids: NDArray[np.float64], center: NDArray[np.float64]):
    if centroids.size == 0:
        return np.array([])
    d = centroids - center
    return np.hypot(d[:, 0], d[:, 1])


def _null_hetero_nn(
    centroids_a: NDArray[np.float64],
    contour: NDArray[np.float64],
    n_b: int,
    n_iter: int,
    annulus_half_width_nm: float,
    grid_spacing_nm: float,
    seed: int,
) -> NDArray[np.float64]:
    """
    Median heterotypic 1NN when channel B is scattered at random through
    the same annulus its clusters actually occupy.

    Without this, a small measured distance means nothing: pack enough
    points around a thin ring and the nearest one is close by construction.
    """
    from tools.mps_randomization import (
        build_annulus_candidates,
        smooth_contour_bspline,
    )

    smoothed = smooth_contour_bspline(contour)
    candidates = build_annulus_candidates(
        smoothed, annulus_half_width_nm, grid_spacing_nm)
    if len(candidates) < n_b or centroids_a.size == 0:
        return np.array([])

    rng = np.random.default_rng(seed)
    tree_a = cKDTree(centroids_a)
    out = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.choice(len(candidates), size=n_b, replace=False)
        d, _ = cKDTree(candidates[idx]).query(centroids_a, k=1)
        out[i] = float(np.median(d))
    return out


def cross_channel_transverse(
    x_a: NDArray[np.float64], y_a: NDArray[np.float64], z_a: NDArray[np.float64],
    x_b: NDArray[np.float64], y_b: NDArray[np.float64], z_b: NDArray[np.float64],
    *,
    slab: Optional[Tuple[float, float]] = None,
    slab_half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM,
    registration: Optional[Registration] = None,
    analyze_kwargs_b: Optional[Dict[str, Any]] = None,
    n_null: int = 200,
    annulus_half_width_nm: float = 50.0,
    grid_spacing_nm: float = 5.0,
    random_seed: int = 0,
    **analyze_kwargs: Any,
) -> CrossChannelResult:
    """
    Compare two channels' cluster arrangements within one MPS segment.

    Parameters
    ----------
    x_a, y_a, z_a : reference channel (betaII-spectrin), nm.
    x_b, y_b, z_b : partner channel (alpha-adducin or 4.1B), nm, already
        registered onto channel A.
    slab : explicit (zmin, zmax). None centres a slab on channel A's main
        axial peak -- the MPS segment is defined by spectrin. Note that a
        partner protein sitting antiphase to spectrin is, by construction,
        under-represented in a slab centred on spectrin; check
        ``axial_phase`` before reading the transverse numbers.
    registration : how channel B was registered. Cross-channel distances
        within three times its lateral error are flagged.
    analyze_kwargs_b : overrides of ``analyze_kwargs`` for channel B -- its
        own clustering parameters, pixel size and source name.
    n_null : randomizations for the null distribution of heterotypic 1NN.

    Returns
    -------
    CrossChannelResult
    """
    warnings_: List[str] = []
    registration = registration or NO_REGISTRATION
    lateral_error = registration.lateral_rms_nm
    kwargs_b = {**analyze_kwargs, **(analyze_kwargs_b or {})}
    xa, ya, za = (np.asarray(v, float).ravel() for v in (x_a, y_a, z_a))
    xb, yb, zb = (np.asarray(v, float).ravel() for v in (x_b, y_b, z_b))

    if slab is None:
        ra = fit_z_periodicity(za)
        slab = (ra.main_peak_nm - slab_half_width_nm,
                ra.main_peak_nm + slab_half_width_nm)

    an_a = an_b = None
    try:
        an_a = analyze_axon(xa, ya, za, slab_override=slab,
                            run_randomization=False, **analyze_kwargs)
    except Exception as exc:                              # noqa: BLE001
        warnings_.append(f"Channel A analysis failed: {exc}")
    try:
        an_b = analyze_axon(xb, yb, zb, slab_override=slab,
                            run_randomization=False, **kwargs_b)
    except Exception as exc:                              # noqa: BLE001
        warnings_.append(f"Channel B analysis failed: {exc}")

    ca = an_a.centroids if an_a is not None else np.empty((0, 2))
    cb = an_b.centroids if an_b is not None else np.empty((0, 2))
    n_a, n_b = len(ca), len(cb)

    nn_ab = nn_ba = np.array([])
    med_ab = med_ba = None
    if n_a and n_b:
        d_ab, _ = cKDTree(cb).query(ca, k=1)
        d_ba, _ = cKDTree(ca).query(cb, k=1)
        nn_ab, nn_ba = np.asarray(d_ab, float), np.asarray(d_ba, float)
        med_ab, med_ba = float(np.median(nn_ab)), float(np.median(nn_ba))

        if lateral_error is None:
            warnings_.append(
                "The lateral registration of channel B is unknown, so no "
                "heterotypic distance can be told apart from a registration "
                "artefact.")
        elif med_ab <= 3 * lateral_error:
            warnings_.append(
                f"Median heterotypic distance ({med_ab:.0f} nm) is within "
                f"3x the channel registration error "
                f"({lateral_error:.0f} nm). At this separation the "
                f"two channels cannot be told apart from a registration "
                f"artefact."
            )
    else:
        warnings_.append(
            f"Heterotypic distances need clusters in both channels; "
            f"found {n_a} and {n_b}.")

    # ---- null control -------------------------------------------------
    null_med = null_iqr = frac_below = None
    if n_a and n_b and an_a is not None and an_a.perimeter is not None:
        null = _null_hetero_nn(
            ca, an_a.perimeter.contour, n_b, n_null,
            annulus_half_width_nm, grid_spacing_nm, random_seed)
        if null.size:
            null_med = float(np.median(null))
            null_iqr = (float(np.percentile(null, 25)),
                        float(np.percentile(null, 75)))
            frac_below = float(np.mean(null <= med_ab))
            if frac_below > 0.05:
                warnings_.append(
                    f"Randomly placed clusters achieve a heterotypic "
                    f"distance at least as small as the measured one in "
                    f"{100 * frac_below:.0f}% of draws: the measured "
                    f"proximity is not distinguishable from chance."
                )

    # ---- angular and radial -------------------------------------------
    ang_med = best_rot = max_corr = rot_frac = None
    rad_a = rad_b = None
    if n_a >= MIN_CLUSTERS_FOR_ANGLE and n_b >= MIN_CLUSTERS_FOR_ANGLE:
        center = np.concatenate([ca, cb], axis=0).mean(axis=0)
        aa = cluster_angles(ca, center)
        ab = cluster_angles(cb, center)
        nn = angular_nn_offsets_deg(aa, ab)
        if nn.size:
            ang_med = float(np.median(nn))
        off, corr, _bw = circular_cross_correlation(aa, ab)
        if corr.size and np.any(np.isfinite(corr)):
            k = int(np.nanargmax(corr))
            best_rot = float(off[k])
            max_corr = float(corr[k])
            rot_frac = fold_rotation(best_rot, 360.0 / ((n_a + n_b) / 2.0))
        rad_a = float(np.median(_radii(ca, center)))
        rad_b = float(np.median(_radii(cb, center)))

    for an, tag in ((an_a, "ch A"), (an_b, "ch B")):
        if an is not None:
            warnings_.extend(f"[{tag}] {w}" for w in an.warnings
                             if not w.startswith("Axial slab set manually"))

    return CrossChannelResult(
        analysis_a=an_a, analysis_b=an_b, slab_nm=slab,
        n_clusters_a=n_a, n_clusters_b=n_b,
        hetero_nn_a_to_b_nm=nn_ab, hetero_nn_b_to_a_nm=nn_ba,
        median_hetero_nn_a_to_b=med_ab, median_hetero_nn_b_to_a=med_ba,
        null_median_a_to_b=null_med, null_iqr_a_to_b=null_iqr,
        fraction_null_below_measured=frac_below,
        angular_nn_median_deg=ang_med, best_rotation_deg=best_rot,
        max_correlation=max_corr, rotation_fraction_of_spacing=rot_frac,
        median_radius_a_nm=rad_a, median_radius_b_nm=rad_b,
        registration_rms_nm=lateral_error, registration=registration,
        warnings=warnings_,
    )


def export_cross_channel(
    axial: Optional[AxialPhaseResult],
    transverse: Optional[CrossChannelResult],
    source_a: str = "",
    source_b: str = "",
) -> Dict[str, Any]:
    """
    Flat record for CSV export, one row per axon per channel pair.

    The columns are the same whatever was computed -- a part that was not
    is left empty -- so rows from 2D and 3D pairs share one table.
    """
    row: Dict[str, Any] = {"source_channel_a": source_a,
                           "source_channel_b": source_b}
    registration = (transverse.registration if transverse is not None
                    else axial.registration if axial is not None
                    else NO_REGISTRATION)
    row.update(export_registration(registration))
    a = axial
    row.update({
        "axial_peak_a_nm": None if a is None else round(a.peak_a_nm, 2),
        "axial_peak_b_nm": None if a is None else round(a.peak_b_nm, 2),
        "axial_offset_nm": None if a is None else round(a.offset_nm, 2),
        "period_a_nm": None if a is None else a.period_a_nm,
        "period_b_nm": None if a is None else a.period_b_nm,
        "axial_phase_fraction": None if a is None else a.phase_fraction,
        "axial_phase_uncertainty": None if a is None else a.phase_uncertainty,
        "axial_phase_label": None if a is None else a.interpretation_hint,
    })
    t = transverse
    row.update({
        "slab_zmin_nm": None if t is None else round(t.slab_nm[0], 2),
        "slab_zmax_nm": None if t is None else round(t.slab_nm[1], 2),
        "n_clusters_a": None if t is None else t.n_clusters_a,
        "n_clusters_b": None if t is None else t.n_clusters_b,
        "median_hetero_nn_a_to_b_nm":
            None if t is None else t.median_hetero_nn_a_to_b,
        "median_hetero_nn_b_to_a_nm":
            None if t is None else t.median_hetero_nn_b_to_a,
        "null_median_hetero_nn_nm":
            None if t is None else t.null_median_a_to_b,
        "fraction_null_below_measured":
            None if t is None else t.fraction_null_below_measured,
        "angular_nn_median_deg":
            None if t is None else t.angular_nn_median_deg,
        "rotation_fraction_of_spacing":
            None if t is None else t.rotation_fraction_of_spacing,
        "max_angular_correlation": None if t is None else t.max_correlation,
        "median_radius_a_nm": None if t is None else t.median_radius_a_nm,
        "median_radius_b_nm": None if t is None else t.median_radius_b_nm,
        "n_warnings": None if t is None else len(t.warnings),
    })
    return row
