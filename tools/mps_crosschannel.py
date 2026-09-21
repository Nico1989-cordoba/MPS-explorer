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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from tools.mps_occupancy import SIGMA_CAP_FRACTION, compute_occupancy
from tools.mps_periodicity import (
    DEFAULT_SLAB_HALF_WIDTH_NM,
    ZPeriodicityResult,
    fit_z_periodicity,
)
from tools.mps_randomization import (
    build_annulus_candidates,
    place_with_min_distance,
    smooth_contour_bspline,
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
    # How much of channel A's covered perimeter channel B also covers.
    shared: Optional["SharedOccupancyResult"] = None
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


# ============================================================================
# Shared perimeter occupancy
# ============================================================================
#
# The thesis question in one number: what fraction of the betaII-spectrin
# ring also carries the partner protein? Occupancy already answers "how
# much of the perimeter does one channel cover" (parameter 6, Gazal et
# al. 2026). Two channels give two covered sets on the SAME sampled
# perimeter, and their intersection is the shared part.
#
# Spectrin defines the MPS, so the perimeter is spectrin's: channel B's
# clusters are projected onto channel A's contour by the same
# compute_occupancy, with the same Mahalanobis threshold, the same
# ellipse mode and the same sample points. That last one is checked
# rather than assumed -- two different samplings would make the two masks
# incomparable element by element, and nothing downstream would notice.
#
# Nothing here is from a paper. No published method measures this: Gazal
# et al. analyse one channel, Xu et al. (2013) read alternation off 1D
# projected histograms. What is borrowed is the machinery -- the
# constrained Gaussian, the Mahalanobis criterion, and the annulus
# randomization of parameter 8 -- so the shared number is built out of
# quantities this program already validates.

# A registration error above this fraction of channel A's median covered
# patch makes the overlap worth reading only with its error bar; above
# the second, the bare number is not a measurement at all. The scale is
# the patch and not the perimeter because that is what an overlap is
# resolved against: measured on the one loadable real pair, channel A's
# covered patches are a median of 86 nm.
SHARED_REGISTRATION_WARN = 0.10
SHARED_REGISTRATION_REFUSE = 0.50
# Matching parameter 8, which uses 1000 randomizations.
DEFAULT_SHARED_NULL = 1000
# Above this, one channel-B cluster alone is carrying the answer.
SINGLE_CLUSTER_SHARE_WARN = 0.25


@dataclass
class SharedOccupancyResult:
    """
    How much of one channel's covered perimeter the other also covers.

    ``shared_of_a`` is the headline: the fraction of the perimeter
    covered by channel A that channel B also covers. ``shared_of_b`` is
    the same quantity the other way round -- a specificity check, "is the
    partner confined to spectrin?" -- and ``jaccard`` is the symmetric
    version, reported because it is cheap and because neither one-sided
    fraction alone shows when both channels cover almost everything.

    All three are fractions in [0, 1], not percentages, and every one of
    them is None when ``reason`` says why nothing could be measured.
    """

    shared_of_a: Optional[float]
    shared_of_b: Optional[float]
    jaccard: Optional[float]
    occupancy_a_percent: Optional[float]
    occupancy_b_percent: Optional[float]
    shared_length_nm: Optional[float]
    perimeter_length_nm: Optional[float]
    # A channel-B cluster can sit far enough from channel A's contour to
    # reach none of its sample points. It then contributes nothing, and
    # how many did so is part of reading the number.
    n_clusters_b_on_contour: int
    n_clusters_b_off_contour: int
    # The largest share any SINGLE channel-B cluster covers on its own.
    b_largest_single_share: Optional[float]
    # The scale the overlap is resolved at: the median length of channel
    # A's covered patches.
    median_patch_a_nm: Optional[float]
    null_median_shared_of_a: Optional[float]
    null_ci_shared_of_a: Optional[Tuple[float, float]]
    # P(null >= measured): small means the overlap exceeds chance.
    p_null_at_least_measured: Optional[float]
    n_null: int
    # What the registration error alone does to shared_of_a, as a 95 %
    # band. None when the registration error is unknown.
    registration_band_shared_of_a: Optional[Tuple[float, float]]
    # Empty when there is a number; otherwise why there is not.
    reason: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.shared_of_a is not None


def _arc_runs(mask: NDArray[np.bool_], spacing_nm: float
              ) -> NDArray[np.float64]:
    """
    The lengths of the covered patches, treating the perimeter as closed.

    A patch that straddles the sampling's start and end is one patch, not
    two: the perimeter is a loop and the sample index zero is arbitrary.
    """
    if not mask.any():
        return np.array([])
    if mask.all():
        return np.array([float(mask.sum()) * spacing_nm])
    start = np.flatnonzero(mask & ~np.roll(mask, 1))
    end = np.flatnonzero(mask & ~np.roll(mask, -1))
    end = np.roll(end, -int(np.searchsorted(end, start[0])))
    return np.asarray(((end - start) % len(mask) + 1) * spacing_nm,
                      dtype=float)


def _ellipse_mask(ellipses: Sequence[Any], probes: NDArray[np.float64],
                  threshold: float,
                  shift: Tuple[float, float] = (0.0, 0.0)) -> NDArray[np.bool_]:
    """Which sampled perimeter points any of these ellipses covers.

    The union over clusters, exactly as ``compute_occupancy`` does it, so
    overlapping ellipses never count a length twice. ``shift`` moves every
    ellipse rigidly, which is how both the null and the registration band
    are built.
    """
    covered = np.zeros(len(probes), dtype=bool)
    limit = float(threshold) ** 2
    offset = np.asarray(shift, dtype=float)
    for ellipse in ellipses:
        d = probes - (ellipse.mean + offset)
        covered |= np.einsum("ij,jk,ik->i", d, ellipse.inv_cov, d) <= limit
    return covered


def _kept_labels(analysis: AxonAnalysis) -> set:
    """The clusters an analysis removed, automatically or by hand."""
    return set(analysis.bad_report.bad_labels) | set(analysis.discarded_labels)


def shared_occupancy(
    analysis_a: Optional[AxonAnalysis],
    analysis_b: Optional[AxonAnalysis],
    *,
    registration: Optional[Registration] = None,
    n_null: int = DEFAULT_SHARED_NULL,
    annulus_half_width_nm: float = 50.0,
    grid_spacing_nm: float = 5.0,
    n_registration_draws: int = 200,
    random_seed: int = 0,
) -> SharedOccupancyResult:
    """
    The fraction of channel A's covered perimeter that channel B covers.

    Parameters
    ----------
    analysis_a : the reference channel, betaII-spectrin. Its contour and
        its occupancy define the perimeter and its sampling; both must
        exist or nothing is measured.
    analysis_b : the partner channel, ALREADY registered onto A and
        analysed on the same axial slab.
    registration : how channel B was put in A's frame. Its lateral error
        becomes an error bar, and a large one withdraws the number.
    n_null : draws of the null, in which channel B's ellipses are moved
        rigidly to random positions in the annulus around A's contour.
    n_registration_draws : draws of the registration band.

    Returns
    -------
    SharedOccupancyResult, whose ``reason`` is non-empty and whose
    numbers are all None when the pair cannot be measured.

    Notes
    -----
    Three things this does NOT do, each deliberately.

    It does not re-sample the perimeter for channel B. It asks
    ``compute_occupancy`` for the same ``n_points`` on the same contour
    and then checks that the sample points came back identical; if they
    did not, the two masks would not be comparable point by point and the
    intersection would be meaningless.

    It does not decide that a channel-B cluster away from A's contour is
    an error. Both channels straddle the contour -- it is a polyline
    through cluster centres, not a boundary -- so "outside" is not a
    defect. A cluster that reaches no sample point simply contributes
    nothing, and the count of those is returned so that a low shared
    fraction built on two of three clusters can be read as such.

    It does not compare across different occupancy settings. A different
    Mahalanobis threshold or ellipse mode changes what "covered" means in
    each channel, and an intersection of two different definitions is not
    a measurement, so it is refused rather than computed.
    """
    reg = registration or NO_REGISTRATION
    warnings_: List[str] = []
    nothing: Dict[str, Any] = dict(
        shared_of_a=None, shared_of_b=None, jaccard=None,
        occupancy_a_percent=None, occupancy_b_percent=None,
        shared_length_nm=None, perimeter_length_nm=None,
        n_clusters_b_on_contour=0, n_clusters_b_off_contour=0,
        b_largest_single_share=None, median_patch_a_nm=None,
        null_median_shared_of_a=None, null_ci_shared_of_a=None,
        p_null_at_least_measured=None, n_null=0,
        registration_band_shared_of_a=None)

    if (analysis_a is None or analysis_a.perimeter is None
            or analysis_a.occupancy is None):
        return SharedOccupancyResult(
            **nothing, warnings=warnings_,
            reason="channel A has no contour, so there is no perimeter to "
                   "share")
    if analysis_b is None or analysis_b.n_clusters_kept == 0:
        return SharedOccupancyResult(
            **nothing, warnings=warnings_,
            reason="channel B has no clusters")
    if (analysis_a.mahalanobis_threshold != analysis_b.mahalanobis_threshold
            or analysis_a.ellipse_mode != analysis_b.ellipse_mode):
        return SharedOccupancyResult(
            **nothing, warnings=warnings_,
            reason=(f"the two channels were measured with different occupancy "
                    f"settings (Mahalanobis "
                    f"{analysis_a.mahalanobis_threshold:g} vs "
                    f"{analysis_b.mahalanobis_threshold:g}, ellipse "
                    f"{analysis_a.ellipse_mode} vs {analysis_b.ellipse_mode})"))

    occ_a = analysis_a.occupancy
    probes = occ_a.perimeter_points
    spacing = occ_a.point_spacing_nm
    mask_a = occ_a.occupied_mask
    threshold = analysis_a.mahalanobis_threshold

    occ_b = compute_occupancy(
        analysis_b.x_slab, analysis_b.y_slab, analysis_b.labels,
        analysis_a.perimeter.contour,
        exclude_labels=_kept_labels(analysis_b),
        n_points=occ_a.n_points,
        mahalanobis_threshold=threshold,
        sigma_cap_fraction=SIGMA_CAP_FRACTION,
        ellipse_mode=analysis_a.ellipse_mode,
        # occ_a already settled the sampling; re-deciding it here could
        # return a different number of points for the same perimeter.
        ensure_subnm_spacing=False)
    if not np.array_equal(occ_b.perimeter_points, probes):
        return SharedOccupancyResult(
            **nothing, warnings=warnings_,
            reason="the two channels were sampled at different points on the "
                   "perimeter, so their covered sets cannot be intersected")
    mask_b = occ_b.occupied_mask
    ellipses_b = occ_b.ellipses

    length_a = float(mask_a.sum()) * spacing
    length_b = float(mask_b.sum()) * spacing
    length_shared = float((mask_a & mask_b).sum()) * spacing
    length_union = float((mask_a | mask_b).sum()) * spacing
    shared_of_a = length_shared / length_a if length_a else None
    shared_of_b = length_shared / length_b if length_b else None
    jaccard = length_shared / length_union if length_union else None

    per_cluster = [_ellipse_mask([e], probes, threshold) for e in ellipses_b]
    off_contour = sum(1 for m in per_cluster if not m.any())
    largest = max((float(m.mean()) for m in per_cluster), default=0.0)
    patches = _arc_runs(mask_a, spacing)
    median_patch = float(np.median(patches)) if patches.size else None

    if off_contour:
        warnings_.append(
            f"{off_contour} of the {len(ellipses_b)} channel-B clusters reach "
            f"no point of channel A's contour, so the shared fraction "
            f"describes only the {len(ellipses_b) - off_contour} that do.")
    if largest > SINGLE_CLUSTER_SHARE_WARN:
        warnings_.append(
            f"One channel-B cluster alone covers {100 * largest:.0f} % of the "
            f"perimeter. A shared fraction built on it is a statement about "
            f"one blob, not about a pattern of clusters.")
    if length_b == 0:
        warnings_.append(
            "Channel B covers no part of channel A's contour.")

    # ---- the null -------------------------------------------------------
    # Channel B's ellipses, each carrying its own size and orientation,
    # moved rigidly to random positions in the annulus around A's
    # contour. Everything else is held: channel A entirely, the contour,
    # the sampling, how many B clusters there are, and their own smallest
    # separation -- so the only thing randomized is WHERE they sit.
    null_median = null_ci = p_null = None
    n_null_done = 0
    if n_null > 0 and ellipses_b and length_a > 0:
        centres_b = np.array([e.mean for e in ellipses_b], dtype=float)
        if len(centres_b) >= 2:
            d, _ = cKDTree(centres_b).query(centres_b, k=2)
            min_separation = float(np.min(d[:, 1]))
        else:
            min_separation = 0.0
        candidates = build_annulus_candidates(
            smooth_contour_bspline(analysis_a.perimeter.contour),
            annulus_half_width_nm, grid_spacing_nm)
        if len(candidates) < len(ellipses_b):
            warnings_.append(
                "The annulus around channel A's contour holds fewer "
                "positions than channel B has clusters, so no null could be "
                "built.")
        else:
            rng = np.random.default_rng(random_seed)
            values = np.empty(int(n_null))
            for draw in range(int(n_null)):
                centres = place_with_min_distance(
                    candidates, len(ellipses_b), min_separation, rng)
                covered = np.zeros(len(probes), dtype=bool)
                limit = threshold ** 2
                for ellipse, centre in zip(ellipses_b, centres):
                    d = probes - centre
                    covered |= np.einsum(
                        "ij,jk,ik->i", d, ellipse.inv_cov, d) <= limit
                values[draw] = float((mask_a & covered).sum()) * spacing \
                    / length_a
            null_median = float(np.median(values))
            null_ci = (float(np.percentile(values, 2.5)),
                       float(np.percentile(values, 97.5)))
            n_null_done = int(n_null)
            if shared_of_a is not None:
                p_null = float(np.mean(values >= shared_of_a))
                if p_null > 0.05:
                    warnings_.append(
                        f"Channel-B clusters placed at random cover at least "
                        f"as much of channel A's perimeter in "
                        f"{100 * p_null:.0f} % of draws: this overlap is not "
                        f"distinguishable from chance.")

    # ---- what the registration error alone does -------------------------
    band = None
    lateral = reg.lateral_rms_nm
    if lateral is None:
        warnings_.append(
            "The lateral registration of channel B is unknown, so this "
            "shared fraction has no error bar: a shift the size of one "
            "covered patch changes it completely.")
    elif shared_of_a is not None and n_registration_draws > 0:
        rng = np.random.default_rng(random_seed + 1)
        # An isotropic 2-D error of total RMS ``lateral`` has this per
        # axis, which is how Registration reports it.
        draws = rng.normal(0.0, lateral / np.sqrt(2.0),
                           size=(int(n_registration_draws), 2))
        values = np.empty(len(draws))
        for i, shift in enumerate(draws):
            covered = _ellipse_mask(ellipses_b, probes, threshold,
                                    shift=(float(shift[0]), float(shift[1])))
            values[i] = float((mask_a & covered).sum()) * spacing / length_a
        band = (float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)))
        if median_patch:
            ratio = lateral / median_patch
            if ratio >= SHARED_REGISTRATION_REFUSE:
                warnings_.append(
                    f"The registration error ({lateral:.0f} nm) is "
                    f"{ratio:.2f} of channel A's median covered patch "
                    f"({median_patch:.0f} nm): this overlap is not a "
                    f"measurement.")
            elif ratio >= SHARED_REGISTRATION_WARN:
                warnings_.append(
                    f"The registration error ({lateral:.0f} nm) is "
                    f"{ratio:.2f} of channel A's median covered patch "
                    f"({median_patch:.0f} nm); read the band, not the "
                    f"number.")

    return SharedOccupancyResult(
        shared_of_a=shared_of_a, shared_of_b=shared_of_b, jaccard=jaccard,
        occupancy_a_percent=100.0 * float(mask_a.mean()),
        occupancy_b_percent=100.0 * float(mask_b.mean()),
        shared_length_nm=length_shared,
        perimeter_length_nm=occ_a.perimeter_length_nm,
        n_clusters_b_on_contour=len(ellipses_b) - off_contour,
        n_clusters_b_off_contour=off_contour,
        b_largest_single_share=largest,
        median_patch_a_nm=median_patch,
        null_median_shared_of_a=null_median,
        null_ci_shared_of_a=null_ci,
        p_null_at_least_measured=p_null,
        n_null=n_null_done,
        registration_band_shared_of_a=band,
        warnings=warnings_,
    )


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
        # The contour's area centroid, which is this project's axon
        # centre since 2026-09-19. The mean of the two channels'
        # centroids used to stand here, and it moves with how channel B
        # was clustered: measured on the one loadable real pair it sits
        # 77 to 151 nm from the area centroid, and channel A's own median
        # radius changed by 54 to 59 nm depending on channel B's DBSCAN
        # parameters -- a number about channel A that channel B could
        # move is not a measurement of channel A.
        centre_a = an_a.centre if an_a is not None else None
        if centre_a is not None:
            center = np.array([centre_a.x_nm, centre_a.y_nm], dtype=float)
        else:
            center = np.concatenate([ca, cb], axis=0).mean(axis=0)
            warnings_.append(
                "Channel A has no contour centre, so the angular and radial "
                "numbers are measured about the mean of both channels' "
                "centroids, which moves with how channel B was clustered.")
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

    # ---- shared perimeter occupancy -------------------------------
    shared = shared_occupancy(
        an_a, an_b, registration=registration, n_null=n_null,
        annulus_half_width_nm=annulus_half_width_nm,
        grid_spacing_nm=grid_spacing_nm, random_seed=random_seed)
    if shared.reason:
        warnings_.append(
            f"No shared perimeter occupancy: {shared.reason}.")
    warnings_.extend(shared.warnings)

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
        registration_rms_nm=lateral_error, shared=shared,
        registration=registration,
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
    s = t.shared if t is not None else None
    ci = s.null_ci_shared_of_a if s is not None else None
    band = s.registration_band_shared_of_a if s is not None else None
    row.update({
        "shared_of_a": None if s is None else s.shared_of_a,
        "shared_of_b": None if s is None else s.shared_of_b,
        "shared_jaccard": None if s is None else s.jaccard,
        "shared_length_nm": None if s is None else s.shared_length_nm,
        "occupancy_a_percent": None if s is None else s.occupancy_a_percent,
        "occupancy_b_percent": None if s is None else s.occupancy_b_percent,
        "n_clusters_b_on_contour":
            None if s is None else s.n_clusters_b_on_contour,
        "n_clusters_b_off_contour":
            None if s is None else s.n_clusters_b_off_contour,
        "b_largest_single_share":
            None if s is None else s.b_largest_single_share,
        "median_patch_a_nm": None if s is None else s.median_patch_a_nm,
        "null_median_shared_of_a":
            None if s is None else s.null_median_shared_of_a,
        "null_ci_lo_shared_of_a": None if ci is None else ci[0],
        "null_ci_hi_shared_of_a": None if ci is None else ci[1],
        "p_null_at_least_measured":
            None if s is None else s.p_null_at_least_measured,
        "n_null_shared": None if s is None else s.n_null,
        "registration_band_lo_shared_of_a": None if band is None else band[0],
        "registration_band_hi_shared_of_a": None if band is None else band[1],
        "shared_not_measured_because": "" if s is None else s.reason,
    })
    return row
