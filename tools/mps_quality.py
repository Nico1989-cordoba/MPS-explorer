# -*- coding: utf-8 -*-
"""
Is this dataset good enough to answer the question being asked of it?

Every number the rest of MPS Explorer produces is a spread of some kind:
the axial width of a ring, the size of a cluster, the gap between two
patches. A spread is only meaningful next to the spread the measurement
could possibly resolve, and until the loader read the whole file there
was no way to put the two side by side.

The checks here all answer the same shape of question -- "is this
structure, or is this the microscope?" -- and they exist because the ring
analysis ran aground on exactly that ambiguity: 19 of 33 segment
boundaries showed no density valley, and nothing in the pipeline could
say whether that meant the rings are genuinely merged or merely
unresolved.

What is here
------------
``nena``               experimental localization precision, no calibration
``precision_check``    NeNA against the precision the fit reported
``box_size_check``     is the fitting box clipping the PSF?
``axial_resolvedness`` fitted sigma_z against lpz, per GMM component
``drift_check``        is there a coherent drift left after undrifting?
``axial_coverage``     how much of the astigmatic range the data uses

What is NOT here, on purpose
----------------------------
FRC (Fourier Ring Correlation) is the other standard resolution number
and Picasso has it. It measures something different -- the resolution of
the rendered IMAGE, which folds in labelling density -- and it needs a
rendering step this module otherwise does not. It is deferred rather
than half-done.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

# A component wider than this multiple of the localization precision is
# carrying structure, not just noise. 1.5 is the upper sigma bound G5M
# imposes on a molecule (picasso.g5m.MAX_SIGMA_FACTOR), i.e. the point
# past which the Jungmann lab stops calling a localization cloud a single
# emitter. Borrowed rather than invented, but it is a convention.
RESOLVED_SIGMA_FACTOR = 1.5

# Below this many next-frame neighbour pairs the NeNA fit is not worth
# reporting. Endesfelder's method needs the single-molecule peak to
# stand out of the random-neighbour background.
MIN_NENA_PAIRS = 500


# ===================================================================
#  NeNA -- experimental localization precision
# ===================================================================
@dataclass
class NeNAResult:
    """
    Nearest-Neighbour-based Analysis precision, in nanometres.

    Endesfelder et al., Histochem Cell Biol 141:629 (2014),
    DOI 10.1007/s00418-014-1192-3. The same molecule detected in
    consecutive frames appears twice, separated only by measurement
    error, so the distribution of next-frame neighbour distances has a
    peak whose width IS the localization precision. It needs no
    calibration and no knowledge of the photon count, which is what makes
    it the check on the precision the fit reports.
    """

    precision_nm: float
    n_pairs: int
    converged: bool
    distances_nm: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )
    histogram: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )
    best_fit: NDArray[np.float64] = field(
        default_factory=lambda: np.array([])
    )
    warnings: List[str] = field(default_factory=list)


def _next_frame_distances_nm(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    max_distance_nm: float,
) -> NDArray[np.float64]:
    """
    Distance from each localization to its nearest neighbour in the NEXT
    frame.

    Consecutive frames only. A molecule that is still emitting is in
    both; anything else contributes a random neighbour, which is the
    broad background the fit separates out.
    """
    order = np.argsort(frame, kind="stable")
    frame, x_nm, y_nm = frame[order], x_nm[order], y_nm[order]
    unique, start = np.unique(frame, return_index=True)
    end = np.append(start[1:], frame.size)
    bounds = {int(f): (int(s), int(e)) for f, s, e in zip(unique, start, end)}

    chunks: List[NDArray[np.float64]] = []
    for value, first, last in zip(unique, start, end):
        following = bounds.get(int(value) + 1)
        if following is None:
            continue
        n_first, n_last = following
        if n_last <= n_first or last <= first:
            continue
        tree = cKDTree(np.column_stack([x_nm[n_first:n_last],
                                        y_nm[n_first:n_last]]))
        distances, _ = tree.query(
            np.column_stack([x_nm[first:last], y_nm[first:last]]), k=1
        )
        chunks.append(np.asarray(distances, dtype=float))

    if not chunks:
        return np.array([], dtype=float)
    everything = np.concatenate(chunks)
    return everything[everything < max_distance_nm]


def nena(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    *,
    max_distance_nm: float = 120.0,
    bin_nm: float = 0.5,
    reported_lp_nm: Optional[float] = None,
) -> NeNAResult:
    """
    Localization precision from the next-frame neighbour distances.

    The model is Picasso's (``picasso.postprocess.nena``): a 2D-Gaussian
    displacement term for the same molecule seen twice,

        a * (d / 2s^2) * exp(-d^2 / 4s^2)

    plus a Gaussian for the structured background of nearby OTHER
    molecules. ``s`` is the precision. Three starting points are tried
    and the best residual kept, because a single start diverges on sparse
    data.

    Parameters
    ----------
    max_distance_nm : distances beyond this are dropped before fitting.
        Picasso uses 1 camera pixel; expressed in nm here so the value
        does not silently change meaning with the camera.
    reported_lp_nm : the fit's own precision, used only to seed the fit.
    """
    warnings: List[str] = []
    distances = _next_frame_distances_nm(
        np.asarray(frame, dtype=np.int64),
        np.asarray(x_nm, dtype=float),
        np.asarray(y_nm, dtype=float),
        max_distance_nm,
    )
    if distances.size < MIN_NENA_PAIRS:
        warnings.append(
            f"only {distances.size} next-frame neighbour pairs "
            f"(need about {MIN_NENA_PAIRS}); NeNA not attempted. This is "
            f"normal for a single picked region -- run it on the whole "
            f"field of view instead."
        )
        return NeNAResult(
            precision_nm=float("nan"),
            n_pairs=int(distances.size),
            converged=False,
            warnings=warnings,
        )

    edges = np.arange(0.0, max_distance_nm + bin_nm, bin_nm)
    counts, edges = np.histogram(distances, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = counts.astype(float)

    def model(d, delta_a, s, ac, dc, sc):
        a = ac + delta_a  # keeps a >= ac, as in Picasso
        single = a * (d / (2.0 * s**2)) * np.exp(-(d**2) / (4.0 * s**2))
        nearby = (
            ac / (sc * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((d - dc) / sc) ** 2)
        )
        return single + nearby

    area = float(np.trapezoid(counts, centres))
    seed = float(reported_lp_nm) if reported_lp_nm else max_distance_nm / 12.0
    peak = float(centres[int(np.argmax(counts))])
    starts = [
        [0.8 * area, seed, 0.1 * area, 2.0 * seed, seed],
        [0.8 * area, peak / np.sqrt(2.0), 0.1 * area,
         0.7 * centres[-1], 0.3 * centres[-1]],
        [0.8 * area, seed, 0.1 * area, 0.5 * centres[-1], seed],
    ]
    best: Optional[NDArray[np.float64]] = None
    best_residual = np.inf
    for start in starts:
        if not np.all(np.isfinite(start)):
            continue
        try:
            candidate, _ = curve_fit(
                model, centres, counts, p0=start,
                bounds=([0.0] * 5, [np.inf] * 5), maxfev=20000,
            )
        except (RuntimeError, ValueError):
            continue
        residual = float(np.sum((model(centres, *candidate) - counts) ** 2))
        if residual < best_residual:
            best_residual, best = residual, candidate

    if best is None:
        warnings.append(
            "NeNA did not converge on the next-frame neighbour histogram."
        )
        return NeNAResult(
            precision_nm=float("nan"),
            n_pairs=int(distances.size),
            converged=False,
            distances_nm=centres,
            histogram=counts,
            warnings=warnings,
        )

    return NeNAResult(
        precision_nm=float(best[1]),
        n_pairs=int(distances.size),
        converged=True,
        distances_nm=centres,
        histogram=counts,
        best_fit=model(centres, *best),
        warnings=warnings,
    )


# ===================================================================
#  Reported precision against NeNA
# ===================================================================
@dataclass
class PrecisionCheck:
    """Whether the precision columns in the file can be believed."""

    nena_nm: float
    reported_lateral_nm: float
    ratio: float                    # NeNA / reported
    reported_lpx_nm: float
    reported_lpy_nm: float
    reported_lpz_nm: Optional[float]
    n_pairs: int
    verdict: str
    warnings: List[str] = field(default_factory=list)


def precision_check(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    lpx_nm: NDArray[np.float64],
    lpy_nm: NDArray[np.float64],
    lpz_nm: Optional[NDArray[np.float64]] = None,
    **kwargs: Any,
) -> PrecisionCheck:
    """
    Compare NeNA against the precision the fitting step reported.

    Picasso's G5M guidance names a miscalibrated camera -- wrong photon
    counts, therefore wrong precisions -- as the most common reason a
    localization cloud comes out the wrong size. That failure corrupts
    lpx and lpy too, so NeNA agreeing with them rules it out. It does NOT
    directly validate lpz, which comes from the astigmatism calibration
    by a different route; what it does is remove the shared cause.
    """
    lpx_med = float(np.nanmedian(lpx_nm))
    lpy_med = float(np.nanmedian(lpy_nm))
    reported = 0.5 * (lpx_med + lpy_med)
    result = nena(frame, x_nm, y_nm, reported_lp_nm=reported, **kwargs)

    warnings = list(result.warnings)
    ratio = result.precision_nm / reported if reported > 0 else float("nan")
    if not result.converged:
        verdict = "not measured"
    elif ratio < 1.3:
        verdict = "reported precisions are consistent with NeNA"
    elif ratio < 2.0:
        verdict = "NeNA is worse than reported"
        warnings.append(
            f"NeNA ({result.precision_nm:.1f} nm) is {ratio:.1f}x the "
            f"reported precision ({reported:.1f} nm). Something not in the "
            f"fit's error model is spreading the localizations -- residual "
            f"drift, or an over-optimistic photon calibration. Any width "
            f"measured downstream inherits it."
        )
    else:
        verdict = "reported precisions are not credible"
        warnings.append(
            f"NeNA ({result.precision_nm:.1f} nm) is {ratio:.1f}x the "
            f"reported precision ({reported:.1f} nm). Treat every "
            f"precision-based threshold in this dataset as unfounded until "
            f"the discrepancy is explained."
        )

    return PrecisionCheck(
        nena_nm=result.precision_nm,
        reported_lateral_nm=reported,
        ratio=ratio,
        reported_lpx_nm=lpx_med,
        reported_lpy_nm=lpy_med,
        reported_lpz_nm=(
            None if lpz_nm is None else float(np.nanmedian(lpz_nm))
        ),
        n_pairs=result.n_pairs,
        verdict=verdict,
        warnings=warnings,
    )


# ===================================================================
#  Is the fitting box clipping the PSF?
# ===================================================================
@dataclass
class BoxSizeCheck:
    box_size_px: Optional[int]
    sx_median_px: float
    sy_median_px: float
    sx_p95_px: float
    sy_p95_px: float
    span_3sigma_p95_px: float
    fraction_over_box_2sigma: float
    fraction_over_box_3sigma: float
    warnings: List[str] = field(default_factory=list)


def box_size_check(
    sx_px: NDArray[np.float64],
    sy_px: NDArray[np.float64],
    box_size_px: Optional[int],
) -> BoxSizeCheck:
    """
    Whether the single-emitter images fit inside the box they were fitted
    in.

    Picasso extracts a square of ``Box Size`` pixels around each spot and
    fits the model to it. An astigmatic PSF near the ends of the z range
    is wide, and a spot wider than its box is fitted on a truncated
    image: sx and sy are then biased low, and everything derived from
    them -- z, lpz, the ellipticity cut -- inherits the bias. Picasso's
    own G5M troubleshooting names this and says to enlarge the box and
    re-run.

    Two fractions are reported because there is no single sharp
    threshold: a spot whose +-2 sigma exceeds the box is badly truncated,
    while +-3 sigma is where the tails start being lost.
    """
    sx = np.asarray(sx_px, dtype=float)
    sy = np.asarray(sy_px, dtype=float)
    larger = np.maximum(sx, sy)
    check = BoxSizeCheck(
        box_size_px=box_size_px,
        sx_median_px=float(np.nanmedian(sx)),
        sy_median_px=float(np.nanmedian(sy)),
        sx_p95_px=float(np.nanpercentile(sx, 95)),
        sy_p95_px=float(np.nanpercentile(sy, 95)),
        span_3sigma_p95_px=float(6.0 * np.nanpercentile(larger, 95)),
        fraction_over_box_2sigma=float("nan"),
        fraction_over_box_3sigma=float("nan"),
    )
    if box_size_px is None:
        check.warnings.append(
            "No 'Box Size' in the metadata, so the PSF cannot be compared "
            "against the box it was fitted in."
        )
        return check

    box = float(box_size_px)
    check.fraction_over_box_2sigma = float(np.mean(4.0 * larger > box))
    check.fraction_over_box_3sigma = float(np.mean(6.0 * larger > box))

    if check.fraction_over_box_2sigma > 0.05:
        check.warnings.append(
            f"{100 * check.fraction_over_box_2sigma:.0f} % of spots are wider "
            f"than the {box_size_px} px fitting box at +-2 sigma: those fits "
            f"saw a truncated image, so their sx/sy are biased low and every "
            f"z and lpz derived from them with them. Re-run Localize with a "
            f"box of at least {int(np.ceil(check.span_3sigma_p95_px)) | 1} px."
        )
    elif check.fraction_over_box_3sigma > 0.20:
        check.warnings.append(
            f"{100 * check.fraction_over_box_3sigma:.0f} % of spots exceed "
            f"the {box_size_px} px box at +-3 sigma. The cores are fitted but "
            f"the tails are cut; consider a box of "
            f"{int(np.ceil(check.span_3sigma_p95_px)) | 1} px for 3D data."
        )
    return check


# ===================================================================
#  Fitted axial width against axial precision
# ===================================================================
@dataclass
class ComponentResolvedness:
    """One GMM component in Z, measured against what Z can resolve."""

    index: int
    mean_nm: float
    sigma_nm: float
    weight: float
    lpz_nm: float
    ratio: float                        # sigma / lpz
    structural_width_nm: float          # sqrt(sigma^2 - lpz^2), 0 if <=
    is_resolved: bool                   # sigma <= RESOLVED_SIGMA_FACTOR * lpz


@dataclass
class AxialResolvedness:
    components: List[ComponentResolvedness]
    median_ratio: float
    lpz_nm: float
    n_precision_limited: int
    n_broad: int
    warnings: List[str] = field(default_factory=list)


def axial_resolvedness(
    z_nm: NDArray[np.float64],
    lpz_nm: NDArray[np.float64],
    means_nm: Union[Sequence[float], NDArray[Any]],
    sigmas_nm: Union[Sequence[float], NDArray[Any]],
    weights: Optional[Union[Sequence[float], NDArray[Any]]] = None,
) -> AxialResolvedness:
    """
    For each component of the axial mixture, is its width structure or
    precision?

    A ring that were infinitely thin in z would still produce a Gaussian
    of width lpz, because that is what the microscope does to a point.
    So ``sigma / lpz`` near 1 means "as thin as this data can show", and
    the implied structural width is sqrt(sigma^2 - lpz^2).

    This is the number that was missing when the ring analysis found no
    density valley at 19 of 33 boundaries. Two rings 190 nm apart, each
    of width lpz = 47 nm, are 4 sigma apart and would show a deep valley;
    the valleys were absent because the fitted components came out ~1.8x
    wider than lpz, and nothing could say why.

    ``lpz`` is taken per component, from the localizations nearest that
    component's mean, because axial precision degrades towards the ends
    of the astigmatic range and a single median would flatter the edges.
    """
    z = np.asarray(z_nm, dtype=float)
    lpz = np.asarray(lpz_nm, dtype=float)
    means = np.asarray(means_nm, dtype=float)
    sigmas = np.asarray(sigmas_nm, dtype=float)
    component_weights = (
        np.full(means.size, np.nan) if weights is None
        else np.asarray(weights, dtype=float)
    )

    warnings: List[str] = []
    overall_lpz = float(np.nanmedian(lpz)) if lpz.size else float("nan")
    components: List[ComponentResolvedness] = []

    for i, (mean, sigma) in enumerate(zip(means, sigmas)):
        # Localizations this component actually speaks for: within one
        # sigma of its mean. Falls back to all of them when that is empty.
        near = np.abs(z - mean) <= sigma
        local = lpz[near] if np.any(near) and lpz.size == z.size else lpz
        component_lpz = (
            float(np.nanmedian(local)) if local.size else overall_lpz
        )
        ratio = sigma / component_lpz if component_lpz > 0 else float("nan")
        structural = (
            float(np.sqrt(max(sigma**2 - component_lpz**2, 0.0)))
            if np.isfinite(component_lpz) else float("nan")
        )
        components.append(
            ComponentResolvedness(
                index=i,
                mean_nm=float(mean),
                sigma_nm=float(sigma),
                weight=(float(component_weights[i])
                        if i < component_weights.size else float("nan")),
                lpz_nm=component_lpz,
                ratio=float(ratio),
                structural_width_nm=structural,
                is_resolved=bool(ratio <= RESOLVED_SIGMA_FACTOR),
            )
        )

    ratios = np.array([c.ratio for c in components], dtype=float)
    n_limited = int(np.sum(ratios <= RESOLVED_SIGMA_FACTOR))
    n_broad = int(np.sum(ratios > 2.0))

    # A component NARROWER than the precision of the localizations it is
    # made of cannot be a real structure: the microscope alone would have
    # spread it that wide. It means one of the two numbers is wrong --
    # usually lpz overestimated near the ends of the astigmatic range,
    # sometimes the mixture putting a spurious narrow component on a
    # density bump. Either way the component's width says nothing about
    # the ring, so it is reported rather than passed on as "resolved".
    impossible = [c for c in components if np.isfinite(c.ratio)
                  and c.ratio < 0.7]
    if impossible:
        where = ", ".join(
            f"#{c.index} (sigma {c.sigma_nm:.0f} nm vs lpz {c.lpz_nm:.0f} nm)"
            for c in impossible
        )
        warnings.append(
            f"{len(impossible)} component(s) are narrower than the axial "
            f"precision of their own localizations: {where}. That is not "
            f"physically possible, so either lpz is overestimated there "
            f"(it degrades towards the ends of the astigmatic range) or "
            f"the mixture is over-fitting. Their widths are not usable as "
            f"a ring thickness."
        )

    if components and n_limited == 0:
        warnings.append(
            f"No component is within {RESOLVED_SIGMA_FACTOR}x the axial "
            f"precision: every one is wider than a single thin ring could "
            f"be. Either the rings genuinely have axial extent, or several "
            f"are being absorbed into one component."
        )
    if n_broad:
        warnings.append(
            f"{n_broad} component(s) exceed 2x lpz. A component that wide "
            f"may be two unresolved rings rather than one; read its "
            f"boundary's relative_depth before treating the two sides as "
            f"separate rings."
        )

    return AxialResolvedness(
        components=components,
        median_ratio=float(np.nanmedian(ratios)) if ratios.size else float("nan"),
        lpz_nm=overall_lpz,
        n_precision_limited=n_limited,
        n_broad=n_broad,
        warnings=warnings,
    )


# ===================================================================
#  Residual drift
# ===================================================================
@dataclass
class AxisDrift:
    axis: str
    shift_range_nm: float
    lag1_autocorrelation: float
    p_permutation: float
    is_coherent: bool


@dataclass
class DriftCheck:
    axes: List[AxisDrift]
    n_segments: int
    warnings: List[str] = field(default_factory=list)


def _profile_shifts_nm(
    values_nm: NDArray[np.float64],
    frame: NDArray[np.int64],
    n_segments: int,
    bin_nm: float,
    max_shift_nm: float,
) -> NDArray[np.float64]:
    """
    Per-time-segment displacement of the whole 1D profile along one axis.

    Each segment's histogram is cross-correlated against the histogram of
    the entire acquisition, and the lag of the maximum is that segment's
    shift. Using the whole PROFILE rather than its mean or median matters:
    for an extended object like an axon, which part of it is blinking
    changes over time, and that moves a mean without anything having
    drifted.
    """
    lo, hi = float(np.min(values_nm)), float(np.max(values_nm))
    edges = np.arange(lo - max_shift_nm, hi + max_shift_nm + bin_nm, bin_nm)
    reference, _ = np.histogram(values_nm, bins=edges)
    total = reference.sum()
    if total == 0:
        return np.array([], dtype=float)
    reference = reference / total

    bounds = np.linspace(frame.min(), frame.max() + 1, n_segments + 1)
    which = np.clip(np.digitize(frame, bounds) - 1, 0, n_segments - 1)
    lags = np.arange(
        -int(max_shift_nm / bin_nm), int(max_shift_nm / bin_nm) + 1
    )

    shifts: List[float] = []
    for segment in range(n_segments):
        inside = which == segment
        if int(inside.sum()) < 50:
            continue
        counts, _ = np.histogram(values_nm[inside], bins=edges)
        if counts.sum() == 0:
            continue
        counts = counts / counts.sum()
        scores = [float(np.dot(np.roll(counts, lag), reference))
                  for lag in lags]
        shifts.append(-float(lags[int(np.argmax(scores))]) * bin_nm)
    return np.asarray(shifts, dtype=float)


def drift_check(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: Optional[NDArray[np.float64]] = None,
    *,
    n_segments: int = 20,
    bin_nm: float = 5.0,
    max_shift_nm: float = 200.0,
    n_permutations: int = 2000,
    seed: int = 0,
) -> DriftCheck:
    """
    Is there a drift left in this file, after whatever undrifting was
    applied?

    Real drift is SMOOTH: consecutive time segments move by similar
    amounts, so the series of per-segment shifts is positively
    autocorrelated. Counting noise is not. The test is therefore the
    lag-1 autocorrelation of the shift series, against a permutation null
    that destroys the time order and keeps everything else.

    Worth running on z in particular: Picasso's RCC undrifting is 2D and
    does not touch z at all, so a file undrifted "by RCC" has had no
    axial correction whatsoever. AIM does correct z.
    """
    frame = np.asarray(frame, dtype=np.int64)
    rng = np.random.default_rng(seed)
    warnings: List[str] = []
    axes: List[AxisDrift] = []

    candidates: List[Tuple[str, Optional[NDArray[np.float64]]]] = [
        ("x", x_nm), ("y", y_nm), ("z", z_nm),
    ]
    for name, values in candidates:
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if values.size == 0 or np.allclose(values, values[0]):
            continue
        shifts = _profile_shifts_nm(
            values, frame, n_segments, bin_nm, max_shift_nm
        )
        if shifts.size < 8:
            warnings.append(
                f"{name}: only {shifts.size} usable time segments, too few "
                f"to tell drift from noise."
            )
            continue
        centred = shifts - shifts.mean()
        denominator = float(np.dot(centred, centred))
        if denominator <= 0:
            # Every segment gave the same shift, so there is no variation
            # to be autocorrelated. That is the opposite of drift, and
            # permuting a constant series would divide by zero.
            axes.append(
                AxisDrift(
                    axis=name,
                    shift_range_nm=0.0,
                    lag1_autocorrelation=0.0,
                    p_permutation=1.0,
                    is_coherent=False,
                )
            )
            continue
        acf = float(np.dot(centred[:-1], centred[1:]) / denominator)
        null = np.empty(n_permutations, dtype=float)
        for i in range(n_permutations):
            shuffled = rng.permutation(centred)
            null[i] = np.dot(shuffled[:-1], shuffled[1:]) / denominator
        p_value = float(np.mean(null >= acf))
        coherent = bool(p_value < 0.05 and acf > 0.2)
        axes.append(
            AxisDrift(
                axis=name,
                shift_range_nm=float(shifts.max() - shifts.min()),
                lag1_autocorrelation=acf,
                p_permutation=p_value,
                is_coherent=coherent,
            )
        )
        if coherent:
            warnings.append(
                f"{name}: a smooth residual drift of "
                f"{shifts.max() - shifts.min():.0f} nm across the "
                f"acquisition (lag-1 autocorrelation {acf:.2f}, p = "
                f"{p_value:.3f}). It adds directly to every width measured "
                f"along {name}."
            )

    return DriftCheck(axes=axes, n_segments=n_segments, warnings=warnings)


# ===================================================================
#  Did a drift correction sharpen the data or scramble it?
# ===================================================================
REPEAT_GAP_FRAMES = 200
_REPEAT_NEAREST = 16


def _frame_sample(frame: NDArray[np.int64], target: int,
                  seed: int) -> NDArray[np.bool_]:
    """A fixed pseudo-random share of FRAMES holding about ``target`` rows."""
    if frame.size <= target:
        return np.ones(frame.size, dtype=bool)
    mixed = (frame.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             + np.uint64(seed)) & np.uint64(0xFFFFFFFF)
    return mixed.astype(np.float64) / 2.0 ** 32 < target / frame.size


def repeat_neighbour_fraction(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    radius_nm: float,
    min_gap_frames: int = REPEAT_GAP_FRAMES,
    max_points: int = 200_000,
    seed: int = 0,
    z_scaled: Optional[NDArray[np.float64]] = None,
    stop: Optional[Any] = None,
) -> float:
    """
    Fraction of localizations with a neighbour within ``radius_nm`` taken
    at least ``min_gap_frames`` earlier or later.

    A docking site is revisited during a DNA-PAINT movie, so its
    localizations from different moments should land on top of each other.
    A drift correction that is wrong scrambles them: on the 15.07.26
    sample, AIM with 100-frame segments reported 1.3 um of drift that was
    not there and this fraction fell from 0.51 to 0.09. That is what it is
    for -- telling a file from its own corrected copy. Whether it also
    shows a correct correction helping depends on how sites are revisited:
    when each is revisited often, revisits close in time coincide even
    under drift and the fraction barely moves. It is not meant for
    comparing different samples.

    ``z_scaled`` makes the test 3D: pass z multiplied by the ratio of the
    lateral to the axial radius, so one sphere of ``radius_nm`` stands for
    the anisotropic neighbourhood. Without it, a scrambled z goes unseen.

    The gap keeps the frames of a single binding event from counting as a
    revisit. Above ``max_points`` the fraction is estimated on the
    localizations of a fixed pseudo-random set of FRAMES, still searched
    against every localization. Choosing by frame rather than by row keeps
    the same queries in a corrected copy that lost some rows.

    ``stop``, if given, is called between blocks of work; when it returns
    True the computation is abandoned with InterruptedError.

    Returns NaN when there is nothing to measure.
    """
    frame = np.asarray(frame, dtype=np.int64)
    columns = [np.asarray(x_nm, dtype=float), np.asarray(y_nm, dtype=float)]
    if z_scaled is not None:
        columns.append(np.asarray(z_scaled, dtype=float))
    points = np.column_stack(columns)
    finite = np.all(np.isfinite(points), axis=1)
    frame, points = frame[finite], points[finite]
    radius = float(radius_nm)
    if frame.size < 2 or not (np.isfinite(radius) and radius > 0):
        return float("nan")
    queries = np.nonzero(_frame_sample(frame, max_points, seed))[0]
    if queries.size == 0:
        return float("nan")

    def check_stop() -> None:
        if stop is not None and stop():
            raise InterruptedError("stopped")

    tree = cKDTree(points)
    k = min(_REPEAT_NEAREST, frame.size)
    revisited = 0
    undecided: List[int] = []
    for start in range(0, queries.size, 20_000):
        check_stop()
        chunk = queries[start:start + 20_000]
        dist, idx = tree.query(points[chunk], k=k,
                               distance_upper_bound=radius)
        dist = dist.reshape(chunk.size, k)
        idx = idx.reshape(chunk.size, k)
        found = np.isfinite(dist)
        safe = np.where(found, idx, 0)
        apart = found & (np.abs(frame[safe] - frame[chunk][:, None])
                         >= min_gap_frames)
        hit = apart.any(axis=1)
        revisited += int(np.count_nonzero(hit))
        # Every one of the k nearest is within reach and none is far in
        # time -- a long binding event, typically. The answer may lie
        # further out, so these few are settled exactly.
        undecided.extend(chunk[~hit & found[:, -1]].tolist())
    for n, i in enumerate(undecided):
        if n % 1000 == 0:
            check_stop()
        near = np.asarray(tree.query_ball_point(points[i], radius),
                          dtype=np.int64)
        if np.any(np.abs(frame[near] - frame[i]) >= min_gap_frames):
            revisited += 1
    return revisited / queries.size


# ===================================================================
#  How much of the axial range is actually used
# ===================================================================
@dataclass
class AxialCoverage:
    z_span_nm: float
    z_p1_nm: float
    z_p99_nm: float
    calibration_span_nm: Optional[float]
    periods_captured: Optional[float]
    warnings: List[str] = field(default_factory=list)


def axial_coverage(
    z_nm: NDArray[np.float64],
    calibration: Optional[Dict[str, Any]] = None,
    period_nm: Optional[float] = None,
) -> AxialCoverage:
    """
    The axial extent of the data, against the range the optics can fit
    and against the periodicity being looked for.

    ``periods_captured`` is the number that decides whether a given
    section thickness can answer a ring-to-ring question at all: two
    consecutive rings need at least two periods in the slab, and a
    correlation that decays over a couple of periods needs three or four
    to be measurable.

    The astigmatic calibration's own span comes from the z stack Picasso
    was calibrated on (``Number of frames`` x ``Step size in nm``);
    localizations fitted beyond it are extrapolations.
    """
    z = np.asarray(z_nm, dtype=float)
    warnings: List[str] = []
    low = float(np.percentile(z, 1))
    high = float(np.percentile(z, 99))
    span = high - low

    calibration_span: Optional[float] = None
    if calibration:
        try:
            calibration_span = (
                float(calibration["Number of frames"])
                * float(calibration["Step size in nm"])
            )
        except (KeyError, TypeError, ValueError):
            calibration_span = None

    if calibration_span and span > calibration_span:
        warnings.append(
            f"The data spans {span:.0f} nm in z but the astigmatism "
            f"calibration only covers {calibration_span:.0f} nm. "
            f"Localizations outside it are extrapolated, and their z and "
            f"lpz are not trustworthy."
        )

    periods: Optional[float] = None
    if period_nm and period_nm > 0:
        periods = span / float(period_nm)
        if periods < 2.0:
            warnings.append(
                f"The axial span holds about {periods:.1f} periods of "
                f"{period_nm:.0f} nm. A ring-to-ring comparison needs at "
                f"least two rings in the slab, so this acquisition cannot "
                f"answer it."
            )

    return AxialCoverage(
        z_span_nm=span,
        z_p1_nm=low,
        z_p99_nm=high,
        calibration_span_nm=calibration_span,
        periods_captured=periods,
        warnings=warnings,
    )


# ===================================================================
#  The whole report
# ===================================================================
@dataclass
class QualityReport:
    source: str
    n_locs: int
    precision: Optional[PrecisionCheck] = None
    box: Optional[BoxSizeCheck] = None
    axial: Optional[AxialResolvedness] = None
    drift: Optional[DriftCheck] = None
    coverage: Optional[AxialCoverage] = None
    missing: List[str] = field(default_factory=list)

    @property
    def warnings(self) -> List[str]:
        """Every finding, in the order the checks ran."""
        out: List[str] = []
        for part in (self.precision, self.box, self.axial,
                     self.drift, self.coverage):
            if part is not None:
                out.extend(part.warnings)
        return out


def quality_report(
    loc: Any,
    *,
    means_nm: Optional[Union[Sequence[float], NDArray[Any]]] = None,
    sigmas_nm: Optional[Union[Sequence[float], NDArray[Any]]] = None,
    weights: Optional[Union[Sequence[float], NDArray[Any]]] = None,
    period_nm: Optional[float] = None,
    run_drift: bool = True,
) -> QualityReport:
    """
    Run every check that this file has the columns for.

    Takes a ``tools.mps_io.Localizations``. A check whose input is
    missing is skipped and named in ``missing`` rather than silently
    omitted: "not measured" and "measured, fine" must not look the same.
    """
    report = QualityReport(source=loc.path, n_locs=loc.n)
    frame = loc.frame
    lpx, lpy, lpz = loc.lpx_nm, loc.lpy_nm, loc.lpz_nm

    if frame is None:
        report.missing.append(
            "no 'frame' column: NeNA and the drift check need it"
        )
    elif lpx is None or lpy is None:
        report.missing.append(
            "no 'lpx'/'lpy' columns: the reported precision cannot be "
            "compared against NeNA"
        )
    else:
        report.precision = precision_check(
            frame, loc.x_nm, loc.y_nm, lpx, lpy, lpz
        )

    sx, sy = loc.column("sx"), loc.column("sy")
    if sx is None or sy is None:
        report.missing.append("no 'sx'/'sy' columns: no box-size check")
    else:
        report.box = box_size_check(sx, sy, loc.box_size_px)

    if means_nm is None or sigmas_nm is None:
        report.missing.append(
            "no axial mixture supplied: run the Z periodicity fit first"
        )
    elif lpz is None:
        report.missing.append(
            "no 'lpz' column: a fitted axial width cannot be told apart "
            "from the axial precision"
        )
    else:
        report.axial = axial_resolvedness(
            loc.z_nm, lpz, means_nm, sigmas_nm, weights
        )

    if run_drift and frame is not None:
        report.drift = drift_check(
            frame, loc.x_nm, loc.y_nm, loc.z_nm if loc.is_3d else None
        )

    if loc.is_3d:
        report.coverage = axial_coverage(
            loc.z_nm, loc.z_calibration, period_nm
        )

    return report
