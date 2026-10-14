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
analysis ran aground on exactly that ambiguity: on pilot data many segment
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
from scipy.optimize import brentq, curve_fit, minimize, minimize_scalar
from scipy.spatial import cKDTree
from scipy.special import ndtr

# A component wider than this multiple of the localization precision is
# carrying structure, not just noise. 1.5 is the upper sigma bound G5M
# imposes on a molecule (picasso.g5m.MAX_SIGMA_FACTOR), i.e. the point
# past which the Jungmann lab stops calling a localization cloud a single
# emitter. Borrowed rather than invented, but it is a convention.
RESOLVED_SIGMA_FACTOR = 1.5

# Below this many next-frame neighbour pairs the NeNA fit is not worth
# reporting. The number itself was set by hand (ad hoc) when NeNA was first
# added; it is kept because a derivation agrees with it: for pairs that are
# all the same molecule seen twice, the maximum-likelihood scale of a
# Rayleigh law has a standard error of s / (2 sqrt(n)), so its 95 % interval
# is narrower than +-5 % once n >= (1.96 / (2 x 0.05))^2 = 384. Below that
# even a perfect file cannot give NeNA to 5 %. docs/nena.md, section 6.
MIN_NENA_PAIRS = 500

# --- the maximum-likelihood NeNA estimator (docs/nena.md) -----------------
# Width of the bins the likelihood is evaluated on. The likelihood uses the
# exact probability of each bin (differences of the two components' CDFs),
# so binning only loses the position of a distance inside 0.05 nm -- 1/60 of
# the smallest precision this module expects (3 nm). Derivation, not a fit
# setting: the result does not depend on the 0.5 nm display histogram.
NENA_ML_BIN_NM = 0.05
# 95 % quantile of chi-square with 1 degree of freedom: the profile-likelihood
# interval is where 2 (logL_max - logL_profile(s)) <= 3.841 (Wilks 1938).
NENA_CHI2_95 = 3.841458820694124
# Picasso's constraint a >= ac, restated for components normalized on the
# fitted range: the same-molecule term carries at least half of the pairs.
NENA_MIN_SAME_FRACTION = 0.5
# Lower bound on the "nearby" Gaussian's width, in units of s. The distance
# between localizations of two DIFFERENT molecules a fixed distance apart
# spreads radially by the two localization errors, i.e. sqrt(2) s; a spread
# of separations only widens it. A narrower "nearby" term is not a
# population of neighbours -- it is the fit carving the same-molecule peak.
# Derivation (docs/nena.md, section 3).
NENA_NEARBY_MIN_SD_FACTOR = float(np.sqrt(2.0))
# The nearby term costs three parameters (its weight, centre and width). It
# is kept only when it raises the log-likelihood by more than the Bayesian
# information criterion's price, 3/2 ln(n) (Schwarz 1978); otherwise the
# fit is Picasso's model with ac = 0 -- inside Picasso's own bounds -- and
# the nearby term is not allowed to carve the peak's shoulder out of noise.
# Without this the likelihood fit, like the least-squares one, came out
# 5 % low or more on pure same-molecule data (docs/nena.md, section 2).
NENA_NEARBY_PARAMETERS = 3
# Agreement check: the estimate with an extra random-neighbour term must lie
# inside NeNA's 95 % interval OR within 5 % of it. 5 % is the same tolerance
# the validation counts as a failure (own simulation, docs/sims/nena).
NENA_AGREEMENT_TOLERANCE = 0.05
# A 95 % interval wider than +-10 % of the value means the fit is poorly
# identified (twice the 5 % tolerance; ad hoc, documented).
NENA_MAX_CI_HALF_WIDTH = 0.10
# Share of the close pairs (d < 2 sqrt(2) s) that also have a SECOND
# localization within that distance in the next frame. Above it the
# nearest-neighbour rule itself starts picking another molecule over the
# molecule's own next localization, and NeNA reads low whatever the fit.
# Own simulation (docs/sims/nena/nena_grid.py, 1160 files, seeds in the
# script): below 0.15 at most 2 % of the files read more than 5 % low; from
# 0.15 to 0.20, 20 % did (median -2 %); above 0.20, 70-76 % did (median
# -6 to -10 %).
NENA_CROWDING_LIMIT = 0.15
# Share of the same-molecule distance law beyond max_distance_nm above which
# the window is too short (the likelihood accounts for the cut; it just has
# little left to measure).
NENA_TRUNCATION_LIMIT = 0.10


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

    ``precision_nm`` is Picasso's two-component model fitted by maximum
    likelihood (docs/nena.md). The added fields describe how well that one
    number is pinned down; none of them replaces it.
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
    # --- added with the maximum-likelihood estimator (all optional) ---
    method: str = ""
    ci_low_nm: float = float("nan")      # 95 % profile-likelihood interval
    ci_high_nm: float = float("nan")
    same_molecule_fraction: float = float("nan")   # weight of the peak
    nearby_centre_nm: float = float("nan")
    nearby_sd_nm: float = float("nan")
    # The same fit with a third term for randomly placed neighbours: the
    # independent estimate the agreement check compares against.
    random_term_nm: float = float("nan")
    # Picasso's own least-squares fit of the same model on the 0.5 nm
    # histogram (the estimator used before 2026-10-14), for comparison.
    lsq_nm: float = float("nan")
    crowding_fraction: float = float("nan")
    well_identified: bool = False
    flags: List[str] = field(default_factory=list)
    nearby_term_used: bool = False


def _next_frame_neighbours_nm(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    max_distance_nm: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Distance from each localization to its nearest and second-nearest
    neighbour in the NEXT frame, for the localizations whose nearest one is
    closer than ``max_distance_nm``. The second is ``inf`` when the next
    frame has a single localization.
    """
    order = np.argsort(frame, kind="stable")
    frame, x_nm, y_nm = frame[order], x_nm[order], y_nm[order]
    unique, start = np.unique(frame, return_index=True)
    end = np.append(start[1:], frame.size)
    bounds = {int(f): (int(s), int(e)) for f, s, e in zip(unique, start, end)}

    first_chunks: List[NDArray[np.float64]] = []
    second_chunks: List[NDArray[np.float64]] = []
    for value, first, last in zip(unique, start, end):
        following = bounds.get(int(value) + 1)
        if following is None:
            continue
        n_first, n_last = following
        if n_last <= n_first or last <= first:
            continue
        tree = cKDTree(np.column_stack([x_nm[n_first:n_last],
                                        y_nm[n_first:n_last]]))
        k = 2 if n_last - n_first >= 2 else 1
        distances, _ = tree.query(
            np.column_stack([x_nm[first:last], y_nm[first:last]]), k=k
        )
        distances = np.asarray(distances, dtype=float).reshape(last - first, k)
        first_chunks.append(distances[:, 0])
        second_chunks.append(
            distances[:, 1] if k == 2 else np.full(last - first, np.inf)
        )

    if not first_chunks:
        empty = np.array([], dtype=float)
        return empty, empty
    nearest = np.concatenate(first_chunks)
    second = np.concatenate(second_chunks)
    keep = nearest < max_distance_nm
    return nearest[keep], second[keep]


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
    return _next_frame_neighbours_nm(frame, x_nm, y_nm, max_distance_nm)[0]


def _picasso_model(
    d: NDArray[np.float64], delta_a: float, s: float, ac: float, dc: float,
    sc: float,
) -> NDArray[np.float64]:
    """Picasso's NeNA model on a histogram (``picasso.postprocess.nena``)."""
    a = ac + delta_a  # keeps a >= ac, as in Picasso
    single = a * (d / (2.0 * s**2)) * np.exp(-(d**2) / (4.0 * s**2))
    nearby = (
        ac / (sc * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((d - dc) / sc) ** 2)
    )
    return np.asarray(single + nearby, dtype=float)


def _picasso_lsq(
    centres: NDArray[np.float64],
    counts: NDArray[np.float64],
    max_distance_nm: float,
    reported_lp_nm: Optional[float],
) -> float:
    """
    The estimator used before 2026-10-14, kept for comparison: Picasso's
    least-squares fit of its model on the histogram, three starting points,
    the smallest residual kept. It can settle on a narrower peak while the
    "nearby" term takes the peak's shoulder (docs/nena.md, section 2).
    """
    area = float(np.trapezoid(counts, centres))
    seed = float(reported_lp_nm) if reported_lp_nm else max_distance_nm / 12.0
    peak = float(centres[int(np.argmax(counts))])
    starts = [
        [0.8 * area, seed, 0.1 * area, 2.0 * seed, seed],
        [0.8 * area, peak / np.sqrt(2.0), 0.1 * area,
         0.7 * centres[-1], 0.3 * centres[-1]],
        [0.8 * area, seed, 0.1 * area, 0.5 * centres[-1], seed],
    ]
    best = float("nan")
    best_residual = np.inf
    for start in starts:
        if not np.all(np.isfinite(start)):
            continue
        try:
            candidate, _ = curve_fit(
                _picasso_model, centres, counts, p0=start,
                bounds=([0.0] * 5, [np.inf] * 5), maxfev=20000,
            )
        except (RuntimeError, ValueError):
            continue
        residual = float(np.sum(
            (_picasso_model(centres, *candidate) - counts) ** 2))
        if residual < best_residual:
            best_residual, best = residual, float(candidate[1])
    return best


def _rayleigh_cdf(d: Any, s: float) -> Any:
    """CDF of the same-molecule distance: Rayleigh law of scale sqrt(2) s."""
    return -np.expm1(-(np.asarray(d, dtype=float) ** 2) / (4.0 * s * s))


class _NeNALikelihood:
    """
    Multinomial likelihood of the next-frame distances on [0, D), with
    every component normalized on that range:

        w * Rayleigh(s)  +  (1 - w) * Gaussian(dc, sc)            (Picasso)
        w * Rayleigh(s)  +  (1 - w) * [u * Rayleigh(t)
                                        + (1 - u) * Gaussian(dc, sc)]

    The second adds the nearest-neighbour law of randomly placed
    localizations (2 pi rho d exp(-pi rho d^2), a Rayleigh law of its own
    with t >= 2 s): the independent estimate of the agreement check.
    """

    def __init__(self, distances: NDArray[np.float64], max_distance_nm: float):
        self.d_max = float(max_distance_nm)
        edges = np.arange(0.0, self.d_max, NENA_ML_BIN_NM)
        edges = np.append(edges, self.d_max)
        counts, _ = np.histogram(distances, bins=edges)
        used = counts > 0
        self.lo = edges[:-1][used]
        self.hi = edges[1:][used]
        self.n = counts[used].astype(float)
        self.n_total = float(self.n.sum())
        cumulative = np.cumsum(self.n)
        self.median = float(
            self.hi[int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))])

    # ------------------------------------------------ component masses
    def _rayleigh_mass(self, s: float) -> NDArray[np.float64]:
        total = float(_rayleigh_cdf(self.d_max, s))
        return np.asarray(
            (_rayleigh_cdf(self.hi, s) - _rayleigh_cdf(self.lo, s))
            / max(total, 1e-300), dtype=float)

    def _gauss_mass(self, dc: float, sc: float) -> NDArray[np.float64]:
        total = float(ndtr((self.d_max - dc) / sc) - ndtr(-dc / sc))
        return np.asarray(
            (ndtr((self.hi - dc) / sc) - ndtr((self.lo - dc) / sc))
            / max(total, 1e-300), dtype=float)

    def _nll(self, p: NDArray[np.float64]) -> float:
        return -float(np.sum(self.n * np.log(np.maximum(p, 1e-300))))

    def nll2(self, s: float, w: float, dc: float, sc: float) -> float:
        p = w * self._rayleigh_mass(s)
        if w < 1.0:
            p = p + (1.0 - w) * self._gauss_mass(dc, sc)
        return self._nll(p)

    def nll3(self, s: float, w: float, u: float, t: float, dc: float,
             sc: float) -> float:
        rest = (u * self._rayleigh_mass(t)
                + (1.0 - u) * self._gauss_mass(dc, sc))
        return self._nll(w * self._rayleigh_mass(s) + (1.0 - w) * rest)

    # ------------------------------------------------ display
    def cdf2(self, d: NDArray[np.float64], s: float, w: float, dc: float,
             sc: float) -> NDArray[np.float64]:
        ray = _rayleigh_cdf(d, s) / max(float(_rayleigh_cdf(self.d_max, s)),
                                        1e-300)
        low = float(ndtr(-dc / sc))
        total = float(ndtr((self.d_max - dc) / sc)) - low
        gauss = (ndtr((np.asarray(d) - dc) / sc) - low) / max(total, 1e-300)
        return np.asarray(w * ray + (1.0 - w) * gauss, dtype=float)


def _nena_fit_two(
    lik: _NeNALikelihood,
    starts: Sequence[Tuple[float, float, float, float]],
    s_fixed: Optional[float] = None,
) -> Tuple[float, Optional[Tuple[float, float, float, float]]]:
    """
    Minimize the two-component negative log-likelihood from every start and
    keep the smallest. Parameters inside the optimizer are scaled to order
    one; sc is s x (sqrt(2) + rho) so the nearby term can never be narrower
    than NENA_NEARBY_MIN_SD_FACTOR x s. With ``s_fixed`` s is held (the
    profile).
    """
    d_max = lik.d_max
    s_low, s_high = max(1e-3 * d_max, 0.05), 0.5 * d_max
    best_nll, best = np.inf, None
    for s0, w0, dc0, sc0 in starts:
        s_start = s_fixed if s_fixed is not None else s0
        rho0 = min(max(sc0 / s_start - NENA_NEARBY_MIN_SD_FACTOR, 0.05), 49.0)
        w0 = min(max(w0, NENA_MIN_SAME_FRACTION), 1.0)
        dc0 = min(max(dc0, 0.0), 3.0 * d_max)

        def unpack(v: NDArray[np.float64]) -> Tuple[float, float, float, float]:
            s = s_fixed if s_fixed is not None else float(v[0]) * d_max
            return (s, float(v[1]), float(v[2]) * d_max,
                    s * (NENA_NEARBY_MIN_SD_FACTOR + 10.0 * float(v[3])))

        def objective(v: NDArray[np.float64]) -> float:
            return lik.nll2(*unpack(v))

        x0 = np.array([s_start / d_max, w0, dc0 / d_max, rho0 / 10.0])
        s_bounds = ((s_fixed / d_max, s_fixed / d_max) if s_fixed is not None
                    else (s_low / d_max, s_high / d_max))
        bounds = [s_bounds, (NENA_MIN_SAME_FRACTION, 1.0), (0.0, 3.0),
                  (0.0, 5.0)]
        x0[0] = min(max(x0[0], s_bounds[0]), s_bounds[1])
        try:
            found = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(found.fun) and found.fun < best_nll:
            best_nll, best = float(found.fun), unpack(found.x)
    return best_nll, best


def _nena_starts(
    lik: _NeNALikelihood, reported_lp_nm: Optional[float],
) -> List[Tuple[float, float, float, float]]:
    """Starting points: four seeds for s (the median of a pure Rayleigh law
    is 1.665 s) x two weights x three placements of the nearby term."""
    d_max = lik.d_max
    s_med = lik.median / 1.665
    seeds = [0.6 * s_med, s_med, 1.4 * s_med]
    if reported_lp_nm and np.isfinite(reported_lp_nm) and reported_lp_nm > 0:
        seeds.append(float(reported_lp_nm))
    out: List[Tuple[float, float, float, float]] = []
    for s0 in seeds:
        for w0 in (0.95, 0.7):
            for dc0, sc0 in ((3.0 * s0, 2.0 * s0), (0.5 * d_max, 0.25 * d_max),
                             (d_max, d_max / 3.0)):
                out.append((s0, w0, dc0, sc0))
    return out


def _nena_profile_interval(
    lik: _NeNALikelihood,
    best_nll: float,
    best: Tuple[float, float, float, float],
) -> Tuple[float, float, bool, bool, Optional[Tuple[float, Tuple[float, float, float, float]]]]:
    """
    95 % profile-likelihood interval for s: the values where the
    negative log-likelihood, minimized over the nuisance parameters, rises
    NENA_CHI2_95 / 2 above its minimum. Returns (low, high, low_open,
    high_open, better) where ``better`` is a fit with a smaller
    negative log-likelihood met on the way (the global search missed it).
    """
    s_hat = best[0]
    d_max = lik.d_max
    s_low, s_high = max(1e-3 * d_max, 0.05), 0.5 * d_max
    target = 0.5 * NENA_CHI2_95
    nuisance = [(best[0], best[1], best[2], best[3])]

    def profile(s: float) -> float:
        starts = [(s, w, dc, sc / best[0] * s) for _, w, dc, sc in nuisance[-2:]]
        starts.append((s, 1.0, best[2], best[3]))
        value, params = _nena_fit_two(lik, starts, s_fixed=s)
        if params is not None:
            nuisance.append(params)
            if value < best_nll - 1e-6:
                better.append((value, params))
        return value - best_nll

    better: List[Tuple[float, Tuple[float, float, float, float]]] = []
    step0 = 2.0 * s_hat / np.sqrt(max(lik.n_total, 1.0))
    limits: List[float] = []
    opens: List[bool] = []
    for direction, wall in ((-1.0, s_low), (1.0, s_high)):
        del nuisance[1:]
        inner, step, edge = s_hat, step0, None
        for _ in range(40):
            trial = s_hat + direction * step
            if (trial - wall) * direction >= 0.0:
                trial = wall
            rise = profile(trial)
            if better:
                return float("nan"), float("nan"), True, True, better[-1]
            if rise >= target:
                edge = trial
                break
            if trial == wall:
                break
            inner, step = trial, step * 1.6
        if edge is None:
            limits.append(wall)
            opens.append(True)
            continue
        a, b = inner, edge
        for _ in range(30):            # bisection on the crossing
            mid = 0.5 * (a + b)
            if profile(mid) >= target:
                b = mid
            else:
                a = mid
            if abs(b - a) < 1e-3 * s_hat:
                break
        limits.append(0.5 * (a + b))
        opens.append(False)
    return limits[0], limits[1], opens[0], opens[1], None


def _nena_random_term(
    lik: _NeNALikelihood, best: Tuple[float, float, float, float],
) -> float:
    """s of the three-component model (same molecule + random neighbours +
    nearby Gaussian), started from the two-component fit."""
    d_max = lik.d_max
    s_hat, w_hat, dc_hat, sc_hat = best
    s_low, s_high = max(1e-3 * d_max, 0.05), 0.5 * d_max
    starts = [
        (s_hat, w_hat, 0.5, d_max, dc_hat, sc_hat),
        (s_hat, min(1.0, w_hat + 0.05), 0.9, 0.5 * d_max, dc_hat, sc_hat),
        (s_hat, max(0.0, w_hat - 0.2), 0.8, d_max, 0.5 * d_max, 0.25 * d_max),
        (lik.median / 1.665, 0.7, 0.5, d_max, 0.5 * d_max, 0.25 * d_max),
    ]
    best_nll, best_s = np.inf, float("nan")
    for s0, w0, u0, t0, dc0, sc0 in starts:
        s0 = min(max(s0, s_low), s_high)

        def unpack(v: NDArray[np.float64]) -> Tuple[float, float, float, float, float, float]:
            s = float(v[0]) * d_max
            return (s, float(v[1]), float(v[2]),
                    s * (2.0 + 10.0 * float(v[3])), float(v[4]) * d_max,
                    s * (NENA_NEARBY_MIN_SD_FACTOR + 10.0 * float(v[5])))

        def objective(v: NDArray[np.float64]) -> float:
            return lik.nll3(*unpack(v))

        x0 = np.array([
            s0 / d_max, w0, u0,
            min(max(t0 / s0 - 2.0, 0.01), 999.0) / 10.0,
            min(max(dc0, 0.0), 3.0 * d_max) / d_max,
            min(max(sc0 / s0 - NENA_NEARBY_MIN_SD_FACTOR, 0.05), 49.0) / 10.0,
        ])
        bounds = [(s_low / d_max, s_high / d_max), (0.0, 1.0), (0.0, 1.0),
                  (0.0, 100.0), (0.0, 3.0), (0.0, 5.0)]
        try:
            found = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(found.fun) and found.fun < best_nll:
            best_nll, best_s = float(found.fun), unpack(found.x)[0]
    return best_s


def _nena_fit_rayleigh(lik: _NeNALikelihood) -> Tuple[float, float]:
    """Maximum likelihood of the same-molecule term alone (ac = 0)."""
    d_max = lik.d_max
    found = minimize_scalar(
        lambda s: lik.nll2(float(s), 1.0, 0.0, 1.0),
        bounds=(max(1e-3 * d_max, 0.05), 0.5 * d_max), method="bounded",
        options={"xatol": 1e-5 * d_max},
    )
    return float(found.fun), float(found.x)


def _nena_rayleigh_interval(
    lik: _NeNALikelihood, nll_hat: float, s_hat: float,
) -> Tuple[float, float, bool, bool]:
    """95 % likelihood interval for s when the nearby term is not used (a
    one-parameter likelihood: no nuisance to profile)."""
    d_max = lik.d_max
    s_low, s_high = max(1e-3 * d_max, 0.05), 0.5 * d_max
    target = 0.5 * NENA_CHI2_95

    def rise(s: float) -> float:
        return lik.nll2(s, 1.0, 0.0, 1.0) - nll_hat

    out: List[float] = []
    opens: List[bool] = []
    step0 = 2.0 * s_hat / np.sqrt(max(lik.n_total, 1.0))
    for direction, wall in ((-1.0, s_low), (1.0, s_high)):
        inner, step, edge = s_hat, step0, None
        for _ in range(60):
            trial = s_hat + direction * step
            if (trial - wall) * direction >= 0.0:
                trial = wall
            if rise(trial) >= target:
                edge = trial
                break
            if trial == wall:
                break
            inner, step = trial, step * 1.6
        if edge is None:
            out.append(wall)
            opens.append(True)
            continue
        found = brentq(lambda s: rise(s) - target, min(inner, edge),
                       max(inner, edge), xtol=1e-4 * s_hat)
        out.append(float(found))
        opens.append(False)
    return out[0], out[1], opens[0], opens[1]


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
    molecules. ``s`` is the precision.

    Since 2026-10-14 the model is fitted by maximum likelihood on the
    distances themselves (exact bin probabilities on 0.05 nm bins), from 24
    starting points, with the nearby term no narrower than sqrt(2) s, and
    the profile likelihood double-checks that the optimum is the global
    one. Picasso's least-squares fit on the histogram settled on a peak
    5-20 % too narrow in 5 of 20 synthetic files of known precision
    (docs/nena.md, section 2); its value is still reported as ``lsq_nm``.

    Next to the value: a 95 % profile-likelihood interval, the same fit
    with a random-neighbour term as an independent estimate, and how
    crowded the next frame is. A warning names each problem; the value is
    never replaced by another estimate.

    Parameters
    ----------
    max_distance_nm : distances beyond this are dropped before fitting.
        Picasso uses 1 camera pixel; expressed in nm here so the value
        does not silently change meaning with the camera.
    bin_nm : width of the histogram that is DRAWN (and that Picasso's
        least-squares comparison is fitted on); the estimate does not use
        it.
    reported_lp_nm : the fit's own precision, used only as one more
        starting point.
    """
    warnings: List[str] = []
    distances, second = _next_frame_neighbours_nm(
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
            flags=["too_few_pairs"],
        )

    edges = np.arange(0.0, max_distance_nm + bin_nm, bin_nm)
    counts, edges = np.histogram(distances, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = counts.astype(float)
    lsq = _picasso_lsq(centres, counts, max_distance_nm, reported_lp_nm)

    lik = _NeNALikelihood(distances, max_distance_nm)
    best_nll, best = _nena_fit_two(lik, _nena_starts(lik, reported_lp_nm))
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
            method="maximum likelihood (Picasso model)",
            lsq_nm=lsq,
            flags=["no_fit"],
        )

    # Is the nearby term worth its three parameters? (BIC, Schwarz 1978.)
    ray_nll, ray_s = _nena_fit_rayleigh(lik)
    gain = 2.0 * (ray_nll - best_nll)
    use_nearby = gain > NENA_NEARBY_PARAMETERS * np.log(lik.n_total)

    low = high = float("nan")
    low_open = high_open = True
    if use_nearby:
        # The profile walks away from the optimum; if it ever finds a
        # better one the global search missed it, so restart from there.
        for _ in range(4):
            low, high, low_open, high_open, better = _nena_profile_interval(
                lik, best_nll, best)
            if better is None:
                break
            refit_nll, refit = _nena_fit_two(lik, [better[1]])
            if refit is None or refit_nll >= better[0]:
                best_nll, best = better
            else:
                best_nll, best = refit_nll, refit
    else:
        best_nll = ray_nll
        best = (ray_s, 1.0, float("nan"), float("nan"))
        low, high, low_open, high_open = _nena_rayleigh_interval(
            lik, ray_nll, ray_s)

    s, w, dc, sc = best
    flags: List[str] = []
    converged = True
    s_ceiling = 0.5 * max_distance_nm
    if s >= 0.99 * s_ceiling:
        converged = False
        flags.append("no_peak")
        warnings.append(
            f"NeNA found no same-molecule peak inside {max_distance_nm:g} nm "
            f"(the fit ran to its limit of {s_ceiling:g} nm). Is this file "
            f"linked (Picasso Link merges the consecutive-frame pairs NeNA "
            f"needs), or is the field so dense that random neighbours swamp "
            f"the peak?"
        )

    random_term = _nena_random_term(
        lik, best if use_nearby else (s, 0.95, 3.0 * s, 2.0 * s))
    tail = float(np.exp(-(max_distance_nm**2) / (4.0 * s * s)))
    close = distances < 2.0 * np.sqrt(2.0) * s
    crowding = (float(np.mean(second[close] < 2.0 * np.sqrt(2.0) * s))
                if np.any(close) else float("nan"))

    if converged:
        half = max(s - low, high - s) / s if np.isfinite(low + high) else np.inf
        if low_open or high_open or half > NENA_MAX_CI_HALF_WIDTH:
            flags.append("wide_interval")
            warnings.append(
                f"NeNA {s:.1f} nm is poorly pinned down: its 95 % interval "
                f"is {low:.1f}-{high:.1f} nm"
                + (" and does not close inside the fitted range"
                   if (low_open or high_open) else
                   f" (more than +-{100 * NENA_MAX_CI_HALF_WIDTH:.0f} %)")
                + f", from {distances.size:,} next-frame pairs. Treat any "
                  f"threshold derived from it as uncertain by as much."
            )
        if w <= NENA_MIN_SAME_FRACTION + 0.01:
            flags.append("nearby_dominates")
            warnings.append(
                f"In the NeNA fit the nearby-molecule term carries half of "
                f"the next-frame pairs, the most Picasso's model allows "
                f"(a >= ac). The same-molecule peak is not separated from "
                f"the background; NeNA {s:.1f} nm may be too large. A "
                f"sparser field of view, or a file with fewer background "
                f"localizations, gives a cleaner peak."
            )
        if np.isfinite(random_term):
            outside = not (low <= random_term <= high)
            if outside and abs(random_term - s) > NENA_AGREEMENT_TOLERANCE * s:
                flags.append("disagreement")
                warnings.append(
                    f"NeNA {s:.1f} nm (95 % interval {low:.1f}-{high:.1f}) "
                    f"and the same fit with a term for randomly placed "
                    f"neighbours ({random_term:.1f} nm) disagree by "
                    f"{100 * abs(random_term - s) / s:.0f} %. The background "
                    f"of the next-frame distances is not the shape the model "
                    f"assumes; check the fitted curve in Data quality > "
                    f"Precision before using either value (docs/nena.md)."
                )
        if np.isfinite(crowding) and crowding > NENA_CROWDING_LIMIT:
            flags.append("crowded")
            warnings.append(
                f"{100 * crowding:.0f} % of the close next-frame pairs have a "
                f"second localization just as close. The nearest-neighbour "
                f"rule then often picks another molecule instead of the same "
                f"one, and NeNA reads LOW whatever the fit (on synthetic "
                f"files above {100 * NENA_CROWDING_LIMIT:.0f} %, one in five "
                f"read more than 5 % low; above 20 %, most did). A sparser "
                f"region or file measures it better."
            )
        if tail > NENA_TRUNCATION_LIMIT:
            flags.append("window_short")
            warnings.append(
                f"{100 * tail:.0f} % of the same-molecule distances at "
                f"NeNA {s:.1f} nm fall beyond the {max_distance_nm:g} nm "
                f"window. The fit accounts for the cut, but a longer window "
                f"would measure it better."
            )

    # The fitted curve, on the drawn histogram's bins: the expected count of
    # each bin under the fitted law.
    fitted = lik.n_total * np.diff(lik.cdf2(
        edges.astype(float), s, w, dc if use_nearby else 0.0,
        sc if use_nearby else 1.0))
    well = converged and not (
        {"wide_interval", "nearby_dominates", "disagreement"} & set(flags))
    return NeNAResult(
        precision_nm=float(s),
        n_pairs=int(distances.size),
        converged=converged,
        distances_nm=centres,
        histogram=counts,
        best_fit=np.asarray(fitted, dtype=float),
        warnings=warnings,
        method=("maximum likelihood, Picasso model"
                + (" with its nearby term" if use_nearby
                   else " without its nearby term (BIC)")),
        ci_low_nm=float(low),
        ci_high_nm=float(high),
        same_molecule_fraction=float(w),
        nearby_centre_nm=float(dc),
        nearby_sd_nm=float(sc),
        random_term_nm=float(random_term),
        lsq_nm=float(lsq),
        crowding_fraction=crowding,
        well_identified=bool(well),
        flags=flags,
        nearby_term_used=bool(use_nearby),
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
    # The NeNA measurement itself (interval, fitted curve, diagnostics), so
    # that a window drawing it does not fit it a second time.
    nena_ci_low_nm: float = float("nan")
    nena_ci_high_nm: float = float("nan")
    nena: Optional[NeNAResult] = None


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
        nena_ci_low_nm=result.ci_low_nm,
        nena_ci_high_nm=result.ci_high_nm,
        nena=result,
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
    density valley at many boundaries of pilot data. Two rings 190 nm
    apart, each of width lpz = 50 nm, are nearly 4 sigma apart and would show a
    deep valley; the valleys were absent because the fitted components
    came out clearly wider than lpz, and nothing could say why.

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
    A drift correction that is wrong scrambles them: on an unpublished
    DNA-PAINT sample, AIM with 100-frame segments reported drift of the
    order of a micrometre that was not there and this fraction fell
    several-fold. That is what it is
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
    return float(revisited / queries.size)


# ===================================================================
#  How much of the axial range is actually used
# ===================================================================
@dataclass
class AxialCoverage:
    z_span_nm: float
    z_p1_nm: float
    z_p99_nm: float
    # The nominal calibrated range, in the file's z units: (n_steps - 1) x
    # step x m (``tools.mps_zcalibration``). None when the record does not
    # give it -- never a constant.
    calibration_span_nm: Optional[float]
    periods_captured: Optional[float]
    warnings: List[str] = field(default_factory=list)
    # What the calibration record says about the data, each None when the
    # record (or m) is missing.
    record_found: bool = False
    magnification: Optional[float] = None
    magnification_origin: str = "blank"
    calibrated_min_nm: Optional[float] = None
    calibrated_max_nm: Optional[float] = None
    fit_bound_nm: Optional[float] = None
    fraction_piled: Optional[float] = None
    fraction_outer: Optional[float] = None
    d_zcalib_rms: Optional[float] = None
    filter_range: Optional[float] = None
    notes: List[str] = field(default_factory=list)


def axial_coverage(
    z_nm: NDArray[np.float64],
    calibration: Any = None,
    period_nm: Optional[float] = None,
    *,
    magnification: Optional[float] = None,
    magnification_origin: str = "",
    d_zcalib: Optional[NDArray[Any]] = None,
) -> AxialCoverage:
    """
    The axial extent of the data, against the range the calibration
    covered and against the periodicity being looked for.

    ``periods_captured`` is the number that decides whether a given
    section thickness can answer a ring-to-ring question at all: two
    consecutive rings need at least two periods in the slab (a published
    periodicity analysis used regions holding a few periods, Barabas et
    al. 2017), so 2 is the floor, a convention.

    ``calibration`` is the measurement's calibration record
    (``tools.mps_zcalibration.CalibrationRecord``) or, as before, the
    'Z Calibration' dictionary of the file. From it (Picasso's own
    arithmetic, ``picasso.zfit``): the NOMINAL calibrated range
    +/-(n_steps - 1) x step x m / 2 in the file's units -- nominal because
    Picasso does not store the z_zero it centred on; the fit bounds
    +/-1000 x m, at which a localization beyond the curves is piled rather
    than extrapolated; the fraction piled there (within 1 nm), the
    fraction beyond 80 % of the calibrated half-range, and the rms of
    d_zcalib with the 'Filter range' applied. The magnification factor is
    the record's, or the one given (typed) when the record has none; with
    no m every derived range is blank. The comparison of the data's span
    with 'Number of frames' x 'Step size' that stood here could not fire
    for a +/-1000 nm stack with m <= 1, and overstated the stack by
    'Frames per step'.
    """
    from tools import mps_zcalibration as zcal

    z = np.asarray(z_nm, dtype=float)
    warnings: List[str] = []
    low = float(np.percentile(z, 1))
    high = float(np.percentile(z, 99))
    span = high - low

    record = calibration
    if isinstance(calibration, dict):
        record = zcal.record_from_mapping(
            calibration, layout=zcal.LAYOUT_Z_CALIBRATION,
            origin=zcal.ORIGIN_FILE)
    out = AxialCoverage(z_span_nm=span, z_p1_nm=low, z_p99_nm=high,
                        calibration_span_nm=None, periods_captured=None)
    m: Optional[float] = None
    if record is not None:
        out.record_found = True
        if record.magnification is not None:
            m, out.magnification_origin = record.magnification, record.origin
        elif magnification is not None:
            m = float(magnification)
            out.magnification_origin = magnification_origin or "user"
        out.magnification = m
        out.filter_range = record.filter_range
        nominal = record.nominal_bounds_nm(m)
        if nominal is not None:
            out.calibrated_min_nm, out.calibrated_max_nm = nominal
            out.calibration_span_nm = nominal[1] - nominal[0]
        if m is not None:
            bound = zcal.FIT_BOUND_NM * m
            out.fit_bound_nm = bound
            finite = z[np.isfinite(z)]
            if finite.size:
                out.fraction_piled = float(np.mean(
                    np.abs(finite) >= bound - zcal.PILE_TOLERANCE_NM))
                if nominal is not None:
                    half = 0.5 * (nominal[1] - nominal[0])
                    out.fraction_outer = float(np.mean(
                        np.abs(finite) > zcal.OUTER_FRACTION * half))
        notes = [f"z calibration record: {record.layout} "
                 f"({record.file_name or 'name not recorded'}); m = "
                 + ("- (not recorded)" if m is None else
                    f"{m:g} ({zcal.SOURCE_WORDS.get(out.magnification_origin, out.magnification_origin)})")]
        if nominal is not None:
            notes.append(f"nominal calibrated range {nominal[0]:+.0f} to "
                         f"{nominal[1]:+.0f} nm (nominal: Picasso does "
                         f"not store the z_zero it centred on)")
        if out.fit_bound_nm is not None:
            notes.append(f"fit bounds +/-{out.fit_bound_nm:.0f} nm; "
                         f"{100 * (out.fraction_piled or 0.0):.2f} % of "
                         f"the localizations piled there")
        if out.fraction_outer is not None:
            notes.append(f"{100 * out.fraction_outer:.1f} % beyond "
                         f"{zcal.OUTER_FRACTION:.0%} of the calibrated "
                         f"half-range")
        out.notes = notes
    else:
        out.notes = ["no z calibration record: the calibrated range, the "
                     "fit bounds and m are not known"]
    if d_zcalib is not None:
        d = np.asarray(d_zcalib, dtype=float)
        d = d[np.isfinite(d)]
        if d.size:
            out.d_zcalib_rms = float(np.sqrt(np.mean(d ** 2)))
            out.notes.append(
                f"d_zcalib rms {out.d_zcalib_rms:.3g}"
                + ("" if out.filter_range is None else
                   f" (filtered at {out.filter_range:g} x rms)"))

    calibration_span = out.calibration_span_nm
    if calibration_span and span > calibration_span:
        warnings.append(
            f"The data spans {span:.0f} nm in z but the astigmatism "
            f"calibration only covers {calibration_span:.0f} nm (nominal). "
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
    out.periods_captured = periods
    out.warnings = warnings
    return out


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
    z_record: Any = None,
    z_magnification: Optional[float] = None,
    z_magnification_origin: str = "",
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
        # The record the file carries (either layout), unless the caller
        # gives the one chosen for this measurement.
        from tools import mps_zcalibration as zcal

        record = z_record if z_record is not None else zcal.record_from_info(
            getattr(loc, "info", None) or [])
        report.coverage = axial_coverage(
            loc.z_nm, record, period_nm, magnification=z_magnification,
            magnification_origin=z_magnification_origin,
            d_zcalib=loc.column("d_zcalib"),
        )

    return report
