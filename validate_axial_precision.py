# -*- coding: utf-8 -*-
"""
Checks for tools/mps_axial_precision.py, against synthetic ground truth.

The axial precision this module reports is the number every leak estimate
downstream rests on, and it is one where a wrong answer looks fine:
Picasso's lpz can come out much narrower than the fitted ring
components, and nothing in a table of lpz values says so. Every quantity
here is therefore checked against data built with the answer known in
advance:

  * axial NeNA per stratum, against emitters simulated with an axial
    error that CHANGES with depth (40 nm below z = 0, 70 nm above) and an
    lpz column deliberately 0.6x too optimistic;
  * the robust estimate, against the same data with 5 % of the
    frame-to-frame differences turned into +-500 nm outliers;
  * the pair rule (consecutive frames of one event, stratum of the pair's
    mean z), against a hand-made movie where the arithmetic is exact;
  * leak_fraction, against the closed form and against Jensen's
    inequality;
  * spurious_cluster_probability, against scipy's binomial tail;
  * structural_width_nm, against the arithmetic.

Every tolerance derives from a stated argument (in the check name or in a
comment next to it), never from "looks fine". Seed 0 throughout; no data
files are needed.

Run:  venv\\Scripts\\python.exe validate_axial_precision.py

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from itertools import product
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.stats import binom, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_axial_precision import (  # noqa: E402
    AxialNena,
    axial_nena,
    leak_fraction,
    spurious_cluster_probability,
    structural_width_nm,
)

PASSED = 0
FAILED = 0


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


# ======================================================== the synthetic truth
SEED = 0
N_EMITTERS = 3000
FIELD_NM = 5000.0                  # emitters at random (x, y) over 5 x 5 um
Z_HALF_RANGE_NM = 400.0            # true z uniform over +-400 nm
FRAMES_PER_EMITTER = (2, 6)        # k consecutive frames, both ends included
# Start frames uniform over this many frames: about 0.6 localizations per
# frame over 25 um^2, so the chance that ANOTHER emitter sits within the
# link radius in the next frame is ~0.6 * pi * 40^2 / 5000^2 ~ 1e-4.
N_FRAMES = 20_000
LATERAL_SIGMA_NM = 8.0
SIGMA_BELOW_NM = 40.0              # true axial error for z < 0
SIGMA_ABOVE_NM = 70.0              # true axial error for z >= 0
LPZ_FACTOR = 0.6                   # reported lpz = 0.6 * true sigma (optimistic)
TRUE_RATIO = 1.0 / LPZ_FACTOR      # what sigma_z / lpz_median must recover

# -------------------------------------------------------- axial_nena call
# 40 nm. A consecutive-frame lateral step is Rayleigh with sigma
# sqrt(2) * 8 = 11.3 nm, so P(step > 40 nm) = exp(-40^2 / (2 * 11.3^2))
# = 0.2 %: the link keeps essentially every pair, and 40 nm is stricter
# than the 30 nm at which the spec asks the temporal overlap to be < 1 %.
LINK_RADIUS_NM = 5.0 * LATERAL_SIGMA_NM
Z_BIN_NM = 50.0
MIN_PAIRS = 50
Z_RANGE_NM = (-400.0, 400.0)       # 16 strata of 50 nm on round edges

# ----------------------------------------------------------- tolerances
# sd of a sd estimate is sigma / sqrt(2 n) = 3.2 % at n = 500; 10 % is ~3 sd.
RICH_STRATUM_PAIRS = 500
SIGMA_TOL = 0.10
# ratio = sigma_z / lpz_median: the sampling sd of the ratio at n >= 500 is
# ~0.05 (3.2 % of 1.67), so +-0.2 is ~4 sd.
RATIO_TOL = 0.2
# The MAD-based scale has sd ~1.16 sigma / sqrt(n) and the sd-based one
# 1 / sqrt(2 n); their difference has sd ~6 % at n = 500, where 15 % is
# only 2.5 sd (one stratum in ten crosses it by chance), and ~4.7 % at
# n = 800, where 15 % is 3.2 sd. So this comparison runs on 100-nm strata
# with >= 800 pairs.
ROBUST_VS_SD_TOL = 0.15
ROBUST_VS_SD_MIN_PAIRS = 800
# 5 % of the dz replaced by +-500 nm outliers, with the pair's mean z kept
# (so its stratum is kept): each stratum carries 5 % +- Poisson. That adds
# 0.05 * 500^2 = 12500 nm^2 to var(dz): sigma_z 40 -> 88 and 70 -> 104.
# At 2 % contamination (3 sd below the mean count of ~28 per stratum)
# the 70 nm strata are still inflated 1.22x, so 1.15 is > 3 sd of margin.
OUTLIER_FRACTION = 0.05
OUTLIER_NM = 500.0
SD_INFLATION_MIN = 1.15
# 5 % of |dz| pushed beyond the MAD raises the 50 % quantile of |dz| from
# 0.6745 to 0.7165 sigma (+6 %); with the 5.2 % sampling sd of a MAD at
# n = 500, 25 % is 6 % + 3.6 sd.
ROBUST_UNDER_OUTLIERS_TOL = 0.25
# A stratum whose centre is >= 125 nm from z = 0 receives pairs from the
# other side of z = 0 only when their mean-z noise exceeds 100 nm: 2.0 sd
# at sigma/sqrt2 = 49 nm, 3.6 sd at 28 nm. That per-mille contamination
# leaves a median untouched and moves an RMS by < 1 %.
PURE_STRATUM_MIN_ABS_CENTRE_NM = 125.0
# For the "NaN exactly where n_pairs < min_pairs" run on the explicit
# range: the interior strata hold ~9000 / 16 = 560 pairs (sd ~24) and the
# top edge stratum only ~370 (the uniform density of true z ends at +400
# and the pair mean z carries 49 nm of noise there), so 450 splits the
# strata into both kinds with > 4 sd of margin on each side.
SPLITTING_MIN_PAIRS = 450


@dataclass
class Truth:
    """
    The synthetic movie with everything the estimator has to recover.

    Localizations are stored emitter by emitter, in frame order, so the
    true consecutive-frame pairs are simply the adjacent rows that share
    an emitter (``pair_first``, ``pair_second``).
    """

    frame: NDArray[np.int64]
    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    lpz_nm: NDArray[np.float64]
    emitter: NDArray[np.int64]
    z_true_nm: NDArray[np.float64]          # per localization: its emitter's z
    sigma_true_nm: NDArray[np.float64]      # per localization
    pair_first: NDArray[np.intp]
    pair_second: NDArray[np.intp]
    frames_per_emitter: NDArray[np.int64]   # (N_EMITTERS,) k
    last_of_emitter: NDArray[np.intp]       # (N_EMITTERS,) row of its last frame

    @property
    def n(self) -> int:
        return int(self.frame.size)

    @property
    def n_pairs(self) -> int:
        return int(self.pair_first.size)

    def pair_mean_z(self, z_nm: Optional[NDArray[np.float64]] = None
                    ) -> NDArray[np.float64]:
        z = self.z_nm if z_nm is None else z_nm
        return 0.5 * (z[self.pair_first] + z[self.pair_second])

    @property
    def pair_sigma_true(self) -> NDArray[np.float64]:
        return self.sigma_true_nm[self.pair_first]


def simulate(rng: np.random.Generator) -> Truth:
    """
    3000 emitters, each seen in k in {2..6} consecutive frames.

    The axial error depends on the TRUE depth of the emitter (40 nm below
    z = 0, 70 nm above), so a per-stratum estimator has something to
    resolve, and the lpz column is 0.6x the true error, so the ratio has
    a known answer. Lateral noise 8 nm.
    """
    k = rng.integers(FRAMES_PER_EMITTER[0], FRAMES_PER_EMITTER[1] + 1,
                     N_EMITTERS)
    start = rng.integers(0, N_FRAMES - FRAMES_PER_EMITTER[1], N_EMITTERS)
    emitter = np.repeat(np.arange(N_EMITTERS, dtype=np.int64), k)
    n = emitter.size
    first_row = np.concatenate([[0], np.cumsum(k)[:-1]])
    offset = np.arange(n) - first_row[emitter]
    frame = (start[emitter] + offset).astype(np.int64)

    x_true = rng.uniform(0.0, FIELD_NM, N_EMITTERS)
    y_true = rng.uniform(0.0, FIELD_NM, N_EMITTERS)
    z_true = rng.uniform(-Z_HALF_RANGE_NM, Z_HALF_RANGE_NM, N_EMITTERS)
    sigma_emitter = np.where(z_true < 0.0, SIGMA_BELOW_NM, SIGMA_ABOVE_NM)

    x = x_true[emitter] + rng.normal(0.0, LATERAL_SIGMA_NM, n)
    y = y_true[emitter] + rng.normal(0.0, LATERAL_SIGMA_NM, n)
    sigma_true = sigma_emitter[emitter]
    z = z_true[emitter] + sigma_true * rng.normal(0.0, 1.0, n)
    lpz = LPZ_FACTOR * sigma_true

    pair_first = np.nonzero(emitter[:-1] == emitter[1:])[0]
    return Truth(
        frame=frame, x_nm=x, y_nm=y, z_nm=z, lpz_nm=lpz, emitter=emitter,
        z_true_nm=z_true[emitter], sigma_true_nm=sigma_true,
        pair_first=pair_first,
        pair_second=pair_first + 1, frames_per_emitter=k.astype(np.int64),
        last_of_emitter=first_row + k - 1,
    )


def temporal_overlap_rate(truth: Truth, radius_nm: float) -> Tuple[float, int]:
    """
    Fraction of localizations that have a localization of ANOTHER emitter
    within ``radius_nm`` laterally in the very next frame -- the only
    situation in which the link can pair two different molecules.

    A 3D tree with the frame stretched by 1e6 nm: querying at frame + 1
    returns exactly the next-frame neighbours within the lateral radius.
    """
    stretch = 1e6
    tree = cKDTree(np.column_stack([truth.x_nm, truth.y_nm,
                                    truth.frame * stretch]))
    query = np.column_stack([truth.x_nm, truth.y_nm,
                             (truth.frame + 1) * stretch])
    hits = tree.query_ball_point(query, r=radius_nm)
    n_overlap = 0
    for i, near in enumerate(hits):
        if any(truth.emitter[j] != truth.emitter[i] for j in near):
            n_overlap += 1
    return n_overlap / truth.n, n_overlap


def stratum_truth(
    truth: Truth,
    edges_nm: NDArray[np.float64],
    z_nm: Optional[NDArray[np.float64]] = None,
    exclude_pair: Optional[NDArray[np.bool_]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Pairs per stratum and the RMS of their TRUE sigma, binning the true
    consecutive-frame pairs by their mean z on the edges the module
    returned.

    The RMS is the right truth because sd(dz)/sqrt2 over a mixture of
    Gaussians of different sigma estimates sqrt(E[sigma^2]); in a stratum
    on one side of z = 0 it is simply 40 or 70.
    """
    mean_z = truth.pair_mean_z(z_nm)
    sigma = truth.pair_sigma_true
    keep = np.ones(truth.n_pairs, dtype=bool)
    if exclude_pair is not None:
        keep &= ~exclude_pair
    index = np.digitize(mean_z, edges_nm) - 1
    n_strata = edges_nm.size - 1
    count = np.zeros(n_strata, dtype=np.int64)
    rms = np.full(n_strata, np.nan)
    for s in range(n_strata):
        sel = keep & (index == s)
        count[s] = int(sel.sum())
        if count[s]:
            rms[s] = float(np.sqrt(np.mean(sigma[sel] ** 2)))
    return count, rms


def run_nena(truth: Truth, **overrides) -> AxialNena:
    kwargs = dict(lpz_nm=truth.lpz_nm, link_radius_nm=LINK_RADIUS_NM,
                  z_bin_nm=Z_BIN_NM, min_pairs=MIN_PAIRS,
                  z_range_nm=Z_RANGE_NM)
    kwargs.update(overrides)
    return axial_nena(truth.frame, truth.x_nm, truth.y_nm, truth.z_nm,
                      **kwargs)


def print_strata(result: AxialNena, truth_rms: NDArray[np.float64]) -> None:
    print("      centre   n_pairs  sigma_z  robust  lpz_med   ratio   truth")
    for s in range(result.z_bin_centre_nm.size):
        print(f"      {result.z_bin_centre_nm[s]:7.1f} {result.n_pairs[s]:8d}"
              f" {result.sigma_z_nm[s]:8.1f} {result.sigma_z_robust_nm[s]:7.1f}"
              f" {result.lpz_median_nm[s]:8.1f} {result.ratio[s]:7.2f}"
              f" {truth_rms[s]:7.1f}")


# ===================================================== 1. the dataset itself
def test_dataset(truth: Truth) -> None:
    print("\n1. THE SYNTHETIC MOVIE (truth kept explicitly)")

    def frames_are_consecutive():
        d = truth.frame[truth.pair_second] - truth.frame[truth.pair_first]
        assert np.all(d == 1)
        k = np.bincount(truth.emitter)
        assert k.min() == 2 and k.max() == 6
        assert truth.n_pairs == truth.n - N_EMITTERS
        return (f"{truth.n} localizations, {N_EMITTERS} emitters, "
                f"{truth.n_pairs} true consecutive-frame pairs")

    def overlap_below_one_percent():
        rate, n_overlap = temporal_overlap_rate(truth, LINK_RADIUS_NM)
        assert rate < 0.01, rate
        return (f"{n_overlap} localizations ({rate:.2%}) see another "
                f"emitter within {LINK_RADIUS_NM:.0f} nm in the next frame")

    def truth_is_what_it_says():
        below = truth.sigma_true_nm == SIGMA_BELOW_NM
        assert np.allclose(truth.lpz_nm[below], LPZ_FACTOR * SIGMA_BELOW_NM)
        assert np.allclose(truth.lpz_nm[~below], LPZ_FACTOR * SIGMA_ABOVE_NM)
        # Same-emitter dz has sd sqrt(2) sigma: check the generator once.
        dz = truth.z_nm[truth.pair_second] - truth.z_nm[truth.pair_first]
        for sigma in (SIGMA_BELOW_NM, SIGMA_ABOVE_NM):
            sel = truth.pair_sigma_true == sigma
            got = float(np.std(dz[sel]) / np.sqrt(2.0))
            assert abs(got - sigma) / sigma < 0.05, (sigma, got)
        # The noise is Gaussian: over 12,000 draws the largest |z-score|
        # is ~3.8 and exceeds 5 with probability ~0.3 %.
        score = np.abs(truth.z_nm - truth.z_true_nm) / truth.sigma_true_nm
        assert score.max() < 5.0, score.max()
        return (f"{below.mean():.0%} of localizations at 40 nm, rest at 70; "
                f"lpz = 0.6 sigma; largest |z-score| {score.max():.2f}; "
                f"z spans {truth.z_nm.min():.0f}..{truth.z_nm.max():.0f} nm")

    check("emitters emit k in {2..6} consecutive frames", frames_are_consecutive)
    check("temporal overlap of different emitters < 1 % (spec)",
          overlap_below_one_percent)
    check("generator: sd(dz)/sqrt2 of the TRUE pairs hits 40 and 70 nm "
          "(5 %)", truth_is_what_it_says)


# ========================================================== 2. axial NeNA
def test_axial_nena(truth: Truth, rng: np.random.Generator) -> None:
    print("\n2. AXIAL NeNA PER STRATUM (axial_nena)")
    result = run_nena(truth)
    n_true, rms_true = stratum_truth(truth, result.z_bin_edges_nm)
    print_strata(result, rms_true)
    centre = result.z_bin_centre_nm
    rich = result.n_pairs >= RICH_STRATUM_PAIRS
    pure = np.abs(centre) >= PURE_STRATUM_MIN_ABS_CENTRE_NM
    nominal = np.where(centre < 0.0, SIGMA_BELOW_NM, SIGMA_ABOVE_NM)

    def bins_as_requested():
        edges = result.z_bin_edges_nm
        expected = np.linspace(Z_RANGE_NM[0], Z_RANGE_NM[1], 17)
        assert edges.shape == (17,), edges.shape
        assert np.allclose(edges, expected, atol=1e-9), edges
        assert np.allclose(result.z_bin_centre_nm,
                           0.5 * (edges[:-1] + edges[1:]), atol=1e-9)
        return "16 strata of 50 nm from -400 to 400, centres at midpoints"

    def shapes_and_parameters():
        assert isinstance(result, AxialNena)
        s = result.z_bin_centre_nm.size
        for name in ("n_pairs", "sigma_z_nm", "sigma_z_robust_nm",
                     "lpz_median_nm", "ratio"):
            arr = getattr(result, name)
            assert arr.shape == (s,), (name, arr.shape)
        assert np.issubdtype(result.n_pairs.dtype, np.integer)
        assert result.link_radius_nm == LINK_RADIUS_NM
        assert result.min_pairs == MIN_PAIRS
        assert isinstance(result.warnings, list)
        assert isinstance(result.n_pairs_total, (int, np.integer))
        assert isinstance(result.n_events_total, (int, np.integer))
        return (f"S = {s}; link_radius_nm and min_pairs recorded; "
                f"{len(result.warnings)} warning(s)")

    def pairs_per_stratum_match():
        # The link loses P(step > 40 nm) = 0.2 % of the pairs and the
        # temporal overlap (< 1 %) can add or move a few: 3 % per stratum.
        assert n_true.sum() >= 0.9 * truth.n_pairs
        rel = np.abs(result.n_pairs - n_true) / np.maximum(n_true, 1)
        assert np.all(rel[n_true >= 100] < 0.03), rel
        return (f"module {result.n_pairs.sum()} vs truth {n_true.sum()} "
                f"pairs in range; worst stratum {rel[n_true >= 100].max():.2%}")

    def totals():
        assert abs(result.n_events_total - N_EMITTERS) <= 0.03 * N_EMITTERS, \
            result.n_events_total
        assert result.n_pairs.sum() <= result.n_pairs_total <= 1.03 * truth.n_pairs
        assert result.n_pairs.sum() >= 0.9 * truth.n_pairs
        return (f"n_events_total {result.n_events_total} (3000 emitters, "
                f"3 %); n_pairs_total {result.n_pairs_total} "
                f"(truth {truth.n_pairs})")

    def sigma_per_stratum_10pct():
        assert rich.sum() >= 8, f"only {rich.sum()} strata with >= 500 pairs"
        err = np.abs(result.sigma_z_nm - rms_true) / rms_true
        assert np.all(err[rich] < SIGMA_TOL), err
        return (f"{rich.sum()} strata with >= {RICH_STRATUM_PAIRS} pairs; "
                f"worst |error| {err[rich].max():.1%} (tol 10 % ~ 3 sd)")

    def pure_strata_recover_40_and_70():
        sel = rich & pure
        assert sel.sum() >= 6, sel.sum()
        err = np.abs(result.sigma_z_nm - nominal) / nominal
        assert np.all(err[sel] < SIGMA_TOL), err
        lo = result.sigma_z_nm[sel & (centre < 0)]
        hi = result.sigma_z_nm[sel & (centre > 0)]
        return (f"below z=0: {lo.min():.1f}-{lo.max():.1f} nm (truth 40); "
                f"above: {hi.min():.1f}-{hi.max():.1f} nm (truth 70)")

    def ratio_recovers_one_over_0_6():
        sel = rich & pure
        dev = np.abs(result.ratio - TRUE_RATIO)
        assert np.all(dev[sel] < RATIO_TOL), result.ratio
        return (f"ratio {result.ratio[sel].min():.2f}-"
                f"{result.ratio[sel].max():.2f} vs 1/0.6 = {TRUE_RATIO:.3f} "
                f"(+-0.2)")

    def lpz_median_exact_in_pure_strata():
        sel = pure & (result.n_pairs >= MIN_PAIRS)
        expected = LPZ_FACTOR * nominal
        assert np.all(np.abs(result.lpz_median_nm[sel] - expected[sel]) < 1e-6), \
            result.lpz_median_nm
        return "24.0 nm below z=0, 42.0 nm above (1e-6; a median is exact)"

    def overall_numbers():
        rms_all = float(np.sqrt(np.mean(truth.pair_sigma_true ** 2)))
        # n ~ 9000 pairs: sd of the sd 0.75 %, so 5 % is generous.
        err = abs(result.sigma_z_overall_nm - rms_all) / rms_all
        assert err < 0.05, (result.sigma_z_overall_nm, rms_all)
        lo, hi = LPZ_FACTOR * SIGMA_BELOW_NM, LPZ_FACTOR * SIGMA_ABOVE_NM
        assert lo - 1e-6 <= result.lpz_median_overall_nm <= hi + 1e-6, \
            result.lpz_median_overall_nm
        return (f"sigma_z_overall {result.sigma_z_overall_nm:.1f} nm vs RMS "
                f"truth {rms_all:.1f} (5 %); lpz median overall "
                f"{result.lpz_median_overall_nm:.1f} nm (24..42)")

    def robust_agrees_with_sd_on_gaussian_data():
        # 100-nm strata, so each holds ~1100 pairs (see ROBUST_VS_SD_TOL).
        wide = run_nena(truth, z_bin_nm=2.0 * Z_BIN_NM)
        assert wide.z_bin_centre_nm.size == 8, wide.z_bin_centre_nm
        gaussian = np.abs(wide.z_bin_centre_nm) >= 150.0     # one side of z=0
        sel = gaussian & (wide.n_pairs >= ROBUST_VS_SD_MIN_PAIRS)
        assert sel.sum() >= 4, wide.n_pairs
        rel = np.abs(wide.sigma_z_robust_nm - wide.sigma_z_nm) / wide.sigma_z_nm
        assert np.all(rel[sel] < ROBUST_VS_SD_TOL), rel
        # And the 50-nm strata, for the record (2.5 sd only: printed).
        rel50 = np.abs(result.sigma_z_robust_nm - result.sigma_z_nm) / result.sigma_z_nm
        return (f"worst |robust - sd| / sd {rel[sel].max():.1%} over "
                f"{sel.sum()} Gaussian 100-nm strata of >= "
                f"{ROBUST_VS_SD_MIN_PAIRS} pairs (tol 15 % >= 3.2 sd); on the "
                f"50-nm strata {rel50[rich & pure].max():.1%}")

    def nan_only_where_sparse():
        sparse = result.n_pairs < MIN_PAIRS
        for name in ("sigma_z_nm", "sigma_z_robust_nm", "ratio"):
            arr = getattr(result, name)
            assert np.all(np.isfinite(arr[~sparse])), name
            assert np.all(np.isnan(arr[sparse])), name
        assert np.all(np.isfinite(result.lpz_median_nm[~sparse]))
        assert np.isfinite(result.sigma_z_overall_nm)
        return f"{sparse.sum()} sparse strata on the explicit range"

    def lpz_absent_gives_nan_not_guesses():
        bare = run_nena(truth, lpz_nm=None)
        assert np.all(np.isnan(bare.lpz_median_nm))
        assert np.all(np.isnan(bare.ratio))
        assert np.isnan(bare.lpz_median_overall_nm)
        assert np.array_equal(bare.sigma_z_nm, result.sigma_z_nm, equal_nan=True)
        assert np.array_equal(bare.sigma_z_robust_nm, result.sigma_z_robust_nm,
                              equal_nan=True)
        assert np.array_equal(bare.n_pairs, result.n_pairs)
        return "medians and ratios NaN; sigma_z identical"

    def deterministic():
        again = run_nena(truth)
        for name in ("z_bin_edges_nm", "z_bin_centre_nm", "n_pairs",
                     "sigma_z_nm", "sigma_z_robust_nm", "lpz_median_nm",
                     "ratio"):
            assert np.array_equal(getattr(again, name), getattr(result, name),
                                  equal_nan=True), name
        assert again.sigma_z_overall_nm == result.sigma_z_overall_nm
        assert again.n_pairs_total == result.n_pairs_total
        assert again.n_events_total == result.n_events_total
        assert again.warnings == result.warnings
        return "two calls, identical arrays and warnings"

    def input_order_does_not_matter():
        # The only order dependence allowed is the link's choice between
        # two candidates within the radius in the next frame, which
        # happens for the overlapping localizations counted in section 1
        # (< 1 %); each such choice moves at most two pairs.
        perm = rng.permutation(truth.n)
        shuffled = axial_nena(
            truth.frame[perm], truth.x_nm[perm], truth.y_nm[perm],
            truth.z_nm[perm], lpz_nm=truth.lpz_nm[perm],
            link_radius_nm=LINK_RADIUS_NM, z_bin_nm=Z_BIN_NM,
            min_pairs=MIN_PAIRS, z_range_nm=Z_RANGE_NM,
        )
        _, n_overlap = temporal_overlap_rate(truth, LINK_RADIUS_NM)
        allowed = 2 * n_overlap + 1
        dn = np.abs(shuffled.n_pairs - result.n_pairs)
        assert np.all(dn <= allowed), (dn, allowed)
        rel = np.abs(shuffled.sigma_z_nm - result.sigma_z_nm) / result.sigma_z_nm
        assert np.all(rel[rich] < 0.02), rel
        return (f"shuffled input: pairs per stratum differ by <= {dn.max()} "
                f"(allowed {allowed}), sigma_z by <= {rel[rich].max():.2%}")

    check("bins: explicit range -> 16 strata of 50 nm, centres at midpoints",
          bins_as_requested)
    check("result shapes, integer counts, parameters recorded",
          shapes_and_parameters)
    check("pairs per stratum match the true pairs (3 %)",
          pairs_per_stratum_match)
    check("n_events_total within 3 % of the 3000 emitters; n_pairs_total "
          "consistent", totals)
    check("sigma_z per stratum within 10 % of truth where n_pairs >= 500",
          sigma_per_stratum_10pct)
    check("pure strata recover 40 nm (z<0) and 70 nm (z>=0) within 10 %",
          pure_strata_recover_40_and_70)
    check("ratio sigma_z / lpz_median = 1/0.6 +- 0.2 (lpz was 0.6x optimistic)",
          ratio_recovers_one_over_0_6)
    check("lpz_median per stratum is the reported lpz (1e-6)",
          lpz_median_exact_in_pure_strata)
    check("overall sigma_z within 5 % of the RMS truth; overall lpz median",
          overall_numbers)
    check("robust estimate within 15 % of the sd on Gaussian strata",
          robust_agrees_with_sd_on_gaussian_data)
    check("NaN exactly where n_pairs < min_pairs", nan_only_where_sparse)
    check("lpz absent -> lpz_median and ratio NaN, sigma_z unchanged",
          lpz_absent_gives_nan_not_guesses)
    check("deterministic", deterministic)
    check("input order does not matter (within the overlap allowance)",
          input_order_does_not_matter)


# ===================================================== 3. sparse strata
def test_sparse_strata(truth: Truth) -> None:
    print("\n3. STRATA WITH TOO FEW PAIRS -> NaN AND A WARNING")

    def default_range_has_empty_tails():
        # Default z range = (min z, max z) of the input. The observed z
        # reaches ~3.5 sigma beyond +-400 nm, and a pair lands out there
        # only through its mean-z noise (28 or 49 nm), so the outermost
        # strata hold a handful of pairs at most.
        result = run_nena(truth, z_range_nm=None)
        edges = result.z_bin_edges_nm
        assert edges[0] <= truth.z_nm.min() + 1e-9, (edges[0], truth.z_nm.min())
        assert edges[-1] >= truth.z_nm.max() - 1e-9, (edges[-1], truth.z_nm.max())
        widths = np.diff(edges)
        assert np.all(widths > 0)
        assert np.allclose(widths[:-1], Z_BIN_NM, atol=1e-9), widths
        assert widths[-1] <= Z_BIN_NM + 1e-9
        sparse = result.n_pairs < MIN_PAIRS
        assert sparse.any(), "no sparse stratum on the default range"
        assert (~sparse).any()
        for name in ("sigma_z_nm", "sigma_z_robust_nm", "ratio"):
            arr = getattr(result, name)
            assert np.all(np.isnan(arr[sparse])), name
            assert np.all(np.isfinite(arr[~sparse])), name
        assert any("pairs" in w.lower() for w in result.warnings), result.warnings
        assert result.n_pairs.sum() == result.n_pairs_total, \
            (result.n_pairs.sum(), result.n_pairs_total)
        # Nothing is dropped on the default range: the total is the truth
        # up to the link's 0.2 % loss and the < 1 % overlap.
        assert abs(result.n_pairs_total - truth.n_pairs) <= 0.03 * truth.n_pairs
        return (f"{edges.size - 1} strata from {edges[0]:.0f} to "
                f"{edges[-1]:.0f} nm; {sparse.sum()} below {MIN_PAIRS} pairs "
                f"-> NaN; {len(result.warnings)} warning(s); n_pairs_total "
                f"{result.n_pairs_total} vs truth {truth.n_pairs}")

    def min_pairs_splits_the_strata():
        base = run_nena(truth)
        result = run_nena(truth, min_pairs=SPLITTING_MIN_PAIRS)
        assert result.min_pairs == SPLITTING_MIN_PAIRS
        assert np.array_equal(result.n_pairs, base.n_pairs), \
            "counting must not depend on min_pairs"
        sparse = result.n_pairs < SPLITTING_MIN_PAIRS
        assert sparse.any() and (~sparse).any(), result.n_pairs
        assert np.all(np.isnan(result.sigma_z_nm[sparse]))
        assert np.all(np.isnan(result.sigma_z_robust_nm[sparse]))
        assert np.all(np.isnan(result.ratio[sparse]))
        assert np.all(np.isfinite(result.sigma_z_nm[~sparse]))
        assert np.array_equal(result.sigma_z_nm[~sparse],
                              base.sigma_z_nm[~sparse])
        assert any("pairs" in w.lower() for w in result.warnings), result.warnings
        return (f"min_pairs={SPLITTING_MIN_PAIRS}: {sparse.sum()} of "
                f"{sparse.size} strata NaN (n_pairs "
                f"{result.n_pairs.min()}..{result.n_pairs.max()}); the rest "
                f"identical to the min_pairs={MIN_PAIRS} run")

    check("default z range: tail strata below min_pairs -> NaN + warning",
          default_range_has_empty_tails)
    check("min_pairs raised: NaN exactly where n_pairs < min_pairs, warning",
          min_pairs_splits_the_strata)


# ========================================================== 4. outliers
def test_outliers(truth: Truth, rng: np.random.Generator) -> None:
    print("\n4. ROBUST ESTIMATE UNDER 5 % OUTLIERS OF +-500 nm")
    # An emitter seen in exactly two frames owns exactly one pair. Moving
    # its first localization by -250 s and its second by +250 s (s = +-1)
    # turns that dz into an outlier of 500 nm while the pair's mean z, and
    # therefore its stratum, stays put; no other pair is touched. So every
    # stratum keeps its own pairs and carries 5 % +- Poisson of outliers.
    n_outliers = int(round(OUTLIER_FRACTION * truth.n_pairs))
    two_frame = np.nonzero(truth.frames_per_emitter == 2)[0]
    assert two_frame.size >= n_outliers, (two_frame.size, n_outliers)
    chosen = rng.choice(two_frame, n_outliers, replace=False)
    sign = rng.choice([-1.0, 1.0], n_outliers)
    second_row = truth.last_of_emitter[chosen]
    first_row = second_row - 1
    z_out = truth.z_nm.copy()
    z_out[first_row] -= 0.5 * OUTLIER_NM * sign
    z_out[second_row] += 0.5 * OUTLIER_NM * sign
    pair_of_second = np.full(truth.n, -1, dtype=np.intp)
    pair_of_second[truth.pair_second] = np.arange(truth.n_pairs)
    outlier_pair = np.zeros(truth.n_pairs, dtype=bool)
    outlier_pair[pair_of_second[second_row]] = True
    assert outlier_pair.sum() == n_outliers
    assert np.allclose(truth.pair_mean_z(z_out), truth.pair_mean_z())

    clean = run_nena(truth)
    dirty = axial_nena(truth.frame, truth.x_nm, truth.y_nm, z_out,
                       lpz_nm=truth.lpz_nm, link_radius_nm=LINK_RADIUS_NM,
                       z_bin_nm=Z_BIN_NM, min_pairs=MIN_PAIRS,
                       z_range_nm=Z_RANGE_NM)
    _, rms_clean = stratum_truth(truth, dirty.z_bin_edges_nm, z_nm=z_out,
                                 exclude_pair=outlier_pair)
    print_strata(dirty, rms_clean)
    rich = dirty.n_pairs >= RICH_STRATUM_PAIRS

    def strata_unchanged_by_the_outliers():
        # Same xy, same frames, same pair mean z: the counting must not
        # notice. This is the pair-mean-z rule tested at +-250 nm.
        assert np.array_equal(dirty.n_pairs, clean.n_pairs), \
            (dirty.n_pairs, clean.n_pairs)
        assert dirty.n_pairs_total == clean.n_pairs_total
        assert np.array_equal(dirty.lpz_median_nm, clean.lpz_median_nm,
                              equal_nan=True)
        return (f"{n_outliers} of {truth.n_pairs} pairs ({OUTLIER_FRACTION:.0%})"
                f" made outliers; n_pairs per stratum identical to the clean run")

    def sd_is_inflated():
        assert rich.sum() >= 8
        inflation = dirty.sigma_z_nm / rms_clean
        assert np.all(inflation[rich] > SD_INFLATION_MIN), inflation
        # Overall: var(dz) = 0.95 * 2 * 3250 + 0.05 * 500^2 -> 1.69x.
        assert dirty.sigma_z_overall_nm > 1.3 * clean.sigma_z_overall_nm
        return (f"sd-based sigma_z inflated {inflation[rich].min():.2f}-"
                f"{inflation[rich].max():.2f}x (min {SD_INFLATION_MIN}); "
                f"overall {clean.sigma_z_overall_nm:.1f} -> "
                f"{dirty.sigma_z_overall_nm:.1f} nm")

    def robust_is_closer_to_truth():
        err_sd = np.abs(dirty.sigma_z_nm - rms_clean)
        err_robust = np.abs(dirty.sigma_z_robust_nm - rms_clean)
        assert np.all(err_robust[rich] < err_sd[rich]), (err_robust, err_sd)
        return (f"in all {rich.sum()} strata with >= 500 pairs: median |err| "
                f"robust {np.median(err_robust[rich]):.1f} nm vs sd "
                f"{np.median(err_sd[rich]):.1f} nm")

    def robust_stays_close():
        rel = np.abs(dirty.sigma_z_robust_nm - rms_clean) / rms_clean
        assert np.all(rel[rich] < ROBUST_UNDER_OUTLIERS_TOL), rel
        return (f"worst |robust - truth| / truth {rel[rich].max():.1%} "
                f"(tol {ROBUST_UNDER_OUTLIERS_TOL:.0%} = 6 % bias + 3.6 sd)")

    check("outliers that keep the pair's mean z leave every count unchanged",
          strata_unchanged_by_the_outliers)
    check(f"5 % outliers inflate the sd-based sigma_z by > {SD_INFLATION_MIN}x "
          f"(sanity, > 3 sd)", sd_is_inflated)
    check("robust sigma_z is closer to the truth than the sd in every rich "
          "stratum", robust_is_closer_to_truth)
    check("robust sigma_z within 25 % of the truth despite the outliers",
          robust_stays_close)


# ================================================= 5. hand-made arithmetic
def test_handmade() -> None:
    print("\n5. HAND-MADE MOVIE: THE PAIR RULE AND THE ARITHMETIC, EXACTLY")
    # A: frames 0..200 at (0, 0), z alternating 95 / 145 nm: 200 pairs
    #    with dz = +-50, every pair mean z = 120 -> stratum [100, 150),
    #    although its two localizations sit in different strata.
    frame_a = np.arange(201, dtype=np.int64)
    z_a = np.where(frame_a % 2 == 0, 95.0, 145.0)
    # B: frames 0, 2, ..., 40 at (1000, 1000): never consecutive, so never
    #    paired; z alternating 10 / 190 would give dz = +-180 if it were.
    frame_b = np.arange(0, 41, 2, dtype=np.int64)
    z_b = np.where((frame_b // 2) % 2 == 0, 10.0, 190.0)
    # C: frames 0..3 at z = 20 and frames 500..503 at z = 180, both at
    #    (2000, 2000): two events of 3 pairs each with dz = 0; the 497-frame
    #    gap between them must not become a pair.
    frame_c = np.r_[np.arange(4), np.arange(500, 504)].astype(np.int64)
    z_c = np.r_[np.full(4, 20.0), np.full(4, 180.0)]
    frame = np.concatenate([frame_a, frame_b, frame_c])
    xy = np.concatenate([np.zeros(201), np.full(21, 1000.0),
                         np.full(8, 2000.0)])
    z = np.concatenate([z_a, z_b, z_c])
    lpz = np.concatenate([np.full(201, 10.0), np.full(21, 20.0),
                          np.full(8, 30.0)])
    result = axial_nena(frame, xy, xy, z, lpz_nm=lpz,
                        link_radius_nm=LINK_RADIUS_NM, z_bin_nm=50.0,
                        min_pairs=2, z_range_nm=(0.0, 200.0))
    sqrt2 = np.sqrt(2.0)

    def pairs_land_where_they_should():
        assert np.allclose(result.z_bin_edges_nm, [0, 50, 100, 150, 200])
        assert result.n_pairs.tolist() == [3, 0, 200, 3], result.n_pairs
        assert result.n_pairs_total == 206, result.n_pairs_total
        # Every event is counted, or only the ones that yield pairs:
        # 3 multi-frame events, 21 singletons from B.
        assert 3 <= result.n_events_total <= 24, result.n_events_total
        return (f"n_pairs {result.n_pairs.tolist()}: B (gap 2) and the "
                f"497-frame gap of C never paired; n_events_total "
                f"{result.n_events_total}")

    def sd_and_mad_arithmetic():
        # dz = +-50 alternating: sd(dz) = 50 (ddof 0) or 50.13 (ddof 1),
        # so 0.5 % covers either; MAD(dz) = 50 exactly.
        expected_sd = 50.0 / sqrt2
        expected_robust = 1.4826 * 50.0 / sqrt2
        assert abs(result.sigma_z_nm[2] - expected_sd) / expected_sd < 0.005, \
            result.sigma_z_nm[2]
        assert abs(result.sigma_z_robust_nm[2] - expected_robust) \
            / expected_robust < 1e-3, result.sigma_z_robust_nm[2]
        # dz = 0 in the two C strata: both estimates exactly 0.
        for s in (0, 3):
            assert abs(result.sigma_z_nm[s]) < 1e-9, result.sigma_z_nm
            assert abs(result.sigma_z_robust_nm[s]) < 1e-9
        return (f"stratum [100,150): sigma_z {result.sigma_z_nm[2]:.3f} "
                f"(50/sqrt2 = {expected_sd:.3f}), robust "
                f"{result.sigma_z_robust_nm[2]:.3f} "
                f"(1.4826*50/sqrt2 = {expected_robust:.3f}); dz = 0 strata -> 0")

    def empty_stratum_is_nan_with_warning():
        assert np.isnan(result.sigma_z_nm[1])
        assert np.isnan(result.sigma_z_robust_nm[1])
        assert np.isnan(result.ratio[1])
        assert any("pairs" in w.lower() for w in result.warnings), result.warnings
        return f"stratum [50,100): 0 pairs -> NaN; {len(result.warnings)} warning(s)"

    def lpz_and_ratio_definitions():
        assert abs(result.lpz_median_nm[2] - 10.0) < 1e-9, result.lpz_median_nm
        assert abs(result.lpz_median_nm[0] - 30.0) < 1e-9
        assert abs(result.lpz_median_nm[3] - 30.0) < 1e-9
        assert abs(result.ratio[2] - result.sigma_z_nm[2] / 10.0) < 1e-9
        assert abs(result.ratio[0]) < 1e-9 and abs(result.ratio[3]) < 1e-9
        assert abs(result.lpz_median_overall_nm - 10.0) < 1e-9, \
            result.lpz_median_overall_nm
        # 200 dz of +-50 and 6 of 0: sd = 50 sqrt(200/206) (ddof 0).
        expected_overall = 50.0 * np.sqrt(200.0 / 206.0) / sqrt2
        assert abs(result.sigma_z_overall_nm - expected_overall) \
            / expected_overall < 0.005, result.sigma_z_overall_nm
        return (f"lpz medians [30, nan, 10, 30] as placed; ratio = sigma/lpz; "
                f"overall sigma_z {result.sigma_z_overall_nm:.3f} "
                f"(expected {expected_overall:.3f})")

    check("pairs: consecutive frames of one event, stratum of the mean z",
          pairs_land_where_they_should)
    check("sigma_z = sd(dz)/sqrt2 and robust = 1.4826 MAD(dz)/sqrt2 (0.5 %)",
          sd_and_mad_arithmetic)
    check("an empty stratum is NaN and reported", empty_stratum_is_nan_with_warning)
    check("lpz_median, ratio and the overall numbers follow their definitions",
          lpz_and_ratio_definitions)


# ======================================================= 6. leak fraction
def test_leak_fraction() -> None:
    print("\n6. LEAK FRACTION (leak_fraction)")

    def constant_lpz_closed_form():
        lpz = np.full(50, 47.0)
        worst = 0.0
        cases = [(95.0, 0.0), (95.0, 30.0), (60.0, 20.0), (190.0, 47.0)]
        for delta, sigma_struct in cases:
            expected = float(norm.sf(delta / np.sqrt(sigma_struct**2 + 47.0**2)))
            for boundary in (delta, -delta):
                got = leak_fraction(0.0, boundary, lpz, sigma_struct)
                worst = max(worst, abs(got - expected))
                assert abs(got - expected) < 1e-12, (delta, sigma_struct, got)
        # A centre away from zero: only |boundary - centre| matters.
        got = leak_fraction(200.0, 295.0, lpz)
        assert abs(got - float(norm.sf(95.0 / 47.0))) < 1e-12, got
        return (f"1 - Phi(delta / sqrt(sigma_s^2 + lpz^2)) on 4 cases, both "
                f"sides; worst |diff| {worst:.1e} (tol 1e-12)")

    def two_valued_lpz_exceeds_the_mean_lpz():
        # Jensen: 1 - Phi(delta / lpz) is convex in lpz on the tail, so
        # the average over {30, 90} exceeds the value at their mean 60.
        lpz = np.tile([30.0, 90.0], 25)
        delta = 95.0
        got = leak_fraction(0.0, delta, lpz)
        expected = 0.5 * (float(norm.sf(delta / 30.0)) + float(norm.sf(delta / 90.0)))
        from_mean = float(norm.sf(delta / 60.0))
        assert abs(got - expected) < 1e-12, (got, expected)
        assert got > from_mean, (got, from_mean)
        return (f"F({{30, 90}}) = {got:.4f} > F(60) = {from_mean:.4f}; "
                f"equals the per-localization average to 1e-12")

    def boundary_at_the_centre_is_one_half():
        lpz = np.array([30.0, 47.0, 90.0])
        for sigma_struct in (0.0, 25.0):
            got = leak_fraction(120.0, 120.0, lpz, sigma_struct)
            assert abs(got - 0.5) < 1e-12, got
        return "0.5 to 1e-12, with and without a structural width"

    def monotone_and_bounded():
        lpz = np.array([30.0, 47.0, 90.0])
        deltas = np.array([0.0, 20.0, 50.0, 95.0, 150.0, 300.0])
        values = np.array([leak_fraction(0.0, d, lpz) for d in deltas])
        assert np.all(np.diff(values) < 0), values
        assert values[0] == 0.5 and values[-1] < 1e-3
        assert np.all((values >= 0.0) & (values <= 0.5))
        widened = np.array([leak_fraction(0.0, 95.0, lpz, s) for s in (0.0, 20.0, 40.0)])
        assert np.all(np.diff(widened) > 0), widened
        return (f"F falls with |boundary - centre| ({values[1]:.3f} -> "
                f"{values[-2]:.4f}) and rises with sigma_struct "
                f"({widened[0]:.4f} -> {widened[-1]:.4f})")

    check("constant lpz: closed form 1 - Phi(delta/sqrt(sigma_s^2+lpz^2)) to 1e-12",
          constant_lpz_closed_form)
    check("two-valued lpz (30, 90) > value from their mean (Jensen)",
          two_valued_lpz_exceeds_the_mean_lpz)
    check("boundary at the centre -> 0.5", boundary_at_the_centre_is_one_half)
    check("monotone in the distance and in sigma_struct, bounded by 0.5",
          monotone_and_bounded)


# ======================================= 7. spurious cluster probability
def test_spurious_probability() -> None:
    print("\n7. SPURIOUS CLUSTER PROBABILITY (spurious_cluster_probability)")

    def equals_binomial_tail():
        worst = 0.0
        for n, f, m in product((5, 10, 50, 200), (0.01, 0.05, 0.2), (5, 10)):
            expected = float(binom.sf(m - 1, n, f))
            got = spurious_cluster_probability(n, f, m)
            worst = max(worst, abs(got - expected))
            assert abs(got - expected) < 1e-12, (n, f, m, got, expected)
        return (f"24 (n, F, min_samples) combinations; worst |diff| "
                f"{worst:.1e} (tol 1e-12)")

    def monotone_in_n():
        for f, m in ((0.03, 10), (0.1, 5)):
            p = np.array([spurious_cluster_probability(n, f, m)
                          for n in range(1, 401)])
            assert np.all(np.diff(p) >= -1e-15), (f, m)
            assert p[0] == 0.0 and p[-1] > p[0]
        p10 = spurious_cluster_probability(100, 0.05, 10)
        p200 = spurious_cluster_probability(200, 0.05, 10)
        return (f"non-decreasing over n = 1..400; e.g. F=0.05, "
                f"min_samples=10: n=100 -> {p10:.3f}, n=200 -> {p200:.3f}")

    def edge_cases():
        assert spurious_cluster_probability(4, 0.5, 5) == 0.0
        assert spurious_cluster_probability(50, 0.0, 5) == 0.0
        assert abs(spurious_cluster_probability(50, 1.0, 5) - 1.0) < 1e-12
        assert abs(spurious_cluster_probability(5, 0.3, 5) - 0.3**5) < 1e-12
        return "n < min_samples -> 0; F = 0 -> 0; F = 1 -> 1; n = m -> F^m"

    check("equals scipy.stats.binom.sf(min_samples - 1, n, F) to 1e-12",
          equals_binomial_tail)
    check("monotone in n", monotone_in_n)
    check("edge cases", edge_cases)


# ============================================= 8. structural width
def test_structural_width() -> None:
    print("\n8. STRUCTURAL WIDTH (structural_width_nm)")

    def arithmetic():
        got = structural_width_nm(77.0, 47.0)
        expected = float(np.sqrt(77.0**2 - 47.0**2))     # 60.99 nm
        assert abs(got - expected) < 1e-9, got
        assert abs(got - 61.0) < 0.05, got
        assert structural_width_nm(40.0, 47.0) == 0.0
        assert structural_width_nm(47.0, 47.0) == 0.0
        assert abs(structural_width_nm(50.0, 0.0) - 50.0) < 1e-12
        assert isinstance(got, float)
        return (f"sqrt(77^2 - 47^2) = {got:.2f} nm; 40 vs 47 -> 0 (not NaN, "
                f"not imaginary)")

    check("structural_width_nm(77, 47) ~ 61 nm; (40, 47) == 0", arithmetic)


def main() -> int:
    print("=" * 72)
    print("AXIAL PRECISION CHECKS (tools/mps_axial_precision.py)")
    print("=" * 72)
    rng = np.random.default_rng(SEED)
    truth = simulate(rng)
    test_dataset(truth)
    test_axial_nena(truth, rng)
    test_sparse_strata(truth)
    test_outliers(truth, rng)
    test_handmade()
    test_leak_fraction()
    test_spurious_probability()
    test_structural_width()
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
