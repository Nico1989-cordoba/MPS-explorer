# -*- coding: utf-8 -*-
"""
Checks for tools/mps_leak.py (H4, Module A: the axial-leak diagnostics
(i)-(v) of 01_formalizacion S1.6 point 4 / D-10, the axial profile of
columns and the guard sensitivity) against localization-level synthetic
axons of KNOWN truth (the research plan 03_plan.md S3.3
rows "Fuga sola", "Perfil axial", "Determinismo"; S3.6 H4: detection of
simulated children >= 95 %, N_min reproduced).

Written BEFORE the module (tests first): ``tools.mps_leak`` is imported
lazily through ``require`` so that a missing module reports every module
check as failed instead of crashing at import, and the harness computes
its own truth (truncated-normal densities, KL divergences, Fisher
information, Monte Carlo fractions, majority-of-origin labels) without
the module under test.

The generator (``make_leak_axon``): three rings at z' = -P, 0, +P
(P = 170 nm) IN THE AXON FRAME (no tilt; ``RingsParams(correct_tilt=
False)`` everywhere except the integration check, so H4 is tested on
rings H1 already validated and the run stays fast). Cluster arc
positions on the 1500 x 1000 nm ellipse by ``hard_core_arcs_nm`` (M1: an
independent draw per ring; M5: the same positions on the three rings =
perfect columns of DISTINCT molecules); each cluster has ``n_fluor`` =
100 fluorophores at N(0, 8 nm) lateral offsets, each emitting r ~
Geometric(p = 1 / mean_locs_per_fluor) localizations (mean 2) in
consecutive frames from a random start in 60000 frames; per
localization x, y = fluorophore + N(0, 8 nm) and z' = ring z + N(0,
sigma_struct) [per fluorophore, 0 here] + N(0, lpz_i) [per localization,
independent: this is what leaks]. With P / 2 = 85 nm and sigma_z = 77 nm,
F = 1 - Phi(85 / 77) = 0.135: a 200-localization cluster leaks 27
localizations on average across each boundary, enough for DBSCAN (eps 25
nm, minPts 10) to make a spurious child on the other side (measured on
seed 0: 44 children over the four leak directions, median 24
localizations each). The M1 axon has 3 x 40 x 200 = 24 000
localizations.

Deviations/decisions (each with the number that forced it; measured with
the harness's own truth on seed 0 before the module existed):

* lpz per localization is sigma_z x Uniform(0.9, 1.1) (``lpz_jitter``),
  not the constant the H4 specification wrote, and the z' noise of a
  localization has sd lpz_i, so the leak physics stays exact (F is the
  mean over localizations of 1 - Phi(85 / lpz_i) = 0.1344). Why: the
  pre-specified diagnostic (ii) is a Spearman correlation with the
  clusters' ``lpz_median_nm``, which is undefined (all ties) with a
  constant lpz; a 10 % jitter gives medians that differ by ~1 nm.
* The generator's leak truth (check 1) is compared ONE-SIDED: the
  fraction with z' - ring z > P / 2 and the fraction below -P / 2 are
  each F +- 0.01; the two-sided fraction |z' - ring z| > P / 2 is 2 F
  (0.269 measured), which the specification's sentence conflated with F.
* Linking exactness (check 1): 98.1 % (M1) / 96.9 % (M5) of the
  fluorophores are one event each, not >= 99 %: the loss is not the
  radius (0.2 % of steps, D-17) but temporal merges, two fluorophores of
  the same 16 nm blob emitting within 2 frames of each other (the M5
  axon stacks three rings' clusters on the same x, y, hence the lower
  figure). The check therefore asks >= 99.5 % exactness among the
  fluorophores NOT exposed to a merge (another fluorophore's
  localization within the link radius and <= max_dark_time + 1 frames,
  ``validate_rings_h1.temporal_overlap_fraction``'s rule applied per
  fluorophore), >= 95 % overall, and prints the exposed fraction.
* The tail-vs-centred LLR (check 4) separates the two models on the
  window [85, 255] nm by 2.5 sd (tail draws) and 2.2 sd (centred draws)
  at n = 15, not "> 4 sd" (numerical expectation and sd of the
  per-localization log ratio, computed in the harness): P(right sign)
  is then 0.994 / 0.987 under a Gaussian approximation, so "llr > 0 in
  >= 99 % of 500 repetitions" would fail the centred case by the
  physics. The n = 15 repetitions are asserted against the harness's
  own prediction minus 0.02 (>= 0.95 in any case); the specification's
  >= 99 % is asserted at n = 50, where the separation is 4.5 / 4.1 sd.
* Check 6's closed-form case (n = 200, F = 0.135) is checked as
  IDENTITIES of the module's ``PairLeak`` fields against the harness's
  recomputation on the M1 pairs (expected_leaked_locs = n_parent F to
  1e-9, p_size_leak = binom.sf(n_child - 1, n_parent, F) to 1e-12,
  p_spurious and F_parent to 1e-12), plus the reference numbers from
  ``tools.mps_axial_precision`` (200 x (1 - Phi(85 / 77)) = 26.96; a
  first version of the harness wrote 26.98 and failed on its own
  rounding), since a ``PairLeak`` with exactly those inputs cannot be
  built without a hand-made ``RingsResult``. F_parent is checked with
  the parent RING's ``centre_z_nm`` (the specification's tail-model
  convention); when that fails the check also tries the parent
  cluster's mean z' and names which convention the module used.
* The size rule (checks 5 and 6; review of 2026-09-23, D-27): the
  specification's p_size_leak = P[Bin(n_locs_parent, F) >= n_locs_child]
  with the DETECTED parent count cannot meet its own acceptance -- the
  detected parent has already lost ~F of its localizations to each
  window edge, so the binomial's mean sat 25-35 % below the truth and
  size_ok failed on 37 of 131 children (detection 71.8 %; with the true
  n0 the exact model rejects 0.7-0.9 %, the nominal 1 %). The module now
  tests Bin(n0_hat, F) with n0_hat = round(n_parent / (1 - F_lo - F_hi)),
  the Gaussian mass retained by the parent's own window, and the
  identities of check 6 are those (n_locs_parent_original, F_parent_far
  and parent_retained_fraction recomputed by the harness from the
  parent ring's window edges).
* The acceptance population (checks 5 and 6): the diagnostic is a
  single-parent tail model (the specification's), and 11-15 % of the
  majority-labelled children are BILATERAL -- pure leak from BOTH
  neighbouring rings at a chance-coincident arc (purity 0.5-0.7, own
  localizations 0, z' piled at both window edges), which no single
  truncated Gaussian fits better than the centred model (llr -17 to
  -61 measured). The >= 95 % acceptance is asserted on the unilateral
  children (purity >= 0.9: at most 10 % of the child's localizations
  from a second ring), the bilateral ones are counted and their flagged
  fraction printed as the stated limitation of the model (the H4
  report carries it; a two-sided tail model is H5's).
* Check 12's premise for parent+child columns ("p_bootstrap > 0.05 in
  >= 80 %: one mode with a tail") does not hold: the one-Gaussian
  bootstrap rejects any non-Gaussian tail, and the leak's
  single-localization events sit beyond the boundary, so the mixture
  at the period absorbs them with a minority weight of 0.01-0.06
  (measured 3 of 7 length-2 parent+child columns with p > 0.05 on seed
  0). The check now asserts what the mechanism guarantees -- the
  profile exists per column, its weights sum to 1 and the child's
  component is the minority (pi_hat <= 0.5: the parent's events alone
  are more than half of the column's) -- and PRINTS the fraction with
  p > 0.05 and the pi_hat quartiles as the finding (rule (iv) stays
  None pending H5's simulated-leak null; see ``axial_profile_fit``).
* Check 14: ``make_columns_axon`` has one lpz value (35 nm) for every
  localization, so the Spearman of the has-a-partner indicator against
  the clusters' median lpz is undefined (a constant variable) and rule
  (ii) is None, which the module documents as the extension of the
  specification's "None without lpz" (an undefined correlation cannot
  vote); the check accepts None exactly when the axon's lpz is
  constant and still asks a bool for the other rules.
* Check 7 also asserts the cleaned null the module adds (D-27):
  ``match_clean`` reruns the eclipse test with the flagged children
  unusable on both sides and the run's seed, so on M1 its zeta is a
  standard normal to first order (|zeta_clean_null| <= 3 asserted,
  99.7 % of a standard normal; measured 0.33 / 0.32 on seed 0) while
  the specification's ``zeta_clean`` against the uncleaned null is
  biased low (-6.6 / -7.1 measured: the removed children stay in E*)
  and is asserted only as the specification wrote it (<= 2).
* Re-review of 2026-09-24 (four findings carried by the checks below;
  the numbers are the reviewer's on fresh seeds, reproduced before the
  harness was amended):
  - Detection is regime-specific (check 5 now prints the overall figure
    including the bilateral children, 86 % here): the
    >= 95 % holds on this generator's flat 200-localization clusters
    with an honest lpz of 77 nm; on a realistic generator (dispersed
    sizes, a range of reported lpz, D-19 depth factors) the size rule
    flags well under half of the matched children, more with a larger
    lpz_scale and more of the unilateral ones (``PairLeak`` docstring).
  - Rule (iii) reads True on leak-only axons (check 9, M1 direction):
    beyond the dead zone the tail is 1 - Phi(115 / 77) = 6.8 % of a
    cluster, P[Bin(200, 0.068) >= 10] ~ 0.85, so most children survive
    g = 60 (44 -> 32 here) and zeta stays > 0 (1.6 / 1.2, 2.5 / 2.3 and
    2.4 / 1.2 at g = 60 on three fresh leak-only axons). The check
    asserts the survival and PRINTS the rule's value and the zetas.
  - The parent+child profile premise (check 12): p > 0.05 in 3 of 7
    here, 71 % and 82 % on fresh seeds, 22 of 32 = 69 % pooled.
  - The unrolled pcf under leak (section 15): ``exclude_clusters``
    removes the children only; the real clusters that absorbed a
    neighbour's leak are displaced toward it (+6.5 to +8.8 nm at 20-90
    nm separation, purity ~0.85), so g(0) stays ~1.29 with every
    truth-spurious cluster removed (p_global <= 0.05 on 4 of 12 pairs)
    while the true generator arcs give 0.995. The section prints the
    three configurations, the true-arc pcf and the displacement, and
    asserts only the bookkeeping and the true-arc null.
* Budget (2026-09-23): the seed-0 axons get their guard runs in the
  FIRST ``analyze_leak`` call (``axon_leak`` forces ``guard=True`` for
  seed 0), so no axon is analysed twice; the determinism and driver
  checks of section 13 run on the M1 seed-2 axon (K = 30, the smallest
  cached one) instead of M5 seed 0; the permutation loop of check 8
  passes ``clean_null=False`` (200 eclipse reruns would cost 200 s and
  test nothing about the permutation p). With the module's
  component-major EM the run takes ~8 minutes (102 before).
* ``GuardRun`` carries no rings, so "the g = 0 rebuild reproduces the
  run's rings and clusters exactly" is checked in two halves: the
  module's g = 0 run against ``cols.adjacent`` (ring indices, K and
  n_locs per ring, n_matched, zeta, the null counts bit for bit: same
  seed) and the harness's OWN g = 0 rebuild with the same call against
  the original rings (ring count, K, n_locs per cluster, centroids to
  1e-9). The children count at g = 60 comes from the harness's own
  rebuild relabelled by truth (measured 44 -> 45 -> 32 -> 21 at g = 0 /
  30 / 60 / 90 on seed 0; the middle ring's K RISES at g = 60, 48 ->
  53, because the neighbours still leak 17 localizations into the
  narrower window, so the K per ring is printed, not compared).
* Check 10's tolerances: the sd of the fitted mu of a 0.5 / 0.5 mixture
  at 2.2 sigma separation is 4.4 nm at N = 2000 (measured over 30 seeds
  with a reference EM; 3.2 nm was the first estimate), not the 2.4 nm of
  a Gaussian mean: mu is asked within 10 nm (2.3 sd; the seed is fixed
  and the measured value is -1.18 nm), sigma within 6 nm (3.4 sd of the
  measured 1.76), weights within 0.04 (1.7 sd of 0.023), and the
  module's fit is compared with the harness's
  reference EM on the same data (loglik not lower by more than 1e-6
  relative, mu and sigma within 1 nm, weights within 0.005).
* N_min (check 11): (b) is asserted on the (0.8, 0.2) case, where
  pi_hat = 1 - max(weights) is the minority weight; for (0.5, 0.5) the
  estimator is folded at 0.5 and its sd is 0.85 x CRLB by construction
  (measured), printed only. (c): the harness's Fisher-information
  sample size at the TRUE pi with mu and sigma known is 12.8 events for
  (0.5, 0.5) and 56 for (0.8, 0.2), while E[LLR] = 2 N KL(mixture ||
  best Gaussian) = 0.032 N and 0.045 N puts the bootstrap LRT's 80 %
  power near N = 300 and 200: a factor 20 and 4, not 3. The factor-3
  band the specification wrote contradicts its own docstring (the bound
  with the nuisance known "cannot be reached" by the free fit); the
  check asserts the bound direction (N_80 >= n_min_80 >= n_min_events)
  and prints the ratios as the finding. The module's ``n_min_80_events``
  field is checked against the harness's formula at the fit's own
  (pi_hat, mu, sigma) to 1e-3 relative (different quadratures).
* Power floors of check 11 are the specification's; the KL figures
  above predict them (0.5 / 0.5 at N = 500: E[LLR] 16 against a null
  95th percentile of a few units).
* Determinism (check 13) is run with ``guard=False`` (the 4 rebuilds
  cost 60 s; ``build_rings`` determinism is H1's, section 6 of
  validate_rings_h1); the digest drops every ``seconds`` field. The
  different-seed run uses ``dataclasses.replace(cols, random_seed=1)``
  (the seed is D-22's ``cols.random_seed``).
* Budget: seeds 1 and 2 of check 5 use K = 30 per ring (18 000
  localizations) and ``guard=False``; the guard runs (4 rebuilds of
  ~15 s each) are made on the seed-0 axons only; ``profile_n_bootstrap
  = 99`` and ``RingsParams(n_bootstrap=20)`` everywhere in the loops.
  build_rings on 24 000 localizations: 15-20 s; analyze_columns at
  n_null 199: 2-4 s (measured). Every section prints its time.

Run:  venv\\Scripts\\python.exe validate_leak.py                     (from the repo root)
      venv\\Scripts\\python.exe validate_leak.py --with-regression   (+ H1, H2 and H3 harnesses)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom, norm, spearmanr

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import validate_columns_h3 as h3  # noqa: E402  (imports validate_rings_h1 as h3.h1; runs nothing)
import validate_rings_h1 as h1  # noqa: E402
from tools.mps_axial_precision import (  # noqa: E402
    leak_fraction,
    spurious_cluster_probability,
    structural_width_nm,
)
from tools.mps_columns import (  # noqa: E402
    RingsParams,
    build_rings,
    load_columns_params,
    rings_params_from,
)
from tools.mps_matching import analyze_columns  # noqa: E402
from tools.mps_paint import link_localizations  # noqa: E402

PARAMS_YAML = os.path.join(REPO_ROOT, "config", "columns_params.yaml")
PARAMS = load_columns_params(PARAMS_YAML)

PASSED = 0
FAILED = 0
T_START = time.perf_counter()
N_NULL = 199                 # analyze_columns null in every loop (H3's integration value)
PROFILE_B = 99               # profile_n_bootstrap in the loops
SEEDS = (0, 1, 2)            # check 5 pools these
K_EXTRA_SEEDS = 30           # K per ring for seeds 1 and 2 (budget)


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


def require(*names: str) -> Any:
    """The H4 names of ``tools.mps_leak``, or a RuntimeError naming what is
    missing, so that a check fails with the reason and never with a
    NameError further down (the module does not exist until the
    implementation lands; the harness is written first)."""
    try:
        import tools.mps_leak as ml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"tools.mps_leak not importable ({type(exc).__name__}: {exc}); H4 not implemented")
    missing = [n for n in names if not hasattr(ml, n)]
    if missing:
        raise RuntimeError(f"tools.mps_leak has no {', '.join(missing)} (H4 not implemented)")
    got = tuple(getattr(ml, n) for n in names)
    return got[0] if len(got) == 1 else got


def need(state: Dict[str, Any], key: str) -> Any:
    """The result an earlier check of the section produced, or a clear error."""
    if key not in state:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return state[key]


def is_nan(v: Any) -> bool:
    return isinstance(v, (float, np.floating)) and math.isnan(float(v))


def elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.1f} s"


def leak_params(**kw: Any) -> Any:
    """``LeakParams`` with the loop bootstrap (99) unless overridden."""
    LeakParams = require("LeakParams")
    kw.setdefault("profile_n_bootstrap", PROFILE_B)
    return LeakParams(**kw)


# ============================================================ synthetic truth
@dataclass
class LeakAxon:
    """One leak axon in the axon frame (no tilt), with its truth per
    localization: ring and cluster of origin (global id r * K + j),
    fluorophore id, true position, frame, and the per-localization
    precisions handed to ``build_rings``."""

    name: str
    model: str
    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    true_x_nm: NDArray[np.float64]
    true_y_nm: NDArray[np.float64]
    true_z_nm: NDArray[np.float64]
    frame: NDArray[np.int64]
    ring: NDArray[np.int64]
    cluster: NDArray[np.int64]
    fluorophore: NDArray[np.int64]
    lp_lateral_nm: NDArray[np.float64]
    lpz_nm: NDArray[np.float64]
    ring_z_nm: Tuple[float, ...]
    cluster_xy_nm: NDArray[np.float64]      # (3K, 2) true centre per global cluster id
    cluster_ring: NDArray[np.int64]         # (3K,) ring of each global cluster id
    K: int
    n_fluor: int
    period_nm: float
    sigma_z_nm: float
    sigma_struct_nm: float
    mean_locs_per_fluor: float
    total_frames: int
    lateral_noise_nm: float
    fluor_spread_nm: float
    lpz_jitter: float

    @property
    def n_locs(self) -> int:
        return int(self.x_nm.size)


def make_leak_axon(
    seed: int, *, model: str, K: int = 40, n_fluor: int = 100, mean_locs_per_fluor: float = 2.0,
    sigma_z_nm: float = 77.0, sigma_struct_nm: float = 0.0, period_nm: float = 170.0,
    lateral_noise_nm: float = 8.0, fluor_spread_nm: float = 8.0, d_min_nm: float = 80.0,
    total_frames: int = 60000, lpz_jitter: float = 0.1,
) -> LeakAxon:
    """See the module docstring. Localizations are in fluorophore order,
    consecutive frames within a fluorophore; every random draw comes
    from ``default_rng(seed)`` in a fixed order."""
    if model not in ("M1", "M5"):
        raise ValueError(f"model must be 'M1' or 'M5', got {model!r}")
    rng = np.random.default_rng(seed)
    ring_z = np.array([-period_nm, 0.0, period_nm])
    if model == "M5":
        s = np.sort(h3.hard_core_arcs_nm(rng, K, d_min_nm=d_min_nm))
        centres = np.vstack([h3.ellipse_point_nm(s)] * 3)
    else:
        centres = np.vstack([h3.ellipse_point_nm(np.sort(h3.hard_core_arcs_nm(rng, K, d_min_nm=d_min_nm)))
                             for _ in range(3)])
    n_cl = 3 * K
    n_fl = n_cl * n_fluor
    fl_cluster = np.repeat(np.arange(n_cl), n_fluor)
    fl_ring = fl_cluster // K
    px = centres[fl_cluster, 0] + rng.normal(0.0, fluor_spread_nm, n_fl)
    py = centres[fl_cluster, 1] + rng.normal(0.0, fluor_spread_nm, n_fl)
    pz = ring_z[fl_ring] + rng.normal(0.0, sigma_struct_nm, n_fl)
    r = rng.geometric(1.0 / mean_locs_per_fluor, n_fl)          # support {1, 2, ...}, mean 1 / p
    start = rng.integers(0, total_frames - r + 1)
    loc_fl = np.repeat(np.arange(n_fl), r)
    within = np.arange(int(r.sum())) - np.repeat(np.cumsum(r) - r, r)
    frame = (start[loc_fl] + within).astype(np.int64)
    n = int(loc_fl.size)
    lpz = sigma_z_nm * rng.uniform(1.0 - lpz_jitter, 1.0 + lpz_jitter, n)
    x = px[loc_fl] + rng.normal(0.0, lateral_noise_nm, n)
    y = py[loc_fl] + rng.normal(0.0, lateral_noise_nm, n)
    z = pz[loc_fl] + rng.normal(0.0, 1.0, n) * lpz
    return LeakAxon(
        name=f"leak-{model}-seed{seed}", model=model, x_nm=x, y_nm=y, z_nm=z,
        true_x_nm=px[loc_fl], true_y_nm=py[loc_fl], true_z_nm=pz[loc_fl], frame=frame,
        ring=fl_ring[loc_fl].astype(np.int64), cluster=fl_cluster[loc_fl].astype(np.int64),
        fluorophore=loc_fl.astype(np.int64), lp_lateral_nm=np.full(n, lateral_noise_nm), lpz_nm=lpz,
        ring_z_nm=tuple(float(v) for v in ring_z), cluster_xy_nm=centres,
        cluster_ring=(np.arange(n_cl) // K).astype(np.int64), K=K, n_fluor=n_fluor,
        period_nm=period_nm, sigma_z_nm=sigma_z_nm, sigma_struct_nm=sigma_struct_nm,
        mean_locs_per_fluor=mean_locs_per_fluor, total_frames=total_frames,
        lateral_noise_nm=lateral_noise_nm, fluor_spread_nm=fluor_spread_nm, lpz_jitter=lpz_jitter)


def leak_rings_params(**kw: Any) -> RingsParams:
    """The rings of every leak axon: no tilt (the axon is generated in its
    frame), 20 bootstrap draws (the centroid bootstrap is not under test)."""
    kw.setdefault("correct_tilt", False)
    kw.setdefault("n_bootstrap", 20)
    return RingsParams(**kw)


def run_leak_rings(axon: LeakAxon, params: Optional[RingsParams] = None, **kw: Any) -> Any:
    return build_rings(
        axon.x_nm.copy(), axon.y_nm.copy(), axon.z_nm.copy(),
        frame=kw.pop("frame", axon.frame.copy()),
        lp_lateral_nm=kw.pop("lp_lateral_nm", axon.lp_lateral_nm.copy()),
        lpz_nm=kw.pop("lpz_nm", axon.lpz_nm.copy()),
        params=params if params is not None else leak_rings_params(),
        source_name=axon.name, n_frames=kw.pop("n_frames", axon.total_frames), **kw)


@dataclass
class ClusterTruth:
    """A detected cluster labelled by the majority ring of origin of its
    localizations: ``spurious`` when that ring is not the ring it was
    detected in; ``parent`` the detected cluster of the origin ring
    holding most of its fluorophores (None when none holds any)."""

    ring: int
    index: int
    n_locs: int
    majority_ring: int
    purity: float
    majority_cluster: int
    spurious: bool
    parent: Optional[Tuple[int, int]]


def label_clusters(axon: LeakAxon, res: Any) -> Dict[Tuple[int, int], ClusterTruth]:
    """Truth labels of every detected cluster (see ``ClusterTruth``). The
    detected ring index k must be true ring k (rings ascend in z'); a
    different mapping raises, so the labels below never silently refer
    to the wrong ring."""
    by_ring = {int(r.index): r for r in res.rings}
    for k, ring in by_ring.items():
        maj = int(np.bincount(axon.ring[np.asarray(ring.loc_index)], minlength=3).argmax())
        if maj != k:
            raise AssertionError(f"detected ring {k} holds mostly true ring {maj}")
    fl_sets: Dict[Tuple[int, int], NDArray[np.int64]] = {}
    out: Dict[Tuple[int, int], ClusterTruth] = {}
    for k, ring in by_ring.items():
        for i, cl in enumerate(ring.clusters):
            li = np.asarray(cl.loc_index, dtype=np.intp)
            fl_sets[(k, i)] = np.unique(axon.fluorophore[li])
            counts = np.bincount(axon.ring[li], minlength=3)
            maj = int(counts.argmax())
            cl_counts = np.bincount(axon.cluster[li], minlength=axon.cluster_xy_nm.shape[0])
            out[(k, i)] = ClusterTruth(ring=k, index=i, n_locs=int(cl.n_locs), majority_ring=maj,
                                       purity=float(counts[maj] / li.size),
                                       majority_cluster=int(cl_counts.argmax()), spurious=maj != k, parent=None)
    for key, ct in out.items():
        if not ct.spurious or ct.majority_ring not in by_ring:
            continue
        li = np.asarray(by_ring[ct.ring].clusters[ct.index].loc_index, dtype=np.intp)
        fl = axon.fluorophore[li]
        best, best_n = None, 0
        for i in range(len(by_ring[ct.majority_ring].clusters)):
            n_shared = int(np.isin(fl, fl_sets[(ct.majority_ring, i)]).sum())
            if n_shared > best_n:
                best, best_n = (ct.majority_ring, i), n_shared
        ct.parent = best
    return out


def match_lookup(cols: Any) -> Dict[Tuple[int, int], Any]:
    return {(int(m.ring_a), int(m.ring_b)): m for m in cols.adjacent}


def matched_pair_key(cols: Any, a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """The (ring_a, ring_b, i_a, j_b) key of the adjacent match pairing
    clusters ``a`` and ``b`` (either order), or None when unmatched."""
    (ra, ia), (rb, jb) = (a, b) if a[0] < b[0] else (b, a)
    m = match_lookup(cols).get((ra, rb))
    if m is None:
        return None
    hit = np.flatnonzero((np.asarray(m.i_a) == ia) & (np.asarray(m.j_b) == jb))
    return (ra, rb, ia, jb) if hit.size else None


def children_status(axon: LeakAxon, res: Any, cols: Any) -> List[Tuple[ClusterTruth, Optional[Tuple[int, int, int, int]]]]:
    """Every spurious child with the key of its match to its parent (None
    when unmatched or without a parent in an adjacent ring)."""
    out = []
    for ct in label_clusters(axon, res).values():
        if not ct.spurious:
            continue
        key = None
        if ct.parent is not None and abs(ct.parent[0] - ct.ring) == 1:
            key = matched_pair_key(cols, ct.parent, (ct.ring, ct.index))
        out.append((ct, key))
    return out


def true_pairs(axon: LeakAxon, res: Any, cols: Any) -> List[Tuple[int, int, int, int]]:
    """M5: the matched pairs whose two clusters are non-spurious and sit
    at the same true arc position (global cluster ids congruent mod K)."""
    truth = label_clusters(axon, res)
    keys = []
    for m in cols.adjacent:
        for i, j in zip(np.asarray(m.i_a).tolist(), np.asarray(m.j_b).tolist()):
            ca, cb = truth[(int(m.ring_a), int(i))], truth[(int(m.ring_b), int(j))]
            if not ca.spurious and not cb.spurious and ca.majority_cluster % axon.K == cb.majority_cluster % axon.K:
                keys.append((int(m.ring_a), int(m.ring_b), int(i), int(j)))
    return keys


def pair_index(leak: Any) -> Dict[Tuple[int, int, int, int], Any]:
    """Every ``PairLeak`` of a ``LeakDiagnostics`` by (ring_a, ring_b, i_a, j_b)."""
    out: Dict[Tuple[int, int, int, int], Any] = {}
    for rp in leak.pairs:
        for p in rp.pairs:
            out[(int(p.ring_a), int(p.ring_b), int(p.i_a), int(p.j_b))] = p
    return out


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score_pos > score_neg) + P(equal) / 2 by brute force (finite values)."""
    p = np.asarray([v for v in pos if np.isfinite(v)], dtype=float)
    q = np.asarray([v for v in neg if np.isfinite(v)], dtype=float)
    if p.size == 0 or q.size == 0:
        return float("nan")
    diff = p[:, None] - q[None, :]
    return float((np.mean(diff > 0) + 0.5 * np.mean(diff == 0)))


def quartiles(v: Sequence[float]) -> str:
    a = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    return "n/a" if a.size == 0 else str(np.round(np.percentile(a, [25, 50, 75]), 3).tolist())


# ============================================================ first principles
def trunc_logpdf(z: NDArray[np.float64], mu: Any, sigma: Any, lo: float, hi: float) -> NDArray[np.float64]:
    """log density of N(mu, sigma) truncated to [lo, hi] (mu, sigma
    broadcast). The mass is taken from the survival functions when mu
    lies below the window's midpoint (the window is then in the upper
    tail and sf(lo) - sf(hi) has no cancellation) and from the cdfs
    otherwise: exact down to windows ~37 sigma away, where the module's
    first version hit its 1e-300 floor at ~8 sigma (review 2026-09-23)."""
    mu_a, s_a = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    upper = norm.sf(lo, mu_a, s_a) - norm.sf(hi, mu_a, s_a)
    lower = norm.cdf(hi, mu_a, s_a) - norm.cdf(lo, mu_a, s_a)
    mass = np.where(mu_a <= 0.5 * (lo + hi), upper, lower)
    return np.asarray(norm.logpdf(z, mu, sigma) - np.log(np.maximum(mass, 1e-300)), dtype=float)


def truth_tail_llr(z: NDArray[np.float64], lo: float, hi: float, mu_c: float, s_c: float,
                   mu_p: float, s_p: NDArray[np.float64]) -> float:
    return float(np.sum(trunc_logpdf(z, mu_p, s_p, lo, hi) - trunc_logpdf(z, mu_c, s_c, lo, hi)))


def log_ratio_moments(lo: float, hi: float, mu_c: float, s_c: float, mu_p: float, s_p: float,
                      under: str) -> Tuple[float, float]:
    """Expectation and sd of log f_tail - log f_centred under the tail
    ("tail") or the centred ("centred") truncated density, by trapezoid
    quadrature on the window."""
    z = np.linspace(lo, hi, 200001)
    lt = trunc_logpdf(z, mu_p, s_p, lo, hi)
    lc = trunc_logpdf(z, mu_c, s_c, lo, hi)
    dens = np.exp(lt if under == "tail" else lc)
    lr = lt - lc
    m = float(np.trapezoid(dens * lr, z))
    v = float(np.trapezoid(dens * (lr - m) ** 2, z))
    return m, math.sqrt(max(v, 0.0))


def truncated_draws(rng: np.random.Generator, n: int, mu: float, sigma: float, lo: float, hi: float) -> NDArray[np.float64]:
    """n draws of N(mu, sigma) conditioned on [lo, hi] by rejection."""
    out: List[NDArray[np.float64]] = []
    got = 0
    mass = float(norm.cdf(hi, mu, sigma) - norm.cdf(lo, mu, sigma))
    while got < n:
        d = rng.normal(mu, sigma, int(1.2 * (n - got) / mass) + 100)
        d = d[(d >= lo) & (d <= hi)]
        out.append(d)
        got += d.size
    return np.concatenate(out)[:n]


def mixture_info(pi: float, mu: float, sigma: float, period: float, n_components: int = 2,
                 weights: Optional[NDArray[np.float64]] = None) -> float:
    """Fisher information for pi in f = (1 - pi) f1 + pi f2, f1 = N(mu,
    sigma), f2 = the (1 - w0)-normalised rest of the fixed-offset mixture
    (for 2 components N(mu + P, sigma)), mu and sigma known:
    I = integral (f2 - f1)^2 / (pi f2 + (1 - pi) f1) dz (trapezoid)."""
    z = np.linspace(mu - 8 * sigma, mu + (n_components - 1) * period + 8 * sigma, 400001)
    f1 = norm.pdf(z, mu, sigma)
    if n_components == 2 or weights is None:
        f2 = norm.pdf(z, mu + period, sigma)
    else:
        w = np.asarray(weights, dtype=float)
        rest = w[1:] / w[1:].sum()
        f2 = sum(rest[m - 1] * norm.pdf(z, mu + m * period, sigma) for m in range(1, n_components))
    return float(np.trapezoid((f2 - f1) ** 2 / (pi * f2 + (1.0 - pi) * f1), z))


def n_min_prereg(pi: float, period: float, sigma: float, z: float = 1.959964) -> float:
    return math.inf if pi == 0.0 else z * z / (pi * pi * (math.exp((period / sigma) ** 2) - 1.0))


def n_min_80(pi: float, info: float, power: float = 0.8, z: float = 1.959964) -> float:
    return math.inf if pi == 0.0 else (z + float(norm.ppf(power))) ** 2 / (pi * pi * info)


def gaussian_loglik(z: NDArray[np.float64]) -> Tuple[float, float, float]:
    """(mu, sigma, loglik) of the one-Gaussian MLE (ddof 0)."""
    mu, s = float(z.mean()), float(z.std())
    return mu, s, float(norm.logpdf(z, mu, s).sum())


def reference_em(z: NDArray[np.float64], period: float, m: int, iters: int = 2000, tol: float = 1e-10) -> Tuple[float, float, NDArray[np.float64], float]:
    """First-principles EM of the fixed-offset mixture (components at mu +
    k P, shared sigma, free weights), best of several starts; the
    reference the module's ``fixed_offset_mixture_fit`` is compared with."""
    best: Optional[Tuple[float, float, NDArray[np.float64], float]] = None
    ks = np.arange(m)
    starts = [(float(z.mean()) - k * period * 0.5, float(z.std())) for k in range(m)]
    starts += [(float(z.mean()) - k * period, float(z.std()) / 2.0) for k in range(m)]
    for mu0, s0 in starts:
        mu, s, w = mu0, max(s0, 1.0), np.full(m, 1.0 / m)
        ll_old = -np.inf
        ll = -np.inf
        for _ in range(iters):
            lp = np.log(np.maximum(w, 1e-300))[None, :] + norm.logpdf(z[:, None], (mu + period * ks)[None, :], s)
            mx = lp.max(axis=1, keepdims=True)
            ll = float((mx.ravel() + np.log(np.exp(lp - mx).sum(axis=1))).sum())
            r = np.exp(lp - mx)
            r /= r.sum(axis=1, keepdims=True)
            w = r.mean(axis=0)
            mu = float((r * (z[:, None] - period * ks[None, :])).sum() / z.size)
            s = max(float(np.sqrt((r * (z[:, None] - (mu + period * ks)[None, :]) ** 2).sum() / z.size)), 1e-3)
            if ll - ll_old < tol:
                break
            ll_old = ll
        if best is None or ll > best[3]:
            best = (mu, s, w, ll)
    assert best is not None
    return best


def digest(obj: Any, drop: Tuple[str, ...] = ("seconds",)) -> Any:
    """A JSON-able mirror of a result (dataclasses, arrays, NaN as 'nan'),
    without the timing fields."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: digest(getattr(obj, f.name), drop) for f in dataclasses.fields(obj) if f.name not in drop}
    if isinstance(obj, dict):
        return {str(k): digest(v, drop) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [digest(v, drop) for v in obj]
    if isinstance(obj, np.ndarray):
        return digest(obj.tolist(), drop)
    if isinstance(obj, np.generic):
        return digest(obj.item(), drop)
    if isinstance(obj, float) and math.isnan(obj):
        return "nan"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def with_event_id(res: Any, event_id: NDArray[np.int64]) -> Any:
    """A copy of ``res`` whose events are ``event_id`` (per input
    localization): ring and cluster bookkeeping recomputed."""
    ev = np.asarray(event_id, dtype=np.int64)
    rings = []
    for ring in res.rings:
        li = np.asarray(ring.loc_index, dtype=np.intp)
        clusters = [dataclasses.replace(cl, n_events=int(np.unique(ev[np.asarray(cl.loc_index)][ev[np.asarray(cl.loc_index)] >= 0]).size))
                    for cl in ring.clusters]
        rings.append(dataclasses.replace(ring, event_id=ev[li], n_events=int(np.unique(ev[li][ev[li] >= 0]).size),
                                         clusters=clusters))
    return dataclasses.replace(res, rings=rings, event_id=ev, n_events=int(np.unique(ev[ev >= 0]).size))


def with_cluster_lpz(res: Any, values: Dict[int, NDArray[np.float64]]) -> Any:
    """A copy of ``res`` with the clusters' ``lpz_median_nm`` replaced per ring."""
    rings = []
    for ring in res.rings:
        if ring.index in values:
            v = values[ring.index]
            ring = dataclasses.replace(ring, clusters=[dataclasses.replace(cl, lpz_median_nm=float(v[i]))
                                                       for i, cl in enumerate(ring.clusters)])
        rings.append(ring)
    return dataclasses.replace(res, rings=rings)


# ============================================================ cached runs
RUNS: Dict[Tuple[str, int], Dict[str, Any]] = {}


def axon_base(model: str, seed: int) -> Dict[str, Any]:
    """The leak axon, its rings (harness) and its H3 columns (existing
    module) for (model, seed); built once, no H4 code involved."""
    st = RUNS.setdefault((model, seed), {})
    if "cols" not in st:
        t0 = time.perf_counter()
        axon = make_leak_axon(seed, model=model, K=40 if seed == 0 else K_EXTRA_SEEDS)
        res = run_leak_rings(axon)
        t1 = time.perf_counter()
        cols = analyze_columns(res, PARAMS, n_null=N_NULL)
        st.update(axon=axon, res=res, cols=cols, truth=label_clusters(axon, res))
        print(f"      [{model} seed {seed}: {axon.n_locs} locs, build_rings {t1 - t0:.1f} s -> {len(res.rings)} rings, "
              f"K {[len(r.clusters) for r in res.rings]}; analyze_columns {time.perf_counter() - t1:.1f} s]")
    return st


def axon_leak(model: str, seed: int, *, guard: bool = False) -> Dict[str, Any]:
    """``axon_base`` plus the module's ``analyze_leak``. The seed-0 axons
    always get their guard runs (sections 9 and 13 need them), in the
    FIRST call, so that no axon's profiles are fitted twice; the other
    seeds run without guard unless asked."""
    analyze_leak = require("analyze_leak")      # before the rings: a missing module fails fast
    st = axon_base(model, seed)
    guard = bool(guard or seed == 0)
    if "leak" not in st or (guard and not st.get("guard")):
        axon, res, cols = st["axon"], st["res"], st["cols"]
        t0 = time.perf_counter()
        st["leak"] = analyze_leak(res, cols, params=leak_params(), lpz_nm=axon.lpz_nm.copy(), frame=axon.frame.copy(),
                                  lp_lateral_nm=axon.lp_lateral_nm.copy(), n_frames=axon.total_frames, guard=guard)
        st["guard"] = guard
        print(f"      [{model} seed {seed}: analyze_leak(guard={guard}) {time.perf_counter() - t0:.1f} s]")
    return st


# ============================================================ 0. API
def test_api() -> None:
    print("\n0. API of tools/mps_leak.py and LeakParams")

    def api_present():
        names = ("LeakParams", "PairLeak", "RingPairLeak", "AxialProfileFit", "GuardRun", "LeakDiagnostics",
                 "child_tail_llr", "boundary_fraction", "boundary_fraction_expected", "shared_events", "pair_leak",
                 "ring_pair_leak", "axon_period_nm", "fixed_offset_mixture_fit", "axial_profile_fit",
                 "column_axial_profile", "guard_sensitivity", "analyze_leak", "analyze_columns_and_leak")
        require(*names)
        import tools.mps_leak as ml
        with open(ml.__file__, "r", encoding="utf-8") as fh:
            text = fh.read()
        tree = ast.parse(text)
        top = [alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
        top += [(node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)]
        banned = [m for m in top if m in ("PyQt5", "PyQt6", "PySide2", "PySide6", "matplotlib", "qtpy")]
        assert not banned, banned
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        undocumented = [n for n, node in funcs.items() if ast.get_docstring(node) is None]
        assert not undocumented, undocumented
        assert "@author: Nicolas (ngomez) + Claude" in "\n".join(text.splitlines()[:60]), "author header missing"
        assert "from __future__ import annotations" in text
        return f"{len(names)} names importable; no Qt/matplotlib; {len(funcs)} functions documented"

    def params_defaults_and_ranges():
        LeakParams = require("LeakParams")
        p = LeakParams()
        assert tuple(p.guards_nm) == (0.0, 30.0, 60.0, 90.0) and p.guard_rule_nm == 60.0
        assert p.shared_threshold == 0.25 and p.n_null_shared == 1999 and p.boundary_band_sigma == 1.0
        assert p.lpz_scale == 1.0 and p.size_alpha == 0.01 and p.profile_n_bootstrap == 199
        assert p.profile_min_locs_per_cluster == 100 and p.profile_alpha == 0.05
        assert abs(p.n_min_z - 1.959964) < 1e-9 and p.n_min_power == 0.8
        assert not hasattr(p, "random_seed"), "the seed is cols.random_seed (D-22), not a LeakParams field"
        for kw in ({"guards_nm": (-1.0, 30.0)}, {"shared_threshold": 1.5}, {"guards_nm": (60.0, 30.0)},
                   {"n_null_shared": 0}, {"size_alpha": -0.1}, {"profile_n_bootstrap": 0}):
            try:
                LeakParams(**kw)
            except ValueError:
                pass
            else:
                raise AssertionError(f"LeakParams({kw}) accepted")
        return "defaults as specified; negative guard, non-increasing guards, threshold > 1, count 0 raise ValueError"

    check("tools.mps_leak: 19 public names, no Qt/matplotlib, every function documented, author header", api_present)
    check("LeakParams: defaults as specified, no random_seed field, ValueError on out-of-range values",
          params_defaults_and_ranges)


# ============================================================ 1. generator truth
def test_generator_truth() -> None:
    print("\n1. GENERATOR TRUTH (M1 seed 0, 24 000 localizations, P = 170 nm, sigma_z = 77 nm)")
    st: Dict[str, Any] = {}

    def leak_fraction_truth():
        axon = make_leak_axon(0, model="M1")
        st["axon"] = axon
        assert abs(axon.n_locs - 24000) < 1500, axon.n_locs
        half = axon.period_nm / 2.0
        f_true = float(np.mean(norm.sf(half / axon.lpz_nm)))
        dz = axon.z_nm - np.asarray(axon.ring_z_nm)[axon.ring]
        above, below = float(np.mean(dz > half)), float(np.mean(dz < -half))
        # Binomial sd of a fraction of 24 000 at F = 0.134: 0.0022; +-0.01 is 4.5 sd.
        assert abs(above - f_true) <= 0.01 and abs(below - f_true) <= 0.01, (above, below, f_true)
        assert abs(f_true - norm.sf(half / axon.sigma_z_nm)) < 0.005
        return (f"n {axon.n_locs}; beyond +P/2 {above:.4f}, beyond -P/2 {below:.4f}, F (mean over lpz_i) {f_true:.4f}, "
                f"1 - Phi(85/77) {norm.sf(half / 77.0):.4f}; two-sided {np.mean(np.abs(dz) > half):.4f} = 2F")

    def linking_exact():
        axon = need(st, "axon")
        radius = 5.0 * 8.0
        ev = np.asarray(link_localizations(axon.frame, axon.x_nm, axon.y_nm, radius, max_dark_time=1))
        n_fl = int(axon.fluorophore.max()) + 1
        pairs = np.unique(np.column_stack([axon.fluorophore, ev]), axis=0)
        fl_n_events = np.bincount(pairs[:, 0], minlength=n_fl)
        ev_n_fl = np.bincount(pairs[:, 1])
        ev_of_fl = np.full(n_fl, -1)
        ev_of_fl[pairs[:, 0]] = pairs[:, 1]
        exact = (fl_n_events == 1) & (ev_n_fl[ev_of_fl] == 1)
        # Exposure: another fluorophore's localization within the radius and <= 2 frames away.
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([axon.x_nm, axon.y_nm]))
        pp = np.array(sorted(tree.query_pairs(radius)), dtype=int).reshape(-1, 2)
        i, j = pp[:, 0], pp[:, 1]
        bad = (np.abs(axon.frame[i] - axon.frame[j]) <= 2) & (axon.fluorophore[i] != axon.fluorophore[j])
        exposed = np.zeros(n_fl, dtype=bool)
        exposed[axon.fluorophore[i[bad]]] = True
        exposed[axon.fluorophore[j[bad]]] = True
        frac_all = float(exact.mean())
        frac_clean = float(exact[~exposed].mean())
        # Radius loss: exp(-25 / 4) = 0.2 % of steps, ~1 step per fluorophore -> >= 99.5 % of the unexposed.
        assert frac_clean >= 0.995, frac_clean
        assert frac_all >= 0.95, frac_all
        mean_locs = axon.n_locs / np.unique(ev).size
        assert abs(mean_locs - axon.mean_locs_per_fluor) <= 0.05 * axon.mean_locs_per_fluor, mean_locs
        return (f"exact fluorophores {100 * frac_all:.2f} % overall, {100 * frac_clean:.2f} % of the {100 * (1 - exposed.mean()):.1f} % "
                f"not exposed to a temporal merge; mean locs/event {mean_locs:.3f} (truth 2.0)")

    def m5_truth():
        axon = make_leak_axon(0, model="M5")
        c = axon.cluster_xy_nm
        assert np.array_equal(c[:axon.K], c[axon.K:2 * axon.K]) and np.array_equal(c[:axon.K], c[2 * axon.K:])
        d = np.hypot(*(c[:axon.K, None, :] - c[None, :axon.K, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        assert d.min() >= 60.0, d.min()          # 80 nm arc hard core -> chords >= 60 nm on this ellipse
        assert np.array_equal(axon.ring, axon.cluster // axon.K)
        return f"M5: identical positions on the 3 rings, nearest centres {d.min():.1f} nm apart; {axon.n_locs} locs"

    check("one-sided fraction beyond +-P/2 equals F = mean(1 - Phi(85/lpz_i)) within 0.01 (binomial sd 0.002)",
          leak_fraction_truth)
    check("linking at 5 x 8 nm: >= 99.5 % of unexposed fluorophores are one event (>= 95 % overall); mean locs/event within 5 % of 2",
          linking_exact)
    check("M5 generator: the three rings share their cluster positions (hard core 80 nm arc)", m5_truth)


# ============================================================ 2. children
def test_children() -> None:
    print("\n2. SPURIOUS CHILDREN on the M1 axon (seed 0) and their matching to the parents")
    st: Dict[str, Any] = {}

    def children_exist():
        base = axon_base("M1", 0)
        axon, res, cols = base["axon"], base["res"], base["cols"]
        assert len(res.rings) == 3, [(r.z_lo_nm, r.z_hi_nm) for r in res.rings]
        status = children_status(axon, res, cols)
        st["status"] = status
        n = len(status)
        sizes = [ct.n_locs for ct, _ in status]
        per_dir: Dict[Tuple[int, int], int] = {}
        for ct, _ in status:
            per_dir[(ct.majority_ring, ct.ring)] = per_dir.get((ct.majority_ring, ct.ring), 0) + 1
        assert n >= 20, n
        assert all(ct.parent is not None for ct, _ in status)
        expected = spurious_cluster_probability(200, float(norm.sf(85.0 / 77.0)), 10)
        return (f"{n} spurious children (origin -> ring: {dict(sorted(per_dir.items()))}); sizes quartiles {quartiles(sizes)}; "
                f"P[Bin(200, 0.135) >= 10] = {expected:.4f}; K per ring {[len(r.clusters) for r in res.rings]}; "
                f"windows {[(round(r.z_lo_nm), round(r.z_hi_nm)) for r in res.rings]}")

    def matched_to_parent():
        status = need(st, "status")
        base = axon_base("M1", 0)
        matched = [key for _, key in status if key is not None]
        frac = len(matched) / len(status)
        d = []
        idx = match_lookup(base["cols"])
        for ra, rb, ia, jb in matched:
            m = idx[(ra, rb)]
            k = np.flatnonzero((np.asarray(m.i_a) == ia) & (np.asarray(m.j_b) == jb))[0]
            d.append(float(np.asarray(m.d_nm)[k]))
        assert frac >= 0.9, frac
        return f"{len(matched)} of {len(status)} children matched to their parent within tau0 {PARAMS.tau0_nm:.2f} nm ({100 * frac:.1f} %); d quartiles {quartiles(d)} nm"

    check("M1 seed 0: >= 20 spurious children (majority ring of origin != detected ring), each with a parent", children_exist)
    check(">= 90 % of the spurious children are matched to their parent by cols.adjacent (d ~ few nm)", matched_to_parent)


# ============================================================ 3. shared events
def test_shared_events() -> None:
    print("\n3. SHARED EVENTS: hand-made counts, children (M1) against true pairs (M5)")

    def hand_made():
        shared_events = require("shared_events")
        ev = np.array([0, 0, 1, 2, 3, 3, -1, -1, 4, 5], dtype=np.int64)
        got = shared_events(ev, np.array([0, 1, 2, 6, 8]), np.array([1, 3, 4, 7, 9]))
        assert tuple(int(v) for v in got) == (1, 3, 4), got            # Ea {0,1,4}, Eb {0,2,3,5}
        got2 = shared_events(ev, np.array([2, 3]), np.array([4, 8]))
        assert tuple(int(v) for v in got2) == (0, 2, 2), got2
        got3 = shared_events(ev, np.array([6]), np.array([7]))
        assert tuple(int(v) for v in got3) == (0, 0, 0), got3
        got4 = shared_events(ev, np.array([0, 1, 4, 5]), np.array([0, 5, 6]))
        assert tuple(int(v) for v in got4) == (2, 2, 2), got4
        return "(1, 3, 4), (0, 2, 2), (0, 0, 0) with -1 only, (2, 2, 2): exact"

    def children_share_more():
        m1 = axon_leak("M1", 0)
        m5 = axon_leak("M5", 0)
        p1 = pair_index(m1["leak"])
        p5 = pair_index(m5["leak"])
        child_keys = [key for _, key in children_status(m1["axon"], m1["res"], m1["cols"]) if key is not None]
        true_keys = true_pairs(m5["axon"], m5["res"], m5["cols"])
        sc = [float(p1[k].shared_fraction) for k in child_keys]
        stt = [float(p5[k].shared_fraction) for k in true_keys]
        assert len(sc) >= 20 and len(stt) >= 20, (len(sc), len(stt))
        assert all(np.isfinite(sc)) and all(np.isfinite(stt))
        a = auc(sc, stt)
        assert np.mean(sc) > np.mean(stt) and a >= 0.9, (np.mean(sc), np.mean(stt), a)
        return (f"children (n {len(sc)}): mean {np.mean(sc):.3f}, quartiles {quartiles(sc)}; true pairs (n {len(stt)}): "
                f"mean {np.mean(stt):.3f}, quartiles {quartiles(stt)}; AUC {a:.3f}")

    check("shared_events on hand-made index arrays with -1 ids: exact (|Ea & Eb|, |Ea|, |Eb|)", hand_made)
    check("shared_fraction of M1 children > that of M5 true pairs; AUC >= 0.9 (argument: ~half of a child's events "
          "have a localization on each side; a true child shares only the ~F n_events_parent leaked events)", children_share_more)


# ============================================================ 4. tail LLR
def test_tail_llr() -> None:
    print("\n4. child_tail_llr, boundary_fraction, boundary_fraction_expected against first principles")
    lo, hi, mu_c, s_c, mu_p = 85.0, 255.0, 170.0, 77.0, 0.0

    def llr_identity():
        child_tail_llr = require("child_tail_llr")
        rng = np.random.default_rng(4)
        worst = 0.0
        for _ in range(20):
            n = int(rng.integers(5, 60))
            z = rng.uniform(lo, hi, n)
            s_p = rng.uniform(60.0, 95.0, n)
            got = float(child_tail_llr(z, win_lo_nm=lo, win_hi_nm=hi, mu_centred_nm=mu_c, sigma_centred_nm=s_c,
                                       mu_parent_nm=mu_p, sigma_parent_nm=s_p))
            exp = truth_tail_llr(z, lo, hi, mu_c, s_c, mu_p, s_p)
            worst = max(worst, abs(got - exp) / max(1.0, abs(exp)))
        assert worst <= 1e-9, worst
        # Narrow tail widths (review 2026-09-23): with sigma_parent 5-12 nm the window [85, 255] lies 7-17
        # sigma from mu_parent = 0 and the truncated mass is 1e-13..1e-64; the module's first version
        # floored it at 1e-300 (log -690.8 instead of -39.2 at sigma 10) and 3 localizations at 90-100 nm
        # gave llr +1943 instead of -11.4. Exact to 1e-9 relative is asked down to 5 nm, both window sides.
        worst_narrow = 0.0
        z3 = np.array([90.0, 95.0, 100.0])
        for s in (12.0, 10.0, 8.0, 5.0):
            for (wlo, whi, mp, mc) in ((lo, hi, mu_p, mu_c), (-hi, -lo, -mu_p, -mu_c)):
                zz = z3 if wlo > 0 else -z3
                got = float(child_tail_llr(zz, win_lo_nm=wlo, win_hi_nm=whi, mu_centred_nm=mc, sigma_centred_nm=s_c,
                                           mu_parent_nm=mp, sigma_parent_nm=np.full(3, s)))
                exp = truth_tail_llr(zz, wlo, whi, mc, s_c, mp, np.full(3, s))
                assert math.isfinite(got) and abs(got) < 1e4, (s, got)
                worst_narrow = max(worst_narrow, abs(got - exp) / max(1.0, abs(exp)))
        assert worst_narrow <= 1e-9, worst_narrow
        llr10 = float(child_tail_llr(z3, win_lo_nm=lo, win_hi_nm=hi, mu_centred_nm=mu_c, sigma_centred_nm=s_c,
                                     mu_parent_nm=mu_p, sigma_parent_nm=np.full(3, 10.0)))
        return (f"20 random cases (sigma 60-95), max relative difference {worst:.1e}; narrow widths 5-12 nm on both "
                f"window sides {worst_narrow:.1e} (3 locs at 90-100 nm, sigma 10: llr {llr10:.3f}, exact -11.424)")

    def sign_repetitions():
        child_tail_llr = require("child_tail_llr")
        rng = np.random.default_rng(5)
        R = 500
        out = []
        for n in (15, 50):
            m_t, sd_t = log_ratio_moments(lo, hi, mu_c, s_c, mu_p, 77.0, "tail")
            m_c, sd_c = log_ratio_moments(lo, hi, mu_c, s_c, mu_p, 77.0, "centred")
            z_t, z_c = math.sqrt(n) * m_t / sd_t, math.sqrt(n) * abs(m_c) / sd_c
            pred_t, pred_c = float(norm.cdf(z_t)), float(norm.cdf(z_c))
            sig = np.full(n, 77.0)
            pos = neg = 0
            for _ in range(R):
                zt = truncated_draws(rng, n, mu_p, 77.0, lo, hi)
                zc = truncated_draws(rng, n, mu_c, s_c, lo, hi)
                pos += float(child_tail_llr(zt, win_lo_nm=lo, win_hi_nm=hi, mu_centred_nm=mu_c, sigma_centred_nm=s_c,
                                            mu_parent_nm=mu_p, sigma_parent_nm=sig)) > 0.0
                neg += float(child_tail_llr(zc, win_lo_nm=lo, win_hi_nm=hi, mu_centred_nm=mu_c, sigma_centred_nm=s_c,
                                            mu_parent_nm=mu_p, sigma_parent_nm=sig)) < 0.0
            ft, fc = pos / R, neg / R
            floor_t, floor_c = (max(0.95, pred_t - 0.02), max(0.95, pred_c - 0.02)) if n == 15 else (0.99, 0.99)
            assert ft >= floor_t and fc >= floor_c, (n, ft, fc, floor_t, floor_c)
            out.append(f"n={n}: E[lr] {m_t:.3f}/{m_c:.3f}, separation {z_t:.2f}/{z_c:.2f} sd, predicted {pred_t:.4f}/{pred_c:.4f}, "
                       f"measured llr>0 {ft:.3f} (tail), llr<0 {fc:.3f} (centred), floors {floor_t:.3f}/{floor_c:.3f}")
        return "; ".join(out)

    def expected_fraction_vs_mc():
        boundary_fraction_expected = require("boundary_fraction_expected")
        rng = np.random.default_rng(6)
        out = []
        for (wlo, whi, mu, sg, b, band) in ((85.0, 255.0, 170.0, 77.0, 85.0, 77.0), (-94.0, 86.0, -4.0, 69.0, 86.0, 30.0)):
            got = float(boundary_fraction_expected(win_lo_nm=wlo, win_hi_nm=whi, mu_nm=mu, sigma_nm=sg, boundary_z_nm=b, band_nm=band))
            z = truncated_draws(rng, 1_000_000, mu, sg, wlo, whi)
            mc = float(np.mean(np.abs(z - b) <= band))
            a, c = max(b - band, wlo), min(b + band, whi)
            closed = (norm.cdf(c, mu, sg) - norm.cdf(a, mu, sg)) / (norm.cdf(whi, mu, sg) - norm.cdf(wlo, mu, sg))
            assert abs(got - mc) <= 0.005 and abs(got - closed) <= 1e-9, (got, mc, closed)
            out.append(f"{got:.4f} (MC {mc:.4f}, closed form {closed:.4f})")
        return "; ".join(out)

    def fraction_hand_made():
        boundary_fraction = require("boundary_fraction")
        z = np.array([80.0, 84.0, 90.0, 100.0, 200.0, 70.0])
        got = float(boundary_fraction(z, 85.0, 10.0))
        assert abs(got - 3.0 / 6.0) < 1e-12, got            # 80, 84, 90 within 10 of 85
        empty = float(boundary_fraction(np.array([]), 85.0, 10.0))
        assert empty == 0.0 or is_nan(empty), empty          # either is acceptable; a crash is not
        return f"3 of 6 within +-10 nm of 85: 0.5 exact; empty input gives {empty}"

    check("child_tail_llr equals the harness's truncated-normal log-likelihood ratio (1e-9 relative, per-localization sigma)",
          llr_identity)
    check("500 repetitions: tail draws give llr > 0 and centred draws llr < 0 (n=15: >= predicted - 0.02; n=50: >= 0.99)",
          sign_repetitions)
    check("boundary_fraction_expected equals 10^6 truncated draws within 0.005 and the closed form (1e-9)", expected_fraction_vs_mc)
    check("boundary_fraction on a hand-made array is exact", fraction_hand_made)


# ============================================================ 5. detection
def test_detection() -> None:
    print("\n5. DETECTION (03 S3.6): leak_explained on children (M1) and on true pairs (M5), 3 seeds pooled")
    st: Dict[str, Any] = {}

    def pooled_children():
        # Population (module docstring): the tail model has ONE parent, so the acceptance is asserted on
        # the unilateral children (purity >= 0.9); the bilateral ones (leak from both neighbours at a
        # chance-coincident arc, purity 0.5-0.7, z' at both edges) are the model's stated limitation.
        n_child = n_uni = n_expl = n_tail = n_size = n_bil = n_bil_expl = 0
        per_seed = []
        for seed in SEEDS:
            run = axon_leak("M1", seed)
            pl = pair_index(run["leak"])
            status = children_status(run["axon"], run["res"], run["cols"])
            uni = [k for ct, k in status if k is not None and ct.purity >= 0.9]
            bil = [k for ct, k in status if k is not None and ct.purity < 0.9]
            n_child += len(status)
            n_uni += len(uni)
            n_bil += len(bil)
            e = sum(bool(pl[k].leak_explained) for k in uni)
            t = sum(bool(pl[k].z_tail) for k in uni)
            s = sum(bool(pl[k].size_ok) for k in uni)
            eb = sum(bool(pl[k].leak_explained) for k in bil)
            n_expl += e
            n_tail += t
            n_size += s
            n_bil_expl += eb
            per_seed.append(f"seed {seed}: {e}/{len(uni)} unilateral (z_tail {t}, size_ok {s}), {eb}/{len(bil)} bilateral; "
                            f"{len(status)} children")
            st.setdefault("child_keys", {})[seed] = uni
            st.setdefault("bilateral_keys", {})[seed] = bil
        frac = n_expl / n_uni
        assert frac >= 0.95, (n_expl, n_uni, frac)
        overall = (n_expl + n_bil_expl) / (n_uni + n_bil)
        return (f"leak_explained {n_expl}/{n_uni} = {frac:.3f} of the matched UNILATERAL children (z_tail {n_tail}, size_ok "
                f"{n_size}); bilateral children {n_bil} of {n_uni + n_bil} matched, {n_bil_expl} flagged (single-parent model: "
                f"stated limitation); OVERALL detection {n_expl + n_bil_expl}/{n_uni + n_bil} = {overall:.3f} on THIS regime "
                f"(flat ~200-loc clusters, honest lpz 77 nm; lower in a realistic regime, see the module docstring); "
                + "; ".join(per_seed))

    def pooled_true_pairs():
        n_pairs = n_expl = n_tail = n_size = 0
        per_seed = []
        for seed in SEEDS:
            run = axon_leak("M5", seed)
            pl = pair_index(run["leak"])
            keys = true_pairs(run["axon"], run["res"], run["cols"])
            e = sum(bool(pl[k].leak_explained) for k in keys)
            n_tail += sum(bool(pl[k].z_tail) for k in keys)
            n_size += sum(bool(pl[k].size_ok) for k in keys)
            n_pairs += len(keys)
            n_expl += e
            per_seed.append(f"seed {seed}: {e}/{len(keys)}")
            st.setdefault("true_keys", {})[seed] = keys
        frac = n_expl / n_pairs
        assert frac <= 0.10, (n_expl, n_pairs, frac)
        return f"leak_explained {n_expl}/{n_pairs} = {frac:.3f} (z_tail {n_tail}, size_ok {n_size}); " + "; ".join(per_seed)

    def child_parent_rule():
        # The child is the cluster with fewer localizations (tie -> ring b); indices consistent with (i_a, j_b).
        n = 0
        for seed in SEEDS:
            for model in ("M1", "M5"):
                run = axon_leak(model, seed)
                for rp in run["leak"].pairs:
                    for p in rp.pairs:
                        assert p.n_locs_child <= p.n_locs_parent
                        if p.n_locs_child == p.n_locs_parent:
                            assert p.child_ring == p.ring_b
                        a, b = (p.ring_a, p.i_a), (p.ring_b, p.j_b)
                        assert {(p.parent_ring, p.parent_index), (p.child_ring, p.child_index)} == {a, b}
                        assert p.n_locs_child > 0 and 0 < p.size_ratio <= 1.0
                        assert abs(p.size_ratio - p.n_locs_child / p.n_locs_parent) < 1e-12
                        n += 1
        return f"{n} PairLeak over 6 axons: child = fewer localizations (tie -> ring b), indices and size_ratio consistent"

    check("M1, 3 seeds pooled: >= 95 % of the matched UNILATERAL spurious children (purity >= 0.9) have leak_explained "
          "(z_tail and size_ok printed; bilateral children counted and printed)", pooled_children)
    check("M5, 3 seeds pooled: <= 10 % of the matched TRUE pairs have leak_explained (n_child ~ 150-200 cannot be Bin(200, F))",
          pooled_true_pairs)
    check("PairLeak: child/parent rule and index bookkeeping consistent on every pair", child_parent_rule)


# ============================================================ 6. F and sizes
def test_f_and_sizes() -> None:
    print("\n6. F_parent, expected_leaked_locs, p_size_leak, p_spurious")

    def reference_numbers():
        f = leak_fraction(0.0, 85.0, np.full(200, 77.0))
        assert abs(f - norm.sf(85.0 / 77.0)) < 1e-12
        assert abs(200.0 * f - 26.96) < 0.01           # 200 x 0.134819 = 26.964
        p27 = float(binom.sf(26, 200, 0.135))
        assert abs(200 * 0.135 - 27.0) < 1e-9
        ps = spurious_cluster_probability(200, 0.135, 10)
        assert ps > 0.999 and abs(ps - binom.sf(9, 200, 0.135)) < 1e-12
        return f"F = 1 - Phi(85/77) = {f:.5f}; 200 F = {200 * f:.3f}; P[Bin(200, .135) >= 27] = {p27:.4f}; p_spurious(200, .135, 10) = {ps:.6f}"

    def identities_on_pairs():
        run = axon_leak("M1", 0)
        res, cols, leak = run["res"], run["cols"], run["leak"]
        params = leak.params
        by_ring = {r.index: r for r in res.rings}
        lpz = run["axon"].lpz_nm
        n = 0
        worst = 0.0
        conventions = {"ring": 0, "cluster": 0}
        for rp in leak.pairs:
            for p in rp.pairs:
                parent_ring = by_ring[p.parent_ring]
                child_ring = by_ring[p.child_ring]
                parent = parent_ring.clusters[p.parent_index]
                below = parent_ring.centre_z_nm < child_ring.centre_z_nm
                boundary = child_ring.z_lo_nm if below else child_ring.z_hi_nm
                assert abs(p.boundary_z_nm - boundary) < 1e-9, (p.boundary_z_nm, boundary)
                s_struct = structural_width_nm(parent_ring.sigma_z_nm, parent_ring.lpz_median_nm * params.lpz_scale)
                lp = lpz[np.asarray(parent.loc_index)] * params.lpz_scale
                # The parent's original size (D-27): the detected count over the mass its own window retains.
                f_lo = leak_fraction(parent_ring.centre_z_nm, parent_ring.z_lo_nm, lp, s_struct)
                f_hi = leak_fraction(parent_ring.centre_z_nm, parent_ring.z_hi_nm, lp, s_struct)
                retained = 1.0 - f_lo - f_hi
                n0 = max(p.n_locs_parent, int(round(p.n_locs_parent / retained)))
                assert abs(p.parent_retained_fraction - retained) < 1e-12 and p.n_locs_parent_original == n0, (p.parent_retained_fraction, retained, p.n_locs_parent_original, n0)
                assert abs(p.F_parent_far - (f_lo if below else f_hi)) < 1e-12
                assert 0.5 < retained < 1.0 and n0 > p.n_locs_parent, (retained, n0, p.n_locs_parent)
                assert abs(p.expected_leaked_locs - n0 * p.F_parent) < 1e-9
                assert abs(p.p_size_leak - binom.sf(p.n_locs_child - 1, n0, p.F_parent)) < 1e-12
                assert abs(p.p_spurious - spurious_cluster_probability(n0, p.F_parent, int(cols.params.min_samples))) < 1e-12
                f_ring = leak_fraction(parent_ring.centre_z_nm, boundary, lp, s_struct)
                f_cluster = leak_fraction(float(np.mean(parent.z_values_nm)), boundary, lp, s_struct)
                if abs(p.F_parent - f_ring) < 1e-12:
                    conventions["ring"] += 1
                elif abs(p.F_parent - f_cluster) < 1e-12:
                    conventions["cluster"] += 1
                else:
                    worst = max(worst, abs(p.F_parent - f_ring))
                assert p.n_locs_parent == parent.n_locs and p.n_locs_child == child_ring.clusters[p.child_index].n_locs
                assert bool(p.size_ok) == (p.p_size_leak >= params.size_alpha)
                assert bool(p.z_tail) == (p.llr_tail_child > 0.0)
                assert bool(p.leak_explained) == (bool(p.z_tail) and bool(p.size_ok))
                n += 1
        assert conventions["cluster"] == 0, (f"F_parent uses the parent CLUSTER's mean z' in {conventions['cluster']} pairs; "
                                             "the specification's tail model uses the parent RING's centre_z_nm")
        assert conventions["ring"] == n, (conventions, n, worst)
        return (f"{n} pairs: boundary, n_locs_parent_original = round(n_parent / (1 - F_lo - F_hi)) from the parent ring's "
                f"window edges, F_parent_far, expected_leaked_locs = n0 F (1e-9), p_size_leak = binom.sf(n_child - 1, n0, F), "
                f"p_spurious (1e-12), F_parent (parent ring centre; 1e-12), flags")

    def size_ok_rates():
        n_c = ok_c = n_t = bad_t = 0
        for seed in SEEDS:
            m1, m5 = axon_leak("M1", seed), axon_leak("M5", seed)
            p1, p5 = pair_index(m1["leak"]), pair_index(m5["leak"])
            # unilateral children only (the size test has one parent; see pooled_children)
            ck = [k for ct, k in children_status(m1["axon"], m1["res"], m1["cols"]) if k is not None and ct.purity >= 0.9]
            tk = true_pairs(m5["axon"], m5["res"], m5["cols"])
            n_c += len(ck)
            ok_c += sum(bool(p1[k].size_ok) for k in ck)
            n_t += len(tk)
            bad_t += sum(not bool(p5[k].size_ok) for k in tk)
        assert ok_c / n_c >= 0.95 and bad_t / n_t >= 0.95, (ok_c, n_c, bad_t, n_t)
        return f"size_ok True for {ok_c}/{n_c} unilateral children ({100 * ok_c / n_c:.1f} %), False for {bad_t}/{n_t} true pairs ({100 * bad_t / n_t:.1f} %)"

    check("reference numbers (tools.mps_axial_precision): 200 (1 - Phi(85/77)) = 26.96, binom.sf(26, 200, .135), p_spurious ~ 1",
          reference_numbers)
    check("PairLeak identities on every M1 pair: boundary facing the parent, the parent's original size from its window's "
          "retained mass, expected_leaked_locs = n0 F (1e-9), p_size_leak = binom.sf(., n0, F) (1e-12), p_spurious, "
          "F_parent = leak_fraction(parent ring centre, ...) (1e-12), flags", identities_on_pairs)
    check("size_ok True for >= 95 % of the unilateral children (M1) and False for >= 95 % of the true pairs (M5), 3 seeds pooled",
          size_ok_rates)


# ============================================================ 7. clean excess
def test_clean_excess() -> None:
    print("\n7. CLEAN EXCESS: zeta_clean against the H3 null of the same pair")

    def m1_cleaned():
        run = axon_leak("M1", 0)
        out = []
        for rp, m in zip(run["leak"].pairs, run["cols"].adjacent):
            assert (rp.ring_a, rp.ring_b) == (m.ring_a, m.ring_b) and rp.n_matched == m.n_matched
            min_k = min(m.K_a, m.K_b)
            assert rp.n_matched_clean == rp.n_matched - rp.n_leak_explained
            assert abs(rp.E_dir_clean - rp.n_matched_clean / min_k) < 1e-12
            assert abs(rp.zeta_clean - (rp.n_matched_clean - m.E_star * min_k) / (m.sd_star * min_k)) < 1e-9
            assert abs(rp.E_excess - (m.E_dir - m.E_star)) < 1e-12
            assert m.zeta > 0.0 and rp.zeta_clean <= 2.0, (m.zeta, rp.zeta_clean)
            # The cleaned null (D-27): the eclipse test rerun without the flagged children on both sides, same
            # seed/tau/n_null as the run. On independent rings the cleaned count against the cleaned null is a
            # standard normal to first order: |zeta| <= 3 holds for 99.7 % of draws (measured 0.33 / 0.32).
            mc = rp.match_clean
            assert mc is not None and mc.n_null == m.n_null and mc.random_seed == m.random_seed and mc.tau_nm == m.tau_nm
            assert mc.K_a + mc.K_b == m.K_a + m.K_b - rp.n_leak_explained, (mc.K_a, mc.K_b, m.K_a, m.K_b, rp.n_leak_explained)
            assert rp.zeta_clean_null == mc.zeta and rp.p_excess_clean == mc.p_excess
            assert abs(rp.zeta_clean_null) <= 3.0, rp.zeta_clean_null
            out.append(f"{m.ring_a}-{m.ring_b}: n_matched {m.n_matched} (E* {m.E_star:.3f}, zeta {m.zeta:.2f}) -> clean {rp.n_matched_clean} "
                       f"(E_dir_clean {rp.E_dir_clean:.3f}, zeta_clean {rp.zeta_clean:.2f} against the uncleaned null: biased low); "
                       f"leak_explained {rp.n_leak_explained}; cleaned null: n_matched {mc.n_matched} of K {mc.K_a}/{mc.K_b}, "
                       f"E* {mc.E_star:.3f}, zeta {mc.zeta:.2f}, p_excess {mc.p_excess:.3f}")
        return "; ".join(out)

    def m5_survives():
        run = axon_leak("M5", 0)
        out = []
        for rp, m in zip(run["leak"].pairs, run["cols"].adjacent):
            assert rp.zeta_clean > 3.0, rp.zeta_clean
            assert rp.zeta_clean_null > 3.0, rp.zeta_clean_null
            if rp.n_leak_explained == 0:
                assert rp.match_clean is m, "nothing flagged: match_clean must be the run's own match"
            out.append(f"{m.ring_a}-{m.ring_b}: zeta {m.zeta:.2f} -> zeta_clean {rp.zeta_clean:.2f}, cleaned null zeta "
                       f"{rp.zeta_clean_null:.2f} (leak_explained {rp.n_leak_explained} of {m.n_matched})")
        return "; ".join(out)

    check("M1: zeta > 0 (inflated by the leak) and zeta_clean <= 2 on both adjacent pairs; identities of n_matched_clean, "
          "E_dir_clean, zeta_clean (1e-9) and E_excess; the cleaned null (children removed on both sides, same seed): "
          "|zeta| <= 3", m1_cleaned)
    check("M5: zeta_clean > 3 and the cleaned null's zeta > 3 on both adjacent pairs (the columns survive the cleaning; "
          "match_clean is the run's match when nothing is flagged)", m5_survives)


# ============================================================ 8. (i) and (ii)
def test_reassignment_and_spearman() -> None:
    print("\n8. (i) reassignment null p_shared and (ii) Spearman matched-vs-lpz with its permutation p")

    def p_shared_hand_made():
        ring_pair_leak = require("ring_pair_leak")
        base = axon_base("M5", 0)
        axon, res, cols = base["axon"], base["res"], base["cols"]
        m = cols.adjacent[0]
        n = axon.n_locs
        # (a) every localization its own event: no cluster shares anything.
        rp = ring_pair_leak(with_event_id(res, np.arange(n, dtype=np.int64)), m, params=leak_params(n_null_shared=199),
                            lpz_nm=axon.lpz_nm, min_samples=int(cols.params.min_samples), random_seed=0)
        assert rp.p_shared >= 0.5 and rp.shared_fraction_paired_mean == 0.0 and rp.shared_fraction_unpaired_mean == 0.0, \
            (rp.p_shared, rp.shared_fraction_paired_mean, rp.shared_fraction_unpaired_mean)
        assert rp.n_shared_high == 0 and all(p.shared_events == 0 and p.shared_high is False for p in rp.pairs)
        # (b) every matched child's events are a subset of its parent's: shared fraction 1 per pair.
        ev = np.arange(n, dtype=np.int64)
        ra, rb = {r.index: r for r in res.rings}[m.ring_a], {r.index: r for r in res.rings}[m.ring_b]
        for i, j in zip(np.asarray(m.i_a).tolist(), np.asarray(m.j_b).tolist()):
            a, b = ra.clusters[i], rb.clusters[j]
            parent, child = (a, b) if a.n_locs > b.n_locs else (b, a)
            ev[np.asarray(child.loc_index)] = np.resize(ev[np.asarray(parent.loc_index)], child.n_locs)
        rp2 = ring_pair_leak(with_event_id(res, ev), m, params=leak_params(n_null_shared=199),
                             lpz_nm=axon.lpz_nm, min_samples=int(cols.params.min_samples), random_seed=0)
        assert rp2.p_shared <= 0.05 and abs(rp2.shared_fraction_paired_mean - 1.0) < 1e-12, (rp2.p_shared, rp2.shared_fraction_paired_mean)
        assert rp2.n_shared_high == m.n_matched and abs(rp2.p_shared - 1.0 / 200.0) < 1e-12
        return (f"unique events: p_shared {rp.p_shared:.4f}, paired/unpaired means 0/0; child events within the parent's: "
                f"p_shared {rp2.p_shared:.4f} = 1/(B+1), paired mean 1.0, unpaired mean {rp2.shared_fraction_unpaired_mean:.4f}, "
                f"n_shared_high {rp2.n_shared_high} of {m.n_matched}")

    def rho_equals_spearmanr():
        run = axon_leak("M1", 0)
        res, cols, leak = run["res"], run["cols"], run["leak"]
        out = []
        for rp, m in zip(leak.pairs, cols.adjacent):
            by = {r.index: r for r in res.rings}
            ind, lpz = [], []
            for ring, usable, matched in ((by[m.ring_a], m.usable_a, np.asarray(m.i_a)), (by[m.ring_b], m.usable_b, np.asarray(m.j_b))):
                for i, cl in enumerate(ring.clusters):
                    if usable is None or bool(usable[i]):
                        ind.append(1.0 if i in set(matched.tolist()) else 0.0)
                        lpz.append(float(cl.lpz_median_nm))
            rho = float(spearmanr(ind, lpz).correlation)
            assert abs(rp.spearman_matched_vs_lpz_rho - rho) < 1e-12, (rp.spearman_matched_vs_lpz_rho, rho)
            assert 0.0 < rp.spearman_matched_vs_lpz_p <= 1.0
            assert abs(rp.lpz_pair_median_nm - float(np.median(lpz))) < 1e-9
            out.append(f"{m.ring_a}-{m.ring_b}: rho {rho:.4f} (p {rp.spearman_matched_vs_lpz_p:.3f}), lpz median {rp.lpz_pair_median_nm:.2f} nm")
        return "; ".join(out)

    def permutation_p_uniform():
        ring_pair_leak = require("ring_pair_leak")
        base = axon_base("M1", 0)
        axon, res, cols = base["axon"], base["res"], base["cols"]
        m = cols.adjacent[0]
        rng = np.random.default_rng(8)
        t0 = time.perf_counter()
        ps = []
        for k in range(200):
            values = {r.index: rng.uniform(60.0, 90.0, len(r.clusters)) for r in res.rings}
            # clean_null=False: the cleaned eclipse rerun (n_null 199 x 200 draws, ~200 s) tests nothing here.
            rp = ring_pair_leak(with_cluster_lpz(res, values), m, params=leak_params(n_null_shared=100),
                                lpz_nm=None, min_samples=int(cols.params.min_samples), random_seed=k, clean_null=False)
            ps.append(float(rp.spearman_matched_vs_lpz_p))
        frac = h3.frac_le(np.asarray(ps))
        lo, hi = h3.wilson_interval(0.05, 200)
        assert 0.01 <= frac <= 0.12, frac
        return f"fraction p <= 0.05: {frac:.3f} over 200 draws (Wilson [{lo:.3f}, {hi:.3f}], band [0.01, 0.12] for the +1 discreteness); {elapsed(t0)}"

    check("p_shared: unique events -> p >= 0.5 (means 0); child events within the parent's -> p = 1/(B+1), paired mean 1",
          p_shared_hand_made)
    check("spearman_matched_vs_lpz_rho equals scipy.stats.spearmanr on the pooled usable clusters (1e-12); p in (0, 1]",
          rho_equals_spearmanr)
    check("permutation p with an lpz independent of the indicator (100 permutations, 200 draws): fraction <= 0.05 in [0.01, 0.12]",
          permutation_p_uniform)


# ============================================================ 9. guard
def test_guard() -> None:
    print("\n9. GUARD SENSITIVITY: the g = 0 rebuild reproduces the run; children fall with the guard; M5 survives")
    st: Dict[str, Any] = {}

    def module_g0_reproduces_run():
        run = axon_leak("M1", 0, guard=True)
        leak, cols, res = run["leak"], run["cols"], run["res"]
        guards = tuple(float(g) for g in leak.params.guards_nm)
        assert tuple(sorted(float(g) for g in leak.guard)) == guards, (list(leak.guard), guards)
        g0 = leak.guard[0.0]
        assert g0.n_rings == 3 and list(g0.ring_indices) == [r.index for r in res.rings]
        assert list(g0.K_per_ring) == [len(r.clusters) for r in res.rings]
        assert list(g0.n_locs_per_ring) == [r.n_locs for r in res.rings]
        assert len(g0.matches) == len(cols.adjacent)
        for gm, m in zip(g0.matches, cols.adjacent):
            assert (gm.ring_a, gm.ring_b, gm.n_matched, gm.K_a, gm.K_b) == (m.ring_a, m.ring_b, m.n_matched, m.K_a, m.K_b)
            assert gm.tau_nm == m.tau_nm == cols.tau0_nm and gm.n_null == cols.n_null and gm.random_seed == cols.random_seed
            assert np.array_equal(np.asarray(gm.null_n_matched), np.asarray(m.null_n_matched)) and gm.zeta == m.zeta
        assert all(g.seconds >= 0.0 and all(isinstance(w, str) for w in g.warnings) for g in leak.guard.values())
        st["K"] = {g: list(leak.guard[g].K_per_ring) for g in guards}
        return f"g = 0: rings {list(g0.ring_indices)}, K {list(g0.K_per_ring)}, n_locs {list(g0.n_locs_per_ring)}, matches identical to cols.adjacent (null counts bit for bit)"

    def harness_rebuild_and_children():
        base = axon_base("M1", 0)
        axon, res, cols = base["axon"], base["res"], base["cols"]

        def rebuild(g: float) -> Any:
            return build_rings(res.x_p, res.y_p, res.z_p, frame=axon.frame.copy(), lp_lateral_nm=axon.lp_lateral_nm.copy(),
                               lpz_nm=axon.lpz_nm.copy(),
                               params=dataclasses.replace(rings_params_from(cols.params), guard_nm=g, correct_tilt=False),
                               source_name=res.source_name, n_frames=axon.total_frames)
        t0 = time.perf_counter()
        r0 = rebuild(0.0)
        assert len(r0.rings) == len(res.rings)
        worst = 0.0
        for a, b in zip(res.rings, r0.rings):
            assert (a.index, a.z_lo_nm, a.z_hi_nm, a.n_locs) == (b.index, b.z_lo_nm, b.z_hi_nm, b.n_locs)
            assert len(a.clusters) == len(b.clusters) and [c.n_locs for c in a.clusters] == [c.n_locs for c in b.clusters]
            worst = max(worst, max(float(np.abs(np.asarray(ca.centroid_nm) - np.asarray(cb.centroid_nm)).max())
                                   for ca, cb in zip(a.clusters, b.clusters)))
        assert worst <= 1e-9, worst
        r60 = rebuild(60.0)
        n0 = sum(ct.spurious for ct in label_clusters(axon, res).values())
        n60 = sum(ct.spurious for ct in label_clusters(axon, r60).values())
        assert n60 < n0, (n0, n60)
        st["children"] = (n0, n60)
        ks = st.get("K", {})
        return (f"harness g = 0 rebuild: identical rings/clusters, centroids to {worst:.1e} nm; children {n0} at g = 0 -> {n60} at g = 60 "
                f"(windows {[(round(r.z_lo_nm), round(r.z_hi_nm)) for r in r60.rings]}); module K per ring by guard {ks}; {elapsed(t0)}")

    def m5_positive_at_every_guard():
        run = axon_leak("M5", 0, guard=True)
        leak = run["leak"]
        out = []
        for g in sorted(leak.guard):
            gr = leak.guard[g]
            assert gr.n_rings == 3 and len(gr.matches) == 2, (g, gr.n_rings, len(gr.matches))
            assert all(m.zeta > 0.0 for m in gr.matches), (g, [m.zeta for m in gr.matches])
            out.append(f"g {g:g}: K {list(gr.K_per_ring)}, zeta {[round(m.zeta, 2) for m in gr.matches]}")
        assert leak.rules["iii_guard"] is True
        return "; ".join(out) + "; rule iii True"

    def m1_leak_survives_guard():
        # The M1 direction of rule (iii) (re-review of 2026-09-24): the pre-specified dead zone does not remove
        # the leak. Physics: beyond 85 + 30 = 115 nm the tail of N(0, lpz_i) is mean_i(1 - Phi(115 / lpz_i)),
        # 6.8 % at lpz 77, i.e. 13.5 of 200 localizations, above min_samples 10 with probability
        # P[Bin(200, F60) >= 10] ~ 0.85, so most children persist at g = 60 (DBSCAN fragmentation lowers it;
        # measured 32 of 44 on this axon) and the excess they carry stays positive. Asserted: at least half the
        # children survive g = 60 (the physics' 0.85 minus fragmentation) and the rule's value is a bool; the
        # value itself and the zetas by guard are PRINTED as the finding -- a True here is not evidence
        # against leak, and the report's rules table says so.
        run = axon_leak("M1", 0, guard=True)
        axon, leak = run["axon"], run["leak"]
        n0, n60 = need(st, "children")
        guard_half = leak.params.guard_rule_nm / 2.0
        f60 = float(np.mean(norm.sf((axon.period_nm / 2.0 + guard_half) / axon.lpz_nm)))
        n_per = axon.n_fluor * axon.mean_locs_per_fluor
        p_survive = float(binom.sf(PARAMS.min_samples - 1, int(round(n_per)), f60))
        assert n60 > 0 and n60 >= 0.5 * n0, (n0, n60)
        assert isinstance(leak.rules["iii_guard"], bool)
        rows = []
        for g in sorted(leak.guard):
            gr = leak.guard[g]
            rows.append(f"g {g:g}: K {list(gr.K_per_ring)}, zeta {[round(m.zeta, 2) for m in gr.matches]}, "
                        f"p_excess {[round(m.p_excess, 3) for m in gr.matches]}")
        return (f"leak-only axon: children {n0} at g = 0 -> {n60} at g = 60 ({n60 / n0:.2f} survive; physics: tail beyond "
                f"{axon.period_nm / 2 + guard_half:.0f} nm F60 = {f60:.3f}, P[Bin({int(round(n_per))}, F60) >= "
                f"{PARAMS.min_samples}] = {p_survive:.2f}); rule iii reads {leak.rules['iii_guard']} on INDEPENDENT rings "
                f"(the dead zone does not separate leak from columns); " + "; ".join(rows))

    check("module guard_sensitivity at g = 0: ring indices, K, n_locs per ring and the matches equal the run's (same seed)",
          module_g0_reproduces_run)
    check("harness rebuild on res.x_p/y_p/z_p with correct_tilt=False at g = 0 reproduces the rings exactly (centroids 1e-9); "
          "spurious children at g = 60 < at g = 0 (truth relabel)", harness_rebuild_and_children)
    check("M5: zeta > 0 at every guard, 3 rings at g = 90 (rule iii holds for real columns)", m5_positive_at_every_guard)
    check("M1 (leak only): >= half the children survive g = 60 (tail beyond 115 nm = 6.8 %, P[Bin(200, F) >= 10] ~ 0.85), so "
          "rule iii cannot separate leak from columns; its value and the zetas by guard are PRINTED", m1_leak_survives_guard)


# ============================================================ 10. profile fit
def test_profile_fit() -> None:
    print("\n10. fixed_offset_mixture_fit against the truth and a reference EM")
    P, S = 170.0, 77.0

    def two_components():
        fit = require("fixed_offset_mixture_fit")
        rng = np.random.default_rng(10)
        k = rng.random(2000) < 0.5
        z = rng.normal(0.0, S, 2000) + P * k
        mu, sigma, w, ll = fit(z, P, 2)
        w = np.asarray(w, dtype=float)
        rmu, rs, rw, rll = reference_em(z, P, 2)
        assert abs(mu) <= 10.0 and abs(sigma - S) <= 6.0 and np.abs(w - 0.5).max() <= 0.04, (mu, sigma, w)
        assert ll >= rll - 1e-6 * abs(rll), (ll, rll)
        assert abs(mu - rmu) <= 1.0 and abs(sigma - rs) <= 1.0 and np.abs(w - rw).max() <= 0.005, (mu, rmu, sigma, rs, w, rw)
        assert abs(w.sum() - 1.0) < 1e-9
        return f"mu {mu:.2f} nm (ref {rmu:.2f}), sigma {sigma:.2f} (ref {rs:.2f}), weights {np.round(w, 4).tolist()} (ref {np.round(rw, 4).tolist()}), loglik {ll:.4f} (ref {rll:.4f})"

    def single_gaussian_never_loses():
        fit = require("fixed_offset_mixture_fit")
        rng = np.random.default_rng(11)
        worst_gap, worst_one = 0.0, 0.0
        for _ in range(20):
            z = rng.normal(30.0, S, 200)
            mu1, s1, ll1 = gaussian_loglik(z)
            _, _, _, ll2 = fit(z, P, 2)
            assert ll2 >= ll1 - 1e-9 * abs(ll1), (ll2, ll1)
            worst_gap = max(worst_gap, ll2 - ll1)
            mu, sigma, w, ll = fit(z, P, 1)
            worst_one = max(worst_one, abs(mu - mu1), abs(sigma - s1), abs(ll - ll1) / abs(ll1))
            assert np.allclose(np.asarray(w, dtype=float), [1.0])
        assert worst_one <= 1e-6, worst_one
        return f"20 single-Gaussian samples: loglik2 - loglik1 in [0, {worst_gap:.4f}]; the 1-component fit equals the closed form to {worst_one:.1e}"

    def three_components():
        fit = require("fixed_offset_mixture_fit")
        rng = np.random.default_rng(12)
        comp = rng.choice(3, 2000, p=[0.4, 0.3, 0.3])
        z = rng.normal(0.0, S, 2000) + P * comp
        mu, sigma, w, ll = fit(z, P, 3)
        w = np.asarray(w, dtype=float)
        rmu, rs, rw, rll = reference_em(z, P, 3)
        assert np.abs(w - [0.4, 0.3, 0.3]).max() <= 0.04 and abs(mu) <= 10.0 and abs(sigma - S) <= 6.0, (w, mu, sigma)
        assert ll >= rll - 1e-6 * abs(rll)
        return f"weights {np.round(w, 4).tolist()} (truth .4/.3/.3, ref {np.round(rw, 4).tolist()}), mu {mu:.2f}, sigma {sigma:.2f}"

    check("2 components on 2000 draws of 0.5 N(0,77) + 0.5 N(170,77): mu within 10 nm, sigma 6 nm, weights 0.04; equals the reference EM",
          two_components)
    check("loglik2 >= loglik1 on 20 single-Gaussian samples (one-Gaussian start); the 1-component fit is the closed-form MLE",
          single_gaussian_never_loses)
    check("3 components with weights (0.4, 0.3, 0.3) on 2000 draws: weights within 0.04", three_components)


# ============================================================ 11. profile test
def test_profile_test() -> None:
    print("\n11. axial_profile_fit: FPR, power table and N_min (03 S3.3 'Perfil axial')")
    P, S = 170.0, 77.0
    st: Dict[str, Any] = {}

    def fpr():
        axial_profile_fit = require("axial_profile_fit")
        params = leak_params()
        rng = np.random.default_rng(110)
        t0 = time.perf_counter()
        ps, worst = [], 0.0
        for k in range(200):
            z = rng.normal(0.0, S, 100)
            f = axial_profile_fit(z, P, 2, n_bootstrap=PROFILE_B, random_seed=k, params=params)
            ps.append(float(f.p_bootstrap))
            _, _, ll1 = gaussian_loglik(z)
            assert abs(f.loglik1 - ll1) <= 1e-6 * abs(ll1) and f.llr >= -1e-9, (f.loglik1, ll1, f.llr)
            assert abs(f.llr - 2.0 * (f.loglik2 - f.loglik1)) < 1e-9 and f.n_bootstrap == PROFILE_B and f.random_seed == k
            worst = max(worst, abs(f.p_bootstrap * (PROFILE_B + 1) - round(f.p_bootstrap * (PROFILE_B + 1))))
        frac = h3.frac_le(np.asarray(ps))
        assert worst < 1e-9, "p_bootstrap is not on the (b + 1) / (B + 1) grid"
        assert 0.02 <= frac <= 0.09, frac
        lo, hi = h3.wilson_interval(0.05, 200)
        return f"fraction p <= 0.05: {frac:.3f} of 200 (Wilson [{lo:.3f}, {hi:.3f}]; band [0.02, 0.09]); {elapsed(t0)}"

    def power_table():
        axial_profile_fit = require("axial_profile_fit")
        params = leak_params()
        rng = np.random.default_rng(111)
        R = 100
        grid = (50, 100, 200, 500)
        t0 = time.perf_counter()
        table: Dict[Tuple[float, float], List[float]] = {}
        fits: Dict[Tuple[float, float], List[Any]] = {}
        seed = 0
        for w in ((0.5, 0.5), (0.8, 0.2)):
            powers = []
            for N in grid:
                rej = 0
                for _ in range(R):
                    z = rng.normal(0.0, S, N) + P * (rng.random(N) < w[1])
                    f = axial_profile_fit(z, P, 2, n_bootstrap=PROFILE_B, random_seed=seed, params=params)
                    seed += 1
                    rej += f.p_bootstrap <= 0.05
                    if N == 500:
                        fits.setdefault(w, []).append(f)
                powers.append(rej / R)
            table[w] = powers
        st["table"], st["fits"], st["grid"], st["R"] = table, fits, grid, R
        lines = []
        for w, powers in table.items():
            sd = [math.sqrt(p * (1 - p) / R) for p in powers]
            inversions = [k for k in range(len(powers) - 1) if powers[k + 1] < powers[k]]
            assert len(inversions) <= 1, (w, powers)
            for k in inversions:
                assert powers[k] - powers[k + 1] <= 2.0 * math.hypot(sd[k], sd[k + 1]), (w, powers)
            lines.append(f"{w}: " + ", ".join(f"N={n} {p:.2f}" for n, p in zip(grid, powers)))
        assert table[(0.5, 0.5)][-1] >= 0.95, table[(0.5, 0.5)]
        return "power " + "; ".join(lines) + f" (R = {R}, B = {PROFILE_B}); {elapsed(t0)}"

    def n_min_closed_form():
        fits = need(st, "fits")
        worst = 0.0
        n = 0
        for w, fl in fits.items():
            for f in fl:
                exp = n_min_prereg(float(f.pi_hat), float(f.period_nm), float(f.sigma_nm), 1.959964)
                assert abs(f.pi_hat - (1.0 - float(np.max(f.weights)))) < 1e-12
                if math.isinf(exp):
                    assert math.isinf(f.n_min_events)
                else:
                    worst = max(worst, abs(f.n_min_events - exp) / max(1e-12, abs(exp)))
                n += 1
        assert worst <= 1e-9, worst
        return f"{n} fits: n_min_events = z^2 / (pi^2 (exp((P/sigma)^2) - 1)) to {worst:.1e} relative; pi_hat = 1 - max(weights)"

    def crlb_efficiency():
        fits = need(st, "fits")
        out = []
        for w in ((0.8, 0.2), (0.5, 0.5)):
            pis = np.array([float(f.pi_hat) for f in fits[w]])
            info = mixture_info(w[1], 0.0, S, P)
            crlb = 1.0 / math.sqrt(500 * info)
            ratio = float(np.std(pis, ddof=1)) / crlb
            out.append(f"{w}: sd(pi_hat) {np.std(pis, ddof=1):.4f}, mean {pis.mean():.3f}, CRLB sd {crlb:.4f} (I = {info:.3f}), ratio {ratio:.2f}")
            if w == (0.8, 0.2):
                assert 0.9 <= ratio <= 3.0, ratio
        return "; ".join(out) + " ((0.5, 0.5) folded at 0.5: printed only)"

    def n_80_vs_bounds():
        table, fits, grid = need(st, "table"), need(st, "fits"), need(st, "grid")
        out = []
        for w in ((0.5, 0.5), (0.8, 0.2)):
            powers = table[w]
            n80 = math.inf
            for k in range(len(grid)):
                if powers[k] >= 0.8:
                    if k == 0:
                        n80 = float(grid[0])
                    else:
                        n80 = grid[k - 1] + (grid[k] - grid[k - 1]) * (0.8 - powers[k - 1]) / (powers[k] - powers[k - 1])
                    break
            pi = w[1]
            info = mixture_info(pi, 0.0, S, P)
            wald = n_min_80(pi, info)
            prereg = n_min_prereg(pi, P, S)
            assert n80 >= wald >= prereg, (n80, wald, prereg)
            # The module's field against the harness's formula at the fit's own values (10 fits, 1e-3 relative).
            worst = 0.0
            for f in fits[w][:10]:
                if f.pi_hat > 0.0:
                    mine = n_min_80(float(f.pi_hat), mixture_info(float(f.pi_hat), float(f.mu_nm), float(f.sigma_nm), P),
                                    float(f.n_min_power), float(f.n_min_z) if hasattr(f, "n_min_z") else 1.959964)
                    worst = max(worst, abs(f.n_min_80_events - mine) / mine)
            assert worst <= 1e-3, worst
            out.append(f"{w}: N_80 (interpolated) {n80:.0f}, n_min_80 at true pi {wald:.1f} (ratio {n80 / wald:.1f}), "
                       f"pre-specified n_min {prereg:.3f} (ratio {n80 / prereg:.0f}); module n_min_80_events vs formula {worst:.1e}")
        return "; ".join(out) + " -- the factor-3 band of the specification is not met by the physics (see the docstring)"

    check("FPR: 200 samples of N = 100 from one Gaussian, B = 99: fraction p <= 0.05 in [0.02, 0.09]; llr = 2 (loglik2 - loglik1) >= 0, "
          "loglik1 the closed-form MLE, p on the (b+1)/(B+1) grid", fpr)
    check("power table (0.5, 0.5) and (0.8, 0.2) at N = 50/100/200/500, R = 100, B = 99: non-decreasing in N (one inversion within 2 MC sd); "
          "power >= 0.95 at N = 500 for (0.5, 0.5)", power_table)
    check("N_min (a): n_min_events equals the pre-specified closed form (1e-9)", n_min_closed_form)
    check("N_min (b): sd of pi_hat at N = 500 (0.8, 0.2) within [0.9, 3] x CRLB sd (mu, sigma known); (0.5, 0.5) printed", crlb_efficiency)
    check("N_min (c)/(d): N_80 >= n_min_80 (Wald, true pi) >= n_min_events (pre-specified); ratios printed; module n_min_80_events "
          "equals the harness formula at the fit's values (1e-3)", n_80_vs_bounds)


# ============================================================ 12. column profiles
def test_column_profiles() -> None:
    print("\n12. COLUMN PROFILES: length-3 columns of M5, parent+child columns of M1")

    def event_z_truth(res: Any, column: Any) -> NDArray[np.float64]:
        by = {r.index: r for r in res.rings}
        ids = np.unique(np.concatenate([np.asarray(res.event_id)[np.asarray(by[k].clusters[i].loc_index)] for k, i in column.members]))
        ids = ids[ids >= 0]
        ev = np.asarray(res.event_id)
        out = np.array([float(np.mean(res.z_p[ev == e])) for e in ids])
        return np.sort(out)

    def m5_length_3():
        run = axon_leak("M5", 0)
        axon, res, cols, leak = run["axon"], run["res"], run["cols"], run["leak"]
        axon_period_nm = require("axon_period_nm")
        by = {r.index: r for r in res.rings}
        long_cols = [c for c in cols.columns if c.length >= 2]
        assert len(leak.profiles) == len(long_cols), (len(leak.profiles), len(long_cols))
        centres = np.array([r.centre_z_nm for r in res.rings])
        period = float(np.median(np.diff(centres) / np.diff([r.index for r in res.rings])))
        assert abs(leak.period_nm - period) < 1e-9 and abs(float(axon_period_nm(res.rings)) - period) < 1e-9
        n3 = n_sig = n_checked = 0
        worst_z = 0.0
        for prof, col in zip(leak.profiles, long_cols):
            assert [tuple(m) for m in prof.members] == [tuple(m) for m in col.members] and prof.length == col.length
            assert prof.n_locs == sum(by[k].clusters[i].n_locs for k, i in col.members)
            meets = all(by[k].clusters[i].n_locs >= leak.params.profile_min_locs_per_cluster for k, i in col.members)
            assert bool(prof.meets_min_locs) == meets
            assert prof.unit_is_events is True and prof.period_nm == leak.period_nm
            truth_z = event_z_truth(res, col)
            assert prof.n_events == truth_z.size and np.asarray(prof.z_events_nm).size == prof.n_events
            worst_z = max(worst_z, float(np.abs(np.sort(np.asarray(prof.z_events_nm)) - truth_z).max()))
            assert prof.n_bootstrap == PROFILE_B and np.isfinite(prof.llr) and prof.llr >= -1e-9
            if col.length == 3 and meets:
                n3 += 1
                n_sig += prof.p_bootstrap <= 0.05
            n_checked += 1
        assert worst_z <= 1e-9, worst_z
        assert n3 >= 20 and n_sig / n3 >= 0.95, (n3, n_sig)
        frac = leak.fraction_profiles_bimodal
        return (f"{n_checked} profiles; z_events = mean z' over all localizations of each event ({worst_z:.1e} nm); {n_sig}/{n3} length-3 "
                f"columns (all clusters >= 100 locs) with p_bootstrap <= 0.05; fraction_profiles_bimodal {frac:.3f}; period {period:.2f} nm")

    def m1_parent_child():
        run = axon_leak("M1", 0)
        axon, res, cols, leak = run["axon"], run["res"], run["cols"], run["leak"]
        truth = label_clusters(axon, res)
        long_cols = [c for c in cols.columns if c.length >= 2]
        n = n_big = 0
        pis, llrs, ps = [], [], []
        for prof, col in zip(leak.profiles, long_cols):
            if col.length != 2:
                continue
            a, b = [truth[tuple(m)] for m in col.members]
            child = b if b.spurious else a if a.spurious else None
            parent = a if child is b else b
            if child is None or child.parent != (parent.ring, parent.index):
                continue
            n += 1
            n_big += prof.p_bootstrap > 0.05
            pis.append(float(prof.pi_hat))
            llrs.append(float(prof.llr))
            ps.append(float(prof.p_bootstrap))
            # What the mechanism guarantees (module docstring): a profile per column, weights summing to 1, the
            # child's component the minority (the parent's events alone are more than half of the column's),
            # p on the (b + 1) / (B + 1) grid. The specification's ">= 80 % with p > 0.05" is NOT asserted: the
            # one-Gaussian bootstrap rejects any non-Gaussian tail and the leak's single-localization events
            # sit beyond the boundary (measured 3 of 7 on seed 0; the finding is printed, rule (iv) stays None).
            assert prof.unit_is_events and abs(float(np.sum(prof.weights)) - 1.0) < 1e-9
            assert 0.0 <= prof.pi_hat <= 0.5, prof.pi_hat
            assert abs(prof.p_bootstrap * (PROFILE_B + 1) - round(prof.p_bootstrap * (PROFILE_B + 1))) < 1e-9
        assert n >= 1, n
        return (f"{n} parent+child columns of length 2 (the other children sit in longer columns): p_bootstrap > 0.05 in {n_big} "
                f"({100 * n_big / n:.0f} % on this seed; 71 % and 82 % on fresh seeds 11 / 12, pooled 22 of 32 = 69 %: the "
                f"specification's premise of >= 80 % holds in about two thirds of them, a one-Gaussian null rejects the leak's "
                f"one-sided tail in the rest); p quartiles {quartiles(ps)}, pi_hat quartiles {quartiles(pis)} (<= 0.5 asserted), "
                f"llr quartiles {quartiles(llrs)}")

    check("M5: one profile per column of length >= 2 in cols.columns order; unit_is_events, n_events and z_events_nm from the events "
          "(1e-9); >= 95 % of the length-3 columns with every cluster >= 100 locs have p_bootstrap <= 0.05", m5_length_3)
    check("M1: parent+child columns of length 2: profile per column, weights sum to 1, the child's component is the minority "
          "(pi_hat <= 0.5), p on the bootstrap grid; the fraction with p > 0.05 is PRINTED (the spec's >= 80 % does not hold)",
          m1_parent_child)


# ============================================================ 13. rules, determinism, API
def test_rules_determinism_api() -> None:
    print("\n13. RULES dict, determinism, seeds, analyze_columns_and_leak")

    def rules_dict():
        expected = {"i_shared", "ii_lpz", "iii_guard", "iv_profile", "v_size_ratio"}
        out = []
        for model, seed, guard in (("M1", 0, True), ("M5", 0, True), ("M1", 1, False)):
            leak = axon_leak(model, seed, guard=guard)["leak"]
            rules = leak.rules
            assert set(rules) == expected, set(rules)
            assert rules["iv_profile"] is None
            assert isinstance(rules["v_size_ratio"], bool)
            assert isinstance(rules["i_shared"], bool) and isinstance(rules["ii_lpz"], bool)
            assert isinstance(rules["iii_guard"], bool) if guard else rules["iii_guard"] is None
            assert leak.events_available is True and leak.lpz_per_localization is True
            assert leak.n_matched == sum(m.n_matched for m in axon_base(model, seed)["cols"].adjacent)
            assert leak.n_leak_explained == sum(rp.n_leak_explained for rp in leak.pairs)
            assert abs(leak.fraction_leak_explained - leak.n_leak_explained / leak.n_matched) < 1e-12
            assert leak.k2 is axon_base(model, seed)["cols"].k2 or len(leak.k2) == len(axon_base(model, seed)["cols"].k2)
            out.append(f"{model} seed {seed}: {rules}, leak_explained {leak.n_leak_explained}/{leak.n_matched}")
        return "; ".join(out)

    # The M1 seed-2 axon (K = 30, the smallest cached one; children, leak-explained pairs and columns of both
    # lengths) carries the determinism, other-seed and driver checks: the code path is the same as on M5.
    DET_AXON = ("M1", 2)

    def deterministic():
        # The cached run of section 5 (same call: leak_params(), the axon's arrays, guard=False for seed 2)
        # is the first of the two runs; one fresh run is the second.
        analyze_leak = require("analyze_leak")
        base = axon_leak(*DET_AXON)
        axon, res, cols = base["axon"], base["res"], base["cols"]
        kw = dict(params=leak_params(), lpz_nm=axon.lpz_nm, frame=axon.frame, lp_lateral_nm=axon.lp_lateral_nm,
                  n_frames=axon.total_frames, guard=False)
        t0 = time.perf_counter()
        d1 = json.dumps(digest(base["leak"]), sort_keys=True)
        d2 = json.dumps(digest(analyze_leak(res, cols, **kw)), sort_keys=True)
        assert d1 == d2, "two runs with the same seed differ"
        base["leak_g0"] = d1
        return f"two runs identical ({len(d1)} chars of JSON digest, timing fields dropped); {elapsed(t0)}"

    def other_seed():
        analyze_leak = require("analyze_leak")
        base = axon_leak(*DET_AXON)
        axon, res, cols = base["axon"], base["res"], base["cols"]
        kw = dict(params=leak_params(), lpz_nm=axon.lpz_nm, frame=axon.frame, lp_lateral_nm=axon.lp_lateral_nm,
                  n_frames=axon.total_frames, guard=False)
        a = base["leak"]
        b = analyze_leak(res, dataclasses.replace(cols, random_seed=1), **kw)
        for ra, rb in zip(a.pairs, b.pairs):
            assert ra.n_leak_explained == rb.n_leak_explained and ra.zeta_clean == rb.zeta_clean
            assert ra.spearman_matched_vs_lpz_rho == rb.spearman_matched_vs_lpz_rho
            # the cleaned null reruns the eclipse test with the RUN's seed (match.random_seed), not cols.random_seed
            assert ra.zeta_clean_null == rb.zeta_clean_null
            for pa, pb in zip(ra.pairs, rb.pairs):
                assert pa.llr_tail_child == pb.llr_tail_child and pa.shared_fraction == pb.shared_fraction and pa.F_parent == pb.F_parent
        for fa, fb in zip(a.profiles, b.profiles):
            assert fa.llr == fb.llr and fa.mu_nm == fb.mu_nm and fa.sigma_nm == fb.sigma_nm and np.array_equal(fa.weights, fb.weights)
            assert fa.random_seed != fb.random_seed
        pa_ = [rp.p_shared for rp in a.pairs] + [rp.spearman_matched_vs_lpz_p for rp in a.pairs] + [f.p_bootstrap for f in a.profiles]
        pb_ = [rp.p_shared for rp in b.pairs] + [rp.spearman_matched_vs_lpz_p for rp in b.pairs] + [f.p_bootstrap for f in b.profiles]
        n_diff = sum(1 for x, y in zip(pa_, pb_) if x != y)
        assert n_diff > 0, "a different random_seed changed no Monte Carlo p-value"
        return f"observed statistics identical; {n_diff} of {len(pa_)} Monte Carlo p-values differ between seeds 0 and 1"

    def driver_and_warnings():
        analyze_columns_and_leak, LeakDiagnostics = require("analyze_columns_and_leak", "LeakDiagnostics")
        base = axon_base(*DET_AXON)
        axon, res, cols = base["axon"], base["res"], base["cols"]
        out = analyze_columns_and_leak(res, PARAMS, n_null=N_NULL, leak_params=leak_params(), lpz_nm=axon.lpz_nm, frame=axon.frame,
                                       lp_lateral_nm=axon.lp_lateral_nm, n_frames=axon.total_frames, guard=False)
        assert type(out).__name__ == "AxonColumnsResult" and isinstance(out.leak, LeakDiagnostics)
        assert [(m.n_matched, m.zeta) for m in out.adjacent] == [(m.n_matched, m.zeta) for m in cols.adjacent]
        assert analyze_columns(res, PARAMS, n_null=N_NULL).leak is None
        if "leak_g0" in base:
            assert json.dumps(digest(out.leak), sort_keys=True) == base["leak_g0"], "analyze_columns_and_leak differs from analyze_leak"
        n_w = 0
        for key in RUNS:
            leak = RUNS[key].get("leak")
            if leak is None:
                continue
            items = [leak] + list(leak.pairs) + list(leak.profiles) + list(leak.guard.values())
            for it in items:
                assert all(isinstance(w, str) for w in it.warnings)
                n_w += len(it.warnings)
        assert out.leak.params.profile_n_bootstrap == PROFILE_B and out.leak.source_name == cols.source_name
        assert out.leak.tau0_nm == cols.tau0_nm and out.leak.seconds >= 0.0
        return f"analyze_columns_and_leak -> AxonColumnsResult with .leak LeakDiagnostics (same digest as analyze_leak); analyze_columns keeps leak None; {n_w} warnings, all str"

    check("rules dict: the five keys with bool/None as specified (iii_guard None without guard runs, iv_profile None); counts consistent",
          rules_dict)
    check("determinism: two analyze_leak runs with the same seed give identical LeakDiagnostics (JSON digest)", deterministic)
    check("a different cols.random_seed changes the Monte Carlo p-values but not the observed statistics", other_seed)
    check("analyze_columns_and_leak returns an AxonColumnsResult whose .leak is a LeakDiagnostics; analyze_columns unchanged (leak None); "
          "warnings are strings", driver_and_warnings)


# ============================================================ 14. integration
def test_integration() -> None:
    print("\n14. INTEGRATION: make_columns_axon(0) (tilted) through run_build_rings + analyze_columns_and_leak")
    st: Dict[str, Any] = {}

    def with_frame():
        analyze_columns_and_leak = require("analyze_columns_and_leak")
        axon = h3.make_columns_axon(0)
        t0 = time.perf_counter()
        res = h1.run_build_rings(axon)
        t1 = time.perf_counter()
        out = analyze_columns_and_leak(res, PARAMS, n_null=N_NULL, leak_params=leak_params(), lpz_nm=axon.lpz_nm, frame=axon.frame,
                                       lp_lateral_nm=axon.lp_lateral_nm, n_frames=60000, guard=True)
        leak = out.leak
        st["axon"], st["out"] = axon, out
        assert len(res.rings) == 3 and len(leak.pairs) == 2 and len(leak.k2) == 1
        assert len(leak.guard) == 4 and all(g.n_rings == 3 for g in leak.guard.values()), {g: r.n_rings for g, r in leak.guard.items()}
        assert all(isinstance(leak.rules[k], bool) for k in ("i_shared", "iii_guard", "v_size_ratio")) and leak.rules["iv_profile"] is None
        # Rule (ii) on this axon: make_columns_axon has ONE lpz value (35 nm) for every localization, so every
        # cluster's median lpz is 35 and the Spearman against it is undefined (a constant variable): None is
        # the module's documented extension of "None without lpz" (an undefined correlation cannot vote).
        if np.unique(axon.lpz_nm).size == 1:
            assert leak.rules["ii_lpz"] is None, leak.rules["ii_lpz"]
            assert all(math.isnan(rp.spearman_matched_vs_lpz_rho) for rp in leak.pairs)
        else:
            assert isinstance(leak.rules["ii_lpz"], bool)
        n3 = sum(1 for c in out.columns if c.length == 3)
        kept = min(len(r.clusters) for r in res.rings)
        # H3's own rule for this axon: >= 80 % of the clusters in columns of length 3.
        assert len(leak.profiles) == sum(1 for c in out.columns if c.length >= 2) and n3 >= 0.8 * kept, (n3, kept)
        assert all(f.unit_is_events for f in leak.profiles)
        return (f"build_rings {t1 - t0:.1f} s (tilt {res.frame.beta_deg:.2f} deg), analyze_columns_and_leak {time.perf_counter() - t1:.1f} s; "
                f"guard K {[list(g.K_per_ring) for g in leak.guard.values()]}, rules {leak.rules}, {len(leak.profiles)} profiles "
                f"({n3} of length 3, bimodal fraction {leak.fraction_profiles_bimodal:.2f}), leak_explained {leak.n_leak_explained}/{leak.n_matched}")

    def without_frame():
        analyze_columns_and_leak = require("analyze_columns_and_leak")
        axon = need(st, "axon")
        res = h1.run_build_rings(axon, frame=None)
        assert res.event_id is None
        out = analyze_columns_and_leak(res, PARAMS, n_null=N_NULL, leak_params=leak_params(), lpz_nm=axon.lpz_nm, guard=False)
        leak = out.leak
        assert leak.events_available is False and leak.rules["i_shared"] is None and leak.guard == {}
        for rp in leak.pairs:
            assert is_nan(rp.p_shared) and is_nan(rp.shared_fraction_paired_mean) and rp.n_shared_high is None
            assert all(p.shared_events is None and is_nan(p.shared_fraction) and p.shared_high is None for p in rp.pairs)
            assert all(p.n_events_parent is None and p.n_events_child is None for p in rp.pairs)
        assert all(f.unit_is_events is False and f.n_events is None for f in leak.profiles)
        assert any("event" in w.lower() for w in leak.warnings + [w for f in leak.profiles for w in f.warnings])
        return f"events None: shared quantities NaN/None, rules i_shared None, guard {{}} (guard=False), {len(leak.profiles)} profiles per localization with a warning"

    check("tilted columns axon: 3 rings, 4 guard runs of 3 rings, rules evaluated (ii None: constant lpz), profiles for every "
          "column of length >= 2", with_frame)
    check("without frame numbers: shared quantities NaN/None, i_shared None, profiles per localization, no crash", without_frame)


# ============================================================ 15. unrolled pcf under leak (finding)
def test_pcf_under_leak() -> None:
    print("\n15. UNROLLED PCF UNDER LEAK (finding): exclude_clusters removes the children, not the centroid attraction")

    def configurations_and_true_arcs():
        # Re-review of 2026-09-24: on independent rings with leak, analyze_unroll's exclude_clusters remedy does
        # not restore the pcf null even with EVERY truth-spurious cluster removed, because a real cluster that
        # absorbed its neighbour's leak (DBSCAN merged the leaked localizations into it, purity ~0.85) has its
        # centroid pulled toward that neighbour, which at h ~ 30 nm inflates g(0) (reviewer's twelve pairs:
        # g(0) 1.29 with every spurious cluster removed against 0.995 for the true generator arcs). Asserted
        # here: the bookkeeping (K per configuration) and the true-arc null (|g(0) - 1| <= 0.3, five sd of the
        # reviewer's 0.06); the three configurations, the true-arc pcf and the displacement toward the nearest
        # neighbouring-ring cluster are PRINTED for the report. The displacement is not asserted: on one axon
        # its standard error is 2-9 nm per separation bin.
        try:
            import tools.mps_unroll as mu
            from tools.mps_matching import ring_geometry
        except Exception as exc:  # pragma: no cover - reported as a failed check
            raise AssertionError(f"tools.mps_unroll not importable: {exc}")
        run = axon_leak("M1", 0)
        axon, res, cols, leak, truth = run["axon"], run["res"], run["cols"], run["leak"], run["truth"]
        t0 = time.perf_counter()
        flagged = [(int(p.child_ring), int(p.child_index)) for rp in leak.pairs for p in rp.pairs if p.leak_explained]
        spurious = [(int(k), int(i)) for (k, i), ct in truth.items() if ct.spurious]
        up = mu.UnrollParams(n_null=N_NULL)
        rows: Dict[str, List[Tuple[int, int, int, int, float, float]]] = {}
        k_ring: Dict[str, Dict[int, int]] = {}
        for label, excl in (("computed", None), ("minus leak_explained", flagged), ("minus truth-spurious", spurious)):
            u = mu.analyze_unroll(res, cols, params=up, exclude_clusters=excl)
            ri, usable = np.asarray(u.unrolled.ring_index), np.asarray(u.unrolled.usable, dtype=bool)
            k_ring[label] = {int(k): int(np.count_nonzero(usable & (ri == k))) for k in np.unique(ri)}
            rows[label] = [(int(c.ring_a), int(c.ring_b), int(c.K_a), int(c.K_b), float(c.g_at_zero), float(c.p_global))
                           for c in u.pcf_adjacent]
            if excl is not None:
                removed: Dict[int, int] = {}
                for k, _ in excl:
                    removed[k] = removed.get(k, 0) + 1
                for k, n_k in k_ring["computed"].items():
                    assert k_ring[label][k] == n_k - removed.get(k, 0), (label, k, n_k, k_ring[label][k], removed)
        # The true generator arcs on the same reference curve: the pcf of the positions the rings were drawn at.
        u0 = mu.unroll(res, params=mu.UnrollParams(include_locs=False))
        ref = next(r for r in res.rings if r.index == u0.reference_ring)
        path = ring_geometry(ref, include_suspect=True).path
        s_true, _ = mu.project_on_path(path, axon.cluster_xy_nm, outward_sign=1.0)
        L = float(u0.length_nm)
        true_rows = []
        for a, b in ((0, 1), (1, 2)):
            sa, sb = s_true[axon.cluster_ring == a], s_true[axon.cluster_ring == b]
            p_bar = (L / sa.size + L / sb.size) / 2.0
            c = mu.cross_pcf_circular(sa, sb, L, p_bar_nm=p_bar, params=up, random_seed=cols.random_seed, ring_a=a, ring_b=b,
                                      reference_ring=u0.reference_ring)
            assert abs(c.g_at_zero - 1.0) <= 0.3, (a, b, c.g_at_zero)
            true_rows.append((a, b, int(sa.size), int(sb.size), float(c.g_at_zero), float(c.p_global)))
        # Displacement of every detected REAL cluster's arc from its true arc, signed toward the nearest true
        # cluster of a neighbouring ring, by separation (the mechanism).
        ri, ci, s_det = np.asarray(u0.ring_index), np.asarray(u0.cluster_index), np.asarray(u0.s_nm, dtype=float)
        sep_l, toward_l, pur_l = [], [], []
        for j in range(ri.size):
            ct = truth[(int(ri[j]), int(ci[j]))]
            if ct.spurious:
                continue
            s_t = s_true[ct.majority_cluster]
            d_arc = (s_det[j] - s_t + L / 2.0) % L - L / 2.0
            nb = [kk for kk in (int(ri[j]) - 1, int(ri[j]) + 1) if 0 <= kk <= 2]
            gaps = (s_true[np.isin(axon.cluster_ring, nb)] - s_t + L / 2.0) % L - L / 2.0
            i_n = int(np.argmin(np.abs(gaps)))
            sep_l.append(abs(gaps[i_n]))
            toward_l.append(float(np.sign(gaps[i_n]) * d_arc))
            pur_l.append(float(ct.purity))
        sep, toward, pur = np.asarray(sep_l), np.asarray(toward_l), np.asarray(pur_l)
        disp = []
        for lo, hi in ((0.0, 20.0), (20.0, 90.0), (90.0, float("inf"))):
            m = (sep >= lo) & (sep < hi)
            if m.sum() >= 2:
                disp.append(f"sep [{lo:.0f}, {hi:.0f}): n {int(m.sum())}, toward {toward[m].mean():+.1f} nm (se "
                            f"{toward[m].std(ddof=1) / math.sqrt(m.sum()):.1f}), purity median {np.median(pur[m]):.3f}")
        fmt = lambda r: "; ".join(f"{a}-{b} K {ka}/{kb} g(0) {g:.2f} p {p:.3f}" for a, b, ka, kb, g, p in r)  # noqa: E731
        return (f"{len(flagged)} flagged children, {len(spurious)} truth-spurious clusters; " +
                " | ".join(f"{label}: {fmt(r)}" for label, r in rows.items()) +
                f" | TRUE generator arcs: {fmt(true_rows)} (|g(0) - 1| <= 0.3 asserted) | displacement of detected real "
                f"clusters toward the nearest neighbouring-ring true cluster: " + "; ".join(disp) + f"; {elapsed(t0)}")

    check("M1 seed 0: pcf as computed / minus leak_explained / minus truth-spurious (K bookkeeping asserted) against the TRUE "
          "generator arcs (null: |g(0) - 1| <= 0.3); g(0), p_global and the centroid displacement PRINTED as the finding",
          configurations_and_true_arcs)


# ============================================================ 16. regression
def test_regression(with_regression: bool) -> None:
    print("\n16. Regression: validate_columns_h3 importable; H1-H3 harnesses in subprocesses (--with-regression)")

    def importable():
        assert hasattr(h3, "make_columns_axon") and hasattr(h3, "hard_core_arcs_nm") and hasattr(h1, "run_build_rings")
        assert h3.PASSED == 0 and h3.FAILED == 0, "importing validate_columns_h3 ran its checks"
        return "validate_columns_h3 and validate_rings_h1 imported without running a check"

    def run(script: str, expected: Tuple[int, int]) -> str:
        proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, script)], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=3600)
        tail = [ln for ln in proc.stdout.splitlines() if re.search(r"\d+ passed, \d+ failed", ln)]
        assert tail, proc.stdout[-2000:] + proc.stderr[-2000:]
        m = re.search(r"(\d+) passed, (\d+) failed", tail[-1])
        assert m is not None and (int(m.group(1)), int(m.group(2))) == expected, tail[-1]
        return tail[-1]

    check("validate_columns_h3.py importable (helpers reused; main() guarded)", importable)
    if with_regression:
        check("validate_rings_h1.py: 70 passed, 0 failed (unchanged)", lambda: run("validate_rings_h1.py", (70, 0)))
        check("validate_columns_h2.py: 49 passed, 0 failed (unchanged)", lambda: run("validate_columns_h2.py", (49, 0)))
        check("validate_columns_h3.py: 67 passed, 0 failed (unchanged)", lambda: run("validate_columns_h3.py", (67, 0)))
    else:
        print("      (H1/H2/H3 harnesses not re-run: pass --with-regression)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--with-regression", action="store_true",
                        help="also run validate_rings_h1.py (70), validate_columns_h2.py (49) and validate_columns_h3.py (67)")
    args = parser.parse_args()
    print("=" * 72)
    print("LEAK H4 CHECKS: tools/mps_leak.py (diagnostics (i)-(v), axial profiles, guard sensitivity)")
    print("=" * 72)
    sections: Tuple[Callable[[], None], ...] = (
        test_api, test_generator_truth, test_children, test_shared_events, test_tail_llr, test_detection,
        test_f_and_sizes, test_clean_excess, test_reassignment_and_spearman, test_guard, test_profile_fit,
        test_profile_test, test_column_profiles, test_rules_determinism_api, test_integration, test_pcf_under_leak)
    timings: List[Tuple[str, float]] = []
    for fn in sections:
        t0 = time.perf_counter()
        fn()
        timings.append((fn.__name__, time.perf_counter() - t0))
    t0 = time.perf_counter()
    test_regression(args.with_regression)
    timings.append(("test_regression", time.perf_counter() - t0))
    print("\n" + "=" * 72)
    print("timings: " + ", ".join(f"{n[5:]} {t:.0f} s" for n, t in timings) + f"; total {time.perf_counter() - T_START:.0f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
