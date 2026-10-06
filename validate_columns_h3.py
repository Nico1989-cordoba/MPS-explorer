# -*- coding: utf-8 -*-
"""
Checks for tools/mps_matching.py (H3: ring-to-ring matching within tau,
arc-shift null with reflection, eclipse curve over tau, joint null per
axon, columns, per-axon driver) against synthetic rings of KNOWN truth
(the research plan 03_plan.md S3.6, milestone H3;
DECISIONES.md D-02, D-03, D-04, D-11, D-22).

Synthetic rings: K clusters on an ellipse of semi-axes 1500 x 1000 nm
(perimeter 7.93 um) at arc positions drawn as a hard-core process
(minimum cyclic arc spacing 60 nm); for the realistic-scale diagnostic, a
circle of radius 1200 nm. Ring b given ring a: M1 fresh and
independent; M5(f, sigma_col) a fraction f of a's clusters copied at the
same arc position plus N(0, sigma_col) lateral jitter, the rest fresh
respecting the hard core; M6 clusters at the midpoints of a's gaps with
5 nm arc jitter; a rigid rotation by a constant arc shift; a rigid
translation. Every ring keeps its truth (arc positions, parent cluster).
Where a tolerance appears, the argument for it is next to the check.
The numbers quoted as "measured" come from this harness run on the
module after the D-24 amendments (seed 0), unless a sentence says they
are the first version's.

Points where this harness departs from the letter of the H3
specification, each with the number that forced it and, where the
orchestrator amended the pre-registration after the review of
2026-09-23, the decision that ratifies it (DECISIONES D-24a-e and the
fix decisions of that review, "items"):

* The fast generator builds ``RingGeometry`` directly (convex tour =
  polar angle about the mean) instead of going through
  ``ring_geometry_from_centroids``: ``reconstruct_perimeter(
  all_starts=True)`` costs 98 ms per ring at K = 40 (measured), and the
  loops below use ~5000 rings (8 min of polygons alone). Section 0
  checks the harness polygon against ``reconstruct_perimeter`` and
  section 2 against ``ring_geometry_from_centroids`` (order, contour,
  polygon arc and length to 1e-9; the curve arcs are the dataclass's
  own in both constructions), so every mass simulation runs on exactly
  the geometry the module would build.
* The hard core is drawn by sequential rejection (each new position is
  redrawn until it is >= 60 nm from every placed one): rejecting whole
  rings accepts (1 - K d / L)^(K - 1) = 8e-7 of the draws at K = 40.
* The null mean E* under the hard core is HIGHER than the Poisson
  formula 1 - exp(-2 tau / p_bar), not lower (measured 0.519 vs 0.454
  at tau = 60, K = 40): the hard core removes overlaps between the
  +/-tau intervals of neighbouring clusters, so a random point is
  covered more often. The +/-0.08 of the specification holds either way.
* The null's path (D-04 -> D-24a). The specification wrote the null as a
  shift along the polygon (``point_on_polygon``); on the chords the
  shifted clusters sit inside a convex membrane by the chord sagitta
  g^2 / (8 R) (up to 30 nm for a 400 nm gap on R = 670 nm), so the null
  matched fewer pairs than the observed vertices and zeta was biased in
  favour of columns: mean zeta +0.19 to +0.24 at K = 40 and +0.85 at
  K = 20 against +0.03 and -0.05 along the exact ellipse; the first
  version of this harness failed its own mean-zeta check on it. D-24a
  moves the clusters along the interpolating periodic cubic spline
  through the vertices (``splprep(s=0, per=True)``, tabulated with
  >= 50 samples per edge and >= 2000 in total); ``RingGeometry.arc_nm``
  and ``length_nm`` are the curve's, ``point_on_polygon`` is the alias
  of ``point_on_curve``, ``contour_nm`` keeps the polygon. Section 2
  checks the module's table against a first-principles CubicSpline
  (another construction of the same interpolant; 1.5e-3 nm), that the
  vertices are exact rows of it, that ``arc_shift`` stays on it
  (< 1e-3 nm) and that on M1 rings it is within 4 nm of the true
  ellipse (chord midpoints 119 nm off); section 4 runs the FPR at K = 40
  (specification: mean zeta +0.03, the check that failed, band kept)
  and at K = 20 (added by D-24a, R = 300: rates in [0.02, 0.09] and mean
  zeta within 0.15; measured +0.08 against +0.02 along the ellipse),
  both against the exact-ellipse diagnostic on the same rings, and
  prints the realistic-scale case (circle R = 1200 nm, K = 60, tau = 80:
  -0.075 vs -0.066 along the circle).
* Pending design point of D-24a (re-review of 2026-09-23): the FPR
  loops above put every cluster exactly on the membrane, and the
  interpolating spline through clusters that SCATTER about it (15 nm
  lateral jitter, a cluster's precision) is not the membrane: it
  overshoots between the vertices (median 50-60 nm) and its null is
  biased in the hypothesis direction on sparse rings. Section 4 prints
  the jittered-M1 diagnostic the re-review asked for: the module
  against a reference that shifts each cluster along the exact ellipse
  carrying its own jitter vector, same rings and the SAME recorded
  draws (paired). Reviewer's figures (R = 600): K = 40 / 15 nm +0.125
  sd, K = 20 / 15 nm +0.219 (module mean zeta +0.229, outside the
  +/-0.15 band), K = 20 / 8 nm +0.097; realistic scale (circle, K = 60,
  tau 80, 15 nm) -0.088, conservative. Printed, not asserted: a null
  along a smoothed membrane needs a pre-registered smoothing scale,
  the orchestrator's call; the H3 report must say that the null's
  validity was verified on membrane-exact rings only.
* The assignment rule (D-24b). The specification described the
  dummies-at-tau matrix as "maximum cardinality within tau, then
  minimum total distance", which it is not (a = (0,0),(55,0),(110,0)
  against b = (-55,0),(0,0),(55,0) at tau 60 gave 2 pairs at 0 nm, not
  3 at 55 nm). D-24b sets the dummy cost to M = tau (K_a + K_b + 1),
  which makes the semantics the rule; section 1 checks both
  counterexamples, 30 random 6 x 6 rings against a brute-force optimum,
  and the monotonicity in tau that follows; section 8 asserts it
  replicate by replicate on the curve's null matrix.
* SMALL cannot carry the power check (z_A > 3) of the integration: its
  clusters form a regular 24-gon (spacing 178 nm) and at tau_0 = 80.6
  nm an arc shift lands within tau_0 of the lattice 90 % of the time
  (z_A = 1.7 measured with every cluster paired). SMALL keeps the
  pipeline and bookkeeping checks; the power check runs on a second
  synthetic axon whose three rings share hard-core (irregular) cluster
  positions (``make_columns_axon``).
* Power floors (item 7): every power floor is "measured - 2 MC sd" at
  this seed (Wilson lower bound when the measurement is 100 %), with
  the measured value printed: M5 f = 0.5 at R = 400 (MC sd 1.6 %):
  88.2 % -> 84.5 % (the specification's 95 % is not reached by the
  model: half the clusters are fresh and compete for the parents the
  copies left); M5 f = 1: 100 % of 200 -> 98.0 %; M6 tau = 40: zeta < 0
  in 100 % -> 98.0 %, p_deficit <= 0.05 in 94.0 % of 200 -> 90.5 %.
* M5 excess: E_dir - E* is compared with f (1 - E*) within 0.10, not
  0.08 (item 7, ratified): the fresh clusters compete for the parents
  the copies did not take, so the excess is below the formula (measured
  deviation 0.079 at f = 0.5, 0.041 at f = 0.2, 0.000 at f = 1).
* The FPR bands [0.02, 0.09] with the Wilson argument (item 7): the 95 %
  score interval of a fraction of 0.05 is [0.034, 0.073] for R = 500,
  [0.030, 0.083] for R = 300 and [0.027, 0.089] for R = 200; the band
  holds each with a margin of about one MC sd. The "discreteness" the
  first version invoked could only lower a rate -- and it does (re-review
  of 2026-09-23): n_matched takes ~10 distinct values at K = 20, so the
  EXACT exchangeable rejection rate of the Phipson-Smyth p at B = 199
  (the fraction of the 200 exchangeable values whose p would be <= 0.05)
  is 0.029 at K = 20 and 0.033 at K = 40, not 0.05, and the reference
  for a band's lower edge is that rate's Wilson interval, [0.015, 0.054]
  for 0.029 at R = 300: the lower edge 0.02 rejected a correct null at 2
  of 6 fresh seeds. The K = 20 loop and the joint null (R = 300; the
  joint T is a sum of two counts, exact two-sided rate 0.034 measured
  on 100 axons, Wilson [0.019, 0.061]) use [0.01, 0.09]; each check
  computes and prints its exact exchangeable rate next to the measured
  fraction. The K = 40 loop keeps [0.02, 0.09] as the decision wrote it
  (exact rate 0.030-0.033, Wilson [0.018, 0.049] at R = 500: the lower
  edge is marginal there too, reported as pending); the mean-zeta
  check (+/-0.15) carries the bias detection in every loop.
* M6 at tau = 90 nm: the deficit weakens (median zeta -1.3 against -3.3
  at 40 nm) but E_dir is 0.60, not "~ 1": half the mean spacing is 99
  nm, so tau = 90 still misses the midpoints of gaps longer than 180 nm
  (at tau = 150 the sign flips to an excess, median zeta +1.2). The
  check asks for a weaker deficit and prints E_dir and E*.
* Rotation by a constant arc shift of 30 nm: |delta_bar| is ~3 nm, not
  ~30 (measured median 3.3, max 8.9 over 20 rings): the pair
  displacements are tangent vectors that cancel around a closed contour.
  delta_bar is checked exactly against the mean over the true pairs and
  d_median against 30 nm (chord of a 30 nm arc: 29.9995). A rigid
  translation by (20, -22) nm is added: there delta_bar is the shift.
* Joint null seeds: ``np.random.SeedSequence(seed, spawn_key=("joint",))``
  raises ValueError in numpy 2.4.6 (spawn keys are integers), so the
  draws cannot be reproduced from the specification's text. The harness
  requires ``JointNull`` to record them: ``null_shift_nm`` and
  ``null_reflect`` of shape (B, n_rings) with column 0 equal to 0 and
  False (the first ring is never shifted), ``null_n_matched`` of shape
  (B, n_pairs) with row sums equal to ``T_null``; and the helper
  ``joint_shift_statistic(geoms, tau_nm, shift_nm, reflect) ->
  (T, n_matched per pair)`` (one entry per ring; a non-zero entry for
  the first ring raises ValueError), which must reproduce every T_null
  from the recorded draws and equal ``arc_shift`` + ``match_rings`` done
  by hand. D-24d adds the relative-shift exclusion (a ring after the
  first shifted one is redrawn while its draw re-creates the previous
  pair's observed phase: 1.1 % of the replicates re-created the phase
  of pair (1, 2) before it). Re-review of 2026-09-23: reflection is
  s -> L - s about each ring's own TOUR ORIGIN, so with both rings
  reflected the (mirrored) phase is re-created at U_k - U_{k-1} = 2 o
  + L_{k-1} (mod L_k), o the arc on ring k's curve of ring k-1's
  first vertex, and not at U_k = U_{k-1}; the first rule tested the
  latter for both cases, which is right only when the tour origins
  coincide (this harness's copies rings share their first vertex) and
  otherwise left 8-11 of 1000 reflect/reflect replicates carrying the
  observed count of one pair into T_null. The module now measures the
  window against 2 o + sigma L_{k-1} in the reflected case (sigma the
  relative sense of the two tours), U_k - sigma U_{k-1} in the
  unreflected one, and records o and sigma (``origin_offset_nm``,
  ``tour_orientation``). Section 9 recomputes o and sigma from first
  principles (nearest point of ring k-1's first vertex on ring k's
  table; shoelace signs), checks that no recorded draw violates the
  rule, that the draws still reproduce every T_null, and adds the
  reviewer's case: a copies axon whose third ring lacks the second
  ring's origin cluster (o one spacing off; reconstruct_perimeter
  tour), asserting that no reflect/reflect replicate re-creates the
  pair's count (>= 85 % of it; 6 of 483 did at seed 1 before the fix)
  and the same with the third ring's tour run the other way round
  (sigma = -1, where the re-creation moves to U_k + U_{k-1}).
* Curve consistency: ``eclipse_curve`` must give, at every tau, the same
  E*/sd* as ``eclipse_test`` with the same seed (the same B shifted
  configurations serve every tau, drawn as eclipse_test draws them), so
  the curve at tau_0 reproduces the primary test of the report.
* ``p_global`` (D-24c): the specification scored each replicate against
  the mean/sd of all B replicates, itself included, calling the O(1/B)
  difference conservative; it is liberal (measured p_global <= 0.05 in
  18 % of exchangeable Gaussian curves at B = 19 against 5.0 %
  leave-one-out; 11.0 % / 5.8 % at B = 19 / 99 in the review's probe).
  The module scores each of the B + 1 curves against the other B in
  closed form; section 8 recomputes t_max_obs, every t_r and p_global
  by brute force, checks the closed form against the explicit
  leave-one-out on a small case (1e-9) and asks the FPR in [0.02, 0.09].
* ``build_columns`` refuses inconsistent input instead of dropping
  clusters (item 6: a match naming a ring outside ``ring_indices``, a
  matched cluster that is not a usable cluster of its ring, a ring
  nothing describes) and checks at the end that every registered
  cluster is placed once; section 10 checks the ValueErrors.
* ``include_suspect`` on a geometry (item 5): ``RingGeometry`` carries
  the H2 suspect marks (``suspect``) so that ``include_suspect=False``
  excludes them, and the stamp of a result built from geometries is
  the flag the masks imply (True only when every cluster entered), with
  a warning when the argument disagreed; section 3 checks both.
* ``build_columns`` receives no labels (``RingPairMatch`` carries cluster
  indices only), so ``Column.labels`` is checked against the DBSCAN
  labels in the integration test and only for length elsewhere.
* Budget (measured, this machine): arc geometry 41-55 s, FPR ~130 s
  (K = 40, K = 20, the realistic-scale circle and the jittered-M1
  diagnostic, whose rings go through ``reconstruct_perimeter`` as real
  rings do: 74 ms per ring at K = 40), M5 19-29 s, M6 8-12 s, curve
  31-46 s, joint ~25 s, integration 9-15 s: ~5 min in total (315 s
  measured); every loop prints its time.

Run:  venv\\Scripts\\python.exe validate_columns_h3.py                     (from the repo root)
      venv\\Scripts\\python.exe validate_columns_h3.py --with-regression   (+ H1 and H2 harnesses, ~15 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import validate_rings_h1 as h1  # noqa: E402  builds SMALL, imports tools.mps_columns
from tools.mps_columns import load_columns_params  # noqa: E402
from tools.mps_geometry import reconstruct_perimeter  # noqa: E402

PARAMS_YAML = os.path.join(REPO_ROOT, "config", "columns_params.yaml")

PASSED = 0
FAILED = 0
T_START = time.perf_counter()


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
    """The H3 names of ``tools.mps_matching``, or a RuntimeError naming
    what is missing, so that a check fails with the reason and never
    with a NameError further down (the module does not exist until the
    implementation lands; the harness is written first)."""
    try:
        import tools.mps_matching as mm
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"tools.mps_matching not importable ({type(exc).__name__}: {exc}); H3 not implemented")
    missing = [n for n in names if not hasattr(mm, n)]
    if missing:
        raise RuntimeError(f"tools.mps_matching has no {', '.join(missing)} (H3 not implemented)")
    got = tuple(getattr(mm, n) for n in names)
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


# ============================================================ synthetic truth
ELLIPSE_A_NM = 1500.0
ELLIPSE_B_NM = 1000.0
HARD_CORE_NM = 60.0
K_RING = 40

# Arc-length table of the ellipse: 40000 chords, so the position at an
# arc s is exact to < 1e-3 nm (the chord-arc gap of a 0.2 nm chord).
_ELL_T = np.linspace(0.0, 2.0 * np.pi, 40001)
_ELL_XY = np.column_stack([ELLIPSE_A_NM * np.cos(_ELL_T), ELLIPSE_B_NM * np.sin(_ELL_T)])
_ELL_CUM = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(_ELL_XY, axis=0).T))])
L_ELLIPSE_NM = float(_ELL_CUM[-1])


def ramanujan_perimeter_nm(a: float, b: float) -> float:
    return float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def ellipse_point_nm(s_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """(n, 2) point of the ellipse at arc position s (taken modulo L)."""
    t = np.interp(np.mod(np.asarray(s_nm, dtype=float), L_ELLIPSE_NM), _ELL_CUM, _ELL_T)
    return np.column_stack([ELLIPSE_A_NM * np.cos(t), ELLIPSE_B_NM * np.sin(t)])


def cyclic_distance_nm(d_nm: NDArray[np.float64], length_nm: float) -> NDArray[np.float64]:
    d = np.mod(np.asarray(d_nm, dtype=float), length_nm)
    return np.minimum(d, length_nm - d)


def hard_core_arcs_nm(
    rng: np.random.Generator, n: int, *, fixed: Sequence[float] = (),
    d_min_nm: float = HARD_CORE_NM, length_nm: float = L_ELLIPSE_NM,
) -> NDArray[np.float64]:
    """``n`` arc positions on the circle of length ``length_nm``, each
    drawn uniformly and redrawn until it lies >= ``d_min_nm`` (cyclic)
    from every position already placed, ``fixed`` included: a sequential
    rejection sampler of a hard-core process. Returned in placement
    order (not sorted)."""
    placed = np.asarray(list(fixed), dtype=float)
    out = np.empty(n)
    for k in range(n):
        for _attempt in range(200000):
            s = float(rng.uniform(0.0, length_nm))
            if placed.size == 0 or np.all(cyclic_distance_nm(placed - s, length_nm) >= d_min_nm):
                break
        else:
            raise RuntimeError(f"hard core jammed after placing {placed.size} of {n + len(fixed)}")
        placed = np.append(placed, s)
        out[k] = s
    return out


@dataclass
class SynthRing:
    """One synthetic ring with its truth."""

    index: int
    model: str
    s_nm: NDArray[np.float64]           # (K,) true arc position on the ellipse, ascending
    xy_nm: NDArray[np.float64]          # (K, 2) cluster position (ellipse point + jitter)
    parent: NDArray[np.int64]           # (K,) cluster of the parent ring it derives from, -1 fresh
    usable: NDArray[np.bool_]

    @property
    def K(self) -> int:
        return int(self.s_nm.size)


def _assemble(index: int, model: str, s: NDArray[np.float64], xy: NDArray[np.float64],
              parent: NDArray[np.int64]) -> SynthRing:
    o = np.argsort(s)
    return SynthRing(index=index, model=model, s_nm=s[o], xy_nm=xy[o], parent=parent[o],
                     usable=np.ones(s.size, dtype=bool))


def ring_m1(rng: np.random.Generator, index: int, K: int = K_RING, jitter_nm: float = 0.0) -> SynthRing:
    """An M1 ring: K hard-core arc positions on the ellipse. With
    ``jitter_nm`` > 0 each cluster is displaced by N(0, jitter) on each
    lateral axis (a cluster's precision; the jittered-M1 diagnostic of
    section 4); ``xy_nm - ellipse_point_nm(s_nm)`` recovers the jitter
    vector. No random draw is consumed when the jitter is 0, so every
    other loop's rings are unchanged."""
    s = hard_core_arcs_nm(rng, K)
    xy = ellipse_point_nm(s)
    if jitter_nm > 0.0:
        xy = xy + rng.normal(0.0, jitter_nm, xy.shape)
    return _assemble(index, "M1" if jitter_nm <= 0.0 else f"M1+jitter({jitter_nm:g})", s, xy,
                     np.full(K, -1, dtype=np.int64))


# realistic-scale membrane for the printed diagnostic of section 4: a circle
# of radius 1200 nm (perimeter 7.54 um) with K = 60 clusters (mean
# spacing 126 nm, a dense ring) and tau = 80 nm (~ tau_0).
CIRCLE_R_NM = 1200.0
L_CIRCLE_NM = 2.0 * np.pi * CIRCLE_R_NM


def circle_point_nm(s_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """(n, 2) point of the circle of radius CIRCLE_R_NM at arc position s."""
    t = np.mod(np.asarray(s_nm, dtype=float), L_CIRCLE_NM) / CIRCLE_R_NM
    return np.column_stack([CIRCLE_R_NM * np.cos(t), CIRCLE_R_NM * np.sin(t)])


def ring_m1_circle(rng: np.random.Generator, index: int, K: int) -> SynthRing:
    """An M1 ring on the realistic-scale circle (hard core 60 nm)."""
    s = hard_core_arcs_nm(rng, K, length_nm=L_CIRCLE_NM)
    return _assemble(index, "M1c", s, circle_point_nm(s), np.full(K, -1, dtype=np.int64))


def ring_m5(rng: np.random.Generator, index: int, parent: SynthRing, f: float,
            sigma_col_nm: float = 10.0) -> SynthRing:
    """Fraction ``f`` of the parent's clusters copied at the same arc
    position with N(0, sigma_col) jitter on each lateral axis; the rest
    fresh, respecting the hard core against the copied positions."""
    K = parent.K
    n_copy = int(round(f * K))
    parents = np.sort(rng.choice(K, n_copy, replace=False)) if n_copy else np.array([], dtype=int)
    s_copy = parent.s_nm[parents]
    xy_copy = parent.xy_nm[parents] + rng.normal(0.0, sigma_col_nm, (n_copy, 2))
    s_fresh = hard_core_arcs_nm(rng, K - n_copy, fixed=s_copy.tolist())
    s = np.concatenate([s_copy, s_fresh])
    xy = np.vstack([xy_copy, ellipse_point_nm(s_fresh)])
    par = np.concatenate([parents, np.full(K - n_copy, -1)]).astype(np.int64)
    return _assemble(index, f"M5(f={f})", s, xy, par)


def ring_m6(rng: np.random.Generator, index: int, parent: SynthRing,
            jitter_nm: float = 5.0) -> SynthRing:
    """Clusters at the midpoints of the parent's cyclic gaps, jittered
    along the arc by N(0, jitter)."""
    s_a = np.sort(parent.s_nm)
    gap = np.mod(np.roll(s_a, -1) - s_a, L_ELLIPSE_NM)
    s = np.mod(s_a + gap / 2.0 + rng.normal(0.0, jitter_nm, s_a.size), L_ELLIPSE_NM)
    return _assemble(index, "M6", s, ellipse_point_nm(s), np.full(s.size, -1, dtype=np.int64))


def ring_rotated(index: int, parent: SynthRing, shift_nm: float) -> SynthRing:
    """The parent moved by a constant arc shift along the ellipse (no jitter)."""
    s = np.mod(parent.s_nm + shift_nm, L_ELLIPSE_NM)
    return _assemble(index, f"rot({shift_nm})", s, ellipse_point_nm(s),
                     np.arange(parent.K, dtype=np.int64))


def ring_translated(index: int, parent: SynthRing, vec_nm: Tuple[float, float]) -> SynthRing:
    """The parent moved rigidly by ``vec_nm`` in the lateral plane."""
    xy = parent.xy_nm + np.asarray(vec_nm, dtype=float)
    return _assemble(index, f"trans{vec_nm}", parent.s_nm.copy(), xy,
                     np.arange(parent.K, dtype=np.int64))


def ring_at_arcs(index: int, arcs_nm: Sequence[float]) -> SynthRing:
    """A ring with clusters at the given arc positions (hand-built cases)."""
    s = np.asarray(arcs_nm, dtype=float)
    return _assemble(index, "hand", s, ellipse_point_nm(s), np.full(s.size, -1, dtype=np.int64))


def make_columns_axon(seed: int, *, K: int = K_RING, d_min_nm: float = 80.0) -> h1.SyntheticAxon:
    """
    Three rings (z' = -190, 0, 190 nm) sharing the SAME irregular cluster
    positions: perfect columns on a hard-core ring, in the layout of
    ``validate_rings_h1.make_synthetic_axon`` (4 fluorophores x 5 frames
    per cluster, 8 nm spread, 8 nm lateral / 35 nm axial noise, tilt 2
    deg at azimuth 30 deg, 60000 frames).

    SMALL cannot serve for the power check: its clusters sit on a
    regular 24-gon (spacing 178 nm), so at tau_0 = 80.6 nm an arc shift
    lands within tau_0 of the lattice 90 % of the time and z_A stays
    near 1.7 whatever the columns (measured). Here the arc positions are
    a hard-core draw with minimum spacing ``d_min_nm`` (80 nm, so that
    DBSCAN at eps 25 nm never merges two clusters of 8 nm spread) on the
    1500 x 1000 nm ellipse: E* ~ 0.7 at tau_0 and z_A ~ 7 when every
    cluster has its partner.
    """
    rng = np.random.default_rng(seed)
    s = np.sort(hard_core_arcs_nm(rng, K, d_min_nm=d_min_nm))
    cx, cy = ellipse_point_nm(s).T
    ring_z = (-190.0, 0.0, 190.0)
    n_fluor, n_per, total_frames = 4, 5, 60000
    rows: List[Tuple[float, ...]] = []
    fl = 0
    for r, zr in enumerate(ring_z):
        for j in range(K):
            for _k in range(n_fluor):
                px = cx[j] + rng.normal(0.0, 8.0)
                py = cy[j] + rng.normal(0.0, 8.0)
                start = int(rng.integers(0, total_frames - n_per))
                for m in range(n_per):
                    rows.append((r, r * K + j, fl, start + m, px, py, zr))
                fl += 1
    table = np.array(rows, dtype=float)
    ring = table[:, 0].astype(np.int64)
    cluster = table[:, 1].astype(np.int64)
    fluor = table[:, 2].astype(np.int64)
    frame = table[:, 3].astype(np.int64)
    tx, ty, tz = table[:, 4], table[:, 5], table[:, 6]
    n = tx.size
    x = tx + rng.normal(0.0, 8.0, n)
    y = ty + rng.normal(0.0, 8.0, n)
    z = tz + rng.normal(0.0, 35.0, n)
    b, ph = np.radians(2.0), np.radians(30.0)
    u = np.array([np.sin(b) * np.cos(ph), np.sin(b) * np.sin(ph), np.cos(b)])
    rot = h1.rotation_z_to(u)
    lab = rot @ np.vstack([x, y, z])
    lab_true = rot @ np.vstack([tx, ty, tz])
    return h1.SyntheticAxon(
        name="columns", x_nm=lab[0], y_nm=lab[1], z_nm=lab[2],
        true_x_nm=lab_true[0], true_y_nm=lab_true[1], true_z_nm=lab_true[2],
        frame=frame, ring=ring, cluster=cluster, fluorophore=fluor,
        lp_lateral_nm=np.full(n, 8.0), lpz_nm=np.full(n, 35.0),
        axis=u, tilt_deg=2.0, semi_axes_nm=(ELLIPSE_A_NM, ELLIPSE_B_NM),
        ring_z_nm=ring_z, n_clusters_per_ring=K, n_fluor_per_cluster=n_fluor,
        n_frames_per_fluor=n_per, lateral_noise_nm=8.0, axial_noise_nm=35.0)


def convex_tour_polygon(
    xy_nm: NDArray[np.float64],
) -> Tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64], float]:
    """(order, contour, arc, length) of the polygon through the points in
    the order of their polar angle about the mean -- what
    ``reconstruct_perimeter`` returns for a convex ring (2-opt finds no
    improvement, the winning start is the first), computed here from
    first principles so the harness never depends on the module for the
    geometry of its own truth."""
    xy = np.asarray(xy_nm, dtype=float)
    c = xy.mean(axis=0)
    order = np.argsort(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))
    contour = xy[order]
    edges = np.hypot(*(np.roll(contour, -1, axis=0) - contour).T)
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    arc = np.empty(xy.shape[0])
    arc[order] = cum[:-1]
    return order.astype(np.intp), contour, arc, float(cum[-1])


def geometry_of(ring: SynthRing) -> Any:
    """The module's ``RingGeometry`` for a synthetic ring, built directly
    from the harness polygon (see the module docstring: 98 ms per
    ``reconstruct_perimeter`` at K = 40 would put the loops out of
    budget); the curve and every arc are the dataclass's own (D-24a)."""
    RingGeometry = require("RingGeometry")
    order, contour, _arc, _length = convex_tour_polygon(ring.xy_nm)
    return RingGeometry(
        index=ring.index, centroids_nm=ring.xy_nm.copy(), contour_nm=contour, order=order,
        usable=ring.usable.copy(), labels=np.arange(ring.K, dtype=np.int64))


# ============================================================ first principles
def polygon_edges(contour: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(edge lengths, cumulative arc at each vertex incl. the closing one)."""
    nxt = np.roll(contour, -1, axis=0)
    edges = np.hypot(*(nxt - contour).T)
    return edges, np.concatenate([[0.0], np.cumsum(edges)])


def nearest_on_polygon(
    p_nm: NDArray[np.float64], contour: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """For each point: distance to the closed polygon and the arc
    position of its nearest point on it (projection on every edge,
    clipped to the segment; the smallest distance wins)."""
    p = np.asarray(p_nm, dtype=float).reshape(-1, 2)
    a = contour
    b = np.roll(contour, -1, axis=0)
    ab = b - a
    edges, cum = polygon_edges(contour)
    ap = p[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nkd,kd->nk", ap, ab) / np.maximum(edges ** 2, 1e-300), 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.hypot(*(p[:, None, :] - proj).transpose(2, 0, 1))
    v = np.argmin(d, axis=1)
    rows = np.arange(p.shape[0])
    s = cum[v] + t[rows, v] * edges[v]
    return d[rows, v], np.mod(s, cum[-1])


def greedy_nn_pairs(ca: NDArray[np.float64], cb: NDArray[np.float64], tau: float) -> List[Tuple[int, int]]:
    """Greedy matching by increasing distance (the alternative the
    assignment is compared with in B3): a pair is taken when both
    clusters are still free and d <= tau."""
    D = np.hypot(ca[:, None, 0] - cb[None, :, 0], ca[:, None, 1] - cb[None, :, 1])
    free_a = np.ones(ca.shape[0], bool)
    free_b = np.ones(cb.shape[0], bool)
    out = []
    for flat in np.argsort(D, axis=None):
        i, j = divmod(int(flat), cb.shape[0])
        if D[i, j] > tau:
            break
        if free_a[i] and free_b[j]:
            free_a[i] = free_b[j] = False
            out.append((i, j))
    return out


def phipson_smyth(null: NDArray[np.int64], obs: int) -> Tuple[float, float, float]:
    B = null.size
    pe = (int(np.sum(null >= obs)) + 1) / (B + 1)
    pd = (int(np.sum(null <= obs)) + 1) / (B + 1)
    return pe, pd, min(1.0, 2.0 * min(pe, pd))


def n_pairs(match_rings: Any, ca: Any, cb: Any, tau: float, ua: Any = None, ub: Any = None) -> int:
    return int(np.asarray(match_rings(ca, cb, tau, usable_a=ua, usable_b=ub)[0]).size)


def membrane_path_null(
    match_rings: Any, ca: NDArray[np.float64], b: SynthRing, tau: float, n_null: int,
    rng: np.random.Generator, min_shift_nm: float,
    point_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] = ellipse_point_nm,
    length_nm: float = L_ELLIPSE_NM,
) -> NDArray[np.int64]:
    """The arc-shift null of the specification with the shift taken
    along the EXACT membrane (the harness knows it: the ellipse, or the
    circle of the realistic-scale diagnostic): the reference that separates
    a path bias from the matching."""
    out = np.empty(n_null, dtype=np.int64)
    for r in range(n_null):
        U = rng.uniform(min_shift_nm, length_nm - min_shift_nm)
        s = b.s_nm
        if rng.random() < 0.5:
            s = np.mod(length_nm - s, length_nm)
        out[r] = n_pairs(match_rings, ca, point_fn(np.mod(s + U, length_nm)), tau, None, b.usable)
    return out


ellipse_path_null = membrane_path_null


def carried_jitter_null(
    match_rings: Any, ca: NDArray[np.float64], b: SynthRing, tau: float,
    shift_nm: NDArray[np.float64], reflect: NDArray[np.bool_],
    point_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] = ellipse_point_nm,
    length_nm: float = L_ELLIPSE_NM,
) -> NDArray[np.int64]:
    """The reference null of the jittered-M1 diagnostic (re-review of
    2026-09-23): ring b's clusters shifted along the EXACT membrane by
    the module's own recorded draws (U, reflect), each cluster carrying
    its jitter vector ``xy_nm - point_fn(s_nm)`` with it. Under iid
    isotropic jitter the jitter vectors are exchangeable across arc
    positions, so this null keeps the clusters' scatter about the
    membrane and loses the phase only; the module's null on the same
    draws differs from it by the spline path alone (paired)."""
    jitter = b.xy_nm - point_fn(b.s_nm)
    out = np.empty(shift_nm.size, dtype=np.int64)
    for r in range(shift_nm.size):
        s = np.mod(length_nm - b.s_nm, length_nm) if reflect[r] else b.s_nm
        out[r] = n_pairs(match_rings, ca, point_fn(np.mod(s + shift_nm[r], length_nm)) + jitter, tau, None, b.usable)
    return out


def exact_exchangeable_rates(null: NDArray[np.int64], obs: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """The rejection rate of the Phipson-Smyth p at level ``alpha`` that
    a CORRECT null yields on this pair, given the discreteness of the
    count: each of the B + 1 exchangeable values (observed + null) is
    taken as the observed one in turn and scored against the other B;
    the fraction with p <= alpha is returned for p_excess, p_deficit and
    p_two_sided. It is below alpha whenever the count takes few
    distinct values (re-review of 2026-09-23: 0.029 at K = 20, B = 199),
    and it is the reference a measured FPR must be compared with."""
    allv = np.concatenate([[int(obs)], np.asarray(null, dtype=np.int64)])
    n = allv.size
    B = n - 1
    order = np.sort(allv)
    # count of the OTHER values >= v (resp. <= v): ranks in the sorted array minus v itself
    ge = n - np.searchsorted(order, allv, side="left") - 1
    le = np.searchsorted(order, allv, side="right") - 1
    pe = (ge + 1) / (B + 1)
    pd = (le + 1) / (B + 1)
    p2 = np.minimum(1.0, 2.0 * np.minimum(pe, pd))
    return float(np.mean(pe <= alpha)), float(np.mean(pd <= alpha)), float(np.mean(p2 <= alpha))


def signed_area_nm2(contour: NDArray[np.float64]) -> float:
    """Shoelace signed area (first principles; > 0 counter-clockwise)."""
    x, y = np.asarray(contour, dtype=float).T
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def wilson_interval(p: float, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson 95 % score interval for a binomial fraction ``p`` of ``n``."""
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def power_floor(measured: float, n: int) -> float:
    """The floor of a power check (fix decisions item 7): the measured
    fraction minus 2 Monte Carlo sd, sqrt(p (1 - p) / n); when every
    trial succeeded the sd estimate is 0, and the Wilson 95 % lower
    bound (n / (n + 3.84)) is used instead. Rounded down to 0.5 %."""
    if measured >= 1.0:
        lo = wilson_interval(1.0, n)[0]
    else:
        lo = measured - 2.0 * math.sqrt(measured * (1.0 - measured) / n)
    return math.floor(lo * 200.0) / 200.0


def smooth_path_polyline(contour: NDArray[np.float64], step_nm: float = 1.0) -> NDArray[np.float64]:
    """The periodic cubic spline through the polygon's vertices on the
    chord-length parameter, sampled every ``step_nm`` of parameter
    (first principles, scipy's CubicSpline; the module's table is
    sampled at 2 nm, so the two polylines differ by the sagitta of a
    2 nm sub-chord, < 1e-3 nm on these rings)."""
    from scipy.interpolate import CubicSpline
    closed = np.vstack([contour, contour[:1]])
    u = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    spline = CubicSpline(u, closed, bc_type="periodic", axis=0)
    n = int(np.ceil(u[-1] / step_nm))
    return np.asarray(spline(u[-1] * np.arange(n) / n), dtype=float)


def ellipse_deviation_nm(p_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance of each point to the ellipse, along the ray of its
    parametric angle (exact on the ellipse, a close upper bound near it)."""
    p = np.asarray(p_nm, dtype=float).reshape(-1, 2)
    t = np.arctan2(p[:, 1] / ELLIPSE_B_NM, p[:, 0] / ELLIPSE_A_NM)
    e = np.column_stack([ELLIPSE_A_NM * np.cos(t), ELLIPSE_B_NM * np.sin(t)])
    return np.hypot(*(p - e).T)


def frac_le(p: NDArray[np.float64], alpha: float = 0.05) -> float:
    return float(np.mean(np.asarray(p) <= alpha))


def assert_band(name: str, value: float, lo: float, hi: float) -> None:
    assert lo <= value <= hi, f"{name} = {value:.4f} outside [{lo}, {hi}]"


# ============================================================ 0. truth and API
def test_truth_and_api() -> None:
    print("\n0. SYNTHETIC TRUTH (ellipse 1500 x 1000 nm, hard core 60 nm, K = 40) and the H3 API")

    def api_present():
        require("RingGeometry", "ring_geometry", "ring_geometry_from_centroids", "point_on_polygon",
                "arc_shift", "BIG_COST", "match_rings", "RingPairMatch", "eclipse_test",
                "EclipseCurve", "eclipse_curve", "JointNull", "axon_joint_null",
                "joint_shift_statistic", "Column", "build_columns", "AxonColumnsResult",
                "analyze_columns", "SmoothPath", "smooth_closed_path", "point_on_path", "point_on_curve")
        big = require("BIG_COST")
        assert float(big) == 1e9, big
        import tools.mps_matching as mm
        with open(mm.__file__, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        top = [alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names]
        top += [(node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)]
        banned = [m for m in top if m in ("PyQt5", "PyQt6", "PySide2", "PySide6", "matplotlib", "qtpy")]
        assert not banned, banned
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        undocumented = [n for n, node in funcs.items() if ast.get_docstring(node) is None]
        assert not undocumented, undocumented
        head = "\n".join(open(mm.__file__, encoding="utf-8").read().splitlines()[:40])
        assert "@author: Nicolas (ngomez) + Claude" in head, "author header missing"
        return f"22 names importable; BIG_COST 1e9; no Qt/matplotlib import; {len(funcs)} functions documented"

    def generator_truth():
        rng = np.random.default_rng(0)
        expected = ramanujan_perimeter_nm(ELLIPSE_A_NM, ELLIPSE_B_NM)
        assert abs(L_ELLIPSE_NM - expected) / expected < 1e-4, (L_ELLIPSE_NM, expected)
        min_gap, lengths = np.inf, []
        for r in range(50):
            a = ring_m1(rng, 0)
            assert a.K == K_RING and np.all(np.diff(a.s_nm) > 0)
            gap = np.mod(np.roll(a.s_nm, -1) - a.s_nm, L_ELLIPSE_NM)
            min_gap = min(min_gap, float(gap.min()))
            assert np.allclose(a.xy_nm, ellipse_point_nm(a.s_nm))
            _, _, _, length = convex_tour_polygon(a.xy_nm)
            lengths.append(length)
            b = ring_m5(rng, 1, a, 0.5)
            copies = np.flatnonzero(b.parent >= 0)
            assert copies.size == 20 and b.K == K_RING
            d = np.hypot(*(b.xy_nm[copies] - a.xy_nm[b.parent[copies]]).T)
            assert np.all(d < 6 * 10.0), d.max()
            assert np.array_equal(b.s_nm[copies], a.s_nm[b.parent[copies]])
            gap_b = np.mod(np.roll(b.s_nm, -1) - b.s_nm, L_ELLIPSE_NM)
            assert gap_b.min() >= HARD_CORE_NM - 1e-9, gap_b.min()
            c = ring_m6(rng, 2, a)
            mids = np.mod(a.s_nm + gap / 2.0, L_ELLIPSE_NM)
            dm = cyclic_distance_nm(np.sort(mids)[:, None] - c.s_nm[None, :], L_ELLIPSE_NM).min(axis=1)
            assert np.all(dm < 30.0), dm.max()
        assert min_gap >= HARD_CORE_NM, min_gap
        # A 40-gon inscribed in the ellipse: each chord is shorter than its
        # arc by (gap / R)^2 / 24, up to 2 % for a 500 nm gap on R = 670 nm.
        lengths_arr = np.asarray(lengths)
        assert np.all(np.abs(lengths_arr - expected) / expected < 0.03), lengths_arr
        # Rotation and translation truths.
        a = ring_m1(rng, 0)
        rot = ring_rotated(1, a, 30.0)
        d_rot = np.hypot(*(rot.xy_nm - a.xy_nm[rot.parent]).T)
        assert np.all(np.abs(d_rot - 30.0) < 0.05), (d_rot.min(), d_rot.max())
        tr = ring_translated(1, a, (20.0, -22.0))
        assert np.allclose(tr.xy_nm - a.xy_nm[tr.parent], [20.0, -22.0])
        return (f"perimeter {L_ELLIPSE_NM:.1f} nm (Ramanujan {expected:.1f}); 50 rings: smallest cyclic "
                f"gap {min_gap:.1f} nm, polygon length {lengths_arr.mean():.0f} nm ({100 * (1 - lengths_arr.mean() / expected):.2f} % "
                f"below the ellipse); M5 copies within 6 sigma_col, hard core kept; M6 midpoints within 30 nm; "
                f"rotation step {d_rot.mean():.4f} nm")

    def polygon_equals_reconstruct_perimeter():
        # The fast generator's polygon must be the one the module builds
        # through reconstruct_perimeter(all_starts=True): on a convex ring
        # the polar order is the shortest tour and the first start wins.
        rng = np.random.default_rng(1)
        worst = 0.0
        for r in range(20):
            ring = ring_m5(rng, r, ring_m1(rng, r), 0.5) if r % 2 else ring_m1(rng, r)
            order, contour, _arc, length = convex_tour_polygon(ring.xy_nm)
            per = reconstruct_perimeter(ring.xy_nm, all_starts=True)
            assert np.array_equal(np.asarray(per.order), order), (r, per.order[:5], order[:5])
            assert np.array_equal(per.contour, contour)
            worst = max(worst, abs(per.perimeter_nm - length))
            assert per.self_intersections_after == 0 and per.n_2opt_improvements == 0
        assert worst < 1e-9, worst
        return f"20 rings (M1 and M5): identical order and contour, |length difference| <= {worst:.1e} nm"

    check("H3 API: 22 names importable from tools.mps_matching; BIG_COST == 1e9; no Qt/matplotlib; "
          "every function documented; author header", api_present)
    check("generator truth: perimeter vs Ramanujan, cyclic spacing >= 60 nm, M5 copies and hard core, "
          "M6 midpoints, rotation step 30 nm, translation vector", generator_truth)
    check("harness polygon == reconstruct_perimeter(all_starts=True) on 20 rings (order, contour, length 1e-9)",
          polygon_equals_reconstruct_perimeter)


# ============================================================ 1. match_rings
def test_match_rings() -> None:
    print("\n1. match_rings: one-to-one assignment truncated at tau (B3 trap)")
    tau = 60.0

    def assignment_trap():
        match_rings = require("match_rings")
        # Clusters 1000 nm apart along a line; the partner sits at 1.5 tau
        # (then 0.9 tau) on the same line, nothing else closer than 910 nm.
        ca = np.column_stack([1000.0 * np.arange(6), np.zeros(6)])
        for factor, expect in ((1.5, 0), (0.9, 6)):
            cb = ca + np.array([factor * tau, 0.0])
            ia, jb, d = match_rings(ca, cb, tau)
            ia, jb, d = np.asarray(ia), np.asarray(jb), np.asarray(d)
            assert ia.size == jb.size == d.size == expect, (factor, ia, jb)
            if expect:
                assert np.array_equal(ia, np.arange(6)) and np.array_equal(jb, np.arange(6))
                assert np.allclose(d, factor * tau, atol=1e-9)
        # Without the BIG truncation a pair at 1.5 tau costs 90 < 120 of two
        # dummies and would be taken: the 0 above is the whole point.
        return "pairs at 1.5 tau -> 0 pairs (cost 90 < 2 tau = 120 would pair them untruncated); at 0.9 tau -> 6 of 6"

    def greedy_loses():
        match_rings = require("match_rings")
        # a1 = (0, 0) near b1 = (-40, 0) and b2 = (20, 0); a2 = (50, 0) near b2 only
        # (30 nm; b1 at 90 > tau). Greedy takes the globally closest pair a1-b2
        # (20 nm) first and strands a2: 1 pair. The assignment pairs a1-b1 and
        # a2-b2 (40 + 30 = 70 < 20 + tau + tau = 140): 2 pairs.
        ca = np.array([[0.0, 0.0], [50.0, 0.0]])
        cb = np.array([[-40.0, 0.0], [20.0, 0.0]])
        greedy = greedy_nn_pairs(ca, cb, tau)
        assert greedy == [(0, 1)], greedy
        ia, jb, d = match_rings(ca, cb, tau)
        pairs = sorted(zip(np.asarray(ia).tolist(), np.asarray(jb).tolist()))
        assert pairs == [(0, 0), (1, 1)], pairs
        assert np.allclose(np.asarray(d), [40.0, 30.0])
        return f"greedy {greedy} (1 pair); assignment {pairs} (2 pairs), distances {np.asarray(d).tolist()}"

    def maximum_cardinality():
        # D-24b: dummies at M = tau (K_a + K_b + 1) give the maximum
        # cardinality among admissible pairs, then the minimum distance
        # sum. Before it (dummies at tau) both cases returned 2 pairs at
        # 0 or 1 nm, because two pairs worth 2 tau each beat three worth
        # 2 tau - 55 (reviewers' counterexamples, 2026-09-23).
        match_rings = require("match_rings")
        ca = np.array([[0.0, 0.0], [55.0, 0.0], [110.0, 0.0]])
        cb = np.array([[-55.0, 0.0], [0.0, 0.0], [55.0, 0.0]])
        ia, jb, d = (np.asarray(v) for v in match_rings(ca, cb, tau))
        pairs = sorted(zip(ia.tolist(), jb.tolist()))
        assert pairs == [(0, 0), (1, 1), (2, 2)], pairs
        assert np.allclose(d, 55.0), d
        ca2 = np.array([[0.0, 0.0], [60.5, 0.0], [121.0, 0.0]])
        cb2 = np.array([[59.5, 0.0], [120.0, 0.0], [180.5, 0.0]])
        ia2, jb2, d2 = (np.asarray(v) for v in match_rings(ca2, cb2, tau))
        pairs2 = sorted(zip(ia2.tolist(), jb2.tolist()))
        assert pairs2 == [(0, 0), (1, 1), (2, 2)], pairs2
        assert np.allclose(d2, 59.5), d2
        # brute force over every matching of admissible pairs on 30 random
        # pairs of small rings: the module's count is the maximum and its
        # distance sum the minimum among matchings of that size
        import itertools
        rng = np.random.default_rng(11)
        n_checked = 0
        for r in range(30):
            k = 6
            xa = np.column_stack([rng.uniform(0, 300, k), rng.uniform(0, 60, k)])
            xb = np.column_stack([rng.uniform(0, 300, k), rng.uniform(0, 60, k)])
            D = np.hypot(*(xa[:, None, :] - xb[None, :, :]).transpose(2, 0, 1))
            best_n, best_sum = 0, np.inf
            for perm in itertools.permutations(range(k)):
                for mask in range(1 << k):
                    n = 0
                    s = 0.0
                    ok = True
                    for i in range(k):
                        if mask >> i & 1:
                            if D[i, perm[i]] > tau:
                                ok = False
                                break
                            n += 1
                            s += D[i, perm[i]]
                    if ok and (n > best_n or (n == best_n and s < best_sum - 1e-9)):
                        best_n, best_sum = n, s
            ia, jb, d = (np.asarray(v) for v in match_rings(xa, xb, tau))
            assert ia.size == best_n and abs(float(d.sum()) - best_sum) < 1e-9, (r, ia.size, best_n, d.sum(), best_sum)
            n_checked += 1
        return (f"a = (0,0),(55,0),(110,0) vs b = (-55,0),(0,0),(55,0): {pairs} at 55 nm (dummies at tau gave 2 at 0 nm); "
                f"second case {pairs2} at 59.5 nm; {n_checked} random 6 x 6 rings: count and distance sum equal the brute-force "
                f"maximum-cardinality minimum-sum matching")

    def monotone_in_tau():
        # A consequence of D-24b: the admissible set grows with tau, so the
        # maximum matching cannot shrink. Checked on 100 M1 pairs over a
        # 5 nm grid of tau (exact assertion, not a diagnostic).
        match_rings = require("match_rings")
        rng = np.random.default_rng(12)
        grid = np.arange(10.0, 201.0, 5.0)
        worst = 0
        for r in range(100):
            a, b = ring_m1(rng, 0), ring_m1(rng, 1)
            counts = np.array([n_pairs(match_rings, a.xy_nm, b.xy_nm, float(t)) for t in grid])
            assert np.all(np.diff(counts) >= 0), (r, counts)
            worst = max(worst, int(counts.max()))
        return f"100 M1 pairs x {grid.size} tau: n_matched non-decreasing in tau in every pair (up to {worst} of 40 at 200 nm)"

    def distances_and_bookkeeping():
        match_rings = require("match_rings")
        rng = np.random.default_rng(2)
        a, b = ring_m1(rng, 0), ring_m5(rng, 1, ring_m1(rng, 0), 0.5)
        ia, jb, d = (np.asarray(v) for v in match_rings(a.xy_nm, b.xy_nm, tau))
        assert ia.dtype.kind in "iu" and jb.dtype.kind in "iu" and d.dtype.kind == "f"
        assert ia.size == jb.size == d.size > 0
        assert np.all(np.diff(ia) > 0), "not sorted by i_a / repeated a-cluster"
        assert np.unique(jb).size == jb.size, "a b-cluster used twice"
        D = np.hypot(*(a.xy_nm[ia] - b.xy_nm[jb]).T)
        assert np.allclose(d, D, atol=1e-9) and np.all(d <= tau + 1e-12)
        return f"{ia.size} pairs: d == Euclidean (1e-9), i_a strictly increasing, j_b unique, d <= tau"

    def symmetry():
        match_rings = require("match_rings")
        rng = np.random.default_rng(3)
        n_pairs_total = 0
        for r in range(10):
            a, b = ring_m1(rng, 0), ring_m1(rng, 1)
            ia, jb, d = (np.asarray(v) for v in match_rings(a.xy_nm, b.xy_nm, tau))
            jb2, ia2, d2 = (np.asarray(v) for v in match_rings(b.xy_nm, a.xy_nm, tau))
            fwd = sorted(zip(ia.tolist(), jb.tolist()))
            bwd = sorted(zip(ia2.tolist(), jb2.tolist()))
            assert fwd == bwd, (r, set(fwd) ^ set(bwd))
            assert abs(d.sum() - d2.sum()) < 1e-9
            n_pairs_total += ia.size
        return f"10 M1 pairs, {n_pairs_total} pairs in total: match_rings(b, a) == match_rings(a, b) transposed"

    def usable_masks():
        match_rings = require("match_rings")
        rng = np.random.default_rng(4)
        a, b = ring_m1(rng, 0), ring_m5(rng, 1, ring_m1(rng, 0), 1.0)
        ua = rng.random(a.K) < 0.7
        ub = rng.random(b.K) < 0.7
        ia, jb, d = (np.asarray(v) for v in match_rings(a.xy_nm, b.xy_nm, tau, usable_a=ua, usable_b=ub))
        assert np.all(ua[ia]) and np.all(ub[jb]), "an excluded cluster was paired"
        sub_a, sub_b = np.flatnonzero(ua), np.flatnonzero(ub)
        i2, j2, d2 = (np.asarray(v) for v in match_rings(a.xy_nm[sub_a], b.xy_nm[sub_b], tau))
        got = sorted(zip(ia.tolist(), jb.tolist()))
        exp = sorted(zip(sub_a[i2].tolist(), sub_b[j2].tolist()))
        assert got == exp, (got[:5], exp[:5])
        assert abs(d.sum() - d2.sum()) < 1e-9
        return (f"{int(ua.sum())} x {int(ub.sum())} usable of 40 x 40: {ia.size} pairs, none excluded, equal to "
                f"matching the subsets with indices mapped back")

    def empty_inputs():
        match_rings = require("match_rings")
        rng = np.random.default_rng(5)
        a, b = ring_m1(rng, 0), ring_m1(rng, 1)
        for ua, ub, what in ((np.zeros(a.K, bool), None, "no usable a"),
                             (None, np.zeros(b.K, bool), "no usable b")):
            out = match_rings(a.xy_nm, b.xy_nm, tau, usable_a=ua, usable_b=ub)
            assert len(out) == 3 and all(np.asarray(v).size == 0 for v in out), (what, out)
        out = match_rings(a.xy_nm, b.xy_nm + np.array([0.0, 5000.0]), tau)
        assert all(np.asarray(v).size == 0 for v in out), out
        out = match_rings(np.zeros((0, 2)), b.xy_nm, tau)
        assert all(np.asarray(v).size == 0 for v in out), out
        return "no usable cluster / nothing within tau / zero clusters -> three empty arrays"

    check("assignment trap: clusters pairwise at 1.5 tau (nothing closer) -> 0 pairs; at 0.9 tau -> all paired",
          assignment_trap)
    check("greedy nearest neighbour takes a1-b2 and strands a2 (1 pair); the assignment gives a1-b1, a2-b2 (2 pairs)",
          greedy_loses)
    check("maximum cardinality within tau, then minimum distance (D-24b): the reviewers' counterexamples give 3 pairs; "
          "30 random 6 x 6 rings equal the brute-force optimum", maximum_cardinality)
    check("n_matched is non-decreasing in tau, pair by pair (100 M1 pairs, tau 10..200 nm)", monotone_in_tau)
    check("matched distances equal the Euclidean distances (1e-9); sorted by i_a; one-to-one; d <= tau",
          distances_and_bookkeeping)
    check("symmetry: swapping a and b gives the same pairs (10 M1 ring pairs, K = 40)", symmetry)
    check("usable masks: excluded clusters never paired; equals matching the usable subsets", usable_masks)
    check("empty and degenerate inputs -> three empty arrays", empty_inputs)


# ============================================================ 2. arc geometry
def test_arc_geometry() -> None:
    print("\n2. Arc geometry: ring_geometry_from_centroids, point_on_polygon, arc_shift")
    st: Dict[str, Any] = {}

    def from_centroids_equals_harness():
        fn = require("ring_geometry_from_centroids")
        rng = np.random.default_rng(6)
        worst = worst_curve = 0.0
        for r in range(20):
            ring = ring_m5(rng, r, ring_m1(rng, r), 0.5) if r % 2 else ring_m1(rng, r)
            g = fn(r, ring.xy_nm)
            order, contour, arc, length = convex_tour_polygon(ring.xy_nm)
            assert g.index == r
            assert np.array_equal(np.asarray(g.order), order), r
            assert np.allclose(g.contour_nm, contour, atol=1e-9)
            assert np.allclose(g.centroids_nm, ring.xy_nm, atol=1e-9)
            # the polygon's arcs and length are the harness's (1e-9); the
            # curve's arcs are the cumulative length of the module's table
            # at the vertex rows and its length the table's closed length
            worst = max(worst, float(np.abs(np.asarray(g.polygon_arc_nm) - arc).max()), abs(g.polygon_length_nm - length))
            rows = np.asarray(g.path.vertex_row)
            worst_curve = max(worst_curve, float(np.abs(np.asarray(g.arc_nm)[order] - np.asarray(g.path.cum_nm)[rows]).max()),
                              abs(g.length_nm - float(g.path.length_nm)))
            assert g.length_nm >= length, "the curve through the vertices is at least as long as the chords"
            assert np.asarray(g.usable).dtype == bool and np.all(g.usable) and g.usable.shape == (ring.K,)
            assert np.asarray(g.labels).shape == (ring.K,) and np.asarray(g.labels).dtype.kind in "iu"
            if r == 0:
                st["g0"], st["ring0"] = g, ring
        assert worst < 1e-9, worst
        assert worst_curve < 1e-9, worst_curve
        g = fn(3, st["ring0"].xy_nm, labels=np.arange(100, 140), usable=np.arange(40) % 3 != 0)
        assert np.array_equal(g.labels, np.arange(100, 140)) and int(g.usable.sum()) == 26
        # the same geometry built directly (as the loops do) is the same object field by field
        gd = geometry_of(st["ring0"])
        g0 = st["g0"]
        assert np.array_equal(gd.order, g0.order) and np.array_equal(gd.arc_nm, g0.arc_nm) and gd.length_nm == g0.length_nm
        assert np.array_equal(np.asarray(gd.path.points_nm), np.asarray(g0.path.points_nm))
        return (f"20 rings: order/contour/centroids identical, polygon arc and length within {worst:.1e} nm, curve arcs == table "
                f"({worst_curve:.1e} nm), curve >= polygon; labels/usable forwarded; direct construction identical")

    def arc_and_point_on_curve():
        pop, poc = require("point_on_polygon", "point_on_curve")
        assert pop is poc, "point_on_polygon must be the alias of point_on_curve"
        g = need(st, "g0")
        order, contour, arc, L = np.asarray(g.order), np.asarray(g.contour_nm), np.asarray(g.arc_nm), float(g.length_nm)
        assert arc[order[0]] == 0.0 and np.all(np.diff(arc[order]) > 0)
        edges, cum = polygon_edges(contour)
        assert np.all(np.diff(arc[order]) >= edges[:-1]), "a curve arc between two vertices is at least their chord"
        assert L >= cum[-1] and np.allclose(contour, np.asarray(g.centroids_nm)[order])
        ref = smooth_path_polyline(contour, 1.0)
        worst = worst_mid = 0.0
        sagitta = 0.0
        for v in range(contour.shape[0]):
            for s in (arc[order[v]], arc[order[v]] + L, arc[order[v]] - L):
                p = np.asarray(pop(contour, L, s), dtype=float).reshape(2)
                worst = max(worst, float(np.abs(p - contour[v]).max()))
            s_next = arc[order[(v + 1) % contour.shape[0]]] if v + 1 < contour.shape[0] else L
            mid = np.asarray(pop(contour, L, (arc[order[v]] + s_next) / 2.0), dtype=float).reshape(2)
            worst_mid = max(worst_mid, float(nearest_on_polygon(mid, ref)[0].max()))
            chord_mid = (contour[v] + contour[(v + 1) % contour.shape[0]]) / 2.0
            sagitta = max(sagitta, float(np.hypot(*(mid - chord_mid))))
        p_end = np.asarray(pop(contour, L, L), dtype=float).reshape(2)
        worst = max(worst, float(np.abs(p_end - contour[0]).max()))
        assert worst < 1e-9, worst
        assert worst_mid < 0.01, worst_mid
        assert sagitta > 1.0, sagitta
        # the polygon's length is refused: it would wrap the arc at the wrong place
        try:
            pop(contour, cum[-1], 0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("point_on_curve accepted the polygon's length")
        return (f"K = {contour.shape[0]}: arc[order[0]] == 0, arcs increase, each >= the chord; vertices, +/-L, s = L within "
                f"{worst:.1e} nm; mid-arc points within {worst_mid:.1e} nm of the first-principles spline and up to "
                f"{sagitta:.1f} nm off the chord midpoints; the polygon's length is refused")

    def smooth_path_truth():
        # The module's splprep curve against a first-principles periodic
        # spline (scipy CubicSpline, another construction of the same
        # interpolant, 1 nm sampling) and against the true ellipse, on the
        # 20 rings of the first check. Tolerance 0.01 nm: the two tables
        # differ by the sagitta of a 2 nm sub-chord on R >= 670 nm
        # (< 1e-3 nm), x 10. The deviation from the ellipse is asserted on
        # the M1 rings only, whose vertices ARE on the ellipse (an M5 ring's
        # copied clusters carry 10 nm of lateral jitter, so its curve
        # overshoots between them: printed).
        fn, pop, path_fn, n_min, n_edge = require("ring_geometry_from_centroids", "point_on_path", "smooth_closed_path",
                                                   "PATH_MIN_SAMPLES", "PATH_MIN_SAMPLES_PER_EDGE")
        rng = np.random.default_rng(6)
        worst_table = worst_len = worst_vertex = 0.0
        dev_m1, dev_m5, dev_chord, excess, n_rows = [], [], [], [], []
        for r in range(20):
            ring = ring_m5(rng, r, ring_m1(rng, r), 0.5) if r % 2 else ring_m1(rng, r)
            g = fn(r, ring.xy_nm)
            path = g.path
            contour = np.asarray(g.contour_nm)
            pts, rows = np.asarray(path.points_nm), np.asarray(path.vertex_row)
            assert pts.ndim == 2 and pts.shape[1] == 2 and rows.shape == (40,)
            assert pts.shape[0] >= max(int(n_min), 40 * int(n_edge)), pts.shape
            assert np.array_equal(pts[rows], contour), "a vertex is not a row of the table"
            edges, cum = polygon_edges(pts)
            assert np.allclose(np.asarray(path.edges_nm), edges) and np.allclose(np.asarray(path.cum_nm), cum)
            assert abs(float(path.length_nm) - cum[-1]) < 1e-9
            assert path.length_nm >= g.polygon_length_nm, "a curve through the vertices is at least as long as the chords"
            arc = np.asarray(g.arc_nm)
            assert np.allclose(arc[np.asarray(g.order)], cum[rows], atol=1e-9)
            assert arc[g.order[0]] == 0.0 and np.all(np.diff(arc[np.asarray(g.order)]) > 0)
            assert abs(g.length_nm - path.length_nm) < 1e-12 and g.path_length_nm == g.length_nm
            assert np.array_equal(np.asarray(g.path_arc_nm), arc)
            ref = smooth_path_polyline(contour, 1.0)
            d, _ = nearest_on_polygon(pts, ref)
            worst_table = max(worst_table, float(d.max()))
            worst_len = max(worst_len, abs(polygon_edges(ref)[1][-1] - path.length_nm))
            p_v = np.asarray(pop(g, arc), dtype=float)
            worst_vertex = max(worst_vertex, float(np.abs(p_v - np.asarray(g.centroids_nm)).max()))
            (dev_m5 if r % 2 else dev_m1).append(float(ellipse_deviation_nm(pts).max()))
            mids = (contour + np.roll(contour, -1, axis=0)) / 2.0
            dev_chord.append(float(ellipse_deviation_nm(mids).max()))
            excess.append(100.0 * (path.length_nm / L_ELLIPSE_NM - 1.0))
            n_rows.append(pts.shape[0])
        assert worst_table < 0.01, worst_table
        assert worst_len < 0.01, worst_len
        assert worst_vertex < 1e-6, worst_vertex
        # Measured on the M1 rings: path at most 4 nm off the ellipse at
        # K = 40 (chord midpoints 112 nm); 20 nm allows a 5 x worse fit.
        assert max(dev_m1) < 20.0, dev_m1
        assert max(dev_m1) < max(dev_chord) / 3.0, (max(dev_m1), max(dev_chord))
        assert max(abs(e) for e in excess) < 1.0, excess
        # Direct construction: smooth_closed_path on a contour with a repeated
        # vertex shares the row; fewer than 3 distinct vertices raise.
        tri = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 0.0], [50.0, 80.0]])
        sp = path_fn(tri)
        rows = np.asarray(sp.vertex_row)
        assert rows[1] == rows[2] and np.array_equal(np.asarray(sp.points_nm)[rows], tri)
        for bad in (np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])):
            try:
                path_fn(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("a degenerate contour was accepted")
        return (f"20 rings: {min(n_rows)}-{max(n_rows)} rows; table within {worst_table:.1e} nm of the CubicSpline reference, "
                f"length within {worst_len:.1e} nm, vertices exact ({worst_vertex:.1e} nm); M1 rings off the ellipse by "
                f"<= {max(dev_m1):.1f} nm (chord midpoints {max(dev_chord):.0f} nm; M5 rings with 10 nm jitter: "
                f"<= {max(dev_m5):.0f} nm, printed); curve length {np.mean(excess):+.2f} % of the ellipse; repeated vertex "
                f"shares a row; < 3 distinct vertices raise")

    def shift_stays_on_path():
        arc_shift = require("arc_shift")
        g = need(st, "g0")
        contour = np.asarray(g.contour_nm)
        pts = np.asarray(g.path.points_nm)
        ref = smooth_path_polyline(contour, 1.0)
        L, s0 = float(g.length_nm), np.asarray(g.arc_nm)
        worst_d, worst_s, worst_ref = 0.0, 0.0, 0.0
        n = 0
        for U in (0.0, 123.4, L / 3.0, L - 5.0, L + 77.0, 2.5 * L):
            for reflect in (False, True):
                p = np.asarray(arc_shift(g, U, reflect), dtype=float)
                assert p.shape == (contour.shape[0], 2), p.shape
                d, s = nearest_on_polygon(p, pts)                 # the module's own dense table
                expect = np.mod((np.mod(L - s0, L) if reflect else s0) + U, L)
                worst_d = max(worst_d, float(d.max()))
                worst_s = max(worst_s, float(cyclic_distance_nm(s - expect, L).max()))
                d_ref, _ = nearest_on_polygon(p, ref)              # the first-principles spline
                worst_ref = max(worst_ref, float(d_ref.max()))
                n += 1
        # D-24a: on the curve = within 1e-3 nm of its dense polyline (the
        # module's own table gives 1e-13; the 1 nm reference 1e-3 x 10).
        assert worst_d < 1e-3, worst_d
        assert worst_s < 1e-6, worst_s
        assert worst_ref < 0.01, worst_ref
        # The shifted clusters are NOT on the polygon's chords: on a convex
        # ring the curve lies outside them by up to the chord sagitta.
        p = np.asarray(arc_shift(g, 123.4, False), dtype=float)
        d_poly, _ = nearest_on_polygon(p, contour)
        assert d_poly.max() > 1.0, d_poly.max()
        return (f"{n} (U, reflect) cases: every cluster within {worst_d:.1e} nm of the curve at arc (s' + U) mod L within "
                f"{worst_s:.1e} nm; within {worst_ref:.1e} nm of the first-principles spline; up to {d_poly.max():.1f} nm "
                f"off the chords")

    def spacings_and_order():
        arc_shift = require("arc_shift")
        g = need(st, "g0")
        pts, L, s0, order = np.asarray(g.path.points_nm), float(g.length_nm), np.asarray(g.arc_nm), np.asarray(g.order)
        gaps0 = np.diff(np.concatenate([s0[order], [L]]))          # consecutive spacings along the curve

        def cyclic_equal(x: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
            return any(np.allclose(np.roll(x, k), y, atol=1e-6) for k in range(x.size))

        U = 1357.9
        for reflect in (False, True):
            p = np.asarray(arc_shift(g, U, reflect), dtype=float)
            _, s = nearest_on_polygon(p, pts)
            new_order = np.argsort(s)
            gaps = np.diff(np.concatenate([s[new_order], [s[new_order[0]] + L]]))
            if reflect:
                assert cyclic_equal(gaps, gaps0[::-1]), "reflected spacings are not the reversed cyclic sequence"
                # reversed cyclic order of the clusters along the tour
                seq = new_order[::-1]
                assert cyclic_equal(seq.astype(float), order.astype(float)), "reflection did not reverse the order"
            else:
                assert cyclic_equal(gaps, gaps0), "spacings are not a cyclic permutation"
                assert cyclic_equal(new_order.astype(float), order.astype(float)), "shift changed the cyclic order"
        return "shift: spacings a cyclic permutation, order kept; reflect + shift: spacings reversed, cyclic order reversed"

    def zero_shift_identity():
        arc_shift = require("arc_shift")
        g = need(st, "g0")
        p = np.asarray(arc_shift(g, 0.0, False), dtype=float)
        d = float(np.abs(p - np.asarray(g.centroids_nm)).max())
        assert d < 1e-9, d
        fn = require("ring_geometry_from_centroids")
        g2 = fn(9, st["ring0"].xy_nm, usable=np.arange(40) % 2 == 0)
        p2 = np.asarray(arc_shift(g2, 0.0, False), dtype=float)
        assert p2.shape == (40, 2) and np.abs(p2 - st["ring0"].xy_nm).max() < 1e-9
        return f"arc_shift(0, False) == centroids within {d:.1e} nm; every cluster returned with 20 of 40 usable"

    check("ring_geometry_from_centroids on 20 rings == harness polygon (order, contour, polygon arc and length 1e-9); "
          "arc_nm/length_nm are the curve's (D-24a); labels/usable defaults and forwarding; direct construction identical",
          from_centroids_equals_harness)
    check("arc_nm increases along the tour, arc_nm[order[0]] == 0; point_on_curve (alias point_on_polygon) returns every "
          "vertex at its curve arc (1e-9), modulo L; mid-arc points on the spline, off the chords; polygon length refused",
          arc_and_point_on_curve)
    check("smooth path (splprep, s = 0, per = True): >= 2000 rows and >= 50 per edge, vertices are rows of the table, arcs "
          "cumulative, table within 0.01 nm of a first-principles CubicSpline, M1 rings within 20 nm of the ellipse "
          "(chords > 3x farther), repeated vertex, degenerate contours raise", smooth_path_truth)
    check("arc_shift keeps every cluster on the curve (< 1e-3 nm of its dense polyline, D-24a; < 0.01 nm from the "
          "first-principles spline) at curve arc (s + U) mod L, reflect: (L - s + U) mod L; off the chords", shift_stays_on_path)
    check("arc_shift preserves the cyclic sequence of consecutive spacings; reflection reverses the cyclic order",
          spacings_and_order)
    check("arc_shift(0, False) returns the original centroids (1e-9); every cluster returned, usable or not",
          zero_shift_identity)


# ============================================================ 3. eclipse_test
def test_eclipse_test_bookkeeping() -> None:
    print("\n3. eclipse_test on one pair: bookkeeping, seeds, usable subset, degenerate cases")
    st: Dict[str, Any] = {}
    tau, B = 60.0, 199

    def bookkeeping():
        eclipse_test, match_rings = require("eclipse_test", "match_rings")
        rng = np.random.default_rng(7)
        a, b = ring_m1(rng, 0), ring_m5(rng, 1, ring_m1(rng, 0), 0.5)
        ga, gb = geometry_of(a), geometry_of(b)
        st["ga"], st["gb"], st["b"] = ga, gb, b
        m = eclipse_test(ga, gb, tau, n_null=B, random_seed=0)
        st["m"] = m
        ia, jb, d = (np.asarray(v) for v in match_rings(a.xy_nm, b.xy_nm, tau))
        assert np.array_equal(np.asarray(m.i_a), ia) and np.array_equal(np.asarray(m.j_b), jb)
        assert np.allclose(np.asarray(m.d_nm), d, atol=1e-12)
        assert m.n_matched == ia.size and m.K_a == 40 and m.K_b == 40
        assert m.ring_a == 0 and m.ring_b == 1 and m.tau_nm == tau
        assert abs(m.E_dir - ia.size / 40.0) < 1e-12 and abs(m.E_sym - 2.0 * ia.size / 80.0) < 1e-12
        null = np.asarray(m.null_n_matched)
        assert null.shape == (B,) and null.dtype.kind in "iu" and np.all(null >= 0) and np.all(null <= 40)
        assert abs(m.E_star - null.mean() / 40.0) < 1e-12
        sd0, sd1 = null.std(ddof=0), null.std(ddof=1)
        ok = [abs(m.sd_star - sd / 40.0) < 1e-12 and abs(m.zeta - (m.n_matched - null.mean()) / sd) < 1e-9
              for sd in (sd0, sd1)]
        assert any(ok), (m.sd_star, sd0 / 40, sd1 / 40, m.zeta)
        pe, pd, p2 = phipson_smyth(null, m.n_matched)
        assert abs(m.p_excess - pe) < 1e-12 and abs(m.p_deficit - pd) < 1e-12 and abs(m.p_two_sided - p2) < 1e-12
        dbar = (b.xy_nm[jb] - a.xy_nm[ia]).mean(axis=0)
        assert np.allclose(np.asarray(m.delta_bar_nm), dbar, atol=1e-9)
        assert abs(m.d_median_nm - np.median(d)) < 1e-9
        L, K_all = float(gb.path_length_nm), 40          # the null's L: the curve's closed length
        m_shift = 0.5 * L / K_all
        U = np.asarray(m.null_shift_nm)
        ref = np.asarray(m.null_reflect)
        assert U.shape == (B,) and np.all(U >= m_shift - 1e-9) and np.all(U <= L - m_shift + 1e-9), (U.min(), U.max())
        assert ref.shape == (B,) and ref.dtype == bool and 0 < int(ref.sum()) < B
        assert abs(m.min_shift_nm - m_shift) < 1e-9 and m.n_null == B and m.random_seed == 0
        assert bool(m.include_suspect) is True and isinstance(m.warnings, list)
        return (f"n {m.n_matched}/40, E_dir {m.E_dir:.3f}, E* {m.E_star:.3f}, zeta {m.zeta:.2f}, p_exc {m.p_excess:.3f}; "
                f"ddof {'1' if ok[1] else '0'}; U in [{U.min():.0f}, {U.max():.0f}] of [{m_shift:.0f}, {L - m_shift:.0f}], "
                f"{int(ref.sum())} reflections")

    def determinism_and_seeds():
        eclipse_test, from_centroids = require("eclipse_test", "ring_geometry_from_centroids")
        ga, gb, m = need(st, "ga"), need(st, "gb"), need(st, "m")
        again = eclipse_test(ga, gb, tau, n_null=B, random_seed=0)
        assert np.array_equal(np.asarray(again.null_n_matched), np.asarray(m.null_n_matched))
        assert np.array_equal(np.asarray(again.null_shift_nm), np.asarray(m.null_shift_nm))
        assert again.p_two_sided == m.p_two_sided and again.zeta == m.zeta
        other = eclipse_test(ga, gb, tau, n_null=B, random_seed=1)
        assert not np.array_equal(np.asarray(other.null_shift_nm), np.asarray(m.null_shift_nm))
        assert other.n_matched == m.n_matched and np.array_equal(np.asarray(other.i_a), np.asarray(m.i_a))
        b = need(st, "b")
        gb2 = from_centroids(6, b.xy_nm)
        ga2 = from_centroids(5, np.asarray(ga.centroids_nm))
        reindexed = eclipse_test(ga2, gb2, tau, n_null=B, random_seed=0)
        assert not np.array_equal(np.asarray(reindexed.null_shift_nm), np.asarray(m.null_shift_nm)), \
            "spawn_key must depend on the ring indices"
        assert reindexed.n_matched == m.n_matched
        return "same seed: identical; seed 1 or ring indices (5, 6): other shifts, same observed pairs"

    def usable_subset():
        eclipse_test, match_rings, from_centroids = require("eclipse_test", "match_rings", "ring_geometry_from_centroids")
        ga, b = need(st, "ga"), need(st, "b")
        rng = np.random.default_rng(8)
        ua = rng.random(40) < 0.75
        ub = rng.random(40) < 0.75
        ga2 = from_centroids(0, np.asarray(ga.centroids_nm), usable=ua)
        gb2 = from_centroids(1, b.xy_nm, usable=ub)
        m = eclipse_test(ga2, gb2, tau, n_null=B, random_seed=0)
        assert m.K_a == int(ua.sum()) and m.K_b == int(ub.sum())
        assert np.all(ua[np.asarray(m.i_a)]) and np.all(ub[np.asarray(m.j_b)])
        assert m.n_matched == n_pairs(match_rings, np.asarray(ga.centroids_nm), b.xy_nm, tau, ua, ub)
        assert abs(m.E_dir - m.n_matched / min(m.K_a, m.K_b)) < 1e-12
        assert np.all(np.asarray(m.null_n_matched) <= min(m.K_a, m.K_b))
        # p_bar of the null uses ALL clusters of b (polygon vertices) and the curve's length
        assert abs(m.min_shift_nm - 0.5 * float(gb2.path_length_nm) / 40) < 1e-9
        return f"K_a {m.K_a}, K_b {m.K_b}: pairs among usable only, null <= min K, min_shift from L / K_all"

    def include_suspect_on_geometry():
        eclipse_test, from_centroids, RingGeometry = require("eclipse_test", "ring_geometry_from_centroids", "RingGeometry")
        ga, b = need(st, "ga"), need(st, "b")
        base = from_centroids(1, b.xy_nm)
        assert np.asarray(base.suspect).dtype == bool and not np.any(base.suspect), "default suspect mask must be all False"
        marks = np.arange(40) % 5 == 0                                       # 8 suspect clusters, all usable
        gs = RingGeometry(index=1, centroids_nm=np.asarray(base.centroids_nm), contour_nm=np.asarray(base.contour_nm),
                          order=np.asarray(base.order), usable=np.ones(40, bool), labels=np.arange(40), suspect=marks)
        assert np.allclose(gs.arc_nm, base.arc_nm) and gs.length_nm == base.length_nm, "arcs are the dataclass's own"
        m_in = eclipse_test(ga, gs, tau, n_null=B, random_seed=0, include_suspect=True)
        m_out = eclipse_test(ga, gs, tau, n_null=B, random_seed=0, include_suspect=False)
        assert m_in.K_b == 40 and bool(m_in.include_suspect) is True and not m_in.warnings, m_in.warnings
        assert m_out.K_b == 32 and bool(m_out.include_suspect) is False, (m_out.K_b, m_out.include_suspect)
        assert not np.any(marks[np.asarray(m_out.j_b)]), "a suspect cluster was paired under include_suspect=False"
        assert np.array_equal(np.asarray(m_out.usable_b), ~marks) and np.all(gs.usable), "the input geometry was modified"
        same = eclipse_test(ga, dataclasses.replace(gs, usable=~marks), tau, n_null=B, random_seed=0, include_suspect=False)
        assert same.n_matched == m_out.n_matched and np.array_equal(np.asarray(same.null_n_matched), np.asarray(m_out.null_n_matched))
        # the other direction cannot be honoured: the mask decides, with a warning
        m_warn = eclipse_test(ga, dataclasses.replace(gs, usable=~marks), tau, n_null=B, random_seed=0, include_suspect=True)
        assert m_warn.K_b == 32 and any("include_suspect" in w and "8 suspect" in w for w in m_warn.warnings), m_warn.warnings
        assert bool(m_warn.include_suspect) is False, "the stamp must say what was applied"
        return (f"8 marked of 40: include_suspect True -> K_b 40, no warning; False -> K_b 32, none paired, stamp False, input "
                f"unchanged, equal to the explicit mask; True on a mask that excludes them -> K_b 32, stamp False, warning")

    def stamp_follows_the_masks():
        # Fix decisions item 5: with a RingGeometry the recorded flag is
        # the one the masks imply (True only when every cluster entered),
        # and a warning names the disagreement with the argument. Before
        # the fix the argument was stamped: False with every cluster used.
        eclipse_test, eclipse_curve, axon_joint_null, from_centroids = require(
            "eclipse_test", "eclipse_curve", "axon_joint_null", "ring_geometry_from_centroids")
        ga, gb, b = need(st, "ga"), need(st, "gb"), need(st, "b")
        m = eclipse_test(ga, gb, tau, n_null=19, random_seed=0, include_suspect=False)
        assert m.K_a == 40 and m.K_b == 40 and bool(m.include_suspect) is True, (m.K_b, m.include_suspect)
        assert any("include_suspect=False" in w and "every cluster entered" in w for w in m.warnings), m.warnings
        gb2 = from_centroids(1, b.xy_nm, usable=np.arange(40) % 4 != 0)         # 10 left out, no suspect mark
        m2 = eclipse_test(ga, gb2, tau, n_null=19, random_seed=0, include_suspect=True)
        assert m2.K_b == 30 and bool(m2.include_suspect) is False, (m2.K_b, m2.include_suspect)
        assert any("include_suspect=True" in w and "ring 1: 10" in w for w in m2.warnings), m2.warnings
        m3 = eclipse_test(ga, gb2, tau, n_null=19, random_seed=0, include_suspect=False)
        assert bool(m3.include_suspect) is False and not any("include_suspect" in w for w in m3.warnings), m3.warnings
        c = eclipse_curve(ga, gb2, [40.0, 60.0], n_null=19, random_seed=0, include_suspect=True)
        assert bool(c.include_suspect) is False and any("include_suspect=True" in w for w in c.warnings)
        j = axon_joint_null([ga, gb2, from_centroids(2, np.asarray(ga.centroids_nm))], tau, n_null=19, random_seed=0,
                            include_suspect=False)
        assert bool(j.include_suspect) is False and not any("include_suspect" in w for w in j.warnings)
        j2 = axon_joint_null([ga, gb, from_centroids(2, np.asarray(ga.centroids_nm))], tau, n_null=19, random_seed=0,
                             include_suspect=False)
        assert bool(j2.include_suspect) is True and any("include_suspect=False" in w for w in j2.warnings)
        return ("all-True masks + include_suspect=False -> stamp True with a warning; a mask leaving 10 out + True -> stamp "
                "False with a warning; agreeing argument -> no warning; eclipse_curve and axon_joint_null stamp the same way")

    def degenerate_pair():
        eclipse_test, from_centroids = require("eclipse_test", "ring_geometry_from_centroids")
        ang = np.radians([0.0, 120.0, 240.0])
        tri = np.column_stack([1000.0 * np.cos(ang), 1000.0 * np.sin(ang)])
        ga = from_centroids(0, tri)
        gb = from_centroids(1, tri + np.array([9000.0, 9000.0]))
        m = eclipse_test(ga, gb, tau, n_null=B, random_seed=0)
        null = np.asarray(m.null_n_matched)
        assert m.n_matched == 0 and np.all(null == 0) and m.E_dir == 0.0 and m.E_star == 0.0 and m.sd_star == 0.0
        assert is_nan(m.zeta), m.zeta
        assert any("sd" in w.lower() or "zeta" in w.lower() or "constant" in w.lower() for w in m.warnings), m.warnings
        assert m.p_excess == 1.0 and m.p_deficit == 1.0 and m.p_two_sided == 1.0
        assert np.asarray(m.delta_bar_nm).shape == (2,) and np.all(np.isnan(np.asarray(m.delta_bar_nm)))
        assert is_nan(m.d_median_nm) and np.asarray(m.i_a).size == 0
        return f"two triangles 12.7 um apart: n 0, null all 0, zeta NaN + warning ({m.warnings[0][:50]}...), p = 1, delta_bar/d_median NaN"

    def min_shift_too_large():
        eclipse_test = require("eclipse_test")
        ga, gb = need(st, "ga"), need(st, "gb")
        L = float(gb.path_length_nm)
        # m = fraction * L / K: 2 m >= L needs fraction >= K / 2 = 20.
        m = eclipse_test(ga, gb, tau, n_null=B, random_seed=0, min_shift_fraction=25.0)
        U = np.asarray(m.null_shift_nm)
        assert np.all(U >= 0.0) and np.all(U <= L), (U.min(), U.max())
        assert any("shift" in w.lower() for w in m.warnings), m.warnings
        assert U.min() < 0.5 * L / 40 or U.max() > L - 0.5 * L / 40, "U was still restricted"
        return f"min_shift_fraction 25 (2 m >= L): warning, U in [{U.min():.0f}, {U.max():.0f}] of [0, {L:.0f}]"

    check("RingPairMatch bookkeeping on an M5 pair: pairs == match_rings, K/E_dir/E_sym/E*/sd*/zeta/p/delta_bar/d_median "
          "recomputed (1e-12); U in [m, L - m], both reflections, n_null/seed/min_shift recorded", bookkeeping)
    check("determinism: same seed -> identical; other seed or other ring indices -> other shifts, same observed",
          determinism_and_seeds)
    check("usable subset: K counts usable clusters, pairs among usable only, null <= min K, p_bar from all clusters",
          usable_subset)
    check("include_suspect on a RingGeometry with suspect marks: False excludes them (stamp, masks, pairs), True on a mask "
          "that already excludes them warns and stamps False", include_suspect_on_geometry)
    check("the include_suspect stamp of a geometry result is what the masks imply, with a warning when the argument "
          "disagrees (eclipse_test, eclipse_curve, axon_joint_null)", stamp_follows_the_masks)
    check("degenerate pair (nothing ever within tau): n 0, null all 0, zeta NaN with a warning, p = 1, NaN delta_bar/d_median",
          degenerate_pair)
    check("min_shift_fraction with 2 m >= L: warning and U ~ U(0, L)", min_shift_too_large)


# ============================================================ 4. FPR under M1
def test_fpr_m1() -> None:
    print("\n4. Null validity: FPR under M1 (R = 500 pairs, K = 40, n_null = 199, tau = 60 nm)")
    st: Dict[str, Any] = {}
    R, B, tau = 500, 199, 60.0
    # The band [0.02, 0.09] of the specification, with the Wilson argument
    # (fix decisions item 7): the 95 % score interval of a fraction of
    # 0.05 is [0.034, 0.073] for R = 500 and [0.030, 0.083] for the R = 300
    # loops (K = 20 here, joint null in section 9); the band contains both
    # with a margin of one MC sd (0.010-0.013) on each side, so a rate
    # outside it is at least 3 sd from nominal on the wide side. The first
    # H3 version explained the band by "the discreteness of n_matched",
    # which can only make the Phipson-Smyth p conservative (its rate can
    # only fall) and never justified the upper end; the +0.19 sd mean-zeta
    # bias it measured there was the polygon path's (D-24a), and the
    # mean-zeta check below is kept at +/-0.15 as written. The lower
    # edge, though, IS where the discreteness bites (re-review of
    # 2026-09-23): the exact exchangeable rate of the discrete p at
    # B = 199 is 0.029 at K = 20 (Wilson [0.015, 0.054] at R = 300), so
    # the K = 20 loop uses LO_SPARSE = 0.01 (module docstring); every
    # loop prints its exact exchangeable rate next to the measured one.
    LO, HI = 0.02, 0.09
    LO_SPARSE = 0.01

    def fpr_loop(K: int, R_loop: int, seed: int, *, tau_nm: float = tau,
                 membrane: str = "ellipse") -> Dict[str, NDArray[np.float64]]:
        """R_loop M1 pairs of K clusters through eclipse_test and, on the
        same rings, the exact-membrane diagnostic (the ellipse, or the
        realistic-scale circle)."""
        eclipse_test, match_rings = require("eclipse_test", "match_rings")
        rng = np.random.default_rng(seed)
        if membrane == "ellipse":
            gen, point_fn, length = ring_m1, ellipse_point_nm, L_ELLIPSE_NM
        else:
            gen, point_fn, length = ring_m1_circle, circle_point_nm, L_CIRCLE_NM
        keys = ("pe", "pd", "p2", "zeta", "estar", "pbar", "alt_pe", "alt_pd", "alt_p2", "alt_zeta", "alt_estar",
                "ex_pe", "ex_pd", "ex_p2")
        acc: Dict[str, List[float]] = {k: [] for k in keys}
        t_mod = t_alt = 0.0
        for r in range(R_loop):
            a, b = gen(rng, 0, K), gen(rng, 1, K)
            ga, gb = geometry_of(a), geometry_of(b)
            t0 = time.perf_counter()
            m = eclipse_test(ga, gb, tau_nm, n_null=B, random_seed=r)
            t_mod += time.perf_counter() - t0
            acc["pe"].append(m.p_excess); acc["pd"].append(m.p_deficit); acc["p2"].append(m.p_two_sided)
            acc["zeta"].append(m.zeta); acc["estar"].append(m.E_star); acc["pbar"].append(float(gb.length_nm) / K)
            ex = exact_exchangeable_rates(np.asarray(m.null_n_matched), m.n_matched)
            acc["ex_pe"].append(ex[0]); acc["ex_pd"].append(ex[1]); acc["ex_p2"].append(ex[2])
            t0 = time.perf_counter()
            null = membrane_path_null(match_rings, a.xy_nm, b, tau_nm, B,
                                      np.random.default_rng(np.random.SeedSequence(7, spawn_key=(r,))),
                                      0.5 * float(gb.length_nm) / K, point_fn, length)
            t_alt += time.perf_counter() - t0
            pe, pd, p2 = phipson_smyth(null, m.n_matched)
            sd = null.std(ddof=1)
            acc["alt_pe"].append(pe); acc["alt_pd"].append(pd); acc["alt_p2"].append(p2)
            acc["alt_zeta"].append((m.n_matched - null.mean()) / sd if sd > 0 else np.nan)
            acc["alt_estar"].append(null.mean() / K)
        out = {k: np.asarray(acc[k], dtype=float) for k in keys}
        out["time"] = np.array([t_mod, t_alt])
        return out

    def loop():
        st.update(fpr_loop(K_RING, R, 41))
        t_mod, t_alt = st["time"]
        return f"{R} pairs: eclipse_test {t_mod:.1f} s ({1000 * t_mod / R:.0f} ms each), ellipse-path diagnostic {t_alt:.1f} s"

    def exact_note(key: str, R_loop: int) -> str:
        # the exact exchangeable rate of this loop's discrete p and its
        # Wilson 95 % interval at this R: the reference of the lower edge
        ex = float(need(st, key).mean())
        lo_w, hi_w = wilson_interval(ex, R_loop)
        return f"exact exchangeable rate {ex:.3f} (Wilson for {R_loop}: [{lo_w:.3f}, {hi_w:.3f}])"

    def excess_band():
        f = frac_le(need(st, "pe"))
        assert_band("p_excess <= 0.05", f, LO, HI)
        return f"fraction {f:.3f}; {exact_note('ex_pe', R)}"

    def deficit_band():
        f = frac_le(need(st, "pd"))
        assert_band("p_deficit <= 0.05", f, LO, HI)
        return f"fraction {f:.3f}; {exact_note('ex_pd', R)}"

    def two_sided_band():
        f = frac_le(need(st, "p2"))
        assert_band("p_two_sided <= 0.05", f, LO, HI)
        return f"fraction {f:.3f}; {exact_note('ex_p2', R)}"

    def zeta_centred():
        # sd of zeta ~ 1, so the mean over 500 has sd 0.045: 0.15 is 3.3 sd.
        # This is the check that failed with the polygon path (+0.219, the
        # chord bias); the band is the specification's, not widened.
        z = need(st, "zeta")
        mean = float(np.nanmean(z))
        assert abs(mean) <= 0.15, mean
        return f"mean zeta {mean:+.3f} (sd {np.nanstd(z):.2f}, se {np.nanstd(z) / math.sqrt(z.size):.3f}, {int(np.isnan(z).sum())} NaN)"

    def estar_vs_formula():
        # B3's Poisson formula; the hard core RAISES E* (fewer overlapping
        # +/-tau intervals): measured +0.05 at tau = 60 with the reference.
        es, pbar = need(st, "estar"), need(st, "pbar")
        formula = float(np.mean(1.0 - np.exp(-2.0 * tau / pbar)))
        mean = float(es.mean())
        assert abs(mean - formula) <= 0.08, (mean, formula)
        return f"mean E* {mean:.3f} vs 1 - exp(-2 tau / p_bar) = {formula:.3f} (p_bar {pbar.mean():.1f} nm)"

    def ellipse_path_diagnostic():
        # The same 500 rings with the null shifted along the exact ellipse
        # (harness path, module match_rings): the reference the module's
        # curve must agree with. The module's mean zeta minus this one is
        # the path bias; with the polygon path it was +0.19 (chord sagitta),
        # with the curve it must be within 0.10 (2.2 se of a difference of
        # two means of 500 zetas, which share the observed count).
        fe, fd, f2 = frac_le(need(st, "alt_pe")), frac_le(need(st, "alt_pd")), frac_le(need(st, "alt_p2"))
        z = float(np.nanmean(need(st, "alt_zeta")))
        for name, f in (("p_excess", fe), ("p_deficit", fd), ("p_two_sided", f2)):
            assert_band(f"ellipse path {name} <= 0.05", f, LO, HI)
        assert abs(z) <= 0.15, z
        dz = float(np.nanmean(need(st, "zeta"))) - z
        assert abs(dz) <= 0.10, dz
        d_estar = float(need(st, "estar").mean() - need(st, "alt_estar").mean())
        return (f"ellipse path: fractions {fe:.3f} / {fd:.3f} / {f2:.3f}, mean zeta {z:+.3f}, E* {need(st, 'alt_estar').mean():.3f}; "
                f"module minus ellipse path: mean zeta {dz:+.3f}, E* {d_estar:+.4f} (the path bias; +0.19 with the polygon path)")

    def sparse_rings_k20():
        # D-24a, added case: K = 20 on the same ellipse (400 nm mean gaps,
        # chord sagitta up to 190 nm for a 1 um gap on R = 670 nm), R = 300,
        # n_null 199. With the polygon path the null gave mean zeta +0.85
        # and p_excess <= 0.05 in 13.7 % of pairs (the ellipse path -0.05
        # and 2.0 %): a chord path cannot pass here. The upper edge is the
        # decision's (Wilson for 300 at 0.05: [0.030, 0.083]); the lower
        # edge is 0.01 (re-review of 2026-09-23): n_matched takes ~10
        # distinct values here, the exact exchangeable rate of the
        # discrete p at B = 199 is 0.029 (Wilson [0.015, 0.054] at R = 300;
        # printed from this loop's own nulls), and 0.02 rejected a correct
        # null at 2 of 6 fresh seeds (0.017 and 0.023 measured). Mean zeta
        # within +/-0.15 (se 0.058 at R = 300, 2.6 se) carries the bias
        # detection; measured with the spline +0.08 (the ellipse path
        # +0.02).
        out = fpr_loop(20, 300, 42)
        st["k20"] = out
        z_mod, z_alt = float(np.nanmean(out["zeta"])), float(np.nanmean(out["alt_zeta"]))
        fe, fd, f2 = frac_le(out["pe"]), frac_le(out["pd"]), frac_le(out["p2"])
        for name, f in (("K = 20 p_excess", fe), ("K = 20 p_deficit", fd), ("K = 20 p_two_sided", f2)):
            assert_band(f"{name} <= 0.05", f, LO_SPARSE, HI)
        assert abs(z_mod) <= 0.15, z_mod
        ex = [float(out[k].mean()) for k in ("ex_pe", "ex_pd", "ex_p2")]
        lo_w, hi_w = wilson_interval(min(ex), 300)
        return (f"300 pairs, K = 20: p_excess / p_deficit / p_two_sided <= 0.05: {fe:.3f} / {fd:.3f} / {f2:.3f} (ellipse path "
                f"{frac_le(out['alt_pe']):.3f} / {frac_le(out['alt_pd']):.3f} / {frac_le(out['alt_p2']):.3f}; exact "
                f"exchangeable rates {ex[0]:.3f} / {ex[1]:.3f} / {ex[2]:.3f}, Wilson for 300 at {min(ex):.3f}: [{lo_w:.3f}, "
                f"{hi_w:.3f}]); mean zeta {z_mod:+.3f} (ellipse path {z_alt:+.3f}, difference {z_mod - z_alt:+.3f}); E* "
                f"{out['estar'].mean():.3f} vs {out['alt_estar'].mean():.3f}; {out['time'].sum():.0f} s")

    def jittered_rings_diagnostic():
        # Printed, not asserted (re-review of 2026-09-23, pending design
        # point of D-24a): M1 rings whose clusters scatter about the
        # ellipse by N(0, jitter) on each axis, built as real rings are
        # (ring_geometry_from_centroids: reconstruct_perimeter tour, so a
        # jitter-induced swap of two neighbours is untangled by 2-opt as it
        # would be on a measured ring); the module's zeta against the
        # reference that shifts each cluster along the exact ellipse
        # carrying its jitter vector, on the SAME recorded draws
        # (carried_jitter_null): the paired difference is the spline
        # path's bias on scattered vertices. Reviewer's figures at R = 600:
        # +0.125 (K = 40, 15 nm), +0.219 (K = 20, 15 nm; module mean zeta
        # +0.229), +0.097 (K = 20, 8 nm); -0.088 at the realistic scale.
        eclipse_test, match_rings, from_centroids = require("eclipse_test", "match_rings", "ring_geometry_from_centroids")
        lines = []
        for K, jitter, R_j, seed in ((40, 15.0, 120, 44), (20, 15.0, 150, 45), (20, 8.0, 150, 46)):
            rng = np.random.default_rng(seed)
            z_mod, z_ref, es_mod, es_ref, pe_mod, pe_ref, dev = [], [], [], [], [], [], []
            t0 = time.perf_counter()
            for r in range(R_j):
                a, b = ring_m1(rng, 0, K, jitter), ring_m1(rng, 1, K, jitter)
                ga, gb = from_centroids(0, a.xy_nm), from_centroids(1, b.xy_nm)
                m = eclipse_test(ga, gb, tau, n_null=B, random_seed=r)
                null = carried_jitter_null(match_rings, a.xy_nm, b, tau, np.asarray(m.null_shift_nm, dtype=float),
                                           np.asarray(m.null_reflect))
                sd = null.std(ddof=1)
                z_mod.append(m.zeta); z_ref.append((m.n_matched - null.mean()) / sd if sd > 0 else np.nan)
                es_mod.append(m.E_star); es_ref.append(null.mean() / K)
                pe_mod.append(m.p_excess); pe_ref.append(phipson_smyth(null, m.n_matched)[0])
                dev.append(float(ellipse_deviation_nm(np.asarray(gb.path.points_nm)).max()))
            zm, zr = np.asarray(z_mod), np.asarray(z_ref)
            d = zm - zr
            lines.append(f"K = {K}, {jitter:g} nm, R = {R_j}: module mean zeta {np.nanmean(zm):+.3f} (se "
                         f"{np.nanstd(zm) / math.sqrt(R_j):.3f}), jitter-carrying ellipse reference {np.nanmean(zr):+.3f}, paired "
                         f"difference {np.nanmean(d):+.3f} (se {np.nanstd(d) / math.sqrt(R_j):.3f}); E* {np.mean(es_mod):.3f} vs "
                         f"{np.mean(es_ref):.3f}; p_excess <= 0.05: {frac_le(np.asarray(pe_mod)):.3f} vs "
                         f"{frac_le(np.asarray(pe_ref)):.3f}; spline off the ellipse median {np.median(dev):.0f} nm (max "
                         f"{np.max(dev):.0f}); {time.perf_counter() - t0:.0f} s")
        st["jitter_lines"] = lines
        return "PENDING (D-24a, printed): " + " | ".join(lines)

    def realistic_scale_diagnostic():
        # Printed, not asserted (fix decisions item 1): the spacing and
        # tolerance of a dense ring -- a circle of radius 1200 nm,
        # K = 60 (mean spacing 126 nm), tau = 80 nm, R = 200 pairs. The
        # chord sagitta is 126^2 / (8 x 1200) = 1.6 nm there, so even the
        # polygon path was unbiased at this density (+0.125 vs +0.129 along
        # the exact circle, review of 2026-09-23); the spline's figure and
        # the exact-circle reference are printed for the H3 report.
        out = fpr_loop(60, 200, 43, tau_nm=80.0, membrane="circle")
        z_mod, z_alt = float(np.nanmean(out["zeta"])), float(np.nanmean(out["alt_zeta"]))
        se = float(np.nanstd(out["zeta"]) / math.sqrt(out["zeta"].size))
        return (f"200 pairs, circle R 1200 nm, K = 60 (p_bar {out['pbar'].mean():.0f} nm), tau 80: module mean zeta {z_mod:+.3f} "
                f"(se {se:.3f}), exact-circle path {z_alt:+.3f} (difference {z_mod - z_alt:+.3f}); p_excess / p_deficit / "
                f"p_two_sided <= 0.05: {frac_le(out['pe']):.3f} / {frac_le(out['pd']):.3f} / {frac_le(out['p2']):.3f} (circle "
                f"path {frac_le(out['alt_pe']):.3f} / {frac_le(out['alt_pd']):.3f} / {frac_le(out['alt_p2']):.3f}); E* "
                f"{out['estar'].mean():.3f} vs {out['alt_estar'].mean():.3f}; {out['time'].sum():.0f} s")

    check("FPR loop: 500 M1 pairs through eclipse_test (n_null 199) and the ellipse-path diagnostic", loop)
    check("FPR under M1: fraction of p_excess <= 0.05 in [0.02, 0.09]", excess_band)
    check("FPR under M1: fraction of p_deficit <= 0.05 in [0.02, 0.09]", deficit_band)
    check("FPR under M1: fraction of p_two_sided <= 0.05 in [0.02, 0.09]", two_sided_band)
    check("FPR under M1: mean zeta within 0.15 of 0", zeta_centred)
    check("null mean E* within 0.08 of 1 - exp(-2 tau / p_bar) (Poisson formula; hard core raises it)", estar_vs_formula)
    check("reference: same rings, null shifted along the exact ellipse -> fractions in [0.02, 0.09], mean zeta within 0.15, "
          "and the module's mean zeta within 0.10 of it (no path bias)", ellipse_path_diagnostic)
    check("sparse rings (K = 20, 400 nm gaps; R = 300, n_null 199; D-24a): p_excess / p_deficit / p_two_sided rates in "
          "[0.01, 0.09] (lower edge: exact exchangeable rate 0.029 of the discrete p, Wilson [0.015, 0.054]) and mean zeta "
          "within 0.15 of 0 (the polygon path gave +0.85 and 0.137)", sparse_rings_k20)
    check("realistic-scale diagnostic (printed): circle R = 1200 nm, K = 60, tau = 80 nm, R = 200 pairs -- module vs the exact "
          "circle path", realistic_scale_diagnostic)
    check("jittered-M1 diagnostic (printed, pending design point of D-24a): clusters scattered about the ellipse by 15 / 8 nm, "
          "K = 40 / 20, reconstruct_perimeter tours; module vs the reference shifting each cluster along the exact ellipse "
          "with its jitter vector on the same draws", jittered_rings_diagnostic)


# ============================================================ 5. M5 columns
def test_implanted_columns() -> None:
    print("\n5. Implanted columns M5 (f = 0.2, 0.5, 1.0; sigma_col 10 nm; tau 60 nm; R = 200 / 400 / 200 pairs; n_null 199)")
    st: Dict[str, Any] = {}
    B, tau = 199, 60.0
    FS = (0.2, 0.5, 1.0)
    R_OF = {0.2: 200, 0.5: 400, 1.0: 200}      # R = 400 at f = 0.5 (fix decisions item 7): MC sd of the power 1.6 %
    # Power floors (item 7): "measured - 2 MC sd" at this seed, measured
    # with the D-24 module (spline path, maximum-cardinality matching):
    # f = 0.5: 88.2 % of 400 (sd 1.6 %) -> 84.5 %; f = 1: 100 % of 200 ->
    # Wilson lower bound 98.1 % -> 98.0 % (power_floor, rounded down to
    # 0.5 %). The specification asked 95 % at both; at f = 0.5 the model
    # does not reach it (half the clusters are fresh and compete for the
    # parents the copies left; the first version measured 86.5-91 % at
    # R = 200), and the ratified floor is the measured power minus 2 sd.
    MEASURED = {0.5: (0.882, 400), 1.0: (1.0, 200)}
    FLOOR = {f: power_floor(p, n) for f, (p, n) in MEASURED.items()}

    def loop():
        eclipse_test = require("eclipse_test")
        t0 = time.perf_counter()
        for f in FS:
            rng = np.random.default_rng(51)
            zeta, pe, dE, es = [], [], [], []
            for r in range(R_OF[f]):
                a = ring_m1(rng, 0)
                b = ring_m5(rng, 1, a, f)
                m = eclipse_test(geometry_of(a), geometry_of(b), tau, n_null=B, random_seed=r)
                zeta.append(m.zeta); pe.append(m.p_excess); dE.append(m.E_dir - m.E_star); es.append(m.E_star)
            st[f] = {k: np.asarray(v, dtype=float) for k, v in
                     (("zeta", zeta), ("pe", pe), ("dE", dE), ("estar", es))}
        return f"{' + '.join(str(R_OF[f]) for f in FS)} pairs in {elapsed(t0)}"

    def median_zeta_increases():
        med = [float(np.nanmedian(need(st, f)["zeta"])) for f in FS]
        assert med[0] < med[1] < med[2], med
        return "median zeta " + ", ".join(f"f = {f}: {m:.2f}" for f, m in zip(FS, med))

    def zeta_positive():
        out = []
        for f in (0.5, 1.0):
            frac = float(np.mean(need(st, f)["zeta"] > 0))
            assert frac >= 0.95, (f, frac)
            out.append(f"f = {f}: {100 * frac:.1f} %")
        return "zeta > 0 in " + ", ".join(out)

    def power():
        out = []
        for f in (0.5, 1.0):
            frac = frac_le(need(st, f)["pe"])
            p, n = MEASURED[f]
            sd = math.sqrt(p * (1 - p) / n)
            assert frac >= FLOOR[f], (f, frac, FLOOR[f])
            out.append(f"f = {f}: {100 * frac:.1f} % of {R_OF[f]} (floor {100 * FLOOR[f]:.1f} % = measured "
                       f"{100 * p:.1f} % - 2 x {100 * sd:.1f} %{'; Wilson lower bound' if p >= 1.0 else ''})")
        f02 = frac_le(need(st, 0.2)["pe"])
        return "p_excess <= 0.05 in " + "; ".join(out) + f"; f = 0.2: {100 * f02:.1f} % (printed)"

    def excess_approximation():
        # Copies always matched (jitter 10 nm << tau), fresh clusters at the
        # chance rate: E_dir - E* ~ f (1 - E*). The fresh clusters actually
        # compete for the parents the copies left (a fresh cluster that
        # lands within tau of a parent whose copy already took it adds
        # nothing), so the excess is below the formula by up to f (1 - f)
        # x (chance rate) ~ 0.25 x 0.3 = 0.08 at f = 0.5 (measured deviation
        # 0.079 at f = 0.5, 0.041 at f = 0.2, 0.000 at f = 1, R = 400 / 200):
        # the tolerance is 0.10, ratified (item 7) instead of the
        # specification's 0.08, which the measured f = 0.5 deviation sits
        # on; the approximation is stated as such.
        out = []
        for f in FS:
            d = need(st, f)
            got, exp = float(d["dE"].mean()), f * (1.0 - float(d["estar"].mean()))
            assert abs(got - exp) <= 0.10, (f, got, exp)
            out.append(f"f = {f}: {got:.3f} vs {exp:.3f} (deviation {got - exp:+.3f})")
        return "mean(E_dir - E*) vs f (1 - E*): " + "; ".join(out)

    check("M5 loop: 200 + 400 + 200 pairs through eclipse_test", loop)
    check("M5: median zeta increases with f", median_zeta_increases)
    check("M5 f = 0.5 and f = 1: zeta > 0 in >= 95 % of the pairs", zeta_positive)
    check(f"M5 power: p_excess <= 0.05 in >= {100 * FLOOR[0.5]:.1f} % at f = 0.5 (R = 400; measured 88.2 % - 2 MC sd) and "
          f">= {100 * FLOOR[1.0]:.1f} % at f = 1 (measured 100 % of 200, Wilson lower bound)", power)
    check("M5: E_dir - E* within 0.10 of f (1 - E*) for every f (fresh clusters compete for the parents the copies left: "
          "measured deviation 0.08 at f = 0.5)", excess_approximation)


# ============================================================ 6. M6 alternation
def test_alternation() -> None:
    print("\n6. Alternation M6 (midpoints of the gaps, jitter 5 nm; R = 200; n_null 199)")
    st: Dict[str, Any] = {}
    R, B = 200, 199

    def loop():
        eclipse_test = require("eclipse_test")
        t0 = time.perf_counter()
        for tau in (40.0, 90.0):
            rng = np.random.default_rng(61)
            zeta, pd, ed, es = [], [], [], []
            for r in range(R):
                a = ring_m1(rng, 0)
                b = ring_m6(rng, 1, a)
                m = eclipse_test(geometry_of(a), geometry_of(b), tau, n_null=B, random_seed=r)
                zeta.append(m.zeta); pd.append(m.p_deficit); ed.append(m.E_dir); es.append(m.E_star)
            st[tau] = {k: np.asarray(v, dtype=float) for k, v in
                       (("zeta", zeta), ("pd", pd), ("E_dir", ed), ("estar", es))}
        return f"2 x {R} pairs in {elapsed(t0)}"

    # Power floors as "measured - 2 MC sd" (fix decisions item 7), measured
    # with the D-24 module at this seed: zeta < 0 in 100 % of 200 (Wilson
    # lower bound 98.1 % -> 98.0 %), p_deficit <= 0.05 in 94.0 % of 200
    # (sd 1.7 %; the first version, dummies at tau, measured 96.5 %: the
    # maximum-cardinality null matches a little more) -> 90.5 %. Both are
    # at or above the specification's 90 %.
    MEASURED = {"zeta": (1.0, 200), "pd": (0.94, 200)}
    FLOOR = {k: power_floor(p, n) for k, (p, n) in MEASURED.items()}

    def deficit_at_40():
        # tau = 40 < half the mean spacing (99 nm): a midpoint is matched only
        # when its gap is < 80 nm (measured E_dir 0.17 vs E* 0.39, zeta -3.4).
        d = need(st, 40.0)
        fz, fp = float(np.mean(d["zeta"] < 0)), frac_le(d["pd"])
        assert fz >= FLOOR["zeta"] and fp >= FLOOR["pd"], (fz, fp, FLOOR)
        return (f"zeta < 0 in {100 * fz:.1f} % (floor {100 * FLOOR['zeta']:.1f} %), p_deficit <= 0.05 in {100 * fp:.1f} % "
                f"(floor {100 * FLOOR['pd']:.1f} % = measured 94.0 % - 2 x 1.7 %); median zeta {np.nanmedian(d['zeta']):.2f}, "
                f"E_dir {d['E_dir'].mean():.3f} vs E* {d['estar'].mean():.3f}")

    def weaker_at_90():
        d40, d90 = need(st, 40.0), need(st, 90.0)
        assert np.nanmedian(d90["zeta"]) > np.nanmedian(d40["zeta"])
        assert frac_le(d90["pd"]) < frac_le(d40["pd"])
        return (f"tau = 90: median zeta {np.nanmedian(d90['zeta']):.2f}, p_deficit <= 0.05 in {100 * frac_le(d90['pd']):.1f} %, "
                f"E_dir {d90['E_dir'].mean():.3f} vs E* {d90['estar'].mean():.3f} (the specification's 'E_dir ~ 1' is not reached: "
                f"half the mean spacing is 99 nm)")

    check("M6 loop: 200 pairs at tau = 40 and at tau = 90", loop)
    check(f"M6 tau = 40 nm: zeta < 0 in >= {100 * FLOOR['zeta']:.1f} % and p_deficit <= 0.05 in >= {100 * FLOOR['pd']:.1f} % "
          "of the pairs (measured 100 % and 94.0 % - 2 MC sd; specification 90 %)", deficit_at_40)
    check("M6 tau = 90 nm: the deficit weakens (median zeta higher, fewer p_deficit <= 0.05); E_dir and E* printed",
          weaker_at_90)


# ============================================================ 7. delta_bar
def test_delta_bar() -> None:
    print("\n7. delta_bar and torsion: rotation by 30 nm of arc; rigid translation by (20, -22) nm")
    tau = 60.0

    def rotation():
        eclipse_test = require("eclipse_test")
        rng = np.random.default_rng(71)
        norms, dmed = [], []
        for r in range(5):
            a = ring_m1(rng, 0)
            b = ring_rotated(1, a, 30.0)
            m = eclipse_test(geometry_of(a), geometry_of(b), tau, n_null=19, random_seed=r)
            assert m.n_matched == 40, m.n_matched
            ia, jb = np.asarray(m.i_a), np.asarray(m.j_b)
            assert np.array_equal(b.parent[jb], ia), "pairs are not the rotated clusters"
            truth = (b.xy_nm[jb] - a.xy_nm[ia]).mean(axis=0)
            assert np.allclose(np.asarray(m.delta_bar_nm), truth, atol=1e-9)
            # chord of a 30 nm arc on the ellipse: 30 (1 - (30 / R)^2 / 24) >= 29.99
            assert abs(m.d_median_nm - 30.0) < 0.005 * 30.0, m.d_median_nm
            norms.append(float(np.linalg.norm(truth)))
            dmed.append(m.d_median_nm)
        assert max(norms) < 15.0, norms
        return (f"5 rings: all 40 matched to their rotated selves; d_median {np.mean(dmed):.4f} nm; "
                f"|delta_bar| {np.round(norms, 2).tolist()} nm (tangent displacements cancel; not ~30)")

    def translation():
        eclipse_test = require("eclipse_test")
        rng = np.random.default_rng(72)
        a = ring_m1(rng, 0)
        b = ring_translated(1, a, (20.0, -22.0))
        m = eclipse_test(geometry_of(a), geometry_of(b), tau, n_null=19, random_seed=0)
        assert m.n_matched == 40 and np.array_equal(b.parent[np.asarray(m.j_b)], np.asarray(m.i_a))
        assert np.allclose(np.asarray(m.delta_bar_nm), [20.0, -22.0], atol=1e-9), m.delta_bar_nm
        assert abs(m.d_median_nm - math.hypot(20.0, 22.0)) < 1e-9, m.d_median_nm
        return f"all matched; delta_bar {np.round(np.asarray(m.delta_bar_nm), 9).tolist()} nm; d_median {m.d_median_nm:.4f} nm"

    check("rotation by 30 nm of arc (no jitter, tau 60): all matched to the rotated clusters, d_median within 0.5 % of 30 nm, "
          "delta_bar == mean over the true pairs (1e-9) with |delta_bar| < 15 nm", rotation)
    check("rigid translation by (20, -22) nm: all matched, delta_bar == (20, -22) (1e-9), d_median == 29.732 nm (1e-9)",
          translation)


# ============================================================ 8. curve
def test_curve() -> None:
    print("\n8. eclipse_curve over tau: monotonicity, envelope, shared replicates, p_global")
    st: Dict[str, Any] = {}
    B = 199
    grid = np.arange(20.0, 201.0, 10.0)
    grid9 = np.arange(20.0, 181.0, 20.0)

    def curve_on_m5():
        eclipse_curve, match_rings = require("eclipse_curve", "match_rings")
        rng = np.random.default_rng(81)
        a = ring_m1(rng, 0)
        b = ring_m5(rng, 1, a, 0.5)
        ga, gb = geometry_of(a), geometry_of(b)
        st["ga"], st["gb"] = ga, gb
        t0 = time.perf_counter()
        c = eclipse_curve(ga, gb, grid, n_null=B, random_seed=0)
        st["c"] = c
        dt = time.perf_counter() - t0
        assert c.ring_a == 0 and c.ring_b == 1 and c.n_null == B and c.random_seed == 0
        assert np.allclose(np.asarray(c.tau_grid_nm), grid)
        obs = np.asarray(c.n_matched_obs)
        exp = np.array([n_pairs(match_rings, a.xy_nm, b.xy_nm, t) for t in grid])
        assert np.array_equal(obs, exp), (obs, exp)
        E = np.asarray(c.E_dir_obs)
        assert np.allclose(E, obs / 40.0, atol=1e-12)
        Es, sd, lo, hi, z = (np.asarray(getattr(c, k), dtype=float) for k in
                             ("E_star", "sd_star", "envelope_lo", "envelope_hi", "zeta"))
        assert np.all(np.diff(E) >= -1e-12) and np.all(np.diff(Es) >= -1e-12)
        assert np.all(lo <= Es + 1e-12) and np.all(Es <= hi + 1e-12), (lo, Es, hi)
        assert np.all(lo <= hi) and np.all(lo >= 0.0) and np.all(hi <= 1.0)
        ok = sd > 0
        assert np.allclose(z[ok], (E - Es)[ok] / sd[ok], atol=1e-9)
        t_max = float(np.max(np.abs(E - Es)[ok] / sd[ok]))
        assert abs(c.t_max_obs - t_max) < 1e-9, (c.t_max_obs, t_max)
        assert 1.0 / (B + 1) <= c.p_global <= 1.0 and c.p_global <= 0.05, c.p_global
        assert bool(c.include_suspect) is True and isinstance(c.warnings, list)
        return (f"{grid.size} tau in {dt:.1f} s; E_dir {E[0]:.2f} -> {E[-1]:.2f}, E* {Es[0]:.2f} -> {Es[-1]:.2f}, "
                f"t_max {t_max:.2f}, p_global {c.p_global:.3f}")

    def shared_replicates():
        eclipse_test = require("eclipse_test")
        ga, gb, c = need(st, "ga"), need(st, "gb"), need(st, "c")
        Es, sd = np.asarray(c.E_star, dtype=float), np.asarray(c.sd_star, dtype=float)
        for k in (0, grid.size // 2, grid.size - 1):
            m = eclipse_test(ga, gb, float(grid[k]), n_null=B, random_seed=0)
            assert abs(m.E_star - Es[k]) < 1e-12 and abs(m.sd_star - sd[k]) < 1e-12, (grid[k], m.E_star, Es[k])
        m_lo = eclipse_test(ga, gb, float(grid[0]), n_null=B, random_seed=0)
        m_hi = eclipse_test(ga, gb, float(grid[-1]), n_null=B, random_seed=0)
        assert np.array_equal(np.asarray(m_lo.null_shift_nm), np.asarray(m_hi.null_shift_nm))
        assert np.array_equal(np.asarray(m_lo.null_reflect), np.asarray(m_hi.null_reflect))
        assert np.all(np.asarray(m_lo.null_n_matched) <= np.asarray(m_hi.null_n_matched))
        # D-24b: the maximum-cardinality matching is monotone in tau, so the
        # (T, B) null matrix must be non-decreasing along tau in EVERY
        # replicate (a real assertion, not a diagnostic).
        N = np.asarray(c.null_n_matched)
        assert N.shape == (grid.size, B), N.shape
        assert np.all(np.diff(N, axis=0) >= 0), "per-tau null matrix not monotone replicate by replicate"
        assert np.array_equal(N[0], np.asarray(m_lo.null_n_matched)) and np.array_equal(N[-1], np.asarray(m_hi.null_n_matched))
        return ("E*/sd* at tau 20, 110, 200 equal eclipse_test with the same seed (1e-12); eclipse_test at 20 and 200 draws "
                "the same shifts; the (T, B) null matrix is non-decreasing in tau in every replicate and its first/last rows "
                "are eclipse_test's")

    def p_global_brute_force():
        # Each of the B + 1 curves (observed first) scored against the other
        # B by brute force: leave one column out, mean and sd (ddof 1) per
        # tau, max over tau of |x - mean| / sd (0/0 = 0, x/0 = inf), then
        # (#{t_r >= t_obs} + 1) / (B + 1). On the M5 curve and on an M1 pair.
        eclipse_curve = require("eclipse_curve")
        rng = np.random.default_rng(83)
        a, b = ring_m1(rng, 0), ring_m1(rng, 1)
        curves = [need(st, "c"), eclipse_curve(geometry_of(a), geometry_of(b), grid, n_null=B, random_seed=3)]
        out = []
        for c in curves:
            N = np.asarray(c.null_n_matched)
            N = N if N.shape == (grid.size, B) else N.T
            E = np.concatenate([np.asarray(c.E_dir_obs)[:, None], N / 40.0], axis=1)
            t = np.empty(B + 1)
            for q in range(B + 1):
                others = np.delete(E, q, axis=1)
                mu, sd = others.mean(axis=1), others.std(axis=1, ddof=1)
                dev = np.abs(E[:, q] - mu)
                with np.errstate(divide="ignore", invalid="ignore"):
                    tt = np.where(sd > 0, dev / sd, np.where(dev > 0, np.inf, 0.0))
                t[q] = tt.max()
            p = (int(np.sum(t[1:] >= t[0])) + 1) / (B + 1)
            assert abs(c.t_max_obs - t[0]) < 1e-9, (c.t_max_obs, t[0])
            assert abs(c.p_global - p) < 1e-12, (c.p_global, p)
            # the observed statistic is against the B replicates' own mean/sd
            mu, sd = E[:, 1:].mean(axis=1), E[:, 1:].std(axis=1, ddof=1)
            ok = sd > 0
            assert abs(t[0] - np.max(np.abs(E[ok, 0] - mu[ok]) / sd[ok])) < 1e-9
            out.append(f"t_obs {t[0]:.2f}, p {p:.3f}")
        return "M5 curve: " + out[0] + "; M1 curve: " + out[1] + " (brute force == module, 1e-12)"

    def p_global_fpr():
        # Max-studentized deviation (Myllymaki et al. 2017 in spirit); each
        # of the B + 1 curves scored against the other B (D-24c), so the
        # statistics are exchangeable and the test exact at every B. The
        # all-replicates scoring of the specification is liberal by O(1/B)
        # (measured p_global <= 0.05 in 18 % of exchangeable Gaussian curves
        # at B = 19 against 5.0 % leave-one-out; 0.060-0.085 on M1 rings at
        # B = 199 with the first H3 version). Band [0.02, 0.09] (item 3):
        # Wilson 95 % for R = 200 at 0.05 is [0.027, 0.089].
        eclipse_curve = require("eclipse_curve")
        rng = np.random.default_rng(82)
        t0 = time.perf_counter()
        ps = np.empty(200)
        for r in range(200):
            a, b = ring_m1(rng, 0), ring_m1(rng, 1)
            ps[r] = eclipse_curve(geometry_of(a), geometry_of(b), grid9, n_null=B, random_seed=r).p_global
        f = frac_le(ps)
        assert_band("p_global <= 0.05", f, 0.02, 0.09)
        lo, hi = wilson_interval(0.05, 200)
        return (f"fraction {f:.3f} (Wilson 95 % for 200 at 0.05: [{lo:.4f}, {hi:.4f}]; <= 0.10: {frac_le(ps, 0.10):.3f}); "
                f"200 curves x 9 tau in {elapsed(t0)}")

    def leave_one_out_closed_form():
        # Item 3: the closed form (per-tau sums and sums of squares) equals
        # the explicit leave-one-out computation to 1e-9 on a small case
        # (T = 5, 12 curves of integers over 8, including a tau where all
        # but one curve share a constant).
        loo = require("_leave_one_out_max_deviation")
        rng = np.random.default_rng(84)
        curves = rng.integers(0, 9, size=(5, 12)).astype(float) / 8.0
        curves[2, :] = 0.5
        curves[2, 3] = 0.75
        t = np.asarray(loo(curves))
        worst = 0.0
        for q in range(12):
            others = np.delete(curves, q, axis=1)
            mu, sd = others.mean(axis=1), others.std(axis=1, ddof=1)
            dev = np.abs(curves[:, q] - mu)
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = np.where(sd > 0, dev / sd, np.where(dev > 0, np.inf, 0.0))
            if np.isinf(tt.max()):
                assert np.isinf(t[q]), (q, t[q])
            else:
                worst = max(worst, abs(float(tt.max()) - float(t[q])))
        assert worst < 1e-9, worst
        assert np.isinf(t[3]) and not np.isinf(t[0]), t
        return f"12 curves x 5 tau: closed form == explicit leave-one-out within {worst:.1e}; the curve alone off a constant tau is +inf"

    check("eclipse_curve on an M5 f = 0.5 pair: grid recorded, n_matched_obs == match_rings per tau, E_dir and E* "
          "non-decreasing, envelope_lo <= E* <= envelope_hi, zeta and t_max_obs recomputed (1e-9), p_global <= 0.05",
          curve_on_m5)
    check("the same replicates serve every tau: curve E*/sd* == eclipse_test at the same seed; eclipse_test null counts "
          "at tau_min <= at tau_max replicate by replicate", shared_replicates)
    check("p_global by brute force (each of the B + 1 curves against the other B): t_max_obs (1e-9) and p_global (1e-12) "
          "on the M5 and an M1 curve", p_global_brute_force)
    check("p_global under M1 (R = 200, 9-point grid, n_null 199): fraction <= 0.05 in [0.02, 0.09] (Wilson [0.027, 0.089])",
          p_global_fpr)
    check("leave-one-out studentisation: closed form from the per-tau sums equals the explicit computation (1e-9)",
          leave_one_out_closed_form)


# ============================================================ 9. joint null
def test_joint_null() -> None:
    print("\n9. Joint null per axon: three rings, the first never shifted")
    st: Dict[str, Any] = {}
    B, tau = 199, 60.0

    def bookkeeping():
        axon_joint_null, eclipse_test = require("axon_joint_null", "eclipse_test")
        rng = np.random.default_rng(91)
        rings = [ring_m1(rng, k) for k in range(3)]
        geoms = [geometry_of(r) for r in rings]
        st["geoms"] = geoms
        t0 = time.perf_counter()
        j = axon_joint_null(geoms, tau, n_null=B, random_seed=0)
        st["j"] = j
        dt = time.perf_counter() - t0
        assert [tuple(p) for p in j.pairs] == [(0, 1), (1, 2)], j.pairs
        m01 = eclipse_test(geoms[0], geoms[1], tau, n_null=B, random_seed=0)
        m12 = eclipse_test(geoms[1], geoms[2], tau, n_null=B, random_seed=0)
        assert np.array_equal(np.asarray(j.n_matched_obs), [m01.n_matched, m12.n_matched])
        assert j.T_obs == m01.n_matched + m12.n_matched and j.tau_nm == tau
        T = np.asarray(j.T_null)
        N = np.asarray(j.null_n_matched)
        assert T.shape == (B,) and N.shape == (B, 2) and np.array_equal(N.sum(axis=1), T)
        mins = [40, 40]
        TA = float(np.mean([(j.n_matched_obs[q] - N[:, q].mean()) / mins[q] for q in range(2)]))
        assert abs(j.T_A - TA) < 1e-12, (j.T_A, TA)
        sd0, sd1 = T.std(ddof=0), T.std(ddof=1)
        assert any(abs(j.z_A - (j.T_obs - T.mean()) / sd) < 1e-9 for sd in (sd0, sd1)), j.z_A
        pe, pd, p2 = phipson_smyth(T, j.T_obs)
        assert abs(j.p_excess - pe) < 1e-12 and abs(j.p_deficit - pd) < 1e-12 and abs(j.p_two_sided - p2) < 1e-12
        assert j.n_null == B and j.random_seed == 0 and isinstance(j.warnings, list)
        return f"T_obs {j.T_obs} = {m01.n_matched} + {m12.n_matched}; T_A {j.T_A:+.3f}, z_A {j.z_A:+.2f}, p {j.p_two_sided:.3f}; {dt:.1f} s"

    def first_ring_fixed():
        helper, arc_shift, match_rings = require("joint_shift_statistic", "arc_shift", "match_rings")
        geoms, j = need(st, "geoms"), need(st, "j")
        U = np.asarray(j.null_shift_nm, dtype=float)
        ref = np.asarray(j.null_reflect)
        assert U.shape == (B, 3) and ref.shape == (B, 3) and ref.dtype == bool
        assert np.all(U[:, 0] == 0.0) and not np.any(ref[:, 0]), "the first ring was shifted"
        for k in (1, 2):
            L, m = float(geoms[k].length_nm), 0.5 * float(geoms[k].length_nm) / 40
            assert np.all(U[:, k] >= m - 1e-9) and np.all(U[:, k] <= L - m + 1e-9), (k, U[:, k].min(), U[:, k].max())
            assert 0 < int(ref[:, k].sum()) < B
        assert not np.array_equal(U[:, 1], U[:, 2]), "rings 2 and 3 share their shifts"
        T0, per0 = helper(geoms, tau, [0.0, 0.0, 0.0], [False, False, False])
        assert int(T0) == j.T_obs and np.array_equal(np.asarray(per0), np.asarray(j.n_matched_obs))
        for bad_shift, bad_ref in (([50.0, 0.0, 0.0], [False, False, False]), ([0.0, 0.0, 0.0], [True, False, False])):
            try:
                helper(geoms, tau, bad_shift, bad_ref)
            except ValueError:
                pass
            else:
                raise AssertionError("a shift of the first ring was accepted")
        T = np.asarray(j.T_null)
        N = np.asarray(j.null_n_matched)
        for r in range(B):
            Tr, per = helper(geoms, tau, U[r], ref[r])
            assert int(Tr) == T[r] and np.array_equal(np.asarray(per), N[r]), r
        # by hand, both rings of a pair shifted
        Ub, Uc = 1234.5, 3210.0
        pb = np.asarray(arc_shift(geoms[1], Ub, True))
        pc = np.asarray(arc_shift(geoms[2], Uc, False))
        n01 = n_pairs(match_rings, np.asarray(geoms[0].centroids_nm), pb, tau, geoms[0].usable, geoms[1].usable)
        n12 = n_pairs(match_rings, pb, pc, tau, geoms[1].usable, geoms[2].usable)
        Th, perh = helper(geoms, tau, [0.0, Ub, Uc], [False, True, False])
        assert int(Th) == n01 + n12 and np.array_equal(np.asarray(perh), [n01, n12]), (Th, n01, n12)
        return (f"null_shift_nm[:, 0] == 0, null_reflect[:, 0] False; zero shifts reproduce T_obs; a first-ring shift raises; "
                f"all {B} T_null reproduced from the recorded draws; (0, {Ub}, {Uc}) == arc_shift + match_rings by hand "
                f"({n01} + {n12})")

    def rule_from_first_principles(j: Any, geoms: Sequence[Any], k: int) -> Tuple[int, int, float, int, float]:
        """The D-24d window of ring k (position k in the sorted geometries)
        recomputed here: o = arc on ring k's dense table of ring k-1's first
        vertex (nearest point), sigma = product of the shoelace signs, m =
        half the mean spacing; the re-creation offset is sigma U_{k-1} when
        neither ring is reflected and sigma U_{k-1} + 2 o + sigma L_{k-1}
        when both are (mod L_k). Returns (offending draws, draws within 2 m
        of the window -- the band just outside the rule --, o, sigma, m)."""
        U, ref = np.asarray(j.null_shift_nm, dtype=float), np.asarray(j.null_reflect)
        gk, gp = geoms[k], geoms[k - 1]
        L, Lp = float(gk.length_nm), float(gp.length_nm)
        m = 0.5 * L / gk.n_clusters
        o = float(nearest_on_polygon(np.asarray(gp.contour_nm)[:1], np.asarray(gk.path.points_nm))[1][0])
        sigma = 1 if signed_area_nm2(gp.contour_nm) * signed_area_nm2(gk.contour_nm) >= 0 else -1
        target = np.where(ref[:, k], 2.0 * o + sigma * Lp, 0.0)
        d = cyclic_distance_nm(U[:, k] - sigma * U[:, k - 1] - target, L)
        same = ref[:, k] == ref[:, k - 1]
        assert abs(float(np.asarray(j.min_shift_nm)[k]) - m) < 1e-9
        assert abs(float(np.asarray(j.origin_offset_nm)[k]) - o) < 1e-6, (j.origin_offset_nm[k], o)
        assert int(np.asarray(j.tour_orientation)[k]) == sigma
        assert np.all(U[:, k] >= m - 1e-9) and np.all(U[:, k] <= L - m + 1e-9)
        return int(np.sum(same & (d < m))), int(np.sum(same & (d < 2 * m))), o, sigma, m

    def relative_shift_rule():
        # D-24d: for every shifted ring after the first, a draw with the
        # previous ring's reflection whose shift lies within m_k (cyclic,
        # modulo L_k) of the offset that re-creates the previous pair's
        # phase is redrawn. Before the fix 1.1 % of the replicates (22 of
        # 1999 on the copies axon; 1 / (2 K) = 1.25 %) re-created the phase
        # of pair (1, 2) and counted 29.8 pairs against 20.8. Checked on
        # four rings (two ring pairs subject to the rule) with n_null 1999,
        # with the window recomputed from first principles (o, sigma), and
        # that every T_null is still reproduced from the recorded draws.
        # These copies rings share their first vertex (o ~ the 10 nm
        # jitter), the case where the first rule was already right.
        axon_joint_null, helper = require("axon_joint_null", "joint_shift_statistic")
        rng = np.random.default_rng(94)
        a = ring_m1(rng, 0)
        geoms = [geometry_of(a)] + [geometry_of(ring_m5(rng, k, a, 1.0)) for k in (1, 2, 3)]
        j = axon_joint_null(geoms, tau, n_null=1999, random_seed=0)
        U, ref = np.asarray(j.null_shift_nm, dtype=float), np.asarray(j.null_reflect)
        assert np.asarray(j.origin_offset_nm).shape == (4,) and np.asarray(j.tour_orientation).shape == (4,)
        assert np.all(np.asarray(j.origin_offset_nm)[:2] == 0.0) and np.all(np.asarray(j.tour_orientation) == 1)
        n_rule, offs = 0, []
        for k in (2, 3):
            n_off, n_near, o, _sigma, _m = rule_from_first_principles(j, geoms, k)
            assert n_off == 0, (k, n_off)
            n_rule += n_near
            offs.append(min(o, float(geoms[k].length_nm) - o))
        # the first shifted ring keeps the rule against ring 0 only: its
        # draws are Uniform(m, L - m) whatever ring 0 is
        assert 0 < int(ref[:, 1].sum()) < 1999
        N = np.asarray(j.null_n_matched)
        for r in range(0, 1999, 25):
            Tr, per = helper(geoms, tau, U[r], ref[r])
            assert int(Tr) == int(j.T_null[r]) and np.array_equal(np.asarray(per), N[r]), r
        assert j.z_A > 3.0 and j.p_excess <= 0.05, (j.z_A, j.p_excess)
        return (f"4 copies rings, 1999 replicates: 0 draws within m of the re-creation offset with equal reflection (rings 2 "
                f"and 3; origins {offs[0]:.0f} and {offs[1]:.0f} nm apart, sigma +1; {n_rule} within 2 m, the band just outside "
                f"the rule); T_null reproduced from the recorded draws (every 25th); z_A {j.z_A:.2f}, p_excess {j.p_excess:.4f}")

    def origin_offset_case():
        # Re-review of 2026-09-23 (probe 5, case iii): a copies axon whose
        # third ring lacks the second ring's origin cluster, so the tour
        # origins differ by one spacing and the reflect/reflect phase of
        # pair (2, 3) is re-created at U_3 - U_2 = 2 o + L_2 (mod L_3), a
        # window the first rule (|U_3 - U_2| < m) did not cover: 6 of 483
        # reflect/reflect replicates counted >= 85 % of the observed 39
        # pairs at seed 1 (1 of 492 at seed 0) before the fix, against a
        # null mean of 19.5 (sd ~3: the 85 % threshold is 4.5 sd away, so a chance
        # re-creation is ruled out). Asserted: no replicate with equal
        # reflections reaches 85 % of the observed count, the recorded draws
        # satisfy the rule recomputed here, T_null reproduces, z_A > 3.
        # Then the same axon with the third ring's tour run the other way
        # round (a legitimate RingGeometry: reversed order and contour, the
        # same curve): sigma = -1, and the re-creation offsets become
        # U_3 + U_2 = 0 (unreflected) and U_3 + U_2 = 2 o - L_2 (reflected),
        # neither of which the first rule covered either.
        axon_joint_null, helper, from_centroids, RingGeometry = require(
            "axon_joint_null", "joint_shift_statistic", "ring_geometry_from_centroids", "RingGeometry")
        rng = np.random.default_rng(52)
        a = ring_m1(rng, 0)
        b = ring_m5(rng, 1, a, 1.0)
        c = ring_m5(rng, 2, a, 1.0)
        g1, g2 = geometry_of(a), geometry_of(b)
        keep = np.ones(40, dtype=bool)
        keep[int(np.asarray(g2.order)[0])] = False
        g3 = from_centroids(2, c.xy_nm[keep])
        order_r = np.asarray(g3.order)[::-1].copy()
        g3r = RingGeometry(index=2, centroids_nm=np.asarray(g3.centroids_nm).copy(), contour_nm=np.asarray(g3.centroids_nm)[order_r],
                           order=order_r, usable=np.ones(39, dtype=bool), labels=np.arange(39, dtype=np.int64))
        assert signed_area_nm2(g3r.contour_nm) * signed_area_nm2(g3.contour_nm) < 0
        out = []
        for name, third in (("origin one cluster off", g3), ("same, tour reversed", g3r)):
            geoms = [g1, g2, third]
            j = axon_joint_null(geoms, tau, n_null=1999, random_seed=1)
            U, ref, N = np.asarray(j.null_shift_nm, dtype=float), np.asarray(j.null_reflect), np.asarray(j.null_n_matched)
            n_obs = int(np.asarray(j.n_matched_obs)[1])
            assert n_obs >= 35, n_obs
            same = ref[:, 1] == ref[:, 2]
            n_high = int(np.sum(same & (N[:, 1] >= 0.85 * n_obs)))
            n_high_any = int(np.sum(N[:, 1] >= 0.85 * n_obs))
            n_off, n_near, o, sigma, m = rule_from_first_principles(j, geoms, 2)
            assert n_off == 0, n_off
            assert n_high == 0, (name, n_high)
            assert n_high_any == 0, (name, n_high_any)
            # the origins really differ: 2 o lies outside the first rule's window (|2 o| > m, i.e. o > m / 2)
            assert min(o, float(third.length_nm) - o) > 0.5 * m, (o, m)
            for r in range(0, 1999, 50):
                Tr, per = helper(geoms, tau, U[r], ref[r])
                assert int(Tr) == int(j.T_null[r]) and np.array_equal(np.asarray(per), N[r]), r
            assert j.z_A > 3.0 and j.p_excess <= 0.05, (j.z_A, j.p_excess)
            out.append(f"{name}: sigma {sigma:+d}, o {o:.0f} nm (m {m:.0f}), pair (2, 3) observed {n_obs} of 39, {int(same.sum())} "
                       f"equal-reflection replicates, 0 with >= 85 % of it (null mean {N[:, 1].mean():.1f}); {n_near} within 2 m; "
                       f"z_A {j.z_A:.2f}")
        return "; ".join(out)

    def fpr_m1():
        axon_joint_null = require("axon_joint_null")
        rng = np.random.default_rng(92)
        t0 = time.perf_counter()
        p2 = np.empty(300)
        z = np.empty(300)
        ex = np.empty(300)
        for r in range(300):
            geoms = [geometry_of(ring_m1(rng, k)) for k in range(3)]
            j = axon_joint_null(geoms, tau, n_null=B, random_seed=r)
            p2[r], z[r] = j.p_two_sided, j.z_A
            ex[r] = exact_exchangeable_rates(np.asarray(j.T_null), j.T_obs)[2]
        f = frac_le(p2)
        # Upper edge 0.09: Wilson 95 % for R = 300 at 0.05 is [0.030, 0.083].
        # Lower edge 0.01 (re-review of 2026-09-23): T is a discrete sum
        # and the exact exchangeable rate of its two-sided p at B = 199 is
        # 0.034 (measured on 100 axons; printed here from these 300), whose
        # Wilson interval at R = 300 is [0.019, 0.061], so 0.02 would sit
        # on its edge. The mean z_A (printed) carries the bias detection.
        assert_band("joint p_two_sided <= 0.05", f, 0.01, 0.09)
        lo_w, hi_w = wilson_interval(float(ex.mean()), 300)
        return (f"fraction {f:.3f} (exact exchangeable rate {ex.mean():.3f}, Wilson for 300: [{lo_w:.3f}, {hi_w:.3f}]; for 0.05: "
                f"[0.030, 0.083]); mean z_A {np.nanmean(z):+.3f}; 300 axons in {elapsed(t0)}")

    def copies():
        axon_joint_null = require("axon_joint_null")
        rng = np.random.default_rng(93)
        zs, ps = [], []
        for r in range(5):
            a = ring_m1(rng, 0)
            geoms = [geometry_of(a), geometry_of(ring_m5(rng, 1, a, 1.0)), geometry_of(ring_m5(rng, 2, a, 1.0))]
            j = axon_joint_null(geoms, tau, n_null=B, random_seed=r)
            zs.append(j.z_A); ps.append(j.p_excess)
        assert min(zs) > 3.0 and max(ps) <= 0.05, (zs, ps)
        return f"5 axons: z_A {np.round(zs, 2).tolist()}, p_excess <= {max(ps):.3f}"

    def gap_and_determinism():
        axon_joint_null, from_centroids = require("axon_joint_null", "ring_geometry_from_centroids")
        geoms = need(st, "geoms")
        g3 = from_centroids(3, np.asarray(geoms[2].centroids_nm))
        j = axon_joint_null([geoms[0], geoms[1], g3], tau, n_null=B, random_seed=0)
        assert [tuple(p) for p in j.pairs] == [(0, 1)], j.pairs
        assert any(re.search(r"gap|consecutive|adjacent|missing", w, re.I) for w in j.warnings), j.warnings
        assert j.T_obs == int(np.asarray(j.n_matched_obs)[0])
        a = axon_joint_null(geoms, tau, n_null=B, random_seed=0)
        b = axon_joint_null(geoms, tau, n_null=B, random_seed=0)
        c = axon_joint_null(geoms, tau, n_null=B, random_seed=1)
        assert np.array_equal(np.asarray(a.T_null), np.asarray(b.T_null)) and a.p_two_sided == b.p_two_sided
        assert not np.array_equal(np.asarray(a.T_null), np.asarray(c.T_null)) and a.T_obs == c.T_obs
        return f"indices 0, 1, 3: pairs {j.pairs}, warning '{[w for w in j.warnings if re.search(r'gap|consecutive|adjacent|missing', w, re.I)][0][:60]}'; seed 0 twice identical, seed 1 differs"

    check("axon_joint_null on three M1 rings: pairs (0,1),(1,2); T_obs == sum of the adjacent n_matched of eclipse_test; "
          "null_n_matched (B, 2) rows sum to T_null; T_A, z_A, p recomputed (1e-12)", bookkeeping)
    check("the first ring is never shifted: recorded draws (column 0 zero), joint_shift_statistic reproduces T_obs with "
          "zero shifts, refuses a first-ring shift, reproduces every T_null and equals arc_shift + match_rings by hand",
          first_ring_fixed)
    check("relative-shift exclusion (D-24d): no replicate has a ring within m of the re-creation offset of the previous pair "
          "with the same reflection (window from first principles: origins and tour sense; 4 rings, n_null 1999); T_null "
          "reproduced from the recorded draws; z_A > 3", relative_shift_rule)
    check("D-24d with distinct tour origins (re-review case iii): third ring without the second ring's origin cluster "
          "(reconstruct_perimeter tour) and the same with its tour reversed (sigma = -1): no equal-reflection replicate "
          "re-creates >= 85 % of the pair's count (6 of 483 did before), rule satisfied, T_null reproduced, z_A > 3",
          origin_offset_case)
    check("joint null under M1 (R = 300, n_null 199): fraction of p_two_sided <= 0.05 in [0.01, 0.09] (lower edge: exact "
          "exchangeable rate 0.034 of the discrete T, Wilson [0.019, 0.061])", fpr_m1)
    check("rings 2 and 3 copies of ring 1 (f = 1): z_A > 3 and p_excess <= 0.05", copies)
    check("ring indices 0, 1, 3: only the pair (0, 1) is used, with a warning; determinism with the seed",
          gap_and_determinism)


# ============================================================ 10. columns
def test_columns() -> None:
    print("\n10. build_columns on hand-built matches over 4 rings")
    st: Dict[str, Any] = {}
    tau = 60.0
    # arcs per ring; cluster index = rank of the arc within the ring
    ARCS = ([0.0, 3000.0, 4500.0], [0.0, 1000.0, 3000.0, 5500.0], [0.0, 1000.0, 2000.0, 4000.0], [0.0, 2000.0, 6000.0])
    EXPECTED = [   # (members, length, touches_first, touches_last), sorted by (start ring, cluster index)
        ([(0, 0), (1, 0), (2, 0), (3, 0)], 4, True, True),
        ([(0, 1), (1, 2)], 2, True, False),
        ([(0, 2)], 1, True, False),
        ([(1, 1), (2, 1)], 2, False, False),
        ([(1, 3)], 1, False, False),
        ([(2, 2), (3, 1)], 2, False, True),
        ([(2, 3)], 1, False, False),
        ([(3, 2)], 1, False, True),
    ]

    def known_chains():
        build_columns, eclipse_test, from_centroids = require("build_columns", "eclipse_test", "ring_geometry_from_centroids")
        rings = [ring_at_arcs(k, arcs) for k, arcs in enumerate(ARCS)]
        geoms = [from_centroids(k, r.xy_nm) for k, r in enumerate(rings)]
        matches = [eclipse_test(geoms[k], geoms[k + 1], tau, n_null=19, random_seed=k) for k in range(3)]
        assert [m.n_matched for m in matches] == [2, 2, 2], [m.n_matched for m in matches]
        cols = build_columns(matches, [0, 1, 2, 3])
        got = [([tuple(int(v) for v in mb) for mb in c.members], int(c.length), bool(c.touches_first), bool(c.touches_last))
               for c in cols]
        assert got == EXPECTED, "\n".join(f"{g}\n{e}" for g, e in zip(got, EXPECTED))
        for c in cols:
            assert len(c.labels) == len(c.members) and c.length == len(c.members)
            assert [m[0] for m in c.members] == sorted(m[0] for m in c.members)
        all_members = [tuple(int(v) for v in mb) for c in cols for mb in c.members]
        assert len(all_members) == len(set(all_members)) == 14
        assert set(all_members) == {(k, i) for k, arcs in enumerate(ARCS) for i in range(len(arcs))}
        starts = [(c.members[0][0], c.members[0][1]) for c in cols]
        assert starts == sorted(starts)
        st["matches"], st["geoms"] = matches, geoms
        return f"{len(cols)} columns, lengths {[c.length for c in cols]}, 14 members = 14 clusters, sorted by (start ring, cluster)"

    def inconsistent_input_raises():
        # Nothing is dropped silently (review of 2026-09-23): before the
        # fix, matches (0,1),(1,2),(2,3) with ring_indices [0, 1, 2] gave 7
        # members of 9 clusters and a match with i_a beyond the ring's
        # clusters lost that chain's tail.
        build_columns = require("build_columns")
        matches, geoms = need(st, "matches"), need(st, "geoms")
        cases = [
            ("match naming a ring outside ring_indices", matches, [0, 1, 2], None, r"outside ring_indices"),
            ("ring nothing describes", matches[:2], [0, 1, 2, 3], None, r"no adjacent match and no geometry"),
            ("cluster index beyond the ring's clusters",
             [dataclasses.replace(matches[0], i_a=np.array([5, 1], dtype=np.intp))] + matches[1:], [0, 1, 2, 3], None,
             r"not a usable cluster"),
            ("matched cluster that the mask excludes",
             [dataclasses.replace(matches[0], usable_a=np.array([False, True, True]))] + matches[1:], [0, 1, 2, 3], None,
             r"not a usable cluster|disagree"),
        ]
        seen = []
        for what, ms, idx, geos, pattern in cases:
            try:
                build_columns(ms, idx, ring_geometries=geos)
            except ValueError as exc:
                assert re.search(pattern, str(exc)), (what, str(exc))
                seen.append(what)
            else:
                raise AssertionError(f"{what}: accepted")
        # consistent input with geometries for a ring no match reaches: singletons
        cols = build_columns(matches[:2], [0, 1, 2, 3], ring_geometries=geoms)
        members = [tuple(int(v) for v in mb) for c in cols for mb in c.members]
        assert len(members) == len(set(members)) == 14 and sum(1 for c in cols if c.members[0][0] == 3) == 3
        k2_only = build_columns([matches[0], dataclasses.replace(matches[1], ring_a=0, ring_b=2)], [0, 1], ring_geometries=geoms)
        assert sum(c.length for c in k2_only) == 7, [c.members for c in k2_only]
        return f"{len(seen)} ValueErrors ({'; '.join(seen)}); geometries cover a ring no match reaches; a k+2 match is ignored"

    check("hand-built matches over 4 rings: exact members, lengths and touches flags of 8 columns; every cluster in exactly "
          "one column; total members == total clusters; sorted by (start ring, cluster index)", known_chains)
    check("build_columns raises ValueError on a match outside ring_indices, a ring nothing describes, a cluster index that "
          "is not usable in its ring; geometries and k+2 matches handled", inconsistent_input_raises)


# ============================================================ 11. integration
def test_integration() -> None:
    print("\n11. Integration: build_rings on SMALL (validate_rings_h1) then analyze_columns with the pre-registered params")
    st: Dict[str, Any] = {}
    B = 199

    def snapshot(res: Any) -> List[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        return [(np.array([cl.centroid_nm for cl in r.clusters], dtype=float), np.asarray(r.contour_nm, dtype=float).copy())
                for r in res.rings]

    def runs():
        analyze_columns = require("analyze_columns")
        t0 = time.perf_counter()
        res = h1.run_build_rings(h1.SMALL)
        t_build = time.perf_counter() - t0
        params = load_columns_params(PARAMS_YAML)
        st["res"], st["params"], st["before"] = res, params, snapshot(res)
        t0 = time.perf_counter()
        out = analyze_columns(res, params, n_null=B)
        t_cols = time.perf_counter() - t0
        st["out"] = out
        assert list(out.rings) == [0, 1, 2], out.rings
        assert len(out.adjacent) == 2 and [(m.ring_a, m.ring_b) for m in out.adjacent] == [(0, 1), (1, 2)]
        assert len(out.k2) == 1 and (out.k2[0].ring_a, out.k2[0].ring_b) == (0, 2)
        assert len(out.curves) == 2 and [(c.ring_a, c.ring_b) for c in out.curves] == [(0, 1), (1, 2)]
        assert set(out.sensitivity) == set(params.tau_sensitivity_nm), (list(out.sensitivity), params.tau_sensitivity_nm)
        assert all(len(v) == 2 and all(m.tau_nm == t for m in v) for t, v in out.sensitivity.items())
        assert out.joint is not None and len(out.columns) > 0
        return (f"build_rings {t_build:.1f} s ({len(res.rings)} rings, {sum(len(r.clusters) for r in res.rings)} clusters); "
                f"analyze_columns {t_cols:.1f} s: 2 adjacent, 1 k+2, 2 curves, sensitivity {sorted(round(t, 2) for t in out.sensitivity)}, "
                f"joint, {len(out.columns)} columns")

    def ring_geometry_from_ring():
        ring_geometry = require("ring_geometry")
        res = need(st, "res")
        n = 0
        for ring in res.rings:
            g = ring_geometry(ring)
            per = ring.analysis.perimeter
            assert g.index == ring.index
            assert np.array_equal(np.asarray(g.contour_nm), np.asarray(ring.contour_nm))
            assert np.array_equal(np.asarray(g.order), np.asarray(per.order))
            cents = np.array([cl.centroid_nm for cl in ring.clusters], dtype=float)
            assert np.array_equal(np.asarray(g.centroids_nm), cents)
            assert np.abs(np.asarray(g.contour_nm) - cents[np.asarray(g.order)]).max() <= 0.01
            assert abs(g.polygon_length_nm - ring.length_nm) < 1e-6
            arc = np.asarray(g.arc_nm)
            assert arc[g.order[0]] == 0.0 and np.all(np.diff(arc[np.asarray(g.order)]) > 0)
            assert np.array_equal(np.asarray(g.labels), [cl.label for cl in ring.clusters])
            assert np.all(g.usable)
            assert np.array_equal(np.asarray(g.suspect), [bool(cl.suspect) for cl in ring.clusters])
            assert g.length_nm >= g.polygon_length_nm and np.array_equal(
                np.asarray(g.path.points_nm)[np.asarray(g.path.vertex_row)], np.asarray(g.contour_nm))
            assert np.allclose(arc[np.asarray(g.order)], np.asarray(g.path.cum_nm)[np.asarray(g.path.vertex_row)])
            g2 = ring_geometry(ring, include_suspect=False)
            assert np.array_equal(np.asarray(g2.usable), [not cl.suspect for cl in ring.clusters])
            n += 1
        ring = res.rings[0]
        cl0 = ring.clusters[0]
        bad = dataclasses.replace(ring, clusters=[dataclasses.replace(cl0, centroid_nm=np.asarray(cl0.centroid_nm) + 0.1)]
                                  + list(ring.clusters[1:]))
        for broken, what in ((bad, "perturbed centroid"), (dataclasses.replace(ring, contour_nm=None), "no contour")):
            try:
                ring_geometry(broken)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{what} accepted")
        return f"{n} rings: contour/order/centroids/length/arc/labels/usable/suspect/path consistent; perturbed centroid and missing contour raise"

    def small_columns():
        out, params = need(st, "out"), need(st, "params")
        # SMALL's three rings draw their lattice offset independently
        # (make_synthetic_axon), and with seed 0 the three offsets agree
        # within 7 nm (H2 fingerprint: cluster 0/1/1 at y 95.4, 90.1, 88.9
        # nm), far below tau_0 = 80.58 nm: every cluster has its partner.
        # z_A is NOT asserted here: on the regular 24-gon the null lands
        # within tau_0 of the lattice 90 % of the time (see make_columns_axon).
        ed = [m.E_dir for m in out.adjacent] + [out.k2[0].E_dir]
        assert all(v >= 0.9 for v in ed), ed
        assert all(m.tau_nm == params.tau0_nm for m in out.adjacent + out.k2)
        return (f"E_dir adjacent {[round(v, 3) for v in ed[:2]]}, k+2 {ed[2]:.3f} at tau0 {params.tau0_nm:.2f} nm; "
                f"E* {[round(m.E_star, 3) for m in out.adjacent]} (regular lattice); z_A {out.z_A:.2f} printed, not asserted")

    def irregular_columns():
        analyze_columns = require("analyze_columns")
        params = need(st, "params")
        axon = make_columns_axon(3)
        t0 = time.perf_counter()
        res = h1.run_build_rings(axon)
        t_build = time.perf_counter() - t0
        assert len(res.rings) == 3, [(r.z_lo_nm, r.z_hi_nm) for r in res.rings]
        kept = [len(r.clusters) for r in res.rings]
        assert min(kept) >= 0.9 * K_RING, kept
        out = analyze_columns(res, params, n_null=B)
        ed = [m.E_dir for m in out.adjacent] + [m.E_dir for m in out.k2]
        assert len(ed) == 3 and all(v >= 0.9 for v in ed), ed
        assert out.z_A > 3.0 and out.p_A <= 0.05, (out.z_A, out.p_A)
        assert all(m.zeta > 3.0 for m in out.adjacent), [m.zeta for m in out.adjacent]
        n3 = sum(1 for c in out.columns if c.length == 3)
        assert n3 >= 0.8 * min(kept), (n3, kept)
        return (f"build_rings {t_build:.1f} s, {kept} clusters kept; E_dir adjacent {[round(v, 3) for v in ed[:2]]}, k+2 {ed[2]:.3f}; "
                f"E* {[round(m.E_star, 3) for m in out.adjacent]}; zeta {[round(m.zeta, 1) for m in out.adjacent]}; "
                f"T_A {out.T_A:.3f}, z_A {out.z_A:.2f}, p_A {out.p_A:.4f}; {n3} columns of length 3 of {len(out.columns)}")

    def without_suspect():
        analyze_columns = require("analyze_columns")
        res, params, out = need(st, "res"), need(st, "params"), need(st, "out")
        out2 = analyze_columns(res, params, include_suspect=False, n_null=B)
        st["out2"] = out2
        assert bool(out2.include_suspect) is False and bool(out.include_suspect) is True
        n_marked = sum(1 for r in res.rings for cl in r.clusters if cl.suspect)
        for m1, m2 in zip(out.adjacent, out2.adjacent):
            assert m2.K_a <= m1.K_a and m2.K_b <= m1.K_b and bool(m2.include_suspect) is False
        used = sum(m.K_a for m in out2.adjacent) + out2.adjacent[-1].K_b
        total = sum(len(r.clusters) for r in res.rings)
        assert used == total - n_marked, (used, total, n_marked)
        members = {tuple(int(v) for v in mb) for c in out2.columns for mb in c.members}
        expected = {(r.index, i) for r in res.rings for i, cl in enumerate(r.clusters) if not cl.suspect}
        assert members == expected, (len(members), len(expected))
        return f"{n_marked} marked clusters excluded: K per ring {[m.K_a for m in out2.adjacent] + [out2.adjacent[-1].K_b]} of {total}; columns cover the usable ones"

    def determinism():
        analyze_columns = require("analyze_columns")
        res, params, out = need(st, "res"), need(st, "params"), need(st, "out")
        again = analyze_columns(res, params, n_null=B)
        for m1, m2 in zip(out.adjacent + out.k2, again.adjacent + again.k2):
            assert np.array_equal(np.asarray(m1.null_n_matched), np.asarray(m2.null_n_matched))
            assert m1.p_two_sided == m2.p_two_sided and m1.zeta == m2.zeta
        for c1, c2 in zip(out.curves, again.curves):
            assert np.array_equal(np.asarray(c1.E_star), np.asarray(c2.E_star)) and c1.p_global == c2.p_global
        assert np.array_equal(np.asarray(out.joint.T_null), np.asarray(again.joint.T_null))
        assert out.T_A == again.T_A and out.z_A == again.z_A and out.p_A == again.p_A
        return "two runs: identical null arrays, p-values, curves, T_null, T_A/z_A/p_A"

    def fields_consistent():
        res, params, out = need(st, "res"), need(st, "params"), need(st, "out")
        assert out.tau0_nm == params.tau0_nm, (out.tau0_nm, params.tau0_nm)
        assert out.params is params or out.params == params
        assert out.n_null == B and out.random_seed == params.random_seed
        assert abs(out.T_A - out.joint.T_A) < 1e-12 and out.z_A == out.joint.z_A and out.p_A == out.joint.p_two_sided
        assert out.joint.tau_nm == params.tau0_nm and out.joint.n_null == B
        for c in out.curves:
            assert np.allclose(np.asarray(c.tau_grid_nm), params.tau_grid_nm) and c.n_null == B
        assert all(m.n_null == B and m.random_seed == params.random_seed for m in out.adjacent + out.k2)
        assert out.leak is None and out.source_name == res.source_name
        assert all(isinstance(w, str) for w in out.warnings)
        after = snapshot(res)
        for (c0, k0), (c1, k1) in zip(need(st, "before"), after):
            assert np.array_equal(c0, c1) and np.array_equal(k0, k1), "analyze_columns changed the RingsResult"
        return f"tau0/grid/sensitivity/seed from params; T_A/z_A/p_A from joint; n_null {B}; leak None; {len(out.warnings)} warnings, all str; input unchanged"

    def columns_cover_clusters():
        res, out = need(st, "res"), need(st, "out")
        members = [tuple(int(v) for v in mb) for c in out.columns for mb in c.members]
        assert len(members) == len(set(members)), "a cluster in two columns"
        expected = {(r.index, i) for r in res.rings for i in range(len(r.clusters))}
        assert set(members) == expected, (len(members), len(expected))
        by_index = {r.index: r for r in res.rings}
        for c in out.columns:
            assert list(c.labels) == [by_index[k].clusters[i].label for k, i in c.members], (c.members, c.labels)
            assert c.length == len(c.members) and c.touches_first == (c.members[0][0] == 0)
            assert c.touches_last == (c.members[-1][0] == 2)
        lengths = np.array([c.length for c in out.columns])
        return f"{len(out.columns)} columns cover {len(members)} clusters once; lengths: {dict(zip(*np.unique(lengths, return_counts=True)))}; labels == DBSCAN labels"

    def few_rings():
        analyze_columns = require("analyze_columns")
        res, params = need(st, "res"), need(st, "params")
        one = analyze_columns(dataclasses.replace(res, rings=res.rings[:1]), params, n_null=B)
        assert one.adjacent == [] and one.k2 == [] and one.curves == [] and one.joint is None
        assert any("ring" in w.lower() for w in one.warnings), one.warnings
        two = analyze_columns(dataclasses.replace(res, rings=res.rings[:2]), params, n_null=B)
        assert len(two.adjacent) == 1 and two.k2 == [] and len(two.curves) == 1
        assert two.joint is not None and [tuple(p) for p in two.joint.pairs] == [(0, 1)]
        return f"1 ring: empty result, warning '{one.warnings[0][:60]}'; 2 rings: 1 adjacent, k2 [], 1 curve, joint over one pair"

    check("build_rings on SMALL then analyze_columns(res, params, n_null=199): 3 rings -> 2 adjacent, 1 k+2, 2 curves, "
          "sensitivity for the 3 tau values, joint, columns", runs)
    check("ring_geometry(Ring): contour/order from analysis.perimeter, centroids from clusters (0.01 nm), arc, labels, usable "
          "per include_suspect; ValueError on a perturbed centroid and without contour", ring_geometry_from_ring)
    check("perfect columns of SMALL: E_dir >= 0.9 at tau0 for both adjacent pairs and k+2 (z_A printed: regular lattice)",
          small_columns)
    check("perfect columns on an irregular axon (hard-core positions shared by 3 rings): 3 rings, E_dir >= 0.9 adjacent and "
          "k+2, zeta > 3 per pair, z_A > 3, p_A <= 0.05, >= 80 % of the clusters in columns of length 3", irregular_columns)
    check("include_suspect=False runs and uses <= clusters; the columns cover exactly the usable clusters", without_suspect)
    check("determinism: two runs of analyze_columns identical", determinism)
    check("result fields consistent: tau0/grid/sensitivity/seed from params (nothing hard-coded), T_A/z_A/p_A from joint, "
          "n_null, leak None, warnings are strings; the input RingsResult is unchanged", fields_consistent)
    check("columns cover every cluster exactly once; labels are the DBSCAN labels; length and censoring flags", columns_cover_clusters)
    check("fewer than 2 rings -> empty result with a warning; 2 rings -> k2 == [] and one pair", few_rings)


# ============================================================ 12. regression
def test_regression() -> None:
    print("\n12. validate_rings_h1.py and validate_columns_h2.py in subprocesses (--with-regression)")

    def run(script: str, expected: Tuple[int, int]) -> str:
        proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, script)], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=3600)
        tail = [ln for ln in proc.stdout.splitlines() if re.search(r"\d+ passed, \d+ failed", ln)]
        assert tail, proc.stdout[-2000:] + proc.stderr[-2000:]
        m = re.search(r"(\d+) passed, (\d+) failed", tail[-1])
        assert m is not None and (int(m.group(1)), int(m.group(2))) == expected, tail[-1]
        return tail[-1]

    check("validate_rings_h1.py: 70 passed, 0 failed (unchanged)", lambda: run("validate_rings_h1.py", (70, 0)))
    check("validate_columns_h2.py: 49 passed, 0 failed (unchanged)", lambda: run("validate_columns_h2.py", (49, 0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--with-regression", action="store_true",
                        help="also run validate_rings_h1.py (70) and validate_columns_h2.py (49) in subprocesses (~15 min)")
    args = parser.parse_args()
    print("=" * 72)
    print("COLUMNS H3 CHECKS: tools/mps_matching.py (matching, arc-shift null, curve, joint null, columns)")
    print("=" * 72)
    sections = (test_truth_and_api, test_match_rings, test_arc_geometry, test_eclipse_test_bookkeeping,
                test_fpr_m1, test_implanted_columns, test_alternation, test_delta_bar, test_curve,
                test_joint_null, test_columns, test_integration)
    timings: List[Tuple[str, float]] = []
    for fn in sections:
        t0 = time.perf_counter()
        fn()
        timings.append((fn.__name__, time.perf_counter() - t0))
    if args.with_regression:
        test_regression()
    else:
        print("\n12. H1/H2 harnesses not re-run (pass --with-regression; ~15 min)")
    print("\n" + "=" * 72)
    print("timings: " + ", ".join(f"{n[5:]} {t:.0f} s" for n, t in timings) + f"; total {time.perf_counter() - T_START:.0f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
