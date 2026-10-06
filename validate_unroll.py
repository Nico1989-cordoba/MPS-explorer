# -*- coding: utf-8 -*-
"""
Checks for tools/mps_unroll.py (H4, Module B: the unrolled cylinder and
the circular cross pair-correlation between rings) against truth the
harness builds itself (the research notes 02_investigacion.md
B5 / P9, 03_plan.md S3.2 ``unroll`` and ``cross_pcf_circular``; the H4
specification, "Harness B"; DECISIONES D-04, D-11, D-24c, D-27).

Written BEFORE the module (tests first): every check that needs the
module fails with "tools.mps_unroll not importable" until it lands,
and never with a NameError further down (``require``). The truth of
every check is computed here from first principles -- the exact wrapped
Gaussian kernel summed over every pair, the ellipse's analytic normal,
the explicit leave-one-out studentisation, the Fourier phase of a pure
shift -- never re-derived from the module under test; the H3 harness
generators (``validate_columns_h3``: hard-core rings on the 1500 x 1000
nm ellipse, M5/M6/rotated rings, ``make_columns_axon``) and the H1
runner (``validate_rings_h1.run_build_rings``) supply the synthetic
rings, and ``tools.mps_matching`` (H3, validated by its own harness)
supplies the smooth path of a ring. Where a tolerance appears, the
argument for it is next to the check; the numbers quoted as "measured"
come from the probe run of 2026-09-23 on a first-principles
implementation of the estimator (nearest-bin histogram at 5 nm, FFT
convolution with the kernel sampled on the grid), seed 0 unless said
otherwise.

Fast by design: no ``build_rings`` except the one integration axon of
sections 6-7 (``make_columns_axon(0)``, 2400 localizations), so that
the whole run stays under ~3 minutes; every section prints its time.

Deviations/decisions (points where the specification left a detail
open or stated an expectation the truth does not bear out; each with
the number that decided it):

* Reflection symmetry (section 2). The specification expects that with
  s_b = (-s_a) mod L the estimate is symmetric in the lag, |g(D) -
  g(-D)| < 1e-9. It is not: the pair differences are then -(s_i + s_j),
  whose multiset is closed under negation only for a ring symmetric
  about the origin (measured max asymmetry 0.72 on a hard-core ring).
  The two exact symmetries of the estimator are asserted instead, at
  1e-9: the auto pair s_b = s_a (differences s_j - s_i, closed under
  i <-> j) is symmetric in D, and reflecting BOTH rings mirrors the
  curve, g_{-a,-b}(D) = g_{a,b}(-D) (measured 1.3e-15 and 1.0e-15) --
  the second is the property that makes the reflection a valid move of
  the null. The asymmetry of the specification's case is printed.
* M6 alternation (section 2). On a hard-core ring (gaps 60 + ~Exp(140)
  nm) the midpoints sit at half-gaps 30 + ~Exp(70) nm from their
  neighbours, so the alternation peak is at the smoothed mode of that
  distribution (measured |peak lag| 55-90 nm over 20 rings, g(0)
  0.53-0.88), not at p_bar / 2 = 99 nm, and g(0) is never below 0.5.
  The specification's expectations (g(0) < 0.5, peak at p_bar / 2 +/-
  20 %) hold exactly on a regular lattice (measured g(0) 0.021, peak
  100 nm) and are asserted there. On ``ring_m6`` a first version
  asserted g(0) < 1 and |peak| in [30, 150] nm, whose 150 nm had no
  argument beyond enclosing the measurements (review 2026-09-23); the
  check now compares the module with the harness's own truth -- the
  exact kernel sum on the same configuration (g(0) within the binning
  error, the truth value at the module's peak within that error of
  the truth's maximum) -- and places THIS ring's exact g(0) and peak
  inside the range the exact estimator takes over 100 fresh M6 rings
  drawn by the harness (a Monte Carlo of the truth, not of the
  module), printing that range against the specification's band.
* Global test (section 3; review 2026-09-23, D-27). The specification's
  ``min_shift_fraction`` 0.5 (the D-11 exclusion of the H3 count test)
  makes the pcf's global test anti-conservative: a rigid shift
  translates the curve, so the excluded replicates are the ones most
  alike to the observed curve and the leave-one-out studentisation
  scores the observed against a set more alike among itself (measured
  p_global <= 0.05 in 0.065 over 2800 M1 pairs at K = 40, 0.178 at
  K_b = 4; 0.0475 and 0.055 without the exclusion). The module's
  default is now 0 (Haar shifts: the B + 1 curves are exchangeable
  and the level is exact), the FPR check asserts the specification's
  band on that default, and the draws check asserts both m = 0 on the
  default and m = 0.5 L / K_b with U in [m, L - m] when 0.5 is passed.
* Power floor (section 4). A first version asserted the measured
  power against "measured - 2 MC sd", a floor computed from the
  measured value itself, which no value can fail. The floor is 0.95
  now: with h = 30 nm and 40 coincident pairs the spike at lag 0 is
  2.5 against a null whose pointwise sd is ~0.25 (t ~ 6-10; measured
  power 1.000 at R = 200, whose Wilson lower bound is 0.98).
* M5 arcs (section 4). ``ring_m5`` jitters the copies laterally (10 nm
  per axis) and keeps the parents' arcs as truth; the unrolled
  coordinate of a jittered copy is the arc of its nearest point on the
  membrane, s_parent + (jitter . tangent) to first order (the curvature
  term is j^2 / (2 R) <= 0.15 nm on R >= 670 nm), which is what the
  harness feeds the estimator (section 0 checks it against the nearest
  point on the 40000-chord ellipse table, < 0.2 nm). The expected g(0)
  of the specification's argument, L/K^2 x K / (sqrt(2 pi) h) = 2.66,
  ignores that jitter; with the 10 nm tangential jitter the exact
  expectation of the coincident pairs' contribution is L/K /
  (sqrt(2 pi) sqrt(h^2 + 10^2)) = 2.52 (measured spike 2.52, g(0) mean
  2.59, min 2.54 over 20 rings), and the floor is 0.8 x 2.52 = 2.02,
  the "20 % slack" of the specification on the exact value.
* Lag grid (section 2). On L = 8000 nm with a 5 nm step the grid
  (-L/2, L/2] is 5 k for k = -799..800 and is asserted exactly; when
  the step does not divide L (the ellipse, 7932.7 nm) a circular
  convolution needs T = round(L / step) bins of L / T each, so on such
  a length the harness asserts only the properties that are
  convention-free: 0 is a grid point, the lags are sorted with a
  constant spacing within 1 % of the step, and they cover (-L/2, L/2]
  within one step.
* Column spread (section 6; changed at the implementation stage, 2026-
  09-23). The specification expects the s of a column's three clusters
  "within 15 nm of each other"; the first version of the check asserted
  max cyclic spread <= 15 nm and failed at 16.69 nm while the module's
  s equalled the brute-force projection on the table (0.0 nm) and the
  same centroids projected on the TRUE ellipse spread 16.78 nm: the
  spread is the centroids' own (per-axis sd 4.38 nm from 4 fluorophores
  x 5 frames; the statistic is the max over 40 groups of the range of 3,
  MC median 16.9 nm, 99.9 % 27.4 nm), so the bound, not the module, was
  wrong. The check now asserts the module's s against the brute-force
  projection (1e-6), the per-group spread against the true-ellipse
  spread of the same centroids (3 nm), the median spread <= 12 nm and
  the max <= 30 nm (the argument with the numbers is next to it).
* Per-localization order (section 6). ``loc_*`` arrays are asserted in
  ring order (``RingsResult.rings`` order) and, within a ring, in
  ``Ring.loc_index`` order, the order the specification's arrow "Ring.
  loc_index -> res.x_p/y_p/z_p" reads naturally.
* Null seeds (section 3). ``cross_pcf_circular`` receives
  ``random_seed = r`` for the r-th M1 pair, so the 300 pairs draw
  distinct nulls; the recorded draws of the first pair rebuild its
  null, and two calls with the same seed are asserted identical.
* ``n_null`` of ``analyze_unroll`` (section 7) is ``UnrollParams.n_null``
  (the specification puts it in the params, not in the call), passed
  as 199 here; ``random_seed`` must be ``cols.random_seed``.
* The public aliases of ``tools.mps_matching`` are asserted to BE the
  private helpers (``is``), not copies, so that the module reuses the
  H3 code verbatim (D-24c). The toy comparison of
  ``leave_one_out_max_deviation`` with the explicit computation is
  relative at 1e-9, not 1e-12: the helper's closed form (row sums and
  sums of squares) cancels digits on a row where the other curves
  nearly coincide (measured 3e-12 at t = 14.8 on the toy), the same
  tolerance the H3 harness applies to that helper.

Run:  venv\\Scripts\\python.exe validate_unroll.py          (from the repo root)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import ast
import dataclasses
import math
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

import validate_columns_h3 as h3  # noqa: E402  hard-core rings, ellipse, make_columns_axon
import validate_rings_h1 as h1  # noqa: E402  run_build_rings, default_params
from tools.mps_columns import load_columns_params  # noqa: E402
from tools.mps_matching import analyze_columns, smooth_closed_path  # noqa: E402

PARAMS_YAML = h3.PARAMS_YAML

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
    """The H4 names of ``tools.mps_unroll``, or a RuntimeError naming
    what is missing, so that a check fails with the reason and never
    with a NameError further down (the module does not exist until the
    implementation lands; this harness is written first)."""
    try:
        import tools.mps_unroll as mu
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"tools.mps_unroll not importable ({type(exc).__name__}: {exc}); H4 module B not implemented")
    missing = [n for n in names if not hasattr(mu, n)]
    if missing:
        raise RuntimeError(f"tools.mps_unroll has no {', '.join(missing)} (H4 module B not implemented)")
    got = tuple(getattr(mu, n) for n in names)
    return got[0] if len(got) == 1 else got


def require_matching(*names: str) -> Any:
    """The public aliases H4 adds to ``tools.mps_matching``."""
    import tools.mps_matching as mm
    missing = [n for n in names if not hasattr(mm, n)]
    if missing:
        raise RuntimeError(f"tools.mps_matching has no public alias {', '.join(missing)} (H4 not implemented)")
    got = tuple(getattr(mm, n) for n in names)
    return got[0] if len(got) == 1 else got


def need(state: Dict[str, Any], key: str) -> Any:
    """The result an earlier check of the section produced, or a clear error."""
    if key not in state:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return state[key]


def elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.1f} s"


# ============================================================ first principles
K_RING = 40
L_CIRCLE_NM = 8000.0          # the round circle of section 2 (grid asserted exactly)
STEP_NM = 5.0                 # UnrollParams.lag_step_nm default
BW_FRACTION = 0.15            # UnrollParams.bandwidth_fraction default
NULL_B = 199


def cyclic(d: NDArray[np.float64], length: float) -> NDArray[np.float64]:
    """Cyclic distance on a circle of ``length``."""
    d = np.mod(np.asarray(d, dtype=float), length)
    return np.minimum(d, length - d)


def wrap_lag(d: NDArray[np.float64], length: float) -> NDArray[np.float64]:
    """``d`` reduced to (-L/2, L/2]."""
    w = np.mod(np.asarray(d, dtype=float) + length / 2.0, length) - length / 2.0
    return np.where(w == -length / 2.0, length / 2.0, w)


def wrapped_gaussian(d: NDArray[np.float64], h: float, length: float) -> NDArray[np.float64]:
    """The Gaussian kernel of sd ``h`` wrapped on the circle of length
    ``length`` (images m = -2..2; the next ones are exp(-(2 L / h)^2 / 2)
    = 0 for any h below L / 10), a density on the circle."""
    w = wrap_lag(d, length)
    out = np.zeros_like(w)
    for m in (-2, -1, 0, 1, 2):
        out += np.exp(-0.5 * ((w + m * length) / h) ** 2)
    return out / (math.sqrt(2.0 * math.pi) * h)


def g_exact(s_a: NDArray[np.float64], s_b: NDArray[np.float64], length: float, h: float,
            lags: NDArray[np.float64]) -> NDArray[np.float64]:
    """The estimator of B5 with the exact kernel, no binning: g(D) =
    L / (K_a K_b) sum_ij k_h(wrap(s_j - s_i - D)) at each lag of
    ``lags`` (the module's grid, so the two are compared point by
    point). Vectorised over (pairs, lags); ~50 ms at K = 40 and 1600
    lags."""
    d = (np.asarray(s_b, float)[None, :] - np.asarray(s_a, float)[:, None]).ravel()
    out = np.empty(lags.size)
    block = 400
    for start in range(0, lags.size, block):
        chunk = lags[start:start + block]
        out[start:start + block] = wrapped_gaussian(d[:, None] - chunk[None, :], h, length).sum(axis=0)
    return length / (s_a.size * s_b.size) * out


def lag_grid_truth(length: float, step: float) -> NDArray[np.float64]:
    """The grid (-L/2, L/2] with step ``step`` when the step divides L:
    T = L / step lags, k step for k = -(T/2 - 1) .. T/2."""
    T = int(round(length / step))
    assert abs(T * step - length) < 1e-9, (length, step)
    return (np.arange(T) - (T // 2 - 1)) * step


def phase_truth(s_a: NDArray[np.float64], s_b: NDArray[np.float64], length: float,
                m: int) -> Tuple[float, float]:
    """(phase_deg in (-180, 180], phase_lag_nm) of harmonic m: F_x(m) =
    sum exp(-2 pi i m s / L), dphi = arg(F_a conj(F_b)), lag = dphi /
    (2 pi) x L / m. For s_b = s_a + delta, F_b = F_a exp(-2 pi i m
    delta / L) and the lag is exactly delta (when |m delta| < L / 2)."""
    fa = np.exp(-2j * np.pi * m * np.asarray(s_a, float) / length).sum()
    fb = np.exp(-2j * np.pi * m * np.asarray(s_b, float) / length).sum()
    dphi = float(np.angle(fa * np.conj(fb)))
    if dphi <= -np.pi:
        dphi += 2.0 * np.pi
    return math.degrees(dphi), dphi / (2.0 * np.pi) * length / m


def loo_max_explicit(curves: NDArray[np.float64]) -> NDArray[np.float64]:
    """Leave-one-out max studentised deviation (D-24c) done the slow,
    explicit way: for each column, the mean and sample sd (ddof 1) of
    the OTHER columns row by row, the max over rows of |x - mean| / sd;
    inf where sd is 0 and the deviation is not, 0 where both are."""
    T, n = curves.shape
    out = np.empty(n)
    for c in range(n):
        others = np.delete(curves, c, axis=1)
        mu = others.mean(axis=1)
        sd = others.std(axis=1, ddof=1)
        dev = np.abs(curves[:, c] - mu)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(sd > 0.0, dev / sd, np.where(dev > 0.0, np.inf, 0.0))
        out[c] = float(t.max())
    return out


def ellipse_outward_normal(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit outward normal of the 1500 x 1000 nm ellipse at the
    parametric angle of each point (gradient of x^2/a^2 + y^2/b^2)."""
    a, b = h3.ELLIPSE_A_NM, h3.ELLIPSE_B_NM
    t = np.arctan2(p[:, 1] / b, p[:, 0] / a)
    n = np.column_stack([np.cos(t) / a, np.sin(t) / b])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def ellipse_tangent(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit tangent of the ellipse at arc ``s`` (central difference over
    1 nm on the 40000-chord table; direction of increasing s)."""
    d = h3.ellipse_point_nm(s + 0.5) - h3.ellipse_point_nm(s - 0.5)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def m5_arcs(ring: Any) -> NDArray[np.float64]:
    """Unrolled arcs of a jittered M5 ring: the parent's arc plus the
    tangential component of the jitter vector (see the docstring)."""
    jit = ring.xy_nm - h3.ellipse_point_nm(ring.s_nm)
    return np.mod(ring.s_nm + np.einsum("ij,ij->i", jit, ellipse_tangent(ring.s_nm)), h3.L_ELLIPSE_NM)


def grid_sane(lags: NDArray[np.float64], length: float, step: float) -> str:
    """Convention-free properties of a lag grid on a length the step
    need not divide (see the docstring)."""
    lags = np.asarray(lags, float)
    T = lags.size
    assert abs(T - round(length / step)) <= 1, (T, length / step)
    assert np.all(np.diff(lags) > 0), "lags not sorted"
    spacing = np.diff(lags)
    assert np.all(np.abs(spacing - step) <= 0.01 * step), (spacing.min(), spacing.max())
    assert np.any(lags == 0.0), "0 is not a grid point"
    assert lags.min() >= -length / 2.0 - step and lags.max() <= length / 2.0 + step, (lags.min(), lags.max())
    return f"T {T}, spacing {spacing.min():.4f}-{spacing.max():.4f} nm, range [{lags.min():.1f}, {lags.max():.1f}]"


def pcf_digest(p: Any) -> Tuple[Any, ...]:
    """A hashable digest of a CrossPcf (for determinism)."""
    arrays = ("lags_nm", "g_obs", "g_null_mean", "env_lo", "env_hi", "null_shift_nm", "null_reflect")
    scalars = ("ring_a", "ring_b", "reference_ring", "length_nm", "K_a", "K_b", "p_bar_nm", "lag_max_nm",
               "t_max_obs", "p_global", "g_at_zero", "peak_lag_nm", "g_peak", "harmonic", "phase_deg",
               "phase_lag_nm", "bandwidth_nm", "lag_step_nm", "n_null", "random_seed", "min_shift_nm")
    return (tuple(np.asarray(getattr(p, a)).tobytes() for a in arrays)
            + tuple(getattr(p, s) for s in scalars) + (tuple(p.warnings),))


def default_unroll_params(**kw: Any) -> Any:
    UnrollParams = require("UnrollParams")
    return UnrollParams(**kw)


def pcf_of(s_a: NDArray[np.float64], s_b: NDArray[np.float64], length: float, *, seed: int = 0,
           n_null: int = 19, ring_a: int = 0, ring_b: int = 1, K_bar: int = K_RING) -> Any:
    """``cross_pcf_circular`` with the harness's conventions: p_bar =
    L / K_bar (both rings on the same reference), default params but
    ``n_null``."""
    cross_pcf_circular = require("cross_pcf_circular")
    return cross_pcf_circular(np.asarray(s_a, float), np.asarray(s_b, float), float(length),
                              p_bar_nm=float(length) / K_bar, params=default_unroll_params(n_null=n_null),
                              random_seed=seed, ring_a=ring_a, ring_b=ring_b, reference_ring=ring_a)


# ============================================================ 0. API and truth
def test_api_and_truth() -> None:
    print("\n0. THE H4 MODULE B API, the public aliases of tools.mps_matching, and the harness's own truth")

    def api_present():
        require("UnrollParams", "UnrolledAxon", "CrossPcf", "UnrollResult", "project_on_path", "unroll",
                "circular_cross_pcf", "cross_pcf_circular", "analyze_unroll")
        import tools.mps_unroll as mu
        with open(mu.__file__, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source)
        top = [alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names]
        top += [(node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)]
        banned = [m for m in top if m in ("PyQt5", "PyQt6", "PySide2", "PySide6", "matplotlib", "qtpy")]
        assert not banned, banned
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        undocumented = [n for n, node in funcs.items() if ast.get_docstring(node) is None]
        assert not undocumented, undocumented
        head = "\n".join(source.splitlines()[:60])
        assert "@author: Nicolas (ngomez) + Claude" in head, "author header missing"
        assert "from __future__ import annotations" in source
        p = default_unroll_params()
        # min_shift_fraction 0.0, not the specification's 0.5: the exclusion breaks the exchangeability of the
        # pcf's global test (module docstring; section 3 measures the level with and without it).
        expected = dict(bandwidth_fraction=0.15, bandwidth_nm=None, lag_step_nm=5.0, lag_max_fraction=1.0,
                        n_null=1999, min_shift_fraction=0.0, include_locs=True)
        got = {k: getattr(p, k) for k in expected}
        assert got == expected, got
        assert dataclasses.is_dataclass(p)
        return (f"9 names importable; no Qt/matplotlib import; {len(funcs)} functions documented; UnrollParams defaults as "
                f"specified except min_shift_fraction 0.0 (D-27: no exclusion on the pcf null)")

    def aliases_are_the_helpers():
        import tools.mps_matching as mm
        loo, nearest = require_matching("leave_one_out_max_deviation", "nearest_arc_on_path")
        assert loo is mm._leave_one_out_max_deviation, "leave_one_out_max_deviation is not the private helper"
        assert nearest is mm._nearest_arc_on_path, "nearest_arc_on_path is not the private helper"
        return "leave_one_out_max_deviation is _leave_one_out_max_deviation; nearest_arc_on_path is _nearest_arc_on_path"

    def harness_truth():
        # (a) the exact estimator integrates to 1 over the circle (E[g] = 1 under independence is
        #     the normalisation the module must state and this harness relies on)
        rng = np.random.default_rng(100)
        s_a = rng.uniform(0.0, L_CIRCLE_NM, K_RING)
        s_b = rng.uniform(0.0, L_CIRCLE_NM, K_RING)
        lags = lag_grid_truth(L_CIRCLE_NM, STEP_NM)
        h = BW_FRACTION * L_CIRCLE_NM / K_RING
        g = g_exact(s_a, s_b, L_CIRCLE_NM, h, lags)
        mean_g = float(g.mean())
        assert abs(mean_g - 1.0) < 1e-6, mean_g          # the kernel is a density on the circle
        # (b) the phase of a pure shift is the shift, exactly
        for delta in (30.0, 60.0, -45.0):
            deg, lag = phase_truth(s_a, np.mod(s_a + delta, L_CIRCLE_NM), L_CIRCLE_NM, K_RING)
            assert abs(lag - delta) < 1e-9, (delta, lag)
        # (c) explicit leave-one-out on a toy against a hand computation
        toy = np.array([[1.0, 2.0, 4.0], [0.5, 0.1, 0.9], [3.0, 3.0, 2.0]])
        t = loo_max_explicit(toy)
        # column 0 against columns 1, 2: rows (2, 4): mean 3, sd sqrt(2); (0.1, 0.9): 0.5, 0.5657; (3, 2): 2.5, 0.7071
        by_hand = max(abs(1.0 - 3.0) / math.sqrt(2.0), abs(0.5 - 0.5) / (0.8 / math.sqrt(2.0)),
                      abs(3.0 - 2.5) / (1.0 / math.sqrt(2.0)))
        assert abs(t[0] - by_hand) < 1e-12, (t[0], by_hand)
        # (d) the M5 arc rule against the nearest point on the 40000-chord ellipse table
        rng = np.random.default_rng(101)
        a = h3.ring_m1(rng, 0, K=K_RING)
        b = h3.ring_m5(rng, 1, a, 1.0)
        _d, s_table = h3.nearest_on_polygon(b.xy_nm, h3._ELL_XY[:-1])
        gap = float(cyclic(m5_arcs(b) - s_table, h3.L_ELLIPSE_NM).max())
        assert gap < 0.2, gap
        return (f"exact g integrates to {mean_g:.8f}; phase of a shift exact; explicit LOO = by hand ({by_hand:.4f}); "
                f"M5 tangential arc vs ellipse table {gap:.3f} nm")

    check("tools.mps_unroll importable: UnrollParams/UnrolledAxon/CrossPcf/UnrollResult/project_on_path/unroll/"
          "circular_cross_pcf/cross_pcf_circular/analyze_unroll; no Qt/matplotlib; docstrings; author header; defaults",
          api_present)
    check("tools.mps_matching public aliases leave_one_out_max_deviation and nearest_arc_on_path ARE the private helpers",
          aliases_are_the_helpers)
    check("harness truth: exact kernel estimator integrates to 1 (1e-6); phase of a pure shift = shift (1e-9); "
          "explicit LOO = by hand (1e-12); M5 tangential arc = nearest ellipse point (< 0.2 nm)", harness_truth)


# ============================================================ 1. projection
def test_projection() -> None:
    print("\n1. project_on_path on the smooth curve of an M1 ring (ellipse 1500 x 1000 nm, K = 40)")
    st: Dict[str, Any] = {}
    t0 = time.perf_counter()

    def build():
        require("project_on_path")
        rng = np.random.default_rng(1)
        ring = h3.ring_m1(rng, 0, K=K_RING)
        geom = h3.geometry_of(ring)                       # counter-clockwise polar tour
        area = h3.signed_area_nm2(np.asarray(geom.contour_nm))
        assert area > 0.0, area
        from tools.mps_matching import RingGeometry
        rev = RingGeometry(index=1, centroids_nm=np.asarray(geom.centroids_nm).copy(),
                           contour_nm=np.asarray(geom.contour_nm)[::-1].copy(),
                           order=np.asarray(geom.order)[::-1].copy(),
                           usable=np.ones(K_RING, dtype=bool), labels=np.arange(K_RING, dtype=np.int64))
        area_rev = h3.signed_area_nm2(np.asarray(rev.contour_nm))
        assert area_rev < 0.0, area_rev
        st["geom"], st["rev"] = geom, rev
        st["sign"], st["sign_rev"] = -math.copysign(1.0, area), -math.copysign(1.0, area_rev)
        return (f"CCW tour area {area:.3e} nm^2 -> outward_sign {st['sign']:+.0f}; CW tour area {area_rev:.3e} "
                f"-> {st['sign_rev']:+.0f}; table {geom.path.points_nm.shape[0]} rows, L {geom.length_nm:.2f} nm")

    def on_table():
        project_on_path = require("project_on_path")
        geom = need(st, "geom")
        path = geom.path
        rows = np.arange(0, path.points_nm.shape[0], 7)
        s, r = project_on_path(path, path.points_nm[rows], outward_sign=st["sign"])
        s, r = np.asarray(s, float), np.asarray(r, float)
        assert s.shape == (rows.size,) and r.shape == (rows.size,)
        ds = float(cyclic(s - path.cum_nm[rows], path.length_nm).max())
        assert ds < 1e-9, ds
        assert float(np.abs(r).max()) < 1e-9, np.abs(r).max()
        assert np.all(s >= 0.0) and np.all(s < path.length_nm)
        return f"{rows.size} table rows: max |s - cum| {ds:.2e} nm, max |r| {np.abs(r).max():.2e} nm"

    def displaced(sign_nm: float, key: str) -> Callable[[], str]:
        def fn():
            project_on_path = require("project_on_path")
            geom = need(st, key)
            path = geom.path
            rows = np.arange(3, path.points_nm.shape[0], 11)
            p = path.points_nm[rows]
            q = p + sign_nm * ellipse_outward_normal(p)
            s, r = project_on_path(path, q, outward_sign=st["sign" if key == "geom" else "sign_rev"])
            s, r = np.asarray(s, float), np.asarray(r, float)
            ds = float(cyclic(s - path.cum_nm[rows], path.length_nm).max())
            # The spline is within 4 nm of the ellipse on these rings (H3 harness), so its normal differs
            # from the ellipse's by an angle theta of a few hundredths: r = 20 cos(theta) is 20 to < 0.1 nm
            # and s moves by 20 sin(theta) <~ 1.5 nm; the table itself has 2 nm steps.
            assert np.all(np.abs(r - sign_nm) <= 1.0), (r.min(), r.max())
            assert ds <= 2.0, ds
            return (f"{rows.size} points at {sign_nm:+.0f} nm along the ellipse normal: r in [{r.min():.3f}, "
                    f"{r.max():.3f}], max |s - s_curve| {ds:.3f} nm")
        return fn

    def vertices():
        project_on_path = require("project_on_path")
        geom = need(st, "geom")
        s, r = project_on_path(geom.path, np.asarray(geom.centroids_nm), outward_sign=st["sign"])
        ds = float(cyclic(np.asarray(s) - np.asarray(geom.arc_nm), geom.length_nm).max())
        assert ds < 1e-9 and float(np.abs(r).max()) < 1e-9, (ds, np.abs(r).max())
        return f"the 40 vertices: s = RingGeometry.arc_nm to {ds:.1e} nm, |r| < {np.abs(r).max():.1e}"

    check("build: CCW geometry (geometry_of(ring_m1)) and the same contour reversed (CW); outward_sign = -sign(area)",
          build)
    check("points ON the dense table: s = table cum (1e-9), |r| < 1e-9", on_table)
    check("+20 nm along the ellipse's analytic outward normal: r = +20 +/- 1 nm, s within 2 nm of the curve point (CCW)",
          displaced(20.0, "geom"))
    check("-20 nm (inward): r = -20 +/- 1 nm, s within 2 nm (CCW)", displaced(-20.0, "geom"))
    check("+20 nm outward on the CLOCKWISE tour: r = +20 +/- 1 nm (outward is + in both senses)",
          displaced(20.0, "rev"))
    check("-20 nm inward on the clockwise tour: r = -20 +/- 1 nm", displaced(-20.0, "rev"))
    check("the cluster centroids (vertices): s = RingGeometry.arc_nm (1e-9), r = 0", vertices)
    print(f"   [{elapsed(t0)}]")


# ============================================================ 2. the estimator
def test_estimator() -> None:
    print("\n2. circular_cross_pcf: grid, normalisation under independence, pure shift, symmetries, alternation")
    st: Dict[str, Any] = {}
    t0 = time.perf_counter()
    h_circle = BW_FRACTION * L_CIRCLE_NM / K_RING            # 30 nm

    def grid():
        circular_cross_pcf = require("circular_cross_pcf")
        rng = np.random.default_rng(2)
        lags, g = circular_cross_pcf(rng.uniform(0, L_CIRCLE_NM, K_RING), rng.uniform(0, L_CIRCLE_NM, K_RING),
                                     L_CIRCLE_NM, bandwidth_nm=h_circle, lag_step_nm=STEP_NM)
        lags, g = np.asarray(lags, float), np.asarray(g, float)
        truth = lag_grid_truth(L_CIRCLE_NM, STEP_NM)
        assert lags.shape == truth.shape == g.shape, (lags.shape, truth.shape, g.shape)
        assert np.abs(lags - truth).max() < 1e-9, (lags[:3], lags[-3:])
        assert np.all(np.isfinite(g)) and np.all(g >= 0.0)
        return f"T {lags.size} lags = 5 k, k = -799..800 ({lags[0]:.0f} .. {lags[-1]:.0f}); g finite and >= 0"

    def independence():
        circular_cross_pcf = require("circular_cross_pcf")
        rng = np.random.default_rng(2)
        R = 200
        grid_means, g0, diff = [], [], []
        for r in range(R):
            s_a = rng.uniform(0.0, L_CIRCLE_NM, K_RING)
            s_b = rng.uniform(0.0, L_CIRCLE_NM, K_RING)
            lags, g = circular_cross_pcf(s_a, s_b, L_CIRCLE_NM, bandwidth_nm=h_circle, lag_step_nm=STEP_NM)
            lags, g = np.asarray(lags, float), np.asarray(g, float)
            grid_means.append(float(g.mean()))
            g0.append(float(g[lags == 0.0][0]))
            if r < 20:
                diff.append(float(np.abs(g - g_exact(s_a, s_b, L_CIRCLE_NM, h_circle, lags)).max()))
        mean_grid, mean_g0 = float(np.mean(grid_means)), float(np.mean(g0))
        # The kernel is a density on the circle, so every replicate's grid mean is 1 up to the kernel's
        # own discretisation; 0.02 leaves room for a kernel normalised analytically rather than on the
        # grid. E[g(0)] = 1 under independence; the per-replicate sd of g(0) is 0.25 (measured), the se
        # of the mean of 200 is 0.018 and 0.1 is 5.5 se. The binned estimator differs from the exact
        # kernel by a first-order term per pair of at most (step / 2) / h = 8 % of k, of random sign
        # over the ~12 pairs within h of a lag: measured max 0.03 over 50 replicates; 0.1 is 3x that.
        assert abs(mean_grid - 1.0) <= 0.02, mean_grid
        assert abs(mean_g0 - 1.0) <= 0.1, mean_g0
        assert max(diff) <= 0.1, max(diff)
        return (f"R {R}: mean grid-average of g {mean_grid:.5f}; mean g(0) {mean_g0:.3f} (sd {np.std(g0):.3f}); "
                f"max |g - exact kernel| over 20 replicates {max(diff):.4f}")

    def pure_shift():
        rng = np.random.default_rng(21)
        a = h3.ring_m1(rng, 0, K=K_RING)
        L = h3.L_ELLIPSE_NM
        s_b = np.mod(a.s_nm + 30.0, L)
        p = pcf_of(a.s_nm, s_b, L, seed=0)
        st["shift_pcf"] = p
        lags, g = np.asarray(p.lags_nm, float), np.asarray(p.g_obs, float)
        sane = grid_sane(lags, L, STEP_NM)
        sel = np.abs(lags) <= p.lag_max_nm
        peak_truth = float(lags[sel][np.argmax(g[sel])])
        assert abs(p.peak_lag_nm - peak_truth) < 1e-9 and abs(p.g_peak - g[sel].max()) < 1e-12
        assert abs(p.peak_lag_nm - 30.0) <= STEP_NM + 1e-9, p.peak_lag_nm
        assert p.harmonic == max(1, round(L / p.p_bar_nm)) == K_RING, p.harmonic
        deg, lag = phase_truth(a.s_nm, s_b, L, p.harmonic)
        assert abs(p.phase_lag_nm - 30.0) <= 0.5, p.phase_lag_nm
        assert abs(p.phase_lag_nm - lag) < 1e-9 and abs(math.remainder(p.phase_deg - deg, 360.0)) < 1e-9
        assert -180.0 < p.phase_deg <= 180.0
        exact = g_exact(a.s_nm, s_b, L, p.bandwidth_nm, lags)
        dmax = float(np.abs(g - exact).max())
        assert dmax <= 0.1, dmax
        assert abs(p.g_at_zero - g[lags == 0.0][0]) < 1e-12
        return (f"grid {sane}; peak at {p.peak_lag_nm:.0f} nm (g_peak {p.g_peak:.3f}, g(0) {p.g_at_zero:.3f}); "
                f"harmonic {p.harmonic}, phase {p.phase_deg:.2f} deg = {p.phase_lag_nm:.4f} nm; "
                f"max |g - exact| {dmax:.4f}")

    def symmetries():
        circular_cross_pcf = require("circular_cross_pcf")
        rng = np.random.default_rng(22)
        a = h3.ring_m1(rng, 0, K=K_RING)
        b = h3.ring_m1(rng, 1, K=K_RING)
        L = h3.L_ELLIPSE_NM
        h = BW_FRACTION * L / K_RING

        def mirror_gap(g1: NDArray[np.float64], g2: NDArray[np.float64], lags: NDArray[np.float64]) -> float:
            """max |g1(D) - g2(-D)| over the lags whose negative is on the grid."""
            idx = {float(v): i for i, v in enumerate(lags)}
            pairs = [(i, idx[-float(v)]) for i, v in enumerate(lags) if -float(v) in idx]
            assert len(pairs) >= lags.size - 2, len(pairs)
            return float(max(abs(g1[i] - g2[j]) for i, j in pairs))

        lags, g_auto = circular_cross_pcf(a.s_nm, a.s_nm, L, bandwidth_nm=h, lag_step_nm=STEP_NM)
        lags, g_auto = np.asarray(lags, float), np.asarray(g_auto, float)
        gap_auto = mirror_gap(g_auto, g_auto, lags)
        _, g_ab = circular_cross_pcf(a.s_nm, b.s_nm, L, bandwidth_nm=h, lag_step_nm=STEP_NM)
        _, g_mm = circular_cross_pcf(np.mod(-a.s_nm, L), np.mod(-b.s_nm, L), L, bandwidth_nm=h, lag_step_nm=STEP_NM)
        gap_both = mirror_gap(np.asarray(g_ab, float), np.asarray(g_mm, float), lags)
        _, g_ref = circular_cross_pcf(a.s_nm, np.mod(-a.s_nm, L), L, bandwidth_nm=h, lag_step_nm=STEP_NM)
        g_ref = np.asarray(g_ref, float)
        gap_spec = mirror_gap(g_ref, g_ref, lags)
        assert gap_auto < 1e-9, gap_auto
        assert gap_both < 1e-9, gap_both
        return (f"auto pair s_b = s_a: max |g(D) - g(-D)| {gap_auto:.1e}; both rings reflected: max |g_ab(D) - "
                f"g_-a-b(-D)| {gap_both:.1e}; (b alone reflected, the specification's case, is NOT symmetric: "
                f"{gap_spec:.3f}, see the docstring)")

    def alternation_lattice():
        s_a = np.arange(K_RING) * (L_CIRCLE_NM / K_RING)
        s_b = np.mod(s_a + L_CIRCLE_NM / K_RING / 2.0, L_CIRCLE_NM)
        p = pcf_of(s_a, s_b, L_CIRCLE_NM, seed=0)
        half = p.p_bar_nm / 2.0
        assert p.g_at_zero < 0.5, p.g_at_zero
        assert abs(abs(p.peak_lag_nm) - half) <= 0.2 * half, (p.peak_lag_nm, half)
        # exact: g(0) = L/K^2 x 2K k_h(100) = 400 x phi(100/30)/30 = 0.021; the peak is at +/-100 exactly
        expected0 = L_CIRCLE_NM / K_RING * 2.0 / (math.sqrt(2 * math.pi) * p.bandwidth_nm) * math.exp(-0.5 * (half / p.bandwidth_nm) ** 2)
        assert abs(p.g_at_zero - expected0) < 0.02, (p.g_at_zero, expected0)
        return f"regular lattice 200 nm, b at the midpoints: g(0) {p.g_at_zero:.4f} (exact {expected0:.4f}), |peak| {abs(p.peak_lag_nm):.0f} nm = p_bar/2 {half:.0f}"

    def alternation_hard_core():
        rng = np.random.default_rng(23)
        a = h3.ring_m1(rng, 0, K=K_RING)
        b = h3.ring_m6(rng, 1, a)
        L = h3.L_ELLIPSE_NM
        p = pcf_of(a.s_nm, b.s_nm, L, seed=0)
        half = p.p_bar_nm / 2.0
        lags = np.asarray(p.lags_nm, float)
        sel = np.abs(lags) <= p.lag_max_nm
        # (a) the module against the harness's exact kernel sum on the SAME configuration: g(0) within the
        #     binning error (measured max |g - exact| 0.03 over replicates; 0.05 asserted) and the truth's
        #     value at the module's peak lag within that error of the truth's maximum (a flat top may move
        #     the argmax by more than one step, the value cannot move by more than the binning error).
        truth = g_exact(a.s_nm, b.s_nm, L, p.bandwidth_nm, lags[sel])
        g0_truth = float(truth[lags[sel] == 0.0][0])
        peak_truth = float(lags[sel][np.argmax(truth)])
        at_module_peak = float(truth[np.argmin(np.abs(lags[sel] - p.peak_lag_nm))])
        assert abs(p.g_at_zero - g0_truth) <= 0.05, (p.g_at_zero, g0_truth)
        assert truth.max() - at_module_peak <= 0.05, (p.peak_lag_nm, peak_truth, at_module_peak, truth.max())
        # (b) where the exact estimator puts g(0) and the peak on M6 rings: 100 fresh hard-core rings and
        #     their midpoint rings, the exact kernel sum on the window (the truth's own Monte Carlo, no
        #     module involved); this ring's exact values must lie within that range. The band the
        #     specification wrote ([0.8, 1.2] x p_bar/2, g(0) < 0.5) is printed against it.
        rng_mc = np.random.default_rng(230)
        g0_mc, peak_mc = [], []
        for _ in range(100):
            a_r = h3.ring_m1(rng_mc, 0, K=K_RING)
            b_r = h3.ring_m6(rng_mc, 1, a_r)
            g_r = g_exact(a_r.s_nm, b_r.s_nm, L, p.bandwidth_nm, lags[sel])
            g0_mc.append(float(g_r[lags[sel] == 0.0][0]))
            peak_mc.append(abs(float(lags[sel][np.argmax(g_r)])))
        g0_lo, g0_hi = min(g0_mc), max(g0_mc)
        pk_lo, pk_hi = min(peak_mc), max(peak_mc)
        assert g0_lo <= g0_truth <= g0_hi, (g0_truth, g0_lo, g0_hi)
        assert pk_lo <= abs(peak_truth) <= pk_hi, (peak_truth, pk_lo, pk_hi)
        in_band = np.mean([0.8 * half <= v <= 1.2 * half for v in peak_mc])
        return (f"ring_m6 on a hard-core M1 ring: module g(0) {p.g_at_zero:.3f} = exact {g0_truth:.3f}, |peak| "
                f"{abs(p.peak_lag_nm):.0f} nm (exact argmax {abs(peak_truth):.0f}, truth value there within {truth.max() - at_module_peak:.3f}); "
                f"exact estimator over 100 M6 rings: g(0) in [{g0_lo:.2f}, {g0_hi:.2f}] (< 0.5 in {np.mean(np.asarray(g0_mc) < 0.5):.0%}), "
                f"|peak| in [{pk_lo:.0f}, {pk_hi:.0f}] nm, median {np.median(peak_mc):.0f} (the specification's band "
                f"[{0.8 * half:.0f}, {1.2 * half:.0f}] holds in {in_band:.0%}); g_peak {p.g_peak:.3f}")

    check("grid on L = 8000 nm, step 5: lags = 5 k for k = -799..800 (1e-9), (-L/2, L/2], T = 1600; g finite, >= 0", grid)
    check("K_a = K_b = 40 uniform independent, R = 200: mean grid-average of g = 1 +/- 0.02, mean g(0) = 1 +/- 0.1; "
          "max |g - exact kernel| <= 0.1 on 20 replicates", independence)
    check("pure shift s_b = s_a + 30 nm on the ellipse: peak lag 30 +/- one step, phase_lag 30 +/- 0.5 nm (exact for a "
          "shift), harmonic round(L / p_bar), fields consistent with g_obs", pure_shift)
    check("exact symmetries: auto pair symmetric in D (1e-9); reflecting both rings mirrors g (1e-9); b alone printed",
          symmetries)
    check("M6 alternation on a regular lattice: g(0) < 0.5 (exact 0.021 to 0.02) and |peak| = p_bar/2 +/- 20 %",
          alternation_lattice)
    check("M6 alternation on a hard-core ring (ring_m6): module g(0) and peak = the exact kernel sum (0.05); this ring's "
          "exact g(0) and |peak| within the exact estimator's range over 100 M6 rings; the spec's band printed",
          alternation_hard_core)
    print(f"   [{elapsed(t0)}]")


# ============================================================ 3. global test under M1
def test_fpr() -> None:
    print("\n3. cross_pcf_circular under M1 (independent hard-core rings): FPR of p_global, the null's draws, rebuild")
    st: Dict[str, Any] = {}
    t0 = time.perf_counter()
    R = 300

    def fpr():
        L = h3.L_ELLIPSE_NM
        rng = np.random.default_rng(3)
        ps, t_obs = [], []
        for r in range(R):
            a = h3.ring_m1(rng, 0, K=K_RING)
            b = h3.ring_m1(rng, 1, K=K_RING)
            p = pcf_of(a.s_nm, b.s_nm, L, seed=r, n_null=NULL_B)
            if r == 0:
                st["pcf"], st["a"], st["b"] = p, a, b
            ps.append(float(p.p_global))
            t_obs.append(float(p.t_max_obs))
        ps_arr = np.asarray(ps)
        frac = h3.frac_le(ps_arr)
        assert np.all(np.abs(ps_arr * (NULL_B + 1) - np.round(ps_arr * (NULL_B + 1))) < 1e-9), "p not k / (B + 1)"
        assert ps_arr.min() >= 1.0 / (NULL_B + 1) and ps_arr.max() <= 1.0
        # Wilson 95 % interval of 0.05 at n = 300: [0.030, 0.082]; the statistic is continuous, so there is
        # no discreteness deficit. With the module's default (no shift exclusion, Haar shifts) the B + 1
        # curves are exchangeable and the level is exact; with the specification's exclusion 0.5 the
        # measured level was 0.065-0.083 (see the docstring). The band [0.02, 0.09] is the specification's.
        lo, hi = h3.wilson_interval(0.05, R)
        h3.assert_band("FPR", frac, 0.02, 0.09)
        return (f"R {R}, B {NULL_B}: p_global <= 0.05 in {frac:.3f} (Wilson of 0.05: [{lo:.3f}, {hi:.3f}]); "
                f"<= 0.10 in {h3.frac_le(ps_arr, 0.10):.3f}; median t_max_obs {np.median(t_obs):.2f}")

    def draws():
        cross_pcf_circular = require("cross_pcf_circular")
        p, a, b = need(st, "pcf"), need(st, "a"), need(st, "b")
        L = h3.L_ELLIPSE_NM
        U = np.asarray(p.null_shift_nm, float)
        refl = np.asarray(p.null_reflect)
        assert U.shape == (NULL_B,) and refl.shape == (NULL_B,) and refl.dtype == bool
        # the default: no exclusion (m = 0, Haar shifts; docstring), U ~ Uniform(0, L)
        assert p.min_shift_nm == 0.0, p.min_shift_nm
        assert np.all(U >= 0.0) and np.all(U < L), (U.min(), U.max())
        assert 0.3 < refl.mean() < 0.7, refl.mean()          # Bernoulli(1/2): 199 draws, sd 0.035
        assert p.n_null == NULL_B and p.random_seed == 0 and p.K_a == p.K_b == K_RING
        assert abs(p.length_nm - L) < 1e-9 and abs(p.p_bar_nm - L / K_RING) < 1e-9
        assert abs(p.bandwidth_nm - BW_FRACTION * p.p_bar_nm) < 1e-9 and abs(p.lag_max_nm - p.p_bar_nm) < 1e-9
        assert p.lag_step_nm == STEP_NM and (p.ring_a, p.ring_b, p.reference_ring) == (0, 1, 0)
        assert all(isinstance(w, str) for w in p.warnings)
        # the pre-registered exclusion when asked for: m = 0.5 L / K_b, every U in [m, L - m], same g_obs
        m = 0.5 * L / K_RING
        q = cross_pcf_circular(a.s_nm, b.s_nm, L, p_bar_nm=L / K_RING,
                               params=default_unroll_params(n_null=NULL_B, min_shift_fraction=0.5),
                               random_seed=0, ring_a=0, ring_b=1, reference_ring=0)
        Uq = np.asarray(q.null_shift_nm, float)
        assert abs(q.min_shift_nm - m) < 1e-9 and np.all(Uq >= m) and np.all(Uq <= L - m), (q.min_shift_nm, Uq.min(), Uq.max())
        assert np.array_equal(np.asarray(q.g_obs), np.asarray(p.g_obs))
        return (f"default min_shift 0: U in [{U.min():.1f}, {U.max():.1f}] of [0, L); reflect in {refl.mean():.2f} of the draws; "
                f"h {p.bandwidth_nm:.2f} nm, lag_max {p.lag_max_nm:.1f} nm; with min_shift_fraction 0.5: m {q.min_shift_nm:.2f} nm "
                f"= 0.5 p_bar, U in [{Uq.min():.1f}, {Uq.max():.1f}] of [m, L - m], p_global {p.p_global:.3f} -> {q.p_global:.3f}")

    def rebuild():
        circular_cross_pcf = require("circular_cross_pcf")
        p, a, b = need(st, "pcf"), need(st, "a"), need(st, "b")
        L = h3.L_ELLIPSE_NM
        lags = np.asarray(p.lags_nm, float)
        g_obs = np.asarray(p.g_obs, float)
        lags_again, g_again = circular_cross_pcf(a.s_nm, b.s_nm, L, bandwidth_nm=p.bandwidth_nm, lag_step_nm=p.lag_step_nm)
        assert np.abs(np.asarray(lags_again, float) - lags).max() < 1e-9
        assert np.abs(np.asarray(g_again, float) - g_obs).max() < 1e-12
        null = np.empty((NULL_B, lags.size))
        for r in range(NULL_B):
            sigma = -1.0 if p.null_reflect[r] else 1.0
            s_b = np.mod(sigma * b.s_nm + p.null_shift_nm[r], L)
            null[r] = np.asarray(circular_cross_pcf(a.s_nm, s_b, L, bandwidth_nm=p.bandwidth_nm,
                                                    lag_step_nm=p.lag_step_nm)[1], float)
        d_mean = float(np.abs(null.mean(axis=0) - np.asarray(p.g_null_mean, float)).max())
        lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
        d_env = max(float(np.abs(lo - np.asarray(p.env_lo, float)).max()), float(np.abs(hi - np.asarray(p.env_hi, float)).max()))
        assert d_mean < 1e-12, d_mean
        assert d_env < 1e-9, d_env
        sel = np.abs(lags) <= p.lag_max_nm
        curves = np.concatenate([g_obs[sel][:, None], null[:, sel].T], axis=1)
        t = loo_max_explicit(curves)
        p_truth = (int(np.sum(t[1:] >= t[0])) + 1) / (NULL_B + 1)
        assert abs(t[0] - p.t_max_obs) < 1e-9, (t[0], p.t_max_obs)
        assert p_truth == p.p_global, (p_truth, p.p_global)
        peak_truth = float(lags[sel][np.argmax(g_obs[sel])])
        assert p.peak_lag_nm == peak_truth and p.g_peak == float(g_obs[sel].max())
        assert p.g_at_zero == float(g_obs[lags == 0.0][0])
        deg, lag = phase_truth(a.s_nm, b.s_nm, L, p.harmonic)
        assert abs(p.phase_lag_nm - lag) < 1e-9 and abs(math.remainder(p.phase_deg - deg, 360.0)) < 1e-9
        return (f"{NULL_B} replicates rebuilt from the draws: |mean - g_null_mean| {d_mean:.1e}, envelope {d_env:.1e}; "
                f"t_max_obs {p.t_max_obs:.3f} = explicit LOO; p_global {p.p_global:.4f} reproduced over {int(sel.sum())} lags; "
                f"peak/g_at_zero/phase reproduced")

    def alias_toy():
        loo = require_matching("leave_one_out_max_deviation")
        # no two equal values in a row: with n = 3 the "other" sd would be 0 and the deviation inf on both sides
        toy = np.array([[1.0, 2.0, 4.0], [0.5, 0.1, 0.9], [3.0, 3.1, 2.0], [1.0, 1.5, 1.2]])
        got = np.asarray(loo(toy), float)
        truth = loo_max_explicit(toy)
        # The helper's closed form (row sums and sums of squares) cancels digits where the other curves
        # nearly coincide (row 3: sd 0.07 from a sum of squares of 22): ~3e-12 at t = 14.8 on this toy, so
        # the comparison is relative at 1e-9, the tolerance the H3 harness uses for the same helper.
        gap = float((np.abs(got - truth) / np.maximum(1.0, np.abs(truth))).max())
        assert got.shape == (3,) and np.all(np.isfinite(truth)) and gap < 1e-9, (got, truth)
        rng = np.random.default_rng(31)
        big = rng.normal(size=(9, 6))
        got2 = np.asarray(loo(big), float)
        truth2 = loo_max_explicit(big)
        gap2 = float((np.abs(got2 - truth2) / np.maximum(1.0, np.abs(truth2))).max())
        assert gap2 < 1e-9, gap2
        return f"3-column toy: {np.round(truth, 6).tolist()} = explicit to {gap:.1e} (relative); 6 random curves to {gap2:.1e}"

    def alias_nearest():
        nearest = require_matching("nearest_arc_on_path")
        rng = np.random.default_rng(32)
        geom = h3.geometry_of(h3.ring_m1(rng, 0, K=K_RING))
        path = geom.path
        pts = h3.ellipse_point_nm(rng.uniform(0, h3.L_ELLIPSE_NM, 12)) + rng.normal(0.0, 15.0, (12, 2))
        _d, s_truth = h3.nearest_on_polygon(pts, path.points_nm)
        got = np.array([nearest(path, q) for q in pts])
        gap = float(cyclic(got - s_truth, path.length_nm).max())
        assert gap < 1e-9, gap
        return f"12 points near the curve: arc = harness nearest_on_polygon on the table to {gap:.1e} nm"

    def determinism():
        a, b, p = need(st, "a"), need(st, "b"), need(st, "pcf")
        L = h3.L_ELLIPSE_NM
        again = pcf_of(a.s_nm, b.s_nm, L, seed=0, n_null=NULL_B)
        assert pcf_digest(again) == pcf_digest(p), "two runs with the same seed differ"
        other = pcf_of(a.s_nm, b.s_nm, L, seed=1, n_null=NULL_B)
        assert not np.array_equal(np.asarray(other.null_shift_nm), np.asarray(p.null_shift_nm))
        assert np.array_equal(np.asarray(other.g_obs), np.asarray(p.g_obs)), "g_obs depends on the seed"
        assert other.random_seed == 1
        return f"same seed: identical digest; seed 1: other draws, same g_obs, p_global {p.p_global:.3f} -> {other.p_global:.3f}"

    check(f"FPR: R = {R} M1 pairs (K 40, ellipse), B = {NULL_B}: p_global <= 0.05 in [0.02, 0.09]; p = k / (B + 1)", fpr)
    check("the null's draws: default min_shift 0 (Haar), every U in [0, L); with min_shift_fraction 0.5: m = 0.5 L / K_b (1e-9), "
          "every U in [m, L - m]; reflect Bernoulli, fields recorded", draws)
    check("the recorded draws rebuild the null: mean (1e-12), envelope (1e-9), t_max_obs = explicit LOO (1e-9), p_global "
          "exact; peak, g_at_zero, phase reproduced", rebuild)
    check("leave_one_out_max_deviation (public alias) on a 3-column toy = explicit computation (relative 1e-9; "
          "the closed form cancels ~3e-12 at t = 15)", alias_toy)
    check("nearest_arc_on_path (public alias) = harness nearest point on the table (1e-9)", alias_nearest)
    check("determinism: same seed identical; another seed changes the draws, not g_obs", determinism)
    print(f"   [{elapsed(t0)}]")


# ============================================================ 4. power
def test_power() -> None:
    print("\n4. Power: M5 f = 1 with 10 nm lateral jitter (ring_m5), R = 200, B = 199")
    t0 = time.perf_counter()
    R = 200

    def power():
        L = h3.L_ELLIPSE_NM
        h = BW_FRACTION * L / K_RING
        # E[spike] = L/K^2 x K x E[k_h(delta)], delta ~ N(0, 10^2) tangential jitter: L/K / (sqrt(2 pi) sqrt(h^2 + 100))
        expected_g0 = (L / K_RING) / (math.sqrt(2.0 * math.pi) * math.sqrt(h * h + 10.0 ** 2))
        floor = 0.8 * expected_g0
        rng = np.random.default_rng(4)
        ps, g0, peak = [], [], []
        for r in range(R):
            a = h3.ring_m1(rng, 0, K=K_RING)
            b = h3.ring_m5(rng, 1, a, 1.0)
            p = pcf_of(a.s_nm, m5_arcs(b), L, seed=r, n_null=NULL_B)
            ps.append(float(p.p_global))
            g0.append(float(p.g_at_zero))
            peak.append(float(p.peak_lag_nm))
        ps_arr, g0_arr, peak_arr = np.asarray(ps), np.asarray(g0), np.asarray(peak)
        pw = h3.frac_le(ps_arr)
        f_g0 = float(np.mean(g0_arr >= floor))
        f_peak = float(np.mean(np.abs(peak_arr) <= STEP_NM + 1e-9))
        # Power floor 0.95 (docstring): the spike at lag 0 is E[g(0)] = 2.52 against a null whose pointwise
        # sd is ~0.25 at K = 40 (section 2 measured 0.195-0.25), so t_max_obs ~ 6-10 against null t's of
        # ~2 and p sits at 1 / (B + 1) in every replicate (measured 1.000, Wilson lower bound 0.981); the
        # first version compared the measured power with "measured - 2 MC sd", which cannot fail.
        assert pw >= 0.95, pw
        assert f_g0 >= 0.95, (f_g0, floor)
        assert f_peak >= 0.95, f_peak
        return (f"p_global <= 0.05 in {pw:.3f} (floor 0.95; Wilson lower bound of the measured value {h3.wilson_interval(pw, R)[0]:.3f}); "
                f"g(0) >= {floor:.2f} (0.8 x exact expectation {expected_g0:.3f}) in {f_g0:.3f} (g(0) min {g0_arr.min():.2f}, "
                f"mean {g0_arr.mean():.2f}); |peak lag| <= one step in {f_peak:.3f}")

    check(f"M5 f = 1, 10 nm jitter, R = {R}: p_global <= 0.05 in >= 95 % (argued floor); g(0) >= 0.8 x exact "
          "expectation in >= 95 %; peak within one lag step of 0 in >= 95 %", power)
    print(f"   [{elapsed(t0)}]")


# ============================================================ 5. torsion
def test_torsion() -> None:
    print("\n5. Torsion: s_b = s_a + 60 nm (ring_rotated)")
    t0 = time.perf_counter()

    def torsion(shift: float) -> Callable[[], str]:
        def fn():
            rng = np.random.default_rng(5)
            a = h3.ring_m1(rng, 0, K=K_RING)
            b = h3.ring_rotated(1, a, shift)
            L = h3.L_ELLIPSE_NM
            p = pcf_of(a.s_nm, b.s_nm, L, seed=0)
            assert abs(p.peak_lag_nm - shift) <= STEP_NM + 1e-9, (p.peak_lag_nm, shift)
            assert abs(p.phase_lag_nm - shift) <= 0.5, (p.phase_lag_nm, shift)
            assert math.copysign(1.0, p.peak_lag_nm) == math.copysign(1.0, shift)
            assert math.copysign(1.0, p.phase_lag_nm) == math.copysign(1.0, shift)
            return f"peak lag {p.peak_lag_nm:+.0f} nm, phase lag {p.phase_lag_nm:+.4f} nm (phase {p.phase_deg:+.2f} deg), g_peak {p.g_peak:.3f}"
        return fn

    check("s_b = s_a + 60 nm: peak_lag 60 +/- one step and phase_lag 60 +/- 0.5 nm, both with the sign +", torsion(60.0))
    check("s_b = s_a - 60 nm: peak_lag -60 +/- one step and phase_lag -60 +/- 0.5 nm (sign -)", torsion(-60.0))
    print(f"   [{elapsed(t0)}]")


# ============================================================ 6. unroll on rings
# The rings built in section 6 (the one build_rings of the harness) are handed to section 7 through here.
_RINGS_STATE: Dict[str, Any] = {}


def test_unroll_rings() -> None:
    print("\n6. unroll on measured rings: make_columns_axon(0) -> run_build_rings (tilt corrected) -> unroll(res)")
    st = _RINGS_STATE
    t0 = time.perf_counter()

    def build():
        require("unroll")
        axon = h3.make_columns_axon(0)
        t1 = time.perf_counter()
        res = h1.run_build_rings(axon)
        st["axon"], st["res"] = axon, res
        assert len(res.rings) == 3, [(r.z_lo_nm, r.z_hi_nm) for r in res.rings]
        kept = [len(r.clusters) for r in res.rings]
        assert min(kept) >= 0.9 * K_RING, kept
        assert all(r.contour_nm is not None for r in res.rings)
        st["kept"] = kept
        return f"build_rings {time.perf_counter() - t1:.1f} s: 3 rings, clusters {kept}, {axon.n_locs} localizations"

    def unroll_clusters():
        unroll = require("unroll")
        res, kept = need(st, "res"), need(st, "kept")
        t1 = time.perf_counter()
        u = unroll(res)
        st["u"] = u
        ref_truth = max(range(len(res.rings)), key=lambda q: (len(res.rings[q].clusters), -res.rings[q].index))
        ref_index = res.rings[ref_truth].index
        assert u.reference_ring == ref_index, (u.reference_ring, ref_index, kept)
        ref = res.rings[ref_truth]
        path = smooth_closed_path(np.asarray(ref.contour_nm))
        assert abs(u.length_nm - path.length_nm) < 1e-9, (u.length_nm, path.length_nm)
        ri, ci = np.asarray(u.ring_index), np.asarray(u.cluster_index)
        n = sum(kept)
        assert ri.shape == ci.shape == (n,)
        for arr, name in ((u.labels, "labels"), (u.usable, "usable"), (u.s_nm, "s"), (u.r_nm, "r"), (u.z_nm, "z")):
            assert np.asarray(arr).shape == (n,), (name, np.asarray(arr).shape)
        pairs = sorted(zip(ri.tolist(), ci.tolist()))
        assert pairs == [(r.index, i) for r in res.rings for i in range(len(r.clusters))]
        by_index = {r.index: r for r in res.rings}
        lab_truth = np.array([by_index[k].clusters[i].label for k, i in zip(ri, ci)])
        assert np.array_equal(np.asarray(u.labels), lab_truth)
        assert np.all(np.asarray(u.usable))
        z_truth = np.array([float(np.median(res.z_p[by_index[k].clusters[i].loc_index])) for k, i in zip(ri, ci)])
        assert np.abs(np.asarray(u.z_nm, float) - z_truth).max() < 1e-9
        s, r = np.asarray(u.s_nm, float), np.asarray(u.r_nm, float)
        assert np.all(s >= 0.0) and np.all(s < u.length_nm)
        r_max = float(np.abs(r).max())
        assert r_max <= 25.0, r_max
        own = ri == ref_index
        cents = np.array([by_index[k].clusters[i].centroid_nm for k, i in zip(ri[own], ci[own])], dtype=float)
        _d, s_truth = h3.nearest_on_polygon(cents, path.points_nm)
        ds = float(cyclic(s[own] - s_truth, path.length_nm).max())
        assert ds < 1e-6 and float(np.abs(r[own]).max()) <= 0.01, (ds, np.abs(r[own]).max())
        assert u.source_name == res.source_name and all(isinstance(w, str) for w in u.warnings)
        return (f"unroll {time.perf_counter() - t1:.1f} s; reference ring {u.reference_ring} (K {kept}), L {u.length_nm:.1f} nm; "
                f"{n} clusters; max |r| {r_max:.2f} nm; reference's own clusters: s = nearest table arc ({ds:.1e}), "
                f"|r| <= {np.abs(r[own]).max():.4f} nm; z = median z'")

    def columns_share_s():
        u, res, axon = need(st, "u"), need(st, "res"), need(st, "axon")
        by_index = {r.index: r for r in res.rings}
        ri, ci = np.asarray(u.ring_index), np.asarray(u.cluster_index)
        s = np.asarray(u.s_nm, float)
        cents = np.array([by_index[k].clusters[i].centroid_nm for k, i in zip(ri, ci)], dtype=float)
        # (a) the module's s IS the nearest point on the reference table for EVERY cluster of every ring
        #     (brute force on the table, the harness's own projection; measured 0.0 nm on three seeds)
        ref = by_index[u.reference_ring]
        path = smooth_closed_path(np.asarray(ref.contour_nm))
        _d, s_bf = h3.nearest_on_polygon(cents, path.points_nm)
        ds_all = float(cyclic(s - s_bf, path.length_nm).max())
        assert ds_all < 1e-6, ds_all
        groups: Dict[int, List[Tuple[float, NDArray[np.float64]]]] = {}
        for k, i, sv, c in zip(ri.tolist(), ci.tolist(), s.tolist(), cents):
            ids = axon.cluster[by_index[k].clusters[i].loc_index]
            vals, counts = np.unique(ids, return_counts=True)
            if counts.max() < 0.9 * ids.size:
                continue
            groups.setdefault(int(vals[np.argmax(counts)]) % K_RING, []).append((sv, c))
        spreads, spreads_ell = [], []
        for j, vals_j in groups.items():
            if len(vals_j) >= 2:
                arr = np.asarray([v for v, _ in vals_j])
                c = np.asarray([q for _, q in vals_j])
                spreads.append(float(cyclic(arr[:, None] - arr[None, :], u.length_nm).max()))
                _d2, s_ell = h3.nearest_on_polygon(c, h3._ELL_XY[:-1])
                spreads_ell.append(float(cyclic(s_ell[:, None] - s_ell[None, :], h3.L_ELLIPSE_NM).max()))
        assert len(spreads) >= 0.9 * K_RING, len(spreads)
        sp, sp_ell = np.asarray(spreads), np.asarray(spreads_ell)
        # (b) The spread across rings is the CENTROIDS' own, not the projection's. The first version of
        #     this check asserted max spread <= 15 nm, "3 sd", and failed at 16.69 nm with the module's s
        #     equal to the brute-force projection (0.0 nm): the bound was wrong, not the module. A
        #     centroid of 4 fluorophores x 5 frames scatters sqrt(8^2/4 + 8^2/20) = 4.38 nm per axis; the
        #     statistic is the MAX over ~40 true positions of the cyclic RANGE of 3 centroids (tangential
        #     component), not one pairwise difference: MC (20000 axons of 40 groups, seed 0) gives that
        #     max a median of 16.9 nm, 95 % 21.9, 99 % 24.3, 99.9 % 27.4 (mean range of 3: 7.4 nm;
        #     measured on seeds 0-2: 16.7, 14.3, 21.8, medians 7.7, 7.7, 5.7). So the check is now:
        #     the per-group spread of the module's s equals the spread of the SAME centroids projected
        #     on the TRUE ellipse (no spline, no module) within 3 nm (the spline is within 4 nm of the
        #     ellipse and a point at |r| <= 16 nm projects on it with an arc error of |r| x the tangent
        #     angle difference, a few hundredths: measured max 0.58, 0.70, 1.08 nm on seeds 0-2), the
        #     median spread <= 12 nm (MC 99.9 %: 9.4 nm) and the max <= 30 nm (MC 99.9 %: 27.4 nm).
        gap = float(np.abs(sp - sp_ell).max())
        assert gap <= 3.0, gap
        assert float(np.median(sp)) <= 12.0, np.median(sp)
        assert float(sp.max()) <= 30.0, sp.max()
        return (f"{len(spreads)} true positions seen in >= 2 rings: module s = brute-force table projection "
                f"({ds_all:.1e} nm, all {s.size} clusters); cyclic spread of s across rings max {sp.max():.2f} nm, "
                f"median {np.median(sp):.2f} (same centroids on the true ellipse: max {sp_ell.max():.2f}, "
                f"per-group gap <= {gap:.2f} nm)")

    def locs():
        u, res = need(st, "u"), need(st, "res")
        n = sum(r.n_locs for r in res.rings)
        for arr, name in ((u.loc_ring, "loc_ring"), (u.loc_s_nm, "loc_s_nm"), (u.loc_r_nm, "loc_r_nm"), (u.loc_z_nm, "loc_z_nm")):
            assert arr is not None and np.asarray(arr).shape == (n,), (name, None if arr is None else np.asarray(arr).shape)
        lr = np.asarray(u.loc_ring)
        assert np.array_equal(lr, np.concatenate([np.full(r.n_locs, r.index) for r in res.rings]))
        z_truth = np.concatenate([res.z_p[r.loc_index] for r in res.rings])
        assert np.array_equal(np.asarray(u.loc_z_nm, float), z_truth)
        ls, lrr = np.asarray(u.loc_s_nm, float), np.asarray(u.loc_r_nm, float)
        assert np.all(ls >= 0.0) and np.all(ls < u.length_nm)
        # localizations scatter 8 nm (spread) + 8 nm (noise) about their cluster, whose |r| <= 25: 3 sd + 25 = 59
        assert float(np.abs(lrr).max()) <= 60.0, np.abs(lrr).max()
        return f"{n} localizations = sum of ring n_locs; loc_z = res.z_p[loc_index] in ring order; |loc_r| <= {np.abs(lrr).max():.1f} nm"

    def no_locs_and_reference():
        unroll = require("unroll")
        res = need(st, "res")
        u2 = unroll(res, params=default_unroll_params(include_locs=False))
        assert u2.loc_ring is None and u2.loc_s_nm is None and u2.loc_r_nm is None and u2.loc_z_nm is None
        assert np.array_equal(np.asarray(u2.s_nm), np.asarray(need(st, "u").s_nm))
        other = [r for r in res.rings if r.index != need(st, "u").reference_ring][0]
        u3 = unroll(res, reference=other.index)
        assert u3.reference_ring == other.index
        assert abs(u3.length_nm - smooth_closed_path(np.asarray(other.contour_nm)).length_nm) < 1e-9
        own = np.asarray(u3.ring_index) == other.index
        assert float(np.abs(np.asarray(u3.r_nm)[own]).max()) <= 0.01
        u4 = unroll(res, include_suspect=False)
        by_index = {r.index: r for r in res.rings}
        usable_truth = np.array([not by_index[k].clusters[i].suspect for k, i in zip(np.asarray(u4.ring_index), np.asarray(u4.cluster_index))])
        assert np.array_equal(np.asarray(u4.usable), usable_truth)
        return (f"include_locs=False: the four loc arrays None, same s; reference={other.index}: honoured, L of ring "
                f"{other.index}; include_suspect=False: usable = not suspect ({int((~usable_truth).sum())} marked)")

    check("make_columns_axon(0) -> run_build_rings: 3 rings, >= 90 % of 40 clusters kept per ring", build)
    check("unroll(res): reference = ring with most clusters (lowest index on a tie); length = reference curve length (1e-9); "
          "one row per cluster, labels/usable/z consistent; |r| <= 25 nm; reference's clusters on the curve", unroll_clusters)
    check("columns (same true position in the 3 rings): module s = brute-force projection (1e-6); the spread of s "
          "across rings = the centroids' own spread on the true ellipse (3 nm), median <= 12 nm, max <= 30 nm",
          columns_share_s)
    check("per-localization arrays: sum of ring n_locs entries, ring order then loc_index order; |loc_r| <= 60 nm", locs)
    check("include_locs=False -> None; reference= honoured; include_suspect=False -> usable = not suspect", no_locs_and_reference)
    print(f"   [{elapsed(t0)}]")


# ============================================================ 7. analyze_unroll
def test_analyze_unroll() -> None:
    print("\n7. analyze_unroll on the same result with analyze_columns(n_null = 199)")
    st: Dict[str, Any] = {}
    t0 = time.perf_counter()

    def runs():
        analyze_unroll = require("analyze_unroll")
        res = need(_RINGS_STATE, "res")
        params = load_columns_params(PARAMS_YAML)
        t1 = time.perf_counter()
        cols = analyze_columns(res, params, n_null=NULL_B)
        t_cols = time.perf_counter() - t1
        assert len(cols.adjacent) == 2 and len(cols.k2) == 1
        t1 = time.perf_counter()
        out = analyze_unroll(res, cols, params=default_unroll_params(n_null=NULL_B))
        t_un = time.perf_counter() - t1
        st["res"], st["cols"], st["out"] = res, cols, out
        assert [(p.ring_a, p.ring_b) for p in out.pcf_adjacent] == [(m.ring_a, m.ring_b) for m in cols.adjacent]
        assert [(p.ring_a, p.ring_b) for p in out.pcf_k2] == [(m.ring_a, m.ring_b) for m in cols.k2]
        assert out.random_seed == cols.random_seed
        assert out.params.n_null == NULL_B
        UnrolledAxon, CrossPcf = require("UnrolledAxon", "CrossPcf")
        assert isinstance(out.unrolled, UnrolledAxon)
        assert all(isinstance(p, CrossPcf) for p in out.pcf_adjacent + out.pcf_k2)
        assert all(isinstance(w, str) for w in out.warnings)
        return (f"analyze_columns {t_cols:.1f} s, analyze_unroll {t_un:.1f} s: pcf for adjacent {[(p.ring_a, p.ring_b) for p in out.pcf_adjacent]} "
                f"and k+2 {[(p.ring_a, p.ring_b) for p in out.pcf_k2]}; random_seed {out.random_seed} = cols.random_seed")

    def fields():
        out, cols = need(st, "out"), need(st, "cols")
        u = out.unrolled
        L = u.length_nm
        ri, usable = np.asarray(u.ring_index), np.asarray(u.usable)
        lines = []
        for p in out.pcf_adjacent + out.pcf_k2:
            k_a = int(np.count_nonzero(usable & (ri == p.ring_a)))
            k_b = int(np.count_nonzero(usable & (ri == p.ring_b)))
            assert (p.K_a, p.K_b) == (k_a, k_b), (p.K_a, p.K_b, k_a, k_b)
            p_bar = (L / k_a + L / k_b) / 2.0
            assert abs(p.p_bar_nm - p_bar) < 1e-9 and abs(p.length_nm - L) < 1e-9
            assert abs(p.bandwidth_nm - BW_FRACTION * p_bar) < 1e-9 and abs(p.lag_max_nm - p_bar) < 1e-9
            assert p.reference_ring == u.reference_ring and p.n_null == NULL_B and p.random_seed == cols.random_seed
            assert p.lag_step_nm == STEP_NM
            grid_sane(np.asarray(p.lags_nm, float), L, STEP_NM)
            s_a = np.asarray(u.s_nm, float)[usable & (ri == p.ring_a)]
            s_b = np.asarray(u.s_nm, float)[usable & (ri == p.ring_b)]
            exact = g_exact(s_a, s_b, L, p.bandwidth_nm, np.asarray(p.lags_nm, float))
            dmax = float(np.abs(np.asarray(p.g_obs, float) - exact).max())
            assert dmax <= 0.1, dmax
            lines.append(f"{p.ring_a}-{p.ring_b}: K {k_a}/{k_b}, p_bar {p_bar:.1f}, h {p.bandwidth_nm:.1f}, |g - exact| {dmax:.3f}")
        return "; ".join(lines)

    def columns_detected():
        out = need(st, "out")
        lines = []
        for p in out.pcf_adjacent + out.pcf_k2:
            assert p.p_global <= 0.05, (p.ring_a, p.ring_b, p.p_global)
            assert abs(p.peak_lag_nm) <= STEP_NM + 1e-9, p.peak_lag_nm
            assert p.g_at_zero > 1.5, p.g_at_zero
            lines.append(f"{p.ring_a}-{p.ring_b}: g(0) {p.g_at_zero:.2f}, peak {p.peak_lag_nm:+.0f}, t {p.t_max_obs:.1f}, p {p.p_global:.4f}, phase lag {p.phase_lag_nm:+.1f}")
        return "; ".join(lines)

    def determinism():
        analyze_unroll = require("analyze_unroll")
        res, cols, out = need(st, "res"), need(st, "cols"), need(st, "out")
        again = analyze_unroll(res, cols, params=default_unroll_params(n_null=NULL_B))
        for p1, p2 in zip(out.pcf_adjacent + out.pcf_k2, again.pcf_adjacent + again.pcf_k2):
            assert pcf_digest(p1) == pcf_digest(p2), (p1.ring_a, p1.ring_b)
        assert np.array_equal(np.asarray(out.unrolled.s_nm), np.asarray(again.unrolled.s_nm))
        assert np.array_equal(np.asarray(out.unrolled.r_nm), np.asarray(again.unrolled.r_nm))
        assert list(out.warnings) == list(again.warnings)
        return "two runs: identical pcf digests, s, r and warnings"

    def without_suspect():
        analyze_unroll = require("analyze_unroll")
        res, cols = need(st, "res"), need(st, "cols")
        out2 = analyze_unroll(res, cols, params=default_unroll_params(n_null=NULL_B), include_suspect=False)
        u = out2.unrolled
        by_index = {r.index: r for r in res.rings}
        truth = np.array([not by_index[k].clusters[i].suspect for k, i in zip(np.asarray(u.ring_index), np.asarray(u.cluster_index))])
        assert np.array_equal(np.asarray(u.usable), truth)
        ri = np.asarray(u.ring_index)
        for p in out2.pcf_adjacent + out2.pcf_k2:
            assert p.K_a == int(np.count_nonzero(truth & (ri == p.ring_a)))
            assert p.K_b == int(np.count_nonzero(truth & (ri == p.ring_b)))
        return f"include_suspect=False: usable = not suspect ({int((~truth).sum())} marked), K_a/K_b count the usable clusters"

    def excluded_clusters():
        analyze_unroll = require("analyze_unroll")
        res, cols, out = need(st, "res"), need(st, "cols"), need(st, "out")
        # Two clusters of the first adjacent pair's rings are excluded (the caller's leak-explained children on
        # data): their rows turn unusable, the pair's K drops by one on each side, the other rows are untouched,
        # an unknown entry is warned about, and the run is otherwise the same (same seed, same reference).
        m = cols.adjacent[0]
        excl = [(int(m.ring_a), 0), (int(m.ring_b), 1), (99, 0)]
        out2 = analyze_unroll(res, cols, params=default_unroll_params(n_null=NULL_B), exclude_clusters=excl)
        u, u0 = out2.unrolled, out.unrolled
        ri, ci = np.asarray(u.ring_index), np.asarray(u.cluster_index)
        hit = np.array([(int(k), int(i)) in set(excl) for k, i in zip(ri, ci)])
        assert int(hit.sum()) == 2 and not np.any(np.asarray(u.usable)[hit])
        assert np.array_equal(np.asarray(u.usable)[~hit], np.asarray(u0.usable)[~hit])
        assert np.array_equal(np.asarray(u.s_nm), np.asarray(u0.s_nm)) and u.reference_ring == u0.reference_ring
        p2, p0 = out2.pcf_adjacent[0], out.pcf_adjacent[0]
        assert (p2.K_a, p2.K_b) == (p0.K_a - 1, p0.K_b - 1), (p2.K_a, p2.K_b, p0.K_a, p0.K_b)
        assert any("exclude_clusters" in w and "1 of 3" in w for w in out2.warnings), out2.warnings
        assert any("2 cluster(s) excluded" in w for w in out2.warnings), out2.warnings
        return (f"exclude_clusters {excl[:2]} + one unknown: 2 rows unusable, pair {p0.ring_a}-{p0.ring_b} K {p0.K_a}/{p0.K_b} -> "
                f"{p2.K_a}/{p2.K_b}, p_global {p0.p_global:.3f} -> {p2.p_global:.3f}; unknown entry warned")

    check("analyze_columns(n_null 199) then analyze_unroll: one CrossPcf per adjacent pair and per k+2 pair; "
          "random_seed = cols.random_seed; types; warnings strings", runs)
    check("each CrossPcf: K from the usable clusters, p_bar = (L/K_a + L/K_b)/2, h = 0.15 p_bar, lag_max = p_bar, "
          "reference ring, grid sane, g_obs = exact kernel within 0.1", fields)
    check("perfect columns: p_global <= 0.05, peak within one step of 0, g(0) > 1.5 for the 2 adjacent and the k+2 pair",
          columns_detected)
    check("determinism: two runs of analyze_unroll identical", determinism)
    check("include_suspect=False: usable = not suspect; K_a/K_b follow", without_suspect)
    check("exclude_clusters: the named rows turn unusable, K_a/K_b drop, other rows untouched, unknown entries warned "
          "(bookkeeping only: the exclusion does not undo the centroid attraction of absorbed leak, validate_leak.py section 15)",
          excluded_clusters)
    print(f"   [{elapsed(t0)}]")


def main() -> int:
    print("=" * 72)
    print("UNROLL H4 CHECKS: tools/mps_unroll.py (projection, circular cross pcf, arc-shift null, analyze_unroll)")
    print("=" * 72)
    sections = (test_api_and_truth, test_projection, test_estimator, test_fpr, test_power, test_torsion,
                test_unroll_rings, test_analyze_unroll)
    timings: List[Tuple[str, float]] = []
    for fn in sections:
        t0 = time.perf_counter()
        fn()
        timings.append((fn.__name__, time.perf_counter() - t0))
    print("\n" + "=" * 72)
    print("timings: " + ", ".join(f"{n[5:]} {t:.0f} s" for n, t in timings) + f"; total {time.perf_counter() - T_START:.0f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
