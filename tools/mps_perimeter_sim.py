"""
Simulator of localization density along an axon perimeter (package P1 of the perimeter density analysis).

It defines the TRUTH of the whole validation: every synthetic axon carries the true curve, the true relative
density lambda(s) of each slice, the true slice of every molecule and, per scenario, the known answer the
analysis has to recover. Pure numpy, no Qt, deterministic for a given seed (one ``numpy.random.Generator``,
draws in a fixed order).

Generative model
----------------
1. A closed curve (counter-clockwise seen from +z, so positive curvature = convex and the outward normal is the
   right-hand side of the tangent) tabulated at uniform arc steps (``table_step_nm``).
2. Per slice k in (-1, 0, +1) a relative density lambda_k(s) on that arc grid, normalised to mean 1.
3. Membrane molecules: a true depth z (which fixes the true slice), an arc s drawn from lambda of that slice, a
   radial offset ~ N(0, radial_sd_nm) along the outward normal.
4. Each molecule emits 1 localization, or with ``blink`` 1 + Geometric(mean ``blink_mean_extra``) localizations;
   every localization gets its own lateral error N(0, precision_nm) per axis and its own axial error N(0, lpz_nm).
5. ``interior_frac`` of the localizations sit uniformly inside the curve (interior spectrin), ``background_frac``
   uniformly in the bounding box widened by ``bg_margin_nm``; both are single localizations without blinking.
6. ``sx`` and ``sy`` (pixels) follow an astigmatic width model of the OBSERVED z whose curves cross at the focus
   z_f (``focus_nm``): sx = w0 sqrt(1 + ((z - z_f - c) / d)^2), sy the same with +c, times (1 + noise).

Localizations are returned in generation order (membrane molecule by molecule, then interior, then background);
nothing is shuffled, so ``molecule_id`` runs are contiguous.

Shapes (``shape``)
------------------
``circle``, ``ellipse``, ``wavy`` (radial harmonics 2, 3, 5, 7), ``c_shape`` (an annular sector closed by two
semicircular caps: NOT star-shaped from its centroid) and ``two_lobes`` (r = R(1 + a cos 2t), concave waist).

Intensities (``intensity["kind"]``)
-----------------------------------
``flat`` | ``cluster`` (flat + big cluster) | ``attractor`` (big cluster + Gaussian excess around it) |
``inhibition`` (big cluster + empty zone on each side) | ``regular`` (big cluster + cosine modes in phase with it,
period rounded so that it closes on the perimeter) | ``curvature`` (lambda proportional to 1 + beta * kappa * R,
R = L / 2 pi; beta = 0 is the curvature-without-effect control) | ``gap`` (flat with an empty stretch) |
``clumps`` (big cluster + small clumps at positions independent of it) | ``random`` (1 + amp * smooth Gaussian field;
the kind used for slice-correlation truths). Any kind accepts ``gap_nm`` (contour with a hole of that length).

Slices (``slices``)
-------------------
Three contiguous slices of ``thickness_nm`` centred on ``z0_nm``. Depth either ``uniform`` (fractions ``fracs``,
uniform within each slice) or ``gaussian`` (N(peak_nm, depth_sd_nm): the axial decay seen in real data; molecules
beyond the three slices are kept and labelled OUTSIDE_LABEL, their pattern is the one of the nearest slice). The
slice patterns are ``identical``, ``independent`` (every random position redrawn per slice) or ``correlated``
(``random`` kind only: field_k = sqrt(rho) F0 + sqrt(1 - rho) G_k). Axial leakage comes from the per-localization
z error lpz_nm; ``leak_matrix`` and ``expected_slice_r`` give its exact expectation for this model.

Truth emitted (``SimPerimeter.truth``; keys always present unless marked)
--------------------------------------------------------------------------
Geometry: ``shape``, ``length_nm``, ``radius_char_nm`` (L / 2 pi), ``min_radius_of_curvature_nm``,
``star_shaped_from_centroid``, ``centroid_nm``. Density: ``kind``, ``cluster_arc_nm`` (per slice, None without a
big cluster), ``gap_arcs_nm`` (per slice, list of (start, end) arc, end may exceed L when the gap wraps),
``clump_arcs_nm`` (per slice), ``period_eff_nm`` (regular), ``attractor_sd_nm`` / ``attractor_amp`` (attractor),
``zone_nm`` (inhibition), ``beta`` and ``curvature_clip_frac`` (curvature), ``rho`` (correlated),
``true_slice_lambda_r`` (3 x 3 Pearson of the true lambda_k on the arc grid). Axial: ``z0_nm``, ``thickness_nm``,
``slice_bounds_nm``, ``depth``, ``peak_nm`` (mode of the true depth; None for uniform depth), ``focus_nm``,
``peak_minus_focus_nm`` (None for uniform depth), ``source_frac`` (molecule mass whose pattern is slice -1/0/+1),
``true_slice_frac`` (molecule mass inside each slice), ``leak_matrix`` (P(observed slice j | pattern of slice k),
lpz error, no guard), ``expected_slice_r`` (3 x 3, from ``expected_slice_r``; a lower bound with blinking). Counts:
``n_membrane``, ``n_interior``, ``n_background``, ``n_molecules``.

What each question can use (P5 = C, P6 = B, P7 = A, P11 = validation):
* C: ``random`` + ``independent`` (negative; with lpz the neighbour r equals ``expected_slice_r``), ``identical``
  (positive, r -> 1), ``correlated`` with rho (dose-response); lpz_nm = 0 removes leakage.
* B: ``wavy`` + ``curvature`` with beta (positive) or beta = 0 (negative), ``true_curvature`` on the arc grid,
  ``true_path`` as the TRUE contour; a small ``radius_nm`` gives an unresolved axon.
* A: ``flat`` / ``clumps`` (negatives), ``attractor``, ``inhibition``, ``regular`` (positives) with
  ``cluster_arc_nm`` and ``period_eff_nm`` as the known answers.
* Contour (P3) and z0 (P2): every shape, ``gap`` / ``gap_nm``, ``c_shape``; ``gaussian`` depth with
  ``peak_nm`` != ``focus_nm``, and ``sx``/``sy``.

Scenario -> known truth (presets in ``SCENARIOS``; defaults: 10 000 localizations, radial sd 20 nm, lpz 47 nm,
blinking on, seed 0; numbers computed with these defaults)
-----------------------------------------------------------------------------------------------------------------
=================  =========================================  ======================================================
scenario           what it is                                 known answer (truth keys)
=================  =========================================  ======================================================
a_flat             ellipse (L = 9.8 um), flat lambda          no structure: A negative (pooled peak at 0 = artefact)
a_clumps           + big cluster + 20 clumps at random arcs   A negative with realistic autocorrelation;
                                                              ``cluster_arc_nm``, ``clump_arcs_nm``
a_attractor        big cluster + Gaussian excess, sd 600 nm,  A "attractor": contrast > 0 around ``cluster_arc_nm``
                   amplitude 0.5                              (``attractor_sd_nm``, ``attractor_amp``)
a_inhibition       big cluster + empty zone of 800 nm/side    A "inhibition zone": contrast < 0 (``zone_nm``)
a_regular          big cluster + cosine modes, period 800 nm  A "regular": period = ``period_eff_nm`` (818 nm here,
                   rounded to close on the perimeter          in phase with the big cluster)
b_null             wavy (L = 9.5 um), beta = 0                B negative: density independent of curvature
b_curvature        wavy, lambda ~ 1 + 0.5 kappa R             B positive (``beta``, ``true_curvature``); min radius
                                                              of curvature 712 nm >= 600 nm (resolved)
b_thin             circle R = 400 nm                          B unresolved (R < 600 nm)
c_identical        random field, same in the 3 slices         C positive: r -> 1
c_independent      random field, independent, lpz = 0         C negative: r = 0, no leakage
c_leak             random field, independent, lpz = 47 nm     leakage only: r(-1,0) = r(0,+1) = 0.275, r(-1,+1) =
                                                              0.020 (``expected_slice_r``; 0.122 / 0.004 with a
                                                              42.5 nm guard, ``leak_matrix(guard_nm=42.5)``);
                                                              11.0 % of a central slice leaks to each neighbour
c_correlated       random field, rho = 0.5                    r(-1,+1) = 0.57, neighbours 0.70 (rho + leakage)
z_peak_focus       Gaussian depth (peak 0, sd 130 nm),        ``peak_nm`` = 0, ``focus_nm`` = -43 (sx = sy there),
                   focus at -43 nm                            ``peak_minus_focus_nm`` = 43; laterals hold 47.6 % of
                                                              the central slice (``true_slice_frac`` 0.232 / 0.487 /
                                                              0.232); 4.9 % of molecules beyond the 3 slices
contour_c_gap      "C" (L = 13.6 um, not star-shaped) with    ``true_path``, ``gap_arcs_nm`` (opposite the start),
                   a 2 um gap                                 ``star_shaped_from_centroid`` = False
contour_two_lobes  two lobes (concave waist), 15 % bg         ``true_path``; ``source`` flags the background
=================  =========================================  ======================================================

``expected_slice_r`` is meaningful for the ``random`` kind (patterns of equal variance); for the deterministic
kinds with an identical pattern it reads 1 by construction. With blinking it is a lower bound.

Origin tags: [lit] literature, [own sim] own simulation, [real data, pilot] the 18-axon training set
(exploratory, one experiment), [ad hoc] chosen here, [derivation] follows from the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from tools.mps_matching import SmoothPath
from tools.mps_perimeter_types import OUTSIDE_LABEL, SLICE_THICKNESS_NM, SLICES

FloatArray: TypeAlias = NDArray[np.float64]

# ----------------------------------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------------------------------
SHAPES: Tuple[str, ...] = ("circle", "ellipse", "wavy", "c_shape", "two_lobes")
KINDS: Tuple[str, ...] = ("flat", "cluster", "attractor", "inhibition", "regular", "curvature", "gap", "clumps",
                          "random")
PATTERNS: Tuple[str, ...] = ("identical", "independent", "correlated")
DEPTH_MODES: Tuple[str, ...] = ("uniform", "gaussian")
SOURCE_MEMBRANE: int = 0
SOURCE_INTERIOR: int = 1
SOURCE_BACKGROUND: int = 2

# ----------------------------------------------------------------------------------------------------
# Defaults, each with its origin
# ----------------------------------------------------------------------------------------------------
# Geometry
RADIUS_NM: float = 1500.0          # [real data, pilot] L = 2 pi R = 9.4 um, inside the 6.7-32 um perimeter range
ELLIPSE_ASPECT: float = 1.6        # [ad hoc] clearly non-circular, still convex
WAVE_AMPS: Tuple[float, ...] = (0.06, 0.04, 0.02, 0.01)   # [ad hoc] harmonics 2,3,5,7; min radius of curvature
WAVE_HARMONICS: Tuple[int, ...] = (2, 3, 5, 7)            # stays >= 600 nm at R = 1500 (the resolved limit of
WAVE_PHASES: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)     # mps_curvature), checked in the tests
LOBE_AMP: float = 0.35             # [derivation] a > 0.2 makes the waist concave; 0.35 -> waist radius ~845 nm
C_INNER_FRAC: float = 0.45         # [ad hoc] inner radius of the "C" / outer radius
C_OPENING_DEG: float = 35.0        # [ad hoc] half-angle of the opening (cap centres at +-35 deg)
TABLE_STEP_NM: float = 2.0         # [ad hoc] arc step of the true curve table; << 8 nm precision
# Emitters
PRECISION_NM: float = 8.0          # [real data, pilot] median lateral precision lpx * pixel (NeNA 7.13 nm)
BLINK_MEAN_EXTRA: float = 2.0      # [ad hoc] mean extra localizations per molecule (1 + Geometric)
BG_MARGIN_NM: float = 500.0        # [ad hoc] background box margin around the curve
RADIAL_SD_NM: float = 20.0         # [own sim] radial scatter of the pilot simulation (plan 2.4)
LPZ_NM: float = 47.0               # [real data, pilot] axial precision lpz
# Big cluster, attractor, inhibition, regular modes, clumps, field
CLUSTER_FRAC: float = 0.05         # [ad hoc] fraction of the membrane density in the big cluster
CLUSTER_SD_NM: float = 40.0        # [ad hoc] larger than the ~21-25 nm effective radius of ordinary clusters [lit]
ATTRACTOR_AMP: float = 0.5         # [ad hoc] the plan's positive control for question A
ATTRACTOR_SD_NM: float = 600.0     # [ad hoc] idem
ZONE_NM: float = 800.0             # [ad hoc] idem, empty zone on each side
ZONE_FLOOR: float = 0.0            # [ad hoc]
PERIOD_NM: float = 800.0           # [ad hoc] idem, regular modes
REGULAR_AMP: float = 0.8           # [ad hoc] < 1 keeps lambda > 0
BETA: float = 0.5                  # [ad hoc] the plan's positive control for question B
GAP_NM: float = 2000.0             # [real data, pilot] the ~2 um holes that defeat the density mask
N_CLUMPS: int = 20                 # [ad hoc]
CLUMP_FRAC: float = 0.3            # [ad hoc]
CLUMP_SD_NM: float = 40.0          # [ad hoc]
FIELD_AMP: float = 0.3             # [ad hoc] clipping at 0 is a 3.3 sd event
FIELD_CORR_NM: float = 100.0       # [ad hoc] one arc bin: fluctuations live at the scale the profile resolves
# Axial
DEPTH_SD_NM: float = 130.0         # [real data, pilot] gives laterals ~47 % of the central (pilot: 36 % and 52 %)
FOCUS_NM: float = 0.0              # nominal focus (z = 0 of the file)
PSF_W0_PX: float = 1.0             # [ad hoc] in-focus PSF sd in pixels
ASTIG_C_NM: float = 400.0          # [ad hoc] focal offset of each axis
ASTIG_D_NM: float = 400.0          # [ad hoc] depth of focus
WIDTH_NOISE_FRAC: float = 0.03     # [ad hoc]

_INTENSITY_KEYS = {"kind", "cluster_frac", "cluster_sd_nm", "cluster_arc_frac", "attractor_amp", "attractor_sd_nm",
                   "zone_nm", "zone_floor", "period_nm", "regular_amp", "beta", "gap_nm", "gap_arc_frac",
                   "n_clumps", "clump_frac", "clump_sd_nm", "field_amp", "field_corr_nm"}
_SLICE_KEYS = {"thickness_nm", "z0_nm", "depth", "fracs", "peak_nm", "depth_sd_nm", "pattern", "rho", "focus_nm"}
_OPTION_KEYS = {"radius_nm", "aspect", "wave_amps", "wave_phases", "lobe_amp", "c_inner_frac", "c_opening_deg",
                "rotation_deg", "centre_nm", "table_step_nm", "precision_nm", "blink_mean_extra", "bg_margin_nm",
                "psf_w0_px", "astig_c_nm", "astig_d_nm", "width_noise_frac"}


@dataclass(frozen=True, eq=False)
class SimPerimeter:
    """Simulated axon: localizations plus their truth. Per-localization arrays share one length and order."""

    x_nm: FloatArray
    y_nm: FloatArray
    z_nm: FloatArray
    true_path: SmoothPath                      # the true curve (uniform arc table, CCW)
    true_lambda: FloatArray                    # true lambda(s) of slice 0 on ``arc_grid_nm`` (mean 1)
    true_slice: NDArray[np.int8]               # true slice of the emitting molecule: -1 / 0 / 1, OUTSIDE_LABEL beyond
    # --- fields added by P1 (all have defaults so the contract constructor still works) ---
    arc_grid_nm: FloatArray = field(default_factory=lambda: np.zeros(0))     # cell starts, step L / N
    true_lambda_by_slice: Dict[int, FloatArray] = field(default_factory=dict)
    true_curvature: FloatArray = field(default_factory=lambda: np.zeros(0))  # signed, 1/nm, on arc_grid_nm
    true_x_nm: FloatArray = field(default_factory=lambda: np.zeros(0))       # emitter position (no lateral error)
    true_y_nm: FloatArray = field(default_factory=lambda: np.zeros(0))
    true_z_nm: FloatArray = field(default_factory=lambda: np.zeros(0))       # emitter depth (no axial error)
    true_arc_nm: FloatArray = field(default_factory=lambda: np.zeros(0))     # NaN off the membrane
    true_radial_nm: FloatArray = field(default_factory=lambda: np.zeros(0))  # + outward; NaN off the membrane
    source: NDArray[np.int8] = field(default_factory=lambda: np.zeros(0, dtype=np.int8))   # SOURCE_* flags
    molecule_id: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    sx: FloatArray = field(default_factory=lambda: np.zeros(0))              # pixels
    sy: FloatArray = field(default_factory=lambda: np.zeros(0))
    truth: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------
# Small numeric helpers
# ----------------------------------------------------------------------------------------------------
_erf_vec = np.vectorize(math.erf, otypes=[np.float64])


def _norm_cdf(x: FloatArray) -> FloatArray:
    out: FloatArray = 0.5 * (1.0 + _erf_vec(np.asarray(x, dtype=np.float64) / math.sqrt(2.0)))
    return out


def _circ_dist(s: FloatArray, s0: float, length: float) -> FloatArray:
    d = np.mod(s - s0, length)
    out: FloatArray = np.minimum(d, length - d)
    return out


def _check_keys(name: str, given: Mapping[str, Any], allowed: set) -> None:
    unknown = sorted(set(given) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} key(s): {unknown}; allowed: {sorted(allowed)}")


def _smooth_path(points: FloatArray) -> SmoothPath:
    nxt = np.roll(points, -1, axis=0)
    edges = np.hypot(nxt[:, 0] - points[:, 0], nxt[:, 1] - points[:, 1]).astype(np.float64)
    cum = np.concatenate([[0.0], np.cumsum(edges)]).astype(np.float64)
    return SmoothPath(points_nm=points, vertex_row=np.arange(len(points), dtype=np.intp), edges_nm=edges,
                      cum_nm=cum, length_nm=float(cum[-1]))


def _signed_area(points: FloatArray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _centroid(points: FloatArray) -> Tuple[float, float]:
    x, y = points[:, 0], points[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * float(np.sum(cross))
    return float(np.sum((x + xn) * cross) / (6.0 * a)), float(np.sum((y + yn) * cross) / (6.0 * a))


def _inside_polygon(px: FloatArray, py: FloatArray, poly: FloatArray) -> NDArray[np.bool_]:
    """Even-odd ray casting, vectorised over points, looped over edges."""
    inside = np.zeros(len(px), dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for i in range(len(poly)):
        crosses = (y0[i] > py) != (y1[i] > py)
        if not np.any(crosses):
            continue
        dy = y1[i] - y0[i]
        xint = x0[i] + (x1[i] - x0[i]) * (py - y0[i]) / (dy if dy != 0.0 else 1e-300)
        inside ^= crosses & (px < xint)
    return inside


def _star_shaped(points: FloatArray, centre: Tuple[float, float]) -> bool:
    """True when the polar angle seen from ``centre`` increases monotonically along the CCW curve."""
    ang = np.unwrap(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
    d = np.diff(np.concatenate([ang, ang[:1] + 2.0 * np.pi]))
    return bool(np.all(d > 0.0)) and abs(float(np.sum(d)) - 2.0 * np.pi) < 1e-6


# ----------------------------------------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------------------------------------
def _parametric(shape: str, opt: Mapping[str, Any]) -> Callable[[FloatArray], Tuple[FloatArray, ...]]:
    """Return t -> (x, y, x', y', x'', y'') for the parametric shapes (analytic derivatives)."""
    R = float(opt.get("radius_nm", RADIUS_NM))

    def polar(r: FloatArray, rd: FloatArray, rdd: FloatArray, t: FloatArray) -> Tuple[FloatArray, ...]:
        c, s = np.cos(t), np.sin(t)
        return (r * c, r * s, rd * c - r * s, rd * s + r * c,
                rdd * c - 2.0 * rd * s - r * c, rdd * s + 2.0 * rd * c - r * s)

    if shape == "circle":
        return lambda t: polar(np.full_like(t, R), np.zeros_like(t), np.zeros_like(t), t)
    if shape == "ellipse":
        asp = float(opt.get("aspect", ELLIPSE_ASPECT))
        a, b = R * math.sqrt(asp), R / math.sqrt(asp)
        return lambda t: (a * np.cos(t), b * np.sin(t), -a * np.sin(t), b * np.cos(t), -a * np.cos(t),
                          -b * np.sin(t))
    if shape == "wavy":
        amps = tuple(float(v) for v in opt.get("wave_amps", WAVE_AMPS))
        phases = tuple(float(v) for v in opt.get("wave_phases", WAVE_PHASES))
        if len(amps) != len(WAVE_HARMONICS) or len(phases) != len(WAVE_HARMONICS):
            raise ValueError("wave_amps and wave_phases need one value per harmonic (2, 3, 5, 7)")
        if sum(abs(a) for a in amps) >= 1.0:
            raise ValueError("sum of |wave_amps| must be < 1 (r > 0)")

        def wavy(t: FloatArray) -> Tuple[FloatArray, ...]:
            r = np.ones_like(t)
            rd = np.zeros_like(t)
            rdd = np.zeros_like(t)
            for k, a, ph in zip(WAVE_HARMONICS, amps, phases):
                r = r + a * np.cos(k * t + ph)
                rd = rd - a * k * np.sin(k * t + ph)
                rdd = rdd - a * k * k * np.cos(k * t + ph)
            return polar(R * r, R * rd, R * rdd, t)
        return wavy
    if shape == "two_lobes":
        la = float(opt.get("lobe_amp", LOBE_AMP))
        if not 0.0 <= la < 1.0:
            raise ValueError("lobe_amp must be in [0, 1)")
        return lambda t: polar(R * (1.0 + la * np.cos(2.0 * t)), -2.0 * R * la * np.sin(2.0 * t),
                               -4.0 * R * la * np.cos(2.0 * t), t)
    raise ValueError(f"unknown shape {shape!r}; allowed: {SHAPES}")


def _c_shape_table(opt: Mapping[str, Any], step: float) -> Tuple[FloatArray, FloatArray]:
    """Annular sector closed by two semicircular caps, built directly at uniform arc steps (exact curvature)."""
    Ro = float(opt.get("radius_nm", RADIUS_NM))
    Ri = Ro * float(opt.get("c_inner_frac", C_INNER_FRAC))
    th0 = math.radians(float(opt.get("c_opening_deg", C_OPENING_DEG)))
    w = Ro - Ri
    h = 0.5 * w
    Rm = 0.5 * (Ro + Ri)
    if not 0.0 < Ri < Ro:
        raise ValueError("c_inner_frac must be in (0, 1)")
    if 2.0 * Rm * math.sin(th0) <= w or th0 >= 0.5 * math.pi:
        raise ValueError("c_opening_deg too small (caps touch) or too large")
    phi1 = 2.0 * math.pi - th0
    lens = [(phi1 - th0) * Ro, math.pi * h, (phi1 - th0) * Ri, math.pi * h]
    L = float(sum(lens))
    n = max(64, int(math.ceil(L / step)))
    s = np.arange(n, dtype=np.float64) * (L / n)
    bounds = np.cumsum([0.0] + lens)
    piece = np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, 3)
    u = s - bounds[piece]
    x = np.empty(n)
    y = np.empty(n)
    k = np.empty(n)
    c1 = (Rm * math.cos(phi1), Rm * math.sin(phi1))
    c0 = (Rm * math.cos(th0), Rm * math.sin(th0))
    m = piece == 0
    a = th0 + u[m] / Ro
    x[m], y[m], k[m] = Ro * np.cos(a), Ro * np.sin(a), 1.0 / Ro
    m = piece == 1
    a = phi1 + u[m] / h
    x[m], y[m], k[m] = c1[0] + h * np.cos(a), c1[1] + h * np.sin(a), 1.0 / h
    m = piece == 2
    a = phi1 - u[m] / Ri
    x[m], y[m], k[m] = Ri * np.cos(a), Ri * np.sin(a), -1.0 / Ri
    m = piece == 3
    a = th0 + math.pi + u[m] / h
    x[m], y[m], k[m] = c0[0] + h * np.cos(a), c0[1] + h * np.sin(a), 1.0 / h
    return np.column_stack([x, y]).astype(np.float64), k.astype(np.float64)


def _shape_table(shape: str, opt: Mapping[str, Any]) -> Tuple[FloatArray, FloatArray]:
    """Uniform-arc table of the CCW curve (points (N, 2), signed curvature (N,)), before rotation/translation."""
    step = float(opt.get("table_step_nm", TABLE_STEP_NM))
    if step <= 0.0:
        raise ValueError("table_step_nm must be > 0")
    if shape == "c_shape":
        return _c_shape_table(opt, step)
    f = _parametric(shape, opt)
    m = 16384
    t = np.arange(m, dtype=np.float64) * (2.0 * np.pi / m)
    _, _, xd, yd, _, _ = f(t)
    speed = np.hypot(xd, yd)
    seg = 0.5 * (speed + np.roll(speed, -1)) * (2.0 * np.pi / m)
    s_t = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s_t[-1])
    n = max(64, int(math.ceil(L / step)))
    s = np.arange(n, dtype=np.float64) * (L / n)
    tt = np.interp(s, s_t, np.concatenate([t, [2.0 * np.pi]]))
    x, y, xd, yd, xdd, ydd = f(tt)
    kappa = (xd * ydd - yd * xdd) / np.hypot(xd, yd) ** 3
    return np.column_stack([x, y]).astype(np.float64), np.asarray(kappa, dtype=np.float64)


# ----------------------------------------------------------------------------------------------------
# Densities
# ----------------------------------------------------------------------------------------------------
def _gaussian_field(rng: np.random.Generator, n: int, ds: float, corr_nm: float) -> FloatArray:
    """Periodic Gaussian random field on the arc grid, Gaussian kernel of sd ``corr_nm``, mean 0, sd 1."""
    white = rng.standard_normal(n)
    if corr_nm > 0.0:
        f = np.fft.rfftfreq(n, d=ds)
        white = np.fft.irfft(np.fft.rfft(white) * np.exp(-0.5 * (2.0 * np.pi * f * corr_nm) ** 2), n=n)
    sd = float(np.std(white))
    out: FloatArray = (white - float(np.mean(white))) / (sd if sd > 0.0 else 1.0)
    return out


def _lambda_for_slice(kind: Dict[str, Any], s_c: FloatArray, L: float, kappa: FloatArray,
                      layout: Dict[str, Any]) -> Tuple[FloatArray, Dict[str, Any]]:
    """One slice's lambda (mean 1) from the kind parameters and the slice's random layout."""
    info: Dict[str, Any] = {}
    k = kind["kind"]
    sc = layout["cluster_s"]
    if k == "attractor":
        base = 1.0 + kind["attractor_amp"] * np.exp(-0.5 * (_circ_dist(s_c, sc, L) / kind["attractor_sd_nm"]) ** 2)
    elif k == "inhibition":
        base = np.where(_circ_dist(s_c, sc, L) <= kind["zone_nm"], kind["zone_floor"], 1.0)
    elif k == "regular":
        base = 1.0 + kind["regular_amp"] * np.cos(2.0 * np.pi * (s_c - sc) / kind["period_eff_nm"])
    elif k == "curvature":
        raw = 1.0 + kind["beta"] * kappa * (L / (2.0 * np.pi))
        info["curvature_clip_frac"] = float(np.mean(raw < 0.0))
        base = np.clip(raw, 0.0, None)
    elif k == "random":
        base = np.clip(1.0 + kind["field_amp"] * layout["field"], 0.0, None)
    else:
        base = np.ones_like(s_c)
    base = np.asarray(base, dtype=np.float64)
    if float(base.mean()) <= 0.0:
        raise ValueError("the base density is zero everywhere")
    fc = kind["cluster_frac"] if sc is not None else 0.0
    fk = kind["clump_frac"] if len(layout["clump_s"]) else 0.0
    if fc + fk >= 1.0:
        raise ValueError("cluster_frac + clump_frac must be < 1")
    lam = (1.0 - fc - fk) * base / base.mean()
    if fc > 0.0:
        g = np.exp(-0.5 * (_circ_dist(s_c, sc, L) / kind["cluster_sd_nm"]) ** 2)
        lam = lam + fc * g / g.mean()
    if fk > 0.0:
        g = np.zeros_like(s_c)
        for sj in layout["clump_s"]:
            g += np.exp(-0.5 * (_circ_dist(s_c, float(sj), L) / kind["clump_sd_nm"]) ** 2)
        lam = lam + fk * g / g.mean()
    if layout["gap_s"] is not None:
        lam = np.where(np.mod(s_c - layout["gap_s"], L) < kind["gap_nm"], 0.0, lam)
    if float(lam.sum()) <= 0.0:
        raise ValueError("lambda is zero everywhere (gap longer than the perimeter?)")
    return np.asarray(lam / lam.mean(), dtype=np.float64), info


def _resolve_kind(intensity: Mapping[str, Any], L: float) -> Dict[str, Any]:
    _check_keys("intensity", intensity, _INTENSITY_KEYS)
    k = str(intensity.get("kind", "flat"))
    if k not in KINDS:
        raise ValueError(f"unknown intensity kind {k!r}; allowed: {KINDS}")
    with_cluster = k in ("cluster", "attractor", "inhibition", "regular", "clumps")
    out: Dict[str, Any] = {
        "kind": k,
        "cluster_frac": float(intensity.get("cluster_frac", CLUSTER_FRAC if with_cluster else 0.0)),
        "cluster_sd_nm": float(intensity.get("cluster_sd_nm", CLUSTER_SD_NM)),
        "cluster_arc_frac": intensity.get("cluster_arc_frac"),
        "attractor_amp": float(intensity.get("attractor_amp", ATTRACTOR_AMP)),
        "attractor_sd_nm": float(intensity.get("attractor_sd_nm", ATTRACTOR_SD_NM)),
        "zone_nm": float(intensity.get("zone_nm", ZONE_NM)),
        "zone_floor": float(intensity.get("zone_floor", ZONE_FLOOR)),
        "regular_amp": float(intensity.get("regular_amp", REGULAR_AMP)),
        "beta": float(intensity.get("beta", BETA)),
        "gap_nm": float(intensity.get("gap_nm", GAP_NM if k == "gap" else 0.0)),
        "gap_arc_frac": intensity.get("gap_arc_frac"),
        "n_clumps": int(intensity.get("n_clumps", N_CLUMPS if k == "clumps" else 0)),
        "clump_frac": float(intensity.get("clump_frac", CLUMP_FRAC)),
        "clump_sd_nm": float(intensity.get("clump_sd_nm", CLUMP_SD_NM)),
        "field_amp": float(intensity.get("field_amp", FIELD_AMP)),
        "field_corr_nm": float(intensity.get("field_corr_nm", FIELD_CORR_NM)),
    }
    period = float(intensity.get("period_nm", PERIOD_NM))
    if period <= 0.0:
        raise ValueError("period_nm must be > 0")
    out["period_eff_nm"] = L / max(1, round(L / period))     # closes on the perimeter
    if k in ("attractor", "inhibition", "regular") and out["cluster_frac"] <= 0.0:
        raise ValueError(f"kind {k!r} is defined around the big cluster: cluster_frac must be > 0")
    if not 0.0 <= out["gap_nm"] < L:
        raise ValueError("gap_nm must be in [0, L)")
    if out["regular_amp"] > 1.0 and k == "regular":
        raise ValueError("regular_amp must be <= 1 (lambda >= 0)")
    return out


def _draw_layout(rng: np.random.Generator, kind: Dict[str, Any], L: float, n: int, ds: float,
                 fixed: bool, shared_field: Optional[FloatArray], rho: Optional[float]) -> Dict[str, Any]:
    """Random positions of one slice. ``fixed`` = honour the user's fixed arcs (slice 0 / identical pattern)."""
    has_cluster = kind["cluster_frac"] > 0.0
    cluster_s: Optional[float] = None
    if has_cluster:
        if fixed and kind["cluster_arc_frac"] is not None:
            cluster_s = float(kind["cluster_arc_frac"]) % 1.0 * L
        else:
            cluster_s = float(rng.uniform(0.0, L))
    gap_s: Optional[float] = None
    if kind["gap_nm"] > 0.0:
        if fixed and kind["gap_arc_frac"] is not None:
            gap_s = float(kind["gap_arc_frac"]) % 1.0 * L
        elif cluster_s is not None:
            gap_s = (cluster_s + 0.5 * L - 0.5 * kind["gap_nm"]) % L     # opposite the big cluster
        else:
            gap_s = float(rng.uniform(0.0, L))
    clump_s = rng.uniform(0.0, L, size=kind["n_clumps"]) if kind["n_clumps"] > 0 else np.zeros(0)
    fld: Optional[FloatArray] = None
    if kind["kind"] == "random":
        own = _gaussian_field(rng, n, ds, kind["field_corr_nm"])
        if shared_field is not None and rho is not None:
            fld = math.sqrt(rho) * shared_field + math.sqrt(1.0 - rho) * own
        else:
            fld = own
    return {"cluster_s": cluster_s, "gap_s": gap_s, "clump_s": clump_s, "field": fld}


# ----------------------------------------------------------------------------------------------------
# Axial model: slices, depth, leakage
# ----------------------------------------------------------------------------------------------------
def slice_bounds(z0_nm: float, thickness_nm: float = SLICE_THICKNESS_NM,
                 guard_nm: float = 0.0) -> Tuple[Tuple[float, float], ...]:
    """Bounds [lo, hi) of slices -1, 0, +1 (contiguous; ``guard_nm`` trims both ends of each)."""
    return tuple((z0_nm + k * thickness_nm - 0.5 * thickness_nm + guard_nm,
                  z0_nm + k * thickness_nm + 0.5 * thickness_nm - guard_nm) for k in SLICES)


def true_slice_of(z_nm: FloatArray, z0_nm: float, thickness_nm: float = SLICE_THICKNESS_NM) -> NDArray[np.int8]:
    """Slice of a depth: floor((z - z0 + T/2) / T) in {-1, 0, 1}, OUTSIDE_LABEL otherwise ([lo, hi) convention)."""
    k = np.floor((np.asarray(z_nm, dtype=np.float64) - z0_nm + 0.5 * thickness_nm) / thickness_nm)
    out = np.where(np.abs(k) <= 1, k, OUTSIDE_LABEL)
    return out.astype(np.int8)


def _depth_density(depth: str, z: FloatArray, z0: float, T: float, fracs: Sequence[float], peak: float,
                   sd: float) -> FloatArray:
    if depth == "uniform":
        lab = true_slice_of(z, z0, T)
        dens = np.zeros_like(z)
        for k, f in zip(SLICES, fracs):
            dens[lab == k] = f / T
        return dens
    out: FloatArray = np.exp(-0.5 * ((z - peak) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))
    return out


def leak_matrix(*, thickness_nm: float = SLICE_THICKNESS_NM, lpz_nm: float = LPZ_NM, guard_nm: float = 0.0,
                z0_nm: float = 0.0, depth: str = "uniform", fracs: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
                peak_nm: Optional[float] = None, depth_sd_nm: float = DEPTH_SD_NM,
                dz_nm: float = 0.5) -> Tuple[FloatArray, FloatArray]:
    """Exact axial leakage of this simulator's model, by quadrature [derivation].

    Returns ``(M, w)``: ``M[k, j]`` = P(a localization is observed in analysis slice j | its molecule carries the
    pattern of slice k) and ``w[k]`` the molecule mass with the pattern of slice k (index 0, 1, 2 = slice -1, 0,
    +1). The pattern of slice -1 covers every depth below z0 + T/2 - T (beyond the three slices too), and that of
    +1 every depth above. Analysis slices are trimmed by ``guard_nm``; a row does not sum to 1 (the rest is
    observed outside the analysis slices).
    """
    T = float(thickness_nm)
    if depth not in DEPTH_MODES:
        raise ValueError(f"depth must be one of {DEPTH_MODES}")
    pk = z0_nm if peak_nm is None else float(peak_nm)
    span = 1.5 * T + 6.0 * max(lpz_nm, 1.0) + (6.0 * depth_sd_nm + abs(pk - z0_nm) if depth == "gaussian" else 0.0)
    z = np.asarray(np.arange(z0_nm - span, z0_nm + span, dz_nm) + 0.5 * dz_nm, dtype=np.float64)
    dens = _depth_density(depth, z, z0_nm, T, fracs, pk, depth_sd_nm)
    src = np.clip(np.floor((z - z0_nm + 0.5 * T) / T), -1, 1).astype(int)
    obs = slice_bounds(z0_nm, T, guard_nm)
    M = np.zeros((3, 3))
    w = np.zeros(3)
    for ki, k in enumerate(SLICES):
        sel = src == k
        mass = float(np.sum(dens[sel])) * dz_nm
        w[ki] = mass
        if mass <= 0.0:
            continue
        for ji, (lo, hi) in enumerate(obs):
            if lpz_nm > 0.0:
                p = _norm_cdf((hi - z[sel]) / lpz_nm) - _norm_cdf((lo - z[sel]) / lpz_nm)
            else:
                p = ((z[sel] >= lo) & (z[sel] < hi)).astype(np.float64)
            M[ki, ji] = float(np.sum(dens[sel] * p)) * dz_nm / mass
    w = w / w.sum()
    return M, w


def expected_slice_r(M: FloatArray, w: FloatArray, rho: float = 0.0) -> FloatArray:
    """Pearson r between the observed slice profiles expected from leakage [derivation].

    Model: each slice pattern is lambda_k = 1 + a F_k with F_k of equal variance and corr(F_k, F_l) = rho; the
    observed profile of slice j is sum_k w_k M[k, j] lambda_k. Valid in the limit of many localizations per bin
    (Poisson noise neglected) and without blinking (a blinking molecule split across slices adds correlation, so
    with ``blink`` this is a lower bound). Returns the 3 x 3 matrix in slice order (-1, 0, +1).
    """
    C = np.full((3, 3), float(rho))
    np.fill_diagonal(C, 1.0)
    A = np.asarray(M) * np.asarray(w)[:, None]          # A[k, j] = w_k M[k, j]
    cov = A.T @ C @ A
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    out: FloatArray = cov / np.outer(d, d)
    return out


# ----------------------------------------------------------------------------------------------------
# The simulator
# ----------------------------------------------------------------------------------------------------
def simulate_perimeter(shape: str, intensity: Mapping[str, Any], *, n_locs: int, radial_sd_nm: float,
                       slices: Mapping[str, Any], lpz_nm: float, background_frac: float, interior_frac: float,
                       blink: bool, seed: int, options: Optional[Mapping[str, Any]] = None) -> SimPerimeter:
    """Simulate one axon cross-section (see the module docstring for the model and the truth it emits).

    ``shape`` in SHAPES; ``intensity`` needs ``kind`` (KINDS) plus optional kind parameters; ``slices`` takes
    ``thickness_nm``, ``z0_nm``, ``depth`` ("uniform" with ``fracs`` | "gaussian" with ``peak_nm`` and
    ``depth_sd_nm``), ``pattern`` ("identical" | "independent" | "correlated" with ``rho``) and ``focus_nm``;
    ``options`` (P1 extension, optional) holds geometry and emitter knobs (``radius_nm``, ``precision_nm``,
    ``blink_mean_extra`` ...). Unknown keys raise ValueError. ``n_locs`` is the exact number of localizations.
    """
    opt: Mapping[str, Any] = options or {}
    _check_keys("options", opt, _OPTION_KEYS)
    _check_keys("slices", slices, _SLICE_KEYS)
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; allowed: {SHAPES}")
    n_locs = int(n_locs)
    if n_locs < 1:
        raise ValueError("n_locs must be >= 1")
    if not (0.0 <= background_frac and 0.0 <= interior_frac and background_frac + interior_frac < 1.0):
        raise ValueError("background_frac and interior_frac must be >= 0 and sum to < 1")
    if radial_sd_nm < 0.0 or lpz_nm < 0.0:
        raise ValueError("radial_sd_nm and lpz_nm must be >= 0")
    rng = np.random.default_rng(seed)

    # ---- geometry ------------------------------------------------------------------------------------
    pts, kappa = _shape_table(shape, opt)
    rot = math.radians(float(opt.get("rotation_deg", 0.0)))
    cx, cy = (float(v) for v in opt.get("centre_nm", (0.0, 0.0)))
    cr, sr = math.cos(rot), math.sin(rot)
    pts = np.column_stack([cr * pts[:, 0] - sr * pts[:, 1] + cx, sr * pts[:, 0] + cr * pts[:, 1] + cy])
    if _signed_area(pts) <= 0.0:
        raise RuntimeError("internal: curve is not counter-clockwise")
    path = _smooth_path(pts)
    n_grid = len(pts)
    L = float(path.length_nm)
    ds = L / n_grid                              # uniform by construction (chord vs arc differs < 1e-6 relative)
    arc_grid = np.arange(n_grid, dtype=np.float64) * ds
    s_c = arc_grid + 0.5 * ds
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tang /= np.hypot(tang[:, 0], tang[:, 1])[:, None]
    normal = np.column_stack([tang[:, 1], -tang[:, 0]])          # outward for a CCW curve
    centroid = _centroid(pts)

    # ---- densities per slice ---------------------------------------------------------------------------
    kind = _resolve_kind(intensity, L)
    pattern = str(slices.get("pattern", "identical"))
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; allowed: {PATTERNS}")
    rho: Optional[float] = None
    if pattern == "correlated":
        if kind["kind"] != "random":
            raise ValueError("pattern 'correlated' is defined for intensity kind 'random' only")
        rho = float(slices.get("rho", 0.5))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
    elif "rho" in slices:
        raise ValueError("rho is only used with pattern 'correlated'")
    shared = _gaussian_field(rng, n_grid, ds, kind["field_corr_nm"]) if pattern == "correlated" else None
    layouts: Dict[int, Dict[str, Any]] = {}
    for k in (0, -1, 1):
        if pattern == "identical" and k != 0:
            layouts[k] = layouts[0]
        else:
            layouts[k] = _draw_layout(rng, kind, L, n_grid, ds, fixed=(k == 0 or pattern == "identical"),
                                      shared_field=shared, rho=rho)
    lam: Dict[int, FloatArray] = {}
    lam_info: Dict[str, Any] = {}
    for k in SLICES:
        lam[k], info = _lambda_for_slice(kind, s_c, L, kappa, layouts[k])
        lam_info.update(info)
    cdf = {k: np.cumsum(lam[k]) / float(np.sum(lam[k])) for k in SLICES}

    # ---- axial model -------------------------------------------------------------------------------------
    T = float(slices.get("thickness_nm", SLICE_THICKNESS_NM))
    if T <= 0.0:
        raise ValueError("thickness_nm must be > 0")
    depth = str(slices.get("depth", "uniform"))
    if depth not in DEPTH_MODES:
        raise ValueError(f"depth must be one of {DEPTH_MODES}")
    fracs = tuple(float(v) for v in slices.get("fracs", (1 / 3, 1 / 3, 1 / 3)))
    if len(fracs) != 3 or min(fracs) < 0.0 or sum(fracs) <= 0.0:
        raise ValueError("fracs needs three non-negative values")
    fracs = tuple(f / sum(fracs) for f in fracs)
    peak: Optional[float] = float(slices["peak_nm"]) if "peak_nm" in slices else (0.0 if depth == "gaussian"
                                                                                   else None)
    if depth == "uniform" and peak is not None:
        raise ValueError("peak_nm is only used with depth 'gaussian'")
    z0 = float(slices.get("z0_nm", peak if peak is not None else 0.0))
    depth_sd = float(slices.get("depth_sd_nm", DEPTH_SD_NM))
    if depth_sd <= 0.0:
        raise ValueError("depth_sd_nm must be > 0")
    focus = float(slices.get("focus_nm", FOCUS_NM))

    def draw_depth(n: int) -> FloatArray:
        if depth == "uniform":
            k = rng.choice(np.array(SLICES), size=n, p=np.array(fracs))
            return np.asarray(z0 + k * T + rng.uniform(-0.5 * T, 0.5 * T, size=n), dtype=np.float64)
        return np.asarray(rng.normal(peak, depth_sd, size=n), dtype=np.float64)

    # ---- counts --------------------------------------------------------------------------------------
    n_bg = int(round(background_frac * n_locs))
    n_int = int(round(interior_frac * n_locs))
    n_mem = n_locs - n_bg - n_int
    precision = float(opt.get("precision_nm", PRECISION_NM))
    mean_extra = float(opt.get("blink_mean_extra", BLINK_MEAN_EXTRA))
    if precision < 0.0 or mean_extra < 0.0:
        raise ValueError("precision_nm and blink_mean_extra must be >= 0")
    if blink and mean_extra > 0.0:
        p_geo = 1.0 / (1.0 + mean_extra)
        chunks: List[NDArray[np.int64]] = []
        total = 0
        while total < n_mem:
            c = rng.geometric(p_geo, size=max(16, int(1.3 * (n_mem - total) / (1.0 + mean_extra)) + 16))
            chunks.append(c.astype(np.int64))
            total += int(c.sum())
        counts = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
        cum = np.cumsum(counts)
        n_mol = int(np.searchsorted(cum, n_mem) + 1) if n_mem > 0 else 0
        counts = counts[:n_mol].copy()
        if n_mol:
            counts[-1] -= int(cum[n_mol - 1] - n_mem)
    else:
        n_mol = n_mem
        counts = np.ones(n_mem, dtype=np.int64)

    # ---- membrane molecules ------------------------------------------------------------------------------
    mz = draw_depth(n_mol)
    mslice = true_slice_of(mz, z0, T)
    pattern_of = np.where(mslice == OUTSIDE_LABEL, np.sign(mz - z0), mslice).astype(int)
    marc = np.empty(n_mol)
    for k in SLICES:
        sel = pattern_of == k
        m = int(np.sum(sel))
        if m:
            idx = np.minimum(np.searchsorted(cdf[k], rng.uniform(0.0, 1.0, size=m), side="right"), n_grid - 1)
            marc[sel] = (idx + rng.uniform(0.0, 1.0, size=m)) * ds
    mrad = rng.normal(0.0, radial_sd_nm, size=n_mol) if radial_sd_nm > 0.0 else np.zeros(n_mol)
    pos = marc / ds
    i0 = np.floor(pos).astype(int) % n_grid
    i1 = (i0 + 1) % n_grid
    fr = (pos - np.floor(pos))[:, None]
    base_xy = (1.0 - fr) * pts[i0] + fr * pts[i1]
    nrm = (1.0 - fr) * normal[i0] + fr * normal[i1]
    nrm /= np.hypot(nrm[:, 0], nrm[:, 1])[:, None]
    mxy = base_xy + mrad[:, None] * nrm

    rep = np.repeat(np.arange(n_mol), counts)
    tx_m, ty_m = mxy[rep, 0], mxy[rep, 1]
    tz_m = mz[rep]

    # ---- interior and background ---------------------------------------------------------------------------
    margin = float(opt.get("bg_margin_nm", BG_MARGIN_NM))
    lo_xy, hi_xy = pts.min(axis=0), pts.max(axis=0)
    poly = pts[:: max(1, n_grid // 1024)]
    ix_l: List[FloatArray] = []
    iy_l: List[FloatArray] = []
    got = 0
    while got < n_int:
        px = rng.uniform(lo_xy[0], hi_xy[0], size=2 * (n_int - got) + 64)
        py = rng.uniform(lo_xy[1], hi_xy[1], size=len(px))
        ok = _inside_polygon(px, py, poly)
        ix_l.append(px[ok])
        iy_l.append(py[ok])
        got += int(ok.sum())
    tx_i = np.concatenate(ix_l)[:n_int] if n_int else np.zeros(0)
    ty_i = np.concatenate(iy_l)[:n_int] if n_int else np.zeros(0)
    tz_i = draw_depth(n_int)
    tx_b = rng.uniform(lo_xy[0] - margin, hi_xy[0] + margin, size=n_bg)
    ty_b = rng.uniform(lo_xy[1] - margin, hi_xy[1] + margin, size=n_bg)
    tz_b = rng.uniform(z0 - 1.5 * T, z0 + 1.5 * T, size=n_bg)

    tx = np.concatenate([tx_m, tx_i, tx_b])
    ty = np.concatenate([ty_m, ty_i, ty_b])
    tz = np.concatenate([tz_m, tz_i, tz_b])
    n = len(tx)
    # lateral and axial errors per localization (background: none, it already is noise)
    x = tx.copy()
    y = ty.copy()
    z = tz.copy()
    nm = n_mem + n_int
    if precision > 0.0:
        x[:nm] += rng.normal(0.0, precision, size=nm)
        y[:nm] += rng.normal(0.0, precision, size=nm)
    if lpz_nm > 0.0:
        z[:nm] += rng.normal(0.0, lpz_nm, size=nm)

    source = np.concatenate([np.full(n_mem, SOURCE_MEMBRANE), np.full(n_int, SOURCE_INTERIOR),
                             np.full(n_bg, SOURCE_BACKGROUND)]).astype(np.int8)
    mol_id = np.concatenate([rep, n_mol + np.arange(n_int + n_bg)]).astype(np.int64)
    true_arc = np.concatenate([marc[rep], np.full(n_int + n_bg, np.nan)])
    true_rad = np.concatenate([mrad[rep], np.full(n_int + n_bg, np.nan)])

    # ---- astigmatic widths ----------------------------------------------------------------------------------
    w0 = float(opt.get("psf_w0_px", PSF_W0_PX))
    c_ast = float(opt.get("astig_c_nm", ASTIG_C_NM))
    d_ast = float(opt.get("astig_d_nm", ASTIG_D_NM))
    wn = float(opt.get("width_noise_frac", WIDTH_NOISE_FRAC))
    # from the OBSERVED z: in a file z is looked up from the widths, so widths and z must agree; computing them
    # from the true z would bias an sx = sy crossing read per bin of observed z (regression dilution, ~8 nm at
    # lpz 47 nm with a peak 63 nm from the focus)
    sx = w0 * np.sqrt(1.0 + ((z - focus - c_ast) / d_ast) ** 2) * (1.0 + wn * rng.standard_normal(n))
    sy = w0 * np.sqrt(1.0 + ((z - focus + c_ast) / d_ast) ** 2) * (1.0 + wn * rng.standard_normal(n))

    # ---- truth -----------------------------------------------------------------------------------------------
    M, w_src = leak_matrix(thickness_nm=T, lpz_nm=lpz_nm, z0_nm=z0, depth=depth, fracs=fracs, peak_nm=peak,
                           depth_sd_nm=depth_sd)
    if depth == "uniform":
        in_slice = np.array(fracs)
    else:
        b = slice_bounds(z0, T)
        in_slice = np.array([float(_norm_cdf(np.array([(hi - peak) / depth_sd]))[0]
                                   - _norm_cdf(np.array([(lo - peak) / depth_sd]))[0]) for lo, hi in b])
    lam_stack = np.vstack([lam[k] for k in SLICES])
    with np.errstate(invalid="ignore", divide="ignore"):
        lam_r = np.corrcoef(lam_stack) if np.all(lam_stack.std(axis=1) > 0) else np.full((3, 3), np.nan)
    truth: Dict[str, Any] = {
        "shape": shape, "length_nm": L, "radius_char_nm": L / (2.0 * np.pi),
        "min_radius_of_curvature_nm": float(1.0 / np.max(np.abs(kappa))),
        "star_shaped_from_centroid": _star_shaped(pts, centroid), "centroid_nm": centroid,
        "kind": kind["kind"], "pattern": pattern, "rho": rho,
        "cluster_arc_nm": {k: layouts[k]["cluster_s"] for k in SLICES},
        "gap_arcs_nm": {k: ([] if layouts[k]["gap_s"] is None
                            else [(layouts[k]["gap_s"], layouts[k]["gap_s"] + kind["gap_nm"])]) for k in SLICES},
        "clump_arcs_nm": {k: np.sort(layouts[k]["clump_s"]) for k in SLICES},
        "period_eff_nm": kind["period_eff_nm"] if kind["kind"] == "regular" else None,
        "attractor_sd_nm": kind["attractor_sd_nm"] if kind["kind"] == "attractor" else None,
        "attractor_amp": kind["attractor_amp"] if kind["kind"] == "attractor" else None,
        "zone_nm": kind["zone_nm"] if kind["kind"] == "inhibition" else None,
        "beta": kind["beta"] if kind["kind"] == "curvature" else None,
        "curvature_clip_frac": lam_info.get("curvature_clip_frac"),
        "cluster_frac": kind["cluster_frac"], "gap_nm": kind["gap_nm"],
        "true_slice_lambda_r": lam_r,
        "z0_nm": z0, "thickness_nm": T, "slice_bounds_nm": slice_bounds(z0, T), "depth": depth,
        "depth_sd_nm": depth_sd if depth == "gaussian" else None, "fracs": fracs if depth == "uniform" else None,
        "peak_nm": peak, "focus_nm": focus, "peak_minus_focus_nm": None if peak is None else peak - focus,
        "lpz_nm": lpz_nm, "radial_sd_nm": radial_sd_nm, "precision_nm": precision,
        "source_frac": w_src, "true_slice_frac": in_slice, "leak_matrix": M,
        "expected_slice_r": expected_slice_r(M, w_src, 1.0 if pattern == "identical" else (rho or 0.0)),
        "n_membrane": n_mem, "n_interior": n_int, "n_background": n_bg, "n_molecules": n_mol,
        "blink": bool(blink), "blink_mean_extra": mean_extra if blink else 0.0, "seed": seed,
    }
    return SimPerimeter(
        x_nm=x, y_nm=y, z_nm=z, true_path=path, true_lambda=lam[0],
        true_slice=true_slice_of(tz, z0, T), arc_grid_nm=arc_grid, true_lambda_by_slice=lam,
        true_curvature=kappa, true_x_nm=tx, true_y_nm=ty, true_z_nm=tz, true_arc_nm=true_arc,
        true_radial_nm=true_rad, source=source, molecule_id=mol_id, sx=np.asarray(sx, dtype=np.float64),
        sy=np.asarray(sy, dtype=np.float64), truth=truth)


# ----------------------------------------------------------------------------------------------------
# Named scenarios (the presets P5 / P6 / P7 / P11 start from)
# ----------------------------------------------------------------------------------------------------
_COMMON: Dict[str, Any] = {"n_locs": 10000, "radial_sd_nm": RADIAL_SD_NM, "lpz_nm": LPZ_NM, "background_frac": 0.0,
                           "interior_frac": 0.0, "blink": True, "seed": 0}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    # question A: negatives, then positives (answer: cluster_arc_nm, attractor_*, zone_nm, period_eff_nm)
    "a_flat": {"shape": "ellipse", "intensity": {"kind": "flat"}, "slices": {}},
    "a_clumps": {"shape": "ellipse", "intensity": {"kind": "clumps"}, "slices": {}},
    "a_attractor": {"shape": "ellipse", "intensity": {"kind": "attractor"}, "slices": {}},
    "a_inhibition": {"shape": "ellipse", "intensity": {"kind": "inhibition"}, "slices": {}},
    "a_regular": {"shape": "ellipse", "intensity": {"kind": "regular"}, "slices": {}},
    # question B (answer: beta, true_curvature; true_path is the true contour)
    "b_null": {"shape": "wavy", "intensity": {"kind": "curvature", "beta": 0.0}, "slices": {}},
    "b_curvature": {"shape": "wavy", "intensity": {"kind": "curvature", "beta": BETA}, "slices": {}},
    "b_thin": {"shape": "circle", "intensity": {"kind": "flat"}, "slices": {}, "options": {"radius_nm": 400.0}},
    # question C (answer: expected_slice_r, true_slice_lambda_r)
    "c_identical": {"shape": "circle", "intensity": {"kind": "random"}, "slices": {"pattern": "identical"}},
    "c_independent": {"shape": "circle", "intensity": {"kind": "random"}, "slices": {"pattern": "independent"},
                      "lpz_nm": 0.0},
    "c_leak": {"shape": "circle", "intensity": {"kind": "random"}, "slices": {"pattern": "independent"}},
    "c_correlated": {"shape": "circle", "intensity": {"kind": "random"},
                     "slices": {"pattern": "correlated", "rho": 0.5}},
    # z0 and slices (answer: peak_nm, focus_nm, true_slice_frac, leak_matrix)
    "z_peak_focus": {"shape": "circle", "intensity": {"kind": "flat"},
                     "slices": {"depth": "gaussian", "peak_nm": 0.0, "focus_nm": -43.0}},
    # contour (answer: true_path, gap_arcs_nm, star_shaped_from_centroid)
    "contour_c_gap": {"shape": "c_shape", "intensity": {"kind": "gap"}, "slices": {}},
    "contour_two_lobes": {"shape": "two_lobes", "intensity": {"kind": "flat"}, "slices": {},
                          "background_frac": 0.15},
}


def scenario_kwargs(name: str, **overrides: Any) -> Dict[str, Any]:
    """Keyword arguments of ``simulate_perimeter`` for a named scenario; ``overrides`` replace any of them."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    kw: Dict[str, Any] = dict(_COMMON)
    kw.update(SCENARIOS[name])
    kw.update(overrides)
    return kw


def simulate_scenario(name: str, **overrides: Any) -> SimPerimeter:
    """``simulate_perimeter(**scenario_kwargs(name, **overrides))``."""
    return simulate_perimeter(**scenario_kwargs(name, **overrides))
