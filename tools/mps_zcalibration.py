# -*- coding: utf-8 -*-
"""
The z calibration of each measurement: its record, its bead stack, and what they say (SCI-1).

Astigmatic z is a calibration-table lookup. Picasso fits two sixth-degree polynomials, the spot width wx(z) and height
wy(z), to a bead (or single-molecule) z stack stepped by "Step size in nm" over "Number of frames" (grouped by
"Frames per step"), shifts the stage axis so that the two curves cross at z = 0, and multiplies every fitted z and lpz
by a "Magnification factor" m that corrects the refractive-index mismatch (Huang et al. 2008, 0.79; Li et al. 2018,
0.75; Li et al. 2019, 0.8; Bratton and Shaevitz 2015 measured 0.57; Besseling et al. 2015, 0.85). m is a property of
the microscope, the objective, the medium and the depth, never a constant, so it is read from the record of each
measurement and never assumed: when the record does not carry it, it is blank.

What this module does, and what it does not
-------------------------------------------
It READS. The z of a Picasso file is already multiplied by the m of its record (``picasso.zfit._fit_z``:
``locs["z"] = z * magnification_factor``); applying m again would apply it twice. Nothing here changes a z, and no
analysis reads anything from here. What it gives is the provenance of the z axis: which record, which m and where m
came from, the calibrated range, the fit bounds, and (from a localized bead stack) the real z_zero, the usable range
and the lateral shift with depth -- the last one is subtracted between rings by a later version, not here.

Two layouts of the record (both written by Picasso)
---------------------------------------------------
* the ``Z Calibration`` dictionary of a Localize step, with ``Z Calibration Path`` beside it (this lab's files);
* the calibration keys merged into the ``Fit 3D`` step of current Picasso, with ``Calibration path`` and
  ``Filter range``.
A calibration YAML (``*_3d_calib.yaml``) holds the same keys at the top level. A ThunderSTORM calibration
(``PolynomialCalibration``) is recognised and named, but it records no m, no step and no frames: what would be derived
from them stays blank.

The calibrated range is NOMINAL: Picasso shifts the stage axis by the curve-crossing offset z_zero before fitting and
does not store it, so the record gives +/-(n_steps - 1) x step x m / 2, uncertain by |z_zero| x m. The bead stack
resolves it.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FIT_BOUND_NM", "OUTER_FRACTION", "PILE_TOLERANCE_NM", "M_RANGE", "MIN_STEPS", "BEAD_LINK_NM",
    "SPLINE_KNOT_SPACING_NM", "CalibrationRecord", "BeadStackReport", "ZCalibrationState", "record_from_mapping",
    "record_from_info", "read_calibration_file", "validate_record", "records_match", "record_differences",
    "bead_stack_report", "load_bead_stack", "refit_z", "Z_CALIBRATION_COLUMNS", "export_columns", "provenance_words", "store_entry",
    "state_for", "clean_store", "SOURCE_WORDS",
]

# Picasso bounds the fitted z to [-1000, 1000] nm before multiplying by m (``_fit_z``: ``bounds=[-1000, 1000]``):
# a localization beyond the calibrated curves is piled at the bound, not extrapolated.
FIT_BOUND_NM = 1000.0
# The outer part of the calibrated half-range whose share is reported (a convention; documented range 0.6-0.95).
OUTER_FRACTION = 0.8
# A localization within this distance of a fit bound counts as piled there (the record's acceptance: exact to 1 nm).
PILE_TOLERANCE_NM = 1.0
# The validation's range for m: every published value (0.57-0.85) inside, with margin; outside it is flagged, never
# clamped.
M_RANGE: Tuple[float, float] = (0.4, 1.2)
# Fewer steps than this cannot constrain a sixth-degree polynomial with any margin (7 coefficients).
MIN_STEPS = 5
# Beads of a calibration slide are microns apart and move by nanometres between steps: localizations within this
# radius of each other, chained, are one bead.
BEAD_LINK_NM = 100.0
# Interior knots of the cubic B-spline of the lateral shift x(z), y(z) (Carlini et al. 2015), in the file's z units.
SPLINE_KNOT_SPACING_NM = 150.0
# The refit's grid step (stage nm): the resolution floor of the per-step spread in Z_usable.
REFIT_GRID_NM = 1.0

LAYOUT_Z_CALIBRATION = "Z Calibration"
LAYOUT_FIT_3D = "Fit 3D"
LAYOUT_CALIBRATION_FILE = "calibration file"
LAYOUT_THUNDERSTORM = "ThunderSTORM"

ORIGIN_FILE = "file record"          # the loaded file's own metadata
ORIGIN_CHOSEN = "chosen record"      # a calibration file chosen in the Measurement panel
ORIGIN_USER = "user"                 # a magnification factor typed by hand
ORIGIN_BLANK = "blank"               # not recorded, not given

# What each origin means, in the words of the panel and the tooltips.
SOURCE_WORDS: Dict[str, str] = {
    ORIGIN_FILE: "read from the file's own record",
    ORIGIN_CHOSEN: "read from the chosen calibration file",
    ORIGIN_USER: "typed by you",
    ORIGIN_BLANK: "not recorded",
}

_X = "X Coefficients"
_Y = "Y Coefficients"
_FRAMES = "Number of frames"
_STEP = "Step size in nm"
_M = "Magnification factor"
_FPS = "Frames per step"
_BOUNDS = "Frame bounds"
_ORDER = "Frame order"


def _base(path: Any) -> str:
    """The file name of a path written on any system ('C:/a/b.yaml', '//server/x/b.yaml', 'b.yaml'); '' for none."""
    text = "" if path is None else str(path).strip()
    if not text or text.upper() == "N/A":
        return ""
    return re.split(r"[\\/]", text)[-1]


def _float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> Optional[int]:
    f = _float(value)
    if f is None or f != int(f):
        return None
    return int(f)


def _coefficients(value: Any) -> Optional[Tuple[float, ...]]:
    if not isinstance(value, (list, tuple)):
        return None
    out = tuple(_float(v) for v in value)
    if len(out) != 7 or any(v is None for v in out):
        return None
    return tuple(float(v) for v in out if v is not None)


# =====================================================================================================================
#  The record
# =====================================================================================================================
@dataclass
class CalibrationRecord:
    """One astigmatism calibration, as recorded. Every field is what the record says, or None when it says nothing."""

    layout: str
    origin: str                                   # ORIGIN_FILE or ORIGIN_CHOSEN
    file_name: str = ""                           # the calibration file's name, never its folder
    x_coeffs: Optional[Tuple[float, ...]] = None  # wx(z) in pixels, highest power first (numpy.polyfit order)
    y_coeffs: Optional[Tuple[float, ...]] = None
    n_frames: Optional[int] = None
    step_nm: Optional[float] = None
    frames_per_step: Optional[int] = None
    frame_bounds: Any = None
    frame_order: Optional[str] = None
    magnification: Optional[float] = None
    filter_range: Optional[float] = None
    unreadable: List[str] = field(default_factory=list)   # keys present but not readable

    # ------------------------------------------------------------------ what follows from it
    @property
    def n_steps(self) -> Optional[int]:
        """Distinct stage positions: Number of frames // Frames per step (1 when absent), as Picasso counts them."""
        if self.n_frames is None:
            return None
        return int(self.n_frames) // max(1, int(self.frames_per_step or 1))

    def stage_z_of_steps(self) -> Optional[NDArray[np.float64]]:
        """Picasso's stage z of every step before the z_zero shift: -(k d - span / 2), centred on the full stack."""
        n, d = self.n_steps, self.step_nm
        if n is None or d is None or n < 1:
            return None
        span = (n - 1) * float(d)
        return np.asarray(-(np.arange(n) * float(d) - span / 2.0), dtype=float)

    def bounded_steps(self) -> Optional[NDArray[np.int64]]:
        """The steps that entered the fit: all of them, or those 'Frame bounds' leaves (inclusive frame segments)."""
        n = self.n_steps
        if n is None or self.n_frames is None:
            return None
        if self.frame_bounds is None:
            return np.arange(n, dtype=np.int64)
        segments = _frame_segments(self.frame_bounds, int(self.n_frames) - 1)
        if not segments:
            return np.arange(n, dtype=np.int64)
        frames = np.arange(int(self.n_frames))
        keep = np.zeros(frames.size, dtype=bool)
        for lo, hi in segments:
            keep |= (frames >= lo) & (frames <= hi)
        steps = _step_of_frame(frames, n, int(self.frames_per_step or 1), self.frame_order)
        return np.unique(steps[keep & (steps >= 0)]).astype(np.int64)

    @property
    def stage_bounds_nm(self) -> Optional[Tuple[float, float]]:
        """The calibrated stage range (nm, before m) of the steps that entered the fit, before the z_zero shift."""
        z = self.stage_z_of_steps()
        steps = self.bounded_steps()
        if z is None or steps is None or steps.size == 0:
            return None
        used = z[steps]
        return float(np.min(used)), float(np.max(used))

    @property
    def stage_range_nm(self) -> Optional[float]:
        b = self.stage_bounds_nm
        return None if b is None else b[1] - b[0]

    def nominal_bounds_nm(self, m: Optional[float]) -> Optional[Tuple[float, float]]:
        """The nominal calibrated range in the file's z units (x m). Nominal: z_zero is not stored."""
        b = self.stage_bounds_nm
        if b is None or m is None:
            return None
        return b[0] * float(m), b[1] * float(m)

    # ------------------------------------------------------------------ the width curves
    @property
    def has_curves(self) -> bool:
        return self.x_coeffs is not None and self.y_coeffs is not None

    def _search_interval(self) -> Tuple[float, float]:
        b = self.stage_bounds_nm
        if b is None:
            return -FIT_BOUND_NM, FIT_BOUND_NM
        half = max(abs(b[0]), abs(b[1]))
        return -half, half

    def crossing_nm(self) -> Optional[float]:
        """Where wx(z) = wy(z), nearest z = 0, in the record's z (stage nm, after Picasso's shift): ~0 by construction."""
        if not self.has_curves:
            return None
        lo, hi = self._search_interval()
        z = np.linspace(lo, hi, 20001)
        g = np.polyval(self.x_coeffs, z) - np.polyval(self.y_coeffs, z)
        sign = np.signbit(g)
        idx = np.flatnonzero(sign[1:] != sign[:-1])
        if idx.size == 0:
            return None
        roots = z[idx] - g[idx] * (z[idx + 1] - z[idx]) / (g[idx + 1] - g[idx])
        return float(roots[np.argmin(np.abs(roots))])

    def monotone_interval_nm(self) -> Optional[Tuple[float, float]]:
        """The interval around the crossing where wx - wy is monotone (stage nm): where the fit's answer is unique."""
        c = self.crossing_nm()
        if c is None:
            return None
        lo, hi = -FIT_BOUND_NM, FIT_BOUND_NM
        z = np.linspace(lo, hi, 20001)
        g = np.polyval(self.x_coeffs, z) - np.polyval(self.y_coeffs, z)
        dg = np.diff(g)
        i0 = int(np.clip(np.searchsorted(z, c) - 1, 0, dg.size - 1))
        s0 = np.sign(dg[i0])
        if s0 == 0:
            return None
        same = np.sign(dg) == s0
        left = i0
        while left > 0 and same[left - 1]:
            left -= 1
        right = i0
        while right < dg.size - 1 and same[right + 1]:
            right += 1
        return float(z[left]), float(z[right + 1])


def _frame_segments(bounds: Any, last_frame: int) -> List[Tuple[int, int]]:
    """Picasso's 'Frame bounds': one (min, max) pair or a list of them, inclusive; None for an open end."""
    if bounds is None:
        return []
    pairs = bounds if (isinstance(bounds, (list, tuple)) and bounds and isinstance(bounds[0], (list, tuple))) \
        else [bounds]
    out: List[Tuple[int, int]] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        lo = 0 if pair[0] is None else _int(pair[0])
        hi = last_frame if pair[1] is None else _int(pair[1])
        if lo is None or hi is None:
            continue
        out.append((int(lo), int(hi)))
    return out


def _step_of_frame(frames: NDArray[Any], n_steps: int, frames_per_step: int,
                   order: Optional[str]) -> NDArray[np.int64]:
    """The stage step of each frame, as ``picasso.zfit.calibrate_z`` assigns it; -1 for trailing frames."""
    f = np.asarray(frames, dtype=np.int64)
    fps = max(1, int(frames_per_step))
    if fps > 1 and str(order or "fov").lower() == "z":
        step = f % n_steps
    else:
        step = f // fps
    valid = (f >= 0) & (f < n_steps * fps)
    return np.where(valid, step, -1).astype(np.int64)


def record_from_mapping(values: Mapping[str, Any], *, layout: str, origin: str,
                        file_name: str = "") -> CalibrationRecord:
    """A record from one mapping of Picasso's keys. A key that is present but unreadable is named, not guessed."""
    rec = CalibrationRecord(layout=layout, origin=origin, file_name=file_name)
    rec.x_coeffs = _coefficients(values.get(_X))
    rec.y_coeffs = _coefficients(values.get(_Y))
    rec.n_frames = _int(values.get(_FRAMES))
    rec.step_nm = _float(values.get(_STEP))
    rec.frames_per_step = _int(values.get(_FPS))
    rec.frame_bounds = values.get(_BOUNDS)
    order = values.get(_ORDER)
    rec.frame_order = None if order is None else str(order)
    rec.magnification = _float(values.get(_M))
    rec.filter_range = _float(values.get("Filter range"))
    for key, parsed in ((_X, rec.x_coeffs), (_Y, rec.y_coeffs), (_FRAMES, rec.n_frames), (_STEP, rec.step_nm),
                        (_FPS, rec.frames_per_step), (_M, rec.magnification)):
        if values.get(key) is not None and parsed is None:
            rec.unreadable.append(key)
    return rec


def record_from_info(info: Sequence[Mapping[str, Any]]) -> Optional[CalibrationRecord]:
    """
    The calibration record a localization file carries in its Picasso metadata chain, newest step first: the merged
    'Fit 3D' step of current Picasso, or the 'Z Calibration' dictionary of a Localize step. None when there is none.
    """
    for doc in reversed(list(info or [])):
        if not isinstance(doc, Mapping):
            continue
        generated = str(doc.get("Generated by", "") or "")
        if "fit 3d" in generated.lower() and (_X in doc or _Y in doc):
            name = _base(doc.get("Calibration path")) or _base(doc.get("Path"))
            return record_from_mapping(doc, layout=LAYOUT_FIT_3D, origin=ORIGIN_FILE, file_name=name)
        block = doc.get("Z Calibration")
        if isinstance(block, Mapping):
            name = _base(doc.get("Z Calibration Path")) or _base(block.get("Path"))
            return record_from_mapping(block, layout=LAYOUT_Z_CALIBRATION, origin=ORIGIN_FILE, file_name=name)
    return None


_THUNDER = re.compile(r"(\w+)\s*:\s*([-+0-9.eE]+)")


def read_calibration_file(path: str) -> CalibrationRecord:
    """
    A calibration chosen by hand: a Picasso calibration YAML (``*_3d_calib.yaml``), a Picasso localization YAML that
    carries the record of its file, or a ThunderSTORM calibration. Raises ValueError when the file holds none.
    """
    name = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    if "ThunderSTORM" in text and "Calibration" in text:
        rec = CalibrationRecord(layout=LAYOUT_THUNDERSTORM, origin=ORIGIN_CHOSEN, file_name=name)
        values = {k: float(v) for k, v in _THUNDER.findall(text)}
        if not values:
            raise ValueError(f"{name}: a ThunderSTORM calibration with no readable parameters")
        return rec
    try:
        import yaml  # type: ignore[import-untyped]

        docs = [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    except Exception as error:  # noqa: BLE001 - any parse failure is "not a calibration file"
        raise ValueError(f"{name} could not be read as YAML: {error}") from error
    for doc in docs:
        if _X in doc or _Y in doc:
            if "fit 3d" in str(doc.get("Generated by", "")).lower():
                return record_from_mapping(doc, layout=LAYOUT_FIT_3D, origin=ORIGIN_CHOSEN, file_name=name)
            return record_from_mapping(doc, layout=LAYOUT_CALIBRATION_FILE, origin=ORIGIN_CHOSEN, file_name=name)
    embedded = record_from_info(docs)
    if embedded is not None:
        embedded.origin = ORIGIN_CHOSEN
        embedded.file_name = name
        return embedded
    raise ValueError(f"{name} holds no astigmatism calibration record ('X Coefficients' / 'Z Calibration').")


def validate_record(rec: CalibrationRecord) -> List[str]:
    """
    The record's problems, in words (design 13.2): keys present, step > 0, at least MIN_STEPS steps, m in M_RANGE,
    width curves crossing at 0 (within half a step) and wx - wy monotone over the calibrated range. An empty list is
    a record that passed. Nothing is changed: a problem is shown, never fixed.
    """
    if rec.layout == LAYOUT_THUNDERSTORM:
        return ["a ThunderSTORM calibration: it records no magnification factor, step or number of frames, so "
                "nothing is derived from it"]
    out: List[str] = []
    missing = [k for k, v in ((_X, rec.x_coeffs), (_Y, rec.y_coeffs), (_FRAMES, rec.n_frames), (_STEP, rec.step_nm),
                              (_M, rec.magnification)) if v is None and k not in rec.unreadable]
    if missing:
        out.append("not recorded: " + ", ".join(missing))
    if rec.unreadable:
        out.append("present but not readable: " + ", ".join(rec.unreadable))
    if rec.step_nm is not None and rec.step_nm <= 0:
        out.append(f"step {rec.step_nm:g} nm is not positive")
    n = rec.n_steps
    if n is not None and n < MIN_STEPS:
        out.append(f"{n} steps: fewer than {MIN_STEPS} cannot constrain the sixth-degree width curves")
    m = rec.magnification
    if m is not None and not (M_RANGE[0] <= m <= M_RANGE[1]):
        out.append(f"magnification factor {m:g} is outside {M_RANGE[0]:g}-{M_RANGE[1]:g}")
    if rec.has_curves:
        c = rec.crossing_nm()
        if c is None:
            out.append("the width curves do not cross: z = 0 is undefined")
        elif rec.step_nm is not None and rec.step_nm > 0 and abs(c) > rec.step_nm / 2.0:
            out.append(f"the width curves cross at {c:+.1f} nm, not at 0 (more than half a step)")
        b = rec.stage_bounds_nm
        mono = rec.monotone_interval_nm()
        if b is not None and mono is not None and rec.step_nm:
            lo, hi = max(b[0], -FIT_BOUND_NM), min(b[1], FIT_BOUND_NM)
            if mono[0] > lo + REFIT_GRID_NM or mono[1] < hi - REFIT_GRID_NM:
                in_file = "" if m is None else \
                    f" = {mono[0] * m:+.0f}..{mono[1] * m:+.0f} nm in the file's z"
                out.append(f"wx - wy is monotone only over {mono[0]:+.0f}..{mono[1]:+.0f} nm (stage{in_file}), not "
                           f"over the nominal calibrated range {lo:+.0f}..{hi:+.0f} nm: beyond it a width pair has "
                           f"two answers, so z there is ambiguous")
    return out


def record_differences(a: Optional[CalibrationRecord], b: Optional[CalibrationRecord]) -> Optional[List[str]]:
    """
    What differs between two records, in words ("the width curves", "the step", ...): an empty list when they are the
    same calibration (the same width curves to 1e-9 relative, and no recorded step, number of frames, frames per
    step or magnification factor that differs; a value recorded on one side only is not a difference). None when
    either is missing or has no width curves (nothing to compare).
    """
    if a is None or b is None or not a.has_curves or not b.has_curves:
        return None
    out: List[str] = []
    if not all(np.allclose(np.asarray(ca), np.asarray(cb), rtol=1e-9, atol=0.0)
               for ca, cb in ((a.x_coeffs, b.x_coeffs), (a.y_coeffs, b.y_coeffs))):
        out.append("the width curves")
    for what, va, vb in (("the number of frames", a.n_frames, b.n_frames), ("the step", a.step_nm, b.step_nm),
                         ("m", a.magnification, b.magnification),
                         ("the frames per step", a.frames_per_step, b.frames_per_step)):
        if va is not None and vb is not None and abs(float(va) - float(vb)) > 1e-9 * max(1.0, abs(float(va))):
            out.append(f"{what} ({float(va):g} vs {float(vb):g})")
    return out


def records_match(a: Optional[CalibrationRecord], b: Optional[CalibrationRecord]) -> Optional[bool]:
    """Whether two records are the same calibration (``record_differences`` is empty); None when not comparable."""
    diff = record_differences(a, b)
    return None if diff is None else not diff


# =====================================================================================================================
#  The bead stack
# =====================================================================================================================
def refit_z(sx: NDArray[Any], sy: NDArray[Any], rec: CalibrationRecord, *, chunk: int = 4000) -> NDArray[np.float64]:
    """
    Picasso's z fit of each localization from its spot width and height (pixels), in the record's stage nm (before
    m): the z in [-FIT_BOUND_NM, FIT_BOUND_NM] minimising (sqrt sx - sqrt wx(z))^2 + (sqrt sy - sqrt wy(z))^2 (Huang
    et al. 2008; ``picasso.zfit._fit_z_target``). A grid of REFIT_GRID_NM refined by a parabola through the minimum.
    """
    if not rec.has_curves:
        raise ValueError("the record has no width curves")
    grid = np.arange(-FIT_BOUND_NM, FIT_BOUND_NM + REFIT_GRID_NM / 2, REFIT_GRID_NM)
    swx = np.sqrt(np.clip(np.polyval(rec.x_coeffs, grid), 0.0, None))
    swy = np.sqrt(np.clip(np.polyval(rec.y_coeffs, grid), 0.0, None))
    rx = np.sqrt(np.clip(np.asarray(sx, dtype=float), 0.0, None))
    ry = np.sqrt(np.clip(np.asarray(sy, dtype=float), 0.0, None))
    out = np.empty(rx.size, dtype=float)
    for start in range(0, rx.size, chunk):
        a, b = rx[start:start + chunk, None], ry[start:start + chunk, None]
        d = (a - swx[None, :]) ** 2 + (b - swy[None, :]) ** 2
        i = np.argmin(d, axis=1)
        z = grid[i].astype(float)
        inner = (i > 0) & (i < grid.size - 1)
        rows = np.flatnonzero(inner)
        if rows.size:
            ii = i[rows]
            y0, y1, y2 = d[rows, ii - 1], d[rows, ii], d[rows, ii + 1]
            den = y0 - 2.0 * y1 + y2
            with np.errstate(invalid="ignore", divide="ignore"):
                off = np.where(den > 0, 0.5 * (y0 - y2) / den, 0.0)
            z[rows] = grid[ii] + np.clip(off, -0.5, 0.5) * REFIT_GRID_NM
        out[start:start + chunk] = z
    return out


@dataclass
class BeadStackReport:
    """What a localized bead stack says about the calibration it was taken for. Lengths in the file's z units
    (stage nm x m); every one None when m is unknown."""

    file_name: str
    n_beads: int
    n_steps_used: int
    magnification: Optional[float]
    magnification_origin: str
    z_zero_stage_nm: Optional[float]                  # the curve crossing on the stage axis (what Picasso shifted by)
    calibrated_min_nm: Optional[float] = None
    calibrated_max_nm: Optional[float] = None
    usable_min_nm: Optional[float] = None
    usable_max_nm: Optional[float] = None
    shift_x_nm: Optional[float] = None                # excursion of x(z) over Z_usable (max - min)
    shift_y_nm: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    _spline_x: Any = None
    _spline_y: Any = None

    @property
    def z_zero_nm(self) -> Optional[float]:
        if self.z_zero_stage_nm is None or self.magnification is None:
            return None
        return self.z_zero_stage_nm * self.magnification

    def shift_at(self, z_nm: Any) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """The lateral shift (dx, dy) at the file's z, relative to z = 0 (the B-spline; Carlini et al. 2015)."""
        if self._spline_x is None or self._spline_y is None:
            return None
        z = np.asarray(z_nm, dtype=float)
        return np.asarray(self._spline_x(z), dtype=float), np.asarray(self._spline_y(z), dtype=float)

    def shift_between(self, z_a_nm: float, z_b_nm: float) -> Optional[Tuple[float, float]]:
        """The rigid lateral offset the optics add between two depths (b minus a): for a later version's ring pairs."""
        sa, sb = self.shift_at([z_a_nm]), self.shift_at([z_b_nm])
        if sa is None or sb is None:
            return None
        return float(sb[0][0] - sa[0][0]), float(sb[1][0] - sa[1][0])


def _medians_by_step(steps: NDArray[np.int64], values: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    out = np.full(n, np.nan)
    order = np.argsort(steps, kind="stable")
    s, v = steps[order], values[order]
    cuts = np.flatnonzero(np.diff(s)) + 1
    for group_s, group_v in zip(np.split(s, cuts), np.split(v, cuts)):
        if group_s.size:
            out[int(group_s[0])] = float(np.median(group_v))
    return out


def bead_stack_report(
    x_nm: NDArray[Any], y_nm: NDArray[Any], frame: NDArray[Any], sx: NDArray[Any], sy: NDArray[Any],
    rec: CalibrationRecord, *, magnification: Optional[float], magnification_origin: str, file_name: str = "",
) -> BeadStackReport:
    """
    The real z_zero, the real calibrated range, Z_usable and the lateral shift x(z), y(z) from a localized bead
    stack and the record it was calibrated with (its step, frames, frames per step, order and width curves).

    * z_zero: the crossing of the per-step median widths, each fitted with a sixth-degree polynomial on Picasso's
      stage axis, as ``picasso.zfit.calibrate_z`` does before it shifts the axis.
    * Z_usable: the contiguous run of steps around z = 0 whose refitted z (``refit_z``) lies within 2 sigma_z of the
      true z, is not piled at a fit bound, and increases with it (PROPOSAL 2.4); sigma_z is the step's robust spread
      (1.4826 MAD), floored at the refit's grid.
    * x(z), y(z): each bead's position relative to its own position at z = 0, averaged per step over the beads and
      fitted with a least-squares cubic B-spline (Carlini et al. 2015).
    """
    from scipy.interpolate import LSQUnivariateSpline
    from sklearn.cluster import DBSCAN

    if rec.n_steps is None or rec.step_nm is None or rec.step_nm <= 0:
        raise ValueError("the record gives no step or number of frames, so the stack's depths are unknown")
    if not rec.has_curves:
        raise ValueError("the record has no width curves")
    x = np.asarray(x_nm, dtype=float)
    y = np.asarray(y_nm, dtype=float)
    sxa = np.asarray(sx, dtype=float)
    sya = np.asarray(sy, dtype=float)
    n = int(rec.n_steps)
    step = _step_of_frame(np.asarray(frame), n, int(rec.frames_per_step or 1), rec.frame_order)
    used = rec.bounded_steps()
    keep = step >= 0
    if used is not None:
        keep &= np.isin(step, used)
    keep &= np.isfinite(x) & np.isfinite(y) & np.isfinite(sxa) & np.isfinite(sya)
    x, y, sxa, sya, step = x[keep], y[keep], sxa[keep], sya[keep], step[keep]
    if x.size < 10:
        raise ValueError("fewer than 10 localizations fall on the calibrated steps")
    z_stage = rec.stage_z_of_steps()
    assert z_stage is not None
    present = np.unique(step)

    # -- z_zero: where the stack's own width curves cross
    msx = _medians_by_step(step, sxa, n)
    msy = _medians_by_step(step, sya, n)
    ok = np.isfinite(msx) & np.isfinite(msy)
    zz = z_stage[ok]
    if zz.size < MIN_STEPS:
        raise ValueError(f"widths on {zz.size} steps: fewer than {MIN_STEPS}")
    cx = np.polyfit(zz, msx[ok], 6)
    cy = np.polyfit(zz, msy[ok], 6)
    grid = np.linspace(float(zz.min()), float(zz.max()), 20001)
    g = np.polyval(cx, grid) - np.polyval(cy, grid)
    sign = np.signbit(g)
    idx = np.flatnonzero(sign[1:] != sign[:-1])
    if idx.size:
        roots = grid[idx] - g[idx] * (grid[idx + 1] - grid[idx]) / (g[idx + 1] - g[idx])
        z_zero: Optional[float] = float(roots[np.argmin(np.abs(roots))])
    else:
        z_zero = None
    report = BeadStackReport(file_name=file_name, n_beads=0, n_steps_used=int(present.size),
                             magnification=magnification, magnification_origin=magnification_origin,
                             z_zero_stage_nm=z_zero)
    if z_zero is None:
        report.notes.append("the stack's width curves do not cross: z_zero, the range and the shift are not given")
        return report
    true_stage = z_stage - z_zero                        # the record's z of each step (stage nm, after the shift)
    if magnification is None:
        report.notes.append("magnification factor not recorded: the stack's depths cannot be put in the file's units")
        return report
    m = float(magnification)
    used_z = true_stage[present]
    report.calibrated_min_nm = float(used_z.min()) * m
    report.calibrated_max_nm = float(used_z.max()) * m

    # -- Z_usable
    zf = refit_z(sxa, sya, rec)
    med = _medians_by_step(step, zf, n)
    mad = _medians_by_step(step, np.abs(zf - med[step]), n)
    counts = np.bincount(step, minlength=n)
    spread = np.maximum(1.4826 * mad, REFIT_GRID_NM)
    usable = ((counts >= 2) & np.isfinite(med) & (np.abs(med - true_stage) <= 2.0 * spread)
              & (np.abs(med) < FIT_BOUND_NM - PILE_TOLERANCE_NM))
    order = np.argsort(true_stage)                       # steps by increasing true z
    ts, us, ms = true_stage[order], usable[order], med[order]
    centre = int(np.argmin(np.abs(ts)))
    if us[centre]:
        lo = hi = centre
        while lo > 0 and us[lo - 1] and ms[lo - 1] < ms[lo]:
            lo -= 1
        while hi < ts.size - 1 and us[hi + 1] and ms[hi + 1] > ms[hi]:
            hi += 1
        report.usable_min_nm = float(ts[lo]) * m
        report.usable_max_nm = float(ts[hi]) * m
    else:
        report.notes.append("the step nearest z = 0 does not refit within 2 sigma_z: no usable range")

    # -- the beads and their lateral shift
    labels = DBSCAN(eps=BEAD_LINK_NM, min_samples=3).fit_predict(np.column_stack([x, y]))
    z_file = true_stage * m
    sum_dx = np.zeros(n)
    sum_dy = np.zeros(n)
    count = np.zeros(n)
    n_beads = 0
    for label in np.unique(labels[labels >= 0]):
        sel = labels == label
        bs = step[sel]
        steps_b = np.unique(bs)
        if steps_b.size < max(MIN_STEPS, present.size // 2):
            continue
        nb = np.bincount(bs, minlength=n).astype(float)
        mx = np.bincount(bs, weights=x[sel], minlength=n)
        my = np.bincount(bs, weights=y[sel], minlength=n)
        have = nb > 0
        zb = z_file[have]
        if not (zb.min() <= 0.0 <= zb.max()):
            continue
        o = np.argsort(zb)
        px = mx[have] / nb[have]
        py = my[have] / nb[have]
        x0 = float(np.interp(0.0, zb[o], px[o]))
        y0 = float(np.interp(0.0, zb[o], py[o]))
        sum_dx[have] += px - x0
        sum_dy[have] += py - y0
        count[have] += 1.0
        n_beads += 1
    report.n_beads = n_beads
    if n_beads == 0:
        report.notes.append("no bead spans z = 0 over most of the stack: the lateral shift is not given")
        return report
    have = count > 0
    zs, o = z_file[have], np.argsort(z_file[have])
    zs = zs[o]
    dx = (sum_dx[have] / count[have])[o]
    dy = (sum_dy[have] / count[have])[o]
    w = np.sqrt(count[have][o])
    span = zs[-1] - zs[0]
    n_knots = int(span // SPLINE_KNOT_SPACING_NM)
    knots = zs[0] + np.arange(1, n_knots) * (span / max(n_knots, 1)) if n_knots >= 2 else np.empty(0)
    if zs.size < knots.size + 5:
        knots = np.empty(0)
    report._spline_x = LSQUnivariateSpline(zs, dx, knots, w=w, k=3)
    report._spline_y = LSQUnivariateSpline(zs, dy, knots, w=w, k=3)
    if report.usable_min_nm is not None and report.usable_max_nm is not None:
        zg = np.linspace(report.usable_min_nm, report.usable_max_nm, 401)
        shift = report.shift_at(zg)
        assert shift is not None
        report.shift_x_nm = float(np.ptp(shift[0]))
        report.shift_y_nm = float(np.ptp(shift[1]))
    return report


_BEAD_CACHE: Dict[Tuple[Any, ...], BeadStackReport] = {}


def load_bead_stack(path: str, rec: CalibrationRecord, *, magnification: Optional[float], magnification_origin: str,
                    pixel_size_nm: Optional[float] = None) -> BeadStackReport:
    """
    Read a LOCALIZED bead stack (Picasso HDF5 with frame, x, y, sx, sy) and report it. A raw stack (a .tif) is not
    localized here: localize it in Picasso first. Cached by the file, its size and time, the record and m.
    """
    from tools import mps_io

    if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
        raise ValueError("this is a raw image stack: localize it in Picasso (Localize, with the box and the camera "
                         "settings of the calibration) and choose the HDF5 it writes")
    st = os.stat(path)
    key = (os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime, rec.x_coeffs, rec.y_coeffs,
           rec.n_frames, rec.step_nm, rec.frames_per_step, rec.frame_order, str(rec.frame_bounds), magnification,
           magnification_origin, pixel_size_nm)
    hit = _BEAD_CACHE.get(key)
    if hit is not None:
        return hit
    loc = mps_io.load_localizations(path, pixel_size_nm=pixel_size_nm)
    frame, sx, sy = loc.frame, loc.column("sx"), loc.column("sy")
    if frame is None or sx is None or sy is None:
        raise ValueError(f"{os.path.basename(path)} has no frame / sx / sy columns: not a localized bead stack")
    report = bead_stack_report(loc.x_nm, loc.y_nm, frame, sx, sy, rec, magnification=magnification,
                               magnification_origin=magnification_origin, file_name=os.path.basename(path))
    if len(_BEAD_CACHE) > 16:
        _BEAD_CACHE.clear()
    _BEAD_CACHE[key] = report
    return report


# =====================================================================================================================
#  The state of one measurement, its store, its words and its columns
# =====================================================================================================================
@dataclass
class ZCalibrationState:
    """Everything known about the z calibration of one loaded file. Built by ``state_for``; shown, exported, never
    used to change a z."""

    file_record: Optional[CalibrationRecord] = None
    chosen_record: Optional[CalibrationRecord] = None
    # The chosen record was confirmed to replace the file's own record although it differs from it.
    chosen_overrides: bool = False
    chosen_error: str = ""
    typed_m: Optional[float] = None
    bead: Optional[BeadStackReport] = None
    bead_name: str = ""
    bead_error: str = ""

    @property
    def chosen_in_use(self) -> bool:
        """A chosen record is in use when the file carries none of its own, or when it was confirmed to replace a
        differing one. Otherwise the file's own record -- the one Picasso fitted this file with -- is in use, and a
        chosen record is only an alternative shown beside it (manual test 4: a stored choice must never claim, in
        the exported provenance, a calibration the file was not fitted with)."""
        if self.chosen_record is None:
            return False
        return self.file_record is None or self.chosen_overrides

    @property
    def alternative(self) -> Optional[CalibrationRecord]:
        """The chosen record when it is not in use (shown, never exported as the record in use)."""
        return self.chosen_record if (self.chosen_record is not None and not self.chosen_in_use) else None

    @property
    def record(self) -> Optional[CalibrationRecord]:
        """The record in use: the file's own, unless it has none or a chosen one was confirmed to replace it."""
        return self.chosen_record if self.chosen_in_use else self.file_record

    @property
    def magnification(self) -> Optional[float]:
        rec = self.record
        if rec is not None and rec.magnification is not None:
            return rec.magnification
        return self.typed_m

    @property
    def magnification_origin(self) -> str:
        """Where m comes from: the record in use when it carries one (a read value beats a typed one), else typed,
        else blank."""
        rec = self.record
        if rec is not None and rec.magnification is not None:
            return rec.origin
        if self.typed_m is not None:
            return ORIGIN_USER
        return ORIGIN_BLANK

    @property
    def match(self) -> Optional[bool]:
        return records_match(self.file_record, self.chosen_record)

    @property
    def is_empty(self) -> bool:
        """Nothing read, nothing given: the program behaves and writes exactly as without SCI-1."""
        return self.record is None and self.typed_m is None and self.bead is None


def provenance_words(state: Optional[ZCalibrationState]) -> Optional[str]:
    """
    'z calibration: <file>, m = x (<source>)' for the provenance line and the axial titles, or None when nothing is
    read or given (the caller keeps today's 'z as fitted' words).
    """
    if state is None or state.is_empty:
        return None
    rec = state.record
    name = "" if rec is None else (rec.file_name or (f"the file's '{rec.layout}' record" if rec.origin == ORIGIN_FILE
                                                      else rec.layout))
    m = state.magnification
    m_words = "-" if m is None else f"{m:g}"
    head = f"z calibration: {name or 'no record'}, m = {m_words} ({SOURCE_WORDS[state.magnification_origin]})"
    return head + "; z as fitted, no z correction applied"


Z_CALIBRATION_COLUMNS: Tuple[str, ...] = (
    "z_calibration_source", "z_calibration_file", "z_magnification", "z_magnification_origin",
    "z_calibrated_min_nm", "z_calibrated_max_nm", "z_calibrated_range_nm", "z_calibrated_range_origin",
    "z_calibration_match", "z_bead_stack_file", "z_zero_nm", "z_usable_min_nm", "z_usable_max_nm",
    "z_lateral_shift_x_nm", "z_lateral_shift_y_nm")


def export_columns(state: Optional[ZCalibrationState]) -> Dict[str, Any]:
    """The per-axon columns (EXP-ORIGIN: a blank value is an empty cell, with its origin in ``*_origin``)."""
    out: Dict[str, Any] = {name: None for name in Z_CALIBRATION_COLUMNS}
    out["z_magnification_origin"] = ORIGIN_BLANK
    if state is None:
        return out
    rec = state.record
    if rec is not None:
        out["z_calibration_source"] = f"{rec.origin} ({rec.layout})"
        out["z_calibration_file"] = rec.file_name or None
    out["z_magnification"] = state.magnification
    out["z_magnification_origin"] = state.magnification_origin
    bead = state.bead
    bounds: Optional[Tuple[float, float]] = None
    if bead is not None and bead.calibrated_min_nm is not None and bead.calibrated_max_nm is not None:
        bounds, out["z_calibrated_range_origin"] = (bead.calibrated_min_nm, bead.calibrated_max_nm), "bead stack"
    elif rec is not None:
        nominal = rec.nominal_bounds_nm(state.magnification)
        if nominal is not None:
            bounds, out["z_calibrated_range_origin"] = nominal, "nominal"
    if bounds is not None:
        out["z_calibrated_min_nm"], out["z_calibrated_max_nm"] = bounds
        out["z_calibrated_range_nm"] = bounds[1] - bounds[0]
    match = state.match
    out["z_calibration_match"] = None if match is None else ("same" if match else "different")
    if bead is not None:
        out["z_bead_stack_file"] = bead.file_name or None
        out["z_zero_nm"] = bead.z_zero_nm
        out["z_usable_min_nm"] = bead.usable_min_nm
        out["z_usable_max_nm"] = bead.usable_max_nm
        out["z_lateral_shift_x_nm"] = bead.shift_x_nm
        out["z_lateral_shift_y_nm"] = bead.shift_y_nm
    return out


# ------------------------------------------------------------------ the store (one entry per folder, per file)
# record_override: the chosen record was confirmed to replace the files' own records although it differs from them.
STORE_FIELDS: Tuple[str, ...] = ("record_path", "record_override", "bead_path", "bead_pixel_size_nm", "m_typed")


def file_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def store_entry(store: Optional[Mapping[str, Any]], data_path: str) -> Dict[str, Any]:
    """The entry for a loaded file: its folder's (one per acquisition), with whatever the file's own entry sets
    taking precedence field by field."""
    from tools.mps_pixel_size import folder_key

    if not store or not data_path:
        return {}
    out: Dict[str, Any] = {}
    for key in (folder_key(data_path), file_key(data_path)):
        entry = store.get(key)
        if isinstance(entry, Mapping):
            out.update(entry)
    return out


def clean_store(store: Any) -> Dict[str, Dict[str, Any]]:
    """The store with every entry that is not one dropped (a settings file is never trusted): paths are strings, a
    typed m lies in M_RANGE, a bead pixel size in 1-5000 nm. Dropped, never clamped."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(store, Mapping):
        return out
    for key, entry in store.items():
        if not isinstance(entry, Mapping):
            continue
        clean: Dict[str, Any] = {}
        for name in ("record_path", "bead_path"):
            value = entry.get(name)
            if isinstance(value, str) and value.strip():
                clean[name] = value
        m = _float(entry.get("m_typed"))
        if m is not None and M_RANGE[0] <= m <= M_RANGE[1]:
            clean["m_typed"] = m
        if entry.get("record_override") is True and "record_path" in clean:
            clean["record_override"] = True
        px = _float(entry.get("bead_pixel_size_nm"))
        if px is not None and 1.0 <= px <= 5000.0:
            clean["bead_pixel_size_nm"] = px
        if clean:
            out[str(key)] = clean
    return out


def state_for(info: Sequence[Mapping[str, Any]], entry: Optional[Mapping[str, Any]] = None) -> ZCalibrationState:
    """The z calibration of one file: its own record (from ``info``), and what its store entry adds."""
    state = ZCalibrationState(file_record=record_from_info(info or []))
    entry = entry or {}
    path = entry.get("record_path")
    if isinstance(path, str) and path:
        try:
            state.chosen_record = read_calibration_file(path)
        except (OSError, ValueError) as error:
            state.chosen_error = f"{os.path.basename(path)}: {error}"
    state.chosen_overrides = entry.get("record_override") is True
    m = _float(entry.get("m_typed"))
    state.typed_m = m if (m is not None and M_RANGE[0] <= m <= M_RANGE[1]) else None
    bead_path = entry.get("bead_path")
    if isinstance(bead_path, str) and bead_path:
        state.bead_name = os.path.basename(bead_path)
        rec = state.record
        if rec is None:
            state.bead_error = "no calibration record: the stack's steps and width curves come from it"
        else:
            try:
                state.bead = load_bead_stack(bead_path, rec, magnification=state.magnification,
                                             magnification_origin=state.magnification_origin,
                                             pixel_size_nm=_float(entry.get("bead_pixel_size_nm")))
            except Exception as error:  # noqa: BLE001 - a stack that cannot be read is reported, never fatal
                state.bead_error = str(error)
    return state
