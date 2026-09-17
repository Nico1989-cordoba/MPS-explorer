# -*- coding: utf-8 -*-
"""
How well are two channels registered, and by how much must one be moved?

Every cross-channel number in ``tools.mps_crosschannel`` inherits the error
of putting the two channels in one frame. This module produces that error
instead of asking for it, from two sources:

  FIDUCIALS IN THE DATA (Exchange-PAINT, and any acquisition with markers).
    Exchange-PAINT images every target with the same dye, in successive
    rounds, so there is no chromatic aberration: what separates the rounds
    is the drift between them, in x, y and z. Markers present in both
    rounds -- gold nanoparticles, or any spot localized in nearly every
    frame -- measure it directly. Their mean offset is the correction; the
    scatter of the individual offsets around it is the error. The error is
    estimated leave-one-out (each marker judged by a shift fitted without
    it), because the residual of a fit on the same markers underestimates
    the error on everything else.

  A CHROMATIC CALIBRATION (two colours imaged at once).
    The user's bead-calibration tool, matriz-transformacion
    (package smlm_affine), writes its quality metrics next to the matrix.
    Only the lateral error is available there -- the tool is 2D -- and the
    matrix itself is assumed to have been applied upstream.

Picasso has no equivalent: ``picasso align`` shifts the channels by
cross-correlating the images themselves, which for two different proteins
aligns away part of the biology, never touches z, and reports no error.

Errors are expressed as RMS: laterally the RMS of the 2D residual length,
axially the RMS of the z residual. Independent sources add in quadrature.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

# Picasso's own rule (imageprocess.find_fiducials): a marker is localized in
# more than 80% of the frames. Counted here in distinct frames, not in
# localizations, so a spot fitted twice in one frame does not count double.
FIDUCIAL_MIN_FRAME_FRACTION = 0.8
# Drift-corrected markers stay within a few nm; this leaves room for a
# poorer correction. On the 15.07.26 DNA-PAINT sample no spot reaches even
# 25% of the frames at any radius up to 1 um, so a generous radius does not
# turn persistent binding sites into markers.
FIDUCIAL_RADIUS_NM = 150.0
# Largest round-to-round offset searched when pairing markers.
MATCH_MAX_OFFSET_NM = 3000.0
# How close two markers must land once the offset is removed.
MATCH_TOLERANCE_NM = 100.0
# Exchange-PAINT rounds share nearly all their markers, so the right offset
# lines up most of them. With tens of markers, a search over micrometres
# finds offsets that line up two by chance; requiring this fraction of the
# smaller set keeps those from passing as a registration.
MATCH_MIN_SHARED_FRACTION = 0.5
# Leave-one-out needs at least this many pairs to say anything.
MIN_PAIRS_FOR_ERROR = 3
# For an isotropic 2D Gaussian error, RMS of the residual length divided by
# its median: sqrt(2) / sqrt(2 ln 2). Used to turn the calibration tool's
# median leave-one-out error into an RMS.
RMS_PER_MEDIAN_2D = math.sqrt(1.0 / math.log(2.0))


def same_pixel_size(a: Optional[float], b: Optional[float]) -> bool:
    """Whether two files agree on the pixel size; unknown counts as agreeing."""
    return not (a and b) or abs(float(a) - float(b)) < 1e-6


def reach_nm(*locs: Any) -> Optional[float]:
    """How far the given files' localizations reach from the camera origin."""
    best = 0.0
    for loc in locs:
        if loc is None or not getattr(loc, "n", 0):
            continue
        best = max(best, float(np.hypot(np.abs(loc.x_nm).max(),
                                        np.abs(loc.y_nm).max())))
    return best or None


def pixel_size_disagreement(
    name_a: str, pixel_a: Optional[float],
    name_b: str, pixel_b: Optional[float],
    reach_nm: Optional[float] = None,
) -> Optional[str]:
    """
    What to tell the user when two channels disagree on the pixel size, or
    None when they agree (or one of them is unknown).

    Two colours split onto one camera and the rounds of an Exchange-PAINT
    acquisition all go through the same optics, so the two files must give
    the same pixel size; when they do not, one of them is wrong. The error
    scales the positions about the camera origin, so it grows with the
    distance from it and no shift can correct it: ``reach_nm`` is how far
    the data reaches from that origin, and the message then says how far
    off its far corner lands.

    The 2023 sciatic-nerve data has exactly this: in staining 6, adducin
    was localized with 133 nm and spectrin with 135 nm.
    """
    if same_pixel_size(pixel_a, pixel_b):
        return None
    assert pixel_a is not None and pixel_b is not None
    text = (f"{name_a} and {name_b} give different pixel sizes "
            f"({pixel_a:g} and {pixel_b:g} nm). Both channels are recorded "
            f"through the same optics -- two colours on one camera, or "
            f"rounds of one acquisition -- so one of the two is wrong.")
    if reach_nm:
        off = float(reach_nm) * abs(pixel_a - pixel_b) / min(pixel_a, pixel_b)
        text += (f" The error scales the positions about the camera origin: "
                 f"{off:.0f} nm at the far corner of the field, which no "
                 f"shift can correct.")
    else:
        text += (" The error scales the positions about the camera origin, "
                 "which no shift can correct.")
    return text


@dataclass
class Fiducial:
    """One marker found in one channel."""

    x_nm: float
    y_nm: float
    z_nm: Optional[float]
    n_locs: int
    n_frames: int
    frame_fraction: float
    # RMS distance of its localizations from its centre, laterally and in z.
    spread_nm: float
    spread_z_nm: Optional[float]


@dataclass
class Registration:
    """What is known about the registration of channel B onto channel A."""

    source: str                      # "none" | "fiducials" | "calibration" | "combined"
    # Added to channel B's coordinates to bring it onto channel A.
    shift_nm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lateral_rms_nm: Optional[float] = None
    axial_rms_nm: Optional[float] = None
    n_pairs: int = 0
    description: str = ""
    paths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Per-pair detail for display: (x_nm, y_nm) of the marker in channel A
    # and its leave-one-out residual (dx, dy, dz) in nm.
    pair_positions_nm: List[Tuple[float, float]] = field(default_factory=list)
    pair_residuals_nm: List[Tuple[float, float, float]] = field(
        default_factory=list)

    @property
    def shifts_channel_b(self) -> bool:
        return any(s != 0.0 for s in self.shift_nm)

    def apply(
        self,
        x_nm: NDArray[np.float64],
        y_nm: NDArray[np.float64],
        z_nm: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Channel B's coordinates moved onto channel A."""
        dx, dy, dz = self.shift_nm
        return (np.asarray(x_nm, float) + dx, np.asarray(y_nm, float) + dy,
                np.asarray(z_nm, float) + dz)


NO_REGISTRATION = Registration(
    source="none",
    description="No registration: channel B is used as loaded.",
)


# ===================================================================
#  Finding markers
# ===================================================================
def find_fiducials(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: Optional[NDArray[np.float64]] = None,
    *,
    n_frames: Optional[int] = None,
    radius_nm: float = FIDUCIAL_RADIUS_NM,
    min_frame_fraction: float = FIDUCIAL_MIN_FRAME_FRACTION,
) -> List[Fiducial]:
    """
    Markers: spots with a localization in at least ``min_frame_fraction``
    of the frames, within ``radius_nm`` of their centre.

    ``n_frames`` is the length of the acquisition; without it the span of
    the frame numbers is used. The data should be drift-corrected: an
    uncorrected marker wanders further than ``radius_nm`` and is missed.
    Returned brightest-first (most frames first).
    """
    frame = np.asarray(frame, dtype=np.int64)
    x = np.asarray(x_nm, dtype=float)
    y = np.asarray(y_nm, dtype=float)
    z = None if z_nm is None else np.asarray(z_nm, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if z is not None:
        ok &= np.isfinite(z)
    frame, x, y = frame[ok], x[ok], y[ok]
    if z is not None:
        z = z[ok]
    if frame.size == 0:
        return []
    total = int(n_frames) if n_frames else int(frame.max() - frame.min() + 1)
    need = min_frame_fraction * total
    if total <= 0 or frame.size < need:
        return []

    # Candidate cells: a marker split across up to four cells still puts a
    # quarter of its frames in one of them.
    cell_x = np.floor(x / radius_nm).astype(np.int64)
    cell_y = np.floor(y / radius_nm).astype(np.int64)
    cell_x -= cell_x.min()
    cell_y -= cell_y.min()
    cell = cell_x * (int(cell_y.max()) + 1) + cell_y
    f0 = frame - frame.min()
    pairs = np.unique(cell * (int(f0.max()) + 1) + f0)
    cells, frames_per_cell = np.unique(pairs // (int(f0.max()) + 1),
                                       return_counts=True)
    candidates = cells[frames_per_cell >= 0.25 * need]
    if candidates.size == 0:
        return []
    candidates = candidates[np.argsort(
        -frames_per_cell[np.searchsorted(cells, candidates)])]

    tree = cKDTree(np.column_stack([x, y]))
    found: List[Fiducial] = []
    for c in candidates:
        members = np.nonzero(cell == c)[0]
        centre = np.array([np.median(x[members]), np.median(y[members])])
        for _ in range(4):
            near = np.asarray(tree.query_ball_point(centre, radius_nm),
                              dtype=np.int64)
            if near.size == 0:
                break
            new = np.array([np.median(x[near]), np.median(y[near])])
            moved = float(np.hypot(*(new - centre)))
            centre = new
            if moved < 0.1:
                break
        near = np.asarray(tree.query_ball_point(centre, radius_nm),
                          dtype=np.int64)
        if near.size == 0:
            continue
        if any(math.hypot(centre[0] - f.x_nm, centre[1] - f.y_nm)
               < 2 * radius_nm for f in found):
            continue
        n_distinct = int(np.unique(frame[near]).size)
        if n_distinct < need:
            continue
        lateral = np.hypot(x[near] - centre[0], y[near] - centre[1])
        zc = spread_z = None
        if z is not None:
            zc = float(np.median(z[near]))
            spread_z = float(np.sqrt(np.mean((z[near] - zc) ** 2)))
        found.append(Fiducial(
            x_nm=float(centre[0]), y_nm=float(centre[1]), z_nm=zc,
            n_locs=int(near.size), n_frames=n_distinct,
            frame_fraction=n_distinct / total,
            spread_nm=float(np.sqrt(np.mean(lateral ** 2))),
            spread_z_nm=spread_z,
        ))
    found.sort(key=lambda f: -f.n_frames)
    return found


def fiducials_in(loc: Any, **kwargs: Any) -> List[Fiducial]:
    """``find_fiducials`` on a ``tools.mps_io.Localizations``."""
    if loc.frame is None:
        return []
    return find_fiducials(
        loc.frame, loc.x_nm, loc.y_nm, loc.z_nm if loc.is_3d else None,
        n_frames=loc.n_frames, **kwargs)


# ===================================================================
#  Pairing markers between channels
# ===================================================================
def match_fiducials(
    a: Sequence[Fiducial],
    b: Sequence[Fiducial],
    *,
    max_offset_nm: float = MATCH_MAX_OFFSET_NM,
    tolerance_nm: float = MATCH_TOLERANCE_NM,
) -> List[Tuple[int, int]]:
    """
    Pairs (index in a, index in b) of the same marker in both channels.

    The rounds may be offset by far more than the markers' spacing, so the
    offset is found first: every a - b difference up to ``max_offset_nm``
    is tried, and the one that brings the most markers within
    ``tolerance_nm`` of a partner wins. It must line up at least
    ``MATCH_MIN_SHARED_FRACTION`` of the smaller set (and two markers) when
    both sides have more than one; otherwise the fields do not share their
    markers and the best offset is a coincidence. Every pair of the right
    offset proposes nearly the same difference, so the winners agree; when
    they do not -- two different offsets explain the markers equally well,
    as with one marker on one side and several on the other -- nothing is
    paired, because any choice would be a guess. Pairs are then mutual
    nearest neighbours under the winning offset.
    """
    if not a or not b:
        return []
    pa = np.array([[f.x_nm, f.y_nm] for f in a])
    pb = np.array([[f.x_nm, f.y_nm] for f in b])
    tree_a = cKDTree(pa)

    offsets: List[NDArray[np.float64]] = []
    support: List[int] = []
    for i in range(len(pa)):
        for j in range(len(pb)):
            d = pa[i] - pb[j]
            if float(np.hypot(*d)) > max_offset_nm:
                continue
            dist, _ = tree_a.query(pb + d, k=1,
                                   distance_upper_bound=tolerance_nm)
            offsets.append(d)
            support.append(int(np.count_nonzero(np.isfinite(dist))))
    if not offsets:
        return []
    top = max(support)
    # With markers on both sides, the right offset lines up more than one,
    # and most of those the smaller side holds.
    smaller = min(len(pa), len(pb))
    if smaller > 1 and top < max(
            2, math.ceil(MATCH_MIN_SHARED_FRACTION * smaller)):
        return []
    winners = np.array([d for d, n in zip(offsets, support) if n == top])
    spread = np.hypot(*(winners - winners[0]).T)
    if np.any(spread > tolerance_nm):
        return []
    best = np.median(winners, axis=0)

    moved = pb + best
    tree_b = cKDTree(moved)
    d_ab, j_of_a = tree_b.query(pa, k=1, distance_upper_bound=tolerance_nm)
    d_ba, i_of_b = tree_a.query(moved, k=1, distance_upper_bound=tolerance_nm)
    out = []
    for i, (dist, j) in enumerate(zip(d_ab, j_of_a)):
        if np.isfinite(dist) and np.isfinite(d_ba[j]) and i_of_b[j] == i:
            out.append((i, int(j)))
    return out


# ===================================================================
#  Measuring the registration
# ===================================================================
def _loo_residuals(offsets: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each offset minus the mean of all the others."""
    n = len(offsets)
    return np.asarray((offsets - offsets.mean(axis=0)) * (n / (n - 1)),
                      dtype=float)


def register_from_fiducials(
    fiducials_a: Sequence[Fiducial],
    fiducials_b: Sequence[Fiducial],
    *,
    tolerance_nm: float = MATCH_TOLERANCE_NM,
    max_offset_nm: float = MATCH_MAX_OFFSET_NM,
) -> Registration:
    """
    Translation of channel B onto channel A, and its error, from markers.

    A translation, not an affine: Exchange-PAINT rounds share the camera,
    the dye and the optical path, so what differs between them is where
    the sample sat. The shift is the mean of the pair offsets. The error is
    the leave-one-out RMS of those offsets, laterally and in z.

    A z shift measured on markers at the coverslip holds at the depth of
    the sample only because a focus difference between rounds moves the
    whole volume together.
    """
    warnings: List[str] = []
    pairs = match_fiducials(fiducials_a, fiducials_b,
                            max_offset_nm=max_offset_nm,
                            tolerance_nm=tolerance_nm)
    n_a, n_b, n = len(fiducials_a), len(fiducials_b), len(pairs)
    head = (f"{n} marker{'s' if n != 1 else ''} found in both channels "
            f"({n_a} in A, {n_b} in B)")
    if n == 0:
        if n_a and n_b:
            reason = (f"the markers of the two channels ({n_a} in A, {n_b} "
                      f"in B) could not be paired")
            detail = (f"No single shift of up to {max_offset_nm:.0f} nm "
                      f"brings at least half of them within "
                      f"{tolerance_nm:.0f} nm of a partner, or two shifts "
                      f"fit equally well. The files may not show the same "
                      f"field, or the markers moved between the rounds.")
        else:
            reason = ("no markers were found in "
                      + ("either channel" if not n_a and not n_b
                         else "channel A" if not n_a else "channel B"))
            detail = (f"A marker must be localized in at least "
                      f"{FIDUCIAL_MIN_FRAME_FRACTION:.0%} of the frames of "
                      f"both rounds (drift-corrected).")
        return Registration(
            source="none",
            description=f"Not registered: {reason}.",
            warnings=[
                f"The registration could not be measured: {reason}. "
                f"{detail} Channel B is used as loaded."],
        )
    if n < min(n_a, n_b):
        warnings.append(
            f"{min(n_a, n_b) - n} marker(s) had no partner in the other "
            f"channel and were not used.")

    both_3d = all(fiducials_a[i].z_nm is not None
                  and fiducials_b[j].z_nm is not None for i, j in pairs)
    offsets = np.array([
        [fiducials_a[i].x_nm - fiducials_b[j].x_nm,
         fiducials_a[i].y_nm - fiducials_b[j].y_nm,
         (fiducials_a[i].z_nm - fiducials_b[j].z_nm) if both_3d else 0.0]
        for i, j in pairs
    ], dtype=float)
    shift = offsets.mean(axis=0)
    shift_t = (float(shift[0]), float(shift[1]), float(shift[2]))

    lateral_rms = axial_rms = None
    positions = [(fiducials_a[i].x_nm, fiducials_a[i].y_nm) for i, _ in pairs]
    residuals: List[Tuple[float, float, float]] = []
    if n >= MIN_PAIRS_FOR_ERROR:
        loo = _loo_residuals(offsets)
        residuals = [(float(r[0]), float(r[1]), float(r[2])) for r in loo]
        lateral_rms = float(np.sqrt(np.mean(loo[:, 0] ** 2 + loo[:, 1] ** 2)))
        if both_3d:
            axial_rms = float(np.sqrt(np.mean(loo[:, 2] ** 2)))
        spread = float(np.median([fiducials_a[i].spread_nm
                                  for i, _ in pairs]))
        if lateral_rms > 3 * max(spread, 1.0):
            warnings.append(
                f"The markers disagree on the shift by {lateral_rms:.0f} nm "
                f"(RMS), far more than each one's own spread "
                f"({spread:.0f} nm). The rounds differ by more than a "
                f"translation -- a residual drift, or a marker that moved.")
    else:
        warnings.append(
            f"Only {n} marker pair(s): the shift is applied, but its error "
            f"cannot be estimated (leave-one-out needs "
            f"{MIN_PAIRS_FOR_ERROR})"
            + (", and a single pair cannot be checked against any other."
               if n == 1 else "."))
    if not both_3d:
        warnings.append(
            "The markers carry no z in at least one channel: the axial "
            "registration is unknown and no z shift is applied.")

    parts = [head, f"shift ({shift_t[0]:+.1f}, {shift_t[1]:+.1f}"
             + (f", {shift_t[2]:+.1f}" if both_3d else "") + ") nm"]
    if lateral_rms is not None:
        parts.append(f"leave-one-out error {lateral_rms:.1f} nm laterally"
                     + (f", {axial_rms:.1f} nm in z" if axial_rms is not None
                        else ""))
    return Registration(
        source="fiducials", shift_nm=shift_t,
        lateral_rms_nm=lateral_rms, axial_rms_nm=axial_rms,
        n_pairs=n, description="; ".join(parts) + ".",
        warnings=warnings, pair_positions_nm=positions,
        pair_residuals_nm=residuals,
    )


def register_localizations(loc_a: Any, loc_b: Any,
                           **kwargs: Any) -> Registration:
    """Markers found in two loaded files, then ``register_from_fiducials``."""
    find_kwargs = {k: kwargs.pop(k) for k in ("radius_nm",
                                              "min_frame_fraction")
                   if k in kwargs}
    reg = register_from_fiducials(fiducials_in(loc_a, **find_kwargs),
                                  fiducials_in(loc_b, **find_kwargs),
                                  **kwargs)
    reg.paths = [str(loc_a.path), str(loc_b.path)]
    note = pixel_size_disagreement(
        os.path.basename(str(loc_a.path)), loc_a.pixel_size_nm,
        os.path.basename(str(loc_b.path)), loc_b.pixel_size_nm,
        reach_nm(loc_a, loc_b))
    if note:
        reg.warnings.append(note)
    return reg


# ===================================================================
#  Reading the calibration tool's output
# ===================================================================
def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _yaml(path: str) -> Any:
    import yaml  # type: ignore[import-untyped]

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _calibration_from_json(path: str) -> Registration:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or \
            data.get("tool") != "smlm-affine-calibrator":
        raise ValueError(
            f"{os.path.basename(path)} is not a calibration_matrix.json "
            f"written by matriz-transformacion.")
    metrics = data.get("metrics") or {}
    rms = _finite(metrics.get("rms_nm"))
    if rms is None:
        raise ValueError(f"{os.path.basename(path)} has no rms_nm.")
    n = int(metrics.get("n_pairs_used") or 0)
    return Registration(
        source="calibration", lateral_rms_nm=rms, n_pairs=n,
        description=(f"Chromatic calibration: {rms:.1f} nm RMS over {n} "
                     f"bead pairs (in-sample)."),
        paths=[path],
        warnings=[
            "This calibration reports only the residual of the fit on its "
            "own beads, after dropping outliers, which underestimates the "
            "error on the sample. The tool's .hdf5 flow writes a "
            "leave-one-out error (matrix.yaml, qc_summary) that does not."],
    )


def _pixel_size_from(data: Any) -> Optional[float]:
    camera = data.get("camera_info") if isinstance(data, dict) else None
    if isinstance(camera, dict):
        return _finite(camera.get("pixel_size_nm"))
    return None


def _from_qc(
    tre_median_px: Any,
    rmse_px: Any,
    pixel_size_nm: float,
    n: int,
    verdict: Optional[str],
    path: str,
) -> Registration:
    warnings: List[str] = []
    tre = _finite(tre_median_px)
    rmse = _finite(rmse_px)
    if tre is not None:
        rms = tre * pixel_size_nm * RMS_PER_MEDIAN_2D
        how = (f"leave-one-out median {tre * pixel_size_nm:.1f} nm, taken as "
               f"{rms:.1f} nm RMS")
    elif rmse is not None:
        rms = rmse * pixel_size_nm
        how = f"{rms:.1f} nm RMS (in-sample)"
        reason = ("it needs at least 4 bead pairs" if n < 4
                  else "none was computed")
        warnings.append(
            f"The calibration has no leave-one-out error ({reason}); its "
            f"in-sample RMS underestimates the error on the sample.")
        # The tool derives its rating from that error, and rates a missing
        # one 'poor': the rating says nothing here.
        verdict = None
    else:
        raise ValueError(
            f"{os.path.basename(path)} has neither a leave-one-out error "
            f"nor an RMS.")
    if verdict in ("marginal", "poor"):
        warnings.append(f"The calibration tool rated this matrix "
                        f"'{verdict}'.")
    return Registration(
        source="calibration", lateral_rms_nm=rms, n_pairs=n,
        description=(f"Chromatic calibration: {how}, {n} bead pairs"
                     + (f", rated '{verdict}'" if verdict else "") + "."),
        paths=[path], warnings=warnings,
    )


def read_calibration(path: str,
                     pixel_size_nm: Optional[float] = None) -> Registration:
    """
    The lateral registration error recorded by matriz-transformacion.

    Accepts its ``calibration_matrix.json`` (TIFF flow), its ``matrix.yaml``
    (the .hdf5 flow's carries the quality summary; the TIFF flow's does
    not, and the JSON beside it is read instead) or its
    ``qc_metrics.yaml``. The tool works in camera pixels; the pixel size is
    taken from the file, else from a ``matrix.yaml`` beside it, else from
    ``pixel_size_nm``.

    The matrix is not applied here: channel B is assumed to have been
    corrected with it already. No z information exists in these files.
    """
    folder = os.path.dirname(os.path.abspath(path))
    if path.lower().endswith(".json"):
        return _calibration_from_json(path)
    data = _yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{os.path.basename(path)} is not a calibration "
                         f"file.")

    if "qc_summary" in data:
        qc = data["qc_summary"] or {}
        px = _pixel_size_from(data) or pixel_size_nm
        if not px:
            raise ValueError(
                f"{os.path.basename(path)} records no pixel size; give it "
                f"explicitly.")
        return _from_qc(qc.get("tre_loo_median_px"), qc.get("rmse_px"), px,
                        int(qc.get("n_inliers") or 0), qc.get("verdict"),
                        path)

    if "tre_loo" in data and "rmse_px" in data:
        px = None
        beside = os.path.join(folder, "matrix.yaml")
        if os.path.exists(beside):
            px = _pixel_size_from(_yaml(beside))
        px = px or pixel_size_nm
        if not px:
            raise ValueError(
                f"{os.path.basename(path)} is in camera pixels and no pixel "
                f"size was found beside it; give it explicitly.")
        tre = data.get("tre_loo") or {}
        return _from_qc(tre.get("median_px"), data.get("rmse_px"), px,
                        int(data.get("n_inliers") or 0), None, path)

    if "matrix" in data or "coefficients" in data:
        beside = os.path.join(folder, "calibration_matrix.json")
        if os.path.exists(beside):
            return _calibration_from_json(beside)
        raise ValueError(
            f"{os.path.basename(path)} holds a matrix but no quality "
            f"metrics. Load the calibration_matrix.json or qc_metrics.yaml "
            f"written with it.")

    raise ValueError(f"{os.path.basename(path)} is not a matriz-"
                     f"transformacion output.")


# ===================================================================
#  Combining sources
# ===================================================================
def combine(fiducials: Registration, calibration: Registration) -> Registration:
    """
    Both sources at once: the round-to-round shift and its error from the
    markers, plus the chromatic residual, added in quadrature.

    Without any marker pair the result stays unregistered: for
    Exchange-PAINT the calibration does not describe the offset between
    rounds, so its error alone would be passed off as the total.
    """
    if fiducials.n_pairs == 0:
        return Registration(
            source="none",
            description=(f"{fiducials.description} The calibration was not "
                         f"used on its own."),
            paths=list(fiducials.paths) + list(calibration.paths),
            warnings=list(fiducials.warnings) + [
                "No markers were found, so the calibration's error was not "
                "taken as the whole registration error: for sequential "
                "rounds it does not describe the offset between them. If "
                "the two channels were imaged at the same time, choose 'Do "
                "not use markers' to use the calibration alone."],
        )
    lateral: Optional[float] = None
    if fiducials.lateral_rms_nm is not None and \
            calibration.lateral_rms_nm is not None:
        lateral = math.hypot(fiducials.lateral_rms_nm,
                             calibration.lateral_rms_nm)
    warnings = list(fiducials.warnings) + list(calibration.warnings)
    if fiducials.lateral_rms_nm is None:
        warnings.append(
            "The lateral error is left unknown: the markers could not "
            "estimate theirs, and the calibration alone would understate "
            "it.")
    return Registration(
        source="combined", shift_nm=fiducials.shift_nm,
        lateral_rms_nm=lateral, axial_rms_nm=fiducials.axial_rms_nm,
        n_pairs=fiducials.n_pairs,
        description=(f"{fiducials.description} {calibration.description} "
                     f"Lateral errors added in quadrature."
                     if lateral is not None else
                     f"{fiducials.description} {calibration.description}"),
        paths=list(fiducials.paths) + list(calibration.paths),
        warnings=warnings,
        pair_positions_nm=list(fiducials.pair_positions_nm),
        pair_residuals_nm=list(fiducials.pair_residuals_nm),
    )


def export_registration(reg: Registration) -> Dict[str, Any]:
    """Flat fields for a CSV row."""
    return {
        "registration_source": reg.source,
        "registration_shift_x_nm": round(reg.shift_nm[0], 2),
        "registration_shift_y_nm": round(reg.shift_nm[1], 2),
        "registration_shift_z_nm": round(reg.shift_nm[2], 2),
        "registration_lateral_rms_nm": reg.lateral_rms_nm,
        "registration_axial_rms_nm": reg.axial_rms_nm,
        "registration_n_pairs": reg.n_pairs,
    }
