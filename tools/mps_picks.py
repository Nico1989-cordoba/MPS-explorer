# -*- coding: utf-8 -*-
"""
Picasso pick regions: reading the file Render's "Save picks" writes, and
finding the pick a per-axon localization file was cut with.

Why this exists (decision P-R59)
--------------------------------
An axon is analysed from a localization file the user cut out of the
whole field in Picasso Render with a pick (usually a polygon drawn by
hand). Where the pick's edge runs through a cluster, only part of that
cluster's localizations reach the file. ``tools.mps_cut_clusters``
flags those clusters; it needs the pick's edge, and this module reads
it. Without the edge the rule is not applied, and says so -- the edge
is never guessed from the localizations.

The two files Picasso writes (v0.9.10, read from its source,
``picasso/gui/render.py``)
------------------------------------------------------------------
* "Save picks" writes a YAML mapping with the regions themselves, in
  CAMERA PIXELS (the frame of the localizations' x and y):

  - ``Shape: Polygon`` and ``Vertices``: one list of [x, y] corners per
    pick; a closed polygon repeats its first corner at the end, and an
    unclosed one (still being drawn) is skipped by Picasso when it picks;
  - ``Shape: Circle``, ``Centers`` ([x, y] per pick) and ``Diameter (nm)``
    (one value for every pick); files written before ``Diameter (nm)``
    existed carry ``Diameter`` in camera pixels, and the oldest ones carry
    no ``Shape`` at all (Picasso reads those as circles);
  - ``Shape: Rectangle``, ``Center-Axis-Points`` ([[x0, y0], [x1, y1]] per
    pick) and ``Width (nm)`` (or ``Width`` in camera pixels);
  - ``Shape: Square``, ``Centers`` and ``Side Length (nm)``.

* "Save picked localizations" writes the localizations inside the picks,
  with a ``group`` column holding the INDEX of the pick each came from in
  the list above, and adds one step to the metadata: ``Pick Shape``,
  ``Number of picks``, ``Pick Areas (um^2)`` (every pick's area) and, for
  circles, ``Pick Diameter (nm)``. "Save picked localizations (one file
  per pick)" writes ``Area (um^2)`` of that one pick instead of the list.
  Neither writes the regions themselves: the edge is only in the "Save
  picks" file, which is why that file is looked for.

Which pick a file came from
---------------------------
The candidates are the pick files beside the per-axon file and one
folder up (or the file the user points to). A pick is a candidate when
EVERY localization of the file lies inside it (Picasso's own test,
strict, with a 0.001 camera-pixel tolerance for the float32 storage of
x and y). It is a STRONG candidate when, besides, nothing the per-axon
file records about its pick contradicts it: the ``group`` value of its
localizations is the pick's index, the recorded shape is the pick's
shape, and the recorded area agrees with the pick's own area within 1 %
(the one pick's area of a file saved one pick per file; with the list of
every pick's area, the entry at the pick's place -- a check that the
pick file is the one the picks were saved with, which by itself does not
tell two of its picks apart). One strong candidate, or several with the same
geometry (copies of one pick file), is a match. Anything else --
several different strong candidates, or only candidates something
contradicts -- is AMBIGUOUS, and the user is asked; no candidate at all
is "none". Nothing is chosen by a score.

No Qt here: the matching is tested headlessly (test_cut_clusters.py).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PICK_SHAPES", "PickFileError", "Pick", "PickFile", "PickBoundary",
    "PickCandidate", "PickMatch", "read_pick_file", "looks_like_pick_file",
    "candidate_files", "match_pick", "picked_record", "CONTAINS_TOLERANCE_PX",
    "AREA_TOLERANCE",
]

PICK_SHAPES: Tuple[str, ...] = ("Polygon", "Circle", "Rectangle", "Square")

# A localization ON the edge counts as inside for the match. Picasso
# stores x and y as float32: a coordinate of ~1000 camera pixels is kept
# to ~6e-5 px, so 1e-3 px (0.13 nm at 130 nm pixels) absorbs the storage
# and nothing else. Derivation, not a fit.
CONTAINS_TOLERANCE_PX = 1e-3
# Relative tolerance of the area check. Picasso writes the area in um^2
# as a Python float from the same corners: the only differences are the
# pixel size used to convert it (the display's, which can be retyped) and
# float rounding. 1 % lets rounding through and stops a different pick
# (ad hoc; two hand-drawn picks of one axon differ by far more).
AREA_TOLERANCE = 0.01
# Pick files are small (a few kB); a larger YAML is something else.
_MAX_PICK_FILE_BYTES = 5_000_000
# A folder with more YAML files than this is not searched file by file.
_MAX_YAML_FILES = 400


class PickFileError(ValueError):
    """The file is not a Picasso pick file, or cannot be used."""


# --------------------------------------------------------------------------- geometry helpers
def _segment_distance(points: NDArray[np.float64], a: NDArray[np.float64],
                      b: NDArray[np.float64]) -> NDArray[np.float64]:
    seg = b - a
    length2 = float(np.dot(seg, seg))
    if length2 == 0.0:
        return np.asarray(np.hypot(points[:, 0] - a[0], points[:, 1] - a[1]), dtype=float)
    t = np.clip(((points - a) @ seg) / length2, 0.0, 1.0)
    proj = a + t[:, None] * seg
    return np.asarray(np.hypot(points[:, 0] - proj[:, 0], points[:, 1] - proj[:, 1]), dtype=float)


def polygon_edge_distance(x: NDArray[np.float64], y: NDArray[np.float64],
                          vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unsigned distance of each point to the closed polygon's edges."""
    pts = np.column_stack([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()])
    v = np.asarray(vertices, float).reshape(-1, 2)
    out = np.full(len(pts), np.inf)
    for i in range(len(v)):
        out = np.minimum(out, _segment_distance(pts, v[i], v[(i + 1) % len(v)]))
    return out


def polygon_contains(x: NDArray[np.float64], y: NDArray[np.float64],
                     vertices: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Even-odd ray casting, the rule of Picasso's ``check_if_in_polygon``."""
    xs = np.asarray(x, float).ravel()
    ys = np.asarray(y, float).ravel()
    v = np.asarray(vertices, float).reshape(-1, 2)
    inside = np.zeros(xs.size, dtype=bool)
    n = len(v)
    for i in range(n):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % n]
        crosses = (y1 > ys) != (y2 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= crosses & (xs < at)
    return inside


def polygon_area(vertices: NDArray[np.float64]) -> float:
    """Shoelace area of a closed polygon (Picasso's ``polygon_area``)."""
    v = np.asarray(vertices, float).reshape(-1, 2)
    x, y = v[:, 0], v[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0)


def _rectangle_corners(start: Sequence[float], end: Sequence[float], width: float) -> NDArray[np.float64]:
    """Picasso's ``get_pick_rectangle_corners``: (4, 2) corners, camera pixels."""
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    alpha = math.pi / 2 if ex == sx else math.atan((ey - sy) / (ex - sx))
    dx = width * math.sin(alpha) / 2
    dy = width * math.cos(alpha) / 2
    return np.array([[sx - dx, sy + dy], [sx + dx, sy - dy], [ex + dx, ey - dy], [ex - dx, ey + dy]])


# --------------------------------------------------------------------------- one pick
@dataclass(frozen=True)
class Pick:
    """One region of a pick file, in camera pixels.

    ``index`` is its position in the file's list: the ``group`` value
    Picasso gives the localizations it picked. ``closed_rank`` is its
    position among the picks Picasso can pick with (an unclosed polygon
    is skipped), the index of its entry in ``Pick Areas (um^2)``.
    """

    index: int
    shape: str
    vertices_px: Optional[NDArray[np.float64]] = None    # polygon, rectangle, square: (n, 2), not repeated
    centre_px: Optional[Tuple[float, float]] = None       # circle
    radius_px: Optional[float] = None                     # circle
    closed: bool = True
    closed_rank: int = -1

    def area_px2(self) -> float:
        if self.shape == "Circle" and self.radius_px is not None:
            return float(math.pi * self.radius_px ** 2)
        if self.vertices_px is None:
            return float("nan")
        return polygon_area(self.vertices_px)

    def contains_px(self, x: NDArray[np.float64], y: NDArray[np.float64],
                    tolerance_px: float = 0.0) -> NDArray[np.bool_]:
        """Inside the pick, or within ``tolerance_px`` of its edge."""
        xs = np.asarray(x, float).ravel()
        ys = np.asarray(y, float).ravel()
        if self.shape == "Circle":
            assert self.centre_px is not None and self.radius_px is not None
            r = np.hypot(xs - self.centre_px[0], ys - self.centre_px[1])
            out: NDArray[np.bool_] = r < self.radius_px + tolerance_px
            return out
        assert self.vertices_px is not None
        inside = polygon_contains(xs, ys, self.vertices_px)
        if tolerance_px > 0 and not inside.all():
            near = polygon_edge_distance(xs[~inside], ys[~inside], self.vertices_px) <= tolerance_px
            inside = inside.copy()
            inside[np.flatnonzero(~inside)[near]] = True
        return inside

    def same_geometry(self, other: "Pick") -> bool:
        if self.shape != other.shape:
            return False
        if self.shape == "Circle":
            return (self.centre_px is not None and other.centre_px is not None
                    and np.allclose(self.centre_px, other.centre_px) and self.radius_px is not None
                    and other.radius_px is not None and math.isclose(self.radius_px, other.radius_px))
        a, b = self.vertices_px, other.vertices_px
        return a is not None and b is not None and a.shape == b.shape and bool(np.allclose(a, b))

    def boundary(self, pixel_size_nm: float, source: str) -> "PickBoundary":
        """The pick's edge in nanometres (the frame of the loaded localizations)."""
        px = float(pixel_size_nm)
        if self.shape == "Circle":
            assert self.centre_px is not None and self.radius_px is not None
            return PickBoundary(kind="circle", shape=self.shape, centre_nm=(self.centre_px[0] * px,
                                                                           self.centre_px[1] * px),
                                radius_nm=self.radius_px * px, source=source)
        assert self.vertices_px is not None
        return PickBoundary(kind="polygon", shape=self.shape, vertices_nm=np.asarray(self.vertices_px) * px,
                            source=source)


@dataclass
class PickFile:
    """A pick file as read: its picks, in the file's order."""

    path: str
    shape: str
    picks: List[Pick]
    size_words: str = ""                 # how the diameter / width was given ("Diameter (nm) 400", ...)
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- the edge, in nm
@dataclass(frozen=True)
class PickBoundary:
    """The edge of the pick an axon was cut with, in the localizations' nm.

    ``kind`` "polygon" (a polygon, rectangle or square pick: its corners)
    or "circle". ``source`` says where it came from, in words, for the
    table and the panel.
    """

    kind: str
    shape: str
    source: str
    vertices_nm: Optional[NDArray[np.float64]] = None
    centre_nm: Optional[Tuple[float, float]] = None
    radius_nm: Optional[float] = None

    def distance_nm(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Unsigned distance of each point to the edge."""
        xs = np.asarray(x, float).ravel()
        ys = np.asarray(y, float).ravel()
        if self.kind == "circle":
            assert self.centre_nm is not None and self.radius_nm is not None
            return np.asarray(np.abs(self.radius_nm - np.hypot(xs - self.centre_nm[0], ys - self.centre_nm[1])),
                              dtype=float)
        assert self.vertices_nm is not None
        return polygon_edge_distance(xs, ys, self.vertices_nm)

    def contains_nm(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.bool_]:
        xs = np.asarray(x, float).ravel()
        ys = np.asarray(y, float).ravel()
        if self.kind == "circle":
            assert self.centre_nm is not None and self.radius_nm is not None
            out: NDArray[np.bool_] = np.hypot(xs - self.centre_nm[0], ys - self.centre_nm[1]) < self.radius_nm
            return out
        assert self.vertices_nm is not None
        return polygon_contains(xs, ys, self.vertices_nm)

    def outline_nm(self, n_circle: int = 181) -> NDArray[np.float64]:
        """A closed polyline of the edge, first point repeated, for drawing."""
        if self.kind == "circle":
            assert self.centre_nm is not None and self.radius_nm is not None
            t = np.linspace(0.0, 2.0 * np.pi, n_circle)
            return np.column_stack([self.centre_nm[0] + self.radius_nm * np.cos(t),
                                    self.centre_nm[1] + self.radius_nm * np.sin(t)])
        assert self.vertices_nm is not None
        v = np.asarray(self.vertices_nm, float).reshape(-1, 2)
        return np.vstack([v, v[:1]])


# --------------------------------------------------------------------------- reading
def _load_yaml(path: str) -> Any:
    import yaml  # type: ignore[import-untyped, unused-ignore]

    if os.path.getsize(path) > _MAX_PICK_FILE_BYTES:
        raise PickFileError(f"{os.path.basename(path)} is too large to be a pick file")
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        docs = [d for d in yaml.safe_load_all(handle)]
    if len(docs) != 1:
        raise PickFileError(f"{os.path.basename(path)} holds {len(docs)} YAML documents; a pick file holds one "
                            f"(a file with several is a localization file's metadata)")
    return docs[0]


def _shape_of(doc: Dict[str, Any]) -> Optional[str]:
    if "Shape" in doc:
        return str(doc["Shape"])
    # Picasso's backwards compatibility: no Shape, Centers and Diameter -> circles.
    if "Centers" in doc and "Diameter" in doc:
        return "Circle"
    return None


def looks_like_pick_file(path: str) -> bool:
    """Whether ``path`` parses as a Picasso pick file (never raises)."""
    try:
        doc = _load_yaml(path)
    except Exception:                                        # noqa: BLE001 - any failure: not a pick file
        return False
    return isinstance(doc, dict) and _shape_of(doc) in PICK_SHAPES


def _size_px(doc: Dict[str, Any], nm_key: str, px_key: str, pixel_size_nm: Optional[float],
             what: str) -> Tuple[float, str]:
    if nm_key in doc:
        if pixel_size_nm is None or not pixel_size_nm > 0:
            raise PickFileError(f"the {what} is given in nm ({nm_key}) and no pixel size is known to convert it")
        value = float(doc[nm_key])
        return value / float(pixel_size_nm), f"{nm_key} {value:g}"
    if px_key in doc:
        value = float(doc[px_key])
        return value, f"{px_key} {value:g} (camera pixels, an older Picasso)"
    raise PickFileError(f"no {nm_key} nor {px_key} in the file")


def _xy(value: Any) -> Tuple[float, float]:
    seq = list(value)
    if len(seq) != 2:
        raise PickFileError(f"a point has {len(seq)} coordinates, not 2")
    return float(seq[0]), float(seq[1])


def read_pick_file(path: str, pixel_size_nm: Optional[float] = None) -> PickFile:
    """Every pick of a Picasso pick file, in camera pixels.

    ``pixel_size_nm`` converts a diameter, width or side given in nm
    (Picasso >= 0.7) into camera pixels; it is the pixel size of the
    localizations the picks are compared with. Raises ``PickFileError``
    for a file that is not a pick file or cannot be read.
    """
    try:
        doc = _load_yaml(path)
    except PickFileError:
        raise
    except Exception as error:                               # noqa: BLE001
        raise PickFileError(f"{os.path.basename(path)} could not be read as YAML ({error})") from error
    if not isinstance(doc, dict):
        raise PickFileError(f"{os.path.basename(path)} is not a Picasso pick file (not a mapping)")
    shape = _shape_of(doc)
    if shape not in PICK_SHAPES:
        raise PickFileError(f"{os.path.basename(path)} is not a Picasso pick file "
                            f"(shape {shape!r}; expected one of {', '.join(PICK_SHAPES)})")
    out = PickFile(path=path, shape=str(shape), picks=[])
    try:
        if shape == "Polygon":
            rank = 0
            for i, raw in enumerate(doc.get("Vertices") or []):
                pts = [_xy(p) for p in (raw or [])]
                closed = len(pts) >= 3 and pts[0] == pts[-1]
                if closed:
                    corners = np.asarray(pts[:-1], dtype=float)
                    out.picks.append(Pick(index=i, shape=shape, vertices_px=corners, closed=True,
                                          closed_rank=rank))
                    rank += 1
                else:
                    out.picks.append(Pick(index=i, shape=shape,
                                          vertices_px=np.asarray(pts, float).reshape(-1, 2),
                                          closed=False))
                    out.warnings.append(f"pick {i} is an unclosed polygon: Picasso picks nothing with it")
        elif shape == "Circle":
            diameter_px, out.size_words = _size_px(doc, "Diameter (nm)", "Diameter", pixel_size_nm, "diameter")
            for i, c in enumerate(doc.get("Centers") or []):
                out.picks.append(Pick(index=i, shape=shape, centre_px=_xy(c), radius_px=diameter_px / 2.0,
                                      closed_rank=i))
        elif shape == "Rectangle":
            width_px, out.size_words = _size_px(doc, "Width (nm)", "Width", pixel_size_nm, "width")
            for i, pair in enumerate(doc.get("Center-Axis-Points") or []):
                start, end = list(pair)
                out.picks.append(Pick(index=i, shape=shape,
                                      vertices_px=_rectangle_corners(_xy(start), _xy(end), width_px),
                                      closed_rank=i))
        else:   # Square: axis-aligned, Picasso's postprocess.picked_locs
            side_px, out.size_words = _size_px(doc, "Side Length (nm)", "Side Length", pixel_size_nm, "side")
            h = side_px / 2.0
            for i, c in enumerate(doc.get("Centers") or []):
                x, y = _xy(c)
                out.picks.append(Pick(index=i, shape=shape,
                                      vertices_px=np.array([[x - h, y - h], [x + h, y - h], [x + h, y + h],
                                                            [x - h, y + h]]), closed_rank=i))
    except PickFileError as error:
        raise PickFileError(f"{os.path.basename(path)}: {error}") from error
    except (TypeError, ValueError) as error:
        raise PickFileError(f"{os.path.basename(path)}: malformed picks ({error})") from error
    if not out.picks:
        raise PickFileError(f"{os.path.basename(path)} holds no picks")
    return out


# --------------------------------------------------------------------------- what the picked file says
@dataclass
class PickedRecord:
    """What a per-axon file records about the pick it was cut with: the
    ``group`` values of its localizations and the metadata step Picasso
    added ("Save picked localizations"). Every field may be missing."""

    groups: Tuple[int, ...] = ()
    shape: Optional[str] = None
    n_picks: Optional[int] = None
    areas_um2: Tuple[float, ...] = ()
    own_area_um2: Optional[float] = None     # one pick per file: that pick's area
    diameter_nm: Optional[float] = None


def picked_record(group: Optional[NDArray[Any]], info: Sequence[Dict[str, Any]]) -> PickedRecord:
    """The record of a loaded per-axon file (``group`` column and metadata chain)."""
    rec = PickedRecord()
    if group is not None and np.asarray(group).size:
        rec.groups = tuple(int(g) for g in np.unique(np.asarray(group).astype(np.int64)))
    step = None
    for doc in reversed(list(info or [])):
        if isinstance(doc, dict) and "Pick Shape" in doc:
            step = doc
            break
    if step is not None:
        rec.shape = str(step.get("Pick Shape"))
        try:
            rec.n_picks = int(step["Number of picks"]) if "Number of picks" in step else None
        except (TypeError, ValueError):
            rec.n_picks = None
        try:
            rec.areas_um2 = tuple(float(a) for a in (step.get("Pick Areas (um^2)") or []))
        except (TypeError, ValueError):
            rec.areas_um2 = ()
        if "Pick Areas (um^2)" not in step and "Area (um^2)" in step:
            try:
                rec.own_area_um2 = float(step["Area (um^2)"])
            except (TypeError, ValueError):
                rec.own_area_um2 = None
        try:
            d = step.get("Pick Diameter (nm)")
            rec.diameter_nm = None if d is None else float(d)
        except (TypeError, ValueError):
            rec.diameter_nm = None
    return rec


# --------------------------------------------------------------------------- matching
@dataclass
class PickCandidate:
    """One pick that holds every localization of the file, and what the
    file's own record says about it (None: the record says nothing)."""

    path: str
    pick: Pick
    pick_file: PickFile
    group_agrees: Optional[bool] = None
    shape_agrees: Optional[bool] = None
    area_agrees: Optional[bool] = None

    @property
    def strong(self) -> bool:
        return not any(v is False for v in (self.group_agrees, self.shape_agrees, self.area_agrees))

    def describe(self) -> str:
        checks = []
        for name, value in (("group", self.group_agrees), ("shape", self.shape_agrees),
                            ("area", self.area_agrees)):
            if value is not None:
                checks.append(f"{name} {'agrees' if value else 'DISAGREES'}")
        how = "; ".join(checks) if checks else "the file records nothing to check it against"
        return (f"pick {self.pick.index} ({self.pick.shape.lower()}) of {os.path.basename(self.path)}: holds every "
                f"localization; {how}")


@dataclass
class PickMatch:
    """The outcome of looking for the pick of one per-axon file.

    ``status`` "matched", "ambiguous" (the user must choose among
    ``candidates``) or "none". ``files`` are the pick files read.
    """

    status: str
    chosen: Optional[PickCandidate] = None
    candidates: List[PickCandidate] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    message: str = ""
    by_hand: bool = False

    def boundary(self, pixel_size_nm: float) -> Optional[PickBoundary]:
        if self.chosen is None:
            return None
        how = "chosen by hand" if self.by_hand else "found automatically"
        return self.chosen.pick.boundary(pixel_size_nm, f"{self.chosen.describe()} ({how})")


def candidate_files(per_axon_path: str, extra: Sequence[str] = ()) -> List[str]:
    """The pick files that could hold this file's pick: the YAML files
    beside it and one folder up that parse as pick files, plus ``extra``."""
    out: List[str] = []
    here = os.path.dirname(os.path.abspath(per_axon_path))
    own_sidecar = os.path.splitext(os.path.abspath(per_axon_path))[0] + ".yaml"
    for folder in (here, os.path.dirname(here)):
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        yamls = [n for n in names if n.lower().endswith((".yaml", ".yml"))]
        if len(yamls) > _MAX_YAML_FILES:
            continue
        for name in yamls:
            path = os.path.join(folder, name)
            if os.path.abspath(path) == own_sidecar or path in out:
                continue
            if looks_like_pick_file(path):
                out.append(path)
    for path in extra:
        if path and path not in out:
            out.append(path)
    return out


def _check_candidate(cand: PickCandidate, record: PickedRecord, pixel_size_nm: Optional[float]) -> None:
    pick = cand.pick
    if record.groups:
        cand.group_agrees = len(record.groups) == 1 and record.groups[0] == pick.index
    if record.shape:
        cand.shape_agrees = record.shape == pick.shape
    if (record.areas_um2 or record.own_area_um2) and pixel_size_nm:
        recorded: Optional[float] = None
        if record.own_area_um2 is not None:
            recorded = record.own_area_um2
        elif pick.shape in ("Circle", "Square"):
            recorded = record.areas_um2[0]
        elif 0 <= pick.closed_rank < len(record.areas_um2):
            recorded = record.areas_um2[pick.closed_rank]
        if recorded is not None and recorded > 0:
            own = pick.area_px2() * (float(pixel_size_nm) * 1e-3) ** 2
            cand.area_agrees = abs(own - recorded) <= AREA_TOLERANCE * recorded


def match_pick(x_px: NDArray[np.float64], y_px: NDArray[np.float64], files: Sequence[str], *,
               record: Optional[PickedRecord] = None, pixel_size_nm: Optional[float] = None,
               by_hand: bool = False) -> PickMatch:
    """Find the pick, among the picks of ``files``, that this file's
    localizations (``x_px``, ``y_px``: camera pixels) were cut with.

    See the module docstring for the rule. ``by_hand``: the user pointed
    to these files; the rule is the same, only the words say so.
    """
    rec = record or PickedRecord()
    xs = np.asarray(x_px, float).ravel()
    ys = np.asarray(y_px, float).ravel()
    match = PickMatch(status="none", files=list(files), by_hand=by_hand)
    if xs.size == 0:
        match.message = "the file holds no localizations to match a pick with"
        return match
    problems: List[str] = []
    lo_x, hi_x, lo_y, hi_y = xs.min(), xs.max(), ys.min(), ys.max()
    for path in files:
        try:
            pf = read_pick_file(path, pixel_size_nm)
        except PickFileError as error:
            problems.append(str(error))
            continue
        for pick in pf.picks:
            if not pick.closed:
                continue
            # cheap rejection by the bounding box before the polygon test
            if pick.shape == "Circle":
                assert pick.centre_px is not None and pick.radius_px is not None
                cx, cy, r = pick.centre_px[0], pick.centre_px[1], pick.radius_px
                if lo_x < cx - r - 1 or hi_x > cx + r + 1 or lo_y < cy - r - 1 or hi_y > cy + r + 1:
                    continue
            else:
                assert pick.vertices_px is not None
                v = pick.vertices_px
                if lo_x < v[:, 0].min() - 1 or hi_x > v[:, 0].max() + 1 or lo_y < v[:, 1].min() - 1 \
                        or hi_y > v[:, 1].max() + 1:
                    continue
            if not pick.contains_px(xs, ys, CONTAINS_TOLERANCE_PX).all():
                continue
            cand = PickCandidate(path=path, pick=pick, pick_file=pf)
            _check_candidate(cand, rec, pixel_size_nm)
            match.candidates.append(cand)
    strong = [c for c in match.candidates if c.strong]
    if strong and all(c.pick.same_geometry(strong[0].pick) for c in strong):
        match.status, match.chosen = "matched", strong[0]
        copies = "" if len(strong) == 1 else f" ({len(strong)} files hold this same pick)"
        match.message = f"{strong[0].describe()}{copies}"
    elif match.candidates:
        match.status = "ambiguous"
        n = len(match.candidates)
        match.message = (f"{n} picks hold every localization of this file and nothing the file records tells "
                         f"them apart: choose one" if len(strong) > 1 else
                         f"{n} pick(s) hold every localization, but the file's own record contradicts "
                         f"{'it' if n == 1 else 'each of them'}: confirm or choose another pick file")
    else:
        where = f"{len(files)} pick file(s) read" if files else "no Picasso pick file found"
        match.message = f"no pick holds every localization of this file ({where})"
        if problems:
            match.message += f"; {len(problems)} file(s) could not be used: {problems[0]}"
    return match
