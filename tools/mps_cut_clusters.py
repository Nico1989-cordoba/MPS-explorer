# -*- coding: utf-8 -*-
"""
Clusters cut by the pick boundary (decision P-R59; docs/cut_clusters.md).

The problem
-----------
A per-axon file is the selection of a Picasso pick drawn on the whole
field. Where the pick's edge runs through a cluster, the localizations
of that cluster outside the edge never reach the file: the cluster is
measured smaller, with fewer localizations and a centre pulled inwards,
and its occupancy and nearest-neighbour distance move with it.

The rule
--------
A cluster is CUT when ANY of its localizations lies within
``multiple`` x NeNA of the pick boundary (``DEFAULT_NENA_MULTIPLE`` = 1).
NeNA is the localization precision of the channel, measured on the
whole-field, unlinked file (Endesfelder et al. 2014; ``tools.mps_quality
.nena``). The reasoning (docs/cut_clusters.md, section 2): two
localizations of one molecule differ by a 2D Gaussian displacement of
sd sqrt(2) sigma per axis, so a localization at distance d inside the
edge has a sibling beyond the edge with probability
Phi(-d / (sqrt(2) sigma)) -- 24 % at d = sigma, 8 % at 2 sigma -- and a
molecule seen n times multiplies that chance. A localization that close
to the edge is the trace a cut leaves; one farther in is not. No
percentage of clusters decides anything (the user rejected such rules
as arbitrary): every cluster is judged on its own localizations.

What is done with it
--------------------
The cut clusters are FLAGGED and counted ("k of K (x %)"), drawn on the
map, and written to the tables. They are left OUT of the analysis only
when the user ticks "Exclude cut clusters from the analysis" (off by
default); the table says which. K is the number of clusters the
automatic curation kept.

When it is not applied
----------------------
Without a known pick boundary, or without a NeNA for the channel, the
rule is not applied, and the status says why ("boundary unknown: not
evaluated", "NeNA unknown: not evaluated"). Neither is ever guessed.
A boundary that does not hold the analysed localizations is not the
boundary of this file, and is refused the same way.

No Qt here.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.mps_picks import PickBoundary

__all__ = [
    "DEFAULT_NENA_MULTIPLE", "SENSITIVITY_MULTIPLES", "STATUS_EVALUATED", "STATUS_NO_BOUNDARY", "STATUS_NO_NENA",
    "STATUS_NOT_HELD", "CutContext", "CutReport", "NenaSource", "find_cut_clusters", "evaluate", "rule_words",
    "CUT_COLUMNS", "CLUSTER_CUT_COLUMN", "export_columns", "is_linked", "nena_from_localizations",
    "nena_from_file", "sibling_outside_probability", "CutSession", "open_session", "choose_pick_file",
    "choose_candidate", "resolve_nena", "stored_nena", "nena_entry", "clean_store", "considered_labels",
    "not_evaluated",
]

# The decided multiple (P-R59). Derivation: docs/cut_clusters.md section 2;
# own simulation of 0.5, 1 and 2: section 4 (docs/sims/cut_clusters).
DEFAULT_NENA_MULTIPLE = 1.0
# The multiples the documentation reports beside the default; never used
# by the program unless passed explicitly.
SENSITIVITY_MULTIPLES: Tuple[float, ...] = (0.5, 1.0, 2.0)

STATUS_EVALUATED = "evaluated"
STATUS_NO_BOUNDARY = "boundary unknown: not evaluated"
STATUS_NO_NENA = "NeNA unknown: not evaluated"
STATUS_NOT_HELD = "boundary does not hold the localizations: not evaluated"

# A localization farther than this outside the boundary means the boundary
# is not this file's (another pick, another pixel size). 1 nm: a hundred
# times the float32 storage of a coordinate in nm (derivation).
_OUTSIDE_TOLERANCE_NM = 1.0


def rule_words(multiple: float = DEFAULT_NENA_MULTIPLE) -> str:
    """The rule, in the words the table and the panel use."""
    return f"cut if any localization lies within {multiple:g} NeNA of the pick boundary"


def sibling_outside_probability(d_over_sigma: float) -> float:
    """P(a second localization of the same molecule lies beyond the edge),
    for a localization at ``d_over_sigma`` precisions inside a straight
    edge: Phi(-d / (sqrt(2) sigma)) (docs/cut_clusters.md, section 2)."""
    return 0.5 * math.erfc(float(d_over_sigma) / 2.0)


# --------------------------------------------------------------------------- what the analysis is given
@dataclass(frozen=True)
class CutContext:
    """What the rule needs, and where each piece came from.

    ``boundary`` None or ``nena_nm`` None: the rule is not applied, and
    ``reason`` says why in words (the status names which piece is
    missing).
    """

    boundary: Optional[PickBoundary] = None
    nena_nm: Optional[float] = None
    nena_source: str = ""
    reason: str = ""
    multiple: float = DEFAULT_NENA_MULTIPLE

    @property
    def usable(self) -> bool:
        return (self.boundary is not None and self.nena_nm is not None and math.isfinite(self.nena_nm)
                and self.nena_nm > 0)


# --------------------------------------------------------------------------- what it found
@dataclass
class CutReport:
    """The rule's outcome for one analysis."""

    status: str
    reason: str = ""
    multiple: float = DEFAULT_NENA_MULTIPLE
    nena_nm: Optional[float] = None
    nena_source: str = ""
    boundary_source: str = ""
    boundary_shape: str = ""
    cut_labels: FrozenSet[int] = frozenset()
    considered_labels: FrozenSet[int] = frozenset()
    excluded: bool = False
    cut_centroids: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 2)))
    min_distance_nm: Dict[int, float] = field(default_factory=dict)
    boundary: Optional[PickBoundary] = None

    @property
    def evaluated(self) -> bool:
        return self.status == STATUS_EVALUATED

    @property
    def margin_nm(self) -> Optional[float]:
        return None if self.nena_nm is None else float(self.multiple) * float(self.nena_nm)

    @property
    def k(self) -> Optional[int]:
        return len(self.cut_labels) if self.evaluated else None

    @property
    def n_considered(self) -> Optional[int]:
        return len(self.considered_labels) if self.evaluated else None

    @property
    def percent(self) -> Optional[float]:
        if not self.evaluated or not self.considered_labels:
            return None
        return 100.0 * len(self.cut_labels) / len(self.considered_labels)

    def count_words(self) -> str:
        """"k of K (x %)", or the status when not evaluated."""
        if not self.evaluated:
            return self.status
        pct = self.percent
        return (f"{len(self.cut_labels)} of {len(self.considered_labels)}"
                + ("" if pct is None else f" ({pct:.0f} %)"))

    def note_words(self) -> str:
        """One line for the results table's note column."""
        if not self.evaluated:
            return self.reason
        margin = self.margin_nm
        assert margin is not None and self.nena_nm is not None
        what = "left out of the analysis" if self.excluded else "kept in the analysis (flagged only)"
        return (f"{rule_words(self.multiple)}: {margin:.1f} nm (NeNA {self.nena_nm:.1f} nm, {self.nena_source}); "
                f"{what}")


def not_evaluated(status: str, reason: str, context: Optional[CutContext] = None) -> CutReport:
    c = context or CutContext()
    return CutReport(status=status, reason=reason, multiple=c.multiple, nena_nm=c.nena_nm,
                     nena_source=c.nena_source,
                     boundary_source="" if c.boundary is None else c.boundary.source,
                     boundary_shape="" if c.boundary is None else c.boundary.shape, boundary=c.boundary)


# --------------------------------------------------------------------------- the rule
def find_cut_clusters(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], labels: NDArray[np.int64],
                      considered: Iterable[int], boundary: PickBoundary,
                      margin_nm: float) -> Tuple[FrozenSet[int], Dict[int, float]]:
    """The labels among ``considered`` with a localization within
    ``margin_nm`` of ``boundary``, and each considered label's smallest
    distance to it."""
    xs = np.asarray(x_nm, float).ravel()
    ys = np.asarray(y_nm, float).ravel()
    lab = np.asarray(labels).ravel()
    wanted = sorted({int(v) for v in considered} - {-1})
    if not wanted:
        return frozenset(), {}
    mask = np.isin(lab, wanted)
    d = boundary.distance_nm(xs[mask], ys[mask])
    sub = lab[mask]
    closest: Dict[int, float] = {}
    for label in wanted:
        sel = sub == label
        closest[label] = float(d[sel].min()) if sel.any() else float("inf")
    cut = frozenset(label for label, dist in closest.items() if dist <= margin_nm)
    return cut, closest


def evaluate(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], labels: NDArray[np.int64],
             considered: Iterable[int], context: Optional[CutContext], *, exclude: bool = False) -> CutReport:
    """Apply the rule to one analysis' clusters (``considered``: the
    labels the curation kept), or record why it was not applied."""
    if context is None:
        return not_evaluated(STATUS_NO_BOUNDARY, "no Picasso pick boundary was given for this file")
    if context.boundary is None:
        return not_evaluated(STATUS_NO_BOUNDARY, context.reason or "no Picasso pick boundary is known for this "
                                                                   "file", context)
    if context.nena_nm is None or not math.isfinite(context.nena_nm) or context.nena_nm <= 0:
        return not_evaluated(STATUS_NO_NENA, context.reason or "no NeNA is known for this channel: measure it on "
                                                               "the whole-field, unlinked file", context)
    xs = np.asarray(x_nm, float).ravel()
    ys = np.asarray(y_nm, float).ravel()
    if xs.size:
        outside = ~context.boundary.contains_nm(xs, ys)
        if outside.any():
            far = context.boundary.distance_nm(xs[outside], ys[outside]) > _OUTSIDE_TOLERANCE_NM
            if far.any():
                return not_evaluated(
                    STATUS_NOT_HELD,
                    f"{int(far.sum()):,} of the analysed localizations lie outside the pick boundary: it is not "
                    f"the boundary of this file (another pick, or another pixel size)", context)
    margin = float(context.multiple) * float(context.nena_nm)
    kept = frozenset(int(v) for v in considered) - {-1}
    cut, closest = find_cut_clusters(xs, ys, labels, kept, context.boundary, margin)
    lab = np.asarray(labels).ravel()
    centres = [(float(xs[lab == c].mean()), float(ys[lab == c].mean())) for c in sorted(cut)]
    return CutReport(status=STATUS_EVALUATED, reason="", multiple=float(context.multiple),
                     nena_nm=float(context.nena_nm), nena_source=context.nena_source,
                     boundary_source=context.boundary.source, boundary_shape=context.boundary.shape,
                     cut_labels=cut, considered_labels=kept, excluded=bool(exclude),
                     cut_centroids=np.asarray(centres, float).reshape(-1, 2), min_distance_nm=closest,
                     boundary=context.boundary)


# --------------------------------------------------------------------------- the tables
# Written once, at the very end of the axon row (no earlier column moves),
# for every row: a row whose rule was not applied says why.
CUT_COLUMNS: Tuple[str, ...] = (
    "cut_status", "cut_rule", "cut_nena_multiple", "cut_nena_nm", "cut_nena_source", "cut_margin_nm",
    "cut_boundary_source", "cut_boundary_shape", "n_clusters_cut", "n_clusters_cut_of", "cut_percent",
    "cut_excluded")
# The per-cluster column: True / False, empty when the rule was not applied.
CLUSTER_CUT_COLUMN = "cut_by_pick"


def export_columns(report: Optional[CutReport]) -> Dict[str, Any]:
    """The CUT_COLUMNS of one analysis; None: not given to the analysis."""
    r = report if report is not None else not_evaluated(STATUS_NO_BOUNDARY,
                                                        "no Picasso pick boundary was given for this file")
    ev = r.evaluated
    pct = r.percent
    return {
        "cut_status": r.status if ev else f"{r.status} ({r.reason})" if r.reason else r.status,
        "cut_rule": rule_words(r.multiple) if ev else None,
        "cut_nena_multiple": float(r.multiple) if ev else None,
        "cut_nena_nm": None if r.nena_nm is None else round(float(r.nena_nm), 2),
        "cut_nena_source": r.nena_source or None,
        "cut_margin_nm": None if not ev or r.margin_nm is None else round(float(r.margin_nm), 2),
        "cut_boundary_source": r.boundary_source or None,
        "cut_boundary_shape": r.boundary_shape or None,
        "n_clusters_cut": r.k,
        "n_clusters_cut_of": r.n_considered,
        "cut_percent": None if pct is None else round(pct, 2),
        "cut_excluded": bool(r.excluded and ev),
    }


# --------------------------------------------------------------------------- the NeNA the rule uses
@dataclass
class NenaSource:
    """A channel's NeNA and where it came from, or why there is none."""

    nena_nm: Optional[float] = None
    words: str = ""             # "this file (2,345 next-frame pairs)", "whole-field file X.hdf5 (...)"
    n_pairs: int = 0
    reason: str = ""            # "" when there is a value
    source_file: str = ""       # name only

    @property
    def measured(self) -> bool:
        return self.nena_nm is not None and math.isfinite(self.nena_nm) and self.nena_nm > 0


def is_linked(loc: Any) -> bool:
    """Whether a loaded file was linked (Picasso Link): it carries Link's
    ``len`` column or its processing chain names Link. Linking merges the
    consecutive-frame pairs NeNA measures."""
    columns = getattr(loc, "columns", {}) or {}
    if "len" in columns:
        return True
    for step in getattr(loc, "info", []) or []:
        if isinstance(step, dict) and "link" in str(step.get("Generated by", "")).lower():
            return True
    return False


def nena_from_localizations(loc: Any, words: str) -> NenaSource:
    """NeNA of a loaded file (``tools.mps_io.Localizations``), or why not:
    a linked file, no frame numbers, fewer than MIN_NENA_PAIRS next-frame
    pairs (a picked region usually has too few), or a fit that failed."""
    from tools.mps_quality import MIN_NENA_PAIRS, nena

    name = os.path.basename(str(getattr(loc, "path", "") or ""))
    out = NenaSource(source_file=name)
    if is_linked(loc):
        out.reason = (f"{words} is linked (Picasso Link): its consecutive-frame pairs are merged, so NeNA "
                      f"cannot be measured on it")
        return out
    frame = loc.frame if hasattr(loc, "frame") else None
    if frame is None:
        out.reason = f"{words} has no frame numbers, which NeNA needs"
        return out
    result = nena(np.asarray(frame, dtype=np.int64), loc.x_nm, loc.y_nm)
    out.n_pairs = int(result.n_pairs)
    if result.n_pairs < MIN_NENA_PAIRS:
        out.reason = (f"{words} has only {result.n_pairs:,} next-frame neighbour pairs (NeNA needs "
                      f"{MIN_NENA_PAIRS}): measure it on the whole-field file")
        return out
    if not result.converged or not math.isfinite(result.precision_nm) or result.precision_nm <= 0:
        out.reason = f"the NeNA fit did not converge on {words}"
        return out
    out.nena_nm = float(result.precision_nm)
    flags = f"; flags: {', '.join(result.flags)}" if result.flags else ""
    out.words = f"{words} ({result.n_pairs:,} next-frame pairs{flags})"
    return out


def nena_from_file(path: str, *, loader: Optional[Callable[[str], Any]] = None) -> NenaSource:
    """NeNA of the whole-field file ``path`` (loaded with its own pixel size)."""
    out = NenaSource(source_file=os.path.basename(path or ""))
    if not path or not os.path.isfile(path):
        out.reason = "the whole-field file does not exist" if path else "no whole-field file was chosen"
        return out
    if loader is None:
        from tools.mps_io import load_localizations as loader
    try:
        loc = loader(path)
    except Exception as error:                                # noqa: BLE001 - shown to the user
        out.reason = f"the whole-field file could not be loaded ({error})"
        return out
    return nena_from_localizations(loc, f"whole-field file {os.path.basename(path)}")


def nena_entry(source: NenaSource) -> Dict[str, Any]:
    """What is stored per acquisition folder (``MPSSettings.cut_clusters_by_folder``)."""
    return {"nena_nm": source.nena_nm, "nena_words": source.words, "nena_source_file": source.source_file,
            "n_pairs": source.n_pairs}


def stored_nena(entry: Optional[Dict[str, Any]]) -> Optional[NenaSource]:
    """A NeNA stored for this acquisition folder, or None."""
    if not isinstance(entry, dict):
        return None
    value = entry.get("nena_nm")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) \
            or not 0 < float(value) < 1000:
        return None
    words = str(entry.get("nena_words") or entry.get("nena_source_file") or "a whole-field file")
    return NenaSource(nena_nm=float(value), words=f"stored for this folder: {words}",
                      n_pairs=int(entry.get("n_pairs") or 0), source_file=str(entry.get("nena_source_file") or ""))


def clean_store(store: Any) -> Dict[str, Dict[str, Any]]:
    """The settings' per-folder entries, cleaned: a NeNA that is not one
    is dropped, the pick files are kept as a list of strings."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(store, dict):
        return out
    for folder, entry in store.items():
        if not isinstance(entry, dict):
            continue
        kept: Dict[str, Any] = {}
        stored = stored_nena(entry)
        if stored is not None:
            kept.update({"nena_nm": stored.nena_nm, "nena_words": str(entry.get("nena_words") or ""),
                         "nena_source_file": stored.source_file, "n_pairs": stored.n_pairs})
        picks = entry.get("pick_files")
        if isinstance(picks, list):
            kept["pick_files"] = [str(p) for p in picks if isinstance(p, str)]
        if kept:
            out[str(folder)] = kept
    return out


def considered_labels(labels: NDArray[np.int64], bad: Iterable[int]) -> List[int]:
    """The cluster labels the curation kept (not noise, not bad)."""
    gone = {int(b) for b in bad}
    return [int(v) for v in np.unique(np.asarray(labels)) if v != -1 and int(v) not in gone]


# --------------------------------------------------------------------------- one file's state (the main window's)
@dataclass
class CutSession:
    """Where the rule stands for the loaded channel-1 file: the pick
    match, and the NeNA once a boundary is known. Qt-free: the main window
    holds one, the Pick boundary dialog changes it."""

    path: str
    pixel_size_nm: Optional[float]
    match: Any                                   # tools.mps_picks.PickMatch
    nena: Optional[NenaSource] = None
    set_aside: bool = False                      # the user said "do not use a boundary for this file"

    @property
    def boundary(self) -> Optional[PickBoundary]:
        if self.set_aside or self.pixel_size_nm is None:
            return None
        found: Optional[PickBoundary] = self.match.boundary(float(self.pixel_size_nm))
        return found

    def boundary_reason(self) -> str:
        if self.set_aside:
            return "the boundary was set aside by hand for this file"
        if self.pixel_size_nm is None:
            return "the file has no pixel size, so a pick in camera pixels cannot be placed"
        if self.match.status == "ambiguous":
            return f"pick ambiguous: {self.match.message} (Pick boundary...)"
        return self.match.message or "no Picasso pick boundary is known for this file"

    def worth_passing(self) -> bool:
        """Whether there is anything to tell the analysis beyond the default
        "no boundary given": a pick file was found or chosen, or the user
        acted."""
        return bool(self.match.files or self.match.candidates or self.match.by_hand or self.set_aside)

    def context(self, multiple: float = DEFAULT_NENA_MULTIPLE) -> Optional[CutContext]:
        if not self.worth_passing():
            return None
        b = self.boundary
        if b is None:
            return CutContext(reason=self.boundary_reason(), multiple=multiple)
        if self.nena is None or not self.nena.measured:
            reason = (self.nena.reason if self.nena is not None and self.nena.reason
                      else "no NeNA is known for this channel")
            return CutContext(boundary=b, reason=f"{reason}; measure it on the whole-field, unlinked file "
                                                 f"(Pick boundary...)", multiple=multiple)
        return CutContext(boundary=b, nena_nm=self.nena.nena_nm, nena_source=self.nena.words, multiple=multiple)

    def status_words(self) -> str:
        """One line for the strip before any analysis: what the next
        analysis will do."""
        if self.boundary is None:
            return f"{STATUS_NO_BOUNDARY}: {self.boundary_reason()}"
        if self.nena is None or not self.nena.measured:
            return f"{STATUS_NO_NENA}" + (f": {self.nena.reason}" if self.nena is not None else "")
        return f"boundary and NeNA ({self.nena.nena_nm:.1f} nm) known"


def open_session(path: str, loc: Any, store_entry: Optional[Dict[str, Any]] = None) -> CutSession:
    """Look for the pick of the loaded per-axon file ``loc`` (a Picasso
    file: x and y in camera pixels in its ``columns``). The pick files the
    user pointed to in this folder (``store_entry['pick_files']``) are
    searched with the ones beside it."""
    from tools import mps_picks

    pixel = getattr(loc, "pixel_size_nm", None)
    extra = [p for p in ((store_entry or {}).get("pick_files") or []) if isinstance(p, str) and os.path.isfile(p)]
    files = mps_picks.candidate_files(path, extra)
    x_px, y_px = _camera_pixels(loc)
    record = mps_picks.picked_record(loc.column("group") if hasattr(loc, "column") else None,
                                     getattr(loc, "info", []) or [])
    if x_px is None or y_px is None:
        match = mps_picks.PickMatch(status="none", files=files,
                                    message="the file is not a Picasso file in camera pixels")
    else:
        match = mps_picks.match_pick(x_px, y_px, files, record=record, pixel_size_nm=pixel)
    return CutSession(path=path, pixel_size_nm=pixel, match=match)


def _camera_pixels(loc: Any) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    pixel = getattr(loc, "pixel_size_nm", None)
    if getattr(loc, "fileformat", None) != 0 or pixel is None:
        return None, None
    columns = getattr(loc, "columns", {}) or {}
    if "x" in columns and "y" in columns:
        return np.asarray(columns["x"], float), np.asarray(columns["y"], float)
    return np.asarray(loc.x_nm, float) / float(pixel), np.asarray(loc.y_nm, float) / float(pixel)


def choose_pick_file(session: CutSession, loc: Any, pick_path: str) -> CutSession:
    """The pick file the user pointed to: its picks are matched with the
    same rule; one that fits is used, several are offered to choose."""
    from tools import mps_picks

    x_px, y_px = _camera_pixels(loc)
    record = mps_picks.picked_record(loc.column("group") if hasattr(loc, "column") else None,
                                     getattr(loc, "info", []) or [])
    if x_px is None or y_px is None:
        session.match = mps_picks.PickMatch(status="none", files=[pick_path], by_hand=True,
                                            message="the file is not a Picasso file in camera pixels")
    else:
        session.match = mps_picks.match_pick(x_px, y_px, [pick_path], record=record,
                                             pixel_size_nm=session.pixel_size_nm, by_hand=True)
    session.set_aside = False
    session.nena = None if session.match.status == "matched" else session.nena
    return session


def choose_candidate(session: CutSession, index: int) -> CutSession:
    """The user chose one of the candidates of an ambiguous match."""
    cands = session.match.candidates
    if not 0 <= index < len(cands):
        raise IndexError(f"there are {len(cands)} candidate picks")
    session.match.chosen = cands[index]
    session.match.status = "matched"
    session.match.by_hand = True
    session.match.message = cands[index].describe()
    session.set_aside = False
    return session


def resolve_nena(session: CutSession, loc: Any, store_entry: Optional[Dict[str, Any]] = None) -> NenaSource:
    """The NeNA the rule uses for this file: the one measured on the
    whole-field, unlinked file of its acquisition folder when one is stored
    (the decided definition: the same value for every axon of the
    acquisition), otherwise the file's own when it has enough next-frame
    pairs, otherwise none, with the reason."""
    stored = stored_nena(store_entry)
    if stored is not None:
        return stored
    return nena_from_localizations(loc, "this file")
