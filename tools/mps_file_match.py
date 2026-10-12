# -*- coding: utf-8 -*-
"""
Which whole-field file and which widefield images belong to a picked axon, by
their names alone (manual test 2, item 16).

A picked axon (``..._picked_axon3.hdf5``) is analysed with the Axoplasm panel
against three other files of the same acquisition: the whole field of the
movie it was picked from (to measure the alignment), a betaIII-tubulin
widefield image and a betaII-spectrin widefield image. The lab files them
under a few conventions, and finding them by hand in a session folder is
slow and error-prone. ``match_files`` reads the NAMES of the files around the
picked one - never their contents - and proposes, for each of the three, a
ranked list of candidates with the reason each was proposed. Nothing here
loads anything: the panel shows the best candidate pre-filled with its reason
and the user decides.

The conventions (observed in the lab's folders, names only):

A (2026-04-30, Picasso 0.9.10)
    The session folder (``30.4.26``) holds the whole-field SR folder
    (``26.04.30_bIIspt_50ms_90mW_TIRF_3con5_1``) with the chain
    ``<stem>.ome_locs[_undrift][_filter][_permisive|_render|_linked...].hdf5``
    and the picked files ``<stem>.ome_locs<chain>_picked(_axonN|N|N_M).hdf5``;
    its widefield images sit in sibling folders ``WF_spec_{n}`` and
    ``WF_tub_{n}`` whose index n is the SR folder's trailing ``_{n}`` (an
    inference, said so).
B (2026-06-09)
    ``20260609-3DSTORM_add{WT|KO}_b2spec_41_ROI{n}_1``; widefield siblings
    in the same genotype folder named ``ROI{n}_WF`` / ``ROI{n}_wf`` /
    ``ROI{n} wf`` / ``ROI{n}_WF_1`` (spectrin) and ``ROI{n}_t`` (tubulin) -
    which token is which is not confirmed, and the proposal says so.
    Two-colour crops ``cropped/*_ch0.tif`` / ``*_ch1.tif`` are listed, never
    assigned to a protein.
C (2023)
    ``ROI{n}/Seleccion de axones/ROI{n}_{protein}_locs..._apicked_?{k}[_filter].hdf5``
    (also ``t{d}_ROI{n}_`` prefixes and ``-`` separators); the whole field is
    the same name without ``_apicked_?k`` in the ROI folder; the widefield
    images in the day folder, ``MPS_{tag}_ROI{n}_50ms_calib3*_wf{W}x{H}*.tif``:
    ``_t_wf`` tubulin, plain ``_wf`` spectrin (inferred, to confirm),
    ``_ch0`` / ``_ch1`` listed only.

Rules for every convention: tokens are case-insensitive; a file whose size
equals another candidate's is one acquisition rendered twice (undrift,
render), so only the shortest chain of them is proposed; a folder or file
with ``_roi2`` is not preferred (it is still listed). When nothing or several
equally good candidates are found, no best is given and the options are
listed.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Candidate", "MatchResult", "match_files", "date_key", "KINDS"]

WHOLE_FIELD = "whole_field"
TUBULIN = "tubulin"
SPECTRIN = "spectrin"
KINDS: Tuple[str, ...] = (WHOLE_FIELD, TUBULIN, SPECTRIN)
KIND_WORDS: Dict[str, str] = {WHOLE_FIELD: "whole-field localizations", TUBULIN: "betaIII-tubulin widefield",
                              SPECTRIN: "betaII-spectrin widefield"}

SPECTRIN_TOKENS = ("bIIspt", "b2spec", "spectrin", "spec", "spt")
TUBULIN_TOKENS = ("bIIItub", "b3tub", "tubulin", "tub")
IMAGE_SUFFIXES = (".tif", ".tiff")
LOC_SUFFIXES = (".hdf5", ".h5")

# --- convention A
_A_PICKED = re.compile(
    r"^(?P<stem>.+?\.ome_locs)(?P<chain>(?:_undrift|_filter|_permisive|_render|_linked|_)*)"
    r"(?P<pick>_picked(?:_?axon\d+|\d*(?:_\d+)?)?)\.hdf5$", re.IGNORECASE)
_A_CHAIN = re.compile(r"^(?P<stem>.+?\.ome_locs)(?P<chain>(?:_undrift|_filter|_permisive|_render|_linked|_)*)"
                      r"\.hdf5$", re.IGNORECASE)
_A_WF_FOLDER = re.compile(r"^WF_(?P<kind>[A-Za-z0-9]+)_(?P<n>\d+)$", re.IGNORECASE)
_TRAILING_INDEX = re.compile(r"_(\d+)$")
# --- convention B
_B_SR = re.compile(r"ROI\s*_?(?P<n>\d+)", re.IGNORECASE)
_B_WF = re.compile(r"^ROI[ _]?(?P<n>\d+)[ _]+(?P<tok>WF|t)(?:[ _](?P<rep>\d+))?(?:\.tiff?)?$", re.IGNORECASE)
# --- convention C
_C_PICKED = re.compile(r"^(?P<base>.+?)(?P<pick>[_-]apicked[_-]?\d+)(?P<post>[_-]filter)?\.hdf5$", re.IGNORECASE)
_C_ROI = re.compile(r"ROI\s*_?(?P<n>\d+)", re.IGNORECASE)
_C_WF = re.compile(r"^MPS_.*?ROI_?(?P<n>\d+)_.*?wf\d+x\d+.*\.tiff?$", re.IGNORECASE)
_CH = re.compile(r"_ch(?P<c>[01])(?:\.|_)", re.IGNORECASE)

_DATE_SEP = re.compile(r"(?<!\d)(\d{2})[._](\d{1,2})[._](\d{1,2})(?!\d)")
_DATE_COMPACT = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")


@dataclass
class Candidate:
    """One proposed file: its kind, a score (higher is better) and why it was proposed."""

    path: str
    kind: str
    score: float
    reason: str
    to_confirm: bool = False


@dataclass
class MatchResult:
    """The candidates of each kind for one picked file, best first."""

    picked: str
    convention: str = ""
    candidates: Dict[str, List[Candidate]] = field(default_factory=lambda: {k: [] for k in KINDS})
    unassigned: List[Candidate] = field(default_factory=list)   # two-colour crops: listed, never guessed
    notes: List[str] = field(default_factory=list)

    def of(self, kind: str) -> List[Candidate]:
        return list(self.candidates.get(kind, []))

    def best(self, kind: str) -> Optional[Candidate]:
        """The top candidate when it is the only one with the top score; None when there is none or a tie."""
        found = self.candidates.get(kind, [])
        if not found:
            return None
        if len(found) > 1 and abs(found[0].score - found[1].score) < 1e-9:
            return None
        return found[0]

    def is_ambiguous(self, kind: str) -> bool:
        found = self.candidates.get(kind, [])
        return len(found) > 1 and abs(found[0].score - found[1].score) < 1e-9


# ----------------------------------------------------------------------------------------------- helpers
def date_key(name: str) -> Optional[Tuple[int, int, int]]:
    """The date a name carries, as its three numbers sorted (26.04.30, 26.4.30, 26_4_30, 30.4.26 and 20260430 are
    the same date), or None. Sorting makes day-month-year and year-month-day folders comparable; it is a matching
    hint, not a calendar date."""
    m = _DATE_COMPACT.search(name)
    if m:
        y, mo, d = int(m.group(1)) % 100, int(m.group(2)), int(m.group(3))
        return tuple(sorted((y, mo, d)))  # type: ignore[return-value]
    m = _DATE_SEP.search(name)
    if m:
        return tuple(sorted(int(g) for g in m.groups()))  # type: ignore[return-value]
    return None


def token_kind(token: str) -> Optional[str]:
    """SPECTRIN or TUBULIN for a protein token of a folder name (case-insensitive), None for any other."""
    low = token.lower()
    if low in {t.lower() for t in SPECTRIN_TOKENS}:
        return SPECTRIN
    if low in {t.lower() for t in TUBULIN_TOKENS} or low == "t":
        return TUBULIN
    return None


def _listdir(folder: str) -> List[str]:
    try:
        return sorted(os.listdir(folder))
    except OSError:
        return []


def _files(folder: str, suffixes: Sequence[str]) -> List[str]:
    return [os.path.join(folder, n) for n in _listdir(folder)
            if os.path.isfile(os.path.join(folder, n)) and n.lower().endswith(tuple(suffixes))]


def _size(path: str) -> Optional[int]:
    try:
        return int(os.path.getsize(path))
    except OSError:
        return None


def _roi2_penalty(path: str) -> Tuple[float, str]:
    return (-0.5, "; '_roi2' not preferred") if "_roi2" in path.lower() else (0.0, "")


def _rank(cands: List[Candidate]) -> List[Candidate]:
    return sorted(cands, key=lambda c: (-c.score, len(os.path.basename(c.path)), c.path.lower()))


def _collapse_duplicates(cands: List[Candidate], chain_of: Dict[str, str]) -> List[Candidate]:
    """Files of identical size are one acquisition (an undrift or a render of it): keep the shortest chain."""
    by_size: Dict[Optional[int], List[Candidate]] = {}
    for c in cands:
        by_size.setdefault(_size(c.path), []).append(c)
    out: List[Candidate] = []
    for size, group in by_size.items():
        if size is None or len(group) == 1:
            out.extend(group)
            continue
        group = sorted(group, key=lambda c: (len(chain_of.get(c.path, "")), -c.score))
        keep = group[0]
        others = ", ".join(os.path.basename(c.path) for c in group[1:])
        keep.reason += f"; same size as {others} (one acquisition): the shortest chain is proposed"
        out.append(keep)
    return out


# ----------------------------------------------------------------------------------------------- convention A
def _match_a(picked: str, result: MatchResult) -> bool:
    name = os.path.basename(picked)
    m = _A_PICKED.match(name)
    if not m:
        return False
    result.convention = "A"
    sr_folder = os.path.dirname(os.path.abspath(picked))
    stem, chain = m.group("stem").lower(), m.group("chain").lower()
    chains: Dict[str, str] = {}
    whole: List[Candidate] = []
    for folder in (sr_folder, os.path.dirname(sr_folder)):
        for path in _files(folder, (".hdf5",)):
            other = _A_CHAIN.match(os.path.basename(path))
            if not other or other.group("stem").lower() != stem or "_picked" in path.lower():
                continue
            their = other.group("chain").lower()
            chains[path] = their
            if their == chain:
                score, why = 3.0, "same name without the '_picked' part"
            else:
                common = os.path.commonprefix([their, chain])
                score = 1.0 + len(common) / 100.0
                why = f"same stem, chain '{their or '(none)'}' (picked file's: '{chain or '(none)'}')"
            pen, pen_why = _roi2_penalty(path)
            whole.append(Candidate(path, WHOLE_FIELD, score + pen, why + pen_why))
        if whole:
            break
    result.candidates[WHOLE_FIELD] = _rank(_collapse_duplicates(whole, chains))
    # the widefield folders: siblings of the SR folder, WF_spec_{n} / WF_tub_{n}
    sr_name = os.path.basename(sr_folder)
    idx = _TRAILING_INDEX.search(sr_name)
    session = os.path.dirname(sr_folder)
    session_date = date_key(os.path.basename(session)) or date_key(sr_name)
    for sub in _listdir(session):
        mf = _A_WF_FOLDER.match(sub)
        folder = os.path.join(session, sub)
        kind = None if mf is None else token_kind(mf.group("kind"))
        if mf is None or kind is None or not os.path.isdir(folder):
            continue
        for path in _files(folder, IMAGE_SUFFIXES):
            score, parts = 1.0, []
            if session_date is not None:
                parts.append("session date")
                score += 1.0
            if idx is not None and mf.group("n") == idx.group(1):
                score += 2.0
                parts.append(f"index {idx.group(1)} (inferred: {sr_name} -> {sub})")
            elif idx is not None:
                parts.append(f"index {mf.group('n')} is not the SR folder's {idx.group(1)}")
                score -= 1.0
            pen, pen_why = _roi2_penalty(path)
            reason = "matched by: " + " + ".join(parts or ["folder name " + sub]) + pen_why
            result.candidates[kind].append(Candidate(path, kind, score + pen, reason, to_confirm=True))
    for kind in (TUBULIN, SPECTRIN):
        result.candidates[kind] = _rank(result.candidates[kind])
    return True


# ----------------------------------------------------------------------------------------------- convention C
def _match_c(picked: str, result: MatchResult) -> bool:
    name = os.path.basename(picked)
    m = _C_PICKED.match(name)
    if not m:
        return False
    result.convention = "C"
    here = os.path.dirname(os.path.abspath(picked))
    roi_folder = os.path.dirname(here)            # ROI{n}/Seleccion de axones -> ROI{n}
    day = os.path.dirname(roi_folder)
    roi = _C_ROI.search(name) or _C_ROI.search(os.path.basename(roi_folder))
    n = roi.group("n") if roi else None
    base, post = m.group("base"), m.group("post") or ""
    wanted = {(base + post).lower() + ".hdf5": (3.0, "same name without '_apicked_k', in the ROI folder"),
              base.lower() + ".hdf5": (2.0 if post else 3.0, "same name without '_apicked_k'"
                                       + (" and its '_filter'" if post else "") + ", in the ROI folder")}
    whole: List[Candidate] = []
    for folder in (roi_folder, here):
        for path in _files(folder, LOC_SUFFIXES):
            hit = wanted.get(os.path.basename(path).lower())
            if hit is not None and "apicked" not in os.path.basename(path).lower():
                whole.append(Candidate(path, WHOLE_FIELD, hit[0], hit[1]))
    result.candidates[WHOLE_FIELD] = _rank(_collapse_duplicates(whole, {}))
    for path in _files(day, IMAGE_SUFFIXES):
        low = os.path.basename(path).lower()
        mw = _C_WF.match(os.path.basename(path))
        if not mw:
            continue
        score = 2.0 if (n is not None and mw.group("n") == n) else 0.0
        where = f"ROI{mw.group('n')}" + ("" if score else f" (not this axon's ROI{n})")
        pen, pen_why = _roi2_penalty(path)
        if _CH.search(low):
            result.unassigned.append(Candidate(path, "", score + pen,
                                               f"{where}; a two-colour crop (ch0/ch1): which protein it is "
                                               "is not guessed" + pen_why))
            continue
        if "_t_wf" in low:
            result.candidates[TUBULIN].append(Candidate(path, TUBULIN, score + pen, f"matched by: {where} + '_t_wf' "
                                                        "-> tubulin (to confirm)" + pen_why, to_confirm=True))
        else:
            result.candidates[SPECTRIN].append(Candidate(path, SPECTRIN, score + pen, f"matched by: {where} + "
                                                         "plain '_wf' -> spectrin (to confirm)" + pen_why,
                                                         to_confirm=True))
    for kind in (TUBULIN, SPECTRIN):
        result.candidates[kind] = _rank(result.candidates[kind])
    result.unassigned = _rank(result.unassigned)
    return True


# ----------------------------------------------------------------------------------------------- convention B
def _match_b(picked: str, result: MatchResult) -> bool:
    here = os.path.dirname(os.path.abspath(picked))
    sr_folder = None
    for folder in (here, os.path.dirname(here)):
        base = os.path.basename(folder)
        if _DATE_COMPACT.search(base) and _B_SR.search(base):
            sr_folder = folder
            break
    if sr_folder is None:
        return False
    result.convention = "B"
    roi = _B_SR.search(os.path.basename(sr_folder))
    n = roi.group("n") if roi is not None else ""
    genotype = os.path.dirname(sr_folder)
    for entry in _listdir(genotype):
        full = os.path.join(genotype, entry)
        mb = _B_WF.match(entry)
        if mb and mb.group("n") == n:
            tubulin = mb.group("tok").lower() == "t"
            kind = TUBULIN if tubulin else SPECTRIN
            word = "token t -> tubulin (to confirm)" if tubulin else "token WF -> spectrin (to confirm)"
            paths = _files(full, IMAGE_SUFFIXES) if os.path.isdir(full) else (
                [full] if entry.lower().endswith(IMAGE_SUFFIXES) else [])
            for path in paths:
                pen, pen_why = _roi2_penalty(path)
                result.candidates[kind].append(Candidate(path, kind, 2.0 + pen, f"matched by: ROI{n} + {word}"
                                                         + pen_why, to_confirm=True))
        if entry.lower() == "cropped" and os.path.isdir(full):
            for path in _files(full, IMAGE_SUFFIXES):
                crop_roi = _B_SR.search(os.path.basename(path))
                if _CH.search(os.path.basename(path).lower()) and crop_roi is not None and crop_roi.group("n") == n:
                    result.unassigned.append(Candidate(path, "", 1.0, f"ROI{n}; a two-colour crop (ch0/ch1): which "
                                                       "protein it is is not guessed"))
    for kind in (TUBULIN, SPECTRIN):
        result.candidates[kind] = _rank(result.candidates[kind])
    result.unassigned = _rank(result.unassigned)
    # the whole field: the SR folder's localization files that are not picked ones
    whole: List[Candidate] = []
    for path in _files(sr_folder, LOC_SUFFIXES):
        if "pick" in os.path.basename(path).lower() or os.path.abspath(path) == os.path.abspath(picked):
            continue
        whole.append(Candidate(path, WHOLE_FIELD, 1.0, f"a localization file of {os.path.basename(sr_folder)} "
                               "that is not a picked one"))
    result.candidates[WHOLE_FIELD] = _rank(_collapse_duplicates(whole, {}))
    return True


# ----------------------------------------------------------------------------------------------- entry point
def match_files(picked: str) -> MatchResult:
    """The candidates for the whole field and the two widefield images of the picked file ``picked``, by names
    (and file sizes, for duplicates) only."""
    result = MatchResult(picked=str(picked))
    if not picked:
        result.notes.append("no file")
        return result
    for matcher in (_match_a, _match_c, _match_b):
        if matcher(str(picked), result):
            break
    if not result.convention:
        result.notes.append("the file name follows none of the known conventions: nothing is proposed")
    for kind in KINDS:
        if result.is_ambiguous(kind):
            result.notes.append(f"{KIND_WORDS[kind]}: {len(result.candidates[kind])} equally good candidates; "
                                "choose one")
    return result


def describe(candidate: Candidate) -> str:
    """One line for a menu or a tooltip."""
    return f"{os.path.basename(candidate.path)} - {candidate.reason}"
