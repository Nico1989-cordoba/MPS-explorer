# -*- coding: utf-8 -*-
"""
Which axon a row describes, in the terms the experiment is designed in.

Every exported table says which file and which ROI it came from, and that
is enough to tell two rows apart but not to compare groups. The design is
nested: a nerve is of one genotype (KO or WT) for one protein (alpha-
adducin or 4.1B); a slide carries several sections of that nerve; a
measurement is one ROI; and an ROI holds at least six axons. The axon
number itself is arbitrary -- it is the order in which the user picked
them -- so it identifies an axon only together with its ROI, its slide and
its nerve.

Those five things are in the folder names the user already keeps, and
nowhere in the files. This module proposes them from the path and stops
there.

Why nothing here guesses
------------------------
A genotype attached to the wrong axon does not produce a visible error: it
moves a measurement from one group to the other, and every statistic
afterwards is computed from it as if it had been read off the sample. A
field this module cannot find in the path is therefore left empty, with a
line saying why, and the user fills it in before the export -- the same
rule ``mps_batch.parse_metadata`` already follows for the headless batch.
The patterns are the user's, written once and stored; what they do not
match is not invented.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Tuple

# The five fields, in the order the nesting goes from the widest to the
# narrowest. Every table carries all of them.
FIELDS: Tuple[str, ...] = ("genotype", "protein", "sample", "roi_name",
                           "axon_name")

# What each one is, in the user's terms. Shown beside the field.
FIELD_LABELS: Dict[str, str] = {
    "genotype": "Genotype",
    "protein": "Protein",
    "sample": "Slide / section",
    "roi_name": "ROI (the measurement)",
    "axon_name": "Axon",
}

FIELD_HINTS: Dict[str, str] = {
    "genotype": "KO or WT, for the protein below. The nerve's genotype.",
    "protein": "Which knockout this nerve is: alpha-adducin or 4.1B.",
    "sample": "The slide, or the section of the nerve on it.",
    "roi_name": "One measurement. A slide may hold several.",
    "axon_name": "The axon within the ROI. Its number is the order it was "
                 "picked in, so it means nothing without the ROI.",
}

# Patterns applied to the whole path, with '/' as the separator whatever
# the platform writes. Each is a regular expression; the first capturing
# group is the value, or the whole match when there is no group.
#
# The defaults match how the test data is filed ('.../Abril/ROI 1/Axon 7/')
# and the two knockouts of the thesis. They are a starting point, not a
# rule: the user edits them once for their own folders.
DEFAULT_PATTERNS: Dict[str, str] = {
    "genotype": r"\b(KO|WT)\b",
    "protein": r"(aducina|adducina|adducin|4\.1B)",
    # Empty: the slide is taken as the folder that holds the ROI folder,
    # which is where it sits in the test data. A pattern typed here wins.
    "sample": "",
    "roi_name": r"(ROI[ _-]*\d+)",
    "axon_name": r"([Aa]xon[ _-]*\d+)",
}

# Origins, so the panel can say where each proposal came from rather than
# presenting all five as equally certain.
FROM_PATTERN = "pattern"
FROM_FOLDER = "folder above the ROI"
FROM_PREVIOUS = "previous axon"
NOT_FOUND = "not found"
NO_PATTERN = "no pattern"
BAD_PATTERN = "bad pattern"


@dataclass
class AxonIdentity:
    """Where one axon sits in the experiment. Empty means not known."""

    genotype: str = ""
    protein: str = ""
    sample: str = ""
    roi_name: str = ""
    axon_name: str = ""

    def __post_init__(self) -> None:
        for name in FIELDS:
            value = getattr(self, name)
            setattr(self, name, "" if value is None else str(value).strip())

    @property
    def missing(self) -> Tuple[str, ...]:
        """The fields still empty, in the order of FIELDS."""
        return tuple(n for n in FIELDS if not getattr(self, n))

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def columns(self) -> Dict[str, Optional[str]]:
        """The identity as export columns. Empty fields are empty cells,
        never a placeholder that a reader could take for a value."""
        return {n: (getattr(self, n) or None) for n in FIELDS}

    def warnings(self) -> List[str]:
        """One line per field the tables will carry empty."""
        out = []
        for name in self.missing:
            out.append(
                f"{FIELD_LABELS[name]} is not filled in, so the exported "
                f"rows carry no {name}. A group comparison cannot use them.")
        return out

    def describe(self) -> str:
        """One line for a message box: the fields that are filled in."""
        parts = [f"{FIELD_LABELS[n]}: {getattr(self, n)}"
                 for n in FIELDS if getattr(self, n)]
        return ", ".join(parts) if parts else "nothing filled in"


@dataclass
class Proposal:
    """What the path suggests, and where each field came from."""

    identity: AxonIdentity
    origin: Dict[str, str] = field(default_factory=dict)
    detail: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def explain(self, name: str) -> str:
        """The sentence shown under one field."""
        return self.detail.get(name, "")


def _as_posix(path: str) -> str:
    return str(path).replace("\\", "/")


def _search(pattern: str, text: str) -> Optional[str]:
    """The first capturing group of ``pattern`` in ``text``, or the whole
    match when the pattern captures nothing. None when it does not match.

    Raises re.error for a pattern that does not compile, which the caller
    reports rather than letting an export fail on it.
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if match is None:
        return None
    if match.groups():
        for group in match.groups():
            if group:
                return str(group)
        return match.group(0)
    return match.group(0)


def _folder_above_roi(path: str, roi_pattern: str) -> Tuple[Optional[str],
                                                            str]:
    """
    The folder that holds the ROI folder, which is where the slide sits in
    the way the test data is filed, plus a sentence saying so.

    The file name itself is never the answer: a whole-field file whose name
    contains "ROI 1" would otherwise make its own folder the slide.
    """
    parts = [p for p in _as_posix(path).split("/") if p]
    if len(parts) < 2:
        return None, "the path has no folders to read a slide from"
    folders = parts[:-1]                 # drop the file name
    for index in range(len(folders) - 1, 0, -1):
        try:
            hit = _search(roi_pattern, folders[index]) if roi_pattern else None
        except re.error:
            return None, "the ROI pattern is not a valid regular expression"
        if hit is not None:
            return folders[index - 1], (
                f"the folder above '{folders[index]}'")
    return None, ("no folder in the path looks like an ROI, so the folder "
                  "above it cannot be found")


def _carry_over(path: str, remembered: "AxonIdentity",
                remembered_folder: str, proposed: AxonIdentity) -> bool:
    """
    Whether the values confirmed for the previous axon still apply here.

    They do when this file sits in the same slide -- the same ROI folder,
    or another ROI of the same slide. Two axons of one measurement share
    genotype, protein and slide by construction, and retyping them for each
    of the six or more axons is how a typo gets into one row of a group.
    Across slides nothing is carried: that is where the genotype changes.
    """
    if not remembered_folder:
        return False
    here = _as_posix(os.path.dirname(path)).rstrip("/").lower()
    there = _as_posix(remembered_folder).rstrip("/").lower()
    if not here or not there:
        return False
    if here == there or here.startswith(there + "/") or \
            there.startswith(here + "/"):
        return True
    # Different folders: the same slide still counts, when both paths name
    # it and it is the same one.
    if proposed.sample and remembered.sample:
        return proposed.sample.lower() == remembered.sample.lower()
    return False


def propose(
    path: str,
    patterns: Optional[Dict[str, str]] = None,
    remembered: Optional[AxonIdentity] = None,
    remembered_folder: str = "",
) -> Proposal:
    """
    Read the five fields off ``path`` with the user's patterns.

    Nothing is invented: a field whose pattern does not match is returned
    empty, with the reason, for the user to type before exporting. Values
    confirmed for the previous axon of the same slide fill in what the path
    does not say, and are reported as coming from there.
    """
    pats = dict(DEFAULT_PATTERNS)
    for name, value in (patterns or {}).items():
        if name in pats and isinstance(value, str):
            pats[name] = value

    text = _as_posix(path)
    identity = AxonIdentity()
    origin: Dict[str, str] = {}
    detail: Dict[str, str] = {}
    warnings: List[str] = []

    for name in FIELDS:
        pattern = pats.get(name, "")
        if pattern:
            try:
                hit = _search(pattern, text)
            except re.error as error:
                origin[name] = BAD_PATTERN
                detail[name] = (f"the pattern for {name} is not a valid "
                                f"regular expression ({error})")
                warnings.append(
                    f"The pattern for {FIELD_LABELS[name].lower()} is not a "
                    f"valid regular expression and was not used: {error}")
                continue
            if hit:
                setattr(identity, name, hit.strip())
                origin[name] = FROM_PATTERN
                detail[name] = f"'{hit.strip()}' found in the path"
                continue
            origin[name] = NOT_FOUND
            detail[name] = "the pattern found nothing in this path"
        elif name == "sample":
            folder, why = _folder_above_roi(path, pats.get("roi_name", ""))
            if folder:
                identity.sample = folder.strip()
                origin[name] = FROM_FOLDER
                detail[name] = why
            else:
                origin[name] = NOT_FOUND
                detail[name] = why
        else:
            origin[name] = NO_PATTERN
            detail[name] = "no pattern is set for this field"

    if remembered is not None and _carry_over(path, remembered,
                                              remembered_folder, identity):
        where = remembered.sample or "the previous axon"
        for name in FIELDS:
            # The axon is the one field that changes within a measurement,
            # so it is never carried over.
            if name == "axon_name" or getattr(identity, name):
                continue
            kept = getattr(remembered, name)
            if kept:
                setattr(identity, name, kept)
                origin[name] = FROM_PREVIOUS
                detail[name] = f"kept from the previous axon of {where}"

    warnings.extend(identity.warnings())
    return Proposal(identity=identity, origin=origin, detail=detail,
                    warnings=warnings)


def check_patterns(patterns: Dict[str, str]) -> Dict[str, str]:
    """
    The patterns that do not compile, as {field: reason}. Empty when all of
    them are usable. The panel calls this before storing them, so a typo is
    rejected while it can still be fixed rather than at the next export.
    """
    broken = {}
    for name, pattern in patterns.items():
        if name not in DEFAULT_PATTERNS or not pattern:
            continue
        try:
            re.compile(pattern)
        except re.error as error:
            broken[name] = str(error)
    return broken


def axon_id(identity: Optional[AxonIdentity], source_name: str = "",
            roi: str = "") -> str:
    """
    A short stable name for this axon, to join the per-axon row with the
    tables of its clusters and its localizations.

    It is a fingerprint of the identity together with the file and the ROI
    the localizations were taken from, so re-exporting the same axon yields
    the same value and two ROIs of one whole-field file yield two. It is
    not a number to be read: the columns beside it say who the axon is.
    """
    ident = identity or AxonIdentity()
    parts = [getattr(ident, n) for n in FIELDS]
    parts.append(os.path.basename(str(source_name or "")))
    parts.append(str(roi or ""))
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:10]


def identity_from_dict(data: Optional[Dict[str, str]]) -> AxonIdentity:
    """An identity from stored settings, ignoring anything unexpected."""
    values = {}
    for f in fields(AxonIdentity):
        value = (data or {}).get(f.name)
        if isinstance(value, str):
            values[f.name] = value
    return AxonIdentity(**values)


def identity_to_dict(identity: Optional[AxonIdentity]) -> Dict[str, str]:
    """The identity as plain strings, for the settings file."""
    ident = identity or AxonIdentity()
    return {n: getattr(ident, n) for n in FIELDS}
