# -*- coding: utf-8 -*-
"""
The pixel size an analysis uses, against the one the microscope recorded.

Picasso's Localize asks for the pixel size and writes it into the YAML,
and every distance downstream comes from that number. It is typed, not
measured, and nothing used to check it. The acquisition software does
record what the microscope was configured with -- Micro-Manager in the
movie's metadata, Tormenta in a sidecar beside it -- and TIFF files carry
it in their resolution tags. This module reads those and says when they
disagree with the value in use.

The 2023 sciatic-nerve data is the reason. All 202 of its acquisition
metadata files record 0.133 um, while 196 of the 200 Picasso YAMLs say
135 nm and the other 4 say 133; its widefield images record 133 nm in
their ImageJ resolution tags. Nobody remembers which is right, so the
software reports the disagreement instead of choosing.

A recorded value is believed only inside PLAUSIBLE_NM. The split movies
of that same data show why: ImageJ wrote their resolution as 0.133 with
the unit "cm", which reads as 1.33 mm per pixel.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools import mps_metadata

# A camera pixel, projected on the sample, between these bounds. Wider
# than any objective in use: the point is to reject a value whose unit is
# wrong or whose field means something else, not to police the optics.
PLAUSIBLE_NM = (1.0, 5000.0)
# Below this the two values are the same number.
SAME_NM = 1e-6
# How the length units seen in TIFF and acquisition metadata convert.
_TO_NM = {
    "nm": 1.0, "nanometer": 1.0, "nanometre": 1.0,
    "um": 1e3, "µm": 1e3, "\\u00b5m": 1e3, "micron": 1e3, "microns": 1e3,
    "micrometer": 1e3, "micrometre": 1e3,
    "mm": 1e6, "millimeter": 1e6, "millimetre": 1e6,
    "cm": 1e7, "centimeter": 1e7, "centimetre": 1e7,
    "m": 1e9, "meter": 1e9, "metre": 1e9,
    "inch": 2.54e7, "in": 2.54e7,
}


@dataclass
class RecordedPixel:
    """A pixel size the acquisition or the file itself records."""

    nm: float
    source: str          # named in the message, e.g. "the ImageJ tags of x"


def plausible(nm: Optional[float]) -> bool:
    return nm is not None and PLAUSIBLE_NM[0] <= nm <= PLAUSIBLE_NM[1]


def _as_micrometres(value: Any) -> Optional[float]:
    """A number of micrometres, or nanometres when it reads that way."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    if plausible(number * 1e3):
        return number
    # Written in nanometres already: keep it rather than reject it.
    if plausible(number):
        return number / 1e3
    return None


def from_micromanager(info: Sequence[Dict[str, Any]],
                      name: str = "the movie") -> Optional[RecordedPixel]:
    """The pixel size Micro-Manager recorded, from Picasso's metadata."""
    meta = mps_metadata.get_value(info, "Micro-Manager Metadata")
    if not isinstance(meta, dict):
        return None
    micrometres = _as_micrometres(meta.get("PixelSizeUm"))
    if micrometres is None:
        return None            # 0 means Micro-Manager was never calibrated
    return RecordedPixel(micrometres * 1e3,
                         f"the Micro-Manager metadata of {name}")


def _from_text_sidecar(path: str) -> Optional[RecordedPixel]:
    """``Pixel size= 0.133`` in the text file the acquisition wrote."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip().lower() not in ("pixel size", "pixelsize"):
            continue
        micrometres = _as_micrometres(value.strip())
        if micrometres is None:
            return None
        return RecordedPixel(micrometres * 1e3,
                             f"{os.path.basename(path)}")
    return None


def _from_hdf5_sidecar(path: str) -> Optional[RecordedPixel]:
    """``Pixel size`` or ``element_size_um`` in an acquisition metadata file."""
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            found = None
            for key in ("Pixel size", "Pixelsize", "pixel size"):
                if key in handle:
                    found = handle[key][()]
                    break
                if key in handle.attrs:
                    found = handle.attrs[key]
                    break
            if found is None and "element_size_um" in handle:
                sizes = list(handle["element_size_um"][()])
                found = sizes[-1] if sizes else None
    except (OSError, KeyError, ValueError, TypeError):
        return None
    micrometres = _as_micrometres(found)
    if micrometres is None:
        return None
    return RecordedPixel(micrometres * 1e3, f"{os.path.basename(path)}")


def from_sidecar(path: str) -> Optional[RecordedPixel]:
    """
    The acquisition metadata beside a file, when it is there.

    Tormenta, the acquisition software of the 2023 data, writes both a
    ``<stem>.txt`` of ``key= value`` lines and a ``<stem>_metadata.hdf5``
    next to each movie. A movie processed afterwards keeps the suffix at
    the end of the name while its metadata keeps it after "_metadata":
    beside MPS_t2_ROI1_50ms_calib3_corrected.tiff sits
    MPS_t2_ROI1_50ms_calib3_metadata_corrected.hdf5.
    """
    stem = os.path.splitext(path)[0]
    text = _from_text_sidecar(stem + ".txt")
    if text is not None:
        return text
    candidates = [stem + "_metadata.hdf5", stem + "_metadata.h5"]
    for suffix in ("_corrected",):
        if stem.endswith(suffix):
            candidates.append(stem[:-len(suffix)] + "_metadata" + suffix
                              + ".hdf5")
    for candidate in candidates:
        if os.path.exists(candidate):
            recorded = _from_hdf5_sidecar(candidate)
            if recorded is not None:
                return recorded
    return None


def from_tiff(path: str) -> Tuple[Optional[RecordedPixel], List[str]]:
    """
    The pixel size a TIFF records, and what was rejected on the way.

    Micro-Manager's own field comes first, then ImageJ's unit and
    resolution, then the plain TIFF resolution tags. A value outside
    PLAUSIBLE_NM is reported rather than used: ImageJ files written by a
    script can carry a resolution in the wrong unit.
    """
    import tifffile

    notes: List[str] = []
    name = os.path.basename(path)
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tags = {key: page.tags[key].value
                    for key in ("XResolution", "ResolutionUnit")
                    if key in page.tags}
            imagej = dict(tif.imagej_metadata or {})
            mm = page.tags.get("MicroManagerMetadata")
            mm_value = mm.value if mm is not None else None
    except (OSError, ValueError, IndexError) as error:
        return None, [f"{name}: {error}"]

    if isinstance(mm_value, dict):
        # Micro-Manager's own field settles it: a zero there means the
        # microscope was never calibrated, and the resolution tags of its
        # files hold a placeholder rather than a pixel size.
        micrometres = _as_micrometres(mm_value.get("PixelSizeUm"))
        if micrometres is None:
            return None, notes
        return (RecordedPixel(micrometres * 1e3,
                              f"the Micro-Manager metadata of {name}"),
                notes)

    resolution = tags.get("XResolution")
    per_unit: Optional[float] = None
    if isinstance(resolution, tuple) and len(resolution) == 2 and resolution[1]:
        per_unit = float(resolution[0]) / float(resolution[1])
    elif isinstance(resolution, (int, float)) and resolution:
        per_unit = float(resolution)
    if not per_unit:
        return None, notes

    # ImageJ states the unit itself; a plain TIFF only in ResolutionUnit.
    unit = str(imagej.get("unit", "")).strip().lower()
    where = f"the ImageJ tags of {name}"
    if unit not in _TO_NM:
        unit = {2: "inch", 3: "cm"}.get(int(tags.get("ResolutionUnit", 0)
                                             or 0), "")
        where = f"the resolution tags of {name}"
    if unit not in _TO_NM:
        return None, notes
    nm = _TO_NM[unit] / per_unit
    if not plausible(nm):
        # Too large to be a pixel is worth saying: it is a real number
        # under the wrong unit, as in the 2023 movies written with "cm".
        # Too small is a placeholder (ImageJ writes 2^32-1 for "unset").
        if nm > PLAUSIBLE_NM[1]:
            notes.append(
                f"{name} records {nm / 1e3:,.0f} um per pixel in {where}, "
                f"which is not a pixel size: it is ignored. Its unit is "
                f"probably wrong ('{unit}').")
        return None, notes
    return RecordedPixel(nm, where), notes


def for_localizations(loc: Any) -> Tuple[Optional[RecordedPixel], List[str]]:
    """
    What the acquisition recorded for a localization file, if anything.

    Micro-Manager's value travels inside Picasso's metadata. Tormenta's
    sits beside the movie, whose path Picasso records, and is looked for
    beside the localization file as well, since data is usually copied
    away from the acquisition folder.
    """
    notes: List[str] = []
    recorded = from_micromanager(
        loc.info, os.path.basename(str(
            mps_metadata.get_value(loc.info, "File") or "the movie")))
    if recorded is not None:
        return recorded, notes
    movie = mps_metadata.get_value(loc.info, "File")
    for candidate in (str(movie) if movie else "", str(loc.path)):
        if not candidate:
            continue
        recorded = from_sidecar(candidate)
        if recorded is not None:
            return recorded, notes
    return None, notes


def disagreement(name: str, used_nm: Optional[float],
                 recorded: Optional[RecordedPixel]) -> Optional[str]:
    """
    What to tell the user when the value in use is not the recorded one,
    or None when they agree, or when there is nothing to compare.
    """
    if recorded is None or not used_nm:
        return None
    if abs(float(used_nm) - recorded.nm) <= SAME_NM:
        return None
    ratio = abs(float(used_nm) - recorded.nm) / recorded.nm
    return (f"{name} is analysed with a pixel size of {used_nm:g} nm, but "
            f"{recorded.source} records {recorded.nm:g} nm. Every distance "
            f"scales with it -- {ratio:.1%} here, and areas by "
            f"{(1 + ratio) ** 2 - 1:.1%} -- so one of the two is wrong.")


# ---------------------------------------------------------------------
# Finding a pixel size for a file that does not carry one
# ---------------------------------------------------------------------
#
# The readers above answer "what does THIS file record?". Two of the
# user's own datasets answer "nothing": the 2023 sciatic-nerve exports
# and the TIRF4 DNA-PAINT set both contain localization files whose
# Picasso YAML was lost somewhere between the acquisition computer and
# the analysis folder. Until now the program either refused them or --
# worse, until 2026-09-20 -- loaded them at a hardcoded 133 nm.
#
# A file that records nothing is rarely alone. It sits in a folder with
# its siblings from the same acquisition, and those siblings usually
# kept their metadata; the movie it came from is often one folder up.
# So look there before asking a person to remember a number, and when a
# person does have to be asked, ask once per folder rather than once per
# file.

# Folders that never hold acquisition metadata and can cost minutes to
# walk. Skipped by name, not by content.
_SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv",
              "node_modules", ".mypy_cache", ".pytest_cache"}
# Localization files whose own metadata is worth reading.
_LOC_SUFFIXES = (".hdf5", ".h5")
_TIFF_SUFFIXES = (".tif", ".tiff")
# Opening files is not free and a data folder can hold thousands. The
# search stops at this many and says that it did.
DEFAULT_FILE_BUDGET = 300
# How far up the tree to walk. Two levels reaches the acquisition folder
# from <acquisition>/<roi>/<picked>/file.hdf5 without reaching the disk.
DEFAULT_LEVELS_UP = 2


@dataclass
class Candidate:
    """A pixel size found somewhere near a file that has none."""

    nm: float
    # Where it was read from, in words, for a sentence the user reads.
    source: str
    # The file it was read from.
    path: str
    # 0 beside the file, 1 one folder up, 2 two folders up.
    steps_away: int

    @property
    def where(self) -> str:
        if self.steps_away == 0:
            return "in the same folder"
        if self.steps_away == 1:
            return "one folder up"
        return f"{self.steps_away} folders up"


def _from_localization_file(path: str) -> Optional[RecordedPixel]:
    """The pixel size a neighbouring localization file carries, if any."""
    try:
        info, _ = mps_metadata.load_metadata(path)
    except Exception:                                     # noqa: BLE001
        return None
    nm = mps_metadata.pixel_size_nm(info)
    if not plausible(nm):
        return None
    recorded = from_micromanager(info, os.path.basename(path))
    if recorded is not None:
        return recorded
    return RecordedPixel(
        float(nm),
        f"the Picasso metadata of {os.path.basename(path)}")


def _scan_folder(folder: str, steps_away: int, skip: str,
                 budget: List[int]) -> Tuple[List["Candidate"], List[str]]:
    """Every pixel size the files directly inside one folder record."""
    found: List[Candidate] = []
    notes: List[str] = []
    try:
        names = sorted(os.listdir(folder))
    except OSError as error:
        return found, [f"{folder} could not be read ({error})."]
    for name in names:
        if budget[0] <= 0:
            notes.append(
                f"The search stopped after {DEFAULT_FILE_BUDGET} files; "
                f"{folder} was not read to the end.")
            break
        full = os.path.join(folder, name)
        if os.path.normcase(full) == os.path.normcase(skip):
            continue
        if not os.path.isfile(full):
            continue
        lower = name.lower()
        recorded: Optional[RecordedPixel] = None
        if lower.endswith(_LOC_SUFFIXES):
            budget[0] -= 1
            recorded = _from_localization_file(full)
            if recorded is None:
                recorded = _from_hdf5_sidecar(full)
        elif lower.endswith(_TIFF_SUFFIXES):
            budget[0] -= 1
            try:
                recorded, tiff_notes = from_tiff(full)
            except Exception as error:                    # noqa: BLE001
                recorded, tiff_notes = None, [f"{name}: {error}"]
            notes.extend(tiff_notes)
        elif lower.endswith(".txt"):
            budget[0] -= 1
            recorded = _from_text_sidecar(full)
        if recorded is not None and plausible(recorded.nm):
            found.append(Candidate(recorded.nm, recorded.source, full,
                                   steps_away))
    return found, notes


def discover(path: str, *, levels_up: int = DEFAULT_LEVELS_UP,
             file_budget: int = DEFAULT_FILE_BUDGET
             ) -> Tuple[List["Candidate"], List[str]]:
    """
    Look around a file for a pixel size it does not itself carry.

    The file's own sidecars first, then the files beside it, then the
    folders above it, stopping as soon as a level yields anything: a
    value from the same folder describes the same acquisition, and one
    from three folders up may describe a different day.

    Returns every candidate found at that level and the notes worth
    showing (a unit that did not parse, a folder that could not be read,
    the budget running out). It decides nothing: candidates that
    disagree are all returned, because two different numbers near one
    file is exactly the situation a person has to settle.
    """
    candidates: List[Candidate] = []
    notes: List[str] = []
    budget = [int(file_budget)]
    # The file itself is never a candidate: this is only ever called
    # because it has no pixel size of its own.
    itself = os.path.abspath(path)

    beside = from_sidecar(path)
    if beside is not None and plausible(beside.nm):
        candidates.append(Candidate(beside.nm, beside.source, path, 0))

    folder = os.path.dirname(os.path.abspath(path))
    for step in range(max(0, int(levels_up)) + 1):
        if not folder or os.path.basename(folder) in _SKIP_DIRS:
            break
        found, more = _scan_folder(folder, step, itself, budget)
        notes.extend(more)
        candidates.extend(found)
        if candidates:
            break
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return candidates, notes


def distinct(candidates: Sequence["Candidate"],
             tolerance_nm: float = 0.5) -> List[float]:
    """
    The distinct values among candidates, smallest first.

    Two readings of the same acquisition can differ in the last digit
    without disagreeing; 135 and 133 nm, the two values the 2023 data
    carries, are a real disagreement and stay two values.
    """
    values: List[float] = []
    for candidate in sorted(candidates, key=lambda c: c.nm):
        if not any(abs(candidate.nm - v) <= tolerance_nm for v in values):
            values.append(candidate.nm)
    return values


def describe(candidates: Sequence["Candidate"],
             notes: Sequence[str] = ()) -> str:
    """What the search found, as the sentences a dialog can show."""
    lines: List[str] = []
    if not candidates:
        lines.append("Nothing near this file records a pixel size.")
    else:
        values = distinct(candidates)
        if len(values) == 1:
            lines.append(f"{values[0]:g} nm was found near this file:")
        else:
            lines.append(
                f"{len(values)} different pixel sizes were found near this "
                f"file, so they cannot all be right:")
        for candidate in candidates[:8]:
            lines.append(f"  {candidate.nm:g} nm -- {candidate.source}, "
                         f"{candidate.where}")
        if len(candidates) > 8:
            lines.append(f"  ... and {len(candidates) - 8} more")
    lines.extend(str(note) for note in notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Remembering the answer
# ---------------------------------------------------------------------

def folder_key(path: str) -> str:
    """
    The key a remembered pixel size is stored under: the file's folder.

    One acquisition is one folder in this user's layout, and the six or
    more axons picked out of it are the files inside it. Keying on the
    folder asks once per acquisition instead of once per axon, and never
    carries a value across to the next day's folder -- which is the
    failure a global "last pixel size" setting would have.
    """
    return os.path.normcase(os.path.abspath(os.path.dirname(path)))


@dataclass
class Resolution:
    """A pixel size to use, and where it came from."""

    nm: float
    # One of mps_analysis.PIXEL_SIZE_SOURCES, for the export column. All
    # three possible values here are in GUESSED_PIXEL_SIZE_SOURCES: none
    # of them is this file's own record, so all three warn.
    token: str
    # The same thing in words, for the log and the provenance line.
    source: str

    @property
    def by_hand(self) -> bool:
        """A person supplied it, now or for an earlier file."""
        return self.token in ("manual", "remembered")


def resolve(path: str, remembered: Dict[str, float], ask: Any,
            *, levels_up: int = DEFAULT_LEVELS_UP,
            file_budget: int = DEFAULT_FILE_BUDGET
            ) -> Tuple[Optional["Resolution"], List[str]]:
    """
    Settle the pixel size of a file that does not carry one.

    ``remembered`` maps ``folder_key`` to a value a person already gave
    for that folder; ``ask`` is called as ``ask(candidates, notes)`` and
    returns a value in nanometres, or None if the person declined. It is
    the only part of this that needs a screen, which is why it is passed
    in: everything above is testable without one.

    A single agreed value found near the file is used without asking --
    that is the whole point of looking. Two different values are never
    resolved here, not even by majority: the program does not know which
    microscope wrote which file, and a wrong pixel size rescales every
    distance in the thesis silently. It asks, and it shows both.

    Returns (Resolution, notes), or (None, notes) when nothing was
    settled -- in which case the caller must refuse the file, as every
    other path in this program does.
    """
    candidates, notes = discover(path, levels_up=levels_up,
                                 file_budget=file_budget)
    values = distinct(candidates)

    if len(values) == 1:
        best = min(candidates, key=lambda c: c.steps_away)
        return (Resolution(values[0], "neighbour",
                           f"found in {best.source}, {best.where}"),
                notes)

    key = folder_key(path)
    if not values and key in remembered:
        stored = remembered.get(key)
        if plausible(stored):
            return (Resolution(float(stored),
                               "remembered",
                               "given by hand for this folder earlier"),
                    notes)

    given = ask(candidates, notes)
    if given is None or not plausible(given):
        return None, notes
    remembered[key] = float(given)
    return Resolution(float(given), "manual", "typed by hand"), notes
