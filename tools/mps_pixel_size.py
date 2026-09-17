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
