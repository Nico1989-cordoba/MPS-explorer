"""
Readable year.ROI.axon label of a trace (the perimeter density analysis).

The label is for reading, not a key: the key stays ``axon_id`` (see
``tools/mps_identity.py``), and nothing here changes it. Nothing is guessed:
a part that cannot be read from the file name or from the identity reads
``NA``, and ``trace_warnings`` says which part is missing and why.

Year: the acquisition date at the START of the file name (``yy.m.d_`` or
``yy.mm.dd_``). A date in a folder name is not read. A year typed for the
folder wins over the file name.

ROI and axon: the digit runs of ``identity.roi_name`` and
``identity.axon_name``, which ``mps_identity.propose`` takes from the path.
Leading zeros are dropped, so ``ROI 01`` and ``ROI_1`` both read ``1``.

Pure functions: no Qt, no file access, no randomness.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from tools.mps_perimeter_types import TraceLabel

if TYPE_CHECKING:
    from tools.mps_identity import AxonIdentity

__all__ = ["YEAR_PATTERN", "TraceLabel", "acquisition_date", "trace_label", "trace_warnings",
           "duplicate_labels"]

# Shown in place of any part that cannot be read.
NA = "NA"

# acquisition date yy.m.d at the start of the file name  [ad hoc, from the April file names]
# Groups: year, month, day. Applied with re.match to the file name only.
YEAR_PATTERN: str = r"(?:^|[\\/_ ])(\d{2})\.(\d{1,2})\.(\d{1,2})_"

# The parts in the order of the label, and the wording of each one's warning.
_MISSING_TEXT: Dict[str, str] = {
    "year": ("no acquisition date at the start of the file name (yy.m.d_), so the year reads NA; "
             "type the year for this folder, or rename the file so it starts with its date"),
    "roi": "no ROI number in the identity, so the ROI reads NA",
    "axon": "no axon number in the identity, so the axon reads NA",
}


def _file_name(source_path: str) -> str:
    """The last path component, split on both '/' and '\\' whatever the platform."""
    text = str(source_path or "").replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def _read_date(source_path: str, pattern: str) -> Tuple[Optional[Tuple[str, str, str]], str]:
    """The date at the start of the file name, and a sentence saying what was found.

    The pattern must have three groups (year, month, day); anything else is a
    programming error and raises ValueError. Month and day are checked, so a
    run of digits that is not a date is not taken for one.
    """
    if re.compile(pattern).groups != 3:
        raise ValueError("the date pattern must have three groups: year, month, day")
    match = re.match(pattern, _file_name(source_path))
    if match is None:
        return None, "no date at the start of the file name"
    year, month, day = match.groups()
    if year is None or month is None or day is None:
        return None, "no date at the start of the file name"
    month_n, day_n = int(month), int(day)
    if not (1 <= month_n <= 12 and 1 <= day_n <= 31):
        return None, f"'{match.group(0).rstrip('_')}' is not a valid date"
    return (f"{int(year):02d}", f"{month_n:02d}", f"{day_n:02d}"), ""


def _typed_year(year: str) -> Tuple[str, str]:
    """The typed year as two digits, and a note when it was not usable."""
    text = str(year or "").strip()
    if not text:
        return "", ""
    if re.fullmatch(r"\d{2}", text):
        return text, ""
    if re.fullmatch(r"\d{4}", text):
        return text[-2:], ""
    return "", f"'{text}' is not a year (two or four digits), so it was not used"


def _number(text: str) -> Optional[str]:
    """The first run of digits, without leading zeros, or None."""
    match = re.search(r"\d+", str(text or ""))
    if match is None:
        return None
    return str(int(match.group(0)))


def acquisition_date(source_path: str, *, pattern: str = YEAR_PATTERN) -> Optional[Tuple[str, str, str]]:
    """(year, month, day) read from the start of the file name, zero-padded, or None.

    Folder names are not read. An impossible date (month above 12, day above 31) is None.
    """
    return _read_date(source_path, pattern)[0]


def trace_label(identity: "AxonIdentity", source_path: str, *, year: str = "",
                pattern: str = YEAR_PATTERN) -> TraceLabel:
    """The label ``year.ROI.axon``, e.g. ``26.1.2``.

    A ``year`` typed for the folder wins over the file name (two or four digits;
    four are cut to two). A missing part reads ``NA`` and ``complete`` is False.
    ``origin`` says where each part came from; ``trace_warnings`` gives the lines to show.
    """
    date, date_note = _read_date(source_path, pattern)
    typed, typed_note = _typed_year(year)

    missing: List[str] = []
    parts: List[str] = []

    if typed:
        year_value = typed
        year_origin = f"year typed '{typed}'"
        if date is not None and date[0] != typed:
            year_origin += f" (the file name says {date[0]})"
    elif date is not None:
        year_value = date[0]
        year_origin = f"year from the file name ({date[0]}.{date[1]}.{date[2]})"
    else:
        year_value = NA
        year_origin = f"year not found ({date_note})"
        missing.append("year")
    if typed_note:
        year_origin += f"; {typed_note}"
    parts.append(year_value)

    roi = _number(identity.roi_name)
    if roi is None:
        missing.append("roi")
        roi_origin = "ROI not found in the identity"
    else:
        roi_origin = f"ROI from '{identity.roi_name}'"
    parts.append(roi or NA)

    axon = _number(identity.axon_name)
    if axon is None:
        missing.append("axon")
        axon_origin = "axon not found in the identity"
    else:
        axon_origin = f"axon from '{identity.axon_name}'"
    parts.append(axon or NA)

    return TraceLabel(
        label=".".join(parts),
        year=year_value,
        roi_number=roi or NA,
        axon_number=axon or NA,
        acquisition_date=date,
        complete=not missing,
        missing=tuple(missing),
        origin="; ".join([year_origin, roi_origin, axon_origin]),
    )


def trace_warnings(label: TraceLabel) -> List[str]:
    """One line per missing part of the label; empty when the label is complete."""
    return [f"Trace {label.label}: {_MISSING_TEXT[part]}." for part in label.missing]


def duplicate_labels(labels: Sequence[TraceLabel]) -> Dict[str, List[int]]:
    """The labels used by more than one trace, each with the positions that share it.

    Two files of the same year, ROI and axon get the same label; the label is not a
    key, so this is the warning, and ``acquisition_date`` is what tells them apart.
    """
    positions: Dict[str, List[int]] = {}
    for index, item in enumerate(labels):
        positions.setdefault(item.label, []).append(index)
    return {name: where for name, where in positions.items() if len(where) > 1}
