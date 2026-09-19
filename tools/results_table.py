# -*- coding: utf-8 -*-
"""
Results tables that grow one export at a time, without losing rows.

The panels add each axon to a CSV the user picks, so a folder's worth of
axons accumulates into one table. Appending is only safe onto a table with
the same columns. Anything else -- another export's table, or this one
after Excel re-saved it with ';' separators or in its own code page -- used
to be replaced by a one-row table without a word. Such a file is now left
as it is and the export refused, with the reason.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import io
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


class TableMismatch(ValueError):
    """The file exists and is not a table these rows can be added to."""


def _existing(path: str) -> Tuple[Optional[List[str]], bool]:
    """The header of the table at ``path`` (None when there is no table)
    and whether its last line is terminated."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None, True
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        # UTF-8 rows appended to it would mix two encodings in one file.
        raise TableMismatch(
            f"{os.path.basename(path)} is not UTF-8 text (byte "
            f"{raw[error.start]:#04x} at position {error.start}). Excel's "
            f"plain 'CSV' format writes its own code page: save the table "
            f"as 'CSV UTF-8', or export to a new file.") from None
    header = next(csv.reader(io.StringIO(text)), None)
    return (header or None), text.endswith(("\n", "\r"))


def read_header(path: str) -> Optional[List[str]]:
    """The column names of the table at ``path``; None when there is none."""
    return _existing(path)[0]


def refuse_other_analysis(path: str, record: Dict[str, Any],
                          columns: Sequence[str]) -> None:
    """
    Raise TableMismatch when the table at ``path`` holds rows of another
    analysis than ``record``: rows that differ in any of ``columns`` (which
    clusters, which contour) describe the same axons differently, and a
    statistic over a table pooling them would count axons twice or mix two
    methods. A value missing on either side is not a difference.
    """
    for column in columns:
        value = record.get(column)
        if value is None:
            continue
        others = column_values(path, column) - {str(value), ""}
        if others:
            raise TableMismatch(
                f"{os.path.basename(path)} holds rows with {column} = "
                f"{', '.join(sorted(others))}, and this row has {value}. "
                f"Each analysis goes to a table of its own, so that no "
                f"statistic pools two analyses of the same axons: choose "
                f"another file.")


def column_values(path: str, column: str) -> Set[str]:
    """
    The values ``column`` takes in the table at ``path``: empty when there
    is no table, no such column, or text that is not UTF-8 (append_rows
    then says what is wrong with the file).
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column not in reader.fieldnames:
                return set()
            return {row[column] for row in reader
                    if row.get(column) is not None}
    except UnicodeDecodeError:
        return set()


def _mismatch(path: str, header: List[str], fields: List[str]) -> str:
    name = os.path.basename(path)
    if len(header) == 1 and any(sep in header[0] for sep in (";", "\t")):
        cause = ("its columns are separated by ';' or tabs, as Excel saves "
                 "them where the decimal separator is a comma")
    else:
        missing = [f for f in fields if f not in header]
        extra = [h for h in header if h not in fields]
        if not missing and not extra:
            cause = "its columns are in another order"
        else:
            parts = []
            if missing:
                parts.append("it lacks " + ", ".join(missing[:4])
                             + (" ..." if len(missing) > 4 else ""))
            if extra:
                parts.append("it has " + ", ".join(extra[:4])
                             + (" ..." if len(extra) > 4 else "")
                             + ", which this export does not write")
            cause = "; ".join(parts)
    return (f"{name} already exists with other columns: {cause}. Adding "
            f"these rows would misalign them, and replacing the file would "
            f"lose its rows, so nothing was written. Choose another file.")


def _fields(rows: Sequence[Dict[str, Any]],
            fieldnames: Optional[Sequence[str]]) -> List[str]:
    if not rows:
        raise ValueError("No rows to write.")
    return (list(fieldnames) if fieldnames is not None
            else list(dict.fromkeys(k for row in rows for k in row)))


def check_appendable(path: str, rows: Sequence[Dict[str, Any]],
                     fieldnames: Optional[Sequence[str]] = None) -> None:
    """
    Raise TableMismatch, writing nothing, when append_rows would refuse
    ``rows`` at ``path``. An export that writes several tables checks them
    all first, so that a refusal never leaves some of them written: a
    second attempt would then add the same axon twice to those.
    """
    fields = _fields(rows, fieldnames)
    header, _ = _existing(path)
    if header is not None and header != fields:
        raise TableMismatch(_mismatch(path, header, fields))


def append_rows(path: str, rows: Sequence[Dict[str, Any]],
                fieldnames: Optional[Sequence[str]] = None) -> bool:
    """
    Add ``rows`` to the table at ``path``, creating the table if needed.

    The columns are ``fieldnames``, or every key of ``rows`` in order of
    appearance. Returns True when the rows went onto an existing table and
    False when a new one was written. A file with other columns is not
    touched: TableMismatch says what differs.
    """
    fields = _fields(rows, fieldnames)
    header, terminated = _existing(path)
    if header is not None and header != fields:
        raise TableMismatch(_mismatch(path, header, fields))
    append = header is not None
    with open(path, "a" if append else "w", encoding="utf-8",
              newline="") as handle:
        if append and not terminated:
            handle.write("\r\n")
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not append:
            writer.writeheader()
        writer.writerows(rows)
    return append
