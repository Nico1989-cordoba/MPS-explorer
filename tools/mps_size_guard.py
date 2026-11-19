# -*- coding: utf-8 -*-
"""
How big a file or a selection is before the program reads or analyses it (manual test of 2026-11-17), without Qt.

A person loaded a whole acquisition by mistake instead of a cropped axon: the program grew to 4.2 GB and stopped
answering for more than ten minutes, with no bar and nothing to cancel. The program is built for ONE axon at a time
(``AXON_LOCS_TYPICAL_MAX`` localizations: the largest cropped axon of the pilot sets), so a file or a selection many
times that is, nearly always, a mistake worth one question before the cost is paid.

The rule, three numbers [ad hoc: calibrated on the size of a cropped axon, not on a measured failure]:

* ``LOAD_ASK_LOCS`` = 10 x the largest axon: before READING a file with more localizations than this, ask. The count
  is taken from the file's header (HDF5: the shape of ``/locs``; CSV: the size over the length of the first lines),
  so asking costs nothing. A loaded file is drawn and selected whole, and its memory is about twice its table.
* ``ANALYSIS_BACKGROUND_LOCS`` = a little over one axon: from here an interactive analysis runs in the background,
  with a progress bar and a Cancel button, as the slow methods already do (the window never stops answering).
* ``ANALYSIS_ASK_LOCS`` = 50,000, about 3 axons: before ANALYSING a selection with more localizations than this, ask,
  with the estimate of the chosen method.

Asking never refuses: "Continue" goes on exactly as before the question existed, "Cancel" stops with nothing
changed. No result changes.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

# The largest cropped axon of the pilot sets, in localizations (user's figure: 10 to 15 thousand).
AXON_LOCS_TYPICAL_MAX = 15_000
LOAD_ASK_LOCS = 10 * AXON_LOCS_TYPICAL_MAX            # 150,000
ANALYSIS_BACKGROUND_LOCS = 20_000
ANALYSIS_ASK_LOCS = 50_000
# What a loaded table costs in memory, as a multiple of its size on disk (the table, then one copy per column).
MEMORY_FACTOR = 2.5


def peek_count(path: str, fileformat: Optional[int] = None) -> Tuple[Optional[int], int]:
    """(localizations, bytes on disk) of a file without reading its table; the count is None when it cannot be told
    cheaply (never raises: a file that cannot be peeked is simply loaded as before)."""
    try:
        size = int(os.path.getsize(path))
    except OSError:
        return None, 0
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".hdf5", ".h5"):
            import h5py
            with h5py.File(path, "r") as handle:
                if "locs" in handle:
                    return int(handle["locs"].shape[0]), size
            return None, size
        if ext == ".csv":
            with open(path, "rb") as fh:
                head = fh.read(256 * 1024)
            lines = head.split(b"\n")
            if len(lines) < 3:
                return None, size
            body = lines[1:-1]
            if not body:
                return None, size
            per_line = sum(len(b) + 1 for b in body) / len(body)
            return max(0, int(size / per_line) - 1), size
    except Exception:                                    # noqa: BLE001 - a peek must never stop a load
        return None, size
    return None, size


def load_needs_question(n_locs: Optional[int]) -> bool:
    return n_locs is not None and int(n_locs) > LOAD_ASK_LOCS


def analysis_needs_question(n_locs: int) -> bool:
    return int(n_locs) > ANALYSIS_ASK_LOCS


def analysis_in_background(n_locs: int) -> bool:
    return int(n_locs) >= ANALYSIS_BACKGROUND_LOCS


def _gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.0f} MB"


def load_question_text(name: str, n_locs: int, n_bytes: int) -> str:
    """The question before reading a very large file."""
    return (
        f"{name} has about {int(n_locs):,} localizations ({_gb(n_bytes)} on disk). A cropped axon has "
        f"{AXON_LOCS_TYPICAL_MAX:,} at most, so this looks like a whole acquisition rather than one axon.\n\n"
        f"Loading it needs about {_gb(n_bytes * MEMORY_FACTOR)} of memory and the program draws and selects "
        f"all of it: it can take minutes and the analysis of that many localizations can take hours. "
        f"Crop one axon first (Picasso: pick, then Save picked localizations).\n\n"
        f"Read it anyway? (A bar shows the reading; Cancel stops it.)")


def analysis_question_text(n_locs: int, method: str, estimate_words: str) -> str:
    """The question before analysing a selection with many localizations."""
    return (
        f"The selection has {int(n_locs):,} localizations; a cropped axon has {AXON_LOCS_TYPICAL_MAX:,} at most. "
        f"The analysis ({method}) is expected to take {estimate_words} or more (a lower bound).\n\n"
        f"It will run in the background with a progress bar and a Cancel button. Select a smaller region first "
        f"if this is not one axon.\n\nAnalyse it anyway?")
