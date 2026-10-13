# -*- coding: utf-8 -*-
"""
Where the user left the dividers of a window (manual test 2, item 4).

The MPS analysis window is three panels the user may want to read larger at
different times: the results table, the axon map and its layer list, and
the plot tabs under the map. Each boundary is a splitter, and the sizes the
user drags them to are kept for the next session in a small JSON file of
their own beside the analysis settings (``mps_ui_layout.json``). They are
not analysis settings, so the settings file and every export stay as they
were.

Only a drag by the user writes the file; opening a window never does.
Reading never raises: a missing or corrupt file gives the window's own
defaults.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtWidgets

from tools import mps_settings

logger = logging.getLogger(__name__)

LAYOUT_FILENAME = "mps_ui_layout.json"

# A handle wide enough to be seen and caught, drawn darker than the window's chrome; the colour of a link when
# the pointer is over it.
SPLITTER_STYLE = ("QSplitter::handle { background: #c9c9c9; }"
                  "QSplitter::handle:hover { background: #0078d7; }")
HANDLE_PX = 6


def layout_path() -> str:
    """The layout file, beside the settings file (wherever ``mps_settings.settings_path`` points)."""
    return os.path.join(os.path.dirname(str(mps_settings.settings_path())), LAYOUT_FILENAME)


def _read() -> Dict[str, List[int]]:
    path = layout_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s (%s); the default layout is used", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, List[int]] = {}
    for key, value in data.items():
        if isinstance(value, list) and value and all(isinstance(v, int) and v >= 0 for v in value):
            out[str(key)] = [int(v) for v in value]
    return out


def load_sizes(key: str) -> Optional[List[int]]:
    """The sizes stored for splitter ``key``, or None."""
    sizes = _read().get(key)
    return sizes if sizes and sum(sizes) > 0 else None


def save_sizes(key: str, sizes: List[int]) -> bool:
    """Store the sizes of splitter ``key`` (atomic replace). False when the file cannot be written."""
    data = _read()
    data[key] = [int(v) for v in sizes]
    path = layout_path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.warning("Could not write %s (%s)", path, exc)
        return False


def style_splitter(splitter: QtWidgets.QSplitter) -> None:
    """A handle that is seen and caught."""
    splitter.setHandleWidth(HANDLE_PX)
    splitter.setStyleSheet(SPLITTER_STYLE)


class PersistentSplitter(QtCore.QObject):
    """Keeps a splitter's sizes under ``key``: saved after the user drags it (once the drag settles), restored by
    ``restore()``."""

    def __init__(self, splitter: QtWidgets.QSplitter, key: str) -> None:
        super().__init__(splitter)
        self.splitter = splitter
        self.key = key
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.save)
        splitter.splitterMoved.connect(lambda *_a: self._timer.start())

    def save(self) -> bool:
        return save_sizes(self.key, list(self.splitter.sizes()))

    def restore(self) -> bool:
        """Apply the stored sizes; False when there are none (or not one per panel)."""
        sizes = load_sizes(self.key)
        if sizes is None or len(sizes) != self.splitter.count():
            return False
        self.splitter.setSizes(sizes)
        return True
