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


def load_flag(key: str, default: bool = False) -> bool:
    """A yes/no the user left in a window (F4: the clustering method's panel collapsed), stored as [1] or [0]."""
    data = _read()
    if key not in data:
        return bool(default)
    return bool(data[key] and data[key][0])


def save_flag(key: str, on: bool) -> bool:
    """Store a yes/no under ``key``. False when the file cannot be written."""
    return save_sizes(key, [1 if on else 0])


# A window that opens without a stored size takes this share of the screen's available area (manual test of
# 2026-10-23: "the information is very compressed in a window that could grow") [ad hoc].
WINDOW_SCREEN_SHARE = 0.85
# Table columns sized to their contents are capped at this width; the last one stretches to the table's width.
TABLE_COLUMN_MAX_PX = 320


class _SizeKeeper(QtCore.QObject):
    """Stores the window's size under ``key`` each time it is hidden (closed, accepted, rejected)."""

    def __init__(self, window: QtWidgets.QWidget, key: str) -> None:
        super().__init__(window)
        self.key = key
        window.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802 - Qt's name
        # never raise out of a filter: PyQt aborts the program on an exception here (a window being destroyed)
        try:
            if event.type() == QtCore.QEvent.Type.Hide and isinstance(obj, QtWidgets.QWidget) and obj.isWindow():
                remember_size(obj, self.key)
        except Exception:                                  # noqa: BLE001
            pass
        return False


def open_sized(window: QtWidgets.QWidget, key: str, share: float = WINDOW_SCREEN_SHARE) -> None:
    """Give a window with maximize buttons the size the user left it at (``key`` = [width, height, maximized]),
    or ``share`` of its screen's available area, centred; never larger than the screen. The size is stored again
    each time the window is hidden."""
    if getattr(window, "_size_keeper", None) is None:
        setattr(window, "_size_keeper", _SizeKeeper(window, key))
    flags = window.windowFlags() | QtCore.Qt.WindowType.WindowMaximizeButtonHint         | QtCore.Qt.WindowType.WindowMinimizeButtonHint
    window.setWindowFlags(flags)
    if isinstance(window, QtWidgets.QDialog):
        window.setSizeGripEnabled(True)
    screen = None
    parent = window.parentWidget()
    if parent is not None and parent.window().windowHandle() is not None:
        screen = parent.window().windowHandle().screen()
    screen = screen or QtWidgets.QApplication.primaryScreen()
    if screen is None:
        return
    avail = screen.availableGeometry()
    stored = load_sizes(key)
    if stored is not None and len(stored) >= 2:
        w, h = min(stored[0], avail.width()), min(stored[1], avail.height())
    else:
        w, h = int(avail.width() * share), int(avail.height() * share)
    window.resize(max(w, 200), max(h, 150))
    window.move(avail.x() + (avail.width() - window.width()) // 2, avail.y() + (avail.height() - window.height()) // 2)
    if stored is not None and len(stored) >= 3 and stored[2]:
        window.setWindowState(window.windowState() | QtCore.Qt.WindowState.WindowMaximized)


def remember_size(window: QtWidgets.QWidget, key: str) -> bool:
    """Store the window's size (its normal size when maximized) and whether it was maximized, under ``key``."""
    maxed = bool(window.isMaximized())
    size = window.normalGeometry().size() if maxed and window.normalGeometry().isValid() else window.size()
    return save_sizes(key, [int(size.width()), int(size.height()), 1 if maxed else 0])


def fit_columns(table: QtWidgets.QTableView, cap: int = TABLE_COLUMN_MAX_PX) -> None:
    """Columns as wide as their contents up to ``cap``, the last one stretched to the table's width."""
    table.resizeColumnsToContents()
    header = table.horizontalHeader()
    for c in range(header.count()):
        if header.sectionSize(c) > cap:
            header.resizeSection(c, cap)
    header.setStretchLastSection(True)


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
