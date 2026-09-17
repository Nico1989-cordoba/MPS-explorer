# -*- coding: utf-8 -*-
"""
Dragging a file onto a field, beside the Browse buttons.

The paths in this project are long -- five folders deep, with spaces and
accents, and often two copies of the same tree -- so picking a file in a
dialog is slow and easy to get wrong. Dragging it out of the file
explorer names the file itself, which no dialog can misinterpret.

Qt's line edits already take a drop, but they paste what the explorer
hands them: "file:///C:/...", which no loader opens. So the drop is
intercepted before the widget sees it, turned into a local path, and
accepted only when the file is of a kind that target takes. The handler
runs from the event loop rather than inside the drop, so the window it
was dragged from is released first and the handler may open dialogs.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

# What each kind of target takes.
LOCALIZATION_SUFFIXES = (".hdf5", ".h5", ".csv")
IMAGE_SUFFIXES = (".tif", ".tiff")


def dropped_paths(event: Any, suffixes: Sequence[str] = ()) -> List[str]:
    """
    The local files a drag carries, keeping only the suffixes asked for.

    A drag from the file explorer carries URLs; one from a browser or a
    text editor may carry only text, which is not a file and is ignored.
    """
    data = event.mimeData()
    if data is None or not data.hasUrls():
        return []
    wanted = tuple(suffix.lower() for suffix in suffixes)
    paths: List[str] = []
    for url in data.urls():
        path = url.toLocalFile()
        if not path or os.path.isdir(path):
            continue
        if wanted and not path.lower().endswith(wanted):
            continue
        paths.append(os.path.normpath(path))
    return paths


class _FileDropFilter(QtCore.QObject):
    """Turns a drag of files onto one widget into a call with their paths."""

    def __init__(self, widget: QtWidgets.QWidget, suffixes: Sequence[str],
                 handler: Callable[[List[str]], None]) -> None:
        super().__init__(widget)
        self._suffixes = tuple(suffixes)
        self._handler = handler

    def eventFilter(self, obj: Any, event: Any) -> bool:
        kind = event.type()
        if kind in (QtCore.QEvent.Type.DragEnter,
                    QtCore.QEvent.Type.DragMove):
            if not dropped_paths(event, self._suffixes):
                return False          # let the widget refuse it as usual
            event.acceptProposedAction()
            return True
        if kind == QtCore.QEvent.Type.Drop:
            paths = dropped_paths(event, self._suffixes)
            if not paths:
                return False
            event.acceptProposedAction()
            # Out of the drop, so the explorer is not left waiting while
            # a file loads or a message box asks something.
            QtCore.QTimer.singleShot(0, lambda: self._handler(paths))
            return True
        return False


def accept_files(
    widget: QtWidgets.QWidget,
    suffixes: Sequence[str],
    handler: Callable[[List[str]], None],
    *,
    hint: Optional[str] = None,
) -> None:
    """
    Let ``widget`` take files dragged onto it.

    ``handler`` is called with the dropped paths, most-recent drag first,
    once the drop has been accepted. ``hint`` is shown as the field's
    placeholder, where the widget has one, and added to its tooltip, so
    the way in is visible without being written in a manual.
    """
    widget.setAcceptDrops(True)
    widget.installEventFilter(_FileDropFilter(widget, suffixes, handler))
    if hint:
        if hasattr(widget, "setPlaceholderText") and not widget.text():
            widget.setPlaceholderText(hint)
        tooltip = widget.toolTip()
        widget.setToolTip(f"{tooltip}\n{hint}" if tooltip else hint)


def first_existing_dir(*candidates: Optional[str]) -> str:
    """
    The first of these that names an existing directory, or the home.

    A candidate that is a file, or a folder that has since been renamed
    or removed, falls back to the folder holding it: a dialog opening one
    level up from where the user was is better than one opening wherever
    it happens to have been left.
    """
    for candidate in candidates:
        if not candidate:
            continue
        path = candidate if os.path.isdir(candidate) \
            else os.path.dirname(candidate)
        if path and os.path.isdir(path):
            return path
    return os.path.expanduser("~")
