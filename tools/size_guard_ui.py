# -*- coding: utf-8 -*-
"""
The questions and the reading bar of ``tools.mps_size_guard`` (Qt side).

``confirm`` asks a yes/no question (tests and scripts answer through ``ANSWER``; offscreen, with nobody to ask, the
answer is "continue": nothing that ran before the question existed changes). ``read_with_progress`` runs a long
read in a worker thread while a dialog with a moving bar, the time spent and a Cancel button keeps the window
answering. Python cannot stop a read from outside: Cancel closes the dialog at once and the read, which cannot be
interrupted half-way, finishes in the background and its result is thrown away (the thread is held until then).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Optional, Set

from PyQt5 import QtCore, QtWidgets

from tools.perimeter_cost_ui import interactive

# Tests and scripts: answer without a dialog (None: ask the user). Takes the text, returns True to go on.
ANSWER: Optional[Callable[[str], bool]] = None
ASKED: List[str] = []

# Threads whose result was cancelled, kept until they end.
_ORPHANS: Set[threading.Thread] = set()


class ReadCancelled(Exception):
    """The user pressed Cancel while a file was being read: nothing was loaded."""


def confirm(parent: Any, title: str, text: str, go_label: str = "Continue") -> bool:
    ASKED.append(text)
    if ANSWER is not None:
        return bool(ANSWER(text))
    if not interactive():
        return True
    box = QtWidgets.QMessageBox(parent)
    box.setObjectName("size_guard_dialog")
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    go = box.addButton(go_label, QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    cancel = box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    box.exec_()
    return box.clickedButton() is go


class ReadDialog(QtWidgets.QDialog):
    """A moving bar, the time spent and Cancel, for one long read."""

    def __init__(self, words: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("size_guard_read_dialog")
        self.setWindowTitle("Reading the file")
        self.setModal(True)
        lay = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(words)
        self.label.setWordWrap(True)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 0)               # a read has no count to show: the bar moves, the time says it lives
        self.bar.setObjectName("size_guard_read_bar")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setObjectName("btn_size_guard_cancel")
        self.btn_cancel.clicked.connect(self.reject)
        lay.addWidget(self.label)
        lay.addWidget(self.bar)
        lay.addWidget(self.btn_cancel)
        self._words = words
        self._t0 = time.monotonic()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.resize(420, 120)

    def _tick(self) -> None:
        s = int(time.monotonic() - self._t0)
        self.label.setText(f"{self._words}\nElapsed {s // 60:02d}:{s % 60:02d}. Cancel closes this window; "
                           f"the read, which cannot be interrupted half-way, ends in the background and is discarded.")


def read_with_progress(parent: Any, fn: Callable[[], Any], words: str = "Reading the file...") -> Any:
    """``fn()`` in a worker thread with the reading dialog; its value (or its exception). Raises ``ReadCancelled``
    after Cancel. Offscreen or from a worker thread: ``fn()`` here, as before."""
    if not interactive():
        return fn()
    box: dict = {}

    def body() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:                     # noqa: BLE001 - handed to the GUI thread
            box["error"] = exc
        finally:
            box["done"] = True

    thread = threading.Thread(target=body, daemon=True, name="size-guard-read")
    dialog = ReadDialog(words, parent)
    poll = QtCore.QTimer(dialog)
    poll.setInterval(100)

    def check() -> None:
        if box.get("done"):
            dialog.accept()
    poll.timeout.connect(check)
    poll.start()
    thread.start()
    dialog.exec_()
    poll.stop()
    if not box.get("done"):                              # Cancel: let the read end by itself, drop what it returns
        _ORPHANS.add(thread)
        QtCore.QTimer.singleShot(2000, _reap)
        raise ReadCancelled("Reading was cancelled; nothing was loaded.")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _reap() -> None:
    for t in list(_ORPHANS):
        if not t.is_alive():
            _ORPHANS.discard(t)
    if _ORPHANS:
        QtCore.QTimer.singleShot(2000, _reap)
