# -*- coding: utf-8 -*-
"""
A parameter strip that scrolls instead of growing the window past the screen (F5d, manual test of 2026-10-19).

The MPS analysis window lays its parameter groups out in one wrapping strip above the results and the plots. A
group that grows -- the Emitters box of the clustering method, a "More" line opened -- used to raise the
window's minimum height with it: with the Emitters box shown the window asked for ~1,275 px, more than a laptop
screen (768 px, or 1080 px at 125-150 % scaling), and the box itself was drawn below the bottom of the screen,
where the user never saw it.

``CappedScrollArea`` holds such a strip. It asks for the strip's own height for the window's width, up to a cap
the window sets (``set_cap``), and never for more than the strip; above the cap the strip scrolls vertically,
never horizontally (the strip wraps to the width). Its minimum is a small constant (``MIN_CAP_PX``), so the
window's minimum height no longer depends on what the strip holds, and a short window squeezes the strip (which
then scrolls) rather than the other way round.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt5 import QtCore, QtWidgets

# The strip is never squeezed below this height [ad hoc: two lines of the strip's controls, ~ 2 x 60 px, so the
# first line of every group stays readable on a short screen].
MIN_CAP_PX = 120
# How much of a revealed box is brought into sight [ad hoc: its title and its first three rows, ~ 4 x 30 px].
REVEAL_LAST_PX = 130


class CappedScrollArea(QtWidgets.QScrollArea):
    """A vertical-only scroll area whose height is its content's height for the current width, at most ``cap``."""

    def __init__(self, content: QtWidgets.QWidget, object_name: str = "",
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        if object_name:
            self.setObjectName(object_name)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setWidget(content)
        self._cap = 10 ** 6
        self._pending = False
        content.installEventFilter(self)

    # -- the height ---------------------------------------------------------
    def content_height(self) -> int:
        """The content's height when laid out at the viewport's width (its wrapped height)."""
        w = self.widget()
        if w is None:
            return 0
        width = max(1, self.viewport().width())
        lay = w.layout()
        if lay is not None and lay.hasHeightForWidth():
            return int(lay.totalHeightForWidth(width))
        if w.hasHeightForWidth():
            return int(w.heightForWidth(width))
        return int(w.sizeHint().height())

    def cap(self) -> int:
        return int(self._cap)

    def set_cap(self, cap_px: int) -> None:
        """The most height the strip takes; above it, it scrolls (never below ``MIN_CAP_PX``)."""
        cap = max(MIN_CAP_PX, int(cap_px))
        if cap != self._cap:
            self._cap = cap
            self.fit()

    def wanted_height(self) -> int:
        """The content's height, up to the cap (frame included)."""
        return int(min(self.content_height(), self._cap) + 2 * self.frameWidth())

    def fit(self) -> None:
        """Never taller than the content; ask the layout again for the height wanted now."""
        self._pending = False
        most = max(1, self.content_height() + 2 * self.frameWidth())
        if self.maximumHeight() != most:
            self.setMaximumHeight(most)
        self.updateGeometry()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt's name
        return QtCore.QSize(super().sizeHint().width(), max(1, self.wanted_height()))

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt's name
        low = min(self.content_height(), MIN_CAP_PX) + 2 * self.frameWidth()
        return QtCore.QSize(0, max(1, low))

    def schedule_fit(self) -> None:
        """Fit once the event loop has laid the content out (coalesced)."""
        if not self._pending:
            self._pending = True
            QtCore.QTimer.singleShot(0, self.fit)

    def scrolls(self) -> bool:
        """True when part of the content is out of sight (the vertical bar has somewhere to go)."""
        return bool(self.verticalScrollBar().maximum() > 0)

    def reveal(self, first: QtWidgets.QWidget, last: Optional[QtWidgets.QWidget] = None,
               last_px: int = REVEAL_LAST_PX) -> None:
        """Once the layout has settled, scroll so that ``first`` starts in sight and then so that the first
        ``last_px`` of ``last`` are in sight too: with a short viewport ``last`` wins (the top of ``first`` may
        then be cut). Widgets outside the content, or hidden, are skipped."""
        def go() -> None:
            self.fit()
            content = self.widget()
            if content is None:
                return
            for w, px in ((first, 0), (last, int(last_px))):
                if w is None or not content.isAncestorOf(w) or not w.isVisibleTo(content):
                    continue
                top = w.mapTo(content, QtCore.QPoint(0, 0))
                self.ensureVisible(top.x(), top.y(), 0, 0)
                if px:
                    self.ensureVisible(top.x(), top.y() + min(px, w.height()), 0, 0)
        QtCore.QTimer.singleShot(0, lambda: QtCore.QTimer.singleShot(0, go))

    # -- Qt -------------------------------------------------------------------
    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt's name
        if obj is self.widget() and event.type() in (QtCore.QEvent.Type.LayoutRequest,
                                                      QtCore.QEvent.Type.Resize):
            self.schedule_fit()
        return bool(super().eventFilter(obj, event))

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        # a new width wraps the strip differently
        self.schedule_fit()
