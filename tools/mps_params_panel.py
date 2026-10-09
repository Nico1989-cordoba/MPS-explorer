# -*- coding: utf-8 -*-
"""
One place per parameter (UI stage 2, design section 6): the pieces the parameter strip and its mirrors share.

``SlabChoice``
    How the axial slab of the MPS analysis is chosen (design 6.3): automatic (the density peak of the fitted mixture
    +/- the half-width), a mixture component chosen in the MPS analysis window (its mean +/- the half-width), or a
    typed range (the main window's Z min / Z max, which then decide the slab). The main window holds the one
    ``SlabChoice``; the typed range itself lives where it always has, in the main window's cut fields.
``FlowLayout``
    A layout that places its widgets left to right and wraps them onto the next line when the row is full, so the
    strip's groups fit a 1366-px screen without clipping a value, a badge or a range hint (design 6.1).
``mode_words`` / ``origin_words``
    The words beside the slab selector, and "eps 25 nm (paper)" / "eps 30 nm (user; default 25 nm, paper)" for the
    read-only places (the main-window mirrors, the batch label).

Nothing here stores a value: the one value of each parameter is the main window's settings object (design 6.2), and
every origin comes from the parameter registry (``tools.mps_param_registry``).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from PyQt5 import QtCore, QtWidgets

from tools import mps_param_registry as reg

__all__ = ["SlabChoice", "FlowLayout", "mode_words", "origin_words", "SLAB_MODES"]

SLAB_MODES: Tuple[str, ...] = ("automatic", "component", "typed")


@dataclass(frozen=True)
class SlabChoice:
    """How the slab is chosen: ``mode`` in SLAB_MODES; ``centre_nm`` for a chosen component (else None)."""

    mode: str = "automatic"
    centre_nm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode not in SLAB_MODES:
            raise ValueError(f"unknown slab mode {self.mode!r}")
        if self.mode == "component" and self.centre_nm is None:
            raise ValueError("a chosen component needs its centre")


def mode_words(choice: SlabChoice, typed: Optional[Tuple[float, float]] = None) -> str:
    """The slab's mode in words, beside the centre selector (design 6.3, G4: the selector's items are today's)."""
    if choice.mode == "typed" or typed is not None:
        if typed is None:
            return "typed range"
        return f"typed range {typed[0]:,.1f} .. {typed[1]:,.1f} nm: it decides the slab"
    if choice.mode == "component":
        return f"a component you chose (z = {float(choice.centre_nm or 0.0):,.1f} nm)"
    return "automatic: the density peak of the fitted mixture"


def origin_words(key: str, value: Any) -> str:
    """'eps 25 nm (paper)', or 'eps 30 nm (user; default 25 nm, paper)' when the value departs from its default."""
    e = reg.info(key)
    shown = reg.format_value(key, value)
    badge = reg.badge(key, value)
    name = {"dbscan.eps_nm": "eps", "dbscan.min_samples": "min samples", "slab.half_width_nm": "half-width",
            "occupancy.mahalanobis": "Mahalanobis"}.get(key, e.label)
    if badge == "user":
        return f"{name} {shown} (user; default {reg.format_value(key)}, {e.origin})"
    return f"{name} {shown} ({badge})"


class FlowLayout(QtWidgets.QLayout):
    """Left to right, wrapping onto a new line when the next widget would not fit (the Qt flow-layout example)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: List[QtWidgets.QLayoutItem] = []
        self._spacing = int(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802 - Qt's name
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Any:  # noqa: N802
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return int(self._place(QtCore.QRect(0, 0, width, 0), move=False))

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect: QtCore.QRect, move: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line = area.x(), area.y(), 0
        for item in self._items:
            w = item.widget()
            if w is not None and w.isHidden():
                continue
            hint = item.sizeHint()
            nxt = x + hint.width() + self._spacing
            if nxt - self._spacing > area.right() + 1 and line > 0:
                x = area.x()
                y = y + line + self._spacing
                nxt = x + hint.width() + self._spacing
                line = 0
            if move:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = nxt
            line = max(line, hint.height())
        return int(y + line - rect.y() + m.bottom())
