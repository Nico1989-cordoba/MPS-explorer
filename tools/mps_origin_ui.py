# -*- coding: utf-8 -*-
"""
The small widgets that say where a value comes from (UI stage 2, design 12.3, 12.4, 12.8).

``OriginBadge``
    The badge word after a value - paper, derived, simulation, pilot suggestion, user, blank, ad hoc, unreviewed -
    read from the parameter registry (``tools.mps_param_registry``), with the registry's tooltip. Dim text; "user" in
    bold; "ad hoc" in the existing warning text colour. No new colour role (P5).
``RangeHint``
    The documented range beside an editor ("40-200"); "outside 40-200" in the warning text colour when the value is
    outside it. Nothing is clamped: the editor's limits are today's (12.4).
``ParamField``
    An existing editor (its object name, limits and signals untouched: the golden drives it by name, G4) with its
    badge and range hint after it. The badge follows the editor's value - display only, nothing is written back.
``MoreLine``
    A strip group's collapsed "More" line: the read-only research values with their badges and ranges (12.4).
``DetailsPanel``
    Read-only rows of name / value / note under a plot, selectable, collapsible, filled by pure functions with
    numbers the program already computes (12.8). Nothing is computed here.

These are plain QWidgets with their texts in one place, so stage 3 can move them into the side-bar window and stage 4
translate them.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from tools import mps_param_registry as reg
from tools.mps_plot_style import TEXT_DIM, TEXT_DIM_LIGHT, verdict

__all__ = ["OriginBadge", "RangeHint", "ParamField", "MoreLine", "DetailsPanel", "badge_style"]

MORE_TEXT = "More"
DETAILS_TITLE = "Details"
DETAILS_TIP = ("Numbers the analysis already holds for this plot, read-only. Select a row to copy it. Nothing here is "
               "computed for this panel.")


def _dim(dark: bool) -> str:
    return TEXT_DIM if dark else TEXT_DIM_LIGHT


def badge_style(badge: str, dark: bool = False) -> str:
    """The style sheet of a badge word: dim; "user" bold; "ad hoc" and an out-of-range hint in the warning text
    colour of the background they are read on."""
    if badge == "ad hoc":
        return f"color: {verdict('warn', dark=dark)};"
    if badge == "user":
        return f"color: {_dim(dark)}; font-weight: bold;"
    return f"color: {_dim(dark)};"


class OriginBadge(QtWidgets.QLabel):
    """The origin badge of one registry key, for the value shown beside it."""

    def __init__(self, key: str, value: Any = reg.UNSET, parent: Optional[QtWidgets.QWidget] = None, *,
                 dark: bool = False) -> None:
        super().__init__(parent)
        reg.info(key)                       # an unknown key fails here, not later
        self.key = key
        self._dark = bool(dark)
        self.setObjectName(f"badge_{key.replace('.', '_')}")
        self.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.set_value(value)

    def set_value(self, value: Any = reg.UNSET) -> None:
        word = reg.badge(self.key, value)
        self.badge = word
        self.setText(word)
        self.setStyleSheet(badge_style(word, self._dark))
        self.setToolTip(reg.tooltip(self.key, value))


class RangeHint(QtWidgets.QLabel):
    """The documented range of one registry key, flagged (never enforced) when the value is outside it."""

    def __init__(self, key: str, value: Any = reg.UNSET, parent: Optional[QtWidgets.QWidget] = None, *,
                 dark: bool = False) -> None:
        super().__init__(parent)
        reg.info(key)
        self.key = key
        self._dark = bool(dark)
        self.setObjectName(f"range_{key.replace('.', '_')}")
        self.set_value(value)

    def set_value(self, value: Any = reg.UNSET) -> None:
        text = reg.range_hint(self.key, value)
        self.outside = bool(value is not reg.UNSET and reg.outside_range(self.key, value))
        self.setText(text)
        self.setVisible(bool(text))
        self.setStyleSheet(f"color: {verdict('warn', dark=self._dark)};" if self.outside
                           else f"color: {_dim(self._dark)};")
        self.setToolTip(reg.tooltip(self.key, value))


def _value_of(editor: QtWidgets.QWidget) -> Any:
    """The value an editor shows: a spin box's value, a check box's state, a combo's data or text."""
    if isinstance(editor, QtWidgets.QSpinBox):
        return int(editor.value())
    if isinstance(editor, QtWidgets.QDoubleSpinBox):
        return float(editor.value())
    if isinstance(editor, QtWidgets.QAbstractButton) and editor.isCheckable():
        return bool(editor.isChecked())
    if isinstance(editor, QtWidgets.QComboBox):
        data = editor.currentData()
        return data if data is not None else editor.currentText()
    if isinstance(editor, QtWidgets.QLineEdit):
        text = editor.text().strip()
        try:
            return float(text)
        except ValueError:
            return text or None
    raise TypeError(f"no value reader for {type(editor).__name__}")


class ParamField(QtWidgets.QWidget):
    """An editor with its badge and range hint after it.

    ``editor`` keeps its object name, limits and signals; the field only listens (valueChanged / toggled /
    currentIndexChanged / textChanged) and redraws the badge and the hint. ``value`` replaces the editor's own
    reading (e.g. a mirror rendered as text). The registry tooltip goes on the badge, the hint and, appended to its
    own, the editor.
    """

    def __init__(self, key: str, editor: QtWidgets.QWidget, parent: Optional[QtWidgets.QWidget] = None, *,
                 dark: bool = False, value: Optional[Callable[[], Any]] = None, show_range: bool = True) -> None:
        super().__init__(parent)
        reg.info(key)
        self.key = key
        self.editor = editor
        self._read = value if value is not None else (lambda: _value_of(editor))
        self._editor_tip = str(editor.toolTip())
        self.setObjectName(f"field_{key.replace('.', '_')}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(editor)
        self.badge = OriginBadge(key, dark=dark)
        lay.addWidget(self.badge)
        self.hint = RangeHint(key, dark=dark)
        self.hint.setVisible(show_range and bool(self.hint.text()))
        self._show_range = bool(show_range)
        lay.addWidget(self.hint)
        lay.addStretch(1)
        for name in ("valueChanged", "toggled", "currentIndexChanged", "textChanged"):
            signal = getattr(editor, name, None)
            if signal is not None:
                try:
                    signal.connect(lambda *_a: self.refresh())
                except (TypeError, AttributeError):
                    continue
        self.refresh()

    def value(self) -> Any:
        return self._read()

    def refresh(self) -> None:
        """Badge, hint and tooltips from the value the editor shows now (display only)."""
        value = self._read()
        self.badge.set_value(value)
        self.hint.set_value(value)
        self.hint.setVisible(self._show_range and bool(self.hint.text()))
        tip = reg.tooltip(self.key, value)
        self.editor.setToolTip((self._editor_tip + "\n\n" if self._editor_tip else "") + tip)

    @property
    def outside(self) -> bool:
        return bool(self.hint.outside)


class MoreLine(QtWidgets.QWidget):
    """A strip group's "More" line: collapsed by default, the read-only values of the group with their badges and
    ranges, each item's registry tooltip on the line."""

    def __init__(self, group: str, parent: Optional[QtWidgets.QWidget] = None, *, dark: bool = False,
                 expanded: bool = False) -> None:
        super().__init__(parent)
        self.group = group
        self.setObjectName(f"more_{group}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.button = QtWidgets.QToolButton()
        self.button.setObjectName(f"more_toggle_{group}")
        self.button.setText(MORE_TEXT)
        self.button.setCheckable(True)
        self.button.setAutoRaise(True)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        lay.addWidget(self.button, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.label = QtWidgets.QLabel()
        self.label.setObjectName(f"more_text_{group}")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setStyleSheet(f"color: {_dim(dark)};")
        lay.addWidget(self.label, 1)
        self.button.toggled.connect(self._fold)
        self.refresh()
        self.button.setChecked(bool(expanded))
        self._fold(bool(expanded))

    def refresh(self) -> None:
        self.label.setText(reg.more_line(self.group))
        tips = [reg.tooltip(key) for key, *_rest in reg.more_items(self.group)]
        self.setToolTip("\n\n".join(tips))
        self.label.setToolTip(self.toolTip())

    def _fold(self, expanded: bool) -> None:
        self.label.setVisible(bool(expanded))
        self.button.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)

    def is_expanded(self) -> bool:
        return bool(self.button.isChecked())


class DetailsPanel(QtWidgets.QWidget):
    """Read-only rows of name / value / note (and an optional badge), selectable, collapsible.

    ``set_rows`` takes any sequence of tuples ``(name, value[, note[, badge]])`` - what the pure builders return -
    so the builders need no Qt."""

    def __init__(self, title: str = DETAILS_TITLE, parent: Optional[QtWidgets.QWidget] = None, *,
                 collapsed: bool = False, object_name: str = "details") -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.button = QtWidgets.QToolButton()
        self.button.setObjectName(f"{object_name}_toggle")
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setAutoRaise(True)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setToolTip(DETAILS_TIP)
        lay.addWidget(self.button)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setObjectName(f"{object_name}_table")
        self.table.setHorizontalHeaderLabels(["", "value", "note"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setWordWrap(True)
        self.table.setToolTip(DETAILS_TIP)
        lay.addWidget(self.table)
        self.button.toggled.connect(self._fold)
        self.button.setChecked(not collapsed)
        self._fold(not collapsed)
        self._rows: List[tuple] = []

    def set_rows(self, rows: Sequence[Sequence[Any]]) -> None:
        self._rows = [tuple(r) for r in rows]
        self.table.setRowCount(len(self._rows))
        for i, row in enumerate(self._rows):
            name = str(row[0])
            value = str(row[1])
            note = str(row[2]) if len(row) > 2 and row[2] else ""
            badge = str(row[3]) if len(row) > 3 and row[3] else ""
            for col, text in enumerate((name, value + (f"  [{badge}]" if badge else ""), note)):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(text)
                self.table.setItem(i, col, item)
        self.table.resizeColumnsToContents()

    def rows(self) -> List[tuple]:
        return list(self._rows)

    def text(self) -> str:
        """Every row as one line of text (what a copy of the panel gives)."""
        out = []
        for row in self._rows:
            parts = [str(row[0]), str(row[1])]
            if len(row) > 3 and row[3]:
                parts[-1] += f" [{row[3]}]"
            if len(row) > 2 and row[2]:
                parts.append(str(row[2]))
            out.append(": ".join(parts[:2]) + ("" if len(parts) < 3 else f" ({parts[2]})"))
        return "\n".join(out)

    def _fold(self, expanded: bool) -> None:
        self.table.setVisible(bool(expanded))
        self.button.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)

    def is_collapsed(self) -> bool:
        return not self.button.isChecked()
