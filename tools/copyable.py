# -*- coding: utf-8 -*-
"""
Text that can be copied (manual test 2, item 3).

Every number a window shows should be one selection and Ctrl+C away from a
spreadsheet or a message, without retyping it:

``copyable_table``
    A table (``QTableWidget`` / ``QTableView``): the selected cells, as
    tab-separated rows (the cells between two selected ones left empty), go
    to the clipboard with Ctrl+C or the context menu's "Copy"; with nothing
    selected, the whole table with its header line. Excel and LibreOffice
    paste that as cells.
``copyable_list``
    A list: the selected lines (all of them with none selected), one per
    line.
``selectable_label``
    A label whose text the mouse selects; Qt's own context menu then offers
    "Copy", and Ctrl+C copies the selection.

Nothing here changes what a widget shows; only how its text leaves it.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

COPY_TEXT = "Copy"
COPY_ALL_TEXT = "Copy all"


def _cell_text(model: Any, row: int, col: int) -> str:
    value = model.data(model.index(row, col), QtCore.Qt.ItemDataRole.DisplayRole)
    text = "" if value is None else str(value)
    # A cell's own tab or line break would split it into two cells when pasted.
    return " ".join(text.replace("\t", " ").splitlines())


def table_text(view: QtWidgets.QAbstractItemView, everything: bool = False) -> str:
    """The selected cells as tab-separated rows; every visible cell with the header line when nothing is selected
    (or ``everything``)."""
    model = view.model()
    if model is None:
        return ""
    hidden_col = getattr(view, "isColumnHidden", lambda _c: False)
    hidden_row = getattr(view, "isRowHidden", lambda _r: False)
    selection = view.selectionModel()
    indexes = [] if (everything or selection is None) else list(selection.selectedIndexes())
    if not indexes:
        cols = [c for c in range(model.columnCount()) if not hidden_col(c)]
        rows = [r for r in range(model.rowCount()) if not hidden_row(r)]
        lines: List[str] = []
        header = [model.headerData(c, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.DisplayRole)
                  for c in cols]
        if any(h not in (None, "") for h in header):
            lines.append("\t".join("" if h is None else str(h) for h in header))
        lines += ["\t".join(_cell_text(model, r, c) for c in cols) for r in rows]
        return "\n".join(lines)
    chosen = {(i.row(), i.column()) for i in indexes}
    rows = sorted({r for r, _c in chosen})
    cols = sorted({c for _r, c in chosen if not hidden_col(c)})
    return "\n".join("\t".join(_cell_text(model, r, c) if (r, c) in chosen else "" for c in cols)
                     for r in rows)


def list_text(view: QtWidgets.QListWidget) -> str:
    """The selected lines, or every line when none is selected."""
    items = view.selectedItems() or [view.item(i) for i in range(view.count())]
    return "\n".join(item.text() for item in items if item is not None)


def _to_clipboard(text: str) -> None:
    QtWidgets.QApplication.clipboard().setText(text)


class _CopyKey(QtCore.QObject):
    """Ctrl+C (the platform's copy key) on a widget: ``copy`` instead of the view's own single-cell copy."""

    def __init__(self, parent: QtWidgets.QWidget, copy: Any) -> None:
        super().__init__(parent)
        self._copy = copy

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt's name
        if event.type() == QtCore.QEvent.Type.KeyPress and event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self._copy()
            return True
        return False


def _install(widget: QtWidgets.QWidget, copy: Any, copy_all: Optional[Any]) -> None:
    if getattr(widget, "_mps_copyable", False):
        return
    widget._mps_copyable = True                                   # type: ignore[attr-defined]
    widget.installEventFilter(_CopyKey(widget, copy))
    widget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
    action = QtWidgets.QAction(COPY_TEXT, widget)
    action.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy))
    # Shown in the menu; the key itself is handled by the event filter (a widget shortcut would also need focus
    # rules the filter does not).
    action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
    action.setShortcutVisibleInContextMenu(True)
    action.triggered.connect(lambda _c=False: copy())
    widget.addAction(action)
    if copy_all is not None:
        everything = QtWidgets.QAction(COPY_ALL_TEXT, widget)
        everything.triggered.connect(lambda _c=False: copy_all())
        widget.addAction(everything)


def copyable_table(view: QtWidgets.QAbstractItemView) -> QtWidgets.QAbstractItemView:
    """Selected cells to the clipboard as tab-separated rows (Ctrl+C, "Copy"); "Copy all" for the whole table."""
    view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    _install(view, lambda: _to_clipboard(table_text(view)),
             lambda: _to_clipboard(table_text(view, everything=True)))
    return view


def copyable_list(view: QtWidgets.QListWidget) -> QtWidgets.QListWidget:
    """Selected lines to the clipboard (Ctrl+C, "Copy")."""
    view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    _install(view, lambda: _to_clipboard(list_text(view)), None)
    return view


def selectable_label(label: QtWidgets.QLabel) -> QtWidgets.QLabel:
    """A label whose text the mouse (and the keyboard) selects; Qt's context menu then has "Copy"."""
    flags = label.textInteractionFlags()
    label.setTextInteractionFlags(flags | QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                                  | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard)
    return label


def selectable_labels(labels: Iterable[Optional[QtWidgets.QLabel]]) -> None:
    for label in labels:
        if label is not None:
            selectable_label(label)
