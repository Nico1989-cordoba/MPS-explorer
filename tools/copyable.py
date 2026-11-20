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

import os
from typing import Any, Iterable, List, Optional

# The exports' rule, defined without Qt in tools.results_table (the computing modules write CSVs too);
# re-exported here, where the windows have always imported it from.
from tools.results_table import free_path  # noqa: F401

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
    setattr(widget, "_mps_copyable", True)
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


def write_text_file(path: str, text: str) -> str:
    """Write ``text`` (UTF-8) to ``path`` or, when a file is there, to the next free name; returns the path
    written."""
    target = free_path(path)
    with open(target, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return target


# ---------------------------------------------------------------------------------------------------------------
# Every informative label of the program, and every plot title (manual test of 2026-10-23: "I cannot select or
# copy the informative texts"): applied once, application-wide, when each widget is first polished.
# ---------------------------------------------------------------------------------------------------------------
COPY_TITLE_TEXT = "Copy title"
_SELECT = QtCore.Qt.TextInteractionFlag.TextSelectableByMouse


def label_is_informative(label: QtWidgets.QLabel) -> bool:
    """A label that shows text (not an icon or a picture) in a window of the program (not a tooltip)."""
    if type(label).__name__ == "QTipLabel" or label.objectName() == "qtooltip_label":
        return False
    win = label.window()
    if win is not None and win.windowType() == QtCore.Qt.WindowType.ToolTip:
        return False
    pix = label.pixmap()
    if pix is not None and not pix.isNull():
        return False
    return label.movie() is None


def copy_title_action(plot_widget: Any) -> None:
    """Add 'Copy title' to a pyqtgraph plot's own context menu (the title is drawn, not a label)."""
    try:
        vb = plot_widget.getPlotItem().getViewBox()
        menu = getattr(vb, "menu", None)
    except Exception:                                  # noqa: BLE001 - not a plot after all
        return
    if menu is None or any(a.text() == COPY_TITLE_TEXT for a in menu.actions()):
        return

    # a weak reference: the menu's action must not keep a destroyed plot's Python side (and its items) alive
    import weakref
    ref = weakref.ref(plot_widget)

    def copy() -> None:
        from tools.figure_export import plain_title
        widget = ref()
        if widget is None:
            return
        try:
            _to_clipboard(plain_title(widget.getPlotItem()))
        except RuntimeError:                           # the plot is gone
            pass

    action = QtWidgets.QAction(COPY_TITLE_TEXT, menu)
    action.triggered.connect(lambda _checked=False: copy())
    menu.addAction(action)


def make_labels_selectable(root: QtWidgets.QWidget) -> int:
    """Every informative QLabel under ``root`` (and ``root`` itself) selectable with the mouse: Ctrl+C and Qt's
    context menu 'Copy' then copy it. Links and the flags a label already had are kept; mouse only, so the Tab
    order does not change. Returns how many labels were made selectable now."""
    labels = list(root.findChildren(QtWidgets.QLabel))
    if isinstance(root, QtWidgets.QLabel):
        labels.append(root)
    n = 0
    for label in labels:
        try:
            if label_is_informative(label) and not (label.textInteractionFlags() & _SELECT):
                label.setTextInteractionFlags(label.textInteractionFlags() | _SELECT)
                n += 1
        except RuntimeError:                           # a label being destroyed
            continue
    return n


def install_text_copy_everywhere(app: Optional[QtWidgets.QApplication] = None) -> None:
    """Every window the program shows gets its informative labels selectable (``make_labels_selectable``) each
    time it becomes the active window, as Ctrl+F is installed (tools.ui_search). Not an application event filter:
    a Python filter sees every event of every object, pyqtgraph's items too, and changed their teardown (an abort
    in the GUI tests). Windows whose labels change after they are shown call ``make_labels_selectable``
    themselves. Call once, after the QApplication exists; a second call does nothing."""
    qapp: Any = app or QtWidgets.QApplication.instance()
    if qapp is None or qapp.property("text_copy_everywhere"):
        return
    qapp.setProperty("text_copy_everywhere", True)

    def on_active(_qwindow: Any) -> None:
        w = QtWidgets.QApplication.activeWindow()
        if w is not None:
            make_labels_selectable(w)

    qapp.focusWindowChanged.connect(on_active)
