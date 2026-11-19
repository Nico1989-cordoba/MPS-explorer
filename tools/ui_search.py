# -*- coding: utf-8 -*-
"""
Ctrl+F in every window of MPS Explorer: find a word among what the window shows (F3, the user's request of
2026-10-22).

The windows hold many controls, groups and tables, and much of what they mean is in tooltips; the user asked for a
keyword search in all of them. ``install(window)`` gives a window Ctrl+F with one line. The bar it opens is small
and sits at the window's top-right corner: a field, previous / next, a counter ("3 of 12") and a close button.

What is searched, in the window and only there: the text of labels, buttons and check boxes, group titles, tab
titles, combo boxes' current text, read-only text panes, table and tree headers and cells (the layer panels are
check boxes and labels, so they are in), and the tooltip of each of them (a tab's and a cell's own tooltip too).
Upper and lower case and accents are alike ("Perimetro" finds "Perímetro"). A widget hidden on purpose (a closed
"More" line, a collapsed panel, a note that does not apply) is not searched; a page of a tab or of a stacked widget
that is merely not the current one is, and going to a match opens it.

Going to a match opens the tab (or stacked page) that holds it, scrolls every scroll area around it (and a table
to its cell), and outlines it for a moment (FLASH_MS). When the word was found in a tooltip, the counter says so
and the tooltip is shown beside the widget.

Keys: Ctrl+F opens the bar in the active window (or selects its text); Enter or F3 goes to the next match,
Shift+Enter or Shift+F3 to the previous one; Esc closes the bar (and only the bar: a dialog stays open).

Cost when closed: one QShortcut per window. The bar, its two F3 shortcuts and its event filter are made the first
time Ctrl+F is pressed in that window; the filter (which keeps the bar in its corner when the window is resized) is
installed only while the bar is open. ``install_everywhere(app)`` installs the search in every window the
program opens, when it first becomes active or a widget in it first takes the focus (two signals, no event filter).

Nothing here reads or changes an analysis, a setting or an export.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5 import sip                       # PyQt5 >= 5.11 ships it as PyQt5.sip

CONTROLLER_NAME = "ui_search_controller"
BAR_NAME = "ui_search_bar"
# How long a match stays outlined [ad hoc: long enough to be found by the eye, short enough not to cover anything].
FLASH_MS = 1600
# The search waits this long after the last key typed [ad hoc: a typing pause].
DEBOUNCE_MS = 150
# Cells read per table or tree at most [ad hoc: a table of a few thousand clusters is searched whole; a huge one
# does not freeze the window].
MAX_CELLS_PER_VIEW = 20000
# The outline: Okabe-Ito orange, the palette of tools/mps_plot_style (seen on light and dark backgrounds).
FLASH_COLOUR = "#E69F00"

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lower case, without accents, tags or mnemonic '&', spaces collapsed: what a search compares."""
    s = str(text or "")
    if "<" in s and ">" in s:
        s = html.unescape(_TAG.sub(" ", s.replace("<br>", "\n")))
    s = s.replace("&&", "\0").replace("&", "").replace("\0", "&")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return _SPACE.sub(" ", s.casefold()).strip()


def plain(text: str) -> str:
    """The text as shown (tags removed), for the counter's tooltip."""
    s = str(text or "")
    if "<" in s and ">" in s:
        s = html.unescape(_TAG.sub(" ", s.replace("<br>", "\n")))
    return _SPACE.sub(" ", s).strip()


@dataclass
class Match:
    """One place where the words were found."""

    widget: Any                        # the QWidget (a view for a cell or a header, a tab bar for a tab)
    kind: str                          # label, button, group, tab, combo, text, cell, header, tooltip
    text: str                          # the text it was found in
    in_tooltip: bool = False
    index: Any = None                  # QPersistentModelIndex of a cell
    section: int = -1                  # a header's section or a tab's index
    orientation: int = 1               # a header's orientation (Qt.Horizontal = 1)

    def alive(self) -> bool:
        if sip.isdeleted(self.widget):
            return False
        return self.index is None or bool(self.index.isValid())


# ---------------------------------------------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------------------------------------------
def _reachable(w: QtWidgets.QWidget, root: QtWidgets.QWidget) -> bool:
    """Shown, or hidden only because it is a page of a tab / stacked widget that is not the current one."""
    node: Optional[QtWidgets.QWidget] = w
    while node is not None and node is not root:
        if node.isHidden():
            parent = node.parentWidget()
            if not isinstance(parent, QtWidgets.QStackedWidget):
                return False
        node = node.parentWidget()
    return node is root


def _model_texts(view: QtWidgets.QAbstractItemView) -> Iterator[Tuple[Any, str, bool]]:
    """(index, text, is tooltip) of the cells of a table, tree or list, up to MAX_CELLS_PER_VIEW."""
    model = view.model()
    if model is None:
        return
    budget = [MAX_CELLS_PER_VIEW]
    table = view if isinstance(view, QtWidgets.QTableView) else None
    tree = view if isinstance(view, QtWidgets.QTreeView) else None

    def walk(parent: QtCore.QModelIndex) -> Iterator[Tuple[Any, str, bool]]:
        for r in range(model.rowCount(parent)):
            if table is not None and table.isRowHidden(r):
                continue
            if tree is not None and tree.isRowHidden(r, parent):
                continue
            for c in range(model.columnCount(parent)):
                if table is not None and table.isColumnHidden(c):
                    continue
                if tree is not None and tree.isColumnHidden(c):
                    continue
                if budget[0] <= 0:
                    return
                budget[0] -= 1
                idx = model.index(r, c, parent)
                for role, tip in ((QtCore.Qt.ItemDataRole.DisplayRole, False),
                                  (QtCore.Qt.ItemDataRole.ToolTipRole, True)):
                    value = model.data(idx, role)
                    if value is not None and str(value) != "":
                        yield idx, str(value), tip
                if tree is not None and c == 0 and model.hasChildren(idx):
                    yield from walk(idx)

    yield from walk(QtCore.QModelIndex())


def _header_texts(view: QtWidgets.QAbstractItemView) -> Iterator[Tuple[int, int, str, bool]]:
    """(orientation, section, text, is tooltip) of a table's or tree's visible headers."""
    model = view.model()
    if model is None:
        return
    headers: List[Tuple[int, Any, int]] = []
    if isinstance(view, QtWidgets.QTableView):
        headers = [(1, view.horizontalHeader(), model.columnCount()), (2, view.verticalHeader(), model.rowCount())]
    elif isinstance(view, QtWidgets.QTreeView):
        headers = [(1, view.header(), model.columnCount())]
    for orient, header, n in headers:
        if header is None or header.isHidden():
            continue
        qt_orient = QtCore.Qt.Orientation(orient)
        for section in range(min(n, MAX_CELLS_PER_VIEW)):
            if header.isSectionHidden(section):
                continue
            for role, tip in ((QtCore.Qt.ItemDataRole.DisplayRole, False),
                              (QtCore.Qt.ItemDataRole.ToolTipRole, True)):
                value = model.headerData(section, qt_orient, role)
                if value is not None and str(value) != "":
                    yield orient, section, str(value), tip


def find_matches(root: QtWidgets.QWidget, query: str,
                 exclude: Optional[QtWidgets.QWidget] = None) -> List[Match]:
    """Every place under ``root`` whose text or tooltip holds ``query`` (case and accents alike), in the order
    of the widgets (as they were made, depth first). ``exclude``: a widget not to search (the bar itself)."""
    q = normalise(query)
    if not q:
        return []
    out: List[Match] = []

    def hit(text: str) -> bool:
        return bool(text) and q in normalise(text)

    for w in root.findChildren(QtWidgets.QWidget):
        if exclude is not None and (w is exclude or exclude.isAncestorOf(w)):
            continue
        if w.isWindow() or not _reachable(w, root):
            continue
        if isinstance(w, QtWidgets.QHeaderView):
            continue                                   # searched with its view (_header_texts)
        own: List[Tuple[str, str]] = []
        if isinstance(w, QtWidgets.QLabel):
            own.append(("label", w.text()))
        elif isinstance(w, QtWidgets.QAbstractButton):
            own.append(("button", w.text()))
        elif isinstance(w, QtWidgets.QGroupBox):
            own.append(("group", w.title()))
        elif isinstance(w, QtWidgets.QComboBox):
            own.append(("combo", w.currentText()))
        elif isinstance(w, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)) and w.isReadOnly():
            own.append(("text", w.toPlainText()))
        for kind, text in own:
            if hit(text):
                out.append(Match(w, kind, plain(text)))
        if isinstance(w, QtWidgets.QTabBar):
            for i in range(w.count()):
                if hit(w.tabText(i)):
                    out.append(Match(w, "tab", plain(w.tabText(i)), section=i))
                if hit(w.tabToolTip(i)):
                    out.append(Match(w, "tab", plain(w.tabToolTip(i)), in_tooltip=True, section=i))
        if isinstance(w, QtWidgets.QAbstractItemView):
            for orient, section, text, tip in _header_texts(w):
                if hit(text):
                    out.append(Match(w, "header", plain(text), in_tooltip=tip, section=section,
                                     orientation=orient))
            for idx, text, tip in _model_texts(w):
                if hit(text):
                    out.append(Match(w, "cell", plain(text), in_tooltip=tip,
                                     index=QtCore.QPersistentModelIndex(idx)))
        tip_text = w.toolTip()
        if hit(tip_text):
            out.append(Match(w, "tooltip", plain(tip_text), in_tooltip=True))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Going to a match
# ---------------------------------------------------------------------------------------------------------------
def open_pages(w: QtWidgets.QWidget) -> None:
    """Make current every tab / stacked page between ``w`` and its window."""
    node: Optional[QtWidgets.QWidget] = w
    while node is not None and not node.isWindow():
        parent = node.parentWidget()
        if isinstance(parent, QtWidgets.QStackedWidget) and parent.currentWidget() is not node:
            holder = parent.parentWidget()
            if isinstance(holder, QtWidgets.QTabWidget) and holder.indexOf(node) >= 0:
                holder.setCurrentWidget(node)
            else:
                parent.setCurrentWidget(node)
        node = parent


def target_rect(m: Match) -> Tuple[QtWidgets.QWidget, QtCore.QRect]:
    """(widget, rectangle in its coordinates) to scroll to and outline."""
    w = m.widget
    if m.kind == "tab" and isinstance(w, QtWidgets.QTabBar):
        return w, w.tabRect(m.section)
    if m.kind == "cell" and isinstance(w, QtWidgets.QAbstractItemView) and m.index is not None:
        idx = QtCore.QModelIndex(m.index)
        vp = w.viewport()
        return vp, w.visualRect(idx)
    if m.kind == "header" and isinstance(w, (QtWidgets.QTableView, QtWidgets.QTreeView)):
        header = (w.verticalHeader() if (m.orientation == 2 and isinstance(w, QtWidgets.QTableView))
                  else w.horizontalHeader() if isinstance(w, QtWidgets.QTableView) else w.header())
        pos = header.sectionViewportPosition(m.section)
        size = header.sectionSize(m.section)
        if header.orientation() == QtCore.Qt.Orientation.Horizontal:
            return header.viewport(), QtCore.QRect(pos, 0, size, header.height())
        return header.viewport(), QtCore.QRect(0, pos, header.width(), size)
    return w, w.rect()


def reveal(m: Match) -> None:
    """Open the pages, select a matched tab, scroll a table to its cell and every scroll area to the widget."""
    w = m.widget
    open_pages(w)
    if m.kind == "tab" and isinstance(w, QtWidgets.QTabBar):
        w.setCurrentIndex(m.section)
    if m.kind == "cell" and isinstance(w, QtWidgets.QAbstractItemView) and m.index is not None:
        idx = QtCore.QModelIndex(m.index)
        if isinstance(w, QtWidgets.QTreeView):
            p = idx.parent()
            while p.isValid():
                w.expand(p)
                p = p.parent()
        w.scrollTo(idx)
    if m.kind == "header" and isinstance(w, QtWidgets.QAbstractItemView) and w.model() is not None:
        model = w.model()
        if m.orientation == 1 and model.rowCount() > 0:
            w.scrollTo(model.index(0, m.section))
        elif m.orientation == 2 and model.columnCount() > 0:
            w.scrollTo(model.index(m.section, 0))
    # every scroll area around it, innermost first
    node = w.parentWidget()
    while node is not None and not node.isWindow():
        if isinstance(node, QtWidgets.QScrollArea) and node.widget() is not None \
                and node.widget().isAncestorOf(w):
            target, rect = target_rect(m)
            top_left = target.mapTo(node.widget(), rect.topLeft())
            node.ensureVisible(top_left.x() + rect.width() // 2, top_left.y() + rect.height() // 2,
                               rect.width() // 2 + 8, rect.height() // 2 + 8)
        node = node.parentWidget()


class _Flash(QtWidgets.QWidget):
    """An outline drawn over a match for a moment; it lets every click through."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.setObjectName("ui_search_flash")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def flash(self, target: QtWidgets.QWidget, rect: QtCore.QRect) -> None:
        window = self.parentWidget()
        if window is None:
            return
        top_left = target.mapTo(window, rect.topLeft())
        shown = QtCore.QRect(top_left, rect.size()).adjusted(-3, -3, 3, 3)
        self.setGeometry(shown.intersected(window.rect()))
        self.raise_()
        self.show()
        self._timer.start(FLASH_MS)

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt's name
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        colour = QtGui.QColor(FLASH_COLOUR)
        fill = QtGui.QColor(colour)
        fill.setAlpha(50)
        p.setBrush(fill)
        p.setPen(QtGui.QPen(colour, 3))
        p.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5), 4, 4)


# ---------------------------------------------------------------------------------------------------------------
# The bar
# ---------------------------------------------------------------------------------------------------------------
class _Field(QtWidgets.QLineEdit):
    """The search field: Enter / Shift+Enter move, Esc closes the bar and is not passed on (a dialog stays)."""

    step = QtCore.pyqtSignal(int)
    escape = QtCore.pyqtSignal()

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            back = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
            self.step.emit(-1 if back else 1)
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_Escape:
            self.escape.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class SearchBar(QtWidgets.QFrame):
    """The bar at the top-right corner of a window (see the module's note)."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.window_ = window
        self.setObjectName(BAR_NAME)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"QFrame#{BAR_NAME} {{ border: 1px solid palette(mid); border-radius: 4px; }}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 4, 3)
        lay.setSpacing(3)
        self.field = _Field()
        self.field.setObjectName("ui_search_field")
        self.field.setPlaceholderText("Find in this window")
        self.field.setClearButtonEnabled(True)
        self.field.setMinimumWidth(190)
        self.field.setToolTip("A word to find among this window's labels, buttons, check boxes, group and tab\n"
                              "titles, table headers and cells, layer panels and tooltips. Upper and lower case\n"
                              "and accents are alike. Enter: next match; Shift+Enter: previous; Esc: close.")
        self.btn_prev = QtWidgets.QToolButton()
        self.btn_prev.setObjectName("ui_search_prev")
        self.btn_prev.setArrowType(QtCore.Qt.ArrowType.UpArrow)
        self.btn_prev.setAutoRaise(True)
        self.btn_prev.setToolTip("Previous match (Shift+F3 or Shift+Enter).")
        self.btn_next = QtWidgets.QToolButton()
        self.btn_next.setObjectName("ui_search_next")
        self.btn_next.setArrowType(QtCore.Qt.ArrowType.DownArrow)
        self.btn_next.setAutoRaise(True)
        self.btn_next.setToolTip("Next match (F3 or Enter).")
        self.lbl_count = QtWidgets.QLabel("")
        self.lbl_count.setObjectName("ui_search_count")
        self.lbl_count.setMinimumWidth(70)
        self.btn_close = QtWidgets.QToolButton()
        self.btn_close.setObjectName("ui_search_close")
        self.btn_close.setText("✕")
        self.btn_close.setAutoRaise(True)
        self.btn_close.setToolTip("Close the search (Esc).")
        for wdg in (self.field, self.btn_prev, self.btn_next, self.lbl_count, self.btn_close):
            lay.addWidget(wdg)
        self.matches: List[Match] = []
        self.current = -1
        self._query = ""
        self._flash = _Flash(window)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.search)
        self.field.textChanged.connect(lambda _t: self._debounce.start())
        self.field.step.connect(self.step)
        self.field.escape.connect(self.close_bar)
        self.btn_next.clicked.connect(lambda: self.step(1))
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_close.clicked.connect(self.close_bar)
        context = QtCore.Qt.ShortcutContext.WindowShortcut
        self._f3 = QtWidgets.QShortcut(QtGui.QKeySequence("F3"), window, context=context)
        self._f3.activated.connect(lambda: self._from_key(1))
        self._shift_f3 = QtWidgets.QShortcut(QtGui.QKeySequence("Shift+F3"), window, context=context)
        self._shift_f3.activated.connect(lambda: self._from_key(-1))
        self.hide()

    # -- open / close ---------------------------------------------------------------------------------------
    def open_bar(self) -> None:
        self.adjustSize()
        self._place()
        self.show()
        self.raise_()
        self.window_.installEventFilter(self)
        self.field.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        self.field.selectAll()

    def close_bar(self) -> None:
        self.window_.removeEventFilter(self)
        self._flash.hide()
        QtWidgets.QToolTip.hideText()
        self.hide()

    def _place(self) -> None:
        win = self.window_
        top = 6
        if isinstance(win, QtWidgets.QMainWindow):
            mb = win.menuWidget()
            if mb is not None and mb.isVisible():
                top += mb.height()
        width = min(self.sizeHint().width(), max(120, win.width() - 12))
        self.setGeometry(max(0, win.width() - width - 12), top, width, self.sizeHint().height())

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 - Qt's name
        if obj is self.window_ and event.type() == QtCore.QEvent.Type.Resize:
            self._place()
        return False

    # -- search ---------------------------------------------------------------------------------------------
    def search(self) -> None:
        """Find the field's words again and go to the first match."""
        self._debounce.stop()
        self._query = self.field.text()
        self.matches = find_matches(self.window_, self._query, exclude=self)
        self.current = -1
        if self.matches:
            self.go(0)
        else:
            self._flash.hide()
            self._show_count()

    def step(self, delta: int) -> None:
        """The next (1) or previous (-1) match, round the end; a new or stale search is made first."""
        if self._debounce.isActive() or self.field.text() != self._query \
                or any(not m.alive() for m in self.matches):
            keep = self.current
            self.search()
            if keep < 0 or not self.matches:
                return
            self.current = keep
        if not self.matches:
            self._show_count()
            return
        self.go((self.current + delta) % len(self.matches))

    def _from_key(self, delta: int) -> None:
        if self.isHidden():
            self.open_bar()
            if not self.field.text():
                return
        self.step(delta)

    def go(self, i: int) -> None:
        """Reveal, outline and count match ``i``; when it is in a tooltip, show the tooltip beside it."""
        self.current = int(i)
        m = self.matches[self.current]
        reveal(m)
        self._show_count()

        def outline() -> None:
            if not m.alive() or self.isHidden():
                return
            target, rect = target_rect(m)
            self._flash.flash(target, rect)
            self.raise_()
            if m.in_tooltip:
                pos = target.mapToGlobal(rect.bottomLeft())
                QtWidgets.QToolTip.showText(pos, m.text, target)
            else:
                QtWidgets.QToolTip.hideText()
        # after the layout has settled on the opened page / scrolled area (as tools.mps_window_scroll.reveal)
        QtCore.QTimer.singleShot(0, lambda: QtCore.QTimer.singleShot(0, outline))

    def _show_count(self) -> None:
        if not self.field.text().strip():
            self.lbl_count.setText("")
            self.lbl_count.setToolTip("")
            return
        if not self.matches:
            self.lbl_count.setText("No matches")
            self.lbl_count.setToolTip("Nothing shown in this window holds these words (hidden notes and closed\n"
                                      "'More' lines are not searched).")
            return
        m = self.matches[self.current]
        where = " (tooltip)" if m.in_tooltip else ""
        self.lbl_count.setText(f"{self.current + 1} of {len(self.matches)}{where}")
        self.lbl_count.setToolTip(("Found in a tooltip: " if m.in_tooltip else f"Found in a {m.kind}: ")
                                  + (m.text if len(m.text) <= 300 else m.text[:297] + "..."))


class _Controller(QtCore.QObject):
    """Owns a window's Ctrl+F shortcut; makes its bar the first time it is used."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window)
        self.setObjectName(CONTROLLER_NAME)
        self.window_ = window
        self.bar: Optional[SearchBar] = None
        self.shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Find), window,
                                            context=QtCore.Qt.ShortcutContext.WindowShortcut)
        self.shortcut.activated.connect(self._on_key)

    def _on_key(self) -> None:
        self.open()

    def open(self) -> SearchBar:
        if self.bar is None or sip.isdeleted(self.bar):
            self.bar = SearchBar(self.window_)
        self.bar.open_bar()
        return self.bar


def controller(window: QtWidgets.QWidget) -> Optional[_Controller]:
    """The search installed in ``window``, or None."""
    found = window.findChild(QtCore.QObject, CONTROLLER_NAME, QtCore.Qt.FindChildOption.FindDirectChildrenOnly)
    return found if isinstance(found, _Controller) else None


def install(window: QtWidgets.QWidget) -> _Controller:
    """Give ``window`` Ctrl+F (once: a second call returns the same search)."""
    existing = controller(window)
    if existing is not None:
        return existing
    return _Controller(window)


def open_search(window: QtWidgets.QWidget, text: str = "") -> SearchBar:
    """Open the bar of ``window`` (installing the search if needed), with ``text`` searched at once (tests,
    scripts)."""
    bar = install(window).open()
    if text:
        bar.field.setText(text)
        bar.search()
    return bar


def _searchable(w: Any) -> bool:
    if not isinstance(w, QtWidgets.QWidget) or not w.isWindow():
        return False
    kind = w.windowType()
    return kind in (QtCore.Qt.WindowType.Window, QtCore.Qt.WindowType.Dialog, QtCore.Qt.WindowType.Sheet)


def install_everywhere(app: Optional[QtWidgets.QApplication] = None) -> None:
    """Install the search in every window the program shows, when it first becomes the active window (menus,
    popups and tooltips are left alone). Call once, after the QApplication exists."""
    qapp: Any = app or QtWidgets.QApplication.instance()
    if qapp is None or qapp.property("ui_search_everywhere"):
        return
    qapp.setProperty("ui_search_everywhere", True)

    def _install_in(w: Any) -> None:
        try:
            if _searchable(w):
                install(w)
        except RuntimeError:                           # a window being destroyed
            return

    def on_active(qwindow: Any) -> None:
        # The window that became active, found by its QWindow: ``activeWindow()`` can still answer the previous
        # window when this signal fires (manual test of 2026-11-17: Ctrl+F did nothing in the lumen review and in
        # "Two rings" until another window had been active).
        found = None
        if qwindow is not None:
            for top in qapp.topLevelWidgets():
                try:
                    if top.windowHandle() is qwindow:
                        found = top
                        break
                except RuntimeError:
                    continue
        _install_in(found if found is not None else QtWidgets.QApplication.activeWindow())

    def on_focus(_old: Any, new: Any) -> None:
        # A widget took the keyboard focus: the window that holds it has Ctrl+F before the next key is pressed.
        if isinstance(new, QtWidgets.QWidget):
            _install_in(new.window())

    qapp.focusWindowChanged.connect(on_active)
    qapp.focusChanged.connect(on_focus)
