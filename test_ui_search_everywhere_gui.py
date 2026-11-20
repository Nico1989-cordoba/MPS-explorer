"""Ctrl+F in EVERY window (the user's request of 2026-11-17: it did nothing in the lumen review and in "Two rings").

``ui_search.install_everywhere`` installs the search in a window when it becomes active or when a widget in it first
takes the keyboard focus. Windows of every kind the program opens (a main window, a dialog with a parent, a dialog
without one, a window opened from another window) must have the search installed once they show a widget with focus,
whatever the order in which they were opened. Offscreen Qt, no data.

Run: py -3 -m pytest test_ui_search_everywhere_gui.py -q
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402

from tools import ui_search  # noqa: E402

_APP = []


def _app():
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _window(kind: str, parent=None):
    if kind == "main":
        w = QtWidgets.QMainWindow(parent)
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addWidget(QtWidgets.QLabel("Tolerance tau"))
        lay.addWidget(QtWidgets.QPushButton("Apply and re-run analyses"))
        w.setCentralWidget(central)
        return w, w.centralWidget().findChild(QtWidgets.QPushButton)
    d = QtWidgets.QDialog(parent)
    lay = QtWidgets.QVBoxLayout(d)
    lay.addWidget(QtWidgets.QLabel("Rings 0-1 along the membrane"))
    btn = QtWidgets.QPushButton("Export CSV...")
    lay.addWidget(btn)
    return d, btn


def test_every_window_gets_ctrl_f_without_prior_activation():
    app = _app()
    ui_search.install_everywhere(app)
    opened = []
    main, main_btn = _window("main")
    opened.append((main, main_btn))
    main.show()
    dlg_parent, dlg_parent_btn = _window("dialog", parent=main)        # a dialog opened from a window
    opened.append((dlg_parent, dlg_parent_btn))
    dlg_parent.show()
    dlg_free, dlg_free_btn = _window("dialog")                           # a dialog with no parent
    opened.append((dlg_free, dlg_free_btn))
    dlg_free.show()
    sub, sub_btn = _window("main", parent=dlg_free)                      # a window opened from a dialog
    opened.append((sub, sub_btn))
    sub.show()
    for w, btn in opened:
        w.activateWindow()
        btn.setFocus()
        QTest.qWait(50)
        app.processEvents()
    missing = [type(w).__name__ for w, _b in opened if ui_search.controller(w) is None]
    assert not missing, f"no Ctrl+F in: {missing}"
    for w, _b in opened:
        w.close()
        w.deleteLater()
    app.processEvents()


def test_focus_alone_installs_the_search():
    """A widget taking the focus is enough, even if no window-activation signal told which window it is."""
    app = _app()
    ui_search.install_everywhere(app)
    w, btn = _window("dialog")
    w.show()
    btn.setFocus()
    QTest.qWait(50)
    app.processEvents()
    assert ui_search.controller(w) is not None
    bar = ui_search.open_search(w, "export")
    assert bar is not None
    w.close()
    w.deleteLater()
    app.processEvents()
