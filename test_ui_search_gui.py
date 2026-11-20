"""F3 (the user's request of 2026-10-22): Ctrl+F finds a word in every window of MPS Explorer.

A synthetic window (tabs, a scroll area, a table, a hidden note, tooltips) checks what is searched and how a match is
reached; then the real windows (MPS analysis, batch, method comparison) check that the search is installed and finds
their texts. Offscreen Qt, no data.

Run: py -3 -m pytest test_ui_search_gui.py -q
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402

from tools import ui_search  # noqa: E402

_APP = []


def _app():
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _pump(seconds: float = 0.15) -> None:
    app = _app()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def _settle(widget) -> None:
    app = _app()
    widget.close()
    widget.deleteLater()
    for _ in range(10):
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.fixture()
def layout_dir(tmp_path, monkeypatch):
    from tools import mps_settings
    original = mps_settings.settings_path
    monkeypatch.setattr(mps_settings, "settings_path",
                        lambda directory=None: original(directory or str(tmp_path)))
    return tmp_path


def _synthetic_window(dialog: bool = False):
    """Two tabs; the second holds a tall scroll area whose last label says 'Mahalanobis threshold'."""
    _app()
    win = QtWidgets.QDialog() if dialog else QtWidgets.QMainWindow()
    tabs = QtWidgets.QTabWidget()
    first = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(first)
    group = QtWidgets.QGroupBox("Occupancy of the perimeter")
    gl = QtWidgets.QVBoxLayout(group)
    btn = QtWidgets.QPushButton("Run")
    btn.setToolTip("Runs the randomization with a hidden keyword: zanahoria.")
    gl.addWidget(btn)
    hidden = QtWidgets.QLabel("Mahalanobis in a hidden note")
    hidden.setVisible(False)
    gl.addWidget(hidden)
    lay.addWidget(group)
    table = QtWidgets.QTableWidget(40, 2)
    table.setHorizontalHeaderLabels(["Perímetro (µm)", "Note"])
    for r in range(40):
        table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{r}.0"))
        item = QtWidgets.QTableWidgetItem("árbol" if r == 35 else "-")
        if r == 3:
            item.setToolTip("ocupación")
        table.setItem(r, 1, item)
    lay.addWidget(table)
    second = QtWidgets.QWidget()
    sl = QtWidgets.QVBoxLayout(second)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    content = QtWidgets.QWidget()
    cl = QtWidgets.QVBoxLayout(content)
    for i in range(60):
        cl.addWidget(QtWidgets.QLabel(f"filler line {i}"))
    target = QtWidgets.QLabel("<b>Mahalanobis</b> threshold")
    cl.addWidget(target)
    scroll.setWidget(content)
    sl.addWidget(scroll)
    tabs.addTab(first, "Results")
    tabs.addTab(second, "Settings")
    tabs.setTabToolTip(1, "Where the parameters are")
    if dialog:
        QtWidgets.QVBoxLayout(win).addWidget(tabs)
    else:
        win.setCentralWidget(tabs)
    win.resize(600, 400)
    ui_search.install(win)
    win.show()
    win.activateWindow()
    _pump()
    return win, dict(tabs=tabs, scroll=scroll, target=target, table=table, hidden=hidden, button=btn)


def test_normalise_ignores_case_accents_tags_and_mnemonics():
    assert ui_search.normalise("Perímetro") == ui_search.normalise("PERIMETRO") == "perimetro"
    assert ui_search.normalise("<b>Ocupación</b>&nbsp;total") == "ocupacion total"
    assert ui_search.normalise("&Cancel") == "cancel"
    assert ui_search.normalise("A && B") == "a & b"


def test_nothing_is_built_before_ctrl_f():
    win, _w = _synthetic_window()
    try:
        ctl = ui_search.controller(win)
        assert ctl is not None and ctl.bar is None
        assert win.findChild(QtWidgets.QFrame, ui_search.BAR_NAME) is None
        assert ui_search.install(win) is ctl                        # a second install is the same search
    finally:
        _settle(win)


def test_ctrl_f_opens_the_bar_and_finds_on_another_tab_inside_a_scroll_area():
    win, w = _synthetic_window()
    try:
        QTest.keyClick(win, QtCore.Qt.Key.Key_F, QtCore.Qt.KeyboardModifier.ControlModifier)
        _pump()
        bar = ui_search.controller(win).bar
        assert bar is not None and bar.isVisible() and bar.field.hasFocus()
        # top-right corner, inside the window
        assert bar.geometry().right() <= win.width() and bar.geometry().top() >= 0
        QTest.keyClicks(bar.field, "mahalanobis")
        _pump(0.4)                                                     # the debounce, then the first match
        # the hidden note is not a match; the label on the second tab is (its tags do not hide the word)
        assert len(bar.matches) == 1 and bar.matches[0].widget is w["target"]
        assert w["tabs"].currentIndex() == 1
        assert w["scroll"].verticalScrollBar().value() > 0
        vp = w["scroll"].viewport()
        top = w["target"].mapTo(vp, QtCore.QPoint(0, 0)).y()
        assert 0 <= top and top + w["target"].height() <= vp.height()
        assert bar.lbl_count.text() == "1 of 1"
        flash = win.findChild(QtWidgets.QWidget, "ui_search_flash")
        assert flash is not None and flash.isVisible()
    finally:
        _settle(win)


def test_headers_cells_tabs_groups_and_tooltips():
    win, w = _synthetic_window()
    try:
        bar = ui_search.open_search(win, "PERIM")
        kinds = sorted(m.kind for m in bar.matches)
        assert kinds == ["group", "header"], kinds          # "Perímetro (µm)" and "Occupancy of the perimeter"
        bar = ui_search.open_search(win, "ARBOL")
        assert [m.kind for m in bar.matches] == ["cell"]
        _pump()
        table = w["table"]
        rect = table.visualRect(table.model().index(35, 1))
        assert table.viewport().rect().intersects(rect), "scrolled to row 35"
        bar = ui_search.open_search(win, "zanahoria")
        assert len(bar.matches) == 1 and bar.matches[0].in_tooltip
        assert "(tooltip)" in bar.lbl_count.text()
        bar = ui_search.open_search(win, "ocupacion")                   # a cell's own tooltip
        assert [(m.kind, m.in_tooltip) for m in bar.matches] == [("cell", True)]
        bar = ui_search.open_search(win, "parameters are")              # a tab's tooltip
        assert [(m.kind, m.in_tooltip) for m in bar.matches] == [("tab", True)]
        assert w["tabs"].currentIndex() == 1
        bar = ui_search.open_search(win, "no such words anywhere")
        assert bar.matches == [] and bar.lbl_count.text() == "No matches"
    finally:
        _settle(win)


def test_next_previous_wrap_with_enter_and_f3():
    win, _w = _synthetic_window()
    try:
        bar = ui_search.open_search(win, "filler line 1")              # 1, 10..19: 11 labels
        n = len(bar.matches)
        assert n == 11 and bar.current == 0
        QTest.keyClick(bar.field, QtCore.Qt.Key.Key_Return)
        assert bar.current == 1
        QTest.keyClick(bar.field, QtCore.Qt.Key.Key_Return, QtCore.Qt.KeyboardModifier.ShiftModifier)
        QTest.keyClick(bar.field, QtCore.Qt.Key.Key_Return, QtCore.Qt.KeyboardModifier.ShiftModifier)
        assert bar.current == n - 1 and bar.lbl_count.text() == f"{n} of {n}"
        QTest.keyClick(win, QtCore.Qt.Key.Key_F3)
        assert bar.current == 0
        QTest.keyClick(win, QtCore.Qt.Key.Key_F3, QtCore.Qt.KeyboardModifier.ShiftModifier)
        assert bar.current == n - 1
    finally:
        _settle(win)


def test_escape_closes_the_bar_and_not_the_dialog():
    win, _w = _synthetic_window(dialog=True)
    try:
        bar = ui_search.open_search(win, "run")
        assert bar.isVisible()
        QTest.keyClick(bar.field, QtCore.Qt.Key.Key_Escape)
        _pump()
        assert not bar.isVisible() and win.isVisible()
        assert win.result() == 0
    finally:
        _settle(win)


def test_install_everywhere_reaches_a_window_nobody_installed_it_in():
    app = _app()
    ui_search.install_everywhere(app)
    win = QtWidgets.QMainWindow()
    win.setCentralWidget(QtWidgets.QLabel("A window of its own"))
    assert ui_search.controller(win) is None
    win.show()
    win.activateWindow()
    QtWidgets.QApplication.setActiveWindow(win)
    _pump(0.3)
    try:
        assert ui_search.controller(win) is not None
    finally:
        _settle(win)


# ------------------------------------------------------------------------------------------------ real windows
def test_mps_analysis_window_is_searchable(layout_dir):
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    _app()
    w = MPSResultsWindow(None, links=WindowLinks(), rerun_callback=lambda **k: None)
    w.resize(1366, 700)
    w.show()
    _pump()
    try:
        assert ui_search.controller(w) is not None
        bar = ui_search.open_search(w, "mahalanobis")
        assert bar.matches and any(m.in_tooltip for m in bar.matches) and any(not m.in_tooltip for m in bar.matches)
        bar = ui_search.open_search(w, "nearest neighbours")           # a tab title
        assert any(m.kind == "tab" for m in bar.matches)
        # within the window at 1366 x 700
        assert bar.geometry().right() <= w.width()
    finally:
        _settle(w)


def test_batch_and_comparison_windows_are_searchable():
    _app()
    from tools import cluster_method_ui as cm
    from tools.batch_window import BatchWindow
    from tools.mps_settings import MPSSettings
    b = BatchWindow(MPSSettings())
    b.show()
    _pump()
    try:
        bar = ui_search.open_search(b, "slow methods")
        assert any(m.kind == "button" for m in bar.matches)
    finally:
        _settle(b)
    dlg = cm.CompareMethodsDialog(lambda *a, **k: None, 1000, True, "DBSCAN", lambda n: {}, verdicts={})
    dlg.show()
    _pump()
    try:
        assert ui_search.controller(dlg) is not None
        bar = ui_search.open_search(dlg, "overlay")
        assert bar.matches
    finally:
        _settle(dlg)
