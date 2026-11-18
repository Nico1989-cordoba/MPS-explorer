"""The axon map: a right-button drag moves the view (manual test of 2026-10-23), a right-button click with no drag
still opens the menu, the wheel still zooms and the left-button drag still pans.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_map_right_drag.py -q
"""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _pump(n: int = 10) -> None:
    for _ in range(n):
        _app().processEvents()


def _map():
    from tools import mps_axon_map_layers as L
    from tools.mps_analysis import analyze_axon
    from tools.mps_axon_map import AxonMap
    rng = np.random.default_rng(1)
    ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    x = np.concatenate([400 * np.cos(a) + rng.normal(0, 9, 25) for a in ang]) + 2000.0
    y = np.concatenate([400 * np.sin(a) + rng.normal(0, 9, 25) for a in ang]) + 2000.0
    z = rng.normal(300, 20, x.size)
    a = analyze_axon(x, y, z, run_randomization=False)
    m = AxonMap(dark=True)
    m.set_inputs(L.MapInputs(analysis=a))
    m.resize(900, 600)
    m.show()
    _pump(20)
    return m


def _send(widget, kind, pos, button, buttons):
    from PyQt5 import QtCore, QtGui, QtWidgets
    ev = QtGui.QMouseEvent(kind, QtCore.QPointF(pos), button, buttons, QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, ev)
    _pump(2)


def _drag(m, button, start, stop, steps: int = 8):
    from PyQt5 import QtCore
    vp = m.plot.viewport()
    E = QtCore.QEvent.Type
    _send(vp, E.MouseButtonPress, start, button, button)
    for k in range(1, steps + 1):
        p = start + (stop - start) * (k / steps)
        _send(vp, E.MouseMove, p, QtCore.Qt.MouseButton.NoButton, button)
    _send(vp, E.MouseButtonRelease, stop, button, QtCore.Qt.MouseButton.NoButton)


def _range(m):
    (x0, x1), (y0, y1) = m.plot.getViewBox().viewRange()
    return np.array([x0, x1, y0, y1])


def _centre(m):
    from PyQt5 import QtCore
    vp = m.plot.viewport()
    return QtCore.QPointF(vp.width() / 2.0, vp.height() / 2.0)


def test_right_drag_moves_the_map_without_zooming():
    _app()
    from PyQt5 import QtCore
    m = _map()
    try:
        before = _range(m)
        c = _centre(m)
        _drag(m, QtCore.Qt.MouseButton.RightButton, c, c + QtCore.QPointF(120, 60))
        after = _range(m)
        assert not np.allclose(before, after), "a right drag did not move the view"
        # a pan, not a zoom: the widths are the same, both axes moved (the data follows the cursor: x left, y up)
        assert np.isclose(after[1] - after[0], before[1] - before[0], rtol=1e-6)
        assert np.isclose(after[3] - after[2], before[3] - before[2], rtol=1e-6)
        assert after[0] < before[0] and after[2] > before[2]
        assert m._user_range, "the map must keep a range moved by hand"
        # the left button pans as before
        mid = _range(m)
        _drag(m, QtCore.Qt.MouseButton.LeftButton, c, c - QtCore.QPointF(80, 0))
        left = _range(m)
        assert left[0] > mid[0] and np.isclose(left[1] - left[0], mid[1] - mid[0], rtol=1e-6)
    finally:
        m.close()
        m.deleteLater()
        _pump()


def test_right_click_without_drag_opens_the_menu_and_the_wheel_zooms():
    _app()
    from PyQt5 import QtCore, QtGui, QtWidgets
    m = _map()
    try:
        vb = m.plot.getViewBox()
        opened = []
        vb.raiseContextMenu = lambda ev: opened.append(True)       # stand-in for the real menu (it would block)
        before = _range(m)
        c = _centre(m)
        _drag(m, QtCore.Qt.MouseButton.RightButton, c, c + QtCore.QPointF(1, 1), steps=1)   # a click: < drag distance
        assert opened, "a right click no longer opens the menu"
        assert np.allclose(_range(m), before), "a right click moved the view"
        # the wheel still zooms
        vp = m.plot.viewport()
        wheel = QtGui.QWheelEvent(c, vp.mapToGlobal(c.toPoint()), QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
                                  QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
                                  QtCore.Qt.ScrollPhase.NoScrollPhase, False)
        QtWidgets.QApplication.sendEvent(vp, wheel)
        _pump(5)
        after = _range(m)
        assert (after[1] - after[0]) < (before[1] - before[0]), "the wheel no longer zooms"
        assert "right button" in m.lbl_mouse.text()
    finally:
        m.close()
        m.deleteLater()
        _pump()
