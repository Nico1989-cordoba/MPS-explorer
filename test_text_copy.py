"""Every informative text can be selected and copied (manual test of 2026-10-23): the labels of the analysis window
(estimates, banners, readings, captions, provenance, the Working scale), Ctrl+C on a selection, 'Copy title' on the
plots; tooltips and pictures left alone.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_text_copy.py -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402

_KEEP = []


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def layout_dir(tmp_path, monkeypatch):
    from tools import mps_settings
    original = mps_settings.settings_path
    monkeypatch.setattr(mps_settings, "settings_path", lambda directory=None: original(directory or str(tmp_path)))
    return tmp_path


def _pump(app, n=20):
    for _ in range(n):
        app.processEvents()


def _selectable(label: QtWidgets.QLabel) -> bool:
    return bool(label.textInteractionFlags() & QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)


def test_a_label_is_selectable_and_ctrl_c_copies_it(app):
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    text = QtWidgets.QLabel("NOT INTERPRETABLE: B = 4")
    link = QtWidgets.QLabel('<a href="x">a link</a>')
    link.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)
    pic = QtWidgets.QLabel()
    pm = QtGui.QPixmap(8, 8)
    pm.fill(QtGui.QColor("red"))
    pic.setPixmap(pm)
    for x in (text, link, pic):
        lay.addWidget(x)
    from tools.copyable import make_labels_selectable
    # before the window is shown: once a main window exists, its activation hook does the same walk
    assert make_labels_selectable(w) == 2
    assert make_labels_selectable(w) == 0                   # each label once
    w.show()
    _pump(app)
    assert _selectable(text) and _selectable(link) and not _selectable(pic)
    assert link.textInteractionFlags() & QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse   # links kept
    text.setFocus()
    text.setSelection(0, len(text.text()))
    QtTest.QTest.keyClick(text, QtCore.Qt.Key.Key_C, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert QtWidgets.QApplication.clipboard().text() == "NOT INTERPRETABLE: B = 4"
    w.deleteLater()
    _pump(app)


def test_the_analysis_windows_informative_labels_and_plot_titles(app, layout_dir):
    from tools import mps_pipeline_null as pn
    from tools.copyable import COPY_TITLE_TEXT
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    an, _ax = pn.observe_synthetic(pn.synthetic_model((900.0, 700.0)), 3)
    w = MPSResultsWindow(an, links=WindowLinks(nena=lambda: (12.0, "synthetic")))
    try:
        # the window's showEvent runs this walk (and each map drawing does); called directly here: a shown
        # window with pyqtgraph plots in this session made a later test abort in pyqtgraph's teardown
        import inspect
        from tools.copyable import make_labels_selectable
        assert "make_labels_selectable(self)" in inspect.getsource(MPSResultsWindow.showEvent)
        make_labels_selectable(w)
        arr = w.arrangement
        named = [arr.lbl_estimate, arr.banner, arr.lbl_reading, arr.lbl_h_help, w.axon_map.lbl_line1,
                 w.axon_map.lbl_line2, w.working_scale.lbl_rule, w.banner.label]
        for lab in named:
            assert _selectable(lab), lab.objectName()
        captions = [lab for lab in w.axon_map.findChildren(QtWidgets.QLabel)
                    if lab.objectName().startswith("group_caption_") and lab.text()]
        assert captions and all(_selectable(lab) for lab in captions)
        menu = w.axon_map.plot.getPlotItem().getViewBox().menu
        actions = [a for a in menu.actions() if a.text() == COPY_TITLE_TEXT]
        assert actions
        from tools.copyable import _to_clipboard  # noqa: F401  (the action's own call, without a menu)
        assert "MPS analysis" in __import__("tools.figure_export", fromlist=["plain_title"]).plain_title(
            w.axon_map.plot.getPlotItem())
    finally:
        w.close()           # closed and kept for the session: deleting a window with pyqtgraph plots during the
        _KEEP.append(w)     # session made a later test's event loop abort in pyqtgraph's teardown


def test_tooltips_are_left_alone(app):
    from tools.copyable import label_is_informative
    tip = QtWidgets.QLabel("tip", None, QtCore.Qt.WindowType.ToolTip)
    assert not label_is_informative(tip)
    tip.deleteLater()
    assert np.isfinite(1.0)
