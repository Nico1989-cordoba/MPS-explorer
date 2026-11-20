"""The wave's dialogs use the room a window can give (manual test of 2026-10-23: "the information is very compressed
in a window that could grow"): maximize buttons and a size grip, most of the screen at first, the size the user left
kept in mps_ui_layout.json; the comparison's table and tabs behind a divider; no table capped in height.

Synthetic data only. Offscreen Qt (its screen is 800 x 600). Run: py -3 -m pytest test_window_sizes.py -q
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = []
_KEEP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


@pytest.fixture(autouse=True)
def no_worker_of_an_earlier_test():
    """Workers an earlier test file left running (a dialog closed while running) end first: the event loop these
    tests run must not be the one that destroys them."""
    import time
    from tools import arrangement_ui, cluster_method_ui
    app = _app()
    t0 = time.time()
    while time.time() - t0 < 120 and any(w.isRunning() for w in list(cluster_method_ui._LIVE_WORKERS)
                                         + list(arrangement_ui._LIVE)):
        app.processEvents()
        time.sleep(0.05)
    for w in list(cluster_method_ui._LIVE_WORKERS) + list(arrangement_ui._LIVE):
        w.wait()
    yield


@pytest.fixture
def layout_dir(tmp_path, monkeypatch):
    from tools import mps_settings
    original = mps_settings.settings_path
    monkeypatch.setattr(mps_settings, "settings_path", lambda directory=None: original(directory or str(tmp_path)))
    return tmp_path


def _dialog():
    from tools.compare_analyses_ui import CompareAnalysesDialog
    return CompareAnalysesDialog(lambda m, p: None, 1000, True, "DBSCAN",
                                 lambda name: {}, verdicts={})


def test_compare_methods_opens_large_resizable_and_remembers(layout_dir):
    app = _app()
    from PyQt5 import QtCore
    from tools import mps_ui_layout as U
    from tools.cluster_method_ui import COMPARE_SIZE_KEY
    avail = app.primaryScreen().availableGeometry()
    d = _dialog()
    try:
        assert d.windowFlags() & QtCore.Qt.WindowType.WindowMaximizeButtonHint
        assert d.isSizeGripEnabled()
        assert d.width() == int(avail.width() * U.WINDOW_SCREEN_SHARE)
        assert d.height() == int(avail.height() * U.WINDOW_SCREEN_SHARE)
        # three panes: the choices, the methods' table, the comparison's tabs
        sp = d.results_splitter
        assert sp.count() == 3 and sp.widget(1) is d.table and sp.widget(2) is d.comparison_result_tabs
        assert d.table.horizontalHeader().stretchLastSection()
    finally:
        d.close()           # never shown; closed, not deleted (the event loop here must not run other tests'
        _KEEP.append(d)     # deferred deletions of pyqtgraph items, which aborted the session)


def test_a_window_reopens_at_the_size_the_user_left(layout_dir):
    """open_sized / remember_size on a plain dialog (the comparison's own minimum width depends on the fonts)."""
    app = _app()
    from PyQt5 import QtWidgets
    from tools import mps_ui_layout as U
    d = QtWidgets.QDialog()
    U.open_sized(d, "test.size")
    d.show()
    app.processEvents()
    d.resize(500, 400)
    app.processEvents()
    d.hide()                                   # stored on hide
    app.processEvents()
    with open(U.layout_path(), encoding="utf-8") as fh:
        assert json.load(fh)["test.size"] == [500, 400, 0]
    d.deleteLater()
    d2 = QtWidgets.QDialog()
    U.open_sized(d2, "test.size")
    assert (d2.width(), d2.height()) == (500, 400), "the size the user left is not restored"
    d2.deleteLater()
    app.processEvents()


def test_comparison_tables_fit_their_columns_and_stretch():
    _app()
    from tools import mps_ui_layout as U
    from tools.compare_analyses_ui import _table, fill
    t = _table([], "", "t")
    fill(t, [{"a": "x" * 400, "b": 1.0}], ["a", "b"])
    assert t.horizontalHeader().sectionSize(0) <= U.TABLE_COLUMN_MAX_PX
    assert t.horizontalHeader().stretchLastSection()
    t.deleteLater()


def test_no_table_of_the_arrangement_tab_is_capped_in_height():
    import inspect
    from tools import arrangement_ui
    src = inspect.getsource(arrangement_ui)
    assert "setMaximumHeight" not in src
