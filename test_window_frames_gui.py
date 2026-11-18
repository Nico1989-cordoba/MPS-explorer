"""The main window hands the frame numbers to every analysis (manual test of 2026-10-23: the MPS analysis window
said "this file has no frame numbers: events cannot be linked" for the example axon, whose frames the batch read):
the event layer, its map rows and the events_* columns follow.

The SIMULATED example axon (examples/demo_axon, tools/make_demo_axon.py). Offscreen Qt; the user's settings file
is never touched. Run: py -3 -m pytest test_window_frames_gui.py -q
"""
from __future__ import annotations

import functools
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets  # noqa: E402

_KEEP = []


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _KEEP.append(a)
    return a


def _pump(app, seconds=0.2):
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def main_window(app, tmp_path, monkeypatch):
    from tools import mps_settings
    original = mps_settings.settings_path
    monkeypatch.setattr(mps_settings, "settings_path", lambda directory=None: original(directory or str(tmp_path)))
    import MPS_explorer
    monkeypatch.setattr(MPS_explorer, "load_settings",
                        functools.partial(mps_settings.load_settings, directory=str(tmp_path)))
    monkeypatch.setattr(MPS_explorer, "save_settings",
                        functools.partial(mps_settings.save_settings, directory=str(tmp_path)))
    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QtWidgets.QMessageBox, kind, staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok))
    monkeypatch.chdir(tmp_path)
    mw = MPS_explorer.MPS_explorer()
    mw.resize(1300, 800)
    _KEEP.append(mw)
    return mw


def test_the_window_analysis_carries_the_frames_and_links_events(app, main_window):
    mw = main_window
    assert mw.open_demo_axon()
    _pump(app)
    mw.ui.pushButton_clusterch1.click()
    t0 = time.perf_counter()
    while (mw.mps_analysis is None or mw.mps_window is None) and time.perf_counter() - t0 < 120:
        _pump(app, 0.1)
    a = mw.mps_analysis
    assert a is not None and a.cluster_method == "DBSCAN"
    frames = a.localization_frame_slab
    assert frames is not None, "the analysis was given no frame numbers"
    assert np.asarray(frames).size == np.asarray(a.x_slab).size
    from tools import mps_axon_map_layers as L
    w = mw.mps_window
    groups = {g.key: g for g in L.build_map(w._map_inputs())}
    fus = groups.get("events")
    assert fus is not None and fus.enabled, (None if fus is None else fus.reason)
    assert "no frame numbers" not in (fus.reason or "")
    cols = w.integration_columns() or {}
    filled = [k for k, v in cols.items() if k.startswith("events_") and v not in (None, "")]
    assert filled, {k: v for k, v in cols.items() if k.startswith("events_")}
    w.close()
    mw.close()
