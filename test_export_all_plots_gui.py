"""'Export image...' of every plot of the analysis window after a small per-axon null (manual test of 2026-10-23,
i15: H1, H2 and the regularity plots came out with axes and no curve, the title pale grey on white): on white and
on black, each figure has marks inside its data area and a title that stands out from the background.

SYNTHETIC axon (tools.mps_pipeline_null.observe_synthetic), B = 4. Offscreen Qt.
Run: py -3 -m pytest test_export_all_plots_gui.py -q
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir(r"C:\Windows\Fonts"):
    # the offscreen platform draws no text without a font folder: the titles are part of what is tested
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PyQt5 import QtGui, QtWidgets  # noqa: E402

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


def _pixels(path):
    img = QtGui.QImage(path).convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    b = img.bits()
    b.setsize(img.byteCount())
    return np.frombuffer(b, np.uint8).reshape(h, img.bytesPerLine())[:, :w * 3].reshape(h, w, 3).astype(int)


@pytest.fixture(scope="module")
def window_after_null(app, tmp_path_factory):
    from tools import mps_pipeline_null as pn
    from tools import mps_settings
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    d = tmp_path_factory.mktemp("settings")
    original = mps_settings.settings_path
    mps_settings.settings_path = lambda directory=None: original(directory or str(d))
    an, _ax = pn.observe_synthetic(pn.synthetic_model((900.0, 700.0)), 3)
    w = MPSResultsWindow(an, rerun_callback=lambda **kw: an, links=WindowLinks(nena=lambda: (12.0, "synthetic")))
    _KEEP.append(w)
    w.resize(1500, 900)
    w.show()
    _pump(app, 0.5)
    w._sync_integration()
    arr = w.arrangement
    arr.combo_b.setItemData(0, 4)
    assert arr.start()
    t0 = time.perf_counter()
    while arr.running() and time.perf_counter() - t0 < 600:
        _pump(app, 0.2)
    _pump(app, 0.3)
    assert arr.outcomes and all(o.ok for o in arr.outcomes.values())
    w.tabs.setCurrentIndex(0)                    # another tab in front, as after reading the results
    arr.table.clearSelection()                   # and no row selected in the nulls' table
    yield w
    mps_settings.settings_path = original
    w.close()


@pytest.mark.parametrize("white", [True, False])
def test_every_plot_exports_its_marks_and_a_readable_title(app, window_after_null, tmp_path, monkeypatch, white):
    from tools import figure_export as FE
    from tools import figure_export_ui
    w = window_after_null
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, kind, staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok))
    names = list(w._plots())
    assert len(names) == 11, names
    bad = []
    for i, name in enumerate(names):
        path = str(tmp_path / f"plot_{i}_{'white' if white else 'dark'}.png")

        def fake(plots, suggested, parent=None, offer_white=True, titles=None, legend=False, _n=name, _p=path):
            return FE.FigureRequest(plot=_n, path=_p, width_mm=85.0, dpi=150, white=white,
                                    title=(titles or {}).get(_n, ""), legend=False)

        monkeypatch.setattr(figure_export_ui, "ask", fake)
        w._on_export_image()
        _pump(app, 0.1)
        a = _pixels(path)
        h, wd = a.shape[:2]
        bg = np.array([255, 255, 255] if white else [0, 0, 0])
        data = a[int(h * 0.2):int(h * 0.8), int(wd * 0.2):int(wd * 0.95)]
        marks = float((np.abs(data - bg).sum(-1) > 90).mean())
        title = a[: max(4, int(h * 0.12)), :]
        strong = int((np.abs(title - bg).sum(-1) > 450).sum())        # nearly the opposite of the background (a pale grey title on white gives none)
        has_title = bool(FE.plain_title(w._plots()[name].getPlotItem()))
        if marks < 0.003 or (has_title and strong < 1):
            bad.append((name, round(marks, 4), strong))
    assert not bad, " | ".join(map(str, bad))
