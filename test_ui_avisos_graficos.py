"""Warnings, labels and legends of the Regularity panel (offscreen Qt, synthetic data only).

Run: QT_QPA_PLATFORM=offscreen py -3 -m pytest test_ui_avisos_graficos.py -q
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
import pytest  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

from test_ui_tanda2_gui import _window, observed  # noqa: E402,F401
from tools import arrangement_ui as A  # noqa: E402
from tools import clustering_integration as ci  # noqa: E402
from tools import mps_gap_measures as gm  # noqa: E402
from tools.cluster_quality import PolygonROI, identify_bad_clusters  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _ring_report(half: bool):
    """A ring of 40 small clusters; the ROI hugs the whole ring (guard acts) or cuts half of it."""
    rng = np.random.default_rng(0)
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    xs, ys, lab = [], [], []
    for k, a in enumerate(ang):
        cx, cy = 1000 * np.cos(a), 1000 * np.sin(a)
        xs.append(cx + rng.normal(0, 15, 30))
        ys.append(cy + rng.normal(0, 15, 30))
        lab.append(np.full(30, k))
    x, y, lab = np.concatenate(xs), np.concatenate(ys), np.concatenate(lab)
    r = 1030.0
    if half:      # a rectangle that clips the left part of the ring only
        poly = np.array([[-900, -3000], [3000, -3000], [3000, 3000], [-900, 3000]], float)
        keep = x > -900
        x, y, lab = x[keep], y[keep], lab[keep]
    else:
        t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        poly = np.c_[r * np.cos(t), r * np.sin(t)]
    return identify_bad_clusters(x, y, lab, PolygonROI(vertices=poly), edge_margin_nm=50.0, dbcv_threshold=-2.0)


def test_guard_report_and_cutting_report_differ():
    tight, cut = _ring_report(False), _ring_report(True)
    assert tight.edge_criterion_disabled and not tight.edge_touching
    assert not cut.edge_criterion_disabled and 0 < len(cut.edge_touching)


def test_regularity_line_warns_when_the_edge_criterion_was_switched_off(app, observed, monkeypatch):
    monkeypatch.setattr(observed, "bad_report", _ring_report(False), raising=False)
    w = _window(observed)
    try:
        w._sync_integration()
        arr = w.arrangement
        text = arr.lbl_reading.text()
        assert "500 nm" in text and ("ADVERTENCIA" in text or "WARNING" in text)
        assert "apagado" in text or "switched off" in text
        assert "color" in arr.lbl_reading.styleSheet()
    finally:
        w.close()


def test_regularity_line_warns_when_the_roi_cuts_part_of_the_ring(app, observed, monkeypatch):
    monkeypatch.setattr(observed, "bad_report", _ring_report(True), raising=False)
    w = _window(observed)
    try:
        w._sync_integration()
        text = w.arrangement.lbl_reading.text()
        assert ci.EDGE_WARNING in text and "apagado" not in text and "switched off" not in text
        assert "color" in w.arrangement.lbl_reading.styleSheet()
    finally:
        w.close()


def test_no_warning_without_edge_problem(app, observed, monkeypatch):
    monkeypatch.setattr(ci, "edge_note", lambda _a: "")
    monkeypatch.setattr(ci, "edge_disabled_note", lambda _a: "")
    w = _window(observed)
    try:
        w._sync_integration()
        assert w.arrangement._edge_warning_text(observed) == ""
        assert "500 nm" not in w.arrangement.lbl_reading.text()
    finally:
        w.close()


def test_reading_labels_are_readable_and_cover_every_level():
    for lv in gm.READING_LEVELS:
        assert ci.reading_label(lv, "es") != lv and ci.reading_label(lv, "en") != lv
        assert "_" not in ci.reading_label(lv, "es") and "_" not in ci.reading_label(lv, "en")
    assert ci.reading_label(gm.KIND_NOT_IDENTIFIED, "es") == "más regular que el azar, tipo no identificado"
    assert ci.reading_label("unknown_code") == "unknown_code"


def test_legend_goes_to_the_free_corner(app):
    plot = pg.PlotWidget()
    x = np.linspace(0, 1, 50)
    plot.plot(x, 1 - x)                      # busy top-left and bottom-right; free top-right? no: curve is diagonal
    plot.plot(x, np.where(x > 0.5, 1.0, 0.0))  # fills the top-right: legend must leave it
    off = A.free_legend_corner(plot, 3)
    assert off != (-4, 4)
    empty = pg.PlotWidget()
    assert A.free_legend_corner(empty, 3) == (-4, 4)


def test_closure_pens_are_clearly_different(app):
    pens = [A.ClosureComparisonDialog.closure_pen(w, k) for w in ("N_loc", "K") for k in ("median", "envelope")]
    keys = {(p.style(), p.widthF()) for p in pens}
    assert len(keys) == 4


def test_scale_bar_length_is_round_and_fits():
    import MPS_explorer as M
    cls = next(v for v in vars(M).values() if isinstance(v, type) and hasattr(v, "_scale_bar_length_nm"))
    assert cls._scale_bar_length_nm(10000) == 2000
    assert cls._scale_bar_length_nm(3000) == 500
    assert cls._scale_bar_length_nm(10) == 50
