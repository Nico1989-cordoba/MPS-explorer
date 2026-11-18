"""F6 (the user's request of 2026-10-22): every plot exports a PNG for publication -- the axon map with any rows and
methods ticked, with a legend of what is drawn and an editable title, never over an existing file; the methods'
diagnostic plots; the comparison dialog's table.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_figure_export_f6.py -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import figure_export as FE  # noqa: E402

_APP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _settle(widget=None):
    from PyQt5 import QtCore
    app = _app()
    if widget is not None:
        widget.close()
        widget.deleteLater()
    for _ in range(20):
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.fixture()
def layout_dir(tmp_path, monkeypatch):
    from tools import mps_settings
    original = mps_settings.settings_path
    monkeypatch.setattr(mps_settings, "settings_path",
                        lambda directory=None: original(directory or str(tmp_path)))
    return tmp_path


@pytest.fixture()
def quiet_boxes(monkeypatch):
    from PyQt5 import QtWidgets
    seen = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: seen.append(("information",) + a[1:3]))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: seen.append(("critical",) + a[1:3]))
    return seen


def test_free_path_never_overwrites(tmp_path):
    p = str(tmp_path / "map.png")
    assert FE.free_path(p) == p
    open(p, "wb").close()
    assert FE.free_path(p) == str(tmp_path / "map_2.png")
    open(str(tmp_path / "map_2.png"), "wb").close()
    assert FE.free_path(p) == str(tmp_path / "map_3.png")


def test_the_dialog_proposes_a_free_name_a_title_and_a_legend(tmp_path):
    _app()
    from tools.figure_export_ui import ExportFigureDialog
    existing = tmp_path / "axon_axon_map.png"
    existing.write_bytes(b"x")
    dlg = ExportFigureDialog(["Axon map", "Cluster area"], str(existing),
                             titles={"Axon map": "MPS analysis | methods", "Cluster area": "Cluster area"},
                             legend={"Axon map"})
    try:
        r = dlg.request()
        assert r.path == str(tmp_path / "axon_axon_map_2.png"), r.path      # the proposal never overwrites
        assert r.title == "MPS analysis | methods" and r.legend
        assert not dlg.chk_legend.isHidden()
        dlg.combo_plot.setCurrentText("Cluster area")
        assert dlg.request().title == "Cluster area" and not dlg.request().legend and dlg.chk_legend.isHidden()
        dlg.edit_title.setText("Figure 2B")
        dlg._on_title_typed("Figure 2B")
        dlg.combo_plot.setCurrentText("Axon map")
        assert dlg.request().title == "Figure 2B"                          # a typed title is kept
        # a typed path that exists becomes _2 too; one confirmed through Browse... is kept
        dlg.edit_path.setText(str(existing))
        assert dlg.request().path.endswith("_2.png") and "nothing is overwritten" in dlg.label_size.text()
        dlg._confirmed = os.path.normcase(os.path.abspath(str(existing)))
        assert dlg.request().path == str(existing)
    finally:
        _settle(dlg)
    # without titles (a caller of before F6): the plot's own title, no legend
    dlg2 = ExportFigureDialog(["Axon map"], str(tmp_path / "new.png"))
    try:
        r2 = dlg2.request()
        assert r2.title is None and r2.legend is False and r2.path == str(tmp_path / "new.png")
    finally:
        _settle(dlg2)


def _png_pixels(path):
    from PyQt5 import QtGui
    img = QtGui.QImage(path)
    assert not img.isNull()
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    ptr = img.constBits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :3]
    return img.width(), img.height(), arr[:, :, ::-1].copy()           # BGRA -> RGB


def test_the_map_with_two_methods_exports_a_png_with_legend_and_title(tmp_path, layout_dir, quiet_boxes, monkeypatch):
    _app()
    from test_method_layers import _runs
    from tools import figure_export_ui
    from tools.mps_plot_style import role
    from tools.mps_results_window import PLOT_MAP, MPSResultsWindow, WindowLinks
    a, runs = _runs(("DBSCAN", "voronoi_factor"))
    w = MPSResultsWindow(a, links=WindowLinks())
    w.resize(1366, 700)
    w.show()
    _settle()
    try:
        assert w.show_methods(runs)
        keys = [r.key for r in w._methods.runs]
        # the analysis shown's own outline starts off (manual test of 2026-10-23); ticked here for the legend
        assert not w.axon_map.layers.is_visible(keys[0])
        w.axon_map.layers.set_visible(keys[0], True)
        assert all(w.axon_map.layers.is_visible(k) for k in keys)
        entries = w.legend_entries(PLOT_MAP)
        labels = [label for label, _s in entries]
        assert any(lab.startswith("DBSCAN (shown): solid") for lab in labels), labels
        assert any(lab.startswith("voronoi_factor: dotted") for lab in labels), labels
        title_before = w.axon_map.plot.getPlotItem().titleLabel.text
        path = str(tmp_path / "map_two_methods.png")
        asked = {}

        def fake(plots, suggested, parent=None, offer_white=True, titles=None, legend=False):
            asked.update(titles=titles, legend=legend)
            return FE.FigureRequest(plot=PLOT_MAP, path=path, width_mm=85.0, dpi=300, white=True,
                                    title="DBSCAN and voronoi_factor | simulated ring", legend=True)

        monkeypatch.setattr(figure_export_ui, "ask", fake)
        w._on_export_image()
        assert quiet_boxes and quiet_boxes[-1][1] == "Image written", quiet_boxes
        assert PLOT_MAP in asked["legend"] and "methods' cluster outlines" in asked["titles"][PLOT_MAP]
        width, height, rgb = _png_pixels(path)
        assert width == FE.pixels_for(85.0, 300) and height > 100
        if os.environ.get("F6_SHOT"):
            import shutil
            shutil.copy(path, os.environ["F6_SHOT"])
        assert rgb.std() > 10, "not an empty image"
        # on white, with both methods' colours somewhere in it (outlines and their legend samples)
        assert (rgb.reshape(-1, 3) > 245).all(axis=1).mean() > 0.3

        def near(hexcolour, tol=40):
            c = np.array([int(hexcolour[i:i + 2], 16) for i in (1, 3, 5)])
            return int((np.abs(rgb.reshape(-1, 3).astype(int) - c).sum(axis=1) < tol).sum())

        # the shown method's thin solid line is antialiased on white: a wider tolerance (with the taller map of
        # 2026-10-23 its outlines on the ring sit under the contour and the legend box)
        assert near(role("locs"), tol=80) > 5 and near(role("method_b")) > 5, (near(role("locs"), tol=80),
                                                                                near(role("method_b")))
        # the window shows what it showed: its own title, no legend left in the plot, on black again
        assert w.axon_map.plot.getPlotItem().titleLabel.text == title_before
        from pyqtgraph import LegendItem
        assert not [it for it in w.axon_map.plot.getPlotItem().vb.childItems() if isinstance(it, LegendItem)]
        assert w.dark
        # a second export to the same name through the dialog's rule never overwrites
        assert FE.free_path(path).endswith("map_two_methods_2.png")
    finally:
        _settle(w)


def test_every_plot_of_the_window_exports(tmp_path, layout_dir, quiet_boxes, monkeypatch):
    _app()
    from test_method_layers import _runs
    from tools import figure_export_ui
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    a, runs = _runs(("DBSCAN", "ToMATo"))
    w = MPSResultsWindow(a, links=WindowLinks())
    w.resize(1366, 700)
    w.show()
    _settle()
    try:
        w.show_methods(runs)
        for name in w._plots():
            path = str(tmp_path / ("plot_" + name.replace(" ", "_") + ".png"))
            monkeypatch.setattr(figure_export_ui, "ask", lambda *a_, _n=name, _p=path, **k: FE.FigureRequest(
                plot=_n, path=_p, width_mm=85.0, dpi=300, legend=True, title=None))
            w._on_export_image()
            assert quiet_boxes[-1][1] == "Image written", (name, quiet_boxes[-1])
            width, _h, rgb = _png_pixels(path)
            assert width == FE.pixels_for(85.0, 300) and rgb.std() > 5, name
    finally:
        _settle(w)


def test_the_comparison_table_exports_at_print_resolution(tmp_path):
    _app()
    from tools import cluster_method_ui as ui
    dlg = ui.CompareMethodsDialog(lambda *a, **k: None, 1000, True, "DBSCAN", lambda n: {}, verdicts={})
    dlg.resize(900, 500)
    dlg.show()
    _settle()
    try:
        dlg.table.setRowCount(1)
        from PyQt5 import QtWidgets
        dlg.table.setItem(0, 0, QtWidgets.QTableWidgetItem("DBSCAN"))
        p = str(tmp_path / "table.png")
        first = dlg.export_table_image(p)
        second = dlg.export_table_image(p)
        assert first == p and second.endswith("table_2.png")
        width, height, rgb = _png_pixels(first)
        assert width >= 3 * dlg.table.width() and rgb.std() > 5
    finally:
        _settle(dlg)
