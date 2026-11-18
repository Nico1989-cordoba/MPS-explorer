"""The axon map's Voronoi layer (voronoi_factor / voronoi_csr): the cells are the fit's own, nothing is clustered
again, the scale puts the threshold on a class edge, and the map builds the cells only when a row is ticked.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_voronoi_layer.py -q
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import voronoi_layer as V  # noqa: E402
from tools.cluster_tessellation import VoronoiDensityStrategy  # noqa: E402

_APP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _settle(widget):
    """Delete the widget inside this test, so the garbage collector never destroys it while a later test runs Qt."""
    from PyQt5 import QtCore
    app = _app()
    widget.deleteLater()
    for _ in range(10):
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _ring(seed: int = 0, n_clusters: int = 24, radius_nm: float = 400.0):
    """A ring of clusters with background, in nm (as test_phase_d_integration._ring_axon), with one duplicate."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False) + rng.normal(0, 0.03, n_clusters)
    xs, ys, zs = [], [], []
    for a in ang:
        n = int(rng.integers(15, 40))
        xs.append(radius_nm * np.cos(a) + rng.normal(0, 9, n))
        ys.append(radius_nm * np.sin(a) + rng.normal(0, 9, n))
        zs.append(rng.normal(0, 20, n))
    nb = 150
    xs.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    ys.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    zs.append(rng.normal(0, 60, nb))
    x = np.concatenate(xs) + 2000.0
    y = np.concatenate(ys) + 2000.0
    z = np.concatenate(zs) + 300.0
    # a coincident copy: it must share its cell (REVIEW_D0 A2)
    return np.append(x, x[3]), np.append(y, y[3]), np.append(z, z[3])


MODES = [("factor", {}), ("csr", {"csr_iterations": 6, "density_rank": 0})]


@pytest.mark.parametrize("mode,kw", MODES)
def test_labels_are_identical_with_and_without_polygons(mode, kw):
    x, y, _ = _ring()
    xy = np.column_stack([x, y])
    a = VoronoiDensityStrategy(threshold_mode=mode, keep_polygons=False, **kw)
    b = VoronoiDensityStrategy(threshold_mode=mode, keep_polygons=True, **kw)
    la, lb = a.fit(xy), b.fit(xy)
    np.testing.assert_array_equal(la, lb)
    assert a.diagnostics is not None and b.diagnostics is not None
    assert a.diagnostics.threshold_density == b.diagnostics.threshold_density
    np.testing.assert_array_equal(a.diagnostics.density, b.diagnostics.density)
    assert a.diagnostics.cell_polygons is None and b.diagnostics.cell_polygons is not None


@pytest.mark.parametrize("mode,kw", MODES)
def test_rebuilt_cells_are_the_fits_own(mode, kw):
    x, y, _ = _ring()
    xy = np.column_stack([x, y])
    plain = VoronoiDensityStrategy(threshold_mode=mode, **kw)
    kept = VoronoiDensityStrategy(threshold_mode=mode, keep_polygons=True, **kw)
    plain.fit(xy)
    kept.fit(xy)
    V._CACHE.clear()
    rebuilt, why = V.cell_polygons(plain.diagnostics, x, y)
    assert why == "" and rebuilt.source == "rebuilt from the fit's window"
    stored, why2 = V.cell_polygons(kept.diagnostics, x, y)
    assert why2 == "" and stored.source == "kept by the fit"
    np.testing.assert_array_equal(rebuilt.first, stored.first)
    assert len(rebuilt.polygons) == np.unique(xy, axis=0).shape[0] == xy.shape[0] - 1
    for p, q in zip(rebuilt.polygons, stored.polygons):
        np.testing.assert_allclose(p, q)


def test_a_window_that_is_not_the_fits_is_refused():
    x, y, _ = _ring()
    st = VoronoiDensityStrategy()
    st.fit(np.column_stack([x, y]))
    diag = dataclasses.replace(st.diagnostics, window=st.diagnostics.window * 1.05)
    V._CACHE.clear()
    cells, why = V.cell_polygons(diag, x, y)
    assert "differ" in why and not cells.polygons
    _cells, why2 = V.cell_polygons(st.diagnostics, x[:-5], y[:-5])
    assert "cells for" in why2


def test_the_threshold_sits_on_a_class_edge():
    x, y, _ = _ring()
    st = VoronoiDensityStrategy()
    labels = st.fit(np.column_stack([x, y]))
    diag = st.diagnostics
    scale = V.density_scale(diag)
    assert scale is not None and scale.threshold_edge == V.N_CLASSES_BELOW
    assert np.isclose(10.0 ** scale.edges[scale.threshold_edge], diag.threshold_density)
    assert np.all(np.diff(scale.edges) > 0)
    cls = V.density_classes(diag.density, scale)
    above = diag.density >= diag.threshold_density
    assert np.all(cls[above] >= V.N_CLASSES_BELOW)
    assert np.all(cls[~above & (diag.density > 0)] < V.N_CLASSES_BELOW)
    # every localization of a cluster is in a cell at or above the threshold (voronoi_factor selects >=)
    assert np.all(cls[labels >= 0] >= V.N_CLASSES_BELOW)


def test_a_threshold_outside_the_range_is_said():
    x, y, _ = _ring()
    st = VoronoiDensityStrategy()
    st.fit(np.column_stack([x, y]))
    diag = dataclasses.replace(st.diagnostics, threshold_density=float("inf"))
    scale = V.density_scale(diag)
    assert scale is not None and scale.threshold_edge is None and scale.note == "no finite threshold"
    diag2 = dataclasses.replace(st.diagnostics, threshold_density=float(st.diagnostics.density.max()) * 10)
    assert "above" in V.density_scale(diag2).note


def test_the_drawing_splits_cells_in_and_out_of_clusters():
    x, y, _ = _ring()
    st = VoronoiDensityStrategy()
    labels = st.fit(np.column_stack([x, y]))
    diag = st.diagnostics
    scale = V.density_scale(diag)
    din, why = V.cell_drawing(diag, x, y, labels, scale, True)
    dout, why2 = V.cell_drawing(diag, x, y, labels, scale, False)
    assert why == why2 == ""
    n_in, n_out = V.cell_counts(diag, x, y, labels)
    assert sum(din.n_cells.values()) == n_in and sum(dout.n_cells.values()) == n_out
    assert n_in + n_out == np.unique(np.column_stack([x, y]), axis=0).shape[0]
    assert min(din.n_cells) >= V.N_CLASSES_BELOW
    for k, (px, py, connect) in din.classes.items():
        assert px.shape == py.shape == connect.shape
        assert int((connect == 0).sum()) == din.n_cells[k]      # one 0 closes each polygon


def test_the_analysis_keeps_the_diagnostics_and_its_numbers():
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring()
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method="voronoi_factor", cluster_params={})
    diag = a.cluster_diagnostics
    assert diag is not None and np.asarray(diag.density).shape == np.asarray(a.x_slab).shape
    # the labels are those of a fit of the same method on the slab: keeping the diagnostics changes nothing
    again = VoronoiDensityStrategy(**{k: v for k, v in a.cluster_params.items() if k != "strategy"})
    np.testing.assert_array_equal(again.fit(np.column_stack([a.x_slab, a.y_slab])), a.labels)
    b = analyze_axon(x, y, z, run_randomization=False)
    assert b.cluster_diagnostics is None


def test_the_map_group_exists_only_for_a_voronoi_analysis():
    from tools import mps_axon_map_layers as L
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring()
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method="voronoi_factor", cluster_params={})
    groups = {g.key: g for g in L.build_map(L.MapInputs(analysis=a))}
    g = groups["voronoi"]
    assert [layer.key for layer in g.layers] == ["vor_in", "vor_out"]
    assert g.enabled and "threshold" in g.caption and "loc/um^2" in g.caption
    assert all(layer.kind == "voronoi" for layer in g.layers)
    assert not L.view_on("mps", "vor_in") and not L.view_on("mps", "vor_out")   # off by default
    title = L.map_title(L.MapInputs(analysis=a), {"vor_in"})
    assert "Voronoi cells in clusters by density" in title
    # an analysis without its diagnostics: the rows say why
    b = dataclasses.replace(a, cluster_diagnostics=None)
    gb = {g.key: g for g in L.build_map(L.MapInputs(analysis=b))}["voronoi"]
    assert not gb.enabled and "re-run" in gb.reason
    d = analyze_axon(x, y, z, run_randomization=False)
    assert "voronoi" not in {g.key for g in L.build_map(L.MapInputs(analysis=d))}


def test_too_many_cells_are_not_drawn(monkeypatch):
    from tools import mps_axon_map_layers as L
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring()
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method="voronoi_factor", cluster_params={})
    monkeypatch.setattr(V, "MAX_DRAWN_CELLS", 50)
    g = {g.key: g for g in L.build_map(L.MapInputs(analysis=a))}["voronoi"]
    assert all(not layer.enabled and "more than the 50" in layer.reason for layer in g.layers)


def test_the_map_builds_the_cells_when_ticked():
    _app()
    from PyQt5 import QtWidgets
    from tools import mps_axon_map_layers as L
    from tools.mps_analysis import analyze_axon
    from tools.mps_axon_map import AxonMap
    x, y, z = _ring()
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method="voronoi_factor", cluster_params={})
    m = AxonMap(dark=True)
    m.set_inputs(L.MapInputs(analysis=a))
    assert "voronoi" in m.layers.groups()
    assert not m.layers.is_visible("vor_in") and m.items("vor_in") == []      # off, nothing built
    assert m.findChild(QtWidgets.QWidget, "voronoi_scale") is not None
    text = m.findChild(QtWidgets.QLabel, "voronoi_scale_text").text()
    assert "threshold" in text
    m.layers.set_visible("vor_in", True)
    items = m.items("vor_in")
    assert items and all(isinstance(i, QtWidgets.QGraphicsPathItem) for i in items)
    scale = V.density_scale(a.cluster_diagnostics)
    din, _ = V.cell_drawing(a.cluster_diagnostics, a.x_slab, a.y_slab, a.labels, scale, True)
    assert len(items) == len(din.classes)               # one item per class, not one per cell
    assert all(i.isVisible() for i in items)
    m.layers.set_visible("vor_in", False)
    assert not any(i.isVisible() for i in items)
    # ticked rows are built at the next redraw directly
    m.layers.set_visible("vor_in", True)
    m.redraw()
    assert m.items("vor_in") and all(i.isVisible() for i in m.items("vor_in"))
    assert "Voronoi cells in clusters" in m.title()
    _settle(m)


def test_the_method_panel_says_where_the_cells_are():
    _app()
    from tools.cluster_method_ui import MethodPanel
    p = MethodPanel()
    p.set_state("voronoi_csr", {}, True)
    assert not p.lbl_hint.isHidden() and "Voronoi cells" in p.lbl_hint.text()
    p.set_state("DBSCAN", {}, True)
    assert p.lbl_hint.isHidden()
    _settle(p)


def test_densities_are_printed_readably():
    assert V.density_words(3190.4) == "3,190"
    assert V.density_words(148.2) == "148"
    assert V.density_words(1.2) == "1.2"
    assert V.density_words(2.5e8) == "2.5e+08"


def test_every_localization_is_drawn_inside_its_cell():
    """Manual test of 2026-10-23: the cells were drawn with no localization in them (the localization rows were
    off), and that read as "each point no longer has its cell". Ticking a cell row with no localization row drawn
    ticks the matching one; the points are over the cells; one polygon per unique localization of the row."""
    _app()
    from tools import mps_axon_map_layers as L
    from tools.mps_analysis import analyze_axon
    from tools.mps_axon_map import AxonMap
    x, y, z = _ring()
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method="voronoi_factor", cluster_params={})
    m = AxonMap(dark=True)
    m.set_inputs(L.MapInputs(analysis=a))
    m.layers.hide_all()
    assert not m.layers.is_visible("slab_kept")
    m.layers.set_visible("vor_in", True)
    assert m.layers.is_visible("slab_kept"), "the points of the cells were left off"
    assert not m.layers.is_visible("slab_noise")
    cells, points = m.items("vor_in"), m.items("slab_kept")
    assert cells and points and all(p.isVisible() for p in points)
    assert min(p.zValue() for p in points) > max(c.zValue() for c in cells)      # points over their cells
    # one polygon per unique localization of the row (the coincident copy shares its cell)
    scale = V.density_scale(a.cluster_diagnostics)
    din, _ = V.cell_drawing(a.cluster_diagnostics, a.x_slab, a.y_slab, a.labels, scale, True)
    n_polys = sum(int(np.sum(np.asarray(c) == 0)) for _x, _y, c in din.classes.values())
    n_in, _n_out = V.cell_counts(a.cluster_diagnostics, a.x_slab, a.y_slab, a.labels)
    assert n_polys == n_in, (n_polys, n_in)
    kept = np.asarray(a.labels) >= 0
    uniq = np.unique(np.round(np.c_[np.asarray(a.x_slab)[kept], np.asarray(a.y_slab)[kept]], 6), axis=0)
    assert abs(n_in - len(uniq)) <= 0.02 * len(uniq) + 1, (n_in, len(uniq))
    # a deliberate choice stays: with a localization row already on, nothing else is ticked
    m.layers.set_visible("vor_in", False)
    m.layers.set_visible("slab_kept", False)
    m.layers.set_visible("slab_curated", True)
    m.layers.set_visible("vor_out", True)
    assert not m.layers.is_visible("slab_noise") and not m.layers.is_visible("slab_kept")
    m.layers.hide_all()
    m.layers.set_visible("vor_out", True)
    assert m.layers.is_visible("slab_noise")
    _settle(m)


def test_the_methods_group_never_hides_the_points_of_the_analysis_shown():
    """Manual test of 2026-10-23: with 'Methods on the map' present, the points of the analysis shown must depend on
    the 'Localizations' group only; the methods' group adds or removes the compared methods' outlines."""
    _app()
    from tools import mps_axon_map_layers as L
    from tools.method_overlay import build_method_layers
    from tools.mps_analysis import analyze_axon
    from tools.mps_axon_map import AxonMap
    x, y, z = _ring()
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method="voronoi_factor", cluster_params={})
    slab = dict(run_randomization=False, slab_override=(float(a.slab_zmin_nm), float(a.slab_zmax_nm)))
    db = analyze_axon(np.asarray(a.x_slab), np.asarray(a.y_slab), np.asarray(a.z_slab), **slab)
    m = AxonMap(dark=True)
    m.set_inputs(L.MapInputs(analysis=a, methods=build_method_layers(a, {"DBSCAN": db, "voronoi_factor": a})))
    assert "methods" in m.layers.groups()
    m.layers.set_visible("vor_in", True)
    methods = m.layers.group_rows("methods")
    assert methods and not set(methods) & set(m.layers.group_rows("localizations"))
    toggle = m.layers._boxes["methods"].toggle
    for _ in range(3):                      # on, off, on: the points never move
        toggle.click()
        assert m.layers.is_visible("slab_kept") and all(i.isVisible() for i in m.items("slab_kept"))
        assert "localizations:" in m.title()
    m.layers.set_group_shown("methods", False)
    assert not any(m.layers.is_visible(k) for k in methods)
    assert m.layers.is_visible("slab_kept") and all(i.isVisible() for i in m.items("slab_kept"))
    _settle(m)


def test_the_points_get_a_rim_over_the_cells():
    """Over green / yellow cells a sky-blue dot of 3 px could not be told (manual test of 2026-10-23)."""
    _app()
    from tools import mps_axon_map as M
    from tools import mps_axon_map_layers as L
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring()
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method="voronoi_factor", cluster_params={})
    m = M.AxonMap(dark=True)
    m.set_inputs(L.MapInputs(analysis=a))
    (pts,) = m.items("slab_kept")
    size0 = pts.opts["size"]
    m.layers.set_visible("vor_in", True)
    assert pts.opts["size"] >= M.POINTS_OVER_CELLS_SIZE > size0
    assert pts.opts["pen"].color().name() == M.DARK_RIM
    m.redraw()                                       # a redraw keeps it while the cells are drawn
    (pts,) = m.items("slab_kept")
    assert pts.opts["size"] >= M.POINTS_OVER_CELLS_SIZE
    m.layers.set_visible("vor_in", False)
    assert pts.opts["size"] == size0
    _settle(m)
