"""F5 (the user's request of 2026-10-22): every method of 'Compare methods...' on the axon map at once, each a row
to tick and untick, with its own colour and line; the Voronoi cells of a compared Voronoi method; the map's noise row
named after the method; and each method's diagnostic plot where one is cheap.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_method_layers.py -q
"""
from __future__ import annotations

import os
import time
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import method_overlay as MO  # noqa: E402
from tools import mps_axon_map_layers as L  # noqa: E402

_APP = []
_CACHE: dict = {}


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


def _ring(seed: int = 0, n_clusters: int = 24, radius_nm: float = 400.0):
    """A ring of clusters with background, in nm (as test_method_overlay)."""
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
    return np.concatenate(xs) + 2000.0, np.concatenate(ys) + 2000.0, np.concatenate(zs) + 300.0


def _runs(methods=("DBSCAN", "voronoi_factor", "ToMATo")):
    """The analysis shown (DBSCAN) and each method run on its slab, as 'Compare methods...' runs them."""
    from tools.mps_analysis import analyze_axon
    if "base" not in _CACHE:
        x, y, z = _ring()
        _CACHE["base"] = analyze_axon(x, y, z, run_randomization=False)
    a = _CACHE["base"]
    out = {}
    for m in methods:
        if m not in _CACHE:
            _CACHE[m] = analyze_axon(np.asarray(a.x_slab), np.asarray(a.y_slab), np.asarray(a.z_slab),
                                     run_randomization=False,
                                     slab_override=(float(a.slab_zmin_nm), float(a.slab_zmax_nm)),
                                     cluster_method=m, cluster_params={})
        out[m] = _CACHE[m]
    return a, out


# ------------------------------------------------------------------------------------------------ pure
def test_every_method_is_a_layer_with_its_own_key_and_style():
    a, runs = _runs()
    ml = MO.build_method_layers(a, runs)
    assert [r.method for r in ml.runs] == ["DBSCAN", "voronoi_factor", "ToMATo"]
    assert ml.runs[0].analysis is a and ml.runs[0].is_shown            # the shown method is the analysis itself
    assert ml.reference == "DBSCAN"
    keys = [r.key for r in ml.runs]
    assert keys == ["meth0_DBSCAN", "meth1_voronoi_factor", "meth2_ToMATo"] and len(set(keys)) == 3
    styles = [L.style_of(k) for k in keys]
    assert len({(s.role, s.pen) for s in styles}) == 3
    assert ml.runs[0].summary is None
    for r in ml.runs[1:]:
        assert r.summary is not None and r.summary.n_a == a.n_clusters_kept
        assert len(r.hulls) >= 1 and not r.note


def test_reference_is_the_analysis_shown_without_dbscan_and_other_localizations_are_refused():
    a, runs = _runs()
    vf = runs["voronoi_factor"]
    ml = MO.build_method_layers(vf, {"ToMATo": runs["ToMATo"]})
    assert ml.reference == "voronoi_factor"
    odd = types.SimpleNamespace(x_slab=np.asarray(a.x_slab)[:-1], y_slab=np.asarray(a.y_slab)[:-1],
                                labels=np.asarray(a.labels)[:-1], bad_report=None)
    ml2 = MO.build_method_layers(a, {"FOCAL": odd})
    assert "same localizations" in ml2.runs[1].note and ml2.runs[1].summary is None


def test_the_map_group_rows_disabled_with_images_and_segments():
    a, runs = _runs()
    ml = MO.build_method_layers(a, runs)
    groups = L.build_map(L.MapInputs(analysis=a, methods=ml))
    assert groups[0].key == "methods"
    g = groups[0]
    assert [layer.key for layer in g.layers] == [r.key for r in ml.runs]
    assert [layer.count for layer in g.layers] == [len(r.hulls) for r in ml.runs]
    assert "reference: DBSCAN" in g.caption and "ARI" in g.caption
    assert "DBSCAN (shown): solid blue" == g.layers[0].label
    assert all(layer.enabled for layer in g.layers)
    gi = {x.key: x for x in L.build_map(L.MapInputs(analysis=a, methods=ml, colour_by="images"))}["methods"]
    assert all(not layer.enabled for layer in gi.layers)
    gs = {x.key: x for x in L.build_map(L.MapInputs(analysis=a, methods=ml, source="segments"))}
    assert all(not layer.enabled for x in gs.values() for layer in x.layers if L.is_method_key(layer.key))
    title = L.map_title(L.MapInputs(analysis=a, methods=ml), {"meth0_DBSCAN", "meth2_ToMATo"})
    assert "methods' cluster outlines: DBSCAN solid, ToMATo dashed" in title
    # another analysis: the rows belong to the one they were drawn on
    assert "methods" not in {x.key for x in L.build_map(L.MapInputs(analysis=runs["ToMATo"], methods=ml))}


def test_a_compared_voronoi_method_brings_its_cells():
    a, runs = _runs()
    ml = MO.build_method_layers(a, runs)
    groups = {g.key: g for g in L.build_map(L.MapInputs(analysis=a, methods=ml))}
    g = groups["voronoi_voronoi_factor"]
    assert [layer.key for layer in g.layers] == ["vor_in_voronoi_factor", "vor_out_voronoi_factor"]
    assert all(layer.enabled and layer.kind == "voronoi" for layer in g.layers)
    assert L.style_of("vor_in_voronoi_factor") == L.ROW_STYLES["vor_in"]
    assert "voronoi" not in groups                     # the analysis shown is DBSCAN: no cells of its own


def test_the_noise_row_and_caption_name_the_method():
    a, runs = _runs()
    loc = {g.key: g for g in L.build_map(L.MapInputs(analysis=runs["ToMATo"]))}["localizations"]
    noise = next(layer for layer in loc.layers if layer.key == "slab_noise")
    assert noise.label == "ToMATo: in no cluster" and "DBSCAN" not in noise.tip
    assert "DBSCAN eps" not in loc.caption and "ToMATo" in loc.caption
    loc_d = {g.key: g for g in L.build_map(L.MapInputs(analysis=a))}["localizations"]
    assert next(layer for layer in loc_d.layers if layer.key == "slab_noise").label == "DBSCAN noise"
    assert "DBSCAN eps 25 nm" in loc_d.caption


FAST_METHODS = ("DBSCAN", "HDBSCAN", "HDBSCAN (scikit-learn)", "FOCAL", "ToMATo", "GraphCommunity", "voronoi_factor",
                "OPTICS")


def test_every_runnable_method_has_its_drawing_on_the_map():
    """Method by method (the slow voronoi_csr and Emitters, and External labels, are in the report): each run
    is drawn as outlines of its kept clusters."""
    from tools.clustering_strategies import strategy_registry
    reg = strategy_registry()
    methods = [m for m in FAST_METHODS if m in reg and reg[m].available]
    a, runs = _runs(tuple(methods))
    ml = MO.build_method_layers(a, runs)
    drawn = {r.method: len(r.hulls) for r in ml.runs}
    assert set(drawn) == set(methods)
    empty = [m for m, n in drawn.items() if n == 0]
    assert not empty, drawn
    assert len({r.key for r in ml.runs}) == len(methods)


# ------------------------------------------------------------------------------------------------ the window
def test_the_window_shows_ticks_and_clears_the_methods(layout_dir):
    _app()
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    a, runs = _runs()
    w = MPSResultsWindow(a, links=WindowLinks())
    try:
        labels = a.labels.copy()
        assert w.show_methods(runs)
        assert "methods" in w.axon_map.layers.groups()
        keys = [r.key for r in w._methods.runs]
        # the compared methods start ticked; the analysis shown (slot 0) is drawn as its localizations, its
        # outline starts off (manual test of 2026-10-23)
        assert not w.axon_map.layers.is_visible(keys[0]) and w.axon_map.items(keys[0])
        for k in keys[1:]:
            assert w.axon_map.layers.is_visible(k) and w.axon_map.items(k), k
        # untick one, the others stay; tick it back
        w.axon_map.layers.set_visible(keys[1], False)
        assert not w.axon_map.layers.is_visible(keys[1]) and w.axon_map.layers.is_visible(keys[2])
        assert "voronoi_factor" not in w.axon_map.title().split("outlines:")[-1]
        w.axon_map.layers.set_visible(keys[1], True)
        assert "voronoi_factor dotted" in w.axon_map.title()
        np.testing.assert_array_equal(w.analysis.labels, labels)       # display only
        w.clear_methods()
        assert "methods" not in w.axon_map.layers.groups()
        # they belong to their analysis: a new one drops them
        assert w.show_methods(runs)
        w.analysis = runs["ToMATo"]
        w.refresh()
        assert w._methods is None and "methods" not in w.axon_map.layers.groups()
        # nothing but the shown method ran: nothing to add
        w.analysis = a
        w.refresh()
        assert not w.show_methods({"DBSCAN": runs["DBSCAN"]})
    finally:
        _settle(w)


def test_closing_compare_methods_keeps_every_run_on_the_map(layout_dir, monkeypatch):
    """The user's case: run DBSCAN and voronoi_factor in 'Compare methods...', close it, see both."""
    _app()
    from PyQt5 import QtWidgets
    from tools import cluster_method_ui as ui
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    a, runs = _runs(("DBSCAN", "voronoi_factor"))

    class FakeDialog:
        def __init__(self, *args, **kw):
            self.analyses = dict(runs)
            self.overlay_method = None
            self.chosen = []
            self.ran = list(runs)
            self.comparison_result = None                          # the clustering integration's result

        def exec_(self):
            return QtWidgets.QDialog.Rejected                         # 'Close'

        def run_for(self, label):
            return None

    # the clustering integration: the window opens tools.compare_analyses_ui.CompareAnalysesDialog. Imported
    # BEFORE the base class is faked: a first import under the patch made the real class a subclass of
    # FakeDialog for every later test of the session (test_clustering_integration_gui failed after this one).
    from tools import compare_analyses_ui as cau
    monkeypatch.setattr(ui, "CompareMethodsDialog", FakeDialog)
    monkeypatch.setattr(cau, "CompareAnalysesDialog", FakeDialog)
    w = MPSResultsWindow(a, links=WindowLinks())
    try:
        w._on_compare()
        assert w._methods is not None
        assert [r.method for r in w._methods.runs] == ["DBSCAN", "voronoi_factor"]
        assert w.axon_map.layers.is_visible("meth1_voronoi_factor")
        # the analysis shown is drawn as itself (its localizations): its outline starts off (manual test 2026-10-23)
        assert not w.axon_map.layers.is_visible("meth0_DBSCAN")
        assert w.axon_map.layers.is_visible("slab_kept")
        assert w.method_panel.current_method() == "DBSCAN"
    finally:
        _settle(w)


# ------------------------------------------------------------------------------------------------ diagnostics
def test_diagnostics_offered_and_explained():
    from tools import method_diagnostics as D
    a, runs = _runs()
    d, why = D.diagnostic_for("DBSCAN", a)
    assert d is not None and not why and d.kind == "curve" and d.hlines[0][0] == pytest.approx(25.0)
    assert np.all(np.diff(d.y) >= 0) and d.y.size == np.asarray(a.x_slab).size
    d, why = D.diagnostic_for("voronoi_factor", runs["voronoi_factor"])
    assert d is not None and d.kind == "histogram" and d.vlines and d.y.sum() > 0
    d, why = D.diagnostic_for("ToMATo", runs["ToMATo"])
    assert d is not None and d.kind == "points" and d.diagonal is not None and d.x.size >= runs["ToMATo"].n_clusters_raw
    for name in ("OPTICS", "HDBSCAN", "FOCAL", "GraphCommunity", "External labels", "emitters_bagol_like"):
        d, why = D.diagnostic_for(name, a)
        assert d is None and why == D.NOT_OFFERED[name]


def test_diagnostics_dialog_exports_a_png(tmp_path):
    _app()
    from tools import figure_export
    from tools.method_diagnostics import MethodDiagnosticsDialog
    a, runs = _runs()
    dlg = MethodDiagnosticsDialog([("DBSCAN", a), ("voronoi_factor", runs["voronoi_factor"]),
                                   ("ToMATo", runs["ToMATo"]), ("FOCAL", a)])
    dlg.resize(820, 560)
    dlg.show()
    _settle()
    try:
        assert sorted(dlg.plots) == [0, 1, 2]
        dlg.tabs.setCurrentIndex(3)
        assert not dlg.btn_export.isEnabled()                         # FOCAL: why, no plot
        dlg.tabs.setCurrentIndex(0)
        assert dlg.btn_export.isEnabled()
        path = str(tmp_path / "kdist.png")
        req = figure_export.FigureRequest(plot="DBSCAN diagnostic", path=path, width_mm=85.0, dpi=300)
        written = dlg.export_tab(0, req)
        assert os.path.getsize(written) > 1000
    finally:
        _settle(dlg)
