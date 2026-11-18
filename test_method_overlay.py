"""A second clustering method over the first on the axon map (Compare methods -> Overlay on the map): the
correspondence counts are the benchmark's matching rules, the outlines are each analysis' kept clusters, nothing of
either analysis changes, and the overlay follows the analysis it was drawn on.

Synthetic data only. Offscreen Qt. Run: py -3 -m pytest test_method_overlay.py -q
"""
from __future__ import annotations

import os
import time
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import method_overlay as MO  # noqa: E402

_APP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


def _ring(seed: int = 0, n_clusters: int = 24, radius_nm: float = 400.0):
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


def _labels_case():
    """A: five clusters and noise. B merges A0+A1, splits A2 in two, misses A3, keeps A4, adds one on A's noise."""
    a = np.concatenate([np.full(20, 0), np.full(20, 1), np.full(40, 2), np.full(20, 3), np.full(20, 4),
                        np.full(30, -1)])
    b = np.concatenate([np.full(20, 0), np.full(20, 0), np.full(20, 1), np.full(20, 2), np.full(20, -1),
                        np.full(20, 4), np.full(20, 3), np.full(10, -1)])
    return a, b


def test_the_correspondence_counts_each_kind_once():
    a, b = _labels_case()
    c = MO.correspondence(a, b)
    assert (c.n_a, c.n_b) == (5, 5)
    assert c.n_one_to_one == 1          # A4 = B4
    assert c.n_a_split == 1             # A2 in B1 and B2
    assert c.n_b_merged == 1            # B0 holds A0 and A1
    assert c.n_a_unmatched == 1         # A3 in no B cluster
    assert c.n_b_unmatched == 1         # B3 on A's noise
    assert 0.0 < c.ari < 1.0
    words = c.words("DBSCAN", "ToMATo")
    assert "1 of DBSCAN's split by ToMATo" in words and "ARI" in words


def test_identical_labellings_correspond_one_to_one():
    a, _ = _labels_case()
    c = MO.correspondence(a, a.copy())
    assert c.n_one_to_one == c.n_a == c.n_b == 5
    assert c.n_a_split == c.n_b_merged == c.n_a_unmatched == c.n_b_unmatched == 0
    assert c.ari == pytest.approx(1.0)
    empty = MO.correspondence(np.full(10, -1), np.r_[np.zeros(5, int), np.full(5, -1)])
    assert empty.n_a == 0 and empty.n_b == 1 and empty.n_b_unmatched == 1


def test_kept_labels_and_hulls():
    an = types.SimpleNamespace(labels=np.array([0, 0, 0, 1, 1, 1, 2, 2, -1]),
                               bad_report=types.SimpleNamespace(bad_labels={1}))
    kept = MO.kept_labels(an)
    np.testing.assert_array_equal(kept, [0, 0, 0, -1, -1, -1, 2, 2, -1])
    x = np.array([0.0, 10.0, 0.0, 5.0, 6.0, 7.0, 50.0, 60.0, 99.0])
    y = np.array([0.0, 0.0, 10.0, 5.0, 6.0, 9.0, 50.0, 50.0, 99.0])
    hulls, none = MO.cluster_hulls(x, y, kept)
    assert len(hulls) == 1 and none == 1                 # cluster 2 has two points: no hull
    assert np.array_equal(hulls[0][0], hulls[0][-1])     # closed
    hx, hy, connect = MO.joined_outlines(hulls + hulls)
    assert hx.size == 2 * hulls[0].shape[0] and int((connect == 0).sum()) == 2


def _analyses():
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring()
    a = analyze_axon(x, y, z, run_randomization=False)
    b = analyze_axon(np.asarray(a.x_slab), np.asarray(a.y_slab), np.asarray(a.z_slab), run_randomization=False,
                     slab_override=(float(a.slab_zmin_nm), float(a.slab_zmax_nm)), cluster_method="voronoi_factor",
                     cluster_params={})
    return a, b


def test_the_overlay_needs_the_same_localizations_and_changes_nothing():
    a, b = _analyses()
    labels_a, labels_b = a.labels.copy(), b.labels.copy()
    ov = MO.build_overlay(a, b, "voronoi_factor")
    assert ov.ok and ov.method_a == "DBSCAN" and ov.method_b == "voronoi_factor"
    assert ov.summary.n_a == a.n_clusters_kept and ov.summary.n_b == b.n_clusters_kept
    assert len(ov.hulls_a) <= a.n_clusters_kept
    np.testing.assert_array_equal(a.labels, labels_a)
    np.testing.assert_array_equal(b.labels, labels_b)
    other = types.SimpleNamespace(x_slab=np.asarray(b.x_slab)[:-1], y_slab=np.asarray(b.y_slab)[:-1],
                                  labels=np.asarray(b.labels)[:-1])
    bad = MO.build_overlay(a, other, "X")
    assert not bad.ok and "same localizations" in bad.note


def test_the_map_group_follows_the_overlay():
    from tools import mps_axon_map_layers as L
    a, b = _analyses()
    ov = MO.build_overlay(a, b, "voronoi_factor")
    groups = L.build_map(L.MapInputs(analysis=a, overlay=ov))
    assert groups[0].key == "overlay"
    g = groups[0]
    assert [layer.key for layer in g.layers] == ["ov_a", "ov_b"]
    assert g.layers[0].count == len(ov.hulls_a) and g.layers[1].count == len(ov.hulls_b)
    assert "one-to-one" in g.caption and "benchmark" in g.caption
    assert all(layer.enabled for layer in g.layers)
    title = L.map_title(L.MapInputs(analysis=a, overlay=ov), {"ov_a", "ov_b"})
    assert "cluster outlines: DBSCAN solid, voronoi_factor dotted" in title
    # another analysis: no group
    assert "overlay" not in {gg.key for gg in L.build_map(L.MapInputs(analysis=b, overlay=ov))}
    # the images' colouring uses the same purple: off, with the reason
    gi = {gg.key: gg for gg in L.build_map(L.MapInputs(analysis=a, overlay=ov, colour_by="images"))}["overlay"]
    assert all(not layer.enabled and "same purple" in layer.reason for layer in gi.layers)


def test_the_dialog_offers_the_overlay_for_another_method_only():
    app = _app()
    from tools import cluster_method_ui as ui
    a, _b = _analyses()

    def run_fn(method, params):
        from tools.mps_analysis import analyze_axon
        return analyze_axon(np.asarray(a.x_slab), np.asarray(a.y_slab), np.asarray(a.z_slab),
                            run_randomization=False, slab_override=(float(a.slab_zmin_nm), float(a.slab_zmax_nm)),
                            cluster_method=method, cluster_params=params)

    dlg = ui.CompareMethodsDialog(run_fn, 1000, True, "DBSCAN", lambda n: {}, verdicts={})
    for name, cb in dlg.checks.items():
        cb.setChecked(name in ("DBSCAN", "FOCAL"))
    assert not dlg.btn_overlay.isEnabled()
    dlg.start()
    t0 = time.time()
    while dlg.running() and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.01)
    assert not dlg.running() and set(dlg.analyses) == {"DBSCAN", "FOCAL"}
    assert "analysis" not in dlg.rows["FOCAL"]                     # the table keeps plain values only
    row_of = {dlg.table.item(r, 0).text(): r for r in range(dlg.table.rowCount())}
    dlg.table.clearSelection()
    dlg.table.selectRow(row_of["DBSCAN"])
    assert not dlg.btn_overlay.isEnabled()                         # the method of the analysis shown
    assert "method of the analysis shown" in dlg.overlay_candidate()[1]
    dlg.table.clearSelection()
    dlg.table.selectRow(row_of["FOCAL"])
    assert dlg.btn_overlay.isEnabled()
    dlg.btn_overlay.click()
    assert dlg.overlay_method == "FOCAL" and dlg.result() == 1
    assert dlg.chosen == []                                        # the method in use does not change
    _settle(app, dlg)


def _settle(app, widget=None):
    """Join the comparison workers and delete the widget inside this test (nothing of it is left for the
    garbage collector to destroy while a later test runs Qt)."""
    from PyQt5 import QtCore
    from tools import cluster_method_ui as ui
    t0 = time.time()
    while ui.live_workers() and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.01)
    if widget is not None:
        widget.deleteLater()
    for _ in range(20):
        app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_the_window_draws_and_clears_the_overlay():
    _app()
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    a, b = _analyses()
    w = MPSResultsWindow(a, links=WindowLinks())
    labels = a.labels.copy()
    assert w.show_overlay("voronoi_factor", b)
    assert "overlay" in w.axon_map.layers.groups()
    assert w.axon_map.layers.is_visible("ov_b") and w.axon_map.items("ov_b")
    assert "dotted" in w.axon_map.title()
    np.testing.assert_array_equal(w.analysis.labels, labels)       # display only
    assert w.method_panel.current_method() == "DBSCAN"
    w.clear_overlay()
    assert "overlay" not in w.axon_map.layers.groups()
    # an overlay belongs to its analysis: a new one drops it
    assert w.show_overlay("voronoi_factor", b)
    w.analysis = b
    w.refresh()
    assert w._overlay is None and "overlay" not in w.axon_map.layers.groups()
    w.close()
    _settle(_app(), w)
