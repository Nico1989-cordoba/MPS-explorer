"""BaGoL phase F5: the 'Events -> Emitters' layer in the analysis, the panel, the settings, the batch and the
axon table v11 (synthetic data only). Run offscreen: QT_QPA_PLATFORM=offscreen."""

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import emitters_layer as el  # noqa: E402

# Short chains, one process: the layer's wiring is tested here, not the RJMCMC (test_cluster_emitters.py).
FAST = {"n_burnin": 200, "n_trials": 300, "processes": 1, "seed": 3}


def _ring(seed: int = 0, n_clusters: int = 18, radius_nm: float = 400.0):
    """A ring of label stacks (one axon's cross-section) with background, in nm: lpx / lpy and the frame of every
    localization. Each stack blinks in events of 1-3 consecutive frames (a switching cycle), each localization
    scattered by its own precision around the stack."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    xs, ys, zs, fs = [], [], [], []
    for a in ang:
        cx, cy = radius_nm * np.cos(a), radius_nm * np.sin(a)
        for _ in range(int(rng.integers(6, 11))):            # events of this stack
            k = int(rng.integers(1, 4))
            f0 = int(rng.integers(1, 15000))
            xs.append(cx + rng.normal(0, 8, k))
            ys.append(cy + rng.normal(0, 8, k))
            zs.append(rng.normal(0, 20, k))
            fs.append(np.arange(f0, f0 + k))
    nb = 60
    xs.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    ys.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    zs.append(rng.normal(0, 60, nb))
    fs.append(rng.integers(1, 15000, nb))
    x = np.concatenate(xs) + 2000.0
    y = np.concatenate(ys) + 2000.0
    z = np.concatenate(zs) + 300.0
    frames = np.concatenate(fs).astype(np.int64)
    sig = np.column_stack([rng.uniform(6, 10, x.size), rng.uniform(6, 10, x.size)])
    return x, y, z, sig, frames


def _params(**kw):
    p = dict(FAST)
    p.update(kw)
    return p


@pytest.fixture(scope="module")
def emitters_analysis():
    from tools.mps_analysis import analyze_axon
    x, y, z, sig, frames = _ring()
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method=el.EMITTERS_METHOD,
                     cluster_params=_params(xi_source="literature_MEA"), localization_sigma_nm=sig,
                     localization_frame=frames)
    return a, (x, y, z, sig, frames)


# ----------------------------------------------------------------- the layer
def test_xi_sources_set_the_strategy():
    mea = el.LayerChoice.from_params({"xi_source": "literature", "buffer": "MEA", "dye": "AF647"}).strategy_kwargs()
    assert mea == {"xi_kind": "gamma", "xi_mean": 14.0, "xi_shape": 1.0, "hierarchical": False}
    bme = el.LayerChoice.from_params({"xi_source": "literature", "buffer": "bME",
                                      "dye": "Alexa Fluor 568"}).strategy_kwargs()
    assert bme["xi_mean"] == 52.0 and bme["hierarchical"] is False
    # learned starts from the literature value of the buffer chosen (F5c), not always from MEA
    learned = el.LayerChoice.from_params({"xi_source": "learned", "buffer": "MEA", "dye": "AF568", "xi_shape": 2.0})
    kw = learned.strategy_kwargs()
    assert kw["hierarchical"] is True and kw["xi_mean"] == 7.0 and kw["xi_shape"] == 2.0
    assert el.LayerChoice.from_params({"xi_source": "learned", "buffer": "bME", "dye": "AF568"}).prior().mean == 52.0
    cal = el.LayerChoice.from_params({"xi_source": "calibrated", "buffer": "bME",
                                      "xi_calibrated_mean": 9.5}).strategy_kwargs()
    assert cal["xi_mean"] == 9.5 and cal["hierarchical"] is False
    with pytest.raises(ValueError, match="calibrated"):
        el.LayerChoice.from_params({"xi_source": "calibrated", "buffer": "MEA"})
    with pytest.raises(ValueError):
        el.LayerChoice.from_params({"xi_source": "made_up", "buffer": "MEA"})
    with pytest.raises(ValueError):
        el.LayerChoice.from_params({"dye": "Cy5", "buffer": "MEA"})
    with pytest.raises(ValueError):
        el.LayerChoice.from_params({"r_mol_nm": -1, "buffer": "MEA"})
    with pytest.raises(ValueError):
        el.LayerChoice.from_params({"buffer": "Trolox"})
    d = el.LayerChoice()
    assert (d.xi_source, d.dye, d.k, d.r_mol_nm, d.buffer) == ("literature", "Alexa Fluor 647", 1.0, 50.0, None)


def test_the_buffer_has_no_default_and_old_sources_still_read():
    """F5c: every source needs the buffer, and nothing runs without it; the sources of F5b read as literature +
    their buffer; a new file forgets the buffer (and the buffer an F5b source implied)."""
    for src in ("literature", "learned"):
        with pytest.raises(ValueError, match="imaging buffer"):
            el.LayerChoice.from_params({"xi_source": src})
    with pytest.raises(ValueError, match="imaging buffer"):
        el.LayerChoice.from_params({"xi_source": "calibrated", "xi_calibrated_mean": 9.5})
    with pytest.raises(ValueError, match="imaging buffer"):
        el.LayerChoice().prior()
    assert el.layer_source({"xi_source": "literature_MEA"}) == ("literature", "MEA")
    assert el.layer_source({"xi_source": "literature_bME"}) == ("literature", "bME")
    assert el.layer_source({"xi_source": "literature_bME", "buffer": "MEA"}) == ("literature", "MEA")
    assert el.layer_source({"xi_source": "learned", "buffer": "\u03b2ME"}) == ("learned", "bME")
    assert el.layer_source({}) == ("literature", None)
    old = el.LayerChoice.from_params({"xi_source": "literature_bME", "dye": "AF647"})
    assert (old.xi_source, old.buffer, old.prior().mean) == ("literature", "bME", 26.0)
    assert old.as_params()["buffer"] == "bME" and old.as_params()["xi_source"] == "literature"
    assert el.forget_buffer({"xi_source": "literature_MEA", "r_mol_nm": 40.0}) == {"xi_source": "literature",
                                                                                     "r_mol_nm": 40.0}
    assert el.forget_buffer({"xi_source": "learned", "buffer": "bME"}) == {"xi_source": "learned"}
    assert el.forget_buffer({}) == {}
    from tools.mps_analysis import analyze_axon
    x, y, z, sig, frames = _ring(seed=6)
    with pytest.raises(ValueError, match="imaging buffer"):
        analyze_axon(x, y, z, run_randomization=False, cluster_method=el.EMITTERS_METHOD,
                     cluster_params=_params(xi_source="literature"), localization_sigma_nm=sig,
                     localization_frame=frames)


def test_link_emitters_single_linkage():
    x = np.array([0.0, 40.0, 80.0, 300.0, 0.0])
    y = np.array([0.0, 0.0, 0.0, 0.0, 500.0])
    assert el.link_emitters(x, y, 50.0).tolist() == [0, 0, 0, 1, 2]     # chained 0-40-80
    assert el.link_emitters(x, y, 30.0).tolist() == [0, 1, 2, 3, 4]
    assert el.link_emitters(x, y, 0.0).tolist() == [0, 1, 2, 3, 4]
    assert el.link_emitters(np.empty(0), np.empty(0), 50.0).size == 0


def test_clean_precisions():
    s = np.array([[5.0, 6.0], [0.0, 7.0], [np.nan, np.nan], [9.0, 8.0]])
    out, n = el.clean_precisions(s, 4)
    assert n == 2 and np.all(out > 0) and out[1, 0] == 7.0 and out[2, 1] == 7.0
    assert el.clean_precisions(None, 4) == (None, 0)
    assert el.clean_precisions(np.zeros((4, 2)), 4) == (None, 0)
    assert el.clean_precisions(s, 5)[0] is None        # another slab's array is not used


def test_seconds_per_kloc_marks_the_method_slow():
    from tools.cluster_method_ui import SECONDS_PER_KLOC, estimate_seconds
    assert SECONDS_PER_KLOC[el.EMITTERS_METHOD] >= 0.5
    assert estimate_seconds([el.EMITTERS_METHOD], 10000) > 20.0


def test_link_events_weighted_mean_and_its_precision():
    xy = np.array([[0.0, 0.0], [4.0, 0.0], [300.0, 0.0], [302.0, 1.0]])
    sig = np.array([[5.0, 5.0], [10.0, 10.0], [8.0, 8.0], [8.0, 8.0]])
    frames = np.array([10, 11, 10, 13])                  # the last pair is 3 frames apart: two events
    ev = el.link_events(xy, sig, frames)
    assert ev.radius_nm == 3.0 * np.median([5.0, 10.0, 8.0, 8.0])     # LINK_RADIUS_IN_PRECISIONS x median
    assert ev.max_dark_frames == 1 and ev.n == 3
    assert ev.event_of_loc.tolist() == [0, 0, 1, 2]
    w = np.array([1 / 25, 1 / 100])
    assert np.isclose(ev.xy[0, 0], (0.0 * w[0] + 4.0 * w[1]) / w.sum())
    assert np.allclose(ev.sigma[0], 1.0 / np.sqrt(w.sum()))           # 1 / sqrt(sum 1 / lp^2)
    assert np.allclose(ev.sigma[1], 8.0)
    ev2 = el.link_events(xy, sig, frames, max_dark_frames=2)
    assert ev2.n == 2                                     # 10 -> 13 with two dark frames
    assert el.link_events(xy, sig, frames, radius_nm=3.0).n == 4


# ----------------------------------------------------------------- the analysis
def test_the_model_groups_events_linked_as_the_dnapaint_panel(emitters_analysis):
    from tools.mps_paint import LINK_RADIUS_IN_PRECISIONS, build_events
    a, _ = emitters_analysis
    data = a.emitters
    assert data.events_linked and a.localization_frame_slab is not None
    s = a.localization_sigma_slab
    lp = 0.5 * (s[:, 0] + s[:, 1])
    radius = LINK_RADIUS_IN_PRECISIONS * float(np.median(lp))
    ref = build_events(a.localization_frame_slab, np.asarray(a.x_slab), np.asarray(a.y_slab), lp_nm=lp,
                       radius_nm=radius, max_dark_time=1, remove_ambiguous_lengths=False)
    assert data.link_radius_nm == radius and data.max_dark_frames == 1
    assert data.n_events == ref.n < a.n_locs_slab
    assert np.array_equal(data.event_of_loc, ref.link_group)
    # the model saw one point per event, and every localization of an event has that event's emitter
    assert data.result.labels.shape == (ref.n,)
    for e in np.unique(data.event_of_loc):
        assert np.unique(data.emitter_labels[data.event_of_loc == e]).size == 1
    from tools import axon_export
    row = axon_export.axon_row(a, exported_at="t")
    assert row["emitters_n_events"] == ref.n and row["emitters_max_dark_frames"] == 1
    assert any(r[0].startswith("  events") and r[1] == str(ref.n) for r in a.summary_rows())


def test_link_radius_and_dark_frames_are_editable_and_recorded():
    from tools.mps_analysis import analyze_axon
    x, y, z, sig, frames = _ring(seed=4)
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method=el.EMITTERS_METHOD,
                     cluster_params=_params(xi_source="literature_MEA", link_radius_nm=15.0, max_dark_frames=0),
                     localization_sigma_nm=sig, localization_frame=frames)
    assert a.emitters.link_radius_nm == 15.0 and a.emitters.max_dark_frames == 0
    assert a.cluster_params["link_radius_nm"] == 15.0 and a.cluster_params["max_dark_frames"] == 0


def test_analysis_runs_the_layer_with_the_files_precisions(emitters_analysis):
    a, (_x, _y, _z, sig, _f) = emitters_analysis
    assert a.cluster_method == el.EMITTERS_METHOD
    data = a.emitters
    assert data is not None and data.per_localization_precision
    assert data.emitter_labels.shape == a.labels.shape
    # the precision the model saw is the file's, not sigma_nm = 12
    assert 6.0 <= data.precision_before_nm <= 10.0
    assert a.cluster_params["per_localization_sigma"] is True
    assert a.cluster_params["xi_source"] == "literature" and a.cluster_params["buffer"] == "MEA"
    assert a.cluster_params["r_mol_nm"] == 50.0
    assert a.cluster_scale_nm == 50.0
    r = a.emitter_readouts
    assert r is not None and r.n >= r.n_with_locs > 0
    assert r.xi_source == "literature" and r.buffer == "MEA" and r.xi_mean == 14.0
    assert r.per_cluster_q25 <= r.per_cluster_median <= r.per_cluster_q75
    assert r.per_um is not None and a.clusters_per_um is not None
    # the readouts count non-singleton emitters (F4 (d)); singletons are reported apart
    assert r.n_nonsingleton + r.n_singletons == r.n_with_locs
    assert r.n_emitters_in_clusters >= r.n_clusters      # every counted cluster has >= 1 non-singleton emitter
    assert r.n_clusters + r.n_singleton_only_clusters >= 1
    assert r.checks is not None and not r.checks.calibrate and not r.checks.mixture     # fixed xi: T1/T1' n/a
    assert r.precision_after_nm < r.precision_before_nm
    words = [row[0] for row in a.summary_rows()]
    assert "Emitters (BaGoL-like reimplementation)" in words


def test_molecules_are_emitters_linked_within_r_mol(emitters_analysis):
    a, _ = emitters_analysis
    data = a.emitters
    ok = data.emitter_labels >= 0
    # one emitter never spans two clusters
    for e in np.unique(data.emitter_labels[ok]):
        assert np.unique(a.labels[data.emitter_labels == e]).size == 1


def test_without_precisions_the_layer_says_so():
    from tools.mps_analysis import analyze_axon
    x, y, z, _, _f = _ring(seed=2)
    a = analyze_axon(x, y, z, run_randomization=False, cluster_method=el.EMITTERS_METHOD,
                     cluster_params=_params(xi_source="literature_bME", r_mol_nm=0.0))
    assert a.emitters is not None and not a.emitters.per_localization_precision
    assert a.emitters.precision_before_nm == 12.0      # the strategy's sigma_nm
    assert a.cluster_scale_nm is None                    # no linking: every emitter is a cluster
    assert a.emitter_readouts.per_cluster_median == 1.0
    assert any("sigma_nm for all" in row[3] for row in a.summary_rows())
    assert not a.emitters.events_linked and a.emitters.n_events == a.n_locs_slab
    assert any("not on frame-linked events" in w for w in a.warnings)
    from tools import axon_export
    row = axon_export.axon_row(a, exported_at="t")
    assert row["emitters_n_events"] is None and row["emitters_link_radius_nm"] is None


def test_dbscan_is_untouched_and_has_no_emitters():
    from tools.mps_analysis import analyze_axon
    x, y, z, sig, _f = _ring(seed=1)
    a = analyze_axon(x, y, z, run_randomization=False)
    b = analyze_axon(x, y, z, run_randomization=False, localization_sigma_nm=sig)
    assert a.emitters is None and a.emitter_readouts is None
    assert np.array_equal(a.labels, b.labels)


# ----------------------------------------------------------------- the table v12 and the files
def test_axon_row_v12_puts_the_emitters_block_last(emitters_analysis):
    from tools import axon_export
    from tools.column_dictionary import describe, known_columns, missing
    a, _ = emitters_analysis
    assert axon_export.TABLE_VERSION == "axon tables v12"
    row = axon_export.axon_row(a, exported_at="t")
    names = list(row)
    n = len(el.EMITTERS_COLUMNS)
    assert names[-n:] == list(el.EMITTERS_COLUMNS)
    assert names[-n - len(axon_export.CLUSTER_METHOD_COLUMNS):-n] == list(axon_export.CLUSTER_METHOD_COLUMNS)
    assert row["emitters_method"] == "Emitters (BaGoL-like reimplementation)"
    assert row["emitters_xi_source"] == "literature" and row["emitters_r_mol_nm"] == 50.0
    assert row["emitters_buffer"] == "MEA"                                   # v12 (F5c)
    assert row["emitters_precision_flag"] in ("ok", "under-reported", "not measured")
    assert list(el.EMITTERS_COLUMNS[-3:]) == ["emitters_buffer", "emitters_nena_over_sigma",
                                               "emitters_precision_flag"]
    assert json.loads(row["cluster_params"])["xi_source"] == "literature"
    assert json.loads(row["cluster_params"])["buffer"] == "MEA"
    assert known_columns()[-n:] == list(el.EMITTERS_COLUMNS)
    assert not missing(list(el.EMITTERS_COLUMNS))
    for c in el.EMITTERS_COLUMNS:
        assert describe(c), c


def test_dbscan_row_has_empty_emitters_cells():
    from tools import axon_export
    from tools.mps_analysis import analyze_axon
    x, y, z, _, _f = _ring(seed=1)
    row = axon_export.axon_row(analyze_axon(x, y, z, run_randomization=False), exported_at="t")
    assert row["emitters_method"] == "" and row["emitters_n"] is None and row["emitters_per_um"] is None
    assert row["emitters_buffer"] == "" and row["emitters_precision_flag"] == ""
    assert row["emitters_nena_over_sigma"] is None


def test_export_and_reimport_as_external_labels_gives_the_same_k_per_um(emitters_analysis, tmp_path):
    from tools.mps_analysis import analyze_axon
    a, _ = emitters_analysis
    written = el.write_emitter_files(a, str(tmp_path / "axons.csv"), "axon 1")
    assert [os.path.basename(p) for p in written] == ["axons_emitters_axon_1.csv", "axons_emitter_labels_axon_1.csv"]
    with open(written[0], encoding="utf-8") as fh:
        head = fh.readline()
    assert "method" in head and "x_nm" in head
    b = analyze_axon(np.asarray(a.x_slab), np.asarray(a.y_slab), np.asarray(a.z_slab), run_randomization=False,
                     slab_override=(a.slab_zmin_nm, a.slab_zmax_nm), cluster_method="External labels",
                     cluster_params={"csv_path": written[1], "scale_nm": a.cluster_scale_nm})
    assert np.array_equal(a.labels, b.labels)
    assert a.n_clusters_kept == b.n_clusters_kept
    assert a.clusters_per_um == b.clusters_per_um


# ----------------------------------------------------------------- the panel and the settings
@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])


def test_panel_shows_the_layer_only_for_the_method(qapp):
    from PyQt5 import QtWidgets
    from tools.cluster_method_ui import MethodPanel
    p = MethodPanel()
    assert el.EMITTERS_METHOD not in [p.combo.itemText(i) for i in range(p.combo.count())]   # D10
    p.chk_experimental.setChecked(True)
    emitted = []
    p.method_changed.connect(lambda: emitted.append(p.current_method()))
    p.combo.setCurrentIndex(p.combo.findText(el.EMITTERS_METHOD))
    assert emitted == []                       # chosen, not run: it is slow (D12)
    # F5c: no buffer chosen, nothing runs, and the panel says why
    buf = p.findChild(QtWidgets.QComboBox, "combo_emitters_buffer")
    assert buf.currentData() is None and p.buffer() is None
    assert not p.btn_layer_run.isEnabled()
    assert p.lbl_buffer_missing.isVisibleTo(p) and "MEA or \u03b2ME" in p.lbl_buffer_missing.text()
    p.btn_layer_run.click()
    assert emitted == []
    buf.setCurrentIndex(buf.findData("bME"))
    assert p.btn_layer_run.isEnabled() and not p.lbl_buffer_missing.isVisibleTo(p)
    assert buf.currentText() == "\u03b2ME" and p.current_params()["buffer"] == "bME"
    p.btn_layer_run.click()
    assert emitted == [el.EMITTERS_METHOD]     # 'Run with these values' runs it
    box = p.findChild(QtWidgets.QWidget, "emitters_layer_box")
    assert box is not None and not box.isHidden()
    assert p.findChild(QtWidgets.QWidget, "param_xi_mean") is None        # set by the xi source
    assert p.findChild(QtWidgets.QWidget, "param_xi_shape") is None       # shown as k
    src = p.findChild(QtWidgets.QComboBox, "combo_xi_source")
    assert [src.itemData(i) for i in range(src.count())] == ["literature", "learned", "calibrated"]
    assert src.currentData() == "literature"                 # the default source; the buffer has none
    cal = p.findChild(QtWidgets.QDoubleSpinBox, "spin_xi_calibrated_mean")
    assert not cal.isEnabled()
    src.setCurrentIndex(src.findData("calibrated"))
    assert cal.isEnabled()
    cal.setValue(11.0)
    p.findChild(QtWidgets.QDoubleSpinBox, "spin_r_mol_nm").setValue(40.0)
    p.findChild(QtWidgets.QDoubleSpinBox, "spin_xi_k").setValue(2.0)
    dye = p.findChild(QtWidgets.QComboBox, "combo_emitters_dye")
    dye.setCurrentIndex(dye.findData("Alexa Fluor 568"))
    params = p.current_params()
    assert params["xi_source"] == "calibrated" and params["xi_calibrated_mean"] == 11.0
    assert params["r_mol_nm"] == 40.0 and params["xi_shape"] == 2.0 and params["dye"] == "Alexa Fluor 568"
    assert "xi_mean" not in params
    assert params.get("link_radius_nm") is None and params["max_dark_frames"] == 1    # auto, the DNA-PAINT default
    p.findChild(QtWidgets.QDoubleSpinBox, "spin_link_radius_nm").setValue(20.0)
    p.findChild(QtWidgets.QSpinBox, "spin_max_dark_frames").setValue(2)
    params = p.current_params()
    assert params["link_radius_nm"] == 20.0 and params["max_dark_frames"] == 2
    assert "[deriv]" in p.findChild(QtWidgets.QDoubleSpinBox, "spin_r_mol_nm").toolTip()
    assert p.lbl_slow.isVisibleTo(p)
    # back from the settings
    q = MethodPanel()
    q.set_state(el.EMITTERS_METHOD, params, True, [])
    assert q.current_params()["xi_source"] == "calibrated" and q.buffer() == "bME"
    q.clear_buffer()                                         # a new file (F5c): chosen again
    assert q.buffer() is None and "buffer" not in q.current_params() and not q.btn_layer_run.isEnabled()
    assert q.current_params()["xi_source"] == "calibrated"   # the rest of the box is kept
    # an F5b source in the settings shows as 'literature' with the buffer NOT taken from it
    r = MethodPanel()
    r.set_state(el.EMITTERS_METHOD, {"xi_source": "literature_MEA"}, True, [])
    assert r.findChild(QtWidgets.QComboBox, "combo_xi_source").currentData() == "literature"
    assert r.buffer() is None
    assert q.findChild(QtWidgets.QDoubleSpinBox, "spin_r_mol_nm").value() == 40.0
    assert q.findChild(QtWidgets.QDoubleSpinBox, "spin_link_radius_nm").value() == 20.0
    assert q.findChild(QtWidgets.QSpinBox, "spin_max_dark_frames").value() == 2
    q.set_state("DBSCAN", {}, False, [])
    assert q.findChild(QtWidgets.QWidget, "emitters_layer_box").isHidden()


def test_settings_keep_the_layer(tmp_path):
    from tools.mps_settings import MPSSettings, load_settings, save_settings
    s = MPSSettings()
    s.cluster_method = el.EMITTERS_METHOD
    s.cluster_params = {"xi_source": "literature_bME", "dye": "Alexa Fluor 647", "xi_shape": 1.0,
                        "r_mol_nm": 45.0, "n_trials": 3000}
    s.experimental_methods_enabled = True
    save_settings(s, str(tmp_path))
    t = load_settings(str(tmp_path))
    assert t.cluster_method == el.EMITTERS_METHOD
    assert t.cluster_params["xi_source"] == "literature_bME" and t.cluster_params["r_mol_nm"] == 45.0


# ----------------------------------------------------------------- the batch
def _batch_settings(mb, **kw):
    return mb.BatchSettings(eps_nm=25.0, min_samples=10, slab_half_width_nm=90.0, dbcv_threshold=-1.0,
                            mahalanobis_threshold=3.0, **kw)


def test_batch_never_runs_the_layer_without_the_slow_box(tmp_path):
    from tools import mps_batch as mb
    from tools.mps_identity import AxonIdentity
    settings = _batch_settings(mb, cluster_method=el.EMITTERS_METHOD, cluster_params=_params())
    item = mb.BatchFile(path=str(tmp_path / "missing.hdf5"), identity=AxonIdentity())
    rec = mb.analyse_file(item, settings)
    assert not rec.row
    assert "Slow methods" in (rec.error or "")
    assert el.EMITTERS_METHOD in mb.BATCH_SLOW_METHODS
    # with the box ticked the method is attempted (here the file is missing, so the reason is another)
    rec2 = mb.analyse_file(item, settings, allow_slow=True)
    assert "Slow methods" not in (rec2.error or "")


def test_batch_comparison_skips_the_layer_without_the_slow_box(tmp_path):
    from tools import mps_batch as mb
    from tools.mps_identity import AxonIdentity
    settings = _batch_settings(mb)
    item = mb.BatchFile(path=str(tmp_path / "missing.hdf5"), identity=AxonIdentity())
    rows = mb.compare_methods([item], settings, ["DBSCAN", el.EMITTERS_METHOD])
    by = {r["cluster_method"]: r for r in rows}
    assert "Slow methods" in by[el.EMITTERS_METHOD]["error"]
    assert "Slow methods" not in by["DBSCAN"]["error"]


def test_batch_window_slow_box_is_off_by_default(qapp):
    from PyQt5 import QtWidgets
    from tools.batch_window import BatchWindow
    from tools.mps_settings import MPSSettings
    w = BatchWindow(MPSSettings())
    box = w.findChild(QtWidgets.QCheckBox, "check_slow_methods")
    assert box is not None and not box.isChecked()
    w.close()


def test_compare_methods_reruns_keep_the_slabs_precisions_and_frames(emitters_analysis, qapp, monkeypatch):
    """'Compare methods' re-runs the slab: the Emitters row must see the same precisions and frames."""
    import tools.mps_analysis as ma
    from tools.mps_results_window import MPSResultsWindow, WindowLinks
    a, _ = emitters_analysis
    seen = {}
    real = ma.analyze_axon

    def spy(*args, **kw):
        seen.update(kw)
        return real(*args, **kw)

    w = MPSResultsWindow(a, links=WindowLinks())
    monkeypatch.setattr(ma, "analyze_axon", spy)      # after the window: its registry reads the signature
    b = w._compare_run(el.EMITTERS_METHOD, _params(xi_source="literature_MEA"))
    assert np.array_equal(seen["localization_sigma_nm"], a.localization_sigma_slab)
    assert np.array_equal(seen["localization_frame"], a.localization_frame_slab)
    assert b.emitters.events_linked and b.emitters.n_events == a.emitters.n_events
    w.close()
