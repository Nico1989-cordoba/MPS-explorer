"""The interface batch's headless parts (2026-11-07; synthetic data only, rule R8): the radii of B1, the null's
membrane and the deformation indicator of C2, the closures of C3 / C4, the Batch's refusal of every slow method
(A2), the two-rings view's numbers (D1, rule D4) and the D2b split with the settings' c_label in H-ECL's arc tests.

Run: py -3 -m pytest test_ui_tanda2.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tools import clustering_integration as ci
from tools import null_membrane as nmb
from tools import two_rings_view as trv


def _circle(r, n=400, cx=0.0, cy=0.0):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


# ------------------------------------------------------------------------------------------------------------ B1
def test_radii_default_words_and_exploratory_mark():
    assert ci.radii_exploratory(None, None) == "no"
    assert ci.radii_exploratory(ci.LPC_RULE_RADIUS_IN_NENA, 3.0) == "no"
    w = ci.radii_exploratory(3.0, None)
    assert w.startswith("EXPLORATORY") and "lpc of the rule at 3 NeNA" in w and ";" not in w
    w2 = ci.radii_exploratory(None, 2.5)
    assert "events / fusion at 2.5 NeNA" in w2
    assert "+" in ci.radii_exploratory(3.0, 2.5)


def test_v16_columns_are_last_and_described():
    from tools import axon_export, column_dictionary
    assert axon_export.TABLE_VERSION == "axon tables v21"
    cols = ci.integration_columns()
    assert cols[-len(ci.V16_COLUMNS):] == ci.V16_COLUMNS
    for c in ci.V16_COLUMNS:
        assert column_dictionary.describe(c), c
    assert len(set(cols)) == len(cols)


def test_axon_columns_say_which_radius_was_used():
    from tools import mps_pipeline_null as pn
    an, _ax = pn.observe_synthetic(pn.synthetic_model((900.0, 700.0)), 3)
    row = ci.axon_columns(an, nena_nm=12.0, nena_source="synthetic")
    assert row["radii_exploratory"] == "no" and row["link_radius_nena"] == 3.0
    assert row["lpc_rule_radius_nena"] == pytest.approx(4.2426, abs=1e-3)
    row2 = ci.axon_columns(an, nena_nm=12.0, nena_source="synthetic", lpc_radius_in_nena=3.0,
                           events_radius_in_nena=2.0)
    assert row2["radii_exploratory"].startswith("EXPLORATORY")
    assert row2["link_radius_nena"] == 2.0 and row2["link_radius_nm"] == pytest.approx(24.0)
    assert row2["lpc_rule_radius_nena"] == 3.0
    # the events plug-in reads the radius: a smaller one links fewer localizations per event
    ev_def = row["events_locs_per_event_mean"]
    ev_small = row2["events_locs_per_event_mean"]
    if ev_def is not None and ev_small is not None:
        assert ev_small <= ev_def + 1e-9


# ------------------------------------------------------------------------------------------------------------ C2
def test_deformation_of_two_concentric_circles_is_their_distance():
    d = nmb.membrane_deformation(_circle(1000.0), _circle(1030.0))
    assert d.mean_nm == pytest.approx(30.0, abs=1.5) and d.max_nm == pytest.approx(30.0, abs=1.5)
    assert d.n_flagged == 0 and d.level() == "look"
    far = nmb.membrane_deformation(_circle(1000.0), _circle(1080.0))
    assert far.n_flagged == len(far.stretch_max_nm) and far.level() == "warn" and "WARNING" in far.sentence()
    same = nmb.membrane_deformation(_circle(1000.0), _circle(1000.0, n=777))
    assert same.max_nm < 2.0 and same.level() == "ok"


def test_deformation_flags_only_the_bulging_stretch():
    c = _circle(1000.0, n=800)
    t = np.linspace(0.0, 2.0 * math.pi, 800, endpoint=False)
    bump = 1000.0 + 120.0 * np.exp(-((t - 1.0) / 0.12) ** 2)       # one bulge of 120 nm, ~240 nm wide
    m = np.column_stack([bump * np.cos(t), bump * np.sin(t)])
    d = nmb.membrane_deformation(c, m)
    assert 1 <= d.n_flagged <= 3 and d.max_nm == pytest.approx(120.0, abs=5.0)
    for a, b, top in d.flagged:
        assert top > nmb.WARN_NM and b - a <= nmb.STRETCH_NM + 1e-6


def test_null_membrane_default_is_dbscan_and_the_reference_null_is_unchanged():
    from tools import mps_pipeline_null as pn
    from tools import multimethod as mm
    an, _ax = pn.observe_synthetic(pn.synthetic_model((900.0, 700.0)), 3)
    assert nmb.reference_analysis(an) is an                        # the synthetic analysis is DBSCAN 25/10
    m = nmb.null_membrane_for(an)
    assert m.choice == nmb.MEMBRANE_DBSCAN and m.curve_nm is not None and m.deformation is not None
    np.testing.assert_allclose(m.curve_nm, nmb.centroid_curve(an))
    run = ci.run_of(an)
    out_ref = ci.run_null_for(run, n_replicates=3, nena_nm=12.0, seed=1)
    assert out_ref.ok and out_ref.membrane == nmb.MEMBRANE_WORDS[nmb.MEMBRANE_DBSCAN]
    # the reference keeps the null's own reading (no override): its origin is the old one
    assert out_ref.result.model.origins["membrane_nm"].startswith("read: 600 nm P-spline through the observed")
    out_loc = ci.run_null_for(run, n_replicates=3, nena_nm=12.0, seed=1, membrane=nmb.MEMBRANE_LOCALIZATIONS)
    assert out_loc.ok and out_loc.membrane == nmb.MEMBRANE_WORDS[nmb.MEMBRANE_LOCALIZATIONS]
    assert "localizations" in out_loc.result.model.origins["membrane_nm"]
    # another method's null gets DBSCAN 25/10's curve
    focal = mm.analyze_like(an, an.x_slab, an.y_slab, an.z_slab, "FOCAL", {})
    out_f = ci.run_null_for(ci.run_of(focal), n_replicates=3, nena_nm=12.0, seed=1)
    assert out_f.status in ("ok", "refused", "error")
    if out_f.ok:
        assert "DBSCAN 25/10" in out_f.result.model.origins["membrane_nm"]
    with pytest.raises(ValueError):
        nmb.check_choice("mean of both")


# --------------------------------------------------------------------------------------------------------- C3/C4
def test_run_nulls_both_closures_keys_and_comparison_rows():
    from tools import mps_pipeline_null as pn
    an, _ax = pn.observe_synthetic(pn.synthetic_model((900.0, 700.0)), 3)
    run = ci.run_of(an)
    out = ci.run_nulls([run], n_replicates=3, nena_nm=12.0, seed=2, closure="both")
    assert list(out) == [run.key, run.key + ci.K_CLOSURE_SUFFIX]
    assert out[run.key].closure == "n_locs" and out[run.key + ci.K_CLOSURE_SUFFIX].closure == "k_clusters"
    pairs = ci.closure_pairs(out)
    assert [p[0] for p in pairs] == [run.key]
    rows = ci.closure_comparison_rows(out)
    assert {r["test"] for r in rows} >= {"H1 global (arc)", "H2 global (band)", "clusters per um"}
    assert all(set(r) >= {"p_n_locs", "p_k_clusters", "same_verdict_at_0_05"} for r in rows)
    # the default stays one null, N_loc, under the run's own key
    one = ci.run_nulls([run], n_replicates=3, nena_nm=12.0, seed=2)
    assert list(one) == [run.key] and one[run.key].closure == "n_locs"
    assert ci.closure_key("x", "k_clusters", "k") == "x"


# ------------------------------------------------------------------------------------------------------------ A2
@pytest.mark.parametrize("method", ["OPTICS", "voronoi_csr", "emitters_bagol_like"])
def test_the_batch_refuses_every_slow_method_without_the_box(tmp_path, method):
    from tools import mps_batch as mb
    from tools.mps_identity import AxonIdentity
    settings = mb.BatchSettings(eps_nm=25.0, min_samples=10, slab_half_width_nm=90.0, dbcv_threshold=-1.0,
                                mahalanobis_threshold=3.0, cluster_method=method, cluster_params={})
    item = mb.BatchFile(path=str(tmp_path / "missing.hdf5"), identity=AxonIdentity())
    rec = mb.analyse_file(item, settings)
    assert not rec.row and "Allow slow methods" in (rec.error or "")
    rec2 = mb.analyse_file(item, settings, allow_slow=True)
    assert "Allow slow methods" not in (rec2.error or "")


def test_slow_confirm_text_and_answer_hook():
    from tools import slow_confirm as sc
    text = sc.confirm_text("The Batch", [("OPTICS", 600.0), ("voronoi_csr", 7200.0)])
    assert "OPTICS: about 10 min" in text and "voronoi_csr: about 2.0 h" in text and "lower bound" in text
    sc.ANSWER, sc.ASKED[:] = (lambda t: False), []
    try:
        assert sc.confirm_slow(None, "X", [("OPTICS", 10.0)]) is False
        assert sc.confirm_slow(None, "X", []) is True                  # nothing slow: never asked
        assert len(sc.ASKED) == 1
    finally:
        sc.ANSWER = None
        sc.ASKED[:] = []


# ------------------------------------------------------------------------------------------------------------ D1
def _pair(shift=0.0, k=30, length=6000.0):
    s_a = np.linspace(0.0, length, k, endpoint=False) + 7.0
    s_b = np.mod(s_a + shift, length)
    t = 2.0 * math.pi * s_a / length
    r = length / (2.0 * math.pi)
    xy_a = np.column_stack([r * np.cos(t), r * np.sin(t)])
    tb = 2.0 * math.pi * s_b / length
    xy_b = np.column_stack([r * np.cos(tb), r * np.sin(tb)])
    return trv.PairData(0, 1, length, s_a, s_b, xy_a, xy_b, np.zeros(2))


def test_two_rings_counts_roles_and_export():
    d = _pair(shift=10.0)
    r = trv.read_pair(d, 31.0, n_rotations=50)
    assert r.role == trv.ROLE_PRIMARY and not r.exploratory
    assert r.n_arc == 30 and r.n_plane == 30
    assert 0.0 < r.expected_arc < 30.0 and r.expected_poisson == pytest.approx(30 * (1 - math.exp(-30 / 6000 * 62)))
    r2 = trv.read_pair(d, 5.0, n_rotations=50)
    assert r2.exploratory and r2.n_arc == 0
    rows = trv.export_rows(d, r2, "synthetic")
    assert len(rows) == 60 and all(row["tau_role"] == trv.ROLE_EXPLORATORY for row in rows)
    assert list(rows[0]) == list(trv.EXPORT_COLUMNS)
    assert trv.tau_role(31.0) == trv.ROLE_PRIMARY and trv.tau_role(31.004) == trv.ROLE_PRIMARY
    assert trv.tau_role(40.0) == trv.ROLE_EXPLORATORY
    with pytest.raises(ValueError):
        trv.read_pair(d, 0.0)


# ------------------------------------------------------------------------------------------------- H-ECL (D2b)
def test_hecl_arc_tests_split_merged_clusters_with_the_given_c_label(tmp_path):
    pytest.importorskip("PyQt5")                    # the review's module is a Qt window module (CI: headless)
    import batch_columns
    from tools import mps_columns_window as mcw
    from tools import mps_zquality_window as zw
    from tools.mps_lumen import LumenDecisions, run_cleaned_analyses
    contour, seed, scale = zw.DEMO_CASES["viable"]
    npz = batch_columns.write_simulated_input(str(tmp_path / "sim.npz"), contour, seed, scale)
    inputs = zw.inputs_from_npz(npz)
    review = mcw.prepare_review(inputs)
    dec = LumenDecisions.from_classification(review.classification)
    res = run_cleaned_analyses(review.res, dec, n_null=9, lpz_nm=review.lpz_nm, c_label=26.0)
    assert res.merged_handling == "split" and res.c_label == 26.0
    assert res.rings_of("2d") is res.rings
    if res.n_merged_split:
        assert res.arc_rings is not None and res.rings_of("arc") is res.arc_rings
    mid = run_cleaned_analyses(review.res, dec, n_null=9, lpz_nm=review.lpz_nm, merged_handling="middle")
    assert mid.merged_handling == "middle" and mid.arc_rings is None and mid.c_label == 14.0


# --------------------------------------------------------------------------------------------- G2 / G3 / G4 / G5
def test_g_decisions_are_recorded_as_decided():
    for key in ("G2", "G3", "G4", "G5", "A2", "B1", "C2"):
        p = ci.PROVISIONAL[key]
        assert "user decision" in p.origin and p.applied, key
    for key in ("G2", "G3", "G4", "G5"):
        assert "2026-11-07" in ci.PROVISIONAL[key].origin, key
    assert "500 nm" in ci.PROVISIONAL["G5"].origin and "4 of 12" in ci.LATTICE_WORDS


def test_batch_summary_says_the_per_axon_label_is_descriptive():
    rows = []
    rng = np.random.default_rng(5)
    for animal in ("m1", "m2"):
        for i in range(3):
            rows.append({"animal": animal, "genotype": "WT", "regularity_reading": "random_with_contact",
                         "gap_cv_z": float(rng.normal()), "gap_excess_cv_z": float(rng.normal()),
                         "n_clusters_kept": 40 + i, "perimeter_um": 10.0 + i})
    text = " ".join(ci.regularity_batch_summary(rows)["lines"])
    assert "descriptive only" in text and "4 of 12" in text and "[own sim" in text


def test_roi_tooltips_recommend_the_margin():
    pytest.importorskip("PyQt5")                    # tools.mps_tooltips imports Qt (CI: headless)
    from tools.mps_tooltips import MAIN_WINDOW as TOOLTIPS
    for name in ("radioButton_circROI", "radioButton_squareROI", "radioButton_polygonROI"):
        assert "at least 500 nm" in TOOLTIPS[name] and "0 clusters removed in 60" in TOOLTIPS[name]
