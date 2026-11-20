# -*- coding: utf-8 -*-
"""
tools/mps_pipeline_null.py and tools/mps_working_scale.py on SYNTHETIC axons only (rule R8): the generator
(determinism, cycles sharing their label's position, loc/cycle read back by event linking), the working scale,
the refusals (K < 8, slow methods with an estimate, External labels, a jammed hard core), the closure, a small run
(deterministic, NOT INTERPRETABLE below 19 replicates, p in range), method-agnosticism (the null re-runs HDBSCAN when
the observed was HDBSCAN), the 2D fallback, n_jobs = 2 equal to n_jobs = 1, and the committed calibration tables.

The rejection rate under the null and the power are NOT computed here (hours): they are in
docs/sims/pipeline_null/ with their output tables committed, and the last test checks those tables.

Run:  py -3 -m pytest test_pipeline_null.py -q
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import mps_pipeline_null as pn  # noqa: E402
from tools.mps_working_scale import (  # noqa: E402
    d50_rule_nm, fusion_nm_from_d50, working_scale, working_scale_for_method)
from tools.spatial_stats import (  # noqa: E402
    csr_envelopes, envelope_from_curves, ripley_k_1d_arc, ripley_k_2d, ripley_k_2d_area, simulate_csr_arc)

SMALL = (1000.0, 700.0)          # perimeter ~5.4 um: ~20 molecules per ring, ~1.3k slab localizations
SIMS = os.path.join(REPO_ROOT, "docs", "sims", "pipeline_null")


@pytest.fixture(scope="module")
def observed_small():
    model = pn.synthetic_model(SMALL)
    an, ax = pn.observe_synthetic(model, 3)
    return model, an, ax


# ------------------------------------------------------------------ generator

def test_generator_is_deterministic() -> None:
    model = pn.synthetic_model(SMALL)
    a = pn.simulate_null_axon(model, np.random.default_rng(5))
    b = pn.simulate_null_axon(model, np.random.default_rng(5))
    c = pn.simulate_null_axon(model, np.random.default_rng(6))
    for f in ("x_nm", "y_nm", "z_nm", "frame", "molecule", "label"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    assert a.n_locs != c.n_locs or not np.array_equal(a.x_nm, c.x_nm)
    assert np.all(np.diff(a.frame) >= 0)                       # frame order, as a file


def test_cycles_of_one_label_share_its_position() -> None:
    # sigma_xy 5 nm, linkage 12 nm: the localizations of one label scatter by sigma_xy only (the linkage is drawn
    # once per label), whereas two labels of one molecule differ by the linkage and the epitope disc
    model = pn.synthetic_model(SMALL, sigma_xy_nm=5.0, background_per_um2=0.0)
    ax = pn.simulate_null_axon(model, np.random.default_rng(11))
    sds = []
    for lab in np.unique(ax.label[ax.label >= 0])[:300]:
        sel = ax.label == lab
        if np.count_nonzero(sel) >= 8:
            sds.append(np.std(ax.x_nm[sel], ddof=1))
    assert 4.0 < float(np.median(sds)) < 6.0


def test_locs_per_cycle_read_back_by_event_linking() -> None:
    model = pn.synthetic_model(SMALL, locs_per_cycle=2.0, background_per_um2=0.0)
    ax = pn.simulate_null_axon(model, np.random.default_rng(2))
    lpc, n_ev = pn.locs_per_cycle_from_events(ax.frame, ax.x_nm, ax.y_nm, 12.0, labels=ax.molecule)
    assert n_ev > 100
    assert abs(lpc - 2.0) < 0.15


def test_slab_capture_matches_simulation() -> None:
    model = pn.synthetic_model(SMALL, background_per_um2=0.0)
    ax = pn.simulate_null_axon(model, np.random.default_rng(4))
    in_slab = np.count_nonzero(np.abs(ax.z_nm - model.z_centre_nm) <= model.slab_half_width_nm)
    expected = model.n_molecules_per_ring * model.slab_capture_per_molecule()
    assert abs(in_slab / expected - 1.0) < 0.25


# ------------------------------------------------------------------ working scale and spatial_stats helpers

def test_working_scale_rule_and_origins() -> None:
    ws = working_scale(25.0, 10, 12.0)
    assert d50_rule_nm(25.0, 12.0) == pytest.approx(80.0)
    assert ws.d50_nm == pytest.approx(80.0) and ws.fusion_nm == pytest.approx(60.0)
    assert fusion_nm_from_d50(10.0) == 0.0
    # A9 manual test (ii): eps 25 -> 20 lowers d50 by 10 nm
    assert working_scale(20.0, 10, 12.0).d50_nm == pytest.approx(70.0)
    origins = {m.key: m.origin for m in ws.marks()}
    assert origins["eps_nm"] == "literature" and origins["d50_nm"] == "simulation"
    assert origins["nena_nm"] == "measured"
    cols = ws.export_columns()
    assert set(cols) == {"scale_eps_nm", "scale_min_samples", "scale_nena_nm", "scale_fusion_nm", "scale_d50_nm",
                         "scale_slab_nm"}
    other = working_scale_for_method("HDBSCAN", {"min_samples": 5}, 12.0, separation_scale_nm=221.0)
    assert other.eps_nm == pytest.approx(25.0) and other.fusion_nm == pytest.approx(60.0)
    assert other.method_scale_nm == pytest.approx(221.0) and {m.key: m.origin for m in other.marks()}["d50_nm"] == "ad hoc"
    assert working_scale(25.0, 10, 22.0).rule_applies is False
    assert working_scale(25.0, 10, None).nena_source == "paper"


def test_envelope_from_curves_equals_csr_envelopes() -> None:
    r = pn.default_r_grid(200.0)
    fn = lambda s: ripley_k_1d_arc(s, 3000.0, r).h  # noqa: E731
    obs = fn(simulate_csr_arc(30, 3000.0, np.random.default_rng(1)))
    e1 = csr_envelopes(fn, 30, 3000.0, r, n_sim=19, seed=3, observed=obs, r_min=60.0)
    e2 = envelope_from_curves(e1.sims, r, observed=obs, r_min=60.0, seed=3)
    assert e1.p_global == e2.p_global and e1.r_star == e2.r_star
    assert np.array_equal(e1.global_hi, e2.global_hi)


def test_ripley_area_form_equals_polygon_form() -> None:
    rect = np.array([[0.0, 0.0], [3000.0, 0.0], [3000.0, 2000.0], [0.0, 2000.0]])
    xy = np.random.default_rng(0).uniform([0, 0], [3000, 2000], (80, 2))
    r = pn.default_r_grid()
    a = ripley_k_2d(xy, rect, r, correction="none")
    b = ripley_k_2d_area(xy, 6.0e6, r)
    assert np.allclose(a.k, b.k) and np.allclose(a.h, b.h)


# ------------------------------------------------------------------ refusals

def test_refuses_fewer_than_eight_clusters() -> None:
    model = pn.synthetic_model((260.0, 200.0), molecules_per_um=3.0)
    an, _ax = pn.observe_synthetic(model, 1)
    assert an.n_clusters_kept < pn.MIN_CLUSTERS
    with pytest.raises(pn.NullRefused, match="not testable"):
        pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=3)


def test_refuses_slow_method_with_an_estimate_and_external_labels(observed_small) -> None:
    _m, an, _ax = observed_small
    slow = pn.PipelineSpec.for_method("emitters_bagol_like")
    with pytest.raises(pn.SlowMethodRefused) as info:
        pn.check_method(slow, int(an.n_locs_slab), 199)
    assert info.value.estimate_s > 60.0 and "allow_slow" in str(info.value)
    assert pn.check_method(slow, int(an.n_locs_slab), 199, allow_slow=True) > 0
    with pytest.raises(pn.NullRefused, match="file"):
        pn.check_method(pn.PipelineSpec.for_method("External labels"), int(an.n_locs_slab), 9, allow_slow=True)
    with pytest.raises(pn.SlowMethodRefused):
        pn.run_pipeline_null(an, spec=slow, nena_nm=12.0, n_replicates=3)


def test_refuses_a_hard_core_that_would_jam(observed_small) -> None:
    _m, an, _ax = observed_small
    with pytest.raises(pn.NullRefused, match="jams"):
        pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=2, hard_core_nm=400.0)


# ------------------------------------------------------------------ model, closure, runs

def test_model_read_from_the_axon_says_where_every_value_comes_from(observed_small) -> None:
    truth, an, _ax = observed_small
    m = pn.null_model_from_analysis(an, nena_nm=12.0)
    assert m.geometry == "ring"
    for key in ("sigma_xy_nm", "membrane_nm", "hard_core_nm", "radial_sd_nm", "sigma_z_nm", "period_nm",
                "locs_per_cycle", "c_label", "background_per_um2", "epitope_radius_nm", "n_frames"):
        assert key in m.origins, key
    assert m.origins["locs_per_cycle"].startswith("read")          # frames were kept by the analysis
    assert abs(m.locs_per_cycle - truth.locs_per_cycle) < 0.4
    assert abs(m.period_nm - truth.period_nm) < 25.0
    assert 0.5 * truth.background_per_um2 < m.background_per_um2 < 2.0 * truth.background_per_um2
    assert m.hard_core_nm == pytest.approx(2.0 * an.median_r_eff_nm)


def test_closure_on_localizations_and_on_clusters(observed_small) -> None:
    _m, an, _ax = observed_small
    model = pn.null_model_from_analysis(an, nena_nm=12.0)
    spec = pn.PipelineSpec.from_analysis(an)
    rec = pn.close_null(model, spec, closure="n_locs", target=float(an.n_locs_slab), seed=0)
    assert len(rec.trajectory) == pn.N_PILOTS
    m2 = pn.dataclasses.replace(model, n_molecules_per_ring=rec.n_molecules_per_ring)
    counts = [pn._slab_count(m2, pn.simulate_null_axon(m2, np.random.default_rng(100 + j)))[0] for j in range(40)]
    assert abs(np.median(counts) / an.n_locs_slab - 1.0) < 0.06
    rec_k = pn.close_null(model, spec, closure="k_clusters", target=float(an.n_clusters_kept), seed=0, n_pilots=3)
    assert abs(rec_k.trajectory[-1][1] / an.n_clusters_kept - 1.0) < 0.35


def test_small_run_is_deterministic_and_not_interpretable(observed_small) -> None:
    _m, an, _ax = observed_small
    r1 = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=6, seed=4)
    r2 = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=6, seed=4)
    d1, d2 = r1.export_dict(), r2.export_dict()
    assert d1 == d2
    assert r1.replicate_rows() == r2.replicate_rows()
    assert not r1.interpretable and r1.headline().startswith("NOT INTERPRETABLE")
    assert any("never reach 0.05" in s for s in r1.not_interpretable_reasons())
    assert d1["null_interpretable"] is False and d1["null_B"] == 6 and d1["null_closure"] == "n_locs"
    for name in ("k_per_um", "nn_median", "reff_median", "occupancy", "regularity"):
        p = d1[f"p_mc_{name}"]
        assert p is None or (1.0 / 7.0 - 1e-9 <= p <= 1.0)
    assert d1["ripley1d_window"] == "closed" and d1["ripley2d_window"] == "band"
    assert d1["scale_fusion_nm"] == pytest.approx(60.0) and d1["null_type"] == pn.NULL_TYPE_PROVISIONAL
    assert len(r1.ripley_rows()) == r1.r_grid.size
    r3 = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=6, seed=5)
    assert r3.replicate_rows() != r1.replicate_rows()


def test_k_closure_leaves_the_k_p_value_empty(observed_small) -> None:
    _m, an, _ax = observed_small
    res = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=3, closure="k_clusters")
    d = res.export_dict()
    assert d["p_mc_k_per_um"] is None and d["null_closure"] == "k_clusters"
    assert any(c.name == "detected clusters K" for c in res.fidelity)


def test_the_null_re_runs_the_observed_method() -> None:
    spec = pn.PipelineSpec.for_method("HDBSCAN")
    an, _ax = pn.observe_synthetic(pn.synthetic_model(SMALL), 3, spec)
    assert an.cluster_method == "HDBSCAN"
    res = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=2)
    assert res.method == "HDBSCAN"
    assert res.example is not None and res.example["cluster_method"] == "HDBSCAN"
    assert res.export_dict()["null_method"] == "HDBSCAN"


def test_two_dimensional_fallback(observed_small) -> None:
    _m, an, _ax = observed_small
    res = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=3, no_ring=True)
    d = res.export_dict()
    assert res.geometry == "roi2d" and res.h1 is None
    assert d["ripley1d_window"] == "none" and d["ripley2d_window"] == "roi"
    assert "p_mc_regularity" not in d and "p_mc_occupancy" not in d
    assert any("2D only" in w for w in res.warnings)


def test_registered_readout_hook(observed_small) -> None:
    _m, an, _ax = observed_small
    spec = pn.ReadoutSpec("n_clusters_test", _n_clusters, "", False, True, "test hook")
    pn.register_readout(spec)
    try:
        res = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=3, readouts=["n_clusters_test"])
        assert "p_mc_n_clusters_test" in res.export_dict()
        assert res.observed["n_clusters_test"] == float(an.n_clusters_kept)
    finally:
        pn.READOUTS.pop("n_clusters_test", None)


def _n_clusters(view: pn.ReplicateView) -> float:
    return float(view.analysis.n_clusters_kept)


def test_parallel_equals_serial(observed_small) -> None:
    _m, an, _ax = observed_small
    serial = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=2, seed=7, n_jobs=1)
    parallel = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=2, seed=7, n_jobs=2)
    assert serial.replicate_rows() == parallel.replicate_rows()


def test_multimethod_readout_names_follow_the_convention() -> None:
    for name in list(pn.MULTIMETHOD_NAMES.values()) + ["null_closure_ratio", "null_seconds_per_replicate_s",
                                                       "null_interpretable_verdict"]:
        assert name == name.lower() and name.replace("_", "").isalnum()
        assert name.startswith("null_") or name.startswith("n_null")
        if "p_global" in name or name.endswith("_p"):
            assert name.endswith("_p")


# ------------------------------------------------------------------ the committed calibration tables

def test_committed_calibration_meets_the_level() -> None:
    path = os.path.join(SIMS, "calibration_null.json")
    if not os.path.exists(path):
        pytest.skip("calibration table not committed yet")
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)
    n = int(table["n_axons"])
    rej = table["rejections_at_0.05"]
    # A9 sec. 6.3: global 1D rejections <= 7 of 60 (P(fail | level 5 %) = 0.03), "and the same for each p_mc_*";
    # scaled to n. H2 has no criterion in A9: its rate is reported in docs/null_and_scale.md section 6.2, not gated.
    limit = int(np.ceil(7.0 * n / 60.0))
    assert rej["ripley1d_p_global"] <= limit
    for key in ("p_mc_k_per_um", "p_mc_nn_median", "p_mc_reff_median", "p_mc_occupancy", "p_mc_regularity"):
        assert rej[key] <= limit, key
    # the pointwise envelope is NOT a test: it must exceed the nominal 5 % by far (A9 sec. 2.6)
    assert table["pointwise_exceedances"]["pointwise1d_exceeds"] > 3 * limit
