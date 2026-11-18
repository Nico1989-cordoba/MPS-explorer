# -*- coding: utf-8 -*-
"""
The three validations of branch ``feat/validations-d2-e1-e5`` (2026-10-29). SYNTHETIC DATA ONLY (R8).

- D2 (``tools/mps_eclipse_merged.py``): the split of a merged-looking cluster is the same two-means split that
  measured its mode separation; split / middle / exclude give the documented centres; the split rings pass the
  geometry check of the matching and the arc test runs on them; the eclipse plug-in reports the handling; the
  committed simulation backs the default.
- E1 (``docs/sims/apparent_size/elongation_cut_clabel.json``): the committed run shows the 2.5 cut within 2 nm of
  the true footprint at c_label 14, 20 and 26.
- E5 (``tools/size_curvature_null.py``): the question-(b) statistic is one function for the observed axon and the
  replicates, it rides the SAME pipeline null (registry), its p reaches the size plug-in and the export columns,
  and the committed calibration has 5 % inside its Wilson interval.

Run: ``py -3 -m pytest test_validations_d2_e1_e5.py -q`` (about 1-2 min, one process).
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

from tools import mps_eclipse_merged as mg
from tools import mps_eclipse_tolerance as et

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "docs", "sims", "eclipse_tolerance"))
import eclipse_sim as es  # noqa: E402


# ============================================================================================== D2: the split
def _two_clusters(rng: np.random.Generator, sep: float) -> Dict[str, Any]:
    t1, n1, e1 = es.draw_cluster(rng, 25.0, 3)
    t2, n2, e2 = es.draw_cluster(rng, 25.0, 3)
    t = np.concatenate([t1, t2 + sep])
    n = np.concatenate([n1, n2])
    ev = np.concatenate([e1, e2 + int(e1.max()) + 1])
    return dict(t=t, n=n, ev=ev, n1=t1.size)


def test_split_is_the_split_that_measured_the_separation() -> None:
    rng = np.random.default_rng(1)
    for i in range(60):
        c = _two_clusters(rng, float(rng.uniform(0.0, 150.0)))
        for ev in (c["ev"], None):
            upper = mg.two_mode_split(c["t"], ev)
            sep = et.two_mode_separation(c["t"], ev, et.MERGED_MIN_UNITS_PER_SIDE if ev is not None
                                         else 2 * et.MERGED_MIN_UNITS_PER_SIDE)
            assert upper is not None and upper.any() and not upper.all()
            if ev is None:
                gap = abs(float(np.mean(c["t"][upper])) - float(np.mean(c["t"][~upper])))
                assert gap == pytest.approx(sep, rel=1e-9, abs=1e-9)
            else:
                # events are the units: every localization of an event on one side; weighted means = separation
                for e in np.unique(ev):
                    assert len(set(upper[ev == e].tolist())) == 1
                gap = abs(float(np.mean(c["t"][upper])) - float(np.mean(c["t"][~upper])))
                assert gap == pytest.approx(sep, rel=1e-9, abs=1e-9)
    assert mg.two_mode_split(np.array([0.0, 1.0, 2.0]), np.array([0, 0, 1])) is None      # too few units


def test_handlings_on_a_constructed_merge() -> None:
    rng = np.random.default_rng(2)
    R = 5000.0
    c = _two_clusters(rng, 110.0)
    # along a circle of radius R around the origin, centred at angle 0 (tangent = +y)
    x = R + c["n"]
    y = c["t"] - 55.0
    single_t, single_n, single_e = es.draw_cluster(rng, 25.0, 3)
    xs = np.concatenate([x, R * math.cos(0.2) + single_n])
    ys = np.concatenate([y, R * math.sin(0.2) + single_t])
    ev = np.concatenate([c["ev"], single_e + int(c["ev"].max()) + 1])
    members = [np.arange(x.size), np.arange(x.size, xs.size)]
    tol = et.axon_tolerances(xs, ys, {0: members}, event_id=ev, centres_nm={0: [0.0, 0.0]})
    flags = tol.groups[0].merged_suspect
    assert flags.tolist() == [True, False]
    mid = mg.centres_with_handling(xs, ys, members, flags, "middle", event=ev)
    spl = mg.centres_with_handling(xs, ys, members, flags, "split", event=ev)
    exc = mg.centres_with_handling(xs, ys, members, flags, "exclude", event=ev)
    assert mid.centres_nm.shape == (2, 2) and abs(mid.centres_nm[0, 1]) < 15.0        # between the two
    assert spl.centres_nm.shape == (3, 2) and spl.n_split == 1 and spl.half.tolist() == [1, 2, 0]
    assert sorted(np.round(spl.centres_nm[:2, 1] / 55.0).tolist()) == [-1.0, 1.0]        # the two real centres
    assert spl.n_locs[:2].sum() == members[0].size
    assert exc.usable.tolist() == [False, True] and exc.n_excluded == 1
    s = mg.half_sigma_c_nm(spl.n_locs[:2], spl.n_events[:2], tol)
    assert np.all(s > tol.groups[0].sigma_c_nm[1] * 0.5) and np.all(np.isfinite(s))
    with pytest.raises(ValueError):
        mg.check_handling("both")
    assert mg.check_handling(None) == mg.MERGED_HANDLING_DEFAULT


@pytest.fixture(scope="module")
def sim_rings() -> Any:
    import tempfile

    import batch_columns as bc
    from tools.mps_columns import build_rings, rings_params_from
    from tools.mps_lumen import default_columns_params
    folder = tempfile.mkdtemp(prefix="d2_rings_")
    z = np.load(bc.write_simulated_input(os.path.join(folder, "sim.npz"), "circle", 39, 0.3))
    res = build_rings(z["x"], z["y"], z["z"], frame=z["frame"], lp_lateral_nm=z["lp"], lpz_nm=z["lpz"],
                      params=rings_params_from(default_columns_params()), n_frames=int(z["n_frames"]))
    return res, {k: z[k] for k in ("x", "y", "z", "frame", "lp", "lpz", "n_frames")}


def test_split_rings_pass_the_geometry_check_and_the_arc_test_runs(sim_rings: Any) -> None:
    from tools.mps_matching import ring_geometry
    from tools.mps_unroll import UnrollParams, analyze_arc_columns
    res, _ = sim_rings
    tol = et.tolerances_from_rings(res)
    # force two clusters of the first two rings to look merged (the simulated axon has few real merges)
    forced = {}
    for ring in res.rings[:2]:
        g = tol.groups[int(ring.index)]
        i = int(np.argmax(g.n_locs))
        g.merged_suspect[:] = False
        g.merged_suspect[i] = True
        forced[int(ring.index)] = i
    stub: Any = SimpleNamespace(include_suspect=True, tau0_nm=31.0, random_seed=0)
    up = UnrollParams(n_null=19)
    for h in mg.MERGED_HANDLINGS:
        mh = mg.apply_merged_handling(res, tol, h)
        assert mh.n_merged == 2
        if h == "split":
            assert mh.n_split + mh.n_split_failed == 2 and mh.n_split >= 1
            for ring, old in zip(mh.res.rings, res.rings):
                geo = ring_geometry(ring)                       # contour / order / centroids still consistent
                assert len(ring.clusters) == len(old.clusters) + (1 if int(ring.index) in forced and
                                                                  len(ring.clusters) != len(old.clusters) else 0)
                assert mh.pair_sigma_nm[int(ring.index)].size == len(ring.clusters) == geo.centroids_nm.shape[0]
                if len(ring.clusters) != len(old.clusters):
                    i = forced[int(ring.index)]
                    a, b = ring.clusters[i], ring.clusters[-1]
                    assert a.n_locs + b.n_locs == old.clusters[i].n_locs
                    assert set(a.loc_index.tolist()) | set(b.loc_index.tolist()) == set(old.clusters[i].loc_index.tolist())
            assert res.rings[0].clusters is not mh.res.rings[0].clusters     # the input rings are not modified
        elif h == "exclude":
            assert sorted(mh.exclude_clusters) == sorted(forced.items()) and mh.res is res
        else:
            assert mh.res is res and not mh.exclude_clusters
        arc = analyze_arc_columns(mh.res, stub, params=up, reference_curve="centroid_membrane",
                                  exclude_clusters=mh.exclude_clusters or None)
        assert arc.adjacent and all(0.0 < m.p_excess <= 1.0 for m in arc.adjacent)
        pp = analyze_arc_columns(mh.res, stub, params=up, reference_curve="centroid_membrane",
                                 exclude_clusters=mh.exclude_clusters or None, pair_sigma_nm=mh.pair_sigma_nm)
        assert pp.tolerance_rule == "per_pair"


def test_split_touches_the_arc_centres_only_the_unified_pool_still_excludes_merged(sim_rings: Any) -> None:
    # integration 3 (2026-10-29): D2's split (feat/validations-d2-e1-e5) and the one footprint pool of
    # fix/core-batch-tanda1 (tools.footprint_pool) side by side: the split never recomputes the pool, and the pool
    # leaves the merged-looking clusters out whatever the arc test does with them
    import dataclasses

    from tools import footprint_pool as fp
    res, _ = sim_rings
    inp = et.ToleranceInputs(exclude_merged_from_pool=True)
    tol = et.tolerances_from_rings(res, inp)
    forced = 0
    for ring in res.rings[:2]:
        g = tol.groups[int(ring.index)]
        g.merged_suspect[:] = False
        g.merged_suspect[int(np.argmax(g.n_locs))] = True
        forced += 1
    before = dataclasses.asdict(tol.pool)
    for h in mg.MERGED_HANDLINGS:
        mh = mg.apply_merged_handling(res, tol, h)
        assert dataclasses.asdict(tol.pool) == before, h               # the pool is the axon's, not the test's
        if h == "split":
            assert mh.n_split >= 1
    groups = list(tol.groups.values())
    s_t = np.concatenate([g.s_t_nm for g in groups])
    n = np.concatenate([g.n_locs for g in groups])
    ev = np.concatenate([g.n_events for g in groups])
    merged = np.concatenate([g.merged_suspect for g in groups])
    kw = dict(sigma_link_nm=inp.sigma_link_nm, c_label=inp.c_label, locs_per_event=inp.locs_per_event)
    every = fp.footprint_pool(s_t, n, ev, inp.sigma_loc_nm, **kw)
    excl = fp.footprint_pool(s_t, n, ev, inp.sigma_loc_nm, exclude=merged, **kw)
    usable_merged = int(np.count_nonzero(merged & np.isfinite(s_t) & (n > 1)))
    assert usable_merged == forced
    # out of the pool unless the event layer had already left them out as probable fusions (E2)
    assert every.n_used - usable_merged <= excl.n_used < every.n_used
    pool, _lam = et.pool_footprint(s_t, n, ev, merged, inp)
    assert pool.n_excluded_merged == usable_merged


def test_plugin_reports_the_handling(sim_rings: Any) -> None:
    from tools import multimethod as MM
    _, z = sim_rings
    assert "eclipse_merged_handling" in et.ECLIPSE_READOUTS and "eclipse_n_merged_split" in et.ECLIPSE_READOUTS
    assert {s.name for s in MM.readout_specs([et.ECLIPSE_ANALYSIS])} == set(et.ECLIPSE_READOUTS)
    assert et._merged_handlings() == mg.MERGED_HANDLINGS
    rin = dict(x=z["x"], y=z["y"], z=z["z"], frame=z["frame"], lp_lateral_nm=z["lp"], lpz_nm=z["lpz"],
               n_frames=int(z["n_frames"]))
    fake = SimpleNamespace(method="DBSCAN", params={"eps": 25.0, "min_samples": 10}, seconds=1.0)
    for h in (None, "exclude"):
        extras: Dict[str, Any] = {"eclipse_rings_input": rin}
        if h is not None:
            extras["eclipse_merged_handling"] = h
        ctx = MM.AnalysisContext(axon_id="a", runs=(), reference=None, n_replicates=19, extras=extras)
        out = et._rings_readouts(fake, ctx, et.ToleranceInputs(), rin)
        assert out["eclipse_merged_handling"] == (h or mg.MERGED_HANDLING_DEFAULT)
        assert out["eclipse_n_merged_split"] >= 0 and 0.0 < out["eclipse_scalar_p"] <= 1.0


def test_committed_d2_simulation_backs_the_default() -> None:
    d = json.load(open(os.path.join(HERE, "docs", "sims", "eclipse_tolerance", "merged_handling_sim.json"),
                       encoding="utf-8"))
    for key, c in d["cells"].items():
        if key.startswith("f=0.0"):
            for h in mg.MERGED_HANDLINGS:                    # the level: 5 % never below the Wilson interval
                assert c[f"{h}|scalar31"]["wilson95"][0] <= 0.05, (key, h)
    if mg.MERGED_HANDLING_DEFAULT == "split":
        c = d["cells"]["f=0.3|rate=0.25"]
        assert c["split|scalar31"]["rate"] > c["middle|scalar31"]["rate"]
        assert c["split_vs_middle|scalar31"]["ci95"][0] > 0.0


# ============================================================================================== E1
def test_committed_e1_run_keeps_the_cut_within_two_nm_at_c_label_20_and_26() -> None:
    d = json.load(open(os.path.join(HERE, "docs", "sims", "apparent_size", "elongation_cut_clabel.json"),
                       encoding="utf-8"))
    for c_label, limit in ((14, 3.0), (20, 2.0), (26, 2.0)):
        for labels in (2, 3):
            cell = d["cells"][f"c_label {c_label}, labels {labels}, rho 25 nm"]["pool"]["dbscan25, cut 2.5"]
            assert abs(cell["bias_vs_true_nm"]) <= limit, (c_label, labels, cell)
    low = d["cells"]["c_label 8, labels 2, rho 25 nm"]["pool"]["dbscan25, cut 2.5"]
    assert low["bias_vs_true_nm"] < -2.0                     # the failure the doc reports at c_label 8


# ============================================================================================== E5
def _ring_analysis(seed: int) -> Any:
    from tools import mps_pipeline_null as pn
    an, _ = pn.observe_synthetic(pn.synthetic_model(), seed, pn.PipelineSpec.for_method("DBSCAN"))
    return an


def test_statistic_is_one_function_for_observed_and_plugin() -> None:
    from tools import size_curvature_null as scn
    from tools import size_plugins as sp
    an = _ring_analysis(3)
    v = scn.nloc_kappa_r_statistic(an)
    assert v is not None and -1.0 <= v <= 1.0
    sa = sp.size_analysis(an, nena_nm=None)
    out = sp.axon_readouts(sa)
    assert out[scn.NULL_STATISTIC] == pytest.approx(v, abs=1e-12)
    assert out[scn.NULL_P_READOUT] is None                    # no null ran: never another p
    assert sp.describe_column(scn.NULL_P_READOUT) and sp.describe_column(scn.NULL_STATISTIC)
    tiny = SimpleNamespace(x_slab=np.r_[0.0, 1, 2], y_slab=np.zeros(3), labels=np.r_[0, 0, 1],
                           bad_report=SimpleNamespace(bad_labels=set()), discarded_labels=frozenset())
    assert scn.nloc_kappa_r_statistic(tiny) is None


def test_question_b_rides_the_same_pipeline_null() -> None:
    from tools import clustering_integration as ci
    from tools import mps_pipeline_null as pn
    from tools import multimethod as MM
    from tools import size_curvature_null as scn
    from tools import size_plugins as sp
    assert scn.NULL_STATISTIC in sp.register_with_pipeline_null(replace=True)
    assert scn.NULL_STATISTIC in pn.READOUTS and scn.NULL_STATISTIC in ci.ALL_NULL_READOUT_NAMES
    # integration 3 (2026-10-29): a v15 column, at the end of the table (no v13 column moves), with G1's companion
    assert scn.NULL_STATISTIC not in ci.NULL_READOUT_NAMES
    cols = ci.integration_columns()
    for c in (f"p_mc_{scn.NULL_STATISTIC}", f"null_median_{scn.NULL_STATISTIC}",
              f"p_mc_{scn.NULL_STATISTIC}_ignoring_failed"):
        assert c in ci.v15_columns() and c in cols and ci.describe_column(c), c
        assert c in ci.null_by_closure_columns(), c
    an = _ring_analysis(4)
    res = pn.run_pipeline_null(an, nena_nm=12.0, n_replicates=9, seed=4,
                               readouts=[scn.NULL_STATISTIC, "size_nloc_nn_spearman"])
    # one null, both size readouts on the same replicates
    assert set(res.p_values) == {scn.NULL_STATISTIC, "size_nloc_nn_spearman"}
    assert res.observed[scn.NULL_STATISTIC] == pytest.approx(scn.nloc_kappa_r_statistic(an), abs=1e-12)
    vals = [r[scn.NULL_STATISTIC] for r in res.replicate_readouts]
    assert sum(v is not None for v in vals) >= 8
    assert f"p_mc_{scn.NULL_STATISTIC}" in res.export_dict()
    assert f"p_mc_{scn.NULL_STATISTIC}_ignoring_failed" in res.export_dict()     # G1 c, integration 3
    # the size plug-in reads that p back from the null's results
    sp.register_with_multimethod(replace=True)
    spec = MM.readout_specs([sp.SIZE_ANALYSIS])
    p_spec = [s for s in spec if s.name == scn.NULL_P_READOUT][0]
    assert p_spec.kind == "p_value" and p_spec.tests == scn.NULL_STATISTIC
    run = SimpleNamespace(key="k", analysis=an, method="DBSCAN", params={})
    ctx = SimpleNamespace(extras={"pipeline_null_results": {"k": res}}, seed=0)
    assert sp.pipeline_null_p(run, ctx, scn.NULL_STATISTIC, 0.1) == pytest.approx(res.p_values[scn.NULL_STATISTIC])


def test_committed_e5_calibration_holds_the_level() -> None:
    d = json.load(open(os.path.join(HERE, "docs", "sims", "apparent_size", "curvature_null_calibration.json"),
                       encoding="utf-8"))
    r = d["readouts"]["size_nloc_kappa_r_spearman"]
    assert d["n_axons"] >= 200 and r["tested"] >= 200
    assert r["wilson95"][0] <= 0.05 <= r["wilson95"][1]
