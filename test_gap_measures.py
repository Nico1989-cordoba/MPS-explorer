# -*- coding: utf-8 -*-
"""
tools/mps_gap_measures.py and tools/mps_gap_batch.py on SYNTHETIC data only (rule R8): each measure on hand-built
gap sequences and on known arrangements (regular lattice, Poisson, hard core, M8 iid gamma gaps) with their
identities; invariance to the origin and the direction; the exact sequential-inhibition sampler; the reading rule's
decision table (rho1 threshold, minimum-below-contact flag, not_testable); p never 0; d_m recovering the true hard
core; a small centre-level simulation of the two-stage rule's false-label rates consistent with the regularity
audit (exclusion is not labelled lattice; the level near 5 %); determinism; the pipeline-null readouts and the
multi-method plug-in; the batch helpers (Holm, per-animal aggregation, P-R83 criterion) and the calibre model's
refusal with < 3 animals and the coverage of its intercept interval; the committed calibration tables.

The pipeline-level confusion matrix (hours) is in docs/sims/gap_measures/pipeline_confusion.py with its output
committed; the last tests check the committed tables.

Run:  py -3 -m pytest test_gap_measures.py -q
"""
from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tools import mps_gap_batch as gb  # noqa: E402
from tools import mps_gap_measures as gm  # noqa: E402
from tools import mps_pipeline_null as pn  # noqa: E402

SIMS = os.path.join(REPO_ROOT, "docs", "sims", "gap_measures")
K, A = 50, 250.0
P = K * A


def _ns(name, obs, mean, p, lo=None, hi=None, n=199, tail="two-sided"):
    return gm.NullSummary(name, obs, mean, 0.1, lo, hi, p, n, tail)


# ------------------------------------------------------------------ measures on hand-built gaps

def test_hand_built_gap_sequences() -> None:
    # alternating gaps 100/300: rho1 = -1 exactly; CV = 0.5; excess = (0/200) -> sd/mean of {0,200} = 1
    g = np.array([100.0, 300.0] * 10)
    s = np.concatenate([[0.0], np.cumsum(g)[:-1]])
    length = float(g.sum())
    assert np.allclose(np.sort(gm.arc_gaps(s, length)), np.sort(g))
    assert gm.gap_cv(g) == pytest.approx(0.5)
    assert gm.gap_rho1(g) == pytest.approx(-1.0)
    assert gm.excess_gap_cv(g) == pytest.approx(1.0)
    # a perfect lattice: CV 0, rho1 undefined (None), S(n) = K exactly at m = n, S_low = 0
    s = np.arange(K) * A + 17.0
    m = gm.gap_measures(s, P)
    assert m.cv == pytest.approx(0.0, abs=1e-12) and m.rho1 is None
    assert m.bragg_m == K and m.bragg_s == pytest.approx(K) and m.bragg_spacing_nm == pytest.approx(A)
    assert m.s_low == pytest.approx(0.0, abs=1e-9)
    assert gm.gap_measures(s[:2], P).cv is None                    # K < 3: nothing computed


def test_identities_on_known_arrangements() -> None:
    rng = np.random.default_rng(11)

    def mean(kind, f, n=300, **kw):
        return float(np.mean([getattr(gm.gap_measures(gm.planted_arrangement(kind, rng, K, P, **kw), P), f)
                              for _ in range(n)]))

    cv2 = float(np.mean([gm.gap_measures(gm.planted_arrangement("poisson", rng, K, P), P).cv ** 2
                         for _ in range(300)]))
    assert cv2 == pytest.approx((K - 1) / (K + 1), abs=0.03)                    # Moran 1947
    assert mean("poisson", "r_arc") == pytest.approx(math.log(2) / 2, abs=0.02)
    assert mean("lattice", "rho1", jitter_nm=30.0) == pytest.approx(-0.5, abs=0.03)
    q = (2 * math.pi * 30.0 / A) ** 2
    assert mean("lattice", "bragg_s", jitter_nm=30.0) == pytest.approx(K * math.exp(-q) + 1 - math.exp(-q), rel=0.05)
    for kind, kw in (("hard_core", {"d_nm": 120.0}), ("m8_gamma", {"sigma_rel": 0.25})):
        assert abs(mean(kind, "rho1", **kw)) < 0.06                                # renewal: rho1 ~ 0
    assert mean("hard_core", "excess_cv", d_nm=80.0) > 0.9                        # exclusion: ~1
    assert mean("lattice", "excess_cv", jitter_nm=30.0) < 0.6                     # bell-shaped gaps
    # S_low ~ CV^2 for a renewal process (hard core): S(q -> 0) = CV^2 [Cox & Lewis 1966]
    rows = [gm.gap_measures(gm.planted_arrangement("hard_core", rng, K, P, d_nm=120.0), P) for _ in range(300)]
    assert np.mean([r.s_low for r in rows]) == pytest.approx(np.mean([r.cv ** 2 for r in rows]), abs=0.03)


def test_invariant_to_origin_and_direction() -> None:
    rng = np.random.default_rng(3)
    s = gm.planted_arrangement("lattice", rng, K, P, jitter_nm=40.0, occupancy=0.9)
    a = gm.gap_measures(s, P)
    for t in (gm.gap_measures(np.mod(s + 1234.5, P), P), gm.gap_measures(np.mod(-s, P), P)):
        for f in ("cv", "rho1", "excess_cv", "bragg_s", "s_low", "r_arc"):
            assert getattr(t, f) == pytest.approx(getattr(a, f), rel=1e-9, abs=1e-12)
        assert t.bragg_m == a.bragg_m


def test_exact_sequential_inhibition_matches_the_rejection_sampler() -> None:
    from tools.mps_simulate_axon import _hard_core_arcs
    rng = np.random.default_rng(5)
    pts, jam = gm.hard_core_arcs(rng, K, P, 150.0, 400)
    assert not jam.any()
    g = np.diff(np.concatenate([pts, pts[:, :1] + P], axis=1), axis=1)
    assert g.min() >= 150.0 - 1e-9
    cv_free = float(np.mean(gm._batch_measures(pts, P)["cv"]))
    cv_rej = float(np.mean([gm.gap_cv(gm.arc_gaps(_hard_core_arcs(rng, K, P, 150.0, np.zeros(0)), P))
                            for _ in range(400)]))
    assert cv_free == pytest.approx(cv_rej, abs=0.015)
    # batched measures equal the scalar ones
    bm = gm._batch_measures(pts[:5], P)
    for i in range(5):
        one = gm.gap_measures(pts[i], P)
        assert bm["cv"][i] == pytest.approx(one.cv) and bm["rho1"][i] == pytest.approx(one.rho1)
        assert bm["bragg_s"][i] == pytest.approx(one.bragg_s)
    _p, jam = gm.hard_core_arcs(rng, K, P, 1.1 * A * 2, 3)            # cannot pack: every pattern jams
    assert jam.all()


def test_pair_correlation_is_one_for_poisson_and_zero_below_the_hard_core() -> None:
    rng = np.random.default_rng(8)
    gs = [gm.pair_correlation_arc(gm.planted_arrangement("poisson", rng, K, P), P)[1] for _ in range(200)]
    assert float(np.mean(gs)) == pytest.approx(1.0, abs=0.05)
    r, g = gm.pair_correlation_arc(gm.planted_arrangement("hard_core", rng, K, P, d_nm=100.0), P)
    assert np.all(g[r < 100.0 - gm.G_BIN_NM] == 0.0)


# ------------------------------------------------------------------ Monte Carlo p and the reading rule

def test_p_is_never_zero() -> None:
    null = list(np.linspace(0.5, 1.0, 199))
    assert gm.summarise("x", 0.1, null, "lower").p == pytest.approx(1 / 200)
    assert gm.summarise("x", 0.1, null, "two-sided").p == pytest.approx(2 / 200)
    assert gm.summarise("x", 5.0, null, "upper").p == pytest.approx(1 / 200)
    assert gm.summarise("x", 0.1, [], "lower").p is None


def test_reading_rule_decision_table() -> None:
    cv_ok = _ns("gap_cv", 0.4, 0.6, 0.01)
    mt = SimpleNamespace(kind="centres_approx", d_m_nm=140.0, capped=False, n_replicates=199)
    R = gm.regularity_reading
    assert R(k=7, cv=cv_ok).level == gm.NOT_TESTABLE
    assert R(k=50, cv=_ns("gap_cv", 0.55, 0.6, 0.3)).level == gm.RANDOM_WITH_CONTACT
    assert R(k=50, cv=_ns("gap_cv", 0.9, 0.6, 0.01)).level == gm.LESS_REGULAR
    assert R(k=50, cv=cv_ok).level == gm.KIND_NOT_IDENTIFIED                    # stage 2 did not run
    assert R(k=50, cv=cv_ok).stage2_kind == "not_run"

    def two(rho, p_rho, p_bragg):
        return R(k=50, cv=cv_ok, rho1_matched=_ns("gap_rho1", rho, 0.0, p_rho, tail="lower"),
                 bragg_matched=_ns("bragg_s", 20.0, 5.0, p_bragg, 3.0, 8.0, tail="upper"), matched=mt,
                 bragg_m=50, bragg_spacing_nm=250.0, excess_cv=0.9)

    assert two(-0.45, 0.01, 0.01).level == gm.LATTICE
    assert two(-0.45, 0.01, 0.40).level == gm.LATTICE_RHO1_ONLY
    assert two(-0.05, 0.40, 0.01).level == gm.PERIODIC_COMPONENT
    assert two(-0.05, 0.40, 0.40).level == gm.KIND_NOT_IDENTIFIED
    # rho1 significant but above -0.25 does not count (written threshold, regularity audit)
    assert two(-0.20, 0.01, 0.01).level == gm.PERIODIC_COMPONENT
    assert two(-0.45, 0.01, 0.01).stage2_kind == "centres_approx"
    assert "approx., centres" in two(-0.45, 0.01, 0.01).sentence
    # the safeguard: minimum below contact is said first, the verdict stays
    rd = R(k=50, cv=cv_ok, min_nn_nm=30.0, d_contact_nm=55.0)
    assert rd.flag_min_below_contact and rd.sentence.startswith("Minimum 1NN")
    assert not R(k=50, cv=cv_ok, min_nn_nm=80.0, d_contact_nm=55.0).flag_min_below_contact
    for lv in (gm.LATTICE, gm.PERIODIC_COMPONENT, gm.KIND_NOT_IDENTIFIED):
        assert lv in gm.READING_LEVELS
    words = " ".join(gm.READING_LEVELS)
    assert "paracrystal" not in words and "vacanc" not in words                  # P-R81 b
    assert "vacanc" not in two(-0.05, 0.40, 0.01).sentence


def test_not_testable_without_running_a_null() -> None:
    an = SimpleNamespace(n_clusters_kept=5, centroids=np.random.default_rng(1).uniform(0, 1000, (5, 2)),
                         perimeter=None)
    res = gm.analyze_gap_regularity(an)
    assert res.reading.level == gm.NOT_TESTABLE and res.null_result is None
    assert res.export_columns()["regularity_reading"] == gm.NOT_TESTABLE


# ------------------------------------------------------------------ stage 2 at the centres level

def test_matched_hard_core_recovers_the_true_one() -> None:
    for d in (120.0, 150.0):
        rng = np.random.default_rng(int(d))
        dms = [gm.matched_null_centres(gm.planted_arrangement("hard_core", rng, K, P, d_nm=d), P, n_replicates=49,
                                       seed=i).d_m_nm for i in range(8)]
        assert float(np.mean(dms)) == pytest.approx(d, rel=0.15)


def test_matched_null_is_deterministic_and_caps() -> None:
    rng = np.random.default_rng(2)
    s = gm.planted_arrangement("lattice", rng, K, P, jitter_nm=20.0)
    a = gm.matched_null_centres(s, P, n_replicates=49, seed=4)
    b = gm.matched_null_centres(s, P, n_replicates=49, seed=4)
    assert a.d_m_nm == b.d_m_nm and a.rho1.p == b.rho1.p and a.bragg_s.p == b.bragg_s.p
    assert a.capped and a.d_m_nm == pytest.approx(gm.MATCHED_CAP_FRACTION * A)   # a lattice beats any exclusion
    assert a.notes and "capped" in a.notes[-1]


def _centres_reading(s, seed, b=99):
    obs = gm.gap_measures(s, P)
    pts, jam = gm.hard_core_arcs(np.random.default_rng([seed, 1]), K, P, 50.0, b)
    cv1 = gm.summarise("gap_cv", obs.cv, list(gm._batch_measures(pts[~jam], P)["cv"]), "two-sided")
    mt = None
    if cv1.p <= 0.05 and obs.cv < cv1.mean:
        mt = gm.matched_null_centres(s, P, n_replicates=b, seed=seed)
    return gm.regularity_reading(k=K, cv=cv1, rho1_matched=None if mt is None else mt.rho1,
                                 bragg_matched=None if mt is None else mt.bragg_s, matched=mt).level


def test_two_stage_false_label_rates_small_simulation() -> None:
    # centre level, consistent with the regularity audit's matched-null check (seed 1011): exclusion d = 150 is
    # never 'lattice' and rarely 'periodic_component'; a full lattice is 'lattice'; with 15 % empty sites
    # 'periodic_component' or 'lattice'; the contact null itself is 'random_with_contact' (level ~5 %).
    rng = np.random.default_rng(20261022)
    lv = [_centres_reading(gm.planted_arrangement("hard_core", rng, K, P, d_nm=150.0), 100 + i) for i in range(30)]
    assert lv.count(gm.LATTICE) <= 1 and lv.count(gm.PERIODIC_COMPONENT) <= 6
    lv = [_centres_reading(gm.planted_arrangement("lattice", rng, K, P, jitter_nm=30.0), 200 + i) for i in range(20)]
    assert lv.count(gm.LATTICE) >= 16
    lv = [_centres_reading(gm.planted_arrangement("lattice", rng, K, P, jitter_nm=30.0, occupancy=0.85), 300 + i)
          for i in range(20)]
    assert lv.count(gm.LATTICE) + lv.count(gm.PERIODIC_COMPONENT) >= 18
    lv = [_centres_reading(gm.planted_arrangement("hard_core", rng, K, P, d_nm=50.0), 400 + i) for i in range(40)]
    assert lv.count(gm.RANDOM_WITH_CONTACT) >= 34
    lv = [_centres_reading(gm.planted_arrangement("m8_gamma", rng, K, P, sigma_rel=0.25), 500 + i) for i in range(20)]
    assert lv.count(gm.LATTICE) <= 1                                              # M8: no lattice label


# ------------------------------------------------------------------ elongation (merged-cluster flag)

def _fake_axon(merged: bool):
    rng = np.random.default_rng(9)
    R0 = 2000.0
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    cen = np.column_stack([R0 * np.cos(theta), R0 * np.sin(theta)])
    xs, ys, labs = [], [], []
    for i, c in enumerate(cen):
        t = np.array([-np.sin(theta[i]), np.cos(theta[i])])
        pts = c + rng.normal(0.0, 15.0, (60, 2))
        if merged and i == 0:
            pts = np.vstack([c + 35 * t + rng.normal(0, 15.0, (30, 2)), c - 35 * t + rng.normal(0, 15.0, (30, 2))])
        xs.append(pts[:, 0]); ys.append(pts[:, 1]); labs.append(np.full(len(pts), i))
    an = SimpleNamespace(centroids=cen, x_slab=np.concatenate(xs), y_slab=np.concatenate(ys),
                         labels=np.concatenate(labs), bad_report=SimpleNamespace(bad_labels=[]))
    mem = np.column_stack([R0 * np.cos(np.linspace(0, 2 * np.pi, 720, endpoint=False)),
                           R0 * np.sin(np.linspace(0, 2 * np.pi, 720, endpoint=False))])
    return an, mem


def test_a_merged_pair_is_elongated_and_flagged() -> None:
    an, mem = _fake_axon(True)
    e = gm.cluster_elongations(an, mem)
    assert e.size == 24 and e[0] > 2.0 and float(np.median(e[1:])) < 1.3
    flag = gm.elongation_flag(an, mem, [1.6] * 50)
    assert flag.n_elongated >= 1 and flag.threshold == pytest.approx(1.6)
    assert flag.expected_under_null == pytest.approx(0.05 * 24)


# ------------------------------------------------------------------ the pipeline null and the plug-in

def test_gap_readouts_are_registered_with_the_pipeline_null() -> None:
    for name in ("gap_cv", "gap_rho1", "gap_excess_cv", "bragg_s", "elongation_p95"):
        assert name in pn.READOUTS
    assert pn.READOUTS["gap_cv"].two_sided and not pn.READOUTS["bragg_s"].two_sided


@pytest.fixture(scope="module")
def observed_small():
    model = pn.synthetic_model((1200.0, 800.0))
    an, _ax = pn.observe_synthetic(model, 4)
    return an


def test_pipeline_level_run_is_deterministic(observed_small) -> None:
    an = observed_small
    a = gm.analyze_gap_regularity(an, n_replicates=19, stage2="none", nena_nm=12.0, seed=2)
    b = gm.analyze_gap_regularity(an, n_replicates=19, stage2="none", nena_nm=12.0, seed=2)
    assert a.export_columns() == b.export_columns()
    assert a.stage1["gap_cv"].n == a.null_result.n_accepted
    assert a.reading.level in gm.READING_LEVELS and a.d_contact_nm == pytest.approx(a.null_result.model.hard_core_nm)
    m1 = gm.matched_null_pipeline(an, a.null_result, n_replicates=4, steps=1, n_bisect=3, seed=1, nena_nm=12.0)
    m2 = gm.matched_null_pipeline(an, a.null_result, n_replicates=4, steps=1, n_bisect=3, seed=1, nena_nm=12.0)
    assert m1.kind == "pipeline" and m1.d_m_nm == m2.d_m_nm and m1.rho1.p == m2.rho1.p
    assert len(m1.bisection) == 2 and m1.n_replicates <= 4
    cols = a.export_columns()
    for c in ("gap_cv", "gap_cv_null_mean", "gap_cv_null_lo", "gap_cv_null_hi", "gap_cv_p_mc", "gap_rho1",
              "gap_rho1_p_mc_matched", "gap_excess_cv_p_mc", "bragg_s", "bragg_m", "bragg_spacing_nm",
              "bragg_p_mc_matched", "s_low", "gap_cv_polygon", "regularity_reading",
              "regularity_flag_min_below_contact", "pipeline_null_B", "pipeline_null_d_contact_nm",
              "matched_null_d_nm", "matched_null_B", "n_clusters_elongated", "elongation_threshold",
              "n_clusters_elongated_null_expected", "nn_regularity_null_kind"):
        assert c in cols


def test_multimethod_plugin(observed_small) -> None:
    from tools import multimethod as mm
    assert gm.MULTIMETHOD_ANALYSIS in mm.registered_analyses()
    specs = {s.name: s for s in mm.readout_specs([gm.MULTIMETHOD_ANALYSIS])}
    for name, s in specs.items():
        assert name.startswith("gap_") or name.startswith("n_gap_")
        if s.kind == "p_value":
            assert name.endswith("_p") and s.tests in specs
        if s.kind == "verdict":
            assert name.endswith("_verdict") and s.levels
    assert tuple(specs["gap_regularity_verdict"].levels) == gm.READING_LEVELS
    an = observed_small
    run = mm.MethodRun(method=an.cluster_method, params=dict(an.cluster_params), key="ref", axon_id="t",
                       analysis=an)
    cheap = mm.apply_analyses([run], [gm.MULTIMETHOD_ANALYSIS], extras={"gap_null": "none"})
    v = {r["readout"]: r["value"] for r in cheap}
    assert v["gap_cv"] is not None and v["gap_regularity_verdict"] is None and not any(r["error"] for r in cheap)
    full = mm.apply_analyses([run], [gm.MULTIMETHOD_ANALYSIS], n_replicates=19, seed=1, extras={"nena_nm": 12.0})
    assert not any(r["error"] for r in full), [r["error"] for r in full if r["error"]]
    v = {r["readout"]: r["value"] for r in full}
    assert v["gap_regularity_verdict"] in gm.READING_LEVELS and 0 < v["gap_cv_p"] <= 1


# ------------------------------------------------------------------ batch helpers

def test_holm() -> None:
    assert gb.holm([0.01, 0.04, 0.03, None]) == pytest.approx([0.03, 0.06, 0.06, None])
    assert gb.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_per_animal_aggregation_and_lattice_criterion() -> None:
    rows = []
    for animal, zr, zb, lat in (("A", -3.0, 4.0, True), ("B", -2.5, 3.0, False), ("C", -0.5, 1.0, False)):
        for j in range(3):
            rows.append({"animal": animal, "group": "wt", "gap_cv_z": -2.0 + 0.1 * j, "gap_excess_cv_z": 0.0 + j,
                         "gap_rho1_z_matched": zr + 0.1 * j, "bragg_s_z_matched": zb - 0.1 * j,
                         "regularity_reading": "lattice" if lat and j == 0 else "periodic_component",
                         "bragg_spacing_nm": 240.0 + j})
    agg = gb.aggregate_by_animal(rows)
    assert [a["animal"] for a in agg] == ["A", "B", "C"] and agg[0]["n_axons"] == 3
    assert agg[0]["mean_gap_cv_z"] == pytest.approx(-1.9)
    assert agg[0]["p_holm_gap_cv_z"] >= agg[0]["p_gap_cv_z"]
    ev = gb.lattice_evidence(rows)
    assert ev.evidence and ev.animals_passing == ["A", "B"] and ev.n_axons_lattice == 1
    no = gb.lattice_evidence([r for r in rows if r["animal"] != "B"])                # one animal only
    assert not no.evidence and "Never write" in no.sentence
    tab = gb.readings_table(rows, gm.READING_LEVELS)
    assert tab[0]["n_lattice"] == 1 and tab[0]["median_spacing_periodic_nm"] == pytest.approx(241.0)
    cmp2 = gb.compare_groups(rows + [dict(r, group="ko", animal=r["animal"] + "2") for r in rows[:6]])
    assert cmp2["descriptive_only"] and all(m["p"] is None for m in cmp2["measures"].values())


def test_calibre_model_refuses_below_three_animals_and_covers_the_intercept() -> None:
    with pytest.raises(gb.CalibreRefused):
        gb.calibre_model([10, 20, 30, 40], [3, 5, 7, 9], ["a", "a", "b", "b"])
    rng = np.random.default_rng(6)
    hits, n_rep = 0, 40
    for _ in range(n_rep):
        n, p, a = [], [], []
        for animal in range(6):
            u = rng.normal(0.0, 3.0)                     # animal intercept (sd 3 clusters) [ad hoc]
            for _j in range(8):
                per = rng.uniform(5.0, 25.0)
                n.append(4.0 * per + u + rng.normal(0.0, 3.0))
                p.append(per)
                a.append(str(animal))
        res = gb.calibre_model(n, p, a)
        lo, hi = res.observed.intercept_ci
        hits += int(lo <= 0.0 <= hi)
    assert hits / n_rep >= 0.85
    assert "intercept" in res.sentence and res.spacing_by_bin


# ------------------------------------------------------------------ committed calibration tables

def test_committed_centre_level_calibration() -> None:
    with open(os.path.join(SIMS, "centres_calibration.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    cells = {c["truth"]: c for c in d["part2"]["cells"]}
    n = d["n_rep_part2"]
    lab = {t: c["labels"] for t, c in cells.items()}
    assert lab["contact_d50"]["random_with_contact"] / n >= 0.85
    for t in ("excl_d120", "excl_d150", "excl_d170"):
        assert lab[t]["lattice"] / n <= 0.05 and lab[t]["periodic_component"] / n <= 0.15
    assert lab["lattice_s30"]["lattice"] / n >= 0.80
    assert (lab["lattice_s30_vac15"]["lattice"] + lab["lattice_s30_vac15"]["periodic_component"]) / n >= 0.90
    for name, c in d["part1"]["checks"].items():
        if name.startswith("lattice_rho1"):
            assert c["sim"] == pytest.approx(-0.5, abs=0.03)


def test_committed_pipeline_level_calibration() -> None:
    # the false-label criteria of the regularity plan (PR 2) through the pipeline; the power criteria (lattice >= 80 %)
    # are NOT met through the pipeline and are documented as a deviation in docs/regularity.md sec. 4.2
    for name in ("pipeline_confusion_roi.json", "pipeline_confusion.json"):
        with open(os.path.join(SIMS, name), encoding="utf-8") as fh:
            d = json.load(fh)
        cells = {c["truth"]: c for c in d["cells"]}
        lab = {t: c["labels_pipeline"] for t, c in cells.items()}
        n = {t: c["n"] for t, c in cells.items()}
        assert n["excl150"] >= 12 and n["contact"] >= 12
        assert lab["excl150"]["lattice"] / n["excl150"] <= 0.05
        assert lab["excl150"]["periodic_component"] / n["excl150"] <= 0.15
        assert lab["m8"]["lattice"] == 0
        assert lab["contact"]["random_with_contact"] / n["contact"] >= 0.85
        assert lab["lattice"]["lattice"] + lab["lattice"]["periodic_component"] >= 0.75 * n["lattice"]
