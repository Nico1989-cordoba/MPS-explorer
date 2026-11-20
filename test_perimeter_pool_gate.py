"""P10b: the automatic contour's assessment through the orchestration, the v19 columns and the quality gate of the
pooled analyses (decisions D9 and D12).

Simulated data only. The gate is checked on results whose assessment is set by hand (``extras``), so that each
rule is exercised on its own; one test runs the real automatic contour end to end. Run:
``python -m pytest test_perimeter_pool_gate.py -q``.
"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import mps_perimeter_analysis as an  # noqa: E402
from tools import mps_perimeter_bc as bc  # noqa: E402
from tools import mps_perimeter_pool as pool  # noqa: E402
from tools import mps_perimeter_types as pdt  # noqa: E402
from tools import perimeter_export as pe  # noqa: E402
from tools.mps_auto_contour import CURVE_KINDS, LOW_QUALITY_PREFIX  # noqa: E402
from tools.mps_identity import AxonIdentity  # noqa: E402
from tools.mps_perimeter_sim import simulate_scenario  # noqa: E402

FAST = replace(pdt.PerimeterParams(), n_null=199)


def _true(sim):
    def fn(x, y, **kw):
        path = sim.true_path
        nan = float("nan")
        return pdt.AutoContour(path=path, length_nm=float(path.length_nm), method="density_nodes", knot_nm=400.0,
                               residual_sd_nm=nan, n_self_crossings=0, nodes_nm=np.asarray(path.points_nm)[::100],
                               polar_distance_nm=nan, star_shaped=True, status=pdt.Status.OK)
    return fn


def _run(seed=1, contour_fn=None, params=FAST, i=0):
    sim = simulate_scenario("a_clumps", seed=seed, n_locs=4000)
    ident = AxonIdentity(genotype="WT", animal="", roi_name="ROI 1", axon_name=f"Axon {i}")
    return analyze(sim, ident, contour_fn if contour_fn is not None else _true(sim), params, i)


def analyze(sim, ident, fn, params, i):
    return an.analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={}, info=[], identity=ident,
                                source_path=f"C:/d/26.1.15_ROI1_axon{i}.hdf5", params=params, contour_fn=fn)


@pytest.fixture(scope="module")
def base():
    r = _run()
    assert r.status == pdt.Status.OK, r.failure
    return r


def _with(r, i, **assessment):
    a = dict(r.extras["contour_assessment"])
    a.update(assessment)
    ident = replace(r.identity, axon_name=f"Axon {i}")
    return replace(r, extras={**r.extras, "contour_assessment": a}, axon_id=f"ax{i:02d}", identity=ident,
                   trace=replace(r.trace, label=f"26.1.{i}", axon_number=str(i)))


# ----------------------------------------------------------------------------------------------------
# The assessment travels with the result and into the v19 block
# ----------------------------------------------------------------------------------------------------
def test_assessment_of_an_injected_contour_and_the_v19_block(base):
    a = base.extras["contour_assessment"]
    assert a["contour_kind"] == an.EXTERNAL_KIND
    assert set(a) >= {"contour_reliable", "contour_frac100", "contour_slice_rms_minus_nm", "contour_slice_misfit",
                      "contour_gate_reasons"}
    b = pe.perimeter_block(base)
    assert list(b) == list(pe.ALL_PERIMETER_COLUMNS)
    assert b["trace_roi"] == "1" and b["pd_contour_kind"] == "external" and b["pd_curve_stratum"] == "external"
    assert b["pd_contour_frac100"] is not None and 0.0 <= b["pd_contour_frac100"] <= 1.0
    assert b["pd_a_reading"] in ("attractor", "inhibition zone", "regular", "attractor + regular",
                                 "inhibition zone + regular", "no structure detectable", "not computable")
    assert b["pd_contour_trim_status"] is None and b["pd_contour_n_excluded"] == 0     # D8: off by default
    # v17's unsupported fraction (bins) and P3's (length without density) are two columns, not one
    assert "pd_contour_unsupported_fraction" in b and "pd_contour_support_unsupported_fraction" in b
    assert not any(isinstance(v, float) and v != v for v in b.values())


def test_not_run_block_holds_v19_empty():
    b = pe.perimeter_block(None)
    assert list(b) == list(pe.ALL_PERIMETER_COLUMNS)
    assert all(b[c] is None for c in pe.PERIMETER_V19_COLUMNS + pe.PERIMETER_V20_COLUMNS + pe.PERIMETER_V21_COLUMNS
               + pe.PERIMETER_V22_COLUMNS)


def test_real_automatic_contour_end_to_end():
    sim = simulate_scenario("a_flat", seed=3, n_locs=6000)
    r = an.analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={}, info=[], identity=AxonIdentity(),
                             source_path="C:/d/26.1.15_ROI2_axon1.hdf5", params=FAST)
    a = r.extras.get("contour_assessment")
    assert a is not None and a["contour_kind"] in CURVE_KINDS
    row = pool.perimeter_row(r)
    if r.status == pdt.Status.OK:
        # the slice labels reached the gate: the misfit between slices was measured
        assert row["pd_contour_slice_rms_minus_nm"] is not None
        assert row["pd_pool_included"] == (not a["contour_gate_reasons"])
    else:
        assert row["pd_pool_included"] is False and row["pd_pool_exclusion_reason"]


def test_trim_flag_when_switched_on():
    r = _run(params=replace(FAST, contour_trim_interior_nm=250.0))
    row = pool.perimeter_row(r)
    assert row["pd_contour_trim_status"] and r.extras["contour_trim_nm"] == 250.0
    assert row["pd_contour_n_excluded"] == r.extras["contour_n_locs_trimmed"]


# ----------------------------------------------------------------------------------------------------
# The gate (D9, D12)
# ----------------------------------------------------------------------------------------------------
def test_gate_rules(base):
    ok = _with(base, 1, contour_gate_reasons="", contour_slice_misfit=False, contour_failure_reason="",
               contour_kind="primary")
    misfit = _with(base, 2, contour_gate_reasons="", contour_slice_misfit=True, contour_reliable=False,
                   contour_kind="stiff fallback")
    bad = _with(base, 3, contour_gate_reasons="only 44% of the localizations within +-100 nm",
                contour_slice_misfit=True, contour_kind="stiff fallback")
    failed = replace(_with(base, 4), status=pdt.Status.CONTOUR_FAILED, failure="contour failed: inward notch")
    g = [pool.pool_gate(r) for r in (ok, misfit, bad, failed)]
    assert [x["pd_pool_included"] for x in g] == [True, True, False, False]           # D12: misfit stays in
    assert [x["pd_pool_included_without_misfit"] for x in g] == [True, False, False, False]
    assert g[1]["pd_contour_slice_misfit"] is True and g[1]["pd_pool_exclusion_reason"] is None
    assert "contour not reliable: only 44%" in g[2]["pd_pool_exclusion_reason"]
    assert g[3]["pd_pool_exclusion_reason"].startswith("analysis contour failed")
    assert [x["pd_curve_stratum"] for x in g[:3]] == ["primary", "stiff", "stiff"]
    nothing = replace(_with(base, 5), extras={})
    assert pool.pool_gate(nothing)["pd_pool_included"] is False                        # cannot vouch for it

    results = [ok, misfit, bad, failed]
    summary = {s["scope"]: s for s in pool.gate_summary_rows(results)}
    assert list(summary) == [pool.SCOPE_MAIN, pool.SCOPE_NO_MISFIT, pool.SCOPE_C_MAIN,
                             pool.SCOPE_CURVE_PREFIX + "primary", pool.SCOPE_CURVE_PREFIX + "stiff"]
    # the main reading of C: the injected base contour's slices coincide in the two included axons
    c = summary[pool.SCOPE_C_MAIN]
    assert c["n_included"] + c["n_excluded_slices_not_coinciding"] == 2
    m, n = summary[pool.SCOPE_MAIN], summary[pool.SCOPE_NO_MISFIT]
    assert (m["n_included"], m["n_excluded_analysis_failed"], m["n_excluded_contour_not_reliable"],
            m["n_included_with_slice_misfit"]) == (2, 1, 1, 1)
    assert (n["n_included"], n["n_excluded_slice_misfit"]) == (1, 1)
    gate = pool.gate_rows(results)
    assert [r["pd_pool_included"] for r in gate] == [True, True, False, False]

    # pooled A takes the gate: main = ok + misfit, without misfit = ok alone, strata within the main set
    scoped = pool.pooled_a_scopes(results, n_null=99)
    assert {k: next(iter(v.values())).n_axons for k, v in scoped.items()} == {
        pool.SCOPE_MAIN: 2, pool.SCOPE_NO_MISFIT: 1, pool.SCOPE_CURVE_PREFIX + "primary": 1,
        pool.SCOPE_CURVE_PREFIX + "stiff": 1}
    rows = pool.pooled_rows_scoped(scoped)
    assert {r["scope"] for r in rows} == set(scoped)

    # per animal and between genotypes: the same gate on the flat rows
    flat = pool.perimeter_rows(results)
    by = pool.by_animal_rows_scoped(flat, pool.row_scopes(flat))
    main = [r for r in by if r["scope"] == pool.SCOPE_MAIN]
    assert sum(r["n_axons"] for r in main) == 2
    cmp = pool.compare_genotypes(flat, pool.SCOPE_NO_MISFIT)
    assert cmp["scope"] == pool.SCOPE_NO_MISFIT and any("quality gate" in n for n in cmp["notes"])


def test_gate_warning_follows_d12():
    only_misfit = {"contour_gate_reasons": "", "contour_slice_misfit": True,
                   "contour_quality_reasons": "the three z slices do not fall on one another: ..."}
    assert an._gate_warning(only_misfit).startswith(an.SLICE_MISFIT_PREFIX)
    other = {"contour_gate_reasons": "radial robust sd 151 nm (limit 130)", "contour_slice_misfit": True}
    assert an._gate_warning(other).startswith(LOW_QUALITY_PREFIX)
    assert an._gate_warning({"contour_gate_reasons": "", "contour_slice_misfit": False}) is None


def test_axon_a_reading_vocabulary(base):
    lab = pool.axon_a_reading(base)
    assert lab is not None
    assert pool.axon_a_reading(replace(base, status=pdt.Status.CONTOUR_FAILED)) is None


# ----------------------------------------------------------------------------------------------------
# R3: the per-axon reading of C says when it cannot be read; by_animal gives no p with one animal
# ----------------------------------------------------------------------------------------------------
def test_c_reading_suffixes():
    ok = {"pd_pool_exclusion_reason": None, "pd_pool_included": True, "pd_c_slices_coincide": True}
    apart = {"pd_pool_exclusion_reason": None, "pd_pool_included": True, "pd_c_slices_coincide": False}
    unmeasured = {"pd_pool_exclusion_reason": None, "pd_pool_included": True, "pd_c_slices_coincide": None}
    bad = {"pd_pool_exclusion_reason": "contour not reliable: radial robust sd 151 nm", "pd_pool_included": False,
           "pd_c_slices_coincide": False}
    rejected = {"pd_pool_exclusion_reason": "contour rejected: inward notch", "pd_pool_included": False,
                "pd_c_slices_coincide": True}
    unassessed = {"pd_pool_exclusion_reason": "contour not assessed by the quality gate", "pd_pool_included": False,
                  "pd_c_slices_coincide": None}
    reg = "registry between slices"
    assert pool.axon_c_reading(reg, ok) == reg                                         # a bare reading only here
    assert pool.axon_c_reading(reg, apart) == reg + pool.C_SUFFIX_SLICES
    assert pool.axon_c_reading(reg, unmeasured) == reg + pool.C_SUFFIX_SLICES          # not measurable: not coinciding
    assert pool.axon_c_reading(reg, bad) == reg + pool.C_SUFFIX_NOT_RELIABLE           # one reason, the worse one
    assert pool.axon_c_reading(reg, rejected) == reg + pool.C_SUFFIX_NOT_RELIABLE
    assert pool.axon_c_reading(reg, unassessed) == reg + pool.C_SUFFIX_NOT_ASSESSED
    for reading in ("neighbours above leakage", "leakage only", "no correlation between slices"):
        assert pool.axon_c_reading(reading, apart) == reading + pool.C_SUFFIX_SLICES   # all read apart
        assert pool.axon_c_reading(reading, ok, bc.VS_LEAK_NOT_ABOVE) == reading       # leak suffix: registry only
    none = "no correlation between slices"
    assert pool.axon_c_reading(none, bad) == none + pool.C_SUFFIX_NOT_RELIABLE         # low power: not readable
    assert pool.axon_c_reading("not computable", bad) == "not computable"
    assert pool.axon_c_reading(None, bad) is None and pool.axon_c_reading("", bad) is None
    # r(-1, +1) of a registry compared with the leak of the pair
    # (v21: 47 nm main, range 36-51 nm; the 77 nm upper bound adds no suffix)
    assert pool.axon_c_reading(reg, ok, bc.VS_LEAK_ABOVE_UPPER) == reg
    assert pool.axon_c_reading(reg, ok, bc.VS_LEAK_ABOVE) == reg
    assert pool.axon_c_reading(reg, ok, bc.VS_LEAK_WITHIN_RANGE) == reg + pool.C_SUFFIX_LEAK_RANGE
    assert pool.axon_c_reading(reg, ok, bc.VS_LEAK_LIMIT) == reg + pool.C_SUFFIX_LEAK_LIMIT
    assert pool.axon_c_reading(reg, ok, bc.VS_LEAK_NOT_ABOVE) == reg + pool.C_SUFFIX_LEAK_NOT_ABOVE


def test_c_reading_in_the_row_follows_the_gate(base):
    reg = "registry between slices"
    fixed = replace(base, diagnostics=replace(base.diagnostics, c_reading=reg))
    fit = dict(contour_gate_reasons="", contour_failure_reason="", contour_slice_rms_minus_nm=20.0,
               contour_slice_rms_plus_nm=80.0)
    ok = _with(fixed, 1, contour_slice_misfit=True, **fit)       # misfit mark, but the better lateral is at 20 nm
    apart = _with(fixed, 2, **dict(fit, contour_slice_rms_minus_nm=40.0), contour_slice_misfit=True)
    bad = _with(fixed, 3, **dict(fit, contour_gate_reasons="only 44% of the localizations within +-100 nm"),
                contour_slice_misfit=False)
    nan = _with(fixed, 4, **dict(fit, contour_slice_rms_plus_nm=float("nan")), contour_slice_misfit=False)
    rows = [pool.perimeter_row(r) for r in (ok, apart, bad, nan)]
    base_reading = [r["pd_c_reading"].split(" (r(-1, +1)")[0] for r in rows]   # leave the leak comparison aside
    assert base_reading == [reg, reg + pool.C_SUFFIX_SLICES, reg + pool.C_SUFFIX_NOT_RELIABLE,
                            reg + pool.C_SUFFIX_SLICES]
    assert [r["pd_c_slices_coincide"] for r in rows] == [True, False, True, False]
    assert [r["pd_c_main_reading"] for r in rows] == [True, False, False, False]
    # the numbers do not change, only the words
    assert len({r["pd_c_r_m1_p1"] for r in rows}) == 1 and len({r["pd_c_p_m1_p1"] for r in rows}) == 1
    assert fixed.diagnostics.c_reading == reg                                          # the result is not mutated
    assert pe.perimeter_block(apart)["pd_c_reading"].startswith(reg + pool.C_SUFFIX_SLICES)   # the export carries it
    # the leak of the three pairs at both widths, and r(-1, +1) against it
    r0 = rows[0]
    for sz in (36, 47, 51, 77):                                  # v21 adds 36 and 51 nm (the measured range)
        for sfx in ("m1_0", "0_p1", "m1_p1"):
            assert 0.0 <= r0[f"pd_c_leak_r_{sfx}_sz{sz}nm"] <= 1.0
    assert (r0["pd_c_leak_r_m1_p1_sz36nm"] < r0["pd_c_leak_r_m1_p1_sz47nm"] < r0["pd_c_leak_r_m1_p1_sz51nm"]
            < r0["pd_c_leak_r_m1_p1_sz77nm"])
    assert r0["pd_c_m1_p1_vs_leak"] == bc.m1_p1_vs_leak(
        r0["pd_c_r_m1_p1"], {float(sz): r0[f"pd_c_leak_r_m1_p1_sz{sz}nm"] for sz in (36, 47, 51, 77)})
    assert r0["pd_c_m1_p1_vs_leak"] in bc.VS_LEAK_VALUES


def test_slice_criterion_rule():
    assert bc.C_SLICE_MATCH_MAX_NM == 32.0
    assert bc.slices_coincide_for_c(46.0, 27.0) is True              # the smaller one rules (a pilot axon)
    assert bc.slices_coincide_for_c(32.0, 200.0) is True and bc.slices_coincide_for_c(32.01, 33.0) is False
    for missing in (None, float("nan"), "x"):
        assert bc.slices_coincide_for_c(missing, 10.0) is False      # not measurable: does not coincide
        assert bc.slices_coincide_for_c(10.0, missing) is False


def test_leak_with_real_fractions_matches_the_simulator_and_the_equal_slice_formula():
    from tools import mps_perimeter_sim as sim
    for sz in (47.0, 77.0):
        M = bc.leak_matrix_uniform(sz, 170.0, 0.0)
        Ms, _w = sim.leak_matrix(thickness_nm=170.0, lpz_nm=sz, guard_nm=0.0, depth="uniform")
        assert M == pytest.approx(Ms, abs=2e-3)
    # equal observed slices: the neighbour value is the old formula's (plus the tiny leak to the slice after next)
    eq = bc.leak_expected_r_pairs([1000, 1000, 1000], 47.0, 170.0, 0.0)
    assert eq[(-1, 0)] == pytest.approx(bc.leak_expected_r(47.0, 170.0, 0.0), abs=0.01)
    assert eq[(-1, 1)] == pytest.approx(0.02, abs=0.01)
    # laterals at 45 % of the central (the pilot): the pair -1/+1 is several times larger (validation report sec. 6)
    real47 = bc.leak_expected_r_pairs([450, 1000, 450], 47.0, 170.0, 0.0)
    real77 = bc.leak_expected_r_pairs([450, 1000, 450], 77.0, 170.0, 0.0)
    assert real47[(-1, 1)] > 3 * eq[(-1, 1)] and 0.06 < real47[(-1, 1)] < 0.15
    assert 0.2 < real77[(-1, 1)] < 0.4
    # the observed fractions are the masses: the same as the simulator's expected r with those fractions
    w = np.array([450.0, 1000.0, 450.0])
    M = bc.leak_matrix_uniform(47.0, 170.0, 0.0)
    want = sim.expected_slice_r(M, w / w.sum())
    assert real47[(-1, 1)] == pytest.approx(want[0, 2], abs=1e-9)
    assert real47[(-1, 0)] == pytest.approx(want[0, 1], abs=1e-9)
    # an empty slice: its pairs are NaN; nothing at all: all NaN
    e = bc.leak_expected_r_pairs([0, 1000, 450], 47.0, 170.0, 0.0)
    assert math.isnan(e[(-1, 0)]) and math.isnan(e[(-1, 1)]) and math.isfinite(e[(0, 1)])
    assert all(math.isnan(v) for v in bc.leak_expected_r_pairs([0, 0, 0], 47.0, 170.0, 0.0).values())
    # r(-1, +1) against the leak: 47 nm main, range 36-51 nm, 77 nm upper bound (v21; NeNA in z, 2026-11-17)
    assert bc.LEAK_SIGMA_Z_MAIN_NM == 47.0 and bc.LEAK_SIGMA_Z_RANGE_NM == (36.0, 51.0)
    assert bc.LEAK_SIGMA_Z_UPPER_NM == 77.0
    lk = {36.0: 0.05, 47.0: 0.10, 51.0: 0.12, 77.0: 0.33}        # about one axon of the pilot
    assert bc.m1_p1_vs_leak(0.5, lk) == bc.VS_LEAK_ABOVE_UPPER
    assert bc.m1_p1_vs_leak(0.2, lk) == bc.VS_LEAK_ABOVE
    assert bc.m1_p1_vs_leak(0.11, lk) == bc.VS_LEAK_WITHIN_RANGE
    assert bc.m1_p1_vs_leak(0.081, lk) == bc.VS_LEAK_LIMIT                # that axon: passes 36, not 47
    assert bc.m1_p1_vs_leak(0.03, lk) == bc.VS_LEAK_NOT_ABOVE
    assert bc.m1_p1_vs_leak(None, lk) is None and bc.m1_p1_vs_leak(0.5, {47.0: 0.1, 77.0: 0.3}) is None


def test_failed_contour_row_has_every_question_column_empty(base):
    failed = replace(base, status=pdt.Status.CONTOUR_FAILED, failure="contour failed: inward notch",
                     a={}, b=None, c={}, profiles=None, diagnostics=None, band=None)
    row = pool.perimeter_row(failed)
    keys = [c for c in pe.ALL_PERIMETER_COLUMNS
            if c.startswith(("pd_a_", "pd_b_", "pd_c_"))]
    assert len(keys) >= 32
    assert [c for c in keys if c not in row] == []                    # present (P11 sec. 9.1) ...
    assert all(row[c] is None for c in keys if c.startswith(("pd_a_contrast", "pd_a_period", "pd_b_spearman",
                                                             "pd_b_z", "pd_b_p", "pd_c_r_", "pd_c_z_", "pd_c_p_")))


def _flat(animal, n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        row = {"pd_status": pdt.Status.OK.value, "pd_pool_included": True, "pd_pool_included_without_misfit": True,
               "pd_curve_stratum": "primary", "animal": animal, "genotype": "WT", "roi_name": "ROI 1"}
        for m in pool.Z_READOUTS:
            row[m] = float(rng.normal(shift, 1.0))
        out.append(row)
    return out


def test_by_animal_no_p_with_one_animal():
    for animal in ("mouse A", pool.PROVISIONAL_ANIMAL):
        (row,) = pool.summarise_by_animal(_flat(animal, 8, 1, shift=2.0))
        assert row["n_animals_total"] == 1 and "one animal: no p" in row["note"]
        assert all(row[f"p_{m}"] is None and row[f"p_holm_{m}"] is None for m in pool.Z_READOUTS)
        assert all(row[f"mean_{m}"] is not None and row[f"n_{m}"] == 8 for m in pool.Z_READOUTS)   # still described
    # a provisional animal next to a real one: still no p for the provisional one
    by = {r["animal"]: r for r in pool.summarise_by_animal(_flat(pool.PROVISIONAL_ANIMAL, 6, 2, 2.0)
                                                           + _flat("mouse B", 6, 3, 2.0))}
    assert "provisional animal: no p" in by[pool.PROVISIONAL_ANIMAL]["note"]
    assert by[pool.PROVISIONAL_ANIMAL]["p_pd_b_z"] is None and by["mouse B"]["p_pd_b_z"] is not None


def test_by_animal_family_leaves_out_the_leak_and_the_period():
    from tools import mps_gap_batch as gb
    flat = _flat("mouse A", 8, 4, 1.5) + _flat("mouse B", 8, 5, 1.5)
    for i, row in enumerate(flat):
        row["pd_c_main_reading"] = i % 2 == 0                           # half of each animal in the main reading of C
    by = pool.summarise_by_animal(flat)
    assert len(by) == 2
    for row in by:
        for m in ("pd_c_z_m1_0", "pd_c_z_0_p1", "pd_a_period_z", "pd_c_z_m1_p1"):   # C: descriptive in the main set
            assert row[f"p_{m}"] is None and row[f"p_holm_{m}"] is None and row[f"mean_{m}"] is not None
        tested = [row[f"p_{m}"] for m in pool.AB_TESTED]
        assert all(p is not None for p in tested)
        assert [row[f"p_holm_{m}"] for m in pool.AB_TESTED] == pytest.approx(gb.holm(tested))
        assert pool.C_ELSEWHERE_NOTE in row["note"]
        assert "no p for pd_c_z_m1_0 and pd_c_z_0_p1" in row["note"] and "treating its axons as replicates" in row["note"]
    # the gap batch's default (every measure tested) is unchanged
    agg = gb.aggregate_by_animal(_flat("mouse A", 5, 6, 1.0), pool.Z_READOUTS)
    assert all(agg[0][f"p_{m}"] is not None for m in pool.Z_READOUTS)
    byc =pool.summarise_by_animal(flat, pool.SCOPE_C_MAIN)
    assert [r["n_axons"] for r in byc] == [4, 4]
    for row in byc:
        assert row["p_pd_c_z_m1_p1"] is not None and row["p_pd_a_contrast_z"] is None and row["p_pd_b_z"] is None
    # across the two scopes: Holm over the family of three, written back into both rows
    both = pool.by_animal_rows_scoped(flat, [pool.SCOPE_MAIN, pool.SCOPE_C_MAIN])
    for animal in ("mouse A", "mouse B"):
        rm = next(r for r in both if r["animal"] == animal and r["scope"] == pool.SCOPE_MAIN)
        rc = next(r for r in both if r["animal"] == animal and r["scope"] == pool.SCOPE_C_MAIN)
        ps = [rm["p_pd_a_contrast_z"], rm["p_pd_b_z"], rc["p_pd_c_z_m1_p1"]]
        assert [rm["p_holm_pd_a_contrast_z"], rm["p_holm_pd_b_z"], rc["p_holm_pd_c_z_m1_p1"]] == pytest.approx(
            gb.holm(ps))
        assert pool.JOINT_HOLM_NOTE in rm["note"] and pool.JOINT_HOLM_NOTE in rc["note"]


def test_pooled_a_below_the_minimum_says_n_too_small(base):
    results = [_with(base, i, contour_gate_reasons="", contour_failure_reason="") for i in range(3)]
    (res,) = pool.pooled_a(results, n_null=49).values()
    assert res.n_axons == 3 < pool.MIN_POOLED_A_AXONS
    label = pool.a_reading_label(res)
    assert pool.MIN_POOLED_A_AXONS == 9                        # 2000 sets at 9-11 (2026-11-19); 10 before, 11 in v20
    assert label.startswith("n too small (n = 3 axons < 9): not interpreted; computed label '")
    eight = [_with(base, i, contour_gate_reasons="", contour_failure_reason="") for i in range(8)]
    (res8,) = pool.pooled_a(eight, n_null=49).values()
    assert pool.a_reading_label(res8).startswith("n too small (n = 8 axons < 9)")
    nine = [_with(base, i, contour_gate_reasons="", contour_failure_reason="") for i in range(9)]
    (res9,) = pool.pooled_a(nine, n_null=49).values()
    assert not pool.a_reading_label(res9).startswith("n too small")
    assert "contrast - null" in res.reading                                    # the numbers are kept
    (bare,) = pool.pooled_a(results, n_null=49, min_axons=1).values()
    assert pool.a_reading_label(bare) in ("attractor", "inhibition zone", "regular", "no structure detectable",
                                          "not computable") or " + " in pool.a_reading_label(bare) \
        or "not robust" in pool.a_reading_label(bare)
    assert res.contrast.p == bare.contrast.p
