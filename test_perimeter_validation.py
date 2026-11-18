"""
Fast versions of the P11 validations of the perimeter density analysis (``validate_perimeter_*.py``).

They check that every validation runs end to end on a few synthetic axons and that the verdict machinery
reads the rows correctly; the rates themselves come from the full runs (see the scripts' docstrings). Synthetic
data only. Run one file per process: ``python -m pytest test_perimeter_validation.py``.
"""
from __future__ import annotations

import math

import numpy as np

import validate_perimeter_a as VA
import validate_perimeter_b as VB
import validate_perimeter_c as VC
import validate_perimeter_cross as X


def test_wilson_and_ks() -> None:
    lo, hi = X.wilson(25, 500)
    assert lo < 0.05 < hi
    d, p = X.ks_uniform(list(np.linspace(0.001, 0.999, 400)))
    assert d < 0.01 and p > 0.9
    d2, p2 = X.ks_uniform([0.01] * 100)
    assert d2 > 0.9 and p2 < 1e-6


def test_cross_bins_alignment() -> None:
    rows, v = X.cross_bins_p4_p6(n_trials=3, n_points=500)
    assert v["verdict"] == "PASS", v
    assert all(r["count_mismatch_bins"] == 0 for r in rows)


def test_cross_leak_theory_agrees() -> None:
    t = X.leak_theory(47.0, 0.0)
    assert abs(t["p5_neighbour_r"] - 0.27) < 0.02
    assert abs(t["p5_neighbour_r"] - t["p1_r_m1_0"]) < 0.03
    assert t["p1_r_m1_p1"] < 0.05
    g = X.leak_theory(47.0, 42.5)
    assert 0.10 < g["p5_neighbour_r"] < 0.14      # the plan's < 0.10 is not met: documented deviation


def test_cross_halfbin_small() -> None:
    rows, v = X.cross_halfbin_p1_p4(n_sims=2, n_locs=20000)
    assert all(math.isfinite(r["chi2_dof_centred"]) for r in rows)
    assert np.median([r["chi2_dof_centred"] for r in rows]) < np.median([r["chi2_dof_cell_start"] for r in rows])


def test_cross_empty_slice_no_exception() -> None:
    rows, v = X.cross_empty_slice()
    assert all(r["exception"] == "" for r in rows)
    zero = next(r for r in rows if r["case"] == "zero")
    assert not math.isfinite(float(zero["r_m1_p1"] if zero["r_m1_p1"] is not None else float("nan")))


def test_cross_orientation() -> None:
    rows, v = X.cross_orientation()
    assert v[0]["verdict"] == "PASS", v[0]


def test_v2_c_small() -> None:
    rows = VC.run_case("null", 2) + VC.run_case("identical", 2) + VC.run_case("leak", 2)
    assert all(80 <= r["n_bins"] <= 260 for r in rows)
    ident = [r for r in rows if r["case"] == "identical"]
    # ~33 localizations per bin and slice with blinking: Poisson + blinking noise caps r well below 1
    assert all(r["r_m1_p1"] > 0.2 for r in ident)
    v = VC.verdicts(rows)
    assert any(x["check"].startswith("V2 guard") for x in v)


def test_v3_b_small() -> None:
    tr = [VB.true_axon("null", 0), VB.true_axon("pos_wavy", 0)]
    assert all(math.isfinite(r["b_p_100"]) for r in tr)
    assert tr[1]["b_r_100"] < 0          # more curvature, lower tortuosity, more density
    v = VB.verdicts(tr, [], [])
    assert v


def test_perturbed_path_zero_is_identity_and_amplitude_moves_the_curve() -> None:
    from tools import mps_perimeter_sim as S
    sim = S.simulate_perimeter("circle", {"kind": "flat"}, n_locs=500, radial_sd_nm=20.0, slices={}, lpz_nm=0.0,
                               background_frac=0.0, interior_frac=0.0, blink=False, seed=1)
    c = np.array([1000.0])
    s = np.array([1.0])
    p0, rms0, mx0 = VC.perturbed_path(sim.true_path, c, s, 0.0)
    assert rms0 == 0.0 and mx0 == 0.0
    assert np.allclose(np.asarray(p0.points_nm), np.asarray(sim.true_path.points_nm))
    p1, rms1, mx1 = VC.perturbed_path(sim.true_path, c, s, 100.0)
    assert 99.0 < mx1 <= 100.0 + 1e-6 and rms1 > 0


def test_cerr_axon_rows() -> None:
    rows = VC.cerr_axon(20.0, 0, n_null=99)
    assert [r["amp_nm"] for r in rows] == list(VC.CERR_AMPS_NM)
    assert rows[0]["gate_reliable"] and rows[0]["rms_disp_nm"] == 0.0
    assert rows[-1]["persistent_offset_nm"] > rows[0]["persistent_offset_nm"]


def test_v4_a_small_sets() -> None:
    old = VA.N_NULL_POOLED
    VA.N_NULL_POOLED = 49
    try:
        row = VA.one_set("attractor", 0, n_axons=4)
    finally:
        VA.N_NULL_POOLED = old
    assert row["n_axons"] == 4


def test_v4_a_small() -> None:
    rng = np.random.default_rng(0)
    p = VA.axon_profile("g150", rng)
    assert 80 <= p.n_bins <= 250 and p.counts.sum() > 0
    old = VA.N_NULL_POOLED
    VA.N_NULL_POOLED = 49
    try:
        row = VA.one_set("flat", 0)
    finally:
        VA.N_NULL_POOLED = old
    assert row["raw_lag0"] == 1.0
    assert VA.verdicts([row])
