# -*- coding: utf-8 -*-
"""Tests of the per-file axial sigma (tools.mps_axial_sigma) with simulated files of known error, and of what the
leak of C does with it (width used, source exported)."""
import math

import numpy as np

from tools import mps_axial_sigma as ax


def simulate_file(*, sc=24.0, sb=88.0, w=0.18, delta=27.0, n_mol=700, events=25, length=15000.0, n_frames=60000,
                  dwell=(1, 2, 3), seed=1, lateral=8.0):
    """Blinking molecules on a band. Each switch-on lasts 1-3 frames; every localization has a z error that is
    good (sd sc) or bad (sd sb, probability w); every switch-on also has its own slow offset N(0, delta).
    Returns x, y, z, frame (all localizations) and the true equivalent sd of the leak."""
    rng = np.random.default_rng(seed)
    mx = rng.uniform(0, length, n_mol)
    my = rng.uniform(-50, 50, n_mol)
    mz = rng.uniform(-85, 85, n_mol)
    xs, ys, zs, fs = [], [], [], []
    for m in range(n_mol):
        k = rng.poisson(events)
        starts = rng.choice(n_frames - 4, size=k, replace=False)
        for s in starts:
            off = rng.normal(0.0, delta) if delta > 0 else 0.0
            for f in range(int(rng.choice(dwell))):
                bad = rng.random() < w
                e = rng.normal(0.0, sb if bad else sc)
                xs.append(mx[m] + rng.normal(0, lateral))
                ys.append(my[m] + rng.normal(0, lateral))
                zs.append(mz[m] + off + e)
                fs.append(s + f)
    truth = (1 - w) * math.hypot(sc, delta) + w * math.hypot(sb, delta)
    return np.array(xs), np.array(ys), np.array(zs), np.array(fs, dtype=np.int64), truth


def test_recovers_known_error_with_slow_component():
    x, y, z, fr, truth = simulate_file(delta=27.0)
    r = ax.measure_axial_sigma(x, y, z, fr, n_boot=39, seed=3)
    assert r.status == ax.STATUS_MEASURED and r.reliable, r.warnings
    frame_truth = 0.82 * 24 + 0.18 * 88
    assert abs(r.frame_nm - frame_truth) < 4.0
    assert r.mixture_by_bic and abs(r.bad_fraction - 0.18) < 0.07
    assert abs(r.slow_nm - 27.0) < 7.0
    assert abs(r.total_nm - truth) < 6.0
    assert r.total_lo_nm <= r.total_nm <= r.total_hi_nm
    assert ax.leak_widths(r) == (min(r.total_lo_nm, r.total_nm), r.total_nm, max(r.total_hi_nm, r.total_nm))


def test_no_slow_component_total_equals_frame():
    x, y, z, fr, truth = simulate_file(delta=0.0, seed=2)
    r = ax.measure_axial_sigma(x, y, z, fr, n_boot=19, seed=1)
    assert r.status == ax.STATUS_MEASURED, r.warnings
    assert r.slow_nm < 9.0
    assert abs(r.total_nm - r.frame_nm) < 4.0


def test_gaussian_error_goes_to_the_single_model():
    x, y, z, fr, _ = simulate_file(sc=40.0, sb=40.0, w=0.0, delta=0.0, seed=4)
    r = ax.measure_axial_sigma(x, y, z, fr, n_boot=0)
    assert abs(r.frame_nm - 40.0) < 4.0 and r.bad_fraction < 0.05 or not r.mixture_by_bic


def test_lpz_is_only_reported_never_used():
    x, y, z, fr, _ = simulate_file(seed=5, n_mol=500)
    a = ax.measure_axial_sigma(x, y, z, fr, n_boot=0)
    b = ax.measure_axial_sigma(x, y, z, fr, lpz=np.full(x.size, 10.0), lp=np.full(x.size, 5.0), n_boot=0)
    assert a.frame_nm == b.frame_nm and a.total_nm == b.total_nm
    assert b.lpz_median_nm == 10.0 and abs(b.ratio_to_lpz - b.frame_nm / 10.0) < 1e-9


def test_without_frame_it_cannot_measure():
    x, y, z, fr, _ = simulate_file(seed=6, n_mol=300)
    r = ax.measure_axial_sigma(x, y, z, None)
    assert r.status == ax.STATUS_NO_FRAME and not r.reliable and "frame" in r.note()
    assert ax.leak_widths(r) is None
    r2 = ax.measure_axial_sigma(x, y, z, fr + 0.5)
    assert r2.status == ax.STATUS_NO_FRAME


def test_too_few_pairs_and_too_short_acquisition():
    x, y, z, fr, _ = simulate_file(seed=7, n_mol=40, events=8)
    r = ax.measure_axial_sigma(x, y, z, fr, n_boot=0)
    assert r.status in (ax.STATUS_TOO_FEW_LOCS, ax.STATUS_TOO_FEW_PAIRS) and not r.reliable
    x, y, z, fr, _ = simulate_file(seed=8, n_frames=1500, n_mol=450, events=8, length=200000.0)
    r = ax.measure_axial_sigma(x, y, z, fr, n_boot=0)
    assert r.status == ax.STATUS_FRAME_ONLY, (r.status, r.warnings)
    assert math.isfinite(r.frame_nm) and not r.reliable and ax.leak_widths(r) is None


def test_dense_acquisition_is_refused():
    # many molecules on every frame: the same-frame control and the ring control must refuse
    x, y, z, fr, _ = simulate_file(seed=9, n_mol=1200, events=40, length=3000.0, n_frames=6000)
    r = ax.measure_axial_sigma(x, y, z, fr, n_boot=0)
    assert r.status == ax.STATUS_CONTAMINATED and not r.reliable
    assert r.same_frame_fraction > ax.SAME_FRAME_MAX or r.contamination > ax.CONTAM_MAX


def test_resampling_by_groups():
    x, y, z, fr, _ = simulate_file(seed=10)
    g = (x // 2500).astype(int)
    r = ax.measure_axial_sigma(x, y, z, fr, groups=g, n_boot=9)
    assert r.resampled_by == "groups"


# ----------------------------------------------------------------------------------------------------
# Wiring: the width the leak of C uses, and where it came from
# ----------------------------------------------------------------------------------------------------
from dataclasses import replace  # noqa: E402

from tools import mps_perimeter_analysis as an  # noqa: E402
from tools import mps_perimeter_bc as bc  # noqa: E402
from tools import mps_perimeter_pool as pool  # noqa: E402
from tools import mps_perimeter_types as pdt  # noqa: E402
from tools.mps_perimeter_sim import simulate_scenario  # noqa: E402

import test_perimeter_export as tpe  # noqa: E402


def test_vocabulary_carries_the_widths_in_use_and_the_pilot_is_unchanged():
    assert bc.VS_LEAK_NOT_ABOVE == "not above the leak (not even at sigma_z 36 nm)"
    assert bc.VS_LEAK_ABOVE_UPPER.startswith("above the leak (sigma_z 47 nm, range 36-51 nm) and above")
    w = bc.measured_widths(40.2, 52.7, 63.0)
    lk = {w.lo: 0.05, w.main: 0.10, w.hi: 0.14, w.upper: 0.30}
    assert bc.m1_p1_vs_leak(0.2, lk, w) == "above the leak (sigma_z 53 nm, range 40-63 nm); within the upper bound (77 nm)"
    assert bc.m1_p1_vs_leak(0.07, lk, w) == "at the limit: above the leak at sigma_z 40 nm, not at 53 nm"
    assert bc.m1_p1_vs_leak(0.01, lk, w) == "not above the leak (not even at sigma_z 40 nm)"
    kinds = [bc.vs_leak_kind(bc.vs_leak_text(k, w)) for k in bc._VS_TEMPLATES]
    assert kinds == list(bc._VS_TEMPLATES)
    assert bc.vs_leak_kind(bc.VS_LEAK_LIMIT) == "limit" and bc.vs_leak_kind(None) == ""
    # a measured "range" label still gets the registry suffix of its kind
    ok = {"pd_pool_exclusion_reason": "", "pd_pool_included": True, "pd_c_slices_coincide": True}
    lab = bc.m1_p1_vs_leak(0.12, lk, w)
    assert bc.vs_leak_kind(lab) == "range"
    assert pool.axon_c_reading("registry between slices", ok, lab).endswith(pool.C_SUFFIX_LEAK_RANGE)


def _blinking_axon(seed=0, events=3, delta=20.0):
    """The a_flat simulated axon with every localization taken as a molecule that blinks `events` times."""
    sim = simulate_scenario("a_flat")
    rng = np.random.default_rng(seed)
    n = sim.x_nm.size
    rep = np.repeat(np.arange(n), events)
    k = rep.size
    off = np.repeat(rng.normal(0, delta, k), 1)
    bad = rng.random(k) < 0.18
    err = np.where(bad, rng.normal(0, 88.0, k), rng.normal(0, 24.0, k))
    two = rng.random(k) < 0.6                                   # a second consecutive frame for 60 % of the events
    start = rng.integers(0, 3_000_000, k)
    x = sim.x_nm[rep] + rng.normal(0, 4.0, k)
    y = sim.y_nm[rep] + rng.normal(0, 4.0, k)
    z = sim.z_nm[rep] + off + err
    x2 = sim.x_nm[rep[two]] + rng.normal(0, 4.0, int(two.sum()))
    y2 = sim.y_nm[rep[two]] + rng.normal(0, 4.0, int(two.sum()))
    bad2 = rng.random(int(two.sum())) < 0.18
    z2 = sim.z_nm[rep[two]] + off[two] + np.where(bad2, rng.normal(0, 88.0, int(two.sum())),
                                                   rng.normal(0, 24.0, int(two.sum())))
    cat = np.concatenate
    cols = {"sx": cat([sim.sx[rep], sim.sx[rep[two]]]), "sy": cat([sim.sy[rep], sim.sy[rep[two]]])}
    return sim, cat([x, x2]), cat([y, y2]), cat([z, z2]), cols, cat([start, start[two] + 1])


def _analyse(sim, x, y, z, cols, params=tpe.FAST):
    return an.analyze_perimeter(x, y, z, columns=cols, info=[], identity=None, source_path=tpe.SOURCE, params=params,
                                contour_fn=tpe._true_contour(sim))


def test_leak_of_c_uses_the_measured_width_and_says_so():
    sim, x, y, z, cols, frame = _blinking_axon()
    res = _analyse(sim, x, y, z, dict(cols, frame=frame), replace(tpe.FAST, sigma_z_n_boot=19))
    assert res.status == pdt.Status.OK, res.failure
    dg = res.diagnostics
    if dg.sigma_z_source != bc.SIGMA_Z_SOURCE_MEASURED:                      # say why before failing
        raise AssertionError(dg.sigma_z_report.status + " " + dg.sigma_z_report.note())
    lo, main, hi = dg.sigma_z_widths
    assert lo <= main <= hi and 25.0 < main < 90.0
    row = pool.perimeter_row(res)
    assert row["pd_sz_source"] == "measured" and row["pd_sz_status"] == "measured"
    assert row["pd_sz_main_nm"] == main
    expected = bc.leak_expected_r_pairs(res.slices.n_locs, main, res.params.slice_thickness_nm,
                                        res.params.slice_guard_nm)[(-1, 1)]
    assert abs(row["pd_c_leak_r_m1_p1_szused"] - expected) < 1e-12
    assert row["pd_c_leak_r_m1_p1_szused_lo"] <= row["pd_c_leak_r_m1_p1_szused"] <= row["pd_c_leak_r_m1_p1_szused_hi"]
    # the fixed pilot columns do not move
    pilot = bc.leak_expected_r_pairs(res.slices.n_locs, 47.0, res.params.slice_thickness_nm,
                                     res.params.slice_guard_nm)[(-1, 1)]
    assert abs(row["pd_c_leak_r_m1_p1_sz47nm"] - pilot) < 1e-12
    if row["pd_c_m1_p1_vs_leak"]:
        assert f"{main:.0f}" in row["pd_c_m1_p1_vs_leak"] or "upper bound" in row["pd_c_m1_p1_vs_leak"] \
            or "limit" in row["pd_c_m1_p1_vs_leak"] or "not above" in row["pd_c_m1_p1_vs_leak"]
    assert row["pd_sz_n_pairs_frame"] >= ax.MIN_PAIRS_FRAME and row["pd_sz_total_nm"] == main


def test_without_frame_the_pilot_width_is_used_with_a_warning():
    sim, x, y, z, cols, _frame = _blinking_axon(seed=1)
    res = _analyse(sim, x, y, z, cols)
    assert res.status == pdt.Status.OK, res.failure
    assert res.diagnostics.sigma_z_source == "pilot 47"
    assert res.diagnostics.sigma_z_widths == (36.0, 47.0, 51.0)
    assert any("sigma_z not measurable" in w and "no_frame" in w and "47" in w for w in res.warnings)
    row = pool.perimeter_row(res)
    assert row["pd_sz_source"] == "pilot 47" and row["pd_sz_status"] == "no_frame"
    assert row["pd_c_leak_r_m1_p1_szused"] == row["pd_c_leak_r_m1_p1_sz47nm"]
    assert row["pd_c_leak_r_m1_p1_szused_lo"] == row["pd_c_leak_r_m1_p1_sz36nm"]
    assert row["pd_c_leak_r_m1_p1_szused_hi"] == row["pd_c_leak_r_m1_p1_sz51nm"]
    assert row["pd_sz_total_nm"] is None and "frame" in row["pd_sz_note"]


def test_switched_off_leaves_the_pilot_and_no_measurement_columns():
    sim, x, y, z, cols, frame = _blinking_axon(seed=2)
    res = _analyse(sim, x, y, z, dict(cols, frame=frame), replace(tpe.FAST, sigma_z_per_file=False))
    assert res.diagnostics.sigma_z_source == "pilot 47" and res.diagnostics.sigma_z_report is None
    row = pool.perimeter_row(res)
    assert row["pd_sz_source"] == "pilot 47" and row["pd_sz_status"] is None
