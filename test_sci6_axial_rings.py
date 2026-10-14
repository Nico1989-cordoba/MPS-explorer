# -*- coding: utf-8 -*-
"""
SCI-6: the axial mixture, the ring boundaries, the axon axis and the ring correlation (synthetic only, R8).

- The valley is the antimode between the two adjacent modes: two equal components 2.1 widths apart (bimodal,
  Behboodian 1970) have a small positive depth referred to the peaks and none referred to the means; at 1.9
  widths there is no valley. Equal components at 2.45 widths dip by about 0.10 (the one level, C8).
- The Bayes boundary: m + sigma^2 ln(w_a / w_b) / P for equal widths (97.5 nm for 0.7 / 0.3, sigma 50, P 170),
  the valley beyond it on the lighter ring's side; no root between the means -> None; the 'bayes' segment mode
  cuts there.
- The foreign fraction F of a window against the normal tails (8.9 % for three equal rings, sigma 50, P 170,
  h 85); Delta-Z's SE and its distances in spread units; the closed-form posterior and guard trade-offs.
- The posterior on the complete mixture: a localization most probably from a discarded component is no ring's.
- The main-peak disagreement at the half-width in use; no warning on near-equal weights any more.
- The analysis exports the five axial-mixture columns at the end of the axon table, the dictionary describes them.
- The axis: the noise floor of delta_beta against simulated untilted rings; the gate by predicted slope error;
  the derived tilt checks; build_rings records how the stages ended and the checks.
- The ring correlation: the p-value arithmetic, the masked correlation, the angle frame's non-monotonicity, the
  arc frame on a simulated axon.
- The autocorrelation: window 0.35 x 170 nm, a local maximum required, three periods minimum.

Run: ``py -3 -m pytest test_sci6_axial_rings.py -q``
"""
from __future__ import annotations

import math
import os
from typing import Any, Tuple

import numpy as np
import pytest
from scipy.stats import norm

from tools import mps_periodicity as mp


def _mixture(means: Tuple[float, ...], sigmas: Tuple[float, ...], weights: Tuple[float, ...],
             extra: Tuple[Tuple[float, float, float], ...] = ()) -> mp.ZPeriodicityResult:
    """An exact fitted mixture (no fitting): dominant components plus optional discarded ones."""
    m = np.array(means, float)
    s = np.array(sigmas, float)
    w = np.array(weights, float)
    all_m = np.concatenate([m, [e[0] for e in extra]])
    all_s = np.concatenate([s, [e[1] for e in extra]])
    all_w = np.concatenate([w, [e[2] for e in extra]])
    dom = np.concatenate([np.ones(m.size, bool), np.zeros(len(extra), bool)])
    return mp.ZPeriodicityResult(
        n_components=int(all_m.size), means_nm=m, weights=w, sigmas_nm=s, delta_z_nm=np.diff(m),
        main_peak_nm=float(m[int(np.argmax(w))]), bic_by_n={}, n_discarded_components=len(extra),
        converged=True, all_means_nm=all_m, all_weights=all_w, all_sigmas_nm=all_s, all_dominant=dom)


# ----------------------------------------------------------------------------------------- boundaries
def test_antimode_near_the_resolution_limit() -> None:
    near = _mixture((0.0, 2.1 * 50.0), (50.0, 50.0), (0.5, 0.5))
    v = mp.density_valleys(near)
    assert bool(v.is_true_valley[0]) and 0.0 < float(v.relative_depth[0]) < 0.03
    assert float(mp.density_valleys(near, depth_reference="means").relative_depth[0]) == 0.0
    assert abs(float(v.positions_nm[0]) - 52.5) < 1.0
    unresolved = _mixture((0.0, 1.9 * 50.0), (50.0, 50.0), (0.5, 0.5))
    u = mp.density_valleys(unresolved)
    assert not bool(u.is_true_valley[0]) and float(u.relative_depth[0]) == 0.0
    assert float(u.positions_nm[0]) == pytest.approx(0.95 * 50.0)


def test_flat_antimode_keeps_the_cut_at_the_midpoint() -> None:
    # Two modes only just (weights 0.55 / 0.45, 2.3 widths): the antimode sits ~21 nm off the midpoint in a
    # nearly flat density, above the density at the lighter mean; it is reported, the cut stays at the midpoint.
    r = _mixture((0.0, 2.3 * 50.0), (50.0, 50.0), (0.55, 0.45))
    v = mp.density_valleys(r)
    assert bool(v.is_true_valley[0]) and 0.0 < float(v.relative_depth[0]) < 0.01
    assert not bool(v.cut_at_antimode[0]) and float(v.positions_nm[0]) == pytest.approx(57.5)
    assert abs(float(v.antimode_nm[0]) - 57.5) > 15.0
    assert any("ill-determined" in w for w in v.warnings)
    # A well-determined valley is the cut.
    clear = mp.density_valleys(_mixture((0.0, 170.0), (50.0, 50.0), (0.7, 0.3)))
    assert bool(clear.cut_at_antimode[0]) and float(clear.positions_nm[0]) == float(clear.antimode_nm[0])


def test_one_depth_level_and_peaks_against_means() -> None:
    at = mp.density_valleys(_mixture((0.0, 2.45 * 50.0), (50.0, 50.0), (0.5, 0.5)))
    assert abs(float(at.relative_depth[0]) - mp.VALLEY_DEPTH_LEVEL) < 0.02
    assert not any("dip less than" in w for w in at.warnings) or float(at.relative_depth[0]) < 0.10
    for gap in (2.3, 2.6, 3.0, 3.4):
        r = _mixture((0.0, gap * 50.0), (50.0, 50.0), (0.5, 0.5))
        peaks = float(mp.density_valleys(r).relative_depth[0])
        means = float(mp.density_valleys(r, depth_reference="means").relative_depth[0])
        assert peaks >= means and peaks - means <= 0.03, (gap, peaks, means)
    shallow = mp.density_valleys(_mixture((0.0, 2.35 * 50.0), (50.0, 50.0), (0.5, 0.5)))
    assert any("barely separated" in w for w in shallow.warnings)
    assert not any("midpoints between means" in w for w in shallow.warnings)


def test_bayes_boundary_and_shifts() -> None:
    zb = mp.bayes_boundary(0.0, 50.0, 0.7, 170.0, 50.0, 0.3)
    assert zb == pytest.approx(85.0 + 50.0 ** 2 * math.log(0.7 / 0.3) / 170.0, abs=1e-9)
    assert zb == pytest.approx(97.46, abs=0.01)
    v = mp.density_valleys(_mixture((0.0, 170.0), (50.0, 50.0), (0.7, 0.3)))
    assert float(v.bayes_positions_nm[0]) == pytest.approx(zb)
    valley = float(v.positions_nm[0])
    assert 85.0 < zb < valley and valley == pytest.approx(104.2, abs=0.6)
    # Positive toward the lighter (upper) ring.
    assert float(v.shift_toward_weaker_from_midpoint_nm[0]) == pytest.approx(valley - 85.0)
    assert float(v.shift_toward_weaker_from_bayes_nm[0]) == pytest.approx(valley - zb)
    assert float(v.separation_d[0]) == pytest.approx(170.0 / math.sqrt(2 * 50.0 ** 2))
    # Equal components: Bayes, valley and midpoint coincide.
    eq = mp.density_valleys(_mixture((0.0, 170.0), (50.0, 50.0), (0.5, 0.5)))
    assert float(eq.bayes_positions_nm[0]) == pytest.approx(85.0)
    assert abs(float(eq.positions_nm[0]) - 85.0) <= 0.5
    # No root between the means: one component outweighs the other everywhere there.
    assert mp.bayes_boundary(0.0, 30.0, 0.999, 60.0, 80.0, 0.001) is None
    # A threshold on the shift is blank by default and a user setting otherwise.
    w = mp.density_valleys(_mixture((0.0, 170.0), (50.0, 50.0), (0.7, 0.3)), shift_warning_nm=5.0)
    assert any("set by you" in t for t in w.warnings)


def test_bayes_segment_mode() -> None:
    from tools.mps_columns import SEGMENT_MODES
    from tools.mps_multisegment import find_axial_segments
    assert "bayes" in SEGMENT_MODES
    rng = np.random.default_rng(3)
    z = np.concatenate([rng.normal(0.0, 45.0, 7000), rng.normal(180.0, 45.0, 3000)])
    segs, zr, valleys, _w = find_axial_segments(z, mode="bayes", min_locs=10, half_width_nm=150.0)
    assert len(segs) == 2
    assert segs[0].zmax_nm == pytest.approx(float(valleys.bayes_positions_nm[0]))
    assert segs[1].zmin_nm == pytest.approx(float(valleys.bayes_positions_nm[0]))


# ----------------------------------------------------------------------------------------- the slab
def test_foreign_fraction_against_the_normal_tails() -> None:
    three = _mixture((-170.0, 0.0, 170.0), (50.0, 50.0, 50.0), (1 / 3, 1 / 3, 1 / 3))
    ff = mp.foreign_fraction(three, -85.0, 85.0)
    own = norm.cdf(1.7) - norm.cdf(-1.7)
    neigh = 2 * (norm.cdf(5.1) - norm.cdf(1.7))
    assert ff.fraction == pytest.approx(neigh / (own + neigh), rel=1e-9)
    assert ff.fraction == pytest.approx(0.089, abs=0.001)
    assert ff.ring_mean_nm == 0.0 and ff.n_dominant == 3
    narrow = _mixture((-170.0, 0.0, 170.0), (35.0, 35.0, 35.0), (1 / 3, 1 / 3, 1 / 3))
    assert mp.foreign_fraction(narrow, -85.0, 85.0).fraction == pytest.approx(0.015, abs=0.001)


def test_delta_z_summary() -> None:
    s = mp.delta_z_summary([160.0, 180.0])
    assert s.mean_nm == 170.0 and s.se_nm == pytest.approx(np.std([160, 180], ddof=1) / math.sqrt(2))
    gazal = s.distances[0]
    assert gazal.reference.label == "Gazal 2026" and gazal.spread_units == pytest.approx(0.0)
    deste = next(d for d in s.distances if d.reference.label == "D'Este 2016")
    assert deste.spread_units == pytest.approx((170.0 - 186.0) / 23.0)
    zhong = next(d for d in s.distances if d.reference.label == "Zhong 2014")
    assert zhong.spread_units is None and zhong.difference_nm == pytest.approx(-20.0)
    assert all("WITHIN" not in line and "OUTSIDE" not in line for line in s.lines())
    one = mp.delta_z_summary([171.0])
    assert one.se_nm is None and one.n_spacings == 1


def test_closed_form_tradeoffs() -> None:
    t = mp.posterior_tradeoff(50.0, 170.0, 0.85)
    assert t.half_width_nm == pytest.approx(25.5, abs=0.1)
    assert t.rejected_fraction == pytest.approx(0.10, abs=0.005)
    assert t.error_among_accepted == pytest.approx(0.015, abs=0.001)
    g = mp.guard_tradeoff(50.0, 85.0, 60.0)
    assert g.lost_fraction == pytest.approx(0.125, abs=0.001) and g.leak_fraction == pytest.approx(0.011, abs=0.001)
    assert mp.guard_tradeoff(50.0, 85.0, 0.0).leak_fraction == pytest.approx(0.045, abs=0.001)
    cap = mp.guard_retention_cap_nm(50.0, 85.0)
    assert cap == pytest.approx(2 * (85.0 - 0.67 * 50.0))
    assert mp.guard_tradeoff(50.0, 85.0, cap).within_retention_bound
    assert not mp.guard_tradeoff(50.0, 85.0, cap + 1.0).within_retention_bound


def test_posterior_on_the_complete_mixture() -> None:
    from tools.mps_columns import posterior_ambiguous
    r = _mixture((0.0, 170.0), (40.0, 40.0), (0.48, 0.48), extra=((85.0, 8.0, 0.04),))
    post, ring = mp.component_posterior(np.array([0.0, 85.0, 170.0]), r)
    assert list(ring) == [0, -1, 1]
    _p, ring_dom = mp.component_posterior(np.array([0.0, 85.0, 170.0]), r, model="dominant")
    assert ring_dom[1] in (0, 1)
    flags = posterior_ambiguous(np.array([0.0, 85.0, 170.0]), r, 0.5)
    assert list(flags) == [False, True, False]
    flags_dom = posterior_ambiguous(np.array([0.0, 170.0]), r, 0.5, model="dominant")
    assert not flags_dom.any()


def test_main_peak_disagreement_at_the_half_width_and_no_weight_test() -> None:
    rng = np.random.default_rng(11)
    # A broad heavy component and a narrow lighter one: the density peak sits on the narrow one.
    z = np.concatenate([rng.normal(0.0, 140.0, 5500), rng.normal(320.0, 25.0, 4500)])
    r = mp.fit_z_periodicity(z, component_range=(2,), half_width_nm=90.0)
    d = abs(r.peak_by_density_nm - r.peak_by_weight_nm)
    assert d > 200 and any("Main-peak estimates disagree" in w and "half-width of 90" in w for w in r.warnings)
    wide = mp.fit_z_periodicity(z, component_range=(2,), half_width_nm=d + 10.0)
    assert not any("Main-peak estimates disagree" in w for w in wide.warnings)
    equal = mp.fit_z_periodicity(np.concatenate([rng.normal(0, 40, 5000), rng.normal(180, 40, 5000)]))
    assert not any("Ambiguous main peak" in w for w in equal.warnings)
    assert equal.main_peak_mode == "density" and math.isfinite(equal.peak_by_weight_nm)
    assert equal.candidate_counts == (2, 3) and equal.n_init == mp.DEFAULT_N_INIT


def test_analysis_exports_the_axial_mixture_columns(tmp_path: Any) -> None:
    from tools.axon_export import AXIAL_MIXTURE_COLUMNS, TABLE_VERSION, axon_row
    from tools.column_dictionary import describe, known_columns
    from tools.mps_analysis import analyze_axon
    rng = np.random.default_rng(5)
    n = 2400
    theta = rng.uniform(0, 2 * np.pi, n)
    ring_z = rng.choice([0.0, 170.0, 340.0], n, p=[0.3, 0.4, 0.3])
    x = 1500 * np.cos(theta) + rng.normal(0, 15, n)
    y = 1500 * np.sin(theta) + rng.normal(0, 15, n)
    z = ring_z + rng.normal(0, 45, n)
    a = analyze_axon(x, y, z, source_name="sim", pixel_size_nm=130.0, pixel_size_source="override",
                     run_randomization=False)
    assert a.slab_foreign is not None and a.slab_foreign_fraction is not None
    row = axon_row(a, exported_at="t")
    assert list(row)[-len(AXIAL_MIXTURE_COLUMNS):] == list(AXIAL_MIXTURE_COLUMNS)
    assert row["table_version"] == TABLE_VERSION == "axon tables v6"
    assert row["main_peak_mode"] == "density" and 0.0 <= row["slab_foreign_fraction"] <= 1.0
    assert all(describe(c) for c in AXIAL_MIXTURE_COLUMNS)
    assert known_columns()[-len(AXIAL_MIXTURE_COLUMNS):] == list(AXIAL_MIXTURE_COLUMNS)
    names = [r[0] for r in a.summary_rows()]
    assert "  foreign fraction of the slab" in names
    note = next(r[3] for r in a.summary_rows() if r[0] == "Delta-Z (axial periodicity)")
    assert "interval(s)" in note and "Gazal 2026" in note
    mode_w = analyze_axon(x, y, z, source_name="sim", pixel_size_nm=130.0, pixel_size_source="override",
                          run_randomization=False, main_peak_mode="weight")
    assert mode_w.z_result.main_peak_mode == "weight"


# ----------------------------------------------------------------------------------------- the axis
def _rings_xyz(rng: np.random.Generator, n_per_ring: int, k: int = 3, sigma_z: float = 50.0,
               radius: float = 1500.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, zs, labels = [], [], [], []
    for ring in range(k):
        t = rng.uniform(0, 2 * np.pi, n_per_ring)
        xs.append(radius * np.cos(t))
        ys.append(radius * np.sin(t))
        zs.append(ring * 400.0 + rng.normal(0, sigma_z, n_per_ring))
        labels.append(np.full(n_per_ring, ring))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), np.concatenate(labels)


def test_delta_beta_noise_floor_against_simulation() -> None:
    from tools.mps_axis import fit_axon_frame
    rng = np.random.default_rng(7)
    db, floors, tilt_floors, betas = [], [], [], []
    for _ in range(60):
        x, y, z, lab = _rings_xyz(rng, 500)
        f = fit_axon_frame(x, y, z, lab, loss="linear")
        db.append(f.delta_beta_deg)
        floors.append(f.delta_beta_floor_deg)
        tilt_floors.append(f.tilt_noise_floor_deg)
        betas.append(f.beta_deg)
    rms_db = math.sqrt(float(np.mean(np.square(db))))
    floor = float(np.median(floors))
    assert 0.7 < rms_db / floor < 1.3, (rms_db, floor)
    # The axis' own error: beta of an untilted axon is pure noise, E[beta^2] = floor^2.
    rms_beta = math.sqrt(float(np.mean(np.square(betas))))
    assert 0.7 < rms_beta / float(np.median(tilt_floors)) < 1.3
    f = fit_axon_frame(*_rings_xyz(rng, 500)[:3], _rings_xyz(rng, 500)[3])
    assert np.isfinite(f.delta_beta_corrected_deg) and f.delta_beta_corrected_deg <= f.delta_beta_deg
    assert f.ring_slope_error_deg.shape == (3,) and np.all(np.isfinite(f.ring_slope_error_deg))


def test_gate_by_predicted_slope_error_and_the_tilt_checks() -> None:
    from tools.mps_axis import fit_axon_frame, tilt_checks
    rng = np.random.default_rng(8)
    x, y, z, lab = _rings_xyz(rng, 400)
    narrow = lab == 2
    x[narrow] *= 0.05
    y[narrow] *= 0.05
    f = fit_axon_frame(x, y, z, lab, max_slope_error_deg=0.3)
    assert list(f.ring_used) == [True, True, False]
    assert any("predicted slope error" in w for w in f.warnings)
    c = tilt_checks(3.0, math.pi * 3000.0, 170.0)
    assert c is not None and c.beta_p_deg == pytest.approx(3.24, abs=0.01)
    assert c.beta_z_deg == pytest.approx(11.86, abs=0.02) and c.z_usable_source == "default"
    assert not c.correction_mandatory and not c.warnings()
    big = tilt_checks(5.0, math.pi * 8000.0, 170.0, 800.0, "calibration")
    assert big is not None and big.correction_mandatory and big.beyond_usable_depth
    assert len(big.warnings()) == 2 and "calibration" in big.warnings()[1]


def test_build_rings_records_the_stages_and_the_tilt_checks(tmp_path: Any) -> None:
    import batch_columns as bc
    from tools.mps_columns import RingsParams, build_rings
    from tools.mps_zquality_window import inputs_from_npz
    npz = bc.write_simulated_input(str(tmp_path / "sim.npz"), "circle", 39, 0.3)
    inp = inputs_from_npz(npz)
    res = build_rings(inp.x_nm, inp.y_nm, inp.z_nm, params=RingsParams(), source_name="sim",
                      pixel_size_nm=130.0, pixel_size_source="override")
    assert res.axis_settle_status in ("settled", "noise-limited", "not converged")
    assert res.axis_stage_tol_used_deg >= RingsParams().axis_stage_tol_deg
    assert res.tilt_checks is not None and res.tilt_checks.z_usable_source == "default"
    judged = build_rings(inp.x_nm, inp.y_nm, inp.z_nm, params=RingsParams(axis_judge="mean"),
                         source_name="sim", pixel_size_nm=130.0, pixel_size_source="override")
    assert judged.n_rings == res.n_rings


# ----------------------------------------------------------------------------------------- ring correlation
def test_rotation_null_arithmetic_and_mask() -> None:
    from tools.mps_gaps import DEFAULT_N_ROTATIONS, _nonmonotonic_fraction, profile_correlation
    rng = np.random.default_rng(9)
    a = (rng.random(3600) < 0.2).astype(float)
    a = np.convolve(np.r_[a, a[:20]], np.ones(20) / 20, mode="valid")[:3600]
    self_c = profile_correlation(a, a)
    assert self_c.n_rotations == DEFAULT_N_ROTATIONS
    assert self_c.p_rotation == pytest.approx(1.0 / (1 + DEFAULT_N_ROTATIONS))
    b = np.roll(a, 50) + 0.01 * rng.normal(size=3600)
    full = profile_correlation(a, b)
    every = np.ones(3600, bool)
    same = profile_correlation(a, b, measured_a=every, measured_b=every)
    assert same.r_at_zero == full.r_at_zero and not same.masked
    hole = np.zeros(3600, bool)
    hole[100:400] = True
    a2, b2 = a.copy(), b.copy()
    a2[hole] = 5.0
    b2[hole] = -3.0
    masked = profile_correlation(a2, b2, measured_a=~hole, measured_b=~hole)
    keep = ~hole
    assert masked.masked and masked.r_at_zero == pytest.approx(float(np.corrcoef(a[keep], b[keep])[0, 1]), abs=1e-9)
    circle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    assert _nonmonotonic_fraction(circle) == 0.0
    folded = np.mod(np.concatenate([np.linspace(0, 4, 200), np.linspace(4, 3, 50), np.linspace(3, 2 * np.pi, 150)]),
                    2 * np.pi)
    assert _nonmonotonic_fraction(folded) == pytest.approx(49 / 400, abs=0.01)


def test_arc_frame_on_a_simulated_axon(tmp_path: Any) -> None:
    import batch_columns as bc
    from tools.mps_gaps import analyze_rings
    from tools.mps_multisegment import analyze_all_segments
    from tools.mps_zquality_window import inputs_from_npz
    npz = bc.write_simulated_input(str(tmp_path / "sim.npz"), "ellipse", 7, 0.3)
    inp = inputs_from_npz(npz)
    ms = analyze_all_segments(inp.x_nm, inp.y_nm, inp.z_nm, source_name="sim", mode="valley",
                              pixel_size_nm=130.0, pixel_size_source="override")
    arc = analyze_rings(ms)
    assert arc.frame == "arc" and arc.pairs
    row = arc.pairs[0].export_dict()
    assert row["correlation_frame"] == "arc" and row["n_rotations"] == 1999 and row["n_bins"] == 3600
    assert row["n_effective_rotations"] is not None and 0 < row["n_effective_rotations"] <= 3600
    angle = analyze_rings(ms, frame="angle", empty_bins="mask")
    assert angle.frame == "angle" and angle.pairs[0].export_dict()["empty_bins"] == "mask"
    pair_rows = ms.pair_rows()
    assert {"boundary_midpoint_nm", "boundary_bayes_nm", "boundary_valley_nm",
            "boundary_separation_d"} <= set(pair_rows[0])


# ----------------------------------------------------------------------------------------- autocorrelation
def test_autocorrelation_window_local_maximum_and_length() -> None:
    from tools.mps_autocorrelation import (DEFAULT_PERIOD_NM, DEFAULT_SEARCH_WINDOW, autocorrelation_amplitude,
                                           min_profile_samples)
    assert DEFAULT_PERIOD_NM == 170.0 and DEFAULT_SEARCH_WINDOW == 0.35
    assert min_profile_samples(10.0) == 51 and min_profile_samples(20.0) == 26
    i = np.arange(600)
    amp, period, _ = autocorrelation_amplitude(np.sin(2 * np.pi * i * 10.0 / 220.0), 10.0)
    assert period == 220.0 and amp is not None
    # 250 nm lies outside +/-35 % of 170 nm: no peak in the window.
    amp2, period2, _ = autocorrelation_amplitude(np.sin(2 * np.pi * i * 10.0 / 250.0), 10.0)
    assert period2 is None and amp2 is None
    rng = np.random.default_rng(0)
    noise = np.convolve(np.cumsum(rng.normal(size=800)), np.ones(25) / 25, mode="same")
    assert autocorrelation_amplitude(noise, 10.0)[1] is None


def test_registry_entries_of_sci6() -> None:
    from tools import mps_param_registry as reg
    for key in ("rings.frame", "rings.empty_bins", "rings.n_rotations", "slab.foreign_fraction_warn",
                "axis.stage_tol_deg", "axis.judge", "acr.search_window", "reference.delta_z_periods"):
        e = reg.info(key)
        assert e.records and e.origin in reg.BADGES
    assert reg.default("rings.valley_depth_colour") == mp.VALLEY_DEPTH_LEVEL == 0.10
    assert reg.default("rings.frame") == "arc" and reg.default("rings.n_rotations") == 1999
    assert reg.badge("rings.valley_shift_warning_nm") == "blank"
    assert not any(reg.info(k).changes_in == "SCI-6" for k in reg.keys())


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
