"""Tests of tools.mps_perimeter_slices (package P2): z0 criteria, foci, three 170 nm slices. Simulated data only."""
from __future__ import annotations

import numpy as np
import pytest

from tools.mps_perimeter_slices import choose_z0, cut_slices, estimate_focus, slice_members, z_peak_nm
from tools.mps_perimeter_types import OUTSIDE_LABEL, FocusEstimate, Status, Z0Criterion
from tools.mps_zcalibration import ORIGIN_FILE, CalibrationRecord


def _rng() -> np.random.Generator:
    return np.random.default_rng(7)


def _sxsy(z: np.ndarray, crossing: float, rng: np.random.Generator):
    """sx - sy linear in z with a zero at ``crossing`` plus noise (pixels)."""
    base = 1.3
    d = 0.002 * (z - crossing)
    return base + d / 2 + rng.normal(0, 0.05, z.size), base - d / 2 + rng.normal(0, 0.05, z.size)


def _record(crossing: float, m: float | None) -> CalibrationRecord:
    # wx - wy = 0.004 z - 0.004 crossing
    return CalibrationRecord(layout="test", origin=ORIGIN_FILE, x_coeffs=(0.002, 1.0),
                             y_coeffs=(-0.002, 1.0 + 0.004 * crossing), magnification=m)


def test_peak_bimodal_within_5nm_of_major_mode():
    rng = _rng()
    z = np.concatenate([rng.normal(-170.0, 40.0, 3000), rng.normal(20.0, 40.0, 6000)])
    assert abs(z_peak_nm(z) - 20.0) <= 5.0


def test_peak_degenerate_inputs():
    assert np.isnan(z_peak_nm(np.array([])))
    assert np.isnan(z_peak_nm(np.array([np.nan, np.inf])))
    assert z_peak_nm(np.array([12.0, 12.0, np.nan])) == 12.0


def test_empirical_focus_crossing_minus_43():
    rng = _rng()
    z = rng.uniform(-400, 400, 20000)
    sx, sy = _sxsy(z, -43.0, rng)
    f = estimate_focus(z, sx, sy, None)
    assert f.empirical_nm is not None and abs(f.empirical_nm + 43.0) <= 5.0
    assert f.empirical_n_bins >= 10 and f.nominal_nm is None


def test_empirical_focus_missing_or_constant():
    rng = _rng()
    z = rng.uniform(-400, 400, 5000)
    f = estimate_focus(z, None, None, None)
    assert f.empirical_nm is None and f.source == "none"
    s = np.full(z.size, 1.3)
    f2 = estimate_focus(z, s, s.copy(), None)
    assert f2.empirical_nm is None and any("constant" in w for w in f2.warnings)


def test_empirical_focus_no_crossing_and_sparse_bins():
    rng = _rng()
    z = rng.uniform(-400, 400, 20000)
    sx, sy = _sxsy(z, 900.0, rng)                      # crossing far outside the range
    assert estimate_focus(z, sx, sy, None).empirical_nm is None
    f = estimate_focus(z[:50], sx[:50], sy[:50], None)   # no bin reaches 200
    assert f.empirical_nm is None and f.empirical_n_bins == 0


def test_empirical_focus_length_mismatch_raises():
    with pytest.raises(ValueError):
        estimate_focus(np.zeros(5), np.zeros(4), np.zeros(5), None)


def test_nominal_focus_from_record_with_magnification():
    f = estimate_focus(np.zeros(3), None, None, _record(112.0, 0.8))
    assert f.nominal_nm == pytest.approx(89.6, abs=0.05)
    assert f.source == "record"


def test_nominal_focus_absent_cases():
    z = np.zeros(3)
    assert estimate_focus(z, None, None, None).nominal_nm is None
    assert estimate_focus(z, None, None, _record(112.0, None)).nominal_nm is None
    bare = CalibrationRecord(layout="test", origin=ORIGIN_FILE, magnification=0.8)
    assert estimate_focus(z, None, None, bare).nominal_nm is None


def test_choose_z0_distances_for_all_criteria():
    rng = _rng()
    z = rng.normal(30.0, 40.0, 8000)
    focus = FocusEstimate(nominal_nm=0.0, empirical_nm=-43.0, empirical_n_bins=20, source="record+sx-sy")
    peak = choose_z0(z, focus, criterion=Z0Criterion.PEAK)
    assert peak.status == Status.OK and peak.z0_nm == peak.peak_nm
    pk = peak.peak_nm
    assert pk is not None and abs(pk - 30.0) < 10.0
    assert peak.peak_minus_focus_nominal_nm == pytest.approx(pk)
    assert peak.peak_minus_focus_empirical_nm == pytest.approx(pk + 43.0)
    assert peak.peak_minus_focus_nm == pytest.approx(pk)           # nominal for PEAK
    nom = choose_z0(z, focus, criterion=Z0Criterion.FOCUS_NOMINAL)
    assert nom.z0_nm == 0.0 and nom.peak_minus_focus_nm == pytest.approx(pk)
    emp = choose_z0(z, focus, criterion=Z0Criterion.FOCUS_EMPIRICAL)
    assert emp.z0_nm == -43.0 and emp.peak_minus_focus_nm == pytest.approx(pk + 43.0)
    assert emp.abs_peak_minus_focus_nm == pytest.approx(abs(pk + 43.0))
    # both distances are always present, whatever the criterion
    for c in (peak, nom, emp):
        assert c.peak_minus_focus_nominal_nm is not None and c.peak_minus_focus_empirical_nm is not None


def test_choose_z0_focus_without_focus_is_no_focus_never_peak():
    z = _rng().normal(0.0, 40.0, 2000)
    none = FocusEstimate(nominal_nm=None, empirical_nm=None, empirical_n_bins=0, source="none")
    for crit in (Z0Criterion.FOCUS_NOMINAL, Z0Criterion.FOCUS_EMPIRICAL):
        c = choose_z0(z, none, criterion=crit)
        assert c.status == Status.NO_FOCUS and c.z0_nm is None and c.peak_nm is not None
        assert c.peak_minus_focus_nm is None
    only_emp = FocusEstimate(nominal_nm=None, empirical_nm=-40.0, empirical_n_bins=9, source="sx-sy")
    c = choose_z0(z, only_emp, criterion=Z0Criterion.FOCUS_NOMINAL)
    assert c.status == Status.NO_FOCUS and c.z0_nm is None
    ok = choose_z0(z, none, criterion=Z0Criterion.PEAK)
    assert ok.status == Status.OK and ok.peak_minus_focus_nm is None


def test_choose_z0_empty_z():
    none = FocusEstimate(nominal_nm=0.0, empirical_nm=None, empirical_n_bins=0, source="record")
    c = choose_z0(np.array([]), none, criterion=Z0Criterion.PEAK)
    assert c.status == Status.TOO_FEW_LOCS and c.z0_nm is None


def test_slice_bounds_exact_and_edges_go_to_one_slice():
    z0 = -50.0
    z = np.array([z0 - 255, z0 - 85, z0 - 85 - 1e-9, z0 + 85, z0 + 255, z0 + 254.999, z0])
    s = cut_slices(z, z0)
    assert s.bounds_nm == ((z0 - 255, z0 - 85), (z0 - 85, z0 + 85), (z0 + 85, z0 + 255))
    assert list(s.label) == [-1, 0, -1, 1, OUTSIDE_LABEL, 1, 0]
    assert sum(s.n_locs) == int((s.label != OUTSIDE_LABEL).sum())
    assert s.n_locs == (2, 2, 2)


def test_guard_makes_slices_85nm():
    s = cut_slices(np.array([0.0]), 0.0, guard_nm=42.5)
    for lo, hi in s.bounds_nm:
        assert hi - lo == pytest.approx(85.0)
    z = np.array([0.0, 50.0, 100.0, 130.0, 170.0])    # 50 and 100 fall in the guard gap [42.5, 127.5)
    s = cut_slices(z, 0.0, guard_nm=42.5)
    assert list(s.label) == [0, OUTSIDE_LABEL, OUTSIDE_LABEL, 1, 1]
    with pytest.raises(ValueError):
        cut_slices(z, 0.0, guard_nm=85.0)


def test_lateral_slice_with_three_locs_is_kept_with_percentage():
    rng = _rng()
    z = np.concatenate([rng.uniform(-80, 80, 300), rng.uniform(90, 250, 90), np.array([-100.0, -150.0, -200.0])])
    s = cut_slices(z, 0.0)
    assert s.n_locs[0] == 3
    assert s.pct_of_central[0] == pytest.approx(100.0 * 3 / s.n_locs[1])
    assert s.pct_of_central[1] == pytest.approx(100.0 * s.n_locs[2] / s.n_locs[1])
    assert s.frac_in_three == pytest.approx(1.0)


def test_z_coverage_truncated_file():
    rng = _rng()
    z = rng.uniform(-190.0, 300.0, 5000)
    s = cut_slices(z, -50.0)
    assert s.z_coverage[0] == pytest.approx(55.0 / 170.0, abs=0.01)     # [-305, -135) vs z >= -190
    assert s.z_coverage[1] == 1.0
    assert s.z_coverage[2] == 1.0


def test_empty_central_and_nonfinite_z():
    z = np.array([200.0, 210.0, np.nan, np.inf])
    s = cut_slices(z, 0.0)
    assert np.isnan(s.pct_of_central[0]) and np.isnan(s.pct_of_central[1])
    assert s.label[2] == OUTSIDE_LABEL and s.label[3] == OUTSIDE_LABEL
    assert s.frac_in_three == pytest.approx(2 / 2)
    e = cut_slices(np.array([]), 0.0)
    assert np.isnan(e.frac_in_three) and e.z_coverage == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        cut_slices(z, float("nan"))


def test_slice_members_trace_to_source_index():
    z = np.array([0.0, 100.0, -100.0, 0.0, 400.0])
    src = np.array([10, 11, 12, 13, 14])
    s = cut_slices(z, 0.0)
    m = slice_members(s, src)
    assert list(m[0]) == [10, 13] and list(m[1]) == [11] and list(m[-1]) == [12]
    assert list(slice_members(s)[0]) == [0, 3]
    with pytest.raises(ValueError):
        slice_members(s, src[:3])


def test_deterministic():
    z = _rng().normal(0, 80, 3000)
    assert z_peak_nm(z) == z_peak_nm(z.copy())
    a, b = cut_slices(z, 5.0), cut_slices(z, 5.0)
    assert np.array_equal(a.label, b.label) and a.n_locs == b.n_locs
