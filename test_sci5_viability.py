# -*- coding: utf-8 -*-
"""
Tests of SCI-5 (ring-pair viability and the reference period), on synthetic data only (R8):

- the reference period (170 nm, Gazal et al. 2026), its presets and range; the axon's own period scales h and the
  peak window, with the reference as the fallback;
- ``ViabilityParams``: the documented ranges flag, never clamp; the store keeps only what differs and drops what
  cannot be a value;
- the closed forms: d -> (sigma / P, valley depth), the per-pair copy tolerance, the guard g*;
- the count floor by thinning, Silverman's reading, the SiZer map;
- the leak diagnostics: the size-ratio permutation p, the s_eff guard grid, the rules dict without (i), (iii), (iv);
- the Criteria dialog (offscreen).

Run: ``py -3 -m pytest test_sci5_viability.py -q``.
"""
from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Any, List, Sequence

import numpy as np
import pytest
from scipy.stats import norm

from tools import mps_axial_precision as ap
from tools import mps_leak as ml


# =========================================================================== reference period and parameters
def test_reference_period_is_the_published_value() -> None:
    assert ap.P_REF_DEFAULT_NM == 170.0
    assert "Gazal et al. 2026" in ap.P_REF_SOURCE and "170" in ap.P_REF_SOURCE
    assert ap.P_REF_RANGE_NM == (150.0, 210.0)
    values = [v for _l, v, _s in ap.P_REF_PRESETS]
    assert values[0] == 170.0 and 186.0 in values and 182.0 in values and 190.0 in values
    assert all(ap.P_REF_RANGE_NM[0] <= v <= ap.P_REF_RANGE_NM[1] for v in values)
    vp = ap.ViabilityParams()
    assert vp.p_ref_nm == 170.0 and vp.out_of_range() == ()


def test_params_flag_never_clamp_and_store() -> None:
    vp = ap.ViabilityParams(d_viable=3.5, h_over_p=0.3, p_ref_nm=140.0)
    assert vp.d_viable == 3.5 and vp.h_over_p == 0.3 and vp.p_ref_nm == 140.0          # kept as typed
    assert set(vp.out_of_range()) == {"d_viable", "h_over_p", "p_ref_nm"}
    assert "d_marginal" in ap.ViabilityParams(d_viable=2.0, d_marginal=2.1).out_of_range()
    stored = vp.to_store()
    assert stored == {"d_viable": 3.5, "h_over_p": 0.3, "p_ref_nm": 140.0}
    back = ap.ViabilityParams.from_store(dict(stored, bogus=1, recall_tolerance="x", keep_floor=-1.0))
    assert back == vp                                                             # bad entries dropped, one by one
    assert ap.ViabilityParams.from_store(None) == ap.ViabilityParams()
    for bad in ({"p_ref_nm": 0.0}, {"recall_tolerance": 1.5}, {"reference_ring": "pick"}, {"x_statistic": "mode"}):
        with pytest.raises(ValueError):
            ap.ViabilityParams(**bad)


def test_settings_store_keeps_only_valid_entries() -> None:
    from tools.mps_settings import MPSSettings
    s = MPSSettings(viability_params={"d_viable": 3.5, "nope": 1, "h_over_p": "a"}).validate()
    assert s.viability_params == {"d_viable": 3.5}
    assert MPSSettings().validate().viability_params == {}


def test_axon_period_and_fallback() -> None:
    rings = [SimpleNamespace(index=0, centre_z_nm=0.0), SimpleNamespace(index=1, centre_z_nm=172.0),
             SimpleNamespace(index=3, centre_z_nm=172.0 + 2 * 168.0)]     # ring 2 missing: index step 2
    assert ap.axon_period_nm(rings) == pytest.approx(170.0)                 # median of 172 and 168
    assert ml.axon_period_nm is ap.axon_period_nm                           # one definition
    p, src = ap.scale_period_nm(rings, 170.0)
    assert p == pytest.approx(170.0) and "measured" in src
    p1, src1 = ap.scale_period_nm(rings[:1], 186.0)
    assert p1 == 186.0 and "fallback" in src1


# =========================================================================== closed forms
def test_d_equivalents_and_valley_depth() -> None:
    sp, depth = ap.d_equivalents(2.2)
    assert sp == pytest.approx(0.3214, abs=1e-4) and depth == pytest.approx(0.41, abs=0.005)
    sp, depth = ap.d_equivalents(2.0)
    assert sp == pytest.approx(0.3536, abs=1e-4) and depth == pytest.approx(0.28, abs=0.005)
    assert ap.valley_depth_equal(0.30) == pytest.approx(0.50, abs=0.005)
    assert ap.valley_depth_equal(0.36) == pytest.approx(0.25, abs=0.01)
    assert ap.valley_depth_equal(0.50) == pytest.approx(0.0, abs=1e-12)
    # sqrt 2 is the equal-weight bimodality floor: sigma / P = 0.5 there, no valley
    assert ap.d_equivalents(math.sqrt(2.0))[1] == pytest.approx(0.0, abs=1e-12)


def test_copy_tolerance_formula() -> None:
    k, length, tau = 40, 3000.0, 35.0
    t = ap.copy_tolerance(k, k, length, length, tau)
    p0 = 1.0 - math.exp(-2.0 * tau * k / length)
    sd = math.sqrt(p0 * (1 - p0) / k)
    assert t.per_pair and t.p0 == pytest.approx(p0) and t.sd_star == pytest.approx(sd)
    assert t.viable == pytest.approx(0.25 * sd / (2 * (1 - p0)))
    assert t.marginal == pytest.approx(2.0 * t.viable)
    # the fixed 2 % is the 0.25 sd tolerance of rings of ~50-60 clusters at p0 ~ 0.55
    p0 = 0.55
    for kk in (50, 60):
        c = 0.25 * math.sqrt(p0 * (1 - p0) / kk) / (2 * (1 - p0))
        assert 0.016 <= c <= 0.022, (kk, c)
    # unknown counts, lengths or tau: the fixed suggestion, said in words
    for args in ((0, 40, 3000.0, 3000.0, 35.0), (40, 40, float("nan"), 3000.0, 35.0), (40, 40, 3000.0, 3000.0, None)):
        u = ap.copy_tolerance(*args)
        assert not u.per_pair and (u.viable, u.marginal) == (0.02, 0.05) and "fixed suggestion" in u.source
    # a larger tolerance (more chance matches) leaves room for more copies
    assert ap.copy_tolerance(40, 40, 3000.0, 3000.0, 80.0).viable > t.viable


def test_guard_star() -> None:
    z = ap.guard_z_star()
    assert z == pytest.approx(1.92, abs=0.01)
    assert ap.guard_star_nm(0.25 * 170, 170.0) < 0                 # no guard needed
    assert ap.guard_star_nm(0.30 * 170, 170.0) == pytest.approx(25.9, abs=1.0)
    assert ap.guard_star_nm(0.40 * 170, 170.0) == pytest.approx(91.2, abs=1.5)


# =========================================================================== synthetic rings
def _ring_res(centres: Sequence[float], n_clusters: Sequence[int], *, sigma: float = 25.0, locs: int = 120,
              seed: int = 0, ring_length: float = 3000.0, lpz: float = 25.0) -> Any:
    """A RingsResult look-alike: rings of compact clusters (2D sd 8 nm on a circle of ``ring_length``) with z' around
    each ring centre."""
    rng = np.random.default_rng(seed)
    x: List[float] = []
    y: List[float] = []
    z: List[float] = []
    rings = []
    radius = ring_length / (2 * math.pi)
    for k, (c, kc) in enumerate(zip(centres, n_clusters)):
        clusters = []
        loc_ring: List[int] = []
        for j in range(kc):
            ang = 2 * math.pi * (j + 0.5 * k) / kc
            cx, cy = radius * math.cos(ang), radius * math.sin(ang)
            cz = c + rng.normal(0.0, 5.0)
            start = len(x)
            x.extend(rng.normal(cx, 8.0, locs))
            y.extend(rng.normal(cy, 8.0, locs))
            zz = rng.normal(cz, sigma, locs)
            z.extend(zz)
            idx = np.arange(start, start + locs)
            loc_ring.extend(idx.tolist())
            clusters.append(SimpleNamespace(n_locs=locs, loc_index=idx, z_values_nm=np.asarray(zz),
                                            lpz_median_nm=lpz, centroid_nm=np.array([cx, cy])))
        rings.append(SimpleNamespace(
            index=k, component_index=k, centre_z_nm=float(c), sigma_z_nm=float(math.hypot(sigma, 5.0)),
            lpz_median_nm=lpz, z_lo_nm=c - 85.0, z_hi_nm=c + 85.0, relative_depth_lo=0.5, relative_depth_hi=0.5,
            loc_index=np.asarray(loc_ring), clusters=clusters, n_events=None, length_nm=ring_length))
    zr = SimpleNamespace(means_nm=np.asarray(centres, float), sigmas_nm=np.full(len(centres), math.hypot(sigma, 5.0)),
                         weights=np.full(len(centres), 1.0 / len(centres)))
    return SimpleNamespace(rings=rings, z_p=np.asarray(z), x_p=np.asarray(x), y_p=np.asarray(y), event_id=None,
                           ms=SimpleNamespace(z_result=zr),
                           params=SimpleNamespace(min_samples=10, eps_nm=25.0, random_seed=0))


def test_z_quality_uses_the_pairs_own_tolerance() -> None:
    res = _ring_res([0.0, 172.0, 344.0], [30, 30, 30])
    zq = ap.z_quality(res, params=ap.ViabilityParams(tau_nm=35.0))
    for b in zq.boundaries:
        t = ap.copy_tolerance(30, 30, 3000.0, 3000.0, 35.0)
        assert b.spur_viable == pytest.approx(t.viable) and b.spur_marginal == pytest.approx(t.marginal)
        assert "per pair" in b.tolerance_source
    assert zq.tau_nm == 35.0 and zq.p_ref_nm == 170.0
    # a pair is VIABLE only under its own tolerance: a copy fraction just above it is MARGINAL
    b = zq.boundaries[0]
    v, _r = ap.boundary_verdict(d_sep=3.0, exp_spur_frac=b.spur_viable * 1.01, info_ratio=float("nan"),
                                valley_depth=0.5, gap_nm=172.0, ring_flags=((), ()), rings=(0, 1), p_ref_nm=170.0,
                                spur_viable=b.spur_viable, spur_marginal=b.spur_marginal)
    assert v == ap.VERDICT_MARGINAL
    # the 'all pairs' aggregation reads the worst pair
    assert ap.axon_verdict(["viable", "not viable"], "any") == "viable"
    assert ap.axon_verdict(["viable", "not viable"], "all") == "not viable"


def test_viability_v2_scales_with_the_axons_period() -> None:
    res = _ring_res([0.0, 190.0, 380.0], [30, 30, 30])
    v2 = ap.viability_v2(res, params=ap.ViabilityParams(tau_nm=35.0), reference_readings=False)
    assert v2.p_nm == pytest.approx(190.0) and "measured" in v2.p_source and v2.p_ref_nm == 170.0
    assert v2.h_nm == pytest.approx(0.10 * 190.0)
    assert len(v2.sizer_map) == len(ap.SIZER_MAP_H_OVER_P)
    assert all(len(r.pair_valley) == 2 and len(r.ring_peak) == 3 for r in v2.sizer_map)
    assert all(len(p.valley_by_h) == len(ap.SIZER_MAP_H_OVER_P) for p in v2.pairs)
    # the row of the map at the verdict's bandwidth agrees with the verdict's valley
    row = [r for r in v2.sizer_map if abs(r.h_over_p - 0.10) < 1e-12][0]
    assert list(row.pair_valley) == [p.valley for p in v2.pairs]
    v2b = ap.viability_v2(res, params=ap.ViabilityParams(tau_nm=35.0, h_over_p=0.15), sizer_map=False,
                          reference_readings=False)
    assert v2b.h_nm == pytest.approx(0.15 * 190.0) and v2b.sizer_map == ()
    # the reference ring and the x statistic
    v2c = ap.viability_v2(res, params=ap.ViabilityParams(reference_ring="pick", reference_ring_index=0,
                                                         x_statistic="median"), sizer_map=False,
                          reference_readings=False)
    assert v2c.central_index == 0 and v2c.x == pytest.approx(120 / 10)
    # the groups' z scale from the file's precision ratio
    v2d = ap.viability_v2(res, precision_nm=(10.0, 25.0), sizer_map=False, reference_readings=False)
    assert v2d.group_z_scale == pytest.approx(0.4) and "from the file" in v2d.group_z_scale_source
    assert ap.viability_v2(res, sizer_map=False, reference_readings=False).group_z_scale == 0.5


def test_limiting_factors_name_the_period_and_the_marks() -> None:
    res = _ring_res([0.0, 172.0], [20, 20])
    v2 = ap.viability_v2(res, sizer_map=False, reference_readings=False)
    lf = {f.key: f for f in v2.limiting_factors}
    assert lf["axial_resolution"].label == "Ring width / period (sigma_z / P)"
    assert "measured on this axon" in lf["axial_resolution"].value
    assert "50 % deep" in lf["axial_resolution"].needed
    texts = " ".join(f.value + f.needed + f.source + f.label for f in v2.limiting_factors) + v2.rule
    for private in ("zmin", "D-39", "D-41", "Q-35", "Q-39"):
        assert private not in texts, private


def test_count_floor_by_thinning() -> None:
    dim = _ring_res([0.0, 172.0, 344.0], [25, 25, 25], locs=30, seed=1)
    bright = _ring_res([0.0, 172.0, 344.0], [25, 25, 25], locs=150, seed=1)
    c_dim = ap.calibrate_count_floor(dim, replicates=10)
    c_bright = ap.calibrate_count_floor(bright, replicates=10)
    assert c_dim.k_clusters == 25 and c_dim.ring_index == 1 and len(c_dim.recall_mean) == len(c_dim.f_grid)
    assert c_dim.recall_mean[0] == pytest.approx(1.0)
    assert all(a >= b - 0.06 for a, b in zip(c_dim.recall_mean, c_dim.recall_mean[1:]))   # falls with thinning
    # brighter clusters survive more thinning: a lower floor
    assert c_bright.f_min < c_dim.f_min <= 1.0
    # the calibration replaces the fitted line in the verdict
    v2 = ap.viability_v2(dim, fmin_calibration=c_dim, sizer_map=False, reference_readings=False)
    assert v2.f_min == pytest.approx(c_dim.f_min) and v2.fmin_source == ap.FMIN_SOURCE_THINNING
    again = ap.calibrate_count_floor(dim, replicates=10)
    assert again.recall_mean == c_dim.recall_mean                                    # deterministic


def test_silverman_reading() -> None:
    rng = np.random.default_rng(3)
    bimodal = np.concatenate([rng.normal(0, 30, 800), rng.normal(170, 30, 800)])
    unimodal = rng.normal(0, 60, 1600)
    h_b, p_b = ap.silverman_reading(bimodal, n_boot=99)
    h_u, p_u = ap.silverman_reading(unimodal, n_boot=99)
    assert h_b > h_u and p_b < 0.05 < p_u
    assert all(math.isnan(v) for v in ap.silverman_reading(np.zeros(5)))
    assert ap.peak_location_sd_nm(51.0, 17.0, 100) > 0


# =========================================================================== leak diagnostics
def _matches_res(sizes_a: Sequence[int], sizes_b: Sequence[int]) -> Any:
    rings = [SimpleNamespace(index=0, clusters=[SimpleNamespace(n_locs=n) for n in sizes_a]),
             SimpleNamespace(index=1, clusters=[SimpleNamespace(n_locs=n) for n in sizes_b])]
    return SimpleNamespace(rings=rings)


def test_size_ratio_permutation() -> None:
    rng = np.random.default_rng(5)
    k = 40
    sizes_a = rng.lognormal(np.log(150), 0.7, k).astype(int) + 10
    sizes_b = rng.lognormal(np.log(150), 0.7, k).astype(int) + 10
    # leak: 12 matched children are thinned copies (~8 %) of their parents
    copies = sizes_b.copy()
    copies[:12] = np.maximum((0.08 * sizes_a[:12]).astype(int), 10)
    m = SimpleNamespace(ring_a=0, ring_b=1, usable_a=None, usable_b=None, i_a=np.arange(12), j_b=np.arange(12))
    obs, null, p, per = ml.size_ratio_permutation(_matches_res(sizes_a, copies), [m], 999, 0)
    assert obs < null and p < 0.01 and per[(0, 1)][1] == pytest.approx(p)
    # genuine partners of the same dispersion: the p is not small
    obs2, null2, p2, _per2 = ml.size_ratio_permutation(_matches_res(sizes_a, sizes_b), [m], 999, 0)
    assert p2 > 0.05
    assert math.isnan(ml.size_ratio_permutation(_matches_res([], []), [], 99, 0)[2])


def test_guard_grid_and_leak_params() -> None:
    p = ml.LeakParams()
    assert p.guards_nm is None and p.guards_seff == (0.0, 0.5, 1.0, 1.5) and p.shared_threshold is None
    assert not hasattr(p, "guard_rule_nm") and p.profile_pi_min == 0.15 and p.size_ratio_n_perm == 999
    s_eff, period = 51.0, 170.0
    g_star = ap.guard_star_nm(s_eff, period)
    grid = ml.guard_grid_nm(p, s_eff, g_star, period)
    assert grid == tuple(sorted({0.0, 25.5, 51.0, 76.5, round(g_star, 1)}))
    assert ml.guard_grid_nm(ml.LeakParams(guards_nm=(0.0, 30.0)), s_eff, g_star, period) == (0.0, 30.0)
    assert ml.guard_grid_nm(p, float("nan"), float("nan"), period) == ()
    rings = [SimpleNamespace(sigma_z_nm=60.0, lpz_median_nm=30.0), SimpleNamespace(sigma_z_nm=50.0,
                                                                                    lpz_median_nm=float("nan"))]
    assert ml.axon_s_eff_nm(rings) == pytest.approx(55.0)
    with pytest.raises(ValueError):
        ml.LeakParams(shared_threshold=1.5)


def test_leak_texts_have_no_private_figures() -> None:
    import inspect
    src = inspect.getsource(ml)
    for private in ("sigma_z = 77", "lpz 77 nm", "(85 + 30) / 77", "03_plan", "01_formalizacion"):
        assert private not in src, private


# =========================================================================== selection and the dialog
def test_selection_reads_the_pairs_tolerance() -> None:
    from tools import mps_selection as ms
    res = _ring_res([0.0, 172.0, 344.0], [30, 30, 30])
    vp = ap.ViabilityParams(tau_nm=35.0)
    v2 = ap.viability_v2(res, ap.z_quality(res, params=vp), params=vp, sizer_map=False, reference_readings=False)
    crit = ms.axon_criteria(v2, None, "not requested")
    for pc, p in zip(crit.pairs, v2.pairs):
        assert pc.spur_viable == pytest.approx(p.spur_viable) and pc.p_nm == pytest.approx(v2.p_nm)
    r = ms.evaluate_selection(crit, ms.DEFAULT_SELECTION)
    assert [x.verdict for x in r.pairs] == [p.verdict for p in v2.pairs]
    assert [tuple(x.reasons) for x in r.pairs] == [tuple(p.reasons) for p in v2.pairs]   # verbatim under the default
    back = ms.AxonCriteria.from_rows(crit.to_rows())
    assert [pc.spur_viable for pc in back.pairs] == [pc.spur_viable for pc in crit.pairs]
    for w in (ms.WARNING_D39, ms.WARNING_TNFIX, ms.WARNING_EXPLORATORY, *ms.LEAK_ESTIMATE_LABELS.values()):
        assert "Q-39" not in w and "pre-specified" not in w


def test_criteria_dialog_offscreen() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    from tools.mps_viability_params_ui import ViabilityParamsDialog
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dlg = ViabilityParamsDialog(ap.ViabilityParams())
    assert dlg.params() == ap.ViabilityParams()
    dlg.preset_combo.setCurrentIndex(1)                      # D'Este 2016, 186 nm
    p = dlg.params()
    assert p.p_ref_nm == 186.0 and "D'Este" in p.p_ref_source
    dlg.spins["d_viable"].setValue(3.5)                      # outside 1.41-3.0: kept and flagged
    assert dlg.params().d_viable == 3.5 and "d_viable" in dlg.out_of_range()
    assert dlg.hints["d_viable"].outside and "outside" in dlg.problem_label.text()
    assert "valley" in dlg.equivalent_d_marginal.text()
    dlg.spins["p_ref_nm"].setValue(175.0)
    assert dlg.params().p_ref_source == "user: 175 nm" and dlg.preset_combo.currentData() is None
    dlg.z_scale_from_file.setChecked(False)
    dlg.z_scale_spin.setValue(0.45)
    assert dlg.params().group_z_scale == pytest.approx(0.45)
    dlg.reset_button.click()
    assert dlg.params() == ap.ViabilityParams()
    dlg.deleteLater()
    app.processEvents()


def test_zquality_window_criteria_and_calibration(monkeypatch: Any) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    from tools import mps_selection as ms
    from tools import mps_zquality_window as zw
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # never the user's settings file: the stored parameters are the defaults here
    monkeypatch.setattr(ms, "stored_viability_params", lambda: ap.ViabilityParams(tau_nm=35.0))
    res = _ring_res([0.0, 172.0, 344.0], [30, 30, 30], locs=40)
    xyz = (res.x_p, res.y_p, res.z_p)
    rep = zw.compute_z_quality(res, xyz, source_name="sim.npz", precision_nm=(10.0, 25.0))
    assert rep.params == ap.ViabilityParams(tau_nm=35.0) and rep.v2.group_z_scale == pytest.approx(0.4)
    w = zw.ZQualityWindow(title_source="sim.npz")
    w.set_report(rep)
    assert w.criteria_button.isEnabled() and w.calibrate_button.isEnabled()
    header = w.header_label.text()
    assert "P = 172.0 nm (measured on this axon)" in header and "P_ref = 170 nm" in header
    tol_col = zw.PAIR_COLUMNS.index("(4) Tolerated")
    assert "%" in w.pair_table.item(0, tol_col).text()
    w.recompute(calibrate=True)
    w.wait(120.0)
    for _ in range(50):
        app.processEvents()
    assert w.report is not rep and w.report.fmin_calibration is not None
    fmin_col = zw.PAIR_COLUMNS.index("(3) Minimum f_min(x)")
    assert "calibrated" in w.pair_table.item(0, fmin_col).text()
    w.recompute(ap.ViabilityParams(tau_nm=35.0, h_over_p=0.15))
    w.wait(120.0)
    for _ in range(50):
        app.processEvents()
    assert w.report.v2.h_nm == pytest.approx(0.15 * 172.0)
    assert w.report.fmin_calibration is not None                     # same recall: the calibration is kept
    summary = "\n".join(zw.plain_summary(w.report))
    assert "SIZER MAP" in summary and "calibrated on this axon" in summary
    for private in ("zmin", "D-41", "Q-35", "D-39"):
        assert private not in summary, private
    w.close()
    w.deleteLater()
    app.processEvents()


def test_exploration_log_counts_across_sessions(tmp_path: Any) -> None:
    from tools import mps_selection as ms
    path = str(tmp_path / "log.csv")
    for sid in ("s1", "s2"):
        log = ms.ExplorationLog(path, session_id=sid)
        for spec in (ms.DEFAULT_SELECTION, ms.SelectionSpec.from_label("v2[peak,valley,count]")):
            log.log_once(("batch", sid, spec.hash), ms.log_row(source="batch", spec=spec, axon_id="a"))
    log = ms.ExplorationLog(path, session_id="s2")
    assert log.variants_tried_all_sessions() == 2 and log.variants_tried_all_sessions("review") == 0
