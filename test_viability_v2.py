# -*- coding: utf-8 -*-
"""
Tests of the ring-pair viability rule v2 (D-41: cluster-robust SiZer peak and valley, localization minimum as a
fraction of the central ring, D-39 leak tiers) in ``tools.mps_sizer`` and ``tools.mps_axial_precision``.

- Parity of the port with the zmin research code (private: sizer_row / peaks_valleys, sign_row / valley_between,
  groups_3d) on fixed synthetic inputs stored in
  ``testdata/viability_v2/sizer_parity.json`` (written by the research functions; exact or 1e-12).
- Closed-form / known-truth cases: one ring has no valley, two separated rings have one, the cluster-robust variance
  equals the iid one with singleton clusters and does not change when every localization is repeated inside its
  cluster (the iid one does).
- f_min(x) with its clamps and flags, the central ring for 1 to 5 rings, the leak tiers, and the whole rule on a
  hand-built ``RingsResult`` look-alike.

Nothing here reads a real axon (the exploratory dataset is checked, geometry only, by a private validator).

Run: ``py -3 -m pytest test_viability_v2.py -q``.
"""
from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

import numpy as np
import pytest

from tools import mps_axial_precision as ap
from tools import mps_sizer as sz

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY = os.path.join(HERE, "testdata", "viability_v2", "sizer_parity.json")
P = ap.P_REF_DEFAULT_NM


@pytest.fixture(scope="module")
def parity() -> Any:
    with open(PARITY, encoding="utf-8") as fh:
        return json.load(fh)


# =========================================================================== parity with the research code
def test_sizer_row_parity(parity: Any) -> None:
    for case in parity["cases"]:
        z = np.asarray(case["z"], float)
        g = np.asarray(case["groups"], np.int64)
        for run in case["runs"]:
            grid = np.asarray(run["grid"], float)
            h = run["c"] * parity["p_ref_nm"]
            np.testing.assert_allclose(sz.sizer_grid(z, h), grid, rtol=0, atol=1e-12)
            r = sz.sizer_row(z, h, grid, g if run["robust"] else None, 0.05, robust=run["robust"])
            assert r.sign.tolist() == run["sign"], (case["name"], run["c"], run["robust"])
            np.testing.assert_allclose(r.fhat, run["fhat"], rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(r.t, run["t"], rtol=1e-12, atol=1e-12)
            assert r.q == pytest.approx(run["qv"], rel=1e-12) and r.l == pytest.approx(run["l"], rel=1e-12)
            pk, vl = sz.peaks_valleys(r.sign, r.fhat, grid)
            assert pk == pytest.approx(run["peaks_row"], abs=1e-12) and vl == pytest.approx(run["valleys_row"], abs=1e-12)


def test_sizer_profile_and_valley_parity(parity: Any) -> None:
    for case in parity["cases"]:
        z = np.asarray(case["z"], float)
        g = np.asarray(case["groups"], np.int64)
        for run in case["runs"]:
            prof = sz.sizer_profile(z, g, p_ref_nm=parity["p_ref_nm"], c=run["c"], robust=run["robust"], q=run["q"])
            assert prof.sign.tolist() == run["sign_q"], (case["name"], run["c"], run["robust"], run["q"])
            assert list(prof.peaks) == pytest.approx(run["peaks"], abs=1e-12)
            assert list(prof.valleys) == pytest.approx(run["valleys"], abs=1e-12)
            cen = run["centres"]
            got = [sz.valley_between(prof.grid, prof.sign, a, b) for a, b in zip(cen[:-1], cen[1:])]
            assert got == run["valley_between"]


def test_groups_3d_parity(parity: Any) -> None:
    xyz = np.asarray(parity["groups3d"]["xyz"], float)
    lab = sz.groups_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    assert lab.tolist() == parity["groups3d"]["labels"]
    # noise points are singleton groups with labels after the clusters
    n_cl = int(np.max(lab[: 100])) + 1
    assert len(set(lab.tolist())) >= n_cl


# =========================================================================== closed form / known truth
def _clustered(rng: np.random.Generator, centres: Sequence[float], n_clusters: int, m: int, ring_sd: float,
               within_sd: float) -> "tuple[np.ndarray, np.ndarray]":
    z, g, gid = [], [], 0
    for c in centres:
        for zc in rng.normal(c, ring_sd, n_clusters):
            z.append(rng.normal(zc, within_sd, m))
            g.append(np.full(m, gid))
            gid += 1
    return np.concatenate(z), np.concatenate(g)


def test_one_ring_has_no_valley_two_separated_rings_have_one() -> None:
    rng = np.random.default_rng(1)
    z1, g1 = _clustered(rng, [0.0], 60, 40, 25.0, 10.0)
    p1 = sz.sizer_profile(z1, g1, p_ref_nm=P)
    assert len(p1.peaks) == 1 and p1.valleys == ()
    z2, g2 = _clustered(rng, [-P / 2, P / 2], 60, 40, 18.0, 10.0)
    p2 = sz.sizer_profile(z2, g2, p_ref_nm=P)
    assert len(p2.peaks) == 2 and len(p2.valleys) == 1 and abs(p2.valleys[0]) < 30.0
    assert sz.valley_between(p2.grid, p2.sign, -P / 2, P / 2)


def test_cluster_robust_variance_singletons_equal_iid() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(0.0, 60.0, 800)
    h = 0.1 * P
    grid = sz.sizer_grid(z, h)
    a = sz.sizer_row(z, h, grid, np.arange(z.size), 0.05, robust=True)
    b = sz.sizer_row(z, h, grid, None, 0.05, robust=False)
    np.testing.assert_allclose(a.t, b.t, rtol=1e-9, atol=1e-9)


def test_cluster_robust_variance_is_invariant_to_repeating_localizations_in_their_cluster() -> None:
    """m copies of every localization inside one cluster: the cluster-robust t is unchanged (each cluster is one unit),
    the iid t grows by sqrt(m)."""
    rng = np.random.default_rng(3)
    zu = np.concatenate([rng.normal(-80.0, 30.0, 150), rng.normal(90.0, 30.0, 150)])
    m = 7
    z = np.repeat(zu, m)
    g = np.repeat(np.arange(zu.size), m)
    h = 0.1 * P
    grid = sz.sizer_grid(zu, h)
    base = sz.sizer_row(zu, h, grid, np.arange(zu.size), 0.05, robust=True)
    rep = sz.sizer_row(z, h, grid, g, 0.05, robust=True)
    iid = sz.sizer_row(z, h, grid, None, 0.05, robust=False)
    ok = np.abs(base.t) > 1e-6
    np.testing.assert_allclose(rep.t[ok], base.t[ok], rtol=1e-9)
    np.testing.assert_allclose(iid.t[ok] / base.t[ok], math.sqrt(m), rtol=0.02)
    # q is not exactly invariant: the ESS >= 5 mask that sets l counts localizations (as in the research code)
    assert rep.q == pytest.approx(base.q, rel=0.05)


# =========================================================================== f_min(x)
def test_f_min_formula_and_reference_point() -> None:
    r = ap.f_min_of_x(8.1)
    assert r.f_min == pytest.approx(1.46 - 0.336 * math.log(8.1), rel=1e-12)
    assert 0.745 <= r.f_min <= 0.792 and r.in_range and r.note == ""      # the 77 % [74.5; 79.2] of the zmin report
    lo, hi = ap.f_min_of_x(5.8), ap.f_min_of_x(13.3)
    assert lo.in_range and hi.in_range
    assert lo.f_min == pytest.approx(0.8693, abs=5e-4) and hi.f_min == pytest.approx(0.5905, abs=5e-4)


def test_f_min_clamps_and_flags() -> None:
    above = ap.f_min_of_x(20.9)
    assert not above.in_range and above.f_min == pytest.approx(ap.f_min_of_x(13.3).f_min, rel=1e-12)
    assert "outside calibrated range" in above.note
    below = ap.f_min_of_x(4.5)
    assert not below.in_range and below.f_min == pytest.approx(1.46 - 0.336 * math.log(4.5), rel=1e-12)
    assert below.f_min < 1.0 and "outside calibrated range" in below.note
    capped = ap.f_min_of_x(3.5)
    assert not capped.in_range and capped.f_min == 1.0 and "100 %" in capped.note
    bad = ap.f_min_of_x(float("nan"))
    assert math.isnan(bad.f_min) and not bad.in_range and bad.note


# =========================================================================== central ring (Q-35)
@pytest.mark.parametrize("n_locs, expected", [
    ([], None), ([500], 0), ([100, 300], 1), ([300, 100], 0), ([200, 200], 0),
    ([10, 1, 10], 1), ([1, 99, 5], 1), ([10, 50, 80, 5], 2), ([10, 80, 50, 5], 1), ([9, 9, 9, 9], 1),
    ([1, 2, 3, 4, 5], 2), ([1, 9, 2, 8, 3, 7], 3),
])
def test_central_ring_position(n_locs: List[int], expected: Optional[int]) -> None:
    assert ap.central_ring_position(n_locs) == expected


# =========================================================================== verdict tiers (criterion 4)
def _verdict(spur: float, *, peak_a: bool = True, peak_b: bool = True, valley: bool = True, f_a: float = 1.0,
             f_b: float = 0.9, f_min: float = 0.77) -> "tuple[str, tuple[str, ...]]":
    return ap.verdict_v2(rings=(0, 1), peak_a=peak_a, peak_b=peak_b, valley=valley, f_a=f_a, f_b=f_b, f_min=f_min,
                         exp_spur_frac=spur)


def test_leak_tiers() -> None:
    assert _verdict(0.0)[0] == ap.VERDICT_VIABLE
    assert _verdict(0.02)[0] == ap.VERDICT_VIABLE
    v, reasons = _verdict(0.03)
    assert v == ap.VERDICT_MARGINAL and any("sensitivity only" in r for r in reasons)
    assert _verdict(0.05)[0] == ap.VERDICT_MARGINAL
    v, reasons = _verdict(0.06)
    assert v == ap.VERDICT_NOT_VIABLE and any("6.0 %" in r for r in reasons)
    assert _verdict(float("nan"))[0] == ap.VERDICT_NOT_VIABLE
    # a pair failing (1)-(3) is NOT VIABLE: its 2-5 % leak reason must not call it marginal (review of H6)
    v, reasons = _verdict(0.036, valley=False)
    assert v == ap.VERDICT_NOT_VIABLE and any("3.6 %" in r for r in reasons)
    assert not any("marginal" in r for r in reasons), reasons


def test_criteria_1_to_3_gate_every_tier() -> None:
    for kw in (dict(peak_a=False), dict(peak_b=False), dict(valley=False), dict(f_b=0.5)):
        v, reasons = _verdict(0.0, **kw)
        assert v == ap.VERDICT_NOT_VIABLE and reasons
    v, reasons = _verdict(0.0, f_b=0.5)
    assert any("50 %" in r and "77 %" in r for r in reasons)
    v, reasons = _verdict(0.0, f_min=float("nan"))
    assert v == ap.VERDICT_NOT_VIABLE
    assert _verdict(0.0, f_a=1.33)[0] == ap.VERDICT_VIABLE           # more populated than the central ring is fine


# =========================================================================== the whole rule on a look-alike
def _fake_res(centres: Sequence[float], n_clusters: Sequence[int], *, m: int = 60, ring_sd: float = 18.0,
              within_sd: float = 10.0, lpz: Sequence[float] = (), min_samples: int = 10, seed: int = 5) -> Any:
    """Rings at ``centres`` (bottom to top) made of ``n_clusters[i]`` clusters of ``m`` localizations each; the
    groups are the clusters. Returned with the groups and a D-39-like ``zq`` whose expected copies are 0."""
    rng = np.random.default_rng(seed)
    z_all, g_all, rings, start, gid = [], [], [], 0, 0
    for i, (c, k) in enumerate(zip(centres, n_clusters)):
        zs, clusters = [], []
        for zc in rng.normal(c, ring_sd, k):
            zz = rng.normal(zc, within_sd, m)
            zs.append(zz)
            g_all.append(np.full(m, gid))
            gid += 1
        z = np.concatenate(zs)
        li = np.arange(start, start + z.size)
        clusters = [SimpleNamespace(n_locs=m, loc_index=li[j * m:(j + 1) * m]) for j in range(k)]
        start += z.size
        z_all.append(z)
        rings.append(SimpleNamespace(index=i, centre_z_nm=float(c), sigma_z_nm=float(np.std(z)),
                                     lpz_median_nm=float(lpz[i]) if lpz else 40.0 + 5 * i, loc_index=li,
                                     clusters=clusters))
    z_p = np.concatenate(z_all)
    res = SimpleNamespace(rings=rings[::-1], z_p=z_p, x_p=np.zeros_like(z_p), y_p=np.zeros_like(z_p),
                          params=SimpleNamespace(min_samples=min_samples))
    zq = SimpleNamespace(boundaries=tuple(SimpleNamespace(ring_a=i, ring_b=i + 1, exp_spur_frac=0.0, d_sep=3.0,
                                                          verdict=ap.VERDICT_VIABLE, reasons=())
                                          for i in range(len(centres) - 1)),
                         rings=tuple(SimpleNamespace(index=i, sigma_over_p=float(r.sigma_z_nm) / P) for i, r in enumerate(rings)))
    return res, np.concatenate(g_all), zq


def test_rule_v2_on_three_separated_rings() -> None:
    res, g, zq = _fake_res([-P, 0.0, P], [40, 40, 36], lpz=(60.0, 50.0, 38.0))
    v = ap.viability_v2(res, zq, groups=g)
    assert v.rule == ap.VIABILITY_RULE_V2 and v.central_index == 1 and v.in_focus_index == 2
    assert [r.is_central for r in v.rings] == [False, True, False]
    assert [r.is_in_focus for r in v.rings] == [False, False, True]
    assert all(r.peak for r in v.rings)
    assert v.x == pytest.approx(60 / 10) and v.x_in_range
    assert [p.verdict for p in v.pairs] == [ap.VERDICT_VIABLE, ap.VERDICT_VIABLE]
    assert v.pairs[1].f_b == pytest.approx(36 / 40) and v.viable_pairs == ((0, 1), (1, 2))
    assert v.groups_source == "given"
    keys = {f.key for f in v.limiting_factors}
    assert {"axial_resolution", "locs_per_cluster", "z_truncation", "focus"} <= keys
    assert ap.viability_v2_key(v, "viable") == (3, ((0, 1), (1, 2)))


def test_rule_v2_weak_edge_ring_fails_the_localization_minimum() -> None:
    res, g, zq = _fake_res([-P, 0.0, P], [40, 40, 14])                   # 35 % of the central ring, f_min(6) ~ 86 %
    v = ap.viability_v2(res, zq, groups=g)
    assert v.pairs[0].verdict == ap.VERDICT_VIABLE
    p = v.pairs[1]
    assert p.verdict == ap.VERDICT_NOT_VIABLE and not p.count_ok
    assert any("of the central ring" in r for r in p.reasons)


def test_rule_v2_merged_rings_have_no_valley() -> None:
    res, g, zq = _fake_res([-40.0, 40.0], [40, 40], ring_sd=45.0)
    v = ap.viability_v2(res, zq, groups=g)
    assert v.pairs[0].verdict == ap.VERDICT_NOT_VIABLE and not v.pairs[0].valley
    assert any("valley" in r for r in v.pairs[0].reasons)


def test_rule_v2_one_ring_has_no_pairs() -> None:
    res, g, zq = _fake_res([0.0], [40])
    v = ap.viability_v2(res, zq, groups=g)
    assert v.pairs == () and v.central_index == 0 and v.viable_pairs == ()


def test_rule_v2_leak_from_z_quality() -> None:
    res, g, zq = _fake_res([-P / 2, P / 2], [40, 40])
    zq.boundaries[0].exp_spur_frac = 0.04
    v = ap.viability_v2(res, zq, groups=g)
    assert v.pairs[0].verdict == ap.VERDICT_MARGINAL and v.marginal_pairs == ((0, 1),)
    assert ap.viability_v2_key(v, "viable") == (2, ())
    assert ap.viability_v2_key(v, "viable+marginal") == (2, ((0, 1),))


def test_rule_v2_groups_from_lab_coordinates() -> None:
    res, g, zq = _fake_res([-P / 2, P / 2], [30, 30])
    rng = np.random.default_rng(9)
    # every cluster a tight 3D blob far from the others: DBSCAN recovers the clusters as groups
    centres = rng.uniform(-3000, 3000, (int(g.max()) + 1, 2))
    x = centres[g, 0] + rng.normal(0, 3, g.size)
    y = centres[g, 1] + rng.normal(0, 3, g.size)
    v = ap.viability_v2(res, zq, xyz_lab_nm=(x, y, res.z_p))
    w = ap.viability_v2(res, zq, groups=g)
    assert v.groups_source == "lab coordinates (3D DBSCAN)"
    assert [p.verdict for p in v.pairs] == [p.verdict for p in w.pairs]


def test_simnull_selection_modes_follow_rule_v2() -> None:
    """simnull --zq-select: the v2-* modes key on rule v2 (observed and replicates alike), the D-39 modes and "off"
    are unchanged."""
    import power_columns as pc
    assert pc.ZQ_SELECT_CHOICES == ("off", "viable", "viable+marginal", "v2-viable", "v2-viable+marginal")
    assert [pc.zq_base_mode(m) for m in pc.ZQ_SELECT_CHOICES] == ["viable", "viable", "viable+marginal", "viable",
                                                                    "viable+marginal"]
    res, g, _zq = _fake_res([-P / 2, P / 2], [40, 40])
    zq: Any = SimpleNamespace(boundaries=(SimpleNamespace(ring_a=0, ring_b=1, exp_spur_frac=0.03, d_sep=2.5,
                                                     verdict=ap.VERDICT_VIABLE, reasons=()),),
                         rings=(SimpleNamespace(index=0, sigma_over_p=0.1), SimpleNamespace(index=1, sigma_over_p=0.1)),
                         viable_pairs=((0, 1),), marginal_pairs=(), axon_verdict=ap.VERDICT_VIABLE)
    v2 = ap.viability_v2(res, zq, groups=g)                               # leak 3 %: marginal under rule v2
    assert v2.marginal_pairs == ((0, 1),)
    with pytest.raises(ValueError):
        pc.zq_row_values(zq, "v2-viable")                                 # rule v2 needs its analysis
    assert pc.zq_row_values(zq, "v2-viable", v2=v2)["zq_key"] == "2|"
    assert pc.zq_row_values(zq, "v2-viable+marginal", (2, ((0, 1),)), v2=v2)["zq_accepted"] == 1
    assert pc.zq_row_values(zq, "viable")["zq_key"] == "2|0-1"           # the D-39 key, as before
    assert pc.zq_row_values(zq, "off")["zq_key"] == "2|0-1"


def test_z_edge_truncation() -> None:
    rng = np.random.default_rng(4)
    z = rng.normal(0.0, 120.0, 20000)
    cut = z[z > -200.0]
    lo, hi = ap.z_edge_truncation(cut)
    assert lo.truncated and lo.z_nm == pytest.approx(-200.0, abs=0.5) and not hi.truncated
    lo2, hi2 = ap.z_edge_truncation(z)
    assert not lo2.truncated and not hi2.truncated
