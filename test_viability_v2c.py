# -*- coding: utf-8 -*-
"""
Tests of rule v2-clusters (EXPLORATORY, D-42: the four criteria of rule v2 read on the CLUSTERS) in
``tools.mps_axial_clusters``.

- Parity of the port with the H6 clusters research code (private: tn_mean_fixed_sigma, leak_pair "tnfix",
  sizer_points + crit_12; at_cut) on fixed synthetic
  inputs stored in ``testdata/viability_v2c/parity.json`` (exact or 1e-12).
- valley_at_cut: a dip built by a slab cut is flagged; a valley in the middle of a 60 nm gap is flagged, one 60 nm
  outside it is not; for the ring-slab variants (A, C) a flagged valley never satisfies (2c), for B it is only flagged.
- The truncated-normal centre: an untruncated sample gives its mean; a sample cut at mu + 0.3 sigma gives mu back.
- 4c: clusters far from the cut leak nothing (its comparison with D-39 on the exploratory axon runs with the
  private research validators).
- The whole rule on a hand-built ``RingsResult`` look-alike: three variants, the verdict tiers, the export columns;
  with planar rings the true valley sits at the slab cut, so only B can reach VIABLE (A and C never do: SPEC R2).
- Q-38: no module that selects pairs (batch, simnull, the column review) uses rule v2-clusters.

Nothing here reads a real axon (the exploratory dataset is checked, geometry only, by a private validator).

Run: ``py -3 -m pytest test_viability_v2c.py -q``.
"""
from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from tools import mps_axial_clusters as ac
from tools import mps_sizer as sz

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY = os.path.join(HERE, "testdata", "viability_v2c", "parity.json")
P = 170.8      # the period the parity fixture was written with (tools.mps_axial_precision.P_REF_DEFAULT_NM)


@pytest.fixture(scope="module")
def parity() -> Any:
    with open(PARITY, encoding="utf-8") as fh:
        return json.load(fh)


# =========================================================================== parity with the research code
def test_tn_centre_parity(parity: Any) -> None:
    import scipy
    # exact with the scipy that wrote the data (the program's venv); another scipy's bounded minimizer stops at a
    # slightly different point inside its own tolerance (xatol 1e-5): 1.3e-5 nm seen between scipy 1.17 and 1.18
    tol = 1e-12 if scipy.__version__ == parity["scipy_version"] else 1e-4
    for c in parity["cases"]["tn"]:
        got = ac.tn_centre_fixed_sigma(np.asarray(c["z"], float), c["sigma"], c["lo"], c["hi"])
        assert got == pytest.approx(c["mu"], abs=tol), c


def test_leak_tnfix_parity(parity: Any) -> None:
    for c in parity["cases"]["leak"]:
        got = ac.exp_spur_frac_clusters(c["n_a"], c["mu_a"], c["n_b"], c["mu_b"], sigma_a_nm=c["sigma_a"],
                                        sigma_b_nm=c["sigma_b"], slab_a=tuple(c["slab_a"]),
                                        slab_b=tuple(c["slab_b"]), cut_nm=c["cut"], min_samples=c["ms"])
        if c["e"] is None:
            assert math.isnan(got)
        else:
            assert got == pytest.approx(c["e"], rel=1e-12, abs=1e-15), c


def test_sizer_on_points_parity(parity: Any) -> None:
    for c in parity["cases"]["sizer"]:
        s = ac.v2c_sizer(c["points"], c["centres"], p_ref_nm=parity["p_ref_nm"], h_over_p=c["c"])
        assert list(s.peaks) == pytest.approx(c["peaks"], abs=1e-12)
        assert list(s.valleys) == pytest.approx(c["valleys"], abs=1e-12)
        assert list(s.ring_peak) == c["ring_peak"]
        cen = c["centres"]
        pv = [bool(s.profile is not None and sz.valley_between(s.profile.grid, s.profile.sign, a, b))
              for a, b in zip(cen[:-1], cen[1:])]
        assert pv == c["pair_valley"]
        if len(c["points"]) < ac.V2C_MIN_POINTS:
            assert s.profile is None and s.peaks == () and s.valleys == ()


def test_valley_at_cut_parity(parity: Any) -> None:
    for c in parity["cases"]["cut"]:
        flag, between = ac.valley_at_cut(c["valleys"], c["lo_c"], c["hi_c"], c["gap_lo"], c["gap_hi"])
        assert flag == c["flag"] and list(between) == pytest.approx(c["between"], abs=0), c


# =========================================================================== valley_at_cut
def test_valley_in_gap_flags() -> None:
    # a 60 nm gap [0, 60] between centres -70 and 130: its middle (30 nm from both borders) is flagged
    assert ac.valley_at_cut([30.0], -70.0, 130.0, 0.0, 60.0) == (True, (30.0,))
    # the borders widened by 15 nm, both sides; 60 nm outside the gap is not flagged
    assert ac.valley_at_cut([-15.0], -70.0, 130.0, 0.0, 60.0)[0]
    assert ac.valley_at_cut([75.0], -70.0, 130.0, 0.0, 60.0)[0]
    assert ac.valley_at_cut([-60.0], -70.0, 130.0, 0.0, 60.0) == (False, (-60.0,))
    assert ac.valley_at_cut([120.0], -70.0, 130.0, 0.0, 60.0) == (False, (120.0,))
    # a valley outside the two centres is not "between" them
    assert ac.valley_at_cut([200.0], -70.0, 130.0, 0.0, 60.0) == (False, ())


def _slab_cut_medians(seed: int, n_clusters: int = 700) -> np.ndarray:
    """ONE broad band of cluster centres (N(0, 45) nm) cut at z' = 0 into two slabs: each cluster's localizations
    (N(centre, 30), 40 of them) on each side of the cut with >= 10 points give one median -- what DBSCAN inside each
    slab gives. No median can cross the cut, so a dip appears there by construction."""
    rng = np.random.default_rng(seed)
    med: List[float] = []
    for c in rng.normal(0.0, 45.0, n_clusters):
        z = rng.normal(c, 30.0, 40)
        for part in (z[z < 0.0], z[z >= 0.0]):
            if part.size >= 10:
                med.append(float(np.median(part)))
    return np.asarray(med)


def test_slab_cut_builds_a_flagged_valley() -> None:
    pts = _slab_cut_medians(7)
    centres = [-60.0, 60.0]
    s = ac.v2c_sizer(pts, centres, p_ref_nm=P, h_over_p=ac.V2C_H_OVER_P)
    assert s.profile is not None
    vraw = sz.valley_between(s.profile.grid, s.profile.sign, *centres)
    flag, between = ac.valley_at_cut(s.valleys, centres[0], centres[1], 0.0, 0.0)
    assert vraw and flag and between and all(abs(v) <= ac.V2C_CUT_FLAG_NM for v in between), (s.valleys, between)
    kw: Dict[str, Any] = dict(rings=(0, 1), peak_a=True, peak_b=True, valley_raw=vraw, valley_at_cut_=flag, k_ratio_a=1.0,
              k_ratio_b=1.0, k_a=10, k_b=10, k_central=10, exp_spur_frac_clusters_=0.0)
    # A and C: the built-in valley never counts for (2c)
    for variant in ac.V2C_CUT_FORCED:
        verdict, valley, count_ok, reasons = ac.verdict_v2c(variant=variant, **kw)
        assert verdict == "not viable" and not valley and count_ok
        assert any(r.startswith(ac.V2C_REASON_AT_CUT) for r in reasons)
    # B: flagged only
    verdict, valley, _ok, reasons = ac.verdict_v2c(variant="groups_3d", **kw)
    assert verdict == "viable" and valley and any("flagged only" in r for r in reasons)


# =========================================================================== truncated-normal centre
def test_tn_centre_untruncated_is_the_mean() -> None:
    rng = np.random.default_rng(3)
    for _ in range(5):
        z = rng.normal(40.0, 30.0, 200)
        assert ac.tn_centre_fixed_sigma(z, 30.0, -2000.0, 2000.0) == pytest.approx(float(z.mean()), abs=1.0)


def test_tn_centre_recovers_a_cut_cluster() -> None:
    mu, sd = 25.0, 40.0
    rng = np.random.default_rng(11)
    errs = []
    for _ in range(10):
        z = rng.normal(mu, sd, 4000)
        z = z[z <= mu + 0.3 * sd][:200]
        assert z.size == 200
        got = ac.tn_centre_fixed_sigma(z, sd, mu - 6 * sd, mu + 0.3 * sd)
        errs.append(got - mu)
        # the raw median sits well below mu: the correction is what moves it back
        assert float(np.median(z)) < mu - 0.4 * sd
    # n = 200 cut at mu + 0.3 sigma: within 0.2 sigma (9 of 10 replicates; the 10th 0.21 sigma), and no bias
    assert sum(abs(e) <= 0.2 * sd for e in errs) >= 9, errs
    assert abs(float(np.mean(errs))) <= 0.05 * sd and max(abs(e) for e in errs) <= 0.3 * sd, errs


# =========================================================================== criterion 4c
def test_leak_far_from_cut_is_zero() -> None:
    e = ac.exp_spur_frac_clusters([50] * 20, [-200.0] * 20, [50] * 20, [200.0] * 20, sigma_a_nm=30.0,
                                  sigma_b_nm=30.0, slab_a=(-400.0, 0.0), slab_b=(0.0, 400.0), cut_nm=0.0,
                                  min_samples=10)
    assert 0.0 <= e < 1e-12
    # a cluster sitting on the cut leaks half its localizations: a copy is almost certain
    e1 = ac.exp_spur_frac_clusters([100], [0.0], [], [], sigma_a_nm=30.0, sigma_b_nm=30.0, slab_a=(-400.0, 0.0),
                                   slab_b=(0.0, 400.0), cut_nm=0.0, min_samples=10)
    assert e1 > 0.99
    assert math.isnan(ac.exp_spur_frac_clusters([], [], [], [], sigma_a_nm=30.0, sigma_b_nm=30.0,
                                                slab_a=(0.0, 1.0), slab_b=(1.0, 2.0), cut_nm=1.0, min_samples=10))


# =========================================================================== the whole rule on a look-alike
def _cluster(z: np.ndarray) -> Any:
    return SimpleNamespace(z_values_nm=np.asarray(z, float), n_locs=int(np.asarray(z).size))


def _fake(seed: int, centres: List[float], k: List[int], sigma: float, tau: float, n_per: int = 60) -> Any:
    """Rings whose clusters are slabs of N(centre_c, sigma) localizations, the cluster centres N(ring centre, tau);
    the slabs cut halfway between centres. Returns (res, v2, xyz)."""
    rng = np.random.default_rng(seed)
    cuts = [0.5 * (a + b) for a, b in zip(centres[:-1], centres[1:])]
    lo = [-1e3] + cuts
    hi = cuts + [1e3]
    rings, zp, xs, ys = [], [], [], []
    for i, (c, kk) in enumerate(zip(centres, k)):
        cls = []
        for j in range(kk):
            z = rng.normal(rng.normal(c, tau), sigma, n_per)
            z = z[(z >= lo[i]) & (z < hi[i])]
            if z.size < 10:
                continue
            cls.append(_cluster(z))
            ang = 2 * math.pi * j / kk
            xs.append(np.full(z.size, 1500 * math.cos(ang)) + rng.normal(0, 3, z.size))
            ys.append(np.full(z.size, 1500 * math.sin(ang)) + rng.normal(0, 3, z.size))
            zp.append(z)
        rings.append(SimpleNamespace(index=i, centre_z_nm=c, z_lo_nm=lo[i], z_hi_nm=hi[i], sigma_z_nm=sigma,
                                     clusters=cls))
    z_all = np.concatenate(zp)
    res = SimpleNamespace(rings=rings, z_p=z_all, params=SimpleNamespace(min_samples=10, eps_nm=25.0))
    n = [sum(cl.n_locs for cl in r.clusters) for r in rings]
    from tools.mps_axial_precision import central_ring_position
    cpos = central_ring_position(n)
    v2 = SimpleNamespace(
        p_ref_nm=P, central_index=cpos,
        rings=[SimpleNamespace(index=r.index, centre_z_nm=r.centre_z_nm) for r in rings],
        pairs=[SimpleNamespace(ring_a=i, ring_b=i + 1, exp_spur_frac=0.001) for i in range(len(rings) - 1)])
    return res, v2, (np.concatenate(xs), np.concatenate(ys), z_all)


def test_rule_on_planar_separated_rings() -> None:
    # narrow, planar, well separated rings: every variant sees two peaks and a valley, and the leak is nil. The true
    # valley of two such rings lies halfway between them, where build_rings puts the slab cut: for the ring-slab
    # variants (A, C) it is indistinguishable from a built-in dip and never counts (SPEC R2, conservative), so only
    # B (slab-free 3D clusters, flagged only) can reach VIABLE.
    res, v2, xyz = _fake(5, [-100.0, 100.0], [120, 120], sigma=12.0, tau=0.0)
    out = ac.viability_v2c(res, v2, xyz_lab_nm=xyz)
    assert out.rule == ac.V2C_RULE == "rule v2-clusters (exploratory)"
    assert [p.variant for p in out.pairs] == list(ac.V2C_VARIANT_ORDER)
    assert len(out.sensitivity_pairs) == 3 and all(p.h_over_p == ac.V2C_H_TABLE for p in out.sensitivity_pairs)
    assert out.leaks[0].exp_spur_frac_clusters < 1e-6 and out.leaks[0].exp_spur_frac_d39 == 0.001
    by = {p.variant: p for p in out.pairs}
    for key in ac.V2C_VARIANT_ORDER:
        p = by[key]
        assert p.peak_a and p.peak_b and p.valley_raw and p.valley_at_cut and p.count_ok, (key, p)
    assert by["groups_3d"].verdict == "viable" and by["groups_3d"].valley, by["groups_3d"].reasons
    for key in ac.V2C_CUT_FORCED:
        assert by[key].verdict == "not viable" and not by[key].valley
        assert by[key].reasons == (ac.V2C_REASON_AT_CUT,), by[key].reasons
    assert out.viable_pairs == {"ring_slab": (), "groups_3d": ((0, 1),), "ring_slab_tn": ()}
    k = out.variants["ring_slab"].rings
    assert [r.k_clusters for r in k] == [len(r.clusters) for r in res.rings]
    rows = ac.v2c_pair_rows(out, {"axon_id": "X"})
    assert len(rows) == 6
    want = ["axon_id", "rule", "variant", "variant_label", "ring_a", "ring_b", "h_over_p", "peak_a", "peak_b",
            "valley", "valley_at_cut", "k_a", "k_b", "k_ratio_a", "k_ratio_b", "k_min", "count_ok",
            "exp_spur_frac_clusters", "exp_spur_frac_d39", "n_points", "n_outside", "verdict", "reasons"]
    assert list(rows[0]) == want
    assert "n_chained" not in rows[0] and "chained" not in rows[0]
    assert all(r["reasons"].startswith("sensitivity only") for r in rows[3:])
    text = "\n".join(ac.v2c_summary_lines(out))
    assert "EXPLORATORY" in text and "never" in text and "Q-38" in text


def test_count_criterion_is_a_fraction_of_the_central_ring() -> None:
    # three rings, the top one with half the central ring's clusters: (3c) fails there whatever the counts
    res, v2, xyz = _fake(9, [-170.0, 0.0, 170.0], [100, 100, 50], sigma=12.0, tau=0.0)
    out = ac.viability_v2c(res, v2, xyz_lab_nm=xyz)
    var = out.variants["ring_slab"]
    assert out.central_index == 1 and var.k_central == len(res.rings[1].clusters)
    assert var.rings[2].k_ratio == pytest.approx(len(res.rings[2].clusters) / var.k_central)
    p12 = [p for p in out.pairs_of("ring_slab") if (p.ring_a, p.ring_b) == (1, 2)][0]
    assert not p12.count_ok and p12.verdict == "not viable"
    assert any("of the central ring" in r and "minimum is 89 %" in r for r in p12.reasons)


def test_verdict_tiers() -> None:
    base: Dict[str, Any] = dict(variant="groups_3d", rings=(0, 1), peak_a=True, peak_b=True, valley_raw=True, valley_at_cut_=False,
                k_ratio_a=0.95, k_ratio_b=1.0, k_a=95, k_b=100, k_central=100)
    assert ac.verdict_v2c(exp_spur_frac_clusters_=0.02, **base)[0] == "viable"
    assert ac.verdict_v2c(exp_spur_frac_clusters_=0.03, **base)[0] == "marginal"
    assert ac.verdict_v2c(exp_spur_frac_clusters_=0.051, **base)[0] == "not viable"
    assert ac.verdict_v2c(exp_spur_frac_clusters_=float("nan"), **base)[0] == "not viable"
    b2: Dict[str, Any] = dict(base, k_ratio_a=0.88)
    assert ac.verdict_v2c(exp_spur_frac_clusters_=0.0, **b2)[0] == "not viable"
    b3: Dict[str, Any] = dict(base, k_central=0, k_ratio_a=float("nan"), k_ratio_b=float("nan"))
    v, _valley, ok, reasons = ac.verdict_v2c(exp_spur_frac_clusters_=0.0, **b3)
    assert v == "not viable" and not ok and any("central ring without clusters" in r for r in reasons)


# =========================================================================== Q-38: never selects pairs
def test_no_pair_selection_uses_rule_v2c() -> None:
    for name in ("batch_columns.py", "power_columns.py", os.path.join("tools", "mps_columns_window.py"),
                 os.path.join("tools", "mps_axial_precision.py"), os.path.join("tools", "mps_columns.py")):
        with open(os.path.join(HERE, name), encoding="utf-8") as fh:
            src = fh.read()
        assert "viability_v2c" not in src and "V2C_" not in src, name
