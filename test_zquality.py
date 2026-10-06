# -*- coding: utf-8 -*-
"""
Unit tests of ``tools.mps_axial_precision.z_quality`` (D-39, H5-E S2).

Every case is a hand-built ``RingsResult`` look-alike whose answer is
known in closed form: the separation index, the analytic leak of a
Gaussian component past the cut, the expected leak copies of D-27c, and
the verdict table of D-39(b). The simulated known-truth cases (the
research classification reproduced on new seeds) live with the private
research validators.

Run: ``py -3 -m pytest test_zquality.py -q``.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import pytest
from scipy.stats import binom, norm

from tools import mps_axial_precision as ap

P = 170.0


def _fake_res(mus: Sequence[float], sds: Sequence[float], *, weights: Optional[Sequence[float]] = None,
              n_locs_ring: int = 800, cluster_sizes: Optional[List[List[int]]] = None, depth: Optional[float] = 0.5,
              events: bool = True, lpz: float = 25.0, seed: int = 11, min_samples: int = 10) -> SimpleNamespace:
    """Rings at ``mus`` with Gaussian z (sd ``sds``), cut halfway between
    neighbours; each ring's slab is [cut below, cut above] (the outer
    edges at mu -+ 3 sd). Clusters partition the ring's localizations
    into ``cluster_sizes`` (default 40 clusters of 20)."""
    rng = np.random.default_rng(seed)
    n = len(mus)
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, float)
    cuts = [0.5 * (mus[i] + mus[i + 1]) for i in range(n - 1)]
    z_all: List[np.ndarray] = []
    rings = []
    start = 0
    for i in range(n):
        z = rng.normal(mus[i], sds[i], n_locs_ring)
        z_all.append(z)
        li = np.arange(start, start + n_locs_ring)
        start += n_locs_ring
        sizes = cluster_sizes[i] if cluster_sizes is not None else [20] * (n_locs_ring // 20)
        clusters, k0 = [], 0
        for s in sizes:
            clusters.append(SimpleNamespace(n_locs=int(s), loc_index=li[k0:k0 + s]))
            k0 += s
        z_lo = cuts[i - 1] if i > 0 else mus[i] - 3 * sds[i]
        z_hi = cuts[i] if i < n - 1 else mus[i] + 3 * sds[i]
        rings.append(SimpleNamespace(
            index=i, component_index=i, centre_z_nm=float(mus[i]), sigma_z_nm=float(sds[i]), lpz_median_nm=lpz,
            loc_index=li, clusters=clusters, n_events=(n_locs_ring // 2 if events else None), z_lo_nm=float(z_lo),
            z_hi_nm=float(z_hi), relative_depth_hi=(depth if i < n - 1 else None)))
    z_p = np.concatenate(z_all) if z_all else np.zeros(0)
    ev = np.repeat(np.arange(z_p.size // 2), 2) if events else None
    if ev is not None and ev.size < z_p.size:
        ev = np.concatenate([ev, [-1]])
    zr = SimpleNamespace(means_nm=np.asarray(mus, float), sigmas_nm=np.asarray(sds, float), weights=w)
    return SimpleNamespace(rings=rings[::-1], z_p=z_p, event_id=ev, ms=SimpleNamespace(z_result=zr),
                           params=SimpleNamespace(min_samples=min_samples))


# --------------------------------------------------------------- metrics
def test_separation_leak_and_copies_closed_form() -> None:
    res = _fake_res([0.0, P], [30.0, 40.0])
    zq = ap.z_quality(res, p_ref_nm=P)
    assert [r.index for r in zq.rings] == [0, 1]
    b = zq.boundaries[0]
    assert (b.ring_a, b.ring_b) == (0, 1)
    assert b.d_sep == pytest.approx(P / math.sqrt(30.0 ** 2 + 40.0 ** 2), rel=1e-12)
    assert b.ashman_d == pytest.approx(b.d_sep * math.sqrt(2.0), rel=1e-12)
    cut = 0.5 * P
    assert b.f_lo == pytest.approx(norm.sf(cut / 30.0), rel=1e-12)
    assert b.f_hi == pytest.approx(norm.sf((P - cut) / 40.0), rel=1e-12)
    assert b.misassign == pytest.approx(0.5 * b.f_lo + 0.5 * b.f_hi, rel=1e-12)
    # D-27c copies: n0 = round(n / keep), keep = the component's mass inside its own slab
    keep_a = norm.cdf(cut / 30.0) - norm.cdf(-3.0)
    keep_b = norm.cdf(3.0) - norm.cdf((cut - P) / 40.0)
    e = 40 * binom.sf(9, int(round(20 / keep_a)), b.f_lo) + 40 * binom.sf(9, int(round(20 / keep_b)), b.f_hi)
    assert b.exp_spur_frac == pytest.approx(e / 80.0, rel=1e-9, abs=1e-300)
    assert b.gap_nm == pytest.approx(P)
    assert b.n_events == 800
    assert b.info_ratio == pytest.approx(800 / b.n_req_events, rel=1e-12)
    assert b.verdict == ap.VERDICT_VIABLE
    assert zq.axon_verdict == ap.VERDICT_VIABLE
    assert zq.viable_pairs == ((0, 1),) and zq.marginal_pairs == ()


def test_copies_grow_with_cluster_size() -> None:
    small = ap.z_quality(_fake_res([0.0, 130.0], [35.0, 35.0]), p_ref_nm=P).boundaries[0]
    big = ap.z_quality(_fake_res([0.0, 130.0], [35.0, 35.0], cluster_sizes=[[400, 400]] * 2), p_ref_nm=P).boundaries[0]
    assert big.exp_spur_frac > 10 * max(small.exp_spur_frac, 1e-12)
    assert big.verdict == ap.VERDICT_NOT_VIABLE
    assert any("leak copies" in r for r in big.reasons)


def test_kl_and_required_events() -> None:
    assert ap.kl_mixture_vs_gauss(np.array([0.0, 0.0]), np.array([30.0, 30.0]), np.array([0.5, 0.5])) == pytest.approx(0.0, abs=1e-9)
    kls = [ap.kl_mixture_vs_gauss(np.array([0.0, d]), np.array([30.0, 30.0]), np.array([0.5, 0.5])) for d in (40, 80, 160)]
    assert kls[0] < kls[1] < kls[2]
    n = ap.n_required(kls[1])
    assert 2 * n * kls[1] - 3 * math.log(n) >= 10.0
    n_less = n * (1 - 1e-6)
    assert 2 * n_less * kls[1] - 3 * math.log(n_less) < 10.0 + 1e-6
    assert ap.n_required(0.0) == math.inf and ap.n_required(float("nan")) == math.inf


def test_ambiguous_fraction_counts_low_posterior() -> None:
    b = ap.z_quality(_fake_res([0.0, 60.0], [30.0, 30.0]), p_ref_nm=P).boundaries[0]
    wide = ap.z_quality(_fake_res([0.0, 200.0], [30.0, 30.0]), p_ref_nm=P).boundaries[0]
    assert 0.0 <= wide.ambiguous_frac < b.ambiguous_frac <= 1.0


# --------------------------------------------------------------- verdict table (D-39 b)
@pytest.mark.parametrize("d,spur,info,valley,gap,flag,expected", [
    (2.5, 0.001, 5.0, 0.3, 170.0, False, "viable"),
    (2.5, 0.001, float("nan"), 0.3, 170.0, False, "viable"),     # no events: the info criterion is not applied
    (2.5, 0.001, 1.5, 0.3, 170.0, False, "marginal"),            # too few events for Delta BIC >= 10
    (2.1, 0.001, 5.0, 0.3, 170.0, False, "marginal"),            # 2.0 <= d < 2.2
    (2.5, 0.03, 5.0, 0.3, 170.0, False, "marginal"),             # 2 % < spur <= 5 %
    (2.5, 0.06, 5.0, 0.3, 170.0, False, "not viable"),
    (1.9, 0.001, 5.0, 0.3, 170.0, False, "not viable"),
    (2.5, 0.001, 5.0, 0.0, 170.0, False, "not viable"),          # no interior valley
    (2.5, 0.001, 5.0, float("nan"), 170.0, False, "not viable"),
    (2.5, 0.001, 5.0, 0.3, 1.31 * P, False, "not viable"),       # a lost ring in between
    (2.5, 0.001, 5.0, 0.3, 170.0, True, "not viable"),           # a ring wider than P/2
    (2.2, 0.02, 2.0, 0.3, 1.3 * P, False, "viable"),             # every edge inclusive
])
def test_boundary_verdict_table(d: float, spur: float, info: float, valley: float, gap: float, flag: bool,
                                expected: str) -> None:
    ring_flags = (("sigma_z 90 > 0.5 P_ref",), ()) if flag else ((), ())
    v, reasons = ap.boundary_verdict(d_sep=d, exp_spur_frac=spur, info_ratio=info, valley_depth=valley, gap_nm=gap,
                                     ring_flags=ring_flags, rings=(0, 1), p_ref_nm=P)
    assert v == expected
    if expected != "viable":
        assert reasons


def test_wide_ring_blocks_both_of_its_boundaries() -> None:
    zq = ap.z_quality(_fake_res([0.0, P, 2 * P], [30.0, 0.55 * P, 30.0]), p_ref_nm=P)
    assert zq.rings[1].flags and not zq.rings[0].flags
    assert [b.verdict for b in zq.boundaries] == [ap.VERDICT_NOT_VIABLE] * 2
    assert zq.axon_verdict == ap.VERDICT_NOT_VIABLE


def test_no_valley_and_lost_ring() -> None:
    zq = ap.z_quality(_fake_res([0.0, P], [30.0, 30.0], depth=None), p_ref_nm=P)
    assert zq.boundaries[0].verdict == ap.VERDICT_NOT_VIABLE and "no interior valley" in zq.boundaries[0].reasons
    zq = ap.z_quality(_fake_res([0.0, 2 * P], [30.0, 30.0]), p_ref_nm=P)
    assert zq.boundaries[0].verdict == ap.VERDICT_NOT_VIABLE
    assert any("lost ring" in r for r in zq.boundaries[0].reasons)


def test_axon_verdict_is_the_best_pair_and_marginal_listed() -> None:
    zq = ap.z_quality(_fake_res([0.0, 150.0, 3 * 150.0 / 2 + 150.0 - 60.0], [30.0, 30.0, 30.0]), p_ref_nm=P)
    verdicts = [b.verdict for b in zq.boundaries]
    assert ap.VERDICT_VIABLE in verdicts
    assert zq.axon_verdict == ap.VERDICT_VIABLE
    assert set(zq.viable_pairs) | set(zq.marginal_pairs) <= {(0, 1), (1, 2)}


def test_fewer_than_two_rings() -> None:
    zq = ap.z_quality(_fake_res([0.0], [30.0]))
    assert zq.boundaries == () and zq.viable_pairs == ()
    assert zq.axon_verdict == ap.AXON_FEWER_THAN_TWO_RINGS == "not viable (fewer than 2 rings)"


# --------------------------------------------------------------- inputs, determinism, selection key
def test_defaults_and_sources() -> None:
    res = _fake_res([0.0, P], [30.0, 30.0], min_samples=7)
    zq = ap.z_quality(res)
    assert zq.p_ref_nm == ap.P_REF_DEFAULT_NM and "default" in zq.p_ref_source
    assert zq.min_samples == 7
    zq2 = ap.z_quality(res, p_ref_nm=160.0)
    assert zq2.p_ref_nm == 160.0 and zq2.p_ref_source == "argument"
    assert ap.z_quality(res) == zq                                   # deterministic, bit for bit
    t = ap.ZQualityThresholds()
    assert (t.d_viable, t.d_marginal, t.spur_viable, t.spur_marginal, t.info_ratio_min, t.merged_sigma_over_p,
            t.merged_gap_over_p) == (2.2, 2.0, 0.02, 0.05, 2.0, 0.5, 1.3)


def test_no_events_gives_nan_info() -> None:
    b = ap.z_quality(_fake_res([0.0, P], [30.0, 30.0], events=False), p_ref_nm=P).boundaries[0]
    assert b.n_events is None and math.isnan(b.info_ratio) and b.verdict == ap.VERDICT_VIABLE


def test_selection_key() -> None:
    zq = ap.z_quality(_fake_res([0.0, P, P + 128.0], [30.0, 30.0, 30.0]), p_ref_nm=P)
    assert ap.z_selection_key(zq, "viable") == (3, zq.viable_pairs)
    both = tuple(sorted(zq.viable_pairs + zq.marginal_pairs))
    assert ap.z_selection_key(zq, "viable+marginal") == (3, both)
    with pytest.raises(ValueError):
        ap.z_selection_key(zq, "all")


# --------------------------------------------------------------- uncalibrated acquisition flag (Q-28, D-38f)
EXAMPLE_PATTERN = r"(?i)roi[ _-]?0*2(?!\d)"     # an example site pattern: folders named "ROI 2" and its spellings


@pytest.mark.parametrize("label,flagged", [
    ("sample_ROI 2_cellB.csv", True), ("roi_02_picked.hdf5", True), ("C:/data/ROI-2/locs.hdf5", True),
    ("ROI2", True), ("ROI 1 axonA", False), ("ROI 12", False), ("ROI 20", False), ("roi_022", False), ("", False),
])
def test_roi2_flag(label: str, flagged: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ap.UNCALIBRATED_LABELS_ENV, EXAMPLE_PATTERN)
    flags = ap.calibration_range_flags(label, None)
    assert bool(flags) == flagged
    if flagged:
        assert flags == (ap.ROI2_FLAG,)


def test_roi2_flag_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ap.UNCALIBRATED_LABELS_ENV, raising=False)
    assert ap.ROI2_PATTERN is None and ap.uncalibrated_label_pattern() is None
    assert ap.calibration_range_flags("sample_ROI 2_cellB.csv", None) == ()
    monkeypatch.setenv(ap.UNCALIBRATED_LABELS_ENV, "(unclosed")
    with pytest.raises(ValueError):
        ap.calibration_range_flags("x", None)


def test_envelope_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    zq = ap.z_quality(_fake_res([0.0, P], [30.0, 30.0]), p_ref_nm=P)
    assert ap.CALIBRATED_ENVELOPE is None
    assert ap.calibration_range_flags("axonA", zq) == ()
    d = zq.boundaries[0].d_sep
    monkeypatch.setattr(ap, "CALIBRATED_ENVELOPE", {"d_sep_min": d + 0.1, "exp_spur_frac_max": 1.0})
    flags = ap.calibration_range_flags("axonA", zq)
    assert len(flags) == 1 and "fuera del rango calibrado" in flags[0] and "0-1" in flags[0]
