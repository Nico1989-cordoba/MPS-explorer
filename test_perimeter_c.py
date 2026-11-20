"""
Question C of the perimeter density analysis (P5): correlation between slices, with hand-built simulated data.

Run alone: python -m pytest test_perimeter_c.py
"""
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.mps_gaps import profile_correlation  # noqa: E402
from tools.mps_perimeter_bc import (leak_expected_r, slice_correlations, slice_reading,  # noqa: E402
                                    LEAK_TOLERANCE)
from tools.mps_perimeter_types import DensityProfile, SliceProfiles, TestReadout  # noqa: E402

T_NM = 170.0
LPZ_NM = 47.0


# ---------------------------------------------------------------------------------------------------
# Helpers (all data is built here: the P1 simulator does not exist yet)
# ---------------------------------------------------------------------------------------------------
def make_profile(counts: np.ndarray, measured: Optional[np.ndarray] = None,
                 bin_nm: float = 100.0, origin: float = 0.0) -> DensityProfile:
    counts = np.asarray(counts, dtype=np.int64)
    n = counts.size
    mx = int(counts.max()) if n else 0
    rel = counts / mx if mx > 0 else np.zeros(n)
    meas = np.ones(n, dtype=bool) if measured is None else np.asarray(measured, dtype=bool)
    return DensityProfile(counts=counts, relative=rel.astype(np.float64), measured=meas, n_bins=n,
                          bin_nm_effective=bin_nm, origin_arc_nm=origin, max_bin=int(np.argmax(counts)) if n else 0,
                          max_tie=False, n_locs=int(counts.sum()))


def make_profiles(c: Dict[int, np.ndarray], frame: str = "z0_max", **kw: object) -> SliceProfiles:
    return SliceProfiles(by_slice={k: make_profile(v, **kw) for k, v in c.items()},  # type: ignore[arg-type]
                         length_nm=float(len(c[0]) * 100.0), frame=frame)  # type: ignore[arg-type]


def smooth_pattern(rng: np.random.Generator, n: int, width: int = 5, depth: float = 0.5) -> np.ndarray:
    """Positive circular pattern with a correlation length of ~width bins (like a real perimeter profile)."""
    g = rng.standard_normal(n)
    k = np.zeros(n)
    k[:width] = 1.0
    sm = np.real(np.fft.ifft(np.fft.fft(g) * np.fft.fft(k)))
    sm = (sm - sm.mean()) / sm.std()
    p = np.exp(depth * sm)
    return p / p.sum()


def leaked_slices(rng: np.random.Generator, n_bins: int, n_mol: int, lpz: float, guard: float,
                  patterns: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, np.ndarray]:
    """Counts of the observed slices -1, 0, +1 from molecules in true slices -2..+2 with Gaussian z error."""
    if patterns is None:
        patterns = {k: smooth_pattern(rng, n_bins) for k in range(-2, 3)}
    obs = {k: np.zeros(n_bins, dtype=np.int64) for k in (-1, 0, 1)}
    for k in range(-2, 3):
        bins = rng.choice(n_bins, size=n_mol, p=patterns[k])
        z = k * T_NM + rng.uniform(-T_NM / 2, T_NM / 2, n_mol) + rng.normal(0.0, lpz, n_mol)
        j = np.floor(z / T_NM + 0.5).astype(int)
        keep = (np.abs(z - j * T_NM) <= T_NM / 2 - guard) & (np.abs(j) <= 1)
        for s in (-1, 0, 1):
            sel = keep & (j == s)
            obs[s] += np.bincount(bins[sel], minlength=n_bins)
    return obs


# ---------------------------------------------------------------------------------------------------
# The sum of the leak
# ---------------------------------------------------------------------------------------------------
def test_leak_expected_r_matches_the_plan_figure():
    r = leak_expected_r(LPZ_NM, T_NM, 0.0)
    assert 0.26 <= r <= 0.28


def test_leak_expected_r_guard_and_monotonic():
    # The plan guessed < 0.10 for a 42.5 nm guard; the sum gives ~0.12 (checked against simulation below).
    assert leak_expected_r(LPZ_NM, T_NM, 42.5) < 0.5 * leak_expected_r(LPZ_NM, T_NM, 0.0)
    assert leak_expected_r(LPZ_NM, T_NM, 42.5) < leak_expected_r(LPZ_NM, T_NM, 20.0)
    assert leak_expected_r(LPZ_NM, T_NM, 20.0) < leak_expected_r(LPZ_NM, T_NM, 0.0)
    assert leak_expected_r(30.0, T_NM, 0.0) < leak_expected_r(47.0, T_NM, 0.0) < leak_expected_r(77.0, T_NM, 0.0)
    assert leak_expected_r(0.0, T_NM, 0.0) == 0.0


def test_leak_expected_r_rejects_bad_inputs():
    with pytest.raises(ValueError):
        leak_expected_r(LPZ_NM, 0.0, 0.0)
    with pytest.raises(ValueError):
        leak_expected_r(LPZ_NM, T_NM, 85.0)
    with pytest.raises(ValueError):
        leak_expected_r(LPZ_NM, T_NM, -1.0)
    with pytest.raises(ValueError):
        leak_expected_r(-1.0, T_NM, 0.0)


# ---------------------------------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------------------------------
def test_negative_control_independent_slices_p_is_uniform():
    """500 axons, three independent patterns, no leak: the p of all three pairs is ~uniform."""
    rng = np.random.default_rng(11)
    n_bins, n_axons = 100, 500
    ps = {pair: [] for pair in ((-1, 0), (0, 1), (-1, 1))}   # type: Dict[Tuple[int, int], list]
    for _ in range(n_axons):
        c = {k: rng.poisson(300 * n_bins * smooth_pattern(rng, n_bins)) for k in (-1, 0, 1)}
        out = slice_correlations(make_profiles(c), n_null=1999, seed=1)
        for pair, ro in out.items():
            ps[pair].append(ro.p)
    allp = np.concatenate([np.asarray(v) for v in ps.values()])
    assert 0.03 <= float(np.mean(allp <= 0.05)) <= 0.07
    assert 0.40 <= float(np.mean(allp)) <= 0.60
    for v in ps.values():
        assert 0.02 <= float(np.mean(np.asarray(v) <= 0.05)) <= 0.09


def test_positive_control_shared_pattern_in_all_slices():
    rng = np.random.default_rng(5)
    n_bins = 100
    base = smooth_pattern(rng, n_bins)
    c = {k: rng.poisson(3000 * n_bins * base) for k in (-1, 0, 1)}
    out = slice_correlations(make_profiles(c), n_null=1999, seed=0)
    for pair, ro in out.items():
        assert ro.value > 0.8, pair
        assert ro.p <= 1.0 / (1 + n_bins - 1) + 1e-12, pair      # the floor with every shift used
        assert ro.z > 4.0
        assert ro.sided == "one"


def test_weak_shared_pattern_between_the_far_slices_is_detected():
    """-1 and +1 share a pattern (the central one is independent): the pair (-1, +1) is the one that sees it."""
    rng = np.random.default_rng(7)
    n_bins = 200
    shared = smooth_pattern(rng, n_bins)
    hits = 0
    for _ in range(20):
        c = {-1: rng.poisson(600 * n_bins * shared), 1: rng.poisson(600 * n_bins * shared),
             0: rng.poisson(600 * n_bins * smooth_pattern(rng, n_bins))}
        out = slice_correlations(make_profiles(c), n_null=1999, seed=0)
        hits += int(out[(-1, 1)].p < 0.05)
    assert hits >= 18


def test_leak_alone_gives_expected_neighbour_r_and_no_far_pair():
    rng = np.random.default_rng(3)
    n_bins, reps = 400, 40
    expected = leak_expected_r(LPZ_NM, T_NM, 0.0)
    r_nb, r_far, sig_nb, sig_far = [], [], [], []
    for _ in range(reps):
        obs = leaked_slices(rng, n_bins, 150000, LPZ_NM, 0.0)
        out = slice_correlations(make_profiles(obs), n_null=999, seed=0)
        r_nb += [out[(-1, 0)].value, out[(0, 1)].value]
        sig_nb += [out[(-1, 0)].p < 0.05, out[(0, 1)].p < 0.05]
        r_far.append(out[(-1, 1)].value)
        sig_far.append(out[(-1, 1)].p < 0.05)
    assert abs(float(np.mean(r_nb)) - expected) <= 0.05
    assert float(np.mean(sig_nb)) >= 0.8
    assert float(np.mean(r_far)) < 0.10
    assert float(np.mean(sig_far)) <= 0.20


def test_guard_removes_the_leak():
    rng = np.random.default_rng(4)
    n_bins, reps = 400, 40
    r_nb = []
    for _ in range(reps):
        obs = leaked_slices(rng, n_bins, 150000, LPZ_NM, 42.5)
        out = slice_correlations(make_profiles(obs), n_null=99, seed=0)
        r_nb += [out[(-1, 0)].value, out[(0, 1)].value]
    expected = leak_expected_r(LPZ_NM, T_NM, 42.5)
    assert abs(float(np.mean(r_nb)) - expected) <= 0.05
    assert float(np.mean(r_nb)) < 0.5 * leak_expected_r(LPZ_NM, T_NM, 0.0)


# ---------------------------------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------------------------------
def test_scaling_a_profile_does_not_change_r():
    rng = np.random.default_rng(9)
    c = {k: rng.poisson(200, 80) for k in (-1, 0, 1)}
    a = slice_correlations(make_profiles(c), n_null=999, seed=2)
    c2 = {-1: c[-1] * 7, 0: c[0], 1: c[1] * 3}
    b = slice_correlations(make_profiles(c2), n_null=999, seed=2)
    for pair in a:
        assert abs(a[pair].value - b[pair].value) < 1e-12
        assert a[pair].p == b[pair].p


def test_value_is_the_plain_pearson_r():
    rng = np.random.default_rng(10)
    c = {k: rng.poisson(100, 60) for k in (-1, 0, 1)}
    out = slice_correlations(make_profiles(c), n_null=999, seed=0)
    assert abs(out[(-1, 1)].value - np.corrcoef(c[-1], c[1])[0, 1]) < 1e-12
    assert abs(out[(0, 1)].null_mean) < 0.1 and out[(0, 1)].null_sd > 0


def test_unmeasured_bins_match_profile_correlation_with_mask():
    rng = np.random.default_rng(12)
    n = 90
    c = {k: rng.poisson(150, n) for k in (-1, 0, 1)}
    meas = np.ones(n, dtype=bool)
    meas[10:25] = False
    meas[60] = False
    profs = SliceProfiles(by_slice={k: make_profile(v, measured=meas) for k, v in c.items()},
                          length_nm=n * 100.0, frame="z0_max")
    out = slice_correlations(profs, n_null=1999, seed=5)
    ref = profile_correlation(c[-1].astype(float), c[1].astype(float), n_rotations=1999, seed=5,
                              measured_a=meas, measured_b=meas)
    assert out[(-1, 1)].value == ref.r_at_zero
    assert out[(-1, 1)].p == ref.p_rotation
    # and the masked bins really are ignored
    keep = meas
    assert abs(out[(-1, 1)].value - np.corrcoef(c[-1][keep], c[1][keep])[0, 1]) < 1e-12


def test_own_max_frame_is_refused():
    rng = np.random.default_rng(1)
    c = {k: rng.poisson(100, 50) for k in (-1, 0, 1)}
    with pytest.raises(ValueError, match="z0_max"):
        slice_correlations(make_profiles(c, frame="own_max"))


def test_missing_slice_and_grid_mismatch_are_refused():
    rng = np.random.default_rng(1)
    c = {k: rng.poisson(100, 50) for k in (-1, 0, 1)}
    with pytest.raises(ValueError, match="Missing slice"):
        slice_correlations(make_profiles({-1: c[-1], 0: c[0]}))
    p = make_profiles(c)
    shifted = SliceProfiles(by_slice={-1: p.by_slice[-1], 0: p.by_slice[0],
                                      1: make_profile(c[1], origin=500.0)},
                            length_nm=p.length_nm, frame="z0_max")
    with pytest.raises(ValueError, match="grid and origin"):
        slice_correlations(shifted)
    short = SliceProfiles(by_slice={-1: p.by_slice[-1], 0: p.by_slice[0], 1: make_profile(c[1][:40])},
                          length_nm=p.length_nm, frame="z0_max")
    with pytest.raises(ValueError, match="grid and origin"):
        slice_correlations(short)


def test_constant_or_empty_profile_returns_nan_not_an_error():
    rng = np.random.default_rng(2)
    c = {-1: np.full(40, 5), 0: rng.poisson(100, 40), 1: np.zeros(40, dtype=int)}
    out = slice_correlations(make_profiles(c))
    assert np.isnan(out[(-1, 0)].value) and np.isnan(out[(-1, 0)].p)
    assert np.isnan(out[(0, 1)].value)
    assert np.isnan(out[(-1, 1)].p)


def test_deterministic_with_seed_when_the_null_is_subsampled():
    rng = np.random.default_rng(6)
    n = 3000                       # n - 1 > n_null: a random subset of shifts
    c = {k: rng.poisson(50, n) for k in (-1, 0, 1)}
    a = slice_correlations(make_profiles(c), n_null=199, seed=3)
    b = slice_correlations(make_profiles(c), n_null=199, seed=3)
    for pair in a:
        assert a[pair] == b[pair]
        assert a[pair].n_null == 199
        assert a[pair].p >= 1.0 / 200


def _ro(r: float, p: float) -> TestReadout:
    return TestReadout(value=r, null_mean=0.0, null_sd=0.1, z=r / 0.1, p=p, n_null=1999, sided="one")


def test_reading_flags_leakage_and_registry():
    exp = 0.27
    leak = {(-1, 0): _ro(0.25, 0.001), (0, 1): _ro(0.30, 0.001), (-1, 1): _ro(0.02, 0.4)}
    assert slice_reading(leak, exp) == "leakage only"
    reg = {(-1, 0): _ro(0.6, 0.001), (0, 1): _ro(0.6, 0.001), (-1, 1): _ro(0.5, 0.001)}
    assert slice_reading(reg, exp) == "registry between slices"
    above = {(-1, 0): _ro(0.27 + LEAK_TOLERANCE + 0.1, 0.001), (0, 1): _ro(0.3, 0.2), (-1, 1): _ro(0.1, 0.2)}
    assert slice_reading(above, exp) == "neighbours above leakage"
    none = {(-1, 0): _ro(0.0, 0.5), (0, 1): _ro(0.05, 0.3), (-1, 1): _ro(-0.1, 0.9)}
    assert slice_reading(none, exp) == "no correlation between slices"
    nan = {(-1, 0): _ro(float("nan"), float("nan")), (0, 1): _ro(0.0, 0.5), (-1, 1): _ro(0.0, 0.5)}
    assert slice_reading(nan, exp) == "not computable"
    assert slice_reading({}, exp) == "not computable"


def test_reading_with_real_fractions_compares_each_pair_with_its_own_leak():
    from tools.mps_perimeter_bc import leak_expected_r_pairs, slice_reading_pairs, slice_reading_real_fractions
    # laterals at ~40 % of the central slice, as in the pilot: the neighbour leak is far above the equal-slice 0.27
    n = (400.0, 1000.0, 400.0)
    leaks = {sz: leak_expected_r_pairs(n, sz, 170.0, 0.0) for sz in (36.0, 47.0, 51.0, 77.0)}
    assert leaks[47.0][(-1, 0)] > 0.27 + LEAK_TOLERANCE
    assert leaks[36.0][(-1, 0)] < leaks[47.0][(-1, 0)] < leaks[51.0][(-1, 0)] < leaks[77.0][(-1, 0)]
    # r = 0.45 on both neighbours: above the equal-slice leak, within the leak of these fractions at both widths
    obs = {(-1, 0): _ro(0.45, 0.001), (0, 1): _ro(0.45, 0.001), (-1, 1): _ro(0.05, 0.3)}
    assert slice_reading(obs, 0.27) == "neighbours above leakage"
    assert leaks[47.0][(-1, 0)] + LEAK_TOLERANCE >= 0.45
    assert slice_reading_pairs(obs, leaks[47.0]) == "leakage only"
    assert slice_reading_real_fractions(obs, leaks).split(" (")[0] == "leakage only"     # 47 nm rules the label
    # each pair against its OWN leak: a neighbour above its leak flags the axon, whatever the other pair
    own = {(-1, 0): 0.10, (0, 1): 0.60, (-1, 1): 0.02}
    mixed = {(-1, 0): _ro(0.45, 0.001), (0, 1): _ro(0.45, 0.001), (-1, 1): _ro(0.0, 0.5)}
    assert slice_reading_pairs(mixed, own) == "neighbours above leakage"
    # v21: the 47 nm reading is the label; an end of the measured range 36-51 nm that disagrees is said; 77 nm (upper
    # bound only) never enters the reading
    lk = {36.0: {(-1, 0): 0.25, (0, 1): 0.25, (-1, 1): 0.04}, 47.0: {(-1, 0): 0.30, (0, 1): 0.30, (-1, 1): 0.05},
          51.0: {(-1, 0): 0.40, (0, 1): 0.40, (-1, 1): 0.06}, 77.0: {(-1, 0): 0.90, (0, 1): 0.90, (-1, 1): 0.2}}
    lab = slice_reading_real_fractions(mixed, lk)
    assert lab.split(" (")[0] == "neighbours above leakage"
    assert lab.endswith("(within the measured range of sigma_z 36-51 nm: leakage only at 51 nm)")
    lk_same = {**lk, 51.0: lk[47.0]}
    assert slice_reading_real_fractions(mixed, lk_same) == "neighbours above leakage"   # 77 nm ignored
    # registry and no correlation do not depend on the leak; a missing leak of a significant pair is not computable
    reg = {(-1, 0): _ro(0.6, 0.001), (0, 1): _ro(0.6, 0.001), (-1, 1): _ro(0.5, 0.001)}
    assert slice_reading_real_fractions(reg, lk) == "registry between slices"
    none = {(-1, 0): _ro(0.0, 0.5), (0, 1): _ro(0.05, 0.3), (-1, 1): _ro(-0.1, 0.9)}
    assert slice_reading_real_fractions(none, {}) == "no correlation between slices"
    assert slice_reading_pairs(obs, {(-1, 0): float("nan"), (0, 1): 0.3}) == "not computable"


def test_the_researcher_name_does_not_appear_in_the_files():
    root = Path(__file__).resolve().parent
    token = "un" + "sa"                                   # split so this file does not match itself
    for rel in ("tools/mps_perimeter_bc.py", "test_perimeter_c.py"):
        assert token not in (root / rel).read_text(encoding="utf-8").lower(), rel
