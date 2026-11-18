"""Question A of the perimeter density analysis (P7): zone, fold, shift null, the two statistics, pooling.

Hand-made simulated profiles only (the P1 simulator does not exist yet). Controls: flat -> no reading and
uniform p; clumps -> no reading; RESOLVED clusters (sd 100 to 300 nm, Gaussian and heavier tails) with no
attraction -> no "attractor"; Gaussian halo -> attractor; depleted flank -> inhibition zone; regular modes ->
regular; and the false-finding control: flat profiles aligned on their maximum must NOT read "attractor".
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import mps_perimeter_a as A  # noqa: E402
from tools.mps_perimeter_types import DensityProfile  # noqa: E402


# ----------------------------------------------------------------------------------------------------
# Hand-made data
# ----------------------------------------------------------------------------------------------------
def profile_from_counts(counts, bin_nm=100.0, measured=None, max_bin=None):
    counts = np.asarray(counts, dtype=np.int64)
    measured = np.ones(counts.size, dtype=bool) if measured is None else np.asarray(measured, dtype=bool)
    top = int(counts[measured].max()) if measured.any() else 0
    k = int(np.argmax(np.where(measured, counts, -1))) if max_bin is None else int(max_bin)
    rel = counts / top if top > 0 else np.zeros(counts.size)
    return DensityProfile(counts=counts, relative=rel.astype(float), measured=measured, n_bins=int(counts.size),
                          bin_nm_effective=float(bin_nm), origin_arc_nm=0.0, max_bin=k,
                          max_tie=bool(top > 0 and np.sum(counts[measured] == top) > 1),
                          n_locs=int(counts[measured].sum()))


def empty_profile(n=80):
    """What the profile package returns for an empty slice."""
    return DensityProfile(counts=np.zeros(n, dtype=np.int64), relative=np.zeros(n), measured=np.zeros(n, dtype=bool),
                          n_bins=n, bin_nm_effective=100.0, origin_arc_nm=0.0, max_bin=-1, max_tie=False, n_locs=0)


def simulate(rng, kind="flat", length_nm=None, per_bin=30.0, cluster=400, blink=False, cluster_sd=40.0,
             cluster_shape="gauss", hole_nm=800.0, hole_level=0.0):
    """One slice: a large cluster at a random arc position plus a background of the given kind.

    kind: "flat"; "attractor" (1 + 0.5 exp(-d^2 / 2 600^2)); "inhibition" (``hole_level`` within ``hole_nm`` of
    the cluster); "regular" (1 + 0.5 cos(2 pi d / 800)); "none" (flat and no large cluster at all).
    cluster_shape: "gauss" (sd), "laplace" (sd) or "t3" (Student t, 3 d.o.f., scale ``cluster_sd``).
    blink: every background molecule leaves 1 + geometric localizations 8 nm apart (clumps).
    """
    length = float(rng.uniform(8000.0, 20000.0)) if length_nm is None else float(length_nm)
    n = int(round(length / 100.0))
    s = np.arange(0.0, length, 5.0)
    centre = float(rng.uniform(0.0, length))
    d = np.abs(s - centre)
    d = np.minimum(d, length - d)
    if kind == "attractor":
        lam = 1.0 + 0.5 * np.exp(-d ** 2 / (2.0 * 600.0 ** 2))
    elif kind == "inhibition":
        lam = np.where(d < hole_nm, hole_level, 1.0)
    elif kind == "regular":
        lam = 1.0 + 0.5 * np.cos(2.0 * np.pi * d / 800.0)
    else:
        lam = np.ones_like(s)
    n_bg = int(rng.poisson(per_bin * n))
    if blink:
        n_mol = max(1, n_bg // 4)
        mol = rng.choice(s, size=n_mol, p=lam / lam.sum()) + rng.uniform(0.0, 5.0, n_mol)
        reps = 1 + rng.geometric(0.25, n_mol)                      # mean 5 localizations per molecule
        pos = np.repeat(mol, reps) + rng.normal(0.0, 8.0, int(reps.sum()))
    else:
        pos = rng.choice(s, size=n_bg, p=lam / lam.sum()) + rng.uniform(0.0, 5.0, n_bg)
    if kind != "none" and cluster:
        if cluster_shape == "t3":
            offsets = cluster_sd * rng.standard_t(3, cluster)
        elif cluster_shape == "laplace":
            offsets = rng.laplace(0.0, cluster_sd / np.sqrt(2.0), cluster)
        else:
            offsets = rng.normal(0.0, cluster_sd, cluster)
        pos = np.concatenate([pos, centre + offsets])
    counts = np.bincount((np.mod(pos, length) / (length / n)).astype(int) % n, minlength=n)
    return profile_from_counts(counts, bin_nm=length / n)


def axon_set(rng, kind, n=18, **kw):
    return [simulate(rng, kind, **kw) for _ in range(n)]


def labels_of(kind, n_sets, seed, n_null=199, **kw):
    rng = np.random.default_rng(seed)
    return [A.pooled_details(axon_set(rng, kind, **kw), n_null=n_null, seed=i) for i in range(n_sets)]


def has(details, what):
    """True when the label holds ``what`` as a full, robust reading (not the 'not robust' variant)."""
    return what in details.label.split(A.READING_JOIN)


# ----------------------------------------------------------------------------------------------------
# Exclusion zone
# ----------------------------------------------------------------------------------------------------
def test_zone_rules_by_hand():
    counts = np.full(60, 10)
    counts[20:25] = [60, 80, 100, 70, 20]            # max at 22, median 10
    p = profile_from_counts(counts)
    assert p.max_bin == 22
    # half height above the median = 55: run 20..23, one buffer bin per side; the minimum is +-2
    assert A.exclusion_zone(p, rule="half_max") == (19, 24)
    # baseline: threshold = median + 10 % of the height = 19 (the MAD is 0): reach 3 bins per side, x 1.5 -> 5
    assert A.DEFAULT_RULE == "baseline" and A.exclusion_zone(p) == (17, 27)
    assert A.exclusion_zone(p, rule="fixed", min_nm=200.0) == (20, 24)
    assert A.exclusion_zone(p, rule="fixed", min_nm=400.0) == (18, 26)
    assert A._widen_zone((17, 27), 22, 2.0, 60) == (12, 32)
    with pytest.raises(ValueError):
        A.exclusion_zone(p, rule="dbscan")


def test_zone_wraps_around_the_origin_and_minimum_follows_the_nominal_bin():
    counts = np.full(50, 10)
    counts[0], counts[49], counts[1] = 100, 90, 12
    p = profile_from_counts(counts, bin_nm=99.2)
    first, last = A.exclusion_zone(p, rule="half_max")
    assert (first, last) == (-2, 2)                   # 49 is in the run, 48 is the buffer; right side = minimum
    assert first <= p.max_bin <= last
    assert np.all(np.isnan(A.folded_profile(p, (first, last))[:3]))
    assert A.exclusion_zone(p) == (-3, 2)             # baseline: reach 2 on the left (49 above, 48 below), x 1.5


def test_zone_does_not_swallow_a_flat_profile():
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = profile_from_counts(rng.poisson(50, 150))    # every bin is above half of the maximum
        for rule in ("baseline", "half_max"):
            first, last = A.exclusion_zone(p, rule=rule)
            assert last - first + 1 <= 13, rule
    assert (np.asarray(p.relative) >= 0.5).mean() > 0.9


@pytest.mark.parametrize("sd", [100.0, 150.0, 300.0])
def test_baseline_zone_contains_a_resolved_gaussian_cluster(sd):
    """The zone must reach where the cluster is gone: at least 2.5 sd per side in the median, half_max does not."""
    rng = np.random.default_rng(int(sd))
    base, half = [], []
    for _ in range(60):
        p = simulate(rng, "flat", cluster=1500, cluster_sd=sd)
        zb = A.exclusion_zone(p)
        zh = A.exclusion_zone(p, rule="half_max")
        base.append((zb[1] - zb[0]) / 2 * p.bin_nm_effective)
        half.append((zh[1] - zh[0]) / 2 * p.bin_nm_effective)
    assert np.median(base) >= 2.5 * sd, np.median(base)
    assert np.median(base) <= 6.0 * sd + 200.0                     # and it does not run away along the noise
    assert np.median(half) < 2.5 * sd


# ----------------------------------------------------------------------------------------------------
# Fold, contrast, periodogram
# ----------------------------------------------------------------------------------------------------
def test_folded_profile_by_hand():
    p = profile_from_counts([10, 4, 6, 2, 8, 2, 4, 6], max_bin=0)
    f = A.folded_profile(p, (0, 0))
    assert np.isnan(f[0])
    np.testing.assert_allclose(f[1:], [0.5, 0.5, 0.2, 0.8])     # (4+6)/2, (6+4)/2, (2+2)/2, antipode 8, over 10
    f = A.folded_profile(p, (-1, 1))
    assert np.isnan(f[1]) and f[2] == 0.5
    measured = np.ones(8, dtype=bool)
    measured[1] = False
    f = A.folded_profile(profile_from_counts([10, 4, 6, 2, 8, 2, 4, 6], measured=measured, max_bin=0), (0, 0))
    assert f[1] == 0.6                                          # only the measured side


def test_near_far_contrast_by_hand():
    lags = np.arange(30) * 100.0
    folded = np.full(30, 0.2)
    folded[:3] = np.nan                                         # zone
    folded[3:13] = 0.5                                          # the 10 bins within 1 um of the zone
    assert A.near_far_contrast(folded, lags) == pytest.approx(0.3)
    assert A.near_far_contrast(folded, lags, near_nm=500.0) == pytest.approx(0.5 - (5 * 0.5 + 17 * 0.2) / 22)
    assert np.isnan(A.near_far_contrast(folded[:10], lags[:10]))          # no far side
    w = np.ones(30)
    w[13:] = 0.0
    assert np.isnan(A.near_far_contrast(folded, lags, weights=w))


def test_periodogram_peak_recovers_a_phase_locked_period():
    lags = np.arange(81) * 100.0
    folded = 0.3 + 0.1 * np.cos(2 * np.pi * lags / 800.0)
    folded[:3] = np.nan
    period, power = A.periodogram_peak(folded, 100.0)
    assert abs(period - 800.0) <= 100.0 and power == pytest.approx(0.1 ** 2 / 2, rel=0.1)
    # in quadrature with the maximum: the cosine locked to the maximum carries much less of it
    quad = 0.3 + 0.1 * np.sin(2 * np.pi * lags / 800.0)
    quad[:3] = np.nan
    assert A.periodogram_peak(quad, 100.0)[1] < 0.5 * power
    assert np.isnan(A.periodogram_peak(np.array([np.nan, 1.0]), 100.0)[0])


def test_plain_periodogram_is_blind_to_the_shift_null():
    """Why the statistic is phase-locked: the plain periodogram of the remainder is the same in every replicate."""
    rng = np.random.default_rng(11)
    p = simulate(rng, "regular", length_nm=16000.0)
    zone = A.exclusion_zone(p)
    rem = (np.arange(zone[1] + 1, zone[0] + p.n_bins)) % p.n_bins
    null = A.remainder_shift_null(p, zone, n_null=1999, rng=rng)
    plain_obs = np.abs(np.fft.rfft(np.asarray(p.relative)[rem]))[1:].max()
    plain_null = np.abs(np.fft.rfft(null[:, rem], axis=1))[:, 1:].max(axis=1)
    np.testing.assert_allclose(plain_null, plain_obs, rtol=1e-9)           # p would be 1 whatever the data
    locked = [A.periodogram_peak(A._fold(np.where(np.isin(np.arange(p.n_bins), rem), r, np.nan)[None, :],
                                         p.max_bin)[0][0], p.bin_nm_effective)[1] for r in null]
    assert np.ptp(locked) > 0.5 * np.mean(locked)                          # the phase-locked one does move


# ----------------------------------------------------------------------------------------------------
# The null
# ----------------------------------------------------------------------------------------------------
def test_null_keeps_the_maximum_the_zone_the_counts_and_is_exhaustive():
    rng = np.random.default_rng(5)
    p = simulate(rng, "flat", length_nm=12000.0, blink=True)
    zone = A.exclusion_zone(p)
    n = p.n_bins
    m = n - (zone[1] - zone[0] + 1)
    null = A.remainder_shift_null(p, zone, n_null=1999, rng=rng)
    assert null.shape == (m - 1, n)                                       # every non-zero shift, once
    assert len({tuple(r) for r in null.round(9)}) == m - 1
    rel = np.asarray(p.relative)
    zone_bins = np.arange(zone[0], zone[1] + 1) % n
    assert np.all(null[:, zone_bins] == rel[zone_bins])                   # maximum and its shape untouched
    assert np.all(np.argmax(null, axis=1) == p.max_bin)                   # the same bin is the maximum in all
    assert np.all(null.max(axis=1) == 1.0)
    np.testing.assert_allclose(np.sort(null, axis=1), np.tile(np.sort(rel), (m - 1, 1)))   # same localizations
    sub = A.remainder_shift_null(p, zone, n_null=20, rng=np.random.default_rng(1))
    assert sub.shape == (20, n)
    np.testing.assert_array_equal(sub, A.remainder_shift_null(p, zone, n_null=20, rng=np.random.default_rng(1)))


def test_readouts_are_deterministic_and_carry_the_exact_p():
    p = simulate(np.random.default_rng(8), "flat", length_nm=10000.0)
    a = A.axon_a_readouts(p, seed=4)
    b = A.axon_a_readouts(p, seed=4)
    assert a == b and set(a) == {"contrast", "period"}
    d = A.axon_a_details(p)
    assert d.contrast == a["contrast"] and d.usable and d.exclusion_rule == "baseline"
    assert a["contrast"].sided == "two" and a["period"].sided == "one"
    assert a["contrast"].n_null == d.n_remainder_bins - 1
    assert d.exclusion_nm == pytest.approx((d.zone[1] - d.zone[0] + 1) * p.bin_nm_effective)
    assert 300.0 <= d.period_nm <= p.n_bins // 2 * p.bin_nm_effective
    for r in a.values():
        assert 1.0 / (r.n_null + 1) <= r.p <= 1.0 and np.isfinite(r.z)
    # the period range reaches the details (it used to be dropped)
    short = A.axon_a_readouts(p, period_range_nm=(300.0, 600.0))["period"]
    assert short.value <= a["period"].value + 1e-12
    assert A.axon_a_details(p, period_range_nm=(300.0, 600.0)).period_nm <= 600.0 + 1e-9
    assert A.axon_a_details(p, widen=2.0).exclusion_nm > d.exclusion_nm


def test_contrast_is_in_units_of_the_remainder_mean():
    """Dividing the remainder by its own mean: the same background under a 10 times larger cluster reads the same."""
    rng = np.random.default_rng(9)
    bg = rng.poisson(40, 150)
    small, large = bg.copy(), bg.copy()
    small[70] += 200
    large[70] += 2000
    a = A.axon_a_details(profile_from_counts(small), rule="fixed")
    b = A.axon_a_details(profile_from_counts(large), rule="fixed")
    assert a.contrast.value == pytest.approx(b.contrast.value, rel=1e-9)
    assert a.contrast.p == b.contrast.p and abs(a.contrast.null_mean) < 1e-3


def test_empty_or_tiny_profiles_give_nan_not_exceptions():
    for p in (empty_profile(), profile_from_counts(np.zeros(40, dtype=int)),
              profile_from_counts([5, 9, 3, 4, 2, 6, 1, 3])):
        d = A.axon_a_details(p)
        assert not d.usable and np.isnan(d.contrast.p) and np.isnan(d.period.value) and np.isnan(d.period_nm)
        assert set(A.axon_a_readouts(p)) == {"contrast", "period"}
        assert A.folded_profile(p, A.exclusion_zone(p)).size == p.n_bins // 2 + 1
        assert A.remainder_shift_null(p, (0, p.n_bins - 1), n_null=10, rng=np.random.default_rng(0)).shape[0] == 0
        assert set(A.zone_sensitivity(p, n_null=19)) >= {"baseline", "baseline_x3"}
    d = A.pooled_details([empty_profile(), empty_profile()], n_null=19)
    assert d.result.n_axons == 0 and d.label == A.READING_NOT_COMPUTABLE and np.isnan(d.result.contrast.p)
    assert A.reading_label(d.result) == A.READING_NOT_COMPUTABLE and d.sensitivity == ()
    assert A.pool_aligned([]).lag_nm.size == 0 and A.pool_aligned([empty_profile()]).n_axons.sum() == 0
    ok = simulate(np.random.default_rng(2), "flat", length_nm=9000.0)
    assert A.pooled_test([ok, empty_profile()], n_null=19).n_axons == 1    # the empty one is left out, no error


def test_unmeasured_bins_travel_with_the_remainder():
    rng = np.random.default_rng(13)
    counts = rng.poisson(30, 120)
    counts[10] = 200
    measured = np.ones(120, dtype=bool)
    measured[60:70] = False                                      # a gap in the contour support
    p = profile_from_counts(counts, measured=measured)
    zone = A.exclusion_zone(p)
    null = A.remainder_shift_null(p, zone, n_null=1999, rng=rng)
    assert np.all(np.isnan(null).sum(axis=1) == 10)
    r = A.axon_a_readouts(p)
    assert np.isfinite(r["contrast"].p) and np.isfinite(r["period"].p)


# ----------------------------------------------------------------------------------------------------
# Negative controls, per axon: flat (Poisson) and clumps (blinking)
# ----------------------------------------------------------------------------------------------------
def _per_axon_p(kind, blink, n_sim, seed):
    rng = np.random.default_rng(seed)
    pc, pp = [], []
    for _ in range(n_sim):
        r = A.axon_a_readouts(simulate(rng, kind, blink=blink))
        pc.append(r["contrast"].p)
        pp.append(r["period"].p)
    return np.array(pc), np.array(pp)


@pytest.mark.parametrize("kind,blink,seed", [("flat", False, 101), ("none", False, 102), ("flat", True, 103)])
def test_negative_controls_per_axon_false_positives_and_uniform_p(kind, blink, seed):
    """Flat with a large cluster, flat with no cluster at all (the maximum is noise), and clumps."""
    pc, pp = _per_axon_p(kind, blink, 600, seed)
    for p in (pc, pp):
        assert 0.03 <= np.mean(p <= 0.05) <= 0.07, np.mean(p <= 0.05)
        grid = np.linspace(0.05, 0.95, 19)
        ecdf = np.array([np.mean(p <= g) for g in grid])
        assert np.max(np.abs(ecdf - grid)) < 0.06, np.max(np.abs(ecdf - grid))        # uniform p


# ----------------------------------------------------------------------------------------------------
# The artefact and the false-finding control
# ----------------------------------------------------------------------------------------------------
def test_raw_pooled_profile_of_flat_axons_shows_the_peak_at_zero_and_the_test_does_not_buy_it():
    rng = np.random.default_rng(21)
    profiles = axon_set(rng, "none")                              # nothing but Poisson noise
    raw = A.pool_aligned(profiles)
    rest = raw.mean_relative[3:60]
    assert raw.mean_relative[0] == 1.0                            # the artefact: lag 0 is 1 by construction
    assert raw.mean_relative[0] > rest.mean() + 5 * rest.std()
    assert np.all(np.isnan(raw.null_lo)) and raw.group == A.NO_ANIMAL_GROUP
    res = A.pooled_test(profiles, n_null=999, seed=1)
    assert A.reading_label(res) == A.READING_NONE
    assert res.profile.null_lo[0] == res.profile.null_hi[0] == 1.0     # the peak is inside its own null
    assert res.profile.global_lo[0] == 1.0
    inside = (res.profile.mean_relative >= res.profile.global_lo - 1e-12) & \
             (res.profile.mean_relative <= res.profile.global_hi + 1e-12)
    assert inside[np.isfinite(res.profile.global_lo)].all()


def test_a_null_that_does_not_repeat_the_alignment_finds_the_artefact_every_time():
    """Peak-minus-rest of the aligned flat profile: against un-aligned uniform data it is always 'significant';
    against the same data re-aligned on their own maximum it is not."""
    rng = np.random.default_rng(22)

    def stat(rows, k):
        return rows[:, k] - (np.nansum(rows, axis=1) - rows[:, k]) / (rows.shape[1] - 1)

    wrong, right = [], []
    for _ in range(40):
        p = simulate(rng, "none", length_nm=12000.0)
        obs = stat(np.asarray(p.relative)[None, :], p.max_bin)[0]
        aligned = A.uniform_realigned_null(p, n_null=199, rng=rng)
        assert np.all(aligned[:, p.max_bin] == 1.0) and aligned.shape == (199, p.n_bins)
        unaligned = np.stack([np.roll(r, rng.integers(1, p.n_bins)) for r in aligned])
        wrong.append((1 + np.sum(stat(unaligned, p.max_bin) >= obs)) / 200)
        right.append((1 + np.sum(stat(aligned, p.max_bin) >= obs)) / 200)
    assert np.mean(np.array(wrong) <= 0.05) > 0.9
    assert np.mean(np.array(right) <= 0.05) < 0.2


@pytest.mark.parametrize("kind,blink,seed", [("none", False, 31), ("flat", True, 32)])
def test_false_finding_control_pooled(kind, blink, seed):
    """Sets of 18 structureless axons aligned on their maximum: 'attractor' must stay at the nominal rate."""
    n_sets = 200
    results = labels_of(kind, n_sets, seed, blink=blink)
    attractor = sum(A.READING_ATTRACTOR in d.label for d in results)          # robust or not
    regular = sum(A.READING_REGULAR in d.label for d in results)
    anything = sum(d.label != A.READING_NONE for d in results)
    raw_c = np.array([d.result.contrast.p for d in results])
    raw_p = np.array([d.result.period.p for d in results])
    assert attractor / n_sets <= 0.05, attractor
    assert regular / n_sets <= 0.05, regular
    assert anything / n_sets <= 0.085, anything               # Holm: family-wise 0.05, +2 sd of 200 sets
    assert 0.02 <= np.mean(raw_c <= 0.05) <= 0.085            # raw p of each statistic near nominal
    assert 0.02 <= np.mean(raw_p <= 0.05) <= 0.085
    assert abs(np.mean([d.result.contrast.null_mean for d in results])) < 0.002    # the null is centred on 0


RESOLVED = [("gauss", 100.0, 1500), ("gauss", 150.0, 1500), ("gauss", 300.0, 1500), ("gauss", 150.0, 400),
            ("gauss", 300.0, 400), ("laplace", 150.0, 1500), ("t3", 150.0, 1500), ("t3", 300.0, 1500)]


@pytest.mark.parametrize("shape,sd,n_cl", RESOLVED)
def test_resolved_cluster_without_attraction_is_not_an_attractor(shape, sd, n_cl):
    """Review R2, B1: a cluster wider than one bin, flat background. With the half-height zone 'attractor' came
    out in every set; with the baseline zone the reading must stay at or below 0.07."""
    n_sets = 100
    results = labels_of("flat", n_sets, int(sd) + n_cl, cluster=n_cl, cluster_sd=sd, cluster_shape=shape)
    attractor = np.mean([has(d, A.READING_ATTRACTOR) for d in results])
    regular = np.mean([has(d, A.READING_REGULAR) for d in results])
    assert attractor <= 0.07, attractor
    assert regular <= 0.07, regular


def test_half_height_zone_reads_the_flank_as_attraction():
    """The failure the baseline rule fixes, kept as a test: same data, zone at half height, 'attractor' everywhere
    at the default zone; the widened zones are what takes the label away."""
    rng = np.random.default_rng(71)
    flagged = robust = 0
    for i in range(20):
        d = A.pooled_details(axon_set(rng, "flat", cluster=1500, cluster_sd=150.0), n_null=199, seed=i,
                             rule="half_max")
        flagged += A.READING_ATTRACTOR in d.label
        robust += has(d, A.READING_ATTRACTOR)
    assert flagged >= 18 and robust < flagged


def test_heavy_tail_is_a_halo_by_any_measure_and_is_reported_as_the_documented_limit():
    """A Student-t (3 d.o.f.) cluster keeps a few per cent of its localizations a micron away. No zone rule can
    tell that from a halo; the module says so. This test only pins the behaviour: the excess shrinks as the zone
    widens (unlike a real halo of sigma 600 nm, where it stays), and the sensitivity table shows it."""
    d = A.pooled_details(axon_set(np.random.default_rng(72), "flat", cluster=1500, cluster_sd=150.0,
                                  cluster_shape="t3"), n_null=499)
    excess = [r.contrast_minus_null for r in d.sensitivity]
    assert [r.widen for r in d.sensitivity] == [1.0, 2.0, 3.0]
    assert excess[0] > excess[1] > excess[2] and excess[2] < 0.4 * excess[0]
    assert d.sensitivity[2].zone_nm_median > d.sensitivity[0].zone_nm_median


# ----------------------------------------------------------------------------------------------------
# The sign of the label (review R2, B2)
# ----------------------------------------------------------------------------------------------------
def test_sign_is_read_against_the_null_with_mixed_axons():
    """12 short axons with a small cluster (high relative background) and 6 long ones with a large cluster, all
    with a depleted flank. The raw pooled contrast used to be positive with a null mean above it."""
    wrong = 0
    for i in range(10):
        rng = np.random.default_rng(900 + i)
        profiles = [simulate(rng, "inhibition", length_nm=6000.0, cluster=60, hole_nm=1500.0, hole_level=0.35)
                    for _ in range(12)] + \
                   [simulate(rng, "inhibition", length_nm=30000.0, cluster=1500, hole_nm=1500.0, hole_level=0.35)
                    for _ in range(6)]
        d = A.pooled_details(profiles, n_null=499, seed=i)
        assert abs(d.result.contrast.null_mean) < 0.03                  # centred (was +0.22): own-mean normalisation
        assert d.contrast_minus_null == pytest.approx(d.result.contrast.value - d.result.contrast.null_mean)
        assert d.contrast_minus_null < 0 and d.result.contrast.z < 0
        wrong += A.READING_ATTRACTOR in d.label
    assert wrong == 0


def test_label_follows_value_minus_null_not_the_raw_value(monkeypatch):
    """Pin the rule itself: a raw value above zero but below its null mean is an inhibition zone."""
    from tools.mps_perimeter_types import TestReadout
    real = A._zone_run

    def shifted(*args, **kw):
        run, rows = real(*args, **kw)
        fake = TestReadout(value=0.111, null_mean=0.2165, null_sd=0.037, z=-2.85, p=0.008, n_null=499, sided="two")
        return A.ZoneRun(widen=run.widen, zone_nm_median=run.zone_nm_median, n_axons=run.n_axons, contrast=fake,
                         period=run.period, contrast_minus_null=fake.value - fake.null_mean,
                         period_nm=run.period_nm, period_at_edge=run.period_at_edge), rows

    monkeypatch.setattr(A, "_zone_run", shifted)
    d = A.pooled_details(axon_set(np.random.default_rng(1), "none"), n_null=99)
    assert d.result.contrast.value > 0 and A.READING_INHIBITION in d.label and A.READING_ATTRACTOR not in d.label


# ----------------------------------------------------------------------------------------------------
# Positive controls, pooled over 18 axons
# ----------------------------------------------------------------------------------------------------
def test_gaussian_halo_reads_attractor():
    results = labels_of("attractor", 30, 41)
    assert all(d.contrast_minus_null > 0 for d in results)
    assert np.mean([has(d, A.READING_ATTRACTOR) for d in results]) >= 0.8      # power with 18 axons, robust label
    assert not any(A.READING_INHIBITION in d.label for d in results)
    assert np.mean([A.READING_REGULAR in d.label for d in results]) <= 0.2     # a halo is not a repeat
    one = A.pooled_details(axon_set(np.random.default_rng(42), "attractor"), n_null=1999)
    assert one.label == A.READING_ATTRACTOR and one.attractor_robust and one.result.contrast.z > 3
    assert all(r.contrast_minus_null > 0 and r.contrast.p <= 0.05 for r in one.sensitivity)
    prof = one.result.profile
    near = (prof.lag_nm >= 400) & (prof.lag_nm <= 1000)
    assert np.any(prof.mean_relative[near] > prof.global_hi[near])             # leaves the global envelope


def test_gaussian_halo_around_a_resolved_cluster_is_seen_but_not_called_robust():
    """The price of the zone, pinned: a halo of sigma 600 nm around a cluster of sd 150 nm is still significant
    with the default zone (about +-650 nm), but it is gone 2 and 3 zone-widths out, exactly as the tail of a
    heavy-tailed cluster is. The module cannot tell them apart and says so in the label."""
    results = labels_of("attractor", 20, 47, cluster=1500, cluster_sd=150.0)
    assert all(d.contrast_minus_null > 0 for d in results)
    assert np.mean([A.READING_ATTRACTOR in d.label for d in results]) >= 0.8
    assert np.mean([has(d, A.READING_ATTRACTOR) for d in results]) <= 0.2
    assert all(A.READING_NOT_ROBUST in d.label for d in results if A.READING_ATTRACTOR in d.label
               and not d.attractor_robust)


def test_depleted_flank_reads_inhibition_zone():
    results = labels_of("inhibition", 15, 43)
    assert all(d.contrast_minus_null < 0 for d in results)
    assert np.mean([A.READING_INHIBITION in d.label for d in results]) >= 0.8
    assert not any(A.READING_ATTRACTOR in d.label for d in results)


def test_regular_modes_in_phase_with_the_maximum_read_regular():
    results = labels_of("regular", 15, 44)
    assert np.mean([has(d, A.READING_REGULAR) for d in results]) >= 0.8
    assert not any(A.READING_ATTRACTOR in d.label for d in results)
    one = A.pooled_details(axon_set(np.random.default_rng(45), "regular"), n_null=1999)
    assert has(one, A.READING_REGULAR) and one.regular_robust and not one.period_at_edge
    assert one.result.period.p <= 0.001 and one.period_p_holm <= 0.002
    assert abs(one.period_nm - 800.0) <= 100.0                                 # within one bin
    assert f"period peak {one.period_nm:.0f} nm" in one.result.reading


def test_one_axon_has_little_power_for_the_period_but_its_z_is_positive():
    """Documented limit: a remainder of period T nearly repeats after a shift by T; the pooled test has the power."""
    rng = np.random.default_rng(46)
    z, p = [], []
    for _ in range(40):
        r = A.axon_a_readouts(simulate(rng, "regular", per_bin=200.0))["period"]
        z.append(r.z)
        p.append(r.p)
    assert np.mean(z) > 1.0 and np.mean(np.array(p) <= 0.05) < 0.8


# ----------------------------------------------------------------------------------------------------
# Pooling: distances, animals, axes, sensitivity, structure
# ----------------------------------------------------------------------------------------------------
def test_n_axons_per_distance_with_different_perimeters():
    rng = np.random.default_rng(51)
    lengths = [6000.0, 6000.0, 12000.0, 32000.0]
    profiles = [simulate(rng, "flat", length_nm=x) for x in lengths]
    raw = A.pool_aligned(profiles)
    assert raw.axis == "nm" and raw.lag_nm[1] - raw.lag_nm[0] == pytest.approx(100.0)
    assert raw.lag_nm[-1] == pytest.approx(16000.0)
    expected = np.array([sum(x / 2 >= lag for x in lengths) for lag in raw.lag_nm])
    np.testing.assert_array_equal(raw.n_axons, expected)
    assert raw.n_axons[0] == 4 and raw.n_axons[31] == 2 and raw.n_axons[61] == 1
    assert np.all(raw.n_animals == 0)
    np.testing.assert_allclose(raw.mean_relative[100], A._fold(np.asarray(profiles[3].relative)[None, :],
                                                                 profiles[3].max_bin)[0][0][100])
    res = A.pooled_test(profiles, n_null=99)
    np.testing.assert_array_equal(res.profile.n_axons, expected)
    np.testing.assert_allclose(res.profile.mean_relative, raw.mean_relative)
    assert res.n_axons == 4 and "n = 4 axons" in res.reading
    # the global band is for display and only where at least GLOBAL_MIN_AXONS axons contribute
    few = res.profile.n_axons < A.GLOBAL_MIN_AXONS
    assert few.any() and np.all(np.isnan(res.profile.global_lo[few])) and np.all(np.isnan(res.profile.global_hi[few]))
    assert np.isfinite(res.profile.global_hi[10]) and np.isfinite(res.profile.null_hi[100])


def test_fraction_axis_is_the_alternative():
    rng = np.random.default_rng(52)
    profiles = [simulate(rng, "flat", length_nm=x) for x in (6000.0, 9000.0, 20000.0)]
    frac = A.pool_aligned(profiles, axis="fraction")
    assert frac.axis == "fraction" and frac.lag_nm[0] == 0.0 and frac.lag_nm[-1] == pytest.approx(0.5)
    np.testing.assert_array_equal(frac.n_axons, 3)                 # every axon reaches every fraction
    assert frac.mean_relative[0] > frac.mean_relative[3:].max()      # the artefact survives the change of axis
    res_f = A.pooled_test(profiles, n_null=99, seed=3, axis="fraction")
    res_n = A.pooled_test(profiles, n_null=99, seed=3)
    assert res_f.profile.axis == "fraction" and np.isfinite(res_f.profile.null_hi[5])
    assert res_f.contrast == res_n.contrast and res_f.period == res_n.period   # statistics always on nm
    with pytest.raises(ValueError):
        A.pool_aligned(profiles, axis="degrees")


def test_animals_labels_weights_and_scope():
    rng = np.random.default_rng(53)
    profiles = [simulate(rng, "flat", length_nm=10000.0) for _ in range(6)]
    none = A.pooled_details(profiles, n_null=99)
    assert none.result.profile.group == A.NO_ANIMAL_GROUP and A.NO_ANIMAL_GROUP in none.result.reading
    assert none.exploratory and none.n_animals == 0
    assert none.result.reading.startswith(A.PREFIX_EXPLORATORY)             # the caveat survives a truncated cell
    assert "no inference across animals or genotypes" in none.scope and none.scope in none.result.reading
    assert A.pool_aligned(profiles, group="WT").group == f"WT [{A.NO_ANIMAL_GROUP}]"
    one = A.pooled_details(profiles, n_null=99, animals=["provisional animal 1"] * 6, group="WT")
    assert one.result.profile.group == "WT" and np.all(one.result.profile.n_animals[:40] == 1)
    assert one.exploratory and one.n_animals == 1 and one.result.reading.startswith(A.PREFIX_EXPLORATORY)
    assert "n_animals = 1 (provisional animal 1): no comparison between animals is possible" in one.result.reading
    assert one.result.contrast == none.result.contrast                 # the label changes no number
    animals = ["a", "a", "a", "a", "b", "c"]
    by_animal = A.pool_aligned(profiles, weights="animal", animals=animals)
    folded = np.stack([A._fold(np.asarray(p.relative)[None, :], p.max_bin)[0][0] for p in profiles])
    manual = np.mean([folded[:4].mean(axis=0), folded[4], folded[5]], axis=0)
    np.testing.assert_allclose(by_animal.mean_relative, manual)
    assert np.all(by_animal.n_animals == 3) and np.all(by_animal.n_axons == 6)
    np.testing.assert_allclose(A.pool_aligned(profiles).mean_relative, folded.mean(axis=0))
    several = A.pooled_details(profiles, n_null=99, weights="animal", animals=animals)
    assert not several.exploratory and several.result.reading.startswith(A.PREFIX_WITHIN_SAMPLE)
    assert "n_animals = 3: this p is not an animal-level test" in several.result.reading
    # one animal one weight: the pooled contrast is the mean of the animals' mean contrasts
    per_axon = [A.axon_a_details(p).contrast.value for p in profiles]
    assert several.result.contrast.value == pytest.approx(np.mean([np.mean(per_axon[:4]), per_axon[4], per_axon[5]]))
    assert none.result.contrast.value == pytest.approx(np.mean(per_axon))
    with pytest.raises(ValueError):
        A.pool_aligned(profiles, weights="animal", animals=["a", "", "a", "a", "b", "c"])
    with pytest.raises(ValueError):
        A.pool_aligned(profiles, animals=["a"])


def test_pooled_details_are_structured_and_agree_with_the_reading():
    d = A.pooled_details(axon_set(np.random.default_rng(42), "attractor"), n_null=499, seed=2)
    r = d.result
    assert A.reading_label(r) == d.label == A.READING_ATTRACTOR
    assert r.reading.split(A.READING_SEPARATOR)[0] == A.PREFIX_EXPLORATORY + d.label
    assert (d.contrast_p_holm, d.period_p_holm) == A.holm_two(r.contrast.p, r.period.p)
    assert d.sensitivity[0].contrast == r.contrast and d.sensitivity[0].period == r.period
    assert [s.widen for s in d.sensitivity] == [1.0, *A.ROBUST_WIDEN]
    assert d.zone_nm_min <= d.zone_nm_median <= d.zone_nm_max and d.zone_nm_min >= 400.0   # +-200 nm at least
    assert d.sensitivity[1].zone_nm_median > d.sensitivity[0].zone_nm_median
    assert d.exclusion_rule == "baseline" and d.exclusion_min_nm == 200.0 and d.alpha == 0.05
    assert f"zone baseline, median {d.zone_nm_median:.0f} nm" in r.reading and "widened x2" in r.reading
    assert A.pooled_test(axon_set(np.random.default_rng(42), "attractor"), n_null=499, seed=2).reading == r.reading


def test_holm_and_reading_vocabulary():
    assert A.holm_two(0.01, 0.04) == (0.02, 0.04)
    assert A.holm_two(0.04, 0.03) == (0.06, 0.06)
    assert A.holm_two(0.6, 0.9) == (1.0, 1.0)
    a, b = A.holm_two(0.01, float("nan"))
    assert a == 0.01 and np.isnan(b)
    assert (A.READING_ATTRACTOR, A.READING_INHIBITION, A.READING_REGULAR, A.READING_NONE) == \
        ("attractor", "inhibition zone", "regular", "no structure detectable")
    assert A.READING_NOT_ROBUST == " (not robust to the exclusion zone: cluster flank)"


def test_zone_sensitivity_table_per_axon():
    """Per axon: a heavy-tailed cluster reads as attraction with a narrow fixed zone; widening lowers it."""
    rng = np.random.default_rng(61)
    names = ("baseline", "baseline_x2", "baseline_x3", "half_max", "fixed_200nm", "fixed_400nm")
    z = {k: [] for k in names}
    for _ in range(40):
        table = A.zone_sensitivity(simulate(rng, "flat", length_nm=14000.0, cluster=1500, cluster_sd=150.0,
                                            cluster_shape="t3"))
        assert set(table) == set(names)
        for k in names:
            z[k].append(table[k]["contrast"].z)
    assert np.mean(z["fixed_200nm"]) > 1.0                       # heavy flanks read as attraction: the known limit
    assert np.mean(z["fixed_200nm"]) > np.mean(z["fixed_400nm"]) + 0.3
    assert np.mean(z["baseline_x3"]) < np.mean(z["fixed_200nm"]) - 1.0


def test_module_is_pure_numpy_and_names_nobody():
    src = (Path(__file__).resolve().parent / "tools" / "mps_perimeter_a.py").read_text(encoding="utf-8")
    assert "PyQt" not in src and "pyqtgraph" not in src and "scipy" not in src
    needle = "un" + "sa"
    for path in (Path(__file__), Path(__file__).resolve().parent / "tools" / "mps_perimeter_a.py"):
        assert needle + "in" not in path.read_text(encoding="utf-8").lower()
