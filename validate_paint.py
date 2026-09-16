# -*- coding: utf-8 -*-
"""
Checks for the DNA-PAINT modules, against synthetic ground truth.

Every quantity here is one where a plausible-looking wrong answer is easy
to produce and hard to notice, so each is checked against data built with
the answer known in advance:

  * linking, against events placed by hand;
  * dark times, against gaps placed by hand;
  * tau_dark, against an exponential simulated with a known mean;
  * qPAINT, against sites simulated with a known valency (1, 2, 4, 8);
  * NeNA, against localizations displaced by a known precision.

The qPAINT case is the one that matters most. The whole method rests on
tau_dark falling as 1/N, and nothing about the output looks wrong if that
relationship is off by a factor.

Run:  python validate_paint.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_paint import (  # noqa: E402
    FRAGMENTATION_EXCESS,
    build_events,
    dark_times,
    estimate_kinetic_rate,
    frame_analysis,
    influx_from_reference,
    kinetics,
    link_localizations,
    qpaint,
)
from tools.mps_quality import (  # noqa: E402
    axial_resolvedness,
    box_size_check,
    drift_check,
    nena,
)

PASSED = 0
FAILED = 0


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


# ------------------------------------------------------------ simulation
def simulate_site(
    rng: np.random.Generator,
    position: Tuple[float, float],
    n_frames: int,
    influx_per_frame: float,
    n_sites: int,
    tau_bright: float,
    precision_nm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One docking-site cluster of ``n_sites`` strands, imaged for
    ``n_frames``.

    Each of the N strands binds independently as a Poisson process of
    rate ``influx_per_frame``, so the cluster's aggregate rate is
    N * influx and its mean dark time is 1 / (N * influx). That IS the
    qPAINT relationship, generated here rather than assumed.
    """
    frames: List[int] = []
    for _ in range(n_sites):
        time = rng.exponential(1.0 / influx_per_frame)
        while time < n_frames:
            duration = max(1, int(round(rng.exponential(tau_bright))))
            for step in range(duration):
                if time + step < n_frames:
                    frames.append(int(time + step))
            time += duration + rng.exponential(1.0 / influx_per_frame)
    frame = np.array(sorted(frames), dtype=np.int64)
    x = position[0] + rng.normal(0.0, precision_nm, frame.size)
    y = position[1] + rng.normal(0.0, precision_nm, frame.size)
    return frame, x, y


# ================================================================ linking
def test_linking() -> None:
    print("\n1. LINKING LOCALIZATIONS INTO BINDING EVENTS")

    def consecutive_frames_link():
        frame = np.array([10, 11, 12, 50, 51], dtype=np.int64)
        x = np.zeros(5)
        y = np.zeros(5)
        link = link_localizations(frame, x, y, radius_nm=30.0, max_dark_time=1)
        assert link[0] == link[1] == link[2], link
        assert link[3] == link[4], link
        assert link[0] != link[3], link
        return "3+2 localizations -> 2 events"

    def max_dark_time_respected():
        # One frame missing in the middle.
        frame = np.array([10, 12], dtype=np.int64)
        x = y = np.zeros(2)
        joined = link_localizations(frame, x, y, 30.0, max_dark_time=1)
        split = link_localizations(frame, x, y, 30.0, max_dark_time=0)
        assert joined[0] == joined[1], "a 1-frame gap should be tolerated"
        assert split[0] != split[1], "max_dark_time=0 must split it"
        return "gap of 1 joined at t=1, split at t=0"

    def radius_respected():
        frame = np.array([10, 11], dtype=np.int64)
        x = np.array([0.0, 100.0])
        y = np.zeros(2)
        assert link_localizations(frame, x, y, 30.0, 1)[0] != \
            link_localizations(frame, x, y, 30.0, 1)[1]
        assert link_localizations(frame, x, y, 150.0, 1)[0] == \
            link_localizations(frame, x, y, 150.0, 1)[1]
        return "100 nm apart: split at r=30, joined at r=150"

    def groups_never_span():
        frame = np.array([10, 11], dtype=np.int64)
        x = y = np.zeros(2)
        group = np.array([0, 1], dtype=np.int64)
        link = link_localizations(frame, x, y, 30.0, 1, group)
        assert link[0] != link[1]
        return "an event cannot span two clusters"

    def same_frame_never_linked():
        # Two molecules in the same frame are two molecules, whatever
        # their distance: the search window starts at frame + 1.
        frame = np.array([10, 10], dtype=np.int64)
        x = y = np.zeros(2)
        link = link_localizations(frame, x, y, 1000.0, 3)
        assert link[0] != link[1]
        return "two localizations in one frame stay separate"

    def input_order_preserved():
        frame = np.array([50, 10, 51, 11], dtype=np.int64)
        x = y = np.zeros(4)
        link = link_localizations(frame, x, y, 30.0, 1)
        assert link[1] == link[3], "the two frame-10/11 locs are one event"
        assert link[0] == link[2], "the two frame-50/51 locs are one event"
        assert link[0] != link[1]
        return "unsorted input, labels returned in input order"

    def event_aggregation():
        frame = np.array([10, 11, 12], dtype=np.int64)
        x = np.array([0.0, 10.0, 20.0])
        y = np.zeros(3)
        photons = np.array([100.0, 200.0, 300.0])
        ev = build_events(frame, x, y, photons=photons, radius_nm=30.0,
                          max_dark_time=1)
        assert ev.n == 1
        assert ev.first_frame[0] == 10 and ev.last_frame[0] == 12
        assert ev.length[0] == 3 and ev.n_locs[0] == 3
        assert np.isclose(ev.photons[0], 600.0)
        assert np.isclose(ev.photon_rate[0], 200.0)
        assert np.isclose(ev.x_nm[0], 10.0)
        return "len, n, photons, mean position"

    def precision_weighting():
        frame = np.array([10, 11], dtype=np.int64)
        x = np.array([0.0, 100.0])
        y = np.zeros(2)
        # The second localization is 10x less precise, so it should pull
        # the mean 100x less (weights go as 1/lp^2).
        lp = np.array([1.0, 10.0])
        ev = build_events(frame, x, y, lp_nm=lp, radius_nm=200.0,
                          max_dark_time=1)
        assert ev.n == 1
        assert np.isclose(ev.x_nm[0], 100.0 / 101.0), ev.x_nm[0]
        return "inverse-variance weighted position"

    check("consecutive frames link", consecutive_frames_link)
    check("max_dark_time", max_dark_time_respected)
    check("linking radius", radius_respected)
    check("events never span groups", groups_never_span)
    check("same frame never linked", same_frame_never_linked)
    check("labels returned in input order", input_order_preserved)
    check("event aggregation", event_aggregation)
    check("inverse-variance weighting", precision_weighting)


# ============================================================= dark times
def test_dark_times() -> None:
    print("\n2. DARK TIMES")

    def simple_gaps():
        # events: [10-12], [20-21], [40-40]
        first = np.array([10, 20, 40], dtype=np.int64)
        last = np.array([12, 21, 40], dtype=np.int64)
        dark = dark_times(first, last)
        assert dark[0] == -1, dark
        assert dark[1] == 20 - 12, dark   # 8
        assert dark[2] == 40 - 21, dark   # 19
        return "first event -1, then 8 and 19 frames"

    def per_group():
        first = np.array([10, 20, 12, 30], dtype=np.int64)
        last = np.array([11, 21, 13, 31], dtype=np.int64)
        group = np.array([0, 0, 1, 1], dtype=np.int64)
        dark = dark_times(first, last, group)
        assert dark[0] == -1 and dark[2] == -1
        assert dark[1] == 20 - 11, dark
        assert dark[3] == 30 - 13, dark
        return "gaps measured within each site only"

    def nearest_preceding_end():
        # An event that ends late must be the one the gap is measured
        # from, even if another started earlier.
        first = np.array([10, 12, 30], dtype=np.int64)
        last = np.array([25, 13, 30], dtype=np.int64)
        dark = dark_times(first, last)
        assert dark[2] == 30 - 25, dark
        return "measured from the latest end, not the latest start"

    def unsorted_input():
        first = np.array([40, 10, 20], dtype=np.int64)
        last = np.array([40, 12, 21], dtype=np.int64)
        dark = dark_times(first, last)
        assert dark[1] == -1 and dark[2] == 8 and dark[0] == 19, dark
        return "order preserved"

    check("gaps between consecutive events", simple_gaps)
    check("dark times are per site", per_group)
    check("measured from the latest end", nearest_preceding_end)
    check("unsorted input", unsorted_input)


# ======================================================== frame analysis
def test_frame_analysis() -> None:
    print("\n3. STICKING REJECTION (frame analysis)")
    rng = np.random.default_rng(1)
    n_frames = 10000

    def real_site_passes():
        frame = rng.integers(0, n_frames, 400)
        labels = np.zeros(400, dtype=np.int64)
        fa = frame_analysis(labels, frame, n_frames)
        assert fa.passed[0], (fa.mean_frame_fraction, fa.max_window_share)
        return "uniform in time -> kept"

    def early_burst_rejected():
        frame = rng.integers(0, 300, 400)          # first 3 % of the movie
        labels = np.zeros(400, dtype=np.int64)
        fa = frame_analysis(labels, frame, n_frames)
        assert not fa.passed[0]
        assert fa.n_rejected == 1
        return "burst in the first 3 % -> rejected"

    def late_burst_rejected():
        frame = rng.integers(n_frames - 300, n_frames, 400)
        labels = np.zeros(400, dtype=np.int64)
        assert not frame_analysis(labels, frame, n_frames).passed[0]
        return "burst at the end -> rejected"

    def centred_but_concentrated_rejected():
        # Mean frame is dead centre, so the first criterion passes, but
        # everything happens inside one twentieth of the movie (the
        # windows are [5000, 5500) and so on for a 10000-frame movie).
        frame = rng.integers(5050, 5250, 400)
        labels = np.zeros(400, dtype=np.int64)
        fa = frame_analysis(labels, frame, n_frames)
        assert 0.2 < fa.mean_frame_fraction[0] < 0.8, "mean frame is central"
        assert not fa.passed[0], "but it is concentrated in time"
        return "central mean, concentrated -> still rejected"

    def straddling_burst_escapes():
        # A KNOWN BLIND SPOT of Picasso's rule, kept deliberately rather
        # than patched: the second criterion counts localizations in
        # twenty FIXED windows, so a burst sitting across a window
        # boundary is split between two of them and neither reaches the
        # 80 % threshold. This burst is just as concentrated as the one
        # above -- 200 frames of a 10000-frame movie -- and passes.
        frame = rng.integers(4900, 5100, 400)
        labels = np.zeros(400, dtype=np.int64)
        fa = frame_analysis(labels, frame, n_frames)
        assert fa.passed[0], "documenting the blind spot; it should pass"
        assert fa.max_window_share[0] < 0.8
        return (f"burst across a window edge passes "
                f"(max share {fa.max_window_share[0]:.2f})")

    def mixed_population():
        frame = np.concatenate([
            rng.integers(0, n_frames, 300),
            rng.integers(0, 200, 300),
        ])
        labels = np.concatenate([
            np.zeros(300, dtype=np.int64), np.ones(300, dtype=np.int64)
        ])
        fa = frame_analysis(labels, frame, n_frames)
        assert list(fa.passed) == [True, False], fa.passed
        assert list(fa.labels) == [0, 1]
        return "one kept, one rejected, labels aligned"

    check("a real docking site passes", real_site_passes)
    check("an early burst is rejected", early_burst_rejected)
    check("a late burst is rejected", late_burst_rejected)
    check("concentrated but centred is rejected", centred_but_concentrated_rejected)
    check("a burst across a window edge escapes (known blind spot)",
          straddling_burst_escapes)
    check("labels stay aligned with verdicts", mixed_population)


# ================================================================ kinetics
def test_kinetics() -> None:
    print("\n4. KINETICS")
    rng = np.random.default_rng(2)

    def recovers_exponential_mean():
        for true_tau in (20.0, 100.0, 500.0):
            sample = rng.exponential(true_tau, 4000)
            estimate = estimate_kinetic_rate(sample)
            assert abs(estimate - true_tau) / true_tau < 0.15, \
                (true_tau, estimate)
        return "tau recovered within 15 % at 20, 100, 500 frames"

    def cdf_fit_beats_the_mean_when_censored():
        # Dark periods longer than the acquisition are never observed.
        # The sample mean is then biased low; the CDF fit should be less so.
        true_tau = 400.0
        movie = 1000.0
        sample = rng.exponential(true_tau, 4000)
        observed = sample[sample < movie]
        by_mean = float(np.mean(observed))
        by_fit = estimate_kinetic_rate(observed)
        assert by_mean < true_tau * 0.85, by_mean
        assert abs(by_fit - true_tau) < abs(by_mean - true_tau), \
            (by_fit, by_mean, true_tau)
        return (f"censored at {movie:.0f}: mean {by_mean:.0f}, "
                f"fit {by_fit:.0f}, true {true_tau:.0f}")

    def degenerate_inputs():
        assert np.isnan(estimate_kinetic_rate([]))
        assert estimate_kinetic_rate([5.0]) == 5.0
        assert estimate_kinetic_rate([7.0, 7.0, 7.0, 7.0]) == 7.0
        return "empty, single, constant"

    check("recovers a known exponential mean", recovers_exponential_mean)
    check("less biased than the mean under censoring",
          cdf_fit_beats_the_mean_when_censored)
    check("degenerate inputs", degenerate_inputs)


# ================================================================== qPAINT
def test_qpaint() -> None:
    print("\n5. qPAINT COUNTING")
    rng = np.random.default_rng(3)
    # Long enough that even the single-site cluster gets several
    # hundred binding events: at 40,000 frames it got 88, and the
    # scatter of tau_dark over 88 exponential samples (SE = 11 %)
    # swamps the effect being tested.
    n_frames = 250000
    influx = 1.0 / 400.0          # one binding per 400 frames per strand
    tau_bright = 4.0
    valencies = [1, 2, 4, 8]

    frames: List[np.ndarray] = []
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for i, valency in enumerate(valencies):
        f, x, y = simulate_site(
            rng, (1000.0 * i, 0.0), n_frames, influx, valency,
            tau_bright, precision_nm=5.0,
        )
        frames.append(f)
        xs.append(x)
        ys.append(y)
        labels.append(np.full(f.size, i, dtype=np.int64))

    frame = np.concatenate(frames)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    label = np.concatenate(labels)
    events = build_events(frame, x, y, group=label, radius_nm=30.0,
                          max_dark_time=1)

    def tau_dark_scales_as_one_over_n():
        result = qpaint(events, events.group,
                        tau_dark_reference_frames=1.0 / influx)
        tau = result.tau_dark_frames
        for i, valency in enumerate(valencies):
            expected = 1.0 / (valency * influx)
            assert abs(tau[i] - expected) / expected < 0.15, \
                (valency, tau[i], expected)
        return "  ".join(
            f"N={v}: tau {tau[i]:.0f} (expect {1/(v*influx):.0f})"
            for i, v in enumerate(valencies)
        )

    def counts_recovered():
        result = qpaint(events, events.group,
                        tau_dark_reference_frames=1.0 / influx)
        assert result.is_calibrated
        for i, valency in enumerate(valencies):
            assert abs(result.n_units[i] - valency) / valency < 0.15, \
                (valency, result.n_units[i])
        return "  ".join(
            f"{v} -> {result.n_units[i]:.2f}"
            for i, v in enumerate(valencies)
        )

    def influx_helper_agrees():
        by_rate = qpaint(events, events.group, influx_rate=influx)
        by_ref = qpaint(events, events.group,
                        tau_dark_reference_frames=1.0 / influx)
        assert np.allclose(by_rate.n_units, by_ref.n_units, equal_nan=True)
        assert np.isclose(influx_from_reference(1.0 / influx), influx)
        return "influx_rate and a single-site reference agree"

    def uncalibrated_is_flagged():
        result = qpaint(events, events.group)
        assert not result.is_calibrated
        assert any("RELATIVE" in w for w in result.warnings), result.warnings
        # Relative ordering must still be right: more sites, more units.
        assert np.all(np.diff(result.n_units) > 0), result.n_units
        return "flagged, but the ranking is still monotonic"

    def both_calibrations_refused():
        try:
            qpaint(events, events.group, influx_rate=influx,
                   tau_dark_reference_frames=100.0)
        except ValueError:
            return "giving both raises"
        raise AssertionError("accepted two conflicting calibrations")

    def sparse_cluster_is_nan_not_zero():
        f = np.array([10, 11, 5000, 5001], dtype=np.int64)
        xx = np.zeros(4)
        lab = np.zeros(4, dtype=np.int64)
        ev = build_events(f, xx, xx, group=lab, radius_nm=30.0,
                          max_dark_time=1)
        result = qpaint(ev, ev.group, influx_rate=influx, min_events=5)
        assert np.isnan(result.n_units[0]), result.n_units
        assert any("NaN" in w for w in result.warnings)
        return "too few events -> NaN, never 0"

    check("tau_dark falls as 1/N", tau_dark_scales_as_one_over_n)
    check("valency recovered from a single-site reference", counts_recovered)
    check("the two calibration routes agree", influx_helper_agrees)
    check("uncalibrated output is flagged", uncalibrated_is_flagged)
    check("two calibrations at once are refused", both_calibrations_refused)
    check("a cluster with too few events is NaN", sparse_cluster_is_nan_not_zero)

    def kinetics_summary():
        result = kinetics(events, site_labels=events.group,
                          exposure_s=0.05, n_frames=n_frames)
        assert result.n_sites == len(valencies)
        assert np.isfinite(result.tau_bright_frames)
        assert result.tau_bright_s == result.tau_bright_frames * 0.05
        # tau_bright was simulated at 4 frames.
        assert 2.0 < result.tau_bright_frames < 8.0, result.tau_bright_frames
        return (f"tau_bright {result.tau_bright_frames:.1f} frames "
                f"(simulated 4), {result.n_sites} sites")

    check("kinetics summary", kinetics_summary)

    def fragmentation_is_caught():
        # Simulate the failure the project's own DNA-PAINT data showed:
        # binding events long enough that max_dark_time splits them, so
        # the dark-time distribution gains a spike of 1-2 frame gaps that
        # are not unbindings at all.
        rng2 = np.random.default_rng(11)
        f_list, x_list, l_list = [], [], []
        for site in range(30):
            t = 0.0
            while t < 20000:
                # A long event that flickers: on, 3 dark, on again.
                start = int(t)
                for offset in (0, 1, 2, 5, 6, 7):
                    f_list.append(start + offset)
                    l_list.append(site)
                t += 6 + rng2.exponential(600.0)
        frame_f = np.array(f_list, dtype=np.int64)
        lab_f = np.array(l_list, dtype=np.int64)
        xx = np.zeros(frame_f.size) + lab_f * 1000.0
        ev = build_events(frame_f, xx, np.zeros_like(xx), group=lab_f,
                          radius_nm=30.0, max_dark_time=1)
        res = kinetics(ev, site_labels=ev.group, n_frames=20000)
        assert any("split in two" in w for w in res.warnings), res.warnings
        assert res.tau_dark_excluding_short_frames > res.tau_dark_frames
        return (f"tau_dark {res.tau_dark_frames:.0f} vs "
                f"{res.tau_dark_excluding_short_frames:.0f} without the "
                f"{100*res.short_gap_fraction:.0f} % short gaps -- flagged")

    def clean_kinetics_not_flagged():
        # The well-behaved simulation from above must NOT trip the check.
        res = kinetics(events, site_labels=events.group, n_frames=n_frames)
        assert not any("split in two" in w for w in res.warnings), res.warnings
        return "no false alarm on clean data"

    def qpaint_can_exclude_short_gaps():
        a = qpaint(events, events.group, influx_rate=influx)
        b = qpaint(events, events.group, influx_rate=influx, min_dark_frames=2)
        assert np.allclose(a.n_units, b.n_units, rtol=0.35, equal_nan=True)
        return "excluding short gaps leaves clean data alone"

    check("event fragmentation is caught", fragmentation_is_caught)
    check("clean data is not flagged", clean_kinetics_not_flagged)
    check("qpaint min_dark_frames", qpaint_can_exclude_short_gaps)

    def degenerate_kinetics_fields_land_correctly():
        # The early returns build Kinetics by keyword, not position: two
        # float fields sit before `warnings`, and a positional call put
        # the warning list into one of them. mypy caught it; this keeps it
        # caught.
        from tools.mps_paint import BindingEvents
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=float)
        nothing = BindingEvents(empty_f, empty_f, None, empty_i, empty_i,
                                empty_i, empty_i, None, None, None,
                                empty_i, 25.0, 1)
        res = kinetics(nothing)
        assert isinstance(res.tau_dark_excluding_short_frames, float)
        assert isinstance(res.warnings, list) and res.warnings
        # And the all-noise path.
        ev2 = build_events(np.array([10, 11], dtype=np.int64), np.zeros(2),
                           np.zeros(2), group=np.array([-1, -1], dtype=np.int64),
                           radius_nm=30.0, max_dark_time=1)
        res2 = kinetics(ev2, site_labels=ev2.group)
        assert isinstance(res2.tau_dark_excluding_short_frames, float)
        assert res2.n_sites == 0
        return "no events, and all-noise, both keep their field types"

    check("degenerate Kinetics fields", degenerate_kinetics_fields_land_correctly)


# ================================================================== NeNA
def test_nena() -> None:
    print("\n6. NeNA (tools/mps_quality.py)")
    rng = np.random.default_rng(4)

    def recovers_known_precision():
        detail = []
        for true_sigma in (5.0, 10.0, 20.0):
            n_molecules = 4000
            true_x = rng.uniform(0, 20000, n_molecules)
            true_y = rng.uniform(0, 20000, n_molecules)
            # Each molecule seen in two consecutive frames.
            frame = np.repeat(rng.integers(0, 2000, n_molecules), 2)
            frame[1::2] += 1
            x = np.repeat(true_x, 2) + rng.normal(0, true_sigma, 2 * n_molecules)
            y = np.repeat(true_y, 2) + rng.normal(0, true_sigma, 2 * n_molecules)
            result = nena(frame, x, y, max_distance_nm=8 * true_sigma,
                          bin_nm=true_sigma / 20)
            assert result.converged, result.warnings
            error = abs(result.precision_nm - true_sigma) / true_sigma
            assert error < 0.15, (true_sigma, result.precision_nm)
            detail.append(f"{true_sigma:.0f}->{result.precision_nm:.1f}")
        return "  ".join(detail)

    def too_few_pairs_is_reported():
        frame = np.array([0, 1, 2], dtype=np.int64)
        x = y = np.zeros(3)
        result = nena(frame, x, y)
        assert not result.converged
        assert np.isnan(result.precision_nm)
        assert any("pairs" in w for w in result.warnings)
        return "reported, not guessed"

    check("recovers a known localization precision", recovers_known_precision)
    check("too few pairs is reported, not faked", too_few_pairs_is_reported)


# ====================================================== other quality checks
def test_quality_checks() -> None:
    print("\n7. OTHER QUALITY CHECKS")
    rng = np.random.default_rng(5)

    def box_clipping_detected():
        sx = np.full(1000, 2.5)
        sy = np.full(1000, 2.5)
        tight = box_size_check(sx, sy, box_size_px=9)
        assert tight.fraction_over_box_2sigma == 1.0
        assert any("truncated" in w for w in tight.warnings)
        roomy = box_size_check(sx, sy, box_size_px=21)
        assert roomy.fraction_over_box_3sigma == 0.0
        assert not roomy.warnings
        return "sigma 2.5 px: 9 px box flagged, 21 px box clean"

    def missing_box_size_is_reported():
        result = box_size_check(np.full(10, 1.0), np.full(10, 1.0), None)
        assert np.isnan(result.fraction_over_box_2sigma)
        assert any("Box Size" in w for w in result.warnings)
        return None

    def resolvedness_arithmetic():
        z = rng.normal(0, 60, 5000)
        lpz = np.full(5000, 47.0)
        # A component of sigma 47 is exactly precision-limited; one of
        # sigma 94 carries sqrt(94^2 - 47^2) = 81 nm of real width.
        result = axial_resolvedness(z, lpz, [0.0, 0.0], [47.0, 94.0])
        assert result.components[0].is_resolved
        assert np.isclose(result.components[0].ratio, 1.0, atol=0.05)
        assert np.isclose(result.components[0].structural_width_nm, 0.0,
                          atol=1.0)
        assert not result.components[1].is_resolved
        assert np.isclose(result.components[1].structural_width_nm,
                          np.sqrt(94.0**2 - 47.0**2), atol=1.0)
        return "sigma=lpz -> 0 nm structure; sigma=2lpz -> 81 nm"

    def impossible_component_flagged():
        z = rng.normal(0, 60, 5000)
        lpz = np.full(5000, 47.0)
        result = axial_resolvedness(z, lpz, [0.0], [20.0])
        assert any("not\nphysically possible" in w.replace(" ", "\n") or
                   "physically possible" in w for w in result.warnings), \
            result.warnings
        return "sigma < lpz is called out"

    def drift_detected_when_real():
        n = 40000
        frame = np.sort(rng.integers(0, 20000, n))
        true_drift = 150.0 * frame / 20000.0     # a clean 150 nm ramp
        x = rng.normal(0, 20, n) + true_drift
        y = rng.normal(0, 20, n)
        result = drift_check(frame, x, y, n_permutations=500)
        by_axis = {a.axis: a for a in result.axes}
        assert by_axis["x"].is_coherent, by_axis["x"]
        assert not by_axis["y"].is_coherent, by_axis["y"]
        return (f"x: {by_axis['x'].shift_range_nm:.0f} nm coherent "
                f"(p={by_axis['x'].p_permutation:.3f}); y: noise")

    def no_drift_is_not_invented():
        n = 40000
        frame = np.sort(rng.integers(0, 20000, n))
        x = rng.normal(0, 20, n)
        y = rng.normal(0, 20, n)
        result = drift_check(frame, x, y, n_permutations=500)
        assert not any(a.is_coherent for a in result.axes), result.axes
        return "static data -> no drift claimed"

    check("a clipping fitting box is detected", box_clipping_detected)
    check("a missing Box Size is reported", missing_box_size_is_reported)
    check("resolvedness arithmetic", resolvedness_arithmetic)
    check("a component narrower than lpz is flagged", impossible_component_flagged)
    check("real drift is detected", drift_detected_when_real)
    check("absent drift is not invented", no_drift_is_not_invented)


def main() -> int:
    print("=" * 72)
    print("DNA-PAINT AND DATA-QUALITY CHECKS")
    print("=" * 72)
    test_linking()
    test_dark_times()
    test_frame_analysis()
    test_kinetics()
    test_qpaint()
    test_nena()
    test_quality_checks()
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
