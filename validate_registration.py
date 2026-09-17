# -*- coding: utf-8 -*-
"""
Checks for tools/mps_registration.py: markers, pairing, the measured
registration, and reading the calibration tool's output.

The markers are simulated as in an Exchange-PAINT experiment: two rounds
of the same field, each marker localized in nearly every frame, the second
round shifted by a known amount and each marker displaced by a known
registration error. The background is sparse DNA-PAINT binding, which must
never be taken for a marker.

Run:  python validate_registration.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_io import FORMAT_PICASSO_HDF5, Localizations  # noqa: E402
from tools.mps_registration import (  # noqa: E402
    FIDUCIAL_MIN_FRAME_FRACTION,
    NO_REGISTRATION,
    RMS_PER_MEDIAN_2D,
    Fiducial,
    Registration,
    combine,
    export_registration,
    find_fiducials,
    match_fiducials,
    read_calibration,
    register_from_fiducials,
    register_localizations,
)

PASSED = 0
FAILED = 0
_TEMP_DIRS: list = []

REAL_SAMPLE = os.environ.get(
    "MPS_PAINT_SAMPLE",
    r"C:\Users\nicol\OneDrive\Doctorado\15.07.26"
    r"\260713_DNAPAINT_NCtransversal_bIIspt_TIRF4_Roi2_2_1"
    r"\260713_DNAPAINT_NCtransversal_bIIspt_TIRF4_Roi2_2_1_MMStack.ome_locs.hdf5",
)


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def new_tmp(prefix: str) -> str:
    path = tempfile.mkdtemp(prefix=prefix)
    _TEMP_DIRS.append(path)
    return path


# ------------------------------------------------------------ simulation
def simulate_round(
    rng: np.random.Generator,
    markers: np.ndarray,
    *,
    n_frames: int = 2000,
    presence: float = 0.95,
    marker_precision: Tuple[float, float] = (2.0, 5.0),
    n_sites: int = 3000,
    site_precision: float = 10.0,
    field_nm: float = 20000.0,
    three_d: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One round: markers in ``presence`` of frames, plus sparse binding."""
    frames, xs, ys, zs = [], [], [], []
    for mx, my, mz in markers:
        on = np.nonzero(rng.random(n_frames) < presence)[0]
        frames.append(on)
        xs.append(mx + rng.normal(0, marker_precision[0], on.size))
        ys.append(my + rng.normal(0, marker_precision[0], on.size))
        zs.append(mz + rng.normal(0, marker_precision[1], on.size))
    sites = rng.uniform(0, field_nm, (n_sites, 2))
    site_z = rng.uniform(-300, 300, n_sites)
    per_site = rng.poisson(4, n_sites)
    for (sx, sy), sz, k in zip(sites, site_z, per_site):
        if k == 0:
            continue
        start = rng.integers(0, n_frames - 20)
        on = np.unique(start + rng.integers(0, 20, k))
        frames.append(on)
        xs.append(sx + rng.normal(0, site_precision, on.size))
        ys.append(sy + rng.normal(0, site_precision, on.size))
        zs.append(sz + rng.normal(0, 3 * site_precision, on.size))
    frame = np.concatenate(frames).astype(np.int64)
    x, y, z = (np.concatenate(v) for v in (xs, ys, zs))
    if not three_d:
        z = np.zeros_like(z)
    order = np.argsort(frame, kind="stable")
    return frame[order], x[order], y[order], z[order]


def exchange_pair(
    seed: int,
    *,
    n_markers: int = 12,
    shift: Tuple[float, float, float] = (840.0, -1260.0, 35.0),
    registration_sigma: Tuple[float, float] = (6.0, 12.0),
    **round_kwargs,
):
    """Round A, round B, the true markers of A, and the true shift B -> A."""
    rng = np.random.default_rng(seed)
    markers = np.column_stack([
        rng.uniform(1000, 19000, n_markers),
        rng.uniform(1000, 19000, n_markers),
        rng.uniform(-20, 20, n_markers),
    ])
    noise = np.column_stack([
        rng.normal(0, registration_sigma[0], n_markers),
        rng.normal(0, registration_sigma[0], n_markers),
        rng.normal(0, registration_sigma[1], n_markers),
    ])
    markers_b = markers - np.asarray(shift) + noise
    a = simulate_round(rng, markers, **round_kwargs)
    b = simulate_round(rng, markers_b, **round_kwargs)
    return a, b, markers, np.asarray(shift)


def as_localizations(round_, path: str, pixel: float = 130.0,
                     n_frames: int = 2000) -> Localizations:
    frame, x, y, z = round_
    return Localizations(
        x_nm=x, y_nm=y, z_nm=z, path=path,
        fileformat=FORMAT_PICASSO_HDF5, pixel_size_nm=pixel,
        pixel_size_source="yaml",
        columns={"frame": frame, "x": x / pixel, "y": y / pixel, "z": z},
        info=[{"Frames": n_frames, "Pixelsize": pixel}],
        metadata_source="yaml",
    )


# ================================================================ markers
def test_markers() -> None:
    print("\n1. FINDING MARKERS")

    def finds_exactly_the_markers():
        (frame, x, y, z), _, markers, _ = exchange_pair(1)
        found = find_fiducials(frame, x, y, z, n_frames=2000)
        assert len(found) == len(markers), len(found)
        got = np.array([[f.x_nm, f.y_nm, f.z_nm] for f in found])
        worst = 0.0
        for m in markers:
            d = np.hypot(got[:, 0] - m[0], got[:, 1] - m[1])
            k = int(np.argmin(d))
            worst = max(worst, float(d[k]))
            assert abs(got[k, 2] - m[2]) < 2.0, (got[k, 2], m[2])
        assert worst < 1.0, worst
        return (f"{len(found)} of {len(markers)} markers among "
                f"{frame.size:,} localizations; worst position error "
                f"{worst:.2f} nm")

    def binding_sites_are_not_markers():
        rng = np.random.default_rng(2)
        frame, x, y, z = simulate_round(rng, np.empty((0, 3)),
                                        n_sites=6000)
        assert find_fiducials(frame, x, y, z, n_frames=2000) == []
        return f"{frame.size:,} localizations, no marker"

    def persistent_but_not_enough():
        rng = np.random.default_rng(3)
        markers = np.array([[5000.0, 5000.0, 0.0]])
        frame, x, y, z = simulate_round(rng, markers, presence=0.6,
                                        n_sites=0)
        none = find_fiducials(frame, x, y, z, n_frames=2000)
        some = find_fiducials(frame, x, y, z, n_frames=2000,
                              min_frame_fraction=0.5)
        assert none == [] and len(some) == 1
        return (f"a spot in 60% of frames is not a marker at "
                f"{FIDUCIAL_MIN_FRAME_FRACTION:.0%}, is at 50%")

    def one_marker_on_cell_corners():
        # Centred on the corner of four grid cells, so each cell holds a
        # quarter of its localizations.
        rng = np.random.default_rng(4)
        markers = np.array([[1500.0, 3000.0, 0.0]])
        frame, x, y, z = simulate_round(rng, markers, n_sites=0,
                                        marker_precision=(20.0, 5.0))
        found = find_fiducials(frame, x, y, z, n_frames=2000)
        assert len(found) == 1, len(found)
        assert math.hypot(found[0].x_nm - 1500, found[0].y_nm - 3000) < 3
        return "found once, not four times"

    def frames_not_localizations():
        # Ten localizations in each of 30% of frames: plenty of
        # localizations, too few frames.
        frame = np.repeat(np.arange(0, 2000, 3), 10)
        rng = np.random.default_rng(5)
        x = 7000 + rng.normal(0, 3, frame.size)
        y = 7000 + rng.normal(0, 3, frame.size)
        assert find_fiducials(frame, x, y, None, n_frames=2000) == []
        return f"{frame.size:,} localizations in 667 frames: not a marker"

    def real_sample_has_no_markers():
        if not os.path.exists(REAL_SAMPLE):
            return "skipped, sample not present"
        from tools.mps_io import load_localizations
        from tools.mps_registration import fiducials_in

        loc = load_localizations(REAL_SAMPLE)
        assert fiducials_in(loc) == []
        assert fiducials_in(loc, radius_nm=1000.0) == []
        return f"{loc.n:,} DNA-PAINT localizations, no marker at 150 or 1000 nm"

    def large_file_is_fast():
        rng = np.random.default_rng(6)
        markers = np.column_stack([rng.uniform(0, 80000, 20),
                                   rng.uniform(0, 80000, 20),
                                   np.zeros(20)])
        started = time.perf_counter()
        frame, x, y, z = simulate_round(rng, markers, n_frames=20000,
                                        n_sites=250000, field_nm=80000.0)
        built = time.perf_counter() - started
        started = time.perf_counter()
        found = find_fiducials(frame, x, y, z, n_frames=20000)
        seconds = time.perf_counter() - started
        assert len(found) == 20, len(found)
        assert seconds < 30, seconds
        return (f"{frame.size:,} localizations: {seconds:.1f} s "
                f"(simulation {built:.0f} s)")

    check("markers are found, and only markers", finds_exactly_the_markers)
    check("sparse binding is never a marker", binding_sites_are_not_markers)
    check("the frame-fraction rule", persistent_but_not_enough)
    check("a marker straddling grid cells is found once",
          one_marker_on_cell_corners)
    check("frames are counted, not localizations", frames_not_localizations)
    check("the 15.07.26 DNA-PAINT sample holds no marker",
          real_sample_has_no_markers)
    check("a large field stays fast", large_file_is_fast)


# ================================================================ pairing
def _fid(x: float, y: float, z: Optional[float] = 0.0) -> Fiducial:
    return Fiducial(x, y, z, 1000, 1000, 1.0, 2.0, 5.0)


def test_pairing() -> None:
    print("\n2. PAIRING MARKERS BETWEEN ROUNDS")

    def large_offset_and_strays():
        rng = np.random.default_rng(10)
        pa = rng.uniform(0, 20000, (8, 2))
        a = [_fid(*p) for p in pa] + [_fid(30000, 30000)]
        offset = np.array([1200.0, -900.0])
        b = [_fid(*(p - offset + rng.normal(0, 5, 2))) for p in pa[::-1]]
        b.append(_fid(-5000, 2000))
        pairs = match_fiducials(a, b)
        assert len(pairs) == 8, pairs
        assert all(j == 7 - i for i, j in pairs), pairs
        return "8 pairs under a 1.5 um offset; strays on both sides ignored"

    def offset_beyond_the_limit():
        a = [_fid(0, 0), _fid(1000, 0)]
        b = [_fid(-4000, 0), _fid(-3000, 0)]
        assert match_fiducials(a, b) == []
        assert len(match_fiducials(a, b, max_offset_nm=5000)) == 2
        return "a 4 um offset needs a larger search"

    def ambiguous_is_refused():
        # Two markers on each side, and no offset lines up both.
        a = [_fid(0, 0), _fid(1000, 0)]
        b = [_fid(0, 0), _fid(0, 1700)]
        assert match_fiducials(a, b) == []
        return "no single offset explains two pairs: nothing paired"

    def one_each_is_paired():
        assert match_fiducials([_fid(0, 0)], [_fid(300, 200)]) == [(0, 0)]
        return None

    def one_against_several_is_refused():
        # Two particles came off between rounds: which one is left is
        # unknowable, and the nearest is not necessarily it.
        a = [_fid(10000, 10000), _fid(9000, 16000), _fid(10500, 14900)]
        b = [_fid(9000 + 1200, 16000 - 900)]
        assert match_fiducials(a, b) == []
        assert match_fiducials(b, a) == []
        reg = register_from_fiducials(a, b)
        assert reg.source == "none" and not reg.shifts_channel_b
        # Candidates in a row: their median is itself one of them, so only
        # the ambiguity test stops a guess here.
        row = [_fid(0, 0), _fid(1000, 0), _fid(2000, 0)]
        assert match_fiducials(row, [_fid(500, 0)]) == []
        return "3 against 1: no pair, no shift"

    def best_supported_offset_beats_the_smallest():
        # The true offset (2000, 0) lines up three markers; a smaller one
        # lines up a single pair.
        a = [_fid(0, 0), _fid(3000, 0), _fid(0, 3000), _fid(100, 150)]
        b = [_fid(-2000, 0), _fid(1000, 0), _fid(-2000, 3000)]
        pairs = match_fiducials(a, b)
        assert sorted(pairs) == [(0, 0), (1, 1), (2, 2)], pairs
        return "three pairs at 2 um win over one pair at 180 nm"

    def tied_offsets_are_refused():
        # Round A holds two identical pairs of markers, both within reach
        # of round B's pair: two different offsets fit equally well.
        a = [_fid(0, 0), _fid(1000, 0), _fid(2000, 1000), _fid(3000, 1000)]
        b = [_fid(0, -300), _fid(1000, -300)]
        assert len(match_fiducials(a[:2], b)) == 2
        assert match_fiducials(a, b) == []
        return "two equally good offsets: nothing paired"

    def partners_are_mutual():
        # Two round-A markers 60 nm apart both see the same round-B marker.
        a = [_fid(0, 0), _fid(60, 0), _fid(5000, 0), _fid(0, 5000)]
        b = [_fid(0, 0), _fid(5000, 0), _fid(0, 5000)]
        pairs = match_fiducials(a, b)
        partners = [j for _, j in pairs]
        assert len(partners) == len(set(partners)) == 3, pairs
        return "one round-B marker is never paired twice"

    def empty_sides():
        assert match_fiducials([], [_fid(0, 0)]) == []
        assert match_fiducials([_fid(0, 0)], []) == []
        return None

    def random_field(rng, n: int, size: float) -> List[Fiducial]:
        points: List[np.ndarray] = []
        while len(points) < n:
            p = rng.uniform(0, size, 2)
            if all(np.hypot(*(p - q)) > 300 for q in points):
                points.append(p)
        return [_fid(*p) for p in points]

    def unrelated_fields_are_not_registered():
        # Two fields that share no marker. The search over 3 um finds
        # offsets that line up two markers by chance; before the shared-
        # fraction rule a third of these trials came out registered.
        import tools.mps_registration as registration

        rng = np.random.default_rng(60)
        trials = [(random_field(rng, 30, 20000.0),
                   random_field(rng, 30, 20000.0)) for _ in range(60)]
        accepted = sum(bool(match_fiducials(a, b)) for a, b in trials)
        saved = registration.MATCH_MIN_SHARED_FRACTION
        registration.MATCH_MIN_SHARED_FRACTION = 0.0
        try:
            by_chance = sum(bool(match_fiducials(a, b)) for a, b in trials)
        finally:
            registration.MATCH_MIN_SHARED_FRACTION = saved
        assert accepted == 0, accepted
        assert by_chance >= 5, by_chance
        reg = register_from_fiducials(*trials[0])
        assert reg.source == "none" and not reg.shifts_channel_b
        assert "could not be paired" in reg.description, reg.description
        assert "80%" not in reg.warnings[0], reg.warnings
        return (f"0 of {len(trials)} pairs of 30-marker fields registered "
                f"({by_chance} without the rule)")

    def shared_markers_among_strays_still_pair():
        rng = np.random.default_rng(61)
        found = []
        for n_shared, n_strays in ((15, 10), (4, 3), (3, 3)):
            shared = random_field(rng, n_shared + 2 * n_strays, 20000.0)
            common, strays_a = shared[:n_shared], shared[n_shared:n_shared + n_strays]
            strays_b = shared[n_shared + n_strays:]
            offset = rng.uniform(-2000, 2000, 2)
            a = common + strays_a
            b = [_fid(f.x_nm - offset[0] + rng.normal(0, 20),
                      f.y_nm - offset[1] + rng.normal(0, 20))
                 for f in common] + strays_b
            pairs = match_fiducials(a, b)
            assert sorted(pairs) == [(i, i) for i in range(n_shared)], \
                (n_shared, n_strays, pairs)
            found.append(f"{n_shared} of {n_shared + n_strays}")
        return "paired: " + ", ".join(found)

    check("pairs under a large offset, strays ignored", large_offset_and_strays)
    check("the offset search limit", offset_beyond_the_limit)
    check("an ambiguous pairing is refused", ambiguous_is_refused)
    check("one marker on each side", one_each_is_paired)
    check("one marker against several is refused",
          one_against_several_is_refused)
    check("the best-supported offset, not the smallest",
          best_supported_offset_beats_the_smallest)
    check("two equally good offsets are refused", tied_offsets_are_refused)
    check("pairs are mutual", partners_are_mutual)
    check("no markers on one side", empty_sides)
    check("fields that share no marker are not registered",
          unrelated_fields_are_not_registered)
    check("shared markers among strays still pair",
          shared_markers_among_strays_still_pair)


# ================================================================ measuring
def test_measuring() -> None:
    print("\n3. MEASURING THE REGISTRATION")

    def shift_and_error_recovered():
        rows = []
        sigma_xy, sigma_z = 6.0, 12.0
        lateral, axial, shift_err = [], [], []
        for seed in range(20, 30):
            (fa, xa, ya, za), (fb, xb, yb, zb), markers, shift = \
                exchange_pair(seed, n_markers=15,
                              registration_sigma=(sigma_xy, sigma_z))
            reg = register_from_fiducials(
                find_fiducials(fa, xa, ya, za, n_frames=2000),
                find_fiducials(fb, xb, yb, zb, n_frames=2000))
            assert reg.source == "fiducials" and reg.n_pairs == 15, reg
            shift_err.append(np.abs(np.asarray(reg.shift_nm) - shift))
            lateral.append(reg.lateral_rms_nm)
            axial.append(reg.axial_rms_nm)
        shift_err = np.max(np.array(shift_err), axis=0)
        # Mean of 15 offsets with sigma 6 / 12 nm: SE 1.5 / 3.1 nm.
        assert shift_err[0] < 6 and shift_err[1] < 6 and shift_err[2] < 12, \
            shift_err
        want_lat = sigma_xy * math.sqrt(2)
        got_lat, got_ax = float(np.mean(lateral)), float(np.mean(axial))
        assert abs(got_lat / want_lat - 1) < 0.15, (got_lat, want_lat)
        assert abs(got_ax / sigma_z - 1) < 0.15, (got_ax, sigma_z)
        return (f"shift within {shift_err.max():.1f} nm; error "
                f"{got_lat:.1f} nm laterally (true {want_lat:.1f}), "
                f"{got_ax:.1f} nm in z (true {sigma_z:.1f}), 10 pairs of rounds")

    def leave_one_out_is_not_the_fit_residual():
        # With 4 markers the fit residual underestimates by a factor
        # (n-1)/n = 0.75; leave-one-out does not.
        ratios_fit, ratios_loo = [], []
        rng = np.random.default_rng(40)
        for _ in range(400):
            pa = rng.uniform(0, 10000, (4, 2))
            a = [_fid(*p) for p in pa]
            b = [_fid(*(p + rng.normal(0, 5, 2))) for p in pa]
            reg = register_from_fiducials(a, b)
            off = np.array([[fa.x_nm - fb.x_nm, fa.y_nm - fb.y_nm]
                            for fa, fb in zip(a, b)])
            fit = off - off.mean(axis=0)
            ratios_fit.append(np.mean(np.sum(fit ** 2, axis=1)))
            ratios_loo.append(reg.lateral_rms_nm ** 2)
        truth = 2 * 25.0
        fit_ratio = float(np.mean(ratios_fit)) / truth
        loo_ratio = float(np.mean(ratios_loo)) / truth
        assert abs(fit_ratio - 0.75) < 0.08, fit_ratio
        assert abs(loo_ratio - 4 / 3) < 0.15, loo_ratio
        return (f"mean squared error / truth: fit {fit_ratio:.2f}, "
                f"leave-one-out {loo_ratio:.2f} (conservative)")

    def two_pairs_shift_only():
        a = [_fid(0, 0, 0.0), _fid(5000, 0, 0.0)]
        b = [_fid(-100, 50, -20.0), _fid(4900, 50, -20.0)]
        reg = register_from_fiducials(a, b)
        assert reg.n_pairs == 2
        assert np.allclose(reg.shift_nm, (100, -50, 20))
        assert reg.lateral_rms_nm is None and reg.axial_rms_nm is None
        assert any("cannot be estimated" in w for w in reg.warnings)
        return "shift applied, error left unknown"

    def two_d_leaves_z_unknown():
        (fa, xa, ya, za), (fb, xb, yb, zb), _, shift = exchange_pair(
            50, three_d=False)
        reg = register_from_fiducials(
            find_fiducials(fa, xa, ya, None, n_frames=2000),
            find_fiducials(fb, xb, yb, None, n_frames=2000))
        assert reg.axial_rms_nm is None and reg.shift_nm[2] == 0.0
        assert reg.lateral_rms_nm is not None
        assert any("axial registration is unknown" in w for w in reg.warnings)
        return None

    def no_markers_means_no_registration():
        reg = register_from_fiducials([], [_fid(0, 0)])
        assert reg.source == "none" and not reg.shifts_channel_b
        assert "channel A" in reg.description, reg.description
        assert reg.warnings
        return reg.description

    def disagreeing_markers_are_flagged():
        rng = np.random.default_rng(60)
        pa = rng.uniform(0, 10000, (8, 2))
        a = [_fid(*p) for p in pa]
        b = [_fid(*(p + rng.normal(0, 40, 2))) for p in pa]
        reg = register_from_fiducials(a, b)
        assert any("disagree" in w for w in reg.warnings), reg.warnings
        return f"{reg.lateral_rms_nm:.0f} nm against a 2 nm marker spread"

    def applying_the_shift():
        reg = Registration(source="fiducials", shift_nm=(10.0, -5.0, 2.0))
        x, y, z = reg.apply(np.array([1.0]), np.array([1.0]),
                            np.array([1.0]))
        assert (x[0], y[0], z[0]) == (11.0, -4.0, 3.0)
        x, y, z = NO_REGISTRATION.apply(np.array([1.0]), np.array([1.0]),
                                        np.array([1.0]))
        assert (x[0], y[0], z[0]) == (1.0, 1.0, 1.0)
        return None

    def from_two_files():
        a, b, _, shift = exchange_pair(70)
        la = as_localizations(a, "A.hdf5")
        lb = as_localizations(b, "B.hdf5", pixel=122.0)
        # Coordinates are already in nm; only the reported pixel size
        # differs, which must be called out.
        reg = register_localizations(la, lb)
        assert reg.n_pairs == 12 and reg.paths == ["A.hdf5", "B.hdf5"]
        assert np.allclose(reg.shift_nm, shift, atol=12)
        assert any("different pixel sizes" in w for w in reg.warnings)
        return reg.description

    check("a known shift and error are recovered",
          shift_and_error_recovered)
    check("leave-one-out, not the fit residual",
          leave_one_out_is_not_the_fit_residual)
    check("two pairs: shift without an error", two_pairs_shift_only)
    check("2D markers: z left unknown", two_d_leaves_z_unknown)
    check("no markers: no registration", no_markers_means_no_registration)
    check("markers that disagree are flagged", disagreeing_markers_are_flagged)
    check("applying the shift to channel B", applying_the_shift)
    check("from two loaded files", from_two_files)


# ================================================================ calibration
def test_calibration() -> None:
    print("\n4. READING matriz-transformacion OUTPUT")
    tmp = new_tmp("reg_cal_")

    def write(name: str, text: str) -> str:
        folder = os.path.join(tmp, name.split("/")[0])
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    matrix_hdf5_flow = (
        "version: '0.2'\nmodel: affine\nunits: pixels\n"
        "matrix:\n- [1.0, 0.0, 512.0]\n- [0.0, 1.0, 0.5]\n"
        "camera_info:\n  model: Synthetic\n  pixel_size_nm: 130\n"
        "qc_summary:\n  rmse_px: 0.08\n  tre_loo_median_px: 0.1\n"
        "  tre_loo_p90_px: 0.2\n  inlier_fraction: 1.0\n  n_inliers: 15\n"
        "  n_pairs: 15\n  verdict: excellent\n"
    )

    def json_tiff_flow():
        path = write("tiff/calibration_matrix.json", json.dumps({
            "tool": "smlm-affine-calibrator", "version": "0.1.0",
            "metrics": {"rms_nm": 8.15, "n_pairs_used": 40,
                        "median_residual_nm": 7.8},
        }))
        reg = read_calibration(path)
        assert reg.source == "calibration" and reg.lateral_rms_nm == 8.15
        assert reg.axial_rms_nm is None and not reg.shifts_channel_b
        assert reg.n_pairs == 40
        assert any("underestimates" in w for w in reg.warnings)
        return reg.description

    def yaml_hdf5_flow():
        path = write("hdf5/matrix.yaml", matrix_hdf5_flow)
        reg = read_calibration(path)
        want = 0.1 * 130 * RMS_PER_MEDIAN_2D
        assert abs(reg.lateral_rms_nm - want) < 1e-9, reg.lateral_rms_nm
        assert not reg.warnings, reg.warnings
        return reg.description

    def nan_tre_falls_back():
        # The tool rates a missing leave-one-out error 'poor' by
        # construction; that rating is not passed on.
        path = write("nan/matrix.yaml", matrix_hdf5_flow.replace(
            "tre_loo_median_px: 0.1", "tre_loo_median_px: .nan").replace(
            "verdict: excellent", "verdict: poor"))
        reg = read_calibration(path)
        assert abs(reg.lateral_rms_nm - 0.08 * 130) < 1e-9
        assert any("none was computed" in w for w in reg.warnings), \
            reg.warnings
        assert not any("poor" in w for w in reg.warnings), reg.warnings
        assert "poor" not in reg.description
        few = write("nan_few/matrix.yaml", matrix_hdf5_flow.replace(
            "tre_loo_median_px: 0.1", "tre_loo_median_px: .nan").replace(
            "n_inliers: 15", "n_inliers: 3"))
        assert any("at least 4" in w for w in read_calibration(few).warnings)
        rated = write("rated/matrix.yaml", matrix_hdf5_flow.replace(
            "verdict: excellent", "verdict: marginal"))
        assert any("'marginal'" in w for w in read_calibration(rated).warnings)
        return "in-sample fallback; a rating kept only when it rests on the error"

    def qc_metrics_beside_matrix():
        write("qc/matrix.yaml", matrix_hdf5_flow)
        path = write("qc/qc_metrics.yaml", (
            "n_pairs: 15\nn_inliers: 14\nrmse_px: 0.07\n"
            "tre_loo:\n  median_px: 0.12\n  p90_px: 0.3\n  max_px: 0.4\n"
            "  n: 14\n"))
        reg = read_calibration(path)
        assert abs(reg.lateral_rms_nm - 0.12 * 130 * RMS_PER_MEDIAN_2D) < 1e-9
        assert reg.n_pairs == 14
        alone = write("qc_alone/qc_metrics.yaml",
                      "rmse_px: 0.07\ntre_loo:\n  median_px: 0.12\n")
        try:
            read_calibration(alone)
        except ValueError:
            pass
        else:
            raise AssertionError("pixels converted without a pixel size")
        assert read_calibration(alone, pixel_size_nm=100).lateral_rms_nm \
            == 0.12 * 100 * RMS_PER_MEDIAN_2D
        return "pixel size from the matrix beside it, or given"

    def tiff_matrix_points_to_its_json():
        write("tm/calibration_matrix.json", json.dumps({
            "tool": "smlm-affine-calibrator",
            "metrics": {"rms_nm": 9.0, "n_pairs_used": 30}}))
        path = write("tm/matrix.yaml", (
            "version: '0.2'\nmodel: affine\nunits: pixels\n"
            "matrix:\n- [1.0, 0.0, 3.2]\n- [0.0, 1.0, -1.5]\n"))
        assert read_calibration(path).lateral_rms_nm == 9.0
        lonely = write("tm_alone/matrix.yaml", (
            "version: '0.2'\nmodel: affine\nunits: pixels\n"
            "matrix:\n- [1.0, 0.0, 3.2]\n- [0.0, 1.0, -1.5]\n"))
        try:
            read_calibration(lonely)
        except ValueError as error:
            assert "no quality" in str(error), error
            return "reads the JSON beside it; refuses when alone"
        raise AssertionError("accepted a matrix with no metrics")

    def unrelated_files_refused():
        bad = [
            write("bad/other.json", json.dumps({"tool": "something"})),
            write("bad2/calibration_matrix.json", json.dumps({
                "tool": "smlm-affine-calibrator",
                "metrics": {"rms_nm": float("nan")}})),
            write("bad3/picasso.yaml", "Pixelsize: 130\nFrames: 10\n"),
        ]
        for path in bad:
            try:
                read_calibration(path)
            except ValueError:
                continue
            raise AssertionError(f"accepted {path}")
        return f"{len(bad)} refused"

    def rms_per_median_constant():
        rng = np.random.default_rng(80)
        r = np.hypot(*rng.normal(0, 1, (2, 400000)))
        ratio = float(np.sqrt(np.mean(r ** 2)) / np.median(r))
        assert abs(ratio - RMS_PER_MEDIAN_2D) < 0.01, ratio
        return f"simulated {ratio:.3f}, used {RMS_PER_MEDIAN_2D:.3f}"

    check("calibration_matrix.json (TIFF flow)", json_tiff_flow)
    check("matrix.yaml with its quality summary (HDF5 flow)", yaml_hdf5_flow)
    check("a missing leave-one-out error falls back, with warnings",
          nan_tre_falls_back)
    check("qc_metrics.yaml", qc_metrics_beside_matrix)
    check("a bare matrix.yaml", tiff_matrix_points_to_its_json)
    check("files that are not calibrations", unrelated_files_refused)
    check("median-to-RMS conversion", rms_per_median_constant)


# ================================================================ combining
def test_combining() -> None:
    print("\n5. COMBINING SOURCES")

    def quadrature():
        fid = Registration(source="fiducials", shift_nm=(1.0, 2.0, 3.0),
                           lateral_rms_nm=6.0, axial_rms_nm=10.0, n_pairs=9)
        cal = Registration(source="calibration", lateral_rms_nm=8.0)
        both = combine(fid, cal)
        assert both.source == "combined" and both.shift_nm == (1.0, 2.0, 3.0)
        assert both.lateral_rms_nm == 10.0 and both.axial_rms_nm == 10.0
        row = export_registration(both)
        assert row["registration_lateral_rms_nm"] == 10.0
        assert row["registration_shift_z_nm"] == 3.0
        return "6 and 8 nm give 10 nm"

    def unknown_stays_unknown():
        fid = Registration(source="fiducials", shift_nm=(1.0, 2.0, 0.0),
                           n_pairs=2)
        cal = Registration(source="calibration", lateral_rms_nm=8.0)
        both = combine(fid, cal)
        assert both.lateral_rms_nm is None
        assert any("left unknown" in w for w in both.warnings)
        return "the calibration alone is not passed off as the total"

    def no_markers_means_unregistered():
        fid = register_from_fiducials([], [])
        cal = Registration(source="calibration", lateral_rms_nm=8.0)
        both = combine(fid, cal)
        assert both.source == "none" and both.lateral_rms_nm is None
        assert any("Do not use markers" in w for w in both.warnings)
        return "the calibration is not passed off as the round offset"

    check("independent errors add in quadrature", quadrature)
    check("no markers: not registered, even with a calibration",
          no_markers_means_unregistered)
    check("an unknown marker error is not replaced", unknown_stays_unknown)


# ================================================================ end to end
def test_end_to_end() -> None:
    print("\n6. TWO CHANNELS END TO END  (synthetic Exchange-PAINT pair)")
    from tools.cluster_quality import CircularROI
    from tools.mps_crosschannel import export_cross_channel
    from tools.mps_io import load_localizations
    from tools.mps_synthetic_pair import write_pair
    from tools.mps_twochannel_window import (
        MARKERS_LOADED,
        MARKERS_NONE,
        MARKERS_OTHER,
        TwoChannelInputs,
        measure_registration,
        run_two_channels,
    )

    tmp = new_tmp("reg_e2e_")
    pair = write_pair(tmp, seed=3, phase=0.5)
    loc_a = load_localizations(pair.path_a)
    loc_b = load_localizations(pair.path_b)

    def inputs(radius: float) -> TwoChannelInputs:
        roi = CircularROI(pair.centre_nm[0], pair.centre_nm[1], radius)
        common = dict(eps_nm=25.0, min_samples=10, dbcv_threshold=-1.0,
                      roi=roi)
        return TwoChannelInputs(
            loc_a=loc_a, loc_b=loc_b, roi=roi, slab=None,
            slab_half_width_nm=90.0,
            kwargs_a=dict(common, pixel_size_nm=loc_a.pixel_size_nm,
                          pixel_size_source=loc_a.pixel_size_source,
                          source_name=pair.path_a),
            kwargs_b=dict(pixel_size_nm=loc_b.pixel_size_nm,
                          pixel_size_source=loc_b.pixel_size_source,
                          source_name=pair.path_b))

    results: dict = {}

    def registered():
        reg = measure_registration(inputs(700.0), MARKERS_LOADED, ("", ""), "")
        assert reg.source == "fiducials" and reg.n_pairs == 8, reg.description
        assert np.allclose(reg.shift_nm, pair.shift_nm, atol=8), reg.shift_nm
        out = run_two_channels(inputs(700.0), reg)
        results["registered"] = out
        assert not out.notes, out.notes
        assert out.axial is not None and out.transverse is not None
        assert abs(out.axial.phase_fraction - 0.5) < 0.07, \
            out.axial.phase_fraction
        assert out.axial.interpretation_hint == "near antiphase"
        assert out.axial.phase_uncertainty == \
            reg.axial_rms_nm / out.axial.period_used_nm
        assert out.axial.phase_uncertainty < 0.1
        assert reg.axial_rms_nm > 1.5 * reg.lateral_rms_nm, \
            (reg.axial_rms_nm, reg.lateral_rms_nm)
        t = out.transverse
        assert t.n_clusters_a >= 14 and t.n_clusters_b >= 14, \
            (t.n_clusters_a, t.n_clusters_b)
        assert t.registration_rms_nm == reg.lateral_rms_nm
        return (f"shift {tuple(round(v) for v in reg.shift_nm)} nm (true "
                f"{pair.shift_nm}); phase {out.axial.phase_fraction:.2f} "
                f"+/- {out.axial.phase_uncertainty:.2f}; clusters "
                f"{t.n_clusters_a}/{t.n_clusters_b}")

    def fold(fraction: float) -> float:
        f = abs(fraction) % 1.0
        return min(f, 1.0 - f)

    def unregistered_misleads_and_says_so():
        # A ROI wide enough to hold channel 2 where it was recorded.
        reg = measure_registration(inputs(2500.0), MARKERS_NONE, ("", ""), "")
        assert reg.source == "none"
        out = run_two_channels(inputs(2500.0), reg)
        assert out.axial is not None, out.notes
        # Round 2 was recorded at true - shift: the 30 nm focus difference
        # between rounds is inside the offset, with that sign.
        wrong = fold(0.5 - pair.shift_nm[2] / pair.period_nm)
        assert abs(out.axial.phase_fraction - wrong) < 0.07, \
            (out.axial.phase_fraction, wrong)
        assert any("axial registration of channel B is unknown" in w
                   for w in out.axial.warnings)
        assert any("lateral registration of channel B is unknown" in w
                   for w in out.transverse.warnings)
        return (f"phase {out.axial.phase_fraction:.2f} instead of 0.50, "
                f"with the warning that says why")

    def shift_before_selection():
        reg = measure_registration(inputs(700.0), MARKERS_NONE, ("", ""), "")
        out = run_two_channels(inputs(700.0), reg)
        assert out.n_b < 50 and out.axial is None, out.n_b
        assert any("Register them" in n for n in out.notes), out.notes
        # 16 clusters of 30 + 60 + 30 localizations: all of them.
        registered_n = results["registered"].n_b
        assert registered_n == 16 * 120, registered_n
        return (f"channel 2 in a 700 nm ROI: {out.n_b} localizations as "
                f"loaded, {registered_n:,} once registered")

    def quarter_phase_and_its_sign():
        quarter = write_pair(tmp, seed=4, phase=0.25, name="quarter")
        la, lb = load_localizations(quarter.path_a), \
            load_localizations(quarter.path_b)
        roi = CircularROI(quarter.centre_nm[0], quarter.centre_nm[1], 2500.0)
        base = TwoChannelInputs(
            loc_a=la, loc_b=lb, roi=roi, slab=None, slab_half_width_nm=90.0,
            kwargs_a=dict(eps_nm=25.0, min_samples=10, roi=roi,
                          pixel_size_nm=130.0, pixel_size_source="yaml"))
        raw = run_two_channels(base, NO_REGISTRATION)
        registered = run_two_channels(
            dataclasses.replace(base, roi=CircularROI(
                quarter.centre_nm[0], quarter.centre_nm[1], 700.0),
                kwargs_a=dict(base.kwargs_a, roi=None)),
            measure_registration(base, MARKERS_LOADED, ("", ""), ""))
        expected_raw = fold(0.25 - quarter.shift_nm[2] / quarter.period_nm)
        assert abs(raw.axial.phase_fraction - expected_raw) < 0.07, \
            (raw.axial.phase_fraction, expected_raw)
        assert abs(registered.axial.phase_fraction - 0.25) < 0.07, \
            registered.axial.phase_fraction
        return (f"true 0.25: {registered.axial.phase_fraction:.2f} "
                f"registered, {raw.axial.phase_fraction:.2f} unregistered "
                f"(expected {expected_raw:.2f})")

    def channel_b_parameters_are_its_own():
        own = dataclasses.replace(
            inputs(700.0),
            kwargs_b=dict(inputs(700.0).kwargs_b, eps_nm=31.0,
                          min_samples=7))
        reg = results["registered"].registration
        out = run_two_channels(own, reg)
        t = out.transverse
        assert t.analysis_a.eps_nm == 25.0 and t.analysis_a.min_samples == 10
        assert t.analysis_b.eps_nm == 31.0 and t.analysis_b.min_samples == 7
        return "eps 25 / 31, min samples 10 / 7"

    def pixel_sizes_are_checked():
        import shutil as _shutil

        other = os.path.join(tmp, "px122")
        os.makedirs(other, exist_ok=True)
        wrong = []
        for path in (pair.path_a, pair.path_b):
            target = os.path.join(other, os.path.basename(path))
            _shutil.copy(path, target)
            with open(os.path.splitext(path)[0] + ".yaml",
                      encoding="utf-8") as handle:
                text = handle.read().replace("Pixelsize: 130",
                                             "Pixelsize: 122")
            with open(os.path.splitext(target)[0] + ".yaml", "w",
                      encoding="utf-8") as handle:
                handle.write(text)
            wrong.append(target)
        try:
            measure_registration(inputs(700.0), MARKERS_OTHER,
                                 (wrong[0], wrong[1]), "")
        except ValueError as error:
            assert "cannot be applied" in str(error), error
        else:
            raise AssertionError("a shift in another pixel scale was used")

        bare = os.path.join(tmp, "bare")
        os.makedirs(bare, exist_ok=True)
        naked = []
        for path in (pair.path_a, pair.path_b):
            target = os.path.join(bare, os.path.basename(path))
            _shutil.copy(path, target)
            naked.append(target)
        reg = measure_registration(inputs(700.0), MARKERS_OTHER,
                                   (naked[0], naked[1]), "")
        assert reg.n_pairs == 8, reg.description
        assert sum("records no pixel size" in w for w in reg.warnings) == 2

        mixed = dataclasses.replace(
            inputs(700.0),
            loc_b=load_localizations(wrong[1]))
        out = run_two_channels(mixed, NO_REGISTRATION)
        assert any("different pixel sizes" in n for n in out.notes), out.notes
        return ("marker files in another scale refused; files without one "
                "take the channel's; the channels themselves are compared")

    def two_d_against_three_d():
        from tools.mps_io import Localizations

        flat_b = Localizations(
            x_nm=loc_b.x_nm, y_nm=loc_b.y_nm,
            z_nm=np.zeros_like(loc_b.z_nm), path="flat.hdf5",
            fileformat=loc_b.fileformat, pixel_size_nm=130.0,
            pixel_size_source="yaml", columns=dict(loc_b.columns),
            info=loc_b.info)
        reg = results["registered"].registration
        out = run_two_channels(
            dataclasses.replace(inputs(700.0), loc_b=flat_b), reg)
        assert out.axial is None and out.transverse is not None
        assert any("2D projections" in n for n in out.notes), out.notes
        lo, hi = out.transverse.slab_nm
        assert lo < out.z_a.min() and hi > out.z_a.max(), (lo, hi)
        results["projected"] = out
        return f"slab {lo:.0f}..{hi:.0f} nm keeps both channels whole"

    def calibration_only():
        path = os.path.join(tmp, "calibration_matrix.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"tool": "smlm-affine-calibrator",
                       "metrics": {"rms_nm": 7.0, "n_pairs_used": 25}},
                      handle)
        reg = measure_registration(inputs(700.0), MARKERS_NONE, ("", ""),
                                   path)
        assert reg.source == "calibration" and not reg.shifts_channel_b
        both = measure_registration(inputs(700.0), MARKERS_LOADED, ("", ""),
                                    path)
        assert both.source == "combined", both.description
        assert both.lateral_rms_nm > 7.0 and both.axial_rms_nm is not None
        return (f"alone: {reg.lateral_rms_nm:.1f} nm, no shift; with the "
                f"markers: {both.lateral_rms_nm:.1f} nm")

    def export_carries_the_registration():
        out = results["registered"]
        row = export_cross_channel(out.axial, out.transverse,
                                   pair.path_a, pair.path_b)
        for key in ("registration_source", "registration_shift_z_nm",
                    "registration_lateral_rms_nm",
                    "registration_axial_rms_nm", "axial_phase_uncertainty"):
            assert key in row, key
        assert row["registration_source"] == "fiducials"
        # A 2D pair leaves the axial columns empty instead of dropping
        # them, so both rows go into one table.
        flat = results["projected"]
        row_2d = export_cross_channel(flat.axial, flat.transverse,
                                      pair.path_a, pair.path_b)
        assert list(row_2d) == list(row), set(row) ^ set(row_2d)
        assert row_2d["axial_phase_fraction"] is None
        empty = export_cross_channel(None, None)
        assert list(empty) == list(row)
        return f"{len(row)} columns, the same for a 2D pair"

    def export_records_the_parameters():
        out = results["registered"]
        assert out.parameters_a == {"eps_nm": 25.0, "min_samples": 10}
        assert out.parameters_b == out.parameters_a
        b_only = dataclasses.replace(
            inputs(700.0), kwargs_b=dict(inputs(700.0).kwargs_b,
                                         eps_nm=31.0))
        from tools.mps_twochannel_window import clustering_parameters

        a, b = clustering_parameters(b_only)
        assert a["eps_nm"] == 25.0 and b == {"eps_nm": 31.0,
                                             "min_samples": 10}
        return None

    def export_name_is_not_an_input():
        from tools.mps_io import is_derived_output

        assert is_derived_output("x_picked_axon7_two_channels.csv")
        return None

    check("registered by markers: antiphase recovered", registered)
    check("unregistered: the focus offset enters the phase, and is flagged",
          unregistered_misleads_and_says_so)
    check("channel 2 is selected after it is moved", shift_before_selection)
    check("a quarter-period phase, with the right sign",
          quarter_phase_and_its_sign)
    check("channel 2 uses its own clustering parameters",
          channel_b_parameters_are_its_own)
    check("pixel sizes are checked", pixel_sizes_are_checked)
    check("one 2D channel: compared as projections", two_d_against_three_d)
    check("with a calibration file", calibration_only)
    check("the export carries the registration",
          export_carries_the_registration)
    check("the clustering parameters are recorded",
          export_records_the_parameters)
    check("the export is not read back as an axon", export_name_is_not_an_input)


def main() -> int:
    print("=" * 72)
    print("CHANNEL REGISTRATION CHECKS")
    print("=" * 72)
    test_markers()
    test_pairing()
    test_measuring()
    test_calibration()
    test_combining()
    test_end_to_end()
    for path in _TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
