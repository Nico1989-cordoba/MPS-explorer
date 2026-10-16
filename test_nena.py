# -*- coding: utf-8 -*-
"""
Tests of the NeNA estimator in ``tools.mps_quality`` (docs/nena.md).

Synthetic data only: whole-field files of known precision from
``docs/sims/nena/nena_sim.py`` and, for the regression, the generator of the
channel-2 tests (600 emitters, 4000 frames) on the seeds where Picasso's
least-squares fit came out low. The full validation grid is
``docs/sims/nena/nena_grid.py``; this is its fast subset.

Run: ``py -3 -m pytest test_nena.py -q``.
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "docs", "sims", "nena"))

from nena_sim import background_for, simulate  # noqa: E402
from tools import mps_quality as q  # noqa: E402

PIXEL_NM = 130.0


def _channel2_field(sigma_nm: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole-field generator of the channel-2 tests (test_channel2_params.write_whole_field), in memory, with the
    float32 pixel round trip of the file it writes."""
    rng = np.random.default_rng(seed)
    n_emitters, frames, field_nm = 600, 4000, 25000.0
    cx = rng.uniform(0, field_nm, n_emitters)
    cy = rng.uniform(0, field_nm, n_emitters)
    fr, xs, ys = [], [], []
    for i in range(n_emitters):
        for _ in range(rng.poisson(3) + 1):
            start = int(rng.integers(0, frames - 10))
            length = int(rng.geometric(0.4))
            for k in range(length):
                fr.append(start + k)
                xs.append(cx[i] + rng.normal(0, sigma_nm))
                ys.append(cy[i] + rng.normal(0, sigma_nm))
    order = np.argsort(fr, kind="stable")
    x = (np.asarray(xs)[order] / PIXEL_NM).astype(np.float32).astype(float) * PIXEL_NM
    y = (np.asarray(ys)[order] / PIXEL_NM).astype(np.float32).astype(float) * PIXEL_NM
    return np.asarray(fr)[order].astype(np.int64), x, y


# ------------------------------------------------------------------ regression
@pytest.mark.parametrize("seed", [2, 3, 7, 8, 11])
def test_previous_failures_now_within_5_percent(seed: int) -> None:
    """The five files where Picasso's least squares read 0.81-0.95 sigma (sigma 12 nm) now read within 5 %."""
    frame, x, y = _channel2_field(12.0, seed)
    r = q.nena(frame, x, y)
    assert r.converged and r.well_identified, r.warnings
    assert r.lsq_nm / 12.0 < 0.96, r.lsq_nm           # the old estimator's failure, kept for comparison
    assert abs(r.precision_nm / 12.0 - 1.0) < 0.05, r.precision_nm
    assert r.ci_low_nm <= r.precision_nm <= r.ci_high_nm


def test_seed_3_is_the_worst_previous_case() -> None:
    frame, x, y = _channel2_field(12.0, 3)
    r = q.nena(frame, x, y)
    assert r.lsq_nm / 12.0 == pytest.approx(0.805, abs=0.01)
    assert r.precision_nm / 12.0 == pytest.approx(1.0, abs=0.03)


# ------------------------------------------------------------------ grid subset
@pytest.mark.parametrize("sigma", [6.0, 20.0, 30.0])
def test_known_sigma_no_contamination(sigma: float) -> None:
    for seed in range(3):
        frame, x, y, _ = simulate(sigma, n_mol=330, seed=7000 + seed)
        r = q.nena(frame, x, y)
        assert r.converged and abs(r.precision_nm / sigma - 1.0) < 0.05, (sigma, seed, r.precision_nm, r.warnings)


def test_uniform_background_30_percent() -> None:
    n_mol = 330
    for seed in range(3):
        frame, x, y, _ = simulate(12.0, n_mol=n_mol, bg_per_frame=background_for(0.3, n_mol), seed=7100 + seed)
        r = q.nena(frame, x, y)
        assert r.converged and abs(r.precision_nm / 12.0 - 1.0) < 0.05, (seed, r.precision_nm)


def test_interval_coverage_small_set() -> None:
    """20 files, sigma 12 nm, ~1000 pairs: the 95 % interval covers the true sigma in at least 17 (P(<= 16 | 0.95)
    = 1.6 %)."""
    covered = 0
    for seed in range(20):
        frame, x, y, _ = simulate(12.0, n_mol=170, seed=7200 + seed)
        r = q.nena(frame, x, y)
        covered += int(r.ci_low_nm <= 12.0 <= r.ci_high_nm)
    assert covered >= 17, covered


# ------------------------------------------------------------------ warnings
def test_crowded_field_is_flagged_not_corrected() -> None:
    """Clusters of 10 molecules with 0.4 of the mates on in the next frame: the nearest-neighbour rule picks other
    molecules and NeNA reads low. The value is reported, with a warning; nothing replaces it."""
    frame, x, y, _ = simulate(12.0, n_mol=1700, cluster_size=10, frames=225, seed=7300)
    r = q.nena(frame, x, y)
    assert r.converged
    assert "crowded" in r.flags and any("reads LOW" in w for w in r.warnings)
    assert r.crowding_fraction > q.NENA_CROWDING_LIMIT


def test_swamped_peak_is_not_converged() -> None:
    """93 % of the pairs are random neighbours: no same-molecule peak; NeNA says so instead of giving a number the
    channel-2 parameters would use."""
    frame, x, y, _ = simulate(12.0, n_mol=600, bg_per_frame=150.0, seed=7400)
    r = q.nena(frame, x, y)
    assert not r.converged and "no_peak" in r.flags
    assert any("no same-molecule peak" in w for w in r.warnings)


def test_disagreement_is_reported_and_value_kept() -> None:
    """sigma 20 nm, background at half the pairs, ~2000 same-molecule pairs (a file of the validation grid, cell
    "uniform 50 %", seed 73003): Picasso's model and the random-neighbour model disagree beyond the interval. The
    warning names both values, the fit is not well identified, and precision_nm stays the Picasso-model value."""
    n_mol = 333
    frame, x, y, _ = simulate(20.0, n_mol=n_mol, bg_per_frame=background_for(0.5, n_mol), seed=73003)
    r = q.nena(frame, x, y)
    assert r.converged and "disagreement" in r.flags, r.flags
    assert not r.well_identified
    assert not (r.ci_low_nm <= r.random_term_nm <= r.ci_high_nm)
    assert r.ci_low_nm <= r.precision_nm <= r.ci_high_nm
    assert any(f"{r.random_term_nm:.1f} nm" in w and f"{r.precision_nm:.1f} nm" in w for w in r.warnings)


def test_checks_do_not_depend_on_the_reported_precision() -> None:
    """Data quality passes the reported lpx/lpy as one more starting point; the value, the interval, the independent
    estimate and the flags must not change with it."""
    n_mol = 333
    frame, x, y, _ = simulate(20.0, n_mol=n_mol, bg_per_frame=background_for(0.5, n_mol), seed=73003)
    plain = q.nena(frame, x, y)
    seeded = q.nena(frame, x, y, reported_lp_nm=20.0)
    assert plain.flags == seeded.flags
    for name in ("precision_nm", "ci_low_nm", "ci_high_nm", "random_term_nm"):
        assert getattr(plain, name) == pytest.approx(getattr(seeded, name), rel=1e-3), name


def test_background_at_half_the_pairs_is_flagged() -> None:
    """sigma 12 nm, background at half the pairs: the same-molecule share sits at Picasso's limit (a >= ac) and the
    warning says so (grid cell "uniform 50 %", sigma 12, 2000 pairs, seed 69000)."""
    n_mol = 333
    frame, x, y, _ = simulate(12.0, n_mol=n_mol, bg_per_frame=background_for(0.5, n_mol), seed=69000)
    r = q.nena(frame, x, y)
    assert r.converged
    assert set(r.flags) & {"nearby_dominates", "disagreement"}, r.flags


def test_too_few_pairs() -> None:
    r = q.nena(np.array([0, 1, 2]), np.zeros(3), np.zeros(3))
    assert not r.converged and np.isnan(r.precision_nm) and r.flags == ["too_few_pairs"]


def test_linked_file_is_not_measured() -> None:
    frame, x, y, _ = simulate(12.0, n_mol=1700, linked=True, seed=7600)
    r = q.nena(frame, x, y)
    assert not r.converged


# ------------------------------------------------------------------ compatibility
def test_result_and_check_stay_backward_compatible() -> None:
    old_style = q.NeNAResult(12.0, 900, True)          # the three positional fields callers used
    assert old_style.precision_nm == 12.0 and old_style.flags == [] and np.isnan(old_style.ci_low_nm)
    frame, x, y, _ = simulate(12.0, n_mol=330, seed=7700)
    lp = np.full(frame.size, 12.0)
    check = q.precision_check(frame, x, y, lp, lp)
    assert check.nena is not None and check.nena_nm == check.nena.precision_nm
    assert check.nena_ci_low_nm == check.nena.ci_low_nm and check.verdict.startswith("reported precisions are")
    d = q._next_frame_distances_nm(frame, x, y, 120.0)
    assert d.size == check.n_pairs
    assert check.nena.best_fit.shape == check.nena.histogram.shape
    assert check.nena.best_fit.sum() == pytest.approx(check.nena.histogram.sum(), rel=1e-6)
