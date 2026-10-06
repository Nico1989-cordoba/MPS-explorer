# -*- coding: utf-8 -*-
"""H5-E S3 (D-40, spec 2.2-2.3): unit tests of the z-selection plumbing, the
selected-pair statistic, the leak family and the closure-v2 binding solver.
(The checks of the calibration harness itself run with the private research validators.)

Run: py -3 -m pytest test_h5e_selection.py -q
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import power_columns as pc
from tools import mps_simulate_axon as sa
from tools.mps_unroll import arc_joint_null


def test_key_text_round_trip() -> None:
    for key in [(3, ((0, 1), (1, 2))), (3, ()), (2, ((0, 1),))]:
        assert pc.zq_key_parse(pc.zq_key_text(key)) == key
    assert pc.zq_key_text((3, ((0, 1), (1, 2)))) == "3|0-1,1-2"


def test_rows_v5_extend_v4_and_leave_v1_v4_alone() -> None:
    v4 = pc.columns_of(pc.POWER_ROWS_VERSION_V4)
    v5 = pc.columns_of(pc.POWER_ROWS_VERSION_V5)
    tail = len(pc.SECONDS_COLUMNS) + 2          # the timings, n_warnings and table_version close every header
    assert v5[:len(v4) - tail] == v4[:-tail] and v5[-tail:] == v4[-tail:]
    assert set(pc.V5_EXTRA_COLUMNS) <= set(v5)
    assert len(v5) == len(v4) + len(pc.V5_EXTRA_COLUMNS)
    assert pc.version_of_header(v5) == pc.POWER_ROWS_VERSION_V5
    assert pc.version_of_header(v4) == pc.POWER_ROWS_VERSION_V4
    assert pc._simnull_table_version("v5") == pc.POWER_ROWS_VERSION_V5


def _joint(rings, seed=11, n=400):
    rng = np.random.default_rng(5)
    arcs = [np.sort(rng.uniform(0.0, 3000.0, 40)) for _ in rings]
    return arc_joint_null(rings, arcs, 3000.0, 60.0, reference_ring=1, n_null=n, random_seed=seed)


def test_z_over_pairs_all_pairs_is_joint_z() -> None:
    j3 = _joint([0, 1, 2])
    arc = cast(Any, SimpleNamespace(joint=j3))
    assert pc.z_over_pairs(arc, None) == pytest.approx(float(j3.z_A), abs=0, rel=0)
    assert pc.z_over_pairs(arc, [(0, 1), (1, 2)]) == float(j3.z_A)
    assert math.isnan(pc.z_over_pairs(arc, [(5, 6)]))
    assert math.isnan(pc.z_over_pairs(None, None))


def test_z_over_one_pair_equals_two_ring_joint() -> None:
    # the per-ring shift streams are spawned by ring position, so rings (0, 1) draw the same shifts in both calls
    rng = np.random.default_rng(5)
    arcs = [np.sort(rng.uniform(0.0, 3000.0, 40)) for _ in range(3)]
    j3 = arc_joint_null([0, 1, 2], arcs, 3000.0, 60.0, reference_ring=1, n_null=300, random_seed=7)
    j2 = arc_joint_null([0, 1], arcs[:2], 3000.0, 60.0, reference_ring=1, n_null=300, random_seed=7)
    assert pc.z_over_pairs(cast(Any, SimpleNamespace(joint=j3)), [(0, 1)]) == pytest.approx(float(j2.z_A), rel=1e-12)


def test_accepted_rows_first_by_index() -> None:
    rows = [{"replicate": str(i), "zq_accepted": ("1" if i % 3 else "0")} for i in (7, 2, 5, 0, 1, 4, 3)]
    got = [int(r["replicate"]) for r in pc.accepted_rows(rows, 3)]
    assert got == [1, 2, 4]


def test_scale_axial_leak() -> None:
    cfg = sa.SimConfig(axial_scale_by_bin=np.array([1.2, 1.4]), sigma_struct_nm=30.0, lpz_bin_edges_lab_nm=np.array([0.0, 100.0, 200.0]),
                       lpz_samples_by_bin_nm=[np.array([40.0]), np.array([45.0])])
    same = sa.scale_axial_leak(cfg, 1.0)
    for f in dataclasses.fields(cfg):
        if f.name == "provenance":
            continue
        a, b = getattr(cfg, f.name), getattr(same, f.name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        elif not isinstance(a, (list, tuple)) or a != b:
            assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)), f.name
    half = sa.scale_axial_leak(cfg, 0.5)
    assert np.allclose(half.axial_scale_by_bin, [0.6, 0.7]) and half.sigma_struct_nm == 15.0
    assert half.provenance["leak_scale"] == 0.5 and "leak_scale" not in cfg.provenance
    with pytest.raises(ValueError):
        sa.scale_axial_leak(cfg, 0.0)


def test_binding_solver_recovers_power_law() -> None:
    thetas = [10.0, 8.0, 8.8, 8.5]
    pairs = [(t, 3.0 * t ** 0.7) for t in thetas]
    theta, info = sa._binding_solve(pairs, 3.0 * 9.3 ** 0.7)
    assert theta == pytest.approx(9.3, rel=1e-9) and info["slope"] == pytest.approx(0.7)
    assert sa._binding_solve(pairs[:2], 5.0)[0] is None                       # too few iterates
    assert sa._binding_solve([(t, 3.0 * t ** 5) for t in thetas], 50.0)[0] is None   # slope out of range
    assert sa._binding_solve([(9.0, 3.0), (9.0, 3.1), (9.0, 2.9)], 3.0)[0] is None   # parameter never moved


def test_closure_argument_checked() -> None:
    with pytest.raises(ValueError):
        sa._calibrate_by_simulation(sa.SimConfig(), None, 10, None, 0, 1, [], closure="v3")


def test_v5_k_mean_is_the_expected_k_the_simulator_draws() -> None:
    # review 2026-10-01: ring k draws Poisson(lambda x factor_k x L / 1000) clusters, so the K per ring of a row is
    # lambda x mean(ring_rate_factors) x L / 1000; a truth configuration with factors of mean
    # above 1 is read by lambda x L alone as a K too low
    from tools.mps_matching import smooth_closed_path
    t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
    circle = np.column_stack([400.0 * np.cos(t), 400.0 * np.sin(t)])
    length = float(smooth_closed_path(circle).length_nm)
    plain = sa.SimConfig(contour_nm=circle, clusters_per_um=4.0)
    assert pc.v5_constant_values(plain, closure="v1")["m_k_mean"] == pytest.approx(4.0 * length / 1000.0, rel=1e-12)
    rated = dataclasses.replace(plain, ring_rate_factors=(1.0, 1.5, 1.5))
    assert pc.v5_constant_values(rated, closure="v1")["m_k_mean"] == pytest.approx(4.0 * (4.0 / 3.0) * length / 1000.0, rel=1e-12)
