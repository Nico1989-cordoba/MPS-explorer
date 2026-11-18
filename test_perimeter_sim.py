"""Tests of the perimeter density simulator (P1): the truth every validation of the analysis relies on."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.mps_perimeter_sim import (  # noqa: E402
    SCENARIOS, SHAPES, SOURCE_BACKGROUND, SOURCE_INTERIOR, SOURCE_MEMBRANE, _inside_polygon, expected_slice_r,
    leak_matrix, scenario_kwargs, simulate_perimeter, simulate_scenario, true_slice_of)
from tools.mps_perimeter_types import OUTSIDE_LABEL  # noqa: E402


def sim(shape="circle", intensity=None, **kw):
    base = dict(n_locs=5000, radial_sd_nm=20.0, slices={}, lpz_nm=47.0, background_frac=0.0, interior_frac=0.0,
                blink=False, seed=1)
    base.update(kw)
    return simulate_perimeter(shape, intensity or {"kind": "flat"}, **base)


def chi2_z(obs, exp):
    """Wilson-Hilferty normal deviate of a Pearson chi2 with len - 1 dof (no scipy needed)."""
    keep = exp >= 5.0
    x2 = float(np.sum((obs[keep] - exp[keep]) ** 2 / exp[keep]))
    k = int(keep.sum()) - 1
    return ((x2 / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))


def expected_per_bin(s, n_bins, n):
    """Expected counts per arc bin from the true lambda on the fine grid (cell-exact integration)."""
    L = s.true_path.length_nm
    ds = L / len(s.true_lambda)
    cell_edges = np.arange(len(s.true_lambda) + 1) * ds
    cum = np.concatenate([[0.0], np.cumsum(s.true_lambda * ds)])
    edges = np.linspace(0.0, L, n_bins + 1)
    mass = np.diff(np.interp(edges, cell_edges, cum))
    return n * mass / mass.sum(), edges


# ---------------------------------------------------------------------------------------------------- basics
def test_determinism_and_exact_count():
    a = sim("wavy", {"kind": "clumps"}, blink=True, background_frac=0.1, interior_frac=0.1, seed=7)
    b = sim("wavy", {"kind": "clumps"}, blink=True, background_frac=0.1, interior_frac=0.1, seed=7)
    c = sim("wavy", {"kind": "clumps"}, blink=True, background_frac=0.1, interior_frac=0.1, seed=8)
    for f in ("x_nm", "y_nm", "z_nm", "sx", "true_arc_nm", "molecule_id"):
        assert np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)
    assert not np.array_equal(a.x_nm, c.x_nm)
    assert len(a.x_nm) == 5000
    n = len(a.x_nm)
    for f in ("y_nm", "z_nm", "true_slice", "true_x_nm", "true_z_nm", "true_arc_nm", "source", "molecule_id",
              "sx", "sy"):
        assert len(getattr(a, f)) == n, f
    assert a.truth["n_membrane"] + a.truth["n_interior"] + a.truth["n_background"] == n


def test_unknown_keys_and_values_raise():
    with pytest.raises(ValueError):
        sim("square")
    with pytest.raises(ValueError):
        sim(intensity={"kind": "flat", "amplitude": 1.0})
    with pytest.raises(ValueError):
        sim(slices={"thickness": 170.0})
    with pytest.raises(ValueError):
        sim(options={"radius": 10.0})
    with pytest.raises(ValueError):
        sim(intensity={"kind": "attractor"}, slices={"pattern": "correlated", "rho": 0.5})
    with pytest.raises(ValueError):
        sim(background_frac=0.6, interior_frac=0.5)


@pytest.mark.parametrize("shape", SHAPES)
def test_shapes_are_closed_ccw_curves_with_turning_number_one(shape):
    s = sim(shape, n_locs=10)
    p = s.true_path
    L = p.length_nm
    assert np.allclose(p.edges_nm, L / len(p.points_nm), rtol=1e-3)          # uniform arc table
    # 5e-3: the "C" has curvature jumps between grid nodes, a rectangle-rule error of ~ step * jump
    assert abs(float(np.sum(s.true_curvature)) * L / len(s.true_curvature) - 2.0 * np.pi) < 5e-3
    x, y = p.points_nm[:, 0], p.points_nm[:, 1]
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0.0
    assert s.truth["star_shaped_from_centroid"] == (shape != "c_shape")


def test_circle_geometry_is_exact():
    s = sim("circle", n_locs=10, options={"radius_nm": 1500.0})
    assert abs(s.true_path.length_nm - 2.0 * np.pi * 1500.0) / (2.0 * np.pi * 1500.0) < 1e-5
    assert np.allclose(s.true_curvature, 1.0 / 1500.0, rtol=1e-6)


def test_wavy_and_two_lobes_curvature():
    w = sim("wavy", n_locs=10)
    assert w.truth["min_radius_of_curvature_nm"] >= 600.0       # resolved by the default contour
    assert np.any(w.true_curvature < 0.0)                        # it has concavities
    t = sim("two_lobes", n_locs=10)
    assert t.true_curvature.min() < 0.0                          # concave waist
    c = sim("c_shape", n_locs=10)
    assert np.isclose(c.true_curvature.min(), -1.0 / (1500.0 * 0.45))


# ---------------------------------------------------------------------------------------------------- lambda(s)
@pytest.mark.parametrize("shape,intensity", [
    ("ellipse", {"kind": "flat"}),
    ("ellipse", {"kind": "attractor", "cluster_arc_frac": 0.3}),
    ("ellipse", {"kind": "inhibition", "cluster_arc_frac": 0.3}),
    ("ellipse", {"kind": "regular", "cluster_arc_frac": 0.3}),
    ("wavy", {"kind": "curvature", "beta": 0.5}),
    ("c_shape", {"kind": "gap"}),
    ("circle", {"kind": "random"}),
])
def test_arc_histogram_reproduces_lambda(shape, intensity):
    s = sim(shape, intensity, n_locs=100_000, blink=False, seed=3)
    mem = s.source == SOURCE_MEMBRANE
    exp, edges = expected_per_bin(s, 100, int(mem.sum()))
    obs = np.histogram(s.true_arc_nm[mem], bins=edges)[0].astype(float)
    assert chi2_z(obs, exp) < 3.09                               # chi2 does not reject at p = 0.001
    assert np.all(obs[exp == 0.0] == 0.0)


def test_arc_histogram_of_molecules_with_blinking():
    s = sim("ellipse", {"kind": "attractor", "cluster_arc_frac": 0.6}, n_locs=150_000, blink=True, seed=4)
    mem = s.source == SOURCE_MEMBRANE
    _, first = np.unique(s.molecule_id[mem], return_index=True)
    arcs = s.true_arc_nm[mem][first]
    exp, edges = expected_per_bin(s, 100, len(arcs))
    assert chi2_z(np.histogram(arcs, bins=edges)[0].astype(float), exp) < 3.09
    n_per = np.bincount(s.molecule_id[mem])
    assert abs(n_per.mean() - 3.0) < 0.05                        # 1 + Geometric(mean 2)
    dx = s.x_nm[mem] - s.true_x_nm[mem]
    assert abs(dx.std() - 8.0) < 0.2                            # localizations at 8 nm of their molecule


def test_curvature_lambda_is_the_formula():
    s = sim("wavy", {"kind": "curvature", "beta": 0.5}, n_locs=10)
    R = s.true_path.length_nm / (2.0 * np.pi)
    f = np.clip(1.0 + 0.5 * s.true_curvature * R, 0.0, None)
    assert np.allclose(s.true_lambda, f / f.mean())
    flat = sim("circle", {"kind": "curvature", "beta": 0.5}, n_locs=10)
    assert np.allclose(flat.true_lambda, 1.0)                   # a circle has constant curvature


def test_inhibition_regular_gap_truths():
    s = sim("ellipse", {"kind": "inhibition", "cluster_arc_frac": 0.25, "gap_nm": 1000.0}, n_locs=10)
    L = s.truth["length_nm"]
    sc = s.truth["cluster_arc_nm"][0]
    assert np.isclose(sc, 0.25 * L)
    d = np.abs((s.arc_grid_nm + 0.5 * L / len(s.arc_grid_nm) - sc + 0.5 * L) % L - 0.5 * L)
    zone = (d > 300.0) & (d <= 800.0)                          # beyond the big cluster's Gaussian, inside the zone
    assert s.true_lambda[zone].max() < 1e-6
    g0, g1 = s.truth["gap_arcs_nm"][0][0]
    assert np.isclose(g1 - g0, 1000.0)
    r = sim("ellipse", {"kind": "regular", "period_nm": 800.0}, n_locs=10)
    P = r.truth["period_eff_nm"]
    assert abs(P - 800.0) < 0.5 * 800.0 / round(r.truth["length_nm"] / 800.0) + 1e-9
    assert np.isclose(r.truth["length_nm"] / P, round(r.truth["length_nm"] / P))


def test_radial_scatter_and_distance_to_true_curve():
    s = sim("circle", n_locs=40_000, radial_sd_nm=20.0, options={"radius_nm": 1500.0})
    mem = s.source == SOURCE_MEMBRANE
    assert abs(np.nanstd(s.true_radial_nm[mem]) - 20.0) < 0.5
    r = np.hypot(s.x_nm[mem], s.y_nm[mem]) - 1500.0
    assert abs(r.std() - math.hypot(20.0, 8.0)) < 0.6


def test_interior_and_background():
    s = sim("c_shape", n_locs=20_000, interior_frac=0.2, background_frac=0.1)
    assert int(np.sum(s.source == SOURCE_INTERIOR)) == 4000
    assert int(np.sum(s.source == SOURCE_BACKGROUND)) == 2000
    inn = s.source == SOURCE_INTERIOR
    assert np.all(_inside_polygon(s.true_x_nm[inn], s.true_y_nm[inn], s.true_path.points_nm))
    assert np.all(np.isnan(s.true_arc_nm[~(s.source == SOURCE_MEMBRANE)]))
    bg = s.source == SOURCE_BACKGROUND
    lo, hi = s.true_path.points_nm.min(axis=0) - 500.0, s.true_path.points_nm.max(axis=0) + 500.0
    assert np.all((s.x_nm[bg] >= lo[0]) & (s.x_nm[bg] <= hi[0]) & (s.y_nm[bg] >= lo[1]) & (s.y_nm[bg] <= hi[1]))


# ---------------------------------------------------------------------------------------------------- slices
def test_true_slice_convention():
    z = np.array([-255.0, -85.0 - 1e-9, -85.0, 84.999, 85.0, 254.999, 255.0])
    assert true_slice_of(z, 0.0, 170.0).tolist() == [-1, -1, 0, 0, 1, 1, OUTSIDE_LABEL]


def test_slice_fractions_match_request_and_leak_prediction():
    fr = (0.2, 0.5, 0.3)
    s = sim(slices={"fracs": fr, "z0_nm": -50.0}, n_locs=60_000, lpz_nm=47.0, seed=5)
    n = len(s.true_slice)
    for k, f in zip((-1, 0, 1), fr):
        got = float(np.mean(s.true_slice == k))
        assert abs(got - f) < 4.0 * math.sqrt(f * (1 - f) / n)
    obs = true_slice_of(s.z_nm, -50.0, 170.0)
    pred = np.asarray(fr) @ s.truth["leak_matrix"]
    for j, k in enumerate((-1, 0, 1)):
        got = float(np.mean(obs == k))
        assert abs(got - pred[j]) < 4.0 * math.sqrt(pred[j] * (1 - pred[j]) / n)


def test_leak_matrix_matches_closed_form():
    T, sig = 170.0, 47.0
    M, w = leak_matrix(thickness_nm=T, lpz_nm=sig)
    # uniform depth in a slice of width T: fraction beyond one edge = (sig / T) (phi(0) - G(T / sig)),
    # G(u) = u Phi(-u) ... written with the standard identity int_0^T Phi(-z / sig) dz / T
    u = T / sig
    G = lambda v: v * 0.5 * (1 + math.erf(v / math.sqrt(2))) + math.exp(-0.5 * v * v) / math.sqrt(2 * math.pi)
    beyond = (sig / T) * (G(0.0) - G(-u))
    beyond_two = (sig / T) * (G(-u) - G(-2 * u))                 # past the next slice too
    assert abs(M[1, 0] - (beyond - beyond_two)) < 1e-4
    assert abs(M[1, 0] - 0.110) < 0.002                          # the plan's ~11 %
    assert np.allclose(w, 1 / 3)
    r = expected_slice_r(M, w)
    assert abs(r[0, 1] - 0.27) < 0.01 and abs(r[1, 2] - 0.27) < 0.01      # the plan's derivation, r ~ 0.27
    assert r[0, 2] < 0.03
    M0, _ = leak_matrix(thickness_nm=T, lpz_nm=0.0)
    assert np.allclose(M0, np.eye(3))
    Mg, _ = leak_matrix(thickness_nm=T, lpz_nm=sig, guard_nm=42.5)
    assert Mg[1, 0] < M[1, 0] / 3


def test_gaussian_depth_peak_focus_and_widths():
    s = sim(slices={"depth": "gaussian", "peak_nm": 20.0, "focus_nm": -43.0, "depth_sd_nm": 130.0},
            n_locs=60_000, seed=6)
    assert s.truth["z0_nm"] == 20.0
    assert s.truth["peak_minus_focus_nm"] == 63.0
    assert abs(float(np.mean(s.z_nm)) - 20.0) < 3.0 * math.hypot(130.0, 47.0) / math.sqrt(60_000)
    n = len(s.true_slice)
    for j, k in enumerate((-1, 0, 1)):
        f = s.truth["true_slice_frac"][j]
        assert abs(float(np.mean(s.true_slice == k)) - f) < 4.0 * math.sqrt(f * (1 - f) / n)
    assert np.any(s.true_slice == OUTSIDE_LABEL)
    pct = s.truth["true_slice_frac"][[0, 2]] / s.truth["true_slice_frac"][1]
    assert np.all((pct > 0.3) & (pct < 0.6))                     # axial decay like the pilot (36-52 %)
    # sx = sy crossing at the focus, read the way an analysis would: median(sx - sy) per 40 nm bin of observed z
    edges = np.arange(-300.0, 301.0, 40.0)
    centres = 0.5 * (edges[1:] + edges[:-1])
    med = np.array([np.median((s.sx - s.sy)[(s.z_nm >= a) & (s.z_nm < b)]) for a, b in zip(edges[:-1], edges[1:])])
    i = int(np.where(np.diff(np.sign(med)) != 0)[0][0])
    cross = centres[i] - med[i] * (centres[i + 1] - centres[i]) / (med[i + 1] - med[i])
    assert abs(cross - (-43.0)) < 5.0


def test_identical_and_correlated_patterns():
    s = sim(intensity={"kind": "random"}, slices={"pattern": "identical"}, n_locs=10)
    assert np.allclose(s.truth["true_slice_lambda_r"], 1.0)
    assert np.allclose(s.truth["expected_slice_r"], 1.0)
    rs = []
    for seed in range(5):
        c = sim(intensity={"kind": "random"}, slices={"pattern": "correlated", "rho": 0.5}, n_locs=10, seed=seed,
                options={"radius_nm": 5000.0})
        rs.append(c.truth["true_slice_lambda_r"][0, 2])
    assert abs(float(np.mean(rs)) - 0.5) < 0.08


def test_measured_leak_matches_expected_r():
    """Independent slice patterns: the neighbour r produced by lpz alone matches ``expected_slice_r`` (+-0.03)."""
    T = 170.0
    rs = {(0, 1): [], (1, 2): [], (0, 2): []}
    exp = None
    for seed in range(6):
        s = sim(intensity={"kind": "random", "field_amp": 0.3}, slices={"pattern": "independent"},
                n_locs=300_000, lpz_nm=47.0, blink=False, seed=100 + seed, options={"radius_nm": 5000.0})
        L = s.true_path.length_nm
        edges = np.linspace(0.0, L, int(round(L / 100.0)) + 1)
        obs = true_slice_of(s.z_nm, 0.0, T)
        prof = [np.histogram(s.true_arc_nm[obs == k], bins=edges)[0].astype(float) for k in (-1, 0, 1)]
        for a, b in rs:
            rs[(a, b)].append(float(np.corrcoef(prof[a], prof[b])[0, 1]))
        exp = s.truth["expected_slice_r"]
    for (a, b), v in rs.items():
        assert abs(float(np.mean(v)) - exp[a, b]) < 0.03, ((a, b), float(np.mean(v)), exp[a, b])
    assert exp[0, 1] > 0.15 and exp[0, 2] < 0.05                 # neighbours leak, the outer pair barely


def test_expected_slice_r_limits():
    M = np.eye(3)
    w = np.full(3, 1 / 3)
    assert np.allclose(expected_slice_r(M, w, 0.0), np.eye(3))
    assert np.allclose(expected_slice_r(M, w, 1.0), 1.0)


# ---------------------------------------------------------------------------------------------------- presets
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_runs_and_emits_its_truth(name):
    s = simulate_scenario(name, n_locs=3000)
    assert len(s.x_nm) == 3000
    t = s.truth
    for key in ("length_nm", "kind", "cluster_arc_nm", "gap_arcs_nm", "z0_nm", "focus_nm", "leak_matrix",
                "expected_slice_r", "true_slice_frac", "source_frac"):
        assert key in t
    assert set(s.true_lambda_by_slice) == {-1, 0, 1}
    assert scenario_kwargs(name, seed=9)["seed"] == 9
    if name.startswith("a_") and name not in ("a_flat",):
        assert t["cluster_arc_nm"][0] is not None
    if name == "a_regular":
        assert t["period_eff_nm"] is not None
    if name == "contour_c_gap":
        assert not t["star_shaped_from_centroid"] and t["gap_nm"] == 2000.0
