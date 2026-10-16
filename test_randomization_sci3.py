# -*- coding: utf-8 -*-
"""
SCI-3: the randomization control, D's own null, the Monte Carlo p-values, the CDF band, the 1NN regularity index,
the pooled-mode helper, the occupancy's sigma cap, the batch ICC in closed form and the constancy script.

Synthetic only (R8): rings of cluster centres drawn here, simulated axons (``tools.mps_simulate_axon``), Gaussian
groups. The heavy calibration (rejection rates over hundreds of replicates, the band sweep) is
``validate_randomization.py``; this file checks the mechanics in seconds:

  * D is the number ``scipy.stats.ks_2samp`` gives; each iteration's D_i, z_D and the Monte Carlo p-values follow
    their definitions; no KS p anywhere (result, export, dictionary).
  * the open-contour bug: a periodic spline fitted to the open contour leaves its last vertex out; closed, every
    vertex is within 5 nm of the curve at s = K (1 nm RMS).
  * every 1NN = 0 refuses the control; a jammed band refuses it when incomplete iterations are excluded and pools
    them (with a warning) when not; centres outside the band are counted.
  * the CDF band's crossing: none on a null difference, found (and significant) on a planted excess.
  * the regularity index r = median 1NN x K / P; the exported columns and their dictionary entries.
  * the ICC chance level in closed form against 20,000 Monte Carlo draws (within 0.01); no adjectives.
  * the pooled mode: Freedman-Diaconis width, the literal 50 bins, an axon-bootstrap interval that covers.
  * the sigma cap: editable and recorded; 1/3 changes nothing.
  * the constancy script refuses with fewer than three animals.

Run:  py -3 -m pytest test_randomization_sci3.py      or      venv\\Scripts\\python.exe test_randomization_sci3.py

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools import mps_randomization as R  # noqa: E402


def _ring(k: int, spacing: float = 250.0, d_min: float = 80.0, seed: int = 0, jitter: float = 0.0) -> np.ndarray:
    """K centres on a circle of perimeter K x spacing, placed by sequential inhibition along the arc."""
    rng = np.random.default_rng(seed)
    perim = k * spacing
    assert k * d_min < 0.7 * perim, "sequential inhibition on a circle jams near 0.75 coverage"
    radius = perim / (2 * np.pi)
    arcs: list = []
    while len(arcs) < k:
        a = rng.uniform(0, perim)
        if all(min(abs(a - b), perim - abs(a - b)) >= d_min for b in arcs):
            arcs.append(a)
    th = np.sort(np.array(arcs)) / radius
    r = radius + rng.normal(0, jitter, k)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def test_ks_distance_is_scipys() -> None:
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = np.round(rng.gamma(4, 50, rng.integers(5, 60)), int(rng.integers(0, 2)))
        b = np.round(rng.gamma(4, 55, rng.integers(50, 3000)), int(rng.integers(0, 2)))
        d = R.ks_distance(np.sort(a), np.sort(b))
        assert abs(d - ks_2samp(a, b).statistic) < 1e-12


def test_monte_carlo_p() -> None:
    null = np.arange(1.0, 100.0)            # 99 values
    assert R.monte_carlo_p(200.0, null) == 1 / 100
    assert R.monte_carlo_p(0.0, null) == 1.0
    assert abs(R.monte_carlo_p(50.0, null) - 51 / 100) < 1e-12
    two = R.monte_carlo_p(50.0, null, two_sided=True)
    assert two is not None and abs(two - min(1.0, 2 * 50 / 100)) < 1e-12
    assert R.monte_carlo_p(1000.0, null, two_sided=True) == 2 / 100


def test_closed_contour_keeps_the_last_vertex() -> None:
    # A ring whose LAST vertex sits 60 nm outside the others: the open fit overwrites it with the first and
    # passes far from it; the closed fit passes within 5 nm of every vertex at s = K (1 nm RMS).
    c = _ring(20, seed=1, jitter=15.0)
    c[-1] *= (np.linalg.norm(c[-1]) + 60.0) / np.linalg.norm(c[-1])
    from scipy.spatial import cKDTree
    open_fit = R.smooth_contour_bspline(c)
    closed_fit = R.smooth_contour_bspline(R.close_contour(c), smoothing=float(len(c)))
    d_open = cKDTree(open_fit).query(c)[0]
    d_closed = cKDTree(closed_fit).query(c)[0]
    assert d_open[-1] > 20.0, d_open[-1]
    assert d_closed.max() <= 5.0, d_closed.max()
    assert R.close_contour(R.close_contour(c)).shape[0] == len(c) + 1


def test_null_statistics_follow_their_definitions() -> None:
    c = _ring(40, seed=2, jitter=10.0)
    r = R.randomize_cluster_positions(c, c, n_iterations=199, random_seed=4)
    assert r.n_iterations == 199 and r.n_iterations_used == 199 and r.per_iteration_d.size == 199
    pool = np.sort(r.randomized_1nn_nm)
    assert abs(r.ks_statistic - R.ks_distance(np.sort(r.experimental_1nn_nm), pool)) < 1e-12
    assert abs(r.ks_null_mean - float(np.mean(r.per_iteration_d))) < 1e-12
    assert r.ks_null_sd is not None
    assert abs(r.ks_z - (r.ks_statistic - r.ks_null_mean) / r.ks_null_sd) < 1e-9
    assert abs(r.ks_p_mc_d - (1 + np.sum(r.per_iteration_d >= r.ks_statistic - 1e-12)) / 200) < 1e-12
    assert 1 / 200 <= r.ks_p_mc_median <= 1.0
    assert not hasattr(r, "ks_pvalue")
    assert r.cdf_grid_nm.size == R.CDF_GRID_POINTS and r.cdf_band_lo.shape == r.cdf_band_hi.shape
    assert np.all(r.cdf_band_lo <= r.cdf_band_hi + 1e-12)
    assert abs(r.cdf_dkw_floor - math.sqrt(math.log(2 / 0.05) / (2 * 40))) < 1e-12
    assert r.packing_fraction is not None and 0 < r.packing_fraction < 1
    assert r.n_outside_band == 0 and r.hard_core == "observed minimum"
    # the same seed gives the same control
    again = R.randomize_cluster_positions(c, c, n_iterations=199, random_seed=4)
    assert again.ks_statistic == r.ks_statistic and again.ks_p_mc_d == r.ks_p_mc_d


def test_zero_1nn_refuses_and_nonzero_falls_back() -> None:
    c = _ring(10, seed=3)
    same = np.repeat(c[:5], 2, axis=0)
    try:
        R.randomize_cluster_positions(same, c[:5], n_iterations=19)
    except ValueError as exc:
        assert "refused" in str(exc)
    else:
        raise AssertionError("every 1NN = 0 must refuse the control")
    dup = np.vstack([c, c[:1]])
    r = R.randomize_cluster_positions(dup, c, n_iterations=19)
    assert r.min_distance_nm > 0 and any("smallest non-zero" in w for w in r.warnings)


def test_incomplete_iterations_excluded_or_refused() -> None:
    c = _ring(30, spacing=250.0, d_min=120.0, seed=4)
    # a minimum separation near the mean spacing jams sequential placement in a narrow band
    params = R.RandomizationParams(band_half_width_nm=20.0, max_redraws=2, max_attempts_factor=5)
    try:
        R.randomize_cluster_positions(c, c, n_iterations=9, min_distance_nm=245.0, params=params)
    except RuntimeError as exc:
        assert "refused" in str(exc)
    else:
        raise AssertionError("a jammed band with exclusion must refuse")
    kept = R.RandomizationParams(band_half_width_nm=20.0, exclude_incomplete=False, max_attempts_factor=5)
    r = R.randomize_cluster_positions(c, c, n_iterations=9, min_distance_nm=245.0, params=kept)
    assert r.n_incomplete_iterations > 0 and r.n_redrawn == 0
    assert any("sparser" in w for w in r.warnings)
    # an easy band: nothing incomplete, nothing re-drawn
    ok = R.randomize_cluster_positions(c, c, n_iterations=19)
    assert ok.n_incomplete_iterations == 0 and ok.n_redrawn == 0


def test_centres_outside_the_band_are_counted() -> None:
    c = _ring(30, seed=5)
    inside = np.vstack([c, [[0.0, 0.0]]])        # a centre at the ring's middle, not on the contour
    r = R.randomize_cluster_positions(inside, c, n_iterations=19)
    assert r.n_outside_band == 1 and any("farther than" in w for w in r.warnings)


def test_bias_corrected_hard_core() -> None:
    assert abs(R.corrected_hard_core(80.0, 50, 12500.0) - (2500 * 80.0 - 12500.0) / (50 * 49)) < 1e-12
    c = _ring(40, seed=6)
    r = R.randomize_cluster_positions(c, c, n_iterations=19,
                                      params=R.RandomizationParams(hard_core="bias-corrected"))
    assert r.hard_core == "bias-corrected" and r.min_distance_nm < r.observed_min_1nn_nm


def test_cdf_band_crossing() -> None:
    grid = np.linspace(0, 1, 256)
    f_null = grid.copy()
    lo, hi = np.full(256, -0.05), np.full(256, 0.05)
    # null: inside the band everywhere
    assert R.cdf_band_crossing(grid, f_null + 0.01 * np.sin(20 * grid), f_null, lo, hi, 0.19) == (None, False)
    # an excess of short distances up to F ~ 0.6, then a deficit
    bump = 0.3 * np.sin(np.pi * grid / 0.6) * (grid < 0.6) - 0.25 * np.sin(np.pi * (grid - 0.6) / 0.4) * (grid >= 0.6)
    level, sig = R.cdf_band_crossing(grid, f_null + bump, f_null, lo, hi, 0.19)
    assert level is not None and abs(level - 0.6) < 0.01 and sig
    # the same shape, smaller than the DKW floor: a crossing outside the band, not significant
    level, sig = R.cdf_band_crossing(grid, f_null + 0.5 * bump, f_null, lo, hi, 0.19)
    assert level is not None and not sig


def test_crossing_strength_and_its_monte_carlo() -> None:
    grid = np.linspace(0, 1, 11)
    d = np.array([0.0, 0.1, 0.3, 0.1, 0.0, -0.1, -0.2, -0.05, 0.0, 0.0, 0.0])
    support = np.ones(11, dtype=bool)
    assert abs(float(R.crossing_strength(d, support)) - 0.2) < 1e-12
    support[:3] = False                     # the excess where the null has no support does not count
    assert abs(float(R.crossing_strength(d, support)) - 0.1) < 1e-12
    assert float(R.crossing_strength(-np.abs(d), np.ones(11, bool))) == 0.0
    # a strong planted crossing (excess of short AND deficit of long distances) is significant; a null draw is not
    rng = np.random.default_rng(9)
    iters = [np.sort(rng.gamma(9.0, 25.0, 50)) for _ in range(199)]
    pool = np.sort(np.concatenate(iters))
    m = 25
    planted = np.sort(np.concatenate([0.4 * rng.gamma(9.0, 25.0, m), 1.6 * rng.gamma(9.0, 25.0, m)]))
    out = R.cdf_band_from_samples(planted, iters, pool, 0.05, 50)
    assert out["crossing"] is not None and out["significant"] and out["p_mc"] <= 0.05, out["p_mc"]
    null = R.cdf_band_from_samples(np.sort(rng.gamma(9.0, 25.0, 50)), iters, pool, 0.05, 50)
    assert not null["significant"]


def test_params_store_never_clamps() -> None:
    p = R.RandomizationParams.from_dict({"band_half_width_nm": 300, "grid_spacing_nm": "x", "foo": 1,
                                         "hard_core": "bias-corrected", "exclude_incomplete": False})
    assert p.band_half_width_nm == 300.0 and p.grid_spacing_nm == R.DEFAULT_GRID_SPACING_NM
    assert p.outside_range() == ["band_half_width_nm"] and not p.exclude_incomplete
    from tools.mps_settings import MPSSettings
    s = MPSSettings(randomization_params={"band_half_width_nm": 300, "seed": 7, "n_iterations": 0,
                                          "smoothing_rms_nm": -1}, occupancy_sigma_cap_fraction=1.5).validate()
    assert s.randomization_params == {"seed": 7, "band_half_width_nm": 300.0}
    assert s.occupancy_sigma_cap_fraction == 1.5


def _sim_analysis(**kw):
    from tools.mps_analysis import analyze_axon
    from tools.mps_simulate_axon import SimConfig, simulate_axon
    sim = simulate_axon(SimConfig(), 3)
    return analyze_axon(sim.x_nm, sim.y_nm, sim.z_nm, source_name="sim.hdf5", pixel_size_nm=113.0,
                        pixel_size_source="yaml", n_randomizations=49, **kw)


def test_analysis_rows_and_columns() -> None:
    from tools import column_dictionary as cd
    from tools.axon_export import MEASURED_COLUMNS, SHARED_COLUMNS, TABLE_VERSION
    a = _sim_analysis()
    rec = a.export_dict()
    assert "ks_pvalue" not in rec and "ks_pvalue" not in MEASURED_COLUMNS
    for name in ("ks_null_mean", "ks_z", "ks_p_mc_median", "ks_p_mc_d", "ks_cdf_crossing_significant",
                 "randomization_n_redrawn", "randomization_packing_fraction", "randomization_n_outside_band",
                 "nn_regularity_index"):
        assert name in rec and name in MEASURED_COLUMNS and cd.describe(name), name
    for name in ("randomization_band_half_width_nm", "randomization_smoothing_rms_nm",
                 "randomization_exclude_incomplete", "randomization_max_attempts_factor",
                 "randomization_grid_spacing_nm", "randomization_hard_core", "ks_band_alpha",
                 "occupancy_sigma_cap_fraction"):
        assert name in rec and name in SHARED_COLUMNS and cd.describe(name), name
    assert TABLE_VERSION == "axon tables v7"   # v6 = SCI-3; v7 = SCI-6 after it
    r_index = a.median_1nn_nm * a.n_clusters_kept / (a.perimeter.perimeter_um * 1000.0)
    assert abs(a.nn_regularity_index - r_index) < 1e-12
    names = [row[0] for row in a.summary_rows()]
    assert "  regularity r" in names and "  Monte Carlo p" in names and "  centres outside the band" in names
    for name in ("nn_regularity_null_mean", "nn_regularity_null_lo", "nn_regularity_null_hi"):
        assert name in rec and name in MEASURED_COLUMNS and cd.describe(name), name
    null = a.nn_regularity_null
    assert null is not None and null[1] <= null[0] <= null[2]
    reg_row = dict((row[0], row) for row in a.summary_rows())["  regularity r"]
    assert "null r (this axon)" in reg_row[3] and "0.47" not in reg_row[3] and a.nn_regularity_reading in reg_row[1]
    ks = dict((row[0], row) for row in a.summary_rows())["Randomization KS test"]
    assert "p=" not in ks[1] and "z_D" in ks[1]
    assert rec["randomization_n_iterations"] == 49


def test_sigma_cap_editable_and_recorded() -> None:
    a = _sim_analysis(run_randomization=False)
    b = _sim_analysis(run_randomization=False, sigma_cap_fraction=1.0 / 3.0)
    assert a.occupancy_percent == b.occupancy_percent
    c = _sim_analysis(run_randomization=False, sigma_cap_fraction=1.0)
    assert c.export_dict()["occupancy_sigma_cap_fraction"] == 1.0
    assert c.occupancy_percent >= a.occupancy_percent
    assert c.occupancy.n_capped <= a.occupancy.n_capped


def test_icc_closed_form_against_monte_carlo() -> None:
    from tools import mps_batch as mb
    for sizes in ((9, 9), (9,) * 6, (6, 6, 6), (2, 3, 12, 20)):
        closed = mb.icc_chance_level(tuple(sizes))
        mc = mb.icc_null_for_sizes(tuple(sizes), 95.0, 20000, 1)
        assert closed is not None and mc is not None and abs(closed - mc) <= 0.01, (sizes, closed, mc)
    assert mb.icc_chance_level((9, 9), 99.0) > mb.icc_chance_level((9, 9), 95.0)


def test_icc_reads_without_adjectives() -> None:
    from tools import mps_batch as mb
    rng = np.random.default_rng(0)
    groups = [list(rng.normal(m, 1.0, 8)) for m in (0, 3, 6, 9, 12, 15)]
    d = mb.intraclass_correlation(groups, "x")
    text = d.severity
    assert "design effect" in text and "worth" in text
    for word in ("severe", "substantial", "mild"):
        assert word not in text and not any(word in w for w in d.warnings)
    labelled = mb.intraclass_correlation(groups, "x", labels=True)
    assert "large" in labelled.severity and "Zyzanski" in labelled.severity
    assert "ks_z" in mb.COMPARABLE_COLUMNS and "ks_statistic" not in mb.COMPARABLE_COLUMNS
    assert not any("p_mc" in c or "pvalue" in c for c in mb.COMPARABLE_COLUMNS)


def test_pooled_mode_helper() -> None:
    from tools.mps_spatial import estimate_mode, fd_bin_width, pool_nn_across_axons
    rng = np.random.default_rng(1)
    axons = [rng.normal(200.0, 40.0, int(rng.integers(30, 80))) for _ in range(15)]
    pooled = np.concatenate(axons)
    q1, q3 = np.percentile(pooled, [25, 75])
    assert abs(fd_bin_width(pooled) - 2 * (q3 - q1) * pooled.size ** (-1 / 3)) < 1e-9
    res = pool_nn_across_axons(axons, n_bootstrap=200)
    assert res.kde_peak_ci_nm is not None and res.kde_peak_ci_nm[0] <= 200.0 <= res.kde_peak_ci_nm[1]
    assert res.hist_peak_ci_nm is not None and res.hist_bins == "fd"
    assert abs(res.pooled_median_nm - float(np.median(pooled))) < 1e-9
    lines = res.summary_lines()
    assert lines[1].startswith("pooled median") and any("KDE mode" in x for x in lines)
    literal = estimate_mode(pooled, method="histogram", bins=50)
    counts, edges = np.histogram(pooled, bins=50)
    i = int(np.argmax(counts))
    assert literal == (edges[i] + edges[i + 1]) / 2


def test_constancy_refuses_without_animals() -> None:
    from tools.mps_constancy import occupancy_vs_calibre
    rng = np.random.default_rng(2)
    per = rng.uniform(8, 40, 24)
    occ = 20 + rng.normal(0, 2, 24)
    try:
        occupancy_vs_calibre(occ, per, ["A"] * 12 + ["B"] * 12)
    except ValueError as exc:
        assert "animal" in str(exc)
    else:
        raise AssertionError("two animals must be refused")
    animals = [f"A{i // 4}" for i in range(24)]
    res = occupancy_vs_calibre(occ, per, animals, n_bootstrap=50)
    assert res.n_animals == 6 and res.slope_ci_pp_per_um[0] < res.slope_ci_pp_per_um[1]
    assert any("pseudoreplicated" in line for line in res.lines())


def _r_of(points: np.ndarray, medians: np.ndarray) -> tuple:
    """Observed r and the null r of a ring, with P the closed polygon through its centres (as perimeter_um)."""
    from tools.mps_analysis import regularity_null
    closed = np.vstack([points, points[:1]])
    p_um = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1))) / 1000.0
    nn = R._self_excluded_1nn(points)
    k = len(points)
    return float(np.median(nn)) * k / (p_um * 1000.0), regularity_null(medians, k, p_um)


def test_null_regularity_covers_a_null_draw() -> None:
    # An exchangeable null (observed = one more draw of the same process, the true hard core given): the observed r
    # falls inside this axon's 95 % null interval of r in about 95 % of draws.
    contour = np.column_stack([1200 * np.cos(np.linspace(0, 2 * np.pi, 200, endpoint=False)),
                               1200 * np.sin(np.linspace(0, 2 * np.pi, 200, endpoint=False))])
    smoothed = R.smooth_contour_bspline(R.close_contour(contour), smoothing=200.0)
    cand = R.build_annulus_candidates(smoothed)
    rng = np.random.default_rng(14)
    inside = n = 0
    for _ in range(40):
        obs = R.place_with_min_distance(cand, 30, 80.0, rng)
        if len(obs) < 30:
            continue
        order = np.argsort(np.arctan2(obs[:, 1], obs[:, 0]))
        obs = obs[order]
        res = R.randomize_cluster_positions(obs, contour, n_iterations=99, min_distance_nm=80.0,
                                            random_seed=int(rng.integers(1 << 30)))
        r_obs, null = _r_of(obs, res.per_iteration_medians_nm)
        inside += int(null[1] <= r_obs <= null[2])
        n += 1
    assert n >= 35 and inside / n >= 0.85, (inside, n)


def test_null_regularity_reads_regular_and_twins() -> None:
    from tools.mps_analysis import regularity_reading
    th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.column_stack([1600 * np.cos(th), 1600 * np.sin(th)])
    res = R.randomize_cluster_positions(ring + 0.0, ring, n_iterations=99, min_distance_nm=80.0)
    r_obs, null = _r_of(ring, res.per_iteration_medians_nm)
    assert abs(r_obs - 1.0) < 0.01 and regularity_reading(r_obs, null).startswith("above null"), (r_obs, null)
    # 20 sites, each split into twins 40 nm apart along the ring: the median 1NN is the twin distance
    th20 = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    twins = np.vstack([np.column_stack([1600 * np.cos(th20 + s), 1600 * np.sin(th20 + s)])
                       for s in (0.0, 40.0 / 1600)])
    twins = twins[np.argsort(np.arctan2(twins[:, 1], twins[:, 0]))]
    res = R.randomize_cluster_positions(twins, twins, n_iterations=99)
    r_obs, null = _r_of(twins, res.per_iteration_medians_nm)
    assert regularity_reading(r_obs, null).startswith("below null"), (r_obs, null)


def test_methods_page_exists_and_is_linked() -> None:
    from tools import mps_param_registry as reg
    from tools import mps_results_window as rw
    page = os.path.join(REPO_ROOT, R.METHODS_DOC)
    text = open(page, encoding="utf-8").read()
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 10. Decision record",
                    "## References"):
        assert heading in text, heading
    for script in ("sim1_ks_mc.py", "sim1b_exchangeable.py", "sim2_icc_null.py", "sim3_nn_factors.py",
                   "sim4_sigma_cap.py"):
        assert script in text and os.path.exists(os.path.join(REPO_ROOT, "docs", "sims", "randomization", script))
    assert R.METHODS_DOC in open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8").read()
    assert R.METHODS_DOC in rw.METHODS_TIP
    for key in ("randomization.band_half_width_nm", "randomization.cdf_band_alpha", "reference.ks",
                "occupancy.sigma_cap_fraction", "batch.icc_percentile"):
        assert f"Methods: {R.METHODS_DOC}" in reg.tooltip(key), key


def _run_all() -> int:
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL  {name}: {error}")
            traceback.print_exc()
    print(f"{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
