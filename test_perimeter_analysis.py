"""Orchestration and batch of the perimeter density analysis (P10).

Simulated data only (``tools.mps_perimeter_sim``). The automatic contour (P3) is injected: the TRUE curve of the
simulator, a circle fitted to the cloud, or a drawn trace subsampled from the true curve, so these tests do not
depend on P3. Run one file per process: ``python -m pytest test_perimeter_analysis.py``. Every test < 60 s.
"""
import csv
import hashlib
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import batch_perimeter as bp  # noqa: E402
from tools import mps_perimeter_pool as pool  # noqa: E402
from tools import mps_perimeter_types as pdt  # noqa: E402
from tools.mps_identity import AxonIdentity  # noqa: E402
from tools.mps_matching import SmoothPath  # noqa: E402
from tools.mps_perimeter_analysis import analyze_perimeter, counter_clockwise  # noqa: E402
from tools.mps_perimeter_sim import simulate_scenario  # noqa: E402
from tools.perimeter_export import PERIMETER_COLUMNS  # noqa: E402

SOURCE = "C:/data/26.1.15_ROI 1/26.1.15_ROI1_axon2.hdf5"   # synthetic path: year 26, ROI 1, axon 2
N_NULL = 499            # [ad hoc] enough for the p's the assertions use (floor 0.002), keeps every test fast
FAST = replace(pdt.PerimeterParams(), n_null=N_NULL)


def _path(points):
    p = np.asarray(points, dtype=np.float64)
    nxt = np.roll(p, -1, axis=0)
    edges = np.hypot(nxt[:, 0] - p[:, 0], nxt[:, 1] - p[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    return SmoothPath(points_nm=p, vertex_row=np.arange(len(p), dtype=np.intp), edges_nm=edges, cum_nm=cum,
                      length_nm=float(cum[-1]))


def _contour(path, method="density_nodes"):
    nan = float("nan")
    return pdt.AutoContour(path=path, length_nm=float(path.length_nm), method=method, knot_nm=400.0,
                           residual_sd_nm=nan, n_self_crossings=0, nodes_nm=np.asarray(path.points_nm)[::100],
                           polar_distance_nm=nan, star_shaped=True, status=pdt.Status.OK)


def true_contour(sim, reverse=False):
    """Contour function that returns the simulator's true curve (reversed to clockwise when asked)."""
    path = sim.true_path
    if reverse:
        pts = np.asarray(path.points_nm)
        path = _path(pts[np.mod(-np.arange(len(pts)), len(pts))])     # same row 0, opposite direction

    def fn(x, y, **kw):
        return _contour(path, kw.get("method", "density_nodes"))
    return fn


def circle_contour(x, y, **kw):
    """Reference contour: the circle at the cloud's centroid with its median radius (2 nm table)."""
    cx, cy = float(np.mean(x)), float(np.mean(y))
    rad = float(np.median(np.hypot(x - cx, y - cy)))
    n = int(math.ceil(2 * math.pi * rad / 2.0))
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return _contour(_path(np.column_stack([cx + rad * np.cos(th), cy + rad * np.sin(th)])), kw.get("method", ""))


def run(sim, params=FAST, contour_fn=None, source=SOURCE, **kw):
    return analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={"sx": sim.sx, "sy": sim.sy}, info=[],
                             identity=None, source_path=source, params=params,
                             contour_fn=contour_fn or true_contour(sim), **kw)


def _row_complete(res):
    row = pool.perimeter_row(res)
    missing = [c for c in PERIMETER_COLUMNS if c not in row]
    # a failed axon keeps every identity / z0 / status column; the later blocks are absent only past the failure
    return row, missing


# ----------------------------------------------------------------------------------------------------
# End to end, scenarios with known truth
# ----------------------------------------------------------------------------------------------------
def test_end_to_end_flat_under_30s_with_default_nulls():
    sim = simulate_scenario("a_flat")
    t0 = time.perf_counter()
    res = run(sim, params=pdt.PerimeterParams())
    elapsed = time.perf_counter() - t0
    assert res.status == pdt.Status.OK, res.failure
    assert elapsed < 30.0, elapsed
    row, missing = _row_complete(res)
    assert missing == []
    assert row["pd_status"] == "ok" and row["trace_label"] == "26.1.2" and row["trace_complete"]
    assert set(res.a) == {"contrast", "period"} and set(res.c) == {(-1, 0), (0, 1), (-1, 1)}
    assert res.b is not None and math.isfinite(res.b.test.z)
    assert res.a["contrast"].p > 0.01                       # flat: no attractor (deterministic seed)
    assert set(res.sensitivity) == {50.0, 150.0} and set(res.guard_sensitivity) == {42.5}
    assert res.profiles.frame == "z0_max"
    # each lateral reports its percentage of the central slice; nothing is dropped
    n = res.slices.n_locs
    assert row["pd_pct_m1_of_0"] == pytest.approx(100.0 * n[0] / n[1])
    # contour support complete on a flat profile; no contour reversal for the CCW true curve
    assert row["pd_contour_unsupported_fraction"] == 0.0 and not res.diagnostics.contour_reversed
    # per-bin radial median exists for the three slices on the central grid
    assert all(res.diagnostics.radial_median_by_bin_nm[k].size == res.profiles.by_slice[0].n_bins
               for k in pdt.SLICES)
    # identity proposed from the path, axon_id is the per-axon table key of a pick without ROI
    assert res.identity is not None and res.identity.roi_name and len(res.axon_id) == 10


def test_attractor_gives_positive_contrast():
    res = run(simulate_scenario("a_attractor", n_locs=30000))
    assert res.status == pdt.Status.OK
    assert res.a["contrast"].value > 0 and res.a["contrast"].z > 2.0


def test_repulsor_gives_negative_contrast():
    # one axon: only the sign (the per-axon contrast has little power against the shift null, z = -1.5 here);
    # the pooled test is the one with power (P7, and below on several axons)
    res = run(simulate_scenario("a_inhibition", n_locs=30000))
    assert res.status == pdt.Status.OK
    assert res.a["contrast"].value < 0 and res.a["contrast"].z < 0
    many = [run(simulate_scenario("a_inhibition", n_locs=20000, seed=s)) for s in range(6)]
    pooled = pool.pooled_a(many)
    (pr,) = pooled.values()
    assert pr.contrast.value < 0 and pr.contrast.p < 0.05


def test_curvature_effect_detected_and_null_not():
    pos = run(simulate_scenario("b_curvature", n_locs=20000))
    neg = run(simulate_scenario("b_null", n_locs=20000))
    # denser where more convex = denser where the tortuosity (chord / arc) is LOWER: Spearman < 0
    assert pos.b.test.value < 0 and pos.b.test.p < 0.05 and pos.b.resolved
    assert neg.b.test.p > 0.05
    assert math.isfinite(pos.b.tau_vs_offset)               # the contour-bias control is computed


def test_axial_leak_between_slices():
    res = run(simulate_scenario("c_leak", n_locs=60000))
    r = {k: v.value for k, v in res.c.items()}
    assert res.diagnostics.leak_expected_r == pytest.approx(0.27, abs=0.02)      # lpz 47 nm, no lpz column
    assert any("no lpz column" in w for w in res.warnings)
    assert 0.5 * (r[(-1, 0)] + r[(0, 1)]) > r[(-1, 1)] + 0.1                     # neighbours carry the leak
    assert abs(r[(-1, 1)]) < 0.2
    g = res.guard_sensitivity[42.5]
    assert 0.5 * (g.c[(-1, 0)].value + g.c[(0, 1)].value) < 0.5 * (r[(-1, 0)] + r[(0, 1)])
    assert g.leak_expected_r < res.diagnostics.leak_expected_r


def test_identical_slices_register():
    res = run(simulate_scenario("c_identical", n_locs=30000))
    # ~94 bins: every rotation is used and the p floor is 1/95
    assert res.c[(-1, 1)].value > 0.5 and res.c[(-1, 1)].p < 0.05 and res.c[(-1, 1)].z > 3
    assert res.diagnostics.c_reading == "registry between slices"


def test_focus_differs_from_peak_both_distances_reported():
    sim = simulate_scenario("z_peak_focus", n_locs=30000)
    res = run(sim)
    assert res.status == pdt.Status.OK
    assert res.z0.peak_nm == pytest.approx(0.0, abs=20.0)
    assert res.z0.focus_nominal_nm is None                  # no calibration record in a simulated file
    assert res.z0.focus_empirical_nm == pytest.approx(-43.0, abs=15.0)
    assert res.z0.peak_minus_focus_empirical_nm == pytest.approx(43.0, abs=25.0)
    row = pool.perimeter_row(res)
    assert row["pd_abs_peak_minus_focus_empirical_nm"] == pytest.approx(abs(res.z0.peak_minus_focus_empirical_nm))
    emp = run(sim, params=replace(FAST, z0_criterion=pdt.Z0Criterion.FOCUS_EMPIRICAL))
    assert emp.status == pdt.Status.OK and emp.z0.z0_nm == pytest.approx(res.z0.focus_empirical_nm)
    nom = run(sim, params=replace(FAST, z0_criterion=pdt.Z0Criterion.FOCUS_NOMINAL))
    assert nom.status == pdt.Status.NO_FOCUS and nom.z0.z0_nm is None
    nrow, missing = _row_complete(nom)
    assert nrow["pd_status"] == "no focus" and nrow["pd_failure"]
    assert nrow["pd_peak_minus_focus_empirical_nm"] == pytest.approx(res.z0.peak_minus_focus_empirical_nm)
    assert "pd_z_peak_nm" not in missing and "trace_label" not in missing


def test_gap_is_not_measured_and_counted_as_unsupported():
    sim = simulate_scenario("contour_c_gap", n_locs=20000)
    res = run(sim)
    assert res.status == pdt.Status.OK, res.failure
    frac = res.diagnostics.unsupported_fraction
    expected = sim.truth["gap_nm"] / sim.truth["length_nm"]
    assert expected - 0.06 < frac < expected + 0.03
    assert not res.profiles.by_slice[0].measured.all()


def test_clockwise_contour_is_normalised():
    sim = simulate_scenario("c_identical", n_locs=20000)
    ccw = run(sim)
    cw = run(sim, contour_fn=true_contour(sim, reverse=True))
    assert cw.diagnostics.contour_reversed and not ccw.diagnostics.contour_reversed
    assert any("clockwise" in w for w in cw.warnings)
    for pair in ccw.c:
        assert cw.c[pair].value == pytest.approx(ccw.c[pair].value, abs=1e-6)
    assert cw.a["contrast"].value == pytest.approx(ccw.a["contrast"].value, abs=1e-6)
    p_cw, _ = counter_clockwise(cw.contour.path)
    assert np.allclose(p_cw.points_nm, cw.contour.path.points_nm)      # already counter-clockwise


def test_band_sensitivity_recomputes_profiles():
    res = run(simulate_scenario("a_flat"))
    main0 = res.profiles.by_slice[0]
    for hw, s in res.sensitivity.items():
        assert s.half_width_nm == hw
    assert res.diagnostics.band_sensitivity_flag.startswith(("consistent", "differs"))
    # the profiles behind 50 nm hold fewer localizations than the 100 nm ones: they were recomputed
    from tools.mps_perimeter_density import project, slice_profiles
    s_, r_ = project(res.contour.path, simulate_scenario("a_flat").x_nm, simulate_scenario("a_flat").y_nm)
    narrow = slice_profiles(s_, r_, res.slices.label, res.contour.path.length_nm, half_width_nm=50.0)
    assert narrow.by_slice[0].n_locs < main0.n_locs
    rows = pool.sensitivity_rows([res])
    assert [r["setting"] for r in rows] == ["band (main)", "band", "band", "axial guard"]


def test_drawn_contour_method_and_comparison():
    sim = simulate_scenario("a_flat")
    trace = np.asarray(sim.true_path.points_nm)[::25]           # ~200 vertices, a hand-drawn-like trace
    res = run(sim, params=replace(FAST, contour_method="drawn"), drawn_contour=trace)
    assert res.status == pdt.Status.OK and res.contour.method == "drawn"
    auto = run(sim, drawn_contour=trace)                         # auto (injected) + drawn only compared
    assert auto.contour.method == "density_nodes" and auto.diagnostics.drawn_vs_auto_nm < 5.0


# ----------------------------------------------------------------------------------------------------
# Failure states: a complete row with the reason, never an exception
# ----------------------------------------------------------------------------------------------------
def _failed_contour(x, y, **kw):
    return replace(_contour(_path([[0, 0], [1, 0], [0, 1]])), path=None, status=pdt.Status.CONTOUR_FAILED,
                   warnings=("2 self-crossings",))


def _raising_contour(x, y, **kw):
    raise RuntimeError("fit aborted")


@pytest.mark.parametrize("case,status", [
    ("too_few", pdt.Status.TOO_FEW_LOCS), ("contour_rejected", pdt.Status.CONTOUR_FAILED),
    ("contour_raises", pdt.Status.CONTOUR_FAILED), ("p3_not_integrated", pdt.Status.CONTOUR_FAILED),
    ("lengths", pdt.Status.INVALID_INPUT), ("all_nan_z", pdt.Status.TOO_FEW_LOCS),
    ("drawn_without_trace", pdt.Status.CONTOUR_FAILED)])
def test_failure_states(case, status):
    sim = simulate_scenario("a_flat", n_locs=3000)
    kw = {}
    if case == "too_few":
        sim = simulate_scenario("a_flat", n_locs=40)
    elif case == "contour_rejected":
        kw["contour_fn"] = _failed_contour
    elif case == "contour_raises":
        kw["contour_fn"] = _raising_contour
    if case == "p3_not_integrated":
        res = analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={}, info=[], identity=None,
                                source_path=SOURCE, params=FAST)
    elif case == "lengths":
        res = analyze_perimeter(sim.x_nm, sim.y_nm[:-1], sim.z_nm, columns={}, info=[], identity=None,
                                source_path=SOURCE, params=FAST)
    elif case == "all_nan_z":
        res = analyze_perimeter(sim.x_nm, sim.y_nm, np.full_like(sim.z_nm, np.nan), columns={}, info=[],
                                identity=None, source_path=SOURCE, params=FAST)
    elif case == "drawn_without_trace":
        res = run(sim, params=replace(FAST, contour_method="drawn"))
    else:
        res = run(sim, **kw)
    if case == "p3_not_integrated":
        try:
            from tools.mps_auto_contour import auto_contour
            auto_contour(sim.x_nm, sim.y_nm)
            pytest.skip("P3 is integrated: the default contour works")
        except NotImplementedError:
            pass
    assert res.status == status, (res.status, res.failure)
    assert res.failure
    row = pool.perimeter_row(res)
    assert row["pd_status"] == status.value and row["trace_label"] == "26.1.2"
    assert row["axon_id"] == res.axon_id


# ----------------------------------------------------------------------------------------------------
# Pooling
# ----------------------------------------------------------------------------------------------------
def _three(animal="", genotype=""):
    out = []
    for i, seed in enumerate((1, 2, 3)):
        sim = simulate_scenario("a_clumps", seed=seed, n_locs=4000)
        ident = AxonIdentity(genotype=genotype, animal=animal, roi_name=f"ROI {1 + i % 2}", axon_name=f"Axon {i}")
        out.append(analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={}, info=[], identity=ident,
                                     source_path=f"C:/d/26.1.15_ROI{1 + i % 2}_axon{i}.hdf5", params=FAST,
                                     contour_fn=true_contour(sim)))
    return out


def test_pool_without_animal_and_with_a_provisional_single_animal():
    res = _three()
    rows = pool.perimeter_rows(res)
    by = pool.summarise_by_animal(rows)
    assert {r["animal"] for r in by} == {pool.NO_ANIMAL_GROUP} and len(by) == 2      # one row per ROI
    assert all(r["p_pd_b_z"] is None for r in by)
    cmp = pool.compare_genotypes(rows)
    assert cmp["descriptive_only"] and any(pool.NO_ANIMAL_GROUP in n for n in cmp["notes"])
    pooled = pool.pooled_a(res)
    assert list(pooled) == [pool.NO_ANIMAL_GROUP]

    one = _three(animal=pool.PROVISIONAL_ANIMAL, genotype="WT")
    rows1 = pool.perimeter_rows(one)
    by1 = pool.summarise_by_animal(rows1)
    assert len(by1) == 1 and by1[0]["provisional"] and "n_animals = 1" in by1[0]["note"]
    cmp1 = pool.compare_genotypes(rows1)
    assert cmp1["descriptive_only"] and all(e["p"] is None for e in cmp1["measures"].values())
    p1 = pool.pooled_a(one)
    (name, pr), = p1.items()
    assert "n_animals = 1" in pr.reading
    # v20: three axons are below the minimum of a pooled reading of A (MIN_POOLED_A_AXONS)
    assert pool.a_reading_label(pr).startswith("n too small (n = 3 axons < 9): not interpreted; computed label")
    (bare,) = pool.pooled_a(one, min_axons=1).values()
    assert pool.a_reading_label(bare) in ("attractor", "regular", "inhibition zone", "no structure detectable",
                                          "not computable", "inhibition zone + regular", "attractor + regular")
    # the order of the axons does not change the pooled result (sorted by axon_id inside)
    p2 = pool.pooled_a(list(reversed(one)))
    assert p2[name].contrast.p == pr.contrast.p and p2[name].period.value == pr.period.value
    focus = pool.summarise_focus_by_roi(rows1)
    assert focus[0]["group"] == "(all axons)" and {f["group"] for f in focus} >= {"ROI 1", "ROI 2"}


def test_long_table_rows():
    res = _three()
    prof = pool.profile_rows(res)
    n_bins = sum(r.profiles.by_slice[0].n_bins for r in res)
    assert len(prof) == 3 * n_bins
    lag0 = [p for p in prof if p["slice"] == 0 and p["lag_nm"] == 0.0]
    assert all(p["relative"] == 1.0 for p in lag0)            # lag 0 is the central slice's maximum
    assert len(pool.contour_rows(res)) == sum(len(r.contour.path.points_nm) for r in res)


# ----------------------------------------------------------------------------------------------------
# Batch
# ----------------------------------------------------------------------------------------------------
def _write_sim_csv(path, sim):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x [nm]", "y [nm]", "z [nm]", "sigma1 [nm]", "sigma2 [nm]"])
        for row in zip(sim.x_nm, sim.y_nm, sim.z_nm, sim.sx, sim.sy):
            w.writerow([f"{v:.3f}" for v in row])


def _digest(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def test_batch_over_a_folder_never_overwrites(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    names = ["26.1.15_ROI1_axon1.csv", "26.1.15_ROI1_axon2.csv", "ROI2_axon1.csv"]
    for i, name in enumerate(names):
        _write_sim_csv(data / name, simulate_scenario("c_identical", seed=10 + i, n_locs=5000))
    (data / "notes_axon.txt").write_text("not data")
    before = {n: _digest(data / n) for n in names}
    params = replace(pdt.PerimeterParams(), n_null=199)
    stem = str(tmp_path / "out" / "run")
    res = bp.process_folder(str(data), stem, params, animal=pool.PROVISIONAL_ANIMAL, contour_fn=circle_contour)
    assert len(res) == 3 and all(r.status == pdt.Status.OK for r in res), [r.failure for r in res]
    assert bp.process_folder.last_stem == stem
    for t in bp.TABLES:
        assert Path(f"{stem}_perimeter_{t}.csv").exists(), t
    with open(f"{stem}_perimeter_axons.csv", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert set(PERIMETER_COLUMNS) <= set(header)
    labels = {r.trace.label for r in res}
    assert {"26.1.1", "26.1.2", "NA.2.1"} == labels
    log = list(csv.DictReader(open(f"{stem}_perimeter_log.csv", encoding="utf-8")))
    assert any("year reads NA" in (r["warnings"] or "") for r in log)
    # second run: a new stem, nothing overwritten, inputs untouched
    first = _digest(f"{stem}_perimeter_axons.csv")
    bp.process_folder(str(data), stem, params, contour_fn=circle_contour)
    assert bp.process_folder.last_stem == stem + "_run2"
    assert _digest(f"{stem}_perimeter_axons.csv") == first
    assert {n: _digest(data / n) for n in names} == before


def test_batch_main_arguments(tmp_path):
    assert bp.main([str(tmp_path / "missing"), "--out", str(tmp_path / "o")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert bp.main([str(empty), "--out", str(tmp_path / "o")]) == 2
    assert not list(tmp_path.glob("o_perimeter_*.csv"))
    assert bp.main(["--bogus"]) == 2
