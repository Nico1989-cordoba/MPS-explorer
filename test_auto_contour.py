"""Tests of the automatic contour of the perimeter density analysis (tools/mps_auto_contour.py, package P3).

Simulated data only (validate_auto_contour.simulate_ring: known membrane curve). Each test runs in well under
60 s on the laptop; run this file in its own pytest process.
"""
import ast
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tools import mps_auto_contour as ac  # noqa: E402
from tools.mps_membrane import _linear_closed_path  # noqa: E402
from tools.mps_perimeter_types import AutoContour, Status  # noqa: E402
import validate_auto_contour as V  # noqa: E402


def _signed_area(points):
    p = np.asarray(points)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _run(sim, **kw):
    det = ac.contour_details(sim.x, sim.y, **kw)
    return det, V.score(sim, det)


# ----------------------------------------------------------------------------------------------------
# Independence from clustering, signature
# ----------------------------------------------------------------------------------------------------
def test_module_imports_nothing_from_clustering():
    tree = ast.parse((ROOT / "tools" / "mps_auto_contour.py").read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert names, "no imports parsed"
    for n in names:
        low = n.lower()
        assert "cluster" not in low and "dbscan" not in low and "hdbscan" not in low and "sklearn" not in low, n


def test_signature_takes_no_cluster_labels_or_cluster_parameters():
    for fn in (ac.auto_contour, ac.contour_details, ac.density_nodes):
        params = set(inspect.signature(fn).parameters)
        for banned in ("labels", "label", "eps", "min_samples", "clusters", "cluster_labels"):
            assert banned not in params, (fn.__name__, banned)


def test_public_names_of_the_contract():
    for name in ("density_nodes", "polar_contour", "is_star_shaped", "auto_contour"):
        assert callable(getattr(ac, name))
    sig = inspect.signature(ac.auto_contour)
    for name in ("method", "knot_nm", "node_drop_nm", "rounds", "drawn"):
        assert name in sig.parameters


# ----------------------------------------------------------------------------------------------------
# Positive control
# ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", V.SHAPES)
def test_positive_control_each_shape(shape):
    for seed, n in ((0, 4000), (1, 9000)):
        sim = V.simulate_ring(shape, 500 + seed, n_membrane=n, background_frac=0.15)
        det, sc = _run(sim)
        c = det.contour
        assert c.status is Status.OK, (shape, seed, c.warnings)
        assert c.n_self_crossings == 0
        assert sc["err_mean"] <= 25.0, (shape, seed, sc)
        assert sc["err_p95"] <= 80.0, (shape, seed, sc)
        assert abs(sc["len_ratio"] - 1.0) <= 0.05, (shape, seed, sc)
        assert c.path is not None and c.length_nm == pytest.approx(c.path.length_nm)


@pytest.mark.parametrize("bg", [0.0, 0.15, 0.30])
def test_background_levels_same_thresholds(bg):
    for shape in ("ellipse", "C"):
        sim = V.simulate_ring(shape, 600, background_frac=bg)
        det, sc = _run(sim)
        assert det.contour.status is Status.OK, (shape, bg, det.contour.warnings)
        assert sc["err_mean"] <= 25.0 and sc["err_p95"] <= 80.0, (shape, bg, sc)


def test_deterministic():
    sim = V.simulate_ring("star", 7, **V.CONDITIONS["hard"])
    a = ac.auto_contour(sim.x, sim.y)
    b = ac.auto_contour(sim.x, sim.y)
    assert a.status is b.status
    assert a.path is not None and b.path is not None
    np.testing.assert_array_equal(a.path.points_nm, b.path.points_nm)
    np.testing.assert_array_equal(a.nodes_nm, b.nodes_nm)
    assert a.warnings == b.warnings


# ----------------------------------------------------------------------------------------------------
# Hazards of the plan
# ----------------------------------------------------------------------------------------------------
def test_dominant_cluster_does_not_pull_the_curve():
    # "wavy" is left out: its own bends (radius 267 nm) are below what 400 nm knots follow, so its error 500 nm
    # from any point is the shape's, not the cluster's (validation table, condition dominant20)
    for shape in ("ellipse", "bilobe", "C"):
        sim = V.simulate_ring(shape, 700, **V.CONDITIONS["dominant20"])
        det, sc = _run(sim)
        assert det.contour.status is Status.OK
        assert sc["err_dominant_500"] <= 30.0, (shape, sc)


def test_gap_of_two_microns_closes_and_leaves_an_empty_stretch():
    # no background: with background the band still holds localizations every few hundred nm across the gap,
    # so a rule that marks "no localization in the band" cannot see the gap (reported to the integration)
    for shape in ("ellipse", "star"):
        sim = V.simulate_ring(shape, 800, gap_nm=2000.0, background_frac=0.0)
        det, sc = _run(sim)
        c = det.contour
        assert c.status is Status.OK, c.warnings
        assert c.path is not None
        s, r = ac.project(c.path, sim.x, sim.y)
        s_band = np.sort(s[np.abs(r) <= 100.0])
        gaps = np.diff(np.concatenate([s_band, [s_band[0] + c.length_nm]]))
        # the curve bridges the gap; the empty stretch (> UNSUPPORTED_GAP_NM = 400 nm) is what the profile marks
        assert gaps.max() >= 1500.0, (shape, gaps.max())
        assert sc["err_mean"] <= 25.0, (shape, sc)


def test_interior_lump_mostly_left_out_and_the_silent_rate_is_bounded():
    # review R1 (H2): 5 of 50 lump cases went through the lump with status ok and no warning. DECLARED limitation
    # (report v2): the quality gate reads stretches of >= 700 nm, a lump dent is shorter, so a curve through a lump
    # can still come out "reliable". V1 (blob10, 10 seeds x 8 shapes): 11 of 77 accepted. This test pins the rate on
    # the five base shapes x 8 seeds so that it cannot silently grow.
    silent = []
    left_out = 0
    for shape in V.SHAPES:
        for seed in range(60, 68):
            sim = V.simulate_ring(shape, seed, **V.CONDITIONS["blob10"])
            det, sc = _run(sim)
            if det.contour.status is not Status.OK:
                continue
            assert det.assessment is not None and det.assessment.quality is not None
            if sc["blob_node_min_nm"] > 200.0 and sc["err_p95"] <= 170.0:
                left_out += 1
            elif sc["err_p95"] > 170.0 and det.assessment.reliable:
                silent.append((shape, seed, round(sc["err_p95"])))
    assert len(silent) <= 5, silent
    assert left_out >= 30, left_out          # of 40: the lump rule itself keeps working


def test_negative_control_filled_disk_is_rejected():
    for seed in range(3):
        x, y = V.disk_cloud(seed)
        c = ac.auto_contour(x, y)
        assert c.status is Status.CONTOUR_FAILED
        assert c.path is None
        assert ac.failure_reason(c) in ac.FAILURE_REASONS
        assert c.warnings[0].startswith(ac.REJECT_PREFIX)


def test_too_few_localizations():
    sim = V.simulate_ring("ellipse", 1, n_membrane=150, background_frac=0.0)
    c = ac.auto_contour(sim.x[:150], sim.y[:150])
    assert c.status is Status.CONTOUR_FAILED
    assert ac.failure_reason(c) == ac.FAIL_TOO_FEW_LOCS


def test_too_few_nodes():
    rng = np.random.default_rng(3)
    t = np.repeat(np.arange(5) * 2 * np.pi / 5, 200)
    x = 1000 * np.cos(t) + rng.normal(0, 15, t.size)
    y = 1000 * np.sin(t) + rng.normal(0, 15, t.size)
    c = ac.auto_contour(x, y, fallbacks=False)
    assert c.status is Status.CONTOUR_FAILED
    assert ac.failure_reason(c) == ac.FAIL_TOO_FEW_NODES


# ----------------------------------------------------------------------------------------------------
# The rejection rules on constructed curves
# ----------------------------------------------------------------------------------------------------
def _ring_cloud(n=6000, r=1200.0, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    rr = r + rng.normal(0, 20, n)
    return rr * np.cos(t), rr * np.sin(t)


def test_self_crossing_curve_is_rejected():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    eight = np.column_stack([1200 * np.sin(t), 600 * np.sin(2 * t)])
    x, y = _ring_cloud()
    cand = ac._check("test", "drawn", 400.0, _linear_closed_path(eight), eight, 20.0, x, y, ())
    assert cand.reason == ac.FAIL_SELF_CROSSING
    assert cand.n_self_crossings >= 1


def test_snaking_curve_is_rejected_as_implausible_length():
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    r = 1000 + 300 * np.cos(30 * t)                    # 30 deep lobes: length ~3x its hull's perimeter
    snake = np.column_stack([r * np.cos(t), r * np.sin(t)])
    path = _linear_closed_path(snake)
    assert ac.hull_ratio(path) > ac.MAX_HULL_RATIO
    x, y = _ring_cloud()
    cand = ac._check("test", "drawn", 400.0, path, snake, 20.0, x, y, ())
    assert cand.reason == ac.FAIL_IMPLAUSIBLE_LENGTH


def test_inward_notch_is_rejected():
    t = np.linspace(0, 2 * np.pi, 1500, endpoint=False)
    dip = 300.0 * np.exp(-0.5 * (((t - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi) / 0.12) ** 2)   # 300 nm deep, ~150 nm wide
    r = 1200.0 - dip
    notched = np.column_stack([r * np.cos(t), r * np.sin(t)])
    path = _linear_closed_path(notched)
    radius, where = ac.tightest_concave(path)
    assert radius < ac.NOTCH_MIN_RADIUS_NM
    assert abs(where[0]) < 200 and where[1] > 0
    x, y = _ring_cloud()
    cand = ac._check("test", "drawn", 400.0, path, notched, 20.0, x, y, ())
    assert cand.reason == ac.FAIL_NOTCH
    smooth = _linear_closed_path(np.column_stack([1200 * np.cos(t), 1200 * np.sin(t)]))
    assert ac.tightest_concave(smooth)[0] == float("inf")


def test_lumen_ratio_ring_versus_disk():
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = _linear_closed_path(np.column_stack([1200 * np.cos(t), 1200 * np.sin(t)]))
    x, y = _ring_cloud()
    ring_ratio, _ = ac.lumen_ratio(circle, x, y)
    xd, yd = V.disk_cloud(0, radius_nm=1250.0)
    disk_ratio, _ = ac.lumen_ratio(circle, xd, yd)
    assert ring_ratio < 0.05
    assert disk_ratio > ac.MAX_LUMEN_RATIO


def test_drawn_method_needs_a_polygon():
    x, y = _ring_cloud()
    c = ac.auto_contour(x, y, method="drawn")
    assert c.status is Status.CONTOUR_FAILED and ac.failure_reason(c) == ac.FAIL_NO_DRAWN
    with pytest.raises(ValueError):
        ac.auto_contour(x, y, method="snake")


# ----------------------------------------------------------------------------------------------------
# Orientation: always counter-clockwise
# ----------------------------------------------------------------------------------------------------
def test_counter_clockwise_keeps_origin_and_length():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    cw = _linear_closed_path(np.column_stack([1500 * np.cos(-t), 900 * np.sin(-t)]))
    assert _signed_area(cw.points_nm) < 0
    ccw = ac.counter_clockwise(cw)
    assert _signed_area(ccw.points_nm) > 0
    np.testing.assert_array_equal(ccw.points_nm[0], cw.points_nm[0])
    assert ccw.length_nm == pytest.approx(cw.length_nm, rel=1e-12)
    assert ccw.cum_nm[-1] == pytest.approx(ccw.length_nm)
    assert ac.counter_clockwise(ccw) is ccw


@pytest.mark.parametrize("clockwise", [False, True])
def test_drawn_contour_comes_out_counter_clockwise(clockwise):
    x, y = _ring_cloud(seed=4)
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    if clockwise:
        t = -t
    drawn = np.column_stack([1230 * np.cos(t), 1170 * np.sin(t)])
    c = ac.auto_contour(x, y, method="drawn", drawn=drawn)
    assert c.status is Status.OK, c.warnings
    assert c.method == "drawn"
    assert c.path is not None and _signed_area(c.path.points_nm) > 0
    assert c.length_nm == pytest.approx(2 * np.pi * 1200, rel=0.03)


def test_automatic_contours_come_out_counter_clockwise():
    for shape in ("ellipse", "C"):
        sim = V.simulate_ring(shape, 900, background_frac=0.15)
        # mirror the cloud: whatever orientation the tour takes, the output must be counter-clockwise
        for sx in (1.0, -1.0):
            det = ac.contour_details(sx * sim.x, sim.y)
            assert det.contour.path is not None
            assert _signed_area(det.contour.path.points_nm) > 0
            if det.polar_path is not None:
                assert _signed_area(det.polar_path.points_nm) > 0


# ----------------------------------------------------------------------------------------------------
# Polar contour and star shape
# ----------------------------------------------------------------------------------------------------
def test_star_shape_and_polar_contour():
    t = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    circle = _linear_closed_path(np.column_stack([1000 * np.cos(t), 1000 * np.sin(t)]))
    assert ac.is_star_shaped(circle, (0.0, 0.0))
    assert not ac.is_star_shaped(circle, (2000.0, 0.0))
    c_shape = _linear_closed_path(V.shape_polygon("C"))
    assert not ac.is_star_shaped(c_shape, ac._area_centroid(np.asarray(c_shape.points_nm)))
    x, y = _ring_cloud(r=1000.0)
    poly = ac.polar_contour(x, y)
    assert poly is not None and poly.shape == (ac.POLAR_N_SECTORS, 2)
    assert np.allclose(np.hypot(poly[:, 0], poly[:, 1]), 1000.0, atol=40.0)
    assert ac.polar_contour(x[:10], y[:10]) is None


def test_polar_method_and_cross_check():
    sim = V.simulate_ring("ellipse", 3, background_frac=0.15)
    p = ac.auto_contour(sim.x, sim.y, method="polar")
    assert p.status is Status.OK and p.method == "polar"
    assert math.isnan(p.polar_distance_nm)          # the polar curve against itself is not an agreement
    n = ac.auto_contour(sim.x, sim.y)
    assert n.star_shaped
    assert n.polar_distance_nm < ac.POLAR_WARN_NM


def test_density_nodes_lie_on_the_ring():
    sim = V.simulate_ring("circle", 2, background_frac=0.15)
    nodes, w = ac.density_nodes(sim.x, sim.y)
    assert nodes.shape[0] >= ac.MIN_NODES and nodes.shape == (w.size, 2)
    assert np.all(np.diff(w) <= 0)
    assert np.all(np.abs(np.hypot(nodes[:, 0], nodes[:, 1]) - 1300.0) < 60.0)
    empty_nodes, empty_w = ac.density_nodes(np.zeros(0), np.zeros(0))
    assert empty_nodes.shape == (0, 2) and empty_w.size == 0


def test_failed_contour_is_a_complete_autocontour():
    x, y = V.disk_cloud(1)
    c = ac.auto_contour(x, y)
    assert isinstance(c, AutoContour)
    assert c.path is None and math.isnan(c.length_nm)
    assert c.status is Status.CONTOUR_FAILED
    assert ac.failure_reason(ac.auto_contour(*_ring_cloud())) == ""


# ----------------------------------------------------------------------------------------------------
# Review R1: quality gate, support, structured assessment, cross-checks, node cap
# ----------------------------------------------------------------------------------------------------
def test_quality_of_a_correct_curve_is_not_low():
    x, y = _ring_cloud(seed=11)
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = _linear_closed_path(np.column_stack([1200 * np.cos(t), 1200 * np.sin(t)]))
    q = ac.contour_quality(circle, x, y)
    assert not q.low_quality, q.reasons
    assert q.frac_within[100.0] > 0.99 and q.robust_sd_nm < 30.0
    assert q.max_local_offset_nm < ac.QUALITY_MAX_LOCAL_OFFSET_NM
    assert set(q.frac_within) == {50.0, 100.0, 150.0}


def test_quality_flags_a_curve_that_cuts_a_bay():
    sim = V.simulate_ring("bay700", 1, background_frac=0.15)
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    smooth = _linear_closed_path(np.column_stack([1500 * np.cos(t), 1500 * np.sin(t)]))   # ignores the 700 nm bay
    q = ac.contour_quality(smooth, sim.x, sim.y)
    assert q.low_quality
    assert q.max_local_offset_nm > ac.QUALITY_MAX_LOCAL_OFFSET_NM
    assert abs(((q.local_offset_arc_nm / smooth.length_nm) * 2 * np.pi) - np.pi) < 0.6   # at the bay (angle pi)


def test_quality_flags_a_cloud_wider_than_the_band():
    x, y = _ring_cloud(seed=12)
    rng = np.random.default_rng(5)
    r = np.hypot(x, y)
    scale = (1200 + rng.normal(0, 180, r.size)) / r       # radial sd 180 nm
    q = ac.contour_quality(_linear_closed_path(np.column_stack([1200 * np.cos(np.linspace(0, 2 * np.pi, 400, endpoint=False)),
                                                                1200 * np.sin(np.linspace(0, 2 * np.pi, 400, endpoint=False))])),
                           x * scale, y * scale)
    assert q.low_quality
    assert q.frac_within[100.0] < ac.QUALITY_MIN_FRAC_100 or q.robust_sd_nm > ac.QUALITY_MAX_SD_NM


def test_support_marks_a_two_micron_gap_and_nothing_else():
    sim = V.simulate_ring("ellipse", 4, gap_nm=2000.0, background_frac=0.15)
    c = ac.auto_contour(sim.x, sim.y)
    assert c.path is not None
    sup = ac.contour_support(c.path, sim.x, sim.y)
    assert len(sup.arcs_nm) >= 1
    # the gap, plus the stretches between Poisson clusters (4 per um) longer than 400 nm that truly hold no
    # localization (15-20 % of a curve without a gap, see SUPPORT_LEVEL_FRAC)
    assert 1000.0 / c.length_nm <= sup.unsupported_fraction <= (2000.0 + 0.25 * (c.length_nm - 2000.0)) / c.length_nm
    full = V.simulate_ring("ellipse", 4, background_frac=0.15, cluster_rate_per_um=8.0)
    cf = ac.auto_contour(full.x, full.y)
    assert cf.path is not None
    # 8 clusters per um: a Poisson stretch > 400 nm without clusters still happens now and then (one of 480 nm here)
    assert ac.contour_support(cf.path, full.x, full.y).unsupported_fraction < 0.10


def test_assessment_is_structured():
    sim = V.simulate_ring("ellipse", 5, background_frac=0.15)
    det = ac.contour_details(sim.x, sim.y)
    a = det.assessment
    assert a is not None and a.kind == "primary" and a.failure_reason == ""
    assert a.quality is not None and a.support is not None
    xd, yd = V.disk_cloud(2)
    bad = ac.contour_details(xd, yd).assessment
    assert bad is not None and bad.kind == "none" and bad.failure_reason in ac.FAILURE_REASONS
    assert bad.quality is None
    pol = ac.contour_details(sim.x, sim.y, method="polar").assessment
    assert pol is not None and pol.kind == "polar"


def test_not_star_shaped_curve_gets_a_polar_cross_check():
    sim = V.simulate_ring("C", 6, background_frac=0.15)
    c = ac.auto_contour(sim.x, sim.y)
    assert c.status is Status.OK and not c.star_shaped
    if math.isfinite(c.polar_distance_nm) and c.polar_distance_nm > ac.POLAR_WARN_NM:
        assert any("not star-shaped" in w for w in c.warnings)


def test_node_cap(monkeypatch):
    monkeypatch.setattr(ac, "NODE_MAX_COUNT", 12)
    monkeypatch.setattr(ac, "NODE_WARN_COUNT", 10)
    sim = V.simulate_ring("ellipse", 7, background_frac=0.15)
    det = ac.contour_details(sim.x, sim.y)
    assert det.n_nodes_initial == 12
    assert any("strongest" in n for n in det.notes)
    assert any("warning above 10" in n for n in det.notes)


# ----------------------------------------------------------------------------------------------------
# Report v2: persistent offset, the general quality gate (D9), slice misfit (D7), interior trimming (D8)
# ----------------------------------------------------------------------------------------------------
def _circle_path(r=1200.0, n=400):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return _linear_closed_path(np.column_stack([r * np.cos(t), r * np.sin(t)]))


def test_persistent_offset_ignores_a_single_window_spike():
    offs = np.zeros(60)
    offs[10] = 200.0                              # one noisy window
    v, _i = ac.persistent_offset(offs, 3)
    assert v == pytest.approx(0.0)
    offs[30:34] = [120.0, -130.0, 125.0, 140.0]   # a stretch of four windows on one side or the other
    v, i = ac.persistent_offset(offs, 3)
    assert v == pytest.approx(125.0) and 30 <= i <= 31
    offs[31] = np.nan                             # a window without a band breaks the run
    v, _i = ac.persistent_offset(offs, 3)
    assert v == pytest.approx(0.0)
    assert math.isnan(ac.persistent_offset(np.full(5, np.nan), 3)[0])
    wrap = np.zeros(20)
    wrap[[18, 19, 0]] = 90.0                      # the run wraps around the arc origin
    assert ac.persistent_offset(wrap, 3)[0] == pytest.approx(90.0)


def test_quality_gate_is_a_general_rule_on_the_numbers():
    # the numbers of a wide, poorly held cloud (as in the three pilot axons the gate must leave out, D9) and of a
    # well held one: the rule reads only the numbers, so any axon with the same numbers gets the same verdict
    bad = ac.quality_reasons(0.44, 157.0, 40.0, 0.0, 0.10)
    assert len(bad) == 2 and "within +-100 nm" in bad[0] and "robust sd" in bad[1]
    assert ac.quality_reasons(0.80, 60.0, 30.0, 0.0, 0.05) == ()
    assert len(ac.quality_reasons(0.80, 60.0, ac.QUALITY_MAX_LOCAL_OFFSET_NM + 1, 500.0, 0.05)) == 1
    assert len(ac.quality_reasons(0.80, 60.0, 30.0, 0.0, ac.QUALITY_MAX_OFF_BAND_FRAC + 0.01)) == 1
    assert ac.quality_reasons(0.80, 60.0, float("nan"), float("nan"), float("nan")) == ()


def test_wide_cloud_gives_a_contour_marked_not_reliable():
    # radial sd 150 nm: the band of +-100 nm cannot hold the cloud whatever the curve, so every accepted contour
    # must come out not reliable, with the cloud-level reason
    flagged, accepted = 0, 0
    for shape, seed in (("ellipse", 1), ("wavy", 2), ("star", 3)):
        sim = V.simulate_ring(shape, 900 + seed, **V.CONDITIONS["real150"])
        det = ac.contour_details(sim.x, sim.y)
        a = det.assessment
        assert a is not None
        if det.contour.status is not Status.OK:
            assert not a.reliable and a.failure_reason in ac.FAILURE_REASONS
            continue
        assert a.quality is not None
        assert a.reliable is (not a.quality.low_quality)
        accepted += 1
        if not a.reliable:
            flagged += 1
            assert det.contour.warnings[0].startswith(ac.LOW_QUALITY_PREFIX)
            assert a.quality.frac_within[100.0] < ac.QUALITY_MIN_FRAC_100 or a.quality.robust_sd_nm > ac.QUALITY_MAX_SD_NM
    assert accepted >= 1 and flagged == accepted


def _sliced_ring(seed=0, n=9000, r=1500.0, sd=40.0, dr_plus=0.0, shift_minus=(0.0, 0.0)):
    """A ring cut in three slices; slice +1's ring is dr_plus larger, slice -1's ring is shifted laterally."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    lab = rng.integers(-1, 2, n)
    rr = r + rng.normal(0, sd, n) + np.where(lab == 1, dr_plus, 0.0)
    x = rr * np.cos(t) + np.where(lab == -1, shift_minus[0], 0.0)
    y = rr * np.sin(t) + np.where(lab == -1, shift_minus[1], 0.0)
    return x, y, lab


def test_slice_misfit_recovers_a_known_displacement():
    x, y, lab = _sliced_ring(dr_plus=60.0, shift_minus=(80.0, 0.0))
    sm = ac.slice_misfit(_circle_path(1500.0), x, y, lab)
    assert sm.radius_change_nm[2] - sm.radius_change_nm[1] == pytest.approx(60.0, abs=6.0)
    assert sm.lateral_vs_central_nm[0] == pytest.approx(80.0, abs=8.0)
    assert sm.lateral_vs_central_nm[1] < 8.0
    assert sm.rms_displacement_nm[1] == pytest.approx(60.0, abs=6.0)
    assert sm.rms_displacement_nm[0] == pytest.approx(80.0 / math.sqrt(2.0), abs=6.0)
    assert sm.radius_change_per_slice_nm == pytest.approx(30.0, abs=5.0)
    assert sm.low and "z slices" in sm.reason
    assert sm.frac100_all < sm.frac100_central
    assert sum(sm.n_locs) == x.size


def test_slice_misfit_of_slices_that_coincide_is_not_flagged():
    x, y, lab = _sliced_ring(seed=1)
    sm = ac.slice_misfit(_circle_path(1500.0), x, y, lab)
    assert not sm.low and sm.reason == ""
    assert max(sm.rms_displacement_nm) < 8.0
    assert abs(sm.frac100_all - sm.frac100_central) < 0.03


def test_slice_labels_enter_the_quality_gate():
    x, y, lab = _sliced_ring(seed=2, dr_plus=150.0)
    det = ac.contour_details(x, y, slice_label=lab)
    a = det.assessment
    assert a is not None and a.quality is not None and a.quality.slices is not None
    assert a.quality.slices.low and not a.reliable
    assert any("z slices" in r for r in a.quality.reasons)
    row = ac.assessment_row(det)
    assert row["contour_slice_rms_plus_nm"] > ac.SLICE_MAX_RMS_NM
    with pytest.raises(ValueError):
        ac.contour_details(x, y, slice_label=lab[:-1])


def test_interior_exclusion_mask():
    d = np.array([-50.0, 0.0, 249.0, 251.0, 900.0, np.nan])
    np.testing.assert_array_equal(ac.interior_exclusion(d), [False, False, False, True, True, False])
    np.testing.assert_array_equal(ac.interior_exclusion(d, depth_limit_nm=100.0), [False, False, True, True, True, False])


def test_trimming_is_off_by_default_and_an_empty_mask_changes_nothing():
    sim = V.simulate_ring("ellipse", 21, background_frac=0.15)
    a = ac.contour_details(sim.x, sim.y)
    b = ac.contour_details(sim.x, sim.y, exclude=np.zeros(sim.x.size, dtype=bool))
    assert a.contour.path is not None and b.contour.path is not None
    np.testing.assert_array_equal(a.contour.path.points_nm, b.contour.path.points_nm)
    assert a.assessment is not None and a.assessment.n_excluded == 0
    assert inspect.signature(ac.auto_contour).parameters["exclude"].default is None


def test_trimming_leaves_out_the_masked_localizations():
    sim = V.simulate_ring("ellipse", 22, **V.CONDITIONS["real75"])
    mask = sim.kind == V.KIND_INTERIOR
    det = ac.contour_details(sim.x, sim.y, exclude=mask)
    assert det.assessment is not None and det.assessment.n_excluded == int(mask.sum()) > 0
    assert any("interior trimming on" in n for n in det.notes)
    with pytest.raises(ValueError):
        ac.contour_details(sim.x, sim.y, exclude=mask[:-1])


def test_assessment_row_has_stable_keys():
    sim = V.simulate_ring("ellipse", 23, background_frac=0.15)
    ok = ac.assessment_row(ac.contour_details(sim.x, sim.y))
    xd, yd = V.disk_cloud(3)
    bad = ac.assessment_row(ac.contour_details(xd, yd))
    assert list(ok) == list(bad)
    assert ok["contour_kind"] == "primary" and ok["contour_reliable"] is True and ok["contour_failure_reason"] == ""
    assert bad["contour_kind"] == "none" and bad["contour_reliable"] is False
    assert bad["contour_failure_reason"] in ac.FAILURE_REASONS
    assert all(k.startswith("contour_") for k in ok)
