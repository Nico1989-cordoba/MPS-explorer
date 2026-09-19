# -*- coding: utf-8 -*-
"""
Validation harness for tools/mps_geometry.py against real axon data.

Chains the full pipeline built so far:
  Z-GMM slab (step 2) -> DBSCAN -> automatic bad-cluster removal (step 1)
  -> perimeter / cluster area / effective radius (step 3)

and compares against the paper's reference values.

Run:  python validate_geometry.py
"""

from __future__ import annotations

import os
import sys

import h5py as h5
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.cluster_quality import (  # noqa: E402
    PolygonROI, identify_bad_clusters, good_cluster_centroids,
)
from tools.mps_periodicity import fit_z_periodicity, select_mps_slab  # noqa: E402
from tools.mps_geometry import (  # noqa: E402
    reconstruct_perimeter, compute_cluster_areas, contour_health,
    PAPER_MEDIAN_CLUSTER_AREA_NM2, PAPER_MEDIAN_R_EFF_NM,
    DEEP_VERTEX_FRACTION, MAX_OVER_MEDIAN_LIMIT,
)

PXSIZE_NM = 133.0
EPS_NM, MIN_SAMPLES = 25.0, 10
EXAMPLE_DIR = "example data"
FILES = [
    "ROI6_B2spectrin_unified_locs_rcc_apicked3_filter.hdf5",
    "ROI6_B2spectrin_unified_locs_rcc_apicked6_filter.hdf5",
    "ROI6_B2spectrin_unified_locs_rcc_apicked_4_filter.hdf5",
    "ROI7_B2spectrin_unified_locs_rcc_apicked1_filter.hdf5",
    "ROI7_B2spectrin_unified_locs_rcc_apicked3_filter.hdf5",
]


def run_one(filename: str):
    with h5.File(os.path.join(EXAMPLE_DIR, filename), "r") as f:
        ds = f["locs"]
        x = np.asarray(ds["x"], float) * PXSIZE_NM
        y = np.asarray(ds["y"], float) * PXSIZE_NM
        z = np.asarray(ds["z"], float)

    print(f"\n{'=' * 78}\n{filename}\n{'=' * 78}")

    per = fit_z_periodicity(z)
    slab = select_mps_slab(z, per.main_peak_nm)
    xs, ys = x[slab], y[slab]

    labels = DBSCAN(eps=EPS_NM, min_samples=MIN_SAMPLES).fit(
        np.column_stack([xs, ys])).labels_
    n_raw = len(set(labels.tolist()) - {-1})
    if n_raw < 3:
        print(f"  only {n_raw} clusters - skipping (cannot close a contour)")
        return None

    hull = ConvexHull(np.column_stack([xs, ys]))
    roi = PolygonROI(vertices=np.column_stack([xs, ys])[hull.vertices])
    rep = identify_bad_clusters(xs, ys, labels, roi, edge_margin_nm=EPS_NM)
    cms = good_cluster_centroids(xs, ys, labels, rep.bad_labels)

    print(f"  clusters raw / kept  : {n_raw} / {len(cms)}")

    # --- perimeter -----------------------------------------------------
    pr = reconstruct_perimeter(cms)
    print(f"  perimeter            : {pr.perimeter_nm:,.0f} nm "
          f"({pr.perimeter_um:.2f} um)")
    print(f"  clusters per um      : {pr.clusters_per_um:.2f}"
          f"   [paper slope: 4.08]")
    exp_n = pr.expected_n_clusters_from_paper()
    print(f"  N predicted by paper : {exp_n:.1f}   vs measured {pr.n_clusters}"
          f"   (diff {pr.n_clusters - exp_n:+.1f})")
    print(f"  2-opt improvements   : {pr.n_2opt_improvements}"
          f"   self-intersections {pr.self_intersections_before} -> "
          f"{pr.self_intersections_after}")

    # --- areas ---------------------------------------------------------
    ar = compute_cluster_areas(xs, ys, labels, exclude_labels=rep.bad_labels)
    a, r = ar.areas_nm2, ar.r_eff_nm
    print(f"  cluster area   nm^2  : median {ar.median_area_nm2:,.0f}"
          f"   [paper: {PAPER_MEDIAN_CLUSTER_AREA_NM2:,.0f}]")
    print(f"      quartiles        : Q1 {np.percentile(a,25):,.0f} | "
          f"Q3 {np.percentile(a,75):,.0f} | max {a.max():,.0f}")
    print(f"  r_eff          nm    : median {ar.median_r_eff_nm:.1f}"
          f"   [paper: {PAPER_MEDIAN_R_EFF_NM:.0f}]")
    print(f"      quartiles        : Q1 {np.percentile(r,25):.1f} | "
          f"Q3 {np.percentile(r,75):.1f} | max {r.max():.1f}")
    print(f"  locs per cluster     : median {np.median(ar.n_locs):.0f} "
          f"(min {ar.n_locs.min()}, max {ar.n_locs.max()})")

    for w in pr.warnings + ar.warnings:
        print(f"  ! {w}")

    return dict(areas=a, r_eff=r, perim_um=pr.perimeter_um, n=pr.n_clusters)


# ============================================================================
# Contour health checks
# ============================================================================
#
# The failure these detect is silent: with vertices from inside the axon,
# or with a chord closing a gap in the ring, the perimeter comes out wrong
# and the only validity check the module had -- the self-intersection
# count -- stays at zero. So the checks below are run against rings whose
# true perimeter is known, and they have to do two things: fire on the
# broken ones, and stay quiet on every healthy one.

PASSED = 0
FAILED = 0
RING_RADIUS_NM = 1600.0


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - a harness
        FAILED += 1
        print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def make_ring(rng, k=70, gap_deg=0.0, n_interior=0, scatter_nm=40.0,
              even=False, radius=RING_RADIUS_NM):
    """Cluster centres on a ring, optionally gapped or contaminated."""
    span = 360.0 - gap_deg
    ang = (np.linspace(0.0, span, k, endpoint=False) if even
           else np.sort(rng.uniform(0.0, span, k)))
    ang = np.deg2rad(ang)
    r = radius + rng.normal(0.0, scatter_nm, k)
    pts = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
    if n_interior:
        ia = rng.uniform(0.0, 2 * np.pi, n_interior)
        # Well inside: never within the scatter of the ring itself.
        ir = rng.uniform(0.0, radius - 400.0, n_interior)
        pts = np.vstack([pts,
                         np.column_stack([ir * np.cos(ia), ir * np.sin(ia)])])
    return pts


def contour_health_checks() -> None:
    print(f"\n{'=' * 78}\nCONTOUR HEALTH\n{'=' * 78}")

    def square_exactly_its_own_hull():
        h = contour_health(np.array([[0.0, 0.0], [1000.0, 0.0],
                                     [1000.0, 1000.0], [0.0, 1000.0]]))
        assert abs(h.tour_over_hull - 1.0) < 1e-12, h.tour_over_hull
        assert h.n_deep_vertices == 0 and h.max_depth_nm == 0.0
        assert not h.warnings
        return f"ratio {h.tour_over_hull:.12f}"

    def depth_is_a_real_distance():
        # A square of side 1000 nm with one point at its centre: the centre
        # is 500 nm from the nearest side, whatever order they are visited.
        pts = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0],
                        [0.0, 1000.0], [500.0, 500.0]])
        h = contour_health(pts)
        assert abs(h.max_depth_nm - 500.0) < 1e-9, h.max_depth_nm
        return f"{h.max_depth_nm:.1f} nm"

    def a_non_finite_centre_is_reported():
        # NaN compares false against every limit, so a contour carrying one
        # would otherwise pass every check while its perimeter is NaN.
        rng = np.random.default_rng(31)
        pts = make_ring(rng, k=40)
        pts[7, 1] = np.nan
        res = reconstruct_perimeter(pts)
        assert np.isnan(res.perimeter_nm), res.perimeter_nm
        assert res.health is None, "a NaN contour must not get a clean bill"
        assert any("not a finite coordinate" in w for w in res.warnings), (
            res.warnings)
        return f"perimeter NaN, {len(res.warnings)} warning(s), no health"

    def ratios_do_not_depend_on_scale():
        rng = np.random.default_rng(1)
        pts = make_ring(rng, k=60, n_interior=4)
        one = contour_health(reconstruct_perimeter(pts).contour)
        ten = contour_health(reconstruct_perimeter(pts * 10.0).contour)
        assert abs(one.tour_over_hull - ten.tour_over_hull) < 1e-9
        assert abs(one.max_over_median - ten.max_over_median) < 1e-9
        assert abs(ten.max_depth_nm - 10.0 * one.max_depth_nm) < 1e-6
        return f"ratio {one.tour_over_hull:.3f} either way"

    def degenerate_inputs_give_no_health():
        line = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        assert contour_health(line) is None
        assert contour_health(np.zeros((2, 2))) is None
        assert contour_health(np.zeros((0, 2))) is None
        # And the perimeter itself still comes back, without health.
        res = reconstruct_perimeter(line)
        assert res.health is None and res.perimeter_nm > 0
        return "collinear and too-few both return None"

    def healthy_rings_never_fire():
        rng = np.random.default_rng(7)
        fired, total = [], 0
        for k in (30, 70, 140):
            for scatter in (20.0, 40.0, 80.0):
                for _ in range(8):
                    total += 1
                    h = contour_health(reconstruct_perimeter(
                        make_ring(rng, k=k, scatter_nm=scatter)).contour)
                    if h.warnings:
                        fired.append((k, scatter, h.tour_over_hull,
                                      h.max_over_median))
        assert not fired, f"{len(fired)}/{total} false alarms: {fired[:3]}"
        return f"0 false alarms in {total} rings, K 30-140"

    def a_gap_is_not_mistaken_for_an_interior():
        rng = np.random.default_rng(11)
        rows = []
        for gap in (40, 60, 90, 120, 160):
            chords, deep = [], []
            for _ in range(10):
                h = contour_health(reconstruct_perimeter(
                    make_ring(rng, gap_deg=gap)).contour)
                chords.append(h.bridges_a_gap)
                deep.append(h.n_deep_vertices)
            assert max(deep) == 0, (
                f"a gap of {gap} deg was read as {max(deep)} interior "
                f"centre(s): the two failures are not separated")
            rows.append((gap, float(np.mean(chords))))
        # The chord check has to see the wide ones.
        assert rows[-1][1] == 1.0 and rows[-2][1] == 1.0, rows
        return "; ".join(f"{g} deg: chord in {f:.0%}" for g, f in rows)

    def a_concave_axon_stays_quiet():
        # An oval squeezed to a waist a third of its width, which is
        # further than any sciatic cross-section in this dataset goes.
        rng = np.random.default_rng(23)
        worst, fired = 0.0, 0
        for _ in range(20):
            ang = np.sort(rng.uniform(0.0, 2 * np.pi, 70))
            r = RING_RADIUS_NM * (1.0 + 0.5 * np.cos(2 * ang))
            r = r + rng.normal(0.0, 40.0, 70)
            pts = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
            h = contour_health(reconstruct_perimeter(pts).contour)
            worst = max(worst, h.max_depth_nm / h.hull_radius_nm)
            fired += h.has_interior_vertices
        assert fired == 0, (
            f"{fired}/20 peanuts flagged as having an interior")
        assert worst < DEEP_VERTEX_FRACTION - 0.05, (
            f"only {DEEP_VERTEX_FRACTION - worst:.3f} of margin left on a "
            f"concave cross-section; the limit is too tight")
        return (f"deepest vertex {worst:.2f} of the radius, limit "
                f"{DEEP_VERTEX_FRACTION:.2f}")

    def a_dense_noisy_ring_is_not_called_broken():
        # The case that made tour/hull unusable as a threshold: a correct
        # contour of a ring with many scattered centres is legitimately
        # much longer than its hull.
        rng = np.random.default_rng(29)
        ratios, fired = [], 0
        for _ in range(10):
            h = contour_health(reconstruct_perimeter(
                make_ring(rng, k=140, scatter_nm=80.0)).contour)
            ratios.append(h.tour_over_hull)
            fired += bool(h.warnings)
        assert fired == 0, f"{fired}/10 healthy dense rings flagged"
        assert max(ratios) > 1.4, (
            f"this check is pointless unless the ratio really is high: "
            f"{max(ratios):.2f}")
        return (f"tour/hull up to {max(ratios):.2f} and no warning")

    def interior_vertices_fire():
        rng = np.random.default_rng(13)
        rows = []
        for n in (3, 5, 10, 20):
            fired, found = [], []
            for _ in range(10):
                h = contour_health(reconstruct_perimeter(
                    make_ring(rng, n_interior=n)).contour)
                fired.append(h.has_interior_vertices)
                found.append(h.n_deep_vertices)
            rows.append((n, float(np.mean(fired)), float(np.mean(found))))
        for n, rate, found in rows:
            assert rate == 1.0, (
                f"{n} interior clusters flagged only {rate:.0%}")
            assert found >= n * 0.8, f"{n} planted, {found:.1f} found"
        return "; ".join(f"{n}: {r:.0%} flagged, {f:.1f} found"
                         for n, r, f in rows)

    def the_warning_reaches_the_analysis():
        rng = np.random.default_rng(17)
        res = reconstruct_perimeter(make_ring(rng, n_interior=10))
        assert res.self_intersections_after == 0, (
            "this ring was chosen because the old check cannot see it")
        assert any("convex hull" in w for w in res.warnings), res.warnings
        assert res.health.n_deep_vertices > 0
        return (f"self-intersections 0 but ratio "
                f"{res.health.tour_over_hull:.2f}, {len(res.warnings)} "
                f"warning(s)")

    def a_manual_order_is_checked_too():
        rng = np.random.default_rng(19)
        pts = make_ring(rng, k=40, n_interior=8)
        order = reconstruct_perimeter(pts).order
        manual = reconstruct_perimeter(pts, custom_order=order)
        assert manual.health is not None
        assert any("convex hull" in w for w in manual.warnings), \
            manual.warnings
        return f"{len(manual.warnings)} warning(s) on a hand-given order"

    check("a square is exactly its own convex hull",
          square_exactly_its_own_hull)
    check("depth inside the hull is a real distance", depth_is_a_real_distance)
    check("the ratios do not change with scale", ratios_do_not_depend_on_scale)
    check("degenerate contours return no health",
          degenerate_inputs_give_no_health)
    check("a non-finite centre is caught, not swallowed",
          a_non_finite_centre_is_reported)
    check("healthy rings never fire a warning", healthy_rings_never_fire)
    check("a gap is not read as an interior",
          a_gap_is_not_mistaken_for_an_interior)
    check("a concave cross-section stays quiet", a_concave_axon_stays_quiet)
    check("a dense, noisy ring is not called broken",
          a_dense_noisy_ring_is_not_called_broken)
    check("interior vertices always fire", interior_vertices_fire)
    check("the warning reaches PerimeterResult",
          the_warning_reaches_the_analysis)
    check("a manual contour order is checked too",
          a_manual_order_is_checked_too)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    all_areas, all_r, pts = [], [], []
    for fn in FILES:
        if os.path.exists(os.path.join(EXAMPLE_DIR, fn)):
            res = run_one(fn)
            if res:
                all_areas.append(res["areas"])
                all_r.append(res["r_eff"])
                pts.append((res["perim_um"], res["n"]))

    if all_areas:
        A = np.concatenate(all_areas)
        R = np.concatenate(all_r)
        print(f"\n{'=' * 78}\nPOOLED across example axons\n{'=' * 78}")
        print(f"  clusters pooled      : {A.size}")
        print(f"  area median          : {np.median(A):,.0f} nm^2"
              f"   [paper 1,965]")
        print(f"  r_eff median         : {np.median(R):.1f} nm   [paper ~25]")

        if len(pts) >= 3:
            P = np.array(pts)
            slope, intercept = np.polyfit(P[:, 0], P[:, 1], 1)
            rr = np.corrcoef(P[:, 0], P[:, 1])[0, 1]
            print(f"  N vs perimeter fit   : y = {slope:.2f}x {intercept:+.2f}"
                  f"   r = {rr:.2f}")
            print(f"    paper              : y = 4.08x -12.62   r = 0.81")
            print(f"    (only {len(pts)} axons here -- the paper used 34)")

    contour_health_checks()
    print(f"\n{'=' * 78}\n{PASSED} passed, {FAILED} failed\n{'=' * 78}")
    sys.exit(1 if FAILED else 0)
