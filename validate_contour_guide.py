# -*- coding: utf-8 -*-
"""
A contour drawn by hand: the path along the membrane, and what it orders.

The contour set by hand is kept as the path a person drew, not as an order
of cluster numbers, because the path still means something after the
clusters change. These checks hold the design to that: the path recovers
the true order of a known outline from shuffled centres; it survives a
change of eps that changes the clusters; it orders the clusters the
discard leaves, so the "_discard" columns compare like with like; and it
is written to the table and to the analysis fingerprint, so two contours
drawn two ways are never taken for one analysis.

Run:  python validate_contour_guide.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.axon_export import SHARED_COLUMNS, analysis_id  # noqa: E402
from tools.column_dictionary import describe  # noqa: E402
from tools.mps_analysis import (  # noqa: E402
    analyze_axon,
    format_guide,
    parse_guide,
    with_every_start,
    without_clusters,
)
from tools.mps_axoplasm import anchored_clusters  # noqa: E402
from tools.mps_geometry import (  # noqa: E402
    GUIDE_FAR_NM,
    check_guide,
    order_along_guide,
    reconstruct_perimeter,
)

PASSED = 0
FAILED = 0


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


def peanut(n: int, a: float = 600.0, b: float = 260.0,
           waist: float = 190.0) -> np.ndarray:
    """Points in order along a closed peanut-shaped outline: concave."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([a * np.cos(t),
                            (b + waist * np.cos(2 * t)) * np.sin(t)])


def tour_length(points: np.ndarray) -> float:
    closed = np.vstack([points, points[:1]])
    return float(np.hypot(*np.diff(closed, axis=0).T).sum())


def ring_axon(seed: int, n: int = 16, radius: float = 800.0,
              per: int = 40, sigma: float = 12.0):
    """Localizations around n clusters on a ring, for analyze_axon."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts, zs = [], []
    for angle in angles:
        c = (radius * math.cos(angle), radius * math.sin(angle))
        pts.append(rng.normal(c, sigma, size=(per, 2)))
        zs.append(rng.normal(0.0, 8.0, size=per))
    xy = np.concatenate(pts)
    return xy[:, 0], xy[:, 1], np.concatenate(zs)


def analyse(x, y, z, **kw):
    kw.setdefault("eps_nm", 60.0)
    kw.setdefault("min_samples", 5)
    return analyze_axon(x, y, z, source_name="synthetic",
                        pixel_size_nm=100.0, pixel_size_source="manual",
                        n_randomizations=0, **kw)


def circle_guide(radius: float, n: int = 12) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def main() -> int:
    print("\n1. THE PATH GIVES THE ORDER")

    def recovers_known_order():
        truth = peanut(40)
        rng = np.random.default_rng(1)
        shuffled = rng.permutation(len(truth))
        # A coarse path: every fourth point of the true outline, pushed
        # 3 % outward -- the kind of thing a person drags into place.
        guide = truth[::4] * 1.03
        order, dist = order_along_guide(truth[shuffled], guide)
        recovered = shuffled[order]
        steps = np.diff(np.concatenate([recovered, recovered[:1]])) % 40
        assert set(steps.tolist()) <= {1} or set(steps.tolist()) <= {39}, \
            steps
        return (f"40 shuffled centres of a concave outline back in order "
                f"from a 10-point path; furthest {dist.max():.0f} nm off it")
    check("the true order of a concave outline, from shuffled centres",
          recovers_known_order)

    def perimeter_is_the_outline():
        truth = peanut(40)
        shuffled = truth[np.random.default_rng(2).permutation(40)]
        result = reconstruct_perimeter(shuffled, guide=truth[::4] * 1.03)
        assert abs(result.perimeter_nm - tour_length(truth)) < 1e-6, \
            (result.perimeter_nm, tour_length(truth))
        assert result.order_source == "set by hand"
        assert result.guide is not None and len(result.guide) == 10
        return (f"{result.perimeter_um:.3f} um, the outline's own "
                f"{tour_length(truth) / 1000:.3f} um; kept as set by hand")
    check("the perimeter along the path is the outline's",
          perimeter_is_the_outline)

    def ties_are_stable():
        pts = np.array([[100.0, 0.0], [100.0, 0.0], [0.0, 100.0],
                        [-100.0, 0.0], [0.0, -100.0]])
        a, _ = order_along_guide(pts, circle_guide(120.0))
        b, _ = order_along_guide(pts, circle_guide(120.0))
        assert np.array_equal(a, b)
        return "two centres on one point of the path come out in one order"
    check("the same input gives the same order", ties_are_stable)

    def far_centres_are_named():
        pts = np.vstack([circle_guide(400.0, 16), [[0.0, 0.0]]])
        result = reconstruct_perimeter(pts, guide=circle_guide(410.0, 12))
        named = [w for w in result.warnings
                 if f"more than {GUIDE_FAR_NM:g} nm from the path" in w]
        assert len(named) == 1 and named[0].startswith("1 of the 17"), \
            result.warnings
        return "a centre at the axon's centre is named as off the path"
    check("a centre the path does not run through is named",
          far_centres_are_named)

    def untouched_changes_nothing():
        # The editor opens on the measured contour, every centre a vertex,
        # so that "Use this contour" without a drag is a no-op. Measured
        # on the 18 real April axons: 0.00 % change in 18 of 18, where
        # the convex hull -- the first version's start -- lengthened every
        # one (median +12 %, up to +49 %).
        x, y, z = ring_axon(11)
        a = analyse(x, y, z)
        same = reconstruct_perimeter(a.centroids, guide=a.perimeter.contour)
        assert abs(same.perimeter_nm - a.perimeter.perimeter_nm) < 1e-6, \
            (same.perimeter_nm, a.perimeter.perimeter_nm)
        return f"{a.perimeter_um:.4f} um both ways"
    check("a path ON the measured contour changes nothing",
          untouched_changes_nothing)

    print("\n2. WHAT IS REFUSED")

    def refuses_crossing():
        bow = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        try:
            check_guide(bow)
        except ValueError as error:
            assert "crosses itself" in str(error), error
            return "a bow tie: crosses itself"
        raise AssertionError("a self-crossing path was accepted")
    check("a path that crosses itself", refuses_crossing)

    def refuses_short_and_nan():
        for bad, why in ((np.zeros((2, 2)), "at least 3"),
                         (np.array([[0, 0], [1, 0], [np.nan, 1.0]]),
                          "finite")):
            try:
                check_guide(bad)
            except ValueError as error:
                assert why in str(error), error
                continue
            raise AssertionError(f"accepted: {bad}")
        return "two points; a NaN"
    check("a path too short to close, and one with a NaN",
          refuses_short_and_nan)

    def refuses_both():
        try:
            reconstruct_perimeter(circle_guide(400.0, 8),
                                  custom_order=np.arange(8),
                                  guide=circle_guide(410.0, 8))
        except ValueError as error:
            return str(error)[:60]
        raise AssertionError("an order and a path were both accepted")
    check("an order and a path at once", refuses_both)

    print("\n3. THE ANALYSIS KEEPS THE PATH")
    x, y, z = ring_axon(3)
    guide = circle_guide(820.0, 12)
    drawn = analyse(x, y, z, contour_guide=guide)
    automatic = analyse(x, y, z)

    def carried():
        assert drawn.contour_order_source == "set by hand"
        assert drawn.contour_guide is not None
        assert np.allclose(drawn.contour_guide, guide)
        assert automatic.contour_guide is None
        row = dict((r[0], r) for r in drawn.summary_rows())
        assert "path drawn by hand" in row["Perimeter"][3], row["Perimeter"]
        return f"{drawn.n_clusters_kept} clusters, {row['Perimeter'][3]}"
    check("it is on the analysis and said on screen", carried)

    def untouched_by_every_start():
        again = with_every_start(drawn)
        assert again.perimeter_um == drawn.perimeter_um
        assert again.contour_guide is not None
        return "with_every_start returns it as it was"
    check("nothing rebuilds it", untouched_by_every_start)

    def survives_new_clusters():
        # Two clusters 70 nm apart along the ring: one cluster at eps 60,
        # two at eps 20 (measured: 16 and 17). An order of the clusters
        # found at one eps means nothing at the other -- there is not even
        # the same number of them -- while the path still orders whichever
        # clusters there are.
        rng = np.random.default_rng(8)
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        pts = [rng.normal((800 * math.cos(a), 800 * math.sin(a)), 5.0,
                          size=(40, 2)) for a in angles]
        pts.append(rng.normal((800.0, 70.0), 5.0, size=(40, 2)))
        xy = np.concatenate(pts)
        zz = rng.normal(0.0, 8.0, size=len(xy))
        wide = analyse(xy[:, 0], xy[:, 1], zz, eps_nm=60.0,
                       contour_guide=guide)
        tight = analyse(xy[:, 0], xy[:, 1], zz, eps_nm=20.0,
                        contour_guide=wide.contour_guide)
        assert wide.n_clusters_kept != tight.n_clusters_kept,             (wide.n_clusters_kept, tight.n_clusters_kept)
        for a in (wide, tight):
            assert a.contour_order_source == "set by hand"
            want, _ = order_along_guide(a.centroids, guide)
            assert np.array_equal(a.perimeter.order, want)
        return (f"eps 60 -> 20 nm: {wide.n_clusters_kept} -> "
                f"{tight.n_clusters_kept} clusters, both along the path")
    check("a change of eps keeps it, ordering the new clusters",
          survives_new_clusters)

    def discard_follows_it():
        flags = np.zeros(drawn.n_clusters_kept, dtype=bool)
        flags[[1, 5, 9]] = True
        rest = without_clusters(drawn, flags, margin_nm=150.0)
        assert rest.contour_order_source == "set by hand", \
            rest.contour_order_source
        want, _ = order_along_guide(drawn.centroids[~flags], guide)
        assert np.array_equal(rest.perimeter.order, want)
        assert np.allclose(rest.contour_guide, guide)
        return (f"3 of {drawn.n_clusters_kept} discarded: the other "
                f"{rest.n_clusters_kept} joined along the same path")
    check("the discard is joined along the same path", discard_follows_it)

    def panel_follows_it():
        c = drawn.centroids
        n = len(c)
        far = np.full(n, 0.0)
        far[[2, 7]] = 500.0          # two clusters deep in both images

        class Mask:
            def distance_at(self, col, row):
                return far.copy()

        anchored = anchored_clusters(
            c, Mask(), np.zeros(n), np.zeros(n), Mask(), np.zeros(n),
            np.zeros(n), margin_nm=100.0, contour_all=drawn.perimeter)
        kept = anchored.contour_anchored
        assert kept is not None and kept.order_source == "set by hand"
        want, _ = order_along_guide(c[~anchored.discarded], guide)
        assert np.array_equal(kept.order, want)
        return (f"the axoplasm panel's contour of the {n - 2} kept clusters "
                f"follows the path")
    check("and so is the axoplasm panel's", panel_follows_it)

    print("\n4. THE TABLE AND THE FINGERPRINT")

    def written():
        d = drawn.export_dict()
        assert "contour_guide_nm" in SHARED_COLUMNS
        assert describe("contour_guide_nm") is not None
        back = parse_guide(d["contour_guide_nm"])
        assert back is not None and np.allclose(back, guide, atol=0.05)
        assert automatic.export_dict()["contour_guide_nm"] is None
        rebuilt = analyse(x, y, z, contour_guide=back)
        assert rebuilt.perimeter_um == drawn.perimeter_um
        return (f"{len(d['contour_guide_nm'])} characters; read back, it "
                f"rebuilds the same {drawn.perimeter_um:.3f} um")
    check("written to the table, and rebuildable from it", written)

    def fingerprint():
        other = analyse(x, y, z, contour_guide=circle_guide(840.0, 12))
        same = analyse(x, y, z, contour_guide=guide)
        ids = {analysis_id(automatic), analysis_id(drawn),
               analysis_id(other)}
        assert len(ids) == 3, ids
        assert analysis_id(same) == analysis_id(drawn)
        return "automatic, one path, another path: three ids; same path: same id"
    check("two contours drawn two ways are two analyses", fingerprint)

    def format_round_trip():
        g = np.array([[1.04, -2.26], [3.0, 4.0], [-5.55, 6.0]])
        assert format_guide(g) == "1.0,-2.3;3.0,4.0;-5.5,6.0" or \
            format_guide(g) == "1.0,-2.3;3.0,4.0;-5.6,6.0", format_guide(g)
        assert parse_guide("") is None and parse_guide(None) is None
        assert format_guide(None) is None
        return format_guide(g)
    check("the cell format", format_round_trip)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
