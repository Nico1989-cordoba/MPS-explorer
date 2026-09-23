# -*- coding: utf-8 -*-
"""
Every cross-channel number, against a synthetic pair whose answer is known.

The two-channel module measures things no published method measures --
the shared perimeter occupancy, the signed radial offset, the heterotypic
distances and their nulls, the angular registry, the axial phase -- and
until now only the axial phase and the registration had ever been checked
against a known answer (validate_registration.py, section 6). This is the
rest, and it has to exist BEFORE good two-colour data does: once real
data arrives there is nothing to compare it with.

Two rules, both for the same reason.

  The right answer is computed on the TRUTH -- the true cluster centres
  the generator hands back -- and with this file's own code, never with
  the functions being validated. Checking a function against itself
  proves only that it agrees with itself.

  Where the model allows it, the answer is written down analytically and
  checked with no jitter at all: the partner interleaved at half the
  spacing sits exactly R(1 - cos(pi/n)) outside the spectrin polygon, and
  a partner that IS spectrin shares all of it. Those are the sharpest
  tests there are, because nothing about them is statistical.

Run:  python validate_crosschannel.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_crosschannel import (  # noqa: E402
    CHANNEL2_PARAMETERS_REQUIRED,
    axial_phase,
    cross_channel_transverse,
    export_cross_channel,
    locs_per_cell,
)
from tools.mps_registration import Registration  # noqa: E402
from tools.mps_synthetic_pair import SyntheticChannels, ring_pair  # noqa: E402

PASSED = 0
FAILED = 0

SLAB = (-90.0, 90.0)
N = 16                      # clusters per ring
R = 400.0                   # ring radius, nm
HALF_SPACING = math.pi / N  # interleaves the two rings


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


def compare(pair: SyntheticChannels, *, slab=SLAB, n_null: int = 200,
            registration=None, **kwargs):
    """The pipeline's answer, with channel 2 clustered like channel 1.

    The same eps and min_samples on purpose: every channel here is drawn
    from the same cluster model, so equal parameters are the right choice
    and the comparison measures the arithmetic, not a clustering mismatch.
    """
    return cross_channel_transverse(
        pair.x_a, pair.y_a, pair.z_a, pair.x_b, pair.y_b, pair.z_b,
        slab=slab, pixel_size_source="not_applicable",
        eps_nm=25.0, min_samples=10,
        analyze_kwargs_b=dict(eps_nm=25.0, min_samples=10),
        n_null=n_null, registration=registration, **kwargs)


# ---------------------------------------------------------------------------
# The truth, computed here and nowhere else
# ---------------------------------------------------------------------------

def true_nn(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Nearest distance from each src point to dst, by brute force."""
    d = np.sqrt(((src[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2))
    return d.min(axis=1)


def true_self_nn(pts: np.ndarray) -> np.ndarray:
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def polygon_boundary(poly: np.ndarray, step: float = 0.05) -> np.ndarray:
    """The closed polygon sampled every ``step`` nm along its edges."""
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(2, int(math.ceil(np.hypot(*(b - a)) / step)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out.append(a + t * (b - a))
    return np.concatenate(out)


def winding_inside(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Inside by winding number: a different test from the code's rays."""
    total = np.zeros(len(points))
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        va = a[None, :] - points
        vb = b[None, :] - points
        cross = va[:, 0] * vb[:, 1] - va[:, 1] * vb[:, 0]
        dot = (va * vb).sum(axis=1)
        total += np.arctan2(cross, dot)
    return np.abs(total) > math.pi


def true_signed_offsets(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Signed distance to the polygon, inside negative -- independently."""
    boundary = polygon_boundary(poly)
    d = true_nn(points, boundary)
    return np.where(winding_inside(points, poly), -d, d)


def shoelace_centroid(poly: np.ndarray) -> Tuple[float, float]:
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    return (float(((x + x1) * cross).sum() / (6 * area)),
            float(((y + y1) * cross).sum() / (6 * area)))


def perimeter(poly: np.ndarray) -> float:
    return float(np.hypot(*(np.roll(poly, -1, axis=0) - poly).T).sum())


def brute_locs_per_cell(x: np.ndarray, y: np.ndarray, r: float) -> float:
    pts = np.column_stack([x, y])
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    return float(np.median((d2 <= r * r).sum(axis=1)))


def angle_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute angular distance from each angle in a to the nearest in b."""
    d = np.abs(np.mod(a[:, None] - b[None, :] + math.pi, 2 * math.pi)
               - math.pi)
    return d.min(axis=1)


def in_slab(pair: SyntheticChannels, which: str) -> np.ndarray:
    z = pair.z_a if which == "a" else pair.z_b
    return (z >= SLAB[0]) & (z <= SLAB[1])


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

IDENTITY = ring_pair(1, b_is_a=True)
INTERLEAVED_EXACT = ring_pair(6, angle_offset_b=HALF_SPACING,
                              angular_jitter=0.0)
INSIDE_EXACT = ring_pair(5, radius_b_nm=300.0, angular_jitter=0.0)
COLOCATED = ring_pair(4)
INTERLEAVED = ring_pair(3, angle_offset_b=HALF_SPACING)
OVERSAMPLED = ring_pair(10, n_per_cluster_b=600)


def half_of(pair: SyntheticChannels) -> SyntheticChannels:
    """Channel 2 = channel 1's own localizations, every other cluster.

    Built from the SAME localizations, so its clusters are exactly
    channel 1's even clusters and every covered point of channel 2 is a
    covered point of channel 1: shared_of_b has to be exactly 1.
    """
    pts = np.column_stack([pair.x_a, pair.y_a])
    centres = pair.truth_a.centres_nm
    nearest = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(
        axis=2).argmin(axis=1)
    keep = nearest % 2 == 0
    from dataclasses import replace
    truth_b = replace(pair.truth_a, centres_nm=centres[::2],
                      angles=pair.truth_a.angles[::2])
    return replace(pair, x_b=pair.x_a[keep], y_b=pair.y_a[keep],
                   z_b=pair.z_a[keep], truth_b=truth_b)


HALF = half_of(ring_pair(2))


def main() -> int:
    runs = {}

    def run(key, pair, **kw):
        if key not in runs:
            runs[key] = compare(pair, **kw)
        return runs[key]

    # =====================================================================
    print("\n1. A PARTNER THAT IS SPECTRIN  (every answer exact)")

    def identity_counts():
        t = run("identity", IDENTITY)
        assert t.n_clusters_a == N and t.n_clusters_b == N, \
            (t.n_clusters_a, t.n_clusters_b)
        return f"{t.n_clusters_a} and {t.n_clusters_b} clusters"
    check("both channels find all 16 clusters", identity_counts)

    def identity_distance():
        t = run("identity", IDENTITY)
        assert t.median_hetero_nn_a_to_b == 0.0, t.median_hetero_nn_a_to_b
        assert t.median_hetero_nn_b_to_a == 0.0, t.median_hetero_nn_b_to_a
        return "0 nm both ways"
    check("the heterotypic distance is exactly zero", identity_distance)

    def identity_shared():
        s = run("identity", IDENTITY).shared
        assert s is not None and s.measured, None if s is None else s.reason
        assert s.shared_of_a == 1.0, s.shared_of_a
        assert s.shared_of_b == 1.0, s.shared_of_b
        assert s.jaccard == 1.0, s.jaccard
        assert s.n_clusters_b_off_contour == 0
        return "shared_of_a = shared_of_b = Jaccard = 1"
    check("the whole ring is shared, exactly", identity_shared)

    def identity_null():
        s = run("identity", IDENTITY).shared
        assert s.p_null_at_least_measured is not None
        assert s.p_null_at_least_measured < 0.01, s.p_null_at_least_measured
        assert s.n_null == 200, s.n_null
        lo, hi = s.null_ci_shared_of_a
        assert 0.0 <= lo <= s.null_median_shared_of_a <= hi < 1.0, \
            (lo, s.null_median_shared_of_a, hi)
        return (f"chance reaches it in {100 * s.p_null_at_least_measured:.1f}"
                f" % of 200 draws; null median "
                f"{s.null_median_shared_of_a:.3f}")
    check("and chance does not reach it", identity_null)

    def identity_radial():
        r = run("identity", IDENTITY).radial
        assert r is not None and r.median_nm is not None, r
        assert abs(r.median_nm) < 1e-6, r.median_nm
        return f"{r.median_nm:+.2e} nm"
    check("a cluster on the contour sits at zero offset", identity_radial)

    def identity_angles():
        t = run("identity", IDENTITY)
        assert t.angular_nn_median_deg == 0.0, t.angular_nn_median_deg
        assert abs(t.median_radius_a_nm - t.median_radius_b_nm) < 1e-9
        return (f"angular offset {t.angular_nn_median_deg} deg; radii "
                f"{t.median_radius_a_nm:.1f} = {t.median_radius_b_nm:.1f}")
    check("no angular offset, the same radius", identity_angles)

    def identity_density():
        t = run("identity", IDENTITY)
        assert t.locs_per_cell_a == t.locs_per_cell_b, \
            (t.locs_per_cell_a, t.locs_per_cell_b)
        assert not any("different regimes" in w for w in t.warnings)
        return f"{t.locs_per_cell_a:.0f} per cell in both, no warning"
    check("the same density, and no regime warning", identity_density)

    # =====================================================================
    print("\n2. HALF THE POSITIONS, SAME LOCALIZATIONS  (exact containment)")

    def half_shared():
        t = run("half", HALF)
        s = t.shared
        assert t.n_clusters_b == N // 2, t.n_clusters_b
        assert s.shared_of_b == 1.0, s.shared_of_b
        assert 0.40 <= s.shared_of_a <= 0.60, s.shared_of_a
        return (f"shared_of_b = {s.shared_of_b}, shared_of_a = "
                f"{s.shared_of_a:.3f} with 8 of 16 clusters")
    check("everything channel 2 covers, channel 1 covers", half_shared)

    def half_distance():
        t = run("half", HALF)
        assert t.median_hetero_nn_b_to_a == 0.0, t.median_hetero_nn_b_to_a
        want = float(np.median(true_nn(HALF.truth_a.centres_nm,
                                       HALF.truth_b.centres_nm)))
        got = t.median_hetero_nn_a_to_b
        assert abs(got - want) < 3.0, (got, want)
        return f"b->a 0 nm; a->b {got:.1f} nm (truth {want:.1f})"
    check("and every channel-2 cluster is a channel-1 cluster", half_distance)

    # =====================================================================
    print("\n3. INTERLEAVED AT HALF THE SPACING, NO JITTER  (analytic)")

    def interleaved_radial_exact():
        r = run("interleaved_exact", INTERLEAVED_EXACT).radial
        want = R * (1 - math.cos(math.pi / N))
        assert abs(r.median_nm - want) < 1.5, (r.median_nm, want)
        assert r.n_outside == N and r.n_inside == 0, (r.n_outside, r.n_inside)
        return (f"{r.median_nm:+.2f} nm; R(1 - cos(pi/16)) = {want:+.2f} nm; "
                f"all {r.n_outside} outside")
    check("channel 2 sits just outside the chord, by the right amount",
          interleaved_radial_exact)

    def interleaved_distance_exact():
        t = run("interleaved_exact", INTERLEAVED_EXACT)
        want = 2 * R * math.sin(math.pi / (2 * N))
        assert abs(t.median_hetero_nn_a_to_b - want) < 2.0, \
            (t.median_hetero_nn_a_to_b, want)
        return (f"{t.median_hetero_nn_a_to_b:.2f} nm; 2R sin(pi/32) = "
                f"{want:.2f} nm")
    check("the heterotypic distance is the half-spacing chord",
          interleaved_distance_exact)

    def interleaved_angles_exact():
        t = run("interleaved_exact", INTERLEAVED_EXACT)
        want = 180.0 / N
        assert abs(t.angular_nn_median_deg - want) < 0.3, \
            (t.angular_nn_median_deg, want)
        assert abs(t.rotation_fraction_of_spacing - 0.5) < 0.05, \
            t.rotation_fraction_of_spacing
        return (f"offset {t.angular_nn_median_deg:.2f} deg (true {want:.2f});"
                f" best rotation {t.rotation_fraction_of_spacing:.2f} of the "
                f"spacing")
    check("the angular offset is half a spacing", interleaved_angles_exact)

    def interleaved_centre_exact():
        c = run("interleaved_exact", INTERLEAVED_EXACT).analysis_a.centre
        dx = c.x_nm - INTERLEAVED_EXACT.centre_nm[0]
        dy = c.y_nm - INTERLEAVED_EXACT.centre_nm[1]
        assert math.hypot(dx, dy) < 1.5, (dx, dy)
        return f"{math.hypot(dx, dy):.2f} nm from the true centre"
    check("the contour centre is the true centre", interleaved_centre_exact)

    def interleaved_shared_exact():
        s = run("interleaved_exact", INTERLEAVED_EXACT).shared
        assert s.shared_of_a < 0.02, s.shared_of_a
        return f"shared_of_a = {s.shared_of_a:.4f}"
    check("two interleaved rings of small clusters share nothing",
          interleaved_shared_exact)

    # =====================================================================
    print("\n4. A PARTNER 100 nm FURTHER IN, NO JITTER  (analytic)")

    def inside_radial_exact():
        r = run("inside_exact", INSIDE_EXACT).radial
        want = -(R - 300.0) * math.cos(math.pi / N)
        assert abs(r.median_nm - want) < 2.0, (r.median_nm, want)
        assert r.n_inside == N, r.n_inside
        return (f"{r.median_nm:+.2f} nm; -(R - r) cos(pi/16) = {want:+.2f} "
                f"nm; all {r.n_inside} inside")
    check("the offset is negative, and the right size", inside_radial_exact)

    def inside_radii_exact():
        t = run("inside_exact", INSIDE_EXACT)
        assert abs(t.median_radius_a_nm - R) < 2.0, t.median_radius_a_nm
        assert abs(t.median_radius_b_nm - 300.0) < 2.0, t.median_radius_b_nm
        return (f"{t.median_radius_a_nm:.1f} and {t.median_radius_b_nm:.1f} "
                f"nm (true 400 and 300)")
    check("each channel's radius about the contour centre", inside_radii_exact)

    def inside_shares_nothing():
        s = run("inside_exact", INSIDE_EXACT).shared
        assert s.shared_of_a == 0.0, s.shared_of_a
        assert s.n_clusters_b_off_contour == N, s.n_clusters_b_off_contour
        assert any("reach no point" in w for w in s.warnings)
        return f"0 shared; {s.n_clusters_b_off_contour} of 16 off the contour"
    check("a ring 100 nm inside shares nothing, and says why",
          inside_shares_nothing)

    def reference_scatter():
        r = run("inside_exact", INSIDE_EXACT).radial
        lo, hi = r.reference_iqr_nm
        # Channel 1's own localizations scatter about their own contour
        # by the cluster sigma: for a straight edge the signed distance
        # is N(0, 8 nm), whose IQR is 1.349 x 8 = 10.8 nm. The polygon's
        # vertices bend it a little, never by a factor.
        width = hi - lo
        assert 7.0 < width < 15.0, (lo, hi)
        assert abs(r.reference_median_nm) < 8.0, r.reference_median_nm
        return (f"channel 1's own scatter: median "
                f"{r.reference_median_nm:+.1f} nm, IQR {width:.1f} nm "
                f"(N(0, 8) gives 10.8)")
    check("the reference scatter is the cluster sigma", reference_scatter)

    # =====================================================================
    print("\n5. WITH JITTER, AGAINST THE TRUE CENTRES  (independent code)")

    for label, key, pair in (("co-located", "colocated", COLOCATED),
                             ("interleaved", "interleaved", INTERLEAVED)):
        def hetero(pair=pair, key=key):
            t = run(key, pair)
            want_ab = float(np.median(true_nn(pair.truth_a.centres_nm,
                                              pair.truth_b.centres_nm)))
            want_ba = float(np.median(true_nn(pair.truth_b.centres_nm,
                                              pair.truth_a.centres_nm)))
            assert abs(t.median_hetero_nn_a_to_b - want_ab) < 3.0, \
                (t.median_hetero_nn_a_to_b, want_ab)
            assert abs(t.median_hetero_nn_b_to_a - want_ba) < 3.0, \
                (t.median_hetero_nn_b_to_a, want_ba)
            return (f"a->b {t.median_hetero_nn_a_to_b:.1f} (true "
                    f"{want_ab:.1f}), b->a {t.median_hetero_nn_b_to_a:.1f} "
                    f"(true {want_ba:.1f}) nm")
        check(f"{label}: heterotypic distances", hetero)

        def angles(pair=pair, key=key):
            t = run(key, pair)
            want = math.degrees(float(np.median(angle_gap(
                pair.truth_a.angles, pair.truth_b.angles))))
            assert abs(t.angular_nn_median_deg - want) < 1.0, \
                (t.angular_nn_median_deg, want)
            return (f"{t.angular_nn_median_deg:.2f} deg (true {want:.2f})")
        check(f"{label}: angular offset", angles)

        def radial(pair=pair, key=key):
            r = run(key, pair).radial
            poly = pair.truth_a.centres_nm
            want = float(np.median(true_signed_offsets(
                pair.truth_b.centres_nm, poly)))
            assert abs(r.median_nm - want) < 3.0, (r.median_nm, want)
            return f"{r.median_nm:+.1f} nm (true {want:+.1f})"
        check(f"{label}: signed radial offset", radial)

    def colocated_attracts():
        t = run("colocated", COLOCATED)
        assert t.fraction_null_below_measured is not None
        assert t.fraction_null_below_measured < 0.05, \
            t.fraction_null_below_measured
        return (f"random placement gets as close in "
                f"{100 * t.fraction_null_below_measured:.1f} % of draws")
    check("co-located clusters are closer than chance", colocated_attracts)

    def interleaved_is_not_called_close():
        # The first version of this check expected the null to call an
        # interleaved partner FURTHER than chance. It does not, and that
        # is the finding: sixteen clusters placed at random on a 2.5 um
        # ring already sit a median ~62 nm from their nearest partner, so
        # an interleaved one (64 nm with angular scatter) is inside the
        # null. The null sees attraction and not avoidance. What must
        # hold is that it does not call the interleaved partner CLOSE,
        # that the rotation places it at half the spacing, and that the
        # warning says the test is one-sided.
        t = run("interleaved", INTERLEAVED)
        assert t.fraction_null_below_measured > 0.05, \
            t.fraction_null_below_measured
        assert abs(t.rotation_fraction_of_spacing - 0.5) < 0.1, \
            t.rotation_fraction_of_spacing
        assert any("sees attraction, not avoidance" in w
                   for w in t.warnings), t.warnings
        return (f"not called close ({100 * t.fraction_null_below_measured:.0f}"
                f" % of random draws get as close), the rotation says "
                f"{t.rotation_fraction_of_spacing:.2f} of the spacing, and "
                f"the warning says the test is one-sided")
    check("the 1NN null does not see avoidance, and says so",
          interleaved_is_not_called_close)

    def colocated_rotation():
        t = run("colocated", COLOCATED)
        assert t.rotation_fraction_of_spacing < 0.15, \
            t.rotation_fraction_of_spacing
        return f"{t.rotation_fraction_of_spacing:.3f} of the spacing"
    check("co-located rings need no rotation", colocated_rotation)

    def colocated_shared():
        s = run("colocated", COLOCATED).shared
        assert s.shared_of_a > 0.5, s.shared_of_a
        assert s.p_null_at_least_measured < 0.01, s.p_null_at_least_measured
        return (f"shared_of_a {s.shared_of_a:.3f} against a null of "
                f"{s.null_median_shared_of_a:.3f}")
    check("co-located rings share most of the ring, beyond chance",
          colocated_shared)

    def interleaved_null():
        s = run("interleaved", INTERLEAVED).shared
        assert s.p_null_at_least_measured > 0.5, s.p_null_at_least_measured
        return (f"shared_of_a {s.shared_of_a:.3f}; chance reaches it in "
                f"{100 * s.p_null_at_least_measured:.0f} % of draws")
    check("interleaved rings do not beat chance", interleaved_null)

    def centre_vs_true_polygon():
        c = run("colocated", COLOCATED).analysis_a.centre
        want = shoelace_centroid(COLOCATED.truth_a.centres_nm)
        d = math.hypot(c.x_nm - want[0], c.y_nm - want[1])
        assert d < 3.0, (c.x_nm, c.y_nm, want)
        return f"{d:.2f} nm from the true polygon's area centroid"
    check("the contour centre, against the true polygon", centre_vs_true_polygon)

    # =====================================================================
    print("\n6. EACH CHANNEL'S OWN NUMBERS, AS EXPORTED")

    def per_channel():
        t = run("colocated", COLOCATED)
        row = export_cross_channel(None, t, "a", "b")
        want_p = perimeter(COLOCATED.truth_a.centres_nm) / 1000.0
        assert abs(row["perimeter_a_um"] - want_p) / want_p < 0.015, \
            (row["perimeter_a_um"], want_p)
        want_nn = float(np.median(true_self_nn(COLOCATED.truth_a.centres_nm)))
        assert abs(row["median_1nn_a_nm"] - want_nn) < 3.0, \
            (row["median_1nn_a_nm"], want_nn)
        assert row["n_locs_slab_a"] == int(in_slab(COLOCATED, "a").sum())
        assert abs(row["clusters_per_um_a"] - N / row["perimeter_a_um"]) < 1e-9
        return (f"perimeter {row['perimeter_a_um']:.3f} um (true "
                f"{want_p:.3f}); 1NN {row['median_1nn_a_nm']:.1f} nm (true "
                f"{want_nn:.1f}); {row['n_locs_slab_a']} in the slab")
    check("perimeter, 1NN and slab population of channel 1", per_channel)

    def export_is_the_result():
        t = run("colocated", COLOCATED)
        row = export_cross_channel(None, t, "a", "b")
        s, r = t.shared, t.radial
        pairs = [("shared_of_a", s.shared_of_a), ("shared_of_b", s.shared_of_b),
                 ("median_hetero_nn_a_to_b_nm", t.median_hetero_nn_a_to_b),
                 ("radial_offset_b_median_nm", r.median_nm),
                 ("locs_per_cell_b", t.locs_per_cell_b),
                 ("n_clusters_b_on_contour", s.n_clusters_b_on_contour)]
        for name, value in pairs:
            assert row[name] == value, (name, row[name], value)
        return f"{len(pairs)} columns read back unchanged"
    check("the exported row carries the numbers, not approximations",
          export_is_the_result)

    # =====================================================================
    print("\n7. DENSITY, IN THE UNIT MIN PTS COUNTS")

    def density_brute_force():
        sel = in_slab(COLOCATED, "a")
        x, y = COLOCATED.x_a[sel], COLOCATED.y_a[sel]
        want = brute_locs_per_cell(x, y, 20.0)
        got = locs_per_cell(x, y, sample=len(x) + 1)
        assert got == want, (got, want)
        return f"{got:.0f} localizations per 20 nm cell, both ways"
    check("locs_per_cell agrees with a brute-force count", density_brute_force)

    def density_analytic():
        # For an isotropic Gaussian cluster of n points and sigma s, the
        # other points within r of a point p number (n - 1) P(|X - p| < r).
        # Averaged over p, X - p ~ N(0, 2 s^2): the MEAN count is
        # 1 + (n - 1)(1 - exp(-r^2 / 4 s^2)). For p at the very centre,
        # X - p ~ N(0, s^2): the MOST any point can have is
        # 1 + (n - 1)(1 - exp(-r^2 / 2 s^2)). Edge points have few, so
        # the distribution leans left and the median sits above the mean.
        # (The first version of this check called the mean a bound.)
        sel = in_slab(COLOCATED, "a")
        x, y = COLOCATED.x_a[sel], COLOCATED.y_a[sel]
        mean_th = 1 + 59 * (1 - math.exp(-400.0 / (4 * 64.0)))
        top_th = 1 + 59 * (1 - math.exp(-400.0 / (2 * 64.0)))
        pts = np.column_stack([x, y])
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        mean_got = float((d2 <= 400.0).sum(axis=1).mean())
        median_got = locs_per_cell(x, y, sample=len(x) + 1)
        assert abs(mean_got - mean_th) < 3.0, (mean_got, mean_th)
        assert mean_th < median_got <= top_th, (median_got, mean_th, top_th)
        return (f"mean {mean_got:.1f} (theory {mean_th:.1f}); median "
                f"{median_got:.0f}, between the mean and the centre point's "
                f"{top_th:.1f}")
    check("and with the Gaussian cluster's own arithmetic", density_analytic)

    def oversampled_is_named():
        t = run("oversampled", OVERSAMPLED)
        ratio = t.locs_per_cell_b / t.locs_per_cell_a
        assert 6.0 < ratio < 14.0, ratio
        assert any("different regimes" in w for w in t.warnings), t.warnings
        return (f"{t.locs_per_cell_a:.0f} against {t.locs_per_cell_b:.0f}: "
                f"a factor of {ratio:.1f} for ten times the localizations")
    check("a ten-times oversampled partner is named as such",
          oversampled_is_named)

    # =====================================================================
    print("\n8. THE REGISTRATION BAND")

    def band_grows():
        small = compare(IDENTITY, n_null=0, registration=Registration(
            source="fiducials", lateral_rms_nm=3.0)).shared
        large = compare(IDENTITY, n_null=0, registration=Registration(
            source="fiducials", lateral_rms_nm=20.0)).shared
        w_small = small.registration_band_shared_of_a[1] \
            - small.registration_band_shared_of_a[0]
        w_large = large.registration_band_shared_of_a[1] \
            - large.registration_band_shared_of_a[0]
        assert w_large > w_small, (w_small, w_large)
        assert large.registration_band_shared_of_a[0] < 1.0
        return (f"3 nm: {small.registration_band_shared_of_a[0]:.3f}.."
                f"{small.registration_band_shared_of_a[1]:.3f}; 20 nm: "
                f"{large.registration_band_shared_of_a[0]:.3f}.."
                f"{large.registration_band_shared_of_a[1]:.3f}")
    check("a worse registration gives a wider band", band_grows)

    def band_withdraws():
        s = compare(IDENTITY, n_null=0, registration=Registration(
            source="manual", lateral_rms_nm=200.0)).shared
        assert any("not a measurement" in w for w in s.warnings), s.warnings
        u = compare(IDENTITY, n_null=0).shared
        assert u.registration_band_shared_of_a is None
        assert any("no error bar" in w for w in u.warnings)
        return (f"200 nm against patches of {s.median_patch_a_nm:.0f} nm: "
                f"withdrawn; unknown: no band, and said")
    check("and one past the patch scale withdraws the number",
          band_withdraws)

    # =====================================================================
    print("\n9. THE AXIAL PHASE")

    for phase in (0.0, 0.25, 0.5):
        def axial(phase=phase):
            pair = ring_pair(20 + int(phase * 100), phase=phase)
            a = axial_phase(pair.z_a, pair.z_b)
            want = min(phase % 1.0, 1.0 - phase % 1.0)
            assert abs(a.phase_fraction - want) < 0.04, (a.phase_fraction,
                                                         want)
            assert abs(a.offset_nm - phase * pair.period_nm) < 8.0, \
                (a.offset_nm, phase * pair.period_nm)
            assert abs(a.period_a_nm - pair.period_nm) < 5.0, a.period_a_nm
            assert a.peaks_separated_a is True, a.peaks_separated_a
            return (f"{a.phase_fraction:.3f}, offset {a.offset_nm:+.1f} nm "
                    f"(true {phase * pair.period_nm:+.1f}), period "
                    f"{a.period_a_nm:.1f} nm, '{a.interpretation_hint}'")
        check(f"phase {phase}", axial)

    def unseparated():
        rng = np.random.default_rng(30)
        blob = rng.normal(0.0, 90.0, 6000)
        a = axial_phase(blob, blob + 95.0)
        assert a.peaks_separated_a is False, a.peaks_separated_a
        assert a.interpretation_hint == "undetermined", a.interpretation_hint
        return (f"one broad blob: phase {a.phase_fraction} withheld as "
                f"'{a.interpretation_hint}'")
    check("a period that was never there is not named", unseparated)

    # =====================================================================
    print("\n10. END TO END, WITH THE AUTOMATIC SLAB")

    def automatic_slab():
        pair = ring_pair(40)
        t = compare(pair, slab=None)
        assert t.n_clusters_a == N and t.n_clusters_b == N, \
            (t.n_clusters_a, t.n_clusters_b)
        assert -40.0 < t.slab_nm[0] + 90.0 < 40.0, t.slab_nm
        row = export_cross_channel(axial_phase(pair.z_a, pair.z_b), t, "a", "b")
        numbers = [v for v in row.values() if isinstance(v, float)]
        assert numbers and all(math.isfinite(v) for v in numbers)
        return (f"slab {t.slab_nm[0]:.0f}..{t.slab_nm[1]:.0f} nm, "
                f"{t.n_clusters_a}/{t.n_clusters_b} clusters, "
                f"{len(numbers)} numeric columns, all finite")
    check("three rings, slab found by the axial fit", automatic_slab)

    def refused_without_parameters():
        try:
            cross_channel_transverse(
                IDENTITY.x_a, IDENTITY.y_a, IDENTITY.z_a, IDENTITY.x_b,
                IDENTITY.y_b, IDENTITY.z_b, slab=SLAB,
                pixel_size_source="not_applicable", eps_nm=25.0,
                min_samples=10, n_null=0)
        except ValueError as error:
            assert str(error) == CHANNEL2_PARAMETERS_REQUIRED
            return "refused, with the message that says what to do"
        raise AssertionError("it ran on channel 1's parameters")
    check("and still refuses to borrow channel 1's parameters",
          refused_without_parameters)

    # =====================================================================
    print("\n11. A PARTNER OUT OF PHASE  (the slab cuts its rings' tails)")

    def antiphase_inflates_and_says_so():
        # Same seed, so the partner's positions in the plane are identical
        # and only its axial phase moves. In antiphase, spectrin's slab
        # holds the tails of the partner's rings above and below; those
        # merge into wider clusters, which cover more perimeter. The
        # number moves the WRONG way -- more shared for a protein that is
        # less in register -- so it must be said, not just computed.
        shared, warned = {}, {}
        for phase in (0.0, 0.5):
            t = compare(ring_pair(60, phase=phase), n_null=0)
            shared[phase] = t.shared.shared_of_a
            warned[phase] = any("tails of the rings" in w for w in t.warnings)
        assert shared[0.5] > shared[0.0] + 0.05, shared
        assert not warned[0.0] and warned[0.5], warned
        return (f"shared_of_a {shared[0.0]:.3f} in phase, {shared[0.5]:.3f} "
                f"in antiphase with the same positions; warned only in "
                f"antiphase")
    check("antiphase inflates the shared occupancy, and it is said",
          antiphase_inflates_and_says_so)

    def quarter_phase_is_quiet():
        t = compare(ring_pair(61, phase=0.25), n_null=0)
        assert not any("tails of the rings" in w for w in t.warnings)
        return "a quarter period keeps the partner's ring in the slab"
    check("and a quarter period does not trip it", quarter_phase_is_quiet)

    print(f"\n{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
