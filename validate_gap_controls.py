# -*- coding: utf-8 -*-
"""
Three controls on the correlation between the gap/patch patterns of
consecutive MPS segments.

validate_gaps.py measures that correlation: mean r(0) = +0.09 with 10 of
33 segment pairs significant against the rotation null. This script asks
whether that is a relationship between rings, or an artefact.

The suspicion is axial bleed-through. Axial localization precision in
3D dSTORM (~50-80 nm) is comparable to the 180 nm slab thickness, and the
fitted component sigmas (55-112 nm) are consistent with being dominated by
it. One physical ring then deposits localizations on both sides of a
boundary, so two slabs that share no localization still share clusters --
which by itself puts the same patches at the same angles in both.

Control 1 -- guard band.
    Widen a dead zone around each boundary. Bleed-through falls off with
    the guard; a real relationship between rings does not.

Control 2 -- axial distance.
    Correlate every pair of segments in an axon, not only neighbours, and
    look at r(0) against their axial separation. Bleed-through decays on
    the scale of the axial precision; register between rings should not
    care how many rings apart they are.

Control 3 -- different axons.
    Correlate segments belonging to DIFFERENT axons. Angles are measured in
    the lab frame, so an illumination or labelling gradient across the
    field of view would correlate any two axons. This pair set shares no
    biology at all, so whatever it shows is the floor of the method.

Run:  python validate_gap_controls.py
"""

from __future__ import annotations

import itertools
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_gaps import coverage_profile, profile_correlation  # noqa: E402
from tools.mps_multisegment import analyze_all_segments  # noqa: E402
from validate_gaps import find_axons, label, load  # noqa: E402

GUARDS_NM = (0.0, 30.0, 60.0, 90.0)


def build_profiles(x, y, z, px, source, mode="valley", guard_nm=0.0):
    """Coverage profile and axial centre of every analysable segment."""
    ms = analyze_all_segments(
        x, y, z, source_name=source, mode=mode, guard_nm=guard_nm,
        pixel_size_nm=px, pixel_size_source="yaml")
    if ms.axon_center is None:
        return [], ms

    out = []
    for seg, an in zip(ms.segments, ms.analyses):
        if an is None or an.occupancy is None or an.perimeter is None:
            continue
        try:
            prof = coverage_profile(
                an.occupancy, an.perimeter.contour, ms.axon_center)
        except ValueError:
            continue
        out.append((seg, prof))
    return out, ms


def kept_labels(an) -> np.ndarray:
    """Cluster labels in the same order as ``AxonAnalysis.centroids``:
    ascending, noise and curated-out clusters removed (see
    ``cluster_quality.good_cluster_centroids``). Indexing centroids by any
    other label list silently pairs a centroid with another cluster's
    label as soon as one cluster is curated out."""
    bad = an.bad_report.bad_labels
    return np.array([int(l) for l in np.unique(an.labels)
                     if l != -1 and int(l) not in bad], dtype=int)


def summarize(name: str, r0: np.ndarray, indent: str = "  ") -> Dict:
    if r0.size == 0:
        print(f"{indent}{name:<26} (no pairs)")
        return {}
    n_sig_hi = int((r0 > 0).sum())
    print(f"{indent}{name:<26} n={r0.size:>4}   mean r(0)={r0.mean():+.3f}   "
          f"median={np.median(r0):+.3f}   r>0: {n_sig_hi}/{r0.size}")
    return {"n": r0.size, "mean": float(r0.mean()),
            "median": float(np.median(r0))}


def main() -> int:
    files = find_axons()
    if not files:
        print("No axon files found; set MPS_VALIDATION_DATA.")
        return 1

    data = {f: load(f) for f in files}
    print(f"{len(files)} axons\n")

    # ================================================== control 1: guard
    print("=" * 92)
    print("CONTROL 1 -- GUARD BAND: does the similarity survive separating "
          "the slabs?")
    print("=" * 92)
    print(f"{'guard nm':>9} {'pairs':>6} {'segments':>9} {'mean r(0)':>10} "
          f"{'median':>8} {'p<0.05':>8} {'expected':>9}")
    print("-" * 92)

    guard_r0: Dict[float, np.ndarray] = {}
    for guard in GUARDS_NM:
        r0s: List[float] = []
        n_seg = 0
        for f in files:
            x, y, z, px = data[f]
            profs, ms = build_profiles(x, y, z, px, f, guard_nm=guard)
            n_seg += len(profs)
            for (sa, pa), (sb, pb) in zip(profs[:-1], profs[1:]):
                if sb.index != sa.index + 1:
                    continue
                c = profile_correlation(pa.coverage, pb.coverage)
                if np.isfinite(c.r_at_zero):
                    r0s.append(c.r_at_zero)
        arr = np.asarray(r0s, float)
        guard_r0[guard] = arr
        if arr.size:
            # Significance re-derived per pair would need the full curve; the
            # mean is what the control is about, so only r(0) is kept here.
            print(f"{guard:>9.0f} {arr.size:>6} {n_seg:>9} {arr.mean():>+10.3f} "
                  f"{np.median(arr):>+8.3f} {'--':>8} {'--':>9}")
        else:
            print(f"{guard:>9.0f} {0:>6} {n_seg:>9}   (no pairs survived)")

    base = guard_r0[0.0]
    wide = guard_r0[max(GUARDS_NM)]
    if base.size and wide.size:
        print(f"\n  r(0) at guard 0 : {base.mean():+.3f} (n={base.size})")
        print(f"  r(0) at guard {max(GUARDS_NM):.0f}: {wide.mean():+.3f} "
              f"(n={wide.size})")
        print(f"  change          : {wide.mean() - base.mean():+.3f}")
        print("  Reading: a large drop means the similarity was axial "
              "bleed-through.")

    # ======================================= control 2: axial separation
    print("\n" + "=" * 92)
    print("CONTROL 2 -- AXIAL DISTANCE: does the similarity decay with "
          "separation?")
    print("=" * 92)

    sep: List[float] = []
    r0_all: List[float] = []
    adjacency: List[int] = []
    for f in files:
        x, y, z, px = data[f]
        profs, ms = build_profiles(x, y, z, px, f, guard_nm=0.0)
        for (sa, pa), (sb, pb) in itertools.combinations(profs, 2):
            c = profile_correlation(pa.coverage, pb.coverage)
            if not np.isfinite(c.r_at_zero):
                continue
            sep.append(abs(sb.center_nm - sa.center_nm))
            r0_all.append(c.r_at_zero)
            adjacency.append(abs(sb.index - sa.index))

    sep_a = np.asarray(sep, float)
    r0_a = np.asarray(r0_all, float)
    adj_a = np.asarray(adjacency, int)

    if sep_a.size >= 3:
        print(f"{'separation':>22} {'pairs':>6} {'mean r(0)':>10} {'median':>8}")
        print("-" * 92)
        for k in sorted(set(adj_a.tolist())):
            m = adj_a == k
            print(f"{f'{k} segment(s) apart':>22} {int(m.sum()):>6} "
                  f"{r0_a[m].mean():>+10.3f} {np.median(r0_a[m]):>+8.3f}"
                  f"   (mean dz = {sep_a[m].mean():.0f} nm)")

        rho, p_rho = stats.spearmanr(sep_a, r0_a)
        print(f"\n  Spearman r(0) vs axial separation: rho = {rho:+.3f}, "
              f"p = {p_rho:.3f}  (n = {sep_a.size})")
        print("  Reading: a clearly negative rho means the similarity is "
              "short-range, i.e. bleed-through.")

    # ============================================ control 3: other axons
    print("\n" + "=" * 92)
    print("CONTROL 3 -- DIFFERENT AXONS: what does the method report for "
          "pairs that share no biology?")
    print("=" * 92)

    by_axon: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for f in files:
        x, y, z, px = data[f]
        profs, ms = build_profiles(x, y, z, px, f, guard_nm=0.0)
        by_axon[f] = [(seg.index, p.coverage) for seg, p in profs]

    cross: List[float] = []
    for fa, fb in itertools.combinations(files, 2):
        for (ia, ca) in by_axon.get(fa, []):
            for (ib, cb) in by_axon.get(fb, []):
                if ia != ib:
                    continue          # same segment rank, different axons
                c = profile_correlation(ca, cb)
                if np.isfinite(c.r_at_zero):
                    cross.append(c.r_at_zero)

    cross_a = np.asarray(cross, float)
    within = np.asarray(guard_r0[0.0], float)

    print()
    summarize("within axon, adjacent", within)
    summarize("across axons", cross_a)

    if cross_a.size and within.size:
        u = stats.mannwhitneyu(within, cross_a, alternative="greater")
        print(f"\n  Mann-Whitney (within > across): U p = {u.pvalue:.2e}")
        print(f"  across-axon mean r(0) = {cross_a.mean():+.4f}   "
              f"(should sit at 0 if the lab frame carries no gradient)")
        t = stats.ttest_1samp(cross_a, 0.0)
        print(f"  across-axon vs 0      : t p = {t.pvalue:.2e}")
        print("  Reading: an across-axon mean away from 0 means a field-wide "
              "gradient, and the within-axon number must be read against "
              "this floor rather than against 0.")

    # ====================================== control 4: cluster identity
    #
    # Controls 1-3 leave one alternative alive. A cluster centred in one
    # slab can still spread beyond the guard band into the next, because
    # the axial spread (sigma 55-112 nm) is a large fraction of the
    # distance to the boundary. The same physical cluster would then be
    # detected in both segments at the same angle, which produces exactly
    # the observed short-range, guard-resistant, axon-specific
    # correlation. So: how many clusters do two consecutive segments
    # actually share, and does the correlation survive removing them?
    print("\n" + "=" * 92)
    print("CONTROL 4 -- CLUSTER IDENTITY: are the two segments detecting the "
          "same clusters?")
    print("=" * 92)
    print(f"{'axon':<18} {'pair':>5} {'Nb':>4} {'matched':>8} {'%':>6} "
          f"{'chance %':>9} {'r(0)':>7} {'r(0) unmatched':>15}")
    print("-" * 92)

    rng = np.random.default_rng(0)
    match_frac: List[float] = []
    chance_frac: List[float] = []
    r_before: List[float] = []
    r_after: List[float] = []

    for f in files:
        x, y, z, px = data[f]
        profs, ms = build_profiles(x, y, z, px, f, guard_nm=0.0)
        segs = [s for s, _ in profs]
        by_index = {s.index: (s, p) for s, p in profs}
        analyses = {s.index: a for s, a in zip(ms.segments, ms.analyses)}

        for sa in segs:
            sb = by_index.get(sa.index + 1)
            if sb is None:
                continue
            an_a, an_b = analyses.get(sa.index), analyses.get(sa.index + 1)
            if an_a is None or an_b is None:
                continue
            ca, cb = an_a.centroids, an_b.centroids
            if ca.size == 0 or cb.size == 0:
                continue

            d = np.linalg.norm(cb[:, None, :] - ca[None, :, :], axis=2)
            nn = d.min(axis=1)
            matched = nn <= an_b.eps_nm          # same cluster, seen twice
            frac = float(matched.mean())

            # Chance level: rotate B's centroids about the axon centre,
            # which keeps the ring geometry and the cluster count but
            # destroys identity.
            ctr = ms.axon_center
            rel = cb - ctr
            hits = []
            for _ in range(200):
                a_ = rng.uniform(0, 2 * np.pi)
                rot = np.column_stack([
                    rel[:, 0] * np.cos(a_) - rel[:, 1] * np.sin(a_),
                    rel[:, 0] * np.sin(a_) + rel[:, 1] * np.cos(a_)]) + ctr
                dr = np.linalg.norm(rot[:, None, :] - ca[None, :, :], axis=2)
                hits.append(float((dr.min(axis=1) <= an_b.eps_nm).mean()))
            chance = float(np.mean(hits))

            pa, pb = by_index[sa.index][1], sb[1]
            c_before = profile_correlation(pa.coverage, pb.coverage)

            # Drop the shared clusters from BOTH profiles and re-measure.
            drop_a: set = set()
            drop_b: set = set()
            if matched.any():
                lab_a, lab_b = kept_labels(an_a), kept_labels(an_b)
                if lab_a.size != len(ca) or lab_b.size != len(cb):
                    print(f"{label(f):<18}  (label/centroid mismatch, skipped)")
                    continue
                drop_b = {int(l) for l, keep in zip(lab_b, matched) if keep}
                near_a = d.min(axis=0) <= an_b.eps_nm
                drop_a = {int(l) for l, keep in zip(lab_a, near_a) if keep}

            r_aft: Optional[float] = None
            if drop_b or drop_a:
                try:
                    pa2 = coverage_profile(
                        an_a.occupancy, an_a.perimeter.contour, ctr,
                        exclude_labels=drop_a)
                    pb2 = coverage_profile(
                        an_b.occupancy, an_b.perimeter.contour, ctr,
                        exclude_labels=drop_b)
                    c_after = profile_correlation(pa2.coverage, pb2.coverage)
                    if np.isfinite(c_after.r_at_zero):
                        r_aft = float(c_after.r_at_zero)
                except ValueError:
                    r_aft = None
            else:
                r_aft = float(c_before.r_at_zero)

            match_frac.append(frac)
            chance_frac.append(chance)
            r_before.append(float(c_before.r_at_zero))
            if r_aft is not None:
                r_after.append(r_aft)

            print(f"{label(f):<18} {f'{sa.index}-{sa.index+1}':>5} "
                  f"{len(cb):>4} {int(matched.sum()):>8} {100 * frac:>6.1f} "
                  f"{100 * chance:>9.1f} {c_before.r_at_zero:>+7.3f} "
                  f"{'n/a' if r_aft is None else f'{r_aft:+.3f}':>15}")

    mf = np.asarray(match_frac, float)
    cf = np.asarray(chance_frac, float)
    rb = np.asarray(r_before, float)
    ra = np.asarray(r_after, float)

    if mf.size:
        print(f"\n  shared clusters   : {100 * mf.mean():.1f}% of the later "
              f"segment's clusters lie within eps of one in the earlier "
              f"segment")
        print(f"  chance level      : {100 * cf.mean():.1f}% "
              f"(same clusters, randomly rotated)")
        w = stats.wilcoxon(mf, cf) if mf.size > 5 else None
        if w is not None:
            print(f"  above chance      : Wilcoxon p = {w.pvalue:.2e}")
        print(f"\n  r(0) with shared clusters   : {rb.mean():+.3f} "
              f"(n = {rb.size})")
        print(f"  r(0) with them removed      : {ra.mean():+.3f} "
              f"(n = {ra.size})")
        if ra.size == rb.size:
            w2 = stats.wilcoxon(rb, ra)
            print(f"  paired difference           : "
                  f"{(rb - ra).mean():+.3f}   Wilcoxon p = {w2.pvalue:.3f}")
        print("  Reading: if the correlation collapses once shared clusters "
              "are removed, the two segments were detecting one ring twice.")

    # ================================ control 5: resolved vs unresolved
    #
    # Most boundaries have no interior density minimum at all: the
    # components are too broad relative to their spacing, so the two
    # segments are two halves of one axial distribution rather than two
    # rings. If the correlation lives only in those pairs, it is a
    # restatement of that, not a relationship between rings.
    print("\n" + "=" * 92)
    print("CONTROL 5 -- RESOLVED vs UNRESOLVED BOUNDARIES: where does the "
          "correlation live?")
    print("=" * 92)
    print(f"{'axon':<18} {'pair':>5} {'depth':>7} {'valley?':>8} "
          f"{'sigma a':>8} {'sigma b':>8} {'dz':>6} {'r(0)':>7}")
    print("-" * 92)

    depth_l: List[float] = []
    r_l: List[float] = []
    resolved: List[bool] = []
    for f in files:
        x, y, z, px = data[f]
        profs, ms = build_profiles(x, y, z, px, f, guard_nm=0.0)
        by_index = {s.index: (s, p) for s, p in profs}
        for pair in ms.pairs:
            a, b = by_index.get(pair.index_a), by_index.get(pair.index_b)
            if a is None or b is None or pair.boundary_relative_depth is None:
                continue
            c = profile_correlation(a[1].coverage, b[1].coverage)
            if not np.isfinite(c.r_at_zero):
                continue
            depth_l.append(pair.boundary_relative_depth)
            r_l.append(c.r_at_zero)
            resolved.append(bool(pair.boundary_is_true_valley))
            print(f"{label(f):<18} {f'{pair.index_a}-{pair.index_b}':>5} "
                  f"{pair.boundary_relative_depth:>7.3f} "
                  f"{str(bool(pair.boundary_is_true_valley)):>8} "
                  f"{a[0].sigma_nm:>8.0f} {b[0].sigma_nm:>8.0f} "
                  f"{pair.delta_z_nm:>6.0f} {c.r_at_zero:>+7.3f}")

    dl = np.asarray(depth_l, float)
    rl = np.asarray(r_l, float)
    rs = np.asarray(resolved, bool)

    if rl.size:
        print()
        summarize("real valley", rl[rs])
        summarize("no valley (midpoint)", rl[~rs])
        deep = dl > 0.10
        summarize("depth > 0.10", rl[deep])
        summarize("depth <= 0.10", rl[~deep])

        if rs.any() and (~rs).any():
            u = stats.mannwhitneyu(rl[rs], rl[~rs], alternative="two-sided")
            print(f"\n  resolved vs unresolved : Mann-Whitney p = "
                  f"{u.pvalue:.3f}")
        if dl.size > 3:
            rho, p_rho = stats.spearmanr(dl, rl)
            print(f"  r(0) vs valley depth   : rho = {rho:+.3f}, "
                  f"p = {p_rho:.3f}  (n = {dl.size})")
        print("  Reading: if the correlation is confined to the unresolved "
              "pairs, it restates that those two segments are one "
              "distribution; if it is present in the resolved pairs too, it "
              "is a relationship between rings.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
