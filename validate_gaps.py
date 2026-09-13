# -*- coding: utf-8 -*-
"""
Validation of tools/mps_gaps.py on the 18 real axons.

Answers three questions, in order:

1. Does cutting consecutive slabs at the density valley differ from cutting
   them at the midpoint between component means? (If not, the choice does
   not matter and the simpler rule wins.)
2. What do the gaps and patches of a single MPS segment look like -- how
   many, how long?
3. Are the patches of one segment at the same angles as those of the next?
   Reported per pair and pooled, against the rotation null, and with the
   overlapping ("paper") and disjoint ("valley") segmentations side by side
   because overlapping slabs share localizations and inflate any
   similarity between them.

Run:  python validate_gaps.py
"""

from __future__ import annotations

import glob
import os
import sys
from typing import List, Optional

import h5py as h5
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_gaps import (  # noqa: E402
    coverage_profile,
    gap_patch_runs,
    profile_correlation,
)
from tools.mps_multisegment import (  # noqa: E402
    analyze_all_segments,
    fold_rotation,
)
from tools.mps_periodicity import (  # noqa: E402
    density_valleys,
    fit_z_periodicity,
)

DATA_ROOT = os.environ.get(
    "MPS_VALIDATION_DATA",
    r"C:\Users\nicol\OneDrive\Doctorado\1°Reunión de avances de tesis\Abril",
)


def read_pixelsize(path: str) -> float:
    y = os.path.splitext(path)[0] + ".yaml"
    for line in open(y, encoding="utf-8", errors="ignore"):
        if line.strip().startswith("Pixelsize:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"no Pixelsize for {path}")


def find_axons() -> List[str]:
    out = []
    for pat in ["ROI 1/*.hdf5", "ROI 1/*/*.hdf5", "ROI 2/*.hdf5"]:
        for f in glob.glob(os.path.join(DATA_ROOT, pat)):
            if "axon" in os.path.basename(f).lower():
                out.append(f)
    return sorted(set(out))


def label(path: str) -> str:
    roi = "ROI1" if "ROI 1" in path else "ROI2"
    return roi + "/" + os.path.basename(path).split("icked")[-1][:10]


def load(path: str):
    px = read_pixelsize(path)
    with h5.File(path, "r") as h:
        ds = h["locs"]
        return (np.asarray(ds["x"], float) * px,
                np.asarray(ds["y"], float) * px,
                np.asarray(ds["z"], float), px)


def fmt(v: Optional[float], n: int = 1) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:,.{n}f}"


# ============================================================ 1. valleys
def report_valleys(files) -> None:
    print("=" * 104)
    print("1. VALLEY vs MIDPOINT boundaries between consecutive segments")
    print("=" * 104)
    print(f"{'axon':<18} {'sigmas (nm)':<22} {'valleys':<18} "
          f"{'midpoints':<18} {'shift':>6} {'depth':<14} {'real?':<8}")
    print("-" * 104)

    shifts: List[float] = []
    depths: List[float] = []
    true_flags: List[bool] = []
    for f in files:
        x, y, z, px = load(f)
        r = fit_z_periodicity(z)
        means = np.asarray(r.means_nm, float)
        if means.size < 2:
            print(f"{label(f):<18} (single dominant component)")
            continue
        vr = density_valleys(r)
        shifts.extend(vr.shift_from_midpoint_nm.tolist())
        depths.extend(vr.relative_depth.tolist())
        true_flags.extend(vr.is_true_valley.tolist())
        print(f"{label(f):<18} {str(np.round(r.sigmas_nm, 0)):<22} "
              f"{str(np.round(vr.positions_nm, 0)):<18} "
              f"{str(np.round(vr.midpoints_nm, 0)):<18} "
              f"{np.max(vr.shift_from_midpoint_nm):>6.0f} "
              f"{str(np.round(vr.relative_depth, 2)):<14} "
              f"{str(vr.is_true_valley.astype(int)):<8}")

    if shifts:
        s = np.asarray(shifts)
        d = np.asarray(depths)
        t = np.asarray(true_flags)
        print(f"\n  boundaries: n = {s.size}   "
              f"median shift {np.median(s):.1f} nm   mean {s.mean():.1f} nm   "
              f"max {s.max():.1f} nm")
        print(f"  moved > 5 nm  : {int((s > 5).sum())} of {s.size}      "
              f"moved > 20 nm: {int((s > 20).sum())} of {s.size}")
        print(f"\n  ARE THE TWO SEGMENTS AXIALLY RESOLVED AT ALL?")
        print(f"  real interior valley : {int(t.sum())} of {t.size} boundaries")
        print(f"  no valley (midpoint) : {int((~t).sum())} of {t.size} "
              f"-- components too broad relative to their spacing")
        if t.any():
            print(f"  depth where real     : median {np.median(d[t]):.3f}  "
                  f"range {d[t].min():.3f} .. {d[t].max():.3f}   "
                  f"(fraction below the shallower adjacent peak)")
        print(f"  deep (> 0.10)        : {int((d > 0.10).sum())} of {d.size}")


# ================================================== 2 & 3. gaps + pairing
def report_mode(files, mode: str) -> dict:
    print("\n" + "=" * 104)
    print(f"2+3. GAPS / PATCHES and CONSECUTIVE-SEGMENT CORRELATION "
          f"-- mode = {mode!r}")
    print("=" * 104)
    print(f"{'axon':<20} {'seg':>3} {'N':>4} {'occ%':>6} {'patch':>6} "
          f"{'gap':>6} {'maxgap':>7} {'#pch':>5} | "
          f"{'pair':>5} {'r(0)':>7} {'p_rot':>8} {'z':>6} {'rotfrac':>8} "
          f"{'nn_deg':>7}")
    print("-" * 104)

    r0_all, p_all, z_all, frac_all, nn_all = [], [], [], [], []
    n_seg = n_pair = 0

    for f in files:
        x, y, z, px = load(f)
        ms = analyze_all_segments(x, y, z, source_name=f, mode=mode,
                                  pixel_size_nm=px, pixel_size_source="yaml")
        if ms.axon_center is None:
            print(f"{label(f):<20}  (no clusters in any segment)")
            continue

        profiles = []
        for seg, an in zip(ms.segments, ms.analyses):
            if an is None or an.occupancy is None or an.perimeter is None:
                profiles.append(None)
                print(f"{label(f):<20} {seg.index:>3}   (no occupancy)")
                continue
            n_seg += 1
            runs = gap_patch_runs(an.occupancy.occupied_mask,
                                  an.occupancy.point_spacing_nm)
            prof = coverage_profile(an.occupancy, an.perimeter.contour,
                                    ms.axon_center)
            profiles.append(prof)
            print(f"{label(f):<20} {seg.index:>3} {an.n_clusters_kept:>4} "
                  f"{fmt(an.occupancy_percent):>6} "
                  f"{fmt(runs.median_patch_nm, 0):>6} "
                  f"{fmt(runs.median_gap_nm, 0):>6} "
                  f"{fmt(runs.max_gap_nm, 0):>7} {runs.n_patches:>5} |")

        for i, pair in enumerate(ms.pairs):
            pa, pb = profiles[i], profiles[i + 1]
            if pa is None or pb is None:
                continue
            c = profile_correlation(pa.coverage, pb.coverage)
            if not np.isfinite(c.r_at_zero):
                continue
            n_pair += 1
            frac = fold_rotation(c.best_offset_deg,
                                 pair.mean_angular_spacing_deg or float("nan"))
            r0_all.append(c.r_at_zero)
            p_all.append(c.p_rotation)
            if c.z_vs_null is not None:
                z_all.append(c.z_vs_null)
            if frac is not None:
                frac_all.append(frac)
            if pair.angular_nn_median_deg is not None:
                nn_all.append(pair.angular_nn_median_deg)
            print(f"{'':<20} {'':>3} {'':>4} {'':>6} {'':>6} {'':>6} "
                  f"{'':>7} {'':>5} | {f'{i}-{i+1}':>5} "
                  f"{c.r_at_zero:>7.3f} {c.p_rotation:>8.4f} "
                  f"{fmt(c.z_vs_null, 2):>6} {fmt(frac, 3):>8} "
                  f"{fmt(pair.angular_nn_median_deg, 1):>7}")

    r0 = np.asarray(r0_all, float)
    p = np.asarray(p_all, float)
    zz = np.asarray(z_all, float)
    fr = np.asarray(frac_all, float)

    print(f"\n  segments: {n_seg}    consecutive pairs compared: {n_pair}")
    if r0.size == 0:
        return {}

    fisher = np.arctanh(np.clip(r0, -0.999999, 0.999999))
    n_sig = int((p < 0.05).sum())
    binom = stats.binomtest(n_sig, r0.size, 0.05, alternative="greater")

    print(f"\n  {'-' * 78}")
    print(f"  COVERAGE-PROFILE CORRELATION AT ZERO ROTATION (n = {r0.size})")
    print(f"  {'-' * 78}")
    print(f"  r(0)          : median {np.median(r0):+.3f}   "
          f"mean {r0.mean():+.3f}   range {r0.min():+.3f} .. {r0.max():+.3f}")
    print(f"  Fisher-z mean : {fisher.mean():+.3f}  "
          f"-> back-transformed r = {np.tanh(fisher.mean()):+.3f}")
    print(f"  r(0) > 0      : {int((r0 > 0).sum())} of {r0.size} pairs")
    print(f"  z vs null     : median {np.median(zz):+.2f} "
          f"(r(0) in SDs of that pair's own rotation null)")
    print(f"  p_rot < 0.05  : {n_sig} of {r0.size} pairs "
          f"({100.0 * n_sig / r0.size:.0f}%; 5% expected under the null)")
    print(f"  binomial test : p = {binom.pvalue:.2e} "
          f"(more significant pairs than chance)")
    if fr.size:
        ks = stats.kstest(fr * 2.0, "uniform")
        print(f"\n  ROTATION OFFSET as a fraction of the mean angular spacing "
              f"(n = {fr.size})")
        print(f"  folded [0, 0.5]: median {np.median(fr):.3f}   mean {fr.mean():.3f}"
              f"   (0 = aligned, 0.5 = maximally staggered)")
        print(f"  KS vs uniform  : D = {ks.statistic:.3f}, p = {ks.pvalue:.3f}")
    if nn_all:
        nn = np.asarray(nn_all, float)
        print(f"\n  CROSS-CHECK, point-based angular 1NN offset (n = {nn.size})")
        print(f"  median {np.median(nn):.1f} deg   "
              f"(random expectation ~ 90/N deg for N clusters)")

    return {"r0": r0, "p": p, "n_sig": n_sig, "fisher": fisher}


def main() -> int:
    files = find_axons()
    if not files:
        print(f"No axon HDF5 files under {DATA_ROOT!r}.")
        print("Set MPS_VALIDATION_DATA to the dataset root.")
        return 1
    print(f"{len(files)} axons under {DATA_ROOT}\n")

    report_valleys(files)
    valley = report_mode(files, "valley")
    paper = report_mode(files, "paper")

    if valley and paper:
        print("\n" + "=" * 104)
        print("4. DOES THE AXIAL OVERLAP INFLATE THE CORRELATION?")
        print("=" * 104)
        print(f"  disjoint slabs  (valley): mean r(0) = {valley['r0'].mean():+.3f}"
              f"   significant {valley['n_sig']}/{valley['r0'].size}")
        print(f"  overlapping     (paper) : mean r(0) = {paper['r0'].mean():+.3f}"
              f"   significant {paper['n_sig']}/{paper['r0'].size}")
        print(f"  difference in mean r(0) : "
              f"{paper['r0'].mean() - valley['r0'].mean():+.3f} "
              f"(positive = overlap inflates similarity, as expected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
