"""
Validation of the exclusion depth of the automatic discard of unanchored clusters (``tools.mps_unanchored``,
``DEFAULT_EXCLUDE_NM``) WITHOUT labels made by hand.

    python validate_unanchored_threshold.py sim --out DIR --seeds 0:200        (simulated axons, known truth)
    python validate_unanchored_threshold.py summarize --out DIR                (error curves and the criterion)
    python validate_unanchored_threshold.py real --file PATH --tag NAME --out DIR [--reference-flags CSV]
    python validate_unanchored_threshold.py agree --out DIR                    (agreement with a reference method)

``sim``: each simulated axon is a closed membrane with clusters scattered about it, optional concave bays, gaps,
interior clusters planted at known depths (150-800 nm below the TRUE membrane) and diffuse interior lumps too small
to be clusters. The rule of the programme is applied as it is (depth of the cluster centre below the polar ring of
the localizations, and below the smooth ring of the centres), and the depth of every cluster is written with its
truth, so that any threshold can be scored afterwards. The star-shape test is evaluated at every threshold, as the
programme does (a rejected ring excludes nothing).

``summarize``: FPR (membrane clusters excluded) and FNR (interior clusters kept) from 250 to 550 nm, the expected
error per cluster E = p * FNR + (1 - p) * FPR at a stated prevalence p, the guard on FPR, and bootstrap intervals
over whole simulated axons. The criterion has to be fixed BEFORE the results are looked at; this script only
computes it.

``real``: one data file per process, READ ONLY: the main analysis, the depth of every cluster under both rings, and
the analysis without the clusters deeper than each threshold (perimeter, occupancy, 1NN). ``--reference-flags``:
a CSV with columns ``cx``, ``cy`` (nm) and ``flag`` (1 = discarded by a reference method, e.g. the widefield one),
matched to the clusters by position. ``agree`` pools the per-file tables: 2 x 2 table, Cohen's kappa with a
bootstrap interval over files, sensitivity with an exact interval, positive and negative agreement.

Origin: [own sim]. Simplification stated: the simulated cluster centres are the centroids of the planted
localizations (no DBSCAN step), so merged or split clusters are not simulated.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tools import mps_unanchored as mu  # noqa: E402

THRESHOLDS: Tuple[int, ...] = (250, 300, 350, 400, 450, 500, 550)
SHAPES: Tuple[str, ...] = ("circle900", "circle1500", "circle2200", "ellipse", "lobed")
SIGMAS: Tuple[float, ...] = (30.0, 60.0, 100.0, 150.0)          # 150: stress, reported apart
BAYS: Tuple[str, ...] = ("none", "300x40", "500x40", "500x20", "800x20")   # 800x20: stress
N_INTERIOR: Tuple[int, ...] = (0, 3, 8, 15)
LUMP_FRACTIONS: Tuple[float, ...] = (0.0, 0.05, 0.15)
LOCS_PER_CLUSTER: Tuple[int, ...] = (25, 80)
GAP_FRACTIONS: Tuple[float, ...] = (0.0, 0.25)
DEPTH_RANGE_NM: Tuple[float, float] = (150.0, 800.0)
SPACING_NM: float = 200.0
STRESS_SIGMA: float = 150.0
STRESS_BAY: str = "800x20"


# ----------------------------------------------------------------------------------------------------
# Simulated axon
# ----------------------------------------------------------------------------------------------------
def _radius_fn(shape: str, bay: str, rotation: float) -> Any:
    def base(t: np.ndarray) -> np.ndarray:
        if shape.startswith("circle"):
            return np.full_like(t, float(shape[6:]))
        if shape == "ellipse":
            a, b = 2000.0, 1100.0
            return a * b / np.sqrt((b * np.cos(t)) ** 2 + (a * np.sin(t)) ** 2)
        if shape == "lobed":
            return 1500.0 * (1.0 + 0.12 * np.cos(3.0 * t))
        raise ValueError(shape)

    def fn(t: np.ndarray) -> np.ndarray:
        r = base(t)
        if bay != "none":
            depth, width = (float(v) for v in bay.split("x"))
            w = math.radians(width) / 2.0
            u = np.angle(np.exp(1j * (t - rotation)))
            r = r - depth * np.where(np.abs(u) < w, 0.5 * (1.0 + np.cos(np.pi * u / w)), 0.0)
        return r
    return fn


def _curve(shape: str, bay: str, rotation: float, n: int = 4000) -> np.ndarray:
    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = _radius_fn(shape, bay, rotation)(t)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def simulate_axon(seed: int) -> Dict[str, Any]:
    """One simulated axon: its drawn factors, the slab localizations, the cluster centres, their truth (1 = interior)
    and their true depth below the TRUE membrane (nm; nan for membrane clusters)."""
    rng = np.random.default_rng(seed)
    f = dict(shape=str(rng.choice(SHAPES)), sigma=float(rng.choice(SIGMAS)), bay=str(rng.choice(BAYS)),
             n_interior=int(rng.choice(N_INTERIOR)), lump_fraction=float(rng.choice(LUMP_FRACTIONS)),
             locs_per_cluster=int(rng.choice(LOCS_PER_CLUSTER)), gap_fraction=float(rng.choice(GAP_FRACTIONS)),
             grouped=int(rng.integers(0, 2)))
    c = _curve(f["shape"], f["bay"], float(rng.uniform(-np.pi, np.pi)))
    nxt = np.roll(c, -1, axis=0)
    seg = np.hypot(*(nxt - c).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(cum[-1])
    tang = nxt - np.roll(c, 1, axis=0)
    tang /= np.hypot(*tang.T)[:, None]
    outward = np.column_stack([tang[:, 1], -tang[:, 0]])

    def at(s: float) -> int:
        return int(np.searchsorted(cum, s % L, side="right") - 1) % len(c)

    # two gaps of membrane, each half of the gap fraction of the arc
    gaps = []
    if f["gap_fraction"] > 0:
        g0 = rng.uniform(0, L)
        half = 0.5 * f["gap_fraction"] * L
        gaps = [(g0, half), (g0 + 0.5 * L, half)]

    def in_gap(s: np.ndarray) -> np.ndarray:
        out = np.zeros(np.shape(s), dtype=bool)
        for g, w in gaps:
            out |= ((np.asarray(s) - g) % L) < w
        return out

    xs: List[np.ndarray] = []
    cents: List[np.ndarray] = []
    truth: List[int] = []
    n_cl = max(6, int(L // SPACING_NM))
    s0 = rng.uniform(0, SPACING_NM)
    for k in range(n_cl):
        s = (s0 + k * L / n_cl + rng.normal(0, 0.1 * SPACING_NM)) % L
        if bool(in_gap(np.array([s]))[0]):
            continue
        i = at(s)
        centre = c[i] + outward[i] * rng.normal(0, f["sigma"])
        p = centre + rng.normal(0, 15.0, (max(10, rng.poisson(f["locs_per_cluster"])), 2))
        xs.append(p); cents.append(p.mean(axis=0)); truth.append(0)
    n_mem_locs = int(sum(len(p) for p in xs))
    # diffuse localizations on the membrane (a third of the clustered ones)
    sd = rng.uniform(0, L, n_mem_locs // 3)
    sd = sd[~in_gap(sd)]
    i = np.array([at(s) for s in sd], dtype=int)
    if i.size:
        xs.append(c[i] + outward[i] * rng.normal(0, max(40.0, f["sigma"]), i.size)[:, None])

    # interior clusters at known depths below the true membrane
    true_depth: List[float] = [float("nan")] * len(cents)
    s_group = rng.uniform(0, L)
    made, tries = 0, 0
    while made < f["n_interior"] and tries < 400:
        tries += 1
        d = rng.uniform(*DEPTH_RANGE_NM)
        s = (s_group + rng.normal(0, 300.0)) if f["grouped"] else rng.uniform(0, L)
        i = at(s)
        centre = c[i] - outward[i] * d
        real = float(mu.signed_depth(centre[None, :], c)[0])
        if not (real >= 0.9 * d and real >= DEPTH_RANGE_NM[0]):
            continue                                   # too deep for this section: it came out near the far side
        p = centre + rng.normal(0, 15.0, (max(10, rng.poisson(f["locs_per_cluster"])), 2))
        xs.append(p); cents.append(p.mean(axis=0)); truth.append(1)
        true_depth.append(float(mu.signed_depth(p.mean(axis=0)[None, :], c)[0]))
        made += 1

    # diffuse interior lumps, too small to be clusters (8 localizations each)
    n_lump_locs = int(f["lump_fraction"] * sum(len(p) for p in xs))
    lo, hi = c.min(axis=0), c.max(axis=0)
    made_l, tries = 0, 0
    while made_l < n_lump_locs // 8 and tries < 20 * (n_lump_locs // 8 + 1):
        tries += 1
        q = rng.uniform(lo, hi)
        if float(mu.signed_depth(q[None, :], c)[0]) < DEPTH_RANGE_NM[0]:
            continue
        xs.append(q + rng.normal(0, 20.0, (8, 2)))
        made_l += 1
    xy = np.concatenate(xs)
    return dict(factors=f, x=xy[:, 0], y=xy[:, 1], centroids=np.asarray(cents), truth=np.asarray(truth, dtype=int),
                true_depth=np.asarray(true_depth), curve=c)


def score_axon(sim: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Depth of every cluster under both rings (the programme's functions), and the inner-band arc of the
    star-shape test at every threshold."""
    x, y, cen = sim["x"], sim["y"], sim["centroids"]
    axon: Dict[str, Any] = dict(sim["factors"])
    axon.update(n_locs=int(x.size), n_clusters=int(cen.shape[0]), n_true_interior=int(sim["truth"].sum()))
    depth: Dict[str, np.ndarray] = {}
    for kind, short in ((mu.REFERENCE_POLAR, "polar"), (mu.REFERENCE_SMOOTH, "smooth")):
        ring = mu.reference_ring(kind, x, y, cen, check_star=False)
        axon[f"{short}_status"] = ring.status
        if not ring.ok or ring.polygon is None:
            depth[short] = np.full(cen.shape[0], np.nan)
            for t in THRESHOLDS:
                axon[f"{short}_arc_{t}"] = float("nan")
            continue
        depth[short] = mu.signed_depth(cen, ring.polygon)
        mem = sim["truth"] == 0
        # error of the ring itself: how far the true membrane clusters' ideal positions sit from it
        axon[f"{short}_ring_bias_nm"] = round(float(np.median(depth[short][mem])), 1) if mem.any() else float("nan")
        for t in THRESHOLDS:
            axon[f"{short}_arc_{t}"] = round(float(mu.inner_band_arc_nm(x, y, ring.polygon, ring.centre,
                                                                         exclude_nm=float(t))), 1)
    rows = []
    for k in range(cen.shape[0]):
        rows.append(dict(k=k, truth=int(sim["truth"][k]),
                         true_depth_nm=round(float(sim["true_depth"][k]), 1) if sim["truth"][k] else "",
                         depth_polar_nm=round(float(depth["polar"][k]), 1),
                         depth_smooth_nm=round(float(depth["smooth"][k]), 1)))
    return rows, axon


def _write(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _read(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cmd_sim(args: argparse.Namespace) -> int:
    a, b = (int(v) for v in args.seeds.split(":"))
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"sim_{a:05d}_{b:05d}")
    if os.path.exists(stem + "_axons.csv"):
        print("already done:", stem)
        return 0
    t0 = time.time()
    crows: List[Dict[str, Any]] = []
    arows: List[Dict[str, Any]] = []
    for seed in range(a, b):
        sim = simulate_axon(seed)
        rows, axon = score_axon(sim)
        axon = dict(seed=seed, **axon)
        arows.append(axon)
        crows.extend(dict(seed=seed, **r) for r in rows)
        if time.time() - t0 > args.max_seconds:
            print(f"time limit: stopped after seed {seed}")
            break
    _write(stem + "_clusters.csv", crows)
    _write(stem + "_axons.csv", arows)
    print(f"{len(arows)} axons, {len(crows)} clusters, {time.time() - t0:.0f} s -> {stem}")
    return 0


# ----------------------------------------------------------------------------------------------------
# Error curves and the criterion
# ----------------------------------------------------------------------------------------------------
def _f(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_sim(out: str) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, np.ndarray]]]:
    axons: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(os.path.join(out, "sim_*_axons.csv"))):
        axons.extend(_read(p))
    per: Dict[int, Dict[str, List[float]]] = {}
    for p in sorted(glob.glob(os.path.join(out, "sim_*_clusters.csv"))):
        for r in _read(p):
            d = per.setdefault(int(r["seed"]), dict(truth=[], true=[], polar=[], smooth=[]))
            d["truth"].append(float(r["truth"])); d["true"].append(_f(r["true_depth_nm"]))
            d["polar"].append(_f(r["depth_polar_nm"])); d["smooth"].append(_f(r["depth_smooth_nm"]))
    return axons, {s: {k: np.asarray(v) for k, v in d.items()} for s, d in per.items()}


def axon_counts(axon: Dict[str, Any], cl: Dict[str, np.ndarray], ref: str, t: int, *,
                min_true_depth: float = 0.0, star: bool = True) -> Tuple[int, int, int, int, int]:
    """(false positives, membrane clusters, false negatives, interior clusters, rejected) of one axon at ``t``."""
    mem = cl["truth"] == 0
    inter = (cl["truth"] == 1) & (np.nan_to_num(cl["true"], nan=0.0) >= min_true_depth)
    arc = _f(axon.get(f"{ref}_arc_{t}", "nan"))
    rejected = axon.get(f"{ref}_status") != mu.STATUS_OK or (star and arc > mu.INNER_BAND_MAX_ARC_NM)
    if rejected:
        return 0, int(mem.sum()), int(inter.sum()), int(inter.sum()), 1
    flag = cl[ref] > t
    return int((flag & mem).sum()), int(mem.sum()), int((~flag & inter).sum()), int(inter.sum()), 0


def is_main(axon: Dict[str, Any]) -> bool:
    return float(axon["sigma"]) < STRESS_SIGMA and axon["bay"] != STRESS_BAY


def curve(axons: Sequence[Dict[str, Any]], per: Dict[int, Dict[str, np.ndarray]], ref: str, *, prevalence: float,
          cost_fp: float = 1.0, min_true_depth: float = 0.0, star: bool = True, n_boot: int = 1000,
          seed: int = 0) -> List[Dict[str, Any]]:
    """FPR, FNR and E per threshold with bootstrap intervals over whole axons."""
    cnt = np.array([[axon_counts(a, per[int(a["seed"])], ref, t, min_true_depth=min_true_depth, star=star)
                     for t in THRESHOLDS] for a in axons], dtype=float)          # (axons, thresholds, 5)

    def rates(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = c.sum(axis=0)
        fpr = s[:, 0] / np.maximum(s[:, 1], 1)
        fnr = s[:, 2] / np.maximum(s[:, 3], 1)
        return fpr, fnr, prevalence * fnr + (1 - prevalence) * cost_fp * fpr

    fpr, fnr, e = rates(cnt)
    rng = np.random.default_rng(seed)
    boots = np.array([np.stack(rates(cnt[rng.integers(0, len(axons), len(axons))])) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    i400 = THRESHOLDS.index(400)
    out = []
    for j, t in enumerate(THRESHOLDS):
        d = boots[:, 2, i400] - boots[:, 2, j]
        out.append(dict(reference=ref, threshold_nm=t, n_axons=len(axons), n_membrane=int(cnt[:, j, 1].sum()),
                        n_interior=int(cnt[:, j, 3].sum()), n_rejected=int(cnt[:, j, 4].sum()),
                        fpr=round(float(fpr[j]), 5), fpr_lo=round(float(lo[0, j]), 5), fpr_hi=round(float(hi[0, j]), 5),
                        fnr=round(float(fnr[j]), 4), fnr_lo=round(float(lo[1, j]), 4), fnr_hi=round(float(hi[1, j]), 4),
                        E=round(float(e[j]), 5), E_lo=round(float(lo[2, j]), 5), E_hi=round(float(hi[2, j]), 5),
                        E400_minus_E=round(float(e[i400] - e[j]), 5),
                        E400_minus_E_lo=round(float(np.percentile(d, 2.5)), 5),
                        E400_minus_E_hi=round(float(np.percentile(d, 97.5)), 5)))
    return out


def cmd_summarize(args: argparse.Namespace) -> int:
    axons, per = load_sim(args.out)
    axons = [a for a in axons if int(a["seed"]) in per]
    main = [a for a in axons if is_main(a)]
    print(f"{len(axons)} simulated axons, {len(main)} in the main set (sigma < {STRESS_SIGMA:g}, no {STRESS_BAY} bay)")
    rows: List[Dict[str, Any]] = []

    def add(label: str, subset: Sequence[Dict[str, Any]], **kw: Any) -> None:
        if len(subset) < 5:
            return
        for ref in ("polar", "smooth"):
            for r in curve(subset, per, ref, **kw):
                rows.append(dict(analysis=label, **r))

    p = float(args.prevalence)
    add("main", main, prevalence=p)
    add("main, prevalence 0.02", main, prevalence=0.02, n_boot=200)
    add("main, prevalence 0.10", main, prevalence=0.10, n_boot=200)
    add("main, false discard costs 3x", main, prevalence=p, cost_fp=3.0, n_boot=200)
    add("main, interior >= 250 nm deep", main, prevalence=p, min_true_depth=250.0, n_boot=200)
    add("main, star-shape test off", main, prevalence=p, star=False, n_boot=200)
    add("all (with stress)", axons, prevalence=p, n_boot=200)
    for key, values in (("sigma", SIGMAS), ("bay", BAYS), ("shape", SHAPES), ("lump_fraction", LUMP_FRACTIONS),
                        ("gap_fraction", GAP_FRACTIONS), ("locs_per_cluster", LOCS_PER_CLUSTER),
                        ("n_interior", N_INTERIOR), ("grouped", (0, 1))):
        for v in values:
            pool = axons if (key == "sigma" and float(v) >= STRESS_SIGMA) or v == STRESS_BAY else main
            sub = [a for a in pool if (a[key] == v if isinstance(v, str) else abs(float(a[key]) - float(v)) < 1e-9)]
            if key == "sigma" and float(v) >= STRESS_SIGMA:
                sub = [a for a in sub if a["bay"] != STRESS_BAY]
            if v == STRESS_BAY:
                sub = [a for a in sub if float(a["sigma"]) < STRESS_SIGMA]
            add(f"{key} = {v}", sub, prevalence=p, n_boot=200)
    _write(os.path.join(args.out, "summary_curves.csv"), rows)

    # detection against the TRUE depth of the interior clusters (main set, polar ring)
    edges = [150, 250, 300, 350, 400, 450, 500, 550, 650, 800]
    drow: List[Dict[str, Any]] = []
    for ref in ("polar", "smooth"):
        for lo, hi in zip(edges[:-1], edges[1:]):
            r: Dict[str, Any] = dict(reference=ref, true_depth_from=lo, true_depth_to=hi)
            n = 0
            hit = {t: 0 for t in THRESHOLDS}
            err: List[float] = []
            for a in main:
                cl = per[int(a["seed"])]
                m = (cl["truth"] == 1) & (cl["true"] >= lo) & (cl["true"] < hi)
                if not m.any():
                    continue
                n += int(m.sum())
                err.extend((cl[ref][m] - cl["true"][m]).tolist())
                for t in THRESHOLDS:
                    arc = _f(a.get(f"{ref}_arc_{t}", "nan"))
                    if a.get(f"{ref}_status") == mu.STATUS_OK and not arc > mu.INNER_BAND_MAX_ARC_NM:
                        hit[t] += int((cl[ref][m] > t).sum())
            r["n"] = n
            e = np.asarray(err)
            e = e[np.isfinite(e)]
            r["depth_error_median_nm"] = round(float(np.median(e)), 1) if e.size else ""
            r["depth_error_sd_nm"] = round(float(1.4826 * np.median(np.abs(e - np.median(e)))), 1) if e.size else ""
            for t in THRESHOLDS:
                r[f"detected_{t}"] = round(hit[t] / n, 4) if n else ""
            drow.append(r)
    _write(os.path.join(args.out, "summary_detection_by_depth.csv"), drow)

    # measured depth of the membrane clusters (the null the threshold must clear)
    mrow: List[Dict[str, Any]] = []
    for ref in ("polar", "smooth"):
        for sig in SIGMAS:
            v = np.concatenate([per[int(a["seed"])][ref][per[int(a["seed"])]["truth"] == 0] for a in axons
                                if abs(float(a["sigma"]) - sig) < 1e-9 and a["bay"] != STRESS_BAY] or [np.zeros(0)])
            v = v[np.isfinite(v)]
            if v.size:
                q = np.percentile(v, [50, 95, 99, 99.9])
                mrow.append(dict(reference=ref, sigma=sig, n=int(v.size), median=round(float(q[0]), 1),
                                 p95=round(float(q[1]), 1), p99=round(float(q[2]), 1), p999=round(float(q[3]), 1),
                                 max=round(float(v.max()), 1)))
    _write(os.path.join(args.out, "summary_membrane_depth.csv"), mrow)
    for r in rows:
        if r["analysis"] == "main":
            print(r["reference"], r["threshold_nm"], "FPR", r["fpr"], "FNR", r["fnr"], "E", r["E"],
                  "rejected", r["n_rejected"])
    return 0


# ----------------------------------------------------------------------------------------------------
# Real data: one file per process, read only
# ----------------------------------------------------------------------------------------------------
def _summary(an: Any) -> Dict[str, Any]:
    def g(name: str) -> Any:
        v = getattr(an, name, None)
        return round(float(v), 3) if v is not None else ""
    return dict(n=int(an.n_clusters_kept), perimeter_um=g("perimeter_um"), nn1_nm=g("median_1nn_nm"),
                occupancy_pct=g("occupancy_percent"))


def cmd_real(args: argparse.Namespace) -> int:
    from tools import mps_analysis as ma
    from tools.mps_io import load_localizations
    os.makedirs(args.out, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.tag)
    loc = load_localizations(args.file)
    an = ma.analyze_axon(loc.x_nm, loc.y_nm, loc.z_nm, pixel_size_nm=loc.pixel_size_nm, run_randomization=False)
    del loc
    cen = np.asarray(an.centroids, dtype=float).reshape(-1, 2)
    res = mu.auto_for_analysis(an, noise_replicates=mu.NOISE_REPLICATES)
    ref_flag = np.full(cen.shape[0], -1, dtype=int)
    if args.reference_flags:
        ref = [(float(r["cx"]), float(r["cy"]), int(float(r["flag"]))) for r in _read(args.reference_flags)
               if not args.reference_tag or r.get(args.reference_tag_column, "") == args.reference_tag]
        if ref:
            pts = np.array([[a, b] for a, b, _ in ref])
            for k in range(cen.shape[0]):
                d = np.hypot(*(pts - cen[k]).T)
                j = int(np.argmin(d))
                if d[j] <= mu.MANUAL_TOLERANCE_NM:
                    ref_flag[k] = ref[j][2]
    other = res.other
    rows = []
    for k in range(cen.shape[0]):
        rows.append(dict(tag=args.tag, k=k, cx=round(float(cen[k, 0]), 1), cy=round(float(cen[k, 1]), 1),
                         depth_polar_nm=round(float(res.used.depth_nm[k]), 1),
                         depth_sd_nm=round(float(res.used.depth_sd_nm[k]), 1),
                         depth_smooth_nm=round(float(other.depth_nm[k]), 1) if other is not None else "",
                         reference_flag=int(ref_flag[k])))
    _write(os.path.join(args.out, f"real_{safe}_clusters.csv"), rows)
    base = _summary(an)
    arow: Dict[str, Any] = dict(tag=args.tag, n_clusters=cen.shape[0], n_slab_locs=int(np.asarray(an.x_slab).size),
                                status=res.status, failure=res.failure,
                                noise_nm=round(res.used.noise_nm, 1),
                                smooth_status=other.ring.status if other is not None else "",
                                **{f"all_{k}": v for k, v in base.items()})
    depth = np.asarray(res.used.depth_nm, dtype=float)
    for t in args.thresholds:
        ring = mu.reference_ring(mu.REFERENCE_POLAR, an.x_slab, an.y_slab, cen, exclude_nm=float(t))
        arow[f"arc_{t}"] = round(float(ring.inner_band_arc_nm), 1) if math.isfinite(ring.inner_band_arc_nm) else ""
        arow[f"ring_ok_{t}"] = int(ring.ok)
        flag = np.nan_to_num(depth, nan=-1e9) > float(t) if ring.ok else np.zeros(cen.shape[0], dtype=bool)
        arow[f"n_excl_{t}"] = int(flag.sum())
        s = base
        if flag.any() and (~flag).sum() >= 3:
            try:
                s = _summary(ma.without_clusters(an, flag, margin_nm=float(t), source="automatic"))
            except Exception as exc:  # noqa: BLE001 - reported in the row, the other thresholds still run
                arow[f"error_{t}"] = f"{type(exc).__name__}: {exc}"[:120]
        for k, v in s.items():
            arow[f"t{t}_{k}"] = v
    _write(os.path.join(args.out, f"real_{safe}_axon.csv"), [arow])
    print(arow)
    return 0


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def exact_interval(k: int, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Clopper-Pearson interval of a proportion."""
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def cmd_agree(args: argparse.Namespace) -> int:
    rows: List[Dict[str, str]] = []
    for p in sorted(glob.glob(os.path.join(args.out, "real_*_clusters.csv"))):
        rows.extend(_read(p))
    rows = [r for r in rows if int(r["reference_flag"]) >= 0 and math.isfinite(_f(r["depth_polar_nm"]))]
    if args.only_tags:
        keep = set(args.only_tags.split(","))
        rows = [r for r in rows if r["tag"] in keep]
    tags = np.array([r["tag"] for r in rows])
    ref = np.array([int(r["reference_flag"]) == 1 for r in rows])
    out = []
    rng = np.random.default_rng(0)
    uniq = sorted(set(tags.tolist()))
    index = {t: np.flatnonzero(tags == t) for t in uniq}
    for col in ("depth_polar_nm", "depth_smooth_nm"):
        depth = np.array([_f(r[col]) for r in rows])
        for t in args.thresholds:
            rule = np.nan_to_num(depth, nan=-1e9) > t
            tp, fp = int((rule & ref).sum()), int((rule & ~ref).sum())
            fn, tn = int((~rule & ref).sum()), int((~rule & ~ref).sum())
            ks = []
            for _ in range(2000):
                pick = np.concatenate([index[u] for u in rng.choice(uniq, len(uniq))])
                v = kappa(rule[pick], ref[pick])
                if math.isfinite(v):
                    ks.append(v)
            lo, hi = (np.percentile(ks, [2.5, 97.5]) if ks else (float("nan"), float("nan")))
            slo, shi = exact_interval(tp, tp + fn) if tp + fn else (float("nan"), float("nan"))
            out.append(dict(depth=col, threshold_nm=t, n_files=len(uniq), n=len(rows), both=tp, rule_only=fp,
                            reference_only=fn, neither=tn, kappa=round(kappa(rule, ref), 3),
                            kappa_lo=round(float(lo), 3), kappa_hi=round(float(hi), 3),
                            sensitivity=round(tp / (tp + fn), 3) if tp + fn else "",
                            sens_lo=round(slo, 3), sens_hi=round(shi, 3),
                            positive_agreement=round(2 * tp / (2 * tp + fp + fn), 3) if tp + fp + fn else "",
                            negative_agreement=round(2 * tn / (2 * tn + fp + fn), 3) if tn + fp + fn else "",
                            raw_agreement=round((tp + tn) / len(rows), 4)))
    _write(os.path.join(args.out, args.name), out)
    for r in out:
        print(r)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.add_argument("--out", required=True); s.add_argument("--seeds", default="0:100")
    s.add_argument("--max-seconds", type=float, default=480.0); s.set_defaults(fn=cmd_sim)
    s = sub.add_parser("summarize"); s.add_argument("--out", required=True)
    s.add_argument("--prevalence", type=float, default=0.05); s.set_defaults(fn=cmd_summarize)
    s = sub.add_parser("real"); s.add_argument("--file", required=True); s.add_argument("--tag", required=True)
    s.add_argument("--out", required=True); s.add_argument("--reference-flags", default="")
    s.add_argument("--reference-tag", default=""); s.add_argument("--reference-tag-column", default="tag")
    s.add_argument("--thresholds", type=int, nargs="+", default=[250, 300, 400, 500]); s.set_defaults(fn=cmd_real)
    s = sub.add_parser("agree"); s.add_argument("--out", required=True); s.add_argument("--only-tags", default="")
    s.add_argument("--name", default="agreement.csv")
    s.add_argument("--thresholds", type=int, nargs="+", default=[250, 300, 400, 500]); s.set_defaults(fn=cmd_agree)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
