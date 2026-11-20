"""
Validation of the automatic contour of the perimeter density analysis (package P3).

    python validate_auto_contour.py sweep --param sigma_nm [--seeds 6] --out DIR
    python validate_auto_contour.py v1 --condition base_bg15 [--seeds 20] --out DIR
    python validate_auto_contour.py real --file PATH.hdf5 --out DIR      (one file per process)
    python validate_auto_contour.py real-list --root DIR
    python validate_auto_contour.py calib --condition real100 [--seeds 6] --out DIR   (each curve type alone)
    python validate_auto_contour.py trim --condition real100 [--seeds 6] --out DIR    (D8: with / without trimming)

Simulation (``simulate_ring``): this package's own minimal ring simulator, used until the shared
simulator of the analysis is merged. Five closed shapes (ellipse, wavy, star, "C", two lobes) with a
known membrane curve; clusters of localizations along it, optional uniform background, a dominant
cluster, an interior lump and a 2 um gap. The error of a fitted curve is measured only on the true
curve where it has data (a true point whose fifth-nearest localization is within 100 nm): in a gap the
curve is an interpolation and has no ground truth to be judged against.

Real data: one file per process, read-only; nothing is copied. The output (a small CSV row and a PNG
per axon) goes to the directory given, outside the repository, and every row says
"real data, pilot (exploratory)". There is no ground truth on real axons: the script reports
DIFFERENCES between curves (density nodes, polar, the old software's centroid polygon), never an error.

The PNG is written with numpy + zlib (no Qt): the cloud as a grey 2D histogram, the accepted curve in
blue, the polar curve in orange, the old centroid polygon in green, a rejected curve in red, the final
nodes as black dots.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools import mps_auto_contour as ac  # noqa: E402
from tools.mps_perimeter_types import Status  # noqa: E402

SHAPES: Tuple[str, ...] = ("ellipse", "wavy", "star", "C", "bilobe")
KIND_MEMBRANE, KIND_BACKGROUND, KIND_LUMP, KIND_INTERIOR = 0, 1, 2, 3   # SimRing.kind
CONCAVE_SHAPES: Tuple[str, ...] = ("bay500", "bay700", "star30")   # review R1: concavities tighter than 250 nm


# ----------------------------------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------------------------------
def _dense(poly: NDArray[np.float64], step: float = 5.0) -> NDArray[np.float64]:
    nxt = np.roll(poly, -1, axis=0)
    out = []
    for a, b in zip(poly, nxt):
        n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / step)))
        t = np.arange(n) / n
        out.append(a + t[:, None] * (b - a))
    return np.vstack(out)


def shape_polygon(name: str) -> NDArray[np.float64]:
    """True membrane polygon (counter-clockwise), nm, centred near the origin."""
    t = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    if name == "circle":
        return np.column_stack([1300.0 * np.cos(t), 1300.0 * np.sin(t)])
    if name == "ellipse":
        return np.column_stack([1500.0 * np.cos(t), 1000.0 * np.sin(t)])
    if name == "wavy":
        r = 1300.0 * (1 + 0.10 * np.cos(2 * t) + 0.08 * np.cos(3 * t + 1) + 0.06 * np.cos(5 * t + 2) + 0.05 * np.cos(7 * t))
        return np.column_stack([r * np.cos(t), r * np.sin(t)])
    if name == "star":
        r = 1500.0 * (1 + 0.15 * np.cos(5 * t))
        return np.column_stack([r * np.cos(t), r * np.sin(t)])
    if name == "bilobe":
        a, b = 1250.0, 1350.0                  # Cassini oval with a neck ~1 um wide: two lobes
        r2 = a * a * np.cos(2 * t) + np.sqrt(b ** 4 - a ** 4 * np.sin(2 * t) ** 2)
        r = np.sqrt(r2)
        return np.column_stack([r * np.cos(t), r * np.sin(t)])
    if name.startswith("bay"):            # circle R 1500 nm with one smooth inward bay (depth d, angular sd 0.2 rad)
        dep = float(name[3:])
        r = 1500.0 - dep * np.exp(-0.5 * ((t - np.pi) / 0.2) ** 2)
        return np.column_stack([r * np.cos(t), r * np.sin(t)])
    if name == "star30":                  # five valleys of radius ~110 nm
        r = 1500.0 * (1 + 0.30 * np.cos(5 * t))
        return np.column_stack([r * np.cos(t), r * np.sin(t)])
    if name == "C":
        big_r, w, span = 2000.0, 600.0, math.radians(210)   # band 1200 nm wide on a 210 degree arc: not star-shaped
        a_ = np.linspace(-span / 2, span / 2, 300)
        outer = np.column_stack([(big_r + w) * np.cos(a_), (big_r + w) * np.sin(a_)])
        inner = np.column_stack([(big_r - w) * np.cos(a_[::-1]), (big_r - w) * np.sin(a_[::-1])])
        bb = np.linspace(0, np.pi, 60)[1:-1]
        c1 = np.array([big_r * np.cos(span / 2), big_r * np.sin(span / 2)])
        cap1 = c1 + w * np.column_stack([np.cos(span / 2 + bb), np.sin(span / 2 + bb)])
        c2 = np.array([big_r * np.cos(-span / 2), big_r * np.sin(-span / 2)])
        cap2 = c2 + w * np.column_stack([np.cos(-span / 2 + np.pi + bb), np.sin(-span / 2 + np.pi + bb)])
        return np.vstack([outer, cap1, inner, cap2])
    raise ValueError(name)


@dataclass(frozen=True, eq=False)
class SimRing:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    truth: NDArray[np.float64]           # dense true curve (N, 2), ~5 nm steps, counter-clockwise
    truth_arc: NDArray[np.float64]       # arc of each truth row
    length_nm: float
    support: NDArray[np.bool_]           # truth rows with data nearby
    dominant_arc: float                  # nan when none
    blob_centre: Optional[Tuple[float, float]]
    n_membrane: int
    gap_start_arc: float = float("nan")  # the true gap [start, start + gap_nm) along truth_arc
    gap_nm: float = 0.0
    kind: NDArray[np.int8] = field(default_factory=lambda: np.zeros(0, dtype=np.int8))   # KIND_* per localization


def simulate_ring(shape: str, seed: int, *, n_membrane: int = 8000, radial_sd_nm: float = 20.0, loc_sd_nm: float = 8.0,
                  background_frac: float = 0.15, dominant_frac: float = 0.0, blob_frac: float = 0.0,
                  gap_nm: float = 0.0, cluster_rate_per_um: float = 4.0, blob_depth_nm: float = 400.0,
                  interior_frac: float = 0.0) -> SimRing:
    """
    Localizations around the true curve of ``shape``. Clusters: centres Poisson along the arc at
    ``cluster_rate_per_um`` (>= 80 nm apart), sizes lognormal (median 150, sigma 0.7) rescaled so the
    clusters hold ``n_membrane`` localizations; each localization is 15 nm along and ``radial_sd_nm`` across
    the curve from its centre, plus ``loc_sd_nm`` isotropic precision. ``gap_nm``: no cluster in one arc of
    that length. ``dominant_frac``: one more cluster holding that fraction of n_membrane (40 nm along).
    ``blob_frac``: an interior lump (sd 80 nm) ``blob_depth_nm`` inside the curve. ``background_frac``:
    uniform over the bounding box + 300 nm. ``interior_frac``: spectrin off the membrane, uniform inside the
    curve (review R1: the real cloud is 2-6x wider than the simulated one and holds interior spectrin).
    Deterministic for a seed.
    """
    shape_key = SHAPES.index(shape) if shape in SHAPES else (100 + CONCAVE_SHAPES.index(shape) if shape in CONCAVE_SHAPES else 99)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), shape_key]))
    d = _dense(shape_polygon(shape))
    seg = np.hypot(*(np.roll(d, -1, axis=0) - d).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    tan = np.roll(d, -1, axis=0) - np.roll(d, 1, axis=0)
    tan /= np.hypot(*tan.T)[:, None]
    nor = np.column_stack([tan[:, 1], -tan[:, 0]])     # outward for a counter-clockwise curve
    k = int(rng.poisson(cluster_rate_per_um * total / 1000.0))
    s = np.sort(rng.uniform(0, total, k))
    if s.size:
        s = s[np.concatenate([[True], np.diff(s) > 80.0])]
    g0 = float(rng.uniform(0, total))
    if gap_nm > 0:
        s = s[((s - g0) % total) >= gap_nm]
    sizes = np.maximum(5.0, rng.lognormal(math.log(150.0), 0.7, s.size))
    sizes = np.maximum(3, np.round(sizes * n_membrane / max(sizes.sum(), 1.0))).astype(int)
    xs: List[NDArray[np.float64]] = []
    ys: List[NDArray[np.float64]] = []
    ks: List[NDArray[np.int8]] = []

    def put(sc: float, m: int, along: float, radial: float) -> None:
        ss = (sc + rng.normal(0, along, m)) % total
        i = np.clip(np.searchsorted(cum, ss) - 1, 0, d.shape[0] - 1)
        r = rng.normal(0, radial, m)
        p = d[i] + nor[i] * r[:, None] + rng.normal(0, loc_sd_nm, (m, 2))
        xs.append(p[:, 0])
        ys.append(p[:, 1])
        ks.append(np.full(m, KIND_MEMBRANE, dtype=np.int8))

    for sc, m in zip(s, sizes):
        put(float(sc), int(m), 15.0, radial_sd_nm)
    n_mem = int(sizes.sum())
    dom_arc = float("nan")
    if dominant_frac > 0:
        dom_arc = float((g0 + gap_nm + rng.uniform(500, max(600.0, total - gap_nm - 500))) % total)
        put(dom_arc, int(dominant_frac * n_mem), 40.0, radial_sd_nm)
    blob: Optional[Tuple[float, float]] = None
    if blob_frac > 0:
        j = int(rng.integers(0, d.shape[0]))
        c = d[j] - nor[j] * blob_depth_nm
        blob = (float(c[0]), float(c[1]))
        nb = int(blob_frac * n_mem)
        xs.append(c[0] + rng.normal(0, 80.0, nb))
        ys.append(c[1] + rng.normal(0, 80.0, nb))
        ks.append(np.full(nb, KIND_LUMP, dtype=np.int8))
    if background_frac > 0:
        nb = int(background_frac * n_mem)
        lo, hi = d.min(0) - 300, d.max(0) + 300
        xs.append(rng.uniform(lo[0], hi[0], nb))
        ys.append(rng.uniform(lo[1], hi[1], nb))
        ks.append(np.full(nb, KIND_BACKGROUND, dtype=np.int8))
    if interior_frac > 0:
        ni = int(interior_frac * n_mem)
        lo, hi = d.min(0), d.max(0)
        cand = np.column_stack([rng.uniform(lo[0], hi[0], 4 * ni + 50), rng.uniform(lo[1], hi[1], 4 * ni + 50)])
        inside = ac._points_in_polygon(cand[:, 0], cand[:, 1], d[::4])
        cand = cand[inside][:ni]
        xs.append(cand[:, 0])
        ys.append(cand[:, 1])
        ks.append(np.full(cand.shape[0], KIND_INTERIOR, dtype=np.int8))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mem = np.column_stack([np.concatenate(xs[:len(s)]), np.concatenate(ys[:len(s)])]) if len(s) else np.zeros((0, 2))
    if mem.shape[0] >= 5:
        dd, _ = cKDTree(mem).query(d, k=5)
        support = np.asarray(dd[:, -1] < 100.0)
    else:
        support = np.zeros(d.shape[0], dtype=bool)
    return SimRing(x=x, y=y, truth=d, truth_arc=cum[:-1], length_nm=total, support=support, dominant_arc=dom_arc,
                   blob_centre=blob, n_membrane=n_mem, gap_start_arc=g0 if gap_nm > 0 else float("nan"), gap_nm=float(gap_nm),
                   kind=np.concatenate(ks) if ks else np.zeros(0, dtype=np.int8))


def disk_cloud(seed: int, *, n: int = 9000, radius_nm: float = 1500.0) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Negative control: localizations uniform in a filled disk (no ring)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 777]))
    r = radius_nm * np.sqrt(rng.uniform(0, 1, n))
    t = rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(t), r * np.sin(t)


# ----------------------------------------------------------------------------------------------------
# Scoring against the truth
# ----------------------------------------------------------------------------------------------------
def score(sim: SimRing, details: ac.ContourDetails) -> Dict[str, float]:
    c = details.contour
    out: Dict[str, float] = {"ok": float(c.status is Status.OK), "n_nodes0": float(details.n_nodes_initial),
                             "n_nodes": float(c.nodes_nm.shape[0])}
    primary = details.candidates[0]
    out["primary_ok"] = float(primary.reason == "")
    out["primary_cross"] = float(primary.n_self_crossings)
    out["hull_ratio"] = float(primary.hull_ratio)
    out["lumen_ratio"] = float(primary.lumen_ratio)
    if c.path is None:
        return out
    tree = cKDTree(np.asarray(c.path.points_nm))
    e, _ = tree.query(sim.truth)
    sup = sim.support
    out["err_mean"] = float(np.mean(e[sup])) if sup.any() else float("nan")
    out["err_p95"] = float(np.percentile(e[sup], 95)) if sup.any() else float("nan")
    dc, near = cKDTree(sim.truth).query(np.asarray(c.path.points_nm))
    dc = dc[sup[near]] if sup[near].any() else dc          # curve rows facing a gap are an interpolation: not judged
    out["curve_p95"] = float(np.percentile(dc, 95))          # curve -> truth: catches excursions away from the membrane
    out["curve_max"] = float(np.max(dc))
    out["err_gap_max"] = float(np.max(e[~sup])) if (~sup).any() else float("nan")
    out["len_ratio"] = float(c.length_nm / sim.length_nm)
    out["polar_dist"] = float(c.polar_distance_nm)
    out["star"] = float(c.star_shaped)
    out["fallback"] = float(c.method != "density_nodes" or c.knot_nm != ac.CONTOUR_KNOT_NM)
    if math.isfinite(sim.dominant_arc):
        da = np.abs(sim.truth_arc - sim.dominant_arc)
        da = np.minimum(da, sim.length_nm - da)
        win = (da >= 450.0) & (da <= 550.0) & sup
        out["err_dominant_500"] = float(np.max(e[win])) if win.any() else float("nan")
    if sim.blob_centre is not None:
        nodes = c.nodes_nm
        bc = np.asarray(sim.blob_centre)
        out["blob_node_min_nm"] = float(np.min(np.hypot(*(nodes - bc).T))) if nodes.size else float("nan")
        dn, _ = cKDTree(sim.truth).query(nodes) if nodes.size else (np.zeros(0), None)
        out["nodes_off_truth_150"] = float(np.count_nonzero(np.asarray(dn) > 150.0))
    return out


# ----------------------------------------------------------------------------------------------------
# Batch runners
# ----------------------------------------------------------------------------------------------------
SWEEP: Dict[str, Sequence[float]] = {
    "sigma_nm": (25.0, 50.0, 75.0, 100.0),
    "min_separation_nm": (100.0, 200.0, 300.0, 400.0),
    "threshold_frac": (0.05, 0.15, 0.30),
    "knot_nm": (300.0, 400.0, 600.0),
    "node_drop_nm": (100.0, 150.0, 250.0),
}

CONDITIONS: Dict[str, Dict[str, float]] = {
    "base_bg0": dict(background_frac=0.0),
    "base_bg15": dict(background_frac=0.15),
    "base_bg30": dict(background_frac=0.30),
    "gap2um": dict(background_frac=0.15, gap_nm=2000.0),
    "dominant20": dict(background_frac=0.15, dominant_frac=0.20 / 0.80),   # 20 % of ALL membrane localizations
    "blob10": dict(background_frac=0.15, blob_frac=0.10),
    "wide50": dict(background_frac=0.15, radial_sd_nm=50.0),
    "low4000": dict(background_frac=0.15, n_membrane=4000),
    "low2000": dict(background_frac=0.15, n_membrane=2000),
    "low1000": dict(background_frac=0.15, n_membrane=1000),
    "hard": dict(background_frac=0.15, gap_nm=2000.0, dominant_frac=0.15, blob_frac=0.08),
    "hard_wide": dict(background_frac=0.15, gap_nm=2000.0, dominant_frac=0.15, blob_frac=0.08, radial_sd_nm=50.0),
    # review R1: a cloud like the real one (radial sd 75-150 nm, interior spectrin), and the gap at three backgrounds
    "real75": dict(background_frac=0.15, radial_sd_nm=75.0, interior_frac=0.10),
    "real100": dict(background_frac=0.15, radial_sd_nm=100.0, interior_frac=0.10),
    "real150": dict(background_frac=0.15, radial_sd_nm=150.0, interior_frac=0.10),
    "gap_bg0": dict(background_frac=0.0, gap_nm=2000.0),
    "gap_bg30": dict(background_frac=0.30, gap_nm=2000.0),
    "gap_real100": dict(background_frac=0.15, gap_nm=2000.0, radial_sd_nm=100.0, interior_frac=0.10),
    "blob_real100": dict(background_frac=0.15, blob_frac=0.10, radial_sd_nm=100.0, interior_frac=0.10),
}

CALIB_CONDITIONS: Tuple[str, ...] = ("base_bg15", "wide50", "real75", "real100", "real150", "gap2um", "gap_bg0", "gap_bg30",
                                     "gap_real100", "blob10", "blob_real100", "low1000", "hard_wide")


def curve_vs_truth(sim: SimRing, path: object) -> Dict[str, float]:
    """Errors of any fitted curve (accepted or not) against the truth, plus the lump and gap read-outs."""
    pts = np.asarray(getattr(path, "points_nm"))
    out: Dict[str, float] = {}
    e, _ = cKDTree(pts).query(sim.truth)
    sup = sim.support
    out["err_mean"] = float(np.mean(e[sup])) if sup.any() else float("nan")
    out["err_p95"] = float(np.percentile(e[sup], 95)) if sup.any() else float("nan")
    dc, near = cKDTree(sim.truth).query(pts)
    dcs = dc[sup[near]] if sup[near].any() else dc
    out["curve_p95"] = float(np.percentile(dcs, 95))
    out["len_ratio"] = float(getattr(path, "length_nm")) / sim.length_nm
    if sim.blob_centre is not None:
        out["blob_curve_min_nm"] = float(np.min(np.hypot(*(pts - np.asarray(sim.blob_centre)).T)))
    return out


def run_calibration(condition: str, seeds: int, out: str) -> None:
    """Each curve type ON ITS OWN (density nodes at 400 nm, at 600 nm, polar), with its quality, notch and
    support read-outs and its true error, to calibrate the quality gate and the support level."""
    rows: List[Dict[str, object]] = []
    t0 = time.time()
    variants = (("k400", dict(fallbacks=False, compute_polar=False)),
                ("k600", dict(knot_nm=600.0, fallbacks=False, compute_polar=False)),
                ("polar", dict(method="polar", fallbacks=False)))
    for shape in SHAPES + CONCAVE_SHAPES:
        for seed in range(seeds):
            sim = simulate_ring(shape, 3000 + seed, **CONDITIONS[condition])  # type: ignore[arg-type]
            for vname, kw in variants:
                t1 = time.time()
                det = ac.contour_details(sim.x, sim.y, **kw)
                cand = det.candidates[0]
                row: Dict[str, object] = dict(condition=condition, shape=shape, seed=seed, variant=vname, n_locs=sim.x.size,
                                              reason=cand.reason, seconds=round(time.time() - t1, 2))
                if cand.path is not None:
                    row.update({k: round(v, 4) for k, v in curve_vs_truth(sim, cand.path).items()})
                    q = ac.contour_quality(cand.path, sim.x, sim.y)
                    row.update(frac50=round(q.frac_within[50.0], 4), frac100=round(q.frac_within[100.0], 4),
                               frac150=round(q.frac_within[150.0], 4), robust_sd=round(q.robust_sd_nm, 1),
                               local_offset=round(q.max_local_offset_nm, 1),
                               off_band=round(q.off_band_fraction, 4),
                               **{f"pers{k}": round(ac.persistent_offset(q.window_offset_nm, k)[0], 1) for k in (2, 3, 4, 5)},
                               notch_r=round(min(ac.tightest_concave(cand.path)[0], 1e6), 1),
                               hull=round(cand.hull_ratio, 3))
                    if sim.gap_nm > 0:
                        for k in (0.01, 0.02, 0.03, 0.05, 0.08):
                            sp = ac.contour_support(cand.path, sim.x, sim.y, k=k)
                            # gap sensitivity: true gap rows whose nearest curve sample is unsupported (in a run)
                            qpts = np.column_stack([np.interp(sp.sample_arc_nm, np.asarray(cand.path.cum_nm),
                                                              np.append(np.asarray(cand.path.points_nm)[:, 0], np.asarray(cand.path.points_nm)[0, 0])),
                                                    np.interp(sp.sample_arc_nm, np.asarray(cand.path.cum_nm),
                                                              np.append(np.asarray(cand.path.points_nm)[:, 1], np.asarray(cand.path.points_nm)[0, 1]))])
                            in_run = np.zeros(sp.sample_arc_nm.size, dtype=bool)
                            for a0, a1 in sp.arcs_nm:
                                in_run |= ((sp.sample_arc_nm >= a0) & (sp.sample_arc_nm <= a1)) if a1 >= a0 else \
                                    ((sp.sample_arc_nm >= a0) | (sp.sample_arc_nm <= a1))
                            ingap = ((sim.truth_arc - sim.gap_start_arc) % sim.length_nm) < sim.gap_nm
                            _dd, nearest = cKDTree(qpts).query(sim.truth[ingap])
                            sens = float(np.mean(in_run[nearest])) if ingap.any() else float("nan")
                            _dd2, tnear = cKDTree(sim.truth).query(qpts[in_run]) if in_run.any() else (None, np.zeros(0, int))
                            # flagged samples facing true membrane WITH data (support) are the false ones
                            false_frac = float(np.mean(sim.support[tnear])) if in_run.any() else 0.0
                            row[f"gap_sens_k{k:g}"] = round(sens, 3)
                            row[f"gap_false_k{k:g}"] = round(false_frac, 3)
                            row[f"unsup_k{k:g}"] = round(sp.unsupported_fraction, 3)
                    else:
                        for k in (0.01, 0.02, 0.03, 0.05, 0.08):
                            row[f"unsup_k{k:g}"] = round(ac.contour_support(cand.path, sim.x, sim.y, k=k).unsupported_fraction, 3)
                rows.append(row)
        print(f"{condition} {shape}: done ({time.time() - t0:.0f} s)", flush=True)
    _write_rows(os.path.join(out, f"calib_{condition}.csv"), rows)


SWEEP_CONDITIONS: Tuple[str, ...] = ("base_bg15", "wide50", "gap2um", "dominant20", "blob10", "low2000")


def _write_rows(path: str, rows: List[Dict[str, object]]) -> None:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    new = not os.path.exists(path)
    if not new:
        with open(path, newline="", encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
        merged: List[Dict[str, object]] = [{k: v for k, v in r.items()} for r in old]
        rows = merged + rows
        for r in old:
            for k in r:
                if k not in keys:
                    keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_sweep(param: str, seeds: int, out: str) -> None:
    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for value in SWEEP[param]:
        for shape in SHAPES:
            for seed in range(seeds):
                cond = SWEEP_CONDITIONS[seed % len(SWEEP_CONDITIONS)]
                sim = simulate_ring(shape, seed, **CONDITIONS[cond])  # type: ignore[arg-type]
                kw: Dict[str, float] = {param: value}
                t1 = time.time()
                det = ac.contour_details(sim.x, sim.y, compute_polar=False, fallbacks=False, **kw)  # type: ignore[arg-type]
                sc = score(sim, det)
                row: Dict[str, object] = dict(param=param, value=value, shape=shape, seed=seed, condition=cond,
                                              n_locs=sim.x.size, seconds=round(time.time() - t1, 2),
                                              reason=det.candidates[0].reason)
                row.update({k: round(v, 4) for k, v in sc.items()})
                rows.append(row)
        print(f"{param}={value}: done ({time.time() - t0:.0f} s)", flush=True)
    _write_rows(os.path.join(out, f"sweep_{param}.csv"), rows)


def run_v1(condition: str, seeds: int, out: str, shapes: Sequence[str] = SHAPES) -> None:
    rows: List[Dict[str, object]] = []
    t0 = time.time()
    if condition == "disk":
        for seed in range(seeds):
            x, y = disk_cloud(seed)
            det = ac.contour_details(x, y)
            rows.append(dict(condition="disk", shape="disk", seed=seed, n_locs=x.size, ok=float(det.contour.status is Status.OK),
                             reason=ac.failure_reason(det.contour), primary_reason=det.candidates[0].reason,
                             hull_ratio=round(det.candidates[0].hull_ratio, 3), lumen_ratio=round(det.candidates[0].lumen_ratio, 3)))
    else:
        for shape in shapes:
            for seed in range(seeds):
                sim = simulate_ring(shape, 1000 + seed, **CONDITIONS[condition])  # type: ignore[arg-type]
                t1 = time.time()
                det = ac.contour_details(sim.x, sim.y)
                sc = score(sim, det)
                row: Dict[str, object] = dict(condition=condition, shape=shape, seed=seed, n_locs=sim.x.size,
                                              seconds=round(time.time() - t1, 2), reason=ac.failure_reason(det.contour),
                                              primary_reason=det.candidates[0].reason, method=det.contour.method,
                                              knot=det.contour.knot_nm)
                row.update({k: round(v, 4) for k, v in sc.items()})
                row.update(_flat(ac.assessment_row(det)))
                rows.append(row)
            print(f"{condition} {shape}: done ({time.time() - t0:.0f} s)", flush=True)
    _write_rows(os.path.join(out, f"v1_{condition}.csv"), rows)


def _flat(d: Dict[str, object], prefix: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in d.items():
        out[prefix + k] = round(v, 4) if isinstance(v, float) else v
    return out


def polar_trim_mask(x: NDArray[np.float64], y: NDArray[np.float64],
                    depth_limit_nm: float = ac.INTERIOR_TRIM_DEPTH_NM,
                    require_valid: bool = True) -> Tuple[Optional[NDArray[np.bool_]], str]:
    """The reference used to validate the optional interior trimming (decision D8), standing in for the
    unanchored-cluster package: depth inside the polar curve (P-spline on the polar contour) and
    ``ac.interior_exclusion``. ``require_valid``: when the polar curve itself is rejected (notch, leaves the
    cloud...) the reference is not valid and nothing is trimmed: returns (None, reason). False = trim against
    any polar curve that exists (what the anchoring pilot did), to measure what that costs."""
    det = ac.contour_details(x, y, method="polar", fallbacks=False)
    cand = det.candidates[0]
    if cand.path is None or (cand.reason and require_valid):
        return None, f"no trimming: polar reference rejected ({cand.reason or 'no curve'})"
    _s, r = ac.project(cand.path, x, y)
    return ac.interior_exclusion(-r, depth_limit_nm=depth_limit_nm), "trimmed against the polar reference"


TRIM_SHAPES: Tuple[str, ...] = SHAPES + CONCAVE_SHAPES + ("bay1000",)
TRIM_CONDITIONS: Tuple[str, ...] = ("base_bg15", "wide50", "real75", "real100", "real150", "blob_real100")


def run_trim(condition: str, seeds: int, out: str, shapes: Sequence[str] = TRIM_SHAPES, require_valid: bool = True) -> None:
    """D8 validation: the full chain without and with the interior trimming on the same cloud; how much TRUE
    membrane the trimming removes, how much interior / lump it removes, and what it does to the curve."""
    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for shape in shapes:
        for seed in range(seeds):
            sim = simulate_ring(shape, 4000 + seed, **CONDITIONS[condition])  # type: ignore[arg-type]
            row: Dict[str, object] = dict(condition=condition, shape=shape, seed=seed, n_locs=sim.x.size)
            t1 = time.time()
            base = ac.contour_details(sim.x, sim.y)
            row["base_seconds"] = round(time.time() - t1, 2)
            row.update(_flat({k: v for k, v in score(sim, base).items()}, "base_"))
            row.update(_flat(ac.assessment_row(base), "base_"))
            t1 = time.time()
            mask, why = polar_trim_mask(sim.x, sim.y, require_valid=require_valid)
            row["trim_note"] = why
            row["trim_applied"] = int(mask is not None)
            if mask is None:
                rows.append(row)
                continue
            mem = sim.kind == KIND_MEMBRANE
            inner = (sim.kind == KIND_INTERIOR) | (sim.kind == KIND_LUMP)
            row.update(n_excluded=int(mask.sum()),
                       membrane_lost_frac=round(float(np.mean(mask[mem])) if mem.any() else float("nan"), 5),
                       interior_removed_frac=round(float(np.mean(mask[inner])) if inner.any() else float("nan"), 4),
                       excluded_that_is_membrane=round(float(np.mean(mem[mask])) if mask.any() else float("nan"), 4))
            # where the lost membrane sits: arcs of the true curve with >= 20 % of their localizations removed
            if mem.any() and mask[mem].any():
                tree = cKDTree(sim.truth)
                _d, near = tree.query(np.column_stack([sim.x[mem], sim.y[mem]]))
                arc = sim.truth_arc[near]
                bins = np.floor(arc / 100.0).astype(int)
                nb = int(math.ceil(sim.length_nm / 100.0))
                tot = np.bincount(bins, minlength=nb)
                lost = np.bincount(bins, weights=mask[mem].astype(float), minlength=nb)
                with np.errstate(invalid="ignore", divide="ignore"):
                    frac_bin = np.where(tot > 0, lost / np.maximum(tot, 1), 0.0)
                row["membrane_arc_lost20_nm"] = int(np.count_nonzero(frac_bin >= 0.2)) * 100
            else:
                row["membrane_arc_lost20_nm"] = 0
            trim = ac.contour_details(sim.x, sim.y, exclude=mask)
            row["trim_seconds"] = round(time.time() - t1, 2)
            row.update(_flat({k: v for k, v in score(sim, trim).items()}, "trim_"))
            row.update(_flat(ac.assessment_row(trim), "trim_"))
            rows.append(row)
        print(f"trim {condition} {shape}: done ({time.time() - t0:.0f} s)", flush=True)
    _write_rows(os.path.join(out, f"trim_{condition}{'' if require_valid else '_anypolar'}.csv"), rows)


# ----------------------------------------------------------------------------------------------------
# Real data (read-only, one file per process)
# ----------------------------------------------------------------------------------------------------
# Folder of the real (unpublished) pilot files, READ ONLY. Set the environment variable MPS_DATA_DIR or pass --root;
# without it the real-data parts report that and do nothing (the simulated parts do not need it).
REAL_ROOT = os.environ.get("MPS_DATA_DIR", "")
REAL_LABEL = "real data, pilot (exploratory; one experiment; no ground truth)"
SLICE_NM = 170.0


def real_files(root: str = REAL_ROOT) -> List[str]:
    files: List[str] = []
    if not root or not os.path.isdir(root):
        print("real data: no folder given (set MPS_DATA_DIR or pass --root); nothing to read", file=sys.stderr)
        return files
    for d, _sub, fs in os.walk(root):
        if "Nueva carpeta" in d:
            continue
        for f in fs:
            if f.endswith(".hdf5") and "picked" in f.lower():
                files.append(os.path.join(d, f))
    return sorted(files)


def real_tag(path: str, root: str = REAL_ROOT) -> str:
    roi = os.path.relpath(path, root).split(os.sep)[0].replace("ROI ", "R")
    return roi + "/" + os.path.basename(path).lower().split("picked_")[1].replace(".hdf5", "")


def _png(path: str, rgb: NDArray[np.uint8]) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def render_png(path: str, x: NDArray[np.float64], y: NDArray[np.float64],
               curves: Sequence[Tuple[NDArray[np.float64], Tuple[int, int, int], bool]],
               nodes: NDArray[np.float64], px_nm: float = 10.0) -> None:
    """Grey histogram of the cloud plus closed curves (colour, dashed) and nodes; 10 nm per pixel."""
    lo = np.array([np.percentile(x, 0.2), np.percentile(y, 0.2)]) - 300
    hi = np.array([np.percentile(x, 99.8), np.percentile(y, 99.8)]) + 300
    w = int(min(1400, math.ceil((hi[0] - lo[0]) / px_nm)))
    h = int(min(1400, math.ceil((hi[1] - lo[1]) / px_nm)))
    px = max((hi[0] - lo[0]) / w, (hi[1] - lo[1]) / h)
    hist, _, _ = np.histogram2d(y, x, bins=[h, w], range=[[lo[1], lo[1] + h * px], [lo[0], lo[0] + w * px]])
    v = np.sqrt(hist)
    v = v / max(float(np.percentile(v[v > 0], 99.5)) if (v > 0).any() else 1.0, 1e-9)
    grey = (255 - np.clip(v, 0, 1) * 200).astype(np.uint8)
    img = np.repeat(grey[::-1, :, None], 3, axis=2)

    def put(cx: NDArray[np.float64], cy: NDArray[np.float64], col: Tuple[int, int, int], r: int) -> None:
        ci = ((cx - lo[0]) / px).astype(int)
        ri = (h - 1 - ((cy - lo[1]) / px)).astype(int)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                a, b = ri + dy, ci + dx
                ok = (a >= 0) & (a < h) & (b >= 0) & (b < w)
                img[a[ok], b[ok]] = col

    for pts, col, dashed in curves:
        p = np.vstack([pts, pts[:1]])
        dense = _dense(p[:-1], step=px / 2)
        if dashed:
            k = np.arange(dense.shape[0])
            dense = dense[(k // 12) % 2 == 0]
        put(dense[:, 0], dense[:, 1], col, 1)
    if nodes.size:
        put(nodes[:, 0], nodes[:, 1], (0, 0, 0), 3)
    _png(path, img)


def run_real(path: str, out: str) -> None:
    """One real axon (read-only): the contour on the three slices with the slice labels (D7), the same with the
    interior trimming against the polar reference (D8), and the central slice alone (comparison for D7; NOT the
    per-slice contour, which is a separate package). One CSV row and one PNG per axon."""
    from tools.mps_geometry import reconstruct_perimeter
    from tools.mps_io import load_localizations
    from tools.mps_membrane import _linear_closed_path
    from tools.mps_periodicity import fit_z_periodicity

    tag = real_tag(path)
    t0 = time.time()
    loc = load_localizations(path)
    x_all, y_all, z_all = (np.asarray(loc.x_nm, dtype=np.float64), np.asarray(loc.y_nm, dtype=np.float64),
                           np.asarray(loc.z_nm, dtype=np.float64))
    del loc
    gc.collect()
    z0 = float(fit_z_periodicity(z_all).main_peak_nm)
    lab_all = np.floor((z_all - z0) / SLICE_NM + 0.5).astype(np.int64)      # slice 0 = [z0 - T/2, z0 + T/2)
    keep = np.abs(lab_all) <= 1
    x, y, lab = x_all[keep], y_all[keep], lab_all[keep]
    del x_all, y_all, z_all, lab_all
    gc.collect()
    t_load = time.time() - t0
    row: Dict[str, object] = dict(label=REAL_LABEL, axon=tag, n_locs_3slices=int(x.size), z0_peak_nm=round(z0, 1),
                                  n_slice_m1=int(np.sum(lab == -1)), n_slice_0=int(np.sum(lab == 0)),
                                  n_slice_p1=int(np.sum(lab == 1)), load_seconds=round(t_load, 1))
    t1 = time.time()
    det = ac.contour_details(x, y, slice_label=lab)
    row["base_seconds"] = round(time.time() - t1, 1)
    c = det.contour
    row.update(status=c.status.value, method_used=c.method, knot_used=c.knot_nm, n_nodes_initial=det.n_nodes_initial,
               n_nodes_final=int(c.nodes_nm.shape[0]), length_um=round(c.length_nm / 1000.0, 3) if c.path is not None else "",
               polar_dist_nm=round(c.polar_distance_nm, 1), star_shaped=c.star_shaped)
    row.update(_flat(ac.assessment_row(det), "base_"))
    for i, cand in enumerate(det.candidates):
        row[f"cand{i}_label"] = cand.label
        row[f"cand{i}_reason"] = cand.reason
        if cand.path is not None:
            cq = ac.contour_quality(cand.path, x, y)
            row[f"cand{i}_frac100"] = round(cq.frac_within[100.0], 4)
            row[f"cand{i}_persistent_nm"] = round(cq.persistent_offset_nm, 1)
            row[f"cand{i}_off_band"] = round(cq.off_band_fraction, 4)
    # the slice misfit measured on the polar curve too (exists also when the contour is rejected)
    if det.polar_path is not None:
        sm = ac.slice_misfit(det.polar_path, x, y, lab)
        row.update(polar_slice_rms_minus_nm=round(sm.rms_displacement_nm[0], 1), polar_slice_rms_plus_nm=round(sm.rms_displacement_nm[1], 1),
                   polar_slice_radius_change_nm=round(sm.radius_change_per_slice_nm, 1),
                   polar_slice_frac100_central=round(sm.frac100_central, 4), polar_slice_frac100_all=round(sm.frac100_all, 4))
    # D8: interior trimming against the polar reference
    t1 = time.time()
    mask, why = polar_trim_mask(x, y)
    row["trim_note"] = why
    trim: Optional[ac.ContourDetails] = None
    if mask is not None:
        trim = ac.contour_details(x, y, exclude=mask, slice_label=lab)
        row.update(_flat(ac.assessment_row(trim), "trim_"))
        row["trim_status"] = trim.contour.status.value
        if trim.contour.path is not None and c.path is not None:
            row["trim_vs_base_nm"] = round(ac.curve_distance_nm(trim.contour.path, c.path), 1)
    row["trim_seconds"] = round(time.time() - t1, 1)
    # the same against any polar curve (what the anchoring pilot did), for comparison only
    trim_any: Optional[ac.ContourDetails] = None
    if mask is None:
        mask_any, why_any = polar_trim_mask(x, y, require_valid=False)
        if mask_any is not None:
            trim_any = ac.contour_details(x, y, exclude=mask_any, slice_label=lab)
    else:
        trim_any = trim
    if trim_any is not None:
        row.update(_flat(ac.assessment_row(trim_any), "trimany_"))
        row["trimany_status"] = trim_any.contour.status.value
    # D7 comparison: the central slice alone
    t1 = time.time()
    m0 = lab == 0
    cen = ac.contour_details(x[m0], y[m0])
    row.update(_flat(ac.assessment_row(cen), "central_"))
    row["central_status"] = cen.contour.status.value
    if cen.contour.path is not None:
        _s, r_all = ac.project(cen.contour.path, x, y)
        row["central_curve_frac100_all3"] = round(float(np.mean(np.abs(r_all) <= 100.0)), 4)
        if c.path is not None:
            row["central_vs_base_nm"] = round(ac.curve_distance_nm(cen.contour.path, c.path), 1)
    row["central_seconds"] = round(time.time() - t1, 1)
    # the old software's centroid polygon
    centres_csv = path[:-5] + "_cluster_centers.csv"
    old_poly: Optional[NDArray[np.float64]] = None
    if os.path.exists(centres_csv):
        cc = np.loadtxt(centres_csv, delimiter=",", ndmin=2)
        if cc.shape[0] >= 3:
            per = reconstruct_perimeter(cc[:, :2], all_starts=True)
            old_poly = per.contour
            row["old_centroid_perimeter_um"] = round(per.perimeter_nm / 1000.0, 3)
            if c.path is not None:
                row["dist_to_old_centroids_mean_nm"] = round(ac.curve_distance_nm(c.path, _linear_closed_path(old_poly)), 1)
    row["warnings"] = " | ".join(c.warnings)[:900]
    row["seconds_total"] = round(time.time() - t0, 1)
    curves: List[Tuple[NDArray[np.float64], Tuple[int, int, int], bool]] = []
    if old_poly is not None:
        curves.append((old_poly, (0, 150, 0), True))
    if det.polar_path is not None:
        curves.append((np.asarray(det.polar_path.points_nm), (230, 140, 0), False))
    if det.rejected_path is not None:
        curves.append((np.asarray(det.rejected_path.points_nm), (220, 0, 0), True))
    if trim is not None and trim.contour.path is not None:
        curves.append((np.asarray(trim.contour.path.points_nm), (200, 0, 200), True))
    if c.path is not None:
        curves.append((np.asarray(c.path.points_nm), (0, 70, 220), False))
    safe = tag.replace("/", "_")
    render_png(os.path.join(out, f"contour_{safe}.png"), x, y, curves, c.nodes_nm)
    _write_rows(os.path.join(out, "real_contours_v2.csv"), [row])
    print({k: v for k, v in row.items() if k != "warnings"}, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sweep")
    s1.add_argument("--param", required=True, choices=sorted(SWEEP))
    s1.add_argument("--seeds", type=int, default=6)
    s1.add_argument("--out", required=True)
    s2 = sub.add_parser("v1")
    s2.add_argument("--condition", required=True, choices=sorted(CONDITIONS) + ["disk"])
    s2.add_argument("--seeds", type=int, default=20)
    s2.add_argument("--out", required=True)
    s3 = sub.add_parser("real")
    s3.add_argument("--file", required=True)
    s3.add_argument("--out", required=True)
    s5 = sub.add_parser("calib")
    s5.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    s5.add_argument("--seeds", type=int, default=6)
    s5.add_argument("--out", required=True)
    s6 = sub.add_parser("trim")
    s6.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    s6.add_argument("--seeds", type=int, default=6)
    s6.add_argument("--out", required=True)
    s6.add_argument("--any-polar", action="store_true", help="trim against any polar curve, valid or not")
    s4 = sub.add_parser("real-list")
    s4.add_argument("--root", default=REAL_ROOT)
    a = ap.parse_args(argv)
    if a.cmd == "sweep":
        os.makedirs(a.out, exist_ok=True)
        run_sweep(a.param, a.seeds, a.out)
    elif a.cmd == "v1":
        os.makedirs(a.out, exist_ok=True)
        run_v1(a.condition, a.seeds, a.out)
    elif a.cmd == "calib":
        os.makedirs(a.out, exist_ok=True)
        run_calibration(a.condition, a.seeds, a.out)
    elif a.cmd == "trim":
        os.makedirs(a.out, exist_ok=True)
        run_trim(a.condition, a.seeds, a.out, require_valid=not a.any_polar)
    elif a.cmd == "real":
        os.makedirs(a.out, exist_ok=True)
        run_real(a.file, a.out)
    else:
        for f in real_files(a.root):
            print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
