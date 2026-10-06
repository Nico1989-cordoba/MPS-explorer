# -*- coding: utf-8 -*-
"""
The simulated-axon harness of the column-analysis tools: SYNTHETIC inputs only.

What the program's demonstration modes (``python -m tools.mps_zquality_window --demo``,
``python -m tools.mps_viability_explorer --demo``), ``batch_columns.write_simulated_input`` and the tests simulate
from. Every value here is a round, first-principles choice (no measured contour, precision sample or count of any
real axon):

- ``base_config()``: three spectrin rings 170 nm apart (the literature's axial period, sigma 15 nm), ~3.5 clusters
  per um of membrane, 20 fluorophores per cluster, lateral precisions drawn from a gamma law with a median of ~9 nm,
  axial precisions ~50 nm (better near the focal plane), an axial "leak" scale of 1.3 per stratum plus 50 nm of
  structural axial spread (``tools.mps_simulate_axon.scale_axial_leak`` scales both), background 400 per um^3;
- ``CASES``: "circle" (radius 2200 nm), "ellipse" (2400 x 1600 nm) and "slot" (a concave, slot-shaped Fourier
  contour); each pick is the contour pushed out by ``PICK_MARGIN_NM`` (its convex hull, as ``pick_keep`` reads it);
- synthetic lumen objects (``objects="syn"``): Gaussian clusters inside the contour, drawn from ``templates()``.

API (the names the tools and the tests use): ``SEED_BASE``, ``LUMEN_OFFSET``, ``base_config``, ``case_spec``,
``cell_spec``, ``sim_config``, ``pick_keep``, ``simulate_and_build``, ``truth_of``, ``ring_signature``,
``signature_differences``, ``synthetic_objects``, ``templates``.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SEED_BASE = 2026102800000
LUMEN_OFFSET = 100000          # source label of a lumen object's localizations (LUMEN_OFFSET + object index)
CENTRE_NM = (3500.0, 3500.0)    # where the synthetic contours sit (positive coordinates, like a camera field)
PICK_MARGIN_NM = 370.0         # how far the pick polygon reaches outside the membrane
N_TEMPLATES = 12

# Round, first-principles nuisance values of a case (lambda per ring, radial scatter, background density).
CASE_OVERRIDES: Dict[str, Any] = dict(ring_rate_factors=(1.0, 1.25, 1.25), radial_offset_sd_nm=100.0,
                                      background_per_um3=400.0)

_CACHE: Dict[str, Any] = {}


# ============================================================ the configuration
def _gamma_quantiles(median: float, shape: float, n: int) -> np.ndarray:
    """``n`` evenly spaced quantiles (0.5/n ... 1 - 0.5/n) of a gamma law with this shape and median."""
    from scipy.stats import gamma
    q = (np.arange(n) + 0.5) / n
    scale = median / float(gamma.ppf(0.5, shape))
    return np.asarray(gamma.ppf(q, shape, scale=scale), dtype=np.float64)


def base_config() -> Any:
    """The harness's synthetic SimConfig (see the module docstring); the contour comes from the case."""
    if "base" not in _CACHE:
        from tools.mps_simulate_axon import SimConfig
        edges = np.array([-450.0, -255.0, -85.0, 85.0, 255.0, 450.0])
        medians = (60.0, 55.0, 45.0, 50.0, 60.0)
        _CACHE["base"] = SimConfig(
            contour_nm=None, ellipse_semi_axes_nm=(1500.0, 1000.0), fourier_perturbation=(), n_rings=3,
            period_nm=170.0, sigma_period_nm=15.0, z_middle_lab_nm=0.0, clusters_per_um=3.5,
            n_clusters_per_ring=None, d_min_nm=50.0, radial_offset_sd_nm=100.0, model="M1", f=0.0,
            sigma_col_nm=10.0, q=1.0, alpha_deg_per_ring=0.0, jitter_nm=5.0, m2_lattice_nm=200.0,
            m2_p_occupied=0.8, m2_h_inherit=0.0, m7_z_half_width_nm=None, n_fluor_mean=20.0, p_lab=1.0,
            n_locs_per_cluster_quantiles=None, epitope_radius_nm=10.0, sigma_link_nm=12.0,
            locs_per_fluor_mean=2.5, max_dark_frames=0, n_frames=60000,
            lp_lateral_samples_nm=_gamma_quantiles(9.0, 6.0, 200), lpz_bin_edges_lab_nm=edges,
            lpz_samples_by_bin_nm=[_gamma_quantiles(m, 8.0, 200) for m in medians],
            axial_scale_by_bin=np.full(len(medians), 1.3), sigma_struct_nm=50.0, background_per_um3=400.0,
            background_margin_nm=150.0, tilt_deg=0.0, azimuth_deg=0.0, pixel_size_nm=130.0,
            provenance={"dataset_role": "check", "date": "2026-10-06", "code_commit": "harness",
                        "note": "tools/mps_sim_harness.py synthetic configuration (no real axon)"})
    return _CACHE["base"]


# ============================================================ the contours
def _polar_contour(radius: Any, n: int = 200) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    r = np.asarray(radius(t), float)
    return np.column_stack([CENTRE_NM[0] + r * np.cos(t), CENTRE_NM[1] + r * np.sin(t)])


def _contours() -> Dict[str, np.ndarray]:
    def circle(t: np.ndarray) -> np.ndarray:
        return np.full(t.shape, 2200.0)

    def ellipse(t: np.ndarray) -> np.ndarray:
        a, b = 2400.0, 1600.0
        return np.asarray(a * b / np.sqrt((b * np.cos(t)) ** 2 + (a * np.sin(t)) ** 2), float)

    def slot(t: np.ndarray) -> np.ndarray:
        # an elongated outline with a shallow indentation on each long side (concave, like a flattened axon)
        return np.asarray(1750.0 * (1.0 + 0.28 * np.cos(2 * t) - 0.10 * np.cos(4 * t) + 0.04 * np.cos(3 * t + 0.7)),
                          float)

    return {"circle": _polar_contour(circle), "ellipse": _polar_contour(ellipse), "slot": _polar_contour(slot)}


def _outward_offset(contour: np.ndarray, margin: float) -> np.ndarray:
    """The convex hull of the contour pushed ``margin`` nm along its outward normals (what ``pick_keep`` uses: the
    Delaunay triangulation of the pick's vertices covers their convex hull)."""
    from scipy.spatial import ConvexHull
    c = np.asarray(contour, float)
    tan = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    nrm = np.column_stack([tan[:, 1], -tan[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    area2 = float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))
    if area2 < 0:                 # clockwise: the normal above points inward
        nrm = -nrm
    pts = c + margin * nrm
    hull = ConvexHull(pts)
    return np.asarray(pts[hull.vertices], float)


def _max_depth(contour: np.ndarray) -> float:
    """The largest distance from an interior point to the contour (a grid search; enough for the object placement)."""
    from scipy.spatial import cKDTree
    c = np.asarray(contour, float)
    dense = _densify(c, 10.0)
    tree = cKDTree(dense)
    x0, y0 = c.min(axis=0)
    x1, y1 = c.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y0, y1, 120))
    g = np.column_stack([gx.ravel(), gy.ravel()])
    inside = _point_in_polygon(g, c)
    if not inside.any():
        return 0.0
    d, _ = tree.query(g[inside])
    return float(d.max())


def _densify(c: np.ndarray, step: float) -> np.ndarray:
    out = []
    for a, b in zip(c, np.roll(c, -1, axis=0)):
        n = max(1, int(math.ceil(float(np.hypot(*(b - a))) / step)))
        out.append(a + (b - a) * (np.arange(n)[:, None] / n))
    return np.vstack(out)


CASE_NAMES: Tuple[str, ...] = ("circle", "ellipse", "slot")


def case_spec(name: str) -> Dict[str, Any]:
    """A case of the harness: its contour, pick polygon, nuisance overrides, ring count, background box margin and
    maximal inscribed depth (``dmax``, where synthetic lumen objects may sit)."""
    if name not in CASE_NAMES:
        raise KeyError(f"unknown harness case {name!r}; known: {list(CASE_NAMES)}")
    key = "case_" + name
    if key not in _CACHE:
        contour = _contours()[name]
        pick = _outward_offset(contour, PICK_MARGIN_NM)
        margin = max(float(contour[:, 0].min() - pick[:, 0].min()), float(pick[:, 0].max() - contour[:, 0].max()),
                     float(contour[:, 1].min() - pick[:, 1].min()), float(pick[:, 1].max() - contour[:, 1].max()),
                     0.0) + 60.0
        _CACHE[key] = dict(name=name, index=CASE_NAMES.index(name), contour=contour, pick=pick,
                           overrides=dict(CASE_OVERRIDES), n_rings=3, box=float(margin),
                           dmax=_max_depth(contour))
    spec: Dict[str, Any] = dict(_CACHE[key])
    spec["overrides"] = dict(spec["overrides"])
    return spec


def cell_spec(name: str) -> Dict[str, Any]:
    """The same cases under the name the lumen-review tests use (a cell = a case with its object budget)."""
    spec = case_spec(name)
    spec["box"] = float(max(500.0, spec["box"] + 90.0))
    return spec


def sim_config(contour: np.ndarray, overrides: Dict[str, Any], n_rings: int, leak: bool, box_margin_nm: float) -> Any:
    """The harness configuration with this contour (as is), model M1, the leak off (axial_scale_by_bin 0.05 in every
    stratum, sigma_struct 0) unless ``leak``, the background box widened to cover the pick, then ``overrides``."""
    base = base_config()
    n_bins = int(np.asarray(base.axial_scale_by_bin).size)
    kw: Dict[str, Any] = dict(contour_nm=np.asarray(contour, float), contour_name=None, contour_library_npz=None,
                              contour_smoothing_knot_nm=None, model="M1", f=0.0, q=1.0, alpha_deg_per_ring=0.0,
                              n_rings=int(n_rings), background_margin_nm=float(box_margin_nm))
    if not leak:
        kw["axial_scale_by_bin"] = [0.05] * n_bins
        kw["sigma_struct_nm"] = 0.0
    for k, v in overrides.items():
        if k == "n_rings":
            kw[k] = int(v)
        else:
            kw[k] = tuple(v) if k == "ring_rate_factors" and v is not None else v
    return dataclasses.replace(base, **kw)


def pick_keep(cfg: Any, axon: Any, pick: np.ndarray) -> np.ndarray:
    """Every localization (background or cluster) outside the pick polygon (simulator axon frame) dropped; the
    background box must cover the pick."""
    from scipy.spatial import Delaunay
    c = np.asarray(axon.contour_used_nm, float)
    m = float(cfg.background_margin_nm)
    if (pick[:, 0].min() < c[:, 0].min() - m or pick[:, 0].max() > c[:, 0].max() + m
            or pick[:, 1].min() < c[:, 1].min() - m or pick[:, 1].max() > c[:, 1].max() + m):
        raise RuntimeError("the simulator's background box does not cover the pick polygon")
    lab = np.vstack([np.asarray(axon.x_nm), np.asarray(axon.y_nm), np.asarray(axon.z_nm)])
    ax = np.asarray(axon.frame_rotation, float).T @ lab
    return np.asarray(Delaunay(pick).find_simplex(ax[:2].T) >= 0, dtype=bool)


# ============================================================ synthetic lumen objects
def templates() -> List[Dict[str, np.ndarray]]:
    """``N_TEMPLATES`` synthetic clusters (offsets about their centre, frames, precisions): 30-60 localizations,
    lateral sd 15 nm, axial sd 35 nm, bursts of consecutive frames, precisions from ``base_config``'s samples."""
    if "tpl" not in _CACHE:
        base = base_config()
        rng = np.random.default_rng(20261006)
        lp_s = np.asarray(base.lp_lateral_samples_nm, float)
        lpz_s = np.asarray(base.lpz_samples_by_bin_nm[2], float)
        out = []
        for _ in range(N_TEMPLATES):
            n = int(rng.integers(30, 61))
            starts = np.sort(rng.integers(0, int(base.n_frames) - 10, size=max(1, n // 3)))
            frames = np.sort(np.concatenate([s + np.arange(3) for s in starts])[:n]).astype(np.int64)
            n = frames.size
            out.append(dict(dx=rng.normal(0.0, 15.0, n), dy=rng.normal(0.0, 15.0, n), dz=rng.normal(0.0, 35.0, n),
                            frame=frames, lp=rng.choice(lp_s, n), lpz=rng.choice(lpz_s, n)))
        _CACHE["tpl"] = out
    got: List[Dict[str, np.ndarray]] = _CACHE["tpl"]
    return got


def _point_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    xj, yj = poly[-1]
    for xi, yi in poly:
        den = np.where(yj - yi == 0, 1e-12, yj - yi)
        inside ^= ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / den + xi)
        xj, yj = xi, yi
    return inside


def _depth_in(tree: Any, pts: np.ndarray, pip_pts: np.ndarray) -> np.ndarray:
    d, _ = tree.query(pts)
    return np.asarray(np.where(_point_in_polygon(pts, pip_pts), d, -d), dtype=float)


def synthetic_objects(axon: Any, rng: np.random.Generator, dmax_contour: float) -> List[Dict[str, Any]]:
    """4 isolated single clusters (depth U(250, min(1200, max depth - 100)) inside the TRUE contour, >= 800 nm
    apart) + 1 group of 3 (the first U(400, ...), the next two 150-300 nm away, each >= 100 nm deep); a synthetic
    template each, the ring drawn uniformly."""
    from scipy.spatial import cKDTree
    from tools.mps_matching import smooth_closed_path
    path = smooth_closed_path(np.asarray(axon.contour_used_nm, float))
    P = np.asarray(path.points_nm, float)
    tree = cKDTree(P)
    P_pip = P[::4]
    x0, y0 = P.min(axis=0)
    x1, y1 = P.max(axis=0)
    cand = np.column_stack([rng.uniform(x0, x1, 20000), rng.uniform(y0, y1, 20000)])
    dep = _depth_in(tree, cand, P_pip)
    hi = min(1200.0, dmax_contour - 100.0)
    tpl = templates()
    n_rings = int(np.asarray(axon.ring_z_nm).size)
    objs: List[Dict[str, Any]] = []

    def far_enough(p: np.ndarray) -> bool:
        return all(np.hypot(*(p - o["centre"])) >= 800.0 for o in objs)

    def pick_at(lo_: float, hi_: float) -> Tuple[Optional[np.ndarray], float]:
        for _ in range(200):
            d_target = rng.uniform(lo_, hi_)
            ok = np.flatnonzero(np.abs(dep - d_target) < 15.0)
            if ok.size == 0:
                continue
            p = cand[rng.choice(ok)]
            if far_enough(p):
                return p, float(d_target)
        return None, float("nan")

    for _i in range(4):
        p, dt = pick_at(250.0, max(hi, 260.0))
        if p is not None:
            objs.append(dict(centre=p, ring=int(rng.integers(n_rings)), tpl=int(rng.integers(len(tpl))), design_depth=dt,
                             kind="single"))
    p, dt = pick_at(400.0, max(hi, 410.0))
    if p is not None:
        grp = [p]
        for _ in range(2):
            for _try in range(100):
                a = rng.uniform(0, 2 * math.pi)
                q = grp[-1] + rng.uniform(150.0, 300.0) * np.array([math.cos(a), math.sin(a)])
                dq = float(_depth_in(tree, q[None], P_pip)[0])
                if dq >= 100.0 and all(np.hypot(*(q - o["centre"])) >= 800.0 for o in objs):
                    grp.append(q)
                    break
        for q in grp:
            objs.append(dict(centre=q, ring=int(rng.integers(n_rings)), tpl=int(rng.integers(len(tpl))),
                             design_depth=float(_depth_in(tree, q[None], P_pip)[0]), kind="group"))
    return objs


def object_arrays(axon: Any, objs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    rot = np.asarray(axon.frame_rotation, float)
    ring_z = np.asarray(axon.ring_z_nm, float)
    tpl = templates()
    parts: Dict[str, List[Any]] = {k: [] for k in ("x", "y", "z", "frame", "lp", "lpz", "src")}
    for j, o in enumerate(objs):
        t = tpl[o["tpl"]]
        rz = ring_z[min(int(o["ring"]), ring_z.size - 1)]
        xa, ya, za = o["centre"][0] + t["dx"], o["centre"][1] + t["dy"], rz + t["dz"]
        labp = rot @ np.vstack([xa, ya, za])
        parts["x"].append(labp[0])
        parts["y"].append(labp[1])
        parts["z"].append(labp[2])
        parts["frame"].append(t["frame"])
        parts["lp"].append(t["lp"])
        parts["lpz"].append(t["lpz"])
        parts["src"].append(np.full(t["dx"].size, LUMEN_OFFSET + j, np.int64))
    return {k: np.concatenate(v) for k, v in parts.items()}


# ============================================================ simulate, build, truth
def simulate_and_build(spec: Dict[str, Any], seed: int, *, leak: bool, objects: Optional[str] = None) -> Dict[str, Any]:
    """simulate_axon (the pick background) + synthetic lumen objects (``objects="syn"``) + build_rings as
    power_columns builds a replicate (_rings_params(0, tilt)). Returns cfg, axon, res, src (per input localization)
    and the arrays given."""
    import power_columns as pc
    from tools.mps_columns import build_rings
    from tools.mps_simulate_axon import simulate_axon
    if objects not in (None, "syn"):
        raise ValueError(f"objects must be None or 'syn', got {objects!r}")
    cfg = sim_config(spec["contour"], spec["overrides"], spec["n_rings"], leak, spec["box"])
    axon = simulate_axon(cfg, int(seed))
    keep = pick_keep(cfg, axon, spec["pick"])
    x = np.array(axon.x_nm, float)[keep]
    y = np.array(axon.y_nm, float)[keep]
    z = np.array(axon.z_nm, float)[keep]
    fr = np.array(axon.frame, np.int64)[keep]
    lp = np.array(axon.lp_lateral_nm, float)[keep]
    lpz = np.array(axon.lpz_nm, float)[keep]
    src = np.array(axon.cluster_true, np.int64)[keep]
    objs: List[Dict[str, Any]] = []
    if objects == "syn":
        objs = synthetic_objects(axon, np.random.default_rng(np.random.SeedSequence([int(seed), 7])), spec["dmax"])
    if objs:
        t = object_arrays(axon, objs)
        x = np.concatenate([x, t["x"]])
        y = np.concatenate([y, t["y"]])
        z = np.concatenate([z, t["z"]])
        fr = np.concatenate([fr, t["frame"]])
        lp = np.concatenate([lp, t["lp"]])
        lpz = np.concatenate([lpz, t["lpz"]])
        src = np.concatenate([src, t["src"]])
        order = np.argsort(fr, kind="stable")
        x, y, z, fr, lp, lpz, src = x[order], y[order], z[order], fr[order], lp[order], lpz[order], src[order]
    res = build_rings(x.copy(), y.copy(), z.copy(), frame=fr.copy(), lp_lateral_nm=lp.copy(), lpz_nm=lpz.copy(),
                      params=pc._rings_params(0.0, float(cfg.tilt_deg)), source_name=f"{spec['name']}/{seed}",
                      pixel_size_nm=float(cfg.pixel_size_nm), pixel_size_source="override", n_frames=int(cfg.n_frames))
    return dict(cfg=cfg, axon=axon, res=res, src=src, objs=objs, x=x, y=y, z=z, frame=fr, lp=lp, lpz=lpz, seed=int(seed))


def keys_and_centroids(res: Any) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    keys, P = [], []
    for r in sorted(res.rings, key=lambda q: int(q.index)):
        for j, c in enumerate(r.clusters):
            keys.append((int(r.index), j))
            P.append(np.asarray(c.centroid_nm, float).reshape(2))
    return keys, np.asarray(P, float).reshape(-1, 2)


def true_path_in_rings_frame(axon: Any, res: Any) -> Any:
    """The simulated contour carried into the rings' axon frame."""
    from tools.mps_axis import to_axon_frame
    from tools.mps_matching import smooth_closed_path
    c = np.asarray(axon.contour_used_nm, float)
    zc = float(np.mean(np.asarray(axon.ring_z_nm, float)))
    c_axon = np.column_stack([c, np.full(c.shape[0], zc)])
    c_lab = c_axon @ np.asarray(axon.frame_rotation, float).T
    xc, yc, _ = to_axon_frame(res.frame, c_lab[:, 0].copy(), c_lab[:, 1].copy(), c_lab[:, 2].copy())
    return smooth_closed_path(np.column_stack([xc, yc]))


def truth_of(sim: Dict[str, Any]) -> Dict[str, Any]:
    """Per detected cluster (ring index, then cluster order): the majority source of its localizations (lumen object /
    simulated membrane cluster / background) and its depth inside the TRUE contour (+ inside)."""
    from tools.mps_simulate_axon import _outward_sign
    from tools.mps_unroll import project_on_path
    res = sim["res"]
    keys, P = keys_and_centroids(res)
    tp = true_path_in_rings_frame(sim["axon"], res)
    if P.shape[0]:
        _s, off = project_on_path(tp, P, outward_sign=_outward_sign(tp))
        depth = -np.asarray(off, float)
    else:
        depth = np.zeros(0)
    by = {int(r.index): r for r in res.rings}
    truth = []
    for k, j in keys:
        s = sim["src"][np.asarray(by[k].clusters[j].loc_index, np.intp)]
        vals, cnt = np.unique(s, return_counts=True)
        top = int(vals[np.argmax(cnt)])
        truth.append("lumen" if top >= LUMEN_OFFSET else ("membrane" if top >= 0 else "background"))
    return dict(keys=keys, P=P, depth=depth, truth=truth, tp=tp)


# ============================================================ ring signatures (what two cleanings must agree on)
def ring_signature(res: Any) -> List[Dict[str, Any]]:
    sig = []
    for r in res.rings:
        per = r.analysis.perimeter
        sig.append(dict(
            index=int(r.index), n_clusters=len(r.clusters), labels=[int(c.label) for c in r.clusters],
            centroids=np.array([np.asarray(c.centroid_nm, float) for c in r.clusters]).reshape(-1, 2),
            contour=None if r.contour_nm is None else np.asarray(r.contour_nm, float).copy(),
            order=None if per is None else np.asarray(per.order).copy(),
            length=float(r.length_nm), centroid=None if r.centroid_nm is None else np.asarray(r.centroid_nm, float).copy(),
            loc_index=np.asarray(r.loc_index).copy(), n_locs=int(r.n_locs), n_events=r.n_events,
            event_id=None if r.event_id is None else np.asarray(r.event_id).copy(),
            an_centroids=np.asarray(r.analysis.centroids, float).copy()))
    return sig


def signature_differences(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> List[str]:
    if len(a) != len(b):
        return [f"{len(a)} vs {len(b)} rings"]
    bad = []
    for s, t in zip(a, b):
        for key in s:
            x, y = s[key], t[key]
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                if x is None or y is None or np.asarray(x).shape != np.asarray(y).shape or not np.array_equal(np.asarray(x), np.asarray(y)):
                    bad.append(f"ring {s['index']}: {key}")
            elif isinstance(x, float) and isinstance(y, float):
                if not (x == y or (math.isnan(x) and math.isnan(y))):
                    bad.append(f"ring {s['index']}: {key}")
            elif x != y:
                bad.append(f"ring {s['index']}: {key}")
    return bad
