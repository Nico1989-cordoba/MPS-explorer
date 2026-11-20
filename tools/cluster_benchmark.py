# -*- coding: utf-8 -*-
"""
Multi-algorithm clustering benchmark on synthetic SMLM data (synthetic only, rule R8).

The harness has four parts:

1. A **scenario generator** (:func:`generate`): molecules along a closed contour (the
   transverse section of an axon) or in a flat window, two epitopes per molecule, a
   binomial number of labels per molecule, geometric blinking cycles per label with
   1-3 localizations per cycle, per-localization precision, linkage error and two
   kinds of background. Every localization carries its true molecule id. The
   model is the one of the clustering study (SIM_CLUSTERING_REPORT.md section 2.1,
   own simulation, 2026-10-15), ported without anything that depends on private
   files.
2. **Metrics** (:func:`evaluate`): molecule-level outcomes by majority matching
   (one cluster = one molecule, merges, splits, missed molecules, false clusters),
   the adjusted Rand index between molecule ids and labels on non-noise
   localizations, the mean per-cluster IoU against the majority molecule, and the
   bias of three pipeline readouts (clusters per um of arc, median 1NN between
   centres, median r_eff from the convex hull) against the oracle labelling.
3. A **runner** (:func:`run_benchmark`) over any object with ``fit(xy) -> labels``
   (the ``ClusteringStrategy`` interface of ``tools.clustering_strategies``), with
   a :class:`CallableStrategy` adapter for plain functions and an
   :class:`ExternalLabelsStrategy` that reads labels produced by another program.
4. Writers for the long table (CSV), the summary (CSV, Markdown) and a JSON config,
   plus a command line (``python -m tools.cluster_benchmark --help``).

Every default states its origin in brackets: [literature], [derivation],
[simulation] (this project's own simulations) or [ad hoc].
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
import warnings
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError, cKDTree

from tools.clustering_strategies import ClusteringStrategy, DBSCANStrategy

# --------------------------------------------------------------------------- defaults (origin in brackets)
SIGMA_LINK_NM = 10.0          # [literature + derivation] Fruh et al. 2021: linkage error up to ~19 nm (3D, antibody
                              # pair); 19 / sqrt(3) ~ 11 -> ~10 nm per axis
EPITOPE_SEP_NM = (5.0, 10.0)  # [inference, no published value] the two beta C-termini of a spectrin tetramer
MEMBRANE_ROUGH_NM = 5.0       # [ad hoc] normal offset of a molecule centre from the ideal contour
N_FRAMES = 20000              # [ad hoc] typical dSTORM movie length
ON_FRAMES = (1, 2, 3)         # [ad hoc] on-time of one switching cycle, frames (mean 2 = project locs_per_fluor_mean)
BG_BAND_NM = 150.0            # [ad hoc] band around the contour that receives background
HARD_CORE_FRACTION = 0.4      # [ad hoc] hard core = 0.4 x mean spacing (project simulator: 80 nm at 200 nm)
PAIR_SEP_NM = (30.0, 60.0)    # [ad hoc] arc separation of a molecule and its partner (oligomer / neighbour)
SIGMA_LOC_CV = 0.25           # [simulation] log-normal spread of the per-localization precision (P-R61 generator)
SPACING_NM = {"sparse": 200.0, "medium": 100.0, "dense": 40.0}  # sparse [literature, preprint: ~5 clusters/um,
                              # Gazal et al. 2026]; medium and dense [ad hoc: stress tests]
TAU_EXACT = 0.8               # [ad hoc] share threshold for "one cluster = one molecule" (used in the clustering
                              # study, SIM_CLUSTERING_REPORT section 8.1; not derived)
PRESENT_SHARE = 0.2           # [ad hoc] a molecule is "present" in a cluster above this share of its locs
FALSE_BG_SHARE = 0.5          # [ad hoc] a cluster is "false" when more than half of its locs are background
MISSED_SHARE = 0.5            # [ad hoc] a molecule is "missed" when less than half of its locs are in any cluster
EPS_NM_DEFAULT, MIN_SAMPLES_DEFAULT = 25.0, 10  # [literature] Gazal et al. 2026 / project DBSCAN defaults


@dataclass
class Scenario:
    """One synthetic condition. Values in nm, um and frames; see module docstring for origins."""

    name: str
    geometry: str = "ring"            # "ring": contour of perimeter L; "window": flat square of side window_um
    spacing_nm: float = 200.0         # mean arc spacing between molecules (ring) or sqrt(area per molecule) (window)
    p_label: float = 0.6              # probability that one epitope carries a label (two epitopes per molecule)
                                      # [ad hoc; the presets use 0.5 and 1.0]
    c_label: float = 14.0             # mean switching cycles per label [literature] Dempsey 2011: AF647 MEA 14, bME 26
    sigma_loc: float = 12.0           # nominal localization precision per axis [literature] ~NeNA of Gazal 2026
    bg_labels_per_um: float = 0.5     # non-specific labels per um of contour (window: per um^2 x 0.3) [ad hoc]
    bg_locs_per_um2: float = 30.0     # isolated single localizations per um^2 of band [ad hoc]
    f_pair: float = 0.15              # fraction of molecules with a partner at PAIR_SEP_NM [ad hoc]
    L_um_range: Tuple[float, float] = (10.0, 30.0)  # perimeter range of the ring [ad hoc, range of the 18-axon set]
    window_um: float = 2.0            # side of the flat window [ad hoc]
    sigma_link: float = SIGMA_LINK_NM
    pair_sep: Tuple[float, float] = PAIR_SEP_NM
    origin: str = "simulation: SIM_CLUSTERING_REPORT.md section 2.1"

    def as_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def _preset(name: str, **kw: Any) -> Scenario:
    return Scenario(name=name, **kw)


# Presets of the clustering study (own simulation; report sections 2.1 and 8.1). "lab1" is p_label 0.5
# (about 1 label per labelled molecule), "lab2" is p_label 1.0 (2 labels per molecule).
PRESETS: Dict[str, Scenario] = {
    "sparse_lab1": _preset("sparse_lab1", spacing_nm=200.0, p_label=0.5),
    "sparse_lab2": _preset("sparse_lab2", spacing_nm=200.0, p_label=1.0),
    "medium_lab1": _preset("medium_lab1", spacing_nm=100.0, p_label=0.5),
    "medium_lab2": _preset("medium_lab2", spacing_nm=100.0, p_label=1.0),
    "sparse_lab2_sig20": _preset("sparse_lab2_sig20", spacing_nm=200.0, p_label=1.0, sigma_loc=20.0),
    "medium_lab2_sig20": _preset("medium_lab2_sig20", spacing_nm=100.0, p_label=1.0, sigma_loc=20.0),
    "sparse_lab2_c26": _preset("sparse_lab2_c26", spacing_nm=200.0, p_label=1.0, c_label=26.0),
    "medium_lab2_c26": _preset("medium_lab2_c26", spacing_nm=100.0, p_label=1.0, c_label=26.0),
    "sparse_lab2_bghigh": _preset("sparse_lab2_bghigh", spacing_nm=200.0, p_label=1.0,
                                  bg_labels_per_um=2.0, bg_locs_per_um2=120.0),
    "sparse_lab2_bgnone": _preset("sparse_lab2_bgnone", spacing_nm=200.0, p_label=1.0,
                                  bg_labels_per_um=0.0, bg_locs_per_um2=0.0),
    "window_sparse_lab2": _preset("window_sparse_lab2", geometry="window", spacing_nm=200.0, p_label=1.0,
                                  origin="ad hoc: flat window variant of sparse_lab2 (no contour)"),
}


@dataclass
class SyntheticSample:
    """Localizations with ground truth. ``molecule_id`` >= 0 real molecule, -1 isolated background
    localization, <= -2 non-specific (background) label."""

    xy: NDArray[np.float64]
    frame: NDArray[np.int64]
    molecule_id: NDArray[np.int64]
    label_id: NDArray[np.int64]
    event_id: NDArray[np.int64]
    sigma: NDArray[np.float64]
    mol_xy: NDArray[np.float64]        # (K, 2) true centres of all molecules (labelled or not)
    mol_n_labels: NDArray[np.int64]    # labels per molecule
    arc_length_nm: float               # perimeter (ring) or 4 x side (window), for "per um" readouts
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_background(self) -> NDArray[np.bool_]:
        return self.molecule_id < 0

    @property
    def oracle_labels(self) -> NDArray[np.int64]:
        """Labels that reproduce the truth: molecule id for real localizations, -1 for background."""
        return np.where(self.molecule_id >= 0, self.molecule_id, -1).astype(np.int64)


# --------------------------------------------------------------------------- geometry helpers
@dataclass
class _Contour:
    xy: NDArray[np.float64]
    s: NDArray[np.float64]
    norm: NDArray[np.float64]
    L: float

    def at(self, s_query: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        sq = np.mod(np.asarray(s_query, float), self.L)
        idx = np.clip(np.searchsorted(self.s, sq, side="right") - 1, 0, self.s.size - 1)
        nxt = (idx + 1) % self.s.size
        seg = (self.s[nxt] - self.s[idx]) % self.L
        seg = np.where(seg <= 0, self.L - self.s[idx], seg)
        f = np.clip((sq - self.s[idx]) / np.maximum(seg, 1e-9), 0, 1)
        p = self.xy[idx] + f[:, None] * (self.xy[nxt] - self.xy[idx])
        return p, self.norm[idx]


def _make_ellipse(L_nm: float, aspect: float, M: int = 4000) -> _Contour:
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    xy = np.column_stack([aspect * np.cos(th), np.sin(th)])
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    xy *= L_nm / seg.sum()
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    d = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    tang = d / np.linalg.norm(d, axis=1)[:, None]
    norm = np.column_stack([tang[:, 1], -tang[:, 0]])
    return _Contour(xy=xy, s=s, norm=norm, L=float(seg.sum()))


def _place_hard_core_1d(rng: np.random.Generator, L: float, K: int, d_min: float) -> NDArray[np.float64]:
    """Sequential rejection on a circle of length L (same rule as the project simulator)."""
    K = min(int(K), int(0.6 * L / d_min))
    placed: List[float] = []
    for _ in range(K):
        ok = False
        for _ in range(5000):
            s = float(rng.uniform(0.0, L))
            if not placed:
                ok = True
                break
            d = np.abs(np.asarray(placed) - s)
            d = np.minimum(d, L - d)
            if bool(np.all(d >= d_min)):
                ok = True
                break
        if not ok:
            break
        placed.append(s)
    return np.sort(np.asarray(placed, dtype=float))


def _place_hard_core_2d(rng: np.random.Generator, side: float, K: int, d_min: float) -> NDArray[np.float64]:
    pts: List[NDArray[np.float64]] = []
    for _ in range(K):
        ok = False
        for _ in range(5000):
            p = rng.uniform(0.0, side, 2)
            if not pts or bool(np.all(np.linalg.norm(np.asarray(pts) - p, axis=1) >= d_min)):
                ok = True
                break
        if not ok:
            break
        pts.append(p)
    return np.asarray(pts, dtype=float).reshape(-1, 2)


# --------------------------------------------------------------------------- generator
def generate(sc: Scenario, rng: np.random.Generator, L_nm: Optional[float] = None) -> SyntheticSample:
    """Draw one synthetic sample of scenario ``sc``. ``L_nm`` fixes the ring perimeter (default: uniform in
    ``sc.L_um_range``). The returned ``params`` record the realised geometry."""
    d_min = HARD_CORE_FRACTION * sc.spacing_nm
    if sc.geometry == "ring":
        if L_nm is None:
            L_nm = 1000.0 * float(rng.uniform(*sc.L_um_range))
        con: Optional[_Contour] = _make_ellipse(L_nm, float(rng.uniform(1.0, 1.6)))
        s_main = _place_hard_core_1d(rng, L_nm, int(round(L_nm / sc.spacing_nm)), d_min)
        K_main = s_main.size
        n_pair = int(rng.binomial(K_main, sc.f_pair)) if K_main else 0
        idx_pair = rng.choice(K_main, n_pair, replace=False) if n_pair else np.empty(0, int)
        sep = rng.uniform(*sc.pair_sep, n_pair)
        s_part = np.mod(s_main[idx_pair] + sep * rng.choice([-1.0, 1.0], n_pair), L_nm)
        s_all = np.concatenate([s_main, s_part])
        assert con is not None
        p, nm = con.at(s_all)
        mol_xy = p + rng.normal(0.0, MEMBRANE_ROUGH_NM, s_all.size)[:, None] * nm
        arc_nm = float(L_nm)
        bg_region_um2 = (L_nm / 1000.0) * (2 * BG_BAND_NM / 1000.0)
        n_bg_lab_mean = sc.bg_labels_per_um * L_nm / 1000.0
    elif sc.geometry == "window":
        side = 1000.0 * sc.window_um
        con = None
        K0 = int(round(side * side / sc.spacing_nm ** 2))
        main_xy = _place_hard_core_2d(rng, side, K0, d_min)
        K_main = main_xy.shape[0]
        n_pair = int(rng.binomial(K_main, sc.f_pair)) if K_main else 0
        idx_pair = rng.choice(K_main, n_pair, replace=False) if n_pair else np.empty(0, int)
        ang = rng.uniform(0, 2 * np.pi, n_pair)
        sep = rng.uniform(*sc.pair_sep, n_pair)
        part_xy = main_xy[idx_pair] + sep[:, None] * np.column_stack([np.cos(ang), np.sin(ang)])
        mol_xy = np.concatenate([main_xy, part_xy])
        arc_nm = 4.0 * side
        bg_region_um2 = sc.window_um ** 2
        n_bg_lab_mean = sc.bg_labels_per_um * 0.3 * bg_region_um2
    else:
        raise ValueError(f"unknown geometry {sc.geometry!r}")
    K = mol_xy.shape[0]
    # epitopes: 2 per molecule, 5-10 nm apart, random orientation; each labelled with p_label
    ang_e = rng.uniform(0, 2 * np.pi, K)
    half = 0.5 * np.asarray(rng.uniform(*EPITOPE_SEP_NM, K), dtype=float)
    u = np.column_stack([np.cos(ang_e), np.sin(ang_e)])
    epi = np.concatenate([mol_xy + half[:, None] * u, mol_xy - half[:, None] * u])
    epi_mol = np.concatenate([np.arange(K), np.arange(K)])
    labelled = rng.uniform(size=2 * K) < sc.p_label
    lab_pos = epi[labelled] + rng.normal(0.0, sc.sigma_link, (int(labelled.sum()), 2))
    lab_mol = epi_mol[labelled].astype(np.int64)
    mol_n_labels = np.bincount(lab_mol, minlength=K).astype(np.int64)
    # background labels (non-specific stacks that blink like a label)
    n_bg_lab = int(rng.poisson(n_bg_lab_mean))
    if n_bg_lab:
        bg_pos = _background_positions(rng, con, arc_nm, sc.window_um, n_bg_lab)
        lab_pos = np.concatenate([lab_pos, bg_pos])
        lab_mol = np.concatenate([lab_mol, -2 - np.arange(n_bg_lab)])
    n_lab = lab_pos.shape[0]
    # blinking: geometric number of cycles per label, each 1-3 consecutive frames
    n_cyc = rng.geometric(1.0 / sc.c_label, n_lab).astype(np.int64)
    cyc_label = np.repeat(np.arange(n_lab), n_cyc)
    n_cyc_tot = int(n_cyc.sum())
    start = rng.integers(0, N_FRAMES, n_cyc_tot)
    on = rng.choice(ON_FRAMES, n_cyc_tot)
    loc_cyc = np.repeat(np.arange(n_cyc_tot), on)
    frame = start[loc_cyc] + (np.arange(loc_cyc.size) - np.repeat(np.cumsum(on) - on, on))
    keep = frame < N_FRAMES
    loc_cyc = loc_cyc[keep]
    frame = frame[keep]
    loc_label = cyc_label[loc_cyc]
    n = loc_label.size
    lv = math.log1p(SIGMA_LOC_CV ** 2)
    sig = sc.sigma_loc * np.exp(rng.normal(-0.5 * lv, math.sqrt(lv), n))
    xy = lab_pos[loc_label] + rng.normal(size=(n, 2)) * sig[:, None]
    mol = lab_mol[loc_label]
    event = loc_cyc.astype(np.int64)
    # isolated background localizations
    n_bg_loc = int(rng.poisson(sc.bg_locs_per_um2 * bg_region_um2))
    if n_bg_loc:
        xy = np.concatenate([xy, _background_positions(rng, con, arc_nm, sc.window_um, n_bg_loc)])
        frame = np.concatenate([frame, rng.integers(0, N_FRAMES, n_bg_loc)])
        mol = np.concatenate([mol, np.full(n_bg_loc, -1)])
        loc_label = np.concatenate([loc_label, np.full(n_bg_loc, -1)])
        event = np.concatenate([event, n_cyc_tot + np.arange(n_bg_loc)])
        sig = np.concatenate([sig, np.full(n_bg_loc, sc.sigma_loc)])
    order = np.argsort(frame, kind="stable")
    params = sc.as_dict()
    params.update({"realised_L_nm": arc_nm, "n_molecules": int(K), "n_molecules_labelled": int((mol_n_labels > 0).sum()),
                   "n_background_labels": n_bg_lab, "n_background_locs": n_bg_loc, "n_locs": int(xy.shape[0])})
    return SyntheticSample(xy=xy[order], frame=frame[order].astype(np.int64), molecule_id=mol[order].astype(np.int64),
                           label_id=loc_label[order].astype(np.int64), event_id=event[order].astype(np.int64),
                           sigma=sig[order], mol_xy=mol_xy, mol_n_labels=mol_n_labels, arc_length_nm=arc_nm,
                           params=params)


def _background_positions(rng: np.random.Generator, con: Optional[_Contour], arc_nm: float, window_um: float,
                          n: int) -> NDArray[np.float64]:
    if con is not None:
        pb, nb = con.at(rng.uniform(0, arc_nm, n))
        return pb + rng.uniform(-BG_BAND_NM, BG_BAND_NM, n)[:, None] * nb
    return rng.uniform(0.0, 1000.0 * window_um, (n, 2))


# --------------------------------------------------------------------------- strategies
class CallableStrategy(ClusteringStrategy):
    """Adapter: any ``fn(xy) -> labels`` (int array, -1 = noise) as a ClusteringStrategy."""

    def __init__(self, fn: Callable[[NDArray[np.float64]], NDArray[np.int64]], name: str,
                 params: Optional[Mapping[str, Any]] = None):
        self.fn = fn
        self.name = name
        self.params: Dict[str, Any] = dict(params or {})

    def fit(self, data: np.ndarray) -> np.ndarray:
        labels = np.asarray(self.fn(np.asarray(data, dtype=float)), dtype=np.int64)
        if labels.shape != (data.shape[0],):
            raise ValueError(f"{self.name}: expected {data.shape[0]} labels, got shape {labels.shape}")
        return labels

    def get_strategy_name(self) -> str:
        return self.name

    def get_params(self) -> dict:
        return dict(self.params)


EXTERNAL_MIN_MATCHED = 0.99
"""[ad hoc] Smallest fraction of the data's localizations that must find a row of the external CSV within tol_nm;
below it ``ExternalLabelsStrategy.fit`` raises instead of returning (mostly) noise without a word (REVIEW_D0 M6: a
CSV written with 2 decimals matched 6 of 200 rows at 1e-3 nm)."""


class ExternalLabelsStrategy(ClusteringStrategy):
    """Labels computed outside Python (BaGoL, a Bayesian method in R, ...) read from a CSV with columns
    ``x, y, label``. Rows are matched to the data by nearest position (tolerance ``tol_nm``, [ad hoc] 1e-3 nm
    for exact round trips; raise it to the rounding of the file, e.g. 0.01 nm for 2 decimals); unmatched points
    become noise (-1). If fewer than ``EXTERNAL_MIN_MATCHED`` of the localizations match, ``fit`` raises
    ``ValueError`` naming the fraction; if two localizations match the same CSV row it warns. The row order of the
    CSV does not matter (the match is by position). ``n_matched_`` / ``frac_matched_`` keep the last match."""

    def __init__(self, csv_path: str, name: Optional[str] = None, tol_nm: float = 1e-3,
                 scale_nm: Optional[float] = None):
        self.csv_path = csv_path
        self.name = name or f"external:{os.path.basename(csv_path)}"
        self.tol_nm = tol_nm
        self.declared_scale_nm = None if scale_nm is None else float(scale_nm)
        self.n_matched_: int = 0
        self.frac_matched_: float = float("nan")

    def scale_nm(self) -> Optional[float]:
        """The scale the user declares for the external method [user]; None when not declared."""
        return self.declared_scale_nm

    def fit(self, data: np.ndarray) -> np.ndarray:
        xs: List[float] = []
        ys: List[float] = []
        ls: List[int] = []
        with open(self.csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
                ls.append(int(float(row["label"])))
        ext = np.column_stack([xs, ys]) if xs else np.empty((0, 2))
        out = np.full(data.shape[0], -1, dtype=np.int64)
        n = int(data.shape[0])
        ok = np.zeros(n, dtype=bool)
        if ext.shape[0] and n:
            d, j = cKDTree(ext).query(np.asarray(data, dtype=float)[:, :2], k=1)
            ok = d <= self.tol_nm
            out[ok] = np.asarray(ls, dtype=np.int64)[j[ok]]
            if np.unique(j[ok]).size < int(ok.sum()):
                warnings.warn(f"{self.name}: {int(ok.sum()) - np.unique(j[ok]).size} localizations matched a CSV row "
                              f"already matched by another one (tol_nm={self.tol_nm})", RuntimeWarning, stacklevel=2)
        self.n_matched_ = int(ok.sum())
        self.frac_matched_ = float(ok.mean()) if n else float("nan")
        if n and self.frac_matched_ < EXTERNAL_MIN_MATCHED:
            raise ValueError(f"{self.name}: only {self.frac_matched_:.0%} of the {n} localizations matched a row of "
                             f"{self.csv_path} within tol_nm={self.tol_nm}; check the units and raise tol_nm to the "
                             f"rounding of the file")
        return out

    def get_strategy_name(self) -> str:
        return self.name

    def get_params(self) -> dict:
        return {"csv_path": self.csv_path, "tol_nm": self.tol_nm, "scale_nm": self.declared_scale_nm}


def _sklearn_hdbscan(min_cluster_size: int, min_samples: int) -> Callable[[NDArray[np.float64]], NDArray[np.int64]]:
    def fn(xy: NDArray[np.float64]) -> NDArray[np.int64]:
        from sklearn.cluster import HDBSCAN
        labels: NDArray[np.int64] = np.asarray(
            HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=int(min_samples)).fit(xy).labels_, dtype=np.int64)
        return labels
    return fn


def make_strategy(spec: str) -> ClusteringStrategy:
    """Build a strategy from a short spec: ``dbscan_<eps>_<min_samples>``, ``hdbscan_<min_cluster_size>_<min_samples>``
    (the repo's HDBSCANStrategy when the ``hdbscan`` package is installed, else scikit-learn's HDBSCAN),
    or ``external:<csv_path>``. The oracle (truth) is added by the runner, not built here."""
    if spec.startswith("external:"):
        return ExternalLabelsStrategy(spec[len("external:"):])
    if spec.startswith("reg:"):
        # A registered method (D0): ``reg:<name>`` or ``reg:<name>:k=v,k=v`` with numeric overrides.
        from tools.clustering_strategies import make_strategy as registry_make
        body = spec[len("reg:"):]
        name, _, kv = body.partition(":")
        params: Dict[str, Any] = {}
        for item in (kv.split(",") if kv else []):
            k, _, v = item.partition("=")
            try:
                params[k] = json.loads(v)          # numbers, null, true/false, quoted strings
            except ValueError:
                params[k] = {"None": None, "inf": float("inf")}.get(v, v)   # bare words: cluster_method=dbscan
        return registry_make(name, params)
    parts = spec.split("_")
    if parts[0] == "dbscan" and len(parts) == 3:
        return DBSCANStrategy(eps=float(parts[1]), min_samples=int(parts[2]))
    if parts[0] == "hdbscan" and len(parts) == 3:
        mcs, ms = int(parts[1]), int(parts[2])
        try:
            from tools.clustering_strategies import HDBSCANStrategy
            return HDBSCANStrategy(min_samples=ms, min_cluster_size=mcs)
        except ImportError:
            return CallableStrategy(_sklearn_hdbscan(mcs, ms), spec, {"min_cluster_size": mcs, "min_samples": ms})
    raise ValueError(f"unknown strategy spec {spec!r}")


def strategy_name(st: ClusteringStrategy, spec: Optional[str] = None) -> str:
    return spec if spec is not None else st.get_strategy_name()


def strategy_params(st: ClusteringStrategy) -> Dict[str, Any]:
    getter = getattr(st, "get_params", None)
    return dict(getter()) if callable(getter) else {}


# --------------------------------------------------------------------------- metrics
def _hull_r_eff(xy: NDArray[np.float64], labels: NDArray[np.int64], ids: NDArray[np.int64]) -> NDArray[np.float64]:
    """Effective radius sqrt(area / pi) of the convex hull of every cluster (0 for degenerate hulls)."""
    out = np.zeros(ids.size)
    for i, cid in enumerate(ids):
        pts = xy[labels == cid]
        if pts.shape[0] < 3:
            continue
        try:
            area = float(ConvexHull(pts).volume)
        except QhullError:
            area = 0.0
        out[i] = math.sqrt(area / math.pi)
    return out


def readouts(xy: NDArray[np.float64], labels: NDArray[np.int64], arc_length_nm: float) -> Dict[str, float]:
    """Pipeline readouts of one labelling: clusters per um of arc, median 1NN between centres, median r_eff."""
    clustered = labels >= 0
    ids = np.unique(labels[clustered])
    res: Dict[str, float] = {"n_clusters": float(ids.size), "clusters_per_um": float(ids.size / (arc_length_nm / 1000.0))}
    if ids.size == 0:
        res.update(median_1nn_nm=float("nan"), median_r_eff_nm=float("nan"), median_n_loc=float("nan"))
        return res
    _, inv = np.unique(labels[clustered], return_inverse=True)
    cnt = np.bincount(inv)
    cents = np.column_stack([np.bincount(inv, weights=xy[clustered, 0]) / cnt, np.bincount(inv, weights=xy[clustered, 1]) / cnt])
    if ids.size >= 2:
        d, _ = cKDTree(cents).query(cents, k=2)
        res["median_1nn_nm"] = float(np.median(d[:, 1]))
    else:
        res["median_1nn_nm"] = float("nan")
    res["median_r_eff_nm"] = float(np.median(_hull_r_eff(xy, labels, ids)))
    res["median_n_loc"] = float(np.median(cnt))
    return res


def molecule_metrics(labels: NDArray[np.int64], sample: SyntheticSample, tau: float = TAU_EXACT) -> Dict[str, float]:
    """Molecule-level outcomes by majority matching (report section 8.1).

    exact   fraction of clusters whose majority molecule holds >= tau of the cluster's localizations AND the
            cluster holds >= tau of that molecule's localizations (one cluster = one molecule);
    merge   fraction of non-false clusters with >= 2 molecules present (each with >= 20 % of its locs);
    split   fraction of labelled molecules present in >= 2 clusters;
    missed  fraction of labelled molecules with < 50 % of their locs in any cluster;
    false   fraction of clusters with > 50 % background localizations;
    ari     adjusted Rand index between molecule id and label on clustered, non-background localizations;
    iou_mean  mean over clusters of |cluster & majority molecule| / |cluster | majority molecule|.
    """
    from sklearn.metrics import adjusted_rand_score
    mol = sample.molecule_id
    real = mol >= 0
    K = sample.mol_xy.shape[0]
    N_m = np.bincount(mol[real], minlength=K)
    labelled = N_m > 0
    n_mol = int(labelled.sum())
    clustered = labels >= 0
    ids, lab_c = np.unique(labels[clustered], return_inverse=True)
    C = ids.size
    res: Dict[str, float] = {"n_clusters": float(C), "n_mol_labelled": float(n_mol),
                             "frac_locs_clustered": float(clustered.mean()) if clustered.size else 0.0}
    if C == 0:
        res.update(exact=float("nan"), merge=float("nan"), split=0.0, missed=1.0, false=float("nan"), ari=0.0,
                   iou_mean=float("nan"), n_false=0.0, n_merged=0.0)
        return res
    n_c = np.bincount(lab_c, minlength=C)
    bg_in = np.bincount(lab_c[~real[clustered]], minlength=C)
    false = (bg_in / n_c) > FALSE_BG_SHARE
    m_cl = clustered & real
    key = lab_c[real[clustered]] * K + mol[m_cl]
    uk, cnt = np.unique(key, return_counts=True)
    kc = uk // K
    km = uk % K
    share_mol = cnt / N_m[km]
    share_cl = cnt / n_c[kc]
    present = share_mol >= PRESENT_SHARE
    n_present = np.bincount(kc[present], minlength=C)
    merged = (n_present >= 2) & ~false
    exact = np.zeros(C, bool)
    exact[kc[(share_cl >= tau) & (share_mol >= tau)]] = True
    exact &= ~false
    n_cl_present = np.bincount(km[present], minlength=K)
    split = (n_cl_present >= 2) & labelled
    in_any = np.bincount(km, weights=cnt, minlength=K)
    missed = labelled & (in_any / np.maximum(N_m, 1) < MISSED_SHARE)
    n_real_cl = int((~false).sum())
    # IoU of every cluster with its majority molecule
    iou = np.zeros(C)
    order = np.argsort(share_cl, kind="stable")
    best_cnt = np.zeros(C)
    best_mol = np.full(C, -1)
    best_cnt[kc[order]] = cnt[order]       # last write wins = largest share_cl per cluster
    best_mol[kc[order]] = km[order]
    has = best_mol >= 0
    iou[has] = best_cnt[has] / (n_c[has] + N_m[best_mol[has]] - best_cnt[has])
    res.update({
        "exact": float(exact.sum() / C), "merge": float(merged.sum() / max(n_real_cl, 1)),
        "false": float(false.sum() / C), "split": float(split.sum() / max(n_mol, 1)),
        "missed": float(missed.sum() / max(n_mol, 1)), "n_false": float(false.sum()), "n_merged": float(merged.sum()),
        "iou_mean": float(iou.mean()),
    })
    sel = m_cl
    res["ari"] = float(adjusted_rand_score(mol[sel], labels[sel])) if int(sel.sum()) > 1 else 0.0
    return res


def evaluate(labels: NDArray[np.int64], sample: SyntheticSample, oracle_readouts: Optional[Mapping[str, float]] = None,
             tau: float = TAU_EXACT) -> Dict[str, float]:
    """All metrics of one labelling; with ``oracle_readouts`` also the relative bias (%) of each readout."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (sample.xy.shape[0],):
        raise ValueError(f"labels must have one entry per localization ({sample.xy.shape[0]}), got {labels.shape}")
    res = molecule_metrics(labels, sample, tau)
    rd = readouts(sample.xy, labels, sample.arc_length_nm)
    res.update({k: v for k, v in rd.items() if k != "n_clusters"})
    if oracle_readouts is not None:
        for k in ("clusters_per_um", "median_1nn_nm", "median_r_eff_nm"):
            ref = float(oracle_readouts.get(k, float("nan")))
            res[f"bias_{k}_pct"] = float(100.0 * (rd[k] - ref) / ref) if ref and np.isfinite(ref) else float("nan")
    return res


# --------------------------------------------------------------------------- runner
METRIC_KEYS = ("exact", "merge", "split", "missed", "false", "ari", "iou_mean", "frac_locs_clustered", "n_clusters",
               "clusters_per_um", "median_1nn_nm", "median_r_eff_nm", "median_n_loc",
               "bias_clusters_per_um_pct", "bias_median_1nn_nm_pct", "bias_median_r_eff_nm_pct", "time_s")


def scenario_seed(name: str) -> int:
    """Stable 32-bit integer of a scenario name (CRC-32), the middle entry of the replicate seed."""
    return int(zlib.crc32(name.encode("utf-8")))


def run_benchmark(scenarios: Iterable[Scenario], strategies: Mapping[str, ClusteringStrategy], n_replicates: int = 3,
                  seed: int = 0, grid: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
                  include_oracle: bool = True, progress: Optional[Callable[[str], None]] = None) -> List[Dict[str, Any]]:
    """Run every strategy on every replicate of every scenario. Returns the long table (one row per
    scenario x replicate x strategy x parameter set). ``grid`` maps a strategy name to a list of keyword
    overrides applied as attributes before ``fit`` (e.g. ``{"dbscan_25_10": [{"eps": 20}, {"eps": 30}]}``);
    every parameter set is its own row and runs on a fresh deep copy of the strategy, so one set never leaks
    into the next (REVIEW_D0 B1). Replicate ``r`` of scenario ``sc`` uses
    ``default_rng([seed, crc32(sc.name), r])``: two scenarios never share molecules or background, so their
    paired differences are independent (REVIEW_D0 M2; before 2026-10-16 ``[seed, r]`` made sparse_lab1 and
    sparse_lab2 replicate r share positions)."""
    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        for rep in range(int(n_replicates)):
            rng = np.random.default_rng([int(seed), scenario_seed(sc.name), rep])
            sample = generate(sc, rng)
            oracle_rd = readouts(sample.xy, sample.oracle_labels, sample.arc_length_nm)
            if include_oracle:
                rows.append(_row(sc, rep, "oracle", {}, sample, sample.oracle_labels, 0.0, oracle_rd))
            for name, st in strategies.items():
                variants = list(grid.get(name, [{}])) if grid else [{}]
                for over in variants:
                    st_v = copy.deepcopy(st) if over else st
                    for k, v in over.items():
                        setattr(st_v, k, v)
                    t0 = time.perf_counter()
                    labels = np.asarray(st_v.fit(sample.xy), dtype=np.int64)
                    dt = time.perf_counter() - t0
                    rows.append(_row(sc, rep, name, {**strategy_params(st_v), **over}, sample, labels, dt, oracle_rd))
            if progress is not None:
                progress(f"{sc.name} replicate {rep + 1}/{n_replicates}: {sample.params['n_locs']} localizations")
    return rows


def _row(sc: Scenario, rep: int, name: str, params: Mapping[str, Any], sample: SyntheticSample,
         labels: NDArray[np.int64], dt: float, oracle_rd: Mapping[str, float]) -> Dict[str, Any]:
    m = evaluate(labels, sample, oracle_rd)
    m["time_s"] = float(dt)
    row: Dict[str, Any] = {"scenario": sc.name, "replicate": rep, "strategy": name,
                           "params": json.dumps(_json_safe(params), sort_keys=True),
                           "n_locs": sample.params["n_locs"], "n_molecules_labelled": sample.params["n_molecules_labelled"],
                           "L_um": sample.arc_length_nm / 1000.0}
    row.update(m)
    return row


def summarize(rows: Sequence[Mapping[str, Any]], keys: Sequence[str] = METRIC_KEYS) -> List[Dict[str, Any]]:
    """Mean and standard error over replicates per (scenario, strategy, params)."""
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r["scenario"]), str(r["strategy"]), str(r["params"])), []).append(r)
    out: List[Dict[str, Any]] = []
    for (sc, st, pr), g in groups.items():
        d: Dict[str, Any] = {"scenario": sc, "strategy": st, "params": pr, "n_replicates": len(g)}
        for k in keys:
            v = np.asarray([float(r.get(k, float("nan"))) for r in g], dtype=float)
            v = v[np.isfinite(v)]
            d[f"{k}_mean"] = float(v.mean()) if v.size else float("nan")
            d[f"{k}_se"] = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan")
        out.append(d)
    return out


# --------------------------------------------------------------------------- writers
def write_csv(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in cols})


def read_csv(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d: Dict[str, Any] = {}
            for k, v in row.items():
                if k in ("scenario", "strategy", "params"):
                    d[k] = v
                else:
                    try:
                        d[k] = float(v) if v not in ("", None) else float("nan")
                    except ValueError:
                        d[k] = v
            out.append(d)
    return out


def _cell(v: Any) -> Any:
    if isinstance(v, float):
        return "" if not np.isfinite(v) else repr(v)
    return v


MD_COLUMNS: Tuple[Tuple[str, str, bool, int], ...] = (
    ("exact", "1 cluster = 1 molecule (%)", True, 0), ("merge", "merge (%)", True, 0), ("split", "split (%)", True, 0),
    ("missed", "missed (%)", True, 0), ("false", "false (%)", True, 0), ("ari", "ARI", False, 2),
    ("iou_mean", "IoU mean", False, 2), ("clusters_per_um", "clusters/um", False, 1),
    ("bias_clusters_per_um_pct", "bias K/um (%)", False, 0), ("bias_median_1nn_nm_pct", "bias 1NN (%)", False, 0),
    ("bias_median_r_eff_nm_pct", "bias r_eff (%)", False, 0), ("time_s", "time (s)", False, 2),
)


def _fmt(d: Mapping[str, Any], key: str, pct: bool, nd: int) -> str:
    m = d.get(f"{key}_mean")
    se = d.get(f"{key}_se")
    if m is None or not np.isfinite(float(m)):
        return "-"
    mf, sf = float(m), (float(se) if se is not None and np.isfinite(float(se)) else None)
    if pct:
        mf *= 100.0
        sf = None if sf is None else sf * 100.0
    return f"{mf:.{nd}f}" if sf is None else f"{mf:.{nd}f} +/- {sf:.{nd}f}"


def write_markdown(summary: Sequence[Mapping[str, Any]], path: str, config: Optional[Mapping[str, Any]] = None) -> None:
    """One table per scenario: mean +/- standard error over replicates."""
    lines: List[str] = ["# Clustering benchmark (synthetic data only)", ""]
    if config:
        lines += [f"Seed {config.get('seed')}, {config.get('n_replicates')} replicates per scenario; "
                  f"strategies: {', '.join(config.get('strategies', []))}.", ""]
    by_sc: Dict[str, List[Mapping[str, Any]]] = {}
    for d in summary:
        by_sc.setdefault(str(d["scenario"]), []).append(d)
    for sc, ds in by_sc.items():
        lines += [f"## Scenario `{sc}`", "", "| strategy | params | " + " | ".join(c[1] for c in MD_COLUMNS) + " |",
                  "|---|---|" + "---|" * len(MD_COLUMNS)]
        for d in ds:
            pr = str(d.get("params", "{}"))
            pr = "" if pr in ("{}", "") else f"`{pr}`"
            lines.append(f"| {d['strategy']} | {pr} | " + " | ".join(_fmt(d, c[0], c[2], c[3]) for c in MD_COLUMNS) + " |")
        lines.append("")
    lines += ["Definitions: docs/sims/clustering_benchmark/README.md. Bias = (strategy - oracle) / oracle, paired by replicate.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# --------------------------------------------------------------------------- D11 verdicts
PRIMARY_METRICS: Tuple[Tuple[str, bool], ...] = (("exact", True), ("merge", False), ("missed", False))
"""The three primary molecule metrics of decision D11 and whether higher is better:
'exact' (1:1 molecule-cluster), 'merge' (fusions), 'missed' (losses)."""
REGIME_OF_SCENARIO = {"sparse": "sparse", "medium": "medium", "dense": "dense", "window": "window"}
"""Regime = first word of the scenario name. 'window' (flat field, other geometry) is its own regime (REVIEW_D0 M2).
The project's 18 axons sit in the sparse regime (~5 clusters/um); medium is a stress test."""

VERDICT_STATUS = "defaults verificados 2026-10-16; benchmark sintético"
"""Status written into verdicts.json: the labels judge the registry defaults verified on 2026-10-16
(VERIF_DEFAULTS_D0), on synthetic data only (D11)."""

DIAGNOSTIC_METHODS: Dict[str, str] = {
    "OPTICS": "no recomendable / diagnóstico: usar el reachability plot, no la segmentación xi (corrida F de "
              "verificación, 2026-10-16: ningún xi / min_samples / max_eps alcanza a DBSCAN 25/10 y tarda ~200x)",
}
"""Methods kept only as diagnostics: their label is always 'no recomendable' and the entry carries this note."""


def _regime(scenario: str) -> str:
    return REGIME_OF_SCENARIO.get(scenario.split("_")[0], scenario.split("_")[0])


def _paired_ci(diff: NDArray[np.float64]) -> Tuple[float, float, float]:
    """Mean paired difference and its 95 % t interval (n - 1 degrees of freedom)."""
    from scipy import stats
    n = diff.size
    mean = float(np.mean(diff))
    if n < 2:
        return mean, float("nan"), float("nan")
    se = float(np.std(diff, ddof=1) / math.sqrt(n))
    half = float(stats.t.ppf(0.975, n - 1)) * se
    return mean, mean - half, mean + half


def verdicts_from_rows(rows: Sequence[Mapping[str, Any]], reference: str = "dbscan_25_10", delta: float = 0.05,
                       time_factor: float = 10.0) -> Dict[str, Any]:
    """Decision D11 on the long table of the benchmark (synthetic data only).

    For every strategy other than ``reference`` and every regime (sparse / medium: the first word of the
    scenario name), the paired difference (strategy - reference, same scenario and replicate) of the three
    primary molecule metrics is averaged with its 95 % t interval. Labels:

    * ``recomendable``: non-inferior on all three -- the interval of 'exact' stays above -delta and the
      intervals of 'merge' and 'missed' stay below +delta -- and median time <= time_factor x the reference's;
    * ``preferible``: recomendable and superior (the interval excludes 0 on the favourable side) on at least one
      primary metric [ad hoc: D11 names the label, not the rule for it];
    * ``no recomendable``: otherwise.

    delta = 0.05 is D11's 5 points (the metrics are fractions) [ad hoc: decision D11 of the user], as are the 10x
    time factor and the 10 replicates. Strategy names drop a leading 'reg:' so the interface finds them by registry
    name. A strategy run with several parameter sets (``grid``) gets one entry per set, named
    ``<strategy> <params JSON>`` (REVIEW_D0 B1). Methods in ``DIAGNOSTIC_METHODS`` are always 'no recomendable' and
    carry their note.
    """
    params_of: Dict[str, List[str]] = {}
    for r in rows:
        ps = params_of.setdefault(str(r["strategy"]), [])
        if str(r.get("params", "")) not in ps:
            ps.append(str(r.get("params", "")))

    def ident(r: Mapping[str, Any]) -> str:
        st_ = str(r["strategy"])
        return st_ if len(params_of[st_]) <= 1 else f"{st_} {r.get('params', '')}"

    by_key: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    strategies: List[str] = []
    base_of: Dict[str, str] = {}
    for r in rows:
        st = ident(r)
        base_of[st] = str(r["strategy"])
        by_key[(st, str(r["scenario"]), int(r["replicate"]))] = r
        if st not in strategies:
            strategies.append(st)
    out: Dict[str, Any] = {"criterion": {"reference": reference, "delta": delta, "time_factor": time_factor,
                                         "metrics": [m for m, _ in PRIMARY_METRICS], "paired": True,
                                         "interval": "95 % t, paired by scenario and replicate; scenarios use "
                                                     "independent seeds (crc32 of the name)",
                                         # The labels judge the methods AT THEIR REGISTRY DEFAULTS, verified on
                                         # 2026-10-16 (VERIF_DEFAULTS_D0), on synthetic data only.
                                         "status": VERDICT_STATUS,
                                         "real_regime": "sparse (the project's axons, ~5 clusters/um); medium is a "
                                                        "stress test"},
                           "verdicts": {}}
    ref_keys = [(sc, rep) for (st, sc, rep) in by_key if st == reference]
    for st in strategies:
        if st == reference or st == "oracle":
            # the oracle is the truth, not a method anyone can choose
            continue
        # "reg:voronoi_csr:csr_iterations=10" -> "voronoi_csr": the interface looks the verdict up by registry
        # name; the full spec (with its overrides) is kept in the entry.
        name = (st[4:].split(":")[0] if st.startswith("reg:") else st)
        if base_of.get(st, st) != st:
            name = st                                  # several parameter sets: keep them apart
        per_regime: Dict[str, Any] = {}
        regimes = sorted({_regime(sc) for sc, _ in ref_keys})
        for regime in regimes:
            pairs = [(by_key.get((st, sc, rep)), by_key[(reference, sc, rep)])
                     for sc, rep in ref_keys if _regime(sc) == regime]
            pairs = [(a, b) for a, b in pairs if a is not None]
            if not pairs:
                continue
            entry: Dict[str, Any] = {"spec": st, "n_pairs": len(pairs), "metrics": {}}
            non_inferior = True
            superior = False
            for metric, higher_better in PRIMARY_METRICS:
                diff = np.array([float(a[metric]) - float(b[metric]) for a, b in pairs], dtype=float)
                mean, lo, hi = _paired_ci(diff)
                entry["metrics"][metric] = {"mean_diff": mean, "ci95": [lo, hi]}
                if higher_better:
                    non_inferior &= bool(lo > -delta)
                    superior |= bool(lo > 0)
                else:
                    non_inferior &= bool(hi < delta)
                    superior |= bool(hi < 0)
            t_st = float(np.median([float(a["time_s"]) for a, _ in pairs]))
            t_ref = float(np.median([float(b["time_s"]) for _, b in pairs]))
            entry["median_time_s"] = t_st
            entry["reference_median_time_s"] = t_ref
            time_ok = bool(t_st <= time_factor * max(t_ref, 1e-9))
            entry["time_ok"] = time_ok
            entry["label"] = ("preferible" if non_inferior and time_ok and superior
                              else "recomendable" if non_inferior and time_ok
                              else "no recomendable")
            if name in DIAGNOSTIC_METHODS:
                entry["label"] = "no recomendable"
                entry["role"] = DIAGNOSTIC_METHODS[name]
            per_regime[regime] = entry
        out["verdicts"][name] = per_regime
    return out


def write_verdicts(rows: Sequence[Mapping[str, Any]], path: str, **kw: Any) -> Dict[str, Any]:
    v = verdicts_from_rows(rows, **kw)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(v), fh, indent=2)
    return v


def _json_safe(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _run_one_spec(args: Tuple[List[Scenario], str, int, int, bool]) -> List[Dict[str, Any]]:
    """Worker of ``run_and_write(jobs > 1)``: one strategy spec over every scenario and replicate."""
    scenarios, spec, n_replicates, seed, with_oracle = args
    return run_benchmark(scenarios, {spec: make_strategy(spec)}, n_replicates, seed, include_oracle=with_oracle)


def _merge_rows(parts: Sequence[List[Dict[str, Any]]], scenarios: Sequence[Scenario],
                specs: Sequence[str]) -> List[Dict[str, Any]]:
    """Rows of the per-spec workers in the order of the serial runner (scenario, replicate, oracle, specs)."""
    sc_order = {sc.name: i for i, sc in enumerate(scenarios)}
    st_order = {"oracle": -1, **{s: i for i, s in enumerate(specs)}}
    rows = [r for part in parts for r in part]
    rows.sort(key=lambda r: (sc_order[str(r["scenario"])], int(r["replicate"]), st_order.get(str(r["strategy"]), 99)))
    return rows


def run_and_write(scenarios: Sequence[Scenario], specs: Sequence[str], n_replicates: int, seed: int, out_dir: str,
                  progress: Optional[Callable[[str], None]] = None, jobs: int = 1) -> Dict[str, str]:
    """Run the benchmark for strategy specs (see :func:`make_strategy`) and write the four output files.

    ``jobs`` > 1 runs one process per strategy spec (at most ``jobs`` at a time). The labels are the same as the
    serial run (every spec regenerates the same samples from the seed); the times are measured under the load of the
    other processes, so the time ratios of D11 are approximate (recorded as ``jobs`` in the config)."""
    os.makedirs(out_dir, exist_ok=True)
    strategies = {s: make_strategy(s) for s in specs if s != "oracle"}
    t0 = time.perf_counter()
    if int(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        todo = [(list(scenarios), s, int(n_replicates), int(seed), i == 0) for i, s in enumerate(strategies)]
        parts: List[List[Dict[str, Any]]] = []
        with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            for spec_rows in ex.map(_run_one_spec, todo):
                parts.append(spec_rows)
                if progress is not None and spec_rows:
                    progress(f"done: {spec_rows[-1]['strategy']} ({time.perf_counter() - t0:.0f} s)")
        rows = _merge_rows(parts, scenarios, list(strategies))
    else:
        rows = run_benchmark(scenarios, strategies, n_replicates, seed, progress=progress)
    summary = summarize(rows)
    config = {"seed": seed, "n_replicates": n_replicates, "strategies": list(specs), "jobs": int(jobs),
              "replicate_seed": "numpy default_rng([seed, crc32(scenario name), replicate])",
              "verdict_status": VERDICT_STATUS,
              "strategy_params": {s: _json_safe(strategy_params(st)) for s, st in strategies.items()},
              "scenarios": [sc.as_dict() for sc in scenarios], "n_rows": len(rows),
              "elapsed_s": time.perf_counter() - t0,
              "constants": {"SIGMA_LINK_NM": SIGMA_LINK_NM, "EPITOPE_SEP_NM": list(EPITOPE_SEP_NM),
                            "MEMBRANE_ROUGH_NM": MEMBRANE_ROUGH_NM, "N_FRAMES": N_FRAMES, "ON_FRAMES": list(ON_FRAMES),
                            "BG_BAND_NM": BG_BAND_NM, "HARD_CORE_FRACTION": HARD_CORE_FRACTION,
                            "SIGMA_LOC_CV": SIGMA_LOC_CV, "TAU_EXACT": TAU_EXACT, "PRESENT_SHARE": PRESENT_SHARE}}
    paths = {"long": os.path.join(out_dir, "benchmark_long.csv"), "summary": os.path.join(out_dir, "benchmark_summary.csv"),
             "markdown": os.path.join(out_dir, "benchmark_summary.md"), "config": os.path.join(out_dir, "benchmark_config.json")}
    write_csv(rows, paths["long"])
    write_csv(summary, paths["summary"])
    write_markdown(summary, paths["markdown"], config)
    # D11: the verdicts against the first strategy given (the reference, DBSCAN 25/10 by default).
    paths["verdicts"] = os.path.join(out_dir, "verdicts.json")
    write_verdicts(rows, paths["verdicts"], reference=list(specs)[0])
    with open(paths["config"], "w", encoding="utf-8") as fh:
        json.dump(_json_safe(config), fh, indent=2)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.cluster_benchmark",
                                 description="Clustering benchmark on synthetic SMLM scenarios (synthetic data only).")
    ap.add_argument("--scenario", action="append", default=[], help="preset name (repeatable); see --list")
    ap.add_argument("--all-scenarios", action="store_true", help="run every preset")
    ap.add_argument("--strategies", default=f"dbscan_{EPS_NM_DEFAULT:g}_{MIN_SAMPLES_DEFAULT},hdbscan_10_5",
                    help="comma-separated specs: dbscan_<eps>_<min_samples>, hdbscan_<mcs>_<ms>, external:<csv>")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmark_out")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    ap.add_argument("--verdicts-from", default="", help="only (re)write verdicts.json (D11) from this long CSV")
    ap.add_argument("--jobs", type=int, default=1, help="processes, one per strategy spec (labels identical; times "
                                                        "measured under load)")
    ap.add_argument("--strategies-json", default="", help="JSON file with a list of specs (for specs with commas, "
                                                          "e.g. reg:OPTICS:xi=0.1,max_eps=100); overrides --strategies")
    a = ap.parse_args(argv)
    if a.list:
        for k, sc in PRESETS.items():
            print(f"{k:22s} geometry={sc.geometry} spacing={sc.spacing_nm:g} p_label={sc.p_label:g} c_label={sc.c_label:g} "
                  f"sigma={sc.sigma_loc:g} bg={sc.bg_labels_per_um:g}/um,{sc.bg_locs_per_um2:g}/um2  [{sc.origin}]")
        return 0
    names = list(PRESETS) if a.all_scenarios else list(a.scenario)
    if not names:
        ap.error("give --scenario NAME (repeatable) or --all-scenarios")
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        ap.error(f"unknown scenario(s) {unknown}; see --list")
    specs = [s.strip() for s in a.strategies.split(",") if s.strip()]
    if a.strategies_json:
        with open(a.strategies_json, encoding="utf-8") as fh:
            specs = [str(x) for x in json.load(fh)]
    if a.verdicts_from:
        rows = read_csv(a.verdicts_from)
        v = write_verdicts(rows, os.path.join(a.out, "verdicts.json"), reference=specs[0])
        for name, per in v["verdicts"].items():
            print(name, {k: e["label"] for k, e in per.items()})
        return 0
    paths = run_and_write([PRESETS[n] for n in names], specs, a.replicates, a.seed, a.out, progress=print,
                          jobs=a.jobs)
    for k, p in paths.items():
        print(f"{k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
