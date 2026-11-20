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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError, cKDTree

from tools.clustering_strategies import ClusteringStrategy, DBSCANStrategy

# --------------------------------------------------------------------------- defaults (origin in brackets)
SIGMA_LINK_NM = 10.0          # [ad hoc, of the order of Fruh et al. 2021] they measure, for primary + secondary IgG,
                              # a mean displacement of 10.0 nm, radial sigma ~7.5 nm and a maximum of ~19 nm; the
                              # old "19 / sqrt(3)" divided a maximum, not an SD: 10 nm per axis is not their value
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

# Emitter-level truth (phase F2 of the BaGoL plan). An "emitter" is a fluorescent label stack (one secondary antibody
# with its dyes), never a protein (Fazel et al. 2022: "an 'emitter' as the fluorescently tagged molecule").
ELE_DEFAULT = 0.65            # [literature] Thevathasan et al. 2019 (Nat Methods 16:1045): effective labelling
                              # efficiency ~65 % for indirect immunofluorescence, as quoted in the project's clustering
                              # report (section 3); page not re-checked here
N_SECONDARY_PMF = (0.554, 0.327, 0.119)  # P(1, 2, 3 secondaries per primary) [derivation + ad hoc]: 1 + Poisson(0.59)
                              # truncated at 3 (mean 1.57), close to the ~1.59 secondaries per primary of Ehmann et al.
                              # 2014; the truncated-Poisson shape and the cap at 3 are ad hoc (task range 1-3)
DOL_MEAN_DEFAULT = 1.5        # [literature] Dempsey et al. 2011 Online Methods (PDF p. 13): ~1.5 Alexa 647 per bovine
                              # anti-goat secondary; Poisson number of dyes per secondary [ad hoc: random lysine
                              # conjugation]. The lab's Invitrogen secondaries have no published DOL.
SIGMA_LAYER_NM = 6.5          # [literature] Jimenez, Friedl & Leterrier 2020 (Methods 174:100, PDF p. 10): "s.d. ~6.5 nm"
                              # per antibody layer, read here as per-axis s.d. [assumption]. Two layers give 9.2 nm per
                              # axis, close to SIGMA_LINK_NM; Zwettler et al. 2020 report ~17.5 nm as the geometric
                              # primary + secondary offset (a shell radius, not a dispersion; ADDENDUM_biologia B)
DYE_SPREAD_NM = 3.0           # [derivation] dyes anywhere on an IgG of ~10-15 nm (ADDENDUM_biologia B): uniform in a
                              # ball of radius ~6.5 nm has per-axis s.d. r / sqrt(5) ~ 2.9 nm
CYCLES_MEAN_DEFAULT = 14.0    # [literature] Dempsey et al. 2011 Table 1: Alexa Fluor 647, 10 mM MEA, 14 switching cycles
                              # (a lower bound by the authors' own statement)
EMITTER_MATCH_FACTOR = 3.0    # [literature] Fazel et al. 2022 (Nat Commun 13:7152, Fig. 4 and Methods): a true and a
                              # found emitter are a pair when within 3 sigma, pairs chosen by the Hungarian algorithm
CYCLE_DISTS = ("geometric", "gamma", "poisson")


@dataclass
class EmitterModel:
    """Antibody-stack labelling of every molecule, with truth per emitter (BaGoL plan, phase F2).

    Hierarchy: molecule -> 2 epitopes (EPITOPE_SEP_NM apart) -> a primary IgG on each epitope with probability
    ``ele`` (offset N(0, ``sigma_primary_nm``) per axis) -> ``n_secondary`` secondaries on that primary, each offset
    N(0, ``sigma_secondary_nm``) from it -> ``dol`` dyes per secondary, each offset N(0, ``dye_spread_nm``) from it ->
    switching cycles per dye (``cycles_dist`` with mean ``cycles_mean``) -> 1-3 localizations per cycle (ON_FRAMES).

    One secondary with its dyes is a **stack** (``stack_id``): its dyes sit closer than the localization precision,
    so a BaGoL-like method sees one emitter per stack when stacks are more than ~sigma apart. Emitters are
    fluorescent label stacks, not proteins. Non-specific (background) labels are stacks too, with dyes drawn from the
    same DOL distribution.

    ``n_secondary``: an int (fixed number) or a sequence of probabilities of 1, 2, 3, ... secondaries.
    ``dol_dist``: "poisson" (mean ``dol``; 0 dyes = a dark secondary) or "fixed" (exactly ``int(dol)`` dyes).
    ``cycles_dist``: "geometric" (>= 1 cycle, mean ``cycles_mean``; the legacy model), "gamma" (Poisson-gamma
    mixture with shape ``cycles_shape`` and mean ``cycles_mean``; 0 allowed: a dye that never switches) or
    "poisson" (mean ``cycles_mean``). Defaults and their origins: module constants. ``EmitterModel.legacy(sc)``
    gives the parameters that reproduce the generator of 26a1cd4 bit for bit.
    """

    ele: float = ELE_DEFAULT
    n_secondary: Union[int, Tuple[float, ...]] = N_SECONDARY_PMF
    dol: float = DOL_MEAN_DEFAULT
    dol_dist: str = "poisson"
    sigma_primary_nm: float = SIGMA_LAYER_NM
    sigma_secondary_nm: float = SIGMA_LAYER_NM
    dye_spread_nm: float = DYE_SPREAD_NM
    cycles_dist: str = "geometric"
    cycles_mean: float = CYCLES_MEAN_DEFAULT
    cycles_shape: float = 1.0      # [literature] Fazel et al. 2022 Methods: exponential (gamma k = 1) blinking counts
    origin: str = "BaGoL plan F2: antibody stacks (constants of tools.cluster_benchmark)"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.ele) <= 1.0:
            raise ValueError(f"ele must be in [0, 1], got {self.ele}")
        if isinstance(self.n_secondary, (int, np.integer)):
            if int(self.n_secondary) < 1:
                raise ValueError(f"n_secondary must be >= 1, got {self.n_secondary}")
        else:
            pmf = np.asarray(self.n_secondary, dtype=float)
            if pmf.ndim != 1 or pmf.size == 0 or np.any(pmf < 0) or not pmf.sum() > 0:
                raise ValueError(f"n_secondary must be an int or a probability vector, got {self.n_secondary!r}")
            self.n_secondary = tuple(float(p) for p in pmf)
        if self.dol_dist not in ("poisson", "fixed"):
            raise ValueError(f"dol_dist must be 'poisson' or 'fixed', got {self.dol_dist!r}")
        if self.dol_dist == "fixed" and (float(self.dol) != int(self.dol) or int(self.dol) < 1):
            raise ValueError(f"a fixed dol must be an integer >= 1, got {self.dol}")
        if not float(self.dol) > 0:
            raise ValueError(f"dol must be > 0, got {self.dol}")
        if self.cycles_dist not in CYCLE_DISTS:
            raise ValueError(f"cycles_dist must be one of {CYCLE_DISTS}, got {self.cycles_dist!r}")
        if self.cycles_dist == "geometric" and not float(self.cycles_mean) >= 1.0:
            raise ValueError(f"a geometric cycle count has mean >= 1, got {self.cycles_mean}")
        if not (float(self.cycles_mean) > 0 and float(self.cycles_shape) > 0):
            raise ValueError("cycles_mean and cycles_shape must be > 0")
        for k in ("sigma_primary_nm", "sigma_secondary_nm", "dye_spread_nm"):
            if not float(getattr(self, k)) >= 0.0:
                raise ValueError(f"{k} must be >= 0")

    @classmethod
    def legacy(cls, sc: "Scenario") -> "EmitterModel":
        """The labelling of the generator of 26a1cd4: one label per labelled epitope (p_label), one linkage draw of
        ``sc.sigma_link`` per label, geometric cycles of mean ``sc.c_label``. Same random draws, same samples."""
        return cls(ele=sc.p_label, n_secondary=1, dol=1, dol_dist="fixed", sigma_primary_nm=0.0,
                   sigma_secondary_nm=sc.sigma_link, dye_spread_nm=0.0, cycles_dist="geometric",
                   cycles_mean=sc.c_label, origin="legacy: one label per epitope (generator of 26a1cd4)")

    @classmethod
    def from_prior(cls, prior: Any, loc_per_cycle: Optional[float] = None, **overrides: Any) -> "EmitterModel":
        """Switching cycles from a xi prior (duck-typed: ``tools.emitters_prior.EmitterPrior`` of phase F1, or any
        object with ``mean``, ``shape`` and optionally ``k``, ``switching_cycles``, ``loc_per_cycle``).

        The cycle mean is ``prior.switching_cycles`` when present, else ``prior.mean / loc_per_cycle`` with
        ``loc_per_cycle`` from the argument, the prior, or the mean of ON_FRAMES (2) in this order. A gamma prior
        gives ``cycles_dist="gamma"`` with its shape k; a Poisson prior gives "poisson"."""
        cyc = getattr(prior, "switching_cycles", None)
        if cyc is None:
            lpc = loc_per_cycle if loc_per_cycle is not None else getattr(prior, "loc_per_cycle", None)
            lpc = float(lpc) if lpc else float(np.mean(ON_FRAMES))
            cyc = float(prior.mean) / lpc
        shape = str(getattr(prior, "shape", "gamma"))
        kw: Dict[str, Any] = {"cycles_mean": float(cyc)}
        if shape == "poisson":
            kw["cycles_dist"] = "poisson"
        else:
            kw["cycles_dist"] = "gamma"
            kw["cycles_shape"] = float(getattr(prior, "k", None) or 1.0)
        src = getattr(prior, "origin", "") or getattr(prior, "source", "")
        kw["origin"] = f"cycles from xi prior ({src})" if src else "cycles from xi prior"
        kw.update(overrides)
        return cls(**kw)


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
    emitters: Optional[EmitterModel] = None   # None: the legacy labelling (p_label, sigma_link, c_label), exactly
                                              # as in 26a1cd4; an EmitterModel replaces those three fields

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        em = d.pop("emitters", None)             # legacy scenarios keep the keys (and the configs) of 26a1cd4
        if isinstance(em, dict):
            d["emitters"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in em.items()}
        return d


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
    # Emitter-level truth (BaGoL plan F2). A label (``label_id``) is one dye; a stack (``stack_id``) is one secondary
    # antibody with its dyes, i.e. one emitter. In the legacy model every label is its own stack (stack_id == label_id).
    stack_id: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, np.int64))        # per localization;
                                                                                            # -1 isolated background
    label_xy: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 2)))         # (n_labels, 2) dye positions
    label_stack: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, np.int64))   # stack of every dye
    stack_xy: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 2)))         # (n_stacks, 2) centroid of
                                                                                            # the stack's dyes
    stack_molecule: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, np.int64))  # >= 0 molecule, <= -2
                                                                                            # non-specific stack
    stack_epitope: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, np.int64))   # index into the 2K
                                                                                            # epitopes; -1 background

    @property
    def is_background(self) -> NDArray[np.bool_]:
        return self.molecule_id < 0

    @property
    def stack_n_locs(self) -> NDArray[np.int64]:
        """Localizations per stack (0 for dark stacks: no dye, or no dye that switched inside the movie)."""
        sid = self.stack_id[self.stack_id >= 0]
        return np.bincount(sid, minlength=self.stack_xy.shape[0]).astype(np.int64)

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
    # The legacy model is the EmitterModel with one label per epitope: same draws in the same order (no draw is made
    # for a fixed n_secondary, a fixed DOL, a zero primary offset or a zero dye spread), so the samples of 26a1cd4
    # are reproduced bit for bit (test_cluster_benchmark_emitters.py).
    em = sc.emitters if sc.emitters is not None else EmitterModel.legacy(sc)
    # epitopes: 2 per molecule, 5-10 nm apart, random orientation; each carries a primary with probability ele
    ang_e = rng.uniform(0, 2 * np.pi, K)
    half = 0.5 * np.asarray(rng.uniform(*EPITOPE_SEP_NM, K), dtype=float)
    u = np.column_stack([np.cos(ang_e), np.sin(ang_e)])
    epi = np.concatenate([mol_xy + half[:, None] * u, mol_xy - half[:, None] * u])
    epi_mol = np.concatenate([np.arange(K), np.arange(K)])
    labelled = rng.uniform(size=2 * K) < em.ele
    epi_idx = np.flatnonzero(labelled).astype(np.int64)
    st_pos, st_parent = _draw_stacks(rng, em, epi[labelled])
    st_mol = epi_mol[labelled].astype(np.int64)[st_parent]
    st_epi = epi_idx[st_parent]
    n_spec = st_pos.shape[0]
    dye_stack = np.repeat(np.arange(n_spec, dtype=np.int64), _draw_dol(rng, em, n_spec))
    lab_pos = _spread_dyes(rng, em, st_pos, dye_stack)
    mol_n_labels = np.bincount(st_mol[dye_stack], minlength=K).astype(np.int64)
    # background labels (non-specific stacks that blink like a label)
    n_bg_lab = int(rng.poisson(n_bg_lab_mean))
    if n_bg_lab:
        bg_pos = _background_positions(rng, con, arc_nm, sc.window_um, n_bg_lab)
        bg_dye_stack = np.repeat(np.arange(n_bg_lab, dtype=np.int64), _draw_dol(rng, em, n_bg_lab))
        lab_pos = np.concatenate([lab_pos, _spread_dyes(rng, em, bg_pos, bg_dye_stack)])
        dye_stack = np.concatenate([dye_stack, n_spec + bg_dye_stack])
        st_pos = np.concatenate([st_pos, bg_pos])
        st_mol = np.concatenate([st_mol, -2 - np.arange(n_bg_lab, dtype=np.int64)])
        st_epi = np.concatenate([st_epi, np.full(n_bg_lab, -1, dtype=np.int64)])
    lab_mol = st_mol[dye_stack]
    n_lab = lab_pos.shape[0]
    # blinking: number of cycles per label (geometric in the legacy model), each 1-3 consecutive frames
    n_cyc = _draw_cycles(rng, em, n_lab)
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
    loc_stack = dye_stack[loc_label]
    event = loc_cyc.astype(np.int64)
    # isolated background localizations
    n_bg_loc = int(rng.poisson(sc.bg_locs_per_um2 * bg_region_um2))
    if n_bg_loc:
        xy = np.concatenate([xy, _background_positions(rng, con, arc_nm, sc.window_um, n_bg_loc)])
        frame = np.concatenate([frame, rng.integers(0, N_FRAMES, n_bg_loc)])
        mol = np.concatenate([mol, np.full(n_bg_loc, -1)])
        loc_label = np.concatenate([loc_label, np.full(n_bg_loc, -1)])
        loc_stack = np.concatenate([loc_stack, np.full(n_bg_loc, -1, dtype=np.int64)])
        event = np.concatenate([event, n_cyc_tot + np.arange(n_bg_loc)])
        sig = np.concatenate([sig, np.full(n_bg_loc, sc.sigma_loc)])
    order = np.argsort(frame, kind="stable")
    params = sc.as_dict()
    params.update({"realised_L_nm": arc_nm, "n_molecules": int(K), "n_molecules_labelled": int((mol_n_labels > 0).sum()),
                   "n_background_labels": n_bg_lab, "n_background_locs": n_bg_loc, "n_locs": int(xy.shape[0])})
    # The centre of a stack is the centroid of its dyes (what an emitter fit converges to); with no dye spread (and
    # in the legacy model) it is the secondary's own position, untouched.
    stack_xy = st_pos.copy()
    if em.dye_spread_nm > 0 and dye_stack.size:
        n_d = np.bincount(dye_stack, minlength=st_pos.shape[0])
        has = n_d > 0
        stack_xy[has, 0] = np.bincount(dye_stack, weights=lab_pos[:, 0], minlength=st_pos.shape[0])[has] / n_d[has]
        stack_xy[has, 1] = np.bincount(dye_stack, weights=lab_pos[:, 1], minlength=st_pos.shape[0])[has] / n_d[has]
    if sc.emitters is not None:
        loc_st = loc_stack[loc_stack >= 0]
        params.update({"n_stacks": int(st_pos.shape[0]), "n_dyes": int(n_lab),
                       "n_stacks_visible": int(np.unique(loc_st).size)})
    return SyntheticSample(xy=xy[order], frame=frame[order].astype(np.int64), molecule_id=mol[order].astype(np.int64),
                           label_id=loc_label[order].astype(np.int64), event_id=event[order].astype(np.int64),
                           sigma=sig[order], mol_xy=mol_xy, mol_n_labels=mol_n_labels, arc_length_nm=arc_nm,
                           params=params, stack_id=loc_stack[order].astype(np.int64), label_xy=lab_pos,
                           label_stack=dye_stack.astype(np.int64), stack_xy=stack_xy,
                           stack_molecule=st_mol.astype(np.int64), stack_epitope=st_epi.astype(np.int64))


def _draw_stacks(rng: np.random.Generator, em: EmitterModel,
                 epi_pos: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Secondary-antibody stacks on the labelled epitopes: their positions and the index of their epitope.

    Draw order (fixed, part of the bit-for-bit contract): number of secondaries (only if not a fixed int), primary
    offsets (only if ``sigma_primary_nm`` > 0), secondary offsets (always, as the legacy linkage draw)."""
    n_epi = int(epi_pos.shape[0])
    if isinstance(em.n_secondary, (int, np.integer)):
        n2: NDArray[np.int64] = np.full(n_epi, int(em.n_secondary), dtype=np.int64)
    else:
        pmf = np.asarray(em.n_secondary, dtype=float)
        n2 = 1 + rng.choice(pmf.size, n_epi, p=pmf / pmf.sum()).astype(np.int64)
    prim = epi_pos + rng.normal(0.0, em.sigma_primary_nm, (n_epi, 2)) if em.sigma_primary_nm > 0 else epi_pos
    parent = np.repeat(np.arange(n_epi, dtype=np.int64), n2)
    pos = prim[parent] + rng.normal(0.0, em.sigma_secondary_nm, (int(parent.size), 2))
    return pos, parent


def _draw_dol(rng: np.random.Generator, em: EmitterModel, n: int) -> NDArray[np.int64]:
    """Dyes per stack: fixed (no draw) or Poisson(dol)."""
    if em.dol_dist == "fixed":
        return np.full(int(n), int(em.dol), dtype=np.int64)
    return rng.poisson(float(em.dol), int(n)).astype(np.int64)


def _spread_dyes(rng: np.random.Generator, em: EmitterModel, stack_pos: NDArray[np.float64],
                 dye_stack: NDArray[np.int64]) -> NDArray[np.float64]:
    """Dye positions: the stack's position plus N(0, dye_spread_nm) per axis (no draw when the spread is 0)."""
    pos = stack_pos[dye_stack]
    if em.dye_spread_nm > 0:
        pos = pos + rng.normal(0.0, em.dye_spread_nm, (int(dye_stack.size), 2))
    return np.asarray(pos, dtype=float).reshape(-1, 2)


def _draw_cycles(rng: np.random.Generator, em: EmitterModel, n: int) -> NDArray[np.int64]:
    """Switching cycles per dye. geometric: support >= 1, mean ``cycles_mean`` (the legacy draw, unchanged);
    gamma: Poisson(Gamma(k, mean / k)), support >= 0 (a negative binomial with mean ``cycles_mean``); poisson."""
    if em.cycles_dist == "geometric":
        return rng.geometric(1.0 / em.cycles_mean, int(n)).astype(np.int64)
    if em.cycles_dist == "gamma":
        lam = rng.gamma(float(em.cycles_shape), float(em.cycles_mean) / float(em.cycles_shape), int(n))
        return rng.poisson(lam).astype(np.int64)
    return rng.poisson(float(em.cycles_mean), int(n)).astype(np.int64)


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


# --------------------------------------------------------------------------- emitter metrics (BaGoL plan F2)
def emitter_truth(sample: SyntheticSample, min_locs: int = 1,
                  include_background: bool = True) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """True emitters of a sample: positions and stack ids of the stacks with at least ``min_locs`` localizations
    (a stack that never switched cannot be recovered by any method). ``include_background`` keeps the non-specific
    stacks, which are real fluorescent marks for an emitter method."""
    n_loc = sample.stack_n_locs
    keep = n_loc >= int(min_locs)
    if not include_background:
        keep &= sample.stack_molecule >= 0
    ids = np.flatnonzero(keep).astype(np.int64)
    return sample.stack_xy[ids].reshape(-1, 2), ids


def oracle_emitters(sample: SyntheticSample, min_locs: int = 1, include_background: bool = True) -> NDArray[np.float64]:
    """An oracle that returns the true position of every visible stack (scores JAC = 1, RMSE = 0)."""
    return emitter_truth(sample, min_locs, include_background)[0]


def _match_hungarian(est: NDArray[np.float64], true: NDArray[np.float64],
                     radius: NDArray[np.float64]) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """One-to-one pairs (estimate i, truth j) with |est_i - true_j| <= radius_i that maximise the number of pairs
    and, among those, minimise the summed distance (Hungarian algorithm on each connected component of the
    "within radius" graph, so the cost matrices stay small)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n_e, n_t = est.shape[0], true.shape[0]
    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0))
    if n_e == 0 or n_t == 0:
        return empty
    r_max = float(np.max(radius))
    cand = cKDTree(est).query_ball_tree(cKDTree(true), r_max)
    ei: List[int] = []
    tj: List[int] = []
    for i, js in enumerate(cand):
        for j in js:
            if float(np.hypot(*(est[i] - true[j]))) <= float(radius[i]):
                ei.append(i)
                tj.append(j)
    if not ei:
        return empty
    e_arr, t_arr = np.asarray(ei, np.int64), np.asarray(tj, np.int64)
    g = coo_matrix((np.ones(e_arr.size), (e_arr, n_e + t_arr)), shape=(n_e + n_t, n_e + n_t))
    _, comp = connected_components(g, directed=False)
    d_all = np.hypot(est[e_arr, 0] - true[t_arr, 0], est[e_arr, 1] - true[t_arr, 1])
    out_e: List[int] = []
    out_t: List[int] = []
    big = 1e9          # a forbidden pair costs more than any sum of allowed distances
    for c in np.unique(comp[e_arr]):
        sel = comp[e_arr] == c
        es, ts = np.unique(e_arr[sel]), np.unique(t_arr[sel])
        cost = np.full((es.size, ts.size), big)
        cost[np.searchsorted(es, e_arr[sel]), np.searchsorted(ts, t_arr[sel])] = d_all[sel]
        r, k = linear_sum_assignment(cost)
        ok = cost[r, k] < big
        out_e.extend(es[r[ok]].tolist())
        out_t.extend(ts[k[ok]].tolist())
    oe, ot = np.asarray(out_e, np.int64), np.asarray(out_t, np.int64)
    return oe, ot, np.hypot(est[oe, 0] - true[ot, 0], est[oe, 1] - true[ot, 1])


def emitter_metrics(est_xy: Any, sample: SyntheticSample, est_sigma: Any = None,
                    match_factor: float = EMITTER_MATCH_FACTOR, min_locs: int = 1, include_background: bool = True,
                    details: bool = False) -> Dict[str, Any]:
    """Recovered emitters against the true stacks of ``sample``, as in Fazel et al. 2022 (Fig. 4, Methods).

    ``est_xy`` (M, 2) are estimated emitter positions in nm. A true stack and an estimate are a pair when the truth
    falls within ``match_factor`` x sigma of the estimate; sigma is ``est_sigma`` (per estimate, or one number: the
    precision of each estimate, as in the paper) or, when not given, the nominal localization precision of the
    scenario (``params['sigma_loc']``) [ad hoc: the lenient choice for dSTORM, where an emitter's precision is
    unknown before a fit]. Pairs come from the Hungarian algorithm (most pairs, then least summed distance).

    Returns (fractions in 0-1, distances in nm):
      n_true, n_est, tp, fp, fn, jac = tp / (tp + fp + fn), recall, precision, rmse_nm (root mean square distance
      of the pairs), mean_dist_nm (Fazel's "mean of the differences"), match_radius_nm (median radius used);
      over-grouping: ``n_true_absorbed`` / ``frac_true_absorbed`` -- unmatched true stacks with an estimate within
      the radius (the estimate went to a neighbour: two stacks merged into one emitter);
      under-grouping: ``n_est_split`` / ``frac_est_split`` -- unmatched estimates within the radius of a true stack
      (an extra emitter for a stack already found);
      per molecule (labelled molecules with >= 1 visible specific stack): every estimate goes to the molecule of
      its nearest visible true stack within its radius (not one-to-one); ``k_true_mean`` and ``k_est_mean`` (stacks
      and estimates per molecule), ``k_bias_mean`` (mean of K_est - K_true), ``frac_k_exact``, ``frac_k_under``
      (K_est < K_true: fusion or loss) and ``frac_k_over`` (K_est > K_true: splitting).
    With ``details`` the per-pair and per-molecule arrays are added (``pairs_est``, ``pairs_true_stack``,
    ``pairs_dist_nm``, ``molecule_ids``, ``k_true``, ``k_est``).
    """
    est = np.asarray(est_xy, dtype=float).reshape(-1, 2)
    true_xy, true_ids = emitter_truth(sample, min_locs, include_background)
    if est_sigma is None:
        sig: NDArray[np.float64] = np.full(est.shape[0], float(sample.params.get("sigma_loc", float(np.median(sample.sigma)))))
    else:
        sig = np.broadcast_to(np.asarray(est_sigma, dtype=float), (est.shape[0],)).astype(float)
    radius = float(match_factor) * sig
    pe, pt, d = _match_hungarian(est, true_xy, radius)
    n_t, n_e, tp = int(true_ids.size), int(est.shape[0]), int(pe.size)
    fp, fn = n_e - tp, n_t - tp
    res: Dict[str, Any] = {
        "n_true": float(n_t), "n_est": float(n_e), "tp": float(tp), "fp": float(fp), "fn": float(fn),
        "jac": float(tp / (tp + fp + fn)) if (tp + fp + fn) else float("nan"),
        "recall": float(tp / n_t) if n_t else float("nan"), "precision": float(tp / n_e) if n_e else float("nan"),
        "rmse_nm": float(np.sqrt(np.mean(d ** 2))) if tp else float("nan"),
        "mean_dist_nm": float(np.mean(d)) if tp else float("nan"),
        "match_radius_nm": float(np.median(radius)) if n_e else float("nan"),
    }
    # over- and under-grouping
    matched_t = np.zeros(n_t, bool)
    matched_t[pt] = True
    matched_e = np.zeros(n_e, bool)
    matched_e[pe] = True
    if n_e and n_t:
        tree_t = cKDTree(true_xy)
        d_nn, j_nn = tree_t.query(est, k=1)
        near_e = d_nn <= radius                                  # estimate within its radius of some true stack
        absorbed = np.zeros(n_t, bool)
        tree_e = cKDTree(est)
        for j in np.flatnonzero(~matched_t):
            for i in tree_e.query_ball_point(true_xy[j], float(np.max(radius))):
                if float(np.hypot(*(est[i] - true_xy[j]))) <= float(radius[i]):
                    absorbed[j] = True
                    break
        n_abs = int(absorbed.sum())
        n_split = int((~matched_e & near_e).sum())
    else:
        near_e = np.zeros(n_e, bool)
        j_nn = np.zeros(n_e, np.int64)
        n_abs, n_split = 0, 0
    res.update({"n_true_absorbed": float(n_abs), "frac_true_absorbed": float(n_abs / n_t) if n_t else float("nan"),
                "n_est_split": float(n_split), "frac_est_split": float(n_split / n_e) if n_e else float("nan")})
    # emitters per molecule
    t_mol = sample.stack_molecule[true_ids]
    spec = t_mol >= 0
    mols = np.unique(t_mol[spec])
    k_true = np.array([int(np.sum(t_mol == m)) for m in mols], dtype=np.int64)
    est_mol = np.where(near_e, t_mol[j_nn] if n_t else -1, -1) if n_e else np.empty(0, np.int64)
    k_est = np.array([int(np.sum(est_mol == m)) for m in mols], dtype=np.int64)
    if mols.size:
        res.update({"n_molecules_visible": float(mols.size), "k_true_mean": float(k_true.mean()),
                    "k_est_mean": float(k_est.mean()), "k_bias_mean": float(np.mean(k_est - k_true)),
                    "frac_k_exact": float(np.mean(k_est == k_true)), "frac_k_under": float(np.mean(k_est < k_true)),
                    "frac_k_over": float(np.mean(k_est > k_true))})
    else:
        res.update({k: float("nan") for k in ("k_true_mean", "k_est_mean", "k_bias_mean", "frac_k_exact",
                                              "frac_k_under", "frac_k_over")})
        res["n_molecules_visible"] = 0.0
    if details:
        res.update({"pairs_est": pe, "pairs_true_stack": true_ids[pt], "pairs_dist_nm": d, "molecule_ids": mols,
                    "k_true": k_true, "k_est": k_est})
    return res


# --------------------------------------------------------------------------- SMD export (BaGoL input)
SMD_COLUMNS = ("X", "Y", "X_SE", "Y_SE", "FrameNum")
"""Columns of the SMD structure read by BaGoL (LidkeLab/BaGoL README: X, Y, X_SE, Y_SE, FrameNum, in nm)."""


def event_table(sample: SyntheticSample) -> Dict[str, NDArray[Any]]:
    """Localizations merged per switching event (``event_id``): inverse-variance weighted position, standard error
    1 / sqrt(sum 1 / sigma^2), first frame, and the truth of the event (one stack and one molecule per event).
    This is what a perfect frame connection returns; BaGoL expects frame-connected input."""
    ev, inv = np.unique(sample.event_id, return_inverse=True)
    w = 1.0 / sample.sigma ** 2
    sw = np.bincount(inv, weights=w)
    x = np.bincount(inv, weights=w * sample.xy[:, 0]) / sw
    y = np.bincount(inv, weights=w * sample.xy[:, 1]) / sw
    first = np.full(ev.size, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, inv, sample.frame)
    pick = np.zeros(ev.size, dtype=np.int64)
    pick[inv] = np.arange(inv.size)             # any localization of the event carries the event's truth
    return {"event_id": ev.astype(np.int64), "x": x, "y": y, "se": 1.0 / np.sqrt(sw), "frame": first,
            "n_locs": np.bincount(inv).astype(np.int64), "stack_id": sample.stack_id[pick],
            "molecule_id": sample.molecule_id[pick], "label_id": sample.label_id[pick]}


def export_smd_csv(sample: SyntheticSample, path: str, level: str = "localization", frame_base: int = 1,
                   truth: bool = False) -> int:
    """Write the sample as an SMD table for the MATLAB reference (BaGoL plan F3'): columns X, Y, X_SE, Y_SE,
    FrameNum, positions and errors in nm (6 decimals, i.e. 1e-6 nm). ``level`` "localization" writes every
    localization (X_SE = Y_SE = its sigma); "event" writes one row per switching event (:func:`event_table`).
    ``frame_base`` 1 writes MATLAB-style frame numbers starting at 1 [ad hoc: SMITE numbers frames from 1; to verify
    against the reference run]. ``truth`` appends StackID, MoleculeID and LabelID (synthetic truth; drop them before
    handing the file to a method). Returns the number of rows."""
    if level == "localization":
        x, y = sample.xy[:, 0], sample.xy[:, 1]
        se, fr = sample.sigma, sample.frame
        st, mo, la = sample.stack_id, sample.molecule_id, sample.label_id
    elif level == "event":
        t = event_table(sample)
        x, y, se, fr = t["x"], t["y"], t["se"], t["frame"]
        st, mo, la = t["stack_id"], t["molecule_id"], t["label_id"]
    else:
        raise ValueError(f"level must be 'localization' or 'event', got {level!r}")
    cols = list(SMD_COLUMNS) + (["StackID", "MoleculeID", "LabelID"] if truth else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(x.size):
            row = [f"{x[i]:.6f}", f"{y[i]:.6f}", f"{se[i]:.6f}", f"{se[i]:.6f}", str(int(fr[i]) + int(frame_base))]
            if truth:
                row += [str(int(st[i])), str(int(mo[i])), str(int(la[i]))]
            w.writerow(row)
    return int(x.size)


def write_stack_truth_csv(sample: SyntheticSample, path: str) -> int:
    """One row per stack: StackID, X, Y (centroid of its dyes, nm), MoleculeID (<= -2 non-specific), EpitopeID,
    NDyes, NLocs. The truth file that goes next to an SMD export."""
    n_d = np.bincount(sample.label_stack, minlength=sample.stack_xy.shape[0])
    n_l = sample.stack_n_locs
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["StackID", "X", "Y", "MoleculeID", "EpitopeID", "NDyes", "NLocs"])
        for j in range(sample.stack_xy.shape[0]):
            w.writerow([j, f"{sample.stack_xy[j, 0]:.6f}", f"{sample.stack_xy[j, 1]:.6f}",
                        int(sample.stack_molecule[j]), int(sample.stack_epitope[j]), int(n_d[j]), int(n_l[j])])
    return int(sample.stack_xy.shape[0])


def emitter_scenario(name: str, base: Optional[str] = None, model: Optional[EmitterModel] = None,
                     **scenario_kw: Any) -> Scenario:
    """A scenario with antibody stacks: a copy of preset ``base`` (or a default Scenario) with ``emitters`` set to
    ``model`` (default EmitterModel(): ELE 0.65, 1-3 secondaries, Poisson DOL 1.5, AF647-MEA cycles)."""
    sc = copy.deepcopy(PRESETS[base]) if base else Scenario(name=name)
    sc.name = name
    for k, v in scenario_kw.items():
        setattr(sc, k, v)
    sc.emitters = model if model is not None else EmitterModel()
    sc.origin = f"BaGoL plan F2: {('preset ' + base) if base else 'default scenario'} with antibody stacks"
    return sc


EMITTER_PRESETS: Dict[str, Scenario] = {
    # Not part of PRESETS: --all-scenarios and the D11 verdicts keep running on the legacy presets only.
    "sparse_emitters": emitter_scenario("sparse_emitters", base="sparse_lab2"),
    "sparse_emitters_bme": emitter_scenario("sparse_emitters_bme", base="sparse_lab2",
                                            model=EmitterModel(cycles_mean=26.0)),   # [literature] Dempsey 2011
                                                                                     # Table 1: AF647 in bME, 26
    "sparse_emitters_af568": emitter_scenario("sparse_emitters_af568", base="sparse_lab2",
                                              model=EmitterModel(cycles_mean=7.0)),  # [literature] Dempsey 2011
                                                                                     # Table 1: AF568 in MEA, 7
    "sparse_emitters_dol1": emitter_scenario("sparse_emitters_dol1", base="sparse_lab2",
                                             model=EmitterModel(n_secondary=1, dol=1, dol_dist="fixed")),
}


# --------------------------------------------------------------------------- runner
METRIC_KEYS =("exact", "merge", "split", "missed", "false", "ari", "iou_mean", "frac_locs_clustered", "n_clusters",
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
    ap.add_argument("--export-smd", action="store_true",
                    help="write each replicate of each --scenario (legacy or emitter preset) as an SMD CSV (X, Y, "
                         "X_SE, Y_SE, FrameNum in nm) plus a per-stack truth CSV into --out, and exit")
    ap.add_argument("--smd-level", choices=("localization", "event"), default="localization",
                    help="SMD rows: every localization, or one per switching event (perfect frame connection)")
    a = ap.parse_args(argv)
    if a.list:
        for k, sc in PRESETS.items():
            print(f"{k:22s} geometry={sc.geometry} spacing={sc.spacing_nm:g} p_label={sc.p_label:g} c_label={sc.c_label:g} "
                  f"sigma={sc.sigma_loc:g} bg={sc.bg_labels_per_um:g}/um,{sc.bg_locs_per_um2:g}/um2  [{sc.origin}]")
        print("emitter presets (antibody stacks; for --export-smd, not in --all-scenarios):")
        for k, sc in EMITTER_PRESETS.items():
            em = sc.emitters
            assert em is not None
            print(f"{k:22s} spacing={sc.spacing_nm:g} sigma={sc.sigma_loc:g} ele={em.ele:g} n2={em.n_secondary} "
                  f"dol={em.dol:g} ({em.dol_dist}) cycles={em.cycles_mean:g} ({em.cycles_dist})  [{sc.origin}]")
        return 0
    names = list(PRESETS) if a.all_scenarios else list(a.scenario)
    if not names:
        ap.error("give --scenario NAME (repeatable) or --all-scenarios")
    if a.export_smd:
        pool = {**PRESETS, **EMITTER_PRESETS}
        unknown = [n for n in names if n not in pool]
        if unknown:
            ap.error(f"unknown scenario(s) {unknown}; see --list")
        os.makedirs(a.out, exist_ok=True)
        for n in names:
            for rep in range(int(a.replicates)):
                s = generate(pool[n], np.random.default_rng([int(a.seed), scenario_seed(n), rep]))
                base = os.path.join(a.out, f"{n}_seed{a.seed}_rep{rep}")
                n_rows = export_smd_csv(s, base + "_smd.csv", level=a.smd_level)
                write_stack_truth_csv(s, base + "_stacks.csv")
                print(f"{base}_smd.csv: {n_rows} rows ({a.smd_level}); truth: {base}_stacks.csv")
        return 0
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
