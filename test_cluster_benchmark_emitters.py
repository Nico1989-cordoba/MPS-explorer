# -*- coding: utf-8 -*-
"""
Tests of the emitter-level truth of the benchmark generator (BaGoL plan, phase F2): antibody stacks
(``EmitterModel``), emitter metrics (Hungarian matching at 3 sigma, as Fazel et al. 2022), the SMD export, and the
bit-for-bit reproduction of the generator of commit 26a1cd4 when the legacy labelling is used.

Synthetic data only (rule R8). Run: ``py -3 -m pytest test_cluster_benchmark_emitters.py -q``.

The block between the FROZEN markers is a verbatim copy of ``tools/cluster_benchmark.py`` at 26a1cd4 (constants,
geometry helpers and ``generate``; only the function name and the ``cb.`` prefix changed). Do not edit it: it is the
reference the legacy path must keep matching.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from tools import cluster_benchmark as cb

# ---------------------------------------------------------------- FROZEN: tools/cluster_benchmark.py at 26a1cd4
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
def generate_26a1cd4(sc: cb.Scenario, rng: np.random.Generator, L_nm: Optional[float] = None) -> cb.SyntheticSample:
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
    return cb.SyntheticSample(xy=xy[order], frame=frame[order].astype(np.int64), molecule_id=mol[order].astype(np.int64),
                           label_id=loc_label[order].astype(np.int64), event_id=event[order].astype(np.int64),
                           sigma=sig[order], mol_xy=mol_xy, mol_n_labels=mol_n_labels, arc_length_nm=arc_nm,
                           params=params)


def _background_positions(rng: np.random.Generator, con: Optional[_Contour], arc_nm: float, window_um: float,
                          n: int) -> NDArray[np.float64]:
    if con is not None:
        pb, nb = con.at(rng.uniform(0, arc_nm, n))
        return pb + rng.uniform(-BG_BAND_NM, BG_BAND_NM, n)[:, None] * nb
    return rng.uniform(0.0, 1000.0 * window_um, (n, 2))


# ---------------------------------------------------------------- END FROZEN


# =========================================================================== tests
def _frozen(sc, rng):
    """The generator of 26a1cd4 (frozen copy above)."""
    return generate_26a1cd4(sc, rng)


@pytest.mark.parametrize("name", sorted(cb.PRESETS))
def test_legacy_generator_is_bit_for_bit_26a1cd4(name):
    """Scenario.emitters = None, and the explicit EmitterModel.legacy, reproduce the generator of 26a1cd4 exactly."""
    fields = ("xy", "frame", "molecule_id", "label_id", "event_id", "sigma", "mol_xy", "mol_n_labels")
    sc = cb.PRESETS[name]
    sc_explicit = copy.deepcopy(sc)
    sc_explicit.emitters = cb.EmitterModel.legacy(sc)
    for seed in (0, 5):
        for rep in (0, 1):
            key = [seed, cb.scenario_seed(name), rep]
            old = _frozen(sc, np.random.default_rng(key))
            new = cb.generate(sc, np.random.default_rng(key))
            exp = cb.generate(sc_explicit, np.random.default_rng(key))
            for f in fields:
                a, b, c = getattr(old, f), getattr(new, f), getattr(exp, f)
                assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (seed, rep, f)
                assert a.shape == c.shape and a.tobytes() == c.tobytes(), ("explicit legacy model", seed, rep, f)
            assert old.arc_length_nm == new.arc_length_nm
            assert old.params == new.params                       # no new key in legacy configs
            # in the legacy model every label is its own stack
            assert np.array_equal(new.stack_id, new.label_id)
            assert np.array_equal(new.label_stack, np.arange(new.label_xy.shape[0]))
            assert np.array_equal(new.stack_xy, new.label_xy)


def test_legacy_as_dict_has_no_emitter_key_and_emitter_scenario_serialises():
    assert "emitters" not in cb.PRESETS["sparse_lab2"].as_dict()
    d = cb.EMITTER_PRESETS["sparse_emitters"].as_dict()
    assert d["emitters"]["ele"] == pytest.approx(cb.ELE_DEFAULT)
    json.dumps(d)                                                  # JSON-safe (tuples became lists)
    assert not set(cb.EMITTER_PRESETS) & set(cb.PRESETS)          # --all-scenarios and D11 stay on the legacy set


def _emitter_sample(seed=2, **model_kw):
    model = cb.EmitterModel(**model_kw)
    sc = cb.emitter_scenario("t_emitters", spacing_nm=200.0, L_um_range=(10.0, 10.0), model=model)
    return cb.generate(sc, np.random.default_rng(seed))


def test_emitter_truth_is_consistent():
    s = _emitter_sample()
    n = s.xy.shape[0]
    assert s.stack_id.shape == (n,)
    on_stack = s.stack_id >= 0
    assert np.array_equal(on_stack, s.label_id >= 0)              # only isolated background locs have no stack
    assert np.array_equal(s.label_stack[s.label_id[on_stack]], s.stack_id[on_stack])
    # molecule of a localization = molecule of its stack (specific >= 0, non-specific <= -2)
    assert np.array_equal(s.stack_molecule[s.stack_id[on_stack]], s.molecule_id[on_stack])
    K = s.mol_xy.shape[0]
    spec = s.stack_molecule >= 0
    assert np.array_equal(s.stack_epitope[spec] % K, s.stack_molecule[spec])
    assert np.all(s.stack_epitope[~spec] == -1)
    dye_mol = s.stack_molecule[s.label_stack]
    assert np.array_equal(s.mol_n_labels, np.bincount(dye_mol[dye_mol >= 0], minlength=K))
    assert s.params["n_stacks"] == s.stack_xy.shape[0]
    assert s.params["n_dyes"] == s.label_xy.shape[0]
    assert s.params["n_stacks_visible"] == int((s.stack_n_locs > 0).sum())
    # stack centre = centroid of its dyes
    j = int(np.flatnonzero(np.bincount(s.label_stack) >= 2)[0])
    assert np.allclose(s.stack_xy[j], s.label_xy[s.label_stack == j].mean(axis=0))
    # dyes stay within a few spreads of their stack; stacks within a few linkage s.d. of their molecule
    assert np.max(np.abs(s.label_xy - s.stack_xy[s.label_stack])) < 6 * cb.DYE_SPREAD_NM
    d_mol = np.linalg.norm(s.stack_xy[spec] - s.mol_xy[s.stack_molecule[spec]], axis=1)
    assert np.max(d_mol) < 6 * math.sqrt(2 * cb.SIGMA_LAYER_NM ** 2 + cb.DYE_SPREAD_NM ** 2) + cb.EPITOPE_SEP_NM[1]


def test_fixed_hierarchy_counts():
    """ELE 1, 2 secondaries, 3 dyes, never a dark dye: every count is exact."""
    s = _emitter_sample(ele=1.0, n_secondary=2, dol=3, dol_dist="fixed", cycles_dist="poisson", cycles_mean=40.0)
    K = s.mol_xy.shape[0]
    n_spec = int((s.stack_molecule >= 0).sum())
    assert n_spec == 2 * 2 * K
    assert np.all(np.bincount(s.label_stack) == 3)
    assert np.all(s.mol_n_labels == 12)
    assert np.all(np.bincount(s.stack_epitope[s.stack_molecule >= 0], minlength=2 * K) == 2)


def test_sampling_statistics_match_the_model():
    """ELE, secondaries per primary and dyes per stack follow their parameters (window, many molecules)."""
    model = cb.EmitterModel(ele=0.65, dol=1.5, cycles_dist="gamma", cycles_mean=14.0)
    sc = cb.emitter_scenario("t_window", base="window_sparse_lab2", model=model, window_um=6.0,
                             bg_labels_per_um=0.0, bg_locs_per_um2=0.0)
    s = cb.generate(sc, np.random.default_rng(11))
    K = s.mol_xy.shape[0]
    n_epi_lab = np.unique(s.stack_epitope).size
    p = n_epi_lab / (2 * K)
    assert abs(p - 0.65) < 4 * math.sqrt(0.65 * 0.35 / (2 * K))
    n2 = np.bincount(s.stack_epitope)[np.unique(s.stack_epitope)]
    pmf = np.asarray(cb.N_SECONDARY_PMF) / sum(cb.N_SECONDARY_PMF)
    m2 = float(np.dot(np.arange(1, 4), pmf))
    v2 = float(np.dot((np.arange(1, 4) - m2) ** 2, pmf))
    assert n2.min() >= 1 and n2.max() <= 3
    assert abs(n2.mean() - m2) < 4 * math.sqrt(v2 / n2.size)
    dol = np.bincount(s.label_stack, minlength=s.stack_xy.shape[0])
    assert abs(dol.mean() - 1.5) < 4 * math.sqrt(1.5 / dol.size)


@pytest.mark.parametrize("dist,mean,shape", [("geometric", 14.0, 1.0), ("gamma", 14.0, 1.0), ("gamma", 7.0, 3.0),
                                             ("poisson", 26.0, 1.0)])
def test_cycle_distributions(dist, mean, shape):
    em = cb.EmitterModel(cycles_dist=dist, cycles_mean=mean, cycles_shape=shape)
    n = cb._draw_cycles(np.random.default_rng(4), em, 200_000)
    var = {"geometric": mean * (mean - 1), "gamma": mean + mean ** 2 / shape, "poisson": mean}[dist]
    assert abs(n.mean() - mean) < 5 * math.sqrt(var / n.size)
    if dist == "geometric":
        assert n.min() >= 1
    if dist == "gamma" and shape == 1.0:
        assert abs(np.mean(n == 0) - 1.0 / (1.0 + mean)) < 0.005     # exponential mixture: P(0) = 1 / (1 + mean)


def test_emitter_model_validation():
    for kw in ({"ele": 1.2}, {"n_secondary": 0}, {"n_secondary": (0.5, -0.1)}, {"dol_dist": "binomial"},
               {"dol": 1.5, "dol_dist": "fixed"}, {"cycles_dist": "lognormal"}, {"cycles_mean": 0.5},
               {"sigma_primary_nm": -1.0}):
        with pytest.raises(ValueError):
            cb.EmitterModel(**kw)


def test_from_prior_duck_typed():
    gam = SimpleNamespace(shape="gamma", mean=28.0, k=1.0, theta=28.0, switching_cycles=14.0, origin="Dempsey MEA")
    em = cb.EmitterModel.from_prior(gam)
    assert em.cycles_dist == "gamma" and em.cycles_mean == 14.0 and em.cycles_shape == 1.0
    assert "Dempsey MEA" in em.origin
    em2 = cb.EmitterModel.from_prior(SimpleNamespace(shape="gamma", mean=28.0, k=2.0), dol=2.0)
    assert em2.cycles_mean == pytest.approx(28.0 / np.mean(cb.ON_FRAMES)) and em2.cycles_shape == 2.0
    assert em2.dol == 2.0
    em3 = cb.EmitterModel.from_prior(SimpleNamespace(shape="poisson", mean=30.0, loc_per_cycle=3.0))
    assert em3.cycles_dist == "poisson" and em3.cycles_mean == pytest.approx(10.0)


@pytest.mark.parametrize("name", sorted(cb.EMITTER_PRESETS))
def test_oracle_emitters_score_jac_one(name):
    s = cb.generate(cb.EMITTER_PRESETS[name], np.random.default_rng([0, cb.scenario_seed(name), 0]))
    m = cb.emitter_metrics(cb.oracle_emitters(s), s, details=True)
    assert m["jac"] == 1.0 and m["rmse_nm"] == 0.0 and m["fp"] == 0 and m["fn"] == 0
    assert m["n_true_absorbed"] == 0 and m["n_est_split"] == 0
    assert m["frac_k_exact"] == 1.0 and m["k_bias_mean"] == 0.0
    assert np.array_equal(np.sort(m["pairs_true_stack"]), cb.emitter_truth(s)[1])
    # only specific stacks: background stacks become false positives of a full oracle
    m2 = cb.emitter_metrics(cb.oracle_emitters(s), s, include_background=False)
    n_bg = int(((s.stack_molecule < 0) & (s.stack_n_locs > 0)).sum())
    assert m2["fp"] == n_bg


def _toy(stack_xy, stack_mol, sigma_loc=10.0):
    """One localization at every stack (so every stack is visible)."""
    stack_xy = np.asarray(stack_xy, float).reshape(-1, 2)
    n = stack_xy.shape[0]
    mol = np.asarray(stack_mol, np.int64)
    K = int(mol.max()) + 1 if np.any(mol >= 0) else 0
    return cb.SyntheticSample(
        xy=stack_xy.copy(), frame=np.arange(n, dtype=np.int64), molecule_id=mol.copy(),
        label_id=np.arange(n, dtype=np.int64), event_id=np.arange(n, dtype=np.int64), sigma=np.full(n, sigma_loc),
        mol_xy=np.zeros((K, 2)), mol_n_labels=np.bincount(mol[mol >= 0], minlength=K).astype(np.int64),
        arc_length_nm=1000.0, params={"sigma_loc": sigma_loc}, stack_id=np.arange(n, dtype=np.int64),
        label_xy=stack_xy.copy(), label_stack=np.arange(n, dtype=np.int64), stack_xy=stack_xy.copy(),
        stack_molecule=mol.copy(), stack_epitope=np.where(mol >= 0, mol, -1))


def test_emitter_metrics_known_cases():
    truth = [[0, 0], [1000, 0], [2000, 0], [3000, 0]]
    s = _toy(truth, [0, 0, 1, 2])
    # a constant shift of 5 nm: every pair found, RMSE 5
    m = cb.emitter_metrics(np.asarray(truth, float) + [3.0, 4.0], s)
    assert m["jac"] == 1.0 and m["rmse_nm"] == pytest.approx(5.0) and m["mean_dist_nm"] == pytest.approx(5.0)
    assert m["match_radius_nm"] == pytest.approx(30.0)
    # two found, one far away (false positive), two missing
    m = cb.emitter_metrics([[0, 0], [1000, 0], [5000, 5000]], s)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 2) and m["jac"] == pytest.approx(2 / 5)
    assert m["frac_k_exact"] == pytest.approx(1 / 3) and m["frac_k_under"] == pytest.approx(2 / 3)
    # a duplicate next to a found stack is under-grouping (split)
    m = cb.emitter_metrics(np.vstack([truth, [[2010, 0]]]), s)
    assert m["n_est_split"] == 1 and m["fp"] == 1 and m["frac_k_over"] == pytest.approx(1 / 3)
    # the per-estimate sigma sets the radius: 5 nm away fails at sigma 1 (radius 3) and passes at sigma 2
    assert cb.emitter_metrics([[5, 0]], s, est_sigma=1.0)["tp"] == 0
    assert cb.emitter_metrics([[5, 0]], s, est_sigma=2.0)["tp"] == 1
    # nothing found
    m = cb.emitter_metrics(np.empty((0, 2)), s)
    assert m["jac"] == 0.0 and m["fn"] == 4 and np.isnan(m["rmse_nm"])


def test_two_close_stacks_merged_into_one_emitter_are_over_grouping():
    s = _toy([[0, 0], [20, 0], [1000, 0]], [0, 0, 1])          # two stacks of molecule 0, 20 nm apart
    m = cb.emitter_metrics([[10, 0], [1000, 0]], s)               # one emitter between them
    assert m["tp"] == 2 and m["fn"] == 1 and m["n_true_absorbed"] == 1
    assert m["frac_k_under"] == pytest.approx(0.5) and m["k_true_mean"] == 1.5 and m["k_est_mean"] == 1.0


def test_hungarian_maximises_the_number_of_pairs():
    """Greedy nearest matching pairs A with t0 and leaves B alone; the Hungarian pairing finds both."""
    s = _toy([[0, 0], [25, 0]], [0, 1])
    m = cb.emitter_metrics([[12, 0], [-20, 0]], s)               # radius 30: A near both, B only near t0
    assert m["tp"] == 2 and m["jac"] == 1.0


def test_export_smd_csv_round_trip(tmp_path):
    s = _emitter_sample(seed=5)
    p = str(tmp_path / "smd.csv")
    assert cb.export_smd_csv(s, p, truth=True) == s.xy.shape[0]
    with open(p, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0])[:5] == list(cb.SMD_COLUMNS)
    x = np.array([float(r["X"]) for r in rows])
    se = np.array([float(r["X_SE"]) for r in rows])
    fr = np.array([int(r["FrameNum"]) for r in rows])
    assert np.max(np.abs(x - s.xy[:, 0])) <= 5e-7 and np.max(np.abs(se - s.sigma)) <= 5e-7
    assert np.array_equal(fr, s.frame + 1)
    assert np.array_equal(np.array([int(r["StackID"]) for r in rows]), s.stack_id)
    # event level: one row per switching event, inverse-variance mean, SE below the best localization
    pe = str(tmp_path / "smd_events.csv")
    t = cb.event_table(s)
    assert cb.export_smd_csv(s, pe, level="event") == np.unique(s.event_id).size
    e0 = int(np.flatnonzero(t["n_locs"] >= 2)[0])
    sel = s.event_id == t["event_id"][e0]
    w = 1.0 / s.sigma[sel] ** 2
    assert np.allclose([t["x"][e0], t["y"][e0]], (s.xy[sel] * w[:, None]).sum(0) / w.sum())
    assert t["se"][e0] < s.sigma[sel].min() and t["frame"][e0] == s.frame[sel].min()
    assert len(set(s.stack_id[sel])) == 1 and t["stack_id"][e0] == s.stack_id[sel][0]
    with open(pe, newline="", encoding="utf-8") as fh:
        assert list(csv.DictReader(fh))[0].keys() == set(cb.SMD_COLUMNS)
    pt = str(tmp_path / "stacks.csv")
    assert cb.write_stack_truth_csv(s, pt) == s.stack_xy.shape[0]
    with pytest.raises(ValueError):
        cb.export_smd_csv(s, p, level="cluster")


def test_cli_export_smd(tmp_path, capsys):
    out = str(tmp_path / "smd")
    assert cb.main(["--export-smd", "--scenario", "sparse_emitters", "--scenario", "sparse_lab1",
                    "--replicates", "1", "--out", out, "--smd-level", "event"]) == 0
    files = sorted(os.listdir(out))
    assert files == ["sparse_emitters_seed0_rep0_smd.csv", "sparse_emitters_seed0_rep0_stacks.csv",
                     "sparse_lab1_seed0_rep0_smd.csv", "sparse_lab1_seed0_rep0_stacks.csv"]
    assert cb.main(["--list"]) == 0
    assert "sparse_emitters_dol1" in capsys.readouterr().out
