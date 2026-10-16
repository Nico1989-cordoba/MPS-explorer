# -*- coding: utf-8 -*-
"""
Synthetic whole-field localization tables of known precision, for the NeNA
validation (docs/nena.md). Synthetic data only: no measured file is read.

Molecules sit uniformly over a square field, or in clusters of nearby
molecules (Gaussian spread around a cluster centre). Each molecule switches on
1 + Poisson(n_act - 1) times, each event lasting Geometric(p_on) frames, and
every localization is the molecule's position plus N(0, sigma) per axis.
Optional single-frame background localizations, uniform over the field.
Optional linking: one localization per event at the mean position, frame =
the event's first frame (Picasso Link with no dark frames allowed).

``next_frame_pairs`` applies NeNA's rule (nearest neighbour in the next frame,
closer than max_distance_nm) and says, for each pair, whether the neighbour is
the same molecule: that is what lets the validation count contamination and
compute the oracle (the width of the same-molecule pairs the rule picked).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

FIELD_NM = 20000.0
FRAMES = 20000
N_ACT = 4.0      # mean switching events per molecule
P_ON = 0.4       # per-frame chance an event ends: mean 2.5 frames on
PAIRS_PER_MOLECULE = N_ACT * (1.0 - P_ON) / P_ON   # 6 consecutive-frame pairs


def simulate(sigma: float, n_mol: int, *, frames: int = FRAMES, field_nm: float = FIELD_NM, n_act: float = N_ACT,
             p_on: float = P_ON, cluster_size: int = 1, cluster_sd_nm: float = 20.0, bg_per_frame: float = 0.0,
             linked: bool = False, seed: int = 0) -> Tuple[NDArray[np.int64], NDArray[np.float64],
                                                           NDArray[np.float64], NDArray[np.int64]]:
    """frame, x_nm, y_nm and the molecule id of every localization (-1: background)."""
    rng = np.random.default_rng(seed)
    n_clusters = max(1, int(np.ceil(n_mol / cluster_size)))
    ccx = rng.uniform(0, field_nm, n_clusters)
    ccy = rng.uniform(0, field_nm, n_clusters)
    cid = np.repeat(np.arange(n_clusters), cluster_size)[:n_mol]
    if cluster_size > 1:
        mx = ccx[cid] + rng.normal(0, cluster_sd_nm, n_mol)
        my = ccy[cid] + rng.normal(0, cluster_sd_nm, n_mol)
    else:
        mx, my = ccx[cid], ccy[cid]
    n_ev_mol = 1 + rng.poisson(n_act - 1.0, n_mol)
    ev_mol = np.repeat(np.arange(n_mol), n_ev_mol)
    ev_len = rng.geometric(p_on, ev_mol.size)
    ev_start = rng.integers(0, max(1, frames - 30), ev_mol.size)
    loc_ev = np.repeat(np.arange(ev_mol.size), ev_len)
    offs = np.arange(loc_ev.size) - np.repeat(np.cumsum(ev_len) - ev_len, ev_len)
    frame = ev_start[loc_ev] + offs
    mol = ev_mol[loc_ev]
    x = mx[mol] + rng.normal(0, sigma, mol.size)
    y = my[mol] + rng.normal(0, sigma, mol.size)
    if linked:
        cnt = np.bincount(loc_ev)
        x = np.bincount(loc_ev, weights=x) / cnt
        y = np.bincount(loc_ev, weights=y) / cnt
        frame = ev_start.copy()
        mol = ev_mol.copy()
    if bg_per_frame > 0:
        nb = int(rng.poisson(bg_per_frame * frames))
        frame = np.concatenate([frame, rng.integers(0, frames, nb)])
        x = np.concatenate([x, rng.uniform(0, field_nm, nb)])
        y = np.concatenate([y, rng.uniform(0, field_nm, nb)])
        mol = np.concatenate([mol, -np.ones(nb, dtype=int)])
    order = np.argsort(frame, kind="stable")
    return (frame[order].astype(np.int64), x[order], y[order], mol[order].astype(np.int64))


def next_frame_pairs(frame: NDArray[np.int64], x: NDArray[np.float64], y: NDArray[np.float64],
                     mol: NDArray[np.int64], max_distance_nm: float) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """NeNA's next-frame nearest-neighbour distances and whether each pair is the same molecule."""
    unique, start = np.unique(frame, return_index=True)
    end = np.append(start[1:], frame.size)
    bounds: Dict[int, Tuple[int, int]] = {int(f): (int(s), int(e)) for f, s, e in zip(unique, start, end)}
    ds, same = [], []
    for value, a, b in zip(unique, start, end):
        nxt = bounds.get(int(value) + 1)
        if nxt is None:
            continue
        c, e = nxt
        tree = cKDTree(np.column_stack([x[c:e], y[c:e]]))
        dist, idx = tree.query(np.column_stack([x[a:b], y[a:b]]), k=1)
        ds.append(np.asarray(dist, dtype=float))
        same.append((mol[a:b] == mol[c:e][idx]) & (mol[a:b] >= 0))
    if not ds:
        return np.array([]), np.array([], dtype=bool)
    d = np.concatenate(ds)
    s = np.concatenate(same)
    keep = d < max_distance_nm
    return d[keep], s[keep]


def background_for(contamination: float, n_mol: int, *, max_distance_nm: float = 120.0, frames: int = FRAMES,
                   field_nm: float = FIELD_NM) -> float:
    """Background localizations per frame that make about ``contamination`` of the next-frame pairs not the same
    molecule (closed-form approximation; the validation reports the share actually reached)."""
    if contamination <= 0:
        return 0.0
    n_same = PAIRS_PER_MOLECULE * n_mol
    n_events = N_ACT * n_mol
    k = np.pi * max_distance_nm ** 2 / field_nm ** 2
    want = contamination / (1.0 - contamination) * n_same
    lo, hi = 0.0, 1e4
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        got = (n_events + mid * frames) * (1.0 - np.exp(-k * mid))
        lo, hi = (mid, hi) if got < want else (lo, mid)
    return 0.5 * (lo + hi)


def rayleigh_rms_scale(d: NDArray[np.float64]) -> float:
    """sqrt(<d^2> / 4): the scale s of a Rayleigh law of distances, from its second moment."""
    return float(np.sqrt(np.mean(np.asarray(d) ** 2) / 4.0)) if np.size(d) else float("nan")
