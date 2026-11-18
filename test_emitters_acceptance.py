"""
Acceptance criteria of the Emitters layer ("Emitters (BaGoL-like reimplementation)"), as fast synthetic regression
tests (decision of the user 2026-10-17, F4 (b); the criteria and their origin are in docs/emitters.md,
'Acceptance criteria'). Each test is a small version of the F4 measurement (a few seconds, one process); the
criteria themselves were set on larger fields. Synthetic data only.

C1a: Poisson regime (the paper's), emitters >= 3 sigma apart, lambda 14: Jaccard (radius 3 x the mean returned
     precision, Hungarian matching; Fazel et al. 2022) >= 0.80.
C1b + C2: exponential counts of mean 14 (dSTORM), xi fixed at the truth, Dirichlet allocation: |emitters / truth - 1|
     <= 10 %, bright emitters (>= 2 x mean) split into >= 2 non-singleton pieces <= 20 %, recall >= 0.85 for 8-14
     localizations and >= 0.95 for >= 15.
C3:  learned xi on isolated emitters: mean within +-25 % of the truth and shape k in [0.5, 2] x the true k (1).
C6:  precisions over-reported by 20 % (model sigma 1.2 x the real one): |emitters / truth - 1| <= 25 %. The other
     side (under-reported by 20 %) does NOT meet it (+28 to +63 % on these fields, F5b): the layer's check T4
     (NeNA / precision > 1.2) asks for se_adjust_nm instead; tested in test_emitters_prior.py.
C4 (time) and C5 (against the MATLAB code) are not tested here (docs/emitters.md).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from tools.cluster_emitters import XiPrior, fit_emitters

SIGMA = 12.0
MEAN = 14.0


def _field(seed: int, n_em: int, side: float, counts: str, min_sep: float = 0.0):
    rng = np.random.default_rng(seed)
    pos: list = []
    while len(pos) < n_em:
        p = rng.uniform(0, side, 2)
        if min_sep <= 0 or all(np.hypot(*(p - q)) >= min_sep for q in pos):
            pos.append(p)
    pos_a = np.array(pos) + 1000.0
    m = (rng.poisson(MEAN, n_em) if counts == "poisson"
         else np.maximum(1, np.round(rng.exponential(MEAN, n_em))).astype(int))
    keep = m > 0
    pos_a, m = pos_a[keep], m[keep]
    xy = np.vstack([pos_a[k] + SIGMA * rng.standard_normal((m[k], 2)) for k in range(pos_a.shape[0])])
    return xy, np.repeat(np.arange(pos_a.shape[0]), m), pos_a, m


def _jaccard(res, pos):
    """Jaccard index and the matched truths: Hungarian matching with cost <= 3 x the mean returned precision
    (Fazel et al. 2022, Methods, 'Jaccard index'), estimates = emitters with localizations."""
    has = res.has_locs
    est = np.column_stack([res.x[has], res.y[has]])
    radius = 3.0 * float(np.mean(0.5 * (res.sigma_x[has] + res.sigma_y[has])))
    d = cdist(pos, est)
    cost = np.where(d <= radius, d, 1e9)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < 1e9
    matched = np.zeros(pos.shape[0], bool)
    matched[r[ok]] = True
    tp = int(ok.sum())
    return tp / (pos.shape[0] + est.shape[0] - tp), matched


def _pieces(res, tr, m):
    """Non-singleton estimates holding >= max(2, 20 %) of a true emitter's localizations (review of F4)."""
    out = np.zeros(m.size, int)
    for e in np.flatnonzero(res.is_nonsingleton):
        ids = tr[res.labels == e]
        if ids.size:
            v, c = np.unique(ids, return_counts=True)
            ok = c >= np.maximum(2, 0.2 * m[v])
            out[v[ok]] += 1
    return out


def test_c1a_poisson_regime_jaccard():
    jac = []
    for seed in (1, 2):
        xy, _tr, pos, _m = _field(seed, 40, 900.0, "poisson", min_sep=3 * SIGMA)
        res = fit_emitters(xy, SIGMA, XiPrior.poisson(MEAN), processes=1, seed=0, n_burnin=500, n_trials=500)
        jac.append(_jaccard(res, pos)[0])
    assert min(jac) >= 0.80, jac            # F5b: 0.91-0.95 on seeds 1-3


def test_c1b_c2_dstorm_regime_counting_splitting_and_recall():
    excess, split, rec8, rec15 = [], [], [], []
    for seed in (1, 2, 3):
        xy, tr, pos, m = _field(seed, 100, 1000.0, "exp")
        res = fit_emitters(xy, SIGMA, XiPrior.gamma(1.0, MEAN), processes=1, seed=0, n_burnin=1000, n_trials=1500)
        assert res.params["allocation_used"] == "dirichlet"
        excess.append(res.n_emitters_with_locs / pos.shape[0] - 1.0)
        bright = m >= 2 * MEAN
        split.extend((_pieces(res, tr, m)[bright] >= 2).tolist())
        _j, matched = _jaccard(res, pos)
        rec8.extend(matched[(m >= 8) & (m <= 14)].tolist())
        rec15.extend(matched[m >= 15].tolist())
    # F5b, seeds 1-3: excess +3 / -11 / +1 % (mean -2 %); split 0-5 %; recall 8-14: 0.95-1.0, >= 15: 0.97-1.0
    assert abs(float(np.mean(excess))) <= 0.10, excess
    assert float(np.mean(split)) <= 0.20, np.mean(split)
    assert float(np.mean(rec8)) >= 0.85 and float(np.mean(rec15)) >= 0.95, (np.mean(rec8), np.mean(rec15))


def test_c3_learned_xi_on_isolated_emitters():
    rng = np.random.default_rng(1)
    pos = np.array([[250.0 * (j % 8), 250.0 * (j // 8)] for j in range(64)]) + 1000.0
    m = np.maximum(1, np.round(rng.exponential(MEAN, 64))).astype(int)
    xy = np.vstack([pos[k] + SIGMA * rng.standard_normal((m[k], 2)) for k in range(64)])
    res = fit_emitters(xy, SIGMA, XiPrior.gamma(1.0, 7.0), hierarchical=True, processes=1, seed=0,
                       n_burnin=1000, n_trials=1000)
    assert res.params["xi_counts"] == "nonsingleton"
    k, theta = res.xi_used
    assert k * theta == pytest.approx(float(m.mean()), rel=0.25), (k * theta, m.mean())   # F5b: 13.8 vs 15.2
    assert 0.5 <= k <= 2.0, k                                                            # F5b: 1.08


def test_c6_overreported_precision_keeps_the_count():
    xy, _tr, pos, _m = _field(3, 100, 1000.0, "exp")
    res = fit_emitters(xy, 1.2 * SIGMA, XiPrior.gamma(1.0, MEAN), processes=1, seed=0, n_burnin=1000,
                       n_trials=1500)
    assert abs(res.n_emitters_with_locs / pos.shape[0] - 1.0) <= 0.25      # F5b: -13 / -23 / -14 % on seeds 1-3
