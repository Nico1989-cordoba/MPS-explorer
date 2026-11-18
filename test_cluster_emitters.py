# -*- coding: utf-8 -*-
"""
Tests of ``tools.cluster_emitters`` (Emitters, BaGoL-like reimplementation), synthetic data only (R8):

- the birth/death acceptance ratio against the exact posterior of K in {1, 2}, computed by integrating the
  model over the support grid (the strongest check of the RJMCMC: a wrong Jacobian, proposal density or area
  term moves the chain away from it); the original code's death ratio (``legacy_death_ratio``) does not match;
- the hierarchical xi: the conjugate Poisson draw, the gamma MH update, and learning lambda from isolated
  emitters;
- the acceptance criteria of phase F3: a dimer at 10 nm (lambda 50, sigma 4 nm) gives K = 2 in > 90 % of the
  post-burn-in chain and positions within 2 nm (Fazel et al. 2022, Suppl. Fig. 1); an 8-mer of radius 10 nm with
  lambda 50 (precisions sigma_PSF / sqrt(I), I ~ Exp(1800), the paper's generator) gives a K mode of 8; an
  isolated emitter with lambda 5 (dSTORM regime) gives a K mode of 1;
- subregions: emitters on tile borders are neither lost nor duplicated; z is the weighted mean;
- drift: a linearly moving emitter is one emitter with its drift, several without;
- determinism with the seed, also across process counts; the registry entry.

Run: ``py -3 -m pytest test_cluster_emitters.py -q``.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tools import cluster_emitters as ce
from tools.cluster_emitters import ChainConfig, EmitterChain, XiPrior, fit_emitters, sample_xi

_LOG2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------------- helpers
def _emitters(pos: np.ndarray, lam: float, sigma: float, rng: np.random.Generator, offset: float = 1000.0):
    pts, truth = [], []
    for k, p in enumerate(pos):
        m = int(rng.poisson(lam))
        pts.append(p + sigma * rng.standard_normal((m, 2)))
        truth += [k] * m
    return np.vstack(pts) + offset, np.asarray(truth)


def _paper_emitters(pos: np.ndarray, lam: float, rng: np.random.Generator, offset: float = 1000.0):
    """Fazel et al. (2022) Methods: Poisson(lambda) events per emitter, I ~ Exp(mean 1800 photons),
    sigma = sigma_PSF / sqrt(I) with sigma_PSF = 120 nm ("PSF size of 120 nm")."""
    pts, sig = [], []
    for p in pos:
        m = int(rng.poisson(lam))
        s = 120.0 / np.sqrt(rng.exponential(1800.0, m))
        pts.append(p + s[:, None] * rng.standard_normal((m, 2)))
        sig.append(s)
    return np.vstack(pts) + offset, np.concatenate(sig)


def _exact_p_k2(ch: EmitterChain, xi: XiPrior) -> float:
    """P(K = 2 | X) for the model restricted to K in {1, 2}: integrate prod_k u_S(mu_k) L_K(mu) over the
    support cells (cell centres, cell area g^2) and weight by p(N | K)."""
    g = ch.g
    iy, ix = np.divmod(ch.scells, ch.gnx)
    cx = ch.gx0 + (ix + 0.5) * g
    cy = ch.gy0 + (iy + 0.5) * g
    lp = ch._logphi(cx, cy)                                   # (N, M)
    l1 = lp.sum(axis=0)
    m1 = l1.max()
    log_z1 = math.log(np.exp(l1 - m1).sum() * g * g / ch.area) + m1
    rowmax = lp.max(axis=1, keepdims=True)
    p = np.exp(lp - rowmax)
    l2 = np.array([np.log(0.5 * (p[:, a][:, None] + p)).sum(axis=0) for a in range(p.shape[1])])
    m2 = l2.max()
    log_z2 = math.log(np.exp(l2 - m2).sum() * g ** 4 / ch.area ** 2) + m2 + float(rowmax.sum())
    lw1 = xi.log_pN(ch.n, 1) + log_z1
    lw2 = xi.log_pN(ch.n, 2) + log_z2
    return 1.0 / (1.0 + math.exp(lw1 - lw2))


def _chain_p_k2(xy: np.ndarray, sigma: float, xi: XiPrior, legacy: bool, n_steps: int = 60000,
                matlab: bool = False) -> EmitterChain:
    if matlab:
        cfg = ChainConfig.matlab_compat(n_burnin=1000, n_trials=n_steps)
    else:
        cfg = ChainConfig(n_burnin=1000, n_trials=n_steps, legacy_death_ratio=legacy)
    n = xy.shape[0]
    ch = EmitterChain(xy[:, 0], xy[:, 1], np.full(n, sigma), np.full(n, sigma), 7, cfg, xi)
    ch.kmax = 2                                               # restrict to K in {1, 2} for the exact check
    ch.run(n_steps + 1000, xi)
    return ch


# ----------------------------------------------------------------------------- RJMCMC correctness
@pytest.mark.parametrize("seed,sep", [(0, 10.0), (1, 5.0)])
def test_birth_death_matches_exact_posterior_of_k(seed, sep):
    rng = np.random.default_rng(seed)
    s = 4.0
    xy = np.vstack([s * rng.standard_normal((3, 2)), np.array([sep, 0.0]) + s * rng.standard_normal((3, 2))])
    xi = XiPrior.poisson(3.0)
    ch = _chain_p_k2(xy, s, xi, legacy=False)
    exact = _exact_p_k2(ch, xi)
    got = float(np.mean(np.asarray(ch.k_rec) == 2))
    assert 0.2 < exact < 0.9                                  # a case where the ratio matters
    assert abs(got - exact) < 0.03, (got, exact)


@pytest.mark.parametrize("matlab", [False, True])
def test_original_death_rule_is_biased_towards_more_emitters(matlab):
    """The original code's Death rule omits Area * q(mu_j) (Supplementary Note 1, eq. 27 asks for the full
    inverse of the Birth ratio). On these data the rule alone (``legacy_death_ratio``) over-weights K = 2 by
    about +0.10; the full original code (``matlab_compat``: the rule plus the pruning of empty emitters, which
    pulls K down) by about +0.08 (independent review of F3, section 3.2)."""
    rng = np.random.default_rng(0)
    s = 4.0
    xy = np.vstack([s * rng.standard_normal((3, 2)), np.array([10.0, 0.0]) + s * rng.standard_normal((3, 2))])
    xi = XiPrior.poisson(3.0)
    ch = _chain_p_k2(xy, s, xi, legacy=True, matlab=matlab)
    assert ch.cfg.legacy_death_ratio and ch.cfg.prune_empty == matlab
    exact = _exact_p_k2(ch, xi)
    got = float(np.mean(np.asarray(ch.k_rec) == 2))
    assert got - exact > (0.04 if matlab else 0.05), (got, exact)


def test_matlab_compat_switches_both_deviations_and_the_fixed_overlap():
    rng = np.random.default_rng(9)
    xy = np.array([500.0, 500.0]) + 12.0 * rng.standard_normal((6, 2))
    r = fit_emitters(xy, 12.0, XiPrior.poisson(6.0), processes=1, seed=0, n_burnin=50, n_trials=50,
                     matlab_compat=True)
    assert r.params["legacy_death_ratio"] and r.params["prune_empty"] and r.params["matlab_compat"]
    assert r.params["overlap_used_nm"] == ce.OVERLAP_NM
    r = fit_emitters(xy, 12.0, XiPrior.poisson(6.0), processes=1, seed=0, n_burnin=50, n_trials=50)
    assert not (r.params["legacy_death_ratio"] or r.params["prune_empty"] or r.params["matlab_compat"])


def test_log_q_density_is_finite_far_from_sharp_localizations():
    """DNA-PAINT-like precision (1 nm): 60 nm away the kernel density underflows to 0 in double precision;
    its log must stay finite (it enters the birth and death ratios) and agree with the direct sum near the data."""
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0]]) + 500.0
    n = xy.shape[0]
    ch = EmitterChain(xy[:, 0], xy[:, 1], np.ones(n), np.ones(n), 0, ChainConfig(n_burnin=0, n_trials=1),
                      XiPrior.poisson(3.0))
    far = ch.log_q_density(560.0, 500.0)
    assert math.isfinite(far) and far < -1000.0
    direct = np.mean(np.exp(-math.log(2 * math.pi) - 0.5 * ((xy[:, 0] - 501.0) ** 2 + (xy[:, 1] - 500.5) ** 2)))
    assert ch.log_q_density(501.0, 500.5) == pytest.approx(math.log(direct), rel=1e-12)


def _exact_k_posterior(xy: np.ndarray, sigma: float, xi: XiPrior, ch: EmitterChain, alpha: float = 0.0):
    """Exact P(K | X) and P(m | X) (m = emitters with >= 1 localization) of the model for ANY K, by enumerating the
    partitions of the localizations (the independent review of F3, ``exact_enum.py``; no grid integration):

        P(K | X) ∝ p(N | K) K^-N sum_{partitions with m <= K blocks} K! / (K - m)! prod_B G(B),

    and with ``alpha`` > 0 (Dirichlet-multinomial allocation of concentration alpha, F4) K^-N is replaced by
    Gamma(K alpha) / Gamma(N + K alpha) and every block B gets the factor Gamma(|B| + alpha) / Gamma(alpha) (the
    K - m empty components contribute 1).

    G(B) = (1/A) int_S prod_{i in B} phi(x_i; mu, sigma) dmu: the Gaussian integral in closed form times the exact
    mass of the block's posterior inside S (cell by cell, with the normal CDF). The support S and its area A are
    the model's (the chain's) definition."""
    from scipy.special import gammaln, ndtr
    n = xy.shape[0]
    mask = ch.smask.astype(float)
    ex = ch.gx0 + np.arange(ch.gnx + 1) * ch.g
    ey = ch.gy0 + np.arange(ch.gny + 1) * ch.g
    log_g = np.zeros(1 << n)
    for b in range(1, 1 << n):
        idx = [i for i in range(n) if b >> i & 1]
        k = len(idx)
        w = k / sigma ** 2
        lg = -math.log(ch.area)
        mass = []
        for v, e in ((xy[idx, 0], ex), (xy[idx, 1], ey)):
            m = float(v.mean())
            q = float(((v - m) ** 2).sum()) / sigma ** 2
            lg += -0.5 * (k - 1) * _LOG2PI - k * math.log(sigma) - 0.5 * math.log(w) - 0.5 * q
            mass.append(np.diff(ndtr((e - m) * math.sqrt(w))))
        log_g[b] = lg + math.log(max(float(mass[1] @ mask @ mass[0]), 1e-300))
    by_m: dict = {}
    lab = [0] * n

    def rec(i: int, m: int) -> None:                      # restricted growth strings = set partitions
        if i == n:
            masks = [0] * m
            for j, a in enumerate(lab):
                masks[a] |= 1 << j
            v = sum(log_g[mk] for mk in masks)
            if alpha > 0:
                v += sum(gammaln(bin(mk).count("1") + alpha) - gammaln(alpha) for mk in masks)
            by_m.setdefault(m, []).append(v)
            return
        for a in range(m + 1):
            lab[i] = a
            rec(i + 1, max(m, a + 1))

    rec(0, 0)
    log_s = {m: float(np.logaddexp.reduce(v)) for m, v in by_m.items()}
    kmax = max(n, ce.KMAX_FLOOR)
    lw = np.full(kmax + 1, -np.inf)
    lwm = np.full((n + 1, kmax + 1), -np.inf)
    for kk in range(1, kmax + 1):
        base = xi.log_pN(n, kk) + ((gammaln(kk * alpha) - gammaln(n + kk * alpha)) if alpha > 0
                                   else -n * math.log(kk))
        for m, ls in log_s.items():
            if m <= kk:
                lwm[m, kk] = base + gammaln(kk + 1) - gammaln(kk - m + 1) + ls
        lw[kk] = np.logaddexp.reduce(lwm[:, kk])
    pk = np.exp(lw - lw.max())
    pm = np.exp(np.logaddexp.reduce(lwm, axis=1) - lw.max())
    return pk / pk.sum(), pm / pm.sum()


def test_rjmcmc_matches_the_exact_posterior_for_any_k():
    """Case B of the review: 8 localizations in two groups 8 nm apart, sigma 4 nm, Poisson(4): the posterior
    spreads over K = 1..5 (E[K] = 2.18). The chain must match P(K = k), E[K] and the number of occupied emitters;
    the original code's death rule alone moves E[K] by +0.08. With K = 1 the position follows N(x_bar, s^2 / N)."""
    s, lam = 4.0, 4.0
    rng = np.random.default_rng(1001)
    xy = np.vstack([s * rng.standard_normal((4, 2)), np.array([8.0, 0.0]) + s * rng.standard_normal((4, 2))]) + 500.0
    n = xy.shape[0]
    xi = XiPrior.poisson(lam)
    ch = EmitterChain(xy[:, 0], xy[:, 1], np.full(n, s), np.full(n, s), 5, ChainConfig(n_burnin=2000,
                      n_trials=120000), xi)
    pk, pm = _exact_k_posterior(xy, s, xi, ch)
    ch.run(122000, xi)
    k = np.asarray(ch.k_rec)
    occ = np.array([np.unique(z).size for z in ch.z_rec])
    ek = float(np.sum(np.arange(pk.size) * pk))
    em = float(np.sum(np.arange(pm.size) * pm))
    assert 2.1 < ek < 2.3 and pk[3] > 0.1                   # K > 2 matters in this case
    assert abs(k.mean() - ek) < 0.02, (k.mean(), ek)
    assert abs(occ.mean() - em) < 0.02, (occ.mean(), em)
    for kk in range(1, 6):
        assert abs(np.mean(k == kk) - pk[kk]) < 0.02, (kk, np.mean(k == kk), pk[kk])
    one = np.flatnonzero(k == 1)
    mx = np.array([ch.mux_rec[i][0] for i in one])
    my = np.array([ch.muy_rec[i][0] for i in one])
    assert mx.mean() == pytest.approx(xy[:, 0].mean(), abs=0.3)
    assert my.mean() == pytest.approx(xy[:, 1].mean(), abs=0.3)
    assert mx.std() == pytest.approx(s / math.sqrt(n), rel=0.15)


def test_xi_log_pn_matches_scipy():
    from scipy import stats
    assert XiPrior.poisson(50.0).log_pN(97, 2) == pytest.approx(stats.poisson.logpmf(97, 100.0))
    assert XiPrior.gamma(1.0, 14.0).log_pN(30, 3) == pytest.approx(stats.gamma.logpdf(30, 3.0, scale=14.0))


# ----------------------------------------------------------------------------- hierarchical xi
def test_poisson_xi_update_is_the_conjugate_posterior():
    rng = np.random.default_rng(0)
    n, k = [40, 55, 61], [1, 1, 1]
    draws = np.array([sample_xi(XiPrior.poisson(10.0), n, k, rng).lam for _ in range(20000)])
    shape, scale = 1.0 + 156, 1.0 / (1.0 / 50.0 + 3)
    assert draws.mean() == pytest.approx(shape * scale, rel=0.01)
    assert draws.std() == pytest.approx(math.sqrt(shape) * scale, rel=0.03)


def test_gamma_xi_update_recovers_the_mean():
    rng = np.random.default_rng(1)
    n = np.maximum(1, np.round(rng.gamma(2.0, 10.0, 60))).astype(int)    # 60 emitters, mean 20
    k = np.ones_like(n)
    xi = XiPrior.gamma(1.0, 5.0)
    means = []
    for t in range(1500):
        xi = sample_xi(xi, n, k, rng)
        if t >= 500:
            means.append(xi.mean)
    assert np.mean(means) == pytest.approx(n.mean(), rel=0.15)


def test_hierarchical_learns_lambda_from_isolated_emitters():
    rng = np.random.default_rng(3)
    pos = np.array([[500.0 * (j % 4), 500.0 * (j // 4)] for j in range(12)])
    xy, _ = _emitters(pos, 20.0, 4.0, rng)
    r = fit_emitters(xy, 4.0, XiPrior.poisson(5.0), hierarchical=True, processes=1, seed=0,
                     n_burnin=1000, n_trials=1000)
    assert r.xi_kind == "poisson"
    assert r.xi_used[0] == pytest.approx(xy.shape[0] / 12.0, rel=0.25)
    assert r.n_emitters == 12


# ----------------------------------------------------------------------------- acceptance criteria F3
def test_dimer_10nm_lambda50():
    rng = np.random.default_rng(1)
    pos = np.array([[0.0, 0.0], [10.0, 0.0]])
    xy, truth = _emitters(pos, 50.0, 4.0, rng)
    r = fit_emitters(xy, 4.0, XiPrior.poisson(50.0), processes=1, seed=3)
    assert len(r.k_hist) == 1
    h = r.k_hist[0]
    assert h.get(2, 0) / sum(h.values()) > 0.90
    assert r.n_emitters == 2
    est = np.column_stack([r.x, r.y]) - 1000.0
    d = min(np.hypot(*(est - pos).T).max(), np.hypot(*(est[::-1] - pos).T).max())
    assert d < 2.0
    # labels: each localization goes to the emitter of its true source in >= 90 % of the cases
    order = 0 if np.hypot(*(est[0] - pos[0])) < np.hypot(*(est[1] - pos[0])) else 1
    mapped = np.where(r.labels == order, 0, 1)
    assert np.mean(mapped == truth) > 0.9


def test_eight_mer_radius_10nm_lambda50():
    rng = np.random.default_rng(200)
    ang = np.arange(8) * 2 * np.pi / 8
    pos = 10.0 * np.column_stack([np.cos(ang), np.sin(ang)])
    xy, sig = _paper_emitters(pos, 50.0, rng)
    r = fit_emitters(xy, sig, XiPrior.poisson(50.0), processes=1, seed=0)
    main = int(np.argmax([sum(k * v for k, v in h.items()) for h in r.k_hist]))
    assert r.k_mapn[main] == 8


def test_isolated_emitter_lambda5_dstorm():
    rng = np.random.default_rng(5)
    xy = np.array([500.0, 500.0]) + 12.0 * rng.standard_normal((5, 2))
    r = fit_emitters(xy, 12.0, XiPrior.poisson(5.0), processes=1, seed=0)
    assert r.k_mapn.tolist() == [1]
    assert r.n_emitters == 1
    assert (r.labels == 0).all()


# ----------------------------------------------------------------------------- subregions, z, seeds, registry
def test_tile_borders_neither_lose_nor_duplicate_emitters():
    rng = np.random.default_rng(4)
    pos = np.column_stack([np.arange(11) * 60.0, np.zeros(11)])         # crosses the 200-nm tile borders
    xy, truth = _emitters(pos, 30.0, 4.0, rng, offset=0.0)
    r = fit_emitters(xy, 4.0, XiPrior.poisson(30.0), processes=1, seed=1, n_burnin=500, n_trials=1000)
    assert r.n_emitters == 11
    assert (r.labels >= 0).all()
    est = np.sort(r.x)
    assert np.abs(est - pos[:, 0]).max() < 3.0


def _border_counts(overlap_sigma_factor: float, n_sets: int = 12):
    """dSTORM-like emitter (sigma 12 nm, lambda 14) within 6 nm of the x = 200 tile border (an anchor localization
    at the origin fixes the tiles); returns how many emitters were reported within 40 nm of it, per data set
    (the independent review of F3, section 3.6)."""
    out = []
    for d in (-6.0, -2.0, 2.0, 6.0):
        for sd in range(n_sets):
            rng = np.random.default_rng(20_000 + sd)
            m = max(2, int(rng.poisson(14.0)))
            xy = np.vstack([[0.0, 0.0], np.array([200.0 + d, 100.0]) + 12.0 * rng.standard_normal((m, 2))])
            r = fit_emitters(xy, 12.0, XiPrior.poisson(14.0), processes=1, seed=sd, n_burnin=1000, n_trials=2000,
                             overlap_sigma_factor=overlap_sigma_factor)
            out.append(int((np.hypot(r.x - (200.0 + d), r.y - 100.0) < 40.0).sum()))
    return np.asarray(out)


def test_emitter_on_a_border_is_not_duplicated_with_dstorm_precision():
    """With sigma 12 nm the code's fixed 20-nm overlap cuts the emitter in the far subregion and both keep it;
    the default overlap (>= 4 sigma_95 = 48 nm) sees it whole on both sides."""
    c = _border_counts(ce.OVERLAP_SIGMA_FACTOR)
    assert np.mean(c >= 2) <= 0.05, c
    assert np.mean(c == 0) <= 0.05, c
    fixed = _border_counts(0.0)                       # the code's overlap: the failure this guards against
    assert np.mean(fixed >= 2) >= 0.10, fixed


def test_resolve_overlap():
    s = np.full((100, 2), 12.0)
    assert ce.resolve_overlap(s) == pytest.approx(48.0)
    assert ce.resolve_overlap(s, overlap_sigma_factor=0.0) == pytest.approx(ce.OVERLAP_NM)
    assert ce.resolve_overlap(np.full((100, 2), 1.5)) == pytest.approx(ce.OVERLAP_NM)   # DNA-PAINT: the floor


def test_emitter_beyond_the_data_range_is_kept():
    """The outer core tiles are unbounded: an emitter estimated below the data minimum is not dropped."""
    rng = np.random.default_rng(14)
    # one localization at the data minimum (x and y) in the same pre-clusters as a group 120 nm away: the chain's
    # estimate of its emitter is below the minimum in x or y in ~3/4 of the runs
    xy = np.vstack([[0.0, 0.0], np.array([120.0, 0.0]) + 4.0 * rng.standard_normal((10, 2)) + [0.0, 30.0]])
    for sd in range(4):
        r = fit_emitters(xy, 4.0, XiPrior.poisson(8.0), processes=1, seed=sd, n_burnin=300, n_trials=300)
        assert r.n_emitters == 2, (sd, r.x, r.y)
        assert (r.labels >= 0).all()


def test_empty_emitters_are_reported_apart_from_k(tmp_path):
    """With a small lambda the posterior keeps emitters without localizations (exact for these data, review case
    A: E[K] = 2.44, E[occupied] = 2.27); the result reports the occupied ones per pre-cluster and per emitter."""
    s = 4.0
    rng = np.random.default_rng(1000)
    xy = np.vstack([s * rng.standard_normal((3, 2)), np.array([10.0, 0.0]) + s * rng.standard_normal((3, 2))]) + 500.0
    r = fit_emitters(xy, s, XiPrior.poisson(3.0), processes=1, seed=0, n_burnin=1000, n_trials=8000)
    assert r.k_occupied_mean.shape == r.k_mean.shape == (1,)
    assert 0.08 < r.k_mean[0] - r.k_occupied_mean[0] < 0.3, (r.k_mean, r.k_occupied_mean)
    assert sum(r.k_occupied_hist[0].values()) == 8000
    assert r.n_emitters_with_locs == int(np.count_nonzero(r.n_locs >= 0.5)) <= r.n_emitters
    labelled = set(np.unique(r.labels[r.labels >= 0]).tolist())
    assert labelled <= set(np.flatnonzero(r.has_locs).tolist())
    out = tmp_path / "e.csv"
    ce.write_emitters_csv(r, str(out))
    head = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "has_locs" in head and "k_occupied_mean" in head
    assert r.p_occupied.shape == r.n_locs.shape
    assert np.all((r.p_occupied >= 0.0) & (r.p_occupied <= 1.0))


def test_has_locs_threshold_is_editable(tmp_path):
    """F3 (d): ``has_locs_min`` (API, strategy/registry, CLI) changes only the flag and its count."""
    from tools.clustering_strategies import make_strategy, strategy_registry
    s = 4.0
    rng = np.random.default_rng(1000)
    xy = np.vstack([s * rng.standard_normal((3, 2)), np.array([10.0, 0.0]) + s * rng.standard_normal((3, 2))]) + 500.0
    kw = dict(processes=1, seed=0, n_burnin=500, n_trials=2000)
    lo = fit_emitters(xy, s, XiPrior.poisson(3.0), **kw)
    hi = fit_emitters(xy, s, XiPrior.poisson(3.0), has_locs_min=100.0, **kw)
    assert lo.has_locs_min == ce.HAS_LOCS_MIN == 0.5 and hi.params["has_locs_min"] == 100.0
    assert lo.params["legacy_death_ratio"] is False and lo.params["prune_empty"] is False   # F3 (a): corrected Death
    np.testing.assert_array_equal(lo.n_locs, hi.n_locs)                    # same chain, only the flag moves
    assert lo.n_emitters_with_locs == int(np.count_nonzero(lo.n_locs >= 0.5))
    assert hi.n_emitters_with_locs == int(np.count_nonzero(lo.n_locs >= 100.0)) == 0 < lo.n_emitters_with_locs
    with pytest.raises(ValueError):
        fit_emitters(xy, s, XiPrior.poisson(3.0), has_locs_min=-1.0, **kw)
    d = strategy_registry()["emitters_bagol_like"].defaults["has_locs_min"]
    assert d["value"] == 0.5 and "simulation, this project" in d["origin"] and "recommended" in d["origin"]
    st = make_strategy("emitters_bagol_like", {"processes": 1, "n_burnin": 500, "n_trials": 2000,
                                               "xi_kind": "poisson", "xi_mean": 3.0, "hierarchical": False,
                                               "sigma_nm": s, "has_locs_min": 100.0})
    st.fit(xy)
    assert st.get_params()["has_locs_min"] == 100.0
    assert st.result_ is not None and st.result_.n_emitters_with_locs == hi.n_emitters_with_locs
    src = tmp_path / "locs.csv"
    src.write_text("x,y\n" + "".join(f"{float(a)!r},{float(b)!r}\n" for a, b in xy), encoding="utf-8")
    for thr, expect in (("0.5", lo), ("100", hi)):
        out = str(tmp_path / ("run" + thr))
        assert ce.main([str(src), out, "--sigma", "4", "--xi-kind", "poisson", "--xi-mean", "3", "--processes",
                        "1", "--n-burnin", "500", "--n-trials", "2000", "--has-locs-min", thr]) == 0
        rows = (tmp_path / ("run" + thr + "_emitters.csv")).read_text(encoding="utf-8").splitlines()
        col = rows[0].split(",").index("has_locs")
        assert sum(int(r.split(",")[col]) for r in rows[1:]) == expect.n_emitters_with_locs


def test_orphans_go_to_the_nearest_emitter_within_three_combined_sigmas():
    """F3 (b): sigma_loc 3 nm and the emitter's posterior sd 4 nm combine to 5 nm, so the limit is 15 nm (not
    9 nm, three of the localization's own sigmas); beyond it, or with no emitters, the orphan stays -1 (noise)."""
    ex, ey = np.array([0.0, 100.0]), np.array([0.0, 0.0])
    esx, esy = np.array([4.0, 4.0]), np.array([1.0, 1.0])
    xy = np.array([[12.0, 0.0], [0.0, 14.9], [0.0, -15.5], [100.0, 6.0], [50.0, 0.0]])
    sxy = np.array([[3.0, 3.0], [3.0, 1.0], [3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])
    got = ce.assign_orphans_to_nearest(xy, sxy, ex, ey, esx, esy)
    assert got.tolist() == [0, 0, -1, 1, -1]
    assert ce.assign_orphans_to_nearest(xy, sxy, ex, ey, esx, esy, n_sigma=1.5).tolist() == [-1, -1, -1, 1, -1]
    none = np.empty(0)
    assert ce.assign_orphans_to_nearest(xy, sxy, none, none, none, none).tolist() == [-1] * 5


def test_z_is_the_weighted_mean():
    rng = np.random.default_rng(6)
    pos = np.array([[0.0, 0.0], [80.0, 0.0]])
    xy, truth = _emitters(pos, 30.0, 4.0, rng)
    z = np.where(truth == 0, 100.0, -50.0) + rng.normal(0.0, 20.0, truth.size)
    r = fit_emitters(xy, 4.0, XiPrior.poisson(30.0), z=z, processes=1, seed=0, n_burnin=500, n_trials=1000)
    i0 = int(np.argmin(r.x))
    assert r.z[i0] == pytest.approx(z[truth == 0].mean(), abs=1.0)
    assert r.z[1 - i0] == pytest.approx(z[truth == 1].mean(), abs=1.0)


def test_same_seed_same_result_and_process_count_does_not_matter():
    rng = np.random.default_rng(8)
    pos = np.array([[0.0, 0.0], [12.0, 0.0], [400.0, 0.0], [409.0, 5.0]])
    xy, _ = _emitters(pos, 30.0, 4.0, rng)
    kw = dict(processes=1, seed=11, n_burnin=300, n_trials=600)
    a = fit_emitters(xy, 4.0, XiPrior.gamma(1.0, 30.0), hierarchical=True, **kw)
    b = fit_emitters(xy, 4.0, XiPrior.gamma(1.0, 30.0), hierarchical=True, **kw)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.x, b.x)
    assert a.xi_chain == b.xi_chain
    kw["processes"] = 2
    c = fit_emitters(xy, 4.0, XiPrior.gamma(1.0, 30.0), hierarchical=True, **kw)
    np.testing.assert_array_equal(a.labels, c.labels)
    np.testing.assert_array_equal(a.x, c.x)
    assert a.xi_chain == c.xi_chain


def test_drift_needs_frames():
    with pytest.raises(ValueError):
        fit_emitters(np.zeros((3, 2)), 4.0, XiPrior.poisson(5.0), drift=1.0)


def test_drifting_emitter_is_one_emitter_with_drift_and_several_without():
    rng = np.random.default_rng(12)
    n = 60
    t = np.sort(rng.uniform(0.0, 400.0, n))
    xy = np.column_stack([1000.0 + 0.05 * t, np.full(n, 1000.0)]) + 3.0 * rng.standard_normal((n, 2))
    xi = XiPrior.poisson(60.0)
    still = fit_emitters(xy, 3.0, xi, processes=1, seed=0, n_burnin=1000, n_trials=2000)
    moving = fit_emitters(xy, 3.0, xi, processes=1, seed=0, n_burnin=1000, n_trials=2000, drift=0.1, frames=t)
    assert still.k_mapn[0] > 1
    assert moving.k_mapn.tolist() == [1]
    assert moving.alpha_x[0] == pytest.approx(0.05, abs=0.01)
    assert abs(moving.alpha_y[0]) < 0.01
    assert moving.x[0] == pytest.approx(1000.0 + 0.05 * t.mean(), abs=1.5)     # position at the mean frame


def test_registry_entry_is_experimental_and_labelled():
    from tools.clustering_strategies import make_strategy, strategy_params_json, strategy_registry
    info = strategy_registry()["emitters_bagol_like"]
    assert info.experimental is True
    assert info.family == "model-based"
    assert "BaGoL-like reimplementation" in info.description
    assert all(d["origin"] for d in info.defaults.values())
    rng = np.random.default_rng(9)
    xy, truth = _emitters(np.array([[0.0, 0.0], [60.0, 0.0]]), 20.0, 4.0, rng)
    st = make_strategy("emitters_bagol_like", {"processes": 1, "n_burnin": 300, "n_trials": 300,
                                               "xi_kind": "poisson", "xi_mean": 20.0, "hierarchical": False})
    data = np.column_stack([xy, np.full(xy.shape[0], 4.0), np.full(xy.shape[0], 4.0)])
    labels = st.fit(data)
    assert labels.shape == (xy.shape[0],)
    assert set(np.unique(labels)) == {0, 1}
    assert st.get_strategy_name() == ce.PUBLIC_LABEL
    json.dumps(strategy_params_json(st))


def test_strategy_uses_per_localization_precisions_when_given():
    rng = np.random.default_rng(9)
    xy, _ = _emitters(np.array([[0.0, 0.0], [60.0, 0.0]]), 20.0, 2.0, rng)
    st = ce.EmittersBagolLikeStrategy(processes=1, n_burnin=200, n_trials=200, xi_kind="poisson", xi_mean=20.0,
                                      hierarchical=False)
    st.fit(xy)                                                     # (n, 2): sigma_nm = 12 for every localization
    assert st.result_ is not None and st.result_.params["overlap_used_nm"] == pytest.approx(48.0)
    st.set_localization_precision(np.full((xy.shape[0], 2), 2.0))   # e.g. the file's X_SE, Y_SE
    labels = st.fit(xy)
    assert st.result_.params["overlap_used_nm"] == pytest.approx(ce.OVERLAP_NM)
    assert set(np.unique(labels)) == {0, 1}
    assert st.get_params()["per_localization_sigma"] is True
    with pytest.raises(ValueError):
        st.fit(xy[:-1])                                            # stale precisions of another slab
    st.set_localization_precision(None)
    assert st.get_params()["per_localization_sigma"] is False


def test_cli_writes_emitters_and_external_labels(tmp_path):
    from tools.cluster_benchmark import ExternalLabelsStrategy
    rng = np.random.default_rng(13)
    xy, _ = _emitters(np.array([[0.0, 0.0], [80.0, 0.0]]), 20.0, 4.0, rng)
    src = tmp_path / "locs.csv"
    with open(src, "w", encoding="utf-8") as fh:
        fh.write("x,y,sigma\n")
        for xv, yv in xy:
            fh.write(f"{float(xv)!r},{float(yv)!r},4.0\n")
    out = str(tmp_path / "run")
    assert ce.main([str(src), out, "--xi-kind", "poisson", "--xi-mean", "20", "--processes", "1",
                    "--n-burnin", "300", "--n-trials", "300"]) == 0
    rows = (tmp_path / "run_emitters.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 and "BaGoL-like reimplementation" in rows[1]
    labels = ExternalLabelsStrategy(out + "_labels.csv").fit(xy)
    assert set(np.unique(labels)) == {0, 1}


# ----------------------------------------------------------------------------- allocation prior (F4)
def _exp_field(seed: int, n_em: int, side: float, mean: float, sigma: float):
    """Emitters uniform in a square, max(1, round(Exp(mean))) localizations each (the F3/F4 dense synthetic)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, side, (n_em, 2)) + 1000.0
    m = np.maximum(1, np.round(rng.exponential(mean, n_em))).astype(int)
    xy = np.vstack([pos[k] + sigma * rng.standard_normal((m[k], 2)) for k in range(n_em)])
    tr = np.repeat(np.arange(n_em), m)
    return xy, tr, pos, m


def _k_est_per_true(res, tr: np.ndarray, n_em: int) -> np.ndarray:
    """Estimated emitters per true emitter: every estimate goes to the true emitter that owns most of its
    localizations (by the labels)."""
    owner = np.full(res.n_emitters, -1, dtype=int)
    for e in range(res.n_emitters):
        ids = tr[res.labels == e]
        if ids.size:
            v, c = np.unique(ids, return_counts=True)
            owner[e] = v[np.argmax(c)]
    return np.bincount(owner[owner >= 0], minlength=n_em)


@pytest.mark.parametrize("alloc", ["uniform", "dirichlet"])
def test_allocation_priors_match_their_exact_posterior(alloc):
    """Case B (8 localizations, two groups 8 nm apart, sigma 4) with the gamma xi Gamma(1, 4): each allocation prior
    has its own exact posterior by enumeration (uniform: K^-N; Dirichlet: the Dirichlet-multinomial) and the chain
    must reproduce it. The two posteriors differ (E[K] 2.09 vs 2.39), so the test tells the priors apart.
    150,000 steps: with 60,000 the Monte Carlo error of E[K] (about 0.03) is as large as the tolerance."""
    s = 4.0
    rng = np.random.default_rng(11)
    xy = np.vstack([s * rng.standard_normal((4, 2)), np.array([8.0, 0.0]) + s * rng.standard_normal((4, 2))]) + 500.0
    n = xy.shape[0]
    xi = XiPrior.gamma(1.0, 4.0)
    cfg = ChainConfig(n_burnin=2000, n_trials=150000, allocation=alloc)
    ch = EmitterChain(xy[:, 0], xy[:, 1], np.full(n, s), np.full(n, s), 5, cfg, xi)
    assert (ch.logw is not None) == (alloc == "dirichlet")
    pk, _ = _exact_k_posterior(xy, s, xi, ch, alpha=(xi.k if alloc == "dirichlet" else 0.0))
    ek = float(np.sum(np.arange(pk.size) * pk))
    assert (2.0 < ek < 2.2) if alloc == "uniform" else (2.3 < ek < 2.5)
    ch.run(152000, xi)
    k = np.asarray(ch.k_rec)
    assert abs(k.mean() - ek) < 0.03, (alloc, k.mean(), ek)     # 3 seeds x 150k: +0.002 (F4)
    for kk in range(1, 5):
        assert abs(np.mean(k == kk) - pk[kk]) < 0.025, (alloc, kk, np.mean(k == kk), pk[kk])


def test_dirichlet_allocation_does_not_split_bright_emitters():
    """Dense field (120 emitters in 800 x 800 nm, exponential counts of mean 14, sigma 12 nm), xi fixed at the truth.
    With the code's uniform allocation the emitters with >= 29 localizations are split into >= 2 estimates most of
    the time (F4: 85-100 %); with the Dirichlet allocation they are not, and the excess of emitters drops."""
    xy, tr, pos, m = _exp_field(4, 120, 800.0, 14.0, 12.0)
    bright = m >= 29
    assert bright.sum() >= 8
    out = {}
    for alloc in ("uniform", "dirichlet"):
        res = fit_emitters(xy, 12.0, XiPrior.gamma(1.0, 14.0), processes=1, seed=0, n_burnin=1000, n_trials=1500,
                           allocation=alloc)
        assert res.params["allocation_used"] == alloc
        k_est = _k_est_per_true(res, tr, pos.shape[0])
        out[alloc] = (float(np.mean(k_est[bright] >= 2)), res.n_emitters_with_locs / pos.shape[0] - 1.0)
    assert out["uniform"][0] >= 0.5, out
    assert out["dirichlet"][0] <= 0.25, out
    assert out["dirichlet"][1] < out["uniform"][1] - 0.05, out


def test_poisson_xi_always_uses_the_uniform_allocation():
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 300, (6, 2))
    xy, _ = _emitters(pos, 8.0, 5.0, rng)
    a = fit_emitters(xy, 5.0, XiPrior.poisson(8.0), processes=1, seed=3, n_burnin=200, n_trials=300,
                     allocation="dirichlet")
    b = fit_emitters(xy, 5.0, XiPrior.poisson(8.0), processes=1, seed=3, n_burnin=200, n_trials=300,
                     allocation="uniform")
    assert a.params["allocation_used"] == "uniform" and a.params["allocation"] == "dirichlet"
    assert np.array_equal(a.labels, b.labels) and np.array_equal(a.x, b.x)
    with pytest.raises(ValueError):
        fit_emitters(xy, 5.0, XiPrior.poisson(8.0), processes=1, allocation="equal")


def test_matlab_compat_implies_the_uniform_allocation():
    rng = np.random.default_rng(5)
    xy, _ = _emitters(rng.uniform(0, 200, (4, 2)), 6.0, 5.0, rng)
    res = fit_emitters(xy, 5.0, XiPrior.gamma(1.0, 6.0), processes=1, seed=1, n_burnin=100, n_trials=150,
                       matlab_compat=True)
    assert res.params["allocation"] == "uniform" and res.params["allocation_used"] == "uniform"
    assert ChainConfig.matlab_compat().allocation == "uniform"


# ----------------------------------------------------------------------------- singletons and xi counts (F5b)
def test_xi_counts_leave_out_singletons_and_matlab_compat_restores_the_code():
    """An isolated background of one-localization pre-clusters beside bright emitters: counting them for the
    hierarchical xi dilutes its mean (F4 part B); 'nonsingleton' (default) leaves them out."""
    rng = np.random.default_rng(7)
    pos = np.array([[400.0 * (j % 5), 400.0 * (j // 5)] for j in range(15)])
    m = np.maximum(2, np.round(rng.exponential(20.0, 15))).astype(int)
    em = np.vstack([pos[k] + 8.0 * rng.standard_normal((m[k], 2)) for k in range(15)])
    bg = rng.uniform(-200, 1800, (60, 2))
    bg = bg[np.min(np.hypot(*(bg[:, None, :] - pos[None, :, :]).transpose(2, 0, 1)), axis=1) > 150.0]
    xy = np.vstack([em, bg]) + 1000.0
    kw = dict(hierarchical=True, processes=1, seed=0, n_burnin=600, n_trials=600)
    a = fit_emitters(xy, 8.0, XiPrior.gamma(1.0, 14.0), **kw)
    b = fit_emitters(xy, 8.0, XiPrior.gamma(1.0, 14.0), xi_counts="occupied", **kw)
    assert a.params["xi_counts"] == "nonsingleton" and b.params["xi_counts"] == "occupied"
    mean_a, mean_b = a.xi_used[0] * a.xi_used[1], b.xi_used[0] * b.xi_used[1]
    assert mean_a == pytest.approx(float(m.mean()), rel=0.3), (mean_a, m.mean())
    assert mean_b < 0.6 * mean_a, (mean_a, mean_b)                  # diluted by the singletons
    assert a.n_singletons >= 0.8 * bg.shape[0] and a.n_emitters_nonsingleton <= 20
    assert a.n_singletons + a.n_emitters_nonsingleton == a.n_emitters_with_locs
    c = fit_emitters(xy, 8.0, XiPrior.gamma(1.0, 14.0), matlab_compat=True, processes=1, seed=0, n_burnin=50,
                     n_trials=50)
    assert c.params["xi_counts"] == "occupied" and ChainConfig.matlab_compat().xi_counts == "occupied"
    with pytest.raises(ValueError):
        fit_emitters(xy, 8.0, XiPrior.gamma(1.0, 14.0), processes=1, xi_counts="all")


def test_prune_empty_with_the_dirichlet_allocation_warns():
    rng = np.random.default_rng(5)
    xy, _ = _emitters(rng.uniform(0, 200, (3, 2)), 6.0, 5.0, rng)
    with pytest.warns(UserWarning, match="prune_empty"):
        fit_emitters(xy, 5.0, XiPrior.gamma(1.0, 6.0), processes=1, n_burnin=20, n_trials=20, prune_empty=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_emitters(xy, 5.0, XiPrior.gamma(1.0, 6.0), processes=1, n_burnin=20, n_trials=20, matlab_compat=True)
        fit_emitters(xy, 5.0, XiPrior.gamma(1.0, 6.0), processes=1, n_burnin=20, n_trials=20, prune_empty=True,
                     allocation="uniform")
