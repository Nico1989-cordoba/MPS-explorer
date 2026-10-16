"""Support simulation of docs/viability_and_reference_period.md, section 6 (synthetic only; no real data).

Rule (v) replacement: is the Mann-Whitney p of 'matched size ratios vs all unmatched cross-pair ratios of the
same two rings' calibrated, and does a within-ring permutation (Monte Carlo rank) test fix it?

Model of one ring pair:
  ring a: K_a usable clusters, ring b: K_b usable clusters, sizes (localizations) iid lognormal with median 100 and
  coefficient of variation cv, rounded, floored at 10 (DBSCAN min_samples).
  H0 (genuine columns): the one-to-one matching is unrelated to size -> n_m disjoint cross pairs drawn at random.
  H1 (directional leak): n_leak sources in ring a each leave a binomially thinned copy (fraction F) in ring b;
     copies below 10 localizations are not clusters; matched pairs = the source-copy pairs + n_m - n_leak random
     genuine pairs.
  Ratio of a pair = min(size)/max(size).

Tests (one-sided, small ratios = leak-like), alpha = 0.05:
  MW  : scipy mannwhitneyu(matched ratios, unmatched cross-pair ratios, alternative='less'), tabulated/asymptotic p.
  MCm : statistic = median matched ratio; null = sizes permuted within each ring with the matching fixed
        (equivalently n_m disjoint cross pairs redrawn at random); p = (b + 1)/(B + 1), b = #{null <= observed}.
  MCr : same null, statistic = sum of the ranks of the matched ratios among all K_a x K_b cross-pair ratios
        (the Mann-Whitney U as a ranking statistic only).
"""
import json
import time

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

rng = np.random.default_rng(20261009)
B = 199          # Monte Carlo null draws per test (alpha (B + 1) integer at 0.05)
N_REP = 1000     # replicate ring pairs per scenario
ALPHA = 0.05


def sizes(n, cv):
    s_log = np.sqrt(np.log1p(cv ** 2))
    return np.maximum(10, np.rint(100.0 * np.exp(rng.normal(0.0, s_log, n)))).astype(int)


def ratio(x, y):
    return np.minimum(x, y) / np.maximum(x, y)


def one_test(sa, sb, ia, jb):
    """sa, sb: sizes of the usable clusters of ring a / ring b; (ia, jb): matched index pairs."""
    Ka, Kb = sa.size, sb.size
    R = ratio(sa[:, None], sb[None, :])               # all cross-pair ratios
    mask = np.zeros((Ka, Kb), bool)
    mask[ia, jb] = True
    r_m = R[mask]
    r_u = R[~mask]
    p_mw = mannwhitneyu(r_m, r_u, alternative="less").pvalue
    t_med = np.median(r_m)
    ranks = rankdata(R.ravel()).reshape(Ka, Kb)
    t_rank = ranks[ia, jb].sum()
    nm = ia.size
    # null: n_m disjoint cross pairs at random (= within-ring permutation of sizes, matching fixed)
    a_idx = np.argsort(rng.random((B, Ka)), axis=1)[:, :nm]
    b_idx = np.argsort(rng.random((B, Kb)), axis=1)[:, :nm]
    null_med = np.median(R[a_idx, b_idx], axis=1)
    null_rank = ranks[a_idx, b_idx].sum(axis=1)
    p_med = (1 + np.sum(null_med <= t_med)) / (B + 1)
    p_rank = (1 + np.sum(null_rank <= t_rank)) / (B + 1)
    return p_mw, p_med, p_rank


def scenario(K, nm, cv, n_leak=0, F=0.15):
    rej = np.zeros(3)
    for _ in range(N_REP):
        sa = sizes(K, cv)
        if n_leak == 0:
            sb = sizes(K, cv)
            ia = rng.choice(K, nm, replace=False)
            jb = rng.choice(K, nm, replace=False)
        else:
            src = rng.choice(K, n_leak, replace=False)
            copies = rng.binomial(sa[src], F)
            keep = copies >= 10
            src, copies = src[keep], copies[keep]
            n_gen_b = K - src.size
            sb = np.concatenate([copies, sizes(n_gen_b, cv)])
            # matched: copies (b indices 0..len(src)-1) with their sources, plus random genuine pairs
            n_rand = max(0, nm - src.size)
            others_a = np.setdiff1d(np.arange(K), src)
            ia_r = rng.choice(others_a, n_rand, replace=False)
            jb_r = src.size + rng.choice(n_gen_b, n_rand, replace=False)
            ia = np.concatenate([src, ia_r])
            jb = np.concatenate([np.arange(src.size), jb_r])
        p = one_test(sa, sb, ia, jb)
        rej += np.array(p) <= ALPHA
    return (rej / N_REP).round(3).tolist()


t0 = time.time()
out = {"B": B, "replicates": N_REP, "alpha": ALPHA, "columns": ["MW tabulated", "MC median", "MC rank-sum"]}
for K, nm in [(20, 8), (40, 12), (60, 20)]:
    for cv in (0.7, 1.0):
        out[f"H0_K{K}_nm{nm}_cv{cv}"] = scenario(K, nm, cv)
# extra H0 regimes: few clusters, most clusters matched, narrow and wide size spreads
for K, nm, cv in [(10, 6, 0.7), (20, 16, 0.7), (40, 36, 0.7), (40, 12, 0.4), (40, 12, 1.5), (40, 36, 1.5)]:
    out[f"H0_K{K}_nm{nm}_cv{cv}"] = scenario(K, nm, cv)
for K, nm, nl in [(40, 12, 6), (40, 12, 10), (60, 20, 10)]:
    out[f"H1_leak_K{K}_nm{nm}_nleak{nl}_cv0.7_F0.15"] = scenario(K, nm, 0.7, n_leak=nl, F=0.15)
out["seconds"] = round(time.time() - t0, 1)
print(json.dumps(out, indent=1))
with open(__file__.replace(".py", "_out.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
