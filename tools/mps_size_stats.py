# -*- coding: utf-8 -*-
"""
Across axons and animals: how the per-axon size readouts are combined without pseudoreplication
(``docs/apparent_size.md``, sec. "Units and covariates").

Decisions of 2026-10-22 (answers to the apparent-size report, sec. 9):

* 5a -- for question (b), size vs curvature, the AXON is the unit: the per-axon Fisher z of Spearman(content,
  kappa R) combined by a one-sample t over axons, with the animal as second level (the mean z per animal, t over
  animals; :func:`combine_axons_fisher`), or a mixed model with a random intercept per axon on the clusters
  (:func:`mixed_random_intercept`). For group comparisons (genotypes) the ANIMAL is the unit, or a mixed model with
  a random intercept per animal (:func:`compare_groups`). With clusters as units the simulated null was rejected
  71-92 % of the time, with axons as units 28-44 %, with animals or the mixed model 2-8 % [simulation: SIZE_SIM
  sec. 4.2; lit: Lazic 2010; Aarts et al. 2014].
* 6a -- covariates: the APPARENT estimators (cloud SD, r_rare, gyration radius) always with the file's NeNA and
  log N_loc (the precision enters them at 1-2.3 nm per nm of NeNA); the footprint rho_pool with NONE (it already
  deconvolves the NeNA, and two covariates on 6 animals leave 2 degrees of freedom: power 87 % -> 31 %)
  [simulation: SIZE_SIM sec. 3.1, 4.2]. The CONTENT gets none either [ad hoc: log N_loc cannot be its own
  covariate; NeNA does not change a count of blinks].
* 7 -- the design size (animals per group) is decided after a pilot that measures the label regime (P-R75):
  :func:`design_guidance` documents the simulation's table and sets NO default.

No Qt; numpy + scipy only (no statsmodels in the venv: the REML model is implemented here, ported from the
simulation, where it gave 4.7 % rejection under the null).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist

__all__ = ["ESTIMATOR_KINDS", "COVARIATES_BY_KIND", "combine_axons_fisher", "mixed_random_intercept",
           "GroupComparison", "compare_groups", "design_guidance"]

ESTIMATOR_KINDS = ("content", "apparent", "pool")
COVARIATES_BY_KIND: Dict[str, Sequence[str]] = {"content": (), "apparent": ("nena_nm", "log_n_locs"), "pool": ()}


def _t_one_sample(v: NDArray[np.float64]) -> Dict[str, Optional[float]]:
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 2:
        return {"mean": float(v.mean()) if n else None, "t": None, "p": None, "n": n}
    sd = float(v.std(ddof=1))
    if sd == 0:
        return {"mean": float(v.mean()), "t": None, "p": None, "n": n}
    tt = float(v.mean()) / (sd / math.sqrt(n))
    return {"mean": float(v.mean()), "t": tt, "p": float(2 * t_dist.sf(abs(tt), n - 1)), "n": n}


def combine_axons_fisher(z: Sequence[Optional[float]], animal: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Per-axon Fisher z (``tools.mps_apparent_size.fisher_z``) combined across axons: ``axon_*`` the one-sample t
    on the axons' z (axon as unit, decision 5a); ``animal_*`` the same on the mean z of each animal (animal as
    second level; with 3 animals the t has 2 df and loses about half the power [simulation: SIZE_SIM sec. 6.3])."""
    zz = np.array([np.nan if v is None else float(v) for v in z], dtype=np.float64)
    out: Dict[str, Any] = {f"axon_{k}": v for k, v in _t_one_sample(zz).items()}
    if animal is not None:
        an = np.asarray([str(a) for a in animal])
        if an.size != zz.size:
            raise ValueError("combine_axons_fisher: one animal per axon")
        ok = np.isfinite(zz)
        means = np.array([zz[ok & (an == a)].mean() for a in sorted(set(an[ok].tolist()))], dtype=np.float64)
        out.update({f"animal_{k}": v for k, v in _t_one_sample(means).items()})
    return out


def mixed_random_intercept(y: Sequence[float], X: Any, group: Sequence[Any]) -> Dict[str, Any]:
    """y = b0 + X b + a_group + e with a ~ N(0, s_a^2), e ~ N(0, s_e^2), fitted by REML profiled over the variance
    ratio [lit: Aarts et al. 2014 for the use; Patterson & Thompson 1971 for REML]. ``X`` (n,) or (n, p) WITHOUT the
    intercept (added). Returns beta, se, t, p (t on G - p df, conservative: the between-group information), the
    variance ratio, n and G. Rows with a non-finite value are dropped."""
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    Xv = np.asarray(X, dtype=np.float64)
    if Xv.ndim == 1:
        Xv = Xv[:, None]
    gv = np.asarray([str(g) for g in group])
    ok = np.isfinite(yv) & np.all(np.isfinite(Xv), axis=1)
    yv, Xv, gv = yv[ok], np.column_stack([np.ones(int(ok.sum())), Xv[ok]]), gv[ok]
    n, p = Xv.shape
    gids, ginv = np.unique(gv, return_inverse=True)
    G = int(gids.size)
    if n <= p or G < 2:
        raise ValueError(f"mixed_random_intercept: {n} rows, {G} groups for {p} coefficients")
    ng = np.bincount(ginv).astype(np.float64)

    def vinv(theta: float, M: NDArray[np.float64]) -> NDArray[np.float64]:
        sums = np.zeros((G,) + M.shape[1:])
        np.add.at(sums, ginv, M)
        fac = theta / (1.0 + theta * ng)
        corr = (fac[:, None] * sums)[ginv] if M.ndim == 2 else (fac * sums)[ginv]
        return np.asarray(M - corr, dtype=np.float64)

    def fit(log_theta: float) -> Any:
        theta = math.exp(log_theta)
        xtvx = Xv.T @ vinv(theta, Xv)
        beta = np.linalg.solve(xtvx, Xv.T @ vinv(theta, yv))
        r = yv - Xv @ beta
        s2 = float(r @ vinv(theta, r)) / (n - p)
        return theta, beta, xtvx, s2

    def reml(log_theta: float) -> float:
        theta, _beta, xtvx, s2 = fit(log_theta)
        value = 0.5 * ((n - p) * math.log(max(float(s2), 1e-300)) + float(np.sum(np.log1p(float(theta) * ng)))
                       + float(np.linalg.slogdet(xtvx)[1]))
        return float(value)

    res = minimize_scalar(reml, bounds=(-12.0, 8.0), method="bounded")
    theta, beta, xtvx, s2 = fit(float(res.x))
    se = np.sqrt(np.diag(s2 * np.linalg.inv(xtvx)))
    tt = beta / se
    df = max(G - p, 1)
    return {"beta": beta, "se": se, "t": tt, "p": 2 * t_dist.sf(np.abs(tt), df), "df": df, "theta": theta,
            "n": n, "G": G}


@dataclass
class GroupComparison:
    """Two groups compared with the right unit: ``animal_t`` / ``animal_p`` the Welch t on the animal means;
    ``mixed_*`` the group coefficient of the mixed model with a random intercept per animal and the covariates of
    the estimator kind (decision 6a); ``covariates`` which ones went in."""
    estimator_kind: str
    covariates: List[str]
    n_axons: int
    n_animals: int
    animal_diff: Optional[float]
    animal_t: Optional[float]
    animal_p: Optional[float]
    mixed_diff: Optional[float]
    mixed_p: Optional[float]
    mixed_df: Optional[int]
    warnings: List[str]


def compare_groups(value: Sequence[Optional[float]], group: Sequence[Any], animal: Sequence[Any], *,
                   estimator_kind: str, nena_nm: Optional[Sequence[Optional[float]]] = None,
                   log_n_locs: Optional[Sequence[Optional[float]]] = None) -> GroupComparison:
    """Per-axon values (e.g. ``size_rho_pool_nm``, ``size_cloud_sd_median_nm``, ``size_n_locs_median``) of two groups
    (genotypes), animals nested in groups. ``estimator_kind`` in ``ESTIMATOR_KINDS`` sets the covariates
    (``COVARIATES_BY_KIND``): "apparent" needs ``nena_nm`` and ``log_n_locs`` per axon (the median log N_loc of the
    axon's clusters), "pool" and "content" take none."""
    if estimator_kind not in ESTIMATOR_KINDS:
        raise ValueError(f"estimator_kind must be one of {ESTIMATOR_KINDS}")
    v = np.array([np.nan if a is None else float(a) for a in value], dtype=np.float64)
    g = np.asarray([str(a) for a in group])
    an = np.asarray([str(a) for a in animal])
    if not (v.size == g.size == an.size):
        raise ValueError("compare_groups: value, group and animal must have one entry per axon")
    levels = sorted(set(g.tolist()))
    if len(levels) != 2:
        raise ValueError(f"compare_groups: two groups expected, got {levels}")
    warnings: List[str] = []
    covs: List[NDArray[np.float64]] = []
    names = list(COVARIATES_BY_KIND[estimator_kind])
    for nm, arr in (("nena_nm", nena_nm), ("log_n_locs", log_n_locs)):
        if nm in names:
            if arr is None:
                raise ValueError(f"compare_groups: the {estimator_kind} estimators need {nm} per axon (decision 6a)")
            covs.append(np.array([np.nan if a is None else float(a) for a in arr], dtype=np.float64))
    ok = np.isfinite(v)
    for c in covs:
        ok &= np.isfinite(c)
    animals = sorted(set(an[ok].tolist()))
    a_mean = {a: float(v[ok & (an == a)].mean()) for a in animals}
    a_group = {a: g[ok & (an == a)][0] for a in animals}
    gm = [np.array([a_mean[a] for a in animals if a_group[a] == lv]) for lv in levels]
    animal_diff = animal_t = animal_p = None
    if all(x.size >= 2 for x in gm):
        va, vb = gm[1].var(ddof=1) / gm[1].size, gm[0].var(ddof=1) / gm[0].size
        animal_diff = float(gm[1].mean() - gm[0].mean())
        if va + vb > 0:
            animal_t = animal_diff / math.sqrt(va + vb)
            df = (va + vb) ** 2 / (va ** 2 / (gm[1].size - 1) + vb ** 2 / (gm[0].size - 1))
            animal_p = float(2 * t_dist.sf(abs(animal_t), df))
    else:
        warnings.append("fewer than two animals in a group: no animal-level test")
    n_an = len(animals)
    if n_an < 6:
        warnings.append(f"{n_an} animals: a two-sided test with animals as units needs at least 6 [lit: Lazic 2010]")
    mixed_diff = mixed_p = None
    mixed_df: Optional[int] = None
    X = np.column_stack([(g[ok] == levels[1]).astype(np.float64)] + [c[ok] for c in covs])
    try:
        fit = mixed_random_intercept(v[ok].tolist(), X, an[ok].tolist())
        mixed_diff, mixed_p, mixed_df = float(fit["beta"][1]), float(fit["p"][1]), int(fit["df"])
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.append(f"mixed model not fitted: {exc}")
    return GroupComparison(estimator_kind, names, int(ok.sum()), n_an, animal_diff, animal_t, animal_p,
                           mixed_diff, mixed_p, mixed_df, warnings)


def design_guidance() -> List[Dict[str, str]]:
    """The simulation's guidance on animals per group for a 5 nm footprint difference (25 vs 20 nm) with the pool
    and the animal as unit -- NO default (decision 7: decide after a pilot measures events per cluster / c_label,
    P-R75). Every row is [simulation: SIZE_SIM sec. 4.1, 7.3]; the 5-6 animals row is an extrapolation (power of a
    t scales with sqrt(n)), not simulated."""
    src = "simulation: SIZE_SIM sec. 4.1 (3 animals x 6 axons per group, 100 replicates)"
    return [
        {"labels_per_cluster": "~1", "power_3_animals": "not measurable (footprint not identifiable)",
         "guidance": "compare content only", "origin": src},
        {"labels_per_cluster": "~2", "power_3_animals": "44-53 %",
         "guidance": "~5-6 animals per group (extrapolation, not simulated)", "origin": src},
        {"labels_per_cluster": ">= 3", "power_3_animals": "87-89 %", "guidance": "3 animals per group", "origin": src},
        {"labels_per_cluster": "5", "power_3_animals": "97-98 %", "guidance": "3 animals per group", "origin": src},
    ]
