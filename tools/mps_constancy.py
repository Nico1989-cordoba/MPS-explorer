# -*- coding: utf-8 -*-
"""
Perimeter occupancy against axon calibre, with animals as the unit of replication (SCI-3).

Gazal et al. (2026) state that perimeter occupancy is roughly constant across axon calibres, a conclusion drawn from
the linear scaling of cluster number with perimeter; the preprint shows no plot or test of occupancy against calibre
and its public review questions the claim. This module is the kept script that examines the relation on new data;
the program makes no claim about it.

Why not a pooled correlation
----------------------------
Axons are nested in regions (ROIs) and animals. A pooled Pearson r over axons treats them as independent, which is
pseudoreplication (Lazic 2010; Aarts et al. 2014: false positives up to 80 %), and r near 0 is absence of evidence,
not constancy: with two regions it can arise from cancellation. Constancy needs an interval on the slope, compared
with an equivalence bound (e.g. +/-1 percentage point per 10 um of perimeter). Added confound: occupancy depends on
the localizations per cluster through the sigma cap (``tools.mps_occupancy``), so a calibre-linked change in labelling
density can masquerade as a calibre effect.

What it reports
---------------
* the pooled Pearson r with a Fisher interval (Fisher 1915), labelled as axon-level (pseudoreplicated) context;
* the same r with a bootstrap interval that resamples ANIMALS;
* a linear mixed model occupancy = b0 + b1 perimeter + animal + ROI-within-animal + error (random intercepts,
  REML, fitted here: no extra dependency), with the slope's interval on n_animals - 2 degrees of freedom (the
  between-animal information; conservative) and whether that interval lies inside the equivalence bound.

It refuses with fewer than three animals and warns below six (a two-sided test at 5 % with animals as units needs
at least six; the design decision belongs to the experimenter and their director).

Run:  python -m tools.mps_constancy <axon table.csv> [--bound 1.0]

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

MIN_ANIMALS = 3
WARN_ANIMALS = 6
DEFAULT_BOUND_PP_PER_10UM = 1.0
DEFAULT_BOOTSTRAP = 2000


@dataclass
class ConstancyResult:
    n_axons: int
    n_animals: int
    n_rois: int
    pearson_r: float
    fisher_ci: Tuple[float, float]
    animal_bootstrap_ci: Optional[Tuple[float, float]]
    slope_pp_per_um: float
    slope_ci_pp_per_um: Tuple[float, float]
    slope_df: int
    var_animal: float
    var_roi: float
    var_residual: float
    bound_pp_per_10um: float
    within_bound: bool
    warnings: List[str] = field(default_factory=list)

    def lines(self) -> List[str]:
        b = self.bound_pp_per_10um
        out = [
            f"{self.n_axons} axons, {self.n_rois} ROIs, {self.n_animals} animals",
            f"pooled Pearson r = {self.pearson_r:+.2f} (Fisher 95 % CI {self.fisher_ci[0]:+.2f} to "
            f"{self.fisher_ci[1]:+.2f}; axon level, pseudoreplicated: context only)",
        ]
        if self.animal_bootstrap_ci is not None:
            out.append(f"  bootstrap over animals: 95 % CI {self.animal_bootstrap_ci[0]:+.2f} to "
                       f"{self.animal_bootstrap_ci[1]:+.2f}")
        lo, hi = self.slope_ci_pp_per_um
        out.append(f"mixed model (animal and ROI-within-animal random intercepts, REML): slope "
                   f"{10 * self.slope_pp_per_um:+.2f} pp per 10 um (95 % CI {10 * lo:+.2f} to {10 * hi:+.2f}; "
                   f"t on {self.slope_df} df)")
        out.append(f"  variance: animal {self.var_animal:.3g}, ROI {self.var_roi:.3g}, residual "
                   f"{self.var_residual:.3g} (pp^2)")
        out.append(f"  equivalence bound +/-{b:g} pp per 10 um: "
                   + ("the interval lies inside it" if self.within_bound
                      else "the interval is not inside it (constancy not shown)"))
        out += [f"! {w}" for w in self.warnings]
        return out


def _design(groups: Sequence[str]) -> NDArray[np.float64]:
    labels = sorted(set(groups))
    index = {g: i for i, g in enumerate(labels)}
    z = np.zeros((len(groups), len(labels)))
    for row, g in enumerate(groups):
        z[row, index[g]] = 1.0
    return z


def _reml(y: NDArray[np.float64], x: NDArray[np.float64], zs: Sequence[NDArray[np.float64]],
          log_theta: NDArray[np.float64]) -> Tuple[float, NDArray[np.float64], NDArray[np.float64], float]:
    """-2 x the profiled REML log-likelihood at variance ratios exp(log_theta); with beta, cov(beta)/sigma^2,
    sigma^2."""
    n, p = x.shape
    v = np.eye(n)
    for z, lt in zip(zs, log_theta):
        v = v + math.exp(float(lt)) * (z @ z.T)
    chol = np.linalg.cholesky(v)
    xi = np.linalg.solve(chol, x)
    yi = np.linalg.solve(chol, y)
    xtx = xi.T @ xi
    beta = np.linalg.solve(xtx, xi.T @ yi)
    resid = yi - xi @ beta
    sigma2 = float(resid @ resid) / (n - p)
    logdet_v = 2.0 * float(np.sum(np.log(np.diag(chol))))
    logdet_x = float(np.linalg.slogdet(xtx)[1])
    value = (n - p) * math.log(max(sigma2, 1e-300)) + logdet_v + logdet_x
    return value, beta, np.linalg.inv(xtx), sigma2


def fit_random_intercepts(y: Sequence[float], x1: Sequence[float], groups: Sequence[Sequence[str]]
                          ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, List[float]]:
    """REML fit of y = b0 + b1 x1 + sum of random intercepts (one per grouping) + error. Returns beta, cov(beta),
    the residual variance and the variance of each grouping."""
    yv = np.asarray(y, dtype=float)
    xv = np.column_stack([np.ones(len(yv)), np.asarray(x1, dtype=float)])
    zs = [_design(g) for g in groups]

    def f(lt: NDArray[np.float64]) -> float:
        return _reml(yv, xv, zs, lt)[0]

    best = None
    for start in (-2.0, 0.0, 2.0):
        res = optimize.minimize(f, np.full(len(zs), start), method="Nelder-Mead",
                                options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    lt = np.clip(best.x, -30.0, 30.0)
    _v, beta, cov_unit, sigma2 = _reml(yv, xv, zs, lt)
    return beta, cov_unit * sigma2, sigma2, [math.exp(float(t)) * sigma2 for t in lt]


def occupancy_vs_calibre(
    occupancy_percent: Sequence[float], perimeter_um: Sequence[float], animal: Sequence[str],
    roi: Optional[Sequence[str]] = None, bound_pp_per_10um: float = DEFAULT_BOUND_PP_PER_10UM,
    n_bootstrap: int = DEFAULT_BOOTSTRAP, random_seed: int = 0,
) -> ConstancyResult:
    """The relation of occupancy to perimeter with animals as the unit (see the module docstring)."""
    occ = np.asarray(occupancy_percent, dtype=float)
    per = np.asarray(perimeter_um, dtype=float)
    ani = [str(a) for a in animal]
    rois = [f"{a}/{r}" for a, r in zip(ani, roi)] if roi is not None else [f"{a}/-" for a in ani]
    keep = np.isfinite(occ) & np.isfinite(per) & np.array([bool(a.strip()) for a in ani])
    occ, per = occ[keep], per[keep]
    ani = [a for a, k in zip(ani, keep) if k]
    rois = [r for r, k in zip(rois, keep) if k]
    animals = sorted(set(ani))
    warnings: List[str] = []
    if len(animals) < MIN_ANIMALS:
        raise ValueError(
            f"{len(animals)} animal(s): the relation of occupancy to calibre needs animals as the unit of "
            f"replication, at least {MIN_ANIMALS} (a pooled correlation over axons is pseudoreplication).")
    if len(animals) < WARN_ANIMALS:
        warnings.append(f"Only {len(animals)} animals: a two-sided test at 5 % with animals as units needs at "
                        f"least {WARN_ANIMALS}; read the interval, not a verdict.")
    if np.ptp(per) <= 0:
        raise ValueError("Every axon has the same perimeter: there is no calibre to relate occupancy to.")

    r = float(np.corrcoef(per, occ)[0, 1])
    n = occ.size
    if n > 3 and abs(r) < 1:
        zr = math.atanh(r)
        half = 1.959963984540054 / math.sqrt(n - 3)
        fisher = (math.tanh(zr - half), math.tanh(zr + half))
    else:
        fisher = (float("nan"), float("nan"))

    boot_ci = None
    if n_bootstrap > 0:
        rng = np.random.default_rng(random_seed)
        by_animal = {a: np.flatnonzero(np.array(ani) == a) for a in animals}
        rs = []
        for _ in range(int(n_bootstrap)):
            pick = rng.choice(animals, size=len(animals), replace=True)
            idx = np.concatenate([by_animal[a] for a in pick])
            if np.ptp(per[idx]) > 0 and np.ptp(occ[idx]) > 0:
                rs.append(float(np.corrcoef(per[idx], occ[idx])[0, 1]))
        if rs:
            boot_ci = (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))

    groupings = [ani]
    if len(set(rois)) > len(animals):
        groupings.append(rois)
    beta, cov, s2, variances = fit_random_intercepts(occ, per, groupings)
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    df = max(len(animals) - 2, 1)
    t = float(stats.t.ppf(0.975, df))
    lo, hi = float(beta[1] - t * se), float(beta[1] + t * se)
    bound = float(bound_pp_per_10um) / 10.0
    return ConstancyResult(
        n_axons=int(n), n_animals=len(animals), n_rois=len(set(rois)), pearson_r=r, fisher_ci=fisher,
        animal_bootstrap_ci=boot_ci, slope_pp_per_um=float(beta[1]), slope_ci_pp_per_um=(lo, hi), slope_df=df,
        var_animal=float(variances[0]), var_roi=float(variances[1]) if len(variances) > 1 else 0.0,
        var_residual=float(s2), bound_pp_per_10um=float(bound_pp_per_10um),
        within_bound=bool(-bound <= lo and hi <= bound), warnings=warnings)


def from_rows(rows: Sequence[Mapping[str, str]], suffix: str = "", **kw: object) -> ConstancyResult:
    """From the rows of an axon table (``occupancy_percent``, ``perimeter_um``, ``animal``, ``roi_name``);
    ``suffix`` "_discard" reads the analysis with the discard applied."""
    def num(v: Optional[str]) -> float:
        try:
            return float(v) if v not in (None, "") else float("nan")
        except ValueError:
            return float("nan")
    occ = [num(r.get("occupancy_percent" + suffix)) for r in rows]
    per = [num(r.get("perimeter_um" + suffix)) for r in rows]
    ani = [str(r.get("animal") or "") for r in rows]
    roi = [str(r.get("roi_name") or "") for r in rows]
    return occupancy_vs_calibre(occ, per, ani, roi, **kw)  # type: ignore[arg-type]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Occupancy against calibre with animals as the unit.")
    ap.add_argument("table", help="an axon table (CSV) written by MPS Explorer")
    ap.add_argument("--bound", type=float, default=DEFAULT_BOUND_PP_PER_10UM,
                    help="equivalence bound, percentage points per 10 um of perimeter")
    ap.add_argument("--discard", action="store_true", help="read the columns with the discard applied")
    args = ap.parse_args(argv)
    with open(args.table, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        result = from_rows(rows, "_discard" if args.discard else "", bound_pp_per_10um=args.bound)
    except ValueError as exc:
        print(f"refused: {exc}")
        return 2
    print("\n".join(result.lines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
