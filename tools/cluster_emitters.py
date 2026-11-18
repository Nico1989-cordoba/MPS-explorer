#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emitters (BaGoL-like reimplementation): localizations -> emitters by RJMCMC.

This is a Python reimplementation of the Bayesian Grouping of Localizations
(BaGoL) model of Fazel et al. (2022), "High-precision estimation of emitter
positions using Bayesian grouping of localizations", Nat Commun 13:7152,
doi:10.1038/s41467-022-34894-2. It is NOT the authors' code; the public label
is "Emitters (BaGoL-like reimplementation)". The MATLAB code of the authors
(github.com/LidkeLab/BaGoL) was read as a specification and is distributed
under the following licence, reproduced as it requires:

    MIT License

    Copyright (c) 2020 LidkeLab

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Pipeline (as in ``BaGoL.analyze_all``)
--------------------------------------
1. Subregions: square tiles of ``roi_size_nm`` (200 nm) extended by an
   overlap on every side [code: BaGoL.m defaults, genROIs.m]. The overlap is
   max(``overlap_nm``, ``overlap_sigma_factor`` * sigma_95), with
   ``overlap_nm`` = 20 nm (the code's fixed default) and
   ``overlap_sigma_factor`` = 4 (deviation, see "Subregion overlap" below).
2. Pre-clusters: single linkage cut at ``cutoff_nm`` (default = roi_size_nm)
   inside each subregion [code: precluster.m].
3. RJMCMC per pre-cluster with four equiprobable jumps Move / Allocate /
   Birth / Death [paper, Methods: (0.25, 0.25, 0.25, 0.25)], ``n_burnin``
   2000 and ``n_trials`` 3000 [code: BaGoL.m defaults].
4. MAPN: the most frequent K of the post-burn-in chain (ties -> the smallest);
   emitter positions = k-means centres of the pooled positions of the states
   with that K, uncertainties = their standard deviation [code: genMAPN.m].
5. Overlap removal: an emitter is kept only by the subregion whose core tile
   (without the overlap) contains it [code: removeOverlap.m].

Subregion overlap [derivation; deviation from the code's fixed 20 nm]
--------------------------------------------------------------------
The overlap removal only works when both subregions that contain an emitter
near their shared border see *all* of its localizations: then both estimate
the same position and exactly one keeps it. When the overlap is narrower
than the spread of the localizations, the subregion on the far side sees the
emitter truncated, its estimate is pulled towards its own core, and both
subregions keep it (a duplicate) or neither does (a loss). With sigma = 12 nm
(dSTORM) and the code's 20-nm overlap, an emitter within 6 nm of a border is
duplicated in about 19 % of synthetic data sets (independent review of F3;
0 % without borders). The original's 20 nm was chosen for DNA-PAINT
precisions of a few nm, where 20 nm is > 4 sigma.

For an emitter on the border, a localization falls outside an overlap o with
probability Phi(-o / sigma) per axis; for N localizations the chance that at
least one is cut is about N * Phi(-o / sigma): with o = 3 sigma that is 1.9 %
for N = 14 and 6.5 % for N = 50; with o = 4 sigma, 0.04 % and 0.16 %. Hence
o = max(20 nm, 4 * sigma_95), with sigma_95 the 95th percentile of the
per-localization precisions max(sigma_x, sigma_y) [derivation; the 95th
percentile is ad hoc, so that a tail of poor localizations does not inflate
the cost]. ``overlap_sigma_factor=0`` restores the code's fixed overlap.
Measured on the review's case (sigma 12 nm, lambda 14, 96 data sets): see the
F3 report, session 2. Cost: a wider overlap puts more localizations in each
subregion (about ((200 + 2 o) / 240)^2 times more in a dense field).

Allocation prior [derivation; deviation from the code, measured in F4]
----------------------------------------------------------------------
The code allocates localizations to emitters with p(Z | K) = K^-N (every
localization picks one of the K emitters uniformly). That is the exact
conditional p(Z | N, K, xi) only when xi is Poisson: N_k ~ Poisson(lambda)
i.i.d. gives N ~ Poisson(K lambda) and a uniform multinomial allocation. For
the gamma xi the paper recommends for dSTORM (eta = 1, exponential blinking
counts) the counts per emitter are i.i.d. negative binomial NB(k, p) (the
Poisson-gamma mixture) and the exact conditional is the Dirichlet-multinomial
p(Z | N, K) = Gamma(K k) / Gamma(N + K k) * prod_j Gamma(N_j + k) / Gamma(k):
a mixture with weights w ~ Dirichlet(k, ..., k), not equal weights. The
target sampled with ``allocation='dirichlet'`` is a HYBRID: this
Dirichlet-multinomial allocation (exact for negative-binomial counts) times
the paper's continuous p(N | K) = Gamma(N; K k, theta) (eq. 7), not the
negative-binomial p(N | K) of the same model. Against the exact posterior of
that model, by enumeration of partitions, the hybrid's residual in the number
of occupied emitters is a total variation of 0.009-0.058 (0.017-0.032 with
theta = 14), against 0.048-0.234 for the uniform allocation; a negative-
binomial p(N | K) changed no metric on synthetic dense fields (independent
review of F4). With equal
weights and heavy-tailed counts, a second emitter placed on a bright emitter
with N_b localizations gains 2^N_b (those localizations get weight 2/K instead
of 1/K) against a cost of about exp(-N/K) and 1/(A q): in a dense pre-cluster
every emitter with N_b > ~2 x mean(xi) is split into ~N_b / mean(xi) pieces
(F4 part A, 720 emitters, exponential counts of mean 14, sigma 12 nm: K per
true emitter 1.0, 1.4, 1.7, 2.1, 2.7, 3.0 for N_b 8-14, 15-21, 22-28, 29-35,
36-49, >= 50; 16-22 % emitters in excess; with Poisson counts and a Poisson
xi the excess is -3 %). ``allocation='dirichlet'`` (the default for a gamma
xi) keeps an explicit weight vector w: Allocate draws z_i ∝ w_j phi_ij; Move
also draws w | Z ~ Dirichlet(k + N_j); Birth draws w* ~ Beta(k, K k) and
rescales w_j <- w_j (1 - w*); Death is the inverse. The Dirichlet prior
ratio, the Beta proposal density and the Jacobian (1 - w*)^(K-1) cancel
exactly (Richardson & Green 1997, J R Stat Soc B 59:731, birth of a mixture
component), so the acceptance ratios below keep their form with L computed
with the weights; the chain matches the exact posterior by enumeration of
partitions (test). With the hierarchical xi two approximations remain:
the update of the shape k ignores the Dirichlet term prod_r Dir(w_r; k) of
the exact conditional, and theta (like k) is drawn from p(N_r | K_r, xi) with
the counts described under 'Hierarchical xi' below (kept emitters with
localizations), which are not the (N, K) of the chain, whose K also
counts empty emitters. Both were measured benign on synthetic fields (the
learned shape returns to ~1 from a start at k = 2; independent review of
F4). The Dirichlet term is not added: with the occupied-only counts it would
mix two different K, and a small k makes births of near-empty components
more likely, which would feed back into a smaller k. A Poisson xi keeps the
uniform allocation (exact). ``allocation='uniform'`` restores the code's
prior; ``matlab_compat`` implies it. ``prune_empty`` with the Dirichlet
allocation warns: the deterministic pruning deletes the empty emitters that
the Dirichlet prior makes frequent and biases K down (-7 to -12 % on
synthetic dense fields, independent review of F4).

Model and acceptance ratios (derivation in this docstring; the full version
is in the F3 notes of the project)
------------------------------------------------------------------------
State (K, mu_1..mu_K, Z). Target, for one pre-cluster with N localizations
x_i of precisions (sx_i, sy_i) and a given xi::

    pi(K, mu, Z | X) ∝ p(N | K, xi) * prod_k u_S(mu_k) * (1/K)^N * prod_i phi(x_i; mu_{z_i}, sigma_i)

* ``p(N | K, xi)`` = Poisson(N; K lambda) or Gamma(N; K k, theta) [code:
  BaGoL_RJMCMC.m, ``poisspdf(N,(1:LengN)*Xi)`` / ``gampdf(N,(1:LengN)*k,theta)``],
  with a flat prior on K in 1..max(N, 100). For Poisson this is exact: N_k
  ~ Poisson(lambda) per emitter gives Poisson(N; K lambda) times a uniform
  multinomial allocation, (1/K)^N for labelled localizations [derivation].
* ``u_S(mu) = 1/A`` on the support S (1-nm cells where the kernel density of
  the localizations exceeds max/1000, area A) [code: ``Area``], 0 outside
  [derivation: the code uses 1/Area untruncated; truncating makes the
  conditional of an empty emitter proper (uniform on S)].
* Marginalising Z: pi(K, mu | X) ∝ p(N|K) prod_k u_S(mu_k) L_K(mu) with
  L_K = prod_i (1/K) sum_k phi(x_i; mu_k, sigma_i).

Birth K -> K+1: mu* ~ q, the continuous kernel density of the localizations
q(mu) = (1/N) sum_i phi(mu; x_i, sigma_i) (the code's 1-nm grid version of
it), inserted at one of K+1 positions; the reverse death picks it with
1/(K+1), which cancels. Z is redrawn from its full conditional after an
accepted jump, so its proposal cancels and the ratio uses the marginal L.
The map (mu, u) -> (mu, mu* = u) has Jacobian 1. Hence::

    A_birth = min{1, p(N|K+1)/p(N|K) * L'/L * 1[mu* in S] * P_death / (P_birth * A * q(mu*))}
    A_death = min{1, p(N|K-1)/p(N|K) * L'/L * P_birth * A * q(mu_j) / P_death}

The birth term equals the code's ``A = PR*AllocR/(Area*PDFgrid(ID))``. The
code's death omits ``A * q(mu_j)`` (an error: the paper's own Supplementary
Note 1, eq. 27, asks for the exact inverse of the birth ratio); the exact
reverse is used here. The code also prunes empty emitters deterministically
at every step (not a reversible move); here they are kept (their mu is drawn
uniform on S and they can die through Death). ``matlab_compat=True`` (both
deviations together, ``legacy_death_ratio`` + ``prune_empty``, plus the
code's fixed overlap) reproduces the original code for comparisons only;
either flag alone is not the original code.

Move: Gibbs of each mu_k given Z, N(sum w x / sum w, 1/sum w) per axis with
w = 1/sigma^2 (flat prior) [code: Gibbs_Mu], truncated to S by an
independence Metropolis step (propose untruncated, accept if in S).
Allocate: Gibbs of Z with prior 1/K [code: Gibbs_Z], drawn with the Gumbel-max
trick on log probabilities.

Hierarchical xi (``hierarchical=True``) [code: BaGoL.m HierarchFlag branch]:
every ``n_samples`` (10) RJMCMC steps of every pre-cluster, xi is redrawn
once, globally, from p(xi) prod_r p(N_r | K_r, xi), with (N_r, K_r) the
localizations assigned to the kept emitters and the number of kept emitters
of pre-cluster r. Which emitters count (``xi_counts``): ``'nonsingleton'``
(default) only those with >= 2 localizations in the current state, and a
pre-cluster of one localization counts nothing; ``'occupied'`` every kept
emitter with >= 1 localization (the code's K_r, which prunes empty
emitters; ``matlab_compat`` implies it). Reason [simulation, this project,
F4 part B]: an isolated background localization is a legitimate
one-localization emitter of the model, and with ~90 of them beside ~70 real
emitters of ~40 events the learned xi was the mixture (mean diluted, shape
k 0.34-0.55), not the events of a label. Cost [derivation]: for an
exponential (geometric) xi, excluding the one-localization emitters adds
about +1 to the learned mean (E[N | N >= 2] = mean + 1, memorylessness), so
+7 % at a mean of 14 and +20 % at 5; real emitters of one event are lost to
xi as well (P(N = 1) = 7 % at a mean of 14). Hyperprior Gamma(shape alpha = 1, scale beta = 50) [code:
Alpha_Xi, Beta_Xi; MATLAB gampdf convention: beta is a scale]. Poisson:
conjugate draw lambda ~ Gamma(alpha + sum N_r, scale 1/(1/beta + sum K_r))
[derivation; the code runs 10 MH steps with the same stationary law]. Gamma
(eta, gamma): componentwise MH with the code's proposal Gamma(3000, x/3000)
and Hastings correction, 10 steps [code: sampleGam.m], with exact log
densities (a zero-density observation rejects; the code drops it).

Drift (``drift`` > 0, nm per frame, needs ``frames``): mu_k + alpha_k t with
alpha ~ N(0, drift^2) [code: Gibbs_MuAlpha]; off by default (the code's
``Drift = 0``). t is centred per pre-cluster, so mu_k is the position at the
mean frame. Not implemented (deviations, listed in the F3 report): intensity and NND filters
(optional in the code; NND "must not be used for dSTORM"), posterior images.
The axial coordinate is not modelled (decision D-B6): an emitter's z is the
mean of its localizations' z weighted by their posterior membership (and by
1/sigma_z^2 when given).
"""
from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_cancel
from tools.clustering_strategies import ClusteringStrategy

PUBLIC_LABEL = "Emitters (BaGoL-like reimplementation)"
CITATION = ("Fazel et al. (2022) Nat Commun 13:7152 (BaGoL); reimplementation of the MIT-licensed "
            "LidkeLab/BaGoL model, not the authors' code")

# Defaults [code: BaGoL.m property initialisers]
ROI_SIZE_NM = 200.0
OVERLAP_NM = 20.0             # floor of the overlap [code: BaGoL.m default Overlap]
OVERLAP_SIGMA_FACTOR = 4.0    # overlap >= 4 sigma_95 [derivation: module docstring, "Subregion overlap"]
OVERLAP_SIGMA_PERCENTILE = 95.0  # [ad hoc: robust to a tail of poor localizations]
N_BURNIN = 2000
N_TRIALS = 3000
N_SAMPLES_XI = 10
ALPHA_XI = 1.0
BETA_XI = 50.0
P_JUMPS = (0.25, 0.25, 0.25, 0.25)
GRID_NM = 1.0                 # [code: BaGoL_RJMCMC.m, 1-nm grid]
SUPPORT_REL_THRESHOLD = 1e-3  # [code: Area = cells with density > max/1000]
KMAX_FLOOR = 100              # [code: LengN = max(N, 100)]
GAMMA_PROPOSAL_SHAPE = 3000.0  # [code: sampleGam.m / samplePoiss.m Alpha_Prop]
GAMMA_MH_STEPS = 10           # [code: sampleGam.m, 10 steps per call]
DEFAULT_PROCESSES = 2         # [ad hoc: this laptop's budget]
# An emitter "has localizations" when n_locs >= HAS_LOCS_MIN (editable: ``has_locs_min``) [simulation, this
# project: F3 session 3, lambda 3-14, sigma 12 nm; see the EmitterResult docstring]
HAS_LOCS_MIN = 0.5
ORPHAN_N_SIGMA = 3.0          # orphan -> nearest emitter within 3 combined sigmas [user decision F3 (b)]
# Allocation prior for a gamma xi: 'dirichlet' (exact for gamma / negative-binomial counts) or 'uniform' (the
# code's K^-N, exact only for a Poisson xi) [derivation: module docstring 'Allocation prior'; measured in F4]
ALLOCATION = "dirichlet"
ALLOCATIONS = ("dirichlet", "uniform")
# Which emitters the hierarchical xi counts (module docstring, 'Hierarchical xi'): 'nonsingleton' (>= 2
# localizations; decision of the user 2026-10-17, F4 (d)) or 'occupied' (>= 1, the code's K_r)
XI_COUNTS = "nonsingleton"
XI_COUNTS_CHOICES = ("nonsingleton", "occupied")
# An emitter is a "non-singleton" when the posterior mean of the localizations it explains is >= 1.5 (rounded, at
# least two) [simulation, this project: F4 part B, the estimates compared with the truth per epitope]
NONSINGLETON_MIN = 1.5

_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Prior on the number of localizations per emitter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class XiPrior:
    """Distribution of the number of localizations per emitter (xi).

    ``kind='poisson'``: Poisson(lam). ``kind='gamma'``: Gamma(shape k, scale
    theta), mean k * theta (the MATLAB ``gampdf`` convention).
    """

    kind: str
    lam: float = 0.0
    k: float = 0.0
    theta: float = 0.0

    @staticmethod
    def poisson(lam: float) -> "XiPrior":
        if not lam > 0:
            raise ValueError("Poisson xi needs lam > 0")
        return XiPrior("poisson", lam=float(lam))

    @staticmethod
    def gamma(k: float, theta: float) -> "XiPrior":
        if not (k > 0 and theta > 0):
            raise ValueError("Gamma xi needs k > 0 and theta > 0")
        return XiPrior("gamma", k=float(k), theta=float(theta))

    @property
    def mean(self) -> float:
        return self.lam if self.kind == "poisson" else self.k * self.theta

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.lam,) if self.kind == "poisson" else (self.k, self.theta)

    def log_pN(self, n: int, kk: int) -> float:
        """log p(N = n | K = kk, xi) (up to nothing: exact log density)."""
        if self.kind == "poisson":
            mu = kk * self.lam
            return n * math.log(mu) - mu - math.lgamma(n + 1.0)
        a = kk * self.k
        return (a - 1.0) * math.log(n) - n / self.theta - math.lgamma(a) - a * math.log(self.theta)


def _gamma_logpdf(x: float, shape: float, scale: float) -> float:
    if x <= 0 or shape <= 0 or scale <= 0:
        return -math.inf
    return (shape - 1.0) * math.log(x) - x / scale - math.lgamma(shape) - shape * math.log(scale)


def sample_xi(xi: XiPrior, n_points: Sequence[int], n_emitters: Sequence[int], rng: np.random.Generator,
              alpha: float = ALPHA_XI, beta: float = BETA_XI) -> XiPrior:
    """One update of the global xi given (N_r, K_r) of every pre-cluster (see the module docstring).

    Pre-clusters with N_r = 0 or K_r = 0 carry no information and are skipped [code: ``Ind = NPoints > 0``].
    """
    n = np.asarray(n_points, dtype=np.float64)
    kk = np.asarray(n_emitters, dtype=np.float64)
    ok = (n > 0) & (kk > 0)
    n, kk = n[ok], kk[ok]
    if xi.kind == "poisson":
        shape = alpha + float(n.sum())
        scale = 1.0 / (1.0 / beta + float(kk.sum()))
        return XiPrior.poisson(float(rng.gamma(shape, scale)))

    def loglik(k: float, theta: float) -> float:
        tot = 0.0
        for ni, ki in zip(n, kk):
            v = _gamma_logpdf(float(ni), float(ki) * k, theta)
            if not math.isfinite(v):
                return -math.inf
            tot += v
        return tot

    k, theta = xi.k, xi.theta
    cur = loglik(k, theta)
    s = GAMMA_PROPOSAL_SHAPE
    for _ in range(GAMMA_MH_STEPS):
        for comp in (0, 1):
            x = k if comp == 0 else theta
            xp = float(rng.gamma(s, x / s))
            kp, tp = (xp, theta) if comp == 0 else (k, xp)
            new = loglik(kp, tp)
            log_r = (new - cur
                     + _gamma_logpdf(xp, alpha, beta) - _gamma_logpdf(x, alpha, beta)
                     + _gamma_logpdf(x, s, xp / s) - _gamma_logpdf(xp, s, x / s))
            if math.isfinite(log_r) and math.log(rng.random()) < log_r:
                k, theta, cur = kp, tp, new
    return XiPrior.gamma(k, theta)


# ---------------------------------------------------------------------------
# One pre-cluster: the RJMCMC chain
# ---------------------------------------------------------------------------

@dataclass
class ChainConfig:
    n_burnin: int = N_BURNIN
    n_trials: int = N_TRIALS
    p_jumps: Tuple[float, float, float, float] = P_JUMPS
    grid_nm: float = GRID_NM
    legacy_death_ratio: bool = False
    prune_empty: bool = False
    drift_sigma: float = 0.0      # prior sd of the per-emitter drift, nm per frame (0 = no drift)
    allocation: str = ALLOCATION  # 'dirichlet' or 'uniform' (module docstring, 'Allocation prior')
    mapn_occupied: bool = True    # MAPN over the emitters with localizations (see EmitterChain.finalize)
    xi_counts: str = XI_COUNTS    # which emitters the hierarchical xi counts (module docstring)

    @classmethod
    def matlab_compat(cls, **kw: Any) -> "ChainConfig":
        """The chain of the original MATLAB code (``legacy_death_ratio`` and ``prune_empty`` together), for
        comparisons with it only; see :func:`fit_emitters` (``matlab_compat``) for why it is biased."""
        kw.update(legacy_death_ratio=True, prune_empty=True, allocation="uniform", xi_counts="occupied")
        return cls(**kw)


@dataclass
class PreclusterResult:
    """MAPN output of one pre-cluster (positions in nm, labels local to the pre-cluster)."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    sx: NDArray[np.float64]
    sy: NDArray[np.float64]
    z: NDArray[np.float64]
    n_mean: NDArray[np.float64]
    kept: NDArray[np.bool_]
    loc_label: NDArray[np.int64]          # local emitter index per localization (-1 none)
    k_mapn: int
    p_k_mapn: float
    k_mean: float
    k_hist: Dict[int, int]
    accept: Dict[str, float]
    alpha_x: Optional[NDArray[np.float64]] = None
    alpha_y: Optional[NDArray[np.float64]] = None
    # Emitters with >= 1 localization in each post-burn-in state: K also counts empty emitters (the model allows
    # them; with prune_empty off they are not deleted), whose position is just uniform on S.
    k_occ_mean: float = float("nan")
    k_occ_hist: Dict[int, int] = field(default_factory=dict)
    # Per MAPN emitter: fraction of the K = K_MAPN states in which it has >= 1 localization
    p_occ: Optional[NDArray[np.float64]] = None


class EmitterChain:
    """RJMCMC sampler of one pre-cluster (x, y, sx, sy in nm).

    ``core`` = (x0, x1, y0, y1, close_x, close_y): the core tile used by the overlap removal; an emitter is kept
    when x0 <= x < x1 (or <= x1 when ``close_x``) and likewise in y.
    """

    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64], sx: NDArray[np.float64],
                 sy: NDArray[np.float64], seed: Any, cfg: ChainConfig, xi0: XiPrior,
                 core: Optional[Tuple[float, float, float, float, bool, bool]] = None,
                 z: Optional[NDArray[np.float64]] = None, sz: Optional[NDArray[np.float64]] = None,
                 t: Optional[NDArray[np.float64]] = None) -> None:
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.sx = np.ascontiguousarray(sx, dtype=np.float64)
        self.sy = np.ascontiguousarray(sy, dtype=np.float64)
        self.zc = None if z is None else np.asarray(z, dtype=np.float64)
        self.szc = None if sz is None else np.asarray(sz, dtype=np.float64)
        self.n = int(self.x.size)
        if self.n < 1:
            raise ValueError("a pre-cluster needs at least one localization")
        self.cfg = cfg
        # Drift: mu_k is the position at the pre-cluster's mean frame (t centred) [derivation: better conditioned
        # than the code's frame 0; the model is the same linear drift mu + alpha t].
        self.drift = cfg.drift_sigma > 0 and t is not None
        tt = np.zeros(self.n) if t is None else np.asarray(t, dtype=np.float64)
        self.t = tt - tt.mean() if self.drift else np.zeros(self.n)
        self.core = core
        self.rng = np.random.default_rng(seed)
        self.wx = 1.0 / self.sx ** 2
        self.wy = 1.0 / self.sy ** 2
        self.lognorm = -_LOG_2PI - np.log(self.sx) - np.log(self.sy)
        self.kmax = max(self.n, KMAX_FLOOR)
        pj = np.asarray(cfg.p_jumps, dtype=np.float64)
        self.cum_jumps = np.cumsum(pj / pj.sum())
        self.log_jump_ratio = math.log(pj[3] / pj[2])   # log(P_death / P_birth)
        self._build_support()
        # Initial state [code: K0 = ceil(N / mean xi); mu at random localizations]
        k0 = int(min(self.kmax, max(1, math.ceil(self.n / xi0.mean))))
        # Allocation weights: Dirichlet(k) for a gamma xi (module docstring, 'Allocation prior'); None = 1/K
        self.alpha_w: Optional[float] = (float(xi0.k) if (cfg.allocation == "dirichlet" and xi0.kind == "gamma")
                                         else None)
        self.logw: Optional[NDArray[np.float64]] = (np.full(k0, -math.log(k0)) if self.alpha_w is not None
                                                    else None)
        pick = self.rng.integers(0, self.n, size=k0)
        self.mux = self.x[pick].copy()
        self.muy = self.y[pick].copy()
        self.axv: NDArray[np.float64] = np.zeros(k0)                         # drift starts at 0 [code]
        self.ayv: NDArray[np.float64] = np.zeros(k0)
        self.logp = self._logphi(self.mux, self.muy, self.axv, self.ayv)
        self.ll = self._loglik(self.logp)
        self.zlab = self._gibbs_z(self.logp)
        self.step = 0
        self.k_rec: List[int] = []
        self.mux_rec: List[NDArray[np.float64]] = []
        self.muy_rec: List[NDArray[np.float64]] = []
        self.ax_rec: List[NDArray[np.float64]] = []
        self.ay_rec: List[NDArray[np.float64]] = []
        self.z_rec: List[NDArray[np.int32]] = []
        self.n_prop = np.zeros(4, dtype=np.int64)
        self.n_acc = np.zeros(4, dtype=np.int64)

    # -- support S, its area A and the birth proposal q -------------------------------------------------
    def _build_support(self) -> None:
        g = float(self.cfg.grid_nm)
        x0 = float(np.min(self.x - 3.0 * self.sx))
        x1 = float(np.max(self.x + 3.0 * self.sx))
        y0 = float(np.min(self.y - 3.0 * self.sy))
        y1 = float(np.max(self.y + 3.0 * self.sy))
        nx = int(math.ceil((x1 - x0) / g)) + 1
        ny = int(math.ceil((y1 - y0) / g)) + 1
        gx = x0 + (np.arange(nx) + 0.5) * g
        gy = y0 + (np.arange(ny) + 0.5) * g
        phx = np.exp(-0.5 * ((gx[None, :] - self.x[:, None]) / self.sx[:, None]) ** 2) / self.sx[:, None]
        phy = np.exp(-0.5 * ((gy[None, :] - self.y[:, None]) / self.sy[:, None]) ** 2) / self.sy[:, None]
        dens = phy.T @ phx                        # (ny, nx), unnormalised kernel density
        self.gx0, self.gy0, self.gnx, self.gny, self.g = x0, y0, nx, ny, g
        self.smask = dens > dens.max() * SUPPORT_REL_THRESHOLD
        self.scells = np.flatnonzero(self.smask.ravel())
        self.area = float(self.scells.size) * g * g

    def in_support(self, mx: float, my: float) -> bool:
        ix = int(math.floor((mx - self.gx0) / self.g))
        iy = int(math.floor((my - self.gy0) / self.g))
        if ix < 0 or iy < 0 or ix >= self.gnx or iy >= self.gny:
            return False
        return bool(self.smask[iy, ix])

    def log_q_density(self, mx: float, my: float) -> float:
        """log of the kernel density of the localizations at (mx, my), per nm^2, by log-sum-exp: finite even
        far (in units of sigma) from every localization, e.g. with DNA-PAINT precisions of ~1 nm."""
        lp = self.lognorm - 0.5 * ((self.x - mx) ** 2 * self.wx + (self.y - my) ** 2 * self.wy)
        m = float(lp.max())
        return m + math.log(float(np.exp(lp - m).sum())) - math.log(self.n)

    def q_density(self, mx: float, my: float) -> float:
        """Kernel density of the localizations at (mx, my), per nm^2 (integrates to 1 over the plane)."""
        return math.exp(self.log_q_density(mx, my))

    def _uniform_in_support(self) -> Tuple[float, float]:
        c = int(self.scells[self.rng.integers(0, self.scells.size)])
        iy, ix = divmod(c, self.gnx)
        u = self.rng.random(2)
        return self.gx0 + (ix + u[0]) * self.g, self.gy0 + (iy + u[1]) * self.g

    # -- likelihood pieces --------------------------------------------------------------------------------
    def _logphi(self, mx: NDArray[np.float64], my: NDArray[np.float64], ax: Optional[NDArray[np.float64]] = None,
                ay: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        dx = self.x[:, None] - mx[None, :]
        dy = self.y[:, None] - my[None, :]
        if self.drift and ax is not None and ay is not None:
            dx = dx - self.t[:, None] * ax[None, :]
            dy = dy - self.t[:, None] * ay[None, :]
        return np.asarray(self.lognorm[:, None] - 0.5 * (dx * dx * self.wx[:, None] + dy * dy * self.wy[:, None]))

    def _loglik(self, logp: NDArray[np.float64]) -> float:
        """log L_K = sum_i log(sum_k w_k phi_ik), with w_k = 1/K (uniform allocation) or the current weights."""
        if self.logw is not None:
            return self._loglik_w(logp, self.logw)
        kk = logp.shape[1]
        m = logp.max(axis=1)
        s = np.log(np.exp(logp - m[:, None]).sum(axis=1)) + m
        return float(s.sum() - self.n * math.log(kk))

    def _loglik_w(self, logp: NDArray[np.float64], logw: NDArray[np.float64]) -> float:
        lw = logp + logw[None, :]
        m = lw.max(axis=1)
        return float((np.log(np.exp(lw - m[:, None]).sum(axis=1)) + m).sum())

    def _gibbs_z(self, logp: NDArray[np.float64]) -> NDArray[np.int32]:
        gum = -np.log(-np.log(self.rng.random(logp.shape)))
        if self.logw is not None:
            return np.asarray(np.argmax(logp + self.logw[None, :] + gum, axis=1), dtype=np.int32)
        return np.asarray(np.argmax(logp + gum, axis=1), dtype=np.int32)

    def _gibbs_w(self) -> None:
        """w | Z ~ Dirichlet(alpha + N_1, ..., alpha + N_K) [derivation: conjugate update]."""
        assert self.alpha_w is not None
        cnt = np.bincount(self.zlab, minlength=self.mux.size).astype(np.float64)
        w = self.rng.gamma(self.alpha_w + cnt)
        self.logw = np.log(np.maximum(w / w.sum(), 1e-300))

    # -- jumps --------------------------------------------------------------------------------------------
    def _draw_alpha(self) -> Tuple[float, float]:
        if not self.drift:
            return 0.0, 0.0
        e = self.rng.standard_normal(2) * self.cfg.drift_sigma
        return float(e[0]), float(e[1])

    def _move_drift(self) -> None:
        """Gibbs of (mu, alpha) per emitter and axis: precision [[sum w, sum w t], [sum w t, sum w t^2 + 1/s^2]]
        (flat prior on mu, N(0, s^2) on alpha) [code: Gibbs_MuAlpha], truncated to S as in :meth:`_move`."""
        kk = self.mux.size
        inv_s2 = 1.0 / self.cfg.drift_sigma ** 2
        cnt = np.bincount(self.zlab, minlength=kk)
        stats = []
        for w, v in ((self.wx, self.x), (self.wy, self.y)):
            a = np.bincount(self.zlab, weights=w, minlength=kk)
            b = np.bincount(self.zlab, weights=w * self.t, minlength=kk)
            d = np.bincount(self.zlab, weights=w * self.t * self.t, minlength=kk) + inv_s2
            r1 = np.bincount(self.zlab, weights=w * v, minlength=kk)
            r2 = np.bincount(self.zlab, weights=w * v * self.t, minlength=kk)
            stats.append((a, b, d, r1, r2))
        for k in range(kk):
            if cnt[k] == 0:
                self.mux[k], self.muy[k] = self._uniform_in_support()
                self.axv[k], self.ayv[k] = self._draw_alpha()
                continue
            draw = []
            for a, b, d, r1, r2 in stats:
                prec = np.array([[a[k], b[k]], [b[k], d[k]]])
                cov = np.linalg.inv(prec)
                mean = cov @ np.array([r1[k], r2[k]])
                draw.append(mean + np.linalg.cholesky(cov) @ self.rng.standard_normal(2))
            if self.in_support(float(draw[0][0]), float(draw[1][0])):
                self.mux[k], self.axv[k] = draw[0]
                self.muy[k], self.ayv[k] = draw[1]
        self.logp = self._logphi(self.mux, self.muy, self.axv, self.ayv)
        self.ll = self._loglik(self.logp)

    def _move(self) -> None:
        if self.drift:
            self._move_drift()
        else:
            self._move_mu()
        if self.logw is not None:
            self._gibbs_w()
            self.ll = self._loglik(self.logp)

    def _move_mu(self) -> None:
        kk = self.mux.size
        sw_x = np.bincount(self.zlab, weights=self.wx, minlength=kk)
        sw_y = np.bincount(self.zlab, weights=self.wy, minlength=kk)
        sx_ = np.bincount(self.zlab, weights=self.wx * self.x, minlength=kk)
        sy_ = np.bincount(self.zlab, weights=self.wy * self.y, minlength=kk)
        cnt = np.bincount(self.zlab, minlength=kk)
        for k in range(kk):
            if cnt[k] == 0:
                self.mux[k], self.muy[k] = self._uniform_in_support()
                continue
            px = sx_[k] / sw_x[k] + self.rng.standard_normal() / math.sqrt(sw_x[k])
            py = sy_[k] / sw_y[k] + self.rng.standard_normal() / math.sqrt(sw_y[k])
            if self.in_support(px, py):
                self.mux[k], self.muy[k] = px, py
        self.logp = self._logphi(self.mux, self.muy, self.axv, self.ayv)
        self.ll = self._loglik(self.logp)

    def _birth(self, xi: XiPrior) -> bool:
        kk = self.mux.size
        if kk + 1 > self.kmax:
            return False
        i = int(self.rng.integers(0, self.n))
        e = self.rng.standard_normal(2)
        mx = float(self.x[i] + self.sx[i] * e[0])
        my = float(self.y[i] + self.sy[i] * e[1])
        if not self.in_support(mx, my):
            return False
        # The new drift is drawn from its prior, so its density cancels the prior term [derivation; code: same].
        bx, by = self._draw_alpha()
        col = self._logphi(np.array([mx]), np.array([my]), np.array([bx]), np.array([by]))
        logp_new = np.hstack([self.logp, col])
        logw_new: Optional[NDArray[np.float64]] = None
        if self.logw is not None:
            # w* ~ Beta(k, K k) (the marginal of one component of Dirichlet_{K+1}(k)), w_j <- w_j (1 - w*): the
            # Dirichlet ratio, the proposal density and the Jacobian (1 - w*)^(K-1) cancel [Richardson & Green 1997]
            assert self.alpha_w is not None
            ws = min(max(float(self.rng.beta(self.alpha_w, kk * self.alpha_w)), 1e-300), 1.0 - 1e-16)
            logw_new = np.append(self.logw + math.log1p(-ws), math.log(ws))
            ll_new = self._loglik_w(logp_new, logw_new)
        else:
            ll_new = self._loglik(logp_new)
        log_a = (xi.log_pN(self.n, kk + 1) - xi.log_pN(self.n, kk) + ll_new - self.ll
                 + self.log_jump_ratio - math.log(self.area) - self.log_q_density(mx, my))
        if math.log(self.rng.random()) < log_a or not math.isfinite(self.ll):
            self.mux = np.append(self.mux, mx)
            self.muy = np.append(self.muy, my)
            self.axv = np.append(self.axv, bx)
            self.ayv = np.append(self.ayv, by)
            if logw_new is not None:
                self.logw = logw_new
            self.logp, self.ll = logp_new, ll_new
            self.zlab = self._gibbs_z(self.logp)
            return True
        return False

    def _death(self, xi: XiPrior) -> bool:
        kk = self.mux.size
        j = int(self.rng.integers(0, kk))
        logp_new = np.delete(self.logp, j, axis=1)
        logw_new: Optional[NDArray[np.float64]] = None
        if self.logw is not None:
            wj = math.exp(float(self.logw[j]))
            if wj >= 1.0 - 1e-12:
                return False
            logw_new = np.delete(self.logw, j) - math.log1p(-wj)       # inverse of the birth rescaling
            ll_new = self._loglik_w(logp_new, logw_new)
        else:
            ll_new = self._loglik(logp_new)
        log_a = xi.log_pN(self.n, kk - 1) - xi.log_pN(self.n, kk) + ll_new - self.ll - self.log_jump_ratio
        if not self.cfg.legacy_death_ratio:
            log_a += math.log(self.area) + self.log_q_density(float(self.mux[j]), float(self.muy[j]))
        if math.log(self.rng.random()) < log_a:
            self.mux = np.delete(self.mux, j)
            self.muy = np.delete(self.muy, j)
            self.axv = np.delete(self.axv, j)
            self.ayv = np.delete(self.ayv, j)
            if logw_new is not None:
                self.logw = logw_new
            self.logp, self.ll = logp_new, ll_new
            self.zlab = self._gibbs_z(self.logp)
            return True
        return False

    def _prune(self) -> None:
        cnt = np.bincount(self.zlab, minlength=self.mux.size)
        keep = cnt > 0
        if keep.all():
            return
        remap = np.cumsum(keep) - 1
        self.mux, self.muy = self.mux[keep], self.muy[keep]
        self.axv, self.ayv = self.axv[keep], self.ayv[keep]
        if self.logw is not None:
            w = np.exp(self.logw[keep])
            self.logw = np.log(np.maximum(w / w.sum(), 1e-300))
        self.zlab = np.asarray(remap[self.zlab], dtype=np.int32)
        self.logp = self.logp[:, keep]
        self.ll = self._loglik(self.logp)

    def run(self, n_steps: int, xi: XiPrior) -> None:
        """``n_steps`` RJMCMC jumps with ``xi``; states after ``n_burnin`` total steps are recorded."""
        if self.alpha_w is not None:
            self.alpha_w = float(xi.k)            # the Dirichlet concentration follows the (hierarchical) shape
        for _ in range(int(n_steps)):
            if self.cfg.prune_empty:
                self._prune()
            jump = int(np.searchsorted(self.cum_jumps, self.rng.random(), side="right"))
            jump = min(jump, 3)
            if jump == 0:
                self._move()
                acc = True
            elif jump == 1:
                self.zlab = self._gibbs_z(self.logp)
                acc = True
            elif jump == 2:
                acc = self._birth(xi)
            elif self.mux.size > 1:
                acc = self._death(xi)
            else:
                acc = False
                jump = -1     # death not proposable at K = 1: the state is kept [code: K==1 -> skip]
            if jump >= 0:
                self.n_prop[jump] += 1
                self.n_acc[jump] += int(acc)
            self.step += 1
            if self.step > self.cfg.n_burnin:
                self.k_rec.append(int(self.mux.size))
                self.mux_rec.append(self.mux.copy())
                self.muy_rec.append(self.muy.copy())
                if self.drift:
                    self.ax_rec.append(self.axv.copy())
                    self.ay_rec.append(self.ayv.copy())
                self.z_rec.append(self.zlab.copy())

    # -- overlap removal and the counts for the xi update ------------------------------------------------
    def kept_mask(self, mx: NDArray[np.float64], my: NDArray[np.float64]) -> NDArray[np.bool_]:
        if self.core is None:
            return np.ones(mx.size, dtype=bool)
        x0, x1, y0, y1, cx, cy = self.core
        okx = (mx >= x0) & ((mx <= x1) if cx else (mx < x1))
        oky = (my >= y0) & ((my <= y1) if cy else (my < y1))
        return np.asarray(okx & oky)

    def kept_counts(self) -> Tuple[int, int]:
        """(localizations assigned to kept emitters, kept emitters with >= 1 localization) of the current state
        [code: BaGoL.m; the code has no empty emitters (it prunes them), so its K_r is the occupied count; with
        the Dirichlet allocation empty emitters are frequent and counting them would bias xi downwards]. With
        ``cfg.xi_counts == 'nonsingleton'`` only the kept emitters with >= 2 localizations count (module docstring,
        'Hierarchical xi')."""
        keep = self.kept_mask(self.mux, self.muy)
        cnt = np.bincount(self.zlab, minlength=self.mux.size)
        use = keep & (cnt >= (2 if self.cfg.xi_counts == "nonsingleton" else 1))
        return int(np.count_nonzero(use[self.zlab])), int(np.count_nonzero(use))

    # -- MAPN ---------------------------------------------------------------------------------------------
    def finalize(self) -> PreclusterResult:
        """MAPN of the pre-cluster [code: genMAPN.m]. With ``cfg.mapn_occupied`` (default) the most frequent number
        of emitters WITH localizations is taken, and the k-means runs on the positions of those emitters only: an
        empty emitter (allowed by the model, uniform on S) is a nuisance component, and the original code never
        reports one because it prunes them at every step. With the Dirichlet allocation empty components are
        frequent (prior odds about (K + 1) / (1 + mean xi) each), so a MAPN over all K would add positions that
        explain no localization. ``k_hist``/``k_mean`` still describe the full K; ``k_occ_hist``/``k_occ_mean``
        the occupied count; ``k_mapn`` is the mode used."""
        if not self.k_rec:
            raise RuntimeError("no post-burn-in samples: n_trials must be > 0")
        ks_all = np.asarray(self.k_rec, dtype=np.int64)
        occ_idx = [np.unique(zz) for zz in self.z_rec]                 # occupied emitters of every state
        occ_n = np.array([o.size for o in occ_idx], dtype=np.int64)
        use_occ = bool(self.cfg.mapn_occupied)
        ks = occ_n if use_occ else ks_all
        vals, cnts = np.unique(ks, return_counts=True)
        k_map = int(vals[np.argmax(cnts)])          # np.argmax -> first max -> smallest K on ties
        sel = np.flatnonzero(ks == k_map)
        if use_occ:
            px = np.concatenate([self.mux_rec[s][occ_idx[s]] for s in sel])
            py = np.concatenate([self.muy_rec[s][occ_idx[s]] for s in sel])
        else:
            px = np.concatenate([self.mux_rec[s] for s in sel])
            py = np.concatenate([self.muy_rec[s] for s in sel])
        pts = np.column_stack([px, py])
        if k_map == 1:
            lab: NDArray[np.int64] = np.zeros(pts.shape[0], dtype=np.int64)
        else:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k_map, n_init=int(math.ceil(k_map / 3.0)) + 1, max_iter=300,
                        random_state=int(self.rng.integers(0, 2 ** 31 - 1)))
            lab = np.asarray(km.fit_predict(pts), dtype=np.int64)
        ns = sel.size
        lab2 = lab.reshape(ns, k_map)
        if use_occ:
            zs = np.stack([np.searchsorted(occ_idx[s], self.z_rec[s]) for s in sel]).astype(np.int64)
        else:
            zs = np.stack([self.z_rec[s] for s in sel]).astype(np.int64)        # (ns, N)
        loc_lab = np.take_along_axis(lab2, zs, axis=1)                     # (ns, N): k-means label per loc
        member = np.zeros((self.n, k_map), dtype=np.float64)
        np.add.at(member, (np.tile(np.arange(self.n), ns), loc_lab.ravel()), 1.0)
        member /= float(ns)
        ex = np.zeros(k_map)
        ey = np.zeros(k_map)
        esx = np.zeros(k_map)
        esy = np.zeros(k_map)
        ez = np.full(k_map, np.nan)
        eax = np.zeros(k_map)
        eay = np.zeros(k_map)
        if self.drift:
            if use_occ:
                pax = np.concatenate([self.ax_rec[s][occ_idx[s]] for s in sel])
                pay = np.concatenate([self.ay_rec[s][occ_idx[s]] for s in sel])
            else:
                pax = np.concatenate([self.ax_rec[s] for s in sel])
                pay = np.concatenate([self.ay_rec[s] for s in sel])
        for c in range(k_map):
            m = lab == c
            ex[c], ey[c] = px[m].mean(), py[m].mean()
            if self.drift:
                eax[c], eay[c] = pax[m].mean(), pay[m].mean()
            esx[c] = px[m].std(ddof=1) if m.sum() > 1 else 0.0
            esy[c] = py[m].std(ddof=1) if m.sum() > 1 else 0.0
            if self.zc is not None:
                w = member[:, c].copy()
                if self.szc is not None:
                    w = w / self.szc ** 2
                if w.sum() > 0:
                    ez[c] = float(np.sum(w * self.zc) / w.sum())
        n_mean = member.sum(axis=0)
        occ_s = np.zeros((ns, k_map), dtype=bool)
        occ_s[np.repeat(np.arange(ns), self.n), loc_lab.ravel()] = True
        p_occ = occ_s.mean(axis=0)
        kept = self.kept_mask(ex, ey)
        loc_label = np.argmax(member, axis=1).astype(np.int64)
        acc = {name: (float(self.n_acc[j]) / self.n_prop[j] if self.n_prop[j] else float("nan"))
               for j, name in enumerate(("move", "allocate", "birth", "death"))}
        va, ca = np.unique(ks_all, return_counts=True)
        ov, oc = np.unique(occ_n, return_counts=True)
        return PreclusterResult(ex, ey, esx, esy, ez, n_mean, kept, loc_label, k_map,
                                float(cnts.max()) / ks.size, float(ks_all.mean()),
                                {int(v): int(c) for v, c in zip(va, ca)}, acc, eax, eay,
                                float(occ_n.mean()), {int(v): int(c) for v, c in zip(ov, oc)}, p_occ)


def _single_result(x: float, y: float, sx: float, sy: float, z: float, keep: bool) -> PreclusterResult:
    """A pre-cluster of one localization: one emitter at it (no chain)."""
    return PreclusterResult(np.array([x]), np.array([y]), np.array([sx]), np.array([sy]), np.array([z]),
                            np.array([1.0]), np.array([keep]), np.zeros(1, dtype=np.int64), 1, 1.0, 1.0,
                            {1: 1}, {}, k_occ_mean=1.0, k_occ_hist={1: 1}, p_occ=np.array([1.0]))


# ---------------------------------------------------------------------------
# Subregions and pre-clusters
# ---------------------------------------------------------------------------

@dataclass
class Precluster:
    idx: NDArray[np.int64]                       # global localization indices
    core: Tuple[float, float, float, float, bool, bool]
    loc_in_core: NDArray[np.bool_]


def resolve_overlap(sigma_xy: NDArray[np.float64], overlap_nm: float = OVERLAP_NM,
                    overlap_sigma_factor: float = OVERLAP_SIGMA_FACTOR) -> float:
    """Subregion overlap in nm: max(``overlap_nm``, ``overlap_sigma_factor`` * the 95th percentile of the
    per-localization max(sigma_x, sigma_y)) [derivation: module docstring, "Subregion overlap"].
    ``overlap_sigma_factor=0`` gives the code's fixed ``overlap_nm``."""
    s = np.asarray(sigma_xy, dtype=np.float64)
    if overlap_sigma_factor <= 0 or s.size == 0:
        return float(overlap_nm)
    smax = s.reshape(s.shape[0], -1).max(axis=1)
    s95 = float(np.percentile(smax, OVERLAP_SIGMA_PERCENTILE))
    return float(max(float(overlap_nm), float(overlap_sigma_factor) * s95))


def make_preclusters(x: NDArray[np.float64], y: NDArray[np.float64], roi_size_nm: float = ROI_SIZE_NM,
                     overlap_nm: float = OVERLAP_NM, cutoff_nm: Optional[float] = None) -> List[Precluster]:
    """Tiles of ``roi_size_nm`` from the data minimum, extended by ``overlap_nm`` on every side (inclusive bounds),
    then single-linkage pre-clusters cut at ``cutoff_nm`` (default ``roi_size_nm``) inside each tile
    [code: genROIs.m, precluster.m; the origin is the data minimum, not the code's 0-if-below-ROIsize rule].
    The core tiles (overlap removal) of the first and last row/column are unbounded on their outer side."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    cutoff = float(roi_size_nm if cutoff_nm is None else cutoff_nm)
    x0, y0 = float(x.min()), float(y.min())
    nxs = max(1, int(math.ceil((float(x.max()) - x0) / roi_size_nm)))
    nys = max(1, int(math.ceil((float(y.max()) - y0) / roi_size_nm)))
    out: List[Precluster] = []
    for iy in range(nys):
        for ix in range(nxs):
            cx0, cx1 = x0 + ix * roi_size_nm, x0 + (ix + 1) * roi_size_nm
            cy0, cy1 = y0 + iy * roi_size_nm, y0 + (iy + 1) * roi_size_nm
            last_x, last_y = ix == nxs - 1, iy == nys - 1
            if last_x:
                cx1 = max(cx1, float(x.max()))
            if last_y:
                cy1 = max(cy1, float(y.max()))
            inside = ((x >= cx0 - overlap_nm) & (x <= cx1 + overlap_nm)
                      & (y >= cy0 - overlap_nm) & (y <= cy1 + overlap_nm))
            idx = np.flatnonzero(inside).astype(np.int64)
            if idx.size == 0:
                continue
            # The core of an outer tile is unbounded on its outer side: an emitter estimated just beyond the data
            # range (e.g. at the extreme localization of the data) belongs to no other tile and would otherwise be
            # dropped [derivation; the code bounds the core by the tile, which loses such emitters].
            kx0 = -math.inf if ix == 0 else cx0
            ky0 = -math.inf if iy == 0 else cy0
            kx1 = math.inf if last_x else cx1
            ky1 = math.inf if last_y else cy1
            core = (kx0, kx1, ky0, ky1, last_x, last_y)
            cx0, cx1, cy0, cy1 = kx0, kx1, ky0, ky1
            pts = np.column_stack([x[idx], y[idx]])
            if idx.size == 1:
                comp = np.zeros(1, dtype=np.int64)
            else:
                pairs = cKDTree(pts).query_pairs(cutoff, output_type="ndarray")
                adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(idx.size, idx.size))
                _, comp = connected_components(adj, directed=False)
            for c in np.unique(comp):
                sub = idx[comp == c]
                lx, ly = x[sub], y[sub]
                okx = (lx >= cx0) & ((lx <= cx1) if last_x else (lx < cx1))
                oky = (ly >= cy0) & ((ly <= cy1) if last_y else (ly < cy1))
                out.append(Precluster(sub, core, np.asarray(okx & oky)))
    return out


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _build_chains(payload: List[Dict[str, Any]], cfg: ChainConfig, xi0: XiPrior) -> List[Optional[EmitterChain]]:
    chains: List[Optional[EmitterChain]] = []
    for p in payload:
        if p["x"].size == 1:
            chains.append(None)
        else:
            chains.append(EmitterChain(p["x"], p["y"], p["sx"], p["sy"], p["seed"], cfg, xi0, core=p["core"],
                                       z=p.get("z"), sz=p.get("sz"), t=p.get("t")))
    return chains


def _single_counts(p: Dict[str, Any]) -> Tuple[int, int]:
    x0, x1, y0, y1, cx, cy = p["core"]
    xv, yv = float(p["x"][0]), float(p["y"][0])
    ok = (xv >= x0) and ((xv <= x1) if cx else (xv < x1)) and (yv >= y0) and ((yv <= y1) if cy else (yv < y1))
    return (1, 1) if ok else (0, 0)


def _finalize_all(chains: List[Optional[EmitterChain]], payload: List[Dict[str, Any]]) -> List[PreclusterResult]:
    res: List[PreclusterResult] = []
    for ch, p in zip(chains, payload):
        if ch is None:
            keep = _single_counts(p)[1] == 1
            zv = float(p["z"][0]) if p.get("z") is not None else float("nan")
            res.append(_single_result(float(p["x"][0]), float(p["y"][0]), float(p["sx"][0]), float(p["sy"][0]),
                                      zv, keep))
        else:
            res.append(ch.finalize())
    return res


def _counts_all(chains: List[Optional[EmitterChain]], payload: List[Dict[str, Any]],
                cfg: ChainConfig) -> List[Tuple[int, int]]:
    """(N_r, K_r) of every pre-cluster for the xi update; a one-localization pre-cluster is a singleton and counts
    nothing with ``xi_counts='nonsingleton'``."""
    single_ok = cfg.xi_counts != "nonsingleton"
    return [((_single_counts(p) if single_ok else (0, 0)) if ch is None else ch.kept_counts())
            for ch, p in zip(chains, payload)]


def _worker_main(conn: Any, payload: List[Dict[str, Any]], cfg: ChainConfig, xi0: XiPrior) -> None:
    chains = _build_chains(payload, cfg, xi0)
    while True:
        msg = conn.recv()
        if msg[0] == "run":
            n_steps, xi = msg[1], msg[2]
            for ch in chains:
                if ch is not None:
                    ch.run(n_steps, xi)
            conn.send(_counts_all(chains, payload, cfg))
        elif msg[0] == "finalize":
            conn.send(_finalize_all(chains, payload))
        else:
            break
    conn.close()


class _Pool:
    """Persistent workers, each owning a fixed set of pre-clusters (so the chains survive between xi updates)."""

    def __init__(self, payload: List[Dict[str, Any]], cfg: ChainConfig, xi0: XiPrior, processes: int) -> None:
        self.payload = payload
        self.cfg = cfg
        n_proc = max(1, min(int(processes), len(payload)))
        cost = np.array([float(p["x"].size) ** 2 for p in payload])
        groups: List[List[int]] = [[] for _ in range(n_proc)]
        load = np.zeros(n_proc)
        for i in np.argsort(-cost, kind="stable"):            # longest-processing-time first
            w = int(np.argmin(load))
            groups[w].append(int(i))
            load[w] += cost[i]
        self.groups = [g for g in groups if g]
        self.local: Optional[List[Optional[EmitterChain]]] = None
        self.conns: List[Any] = []
        self.procs: List[Any] = []
        if len(self.groups) == 1:
            self.local = _build_chains(payload, cfg, xi0)
            return
        ctx = mp.get_context("spawn")
        for g in self.groups:
            parent, child = ctx.Pipe()
            pr = ctx.Process(target=_worker_main, args=(child, [payload[i] for i in g], cfg, xi0), daemon=True)
            pr.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(pr)

    def run(self, n_steps: int, xi: XiPrior) -> List[Tuple[int, int]]:
        if self.local is not None:
            for ch in self.local:
                # F5d: a cancelled run of the MPS analysis window stops between pre-clusters (tools.mps_cancel;
                # a no-op elsewhere)
                mps_cancel.check("Emitters layer")
                if ch is not None:
                    ch.run(n_steps, xi)
            return _counts_all(self.local, self.payload, self.cfg)
        for c in self.conns:
            c.send(("run", n_steps, xi))
        out: List[Tuple[int, int]] = [(0, 0)] * len(self.payload)
        try:
            for c, g, pr in zip(self.conns, self.groups, self.procs):
                for i, v in zip(g, self._recv(c, pr)):
                    out[i] = v
        except mps_cancel.AnalysisCancelled:
            # the workers are in the middle of their chains and would not read 'stop' before the end of them
            self.abort()
            raise
        return out

    def finalize(self) -> List[PreclusterResult]:
        if self.local is not None:
            return _finalize_all(self.local, self.payload)
        for c in self.conns:
            c.send(("finalize",))
        out: List[Optional[PreclusterResult]] = [None] * len(self.payload)
        try:
            for c, g, pr in zip(self.conns, self.groups, self.procs):
                for i, v in zip(g, self._recv(c, pr)):
                    out[i] = v
        except mps_cancel.AnalysisCancelled:
            self.abort()
            raise
        return [r for r in out if r is not None]

    @staticmethod
    def _recv(conn: Any, proc: Any) -> Any:
        """Wait for a worker's answer; raise instead of hanging when the worker died (e.g. a script started
        without the ``if __name__ == '__main__':`` guard that the 'spawn' start method of Windows needs)."""
        while not conn.poll(0.5):
            mps_cancel.check("Emitters layer")
            if not proc.is_alive():
                raise RuntimeError(f"emitters worker {proc.pid} exited with code {proc.exitcode}; on Windows the "
                                   "calling script needs an `if __name__ == '__main__':` guard")
        return conn.recv()

    def abort(self) -> None:
        """Stop the worker processes at once (a cancelled run, F5d): nothing they compute is wanted."""
        for pr in self.procs:
            if pr.is_alive():
                pr.terminate()
        for pr in self.procs:
            pr.join(timeout=5)
        for c in self.conns:
            try:
                c.close()
            except OSError:
                pass
        self.conns = []
        self.procs = []

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("stop",))
                c.close()
            except (OSError, EOFError):
                pass
        for pr in self.procs:
            pr.join(timeout=10)
            if pr.is_alive():
                pr.terminate()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class EmitterResult:
    """Emitters (BaGoL-like reimplementation) of a set of localizations.

    ``labels``: emitter index per localization (-1 when none of the subregions containing it kept the emitter it
    was allocated to). Per emitter: x, y, sigma_x, sigma_y (sd of the chain positions, nm), z (membership-weighted
    mean of the localizations' z; NaN without z), n_locs (posterior mean of the localizations it explains) and the
    pre-cluster it comes from. Per pre-cluster: K_MAPN, P(K = K_MAPN), posterior mean of K, the K histogram.

    K counts every emitter of the model, also those without localizations (an empty emitter is allowed by the
    model and, with ``prune_empty`` off, is not deleted; its position is uniform on the support). So, per
    pre-cluster, ``k_occupied_mean`` is the posterior mean of the emitters with >= 1 localization, and per
    emitter ``p_occupied`` is the fraction of the K = K_MAPN states in which it has >= 1 localization.

    ``has_locs`` is n_locs >= ``has_locs_min`` (default 0.5), with n_locs the posterior mean of the number of
    localizations the MAPN emitter explains: 0.5 means "rounded, it explains at least one localization". It says
    only whether the emitter is backed by data in this run. It has nothing to do with epitopes or antibodies:
    how many emitters one molecule gives (labels per protein, the label's blinking) belongs to the step from
    emitters to molecules (decision D-B5) and to the prior xi, not to this threshold. Origin [simulation, this
    project, F3 session 3]: on synthetic emitters with the truth (sigma 12 nm, Poisson lambda 3, 5, 7 and 14),
    1.0 and p_occupied >= 0.5 gave the same Jaccard index within the spread between fields; 0.5 keeps the most
    real emitters and is the simplest. ``n_emitters_with_locs`` counts those emitters. They differ from K
    only when lambda is small (<= ~7).
    """

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    sigma_x: NDArray[np.float64]
    sigma_y: NDArray[np.float64]
    z: NDArray[np.float64]
    n_locs: NDArray[np.float64]
    precluster: NDArray[np.int64]
    labels: NDArray[np.int64]
    k_mapn: NDArray[np.int64]
    p_k_mapn: NDArray[np.float64]
    k_mean: NDArray[np.float64]
    k_hist: List[Dict[int, int]]
    accept: List[Dict[str, float]]
    alpha_x: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    alpha_y: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    n_orphans_assigned: int = 0
    xi_chain: List[Tuple[float, ...]] = field(default_factory=list)
    xi_used: Tuple[float, ...] = ()
    xi_kind: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    k_occupied_mean: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    k_occupied_hist: List[Dict[int, int]] = field(default_factory=list)
    p_occupied: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    has_locs_min: float = HAS_LOCS_MIN

    @property
    def n_emitters(self) -> int:
        return int(self.x.size)

    @property
    def has_locs(self) -> NDArray[np.bool_]:
        """Emitters whose n_locs (posterior mean of the localizations they explain) is >= ``has_locs_min``."""
        return np.asarray(self.n_locs >= self.has_locs_min)

    @property
    def n_emitters_with_locs(self) -> int:
        return int(np.count_nonzero(self.has_locs))

    @property
    def is_nonsingleton(self) -> NDArray[np.bool_]:
        """Emitters whose n_locs is >= ``NONSINGLETON_MIN`` (1.5: rounded, at least two localizations)."""
        return np.asarray(self.n_locs >= NONSINGLETON_MIN)

    @property
    def n_emitters_nonsingleton(self) -> int:
        """Emitters with >= 2 localizations (rounded posterior mean): the readouts per cluster count these."""
        return int(np.count_nonzero(self.is_nonsingleton))

    @property
    def n_singletons(self) -> int:
        """Emitters with localizations but fewer than ``NONSINGLETON_MIN``: one localization (one event). With an
        isolated background they are mostly that background, a legitimate one-localization emitter of the model
        (F4 part B); reported apart, never mixed into the readouts per cluster."""
        return int(np.count_nonzero(self.has_locs & ~self.is_nonsingleton))


def assign_orphans_to_nearest(xy: NDArray[np.float64], sxy: NDArray[np.float64], ex: NDArray[np.float64],
                              ey: NDArray[np.float64], esx: NDArray[np.float64], esy: NDArray[np.float64],
                              n_sigma: float = ORPHAN_N_SIGMA) -> NDArray[np.int64]:
    """Emitter index for each orphan localization, or -1 (noise).

    An orphan goes to its nearest emitter (Euclidean) when the distance is <= ``n_sigma`` * s, with the
    combined sigma s = sqrt(sigma_loc^2 + sigma_emitter^2): sigma_loc = max(sigma_x, sigma_y) of the
    localization, sigma_emitter = max(sigma_x, sigma_y) of the emitter's posterior position (sd of the chain).
    If the localization came from that emitter, their difference has variance sigma_loc^2 + sigma_emitter^2 per
    axis, so with s = that sigma the distance exceeds 3 s with probability exp(-4.5) = 1.1 % (2-D, isotropic);
    the larger axis of each makes the test lenient for anisotropic precisions [derivation; 3 = decision of the
    user, 2026-10-16, F3 (b)]. Without emitters every orphan stays -1.
    """
    out = np.full(xy.shape[0], -1, dtype=np.int64)
    if xy.shape[0] == 0 or ex.size == 0:
        return out
    from scipy.spatial import cKDTree
    d, j = cKDTree(np.column_stack([ex, ey])).query(xy, k=1)
    j = np.asarray(j, dtype=np.int64)
    s_loc = np.max(sxy, axis=1)
    s_em = np.maximum(esx, esy)[j]
    ok = d <= float(n_sigma) * np.sqrt(s_loc ** 2 + s_em ** 2)
    out[ok] = j[ok]
    return out


def fit_emitters(xy: NDArray[np.float64], sigma: Any, xi: XiPrior, *, hierarchical: bool = False,
                 z: Optional[NDArray[np.float64]] = None, sigma_z: Optional[NDArray[np.float64]] = None,
                 n_burnin: int = N_BURNIN, n_trials: int = N_TRIALS, n_samples_xi: int = N_SAMPLES_XI,
                 alpha_xi: float = ALPHA_XI, beta_xi: float = BETA_XI,
                 roi_size_nm: float = ROI_SIZE_NM, overlap_nm: float = OVERLAP_NM,
                 overlap_sigma_factor: float = OVERLAP_SIGMA_FACTOR,
                 cutoff_nm: Optional[float] = None, se_adjust_nm: float = 0.0, grid_nm: float = GRID_NM,
                 processes: int = DEFAULT_PROCESSES, seed: int = 0, drift: float = 0.0,
                 frames: Optional[NDArray[np.float64]] = None, assign_orphans: bool = True,
                 legacy_death_ratio: bool = False, prune_empty: bool = False,
                 matlab_compat: bool = False, has_locs_min: float = HAS_LOCS_MIN,
                 allocation: str = ALLOCATION, mapn_occupied: bool = True,
                 xi_counts: str = XI_COUNTS) -> EmitterResult:
    """Group localizations into emitters (see the module docstring for the model).

    ``xy``: (n, 2) positions in nm (frame-connected events, as BaGoL requires). ``sigma``: scalar, (n,) or (n, 2)
    localization precisions in nm (sigma_x, sigma_y). ``xi``: the prior on localizations per emitter (fixed, or the
    starting value when ``hierarchical``). Results depend only on ``seed`` (not on ``processes``): every
    pre-cluster has its own random stream spawned from the seed.

    The subregion overlap is max(``overlap_nm``, ``overlap_sigma_factor`` * sigma_95) (:func:`resolve_overlap`);
    the value used is in ``params['overlap_used_nm']``. ``has_locs_min``: the n_locs an emitter needs to count
    as "with localizations" (:class:`EmitterResult`); it changes no estimate, only that flag and its count.

    ``allocation``: prior of the allocation of localizations to emitters, ``'dirichlet'`` (default; a weight
    vector w ~ Dirichlet(k) for a gamma xi of shape k: the exact conditional for gamma / negative-binomial counts)
    or ``'uniform'`` (the code's K^-N, exact only for a Poisson xi, which always uses it). With the uniform
    allocation and exponential counts, bright emitters in dense data are split into ~N_b / mean(xi) emitters
    (module docstring, "Allocation prior"; measured in F4). ``params['allocation_used']`` says which one ran.
    ``mapn_occupied`` (default True): the MAPN takes the most frequent number of emitters WITH localizations and
    places only those (:meth:`EmitterChain.finalize`); False reproduces the MAPN over all K, empty ones included.
    ``xi_counts`` (hierarchical xi only): ``'nonsingleton'`` (default) learns xi from the kept emitters with >= 2
    localizations, ``'occupied'`` from every kept emitter with >= 1 (the code's); module docstring, 'Hierarchical
    xi'. ``prune_empty`` together with the Dirichlet allocation raises a ``UserWarning`` (it biases K down).

    ``matlab_compat=True`` reproduces the original MATLAB code (LidkeLab/BaGoL ``BaGoL_RJMCMC.m``) and is meant
    ONLY for comparisons with it. It switches on, together:

    * ``legacy_death_ratio``: the code's Death acceptance ``Pk(K-1)/Pk(K) * L'/L``. It omits the factor
      ``Area * q(mu_j)``, so it is not the inverse of the code's own Birth ratio,
      which contains ``1 / (Area * q(mu*))``; no prior on the positions makes both rules valid at once, and the
      paper itself (Fazel et al. 2022, Supplementary Note 1, eq. 27) defines the Death acceptance as the inverse
      of the Birth ratio (eq. 26). Where a surplus emitter sits on the localizations, ``Area * q`` is about 3-6,
      so surplus emitters die that many times too rarely and K is biased upwards;
    * ``prune_empty``: the code deletes emitters without localizations at every step, with no acceptance ratio
      (not a reversible move), which biases K downwards when lambda is small (<= ~4);
    * the code's fixed 20-nm overlap (``overlap_sigma_factor`` is ignored);
    * the code's uniform allocation (``allocation='uniform'``) and its counts for xi (``xi_counts='occupied'``).

    Measured against the exact posterior (independent review of F3): the Death rule alone gives +0.10 to +0.14
    in P(K = 2) (K <= 2, lambda 3); the full code (both together) +0.08 in that case, +0.04 to +0.16 in E[K] with
    lambda 8 to 20 and none when the posterior is decisive; with lambda <= 4 and K unrestricted the pruning can
    win and leave E[K] below the exact value (-0.11), while the number of occupied emitters stays biased upwards.
    Either flag alone is not the original code.
    """
    if matlab_compat:
        legacy_death_ratio, prune_empty, overlap_sigma_factor, allocation = True, True, 0.0, "uniform"
        xi_counts = "occupied"
    if allocation not in ALLOCATIONS:
        raise ValueError(f"allocation must be one of {ALLOCATIONS}, got {allocation!r}")
    if xi_counts not in XI_COUNTS_CHOICES:
        raise ValueError(f"xi_counts must be one of {XI_COUNTS_CHOICES}, got {xi_counts!r}")
    allocation_used = "dirichlet" if (allocation == "dirichlet" and xi.kind == "gamma") else "uniform"
    if prune_empty and allocation_used == "dirichlet":
        import warnings
        warnings.warn("prune_empty with the Dirichlet allocation deletes the empty emitters that this prior makes "
                      "frequent and biases K down (-7 to -12 % on synthetic dense fields); use it only with "
                      "allocation='uniform' or matlab_compat=True", UserWarning, stacklevel=2)
    if drift and frames is None:
        raise ValueError("drift > 0 needs the frame number of every localization (frames)")
    if not has_locs_min >= 0.0:
        raise ValueError("has_locs_min must be >= 0")
    xy = np.asarray(xy, dtype=np.float64)
    n = xy.shape[0]
    sig = np.asarray(sigma, dtype=np.float64)
    if sig.ndim == 0:
        sxy: NDArray[np.float64] = np.full((n, 2), float(sig))
    elif sig.ndim == 1:
        sxy = np.column_stack([sig, sig])
    else:
        sxy = sig[:, :2]
    sxy = sxy + float(se_adjust_nm)                     # [code: precluster.m adds SE_Adjust]
    if not np.all(sxy > 0):
        raise ValueError("localization precisions must be > 0 (BaGoL drops X_SE = 0)")
    overlap_used = resolve_overlap(sxy, overlap_nm, overlap_sigma_factor)
    cfg = ChainConfig(n_burnin=int(n_burnin), n_trials=int(n_trials), grid_nm=float(grid_nm),
                      legacy_death_ratio=bool(legacy_death_ratio), prune_empty=bool(prune_empty),
                      drift_sigma=float(drift), allocation=str(allocation), mapn_occupied=bool(mapn_occupied),
                      xi_counts=str(xi_counts))
    params = dict(xi_kind=xi.kind, xi=xi.as_tuple(), hierarchical=bool(hierarchical), n_burnin=int(n_burnin),
                  n_trials=int(n_trials), n_samples_xi=int(n_samples_xi), alpha_xi=float(alpha_xi),
                  beta_xi=float(beta_xi), roi_size_nm=float(roi_size_nm), overlap_nm=float(overlap_nm),
                  overlap_sigma_factor=float(overlap_sigma_factor), overlap_used_nm=overlap_used,
                  cutoff_nm=cutoff_nm, se_adjust_nm=float(se_adjust_nm), grid_nm=float(grid_nm), seed=int(seed),
                  legacy_death_ratio=bool(legacy_death_ratio), prune_empty=bool(prune_empty),
                  drift=float(drift), assign_orphans=bool(assign_orphans), matlab_compat=bool(matlab_compat),
                  has_locs_min=float(has_locs_min), orphan_n_sigma=ORPHAN_N_SIGMA, allocation=str(allocation),
                  allocation_used=allocation_used, mapn_occupied=bool(mapn_occupied), xi_counts=str(xi_counts),
                  nonsingleton_min=NONSINGLETON_MIN, method=PUBLIC_LABEL)
    empty_f = np.empty(0, dtype=np.float64)
    if n == 0:
        return EmitterResult(empty_f, empty_f, empty_f, empty_f, empty_f, empty_f, np.empty(0, dtype=np.int64),
                             np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), empty_f, empty_f, [], [],
                             xi_used=xi.as_tuple(), xi_kind=xi.kind, params=params,
                             has_locs_min=float(has_locs_min))
    pcs = make_preclusters(xy[:, 0], xy[:, 1], roi_size_nm, overlap_used, cutoff_nm)
    seeds = np.random.SeedSequence(int(seed)).spawn(len(pcs) + 1)
    payload: List[Dict[str, Any]] = []
    for pc, ss in zip(pcs, seeds[:-1]):
        p: Dict[str, Any] = {"x": xy[pc.idx, 0], "y": xy[pc.idx, 1], "sx": sxy[pc.idx, 0], "sy": sxy[pc.idx, 1],
                             "seed": ss, "core": pc.core}
        if drift:
            p["t"] = np.asarray(frames, dtype=np.float64)[pc.idx]
        if z is not None:
            p["z"] = np.asarray(z, dtype=np.float64)[pc.idx]
            if sigma_z is not None:
                p["sz"] = np.asarray(sigma_z, dtype=np.float64)[pc.idx]
        payload.append(p)
    xi_rng = np.random.default_rng(seeds[-1])
    pool = _Pool(payload, cfg, xi, processes)
    xi_chain: List[Tuple[float, ...]] = []
    try:
        total = int(n_burnin) + int(n_trials)
        if hierarchical:
            cur = xi
            done = 0
            while done < total:
                step = min(int(n_samples_xi), total - done)
                counts = pool.run(step, cur)
                done += step
                cur = sample_xi(cur, [c[0] for c in counts], [c[1] for c in counts], xi_rng, alpha_xi, beta_xi)
                xi_chain.append(cur.as_tuple())
            n_burn_xi = int(n_burnin) // max(1, int(n_samples_xi))
            post = np.asarray(xi_chain[n_burn_xi:] or xi_chain, dtype=np.float64)
            xi_used = tuple(float(v) for v in post.mean(axis=0))
        else:
            pool.run(total, xi)
            xi_used = xi.as_tuple()
        results = pool.finalize()
    finally:
        pool.close()
    # Stitch: kept emitters get global ids; a localization takes the label of the subregion whose core
    # contains it when that subregion kept its emitter, else of any subregion that kept it [ad hoc].
    cols: Dict[str, List[NDArray[Any]]] = {k: [] for k in ("x", "y", "sx", "sy", "z", "n", "po", "pc", "ax", "ay")}
    lab_core = np.full(n, -1, dtype=np.int64)
    lab_any = np.full(n, -1, dtype=np.int64)
    next_id = 0
    for r_i, (pc, r) in enumerate(zip(pcs, results)):
        gmap = np.full(r.x.size, -1, dtype=np.int64)
        kk = int(np.count_nonzero(r.kept))
        gmap[r.kept] = np.arange(next_id, next_id + kk)
        next_id += kk
        rax = np.zeros(r.x.size) if r.alpha_x is None else r.alpha_x
        ray = np.zeros(r.x.size) if r.alpha_y is None else r.alpha_y
        rpo = np.ones(r.x.size) if r.p_occ is None else r.p_occ
        for key, arr in (("x", r.x), ("y", r.y), ("sx", r.sx), ("sy", r.sy), ("z", r.z), ("n", r.n_mean),
                         ("po", rpo), ("ax", rax), ("ay", ray)):
            cols[key].append(arr[r.kept])
        cols["pc"].append(np.full(kk, r_i, dtype=np.int64))
        g = gmap[r.loc_label]
        ok = g >= 0
        core_ok = ok & pc.loc_in_core
        lab_core[pc.idx[core_ok]] = g[core_ok]
        lab_any[pc.idx[ok]] = g[ok]
    labels = np.where(lab_core >= 0, lab_core, lab_any)

    def cat(key: str, dtype: Any) -> NDArray[Any]:
        return np.concatenate(cols[key]).astype(dtype) if cols[key] else np.empty(0, dtype=dtype)

    # A localization whose emitter was kept by no subregion that contains it (its emitter sits in a tile whose
    # overlap does not reach it) goes to the nearest kept emitter within 3 combined sigmas, else stays -1
    # (noise): see assign_orphans_to_nearest (decision of the user, 2026-10-16).
    n_orph = 0
    orphans = np.flatnonzero(labels < 0)
    if assign_orphans and orphans.size:
        j = assign_orphans_to_nearest(xy[orphans], sxy[orphans], cat("x", np.float64), cat("y", np.float64),
                                      cat("sx", np.float64), cat("sy", np.float64))
        labels[orphans] = j
        n_orph = int(np.count_nonzero(j >= 0))

    return EmitterResult(
        cat("x", np.float64), cat("y", np.float64), cat("sx", np.float64), cat("sy", np.float64),
        cat("z", np.float64), cat("n", np.float64), cat("pc", np.int64), labels,
        np.array([r.k_mapn for r in results], dtype=np.int64),
        np.array([r.p_k_mapn for r in results], dtype=np.float64),
        np.array([r.k_mean for r in results], dtype=np.float64),
        [r.k_hist for r in results], [r.accept for r in results],
        alpha_x=cat("ax", np.float64), alpha_y=cat("ay", np.float64), n_orphans_assigned=n_orph,
        xi_chain=xi_chain, xi_used=xi_used, xi_kind=xi.kind, params=params,
        k_occupied_mean=np.array([r.k_occ_mean for r in results], dtype=np.float64),
        k_occupied_hist=[r.k_occ_hist for r in results], p_occupied=cat("po", np.float64),
        has_locs_min=float(has_locs_min))


# ---------------------------------------------------------------------------
# Strategy for the clustering registry
# ---------------------------------------------------------------------------

class EmittersBagolLikeStrategy(ClusteringStrategy):
    """Emitters (BaGoL-like reimplementation) as a clustering strategy: the labels are the emitter of each
    localization, so every emitter is a 'cluster' for the rest of the pipeline.

    ``fit`` receives (n, 2) positions (or (n, 4): x, y, sigma_x, sigma_y). The precision of each localization
    comes, in this order, from the (n, 4) columns, from :meth:`set_localization_precision` (the file's X_SE,
    Y_SE), or else ``sigma_nm`` for all. ``cluster_slab`` passes only (n, 2): until phase F5 wires the
    precisions through it, the interface path uses ``sigma_nm``. The precisions also set the subregion overlap
    (``overlap_sigma_factor``). The full result (emitters, posterior K) stays in ``result_`` after ``fit``.
    """

    def __init__(self, sigma_nm: float = 12.0, xi_kind: str = "gamma", xi_mean: float = 14.0,
                 xi_shape: float = 1.0, hierarchical: bool = True, n_burnin: int = N_BURNIN,
                 n_trials: int = N_TRIALS, roi_size_nm: float = ROI_SIZE_NM, overlap_nm: float = OVERLAP_NM,
                 overlap_sigma_factor: float = OVERLAP_SIGMA_FACTOR, has_locs_min: float = HAS_LOCS_MIN,
                 processes: int = DEFAULT_PROCESSES, seed: int = 0, allocation: str = ALLOCATION,
                 xi_counts: str = XI_COUNTS, se_adjust_nm: float = 0.0) -> None:
        self.sigma_nm = float(sigma_nm)
        self.xi_kind = str(xi_kind)
        self.xi_mean = float(xi_mean)
        self.xi_shape = float(xi_shape)
        self.hierarchical = bool(hierarchical)
        self.n_burnin = int(n_burnin)
        self.n_trials = int(n_trials)
        self.roi_size_nm = float(roi_size_nm)
        self.overlap_nm = float(overlap_nm)
        self.overlap_sigma_factor = float(overlap_sigma_factor)
        self.has_locs_min = float(has_locs_min)
        if not self.has_locs_min >= 0.0:
            raise ValueError("has_locs_min must be >= 0")
        self.processes = int(processes)
        self.seed = int(seed)
        self.allocation = str(allocation)
        if self.allocation not in ALLOCATIONS:
            raise ValueError(f"allocation must be one of {ALLOCATIONS}, got {allocation!r}")
        self.xi_counts = str(xi_counts)
        if self.xi_counts not in XI_COUNTS_CHOICES:
            raise ValueError(f"xi_counts must be one of {XI_COUNTS_CHOICES}, got {xi_counts!r}")
        self.se_adjust_nm = float(se_adjust_nm)
        if not (math.isfinite(self.se_adjust_nm) and self.se_adjust_nm >= 0.0):
            raise ValueError(f"se_adjust_nm must be >= 0, got {se_adjust_nm}")
        self.result_: Optional[EmitterResult] = None
        self._sigma_xy: Optional[NDArray[np.float64]] = None

    def set_localization_precision(self, sigma_xy: Optional[np.ndarray]) -> None:
        """Per-localization precisions in nm, (n,) or (n, 2) = (X_SE, Y_SE), used by :meth:`fit` when it receives
        (n, 2) positions; ``None`` clears them. A length that does not match the positions raises in ``fit``
        (a stale array from another slab must not be used silently)."""
        if sigma_xy is None:
            self._sigma_xy = None
            return
        a = np.asarray(sigma_xy, dtype=np.float64)
        if a.ndim == 1:
            a = np.column_stack([a, a])
        if a.ndim != 2 or a.shape[1] < 2:
            raise ValueError("precisions must be (n,) or (n, 2)")
        self._sigma_xy = np.ascontiguousarray(a[:, :2])

    def xi_prior(self) -> XiPrior:
        if self.xi_kind == "poisson":
            return XiPrior.poisson(self.xi_mean)
        return XiPrior.gamma(self.xi_shape, self.xi_mean / self.xi_shape)

    def fit(self, data: np.ndarray) -> np.ndarray:
        d = np.asarray(data, dtype=np.float64)
        sig: Any = self.sigma_nm
        if d.ndim == 2 and d.shape[1] >= 4:
            sig = d[:, 2:4]
        elif self._sigma_xy is not None:
            if self._sigma_xy.shape[0] != d.shape[0]:
                raise ValueError(f"{self._sigma_xy.shape[0]} precisions for {d.shape[0]} localizations")
            sig = self._sigma_xy
        self.result_ = fit_emitters(d[:, :2], sig, self.xi_prior(), hierarchical=self.hierarchical,
                                    n_burnin=self.n_burnin, n_trials=self.n_trials, roi_size_nm=self.roi_size_nm,
                                    overlap_nm=self.overlap_nm, overlap_sigma_factor=self.overlap_sigma_factor,
                                    has_locs_min=self.has_locs_min, processes=self.processes, seed=self.seed,
                                    allocation=self.allocation, xi_counts=self.xi_counts,
                                    se_adjust_nm=self.se_adjust_nm)
        return np.asarray(self.result_.labels, dtype=np.int64)

    def get_strategy_name(self) -> str:
        return PUBLIC_LABEL

    def get_params(self) -> Dict[str, Any]:
        return {"sigma_nm": self.sigma_nm, "xi_kind": self.xi_kind, "xi_mean": self.xi_mean,
                "xi_shape": self.xi_shape, "hierarchical": self.hierarchical, "n_burnin": self.n_burnin,
                "n_trials": self.n_trials, "roi_size_nm": self.roi_size_nm, "overlap_nm": self.overlap_nm,
                "overlap_sigma_factor": self.overlap_sigma_factor, "has_locs_min": self.has_locs_min,
                "processes": self.processes, "seed": self.seed, "allocation": self.allocation,
                "xi_counts": self.xi_counts, "se_adjust_nm": self.se_adjust_nm,
                "per_localization_sigma": self._sigma_xy is not None}


def emitters_family() -> Dict[str, Dict[str, Any]]:
    """Registry entry: name -> {'cls', 'description', 'defaults', 'origins', 'citation'}."""
    return {
        "emitters_bagol_like": {
            "cls": EmittersBagolLikeStrategy,
            "description": PUBLIC_LABEL + ": RJMCMC grouping of localizations into emitters (each emitter is a "
                                          "label stack, not a protein); slow, minutes per 10^4 localizations",
            "defaults": {"sigma_nm": 12.0, "xi_kind": "gamma", "xi_mean": 14.0, "xi_shape": 1.0,
                         "hierarchical": True, "n_burnin": N_BURNIN, "n_trials": N_TRIALS,
                         "roi_size_nm": ROI_SIZE_NM, "overlap_nm": OVERLAP_NM,
                         "overlap_sigma_factor": OVERLAP_SIGMA_FACTOR, "has_locs_min": HAS_LOCS_MIN,
                         "processes": DEFAULT_PROCESSES, "seed": 0, "allocation": ALLOCATION,
                         "xi_counts": XI_COUNTS, "se_adjust_nm": 0.0},
            "origins": {
                "sigma_nm": "ad hoc: the benchmark's sigma_loc (12 nm); real precisions should be passed per "
                            "localization",
                "xi_kind": "literature: Fazel et al. (2022) Methods, gamma with shape 1 for dSTORM blinking",
                "xi_mean": "literature + derivation: Dempsey et al. (2011) Table 1, AF647 in MEA, 14 switching "
                           "cycles x 1 localization per cycle after frame linking",
                "xi_shape": "literature: Fazel et al. (2022) Methods, eta = 1 (exponential) for synthetic dSTORM",
                "hierarchical": "decision D-B3: learned xi, for direct use of the strategy; the analysis layer "
                                "(tools.emitters_layer) sets it from its xi source, literature (fixed; the "
                                "buffer chosen per file, F5c) by "
                                "default since F5b",
                "n_burnin": "code: LidkeLab/BaGoL BaGoL.m default N_Burnin",
                "n_trials": "code: LidkeLab/BaGoL BaGoL.m default N_Trials",
                "roi_size_nm": "code: LidkeLab/BaGoL BaGoL.m default ROIsize",
                "overlap_nm": "code: LidkeLab/BaGoL BaGoL.m default Overlap (here the floor of the overlap)",
                "overlap_sigma_factor": "derivation: overlap >= 4 x the 95th-percentile precision, so that an "
                                        "emitter on a subregion border is seen whole by both subregions (with "
                                        "20 nm and sigma 12 nm, 19 % of such emitters were duplicated); 0 = "
                                        "the code's fixed overlap",
                "has_locs_min": "simulation, this project (F3 session 3: synthetic emitters with the truth, "
                                "sigma 12 nm, lambda 3-14; 0.5, 1 and P(occupied) >= 0.5 gave the same Jaccard "
                                "index). An emitter counts as 'with localizations' when the posterior mean of "
                                "the localizations it explains is >= this value; 0.5 (recommended) = rounded, "
                                "at least one. Only that flag and its count change, not the emitters. Not "
                                "related to epitopes or antibodies (that is the emitters-to-molecules step)",
                "processes": "ad hoc: the laptop's process budget",
                "seed": "ad hoc: fixed seed for reproducibility",
                "allocation": "derivation + simulation (F4): 'dirichlet' = weights w ~ Dirichlet(k) per emitter, "
                              "the exact allocation prior for a gamma xi (negative-binomial counts); 'uniform' = "
                              "the original code's K^-N, exact only for a Poisson xi. With 'uniform' and "
                              "exponential counts, emitters with more than ~2 x mean(xi) localizations are split "
                              "(16-22 % emitters in excess on a dense synthetic field); 'dirichlet' removes that",
                "xi_counts": "decision of the user (F4 (d)) + simulation (F4 part B): the hierarchical xi is learned "
                             "on emitters with >= 2 localizations, because isolated background localizations are "
                             "one-localization emitters and made the learned xi a mixture (shape 0.34-0.55); "
                             "'occupied' = the original code's count. Costs about +1 on the mean of an "
                             "exponential xi (derivation)",
                "se_adjust_nm": "code: BaGoL SE_Adjust, added to every precision; literature: Fazel et al. (2022) "
                                "Methods increased the precisions of their real data by 0.75-2 nm. Raise it when "
                                "NeNA exceeds the reported precision (the layer suggests NeNA - median precision "
                                "when NeNA / precision > 1.2); 0 = the file's precisions",
            },
            "citation": CITATION,
        }
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _read_csv_columns(path: str) -> Dict[str, NDArray[np.float64]]:
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return {k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0].keys() if k is not None}


def write_emitters_csv(result: EmitterResult, path: str) -> None:
    """One row per emitter, with the method label and the xi used in every row."""
    import csv
    xi = "|".join(f"{v:g}" for v in result.xi_used)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["emitter", "x_nm", "y_nm", "sigma_x_nm", "sigma_y_nm", "z_nm", "n_locs", "alpha_x_nm_per_frame",
                    "alpha_y_nm_per_frame", "has_locs", "p_occupied", "precluster", "k_mapn", "p_k_mapn", "k_mean",
                    "k_occupied_mean", "xi_kind", "xi_used", "method"])
        ax = result.alpha_x if result.alpha_x.size == result.x.size else np.zeros(result.x.size)
        ay = result.alpha_y if result.alpha_y.size == result.x.size else np.zeros(result.x.size)
        kocc = result.k_occupied_mean
        has = result.has_locs
        pocc = result.p_occupied if result.p_occupied.size == result.x.size else np.full(result.x.size, np.nan)
        for i in range(result.n_emitters):
            pc = int(result.precluster[i])
            w.writerow([i, f"{result.x[i]:.4f}", f"{result.y[i]:.4f}", f"{result.sigma_x[i]:.4f}",
                        f"{result.sigma_y[i]:.4f}", f"{result.z[i]:.4f}", f"{result.n_locs[i]:.3f}",
                        f"{ax[i]:.6g}", f"{ay[i]:.6g}", int(has[i]), f"{pocc[i]:.4f}", pc, int(result.k_mapn[pc]),
                        f"{result.p_k_mapn[pc]:.4f}", f"{result.k_mean[pc]:.4f}",
                        f"{kocc[pc]:.4f}" if pc < kocc.size else "", result.xi_kind, xi, PUBLIC_LABEL])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``py -3 -m tools.cluster_emitters in.csv out_prefix [options]``.

    ``in.csv`` has columns x, y (nm) and optionally sigma_x, sigma_y (or sigma), z, sigma_z, frame. Writes
    ``out_prefix_emitters.csv`` (one row per emitter) and ``out_prefix_labels.csv`` (x, y, label per
    localization: the format of the 'External labels' strategy).
    """
    import argparse
    import csv
    ap = argparse.ArgumentParser(prog="tools.cluster_emitters", description=PUBLIC_LABEL + ". " + CITATION)
    ap.add_argument("csv_in")
    ap.add_argument("out_prefix")
    ap.add_argument("--xi-kind", choices=("poisson", "gamma"), default="gamma")
    ap.add_argument("--xi-mean", type=float, default=14.0)
    ap.add_argument("--xi-shape", type=float, default=1.0)
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("--sigma", type=float, default=None, help="precision (nm) when the file has none")
    ap.add_argument("--n-burnin", type=int, default=N_BURNIN)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--roi-size", type=float, default=ROI_SIZE_NM)
    ap.add_argument("--overlap", type=float, default=OVERLAP_NM, help="minimum subregion overlap (nm)")
    ap.add_argument("--overlap-sigma-factor", type=float, default=OVERLAP_SIGMA_FACTOR,
                    help="overlap >= this x the 95th-percentile precision (0 = fixed --overlap)")
    ap.add_argument("--se-adjust", type=float, default=0.0)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--processes", type=int, default=DEFAULT_PROCESSES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--has-locs-min", type=float, default=HAS_LOCS_MIN,
                    help="n_locs an emitter needs to count as 'with localizations' (recommended 0.5)")
    ap.add_argument("--matlab-compat", action="store_true",
                    help="reproduce the original MATLAB code (biased; for comparisons only)")
    ap.add_argument("--allocation", choices=ALLOCATIONS, default=ALLOCATION,
                    help="allocation prior for a gamma xi: dirichlet (exact) or uniform (the code's)")
    ap.add_argument("--xi-counts", choices=XI_COUNTS_CHOICES, default=XI_COUNTS,
                    help="emitters the hierarchical xi counts: nonsingleton (>= 2 localizations) or occupied")
    a = ap.parse_args(argv)
    cols = _read_csv_columns(a.csv_in)
    xy = np.column_stack([cols["x"], cols["y"]])
    if "sigma_x" in cols and "sigma_y" in cols:
        sig: Any = np.column_stack([cols["sigma_x"], cols["sigma_y"]])
    elif "sigma" in cols:
        sig = cols["sigma"]
    elif a.sigma is not None:
        sig = float(a.sigma)
    else:
        ap.error("the file has no sigma_x/sigma_y or sigma column: pass --sigma")
    xi = XiPrior.poisson(a.xi_mean) if a.xi_kind == "poisson" else XiPrior.gamma(a.xi_shape, a.xi_mean / a.xi_shape)
    res = fit_emitters(xy, sig, xi, hierarchical=a.hierarchical, z=cols.get("z"), sigma_z=cols.get("sigma_z"),
                       n_burnin=a.n_burnin, n_trials=a.n_trials, roi_size_nm=a.roi_size, overlap_nm=a.overlap,
                       overlap_sigma_factor=a.overlap_sigma_factor,
                       se_adjust_nm=a.se_adjust, drift=a.drift, frames=cols.get("frame"),
                       processes=a.processes, seed=a.seed, matlab_compat=a.matlab_compat,
                       has_locs_min=a.has_locs_min, allocation=a.allocation, xi_counts=a.xi_counts)
    write_emitters_csv(res, a.out_prefix + "_emitters.csv")
    with open(a.out_prefix + "_labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "label"])
        for (xv, yv), lab in zip(xy, res.labels):
            w.writerow([repr(float(xv)), repr(float(yv)), int(lab)])
    print(f"{res.n_emitters} emitters ({res.n_emitters_with_locs} with localizations) from {xy.shape[0]} "
          f"localizations ({PUBLIC_LABEL}); xi {res.xi_kind} "
          f"{res.xi_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
