# -*- coding: utf-8 -*-
"""
SiZer on the axial (z') profile of an axon: significant peaks and valleys
with a cluster-robust variance (D-41, criteria 1 and 2 of the ring-pair
viability rule v2).

SiZer (Chaudhuri & Marron 1999, JASA 94:807-823) smooths the profile with
a Gaussian kernel of width h and asks, at every point of a grid, whether
the SLOPE of the smoothed density is significantly positive, negative or
neither. A significant PEAK is a run of significant rises followed (next
run of non-zero sign) by a run of significant falls; a significant VALLEY
is a fall followed by a rise -- literally "the derivative changes sign",
with a test.

The variance of the slope counts every CLUSTER as one unit (the
cluster-robust / sandwich variance, Liang & Zeger 1986; Cameron & Miller
2015): the localizations of one fluorophore cluster share their z and are
not independent. The groups are 3D DBSCAN clusters of the localizations
(``groups_3d``: eps 25 nm, z scaled by 0.5, min_samples 10; every noise
point is its own group). Significance uses the simultaneous quantile of
Chaudhuri & Marron, q = Phi^-1((1 + (1 - alpha)^(1/l)) / 2), with l = n /
mean effective sample size over the grid points where it is >= 5.

Port of the H5-E zmin research code (private research scripts:
``sizer_row`` / ``peaks_valleys``, ``sign_row`` /
``valley_between``, ``groups_3d``; the same numbers to 1e-12 on the
fixed inputs of ``testdata/viability_v2/sizer_parity.json``). Nothing is
drawn: every function is deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.stats import norm

__all__ = [
    "SQ2PI",
    "SIZER_H_OVER_P",
    "SIZER_ALPHA",
    "SIZER_MIN_ESS",
    "GROUP_EPS_NM",
    "GROUP_MIN_SAMPLES",
    "GROUP_Z_SCALE",
    "SizerRow",
    "SizerProfile",
    "sizer_grid",
    "sizer_row",
    "peaks_valleys",
    "sizer_profile",
    "valley_between",
    "groups_3d",
]

SQ2PI = math.sqrt(2.0 * math.pi)
# D-41: kernel h = 0.10 P, alpha 0.05 (simultaneous), the zmin report's
# recommended variant ("cl_0.1").
SIZER_H_OVER_P = 0.10
SIZER_ALPHA = 0.05
# A grid point whose effective sample size sum K_h(x - X_i) / K_h(0) is
# below this is never significant (Chaudhuri & Marron's rule of thumb).
SIZER_MIN_ESS = 5.0
# The 3D DBSCAN that defines the clusters of the variance (zmin part B/C).
GROUP_EPS_NM = 25.0
GROUP_MIN_SAMPLES = 10
GROUP_Z_SCALE = 0.5


@dataclass(frozen=True)
class SizerRow:
    """One SiZer row: the sign of the significant slope at every grid
    point (+1 / -1 / 0), the smoothed density, the slope t statistic, the
    simultaneous quantile q and the number of independent blocks l."""

    sign: NDArray[np.int64]
    fhat: NDArray[np.float64]
    t: NDArray[np.float64]
    q: float
    l: float


@dataclass(frozen=True)
class SizerProfile:
    """The SiZer reading of one z' profile at one bandwidth: the grid, the
    sign row (with ``q_override`` when a pointwise quantile replaced the
    simultaneous one), the smoothed density, the significant peaks and
    valleys (positions in nm, at the density maximum / minimum between the
    two runs) and the settings."""

    grid: NDArray[np.float64]
    sign: NDArray[np.int64]
    fhat: NDArray[np.float64]
    t: NDArray[np.float64]
    q: float
    l: float
    h_nm: float
    alpha: float
    robust: bool
    q_override: Optional[float]
    peaks: Tuple[float, ...]
    valleys: Tuple[float, ...]


def sizer_grid(z: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """The evaluation grid: from the 0.2 to the 99.8 percentile of z, 2 h
    beyond each end, step max(h / 4, 2 nm)."""
    step = max(h / 4.0, 2.0)
    lo, hi = np.percentile(np.asarray(z, dtype=np.float64), [0.2, 99.8])
    return np.asarray(np.arange(lo - 2 * h, hi + 2 * h + step, step), dtype=np.float64)


def sizer_row(z: NDArray[np.float64], h: float, grid: NDArray[np.float64], groups: Optional[NDArray[np.int64]],
              alpha: float, robust: bool = True, block: int = 64) -> SizerRow:
    """
    The SiZer sign row of the kernel density of ``z`` at bandwidth ``h``.

    f'_h(x) = mean_i d/dx K_h(x - X_i). Its variance is the iid one,
    (mean D^2 - f'^2) / n, or, with ``robust`` and ``groups``, the
    cluster-robust one, sum over groups of (S_g - m_g f')^2 / n^2 with S_g
    the group's summed kernel derivative and m_g its size. A grid point is
    significant when |f'| / sd > q; points with an effective sample size
    below ``SIZER_MIN_ESS`` are set to 0.
    """
    z = np.asarray(z, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n = z.size
    G = grid.size
    fd = np.zeros(G)
    fh = np.zeros(G)
    var = np.zeros(G)
    ess = np.zeros(G)
    M = None
    m_g = None
    if robust and groups is not None:
        g = np.unique(np.asarray(groups), return_inverse=True)[1]
        M = sparse.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(int(g.max()) + 1, n))
        m_g = np.asarray(M.sum(axis=1)).ravel()
    for s in range(0, G, block):
        u = (grid[s:s + block, None] - z[None, :]) / h
        ph = np.exp(-0.5 * u * u)
        D = -u * ph / (SQ2PI * h * h)          # d/dx of K_h(x - X_i)
        fd[s:s + block] = D.mean(axis=1)
        fh[s:s + block] = ph.sum(axis=1) / (SQ2PI * h * n)
        ess[s:s + block] = ph.sum(axis=1)       # sum K_h(x - X_i) / K_h(0)
        if M is not None:
            assert m_g is not None
            S = (M @ D.T)                       # groups x block
            r = S - m_g[:, None] * fd[None, s:s + block]
            var[s:s + block] = (r * r).sum(axis=0) / (n * n)
        else:
            var[s:s + block] = ((D * D).mean(axis=1) - fd[s:s + block] ** 2) / n
    ok = ess >= SIZER_MIN_ESS
    l_blocks = n / max(float(np.mean(ess[ok])), 1e-9) if ok.any() else 1.0
    l_blocks = max(l_blocks, 1.0)
    q = float(norm.ppf((1.0 + (1.0 - alpha) ** (1.0 / l_blocks)) / 2.0))
    t = fd / np.sqrt(np.maximum(var, 1e-300))
    sign = np.where(t > q, 1, np.where(t < -q, -1, 0))
    sign[~ok] = 0
    return SizerRow(sign=np.asarray(sign, dtype=np.int64), fhat=fh, t=t, q=q, l=float(l_blocks))


def peaks_valleys(sign: NDArray[np.int64], fhat: NDArray[np.float64],
                  grid: NDArray[np.float64]) -> Tuple[List[float], List[float]]:
    """Significant peaks (a '+' run then a '-' run, zeros skipped) and
    valleys ('-' then '+'), placed at the density maximum / minimum
    between the end of the first run and the start of the second."""
    idx = np.flatnonzero(sign != 0)
    peaks: List[float] = []
    valleys: List[float] = []
    if idx.size == 0:
        return peaks, valleys
    s = sign[idx]
    brk = np.flatnonzero(np.diff(s) != 0)
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, s.size - 1]
    for a in range(starts.size - 1):
        s1, s2 = s[starts[a]], s[starts[a + 1]]
        i0, i1 = idx[ends[a]], idx[starts[a + 1]]
        seg = slice(i0, i1 + 1)
        if s1 > 0 and s2 < 0:
            peaks.append(float(grid[i0 + int(np.argmax(fhat[seg]))]))
        elif s1 < 0 and s2 > 0:
            valleys.append(float(grid[i0 + int(np.argmin(fhat[seg]))]))
    return peaks, valleys


def sizer_profile(z: NDArray[np.float64], groups: Optional[NDArray[np.int64]], *, p_ref_nm: float,
                  c: float = SIZER_H_OVER_P, alpha: float = SIZER_ALPHA, robust: bool = True,
                  q: Optional[float] = None) -> SizerProfile:
    """
    SiZer of a z' profile at h = c * ``p_ref_nm`` on ``sizer_grid``
    (port of ``zm_real.sign_row``). ``robust`` uses the cluster-robust
    variance over ``groups``; ``q`` replaces the simultaneous quantile by a
    pointwise one (a liberal reading kept for comparison, never the rule).
    """
    z = np.asarray(z, dtype=np.float64)
    h = c * float(p_ref_nm)
    grid = sizer_grid(z, h)
    r = sizer_row(z, h, grid, groups if robust else None, alpha, robust=robust)
    sg = r.sign
    if q is not None:
        sg = np.where(r.t > q, 1, np.where(r.t < -q, -1, 0)).astype(np.int64)
        sg[(r.fhat * z.size * SQ2PI * h) < SIZER_MIN_ESS] = 0
    pk, vl = peaks_valleys(sg, r.fhat, grid)
    return SizerProfile(grid=grid, sign=np.asarray(sg, dtype=np.int64), fhat=r.fhat, t=r.t, q=r.q, l=r.l, h_nm=h,
                        alpha=float(alpha), robust=bool(robust), q_override=q, peaks=tuple(pk), valleys=tuple(vl))


def valley_between(grid: NDArray[np.float64], sign: NDArray[np.int64], a: float, b: float) -> bool:
    """A significant valley strictly between ``a`` and ``b`` (two ring
    centres, a < b): some significant fall before some significant rise in
    the open interval."""
    m = (np.asarray(grid) > a) & (np.asarray(grid) < b)
    s = np.asarray(sign)[m]
    neg = np.flatnonzero(s < 0)
    pos = np.flatnonzero(s > 0)
    return bool(neg.size and pos.size and neg.min() < pos.max())


def groups_3d(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64], eps: float = GROUP_EPS_NM,
              ms: int = GROUP_MIN_SAMPLES, zscale: float = GROUP_Z_SCALE) -> NDArray[np.int64]:
    """The clusters of the cluster-robust variance: DBSCAN on (x, y,
    zscale * z) with ``eps`` and ``ms``; every noise point becomes its own
    group, numbered after the clusters."""
    from sklearn.cluster import DBSCAN
    lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(np.column_stack([x, y, np.asarray(z, dtype=np.float64) * zscale]))
    g = np.asarray(lab, dtype=np.int64).copy()
    noise = g < 0
    g[noise] = g.max() + 1 + np.arange(int(noise.sum()))
    return np.asarray(g, dtype=np.int64)
