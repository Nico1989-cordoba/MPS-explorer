# -*- coding: utf-8 -*-
"""
The k-distance graph of DBSCAN and its valley: how "eps = auto" works (decision P-R58, docs/eps_kdistance.md).

What it is
----------
Typing "auto" for eps never picks a number by itself. It opens the k-distance graph of the localizations the
clustering would run on, marks the VALLEY of the k-distance density as a suggestion -- with the eps it implies, its
bootstrap interval and the share of localizations DBSCAN would leave unclustered there -- draws the published value
(25 nm, Gazal et al. 2026) as a reference line, and lets the person set eps by clicking on the graph. When the density
has no valley, it says so and suggests nothing: there is no fallback number.

The convention
--------------
scikit-learn's ``min_samples`` counts the point itself (a core point has at least ``min_samples`` points, itself
included, within eps). The k-distance that matches it is the distance to the (min_samples - 1)-th OTHER point:
``k = min_samples - 1`` (Ester et al. 1996 define k-dist as the distance to the k-th nearest neighbour; Schubert et
al. 2017, footnote 4: "k corresponds to minPts = k + 1"). Then a localization is a core point exactly when its
k-distance is <= eps, so the share of k-distances above eps is the share of localizations that are NOT core points;
the share DBSCAN leaves unclustered is smaller (border points join a cluster) and is computed by running DBSCAN.

The valley
----------
On u = log10(k-distance) (for a homogeneous Poisson background the k-th neighbour distance scales as density^-1/2,
so on a log scale the background mode keeps its shape whatever its density and only moves), a Gaussian kernel density
estimate with Silverman's rule-of-thumb bandwidth (Silverman 1986, eq. 3.31; it over-smooths a multimodal density, so
it hides weak modes rather than inventing them) is evaluated on a grid (linear binning, Wand 1994). Its local maxima
are the modes; the valley is the lowest point between the two most prominent ones. A valley is REPORTED only when

1. the dip is larger than the estimate's own noise: the lower of the two peaks exceeds the valley by more than
   ``DIP_Z`` standard errors, var f(u) ~ f(u) R(K) / (n h) with R(K) = 1 / (2 sqrt(pi)) for the Gaussian kernel
   (Silverman 1986, section 3.3.1), and
2. it is found again, the same way, in at least ``PRESENCE_LEVEL`` of ``BOOTSTRAP_N`` bootstrap resamples of the
   k-distances (multinomial on the bins: the same estimate, resampled).

The bootstrap also gives the uncertainty of the suggestion: the 2.5-97.5 % interval of the valley's eps over the
resamples that have one. Why a density valley and not the knee of the sorted curve (Kneedle, Satopaa et al. 2011):
docs/eps_kdistance.md and docs/sims/kdist/.

The record
----------
How eps was chosen is kept (settings ``eps_choice``) and written beside every export (``EPS_CHOICE_COLUMNS``): the
published default, typed, the valley suggestion accepted, or a click on the graph, with k, the suggestion and its
interval, the share left unclustered at the chosen eps and the file the graph was drawn on. A record describes ONE
value of eps: when eps is later set some other way, ``export_columns`` no longer applies it.

No Qt here (the dialog is ``tools.mps_kdist_dialog``).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "GAZAL_EPS_NM", "BOOTSTRAP_N", "PRESENCE_LEVEL", "DIP_Z", "GRID_POINTS", "BOOTSTRAP_SEED",
    "EPS_SOURCES", "EPS_SOURCE_WORDS", "EPS_CHOICE_COLUMNS", "Valley", "k_rank", "k_distances", "sorted_curve",
    "noncore_fraction", "unclustered_fraction", "find_valley", "choice_record", "clean_record", "export_columns",
    "source_words", "valley_words",
]

from tools.mps_settings import DEFAULT_EPS_NM

# The published value the graph draws as a reference line (Gazal et al. 2026): the analysis' default, one constant.
GAZAL_EPS_NM = float(DEFAULT_EPS_NM)
# Bootstrap resamples of the k-distances for the valley's presence and interval. With 200, the Monte Carlo standard
# error of a share near 0.95 is sqrt(0.95 x 0.05 / 200) = 0.015 (docs/eps_kdistance.md).
BOOTSTRAP_N = 200
# Share of resamples that must show the valley for it to be reported (a conventional 95 %).
PRESENCE_LEVEL = 0.95
# How many standard errors of the kernel estimate the dip must exceed (two-sided 95 % of a normal).
DIP_Z = 1.96
# Grid of the density estimate (linear binning; Wand 1994 finds 400-500 points ample for a univariate estimate).
GRID_POINTS = 512
# The bootstrap is seeded, so the same localizations give the same suggestion and interval every time.
BOOTSTRAP_SEED = 0

_R_GAUSS = 1.0 / (2.0 * math.sqrt(math.pi))   # R(K) = integral of K^2 for the Gaussian kernel

# How eps was chosen. Stable words: they are written into exported tables.
EPS_SOURCES: Tuple[str, ...] = ("gazal_default", "typed", "kdist_valley", "kdist_click")
EPS_SOURCE_WORDS: Dict[str, str] = {
    "gazal_default": "the published value (Gazal et al. 2026)",
    "typed": "typed",
    "kdist_valley": "the k-distance valley, suggestion accepted",
    "kdist_click": "chosen by clicking on the k-distance graph",
}

# What the per-axon table carries about it, written once at the end of the row.
EPS_CHOICE_COLUMNS: Tuple[str, ...] = (
    "eps_source", "eps_kdist_k", "eps_kdist_valley_nm", "eps_kdist_valley_low_nm", "eps_kdist_valley_high_nm",
    "eps_kdist_unclustered_fraction", "eps_kdist_file")


# --------------------------------------------------------------------------- the graph
def k_rank(min_samples: int) -> int:
    """The neighbour rank of the k-distance that matches ``min_samples``: min_samples - 1 OTHER points."""
    m = int(min_samples)
    if m < 2:
        raise ValueError(f"min samples {m} has no k-distance: it needs at least 2 (k = min samples - 1 >= 1)")
    return m - 1


def k_distances(x_nm: Any, y_nm: Any, min_samples: int) -> np.ndarray:
    """Each localization's distance (nm) to its (min_samples - 1)-th nearest other localization, in input order.

    Queried with the point itself in the set (column 0, distance 0), so column ``min_samples - 1`` is the
    (min_samples - 1)-th other point; exact duplicates are other points at distance 0, as DBSCAN counts them.
    """
    from scipy.spatial import cKDTree

    pts = np.column_stack([np.asarray(x_nm, dtype=np.float64).ravel(), np.asarray(y_nm, dtype=np.float64).ravel()])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    m = int(min_samples)
    k_rank(m)
    if pts.shape[0] < m:
        raise ValueError(f"{pts.shape[0]} localizations: fewer than min samples ({m}), so no point has a "
                         f"k-distance")
    dist, _ = cKDTree(pts).query(pts, k=m)
    return np.asarray(dist[:, m - 1], dtype=np.float64)


def sorted_curve(kdist_nm: np.ndarray) -> np.ndarray:
    """The k-distance graph as drawn: the k-distances in increasing order."""
    return np.sort(np.asarray(kdist_nm, dtype=np.float64))


def noncore_fraction(kdist_nm: np.ndarray, eps_nm: float) -> float:
    """Share of localizations whose k-distance exceeds eps: the ones that are NOT core points at this eps."""
    kd = np.asarray(kdist_nm, dtype=np.float64)
    if kd.size == 0:
        return float("nan")
    return float(np.count_nonzero(kd > float(eps_nm)) / kd.size)


def unclustered_fraction(x_nm: Any, y_nm: Any, eps_nm: float, min_samples: int) -> float:
    """Share of localizations DBSCAN(eps, min_samples) labels noise, before any curation: run, not estimated."""
    from sklearn.cluster import DBSCAN

    pts = np.column_stack([np.asarray(x_nm, dtype=np.float64).ravel(), np.asarray(y_nm, dtype=np.float64).ravel()])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] == 0:
        return float("nan")
    labels = DBSCAN(eps=float(eps_nm), min_samples=int(min_samples)).fit_predict(pts)
    return float(np.count_nonzero(labels == -1) / labels.size)


# --------------------------------------------------------------------------- the valley
@dataclass
class Valley:
    """What the k-distance density says about eps. ``found`` False: no valley, no suggestion (``reason`` says why)."""

    found: bool
    eps_nm: float = float("nan")
    low_nm: float = float("nan")
    high_nm: float = float("nan")
    presence: float = float("nan")          # share of bootstrap resamples with the valley
    n_modes: int = 0
    left_mode_nm: float = float("nan")
    right_mode_nm: float = float("nan")
    dip_z: float = float("nan")             # (lower peak - valley) / its standard error
    reason: str = ""
    other_valleys_nm: Tuple[float, ...] = ()
    bandwidth_log10: float = float("nan")
    n: int = 0
    n_zero: int = 0                         # k-distances of 0 (exact duplicates), left out of the log density
    grid_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    density: np.ndarray = field(default_factory=lambda: np.zeros(0))   # per unit of log10(nm)

    def words(self) -> str:
        return valley_words(self)


def _bandwidth(u: np.ndarray) -> float:
    """Silverman's rule of thumb, 0.9 min(sd, IQR / 1.34) n^(-1/5) (Silverman 1986, eq. 3.31)."""
    n = u.size
    sd = float(np.std(u, ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(u, [75.0, 25.0])
    spread = min(sd, float(q75 - q25) / 1.34) if q75 > q25 else sd
    if not spread > 0:
        return float("nan")
    return float(0.9 * spread * float(n) ** (-0.2))


def _binned_density(counts: np.ndarray, step: float, h: float) -> np.ndarray:
    """Gaussian KDE from bin counts on a regular grid (linear binning already done), per unit of u."""
    n = float(counts.sum())
    half = int(math.ceil(5.0 * h / step))
    t = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (t / h) ** 2) / (h * math.sqrt(2.0 * math.pi))
    return np.convolve(counts, kernel, mode="same") / n


def _linear_binning(u: np.ndarray, grid: np.ndarray) -> np.ndarray:
    step = grid[1] - grid[0]
    pos = (u - grid[0]) / step
    i = np.clip(np.floor(pos).astype(np.int64), 0, grid.size - 2)
    w = np.clip(pos - i, 0.0, 1.0)
    counts = np.bincount(i, weights=1.0 - w, minlength=grid.size)
    counts += np.bincount(i + 1, weights=w, minlength=grid.size)
    return counts


def _modes_and_valley(density: np.ndarray, n: int, h: float) -> Tuple[int, Optional[int], float, Tuple[int, int],
                                                                       Tuple[int, ...]]:
    """(number of modes, valley index or None, dip z, (left mode, right mode), other minima between modes)."""
    d = density
    inner = np.arange(1, d.size - 1)
    is_max = (d[inner] > d[inner - 1]) & (d[inner] >= d[inner + 1])
    maxima = inner[is_max]
    if maxima.size < 2:
        return int(maxima.size), None, float("nan"), (-1, -1), ()
    # Topographic prominence of each maximum: its height above the higher of the two lowest points that separate it
    # from a higher maximum (or from the ends).
    prom = np.empty(maxima.size)
    for j, p in enumerate(maxima):
        left_higher = maxima[:j][d[maxima[:j]] > d[p]]
        right_higher = maxima[j + 1:][d[maxima[j + 1:]] > d[p]]
        lo_left = float(d[left_higher[-1]:p + 1].min()) if left_higher.size else float(d[:p + 1].min())
        lo_right = float(d[p:right_higher[0] + 1].min()) if right_higher.size else float(d[p:].min())
        prom[j] = d[p] - max(lo_left, lo_right)
    top = np.sort(maxima[np.argsort(prom)[-2:]])
    a, b = int(top[0]), int(top[1])
    v = a + int(np.argmin(d[a:b + 1]))
    peak = min(float(d[a]), float(d[b]))
    se = math.sqrt(max(peak, 0.0) * _R_GAUSS / (n * h) + max(float(d[v]), 0.0) * _R_GAUSS / (n * h))
    z = (peak - float(d[v])) / se if se > 0 else float("nan")
    others = []
    for left, right in zip(maxima[:-1], maxima[1:]):
        w = int(left) + int(np.argmin(d[int(left):int(right) + 1]))
        if w != v:
            others.append(w)
    return int(maxima.size), v, z, (a, b), tuple(others)


def find_valley(kdist_nm: Any, *, n_boot: int = BOOTSTRAP_N, level: float = PRESENCE_LEVEL, dip_z: float = DIP_Z,
                seed: int = BOOTSTRAP_SEED, grid_points: int = GRID_POINTS) -> Valley:
    """The valley of the log k-distance density, or why there is none (module docstring)."""
    kd = np.asarray(kdist_nm, dtype=np.float64).ravel()
    kd = kd[np.isfinite(kd)]
    n_zero = int(np.count_nonzero(kd <= 0))
    u = np.log10(kd[kd > 0])
    n = int(u.size)
    if n < 20:
        return Valley(found=False, n=n, n_zero=n_zero,
                      reason=f"only {n} positive k-distances: too few for a density estimate")
    h = _bandwidth(u)
    if not (h > 0 and math.isfinite(h)):
        return Valley(found=False, n=n, n_zero=n_zero, reason="all k-distances are equal: no density to read")
    lo, hi = float(u.min()) - 3.0 * h, float(u.max()) + 3.0 * h
    grid = np.linspace(lo, hi, int(grid_points))
    step = float(grid[1] - grid[0])
    counts = _linear_binning(u, grid)
    density = _binned_density(counts, step, h)
    n_modes, v, z, (a, b), others = _modes_and_valley(density, n, h)
    base: Dict[str, Any] = dict(n=n, n_zero=n_zero, n_modes=n_modes, bandwidth_log10=h, grid_nm=10.0 ** grid,
                                density=density)
    if v is None:
        return Valley(found=False, reason="the k-distance density has a single mode: no valley", **base)
    eps = float(10.0 ** grid[v])
    detail: Dict[str, Any] = dict(base, left_mode_nm=float(10.0 ** grid[a]), right_mode_nm=float(10.0 ** grid[b]), dip_z=z,
                  other_valleys_nm=tuple(float(10.0 ** grid[w]) for w in others))
    if not (z > dip_z):
        return Valley(found=False, eps_nm=eps,
                      reason=(f"the dip between the two main modes is within the estimate's noise ({z:.1f} standard "
                              f"errors, {dip_z:g} needed): no valley"), **detail)
    # Bootstrap: the same estimate on resampled k-distances (multinomial on the linear-binning weights).
    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    p = counts / counts.sum()
    found_eps = []
    for _ in range(int(n_boot)):
        c = rng.multinomial(n, p).astype(np.float64)
        d = _binned_density(c, step, h)
        _, vb, zb, _, _ = _modes_and_valley(d, n, h)
        if vb is not None and zb > dip_z:
            found_eps.append(float(10.0 ** grid[vb]))
    presence = len(found_eps) / float(n_boot) if n_boot > 0 else float("nan")
    if found_eps:
        low, high = (float(q) for q in np.percentile(found_eps, [2.5, 97.5]))
    else:
        low = high = float("nan")
    if not presence >= level:
        return Valley(found=False, eps_nm=eps, low_nm=low, high_nm=high, presence=presence,
                      reason=(f"the valley at {eps:.1f} nm is found again in only {100 * presence:.0f} % of "
                              f"{n_boot} bootstrap resamples ({100 * level:.0f} % needed): no stable valley"),
                      **detail)
    return Valley(found=True, eps_nm=eps, low_nm=low, high_nm=high, presence=presence, **detail)


def valley_words(valley: Valley) -> str:
    """One sentence for the dialog and the log."""
    if not valley.found:
        return f"No valley: {valley.reason.split(': no ')[0]}." if ": no " in valley.reason else \
            f"No valley ({valley.reason})."
    text = (f"Valley at {valley.eps_nm:.1f} nm (95 % bootstrap interval {valley.low_nm:.1f}-{valley.high_nm:.1f} nm; "
            f"found in {100 * valley.presence:.0f} % of {BOOTSTRAP_N} resamples)")
    if valley.n_modes > 2:
        text += (f"; the density has {valley.n_modes} modes, the valley shown lies between the two most prominent"
                 f" (others at {', '.join(f'{e:.0f}' for e in valley.other_valleys_nm)} nm)")
    return text + "."


# --------------------------------------------------------------------------- the record
def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def choice_record(source: str, eps_nm: float, min_samples: int, *, valley: Optional[Valley] = None,
                  unclustered: Optional[float] = None, file_name: str = "", n_locs: int = 0) -> Dict[str, Any]:
    """The settings' record of how eps was chosen on the k-distance graph (``source`` in EPS_SOURCES)."""
    if source not in EPS_SOURCES:
        raise ValueError(f"unknown eps source {source!r}; known: {EPS_SOURCES}")
    rec: Dict[str, Any] = {"source": source, "eps_nm": float(eps_nm), "k": k_rank(min_samples),
                           "min_samples": int(min_samples), "file": str(file_name), "n_locs": int(n_locs)}
    if valley is not None:
        rec["valley_found"] = bool(valley.found)
        if valley.found:
            rec.update(valley_nm=float(valley.eps_nm), valley_low_nm=float(valley.low_nm),
                       valley_high_nm=float(valley.high_nm), valley_presence=float(valley.presence))
    if unclustered is not None and _number(unclustered) is not None:
        rec["unclustered_fraction"] = float(unclustered)
    return rec


_NUMBERS = ("eps_nm", "valley_nm", "valley_low_nm", "valley_high_nm", "valley_presence", "unclustered_fraction")
_INTS = ("k", "min_samples", "n_locs")


def clean_record(entry: Any) -> Dict[str, Any]:
    """A stored record, cleaned: {} when it is not a usable one (then eps reads as typed or the default)."""
    if not isinstance(entry, dict) or entry.get("source") not in EPS_SOURCES:
        return {}
    out: Dict[str, Any] = {"source": str(entry["source"])}
    for key in _NUMBERS:
        v = _number(entry.get(key))
        if v is not None:
            out[key] = v
    for key in _INTS:
        v = _number(entry.get(key))
        if v is not None and float(v).is_integer():
            out[key] = int(v)
    if "eps_nm" not in out:
        return {}
    if isinstance(entry.get("file"), str):
        out["file"] = entry["file"]
    if isinstance(entry.get("valley_found"), bool):
        out["valley_found"] = entry["valley_found"]
    return out


def _same_eps(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))


def export_columns(record: Any, eps_nm: float, default_eps_nm: float = GAZAL_EPS_NM) -> Dict[str, Any]:
    """The EPS_CHOICE_COLUMNS of an analysis run with ``eps_nm``.

    The record applies only when it describes this very eps; otherwise eps was set another way since, and reads as
    the published default when it equals it, typed when not.
    """
    rec = clean_record(record)
    out: Dict[str, Any] = {name: None for name in EPS_CHOICE_COLUMNS}
    if rec and _same_eps(rec["eps_nm"], eps_nm):
        out.update({
            "eps_source": rec["source"],
            "eps_kdist_k": rec.get("k"),
            "eps_kdist_valley_nm": rec.get("valley_nm"),
            "eps_kdist_valley_low_nm": rec.get("valley_low_nm"),
            "eps_kdist_valley_high_nm": rec.get("valley_high_nm"),
            "eps_kdist_unclustered_fraction": rec.get("unclustered_fraction"),
            "eps_kdist_file": rec.get("file") or None,
        })
        return out
    out["eps_source"] = "gazal_default" if _same_eps(eps_nm, default_eps_nm) else "typed"
    return out


def source_words(columns: Dict[str, Any]) -> str:
    """The provenance of eps in words, from ``export_columns``."""
    src = str(columns.get("eps_source") or "")
    words = EPS_SOURCE_WORDS.get(src, src)
    if src in ("kdist_valley", "kdist_click") or columns.get("eps_kdist_k") is not None:
        parts = [f"k = {columns.get('eps_kdist_k')}"]
        if columns.get("eps_kdist_valley_nm") is not None:
            parts.append(f"valley {float(columns['eps_kdist_valley_nm']):.1f} nm")
        else:
            parts.append("no valley")
        if columns.get("eps_kdist_unclustered_fraction") is not None:
            parts.append(f"{100 * float(columns['eps_kdist_unclustered_fraction']):.1f} % unclustered")
        words += " (" + ", ".join(parts) + ")"
    return words

