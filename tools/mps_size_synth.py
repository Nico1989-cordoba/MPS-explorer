# -*- coding: utf-8 -*-
"""
Synthetic clusters and rings with a KNOWN size and a KNOWN curvature, for the tests and the docs of the apparent-size
analysis (branch ``feat/apparent-size``; ``docs/apparent_size.md``). Synthetic data only (rule R8): nothing here
reads a file.

Ported from the apparent-size feasibility simulation (P-R61 measurement model, reused by the size simulation of
2026-10-15) so that the claims of ``docs/apparent_size.md`` can be re-checked from the repository:

* a cluster is seen through four layers: the FOOTPRINT (a disc of radius ``rho`` on the membrane, where the labels
  land at random), the LABELS (1-5 antibody stacks), the LINKAGE (a fixed offset per label, N(0, sigma_link) per
  axis) and the PRECISION (each localization N(0, sigma_loc) about its label, with a log-normal spread of the
  per-localization sigma). Each label blinks a geometric number of times (mean ``c_label``) and each blink leaves a
  geometric number of localizations (mean ``locs_per_event``);
* closed contours (circle, ellipse, ellipse with concave notches) tabulated densely with arc length, tangent,
  outward normal and the TRUE signed curvature (+ convex);
* cluster positions on the arc with a hard core (sequential rejection, as the project's simulator).

Every default carries its origin. No Qt.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SIGMA_LOC_NM", "SIGMA_LINK_NM", "C_LABEL_MEAN", "LOCS_PER_EVENT", "D_MIN_NM",
    "ClusterDraw", "draw_cluster", "ztp_lambda_for_mean", "draw_labels_mean",
    "Contour", "make_contour", "place_clusters", "SyntheticRing", "generate_ring",
]

# Per-localization precision of the synthetic file [lit: 8-20 nm, Huang et al. 2008; the NeNA of a dSTORM file is
# in that range, Endesfelder et al. 2014].
SIGMA_LOC_NM = 10.0
# Linkage error per axis of a primary + secondary IgG stack [lit: Frueh et al. 2021, ~19 nm maximum -> l / sqrt(3)
# per axis], FIXED for a label for the whole movie.
SIGMA_LINK_NM = 10.0
# Blinks (switching cycles) per label [lit: Dempsey et al. 2011, Alexa 647 in MEA; 26 in beta-ME].
C_LABEL_MEAN = 14.0
# Localizations per blink [ad hoc: the project's simulator default locs_per_fluor_mean].
LOCS_PER_EVENT = 2.0
# Hard core between cluster centres along the arc [ad hoc: the project's simulator hard core].
D_MIN_NM = 80.0


@dataclass
class ClusterDraw:
    """One synthetic cluster in its (tangent, normal) frame centred on the true centre: localization coordinates
    ``t``, ``n``; the blink (event) and the label of every localization; how many labels and events it has."""
    t: NDArray[np.float64]
    n: NDArray[np.float64]
    event: NDArray[np.int64]
    label: NDArray[np.int64]
    n_label: int
    n_events: int
    rho_true: float


def ztp_lambda_for_mean(m: float) -> float:
    """lambda of the zero-truncated Poisson whose mean is ``m`` (0 for m <= 1: exactly one label)."""
    if m <= 1.0 + 1e-9:
        return 0.0
    lo, hi = 1e-6, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mid / (1.0 - math.exp(-mid)) < m:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def draw_labels_mean(rng: np.random.Generator, means: NDArray[np.float64]) -> NDArray[np.int64]:
    """Labels per cluster: a zero-truncated Poisson with the given MEAN per cluster (mean 1 -> exactly 1)."""
    m = np.asarray(means, dtype=np.float64).reshape(-1)
    out = np.ones(m.size, dtype=np.int64)
    lam = np.array([ztp_lambda_for_mean(float(v)) for v in m])
    todo = lam > 0
    while todo.any():
        idx = np.flatnonzero(todo)
        k = rng.poisson(lam[idx])
        out[idx] = np.where(k > 0, k, out[idx])
        todo[idx[k > 0]] = False
    return out


def draw_cluster(rng: np.random.Generator, rho: float, n_label: int, *, c_label: float = C_LABEL_MEAN,
                 locs_per_event: float = LOCS_PER_EVENT, sigma_loc: float = SIGMA_LOC_NM,
                 sigma_loc_cv: float = 0.25, sigma_link: float = SIGMA_LINK_NM) -> ClusterDraw:
    """One cluster (module docstring). ``sigma_loc_cv``: log-normal spread of the per-localization sigma about
    ``sigma_loc`` [ad hoc: 25 %, as the P-R61 generator]; 0 gives every localization exactly ``sigma_loc``."""
    k = int(n_label)
    r = float(rho) * np.sqrt(rng.uniform(size=k))
    phi = rng.uniform(0.0, 2.0 * np.pi, k)
    ep_t = r * np.cos(phi)                      # the footprint lies along the membrane (tangent)
    link = rng.normal(0.0, sigma_link, (k, 2))
    lab_t = ep_t + link[:, 0]
    lab_n = link[:, 1]
    n_ev = rng.geometric(1.0 / c_label, k).astype(np.int64)
    n_events = int(n_ev.sum())
    ev_label = np.repeat(np.arange(k), n_ev)
    n_loc = rng.geometric(1.0 / locs_per_event, n_events).astype(np.int64)
    loc_event = np.repeat(np.arange(n_events), n_loc)
    loc_label = ev_label[loc_event]
    total = int(n_loc.sum())
    if sigma_loc_cv > 0:
        s2 = math.log1p(sigma_loc_cv ** 2)
        s = sigma_loc * np.exp(rng.normal(-0.5 * s2, math.sqrt(s2), total))
    else:
        s = np.full(total, float(sigma_loc))
    t = lab_t[loc_label] + rng.normal(size=total) * s
    n = lab_n[loc_label] + rng.normal(size=total) * s
    return ClusterDraw(t=t, n=n, event=loc_event.astype(np.int64), label=loc_label.astype(np.int64), n_label=k,
                       n_events=n_events, rho_true=float(rho))


@dataclass
class Contour:
    """A dense closed contour: vertices ``xy`` (M, 2), arc ``s`` (s[0] = 0), length ``L``, unit tangent, unit
    OUTWARD normal, TRUE signed curvature ``kappa`` (1/nm, + convex: the centre of curvature inside) and
    ``R`` = L / (2 pi), the calibre the curvature is normalised by (kappa R = 1 on a circle)."""
    xy: NDArray[np.float64]
    s: NDArray[np.float64]
    L: float
    tang: NDArray[np.float64]
    norm: NDArray[np.float64]
    kappa: NDArray[np.float64]
    R: float
    kind: str
    outward: float

    def at(self, s_query: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                                                         NDArray[np.float64], NDArray[np.float64]]:
        """Position, tangent, outward normal and curvature at arc positions (linear interpolation)."""
        sq = np.mod(np.asarray(s_query, dtype=np.float64), self.L)
        idx = np.clip(np.searchsorted(self.s, sq, side="right") - 1, 0, self.s.size - 1)
        nxt = (idx + 1) % self.s.size
        seg = np.where(nxt == 0, self.L - self.s[idx], self.s[nxt] - self.s[idx])
        f = np.clip((sq - self.s[idx]) / np.maximum(seg, 1e-12), 0.0, 1.0)[:, None]
        xy = (1 - f) * self.xy[idx] + f * self.xy[nxt]
        tg = (1 - f) * self.tang[idx] + f * self.tang[nxt]
        tg = tg / np.linalg.norm(tg, axis=1, keepdims=True)
        nm = np.column_stack([tg[:, 1], -tg[:, 0]]) * self.outward
        kp = (1 - f[:, 0]) * self.kappa[idx] + f[:, 0] * self.kappa[nxt]
        return xy, tg, nm, kp

    def kappa_at_points(self, xy: NDArray[np.float64]) -> NDArray[np.float64]:
        """TRUE curvature at the nearest contour vertex of each point (the truth a test compares with)."""
        from scipy.spatial import cKDTree
        _, idx = cKDTree(self.xy).query(np.asarray(xy, dtype=np.float64).reshape(-1, 2))
        return np.asarray(self.kappa[idx], dtype=np.float64)


def _polyline_geometry(xy: NDArray[np.float64], kind: str) -> Contour:
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    L = float(seg.sum())
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    ds = 0.5 * (seg + np.roll(seg, 1))
    dxy = (np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)) / (2.0 * ds[:, None])
    tang = dxy / np.linalg.norm(dxy, axis=1, keepdims=True)
    ddxy = (np.roll(xy, -1, axis=0) - 2 * xy + np.roll(xy, 1, axis=0)) / (ds[:, None] ** 2)
    cross = dxy[:, 0] * ddxy[:, 1] - dxy[:, 1] * ddxy[:, 0]
    kappa = cross / np.linalg.norm(dxy, axis=1) ** 3
    area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    outward = 1.0 if area > 0 else -1.0
    norm_v = np.column_stack([tang[:, 1], -tang[:, 0]]) * outward
    return Contour(xy=xy, s=s, L=L, tang=tang, norm=norm_v, kappa=kappa * outward, R=L / (2 * math.pi), kind=kind,
                   outward=outward)


def make_contour(kind: str, R_target: float, rng: Optional[np.random.Generator] = None, *, aspect: float = 1.0,
                 n_notch: int = 0, notch_depth: float = 0.2, notch_width: float = 0.5, M: int = 3000) -> Contour:
    """Closed contour of perimeter 2 pi ``R_target``: 'circle', 'ellipse' (``aspect`` = major / minor) or 'notched'
    (the ellipse with ``n_notch`` inward Gaussian dents of relative depth ``notch_depth`` and angular half-width
    ``notch_width`` rad: concave stretches, as a sectioning artefact leaves [ad hoc: depth 20 %, half-width 0.5 rad,
    the size simulation's scenario C])."""
    th = np.linspace(0.0, 2 * np.pi, M, endpoint=False)
    a = 1.0 if kind == "circle" else float(aspect)
    r_ell = a / np.sqrt(np.cos(th) ** 2 + (a * np.sin(th)) ** 2)
    if kind == "notched" and n_notch > 0:
        gen = rng if rng is not None else np.random.default_rng(0)
        centres = gen.uniform(0, 2 * np.pi, n_notch)
        dent = np.zeros(M)
        for c0 in centres:
            dth = np.angle(np.exp(1j * (th - c0)))
            dent += notch_depth * np.exp(-0.5 * (dth / notch_width) ** 2)
        r_ell = r_ell * (1.0 - dent)
    xy = np.column_stack([r_ell * np.cos(th), r_ell * np.sin(th)])
    c0 = _polyline_geometry(xy, kind)
    xy = xy * (2 * math.pi * R_target / c0.L)
    c1 = _polyline_geometry(xy, kind)
    xy2, _, _, _ = c1.at(np.asarray(np.linspace(0, c1.L, M, endpoint=False), dtype=np.float64))     # uniform in arc: clean differences
    return _polyline_geometry(xy2, kind)


def place_clusters(rng: np.random.Generator, L: float, K: int, d_min: float = D_MIN_NM) -> NDArray[np.float64]:
    """K sorted arc positions on a closed curve of length L with a hard core ``d_min`` (sequential rejection). K is
    capped at 0.6 L / d_min [lit: random sequential adsorption in 1D jams near 75 % coverage, Renyi 1958]."""
    K = min(int(K), int(0.6 * L / d_min))
    placed: List[float] = []
    for _ in range(K):
        for _ in range(20000):
            s = float(rng.uniform(0.0, L))
            if not placed:
                break
            d = np.abs(np.asarray(placed) - s)
            if np.all(np.minimum(d, L - d) >= d_min):
                break
        else:
            raise RuntimeError("hard core jammed")
        placed.append(s)
    return np.sort(np.asarray(placed, dtype=np.float64))


@dataclass
class SyntheticRing:
    """One synthetic ring: localizations ``x``, ``y`` (nm), a global ``event`` id per localization (blinks), the
    TRUE cluster of each localization (``truth``, -1 = background), and per true cluster its arc position,
    footprint, labels, events and localizations. ``contour`` is the truth the curvature is checked against."""
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    event: NDArray[np.int64]
    truth: NDArray[np.int64]
    s_true: NDArray[np.float64]
    rho: NDArray[np.float64]
    n_label: NDArray[np.int64]
    n_events: NDArray[np.int64]
    n_locs: NDArray[np.int64]
    contour: Contour


def generate_ring(rng: np.random.Generator, contour: Contour, *, K: Optional[int] = None, labels_mean: float = 3.0,
                  rho: float = 25.0, sigma_loc: float = SIGMA_LOC_NM, sigma_loc_cv: float = 0.25,
                  sigma_link: float = SIGMA_LINK_NM, c_label: float = C_LABEL_MEAN,
                  background_fraction: float = 0.0, bg_band_nm: float = 150.0,
                  d_min: float = D_MIN_NM, mean_spacing_nm: float = 200.0) -> SyntheticRing:
    """A ring of clusters on ``contour``: K ~ Poisson(L / ``mean_spacing_nm``) unless given [lit: mean spacing ~190-
    200 nm along the ring, Gazal et al. 2026], hard-core positions, labels zero-truncated Poisson of mean
    ``labels_mean``, footprint ``rho``, and an optional uniform background in a +/- ``bg_band_nm`` band (a fraction
    of the cluster localizations)."""
    k = int(rng.poisson(contour.L / mean_spacing_nm)) if K is None else int(K)
    s_true = place_clusters(rng, contour.L, max(k, 3), d_min)
    k = s_true.size
    labs = draw_labels_mean(rng, np.full(k, float(labels_mean)))
    xy_c, tg, nm, _ = contour.at(s_true)
    xs: List[NDArray[np.float64]] = []
    ys: List[NDArray[np.float64]] = []
    ev: List[NDArray[np.int64]] = []
    tru: List[NDArray[np.int64]] = []
    n_ev = np.zeros(k, dtype=np.int64)
    n_lc = np.zeros(k, dtype=np.int64)
    off = 0
    for i in range(k):
        c = draw_cluster(rng, rho, int(labs[i]), c_label=c_label, sigma_loc=sigma_loc, sigma_loc_cv=sigma_loc_cv,
                         sigma_link=sigma_link)
        xs.append(xy_c[i, 0] + c.t * tg[i, 0] + c.n * nm[i, 0])
        ys.append(xy_c[i, 1] + c.t * tg[i, 1] + c.n * nm[i, 1])
        ev.append(c.event + off)
        off += c.n_events
        tru.append(np.full(c.t.size, i, dtype=np.int64))
        n_ev[i] = c.n_events
        n_lc[i] = c.t.size
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    event = np.concatenate(ev)
    truth = np.concatenate(tru)
    n_bg = int(rng.poisson(background_fraction * x.size)) if background_fraction > 0 else 0
    if n_bg:
        sb = rng.uniform(0, contour.L, n_bg)
        rb = rng.uniform(-bg_band_nm, bg_band_nm, n_bg)
        xyb, _, nmb, _ = contour.at(sb)
        xyb = xyb + rb[:, None] * nmb
        x = np.concatenate([x, xyb[:, 0]])
        y = np.concatenate([y, xyb[:, 1]])
        event = np.concatenate([event, np.arange(n_bg, dtype=np.int64) + off])
        truth = np.concatenate([truth, np.full(n_bg, -1, dtype=np.int64)])
    return SyntheticRing(x=x, y=y, event=event, truth=truth, s_true=s_true, rho=np.full(k, float(rho)),
                         n_label=labs, n_events=n_ev, n_locs=n_lc, contour=contour)
