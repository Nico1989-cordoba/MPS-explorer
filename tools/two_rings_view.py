# -*- coding: utf-8 -*-
"""
Two rings unrolled with a hand-set tolerance +/- tau: the numbers behind the H-ECL window's "Two rings (unrolled)"
view (user request D1, 2026-10-23) -- without Qt.

What it is for
--------------
The user wanted to SEE what the eclipse tolerance does on real rings before accepting tau_0 (D1_ESTUDIO_TOLERANCIA
sec. 5): two consecutive rings laid flat along the membrane, a window of +/- tau that can be moved by hand, the
clusters painted when they have a partner inside it, the coincidences counted against what chance gives, and the
same pair read along the arc (the PRIMARY test's coordinate) and in the plane (the 2D sensitivity).

This is a VIEW. It reuses the tests' own coordinates and matchings -- the arcs of the arc test on the centroid
membrane (``CentroidMembrane.arcs_nm``, leave-one-cluster-out), ``tools.mps_matching.match_arcs`` (one-to-one,
maximum cardinality, then minimum total distance, cyclic) and ``match_rings`` in the plane -- but it is not a test:
"expected by chance" is the mean count over rotations of ring B (arc: B shifted uniformly along the membrane and
reflected with probability 1/2, as the arc test's null; plane: B rotated rigidly about its contour centroid), with
no joint null, no viability and no p.

Rule D4 (user decision, 2026-10-23): tau = tau_0 = 31 nm carries no mark; any other value is EXPLORATORY, on screen
and in the export (``tau_role``).

Numbers and their origin: 199 rotations [ad hoc: the arc test's default order of magnitude, enough for a mean to
1-2 %]; seed 0 [ad hoc: deterministic].

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

TAU0_NM = 31.0                 # [derivation: docs/eclipse_tolerance.md sec. 2; config/columns_params.yaml tau0_nm]
N_ROTATIONS = 199              # [ad hoc: see the module docstring]
SEED = 0                       # [ad hoc]
ROLE_PRIMARY = "primary (tau_0, pre-specified)"
ROLE_EXPLORATORY = "EXPLORATORY"

__all__ = ["TAU0_NM", "N_ROTATIONS", "ROLE_PRIMARY", "ROLE_EXPLORATORY", "tau_role", "PairData", "PairReading",
           "pairs_of", "pair_data", "read_pair", "export_rows", "EXPORT_COLUMNS"]


def tau_role(tau_nm: float, tau0_nm: float = TAU0_NM) -> str:
    """D4: the role of a tau -- primary at tau_0 (to 0.005 nm), EXPLORATORY otherwise."""
    return ROLE_PRIMARY if abs(float(tau_nm) - float(tau0_nm)) < 0.005 else ROLE_EXPLORATORY


@dataclass
class PairData:
    """Two consecutive rings as the view draws them: each cluster's arc on the centroid membrane (nm, in
    ``Ring.clusters`` order) and its centroid in the axon frame (nm), the membrane's length, and the rings' contour
    centroids (the plane's rotation centre of ring B)."""
    ring_a: int
    ring_b: int
    length_nm: float
    s_a: FloatArray
    s_b: FloatArray
    xy_a: FloatArray
    xy_b: FloatArray
    centre_b: FloatArray
    source: str = ""


@dataclass
class PairReading:
    """The pair at one tau: the matched clusters along the arc and in the plane, their counts, the counts expected
    by chance, and the tau's role (D4)."""
    tau_nm: float
    role: str
    arc_a: NDArray[np.intp]
    arc_b: NDArray[np.intp]
    arc_d_nm: FloatArray
    plane_a: NDArray[np.intp]
    plane_b: NDArray[np.intp]
    plane_d_nm: FloatArray
    n_arc: int
    n_plane: int
    expected_arc: float
    expected_plane: float
    expected_poisson: float
    n_rotations: int
    notes: List[str] = field(default_factory=list)

    @property
    def exploratory(self) -> bool:
        return self.role == ROLE_EXPLORATORY

    def sentence(self, data: PairData) -> str:
        k = min(data.s_a.size, data.s_b.size)
        mark = "" if not self.exploratory else "  [EXPLORATORY: tau is not tau_0 = 31 nm]"
        return (f"tau = {self.tau_nm:.1f} nm{mark}. Rings {data.ring_a}-{data.ring_b}: {data.s_a.size} and "
                f"{data.s_b.size} clusters on {data.length_nm / 1000.0:.2f} um. Along the arc {self.n_arc} of {k} "
                f"coincide (chance: {self.expected_arc:.1f} by rotation, {self.expected_poisson:.1f} by the Poisson "
                f"approximation); in the plane {self.n_plane} (chance: {self.expected_plane:.1f} by rotation). "
                "A view, not a test: no joint null, no viability, no p.")


def pairs_of(analyses: Any) -> List[Tuple[int, int]]:
    """The adjacent ring pairs of a run's arc test on the centroid membrane (``CleanedAnalyses.arc_centroid``)."""
    arc = getattr(analyses, "arc_centroid", None)
    if arc is None:
        return []
    return [(int(q.ring_a), int(q.ring_b)) for q in arc.adjacent]


def pair_data(analyses: Any, pair: Sequence[int]) -> PairData:
    """The two rings of ``pair`` from a finished run (``CleanedAnalyses``): arcs from the arc test's centroid
    membrane (the same coordinate the PRIMARY test matches on), centroids from the rings it ran on (the split rings
    under decision D2b). Raises ValueError when the run has no such pair."""
    arc = getattr(analyses, "arc_centroid", None)
    cm = None if arc is None else getattr(arc, "centroid_membrane", None)
    if cm is None:
        raise ValueError("this run has no arc test on the centroid membrane")
    a, b = int(pair[0]), int(pair[1])
    if a not in cm.arcs_nm or b not in cm.arcs_nm:
        raise ValueError(f"rings {a}-{b} are not in this run's arc test")
    rings_res = analyses.rings_of("arc") if hasattr(analyses, "rings_of") else analyses.rings
    rings = {int(r.index): r for r in rings_res.rings}

    def cents(ring: Any) -> FloatArray:
        return np.array([np.asarray(c.centroid_nm, dtype=np.float64).reshape(2) for c in ring.clusters],
                        dtype=np.float64).reshape(-1, 2)

    xy_a, xy_b = cents(rings[a]), cents(rings[b])
    s_a = np.asarray(cm.arcs_nm[a], dtype=np.float64).reshape(-1)
    s_b = np.asarray(cm.arcs_nm[b], dtype=np.float64).reshape(-1)
    if s_a.size != xy_a.shape[0] or s_b.size != xy_b.shape[0]:
        raise ValueError(f"rings {a}-{b}: the arcs and the clusters differ in number")
    contour_b = np.asarray(getattr(rings[b], "contour_nm", xy_b), dtype=np.float64).reshape(-1, 2)
    centre = contour_b.mean(axis=0) if contour_b.size else xy_b.mean(axis=0)
    return PairData(a, b, float(cm.length_nm), s_a, s_b, xy_a, xy_b, np.asarray(centre, dtype=np.float64),
                    source=f"arc test on the centroid membrane, {analyses.merged_handling if hasattr(analyses, 'merged_handling') else 'middle'} "
                           "for merged-looking clusters")


def read_pair(data: PairData, tau_nm: float, *, n_rotations: int = N_ROTATIONS, seed: int = SEED,
              tau0_nm: float = TAU0_NM) -> PairReading:
    """The pair at ``tau_nm`` (> 0): one-to-one matches along the arc and in the plane, and the counts by chance."""
    from tools.mps_matching import match_arcs, match_rings
    tau = float(tau_nm)
    if not (math.isfinite(tau) and tau > 0):
        raise ValueError(f"tau must be a positive number of nm, got {tau_nm!r}")
    L = float(data.length_nm)
    ia, jb, d = match_arcs(data.s_a, data.s_b, L, tau)
    pa, pb, pd = match_rings(data.xy_a, data.xy_b, tau)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 431]))
    n_arc_null = np.empty(int(n_rotations))
    n_plane_null = np.empty(int(n_rotations))
    rel = data.xy_b - data.centre_b
    for r in range(int(n_rotations)):
        u = float(rng.uniform(0.0, L))
        refl = bool(rng.random() < 0.5)
        s_b = np.mod((L - data.s_b if refl else data.s_b) + u, L)
        n_arc_null[r] = match_arcs(data.s_a, s_b, L, tau)[0].size
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        c, s = math.cos(th), math.sin(th)
        xy = data.centre_b + rel @ np.array([[c, s], [-s, c]])
        n_plane_null[r] = match_rings(data.xy_a, xy, tau)[0].size
    lam = data.s_b.size / L if L > 0 else 0.0
    poisson = float(data.s_a.size * (1.0 - math.exp(-lam * 2.0 * tau)))   # [derivation: D1 study sec. 1]
    notes = []
    if data.s_a.size == 0 or data.s_b.size == 0:
        notes.append("a ring has no cluster: nothing to match")
    return PairReading(tau, tau_role(tau, tau0_nm), np.asarray(ia, dtype=np.intp), np.asarray(jb, dtype=np.intp),
                       np.asarray(d, dtype=np.float64), np.asarray(pa, dtype=np.intp), np.asarray(pb, dtype=np.intp),
                       np.asarray(pd, dtype=np.float64), int(np.asarray(ia).size), int(np.asarray(pa).size),
                       float(n_arc_null.mean()), float(n_plane_null.mean()), poisson, int(n_rotations), notes)


EXPORT_COLUMNS: Tuple[str, ...] = (
    "source", "ring_a", "ring_b", "ring", "cluster", "arc_nm", "x_nm", "y_nm", "matched_arc", "partner_arc",
    "d_arc_nm", "matched_plane", "partner_plane", "d_plane_nm", "tau_nm", "tau_role", "n_arc", "n_plane",
    "expected_arc_rotation", "expected_plane_rotation", "expected_arc_poisson", "n_rotations", "membrane_length_nm")


def export_rows(data: PairData, reading: PairReading, source: str = "") -> List[Dict[str, Any]]:
    """One row per cluster of both rings (``EXPORT_COLUMNS``): its arc and centroid, whether and with whom it
    matched along the arc and in the plane, and the pair's tau with its role (D4: EXPLORATORY unless tau_0)."""
    pa = {int(i): (int(j), float(dd)) for i, j, dd in zip(reading.arc_a, reading.arc_b, reading.arc_d_nm)}
    pb = {int(j): (int(i), float(dd)) for i, j, dd in zip(reading.arc_a, reading.arc_b, reading.arc_d_nm)}
    qa = {int(i): (int(j), float(dd)) for i, j, dd in zip(reading.plane_a, reading.plane_b, reading.plane_d_nm)}
    qb = {int(j): (int(i), float(dd)) for i, j, dd in zip(reading.plane_a, reading.plane_b, reading.plane_d_nm)}
    rows: List[Dict[str, Any]] = []
    common = {"source": source, "ring_a": data.ring_a, "ring_b": data.ring_b, "tau_nm": round(reading.tau_nm, 2),
              "tau_role": reading.role, "n_arc": reading.n_arc, "n_plane": reading.n_plane,
              "expected_arc_rotation": round(reading.expected_arc, 3),
              "expected_plane_rotation": round(reading.expected_plane, 3),
              "expected_arc_poisson": round(reading.expected_poisson, 3), "n_rotations": reading.n_rotations,
              "membrane_length_nm": round(data.length_nm, 1)}
    for ring, s, xy, am, qm in ((data.ring_a, data.s_a, data.xy_a, pa, qa), (data.ring_b, data.s_b, data.xy_b, pb, qb)):
        for k in range(s.size):
            arc_m, plane_m = am.get(k), qm.get(k)
            row = dict(common, ring=ring, cluster=k, arc_nm=round(float(s[k]), 1), x_nm=round(float(xy[k, 0]), 1),
                       y_nm=round(float(xy[k, 1]), 1), matched_arc=arc_m is not None,
                       partner_arc=None if arc_m is None else arc_m[0],
                       d_arc_nm=None if arc_m is None else round(arc_m[1], 1), matched_plane=plane_m is not None,
                       partner_plane=None if plane_m is None else plane_m[0],
                       d_plane_nm=None if plane_m is None else round(plane_m[1], 1))
            rows.append({c: row.get(c) for c in EXPORT_COLUMNS})
    return rows
