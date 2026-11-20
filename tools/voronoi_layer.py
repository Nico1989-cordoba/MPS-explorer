# -*- coding: utf-8 -*-
"""
What the axon map's Voronoi layer draws, as data (no widgets): the cells of a ``voronoi_factor`` / ``voronoi_csr`` analysis,
painted by their density on a sequential scale, with the threshold the method applied marked on that scale and the
cells that ended in a cluster told apart from the rest.

Nothing here re-runs the clustering. The analysis keeps the method's ``TessellationDiagnostics`` (density per
localization, clip window, threshold, mean density); the cell POLYGONS are either the ones the fit kept
(``keep_polygons=True``) or rebuilt from the same unique points and the same clip window with
``tools.cluster_tessellation.voronoi_cells`` - one qhull run, the same function the fit called, so every cell is the
cell the method measured (``cell_polygons`` checks the rebuilt areas against the fit's own, and refuses to draw when
they differ). The labels, the densities and the threshold are read, never recomputed.

The scale
---------
log10 of the density the method thresholded (first-rank for ``voronoi_factor`` by default, 1/area for
``voronoi_csr``), in ``N_CLASSES_BELOW`` equal classes from the ``CLIP_QUANTILES[0]`` quantile up to the threshold and
``N_CLASSES_ABOVE`` from the threshold up to the ``CLIP_QUANTILES[1]`` quantile (densities beyond are put in the end
classes) [ad hoc: equal steps in log density, the threshold on a class edge so that "above / below the cut" is read
from the colour alone]. The colours are ``tools.mps_plot_style.DENSITY_RAMP`` (viridis, a sequential map of monotone
lightness; checked under the three dichromacies by ``validate_plot_colours.py``). When the threshold falls outside
the drawn range (CSR data with no cut inside it), the eight classes span the range and the scale says so.

What counts as "in a cluster" is the method's own label (>= 0), before the edge curation of the analysis: the layer
shows what the tessellation did; the curation is what the localization rows show.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["VORONOI_METHODS", "DensityScale", "CellDrawing", "density_scale", "density_classes", "cell_polygons",
           "cell_drawing", "cell_counts", "has_tessellation", "density_words", "MAX_DRAWN_CELLS", "PER_UM2"]

VORONOI_METHODS: Tuple[str, ...] = ("voronoi_factor", "voronoi_csr")
N_CLASSES_BELOW = 4          # [ad hoc] classes of the scale under the threshold ...
N_CLASSES_ABOVE = 4          # [ad hoc] ... and over it; with the ramp's 8 colours
CLIP_QUANTILES = (0.01, 0.99)  # [ad hoc] the scale's ends: a few extreme cells do not stretch it
PER_UM2 = 1.0e6              # nm^-2 -> localizations per um^2 (what the scale prints)
# The most cells the layer draws [ad hoc, from a simulation (project), 2026-10-20, synthetic rings on the user's
# notebook, one process: the first tick (rebuilding the polygons and one path per class) took 0.28 s at 5,128 and
# 1.3-1.5 s at 20,290 localizations, ~65 us per cell, and one frame painted in 0.008 / 0.025 s; 50,000 cells would be
# ~3-4 s on the first tick]: above it the rows say why and draw nothing.
MAX_DRAWN_CELLS = 50000


def density_words(per_um2: float) -> str:
    """A density in localizations per um^2 as the layer prints it: grouped digits in the usual range (3,190), three
    significant digits outside it."""
    v = float(per_um2)
    if not np.isfinite(v):
        return str(v)
    return f"{v:,.0f}" if 10.0 <= abs(v) < 1.0e7 else f"{v:.3g}"


def has_tessellation(analysis: Any) -> bool:
    """Whether the analysis was clustered by a Voronoi method (its diagnostics may still be missing)."""
    return str(getattr(analysis, "cluster_method", "")) in VORONOI_METHODS


@dataclass
class DensityScale:
    """The classes of the scale: ``edges`` in log10(nm^-2), ``len(colours) + 1`` of them; ``threshold_edge`` the
    index of the edge that is the method's threshold (None when it lies outside the drawn range)."""
    edges: NDArray[np.float64]
    colours: Tuple[str, ...]
    threshold: float                       # nm^-2, as the method applied it
    threshold_edge: Optional[int]
    mean_density: float                    # nm^-2
    rule: str                              # 'factor', 'first crossing' or 'quantile fallback'
    note: str = ""                         # why the threshold is not on the scale, when it is not

    def edge_values_um2(self) -> NDArray[np.float64]:
        return np.asarray(10.0 ** self.edges * PER_UM2, dtype=np.float64)


def density_scale(diag: Any, colours: Optional[Tuple[str, ...]] = None) -> Optional[DensityScale]:
    """The scale for these diagnostics, or None when no cell has a positive density."""
    from tools.mps_plot_style import DENSITY_RAMP
    ramp = tuple(colours if colours is not None else DENSITY_RAMP)
    if len(ramp) != N_CLASSES_BELOW + N_CLASSES_ABOVE:
        raise ValueError("the ramp must have one colour per class")
    d = np.asarray(diag.density, dtype=np.float64)
    pos = d[np.isfinite(d) & (d > 0)]
    if pos.size == 0:
        return None
    lo = float(np.log10(np.quantile(pos, CLIP_QUANTILES[0])))
    hi = float(np.log10(np.quantile(pos, CLIP_QUANTILES[1])))
    if not hi > lo:
        hi = lo + 1e-6
    thr = float(diag.threshold_density)
    t = float(np.log10(thr)) if (np.isfinite(thr) and thr > 0) else float("nan")
    note = ""
    if np.isfinite(t) and lo < t < hi:
        edges = np.concatenate([np.linspace(lo, t, N_CLASSES_BELOW + 1),
                                np.linspace(t, hi, N_CLASSES_ABOVE + 1)[1:]])
        edge: Optional[int] = N_CLASSES_BELOW
    else:
        edges = np.linspace(lo, hi, len(ramp) + 1)
        edge = None
        if not np.isfinite(t):
            note = "no finite threshold"
        elif t >= hi:
            note = "the threshold is above the densest 1 % of the cells"
        else:
            note = "the threshold is below the sparsest 1 % of the cells"
    return DensityScale(edges=np.asarray(edges, dtype=np.float64), colours=ramp, threshold=thr,
                        threshold_edge=edge, mean_density=float(diag.mean_density),
                        rule=str(getattr(diag, "threshold_rule", "") or diag.threshold_mode), note=note)


def density_classes(density: NDArray[np.float64], scale: DensityScale) -> NDArray[np.int64]:
    """The class of each density (0 .. n-1); -1 for a cell of no area (density 0, not drawn). A density exactly at
    the threshold falls in the class above it, as ``voronoi_factor`` selects it (>=)."""
    d = np.asarray(density, dtype=np.float64)
    out = np.full(d.shape, -1, dtype=np.int64)
    ok = np.isfinite(d) & (d > 0)
    inner = scale.edges[1:-1]
    k = np.searchsorted(inner, np.log10(d[ok]), side="right")
    out[ok] = np.clip(k, 0, len(scale.colours) - 1)
    return out


# The polygons rebuilt for one diagnostics object, kept while it is shown (a redraw, a tick, the other row).
_CACHE: List[Tuple[Any, NDArray[np.float64], "_Cells"]] = []
_CACHE_SIZE = 3


@dataclass
class _Cells:
    first: NDArray[np.int64]               # (u,) the first localization of each unique position
    polygons: List[NDArray[np.float64]]    # (u,) one clipped polygon per unique position
    source: str                            # 'kept by the fit' | 'rebuilt from the fit's window'


def cell_polygons(diag: Any, x: NDArray[np.float64], y: NDArray[np.float64]) -> Tuple[_Cells, str]:
    """The cell of every unique position of (x, y), the localizations the method was given, and '' - or (empty, why)
    when they cannot be the method's own cells. Coincident localizations share one cell, as in the fit."""
    from tools.cluster_tessellation import voronoi_cells
    xy = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    empty = _Cells(np.zeros(0, dtype=np.int64), [], "")
    if xy.shape[0] != np.asarray(diag.density).shape[0]:
        return empty, (f"the tessellation has {np.asarray(diag.density).shape[0]} cells for {xy.shape[0]} "
                       f"localizations")
    for d, key, cells in _CACHE:
        if d is diag and key.shape == xy.shape and np.array_equal(key, xy):
            return cells, ""
    uxy, first = np.unique(xy, axis=0, return_index=True)
    first = np.asarray(first, dtype=np.int64)
    if uxy.shape[0] > MAX_DRAWN_CELLS:
        return empty, f"{uxy.shape[0]:,} cells, more than the {MAX_DRAWN_CELLS:,} this layer draws"
    kept = getattr(diag, "cell_polygons", None)
    if kept is not None and len(kept) == xy.shape[0]:
        cells = _Cells(first, [np.asarray(kept[int(i)], dtype=np.float64) for i in first], "kept by the fit")
    else:
        rebuilt = voronoi_cells(uxy, np.asarray(diag.window, dtype=np.float64), keep_polygons=True)
        area_fit = np.asarray(diag.cell_area, dtype=np.float64)[first]
        if rebuilt.polygons is None or not np.allclose(rebuilt.area, area_fit, rtol=1e-9, atol=1e-6):
            return empty, "the cells rebuilt from the fit's window differ from the fit's own"
        cells = _Cells(first, list(rebuilt.polygons), "rebuilt from the fit's window")
    _CACHE.insert(0, (diag, xy.copy(), cells))
    del _CACHE[_CACHE_SIZE:]
    return cells, ""


@dataclass
class CellDrawing:
    """Per class, the polygons of one row as one polyline: ``x``, ``y`` and ``connect`` (1 joins a vertex to the
    next, 0 ends a polygon), what ``pyqtgraph.arrayToQPath`` takes. ``n_cells`` per class."""
    classes: Dict[int, Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]] = field(
        default_factory=dict)
    n_cells: Dict[int, int] = field(default_factory=dict)


def _join(polys: List[NDArray[np.float64]]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    closed = [np.vstack([p, p[:1]]) for p in polys if p.shape[0] >= 3]
    if not closed:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int32)
    pts = np.concatenate(closed, axis=0)
    connect = np.ones(pts.shape[0], dtype=np.int32)
    ends = np.cumsum([c.shape[0] for c in closed]) - 1
    connect[ends] = 0
    return pts[:, 0].copy(), pts[:, 1].copy(), connect


def cell_drawing(diag: Any, x: NDArray[np.float64], y: NDArray[np.float64], labels: NDArray[np.int64],
                 scale: DensityScale, in_clusters: bool) -> Tuple[CellDrawing, str]:
    """The cells in clusters (``in_clusters``) or the others, by class; (empty drawing, why) when they cannot be
    drawn."""
    cells, why = cell_polygons(diag, x, y)
    if why:
        return CellDrawing(), why
    lab = np.asarray(labels)[cells.first]
    cls = density_classes(np.asarray(diag.density, dtype=np.float64)[cells.first], scale)
    member = (lab >= 0) if in_clusters else (lab < 0)
    out = CellDrawing()
    for k in range(len(scale.colours)):
        idx = np.flatnonzero(member & (cls == k))
        if idx.size == 0:
            continue
        out.classes[k] = _join([cells.polygons[int(i)] for i in idx])
        out.n_cells[k] = int(idx.size)
    return out, ""


def cell_counts(diag: Any, x: NDArray[np.float64], y: NDArray[np.float64],
                labels: NDArray[np.int64]) -> Tuple[int, int]:
    """Cells (unique positions) in clusters and outside them, without building any polygon."""
    xy = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    if xy.shape[0] == 0:
        return 0, 0
    _u, first = np.unique(xy, axis=0, return_index=True)
    lab = np.asarray(labels)[first]
    ok = np.asarray(diag.cell_area, dtype=np.float64)[first] > 0
    return int(np.sum((lab >= 0) & ok)), int(np.sum((lab < 0) & ok))
