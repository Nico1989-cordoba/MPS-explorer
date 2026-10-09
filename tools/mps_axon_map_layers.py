# -*- coding: utf-8 -*-
"""
What the axon map, the axial view and the nearest-neighbours tab draw, as data (UI stage 2, design 3-5, 12.8).

Pure functions, no Qt: each turns the objects an analysis already produced - an ``AxonAnalysis`` and its
``DiscardComparison``, a ``MultiSegmentAnalysis``, the Axoplasm panel's state - into row specifications (key, name,
group, arrays, style role, count, tip, available or not and why), plus the plot titles, the group captions and the
details-panel rows. The widgets (``tools/mps_axon_map.py``, ``tools/mps_axial_view.py``, ``tools/mps_nn_panel.py``)
turn the specs into pyqtgraph items; the headless tests compare the arrays with what the old drawing code drew.

Rules this module keeps (design 1, 3.4):

* A caption, a title and a details row read their numbers from the object drawn, never from an editor: a caption
  cannot name a value the drawing was not computed with (P1, P2).
* Three different sets of localizations are three exclusive sources; one set of cluster centres has two colourings;
  the contours are the analysis' own objects (3.4 rules 1-3).
* Nothing is computed that the program does not already compute; the only arithmetic here is presentation:
  histograms, the closed polyline of a contour, the runs of the occupied stretches, the sampling of at most 20,000
  points per class with seed 0 (as the Axoplasm panel drew them), and the nearest-neighbour query for N > 1 (the
  sklearn KD-tree code the main window used, moved here).
* Styles are role names of ``tools.mps_plot_style`` (no new role, P5); "neutral" is the background-dependent line
  colour, "segment" the segment's own colour. ``validate_plot_colours.py`` checks them against ``ROW_STYLES``.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_param_registry as reg

__all__ = [
    "Style", "Layer", "Group", "MapInputs", "AxoplasmMapState", "AxialInputs", "DetailRow", "ROW_STYLES",
    "MAP_VIEWS", "AXIAL_VIEWS", "MAX_DRAWN_PER_CLASS", "build_map", "map_title", "build_axial", "axial_title",
    "axial_details", "knn_distances", "nn_max_neighbours", "nn_histogram", "nn_title", "nn_layers", "nn_details",
    "cdf_layers", "cdf_title", "cdf_details", "area_details", "distances_table", "default_distances_name",
    "shown_words", "region_rect", "axoplasm_map_state", "provenance_line2", "closed", "occupied_runs",
    "write_distances_csv", "located_points", "image_levels", "component_curves", "rz_grid", "axial_edges",
    "style_of", "view_on", "axial_on",
]

# ============================================================================ styles
@dataclass(frozen=True)
class Style:
    """How a row is drawn. ``role``: a role of tools.mps_plot_style, "neutral" or "segment"; ``mark``: what the
    validator calls the shape ("dot", "x", "diamond", "circle", "plus", "line", "dashed line", "bars", "band"...)."""
    role: str
    alpha: int = 255
    mark: str = "dot"
    symbol: str = "o"
    size: float = 3.0
    width: float = 1.0
    pen: str = "solid"            # "solid" | "dash" | "dot"
    outline: Optional[str] = None  # marker outline: "neutral" | "dark" | None
    cased: bool = False
    hollow: bool = False


# Every row of the three widgets, by key (the segment rows share one entry each). The validator reads this table.
ROW_STYLES: Dict[str, Style] = {
    # map: localizations
    "slab_kept": Style("locs", 200, "dot", "o", 3.0),
    "slab_noise": Style("noise", 90, "dot", "o", 2.0),
    "slab_curated": Style("curated", 160, "x", "x", 5.0),
    "slab_discarded": Style("discarded", 255, "diamond", "d", 4.0),
    "sel_inside": Style("discarded", 255, "dot", "o", 2.0),
    "sel_membrane": Style("locs", 255, "dot", "o", 2.0),
    "sel_free": Style("neutral", 255, "dot", "o", 2.0),
    "seg": Style("segment", 255, "hollow marker", "o", 5.0, hollow=True),
    # map: cluster centres
    "c_kept": Style("centroid", 255, "circle", "o", 7.0, outline="neutral"),
    "c_discarded": Style("discarded", 255, "diamond", "d", 9.0, outline="neutral"),
    "c_both": Style("discarded", 255, "disc", "o", 10.0, outline="dark"),
    "c_tubulin": Style("image_tubulin", 255, "cased ring", "o", 10.0, cased=True, hollow=True),
    "c_spectrin": Style("image_spectrin", 255, "cased square", "s", 10.0, cased=True, hollow=True),
    "c_neither": Style("neutral", 255, "ring", "o", 9.0, hollow=True),
    # map: contour, occupancy and centre
    "contour": Style("neutral", 255, "cased line", width=1.5, cased=True),
    "contour_all": Style("neutral", 255, "cased dashed line", width=1.0, pen="dash", cased=True),
    "occupied": Style("occupied", 255, "thick line", width=4.0),
    "centre": Style("centre", 255, "cased plus", "+", 18.0, width=2.0, cased=True),
    "centre_all": Style("neutral", 255, "plus", "+", 14.0),
    "centres_mean": Style("neutral", 255, "x", "x", 10.0),
    "rand_band": Style("neutral", 40, "cloud of 1 px dots", "o", 1.0),
    # map: widefield images and rings
    "image": Style("image", 255, "image"),
    "tubulin_edge": Style("image_tubulin", 255, "cased line", cased=True),
    "spectrin_edge": Style("image_spectrin", 255, "cased line", cased=True),
    "ring_origin": Style("neutral", 255, "star", "star", 12.0),
    # axial view
    "roi_in": Style("locs", 170, "bars"),
    "roi_out": Style("dim", 100, "bars"),
    "ch2_roi": Style("channel_b", 255, "step line", width=1.5),
    "components": Style("fit", 255, "dashed line", width=2.0, pen="dash"),
    "mixture": Style("neutral", 255, "line", width=2.0),
    "peak": Style("neutral", 255, "dotted line", width=1.0, pen="dot"),
    "slab": Style("slab", 60, "band"),
    "main_cut": Style("neutral", 255, "dotted line", width=1.0, pen="dot"),
    "segband": Style("segment", 55, "band and a numbered dotted line"),
    "valleys": Style("neutral", 255, "labelled line", width=2.0),
    "seghist": Style("segment", 255, "step line", width=1.5),
    # nearest neighbours
    "nn_bars": Style("locs", 190, "bars"),
    "nn_median": Style("summary", 255, "line", width=2.0),
    "nn_reference": Style("paper", 255, "dashed line", width=2.0, pen="dash"),
    "cdf_observed": Style("observed", 255, "line", width=2.0),
    "cdf_randomized": Style("randomized", 255, "line", width=2.0),
    "cdf_crossing": Style("neutral", 255, "dashed line", width=1.0, pen="dash"),
}


def style_of(key: str) -> Style:
    """The style of a row key (``seg3`` -> the ``seg`` entry)."""
    if key in ROW_STYLES:
        return ROW_STYLES[key]
    for prefix in ("segband", "seghist", "seg"):
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return ROW_STYLES[prefix]
    raise KeyError(key)


# ============================================================================ specs
@dataclass
class Layer:
    """One row: what it draws and how it is named.

    kind   "scatter" (data x, y), "polyline" (x, y; closed already when it is a contour), "polylines" (runs: a
           list of (x, y)), "image" (image, rect, levels), "edge" (edge: a boolean array, rect), "bars" (x, height,
           width, y0), "step" (edges, values), "curves" (grid, curves: list of arrays), "vlines" (positions, labels,
           pens), "band" (lo, hi; optional line, label), "bands" (a list of those), "hline" (y, label)
    """
    key: str
    label: str
    group: str
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)
    count: Optional[int] = None
    tip: str = ""
    enabled: bool = True
    reason: str = ""
    segment: Optional[int] = None      # the segment's position (colour and symbol) for segment rows

    @property
    def style(self) -> Style:
        return style_of(self.key)


@dataclass
class Group:
    key: str
    title: str
    caption: str = ""
    layers: List[Layer] = field(default_factory=list)
    enabled: bool = True
    reason: str = ""
    header: Optional[str] = None       # "source" | "colour_by" | "image" | "view" | None

    def keys(self) -> List[str]:
        return [layer.key for layer in self.layers]


DetailRow = Tuple[str, str, str, str]     # name, value, note, badge


def _row(name: str, value: str, note: str = "", badge: str = "") -> DetailRow:
    return (name, value, note, badge)


# ============================================================================ small helpers
def closed(contour: NDArray[np.float64]) -> NDArray[np.float64]:
    """A contour's vertices with the first repeated at the end, as every old plot drew it."""
    c = np.asarray(contour, dtype=float)
    return np.vstack([c, c[:1]])


def occupied_runs(points: NDArray[np.float64], occupied: NDArray[np.bool_]) -> List[NDArray[np.float64]]:
    """The occupied stretches, one polyline per contiguous run of at least two points (the old contour plot's
    runs, ``mps_results_window._draw_contour``)."""
    pts = np.asarray(points, dtype=float)
    occ = np.asarray(occupied, dtype=bool)
    out: List[NDArray[np.float64]] = []
    if not np.any(occ):
        return out
    idx = np.flatnonzero(occ)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(idx) - 1]])
    for s, e in zip(starts, ends):
        run = pts[idx[s]:idx[e] + 1]
        if len(run) >= 2:
            out.append(run)
    return out


def _f0(v: float) -> str:
    return f"{v:,.0f}"


def _good_labels(analysis: Any) -> NDArray[np.int64]:
    from tools.cluster_quality import good_cluster_labels
    return good_cluster_labels(np.asarray(analysis.labels), analysis.bad_report.bad_labels)


def shown_words(shown: Any, comparison: Any, measured: Any) -> str:
    """Which analysis a plot shows, as its title ends with it when a discard comparison exists (3.6): "measured",
    "all N clusters" (every 2-opt start, when that is not the measured analysis itself), "discard applied, K of N
    clusters"; "" without a comparison."""
    if comparison is None or shown is None:
        return ""
    total = int(comparison.all_clusters.n_clusters_kept)
    if shown.discard_applied:
        return f"discard applied, {int(shown.n_clusters_kept)} of {total} clusters"
    if shown is comparison.all_clusters and shown is not measured:
        return f"all {total} clusters"
    return "measured"


# ============================================================================ the axoplasm panel's state
@dataclass
class AxoplasmMapState:
    """What the Axoplasm panel draws, computed where its ``_draw`` computed it (IMPL-C moves the computation into
    ``AxoplasmWindow.map_state()``, which calls ``axoplasm_map_state``)."""
    tubulin_region: Optional[NDArray[Any]] = None
    tubulin_rect: Optional[Tuple[float, float, float, float]] = None
    spectrin_region: Optional[NDArray[Any]] = None
    spectrin_rect: Optional[Tuple[float, float, float, float]] = None
    tubulin_edge: Optional[NDArray[np.bool_]] = None
    tubulin_edge_rect: Optional[Tuple[float, float, float, float]] = None
    spectrin_edge: Optional[NDArray[np.bool_]] = None
    spectrin_edge_rect: Optional[Tuple[float, float, float, float]] = None
    loc_x: Optional[NDArray[np.float64]] = None
    loc_y: Optional[NDArray[np.float64]] = None
    located: Optional[NDArray[Any]] = None         # LOC_* per localization; None before both images placed them
    anchored: Any = None                           # tools.mps_axoplasm.AnchoredClusters
    anchored_centroids: Optional[NDArray[np.float64]] = None
    clusters: Optional[NDArray[np.float64]] = None  # the centres before classification
    threshold_source: str = "otsu"                  # "otsu" | "manual"
    selection_note: Optional[str] = None
    margin_nm: Optional[float] = None
    registration: str = ""                          # how the images were placed, in the panel's words
    has_mask: bool = False


MAX_DRAWN_PER_CLASS = 20000     # tools.mps_axoplasm_window.MAX_DRAWN: points drawn per class (seed 0)


def region_rect(rows: Tuple[int, int], cols: Tuple[int, int], offset: Tuple[float, float],
                shift_px: Tuple[float, float], pixel_nm: float) -> Tuple[float, float, float, float]:
    """Where image pixels [rows) x [cols) lie, in localization nm (``AxoplasmWindow._region_rect``)."""
    sx, sy = shift_px
    px = float(pixel_nm)
    return ((cols[0] - 0.5 - offset[0] - sx) * px, (rows[0] - 0.5 - offset[1] - sy) * px,
            (cols[1] - cols[0]) * px, (rows[1] - rows[0]) * px)


def _edge(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    from scipy import ndimage
    m = np.asarray(mask, dtype=bool)
    out: NDArray[np.bool_] = m & ~ndimage.binary_erosion(m)
    return out


def axoplasm_map_state(*, tubulin_image: Optional[NDArray[Any]], tubulin_offset: Tuple[float, float],
                       reference_image: Optional[NDArray[Any]], reference_offset: Tuple[float, float],
                       mask: Any, interior: Any, shift_px: Tuple[float, float], pixel_nm: float,
                       loc_x: Optional[NDArray[np.float64]], loc_y: Optional[NDArray[np.float64]],
                       located: Optional[NDArray[Any]], anchored: Any,
                       anchored_centroids: Optional[NDArray[np.float64]],
                       clusters: Optional[NDArray[np.float64]], threshold_source: str = "otsu",
                       selection_note: Optional[str] = None, margin_nm: Optional[float] = None,
                       registration: str = "", has_result: bool = True) -> AxoplasmMapState:
    """The Axoplasm panel's drawing as data, computed as ``AxoplasmWindow._draw`` computes it: the tubulin region
    of the mask and, when a reference is loaded, the same area in the spectrin image; the two cased edges; the
    localizations of the selection by class; the anchored classification."""
    state = AxoplasmMapState(loc_x=loc_x, loc_y=loc_y, located=located, anchored=anchored,
                             anchored_centroids=anchored_centroids, clusters=clusters,
                             threshold_source=threshold_source, selection_note=selection_note, margin_nm=margin_nm,
                             registration=registration)
    if not has_result:
        state.loc_x = state.loc_y = None
    if mask is None or tubulin_image is None:
        return state
    state.has_mask = True
    r0, r1, c0, c1 = mask.region
    region = np.asarray(tubulin_image)[r0:r1, c0:c1]
    state.tubulin_region = region
    state.tubulin_rect = region_rect((r0, r1), (c0, c1), tubulin_offset, shift_px, pixel_nm)
    if reference_image is not None:
        dr = reference_offset[1] - tubulin_offset[1]
        dc = reference_offset[0] - tubulin_offset[0]
        height, width = np.asarray(reference_image).shape[:2]
        rows = (int(max(0, r0 + dr)), int(min(height, r1 + dr)))
        cols = (int(max(0, c0 + dc)), int(min(width, c1 + dc)))
        state.spectrin_region = np.asarray(reference_image)[rows[0]:rows[1], cols[0]:cols[1]]
        state.spectrin_rect = region_rect(rows, cols, reference_offset, shift_px, pixel_nm)
    state.tubulin_edge = _edge(mask.mask)
    state.tubulin_edge_rect = region_rect((r0, r1), (c0, c1), tubulin_offset, shift_px, pixel_nm)
    if interior is not None and np.asarray(interior.mask).any():
        state.spectrin_edge = _edge(interior.mask)
        ir0, ir1, ic0, ic1 = interior.region
        state.spectrin_edge_rect = region_rect((ir0, ir1), (ic0, ic1), reference_offset, shift_px, pixel_nm)
    return state


def image_levels(region: NDArray[Any]) -> Tuple[float, float]:
    """Display levels of a widefield region: percentiles 1 and 99.7, as the Axoplasm panel drew it."""
    low, high = np.percentile(region, (1, 99.7))
    return float(low), float(max(high, low + 1))


def located_points(state: AxoplasmMapState) -> Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], int]]:
    """The localizations of the selection by class, at most MAX_DRAWN_PER_CLASS drawn per class (one generator,
    seed 0, drawn in the order no cluster / membrane / inside: the Axoplasm panel's own order). Values:
    (x, y, n of all)."""
    from tools.mps_axoplasm import LOC_INSIDE, LOC_MEMBRANE, LOC_NO_CLUSTER
    out: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64], int]] = {}
    if state.loc_x is None or state.loc_y is None:
        return out
    x = np.asarray(state.loc_x, dtype=float)
    y = np.asarray(state.loc_y, dtype=float)
    rng = np.random.default_rng(0)
    located = (state.located if state.located is not None
               else np.full(x.size, LOC_NO_CLUSTER, dtype=object))
    for label in (LOC_NO_CLUSTER, LOC_MEMBRANE, LOC_INSIDE):
        idx = np.nonzero(located == label)[0]
        n_all = int(idx.size)
        if idx.size > MAX_DRAWN_PER_CLASS:
            idx = rng.choice(idx, MAX_DRAWN_PER_CLASS, replace=False)
        out[label] = (x[idx], y[idx], n_all)
    return out


# ============================================================================ the map
MAP_VIEWS: Tuple[str, ...] = ("mps", "axoplasm", "segments", "custom")
MAP_VIEW_NAMES: Dict[str, str] = {"mps": "MPS analysis", "axoplasm": "Axoplasm", "segments": "Segments",
                                  "custom": "Custom"}
SOURCES: Dict[str, str] = {"slab": "MPS analysis slab", "selection": "Main-window selection (axoplasm classes)",
                           "segments": "Each segment's own slab (rings)"}
COLOURINGS: Dict[str, str] = {"status": "status in the MPS analysis", "images": "the two widefield images"}
IMAGES: Dict[str, str] = {"tubulin": "betaIII-tubulin", "spectrin": "betaII-spectrin"}
# Per view: the localization source, the colouring of the centres, and the rows that start ticked.
VIEW_SOURCE: Dict[str, str] = {"mps": "slab", "axoplasm": "selection", "segments": "segments"}
VIEW_COLOUR_BY: Dict[str, str] = {"mps": "status", "axoplasm": "images", "segments": "status"}
VIEW_ON: Dict[str, FrozenSet[str]] = {
    "mps": frozenset({"slab_kept", "slab_noise", "slab_curated", "slab_discarded", "c_kept", "c_discarded",
                      "contour", "contour_all", "occupied", "centre", "centre_all"}),
    "axoplasm": frozenset({"sel_inside", "sel_membrane", "sel_free", "c_both", "c_tubulin", "c_spectrin",
                           "c_neither", "contour", "contour_all", "centre", "centre_all", "image", "tubulin_edge",
                           "spectrin_edge"}),
    "segments": frozenset({"seg"}),
}
# Rows disabled while the widefield image is drawn (3.5): grey on the grey image.
OVER_IMAGE_DISABLED: Tuple[str, ...] = ("slab_noise", "slab_curated")
OVER_IMAGE_REASON = "grey on the grey image: use the axoplasm classes"
# Rows disabled with the segments' source (3.5): the five segment colours are five roles' colours.
SEGMENT_COLOUR_REASON = "the segments use these colours"
SEGMENTS_DISABLE: Tuple[str, ...] = ("c_kept", "c_discarded", "c_both", "c_tubulin", "c_spectrin", "occupied",
                                     "tubulin_edge", "spectrin_edge")
# What may be drawn over the segments (validator: "segments against the map's overlays").
SEGMENTS_ALLOW: Tuple[str, ...] = ("contour", "contour_all", "centre", "centre_all", "centres_mean", "rand_band",
                                   "c_neither", "image", "ring_origin")
MAP_VIEW_TIP = ("Presets of the rows. MPS analysis: what the analysis drew (its slab, its centres by status, its "
                "contour). Axoplasm: the main-window selection placed by the two widefield images, over the image; "
                "it shows the analysis with the discard applied when the Axoplasm panel discarded clusters, as the "
                "radio above the table does. Segments: each segment's own slab from the Rings window. Any change "
                "of a row makes the view Custom.")


def view_on(view: str, key: str) -> bool:
    """Whether ``key`` starts ticked in ``view``."""
    on = VIEW_ON.get(view, frozenset())
    return key in on or (key.startswith("seg") and key[3:].isdigit() and "seg" in on)


@dataclass
class MapInputs:
    """Everything the map is drawn from. ``analysis`` is the MEASURED analysis; ``shown`` which of the three the
    radio shows ("measured" | "every" | "discard")."""
    analysis: Any = None
    comparison: Any = None
    shown: str = "measured"
    stale: bool = False
    given_after_cut: bool = False
    discard_failed: str = ""                        # why the discard comparison could not be made, when it failed
    axoplasm: Optional[AxoplasmMapState] = None
    axoplasm_reason: str = "the Axoplasm panel is not open on this selection"
    rings: Any = None
    rings_reason: str = "press Rings... to compute the segments"
    source: str = "slab"
    colour_by: str = "status"
    image: str = "tubulin"
    selection_n: Optional[int] = None               # the plain selection, with no axoplasm panel

    def shown_analysis(self) -> Any:
        if self.analysis is None:
            return None
        c = self.comparison
        if c is not None:
            if self.shown == "every":
                return c.all_clusters
            if self.shown == "discard":
                return c.discard_applied
        return self.analysis


def _contour_words(perimeter: Any) -> str:
    if perimeter is None:
        return "no contour (too few clusters)"
    if perimeter.order_source != "automatic":
        return ("contour joined along the drawn path" if perimeter.guide is not None
                else "contour in an order set by hand")
    if perimeter.n_starts > 1:
        return "contour by 2-opt from every start"
    return "contour by 2-opt from one start (as before 2026-09-19)"


def _seg_params(ms: Any) -> Dict[str, Any]:
    first = next((a for a in ms.analyses if a is not None), None)
    if first is None:
        return {}
    return {"eps": float(first.eps_nm), "min": int(first.min_samples), "half": float(first.slab_half_width_nm),
            "maha": float(first.mahalanobis_threshold)}


def _seg_words(ms: Any) -> str:
    p = _seg_params(ms)
    base = f"mode {ms.mode}, guard {float(getattr(ms, 'guard_nm', 0.0)):g} nm"
    if not p:
        return base
    return (f"{base}; eps {p['eps']:g} nm{reg.user_mark('dbscan.eps_nm', p['eps'])}, "
            f"min {p['min']}{reg.user_mark('dbscan.min_samples', p['min'])}, "
            f"half-width {p['half']:g} nm{reg.user_mark('slab.half_width_nm', p['half'])}, "
            f"Mahalanobis {p['maha']:g}{reg.user_mark('occupancy.mahalanobis', p['maha'])}")


def _localization_group(inp: MapInputs, shown: Any) -> Group:
    g = Group("localizations", "Localizations", header="source")
    src = inp.source
    if src == "slab":
        if shown is None:
            g.caption = "No MPS analysis of this selection yet."
            g.enabled, g.reason = False, "no MPS analysis of this selection yet"
            return g
        given = ("the main-window cut selection" if inp.given_after_cut else "the ROI before the Z cut")
        g.caption = (f"analyze_axon on {given}, then its own slab {shown.slab_zmin_nm:,.0f} <= z <= "
                     f"{shown.slab_zmax_nm:,.0f} nm; DBSCAN eps {shown.eps_nm:g} nm"
                     f"{reg.user_mark('dbscan.eps_nm', float(shown.eps_nm))}, min samples {int(shown.min_samples)}"
                     f"{reg.user_mark('dbscan.min_samples', int(shown.min_samples))} (of the analysis shown)")
        if shown.bad_report.edge_criterion_disabled:
            g.caption += f". Edge criterion off: {shown.bad_report.edge_criterion_disabled}"
        x, y = np.asarray(shown.x_slab, float), np.asarray(shown.y_slab, float)
        labels = np.asarray(shown.labels)
        noise = labels == -1
        bad = shown.bad_report.bad_labels
        bad_mask = np.isin(labels, list(bad)) if bad else np.zeros_like(noise)
        gone = (np.isin(labels, list(shown.discarded_labels)) if shown.discarded_labels
                else np.zeros_like(noise))
        kept = (~noise) & (~bad_mask) & (~gone)
        curated_word = ("Removed by the edge criterion or by DBCV" if getattr(shown.bad_report, "low_dbcv", None)
                        else "Removed by the edge criterion")
        g.layers = [
            Layer("slab_kept", "In kept clusters", g.key, "scatter", {"x": x[kept], "y": y[kept]},
                  int(kept.sum()), "Localizations of the analysis slab in the clusters the analysis kept."),
            Layer("slab_noise", "DBSCAN noise", g.key, "scatter", {"x": x[noise], "y": y[noise]}, int(noise.sum()),
                  "Localizations of the slab DBSCAN put in no cluster."),
            Layer("slab_curated", curated_word, g.key, "scatter", {"x": x[bad_mask], "y": y[bad_mask]},
                  int(np.sum(bad_mask)), "Localizations in clusters the automatic curation removed."),
            Layer("slab_discarded", "In discarded clusters", g.key, "scatter", {"x": x[gone], "y": y[gone]},
                  int(np.sum(gone)), "Localizations in the clusters the Axoplasm panel discarded.",
                  enabled=bool(shown.discard_applied),
                  reason="" if shown.discard_applied else "shown only with the discard applied"),
        ]
        return g
    if src == "selection":
        g.caption = ("the ROI after the main-window cut Z min < z < Z max (the set the Axoplasm, Data quality and "
                     "DNA-PAINT panels use; Two channels re-selects from the ROI shape and the slab); each "
                     "localization placed through its MPS-analysis cluster by the two widefield images; at most "
                     f"{MAX_DRAWN_PER_CLASS:,} drawn per class (seed 0)")
        ax_state = inp.axoplasm
        if ax_state is None or ax_state.loc_x is None:
            g.enabled, g.reason = False, inp.axoplasm_reason
            n = inp.selection_n
            g.layers = [Layer("sel_free", "Not placed yet", g.key, "scatter", {"x": np.empty(0), "y": np.empty(0)},
                              n, "The localizations of the selection.")]
            return g
        from tools.mps_axoplasm import LOC_INSIDE, LOC_MEMBRANE, LOC_NO_CLUSTER
        pts = located_points(ax_state)
        placed = ax_state.located is not None
        xi, yi, ni = pts[LOC_INSIDE]
        xm, ym, nm_ = pts[LOC_MEMBRANE]
        xf, yf, nf = pts[LOC_NO_CLUSTER]
        g.layers = [
            Layer("sel_inside", "Inside the axon: in discarded clusters", g.key, "scatter", {"x": xi, "y": yi}, ni,
                  "Localizations whose cluster both images put inside the axon."),
            Layer("sel_membrane", "At the membrane: in clusters not discarded", g.key, "scatter",
                  {"x": xm, "y": ym}, nm_, "Localizations in a cluster the Axoplasm panel did not discard."),
            Layer("sel_free", "In no cluster" if placed else "Not placed yet", g.key, "scatter",
                  {"x": xf, "y": yf}, nf,
                  ("Outside the analysis slab, DBSCAN noise, or removed by the edge criterion." if placed else
                   "Every localization of the selection, before both images placed the clusters.")),
        ]
        if inp.stale:
            g.enabled, g.reason = False, "the analysis is of the previous selection or axial slab"
        return g
    # segments
    ms = inp.rings
    if ms is None:
        g.caption = "Each segment's own slab, from the Rings window."
        g.enabled, g.reason = False, inp.rings_reason
        return g
    g.caption = (f"analyze_all_segments, {_seg_words(ms)} as the segments were computed (read from the segments); "
                 "every localization of the segment's slab (clusters and noise not told apart)")
    for k, (seg, an) in enumerate(zip(ms.segments, ms.analyses)):
        if an is None or not np.asarray(an.x_slab).size:
            continue
        g.layers.append(Layer(f"seg{seg.index}", f"Segment {seg.index}", g.key, "scatter",
                              {"x": np.asarray(an.x_slab, float), "y": np.asarray(an.y_slab, float)},
                              int(np.asarray(an.x_slab).size),
                              f"Segment {seg.index}'s slab, {seg.zmin_nm:,.0f} to {seg.zmax_nm:,.0f} nm.", segment=k))
    return g


def _centres_group(inp: MapInputs, shown: Any) -> Group:
    g = Group("centres", "Cluster centres (MPS analysis)", header="colour_by")
    measured = inp.analysis
    if measured is None:
        g.caption = "No MPS analysis of this selection yet."
        g.enabled, g.reason = False, "no MPS analysis of this selection yet"
        return g
    if inp.colour_by == "status":
        g.caption = "the clusters the analysis shown kept after the edge criterion"
        c = np.asarray(shown.centroids, float).reshape(-1, 2)
        g.layers.append(Layer("c_kept", "Kept", g.key, "scatter", {"x": c[:, 0], "y": c[:, 1]}, int(len(c)),
                              "The centre of mass of each cluster the analysis shown kept (the green centres the "
                              "main window drew)."))
        if shown.discard_applied and inp.comparison is not None:
            labels = _good_labels(measured)
            out = np.isin(labels, list(shown.discarded_labels))
            d = np.asarray(measured.centroids, float).reshape(-1, 2)[out]
            g.layers.append(Layer("c_discarded", "Discarded by the axoplasm panel", g.key, "scatter",
                                  {"x": d[:, 0], "y": d[:, 1]}, int(len(d)),
                                  "The same clusters as 'Inside both images' in the other colouring."))
        else:
            g.layers.append(Layer("c_discarded", "Discarded by the axoplasm panel", g.key, "scatter",
                                  {"x": np.empty(0), "y": np.empty(0)}, 0,
                                  "The same clusters as 'Inside both images' in the other colouring.",
                                  enabled=False, reason="shown only with the discard applied"))
        return g
    # the two widefield images
    total = int(measured.n_clusters_kept)
    left_out = total - int(shown.n_clusters_kept)
    st = inp.axoplasm
    margin = None if st is None else st.margin_nm
    margin_words = "" if margin is None else f" (margin {margin:,.0f} nm)"
    g.caption = (f"all {total} measured clusters, sorted by the two images of the Axoplasm panel{margin_words}; the "
                 f"analysis shown leaves out {left_out} of them")
    if st is None:
        g.enabled, g.reason = False, inp.axoplasm_reason
        return g
    if st.anchored is not None and st.anchored_centroids is not None:
        if not np.array_equal(np.asarray(st.anchored_centroids), np.asarray(measured.centroids)):
            g.enabled, g.reason = False, "the Axoplasm panel classified other clusters than this analysis measured"
        c = np.asarray(st.anchored_centroids, float).reshape(-1, 2)
        for key, name, attr, tip in (
                ("c_both", "Inside both images: discarded", "discarded",
                 "Inside the tubulin mask and the spectrin interior: discarded (the same clusters as 'Discarded' "
                 "in the other colouring)."),
                ("c_tubulin", "Inside the tubulin mask only", "tubulin_only", "Kept."),
                ("c_spectrin", "Inside the spectrin interior only", "spectrin_only", "Kept."),
                ("c_neither", "Inside neither: on the membrane", "inside_neither", "Kept, on the membrane.")):
            members = np.asarray(getattr(st.anchored, attr), dtype=bool)
            g.layers.append(Layer(key, name, g.key, "scatter", {"x": c[members, 0], "y": c[members, 1]},
                                  int(members.sum()), tip))
        return g
    every = st.clusters if st.clusters is not None else np.asarray(measured.centroids, float)
    c = np.asarray(every, float).reshape(-1, 2)
    g.layers.append(Layer("c_neither", "Not classified yet", g.key, "scatter", {"x": c[:, 0], "y": c[:, 1]},
                          int(len(c)), "Every centre, before both images placed the clusters."))
    return g


def _contour_group(inp: MapInputs, shown: Any) -> Group:
    g = Group("contour", "Contour, occupancy and centre")
    if shown is None:
        g.caption = "No MPS analysis of this selection yet."
        g.enabled, g.reason = False, "no MPS analysis of this selection yet"
        return g
    occ = shown.occupancy
    t = float(shown.mahalanobis_threshold)
    cap = ""
    if occ is not None:
        cap = (f", sigma capped at d_max/{1.0 / reg.default('occupancy.sigma_cap_fraction'):.3g} for "
               f"{int(occ.n_capped)} of {len(occ.ellipses)} ellipses")
    g.caption = (f"{_contour_words(shown.perimeter)}; occupied at Mahalanobis {t:g}"
                 f"{reg.user_mark('occupancy.mahalanobis', t)}{cap}; the centre is the contour's area centroid")
    if inp.discard_failed:
        g.caption += f". No discard comparison: {inp.discard_failed}"
    per = shown.perimeter
    comp = inp.comparison
    discard = bool(shown.discard_applied and comp is not None)
    if per is not None:
        cc = closed(per.contour)
        label = f"Contour ({per.perimeter_um:,.2f} um)" + (" (along the drawn path)" if per.guide is not None else "")
        g.layers.append(Layer("contour", label, g.key, "polyline", {"x": cc[:, 0], "y": cc[:, 1]}, None,
                              "The contour through the cluster centres of the analysis shown."))
    every = comp.all_clusters.perimeter if comp is not None else None
    if discard and every is not None:
        ec = closed(every.contour)
        how = "as measured" if comp.all_clusters is inp.analysis else "every start"
        g.layers.append(Layer("contour_all", f"Contour with every cluster ({every.perimeter_um:,.2f} um; {how})",
                              g.key, "polyline", {"x": ec[:, 0], "y": ec[:, 1]}, None,
                              "The contour before the discard: every cluster the analysis kept."))
    else:
        g.layers.append(Layer("contour_all", "Contour with every cluster", g.key, "polyline",
                              {"x": np.empty(0), "y": np.empty(0)}, None, "The contour before the discard.",
                              enabled=False, reason="shown only with the discard applied"))
    if occ is not None:
        runs = occupied_runs(occ.perimeter_points, occ.occupied_mask)
        g.layers.append(Layer("occupied", f"Occupied stretches ({occ.occupancy_percent:.1f} %)", g.key, "polylines",
                              {"runs": runs}, None, "The stretches of the contour within reach of a cluster."))
    if shown.centre is not None:
        g.layers.append(Layer("centre", "Centre of the contour (+)", g.key, "scatter",
                              {"x": np.array([shown.centre.x_nm]), "y": np.array([shown.centre.y_nm])}, None,
                              "The area centroid of the contour of the analysis shown."))
    if discard and every is not None and every.centre is not None:
        g.layers.append(Layer("centre_all", "Centre with every cluster (+)", g.key, "scatter",
                              {"x": np.array([every.centre.x_nm]), "y": np.array([every.centre.y_nm])}, None,
                              "The area centroid of the contour before the discard."))
    if per is not None and len(per.contour):
        m = np.asarray(per.contour, float).reshape(-1, 2).mean(axis=0)
        g.layers.append(Layer("centres_mean", "Mean of the cluster centres (origin of the \"Scatter off the "
                              "outline\" angles)", g.key, "scatter", {"x": np.array([m[0]]), "y": np.array([m[1]])},
                              None, "Used today as the origin of the angles of 'Scatter off the outline'."))
    r = shown.randomization
    if r is not None:
        cp = np.asarray(r.candidate_points, float).reshape(-1, 2)
        sc = np.asarray(r.smoothed_contour, float).reshape(-1, 2)
        g.layers.append(Layer("rand_band", "Randomization band (the null's candidate positions)", g.key, "scatter",
                              {"x": cp[:, 0], "y": cp[:, 1], "contour_x": sc[:, 0], "contour_y": sc[:, 1]},
                              int(len(cp)),
                              f"The positions the randomization draws from: within {r.annulus_half_width_nm:g} nm "
                              "of the smoothed contour (as computed)."))
    else:
        g.layers.append(Layer("rand_band", "Randomization band (the null's candidate positions)", g.key, "scatter",
                              {"x": np.empty(0), "y": np.empty(0)}, None, "The positions the randomization draws "
                              "from.", enabled=False, reason="the randomization was not run"))
    return g


def _image_group(inp: MapInputs) -> Group:
    g = Group("images", "Widefield images (Axoplasm panel)", header="image")
    st = inp.axoplasm
    if st is None or not st.has_mask:
        g.caption = "loaded in the Axoplasm panel"
        g.enabled, g.reason = False, (inp.axoplasm_reason if st is None else "no tubulin image or mask yet")
        return g
    threshold = "Otsu's" if st.threshold_source == "otsu" else "set by hand"
    note = f" ({st.selection_note})" if st.selection_note else ""
    margin = ""
    if st.margin_nm is not None:
        margin = (f"; margin {st.margin_nm:,.0f} nm ({reg.badge('axoplasm.margin_nm', float(st.margin_nm))})")
    reg_words = f" ({st.registration})" if st.registration else ""
    g.caption = (f"loaded in the Axoplasm panel and placed with its shift{reg_words}; threshold {threshold}{note}"
                 f"{margin}; smoothing and margin are set there")
    use_spectrin = inp.image == "spectrin" and st.spectrin_region is not None
    region = st.spectrin_region if use_spectrin else st.tubulin_region
    rect = st.spectrin_rect if use_spectrin else st.tubulin_rect
    label = f"Widefield image: {IMAGES['spectrin' if use_spectrin else 'tubulin']}"
    if region is not None and np.asarray(region).size:
        g.layers.append(Layer("image", label, g.key, "image",
                              {"image": np.asarray(region), "rect": rect, "levels": image_levels(np.asarray(region)),
                               "which": "spectrin" if use_spectrin else "tubulin"}, None,
                              "The widefield image under the map (choose which one below the row)."))
    if st.tubulin_edge is not None:
        g.layers.append(Layer("tubulin_edge", "Tubulin mask edge", g.key, "edge",
                              {"edge": st.tubulin_edge, "rect": st.tubulin_edge_rect}, None,
                              "The edge of the tubulin mask: the outside of the axoplasm."))
    if st.spectrin_edge is not None:
        g.layers.append(Layer("spectrin_edge", "Spectrin interior edge", g.key, "edge",
                              {"edge": st.spectrin_edge, "rect": st.spectrin_edge_rect}, None,
                              "The edge of the dark inside of the spectrin ring."))
    return g


def _rings_group(inp: MapInputs) -> Group:
    g = Group("rings", "Rings")
    ms = inp.rings
    if ms is None:
        g.caption = "the segments of the Rings window"
        g.enabled, g.reason = False, inp.rings_reason
        return g
    g.caption = f"from the Rings window: {_seg_words(ms)}"
    if ms.axon_center is not None:
        c = np.asarray(ms.axon_center, float).ravel()
        g.layers.append(Layer("ring_origin", "Angle origin of the ring profiles (pooled centre of every segment's "
                              "clusters)", g.key, "scatter", {"x": c[:1], "y": c[1:2]}, None,
                              "The origin of the angles of 'Patches around the perimeter'. SCI-6 moves the ring "
                              "correlation to arc length and keeps the angle as an exploratory mode."))
    return g


def build_map(inp: MapInputs) -> List[Group]:
    """Every group of the map for these inputs, with the rows of the chosen source and colouring only, each with
    its availability and reason (3.3, 3.4, 3.5)."""
    shown = inp.shown_analysis()
    groups = [_localization_group(inp, shown), _centres_group(inp, shown), _contour_group(inp, shown),
              _image_group(inp), _rings_group(inp)]
    if inp.source == "segments":
        for g in groups:
            for layer in g.layers:
                if layer.key in SEGMENTS_DISABLE:
                    layer.enabled, layer.reason = False, SEGMENT_COLOUR_REASON
    return groups


def map_title(inp: MapInputs, ticked: Set[str]) -> str:
    """The map's plot title: line 1 the analysis shown, line 2 what else is drawn (only groups with a ticked row);
    two lines joined by <br>, exported with every figure (P7, 3.1)."""
    shown = inp.shown_analysis()
    if shown is None:
        line1 = "No MPS analysis of this selection yet"
    else:
        comp = inp.comparison
        if shown.discard_applied and comp is not None:
            total = int(comp.all_clusters.n_clusters_kept)
            line1 = (f"Discard applied: {int(shown.n_clusters_kept)} of {total} clusters, "
                     f"{len(shown.discarded_labels)} discarded")
        elif comp is not None and inp.shown == "every" and comp.all_clusters is not inp.analysis:
            line1 = f"All {int(shown.n_clusters_kept)} clusters, contour from every 2-opt start"
        else:
            line1 = f"MPS analysis, measured: {int(shown.n_clusters_kept)} clusters"
        if inp.stale:
            line1 += " (previous selection)"
    parts: List[str] = []
    src = inp.source
    if src == "slab" and ticked & {"slab_kept", "slab_noise", "slab_curated", "slab_discarded"} and shown is not None:
        parts.append(f"localizations: MPS analysis slab {shown.slab_zmin_nm:,.0f}..{shown.slab_zmax_nm:,.0f} nm")
    elif src == "selection" and ticked & {"sel_inside", "sel_membrane", "sel_free"}:
        parts.append("localizations: main-window cut, axoplasm classes")
    elif src == "segments" and inp.rings is not None and any(k.startswith("seg") and k[3:].isdigit()
                                                              for k in ticked):
        parts.append(f"localizations: each segment's slab, mode {inp.rings.mode}")
    if ticked & {"c_kept", "c_discarded"} and inp.colour_by == "status":
        parts.append("centres by status in the analysis")
    elif ticked & {"c_both", "c_tubulin", "c_spectrin", "c_neither"} and inp.colour_by == "images":
        parts.append("centres by the two widefield images")
    if "image" in ticked:
        st = inp.axoplasm
        which = "spectrin" if (inp.image == "spectrin" and st is not None and st.spectrin_region is not None) \
            else "tubulin"
        parts.append(f"image: {IMAGES[which]}")
    return line1 if not parts else line1 + "<br>" + "; ".join(parts)


def provenance_line2(analysis: Any, *, roi_words: str, cut: Optional[Tuple[float, float]],
                     given_after_cut: bool = False) -> str:
    """The map's second provenance line (3.1; screen and export message, not the figure): frame, ROI, how many
    localizations the analysis was given, slab, cut, Mahalanobis, and "z as fitted". A value departing from its
    default carries [user]."""
    parts = ["lab x, y [nm]", roi_words]
    if analysis is not None:
        given = "the cut selection" if given_after_cut else "the ROI before the cut"
        parts.append(f"{int(analysis.n_locs_total):,} localizations given to the analysis ({given})")
        half = float(analysis.slab_half_width_nm)
        parts.append(f"MPS analysis slab {analysis.slab_zmin_nm:,.0f}..{analysis.slab_zmax_nm:,.0f} nm "
                     f"({analysis.slab_source}; half-width {half:g} nm{reg.user_mark('slab.half_width_nm', half)})")
    if cut is not None:
        parts.append(f"main-window cut {cut[0]:,.0f}..{cut[1]:,.0f} nm")
    if analysis is not None:
        t = float(analysis.mahalanobis_threshold)
        parts.append(f"Mahalanobis {t:g}{reg.user_mark('occupancy.mahalanobis', t)}")
    parts.append("z as fitted, no z-calibration record read")
    return "; ".join(parts)


# ============================================================================ the axial view
AXIAL_VIEWS: Tuple[str, ...] = ("slab", "segments")
AXIAL_VIEW_NAMES: Dict[str, str] = {"slab": "MPS analysis slab", "segments": "Every segment"}
AXIAL_BIN_NM = 10.0
AXIAL_ON: Dict[str, FrozenSet[str]] = {
    "slab": frozenset({"roi_in", "roi_out", "components", "mixture", "slab", "main_cut"}),
    "segments": frozenset({"roi_out", "mixture", "segband", "valleys"}),
}
AXIAL_LEFT_LABEL = "density over the ROI [1/nm]"
Z_CALIBRATION_LINE = "z as fitted; no z-calibration record read"
CH2_HIDES_COMPONENTS = ("channel 2 is drawn: its vermillion is too close to the components' orange under "
                        "deuteranopia; the neutral mixture stays")


def axial_on(view: str, key: str) -> bool:
    on = AXIAL_ON.get(view, frozenset())
    for prefix in ("segband", "seghist"):
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return prefix in on
    return key in on


@dataclass
class AxialInputs:
    """What the axial view is drawn from (4.2, 4.5). ``z_roi`` is the z of the selection the analysis was given
    (the ROI before the cut, or the cut selection in the fallback) or, with no analysis, of the current ROI."""
    z_roi: Optional[NDArray[np.float64]] = None
    analysis: Any = None                   # the analysis shown
    fit: Any = None                        # ZPeriodicityResult with no analysis (the prefill's fit)
    cut: Optional[Tuple[float, float]] = None
    cut_from_histogram_mode: bool = False
    cut_stale: bool = False
    given_after_cut: bool = False
    ch2_z: Optional[NDArray[np.float64]] = None
    ch2_reason: str = "no channel 2 is loaded"
    ch2_shown: bool = False
    rings: Any = None
    rings_reason: str = "press Rings... to compute the segments"
    view: str = "slab"
    fit_reason: str = "the mixture could not be fitted"

    def z_result(self) -> Any:
        if self.analysis is not None:
            return self.analysis.z_result
        if self.rings is not None and self.view == "segments":
            return self.rings.z_result
        return self.fit


def axial_edges(z: NDArray[np.float64], anchor: float, bin_nm: float = AXIAL_BIN_NM) -> NDArray[np.float64]:
    """Bin edges every ``bin_nm`` on a grid through ``anchor`` (the slab's lower bound), covering ``z``."""
    zz = np.asarray(z, float)
    zz = zz[np.isfinite(zz)]
    if zz.size == 0:
        return np.array([anchor, anchor + bin_nm])
    k0 = int(np.floor((zz.min() - anchor) / bin_nm))
    k1 = int(np.ceil((zz.max() - anchor) / bin_nm))
    if k1 <= k0:
        k1 = k0 + 1
    if anchor + k1 * bin_nm <= zz.max():
        k1 += 1
    return anchor + bin_nm * np.arange(k0, k1 + 1, dtype=float)


def component_curves(zr: Any, grid: NDArray[np.float64]) -> List[NDArray[np.float64]]:
    """Each dominant component's weighted density on ``grid`` (the old axial plot's curves)."""
    out: List[NDArray[np.float64]] = []
    for m, wgt, s in zip(zr.means_nm, zr.weights, zr.sigmas_nm):
        if s <= 0:
            continue
        dens = wgt * np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        out.append(np.asarray(dens, float))
    return out


def rz_grid(analysis: Any) -> NDArray[np.float64]:
    """The grid the old axial plot drew the components on: the slab +/- two slab widths, 1,024 points."""
    lo, hi = float(analysis.slab_zmin_nm), float(analysis.slab_zmax_nm)
    span = max(hi - lo, 1.0)
    return np.linspace(lo - 2 * span, hi + 2 * span, 1024)


def _slab_mask(a: Any, z: NDArray[np.float64]) -> NDArray[np.bool_]:
    idx = np.asarray(getattr(a, "slab_index", np.empty(0)), dtype=np.intp)
    inside = np.zeros(z.size, dtype=bool)
    if idx.size and idx.max() < z.size:
        inside[idx] = True
        return inside
    zf = np.where(np.isfinite(z), z, np.nan)
    with np.errstate(invalid="ignore"):
        out: NDArray[np.bool_] = (zf >= a.slab_zmin_nm) & (zf <= a.slab_zmax_nm)
    return out


def build_axial(inp: AxialInputs) -> List[Group]:
    """Every group of the axial view (4.2)."""
    groups: List[Group] = []
    a = inp.analysis if inp.view == "slab" else None
    z = np.asarray(inp.z_roi if inp.z_roi is not None else np.empty(0), float)
    zf = z[np.isfinite(z)]
    n_roi = int(zf.size)
    anchor = (float(a.slab_zmin_nm) if a is not None else
              float(inp.cut[0]) if inp.cut is not None else float(zf.min()) if zf.size else 0.0)
    edges = axial_edges(zf, anchor)
    width = float(edges[1] - edges[0])
    centres = (edges[:-1] + edges[1:]) / 2.0
    norm = max(n_roi, 1) * width
    # --- the ROI before the cut
    given = ("Localizations the analysis was given: the main-window cut selection" if inp.given_after_cut
             else "ROI localizations before the Z cut")
    g = Group("roi", given, caption=("the input of the MPS analysis and of the rings (ROI, no axial cut); 10 nm "
                                     "bars, density over the whole ROI so the fitted mixture sits on them"))
    if a is not None and n_roi:
        inside = _slab_mask(a, z)
        z_in = z[inside & np.isfinite(z)]
        z_out = z[~inside & np.isfinite(z)]
        h_in, _ = np.histogram(z_in, bins=edges)
        h_out, _ = np.histogram(z_out, bins=edges)
        d_in, d_out = h_in / norm, h_out / norm
        g.layers.append(Layer("roi_in", "Inside the MPS analysis slab", g.key, "bars",
                              {"x": centres, "height": d_in, "width": width, "y0": np.zeros_like(d_in),
                               "edges": edges, "z": z_in}, int(z_in.size),
                              "The localizations between the main-window cut lines are what the old 'z ROI "
                              "Histogram' showed for channel 1."))
        g.layers.append(Layer("roi_out", "Outside it", g.key, "bars",
                              {"x": centres, "height": d_out, "width": width, "y0": d_in, "edges": edges,
                               "z": z_out},
                              int(z_out.size), "The rest of the ROI, stacked on the slab's bars."))
    elif n_roi:
        h_all, _ = np.histogram(zf, bins=edges)
        d_all = h_all / norm
        g.layers.append(Layer("roi_out", "All", g.key, "bars",
                              {"x": centres, "height": d_all, "width": width, "y0": np.zeros_like(d_all),
                               "edges": edges, "z": zf}, n_roi, "Every localization of the ROI before the cut."))
    ch2 = None if inp.ch2_z is None else np.asarray(inp.ch2_z, float)
    ch2f = None if ch2 is None else ch2[np.isfinite(ch2)]
    if ch2f is not None and ch2f.size and n_roi:
        e2 = axial_edges(np.concatenate([zf, ch2f]), anchor)
        h2, _ = np.histogram(ch2f, bins=e2)
        g.layers.append(Layer("ch2_roi", "Channel 2 in the ROI, before the cut, as loaded - not registered", g.key,
                              "step", {"edges": e2, "values": h2 / (ch2f.size * float(e2[1] - e2[0])),
                                       "z": ch2f},
                              int(ch2f.size),
                              "Not registered to channel 1: an axial offset between the two may be the rounds' or "
                              "the colours' shift, not a phase. Two channels -> Axial phase puts them in one frame."))
    else:
        g.layers.append(Layer("ch2_roi", "Channel 2 in the ROI, before the cut, as loaded - not registered", g.key,
                              "step", {}, None, "Channel 2's z in the ROI, as loaded.", enabled=False,
                              reason=(inp.ch2_reason if ch2f is None else "channel 2 has no z in the ROI")))
    groups.append(g)
    # --- the mixture
    zr = inp.z_result()
    gm = Group("mixture", "Gaussian mixture of the ROI's z",
               caption=("fit_z_periodicity on the finite z of the ROI before the cut, seeded: the analysis and the "
                        "rings get the same fit. Components = the dominant ones (weight > 5 %; "
                        + (f"{int(zr.n_discarded_components)} left out" if zr is not None else "k left out")
                        + ", n_discarded_components); their weights add up to 1 minus the left-out ones. The "
                        "mixture includes every component and integrates to 1 over the ROI"))
    if zr is None or len(getattr(zr, "means_nm", [])) == 0:
        gm.enabled, gm.reason = False, inp.fit_reason
    else:
        grid_c = rz_grid(a) if a is not None else (np.linspace(float(zf.min()), float(zf.max()), 1024)
                                                   if zf.size else np.linspace(0.0, 1.0, 2))
        curves = component_curves(zr, grid_c)
        comp = Layer("components", "Components (dashed) and their means (dotted)", gm.key, "curves",
                     {"grid": grid_c, "curves": curves, "means": [float(m) for m, s in zip(zr.means_nm, zr.sigmas_nm)
                                                                  if s > 0]},
                     None, "Each dominant component, weighted, and its mean.")
        if inp.view == "segments":
            comp.enabled, comp.reason = False, "orange is segment 3's colour; the segment lines mark the same places"
        elif inp.ch2_shown and ch2f is not None and ch2f.size and n_roi:
            # only while channel 2 is really drawn (its row ticked and available)
            comp.enabled, comp.reason = False, CH2_HIDES_COMPONENTS
        gm.layers.append(comp)
        grid_m = (np.linspace(float(zf.min()), float(zf.max()), 1024) if zf.size else grid_c)
        gm.layers.append(Layer("mixture", "Mixture density (every component)", gm.key, "curves",
                               {"grid": grid_m, "curves": [np.asarray(zr.mixture_density(grid_m), float)]}, None,
                               "The whole fitted mixture, every component included."))
        gm.layers.append(Layer("peak", f"Density peak (the automatic slab centre, {zr.main_peak_nm:,.1f} nm)",
                               gm.key, "vlines", {"positions": [float(zr.main_peak_nm)], "labels": ["density peak"],
                                                  "pens": ["dot"]}, None,
                               "The mode of the fitted mixture: why the automatic slab need not sit on a component "
                               "mean."))
    groups.append(gm)
    # --- the slab and the cut
    gs = Group("slab", "Axial slab", caption=("the analysis keeps a <= z <= b; the main window's selection keeps "
                                              "Z min < z < Z max (rounded to 0.1 nm); one setting since stage 2"))
    if a is not None:
        half = float(a.slab_half_width_nm)
        gs.layers.append(Layer("slab", f"MPS analysis slab ({a.slab_zmin_nm:,.0f} to {a.slab_zmax_nm:,.0f} nm, "
                               f"{a.slab_source}; half-width {half:g} nm, {reg.badge('slab.half_width_nm', half)})",
                               gs.key, "band", {"lo": float(a.slab_zmin_nm), "hi": float(a.slab_zmax_nm)}, None,
                               "The slab the MPS analysis kept."))
    if inp.cut is not None:
        name = "Main-window cut (Z min < z < Z max)"
        if inp.cut_from_histogram_mode:
            name += ", centred on the histogram mode, not on a fitted peak"
        layer = Layer("main_cut", name, gs.key, "vlines",
                      {"positions": [float(inp.cut[0]), float(inp.cut[1])], "labels": ["Z min", "Z max"],
                       "pens": ["dot", "dot"]}, None, "The cut the main window applies to its selection.")
        if inp.cut_stale:
            layer.enabled, layer.reason = False, "the cut of the current selection; the bars are of the analysed one"
        gs.layers.append(layer)
    groups.append(gs)
    # --- the segments
    ms = inp.rings
    gseg = Group("segments", "Segments (rings)")
    if ms is None:
        gseg.caption = "the segments of the Rings window"
        gseg.enabled, gseg.reason = False, inp.rings_reason
    else:
        gseg.caption = (f"find_axial_segments, {_seg_words(ms)}. In paper mode the slabs overlap and a localization "
                        "is in two of them. No analysis slab is marked in this view: the grey bars are all of the ROI")
        for k, seg in enumerate(ms.segments):
            gseg.layers.append(Layer(f"segband{seg.index}", f"Segment {seg.index} slab ({seg.zmin_nm:,.0f} to "
                                     f"{seg.zmax_nm:,.0f} nm)", gseg.key, "band",
                                     {"lo": float(seg.zmin_nm), "hi": float(seg.zmax_nm),
                                      "line": float(seg.center_nm), "label": f"seg {seg.index}"}, None,
                                     f"Segment {seg.index}'s slab and its centre.", segment=k))
        if ms.valleys is not None:
            pos = [float(p) for p in ms.valleys.positions_nm]
            real = [bool(r) for r in ms.valleys.is_true_valley]
            labels = [f"valley {d:.2f}" if r else "no valley" for r, d in zip(real, ms.valleys.relative_depth)]
            gseg.layers.append(Layer("valleys", "Boundaries: valley (solid, depth d) / no valley (dashed)", gseg.key,
                                     "vlines", {"positions": pos, "labels": labels,
                                                "pens": ["solid" if r else "dash" for r in real]}, len(pos),
                                     "Where consecutive slabs meet."))
        for k, (seg, an) in enumerate(zip(ms.segments, ms.analyses)):
            if an is None or not np.asarray(an.z_slab).size:
                continue
            zs = np.asarray(an.z_slab, float)
            e = axial_edges(np.concatenate([zs, zf]) if zf.size else zs, anchor)
            h, _ = np.histogram(zs, bins=e)
            gseg.layers.append(Layer(f"seghist{seg.index}", f"Segment {seg.index}'s own localizations (outline)",
                                     gseg.key, "step", {"edges": e, "values": h / norm, "z": zs, "bounds":
                                                        (float(seg.zmin_nm), float(seg.zmax_nm))},
                                     int(zs.size), f"Segment {seg.index}'s slab, 10 nm, density over the ROI.",
                                     segment=k))
    groups.append(gseg)
    return groups


def axial_title(inp: AxialInputs) -> str:
    """The axial view's exported title, three lines (4.4)."""
    z = np.asarray(inp.z_roi if inp.z_roi is not None else np.empty(0), float)
    n = int(np.isfinite(z).sum())
    what = "z of the main-window cut selection" if inp.given_after_cut else "z of the ROI before the cut"
    line1 = f"{what} ({n:,}), density over the ROI"
    cut = "" if inp.cut is None else f"main-window cut {inp.cut[0]:,.0f}..{inp.cut[1]:,.0f} nm"
    if inp.view == "segments":
        ms = inp.rings
        line2 = ("segments: " + _seg_words(ms) + " (as computed)") if ms is not None else "segments: none computed"
    elif inp.analysis is not None:
        a = inp.analysis
        line2 = f"MPS analysis slab {a.slab_zmin_nm:,.0f}..{a.slab_zmax_nm:,.0f} nm ({a.slab_source})"
        if cut:
            line2 += f"; {cut}"
    else:
        line2 = "No MPS analysis of this selection yet" + (f"; {cut}" if cut else "")
    return f"{line1}<br>{line2}<br>{Z_CALIBRATION_LINE}"


def axial_details(inp: AxialInputs) -> List[DetailRow]:
    """The axial details panel (4.6): numbers the fit already holds; nothing computed."""
    zr = inp.z_result()
    rows: List[DetailRow] = []
    if zr is None:
        return [_row("Gaussian mixture", "- (not fitted)", inp.fit_reason)]
    rows.append(_row("components chosen by BIC", str(int(zr.n_components))))
    for n_comp, bic in sorted((int(k), float(v)) for k, v in dict(zr.bic_by_n).items()):
        rows.append(_row(f"BIC with {n_comp} component{'s' if n_comp != 1 else ''}", f"{bic:,.1f}"))
    for i, (m, s, w) in enumerate(zip(zr.means_nm, zr.sigmas_nm, zr.weights), 1):
        rows.append(_row(f"component {i}", f"mean {float(m):,.1f} nm, sigma {float(s):,.1f} nm, weight {float(w):.2f}"))
    rows.append(_row("components left out (weight <= 5 %)", str(int(zr.n_discarded_components))))
    rows.append(_row("density peak", f"{float(zr.main_peak_nm):,.1f} nm", "the automatic slab centre: the mode of "
                     "the fitted mixture", reg.info("slab.centre").origin))
    dz = zr.mean_delta_z_nm
    rows.append(_row("mean Delta-Z", "- (one component)" if dz is None else f"{dz:,.1f} nm"))
    a = inp.analysis
    if a is not None:
        half = float(a.slab_half_width_nm)
        rows.append(_row("MPS analysis slab", f"{a.slab_zmin_nm:,.1f} to {a.slab_zmax_nm:,.1f} nm", a.slab_source))
        rows.append(_row("slab half-width", f"{half:g} nm", reg.range_hint("slab.half_width_nm", half),
                         reg.badge("slab.half_width_nm", half)))
    if inp.cut is not None:
        rows.append(_row("main-window cut", f"{inp.cut[0]:,.1f} to {inp.cut[1]:,.1f} nm",
                         "centred on the histogram mode" if inp.cut_from_histogram_mode else ""))
    return rows


# ============================================================================ nearest neighbours
NN_DEFAULT_BINS = 30
NN_MAX_N = 10
NN_TITLE_1 = "1NN distance between cluster centres"
CDF_TITLE = "1NN CDF: observed vs randomized"
NN_FIRST_ONLY = "they are of the first neighbour"


def knn_distances(centroids: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Distances from each centre to its 1st..Nth nearest centres: the sklearn KD-tree query the main window ran
    (``tree.query(c, N + 1)``, the centre itself excluded). (K, N)."""
    from sklearn.neighbors import KDTree
    c = np.asarray(centroids, dtype=float).reshape(-1, 2)
    tree = KDTree(c)
    distances, _indexes = tree.query(c, int(n) + 1)
    out: NDArray[np.float64] = np.asarray(distances[:, 1:], dtype=float)
    return out


def nn_max_neighbours(n_clusters: int) -> int:
    """The largest N the spin offers: min(10, clusters - 1); 0 with fewer than two clusters."""
    return int(max(0, min(NN_MAX_N, int(n_clusters) - 1)))


def nn_histogram(shown: Any, n: int = 1, bins: int = NN_DEFAULT_BINS,
                 value_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """The nearest-neighbour histogram of the analysis shown. N = 1 with the range automatic is the old 1NN plot
    exactly (``a.nn.first_nn_nm``, ``np.histogram(v, bins=30)``); N > 1 pools the 1st..Nth distances as the main
    window's histogram did; a range is ``np.histogram(d, bins, range=(a, b))`` and counts what falls outside."""
    if shown is None:
        return {"values": np.empty(0), "counts": np.empty(0), "edges": np.empty(0), "n": n, "outside": 0}
    k = int(shown.n_clusters_kept)
    n = int(max(1, min(int(n), max(1, nn_max_neighbours(k)))))
    if n == 1:
        vals = np.asarray(shown.nn.first_nn_nm if shown.nn is not None else np.empty(0), float)
    else:
        vals = knn_distances(shown.centroids, n).ravel()
    out: Dict[str, Any] = {"values": vals, "n": n, "outside": 0}
    if vals.size == 0:
        out.update(counts=np.empty(0), edges=np.empty(0))
        return out
    if value_range is None:
        counts, edges = np.histogram(vals, bins=int(bins))
    else:
        lo, hi = float(value_range[0]), float(value_range[1])
        if not hi > lo:
            # A range being typed passes through "from 900 to 800": nothing is drawn, and the title says why.
            out.update(counts=np.empty(0), edges=np.empty(0), range_invalid=True)
            return out
        counts, edges = np.histogram(vals, bins=int(bins), range=(lo, hi))
        out["outside"] = int(np.sum((vals < lo) | (vals > hi)))
    out["counts"], out["edges"] = counts, edges
    return out


def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th ... 11th, 12th, 13th, 21st."""
    n = int(n)
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def nn_title(hist: Dict[str, Any], shown: Any, comparison: Any, measured: Any,
             value_range: Optional[Tuple[float, float]]) -> str:
    n = int(hist.get("n", 1))
    words = shown_words(shown, comparison, measured)
    if n == 1:
        title = NN_TITLE_1 + (f" ({words})" if words else "")
    else:
        title = (f"Distances to the 1st..{ordinal(n)} nearest centre, pooled (N = {n}"
                 + (f"; {words}" if words else "") + ")")
    if value_range is not None and hist.get("range_invalid"):
        title += "; nothing drawn: the range's upper bound must be above its lower bound"
    elif value_range is not None and hist.get("outside"):
        title += f"; {int(hist['outside'])} distances outside the range not shown"
    return title


def nn_layers(shown: Any, hist: Dict[str, Any]) -> List[Layer]:
    """Bars, the median line and the reference layer (off by default, P-R25) of the nearest-neighbour tab."""
    layers: List[Layer] = []
    n = int(hist.get("n", 1))
    edges = np.asarray(hist.get("edges", np.empty(0)), float)
    counts = np.asarray(hist.get("counts", np.empty(0)), float)
    if edges.size >= 2:
        centres = (edges[:-1] + edges[1:]) / 2
        width = float(np.mean(np.diff(edges)))
        layers.append(Layer("nn_bars", "Distances" if n > 1 else "1NN distances", "nn", "bars",
                            {"x": centres, "height": counts, "width": width, "y0": np.zeros_like(counts)},
                            int(np.asarray(hist["values"]).size), "The histogram."))
    med = None if shown is None or shown.nn is None else shown.nn.median_1nn_nm
    median = Layer("nn_median", "Median" + ("" if med is None else f" {med:,.0f} nm"), "nn", "vlines",
                   {"positions": [] if med is None else [float(med)],
                    "labels": [] if med is None else [f"median {med:,.0f} nm"], "pens": ["solid"]}, None,
                   "The median 1NN distance.")
    ref_value = float(reg.default("reference.nn1_median_nm"))
    reference = Layer("nn_reference", f"Reference: {reg.PAPER}, another dataset and pipeline", "nn", "vlines",
                      {"positions": [ref_value], "labels": [f"Gazal 2026 (preprint v1): {ref_value:,.0f} nm"],
                       "pens": ["dash"]}, None, reg.tooltip("reference.nn1_median_nm"))
    if n > 1:
        for layer in (median, reference):
            layer.enabled, layer.reason = False, NN_FIRST_ONLY
    if med is None and n == 1:
        median.enabled, median.reason = False, "no 1NN distance"
    layers += [median, reference]
    return layers


NN_ON: FrozenSet[str] = frozenset({"nn_bars", "nn_median"})      # the reference starts off (P-R25)


def nn_details(shown: Any, n: int) -> List[DetailRow]:
    """K, P, clusters per um, the median 1NN, N (12.8)."""
    if shown is None:
        return [_row("clusters", "- (no MPS analysis)")]
    rows = [_row("clusters K (analysis shown)", str(int(shown.n_clusters_kept)))]
    per = shown.perimeter
    rows.append(_row("perimeter P", "- (no contour)" if per is None else f"{per.perimeter_um:,.2f} um"))
    cpu = shown.clusters_per_um
    rows.append(_row("clusters per um", "-" if cpu is None else f"{cpu:,.2f}"))
    med = None if shown.nn is None else shown.nn.median_1nn_nm
    rows.append(_row("median 1NN", "-" if med is None else f"{med:,.1f} nm"))
    rows.append(_row("neighbours N", str(int(n))))
    return rows


def cdf_layers(shown: Any) -> List[Layer]:
    """Observed and randomized 1NN CDFs (the old CDF plot's arrays) and the crossing line, which is this program's
    heuristic: neutral, never the paper colour (P5)."""
    if shown is None or shown.randomization is None:
        return []
    r = shown.randomization

    def cdf(v: Any) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = np.sort(np.asarray(v, dtype=float))
        return s, np.arange(1, s.size + 1) / s.size

    xe, ye = cdf(r.experimental_1nn_nm)
    xr, yr = cdf(r.randomized_1nn_nm)
    layers = [Layer("cdf_randomized", "randomized", "cdf", "curves", {"x": xr, "y": yr}, None,
                    "Every randomized 1NN distance, pooled over the iterations."),
              Layer("cdf_observed", "observed", "cdf", "curves", {"x": xe, "y": ye}, None, "The measured 1NN.")]
    if r.cdf_crossing is not None:
        layers.append(Layer("cdf_crossing", f"crossing {r.cdf_crossing:.2f} (heuristic)", "cdf", "hline",
                            {"y": float(r.cdf_crossing), "label": f"crossing {r.cdf_crossing:.2f} (heuristic)"},
                            None, reg.tooltip("randomization.cdf_crossing_floor")))
    return layers


def cdf_title(shown: Any, comparison: Any, measured: Any) -> str:
    words = shown_words(shown, comparison, measured)
    return CDF_TITLE + (f" ({words})" if words else "")


def cdf_details(shown: Any) -> List[DetailRow]:
    """D, B, the minimum separation, the candidates, the band (paper), incomplete iterations, the mean placed
    fraction, the randomized median. Not the KS p (P-R23)."""
    if shown is None or shown.randomization is None:
        return [_row("randomization", "- (not run)")]
    r = shown.randomization
    band = float(r.annulus_half_width_nm)
    med = r.randomized_median_nm
    return [
        _row("KS statistic D", f"{float(r.ks_statistic):.3f}"),
        _row("randomizations B", f"{int(r.n_iterations):,}", "", reg.badge("randomization.iterations",
                                                                            int(r.n_iterations))),
        _row("minimum separation used", f"{float(r.min_distance_nm):,.1f} nm"),
        _row("candidate positions", f"{int(r.n_candidates):,}"),
        _row("band half-width", f"{band:g} nm", reg.range_hint("randomization.band_half_width_nm", band),
             reg.badge("randomization.band_half_width_nm", band)),
        _row("incomplete iterations", f"{int(r.n_incomplete_iterations):,}", "kept in the pool",
             reg.badge("randomization.incomplete_iterations")),
        _row("mean placed fraction", f"{float(r.mean_placed_fraction):.3f}"),
        _row("randomized median 1NN", "-" if med is None else f"{med:,.1f} nm"),
    ]


def area_details(shown: Any) -> List[DetailRow]:
    """The median area and effective radius with the reference values (12.8)."""
    if shown is None or shown.areas is None:
        return [_row("cluster areas", "- (none)")]
    ar = shown.areas
    med_a = ar.median_area_nm2
    med_r = float(np.median(ar.r_eff_nm)) if np.asarray(ar.r_eff_nm).size else None
    ref_a = float(reg.default("reference.cluster_area_nm2"))
    ref_r = float(reg.default("reference.r_eff_nm"))
    return [
        _row("median area", "-" if med_a is None else f"{med_a:,.0f} nm^2"),
        _row("median effective radius", "-" if med_r is None else f"{med_r:,.1f} nm"),
        _row("reference area", f"{ref_a:,.0f} nm^2", reg.PAPER, "paper"),
        _row("reference effective radius", f"~{ref_r:g} nm", reg.PAPER, "paper"),
    ]


def distances_table(shown: Any, measured: Any, n: int) -> Tuple[List[str], List[NDArray[Any]]]:
    """The "Save distances..." table: ``cluster_label`` and ``nn1_nm..nnN_nm`` of the analysis shown, as the main
    window's "save dist data" wrote it for the measured analysis (the labels of the kept clusters, in the order of
    the centroids; with the discard applied, those not discarded). Returns (column names, column arrays)."""
    dist = knn_distances(shown.centroids, int(n))
    labels = _good_labels(measured)
    if shown.discard_applied:
        labels = labels[~np.isin(labels, list(shown.discarded_labels))]
    names: List[str] = []
    columns: List[NDArray[Any]] = []
    if len(labels) == len(dist):
        names.append("cluster_label")
        columns.append(np.asarray(labels, dtype=int))
    for k in range(dist.shape[1]):
        names.append(f"nn{k + 1}_nm")
        columns.append(dist[:, k])
    return names, columns


def write_distances_csv(path: str, shown: Any, measured: Any, n: int) -> int:
    """Write the table with the formatting "save dist data" used (pandas, ``float_format="%.2f"``, no index): the
    same bytes for the measured analysis. Returns the number of rows."""
    import pandas as pd
    names, columns = distances_table(shown, measured, n)
    data: Dict[str, Any] = dict(zip(names, columns))
    pd.DataFrame(data).to_csv(path, index=False, float_format="%.2f")
    return int(len(columns[-1])) if columns else 0


def default_distances_name(root: str, n: int, shown: Any = None, measured: Any = None) -> str:
    """``<root>_<N>neighbor_distances.csv`` (today's name, for the measured analysis), with ``_discard_applied`` or
    ``_all_clusters`` before the extension for the other two."""
    suffix = ""
    if shown is not None and measured is not None and shown is not measured:
        suffix = "_discard_applied" if shown.discard_applied else "_all_clusters"
    return f"{root}_{int(n)}neighbor_distances{suffix}.csv"
