# -*- coding: utf-8 -*-
"""
The three tables one axon is exported as: the axon, its clusters, its
localizations.

Until now each panel wrote its own file from its own state, and the same
axon could end up described twice, differently: the results window's row
measured with one set of clusters, the axoplasm panel's row with another,
and nothing in either saying so. Here the three tables are built together,
from one state, in one act -- so an axon that reaches the tables at all
reaches them consistent.

One row per axon, with columns
------------------------------
The analysis of every kept cluster and the analysis without the clusters
the axoplasm panel discarded are the SAME axon measured two ways. They go
in one row: the measured columns, and the same ones again with a
``_discard`` suffix. Filtering a column is a decision a reader makes on
purpose; filtering rows of a mixed table is a decision they forget to
make. What the two analyses share -- pixel size, axial slab, DBSCAN,
curation, the settings -- is written once.

What is written once and checked twice
--------------------------------------
Three numbers exist in two places at export time: how many clusters were
discarded, with what margin, and how the widefield images were placed. The
analysis holds them because the discard was applied to it; the panel holds
them because it decided them. They are written once, and the two sources
are compared first: if they disagree, the panel has moved on since the
analysis was re-run, and the export is refused rather than writing a row
that is wrong in a way no reader could detect.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.cluster_quality import describe_roi, good_cluster_labels
from tools.mps_analysis import AxonAnalysis
from tools.mps_axoplasm import SUMMARY_COLUMNS as PANEL_COLUMNS
from tools.mps_identity import FIELDS as IDENTITY_FIELDS
from tools.mps_identity import AxonIdentity, axon_id
from tools.results_table import cell_text

# Bumped when the columns of these tables change, so a table written by
# another version is recognisable in its own column rather than only by
# the export being refused.
#
# v2 (2026-09-22): the animal among the identity columns, and the axoplasm
# panel's columns written for every axon, empty where the panel did not
# measure it -- in v1 a row without the panel had fewer columns than a row
# with it, and the two could not share a table.
PROGRAM_VERSION = "MPS Explorer 2026.09"
TABLE_VERSION = "axon tables v2"

# Which rows describe the same thing, for the duplicate check. ``axon_id``
# is the file and the selection, and nothing else: the same axon exported
# before and after its genotype was typed in must still be recognised as
# the same axon, in all three tables at once.
AXON_KEY_COLUMNS: Tuple[str, ...] = ("axon_id",)
CLUSTER_KEY_COLUMNS: Tuple[str, ...] = ("axon_id", "cluster_label")
LOCALIZATION_KEY_COLUMNS: Tuple[str, ...] = ("axon_id", "source_index")

# --------------------------------------------------------------------------
# How the per-axon row is laid out
# --------------------------------------------------------------------------

# Who this axon is. ``roi`` is the geometric selection ("circle centred at
# ..."), roi_name the measurement it belongs to ("ROI 1"): two axons of one
# ROI share the second and not the first.
HEAD_COLUMNS: Tuple[str, ...] = (
    ("axon_id", "analysis_id") + IDENTITY_FIELDS
    + ("source", "roi", "exported_at", "program_version", "table_version"))

# What this row describes. Each of these is written once even where two
# objects hold it.
STATE_COLUMNS: Tuple[str, ...] = (
    "discard_applied", "discard_margin_nm", "discard_registration",
    "n_clusters_discarded", "axoplasm_measured")

# Columns of AxonAnalysis.export_dict that the discard cannot change: the
# file, the axial slab, the clustering and the settings steps 3-6 ran with.
# Written once.
SHARED_COLUMNS: Tuple[str, ...] = (
    "pixel_size_nm", "pixel_size_source", "eps_nm", "min_samples",
    "dbcv_threshold", "edge_reference", "slab_half_width_nm",
    "slab_zmin_nm", "slab_zmax_nm", "slab_source", "z_main_peak_auto_nm",
    "n_locs_total", "n_locs_slab", "gmm_n_components", "gmm_means_nm",
    "gmm_weights", "gmm_sigmas_nm", "gmm_converged",
    "gmm_n_discarded_components", "delta_z_mean_nm", "delta_z_values_nm",
    "n_clusters_raw", "n_clusters_removed", "removed_edge_touching",
    "removed_low_dbcv", "edge_criterion_disabled", "mahalanobis_threshold",
    "ellipse_mode", "random_seed", "randomization_requested")

# Columns the discard does change: everything from the clusters that are
# left onwards. These are the ones written twice, the second time with a
# "_discard" suffix.
MEASURED_COLUMNS: Tuple[str, ...] = (
    "n_clusters_kept", "perimeter_um", "clusters_per_um", "contour_2opt",
    # Where the order of the tour came from. Written twice, plain and
    # _discard, because the discard analysis has its own contour: an
    # order set by hand on the measured one does not make the other
    # hand-set.
    "contour_order_source",
    "contour_hull_um", "contour_tour_over_hull", "contour_max_over_median",
    "contour_n_deep_vertices", "contour_hull_radius_nm",
    "contour_deep_limit_nm", "contour_max_depth_nm",
    "contour_length_in_long_edges", "centre_x_nm", "centre_y_nm",
    "contour_area_um2", "centre_max_shift_nm", "median_area_nm2",
    "median_r_eff_nm", "median_1nn_nm", "occupancy_percent",
    "occupied_length_nm", "occupancy_n_points",
    "occupancy_n_capped_ellipses", "ks_statistic", "ks_cdf_crossing",
    "randomization_n_iterations", "randomization_min_sep_nm",
    "randomization_incomplete_iters", "randomized_median_1nn_nm",
    "ks_pvalue", "n_warnings", "warnings")

DISCARD_SUFFIX = "_discard"

# Columns of the axoplasm panel's row that the axon row already holds, and
# that must not be written a second time under another name.
_AXOPLASM_DROPPED: Tuple[str, ...] = (
    "source_localizations", "roi", "pixel_size_nm", "n_clusters_discarded",
    "discard_registration", "margin_nm")

# Prefixed, so "n_warnings" of the panel and "n_warnings" of the analysis
# cannot be confused for one another in a wide row.
AXOPLASM_PREFIX = "axoplasm_"


# How much of a number is kept, by what the number is. One length in a row
# written to 15 digits beside another written to 3 invites the reader to
# believe the first one: nothing here is measured to a femtometre.
ROUNDING: Tuple[Tuple[str, int], ...] = (
    ("_nm2", 1), ("_um2", 4), ("_nm", 1), ("_um", 3), ("_percent", 2),
)
# Fractions and the statistics that behave like one.
_FOUR_DECIMALS = frozenset({"ks_statistic", "contour_tour_over_hull",
                            "contour_length_in_long_edges"})
# Not measurements: what the analysis was RUN with is written exactly as
# it was given, however many digits that takes.
_SETTINGS: frozenset = frozenset({
    "pixel_size_nm", "eps_nm", "min_samples", "dbcv_threshold",
    "mahalanobis_threshold", "slab_half_width_nm", "discard_margin_nm",
    "random_seed", "randomization_requested", "randomization_min_sep_nm",
})


def _rounded(name: str, value: Any) -> Any:
    """``value`` kept to the precision its name says it has."""
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    if not math.isfinite(value):
        return value
    base = (name[:-len(DISCARD_SUFFIX)] if name.endswith(DISCARD_SUFFIX)
            else name)
    if base in _SETTINGS or base.endswith("pvalue"):
        return value
    if base in _FOUR_DECIMALS or base.startswith("fraction_"):
        return round(value, 4)
    for suffix, digits in ROUNDING:
        if base.endswith(suffix):
            return round(value, digits)
    return value


class ExportConflict(ValueError):
    """Two parts of the program describe this axon differently."""


def _unknown_columns(record: Dict[str, Any]) -> List[str]:
    """Columns of an export_dict this module does not place anywhere.

    A column added to AxonAnalysis.export_dict and to no list here would
    vanish from the axon table without a word; it is raised instead.
    """
    placed = set(SHARED_COLUMNS) | set(MEASURED_COLUMNS) | {
        "source", "roi", "cluster_set", "discard_margin_nm",
        "discard_registration", "n_clusters_discarded"}
    return sorted(set(record) - placed)


def analysis_id(analysis: AxonAnalysis) -> str:
    """
    A short name for THIS analysis: these localizations, these parameters,
    this version of the program.

    Two exports of one analysis give the same value; changing any
    parameter, or analysing another selection, gives another. It is what
    joins the per-cluster and per-localization tables to the row they were
    written with, and what says that two rows of a pooled table were not
    measured the same way.
    """
    coords = np.concatenate([
        np.asarray(analysis.x_slab, dtype=np.float64).ravel(),
        np.asarray(analysis.y_slab, dtype=np.float64).ravel(),
        np.asarray(analysis.z_slab, dtype=np.float64).ravel()])
    data = hashlib.sha1(np.ascontiguousarray(coords).tobytes()).hexdigest()
    parts = [
        PROGRAM_VERSION, TABLE_VERSION, data,
        os.path.basename(str(analysis.source_name)),
        describe_roi(analysis.roi),
        f"{analysis.pixel_size_nm}", analysis.pixel_size_source,
        f"{analysis.eps_nm}", f"{analysis.min_samples}",
        f"{analysis.dbcv_threshold}", analysis.edge_reference,
        f"{analysis.slab_zmin_nm:.4f}", f"{analysis.slab_zmax_nm:.4f}",
        analysis.slab_source, f"{analysis.mahalanobis_threshold}",
        analysis.ellipse_mode, f"{analysis.random_seed}",
        f"{analysis.n_randomizations if analysis.run_randomization else 0}",
        str(analysis.contour_2opt), analysis.cluster_set,
        f"{analysis.discard_margin_nm}",
    ]
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:10]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _same(a: Any, b: Any) -> bool:
    """Whether two values from two parts of the program agree. A value
    missing on one side is not a disagreement: it is not measured there."""
    if a is None or b is None or a == "" or b == "":
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-6
    return str(a) == str(b)


def _check_agreement(measured: AxonAnalysis,
                     discard: Optional[AxonAnalysis],
                     axoplasm: Optional[Dict[str, Any]]) -> None:
    """Refuse an export whose parts describe different states of one axon."""
    if axoplasm is None:
        return
    source = os.path.basename(str(measured.source_name))
    panel_source = os.path.basename(
        str(axoplasm.get("source_localizations") or ""))
    if panel_source and panel_source != source:
        raise ExportConflict(
            f"The axoplasm panel is showing {panel_source} and the analysis "
            f"is of {source}. Reopen the panel on the axon being exported.")
    roi = describe_roi(measured.roi)
    panel_roi = str(axoplasm.get("roi") or "")
    if panel_roi and roi and panel_roi != roi:
        raise ExportConflict(
            f"The axoplasm panel is showing another selection of this file "
            f"({panel_roi}) than the analysis ({roi}). Re-run the analysis "
            f"on the selection you want to export, or reopen the panel.")
    if discard is None:
        return
    pairs = (
        ("how many clusters were discarded", discard.n_clusters_discarded,
         axoplasm.get("n_clusters_discarded")),
        ("the discard margin", discard.discard_margin_nm,
         axoplasm.get("margin_nm")),
        ("how the images were placed", discard.discard_registration,
         axoplasm.get("discard_registration")),
    )
    for what, from_analysis, from_panel in pairs:
        if not _same(from_analysis, from_panel):
            raise ExportConflict(
                f"The analysis and the axoplasm panel disagree about {what}: "
                f"{from_analysis!r} against {from_panel!r}. The panel has "
                f"changed since the analysis was re-run with the discard; "
                f"press 'Discard applied' again before exporting.")


def axon_row(
    measured: AxonAnalysis,
    *,
    identity: Optional[AxonIdentity] = None,
    discard: Optional[AxonAnalysis] = None,
    axoplasm: Optional[Dict[str, Any]] = None,
    exported_at: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One axon as one row: who it is, what state it is in, the analysis of
    every kept cluster, the same without the discarded ones, and what the
    axoplasm panel measured.

    ``measured`` is the analysis of all the kept clusters and ``discard``
    the same analysis without the ones the panel discarded -- the pair
    ``compare_discard`` returns, which differ only by those clusters.
    ``axoplasm`` is the panel's own row (``mps_axoplasm.summary_row``).
    """
    if measured.discard_applied:
        raise ValueError(
            "The measured analysis is the one with all the kept clusters; "
            "this one has the discard applied.")
    if discard is not None and not discard.discard_applied:
        raise ValueError(
            "The second analysis given has no discard applied, so it is not "
            "the other half of the comparison.")
    _check_agreement(measured, discard, axoplasm)

    record = measured.export_dict()
    unknown = _unknown_columns(record)
    if unknown:
        raise ValueError(
            f"The analysis exports columns this table does not place: "
            f"{', '.join(unknown)}. Add them to SHARED_COLUMNS or to "
            f"MEASURED_COLUMNS in tools/axon_export.py.")

    roi = describe_roi(measured.roi)
    ident = identity or AxonIdentity()
    row: Dict[str, Any] = {
        "axon_id": axon_id(measured.source_name, roi),
        "analysis_id": analysis_id(measured),
    }
    row.update(ident.columns())
    row.update({
        "source": measured.source_name,
        "roi": roi,
        "exported_at": exported_at or _now(),
        "program_version": PROGRAM_VERSION,
        "table_version": TABLE_VERSION,
    })

    # --- state, written once whichever object holds it ------------------
    from_panel = axoplasm or {}
    n_discarded = (discard.n_clusters_discarded if discard is not None
                   else from_panel.get("n_clusters_discarded"))
    margin = (discard.discard_margin_nm if discard is not None
              else from_panel.get("margin_nm"))
    registration = (discard.discard_registration if discard is not None
                    else from_panel.get("discard_registration")) or None
    row.update({
        "discard_applied": discard is not None,
        "discard_margin_nm": margin,
        "discard_registration": registration,
        "n_clusters_discarded": n_discarded,
        "axoplasm_measured": axoplasm is not None,
    })

    # --- the analysis ---------------------------------------------------
    for name in SHARED_COLUMNS:
        row[name] = _rounded(name, record[name])
    for name in MEASURED_COLUMNS:
        row[name] = _rounded(name, record[name])
    second = discard.export_dict() if discard is not None else None
    for name in MEASURED_COLUMNS:
        row[name + DISCARD_SUFFIX] = (
            None if second is None else _rounded(name, second[name]))

    # --- the panel ------------------------------------------------------
    # Every column of the panel's row, whether the panel measured this
    # axon or not. A table holds one layout of columns, and the row of an
    # axon the panel never saw -- every row of a batch -- has to fit beside
    # one it did; axoplasm_measured says which is which. A column of the
    # panel this list does not know is raised, as for the analysis above:
    # left out, it would vanish from the table without a word.
    panel = dict(axoplasm or {})
    unplaced = sorted(set(panel) - set(PANEL_COLUMNS))
    if unplaced:
        raise ValueError(
            f"The axoplasm panel exports columns this table does not place: "
            f"{', '.join(unplaced)}. Add them to SUMMARY_COLUMNS in "
            f"tools/mps_axoplasm.py.")
    for name in PANEL_COLUMNS:
        if name in _AXOPLASM_DROPPED:
            continue
        row[AXOPLASM_PREFIX + name] = panel.get(name)
    return row


# --------------------------------------------------------------------------
# One row per cluster
# --------------------------------------------------------------------------

CLUSTER_COLUMNS: Tuple[str, ...] = (
    "cluster_label", "centroid_x_nm", "centroid_y_nm", "n_localizations",
    "area_nm2", "r_eff_nm", "nn_1_nm", "nn_1_label", "contour_position",
    "discarded", "contour_position_discard", "nn_1_nm_discard",
    "nn_1_label_discard")


def _contour_positions(analysis: AxonAnalysis,
                       labels: NDArray[np.int64]) -> Dict[int, int]:
    """Where each cluster sits along the reconstructed contour.

    Without it the perimeter cannot be recomputed from an exported table:
    the tour is the measurement, and the centroids alone do not give it.
    """
    perimeter = analysis.perimeter
    if perimeter is None or len(labels) == 0:
        return {}
    order = np.asarray(perimeter.order, dtype=int).ravel()
    if order.size != len(labels):
        return {}
    return {int(labels[row]): position for position, row in enumerate(order)}


def _by_label(analysis: AxonAnalysis) -> Tuple[NDArray[np.int64],
                                               Dict[str, Dict[int, Any]]]:
    """The clusters this analysis measured, and what it measured per label.

    The clusters the discard left out are left out here too: an analysis
    with the discard applied keeps the same curation report, so they are
    only recognisable by ``discarded_labels``, and counting them in would
    misalign every per-cluster array by as many rows.
    """
    labels = good_cluster_labels(
        analysis.labels,
        set(analysis.bad_report.bad_labels) | set(analysis.discarded_labels))
    per: Dict[str, Dict[int, Any]] = {"area": {}, "r_eff": {}, "n_locs": {},
                                      "nn": {}, "nn_label": {},
                                      "position": {}}
    areas = analysis.areas
    if areas is not None:
        for label, area, r_eff, n_locs in zip(areas.labels, areas.areas_nm2,
                                              areas.r_eff_nm, areas.n_locs):
            per["area"][int(label)] = round(float(area), 1)
            per["r_eff"][int(label)] = round(float(r_eff), 1)
            per["n_locs"][int(label)] = int(n_locs)
    nn = analysis.nn
    if nn is not None and nn.first_nn_nm.size == len(labels):
        index = nn.neighbour_index
        for row, distance in enumerate(nn.first_nn_nm):
            per["nn"][int(labels[row])] = round(float(distance), 1)
            if index is not None and len(index) == len(labels):
                per["nn_label"][int(labels[row])] = int(
                    labels[int(index[row][0])])
    per["position"] = _contour_positions(analysis, labels)
    return labels, per


def cluster_table(
    measured: AxonAnalysis,
    *,
    discard: Optional[AxonAnalysis] = None,
    axoplasm_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    One row per kept cluster: where it is, how big it is, who its nearest
    neighbour is, where it sits on the contour, and whether the discard
    left it out.

    ``axoplasm_rows`` are the panel's own cluster rows
    (``mps_axoplasm.cluster_rows``), joined by ``cluster_label``. The
    columns they bring -- which side of each image a cluster is on, and the
    group that follows -- are the panel's decision, not a recomputation
    from depths rounded past it.
    """
    labels, per = _by_label(measured)
    head = _head(measured)
    dropped = set(discard.discarded_labels) if discard is not None else set()
    second: Dict[str, Dict[int, Any]] = {}
    if discard is not None:
        _, second = _by_label(discard)

    panel: Dict[int, Dict[str, Any]] = {}
    for row in (axoplasm_rows or []):
        label = row.get("cluster_label")
        if label is None or label == "":
            continue
        panel[int(label)] = row

    centroids = np.asarray(measured.centroids, dtype=float)
    rows: List[Dict[str, Any]] = []
    for position, label in enumerate(labels):
        label = int(label)
        out: Dict[str, Any] = dict(head)
        out.update({
            "cluster_label": label,
            "centroid_x_nm": (round(float(centroids[position][0]), 1)
                              if position < len(centroids) else None),
            "centroid_y_nm": (round(float(centroids[position][1]), 1)
                              if position < len(centroids) else None),
            "n_localizations": per["n_locs"].get(label),
            "area_nm2": per["area"].get(label),
            "r_eff_nm": per["r_eff"].get(label),
            "nn_1_nm": per["nn"].get(label),
            "nn_1_label": per["nn_label"].get(label),
            "contour_position": per["position"].get(label),
            "discarded": (None if discard is None else label in dropped),
            "contour_position_discard": (
                None if discard is None else second["position"].get(label)),
            "nn_1_nm_discard": (None if discard is None
                                else second["nn"].get(label)),
            "nn_1_label_discard": (None if discard is None
                                   else second["nn_label"].get(label)),
        })
        for name, value in panel.get(label, {}).items():
            if name in ("source_localizations", "roi", "cluster_label",
                        "x_nm", "y_nm", "discarded"):
                continue
            out[AXOPLASM_PREFIX + name] = value
        rows.append(out)
    return rows


# --------------------------------------------------------------------------
# One row per localization
# --------------------------------------------------------------------------

def localization_table(
    measured: AxonAnalysis,
    *,
    discard: Optional[AxonAnalysis] = None,
    x_nm: Optional[NDArray[np.float64]] = None,
    y_nm: Optional[NDArray[np.float64]] = None,
    z_nm: Optional[NDArray[np.float64]] = None,
    axoplasm_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    One row per localization of the selection, and the warnings the caller
    should show.

    ``x_nm``/``y_nm``/``z_nm`` are the localizations the analysis was given,
    all of them; when they are not supplied only the ones inside the axial
    slab are written, which is what the analysis kept. ``source_index`` is
    the position in that selection, so the table joins back to the file the
    localizations were read from.

    ``axoplasm_rows`` are the panel's per-localization rows. They are only
    joined when they describe the same localizations in the same order,
    which is checked coordinate by coordinate: the panel's selection is cut
    at 0.1 nm on the text of the range and the analysis' slab is inclusive,
    so the two have differed by one localization, and a join by position
    would then attribute every distance to the wrong point.
    """
    warnings: List[str] = []
    head = _head(measured)
    slab_index = np.asarray(measured.slab_index, dtype=np.intp).ravel()
    labels = np.asarray(measured.labels, dtype=np.int64).ravel()
    kept = set(int(v) for v in good_cluster_labels(
        labels, set(measured.bad_report.bad_labels)))
    dropped = set(discard.discarded_labels) if discard is not None else set()

    have_all = (x_nm is not None and y_nm is not None and z_nm is not None)
    if have_all:
        xs = np.asarray(x_nm, dtype=float).ravel()
        ys = np.asarray(y_nm, dtype=float).ravel()
        zs = np.asarray(z_nm, dtype=float).ravel()
        if not (len(xs) == len(ys) == len(zs) == measured.n_locs_total):
            raise ValueError(
                f"{len(xs)} localizations given for an analysis of "
                f"{measured.n_locs_total}.")
        if slab_index.size != len(measured.x_slab):
            have_all = False
            warnings.append(
                "This analysis did not record which localizations its slab "
                "holds, so only the ones it analysed are written.")
    if have_all:
        # Which of the localizations given are in the slab, and where each
        # one sits in the file they were read from.
        in_slab = np.zeros(len(xs), dtype=bool)
        in_slab[slab_index] = True
        label_of = np.full(len(xs), -1, dtype=np.int64)
        if labels.size == slab_index.size:
            label_of[slab_index] = labels
        index_of = np.arange(len(xs), dtype=np.intp)
    else:
        xs = np.asarray(measured.x_slab, dtype=float).ravel()
        ys = np.asarray(measured.y_slab, dtype=float).ravel()
        zs = np.asarray(measured.z_slab, dtype=float).ravel()
        in_slab = np.ones(len(xs), dtype=bool)
        label_of = (labels if labels.size == len(xs)
                    else np.full(len(xs), -1, dtype=np.int64))
        # Still the position in the selection, not in the slab, whenever
        # the analysis recorded it: a table whose index means one thing in
        # one export and another in the next cannot be joined to anything.
        index_of = (slab_index if slab_index.size == len(xs)
                    else np.arange(len(xs), dtype=np.intp))

    panel = _panel_by_localization(axoplasm_rows, xs, ys, zs,
                                   warnings)

    rows: List[Dict[str, Any]] = []
    for i in range(len(xs)):
        label = int(label_of[i])
        clustered = bool(in_slab[i]) and label >= 0
        out: Dict[str, Any] = dict(head)
        out.update({
            "source_index": int(index_of[i]),
            "x_nm": round(float(xs[i]), 2),
            "y_nm": round(float(ys[i]), 2),
            "z_nm": round(float(zs[i]), 2),
            "in_slab": bool(in_slab[i]),
            # Empty rather than -1: a cluster label of -1 is DBSCAN's noise
            # and reads as a number in every program that opens the table.
            "cluster_label": label if clustered else None,
            "cluster_kept": (None if not clustered else label in kept),
            "cluster_discarded": (None if discard is None or not clustered
                                  else label in dropped),
        })
        for name, value in (panel[i] if panel is not None else {}).items():
            out[AXOPLASM_PREFIX + name] = value
        rows.append(out)
    return rows, warnings


_PANEL_LOCALIZATION_DROPPED = ("source_localizations", "roi", "x_nm", "y_nm",
                               "z_nm", "cluster_label")


def _panel_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    return {name: value for name, value in row.items()
            if name not in _PANEL_LOCALIZATION_DROPPED}


def _panel_by_localization(
    axoplasm_rows: Optional[Sequence[Dict[str, Any]]],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    warnings: List[str],
) -> Optional[List[Dict[str, Any]]]:
    """
    The panel's rows lined up with these localizations, by where each one
    is -- never by its position in the list.

    The panel measures the selection cut to the axial range typed in the
    main window, and the analysis is given the selection before that cut,
    so the two lists differ in length on any real axon (10,034 against
    23,743 on axon 7) and their order is not the same either. Lining them
    up by position would put every distance on the wrong localization,
    quietly. Localizations that share a position to the centinanometre are
    left out of the join rather than guessed between.
    """
    if not axoplasm_rows:
        return None
    if len(axoplasm_rows) == len(xs) and all(
            abs(float(row.get("x_nm", 0.0)) - round(float(x), 2)) <= 0.011
            and abs(float(row.get("y_nm", 0.0)) - round(float(y), 2)) <= 0.011
            and abs(float(row.get("z_nm", 0.0)) - round(float(z), 2)) <= 0.011
            for row, x, y, z in zip(axoplasm_rows, xs, ys, zs)):
        return [_panel_columns(row) for row in axoplasm_rows]

    by_place: Dict[Tuple[float, float, float], Optional[Dict[str, Any]]] = {}
    ambiguous = 0
    for row in axoplasm_rows:
        try:
            key = (round(float(row["x_nm"]), 2), round(float(row["y_nm"]), 2),
                   round(float(row["z_nm"]), 2))
        except (KeyError, TypeError, ValueError):
            continue
        if key in by_place:
            if by_place[key] is not None:
                ambiguous += 1
            by_place[key] = None          # two rows here: join neither
            continue
        by_place[key] = _panel_columns(row)

    out: List[Dict[str, Any]] = []
    matched = 0
    for x, y, z in zip(xs, ys, zs):
        found = by_place.get((round(float(x), 2), round(float(y), 2),
                              round(float(z), 2)))
        out.append(found or {})
        if found:
            matched += 1
    if matched == 0:
        warnings.append(
            f"The axoplasm panel is showing {len(axoplasm_rows)} other "
            f"localizations than the ones analysed, so what it measured per "
            f"localization is not written.")
        return None
    if matched < len(xs):
        warnings.append(
            f"{matched:,} of the {len(xs):,} localizations carry what the "
            f"axoplasm panel measured; the panel sees the selection cut to "
            f"the axial range, and those cells are empty for the rest.")
    if ambiguous:
        warnings.append(
            f"{ambiguous} localization(s) of the panel share a position with "
            f"another to 0.01 nm, so what it measured there is not written: "
            f"which of the two a row would carry cannot be decided.")
    return out


def _head(measured: AxonAnalysis) -> Dict[str, Any]:
    """
    What a row of the clusters or the localizations table starts with.

    Two columns, and not the path, the ROI or the identity: those are in
    the axon's own row, which these join to by ``axon_id``. Repeating them
    per localization was most of the bytes of that table -- and a second
    place for them to disagree.
    """
    return {"axon_id": axon_id(measured.source_name,
                               describe_roi(measured.roi)),
            "analysis_id": analysis_id(measured)}


# --------------------------------------------------------------------------
# The three together
# --------------------------------------------------------------------------

@dataclass
class AxonTables:
    """What one export writes, and what it has to say about it."""

    axon: Dict[str, Any]
    clusters: List[Dict[str, Any]] = field(default_factory=list)
    localizations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def build_tables(
    measured: AxonAnalysis,
    *,
    identity: Optional[AxonIdentity] = None,
    discard: Optional[AxonAnalysis] = None,
    axoplasm: Optional[Dict[str, Any]] = None,
    axoplasm_clusters: Optional[Sequence[Dict[str, Any]]] = None,
    axoplasm_localizations: Optional[Sequence[Dict[str, Any]]] = None,
    with_clusters: bool = False,
    with_localizations: bool = False,
    x_nm: Optional[NDArray[np.float64]] = None,
    y_nm: Optional[NDArray[np.float64]] = None,
    z_nm: Optional[NDArray[np.float64]] = None,
) -> AxonTables:
    """
    Build the axon's row, and the tables that were asked for, from one
    state and one timestamp.

    The per-cluster and per-localization tables are written on request: a
    folder of axons needs the row, and the other two are what a particular
    question needs -- and they are large.
    """
    stamp = _now()
    row = axon_row(measured, identity=identity, discard=discard,
                   axoplasm=axoplasm, exported_at=stamp)
    tables = AxonTables(axon=row)
    tables.warnings.extend(
        w for w in (identity.warnings() if identity is not None else []))
    if with_clusters:
        tables.clusters = cluster_table(
            measured, discard=discard, axoplasm_rows=axoplasm_clusters)
    if with_localizations:
        rows, warnings = localization_table(
            measured, discard=discard, x_nm=x_nm, y_nm=y_nm, z_nm=z_nm,
            axoplasm_rows=axoplasm_localizations)
        tables.localizations = rows
        tables.warnings.extend(warnings)
    return tables


def table_paths(path: str) -> Dict[str, str]:
    """Where the three tables of one export go, given the axon table."""
    base, ext = os.path.splitext(path)
    ext = ext or ".csv"
    return {"axon": base + ext,
            "clusters": f"{base}_clusters{ext}",
            "localizations": f"{base}_localizations{ext}"}


def describe(tables: AxonTables) -> str:
    """One line per table, for the message that says what was written."""
    parts = [f"1 axon ({tables.axon.get('source', '')})"]
    if tables.clusters:
        parts.append(f"{len(tables.clusters)} clusters")
    if tables.localizations:
        parts.append(f"{len(tables.localizations)} localizations")
    return cell_text(", ".join(parts))
