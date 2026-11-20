"""
Columns of the automatic discard of unanchored clusters (axon tables v18, package P14).

* ``UNANCHORED_COLUMNS`` is the block written at the very END of the per-axon row, after the v17 block of the
  perimeter density analysis (``tools.perimeter_export.PERIMETER_COLUMNS``). Its order is fixed; a column is only
  ever appended, never moved. ``not_run_block()`` is the block of an axon on which the discard was not evaluated:
  every column empty but ``unanchored_status`` = "not run".
* ``CLUSTER_UNANCHORED_COLUMNS`` are written at the end of the per-cluster table (after ``cut_by_pick``), empty when
  the discard was not evaluated.

The values come from ``tools.mps_unanchored.unanchored_block`` and ``tools.mps_unanchored.cluster_rows``. This
module holds only the names, so that the export and the column dictionary can import it without numpy or scipy.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

UNANCHORED_COLUMNS: Tuple[str, ...] = (
    # --- how it was run and whether it could be
    "unanchored_method", "unanchored_status", "unanchored_failure",
    # --- the automatic rule (decisions D1-D3)
    "unanchored_reference", "unanchored_exclude_nm", "unanchored_doubt_nm",
    "unanchored_reference_noise_nm", "unanchored_inner_band_arc_nm",
    "unanchored_n_clusters", "unanchored_n_auto", "unanchored_n_doubtful", "unanchored_n_borderline",
    "unanchored_max_depth_nm",
    # --- the other reference ring (D3: how many clusters change verdict with the ring)
    "unanchored_other_reference", "unanchored_other_status", "unanchored_n_auto_other",
    "unanchored_n_changed_by_reference",
    # --- the old widefield method, when it was run (D5)
    "unanchored_n_widefield",
    # --- the decisions by hand (D11) and the final set
    "unanchored_n_excluded_manual", "unanchored_n_restored_manual", "unanchored_n_final", "unanchored_fraction",
    "unanchored_gate_exceeded", "unanchored_discard_source",
    "unanchored_labels_auto", "unanchored_labels_doubtful", "unanchored_labels_excluded_manual",
    "unanchored_labels_restored_manual", "unanchored_manual_reasons", "unanchored_manual_file",
    "unanchored_manual_fingerprint", "unanchored_manual_n_orphan",
    "unanchored_warnings",
    # --- the trim of interior localizations before the perimeter contour (D8, off by default)
    "pd_contour_trim_nm", "pd_contour_n_locs_trimmed",
)

CLUSTER_UNANCHORED_COLUMNS: Tuple[str, ...] = (
    "depth_below_ring_nm", "anchor_state", "anchor_borderline", "anchor_source", "anchor_reason")

NOT_RUN: str = "not run"


def not_run_block() -> Dict[str, Any]:
    """The block of an axon on which the discard was not evaluated: every column empty but the status."""
    block: Dict[str, Any] = {name: None for name in UNANCHORED_COLUMNS}
    block["unanchored_status"] = NOT_RUN
    block["unanchored_method"] = NOT_RUN
    return block


def empty_cluster_row() -> Dict[str, Any]:
    """The cluster columns of a cluster the discard did not evaluate."""
    return {name: None for name in CLUSTER_UNANCHORED_COLUMNS}
