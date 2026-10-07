# -*- coding: utf-8 -*-
"""
The lumen review window of the H-ECL column analysis (milestone H5-D;
DECISIONES D-35(b)): the clusters of every ring of one axon, which of them
the automatic lumen rule removed and which it doubts, the user's own
removals and restorations, and the ring analyses re-run on the final set.

What the window is for (the user's request, Q-18/Q-19)
------------------------------------------------------
The automatic rule (``tools.mps_lumen``, D-35(a)) removes a cluster only
when it is sure the cluster sits in the lumen ("sacar solo con
seguridad"). Everything else it has a reason to doubt -- flagged by the
widefield images at margin 0 or 250 nm, or by the sensitive isolation
criterion -- is DOUBTFUL and kept unless the user removes it. This window
shows the doubtful clusters differently from the rest, lets the user
remove or keep them one by one (a checkbox in the table, or a click on
the marker) or all together, remove any other cluster by hand and give
back one the rule removed, and then re-run the analyses of the rings on
the final set with one button, in a worker thread, so the window never
freezes.

What it shows, and what it does not claim
-----------------------------------------
* The primary test is the arc test on the centroid membrane of the
  cleaned rings (``cmX-k400-it2-L``, D-35c); the 2D test is a
  sensitivity; the arc test on the localization membrane is a diagnostic
  with its known conservative bias printed beside it.
* None of them is calibrated for the axial leak yet (H5-E): a banner
  says so whenever results are on screen.
* Provenance (R9/R10): every decision is logged in order; a change of the
  cluster set made AFTER column results of the axon were shown sets
  "edited after results were shown" in the window and in every export,
  and is never reset. A set with manual edits cannot be reproduced by the
  simulated null (``power_columns.py simnull --lumen-clean`` applies the
  automatic rule only); the results say so.
* Without widefield images the rule is isolation only and its
  completeness is not certified: a warning stays visible.
* Which ring pairs are analysed (H6, D-41): the viability rule v2
  (``tools.mps_axial_precision.viability_v2``: SiZer peak and valley,
  the localization minimum of the central ring, the leak tiers) is
  computed on the rings as built when the window opens and shown pair by
  pair. The run analyses the VIABLE pairs only (the arc test's joint null
  restricted to them, ``batch_columns.restricted_joint``: the batch
  runner's numbers); MARGINAL pairs add a sensitivity line; NOT VIABLE
  pairs are never shown. With no viable or marginal pair nothing is
  computed and the window says why. Every p is labelled "not calibrated"
  (H5-E closed as not accepted, Q-33). Beside the run: "Z quality..."
  (tools/mps_zquality_window.py), the optional "Simulated-null p (not
  calibrated)..." (tools/mps_simnull_ui.py) and "Columns batch..."
  (tools/mps_columns_batch_ui.py).
* The criteria switches (H6 toggles, D-43, the user's request of
  2026-10-02): above the viability table, the program's one selection
  (``tools.mps_selection_ui``). Each criterion of rule v2 and of rule v2c
  can be checked and unchecked; the table, the run's tiers and the results
  follow in real time. The analyses of a cluster set do not depend on the
  selection (every pair is analysed; the selection restricts the joint and
  the lines shown), so once a run has computed them a toggle re-renders in
  milliseconds; only when the last run computed nothing (no pair was
  selected) does a toggle start a run, in the background, after 300 ms
  ("Live update"; a newer request supersedes an unneeded one). Anything but
  the pre-specified rule is an EXPLORATORY selection: the first line of the
  results, the banner, the viability table and every export say so, and
  each selection a p was shown under is logged (``selection_exploration_log
  .csv``), with "N selection variants tried this session" next to the p.
  Under the pre-specified rule the window's texts and exports are exactly
  what they were.

How the axon gets here
----------------------
``start_columns_review`` (the Rings panel's button, through
``MPS_explorer.open_columns_review``) builds the H-ECL rings once, in a
worker, from the ROI's localizations BEFORE the axial cut with their
frames and precisions (the H1 hook ``roi_indices_unfiltered``), with the
pre-registered parameters (``config/columns_params.yaml``,
D-22) and WITHOUT the ROI's edge (``roi=None``, as simnull and the certified
classification build them), classifies the clusters
(``tools.mps_lumen.classify_lumen``), and opens the window. Pressing the
button again for the same selection brings the open review forward; the
decisions of every axon are kept between sessions
(``tools.mps_lumen.LumenReviewStore``), so re-opening an axon's review
restores them together with its "results were shown" provenance. The widefield part of the rule reads the Axoplasm
panel's own masks and registration (``widefield_from_axoplasm_panel``),
exactly as the research reproduced the panel (D-32a); without them the
rule is isolation only. The re-run button never rebuilds the rings: it
runs ``tools.mps_lumen.run_cleaned_analyses`` (``clean_rings`` with the
final set, then the three tests) on a copy.

Colours and shapes
------------------
Every colour is a role of ``tools.mps_plot_style`` (Okabe-Ito), and the
shape carries the class as well, so the classes survive every dichromacy
and a greyscale print (``LUMEN_CLASS_STYLE``; ``validate_plot_colours.py``
checks the plot as "the lumen review plot"). The ring of a cluster is the
colour of its marker's outline, the Rings panel's segment colour.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt5 import QtCore, QtGui, QtWidgets

from tools import export_ui, mps_axoplasm
from tools.mps_axoplasm import AxoplasmMask
from tools.mps_columns import ColumnsParams, RingsParams, RingsResult, build_rings, rings_params_from
from tools.mps_identity import AxonIdentity, axon_id
from tools.mps_lumen import (
    AUTO_DOUBTFUL,
    AUTO_KEPT,
    AUTO_REMOVED,
    LEAK_NOT_CALIBRATED_NOTE,
    LUMEN_DECISION_COLUMNS,
    LUMEN_RULE_VERSION,
    NO_WIDEFIELD_WARNING,
    TUBULIN_SHIFT_WARNING,
    CleanedAnalyses,
    LumenClassification,
    LumenDecisions,
    LumenParams,
    LumenReviewStore,
    WidefieldLumenFlags,
    classify_lumen,
    cluster_lab_centres,
    default_columns_params,
    run_cleaned_analyses,
    widefield_flags_from_depths,
)
from tools.mps_layer_panel import LayerPanel, Swatch
from tools.mps_plot_style import (
    OKABE_ITO, ROLES, marked, neutral, rgba, segment_colour, set_title, style_dark, verdict)
from tools import mps_selection as msel
from tools import mps_tau as mtau
from tools import mps_tau_ui as mtui
from tools.mps_selection_ui import SelectionWidget, app_log_path, exploratory_banner, git_head
from tools.mps_tooltips import apply_tooltips
from tools.results_table import append_rows, cell_text, check_appendable, refuse_other_analysis, replace_rows

__all__ = [
    "COLUMNS_WINDOW_TOOLTIPS",
    "DECISION_TABLE_COLUMNS",
    "DISPLAY_AUTO_REMOVED",
    "DISPLAY_CLASSES",
    "DISPLAY_DOUBTFUL",
    "DISPLAY_KEPT",
    "DISPLAY_LABELS",
    "DISPLAY_MANUAL_REMOVED",
    "DISPLAY_RESTORED",
    "EDITED_LABEL_TEXT",
    "LEAK_BANNER_TEXT",
    "LUMEN_CLASS_STYLE",
    "RESULTS_TABLE_COLUMNS",
    "UNDERLAY_MAX_GREY",
    "ColumnsReview",
    "ColumnsReviewInputs",
    "ColumnsReviewLauncher",
    "ColumnsWindow",
    "ImagePlacement",
    "ReviewWidefield",
    "decision_rows",
    "display_class_of",
    "prepare_review",
    "results_row",
    "review_signature",
    "review_widefield_flags",
    "ring_colour_name",
    "start_columns_review",
    "widefield_from_axoplasm_panel",
]

# ============================================================================
# The classes on screen
# ============================================================================

# What a cluster is on screen: its automatic class and the user's action
# together. A doubtful cluster the user removed is shown as removed by hand;
# a cluster the rule removed and the user kept is "restored".
DISPLAY_KEPT = "kept"
DISPLAY_AUTO_REMOVED = "auto_removed"
DISPLAY_DOUBTFUL = "doubtful"
DISPLAY_MANUAL_REMOVED = "manual_removed"
DISPLAY_RESTORED = "restored"
DISPLAY_CLASSES: Tuple[str, ...] = (
    DISPLAY_KEPT, DISPLAY_AUTO_REMOVED, DISPLAY_DOUBTFUL, DISPLAY_MANUAL_REMOVED, DISPLAY_RESTORED)

# How each class is drawn: (role of tools.mps_plot_style.ROLES, alpha, pyqtgraph symbol). The roles keep their
# meaning from the other panels: a kept cluster is a centroid (green circle, as in the contour plot); one the rule
# removed as lumen is "discarded" (vermillion diamond, as the Axoplasm panel's discard); a doubtful one needs the
# user's attention, "warn" (orange triangle); one removed by hand is out of the analysis like a curated-away
# cluster (grey x); one the user gave back is in the data again, "locs" (sky blue square). Five symbols for five
# classes: orange against vermillion is 18 apart for a deuteranope, and a triangle against a diamond is what tells
# them apart there (validate_plot_colours.py, "the lumen review plot").
LUMEN_CLASS_STYLE: Dict[str, Tuple[str, int, str]] = {
    DISPLAY_KEPT: ("centroid", 255, "o"),
    DISPLAY_AUTO_REMOVED: ("discarded", 255, "d"),
    DISPLAY_DOUBTFUL: ("warn", 255, "t"),
    DISPLAY_MANUAL_REMOVED: ("curated", 255, "x"),
    DISPLAY_RESTORED: ("locs", 255, "s"),
}
DISPLAY_LABELS: Dict[str, str] = {
    DISPLAY_KEPT: "kept",
    DISPLAY_AUTO_REMOVED: "removed by the rule",
    DISPLAY_DOUBTFUL: "doubtful (kept unless removed)",
    DISPLAY_MANUAL_REMOVED: "removed by hand",
    DISPLAY_RESTORED: "restored by hand",
}
# The rows of the map's layer panel (UI stage 0). Short: the panel is at most 240 px wide and its rows never wrap; the
# tooltips carry the full meaning. Display text only: the keys are the DISPLAY_* constants and two ASCII names.
LAYER_LABELS: Dict[str, str] = {
    DISPLAY_KEPT: "Kept",
    DISPLAY_AUTO_REMOVED: "Removed by the rule",
    DISPLAY_DOUBTFUL: "Doubtful",
    DISPLAY_MANUAL_REMOVED: "Removed by hand",
    DISPLAY_RESTORED: "Restored by hand",
    "membrane_before": "Every cluster",
    "membrane_after": "After cleaning",
    "underlay": "Widefield image underneath",
}
# The order the markers are drawn in, bottom to top. Clusters of different rings often sit on top of each other (a
# column, or an axial leak copy in the next ring at the same x', y'), so what the user must see is drawn last: the
# user's own edits, then the rule's removals, then the doubtful. A click toggles the TOPMOST marker under the mouse
# (pyqtgraph hands the points topmost first), i.e. the one the user sees.
_DRAW_ORDER: Tuple[str, ...] = (
    DISPLAY_KEPT, DISPLAY_DOUBTFUL, DISPLAY_AUTO_REMOVED, DISPLAY_RESTORED, DISPLAY_MANUAL_REMOVED)
MARKER_SIZE_PX = 11.0
# The outline carries the ring (tools.mps_plot_style.segment_colour of Ring.index, the Rings panel's colours).
MARKER_PEN_WIDTH = 2.0
# The x of a cluster removed by hand is thin: at 11 px a 2 px outline covers its grey fill and it reads as the
# ring's colour (the GUI review of H5-D). It is drawn larger, with a thinner outline, so the grey shows.
MANUAL_REMOVED_SIZE_PX = 15.0
MANUAL_REMOVED_PEN_WIDTH = 1.2
HIGHLIGHT_SIZE_PX = 24.0
# The centroid membrane after the cleaning: "contour_kept", the contour without the discarded clusters (the
# Axoplasm panel's role); before the cleaning: the neutral structural line, dashed.
MEMBRANE_AFTER_ROLE = "contour_kept"
# A widefield image drawn under the markers is dimmed to this grey at its brightest, so every class still stands
# off it (validate_plot_colours.py measures the plot against this grey).
UNDERLAY_MAX_GREY = "#404040"
UNDERLAY_OPACITY = 0x40 / 255.0
# How far beyond the clusters the underlay is cropped, and the least margin the view keeps around them.
UNDERLAY_MARGIN_NM = 2000.0
VIEW_MARGIN_NM = 300.0

LEAK_BANNER_TEXT = marked(
    "warn", f"Column statistics {LEAK_NOT_CALIBRATED_NOTE}: z_A and p are read against a null WITHOUT the axial leak, "
            "which inflates them on real axons (D-28, D-30). The leak calibration (H5-E) was closed as not accepted "
            "(D-41): no p of this window is the axon's p-value.")
EDITED_LABEL_TEXT = marked(
    "warn", "Edited after results were shown: the cluster set was changed after column results of this axon were "
            "displayed (D-35b). The flag stays on this axon's decisions and on every export.")
STALE_LABEL_TEXT = marked(
    "dim", "The cluster set changed since the results below were computed: press 'Apply and re-run analyses' to "
           "update them.")
MANUAL_NULL_NOTE = (
    "manual edits are not reproduced by the simulated null: power_columns.py simnull --lumen-clean applies the "
    "automatic rule only (D-35b)")
# D-41 (Q-33): the H5-E calibration of the simulated null was closed as NOT accepted; every p on screen says so.
NOT_CALIBRATED = "not calibrated"
NOT_CALIBRATED_DETAIL = ("every p here is NOT calibrated: the H5-E calibration of the simulated null was closed as not "
                         "accepted (D-41, Q-33)")
NO_VIABLE_PAIR_TEXT = ("No ring pair of this axon passes the viability rule v2 (D-41): the column analysis was NOT run "
                       "and no statistic is shown. The reasons, pair by pair:")
# H6 toggles (D-43): the same under an exploratory selection (no pair selected: no column statistic is shown)
NO_SELECTED_PAIR_TEXT = ("No ring pair of this axon passes the exploratory selection {label} #{hash}: no column "
                         "statistic is shown. The reasons, pair by pair:")
LIVE_DEBOUNCE_MS = 300
# UI stage 1 (D-44): the analyses kept per (cluster set, tau), so that going back to a tau or a cluster set already
# analysed re-draws at once
ANALYSES_CACHE_SIZE = 8
# The matches on the map (T5): between the membranes (z -5, -4) and the markers (z 0 and up), the arc test on top.
MATCH_ARC_Z = -2.0
MATCH_2D_Z = -3.0
# The E(tau) curve's plot: never shorter than this (the section scrolls instead on a short screen).
CURVE_PLOT_MIN_HEIGHT = 150
# How the left column is shared between the map and the curve under it, until the user drags the handle: the curve
# up to 2/5 of the column (3:2), and the map -- the review's main surface, where clusters are clicked -- never less
# than this share of the window's height (on a short screen the curve section starts small and scrolls).
LEFT_CURVE_SHARE = 0.4
MAP_MIN_WINDOW_SHARE = 0.55


# ============================================================================
# Which ring pairs the column analysis runs on (D-41, rule v2)
# ============================================================================

@dataclass(frozen=True)
class ReviewViability:
    """
    The viability of every consecutive ring pair of the review's rings:
    ``rule`` names the rule and its version, ``pairs`` holds one
    ``batch_columns.PairVerdict`` (ring_a, ring_b, verdict, reasons) per
    pair, ``detail`` the full ``AxonViabilityV2`` when the rule is v2,
    ``lab_source`` where the lab coordinates of its 3D groups came from.
    """

    rule: str
    pairs: Tuple[Any, ...]
    detail: Optional[Any] = None
    lab_source: str = ""
    # H6 toggles (D-43): under an exploratory selection, its ``SelectionResult`` and ``SelectionSpec`` (None under the
    # pre-specified rule, so the window's viability then equals rule v2's field by field)
    selection: Any = None
    spec: Any = None

    def of(self, verdict_: str) -> List[Tuple[int, int]]:
        return [(int(p.ring_a), int(p.ring_b)) for p in self.pairs if str(p.verdict) == verdict_]

    @property
    def viable(self) -> List[Tuple[int, int]]:
        return self.of("viable")

    @property
    def marginal(self) -> List[Tuple[int, int]]:
        return self.of("marginal")


def rule_v2_viability(res: RingsResult, xyz_lab_nm: Optional[Sequence[Any]] = None) -> ReviewViability:
    """
    Rule v2 (D-41) on the review's rings as built (before the lumen
    cleaning, as the batch runner does): ``z_quality`` and ``viability_v2``
    with the 3D SiZer groups on the LAB coordinates ``xyz_lab_nm`` (rebuilt
    from the axon frame when not given). Geometry only.
    """
    import batch_columns as bc
    from tools.mps_axial_precision import viability_v2, z_quality
    from tools.mps_zquality_window import lab_xyz_of

    if xyz_lab_nm is None:
        xyz: Sequence[Any] = lab_xyz_of(res)
        src = "lab coordinates rebuilt from the axon frame"
    else:
        xyz = tuple(np.asarray(v, dtype=np.float64).reshape(-1) for v in xyz_lab_nm)
        src = "lab coordinates of the selection"
    v2 = viability_v2(res, z_quality(res), xyz_lab_nm=xyz)
    verdicts = bc.v2_viability(res, None, precomputed=v2)
    return ReviewViability(rule=str(verdicts.rule), pairs=tuple(verdicts.pairs), detail=v2, lab_source=src)


# What a new window uses (a module attribute, read when the window is made, so a test of the lumen review that is
# not about rule v2 can put another function here).
DEFAULT_VIABILITY_FN: Callable[[RingsResult, Optional[Sequence[Any]]], ReviewViability] = rule_v2_viability


@dataclass
class ReviewRun:
    """
    One press of "Apply and re-run analyses": the viability the run used,
    the analyses of the cleaned rings (None when no pair was viable or
    marginal: nothing was computed), and the arc test's joint null
    restricted to the VIABLE pairs (``primary``) and to the viable +
    marginal ones (``sensitivity``, only when there are marginal pairs):
    ``batch_columns.restricted_joint`` dictionaries, the same numbers the
    batch runner writes.
    """

    viability: ReviewViability
    analyses: Optional[CleanedAnalyses]
    primary: Optional[Dict[str, Any]] = None
    sensitivity: Optional[Dict[str, Any]] = None
    seconds: float = 0.0
    # H6 toggles: the lumen fingerprint of the cluster set analysed (``tools.mps_selection.lumen_fp``) and whether a
    # selection change started it ("Live update") rather than the button
    fp: Any = None
    live: bool = False
    # UI stage 1 (D-44): the tolerance the analyses ran at (the window's control); None = the pre-registered tau_0
    # (a run built outside the window)
    tau_nm: Optional[float] = None

    @property
    def ran(self) -> bool:
        return self.analyses is not None

# The export (tools.results_table conventions): what says two rows describe the same thing, and what makes a table
# another analysis (never appended to).
DECISION_TABLE_KEY: Tuple[str, ...] = ("axon_id", "stable_key")
DECISION_TABLE_ANALYSIS: Tuple[str, ...] = ("rule_version",)
RESULTS_TABLE_KEY: Tuple[str, ...] = ("axon_id",)
RESULTS_TABLE_ANALYSIS: Tuple[str, ...] = ("rule_version", "arcc_recipe", "tau0_nm", "viability_rule")
# The arc test's joint null restricted to the selected pairs: the fields of batch_columns.TIER_STAT_FIELDS (the batch
# writes the same numbers under the same names).
TIER_STAT_FIELDS: Tuple[str, ...] = ("pairs", "n_pairs", "T_obs", "null_mean", "z_A", "p_excess_uncalibrated",
                                     "p_two_sided_uncalibrated")
_IDENTITY_FIELDS: Tuple[str, ...] = tuple(AxonIdentity().columns().keys())
DECISION_TABLE_COLUMNS: Tuple[str, ...] = (
    ("axon_id",) + _IDENTITY_FIELDS + ("source", "roi", "exported_at", "program")
    + tuple(c for c in LUMEN_DECISION_COLUMNS if c not in ("axon_id",)))
RESULTS_TABLE_COLUMNS: Tuple[str, ...] = (
    ("axon_id",) + _IDENTITY_FIELDS + (
        "source", "roi", "exported_at", "program", "analysis", "rule_version", "has_widefield", "interior_usable",
        "registration", "n_rings", "n_clusters", "k_per_ring_before", "k_per_ring_after", "lumen_n_removed_auto",
        "lumen_n_doubtful", "lumen_n_removed_final", "lumen_n_manual", "lumen_removed_keys",
        "lumen_automatic_removed_keys", "edited_after_results_shown", "results_match_decisions",
        "simnull_reproduces_set", "leak_calibrated", "leak_note", "n_null", "tau0_nm", "run_seconds", "arcc_recipe",
        "arcc_z_A", "arcc_p_A", "arcc_T_A", "arcc_membrane_length_nm", "arcc_knot_spacing_nm", "arcc_zeta_pair0",
        "arcc_p_excess_pair0", "arcc_zeta_pair1", "arcc_p_excess_pair1", "arcc_pairs", "cols2d_z_A", "cols2d_p_A",
        "cols2d_zeta_pair0", "cols2d_p_excess_pair0", "cols2d_zeta_pair1", "cols2d_p_excess_pair1", "arcl_z_A",
        "arcl_p_A", "arcl_role", "run_warnings", "viability_rule", "pair_verdicts", "viable_pairs", "marginal_pairs",
        "calibration_note")
    + tuple(f"primary_{f}" for f in TIER_STAT_FIELDS) + tuple(f"sensitivity_{f}" for f in TIER_STAT_FIELDS))
_ANALYSIS_NAME = "H-ECL columns after the lumen cleaning (H5-D)"
_PROGRAM = "MPS Explorer (tools/mps_columns_window.py)"

SIMNULL_BUTTON_TIP = (
    "Optional: a p of this axon against a SIMULATED null (power_columns.py simnull, in a separate process with "
    "progress and Cancel), restricted to the pairs rule v2 selects in the observed axon and in every simulated one.\n\n"
    "It is NOT calibrated: the H5-E calibration of the simulated null was closed as not accepted (D-41, Q-33). It "
    "takes minutes (the measurement of the axon, then each simulated axon).")
SIMNULL_DISABLED_TIP = (
    "Disabled: no ring pair of this axon is VIABLE under rule v2 (D-41), so there is nothing to compare with a "
    "simulated null. 'Z quality...' shows why.")

# What each control says when the mouse rests on it, in the style of tools/mps_tooltips.py: what it does, then the
# thing a beginner cannot guess. Keyed by the window's attribute names (apply_tooltips).
COLUMNS_WINDOW_TOOLTIPS: Dict[str, str] = {
    "plot_widget":
        "The clusters of every ring of this axon in the axon's own frame (x', y': the tilt of the axis removed).\n\n"
        "Fill and shape say what happens to a cluster: green circle kept, vermillion diamond removed by the lumen "
        "rule, orange triangle doubtful (kept unless you remove it), grey x removed by hand, sky-blue square "
        "restored by hand. The outline's colour is the ring, as in the Rings panel.\n\n"
        "Click a marker to remove that cluster by hand; click it again to keep it. Any cluster can be clicked, the "
        "ones the rule removed included (that restores them). Hover a marker for its descriptors.\n\n"
        "Clusters of different rings can sit on top of each other (a column, or an axial leak copy): the click acts "
        "on the marker drawn on top, and hovering lists every cluster under the mouse. To reach one underneath, "
        "show its ring alone (below the plot), hide the class on top in the panel at the right, or use the doubtful "
        "table.",
    "layer_panel":
        "Tick what the map draws. A hidden class cannot be clicked; the analysis is not affected.",
    "doubtful_table":
        "The doubtful clusters: flagged by a widefield image (WF0, WF250) or by the sensitive isolation criterion "
        "(ISO_SENS), but not removed by the rule (D-35a).\n\n"
        "Tick a row to remove that cluster; untick it to keep it. Unticked is the automatic state: a doubtful "
        "cluster stays in the analysis unless you remove it. Select a row to ring its marker in the plot.\n\n"
        "Hull depth is the depth inside the convex hull of every ring's centroids; a hop is the longest step from "
        "the cluster to that hull's outline through the other centroids (short in a crowd of membrane clusters, "
        "long for a cluster alone in the lumen). The depths are + inside the tubulin mask and inside the spectrin "
        "ring's dark interior; n/a = off the images, or no images.",
    "select_all_doubtful_button":
        "Remove every doubtful cluster (tick every row).\n\n"
        "Each change is logged in order. Nothing is analysed until you press 'Apply and re-run analyses'.",
    "clear_all_doubtful_button":
        "Keep every doubtful cluster again (untick every row): their automatic state.\n\n"
        "Clusters removed or restored by clicking the plot are not touched.",
    "reset_button":
        "Put every cluster back in its automatic state: the rule's removals removed, everything else kept.\n\n"
        "The undoing is logged like any other edit, and if results were already shown the 'edited after results "
        "were shown' flag stays: going back does not un-see a result.",
    "run_button":
        "Remove the final set of clusters from the rings and run the ring analyses again: the arc test on the "
        "centroid membrane (primary), the 2D test (sensitivity) and the arc test on the localization membrane "
        "(a diagnostic).\n\n"
        "Only the ring pairs that pass the viability rule v2 (D-41) are analysed: the VIABLE pairs are the result, "
        "the MARGINAL ones a sensitivity check, the NOT VIABLE ones are never shown. When no pair is viable or "
        "marginal nothing is computed and the window says why. Every p is NOT calibrated (H5-E not accepted).\n\n"
        "The rings are NOT rebuilt: the clusters are taken out of the rings built when this window opened. It runs "
        "in the background; the window stays usable, and edits made meanwhile wait for the next run.",
    "export_button":
        "Write two tables: one row per cluster with its automatic class, your action, the order of every edit and "
        "the provenance flags, and one row with the results of the last run.\n\n"
        "Rows are added to an existing table of the same kind; a table of another analysis is never written into, "
        "and a row this axon already has is replaced only if you say so.",
    "load_button":
        "Read decisions exported earlier for this axon and apply them to these clusters, matched by their stable "
        "key (ring, DBSCAN label and rounded centroid).\n\n"
        "A table of another axon, or of this axon's rings built otherwise, is refused and nothing changes. Rows of "
        "other axons in the same table are ignored; when the table holds several exports of this axon, the last one "
        "is used. If you have already edited or seen results here, every change the file makes is logged as an edit "
        "(and flagged if results were shown).",
    "ring_filter":
        "Show every ring, or one ring alone. Only the drawing changes: every cluster stays in the analysis as "
        "decided.",
    "underlay_check":
        "Draw a widefield image of the Axoplasm panel under the clusters, dimmed so the markers stay legible.\n\n"
        "It is placed with the panel's registration and projected at the axon's mean depth, so it is a guide for "
        "the eye, a few tens of nm off at most with the tilts seen so far; the classification itself reads the "
        "masks exactly as the panel does.\n\n"
        "Greyed out when this axon has no widefield image.",
    "underlay_combo":
        "Which widefield image to draw underneath: the betaIII-tubulin one (the mask of the axon) or the "
        "betaII-spectrin one (the ring and its dark interior).",
    "progress_bar":
        "What the re-run is doing. The window can be used meanwhile.",
    "results_view":
        "The analyses of the last run, on the cluster set shown in the counts above them. None of the statistics "
        "is calibrated for the axial leak yet (H5-E).",
    "warnings_list":
        "What the rule, the images and the analyses had to say about this axon.",
    "viability_table":
        "The viability of every consecutive ring pair under rule v2 (D-41), computed on the rings as built (before "
        "the lumen cleaning): (1) a significant SiZer peak of each ring, (2) a significant valley between them, "
        "(3) each ring with at least f_min(x) of the central ring's localizations, (4) expected axial-leak copies "
        "<= 2 % (VIABLE) or <= 5 % (MARGINAL).\n\n"
        "The column analysis runs on the VIABLE pairs only; MARGINAL pairs are a sensitivity check; NOT VIABLE pairs "
        "are never analysed. 'Z quality...' shows the profile behind these verdicts.",
    "zquality_button":
        "Open the z-quality view of these rings: the axial profile with the SiZer significance strip, the ring and "
        "pair tables behind the verdicts, and what limits the z quality of this axon (for the microscopy team).",
    "simnull_button":
        SIMNULL_BUTTON_TIP,
    "advanced_toggle":
        "Show or hide the experimental tools: the simulated-null p, whose calibration was not accepted (D-41). "
        "Read its dialog before using it.",
    "columns_batch_button":
        "Run the column test on many axons in a separate process (batch_columns.py): choose files or a folder and an "
        "output folder; live progress, Cancel, and Resume with the same output folder. Rule v2 selects the pairs; "
        "every p is NOT calibrated.",
    "live_update_check":
        "Live update (H6 toggles): when a change of the criteria switches needs analyses this cluster set does not "
        "have yet (the last run computed nothing because no pair was selected), run them in the background 0.3 s "
        "after the last change. Off: the window says 'press Apply and re-run'.\n\n"
        "The analyses do not depend on the selection, so most changes need no run at all: the results are re-drawn "
        "at once. A run nobody needs any more (the selection went back to no pair) is stopped.",
    "selection_counter_label":
        "Under an EXPLORATORY selection (not the pre-specified rule v2, D-41) its label and short hash first; they are "
        "in every export of this window.\n\n"
        "Then how many different selections the p values of this axon were shown under in this session (every one is "
        "logged in selection_exploration_log.csv). The p values are NOT corrected for trying several selections.",
    "selection_note":
        "What the window does after a change of the criteria switches.",
}


def ring_colour_name(ring_index: int) -> str:
    """The name of a ring's outline colour (the Okabe-Ito name of ``segment_colour``), for text: the plot tells rings
    apart by the outline's colour, and the class shapes (circle, square, triangle, diamond) must not be read as rings."""
    colour = segment_colour(int(ring_index)).lower()
    for name, value in OKABE_ITO.items():
        if value.lower() == colour:
            return name.replace("_", " ")
    return colour


def display_class_of(auto_class: str, removed: bool) -> str:
    """What a cluster is on screen, from its automatic class and its final state."""
    if auto_class == AUTO_REMOVED:
        return DISPLAY_AUTO_REMOVED if removed else DISPLAY_RESTORED
    if removed:
        return DISPLAY_MANUAL_REMOVED
    return DISPLAY_DOUBTFUL if auto_class == AUTO_DOUBTFUL else DISPLAY_KEPT


# ============================================================================
# The widefield part, from the Axoplasm panel
# ============================================================================

@dataclass
class ImagePlacement:
    """
    How the laboratory frame (nm) maps onto one image, as the Axoplasm panel
    places it (``_coordinates_for``): a widefield image at column = x /
    pixel + camera offset + shift; an image drawn from localizations
    acquired with the movie at column = x / pixel (placed by construction).
    Pixel c spans [c - 0.5, c + 0.5).
    """

    from_localizations: bool
    offset_px: Tuple[float, float] = (0.0, 0.0)
    shift_px: Tuple[float, float] = (0.0, 0.0)

    def to_pixels(self, x_nm: Any, y_nm: Any, pixel_size_nm: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(column, row) of laboratory positions."""
        x = np.asarray(x_nm, dtype=np.float64) / float(pixel_size_nm)
        y = np.asarray(y_nm, dtype=np.float64) / float(pixel_size_nm)
        if self.from_localizations:
            return x, y
        return (x + float(self.offset_px[0]) + float(self.shift_px[0]),
                y + float(self.offset_px[1]) + float(self.shift_px[1]))

    def lab_origin_nm(self, pixel_size_nm: float) -> Tuple[float, float]:
        """The laboratory position of the image's corner (column, row) = (-0.5, -0.5)."""
        px = float(pixel_size_nm)
        if self.from_localizations:
            return (-0.5 * px, -0.5 * px)
        return ((-0.5 - float(self.offset_px[0]) - float(self.shift_px[0])) * px,
                (-0.5 - float(self.offset_px[1]) - float(self.shift_px[1])) * px)


@dataclass
class ReviewWidefield:
    """
    What the review needs from the Axoplasm panel, read in the GUI thread
    (``widefield_from_axoplasm_panel``) so the worker never touches the
    panel: its tubulin mask (Otsu or the user's threshold, as the panel
    shows it), its spectrin ring interior (None when the panel has not
    built one: then it is built here from the ring clusters, with a
    warning), both images (for the interior and for the drawing), how each
    is placed, the registration in words, the panel's selection (whose
    median centre finds the axon in each image), and notes.
    """

    pixel_size_nm: float
    tubulin_mask: AxoplasmMask
    spectrin_interior: Optional[AxoplasmMask]
    tubulin_image: Optional[NDArray[np.float64]]
    spectrin_image: Optional[NDArray[np.float64]]
    tubulin_placement: ImagePlacement
    spectrin_placement: ImagePlacement
    registration: str = ""
    selection_x_nm: Optional[NDArray[np.float64]] = None
    selection_y_nm: Optional[NDArray[np.float64]] = None
    notes: List[str] = field(default_factory=list)


def _registration_words(reg: Any) -> str:
    """The panel's registration in one phrase, with the shift in pixels and its score."""
    words = {"none": "no shift", "manual": "set by hand", "measured": "measured",
             "adjusted": "measured, then adjusted by hand",
             "same acquisition": "placed by construction (images drawn from localizations of the movie)"}
    source = str(getattr(reg, "source", "none"))
    shift = tuple(float(v) for v in getattr(reg, "shift_px", (0.0, 0.0)))
    text = f"Axoplasm panel: {words.get(source, source)}, shift ({shift[0]:+.2f}, {shift[1]:+.2f}) px"
    score = getattr(reg, "score", None)
    if score is not None:
        text += f", score {float(score):.1f}"
    return text


def widefield_from_axoplasm_panel(panel: Any) -> Tuple[Optional[ReviewWidefield], str]:
    """
    The widefield state of the Axoplasm panel ``panel``
    (``tools.mps_axoplasm_window.AxoplasmWindow``) as the lumen rule needs
    it, or (None, why not). Called in the GUI thread: anything a drag left
    pending is recomputed first (``panel.flush``). Both images and a
    tubulin mask are required (the joint rule reads both); a widefield
    image that has not been placed on the localizations yet (no shift
    measured or typed) is refused, as the panel itself refuses to sort the
    clusters then.
    """
    if panel is None:
        return None, "the Axoplasm panel is not open"
    try:
        panel.flush()
    except Exception as exc:  # noqa: BLE001 - the panel's own state; reported, never fatal here
        return None, f"the Axoplasm panel could not be read ({type(exc).__name__}: {exc})"
    tubulin = getattr(panel, "tubulin", None)
    reference = getattr(panel, "reference", None)
    mask = getattr(panel, "axoplasm", None)
    if tubulin is None:
        return None, "the Axoplasm panel has no betaIII-tubulin image"
    if reference is None:
        return None, "the Axoplasm panel has no betaII-spectrin widefield image (the joint rule needs both images)"
    if mask is None:
        region = "; ".join(str(n) for n in (getattr(panel, "region_notes", None) or []))
        return None, "the Axoplasm panel has no tubulin mask" + (f" ({region})" if region else "")
    needs_shift = any(img is not None and not bool(getattr(img, "from_localizations", False))
                      for img in (tubulin, reference))
    reg = panel.registration()
    if needs_shift and str(reg.source) == "none":
        return None, ("the widefield images are not placed on the localizations yet: measure the shift in the "
                      "Axoplasm panel (section 2) or set one by hand")
    shift = (float(reg.shift_px[0]), float(reg.shift_px[1]))
    t_off = (float(panel.tubulin_offset[0]), float(panel.tubulin_offset[1]))
    s_off = (float(panel.reference_offset[0]), float(panel.reference_offset[1]))
    tub_place = ImagePlacement(bool(getattr(tubulin, "from_localizations", False)), t_off, shift)
    spec_place = ImagePlacement(bool(getattr(reference, "from_localizations", False)), s_off, shift)
    loc = getattr(getattr(panel, "inputs", None), "loc", None)
    notes: List[str] = []
    if mask.threshold_source != "otsu":
        notes.append(f"The tubulin mask uses the threshold set by hand in the Axoplasm panel ({mask.threshold:g}), "
                     "not Otsu's: the classification follows the panel as it is.")
    return ReviewWidefield(
        pixel_size_nm=float(panel.pixel_nm), tubulin_mask=mask, spectrin_interior=getattr(panel, "spectrin_interior", None),
        tubulin_image=np.asarray(tubulin.image, dtype=np.float64), spectrin_image=np.asarray(reference.image, dtype=np.float64),
        tubulin_placement=tub_place, spectrin_placement=spec_place, registration=_registration_words(reg),
        selection_x_nm=None if loc is None else np.asarray(loc.x_nm, dtype=np.float64).copy(),
        selection_y_nm=None if loc is None else np.asarray(loc.y_nm, dtype=np.float64).copy(), notes=notes), ""


def review_widefield_flags(
    res: RingsResult,
    x_lab_nm: Any,
    y_lab_nm: Any,
    widefield: ReviewWidefield,
    params: LumenParams = LumenParams(),
) -> WidefieldLumenFlags:
    """
    The widefield flags of every ring cluster read on the Axoplasm panel's
    own masks (the research's recipe, a private research script: each
    cluster's lab centre = the mean laboratory x, y of its localizations;
    its depth in the tubulin mask and in the spectrin interior at that
    centre, each image placed as the panel places it). When the panel has
    no spectrin interior (the 2D analysis gives its ring level, and there
    was none), it is built here with the panel's own function, the ring
    level read at the ring clusters' lab centres, and a warning says so.
    ``x_lab_nm``, ``y_lab_nm``: every localization ``build_rings`` received.
    """
    px = float(widefield.pixel_size_nm)
    C = cluster_lab_centres(res, x_lab_nm, y_lab_nm)
    warnings_: List[str] = list(widefield.notes)
    if not widefield.tubulin_placement.from_localizations:
        warnings_.append(TUBULIN_SHIFT_WARNING)
    tcol, trow = widefield.tubulin_placement.to_pixels(C[:, 0], C[:, 1], px)
    dt = np.asarray(widefield.tubulin_mask.distance_at(tcol, trow), dtype=np.float64)
    interior = widefield.spectrin_interior
    if interior is None:
        if widefield.spectrin_image is None or widefield.selection_x_nm is None or widefield.selection_y_nm is None:
            raise ValueError("no spectrin interior in the Axoplasm panel and no image to build one from")
        scol_sel, srow_sel = widefield.spectrin_placement.to_pixels(widefield.selection_x_nm, widefield.selection_y_nm, px)
        centre, radius, reach = mps_axoplasm.axon_centre(scol_sel, srow_sel)
        ring_col, ring_row = widefield.spectrin_placement.to_pixels(C[:, 0], C[:, 1], px)
        interior = mps_axoplasm.build_ring_interior(widefield.spectrin_image, centre, radius, px, ring_col=ring_col,
                                                    ring_row=ring_row, reach_px=reach)
        warnings_.append("The Axoplasm panel had no spectrin interior (it reads the ring level at the kept cluster "
                         "centres of the 2D analysis, which was not available): it was built here with the ring "
                         "clusters' own lab centres.")
    scol, srow = widefield.spectrin_placement.to_pixels(C[:, 0], C[:, 1], px)
    ds = np.asarray(interior.distance_at(scol, srow), dtype=np.float64)
    usable = str(interior.threshold_source) == "half maximum"
    flags = widefield_flags_from_depths(res, dt, ds, interior_usable=usable, params=params, centroids_lab_nm=C,
                                        registration=widefield.registration,
                                        interior_source=str(interior.threshold_source))
    if not usable:
        warnings_.append(f"The spectrin interior is cut at its {interior.threshold_source}, not at its half maximum: "
                         "it is not usable, so the WF250 removal term and the veto are off for this axon (D-35a).")
    warnings_.extend(f"tubulin mask: {w}" for w in widefield.tubulin_mask.warnings)
    warnings_.extend(f"spectrin interior: {w}" for w in interior.warnings)
    flags.warnings = warnings_ + list(flags.warnings)
    return flags


# ============================================================================
# Building the review (the worker's job when the window opens; no Qt)
# ============================================================================

@dataclass
class ColumnsReviewInputs:
    """
    One axon as the main window hands it over: every localization of the
    ROI BEFORE the axial cut (laboratory nm, the H1 hook
    ``roi_indices_unfiltered``) with their frames and precisions, the
    acquisition's length, the file, the ROI (``roi``: None -- the review's
    convention, ``RINGS_WITHOUT_ROI_NOTE`` -- builds the rings without its
    edge; ``roi_text`` names the axon in the tables), the pixel size and
    where it came from, the rings' parameters (None: the pre-registered ones), the
    Axoplasm panel's widefield state (None: isolation only, with
    ``widefield_note`` saying why), and the identity (an ``AxonIdentity``
    or a callable returning one, asked at the export).
    """

    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    frame: Optional[NDArray[np.int64]] = None
    lp_lateral_nm: Optional[NDArray[np.float64]] = None
    lpz_nm: Optional[NDArray[np.float64]] = None
    n_frames: Optional[int] = None
    source_name: str = ""
    roi: Optional[Any] = None
    roi_text: str = ""
    pixel_size_nm: Optional[float] = None
    pixel_size_source: str = "unknown"
    rings_params: Optional[RingsParams] = None
    widefield: Optional[ReviewWidefield] = None
    widefield_note: str = ""
    identity: Any = None


@dataclass
class ColumnsReview:
    """What ``prepare_review`` built: the rings, their classification, the
    axial precisions (for ``clean_rings``), the parameters used, the
    centroid membrane through every cluster (None if it could not be
    fitted), the warnings and the time."""

    res: RingsResult
    classification: LumenClassification
    lpz_nm: Optional[NDArray[np.float64]]
    rings_params: RingsParams
    membrane_before: Optional[Any]
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0
    rings_note: str = ""


# How the review builds the rings (the orchestrator's decision for the H5-D fix, 2026-09-29): exactly as simnull,
# the research code and the certified classification of D-35(d)(4) do, WITHOUT the drawn ROI's edge (build_rings
# roi=None: the edge criterion reads the convex hull of the selection), so that the window's clusters, classes and
# stable keys are the ones simnull --lumen-edits finds, and a tight hand-drawn ROI cannot remove membrane clusters.
RINGS_WITHOUT_ROI_NOTE = (
    "Rings built without the ROI's edge (the edge criterion reads the convex hull of the selection), exactly as simnull "
    "and the certified classification build them, so decisions exported here apply unchanged to simnull --lumen-edits.")
RINGS_WITH_ROI_NOTE = (
    "Rings built WITH the drawn ROI's edge: simnull builds them without it, so its clusters and stable keys can differ "
    "from these.")


def review_signature(inputs: ColumnsReviewInputs) -> Tuple[Any, ...]:
    """
    What makes two reviews the same review: the axon (source and ROI text),
    its localizations (a hash of x, y, z), the ROI-edge convention and the
    widefield state the rule reads (registration, the tubulin threshold, the
    spectrin interior's cut, the images' sizes; or why there is none). The
    main window brings an open review forward instead of rebuilding it when
    the signature is unchanged, and rebuilds it (restoring the axon's
    decisions from the review store) when the selection or the Axoplasm
    panel changed.
    """
    h = hashlib.sha256()
    for arr in (inputs.x_nm, inputs.y_nm, inputs.z_nm):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64).reshape(-1))
        h.update(str(a.size).encode("ascii"))
        h.update(a.tobytes())
    wf = inputs.widefield
    if wf is None:
        wf_sig: Tuple[Any, ...] = ("no widefield", str(inputs.widefield_note))
    else:
        interior = wf.spectrin_interior
        wf_sig = (
            str(wf.registration), float(wf.pixel_size_nm), float(getattr(wf.tubulin_mask, "threshold", float("nan"))),
            str(getattr(wf.tubulin_mask, "threshold_source", "")),
            None if interior is None else (str(interior.threshold_source), float(getattr(interior, "threshold", float("nan")))),
            None if wf.tubulin_image is None else tuple(np.shape(wf.tubulin_image)),
            None if wf.spectrin_image is None else tuple(np.shape(wf.spectrin_image)),
            tuple(wf.tubulin_placement.offset_px) + tuple(wf.tubulin_placement.shift_px),
            tuple(wf.spectrin_placement.offset_px) + tuple(wf.spectrin_placement.shift_px))
    return (str(inputs.source_name), str(inputs.roi_text), inputs.roi is None, h.hexdigest(), wf_sig)


def _optional_array(values: Optional[Any], n: int, what: str, dtype: Any) -> Optional[NDArray[Any]]:
    if values is None:
        return None
    out = np.asarray(values, dtype=dtype).reshape(-1)
    if out.size != n:
        raise ValueError(f"{what}: {out.size} values for {n} localizations")
    return out


def build_review_rings(
        inputs: ColumnsReviewInputs) -> Tuple[RingsResult, RingsParams, Optional[NDArray[np.float64]], List[str]]:
    """
    The H-ECL rings of one axon exactly as the review builds them
    (``build_rings`` with the pre-registered parameters unless
    ``inputs.rings_params`` says otherwise, ``roi=inputs.roi``): the rings,
    the parameters used, the axial precisions (None without them) and the
    warnings. Shared by ``prepare_review`` and the z-quality view
    (tools/mps_zquality_window.py), so both see the same rings. No Qt.
    ValueError when the inputs disagree in length or no ring has clusters.
    """
    x = np.asarray(inputs.x_nm, dtype=np.float64).reshape(-1)
    n = int(x.size)
    y = np.asarray(inputs.y_nm, dtype=np.float64).reshape(-1)
    z = np.asarray(inputs.z_nm, dtype=np.float64).reshape(-1)
    if y.size != n or z.size != n:
        raise ValueError(f"the selection has {n} x, {y.size} y and {z.size} z values")
    frame = _optional_array(inputs.frame, n, "frame", np.int64)
    lp = _optional_array(inputs.lp_lateral_nm, n, "lateral precision", np.float64)
    lpz = _optional_array(inputs.lpz_nm, n, "axial precision", np.float64)
    warnings_: List[str] = []
    if inputs.rings_params is not None:
        params = inputs.rings_params
    else:
        try:
            params = rings_params_from(default_columns_params())
        except (OSError, ValueError) as exc:
            params = RingsParams()
            warnings_.append(f"the pre-registered column parameters could not be read ({exc}); the rings were built "
                             "with the defaults of RingsParams")
    res = build_rings(x.copy(), y.copy(), z.copy(), frame=None if frame is None else frame.copy(),
                      lp_lateral_nm=None if lp is None else lp.copy(), lpz_nm=None if lpz is None else lpz.copy(),
                      params=params, source_name=str(inputs.source_name), pixel_size_nm=inputs.pixel_size_nm,
                      pixel_size_source=str(inputs.pixel_size_source), n_frames=inputs.n_frames, roi=inputs.roi)
    n_clusters = sum(len(r.clusters) for r in res.rings)
    if not res.rings or n_clusters == 0:
        raise ValueError("no ring with clusters in this selection: " + "; ".join(res.warnings[:3]))
    return res, params, lpz, warnings_


def prepare_review(inputs: ColumnsReviewInputs,
                   progress: Optional[Callable[[str, float], None]] = None) -> ColumnsReview:
    """
    The rings of the axon (``build_rings``, once), the lumen classification
    of their clusters (isolation, and the widefield part when the Axoplasm
    panel gave its masks) and the centroid membrane through every cluster
    (what the window draws as "before the cleaning"). No Qt: meant for a
    worker thread. ``progress(text, fraction)`` is called before each step.
    ValueError when the inputs disagree in length or no ring has clusters.
    """
    from tools.mps_unroll import centroid_membrane_of_rings

    t0 = time.perf_counter()

    def say(text: str, fraction: float) -> None:
        if progress is not None:
            progress(str(text), float(fraction))

    x = np.asarray(inputs.x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(inputs.y_nm, dtype=np.float64).reshape(-1)
    say(f"Building the rings of {x.size:,} localizations (build_rings)", 0.05)
    res, params, lpz, warnings_ = build_review_rings(inputs)
    say("Classifying the ring clusters (isolation" + (" and widefield)" if inputs.widefield is not None else ")"), 0.75)
    wf_flags: Optional[WidefieldLumenFlags] = None
    if inputs.widefield is not None:
        try:
            wf_flags = review_widefield_flags(res, x, y, inputs.widefield)
        except Exception as exc:  # noqa: BLE001 - the rule falls back to isolation only, with the reason
            warnings_.append(f"the widefield part of the rule could not be read from the Axoplasm panel "
                             f"({type(exc).__name__}: {exc}): isolation only")
    classification = classify_lumen(res, widefield=wf_flags)
    say("Fitting the centroid membrane through every cluster", 0.9)
    membrane = None
    try:
        membrane = centroid_membrane_of_rings(res)
    except Exception as exc:  # noqa: BLE001 - a drawing aid; the window says it is missing
        warnings_.append(f"the centroid membrane through every cluster could not be fitted: {exc}")
    say("Done", 1.0)
    return ColumnsReview(res=res, classification=classification, lpz_nm=lpz, rings_params=params,
                         membrane_before=membrane, warnings=warnings_, seconds=float(time.perf_counter() - t0),
                         rings_note=RINGS_WITHOUT_ROI_NOTE if inputs.roi is None else RINGS_WITH_ROI_NOTE)


# ============================================================================
# The tables (no Qt)
# ============================================================================

def _num(value: Any) -> Optional[float]:
    """A finite float, or None (an empty cell) for NaN, inf and missing values."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _k_text(res: RingsResult) -> str:
    return "/".join(str(len(r.clusters)) for r in sorted(res.rings, key=lambda q: int(q.index)))


def decision_rows(decisions: LumenDecisions, head: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per cluster (``LumenDecisions.to_rows``) after the axon's own
    columns ``head`` (axon_id, identity, source, roi, exported_at,
    program), in ``DECISION_TABLE_COLUMNS`` order."""
    rows = []
    for row in decisions.to_rows():
        full = {c: head.get(c) for c in DECISION_TABLE_COLUMNS if c in head}
        full.update(row)
        rows.append({c: full.get(c) for c in DECISION_TABLE_COLUMNS})
    return rows


def _pair_values(pairs: Sequence[Any], i: int) -> Tuple[Optional[float], Optional[float]]:
    if i >= len(pairs):
        return None, None
    return _num(getattr(pairs[i], "zeta", None)), _num(getattr(pairs[i], "p_excess", None))


def results_row(
    result: CleanedAnalyses,
    decisions: LumenDecisions,
    head: Mapping[str, Any],
    *,
    res_before: Optional[RingsResult] = None,
    tau0_nm: Optional[float] = None,
    run: Optional[ReviewRun] = None,
    variant: Optional[mtau.AnalysisVariant] = None,
) -> Dict[str, Any]:
    """
    The results of one run as one row (``RESULTS_TABLE_COLUMNS``): the
    axon's own columns ``head``, the rule and the cleaning counts of the
    run, the provenance flags -- ``edited_after_results_shown`` is True when
    the run's set was edited after results were shown OR the decisions have
    been since (the export describes both), ``results_match_decisions``
    whether the decisions still give the run's set,
    ``simnull_reproduces_set`` False once the set has manual edits -- the
    three tests (arcc_ = centroid membrane, primary; cols2d_ = the 2D
    sensitivity; arcl_ = the localization-membrane diagnostic) and the
    warnings of the run. With ``run`` (rule v2, D-41) only the selected
    pairs keep their per-pair statistics, and a test's joint over all of
    its pairs is written only when all of them are VIABLE.

    ``variant`` (UI stage 1, D-44: a ``tools.mps_tau.AnalysisVariant``,
    the run's selection and tolerance): at another tau than tau_0,
    ``tau0_nm`` holds the run's tau (an analysis column: rows of two
    tolerances never share a table) and ``analysis`` carries the
    EXPLORATORY tag (``tools.mps_tau.analysis_with_tau``). At tau_0, or
    without it, the row is the pre-registered one.
    """
    from tools.mps_unroll import CENTROID_MEMBRANE_RECIPE

    arc_c, arc_l, cols = result.arc_centroid, result.arc_localization, result.columns_2d
    same = set(result.removed_keys) == set(decisions.final_removed_keys())
    row: Dict[str, Any] = {c: head.get(c) for c in RESULTS_TABLE_COLUMNS if c in head}
    cm = None if arc_c is None else getattr(arc_c, "centroid_membrane", None)
    arc_pairs = [] if arc_c is None else list(arc_c.adjacent)
    col_pairs = [] if cols is None else list(cols.adjacent)
    z0, p0 = _pair_values(arc_pairs, 0)
    z1, p1 = _pair_values(arc_pairs, 1)
    c0, q0 = _pair_values(col_pairs, 0)
    c1, q1 = _pair_values(col_pairs, 1)
    row.update({
        "analysis": _ANALYSIS_NAME,
        "rule_version": result.rule_version,
        "has_widefield": bool(result.has_widefield),
        "interior_usable": None if decisions.classification is None else bool(decisions.classification.interior_usable),
        "registration": None if decisions.classification is None else cell_text(decisions.classification.registration),
        "n_rings": len(result.rings.rings),
        "n_clusters": len(decisions.stable_keys),
        "k_per_ring_before": None if res_before is None else _k_text(res_before),
        "k_per_ring_after": _k_text(result.rings),
        "lumen_n_removed_auto": int(result.n_removed_auto),
        "lumen_n_doubtful": int(result.n_doubtful),
        "lumen_n_removed_final": int(result.n_removed_final),
        "lumen_n_manual": int(result.n_manual),
        "lumen_removed_keys": " ".join(result.removed_keys),
        "lumen_automatic_removed_keys": " ".join(result.automatic_removed_keys),
        "edited_after_results_shown": bool(result.edited_after_results_shown or decisions.results_shown_before_edit),
        "results_match_decisions": bool(same),
        "simnull_reproduces_set": bool(int(result.n_manual) == 0),
        "leak_calibrated": bool(result.leak_calibrated),
        "leak_note": result.leak_note,
        "n_null": int(result.n_null),
        "tau0_nm": _num(tau0_nm),
        "run_seconds": _num(round(float(result.seconds), 2)),
        "arcc_recipe": CENTROID_MEMBRANE_RECIPE,
        "arcc_z_A": None if arc_c is None else _num(arc_c.z_A),
        "arcc_p_A": None if arc_c is None else _num(arc_c.p_A),
        "arcc_T_A": None if arc_c is None else _num(arc_c.T_A),
        "arcc_membrane_length_nm": None if arc_c is None else _num(arc_c.length_nm),
        "arcc_knot_spacing_nm": None if arc_c is None else _num(arc_c.membrane_knot_spacing_nm),
        "arcc_zeta_pair0": z0, "arcc_p_excess_pair0": p0, "arcc_zeta_pair1": z1, "arcc_p_excess_pair1": p1,
        "arcc_pairs": " | ".join(f"{p.ring_a}-{p.ring_b}: zeta {float(p.zeta):+.3f}, p_excess {float(p.p_excess):.4f}"
                                 for p in arc_pairs),
        "cols2d_z_A": None if cols is None else _num(cols.z_A),
        "cols2d_p_A": None if cols is None else _num(cols.p_A),
        "cols2d_zeta_pair0": c0, "cols2d_p_excess_pair0": q0, "cols2d_zeta_pair1": c1, "cols2d_p_excess_pair1": q1,
        "arcl_z_A": None if arc_l is None else _num(arc_l.z_A),
        "arcl_p_A": None if arc_l is None else _num(arc_l.p_A),
        "arcl_role": cell_text(result.arc_localization_note),
        "run_warnings": " | ".join(cell_text(w) for w in result.warnings),
    })
    if cm is None and arc_c is not None:
        row["arcc_recipe"] = f"{CENTROID_MEMBRANE_RECIPE} (no fit recorded)"
    if variant is not None and not variant.tau_is_default:
        # UI stage 1 (D-44): the tolerance the tests ran at, and the tag that says it is not tau_0
        row["tau0_nm"] = _num(variant.tau_nm)
        row["analysis"] = mtau.analysis_with_tau(_ANALYSIS_NAME, variant)
    row["calibration_note"] = NOT_CALIBRATED_DETAIL
    if run is not None:
        v = run.viability
        # D-41: the export carries what the window shows. A pair rule v2 did not select (NOT VIABLE) has no statistic
        # here; a test's joint over all of its pairs only when every one of them is VIABLE (it is then that test's
        # joint over the viable pairs); otherwise the arc test's restricted joints are primary_* / sensitivity_*.
        viable, selected = set(v.viable), set(v.viable) | set(v.marginal)
        for prefix, pairs in (("arcc", arc_pairs), ("cols2d", col_pairs)):
            for k in (0, 1):
                if k < len(pairs) and (int(pairs[k].ring_a), int(pairs[k].ring_b)) not in selected:
                    row[f"{prefix}_zeta_pair{k}"] = row[f"{prefix}_p_excess_pair{k}"] = None
        row["arcc_pairs"] = " | ".join(
            f"{p.ring_a}-{p.ring_b}: zeta {float(p.zeta):+.3f}, p_excess {float(p.p_excess):.4f}"
            + ("" if (int(p.ring_a), int(p.ring_b)) in viable else " (marginal: sensitivity only)")
            for p in arc_pairs if (int(p.ring_a), int(p.ring_b)) in selected)
        for test, joint_cols in ((arc_c, ("arcc_z_A", "arcc_p_A", "arcc_T_A")), (cols, ("cols2d_z_A", "cols2d_p_A")),
                                 (arc_l, ("arcl_z_A", "arcl_p_A"))):
            keys = [] if test is None else [(int(p.ring_a), int(p.ring_b)) for p in test.adjacent]
            if not keys or not set(keys) <= viable:
                for c in joint_cols:
                    row[c] = None
        row.update({
            "viability_rule": v.rule,
            "pair_verdicts": " | ".join(f"{p.ring_a}-{p.ring_b}: {p.verdict}" + (f" ({'; '.join(p.reasons)})" if p.reasons else "")
                                        for p in v.pairs),
            "viable_pairs": " ".join(f"{a}-{b}" for a, b in v.viable),
            "marginal_pairs": " ".join(f"{a}-{b}" for a, b in v.marginal),
        })
        for name, tier in (("primary", run.primary), ("sensitivity", run.sensitivity)):
            for f in TIER_STAT_FIELDS:
                val = None if tier is None else tier.get(f)
                row[f"{name}_{f}"] = val if isinstance(val, (str, int)) else (None if val is None else _num(val))
    return {c: row.get(c) for c in RESULTS_TABLE_COLUMNS}


# ============================================================================
# The window
# ============================================================================

class _PassiveScatter(pg.ScatterPlotItem):
    """A scatter item that never takes a mouse click, so a click on the marker under it reaches that marker (the
    ring drawn around the selected table row sits above the markers; a plain ScatterPlotItem accepts every click in
    its bounding box and swallowed the click on its own marker, the GUI review of H5-D)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

    def mouseClickEvent(self, ev: Any) -> None:
        ev.ignore()


class _Relay(QtCore.QObject):
    """Carries a worker's news into the GUI thread (queued: emitted from the worker, received in the GUI thread).
    Never parented: a worker that outlives its window still has a live object to emit on."""

    progress = QtCore.pyqtSignal(int, str, float)       # generation, text, fraction
    finished = QtCore.pyqtSignal(int, object, object)   # generation, result, error
    membrane = QtCore.pyqtSignal(int, object, object)   # generation, CentroidMembrane or None, error
    viability = QtCore.pyqtSignal(int, object, object)  # generation, ReviewViability or None, error
    criteria = QtCore.pyqtSignal(int, object, object)   # generation, AxonCriteria or None, error text


class _Superseded(BaseException):
    """Raised by a live run's progress callback once a newer state made it unnecessary (H6 toggles): a
    BaseException, so the analyses' own ``except Exception`` (every failure becomes a warning) lets it through."""


@dataclass(frozen=True)
class _SupersededRun:
    """What a stopped live run reports instead of a ``ReviewRun``."""

    seq: int


def _min_height_for_width(layout: Any, width: int) -> int:
    """The height a vertical layout's rows need at ``width``: a word-wrapped label at its wrapped height, every other
    row at its minimum (a scrolling table at its own minimum), plus the spacing and margins."""
    m = layout.contentsMargins()
    inner = max(int(width) - m.left() - m.right(), 50)
    total, n = m.top() + m.bottom(), 0
    for i in range(layout.count()):
        it = layout.itemAt(i)
        w = it.widget()
        if (w is not None and w.isHidden()) or it.spacerItem() is not None:
            continue
        h = it.heightForWidth(inner) if it.hasHeightForWidth() else -1
        total += h if h > 0 else it.minimumSize().height()
        n += 1
    return int(total + max(int(layout.spacing()), 0) * max(n - 1, 0))


class _ColumnHolder(QtWidgets.QWidget):
    """The scroll area's widget: no layout (a layout with word-wrapped labels would make the scroll area use their
    PREFERRED height); the column fills it."""

    def __init__(self, child: QtWidgets.QWidget) -> None:
        super().__init__()
        self.child = child
        child.setParent(self)

    def resizeEvent(self, event: Any) -> None:
        self.child.setGeometry(self.rect())
        super().resizeEvent(event)


class _ColumnScroll(QtWidgets.QScrollArea):
    """H6 toggles: the review's side column scrolls instead of squeezing its rows over each other when the window is
    short (a 1366 x 768 screen with the criteria switches above): the column is as tall as the visible area, and at
    least as tall as its rows need at this width (``_min_height_for_width``)."""

    def __init__(self, child: QtWidgets.QWidget) -> None:
        super().__init__()
        self.setObjectName("side_scroll")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.column = child
        self.holder = _ColumnHolder(child)
        self.setWidget(self.holder)
        self.viewport().installEventFilter(self)
        child.installEventFilter(self)
        self._queued = False

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.LayoutRequest) and not self._queued:
            self._queued = True
            QtCore.QTimer.singleShot(0, self.refit)
        return False

    def refit(self) -> None:
        self._queued = False
        try:
            lay = self.column.layout()
            if lay is None:
                return
            need = _min_height_for_width(lay, self.viewport().width())
            if self.holder.minimumHeight() != need:
                self.holder.setMinimumHeight(need)
        except RuntimeError:        # being destroyed
            return


def _fmt(value: Any, spec: str = "+.3f", missing: str = "n/a") -> str:
    v = _num(value)
    if v is None:
        return missing
    text = format(v, spec)
    # A small negative number rounds to "-0" (a hull depth of -0.3 nm printed as "-0"): print it as a zero.
    if text.startswith("-") and not any(ch in "123456789" for ch in text):
        text = ("+" if "+" in spec else "") + text[1:]
    return text


def _table_number(value: Any, spec: str = ".0f") -> str:
    return _fmt(value, spec, "n/a")


def _clean_via(text: str) -> str:
    """A file name safe inside an edit's ``via`` (the log's text never holds ';' nor '|')."""
    return str(text).replace(";", ",").replace("|", "/")


def _item_flags(*flags: Any) -> Any:
    """Qt.ItemFlags of several Qt.ItemFlag (an OR of two bare flags is an int to the stubs)."""
    out = QtCore.Qt.ItemFlags(flags[0])
    for f in flags[1:]:
        out = out | f
    return out


class ColumnsWindow(QtWidgets.QMainWindow):
    """
    The lumen review of one axon's rings (D-35b): the clusters in the axon
    frame drawn by class, the doubtful ones listed with a checkbox each,
    the re-run of the ring analyses in a worker thread, the results with
    the leak banner, the provenance flags, and the export.

    ``res``: the rings of ``build_rings`` (never modified: every run cleans
    a copy). ``classification``: their ``LumenClassification``.
    ``decisions``: a ``LumenDecisions`` of the same clusters (None: the
    automatic state). ``lpz_nm``: the axial precision of every localization
    ``build_rings`` received (``clean_rings`` recomputes the rings'
    medians with it). ``columns_params`` / ``n_null``: the tests'
    parameters (None: the pre-registered ones, D-22). ``widefield_images``:
    a ``ReviewWidefield`` to draw under the clusters. ``identity``: an
    ``AxonIdentity`` or a callable returning one (asked at the export).
    ``roi_text``: the ROI as ``tools.cluster_quality.describe_roi`` names
    it (with the source it makes the ``axon_id`` the other tables use).
    ``membrane_before``: the centroid membrane through every cluster (None:
    fitted here in the background). ``widefield_note``: why there is no
    widefield, shown in the warning. ``extra_warnings``: listed first.
    ``review_store``: a ``tools.mps_lumen.LumenReviewStore`` (None: nothing
    kept between sessions) -- when given and no ``decisions`` are, the
    axon's saved review is restored (its final states, log and provenance:
    the "edited after results were shown" flag belongs to the axon, D-35b),
    and every change is saved there at once. ``rings_note``: how the rings
    were built, shown in the header.
    """

    analysis_finished = QtCore.pyqtSignal(object)
    # Every finished press of the run button (a ReviewRun), the one that computed nothing included.
    run_finished = QtCore.pyqtSignal(object)

    def __init__(
        self,
        res: RingsResult,
        classification: LumenClassification,
        *,
        decisions: Optional[LumenDecisions] = None,
        lpz_nm: Optional[NDArray[np.float64]] = None,
        columns_params: Optional[ColumnsParams] = None,
        n_null: Optional[int] = None,
        widefield_images: Optional[ReviewWidefield] = None,
        identity: Any = None,
        source_name: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
        roi_text: str = "",
        membrane_before: Optional[Any] = None,
        widefield_note: str = "",
        extra_warnings: Optional[Sequence[str]] = None,
        review_store: Optional[LumenReviewStore] = None,
        rings_note: str = "",
        xyz_lab_nm: Optional[Sequence[Any]] = None,
        review_inputs: Optional[ColumnsReviewInputs] = None,
        viability_fn: Optional[Callable[[RingsResult, Optional[Sequence[Any]]], ReviewViability]] = None,
        zquality_callback: Optional[Callable[[Any], Any]] = None,
        batch_callback: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().__init__(parent)
        c = classification
        # D-41: which pairs the column analysis runs on. ``xyz_lab_nm``: the LAB coordinates of every localization
        # build_rings received (rebuilt from the axon frame when not given); ``review_inputs``: the selection itself
        # (the simulated-null button needs its frames and precisions; None disables that button).
        self.xyz_lab_nm = None if xyz_lab_nm is None else tuple(np.asarray(v, dtype=np.float64).reshape(-1)
                                                                for v in xyz_lab_nm)
        self.review_inputs = review_inputs
        self.viability_fn = viability_fn if viability_fn is not None else DEFAULT_VIABILITY_FN
        self.viability: Optional[ReviewViability] = None
        self.viability_error: Optional[str] = None
        # H6 toggles (D-43): rule v2's own viability (``base_viability``), the per-pair criteria of both rules (None
        # with a custom ``viability_fn``: the switches are off), the selection's result, the analyses of every cluster
        # set already analysed (by lumen fingerprint: they do not depend on the selection), and the live re-run state
        self.base_viability: Optional[ReviewViability] = None
        self.criteria: Optional[Any] = None
        self.criteria_error = ""
        self.selection_result: Optional[Any] = None
        self._analyses: Dict[Any, CleanedAnalyses] = {}
        self._live_armed = False
        self._run_seq = 0
        self._cancel_seq = 0
        self._run_fp: Any = None
        self._run_live = False
        self._pending_live = False
        self._closed = False
        self.n_live_runs = 0
        self.exploration_log: Optional[msel.ExplorationLog] = None
        self.exploration_log_error = ""
        self.selection_banner_text = ""
        self.last_run: Optional[ReviewRun] = None
        self.zquality_window: Optional[Any] = None
        self.simnull_dialog: Optional[Any] = None
        self.batch_dialog: Optional[Any] = None
        # UI stage 0: the program's single Z quality view and Columns batch dialog (MPS_explorer.open_z_quality(
        # inputs) / open_columns_batch); None (a window run on its own): this window opens its own copies
        self.zquality_callback = zquality_callback
        self.batch_callback = batch_callback
        self._viability_generation = 0
        self.res = res
        self.classification = c
        self.source_name = str(source_name or res.source_name or "")
        self.roi_text = str(roi_text or "")
        self.review_store = review_store
        self.rings_note = str(rings_note or "")
        self.store_notes: List[str] = []
        self._store_error: Optional[str] = None
        if decisions is None:
            restored: Optional[LumenDecisions] = None
            if review_store is not None:
                restored, self.store_notes = review_store.restore(self.axon_id, c)
            decisions = restored if restored is not None else LumenDecisions.from_classification(c)
        elif list(decisions.stable_keys) != list(c.stable_keys):
            raise ValueError("ColumnsWindow: the decisions describe other clusters than the classification")
        self.decisions: LumenDecisions = decisions
        self.lpz_nm = None if lpz_nm is None else np.asarray(lpz_nm, dtype=np.float64)
        self.columns_params = columns_params
        self.n_null = None if n_null is None else int(n_null)
        # UI stage 1 (D-44): the tolerance control. ``base_columns_params``: the parameters it reads its choices from
        # (the window's own, or the pre-registered file, read once); a run at tau_0 still passes ``columns_params``
        # exactly as before, a run at another tau a copy of these with that tau (tools.mps_tau.params_at_tau). The
        # control starts at tau_0 in every window and is never saved.
        self.base_columns_params: Optional[ColumnsParams] = None
        self.tau_unavailable = ""
        try:
            base = columns_params if columns_params is not None else default_columns_params()
            mtau.tau_choices(base)
            self.base_columns_params = base
        except Exception as exc:  # noqa: BLE001 - the control is off, with the reason; the rest works as before
            self.tau_unavailable = f"the column parameters could not be read ({type(exc).__name__}: {exc})"
        self.tau0_nm: Optional[float] = (None if self.base_columns_params is None
                                         else float(self.base_columns_params.tau0_nm))
        self.tau_nm: Optional[float] = self.tau0_nm
        self._run_tau: Optional[float] = self.tau0_nm
        self._cancel_reason = ""
        # the results at the current tau are being computed (or wait for the button): no p of another tau is shown,
        # and ``_pending_keep`` holds the analyses they replace for what does not depend on tau (the curve, the
        # membrane after the cleaning)
        self._tau_pending = False
        self._pending_keep: Optional[CleanedAnalyses] = None
        self._counter_line_text = ""
        self.curve_shown: Optional[mtau.CurveView] = None
        self._tau_rerun_wanted = False
        # the control has left tau_0 at least once in this window (review of UI stage 1): from then on every p shown is
        # logged and counted, also without the criteria of the selection (_selection_texts)
        self._tau_touched = False
        self._left_split_by_user = False       # the user dragged the map / curve handle: their split is kept
        self.widefield_images = widefield_images
        self.identity = identity
        self.widefield_note = str(widefield_note or "")
        self.extra_warnings: List[str] = [str(w) for w in (extra_warnings or [])]
        self.last_result: Optional[CleanedAnalyses] = None
        self.membrane_before = membrane_before
        self._run_error: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._generation = 0
        self._syncing = False
        self._relay = _Relay()
        self._relay.progress.connect(self._on_progress)
        self._relay.finished.connect(self._on_run_finished)
        self._relay.membrane.connect(self._on_membrane)
        self._relay.viability.connect(self._on_viability)
        self._relay.criteria.connect(self._on_criteria)
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(LIVE_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._request_live_run)
        self._index ={k: i for i, k in enumerate(c.stable_keys)}
        self._ring_of = {k: int(c.keys[i][0]) for i, k in enumerate(c.stable_keys)}
        self._centroids = np.asarray(c.centroids_nm, dtype=np.float64).reshape(-1, 2)

        self.setWindowTitle(f"H-ECL columns: lumen review - {os.path.basename(self.source_name) or 'axon'}")
        self.resize(1440, 900)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self.header_label = QtWidgets.QLabel()
        self.header_label.setWordWrap(True)
        self.header_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.header_label)
        self.widefield_warning = QtWidgets.QLabel()
        self.widefield_warning.setWordWrap(True)
        self.widefield_warning.setStyleSheet(f"color: {verdict('warn', dark=False)}; font-weight: bold;")
        root.addWidget(self.widefield_warning)
        # H6 toggles (D-43): the criteria switches, the program's one selection, across the whole width (the side
        # column has no height to spare on a 1366 x 768 screen); off until the criteria of both rules are known
        self.selection_widget = SelectionWidget(parent=central)
        self.selection_widget.set_available(False, "computing the criteria of both rules...")
        self.selection_widget.state.changed.connect(self._on_selection)
        root.addWidget(self.selection_widget)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        # UI stage 1: the map on top of the E(tau) curve of the 2D test (a vertical splitter; the split follows the
        # window, _fit_left_split, until the user drags its handle)
        self.left_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.left_splitter.setObjectName("left_splitter")
        self.left_splitter.setChildrenCollapsible(False)
        self.left_splitter.addWidget(self._build_plot_column())
        # the curve section scrolls on a short screen instead of squeezing its plot and texts
        self.curve_scroll = _ColumnScroll(self._build_curve_section())
        self.curve_scroll.setObjectName("curve_scroll")
        self.left_splitter.addWidget(self.curve_scroll)
        self.left_splitter.setStretchFactor(0, 3)
        self.left_splitter.setStretchFactor(1, 2)
        self.left_splitter.setSizes([540, 360])
        self.left_splitter.splitterMoved.connect(self._on_left_split_moved)
        splitter.addWidget(self.left_splitter)
        # H6 toggles: the side column scrolls on a short screen instead of drawing its rows over each other
        self.side_scroll = _ColumnScroll(self._build_side_column())
        splitter.addWidget(self.side_scroll)
        splitter.setSizes([int(self.width() * 0.52), int(self.width() * 0.48)])
        root.addWidget(splitter, stretch=1)
        missing = apply_tooltips(self, COLUMNS_WINDOW_TOOLTIPS)
        if missing:  # a renamed widget; never silent
            self.extra_warnings.append(f"(window) no widget for the tooltips of {missing}")
        # UI stage 1: the control's and the curve's tooltips (tools/mps_tau_ui.py; must stay empty, a test checks)
        self.tau_missing_tooltips: List[str] = apply_tooltips(self, mtui.TAU_UI_TOOLTIPS)

        self._fill_header()
        self._fill_widefield_warning()
        self._fill_table()
        self._draw_underlay()
        self._draw_markers()
        self._draw_membranes()
        self._fit_view()
        self._refresh_state()
        self._fill_warnings()
        if self.membrane_before is None:
            self._fit_membrane_before()
        self._refresh_tau_views()
        self._compute_viability()

    # ------------------------------------------------------------ construction
    def _build_plot_column(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setObjectName("plot_widget")
        style_dark(self.plot_widget)
        # short: at 1366 px the plot is ~450 px wide beside its layer panel, and a pyqtgraph title is clipped, never
        # wrapped; "click to remove / keep" heads the panel's cluster rows right beside it
        set_title(self.plot_widget, "Ring clusters in the axon frame")
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabels(bottom="x' [nm]", left="y' [nm]")
        self._view_target: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.plot_widget.getViewBox().sigResized.connect(self._reapply_view)
        # Once the user pans or zooms, a resize keeps their view (zooming is how overlapping markers are separated).
        self.plot_widget.getViewBox().sigRangeChangedManually.connect(self._on_user_range)
        self.underlay_item = pg.ImageItem()
        self.underlay_item.setZValue(-20)
        self.underlay_item.setOpacity(UNDERLAY_OPACITY)
        self.underlay_item.setVisible(False)
        self.plot_widget.addItem(self.underlay_item)
        self.membrane_before_item = pg.PlotDataItem(pen=self._before_pen())
        self.membrane_before_item.setZValue(-5)
        self.membrane_after_item = pg.PlotDataItem(pen=self._after_pen())
        self.membrane_after_item.setZValue(-4)
        self.plot_widget.addItem(self.membrane_before_item)
        self.plot_widget.addItem(self.membrane_after_item)
        # UI stage 1 (T5): the clusters each column test matched at the current tau for the curve's pair, as thin
        # segments between the membranes and the markers; they never take a click (the markers above them do)
        self.match_items: Dict[str, pg.PlotDataItem] = {}
        for key, pen, z in (
                ("matches_arc", pg.mkPen(ROLES["matched"], width=2.0), MATCH_ARC_Z),
                ("matches_2d", pg.mkPen(ROLES["matched"], width=1.5, style=QtCore.Qt.PenStyle.DashLine), MATCH_2D_Z)):
            item = pg.PlotDataItem(pen=pen, connect="pairs")
            item.setZValue(z)
            for part in (item, item.curve, item.scatter):
                part.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            self.plot_widget.addItem(item)
            self.match_items[key] = item
        # UI stage 0: one marker item per class, stacked in _DRAW_ORDER (bottom to top), so the layer panel hides the
        # REAL markers of a class. Qt delivers no click or hover to a hidden item (a hidden class cannot be clicked),
        # and pyqtgraph hands a click no point of an item claims to the item underneath, so the topmost VISIBLE
        # marker toggles, as within the one item before. The tooltips have one owner (_on_mouse_moved): per-item tips
        # would blank each other (ScatterPlotItem.hoverEvent clears the view's tip when its own item has no point).
        self.marker_items: Dict[str, pg.ScatterPlotItem] = {}
        for cls in _DRAW_ORDER:
            item = pg.ScatterPlotItem(hoverable=False, tip=None)
            item.setZValue(0.1 * _DRAW_ORDER.index(cls))
            item.sigClicked.connect(self._on_points_clicked)
            self.plot_widget.addItem(item)
            self.marker_items[cls] = item
        self._hover_tip = ""
        self._hover_pos: Optional[Any] = None      # the last mouse position over the scene (a redraw re-reads it)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.highlight_item = _PassiveScatter(symbol="o", size=HIGHLIGHT_SIZE_PX, brush=None,
                                              pen=pg.mkPen(neutral(dark=True), width=2))
        self.highlight_item.setZValue(5)
        self.plot_widget.addItem(self.highlight_item)
        plot_row = QtWidgets.QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.addWidget(self.plot_widget, stretch=1)
        plot_row.addWidget(self._build_layer_panel())
        lay.addLayout(plot_row, stretch=1)
        # a class shown or hidden under a still mouse: the tooltip follows at once
        self.layer_panel.toggled.connect(lambda _key, _on: self._refresh_hover())

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Show:"))
        self.ring_filter = QtWidgets.QComboBox()
        self.ring_filter.setObjectName("ring_filter")
        self.ring_filter.addItem("every ring", None)
        rings_sorted = sorted(self.res.rings, key=lambda q: int(q.index))
        for r in rings_sorted:
            self.ring_filter.addItem(f"ring {int(r.index)} only ({ring_colour_name(int(r.index))} outline)", int(r.index))
        self.ring_filter.currentIndexChanged.connect(lambda _i: self._draw_markers())
        row.addWidget(self.ring_filter)
        # Which outline colour is which ring (the shapes are the classes, not the rings).
        self.ring_key_label = QtWidgets.QLabel(
            "Outline = ring: " + ", ".join(f"{int(r.index)} {ring_colour_name(int(r.index))}" for r in rings_sorted))
        row.addWidget(self.ring_key_label)
        row.addStretch(1)
        lay.addLayout(row)
        return w

    def _build_layer_panel(self) -> LayerPanel:
        """The layers of the map, beside it (UI stage 0: the legend that sat over the data is gone): the five
        classes with their counts, the two centroid membranes, the widefield image underneath."""
        panel = LayerPanel()
        panel.setObjectName("layer_panel")
        self.layer_panel = panel
        panel.add_group("Clusters (click to remove / keep)")
        for cls in DISPLAY_CLASSES:
            role_name, alpha, symbol = LUMEN_CLASS_STYLE[cls]
            label = DISPLAY_LABELS[cls]
            panel.add_layer(cls, LAYER_LABELS[cls],
                            Swatch("symbol", self._colour(role_name, alpha), symbol=symbol),
                            items=[self.marker_items[cls]],
                            tip=f"Draw the clusters {label} (the number is how many are drawn). Hidden, they cannot "
                                "be clicked; the analysis is not affected.")
        panel.add_group("Centroid membrane")
        panel.add_layer("membrane_before", LAYER_LABELS["membrane_before"],
                        Swatch("line", neutral(dark=True), dash=True), items=[self.membrane_before_item],
                        tip="The centroid membrane through every cluster of every ring, before the cleaning (dashed).")
        panel.add_layer("membrane_after", LAYER_LABELS["membrane_after"],
                        Swatch("line", ROLES[MEMBRANE_AFTER_ROLE]), items=[self.membrane_after_item],
                        tip="The centroid membrane of the last run, without the removed clusters (solid).")
        panel.add_group("Image")
        self.underlay_check = QtWidgets.QCheckBox("Widefield image underneath")
        self.underlay_check.setObjectName("underlay_check")
        self.underlay_combo = QtWidgets.QComboBox()
        self.underlay_combo.setObjectName("underlay_combo")
        wf = self.widefield_images
        if wf is not None and wf.tubulin_image is not None:
            self.underlay_combo.addItem("betaIII-tubulin", "tubulin")
        if wf is not None and wf.spectrin_image is not None:
            self.underlay_combo.addItem("betaII-spectrin", "spectrin")
        has_images = self.underlay_combo.count() > 0
        self.underlay_check.setChecked(has_images)
        # _draw_underlay owns underlay_item's visibility: it also depends on whether an image covers the clusters
        panel.add_layer("underlay", LAYER_LABELS["underlay"], Swatch("image"), items=(),
                        on_toggle=lambda _on: self._draw_underlay(), checkbox=self.underlay_check,
                        tip=COLUMNS_WINDOW_TOOLTIPS.get("underlay_check", ""))
        panel.add_widget(self.underlay_combo)
        panel.set_enabled("underlay", has_images, "no widefield image for this axon")
        self.underlay_combo.setVisible(has_images)     # an empty, greyed combo says nothing
        self.underlay_combo.currentIndexChanged.connect(lambda _i: self._draw_underlay())
        # UI stage 1 (T5): what each column test matched at the current tau, for the pair of the curve below
        panel.add_group(mtui.MATCH_GROUP_TITLE)
        panel.add_layer("matches_arc", mtui.MATCH_LABELS["matches_arc"], Swatch("line", ROLES["matched"], width=2.0),
                        items=[self.match_items["matches_arc"]], tip=mtui.MATCH_TIPS["matches_arc"], visible=True)
        panel.add_layer("matches_2d", mtui.MATCH_LABELS["matches_2d"],
                        Swatch("line", ROLES["matched"], dash=True, width=1.5), items=[self.match_items["matches_2d"]],
                        tip=mtui.MATCH_TIPS["matches_2d"], visible=False)
        return panel

    def _build_curve_section(self) -> QtWidgets.QWidget:
        """UI stage 1 (T4): the E(tau) curve of the 2D test for one selected pair, as every run computes it (it does
        not depend on tau: changing tau moves the marker only), with its layer panel beside it, its global p, the
        counter and a note that always says what the curve is not."""
        w = QtWidgets.QWidget()
        w.setObjectName("curve_section")
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(3)
        head = QtWidgets.QHBoxLayout()
        self.curve_head_label = QtWidgets.QLabel(mtui.CURVE_HEAD_TEXT)
        f = self.curve_head_label.font()
        f.setBold(True)
        self.curve_head_label.setFont(f)
        head.addWidget(self.curve_head_label)
        self.curve_pair_combo = QtWidgets.QComboBox()
        self.curve_pair_combo.setObjectName("curve_pair_combo")
        self.curve_pair_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.curve_pair_combo.currentIndexChanged.connect(self._on_curve_pair)
        head.addWidget(self.curve_pair_combo)
        head.addStretch(1)
        lay.addLayout(head)
        self.curve_plot = pg.PlotWidget()
        self.curve_plot.setObjectName("curve_plot")
        style_dark(self.curve_plot)
        set_title(self.curve_plot, mtui.CURVE_TITLE)
        self.curve_plot.setLabels(bottom="tau [nm]", left="E (matched fraction)")
        self.curve_plot.setMouseEnabled(x=False, y=False)
        self.curve_plot.setMenuEnabled(False)
        self.curve_plot.hideButtons()
        self.curve_plot.setMinimumHeight(CURVE_PLOT_MIN_HEIGHT)
        dark_neutral = neutral(dark=True)
        self.curve_band_lo = pg.PlotCurveItem()
        self.curve_band_hi = pg.PlotCurveItem()
        self.curve_band_item = pg.FillBetweenItem(self.curve_band_lo, self.curve_band_hi,
                                                  brush=pg.mkBrush(rgba("randomized", 60)))
        self.curve_band_item.setZValue(-10)
        self.curve_mean_item = pg.PlotDataItem(
            pen=pg.mkPen(ROLES["randomized"], width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.curve_mean_item.setZValue(-5)
        self.curve_observed_item = pg.PlotDataItem(
            pen=pg.mkPen(ROLES["observed"], width=2.0), symbol="o", symbolSize=6,
            symbolBrush=pg.mkBrush(ROLES["observed"]), symbolPen=pg.mkPen(dark_neutral, width=0.8))
        self.curve_tau0_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(dark_neutral, width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.curve_tau_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(ROLES["warn"], width=2.0))
        for item in (self.curve_band_item, self.curve_mean_item, self.curve_observed_item, self.curve_tau0_line,
                     self.curve_tau_line):
            self.curve_plot.addItem(item)
        self.curve_tau0_line.setVisible(self.tau0_nm is not None)
        self.curve_tau_line.setVisible(False)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.curve_plot, stretch=1)
        layers = LayerPanel()
        layers.setObjectName("curve_layers")
        self.curve_layers = layers
        labels, tips = mtui.CURVE_LAYER_LABELS, mtui.CURVE_LAYER_TIPS
        layers.add_layer("observed", labels["observed"], Swatch("symbol", ROLES["observed"], symbol="o"),
                         items=[self.curve_observed_item], tip=tips["observed"])
        layers.add_layer("null_mean", labels["null_mean"], Swatch("line", ROLES["randomized"], dash=True),
                         items=[self.curve_mean_item], tip=tips["null_mean"])
        layers.add_layer("null_band", labels["null_band"], Swatch("bar", rgba("randomized", 60)),
                         items=[self.curve_band_item], tip=tips["null_band"])
        layers.add_layer("tau0", labels["tau0"], Swatch("line", dark_neutral, dash=True),
                         items=[self.curve_tau0_line], tip=tips["tau0"])
        # the chosen tau's line is hidden at tau_0 whatever the row says: the window owns its visibility
        layers.add_layer("tau", labels["tau"], Swatch("line", ROLES["warn"]), items=(),
                         on_toggle=lambda _on: self._update_curve_marker(), tip=tips["tau"])
        if self.tau0_nm is None:
            layers.set_enabled("tau0", False, "the column parameters could not be read")
            layers.set_enabled("tau", False, "the column parameters could not be read")
        row.addWidget(layers)
        lay.addLayout(row, stretch=1)
        self.curve_p_label = QtWidgets.QLabel(mtui.CURVE_NO_RUN_TEXT)
        self.curve_p_label.setObjectName("curve_p_label")
        self.curve_p_label.setWordWrap(True)
        self.curve_p_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.curve_p_label)
        self.curve_counter_label = QtWidgets.QLabel()
        self.curve_counter_label.setObjectName("curve_counter_label")
        self.curve_counter_label.setWordWrap(True)
        self.curve_counter_label.setStyleSheet(f"color: {verdict('warn', dark=False)};")
        self.curve_counter_label.setVisible(False)
        lay.addWidget(self.curve_counter_label)
        self.curve_note = QtWidgets.QLabel(mtui.CURVE_NOTE_TEXT)
        self.curve_note.setObjectName("curve_note")
        self.curve_note.setWordWrap(True)
        self.curve_note.setStyleSheet(f"color: {verdict('dim', dark=False)};")
        lay.addWidget(self.curve_note)
        return w

    def _build_side_column(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel("Doubtful clusters (ticked = removed by you; unticked = kept, the automatic state)"))
        self.doubtful_table = QtWidgets.QTableWidget(0, 12)
        self.doubtful_table.setObjectName("doubtful_table")
        self.doubtful_table.setHorizontalHeaderLabels([
            "Remove", "Ring", "Label", "Hull depth [nm]", "Hop 250 [nm]", "Hop 300 [nm]", "Tubulin depth [nm]",
            "Spectrin depth [nm]", "n locs", "n events", "|z' - ring centre| [nm]", "Flags"])
        header = self.doubtful_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        self.doubtful_table.verticalHeader().setVisible(False)
        self.doubtful_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.doubtful_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.doubtful_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.doubtful_table.setAlternatingRowColors(True)
        self.doubtful_table.itemChanged.connect(self._on_table_item_changed)
        self.doubtful_table.itemSelectionChanged.connect(self._on_table_selection)
        lay.addWidget(self.doubtful_table, stretch=3)

        buttons = QtWidgets.QHBoxLayout()
        self.select_all_doubtful_button = QtWidgets.QPushButton("Select all doubtful")
        self.select_all_doubtful_button.setObjectName("select_all_doubtful_button")
        self.select_all_doubtful_button.clicked.connect(self._on_select_all)
        self.clear_all_doubtful_button = QtWidgets.QPushButton("Clear all doubtful")
        self.clear_all_doubtful_button.setObjectName("clear_all_doubtful_button")
        self.clear_all_doubtful_button.clicked.connect(self._on_clear_all)
        self.reset_button = QtWidgets.QPushButton("Back to the automatic rule")
        self.reset_button.setObjectName("reset_button")
        self.reset_button.clicked.connect(self._on_reset)
        for b in (self.select_all_doubtful_button, self.clear_all_doubtful_button, self.reset_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        lay.addLayout(buttons)
        self.counts_label = QtWidgets.QLabel()
        self.counts_label.setWordWrap(True)
        lay.addWidget(self.counts_label)

        # D-41: the verdict of every ring pair, before anything is run
        self.viability_label = QtWidgets.QLabel("Ring-pair viability (rule v2, D-41): computing...")
        self.viability_label.setWordWrap(True)
        lay.addWidget(self.viability_label)
        self.viability_table = QtWidgets.QTableWidget(0, 3)
        self.viability_table.setObjectName("viability_table")
        self.viability_table.setHorizontalHeaderLabels(["Pair", "Verdict (rule v2)", "Why"])
        self.viability_table.verticalHeader().setVisible(False)
        self.viability_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # one line per pair (the failed criteria; the full reasons in the tooltip): two or three pairs stay in view
        # even when the window is squeezed onto a 1366 x 768 screen
        self.viability_table.setWordWrap(False)
        self.viability_table.verticalHeader().setDefaultSectionSize(self.viability_table.fontMetrics().height() + 8)
        self.viability_table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.viability_table, stretch=1)
        tools_row = QtWidgets.QHBoxLayout()
        self.zquality_button = QtWidgets.QPushButton("Z quality...")
        self.zquality_button.setObjectName("zquality_button")
        self.zquality_button.clicked.connect(self._on_zquality)
        tools_row.addWidget(self.zquality_button)
        self.columns_batch_button = QtWidgets.QPushButton("Columns batch...")
        self.columns_batch_button.setObjectName("columns_batch_button")
        self.columns_batch_button.clicked.connect(self._on_columns_batch)
        tools_row.addWidget(self.columns_batch_button)
        tools_row.addStretch(1)
        lay.addLayout(tools_row)
        # UI stage 0: the simulated null is experimental (its calibration was not accepted, D-41): it lives in a
        # collapsed "Advanced (experimental)" section, with one line on what it is
        self.advanced_toggle = QtWidgets.QToolButton()
        self.advanced_toggle.setObjectName("advanced_toggle")
        self.advanced_toggle.setText("Advanced (experimental)")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.advanced_toggle.setAutoRaise(True)
        lay.addWidget(self.advanced_toggle)
        self.advanced_box = QtWidgets.QWidget()
        self.advanced_box.setObjectName("advanced_box")
        adv = QtWidgets.QHBoxLayout(self.advanced_box)
        adv.setContentsMargins(18, 0, 0, 0)
        self.simnull_button = QtWidgets.QPushButton("Simulated-null p (not calibrated)...")
        self.simnull_button.setObjectName("simnull_button")
        self.simnull_button.clicked.connect(self._on_simnull)
        self.simnull_button.setEnabled(False)
        adv.addWidget(self.simnull_button)
        self.advanced_note = QtWidgets.QLabel("A p against simulated axons without columns. Experimental: read the "
                                              "dialog first.")
        self.advanced_note.setObjectName("advanced_note")
        self.advanced_note.setWordWrap(True)
        adv.addWidget(self.advanced_note, stretch=1)
        self.advanced_box.setVisible(False)
        lay.addWidget(self.advanced_box)
        self.advanced_toggle.toggled.connect(self._on_advanced_toggled)

        # UI stage 1 (D-44): the tolerance of the column tests, always in view right above the run
        self.tau_control = mtui.TauControl(self.base_columns_params, unavailable=self.tau_unavailable)
        self.tau_control.changed.connect(self._on_tau_changed)
        lay.addWidget(self.tau_control)

        run_row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Apply and re-run analyses")
        self.run_button.setObjectName("run_button")
        self.run_button.clicked.connect(self._on_run)
        run_row.addWidget(self.run_button)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        run_row.addWidget(self.progress_bar, stretch=1)
        self.live_update_check = QtWidgets.QCheckBox("Live update")
        self.live_update_check.setObjectName("live_update_check")
        self.live_update_check.setChecked(True)
        self.live_update_check.toggled.connect(lambda _on: self._rerender_for_selection())
        run_row.addWidget(self.live_update_check)
        self.load_button = QtWidgets.QPushButton("Load decisions...")
        self.load_button.setObjectName("load_button")
        self.load_button.clicked.connect(self._on_load)
        run_row.addWidget(self.load_button)
        self.export_button = QtWidgets.QPushButton("Export tables...")
        self.export_button.setObjectName("export_button")
        self.export_button.clicked.connect(self._on_export)
        run_row.addWidget(self.export_button)
        lay.addLayout(run_row)

        self.leak_banner = QtWidgets.QLabel(LEAK_BANNER_TEXT)
        self.leak_banner.setWordWrap(True)
        self.leak_banner.setStyleSheet(f"color: {verdict('warn', dark=False)}; font-weight: bold;")
        self.leak_banner.setVisible(False)
        lay.addWidget(self.leak_banner)
        self.edited_label = QtWidgets.QLabel(EDITED_LABEL_TEXT)
        self.edited_label.setWordWrap(True)
        self.edited_label.setStyleSheet(f"color: {verdict('bad', dark=False)}; font-weight: bold;")
        self.edited_label.setVisible(False)
        lay.addWidget(self.edited_label)
        self.stale_label = QtWidgets.QLabel(STALE_LABEL_TEXT)
        self.stale_label.setWordWrap(True)
        self.stale_label.setStyleSheet(f"color: {verdict('dim', dark=False)};")
        self.stale_label.setVisible(False)
        lay.addWidget(self.stale_label)
        # H6 toggles: separate labels (the results text under the pre-specified rule stays exactly as it was): the
        # exploratory banner and the counter in ONE label next to the p values (no height to spare at 1366 x 768)
        self.selection_counter_label = QtWidgets.QLabel()
        self.selection_counter_label.setObjectName("selection_counter_label")
        self.selection_counter_label.setWordWrap(True)
        self.selection_counter_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.selection_counter_label.setStyleSheet(f"color: {verdict('warn', dark=False)};")
        self.selection_counter_label.setVisible(False)
        lay.addWidget(self.selection_counter_label)
        self.selection_note = QtWidgets.QLabel()
        self.selection_note.setObjectName("selection_note")
        self.selection_note.setWordWrap(True)
        self.selection_note.setStyleSheet(f"color: {verdict('dim', dark=False)};")
        self.selection_note.setVisible(False)
        lay.addWidget(self.selection_note)
        self.results_view = QtWidgets.QPlainTextEdit()
        self.results_view.setObjectName("results_view")
        self.results_view.setReadOnly(True)
        self.results_view.setPlaceholderText("No results yet: press 'Apply and re-run analyses'.")
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.results_view.setFont(font)
        lay.addWidget(self.results_view, stretch=5)  # H6 toggles: the results keep their room under the switches
        lay.addWidget(QtWidgets.QLabel("Warnings"))
        self.warnings_list = QtWidgets.QListWidget()
        self.warnings_list.setObjectName("warnings_list")
        self.warnings_list.setWordWrap(True)
        lay.addWidget(self.warnings_list, stretch=1)
        # H6 toggles: the criteria switches above took ~70 px of the height; the scrolling areas may shrink below
        # Qt's default minimum (87 px) so that the window still fits a 1366 x 768 screen (they scroll; the stretch
        # factors share every pixel above these minimums as before)
        for widget, h in ((self.doubtful_table, 52), (self.viability_table, 78), (self.results_view, 60),
                          (self.warnings_list, 44)):
            widget.setMinimumHeight(h)
        return w

    # ------------------------------------------------------------ small helpers
    @staticmethod
    def _colour(role_name: str, alpha: int) -> QtGui.QColor:
        col = QtGui.QColor(ROLES[role_name])
        col.setAlpha(int(alpha))
        return col

    @staticmethod
    def _before_pen() -> Any:
        return pg.mkPen(neutral(dark=True), width=1.5, style=QtCore.Qt.PenStyle.DashLine)

    @staticmethod
    def _after_pen() -> Any:
        return pg.mkPen(ROLES[MEMBRANE_AFTER_ROLE], width=2.5)

    @property
    def axon_id(self) -> str:
        """The axon's name in every table (``tools.mps_identity.axon_id`` of the source and the ROI)."""
        return axon_id(self.source_name, self.roi_text)

    def display_class(self, stable_key: str) -> str:
        """What the cluster ``stable_key`` is on screen now (``DISPLAY_CLASSES``)."""
        d = self.decisions
        return display_class_of(d.auto_class(stable_key), d.is_removed(stable_key))

    def is_running(self) -> bool:
        """True while a re-run is in the worker (until its results are shown)."""
        return bool(self._running)

    def results_text(self) -> str:
        """The results area as plain text."""
        return str(self.results_view.toPlainText())

    # ------------------------------------------------------------ header, warnings
    def _fill_header(self) -> None:
        c = self.classification
        rings = sorted(self.res.rings, key=lambda q: int(q.index))
        per_ring = ", ".join(f"ring {int(r.index)}: {len(r.clusters)}" for r in rings)
        frame = self.res.frame
        tilt = "no tilt correction" if frame.is_identity else f"axis tilt {float(frame.beta_deg):.2f} deg"
        self.header_label.setText(
            f"<b>{os.path.basename(self.source_name) or 'axon'}</b>"
            + (f" &mdash; ROI {self.roi_text}" if self.roi_text else "")
            + f" &mdash; {len(rings)} ring(s), {c.n} clusters ({per_ring}); {tilt}.<br>"
            f"Rule: {c.rule_description()}."
            + (f" Registration: {c.registration}." if c.has_widefield and c.registration else "")
            + (f"<br>{self.rings_note}" if self.rings_note else ""))

    def _fill_widefield_warning(self) -> None:
        if self.classification.has_widefield:
            self.widefield_warning.setVisible(False)
            return
        why = f" ({self.widefield_note})" if self.widefield_note else ""
        self.widefield_warning.setText(marked(
            "warn", f"No widefield images{why}: the lumen rule is isolation only (REMOVE = ISO_STRICT) and its "
                    "completeness is not certified -- a lumen cluster shallower than ~600 nm or close to other "
                    "clusters can stay. Review the doubtful clusters, and open the Axoplasm panel with both images "
                    "to use the full rule."))
        self.widefield_warning.setVisible(True)

    def _fill_warnings(self) -> None:
        self.warnings_list.clear()
        msgs: List[Tuple[str, str]] = []
        if self._store_error:
            msgs.append(("bad", f"this review is NOT kept between sessions ({self._store_error}): export the decisions "
                                "before closing"))
        msgs += [("dim", f"saved review: {w}") for w in self.store_notes]
        msgs += [("warn", w) for w in self.extra_warnings]
        msgs += [("warn", w) for w in self.classification.warnings if w != NO_WIDEFIELD_WARNING]
        msgs += [("warn", w) for w in self.decisions.warnings]
        if self._run_error:
            msgs.append(("bad", f"the last run failed: {self._run_error}"))
        if self.last_result is not None:
            msgs += [("warn", f"run: {w}") for w in self.last_result.warnings]
        msgs += [("dim", f"rings: {w}") for w in self.res.warnings]
        if not msgs:
            msgs = [("good", "No warnings.")]
        for kind, text in msgs:
            item = QtWidgets.QListWidgetItem(marked(kind, text))
            item.setForeground(QtGui.QColor(verdict(kind, dark=False)))
            self.warnings_list.addItem(item)

    # ------------------------------------------------------------ the table
    def _fill_table(self) -> None:
        c = self.classification
        t = self.doubtful_table
        self._syncing = True
        t.blockSignals(True)
        self.highlight_item.setData([])
        try:
            t.setRowCount(0)
            for i, k in enumerate(c.stable_keys):
                if self.decisions.auto_class(k) != AUTO_DOUBTFUL:
                    continue
                row = t.rowCount()
                t.insertRow(row)
                check = QtWidgets.QTableWidgetItem("")
                check.setFlags(_item_flags(QtCore.Qt.ItemFlag.ItemIsUserCheckable, QtCore.Qt.ItemFlag.ItemIsEnabled,
                                           QtCore.Qt.ItemFlag.ItemIsSelectable))
                check.setData(QtCore.Qt.ItemDataRole.UserRole, k)
                check.setCheckState(QtCore.Qt.CheckState.Checked if self.decisions.is_removed(k) else QtCore.Qt.CheckState.Unchecked)
                check.setToolTip(k)
                t.setItem(row, 0, check)
                ring = int(c.keys[i][0])
                flags = [r for r in c.reasons[i]] or ["-"]
                cells = [
                    f"{ring} ({ring_colour_name(ring)})", str(int(c.labels[i])) if len(c.labels) == c.n else "",
                    _table_number(c.hull_depth_nm[i]), _table_number(c.hop_sens_nm[i]), _table_number(c.hop_strict_nm[i]),
                    _table_number(c.depth_tubulin_nm[i]), _table_number(c.depth_spectrin_nm[i]), str(int(c.n_locs[i])),
                    _table_number(c.n_events[i]), _table_number(c.abs_dz_ring_nm[i]), ", ".join(flags)]
                for col, text in enumerate(cells, start=1):
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setFlags(_item_flags(QtCore.Qt.ItemFlag.ItemIsEnabled, QtCore.Qt.ItemFlag.ItemIsSelectable))
                    item.setToolTip(k)
                    t.setItem(row, col, item)
        finally:
            t.blockSignals(False)
            self._syncing = False

    def _row_of(self, key: str) -> int:
        t = self.doubtful_table
        for r in range(t.rowCount()):
            it = t.item(r, 0)
            if it is not None and it.data(QtCore.Qt.ItemDataRole.UserRole) == key:
                return r
        return -1

    def _sync_table(self) -> None:
        t = self.doubtful_table
        self._syncing = True
        t.blockSignals(True)
        try:
            for r in range(t.rowCount()):
                it = t.item(r, 0)
                if it is None:
                    continue
                want = QtCore.Qt.CheckState.Checked if self.decisions.is_removed(str(it.data(QtCore.Qt.ItemDataRole.UserRole))) \
                    else QtCore.Qt.CheckState.Unchecked
                if it.checkState() != want:
                    it.setCheckState(want)
        finally:
            t.blockSignals(False)
            self._syncing = False

    def _on_table_item_changed(self, item: Any) -> None:
        if self._syncing or item is None or item.column() != 0:
            return
        key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if key is None:
            return
        self.decisions.set_removed(str(key), item.checkState() == QtCore.Qt.CheckState.Checked, via="checkbox")
        self._after_edit()

    def _on_table_selection(self) -> None:
        rows = {i.row() for i in self.doubtful_table.selectedIndexes()}
        if not rows:
            self.highlight_item.setData([])
            return
        it = self.doubtful_table.item(min(rows), 0)
        key = None if it is None else it.data(QtCore.Qt.ItemDataRole.UserRole)
        if key is None or str(key) not in self._index:
            self.highlight_item.setData([])
            return
        xy = self._centroids[self._index[str(key)]]
        self.highlight_item.setData([{"pos": (float(xy[0]), float(xy[1]))}])

    # ------------------------------------------------------------ the plot
    def _visible_rings(self) -> Optional[int]:
        data = self.ring_filter.currentData()
        return None if data is None else int(data)

    def _draw_markers(self) -> None:
        only = self._visible_rings()
        spots: Dict[str, List[Dict[str, Any]]] = {cls: [] for cls in DISPLAY_CLASSES}
        for key, i in self._index.items():
            ring = self._ring_of[key]
            if only is not None and ring != only:
                continue
            cls = self.display_class(key)
            role_name, alpha, symbol = LUMEN_CLASS_STYLE[cls]
            xy = self._centroids[i]
            by_hand = cls == DISPLAY_MANUAL_REMOVED
            spots[cls].append({
                "pos": (float(xy[0]), float(xy[1])), "symbol": symbol, "data": key,
                "size": MANUAL_REMOVED_SIZE_PX if by_hand else MARKER_SIZE_PX,
                "brush": pg.mkBrush(self._colour(role_name, alpha)),
                "pen": pg.mkPen(segment_colour(ring), width=MANUAL_REMOVED_PEN_WIDTH if by_hand else MARKER_PEN_WIDTH)})
        # one item per class (the layer panel's rows); a hidden class stays hidden: visibility lives on the item
        for cls in _DRAW_ORDER:
            self.marker_items[cls].setData(spots[cls])
            self.layer_panel.set_count(cls, len(spots[cls]))
        # the tooltip of the NEW markers under the mouse (a click redraws: the old text named the old class)
        self._refresh_hover()

    def _refresh_hover(self) -> None:
        """Recompute the tooltip at the last mouse position (none yet: no tooltip)."""
        self._hover_tip = "\0"      # never a real tip: forces the update below
        if self._hover_pos is not None:
            self._on_mouse_moved(self._hover_pos)
            return
        try:
            self.plot_widget.getViewBox().setToolTip("")
        except RuntimeError:
            pass
        self._hover_tip = ""

    def markers_at(self, pos: QtCore.QPointF) -> List[Any]:
        """The markers drawn at ``pos`` (x', y' in nm) of the VISIBLE classes, topmost first (what a click there
        toggles is the first)."""
        out: List[Any] = []
        for cls in reversed(_DRAW_ORDER):
            item = self.marker_items[cls]
            if not item.isVisible():
                continue
            out.extend(list(item.pointsAt(pos)))
        return out

    def _on_mouse_moved(self, scene_pos: Any) -> None:
        """The tooltip of the markers under the mouse (up to three, topmost first), the plot's only tooltip owner."""
        self._hover_pos = scene_pos
        try:
            vb = self.plot_widget.getViewBox()
            if not vb.sceneBoundingRect().contains(scene_pos):
                tip = ""
            else:
                pts = self.markers_at(vb.mapSceneToView(scene_pos))
                tips = [self._marker_tip(x=pt.pos().x(), y=pt.pos().y(), data=pt.data()) for pt in pts[:3]]
                if len(pts) > 3:
                    tips.append(f"({len(pts) - 3} others...)")
                tip = "\n\n".join(tips)
            if tip != self._hover_tip:
                vb.setToolTip(tip)
                self._hover_tip = tip
        except RuntimeError:      # the window is being destroyed
            return

    def _marker_tip(self, x: float = 0.0, y: float = 0.0, data: Any = None) -> str:
        key = str(data) if data is not None else ""
        if key not in self._index:
            return f"x' {x:.0f} nm, y' {y:.0f} nm"
        c = self.classification
        i = self._index[key]
        reasons = ", ".join(c.reasons[i]) or "none"
        return (f"{key}\nring {self._ring_of[key]}, label {int(c.labels[i]) if len(c.labels) == c.n else '?'}: "
                f"{DISPLAY_LABELS[self.display_class(key)]}\nrule: {c.auto_classes[i]} ({reasons})\n"
                f"hull depth {_table_number(c.hull_depth_nm[i])} nm, hop 250 {_table_number(c.hop_sens_nm[i])} nm, "
                f"hop 300 {_table_number(c.hop_strict_nm[i])} nm\n"
                f"tubulin depth {_table_number(c.depth_tubulin_nm[i])} nm, spectrin depth "
                f"{_table_number(c.depth_spectrin_nm[i])} nm\n"
                f"{int(c.n_locs[i])} locs, {_table_number(c.n_events[i])} events, |z' - ring centre| "
                f"{_table_number(c.abs_dz_ring_nm[i])} nm")

    def _on_points_clicked(self, _item: Any, points: Any, ev: Any = None) -> None:
        pts = list(points) if points is not None else []
        if not pts:
            return
        key = pts[0].data()
        if key is None or str(key) not in self._index:
            return
        self.decisions.toggle(str(key), via="click")
        self._after_edit()
        if ev is not None and hasattr(ev, "accept"):
            ev.accept()

    @staticmethod
    def _path_xy(membrane: Any) -> Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        path = getattr(membrane, "path", None)
        pts = getattr(path, "points_nm", None)
        if pts is None:
            return None
        P = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if P.shape[0] < 2:
            return None
        P = np.vstack([P, P[:1]])
        return P[:, 0], P[:, 1]

    def _draw_membranes(self) -> None:
        before = self._path_xy(self.membrane_before)
        if before is None:
            self.membrane_before_item.setData([], [])
        else:
            self.membrane_before_item.setData(before[0], before[1])
        after = None
        # the membrane after the cleaning does not depend on tau: it stays while results at a new tau are computed
        r = self._tau_free_analyses()
        if r is not None and r.arc_centroid is not None:
            after = self._path_xy(getattr(r.arc_centroid, "centroid_membrane", None))
        if after is None:
            self.membrane_after_item.setData([], [])
        else:
            self.membrane_after_item.setData(after[0], after[1])

    def _fit_membrane_before(self) -> None:
        """The centroid membrane through every cluster, fitted in the background (a drawing aid)."""
        from tools.mps_unroll import centroid_membrane_of_rings

        res = self.res
        relay = self._relay
        gen = self._generation

        def body() -> None:
            try:
                relay.membrane.emit(gen, centroid_membrane_of_rings(res), None)
            except Exception as exc:  # noqa: BLE001 - reported in the warnings list
                relay.membrane.emit(gen, None, exc)

        threading.Thread(target=body, daemon=True, name="lumen-review-membrane").start()

    def _on_membrane(self, gen: int, membrane: Any, error: Any) -> None:
        if gen != self._generation:
            return
        if error is None:
            self.membrane_before = membrane
            try:
                self._draw_membranes()
                return
            except Exception as exc:  # noqa: BLE001 - a slot: reported, never fatal
                self.membrane_before = None
                error = exc
        self.extra_warnings.append(f"the centroid membrane through every cluster could not be fitted: {error}")
        self._fill_warnings()

    def _frame_affine(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(A, b) with (x', y') = A (x, y) + b: the rotation into the axon frame of a laboratory point at the axon's
        mean depth (the frame's origin), i.e. to_axon_frame restricted to that plane."""
        frame = self.res.frame
        if frame.is_identity:
            return np.eye(2), np.zeros(2)
        R = np.asarray(frame.rotation, dtype=np.float64)
        o = np.asarray(frame.origin_nm, dtype=np.float64).reshape(3)
        A = np.asarray(R[:2, :2], dtype=np.float64)
        return A, np.asarray(o[:2] - A @ o[:2], dtype=np.float64)

    def _underlay_transform(self, placement: ImagePlacement, col0: int = 0, row0: int = 0) -> QtGui.QTransform:
        """Image item coordinates (pixel (c, r) of the transposed crop starting at column ``col0``, row ``row0``
        spans [c, c + 1) x [r, r + 1)) to the axon frame: the panel's placement to the laboratory, then the rotation
        into the axon frame projected at the axon's mean depth (the frame's origin)."""
        wf = self.widefield_images
        assert wf is not None
        px = float(wf.pixel_size_nm)
        x0, y0 = placement.lab_origin_nm(px)
        x0 += float(col0) * px
        y0 += float(row0) * px
        A, b = self._frame_affine()
        M = A * px
        t = A @ np.array([x0, y0]) + b
        return QtGui.QTransform(float(M[0, 0]), float(M[1, 0]), float(M[0, 1]), float(M[1, 1]), float(t[0]), float(t[1]))

    def _clusters_lab_nm(self) -> NDArray[np.float64]:
        """The clusters' centroids carried back to the laboratory frame on the plane the underlay is projected on
        (the inverse of ``_frame_affine``)."""
        A, b = self._frame_affine()
        return np.asarray(np.linalg.solve(A, (self._centroids - b).T).T, dtype=np.float64).reshape(-1, 2)

    def _underlay_crop(self, image: NDArray[np.float64], placement: ImagePlacement) -> Tuple[int, int, int, int]:
        """(row0, row1, col0, col1) of the image around the clusters, ``UNDERLAY_MARGIN_NM`` beyond them: a whole
        camera frame is tens of um across, and drawing it would zoom the axon to a speck and set the grey levels by
        the rest of the field."""
        wf = self.widefield_images
        assert wf is not None
        px = float(wf.pixel_size_nm)
        lab = self._clusters_lab_nm()
        col, row = placement.to_pixels(lab[:, 0], lab[:, 1], px)
        m = UNDERLAY_MARGIN_NM / px
        rows, cols = image.shape
        r0 = int(max(0, math.floor(float(np.min(row)) - m)))
        r1 = int(min(rows, math.ceil(float(np.max(row)) + m) + 1))
        c0 = int(max(0, math.floor(float(np.min(col)) - m)))
        c1 = int(min(cols, math.ceil(float(np.max(col)) + m) + 1))
        return r0, r1, c0, c1

    def _draw_underlay(self) -> None:
        wf = self.widefield_images
        which = self.underlay_combo.currentData()
        if wf is None or not self.underlay_check.isChecked() or which is None:
            self.underlay_item.setVisible(False)
            return
        image = wf.tubulin_image if which == "tubulin" else wf.spectrin_image
        placement = wf.tubulin_placement if which == "tubulin" else wf.spectrin_placement
        if image is None or self._centroids.shape[0] == 0:
            self.underlay_item.setVisible(False)
            return
        full = np.asarray(image, dtype=np.float64)
        r0, r1, c0, c1 = self._underlay_crop(full, placement)
        img = full[r0:r1, c0:c1]
        if img.size == 0:
            # The clusters fall outside the image: nothing of it belongs under them.
            self.underlay_item.setVisible(False)
            return
        finite = img[np.isfinite(img)]
        lo, hi = (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.8))) if finite.size else (0.0, 1.0)
        if not hi > lo:
            hi = lo + 1.0
        self.underlay_item.setImage(img.T, levels=(lo, hi), autoLevels=False)
        self.underlay_item.setTransform(self._underlay_transform(placement, col0=c0, row0=r0))
        self.underlay_item.setVisible(True)

    def _fit_view(self) -> None:
        """Frame the clusters (not the underlay, nor a membrane that strays), with a margin. An aspect-locked view
        keeps nm per pixel across a resize, which widens the range as the window is laid out (the Rings panel's
        note), so the range is re-applied on every resize until the user pans or zooms (which emit no resize)."""
        P = self._centroids
        if P.shape[0] == 0:
            return
        lo = P.min(axis=0)
        hi = P.max(axis=0)
        pad = np.maximum(0.08 * (hi - lo), VIEW_MARGIN_NM)
        self._view_target = ((float(lo[0] - pad[0]), float(hi[0] + pad[0])),
                             (float(lo[1] - pad[1]), float(hi[1] + pad[1])))
        self._reapply_view()

    def _reapply_view(self, *_args: Any) -> None:
        target = getattr(self, "_view_target", None)
        if target is None:
            return
        self.plot_widget.getViewBox().setRange(xRange=target[0], yRange=target[1], padding=0.0)

    def _on_user_range(self, *_args: Any) -> None:
        """The user panned or zoomed: their view is kept from now on (a resize no longer re-frames the clusters)."""
        self._view_target = None

    # ------------------------------------------------------------ editing
    def _after_edit(self) -> None:
        self._draw_markers()
        self._sync_table()
        self._refresh_state()
        self._persist()

    def _persist(self) -> bool:
        """Save the decisions in the review store (when there is one); a failure is shown, never raised (a slot's
        caller). True when saved."""
        store = self.review_store
        if store is None:
            return False
        try:
            store.save(self.axon_id, self.decisions)
        except Exception as exc:  # noqa: BLE001 - shown in the warnings list; the review goes on
            first = self._store_error is None
            self._store_error = f"{type(exc).__name__}: {exc}"
            if first:
                self._fill_warnings()
            return False
        if self._store_error is not None:
            self._store_error = None
            self._fill_warnings()
        return True

    def _refresh_state(self) -> None:
        d = self.decisions
        n_restored = sum(1 for k in d.stable_keys if self.display_class(k) == DISPLAY_RESTORED)
        n_hand = sum(1 for k in d.stable_keys if self.display_class(k) == DISPLAY_MANUAL_REMOVED)
        n_db_removed = sum(1 for k in d.doubtful_keys() if d.is_removed(k))
        self.counts_label.setText(
            f"Removed by the rule: <b>{d.n_removed_auto}</b> ({n_restored} restored by hand) &nbsp;|&nbsp; "
            f"doubtful: <b>{d.n_doubtful}</b> ({n_db_removed} removed by hand) &nbsp;|&nbsp; removed by hand: "
            f"<b>{n_hand}</b> &nbsp;|&nbsp; final: <b>{d.n_removed_final}</b> of {len(d.stable_keys)} clusters "
            f"removed, {d.n_manual} differ from the rule; {len(d.edit_log)} edit(s) logged.")
        self.edited_label.setVisible(bool(d.results_shown_before_edit))
        stale = self.last_result is not None and set(self.last_result.removed_keys) != set(d.final_removed_keys())
        self.stale_label.setVisible(bool(stale))
        idle = not self._running
        for b in (self.run_button, self.load_button):
            b.setEnabled(idle)

    def _on_select_all(self) -> None:
        self.decisions.select_all_doubtful()
        self._after_edit()

    def _on_clear_all(self) -> None:
        self.decisions.clear_all_doubtful()
        self._after_edit()

    def _on_reset(self) -> None:
        for k in self.decisions.stable_keys:
            self.decisions.reset(k, via="back to the automatic rule")
        self._after_edit()

    # ------------------------------------------------------------ the re-run
    def _on_run(self) -> None:
        self.start_run()

    def start_run(self, *, live: bool = False) -> bool:
        """Start the re-run in a worker thread; False when one is already running. ``live``: started by a change of
        the criteria switches (H6 toggles), not by the button; such a run is stopped when nobody needs it any more."""
        if self._running:
            return False
        spec = self.selection_widget.spec
        if self.viability is None and spec.exploratory:
            # the selection needs the criteria of both rules: they come with the viability
            self.selection_note.setText("Wait for the ring-pair viability: the exploratory selection needs it.")
            self.selection_note.setVisible(True)
            return False
        snapshot = self.decisions.snapshot()
        fp = msel.lumen_fp(snapshot)
        res, lpz, cp, n_null = self.res, self.lpz_nm, self.columns_params, self.n_null
        # UI stage 1 (D-44): the tau of the control. At tau_0 the window's own parameters go to the analyses exactly as
        # before (None: the pre-registered file); at another tau a copy of them with that tau, in memory only
        run_tau = self._tau_key(self.tau_nm)
        if not self._at_tau0() and self.base_columns_params is not None:
            cp = mtau.params_at_tau(self.base_columns_params, run_tau)
        relay = self._relay
        gen = self._generation
        self._run_seq += 1
        seq = self._run_seq
        vfn, xyz, known = self.viability_fn, self.xyz_lab_nm, self.viability
        window = self

        def report(text: str, fraction: float) -> None:
            # H6 toggles: a live run nobody needs any more stops between two steps of the analyses
            if window._cancel_seq >= seq:
                raise _Superseded()
            relay.progress.emit(gen, str(text), float(fraction))

        def body() -> None:
            t0 = time.perf_counter()
            try:
                # D-41: the pairs are decided on the rings as built (before the cleaning), as the batch does
                report("Ring-pair viability (rule v2)", 0.01)
                viab = known if known is not None else vfn(res, xyz)
                viable, marginal = viab.viable, viab.marginal
                if not viable and not marginal:
                    relay.finished.emit(gen, ReviewRun(viab, None, seconds=time.perf_counter() - t0, fp=fp, live=live,
                                                       tau_nm=run_tau), None)
                    return
                result = run_cleaned_analyses(res, snapshot, columns_params=cp, n_null=n_null, lpz_nm=lpz, progress=report)
                primary = sensitivity = None
                if result.arc_centroid is not None:
                    from batch_columns import restricted_joint
                    if viable:
                        primary = restricted_joint(result.arc_centroid, viable)
                    if marginal:
                        sensitivity = restricted_joint(result.arc_centroid, viable + marginal)
            except _Superseded:
                relay.finished.emit(gen, _SupersededRun(seq), None)
                return
            except Exception as exc:  # noqa: BLE001 - shown in the window
                relay.finished.emit(gen, None, exc)
                return
            relay.finished.emit(gen, ReviewRun(viab, result, primary, sensitivity, time.perf_counter() - t0, fp=fp,
                                               live=live, tau_nm=run_tau), None)

        self._live_armed = True
        self._run_fp = fp
        self._run_live = bool(live)
        self._run_tau = run_tau
        self._cancel_reason = ""        # UI stage 1 (review): a new run starts with no request to stop it
        pending_tau = self.tau_nm if self._tau_pending else None
        if pending_tau is not None:
            # UI stage 1: the results view says which tau is computed (the button pressed after a change of tau, or
            # the live run that change started)
            self.results_view.setPlainText(mtui.pending_live_text(pending_tau))
        if live and pending_tau is not None:
            self.n_live_runs += 1
            self.selection_note.setText(mtui.pending_run_note(pending_tau))
            self.selection_note.setVisible(True)
        elif live:
            self.n_live_runs += 1
            self.selection_note.setText("Live update: computing the analyses of this cluster set in the background "
                                        "(the selection does not change them; later changes re-draw at once)...")
            self.selection_note.setVisible(True)
        self._running = True
        self._run_error = None
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Removing {snapshot.n_removed_final} cluster(s)...")
        self.progress_bar.setVisible(True)
        self._refresh_state()
        self._thread = threading.Thread(target=body, daemon=True, name="lumen-review-rerun")
        self._thread.start()
        return True

    def wait(self, timeout: float = 600.0) -> None:
        """Block until a running re-run is shown (for scripts and tests)."""
        t_end = time.perf_counter() + float(timeout)
        while self._running and time.perf_counter() < t_end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)

    def _on_progress(self, gen: int, text: str, fraction: float) -> None:
        if gen != self._generation or not self._running:
            return
        self.progress_bar.setValue(int(round(100 * max(0.0, min(1.0, float(fraction))))))
        self.progress_bar.setFormat(f"%p%  {text}")

    def _on_run_finished(self, gen: int, result: Any, error: Any) -> None:
        # A slot: an exception escaping it would abort the whole application (PyQt5 without an excepthook).
        if gen != self._generation:
            return
        self._running = False
        self._thread = None
        self.progress_bar.setVisible(False)
        # UI stage 1 (review): why this run was asked to stop, if it was; consumed by every run that ends, so a reason
        # never outlives its run (a run that ended normally after the request leaves none for the next one)
        reason, self._cancel_reason = self._cancel_reason, ""
        if isinstance(result, _SupersededRun):
            # H6 toggles: a live run stopped because nobody needed it any more; the screen already shows the state.
            # UI stage 1: or a run (live or not) stopped because tau changed before it finished
            self._refresh_state()
            self._pending_live = False
            if reason != "tau":
                self.selection_note.setText("A background run was stopped: the selection did not need it any more.")
                self.selection_note.setVisible(True)
            elif self.viability is None:
                self._tau_rerun_wanted = True       # re-rendered when the viability comes (_on_viability)
            self._rerender_for_selection()      # the current selection's state (a new request if it needs one)
            if reason == "tau":
                # UI stage 1: said after the re-render (which hides the selection's notes), until the next run starts
                self.selection_note.setText(mtui.TAU_STOPPED_NOTE)
                self.selection_note.setVisible(True)
            return
        if error is None and result is not None:
            run: ReviewRun = result
            try:
                if run.analyses is not None and run.fp is not None:
                    # the analyses of this cluster set, for every later selection (H6 toggles) and, UI stage 1, at
                    # the tau they ran at
                    self._analyses[self._cache_key(run.fp, run.tau_nm)] = run.analyses
                    while len(self._analyses) > ANALYSES_CACHE_SIZE:
                        self._analyses.pop(next(iter(self._analyses)))
                if not self._tau_is_current(run.tau_nm):
                    # UI stage 1 (D-44): computed at a tau the control has left (the run ended before a step that
                    # could stop it): kept for that tau, never shown, no signal; the current tau's state follows
                    if self.viability is None:
                        self.viability = run.viability
                        self._fill_viability()
                    self._cancel_reason = ""
                    self._pending_live = False
                    self._refresh_state()
                    self._rerender_for_selection(tau_change=True)
                    return
                if self.viability is not None and run.viability is not self.viability:
                    # the selection changed while the run was computing: show the CURRENT selection's tiers
                    run = self._run_under_current(run)
                self.last_run = run
                if self.viability is None:
                    self.viability = run.viability
                    self._fill_viability()
                if self._run_live or run.live:
                    self.selection_note.setVisible(False)
                self.last_result = run.analyses
                if run.analyses is None:
                    # D-41: no viable or marginal pair -- nothing was computed, and nothing is shown as a result
                    self.results_view.setPlainText(self._not_run_lines(run))
                    self.leak_banner.setVisible(False)
                else:
                    self._show_results(run.analyses)
                    a = run.analyses
                    if any(t is not None for t in (a.arc_centroid, a.columns_2d, a.arc_localization)):
                        # A column result of this axon is now on screen: every later edit is flagged (D-35b), in
                        # this window and, through the review store, in every later review of this axon.
                        self.decisions.mark_results_shown()
                        self._persist()
                self._clear_tau_pending()       # UI stage 1: what is on screen is at the current tau
                self._draw_membranes()
                self._refresh_state()
                self._fill_warnings()
                self._selection_texts(run)
                self._refresh_tau_views()
            except Exception as exc:  # noqa: BLE001 - reported in the window
                error = exc
            else:
                self.run_finished.emit(run)
                if run.analyses is not None:
                    self.analysis_finished.emit(run.analyses)
                if self._pending_live:
                    self._pending_live = False
                    self._rerender_for_selection()
                return
        self._run_error = f"{type(error).__name__}: {error}" if error is not None else "no result"
        self.results_view.setPlainText(marked("bad", f"The re-run failed: {self._run_error}\n\n"
                                                     "The previous results, if any, are no longer shown."))
        self.last_result = None
        self.last_run = None
        self._clear_tau_pending()
        self.leak_banner.setVisible(False)
        self._selection_texts(None)
        self._draw_membranes()
        self._refresh_state()
        self._fill_warnings()
        self._refresh_tau_views()

    def _show_results(self, r: CleanedAnalyses) -> None:
        self.results_view.setPlainText(self._results_lines(r, self.last_run))
        self.leak_banner.setVisible(True)

    @staticmethod
    def _verdict_lines(v: ReviewViability) -> List[str]:
        lines = [f"Ring-pair viability ({v.rule})"]
        for p in v.pairs:
            word = {"viable": marked("good", "VIABLE"), "marginal": marked("warn", "MARGINAL (sensitivity only)")}.get(
                str(p.verdict), marked("bad", "NOT VIABLE"))
            lines.append(f"   pair {p.ring_a}-{p.ring_b}: {word}" + (f" -- {'; '.join(p.reasons)}" if p.reasons else ""))
        if not v.pairs:
            lines.append("   no consecutive ring pair (fewer than two rings)")
        return lines

    def _not_run_lines(self, run: ReviewRun) -> str:
        when = datetime.now().strftime("%H:%M:%S")
        spec = getattr(run.viability, "spec", None)
        if spec is not None and not spec.is_default:
            # H6 toggles: the first line names the exploratory selection
            lines = [exploratory_banner(spec), "",
                     marked("bad", NO_SELECTED_PAIR_TEXT.format(label=spec.label, hash=spec.hash)), ""]
        else:
            lines = [marked("bad", NO_VIABLE_PAIR_TEXT), ""]
        lines.extend(self._verdict_lines(run.viability))
        lines.append("")
        lines.append(f"(checked {when}, {run.seconds:.1f} s; the rings as built, before the lumen cleaning.)")
        lines.append("What limits the z quality of this axon, with the numbers and what would be needed: press "
                     "'Z quality...'.")
        return "\n".join(lines)

    @staticmethod
    def _tier_lines(name: str, tier: Optional[Dict[str, Any]], pairs: Sequence[Tuple[int, int]]) -> List[str]:
        text = " ".join(f"{a}-{b}" for a, b in pairs)
        if tier is None or not tier.get("n_pairs"):
            return [f"   {name} ({text}): not in the arc test (a ring or its contour was lost in the cleaning)"]
        return [f"   {name}, pairs {tier.get('pairs')}: T = {tier.get('T_obs')} matched (null mean "
                f"{_fmt(tier.get('null_mean'), '.2f')}), z_A = {_fmt(tier.get('z_A'))}, p_excess = "
                f"{_fmt(tier.get('p_excess_uncalibrated'), '.4f')} ({NOT_CALIBRATED}), two-sided p = "
                f"{_fmt(tier.get('p_two_sided_uncalibrated'), '.4f')} ({NOT_CALIBRATED})"]

    def _results_lines(self, r: CleanedAnalyses, run: Optional[ReviewRun] = None) -> str:
        from tools.mps_unroll import CENTROID_MEMBRANE_RECIPE

        lines: List[str] = []
        when = datetime.now().strftime("%H:%M:%S")
        spec = None if run is None else getattr(run.viability, "spec", None)
        exploratory = spec is not None and not spec.is_default
        variant = self._tau_variant(run)
        if variant is not None:
            # UI stage 1 (D-44): at another tau than tau_0 the first line names the analysis (selection and tau)
            lines.append(mtau.tau_banner(variant))
        elif exploratory:
            # H6 toggles (D-43): the first line of a result under an exploratory selection says so
            lines.append(exploratory_banner(spec))
        lines.append(f"Results on the cleaned rings (run finished {when}, {r.seconds:.1f} s, null size {r.n_null})")
        lines.append(f"!  Every statistic here is {LEAK_NOT_CALIBRATED_NOTE}; {NOT_CALIBRATED_DETAIL}.")
        if r.edited_after_results_shown:
            lines.append("!  This set was edited after results were shown (D-35b).")
        lines.append("")
        if run is not None:
            lines.extend(self._verdict_lines(run.viability))
            lines.append("")
            selected = set(run.viability.viable) | set(run.viability.marginal)
            primary, marginal = run.viability.viable, run.viability.marginal
        else:  # a caller without a run: every pair (the H5-D behaviour)
            selected = {(int(p.ring_a), int(p.ring_b)) for p in (r.arc_centroid.adjacent if r.arc_centroid else [])}
            primary, marginal = sorted(selected), []
        lines.append(f"Cleaning ({r.rule_version}{'' if r.has_widefield else ', isolation only: no widefield'}):")
        lines.append(f"   {r.n_removed_auto} removed by the rule, {r.n_doubtful} doubtful, {r.n_manual} changed by "
                     f"hand; {r.n_removed_final} of {len(self.decisions.stable_keys)} clusters removed in total.")
        lines.append(f"   Clusters per ring: before {_k_text(self.res)}, after {_k_text(r.rings)}.")
        if r.n_manual:
            lines.append(f"   Note: {MANUAL_NULL_NOTE}.")
        lines.append("")

        def joint_lines(test: Any) -> List[str]:
            # D-41 (review B1, N8): a test's joint over all of its pairs only when every one of them is VIABLE -- the
            # rule results_row applies to the export; the numbers are batch_columns.restricted_joint over those pairs
            # (= the test's own z_A and p_A, its two-sided p)
            keys = [(int(q.ring_a), int(q.ring_b)) for q in test.adjacent]
            if run is None:
                return ["   (its joint z_A is shown only under rule v2, when every pair is VIABLE)"]
            if not keys or not set(keys) <= set(primary):
                return [f"   (its joint z_A over every pair is shown only when all of them are VIABLE; here "
                        f"{'no pair is' if not keys else 'not all are'})"]
            from batch_columns import restricted_joint
            return self._tier_lines("JOINT: every pair VIABLE, joint z_A", restricted_joint(test, keys), keys)

        def pair_lines(adjacent: Sequence[Any]) -> List[str]:
            out = []
            for q in adjacent:
                key = (int(q.ring_a), int(q.ring_b))
                if key not in selected:
                    continue
                tag = "" if key in primary else "  [marginal: sensitivity only]"
                out.append(f"   pair {q.ring_a}-{q.ring_b}: zeta {_fmt(q.zeta)}, p_excess {_fmt(q.p_excess, '.4f')} "
                           f"({NOT_CALIBRATED}), {int(q.n_matched)} matched of K {int(q.K_a)} / {int(q.K_b)}{tag}")
            return out or ["   (none of the selected pairs is in this test)"]

        ac = r.arc_centroid
        lines.append(f"Arc test on the centroid membrane ({CENTROID_MEMBRANE_RECIPE}) -- PRIMARY (D-35c), "
                     "VIABLE pairs only")
        if ac is None:
            lines.append("   failed (see the warnings)")
        else:
            if run is not None:
                if primary:
                    lines.extend(self._tier_lines("PRIMARY: viable pairs, joint z_A", run.primary, primary))
                else:
                    lines.append("   PRIMARY: no viable pair (only marginal ones: see the sensitivity below)")
                if marginal:
                    lines.extend(self._tier_lines("SENSITIVITY: viable + marginal pairs, joint z_A", run.sensitivity,
                                                  primary + marginal))
            else:
                lines.append(f"   z_A = {_fmt(ac.z_A)}   p_A = {_fmt(ac.p_A, '.4f')} ({NOT_CALIBRATED})")
            lines.append(f"   membrane length {_fmt(ac.length_nm, ',.0f')} nm, knots {_fmt(ac.membrane_knot_spacing_nm, '.0f')} nm,"
                         f" tau {_fmt(ac.tau_nm, '.1f')} nm, rings {ac.rings}")
            lines.extend(pair_lines(ac.adjacent))
        lines.append("")
        cols = r.columns_2d
        lines.append("2D test (pre-registered, D-24) -- SENSITIVITY (D-29a), selected pairs only")
        if cols is None:
            lines.append("   failed (see the warnings)")
        else:
            if variant is not None and self.tau0_nm is not None:
                # UI stage 1 (D-44): the 2D test ran at the chosen tau, not at tau_0
                lines.append(mtau.tau_2d_line(cols.tau0_nm, self.tau0_nm))
            else:
                lines.append(f"   tau_0 {_fmt(cols.tau0_nm, '.1f')} nm")
            lines.extend(joint_lines(cols))
            lines.extend(pair_lines(cols.adjacent))
        lines.append("")
        al = r.arc_localization
        lines.append("Arc test on the localization membrane -- DIAGNOSTIC only, selected pairs only")
        lines.append(f"   ({r.arc_localization_note or 'diagnostic only (D-35c), known conservative bias (D-34b)'})")
        if al is None:
            lines.append("   failed (see the warnings)")
        else:
            lines.extend(joint_lines(al))
            lines.extend(pair_lines(al.adjacent))
        if run is not None:
            hidden = [p for p in run.viability.pairs if str(p.verdict) not in ("viable", "marginal")]
            if hidden:
                lines.append("")
                under = "the exploratory selection" if exploratory else "rule v2"
                lines.append(f"Not analysed (NOT VIABLE under {under}): "
                             + ", ".join(f"{p.ring_a}-{p.ring_b}" for p in hidden) + ". Their statistics are not shown.")
        notes = [w for w in r.warnings if w.startswith(("cleaned ring", "arc test (", "2D test: "))]
        if notes:
            lines.append("")
            lines.append(f"Notes of the cleaning and the analyses ({len(notes)}; all in the warnings list):")
            lines.extend(f"   - {w}" for w in notes[:6])
            if len(notes) > 6:
                lines.append(f"   ... {len(notes) - 6} more in the warnings list")
        return "\n".join(lines)

    # ------------------------------------------------------------ the tolerance tau (UI stage 1, D-44)
    def _tau_key(self, tau: Optional[float]) -> Optional[float]:
        """A tau as the window holds it: tau_0 itself (None counts as tau_0: a run built outside the window), else
        rounded to 0.01 nm (``tools.mps_tau.normalize_tau``); None without the control."""
        t0 = self.tau0_nm
        if t0 is None:
            return None
        if tau is None or mtau.is_tau0(tau, t0):
            return float(t0)
        return mtau.normalize_tau(tau)

    def _tau_is_current(self, tau: Optional[float]) -> bool:
        """A run's tau is the control's (always, without the control: every run is then at tau_0)."""
        return self._tau_key(tau) == self._tau_key(self.tau_nm)

    def _at_tau0(self) -> bool:
        """The control is at the pre-registered tau_0 (or there is no control)."""
        return self.tau0_nm is None or self.tau_nm is None or mtau.is_tau0(self.tau_nm, self.tau0_nm)

    def _cache_key(self, fp: Any, tau: Optional[float]) -> Tuple[Any, Optional[float]]:
        """The key of the analyses kept per cluster set (lumen fingerprint) and tau."""
        return fp, self._tau_key(tau)

    def _variant(self, spec: Any, tau: Optional[float]) -> Optional[mtau.AnalysisVariant]:
        """The analysis variant (``tools.mps_tau.AnalysisVariant``) of a selection at a tau; None without the
        control."""
        if self.tau0_nm is None or not isinstance(spec, msel.SelectionSpec):
            return None
        key = self._tau_key(tau)
        try:
            return mtau.AnalysisVariant(spec, float(self.tau0_nm if key is None else key), self.tau0_nm)
        except ValueError:
            return None

    def _tau_variant(self, run: Optional[ReviewRun]) -> Optional[mtau.AnalysisVariant]:
        """The variant of a run computed at another tau than tau_0 (its selection: the run's own, the pre-specified
        rule when it has none); None at tau_0, without a run or without the control (the texts are then the
        pre-registered ones)."""
        if run is None:
            return None
        spec = getattr(run.viability, "spec", None)
        v = self._variant(spec if spec is not None else msel.DEFAULT_SELECTION, run.tau_nm)
        return v if v is not None and not v.tau_is_default else None

    def _on_tau_changed(self, tau: float) -> None:
        # A slot of the control: never raises (an exception escaping a slot aborts the application)
        try:
            self.apply_tau(tau)
        except RuntimeError:        # the window is being destroyed
            return
        except Exception as exc:  # noqa: BLE001 - shown
            self.selection_note.setText(marked("bad", f"tau could not be applied: {type(exc).__name__}: {exc}"))
            self.selection_note.setVisible(True)

    def apply_tau(self, tau: float) -> None:
        """
        The control moved to ``tau`` (UI stage 1, D-44). At once: the simulated-null button (tau_0 only) and the
        curve's marker. A run in flight at another tau is stopped (between two steps of the analyses: its result is
        never shown). After the first run of the window: the results of this cluster set at the new tau re-drawn when
        they were computed before; otherwise no p of another tau stays on screen, and with 'Live update' a run starts
        after the usual wait (quick changes give one run, at the last value). Before the first run only the control
        changes: the button then runs at it.
        """
        if self.tau0_nm is None:
            return
        new = self._tau_key(tau)
        if new == self.tau_nm:
            return
        if not self._tau_touched and not mtau.is_tau0(new, self.tau0_nm):
            # UI stage 1 (review): the control leaves tau_0 for the first time. Without the criteria of the selection
            # (switches off) the window logged and counted nothing, as before this stage; from now on it does, so the
            # pre-registered result on screen is logged first and counted with the tau variants that follow
            self._tau_touched = True
            if self.criteria is None and self.last_run is not None and self.last_run.analyses is not None:
                self._selection_texts(self.last_run)
        self.tau_nm = new
        self._refresh_tools()
        self._update_curve_marker()
        if self._running and not self._tau_is_current(self._run_tau) and self._cancel_seq < self._run_seq:
            self._cancel_seq = self._run_seq        # stopped between two steps; reported as superseded
            self._cancel_reason = "tau"
        if not self._live_armed:
            return
        self._rerender_for_selection(tau_change=True)

    def _clear_tau_pending(self) -> None:
        self._tau_pending = False
        self._pending_keep = None

    def _tau_free_analyses(self) -> Optional[CleanedAnalyses]:
        """The analyses of the cluster set on screen for what does not depend on tau (the E(tau) curve, the membrane
        after the cleaning): the results on screen, or, while the results at a new tau are pending, the ones they
        replace."""
        if self.last_result is not None:
            return self.last_result
        return self._pending_keep if self._tau_pending else None

    def _show_tau_pending(self, fp: Any) -> None:
        """No analyses of this cluster set at the current tau yet: the results view says they are being computed (or
        that the button computes them); no p of another tau stays on screen -- the results, the leak banner, the
        counter (the analysis' head only) and the matches on the map; the curve and the membrane, which do not depend
        on tau, stay. The export then writes the decisions only."""
        tau = self.tau_nm
        if tau is None or self.viability is None:
            return
        if self.last_result is not None:
            self._pending_keep = self.last_result
        self._tau_pending = True
        prev = self.last_run
        self.last_run = ReviewRun(self.viability, None, seconds=0.0 if prev is None else prev.seconds, fp=fp,
                                  tau_nm=self._tau_key(tau))
        self.last_result = None
        running_here = self._running and self._tau_is_current(self._run_tau) and self._cancel_seq < self._run_seq
        computing = running_here or self.live_update_check.isChecked()
        self.results_view.setPlainText(mtui.pending_live_text(tau) if computing else mtui.pending_press_text(tau))
        self.leak_banner.setVisible(False)
        self._selection_texts(self.last_run)
        self._draw_membranes()
        self._refresh_state()
        self._fill_warnings()       # UI stage 1 (review): the 'run:' notes of another tau leave with its results
        self._refresh_tau_views()

    def is_tau_pending(self) -> bool:
        """True while the results at the current tau are not on screen yet (being computed, or waiting for the
        button)."""
        return bool(self._tau_pending)

    # ------------------------------------------------------------ the E(tau) curve and the matches (UI stage 1)
    def _curve_pair(self) -> Optional[Tuple[int, int]]:
        data = self.curve_pair_combo.currentData()
        if data is None:
            return None
        try:
            return int(data[0]), int(data[1])
        except (TypeError, ValueError, IndexError):
            return None

    def _fill_curve_pairs(self) -> None:
        """The curve's pair combo: the pairs the selection selects (VIABLE first, then MARGINAL); the pair shown stays
        when it is still selected."""
        v = self.viability
        combo = self.curve_pair_combo
        current = self._curve_pair()
        entries = mtui.curve_pairs(v.viable, v.marginal) if v is not None else []
        combo.blockSignals(True)
        try:
            combo.clear()
            for text, pair in entries:
                combo.addItem(text, pair)
            index = next((i for i, (_t, p) in enumerate(entries) if p == current), 0 if entries else -1)
            combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(False)
        combo.setEnabled(bool(entries))
        self._refresh_tau_views()

    def _on_curve_pair(self, _index: int = 0) -> None:
        # a slot: never raises
        try:
            self._refresh_tau_views()
        except RuntimeError:        # the window is being destroyed
            return

    def _refresh_tau_views(self) -> None:
        """The curve of the chosen pair, its tau marker and the matches on the map, from what is on screen. A drawing:
        never raises (a failure is said under the curve)."""
        try:
            self._draw_curve()
            self._update_curve_marker()
            self._draw_matches()
        except RuntimeError:        # the window is being destroyed
            return
        except Exception as exc:  # noqa: BLE001 - a drawing aid; said, never fatal
            self.curve_p_label.setText(marked("bad", f"The curve could not be drawn: {type(exc).__name__}: {exc}"))

    def _draw_curve(self) -> None:
        pair = self._curve_pair()
        a = self._tau_free_analyses()
        cv: Optional[mtau.CurveView] = None
        if pair is None:
            text = mtui.CURVE_NO_PAIR_TEXT if self.viability is not None else mtui.CURVE_NO_RUN_TEXT
        elif a is None:
            text = mtui.CURVE_NO_RUN_TEXT
        elif a.columns_2d is None:
            text = mtui.CURVE_2D_FAILED_TEXT
        else:
            cv = mtau.curve_view(a, pair)
            text = mtui.curve_pair_missing_text(*pair) if cv is None else ""
        self.curve_shown = cv
        if cv is None:
            self.curve_observed_item.setData([], [])
            self.curve_mean_item.setData([], [])
            self.curve_band_lo.setData([], [])
            self.curve_band_hi.setData([], [])
            self.curve_p_label.setText(text)
            self.curve_counter_label.setVisible(False)
            return
        x = cv.tau_grid_nm
        self.curve_observed_item.setData(x, cv.E_dir_obs)
        self.curve_mean_item.setData(x, cv.E_star)
        self.curve_band_lo.setData(x, cv.envelope_lo)
        self.curve_band_hi.setData(x, cv.envelope_hi)
        v = self.viability
        marginal = v is not None and pair in set(v.marginal) and pair not in set(v.viable)
        spec = self.selection_widget.spec
        sel_hash = spec.hash if self.criteria is not None and spec.exploratory else ""
        self.curve_p_label.setText(mtui.curve_p_text(cv, marginal=marginal, selection_hash=sel_hash))
        # the counter next to every p (D-43): the side column's, as last counted when a p was shown
        self.curve_counter_label.setText(self._counter_line_text)
        self.curve_counter_label.setVisible(bool(self._counter_line_text) and cv.has_global_test)

    def _update_curve_marker(self) -> None:
        """tau_0 and the chosen tau on the curve's axis (the chosen one hidden at tau_0); the x range covers the grid
        and both."""
        t0, tau = self.tau0_nm, self.tau_nm
        if t0 is None or tau is None:
            self.curve_tau0_line.setVisible(False)
            self.curve_tau_line.setVisible(False)
            return
        self.curve_tau0_line.setPos(float(t0))
        self.curve_tau_line.setPos(float(tau))
        self.curve_tau_line.setVisible(self.curve_layers.is_visible("tau") and not self._at_tau0())
        cp = self.base_columns_params
        grid = [float(g) for g in (cp.tau_grid_nm if cp is not None else ())]
        lo, hi = min(grid + [float(t0), float(tau)]), max(grid + [float(t0), float(tau)])
        pad = 0.03 * max(hi - lo, 1.0)
        self.curve_plot.setXRange(lo - pad, hi + pad, padding=0.0)
        self.curve_plot.setYRange(0.0, 1.0, padding=0.03)

    def _draw_matches(self) -> None:
        """The clusters each test matched for the curve's pair, from the analyses on screen (at the current tau);
        empty and greyed while the results at the current tau are not on screen."""
        a = self.last_result
        pair = self._curve_pair()
        shown = a is not None and pair is not None and not self._tau_pending
        for key, test in (("matches_arc", "arc"), ("matches_2d", "2d")):
            item = self.match_items[key]
            seg = np.zeros((0, 2, 2), dtype=np.float64)
            if shown and pair is not None:
                try:
                    seg = mtau.match_segments(a, pair, test)
                except ValueError:
                    seg = np.zeros((0, 2, 2), dtype=np.float64)
            if seg.shape[0]:
                xy = seg.reshape(-1, 2)
                item.setData(xy[:, 0], xy[:, 1], connect="pairs")
            else:
                item.setData([], [])
            self.layer_panel.set_count(key, int(seg.shape[0]) if shown else None)
            self.layer_panel.set_enabled(key, shown, mtui.NO_RESULTS_AT_TAU)

    def match_counts(self) -> Dict[str, int]:
        """How many segments each match layer draws now (for scripts and tests)."""
        out: Dict[str, int] = {}
        for key, item in self.match_items.items():
            x, _y = item.getData()
            out[key] = 0 if x is None else int(len(x)) // 2
        return out

    # ------------------------------------------------------------ rule v2 and the tools beside the run
    def _compute_viability(self) -> None:
        """Rule v2 on the rings as built, in a worker (fills the viability table and enables the buttons); then the
        per-pair criteria of both rules (H6 toggles: rule v2's numbers and rule v2c's, geometry only), which the
        criteria switches evaluate without recomputing anything."""
        self._viability_generation += 1
        gen, relay, fn, res, xyz = self._viability_generation, self._relay, self.viability_fn, self.res, self.xyz_lab_nm

        def body() -> None:
            try:
                v = fn(res, xyz)
            except Exception as exc:  # noqa: BLE001 - shown in the window
                relay.viability.emit(gen, None, exc)
                return
            crit, err = None, ""
            detail = getattr(v, "detail", None)
            if detail is not None and hasattr(detail, "pairs") and hasattr(detail, "rings"):
                try:
                    from tools.mps_zquality_window import lab_xyz_of
                    lab = xyz if xyz is not None else lab_xyz_of(res)
                    crit = msel.compute_axon_criteria(res, lab, v2=detail, axon_id=str(getattr(res, "source_name", "")))
                except Exception as exc:  # noqa: BLE001 - the switches stay off, with the reason
                    crit, err = None, f"{type(exc).__name__}: {exc}"
            else:
                err = "custom viability function: toggles off"
            # the criteria first (queued signals keep their order): the viability is shown under the selection at once
            relay.criteria.emit(gen, crit, err)
            relay.viability.emit(gen, v, None)

        threading.Thread(target=body, daemon=True, name="review-viability").start()

    def wait_viability(self, timeout: float = 120.0) -> None:
        """Block until the viability is shown (for scripts and tests)."""
        t_end = time.perf_counter() + float(timeout)
        while self.viability is None and self.viability_error is None and time.perf_counter() < t_end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)

    def _on_criteria(self, gen: int, crit: Any, error: Any) -> None:
        if gen != self._viability_generation:
            return
        self.criteria = crit
        self.criteria_error = str(error or "")
        sw = self.selection_widget
        if crit is None:
            sw.set_available(False, self.criteria_error or "no criteria")
            return
        sw.set_available(True)
        if not crit.has_v2c:
            sw.enable_v2c(False, f"rule v2c could not be computed for this axon ({crit.v2c_error})")
        else:
            sw.enable_v2c(True)

    def _on_viability(self, gen: int, v: Any, error: Any) -> None:
        if gen != self._viability_generation:
            return
        if error is not None or v is None:
            self.viability_error = f"{type(error).__name__}: {error}" if error is not None else "no result"
            self.viability_label.setText(marked("bad", f"Ring-pair viability could not be computed: "
                                                       f"{self.viability_error}"))
            self._refresh_tools()
            return
        self.base_viability = v
        self.viability = self._selected_viability()
        self._fill_viability()
        if self._tau_rerun_wanted and not self._running:
            # UI stage 1: the first run was stopped by a change of tau before the viability was known: the run at
            # the current tau follows now
            self._tau_rerun_wanted = False
            self._rerender_for_selection(tau_change=True)

    # ------------------------------------------------------------ the criteria switches (H6 toggles, D-43)
    def _selected_viability(self) -> Optional[ReviewViability]:
        """The viability under the program's selection: rule v2's own object under the pre-specified rule (so every
        text and export is exactly what it was), the selection's verdicts otherwise."""
        base = self.base_viability
        if base is None:
            return None
        spec = self.selection_widget.spec
        crit = self.criteria
        if crit is None:
            self.selection_result = None
            self.selection_widget.set_summary("")
            return base
        result = msel.evaluate_selection(crit, spec)
        self.selection_result = result
        self.selection_widget.set_summary("this axon: " + result.summary())
        if spec.is_default:
            return base
        chosen: ReviewViability = msel.selection_viability(base, result)
        return chosen

    def _on_selection(self, _spec: Any = None) -> None:
        # a slot of the program-wide selection: never raises (it would abort the application)
        if self._closed:
            return
        try:
            self.apply_selection()
        except RuntimeError:        # the window is being destroyed
            return
        except Exception as exc:  # noqa: BLE001 - shown
            self.selection_note.setText(marked("bad", f"The selection could not be applied: {type(exc).__name__}: "
                                                      f"{exc}"))
            self.selection_note.setVisible(True)

    def apply_selection(self) -> None:
        """The selection changed: the viability table at once (microseconds), then the results -- re-drawn from the
        analyses of the cluster set when it has them, or a live run in the background when it does not."""
        if self.base_viability is None or self.criteria is None:
            return
        self.viability = self._selected_viability()
        self._fill_viability()
        self._rerender_for_selection()

    def _run_under_current(self, run: ReviewRun) -> ReviewRun:
        """``run``'s analyses under the CURRENT selection: its tiers recomputed (``restricted_joint``, sub-ms); its tau
        is kept (UI stage 1)."""
        v = self.viability if self.viability is not None else run.viability
        if run.analyses is None or not (v.viable or v.marginal):
            return ReviewRun(v, None, seconds=run.seconds, fp=run.fp, live=run.live, tau_nm=run.tau_nm)
        primary, sensitivity = msel.selection_tiers(run.analyses.arc_centroid, v)
        return ReviewRun(v, run.analyses, primary, sensitivity, run.seconds, fp=run.fp, live=run.live,
                         tau_nm=run.tau_nm)

    def _rerender_for_selection(self, tau_change: bool = False) -> None:
        """After a change of the selection (or of 'Live update', or -- UI stage 1, ``tau_change`` -- of tau): nothing
        before the first run of this window; then the results of the analysed cluster set re-drawn under the new
        selection at the current tau, or 'nothing selected', or a live run when the cluster set was never analysed
        at this tau. A change of tau never leaves a p of another tau on screen: until the results at the new tau are
        there, the window says they are being computed (or that the button computes them)."""
        if self._closed or not self._live_armed or self.viability is None:
            return
        # UI stage 1: a change of tau re-renders whether or not the criteria switches are on (a custom viability
        # function turns the switches off, never the tau control)
        if self.criteria is None and not (tau_change or self._tau_pending):
            return
        v = self.viability
        run = self.last_run
        fp = run.fp if run is not None and run.fp is not None else msel.lumen_fp(self.decisions)
        tau = self._tau_key(self.tau_nm)
        analyses = self._analyses.get(self._cache_key(fp, tau))
        if analyses is None and run is not None and run.analyses is not None and self._tau_is_current(run.tau_nm):
            analyses = run.analyses
        if not v.viable and not v.marginal:
            if tau_change:
                # UI stage 1: no pair is selected, so nothing on screen depends on tau (nothing is computed)
                return
            self._debounce.stop()
            if self._running and self._run_live:
                self._cancel_seq = self._run_seq        # nobody needs that run any more
            # (final review F1) at once, even with a run in flight: no p of the previous selection stays on screen
            # (a button run still ends, and is then shown under the selection current at that moment)
            new = ReviewRun(v, None, seconds=0.0 if run is None else run.seconds, fp=fp, tau_nm=tau)
            self.last_run = new
            self.last_result = None
            self._clear_tau_pending()
            self.results_view.setPlainText(self._not_run_lines(new))
            self.leak_banner.setVisible(False)
            self._draw_membranes()
            self._refresh_state()
            self._selection_texts(new)
            self._refresh_tau_views()
            return
        if analyses is not None:
            # (final review F1) also while a button run is in flight: the analyses on screen re-drawn under the new
            # selection at once (they do not depend on it); the run is shown under the selection current when it ends.
            # UI stage 1: the analyses of this cluster set at the current tau (a tau already computed comes back at
            # once)
            self._debounce.stop()
            base = run if run is not None else ReviewRun(v, analyses, fp=fp)
            new = self._run_under_current(ReviewRun(base.viability, analyses, seconds=base.seconds, fp=fp,
                                                    tau_nm=tau))
            shown_before = self.last_result
            self.last_run = new
            self.last_result = analyses
            self._clear_tau_pending()
            self._show_results(analyses)
            if analyses is not shown_before:
                # UI stage 1 (review): other analyses than the ones on screen (another tau): the warnings list
                # follows them, so no note of another tau stays beside these results
                self._fill_warnings()
            if not self.decisions.results_shown and any(
                    t is not None for t in (analyses.arc_centroid, analyses.columns_2d, analyses.arc_localization)):
                # UI stage 1 (review): analyses kept from a run that ended at a tau the control had left were never
                # shown; shown now, they flag every later edit (D-35b) exactly as _on_run_finished does
                self.decisions.mark_results_shown()
                self._persist()
            self._draw_membranes()
            self._refresh_state()
            self._selection_texts(new)
            self.selection_note.setVisible(False)
            self._refresh_tau_views()
            return
        if tau_change or self._tau_pending:
            # UI stage 1: this cluster set has no analyses at this tau yet: no p of another tau stays on screen
            self._show_tau_pending(fp)
        if self._running:
            # the run in flight analyses every pair: it is shown under the selection current when it finishes (UI
            # stage 1: a run at another tau was stopped by the change of tau, and its report comes back here)
            if self._run_live and self._run_fp != msel.lumen_fp(self.decisions):
                self._pending_live = True
            return
        if self.live_update_check.isChecked():
            if self._tau_pending:
                self.selection_note.setVisible(False)      # the results view says what is coming
            else:
                self.selection_note.setText(f"Live update: the analyses of this cluster set start in "
                                            f"{LIVE_DEBOUNCE_MS / 1000:.1f} s (a further change restarts the wait)...")
                self.selection_note.setVisible(True)
            self._debounce.start()
        elif self._tau_pending:
            self.selection_note.setVisible(False)          # the results view says to press the button
        else:
            self.selection_note.setText("This selection needs the analyses of the cluster set, which the last run did "
                                        "not compute: press 'Apply and re-run analyses' (or tick 'Live update').")
            self.selection_note.setVisible(True)

    def _request_live_run(self) -> None:
        """The debounce elapsed: start the live run (or leave a request for when the running one ends)."""
        if self._closed:
            return
        if self._running:
            self._pending_live = True
            return
        v = self.viability
        if v is None or not (v.viable or v.marginal):
            return
        if not self.start_run(live=True):
            self._pending_live = True

    def _log_path(self) -> str:
        store = self.review_store
        return app_log_path(getattr(store, "directory", None) if store is not None else None)

    def _selection_texts(self, run: Optional[ReviewRun]) -> None:
        """The banner of an exploratory selection and the counter next to the p values; logs the selection the p
        values on screen were computed under (once per axon, cluster set and selection in this session)."""
        spec = self.selection_widget.spec
        # UI stage 1 (review): without the criteria (switches off) nothing is logged or counted, as before this stage,
        # until the tau control leaves tau_0; from then on every p shown is, under the selection the window applies
        # (rule v2's own: the switches are off), so that no p of another tau goes unlogged or uncounted
        counting = self.criteria is not None or self._tau_touched
        shows_p = run is not None and run.analyses is not None and counting
        # short: the results' first line carries the whole sentence
        exploratory = run is not None and self.criteria is not None and spec.exploratory
        # UI stage 1 (D-44): at another tau than tau_0 the head names the analysis (selection and tau) while its
        # results are on screen or being computed; otherwise the selection's own head, as before
        tau_variant = (self._tau_variant(run)
                       if run is not None and (run.analyses is not None or self._tau_pending) else None)
        if tau_variant is not None:
            head = mtau.counter_banner_text(tau_variant)
            self.selection_banner_text = mtau.tau_banner(tau_variant)
        else:
            head = f"EXPLORATORY SELECTION {spec.label} #{spec.hash}" if exploratory else ""
            self.selection_banner_text = exploratory_banner(spec) if exploratory else ""
        banner = f"<b style='color:{verdict('bad', dark=False)}'>{head}</b>" if head else ""
        if not shows_p or run is None:
            if banner:
                self.selection_counter_label.setText(banner)
                self.selection_counter_label.setVisible(True)
                return
            self.selection_counter_label.setVisible(False)
            return
        try:
            if self.exploration_log is None:
                self.exploration_log = msel.ExplorationLog(self._log_path())
            log = self.exploration_log
            a = run.analyses
            v = run.viability
            selected = set(v.viable) | set(v.marginal)
            pair_p = msel.pair_p_text(
                (int(q.ring_a), int(q.ring_b), getattr(q, "p_excess", None))
                for q in (a.arc_centroid.adjacent if a is not None and a.arc_centroid is not None else [])
                if (int(q.ring_a), int(q.ring_b)) in selected)
            lumen = msel.lumen_fp_text(run.fp) if run.fp is not None else ""
            kw: Dict[str, Any] = dict(source="review", axon_id=self.axon_id, lumen=lumen, viable=v.viable,
                                      marginal=v.marginal, primary=run.primary, sensitivity=run.sensitivity,
                                      pair_p=pair_p, n_null=None if a is None else a.n_null, git_head=git_head())
            # UI stage 1: the row of the analysis variant (the selection at the run's tau; at tau_0 the selection's
            # own row exactly), so that every (selection, tau) a p was shown under is logged once and counted
            applied = spec if self.criteria is not None else (getattr(v, "spec", None) or msel.DEFAULT_SELECTION)
            variant = self._variant(applied, run.tau_nm)
            if variant is None:
                row = msel.log_row(spec=applied, **kw)
                key_hash = applied.hash
            else:
                row = mtau.variant_log_row(variant, **kw)
                key_hash = variant.hash
            log.log_once(("review", self.axon_id, lumen, key_hash), row)
            n, n_tau = mtau.variant_counts(log, [self.axon_id, os.path.splitext(os.path.basename(self.source_name))[0]])
            line = mtau.variant_counter_line(max(n, 1), n_tau)
            self._counter_line_text = line
            self.exploration_log_error = ""
            self.selection_counter_label.setText((banner + " &mdash; " if banner else "") + line)
            self.selection_counter_label.setToolTip(COLUMNS_WINDOW_TOOLTIPS["selection_counter_label"]
                                                    + f"\n\nLog: {log.path}")
        except Exception as exc:  # noqa: BLE001 - the counter says it could not count; never fatal
            self.exploration_log_error = f"{type(exc).__name__}: {exc}"
            failed = marked(
                "bad", f"The exploration log could not be written ({self.exploration_log_error}): the selections "
                       "tried are NOT counted. p values are not corrected for trying several selections.")
            self._counter_line_text = failed
            self.selection_counter_label.setText(banner + " " + failed)
        self.selection_counter_label.setVisible(True)

    def _fill_viability(self) -> None:
        v = self.viability
        if v is None:
            return
        spec = getattr(v, "spec", None)
        exploratory = spec is not None and not spec.is_default
        n_v, n_m = len(v.viable), len(v.marginal)
        if n_v:
            head = marked("good", f"{n_v} VIABLE pair(s): the column analysis runs on them"
                          + (f"; {n_m} MARGINAL (sensitivity only)" if n_m else "") + ".")
        elif n_m:
            head = marked("warn", f"No VIABLE pair; {n_m} MARGINAL: only a sensitivity analysis will run.")
        else:
            head = marked("bad", "No VIABLE or MARGINAL pair: 'Apply and re-run' will compute nothing and say why.")
        if exploratory:
            self.viability_label.setText(f"Ring-pair viability (EXPLORATORY selection #{spec.hash}): {head}")
        else:
            self.viability_label.setText(f"Ring-pair viability (rule v2, D-41): {head}")
        self.viability_table.setHorizontalHeaderLabels(
            ["Pair", "Verdict (selection)" if exploratory else "Verdict (rule v2)", "Why"])
        from tools.mps_zquality_window import short_why
        detail = {(int(q.ring_a), int(q.ring_b)): q for q in getattr(getattr(v, "detail", None), "pairs", ())}
        chosen = {(int(q.ring_a), int(q.ring_b)): q for q in getattr(getattr(v, "selection", None), "pairs", ())}
        t = self.viability_table
        t.setRowCount(0)
        for p in v.pairs:
            i = t.rowCount()
            t.insertRow(i)
            word, kind = {"viable": ("ok  VIABLE", "good"), "marginal": ("!  MARGINAL", "warn")}.get(
                str(p.verdict), ("x  NOT VIABLE", "bad"))
            why = "; ".join(p.reasons) if p.reasons else "all four criteria met"
            q = detail.get((int(p.ring_a), int(p.ring_b)))
            ps = chosen.get((int(p.ring_a), int(p.ring_b)))
            if exploratory and ps is not None:
                # the CHECKED criteria that fail, then the unchecked ones that would (exploratory)
                brief = ("fails " + ", ".join(ps.failing)) if ps.failing else "every checked criterion met"
                if ps.unchecked_failing:
                    brief += "; not checked, would fail: " + ", ".join(ps.unchecked_failing)
                why = (why if p.reasons else "every checked criterion met")
            else:
                # the failed criteria with their numbers on one line; the full reasons in the tooltip
                brief = short_why(q) if q is not None and str(q.verdict) == str(p.verdict) else why
            for j, text in enumerate((f"{p.ring_a}-{p.ring_b}", word, brief)):
                it = QtWidgets.QTableWidgetItem(text)
                it.setToolTip(why)
                if j == 1:
                    it.setForeground(QtGui.QBrush(QtGui.QColor(verdict(kind, dark=False))))
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                t.setItem(i, j, it)
        t.resizeColumnToContents(0)
        t.resizeColumnToContents(1)
        self._refresh_tools()
        # UI stage 1: the curve's pairs follow the selection
        self._fill_curve_pairs()

    def _refresh_tools(self) -> None:
        v = self.viability
        ok = v is not None and bool(v.viable) and self.review_inputs is not None
        self.simnull_button.setEnabled(bool(ok))
        spec = getattr(v, "spec", None) if v is not None else None
        if v is not None and not v.viable:
            if spec is not None and not spec.is_default:
                self.simnull_button.setToolTip(
                    f"Disabled: no ring pair of this axon is VIABLE under the exploratory selection {spec.label}, so "
                    "there is nothing to compare with a simulated null.")
            else:
                self.simnull_button.setToolTip(SIMNULL_DISABLED_TIP)
        elif self.review_inputs is None:
            self.simnull_button.setToolTip("Disabled: this window was opened without the selection's localizations "
                                           "(frames and precisions), which the simulated null needs.\n\n"
                                           + SIMNULL_BUTTON_TIP)
        elif v is None:
            self.simnull_button.setToolTip("Waiting for the ring-pair viability.\n\n" + SIMNULL_BUTTON_TIP)
        else:
            self.simnull_button.setToolTip(SIMNULL_BUTTON_TIP)
        if not self._at_tau0():
            # UI stage 1 (D-44): the simulated null runs at the pre-registered tau_0 only
            self.simnull_button.setEnabled(False)
            self.simnull_button.setToolTip(mtui.SIMNULL_TAU_TIP)

    def _on_advanced_toggled(self, on: bool) -> None:
        self.advanced_box.setVisible(bool(on))
        self.advanced_toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if on else QtCore.Qt.ArrowType.RightArrow)

    def _on_zquality(self) -> None:
        from tools.mps_zquality_window import open_z_quality_for_rings
        # UI stage 0: one z-quality view for the whole program -- the main window's, on this review's own axon
        if self.zquality_callback is not None and self.review_inputs is not None:
            self.zquality_window = self.zquality_callback(self.review_inputs)
            return
        w = self.zquality_window
        if w is not None and w.isVisible():
            w.raise_()
            w.activateWindow()
            return
        self.zquality_window = open_z_quality_for_rings(self.res, self.xyz_lab_nm, parent=self,
                                                        source_name=self.source_name, roi_text=self.roi_text)

    def _on_simnull(self) -> None:
        from tools.mps_simnull_ui import SimnullDialog
        if self.review_inputs is None or self.viability is None or not self.viability.viable:
            return
        if not self._at_tau0():
            return      # UI stage 1: the simulated null runs at tau_0 only (the button is disabled meanwhile)
        d = self.simnull_dialog
        if d is not None and d.isVisible():
            d.raise_()
            d.activateWindow()
            return
        self.simnull_dialog = SimnullDialog(self.review_inputs, self.decisions, self.viability, parent=self,
                                            source_name=self.source_name, criteria=self.criteria,
                                            base_viability=self.base_viability, log_path=self._log_path(),
                                            axon_ids=[self.axon_id])
        self.simnull_dialog.show()

    def _on_columns_batch(self) -> None:
        from tools.mps_columns_batch_ui import ColumnsBatchDialog
        # UI stage 0: one batch dialog for the whole program -- the main window's
        if self.batch_callback is not None:
            self.batch_dialog = self.batch_callback()
            return
        d = self.batch_dialog
        if d is not None and d.isVisible():
            d.raise_()
            d.activateWindow()
            return
        self.batch_dialog = ColumnsBatchDialog(self)
        self.batch_dialog.show()

    # ------------------------------------------------------------ loading and exporting
    def _identity(self) -> AxonIdentity:
        ident = self.identity
        if callable(ident):
            try:
                ident = ident()
            except Exception:  # noqa: BLE001 - an identity dialog closed; the rows carry empty identity cells
                ident = None
        return ident if isinstance(ident, AxonIdentity) else AxonIdentity()

    def _head(self, exported_at: str) -> Dict[str, Any]:
        head: Dict[str, Any] = {"axon_id": self.axon_id}
        head.update(self._identity().columns())
        head.update({"source": self.source_name, "roi": self.roi_text, "exported_at": exported_at, "program": _PROGRAM})
        return head

    def _tau0(self) -> Optional[float]:
        r = self.last_result
        if r is not None and r.columns_2d is not None:
            return _num(r.columns_2d.tau0_nm)
        if self.columns_params is not None:
            return _num(self.columns_params.tau0_nm)
        return None

    def export_tables(self, path: str, *, duplicates: str = "ask") -> List[str]:
        """
        Write the decision table at ``path`` (one row per cluster,
        ``DECISION_TABLE_COLUMNS``) and, when a run has finished, its
        results at ``<stem>_results.csv`` (one row, ``RESULTS_TABLE_COLUMNS``),
        with ``tools.results_table``'s rules: both tables are checked before
        either is written; a table with other columns, or rows of another
        rule or recipe, is refused (ValueError, nothing written); rows this
        axon already has are replaced or kept beside the new ones as
        ``duplicates`` says ("ask": the export dialog when there are any;
        "replace"; "append"). Returns the paths written ([] when cancelled).
        """
        if duplicates not in ("ask", "replace", "append"):
            raise ValueError(f"export_tables: duplicates must be 'ask', 'replace' or 'append', got {duplicates!r}")
        path = os.path.abspath(str(path))
        stem, ext = os.path.splitext(path)
        results_path = f"{stem}_results{ext or '.csv'}"
        exported_at = datetime.now().isoformat(timespec="seconds")
        head = self._head(exported_at)
        dec_rows = decision_rows(self.decisions, head)
        tables: List[Tuple[str, List[Dict[str, Any]], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]] = [
            (path, dec_rows, DECISION_TABLE_KEY, DECISION_TABLE_ANALYSIS, DECISION_TABLE_COLUMNS)]
        if self.last_result is not None:
            # UI stage 1 (D-44): a run at another tau than tau_0 writes its tau and the EXPLORATORY tag (results_row)
            row = results_row(self.last_result, self.decisions, head, res_before=self.res, tau0_nm=self._tau0(),
                              run=self.last_run, variant=self._tau_variant(self.last_run))
            tables.append((results_path, [row], RESULTS_TABLE_KEY, RESULTS_TABLE_ANALYSIS, RESULTS_TABLE_COLUMNS))
        for target, rows, _key, analysis, columns in tables:
            check_appendable(target, rows, columns)
            refuse_other_analysis(target, rows[0], analysis)
        if duplicates == "ask":
            decision = export_ui.resolve_duplicates(self, [(t, rows, key) for t, rows, key, _a, _c in tables])
        else:
            decision = export_ui.REPLACE if duplicates == "replace" else export_ui.WRITE
        if decision == export_ui.CANCEL:
            return []
        written: List[str] = []
        for target, rows, key, _analysis, columns in tables:
            if decision == export_ui.REPLACE:
                replace_rows(target, rows, key, columns)
            else:
                append_rows(target, rows, columns)
            written.append(target)
        return written

    def _on_export(self) -> None:
        base = os.path.splitext(os.path.basename(self.source_name))[0] or "axon"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export the lumen decisions and results",
                                                        f"{base}_lumen_decisions.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            written = self.export_tables(path)
        except Exception as exc:  # noqa: BLE001 - a slot: surfaced to the user, never fatal
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Nothing was written:\n\n{exc}")
            return
        if not written:
            return
        QtWidgets.QMessageBox.information(self, "Exported", "Written:\n" + "\n".join(written) + self.export_note())

    def export_note(self) -> str:
        """What the export message adds after the files written: why only the decisions were written, if so."""
        if self.last_result is not None:
            return ""
        if self._tau_pending and self.tau_nm is not None:
            # UI stage 1: the results at the current tau are still being computed
            return "\n\n" + mtau.pending_export_message(self.tau_nm)
        return "\n\nNo run has finished yet: only the decisions were written."

    def load_decisions(self, path: str) -> int:
        """
        Apply decisions exported earlier (``export_tables``) to these
        clusters: the rows of this axon (by ``axon_id`` when the table has
        that column) are read back with ``LumenDecisions.from_rows`` against
        this classification. A table of another axon is REFUSED
        (ValueError, nothing changes): no row of this axon_id, rows saved on
        another cluster set (fingerprint: another axon under the same name,
        or this axon's rings built otherwise), no saved cluster among these,
        or a saved manual edit of a cluster these rings lack. When the table
        holds several exports of this axon, the LAST row of each cluster is
        used. Into an untouched window (no edit, no result shown) the saved
        final states, edit log and provenance flags are taken exactly as
        saved; otherwise each change the file makes is logged as an edit of
        this window (flagged if results were shown here) and the file's
        provenance is carried (``LumenDecisions.adopt``). The warnings go to
        the warnings list. Returns how many saved clusters matched these
        clusters. ValueError also while a re-run is running.
        """
        if self._running:
            raise ValueError("a re-run is running: load the decisions when it has finished")
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        mine = [r for r in rows if "axon_id" not in r or str(r.get("axon_id") or "") == self.axon_id]
        if not mine:
            raise ValueError(f"{os.path.basename(path)} has no row of this axon (axon_id {self.axon_id})")
        loaded = LumenDecisions.from_rows(mine, self.classification, strict_manual=True)
        untouched = not self.decisions.edit_log and not self.decisions.results_shown
        name = os.path.basename(path)
        if untouched:
            self.decisions = loaded
        else:
            n_changed = self.decisions.adopt(loaded, via=f"loaded {_clean_via(name)}")
            self.decisions.warnings.extend(w for w in loaded.warnings if w not in self.decisions.warnings)
            self.decisions.warnings.append(f"{name}: {n_changed} cluster(s) changed by the loaded decisions, logged as "
                                           "edits of this review")
        self.decisions.warnings.append(f"{name}: {loaded.rows_matched} of {loaded.rows_read} saved cluster(s) matched "
                                       f"this axon's {len(self.decisions.stable_keys)} clusters")
        self._fill_table()
        self._after_edit()
        self._fill_warnings()
        return int(loaded.rows_matched)

    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load lumen decisions", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            n = self.load_decisions(path)
        except Exception as exc:  # noqa: BLE001 - a slot: surfaced to the user, never fatal
            QtWidgets.QMessageBox.critical(self, "Load failed", f"The decisions were not loaded:\n\n{exc}")
            return
        QtWidgets.QMessageBox.information(
            self, "Loaded", f"{n} saved cluster(s) matched this axon's {len(self.decisions.stable_keys)} clusters and "
                            "were applied. See the warnings list.")

    # ------------------------------------------------------------ closing
    def closeEvent(self, event: Any) -> None:
        # A run started before the close still finishes in its thread; its result is dropped. The decisions are in
        # the review store already (saved at every change); saving once more retries a failed save.
        self._generation += 1
        self._running = False
        self._thread = None
        # H6 toggles: a closed window no longer follows the selection, and a live run is stopped
        self._closed = True
        self._debounce.stop()
        if self._run_live:
            self._cancel_seq = self._run_seq
        self.progress_bar.setVisible(False)
        self._refresh_state()
        self._persist()
        super().closeEvent(event)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        # UI stage 1: the map / curve split follows the window until the user drags the handle (after the layout)
        if not self._left_split_by_user:
            QtCore.QTimer.singleShot(0, self._fit_left_split)

    def _on_left_split_moved(self, _pos: int = 0, _index: int = 0) -> None:
        self._left_split_by_user = True

    def _fit_left_split(self) -> None:
        """The curve gets up to ``LEFT_CURVE_SHARE`` of the left column and the map at least
        ``MAP_MIN_WINDOW_SHARE`` of the window's height (each widget's own minimum still wins)."""
        if self._left_split_by_user:
            return
        try:
            sizes = self.left_splitter.sizes()
            total = int(sum(sizes))
            if total <= 0:
                return
            curve = int(min(round(LEFT_CURVE_SHARE * total), total - MAP_MIN_WINDOW_SHARE * self.height()))
            want = [total - max(curve, 0), max(curve, 0)]
            if want != list(sizes):
                self.left_splitter.setSizes(want)
        except RuntimeError:        # being destroyed
            return

    def showEvent(self, event: Any) -> None:
        if self._closed:
            # shown again: follow the program's selection again
            self._closed = False
            QtCore.QTimer.singleShot(0, self._on_selection)
        super().showEvent(event)


# ============================================================================
# Opening the window: the rings are built once, in a worker
# ============================================================================

class ColumnsReviewLauncher(QtWidgets.QWidget):
    """
    A small window that builds the review (``prepare_review``: the rings,
    their classification and the membrane through every cluster) in a
    worker thread with a progress bar, then opens the ``ColumnsWindow`` and
    closes itself. ``window`` is the review window once open;
    ``window_opened`` is emitted with it. Cancel abandons the build (the
    thread finishes on its own; its result is dropped).
    """

    window_opened = QtCore.pyqtSignal(object)

    def __init__(self, inputs: ColumnsReviewInputs, parent: Optional[QtWidgets.QWidget] = None, *,
                 columns_params: Optional[ColumnsParams] = None, n_null: Optional[int] = None,
                 review_store: Optional[LumenReviewStore] = None,
                 zquality_callback: Optional[Callable[[Any], Any]] = None,
                 batch_callback: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        # passed to the review window (UI stage 0: the program's single Z quality view and batch dialog)
        self.zquality_callback = zquality_callback
        self.batch_callback = batch_callback
        self.inputs = inputs
        self.columns_params = columns_params
        self.n_null = n_null
        self.review_store = review_store
        # What the caller uses to recognise the same review (MPS_explorer.open_columns_review); not read here.
        self.review_signature: Any = None
        self.window: Optional[ColumnsWindow] = None
        self.review: Optional[ColumnsReview] = None
        self.error: Optional[str] = None
        self._generation = 0
        self._thread: Optional[threading.Thread] = None
        self._relay = _Relay()
        self._relay.progress.connect(self._on_progress)
        self._relay.finished.connect(self._on_finished)
        self.setWindowTitle("H-ECL columns: building the rings")
        self.resize(560, 140)
        lay = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel(
            f"Building the H-ECL rings of {os.path.basename(str(inputs.source_name)) or 'this axon'} "
            f"({np.asarray(inputs.x_nm).size:,} localizations of the ROI, before the axial cut) with the pre-registered "
            "parameters, then classifying their clusters. This runs once; the review window opens when it is done.")
        self.label.setWordWrap(True)
        lay.addWidget(self.label)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(True)
        lay.addWidget(self.progress_bar)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setToolTip("Stop waiting. The build finishes in the background and is thrown away.")
        self.cancel_button.clicked.connect(self._on_cancel)
        row.addWidget(self.cancel_button)
        lay.addLayout(row)

    def start(self) -> None:
        """Start the build in a worker thread."""
        if self._thread is not None:
            return
        inputs, relay, gen = self.inputs, self._relay, self._generation

        def report(text: str, fraction: float) -> None:
            relay.progress.emit(gen, str(text), float(fraction))

        def body() -> None:
            try:
                review = prepare_review(inputs, progress=report)
            except Exception as exc:  # noqa: BLE001 - shown to the user
                relay.finished.emit(gen, None, exc)
                return
            relay.finished.emit(gen, review, None)

        self._thread = threading.Thread(target=body, daemon=True, name="lumen-review-build")
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None

    def _on_cancel(self) -> None:
        self.close()

    def wait(self, timeout: float = 900.0) -> None:
        """Block until the build is done and the window open (for scripts and tests)."""
        t_end = time.perf_counter() + float(timeout)
        while self._thread is not None and time.perf_counter() < t_end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
        QtWidgets.QApplication.processEvents()

    def _on_progress(self, gen: int, text: str, _fraction: float) -> None:
        if gen == self._generation:
            self.progress_bar.setFormat(text)
            self.label.setToolTip(text)

    def _on_finished(self, gen: int, review: Any, error: Any) -> None:
        # A slot: an exception escaping it would abort the whole application (PyQt5 without an excepthook).
        self._thread = None
        if gen != self._generation:
            return
        if error is None and review is not None:
            self.review = review
            inp = self.inputs
            try:
                self.window = ColumnsWindow(
                    review.res, review.classification, lpz_nm=review.lpz_nm, columns_params=self.columns_params,
                    n_null=self.n_null, widefield_images=inp.widefield if review.classification.has_widefield else None,
                    identity=inp.identity, source_name=inp.source_name, parent=self.parentWidget(), roi_text=inp.roi_text,
                    membrane_before=review.membrane_before, widefield_note=inp.widefield_note,
                    extra_warnings=review.warnings, review_store=self.review_store, rings_note=review.rings_note,
                    xyz_lab_nm=(inp.x_nm, inp.y_nm, inp.z_nm), review_inputs=inp,
                    zquality_callback=self.zquality_callback, batch_callback=self.batch_callback)
            except Exception as exc:  # noqa: BLE001 - shown to the user below
                self.window = None
                error = exc
            else:
                self.window.show()
                self.window.raise_()
                self.window_opened.emit(self.window)
                self.hide()
                return
        self.error = f"{type(error).__name__}: {error}" if error is not None else "no result"
        self.hide()
        QtWidgets.QMessageBox.critical(self.parentWidget(), "H-ECL columns",
                                       f"The review of this selection could not be opened:\n\n{self.error}")
        self.close()

    def closeEvent(self, event: Any) -> None:
        if self._thread is not None:
            self._generation += 1
            self._thread = None
        super().closeEvent(event)


def start_columns_review(inputs: ColumnsReviewInputs, parent: Optional[QtWidgets.QWidget] = None, *,
                         columns_params: Optional[ColumnsParams] = None,
                         n_null: Optional[int] = None,
                         review_store: Optional[LumenReviewStore] = None,
                         zquality_callback: Optional[Callable[[Any], Any]] = None,
                         batch_callback: Optional[Callable[[], Any]] = None) -> ColumnsReviewLauncher:
    """Show the launcher and start building the review; the review window opens when the build is done
    (``review_store``: where the axon's decisions are kept between sessions, see ``ColumnsWindow``;
    ``zquality_callback`` / ``batch_callback``: the program's single Z quality view and batch dialog)."""
    launcher = ColumnsReviewLauncher(inputs, parent, columns_params=columns_params, n_null=n_null,
                                     review_store=review_store, zquality_callback=zquality_callback,
                                     batch_callback=batch_callback)
    launcher.show()
    launcher.start()
    return launcher
