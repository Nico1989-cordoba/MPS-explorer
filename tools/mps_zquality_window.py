# -*- coding: utf-8 -*-
"""
The "Z quality" view of one axon (D-41, the ring-pair viability rule v2):
can the clusters of two consecutive MPS rings of this axon be compared at
all, and if not, what about the acquisition stops them?

What the window shows
---------------------
* The axial profile of the axon (z' in the axon's own frame, the tilt of
  the axis removed): a histogram of every localization, its kernel density
  (h = 0.10 P = 17 nm), the ring centres -- each labelled with its share of
  the CENTRAL ring's localizations, the central ring marked (Q-35) -- and
  the significant peaks and valleys.
* Under it the SiZer strip (Chaudhuri & Marron 1999, cluster-robust: each
  3D DBSCAN cluster counts as one unit): where the smoothed density rises
  significantly (blue, upper half), falls significantly (vermillion, lower
  half) or neither (grey, thin middle line), with alpha 0.05 over the whole
  profile at once. A peak is a significant rise followed by a fall; a valley
  a fall followed by a rise.
* One row per ring and one row per consecutive ring pair with the four
  criteria of rule v2: (1) a significant peak of each ring, (2) a
  significant valley between them, (3) each ring holds at least f_min(x) of
  the central ring's localizations (x = localizations per cluster of the
  central ring / min_samples; f_min = 1.46 - 0.336 ln x, validated for x in
  5.8-13.3, conservative outside), (4) the expected axial-leak copies of
  D-39: <= 2 % VIABLE, <= 5 % MARGINAL (sensitivity only), more NOT VIABLE;
  every verdict with its reasons in plain words.
* "What limits z quality here": the axon's limiting factors with the
  numbers and what the rule would need, written for the microscopy team.
* "Export z-quality report...": a PNG of the view and of the profile, the
  tables as CSV and the summary as text.
* Under the localization profile, on the same z' axis (H6 clusters,
  descriptive, 2026-10-02): the CLUSTER medians -- one z' per cluster --
  in 20 nm bars with their own SiZer strip (iid, one point per cluster,
  h = 0.10 P), for the slab-free 3D clusters (default) or the ring clusters
  of the column test (``tools.mps_axial_clusters``). No verdict: rule v2
  still reads the localizations. The export adds a PNG of this panel and a
  CSV of the medians as NEW files; the files above are unchanged.
* At the bottom of the view, full width (H6 clusters, EXPLORATORY, D-42): a
  separate table "rule v2-clusters" with the same four criteria read on the
  clusters (one point per cluster) for three cluster sets, A (ring clusters,
  median), B (slab-free 3D clusters, median), C (ring clusters, centre
  corrected for the slab cut); its verdicts never select pairs under the
  pre-specified rule, and rule v2's table, summary file and exports do not
  change. The summary view adds its block in plain words; the export adds
  one NEW file, ``<stem>_zquality_clusters_pairs.csv``.
* Above the plots (H6 toggles, D-43, the user's request of 2026-10-02):
  the criteria switches (``tools.mps_selection_ui.SelectionWidget``, the
  program's one selection). Since Q-38 BOTH rules can select the pairs of
  the column tests, criterion by criterion; anything but the pre-specified
  rule v2[peak,valley,count,leak:D-39] is an EXPLORATORY selection. Live
  (no recomputation): under an exploratory selection the rows of the pairs
  it selects are tinted (green VIABLE, orange MARGINAL) in the table(s) of
  the rule it reads and the rings of those pairs are bold; under the
  pre-specified rule the tables look exactly as before. The export adds
  ``<stem>_zquality_selection.csv`` ONLY under an exploratory selection.

What it does NOT do
-------------------
No column statistic is computed here (no matching, no z_A, no p): this is
the geometry of the z profile only, allowed on real axons (R8). Rule v2 is
a viability filter, and the column test's p-values remain NOT calibrated
(H5-E closed as not accepted, D-41).

The computation (``compute_z_quality``: ``z_quality`` + ``viability_v2``
with the 3D groups on the LAB coordinates, as the batch runner and simnull
do) runs in a worker thread; the window never freezes.

Run on its own (a demonstration on SIMULATED axons, or an NPZ):
    venv\\Scripts\\python.exe -m tools.mps_zquality_window --demo viable
    venv\\Scripts\\python.exe -m tools.mps_zquality_window --demo none
    venv\\Scripts\\python.exe -m tools.mps_zquality_window --npz axon.npz

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # run as a script: make "tools" importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt5 import QtCore, QtGui, QtWidgets

from tools.mps_axial_clusters import (
    CLUSTER_SET_LABELS, CLUSTER_SET_ORDER, CLUSTER_SIZER_H_OVER_P, HIST_BIN_CLUSTERS_NM, cluster_median_rows,
    cluster_z_profile)
from tools.mps_axial_clusters import (
    V2C_CUT_FORCED, V2C_H_TABLE, V2C_K_MIN, V2C_RULE, V2C_VARIANT_ORDER, v2c_pair_rows, v2c_summary_lines,
    viability_v2c)
from tools.mps_layer_panel import LegendLayers
from tools.mps_plot_style import (
    AXIS_FG, PLOT_BG, ROLES, marked, neutral, rgba, set_title, style_dark, verdict)
from tools import mps_selection as msel
from tools.mps_selection_ui import SelectionWidget
from tools.mps_tooltips import apply_tooltips

__all__ = [
    "CLUSTER_NOTE",
    "HIST_BIN_NM",
    "PROGRAM",
    "SIZER_LABELS",
    "ZQUALITY_WINDOW_TOOLTIPS",
    "ZQualityReport",
    "ZQualityWindow",
    "compute_z_quality",
    "compute_z_quality_clusters",
    "factor_rows",
    "lab_xyz_of",
    "open_z_quality_for_inputs",
    "open_z_quality_for_rings",
    "pair_rows",
    "plain_summary",
    "short_why",
    "profile_rows",
    "ring_rows",
    "verdict_text",
    "write_cluster_report_files",
    "write_report_files",
    # rule v2-clusters (EXPLORATORY, D-42)
    "V2C_COLUMNS",
    "compute_z_quality_v2c",
    "short_why_v2c",
    "write_v2c_report_files",
]

PROGRAM = "MPS Explorer (tools/mps_zquality_window.py)"
HIST_BIN_NM = 10.0
# The SiZer strip: role of tools.mps_plot_style.ROLES, the band's bottom and top (the strip runs from -1 to +1), and
# the legend text. The height carries the meaning as well as the colour (validate_plot_colours.py, "the SiZer strip").
SIZER_STYLE: Dict[int, Tuple[str, float, float, str]] = {
    1: ("sizer_rise", 0.0, 1.0, "SiZer: density rises (significant)"),
    -1: ("sizer_fall", -1.0, 0.0, "SiZer: density falls (significant)"),
    0: ("sizer_flat", -0.12, 0.12, "SiZer: no significant slope"),
}
SIZER_LABELS: Dict[int, str] = {k: v[3] for k, v in SIZER_STYLE.items()}
NOT_CALIBRATED_LINE = ("The column test's p-values stay NOT calibrated (H5-E closed as not accepted, D-41); this view "
                       "computes no column statistic, only the geometry of the z profile.")
# What the exploratory dataset taught about the acquisition (not about the axon on screen): the summary's "FOR NEW
# ACQUISITIONS" block, as generic advice. Each line names its evidence; nothing here is a promise about a new sample.
MEASURED_LEVERS: Tuple[str, ...] = (
    "Picasso fitting box: a box narrower than about 6 sigma of the PSF truncates the astigmatic PSF; use a box that "
    "covers +-3 sigma (check sx, sy at their 95th percentile) for new data. A truncated PSF can make the rings wider "
    "than the reported precision. [axial-width budget, 2026-09-13]",
    "Picasso z filter: a z filter that cuts where the density is still high truncates the edge rings; widen the z "
    "filter when exporting from Picasso so that the profile falls to ~0 before the cut. [zmin report sec. 4]",
    "Check with the data before blaming the sample: axial drift (are the z shifts between time segments "
    "autocorrelated?), axon tilt (does removing it change sigma_z?) and the reported precisions (NeNA against the "
    "reported lp). [axial-width budget, 2026-09-13]",
)
_MARK_OF_STATUS = {"limiting": "bad", "warning": "warn", "outside calibrated range": "warn", "ok": "good", "info": "dim"}
# The cluster panel's one-line note (trap (i) of OPT_MAP_C): the ring clusters come from DBSCAN inside each ring's slab.
CLUSTER_NOTE = "Ring-slab medians cannot cross a slab cut (dash-dot): a dip there is built in."
# The height the view needs with the cluster panel shown (the localization plots, the cluster plots and the tables each
# above the size where their axes and rows start to clip, measured at 1920 x 1020 and 1366 x 728 on Windows).
CLUSTER_VIEW_MIN_HEIGHT = 1000
# H6 toggles (orchestrator, review N1 of H6 clusters): rule v2's localization profile and pair table are what the user
# shows the microscopy team, so they keep the height they had before the cluster work (ff2ce46: profile 216 px, pair
# table 253 px at 1920 x 1020). The view is at least the visible height (never below VIEW_BASE_MIN_HEIGHT), shared by
# the localization plots and the tables as before (SPLIT_SHARES), and every later panel -- the cluster medians (a fixed
# CLUSTER_PANEL_HEIGHT inside the splitter), the criteria switches above, the rule v2-clusters table below -- ADDS its
# height; the view scrolls to them.
VIEW_BASE_MIN_HEIGHT = 900
CLUSTER_PANEL_HEIGHT = 250
SPLIT_SHARES: Tuple[float, float] = (0.42, 0.58)
# The tint of the rows an exploratory selection selects (role, alpha): a pale wash under the black text of the table.
SELECTION_TINT: Dict[str, Tuple[str, int]] = {"viable": ("good", 60), "marginal": ("warn", 70)}
PAIRS_TITLE = "<b>Consecutive ring pairs</b>: rule v2 (D-41), criteria (1)-(4)"
TINT_NOTE = ("<span style='background:#cdebe0'>&nbsp;tinted&nbsp;</span> rows: the pairs the EXPLORATORY "
             "selection reads (green VIABLE, orange MARGINAL); the verdicts stay the rule's own")
TINT_SUMMARY = "; their rows are tinted in the tables below (green VIABLE, orange MARGINAL; verdicts unchanged)"
# The rule v2-clusters panel (EXPLORATORY, D-42), full width under everything else: the view grows by its height (the
# table at most this: one row per pair and cluster set, 6 rows for 3 rings), so rule v2's plots and tables keep theirs
# exactly; the panel is reached by scrolling down.
V2C_VIEW_EXTRA_HEIGHT = 380
V2C_SHORT_NAMES: Dict[str, str] = {
    "ring_slab": "A  ring clusters",
    "groups_3d": "B  3D clusters",
    "ring_slab_tn": "C  ring, corrected",
}

# Stable keys of the SiZer legend entries (UI stage 0, the layer state of the window; never shown)
SIZER_LAYER_KEYS: Dict[int, str] = {1: "sizer_up", -1: "sizer_down", 0: "sizer_flat"}

ZQUALITY_WINDOW_TOOLTIPS: Dict[str, str] = {
    "graphics":
        "The axial profile of this axon: every localization's z' (the axon's own axis, the tilt removed) in 10 nm "
        "bars, the smoothed density (orange; kernel h = 0.10 P), the ring centres (grey lines, labelled with the "
        "ring's share of the CENTRAL ring's localizations; the central ring is the thick solid line), and the "
        "significant peaks (up triangles) and valleys (down triangles).\n\n"
        "The strip underneath is SiZer: blue upper bar = the density rises significantly there, vermillion lower bar "
        "= it falls significantly, thin grey = no significant slope. Each 3D cluster counts as ONE unit in the "
        "variance (its localizations share their z), and alpha 0.05 holds over the whole profile at once.\n\n"
        "Two rings can be compared only if each has its own significant peak and a significant valley separates "
        "them: on the strip, blue then vermillion around each ring, and vermillion then blue between them.\n\n"
        "Click a legend entry to hide or show it.",
    "ring_table":
        "One row per ring, bottom to top. '% of central' is the ring's localizations over the central ring's "
        "(Q-35: the middle ring; with 2 rings, or 2 middle rings, the more populated). x = localizations per cluster "
        "of the central ring / min_samples sets the minimum % every ring of a pair needs (criterion 3). sigma_z / P "
        "is the ring's axial width over the MPS period: the smaller, the better separated the rings.",
    "pair_table":
        "One row per consecutive ring pair with the four criteria of rule v2 (D-41): (1) a significant peak of each "
        "ring, (2) a significant valley between them, (3) each ring holds at least f_min(x) of the central ring's "
        "localizations, (4) the expected axial-leak copies (D-39) <= 2 % for VIABLE, <= 5 % for MARGINAL.\n\n"
        "Only VIABLE pairs enter the column test; MARGINAL pairs are a sensitivity analysis only; NOT VIABLE pairs "
        "are never analysed. 'Why' says, in plain words, what failed.\n\n"
        "The verdicts here are always rule v2's own. Under an EXPLORATORY selection of the switches above (D-43) the "
        "rows of the pairs it selects are tinted (green VIABLE, orange MARGINAL) and the cell tooltips say which "
        "criteria fail and which are not checked.",
    "summary_view":
        "The same verdicts in plain words for the microscopy team: what limits the z quality of this axon, with the "
        "numbers, and what the rule would need. Every line of 'What would help' names its evidence in [brackets]: "
        "zmin report = the statistical study behind rule v2 (2026-10-01/02); axial-width budget = the measurement of "
        "why the rings are wider than the precision (exploratory dataset, 2026-09-13); D-39 / D-41 = the decisions.",
    "export_button":
        "Write the report of this axon into a folder you choose: a PNG of this view and one of the profile, the "
        "ring, pair, limiting-factor and profile tables as CSV, and the summary as text. Existing files are never "
        "overwritten (a number is added).\n\n"
        "With the cluster panel shown, two more files: a PNG of the cluster-median histogram (the cluster set "
        "selected) and a CSV with every cluster's median z' for both cluster sets (_zquality_clusters...).\n\n"
        "With the rule v2-clusters table shown (exploratory), one more: _zquality_clusters_pairs.csv, its criteria and "
        "verdicts per pair and cluster set.",
    "graphics_clusters":
        "The axial profile of this axon's CLUSTERS, on the same z' axis as the localization profile above: one value "
        "per cluster, the median z' of its localizations, in 20 nm bars (bluish green); the smoothed density of the "
        "medians (orange; h = 0.10 P) and its significant peaks (up triangles) and valleys (down triangles).\n\n"
        "The strip underneath is SiZer on the medians, each cluster ONE independent point (iid), alpha 0.05 over the "
        "whole profile; same colours as the strip above. Grey lines: the ring centres, labelled with how many "
        "clusters have their median inside that ring's slab. Dash-dot lines: the slab cuts between the rings.\n\n"
        "Descriptive only: rule v2 and its verdicts above read the localizations; nothing here changes them.",
    "cluster_source_combo":
        "Which clusters give the medians.\n\n"
        "Slab-free 3D clusters (default): DBSCAN on the lab coordinates (x, y, 0.5 z; eps 25 nm, min_samples 10), "
        "the clusters that set rule v2's SiZer variance. They know nothing about the rings, so a median can fall "
        "anywhere; but clusters stacked in z within ~50 nm can merge into one.\n\n"
        "Ring clusters (as used by the column test): DBSCAN INSIDE each ring's axial slab. A ring cluster's median "
        "can never cross its slab's cut, so a dip of their histogram at a cut comes from the segmentation itself "
        "(moving the cut moves the dip).",
    "cluster_note":
        "Why the two cluster sets differ near the slab cuts (dash-dot lines): see the cluster set selector.",
    "v2c_title":
        "Rule v2-clusters (EXPLORATORY, D-42): the four criteria of rule v2 read on the CLUSTERS instead of the "
        "localizations. It never selects pairs under the pre-specified rule: rule v2 above decides what enters the "
        "column test. Since the user's decision of 2026-10-02 (Q-38, D-43) it can select them when chosen in the "
        "switches above, always as an EXPLORATORY selection, criterion by criterion.",
    "v2c_table":
        "Rule v2-clusters (EXPLORATORY, D-42; not the rule). One row per ring pair and cluster set; each cluster is ONE "
        "point: A = ring clusters (median z'), B = slab-free 3D clusters (median z'), C = ring clusters with the centre "
        "corrected for the slab cut.\n\n"
        "(1c) a significant SiZer peak of the cluster points (iid, h = 0.10 P) within P/3 of each ring centre; (2c) a "
        "significant valley between them; (3c) each ring holds >= 89 % of the CENTRAL ring's clusters (K / K "
        "central); (4c) the cluster-level expected leak copies (tnfix) <= 2 % VIABLE, <= 5 % MARGINAL, shown beside "
        "rule v2's D-39 number. '(1c)+(2c) at h 0.20 P': the same two criteria with a wider kernel (information "
        "only).\n\n"
        "A valley of A or C inside a slab gap (the dash-dot cuts of the cluster panel, +-15 nm) is built by the "
        "segmentation: each ring cluster is found inside its ring's slab, so no median crosses the cut, and moving "
        "the cut moves the dip. It never counts for (2c); for B it is only flagged.\n\n"
        "It never selects pairs under the pre-specified rule (rule v2). With the switches above (Q-38, D-43) the "
        "column review, the batch and the simulated null can read it as an EXPLORATORY selection: the rows it then "
        "selects are tinted. A and C cannot pass (2c) for two similar rings (the slab cut sits at the valley): "
        "uncheck (2c) to use them; B's selection depends on whether columns exist (lower power).",
    "columns_button":
        "Open the H-ECL column review of this axon (the lumen clusters, then the column test on the VIABLE pairs "
        "only).",
    "progress_bar":
        "The z-quality computation runs in the background; the window stays usable.",
}


# ============================================================================
# The numbers (no Qt)
# ============================================================================

@dataclass
class ZQualityReport:
    """Everything the view draws and exports: the axon, its z' values and
    histogram, ``z_quality`` (D-39) and ``viability_v2`` (D-41), where the
    lab coordinates came from, the warnings and the time."""

    source_name: str
    roi_text: str
    axon_id: str
    n_locs: int
    z_p: NDArray[np.float64]
    hist_edges: NDArray[np.float64]
    hist_counts: NDArray[np.float64]
    zq: Any
    v2: Any
    lab_source: str
    seconds: float
    warnings: List[str] = field(default_factory=list)
    computed_at: str = ""
    # H6 clusters (descriptive): ``tools.mps_axial_clusters.ClusterZProfile`` of the axon, filled by
    # ``compute_z_quality_clusters`` AFTER ``compute_z_quality`` (so every field above stays what it was); None = not
    # computed (the cluster panel stays hidden), with the reason in ``clusters_error`` when it failed.
    clusters: Any = None
    clusters_error: str = ""
    # Rule v2-clusters (EXPLORATORY, D-42): ``tools.mps_axial_clusters.AxonViabilityV2C``, filled by
    # ``compute_z_quality_v2c`` after the two above; None = not computed (the table stays hidden). Never selects pairs.
    v2c: Any = None
    v2c_error: str = ""


def lab_xyz_of(res: Any) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The LAB coordinates of every localization of a ``RingsResult``,
    rebuilt from its axon frame: p' = R (p - o) + o, so p = R^T (p' - o) + o
    (``tools.mps_axis.to_axon_frame``); exact to ~1e-12 nm."""
    rot = np.asarray(res.frame.rotation, dtype=np.float64)
    origin = np.asarray(res.frame.origin_nm, dtype=np.float64).reshape(3, 1)
    p = np.vstack([np.asarray(res.x_p, dtype=np.float64), np.asarray(res.y_p, dtype=np.float64),
                   np.asarray(res.z_p, dtype=np.float64)])
    lab = rot.T @ (p - origin) + origin
    return (np.asarray(lab[0], dtype=np.float64), np.asarray(lab[1], dtype=np.float64),
            np.asarray(lab[2], dtype=np.float64))


def compute_z_quality(res: Any, xyz_lab_nm: Optional[Sequence[Any]] = None, *, p_ref_nm: Optional[float] = None,
                      source_name: str = "", roi_text: str = "") -> ZQualityReport:
    """
    The z-quality report of one axon's rings (``build_rings`` as the column
    review builds them): ``z_quality`` and rule v2 (``viability_v2``) with
    the 3D groups on the LAB coordinates ``xyz_lab_nm`` -- rebuilt from the
    axon frame when not given (``lab_xyz_of``) -- and the 10 nm histogram of
    z'. Geometry only: no column statistic.
    """
    from tools.mps_axial_precision import viability_v2, z_quality
    from tools.mps_identity import axon_id

    t0 = time.perf_counter()
    warnings_: List[str] = []
    if xyz_lab_nm is None:
        xyz = lab_xyz_of(res)
        lab_source = "lab coordinates rebuilt from the axon frame"
    else:
        xyz = tuple(np.asarray(v, dtype=np.float64).reshape(-1) for v in xyz_lab_nm)  # type: ignore[assignment,unused-ignore]
        lab_source = "lab coordinates of the selection"
    zp = np.asarray(res.z_p, dtype=np.float64)
    if xyz[0].size != zp.size:
        raise ValueError(f"{xyz[0].size} lab coordinates for {zp.size} localizations of the rings")
    zq = z_quality(res, p_ref_nm=p_ref_nm)
    v2 = viability_v2(res, zq, xyz_lab_nm=xyz, p_ref_nm=p_ref_nm)
    warnings_.extend(str(w) for w in getattr(zq, "warnings", ()))
    warnings_.extend(str(w) for w in v2.warnings)
    fin = zp[np.isfinite(zp)]
    if fin.size:
        lo = math.floor(float(fin.min()) / HIST_BIN_NM) * HIST_BIN_NM
        hi = math.ceil(float(fin.max()) / HIST_BIN_NM) * HIST_BIN_NM + HIST_BIN_NM
        edges = np.arange(lo, hi + 0.5 * HIST_BIN_NM, HIST_BIN_NM)
        counts, edges = np.histogram(fin, bins=edges)
    else:
        edges, counts = np.array([0.0, HIST_BIN_NM]), np.array([0])
    src = str(source_name or getattr(res, "source_name", "") or "")
    return ZQualityReport(
        source_name=src, roi_text=str(roi_text or ""), axon_id=axon_id(os.path.basename(src), str(roi_text or "")),
        n_locs=int(zp.size), z_p=zp, hist_edges=np.asarray(edges, dtype=np.float64),
        hist_counts=np.asarray(counts, dtype=np.float64), zq=zq, v2=v2, lab_source=lab_source,
        seconds=float(time.perf_counter() - t0), warnings=warnings_,
        computed_at=datetime.now().isoformat(timespec="seconds"))


def compute_z_quality_clusters(res: Any, xyz_lab_nm: Optional[Sequence[Any]], rep: ZQualityReport) -> ZQualityReport:
    """
    Add the cluster-median profile (``tools.mps_axial_clusters.cluster_z_profile``: the slab-free 3D clusters and
    the ring clusters, one median z' each, with their SiZer at h = ``CLUSTER_SIZER_H_OVER_P`` P) to ``rep``, a report
    of ``compute_z_quality`` for the same ``res``. Only ``rep.clusters`` / ``rep.clusters_error`` change; a failure
    here never touches rule v2's report (its warnings, time and verdicts stay as they were). Descriptive: no
    verdict, no column statistic.
    """
    try:
        xyz = lab_xyz_of(res) if xyz_lab_nm is None else tuple(xyz_lab_nm)
        rep.clusters = cluster_z_profile(res, xyz, p_ref_nm=float(rep.v2.p_ref_nm), h_over_p=CLUSTER_SIZER_H_OVER_P)
        rep.clusters_error = ""
    except Exception as exc:  # noqa: BLE001 - the localization view must survive; the reason is kept
        rep.clusters = None
        rep.clusters_error = f"{type(exc).__name__}: {exc}"
    return rep


def compute_z_quality_v2c(res: Any, xyz_lab_nm: Optional[Sequence[Any]], rep: ZQualityReport) -> ZQualityReport:
    """
    Add rule v2-clusters (EXPLORATORY, D-42: ``tools.mps_axial_clusters.viability_v2c``, the four criteria of rule
    v2 on the clusters) to ``rep``, a report of ``compute_z_quality`` (+ ``compute_z_quality_clusters``) for the same
    ``res``; variant B reuses ``rep.clusters`` when it is there. Only ``rep.v2c`` / ``rep.v2c_error`` change: rule
    v2's report, warnings and time stay as they were, and a failure here never touches them. Never selects pairs.
    """
    try:
        xyz = lab_xyz_of(res) if xyz_lab_nm is None else tuple(xyz_lab_nm)
        rep.v2c = viability_v2c(res, rep.v2, xyz_lab_nm=xyz, clusters=rep.clusters, p_ref_nm=float(rep.v2.p_ref_nm))
        rep.v2c_error = ""
    except Exception as exc:  # noqa: BLE001 - the localization view must survive; the reason is kept
        rep.v2c = None
        rep.v2c_error = f"{type(exc).__name__}: {exc}"
    return rep


def _pct(f: Any) -> str:
    try:
        v = float(f)
    except (TypeError, ValueError):
        return "n/a"
    return f"{100.0 * v:.0f} %" if math.isfinite(v) else "n/a"


def _num(v: Any, spec: str = ".1f") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return format(f, spec) if math.isfinite(f) else "n/a"


def verdict_text(verdict_: str) -> Tuple[str, str]:
    """(text with its mark, verdict kind of tools.mps_plot_style) of a rule-v2 verdict."""
    v = str(verdict_)
    if v == "viable":
        return marked("good", "VIABLE"), "good"
    if v == "marginal":
        return marked("warn", "MARGINAL (sensitivity only)"), "warn"
    return marked("bad", "NOT VIABLE"), "bad"


def short_why(p: Any) -> str:
    """One line with the criteria a rule-v2 pair (``PairViabilityV2``) fails and their numbers, e.g. "(1) ring 0: no
    peak; (2) no valley; (3) 55 % / 100 % < 72 %; (4) leak 8.0 % > 5 %" (the full sentences are ``p.reasons``)."""
    from tools.mps_axial_precision import V2_SPUR_MARGINAL, V2_SPUR_VIABLE
    if str(p.verdict) == "viable":
        return "all four criteria met"
    parts: List[str] = []
    miss = [str(k) for k, ok in ((p.ring_a, p.peak_a), (p.ring_b, p.peak_b)) if not ok]
    if miss:
        parts.append(f"(1) ring {', '.join(miss)}: no peak")
    if not p.valley:
        parts.append("(2) no valley")
    if not p.count_ok:
        parts.append(f"(3) {_pct(p.f_a)} / {_pct(p.f_b)} < {_pct(p.f_min)}")
    spur = float(p.exp_spur_frac)
    if math.isfinite(spur) and spur > V2_SPUR_VIABLE:
        limit = V2_SPUR_MARGINAL if spur > V2_SPUR_MARGINAL else V2_SPUR_VIABLE
        parts.append(f"(4) leak {100 * spur:.1f} % > {100 * limit:g} %")
    text = "; ".join(parts) or "; ".join(p.reasons)
    return text + (" (marginal: sensitivity only)" if str(p.verdict) == "marginal" else "")


def short_why_v2c(p: Any) -> str:
    """One line with the criteria a rule v2-clusters pair (``V2CPair``) fails, e.g. "(1c) ring 0: no peak; (2c)
    valley in the slab gap (built in); (3c) 60 % < 89 %; (4c) leak 20.0 % > 5 %" (the full sentences are
    ``p.reasons``)."""
    from tools.mps_axial_clusters import V2C_SPUR_MARGINAL, V2C_SPUR_VIABLE
    parts: List[str] = []
    miss = [str(k) for k, ok in ((p.ring_a, p.peak_a), (p.ring_b, p.peak_b)) if not ok]
    if miss:
        parts.append(f"(1c) ring {', '.join(miss)}: no peak")
    if not p.valley_raw:
        parts.append("(2c) no valley")
    elif p.valley_at_cut and p.variant in V2C_CUT_FORCED:
        parts.append("(2c) valley in the slab gap (built in, not counted)")
    elif p.valley_at_cut:
        parts.append("(2c) valley in the slab gap (flagged)")
    if not p.count_ok:
        parts.append(f"(3c) {_pct(p.k_ratio_a)} / {_pct(p.k_ratio_b)} < {_pct(p.k_min)}")
    spur = float(p.exp_spur_frac_clusters)
    if math.isfinite(spur) and spur > V2C_SPUR_VIABLE:
        limit = V2C_SPUR_MARGINAL if spur > V2C_SPUR_MARGINAL else V2C_SPUR_VIABLE
        parts.append(f"(4c) leak {100 * spur:.1f} % > {100 * limit:g} %")
    elif not math.isfinite(spur):
        parts.append("(4c) leak unknown")
    return "; ".join(parts) or "all four criteria met"


def _ring_role(r: Any) -> str:
    roles = []
    if r.is_central:
        roles.append("central")
    if r.is_in_focus:
        roles.append("in focus")
    if r.truncated_edge:
        roles.append(f"data cut at the {r.truncated_edge} edge")
    return ", ".join(roles)


def _head(report: ZQualityReport) -> Dict[str, Any]:
    return {"axon_id": report.axon_id, "source": report.source_name, "roi": report.roi_text,
            "rule": report.v2.rule, "p_ref_nm": report.v2.p_ref_nm, "computed_at": report.computed_at,
            "program": PROGRAM}


def ring_rows(report: ZQualityReport) -> List[Dict[str, Any]]:
    """One row per ring (bottom to top) with the numbers of the ring table."""
    out = []
    for r in report.v2.rings:
        row = dict(_head(report))
        row.update(ring=r.index, role=_ring_role(r), centre_z_nm=r.centre_z_nm, n_locs=r.n_locs,
                   f_of_central=r.f_of_central, k_clusters=r.k_clusters, locs_per_cluster=r.locs_per_cluster,
                   lpz_median_nm=r.lpz_median_nm, sigma_z_nm=r.sigma_z_nm, sigma_over_p=r.sigma_over_p,
                   sizer_peak=bool(r.peak), sizer_peak_z_nm=r.peak_z_nm, is_central=bool(r.is_central),
                   is_in_focus=bool(r.is_in_focus), truncated_edge=r.truncated_edge)
        out.append(row)
    return out


def pair_rows(report: ZQualityReport) -> List[Dict[str, Any]]:
    """One row per consecutive ring pair with the four criteria, the verdict and its reasons."""
    out = []
    for p in report.v2.pairs:
        row = dict(_head(report))
        row.update(pair=f"{p.ring_a}-{p.ring_b}", ring_a=p.ring_a, ring_b=p.ring_b, peak_a=bool(p.peak_a),
                   peak_b=bool(p.peak_b), valley=bool(p.valley), f_a=p.f_a, f_b=p.f_b, f_min=p.f_min, x=p.x,
                   x_in_range=bool(p.x_in_range), x_note=report.v2.x_note, count_ok=bool(p.count_ok),
                   exp_spur_frac=p.exp_spur_frac, d_sep=p.d_sep, d39_verdict=p.d39_verdict, verdict=p.verdict,
                   reasons=" | ".join(p.reasons))
        out.append(row)
    return out


def factor_rows(report: ZQualityReport) -> List[Dict[str, Any]]:
    """One row per limiting factor of the axon (``AxonViabilityV2.limiting_factors``)."""
    out = []
    for f in report.v2.limiting_factors:
        row = dict(_head(report))
        row.update(key=f.key, factor=f.label, value=f.value, needed=f.needed, status=f.status, source=f.source)
        out.append(row)
    return out


def profile_rows(report: ZQualityReport) -> List[Dict[str, Any]]:
    """The SiZer profile on its grid: z', smoothed density, slope t, and the significant sign (+1 / -1 / 0)."""
    prof = report.v2.profile
    if prof is None:
        return []
    out = []
    for g, f, t, s in zip(prof.grid, prof.fhat, prof.t, prof.sign):
        out.append({"axon_id": report.axon_id, "z_nm": float(g), "density_per_nm": float(f), "slope_t": float(t),
                     "sizer_sign": int(s), "q_simultaneous": float(prof.q), "h_nm": float(prof.h_nm)})
    return out


def plain_summary(report: ZQualityReport) -> List[str]:
    """The axon's verdicts and limiting factors in plain English, for the microscopy team (the text of the
    window's summary and of the exported ``_summary.txt``)."""
    v2 = report.v2
    P = float(v2.p_ref_nm)
    rings, pairs = list(v2.rings), list(v2.pairs)
    n_v = sum(1 for p in pairs if p.verdict == "viable")
    n_m = sum(1 for p in pairs if p.verdict == "marginal")
    lines: List[str] = []
    name = os.path.basename(report.source_name) or "this axon"
    lines.append(f"Z QUALITY OF {name}" + (f" ({report.roi_text})" if report.roi_text else ""))
    lines.append(f"{report.n_locs:,} localizations, {len(rings)} ring(s), {len(pairs)} consecutive pair(s); "
                 f"rule v2 (D-41), P = {P:.1f} nm, SiZer h = {v2.h_nm:.0f} nm, alpha {v2.alpha:g}.")
    lines.append("")
    if not pairs:
        lines.append(marked("bad", "No ring pair: with fewer than two rings there is nothing to compare."))
    elif n_v:
        ok = ", ".join(f"{a}-{b}" for a, b in v2.viable_pairs)
        lines.append(marked("good", f"{n_v} of {len(pairs)} pair(s) VIABLE ({ok}): the column test can run on them."))
        if n_m:
            lines.append(marked("warn", f"{n_m} MARGINAL pair(s) ({', '.join(f'{a}-{b}' for a, b in v2.marginal_pairs)})"
                                ": analysed only as a sensitivity check."))
    elif n_m:
        lines.append(marked("warn", f"No VIABLE pair; {n_m} MARGINAL pair(s): only a sensitivity analysis is possible."))
    else:
        lines.append(marked("bad", f"No pair passes the rule: none of the {len(pairs)} consecutive pair(s) can enter "
                                   "the column test."))
    for p in pairs:
        text, _kind = verdict_text(p.verdict)
        why = "; ".join(p.reasons) if p.reasons else "all four criteria met"
        lines.append(f"   rings {p.ring_a}-{p.ring_b}: {text} -- {why}")
    lines.append("")
    lines.append("WHAT LIMITS Z QUALITY HERE")
    for f in v2.limiting_factors:
        kind = _MARK_OF_STATUS.get(str(f.status), "dim")
        lines.append(marked(kind, f"{f.label}: {f.value}"))
        lines.append(f"      needed: {f.needed}")
    lines.append("")
    lines.append("WHAT WOULD HELP (for the microscopy team; the evidence of each line in [brackets])")
    lines.extend(f"   * {t}" for t in _what_would_help(report))
    lines.append("")
    lines.append("FOR NEW ACQUISITIONS (learnt on the exploratory dataset, not on this axon)")
    lines.extend(f"   * {t}" for t in MEASURED_LEVERS)
    lines.append("")
    lines.append("HOW THE VERDICT IS MADE (rule v2, D-41)")
    lines.append(f"   (1) each ring has a significant SiZer peak within P/3 = {P / 3:.0f} nm of its centre;")
    lines.append("   (2) a significant valley separates the two rings (the density falls, then rises, between them);")
    lines.append("   (3) each ring holds at least f_min(x) of the CENTRAL ring's localizations, x = localizations per "
                 "cluster of the central ring / min_samples, f_min = 1.46 - 0.336 ln x (validated for x 5.8-13.3);")
    lines.append("   (4) the expected axial-leak copies (D-39) are <= 2 % (VIABLE) or <= 5 % (MARGINAL, sensitivity only).")
    lines.append(f"   Central ring: {v2.central_index if v2.central_index is not None else 'none'} (Q-35: the middle "
                 "ring; with 2 rings, or 2 middle rings, the more populated). SiZer groups: "
                 f"{v2.groups_source}, {v2.n_groups:,} units.")
    lines.append("")
    lines.append(NOT_CALIBRATED_LINE)
    if report.warnings:
        lines.append("")
        lines.append("Warnings: " + " | ".join(dict.fromkeys(report.warnings)))
    return lines


def _what_would_help(report: ZQualityReport) -> List[str]:
    """Concrete, numeric statements of what the rule would need on this axon (never a promise); every line names
    its evidence in [brackets]. What the exploratory dataset taught about the acquisition is ``MEASURED_LEVERS``."""
    from tools.mps_axial_precision import LF_SIGMA_OVER_P_NONE, LF_SIGMA_OVER_P_VALLEY

    v2 = report.v2
    P = float(v2.p_ref_nm)
    out: List[str] = []
    rings, pairs = list(v2.rings), list(v2.pairs)
    if len(rings) < 2:
        out.append("A selection with at least two rings in z (a longer z range, or a thicker section of the axon). "
                   "[rule v2, D-41]")
        return out
    by = {r.index: r for r in rings}
    pair_sig = [(math.sqrt(0.5 * (by[p.ring_a].sigma_z_nm ** 2 + by[p.ring_b].sigma_z_nm ** 2)), p) for p in pairs]
    pair_sig = [(s, p) for s, p in pair_sig if math.isfinite(s)]
    sig = [s for s, _p in pair_sig]
    lpz = [r.lpz_median_nm for r in rings if math.isfinite(r.lpz_median_nm)]
    if sig:
        best = min(sig)
        target = LF_SIGMA_OVER_P_VALLEY * P
        bp = min(pair_sig, key=lambda sp: sp[0])[1]
        cut = [k for k in (bp.ring_a, bp.ring_b) if by[k].truncated_edge]
        shown = round(best / P, 3)          # judged on the value shown
        if shown > LF_SIGMA_OVER_P_VALLEY:
            out.append(f"Thinner rings in z: the axial width sigma_z of the best pair is ~{best:.1f} nm (sigma_z / P = "
                       f"{shown:.3f}; median lpz here "
                       + (f"{min(lpz):.0f}-{max(lpz):.0f} nm" if lpz else "unknown")
                       + f"). There is no hard threshold: in the simulations SiZer found the valley between two rings "
                       f"in ~80 % of simulated profiles (4000 localizations in the central ring) at sigma_z / P "
                       f"{LF_SIGMA_OVER_P_VALLEY:.2f} (sigma_z ~{target:.0f} nm here) "
                       f"when the weaker ring holds ~81 % of the central ring, and not in 80 % even with equal rings "
                       f"at {LF_SIGMA_OVER_P_NONE:.2f}. [zmin report sec. 2]")
        elif cut:
            out.append(f"The best pair ({bp.ring_a}-{bp.ring_b}) looks narrow (sigma_z / P = {shown:.3f}), but ring "
                       f"{', '.join(str(k) for k in cut)} sits at a cut z edge, where the fitted width is "
                       "underestimated: not evidence of enough axial resolution (the other pairs reach sigma_z / P = "
                       f"{max(sig) / P:.3f}); widen the z filter first. [zmin report sec. 4]")
        else:
            out.append(f"The axial width is where the simulations found valleys (sigma_z ~{best:.0f} nm, sigma_z / P = "
                       f"{shown:.3f}, at or below {LF_SIGMA_OVER_P_VALLEY:.2f}). [zmin report sec. 2]")
    low = [r for r in rings if math.isfinite(r.f_of_central) and math.isfinite(v2.f_min) and r.f_of_central < v2.f_min]
    if low:
        out.append("More balanced rings: " + ", ".join(f"ring {r.index} has {_pct(r.f_of_central)}" for r in low)
                   + f" of the central ring's localizations and needs >= {_pct(v2.f_min)} (x = {_num(v2.x)}"
                   + (f"; {v2.x_note}" if v2.x_note else "") + "). [rule v2 criterion 3; zmin report sec. 3]")
    if math.isfinite(v2.x) and not v2.x_in_range:
        out.append(f"Localizations per cluster outside the validated range (x = {v2.x:.1f}; validated 5.8-13.3): the "
                   "minimum % used is the conservative end of the formula. [zmin report sec. 3]")
    cut = [e for e in v2.z_edges if e.truncated]
    for e in cut:
        out.append(f"Do not cut the z range: the data end abruptly at the {e.side} ({e.z_nm:.0f} nm, {v2.z_edges_source}) "
                   f"while the density there is still {_pct(e.edge_frac)} of its maximum. Widen the z filter when "
                   "exporting from Picasso so the profile falls to ~0 before the data end; a cut edge ring's share is "
                   "underestimated and its 'peak' can be the cut itself. [zmin report sec. 4]")
    foc = [r for r in rings if r.is_in_focus]
    cen = [r for r in rings if r.is_central]
    if foc and cen and foc[0].index != cen[0].index:
        out.append(f"The ring in focus (smallest median lpz) is ring {foc[0].index}, not the central ring "
                   f"{cen[0].index}: the axial precision is best at ring {foc[0].index}, not in the middle of the stack "
                   "(information). [zmin report sec. 1]")
    sp = [p.exp_spur_frac for p in pairs if math.isfinite(p.exp_spur_frac)]
    if sp and min(sp) > 0.02:
        out.append(f"Less axial leak: even the best pair expects {100 * min(sp):.1f} % of its clusters to be copies "
                   "leaked from the neighbouring ring (<= 2 % needed); it falls with a smaller sigma_z. [D-39]")
    if not out:
        out.append("Nothing limits this axon under rule v2.")
    return out


def _safe_stem(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", os.path.splitext(os.path.basename(text))[0]).strip("_.-")
    return s[:80] or "axon"


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if bool(v) else "False"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return "" if not math.isfinite(f) else repr(f)
    return str(v)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    cols: List[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in cols])


def write_report_files(report: ZQualityReport, folder: str, *, stem: Optional[str] = None,
                       images: Optional[Dict[str, Callable[[str], bool]]] = None) -> List[str]:
    """
    Write the report into ``folder``: ``<stem>_zquality_{rings,pairs,factors,profile}.csv``,
    ``<stem>_zquality_summary.txt`` and one PNG per entry of ``images`` (suffix -> a function that saves the PNG
    at the path it is given). Never overwrites: when any of the files exists, ``_2``, ``_3``... is added to the
    stem. Returns the paths written.
    """
    os.makedirs(folder, exist_ok=True)
    base = _safe_stem(stem) if stem else (
        f"{_safe_stem(report.source_name)[:60]}_{report.axon_id}" if report.source_name else _safe_stem(report.axon_id))
    suffixes = ["rings.csv", "pairs.csv", "factors.csv", "profile.csv", "summary.txt"] + [
        f"{k}.png" for k in (images or {})]
    k = 1
    while True:
        cand = base if k == 1 else f"{base}_{k}"
        if not any(os.path.exists(os.path.join(folder, f"{cand}_zquality_{s}")) for s in suffixes):
            break
        k += 1
    pre = os.path.join(folder, f"{cand}_zquality_")
    written: List[str] = []
    for name, rows in (("rings.csv", ring_rows(report)), ("pairs.csv", pair_rows(report)),
                       ("factors.csv", factor_rows(report)), ("profile.csv", profile_rows(report))):
        _write_csv(pre + name, rows)
        written.append(pre + name)
    with open(pre + "summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(plain_summary(report)) + "\n")
        fh.write(f"\nComputed {report.computed_at} by {PROGRAM}; {report.lab_source}.\n")
    written.append(pre + "summary.txt")
    for suffix, saver in (images or {}).items():
        path = pre + f"{suffix}.png"
        if saver(path):
            written.append(path)
    return written


def write_cluster_report_files(report: ZQualityReport, folder: str, *, stem: Optional[str] = None,
                               images: Optional[Dict[str, Callable[[str], bool]]] = None) -> List[str]:
    """
    The cluster panel's files (NEW files, beside -- never instead of -- ``write_report_files``'):
    ``<stem>_zquality_clusters_medians.csv`` (one row per cluster of both cluster sets) and one
    ``<stem>_zquality_clusters_<suffix>.png`` per entry of ``images``. ``stem`` is used as given (the stem
    ``write_report_files`` chose, so the files sit together); without it, the same default stem. Never overwrites
    (``_2``, ``_3``... as there). Returns the paths written; nothing when the report has no cluster profile.
    """
    if report.clusters is None:
        return []
    os.makedirs(folder, exist_ok=True)
    if stem:
        base = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(stem)).strip("_") or "axon"
    else:
        base = (f"{_safe_stem(report.source_name)[:60]}_{report.axon_id}" if report.source_name
                else _safe_stem(report.axon_id))
    suffixes = ["clusters_medians.csv"] + [f"clusters_{k}.png" for k in (images or {})]
    k = 1
    while True:
        cand = base if k == 1 else f"{base}_{k}"
        if not any(os.path.exists(os.path.join(folder, f"{cand}_zquality_{s}")) for s in suffixes):
            break
        k += 1
    pre = os.path.join(folder, f"{cand}_zquality_")
    head = {"axon_id": report.axon_id, "source": report.source_name, "roi": report.roi_text,
            "p_ref_nm": report.clusters.p_ref_nm, "computed_at": report.computed_at, "program": PROGRAM}
    _write_csv(pre + "clusters_medians.csv", cluster_median_rows(report.clusters, head))
    written = [pre + "clusters_medians.csv"]
    for suffix, saver in (images or {}).items():
        path = pre + f"clusters_{suffix}.png"
        if saver(path):
            written.append(path)
    return written


def write_v2c_report_files(report: ZQualityReport, folder: str, *, stem: Optional[str] = None) -> List[str]:
    """
    Rule v2-clusters' file (EXPLORATORY, D-42; a NEW file beside the others, never instead of them):
    ``<stem>_zquality_clusters_pairs.csv``, one row per pair x cluster set x bandwidth (h 0.10 P: the verdict; h 0.20
    P: sensitivity only), ``tools.mps_axial_clusters.v2c_pair_rows``. ``stem`` as in ``write_cluster_report_files``.
    Never overwrites (``_2``, ``_3``...). Nothing when the report has no rule v2-clusters.
    """
    if report.v2c is None:
        return []
    os.makedirs(folder, exist_ok=True)
    if stem:
        base = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(stem)).strip("_") or "axon"
    else:
        base = (f"{_safe_stem(report.source_name)[:60]}_{report.axon_id}" if report.source_name
                else _safe_stem(report.axon_id))
    k = 1
    while True:
        cand = base if k == 1 else f"{base}_{k}"
        if not os.path.exists(os.path.join(folder, f"{cand}_zquality_clusters_pairs.csv")):
            break
        k += 1
    path = os.path.join(folder, f"{cand}_zquality_clusters_pairs.csv")
    head = {"axon_id": report.axon_id, "source": report.source_name, "roi": report.roi_text,
            "p_ref_nm": report.v2c.p_ref_nm, "computed_at": report.computed_at, "program": PROGRAM}
    _write_csv(path, v2c_pair_rows(report.v2c, head))
    return [path]


# ============================================================================
# The window
# ============================================================================

class _Relay(QtCore.QObject):
    """A worker's news into the GUI thread (never parented: a worker that outlives its window still emits)."""

    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(int, object, object)


def _table(columns: Sequence[str], name: str, headers: Optional[Sequence[str]] = None) -> QtWidgets.QTableWidget:
    t = QtWidgets.QTableWidget(0, len(columns))
    t.setObjectName(name)
    t.setHorizontalHeaderLabels(list(headers or columns))
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    t.setWordWrap(True)
    return t


RING_COLUMNS: Tuple[str, ...] = (
    "Ring", "Role", "Centre z' [nm]", "Localizations", "% of central", "Clusters", "Locs / cluster",
    "Median lpz [nm]", "sigma_z [nm]", "sigma_z / P", "SiZer peak")
# what the header shows (two lines keep the columns narrow; the names above stay the lookup keys)
RING_HEADERS: Tuple[str, ...] = (
    "Ring", "Role", "Centre z'\n[nm]", "Localizations", "% of\ncentral", "Clusters", "Locs /\ncluster",
    "Median\nlpz [nm]", "sigma_z\n[nm]", "sigma_z\n/ P", "SiZer\npeak")
# "Why" is last: it stretches over the rest of the width and wraps (the full text is also its tooltip)
PAIR_COLUMNS: Tuple[str, ...] = (
    "Pair", "(1) Peaks", "(2) Valley", "(3) % of central", "(3) Minimum f_min(x)", "(4) Leak copies",
    "Verdict (rule v2)", "D-39 (old rule)", "Why")
PAIR_HEADERS: Tuple[str, ...] = (
    "Pair", "(1)\nPeaks", "(2)\nValley", "(3) % of\ncentral", "(3) minimum\nf_min(x)", "(4) Leak\ncopies",
    "Verdict\n(rule v2)", "D-39\n(old rule)", "Why")
# Rule v2-clusters (EXPLORATORY, D-42): its own table, never inside the one above
V2C_COLUMNS: Tuple[str, ...] = (
    "Pair", "Cluster set", "(1c) Peaks", "(2c) Valley", "Valley in slab gap", "(3c) K / K central",
    "(4c) Leak (clusters)", "D-39 leak (rule v2)", "(1c)+(2c) at h 0.20 P", "Verdict (v2-clusters)", "Why")
V2C_HEADERS: Tuple[str, ...] = (
    "Pair", "Cluster set", "(1c)\nPeaks", "(2c)\nValley", "Valley in\nslab gap", "(3c) K /\nK central",
    "(4c) Leak\n(clusters)", "D-39 leak\n(rule v2)", "(1c)+(2c)\nh 0.20 P", "Verdict\n(v2-clusters)", "Why")


class _ResizeWatcher(QtCore.QObject):
    """Calls ``callback`` (deferred) whenever the watched widget is resized (rule v2-clusters' panel height)."""

    def __init__(self, callback: Callable[[], None]) -> None:
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if event.type() == QtCore.QEvent.Type.Resize:
            QtCore.QTimer.singleShot(0, self.callback)
        return False


class _FillHolder(QtWidgets.QWidget):
    """The scroll area's widget: holds the view and gives it all its own area. It has no layout on purpose: a layout
    with a word-wrapped label reports a height-for-width, and QScrollArea then sizes its widget to the PREFERRED height
    of everything (1360 px for a 1020 px window), which scrolled even where the view fits."""

    def __init__(self, child: QtWidgets.QWidget) -> None:
        super().__init__()
        self.child = child
        child.setParent(self)

    def resizeEvent(self, event: Any) -> None:
        self.child.setGeometry(self.rect())
        super().resizeEvent(event)


class ZQualityWindow(QtWidgets.QMainWindow):
    """
    The z-quality view of one axon (module docstring). ``start(job)`` runs
    ``job(progress) -> ZQualityReport`` in a worker thread and fills the
    view when it is done; ``set_report`` fills it directly. ``report`` is
    the report on screen (None while computing or after an error).
    ``columns_callback``: shown as "Columns review..." when given.
    """

    report_ready = QtCore.pyqtSignal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, title_source: str = "",
                 columns_callback: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(parent)
        self.report: Optional[ZQualityReport] = None
        self.error: Optional[str] = None
        self.columns_callback = columns_callback
        self.signature: Any = None          # what the opener uses to recognise the same axon
        self._generation = 0
        self._thread: Optional[threading.Thread] = None
        self._relay = _Relay()
        self._relay.progress.connect(self._on_progress)
        self._relay.finished.connect(self._on_finished)
        self.setWindowTitle(f"Z quality - {os.path.basename(title_source) or 'axon'}")
        self.resize(1440, 960)
        central = QtWidgets.QWidget()
        # H6 clusters: the view sits in a scroll area. With the cluster panel the view needs CLUSTER_VIEW_MIN_HEIGHT
        # px; a shorter window (a 1366 x 768 laptop) scrolls instead of squeezing the plots below their axes.
        self.content_widget = central
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setObjectName("scroll_area")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.content_holder = _FillHolder(central)
        self.scroll_area.setWidget(self.content_holder)
        self.setCentralWidget(self.scroll_area)
        root = QtWidgets.QVBoxLayout(central)
        self.header_label = QtWidgets.QLabel("Computing the z quality...")
        self.header_label.setWordWrap(True)
        self.header_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.header_label)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)
        # H6 toggles (D-43): the program's selection, live; its height is ADDED to the view (``_update_min_height``)
        self.criteria: Optional[Any] = None
        self.criteria_error = ""
        self.selection_result: Optional[Any] = None
        self._tinted = False
        self._v2c_title_base = ""
        self.selection_widget = SelectionWidget(parent=central)
        self.selection_widget.state.changed.connect(self._on_selection)
        root.addWidget(self.selection_widget)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setObjectName("graphics")
        self.graphics.setBackground(PLOT_BG)
        self.profile_plot = self.graphics.addPlot(row=0, col=0)
        self.strip_plot = self.graphics.addPlot(row=1, col=0)
        for p in (self.profile_plot, self.strip_plot):
            style_dark(p)
        self.strip_plot.setXLink(self.profile_plot)
        self.strip_plot.setMaximumHeight(110)
        self.strip_plot.setYRange(-1.1, 1.1, padding=0)
        # the two plots share their x: the same left-axis width keeps z' aligned between them
        for p in (self.profile_plot, self.strip_plot):
            p.getAxis("left").setWidth(70)
        self.strip_plot.getAxis("left").setStyle(showValues=False)
        self.strip_plot.setLabel("left", "SiZer")
        self.strip_plot.setMouseEnabled(x=True, y=False)
        self.strip_plot.setLabel("bottom", "z' [nm]  (the axon's own axis, bottom to top)")
        self.profile_plot.setLabel("left", f"localizations per {HIST_BIN_NM:.0f} nm")
        set_title(self.profile_plot, "Axial profile")
        set_title(self.strip_plot, "SiZer: where the density rises or falls significantly (alpha 0.05, cluster-robust)")
        # The legend sits in its own column, beside the plots: never over the profile it explains. UI stage 0: a
        # click on an entry hides or shows the REAL items it names (groups included: every ring-centre line of both
        # plots and its label), and stays so through every redraw (_layer_state, per window)
        self.legend = pg.LegendItem(labelTextColor=AXIS_FG, brush=pg.mkBrush(0, 0, 0, 0))
        self._layer_state: Dict[str, bool] = {}
        self.legend_layers = LegendLayers(self._layer_state)
        self.cluster_legend_layers = LegendLayers(self._layer_state)
        self.graphics.addItem(self.legend, row=0, col=1, rowspan=2)
        split.addWidget(self.graphics)

        # H6 clusters (descriptive): the cluster medians in a plot area of their own UNDER the localization profile,
        # on the same z' axis (x-linked, same left-axis width, the legend column as wide as the one above), so the
        # localization view and its exported profile PNG keep exactly their own items.
        self.cluster_panel = QtWidgets.QWidget()
        self.cluster_panel.setObjectName("cluster_panel")
        cpl = QtWidgets.QVBoxLayout(self.cluster_panel)
        cpl.setContentsMargins(0, 0, 0, 0)
        cpl.setSpacing(2)
        crow = QtWidgets.QHBoxLayout()
        self.cluster_title = QtWidgets.QLabel("<b>Cluster medians</b> (one z' per cluster; descriptive, no verdict)."
                                              " Cluster set:")
        self.cluster_title.setObjectName("cluster_title")
        self.cluster_title.setWordWrap(True)
        crow.addWidget(self.cluster_title, stretch=3)
        self.cluster_source_combo = QtWidgets.QComboBox()
        self.cluster_source_combo.setObjectName("cluster_source_combo")
        for key in CLUSTER_SET_ORDER:
            self.cluster_source_combo.addItem(CLUSTER_SET_LABELS[key], key)
        self.cluster_source_combo.setCurrentIndex(CLUSTER_SET_ORDER.index("groups_3d"))
        self.cluster_source_combo.currentIndexChanged.connect(lambda *_a: self._draw_clusters())
        crow.addWidget(self.cluster_source_combo)
        self.cluster_note = QtWidgets.QLabel(CLUSTER_NOTE)
        self.cluster_note.setObjectName("cluster_note")
        self.cluster_note.setWordWrap(True)
        crow.addWidget(self.cluster_note, stretch=2)
        cpl.addLayout(crow)
        self.graphics_clusters = pg.GraphicsLayoutWidget()
        self.graphics_clusters.setObjectName("graphics_clusters")
        self.graphics_clusters.setBackground(PLOT_BG)
        self.cluster_plot = self.graphics_clusters.addPlot(row=0, col=0)
        self.cluster_strip = self.graphics_clusters.addPlot(row=1, col=0)
        for p in (self.cluster_plot, self.cluster_strip):
            style_dark(p)
            p.getAxis("left").setWidth(70)
            # FOLLOW the localization profile's z' range, never drive it: no setXLink (pyqtgraph links both ways, and
            # a range set here moved the plot above); zoom and pan the profile above, these two follow it
            p.enableAutoRange(x=False)
            p.setMouseEnabled(x=False, y=False)
        self.profile_plot.vb.sigXRangeChanged.connect(lambda *_a: self._follow_profile_x())
        self.cluster_strip.setMaximumHeight(62)
        # the z' ticks are on the strip underneath: the histogram keeps its height for the bars
        self.cluster_plot.getAxis("bottom").setStyle(showValues=False)
        self.cluster_plot.getAxis("bottom").setHeight(4)
        self.cluster_strip.setYRange(-1.1, 1.1, padding=0)
        self.cluster_strip.getAxis("left").setStyle(showValues=False)
        self.cluster_strip.setLabel("left", "SiZer")
        self.cluster_strip.setLabel("bottom", "z' [nm]  (median of each cluster's localizations)")
        # short: the plot is low, and a longer vertical label ran into the strip's (the bar width is in the legend)
        self.cluster_plot.setLabel("left", "clusters")
        self.cluster_legend = pg.LegendItem(labelTextColor=AXIS_FG, brush=pg.mkBrush(0, 0, 0, 0))
        self.graphics_clusters.addItem(self.cluster_legend, row=0, col=1, rowspan=2)
        cpl.addWidget(self.graphics_clusters, stretch=1)
        self.cluster_panel.setVisible(False)
        # a fixed height ADDED to the view (``_update_min_height``): the localization plots and the tables keep theirs
        self.cluster_panel.setFixedHeight(CLUSTER_PANEL_HEIGHT)
        split.addWidget(self.cluster_panel)
        # the legend column under the one above, whatever the window's width (keeps z' aligned between the panels)
        self.graphics.installEventFilter(self)
        self.graphics_clusters.installEventFilter(self)

        lower = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        tables = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tables)
        self._tables_layout = tl
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(QtWidgets.QLabel("<b>Rings</b> (bottom to top)"))
        self.ring_table = _table(RING_COLUMNS, "ring_table", RING_HEADERS)
        tl.addWidget(self.ring_table, stretch=2)
        self.pairs_title = QtWidgets.QLabel(PAIRS_TITLE)
        self.pairs_title.setObjectName("pairs_title")
        tl.addWidget(self.pairs_title)
        self.pair_table = _table(PAIR_COLUMNS, "pair_table", PAIR_HEADERS)
        self.pair_table.horizontalHeader().setStretchLastSection(True)
        # a wider or narrower window re-wraps the Why column (deferred: the signal comes before the new width is set)
        self.pair_table.horizontalHeader().sectionResized.connect(
            lambda *_a: QtCore.QTimer.singleShot(0, self.pair_table.resizeRowsToContents))
        tl.addWidget(self.pair_table, stretch=3)
        lower.addWidget(tables)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QtWidgets.QLabel("<b>What limits z quality here</b> (plain words, for the microscopy team)"))
        self.summary_view = QtWidgets.QPlainTextEdit()
        self.summary_view.setObjectName("summary_view")
        self.summary_view.setReadOnly(True)
        self.summary_view.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        rl.addWidget(self.summary_view, stretch=1)
        buttons = QtWidgets.QHBoxLayout()
        self.export_button = QtWidgets.QPushButton("Export z-quality report...")
        self.export_button.setObjectName("export_button")
        self.export_button.clicked.connect(self._on_export)
        self.export_button.setEnabled(False)
        buttons.addWidget(self.export_button)
        self.columns_button = QtWidgets.QPushButton("Columns review...")
        self.columns_button.setObjectName("columns_button")
        self.columns_button.clicked.connect(self._on_columns)
        self.columns_button.setVisible(columns_callback is not None)
        buttons.addWidget(self.columns_button)
        buttons.addStretch(1)
        rl.addLayout(buttons)
        lower.addWidget(right)
        lower.setSizes([int(self.width() * 0.66), int(self.width() * 0.34)])
        split.addWidget(lower)
        split.setSizes([int(self.height() * SPLIT_SHARES[0]), 0, int(self.height() * SPLIT_SHARES[1])])
        # a taller or shorter view is shared by the localization plots and the tables in the same proportion
        split.setStretchFactor(0, int(round(100 * SPLIT_SHARES[0])))
        split.setStretchFactor(1, 0)
        split.setStretchFactor(2, int(round(100 * SPLIT_SHARES[1])))
        self.split = split
        self._cluster_split_done = False
        # Rule v2-clusters (EXPLORATORY, D-42): a full-width panel of its own UNDER the splitter, outside it: the view
        # grows by exactly the panel's height, so rule v2's plots and tables (the splitter) keep the height they have
        # without it; hidden until computed
        self.v2c_panel = QtWidgets.QWidget()
        self.v2c_panel.setObjectName("v2c_panel")
        vpl = QtWidgets.QVBoxLayout(self.v2c_panel)
        vpl.setContentsMargins(0, 0, 0, 0)
        vpl.setSpacing(2)
        self.v2c_title = QtWidgets.QLabel(
            f"<b>Clusters (exploratory)</b>: {V2C_RULE}, the same criteria (1c)-(4c) on the clusters; never selects "
            "pairs under the pre-specified rule, only as an exploratory selection (Q-38, D-43)")
        self.v2c_title.setObjectName("v2c_title")
        self.v2c_title.setWordWrap(True)
        vpl.addWidget(self.v2c_title)
        self.v2c_table = _table(V2C_COLUMNS, "v2c_table", V2C_HEADERS)
        self.v2c_table.horizontalHeader().setStretchLastSection(True)
        # a wider or narrower window re-wraps the Why column: the rows and the panel's height follow
        self.v2c_table.horizontalHeader().sectionResized.connect(
            lambda *_a: QtCore.QTimer.singleShot(0, self._v2c_fit_height))
        vpl.addWidget(self.v2c_table, stretch=1)
        self.v2c_panel.setVisible(False)
        self._v2c_extra = 0
        root.addWidget(split, stretch=1)
        root.addWidget(self.v2c_panel, stretch=0)
        # the scroll area's visible height sets how tall the view is: watched so the panel always comes ON TOP of it
        self._v2c_watch = _ResizeWatcher(self._v2c_update_min_height)
        self.scroll_area.viewport().installEventFilter(self._v2c_watch)
        self.missing_tooltips = (apply_tooltips(self, ZQUALITY_WINDOW_TOOLTIPS)
                                 + list(self.selection_widget.missing_tooltips))
        self.content_holder.setMinimumSize(central.minimumSizeHint())
        self._update_min_height()

    # ------------------------------------------------------------ the worker
    def start(self, job: Callable[[Callable[[str], None]], ZQualityReport]) -> None:
        """Run ``job(progress)`` in a worker thread; the view fills when it returns."""
        self._generation += 1
        gen, relay = self._generation, self._relay

        def say(text: str) -> None:
            relay.progress.emit(gen, str(text))

        def body() -> None:
            try:
                rep = job(say)
            except Exception as exc:  # noqa: BLE001 - shown in the window
                relay.finished.emit(gen, None, exc)
                return
            relay.finished.emit(gen, rep, None)

        self.report = None
        self.error = None
        self.export_button.setEnabled(False)
        self.progress_bar.setFormat("Computing the z quality...")
        self.progress_bar.setVisible(True)
        self._thread = threading.Thread(target=body, daemon=True, name="z-quality")
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None

    def wait(self, timeout: float = 300.0) -> None:
        """Block until the computation is shown (for scripts and tests)."""
        t_end = time.perf_counter() + float(timeout)
        while self._thread is not None and time.perf_counter() < t_end:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
        QtWidgets.QApplication.processEvents()

    def _on_progress(self, gen: int, text: str) -> None:
        if gen == self._generation:
            self.progress_bar.setFormat(text)

    def _on_finished(self, gen: int, rep: Any, error: Any) -> None:
        # A slot: an exception escaping it would abort the application.
        if gen != self._generation:
            return
        self._thread = None
        self.progress_bar.setVisible(False)
        if error is None and rep is not None:
            try:
                self.set_report(rep)
            except Exception as exc:  # noqa: BLE001 - reported below
                error = exc
            else:
                return
        self.error = f"{type(error).__name__}: {error}" if error is not None else "no result"
        self.header_label.setText(marked("bad", f"The z quality could not be computed: {self.error}"))
        self.summary_view.setPlainText(marked("bad", f"The z quality could not be computed:\n{self.error}"))

    # ------------------------------------------------------------ filling the view
    def set_report(self, rep: ZQualityReport) -> None:
        self.report = rep
        v2 = rep.v2
        n_v, n_m = len(v2.viable_pairs), len(v2.marginal_pairs)
        n_p = len(v2.pairs)
        state = (marked("good", f"{n_v} viable pair(s)") if n_v else
                 marked("warn", "no viable pair, marginal only") if n_m else marked("bad", "no viable pair"))
        self.header_label.setText(
            f"<b>{os.path.basename(rep.source_name) or 'axon'}</b>" + (f" ({rep.roi_text})" if rep.roi_text else "")
            + f" &nbsp;|&nbsp; {rep.n_locs:,} localizations, {len(v2.rings)} ring(s), {n_p} pair(s): <b>{state}</b>"
            + f" &nbsp;|&nbsp; rule v2 (D-41); geometry only, no column statistic; {rep.lab_source}; "
            + f"{rep.seconds:.1f} s")
        self._draw(rep)
        self._fill_rings(rep)
        self._fill_pairs(rep)
        self.summary_view.setPlainText("\n".join(plain_summary(rep)))
        self._show_clusters(rep)
        self._show_v2c(rep)
        self._set_criteria(rep)
        self._apply_selection()
        self.export_button.setEnabled(True)
        self.report_ready.emit(rep)

    # ------------------------------------------------------------ the cluster panel (H6 clusters, descriptive)
    def _show_clusters(self, rep: ZQualityReport) -> None:
        """Show the cluster panel when the report has a cluster profile (hide it otherwise); the first time, give it
        its share of the height: profile 0.33, clusters 0.22, tables 0.45."""
        has = getattr(rep, "clusters", None) is not None
        self.cluster_panel.setVisible(has)
        # the holder has no layout: its minimum is what makes the scroll area scroll (never below the view's own)
        self._update_min_height()
        if not has:
            if getattr(rep, "clusters_error", ""):
                self.cluster_note.setText(f"Cluster medians not computed: {rep.clusters_error}")
            return
        self.cluster_note.setText(CLUSTER_NOTE)
        if not self._cluster_split_done:
            # H6 toggles: the panel ADDS its fixed height; the localization plots and the tables share the rest as
            # they did before it (SPLIT_SHARES), so rule v2's profile and pair table keep their ff2ce46 heights
            rest = max(sum(self.split.sizes()) - CLUSTER_PANEL_HEIGHT, 0) or int(self.height() * 0.9)
            self.split.setSizes([int(rest * SPLIT_SHARES[0]), CLUSTER_PANEL_HEIGHT, int(rest * SPLIT_SHARES[1])])
            self._cluster_split_done = True
        self._draw_clusters()

    def _extra_height(self) -> int:
        """What the panels added below and above rule v2's plots and tables need: the cluster panel (inside the
        splitter), the criteria switches above it, the rule v2-clusters table under it."""
        lay = self.content_widget.layout()
        gap = max(int(lay.spacing()), 0) if lay is not None else 0
        extra = 0
        if not self.cluster_panel.isHidden():
            extra += CLUSTER_PANEL_HEIGHT + max(int(self.split.handleWidth()), 0)
        sw = self.selection_widget
        if not sw.isHidden():
            width = max(int(self.content_widget.width()) - 20, 400)
            h = sw.layout().heightForWidth(width) if sw.layout() is not None and sw.layout().hasHeightForWidth() else -1
            extra += (h if h > 0 else sw.sizeHint().height()) + gap
        if not self.v2c_panel.isHidden() and self._v2c_extra:
            extra += int(self._v2c_extra) + gap
        return extra

    def _update_min_height(self) -> None:
        """The view's height: the visible height (at least ``VIEW_BASE_MIN_HEIGHT``) for the header, rule v2's plots
        and tables -- as before the cluster work -- PLUS every panel added since (``_extra_height``); the scroll area
        reaches them. Never below the view's own minimum."""
        try:
            hint = self.content_widget.minimumSizeHint()
            base = max(VIEW_BASE_MIN_HEIGHT, int(self.scroll_area.viewport().height()))
            want = max(hint.height(), base + self._extra_height())
            width = max(hint.width(), 0)
            if self.content_holder.minimumHeight() != want or self.content_holder.minimumWidth() != width:
                self.content_holder.setMinimumSize(width, want)
        except RuntimeError:        # the window is being destroyed
            return

    # ------------------------------------------------------------ rule v2-clusters (EXPLORATORY, D-42)
    def _show_v2c(self, rep: ZQualityReport) -> None:
        """Show the rule v2-clusters table when the report has it (hide it otherwise) and append its block to the
        summary view (rule v2's summary text above it, and its exported file, stay exactly as they are)."""
        v2c = getattr(rep, "v2c", None)
        has = v2c is not None
        err = str(getattr(rep, "v2c_error", "") or "")
        self.v2c_panel.setVisible(has or bool(err))
        self.v2c_table.setVisible(has)
        if not has:
            if err:
                self.v2c_title.setText(f"<b>Clusters (exploratory)</b>: {V2C_RULE} not computed: {err}")
            self._v2c_fit_height()
            return
        self.v2c_title.setText(
            f"<b>Clusters (exploratory)</b>: {V2C_RULE}, the same criteria (1c)-(4c) on the clusters (one point "
            f"per cluster); <b>never selects pairs</b> under the pre-specified rule -- only as an exploratory "
            f"selection of the switches above (Q-38, D-43)")
        self._v2c_title_base = self.v2c_title.text()
        self._fill_v2c(rep)
        lines = plain_summary(rep) + [""] + v2c_summary_lines(v2c)
        self.summary_view.setPlainText("\n".join(lines))
        self._v2c_fit_height()
        # once the panel has its width the rows re-wrap: measure again
        QtCore.QTimer.singleShot(0, self._v2c_fit_height)

    def _v2c_fit_height(self) -> None:
        """The rule v2-clusters panel's height: the title and all the table's rows (the table at most
        ``V2C_VIEW_EXTRA_HEIGHT``); then the view's height (``_v2c_update_min_height``)."""
        try:
            if self.v2c_panel.isHidden() or self.report is None:
                if self._v2c_extra and self.report is not None:
                    # hidden again: the view's height without it
                    self._v2c_extra = 0
                    self._update_min_height()
                self._v2c_extra = 0
                return
            t = self.v2c_table
            table_h = 0
            if not t.isHidden():
                t.resizeRowsToContents()
                need = (t.horizontalHeader().sizeHint().height() + sum(t.rowHeight(i) for i in range(t.rowCount()))
                        + 2 * t.frameWidth() + 4)
                table_h = min(need, V2C_VIEW_EXTRA_HEIGHT)
                t.setFixedHeight(table_h)
            # the title wraps on a narrow window: its height at the panel's width, fixed (no gap above the table)
            width = max(int(self.v2c_panel.width()), 300)
            title_h = self.v2c_title.heightForWidth(width)
            title_h = title_h if title_h > 0 else self.v2c_title.sizeHint().height()
            self.v2c_title.setFixedHeight(title_h)
            extra = table_h + title_h + 8
            self.v2c_panel.setFixedHeight(extra)
            self._v2c_extra = extra
            self._v2c_update_min_height()
        except RuntimeError:        # the window is being destroyed
            return

    def _v2c_update_min_height(self) -> None:
        """With the rule v2-clusters panel shown, the view is as tall as it would be without it (the scroll area's
        visible height, or the cluster panel's minimum) PLUS the panel: the splitter above -- rule v2's plots and
        tables, and the cluster panel -- keeps exactly the height it has without the panel, and the panel is reached
        by scrolling down."""
        # H6 toggles: one rule for every added panel (``_update_min_height``)
        self._update_min_height()

    def _fill_v2c(self, rep: ZQualityReport) -> None:
        t = self.v2c_table
        t.setRowCount(0)
        v2c = rep.v2c
        sens = {(p.variant, p.ring_a, p.ring_b): p for p in v2c.sensitivity_pairs}
        for p in v2c.pairs:
            var = v2c.variants[p.variant]
            i = t.rowCount()
            t.insertRow(i)
            vtext, vkind = verdict_text(p.verdict)
            forced = p.variant in V2C_CUT_FORCED
            s2 = sens.get((p.variant, p.ring_a, p.ring_b))
            ks = "/".join(str(r.k_clusters) for r in var.rings)
            set_tip = (f"{var.label}: {var.n_points} points, K by ring {ks}"
                       + (f", {var.n_outside} outside every slab" if var.n_outside else "") + f".\n{var.params}")
            if not p.valley_raw:
                valley = ("NO", "bad")
            elif p.valley_at_cut and forced:
                valley = ("in gap: not counted", "bad")
            else:
                valley = ("yes" + (" (flagged)" if p.valley_at_cut else ""), "good")
            gap = ("yes (built in)" if forced else "yes (flag only)") if p.valley_at_cut else "no"
            gap_tip = ("valleys between the centres: " + (", ".join(f"{v:.0f} nm" for v in p.valleys_between) or "none")
                       + "; slab gap +-15 nm counts as 'in the gap'")
            spur, d39 = float(p.exp_spur_frac_clusters), float(p.exp_spur_frac_d39)
            c12 = (s2 is not None and s2.peak_a and s2.peak_b and s2.valley)
            why = "; ".join(p.reasons) if p.reasons else "all four criteria met"
            cells = [
                (f"{p.ring_a}-{p.ring_b}", "", None),
                (V2C_SHORT_NAMES.get(p.variant, var.letter), set_tip, None),
                (f"{'yes' if p.peak_a else 'NO'} / {'yes' if p.peak_b else 'NO'}", "",
                 "good" if (p.peak_a and p.peak_b) else "bad"),
                (valley[0], gap_tip, valley[1]),
                (gap, gap_tip, ("bad" if forced else "warn") if p.valley_at_cut else None),
                (f"{_pct(p.k_ratio_a)} / {_pct(p.k_ratio_b)}",
                 f"K {p.k_a} / {p.k_b} clusters; central ring {var.k_central}; minimum {_pct(V2C_K_MIN)} of it",
                 "good" if p.count_ok else "bad"),
                (f"{100 * spur:.1f} %" if math.isfinite(spur) else "n/a",
                 "cluster-level expected leak copies (tnfix, shared by the three cluster sets): <= 2 % viable, <= 5 % "
                 "marginal", "good" if spur <= 0.02 else ("warn" if spur <= 0.05 else "bad")),
                (f"{100 * d39:.1f} %" if math.isfinite(d39) else "n/a", "rule v2's criterion (4), D-39, for "
                 "comparison", None),
                ("yes" if c12 else "NO", f"(1c) and (2c) at h = {V2C_H_TABLE:g} P (information only, never the "
                 "verdict)", None),
                (f"v2-clusters: {vtext}", f"{V2C_RULE}: {why}", vkind),
                (short_why_v2c(p), why, None),
            ]
            for j, (txt, tip, kind) in enumerate(cells):
                t.setItem(i, j, self._item(txt, tip, kind))
        why_col = V2C_COLUMNS.index("Why")
        for j in range(len(V2C_COLUMNS)):
            if j != why_col:
                t.resizeColumnToContents(j)
        t.resizeRowsToContents()

    # ------------------------------------------------------------ the criteria switches (H6 toggles, D-43)
    def _set_criteria(self, rep: ZQualityReport) -> None:
        """The per-pair criteria of both rules from the report's own v2 / v2c objects (nothing recomputed)."""
        self.criteria, self.criteria_error = None, ""
        try:
            v2c = getattr(rep, "v2c", None)
            err = str(getattr(rep, "v2c_error", "") or "") or ("" if v2c is not None else "not computed")
            self.criteria = msel.axon_criteria(rep.v2, v2c, err, axon_id=str(rep.axon_id))
        except Exception as exc:  # noqa: BLE001 - shown beside the switches; the view itself is unaffected
            self.criteria_error = f"{type(exc).__name__}: {exc}"
        crit = self.criteria
        if crit is not None and not crit.has_v2c:
            self.selection_widget.enable_v2c(False, f"rule v2c was not computed for this axon ({crit.v2c_error})")
        else:
            self.selection_widget.enable_v2c(True)

    def _on_selection(self, _spec: Any = None) -> None:
        # a slot of the program-wide state: never raises (it would abort the application)
        try:
            self._apply_selection()
        except RuntimeError:        # the window is being destroyed
            return
        except Exception as exc:  # noqa: BLE001 - shown
            self.selection_widget.set_summary(f"could not be evaluated here: {type(exc).__name__}: {exc}")

    def _apply_selection(self) -> None:
        """Evaluate the selection on this axon (microseconds) and show it: the summary beside the switches; under an
        exploratory selection the tables' rows of the pairs it selects tinted and their rings bold. Under the
        pre-specified rule the tables are exactly today's."""
        rep = self.report
        if rep is None:
            return
        spec = self.selection_widget.spec
        crit = self.criteria
        if crit is None:
            self.selection_result = None
            self.selection_widget.set_summary(f"not available here: {self.criteria_error or 'no criteria'}")
            return
        res = msel.evaluate_selection(crit, spec)
        self.selection_result = res
        # what the tint means goes in the switches' own line (a label above the tables would take their height)
        self.selection_widget.set_summary("this axon: " + res.summary() + (TINT_SUMMARY if spec.exploratory else ""))
        if self._tinted:
            # back to today's tables, then tint again
            self._fill_rings(rep)
            self._fill_pairs(rep)
            if getattr(rep, "v2c", None) is not None:
                self._fill_v2c(rep)
                self._v2c_fit_height()
            self._tinted = False
        if spec.exploratory:
            self._tint(rep, res)
            self._tinted = True
        if getattr(rep, "v2c", None) is not None and self._v2c_title_base:
            want = self._v2c_title_base + (" &mdash; " + TINT_NOTE if spec.exploratory and spec.uses_v2c else "")
            if self.v2c_title.text() != want:
                self.v2c_title.setText(want)
                self._v2c_fit_height()
        self._update_min_height()

    def _tint(self, rep: ZQualityReport, res: Any) -> None:
        by = {(int(p.ring_a), int(p.ring_b)): p for p in res.pairs}
        spec = res.spec

        def brush(verdict_: str) -> Optional[QtGui.QBrush]:
            if verdict_ not in SELECTION_TINT:
                return None
            role, alpha = SELECTION_TINT[verdict_]
            col = QtGui.QColor(ROLES[role])
            col.setAlpha(alpha)
            return QtGui.QBrush(col)

        def note(ps: Any) -> str:
            word = {"viable": "selected, VIABLE", "marginal": "selected, MARGINAL (sensitivity)"}.get(
                str(ps.verdict), "not selected")
            out = f"Exploratory selection {res.label} #{res.hash}: {word}"
            if ps.failing:
                out += "; fails " + ", ".join(ps.failing)
            if ps.unchecked_failing:
                out += "; not checked but would fail: " + ", ".join(ps.unchecked_failing)
            return out + "."

        def paint(table: QtWidgets.QTableWidget, row: int, ps: Any) -> None:
            b = brush(str(ps.verdict))
            text = note(ps)
            for j in range(table.columnCount()):
                it = table.item(row, j)
                if it is None:
                    continue
                if b is not None:
                    it.setBackground(b)
                tip = it.toolTip()
                it.setToolTip(text + ("\n\n" + tip if tip else ""))

        if spec.uses_v2:
            for i, p in enumerate(rep.v2.pairs):
                ps = by.get((int(p.ring_a), int(p.ring_b)))
                if ps is not None:
                    paint(self.pair_table, i, ps)
        v2c = getattr(rep, "v2c", None)
        if spec.uses_v2c and v2c is not None:
            key = msel.VARIANT_KEYS[spec.variant]
            for i, p in enumerate(v2c.pairs):
                if str(p.variant) != key:
                    continue
                ps = by.get((int(p.ring_a), int(p.ring_b)))
                if ps is not None:
                    paint(self.v2c_table, i, ps)
        rings = {r for ps in res.pairs if ps.selected for r in (int(ps.ring_a), int(ps.ring_b))}
        for i, r in enumerate(rep.v2.rings):
            if int(r.index) not in rings:
                continue
            for j in range(self.ring_table.columnCount()):
                it = self.ring_table.item(i, j)
                if it is not None:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                    it.setToolTip(f"Ring of a pair the exploratory selection {res.label} selects."
                                  + ("\n\n" + it.toolTip() if it.toolTip() else ""))

    def export_selection_report(self, folder: str, *, stem: Optional[str] = None) -> List[str]:
        """``<stem>_zquality_selection.csv`` (``tools.mps_selection.selection_pair_rows``: the selection's label and
        hash, the verdict, tier, failing and unchecked criteria, every criterion of both rules) -- ONLY under an
        exploratory selection; nothing under the pre-specified rule (the export is then exactly today's)."""
        if self.report is None:
            raise RuntimeError("no z-quality report on screen yet")
        spec = self.selection_widget.spec
        if spec.is_default or self.criteria is None:
            return []
        rep = self.report
        res = msel.evaluate_selection(self.criteria, spec)
        os.makedirs(folder, exist_ok=True)
        if stem:
            base = re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(stem)).strip("_") or "axon"
        else:
            base = (f"{_safe_stem(rep.source_name)[:60]}_{rep.axon_id}" if rep.source_name
                    else _safe_stem(rep.axon_id))
        k = 1
        while True:
            cand = base if k == 1 else f"{base}_{k}"
            if not os.path.exists(os.path.join(folder, f"{cand}_zquality_selection.csv")):
                break
            k += 1
        path = os.path.join(folder, f"{cand}_zquality_selection.csv")
        head = {"source": rep.source_name, "roi": rep.roi_text, "computed_at": rep.computed_at, "program": PROGRAM,
                "selection_rule_text": res.rule_text}
        rows = [dict(head, **r) for r in msel.selection_pair_rows(self.criteria, res, input_key=str(rep.axon_id))]
        _write_csv(path, rows)
        return [path]

    def cluster_source(self) -> str:
        """The cluster set on screen (a key of ``tools.mps_axial_clusters.CLUSTER_SET_LABELS``)."""
        key = self.cluster_source_combo.currentData()
        return str(key) if key in CLUSTER_SET_LABELS else "groups_3d"

    def _draw_clusters(self) -> None:
        cp, cs = self.cluster_plot, self.cluster_strip
        cp.clear()
        cs.clear()
        self.cluster_legend.clear()
        layers = self.cluster_legend_layers
        layers.clear()
        rep = self.report
        prof_all = getattr(rep, "clusters", None) if rep is not None else None
        if rep is None or prof_all is None:
            return
        s = prof_all.sets[self.cluster_source()]
        counts, edges = s.hist_counts, s.hist_edges
        bar_brush = pg.mkBrush(*rgba("centroid", 190))
        cl_bars = pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=counts, brush=bar_brush, pen=pg.mkPen(None))
        cp.addItem(cl_bars)
        layers.entry(self.cluster_legend, "cl_bars", f"cluster medians ({HIST_BIN_CLUSTERS_NM:.0f} nm bars)",
                     "scatter", [cl_bars], symbol="s", size=10, brush=bar_brush, pen=pg.mkPen(None))
        top = float(counts.max()) if counts.size and counts.max() > 0 else 1.0
        prof = s.profile
        if prof is not None:
            scale = s.n_clusters * HIST_BIN_CLUSTERS_NM
            kde_pen = pg.mkPen(ROLES["fit"], width=2.5)
            kde = pg.PlotDataItem(prof.grid, prof.fhat * scale, pen=kde_pen)
            cp.addItem(kde)
            kde_members: List[Any] = [kde]
            top = max(top, float(np.max(prof.fhat * scale)))
            for xs, sym in ((prof.peaks, "t1"), (prof.valleys, "t")):
                pts = np.asarray(xs, dtype=np.float64)
                ys = np.interp(pts, prof.grid, prof.fhat * scale) if pts.size else pts
                marks = pg.ScatterPlotItem(pts, ys, symbol=sym, size=13, brush=pg.mkBrush(neutral(dark=True)),
                                           pen=pg.mkPen("k", width=1))
                cp.addItem(marks)
                kde_members.append(marks)
            # the peaks and valleys of this panel have no entry of their own: they go with the smoothed curve
            layers.entry(self.cluster_legend, "cl_kde", f"smoothed (h = {prof.h_nm:.0f} nm)", "line", kde_members,
                         pen=kde_pen)
            step = float(np.median(np.diff(prof.grid))) if prof.grid.size > 1 else 1.0
            for sgn, (role_name, lo, hi, _label) in SIZER_STYLE.items():
                m = np.asarray(prof.sign) == sgn
                if m.any():
                    g = np.asarray(prof.grid)[m]
                    cs.addItem(pg.BarGraphItem(x0=g - step / 2, x1=g + step / 2, y0=np.full(g.size, lo),
                                               y1=np.full(g.size, hi), brush=pg.mkBrush(ROLES[role_name]),
                                               pen=pg.mkPen(None)))
            cs.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(neutral(dark=True), width=0.5)))
        # the slab cuts (where a ring cluster's median can never cross), then the ring centres as above
        col = neutral(dark=True)
        cuts = sorted({round(v, 6) for _i, lo, hi in prof_all.slabs for v in (lo, hi) if math.isfinite(v)})
        cut_pen = pg.mkPen(col, width=1.2, style=QtCore.Qt.PenStyle.DashDotLine)
        cut_lines: List[Any] = []
        for c in cuts:
            for plot in (cp, cs):
                line = pg.InfiniteLine(pos=float(c), angle=90, pen=cut_pen)
                plot.addItem(line)
                cut_lines.append(line)
        if cuts:
            layers.entry(self.cluster_legend, "cl_cut", "ring slab cut", "line", cut_lines, pen=cut_pen)
        centre_items: List[Any] = []
        for r in rep.v2.rings:
            style = QtCore.Qt.PenStyle.SolidLine if r.is_central else QtCore.Qt.PenStyle.DashLine
            a = pg.InfiniteLine(pos=float(r.centre_z_nm), angle=90,
                                pen=pg.mkPen(col, width=4 if r.is_central else 1.5, style=style))
            b = pg.InfiniteLine(pos=float(r.centre_z_nm), angle=90,
                                pen=pg.mkPen(col, width=2 if r.is_central else 1, style=style))
            cp.addItem(a)
            cs.addItem(b)
            txt = pg.TextItem(f"ring {r.index}\n{s.count_in_ring(r.index)} clusters", color=col, anchor=(-0.08, 0.0))
            txt.setPos(float(r.centre_z_nm), top * 1.75)
            cp.addItem(txt)
            centre_items.extend((a, b, txt))
        if len(rep.v2.rings):
            layers.entry(self.cluster_legend, "cl_ring_centre", "ring centre: clusters<br>with the median in its slab",
                         "line", centre_items, pen=pg.mkPen(col, width=2, style=QtCore.Qt.PenStyle.DashLine))
        cp.setYRange(0, top * 1.8, padding=0)
        self._follow_profile_x()
        n_out = int(np.count_nonzero(s.ring_index < 0))
        h_txt = f"h = {s.profile.h_nm:.0f} nm" if s.profile is not None else "no SiZer (< 2 clusters)"
        self.cluster_title.setText(
            f"<b>Cluster medians</b> (one z' per cluster; descriptive, no verdict): <b>{s.n_clusters:,} clusters</b>"
            + (f" ({n_out} outside every slab)" if n_out else "")
            + f"; SiZer under it iid, {h_txt}, alpha 0.05. Cluster set:")
        QtCore.QTimer.singleShot(0, self._align_cluster_columns)

    def _follow_profile_x(self) -> None:
        """The cluster plots take the localization profile's z' range (one way only)."""
        try:
            lo_x, hi_x = self.profile_plot.viewRange()[0]
            for p in (self.cluster_plot, self.cluster_strip):
                p.setXRange(lo_x, hi_x, padding=0)
        except (RuntimeError, AttributeError):     # during construction or destruction
            return

    def _align_cluster_columns(self) -> None:
        """Make the cluster panel's legend column exactly as wide as the localization panel's, so the two plots have
        the same width and the same z' falls at the same x on screen."""
        try:
            want = int(round(self.legend.geometry().width()))
            lay = self.graphics_clusters.ci.layout
            if want > 0 and int(round(lay.columnMinimumWidth(1))) != want:
                lay.setColumnFixedWidth(1, want)
        except RuntimeError:        # the window is being destroyed
            return

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if (event.type() == QtCore.QEvent.Type.Resize
                and obj in (getattr(self, "graphics", None), getattr(self, "graphics_clusters", None))):
            QtCore.QTimer.singleShot(0, self._align_cluster_columns)
        return bool(super().eventFilter(obj, event))

    def _draw(self, rep: ZQualityReport) -> None:
        v2, prof = rep.v2, rep.v2.profile
        pp, sp = self.profile_plot, self.strip_plot
        pp.clear()
        sp.clear()
        self.legend.clear()
        layers = self.legend_layers
        layers.clear()
        counts, edges = rep.hist_counts, rep.hist_edges
        bars = pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=counts, brush=pg.mkBrush(*rgba("locs", 170)),
                               pen=pg.mkPen(None))
        pp.addItem(bars)
        layers.entry(self.legend, "loc_bars", f"localizations ({HIST_BIN_NM:.0f} nm bars)", "scatter", [bars],
                     symbol="s", size=10, brush=pg.mkBrush(*rgba("locs", 170)), pen=pg.mkPen(None))
        top = float(counts.max()) if counts.size else 1.0
        scale = rep.n_locs * HIST_BIN_NM
        if prof is not None:
            kde_pen = pg.mkPen(ROLES["fit"], width=2.5)
            kde = pg.PlotDataItem(prof.grid, prof.fhat * scale, pen=kde_pen)
            pp.addItem(kde)
            layers.entry(self.legend, "loc_kde", f"smoothed density<br>(h = {prof.h_nm:.0f} nm = 0.10 P)", "line",
                         [kde], pen=kde_pen)
            top = max(top, float(np.max(prof.fhat * scale)))
            for xs, sym, label, key in ((prof.peaks, "t1", "significant peak", "peak"),
                                        (prof.valleys, "t", "significant valley", "valley")):
                pts = np.asarray(xs, dtype=np.float64)
                ys = np.interp(pts, prof.grid, prof.fhat * scale) if pts.size else pts
                item = pg.ScatterPlotItem(pts, ys, symbol=sym, size=13, brush=pg.mkBrush(neutral(dark=True)),
                                          pen=pg.mkPen("k", width=1))
                pp.addItem(item)
                layers.entry(self.legend, key, label, "scatter", [item], symbol=sym, size=11,
                             brush=pg.mkBrush(neutral(dark=True)), pen=pg.mkPen("k", width=1))
            step = float(np.median(np.diff(prof.grid))) if prof.grid.size > 1 else 1.0
            for s, (role_name, lo, hi, label) in SIZER_STYLE.items():
                m = np.asarray(prof.sign) == s
                sizer_members: List[Any] = []
                if m.any():
                    g = np.asarray(prof.grid)[m]
                    band = pg.BarGraphItem(x0=g - step / 2, x1=g + step / 2, y0=np.full(g.size, lo),
                                           y1=np.full(g.size, hi), brush=pg.mkBrush(ROLES[role_name]),
                                           pen=pg.mkPen(None))
                    sp.addItem(band)
                    sizer_members.append(band)
                # the legend always lists all three, present or not
                layers.entry(self.legend, SIZER_LAYER_KEYS[s], label, "scatter", sizer_members, symbol="s", size=10,
                             brush=pg.mkBrush(ROLES[role_name]), pen=pg.mkPen(None))
            sp.addItem(pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(neutral(dark=True), width=0.5)))
        else:
            for s, (role_name, _lo, _hi, label) in SIZER_STYLE.items():
                layers.entry(self.legend, SIZER_LAYER_KEYS[s], label, "scatter", [], symbol="s", size=10,
                             brush=pg.mkBrush(ROLES[role_name]), pen=pg.mkPen(None))
        # ring centres, labelled with their share of the central ring
        centre_items: List[Any] = []
        for r in v2.rings:
            # neutral, not a role: the bars (locs) and the density (fit) already hold sky blue and orange here
            col = neutral(dark=True)
            style = QtCore.Qt.PenStyle.SolidLine if r.is_central else QtCore.Qt.PenStyle.DashLine
            a = pg.InfiniteLine(pos=float(r.centre_z_nm), angle=90,
                                pen=pg.mkPen(col, width=4 if r.is_central else 1.5, style=style))
            pp.addItem(a)
            # in the strip blue and vermillion mean "rises" / "falls": the ring centres are neutral there
            b = pg.InfiniteLine(pos=float(r.centre_z_nm), angle=90,
                                pen=pg.mkPen(neutral(dark=True), width=2 if r.is_central else 1, style=style))
            sp.addItem(b)
            # two lines only: on a short plot (a 1366 x 768 screen) a third line reached down into the peaks
            label = (f"ring {r.index}\nCENTRAL" if r.is_central else f"ring {r.index}\n{_pct(r.f_of_central)}")
            if r.truncated_edge:
                label += " (edge cut)"
            # hangs down from just under the top, just right of its own line (never crossed by it)
            txt = pg.TextItem(label, color=col, anchor=(-0.08, 0.0))
            txt.setPos(float(r.centre_z_nm), top * 1.75)
            pp.addItem(txt)
            centre_items.extend((a, b, txt))
        if len(v2.rings):
            # the last entry, as before: one sample for every ring centre of both plots and their labels
            layers.entry(self.legend, "ring_centre", "ring centre: % of the central ring<br>(thick solid = central)",
                         "line", centre_items,
                         pen=pg.mkPen(neutral(dark=True), width=2, style=QtCore.Qt.PenStyle.DashLine))
        # where the data end abruptly (a z filter?), drawn at the end of the z' data
        zp = rep.z_p[np.isfinite(rep.z_p)]
        for e in v2.z_edges:
            if e.truncated and zp.size:
                pos = float(zp.min()) if e.side == "bottom" else float(zp.max())
                pp.addItem(pg.InfiniteLine(pos=pos, angle=90, pen=pg.mkPen(neutral(dark=True), width=1.5,
                                                                         style=QtCore.Qt.PenStyle.DotLine)))
                t = pg.TextItem(f"data cut ({e.side})", color=neutral(dark=True),
                                anchor=(0.0 if e.side == "bottom" else 1.0, 1.0))
                t.setPos(pos, top * 0.55)
                pp.addItem(t)
        pp.setYRange(0, top * 1.8, padding=0)
        if zp.size:
            pp.setXRange(float(zp.min()) - 30, float(zp.max()) + 30, padding=0)
        # the name shortened in the middle: a plot title is never narrower than its text, and a real Picasso name
        # (~95 characters) pushed the legend out of a 1440 px window (the full name is in the header above)
        name = os.path.basename(rep.source_name) or "the axon"
        if len(name) > 44:
            name = name[:16] + "..." + name[-25:]
        set_title(pp, f"Axial profile of {name}: "
                      f"{len(v2.rings)} ring(s), central ring {v2.central_index}, P = {v2.p_ref_nm:.1f} nm")

    def _item(self, text: str, tip: str = "", kind: Optional[str] = None) -> QtWidgets.QTableWidgetItem:
        it = QtWidgets.QTableWidgetItem(text)
        if tip:
            it.setToolTip(tip)
        if kind is not None:
            it.setForeground(QtGui.QBrush(QtGui.QColor(verdict(kind, dark=False))))
            f = it.font()
            f.setBold(True)
            it.setFont(f)
        return it

    def _fill_rings(self, rep: ZQualityReport) -> None:
        t = self.ring_table
        t.setRowCount(0)
        for r in rep.v2.rings:
            i = t.rowCount()
            t.insertRow(i)
            vals = [str(r.index), _ring_role(r), _num(r.centre_z_nm, ".0f"), f"{r.n_locs:,}", _pct(r.f_of_central),
                    str(r.k_clusters), _num(r.locs_per_cluster), _num(r.lpz_median_nm, ".0f"),
                    _num(r.sigma_z_nm, ".0f"), _num(r.sigma_over_p, ".3f"),
                    (f"yes ({r.peak_z_nm:.0f} nm)" if r.peak and r.peak_z_nm is not None else "NO")]
            for j, v in enumerate(vals):
                kind = None
                if j == 10:
                    kind = "good" if r.peak else "bad"
                t.setItem(i, j, self._item(v, kind=kind))
        t.resizeColumnsToContents()
        role = RING_COLUMNS.index("Role")
        t.setColumnWidth(role, min(t.columnWidth(role), 170))     # a long role wraps instead of widening the table
        t.resizeRowsToContents()

    def _fill_pairs(self, rep: ZQualityReport) -> None:
        t = self.pair_table
        t.setRowCount(0)
        v2 = rep.v2
        for p in v2.pairs:
            i = t.rowCount()
            t.insertRow(i)
            vtext, vkind = verdict_text(p.verdict)
            fmin = f"{_pct(p.f_min)} (x = {_num(p.x)})" + ("" if p.x_in_range else " !")
            fmin_tip = ("x inside the validated range 5.8-13.3" if p.x_in_range else
                        "! x outside the calibrated range 5.8-13.3" + (f": {v2.x_note}" if v2.x_note else ""))
            why = "; ".join(p.reasons) if p.reasons else "all four criteria met"
            cells = [
                (f"{p.ring_a}-{p.ring_b}", "", None),
                (f"{'yes' if p.peak_a else 'NO'} / {'yes' if p.peak_b else 'NO'}", "",
                 "good" if (p.peak_a and p.peak_b) else "bad"),
                ("yes" if p.valley else "NO", "", "good" if p.valley else "bad"),
                (f"{_pct(p.f_a)} / {_pct(p.f_b)}", "", "good" if p.count_ok else "bad"),
                (fmin, fmin_tip, None if p.x_in_range else "warn"),
                (f"{100 * p.exp_spur_frac:.1f} %" if math.isfinite(p.exp_spur_frac) else "n/a",
                 "expected fraction of clusters that are axial-leak copies (D-39): <= 2 % viable, <= 5 % marginal",
                 "good" if p.exp_spur_frac <= 0.02 else ("warn" if p.exp_spur_frac <= 0.05 else "bad")),
                (vtext, why, vkind),
                (p.d39_verdict, "the D-39 verdict, shown for comparison (not used)", None),
                (why, why, None),
            ]
            for j, (txt, tip, kind) in enumerate(cells):
                t.setItem(i, j, self._item(txt, tip, kind))
        why_col = PAIR_COLUMNS.index("Why")
        for j in range(len(PAIR_COLUMNS)):
            if j != why_col:
                t.resizeColumnToContents(j)
        t.resizeRowsToContents()

    # ------------------------------------------------------------ export
    def export_report(self, folder: str) -> List[str]:
        """Write the report (``write_report_files``) with a PNG of the whole view and one of the profile."""
        if self.report is None:
            raise RuntimeError("no z-quality report on screen yet")
        QtWidgets.QApplication.processEvents()

        def save_view(path: str) -> bool:
            # the whole view, also the part a short window scrolls away (the scroll area holds it since H6 clusters)
            return bool(self.content_widget.grab().save(path, "PNG"))

        def save_profile(path: str) -> bool:
            return bool(self.graphics.grab().save(path, "PNG"))

        return write_report_files(self.report, folder, images={"view": save_view, "profile": save_profile})

    def export_cluster_report(self, folder: str, *, stem: Optional[str] = None) -> List[str]:
        """The cluster panel's NEW files (``write_cluster_report_files``): the CSV of the medians of both cluster sets
        and a PNG of the panel (the cluster set on screen). Nothing when no cluster profile is shown."""
        if self.report is None:
            raise RuntimeError("no z-quality report on screen yet")
        if self.report.clusters is None:
            return []
        QtWidgets.QApplication.processEvents()

        def save_hist(path: str) -> bool:
            return bool(self.cluster_panel.grab().save(path, "PNG"))

        return write_cluster_report_files(self.report, folder, stem=stem, images={"histogram": save_hist})

    def export_v2c_report(self, folder: str, *, stem: Optional[str] = None) -> List[str]:
        """Rule v2-clusters' NEW file (``write_v2c_report_files``: ``<stem>_zquality_clusters_pairs.csv``). Nothing
        when the report has no rule v2-clusters."""
        if self.report is None:
            raise RuntimeError("no z-quality report on screen yet")
        return write_v2c_report_files(self.report, folder, stem=stem)

    def _on_export(self) -> None:
        if self.report is None:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the z-quality report",
                                                            os.path.dirname(self.report.source_name) or "")
        if not folder:
            return
        try:
            paths = self.export_report(folder)
            # the cluster files beside them, with the stem the report just used (".../<stem>_zquality_rings.csv")
            first = os.path.basename(paths[0]) if paths else ""
            stem = first[:-len("_zquality_rings.csv")] if first.endswith("_zquality_rings.csv") else None
            paths = list(paths) + self.export_cluster_report(folder, stem=stem)
            paths = paths + self.export_v2c_report(folder, stem=stem)
            paths = paths + self.export_selection_report(folder, stem=stem)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            QtWidgets.QMessageBox.critical(self, "Z quality", f"The report could not be written:\n{exc}")
            return
        n_hidden = len(self.hidden_layers())
        note = f"\n\n({n_hidden} layer(s) hidden in the images)" if n_hidden else ""
        QtWidgets.QMessageBox.information(self, "Z quality", "Written:\n" + "\n".join(os.path.basename(p) for p in paths)
                                          + f"\n\nin {folder}" + note)

    def hidden_layers(self) -> List[str]:
        """The legend entries now hidden (the exported images show what is on screen)."""
        out = list(self.legend_layers.hidden())
        if not self.cluster_panel.isHidden():
            out += self.cluster_legend_layers.hidden()
        return out

    def _on_columns(self) -> None:
        if self.columns_callback is not None:
            self.columns_callback()

    def closeEvent(self, event: Any) -> None:
        if self._thread is not None:
            self._generation += 1      # a late result is dropped
            self._thread = None
        super().closeEvent(event)


# ============================================================================
# Opening it
# ============================================================================

def open_z_quality_for_rings(res: Any, xyz_lab_nm: Optional[Sequence[Any]] = None, *,
                             parent: Optional[QtWidgets.QWidget] = None, source_name: str = "", roi_text: str = "",
                             columns_callback: Optional[Callable[[], Any]] = None) -> ZQualityWindow:
    """Show the view for rings already built (the column review's); the numbers are computed in the background."""
    w = ZQualityWindow(parent, title_source=source_name or getattr(res, "source_name", ""),
                       columns_callback=columns_callback)
    w.show()

    def job(say: Callable[[str], None]) -> ZQualityReport:
        say("SiZer and the rule v2 on the rings")
        rep = compute_z_quality(res, xyz_lab_nm, source_name=source_name, roi_text=roi_text)
        say("The cluster medians (descriptive)")
        rep = compute_z_quality_clusters(res, xyz_lab_nm, rep)
        say("Rule v2-clusters (exploratory)")
        return compute_z_quality_v2c(res, xyz_lab_nm, rep)

    w.start(job)
    return w


def open_z_quality_for_inputs(inputs: Any, *, parent: Optional[QtWidgets.QWidget] = None,
                              columns_callback: Optional[Callable[[], Any]] = None) -> ZQualityWindow:
    """Show the view for a selection (a ``ColumnsReviewInputs``): the rings are built in the background exactly as
    the column review builds them (``build_review_rings``), then the z quality is computed."""
    w = ZQualityWindow(parent, title_source=str(getattr(inputs, "source_name", "")), columns_callback=columns_callback)
    w.show()

    def job(say: Callable[[str], None]) -> ZQualityReport:
        from tools.mps_columns_window import build_review_rings
        n = int(np.asarray(inputs.x_nm).size)
        say(f"Building the rings of {n:,} localizations (as the column review does)")
        res, _params, _lpz, warns = build_review_rings(inputs)
        say("SiZer and the rule v2 on the rings")
        rep = compute_z_quality(res, (inputs.x_nm, inputs.y_nm, inputs.z_nm), source_name=str(inputs.source_name),
                                roi_text=str(getattr(inputs, "roi_text", "") or ""))
        rep.warnings = list(warns) + rep.warnings
        say("The cluster medians (descriptive)")
        rep = compute_z_quality_clusters(res, (inputs.x_nm, inputs.y_nm, inputs.z_nm), rep)
        say("Rule v2-clusters (exploratory)")
        return compute_z_quality_v2c(res, (inputs.x_nm, inputs.y_nm, inputs.z_nm), rep)

    w.start(job)
    return w


# ============================================================================
# Stand-alone demonstration (SIMULATED axons, or an NPZ)
# ============================================================================

DEMO_CASES: Dict[str, Tuple[str, int, float]] = {
    # tools.mps_sim_harness's circle contour, seed, axial-leak scale (batch_columns.write_simulated_input): rule v2
    # (and the D-39 first filter) gives
    "viable": ("circle", 39, 0.3),        # both pairs VIABLE
    "marginal": ("circle", 22, 0.6),      # one MARGINAL (pair 0-1), one VIABLE
    "none": ("circle", 30, 0.7),          # no viable pair (wide rings, much leak)
}


def inputs_from_npz(path: str) -> Any:
    """A ``ColumnsReviewInputs`` from an NPZ with x, y, z, frame, lp, lpz (nm) and n_frames (power_columns'
    NPZ_KEYS)."""
    from tools.mps_columns_window import ColumnsReviewInputs
    with np.load(path) as st:
        return ColumnsReviewInputs(
            x_nm=np.asarray(st["x"], float), y_nm=np.asarray(st["y"], float), z_nm=np.asarray(st["z"], float),
            frame=np.asarray(st["frame"], np.int64), lp_lateral_nm=np.asarray(st["lp"], float),
            lpz_nm=np.asarray(st["lpz"], float), n_frames=int(np.asarray(st["n_frames"]).reshape(-1)[0]),
            source_name=os.path.abspath(path), roi=None, roi_text="")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import tempfile
    ap = argparse.ArgumentParser(description="The z-quality view on a SIMULATED axon (--demo) or an NPZ (--npz).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", choices=sorted(DEMO_CASES), help="simulate an axon whose rule-v2 verdict is this")
    g.add_argument("--npz", help="an NPZ with x, y, z, frame, lp, lpz, n_frames")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    # The application first (final audit, 2026-10-06): the demo's simulator once imported a validation harness that
    # sets QT_QPA_PLATFORM=offscreen, and an application created after that was never on screen.
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if args.demo:
        import batch_columns as bc
        contour, seed, s = DEMO_CASES[args.demo]
        folder = tempfile.mkdtemp(prefix="mps_zquality_demo_")
        path = bc.write_simulated_input(os.path.join(folder, f"simulated_{args.demo}.npz"), contour, seed, s)
    else:
        path = args.npz
    inputs = inputs_from_npz(path)
    holder: Dict[str, Any] = {}

    def open_review() -> None:
        from tools.mps_columns_window import start_columns_review
        holder["review"] = start_columns_review(inputs)

    holder["z"] = open_z_quality_for_inputs(inputs, columns_callback=open_review)
    return int(app.exec_())


if __name__ == "__main__":
    sys.exit(main())
