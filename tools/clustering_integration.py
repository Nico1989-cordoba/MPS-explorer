# -*- coding: utf-8 -*-
"""
The clustering integration (wave 2, 2026-10-22): every new analysis, for every clustering method chosen, side by
side -- without Qt.

The user's central request (plan of 2026-10-22): each analysis of an axon must be comparable across clustering
methods. The wave-1 branches built the analyses as pure functions of ONE method's result (``tools.multimethod``
plug-ins); this module is what the windows and the batch call to run them:

* :func:`register_all_plugins` -- every plug-in registered once (core, events, sweep, eclipse, size, size_eps,
  pipeline null, gap measures) plus the pipeline null's extra readouts (size, gap), so one null serves every readout
  (decision D7).
* :data:`PROVISIONAL` -- the open questions of the wave-1 reports (``PREGUNTAS_ABIERTAS.md`` A-E, G) answered by
  their recommendation, each as ONE named value with its origin, so changing an answer is one line here.
* :func:`run_comparison` -- 'Compare methods...' without Qt: the methods on the same slab (DBSCAN 25/10 always
  first, as the reference), the chosen analyses on each, the agreement between methods and the "does the conclusion
  depend on the method?" summary; :func:`write_comparison` writes ``<stem>_methods_long.csv`` with the agreement and
  the summary beside it, and the wide table as before.
* :func:`run_null_for` / :func:`run_nulls` -- the per-axon pipeline null of each method's run, re-running THAT
  method (same parameters, curation and contour), with the regularity reading (gap measures, two stages) on the same
  null; slow methods refused unless asked, with the estimate.
* :func:`axon_columns` -- the columns the axon table gains in v13 (``INTEGRATION_COLUMNS``): the working scale, the
  methods block (A8: how DBSCAN's parameters were chosen, c_label, localizations per cycle, linking), the events
  layer, the eclipse tolerance, the apparent size, the regularity and the pipeline null of the analysis exported.
* :func:`regularity_batch_summary` -- regularity PR 4: per animal z, Holm, the P-R83 lattice criterion and the
  calibre model, over a batch's rows.

Never r_eff as a size (the size readouts are the apparent-size branch's); "Cluster area" is not renamed (the user's
decision of 2026-10-22).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import importlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tools import multimethod as mm
from tools.method_costs import is_slow, method_label, words_seconds

# =====================================================================================================================
# The open questions, answered provisionally by their recommendation (one value each; change it here)
# =====================================================================================================================


@dataclass(frozen=True)
class Provisional:
    """One open question of the wave-1 reports and the value the program uses until the user answers it."""
    key: str
    question: str
    value: Any
    origin: str
    where: str
    applied: bool = True


# G4 (user decision 2026-11-07): said wherever a per-axon regularity label is shown.
LATTICE_WORDS = ("the per-axon label is descriptive only and under-reads 'lattice': a full lattice through the "
                 "pipeline reads lattice in 4 of 12 simulated axons and periodic in 8 [own sim, docs/regularity.md]")
# G5 (user decision 2026-11-07): the visible warning when the edge curation removed clusters.
EDGE_WARNING = "contour stretches removed at the ROI edge: regularity may be biased"

_P: Tuple[Provisional, ...] = (
    Provisional("A1", "edge-margin floor of the methods that are not DBSCAN", "the eps of the analysis shown",
                "[ad hoc: recommendation A1, as 'Compare methods...' did]", "tools.multimethod.analyze_like"),
    Provisional("A2", "slow methods (Emitters, OPTICS, voronoi_csr) in the comparison", True,
                "[user decision A2 (a)+(b), 2026-10-23] ticking a slow method in 'Compare methods...' is the "
                "permission; the Batch and the null refuse them unless 'Allow slow methods' is ticked; and every time "
                "a slow method is about to run (Compare, or the Batch / the null with the box ticked) a confirmation "
                "dialog shows the time estimate first (tools.slow_confirm)",
                "tools.compare_analyses_ui; tools.mps_batch; tools.arrangement_ui"),
    Provisional("A3", "'large spread' between methods", "visual flag only",
                "[ad hoc: recommendation A3] range over the reference > 10 % (tools.multimethod.SPREAD_FLAG_REL), "
                "never a test until the null's SD gives a scale", "tools.multimethod.conclusion_summary"),
    Provisional("A4", "export of the comparison", "long + agreement + summary, wide kept",
                "[ad hoc: recommendation A4]", "write_comparison"),
    Provisional("B1", "radius of the localizations-per-cycle measurement that feeds the DBSCAN rule",
                3.0 * math.sqrt(2.0),
                "[simulation: docs/dbscan_rule.md sec. 4 -- 3 NeNA under-reads it by 9-26 %, 3 sqrt(2) NeNA by "
                "0.4-3.4 %; user decision B1 (a), 2026-10-23] (in NeNA); events and the fusion mark keep 3 NeNA "
                "(decision D5). Both radii are editable in 'Working scale' to explore (EXPLORATORY when not the "
                "default; the export says which radius each used: lpc_rule_radius_nena, link_radius_nena)",
                "rule_for; locs_per_cycle_of(radius_in_nena=); comparison_extras(events_radius_in_nena=)"),
    Provisional("B2", "min samples formula counts the point itself", True,
                "[lit: Ester et al. 1996; recommendation B2] (gives 6 with c_label 14 and 2 loc/cycle)",
                "tools.dbscan_rule.derive_rule(count_self=...)"),
    Provisional("B3", "where localizations per cycle are measured", "the whole loaded file",
                "[ad hoc: recommendation B3] like the NeNA: the photophysics is the movie's, not the axon's",
                "MPS_explorer._file_rule_inputs; tools.mps_batch"),
    Provisional("B4", "ToMATo grid of the sweep", "kept, labelled [ad hoc]", "[ad hoc: recommendation B4]",
                "tools.dbscan_sweep.GRID_PRESETS"),
    Provisional("B5", "the sweep (20 analyses per method)", "unticked by default, with its cost shown",
                "[ad hoc: recommendation B5]", "ANALYSIS_CHOICES"),
    Provisional("C1", "where the global H1 / H2 test starts", "d50",
                "[simulation: docs/null_and_scale.md -- the contact core 2 r_eff reads ~6 nm high and made H1 / H2 "
                "liberal at 60-80 nm; recommendation C1] the working scale's d50 (~80 nm) instead of the fusion mark",
                "run_null_for -> tools.mps_pipeline_null.run_pipeline_null(global_test_from=)"),
    Provisional("C2", "membrane of the null for methods that leave gaps (HDBSCAN)",
                "the centroids of DBSCAN 25/10, common to every method; the localizations as an option",
                "[user decision C2 hybrid (a)+(b), 2026-10-23] with the deformation indicator between the two curves "
                "(mean and largest distance, a warning per 600 nm stretch beyond 50 nm [ad hoc: the H2 band's "
                "half-width]); the DBSCAN 25/10 run's own null is unchanged", "tools.null_membrane; run_null_for"),
    Provisional("C3", "pilots of the K closure", 4,
                "[simulation: docs/null_and_scale.md -- 2 pilots left 9 of 20 axons > 5 % off; recommendation C3]",
                "tools.mps_pipeline_null.PILOT_AXONS_K"),
    Provisional("C4", "default closure", "n_locs (localizations in the slab)",
                "[user decision A9; recommendation C4: 3-4 pilots before changing it]", "run_null_for"),
    Provisional("C5", "power per axon (K 35-50)", "conclusion on the arrangement from the batch per animal",
                "[ad hoc: recommendation C5] said in the 'Perimeter arrangement' tab", "tools.arrangement_ui"),
    Provisional("D1", "2D eclipse test in the plane: tau_0 31 or 40 nm", 40.0,
                "[derivation: 3.46 x 11.3 nm; recommendation D1] NOT applied: config/columns_params.yaml is a frozen "
                "pre-specification of the H-ECL review and its tau control; changing it is the user's decision",
                "config/columns_params.yaml (unchanged)", applied=False),
    Provisional("D2", "the merged-cluster criterion (z 3.29) as a measurement", True,
                "[derivation + simulation: docs/eclipse_tolerance.md sec. 5; recommendation D2] the footprint pool "
                "leaves the merged-looking clusters out", "comparison_extras -> ToleranceInputs(exclude_merged_from_pool)"),
    Provisional("D2b", "what the eclipse arc test does with a merged-looking cluster", "split",
                "[own sim: docs/sims/eclipse_tolerance/merged_handling_sim.py, docs/eclipse_tolerance.md sec. 10; "
                "user decision D2a] split keeps the 5 %% level and gains power with merges (+2.8 / +4.3 points at "
                "10 / 25 %% merged); only the arc test's centres: the footprint pool (tools.footprint_pool) still "
                "leaves the merged-looking clusters out (D2)".replace("%%", "%"),
                "tools.mps_eclipse_merged.MERGED_HANDLING_DEFAULT; extras['eclipse_merged_handling']"),
    Provisional("D3", "per-pair tolerance in the 2D null", "Phase B", "[ad hoc: recommendation D3]", "-",
                applied=False),
    Provisional("D4", "'Free' tau starts at 31 nm without an exploratory mark", "accepted",
                "[ad hoc: recommendation D4]", "tools.mps_tau_ui (unchanged)"),
    Provisional("E1", "elongation cut of the footprint pool", 2.5,
                "[simulation: docs/apparent_size.md sec. 5 and 12; user decision E1a] within 1-2 nm of the true "
                "footprint at c_label 20-26 (betaME), -0.7 to -2.5 nm at 14, fails at 8 (-3 to -7 nm)",
                "tools.mps_apparent_size.POOL_EXCLUDE_ELONGATION"),
    Provisional("E2", "fusions fooling the 1.5-labels filter with one label per cluster", True,
                "[user decision E2, 2026-10-23; simulation: docs/apparent_size.md, 3 of 12 axons] the footprint pool "
                "(size AND eclipse, one definition: tools.footprint_pool) leaves out the clusters the event layer "
                "marks as probable fusions", "tools.footprint_pool; comparison_extras -> size_exclude_event_fusion, "
                "ToleranceInputs(exclude_event_fusion_from_pool)"),
    Provisional("E4", "plug-in size_eps (eps 20 / 25 / 30)", "unticked by default, with its cost shown",
                "[ad hoc: recommendation E4]", "ANALYSIS_CHOICES"),
    Provisional("E5", "pipeline null for the size question (b)", "the axon's one null (G3), readout "
                "size_nloc_kappa_r_spearman", "[user decision E5a; calibration: own sim, docs/sims/apparent_size/"
                "curvature_null_calibration.py] its p counts the failed replicates as G1 says (p_mc_size_nloc_kappa_r_"
                "spearman_ignoring_failed beside it)", "tools.size_curvature_null.register_with_pipeline_null"),
    Provisional("G1", "failed replicates of the pipeline null (stage 1 and stage 2)", 0.05,
                "[user decision G1 (c), 2026-10-23; simulation: docs/regularity.md, 22 of 60 axons lost 1 of 49] "
                "up to floor(0.05 B) tolerated, each counted as at least as extreme as the observed (conservative p); "
                "null_failed_replicates and p_mc_*_ignoring_failed exported; more -> NOT INTERPRETABLE",
                "tools.mps_pipeline_null.TOLERATED_FAILED_FRACTION (= tools.mps_gap_measures."
                "MATCHED_TOLERATED_FAILED_FRACTION)"),
    Provisional("G2", "pipeline-level bands of the regularity plots", True,
                "[user decision G2 (a), 2026-11-07] the null keeps every replicate's arcs (keep_arcs) and the "
                "regularity plots draw their band from them once the null ran (the same null as stage 2's p); before "
                "it, the centres-level band labelled 'approx., centres'; an exported image says which null its band "
                "comes from", "run_null_for -> run_pipeline_null(keep_arcs=True); tools.arrangement_ui"),
    Provisional("G3", "one null for every readout", True, "[decision D7; user decision G3, 2026-11-07] the null is passed "
                "to the gap measures (extras['pipeline_null_results'])", "run_null_for"),
    Provisional("G4", "the type of regularity of a group", "from the per-animal z (P-R83), not per axon",
                "[user decisions P-R83 and G4 (a), 2026-11-07] the window and the Batch summary say that "
                + LATTICE_WORDS, "regularity_batch_summary; tools.arrangement_ui"),
    Provisional("G5", "contour stretches lost at the ROI edge", "a visible warning beside the reading",
                "[simulation: docs/regularity.md, up to 11 of 41 clusters; user decision G5 (a), 2026-11-07] the "
                "regularity tab warns when the edge curation removed a cluster; the ROI tooltips recommend a "
                "margin >= 500 nm around the membrane [own sim: 0 clusters removed in 60 axons]",
                "edge_note; tools.arrangement_ui; tools.mps_tooltips"),
)
PROVISIONAL: Dict[str, Provisional] = {p.key: p for p in _P}


def provisional(key: str) -> Any:
    """The value the program uses for the open question ``key`` (``PROVISIONAL``)."""
    return PROVISIONAL[key].value


NULL_GLOBAL_TEST_FROM: str = str(provisional("C1"))
NULL_CLOSURE = "n_locs"                                 # [user decision A9; C4]
# C3 / C4 (2026-10-23): the closures a caller may ask for -- the null's own two and "both" (one null per closure).
# "k" is the user's word for tools.mps_pipeline_null's "k_clusters".
NULL_CLOSURE_CHOICES: Tuple[str, ...] = ("n_locs", "k", "both")


def null_closures(choice: str) -> List[str]:
    """The null's closures (``tools.mps_pipeline_null.CLOSURES`` names) for a choice of ``NULL_CLOSURE_CHOICES``
    (the null's own names are accepted too): "n_locs" -> [n_locs], "k" -> [k_clusters], "both" -> both, n_locs first
    (the default closure, the one the axon table's row carries)."""
    c = str(choice)
    if c in ("n_locs",):
        return ["n_locs"]
    if c in ("k", "k_clusters"):
        return ["k_clusters"]
    if c == "both":
        return ["n_locs", "k_clusters"]
    raise ValueError(f"null closure must be one of {NULL_CLOSURE_CHOICES}, got {choice!r}")
LPC_RULE_RADIUS_IN_NENA: float = float(provisional("B1"))
# B1 (user decision 2026-10-23): the event-linking radius of the events layer and the fusion mark (decision D5,
# 3 NeNA [simulation: docs/dbscan_rule.md]); like the rule's radius above, editable in 'Working scale' to explore.
EVENTS_RADIUS_IN_NENA: float = 3.0
# C2: the membrane of every method's null (tools.null_membrane).
NULL_MEMBRANE = "dbscan_centroids"


def radii_exploratory(lpc_radius_in_nena: Optional[float], events_radius_in_nena: Optional[float]) -> str:
    """B1: "no" when both radii are the defaults (3 sqrt(2) and 3 NeNA), else which one moved, the words of the
    export's radii_exploratory column and of the panel's EXPLORATORY mark."""
    parts = []
    if lpc_radius_in_nena is not None and not math.isclose(float(lpc_radius_in_nena), LPC_RULE_RADIUS_IN_NENA,
                                                           abs_tol=1e-6):
        parts.append(f"lpc of the rule at {float(lpc_radius_in_nena):g} NeNA (default {LPC_RULE_RADIUS_IN_NENA:.3g})")
    if events_radius_in_nena is not None and not math.isclose(float(events_radius_in_nena), EVENTS_RADIUS_IN_NENA,
                                                              abs_tol=1e-6):
        parts.append(f"events / fusion at {float(events_radius_in_nena):g} NeNA (default {EVENTS_RADIUS_IN_NENA:g})")
    return "EXPLORATORY: " + " + ".join(parts) if parts else "no"
MIN_SAMPLES_COUNT_SELF: bool = bool(provisional("B2"))
ECLIPSE_EXCLUDE_MERGED: bool = bool(provisional("D2"))
SIZE_EXCLUDE_EVENT_FUSION: bool = bool(provisional("E2"))
NULL_KEEP_ARCS: bool = bool(provisional("G2"))
# The stage 2 of the regularity reading: the GUI preview at the centres level (seconds, "approx., centres"), the batch
# at the pipeline level (exact) [regularity plan P-R87 c; the coordinator's instruction: gap_stage2 = "pipeline" in
# the batch].
GAP_STAGE2_WINDOW = "centres"
GAP_STAGE2_BATCH = "pipeline"

# =====================================================================================================================
# Plug-ins
# =====================================================================================================================

# The regularity branch registers itself on import (tools.mps_gap_measures: the gap_ plug-in and its null readouts);
# any later branch adds its module here and the integration picks it up.
OPTIONAL_PLUGIN_MODULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("tools.mps_gap_measures", ("register_null_readouts", "register_multimethod")),
)
_REGISTERED = False


def register_all_plugins(force: bool = False) -> List[str]:
    """Register every analysis plug-in with ``tools.multimethod`` and every extra readout with the pipeline null
    (idempotent; the program calls it at start-up). Returns the registered analyses, in order."""
    global _REGISTERED
    if _REGISTERED and not force:
        return mm.registered_analyses()
    from tools import events_sweep_plugins, mps_eclipse_tolerance, mps_pipeline_null, size_plugins
    mm.register_core()
    events_sweep_plugins.register_with_multimethod(replace=True)
    mps_eclipse_tolerance.register(replace=True)
    size_plugins.register_with_multimethod(replace=True)
    size_plugins.register_with_pipeline_null(replace=True)
    mps_pipeline_null.register_multimethod(replace=True)
    for module, functions in OPTIONAL_PLUGIN_MODULES:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        for fn in functions:
            getattr(mod, fn)(replace=True)
    _REGISTERED = True
    return mm.registered_analyses()


@dataclass(frozen=True)
class AnalysisChoice:
    """One analysis the comparison can run, as the dialog offers it."""
    name: str
    label: str
    default_on: bool
    tip: str


ANALYSIS_CHOICES: Tuple[AnalysisChoice, ...] = (
    AnalysisChoice("per_axon", "Per-axon readouts", True,
                   "Clusters per um, median 1NN, median r_eff, occupancy and clusters kept: what the table shows."),
    AnalysisChoice("events", "Events and probable fusions", True,
                   "Linked switching events per cluster (3 NeNA, 1 dark frame) and the 'probable fusion' mark "
                   "(events / c_label > 2) with c_label 14 and 26. Needs frames and the file's NeNA. "
                   "Methods: docs/dbscan_rule.md."),
    AnalysisChoice("eclipse", "Eclipse tolerance", True,
                   "Per-cluster centre errors and the axon's footprint (P-R61); the scalar tau 31 nm is primary, the "
                   "per-pair tolerance exploratory. Methods: docs/eclipse_tolerance.md."),
    AnalysisChoice("size", "Apparent size", True,
                   "Content (events or localizations), cloud SD, hull at fixed N, footprint, curvature. 'Cluster area' "
                   "stays as it is; r_eff is never used as a size. Methods: docs/apparent_size.md."),
    AnalysisChoice("sweep", "Parameter sweep (slow)", False,
                   "The method's preset grid on the same slab (DBSCAN: eps 15-40 x min samples 5-20, 20 analyses per "
                   "method). Off by default; its cost is shown before Run. Methods: docs/dbscan_rule.md."),
    AnalysisChoice("size_eps", "Size at eps 20 / 25 / 30 (DBSCAN)", False,
                   "The size-neighbour statistic re-clustered at eps 20, 25 and 30 nm (DBSCAN only). Off by default; "
                   "its cost is shown before Run. Methods: docs/apparent_size.md."),
)
DEFAULT_ANALYSES: Tuple[str, ...] = tuple(c.name for c in ANALYSIS_CHOICES if c.default_on)


def estimate_analyses_seconds(names: Sequence[str], methods: Sequence[str], n_locs: int) -> float:
    """A first estimate of the analyses' cost before anything runs [ad hoc: the cheap plug-ins are a few per cent of
    an analysis; the sweep is 20 analyses per DBSCAN-like method, size_eps 2 for DBSCAN only]."""
    from tools import dbscan_sweep
    from tools.method_costs import estimate_seconds
    total = 0.0
    for m in methods:
        one = estimate_seconds([m], int(n_locs))
        if "sweep" in names and m in dbscan_sweep.GRID_PRESETS:
            total += len(dbscan_sweep.grid_cells(dbscan_sweep.GRID_PRESETS[m])) * one
        if "size_eps" in names and m == mm.REFERENCE_METHOD:
            total += 2.0 * one
        total += 0.1 * one * sum(1 for n in names if n in ("events", "eclipse", "size"))
    return float(total)


# =====================================================================================================================
# What the analyses read besides the run: the file's NeNA, c_label, the provisional answers
# =====================================================================================================================

def event_ids_for(analysis: Any, nena_nm: Optional[float],
                  radius_in_nena: Optional[float] = None) -> Optional[np.ndarray]:
    """The event id of every localization the analysis was given, for the eclipse plug-in
    (``extras["event_id"]``, sliced there with ``analysis.slab_index``): the slab's localizations linked at 3 NeNA,
    1 dark frame (decision D5; ``tools.mps_events.link_events``), -1 outside the slab. None without frames or NeNA.
    The same for every method of a comparison (they share the slab)."""
    from tools.mps_events import link_events
    frames = getattr(analysis, "localization_frame_slab", None)
    idx = getattr(analysis, "slab_index", None)
    nena = _num(nena_nm)
    if frames is None or idx is None or nena is None or nena <= 0:
        return None
    idx = np.asarray(idx, dtype=np.intp)
    x = np.asarray(analysis.x_slab, dtype=np.float64)
    if idx.size != x.size or idx.size == 0:
        return None
    k = EVENTS_RADIUS_IN_NENA if radius_in_nena is None else float(radius_in_nena)
    layer = link_events(np.asarray(frames), x, np.asarray(analysis.y_slab, dtype=np.float64), nena_nm=nena,
                        radius_in_nena=k)
    out = np.full(int(idx.max()) + 1, -1, dtype=np.int64)
    out[idx] = np.asarray(layer.link_group, dtype=np.int64)
    return out


def comparison_extras(*, nena_nm: Optional[float], c_label: Optional[float] = None,
                      pipeline_null_results: Optional[Mapping[str, Any]] = None,
                      gap_stage2: str = GAP_STAGE2_WINDOW, gap_null: str = "none",
                      analysis: Optional[Any] = None,
                      events_radius_in_nena: Optional[float] = None) -> Dict[str, Any]:
    """``context.extras`` for :func:`tools.multimethod.apply_analyses`, with the provisional answers applied.
    ``gap_null`` "none" keeps the gap measures to their values (no null runs in a comparison: the null has its own
    button); "pipeline" lets them run (or reuse, ``pipeline_null_results``) the null."""
    from tools.mps_eclipse_tolerance import ToleranceInputs
    from tools.dbscan_rule import C_LABEL_DEFAULT
    c = float(C_LABEL_DEFAULT if c_label is None else c_label)
    extras: Dict[str, Any] = {
        "nena_nm": nena_nm,
        "size_c_label": c,
        "size_exclude_event_fusion": SIZE_EXCLUDE_EVENT_FUSION,
        "eclipse_inputs": ToleranceInputs(c_label=c, exclude_merged_from_pool=ECLIPSE_EXCLUDE_MERGED,
                                          exclude_event_fusion_from_pool=SIZE_EXCLUDE_EVENT_FUSION),
        "null_c_label": c,
        "null_closure": NULL_CLOSURE,
        "null_global_test_from": NULL_GLOBAL_TEST_FROM,
        "gap_stage2": gap_stage2,
        "gap_null": gap_null,
    }
    if events_radius_in_nena is not None:
        # B1: an edited events / fusion radius (the plug-ins read it; absent = 3 NeNA, decision D5)
        extras["events_radius_in_nena"] = float(events_radius_in_nena)
    if pipeline_null_results:
        extras["pipeline_null_results"] = dict(pipeline_null_results)
    if analysis is not None:
        try:
            ev = event_ids_for(analysis, nena_nm, events_radius_in_nena)
        except ValueError:
            ev = None
        if ev is not None:
            extras["event_id"] = ev
    return extras


def rule_for(nena_nm: Optional[float], *, c_label: Optional[float] = None, locs_per_cycle: Optional[Any] = None
             ) -> Optional[Any]:
    """The DBSCAN rule of this file (``tools.dbscan_rule.derive_rule``) or None without a NeNA."""
    from tools.dbscan_rule import C_LABEL_DEFAULT, derive_rule
    if nena_nm is None or not (math.isfinite(float(nena_nm)) and float(nena_nm) > 0):
        return None
    return derive_rule(float(nena_nm), c_label=float(C_LABEL_DEFAULT if c_label is None else c_label),
                       locs_per_cycle=locs_per_cycle, count_self=MIN_SAMPLES_COUNT_SELF)


def locs_per_cycle_of(loc: Any, nena_nm: Optional[float], radius_in_nena: Optional[float] = None) -> Any:
    """Localizations per cycle of a loaded file, measured on the WHOLE file (B3) at 3 sqrt(2) NeNA (B1), or at the
    radius edited in 'Working scale' (``radius_in_nena``, EXPLORATORY)."""
    from tools.mps_events import locs_per_cycle_from_localizations
    k = LPC_RULE_RADIUS_IN_NENA if radius_in_nena is None else float(radius_in_nena)
    return locs_per_cycle_from_localizations(loc, nena_nm=nena_nm, radius_in_nena=k)


def c_label_origin(c_label: float) -> str:
    """Where a c_label comes from, in the words of ``tools.dbscan_rule.derive_rule``: the literature preset for 14
    and 26 (Dempsey et al. 2011), "[user] typed" for any other value."""
    from tools.dbscan_rule import C_LABEL_BME, C_LABEL_MEA
    if math.isclose(float(c_label), C_LABEL_MEA):
        return "[lit: Dempsey et al. 2011, Alexa Fluor 647 in MEA] provisional, not measured here"
    if math.isclose(float(c_label), C_LABEL_BME):
        return "[lit: Dempsey et al. 2011, Alexa Fluor 647 in beta-ME] sensitivity, not measured here"
    return "[user] typed"


RULE_LABEL = "DBSCAN rule (c_label {c:g})"


def rule_method_specs(rule: Optional[Any], rule_c26: Optional[Any] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """The DBSCAN rule's runs for 'Compare methods...' (the double table, D2: 25/10 is the reference and is always
    run first): the rule with c_label 14 and, when given, with 26 (sensitivity, D1)."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    for r in (rule, rule_c26):
        if r is not None:
            out.append((mm.REFERENCE_METHOD, {"eps": float(r.eps_nm), "min_samples": int(r.min_samples)}))
    return out


def dbscan_mode(analysis: Any, rule: Optional[Any] = None) -> str:
    """How the analysis' DBSCAN parameters were chosen: "compatibility 25/10", "rule", "typed", or "not DBSCAN"."""
    if str(getattr(analysis, "cluster_method", "DBSCAN")) != mm.REFERENCE_METHOD:
        return "not DBSCAN"
    eps, ms = float(analysis.eps_nm), int(analysis.min_samples)
    if eps == float(mm.REFERENCE_PARAMS["eps"]) and ms == int(mm.REFERENCE_PARAMS["min_samples"]):
        return "compatibility 25/10"
    if rule is not None and abs(eps - float(rule.eps_nm)) < 1e-6 and ms == int(rule.min_samples):
        return "rule"
    return "typed"


def run_of(analysis: Any, axon_id: str = "") -> mm.MethodRun:
    """The analysis shown as a :class:`tools.multimethod.MethodRun` (its own method and parameters; DBSCAN's eps /
    min samples as its params, so the reference is recognised)."""
    method = str(getattr(analysis, "cluster_method", mm.REFERENCE_METHOD))
    if method == mm.REFERENCE_METHOD:
        params: Dict[str, Any] = {"eps": float(analysis.eps_nm), "min_samples": int(analysis.min_samples)}
    else:
        params = dict(getattr(analysis, "cluster_params", {}) or {})
    ref = mm._is_reference(method, params)
    key = mm.REFERENCE_KEY if ref else method_label(method)
    return mm.MethodRun(method=method, params=params, key=key, axon_id=axon_id, is_reference=ref, analysis=analysis,
                        labels=np.asarray(analysis.labels, dtype=np.int64))


# =====================================================================================================================
# The comparison
# =====================================================================================================================

@dataclass
class ComparisonResult:
    """One axon, several methods: the runs, the long table and its views (``tools.multimethod``)."""
    axon_id: str
    runs: List[mm.MethodRun]
    analyses: List[str]
    long_rows: List[Dict[str, Any]]
    wide_rows: List[Dict[str, Any]]
    agreement_rows: List[Dict[str, Any]]
    summary_rows: List[Dict[str, Any]]
    labels: Dict[str, str] = field(default_factory=dict)      # run key -> how the user calls it (rule runs)
    seconds: float = 0.0

    def run(self, key: str) -> Optional[mm.MethodRun]:
        return next((r for r in self.runs if r.key == key), None)

    def depends_on_method(self) -> List[str]:
        """Readouts whose verdict differs between methods ("yes" in the summary)."""
        return [str(r["readout"]) for r in self.summary_rows if r.get("depends_on_method") == "yes"]

    def double_table(self) -> List[Dict[str, Any]]:
        """D2: the reference 25/10 and the DBSCAN rule runs, one row each (the wide view's rows)."""
        keep = {mm.REFERENCE_KEY} | {k for k, v in self.labels.items() if v.startswith("DBSCAN rule")}
        rows = [dict(r) for r in self.wide_rows if r.get("method_key") in keep]
        for r in rows:
            r["label"] = self.labels.get(str(r.get("method_key")), "DBSCAN 25/10 (reference)")
        return rows


def run_comparison(run_fn: mm.RunFn, methods: Sequence[mm.MethodSpec], *, n_locs: int, axon_id: str = "",
                   params_for: Optional[Callable[[str], Dict[str, Any]]] = None, allow_slow: bool = False,
                   analyses: Sequence[str] = DEFAULT_ANALYSES, extras: Optional[Mapping[str, Any]] = None,
                   seed: int = mm.DEFAULT_SEED, cancelled: Optional[Callable[[], bool]] = None,
                   on_method_done: Optional[Callable[[int, int, mm.MethodRun], None]] = None,
                   labels: Optional[Mapping[str, str]] = None, plan: Optional[mm.RunPlan] = None) -> ComparisonResult:
    """The methods on one axon (DBSCAN 25/10 first), then the analyses on every run that worked.

    A cancellation during the analyses (``tools.mps_cancel.AnalysisCancelled``) propagates; one during the runs
    leaves the remaining runs "cancelled" and the analyses still run on the finished ones."""
    register_all_plugins()
    t0 = time.perf_counter()
    runs = mm.run_methods(run_fn, methods, n_locs=int(n_locs), axon_id=axon_id, params_for=params_for,
                          allow_slow=allow_slow, cancelled=cancelled, on_done=on_method_done, plan=plan)
    names = [n for n in analyses if n in mm.registered_analyses()]
    long_rows = mm.apply_analyses(runs, names, seed=seed, allow_slow=allow_slow, cancelled=cancelled,
                                  extras=dict(extras or {}))
    return ComparisonResult(axon_id=axon_id, runs=runs, analyses=names, long_rows=long_rows,
                            wide_rows=mm.wide_view(long_rows), agreement_rows=mm.pairwise_agreement(runs),
                            summary_rows=mm.conclusion_summary(long_rows), labels=dict(labels or {}),
                            seconds=float(time.perf_counter() - t0))


def rule_labels(plan: mm.RunPlan, rule: Optional[Any], rule_c26: Optional[Any] = None) -> Dict[str, str]:
    """Run key -> "DBSCAN rule (c_label 14)" for the rule's runs of ``plan`` (keys are the method + params JSON)."""
    out: Dict[str, str] = {}
    for r in (rule, rule_c26):
        if r is None:
            continue
        for key, method, params, ref in plan.jobs:
            if (method == mm.REFERENCE_METHOD and not ref and abs(float(params.get("eps", -1)) - float(r.eps_nm)) < 1e-9
                    and int(params.get("min_samples", -1)) == int(r.min_samples)):
                c = float(r.inputs["c_label"].value)
                out[key] = RULE_LABEL.format(c=c) if key not in out else out[key] + f" = {RULE_LABEL.format(c=c)}"
    return out


COMPARISON_SUFFIXES: Dict[str, str] = {"long": "_methods_long.csv", "agreement": "_methods_agreement.csv",
                                       "summary": "_methods_summary.csv", "wide": "_methods_wide.csv"}


def comparison_paths(stem: str) -> Dict[str, str]:
    """Where :func:`write_comparison` writes, given a path stem (no extension)."""
    base = os.path.splitext(stem)[0] if stem.lower().endswith(".csv") else stem
    return {k: base + s for k, s in COMPARISON_SUFFIXES.items()}


def write_comparison(stem: str, result: ComparisonResult,
                     extra_long_rows: Sequence[Mapping[str, Any]] = ()) -> Dict[str, str]:
    """``<stem>_methods_long.csv`` (A4: every readout, one row per axon x method x readout), the agreement and the
    summary beside it, and the wide view (one row per method, as the table before). ``extra_long_rows``: rows of the
    same layout to append (the null's, :func:`null_long_rows`). Never overwrites (``stem_2.csv``, ...)."""
    paths = comparison_paths(stem)
    long_rows = list(result.long_rows) + [dict(r) for r in extra_long_rows]
    out = {"long": mm.write_long_csv(paths["long"], long_rows)}
    out["agreement"] = mm.write_rows_csv(paths["agreement"], result.agreement_rows, mm.AGREEMENT_COLUMNS)
    out["summary"] = mm.write_rows_csv(paths["summary"], mm.conclusion_summary(long_rows), mm.SUMMARY_COLUMNS)
    wide = mm.wide_view(long_rows)
    cols: List[str] = list(mm.WIDE_HEAD_COLUMNS)
    for r in wide:
        cols += [c for c in r if c not in cols]
    out["wide"] = mm.write_rows_csv(paths["wide"], wide, cols)
    return out


# =====================================================================================================================
# The pipeline null of each method's run, with the regularity reading on the same null
# =====================================================================================================================

NOT_RUN = "not run"
# The words the batch's stage bar shows while a null runs (tools.mps_progress, F1).
NULL_PROGRESS_STAGE = "per-axon null (simulated axons)"


@dataclass
class NullOutcome:
    """The null of one method's run: the result (``PipelineNullResult``), the regularity reading on it
    (``GapRegularityResult``), or why there is none."""
    key: str
    method: str
    params: Dict[str, Any]
    result: Optional[Any] = None
    gap: Optional[Any] = None
    status: str = NOT_RUN                        # ok | refused | error | cancelled | not run
    message: str = ""
    estimate_s: float = 0.0
    seconds: float = 0.0
    n_replicates: int = 0
    closure: str = NULL_CLOSURE                  # C3 / C4: the null's closure name (n_locs | k_clusters)
    membrane: str = ""                           # C2: what the null's membrane was (tools.null_membrane words)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.result is not None


def null_estimate_seconds(run: mm.MethodRun, n_replicates: int, closure: str = NULL_CLOSURE,
                          gap_stage2: str = GAP_STAGE2_WINDOW) -> float:
    """A first estimate before the null runs (``tools.mps_pipeline_null.check_method``'s table figure, a lower bound
    for the slow methods), in seconds: per closure (``null_closures``) the B replicates plus, for the K closure, its
    pilots (N_PILOTS x PILOT_AXONS_K pipeline runs; the N_loc closure's pilots only simulate); plus, when stage 2 of
    the regularity runs at the pipeline level (the Batch), its upper bound ((PIPELINE_BISECTION_STEPS + 1) x
    PIPELINE_BISECTION_N + B analyses: it runs only when stage 1 says "more regular")."""
    from tools import mps_gap_measures as gm
    from tools import mps_pipeline_null as pn
    an = run.analysis
    n = int(getattr(an, "n_locs_slab", 0) or 0)
    per = pn.estimated_seconds_per_replicate(run.method, n)
    total = 0.0
    for c in null_closures(closure):
        total += per * (int(n_replicates) + (pn.N_PILOTS * pn.PILOT_AXONS_K if c == "k_clusters" else 0))
        if gap_stage2 in ("pipeline", "both"):
            total += per * ((gm.PIPELINE_BISECTION_STEPS + 1) * gm.PIPELINE_BISECTION_N + int(n_replicates))
    return float(total)


def run_null_for(run: mm.MethodRun, *, n_replicates: int, nena_nm: Optional[float], seed: int = mm.DEFAULT_SEED,
                 c_label: Optional[float] = None, allow_slow: bool = False,
                 cancelled: Optional[Callable[[], bool]] = None,
                 progress: Optional[Callable[[int, int], None]] = None,
                 gap_stage2: str = GAP_STAGE2_WINDOW, frame_slab: Optional[Any] = None,
                 closure: str = NULL_CLOSURE, membrane: str = NULL_MEMBRANE,
                 membrane_curve: Optional[Any] = None) -> NullOutcome:
    """The pipeline null of ``run`` (its own method, parameters, curation and contour: ``PipelineSpec.from_analysis``)
    and the two-stage regularity reading on THE SAME null (D7, G3). ``closure``: "n_locs" (default) or "k"
    ("k_clusters"); for "both" call :func:`run_nulls_by_closure`. Never raises but for a cancellation
    (``tools.mps_cancel.AnalysisCancelled``): a refusal (slow method, K < 8, External labels) or an error is the
    outcome's status and message."""
    from tools import mps_cancel, mps_progress
    from tools import mps_pipeline_null as pn
    from tools.dbscan_rule import C_LABEL_DEFAULT
    register_all_plugins()
    closures = null_closures(closure)
    if len(closures) != 1:
        raise ValueError("run_null_for runs one closure; use run_nulls_by_closure for 'both'")
    out = NullOutcome(key=run.key, method=run.method, params=dict(run.params), n_replicates=int(n_replicates),
                      closure=closures[0])
    if not run.ok:
        out.status, out.message = "error", f"the method's own analysis did not run ({run.status}: {run.error})"
        return out
    out.estimate_s = null_estimate_seconds(run, n_replicates, closures[0], gap_stage2)
    an = run.analysis

    def tick(done: int, total: int) -> None:
        if cancelled is not None and cancelled():
            raise mps_cancel.AnalysisCancelled(f"null of {run.key}: replicate {done} of {total}")
        mps_progress.notify(NULL_PROGRESS_STAGE, done, total)
        if progress is not None:
            progress(done, total)

    # The batch's stage bar (tools.mps_progress, F1) hears the null's own count, not the check points of every
    # simulated axon's analysis (they would make the words jump between "clustering" and "perimeter").
    outer = mps_progress.current_observer()

    def only_the_null(where: str, done: Optional[int], total: Optional[int]) -> None:
        if where == NULL_PROGRESS_STAGE and outer is not None:
            outer(where, done, total)

    # the run's own method and parameters, the analysis' slab, curation and contour settings (as the pipeline_null
    # plug-in builds it): the null of THIS method
    base = pn.PipelineSpec.from_analysis(an)
    spec = pn.PipelineSpec.for_method(str(run.method), dict(run.params or {}), **{
        k: getattr(base, k) for k in ("slab_half_width_nm", "dbcv_threshold", "roi", "mahalanobis_threshold",
                                      "ellipse_mode", "sigma_cap_fraction", "all_starts", "main_peak_mode")})
    t0 = time.perf_counter()
    model_kw = _membrane_kw(run, membrane, membrane_curve, out)
    with mps_progress.observing(only_the_null if outer is not None else None):
        _null_and_reading(out, an, spec, tick, n_replicates=n_replicates, nena_nm=nena_nm, seed=seed,
                          c_label=c_label, allow_slow=allow_slow, gap_stage2=gap_stage2, frame_slab=frame_slab,
                          closure=closures[0], model_kw=model_kw)
    out.seconds = float(time.perf_counter() - t0)
    return out


def _membrane_kw(run: mm.MethodRun, membrane: str, curve: Optional[Any], out: "NullOutcome") -> Optional[Dict[str, Any]]:
    """C2: the null model's membrane for ``run``. The DBSCAN 25/10 run under the default keeps the null's own
    reading (the P-spline through its own centroids: the same curve, its null unchanged); every other run gets the
    common curve (``curve`` when the caller built it once for every method, else :mod:`tools.null_membrane` builds
    it here). Falls back to the run's own centroids, said in ``out.membrane``."""
    from tools import null_membrane as nmb
    choice = nmb.check_choice(membrane)
    if choice == nmb.MEMBRANE_DBSCAN and run.is_reference:
        out.membrane = nmb.MEMBRANE_WORDS[nmb.MEMBRANE_DBSCAN]
        return None
    if curve is None:
        nm = nmb.null_membrane_for(run.analysis, (), choice, with_deformation=False)
        if nm.curve_nm is None:
            out.membrane = f"own centroids ({nm.note})"
            return None
        curve, source = nm.curve_nm, nm.source
    else:
        source = ("read: robust P-spline fitted to the slab's localizations (C2 option)"
                  if choice == nmb.MEMBRANE_LOCALIZATIONS else
                  "read: 600 nm P-spline through the centroids of DBSCAN 25/10, common to every method (C2)")
    out.membrane = nmb.MEMBRANE_WORDS[choice]
    return {"membrane_nm": np.asarray(curve, dtype=np.float64), "membrane_origin": source}


def run_nulls_by_closure(run: mm.MethodRun, *, closure: str = NULL_CLOSURE, **kw: Any) -> Dict[str, NullOutcome]:
    """C3 / C4: :func:`run_null_for` once per closure of ``closure`` ("n_locs", "k" or "both"), keyed by the null's
    closure name ("n_locs", "k_clusters"), n_locs first. "both" costs the two nulls (the K closure's pilots run the
    whole pipeline, ``null_estimate_seconds`` says how much); the same seed for both, so the two differ only by the
    closure."""
    return {c: run_null_for(run, closure=c, **kw) for c in null_closures(closure)}


def _null_and_reading(out: "NullOutcome", an: Any, spec: Any, tick: Callable[[int, int], None], *, n_replicates: int,
                      nena_nm: Optional[float], seed: int, c_label: Optional[float], allow_slow: bool,
                      gap_stage2: str, frame_slab: Optional[Any], closure: str = NULL_CLOSURE,
                      model_kw: Optional[Dict[str, Any]] = None) -> None:
    """The null, then the regularity reading on it, into ``out`` (refusals and errors as its status; a cancellation
    propagates)."""
    from tools import mps_cancel
    from tools import mps_pipeline_null as pn
    from tools.dbscan_rule import C_LABEL_DEFAULT
    try:
        res = pn.run_pipeline_null(
            an, spec=spec, nena_nm=nena_nm, frame_slab=frame_slab, n_replicates=int(n_replicates), seed=int(seed),
            closure=str(closure), c_label=float(C_LABEL_DEFAULT if c_label is None else c_label),
            allow_slow=bool(allow_slow), progress=tick, global_test_from=NULL_GLOBAL_TEST_FROM,
            keep_arcs=NULL_KEEP_ARCS, model_kw=model_kw)
    except pn.SlowMethodRefused as exc:
        out.status, out.message, out.estimate_s = "refused", str(exc), float(exc.estimate_s)
        return
    except pn.NullRefused as exc:
        out.status, out.message = "refused", str(exc)
        return
    except mps_cancel.AnalysisCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - one method's failure is its row, never the others'
        out.status, out.message = "error", f"{type(exc).__name__}: {exc}"
        return
    out.result, out.status = res, "ok"
    out.message = res.headline()
    try:
        from tools import mps_gap_measures as gm
        out.gap = gm.analyze_gap_regularity(an, null_result=res, stage2=gap_stage2, seed=int(seed), spec=spec,
                                            n_replicates=int(n_replicates), n_matched=int(n_replicates),
                                            nena_nm=nena_nm, allow_slow=bool(allow_slow),
                                            progress=lambda what, i, n: tick(i, n))
    except mps_cancel.AnalysisCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - the null stands without the reading
        out.message += f" (regularity reading failed: {type(exc).__name__}: {exc})"


def run_nulls(runs: Sequence[mm.MethodRun], *, n_replicates: int, nena_nm: Optional[float],
              seed: int = mm.DEFAULT_SEED, c_label: Optional[float] = None, allow_slow: bool = False,
              cancelled: Optional[Callable[[], bool]] = None,
              progress: Optional[Callable[[str, int, int], None]] = None,
              gap_stage2: str = GAP_STAGE2_WINDOW, closure: str = NULL_CLOSURE, membrane: str = NULL_MEMBRANE,
              membrane_curve: Optional[Any] = None) -> Dict[str, NullOutcome]:
    """:func:`run_null_for` for every run and every closure of ``closure`` (C3 / C4: "n_locs", "k" or "both"), in
    order; keyed by the run's key, and with "both" the K closure's outcome under ``closure_key(key, "k_clusters")``.
    ``membrane`` / ``membrane_curve``: C2 (one curve for every method). After a cancellation the rest are
    "cancelled"."""
    from tools import mps_cancel
    out: Dict[str, NullOutcome] = {}
    stop = False
    for r in runs:
        if stop or (cancelled is not None and cancelled()):
            stop = True
            out[r.key] = NullOutcome(key=r.key, method=r.method, params=dict(r.params), status="cancelled",
                                     message="cancelled before this method's null ran")
            continue
        tick: Optional[Callable[[int, int], None]] = None
        if progress is not None:
            report = progress

            def tick(d: int, t: int, k: str = r.key) -> None:
                report(k, d, t)
        for c in null_closures(closure):
            key = closure_key(r.key, c, closure)
            if stop:
                out[key] = NullOutcome(key=key, method=r.method, params=dict(r.params), status="cancelled",
                                       message="cancelled before this method's null ran", closure=c)
                continue
            try:
                o = run_null_for(
                    r, n_replicates=n_replicates, nena_nm=nena_nm, seed=seed, c_label=c_label, allow_slow=allow_slow,
                    cancelled=cancelled, gap_stage2=gap_stage2, progress=tick, closure=c, membrane=membrane,
                    membrane_curve=membrane_curve)
                o.key = key
                out[key] = o
            except mps_cancel.AnalysisCancelled:
                stop = True
                out[key] = NullOutcome(key=key, method=r.method, params=dict(r.params), status="cancelled",
                                       message="cancelled while this method's null ran", closure=c)
    return out


K_CLOSURE_SUFFIX = " [K closure]"


def closure_key(key: str, closure: str, choice: str = NULL_CLOSURE) -> str:
    """The key of one closure's null in :func:`run_nulls`'s result: the run's own key for its first closure (n_locs
    under "both", the only one otherwise), the key plus ``K_CLOSURE_SUFFIX`` for the K closure under "both"."""
    if choice == "both" and closure == "k_clusters":
        return key + K_CLOSURE_SUFFIX
    return key


def closure_pairs(outcomes: Mapping[str, NullOutcome]) -> List[Tuple[str, NullOutcome, NullOutcome]]:
    """C4: (run key, N_loc outcome, K outcome) for every run whose two closures ran (``run_nulls`` with "both")."""
    out = []
    for key, o in outcomes.items():
        k = outcomes.get(key + K_CLOSURE_SUFFIX)
        if k is not None and not key.endswith(K_CLOSURE_SUFFIX):
            out.append((key, o, k))
    return out


def closure_comparison_rows(outcomes: Mapping[str, NullOutcome]) -> List[Dict[str, Any]]:
    """C4: one row per run and test with the p of each closure side by side (H1, H2, the five readouts, the
    regularity's stage 1), and the difference of their global envelopes' largest excursion -- what the panel's
    'Compare closures' view shows and writes."""
    rows: List[Dict[str, Any]] = []
    for key, a, b in closure_pairs(outcomes):
        def p_of(o: NullOutcome, what: str) -> Optional[float]:
            if not o.ok:
                return None
            r = o.result
            if what in ("h1", "h2"):
                env = getattr(r, what, None)
                return None if env is None else _num(env.p_global)
            if what == "gap_cv":
                return None if o.gap is None else _num(o.gap.export_columns().get("gap_cv_p_mc"))
            return _num(r.p_values.get(what))

        def hmax(o: NullOutcome, what: str) -> Optional[float]:
            env = getattr(o.result, what, None) if o.ok else None
            return None if env is None else _num(env.h_max)

        for what, words in (("h1", "H1 global (arc)"), ("h2", "H2 global (band)"), ("k_per_um", "clusters per um"),
                            ("nn_median", "median 1NN"), ("occupancy", "occupancy"), ("regularity", "regularity r"),
                            ("gap_cv", "gap CV (stage 1)")):
            pa, pb = p_of(a, what), p_of(b, what)
            rows.append({"method": key, "test": words, "p_n_locs": pa, "p_k_clusters": pb,
                         "same_verdict_at_0_05": None if pa is None or pb is None else (pa <= 0.05) == (pb <= 0.05),
                         "h_max_n_locs": hmax(a, what) if what in ("h1", "h2") else None,
                         "h_max_k_clusters": hmax(b, what) if what in ("h1", "h2") else None})
    return rows


def null_long_rows(runs: Sequence[mm.MethodRun], outcomes: Mapping[str, NullOutcome]) -> List[Dict[str, Any]]:
    """The nulls as rows of the long comparison table (``tools.multimethod.LONG_COLUMNS``, analysis
    "pipeline_null" with the plug-in's readout names), so the export carries them beside every other readout."""
    from tools import mps_pipeline_null as pn
    register_all_plugins()
    specs = {s.name: s for s in mm.readout_specs(["pipeline_null"])}
    rows: List[Dict[str, Any]] = []
    for r in runs:
        o = outcomes.get(r.key)
        if o is None:
            continue
        values: Dict[str, Any] = {}
        if o.ok:
            d = o.result.export_dict()
            values = {new: d.get(old) for old, new in pn.MULTIMETHOD_NAMES.items()}
            values["null_closure_ratio"] = o.result.closure_ratio
            values["null_seconds_per_replicate_s"] = o.result.seconds_per_replicate
            values["null_interpretable_verdict"] = (pn.INTERPRETABLE_LEVELS[0] if o.result.interpretable
                                                    else pn.INTERPRETABLE_LEVELS[1])
        for name, s in specs.items():
            v = mm._coerce(s, values.get(name))
            rows.append({"axon_id": r.axon_id, "method_key": r.key, "cluster_method": r.method,
                         "cluster_params": r.params_json, "is_reference": bool(r.is_reference),
                         "run_status": r.status, "analysis_name": "pipeline_null", "readout": name, "kind": s.kind,
                         "value": v, "unit": s.unit, "alpha": s.alpha, "verdict": s.verdict(v),
                         "error": "" if o.ok else o.message, "seconds": round(float(o.seconds), 3)})
    return rows


def edge_note(analysis: Any) -> str:
    """G5: the clusters the edge curation removed, said beside the regularity reading (they may leave contour
    stretches empty: the gap measures then see a gap that the axon does not have)."""
    rep = getattr(analysis, "bad_report", None)
    n = len(getattr(rep, "edge_touching", ()) or ()) if rep is not None else 0
    if n <= 0:
        return ""
    return (f"{n} cluster(s) removed at the ROI edge: stretches of the contour may be missing, which lengthens gaps; "
            "draw the ROI with a margin around the axon")


# =====================================================================================================================
# The axon table's new block (v13)
# =====================================================================================================================

SCALE_COLUMNS: Tuple[str, ...] = ("scale_eps_nm", "scale_min_samples", "scale_nena_nm", "scale_nena_source",
                                  "scale_fusion_nm", "scale_d50_nm", "scale_slab_nm")
METHODS_COLUMNS: Tuple[str, ...] = (
    "dbscan_mode", "eps_rule_nm", "min_samples_rule", "min_samples_rule_c26", "c_label_used", "c_label_source",
    "loc_per_cycle_used", "loc_per_cycle_source", "loc_per_cycle_n_events", "p_neighbour", "nena_nm_used",
    "cycle_quantile", "link_radius_nena", "link_max_dark_frames", "methods_compared", "methods_long_file",
    "methods_depends_on_method")
# The per-axon readouts of three plug-ins, under their own names (prefixes events_ / eclipse_ / size_).
PLUGIN_BLOCKS: Tuple[str, ...] = ("events", "eclipse", "size")
NULL_HEAD_COLUMNS: Tuple[str, ...] = (
    "null_type", "null_method", "null_closure", "null_B", "null_seed", "null_c_label", "null_locs_per_cycle",
    "null_locs_per_cycle_source", "null_hard_core_nm", "null_interpretable", "null_global_test_from",
    "null_global_r_min_nm", "ripley_points", "ripley1d_window", "ripley1d_rstar_nm", "ripley1d_hmax",
    "ripley1d_p_global", "ripley2d_window", "ripley2d_rstar_nm", "ripley2d_hmax", "ripley2d_p_global")
# The readouts the null tests (tools.mps_pipeline_null.READOUTS once every plug-in registered: A9's five, the size
# statistic and the gap readouts), in a fixed order so the table keeps one layout.
NULL_READOUT_NAMES: Tuple[str, ...] = ("k_per_um", "nn_median", "reff_median", "occupancy", "regularity",
                                       "size_nloc_nn_spearman", "gap_cv", "gap_rho1", "gap_excess_cv", "bragg_s",
                                       "elongation_p95")
# v15 (decision E5a, 2026-10-29): question (b) of the apparent size on the SAME null (tools.size_curvature_null); its
# p_mc_ / null_median_ / p_mc_*_ignoring_failed columns go to the end of the table (v15_columns) so nothing moves.
NULL_READOUT_NAMES_V15: Tuple[str, ...] = ("size_nloc_kappa_r_spearman",)
ALL_NULL_READOUT_NAMES: Tuple[str, ...] = NULL_READOUT_NAMES + NULL_READOUT_NAMES_V15
NULL_COLUMNS: Tuple[str, ...] = (NULL_HEAD_COLUMNS + tuple(f"p_mc_{n}" for n in NULL_READOUT_NAMES)
                                 + tuple(f"null_median_{n}" for n in NULL_READOUT_NAMES))
# GapRegularityResult.export_columns (regularity plan sec. 6.2, P-R86), plus the edge note (G5).
GAP_COLUMNS: Tuple[str, ...] = (
    "gap_cv", "gap_cv_null_mean", "gap_cv_null_lo", "gap_cv_null_hi", "gap_cv_p_mc", "gap_cv_z",
    "gap_rho1", "gap_rho1_null_mean", "gap_rho1_null_lo", "gap_rho1_null_hi", "gap_rho1_z",
    "gap_excess_cv", "gap_excess_cv_null_mean", "gap_excess_cv_null_lo", "gap_excess_cv_null_hi",
    "gap_excess_cv_p_mc", "gap_excess_cv_z", "bragg_s", "bragg_s_null_mean", "bragg_s_null_lo", "bragg_s_null_hi",
    "bragg_s_z", "gap_rho1_p_mc_matched", "gap_rho1_z_matched", "bragg_p_mc_matched", "bragg_s_z_matched", "bragg_m",
    "bragg_spacing_nm", "s_low", "gap_cv_polygon", "gap_rho1_polygon", "gap_excess_cv_polygon", "bragg_s_polygon",
    "regularity_reading", "regularity_flag_min_below_contact", "matched_null_kind", "matched_null_d_nm",
    "matched_null_B", "matched_null_capped", "pipeline_null_B", "pipeline_null_K_sim", "pipeline_null_d_contact_nm",
    "pipeline_null_epitope_R_nm", "pipeline_null_lp_nm", "pipeline_null_K_det_mean", "pipeline_null_closure_ratio",
    "pipeline_null_seed", "n_clusters_elongated", "elongation_threshold", "n_clusters_elongated_null_expected",
    "nn_regularity_null_kind", "regularity_edge_note")


# What the methods block says when there is nothing to say (an empty cell read as a lost value).
METHODS_NOT_RUN = "not run"
METHODS_TABLES_NOT_WRITTEN = "not written ('Export tables...' in Compare methods)"

# v14 (plan A8 completed, 2026-10-23): what the reporting checklist (docs/reporting_checklist.md) still lacked,
# appended after the null so no v13 column moves. How the NeNA was estimated; the event linking radius in nm (the
# v13 block has it only in NeNA units); and P-R103, the occupancy re-expressed as the fraction of the contour within
# 25 nm of a kept cluster's centre (tools.mps_occupancy.arc_fraction_within; occupancy_percent is unchanged).
REPORT_COLUMNS: Tuple[str, ...] = ("nena_method", "link_radius_nm", "occupancy_arc_fraction_within_25nm")
# v15 (2026-10-29, the headless fixes after the design review), appended after v14's last column so nothing earlier
# moves. G1 (c): how many replicates of the null failed and, for every p the null gives, the same p ignoring them
# (descriptive; the p_mc_* / *_p_global columns count the failures in the worst case).
NULL_FAILED_COLUMNS: Tuple[str, ...] = (
    ("null_failed_replicates",) + tuple(f"p_mc_{n}_ignoring_failed" for n in NULL_READOUT_NAMES)
    + ("ripley1d_p_global_ignoring_failed", "ripley2d_p_global_ignoring_failed",
       "gap_cv_p_mc_ignoring_failed", "gap_excess_cv_p_mc_ignoring_failed", "gap_rho1_p_mc_matched_ignoring_failed",
       "bragg_p_mc_matched_ignoring_failed", "matched_null_failed_replicates"))


# v15, second part (feat/validations-d2-e1-e5, 2026-10-29), after the first part so nothing moves: E5's null columns
# (its p counts the failed replicates as G1 says; the companion ignores them) and the plug-in readouts the branch
# added (D2: what the eclipse arc test did with merged-looking clusters; E5: question (b)'s statistic and its p).
E5_NULL_COLUMNS: Tuple[str, ...] = (
    tuple(f"p_mc_{n}" for n in NULL_READOUT_NAMES_V15) + tuple(f"null_median_{n}" for n in NULL_READOUT_NAMES_V15)
    + tuple(f"p_mc_{n}_ignoring_failed" for n in NULL_READOUT_NAMES_V15))
V15_PLUGIN_COLUMNS: Tuple[str, ...] = ("eclipse_merged_handling", "eclipse_n_merged_split",
                                       "size_nloc_kappa_r_spearman", "size_nloc_kappa_r_null_p")


# Where the regularity's stage 2 was set to run (P-R87 c: "centres" in the axon window, "pipeline" in the Batch); the
# two levels read 4 of 60 synthetic axons differently, so a row says which it is (design review 2026-10-29).
REGULARITY_LEVEL_COLUMNS: Tuple[str, ...] = ("regularity_stage2_level",)


# B3 (c): the localizations per cycle by depth, a diagnostic beside the rule (tools.lpc_depth; the rule keeps the
# whole file's value).
LPC_DEPTH_NOT_HERE = "not computed here: the Batch computes it per file (the whole file is needed)"


# v16 (the interface batch, 2026-11-07), appended after v15's last column so nothing earlier moves. B1: the radius
# the rule's localizations per cycle were measured at (the events / fusion radius is link_radius_nena, v13) and
# whether either radius was edited (EXPLORATORY). C2: the membrane the null used and the deformation indicator
# between the DBSCAN 25/10 centroid membrane and the localization membrane.
V16_COLUMNS: Tuple[str, ...] = ("lpc_rule_radius_nena", "radii_exploratory", "null_membrane",
                                "membrane_deformation_mean_nm", "membrane_deformation_max_nm",
                                "membrane_deformation_n_flagged")


def null_by_closure_columns() -> Tuple[str, ...]:
    """The columns of ``<table>_null_by_closure.csv`` (C3 / C4: one row per axon x closure): who the axon is, then
    the null's columns (null_closure among them), the failure columns (G1) and the regularity reading on that null."""
    head = ("axon_id", "analysis_id", "source", "cluster_method", "null_status")
    return head + NULL_COLUMNS + NULL_FAILED_COLUMNS + GAP_COLUMNS + REGULARITY_LEVEL_COLUMNS + E5_NULL_COLUMNS


def null_closure_rows(head: Mapping[str, Any], outcomes: Mapping[str, NullOutcome]) -> List[Dict[str, Any]]:
    """One row per closure of ``outcomes`` (``run_nulls_by_closure``) for the axon ``head`` (its axon_id,
    analysis_id, source, cluster_method), in ``null_by_closure_columns()`` order; a refused or failed null is its row
    with ``null_status`` and the reason in ``null_type``."""
    cols = null_by_closure_columns()
    rows: List[Dict[str, Any]] = []
    for closure, o in outcomes.items():
        row: Dict[str, Any] = {c: None for c in cols}
        for k in ("axon_id", "analysis_id", "source", "cluster_method"):
            row[k] = head.get(k)
        row["null_closure"] = closure
        row["null_status"] = o.status
        if o.ok:
            row.update({k: v for k, v in o.result.export_dict().items() if k in row})
            if o.gap is not None:
                row.update({k: v for k, v in o.gap.export_columns().items() if k in row})
        else:
            row["null_type"] = f"none ({o.status}: {o.message})"
        rows.append({c: row.get(c) for c in cols})
    return rows


def v15_columns() -> Tuple[str, ...]:
    """The v15 block, in order (appended after ``REPORT_COLUMNS``): the core fixes' part (G1 failures, the
    regularity's stage-2 level, lpc by depth), then the validations' part (E5's null columns, D2 / E5 readouts)."""
    from tools.lpc_depth import LPC_DEPTH_COLUMNS
    return (NULL_FAILED_COLUMNS + REGULARITY_LEVEL_COLUMNS + LPC_DEPTH_COLUMNS + E5_NULL_COLUMNS
            + V15_PLUGIN_COLUMNS)


def v16_columns() -> Tuple[str, ...]:
    """The v16 block (``V16_COLUMNS``), appended after ``v15_columns``."""
    return V16_COLUMNS
# The estimator every caller's NeNA comes from (tools.mps_cut_clusters.nena_from_localizations -> tools.mps_quality
# .nena; docs/nena.md); whether the nearby term was kept (BIC) is shown in the Data quality window, not here.
NENA_METHOD_MEASURED = ("NeNA (Endesfelder et al. 2014): maximum likelihood on the next-frame distances, Picasso's "
                        "model, nearby term kept only when BIC asks for it (tools.mps_quality.nena)")
NENA_METHOD_NONE = "none measured: the marks use the paper's 12 nm (Gazal et al. 2026)"


def arc_fraction_of(analysis: Any, distance_nm: Optional[float] = None) -> Optional[float]:
    """P-R103: the fraction of the analysis' contour (``perimeter.contour``, the closed polygon through the kept
    clusters' centres of mass in tour order -- the same curve as ``perimeter_um`` and ``occupancy_percent``) within
    ``distance_nm`` (default 25 nm) of a kept cluster's centre (``centroids``). None without a contour."""
    from tools.mps_occupancy import ARC_FRACTION_DISTANCE_NM, arc_fraction_within
    per = getattr(analysis, "perimeter", None)
    contour = None if per is None else getattr(per, "contour", None)
    if contour is None or len(contour) < 3:
        return None
    centres = getattr(analysis, "centroids", None)
    if centres is None:
        centres = contour
    d = ARC_FRACTION_DISTANCE_NM if distance_nm is None else float(distance_nm)
    try:
        return arc_fraction_within(np.asarray(contour, dtype=np.float64), np.asarray(centres, dtype=np.float64), d)
    except ValueError:
        return None


def plugin_columns() -> Tuple[str, ...]:
    """The per-axon readouts of ``PLUGIN_BLOCKS`` in registration order (their names are the columns), but those
    added in v15 (``V15_PLUGIN_COLUMNS``), which sit at the end of the table so no earlier column moves."""
    register_all_plugins()
    return tuple(s.name for s in mm.readout_specs(list(PLUGIN_BLOCKS)) if s.name not in V15_PLUGIN_COLUMNS)


def integration_columns() -> Tuple[str, ...]:
    """Every column of the block, in order: scale, methods (A8), events / eclipse / size, regularity, null (v13),
    the reporting checklist's last three (v14, ``REPORT_COLUMNS``) and the v15 block (``v15_columns``)."""
    return (SCALE_COLUMNS + METHODS_COLUMNS + plugin_columns() + GAP_COLUMNS + NULL_COLUMNS + REPORT_COLUMNS
            + v15_columns() + v16_columns())


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def axon_columns(analysis: Any, *, nena_nm: Optional[float] = None, nena_source: str = "",
                 c_label: Optional[float] = None, locs_per_cycle: Optional[Any] = None,
                 null: Optional[NullOutcome] = None, comparison: Optional[ComparisonResult] = None,
                 comparison_files: Optional[Mapping[str, str]] = None, seed: int = mm.DEFAULT_SEED,
                 nena_method: Optional[str] = None, lpc_profile: Optional[Any] = None,
                 lpc_profile_note: str = "", lpc_radius_in_nena: Optional[float] = None,
                 events_radius_in_nena: Optional[float] = None,
                 membrane_deformation: Optional[Any] = None) -> Dict[str, Any]:
    """The integration block of one axon's row (``integration_columns()``: v13, plus v14's ``REPORT_COLUMNS``), for
    the analysis exported.

    ``nena_nm`` / ``nena_source``: the file's NeNA (None: the working scale says "paper" and the events, the
    footprint and the rule stay empty). ``locs_per_cycle``: a ``tools.mps_events.LocsPerCycle`` measured on the
    whole file (B3), or None (the rule's labelled fallback). ``null``: this analysis' :class:`NullOutcome` when the
    user ran one (else the null columns stay empty with ``null_type`` "none"). ``comparison``: the last comparison of
    this axon, for the methods block. ``nena_method``: how ``nena_nm`` was estimated (default: the program's NeNA
    estimator when a NeNA is given, ``NENA_METHOD_MEASURED``)."""
    from tools.dbscan_rule import C_LABEL_BME
    from tools.mps_events import LINK_RADIUS_IN_NENA, MAX_DARK_FRAMES
    from tools.mps_working_scale import working_scale_of_analysis
    register_all_plugins()
    cols = integration_columns()
    row: Dict[str, Any] = {c: None for c in cols}
    nena = _num(nena_nm)
    # --- the working scale (A9 sec. 1.1, A5 inside it)
    ws = working_scale_of_analysis(analysis, nena_nm=nena)
    row.update(ws.export_columns())
    row["scale_nena_source"] = (nena_source or "measured") if ws.nena_source == "measured" else "paper (no NeNA)"
    # --- the methods block (A8)
    rule = rule_for(nena, c_label=c_label, locs_per_cycle=locs_per_cycle)
    rule26 = rule_for(nena, c_label=C_LABEL_BME, locs_per_cycle=locs_per_cycle)
    row["dbscan_mode"] = dbscan_mode(analysis, rule)
    # the c_label every analysis of this row used (the rule's, the plug-ins', the null's), said even without a NeNA
    # (no rule then): the Batch used 14 whatever the Working scale panel said before 2026-10-29
    from tools.dbscan_rule import C_LABEL_DEFAULT as _C14
    c_used = float(_C14 if c_label is None else c_label)
    row["c_label_used"] = c_used
    row["c_label_source"] = c_label_origin(c_used)
    if rule is not None:
        row.update({k: v for k, v in rule.as_export().items() if k in row and k != "dbscan_mode"})
        row["min_samples_rule_c26"] = None if rule26 is None else int(rule26.min_samples)
    if locs_per_cycle is not None:
        row["loc_per_cycle_n_events"] = int(getattr(locs_per_cycle, "n_events", 0) or 0)
    ev_k = float(LINK_RADIUS_IN_NENA if events_radius_in_nena is None else events_radius_in_nena)
    row["link_radius_nena"] = ev_k
    row["link_max_dark_frames"] = int(MAX_DARK_FRAMES)
    # --- the reporting checklist's columns (v14)
    row["nena_method"] = (nena_method or NENA_METHOD_MEASURED) if nena is not None else NENA_METHOD_NONE
    row["link_radius_nm"] = None if nena is None else round(ev_k * nena, 1)
    frac = arc_fraction_of(analysis)
    row["occupancy_arc_fraction_within_25nm"] = None if frac is None else round(frac, 4)
    # manual test of 2026-10-23: the three columns were empty, whether 'Compare methods...' had run or not
    if comparison is None:
        for name in ("methods_compared", "methods_long_file", "methods_depends_on_method"):
            row[name] = METHODS_NOT_RUN
    else:
        row["methods_compared"] = "; ".join(comparison.labels.get(r.key, r.key) for r in comparison.runs)
        row["methods_depends_on_method"] = "; ".join(comparison.depends_on_method()) or "none"
        long_file = str((comparison_files or {}).get("long", "") or "")
        row["methods_long_file"] = os.path.basename(long_file) if long_file else METHODS_TABLES_NOT_WRITTEN
    # --- events, eclipse, size of THIS analysis
    run = run_of(analysis, axon_id="exported")
    extras = comparison_extras(nena_nm=nena, c_label=c_label, analysis=analysis,
                               pipeline_null_results=({run.key: null.result} if null is not None and null.ok
                                                      else None),
                               events_radius_in_nena=events_radius_in_nena)
    long_rows = mm.apply_analyses([run], list(PLUGIN_BLOCKS), seed=int(seed), extras=extras)
    for r in long_rows:
        if r["readout"] in row:
            row[r["readout"]] = r["value"]
    # --- regularity (needs the null for its p and reading; the values alone without it)
    if null is not None and null.gap is not None:
        row.update({k: v for k, v in null.gap.export_columns().items() if k in row})
    else:
        try:
            from tools import mps_gap_measures as gm
            if getattr(analysis, "perimeter", None) is not None and int(analysis.n_clusters_kept) >= 3:
                s, length, _m = gm.axon_arcs(analysis)
                if s is not None:
                    g = gm.gap_measures(s, length)
                    row.update({"gap_cv": g.cv, "gap_rho1": g.rho1, "gap_excess_cv": g.excess_cv,
                                "bragg_s": g.bragg_s, "bragg_m": g.bragg_m, "bragg_spacing_nm": g.bragg_spacing_nm,
                                "s_low": g.s_low})
        except Exception:  # noqa: BLE001 - descriptive values only; the row stands without them
            pass
    row["regularity_edge_note"] = edge_note(analysis) or None
    # --- B3 (c): localizations per cycle by depth (a diagnostic; the rule above keeps the whole file's value)
    from tools.lpc_depth import export_columns as _lpc_columns
    row.update(_lpc_columns(lpc_profile, lpc_profile_note or LPC_DEPTH_NOT_HERE))
    # --- the null of this analysis, when it ran
    if null is not None and null.ok:
        d = null.result.export_dict()
        row.update({k: v for k, v in d.items() if k in row and not k.startswith("scale_")})
    else:
        row["null_type"] = "none"
    # --- v16: B1's radii and C2's membrane
    row["lpc_rule_radius_nena"] = round(float(LPC_RULE_RADIUS_IN_NENA if lpc_radius_in_nena is None
                                              else lpc_radius_in_nena), 4)
    row["radii_exploratory"] = radii_exploratory(lpc_radius_in_nena, events_radius_in_nena)
    if null is not None and null.ok:
        row["null_membrane"] = null.membrane or None
    if membrane_deformation is not None:
        row["membrane_deformation_mean_nm"] = round(float(membrane_deformation.mean_nm), 1)
        row["membrane_deformation_max_nm"] = round(float(membrane_deformation.max_nm), 1)
        row["membrane_deformation_n_flagged"] = int(membrane_deformation.n_flagged)
    return {c: row.get(c) for c in cols}


def describe_column(name: str) -> Optional[str]:
    """What a column of the block (v13 / v14) means (for ``tools.column_dictionary.describe``), or None."""
    if name in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[name]
    if name in SIZE_STATS_DESCRIPTIONS:
        return SIZE_STATS_DESCRIPTIONS[name]
    from tools.lpc_depth import COLUMN_DESCRIPTIONS as _LPC_WORDS
    if name in _LPC_WORDS:
        return _LPC_WORDS[name]
    if name.startswith("p_mc_") and name.endswith(IGNORING_FAILED):
        what = NULL_READOUT_WORDS.get(name[len("p_mc_"):-len(IGNORING_FAILED)])
        if what is not None:
            return (f"Monte Carlo p of {what} against the pipeline null over the ANALYSED replicates only, the failed "
                    "ones left out (G1 c): descriptive; the p to read is the p_mc_ column without this suffix, which "
                    "counts every failed replicate as at least as extreme as the observed.")
    if name.startswith("p_mc_") or name.startswith("null_median_"):
        base = name[len("p_mc_"):] if name.startswith("p_mc_") else name[len("null_median_"):]
        what = NULL_READOUT_WORDS.get(base)
        if what is None:
            return None
        return (f"Two-sided Monte Carlo p of {what} against the per-axon pipeline null (same method, curation and "
                "contour; Besag & Diggle 1977); empty without a null or under the K closure for K per um."
                if name.startswith("p_mc_") else f"Median over the pipeline null's simulated axons of {what}.")
    register_all_plugins()
    return mm.describe_column(name)


IGNORING_FAILED = "_ignoring_failed"
NULL_READOUT_WORDS: Dict[str, str] = {
    "k_per_um": "the clusters per um", "nn_median": "the median 1NN", "reff_median": "the median r_eff",
    "occupancy": "the occupancy", "regularity": "the regularity index r",
    "size_nloc_nn_spearman": "the size-neighbour Spearman (apparent-size question a)",
    "size_nloc_kappa_r_spearman": "the size-curvature Spearman (apparent-size question b, localizations, convex "
                                  "clusters)",
    "gap_cv": "the gap CV", "gap_rho1": "the lag-1 gap correlation", "gap_excess_cv": "the excess-gap CV",
    "bragg_s": "the Bragg peak S*", "elongation_p95": "the 95th percentile of the cluster elongation",
}

COLUMN_DESCRIPTIONS: Dict[str, str] = {
    # working scale (docs/null_and_scale.md sec. 2)
    "scale_eps_nm": "The eps the working scale is computed with: DBSCAN's own, or the reference 25 nm for other "
                    "methods (the same tested radii for every method).",
    "scale_min_samples": "DBSCAN's min samples of this analysis (empty for methods without one).",
    "scale_nena_nm": "The NeNA lateral precision the marks use: this file's, or the paper's 12 nm when none was "
                     "measured (scale_nena_source says which).",
    "scale_nena_source": "Where scale_nena_nm comes from: the file measured (and on which file), or the paper.",
    "scale_fusion_nm": "Near-total fusion: below this distance two molecules come out as one cluster (d50 - 20 nm; "
                       "own simulation).",
    "scale_d50_nm": "d50 = 2 eps + 2.5 NeNA: half of the pairs of molecules resolved (own simulation, an empirical "
                    "fit); a 1NN below it is an upper bound.",
    "scale_slab_nm": "Half-width of the axial slab (Gazal et al. 2026: 90 nm).",
    # methods block (A8; docs/dbscan_rule.md)
    "dbscan_mode": "How DBSCAN's parameters were chosen: compatibility 25/10 (Gazal et al. 2026), the rule of "
                   "docs/dbscan_rule.md, typed, or not DBSCAN.",
    "eps_rule_nm": "The rule's eps for this file: min(2.5 NeNA, 25 nm) (own simulation; Gazal et al. 2026 as cap).",
    "min_samples_rule": "The rule's min samples with c_label_used (counting the point itself).",
    "min_samples_rule_c26": "The rule's min samples with c_label 26 (Alexa Fluor 647 in beta-ME; sensitivity).",
    "c_label_used": "Switching cycles per label the rule, the fusion mark, the footprint and the null used "
                    "(14: Dempsey et al. 2011, MEA, provisional; editable in 'Working scale').",
    "c_label_source": "Where c_label_used comes from (literature preset, typed, calibrated).",
    "loc_per_cycle_used": "Localizations per switching cycle the rule used: measured on the whole file by event "
                          "linking at 3 sqrt(2) NeNA, or the labelled fallback 2.",
    "loc_per_cycle_source": "Where loc_per_cycle_used comes from (measured, the file's own columns, fallback).",
    "loc_per_cycle_n_events": "Events the localizations per cycle were measured on.",
    "p_neighbour": "Probability that two localizations of one label fall within eps (the rule's P).",
    "nena_nm_used": "The NeNA the rule used.",
    "cycle_quantile": "Quantile of the cycles per label the rule protects (own simulation).",
    "link_radius_nena": "Event linking radius of the event layer and the fusion mark, in NeNA (decision D5).",
    "link_max_dark_frames": "Dark frames bridged inside one event (Thevathasan et al. 2019).",
    "methods_compared": "The runs of the last 'Compare methods...' of this axon (DBSCAN 25/10 first); 'not run' "
                        "without a comparison.",
    "methods_long_file": "The long comparison table written for this axon (one row per method x readout); 'not "
                         "written' when the comparison ran but 'Export tables...' did not, 'not run' without one.",
    "methods_depends_on_method": "Readouts whose verdict changed with the method in that comparison ('none' when "
                                 "no verdict changed, 'not run' without a comparison).",
    # the reporting checklist (v14; docs/reporting_checklist.md)
    "nena_method": "How the NeNA of the marks and the rule was estimated (Endesfelder et al. 2014; maximum "
                   "likelihood on Picasso's model, docs/nena.md), or that none was measured.",
    "link_radius_nm": "Event linking radius of the event layer in nm: link_radius_nena x the file's NeNA (empty "
                      "without a NeNA).",
    "occupancy_arc_fraction_within_25nm": "P-R103: fraction (0-1) of the contour through the kept clusters' centres "
                                          "(the curve of perimeter_um and occupancy_percent) lying within 25 nm "
                                          "(Euclidean) of a kept cluster's centre; exact, not sampled. With centres "
                                          "on the contour more than 50 nm apart it is 50 K / P. Not the "
                                          "constrained-Gaussian occupancy, which is unchanged.",
    # regularity (docs/regularity.md)
    "gap_cv": "Gap CV: sd / mean of the arc gaps between consecutive centres on the 600 nm membrane.",
    "gap_cv_null_mean": "Mean gap CV of the pipeline null (stage 1).",
    "gap_cv_null_lo": "2.5 % of the pipeline null's gap CV.",
    "gap_cv_null_hi": "97.5 % of the pipeline null's gap CV.",
    "gap_cv_p_mc": "Two-sided Monte Carlo p of the gap CV against the pipeline null (stage 1 of the reading).",
    "gap_cv_z": "Gap CV in standard deviations of its pipeline null (what the batch averages per animal).",
    "gap_rho1": "Lag-1 correlation of consecutive gaps (-1/2 for a lattice, 0 for any renewal process).",
    "gap_rho1_null_mean": "Mean rho1 of the pipeline null.", "gap_rho1_null_lo": "2.5 % of the null's rho1.",
    "gap_rho1_null_hi": "97.5 % of the null's rho1.", "gap_rho1_z": "rho1 in SD of the pipeline null.",
    "gap_excess_cv": "CV of the gaps in excess of the smallest (~1 exclusion, 0.4-0.6 bell-shaped gaps; descriptive).",
    "gap_excess_cv_null_mean": "Mean excess-gap CV of the null.", "gap_excess_cv_null_lo": "2.5 % of it.",
    "gap_excess_cv_null_hi": "97.5 % of it.", "gap_excess_cv_p_mc": "Two-sided Monte Carlo p of the excess-gap CV.",
    "gap_excess_cv_z": "Excess-gap CV in SD of the null.",
    "bragg_s": "Bragg peak S*: the largest periodogram value for m in [K/2, 3K/2] (Bartlett 1963).",
    "bragg_s_null_mean": "Mean S* of the pipeline null.", "bragg_s_null_lo": "2.5 % of the null's S*.",
    "bragg_s_null_hi": "97.5 % of the null's S*.", "bragg_s_z": "S* in SD of the pipeline null.",
    "gap_rho1_p_mc_matched": "Lower-tail p of rho1 against the matched-exclusion null (stage 2).",
    "gap_rho1_z_matched": "rho1 in SD of the matched-exclusion null.",
    "bragg_p_mc_matched": "Upper-tail p of S* against the matched-exclusion null (stage 2).",
    "bragg_s_z_matched": "S* in SD of the matched-exclusion null.",
    "bragg_m": "The mode m* of the Bragg peak.", "bragg_spacing_nm": "a* = P / m*, the spacing the peak implies.",
    "s_low": "Mean S(m) over m = 1..3 (the renewal plateau; 0 for a lattice).",
    "gap_cv_polygon": "Gap CV on the contour polygon (descriptive, no p).",
    "gap_rho1_polygon": "rho1 on the contour polygon (descriptive).",
    "gap_excess_cv_polygon": "Excess-gap CV on the contour polygon (descriptive).",
    "bragg_s_polygon": "S* on the contour polygon (descriptive).",
    "regularity_reading": "The two-stage reading of this axon: random_with_contact, less_regular_than_random, "
                          "more_regular_kind_not_identified, periodic_component, lattice_rho1_only, lattice or "
                          "not_testable (docs/regularity.md).",
    "regularity_flag_min_below_contact": "The smallest 1NN is below the contact distance d_c: split clusters or "
                                         "clusters of another ring.",
    "matched_null_kind": "Where stage 2 ran: pipeline (exact), centres_approx, or not_run.",
    "matched_null_d_nm": "The matched hard core d_m whose mean gap CV equals the observed.",
    "matched_null_B": "Replicates of the matched null.", "matched_null_capped": "d_m reached its cap.",
    "pipeline_null_B": "Replicates of the stage-1 pipeline null.",
    "pipeline_null_K_sim": "Molecules per ring the null simulated.",
    "pipeline_null_d_contact_nm": "Contact hard core d_c of the null (2 x median r_eff, read).",
    "pipeline_null_epitope_R_nm": "Epitope disc radius of the null.",
    "pipeline_null_lp_nm": "Lateral precision the null simulated.",
    "pipeline_null_K_det_mean": "Mean clusters the pipeline detected on the null's axons.",
    "pipeline_null_closure_ratio": "Null median / observed of the closed quantity.",
    "pipeline_null_seed": "Seed of the null.",
    "n_clusters_elongated": "Clusters more elongated along the membrane than the null's 95th percentile (merged "
                            "pairs; a mark, never split).",
    "elongation_threshold": "That 95th percentile.",
    "n_clusters_elongated_null_expected": "How many such clusters the null itself gives.",
    "nn_regularity_null_kind": "Which null the regularity p comes from (pipeline).",
    "regularity_edge_note": "Clusters removed at the ROI edge, which may leave contour stretches empty (G5).",
    # the pipeline null (docs/null_and_scale.md, A9 sec. 5)
    "null_type": "Which per-axon null: molecules+blinking (provisional: c_label from the literature), or none.",
    "null_method": "The clustering method the null re-ran (the analysis' own).",
    "null_closure": "What the null was matched to: n_locs (slab localizations, default) or k_clusters.",
    "null_B": "Simulated axons analysed.", "null_seed": "Seed of the null.",
    "null_c_label": "Switching cycles per label the null simulated.",
    "null_locs_per_cycle": "Localizations per cycle the null simulated.",
    "null_locs_per_cycle_source": "measured (event linking on the slab) or default.",
    "null_hard_core_nm": "Contact hard core of the null's molecules (2 x median r_eff, read).",
    "null_interpretable": "False when the null cannot be read (fewer than 19 replicates, more than 5 % of the "
                          "replicates failed, closure or fidelity failed); the numbers are kept.",
    "null_global_test_from": "The working-scale mark the global H1 / H2 test starts at (d50, provisional C1).",
    "null_global_r_min_nm": "That mark in nm.",
    "ripley_points": "What Ripley's functions are computed on (the cluster centres).",
    "ripley1d_window": "Window of H1 on the arc (closed ring, or none).",
    "ripley1d_rstar_nm": "r of the largest studentised deviation of H1 from the null, above the global-test mark.",
    "ripley1d_hmax": "H1 observed - null median at r* (signed).",
    "ripley1d_p_global": "Global (MAD) Monte Carlo p of H1: the decision; the pointwise envelope is only drawn.",
    "ripley2d_window": "Window of H2 (the +/-50 nm band, or the ROI without a ring).",
    "ripley2d_rstar_nm": "r of the largest studentised deviation of H2.",
    "ripley2d_hmax": "H2 observed - null median at r*.",
    "ripley2d_p_global": "Global (MAD) Monte Carlo p of H2.",
    # v15: failed replicates of the null (G1 c; docs/null_and_scale.md sec. 6.4)
    "null_failed_replicates": "Simulated axons of the pipeline null that failed (jammed sampler, pipeline error). Up "
                              "to floor(0.05 B) are tolerated (2 of 49, 9 of 199) and counted in every p as at least "
                              "as extreme as the observed (the p is conservative); more makes the null NOT "
                              "INTERPRETABLE (user decision G1 c).",
    "ripley1d_p_global_ignoring_failed": "Global (MAD) p of H1 over the analysed replicates only (descriptive; "
                                         "ripley1d_p_global counts the failed ones in the worst case).",
    "ripley2d_p_global_ignoring_failed": "Global (MAD) p of H2 over the analysed replicates only (descriptive).",
    "gap_cv_p_mc_ignoring_failed": "gap_cv_p_mc over the analysed replicates only (descriptive; G1 c).",
    "gap_excess_cv_p_mc_ignoring_failed": "gap_excess_cv_p_mc over the analysed replicates only (descriptive).",
    "gap_rho1_p_mc_matched_ignoring_failed": "gap_rho1_p_mc_matched over the analysed replicates of the matched null "
                                             "only (descriptive).",
    "bragg_p_mc_matched_ignoring_failed": "bragg_p_mc_matched over the analysed replicates of the matched null only "
                                          "(descriptive).",
    "null_status": "How the null of that closure ended: ok, refused (slow method, K < 8, External labels), error or "
                   "cancelled (the reason in null_type).",
    "regularity_stage2_level": "Where stage 2 of the regularity reading was set to run: 'centres' (the axon "
                               "window's preview, approximate: the matched null on the detected centres) or "
                               "'pipeline' (the Batch, exact: the matched null through the whole pipeline); P-R87 c. "
                               "The two can label one axon differently (4 of 60 synthetic axons); compare rows only "
                               "at the same level. Empty without a null.",
    "matched_null_failed_replicates": "Simulated axons of the stage-2 matched null that failed (same G1 rule as the "
                                      "stage-1 null); empty when stage 2 did not run at the pipeline level.",
    # v16 (2026-11-07)
    "lpc_rule_radius_nena": "Radius, in NeNA, of the event linking that measured the localizations per cycle the "
                            "DBSCAN rule used (loc_per_cycle_used): 3 sqrt(2) = 4.243 by default [simulation: "
                            "docs/dbscan_rule.md sec. 4; user decision B1]; another value was typed in 'Working "
                            "scale' (radii_exploratory says so). The events layer's radius is link_radius_nena.",
    "radii_exploratory": "B1: 'no' when both linking radii are the defaults (rule's lpc 3 sqrt(2) NeNA, events and "
                         "fusion mark 3 NeNA); else 'EXPLORATORY: ...' naming the radius typed in 'Working scale'. "
                         "Rows marked EXPLORATORY are not the pre-specified analysis.",
    "null_membrane": "C2: the membrane the per-axon null put its molecules on: the centroids of DBSCAN 25/10 "
                     "(default, common to every method) or the localizations of the slab (option). Empty without a "
                     "null.",
    "membrane_deformation_mean_nm": "C2: mean distance (nm) from the DBSCAN 25/10 centroid membrane to the membrane "
                                    "fitted to the slab's localizations, sampled every 10 nm along it. Empty when "
                                    "the indicator was not computed (it is computed in the axon window's "
                                    "'Perimeter arrangement' tab).",
    "membrane_deformation_max_nm": "C2: largest distance (nm) between the two membranes.",
    "membrane_deformation_n_flagged": "C2: stretches of 600 nm [derivation: the knot spacing] along the centroid "
                                      "membrane where the two membranes are more than 50 nm apart [ad hoc: the H2 "
                                      "band's half-width]: there the centroids deform the membrane the "
                                      "localizations show.",
}


# =====================================================================================================================
# The batch (regularity PR 4)
# =====================================================================================================================

NO_ANIMAL_WORDS = "no animal assigned: no per-animal table; set the animal in the batch table"


def regularity_batch_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Regularity over a batch's rows (the axon table's columns plus ``animal`` / ``genotype``): per animal z with
    Holm across measures, readings per group, the P-R83 lattice criterion per group, and the calibre model (N ~ P,
    random intercept per animal; refused below 3 animals). The type of a group's regularity is read from the
    per-animal z (G4), never from the per-axon labels. Returns {"by_animal": rows, "readings": rows,
    "lattice": {group: sentence}, "calibre": sentence, "lines": [...]}."""
    from tools import mps_gap_batch as gb
    from tools import mps_gap_measures as gm
    use = [dict(r) for r in rows if r.get("regularity_reading") not in (None, "")]
    for r in use:
        r.setdefault("group", r.get("genotype") or "")
    lines: List[str] = []
    out: Dict[str, Any] = {"by_animal": [], "readings": [], "lattice": {}, "calibre": "", "lines": lines}
    if not use:
        lines.append("Regularity: no axon of this batch has a regularity reading (run the batch with 'Null per "
                     "axon').")
        return out
    # manual test of 2026-10-23: with the animal column empty no per-animal table was written, nothing said why,
    # and the text counted the blank as "1 animal(s)". A blank animal is no animal.
    with_animal = [r for r in use if str(r.get("animal") or "").strip()]
    animals = {str(r.get("animal")).strip() for r in with_animal}
    out["by_animal"] = gb.aggregate_by_animal(with_animal) if with_animal else []
    out["readings"] = gb.readings_table(use, gm.READING_LEVELS)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in use:
        groups.setdefault(str(r.get("group") or "(no genotype)"), []).append(r)
    lines.append(f"Regularity over {len(use)} axon(s), {len(animals)} animal(s): the type of a group's regularity "
                 "is read from the per-animal z (P-R83), not from each axon's label; " + LATTICE_WORDS + " (G4).")
    if not with_animal:
        lines.append(f"  {NO_ANIMAL_WORDS}.")
    elif len(with_animal) < len(use):
        lines.append(f"  {len(use) - len(with_animal)} axon(s) with no animal assigned are left out of the "
                     "per-animal table; set the animal in the batch table.")
    for g, rr in sorted(groups.items()):
        ev = gb.lattice_evidence(rr)
        out["lattice"][g] = ev.sentence
        lines.append(f"  {g}: {ev.sentence}")
    try:
        n = [float(_num(r.get("n_clusters_kept")) or 0.0) for r in use]
        p = [float(_num(r.get("perimeter_um")) or 0.0) for r in use]
        a = [str(r.get("animal") or "") for r in use]
        k_det = [_num(r.get("pipeline_null_K_det_mean")) for r in use]
        cal = gb.calibre_model(n, p, a, null_k_detected=k_det)
        out["calibre"] = cal.sentence
        lines.append(f"  Calibre: {cal.sentence}")
    except gb.CalibreRefused as exc:
        out["calibre"] = str(exc)
        lines.append(f"  Calibre model not fitted: {exc}")
    except (ValueError, np.linalg.LinAlgError) as exc:
        out["calibre"] = f"not fitted: {exc}"
        lines.append(f"  Calibre model not fitted: {exc}")
    return out


def write_regularity_summary(stem: str, summary: Mapping[str, Any]) -> Dict[str, str]:
    """``<stem>_regularity_by_animal.csv``, ``<stem>_regularity_readings.csv`` and ``<stem>_regularity.txt`` (the
    lines); never overwrites."""
    from tools.results_table import free_path
    base = os.path.splitext(stem)[0]
    out: Dict[str, str] = {}
    for key, suffix in (("by_animal", "_regularity_by_animal.csv"), ("readings", "_regularity_readings.csv")):
        rows = list(summary.get(key) or [])
        if not rows:
            continue
        cols: List[str] = []
        for r in rows:
            cols += [c for c in r if c not in cols]
        out[key] = mm.write_rows_csv(base + suffix, rows, cols)
    target = free_path(base + "_regularity.txt")
    with open(target, "x", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(summary.get("lines") or []) + "\n")
    out["lines"] = target
    return out


# =====================================================================================================================
# The batch: the size readouts across axons and animals (tools.mps_size_stats, decisions 5a / 6a / 7)
# =====================================================================================================================

# The per-axon size readouts a group comparison is run on, with their estimator kind (tools.mps_size_stats
# .COVARIATES_BY_KIND, decision 6a: the apparent estimators with the file's NeNA and log N_loc, the pool and the
# content with none).
SIZE_GROUP_READOUTS: Tuple[Tuple[str, str], ...] = (
    ("size_rho_pool_nm", "pool"), ("size_cloud_sd_median_nm", "apparent"), ("size_r_rare_median_nm", "apparent"),
    ("size_radius_gyration_median_nm", "apparent"), ("size_n_events_median", "content"),
    ("size_n_locs_median", "content"))
# The per-axon Fisher z combined across axons (decision 5a: the axon is the unit; the animal as second level).
SIZE_COMBINED_Z: Tuple[Tuple[str, str], ...] = (
    ("size_nloc_nn_fisher_z", "question (a): localizations per cluster vs the nearest cluster"),
    ("size_content_kappa_r_fisher_z", "question (b): content vs local curvature (convex clusters)"))
SIZE_STATS_COLUMNS: Tuple[str, ...] = (
    "size_stat", "size_readout", "estimator_kind", "group", "group_a", "group_b", "covariates", "n_axons",
    "n_animals", "animal_diff", "animal_t", "animal_p", "mixed_diff", "mixed_p", "mixed_df", "axon_mean_z", "axon_t",
    "axon_p", "axon_n", "animal_mean_z", "animal_t_z", "animal_p_z", "animal_n", "size_stat_warnings")
SIZE_STATS_DESCRIPTIONS: Dict[str, str] = {
    "size_stat": "Which combination: 'group comparison' (two genotypes, animals nested; tools.mps_size_stats"
                 ".compare_groups) or 'combined z' (per-axon Fisher z combined over axons and over animals; "
                 "combine_axons_fisher).",
    "size_readout": "The per-axon column of the axon table the row combines (size_*).",
    "estimator_kind": "content, apparent or pool: sets the covariates (decision 6a: apparent estimators with the "
                      "file's NeNA and log N_loc; content and pool with none).",
    "group": "For a combined z: the genotype the axons belong to ('all' = every axon of the batch).",
    "group_a": "First genotype of the comparison (alphabetical).",
    "group_b": "Second genotype; differences are group_b - group_a.",
    "covariates": "Covariates in the mixed model (decision 6a).",
    "n_animals": "Animals with a value (the unit of the group comparison; Lazic 2010 asks for at least 6).",
    "animal_diff": "Difference of the animal means, group_b - group_a, in the readout's unit.",
    "animal_t": "Welch t on the animal means (the animal as unit).",
    "animal_p": "Two-sided p of animal_t (Welch df).",
    "mixed_diff": "Group coefficient of the mixed model with a random intercept per animal and the covariates "
                  "(REML; Aarts et al. 2014).",
    "mixed_p": "Two-sided p of mixed_diff (t on the between-animal degrees of freedom, conservative).",
    "mixed_df": "Degrees of freedom of mixed_p.",
    "axon_mean_z": "Mean per-axon Fisher z (the axon as unit, decision 5a).",
    "axon_t": "One-sample t of the per-axon z against 0. NOT against the pipeline null: for question (a) DBSCAN "
              "alone fabricates a positive Spearman (docs/apparent_size.md sec. 7); read with size_nloc_nn_null_p.",
    "axon_p": "Two-sided p of axon_t.",
    "axon_n": "Axons with a finite z.",
    "animal_mean_z": "Mean of the per-animal mean z (the animal as second level).",
    "animal_t_z": "One-sample t over the animals' mean z.",
    "animal_p_z": "Two-sided p of animal_t_z (with 3 animals: 2 df, about half the power; SIZE_SIM sec. 6.3).",
    "animal_n": "Animals with a finite mean z.",
    "size_stat_warnings": "Why a number is missing or weak (fewer than two animals in a group, fewer than 6 "
                          "animals, mixed model not fitted, not two genotypes).",
}


def size_batch_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The batch's size readouts across axons and animals (tools.mps_size_stats; HALLAZGOS: it existed and nothing
    called it): a group comparison per readout of ``SIZE_GROUP_READOUTS`` when the batch holds exactly two genotypes
    (animal as unit, and the mixed model with the covariates of decision 6a), and the per-axon Fisher z of questions
    (a) and (b) combined over axons and animals, for all axons and per genotype. Rows are the axon table's columns plus
    ``animal`` / ``genotype``; a blank animal is no animal. Returns {"rows": [...SIZE_STATS_COLUMNS], "lines": [...]}."""
    from tools import mps_size_stats as ss
    use = [dict(r) for r in rows if str(r.get("animal") or "").strip()]
    lines: List[str] = []
    out_rows: List[Dict[str, Any]] = []
    if not use:
        lines.append(f"Size across axons: {NO_ANIMAL_WORDS}.")
        return {"rows": out_rows, "lines": lines}
    genotypes = sorted({str(r.get("genotype") or "").strip() for r in use} - {""})
    animal = [str(r.get("animal")).strip() for r in use]
    group = [str(r.get("genotype") or "").strip() for r in use]
    lines.append(f"Size across {len(use)} axon(s), {len(set(animal))} animal(s), genotypes: "
                 f"{', '.join(genotypes) or 'none'} (tools.mps_size_stats; the animal is the unit).")
    for name, kind in SIZE_GROUP_READOUTS:
        row: Dict[str, Any] = {c: None for c in SIZE_STATS_COLUMNS}
        row.update(size_stat="group comparison", size_readout=name, estimator_kind=kind)
        if len(genotypes) != 2:
            row["size_stat_warnings"] = f"needs exactly two genotypes, the batch has {len(genotypes)}"
            out_rows.append(row)
            continue
        keep = [i for i, g in enumerate(group) if g in genotypes]
        vals = [_num(use[i].get(name)) for i in keep]
        nena = [_num(use[i].get("size_nena_nm")) for i in keep]
        nloc = [_num(use[i].get("size_n_locs_median")) for i in keep]
        lognloc = [None if v is None or v <= 0 else math.log(v) for v in nloc]
        row.update(group_a=genotypes[0], group_b=genotypes[1])
        if sum(v is not None for v in vals) < 2:
            row["size_stat_warnings"] = "fewer than two axons with a value"
            out_rows.append(row)
            continue
        try:
            gc = ss.compare_groups(vals, [group[i] for i in keep], [animal[i] for i in keep], estimator_kind=kind,
                                   nena_nm=nena, log_n_locs=lognloc)
        except ValueError as exc:
            row["size_stat_warnings"] = str(exc)
            out_rows.append(row)
            continue
        row.update(covariates=", ".join(gc.covariates) or "none", n_axons=gc.n_axons, n_animals=gc.n_animals,
                   animal_diff=gc.animal_diff, animal_t=gc.animal_t, animal_p=gc.animal_p,
                   mixed_diff=gc.mixed_diff, mixed_p=gc.mixed_p, mixed_df=gc.mixed_df,
                   size_stat_warnings="; ".join(gc.warnings) or None)
        out_rows.append(row)
        if gc.animal_p is not None:
            lines.append(f"  {name}: {genotypes[1]} - {genotypes[0]} = {gc.animal_diff:.3g} (animals, p = "
                         f"{gc.animal_p:.3g}; {gc.n_animals} animals)")
    for name, words in SIZE_COMBINED_Z:
        for g in ["all"] + genotypes:
            idx = [i for i in range(len(use)) if g == "all" or group[i] == g]
            row = {c: None for c in SIZE_STATS_COLUMNS}
            row.update(size_stat="combined z", size_readout=name, estimator_kind="z", group=g)
            z = [_num(use[i].get(name)) for i in idx]
            if not any(v is not None for v in z):
                row["size_stat_warnings"] = f"no axon with {name} ({words})"
                out_rows.append(row)
                continue
            comb = ss.combine_axons_fisher(z, [animal[i] for i in idx])
            row.update(axon_mean_z=comb.get("axon_mean"), axon_t=comb.get("axon_t"), axon_p=comb.get("axon_p"),
                       axon_n=comb.get("axon_n"), animal_mean_z=comb.get("animal_mean"),
                       animal_t_z=comb.get("animal_t"), animal_p_z=comb.get("animal_p"),
                       animal_n=comb.get("animal_n"))
            warn = ["against 0, not against the pipeline null (read size_nloc_nn_null_p per axon)"
                    if name == "size_nloc_nn_fisher_z" else ""]
            row["size_stat_warnings"] = "; ".join(w for w in warn if w) or None
            out_rows.append(row)
    return {"rows": out_rows, "lines": lines}


def write_size_summary(stem: str, summary: Mapping[str, Any]) -> Dict[str, str]:
    """``<stem>_size_stats.csv`` (``SIZE_STATS_COLUMNS``) beside the axon table; never overwrites."""
    rows = list(summary.get("rows") or [])
    if not rows:
        return {}
    base = os.path.splitext(stem)[0]
    return {"size_stats": mm.write_rows_csv(base + "_size_stats.csv", rows, SIZE_STATS_COLUMNS)}


def words_estimate(seconds: float) -> str:
    """'about 12 min' (a first estimate, a lower bound for the slow methods)."""
    return f"about {words_seconds(seconds)}"


__all__ = [
    "Provisional", "PROVISIONAL", "provisional", "register_all_plugins", "AnalysisChoice", "ANALYSIS_CHOICES",
    "DEFAULT_ANALYSES", "estimate_analyses_seconds", "comparison_extras", "rule_for", "locs_per_cycle_of",
    "rule_method_specs", "rule_labels", "dbscan_mode", "run_of", "ComparisonResult", "run_comparison",
    "comparison_paths", "write_comparison", "NullOutcome", "null_estimate_seconds", "run_null_for", "run_nulls",
    "NULL_CLOSURE_CHOICES", "null_closures", "run_nulls_by_closure", "null_by_closure_columns", "null_closure_rows",
    "null_long_rows", "edge_note", "event_ids_for", "SCALE_COLUMNS", "METHODS_COLUMNS", "GAP_COLUMNS", "NULL_COLUMNS",
    "REPORT_COLUMNS", "NULL_FAILED_COLUMNS", "NULL_READOUT_NAMES_V15", "ALL_NULL_READOUT_NAMES", "E5_NULL_COLUMNS",
    "V15_PLUGIN_COLUMNS", "v15_columns", "arc_fraction_of", "NENA_METHOD_MEASURED", "NENA_METHOD_NONE",
    "integration_columns", "axon_columns", "describe_column", "regularity_batch_summary", "write_regularity_summary",
    "words_estimate", "is_slow", "SIZE_GROUP_READOUTS", "SIZE_COMBINED_Z", "SIZE_STATS_COLUMNS", "size_batch_summary",
    "write_size_summary",
]
