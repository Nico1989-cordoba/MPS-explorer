# -*- coding: utf-8 -*-
"""
What every column of the exported tables means, written beside them.

A table of 159 columns is only usable by whoever exported it, and only
while they remember. This builds the dictionary from the code that writes
the tables, so a column cannot be documented as something it is not, and a
new column cannot be quietly undocumented either: ``validate_axon_tables``
checks that every column an export produces has an entry here.

The text is deliberately plain. It says what the number is and what it is
in, not how it is computed -- the code and the plan say that.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tools.axon_export import (
    AXIAL_MIXTURE_COLUMNS, AXOPLASM_PREFIX, DISCARD_SUFFIX, EPS_CHOICE_COLUMNS,
    HEAD_COLUMNS,
    MEASURED_COLUMNS, SHARED_COLUMNS, STATE_COLUMNS, Z_CALIBRATION_COLUMNS,
)
from tools.mps_cut_clusters import CUT_COLUMNS
from tools.axon_export import CLUSTER_METHOD_COLUMNS, EMITTERS_COLUMNS
from tools.perimeter_export import (PERIMETER_COLUMNS, PERIMETER_V19_COLUMNS, PERIMETER_V20_COLUMNS,
                                    PERIMETER_V21_COLUMNS)
from tools.unanchored_export import UNANCHORED_COLUMNS
from tools.mps_identity import FIELD_HINTS, FIELDS as IDENTITY_FIELDS

# Units, read off the end of the name. The rule is the convention: a
# quantity says its unit in its own name, and nothing carries a unit it
# does not say.
UNITS = (
    ("_nm2", "nm^2"), ("_um2", "um^2"), ("_nm", "nm"), ("_um", "um"),
    ("_px", "pixels"), ("_percent", "%"), ("_deg", "degrees"),
)


def unit_of(name: str) -> str:
    explicit = perimeter_unit(name)        # v17: units the name does not give (%, fractions, 1/nm, null SD)
    if explicit is not None:
        return explicit
    if name in UNANCHORED_UNITS:            # v18
        return UNANCHORED_UNITS[name]
    base = name[:-len(DISCARD_SUFFIX)] if name.endswith(DISCARD_SUFFIX) \
        else name
    for suffix, unit in UNITS:
        if base.endswith(suffix):
            return unit
    if base.startswith("fraction_") or base.startswith("share_"):
        return "fraction (0-1)"
    return ""


# --------------------------------------------------------------------------
# What each column is
# --------------------------------------------------------------------------

IDENTITY: Dict[str, str] = {
    "axon_id": "Short name of this axon: the file it was picked from and "
               "the selection made on it. Joins the three tables.",
    "analysis_id": "Short name of this analysis: these localizations, every "
                   "parameter, and the version of the program.",
    "source": "The localization file this axon was picked from.",
    "roi": "The selection drawn on it, by shape, place and size.",
    "exported_at": "When the row was written (UTC).",
    "program_version": "Which version of the program wrote it.",
    "table_version": "Which layout of columns this table has.",
}
IDENTITY.update({name: FIELD_HINTS[name] for name in IDENTITY_FIELDS})

STATE: Dict[str, str] = {
    "discard_applied": "Whether the columns ending in _discard are filled "
                       "in, i.e. whether clusters were discarded.",
    "discard_margin_nm": "How far inside both widefield images a cluster "
                         "had to be to be discarded; with the automatic "
                         "discard of unanchored clusters (v18), its exclusion "
                         "depth (unanchored_exclude_nm).",
    "discard_registration": "How the widefield images were placed when "
                            "those clusters were found: measured with its "
                            "score, or set by hand.",
    "n_clusters_discarded": "How many kept clusters the discard left out.",
    "axoplasm_measured": "Whether the axoplasm panel measured this axon, "
                         "i.e. whether the axoplasm_ columns are filled in.",
}

ANALYSIS: Dict[str, str] = {
    # --- the file and the selection
    "pixel_size_nm": "Camera pixel, used to turn pixels into nanometres. "
                     "A wrong value rescales every distance here.",
    "pixel_size_source": "Where that pixel size came from: the Picasso "
                         "sidecar, the file itself, a file beside it "
                         "(neighbour), a value typed for this folder "
                         "earlier (remembered), or typed by hand now.",
    "eps_nm": "DBSCAN's neighbourhood radius.",
    "min_samples": "DBSCAN's minimum number of localizations for a cluster.",
    # --- which clustering method (axon tables v10, Phase D; written last)
    "cluster_method": "Which clustering method made the clusters: 'DBSCAN' "
                      "(the reference pipeline, eps_nm / min_samples) or one "
                      "of the experimental methods of "
                      "docs/clustering_methods.md.",
    "cluster_params": "The parameters that method ran with, as JSON.",
    # --- the Events -> Emitters layer (axon tables v11, BaGoL F5; written
    # last; empty for every method but Emitters). docs/emitters.md
    "emitters_method": "'Emitters (BaGoL-like reimplementation)' when the "
                       "localizations were grouped into emitters before the "
                       "clustering (Fazel et al. 2022, reimplemented; not "
                       "the authors' code); empty otherwise. Emitters are "
                       "fluorescent label stacks, not proteins.",
    "emitters_n": "Emitters of the model in the axial slab (K of the most "
                  "probable configuration, emitters without localizations "
                  "included).",
    "emitters_n_with_locs": "Emitters whose posterior mean of explained "
                            "events is at least has_locs_min (0.5 by "
                            "default).",
    "emitters_n_nonsingleton": "Emitters that explain at least 1.5 events "
                               "(posterior mean; rounded, two or more): the "
                               "emitters the per-um and per-cluster readouts "
                               "count.",
    "emitters_n_singletons": "Emitters with localizations that explain fewer "
                             "than 1.5 events (one event): mostly isolated "
                             "background, a legitimate one-event emitter of "
                             "the model, reported apart.",
    "emitters_per_um": "Distinct non-singleton emitters in the kept clusters "
                       "divided by the perimeter (the denominator of "
                       "clusters_per_um), per um.",
    "emitters_per_cluster_median": "Median number of distinct non-singleton "
                                   "emitters per kept cluster (clusters of "
                                   "singletons only left out): a readout, "
                                   "never a constant of emitters per molecule.",
    "emitters_per_cluster_q25": "First quartile of the non-singleton emitters "
                                "per kept cluster.",
    "emitters_per_cluster_q75": "Third quartile of the non-singleton emitters "
                                "per kept cluster.",
    "emitters_xi_mean": "Mean events (frame-linked localizations) per "
                        "emitter of the prior xi the model used (learned: k x "
                        "theta of the posterior means, learned on "
                        "non-singleton emitters).",
    "emitters_xi_source": "Where xi came from: literature (Dempsey et al. "
                          "2011, Table 1, the channel's dye in the buffer of "
                          "emitters_buffer), learned (from this axon, started "
                          "from that literature value) or calibrated (the "
                          "user's control). Tables before v12 wrote "
                          "literature_MEA / literature_bME.",
    "emitters_precision_nm_before": "Median lateral precision of the slab's "
                                    "localizations, (sigma_x + sigma_y) / 2, "
                                    "in nm, as the model received it.",
    "emitters_precision_nm_after": "Median precision of the emitters with "
                                   "localizations, (sigma_x + sigma_y) / 2 of "
                                   "their posterior position, in nm.",
    "emitters_r_mol_nm": "Emitters closer than this were linked into one "
                         "cluster (single linkage; 50 nm by default, a "
                         "derivation; 0 = no linking).",
    "emitters_n_events": "Events the model grouped: the slab's localizations "
                         "linked across frames as the DNA-PAINT panel links "
                         "them. Empty when the file had no frame numbers "
                         "(the model then ran on raw localizations).",
    "emitters_link_radius_nm": "The linking radius used for the events: 3 x "
                               "the median lateral precision of the slab "
                               "unless set by hand.",
    "emitters_max_dark_frames": "Frames that could be missing inside one "
                                "event when linking (1 by default).",
    # --- v12 (F5c): the buffer and check T4, last
    "emitters_buffer": "The imaging buffer chosen for this file, MEA (GLOX + "
                       "10 mM MEA) or bME (GLOX + 140 mM beta-mercaptoethanol), "
                       "the two of Dempsey et al. 2011 Table 1. No default: it "
                       "is chosen per file (per run in the batch).",
    "emitters_nena_over_sigma": "Check T4: NeNA of the slab / (median reported "
                                "lateral precision + se_adjust_nm). Empty when "
                                "NeNA was not measured (fewer than 500 "
                                "next-frame pairs, no frame numbers) or the file "
                                "had no per-localization precision.",
    "emitters_precision_flag": "ok, under-reported (emitters_nena_over_sigma "
                               "above 1.2: raise se_adjust_nm to NeNA - median "
                               "precision) or not measured.",
    "cluster_scale_nm": "The method's own length scale, informative: eps for "
                        "DBSCAN, the grid for FOCAL, the spacing of the dense "
                        "cells for Voronoi, the radius for ToMATo / "
                        "GraphCommunity, the median core distance for "
                        "HDBSCAN / OPTICS; empty when the method has none. "
                        "The edge-curation margin is eps for DBSCAN and, for "
                        "the other methods, the larger of eps and the "
                        "method's separation scale (not this column).",
    "cluster_methods_compared": "The methods compared in 'Compare methods' "
                                "before this one was chosen, separated by "
                                "'|'; empty when none were.",
    "dbcv_threshold": "Below this DBCV score a cluster is curated away. "
                      "-1 means the criterion is off.",
    "edge_reference": "What the edge criterion measured against: the drawn "
                      "ROI, or the convex hull when none was drawn.",
    # --- the axial slab
    "slab_half_width_nm": "Half-width of the axial slab that was analysed.",
    "slab_zmin_nm": "Lower edge of that slab.",
    "slab_zmax_nm": "Upper edge of that slab.",
    "slab_source": "Whether the slab is the automatic one, one centred on a "
                   "peak chosen by hand, or a range typed in.",
    "z_main_peak_auto_nm": "Where the automatic choice would have put the "
                           "centre of the slab.",
    "n_locs_total": "Localizations in the selection given to the analysis.",
    "n_locs_slab": "How many of them the axial slab holds.",
    "gmm_n_components": "Components of the Gaussian mixture fitted to the "
                        "depths.",
    "gmm_means_nm": "Where each component sits, separated by '|'.",
    "gmm_weights": "How much of the data each component holds.",
    "gmm_sigmas_nm": "How wide each component is.",
    "gmm_converged": "Whether that fit converged.",
    "gmm_n_discarded_components": "Components dropped as too small to be a "
                                  "ring.",
    "delta_z_mean_nm": "Mean spacing between consecutive axial components: "
                       "the periodicity along the axon.",
    "delta_z_values_nm": "Every one of those spacings, separated by '|'.",
    # --- the clustering
    "n_clusters_raw": "Clusters DBSCAN found, before any curation.",
    "n_clusters_removed": "Clusters the automatic curation removed.",
    "removed_edge_touching": "How many of those were touching the edge of "
                             "the selection.",
    "removed_low_dbcv": "How many of those scored below the DBCV threshold.",
    "edge_criterion_disabled": "Why the edge criterion did not run, when it "
                               "did not.",
    "mahalanobis_threshold": "How close to a cluster a point of the "
                             "perimeter counts as occupied. It decides the "
                             "occupancy outright.",
    "ellipse_mode": "How a cluster's ellipse is bounded before the occupancy "
                    "is measured.",
    "random_seed": "Seed of the randomization control, so it repeats.",
    "randomization_requested": "How many randomized rings were asked for; 0 "
                               "means the control was off.",
    "randomization_band_half_width_nm": "How far from the smoothed contour a "
                                        "randomized centre may sit (the "
                                        "band is twice this wide).",
    "randomization_smoothing_rms_nm": "How closely the smoothed contour "
                                      "follows the cluster centres: its RMS "
                                      "residual per centre. 1 nm passes "
                                      "through them.",
    "randomization_exclude_incomplete": "Whether randomized rings that could "
                                        "not place every cluster were drawn "
                                        "again and left out of the null.",
    "randomization_max_attempts_factor": "Candidates tried per cluster before "
                                         "a randomized ring gives up.",
    "randomization_grid_spacing_nm": "Spacing of the grid of positions a "
                                     "randomized centre is drawn from.",
    "randomization_hard_core": "How the minimum separation of the "
                               "randomized centres was set: the smallest "
                               "measured 1NN (the paper's rule) or its bias "
                               "correction.",
    "ks_band_alpha": "Level of the null band the CDF crossing is read "
                     "against.",
    "occupancy_sigma_cap_fraction": "Cap of each cluster's ellipse: its "
                                    "largest standard deviation is at most "
                                    "this fraction of the farthest "
                                    "localization's distance (1/3 in the "
                                    "paper).",
    # --- what the discard changes
    "n_clusters_kept": "Clusters this analysis measured.",
    "perimeter_um": "Length of the contour through the cluster centres.",
    "clusters_per_um": "Clusters per micrometre of that contour.",
    "contour_2opt": "Whether the tour was refined from one start or from "
                    "every start.",
    "contour_order_source": "Where the order of the tour through the centres came from: the program built it, or a person set it. An order set by hand leaves contour_2opt empty, because no 2-opt ran on it.",
    "contour_guide_nm": "The path drawn by hand along the membrane, when "
                        "the contour was ordered along one: x,y pairs in nm "
                        "separated by ';'. The centres are joined in the "
                        "order they fall along it, so the contour can be "
                        "rebuilt from this cell alone. Empty when the "
                        "program ordered the contour.",
    "contour_hull_um": "Perimeter of the convex hull of the same centres.",
    "contour_tour_over_hull": "How much longer the contour is than that "
                              "hull. Far above 1 means it wanders.",
    "contour_max_over_median": "Longest step of the contour over its median "
                               "step.",
    "contour_n_deep_vertices": "Centres sitting deep inside the hull rather "
                               "than on the outline.",
    "contour_hull_radius_nm": "Radius of that hull, which is what 'deep' is "
                              "measured against.",
    "contour_deep_limit_nm": "How deep a centre had to be to be counted.",
    "contour_scatter_nm": "How far this axon's centres scatter off a smooth "
                          "outline, estimated from the centres themselves. "
                          "Rough: from one ring it can be off by a factor "
                          "of two.",
    "contour_scatter_percent": "The same scatter as a percentage of the "
                               "hull radius: how rough this ring is for its "
                               "size. The deep-centre count is calibrated "
                               "for up to 5 %; real axons can measure "
                               "more.",
    "contour_scatter_depth_limit_nm": "How deep a healthy ring with that "
                                      "scatter and this many centres "
                                      "reaches, 99 times in 100. The deep "
                                      "count above is calibrated for 5 % "
                                      "scatter; this is the depth for this "
                                      "axon's own.",
    "contour_n_deep_beyond_scatter": "Deep centres that are deeper than a "
                                     "healthy ring with this axon's own "
                                     "scatter reaches, too: the ones scatter "
                                     "does not explain. Zero with a deep "
                                     "count above zero means the flagged "
                                     "centres may be scatter.",
    "contour_max_depth_nm": "The deepest of them.",
    "contour_length_in_long_edges": "Share of the contour spent on its "
                                    "longest steps.",
    "centre_x_nm": "Centre of the axon: the area centroid of the contour.",
    "centre_y_nm": "The same, in y.",
    "contour_area_um2": "Area the contour encloses.",
    "centre_max_shift_nm": "How far that centre moves when one cluster is "
                           "left out of it.",
    "median_area_nm2": "Median area of a cluster.",
    "median_r_eff_nm": "Median radius of the circle of that area.",
    "median_1nn_nm": "Median distance from a cluster to its nearest "
                     "neighbour. The paper's parameter 5.",
    "occupancy_percent": "Share of the perimeter within reach of a cluster.",
    "occupied_length_nm": "The same as a length.",
    "occupancy_n_points": "Points of the perimeter the occupancy was "
                          "measured at.",
    "occupancy_n_capped_ellipses": "Cluster ellipses that had to be bounded "
                                   "before measuring.",
    "ks_statistic": "How far the measured 1NN distribution is from the "
                    "randomized one (KS D). Descriptive: it shrinks with "
                    "the number of clusters, so compare ks_z between axons.",
    "ks_cdf_crossing": "The CDF level up to which the measured 1NN "
                       "distances are shorter than random, read only where "
                       "the two curves leave the band the randomized rings "
                       "give. Empty when no such crossing is seen. Until "
                       "axon tables v6 this column held a heuristic "
                       "crossing without a band.",
    "ks_cdf_crossing_significant": "Whether that crossing is stronger than "
                                   "random placements of this axon's "
                                   "clusters reach (Monte Carlo, at the "
                                   "band's level): true is a crossing this "
                                   "axon resolves.",
    "randomization_n_iterations": "Randomized rings in the null: the "
                                  "complete ones.",
    "randomization_min_sep_nm": "Closest two randomized centres were allowed "
                                "to be.",
    "randomization_incomplete_iters": "Randomized rings that could not place "
                                      "every cluster.",
    "randomized_median_1nn_nm": "Median 1NN of the randomized rings.",
    "ks_null_mean": "The D that randomized rings of this axon reach "
                    "against the pooled ones: D's own null mean for this "
                    "number of clusters.",
    "ks_z": "How many null standard deviations D lies above that mean. "
            "Comparable between axons, unlike D.",
    "ks_p_mc_median": "Monte Carlo p of the median 1NN against the "
                      "randomized rings' medians, two-sided. Approximate: "
                      "the minimum separation of the null comes from the "
                      "same data. Replaces the KS p-value, which these "
                      "samples do not support.",
    "ks_p_mc_d": "Monte Carlo p of D against the randomized rings' own D. "
                 "Approximate, as above.",
    "randomization_n_redrawn": "Randomized rings drawn again because the "
                               "first draw could not place every cluster.",
    "randomization_packing_fraction": "How crowded the minimum separation "
                                      "makes the band: K pi d^2 / (4 x band "
                                      "area). Near 0.5 or above, random "
                                      "placement jams.",
    "randomization_n_outside_band": "Measured cluster centres farther from "
                                    "the smoothed contour than the band "
                                    "reaches, where no randomized centre "
                                    "can sit.",
    "nn_regularity_index": "Median 1NN times the clusters over the "
                           "perimeter (median 1NN x K / P): 1 for evenly "
                           "spaced centres. Separates regularity from how "
                           "many clusters were detected. What random "
                           "placement gives depends on the minimum "
                           "separation and the band: compare it with this "
                           "axon's null r (the next three columns).",
    "nn_regularity_null_mean": "The same index of the randomized rings of "
                               "this axon (same K and P), averaged: the "
                               "random level for this axon.",
    "nn_regularity_null_lo": "Lower end of the central 95 % of the "
                             "randomized rings' index. Below it: an excess "
                             "of short distances.",
    "nn_regularity_null_hi": "Upper end of the same interval. Above it: "
                             "more regular than random.",
    "n_warnings": "How many warnings this analysis raised.",
    "warnings": "Their text, separated by ' | '.",
}

AXOPLASM: Dict[str, str] = {
    "tubulin_source": "What the axoplasm mask was drawn from: a widefield "
                      "betaIII-tubulin image placed on the localizations by "
                      "a measured shift, or tubulin localizations acquired "
                      "with the movie (dual view), placed by construction.",
    "spectrin_interior_source": "The same for the image the spectrin ring's "
                                "interior was found in.",
    "tubulin_image": "The widefield betaIII-tubulin image the axon's own "
                     "area was taken from.",
    "spectrin_interior_image": "The widefield betaII-spectrin image the "
                               "ring's interior was taken from.",
    "spectrin_interior_status": "Whether that interior could be found at "
                                "all.",
    "spectrin_interior_cut": "Brightness below which the image counts as "
                             "the inside of the ring.",
    "spectrin_interior_cut_source": "How that level was chosen.",
    "spectrin_level_interior": "Brightness inside the ring.",
    "spectrin_level_ring": "Brightness on the ring.",
    "spectrin_level_half_max": "Half way between the two.",
    "spectrin_level_spill": "Brightness just outside the ring, which is what "
                            "the interior must not be confused with.",
    "spectrin_interior_area_um2": "Area the interior of the ring covers.",
    "registration_reference_image": "Widefield image the shift was measured "
                                    "against.",
    "registration_localizations": "Localization file the shift was measured "
                                  "with.",
    "camera_offset_x_px": "Offset between the two cameras, in x.",
    "camera_offset_y_px": "The same, in y.",
    "registration_source": "Whether the shift was measured, set by hand, or "
                           "not placed at all -- or not needed ('same "
                           "acquisition'): every image was drawn from "
                           "localizations acquired with the movie.",
    "shift_x_nm": "How far the images were moved onto the localizations, "
                  "in x.",
    "shift_y_nm": "The same, in y.",
    "registration_peak": "Height of the correlation peak the shift was read "
                         "from.",
    "registration_zero_shift": "What that correlation is at no shift.",
    "registration_runner_up": "The next-best peak, which says whether the "
                              "best one stands out.",
    "registration_score": "Best peak over the runner-up: how sure the "
                          "placement is.",
    "threshold": "Brightness above which the tubulin image counts as axon.",
    "threshold_source": "How that brightness was chosen.",
    "otsu_threshold": "What Otsu's method would have chosen.",
    "smooth_sigma_nm": "How much the image was smoothed first.",
    "margin_nm": "How far inside a mask something has to be to count as "
                 "inside.",
    "mask_status": "Whether the axon's area could be found in the tubulin "
                   "image at all.",
    "mask_area_um2": "Area of the axon in the tubulin image.",
    "ring_area_um2": "Area of the spectrin ring itself.",
    "selection_zmin_nm": "Lower edge of the axial range the panel was given.",
    "selection_zmax_nm": "Upper edge of it.",
    "selection_z_source": "Whether that range was typed, came from the "
                          "axial peak, or was centred on the mixture "
                          "component chosen in the MPS analysis window "
                          "(peak chosen).",
    "n_localizations": "Localizations the panel was given.",
    "n_outside_tubulin_region": "How many fell outside the part of the image "
                                "that was read.",
    "n_localizations_inside": "Localizations inside the axon, through a "
                              "cluster both images put inside.",
    "n_localizations_membrane": "Localizations in a cluster at the membrane.",
    "n_localizations_no_cluster": "Localizations in no cluster at all.",
    "fraction_inside": "The inside ones over all of them.",
    "n_clusters": "Clusters the panel was given.",
    "n_clusters_inside_tubulin_only": "Clusters the tubulin image puts "
                                      "inside but the spectrin image does "
                                      "not.",
    "n_clusters_inside_spectrin_only": "The other way round.",
    "perimeter_all_clusters_um": "Contour through every kept cluster, as the "
                                 "panel drew it.",
    "perimeter_all_clusters_all_starts_um": "The same refined from every "
                                            "2-opt start.",
    "perimeter_anchored_um": "Contour through the clusters that were kept "
                             "after the discard.",
    "perimeter_anchored_2opt_starts": "How many starts that one was refined "
                                      "from.",
    "perimeter_anchored_start_spread_um": "Longest minus shortest of those "
                                          "tours: how much the start "
                                          "mattered.",
    "anchored_contour_deep_vertices": "Centres of that contour sitting deep "
                                      "inside its hull.",
    "n_warnings": "How many warnings the panel showed.",
    "warnings": "Their text, separated by ' | '.",
}

CLUSTERS: Dict[str, str] = {
    "cluster_label": "The cluster's DBSCAN label. The same number in every "
                     "table of this program.",
    "centroid_x_nm": "Its centre of mass, in x.",
    "centroid_y_nm": "The same, in y.",
    "n_localizations": "Localizations it holds.",
    "area_nm2": "Area of its convex hull.",
    "r_eff_nm": "Radius of the circle of that area.",
    "nn_1_nm": "Distance to its nearest neighbouring cluster.",
    "nn_1_label": "Which cluster that is.",
    "contour_position": "Where it sits along the contour, from 0. Without "
                        "this the perimeter cannot be recomputed from the "
                        "table.",
    "discarded": "Whether the discard left this cluster out.",
    "contour_position_discard": "Where it sits along the contour built "
                                "without the discarded clusters.",
    "nn_1_nm_discard": "Its nearest neighbour once those are gone.",
    "nn_1_label_discard": "Which cluster that is.",
    "cut_by_pick": "Whether the pick boundary cut this cluster: one of its "
                   "localizations lies within cut_margin_nm of the edge "
                   "(docs/cut_clusters.md). Empty when the rule was not "
                   "applied (see cut_status in the axon table).",
    # v18, package P14: the automatic discard of unanchored clusters
    "depth_below_ring_nm": "Signed distance of the cluster centre to the "
                           "reference ring of the automatic discard, "
                           "positive inside (the ring is in "
                           "unanchored_reference of the axon table). Empty "
                           "when the discard was not evaluated.",
    "anchor_state": "The automatic rule's verdict: 'unanchored' (deeper than "
                    "unanchored_exclude_nm, excluded), 'doubtful' (deeper "
                    "than unanchored_doubt_nm: only marked, kept), "
                    "'anchored', or 'not evaluated' (the ring could not be "
                    "used; see unanchored_failure).",
    "anchor_borderline": "True when the depth is within 2 of its own noise "
                         "SDs (subsamples of the ring's input) of the "
                         "exclusion depth: the verdict depends on the "
                         "subsample.",
    "anchor_source": "Why the cluster is out of (or back in) the 'discard "
                     "applied' analysis: 'automatic', 'widefield' (the old "
                     "two-image method), 'manual' (excluded by hand), "
                     "'restored by hand' (the rule excluded it, a person "
                     "kept it); empty when kept by the rule.",
    "anchor_reason": "One sentence: the depth that excluded or marked it, or "
                     "the reason and date of the decision by hand.",
}

LOCALIZATIONS: Dict[str, str] = {
    "source_index": "Where this localization sits in the selection given to "
                    "the analysis. Joins the table back to the file.",
    "x_nm": "Its position, in x.",
    "y_nm": "The same, in y.",
    "z_nm": "Its depth.",
    "in_slab": "Whether the axial slab that was analysed holds it.",
    "cluster_label": "The cluster it belongs to; empty when it is in none.",
    "cluster_kept": "Whether that cluster survived the automatic curation.",
    "cluster_discarded": "Whether the discard left that cluster out.",
}

# Rows of the panel's per-localization table, joined into the localization
# table under the axoplasm_ prefix.
AXOPLASM_LOCALIZATIONS: Dict[str, str] = {
    "distance_to_tubulin_edge_nm": "How far this localization is from the "
                                   "edge of the axon in the tubulin image; "
                                   "positive is inside.",
    "label": "What the panel calls it: inside, membrane or no cluster.",
}

AXOPLASM_CLUSTERS: Dict[str, str] = {
    "depth_in_tubulin_mask_nm": "How far inside the tubulin image puts this "
                                "cluster.",
    "depth_in_spectrin_interior_nm": "How far inside the spectrin image puts "
                                     "it.",
    "inside_tubulin_mask": "Whether the tubulin image puts it inside by more "
                           "than the margin.",
    "inside_spectrin_interior": "Whether the spectrin image does.",
    "group": "What follows from the two: discarded, tubulin only, spectrin "
             "only, or membrane.",
    "spectrin_interior_image": "The image the interior came from.",
}

CROSSCHANNEL: Dict[str, str] = {
    # --- which two files, and how they were put in one frame
    "source_channel_a": "The reference channel, betaII-spectrin: it defines "
                        "the MPS and its contour is the perimeter everything "
                        "here is measured on.",
    "source_channel_b": "The partner channel (alpha-adducin or 4.1B), "
                        "already moved into channel A's frame.",

    # --- how channel B was put in channel A's frame
    "registration_source": "How the two channels were aligned: fiducials, "
                           "cross-correlation, set by hand, or none -- in "
                           "which case every number below inherits an "
                           "unknown error.",
    "registration_shift_x_nm": "How far channel B was moved in x.",
    "registration_shift_y_nm": "In y.",
    "registration_shift_z_nm": "In z.",
    "registration_lateral_rms_nm": "What is left over laterally after that "
                                   "move. A cross-channel distance of this "
                                   "size is not a measurement.",
    "registration_axial_rms_nm": "What is left over axially. It bounds the "
                                 "axial phase one to one.",
    "registration_n_pairs": "How many matched pairs the alignment rests on.",

    # --- axial phase
    "axial_peak_a_nm": "Where channel A's axial distribution peaks.",
    "axial_peak_b_nm": "Where channel B's does.",
    "axial_offset_nm": "Signed, B minus A.",
    "period_a_nm": "Channel A's mean spacing between axial peaks.",
    "period_b_nm": "Channel B's.",
    "axial_phase_fraction": "The offset as a fraction of the period, folded "
                            "to 0-0.5: 0 is in phase, 0.5 is antiphase.",
    "axial_phase_uncertainty": "The axial registration error as a fraction of "
                               "the period. A phase measured below this is "
                               "not measured.",
    "axial_peaks_separated_a": "Whether channel A's axial density actually "
                               "dips between the components fitted to it. "
                               "False means the mixture split one broad "
                               "distribution, and the period rests on that "
                               "split; the phase label then reads "
                               "undetermined whatever the fraction is.",
    "axial_peaks_separated_b": "The same for channel B.",
    "axial_phase_label": "A word for the fraction, never a conclusion: near "
                         "in-phase, intermediate, near antiphase.",

    # --- the slab and the clusters in it
    "slab_zmin_nm": "Bottom of the axial slab both channels were cut to.",
    "slab_zmax_nm": "Top of it.",
    "n_clusters_a": "Channel A clusters kept in that slab.",
    "n_clusters_b": "Channel B clusters kept in it.",

    # --- each channel's own measures, on its own clusters
    "perimeter_a_um": "Length of channel A's contour through its cluster "
                      "centres.",
    "perimeter_b_um": "The same for channel B, through ITS centres -- not "
                      "the perimeter the shared occupancy is measured on.",
    "clusters_per_um_a": "Channel A clusters per micrometre of its contour.",
    "clusters_per_um_b": "Channel B clusters per micrometre of its own.",
    "median_area_a_nm2": "Median area of channel A's clusters.",
    "median_area_b_nm2": "Median area of channel B's. Far above channel A's "
                         "usually means DBSCAN chained a diffuse signal "
                         "instead of resolving clusters.",
    "median_1nn_a_nm": "Median nearest-neighbour spacing within channel A.",
    "median_1nn_b_nm": "Within channel B.",
    "n_locs_slab_a": "Channel A localizations in the analysed slab.",
    "n_locs_slab_b": "Channel B localizations in it.",
    "eps_a_nm": "DBSCAN radius used on channel A.",
    "eps_b_nm": "DBSCAN radius used on channel B: its own, never channel "
                "A's. Derived as eps_a x NeNA_b / NeNA_a, or typed; "
                "eps_b_origin says which.",
    "min_samples_a": "DBSCAN minimum cluster size on channel A.",
    "min_samples_b": "On channel B: its own, derived as round(1 + (n_loc_mol_b "
                     "- 1) x (1 - exp(-eps_b^2 / (4 NeNA_b^2)))), or typed; "
                     "min_samples_b_origin says which.",

    # --- heterotypic nearest neighbours
    "median_hetero_nn_a_to_b_nm": "From each channel-A cluster to the nearest "
                                  "channel-B one. Not symmetric.",
    "median_hetero_nn_b_to_a_nm": "The same the other way round.",
    "null_median_hetero_nn_nm": "The same distance when channel B is "
                                "scattered at random through the annulus its "
                                "clusters occupy.",
    "fraction_null_below_measured": "How often chance alone gets that close. "
                                    "Above 0.05 the measured proximity is not "
                                    "distinguishable from chance.",

    # --- angular and radial, about channel A's contour centre
    "angular_nn_median_deg": "Median angle from each channel-A cluster to the "
                             "nearest channel-B one, about channel A's "
                             "contour centre.",
    "rotation_fraction_of_spacing": "The best rigid rotation of channel B "
                                    "onto A, as a fraction of the mean "
                                    "angular spacing.",
    "max_angular_correlation": "The correlation at that rotation.",
    "median_radius_a_nm": "Median distance of channel A's clusters from the "
                          "contour centre.",
    "median_radius_b_nm": "The same for channel B's.",

    "locs_within_eps_a": "Median localizations within channel 1's eps of a "
                         "channel-1 localization, itself included: how "
                         "densely the channel is sampled, in the unit Min Pts "
                         "counts. A diagnostic, never used to cluster.",
    "locs_within_eps_b": "The same for channel 2, within its own eps. A ratio "
                         "of three or more either way is warned about; it is "
                         "never used to set Min Pts, because it also holds "
                         "the partner protein's abundance.",
    "channel_b_parameter_source": "Where channel 2's Epsilon and Min Pts "
                                  "came from, as the caller stated it: "
                                  "derived from channel 2's NeNA and dye, "
                                  "typed by a person, or empty when a "
                                  "script supplied them and nobody can say.",
    "eps_b_origin": "'derived' when channel 2's eps is the value derived "
                    "from the two channels' NeNA (docs/channel2_parameters"
                    ".md); 'user' when it was typed or edited.",
    "min_samples_b_origin": "'derived' when channel 2's Min Pts is the value "
                            "derived from its NeNA and its dye's switching "
                            "cycles; 'user' when typed or edited.",
    "nena_a_nm": "Channel 1's precision used for eps_b: NeNA (Endesfelder et "
                 "al. 2014) on channel 1's whole-field, unlinked file, or the "
                 "published reference when it could not be measured "
                 "(nena_a_origin).",
    "nena_a_source_file": "The whole-field file channel 1's NeNA was measured "
                          "on (name only).",
    "nena_a_origin": "'measured', or 'reference (Gazal et al. 2026)' when "
                     "channel 1's NeNA could not be measured.",
    "eps_a_reference_nm": "The channel-1 radius eps_b was scaled from.",
    "nena_b_nm": "Channel 2's precision: NeNA on its whole-field, unlinked "
                 "file. Empty when it could not be measured.",
    "nena_b_source_file": "The whole-field file channel 2's NeNA was measured "
                          "on (name only).",
    "dye_b": "Channel 2's dye, as chosen from the table of Dempsey et al. "
             "(2011).",
    "buffer_b": "Channel 2's imaging buffer (MEA or bME), as chosen.",
    "n_loc_mol_b": "Localizations per labelled molecule used for Min Pts: "
                   "the dye's mean switching cycles in that buffer.",
    "n_loc_mol_b_origin": "Where n_loc_mol_b comes from: the dye table "
                          "(Dempsey et al. 2011, Table 1), a lower bound "
                          "until the label is calibrated.",

    # --- where channel B sits relative to channel A's outline
    "radial_offset_b_median_nm": "Median signed distance from channel B's "
                                 "cluster centres to channel A's contour. "
                                 "Negative is towards the inside of the "
                                 "axon. This is what tells 'the partner is "
                                 "not there' apart from 'the partner is "
                                 "there, further in'.",
    "radial_offset_b_q1_nm": "First quartile of those distances.",
    "radial_offset_b_q3_nm": "Third quartile.",
    "n_clusters_b_inside_contour": "Channel B clusters inside the contour.",
    "n_clusters_b_outside_contour": "Channel B clusters outside it.",
    "radial_offset_a_median_nm": "The same measurement on channel A's OWN "
                                 "localizations: the contour runs through "
                                 "channel A's cluster centres, so its own "
                                 "data straddles it. Read channel B's offset "
                                 "against this, never alone.",
    "radial_offset_a_q1_nm": "First quartile of channel A's own offsets.",
    "radial_offset_a_q3_nm": "Third quartile of them.",

    # --- shared perimeter occupancy
    "shared_of_a": "THE headline: the fraction of the perimeter covered by "
                   "channel A that channel B also covers. A fraction, not a "
                   "percentage. Read it against null_median_shared_of_a.",
    "shared_of_b": "The same the other way round -- is the partner confined "
                   "to spectrin? Fragile when channel B has few clusters.",
    "shared_jaccard": "The symmetric version: shared over the union.",
    "shared_length_nm": "The length both channels cover.",
    "occupancy_a_percent": "How much of its own perimeter channel A covers.",
    "occupancy_b_percent": "How much of CHANNEL A's perimeter channel B "
                           "covers. Not channel B's own occupancy.",
    "n_clusters_b_on_contour": "Channel-B clusters that reach at least one "
                               "sampled point of channel A's contour.",
    "n_clusters_b_off_contour": "Channel-B clusters that reach none, and so "
                                "contribute nothing. Not an error: the "
                                "contour is a line through cluster centres, "
                                "not a boundary, and both channels straddle "
                                "it.",
    "b_largest_single_share": "The share the largest single channel-B cluster "
                              "covers on its own. High means the answer rests "
                              "on one blob.",
    "median_patch_a_nm": "The median length of channel A's covered patches: "
                         "the scale this overlap is resolved at.",
    "null_median_shared_of_a": "shared_of_a when channel B's ellipses are "
                               "moved rigidly to random positions in the "
                               "annulus, keeping their sizes, shapes and "
                               "their own minimum separation.",
    "null_ci_lo_shared_of_a": "2.5th percentile of that null.",
    "null_ci_hi_shared_of_a": "97.5th percentile of it.",
    "p_null_at_least_measured": "How often the null reaches the measured "
                                "value. Above 0.05 the overlap is not "
                                "distinguishable from chance.",
    "n_null_shared": "Draws behind those three numbers.",
    "registration_band_lo_shared_of_a": "What the registration error alone "
                                        "does to shared_of_a: the 2.5th "
                                        "percentile over random shifts of "
                                        "that size.",
    "registration_band_hi_shared_of_a": "The 97.5th percentile of the same.",
    "shared_not_measured_because": "Why there is no shared fraction in this "
                                   "row, when there is not.",

    "n_warnings": "How many warnings this pair raised. Never zero by "
                  "default; read them.",
}


# The z calibration of the measurement (tools.mps_zcalibration): read and
# written, never used to change a z. Picasso has already multiplied every z
# by the record's magnification factor when it fitted it.
Z_CALIBRATION: Dict[str, str] = {
    "z_calibration_source": "Which astigmatism calibration record describes "
                            "this file's z: the file's own record (in its "
                            "Picasso metadata: the 'Z Calibration' block or "
                            "the 'Fit 3D' step) or a calibration file chosen "
                            "for this measurement. Empty: none read.",
    "z_calibration_file": "The name of that calibration file (never its "
                          "folder). Empty when the record does not name it.",
    "z_magnification": "The magnification factor m Picasso multiplied into "
                       "every z and lpz to correct the refractive-index "
                       "mismatch. A property of the microscope, objective, "
                       "medium and depth (published values 0.57-0.85), "
                       "never assumed: empty when not recorded.",
    "z_magnification_origin": "Where m comes from: 'file record', 'chosen "
                              "record', 'user' (typed, because the record "
                              "has none) or 'blank' (not recorded).",
    "z_calibrated_min_nm": "Lower end of the calibrated z range, in the "
                           "file's z units (stage z x m).",
    "z_calibrated_max_nm": "Upper end of the calibrated z range.",
    "z_calibrated_range_nm": "Width of the calibrated z range: (steps - 1) x "
                             "step x m.",
    "z_calibrated_range_origin": "'nominal': from the record alone, centred "
                                 "on 0 and uncertain by the z_zero Picasso "
                                 "does not store; 'bead stack': placed by "
                                 "the bead stack's own z_zero.",
    "z_calibration_match": "Whether the file's own record is the same "
                           "calibration as the chosen one ('same' / "
                           "'different'); empty unless both exist.",
    "z_bead_stack_file": "The localized bead stack the next five columns "
                         "come from (its name only).",
    "z_zero_nm": "Where the stack's width curves cross: the shift Picasso "
                 "applied to the stage axis and does not store.",
    "z_usable_min_nm": "Lower end of the usable z range: the steps around "
                       "z = 0 whose refitted z lies within 2 sigma_z of the "
                       "true z, off the fit bounds, and in order.",
    "z_usable_max_nm": "Upper end of the usable z range.",
    "z_lateral_shift_x_nm": "How far x moves with depth across the usable "
                            "range (max - min of a cubic B-spline of the "
                            "beads' x(z), Carlini et al. 2015). Not "
                            "subtracted from anything here.",
    "z_lateral_shift_y_nm": "The same for y.",
}

# SCI-6: the axial mixture, after the z calibration.
AXIAL_MIXTURE: Dict[str, str] = {
    "main_peak_mode": "Which reading of 'the main peak of the axial "
                      "localization distribution' centred the automatic slab: "
                      "'density' (the mode of the fitted mixture) or 'weight' "
                      "(the mean of its heaviest component).",
    "z_peak_density_nm": "The mode of the fitted axial mixture density.",
    "z_peak_weight_nm": "The mean of the mixture component with the largest "
                        "weight. The two readings coincide when the rings are "
                        "resolved.",
    "slab_foreign_fraction": "Expected share (0-1) of the slab's localizations "
                             "that the fitted mixture attributes to components "
                             "other than the slab's own ring: sum of w_k x the "
                             "mass of component k inside the slab over the other "
                             "components, divided by the same sum over all. This "
                             "program warns above 0.05.",
    "delta_z_se_nm": "Standard error of delta_z_mean_nm: sd of the spacings / "
                     "sqrt(m), with m >= 2 spacings; empty with one. Delta-Z is "
                     "in the length units of the measurement's z calibration.",
}

# How eps was chosen (decision P-R58; docs/eps_kdistance.md).
EPS_CHOICE: Dict[str, str] = {
    "eps_source": "How eps_nm was chosen: 'gazal_default' (the published "
                  "25 nm of Gazal et al. 2026, kept or chosen on the "
                  "k-distance graph), 'typed', 'kdist_valley' (the valley of "
                  "the k-distance density, suggested and accepted) or "
                  "'kdist_click' (set by clicking on the k-distance graph). "
                  "The eps_kdist_ columns are filled only when the choice "
                  "was made on the graph.",
    "eps_kdist_k": "The neighbour rank of the k-distance graph eps was "
                   "chosen on: k = min samples - 1 other localizations "
                   "(scikit-learn counts the point itself; Schubert et al. "
                   "2017).",
    "eps_kdist_valley_nm": "Where the graph's valley was: the minimum of the "
                           "kernel density of log10(k-distance) between its "
                           "two most prominent modes, Silverman bandwidth. "
                           "Empty when the density had no valley -- nothing "
                           "was then suggested.",
    "eps_kdist_valley_low_nm": "Lower end of the valley's 95 % bootstrap "
                               "interval (200 resamples of the "
                               "k-distances): the uncertainty of where the "
                               "valley is, not of the right eps.",
    "eps_kdist_valley_high_nm": "Upper end of that interval.",
    "eps_kdist_unclustered_fraction": "Share of the localizations the graph "
                                      "was drawn on that DBSCAN(eps_nm, min "
                                      "samples) labels noise, before any "
                                      "cluster is removed by the analysis.",
    "eps_kdist_file": "The file whose selection the k-distance graph was "
                      "drawn on (name only); another file's in a batch.",
}

# The clusters cut by the pick boundary (decision P-R59;
# docs/cut_clusters.md).
CUT_CLUSTERS: Dict[str, str] = {
    "cut_status": "Whether the cut-cluster rule was applied: 'evaluated', or "
                  "why not -- 'boundary unknown: not evaluated' (no Picasso "
                  "pick boundary known for this file, or more than one pick "
                  "fits it and none was chosen), 'NeNA unknown: not "
                  "evaluated' (no localization precision for the channel), "
                  "'boundary does not hold the localizations: not "
                  "evaluated'. Nothing is guessed in their place.",
    "cut_rule": "The rule: a kept cluster is cut when any of its "
                "localizations lies within cut_nena_multiple x NeNA of the "
                "pick boundary (docs/cut_clusters.md).",
    "cut_nena_multiple": "How many NeNA from the boundary count as 'at the "
                         "edge' (1 by default: derivation and simulation in "
                         "docs/cut_clusters.md).",
    "cut_nena_nm": "The NeNA (localization precision) the rule used, in nm.",
    "cut_nena_source": "Where that NeNA came from: this file (when it has "
                       "enough next-frame pairs), a whole-field file "
                       "measured for this acquisition folder, and how many "
                       "next-frame pairs it had.",
    "cut_margin_nm": "cut_nena_multiple x cut_nena_nm: the distance from the "
                     "boundary within which a localization marks its "
                     "cluster as cut.",
    "cut_boundary_source": "The Picasso pick file and the pick in it the "
                           "boundary is, what the file's own record (group, "
                           "shape, area) says about it, and whether it was "
                           "found automatically or chosen by hand.",
    "cut_boundary_shape": "The pick's shape in Picasso: Polygon, Circle, "
                          "Rectangle or Square.",
    "n_clusters_cut": "k: how many of the clusters the curation kept were "
                      "cut by the pick boundary.",
    "n_clusters_cut_of": "K: how many clusters the curation kept (before "
                         "the cut ones were left out, when they were).",
    "cut_percent": "100 k / K.",
    "cut_excluded": "True when the cut clusters were left out of the "
                    "analysis ('Exclude cut clusters from the analysis'); "
                    "they are then counted in n_clusters_removed and every "
                    "measured column is without them. False: they are "
                    "flagged only, and measured as they are in the file.",
}

# --------------------------------------------------------------------------
# The perimeter density analysis (axon tables v17: the pd_* block, written
# last; the long tables <stem>_perimeter_*.csv). Every entry says what the
# number is, how to read it, when it is empty and where it comes from. Origin
# tags: [director] fixed by the thesis director, [lit] literature, [own sim]
# simulation of this programme, [real data, pilot] the 18-axon pilot of
# 2026-11-16 (ONE animal), [ad hoc] chosen here and to be swept, [derivation]
# follows from the geometry. See docs/perimeter_density.md.
# --------------------------------------------------------------------------

_EMPTY_NOT_RUN = (" Empty when the perimeter density analysis was not run "
                  "(pd_status 'not run') or failed before this step.")

PERIMETER: Dict[str, str] = {
    # --- identification of the trace
    "trace_label": "Readable year.ROI.axon label of the trace, for example "
                   "26.1.2. A label for reading, NOT a key: the key is "
                   "axon_id. A part that could not be read is written NA. "
                   "Origin: [ad hoc] format; the year comes from the date at "
                   "the start of the file name (yy.m.d_) or from the year "
                   "typed for the folder." + _EMPTY_NOT_RUN,
    "trace_year": "Two-digit acquisition year of the label; NA when the file "
                  "name has no date at its start and no year was typed. "
                  "Origin: [ad hoc]." + _EMPTY_NOT_RUN,
    "trace_complete": "True when the year, the ROI and the axon number of the "
                      "label were all read; False when any reads NA."
                      + _EMPTY_NOT_RUN,
    "acquisition_date": "Acquisition date yy.mm.dd read at the start of the "
                        "file name; empty when the name has none.",
    "trace_warnings": "Which part of the label is missing and why, separated "
                      "by ' | '; empty when the label is complete.",
    # --- state
    "pd_status": "'ok', or why the analysis has no full result: 'not run', "
                 "'no focus' (a focus criterion was asked and no focus is "
                 "known; it never falls back to the peak), 'contour failed', "
                 "'too few localizations', 'empty profile', 'invalid input', "
                 "'internal error'. Any status but 'ok' still gives a complete "
                 "row: the columns after the failure are empty.",
    "pd_failure": "One sentence saying why pd_status is not 'ok'; empty when "
                  "it is.",
    "pd_version": "Version of the perimeter density analysis that wrote the "
                  "row." + _EMPTY_NOT_RUN,
    # --- z0 and the two foci
    "pd_z0_mode": "How the height z0 of the central slice was chosen: 'peak' "
                  "(the localization peak in z, the default) or 'focus'. "
                  "Origin: [director] decision of 2026-11-16 (peak by "
                  "default, focus selectable)." + _EMPTY_NOT_RUN,
    "pd_focus_kind": "Which focus when pd_z0_mode is 'focus': 'nominal' or "
                     "'empirical'; empty for 'peak'. Both foci are always "
                     "estimated and exported whatever z0 is.",
    "pd_z0_criterion": "The criterion exactly as chosen: 'peak', "
                       "'focus_nominal' or 'focus_empirical'."
                       + _EMPTY_NOT_RUN,
    "pd_z0_nm": "z of the centre of the central slice (slice 0)."
                + _EMPTY_NOT_RUN,
    "pd_z_peak_nm": "z of the highest localization density: the maximum of a "
                    "Gaussian kernel density of the axon's z (sigma 25 nm). "
                    "It differs from the programme's mixture-model peak by a "
                    "median of 7.7 nm (maximum 36 nm) on the pilot. Origin: "
                    "[real data, pilot] for the 25 nm kernel." + _EMPTY_NOT_RUN,
    "pd_z_focus_nominal_nm": "The nominal focus: z = 0 of the file, shifted by "
                             "the z calibration record when the file has "
                             "one. Empty when no record says where it is.",
    "pd_z_focus_empirical_nm": "The empirical focus: the z where the median "
                               "of sx - sy crosses zero (the PSF is round "
                               "there). Empty without sx / sy columns or "
                               "without a crossing. It is a measured estimate; "
                               "see pd_focus_empirical_n_bins.",
    "pd_focus_empirical_n_bins": "z bins (40 nm wide, at least 200 "
                                 "localizations each) behind the empirical "
                                 "focus; few bins make it a weak estimate. "
                                 "Origin: [ad hoc].",
    "pd_peak_minus_focus_nm": "Peak minus focus of the SELECTED kind (the "
                              "nominal one when z0 is the peak), in nm: "
                              "positive when the peak lies at higher z than "
                              "the focus. Both foci are exported below.",
    "pd_abs_peak_minus_focus_nm": "Absolute value of pd_peak_minus_focus_nm: "
                                  "how far the focus is from the peak, "
                                  "whatever the side.",
    "pd_peak_minus_focus_nominal_nm": "Peak minus the NOMINAL focus, signed "
                                      "(nm). Empty without a nominal focus.",
    "pd_abs_peak_minus_focus_nominal_nm": "Absolute peak - nominal focus (nm).",
    "pd_peak_minus_focus_empirical_nm": "Peak minus the EMPIRICAL focus, "
                                        "signed (nm). Empty without an "
                                        "empirical focus.",
    "pd_abs_peak_minus_focus_empirical_nm": "Absolute peak - empirical focus "
                                            "(nm).",
    # --- slices
    "pd_slice_thickness_nm": "Thickness of each of the three contiguous "
                             "slices (-1, 0, +1). Origin: [director] 170 nm, "
                             "the programme's reference period (about 170 "
                             "nm in the spectrin lattice, [lit]).",
    "pd_slice_guard_nm": "Axial guard cut from each side of a slice: 0 for "
                         "contiguous slices (the main run); the 42.5 nm guard "
                         "(T / 4) only exists as the axial-guard sensitivity "
                         "of question C. Origin: [ad hoc].",
    "pd_n_locs_m1": "Localizations in slice -1." + _EMPTY_NOT_RUN,
    "pd_n_locs_0": "Localizations in the central slice (slice 0)."
                   + _EMPTY_NOT_RUN,
    "pd_n_locs_p1": "Localizations in slice +1." + _EMPTY_NOT_RUN,
    "pd_pct_m1_of_0": "Localizations of slice -1 as a percentage of the "
                      "central slice's. No lateral slice is dropped for "
                      "having few localizations: this number says how thin "
                      "it is. Origin: [director] decision of 2026-11-16."
                      + _EMPTY_NOT_RUN,
    "pd_pct_p1_of_0": "Localizations of slice +1 as a percentage of the "
                      "central slice's (same reading)." + _EMPTY_NOT_RUN,
    "pd_frac_locs_in_slices": "Fraction of the axon's localizations that "
                              "fall in the three slices (510 nm in z); the "
                              "rest lie outside them." + _EMPTY_NOT_RUN,
    "pd_z_coverage_m1": "Fraction of slice -1's z interval inside the file's "
                        "z range; below 1 the slice is cut by the edge of the "
                        "data and its counts are biased low. Origin: "
                        "[derivation].",
    "pd_z_coverage_0": "Same for the central slice.",
    "pd_z_coverage_p1": "Same for slice +1.",
    "pd_z_truncated_slices": "The slices (written -1, +1, ...) whose z "
                             "interval leaves the file's z range; empty when "
                             "none. A pair that includes one is to be read "
                             "with care.",
    "pd_lpz_nm": "Axial localization precision used for the expected axial "
                 "leak of question C. Origin: [real data, pilot] about 47 nm; "
                 "pd_lpz_source says where this value came from.",
    "pd_lpz_source": "Where pd_lpz_nm came from (the file's own column, or "
                     "the default).",
    # --- contour
    "pd_contour_method": "How the perimeter contour was made: "
                         "'density_nodes' (automatic, the default), 'polar' "
                         "(automatic, star-shaped axons only) or 'drawn' "
                         "(a manual trace, offered for comparison). Origin: "
                         "[director] decision of 2026-11-16: automatic.",
    "pd_contour_status": "'ok', or why the automatic contour failed.",
    "pd_contour_knot_nm": "Spacing of the spline's knots. Origin: [ad hoc] "
                          "400 nm, the curvature module's knot spacing.",
    "pd_contour_length_nm": "Length of the contour: the perimeter the arc "
                            "bins are laid on." + _EMPTY_NOT_RUN,
    "pd_contour_residual_sd_nm": "Standard deviation of the localizations' "
                                 "distances to the contour.",
    "pd_contour_self_crossings": "Times the contour crosses itself; above 0 "
                                 "arc distances and windows are unreliable."
                                 + _EMPTY_NOT_RUN,
    "pd_contour_vs_polar_nm": "Mean distance between the automatic contour "
                              "and the polar one; a large value, or a length "
                              "that doubles the polar one, means the contour "
                              "does not follow the membrane. Empty when not "
                              "computable.",
    "pd_contour_unsupported_fraction": "Fraction of the central slice's bins "
                                       "that are not measured (arc stretches "
                                       "longer than 400 nm without data, "
                                       "[ad hoc]); those bins are left out "
                                       "of the tests.",
    "pd_contour_star_shaped": "True when the contour is star-shaped about "
                              "its centre (the polar contour is then valid).",
    "pd_contour_n_nodes": "Nodes of the automatic contour before smoothing.",
    "pd_contour_reversed": "True when the contour came out clockwise and was "
                           "reversed to counter-clockwise before use (the "
                           "curvature sign depends on it).",
    "pd_drawn_vs_auto_nm": "Mean absolute offset of a manually drawn trace "
                           "from the automatic contour; empty when no trace "
                           "was given.",
    "pd_radial_shift_m1_nm": "Median radial offset of slice -1 minus that of "
                             "the central slice (radial offset positive "
                             "outside the contour): a systematic shift "
                             "between slices of the same axon.",
    "pd_radial_shift_p1_nm": "Same for slice +1 minus the central slice.",
    # --- band
    "pd_band_mode": "'fixed' (the band half-width is typed) or 'fwhm' (the "
                    "axon's own radial FWHM, clipped to 50-150 nm).",
    "pd_band_half_width_nm": "Half-width of the radial band around the "
                             "perimeter contour inside which a localization "
                             "counts here (100 nm). NOT the 50 nm band of the "
                             "pipeline null H2 (tools.mps_pipeline_null): "
                             "another module and another question. Origin: "
                             "[ad hoc] chosen on [real data, pilot]; 50 and "
                             "150 nm are run as a mandatory sensitivity "
                             "(pd_band_sensitivity).",
    "pd_radial_fwhm_nm": "Gaussian-equivalent full width at half maximum of "
                         "the radial distances (2.3548 x 1.4826 x MAD). "
                         "Origin: [derivation].",
    "pd_radial_sd_nm": "Robust radial standard deviation (1.4826 x MAD).",
    "pd_frac_within_50nm": "Fraction of the localizations within +-50 nm of "
                           "the contour.",
    "pd_frac_within_100nm": "Fraction of the localizations within +-100 nm of "
                            "the contour.",
    "pd_frac_within_150nm": "Fraction of the localizations within +-150 nm of "
                            "the contour.",
    # --- grid
    "pd_bin_nm": "Nominal arc bin. Origin: [director] 100 nm.",
    "pd_bin_nm_effective": "Bin actually used: perimeter divided by the "
                           "number of bins (99-101 nm). Origin: [derivation].",
    "pd_n_bins": "Arc bins around the perimeter." + _EMPTY_NOT_RUN,
    "pd_max_count_0": "Localizations in the fullest bin of the central slice "
                      "(the maximum question A aligns on).",
    "pd_max_tie": "True when another bin ties the maximum; the first one is "
                  "used.",
    # --- question A
    "pd_a_exclusion_nm": "Width of the zone around the maximum left out of "
                         "A's statistics: the run of bins above half height "
                         "(measured above the median) plus one buffer bin, "
                         "and at least 200 nm on each side. Origin: [ad hoc].",
    "pd_a_contrast": "Question A, near-far contrast: mean relative density "
                     "within 1000 nm of the excluded zone minus the mean of "
                     "the rest. Positive reads as an attractor (density "
                     "gathers around the maximum), negative as an inhibition "
                     "zone. The attractor reading is EXPLORATORY: it only "
                     "holds if it survives the sensitivity to the exclusion "
                     "zone (fixed 200 and 400 nm) and to the band. Origin: "
                     "[ad hoc] 1000 nm.",
    "pd_a_contrast_z": "pd_a_contrast in standard deviations of its null: the "
                       "remainder of the profile shifted circularly (toroidal "
                       "shift; Lotwick & Silverman 1982, [lit], not re-read). "
                       "Two-sided.",
    "pd_a_contrast_p": "Two-sided Monte Carlo p of the contrast, (1 + # "
                       "replicates at least as extreme) / (1 + n_null) "
                       "([lit] Phipson & Smyth 2010); its floor is "
                       "1 / (n_null + 1), 0.0005 with 1999. The 18 pilot "
                       "axons are ONE animal: with n_animals = 1 no "
                       "comparison between animals is possible.",
    "pd_a_period_nm": "Period of the strongest phase-locked peak of the "
                      "remainder's periodogram, among periods from 300 nm "
                      "([ad hoc]). Empty when A could not be tested.",
    "pd_a_period": "The statistic behind pd_a_period_nm: the phase-locked "
                   "periodogram power of the strongest peak (relative "
                   "density squared, not divided by the total variance). "
                   "Read it through pd_a_period_z and pd_a_period_p.",
    "pd_a_period_z": "That peak's power in null standard deviations; "
                     "one-sided. Descriptive for ONE axon: a periodic "
                     "remainder looks almost the same after a shift by its "
                     "period, so the per-axon test has little power. The "
                     "test with power is the pooled one.",
    "pd_a_period_p": "One-sided Monte Carlo p of the period peak (same floor "
                     "as pd_a_contrast_p).",
    "pd_a_period_at_edge": "True when the strongest peak sits at the longest "
                           "scanned period: a trend (a halo or a depleted "
                           "zone), not a repeat. Empty when A could not be "
                           "tested.",
    # --- question B
    "pd_b_window_nm": "Tortuosity window (2 bins). Origin: [director] 200 nm.",
    "pd_b_spearman": "Question B: Spearman correlation between the "
                     "tortuosity of the 200 nm windows (chord / arc: 1 is "
                     "straight, lower is more curved) and their density. It "
                     "comes out NEGATIVE when density is higher where the "
                     "contour is more curved. Tortuosity does not tell convex "
                     "from concave. A positive value can also be the "
                     "contour-band bias (see pd_b_tau_vs_offset).",
    "pd_b_z": "pd_b_spearman in standard deviations of the circular-rotation "
              "null of the density profile (curvature fixed).",
    "pd_b_p": "Two-sided p against that null, (1 + #) / (1 + n shifts) ([lit] "
              "Phipson & Smyth 2010); all non-zero shifts are used when "
              "there are fewer than n_null.",
    "pd_b_tau_min": "Lowest tortuosity among the windows (the most curved "
                    "one).",
    "pd_b_curvature_resolved": "True when the perimeter's equivalent radius "
                               "(length / 2 pi) is at least 600 nm; below it "
                               "the tortuosity differences are under the "
                               "resolution and B is not interpretable. "
                               "Origin: [derivation].",
    "pd_b_tau_vs_offset": "Control for B: Spearman between the windows' "
                          "tortuosity and their median radial offset. Far "
                          "from 0 means the contour cutting corners could "
                          "give B without biology. Only simulation controls "
                          "that bias ([own sim]).",
    # --- question C
    "pd_c_r_m1_0": "Question C: correlation between the density profiles of "
                   "slices -1 and 0. Neighbouring slices share axial "
                   "leakage: compare with pd_c_leak_r_m1_0_sz47nm / _sz77nm "
                   "(the leak with this axon's slice fractions) before "
                   "reading it as registry.",
    "pd_c_z_m1_0": "pd_c_r_m1_0 in standard deviations of the circular-shift "
                   "null.",
    "pd_c_p_m1_0": "One-sided p of pd_c_r_m1_0 over the circular shifts of "
                   "the profile. Its floor is 1 / (n_bins + 1), about 0.0105 "
                   "with 94 bins, whatever n_null is.",
    "pd_c_r_0_p1": "Correlation between slices 0 and +1 (neighbours; same "
                   "caution about leakage).",
    "pd_c_z_0_p1": "pd_c_r_0_p1 in null standard deviations.",
    "pd_c_p_0_p1": "One-sided p of pd_c_r_0_p1; floor 1 / (n_bins + 1), about "
                   "0.0105 with 94 bins.",
    "pd_c_r_m1_p1": "Correlation between slices -1 and +1, the pair that can "
                    "show registry. Leakage reaches it too: about 0.02 with "
                    "three equal slices and sigma_z 47 nm, but 0.07-0.09 "
                    "(47 nm) to 0.24-0.28 (77 nm) with laterals at ~45 % of "
                    "the central slice, as in the pilot; compare with "
                    "pd_c_leak_r_m1_p1_sz47nm (main; range _sz36nm-_sz51nm, "
                    "v21; _sz77nm is only an upper bound). Read as the "
                    "main reading of C only when pd_c_main_reading is true.",
    "pd_c_z_m1_p1": "pd_c_r_m1_p1 in null standard deviations.",
    "pd_c_p_m1_p1": "One-sided p of pd_c_r_m1_p1; floor 1 / (n_bins + 1), about "
                   "0.0105 with 94 bins.",
    "pd_c_leak_expected_r": "Largest correlation between NEIGHBOURING slices "
                            "that axial leakage alone produces: about 0.27 "
                            "for 170 nm slices and a 47 nm axial precision. "
                            "With the 42.5 nm axial guard it is about 0.12, "
                            "not below 0.10. Counts are taken noise-free, so "
                            "it is an upper bound. Origin: [derivation] with "
                            "the precision of [real data, pilot]. Reported "
                            "for reference only: since 2026-11-17 "
                            "pd_c_reading compares the neighbours with the "
                            "leak of the axon's own slice fractions "
                            "(pd_c_leak_r_*_sz47nm, range _sz36nm-_sz51nm).",
    "pd_c_reading": "Plain reading of the three correlations: 'registry "
                    "between slices', 'leakage only', 'neighbours above "
                    "leakage' (suggestive), 'no correlation between slices' "
                    "or 'not computable'. A suffix says when the reading "
                    "cannot be read as it stands: ' (contour not reliable: "
                    "not interpretable)' or ' (contour not assessed: not "
                    "interpretable)' when the quality gate leaves the axon "
                    "out of the pooled analyses, and ' (slices do not "
                    "coincide: outside the main reading of C, descriptive)' "
                    "when the contour is reliable but pd_c_slices_coincide "
                    "is false (v20; before v20 a ' (slice misfit: ...)' "
                    "suffix followed pd_contour_slice_misfit). A 'registry "
                    "between slices' whose r(-1, +1) is not clearly above "
                    "the leak expected for the pair says so (v21: sigma_z "
                    "47 nm main, measured range 36-51 nm): ' (r(-1, +1) not "
                    "above the axial leak expected even with sigma_z 36 "
                    "nm)', ' (r(-1, +1) at the limit: above the axial leak "
                    "expected with sigma_z 36 nm, not with 47 nm)' or ' "
                    "(r(-1, +1) above the axial leak expected with sigma_z "
                    "47 nm but within its measured range 36-51 nm)'. The "
                    "numbers do not change. 'leakage only' and 'neighbours "
                    "above leakage' compare each significant neighbouring "
                    "pair with ITS OWN leak expected from this axon's slice "
                    "fractions at 47 nm (pd_c_leak_r_m1_0_sz47nm / "
                    "pd_c_leak_r_0_p1_sz47nm; since 2026-11-17, before "
                    "against pd_c_leak_expected_r of equal slices). When an "
                    "end of the measured range gives another reading the "
                    "label is the 47 nm one followed by ' (within the "
                    "measured range of sigma_z 36-51 nm: <reading> at 36 "
                    "nm; <reading> at 51 nm)'. The 77 nm upper bound does "
                    "not enter the reading. Origin: [ad hoc] rule, "
                    "tolerance 0.10 over the expected leak (neighbours).",
    # --- sensitivity and run
    "pd_band_sensitivity": "'consistent' when A, B and C keep their decision "
                           "with the band at 50 and 150 nm, or which decision "
                           "changes. Origin: [ad hoc]; the sensitivity is "
                           "mandatory, its detail is in the sensitivity table.",
    "pd_n_null": "Replicates of the Monte Carlo nulls. Origin: [lit] 1999, "
                 "as the rest of the programme (Phipson & Smyth 2010).",
    "pd_seed": "Seed of the random generator of the nulls (the same seed "
               "gives the same p).",
    "pd_warnings": "Warnings of the analysis, separated by ' | '.",
}

# v19 (package P10b): the automatic contour's assessment, the quality gate of the pooled analyses and the per-axon
# reading of A. Thresholds and their origin: tools.mps_auto_contour (QUALITY_*, SLICE_*), docs/perimeter_density.md.
_EMPTY_NO_CONTOUR = (" Empty when the analysis was not run or stopped before "
                     "the contour.")
PERIMETER_V19: Dict[str, str] = {
    "trace_roi": "The ROI part of trace_label (year.ROI.axon); NA when it "
                 "could not be read. For reading, not a key.",
    "pd_contour_kind": "Which curve the contour is: 'primary' (density nodes, "
                       "knot 400 nm), 'stiff fallback' (knot 600 nm), 'polar "
                       "fallback' or 'polar', 'drawn' (a drawn trace), "
                       "'external' (a curve given by a caller-supplied "
                       "function) or 'none' (rejected)." + _EMPTY_NO_CONTOUR,
    "pd_curve_stratum": "The kind folded into the strata of the pooled "
                        "report: primary, stiff, polar, drawn (external / "
                        "none otherwise). Origin: [own sim] the three "
                        "automatic curves have different errors (p95 median "
                        "36, 81 and 74 nm).",
    "pd_contour_reliable": "The automatic contour's own verdict: False when "
                           "the contour was rejected or failed ANY rule of its "
                           "quality gate, the slice misfit included. Whether "
                           "the axon is pooled is pd_pool_included, which "
                           "does NOT count the slice misfit (decision D12)."
                           + _EMPTY_NO_CONTOUR,
    "pd_contour_failure_reason": "Why the contour was rejected (self-crossing, "
                                 "implausible length, inward notch, no lumen, "
                                 "too few nodes, ...); empty when accepted.",
    "pd_contour_quality_reasons": "Every rule of the quality gate the contour "
                                  "failed, separated by ' | ', the slice "
                                  "misfit included; empty when none.",
    "pd_contour_gate_reasons": "The reasons of pd_contour_quality_reasons "
                               "that keep the axon out of the pooled "
                               "analyses: all but the slice misfit "
                               "(decision D12).",
    "pd_contour_frac100": "Fraction of the localizations of the three slices "
                          "within +-100 nm of the contour. The gate needs >= "
                          "0.55. Origin: [own sim + review] no correct "
                          "simulated curve with a cloud up to 100 nm falls "
                          "below it.",
    "pd_contour_robust_sd_nm": "Robust SD (1.4826 MAD) of the radial offsets "
                               "within 400 nm of the contour. The gate needs "
                               "<= 130 nm. Origin: [own sim + review].",
    "pd_contour_persistent_offset_nm": "Largest offset of the band from the "
                                       "curve that holds over 3 consecutive "
                                       "windows of 500 nm (step 100 nm). The "
                                       "gate needs <= 75 nm. Origin: [own "
                                       "sim] 1.0 % false alarms on correct "
                                       "curves, 39 % detection of bad ones.",
    "pd_contour_off_band_fraction": "Fraction of the arc windows whose band "
                                    "lies more than 100 nm from the curve. "
                                    "The gate needs <= 0.15. Origin: [own "
                                    "sim] 3x the largest of correct curves.",
    "pd_contour_max_window_offset_nm": "Largest offset of ONE window (a "
                                       "diagnostic only: it flagged every real "
                                       "contour, so the gate reads the "
                                       "persistent one).",
    "pd_contour_support_unsupported_fraction": "Share of the contour's length "
                                               "with no localization density "
                                               "under it (runs > the gap "
                                               "length): the curve is an "
                                               "interpolation there. Not the "
                                               "same as pd_contour_"
                                               "unsupported_fraction (bins "
                                               "not measured).",
    "pd_contour_n_unsupported_arcs": "Number of those unsupported stretches.",
    "pd_contour_slice_misfit": "True when the rings of slices -1 or +1 fall "
                               "more than 50 nm RMS from the central one "
                               "about the same curve (decision D7). REPORTED "
                               "ONLY: the axon stays in the main pooled set "
                               "(decision D12); the set without slice misfit "
                               "leaves it out. Origin: [derivation] 50 nm "
                               "costs about 12 % of the band; not calibrated.",
    "pd_contour_slice_rms_minus_nm": "RMS along the curve of the displacement "
                                     "of slice -1's ring from the central "
                                     "one.",
    "pd_contour_slice_rms_plus_nm": "Same for slice +1.",
    "pd_contour_slice_radius_change_nm": "Change of the ring's radius per "
                                         "slice ((a+1 - a-1) / 2; positive = "
                                         "larger upward).",
    "pd_contour_slice_lateral_minus_nm": "Lateral shift of slice -1's ring "
                                         "from the central one.",
    "pd_contour_slice_lateral_plus_nm": "Lateral shift of slice +1's ring "
                                        "from the central one.",
    "pd_contour_slice_frac100_central": "Fraction within +-100 nm for the "
                                        "central slice alone.",
    "pd_contour_slice_frac100_all": "Fraction within +-100 nm for the three "
                                    "slices together.",
    "pd_contour_n_excluded": "Localizations left out of the contour fit by "
                             "the optional interior trim (decision D8, off by "
                             "default: 0).",
    "pd_contour_trim_status": "'trimmed' or 'not trimmed: <why>' when the "
                              "interior trim was switched on; empty when off "
                              "(the default). The trim acts only when the "
                              "polar curve passes the strict rules of the "
                              "automatic contour (no notch under 150 nm, the "
                              "cloud never leaves it, a lumen, ...); otherwise "
                              "'not trimmed: the polar reference fails the "
                              "strict rules of the automatic contour (<rule>)' "
                              "(user's decision 2026-11-17: against a polar "
                              "curve of a non-star-shaped cloud the trim "
                              "removed 5-22 % of the membrane in simulation).",
    "pd_pool_included": "True when the axon enters the pooled analyses (main "
                        "set): its analysis finished and its contour passed "
                        "the quality gate, slice misfit aside (decisions D9, "
                        "D12).",
    "pd_pool_included_without_misfit": "True when the axon enters the pooled "
                                       "set without slice misfit.",
    "pd_pool_exclusion_reason": "Why the axon is not pooled: the analysis "
                                "status and failure, 'contour rejected: ...' "
                                "or 'contour not reliable: ...'; empty when "
                                "pooled.",
    "pd_a_reading": "Reading of question A. In the per-axon table: this axon "
                    "alone (its contrast and period tests, Holm between the "
                    "two; no exclusion-zone sensitivity per axon). In the "
                    "pooled table: the group's pooled reading. Labels: "
                    "'attractor', 'inhibition zone', 'regular' (or a "
                    "combination), 'no structure detectable' or 'not "
                    "computable'. In the pooled table a group of fewer than "
                    "10 axons reads 'n too small (n = k axons < 10): not "
                    "interpreted; computed label ...' (10 = the smallest "
                    "simulated set, sizes 4-11 and 18, where attractor, "
                    "inhibition zone and regular appear with a lower 95 % "
                    "Wilson bound above 0.075 while every null stays at most "
                    "0.075 [own sim; Bradley 1978]; 11 in v20). "
                    "EXPLORATORY: a pooled reading is valid only "
                    "if it survives the exclusion-zone sensitivity; with one "
                    "animal there is no comparison between animals.",
}

# v20 (decision P-R3-1): the calibrated slice criterion of C and the axial leak with the axon's own slice
# fractions. Rules and origins: tools.mps_perimeter_bc (slices_coincide_for_c, leak_expected_r_pairs).
_LEAK_V20 = ("Correlation between slices {pair} expected from axial leakage "
             "ALONE, computed with this axon's own fractions of "
             "localizations in the three slices (taken as the slices' "
             "molecule masses; this can underestimate the leak when a "
             "lateral slice is mostly leak) and an axial width sigma_z = "
             "{sz} nm. Noise-free counts. {why} Widths decided 2026-11-17 "
             "after the NeNA in z of the pilot: 47 nm main, measured range "
             "36-51 nm, 77 nm upper bound; for new data sigma_z should be "
             "measured per file. Origin: [derivation]; widths [real data, "
             "pilot].")
_SZ47 = ("47 nm is the MAIN width: the equivalent sd of the error in z "
         "measured by the NeNA in z with the slow error component (median "
         "47.0 nm across axons; equals Picasso's median lpz).")
_SZ77 = ("77 nm is only an UPPER BOUND: the sd of the rings' components in "
         "z, which also holds the rings' physical thickness.")
_SZ36 = ("36 nm is the LOWER end of the measured range: frame-to-frame "
         "error only (v21).")
_SZ51 = ("51 nm is the UPPER end of the measured range (across axons, with "
         "the slow error component; v21).")
PERIMETER_V20: Dict[str, str] = {
    "pd_c_slices_coincide": "True when the slices coincide well enough for "
                            "the main reading of question C: the better-"
                            "fitting lateral slice is at most 32 nm RMS from "
                            "the central ring (min(pd_contour_slice_rms_"
                            "minus_nm, pd_contour_slice_rms_plus_nm) <= 32). "
                            "False when either misfit could not be measured. "
                            "Empty when the contour carries no assessment. "
                            "Origin: [own sim; Bradley 1978 robustness band "
                            "7.5 %]: the largest misfit at which the false-"
                            "positive rate of C(-1, +1) stays at most 0.075 in "
                            "the worst simulated geometry (calibrated with up "
                            "to 50 localizations per bin: recalibrate for "
                            "denser data).",
    "pd_c_main_reading": "True when the axon is in the main reading of "
                         "question C: it enters the pooled analyses "
                         "(pd_pool_included) AND its slices coincide "
                         "(pd_c_slices_coincide). The others are reported "
                         "apart, descriptively.",
    "pd_c_leak_r_m1_0_sz47nm": _LEAK_V20.format(pair="-1 and 0", sz=47, why=_SZ47),
    "pd_c_leak_r_0_p1_sz47nm": _LEAK_V20.format(pair="0 and +1", sz=47, why=_SZ47),
    "pd_c_leak_r_m1_p1_sz47nm": _LEAK_V20.format(pair="-1 and +1", sz=47, why=_SZ47),
    "pd_c_leak_r_m1_0_sz77nm": _LEAK_V20.format(pair="-1 and 0", sz=77, why=_SZ77),
    "pd_c_leak_r_0_p1_sz77nm": _LEAK_V20.format(pair="0 and +1", sz=77, why=_SZ77),
    "pd_c_leak_r_m1_p1_sz77nm": _LEAK_V20.format(pair="-1 and +1", sz=77, why=_SZ77),
    "pd_c_m1_p1_vs_leak": "pd_c_r_m1_p1 against the leak expected for the "
                          "pair (since v21: 47 nm main, range 36-51 nm, 77 nm "
                          "upper bound): 'above the leak (sigma_z 47 nm, range "
                          "36-51 nm) and above its upper bound (77 nm)', "
                          "'above the leak (sigma_z 47 nm, range 36-51 nm); "
                          "within the upper bound (77 nm)', 'above the leak "
                          "at sigma_z 47 nm, within its range (not above 51 "
                          "nm)', 'at the limit: above the leak at sigma_z 36 "
                          "nm, not at 47 nm' or 'not above the leak (not even "
                          "at sigma_z 36 nm)'. Says nothing about significance "
                          "(pd_c_p_m1_p1). Origin: [derivation].",
}
# v21 (NeNA in z, 2026-11-17): the leak at the two ends of the measured range of sigma_z.
PERIMETER_V21: Dict[str, str] = {
    "pd_c_leak_r_m1_0_sz36nm": _LEAK_V20.format(pair="-1 and 0", sz=36, why=_SZ36),
    "pd_c_leak_r_0_p1_sz36nm": _LEAK_V20.format(pair="0 and +1", sz=36, why=_SZ36),
    "pd_c_leak_r_m1_p1_sz36nm": _LEAK_V20.format(pair="-1 and +1", sz=36, why=_SZ36),
    "pd_c_leak_r_m1_0_sz51nm": _LEAK_V20.format(pair="-1 and 0", sz=51, why=_SZ51),
    "pd_c_leak_r_0_p1_sz51nm": _LEAK_V20.format(pair="0 and +1", sz=51, why=_SZ51),
    "pd_c_leak_r_m1_p1_sz51nm": _LEAK_V20.format(pair="-1 and +1", sz=51, why=_SZ51),
}

# Units that the name alone does not give (or gives wrongly: kappa_mean_per_nm ends in _nm but is 1 / nm).
PERIMETER_UNITS: Dict[str, str] = {
    "pd_pct_m1_of_0": "%", "pd_pct_p1_of_0": "%",
    "pd_frac_locs_in_slices": "fraction (0-1)", "pd_z_coverage_m1": "fraction (0-1)",
    "pd_z_coverage_0": "fraction (0-1)", "pd_z_coverage_p1": "fraction (0-1)",
    "pd_contour_unsupported_fraction": "fraction (0-1)", "pd_frac_within_50nm": "fraction (0-1)",
    "pd_frac_within_100nm": "fraction (0-1)", "pd_frac_within_150nm": "fraction (0-1)",
    "pd_c_leak_expected_r": "r (correlation)", "kappa_mean_per_nm": "1/nm",
    "pd_c_r_m1_0": "r (correlation)", "pd_c_r_0_p1": "r (correlation)", "pd_c_r_m1_p1": "r (correlation)",
    "pd_b_spearman": "r (Spearman)", "pd_b_tau_vs_offset": "r (Spearman)",
    "pd_b_z": "null SD", "pd_a_contrast_z": "null SD", "pd_a_period_z": "null SD",
    "pd_c_z_m1_0": "null SD", "pd_c_z_0_p1": "null SD", "pd_c_z_m1_p1": "null SD",
    "trace_warnings": "", "pd_warnings": "",
    # v19
    "pd_contour_frac100": "fraction (0-1)", "pd_contour_off_band_fraction": "fraction (0-1)",
    "pd_contour_support_unsupported_fraction": "fraction (0-1)",
    "pd_contour_slice_frac100_central": "fraction (0-1)", "pd_contour_slice_frac100_all": "fraction (0-1)",
    # v20
    "pd_c_leak_r_m1_0_sz47nm": "r (correlation)", "pd_c_leak_r_0_p1_sz47nm": "r (correlation)",
    "pd_c_leak_r_m1_p1_sz47nm": "r (correlation)", "pd_c_leak_r_m1_0_sz77nm": "r (correlation)",
    "pd_c_leak_r_0_p1_sz77nm": "r (correlation)", "pd_c_leak_r_m1_p1_sz77nm": "r (correlation)",
    "pd_c_m1_p1_vs_leak": "",
    # v21
    "pd_c_leak_r_m1_0_sz36nm": "r (correlation)", "pd_c_leak_r_0_p1_sz36nm": "r (correlation)",
    "pd_c_leak_r_m1_p1_sz36nm": "r (correlation)", "pd_c_leak_r_m1_0_sz51nm": "r (correlation)",
    "pd_c_leak_r_0_p1_sz51nm": "r (correlation)", "pd_c_leak_r_m1_p1_sz51nm": "r (correlation)",
}

# --------------------------------------------------------------------------
# v18, package P14: the automatic discard of unanchored clusters
# (tools.mps_unanchored; decisions D1-D11). Origin of the numbers: [pilot] one
# unpublished pilot experiment, [ad hoc] chosen without further support; none
# from the literature.
# --------------------------------------------------------------------------
_EMPTY_UN = (" Empty when the discard was not evaluated (unanchored_status "
             "'not run').")
UNANCHORED: Dict[str, str] = {
    "unanchored_method": "Which discard made the 'discard applied' columns: "
                         "'automatic' (the cluster's depth below a ring the "
                         "spectrin itself draws), 'widefield' (the old "
                         "tubulin + widefield spectrin method), 'off' (only "
                         "decisions by hand) or 'not run'.",
    "unanchored_status": "'ok', 'not run', or why the automatic rule could "
                         "not be evaluated (too few localizations, no polar "
                         "contour, cloud not star-shaped, too few cluster "
                         "centres, smooth fit failed). When it is not 'ok' "
                         "nothing was excluded automatically.",
    "unanchored_failure": "One sentence saying why unanchored_status is not "
                          "'ok'.",
    "unanchored_reference": "The reference ring the depths were measured "
                            "from: 'polar_localizations' (default: the polar "
                            "contour of the slab's localizations) or "
                            "'smooth_centres' (a robust smooth curve through "
                            "the cluster centres).",
    "unanchored_exclude_nm": "Depth below the ring beyond which a cluster is "
                             "excluded automatically. Origin: [pilot + ad "
                             "hoc] 400 nm, PROVISIONAL until the pilot axons "
                             "are labelled by hand." + _EMPTY_UN,
    "unanchored_doubt_nm": "Depth beyond which a cluster is only marked as "
                           "doubtful (never excluded automatically). Origin: "
                           "[ad hoc] 250 nm, the widefield method's margin."
                           + _EMPTY_UN,
    "unanchored_reference_noise_nm": "Median over the clusters of the SD of "
                                     "their depth over 20 subsamples of the "
                                     "ring's input (70 % of the localizations "
                                     "or 90 % of the centres, fixed seed).",
    "unanchored_inner_band_arc_nm": "Arc of the longest dense band deeper "
                                    "than the exclusion depth that runs "
                                    "along the inside of the ring (star-shape "
                                    "test). Above 1500 nm [ad hoc + own sim] "
                                    "the membrane folds inward and the ring "
                                    "is rejected.",
    "unanchored_n_clusters": "Kept clusters the discard looked at.",
    "unanchored_n_auto": "Clusters the automatic rule excludes (deeper than "
                         "unanchored_exclude_nm).",
    "unanchored_n_doubtful": "Clusters the automatic rule marks as doubtful "
                             "(kept).",
    "unanchored_n_borderline": "Clusters whose depth is within 2 noise SDs "
                               "of the exclusion depth.",
    "unanchored_max_depth_nm": "Depth of the deepest cluster below the ring.",
    "unanchored_other_reference": "The other reference ring, evaluated for "
                                  "comparison.",
    "unanchored_other_status": "'ok' or why the other ring could not be used.",
    "unanchored_n_auto_other": "Clusters the automatic rule would exclude "
                               "with the other ring.",
    "unanchored_n_changed_by_reference": "Clusters whose verdict (anchored / "
                                         "doubtful / unanchored) changes "
                                         "with the ring. Empty when either "
                                         "ring could not be used.",
    "unanchored_n_widefield": "Clusters the old widefield method discards, "
                              "when it was run on this axon (empty "
                              "otherwise).",
    "unanchored_n_excluded_manual": "Clusters excluded by hand that the "
                                    "method kept.",
    "unanchored_n_restored_manual": "Clusters the method excluded and a "
                                    "person restored.",
    "unanchored_n_final": "Clusters left out of the 'discard applied' "
                          "analysis (method plus decisions by hand).",
    "unanchored_fraction": "unanchored_n_final / unanchored_n_clusters.",
    "unanchored_gate_exceeded": "True when the automatic rule excludes more "
                                "than 20 % of the clusters [ad hoc, pilot "
                                "maximum 19 %]: check the axon and the ring.",
    "unanchored_discard_source": "Where the final discards came from: "
                                 "'automatic', 'widefield', 'manual', their "
                                 "combination joined by '+', or 'none'.",
    "unanchored_labels_auto": "Labels of the clusters the automatic rule "
                              "excludes, separated by spaces.",
    "unanchored_labels_doubtful": "Labels of the doubtful clusters.",
    "unanchored_labels_excluded_manual": "Labels of the clusters excluded by "
                                         "hand.",
    "unanchored_labels_restored_manual": "Labels of the clusters restored by "
                                         "hand.",
    "unanchored_manual_reasons": "'label: reason' of every decision by hand "
                                 "applied, separated by ' | '.",
    "unanchored_manual_file": "The decisions file beside the data "
                              "(<data>_unanchored.json): decisions are stored "
                              "by position (cluster centre, 50 nm tolerance) "
                              "and survive a re-analysis.",
    "unanchored_manual_fingerprint": "Short hash of the decisions applied "
                                     "(the same decisions give the same "
                                     "hash).",
    "unanchored_manual_n_orphan": "Decisions in the file that match no "
                                  "cluster of this analysis (kept in the "
                                  "file, not applied).",
    "unanchored_warnings": "Warnings of the discard, separated by ' | '.",
    "pd_contour_trim_nm": "Depth below the polar ring beyond which "
                          "localizations were left out before the perimeter "
                          "contour was fitted (off by default: empty). "
                          "Origin: [pilot + ad hoc] 250 nm. Applied only when "
                          "the polar curve passes the strict rules of the "
                          "automatic contour (see pd_contour_trim_status).",
    "pd_contour_n_locs_trimmed": "Localizations that trim left out of the "
                                 "contour fit (the density profiles still "
                                 "count every localization).",
}
UNANCHORED_UNITS: Dict[str, str] = {
    "unanchored_fraction": "fraction (0-1)",
    "unanchored_inner_band_arc_nm": "nm",
}

# The long tables. A name keeps one meaning across them.
PERIMETER_LONG: Dict[str, str] = {
    # profiles
    "slice": "Slice of the row: -1, 0 (central) or +1.",
    "bin": "Index of the arc bin on the central grid (bin j is centred on "
           "origin + j x bin).",
    "lag_nm": "Position along the contour relative to the central slice's "
              "maximum, on the straight axis centred on it (0 at the maximum; "
              "negative one way round, positive the other). In the pooled "
              "table with axis 'fraction' it is a fraction of the perimeter "
              "instead.",
    "count": "Localizations of this slice in this bin.",
    "relative": "count / the largest count of this slice (0-1).",
    "measured": "False where the contour has no data support (a stretch "
                "longer than 400 nm without data, or a bin the band does not "
                "reach): left out of the tests.",
    "tau_window": "Tortuosity (chord / arc, 1 = straight) of the 200 nm "
                  "window that starts at this bin, central slice.",
    "kappa_mean_per_nm": "Signed mean curvature of that window (positive = "
                         "convex, 1/nm).",
    "window_radial_median_nm": "Median signed radial offset (positive "
                               "outside the contour) of the localizations in "
                               "that window, central slice.",
    "radial_median_nm": "Median signed radial offset of this slice's "
                        "localizations in this bin.",
    # contour
    "row": "Row of the contour's point table.",
    "x_nm": "x of the contour point.",
    "y_nm": "y of the contour point.",
    "arc_nm": "Cumulative arc length from the first row, counter-clockwise.",
    # pooled
    "group": "The group of the row: the genotype (or the identity fields "
             "chosen) of the pooled axons, 'SIN ANIMAL' when no axon has an "
             "animal; in the peak - focus table '(all axons)' or 'ROI n'.",
    "axis": "'nm' (absolute distance from the maximum, bins of 100 nm) or "
            "'fraction' (of the perimeter).",
    "mean_relative": "Mean over axons of the folded relative density at this "
                     "distance (one axon, one weight).",
    "n_axons": "Axons contributing at this distance (in the per-animal and "
               "peak - focus tables: axons in the row).",
    "n_animals": "Animals contributing at this distance.",
    "null_lo": "Pointwise 2.5 % of the null band at this distance.",
    "null_hi": "Pointwise 97.5 % of the null band.",
    "global_lo": "Lower edge of the global envelope of the null (Myllymaki "
                 "et al. 2017, [lit]); a profile leaving it is significant "
                 "over all distances at once.",
    "global_hi": "Upper edge of the global envelope.",
    "reading_full": "The reading with its Holm p's, the period peak and the "
                    "scope (axons pooled).",
    "contrast": "Pooled near-far contrast of the group.",
    "contrast_z": "Pooled contrast in null standard deviations.",
    "contrast_p": "Pooled contrast p (two-sided, Monte Carlo).",
    "period": "Pooled phase-locked periodogram power of the group.",
    "period_z": "Pooled period power in null standard deviations.",
    "period_p": "Pooled period p (one-sided, Monte Carlo).",
    "n_axons_pooled": "Axons pooled in the group's statistics.",
    "scope": "Which pooled set the row belongs to: 'reliable contours (slice "
             "misfit included)' (the main set: every finished axon whose "
             "contour passed the quality gate; the misfit between the z "
             "slices is only reported, decision D12), 'reliable contours "
             "without slice misfit' (the same, without the axons the misfit "
             "marked) or 'reliable contours, curve: <kind>' (the main set "
             "restricted to one kind of curve: primary, stiff, polar, drawn; "
             "the kinds are different estimators). v20: 'reliable contours "
             "with coinciding slices (main reading of C)' is the main set "
             "restricted to axons whose slices coincide "
             "(pd_c_main_reading): C(-1, +1) gets its p only there, and A "
             "and B only in the other scopes; elsewhere those columns are "
             "descriptive. Origin: [plan] P10b; decision P-R3-1.",
    # gate summary
    "n_axons_total": "Axons analysed (every file of the run).",
    "n_included": "Axons in this pooled set.",
    "n_excluded": "Axons not in this pooled set (n_axons_total - "
                  "n_included).",
    "n_excluded_analysis_failed": "Axons whose analysis did not finish "
                                  "(pd_status not 'ok'), in any set.",
    "n_excluded_contour_not_reliable": "Finished axons whose contour failed "
                                       "the quality gate for a reason other "
                                       "than the slice misfit.",
    "n_excluded_slice_misfit": "Axons left out ONLY because the slice misfit "
                               "marked them (non-zero only in the set "
                               "without slice misfit).",
    "n_included_with_slice_misfit": "Axons of this set that carry the slice "
                                    "misfit mark (pd_contour_slice_misfit).",
    "included_axons": "trace_label of the axons in this set, separated by "
                      "' | '.",
    "n_excluded_slices_not_coinciding": "Pooled axons left out of the main "
                                        "reading of C because their slices "
                                        "do not coincide (non-zero only in "
                                        "that scope).",
    # by animal
    "animal": "Animal label; 'SIN ANIMAL' for axons without one (descriptive "
              "per ROI, no p); 'provisional-single-animal' is a placeholder "
              "for a folder known to be one animal of unknown label, not an "
              "animal's name.",
    "provisional": "True when the animal label is the provisional placeholder.",
    "n_animals_total": "Animals in the whole analysis; with 1 no comparison "
                       "between animals is possible.",
    "note": "How to read the row (what its p is and is not).",
    # sensitivity
    "setting": "What was varied: 'band (main)' (the main run), 'band' "
               "(another half-width) or 'axial guard'.",
    "value_nm": "The band half-width or the axial guard of the row.",
    "c_leak_expected_r": "Expected neighbour correlation from axial leakage "
                         "alone at this setting (about 0.27 unguarded, about "
                         "0.12 with the 42.5 nm guard).",
    "a_contrast": "As pd_a_contrast at this setting.",
    "a_contrast_z": "As pd_a_contrast_z at this setting.",
    "a_contrast_p": "As pd_a_contrast_p at this setting.",
    "a_period": "Phase-locked period power at this setting.",
    "a_period_z": "As pd_a_period_z at this setting.",
    "a_period_p": "As pd_a_period_p at this setting.",
    "b_spearman": "As pd_b_spearman at this setting (empty for an axial "
                  "guard: only C is repeated).",
    "b_z": "As pd_b_z at this setting.",
    "b_p": "As pd_b_p at this setting.",
    # the batch's axons table and log
    "source_path": "The localization file this axon was analysed from.",
    "file": "The localization file of the log row.",
    "status": "Status of the analysis of that file (as pd_status).",
    "failure": "One sentence saying why the analysis of that file did not "
               "finish; empty when it did.",
    "duplicate_label_with": "Other files that read the same trace_label; a "
                            "label is not a key, so this is a warning.",
    "warnings": "Warnings of the identity and of the analysis, separated by "
                "' | '.",
}
for _suffix in ("m1_0", "0_p1", "m1_p1"):
    PERIMETER_LONG[f"c_r_{_suffix}"] = f"As pd_c_r_{_suffix} at this setting."
    PERIMETER_LONG[f"c_z_{_suffix}"] = f"As pd_c_z_{_suffix} at this setting."
    PERIMETER_LONG[f"c_p_{_suffix}"] = f"As pd_c_p_{_suffix} at this setting."

_ROW_STATS = {
    "n": "How many axons have a value of",
    "mean": "Mean over the axons of",
    "median": "Median over the axons of",
    "sd": "Standard deviation over the axons of",
    "min": "Smallest value of",
    "max": "Largest value of",
    "p": "p of a t test (z against 0) of the axons within one animal, of",
    "p_holm": "That p, Holm-adjusted across the readouts, for",
}
# readouts summarised per animal (tools.mps_perimeter_pool.Z_READOUTS) and per ROI (the four distances)
_ROW_MEASURES = ("pd_a_contrast_z", "pd_a_period_z", "pd_b_z", "pd_c_z_m1_p1", "pd_c_z_m1_0", "pd_c_z_0_p1",
                 "pd_peak_minus_focus_nominal_nm", "pd_abs_peak_minus_focus_nominal_nm",
                 "pd_peak_minus_focus_empirical_nm", "pd_abs_peak_minus_focus_empirical_nm")


def describe_perimeter(name: str) -> Optional[str]:
    """What a column of the perimeter density analysis means (axon table block or long table), or None."""
    if name in PERIMETER:
        return PERIMETER[name]
    if name in PERIMETER_V19:
        return PERIMETER_V19[name]
    if name in PERIMETER_V20:
        return PERIMETER_V20[name]
    if name in PERIMETER_V21:
        return PERIMETER_V21[name]
    if name in PERIMETER_LONG:
        return PERIMETER_LONG[name]
    # n_ / mean_ / sd_ / p_ / p_holm_ / median_ / min_ / max_ + a summarised readout
    for prefix in sorted(_ROW_STATS, key=len, reverse=True):
        if name.startswith(prefix + "_") and name[len(prefix) + 1:] in _ROW_MEASURES:
            measure = name[len(prefix) + 1:]
            base = f"{_ROW_STATS[prefix]} {measure} ({describe_perimeter(measure)})"
            if prefix in ("p", "p_holm"):
                base += (" This is a test within one animal that treats its axons as replicates, NOT a comparison "
                         "between animals, and it does not generalise. It is empty with a single animal or a "
                         "provisional one ('one animal: no p'), in rows 'SIN ANIMAL', and always for "
                         "pd_c_z_m1_0, pd_c_z_0_p1 and pd_a_period_z (not tested: see the note). Holm runs over "
                         "pd_a_contrast_z, pd_b_z and pd_c_z_m1_p1 only.")
            if measure in ("pd_c_z_m1_0", "pd_c_z_0_p1"):
                base += " For the neighbouring pairs a z above 0 is expected from axial leakage alone."
            return base
    return None


def perimeter_unit(name: str) -> Optional[str]:
    """The unit of a perimeter column when it is not the one the name gives (None: use the name's)."""
    if name in PERIMETER_UNITS:
        return PERIMETER_UNITS[name]
    for prefix in sorted(_ROW_STATS, key=len, reverse=True):
        measure = name[len(prefix) + 1:] if name.startswith(prefix + "_") else ""
        if measure in _ROW_MEASURES:
            if prefix == "n":
                return "axons"
            if prefix in ("p", "p_holm"):
                return ""
            return "nm" if measure.endswith("_nm") else "null SD"
    return None


_PANEL = {**AXOPLASM, **AXOPLASM_CLUSTERS, **AXOPLASM_LOCALIZATIONS}


def describe(name: str) -> Optional[str]:
    """What one column means, or None when nothing here describes it."""
    for table in (IDENTITY, STATE, ANALYSIS, CLUSTERS, LOCALIZATIONS,
                  CROSSCHANNEL, Z_CALIBRATION, AXIAL_MIXTURE,
                  EPS_CHOICE, CUT_CLUSTERS, UNANCHORED):
        if name in table:
            return table[name]
    described_pd = describe_perimeter(name)         # v17: the perimeter density analysis
    if described_pd:
        return described_pd
    if name.endswith(DISCARD_SUFFIX):
        base = name[:-len(DISCARD_SUFFIX)]
        described = describe(base)
        if described:
            return (f"{described} Measured without the clusters the "
                    f"axoplasm panel discarded.")
    if name.startswith(AXOPLASM_PREFIX):
        base = name[len(AXOPLASM_PREFIX):]
        if base in _PANEL:
            return f"From the axoplasm panel. {_PANEL[base]}"
    # v13: the clustering integration's block and the comparison tables (tools.clustering_integration,
    # tools.multimethod), described where they are defined
    from tools import clustering_integration
    return clustering_integration.describe_column(name)


def missing(columns: Sequence[str]) -> List[str]:
    """The columns nothing here describes. Empty is the point."""
    return [name for name in columns if not describe(name)]


def rows_for(columns: Sequence[str], table: str) -> List[Dict[str, Any]]:
    """One row per column: where it is, what it is in, what it means."""
    out = []
    for position, name in enumerate(columns, start=1):
        out.append({"table": table, "position": position, "column": name,
                    "unit": unit_of(name),
                    "meaning": describe(name) or ""})
    return out


def write_dictionary(path: str,
                     tables: Mapping[str, Sequence[str]]) -> str:
    """
    Write the dictionary of the tables just exported, beside them.

    ``tables`` maps a table's name to its columns, in the order they are
    written. The file is rewritten each time, since it describes the
    program and not the data.
    """
    base, ext = os.path.splitext(path)
    out = f"{base}_columns{ext or '.csv'}"
    rows: List[Dict[str, Any]] = []
    for name, columns in tables.items():
        rows.extend(rows_for(columns, name))
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["table", "position", "column", "unit",
                                "meaning"])
        writer.writeheader()
        writer.writerows(rows)
    return out


def known_columns() -> List[str]:
    """Every column the axon table itself can hold, in order."""
    columns = list(HEAD_COLUMNS) + list(STATE_COLUMNS)
    columns += list(SHARED_COLUMNS) + list(MEASURED_COLUMNS)
    columns += [name + DISCARD_SUFFIX for name in MEASURED_COLUMNS]
    columns += [AXOPLASM_PREFIX + name for name in AXOPLASM]
    columns += list(Z_CALIBRATION_COLUMNS)
    columns += list(AXIAL_MIXTURE_COLUMNS)
    columns += list(EPS_CHOICE_COLUMNS)
    columns += list(CUT_COLUMNS)
    columns += list(CLUSTER_METHOD_COLUMNS)
    columns += list(EMITTERS_COLUMNS)
    from tools.axon_export import integration_columns
    columns += list(integration_columns())          # v13 (v14: three more at its end)
    columns += list(PERIMETER_COLUMNS)              # v17: the perimeter density analysis
    columns += list(UNANCHORED_COLUMNS)             # v18: the unanchored-cluster discard
    columns += list(PERIMETER_V19_COLUMNS)          # v19: contour assessment and pooling gate
    columns += list(PERIMETER_V20_COLUMNS)          # v20: slice criterion of C, leak with real fractions
    columns += list(PERIMETER_V21_COLUMNS)          # v21: leak at the ends of the measured range of sigma_z, last
    return columns
