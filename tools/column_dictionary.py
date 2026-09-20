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
    AXOPLASM_PREFIX, DISCARD_SUFFIX, HEAD_COLUMNS, MEASURED_COLUMNS,
    SHARED_COLUMNS, STATE_COLUMNS,
)
from tools.mps_identity import FIELD_HINTS, FIELDS as IDENTITY_FIELDS

# Units, read off the end of the name. The rule is the convention: a
# quantity says its unit in its own name, and nothing carries a unit it
# does not say.
UNITS = (
    ("_nm2", "nm^2"), ("_um2", "um^2"), ("_nm", "nm"), ("_um", "um"),
    ("_px", "pixels"), ("_percent", "%"), ("_deg", "degrees"),
)


def unit_of(name: str) -> str:
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
                         "had to be to be discarded.",
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
                         "sidecar, the file itself, or typed by hand.",
    "eps_nm": "DBSCAN's neighbourhood radius.",
    "min_samples": "DBSCAN's minimum number of localizations for a cluster.",
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
    # --- what the discard changes
    "n_clusters_kept": "Clusters this analysis measured.",
    "perimeter_um": "Length of the contour through the cluster centres.",
    "clusters_per_um": "Clusters per micrometre of that contour.",
    "contour_2opt": "Whether the tour was refined from one start or from "
                    "every start.",
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
                    "randomized one (KS D).",
    "ks_cdf_crossing": "Whether those two distributions cross, which makes "
                       "D hard to read on its own.",
    "randomization_n_iterations": "Randomized rings actually built.",
    "randomization_min_sep_nm": "Closest two randomized centres were allowed "
                                "to be.",
    "randomization_incomplete_iters": "Randomized rings that could not place "
                                      "every cluster.",
    "randomized_median_1nn_nm": "Median 1NN of the randomized rings.",
    "ks_pvalue": "p-value of that KS test. With 94 measured against 94,000 "
                 "randomized distances, read D, not this.",
    "n_warnings": "How many warnings this analysis raised.",
    "warnings": "Their text, separated by ' | '.",
}

AXOPLASM: Dict[str, str] = {
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
                           "not placed at all.",
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
    "selection_z_source": "Whether that range was typed or came from the "
                          "axial peak.",
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

_PANEL = {**AXOPLASM, **AXOPLASM_CLUSTERS, **AXOPLASM_LOCALIZATIONS}


def describe(name: str) -> Optional[str]:
    """What one column means, or None when nothing here describes it."""
    for table in (IDENTITY, STATE, ANALYSIS, CLUSTERS, LOCALIZATIONS):
        if name in table:
            return table[name]
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
    return None


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
    return columns
