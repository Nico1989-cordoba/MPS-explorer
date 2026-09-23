# -*- coding: utf-8 -*-
"""
End-to-end orchestration of the Gazal et al. (2026) per-axon analysis.

Chains the individual modules into a single call that takes the raw
localizations of one loaded ROI and returns every parameter implemented so
far, together with the provenance and warnings the GUI must surface.

    step 2  tools.mps_periodicity   Delta-Z + automatic 180 nm axial slab
    step 1  tools.cluster_quality   automatic bad-cluster removal
    step 3  tools.mps_geometry      perimeter, cluster area, effective radius
    step 4  tools.mps_spatial       1NN spacing between cluster centres

    step 5  tools.mps_occupancy     perimeter occupancy (Mahalanobis)
    step 6  tools.mps_randomization  constrained random control + KS test

All eight per-axon parameters of the paper are now covered. Any that cannot
be computed for a given axon (too few surviving clusters, for instance) come
back as None and are shown as "n/a" rather than as zero.

The contour is built with 2-opt from every start, keeping the shortest
tour, whose result does not depend on the cluster the tour starts from.
Before 2026-09-19 it was refined from one start; ``all_starts=False``
still does that, and the exported ``contour_2opt`` column says which.

``without_clusters`` re-runs steps 3-6 on a subset of the clusters: the
ones the axoplasm panel keeps after discarding those both widefield images
place inside the axon. Everything before the clusters are fixed -- the
axial slab, DBSCAN, the automatic curation -- is shared, so the analysis
of all the clusters and this one differ only by the clusters left out,
and the user can take either one to the statistics.

Beyond the paper, the analysis reports the centre of the contour: its
area centroid (tools.mps_geometry.contour_centre).

This module is deliberately free of any Qt dependency so the whole pipeline
can be run and tested headlessly (see validate_full_18axons.py).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from tools.cluster_quality import (
    BadClusterReport,
    PolygonROI,
    ROIShape,
    describe_roi,
    good_cluster_centroids,
    good_cluster_labels,
    identify_bad_clusters,
)
from tools.mps_geometry import (
    PAPER_INTERCEPT_CLUSTERS,
    PAPER_MEDIAN_CLUSTER_AREA_NM2,
    PAPER_MEDIAN_R_EFF_NM,
    PAPER_SLOPE_CLUSTERS_PER_UM,
    DEEP_VERTEX_FRACTION,
    MAX_OVER_MEDIAN_LIMIT,
    ClusterAreaResult,
    ContourCentre,
    ContourHealth,
    PerimeterResult,
    compute_cluster_areas,
    reconstruct_perimeter,
)
from tools.mps_periodicity import (
    DEFAULT_SLAB_HALF_WIDTH_NM,
    ZPeriodicityResult,
    fit_z_periodicity,
    select_mps_slab,
)
from tools.mps_randomization import (
    DEFAULT_N_RANDOMIZATIONS,
    PAPER_KS_D,
    RandomizationResult,
    randomize_cluster_positions,
)
from tools.mps_occupancy import (
    PAPER_OCCUPANCY_PERCENT,
    OccupancyResult,
    compute_occupancy,
)
from tools.mps_spatial import (
    PAPER_MEDIAN_OF_MEDIANS_NM,
    NNResult,
    compute_nn_distances,
)
from tools.mps_settings import DEFAULT_DBCV_THRESHOLD
from tools.results_table import cell_text

# Paper defaults (Gazal et al. 2026)
DEFAULT_EPS_NM = 25.0
DEFAULT_MIN_SAMPLES = 10

# The exported columns that say which analysis a row is. Two rows that
# differ in one of them describe the same axon two ways, and must go to
# separate tables (tools.results_table.refuse_other_analysis).
ANALYSIS_COLUMNS = ("cluster_set", "contour_2opt")

# The columns that say WHICH selection a row describes. Two rows that
# agree on them are one axon exported twice, which no table should gain
# without the user being told (tools.export_ui).
AXON_KEY_COLUMNS = ("source", "roi")

# Reference values for the "vs paper" column of the results panel.
# (label, value, unit, tolerance_fraction) -- tolerance is only used to
# colour the comparison, never to alter a computed number.
PAPER_REFERENCE: Dict[str, Tuple[str, float, str]] = {
    "delta_z_nm": ("Axial periodicity Delta-Z", 170.0, "nm"),
    "median_area_nm2": ("Cluster area (median)", PAPER_MEDIAN_CLUSTER_AREA_NM2, "nm^2"),
    "median_r_eff_nm": ("Effective radius (median)", PAPER_MEDIAN_R_EFF_NM, "nm"),
    "median_1nn_nm": ("1NN spacing (median)", PAPER_MEDIAN_OF_MEDIANS_NM, "nm"),
    "clusters_per_um": ("Clusters per um", PAPER_SLOPE_CLUSTERS_PER_UM, "1/um"),
}


def _joined(values: Any, digits: int) -> str:
    """A list of numbers in one cell, separated by "|".

    Not by ";": the canonical CSV is comma-separated, and a ";" inside a
    cell is what Excel splits on where the decimal mark is a comma, which
    turns one row into several columns.
    """
    return "|".join(f"{float(v):.{digits}f}" for v in np.asarray(values).ravel())


@dataclass
class AxonAnalysis:
    """Every per-axon parameter produced by one run of the pipeline."""

    # --- provenance -----------------------------------------------------
    source_name: str
    pixel_size_nm: Optional[float]
    pixel_size_source: str          # see PIXEL_SIZE_SOURCES
    eps_nm: float
    min_samples: int
    slab_half_width_nm: float
    dbcv_threshold: float

    # --- step 2: axial ---------------------------------------------------
    z_result: ZPeriodicityResult
    slab_zmin_nm: float
    slab_zmax_nm: float
    n_locs_total: int
    n_locs_slab: int

    # --- step 1: clustering + automatic curation -------------------------
    labels: NDArray[np.int64]
    bad_report: BadClusterReport
    n_clusters_raw: int
    n_clusters_kept: int
    centroids: NDArray[np.float64]

    # --- coordinates of the analysed slab (for plotting) -----------------
    x_slab: NDArray[np.float64]
    y_slab: NDArray[np.float64]
    z_slab: NDArray[np.float64]

    # --- step 3 / step 4 --------------------------------------------------
    perimeter: Optional[PerimeterResult]
    areas: Optional[ClusterAreaResult]
    nn: Optional[NNResult]

    # --- step 5: perimeter occupancy (parameter 6) ------------------------
    occupancy: Optional[OccupancyResult] = None

    # --- step 6: randomization control (parameter 8) ----------------------
    randomization: Optional[RandomizationResult] = None

    warnings: List[str] = field(default_factory=list)

    # --- what steps 3-6 ran with, so without_clusters can repeat them ----
    mahalanobis_threshold: float = 3.0
    ellipse_mode: str = "clip"
    run_randomization: bool = True
    n_randomizations: int = DEFAULT_N_RANDOMIZATIONS
    random_seed: int = 0
    # The warnings raised before the clusters were fixed (pixel size, axial
    # slab, curation): an analysis of a subset of the clusters shares them.
    upstream_warnings: List[str] = field(default_factory=list)

    # --- which selection this is -----------------------------------------
    # The ROI the localizations were taken from. Rows of two axons picked
    # from one whole-field file differ in nothing else, so without it an
    # exported row cannot be attributed to an axon.
    roi: Optional[ROIShape] = None
    # What the edge criterion measured against: the drawn ROI, or the
    # convex hull of the analysed localizations when none was given (the
    # headless batch). The two curate differently -- on axon 7, 94 kept
    # clusters against 90 -- so a table pooling both must be able to say
    # which is which.
    edge_reference: str = "none"
    # "automatic", "peak chosen" or "range typed": how the axial slab was
    # decided. The table's note says "auto from GMM main peak" whatever
    # happened, so without this column a slab picked by hand -- another
    # ring of the same axon -- cannot be told from the automatic one.
    slab_source: str = "automatic"
    # Where each localization of the slab sits in the ones the analysis was
    # given, so a per-localization table can be joined back to the file, and
    # so that a localization left out of the slab can be told from one the
    # clustering called noise. Empty when the caller did not record it.
    slab_index: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(0, dtype=np.intp))

    # --- set by without_clusters -----------------------------------------
    # The margin the axoplasm panel discarded clusters with (None: nothing
    # was discarded, this is the analysis of every kept cluster), and the
    # DBSCAN labels of the clusters left out.
    discard_margin_nm: Optional[float] = None
    discarded_labels: FrozenSet[int] = frozenset()
    # How the widefield images were placed when those clusters were found
    # ("measured, score 13.5", "set by hand", ...). The discard is only as
    # good as that placement, and this row is where a reader of the table
    # can see it; the panel's own table carries the same column.
    discard_registration: str = ""

    # ---------------- convenience accessors for the panel ----------------

    @property
    def discard_applied(self) -> bool:
        return self.discard_margin_nm is not None

    @property
    def cluster_set(self) -> str:
        """Which clusters the numbers describe, as the exported column."""
        return "discard applied" if self.discard_applied else "all clusters"

    @property
    def n_clusters_discarded(self) -> Optional[int]:
        return len(self.discarded_labels) if self.discard_applied else None

    @property
    def contour_2opt(self) -> Optional[str]:
        """"one start" or "every start"; None without a contour, and None
        for an order set by hand, where no 2-opt ran at all.

        It used to say "one start" for a hand-set order, because it read
        n_starts and a hand-set order leaves it at its default. That is
        a measured-sounding answer to a question that was never asked.
        """
        if self.perimeter is None:
            return None
        if self.perimeter.order_source != "automatic":
            return None
        return "every start" if self.perimeter.n_starts > 1 else "one start"

    @property
    def contour_order_source(self) -> Optional[str]:
        """Whether the tour through the centres is the one the program
        built or one a person set. None without a contour."""
        if self.perimeter is None:
            return None
        return self.perimeter.order_source

    @property
    def contour_guide(self) -> Optional[NDArray[np.float64]]:
        """The path a person drew along the membrane, when the contour was
        ordered along one; None when the program ordered it."""
        return None if self.perimeter is None else self.perimeter.guide

    @property
    def mean_delta_z_nm(self) -> Optional[float]:
        return self.z_result.mean_delta_z_nm

    @property
    def perimeter_um(self) -> Optional[float]:
        return self.perimeter.perimeter_um if self.perimeter else None

    @property
    def clusters_per_um(self) -> Optional[float]:
        return self.perimeter.clusters_per_um if self.perimeter else None

    @property
    def contour_health(self) -> Optional[ContourHealth]:
        return self.perimeter.health if self.perimeter else None

    @property
    def centre(self) -> Optional[ContourCentre]:
        """The area centroid of the contour. None without one, and when it
        crosses itself, encloses no area or has a non-finite vertex."""
        return self.perimeter.centre if self.perimeter else None

    @property
    def contour_tour_over_hull(self) -> Optional[float]:
        """Exposed on its own so a batch can compare it between groups.

        If the contours of one genotype are more inflated than those of
        the other, every perimeter-derived difference between the groups
        is confounded, and the only way to notice is to compare this.
        """
        health = self.contour_health
        return health.tour_over_hull if health else None

    @property
    def median_area_nm2(self) -> Optional[float]:
        return self.areas.median_area_nm2 if self.areas else None

    @property
    def median_r_eff_nm(self) -> Optional[float]:
        return self.areas.median_r_eff_nm if self.areas else None

    @property
    def median_1nn_nm(self) -> Optional[float]:
        return self.nn.median_1nn_nm if self.nn else None

    @property
    def occupancy_percent(self) -> Optional[float]:
        return self.occupancy.occupancy_percent if self.occupancy else None

    @property
    def ks_statistic(self) -> Optional[float]:
        return self.randomization.ks_statistic if self.randomization else None

    @property
    def ks_pvalue(self) -> Optional[float]:
        return self.randomization.ks_pvalue if self.randomization else None

    def summary_rows(self) -> List[Tuple[str, str, str, str]]:
        """
        Rows for the results table: (parameter, measured, paper, note).

        Values that could not be computed are reported as such rather than
        as zero, and parameters that are not implemented yet are labelled
        "not implemented" so the panel can never be mistaken for a complete
        replication of the paper.
        """
        def fmt(v: Optional[float], nd: int = 1) -> str:
            return "n/a" if v is None else f"{v:,.{nd}f}"

        kept = f"{self.n_clusters_kept}"
        if self.discarded_labels:
            kept += f" ({len(self.discarded_labels)} discarded)"
        starts = 1 if self.perimeter is None else self.perimeter.n_starts
        joined = ("centroids connected, 2-opt" if starts == 1 else
                  f"centroids connected, 2-opt from all {starts} starts")
        if (self.perimeter is not None
                and self.perimeter.order_source != "automatic"):
            joined = f"centroids connected in the order {self.perimeter.order_source}"
            if self.perimeter.guide is not None:
                joined = ("centroids connected along the path drawn by hand "
                          f"({len(self.perimeter.guide)} points)")
        centre = self.centre
        if centre is not None:
            where = f"x {centre.x_nm:.0f}, y {centre.y_nm:.0f} nm"
            where_note = (f"area centroid of the contour, which encloses "
                          f"{centre.area_um2:.2f} um^2; same frame as the "
                          f"localizations")
        else:
            where = "n/a"
            if self.perimeter is None:
                where_note = ""
            elif self.perimeter.self_intersections_after > 0:
                where_note = "the contour crosses itself"
            else:
                where_note = ("the contour encloses no area, or a cluster "
                              "centre is not a finite coordinate")
        shift = None if centre is None else centre.max_shift_nm

        rows: List[Tuple[str, str, str, str]] = [
            ("Localizations (ROI)", f"{self.n_locs_total:,}", "-", ""),
            ("Localizations (180 nm slab)", f"{self.n_locs_slab:,}", "-", ""),
            ("Axial slab z-range",
             f"{self.slab_zmin_nm:,.0f} .. {self.slab_zmax_nm:,.0f} nm", "-",
             "auto from GMM main peak"),
            ("Delta-Z (axial periodicity)", fmt(self.mean_delta_z_nm, 1),
             "170 +/- 15 nm", f"{len(self.z_result.delta_z_nm)} interval(s)"),
            ("Clusters detected (raw)", f"{self.n_clusters_raw}", "-", ""),
            ("Clusters kept (auto-curated)", kept, "-",
             f"{len(self.bad_report.bad_labels)} removed"
             + ("" if not self.discard_applied else
                f"; discarded: inside both widefield images by more than "
                f"{self.discard_margin_nm:,.0f} nm")),
            ("Perimeter", fmt(self.perimeter_um, 2) + " um", "-", joined),
            # These checks and their limits are this program's, not the
            # paper's: the paper column stays "-".
            ("  centres deep inside the hull",
             "n/a" if self.contour_health is None
             else f"{self.contour_health.n_deep_vertices}",
             "-",
             "" if self.contour_health is None
             else f"this program flags any deeper than "
                  f"{self.contour_health.depth_limit_nm:,.0f} nm "
                  f"({DEEP_VERTEX_FRACTION:.0%} of the hull radius); "
                  f"deepest {self.contour_health.max_depth_nm:,.0f} nm"),
            ("  contour / its convex hull",
             "n/a" if self.contour_health is None
             else f"{self.contour_health.tour_over_hull:.2f}",
             "-",
             "" if self.contour_health is None
             else f"hull {self.contour_health.hull_perimeter_um:.2f} um; "
                  f"context only, it has no threshold"),
            ("  longest step / median",
             "n/a" if self.contour_health is None
             else f"{self.contour_health.max_over_median:.1f}",
             "-",
             "" if self.contour_health is None
             else f"this program flags above "
                  f"{MAX_OVER_MEDIAN_LIMIT:.0f}; "
                  f"{self.contour_health.edge_max_nm:,.0f} nm against "
                  f"{self.contour_health.edge_median_nm:,.0f} nm"),
            ("Centre", where, "-", where_note),
            ("  moves with one cluster out",
             "n/a" if shift is None else f"{shift:,.0f} nm", "-",
             "" if shift is None
             else "the most it moves, leaving each cluster of the contour "
                  "out in turn"),
            ("Clusters per um", fmt(self.clusters_per_um, 2), "4.08", ""),
            ("Cluster area (median)", fmt(self.median_area_nm2, 0) + " nm^2",
             "1,965 nm^2", "convex hull"),
            ("Effective radius (median)", fmt(self.median_r_eff_nm, 1) + " nm",
             "~25 nm", "sqrt(A/pi)"),
            ("1NN spacing (median)", fmt(self.median_1nn_nm, 1) + " nm",
             "260 nm", "euclidean, centre-to-centre"),
            ("Perimeter occupancy",
             "n/a" if self.occupancy is None
             else f"{self.occupancy.occupancy_percent:.1f} %",
             "~20 %",
             "" if self.occupancy is None
             else f"Mahalanobis < {self.occupancy.mahalanobis_threshold:g}, "
                  f"{self.occupancy.n_points:,} pts"),
            ("Randomization KS test",
             "n/a" if self.randomization is None
             else f"D={self.randomization.ks_statistic:.3f}, "
                  f"p={self.randomization.ks_pvalue:.1e}",
             "D=0.054, p<1e-3",
             "" if self.randomization is None
             else f"{self.randomization.n_iterations:,} randomizations, "
                  f"min sep {self.randomization.min_distance_nm:.0f} nm"),
            ("  CDF crossing",
             "no crossing" if (self.randomization is None
                               or self.randomization.cdf_crossing is None)
             else f"{self.randomization.cdf_crossing:.2f}",
             "~0.80", "experimental vs randomized"),
        ]
        return rows

    def export_dict(self) -> Dict[str, Any]:
        """Flat record for CSV export -- one row per analysed axon.

        The pixel size and its provenance are included deliberately: a
        miscalibrated pixel size silently rescales every distance here, so
        an exported row must carry enough information to detect that later.
        """
        return {
            "source": self.source_name,
            # Which axon of the file: rows of two ROIs drawn on one
            # whole-field image are identical in every other column, so
            # without this they cannot be told apart or joined to the
            # panels' own tables.
            "roi": describe_roi(self.roi),
            # "all clusters" or "discard applied": the two analyses of one
            # axon are two rows, and a statistic must not pool them.
            "cluster_set": self.cluster_set,
            "pixel_size_nm": self.pixel_size_nm,
            "pixel_size_source": self.pixel_size_source,
            "eps_nm": self.eps_nm,
            "min_samples": self.min_samples,
            "dbcv_threshold": self.dbcv_threshold,
            # What the edge criterion measured against: the drawn ROI, or
            # the convex hull of the localizations when none was drawn, as
            # in the headless batch. They curate differently, so rows of
            # the two must be separable in a pooled table.
            "edge_reference": self.edge_reference,
            "slab_half_width_nm": self.slab_half_width_nm,
            "slab_zmin_nm": round(self.slab_zmin_nm, 2),
            "slab_zmax_nm": round(self.slab_zmax_nm, 2),
            # Whether that slab is the automatic one, and what the
            # automatic one would have been: a row measured on another ring
            # of the same axon is otherwise indistinguishable.
            "slab_source": self.slab_source,
            "z_main_peak_auto_nm": round(self.z_result.main_peak_nm, 2),
            "n_locs_total": self.n_locs_total,
            "n_locs_slab": self.n_locs_slab,
            "gmm_n_components": self.z_result.n_components,
            # The mixture the slab was chosen from. Its weights are what
            # "ambiguous main peak" refers to, and the widths say whether
            # the components are separated at all; both were on screen
            # only. Values are separated by "|", never by ";", which is
            # the field separator Excel uses where the decimal mark is a
            # comma.
            "gmm_means_nm": _joined(self.z_result.means_nm, 1),
            "gmm_weights": _joined(self.z_result.weights, 3),
            "gmm_sigmas_nm": _joined(self.z_result.sigmas_nm, 1),
            "gmm_converged": bool(self.z_result.converged),
            "gmm_n_discarded_components":
                self.z_result.n_discarded_components,
            "delta_z_mean_nm": self.mean_delta_z_nm,
            "delta_z_values_nm": _joined(self.z_result.delta_z_nm, 1),
            "n_clusters_raw": self.n_clusters_raw,
            "n_clusters_kept": self.n_clusters_kept,
            "n_clusters_removed": len(self.bad_report.bad_labels),
            "removed_edge_touching": len(self.bad_report.edge_touching),
            "removed_low_dbcv": len(self.bad_report.low_dbcv),
            "edge_criterion_disabled": bool(
                self.bad_report.edge_criterion_disabled),
            "n_clusters_discarded": self.n_clusters_discarded,
            "discard_margin_nm": self.discard_margin_nm,
            # How the widefield images were placed when the discarded
            # clusters were found: at no shift the same axon gives another
            # set of them, and nothing in this row used to say so.
            "discard_registration": self.discard_registration or None,
            "perimeter_um": self.perimeter_um,
            "clusters_per_um": self.clusters_per_um,
            # How the tour was refined: 2-opt from one start (the legacy
            # behaviour) or the shortest over every start. Tables of the
            # two must not be pooled either.
            "contour_2opt": self.contour_2opt,
            # Where the order of the tour came from: the program, or a
            # person. Empty 2opt with a source of "set by hand" is the
            # pair that says no 2-opt ran.
            "contour_order_source": self.contour_order_source,
            "contour_guide_nm": format_guide(self.contour_guide),
            # Contour health: the perimeter above is only a perimeter if
            # these say so, and a reader of the CSV cannot tell otherwise.
            "contour_hull_um": (
                None if self.contour_health is None
                else round(self.contour_health.hull_perimeter_um, 3)),
            "contour_tour_over_hull": (
                None if self.contour_health is None
                else round(self.contour_health.tour_over_hull, 3)),
            "contour_max_over_median": (
                None if self.contour_health is None
                else round(self.contour_health.max_over_median, 2)),
            "contour_n_deep_vertices": (
                None if self.contour_health is None
                else self.contour_health.n_deep_vertices),
            # What "deep" meant here: the check is a fraction of this
            # axon's own hull radius, so the count alone cannot be read.
            "contour_hull_radius_nm": (
                None if self.contour_health is None
                else round(self.contour_health.hull_radius_nm, 1)),
            "contour_deep_limit_nm": (
                None if self.contour_health is None
                else round(self.contour_health.depth_limit_nm, 1)),
            "contour_max_depth_nm": (
                None if self.contour_health is None
                else round(self.contour_health.max_depth_nm, 1)),
            "contour_length_in_long_edges": (
                None if self.contour_health is None
                else round(self.contour_health.length_in_long_edges, 3)),
            # The area centroid of the contour, in the localizations' own
            # frame, the area it encloses, and the most it moves when one
            # cluster is left out of it.
            "centre_x_nm": (None if self.centre is None
                            else round(self.centre.x_nm, 1)),
            "centre_y_nm": (None if self.centre is None
                            else round(self.centre.y_nm, 1)),
            "contour_area_um2": (None if self.centre is None
                                 else round(self.centre.area_um2, 4)),
            "centre_max_shift_nm": (
                None if self.centre is None
                or self.centre.max_shift_nm is None
                else round(self.centre.max_shift_nm, 1)),
            "median_area_nm2": self.median_area_nm2,
            "median_r_eff_nm": self.median_r_eff_nm,
            "median_1nn_nm": self.median_1nn_nm,
            # The threshold the occupancy was measured with. It decides
            # the number outright (0.1 instead of 3 turned 46.5 % into
            # 1.5 % on axon 7) and was in no column.
            "mahalanobis_threshold": self.mahalanobis_threshold,
            "ellipse_mode": self.ellipse_mode,
            "occupancy_percent": self.occupancy_percent,
            "occupied_length_nm": (
                None if self.occupancy is None
                else round(self.occupancy.occupied_length_nm, 1)),
            "occupancy_n_points": (
                None if self.occupancy is None else self.occupancy.n_points),
            "occupancy_n_capped_ellipses": (
                None if self.occupancy is None else self.occupancy.n_capped),
            "ks_statistic": self.ks_statistic,
            "ks_cdf_crossing": (
                None if self.randomization is None
                else self.randomization.cdf_crossing),
            "randomization_requested": (
                int(self.n_randomizations) if self.run_randomization else 0),
            "random_seed": self.random_seed,
            "randomization_n_iterations": (
                None if self.randomization is None
                else self.randomization.n_iterations),
            "randomization_min_sep_nm": (
                None if self.randomization is None
                else round(self.randomization.min_distance_nm, 1)),
            "randomization_incomplete_iters": (
                None if self.randomization is None
                else self.randomization.n_incomplete_iterations),
            "randomized_median_1nn_nm": (
                None if self.randomization is None
                else self.randomization.randomized_median_nm),
            "ks_pvalue": self.ks_pvalue,
            "n_warnings": len(self.warnings),
            "warnings": " | ".join(cell_text(w) for w in self.warnings),
        }


# Where a pixel size came from, as tools.mps_io records it: the Picasso
# sidecar ("yaml"), the metadata embedded in the HDF5 since Picasso 0.11
# ("hdf5"), a line scan of a sidecar that would not parse ("yaml_scan"), a
# value given in code ("override") or typed by the user ("manual"), nothing
# ("unknown"), or a file already in nanometres ("not_applicable").
# "neighbour": read from the acquisition metadata of a DIFFERENT file
# near this one, because this one carries none. "remembered": a value a
# person typed for another file in the same folder. Both are inferences
# about which acquisition a file belongs to, not records of it, so both
# warn exactly like a typed value does.
def format_guide(guide: Optional[NDArray[np.float64]]) -> Optional[str]:
    """A drawn path as one table cell: "x,y;x,y;..." in nm, to 0.1 nm.

    Written so the contour can be rebuilt from the table alone -- a
    contour set by hand that cannot be reproduced is a number nobody can
    check.
    """
    if guide is None:
        return None
    path = np.asarray(guide, dtype=float).reshape(-1, 2)
    return ";".join(f"{x:.1f},{y:.1f}" for x, y in path)


def parse_guide(text: Optional[str]) -> Optional[NDArray[np.float64]]:
    """The inverse of format_guide; None for an empty cell."""
    if text is None or not str(text).strip():
        return None
    rows = [tuple(float(v) for v in pair.split(","))
            for pair in str(text).strip().split(";") if pair.strip()]
    return np.asarray(rows, dtype=float).reshape(-1, 2)


PIXEL_SIZE_SOURCES = ("yaml", "hdf5", "yaml_scan", "override", "manual",
                      "neighbour", "remembered", "unknown", "not_applicable")
GUESSED_PIXEL_SIZE_SOURCES = ("override", "manual", "neighbour",
                              "remembered", "unknown")


def analyze_axon(
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: NDArray[np.float64],
    *,
    source_name: str = "",
    pixel_size_nm: Optional[float] = None,
    pixel_size_source: str = "unknown",
    eps_nm: float = DEFAULT_EPS_NM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    slab_half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM,
    roi: Optional[ROIShape] = None,
    main_peak_override_nm: Optional[float] = None,
    slab_override: Optional[Tuple[float, float]] = None,
    custom_contour_order: Optional[NDArray[np.intp]] = None,
    dbcv_threshold: float = DEFAULT_DBCV_THRESHOLD,
    mahalanobis_threshold: float = 3.0,
    ellipse_mode: str = "clip",
    run_randomization: bool = True,
    n_randomizations: int = DEFAULT_N_RANDOMIZATIONS,
    random_seed: int = 0,
    all_starts: bool = True,
    contour_guide: Optional[NDArray[np.float64]] = None,
) -> AxonAnalysis:
    """
    Run the full per-axon pipeline on one ROI's localizations.

    Coordinates must already be in nanometres (the GUI converts x,y from
    camera pixels on load; Picasso's z is already in nm and must NOT be
    scaled).

    Overrides -- every automatic choice can be replaced by the user
    --------------------------------------------------------------------
    main_peak_override_nm : centre the axial slab on this z instead of the
        GMM's main peak. This is the hook for the peak selector, used when
        the axial distribution has more than one comparable peak.
    slab_override : explicit (zmin, zmax) in nm, bypassing the automatic
        slab entirely.
    custom_contour_order : explicit ordering of the cluster centroids for
        the perimeter, for axons where the automatic contour is wrong.
    contour_guide : a closed path drawn along the membrane, (M, 2) in nm.
        The centroids are joined in the order they fall along it. This is
        how the GUI keeps a contour set by hand: unlike an order of
        cluster indices, it still means something when eps, min samples
        or the slab change the clusters, and it orders the clusters the
        axoplasm discard leaves as well. Exclusive with
        ``custom_contour_order``.

    all_starts : build the contour with 2-opt from every start and keep
        the shortest tour (default). False refines it from one start, as
        this program did before 2026-09-19; on the 18 April axons that
        tour came out longer in 13, by up to 7.1 %.

    Returns
    -------
    AxonAnalysis -- always returned, even when the axon is too sparse to
    yield a perimeter; in that case the corresponding fields are None and
    the reason is in ``warnings``.
    """
    x_nm = np.asarray(x_nm, dtype=float).ravel()
    y_nm = np.asarray(y_nm, dtype=float).ravel()
    z_nm = np.asarray(z_nm, dtype=float).ravel()

    warnings_: List[str] = []
    if pixel_size_source not in PIXEL_SIZE_SOURCES:
        raise ValueError(f"bad pixel_size_source: {pixel_size_source!r}")
    if pixel_size_source in GUESSED_PIXEL_SIZE_SOURCES:
        # "not_applicable" means the file was already in nanometres
        # (ThunderSTORM / custom CSV), so there is nothing to warn about.
        warnings_.append(
            f"Pixel size ({pixel_size_nm} nm) did not come from the Picasso "
            f"metadata (source: {pixel_size_source}). Every lateral "
            f"distance and, squared, every cluster area scales with it -- "
            f"verify it before using these numbers."
        )

    # ---------------- step 2: axial periodicity and slab ----------------
    z_result = fit_z_periodicity(z_nm)
    warnings_.extend(z_result.warnings)

    if slab_override is not None:
        slab_source = "range typed"
        zmin, zmax = float(slab_override[0]), float(slab_override[1])
        slab_mask = (z_nm >= zmin) & (z_nm <= zmax)
        warnings_.append(
            f"Axial slab set manually to {zmin:.0f} .. {zmax:.0f} nm "
            f"(automatic choice was "
            f"{z_result.main_peak_nm - slab_half_width_nm:.0f} .. "
            f"{z_result.main_peak_nm + slab_half_width_nm:.0f} nm)."
        )
    else:
        peak = (float(main_peak_override_nm)
                if main_peak_override_nm is not None
                else z_result.main_peak_nm)
        # The results window sends back the peak it is showing on every
        # re-run, so a peak equal to the automatic one is not a choice:
        # warning about it added a warning to analyses nobody had steered,
        # and made the same axon export n_warnings 4 or 5 for no reason.
        chosen = (main_peak_override_nm is not None
                  and abs(peak - z_result.main_peak_nm) > 0.05)
        slab_source = "peak chosen" if chosen else "automatic"
        if chosen:
            warnings_.append(
                f"Axial peak selected manually at z = {peak:.0f} nm "
                f"(automatic choice was {z_result.main_peak_nm:.0f} nm)."
            )
        slab_mask = select_mps_slab(z_nm, peak, slab_half_width_nm)
        zmin, zmax = peak - slab_half_width_nm, peak + slab_half_width_nm

    xs, ys, zs = x_nm[slab_mask], y_nm[slab_mask], z_nm[slab_mask]

    base: Dict[str, Any] = dict(
        source_name=source_name,
        pixel_size_nm=pixel_size_nm,
        pixel_size_source=pixel_size_source,
        eps_nm=eps_nm,
        min_samples=min_samples,
        slab_half_width_nm=slab_half_width_nm,
        dbcv_threshold=dbcv_threshold,
        z_result=z_result,
        slab_zmin_nm=zmin,
        slab_zmax_nm=zmax,
        n_locs_total=int(x_nm.size),
        n_locs_slab=int(xs.size),
        x_slab=xs, y_slab=ys, z_slab=zs,
        slab_index=np.flatnonzero(slab_mask).astype(np.intp),
        roi=roi,
        slab_source=slab_source,
    )

    settings: Dict[str, Any] = dict(
        mahalanobis_threshold=mahalanobis_threshold,
        ellipse_mode=ellipse_mode,
        run_randomization=run_randomization,
        n_randomizations=n_randomizations,
        random_seed=random_seed,
    )

    if xs.size < min_samples:
        warnings_.append(
            f"Only {xs.size} localizations inside the axial slab "
            f"(min_samples = {min_samples}): cannot cluster this ROI."
        )
        empty = np.array([], dtype=np.int64)
        return AxonAnalysis(
            labels=empty,
            bad_report=BadClusterReport(set(), set(), set(), {}),
            n_clusters_raw=0, n_clusters_kept=0,
            centroids=np.empty((0, 2)),
            perimeter=None, areas=None, nn=None,
            warnings=warnings_, upstream_warnings=list(warnings_),
            **settings, **base,
        )

    # ---------------- step 1: DBSCAN + automatic curation ---------------
    pts = np.column_stack([xs, ys])
    labels = DBSCAN(eps=eps_nm, min_samples=min_samples).fit(pts).labels_
    n_raw = len({int(l) for l in labels} - {-1})

    # When no explicit ROI shape is supplied, approximate the boundary with
    # the convex hull of the analysed localizations. The runaway guard
    # inside identify_bad_clusters protects against the ring-on-boundary
    # case this can produce.
    roi_for_edges = roi
    if roi_for_edges is None and len(pts) >= 3:
        try:
            roi_for_edges = PolygonROI(vertices=pts[ConvexHull(pts).vertices])
        except Exception:
            roi_for_edges = None
    edge_reference = ("roi" if roi is not None
                      else "convex hull" if roi_for_edges is not None
                      else "none")

    report = identify_bad_clusters(
        xs, ys, labels, roi_for_edges,
        edge_margin_nm=eps_nm, dbcv_threshold=dbcv_threshold,
    )
    if report.edge_criterion_disabled:
        warnings_.append(report.edge_criterion_disabled)

    centroids = good_cluster_centroids(xs, ys, labels, report.bad_labels)
    n_kept = len(centroids)

    base_cluster: Dict[str, Any] = dict(
        labels=labels, bad_report=report,
        n_clusters_raw=n_raw, n_clusters_kept=n_kept,
        centroids=centroids,
    )

    steps = _cluster_steps(
        xs, ys, labels, set(report.bad_labels), centroids,
        custom_contour_order=custom_contour_order,
        contour_guide=contour_guide, all_starts=all_starts,
        **settings)
    return AxonAnalysis(
        perimeter=steps.perimeter, areas=steps.areas, nn=steps.nn,
        occupancy=steps.occupancy, randomization=steps.randomization,
        warnings=warnings_ + steps.warnings,
        upstream_warnings=list(warnings_),
        edge_reference=edge_reference,
        **settings, **base, **base_cluster,
    )


@dataclass
class _ClusterSteps:
    """What steps 3-6 produce from one set of clusters."""

    areas: Optional[ClusterAreaResult]
    perimeter: Optional[PerimeterResult]
    nn: Optional[NNResult]
    occupancy: Optional[OccupancyResult]
    randomization: Optional[RandomizationResult]
    warnings: List[str]


def _cluster_steps(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    labels: NDArray[np.int64],
    excluded: Set[int],
    centroids: NDArray[np.float64],
    *,
    mahalanobis_threshold: float,
    ellipse_mode: str,
    run_randomization: bool,
    n_randomizations: int,
    random_seed: int,
    custom_contour_order: Optional[NDArray[np.intp]] = None,
    all_starts: bool = False,
    contour: Optional[PerimeterResult] = None,
    left_after: str = "curation",
    contour_guide: Optional[NDArray[np.float64]] = None,
) -> _ClusterSteps:
    """
    Steps 3-6 on the clusters whose labels are not in ``excluded``.

    ``centroids`` are those clusters' centres, in label order. The contour
    is ``contour`` when one is given (it must join exactly these centres),
    otherwise it is built from them. ``left_after`` names what the clusters
    went through, for the message when too few are left.
    """
    warnings_: List[str] = []
    n_kept = len(centroids)

    # ---------------- step 3: areas (independent of the contour) --------
    areas = compute_cluster_areas(xs, ys, labels, exclude_labels=excluded)
    warnings_.extend(areas.warnings)

    # ---------------- step 3: perimeter ---------------------------------
    perimeter: Optional[PerimeterResult] = contour
    if perimeter is None and n_kept >= 3:
        perimeter = reconstruct_perimeter(
            centroids, refine=True, custom_order=custom_contour_order,
            all_starts=all_starts, guide=contour_guide)
    if perimeter is not None:
        warnings_.extend(perimeter.warnings)
    else:
        warnings_.append(
            f"Only {n_kept} cluster(s) survived {left_after}: a closed "
            f"contour needs at least 3, so perimeter and occupancy cannot "
            f"be computed."
        )

    # ---------------- step 4: 1NN ---------------------------------------
    nn: Optional[NNResult] = None
    if n_kept >= 2:
        nn = compute_nn_distances(centroids, k=1)
        warnings_.extend(nn.warnings)

    # ---------------- step 5: perimeter occupancy -----------------------
    occupancy: Optional[OccupancyResult] = None
    if perimeter is not None:
        try:
            occupancy = compute_occupancy(
                xs, ys, labels, perimeter.contour,
                exclude_labels=excluded,
                mahalanobis_threshold=mahalanobis_threshold,
                ellipse_mode=ellipse_mode,
            )
            warnings_.extend(occupancy.warnings)
        except Exception as exc:                      # noqa: BLE001
            warnings_.append(f"Occupancy could not be computed: {exc}")

    # ---------------- step 6: randomization control ---------------------
    randomization: Optional[RandomizationResult] = None
    if (run_randomization and perimeter is not None and nn is not None
            and n_kept >= 2):
        try:
            randomization = randomize_cluster_positions(
                centroids, perimeter.contour,
                experimental_1nn_nm=nn.first_nn_nm,
                n_iterations=n_randomizations,
                random_seed=random_seed,
            )
            warnings_.extend(randomization.warnings)
        except Exception as exc:                      # noqa: BLE001
            warnings_.append(f"Randomization control failed: {exc}")

    return _ClusterSteps(areas=areas, perimeter=perimeter, nn=nn,
                         occupancy=occupancy, randomization=randomization,
                         warnings=warnings_)


def without_clusters(
    analysis: AxonAnalysis,
    discarded: NDArray[np.bool_],
    *,
    margin_nm: float,
    registration: str = "",
    contour: Optional[PerimeterResult] = None,
) -> AxonAnalysis:
    """
    The same analysis with some of its kept clusters left out.

    ``discarded`` flags rows of ``analysis.centroids``: the clusters the
    axoplasm panel found more than ``margin_nm`` inside the axon in both
    widefield images. Steps 3-6 -- cluster areas, contour, 1NN, occupancy
    and the randomization -- run again on the rest, with the settings
    ``analysis`` ran with, the same random seed included. The axial slab,
    DBSCAN and the automatic curation are those of ``analysis``.

    The contour is built with 2-opt from every start, as in
    with_every_start, which is what this must be compared with: the two
    differ only by the clusters left out, and with nothing discarded they
    are the same numbers. ``contour``, when given, is that contour already
    built -- the one the panel drew -- and must join exactly the clusters
    that remain.
    """
    return _rerun(analysis, discarded, contour=contour, margin_nm=margin_nm,
                  registration=registration)


def with_every_start(
    analysis: AxonAnalysis,
    *,
    contour: Optional[PerimeterResult] = None,
) -> AxonAnalysis:
    """
    The same analysis with its contour built by 2-opt from every start.

    2-opt is a local search: the tour it settles on depends on the cluster
    it starts from. On the 18 April axons, rotating the start changed the
    perimeter by up to 12.9 %, more than leaving one or two clusters out
    does, and in either direction: with one start, discarding interior
    clusters lengthened two contours of four. The shortest tour over every
    start does not depend on where the centres are listed from, so the
    effect of the discard is measured between this analysis and
    without_clusters, both built that way. Steps 3-6 run again, since the
    contour moves occupancy and the randomization; nothing before them
    changes. ``contour`` is the contour of all the kept clusters already
    built with every start.

    An analysis already built that way -- analyze_axon's default -- comes
    back as it is, and so does one with too few clusters for a contour.
    """
    if not analysis.discard_applied and (
            analysis.perimeter is None
            or analysis.perimeter.n_starts > 1
            # An order a person set is not a 2-opt result to be improved
            # on. Without this line, opening the axoplasm panel silently
            # replaced it: measured, the perimeter went from 19.31 um
            # back to 8.96 um with nothing touched.
            or analysis.perimeter.order_source != "automatic"):
        return analysis
    return _rerun(analysis, np.zeros(analysis.n_clusters_kept, dtype=bool),
                  contour=contour, margin_nm=None)


def _rerun(
    analysis: AxonAnalysis,
    discarded: NDArray[np.bool_],
    *,
    contour: Optional[PerimeterResult],
    margin_nm: Optional[float],
    registration: str = "",
) -> AxonAnalysis:
    """Steps 3-6 again, 2-opt from every start, without the flagged
    clusters; ``margin_nm`` None records that nothing was discarded."""
    flags = np.asarray(discarded, dtype=bool).ravel()
    if flags.size != analysis.n_clusters_kept:
        raise ValueError(
            f"{flags.size} discard flag(s) for the "
            f"{analysis.n_clusters_kept} clusters the analysis kept.")
    if analysis.discard_applied:
        raise ValueError("Start from the analysis of all the clusters.")
    labels = good_cluster_labels(analysis.labels,
                                 analysis.bad_report.bad_labels)
    if labels.size != analysis.n_clusters_kept:
        raise ValueError("The analysis' labels do not match its centroids.")
    centroids = np.asarray(analysis.centroids, dtype=float)[~flags]
    if contour is not None:
        if (contour.n_clusters != len(centroids)
                or not np.array_equal(contour.contour,
                                      centroids[contour.order])):
            raise ValueError(
                "The contour given joins other clusters than the ones left.")
        if (contour.order_source == "automatic"
                and contour.n_starts == 1 and len(centroids) >= 4):
            raise ValueError(
                "The contour given was built from one 2-opt start, not "
                "from every start.")
    dropped = frozenset(int(label) for label in labels[flags])
    steps = _cluster_steps(
        analysis.x_slab, analysis.y_slab, analysis.labels,
        set(analysis.bad_report.bad_labels) | set(dropped), centroids,
        mahalanobis_threshold=analysis.mahalanobis_threshold,
        ellipse_mode=analysis.ellipse_mode,
        run_randomization=analysis.run_randomization,
        n_randomizations=analysis.n_randomizations,
        random_seed=analysis.random_seed,
        all_starts=True, contour=contour,
        # A contour drawn by hand is drawn for this axon, not for this set
        # of clusters: the ones the discard leaves are joined along the
        # same path. Without this the "_discard" columns would compare a
        # contour a person drew with one the program built.
        contour_guide=(analysis.contour_guide if contour is None else None),
        left_after=("curation" if margin_nm is None
                    else "curation and the discard"))
    notes = ([] if margin_nm is None else
             [_discard_note(len(dropped), analysis.n_clusters_kept,
                            margin_nm)])
    return dataclasses.replace(
        analysis,
        n_clusters_kept=int(len(centroids)), centroids=centroids,
        areas=steps.areas, perimeter=steps.perimeter, nn=steps.nn,
        occupancy=steps.occupancy, randomization=steps.randomization,
        warnings=list(analysis.upstream_warnings) + notes + steps.warnings,
        upstream_warnings=list(analysis.upstream_warnings),
        discard_margin_nm=None if margin_nm is None else float(margin_nm),
        discarded_labels=dropped,
        discard_registration="" if margin_nm is None else registration,
    )


@dataclass
class DiscardComparison:
    """
    One axon analysed with all its kept clusters and without the discarded
    ones, both with 2-opt from every start: they differ only by the
    clusters left out.
    """

    all_clusters: AxonAnalysis
    discard_applied: AxonAnalysis


def compare_discard(
    analysis: AxonAnalysis,
    discarded: NDArray[np.bool_],
    *,
    margin_nm: float,
    contour_all: Optional[PerimeterResult] = None,
    contour_kept: Optional[PerimeterResult] = None,
    all_clusters: Optional[AxonAnalysis] = None,
    registration: str = "",
) -> DiscardComparison:
    """
    with_every_start and without_clusters of ``analysis``.

    ``contour_all`` and ``contour_kept`` are the two contours already built
    with every start (the axoplasm panel builds them). ``all_clusters`` is
    a with_every_start result of this same analysis kept from before: it
    does not depend on the discard, so only the other half is recomputed
    when the margin moves. With nothing discarded, the second half is the
    first one with the discard recorded, not a second run.
    """
    flags = np.asarray(discarded, dtype=bool).ravel()
    if all_clusters is not None and (all_clusters.labels is not analysis.labels
                                     or all_clusters.discard_applied):
        raise ValueError("all_clusters is not this analysis of all clusters.")
    every = (all_clusters if all_clusters is not None
             else with_every_start(analysis, contour=contour_all))
    if flags.any():
        applied = without_clusters(analysis, flags, margin_nm=margin_nm,
                                   contour=contour_kept,
                                   registration=registration)
    else:
        if flags.size != analysis.n_clusters_kept:
            raise ValueError(
                f"{flags.size} discard flag(s) for the "
                f"{analysis.n_clusters_kept} clusters the analysis kept.")
        upstream = list(every.upstream_warnings)
        note = _discard_note(0, analysis.n_clusters_kept, margin_nm)
        applied = dataclasses.replace(
            every, discard_margin_nm=float(margin_nm),
            discarded_labels=frozenset(),
            discard_registration=registration,
            warnings=upstream + [note] + every.warnings[len(upstream):])
    return DiscardComparison(all_clusters=every, discard_applied=applied)


def _discard_note(n_dropped: int, n_before: int, margin_nm: float) -> str:
    if n_dropped == 0:
        return (
            f"Discard applied: none of the {n_before} clusters is more than "
            f"{margin_nm:,.0f} nm inside the axon in both widefield images, "
            f"so none was left out.")
    return (
        f"Discard applied: {n_dropped} of {n_before} clusters left out, "
        f"those both widefield images place more than {margin_nm:,.0f} nm "
        f"inside the axon. Gazal et al. (2026) do not describe leaving out "
        f"clusters that lie inside the axon: this step goes beyond their "
        f"Methods.")


def with_discard_margin(analysis: AxonAnalysis, margin_nm: float,
                        registration: Optional[str] = None) -> AxonAnalysis:
    """
    ``analysis`` (from without_clusters) recorded at another margin that
    leaves out the same clusters: every number stays, only the margin it is
    reported with changes, so moving the margin does not re-run steps 3-6.
    """
    if not analysis.discard_applied:
        raise ValueError("This analysis has no discarded clusters.")
    warnings = list(analysis.warnings)
    # without_clusters puts its note right after the inherited warnings.
    at = len(analysis.upstream_warnings)
    n_dropped = len(analysis.discarded_labels)
    warnings[at] = _discard_note(
        n_dropped, analysis.n_clusters_kept + n_dropped, margin_nm)
    return dataclasses.replace(
        analysis, discard_margin_nm=float(margin_nm), warnings=warnings,
        discard_registration=(analysis.discard_registration
                              if registration is None else registration))
