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

This module is deliberately free of any Qt dependency so the whole pipeline
can be run and tested headlessly (see validate_full_18axons.py).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from tools.cluster_quality import (
    BadClusterReport,
    PolygonROI,
    ROIShape,
    good_cluster_centroids,
    identify_bad_clusters,
)
from tools.mps_geometry import (
    PAPER_INTERCEPT_CLUSTERS,
    PAPER_MEDIAN_CLUSTER_AREA_NM2,
    PAPER_MEDIAN_R_EFF_NM,
    PAPER_SLOPE_CLUSTERS_PER_UM,
    ClusterAreaResult,
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

# Paper defaults (Gazal et al. 2026)
DEFAULT_EPS_NM = 25.0
DEFAULT_MIN_SAMPLES = 10

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


@dataclass
class AxonAnalysis:
    """Every per-axon parameter produced by one run of the pipeline."""

    # --- provenance -----------------------------------------------------
    source_name: str
    pixel_size_nm: Optional[float]
    pixel_size_source: str          # "yaml" | "manual" | "unknown"
    eps_nm: float
    min_samples: int
    slab_half_width_nm: float

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

    # ---------------- convenience accessors for the panel ----------------

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

        rows: List[Tuple[str, str, str, str]] = [
            ("Localizations (ROI)", f"{self.n_locs_total:,}", "-", ""),
            ("Localizations (180 nm slab)", f"{self.n_locs_slab:,}", "-", ""),
            ("Axial slab z-range",
             f"{self.slab_zmin_nm:,.0f} .. {self.slab_zmax_nm:,.0f} nm", "-",
             "auto from GMM main peak"),
            ("Delta-Z (axial periodicity)", fmt(self.mean_delta_z_nm, 1),
             "170 +/- 15 nm", f"{len(self.z_result.delta_z_nm)} interval(s)"),
            ("Clusters detected (raw)", f"{self.n_clusters_raw}", "-", ""),
            ("Clusters kept (auto-curated)", f"{self.n_clusters_kept}", "-",
             f"{len(self.bad_report.bad_labels)} removed"),
            ("Perimeter", fmt(self.perimeter_um, 2) + " um", "-",
             "centroids connected, 2-opt"),
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
            "pixel_size_nm": self.pixel_size_nm,
            "pixel_size_source": self.pixel_size_source,
            "eps_nm": self.eps_nm,
            "min_samples": self.min_samples,
            "slab_half_width_nm": self.slab_half_width_nm,
            "slab_zmin_nm": round(self.slab_zmin_nm, 2),
            "slab_zmax_nm": round(self.slab_zmax_nm, 2),
            "n_locs_total": self.n_locs_total,
            "n_locs_slab": self.n_locs_slab,
            "gmm_n_components": self.z_result.n_components,
            "delta_z_mean_nm": self.mean_delta_z_nm,
            "delta_z_values_nm": ";".join(
                f"{d:.1f}" for d in self.z_result.delta_z_nm),
            "n_clusters_raw": self.n_clusters_raw,
            "n_clusters_kept": self.n_clusters_kept,
            "n_clusters_removed": len(self.bad_report.bad_labels),
            "removed_edge_touching": len(self.bad_report.edge_touching),
            "removed_low_dbcv": len(self.bad_report.low_dbcv),
            "edge_criterion_disabled": bool(
                self.bad_report.edge_criterion_disabled),
            "perimeter_um": self.perimeter_um,
            "clusters_per_um": self.clusters_per_um,
            "median_area_nm2": self.median_area_nm2,
            "median_r_eff_nm": self.median_r_eff_nm,
            "median_1nn_nm": self.median_1nn_nm,
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
            "warnings": " | ".join(self.warnings),
        }


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
    dbcv_threshold: float = 0.0,
    mahalanobis_threshold: float = 3.0,
    ellipse_mode: str = "clip",
    run_randomization: bool = True,
    n_randomizations: int = DEFAULT_N_RANDOMIZATIONS,
    random_seed: int = 0,
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
    if pixel_size_source not in ("yaml", "manual", "unknown", "not_applicable"):
        raise ValueError(f"bad pixel_size_source: {pixel_size_source!r}")
    if pixel_size_source in ("manual", "unknown"):
        # "not_applicable" means the file was already in nanometres
        # (ThunderSTORM / custom CSV), so there is nothing to warn about.
        warnings_.append(
            f"Pixel size ({pixel_size_nm} nm) did not come from the Picasso "
            f"YAML sidecar (source: {pixel_size_source}). Every lateral "
            f"distance and, squared, every cluster area scales with it -- "
            f"verify it before using these numbers."
        )

    # ---------------- step 2: axial periodicity and slab ----------------
    z_result = fit_z_periodicity(z_nm)
    warnings_.extend(z_result.warnings)

    if slab_override is not None:
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
        if main_peak_override_nm is not None:
            warnings_.append(
                f"Axial peak selected manually at z = {peak:.0f} nm "
                f"(automatic choice was {z_result.main_peak_nm:.0f} nm)."
            )
        slab_mask = select_mps_slab(z_nm, peak, slab_half_width_nm)
        zmin, zmax = peak - slab_half_width_nm, peak + slab_half_width_nm

    xs, ys, zs = x_nm[slab_mask], y_nm[slab_mask], z_nm[slab_mask]

    base = dict(
        source_name=source_name,
        pixel_size_nm=pixel_size_nm,
        pixel_size_source=pixel_size_source,
        eps_nm=eps_nm,
        min_samples=min_samples,
        slab_half_width_nm=slab_half_width_nm,
        z_result=z_result,
        slab_zmin_nm=zmin,
        slab_zmax_nm=zmax,
        n_locs_total=int(x_nm.size),
        n_locs_slab=int(xs.size),
        x_slab=xs, y_slab=ys, z_slab=zs,
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
            warnings=warnings_, **base,
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

    report = identify_bad_clusters(
        xs, ys, labels, roi_for_edges,
        edge_margin_nm=eps_nm, dbcv_threshold=dbcv_threshold,
    )
    if report.edge_criterion_disabled:
        warnings_.append(report.edge_criterion_disabled)

    centroids = good_cluster_centroids(xs, ys, labels, report.bad_labels)
    n_kept = len(centroids)

    base_cluster = dict(
        labels=labels, bad_report=report,
        n_clusters_raw=n_raw, n_clusters_kept=n_kept,
        centroids=centroids,
    )

    # ---------------- step 3: areas (independent of the contour) --------
    areas = compute_cluster_areas(xs, ys, labels,
                                  exclude_labels=report.bad_labels)
    warnings_.extend(areas.warnings)

    # ---------------- step 3: perimeter ---------------------------------
    perimeter: Optional[PerimeterResult] = None
    if n_kept >= 3:
        perimeter = reconstruct_perimeter(
            centroids, refine=True, custom_order=custom_contour_order)
        warnings_.extend(perimeter.warnings)
    else:
        warnings_.append(
            f"Only {n_kept} cluster(s) survived curation: a closed contour "
            f"needs at least 3, so perimeter and occupancy cannot be computed."
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
                exclude_labels=report.bad_labels,
                mahalanobis_threshold=mahalanobis_threshold,
                ellipse_mode=ellipse_mode,
            )
            warnings_.extend(occupancy.warnings)
        except Exception as exc:                      # noqa: BLE001
            warnings_.append(f"Occupancy could not be computed: {exc}")

    # ---------------- step 6: randomization control ---------------------
    randomization: Optional[RandomizationResult] = None
    if run_randomization and perimeter is not None and nn is not None             and n_kept >= 2:
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

    return AxonAnalysis(
        perimeter=perimeter, areas=areas, nn=nn, occupancy=occupancy,
        randomization=randomization,
        warnings=warnings_, **base, **base_cluster,
    )
