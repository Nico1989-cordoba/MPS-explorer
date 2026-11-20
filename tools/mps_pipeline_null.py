# -*- coding: utf-8 -*-
"""
The per-axon pipeline null with blinking (plan A9, "Null (per axon)"; regularity plan PR 1; decision D7: ONE
pipeline null shared by the cluster readouts and the regularity readouts).

The question it answers
-----------------------
"What would THIS pipeline -- this clustering method with these parameters, this curation, this contour -- report
on this axon if the molecules sat at random on its membrane?" Each replicate is a whole simulated axon:

    molecules at random on the axon's own membrane (arc positions uniform, with a contact hard core d_c)
      -> 0, 1 or 2 labels per molecule (Binomial(2, p_label)), each at a point of the epitope disc + linkage
      -> c_label switching cycles per label, ALL at the label's position (the cycles of one label share its
         linkage: that is what makes a molecule a cloud of localizations rather than one point)
      -> localizations per cycle (geometric, mean loc/cycle) with the file's NeNA (lateral) and axial error
      -> background at the density measured on the file, two neighbouring rings at +/- the axial period
      -> THE SAME PIPELINE: tools.mps_analysis.analyze_axon with the same clustering method and parameters
         (any ClusteringStrategy: DBSCAN, HDBSCAN, ToMATo, Voronoi, ...), the same slab, curation and contour
      -> the readouts (clusters per um, median 1NN, median r_eff, occupancy, regularity r, and any readout
         registered with ``register_readout``) and Ripley's H on the arc (H1) and in the band (H2).

The observed axon is compared with the B replicates by Monte Carlo p per readout (Besag & Diggle, 1977) and,
for H1 / H2, by the global maximum-studentised-deviation test (MAD; ``tools.spatial_stats.envelope_from_curves``)
over the radii above the working scale's fusion mark (``tools.mps_working_scale``); the pointwise envelope is
returned for drawing only.

Why molecules, not localizations
--------------------------------
A null of independent random LOCALIZATIONS is wrong in SMLM: each label leaves a cloud of tens of localizations,
so such a null never forms a cluster and "the observed has clusters, the null does not" is significant whatever
the biology (Platzer et al., 2020; Nieves et al., 2023, scenario 1; Schneider & Schuetz, 2022: 22 DBSCAN clusters
in CSR without blinking, 116 with realistic blinking). The null randomises what biology could randomise (the
molecules) and keeps what the instrument and the algorithm do anyway (blinking, precision, background, fusion).

Method-agnostic by construction
-------------------------------
``PipelineSpec`` carries the clustering method (a name of ``tools.clustering_strategies`` + its parameters, or a
``ClusteringStrategy`` template that is deep-copied for each replicate) and the curation settings, and runs
``analyze_axon`` with them on every replicate exactly as on the observed axon (``PipelineSpec.from_analysis``
reads them back from an ``AxonAnalysis``). Methods the program marks as slow (``tools.cluster_method_ui.is_slow``;
Emitters above all) are refused with a cost estimate unless ``allow_slow=True``; "External labels" is always
refused (its labels come from a file and cannot be recomputed on a simulated axon).

Closure (A9 sec. 3.3, decided 2026-10-22): on the localizations in the slab (default): every replicate is
conditioned on the observed slab's localizations (a pool of molecules, of which the first n per ring that come
closest to the count are kept; a single molecule count for all replicates inherited the observed count's ~10-15 %
noise and made the K per um test liberal [simulation: docs/sims/pipeline_null]), and K per um is
then a free readout of the null whose p means something (provisional while c_label comes from the literature);
or, as a sensitivity, on the detected clusters K (no c_label needed, but K per um is then conditioned and its p
is left empty).

Origins of every default (labels: [lit], [derivation], [simulation], [ad hoc], [read]: measured on the axon):
  B = 199 [user decision 2026-10-22; p_min = 1/(B + 1) = 0.005], 999 on demand.
  c_label = 14 [lit: Dempsey et al. 2011, Table 1, Alexa Fluor 647 in MEA; provisional until measured in the
    lab's buffer], sensitivity 26 [lit: same table, betaME].
  loc/cycle: measured on the file by event linking (radius 3 NeNA, 1 dark frame; tools.mps_paint) when frames
    are given, else 2 [ad hoc: the simulator's value; regularity plan P-R79 (b) with (a) meanwhile].
  labels per molecule 2, p_label 1 [ad hoc: two epitopes per tetramer, every one labelled].
  cycles per label: geometric [ad hoc, the clustering research's generator; Dempsey's per-molecule counts are
    broad], Poisson on request (A9 sec. 3.2 step 3).
  sigma_link 12 nm [ad hoc: the simulator's value, of the order of Frueh et al. 2021].
  epitope disc radius 25 nm [ad hoc, simulator; NOT read: not identifiable under blinking, see the code].
  hard core d_c = 2 x median r_eff [read; regularity plan P-R77 (a) contact, editable (e)].
  membrane: 600 nm P-spline through the observed centroids [regularity plan P-R80 (a); tools.mps_membrane].
  radial offset sd of the molecules: the centroids' scatter about that membrane [read].
  axial: the slab's own GMM component sd minus the label offsets [read; derivation], period = mean Delta-Z
    [read], else 170 nm [lit: Gazal et al. 2026]; three rings, the middle one in the slab.
  r grid 20-400 nm step 10, upper limit editable [ad hoc; A9 sec. 2.5, decided]; alpha 0.05 [ad hoc].
  4 closure pilots [regularity plan PR 1]; closure tolerance 5 %, fidelity tolerance 15 % [A9 sec. 3.4].
  NOT INTERPRETABLE below 19 accepted replicates [derivation: 1/(19 + 1) = 0.05], with failed fidelity, with a
  closure not reached; K < 8 is refused [regularity plan sec. 5 rule 1].

No Qt. Deterministic: replicate b draws from SeedSequence(seed, spawn_key=(0, b)), closure pilots from (1, i, j),
so a run is bit-identical with the same seed and n_jobs does not change any number.

References (APA 7)
------------------
Besag, J., & Diggle, P. J. (1977). Simple Monte Carlo tests for spatial pattern. Applied Statistics, 26(3),
    327-333. https://doi.org/10.2307/2346974
Dempsey, G. T., Vaughan, J. C., Chen, K. H., Bates, M., & Zhuang, X. (2011). Evaluation of fluorophores for optimal
    performance in localization-based super-resolution imaging. Nature Methods, 8(12), 1027-1036.
    https://doi.org/10.1038/nmeth.1768
Nieves, D. J., Pike, J. A., Levet, F., et al. (2023). A framework for evaluating the performance of SMLM cluster
    analysis algorithms. Nature Methods, 20, 259-267. https://doi.org/10.1038/s41592-022-01750-6
Platzer, R., Rossboth, B. K., Schneider, M. C., Sevcsik, E., Baumgart, F., Stockinger, H., Schuetz, G. J.,
    Huppa, J. B., & Brameshuber, M. (2020). Unscrambling fluorophore blinking for comprehensive
    cluster detection via photoactivated localization microscopy. Nature Communications, 11, 4993.
    https://doi.org/10.1038/s41467-020-18726-9
Schneider, M. C., & Schuetz, G. J. (2022). Don't be fooled by randomness: Valid p-values for single molecule
    microscopy. Frontiers in Bioinformatics, 2, 811053. https://doi.org/10.3389/fbinf.2022.811053

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import norm

from tools.clustering_strategies import COMPAT_METHOD, ClusteringStrategy
from tools.cluster_quality import CircularROI, PolygonROI, ROIShape, SquareROI, good_cluster_labels
from tools.mps_analysis import AxonAnalysis, analyze_axon
from tools.mps_matching import smooth_closed_path
from tools.mps_membrane import MEMBRANE_KNOT_SPACING_NM, radial_scatter_nm
from tools.mps_periodicity import select_mps_slab
from tools.mps_randomization import DEFAULT_ANNULUS_HALF_WIDTH_NM, monte_carlo_p
from tools.mps_simulate_axon import _frame_at_arc, _hard_core_arcs
from tools.mps_working_scale import WorkingScale, working_scale_of_analysis
from tools.spatial_stats import (
    CsrEnvelope,
    arc_coordinates,
    envelope_from_curves,
    point_in_polygon,
    polygon_area,
    ripley_k_1d_arc,
    ripley_k_2d,
    ripley_k_2d_area,
    simulate_csr_polygon,
)

FloatArray = NDArray[np.float64]

# ============================================================================
# Defaults (origin in brackets; the module docstring gives the argument)
# ============================================================================
DEFAULT_N_REPLICATES = 199            # [user decision 2026-10-22] p_min = 1 / 200 = 0.005
ON_DEMAND_N_REPLICATES = 999          # [user decision 2026-10-22] on request
MIN_ACCEPTED = 19                     # [derivation] below it the p can never reach 0.05 (as simnull)
MIN_CLUSTERS = 8                      # [regularity plan sec. 5, rule 1] K < 8: not testable
DEFAULT_C_LABEL = 14.0                # [lit: Dempsey et al. 2011, Table 1, AF647 MEA] provisional
SENSITIVITY_C_LABEL = 26.0            # [lit: same table, betaME] the sensitivity column (D1)
DEFAULT_LOCS_PER_CYCLE = 2.0          # [ad hoc: simulator's locs_per_fluor_mean] used when not measured
DEFAULT_LABELS_PER_MOLECULE = 2       # [ad hoc: two epitopes per tetramer, plan sec. 3.1]
DEFAULT_P_LABEL = 1.0                 # [ad hoc]
DEFAULT_SIGMA_LINK_NM = 12.0          # [ad hoc: simulator; order of Frueh et al. 2021]
DEFAULT_EPITOPE_RADIUS_NM = 25.0      # [ad hoc: simulator's epitope_radius_nm] used when not readable
DEFAULT_SIGMA_Z_NM = 50.0             # [ad hoc] axial error when the slab's GMM gives no width
DEFAULT_PERIOD_NM = 170.0             # [lit: Gazal et al. 2026, Delta-Z 170 +/- 15 nm]
DEFAULT_N_FRAMES = 60000              # [ad hoc: simulator default] when no frame numbers are given
N_RINGS = 3                           # [A9 sec. 3.2] the slab's ring and its two neighbours (axial leak)
LINK_RADIUS_IN_NENA = 3.0             # [plan D5; tools.mps_paint.LINK_RADIUS_IN_PRECISIONS] event linking
LINK_MAX_DARK_FRAMES = 1              # [plan D5] one dark frame
BAND_HALF_WIDTH_NM = DEFAULT_ANNULUS_HALF_WIDTH_NM   # [lit: Gazal et al. 2026] +/-50 nm band of H2
BACKGROUND_BAND_SIGMAS = 3.0          # [A9 sec. 3.1] background = slab locs beyond +/-(50 + 3 sigma) of the membrane
BACKGROUND_MARGIN_NM = 500.0          # [ad hoc: tools.mps_simulate_axon.BACKGROUND_MARGIN_DEFAULT_NM] no ROI given
R_MIN_NM, R_STEP_NM, R_MAX_NM = 20.0, 10.0, 400.0   # [ad hoc; A9 sec. 2.5, decided 2026-10-22]
ALPHA = 0.05                          # [ad hoc] conventional level
N_PILOTS = 4                          # [regularity plan PR 1] closure pilots with damped update
PILOT_AXONS_N_LOCS = 10               # [ad hoc] pilot axons per iteration when closing on N_loc (cheap: no clustering;
                                      # the slab count of one axon varies ~20 % with geometric cycles [simulation])
PILOT_AXONS_K = 4                     # [ad hoc, provisional: open question C3 of the clustering integration,
                                      # 2026-10-22 -- 2 (the simulator's CALIBRATION_AXONS_PER_ITERATION) left 9 of
                                      # 20 axons > 5 % off; 4 recommended, doubling the cost] when closing on K
K_CLOSURE_DAMPING = 0.8               # [ad hoc] exponent of the multiplicative update on K (compressive map)
CLOSURE_TOLERANCE = 0.05              # [A9 sec. 3.3] median of the null within 5 % of the observed
FIDELITY_TOLERANCE = 0.15             # [A9 sec. 3.4, as simnull D-29c]
FIDELITY_SCATTER_FLOOR_NM = 3.0       # [ad hoc] a radial-scatter check never stricter than 3 nm
FIDELITY_NENA_NM = 2.0                # [A9 sec. 3.4] re-measured NeNA within 2 nm
RSA_JAMMING = 0.7476                  # [lit: Renyi 1958, the parking constant; text not checked] random sequential packing on a line
JAM_FRACTION = 0.70                   # [ad hoc] refuse a hard core that would pack the ring above this
HARD_CORE_MAX_ATTEMPTS = 20000        # [ad hoc] attempts per molecule before a replicate counts as jammed
NULL_TYPE_PROVISIONAL = "molecules+blinking (provisional: c_label from literature)"
NULL_TYPE_CALIBRATED = "molecules+blinking (calibrated)"
CLOSURES = ("n_locs", "k_clusters")
# Where the global (MAD) test of H1 / H2 starts: the working scale's fusion mark (A9 as calibrated) or its d50.
# The integration starts at d50 [provisional: open question C1, 2026-10-22: the contact hard core 2 r_eff reads
# ~6 nm high, which made H1 / H2 somewhat liberal at 60-80 nm; tools.clustering_integration.PROVISIONAL].
GLOBAL_TEST_FROM = ("fusion", "d50")
GEOMETRIES = ("ring", "roi2d")
CYCLE_DISTS = ("geometric", "poisson")
# A method whose clusters come from a file cannot be recomputed on a simulated axon.
NEVER_IN_NULL = ("External labels",)


def default_r_grid(r_max_nm: float = R_MAX_NM) -> FloatArray:
    """20, 30, ..., r_max nm (A9 sec. 2.5; the upper limit is editable)."""
    if not r_max_nm >= R_MIN_NM:
        raise ValueError(f"r_max_nm must be >= {R_MIN_NM:g} nm, got {r_max_nm}")
    return np.arange(R_MIN_NM, float(r_max_nm) + 1e-9, R_STEP_NM, dtype=np.float64)


# ============================================================================
# Errors
# ============================================================================

class NullRefused(ValueError):
    """The null cannot or must not run on this axon (K < 8, a jammed hard core, a method that cannot be re-run)."""


class SlowMethodRefused(NullRefused):
    """A slow method was asked for without ``allow_slow=True``; ``estimate_s`` is the expected cost of the run."""

    def __init__(self, message: str, estimate_s: float) -> None:
        super().__init__(message)
        self.estimate_s = float(estimate_s)


# ============================================================================
# The pipeline the null repeats
# ============================================================================

def _roi_polygon(roi: Optional[ROIShape]) -> Optional[FloatArray]:
    """The ROI as a closed polygon (vertices only), or None."""
    if roi is None:
        return None
    if isinstance(roi, PolygonROI):
        v: FloatArray = np.asarray(roi.vertices, dtype=np.float64).reshape(-1, 2)
        if v.shape[0] >= 2 and np.allclose(v[0], v[-1]):
            v = v[:-1]
        return v
    if isinstance(roi, SquareROI):
        return np.array([[roi.xmin, roi.ymin], [roi.xmax, roi.ymin], [roi.xmax, roi.ymax], [roi.xmin, roi.ymax]],
                        dtype=np.float64)
    if isinstance(roi, CircularROI):
        t = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        return np.column_stack([roi.center_x + roi.radius * np.cos(t), roi.center_y + roi.radius * np.sin(t)])
    return None


@dataclass
class PipelineSpec:
    """Everything ``analyze_axon`` decides with that the null must repeat: the clustering method and its
    parameters (or a strategy template), and the slab / curation / contour settings.

    ``strategy`` (optional) is a ``ClusteringStrategy`` template: every run gets a deep copy, so a fitted state
    never leaks between replicates. Without it ``analyze_axon`` builds the method from ``method`` + ``params``
    (DBSCAN from ``eps_nm`` / ``min_samples``), exactly as for the observed axon."""

    method: str = COMPAT_METHOD
    params: Dict[str, Any] = field(default_factory=dict)
    strategy: Optional[ClusteringStrategy] = None
    eps_nm: float = 25.0
    min_samples: int = 10
    slab_half_width_nm: float = 90.0
    dbcv_threshold: Optional[float] = None
    roi: Optional[ROIShape] = None
    mahalanobis_threshold: float = 3.0
    ellipse_mode: str = "clip"
    sigma_cap_fraction: Optional[float] = None
    all_starts: bool = True
    main_peak_mode: str = "density"

    @classmethod
    def from_analysis(cls, analysis: AxonAnalysis, strategy: Optional[ClusteringStrategy] = None) -> "PipelineSpec":
        """The pipeline an ``AxonAnalysis`` ran: its method and parameters, slab, curation and contour mode."""
        all_starts = analysis.contour_2opt != "one start"
        return cls(method=str(analysis.cluster_method), params=dict(analysis.cluster_params or {}), strategy=strategy,
                   eps_nm=float(analysis.eps_nm), min_samples=int(analysis.min_samples),
                   slab_half_width_nm=float(analysis.slab_half_width_nm), dbcv_threshold=float(analysis.dbcv_threshold),
                   roi=analysis.roi, mahalanobis_threshold=float(analysis.mahalanobis_threshold),
                   ellipse_mode=str(analysis.ellipse_mode), sigma_cap_fraction=float(analysis.sigma_cap_fraction),
                   all_starts=bool(all_starts), main_peak_mode=str(analysis.z_result.main_peak_mode))

    @classmethod
    def for_method(cls, method: str, params: Optional[Dict[str, Any]] = None, **kw: Any) -> "PipelineSpec":
        """A spec for a method name of the registry; DBSCAN takes eps / min_samples from ``params``."""
        p = dict(params or {})
        if method == COMPAT_METHOD:
            kw.setdefault("eps_nm", float(p.get("eps", 25.0)))
            kw.setdefault("min_samples", int(p.get("min_samples", 10)))
        return cls(method=method, params=p, **kw)

    def method_name(self) -> str:
        return self.method if self.strategy is None else str(self.strategy.get_strategy_name())

    def run(self, x_nm: FloatArray, y_nm: FloatArray, z_nm: FloatArray, *, main_peak_nm: Optional[float] = None,
            lp_nm: Optional[FloatArray] = None, frame: Optional[NDArray[np.int64]] = None,
            source_name: str = "pipeline null") -> AxonAnalysis:
        """``analyze_axon`` with this method and these settings (randomization control off: it is the paper's
        separate centres null, not part of what the readouts measure)."""
        kw: Dict[str, Any] = {}
        if self.dbcv_threshold is not None:
            kw["dbcv_threshold"] = float(self.dbcv_threshold)
        if self.sigma_cap_fraction is not None:
            kw["sigma_cap_fraction"] = float(self.sigma_cap_fraction)
        strategy = None if self.strategy is None else copy.deepcopy(self.strategy)
        return analyze_axon(
            x_nm, y_nm, z_nm, source_name=source_name, pixel_size_source="not_applicable",
            eps_nm=float(self.eps_nm), min_samples=int(self.min_samples),
            slab_half_width_nm=float(self.slab_half_width_nm), roi=self.roi, main_peak_override_nm=main_peak_nm,
            mahalanobis_threshold=float(self.mahalanobis_threshold), ellipse_mode=str(self.ellipse_mode),
            run_randomization=False, all_starts=bool(self.all_starts), main_peak_mode=self.main_peak_mode,
            cluster_method=self.method, cluster_params=dict(self.params), strategy=strategy,
            localization_sigma_nm=lp_nm, localization_frame=frame, **kw)


# ============================================================================
# Cost and refusal of slow methods
# ============================================================================

def estimated_seconds_per_replicate(method: str, n_locs_slab: int) -> float:
    """A FIRST estimate before anything runs: the per-1000-localization cost table of the Compare-methods dialog
    (``tools.cluster_method_ui.SECONDS_PER_KLOC`` and ``SCALING_EXPONENT``, [simulation] on a synthetic ring, the
    clustering alone) plus its ``PIPELINE_PER_KLOC`` [ad hoc] for the rest of the analysis. The run itself
    measures the real figure (``PipelineNullResult.seconds_per_replicate``)."""
    from tools.method_costs import PIPELINE_PER_KLOC, SCALING_EXPONENT, SECONDS_PER_KLOC
    k = max(float(n_locs_slab), 1.0) / 1000.0
    per = float(SECONDS_PER_KLOC.get(method, 0.05)) * k ** float(SCALING_EXPONENT.get(method, 1.0))
    return float(per + PIPELINE_PER_KLOC * k)


def method_is_slow(method: str) -> bool:
    """``tools.method_costs.is_slow`` (the slow rule of the analysis window, without Qt)."""
    from tools.method_costs import is_slow
    return bool(is_slow(method))


def check_method(spec: PipelineSpec, n_locs_slab: int, n_replicates: int, allow_slow: bool = False) -> float:
    """Refuse what the null must not run; return the estimated seconds of the whole run.

    "External labels" is always refused (its labels come from a file of the OBSERVED data). A slow method
    (``is_slow``: Emitters, OPTICS, voronoi_csr) is refused unless ``allow_slow`` with the estimate in the
    message, so that a 199-replicate null never starts an hours-long run silently."""
    name = spec.method_name()
    if name in NEVER_IN_NULL or spec.method in NEVER_IN_NULL:
        raise NullRefused(f"{name}: its labels come from a file computed on the observed localizations; a "
                          "simulated axon has no such file, so the pipeline null cannot re-run this method.")
    per = estimated_seconds_per_replicate(spec.method, n_locs_slab)
    total = per * (int(n_replicates) + N_PILOTS * PILOT_AXONS_K)
    if method_is_slow(spec.method) and not allow_slow:
        raise SlowMethodRefused(
            f"{name} is slow: about {per:.0f} s per simulated axon on {n_locs_slab} slab localizations, so "
            f"{n_replicates} replicates would take about {total / 60.0:.0f} min or more [first estimate from the "
            f"Compare-methods cost table; a lower bound for the slow methods]. Pass allow_slow=True to run it anyway, "
            f"or use fewer replicates.", total)
    return float(total)


# ============================================================================
# The generative model of the null
# ============================================================================

@dataclass
class NullModel:
    """What a replicate is drawn from. Every field that was read on the observed axon says so in ``origins``.

    ``geometry`` "ring": molecules on ``membrane_nm`` (closed curve, axon frame, nm) at three rings ``period_nm``
    apart, the middle one at ``z_centre_nm`` (the slab's centre). "roi2d" (A9 sec. 3.5, an axon without a ring):
    molecules uniform in ``region_nm`` and uniform in z over the three rings' span.
    ``n_molecules_per_ring`` is what the closure solves (a float: each replicate rounds it stochastically)."""

    geometry: str
    membrane_nm: Optional[FloatArray]
    region_nm: FloatArray
    z_centre_nm: float
    period_nm: float
    slab_half_width_nm: float
    n_molecules_per_ring: float
    hard_core_nm: float
    radial_sd_nm: float
    sigma_xy_nm: float
    sigma_z_nm: float
    background_per_um2: float
    n_frames: int
    c_label: float = DEFAULT_C_LABEL
    locs_per_cycle: float = DEFAULT_LOCS_PER_CYCLE
    labels_per_molecule: int = DEFAULT_LABELS_PER_MOLECULE
    p_label: float = DEFAULT_P_LABEL
    epitope_radius_nm: float = DEFAULT_EPITOPE_RADIUS_NM
    sigma_link_nm: float = DEFAULT_SIGMA_LINK_NM
    cycles_dist: str = "geometric"
    n_rings: int = N_RINGS
    # Closure on the slab's localizations, replicate by replicate (None: n_molecules_per_ring molecules per ring).
    # When set, each replicate draws a POOL of molecules (sequential inhibition, so any prefix of the pool is itself
    # a hard-core sample) and keeps the first n per ring whose slab localizations, background included, come
    # closest to this count: the null is conditioned on N_loc as the observed axon is (see run_pipeline_null).
    condition_slab_locs: Optional[int] = None
    origins: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}, got {self.geometry!r}")
        if self.cycles_dist not in CYCLE_DISTS:
            raise ValueError(f"cycles_dist must be one of {CYCLE_DISTS}, got {self.cycles_dist!r}")
        if self.geometry == "ring" and self.membrane_nm is None:
            raise ValueError("a ring null needs membrane_nm")
        for name in ("c_label", "locs_per_cycle", "period_nm", "slab_half_width_nm"):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"{name} must be > 0")
        if self.cycles_dist == "geometric" and float(self.c_label) < 1.0:
            raise ValueError("a geometric cycle count needs c_label >= 1")
        if float(self.locs_per_cycle) < 1.0:
            raise ValueError("locs_per_cycle is a mean of a count >= 1")

    @property
    def locs_per_molecule(self) -> float:
        """E[localizations per molecule] = labels x p_label x c_label x loc/cycle."""
        return float(self.labels_per_molecule * self.p_label * self.c_label * self.locs_per_cycle)

    def membrane_length_nm(self) -> float:
        if self.membrane_nm is None:
            return float("nan")
        c = np.asarray(self.membrane_nm, dtype=np.float64)
        return float(np.sum(np.hypot(*(np.roll(c, -1, axis=0) - c).T)))

    def z_span(self) -> Tuple[float, float]:
        """The axial span the molecules and the background occupy: the three rings +/- half a period."""
        half = 0.5 * self.n_rings * self.period_nm
        return self.z_centre_nm - half, self.z_centre_nm + half

    def slab_capture_per_molecule(self) -> float:
        """Expected slab localizations per molecule of EACH ring, summed over the rings (the ring's own share of
        a Gaussian axial error plus its neighbours' leak; uniform in z for "roi2d") [derivation]."""
        hw = self.slab_half_width_nm
        sz = math.hypot(self.sigma_z_nm, 0.5 * self.epitope_radius_nm, self.sigma_link_nm)
        if self.geometry == "roi2d":
            lo, hi = self.z_span()
            share = 2.0 * hw / (hi - lo) * self.n_rings
        else:
            share = 0.0
            for k in range(self.n_rings):
                dz = (k - 0.5 * (self.n_rings - 1)) * self.period_nm
                share += float(norm.cdf((hw - dz) / sz) - norm.cdf((-hw - dz) / sz))
        return float(self.locs_per_molecule * share)

    def background_count_mean(self) -> float:
        """E[background localizations] over the region and the whole z span, at the slab's areal density."""
        lo, hi = self.z_span()
        area_um2 = polygon_area(self.region_nm) / 1e6
        return float(self.background_per_um2 * area_um2 * (hi - lo) / (2.0 * self.slab_half_width_nm))


@dataclass
class NullAxon:
    """One simulated axon: localizations in frame order with their truth (-1 for the background)."""
    x_nm: FloatArray
    y_nm: FloatArray
    z_nm: FloatArray
    frame: NDArray[np.int64]
    lp_nm: FloatArray
    molecule: NDArray[np.int64]
    label: NDArray[np.int64]
    ring: NDArray[np.int64]
    molecule_xy_nm: FloatArray
    molecule_ring: NDArray[np.int64]
    n_background: int

    @property
    def n_locs(self) -> int:
        return int(self.x_nm.size)


def _stochastic_round(value: float, rng: np.random.Generator) -> int:
    base = math.floor(max(float(value), 0.0))
    return int(base + (1 if rng.uniform() < (float(value) - base) else 0))


def _hard_core_2d(rng: np.random.Generator, n: int, polygon: FloatArray, d_min_nm: float) -> FloatArray:
    """n points uniform in the polygon by sequential inhibition at d_min (RuntimeError when it jams)."""
    pts: FloatArray = np.empty((0, 2), dtype=np.float64)
    attempts = 0
    while pts.shape[0] < n:
        cand = simulate_csr_polygon(1, polygon, rng)
        attempts += 1
        if attempts > HARD_CORE_MAX_ATTEMPTS * max(n, 1):
            raise RuntimeError(f"2D hard core of {d_min_nm} nm jammed after {pts.shape[0]} of {n} molecules")
        if d_min_nm > 0 and pts.shape[0] and float(np.min(np.hypot(*(pts - cand[0]).T))) < d_min_nm:
            continue
        pts = np.vstack([pts, cand])
    return pts


POOL_FACTOR = 1.6                     # [ad hoc] pool of molecules per ring under the N_loc conditioning, x the closed mean
POOL_EXTRA = 6                        # [ad hoc] + this many (small rings)


def _pool_size(model: NullModel, length_nm: float) -> int:
    """Molecules per ring drawn when the replicate is conditioned on the slab's localizations: enough to reach the
    target with a margin (the per-molecule yield of geometric cycles is heavy-tailed), never more than the hard core
    can pack (JAM_FRACTION of the ring)."""
    n = int(math.ceil(POOL_FACTOR * float(model.n_molecules_per_ring))) + POOL_EXTRA
    if model.hard_core_nm > 0 and math.isfinite(length_nm):
        n = min(n, max(1, int(JAM_FRACTION * length_nm / float(model.hard_core_nm))))
    return n


def simulate_null_axon(model: NullModel, rng: np.random.Generator,
                       molecule_arcs: Optional[Sequence[FloatArray]] = None) -> NullAxon:
    """One replicate of the null (module docstring), in a FIXED order of draws: per ring the molecule count, arcs
    and radial offsets; then labels (count, epitope disc, linkage), cycles, localizations per cycle, start frames,
    lateral and axial errors; then the background. ``molecule_arcs`` (one array per ring, nm along the membrane)
    replaces the random arcs -- for planting structure in synthetic tests, never in the null itself."""
    ring_z = model.z_centre_nm + (np.arange(model.n_rings, dtype=np.float64) - 0.5 * (model.n_rings - 1)) * model.period_nm
    conditioned = model.condition_slab_locs is not None and molecule_arcs is None
    xy_parts: List[FloatArray] = []
    z_parts: List[FloatArray] = []
    tan_parts: List[FloatArray] = []
    ring_parts: List[NDArray[np.int64]] = []
    if model.geometry == "ring":
        path = smooth_closed_path(np.asarray(model.membrane_nm, dtype=np.float64))
        length = float(path.length_nm)
        for k in range(model.n_rings):
            if molecule_arcs is not None:
                s = np.asarray(molecule_arcs[k], dtype=np.float64).reshape(-1) % length
            else:
                n = _pool_size(model, length) if conditioned else _stochastic_round(model.n_molecules_per_ring, rng)
                s = _hard_core_arcs(rng, n, length, float(model.hard_core_nm), np.zeros(0),
                                    max_attempts=HARD_CORE_MAX_ATTEMPTS)
            r_off = rng.normal(0.0, float(model.radial_sd_nm), s.size)
            point, tangent, normal_out = _frame_at_arc(path, s)
            xy_parts.append(point + r_off[:, None] * normal_out)
            tan_parts.append(tangent)
            z_parts.append(np.full(s.size, ring_z[k]))
            ring_parts.append(np.full(s.size, k, dtype=np.int64))
    else:
        lo, hi = model.z_span()
        n_total = (_pool_size(model, float("inf")) * model.n_rings if conditioned
                   else _stochastic_round(model.n_molecules_per_ring * model.n_rings, rng))
        xy = _hard_core_2d(rng, n_total, model.region_nm, float(model.hard_core_nm))
        zz = rng.uniform(lo, hi, n_total)
        ang = rng.uniform(0.0, 2.0 * np.pi, n_total)
        xy_parts.append(xy)
        z_parts.append(zz)
        tan_parts.append(np.column_stack([np.cos(ang), np.sin(ang)]))
        ring_parts.append(np.clip(np.round((zz - ring_z[0]) / model.period_nm), 0, model.n_rings - 1).astype(np.int64))
    mol_xy = np.vstack(xy_parts) if xy_parts else np.zeros((0, 2))
    mol_z = np.concatenate(z_parts) if z_parts else np.zeros(0)
    mol_t = np.vstack(tan_parts) if tan_parts else np.zeros((0, 2))
    mol_ring = np.concatenate(ring_parts) if ring_parts else np.zeros(0, dtype=np.int64)
    n_mol = int(mol_xy.shape[0])
    if model.geometry == "ring":
        mol_layer = (np.concatenate([np.arange(p.shape[0], dtype=np.int64) for p in xy_parts]) if xy_parts
                     else np.zeros(0, dtype=np.int64))
    else:
        mol_layer = np.arange(n_mol, dtype=np.int64) // max(int(model.n_rings), 1)
    mol_n = np.column_stack([-mol_t[:, 1], mol_t[:, 0]]) if n_mol else np.zeros((0, 2))
    # labels: Binomial(labels_per_molecule, p_label) per molecule, each on the epitope disc (membrane plane:
    # tangent x axial) plus a 3D linkage drawn ONCE per label
    n_lab_mol = np.asarray(rng.binomial(int(model.labels_per_molecule), float(model.p_label), n_mol), dtype=np.int64)
    lab_mol = np.repeat(np.arange(n_mol, dtype=np.int64), n_lab_mol)
    n_lab = int(lab_mol.size)
    rho = float(model.epitope_radius_nm) * np.sqrt(rng.uniform(size=n_lab))
    phi = 2.0 * np.pi * rng.uniform(size=n_lab)
    d_t, d_z = rho * np.cos(phi), rho * np.sin(phi)
    link = rng.normal(0.0, float(model.sigma_link_nm), (n_lab, 3))
    lab_xy = mol_xy[lab_mol] + (d_t + link[:, 0])[:, None] * mol_t[lab_mol] + link[:, 1][:, None] * mol_n[lab_mol]
    lab_z = mol_z[lab_mol] + d_z + link[:, 2]
    # cycles per label (all at the label's position), localizations per cycle in consecutive frames
    if model.cycles_dist == "geometric":
        n_cyc = np.asarray(rng.geometric(1.0 / float(model.c_label), n_lab), dtype=np.int64)
    else:
        n_cyc = np.asarray(rng.poisson(float(model.c_label), n_lab), dtype=np.int64)
    cyc_label = np.repeat(np.arange(n_lab, dtype=np.int64), n_cyc)
    n_c = int(cyc_label.size)
    per_cyc = np.asarray(rng.geometric(1.0 / float(model.locs_per_cycle), n_c), dtype=np.int64)
    span = np.minimum(per_cyc, int(model.n_frames))
    start = np.asarray(rng.integers(0, int(model.n_frames) - span + 1), dtype=np.int64) if n_c else np.zeros(0, np.int64)
    loc_cyc = np.repeat(np.arange(n_c, dtype=np.int64), span)
    first = np.cumsum(span) - span
    frame = start[loc_cyc] + (np.arange(loc_cyc.size, dtype=np.int64) - first[loc_cyc])
    loc_label = cyc_label[loc_cyc]
    n_loc = int(loc_label.size)
    x = lab_xy[loc_label, 0] + rng.normal(size=n_loc) * float(model.sigma_xy_nm)
    y = lab_xy[loc_label, 1] + rng.normal(size=n_loc) * float(model.sigma_xy_nm)
    z = lab_z[loc_label] + rng.normal(size=n_loc) * float(model.sigma_z_nm)
    # background: uniform in the region and the z span at the slab's areal density
    n_bg = int(rng.poisson(model.background_count_mean()))
    lo, hi = model.z_span()
    bg_xy = simulate_csr_polygon(n_bg, model.region_nm, rng) if n_bg else np.zeros((0, 2))
    bg_z = rng.uniform(lo, hi, n_bg)
    bg_frame = np.asarray(rng.integers(0, int(model.n_frames), n_bg), dtype=np.int64)
    xs = np.concatenate([x, bg_xy[:, 0]])
    ys = np.concatenate([y, bg_xy[:, 1]])
    zs = np.concatenate([z, bg_z])
    fr = np.concatenate([frame, bg_frame])
    mol = np.concatenate([lab_mol[loc_label], np.full(n_bg, -1, dtype=np.int64)])
    lab = np.concatenate([loc_label, np.full(n_bg, -1, dtype=np.int64)])
    rg = np.concatenate([mol_ring[lab_mol[loc_label]] if n_loc else np.zeros(0, np.int64),
                         np.full(n_bg, -1, dtype=np.int64)])
    if conditioned and n_mol:
        # keep the first n layers whose slab localizations (+ the background's) come closest to the target
        in_slab = select_mps_slab(zs, model.z_centre_nm, model.slab_half_width_nm)
        need = float(model.condition_slab_locs) - float(np.count_nonzero(in_slab & (mol < 0)))
        per_layer = np.bincount(mol_layer[mol[in_slab & (mol >= 0)]], minlength=int(mol_layer.max()) + 1)
        cum = np.concatenate([[0.0], np.cumsum(per_layer, dtype=np.float64)])
        n_keep = int(np.argmin(np.abs(cum - need)))
        keep_mol = mol_layer < n_keep
        keep = (mol < 0) | keep_mol[np.maximum(mol, 0)]
        xs, ys, zs, fr, mol, lab, rg = (a[keep] for a in (xs, ys, zs, fr, mol, lab, rg))
        mol_xy, mol_ring = mol_xy[keep_mol], mol_ring[keep_mol]
    order = np.argsort(fr, kind="stable")
    return NullAxon(x_nm=xs[order], y_nm=ys[order], z_nm=zs[order], frame=fr[order],
                    lp_nm=np.full(xs.size, float(model.sigma_xy_nm)), molecule=mol[order], label=lab[order],
                    ring=rg[order], molecule_xy_nm=mol_xy, molecule_ring=mol_ring, n_background=n_bg)


# ============================================================================
# Reading the model off the observed axon
# ============================================================================

def locs_per_cycle_from_events(frame: NDArray[np.int64], x_nm: FloatArray, y_nm: FloatArray, nena_nm: float,
                               labels: Optional[NDArray[np.int64]] = None) -> Tuple[Optional[float], int]:
    """Mean localizations per switching event, by event linking (radius 3 NeNA, 1 dark frame; Picasso's rule via
    ``tools.mps_paint.link_localizations``). The mean is the maximum-likelihood mean of the geometric count the
    generator draws. With ``labels`` the events never span two clusters and the noise (-1) is left out (the
    background's single localizations would pull the mean towards 1). Returns (mean or None, events counted).

    Corrected for the links the radius breaks [derivation]: two localizations of one emitter differ by a vector of
    sd sqrt(2) sigma per axis, so their distance exceeds a radius rho with probability q = exp(-rho^2 / (4 sigma^2))
    (0.105 at rho = 3 sigma); an event of true mean length mu then splits into 1 + Binomial(L - 1, q) pieces of mean
    m = mu / (1 + q (mu - 1)), hence mu = m (1 - q) / (1 - m q). Uncorrected, the synthetic axons read 1.80 for a true
    2.0 [simulation: calibration_null.jsonl of the first run], the null then needed 11 % more molecules to match the
    slab's localizations and rejected K per um in 14 of 60 axons under a true null."""
    from tools.mps_paint import link_localizations
    f = np.asarray(frame, dtype=np.int64).reshape(-1)
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    keep = np.ones(f.size, dtype=bool) if labels is None else (np.asarray(labels).reshape(-1) >= 0)
    if int(np.count_nonzero(keep)) < 2 or not nena_nm > 0:
        return None, 0
    group = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)[keep]
    ev = link_localizations(f[keep], x[keep], y[keep], LINK_RADIUS_IN_NENA * float(nena_nm),
                            max_dark_time=LINK_MAX_DARK_FRAMES, group=group)
    counts = np.bincount(ev)
    counts = counts[counts > 0]
    if counts.size == 0:
        return None, 0
    m = float(np.mean(counts))
    q = math.exp(-(LINK_RADIUS_IN_NENA ** 2) / 4.0)
    mu = m * (1.0 - q) / max(1.0 - m * q, 0.1)       # the floor only bites for m > 8.5 localizations per event
    return float(max(mu, 1.0)), int(counts.size)


def _kept_labels(analysis: AxonAnalysis) -> List[int]:
    return [int(v) for v in good_cluster_labels(analysis.labels, set(analysis.bad_report.bad_labels))]


def _median_tangential_sd(analysis: AxonAnalysis, membrane: FloatArray, min_locs: int = 10) -> Optional[float]:
    """Median over kept clusters of the sd of their localizations ALONG the membrane tangent at the centroid."""
    path = smooth_closed_path(membrane)
    s, _d, _len = arc_coordinates(analysis.centroids, np.asarray(path.points_nm))
    _p, tangent, _n = _frame_at_arc(path, s)
    sds: List[float] = []
    for i, lab in enumerate(_kept_labels(analysis)):
        sel = analysis.labels == lab
        if int(np.count_nonzero(sel)) < min_locs:
            continue
        dx = analysis.x_slab[sel] - analysis.centroids[i, 0]
        dy = analysis.y_slab[sel] - analysis.centroids[i, 1]
        sds.append(float(np.std(dx * tangent[i, 0] + dy * tangent[i, 1], ddof=1)))
    return float(np.median(sds)) if sds else None


CLOUD_MOLECULES = 4000              # [ad hoc] molecules of the cloud Monte Carlo (median sd to ~1 %)
CLOUD_SEED = 20261022               # [ad hoc] fixed: the read values do not depend on the null's seed


def cloud_statistics(*, epitope_radius_nm: float, sigma_link_nm: float, sigma_xy_nm: float, c_label: float,
                     locs_per_cycle: float, labels_per_molecule: int, p_label: float, cycles_dist: str,
                     min_locs: int, n_molecules: int = CLOUD_MOLECULES, seed: int = CLOUD_SEED) -> Tuple[float, float]:
    """(median tangential sd of the localizations of ONE molecule about their centroid, sd of that centroid along
    the membrane normal) over simulated isolated molecules with at least ``min_locs`` localizations -- what the
    clustering keeps [simulation, the null's own label / cycle model]. Used to read the epitope radius off the
    observed clusters' tangential spread and to remove the centroid noise from the observed radial scatter: with
    two labels per molecule and geometric cycles the analytic variances are dominated by one label, so the
    formulas of independent localizations do not apply (the first prototype read R = 0 for a true 25 nm)."""
    rng = np.random.default_rng(int(seed))
    n_lab = rng.binomial(int(labels_per_molecule), float(p_label), n_molecules)
    lab_mol = np.repeat(np.arange(n_molecules), n_lab)
    nl = lab_mol.size
    rho = float(epitope_radius_nm) * np.sqrt(rng.uniform(size=nl))
    t_lab = rho * np.cos(2.0 * np.pi * rng.uniform(size=nl)) + rng.normal(0.0, sigma_link_nm, nl)
    n_lab_pos = rng.normal(0.0, sigma_link_nm, nl)
    if cycles_dist == "geometric":
        cyc = rng.geometric(1.0 / float(c_label), nl)
    else:
        cyc = rng.poisson(float(c_label), nl)
    cyc_lab = np.repeat(np.arange(nl), cyc)
    per = rng.geometric(1.0 / float(locs_per_cycle), cyc_lab.size)
    loc_lab = np.repeat(cyc_lab, per)
    loc_mol = lab_mol[loc_lab]
    t = t_lab[loc_lab] + rng.normal(0.0, sigma_xy_nm, loc_lab.size)
    nn = n_lab_pos[loc_lab] + rng.normal(0.0, sigma_xy_nm, loc_lab.size)
    cnt = np.bincount(loc_mol, minlength=n_molecules).astype(np.float64)
    ok = cnt >= max(int(min_locs), 2)
    if not np.any(ok):
        return float("nan"), float("nan")
    st = np.bincount(loc_mol, weights=t, minlength=n_molecules)
    st2 = np.bincount(loc_mol, weights=t * t, minlength=n_molecules)
    sn = np.bincount(loc_mol, weights=nn, minlength=n_molecules)
    c = cnt[ok]
    var_t = (st2[ok] - st[ok] ** 2 / c) / (c - 1.0)
    cen_n = sn[ok] / c
    return float(np.median(np.sqrt(np.maximum(var_t, 0.0)))), float(np.std(cen_n, ddof=1))


BACKGROUND_MIN_DISTANCE_NM = 200.0    # [ad hoc; above the worst centroid-membrane error seen in the calibration, 133 nm]
BACKGROUND_EXCLUSION_NM = 100.0       # [ad hoc: tools.mps_simulate_axon.BACKGROUND_EXCLUSION_NM] discs around centroids
BACKGROUND_GRID_NM = 10.0             # [ad hoc] grid of the free-area measurement (0.01 um^2 per 100 cells)


def _background_density(pts: FloatArray, region: FloatArray, membrane: FloatArray, centroids: FloatArray,
                        band_half_width_nm: float) -> Tuple[float, float]:
    """(localizations per um^2, free area in um^2): the slab localizations, and the grid cells, that lie inside the
    region and the slab's convex hull, farther than ``band_half_width_nm`` from the membrane and farther than
    ``BACKGROUND_EXCLUSION_NM`` from every centroid."""
    hull = pts[ConvexHull(pts).vertices]
    lo = np.maximum(region.min(axis=0), hull.min(axis=0))
    hi = np.minimum(region.max(axis=0), hull.max(axis=0))
    if np.any(hi <= lo):
        return 0.0, 0.0
    gx = np.arange(lo[0] + 0.5 * BACKGROUND_GRID_NM, hi[0], BACKGROUND_GRID_NM)
    gy = np.arange(lo[1] + 0.5 * BACKGROUND_GRID_NM, hi[1], BACKGROUND_GRID_NM)
    grid = np.column_stack([np.repeat(gx, gy.size), np.tile(gy, gx.size)])

    def free(q: FloatArray) -> NDArray[np.bool_]:
        ok = point_in_polygon(q[:, 0], q[:, 1], region) & point_in_polygon(q[:, 0], q[:, 1], hull)
        ok &= _band_distance(q, membrane) > band_half_width_nm
        if centroids.shape[0]:
            d, _ = cKDTree(centroids).query(q, k=1)
            ok &= d > BACKGROUND_EXCLUSION_NM
        return np.asarray(ok, dtype=bool)

    area_um2 = float(np.count_nonzero(free(grid))) * BACKGROUND_GRID_NM ** 2 / 1e6
    if area_um2 <= 0:
        return 0.0, 0.0
    return float(np.count_nonzero(free(pts))) / area_um2, area_um2


def _band_distance(points: FloatArray, curve: FloatArray) -> FloatArray:
    """Distance of each point to a closed curve (dense: resampled every ~2 nm along it)."""
    path = smooth_closed_path(curve)
    dense = np.asarray(path.points_nm, dtype=np.float64)
    d, _ = cKDTree(dense).query(np.asarray(points, dtype=np.float64), k=1)
    return np.asarray(d, dtype=np.float64)


def null_model_from_analysis(
    analysis: AxonAnalysis,
    *,
    nena_nm: Optional[float] = None,
    frame_slab: Optional[NDArray[np.int64]] = None,
    n_frames: Optional[int] = None,
    c_label: float = DEFAULT_C_LABEL,
    locs_per_cycle: Optional[float] = None,
    hard_core_nm: Optional[float] = None,
    epitope_radius_nm: Optional[float] = None,
    sigma_z_nm: Optional[float] = None,
    sigma_link_nm: float = DEFAULT_SIGMA_LINK_NM,
    labels_per_molecule: int = DEFAULT_LABELS_PER_MOLECULE,
    p_label: float = DEFAULT_P_LABEL,
    cycles_dist: str = "geometric",
    no_ring: bool = False,
) -> NullModel:
    """The null model of one observed ``AxonAnalysis`` (A9 sec. 3.1; regularity plan sec. 4.4). Every argument
    left None is READ from the axon (and labelled "read" in ``origins``) or falls back to its documented default.

    ``nena_nm``: the file's lateral NeNA (the generator's lateral error and the event-linking radius); None falls
    back to 12 nm [lit: Gazal et al. 2026], labelled as not this file's. ``frame_slab``: the frame of each SLAB localization (aligned with ``analysis.x_slab``),
    for loc/cycle; ``analysis.localization_frame_slab`` is used when the analysis kept it."""
    origins: Dict[str, str] = {}
    if nena_nm is None or not (nena_nm > 0):
        nena, origins["sigma_xy_nm"] = 12.0, "lit: Gazal et al. 2026 (no NeNA given for this file)"
    else:
        nena, origins["sigma_xy_nm"] = float(nena_nm), "read: the file's NeNA"
    per = analysis.perimeter
    ring_ok = (not no_ring) and per is not None and analysis.n_clusters_kept >= 3
    # the region: the ROI the observed was curated against, else the convex hull of its slab
    region = _roi_polygon(analysis.roi)
    region_source = "roi"
    pts = np.column_stack([analysis.x_slab, analysis.y_slab])
    if region is None:
        if pts.shape[0] >= 3:
            region = pts[ConvexHull(pts).vertices]
            region_source = "convex hull of the slab"
        else:
            raise NullRefused("fewer than 3 localizations in the slab: no region to simulate in")
    membrane: Optional[FloatArray] = None
    scatter = float("nan")
    if ring_ok:
        assert per is not None
        fit = radial_scatter_nm(np.asarray(per.contour, dtype=np.float64), knot_spacing_nm=MEMBRANE_KNOT_SPACING_NM)
        scatter = float(fit.sigma_nm)
        membrane = np.asarray(fit.curve_nm, dtype=np.float64)
        origins["membrane_nm"] = ("read: 600 nm P-spline through the observed centroids (P-R80 a)" if math.isfinite(scatter)
                                  else "read: the centroid tour itself (P-spline not identifiable)")
        if no_ring is False and region_source != "roi":
            # without an ROI the background region is the membrane's box with the simulator's margin
            lo = membrane.min(axis=0) - BACKGROUND_MARGIN_NM
            hi = membrane.max(axis=0) + BACKGROUND_MARGIN_NM
            region = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            region_source = f"membrane box + {BACKGROUND_MARGIN_NM:.0f} nm (no ROI)"
    origins["region_nm"] = region_source
    # hard core: contact, 2 x median r_eff (P-R77 a), editable (e)
    if hard_core_nm is None:
        r_eff = analysis.median_r_eff_nm
        hard_core = 2.0 * float(r_eff) if r_eff is not None else 0.0
        origins["hard_core_nm"] = "read: contact, 2 x median r_eff (P-R77 a)" if r_eff is not None else "ad hoc: 0 (no r_eff)"
    else:
        hard_core, origins["hard_core_nm"] = float(hard_core_nm), "user"
    # frames and loc/cycle (before the cloud statistics, which depend on them)
    fr = frame_slab if frame_slab is not None else analysis.localization_frame_slab
    if n_frames is None:
        if fr is not None and np.asarray(fr).size:
            n_frames_v, origins["n_frames"] = int(np.max(fr)) + 1, "read: last frame + 1"
        else:
            n_frames_v, origins["n_frames"] = DEFAULT_N_FRAMES, "ad hoc: simulator default"
    else:
        n_frames_v, origins["n_frames"] = int(n_frames), "user"
    if locs_per_cycle is None:
        lpc: Optional[float] = None
        if fr is not None and np.asarray(fr).size == analysis.x_slab.size:
            keep = np.zeros(analysis.labels.size, dtype=np.int64) - 1
            kept = set(_kept_labels(analysis))
            mask = np.isin(analysis.labels, list(kept))
            keep[mask] = analysis.labels[mask]
            lpc, n_ev = locs_per_cycle_from_events(np.asarray(fr), analysis.x_slab, analysis.y_slab, nena, labels=keep)
            if lpc is not None:
                origins["locs_per_cycle"] = f"read: mean localizations per linked event ({n_ev} events; P-R79 b)"
        if lpc is None:
            lpc, origins["locs_per_cycle"] = DEFAULT_LOCS_PER_CYCLE, "ad hoc: default 2 (no frames; P-R79 a)"
    else:
        lpc, origins["locs_per_cycle"] = float(locs_per_cycle), "user"
    lpc = max(float(lpc), 1.0)
    cloud_kw: Dict[str, Any] = dict(sigma_link_nm=float(sigma_link_nm), sigma_xy_nm=nena, c_label=float(c_label),
                                    locs_per_cycle=lpc, labels_per_molecule=int(labels_per_molecule),
                                    p_label=float(p_label), cycles_dist=cycles_dist, min_locs=int(analysis.min_samples))
    # epitope disc radius: NOT read (deviation from the regularity plan sec. 4.4, which read it from the clusters'
    # tangential spread). Under this blinking model the spread hardly moves with R -- the median tangential sd of
    # simulated isolated molecules is 13.0 / 13.8 / 15.0 nm at R = 0 / 25 / 40 nm (NeNA 12, sigma_link 12, c_label
    # 14) [simulation: cloud_statistics] -- while merged clusters add more than that, so a read R is mostly the
    # merges (a synthetic axon with R = 25 read 54 nm). The observed spread is kept as a diagnostic.
    if epitope_radius_nm is None:
        sd_t = _median_tangential_sd(analysis, membrane) if membrane is not None else None
        epi = DEFAULT_EPITOPE_RADIUS_NM
        origins["epitope_radius_nm"] = ("ad hoc: simulator default 25 nm (not identifiable from the clusters' spread"
                                        + ("" if sd_t is None else f"; observed median tangential sd {sd_t:.1f} nm against "
                                           f"{cloud_statistics(epitope_radius_nm=epi, **cloud_kw)[0]:.1f} nm for "
                                           "isolated simulated molecules") + ")")
    else:
        epi, origins["epitope_radius_nm"] = float(epitope_radius_nm), "user"
    # radial offset of the molecules: the observed centroid scatter minus the centroid noise of one molecule
    if math.isfinite(scatter):
        noise = cloud_statistics(epitope_radius_nm=epi, **cloud_kw)[1]
        radial_sd = math.sqrt(max(scatter ** 2 - (noise if math.isfinite(noise) else 0.0) ** 2, 0.0))
        origins["radial_sd_nm"] = (f"read: centroid scatter {scatter:.1f} nm about the 600 nm membrane minus the "
                                   f"simulated centroid noise of one molecule {noise:.1f} nm, in quadrature")
    else:
        radial_sd, origins["radial_sd_nm"] = 0.0, "ad hoc: 0 (scatter not identifiable)"
    # axial: the slab ring's GMM component sd, minus the label offsets in quadrature
    zr = analysis.z_result
    centre = 0.5 * (analysis.slab_zmin_nm + analysis.slab_zmax_nm)
    if sigma_z_nm is None:
        sig = np.asarray(zr.sigmas_nm, dtype=np.float64)
        means = np.asarray(zr.means_nm, dtype=np.float64)
        if sig.size and np.all(np.isfinite(sig)):
            comp = float(sig[int(np.argmin(np.abs(means - centre)))])
            sz = math.sqrt(max(comp ** 2 - (0.5 * epi) ** 2 - sigma_link_nm ** 2, (0.5 * comp) ** 2))
            origins["sigma_z_nm"] = f"read: slab component sd {comp:.0f} nm minus label offsets (derivation)"
        else:
            sz, origins["sigma_z_nm"] = DEFAULT_SIGMA_Z_NM, "ad hoc: no axial component"
    else:
        sz, origins["sigma_z_nm"] = float(sigma_z_nm), "user"
    period = analysis.mean_delta_z_nm
    if period is None or not (period > 0):
        period, origins["period_nm"] = DEFAULT_PERIOD_NM, "lit: Gazal et al. 2026, Delta-Z 170 nm"
    else:
        period, origins["period_nm"] = float(period), "read: mean Delta-Z of the axon"
    origins["c_label"] = ("lit: Dempsey et al. 2011, AF647 MEA (provisional)" if c_label == DEFAULT_C_LABEL else
                          "lit: Dempsey et al. 2011, AF647 betaME (sensitivity)" if c_label == SENSITIVITY_C_LABEL else "user")
    # background: slab localizations beyond +/-(50 + 3 sigma) of the membrane AND beyond 100 nm of every raw
    # cluster's centroid, per um^2 of the area meeting the same conditions inside the region and the slab's convex hull
    # (measured on a 10 nm grid). The centroid exclusion makes the estimate robust to the method: with the band
    # alone, a contour distorted by off-membrane clusters turned molecule localizations into "background" (HDBSCAN
    # read 11.6 +/- 15.4 per um^2 for a true 2 [simulation: docs/sims/pipeline_null]); the simulator excludes the
    # same 100 nm discs (tools.mps_simulate_axon.BACKGROUND_EXCLUSION_NM).
    # and never closer than BACKGROUND_MIN_DISTANCE_NM: the centroid membrane of a method that leaves gaps cuts
    # corners (133 nm from the true membrane on a synthetic HDBSCAN axon, which read 15 per um^2 for a true 2 at
    # +/-86 nm and 2.8 at 200 nm) [simulation: docs/sims/pipeline_null]
    hw_bg = max(BAND_HALF_WIDTH_NM + BACKGROUND_BAND_SIGMAS * nena, BACKGROUND_MIN_DISTANCE_NM)
    bg = 0.0
    if membrane is not None and pts.shape[0] >= 3:
        # every RAW cluster's centroid, the curated-away ones included: their localizations are molecules too (a
        # synthetic HDBSCAN axon had 250 of 279 "background" localizations in the 11 clusters its curation removed)
        lab = np.asarray(analysis.labels, dtype=np.int64)
        raw = lab >= 0
        if np.any(raw):
            cnt = np.bincount(lab[raw]).astype(np.float64)
            has = cnt > 0
            cx = np.bincount(lab[raw], weights=analysis.x_slab[raw])[has] / cnt[has]
            cy = np.bincount(lab[raw], weights=analysis.y_slab[raw])[has] / cnt[has]
            all_centroids = np.column_stack([cx, cy])
        else:
            all_centroids = np.zeros((0, 2))
        bg, area_um2 = _background_density(pts, region, membrane, all_centroids, hw_bg)
        origins["background_per_um2"] = (f"read: slab localizations beyond +/-{hw_bg:.0f} nm of the membrane and "
                                         f"{BACKGROUND_EXCLUSION_NM:.0f} nm of every raw cluster's centroid, per um^2 of "
                                         f"that area ({area_um2:.2f} um^2)")
    else:
        origins["background_per_um2"] = "ad hoc: 0 (no ring: background folded into the molecules)"
    model = NullModel(
        geometry="ring" if ring_ok else "roi2d", membrane_nm=membrane, region_nm=np.asarray(region, dtype=np.float64),
        z_centre_nm=float(centre), period_nm=float(period), slab_half_width_nm=float(analysis.slab_half_width_nm),
        n_molecules_per_ring=1.0, hard_core_nm=float(hard_core), radial_sd_nm=float(radial_sd), sigma_xy_nm=nena,
        sigma_z_nm=float(sz), background_per_um2=float(bg), n_frames=int(n_frames_v), c_label=float(c_label),
        locs_per_cycle=float(lpc), labels_per_molecule=int(labels_per_molecule), p_label=float(p_label),
        epitope_radius_nm=float(epi), sigma_link_nm=float(sigma_link_nm), cycles_dist=cycles_dist, origins=origins)
    # the first guess of the molecules, from the slab's localizations (refined by the closure)
    n_bg_slab = bg * polygon_area(model.region_nm) / 1e6
    model.n_molecules_per_ring = max(1.0, (analysis.n_locs_slab - n_bg_slab) / max(model.slab_capture_per_molecule(), 1e-9))
    origins["n_molecules_per_ring"] = "closure (first guess from the slab's localizations)"
    return model


def check_packing(model: NullModel) -> None:
    """Refuse a hard core that random sequential packing cannot honour on this ring (Renyi's constant 0.7476 of
    the length; refused above 0.70 [ad hoc])."""
    if model.geometry != "ring" or model.hard_core_nm <= 0:
        return
    length = float(smooth_closed_path(np.asarray(model.membrane_nm)).length_nm)
    fill = model.n_molecules_per_ring * model.hard_core_nm / length
    if fill > JAM_FRACTION:
        raise NullRefused(f"the contact hard core {model.hard_core_nm:.0f} nm x {model.n_molecules_per_ring:.0f} molecules "
                          f"would cover {fill:.0%} of the {length / 1000.0:.2f} um ring (random packing jams near "
                          f"{RSA_JAMMING:.0%}): the file has far more localizations per cluster than labels x c_label x loc/cycle "
                          "predicts. Check c_label and loc/cycle, lower the hard core, or close on the detected "
                          "clusters (closure='k_clusters', the sensitivity analysis).")


# ============================================================================
# Readouts and the replicate view
# ============================================================================

@dataclass
class ReplicateView:
    """What a readout function sees of one analysed axon (observed or simulated): the analysis, its kept
    centroids, the 600 nm P-spline membrane through them (ring mode), the arc position of each centroid on it."""
    analysis: AxonAnalysis
    centroids: FloatArray
    membrane_nm: Optional[FloatArray]
    membrane_length_nm: float
    arc_s_nm: Optional[FloatArray]
    radial_scatter_nm: float
    is_observed: bool = False


ReadoutFn = Callable[[ReplicateView], Optional[float]]


@dataclass(frozen=True)
class ReadoutSpec:
    """A scalar readout tested against the null: name (column suffix ``p_mc_<name>``), function, unit,
    whether it needs a ring (contour), two-sided test, and a one-line description with its origin."""
    name: str
    fn: ReadoutFn
    unit: str = ""
    needs_ring: bool = True
    two_sided: bool = True
    description: str = ""
    depends_on_closure: bool = False


def _ro_k_per_um(v: ReplicateView) -> Optional[float]:
    return v.analysis.clusters_per_um


def _ro_nn(v: ReplicateView) -> Optional[float]:
    return v.analysis.median_1nn_nm


def _ro_reff(v: ReplicateView) -> Optional[float]:
    return v.analysis.median_r_eff_nm


def _ro_occ(v: ReplicateView) -> Optional[float]:
    return v.analysis.occupancy_percent


def _ro_reg(v: ReplicateView) -> Optional[float]:
    return v.analysis.nn_regularity_index


READOUTS: Dict[str, ReadoutSpec] = {}


def register_readout(spec: ReadoutSpec, *, replace: bool = False) -> None:
    """Add a readout to the null (the hook the regularity branch uses: gap CV, rho1, excess CV, Bragg peak).

    The function receives a ``ReplicateView`` -- the same for the observed axon and for every replicate -- and
    returns a number or None (not computable). For ``n_jobs > 1`` it must be a module-level function (pickled into
    the worker processes)."""
    if spec.name in READOUTS and not replace:
        raise ValueError(f"readout {spec.name!r} is already registered")
    READOUTS[spec.name] = spec


for _spec in (
    ReadoutSpec("k_per_um", _ro_k_per_um, "1/um", True, True,
                "detected clusters per um of contour (free under the N_loc closure; conditioned under the K one)",
                depends_on_closure=True),
    ReadoutSpec("nn_median", _ro_nn, "nm", False, True, "median 1NN between curated cluster centres"),
    ReadoutSpec("reff_median", _ro_reff, "nm", False, True, "median effective radius sqrt(area / pi)"),
    ReadoutSpec("occupancy", _ro_occ, "%", True, True, "perimeter occupancy (Mahalanobis ellipses)"),
    ReadoutSpec("regularity", _ro_reg, "", True, True, "regularity index r = median 1NN x K / P"),
):
    register_readout(_spec)


def _view(analysis: AxonAnalysis, ring: bool, observed: bool = False) -> ReplicateView:
    cen = np.asarray(analysis.centroids, dtype=np.float64).reshape(-1, 2)
    membrane: Optional[FloatArray] = None
    length, s, scatter = float("nan"), None, float("nan")
    if ring and analysis.perimeter is not None and cen.shape[0] >= 3:
        fit = radial_scatter_nm(np.asarray(analysis.perimeter.contour, dtype=np.float64),
                                knot_spacing_nm=MEMBRANE_KNOT_SPACING_NM)
        scatter = float(fit.sigma_nm)
        membrane = np.asarray(fit.curve_nm, dtype=np.float64)
        s, _d, length = arc_coordinates(cen, membrane)
    return ReplicateView(analysis=analysis, centroids=cen, membrane_nm=membrane, membrane_length_nm=float(length),
                         arc_s_nm=s, radial_scatter_nm=scatter, is_observed=observed)


def ripley_curves(view: ReplicateView, r_grid: FloatArray, window_2d: Optional[FloatArray] = None
                  ) -> Tuple[FloatArray, FloatArray]:
    """(H1 on the closed arc, H2 in the +/-50 nm band) of the centroids; H2 in ``window_2d`` (polygon, correction
    'none') when given -- the 2D fallback. NaN curves where not computable."""
    nan: FloatArray = np.full(np.asarray(r_grid).size, np.nan)
    h1 = nan.copy()
    if view.arc_s_nm is not None and view.arc_s_nm.size >= 2 and view.membrane_length_nm > 0:
        h1 = ripley_k_1d_arc(view.arc_s_nm, view.membrane_length_nm, r_grid, closed=True).h
    h2 = nan.copy()
    if view.centroids.shape[0] >= 2:
        if window_2d is not None:
            h2 = ripley_k_2d(view.centroids, window_2d, r_grid, correction="none").h
        elif view.membrane_nm is not None:
            area = 2.0 * BAND_HALF_WIDTH_NM * view.membrane_length_nm   # tube formula [derivation]
            h2 = ripley_k_2d_area(view.centroids, area, r_grid).h
    return np.asarray(h1, dtype=np.float64), np.asarray(h2, dtype=np.float64)


# ============================================================================
# One replicate (module level: picklable for n_jobs > 1)
# ============================================================================

@dataclass
class ReplicateOutcome:
    index: int
    ok: bool
    reason: str
    readouts: Dict[str, Optional[float]]
    h1: FloatArray
    h2: FloatArray
    n_locs_slab: int
    k_clusters: int
    radial_scatter_nm: float
    seconds: float
    example: Optional[Dict[str, Any]] = None
    arcs: Optional[FloatArray] = None           # keep_arcs: the arc positions of the kept centres
    arc_length_nm: float = float("nan")        # ... and the length of that replicate's membrane


def _rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=tuple(int(k) for k in key)))


def _run_replicate(model: NullModel, spec: PipelineSpec, seed: int, index: int, r_grid: FloatArray,
                   readouts: Sequence[ReadoutSpec], window_2d: Optional[FloatArray], keep_example: bool,
                   keep_arcs: bool = False
                   ) -> ReplicateOutcome:
    t0 = time.perf_counter()
    nan = np.full(r_grid.size, np.nan)
    try:
        ax = simulate_null_axon(model, _rng(seed, 0, index))
    except RuntimeError as exc:
        return ReplicateOutcome(index, False, f"generator: {exc}", {}, nan, nan, 0, 0, float("nan"),
                                time.perf_counter() - t0)
    try:
        an = spec.run(ax.x_nm, ax.y_nm, ax.z_nm, main_peak_nm=model.z_centre_nm, lp_nm=ax.lp_nm, frame=ax.frame,
                      source_name=f"null replicate {index}")
    except Exception as exc:                      # noqa: BLE001 - one failed replicate is reported, not fatal
        return ReplicateOutcome(index, False, f"pipeline: {type(exc).__name__}: {exc}", {}, nan, nan, 0, 0,
                                float("nan"), time.perf_counter() - t0)
    ring = model.geometry == "ring"
    view = _view(an, ring)
    values: Dict[str, Optional[float]] = {}
    for ro in readouts:
        if ro.needs_ring and not ring:
            continue
        try:
            v = ro.fn(view)
            values[ro.name] = None if v is None or not math.isfinite(float(v)) else float(v)
        except Exception:                          # noqa: BLE001 - a readout that fails is None for this replicate
            values[ro.name] = None
    h1, h2 = ripley_curves(view, r_grid, window_2d)
    example = None
    if keep_example:
        example = {"x_slab": an.x_slab, "y_slab": an.y_slab, "labels": an.labels, "centroids": view.centroids,
                   "cluster_method": an.cluster_method, "cluster_params": dict(an.cluster_params),
                   "contour": None if an.perimeter is None else an.perimeter.contour, "seed": int(seed),
                   "spawn_key": (0, int(index)), "molecule_xy_nm": ax.molecule_xy_nm}
    arcs = (np.asarray(view.arc_s_nm, dtype=np.float64).copy()
            if keep_arcs and ring and view.arc_s_nm is not None else None)
    return ReplicateOutcome(index, True, "", values, h1, h2, int(an.n_locs_slab), int(an.n_clusters_kept),
                            float(view.radial_scatter_nm), time.perf_counter() - t0, example,
                            arcs, float(view.membrane_length_nm) if arcs is not None else float("nan"))


def _replicate_star(args: Tuple[Any, ...]) -> ReplicateOutcome:
    return _run_replicate(*args)


# ============================================================================
# Closure
# ============================================================================

@dataclass
class ClosureRecord:
    closure: str
    target: float
    trajectory: List[Tuple[float, float]]     # (molecules per ring, simulated statistic) per pilot
    n_molecules_per_ring: float
    notes: List[str] = field(default_factory=list)


def _slab_count(model: NullModel, ax: NullAxon) -> Tuple[int, int]:
    """(slab localizations, slab background localizations) of a simulated axon, with the pipeline's own slab."""
    mask = select_mps_slab(ax.z_nm, model.z_centre_nm, model.slab_half_width_nm)
    return int(np.count_nonzero(mask)), int(np.count_nonzero(mask & (ax.molecule < 0)))


def close_null(model: NullModel, spec: PipelineSpec, *, closure: str, target: float, seed: int,
               n_pilots: int = N_PILOTS) -> ClosureRecord:
    """Solve the molecules per ring (A9 sec. 3.3; regularity plan PR 1: 4 pilots, damped update).

    "n_locs": slab localizations are linear in the molecules, so the yield per molecule is pooled over every
    pilot axon so far and N = (target - median background) / median yield (no clustering needed: cheap).
    "k_clusters": the detected K is a compressive function of N (fusion), so N <- N (K_obs / K_sim)^0.8 [ad hoc
    damping] with the K of ``PILOT_AXONS_K`` pipeline runs per pilot."""
    if closure not in CLOSURES:
        raise ValueError(f"closure must be one of {CLOSURES}, got {closure!r}")
    traj: List[Tuple[float, float]] = []
    notes: List[str] = []
    n = float(model.n_molecules_per_ring)
    if closure == "n_locs":
        yields: List[float] = []
        bg_counts: List[int] = []
        for i in range(n_pilots):
            m = dataclasses.replace(model, n_molecules_per_ring=n, condition_slab_locs=None)
            check_packing(m)
            stats = []
            for j in range(PILOT_AXONS_N_LOCS):
                ax = simulate_null_axon(m, _rng(seed, 1, i, j))
                tot, bgc = _slab_count(m, ax)
                stats.append(tot)
                yields.append((tot - bgc) / max(n, 1e-9))
                bg_counts.append(bgc)
            traj.append((n, float(np.median(stats))))
            # the MEDIAN slab count is what the closure must match (A9 sec. 3.3); the per-molecule yield pooled over
            # every pilot axon so far is a skewed variable (geometric cycles), so its median is used, not its mean
            n_new = (float(target) - float(np.median(bg_counts))) / max(float(np.median(yields)), 1e-9)
            n = max(1.0, n_new)
    else:
        for i in range(n_pilots):
            m = dataclasses.replace(model, n_molecules_per_ring=n, condition_slab_locs=None)
            check_packing(m)
            ks = []
            for j in range(PILOT_AXONS_K):
                ax = simulate_null_axon(m, _rng(seed, 1, i, j))
                try:
                    an = spec.run(ax.x_nm, ax.y_nm, ax.z_nm, main_peak_nm=m.z_centre_nm, lp_nm=ax.lp_nm, frame=ax.frame)
                    ks.append(float(an.n_clusters_kept))
                except Exception as exc:                  # noqa: BLE001
                    notes.append(f"pilot {i}.{j} failed: {exc}")
            k_sim = float(np.mean(ks)) if ks else float("nan")
            traj.append((n, k_sim))
            if not (k_sim > 0):
                break
            n = max(1.0, n * (float(target) / k_sim) ** K_CLOSURE_DAMPING)
    return ClosureRecord(closure=closure, target=float(target), trajectory=traj, n_molecules_per_ring=float(n),
                         notes=notes)


# ============================================================================
# The result
# ============================================================================

@dataclass
class FidelityCheck:
    name: str
    observed: float
    null_median: float
    tolerance: str
    met: Optional[bool]


@dataclass
class PipelineNullResult:
    """Everything one run of the null gives (the integration exports ``export_dict`` and draws the rest)."""
    method: str
    method_params: Dict[str, Any]
    closure: str
    null_type: str
    seed: int
    n_requested: int
    model: NullModel
    closure_record: ClosureRecord
    scale: WorkingScale
    r_grid: FloatArray
    geometry: str
    observed: Dict[str, Optional[float]]
    replicate_readouts: List[Dict[str, Optional[float]]]
    replicate_index: List[int]
    rejected: List[Tuple[int, str]]
    p_values: Dict[str, Optional[float]]
    null_medians: Dict[str, Optional[float]]
    h1: Optional[CsrEnvelope]
    h2: CsrEnvelope
    fidelity: List[FidelityCheck]
    seconds_per_replicate: float
    n_locs_slab_null: List[int]
    k_null: List[int]
    example: Optional[Dict[str, Any]] = None
    global_test_from: str = "fusion"
    global_r_min_nm: float = float("nan")
    # keep_arcs (regularity PR 3, open question G2): every accepted replicate's centre arcs and membrane length,
    # for drawing the gap histogram, g(s) and S(m) bands at the pipeline level (~B x K numbers)
    replicate_arcs: List[Tuple[FloatArray, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    readout_specs: Dict[str, ReadoutSpec] = field(default_factory=dict)

    @property
    def n_accepted(self) -> int:
        return len(self.replicate_index)

    @property
    def closure_ratio(self) -> Optional[float]:
        vals = self.n_locs_slab_null if self.closure == "n_locs" else self.k_null
        if not vals or not self.closure_record.target > 0:
            return None
        return float(np.median(vals)) / float(self.closure_record.target)

    @property
    def closure_reached(self) -> bool:
        r = self.closure_ratio
        return r is not None and abs(r - 1.0) <= CLOSURE_TOLERANCE

    def not_interpretable_reasons(self) -> List[str]:
        """Why the p cannot be read (A9 sec. 4.4; same criteria and wording style as simnull). Empty = none."""
        out: List[str] = []
        if self.n_accepted < self.n_requested:
            out.append(f"Only {self.n_accepted} of the {self.n_requested} simulated axons asked for were analysed "
                       f"({len(self.rejected)} failed).")
        if self.n_accepted < MIN_ACCEPTED:
            out.append(f"With {self.n_accepted} simulated axons the p cannot go below 1/({self.n_accepted}+1) = "
                       f"{1.0 / (self.n_accepted + 1):.3f}: it can never reach 0.05.")
        if not self.closure_reached:
            r = self.closure_ratio
            what = "slab localizations" if self.closure == "n_locs" else "detected clusters"
            out.append(f"Closure not reached in {N_PILOTS} pilots: the null's median {what} is "
                       f"{'n/a' if r is None else f'{r:.2f}'} x the observed (needs within {CLOSURE_TOLERANCE:.0%}).")
        bad = [c for c in self.fidelity if c.met is False]
        if bad:
            out.append("The simulated axons do not resemble this axon (" + "; ".join(
                f"{c.name}: observed {c.observed:.3g}, null median {c.null_median:.3g}, {c.tolerance}" for c in bad)
                + ").")
        return out

    @property
    def interpretable(self) -> bool:
        return not self.not_interpretable_reasons()

    def headline(self) -> str:
        """The banner text: NOT INTERPRETABLE with its reasons, or the null's own description."""
        reasons = self.not_interpretable_reasons()
        if reasons:
            return "NOT INTERPRETABLE: " + " ".join(reasons)
        return (f"Null per axon ({self.null_type}; {self.method}; closure on {self.closure}; B = {self.n_accepted}, "
                f"seed {self.seed}). p values are approximate (the null reads contour, density and precision off "
                "this axon).")

    def export_dict(self) -> Dict[str, Any]:
        """The A9 sec. 5 columns (``null_*``, ``ripley*``, ``p_mc_*``, ``null_median_*``) plus ``scale_*``."""
        d: Dict[str, Any] = dict(self.scale.export_columns())
        d.update({
            "null_type": self.null_type, "null_closure": self.closure, "null_B": self.n_accepted,
            "null_seed": self.seed, "null_c_label": self.model.c_label,
            "null_locs_per_cycle": round(self.model.locs_per_cycle, 3),
            "null_locs_per_cycle_source": "measured" if self.model.origins.get("locs_per_cycle", "").startswith("read")
            else "default",
            "null_hard_core_nm": round(self.model.hard_core_nm, 1),
            "null_method": self.method,
            "null_interpretable": self.interpretable,
            "ripley_points": "centres",
            "ripley1d_window": "closed" if self.h1 is not None else "none",
            "ripley2d_window": "band" if self.geometry == "ring" else "roi",
            "null_global_test_from": self.global_test_from,
            "null_global_r_min_nm": None if not math.isfinite(self.global_r_min_nm) else round(self.global_r_min_nm, 1),
        })
        for tag, env in (("ripley1d", self.h1), ("ripley2d", self.h2)):
            d[f"{tag}_rstar_nm"] = None if env is None or env.r_star is None else round(env.r_star, 1)
            d[f"{tag}_hmax"] = None if env is None or env.h_max is None else round(env.h_max, 3)
            d[f"{tag}_p_global"] = None if env is None or env.p_global is None else round(env.p_global, 4)
        for name, p in self.p_values.items():
            d[f"p_mc_{name}"] = None if p is None else round(p, 4)
        for name, m in self.null_medians.items():
            d[f"null_median_{name}"] = None if m is None else round(m, 4)
        return d

    def replicate_rows(self) -> List[Dict[str, Any]]:
        """One row per accepted replicate (index, slab localizations, K, every readout): the replicates' table."""
        rows = []
        for i, idx in enumerate(self.replicate_index):
            row: Dict[str, Any] = {"replicate": idx, "n_locs_slab": self.n_locs_slab_null[i], "k": self.k_null[i]}
            row.update(self.replicate_readouts[i])
            rows.append(row)
        return rows

    def ripley_rows(self) -> List[Dict[str, Any]]:
        """The H1 / H2 curves (r, observed, null median, pointwise 2.5 / 97.5 %): the ``<stem>_ripley.csv`` rows."""
        rows = []
        for k, r in enumerate(self.r_grid):
            row: Dict[str, Any] = {"r_nm": float(r)}
            for tag, env in (("h1", self.h1), ("h2", self.h2)):
                if env is None:
                    continue
                row[f"{tag}_observed"] = None if env.observed is None else float(env.observed[k])
                row[f"{tag}_null_median"] = float(env.median[k])
                row[f"{tag}_null_lo"] = float(env.pointwise_lo[k])
                row[f"{tag}_null_hi"] = float(env.pointwise_hi[k])
            rows.append(row)
        return rows


# ============================================================================
# The run
# ============================================================================

def _fidelity(model: NullModel, closure: str, obs_nloc: int, obs_k: int, obs_scatter: float,
              nlocs: Sequence[int], ks: Sequence[int], scatters: Sequence[float]) -> List[FidelityCheck]:
    """A9 sec. 3.4, adapted (see the report): slab localizations within 15 %, the centroids' radial scatter
    within max(15 %, 3 nm); K within 15 % ONLY under the K closure (under the N_loc closure K is a tested
    readout, and requiring it to match would refuse exactly the axons where it differs)."""
    out: List[FidelityCheck] = []
    if nlocs:
        med = float(np.median(nlocs))
        out.append(FidelityCheck("slab localizations", float(obs_nloc), med, "within 15 %",
                                 abs(med / max(obs_nloc, 1) - 1.0) <= FIDELITY_TOLERANCE))
    if closure == "k_clusters" and ks:
        med = float(np.median(ks))
        out.append(FidelityCheck("detected clusters K", float(obs_k), med, "within 15 %",
                                 abs(med / max(obs_k, 1) - 1.0) <= FIDELITY_TOLERANCE))
    sc = [s for s in scatters if math.isfinite(s)]
    if model.geometry == "ring" and sc and math.isfinite(obs_scatter):
        med = float(np.median(sc))
        tol = max(FIDELITY_TOLERANCE * obs_scatter, FIDELITY_SCATTER_FLOOR_NM)
        out.append(FidelityCheck("radial scatter of the centres (nm)", obs_scatter, med,
                                 f"within {tol:.1f} nm", abs(med - obs_scatter) <= tol))
    return out


def run_pipeline_null(
    analysis: AxonAnalysis,
    *,
    spec: Optional[PipelineSpec] = None,
    strategy: Optional[ClusteringStrategy] = None,
    model: Optional[NullModel] = None,
    nena_nm: Optional[float] = None,
    frame_slab: Optional[NDArray[np.int64]] = None,
    n_replicates: int = DEFAULT_N_REPLICATES,
    seed: int = 0,
    closure: str = "n_locs",
    c_label: float = DEFAULT_C_LABEL,
    locs_per_cycle: Optional[float] = None,
    hard_core_nm: Optional[float] = None,
    r_max_nm: float = R_MAX_NM,
    alpha: float = ALPHA,
    no_ring: bool = False,
    allow_slow: bool = False,
    n_jobs: int = 1,
    readouts: Optional[Sequence[str]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    keep_example: bool = True,
    model_kw: Optional[Dict[str, Any]] = None,
    keep_arcs: bool = False,
    global_test_from: str = "fusion",
) -> PipelineNullResult:
    """The per-axon pipeline null of an observed ``AxonAnalysis`` (module docstring).

    The method is the observed analysis's own (``PipelineSpec.from_analysis``) unless ``spec`` or ``strategy``
    says otherwise -- the null MUST repeat the observed pipeline to be its null, so pass a different method only
    together with an observed analysis made with it. ``model`` replaces the model read from the axon (its
    ``n_molecules_per_ring`` is still closed). ``readouts`` names the registered readouts to test (default: all).
    ``progress(done, total)`` is called after each replicate. ``global_test_from`` (``GLOBAL_TEST_FROM``): the
    working-scale mark the global H1 / H2 test starts at, "fusion" (A9) or "d50". Raises ``NullRefused`` /
    ``SlowMethodRefused``. ``keep_arcs``: keep every replicate's centre arcs (``replicate_arcs``) for the
    regularity plots' pipeline-level bands [provisional: open question G2]."""
    if global_test_from not in GLOBAL_TEST_FROM:
        raise ValueError(f"global_test_from must be one of {GLOBAL_TEST_FROM}, got {global_test_from!r}")
    if closure not in CLOSURES:
        raise ValueError(f"closure must be one of {CLOSURES}, got {closure!r}")
    if int(n_replicates) < 1:
        raise ValueError("n_replicates must be >= 1")
    k_obs = int(analysis.n_clusters_kept)
    if k_obs < MIN_CLUSTERS:
        raise NullRefused(f"{k_obs} curated clusters (< {MIN_CLUSTERS}): the arrangement is not testable on this axon "
                          "(regularity plan, rule 1).")
    if spec is None:
        spec = PipelineSpec.from_analysis(analysis, strategy=strategy)
    elif strategy is not None:
        spec = dataclasses.replace(spec, strategy=strategy)
    check_method(spec, int(analysis.n_locs_slab), int(n_replicates), allow_slow=allow_slow)
    warnings: List[str] = []
    if analysis.discard_applied:
        warnings.append("The observed analysis has the axoplasm discard applied; the null cannot simulate the "
                        "widefield images and analyses every simulated cluster: compare it with the analysis of all "
                        "clusters.")
    if analysis.perimeter is not None and analysis.perimeter.order_source != "automatic":
        warnings.append("The observed contour was ordered by hand; the null's contours are the automatic 2-opt.")
    if analysis.cut is not None and getattr(analysis.cut, "cut_labels", None) and analysis.bad_report.cut_by_pick:
        warnings.append("Clusters cut by the pick were excluded in the observed analysis; the null does not cut.")
    if model is None:
        model = null_model_from_analysis(analysis, nena_nm=nena_nm, frame_slab=frame_slab, c_label=c_label,
                                         locs_per_cycle=locs_per_cycle, hard_core_nm=hard_core_nm, no_ring=no_ring,
                                         **(model_kw or {}))
    ring = model.geometry == "ring"
    if not ring:
        warnings.append("No ring (2D only): H1, occupancy and regularity are not tested; H2 uses the ROI as window.")
    scale = working_scale_of_analysis(analysis, nena_nm=nena_nm)
    r_grid = default_r_grid(r_max_nm)
    target = float(analysis.n_locs_slab) if closure == "n_locs" else float(k_obs)
    rec = close_null(model, spec, closure=closure, target=target, seed=seed)
    model = dataclasses.replace(model, n_molecules_per_ring=rec.n_molecules_per_ring, origins=dict(model.origins),
                                condition_slab_locs=int(analysis.n_locs_slab) if closure == "n_locs" else None)
    model.origins["n_molecules_per_ring"] = (
        f"closure on n_locs: {N_PILOTS} pilots for the mean, then every replicate conditioned on the observed slab's "
        "localizations" if closure == "n_locs" else f"closure on k_clusters ({N_PILOTS} pilots)")
    check_packing(model)
    names = list(READOUTS) if readouts is None else [str(n) for n in readouts]
    specs = [READOUTS[n] for n in names]
    window_2d = None if ring else np.asarray(model.region_nm, dtype=np.float64)
    args = [(model, spec, int(seed), b, r_grid, specs, window_2d, bool(keep_example and b == 0), bool(keep_arcs))
            for b in range(int(n_replicates))]
    outcomes: List[ReplicateOutcome] = []
    t_run = time.perf_counter()
    if int(n_jobs) <= 1:
        for a in args:
            outcomes.append(_replicate_star(a))
            if progress is not None:
                progress(len(outcomes), len(args))
    else:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as pool:
            for out in pool.map(_replicate_star, args, chunksize=max(1, len(args) // (4 * int(n_jobs)))):
                outcomes.append(out)
                if progress is not None:
                    progress(len(outcomes), len(args))
    wall = time.perf_counter() - t_run
    ok = [o for o in outcomes if o.ok]
    rejected = [(o.index, o.reason) for o in outcomes if not o.ok]
    if rejected:
        warnings.append(f"{len(rejected)} simulated axons failed (first: {rejected[0][1]}).")
    # observed
    obs_view = _view(analysis, ring, observed=True)
    observed: Dict[str, Optional[float]] = {}
    for ro in specs:
        if ro.needs_ring and not ring:
            continue
        v = ro.fn(obs_view)
        observed[ro.name] = None if v is None or not math.isfinite(float(v)) else float(v)
    h1_obs, h2_obs = ripley_curves(obs_view, r_grid, window_2d)
    p_values: Dict[str, Optional[float]] = {}
    medians: Dict[str, Optional[float]] = {}
    for ro in specs:
        if ro.name not in observed:
            continue
        vals = np.array([o.readouts.get(ro.name) for o in ok if o.readouts.get(ro.name) is not None], dtype=float)
        medians[ro.name] = float(np.median(vals)) if vals.size else None
        obs = observed[ro.name]
        if ro.depends_on_closure and closure == "k_clusters":
            p_values[ro.name] = None              # conditioned on K: its p means nothing (A9 sec. 2.7)
            continue
        p_values[ro.name] = None if obs is None else monte_carlo_p(obs, vals, two_sided=ro.two_sided)
    r_min = float(scale.fusion_nm) if global_test_from == "fusion" else float(scale.d50_nm)
    if r_min >= float(r_grid[-1]):
        warnings.append(f"The {global_test_from} mark ({r_min:.0f} nm) is beyond the largest radius ({r_grid[-1]:.0f} nm): the "
                        "global tests are empty; raise r_max_nm.")
    tested = r_grid >= r_min

    def _curves(rows: List[FloatArray]) -> FloatArray:
        # a replicate whose curve cannot be computed (fewer than 2 centres) is left out of the envelope: counted
        # among the B with T_b = NaN it would never be "as extreme" and would make p_global too small
        good = [h for h in rows if np.all(np.isfinite(h[tested]))]
        return np.array(good) if good else np.full((0, r_grid.size), np.nan)

    h2_env = envelope_from_curves(_curves([o.h2 for o in ok]), r_grid, observed=h2_obs, alpha=alpha, r_min=r_min,
                                  seed=seed)
    h1_env: Optional[CsrEnvelope] = None
    if ring:
        h1_env = envelope_from_curves(_curves([o.h1 for o in ok]), r_grid, observed=h1_obs, alpha=alpha,
                                      r_min=r_min, seed=seed)
    fid = _fidelity(model, closure, int(analysis.n_locs_slab), k_obs, obs_view.radial_scatter_nm,
                    [o.n_locs_slab for o in ok], [o.k_clusters for o in ok], [o.radial_scatter_nm for o in ok])
    example = next((o.example for o in ok if o.example is not None), None)
    null_type = NULL_TYPE_PROVISIONAL
    return PipelineNullResult(
        method=spec.method_name(), method_params=dict(spec.params), closure=closure, null_type=null_type,
        seed=int(seed), n_requested=int(n_replicates), model=model, closure_record=rec, scale=scale, r_grid=r_grid,
        geometry=model.geometry, observed=observed, replicate_readouts=[o.readouts for o in ok],
        replicate_index=[o.index for o in ok], rejected=rejected, p_values=p_values, null_medians=medians,
        h1=h1_env, h2=h2_env, fidelity=fid,
        seconds_per_replicate=(wall / max(len(outcomes), 1)) if int(n_jobs) <= 1 else
        float(np.mean([o.seconds for o in outcomes])) if outcomes else float("nan"),
        n_locs_slab_null=[o.n_locs_slab for o in ok], k_null=[o.k_clusters for o in ok], example=example,
        global_test_from=str(global_test_from), global_r_min_nm=float(r_min),
        replicate_arcs=[(o.arcs, float(o.arc_length_nm)) for o in ok if o.arcs is not None],
        warnings=warnings + list(rec.notes), readout_specs={s.name: s for s in specs})


def measure_cost(analysis: AxonAnalysis, *, spec: Optional[PipelineSpec] = None, n_probe: int = 1, seed: int = 0,
                 nena_nm: Optional[float] = None, allow_slow: bool = False) -> Dict[str, float]:
    """Seconds per replicate MEASURED on this axon's own synthetic replicates [simulation] (``n_probe`` of them,
    spawn key (2, j)): generate + full pipeline + readouts. Returns {"seconds_per_replicate", "n_locs_total",
    "n_locs_slab"} (means over the probes). A slow method is still refused unless ``allow_slow``."""
    if spec is None:
        spec = PipelineSpec.from_analysis(analysis)
    check_method(spec, int(analysis.n_locs_slab), 1, allow_slow=allow_slow)
    model = null_model_from_analysis(analysis, nena_nm=nena_nm)
    rec = close_null(model, spec, closure="n_locs", target=float(analysis.n_locs_slab), seed=seed, n_pilots=2)
    model = dataclasses.replace(model, n_molecules_per_ring=rec.n_molecules_per_ring,
                                condition_slab_locs=int(analysis.n_locs_slab))
    secs, tot, slab = [], [], []
    r_grid = default_r_grid()
    for j in range(int(n_probe)):
        t0 = time.perf_counter()
        ax = simulate_null_axon(model, _rng(seed, 2, j))
        an = spec.run(ax.x_nm, ax.y_nm, ax.z_nm, main_peak_nm=model.z_centre_nm, lp_nm=ax.lp_nm, frame=ax.frame)
        v = _view(an, model.geometry == "ring")
        ripley_curves(v, r_grid, None if model.geometry == "ring" else model.region_nm)
        secs.append(time.perf_counter() - t0)
        tot.append(ax.n_locs)
        slab.append(an.n_locs_slab)
    return {"seconds_per_replicate": float(np.mean(secs)), "n_locs_total": float(np.mean(tot)),
            "n_locs_slab": float(np.mean(slab))}


# ============================================================================
# Synthetic axons (tests and the calibration scripts only: rule R8)
# ============================================================================

def ellipse_contour(a_nm: float, b_nm: float, n: int = 360) -> FloatArray:
    """A closed ellipse polygon (n vertices, uniform parametric angle)."""
    t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    return np.column_stack([float(a_nm) * np.cos(t), float(b_nm) * np.sin(t)]).astype(np.float64)


def synthetic_model(semi_axes_nm: Tuple[float, float] = (2000.0, 1400.0), *, molecules_per_um: float = 4.0,
                    hard_core_nm: float = 50.0, radial_sd_nm: float = 5.0, sigma_xy_nm: float = 12.0,
                    sigma_z_nm: float = 40.0, period_nm: float = 190.0, background_per_um2: float = 2.0,
                    n_frames: int = 20000, c_label: float = DEFAULT_C_LABEL, margin_nm: float = BACKGROUND_MARGIN_NM,
                    **kw: Any) -> NullModel:
    """A synthetic axon's TRUE model (an ellipse membrane; the REG plan's "contact 50 nm" null truth by default:
    4 molecules per um [lit order: Gazal et al. 2026, ~4 clusters per um], NeNA 12 nm [lit: Gazal et al. 2026,
    Methods], period 190 nm [ad hoc, the simulator's], background 2 per um^2 of slab [ad hoc]). For tests and the
    calibration scripts; the null itself never uses it."""
    mem = ellipse_contour(*semi_axes_nm)
    lo, hi = mem.min(axis=0) - margin_nm, mem.max(axis=0) + margin_nm
    region = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], dtype=np.float64)
    length = float(np.sum(np.hypot(*(np.roll(mem, -1, axis=0) - mem).T)))
    return NullModel(geometry="ring", membrane_nm=mem, region_nm=region, z_centre_nm=0.0, period_nm=float(period_nm),
                     slab_half_width_nm=90.0, n_molecules_per_ring=float(molecules_per_um) * length / 1000.0,
                     hard_core_nm=float(hard_core_nm), radial_sd_nm=float(radial_sd_nm), sigma_xy_nm=float(sigma_xy_nm),
                     sigma_z_nm=float(sigma_z_nm), background_per_um2=float(background_per_um2), n_frames=int(n_frames),
                     c_label=float(c_label), origins={"all": "synthetic truth"}, **kw)


def planted_arcs(kind: str, length_nm: float, n_molecules: int, rng: np.random.Generator, *,
                 pair_distance_nm: float = 150.0, lattice_nm: float = 190.0, jitter_nm: float = 30.0,
                 hard_core_nm: float = 50.0) -> FloatArray:
    """Arc positions with KNOWN structure for power tests (A9 sec. 6.4): "pairs" -- n/2 pair centres by
    sequential inhibition at pair_distance + hard_core, each a pair of molecules pair_distance apart; "lattice" --
    sites every lattice_nm (random phase) with N(0, jitter) offsets, n_molecules ignored."""
    if kind == "pairs":
        n_pairs = max(1, int(n_molecules) // 2)
        c = _hard_core_arcs(rng, n_pairs, float(length_nm), float(pair_distance_nm + hard_core_nm), np.zeros(0),
                            max_attempts=HARD_CORE_MAX_ATTEMPTS)
        return np.concatenate([c - 0.5 * pair_distance_nm, c + 0.5 * pair_distance_nm]) % float(length_nm)
    if kind == "lattice":
        n_sites = max(1, int(round(float(length_nm) / float(lattice_nm))))
        step = float(length_nm) / n_sites
        phase = rng.uniform(0.0, step)
        return (phase + step * np.arange(n_sites) + rng.normal(0.0, jitter_nm, n_sites)) % float(length_nm)
    raise ValueError(f"kind must be 'pairs' or 'lattice', got {kind!r}")


def observe_synthetic(model: NullModel, seed: int, spec: Optional[PipelineSpec] = None,
                      molecule_arcs: Optional[Sequence[FloatArray]] = None) -> Tuple[AxonAnalysis, NullAxon]:
    """Simulate one synthetic 'observed' axon from ``model`` (seed key (9, seed), never a null replicate's) and run
    the pipeline on it, keeping the frames and precisions as a loaded file would."""
    ax = simulate_null_axon(model, _rng(int(seed), 9), molecule_arcs=molecule_arcs)
    sp = spec if spec is not None else PipelineSpec()
    an = sp.run(ax.x_nm, ax.y_nm, ax.z_nm, main_peak_nm=model.z_centre_nm, lp_nm=ax.lp_nm, frame=ax.frame,
                source_name=f"synthetic observed {seed}")
    return an, ax


# ============================================================================
# The multi-method plug-ins
# ============================================================================

# Readouts of the multi-method long table (tools.multimethod naming: prefix null_, p-values end in _p). The A9
# axon-table columns (p_mc_*, ripley*_*) are the export's names; the map is MULTIMETHOD_NAMES.
MULTIMETHOD_NAMES: Dict[str, str] = {
    "p_mc_k_per_um": "null_clusters_per_um_p", "p_mc_nn_median": "null_median_1nn_nm_p",
    "p_mc_reff_median": "null_median_r_eff_nm_p", "p_mc_occupancy": "null_occupancy_percent_p",
    "p_mc_regularity": "null_regularity_p", "ripley1d_p_global": "null_ripley1d_p",
    "ripley2d_p_global": "null_ripley2d_p", "ripley1d_rstar_nm": "null_ripley1d_rstar_nm",
    "ripley2d_rstar_nm": "null_ripley2d_rstar_nm", "ripley1d_hmax": "null_ripley1d_hmax_nm",
    "ripley2d_hmax": "null_ripley2d_hmax_nm", "null_B": "n_null_replicates",
}
INTERPRETABLE_LEVELS = ("interpretable", "not interpretable")


def _multimethod() -> Any:
    """``tools.multimethod`` (feat/multimethod-core), imported by name: it is not on this branch."""
    import importlib
    return importlib.import_module("tools.multimethod")


def pipeline_null_analysis(run: Any, context: Any) -> Dict[str, Any]:
    """The ``tools.multimethod`` plug-in: the pipeline null of ONE method's run, re-running that method.

    Reads ``run.analysis``, ``run.method``, ``run.params``; ``context.seed``, ``context.n_replicates``,
    ``context.allow_slow``, ``context.check_cancelled`` and ``context.extras`` keys ``nena_nm`` (the file's NeNA;
    without it 12 nm [lit], labelled), ``null_closure`` ("n_locs"), ``null_c_label`` (14) and ``null_r_max_nm``
    (400). A slow method without allow_slow raises the runner's own ``SlowMethodRefused``."""
    mm = _multimethod()
    an = run.analysis
    extras = dict(getattr(context, "extras", {}) or {})
    spec = PipelineSpec.from_analysis(an)
    spec = PipelineSpec.for_method(str(run.method), dict(run.params or {}), **{
        k: getattr(spec, k) for k in ("slab_half_width_nm", "dbcv_threshold", "roi", "mahalanobis_threshold",
                                      "ellipse_mode", "sigma_cap_fraction", "all_starts", "main_peak_mode")})
    n_rep = int(getattr(context, "n_replicates", DEFAULT_N_REPLICATES))
    try:
        res = run_pipeline_null(
            an, spec=spec, nena_nm=extras.get("nena_nm"), n_replicates=n_rep, seed=int(getattr(context, "seed", 0)),
            closure=str(extras.get("null_closure", "n_locs")), c_label=float(extras.get("null_c_label", DEFAULT_C_LABEL)),
            r_max_nm=float(extras.get("null_r_max_nm", R_MAX_NM)), allow_slow=bool(getattr(context, "allow_slow", False)),
            global_test_from=str(extras.get("null_global_test_from", "fusion")),
            progress=lambda done, total: context.check_cancelled(f"pipeline null, replicate {done} of {total}"))
    except SlowMethodRefused as exc:
        raise mm.SlowMethodRefused(mm.slow_refusal(str(run.method), 1 + n_rep) + f" ({exc})") from exc
    except NullRefused:
        return {}
    d = res.export_dict()
    out: Dict[str, Any] = {new: d.get(old) for old, new in MULTIMETHOD_NAMES.items()}
    out["null_closure_ratio"] = res.closure_ratio
    out["null_seconds_per_replicate_s"] = res.seconds_per_replicate
    out["null_interpretable_verdict"] = INTERPRETABLE_LEVELS[0] if res.interpretable else INTERPRETABLE_LEVELS[1]
    return out


def register_multimethod(replace: bool = False) -> None:
    """Register ``pipeline_null_analysis`` with ``tools.multimethod`` (the integration calls this once; the module
    lives on another branch, hence the lazy import)."""
    mm = _multimethod()
    R = mm.ReadoutSpec
    org = "pipeline null per axon (A9); p: Besag & Diggle 1977; global envelope: MAD test"
    cols = [R("null_clusters_per_um_p", kind="p_value", description="Monte Carlo p of clusters per um "
              "(empty under the K closure)", origin=org),
            R("null_median_1nn_nm_p", kind="p_value", description="Monte Carlo p of the median 1NN", origin=org),
            R("null_median_r_eff_nm_p", kind="p_value", description="Monte Carlo p of the median r_eff", origin=org),
            R("null_occupancy_percent_p", kind="p_value", description="Monte Carlo p of the occupancy", origin=org),
            R("null_regularity_p", kind="p_value", description="Monte Carlo p of the regularity r", origin=org),
            R("null_ripley1d_p", kind="p_value", description="global (MAD) p of H1 on the arc", origin=org),
            R("null_ripley2d_p", kind="p_value", description="global (MAD) p of H2 in the band", origin=org),
            R("null_ripley1d_rstar_nm", description="r of the largest studentised deviation of H1", origin=org),
            R("null_ripley2d_rstar_nm", description="r of the largest studentised deviation of H2", origin=org),
            R("null_ripley1d_hmax_nm", description="H1 observed - null median at r*", origin=org),
            R("null_ripley2d_hmax_nm", description="H2 observed - null median at r*", origin=org),
            R("n_null_replicates", kind="count", description="simulated axons analysed", origin=org),
            R("null_closure_ratio", description="null median / observed of the closed quantity", origin=org),
            R("null_seconds_per_replicate_s", description="measured cost of one simulated axon", origin=org),
            R("null_interpretable_verdict", kind="verdict", levels=INTERPRETABLE_LEVELS,
              description="NOT INTERPRETABLE conditions of A9 sec. 4.4", origin=org)]
    mm.register_analysis("pipeline_null", pipeline_null_analysis, cols, replace=replace,
                         description="Per-axon pipeline null with blinking, same method and curation (A9 + REG PR 1).",
                         cost_seconds=lambda run: (1 + int(DEFAULT_N_REPLICATES)) * estimated_seconds_per_replicate(
                             str(run.method), int(getattr(run.analysis, "n_locs_slab", 0) or 0)))


def null_readouts(run: Any, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Plain plug-in form for the multi-method runner: ``run`` is an ``AxonAnalysis`` (or has ``.analysis``) made
    with ONE method; the null re-runs that same method. ``context`` keys (all optional): ``nena_nm``,
    ``frame_slab``, ``n_replicates`` (199), ``seed`` (0), ``closure`` ("n_locs"), ``c_label`` (14),
    ``allow_slow`` (False), ``n_jobs`` (1), ``r_max_nm`` (400). Returns the ``export_dict`` columns plus
    ``null_headline``; a refusal returns {"null_refused": <reason>} instead of raising, so one slow method does not
    stop the comparison of the others."""
    ctx = dict(context or {})
    analysis = getattr(run, "analysis", run)
    try:
        res = run_pipeline_null(
            analysis, nena_nm=ctx.get("nena_nm"), frame_slab=ctx.get("frame_slab"),
            n_replicates=int(ctx.get("n_replicates", DEFAULT_N_REPLICATES)), seed=int(ctx.get("seed", 0)),
            closure=str(ctx.get("closure", "n_locs")), c_label=float(ctx.get("c_label", DEFAULT_C_LABEL)),
            allow_slow=bool(ctx.get("allow_slow", False)), n_jobs=int(ctx.get("n_jobs", 1)),
            r_max_nm=float(ctx.get("r_max_nm", R_MAX_NM)))
    except NullRefused as exc:
        out: Dict[str, Any] = {"null_refused": str(exc)}
        if isinstance(exc, SlowMethodRefused):
            out["null_estimated_s"] = round(exc.estimate_s, 1)
        return out
    d = res.export_dict()
    d["null_headline"] = res.headline()
    return d
