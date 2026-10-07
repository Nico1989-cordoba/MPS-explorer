# -*- coding: utf-8 -*-
"""
Rings of one axon in the axon's own frame, each keeping the link from its
localizations back to their frame numbers, axial precisions and emission
events: the data model of the column analysis and ``build_rings``
(milestone H1 of the research plan, 03_plan.md, section 3.1).

Why this exists
---------------
The existing pipeline (``tools.mps_multisegment.analyze_all_segments``)
already cuts an axon into axial slabs, one per dominant component of the z
distribution, and runs the per-segment analysis on each. What it returns
is coordinates, DBSCAN labels and cluster centroids. Three things the
column analysis needs are lost on the way:

* which localization of the input each slab localization is, so that its
  frame number and precisions can be looked up again (``Ring.loc_index``,
  ``Cluster.loc_index``: indices into the arrays ``build_rings``
  received);
* how many distinct emission events a cluster is made of, as opposed to
  how many localizations: one molecule seen in five consecutive frames is
  five localizations but one event, and every statistic that treats them
  as five independent observations is overconfident (``n_events``, from
  ``tools.mps_paint.link_localizations`` over ALL input localizations,
  before any axial cut, so an event that straddles a ring boundary is one
  event on both sides, not two);
* the axial precision of the localizations actually in the ring
  (``lpz_median_nm``), which the leak diagnostics of H4 need per ring and
  per cluster.

The rings are built in the axon frame (``tools.mps_axis``): a tilt beta of
the axon axis from the optical axis mixes rings in z by D * tan(beta), up
to hundreds of nm for a wide axon, so the z-segmentation that defines the
rings runs on the rotated coordinates. The rotation is rigid, so every
distance and area the per-segment analysis reports is unchanged in
meaning; only the frame in which the contour and the centroids are
expressed is the axon's.

How the axis is estimated here (and why not in one pass)
---------------------------------------------------------
The H1 specification fits one plane per ring, the rings being the slabs
of a first segmentation in the laboratory frame. That works only while
the slabs are mostly one ring each, i.e. while D * tan(beta) is below
half a period -- and the wide, tilted axon is exactly the case the frame
exists for. Measured on the synthetic 8 um axon tilted 3 deg (D *
tan(beta) = 420 nm): the middle laboratory slab holds 403 / 356 / 399
localizations of the three true rings, every "ring plane" is a
horizontal slice through that mixture (tilts 0.07, 0.06, 0.13 deg), and
the one-pass estimate is 0.045 deg. The rings are then assigned no
better than without any correction (30 % agreement either way).

A plane through ALL localizations needs no labels but assumes every
ring has the same lateral footprint: consecutive rings with incomplete
arcs on opposite sides shift the cloud's centre with z, and the pooled
regression reads that as a tilt (1.7 deg on an 8 um axon that is not
tilted, with no warning; it then mixes the rings and the per-ring
stages cannot recover). Partial rings are the normal case in real
sections, so the pooled plane is not used.

``tools.mps_axis.fit_axis_by_profile_sharpness`` needs neither labels
nor equal footprints: it searches the tilt at which the axial profile
is sharpest. Measured on the same axons it starts within 0.2 deg of the
truth (0.05 deg on the 8 um / 3 deg axon, 0.0-0.06 deg on the
opposite-arc axons where the pooled plane gives 1.2-1.7 deg), and after
that start the rings ARE separable, so a per-ring fit in the rotated
frame removes the residual.

``build_rings`` therefore estimates the axis in stages, each with
``fit_axon_frame`` and each rotating about ONE origin, so that the
stages compose into one rotation:

  stage 0   a starting frame (see below);
  stage k   segment z in the current frame, one plane per ring, rotate;
            repeated until a stage rotates by less than
            ``AXIS_STAGE_TOL_DEG`` or ``AXIS_MAX_RING_STAGES`` is reached.

Two starts are refined this way and the result is chosen between them:
(A) the specification's own start, one plane per ring of the
laboratory-frame segmentation, and (B) the profile-sharpness start. When
their refined axes agree within ``AXIS_STAGE_TOL_DEG`` the
specification's is kept (unless it alone failed to settle); otherwise
the start whose last segmentation is better resolved wins: the total
relative depth of the density valleys between its rings
(``density_valleys``). In a wrong frame the rings are smeared into one
another and the axial density has no deep valleys, whichever way the
mixture cuts it. Two other judges were tried and fooled (measured): the
global profile sharpness rewards stacking rings as much as thinning
them (symmetric half rings on opposite sides can be sheared onto the
middle ring), and the residual scale about the ring planes rewards a
smear that BIC cuts into thin slices. A warning reports a disagreement
beyond ``AXIS_CANDIDATE_DISAGREEMENT_DEG``. The choice, both refined
tilts and both valley depths are in ``RingsResult.axis_start``,
``axis_candidate_beta_deg`` and ``axis_candidate_valley_depth``; the
increments of the chosen start's stages in ``axis_stages_beta_deg``.

In every stage the plane of a ring is fitted on the localizations the
axial mixture assigns to it with a posterior of at least
``AXIS_STAGE_POSTERIOR_MIN``: the hard mask's boundary leaks are
laterally one-sided on a ring with a missing arc and tilt the
neighbour's plane (1.2 deg of drift measured on half rings with
sigma_z = 50 nm). The final rings keep the specification's hard mask.

The per-ring stage is the specification's step, run where it is valid.
The composed ``AxonFrame`` reports the last per-ring stage's normals
(mapped back to the laboratory frame), its delta_beta (an angle between
normals, invariant under the rotation), its method, and as origin the
mean of the localizations of the rings that entered that stage's fit.
Its ``ring_n_locs``/``ring_used`` describe the segmentation that stage
used, in that stage's frame, which can differ from the final rings by a
few localizations near the boundaries. Everything stays deterministic.

What is here (H1)
-----------------
``Cluster``, ``Ring``, ``RingsParams``, ``RingsResult``    data model
``ring_index_from_segments``    ring label per localization from axial slabs
``posterior_ambiguous``         localizations the axial mixture cannot place
``build_rings``                 the whole H1 step

What is here (H2)
-----------------
``bootstrap_centroid_sigma``    centroid precision by a bootstrap over EVENTS
``Cluster.sigma_centroid_nm``   filled by ``build_rings`` (NaN without frames)
``Cluster.suspect``             marks "few_events", "burst", "edge" (informative:
                                nothing is removed; H3 decides what to do)
``RingsResult.n_suspect``       count per mark over all rings
``TauComponents``, ``tau_components``    tau_0 of D-03 and its components
``ColumnsParams``, ``write_columns_params``, ``load_columns_params``,
``rings_params_from``            the pre-registered parameters file (YAML)
``columns_params_from_rings``    an addition beyond the H2 specification: the
                                ``ColumnsParams`` for a measured tau_0 (what
                                the report script needs to write the file)

What is here (H5-D)
-------------------
``clean_rings``                 the rings without some of their clusters (the
                                lumen cleaning of ``tools.mps_lumen``, D-35):
                                each touched ring's contour rebuilt exactly as
                                ``build_rings`` built it, the input untouched
``RemovedCluster``,
``RingsResult.removed_clusters`` what was removed and why

Why the precision is bootstrapped over events (D-03, 02_investigacion B4):
the localizations of one emission event are repeated measurements of the
same molecule, correlated through the antibody offset and the molecule's
position, so resampling localizations treats k correlated copies as k
independent observations and understates the centroid's uncertainty (0.58
of the truth with k = 5 in the harness). The independent units are the
events. The plug-in bootstrap is itself biased low with few events (0.77
of the truth at E = 3, 0.83 at E = 4, 0.98 at E = 30; see
``bootstrap_centroid_sigma``); no correction is applied because D-03 and
the H2 specification define the estimator as the sd of the draw means,
and the H2 report and H3's sensitivity to the marks account for it.

Why the marks are informative and not a removal: a cluster of one or two
molecules seen many times is what an axially leaked molecule looks like
(B4), a burst is what a stuck fluorophore looks like, and a cluster cut
by the ROI has a biased centroid and area; each is a reason not to let
that cluster SET the tolerance nor count as a clean column member, not a
reason to delete it from the rings before H3 has measured what it changes
(the sensitivity with and without is H3's).

Why tau_0 is the median of the PAIR tolerance and not the sum of medians:
the tolerance is applied per pair in H3, so the typical pair tolerance is
the right scalar; both medians are reported anyway (``TauComponents``).

What is NOT here
----------------
Ring-to-ring matching (``match_rings``, the eclipse test and its arc-shift
null), columns across rings (``build_columns``, the axial profile test)
and the leak diagnostics come in milestones H3 and H4. No "dbcv" mark:
the DBCV criterion is off by default in this project
(``tools.mps_settings.DEFAULT_DBCV_THRESHOLD = -1.0``) because its score
correlates strongly with cluster size (log10 area, on real axons: it
doubles as a size filter and removes the large clusters the paper keeps
as oligomers), so a mark from it would be a size mark in disguise.

Design decisions
----------------
* ``build_rings`` takes arrays, not a ``Localizations`` object: the GUI
  hands it the ROI selection before the axial cut (``xroi_unfiltered``
  and friends plus ``roi_indices_unfiltered``), the batch hands it a file,
  the harness hands it synthetic truth. Whatever the caller has, every
  ``loc_index`` is relative to what it passed.
* Membership of a localization in a ring is the hard z mask of the
  segment, exactly as ``analyze_axon`` applies it, so ``Ring.loc_index``
  is ``analysis.slab_index`` mapped back to the input and nothing else.
  Ring membership by posterior (``params.posterior_min``) is offered as a
  control, not as the primary path: it drops the localizations near a
  boundary before the per-ring analysis, so that whatever survives in the
  matching of H3 cannot be blamed on localizations the mixture placed by
  a coin toss.
* The H1 part draws no random numbers. The only randomised step of H1 is
  inside the existing pipeline (the GMM, pinned to ``random_state=0``),
  and ``params.random_seed`` is forwarded to it, so two calls on the same
  input give the same rings, the same clusters and the same warnings.
  The H2 bootstrap draws from a ``numpy.random.SeedSequence`` of
  ``params.random_seed`` spawned in a fixed order (ring index, then
  cluster position), so a cluster's precision does not depend on what
  else ran, on how many rings there are, or on the order of the calls.
* The H2 marks are computed from the ring analysis as it stands, never
  by changing what ``identify_bad_clusters`` reports: the export column
  ``removed_edge_touching`` reads that report, and the runaway guard
  empties ``edge_touching`` when it fires, so the "edge" mark recomputes
  the touching clusters from the ROI itself (nothing is recomputed on
  the convex-hull path: a hull cuts nothing, so nothing is biased there).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import copy
import dataclasses
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from tools.cluster_quality import (
    ROIShape,
    find_edge_touching_clusters,
    good_cluster_labels,
)
from tools.mps_analysis import AxonAnalysis, without_clusters
from tools.mps_axis import (
    METHOD_ALL,
    METHOD_IDENTITY,
    TILT_WARNING_DEG,
    AxonFrame,
    fit_axis_by_profile_sharpness,
    fit_axon_frame,
    identity_axon_frame,
    lateral_shift_per_period_nm,
    profile_sharpness,
    to_axon_frame,
)
from tools.mps_multisegment import (
    AxialSegment,
    MultiSegmentAnalysis,
    analyze_all_segments,
    find_axial_segments,
)
from tools.mps_paint import frame_analysis, link_localizations
from tools.mps_periodicity import ZPeriodicityResult, fit_z_periodicity

__all__ = [
    "AXIS_CANDIDATE_DISAGREEMENT_DEG",
    "AXIS_MAX_RING_STAGES",
    "AXIS_STAGE_TOL_DEG",
    "AXIS_START_LAB_RINGS",
    "AXIS_START_SHARPNESS",
    "COLUMNS_PARAMS_HEADER",
    "DATASET_ROLES",
    "DEFAULT_TAU_GRID_NM",
    "MIN_BOOTSTRAP_DRAWS",
    "MIN_LOCS_FOR_RINGS",
    "PROVENANCE_REQUIRED_KEYS",
    "SEGMENT_MODES",
    "SUSPECT_BURST",
    "SUSPECT_EDGE",
    "SUSPECT_FEW_EVENTS",
    "SUSPECT_MARKS",
    "TAU_QUANTILES_PERCENT",
    "Cluster",
    "ColumnsParams",
    "RemovedCluster",
    "Ring",
    "RingsParams",
    "RingsResult",
    "TauComponents",
    "bootstrap_centroid_sigma",
    "build_rings",
    "clean_rings",
    "columns_params_from_rings",
    "load_columns_params",
    "posterior_ambiguous",
    "ring_index_from_segments",
    "rings_params_from",
    "tau_components",
    "write_columns_params",
]

# Per-ring axis stages after the start. On rings well separated in z
# (sigma_z 35 nm at P = 190 nm) the first stage removes the residual of
# the start and the second confirms it. When rings overlap (sigma_z 50
# nm: the boundary is 1.9 sigma from each centre) the hard mask cuts the
# far side of a still-tilted ring, the plane fitted inside the slab sees
# only ~30 % of the residual per stage (measured ratio 0.7), and
# reaching the 0.05 deg tolerance from a 2 deg start error takes
# ln(0.05 / 2) / ln(0.7) = 10.3 stages. A converged case stops early, so
# the extra stages cost nothing there.
AXIS_MAX_RING_STAGES = 12
# A stage that rotates by less than this has nothing left to correct:
# the slope error of one plane at the typical N and spread (sigma_z 50 nm,
# 5000 localizations, 900 nm lateral sd) is ~0.05 deg, so a smaller
# increment is the fit's own noise, not a refinement.
AXIS_STAGE_TOL_DEG = 0.05
# Two starts whose refined axes end more than this apart did not converge
# to one axis: one of them stopped in a segmentation that mixes rings
# (D * tan(residual) > P/2), which is what rings with different lateral
# footprints do to the specification's start. It is > 3 x the slope error
# of one plane (0.05-0.16 deg at the harness's N) and the tolerance the
# acceptance uses for the tilt itself.
AXIS_CANDIDATE_DISAGREEMENT_DEG = 0.5
# Localizations whose ring the axial mixture cannot name (maximum
# posterior below this) stay out of the per-ring plane fits of the axis
# stages. Near a boundary the hard mask hands some of ring k's
# localizations to ring k+1; on a complete ring those leaks are
# laterally symmetric and only add noise, but on a ring with a missing
# arc they sit on one side and tilt the neighbour's plane (measured:
# half rings with sigma_z = 50 nm drift 1.2 deg over five stages with
# every localization, 0.1 deg without the ambiguous 13 %). The final
# rings keep the specification's hard mask; this only decides which
# localizations vote for the axis. 0.85 is the value the module already
# offers as the posterior_min control.
AXIS_STAGE_POSTERIOR_MIN = 0.85
# Labels of the two starts, in the order of ``RingsResult.axis_candidate_beta_deg``.
AXIS_START_LAB_RINGS = "lab-frame rings"
AXIS_START_SHARPNESS = "profile sharpness"
# The axial mixture needs two values; with fewer there is nothing to
# segment and ``build_rings`` returns an empty result instead of raising.
MIN_LOCS_FOR_RINGS = 2

# H2 suspect marks (``Cluster.suspect``), in the order they are counted.
# "few_events": fewer events than params.min_events_per_cluster (one or
# two molecules seen many times: an axially leaked molecule looks like
# that, and its centroid precision cannot be bootstrapped). "burst":
# Picasso's frame rule as implemented in tools.mps_paint.frame_analysis
# (mean frame outside the middle 20-80 % of the acquisition, or one
# twentieth of it holding > 80 % of the localizations). "edge": kept only
# because the runaway guard disabled the edge criterion, although it
# touches the ROI boundary (biased centroid and area).
SUSPECT_FEW_EVENTS = "few_events"
SUSPECT_BURST = "burst"
SUSPECT_EDGE = "edge"
SUSPECT_MARKS: Tuple[str, ...] = (SUSPECT_FEW_EVENTS, SUSPECT_BURST, SUSPECT_EDGE)

# Quantiles reported by ``tau_components`` for rho and sigma_m, in percent.
TAU_QUANTILES_PERCENT: Tuple[float, ...] = (10.0, 25.0, 50.0, 75.0, 90.0)
# The tolerance grid of the pre-registered file: 20, 30, ..., 200 nm.
DEFAULT_TAU_GRID_NM: Tuple[float, ...] = tuple(float(v) for v in range(20, 201, 10))
# The bootstrap needs two draws for a standard deviation with ddof = 1;
# fewer means "off": sigma stays NaN and n_bootstrap 0.
MIN_BOOTSTRAP_DRAWS = 2
# Comment block written above the YAML body of the pre-registered file.
# PyYAML drops comments, so the block is written by hand and the parser
# skips it (a '#' line is a comment to the YAML reader as well).
COLUMNS_PARAMS_HEADER: Tuple[str, ...] = (
    "# Pre-registered parameters of the column analysis (MPS Explorer, H2).",
    "# tau0_nm is the median pair tolerance measured on the exploratory axon",
    "# (DECISIONES D-03, D-05); the other keys are what build_rings and the",
    "# H3+ steps must run with so that both agree (rings_params_from).",
    "# Written by tools.mps_columns.write_columns_params; read it back with",
    "# load_columns_params, which rejects unknown keys. Frozen once committed.",
)
# The values ``segment_mode`` may take (``tools.mps_multisegment.
# find_axial_segments`` raises on anything else, but only when the rings
# are built; the parameters file must fail on load instead).
SEGMENT_MODES: Tuple[str, ...] = ("paper", "valley", "partition")
# Provenance of the pre-registered file (03_plan S3.1, RunRecord): the
# role of the data tau_0 came from, the date the caller passed in and
# the code commit. ``write_columns_params`` refuses a file without them,
# so a frozen file cannot exist that does not say where it came from.
DATASET_ROLES: Tuple[str, ...] = ("exploratory", "check", "confirmatory")
PROVENANCE_REQUIRED_KEYS: Tuple[str, ...] = ("dataset_role", "date", "code_commit")

# A component width below this is a numerical accident, not a ring; it
# only keeps the posterior finite (as ``ZPeriodicityResult.mixture_density``
# does) and never changes a result on real data.
_MIN_SIGMA_NM = 1e-9

_Z_HAT: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])


# ============================================================================
# Data model
# ============================================================================

@dataclass
class Cluster:
    """One kept cluster of one ring, in the axon frame.

    ``loc_index`` indexes the arrays ``build_rings`` received;
    ``centroid_nm`` is the exact mean of those localizations' x', y'. The
    matching row of ``analysis.centroids`` is the same mean rounded to
    0.01 nm by ``tools.cluster_quality.good_cluster_centroids``, so the
    two agree to 0.005 nm.

    H2 fields: ``sigma_centroid_nm`` is the per-axis centroid precision
    from ``bootstrap_centroid_sigma`` over the cluster's emission events
    (``sigma_centroid_x_nm``/``_y_nm`` its two axes, ``n_bootstrap`` the
    draws; NaN and 0 without frames or with fewer than two events);
    ``suspect`` holds the marks of ``SUSPECT_MARKS`` that apply, in that
    order, empty when clean. There is no "dbcv" mark: the DBCV criterion
    is off by default here because it correlates with cluster size (see
    the module docstring). ``mean_frame_fraction`` and
    ``max_window_share`` are the two statistics the burst rule read
    (NaN without frames), kept so a report can say why a cluster was
    marked.
    """

    ring: int
    label: int                              # DBSCAN label within the ring's slab arrays
    centroid_nm: NDArray[np.float64]        # (2,) x', y' in the axon frame (nm)
    n_locs: int
    n_events: Optional[int]                 # distinct link events with >= 1 localization here; None without frames
    loc_index: NDArray[np.intp]             # indices into the arrays build_rings received
    z_values_nm: NDArray[np.float64]        # z' of its localizations, in loc_index order
    lpz_median_nm: float                    # NaN without lpz
    area_nm2: float                         # from the ring analysis, NaN if unavailable
    rho_nm: float                           # sqrt(area / pi), NaN with area NaN
    sigma_centroid_nm: Optional[float] = None   # H2: bootstrap over events (float or NaN once built)
    suspect: Tuple[str, ...] = ()               # H2: marks of SUSPECT_MARKS ("few_events", "burst", "edge")
    sigma_centroid_x_nm: float = float("nan")   # H2: sd of the draw means along x'
    sigma_centroid_y_nm: float = float("nan")   # H2: idem along y'
    n_bootstrap: int = 0                        # H2: draws behind sigma_centroid_nm; 0 = not bootstrapped
    mean_frame_fraction: float = float("nan")   # H2: mean frame / acquisition length (burst rule)
    max_window_share: float = float("nan")      # H2: largest share of one twentieth of the acquisition


@dataclass
class Ring:
    """One ring: an axial segment of the axon frame and its analysis.

    ``relative_depth_lo``/``relative_depth_hi`` are the depth of the axial
    density valley at the lower/upper boundary of this ring's component
    (0 = no dip, 1 = density falls to zero); None at the outer boundaries,
    where there is no neighbouring component and hence no valley.
    """

    index: int
    component_index: int
    z_lo_nm: float
    z_hi_nm: float
    guard_nm: float
    centre_z_nm: float
    sigma_z_nm: float                       # fitted component width
    lpz_median_nm: float                    # NaN without lpz
    relative_depth_lo: Optional[float]
    relative_depth_hi: Optional[float]
    n_locs: int
    n_events: Optional[int]
    loc_index: NDArray[np.intp]             # indices into the arrays build_rings received
    event_id: Optional[NDArray[np.int64]]   # per localization of loc_index; -1 where not linked; None without frames
    analysis: AxonAnalysis                  # the AxonAnalysis of this segment (existing pipeline)
    clusters: List[Cluster]
    contour_nm: Optional[NDArray[np.float64]]   # (K, 2) axon-frame contour through kept centroids
    length_nm: float                        # NaN without contour
    centroid_nm: Optional[NDArray[np.float64]]  # (2,) area centroid of the contour
    warnings: List[str] = field(default_factory=list)


@dataclass
class RingsParams:
    """Everything ``build_rings`` decides with, recorded in the result."""

    segment_mode: str = "valley"
    guard_nm: float = 0.0
    half_width_nm: float = 90.0
    eps_nm: float = 25.0
    # DBSCAN min_samples; ``analyze_all_segments`` also uses it as the
    # minimum localizations of an axial segment (it derives that from
    # min_samples and accepts no separate value), so there is no
    # ``min_locs`` here: a field that is recorded but changes nothing
    # would misstate the provenance.
    min_samples: int = 10
    correct_tilt: bool = True
    axis_min_locs: int = 200
    # Link radius = factor x median lateral precision. The step between
    # two consecutive localizations of one molecule has sd sqrt(2) * lp
    # per axis, so a step exceeds the radius with probability
    # exp(-factor^2 / 4): 2.5 -> 21 % (1.66 events per fluorophore
    # measured), 3 -> 10.5 % (1.33), 4 -> 1.8 % (1.06), 5 -> 0.2 % (1.00).
    # 5 is the smallest integer factor that keeps n_events within the
    # +/-5 % of the true fluorophores the acceptance asks for; the
    # expected lost-step fraction is recorded in the result.
    link_radius_factor: float = 5.0
    link_radius_default_nm: float = 25.0    # used (with a warning) when no lateral precision is given
    link_max_dark_time: int = 1
    posterior_min: Optional[float] = None   # None = primary path; e.g. 0.85 drops ambiguous localizations
    random_seed: int = 0
    # H2. Bootstrap draws behind each cluster's centroid precision: the
    # sd of a sd from 200 draws is 1 / sqrt(2 * 199) = 5 %, below the
    # per-cluster noise of the estimate itself (an sd of E = 4 event
    # means is uncertain by ~40 %), so more draws buy nothing here.
    n_bootstrap: int = 200
    # A cluster with fewer events than this carries "few_events": one or
    # two molecules seen many times is what an axially leaked molecule
    # looks like, and two events give an sd of two values.
    min_events_per_cluster: int = 3


@dataclass
class RemovedCluster:
    """One cluster ``clean_rings`` took out of the rings (H5-D, D-35).

    ``ring`` is the ``Ring.index`` and ``position`` the cluster's index in
    that ring's ``Ring.clusters`` of the INPUT the cleaning was given (the
    rings of ``build_rings`` for a first cleaning); ``label``,
    ``centroid_nm`` (x', y', a copy) and ``n_locs`` are the cluster's own,
    so the entry names it even after the positions of the rest shift;
    ``reason`` is the caller's (the lumen rule's terms, "manual", ...;
    "" when none was given).
    """

    ring: int
    position: int
    label: int
    centroid_nm: NDArray[np.float64]
    n_locs: int
    reason: str = ""


@dataclass
class RingsResult:
    """What ``build_rings`` returns: the frame, the rings and the bookkeeping.

    ``x_p``, ``y_p``, ``z_p`` and ``event_id`` cover EVERY input
    localization, in input order, whether or not it fell in a ring.

    ``frame.ring_n_locs``, ``ring_used`` and ``ring_normals`` describe the
    segmentation the LAST axis stage fitted, in that stage's frame; the
    final rings come from one more segmentation in the composed frame and
    can differ from them by a few localizations near the boundaries.
    ``frame.origin_nm`` is the mean of the localizations of the rings that
    entered that last stage (of every localization when no ring stage
    could be fitted).
    """

    frame: AxonFrame
    rings: List[Ring]
    ms: MultiSegmentAnalysis                # computed in the axon frame
    x_p: NDArray[np.float64]
    y_p: NDArray[np.float64]
    z_p: NDArray[np.float64]
    event_id: Optional[NDArray[np.int64]]   # per input localization; None without frames
    link_radius_nm: Optional[float]
    n_events: Optional[int]
    ambiguous: Optional[NDArray[np.bool_]]  # per input localization; None when posterior_min is None
    n_ambiguous: int
    params: RingsParams
    source_name: str
    warnings: List[str] = field(default_factory=list)
    # Rotation applied by each axis stage of the chosen start, in degrees:
    # the start first, then one entry per ring stage. Empty with
    # correct_tilt False.
    axis_stages_beta_deg: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    # Which start the frame came from (AXIS_START_LAB_RINGS or
    # AXIS_START_SHARPNESS; "identity" when the tilt was not corrected or
    # no start could be estimated), and the refined tilt of each start in
    # that order (NaN for a start that could not be estimated).
    axis_start: str = METHOD_IDENTITY
    axis_candidate_beta_deg: NDArray[np.float64] = field(
        default_factory=lambda: np.full(2, np.nan))
    # What the choice between the starts was made on: the total relative
    # depth of the density valleys of each start's last segmentation, in
    # the same order; the larger wins when the starts disagree. -inf for
    # a start whose refinement fitted no ring stage, NaN when it could
    # not be estimated.
    axis_candidate_valley_depth: NDArray[np.float64] = field(
        default_factory=lambda: np.full(2, np.nan))
    # Expected fraction of consecutive-frame steps of one molecule longer
    # than the link radius, exp(-(radius / median lp)^2 / 4): what the
    # radius costs in split events. None without frames or without a
    # lateral precision to relate the radius to.
    link_lost_step_fraction: Optional[float] = None
    # H2. Clusters carrying each suspect mark, over every ring (a cluster
    # with two marks counts once under each). Nothing is removed from the
    # rings because of a mark; H3 measures what excluding them changes.
    n_suspect: Dict[str, int] = field(default_factory=dict)
    # H2. The acquisition length the burst rule used: the ``n_frames``
    # given to ``build_rings``, or max(frame) + 1 when it was not (a
    # lower bound, with a warning); None without frame numbers.
    n_frames: Optional[int] = None
    # H5-D. The clusters ``clean_rings`` took out of these rings, what they
    # were and why, in the order they were taken (cumulative over
    # successive cleanings); empty for the rings of ``build_rings``.
    removed_clusters: List[RemovedCluster] = field(default_factory=list)

    @property
    def n_rings(self) -> int:
        return len(self.rings)


# ============================================================================
# Ring labels from segments
# ============================================================================

def ring_index_from_segments(
    z_nm: NDArray[np.float64],
    segments: Sequence[AxialSegment],
) -> NDArray[np.int64]:
    """
    Ring label per localization from a list of axial slabs.

    The label is the position of the segment in ``segments`` (not its
    ``component_index``), so that row k of the per-ring outputs of
    ``fit_axon_frame`` is segment k. Membership is the hard mask
    ``(z >= zmin) & (z <= zmax)``, the same rule ``analyze_axon`` applies
    to a slab; -1 outside every slab (and for a non-finite z).

    Slabs can overlap ("paper" mode, where consecutive slabs overlap
    whenever the period is shorter than 2 * half_width). A localization
    inside two slabs then goes to the one whose centre is nearer: a plane
    fit needs each localization in ONE ring, and the nearer centre is the
    component more likely to have produced it when the two are of similar
    width. An exact tie goes to the lower segment.

    Parameters
    ----------
    z_nm : (M,) array
        Axial coordinate of each localization, nm.
    segments : sequence of AxialSegment
        From ``find_axial_segments``, in ascending axial order.

    Returns
    -------
    (M,) int64 array
        Segment position per localization, -1 where in none.
    """
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    index = np.full(z.shape, -1, dtype=np.int64)
    best = np.full(z.shape, np.inf)
    for k, seg in enumerate(segments):
        inside = (z >= seg.zmin_nm) & (z <= seg.zmax_nm)
        distance = np.abs(z - seg.center_nm)
        take = inside & (distance < best)
        index[take] = k
        best[take] = distance[take]
    return index


# ============================================================================
# Ambiguous localizations by posterior
# ============================================================================

def posterior_ambiguous(
    z_nm: NDArray[np.float64],
    z_result: ZPeriodicityResult,
    posterior_min: float,
) -> NDArray[np.bool_]:
    """
    Localizations whose most probable axial component has a posterior
    probability below ``posterior_min``.

    The posterior is that of the dominant components only (``means_nm``,
    ``sigmas_nm``, ``weights`` of the fit), with the weights renormalised
    to sum to one: a ring is a dominant component, and the question is
    "which ring", not "which of the fitted Gaussians". Near the valley
    between two rings the answer is close to a coin toss, and those are
    the localizations a hard z mask assigns with false confidence.

    Computed in log space, row-centred on the largest term, so a
    localization far from every component does not underflow to 0/0.

    Parameters
    ----------
    z_nm : (M,) array
        Axial coordinate per localization, nm.
    z_result : ZPeriodicityResult
        The mixture fitted on these z values.
    posterior_min : float
        Threshold on the maximum posterior; below it the localization is
        ambiguous.

    Returns
    -------
    (M,) bool array
        True where ambiguous. With fewer than two dominant components
        every localization has posterior 1, so nothing is ambiguous; a
        non-finite z has no posterior and is never flagged.
    """
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    means = np.asarray(z_result.means_nm, dtype=np.float64).ravel()
    sigmas = np.asarray(z_result.sigmas_nm, dtype=np.float64).ravel()
    weights = np.asarray(z_result.weights, dtype=np.float64).ravel()
    if means.size < 2 or means.size != sigmas.size or means.size != weights.size:
        return np.zeros(z.shape, dtype=bool)
    sigmas = np.maximum(sigmas, _MIN_SIGMA_NM)
    weights = weights / weights.sum()
    flags = np.zeros(z.shape, dtype=bool)
    finite = np.isfinite(z)
    if not finite.any():
        return flags
    zf = z[finite]
    logp = (np.log(weights)[None, :] - np.log(sigmas)[None, :]
            - 0.5 * ((zf[:, None] - means[None, :]) / sigmas[None, :]) ** 2)
    logp -= logp.max(axis=1, keepdims=True)
    posterior = np.exp(logp)
    posterior /= posterior.sum(axis=1, keepdims=True)
    flags[finite] = posterior.max(axis=1) < posterior_min
    return flags


# ============================================================================
# The axon frame, in stages
# ============================================================================

def _angle_to_z_deg(u: NDArray[np.float64]) -> float:
    """Angle between ``u`` and the z axis; atan2 keeps its digits near 0."""
    return math.degrees(math.atan2(math.hypot(float(u[0]), float(u[1])), float(u[2])))


def _stage_warnings(label: str, stage: AxonFrame) -> List[str]:
    """A stage's warnings, labelled; its tilt warning is dropped because
    the tilt that matters is the composed one, warned about once below."""
    return [f"axis {label}: {w}" for w in stage.warnings
            if not w.startswith("axis tilt beta")]


@dataclass
class _AxisCandidate:
    """One start refined by per-ring stages, and what the choice needs."""

    label: str
    rotation: NDArray[np.float64]           # composed, lab -> final frame
    increments_deg: List[float]             # the start, then each ring stage
    last: Optional[AxonFrame]               # last per-ring stage; None if none fitted
    rotation_before_last: NDArray[np.float64]
    used_mask: Optional[NDArray[np.bool_]]  # localizations of the rings the last stage used
    sharpness: float                        # profile_sharpness of z' over every localization
    valley_depth: float                     # total relative depth of the valleys of the last stage's segmentation; -inf without one
    settled: bool                           # the last stage rotated by less than AXIS_STAGE_TOL_DEG
    warnings: List[str]

    @property
    def u(self) -> NDArray[np.float64]:
        # rotation @ u == z_hat, so u is rotation^T z_hat: its third row.
        return np.asarray(self.rotation[2, :], dtype=np.float64)

    @property
    def beta_deg(self) -> float:
        return _angle_to_z_deg(self.u)


@dataclass
class _StagedAxis:
    """What ``_fit_frame_staged`` returns."""

    frame: AxonFrame
    increments_deg: NDArray[np.float64]
    start: str
    candidate_beta_deg: NDArray[np.float64]
    candidate_valley_depth: NDArray[np.float64]


def _refine_from_start(
    label: str,
    start: AxonFrame,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    origin: NDArray[np.float64],
    params: RingsParams,
) -> _AxisCandidate:
    """
    Per-ring stages from one starting frame, all rotating about ``origin``.

    Each stage segments z in the current frame, fits one plane per ring
    (``fit_axon_frame``) and rotates; it stops when a stage rotates by
    less than ``AXIS_STAGE_TOL_DEG``, when no plane can be fitted, or
    after ``AXIS_MAX_RING_STAGES``. The total valley depth of the last
    stage's segmentation is what ``_fit_frame_staged`` chooses candidates
    by; the sharpness of the final z' profile is recorded for the report.
    """
    n = int(x.size)
    rotation = np.eye(3)
    x_c, y_c, z_c = x, y, z
    increments: List[float] = []
    warnings_: List[str] = []

    def apply(stage: AxonFrame) -> None:
        nonlocal rotation, x_c, y_c, z_c
        about_origin = dataclasses.replace(stage, origin_nm=origin)
        x_c, y_c, z_c = to_axon_frame(about_origin, x_c, y_c, z_c)
        rotation = np.asarray(stage.rotation @ rotation, dtype=np.float64)
        increments.append(float(stage.beta_deg))

    apply(start)
    last: Optional[AxonFrame] = None
    rotation_before_last = np.eye(3)
    used_mask: Optional[NDArray[np.bool_]] = None
    valley_depth = -math.inf
    settled = False
    for k in range(1, AXIS_MAX_RING_STAGES + 1):
        segments, z_result, valleys, _ = find_axial_segments(
            z_c, half_width_nm=params.half_width_nm, mode=params.segment_mode,
            min_locs=int(params.min_samples), guard_nm=params.guard_nm,
        )
        ring_index = ring_index_from_segments(z_c, segments)
        ring_index[posterior_ambiguous(z_c, z_result, AXIS_STAGE_POSTERIOR_MIN)] = -1
        stage = fit_axon_frame(x_c, y_c, z_c, ring_index,
                               min_locs=int(params.axis_min_locs))
        warnings_.extend(_stage_warnings(f"stage {k} (rings)", stage))
        if stage.method == METHOD_IDENTITY:
            warnings_.append(f"axis stage {k}: no plane could be fitted on the "
                             "rings; the frame stops at the previous stage")
            break
        rotation_before_last = rotation
        last = stage
        in_ring = ring_index >= 0
        used_mask = np.zeros(n, dtype=bool)
        used_mask[in_ring] = stage.ring_used[ring_index[in_ring]]
        valley_depth = (float(np.sum(valleys.relative_depth))
                        if valleys is not None and valleys.n_boundaries else 0.0)
        apply(stage)
        if stage.beta_deg < AXIS_STAGE_TOL_DEG:
            settled = True
            break
    else:
        warnings_.append(
            f"axis refinement did not settle after {AXIS_MAX_RING_STAGES} ring "
            f"stages (last increment {increments[-1]:.3f} deg, tolerance "
            f"{AXIS_STAGE_TOL_DEG:g} deg)"
        )
    return _AxisCandidate(
        label=label, rotation=rotation, increments_deg=increments, last=last,
        rotation_before_last=rotation_before_last, used_mask=used_mask,
        sharpness=profile_sharpness(z_c), valley_depth=valley_depth,
        settled=settled, warnings=warnings_,
    )


def _fit_frame_staged(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    params: RingsParams,
) -> _StagedAxis:
    """
    The axon frame: two starts refined by per-ring planes, the better
    resolved kept.

    See the module docstring for why one per-ring pass is not enough and
    why a pooled plane is not a safe start. The starts are (A) the
    specification's lab-frame per-ring fit and (B)
    ``fit_axis_by_profile_sharpness``; each is refined by
    ``_refine_from_start`` about one origin (the mean of all
    localizations), so that the stages compose into one rotation. When
    the refined axes agree the specification's start is kept (unless it
    alone failed to settle); when they disagree, the candidate whose
    last segmentation is better resolved wins: the total relative depth
    of the density valleys between its rings. In a wrong frame the rings
    are smeared into each other and the axial density has no deep
    valleys, however the mixture is cut (see the module docstring for
    the two statistics that were tried and fooled). The composed frame
    is then re-expressed about the mean of the localizations of the
    rings the last stage used (the specification's origin): same
    rotation, a constant shift of x', y', z'.

    Returns
    -------
    _StagedAxis
        The composed ``AxonFrame``, the rotation each stage of the chosen
        start applied (start first), the label of that start and the
        refined tilt of both starts. An identity frame (with a warning)
        when neither start could be estimated.
    """
    n = int(x.size)
    origin_all = np.array([x.mean(), y.mean(), z.mean()], dtype=np.float64)
    warnings_: List[str] = []
    candidate_beta = np.full(2, np.nan)
    candidate_depth = np.full(2, np.nan)

    # The starts. (A) is the specification's step 1-2 verbatim.
    segments0, _, _, _ = find_axial_segments(
        z, half_width_nm=params.half_width_nm, mode=params.segment_mode,
        min_locs=int(params.min_samples), guard_nm=params.guard_nm,
    )
    start_a = fit_axon_frame(x, y, z, ring_index_from_segments(z, segments0),
                             min_locs=int(params.axis_min_locs))
    start_b = fit_axis_by_profile_sharpness(x, y, z)
    candidates: List[_AxisCandidate] = []
    for position, (label, start) in enumerate(
            ((AXIS_START_LAB_RINGS, start_a), (AXIS_START_SHARPNESS, start_b))):
        warnings_.extend(_stage_warnings(f"start '{label}'", start))
        if start.method == METHOD_IDENTITY:
            warnings_.append(f"axis start '{label}' could not be estimated; skipped")
            continue
        cand = _refine_from_start(label, start, x, y, z, origin_all, params)
        candidate_beta[position] = cand.beta_deg
        candidate_depth[position] = cand.valley_depth
        candidates.append(cand)
    if not candidates:
        warnings_.append("no axis start could be estimated: identity frame, "
                         "no tilt correction")
        return _StagedAxis(
            frame=identity_axon_frame(None, origin_all, min_locs=int(params.axis_min_locs),
                                      warnings=warnings_),
            increments_deg=np.array([], dtype=np.float64),
            start=METHOD_IDENTITY, candidate_beta_deg=candidate_beta,
            candidate_valley_depth=candidate_depth,
        )

    # The choice: when the starts agree, the specification's unless it
    # alone did not settle; when they disagree, the better resolved
    # segmentation (ties keep the first).
    best = candidates[0]
    if len(candidates) == 2:
        apart = math.degrees(math.atan2(
            float(np.linalg.norm(np.cross(candidates[0].u, candidates[1].u))),
            float(candidates[0].u @ candidates[1].u)))
        if apart <= AXIS_STAGE_TOL_DEG:
            if candidates[1].settled and not candidates[0].settled:
                best = candidates[1]
        else:
            best = max(candidates, key=lambda c: c.valley_depth)
        if apart > AXIS_CANDIDATE_DISAGREEMENT_DEG:
            other = candidates[1] if best is candidates[0] else candidates[0]
            warnings_.append(
                f"the two axis starts converged to axes {apart:.2f} deg apart "
                f"('{candidates[0].label}' {candidates[0].beta_deg:.2f} deg, "
                f"'{candidates[1].label}' {candidates[1].beta_deg:.2f} deg): the "
                f"'{other.label}' start stopped in a segmentation that mixes rings, "
                "which happens when the rings' lateral footprints differ; the "
                f"frame whose rings are better resolved ('{best.label}', total valley "
                f"depth {best.valley_depth:.2f} vs {other.valley_depth:.2f}) is kept"
            )
    warnings_.extend(best.warnings)

    # Compose and re-express about the specification's origin.
    rotation = best.rotation
    u = best.u.copy()
    if u[2] < 0.0:                       # cannot follow from small stages; kept as the contract
        u = -u
    beta = _angle_to_z_deg(u)
    if best.last is not None and best.used_mask is not None and best.used_mask.any():
        source = best.last
        m = best.used_mask
        origin = np.array([x[m].mean(), y[m].mean(), z[m].mean()], dtype=np.float64)
        # Normals of the last stage live in the frame before it:
        # n_lab = R_prev^T n, i.e. as row vectors n @ R_prev.
        normals = np.asarray(source.ring_normals @ best.rotation_before_last,
                             dtype=np.float64)
        method = source.method
        delta_beta = float(source.delta_beta_deg)
    else:
        source = start_a if best.label == AXIS_START_LAB_RINGS else start_b
        origin = origin_all
        normals = np.asarray(source.ring_normals, dtype=np.float64)
        method = METHOD_ALL if source.method == METHOD_ALL else source.method
        delta_beta = float("nan")
        warnings_.append("no ring stage could be fitted: the frame is the start "
                         f"'{best.label}' alone (method '{method}'); delta_beta undefined")
    if beta > TILT_WARNING_DEG:
        warnings_.append(
            f"axis tilt beta = {beta:.2f} deg exceeds {TILT_WARNING_DEG:g} deg: "
            "check the section and the ring labels before trusting the frame"
        )
    frame = AxonFrame(
        u=u,
        origin_nm=origin,
        beta_deg=beta,
        delta_beta_deg=delta_beta,
        ring_normals=normals,
        ring_n_locs=np.asarray(source.ring_n_locs, dtype=np.int64),
        ring_used=np.asarray(source.ring_used, dtype=bool),
        rotation=rotation,
        method=method,
        min_locs=int(params.axis_min_locs),
        loss=source.loss,
        warnings=warnings_,
        ring_f_scale_nm=np.asarray(source.ring_f_scale_nm, dtype=np.float64),
        n_locs_used=int(source.n_locs_used),
        profile_sharpness=float(best.sharpness),
    )
    return _StagedAxis(
        frame=frame,
        increments_deg=np.asarray(best.increments_deg, dtype=np.float64),
        start=best.label,
        candidate_beta_deg=candidate_beta,
        candidate_valley_depth=candidate_depth,
    )


# ============================================================================
# build_rings
# ============================================================================

def _as_float(values: NDArray[np.float64], name: str, n: int) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64).ravel()
    if out.size != n:
        raise ValueError(f"{name} has {out.size} values for {n} localizations")
    return out


def _nanmedian_or_nan(values: Optional[NDArray[np.float64]],
                      index: NDArray[np.intp]) -> float:
    """Median of the finite values at ``index``; NaN without any."""
    if values is None or index.size == 0:
        return float("nan")
    picked = values[index]
    finite = picked[np.isfinite(picked)]
    return float(np.median(finite)) if finite.size else float("nan")


def _count_events(event_id: Optional[NDArray[np.int64]],
                  index: NDArray[np.intp]) -> Optional[int]:
    """Distinct event ids at ``index``, ignoring -1; None without events."""
    if event_id is None:
        return None
    ids = event_id[index]
    return int(np.unique(ids[ids >= 0]).size)


def _link_events(
    frame: NDArray[np.int64],
    x_p: NDArray[np.float64],
    y_p: NDArray[np.float64],
    lp_lateral_nm: Optional[NDArray[np.float64]],
    params: RingsParams,
    warnings_: List[str],
) -> Tuple[NDArray[np.int64], float, Optional[float]]:
    """Step 5: one event id per input localization, the radius used and
    the expected fraction of one molecule's consecutive-frame steps that
    exceed it, exp(-(radius / median lp)^2 / 4) (the step has sd
    sqrt(2) * lp per axis); None when no lateral precision relates the
    radius to anything."""
    radius: Optional[float] = None
    lost: Optional[float] = None
    if lp_lateral_nm is not None:
        finite = lp_lateral_nm[np.isfinite(lp_lateral_nm)]
        median_lp = float(np.median(finite)) if finite.size else float("nan")
        if np.isfinite(median_lp) and median_lp > 0.0:
            radius = params.link_radius_factor * median_lp
        else:
            warnings_.append(
                "lateral precision given but without a finite positive median: "
                f"link radius fell back to the default {params.link_radius_default_nm:g} nm"
            )
    else:
        warnings_.append(
            "no lateral precision given: link radius set to the default "
            f"{params.link_radius_default_nm:g} nm instead of "
            f"{params.link_radius_factor:g} x the median precision"
        )
    if radius is None:
        radius = float(params.link_radius_default_nm)
    else:
        lost = float(math.exp(-(radius / median_lp) ** 2 / 4.0))
    event_id = link_localizations(frame, x_p, y_p, radius,
                                  max_dark_time=int(params.link_max_dark_time))
    return np.asarray(event_id, dtype=np.int64), float(radius), lost


def _build_clusters(
    ring_index: int,
    an: AxonAnalysis,
    loc_index: NDArray[np.intp],
    x_p: NDArray[np.float64],
    y_p: NDArray[np.float64],
    z_p: NDArray[np.float64],
    lpz_nm: Optional[NDArray[np.float64]],
    event_id: Optional[NDArray[np.int64]],
    warnings_: List[str],
) -> List[Cluster]:
    """One ``Cluster`` per kept cluster, in the order of ``an.centroids``.

    ``good_cluster_labels`` lists the labels in the order
    ``good_cluster_centroids`` produced the rows of ``an.centroids``
    (ascending label, noise and bad labels skipped), so row i of the
    centroids is the cluster labelled ``labels[i]``. ``centroid_nm`` is
    recomputed as the exact mean of the cluster's localizations (the
    centroid row is that mean rounded to 0.01 nm). The areas come from
    ``compute_cluster_areas`` with the same exclusions and are looked up
    by label rather than by position, so a mismatch is a NaN with a
    warning, never a wrong area.
    """
    labels = good_cluster_labels(an.labels, an.bad_report.bad_labels)
    centroids = np.asarray(an.centroids, dtype=np.float64).reshape(-1, 2)
    if labels.size != centroids.shape[0]:
        warnings_.append(
            f"ring {ring_index}: {labels.size} kept labels for "
            f"{centroids.shape[0]} centroid rows; clusters not built"
        )
        return []
    areas = an.areas
    clusters: List[Cluster] = []
    for i, label in enumerate(labels.tolist()):
        member = np.asarray(an.labels == label)
        cl_index = np.asarray(loc_index[member], dtype=np.intp)
        area = float("nan")
        if areas is not None:
            where = np.flatnonzero(np.asarray(areas.labels) == label)
            if where.size == 1:
                area = float(areas.areas_nm2[where[0]])
            else:
                warnings_.append(
                    f"ring {ring_index}, cluster {label}: {where.size} area entries "
                    "for this label; area_nm2 set to NaN"
                )
        clusters.append(Cluster(
            ring=ring_index,
            label=int(label),
            centroid_nm=np.array([x_p[cl_index].mean(), y_p[cl_index].mean()],
                                 dtype=np.float64),
            n_locs=int(cl_index.size),
            n_events=_count_events(event_id, cl_index),
            loc_index=cl_index,
            z_values_nm=np.asarray(z_p[cl_index], dtype=np.float64),
            lpz_median_nm=_nanmedian_or_nan(lpz_nm, cl_index),
            area_nm2=area,
            rho_nm=float(np.sqrt(area / np.pi)) if np.isfinite(area) else float("nan"),
        ))
    return clusters


# ============================================================================
# H2: centroid precision by bootstrap over emission events
# ============================================================================

def bootstrap_centroid_sigma(
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    event_id: NDArray[np.int64],
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    Precision of a cluster's centroid by a bootstrap over its emission
    events.

    Why events and not localizations: the localizations of one event are
    repeated measurements of the same molecule, and their errors are
    correlated through the molecule's position and the antibody offset
    (02_investigacion.md B4, D-03). A bootstrap that resamples
    localizations treats those k copies as k independent observations
    and understates the centroid's uncertainty (measured in
    validate_columns_h2.py: 0.58 of the truth with k = 5). The
    independent units are the events, so those are what is resampled.

    Algorithm: the localizations are grouped by ``event_id``, an id of -1
    ("not linked") making every such localization its own event; a draw
    resamples the E events with replacement, concatenates their
    localizations and takes the mean x and mean y (computed as the sum of
    the drawn events' coordinate sums over the sum of their counts, which
    is that mean); ``sigma_x_nm`` and ``sigma_y_nm`` are the standard
    deviations (ddof = 1) of the draw means and ``sigma_nm =
    sqrt((sigma_x^2 + sigma_y^2) / 2)`` is the per-axis precision.
    Deterministic given ``rng``.

    Parameters
    ----------
    x_nm, y_nm : (n,) arrays
        Coordinates of the cluster's localizations, nm.
    event_id : (n,) int array
        Event id per localization; -1 where not linked.
    n_draws : int
        Bootstrap draws, at least ``MIN_BOOTSTRAP_DRAWS``.
    rng : numpy.random.Generator
        Source of the draws.

    Returns
    -------
    (sigma_nm, sigma_x_nm, sigma_y_nm)
        All three NaN with fewer than two events: there is nothing to
        resample, and a cluster of one molecule has no measurable
        centroid precision.

    Raises
    ------
    ValueError
        Mismatched lengths, or ``n_draws`` below ``MIN_BOOTSTRAP_DRAWS``.

    Notes
    -----
    Bias with few events. The bootstrap resamples the E observed event
    means, whose spread is the plug-in (ddof = 0) variance of the sample,
    so the bootstrap sd of the mean is on average sqrt((E - 1) / E) of
    the truth, and a little less again because the mean of an sd
    underestimates the sd of the mean (Jensen). Measured (2000
    realisations per E, one localization per event, sd 10 nm, 200
    draws): 0.77 of the truth at E = 3, 0.83 at E = 4, 0.94 at E = 10,
    0.98 at E = 30. ``ddof = 1`` over the DRAWS does not remove it, since
    the draws are many. No correction is applied here: D-03 and the H2
    specification define the estimator as the sd of the draw means, and
    the number a frozen file records must be that estimator. The
    consequence for tau_0 (D-03) is that its 2 * sigma term is slightly
    anti-conservative for clusters near ``min_events_per_cluster``
    (3-4 events: 17-23 % on their own sigma); the rho term is unaffected.
    H3's sensitivity with and without the "few_events" clusters covers
    it, and the H2 report states it next to the sigma_m quantiles.
    """
    x = np.asarray(x_nm, dtype=np.float64).ravel()
    y = np.asarray(y_nm, dtype=np.float64).ravel()
    ids = np.asarray(event_id, dtype=np.int64).ravel()
    n = int(x.size)
    if y.size != n or ids.size != n:
        raise ValueError(
            f"x_nm, y_nm and event_id must have one value per localization "
            f"(got {n}, {y.size}, {ids.size})")
    draws = int(n_draws)
    if draws < MIN_BOOTSTRAP_DRAWS:
        raise ValueError(f"n_draws must be at least {MIN_BOOTSTRAP_DRAWS}, got {draws}")
    nan = float("nan")
    if n == 0:
        return nan, nan, nan
    unlinked = ids < 0
    if unlinked.any():
        ids = ids.copy()
        ids[unlinked] = int(ids.max()) + 1 + np.arange(int(unlinked.sum()), dtype=np.int64)
    _, inverse = np.unique(ids, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.intp).ravel()
    n_events = int(inverse.max()) + 1
    if n_events < 2:
        return nan, nan, nan
    sum_x = np.bincount(inverse, weights=x, minlength=n_events)
    sum_y = np.bincount(inverse, weights=y, minlength=n_events)
    count = np.bincount(inverse, minlength=n_events).astype(np.float64)
    pick = rng.integers(0, n_events, size=(draws, n_events))
    n_drawn = count[pick].sum(axis=1)
    mean_x = sum_x[pick].sum(axis=1) / n_drawn
    mean_y = sum_y[pick].sum(axis=1) / n_drawn
    sigma_x = float(np.std(mean_x, ddof=1))
    sigma_y = float(np.std(mean_y, ddof=1))
    return math.sqrt((sigma_x * sigma_x + sigma_y * sigma_y) / 2.0), sigma_x, sigma_y


def _fill_cluster_precision(
    clusters: List[Cluster],
    x_p: NDArray[np.float64],
    y_p: NDArray[np.float64],
    event_id: Optional[NDArray[np.int64]],
    params: RingsParams,
    ring_seed: np.random.SeedSequence,
) -> None:
    """
    H2 step: ``sigma_centroid_nm`` (and its two axes, ``n_bootstrap``)
    of every cluster, in place.

    The rng of cluster i is ``ring_seed.spawn(len(clusters))[i]``, and
    ``ring_seed`` is the ring's child of ``SeedSequence(params.random_seed)``
    (see ``build_rings``): the sequence a cluster draws from depends only
    on the seed, the ring index and the cluster position, so the value is
    the same whatever else runs and however many clusters or rings there
    are. Without events, or with fewer than ``MIN_BOOTSTRAP_DRAWS`` draws
    asked for, every sigma is NaN and ``n_bootstrap`` 0.
    """
    if event_id is None or int(params.n_bootstrap) < MIN_BOOTSTRAP_DRAWS or not clusters:
        for cl in clusters:
            cl.sigma_centroid_nm = float("nan")
            cl.sigma_centroid_x_nm = float("nan")
            cl.sigma_centroid_y_nm = float("nan")
            cl.n_bootstrap = 0
        return
    seeds = ring_seed.spawn(len(clusters))
    for cl, seed in zip(clusters, seeds):
        index = np.asarray(cl.loc_index, dtype=np.intp)
        sigma, sigma_x, sigma_y = bootstrap_centroid_sigma(
            x_p[index], y_p[index], event_id[index],
            n_draws=int(params.n_bootstrap), rng=np.random.default_rng(seed))
        cl.sigma_centroid_nm = sigma
        cl.sigma_centroid_x_nm = sigma_x
        cl.sigma_centroid_y_nm = sigma_y
        cl.n_bootstrap = int(params.n_bootstrap)


# ============================================================================
# H2: suspect marks
# ============================================================================

def _acquisition_length(
    frame_i: NDArray[np.int64],
    n_frames: Optional[int],
    warnings_: List[str],
) -> int:
    """
    The acquisition length the burst rule divides the movie by.

    ``n_frames`` is the length from the metadata
    (``tools.mps_io.Localizations.n_frames``); without it the largest
    frame number + 1 is used, which can only UNDERESTIMATE the length (a
    movie whose last localizations were filtered out looks shorter), so
    the middle 20-80 % and the twentieths shift and a warning says so.
    """
    last = int(frame_i.max()) if frame_i.size else -1
    if n_frames is None:
        total = last + 1
        warnings_.append(
            f"acquisition length not given (n_frames is None): max(frame) + 1 = "
            f"{total} frames used for the burst mark, which is a lower bound of "
            "the acquisition length (Localizations.n_frames carries the metadata "
            "value); the rule's windows shift if the movie was longer"
        )
        return total
    total = int(n_frames)
    if total <= 0:
        raise ValueError(f"n_frames must be a positive frame count, got {n_frames!r}")
    if last >= total:
        warnings_.append(
            f"n_frames = {total} but the largest frame number is {last}: frames "
            "beyond the acquisition length fall outside the burst rule's windows"
        )
    return total


def _edge_suspects(an: AxonAnalysis) -> Set[int]:
    """
    Labels that touch the ROI boundary and were kept because the runaway
    guard disabled the edge criterion.

    ``identify_bad_clusters`` empties ``bad_report.edge_touching`` when
    its guard fires (and the export column ``removed_edge_touching``
    reads that set, so it is not changed here), so the touching clusters
    are recomputed from the ROI the analysis measured against, with the
    same margin (``eps_nm``) and on the same coordinates (the slab's).
    On the convex-hull path (no ROI forwarded) nothing is recomputed: a
    hull through the ring's own clusters cuts nothing, so no centroid or
    area is biased by it, and the guard fires there on every ring of
    clusters by construction.
    """
    report = an.bad_report
    if report.edge_criterion_disabled is None:
        return set()
    flagged: Set[int] = {int(label) for label in report.edge_touching}
    if an.roi is not None and np.asarray(an.labels).size:
        flagged |= find_edge_touching_clusters(
            np.asarray(an.x_slab, dtype=np.float64), np.asarray(an.y_slab, dtype=np.float64),
            np.asarray(an.labels, dtype=np.int64), an.roi, edge_margin_nm=float(an.eps_nm))
    return flagged


def _mark_suspects(
    clusters: List[Cluster],
    an: AxonAnalysis,
    ring_index: int,
    frame_ring: Optional[NDArray[np.int64]],
    n_frames_total: Optional[int],
    params: RingsParams,
    warnings_: List[str],
) -> None:
    """
    H2 step: ``Cluster.suspect`` of every kept cluster of one ring, in
    place, in the order of ``SUSPECT_MARKS``.

    The burst rule is ``tools.mps_paint.frame_analysis`` called once per
    ring with the labels of the KEPT clusters (the others set to noise)
    and the frames of the ring's localizations, in slab order (the order
    of ``an.labels``); its two statistics are kept on the cluster. The
    few-events rule reads the H1 ``n_events``; the edge rule
    ``_edge_suspects``. Without frames neither the burst rule nor the
    few-events rule can be evaluated (``build_rings`` warns once).
    """
    edge = _edge_suspects(an)
    bursty: Set[int] = set()
    stats: Dict[int, Tuple[float, float]] = {}
    if frame_ring is not None and n_frames_total is not None and clusters:
        kept = np.array([cl.label for cl in clusters], dtype=np.int64)
        labels = np.asarray(an.labels, dtype=np.int64)
        labels_kept = np.where(np.isin(labels, kept), labels, -1)
        fa = frame_analysis(labels_kept, np.asarray(frame_ring, dtype=np.int64), n_frames_total)
        warnings_.extend(f"ring {ring_index} frame analysis: {w}" for w in fa.warnings)
        for label, passed, mean_fraction, share in zip(
                fa.labels.tolist(), fa.passed.tolist(),
                fa.mean_frame_fraction.tolist(), fa.max_window_share.tolist()):
            stats[int(label)] = (float(mean_fraction), float(share))
            if not passed:
                bursty.add(int(label))
    for cl in clusters:
        marks: List[str] = []
        if cl.n_events is not None and cl.n_events < int(params.min_events_per_cluster):
            marks.append(SUSPECT_FEW_EVENTS)
        if cl.label in bursty:
            marks.append(SUSPECT_BURST)
        if cl.label in edge:
            marks.append(SUSPECT_EDGE)
        cl.suspect = tuple(marks)
        if cl.label in stats:
            cl.mean_frame_fraction, cl.max_window_share = stats[cl.label]


def _count_suspects(rings: Sequence[Ring]) -> Dict[str, int]:
    """``RingsResult.n_suspect``: every mark of ``SUSPECT_MARKS`` present
    (0 when none carries it), plus any other mark a cluster carries."""
    counts: Dict[str, int] = {mark: 0 for mark in SUSPECT_MARKS}
    for ring in rings:
        for cl in ring.clusters:
            for mark in cl.suspect:
                counts[mark] = counts.get(mark, 0) + 1
    return counts


def _empty_result(
    n: int,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    frame_i: Optional[NDArray[np.int64]],
    params: RingsParams,
    source_name: str,
    warnings_: List[str],
) -> RingsResult:
    """
    A ``RingsResult`` with no rings for fewer than ``MIN_LOCS_FOR_RINGS``
    localizations (an empty ROI), instead of an exception from the mixture.

    The frame is the identity, ``ms`` an analysis with no segments whose
    ``z_result`` is an unfitted mixture, and with frame numbers every
    localization is its own event (no radius is derived: there is nothing
    to link).
    """
    warnings_.append(
        f"{n} localizations: fewer than {MIN_LOCS_FOR_RINGS}, nothing to segment "
        "(the axial mixture needs at least two values); empty result"
    )
    origin = (np.array([x.mean(), y.mean(), z.mean()], dtype=np.float64)
              if n else np.zeros(3))
    frame = identity_axon_frame(None, origin, min_locs=int(params.axis_min_locs),
                                warnings=[warnings_[-1]])
    z_result = ZPeriodicityResult(
        n_components=0, means_nm=np.array([], dtype=np.float64),
        weights=np.array([], dtype=np.float64), sigmas_nm=np.array([], dtype=np.float64),
        delta_z_nm=np.array([], dtype=np.float64), main_peak_nm=float("nan"),
        bic_by_n={}, n_discarded_components=0, converged=False,
        warnings=[warnings_[-1]],
    )
    ms = MultiSegmentAnalysis(
        source_name=source_name, mode=params.segment_mode, segments=[], analyses=[],
        z_result=z_result, axon_center=None, pairs=[], valleys=None,
        warnings=[warnings_[-1]], guard_nm=float(params.guard_nm),
    )
    event_id: Optional[NDArray[np.int64]] = None
    n_events: Optional[int] = None
    if frame_i is not None:
        event_id = np.arange(n, dtype=np.int64)
        n_events = n
    else:
        warnings_.append("no frame numbers: events unavailable (n_events and "
                         "event_id are None)")
    ambiguous = None if params.posterior_min is None else np.zeros(n, dtype=bool)
    return RingsResult(
        frame=frame, rings=[], ms=ms, x_p=x.copy(), y_p=y.copy(), z_p=z.copy(),
        event_id=event_id, link_radius_nm=None, n_events=n_events,
        ambiguous=ambiguous, n_ambiguous=0, params=params,
        source_name=source_name, warnings=warnings_,
        n_suspect=_count_suspects([]),
    )


def build_rings(
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: NDArray[np.float64],
    *,
    frame: Optional[NDArray[np.int64]] = None,
    lp_lateral_nm: Optional[NDArray[np.float64]] = None,
    lpz_nm: Optional[NDArray[np.float64]] = None,
    params: RingsParams = RingsParams(),
    source_name: str = "",
    pixel_size_nm: Optional[float] = None,
    pixel_size_source: str = "unknown",
    n_frames: Optional[int] = None,
    roi: Optional[ROIShape] = None,
) -> RingsResult:
    """
    The rings of one axon in its own frame, with events, precisions and
    suspect marks.

    Parameters
    ----------
    x_nm, y_nm, z_nm : (M,) arrays
        Every localization of the ROI, in nm, BEFORE any axial cut (the
        mixture must see the whole axial distribution). Must be finite.
    frame : (M,) int array, optional
        Frame number per localization. Without it no events can be
        linked: ``event_id`` and every ``n_events`` are None.
    lp_lateral_nm : (M,) array, optional
        Lateral localization precision per localization, nm; sets the
        link radius (``params.link_radius_factor`` times its median).
        Without it the default radius is used, with a warning.
    lpz_nm : (M,) array, optional
        Axial localization precision per localization, nm. Without it
        every ``lpz_median_nm`` is NaN.
    params : RingsParams
        See the class; recorded in the result.
    source_name : str
        Name carried into every analysis and export.
    pixel_size_nm : float, optional
        Forwarded to ``analyze_axon`` for provenance only: the inputs are
        already in nm.
    pixel_size_source : str
        Where that pixel size came from, one of
        ``tools.mps_analysis.PIXEL_SIZE_SOURCES`` ("yaml", "hdf5", ...).
        Forwarded with the pixel size; the default "unknown" makes every
        ring analysis warn that the value was not read from the metadata,
        so a caller that knows the provenance should say so.
    n_frames : int, optional
        H2. Length of the acquisition in frames
        (``tools.mps_io.Localizations.n_frames`` for a real file), which
        the burst rule divides the movie by. When None, max(frame) + 1
        is used with a warning that it is a lower bound. Ignored without
        ``frame``.
    roi : ROIShape, optional
        H2. The ROI the localizations were selected with, forwarded to
        every ring analysis as its edge reference (H1 forwarded none, so
        the edge criterion fell back to the convex hull of the ring's
        localizations; that remains the behaviour when this is None).
        The ROI is applied to the coordinates the per-ring analysis
        runs on, i.e. the axon frame: with the tilt corrected a boundary
        drawn in the laboratory frame is displaced by about z' * sin(beta)
        laterally, a few nm at the tilts seen so far, and a warning
        records the tilt when both are in play.

    Returns
    -------
    RingsResult
        With fewer than ``MIN_LOCS_FOR_RINGS`` localizations an empty
        result (no rings, identity frame) with a warning, never an
        exception.

    Raises
    ------
    ValueError
        Mismatched lengths, non-finite coordinates, frame numbers that
        are booleans, non-finite or not whole numbers, or a
        non-positive ``n_frames``.

    Notes
    -----
    Steps, every one deterministic:

    1. With ``params.correct_tilt``, the axon frame: two starts (the
       specification's lab-frame per-ring planes, and the tilt at which
       the axial profile is sharpest), each refined by one plane per
       ring of a segmentation in the rotated frame until the rotation
       settles, the better resolved kept (``_fit_frame_staged``; the module
       docstring says why the per-ring fit alone is not enough and why
       a pooled plane is not a safe start); ``to_axon_frame`` gives x',
       y', z'. Otherwise the identity frame and the original coordinates.
    2. With ``params.posterior_min`` set, the mixture is refitted on z'
       and the localizations whose maximum component posterior is below
       it are left out of everything that follows (``posterior_ambiguous``);
       ``ambiguous`` and ``n_ambiguous`` record them. The primary path
       leaves ``ambiguous`` None.
    3. ``analyze_all_segments`` on the (remaining) axon-frame coordinates:
       the segmentation that DEFINES the rings, and the per-segment
       analysis of each (DBSCAN, curation, contour, areas, 1NN,
       occupancy; no randomization control).
    4. With ``frame``, ``link_localizations`` over ALL input localizations
       in the axon frame, so an event is one event wherever its
       localizations fell.
    5. One ``Ring`` per analysed segment, ``loc_index`` being the
       analysis' ``slab_index`` mapped back to the input; one ``Cluster``
       per kept cluster in the order of the centroid rows.
    6. H2, per ring: the centroid precision of every cluster by
       ``bootstrap_centroid_sigma`` over its events (rng from
       ``SeedSequence(params.random_seed)`` spawned per ring index, then
       per cluster position), then the suspect marks (``_mark_suspects``:
       few events, Picasso's burst rule via ``frame_analysis`` with the
       acquisition length, edge-touching kept by the runaway guard) and
       their counts in ``n_suspect``. Marks remove nothing.

    Without a ``roi`` the edge criterion of every ring analysis falls
    back to the convex hull of the analysed localizations (which rotates
    with them); on that path no cluster can carry "edge".
    """
    x = np.asarray(x_nm, dtype=np.float64).ravel()
    n = int(x.size)
    y = _as_float(y_nm, "y_nm", n)
    z = _as_float(z_nm, "z_nm", n)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise ValueError("x_nm, y_nm and z_nm must be finite")
    lp_lateral = None if lp_lateral_nm is None else _as_float(lp_lateral_nm, "lp_lateral_nm", n)
    lpz = None if lpz_nm is None else _as_float(lpz_nm, "lpz_nm", n)
    frame_i: Optional[NDArray[np.int64]] = None
    if frame is not None:
        frame_raw = np.asarray(frame).ravel()
        if frame_raw.size != n:
            raise ValueError(f"frame has {frame_raw.size} values for {n} localizations")
        if frame_raw.dtype == np.bool_:
            raise ValueError("frame numbers must be integers, not booleans")
        if not np.issubdtype(frame_raw.dtype, np.integer):
            # A float column is accepted only when every value is a whole
            # number: truncating 12.7 to 12 would silently change which
            # localizations are consecutive.
            if not np.isfinite(frame_raw).all():
                raise ValueError("frame numbers must be finite")
            if not np.array_equal(frame_raw, np.round(frame_raw)):
                raise ValueError("frame numbers must be integers (whole-number values)")
        frame_i = frame_raw.astype(np.int64)
    warnings_: List[str] = []

    # ---- 0. nothing to segment ------------------------------------------
    if n < MIN_LOCS_FOR_RINGS:
        return _empty_result(n, x, y, z, frame_i, params, source_name, warnings_)

    # ---- 1. the axon frame ---------------------------------------------
    axon_frame: Optional[AxonFrame] = None
    increments = np.array([], dtype=np.float64)
    axis_start = METHOD_IDENTITY
    candidate_beta = np.full(2, np.nan)
    candidate_depth = np.full(2, np.nan)
    if params.correct_tilt:
        staged = _fit_frame_staged(x, y, z, params)
        axon_frame = staged.frame
        increments = staged.increments_deg
        axis_start = staged.start
        candidate_beta = staged.candidate_beta_deg
        candidate_depth = staged.candidate_valley_depth
        warnings_.extend(f"axon frame: {w}" for w in axon_frame.warnings)
        x_p, y_p, z_p = to_axon_frame(axon_frame, x, y, z)
    else:
        # The identity frame is built after the segmentation below, so its
        # per-ring counts describe the rings actually used.
        x_p, y_p, z_p = x.copy(), y.copy(), z.copy()

    # ---- 2. ambiguous localizations (control path) ---------------------
    ambiguous: Optional[NDArray[np.bool_]] = None
    n_ambiguous = 0
    used = np.arange(n, dtype=np.intp)
    if params.posterior_min is not None:
        z_result_p = fit_z_periodicity(z_p)
        ambiguous = posterior_ambiguous(z_p, z_result_p, float(params.posterior_min))
        n_ambiguous = int(np.count_nonzero(ambiguous))
        used = np.flatnonzero(~ambiguous).astype(np.intp)
        fraction = n_ambiguous / n if n else 0.0
        warnings_.append(
            f"posterior_min = {params.posterior_min:g}: {n_ambiguous} of {n} "
            f"localizations ({100.0 * fraction:.2f} %) are ambiguous (no dominant "
            f"axial component reaches that posterior) and were left out of the "
            f"per-ring analysis"
        )

    # ---- 3. segmentation and per-segment analysis in the axon frame ----
    # The ROI (H2) is forwarded only when given, so the H1 call is the
    # same call: analyze_all_segments reads the key for its description.
    roi_kwargs: Dict[str, Any] = {} if roi is None else {"roi": roi}
    ms = analyze_all_segments(
        x_p[used], y_p[used], z_p[used],
        source_name=source_name,
        half_width_nm=params.half_width_nm,
        mode=params.segment_mode,
        guard_nm=params.guard_nm,
        run_randomization=False,
        eps_nm=params.eps_nm,
        min_samples=int(params.min_samples),
        pixel_size_nm=pixel_size_nm,
        pixel_size_source=pixel_size_source,
        random_seed=int(params.random_seed),
        **roi_kwargs,
    )
    warnings_.extend(ms.warnings)
    if axon_frame is None:
        ring_index0 = ring_index_from_segments(z_p, ms.segments)
        in_ring = ring_index0 >= 0
        origin = (np.array([x[in_ring].mean(), y[in_ring].mean(), z[in_ring].mean()])
                  if in_ring.any() else np.array([x.mean(), y.mean(), z.mean()]))
        axon_frame = identity_axon_frame(ring_index0, origin,
                                         min_locs=int(params.axis_min_locs))

    # ---- 4. emission events over every input localization -------------
    event_id: Optional[NDArray[np.int64]] = None
    link_radius: Optional[float] = None
    lost_fraction: Optional[float] = None
    n_events: Optional[int] = None
    if frame_i is not None:
        event_id, link_radius, lost_fraction = _link_events(
            frame_i, x_p, y_p, lp_lateral, params, warnings_)
        n_events = _count_events(event_id, np.arange(n, dtype=np.intp))
    else:
        warnings_.append("no frame numbers: events unavailable (n_events and "
                         "event_id are None)")
    if lpz is None:
        warnings_.append("no axial precision (lpz) given: lpz_median_nm is NaN")

    # ---- H2 preparation: acquisition length, bootstrap seeds, ROI note --
    n_frames_used: Optional[int] = None
    if frame_i is not None:
        n_frames_used = _acquisition_length(frame_i, n_frames, warnings_)
        if int(params.n_bootstrap) < MIN_BOOTSTRAP_DRAWS:
            warnings_.append(
                f"n_bootstrap = {params.n_bootstrap} (fewer than {MIN_BOOTSTRAP_DRAWS}): "
                "the centroid precision was not bootstrapped (sigma_centroid_nm is NaN, "
                "n_bootstrap 0 in every cluster)"
            )
    else:
        warnings_.append(
            "no frame numbers: the centroid precision cannot be bootstrapped over "
            "emission events (sigma_centroid_nm is NaN in every cluster) and the "
            "burst mark cannot be computed (no cluster carries 'burst'; 'few_events' "
            "needs n_events and is not set either)"
        )
    if roi is not None and axon_frame is not None and not axon_frame.is_identity:
        warnings_.append(
            f"roi given with the axis tilted by {axon_frame.beta_deg:.2f} deg: the edge "
            "criterion and the 'edge' mark apply the ROI to the axon-frame coordinates, "
            "where a boundary drawn in the laboratory frame is displaced by about "
            "z' * sin(beta) laterally"
        )
    # The bootstrap seeds: ring k draws from the k-th child of the root
    # sequence, SeedSequence(seed).spawn(n)[k], which is
    # SeedSequence(seed, spawn_key=(k,)) whatever n is; written that way
    # so that a ring's seed depends on nothing but the seed and its index.
    root_seed = int(params.random_seed)

    # ---- 5. rings and clusters -----------------------------------------
    valleys = ms.valleys
    n_boundaries = 0 if valleys is None else int(valleys.n_boundaries)
    rings: List[Ring] = []
    for seg, an in zip(ms.segments, ms.analyses):
        if an is None:
            continue
        ring_warnings: List[str] = list(an.warnings)
        loc_index = np.asarray(used[np.asarray(an.slab_index, dtype=np.intp)],
                               dtype=np.intp)
        c = int(seg.component_index)
        depth_lo: Optional[float] = None
        depth_hi: Optional[float] = None
        if valleys is not None and c >= 0:
            if c > 0 and c - 1 < n_boundaries:
                depth_lo = float(valleys.relative_depth[c - 1])
            if c < n_boundaries:
                depth_hi = float(valleys.relative_depth[c])
        contour: Optional[NDArray[np.float64]] = None
        length = float("nan")
        centroid: Optional[NDArray[np.float64]] = None
        if an.perimeter is not None:
            contour = np.asarray(an.perimeter.contour, dtype=np.float64)
            length = float(an.perimeter.perimeter_nm)
            if an.perimeter.centre is not None:
                centroid = np.array([an.perimeter.centre.x_nm,
                                     an.perimeter.centre.y_nm], dtype=np.float64)
            else:
                ring_warnings.append(
                    f"ring {seg.index}: the contour has no area centroid "
                    "(it crosses itself or encloses no area)")
        clusters = _build_clusters(seg.index, an, loc_index, x_p, y_p, z_p,
                                   lpz, event_id, ring_warnings)
        # ---- 6. H2: centroid precision and suspect marks ----------------
        ring_seed = np.random.SeedSequence(root_seed, spawn_key=(int(seg.index),))
        _fill_cluster_precision(clusters, x_p, y_p, event_id, params, ring_seed)
        _mark_suspects(clusters, an, int(seg.index),
                       None if frame_i is None else frame_i[loc_index],
                       n_frames_used, params, ring_warnings)
        rings.append(Ring(
            index=int(seg.index),
            component_index=c,
            z_lo_nm=float(seg.zmin_nm),
            z_hi_nm=float(seg.zmax_nm),
            guard_nm=float(params.guard_nm),
            centre_z_nm=float(seg.center_nm),
            sigma_z_nm=float(seg.sigma_nm),
            lpz_median_nm=_nanmedian_or_nan(lpz, loc_index),
            relative_depth_lo=depth_lo,
            relative_depth_hi=depth_hi,
            n_locs=int(loc_index.size),
            n_events=_count_events(event_id, loc_index),
            loc_index=loc_index,
            event_id=None if event_id is None else np.asarray(event_id[loc_index], dtype=np.int64),
            analysis=an,
            clusters=clusters,
            contour_nm=contour,
            length_nm=length,
            centroid_nm=centroid,
            warnings=ring_warnings,
        ))

    return RingsResult(
        frame=axon_frame,
        rings=rings,
        ms=ms,
        x_p=x_p, y_p=y_p, z_p=z_p,
        event_id=event_id,
        link_radius_nm=link_radius,
        n_events=n_events,
        ambiguous=ambiguous,
        n_ambiguous=n_ambiguous,
        params=params,
        source_name=source_name,
        warnings=warnings_,
        axis_stages_beta_deg=increments,
        axis_start=axis_start,
        axis_candidate_beta_deg=candidate_beta,
        axis_candidate_valley_depth=candidate_depth,
        link_lost_step_fraction=lost_fraction,
        n_suspect=_count_suspects(rings),
        n_frames=n_frames_used,
    )


# ============================================================================
# H2: tau_0 and its components (D-03)
# ============================================================================

@dataclass
class TauComponents:
    """
    tau_0 of D-03 and everything it was made of, from ``tau_components``.

    ``tau0_nm`` is the median over cross pairs of consecutive rings of
    the pair tolerance ``rho_i + rho_j + 2 * sigma_pair_ij``;
    ``rho_pair_median_nm`` and ``sigma_pair_median_nm`` are the medians
    of the two terms over the same pairs (their sum is NOT tau_0: the
    median of a sum is not the sum of medians; the pair tolerance is what
    H3 applies per pair, so its median is the scalar). ``period_nm``,
    ``delta_beta_deg`` and ``lateral_shift_nm`` are per-axon quantities:
    the axon's own with one axon, the median over the axons' finite
    values with several (a warning then lists them per axon).
    ``n_clusters_excluded`` splits into ``n_excluded_suspect`` (left out
    for a mark, with ``exclude_suspect``) and ``n_excluded_missing``
    (NaN rho or NaN sigma_m). The quantiles are at
    ``quantiles_percent`` (10/25/50/75/90 %).
    """

    tau0_nm: float
    rho_pair_median_nm: float
    sigma_m_median_nm: float
    sigma_pair_median_nm: float
    period_nm: float
    delta_beta_deg: float
    lateral_shift_nm: float
    sigma_d_nm: float
    n_axons: int
    n_clusters_used: int
    n_clusters_excluded: int
    n_pairs: int
    rho_quantiles_nm: NDArray[np.float64]
    sigma_m_quantiles_nm: NDArray[np.float64]
    warnings: List[str] = field(default_factory=list)
    n_excluded_suspect: int = 0
    n_excluded_missing: int = 0
    quantiles_percent: NDArray[np.float64] = field(
        default_factory=lambda: np.asarray(TAU_QUANTILES_PERCENT, dtype=np.float64))


def _cluster_sigma_m(cl: Cluster) -> float:
    """``sigma_centroid_nm`` as a float; None (never bootstrapped) is NaN."""
    return float("nan") if cl.sigma_centroid_nm is None else float(cl.sigma_centroid_nm)


def _usable_for_tau(cl: Cluster, exclude_suspect: bool) -> Tuple[bool, str]:
    """Whether a cluster enters tau_0, and if not, why ("suspect" or "missing")."""
    if not (np.isfinite(float(cl.rho_nm)) and np.isfinite(_cluster_sigma_m(cl))):
        return False, "missing"
    if exclude_suspect and cl.suspect:
        return False, "suspect"
    return True, ""


def _per_axon_summary(values: Sequence[float]) -> float:
    """One axon: its value; several: the median of the finite ones (NaN
    when none is finite); none: NaN."""
    if len(values) == 1:
        return float(values[0])
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def tau_components(
    results: Sequence[RingsResult],
    *,
    sigma_d_nm: float = 0.0,
    exclude_suspect: bool = True,
) -> TauComponents:
    """
    tau_0 (D-03) and its components from one or more ``RingsResult``.

    Definition (this resolves an ambiguity of D-03's wording): for every
    pair of consecutive rings (k, k + 1) of every axon in ``results``
    (consecutive by ``Ring.index``; a gap left by a segment that failed
    to analyse breaks the pair, with a warning), and every cluster i of
    ring k and j of ring k + 1 -- the full cross product, not a matching
    -- the pair tolerance is

        t_ij = rho_i + rho_j + 2 * sqrt(sigma_m_i^2 + sigma_m_j^2
                                        + shift_A^2 + sigma_d^2)

    with ``shift_A = period_A * tan(delta_beta_A)`` of that axon (0 when
    ``delta_beta`` or the period is NaN, with a warning: nothing is then
    known about the rings' disagreement, and adding an unknown would be
    a guess), ``period_A`` the ring spacing of that axon and ``sigma_d``
    the assumed z-dependent lateral distortion (D-12: 0 for the exploratory dataset).
    ``period_A`` is the median over consecutive LISTED rings of their
    centre spacing divided by their ``Ring.index`` step, so a segment
    that failed to analyse (rings 0 and 2 listed, 1 missing) counts as
    two periods and not as one doubled spacing: the shift of the pairs
    that do exist would otherwise be inflated by the gap (measured on
    the harness axon: 286 nm instead of 191 nm for rings 0, 1, 3, with
    the tolerance of the 552 pairs of rings 0-1 rising by 0.16 nm at
    delta_beta = 0.24 deg and by 5.7 nm at 2 deg). Rings with a
    repeated index give a NaN period, with a warning. ``tau0_nm`` is the
    median of every t_ij. Why the median of the pair tolerance and not
    the sum of medians: H3 applies the tolerance per pair, so the
    typical PAIR tolerance is the right scalar; the medians of the two
    terms are reported alongside.

    Clusters with NaN rho or NaN sigma_m, and (with ``exclude_suspect``)
    clusters with any suspect mark, are left out of the pairs, of
    ``sigma_m_median_nm`` and of the quantiles, and are counted in
    ``n_clusters_excluded`` (split by reason); the count runs over every
    ring, including one without a neighbour. With no usable pair every
    median is NaN and a warning says so.

    Parameters
    ----------
    results : sequence of RingsResult
        The axons whose pairs are pooled (D-05: the exploratory axon
        alone for the pre-registered tau_0).
    sigma_d_nm : float
        Assumed lateral distortion between rings, nm (>= 0).
    exclude_suspect : bool
        Leave out clusters carrying any mark of ``SUSPECT_MARKS``.

    Returns
    -------
    TauComponents
        Deterministic: no random numbers are drawn here.

    Raises
    ------
    ValueError
        ``sigma_d_nm`` negative or not finite.
    """
    sigma_d = float(sigma_d_nm)
    if not np.isfinite(sigma_d) or sigma_d < 0.0:
        raise ValueError(f"sigma_d_nm must be a finite non-negative length, got {sigma_d_nm!r}")
    warnings_: List[str] = []
    t_all: List[NDArray[np.float64]] = []
    rho_pair_all: List[NDArray[np.float64]] = []
    sigma_pair_all: List[NDArray[np.float64]] = []
    rho_used: List[float] = []
    sigma_used: List[float] = []
    n_used = 0
    n_excluded_suspect = 0
    n_excluded_missing = 0
    periods: List[float] = []
    deltas: List[float] = []
    shifts: List[float] = []

    for a, res in enumerate(results):
        label = res.source_name or f"axon {a}"
        rings = list(res.rings)
        centres = np.array([float(r.centre_z_nm) for r in rings], dtype=np.float64)
        index_steps = np.diff(np.array([int(r.index) for r in rings], dtype=np.float64))
        if centres.size < 2:
            period = float("nan")
        elif np.any(index_steps == 0.0):
            period = float("nan")
            warnings_.append(
                f"{label}: rings with a repeated index {[int(r.index) for r in rings]}: "
                "period undefined"
            )
        else:
            # Spacing per index step, so a missing segment does not
            # double the spacing of the rings around it (docstring).
            period = float(np.median(np.diff(centres) / index_steps))
        delta = float(res.frame.delta_beta_deg)
        shift_field = lateral_shift_per_period_nm(res.frame, period)
        periods.append(period)
        deltas.append(delta)
        shifts.append(float(shift_field))
        if len(rings) < 2:
            warnings_.append(
                f"{label}: {len(rings)} ring(s): no pair of consecutive rings, "
                "period and pair tolerances undefined"
            )
        if np.isfinite(shift_field):
            shift = float(shift_field)
        else:
            shift = 0.0
            if len(rings) >= 2:
                warnings_.append(
                    f"{label}: lateral shift undefined (delta_beta {delta:g} deg, "
                    f"period {period:g} nm): 0 used in the pair tolerances"
                )

        usable: List[Tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for ring in rings:
            rho_ring: List[float] = []
            sigma_ring: List[float] = []
            for cl in ring.clusters:
                ok, reason = _usable_for_tau(cl, exclude_suspect)
                if not ok:
                    if reason == "suspect":
                        n_excluded_suspect += 1
                    else:
                        n_excluded_missing += 1
                    continue
                n_used += 1
                rho_ring.append(float(cl.rho_nm))
                sigma_ring.append(_cluster_sigma_m(cl))
                rho_used.append(rho_ring[-1])
                sigma_used.append(sigma_ring[-1])
            usable.append((np.asarray(rho_ring, dtype=np.float64),
                           np.asarray(sigma_ring, dtype=np.float64)))

        for k in range(len(rings) - 1):
            if int(rings[k + 1].index) != int(rings[k].index) + 1:
                warnings_.append(
                    f"{label}: rings {rings[k].index} and {rings[k + 1].index} are not "
                    "consecutive (a segment between them was not analysed): no pairs"
                )
                continue
            rho_a, sig_a = usable[k]
            rho_b, sig_b = usable[k + 1]
            if rho_a.size == 0 or rho_b.size == 0:
                continue
            rho_pair = rho_a[:, None] + rho_b[None, :]
            sigma_pair = np.sqrt(sig_a[:, None] ** 2 + sig_b[None, :] ** 2
                                 + shift * shift + sigma_d * sigma_d)
            t_all.append(np.asarray(rho_pair + 2.0 * sigma_pair, dtype=np.float64).ravel())
            rho_pair_all.append(np.asarray(rho_pair, dtype=np.float64).ravel())
            sigma_pair_all.append(np.asarray(sigma_pair, dtype=np.float64).ravel())

    if not results:
        warnings_.append("no results given: nothing to compute")
    t = np.concatenate(t_all) if t_all else np.empty(0, dtype=np.float64)
    if t.size == 0:
        tau0 = rho_pair_median = sigma_pair_median = float("nan")
        if results:
            warnings_.append(
                "no usable pair of clusters in consecutive rings: tau0_nm and the "
                "pair medians are NaN"
            )
    else:
        tau0 = float(np.median(t))
        rho_pair_median = float(np.median(np.concatenate(rho_pair_all)))
        sigma_pair_median = float(np.median(np.concatenate(sigma_pair_all)))
    q = np.asarray(TAU_QUANTILES_PERCENT, dtype=np.float64)
    if sigma_used:
        sigma_m_median = float(np.median(sigma_used))
        rho_q = np.asarray(np.percentile(rho_used, q), dtype=np.float64)
        sigma_q = np.asarray(np.percentile(sigma_used, q), dtype=np.float64)
    else:
        sigma_m_median = float("nan")
        rho_q = np.full(q.size, np.nan)
        sigma_q = np.full(q.size, np.nan)
    if len(results) > 1:
        warnings_.append(
            f"{len(results)} axons pooled: period_nm, delta_beta_deg and "
            "lateral_shift_nm are the medians of the finite per-axon values; per "
            "axon (period nm, delta_beta deg, shift nm): "
            + "; ".join(f"{(r.source_name or f'axon {i}')}: ({p:.1f}, {d:.3f}, {s:.2f})"
                        for i, (r, p, d, s) in enumerate(zip(results, periods, deltas, shifts)))
        )
    return TauComponents(
        tau0_nm=tau0,
        rho_pair_median_nm=rho_pair_median,
        sigma_m_median_nm=sigma_m_median,
        sigma_pair_median_nm=sigma_pair_median,
        period_nm=_per_axon_summary(periods),
        delta_beta_deg=_per_axon_summary(deltas),
        lateral_shift_nm=_per_axon_summary(shifts),
        sigma_d_nm=sigma_d,
        n_axons=len(results),
        n_clusters_used=n_used,
        n_clusters_excluded=n_excluded_suspect + n_excluded_missing,
        n_pairs=int(t.size),
        rho_quantiles_nm=rho_q,
        sigma_m_quantiles_nm=sigma_q,
        warnings=warnings_,
        n_excluded_suspect=n_excluded_suspect,
        n_excluded_missing=n_excluded_missing,
        quantiles_percent=q,
    )


# ============================================================================
# H2: the pre-registered parameters file
# ============================================================================

@dataclass
class ColumnsParams:
    """
    The parameters of the column analysis, frozen before the check set is
    looked at (03_plan S3.6, D-05): tau_0 and the tolerances H3 scans,
    the null (D-11), and the ``RingsParams`` fields the rings must be
    built with so that ``rings_params_from`` gives back exactly what
    produced tau_0. ``provenance`` records where tau_0 came from
    (dataset role, source files, date, the tau components, code commit)
    and is free-form beyond ``PROVENANCE_REQUIRED_KEYS``, which
    ``write_columns_params`` requires; everything else is typed, checked
    for type and for range (``_check_columns_ranges``) on write and on
    load.
    """

    tau0_nm: float
    tau_grid_nm: List[float]                # 20, 30, ..., 200 (DEFAULT_TAU_GRID_NM)
    tau_sensitivity_nm: List[float]         # [30.0, tau0_nm, 100.0]
    n_null: int = 1999                      # B of the null (D-11)
    random_seed: int = 0
    min_shift_fraction: float = 0.5         # null shifts exclude |U| < fraction * mean cluster spacing (D-11)
    guard_nm: float = 0.0                   # provisional until H5 fixes it by simulation
    posterior_min: Optional[float] = None
    segment_mode: str = "valley"
    eps_nm: float = 25.0
    min_samples: int = 10
    link_radius_factor: float = 5.0
    link_max_dark_time: int = 1
    n_bootstrap: int = 200
    min_events_per_cluster: int = 3
    sigma_d_nm: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)


# How each field is read back: the YAML value is coerced to the declared
# type and anything else is an error, so a typo in a frozen file (an int
# where a list belongs, "yes" for a number) cannot pass silently.
_COLUMNS_FLOAT_FIELDS = ("tau0_nm", "min_shift_fraction", "guard_nm", "eps_nm",
                         "link_radius_factor", "sigma_d_nm")
_COLUMNS_INT_FIELDS = ("n_null", "random_seed", "min_samples", "link_max_dark_time",
                       "n_bootstrap", "min_events_per_cluster")
_COLUMNS_FLOAT_LIST_FIELDS = ("tau_grid_nm", "tau_sensitivity_nm")
_COLUMNS_OPTIONAL_FLOAT_FIELDS = ("posterior_min",)
_COLUMNS_STR_FIELDS = ("segment_mode",)
_COLUMNS_DICT_FIELDS = ("provenance",)
# The fields ColumnsParams shares with RingsParams, copied by
# rings_params_from and columns_params_from_rings.
_COLUMNS_RINGS_SHARED = ("segment_mode", "guard_nm", "eps_nm", "min_samples",
                         "link_radius_factor", "link_max_dark_time", "posterior_min",
                         "random_seed", "n_bootstrap", "min_events_per_cluster")


def _param_float(name: str, value: Any) -> float:
    """``value`` as a finite float, or ValueError naming the field. bool is
    rejected although it is an int in Python, because YAML ``yes``/``no``
    would otherwise load as 1.0/0.0 without a word; NaN and inf are
    rejected because no parameter of the file means anything with them."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name}: expected a number, got {value!r}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name}: expected a finite number, got {value!r}")
    return out


def _param_int(name: str, value: Any) -> int:
    """``value`` as an int, or ValueError naming the field. bool is
    rejected (YAML ``yes``/``no``); an integral float (``10.0``, which a
    hand edit or a NumPy value can leave behind) is accepted, ``10.5`` is
    not, since a count truncated silently would be a different run."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name}: expected an integer, got {value!r}")
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(f"{name}: expected an integer, got {value!r}")
    return int(value)


def _param_float_list(name: str, value: Any) -> List[float]:
    """``value`` as a list of finite floats, or ValueError naming the field
    and the offending position. A string, bytes or mapping is rejected
    before the element check because each is iterable and would otherwise
    fail one character or one key at a time with a misleading message."""
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name}: expected a list of numbers, got {value!r}")
    return [_param_float(f"{name}[{i}]", v) for i, v in enumerate(list(value))]


def _coerce_columns_value(name: str, value: Any) -> Any:
    """The YAML value of field ``name`` coerced to the field's declared
    type by the tables above (float, int, list of float, optional float,
    str, dict), or ValueError. A field with no reader is an error too, so
    a field added to ``ColumnsParams`` without a table entry cannot load
    as whatever YAML made of it."""
    if name in _COLUMNS_FLOAT_FIELDS:
        return _param_float(name, value)
    if name in _COLUMNS_INT_FIELDS:
        return _param_int(name, value)
    if name in _COLUMNS_FLOAT_LIST_FIELDS:
        return _param_float_list(name, value)
    if name in _COLUMNS_OPTIONAL_FLOAT_FIELDS:
        return None if value is None else _param_float(name, value)
    if name in _COLUMNS_STR_FIELDS:
        if not isinstance(value, str):
            raise ValueError(f"{name}: expected a string, got {value!r}")
        return value
    if name in _COLUMNS_DICT_FIELDS:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(f"{name}: expected a mapping, got {value!r}")
        return dict(value)
    raise ValueError(f"{name}: no reader for this field")   # a field added without a reader


def _check_columns_ranges(params: ColumnsParams, where: str) -> None:
    """
    The values a typed field may take, or ValueError naming the field.

    The type check alone lets through numbers that cannot mean anything
    and would fail late, in the middle of a run, or not at all: a
    negative tau_0 or eps, zero DBSCAN min_samples, a null of zero
    shifts, a posterior threshold above 1 (no localization is ever
    dropped, silently), a misspelt segment mode (raised only when
    ``find_axial_segments`` runs). Applied on write, on load and by
    ``columns_params_from_rings``, so neither a frozen file nor the
    object that writes it can hold them. Bounds are the loosest that
    make sense: ``n_bootstrap`` 0 means "off" (``MIN_BOOTSTRAP_DRAWS``),
    ``guard_nm``, ``sigma_d_nm`` and ``min_shift_fraction`` 0 are the
    pre-registered values, ``posterior_min`` 0 and 1 are the ends of its range.
    """
    positive = (("tau0_nm", params.tau0_nm), ("eps_nm", params.eps_nm),
                ("link_radius_factor", params.link_radius_factor))
    for name, value in positive:
        if not float(value) > 0.0:
            raise ValueError(f"{where}: {name} must be > 0, got {value!r}")
    non_negative = (("min_shift_fraction", params.min_shift_fraction),
                    ("guard_nm", params.guard_nm), ("sigma_d_nm", params.sigma_d_nm))
    for name, value in non_negative:
        if not float(value) >= 0.0:
            raise ValueError(f"{where}: {name} must be >= 0, got {value!r}")
    at_least_one = (("n_null", params.n_null), ("min_samples", params.min_samples),
                    ("min_events_per_cluster", params.min_events_per_cluster))
    for name, count in at_least_one:
        if int(count) < 1:
            raise ValueError(f"{where}: {name} must be >= 1, got {count!r}")
    for name, count in (("n_bootstrap", params.n_bootstrap),
                        ("link_max_dark_time", params.link_max_dark_time)):
        if int(count) < 0:
            raise ValueError(f"{where}: {name} must be >= 0, got {count!r}")
    for name, values in (("tau_grid_nm", params.tau_grid_nm),
                         ("tau_sensitivity_nm", params.tau_sensitivity_nm)):
        if len(values) == 0:
            raise ValueError(f"{where}: {name} must not be empty")
        bad = [v for v in values if not float(v) > 0.0]
        if bad:
            raise ValueError(f"{where}: {name} must hold tolerances > 0, got {bad!r}")
    if params.posterior_min is not None and not 0.0 <= float(params.posterior_min) <= 1.0:
        raise ValueError(
            f"{where}: posterior_min must be in [0, 1] or null, got {params.posterior_min!r}")
    if params.segment_mode not in SEGMENT_MODES:
        raise ValueError(
            f"{where}: segment_mode must be one of {list(SEGMENT_MODES)}, got {params.segment_mode!r}")


def _check_provenance(provenance: Dict[str, Any], where: str) -> None:
    """
    ``PROVENANCE_REQUIRED_KEYS`` present and non-empty, ``dataset_role``
    one of ``DATASET_ROLES``, or ValueError. The dict stays free-form
    otherwise (source files, the tau components, versions may be added),
    but a file that does not say which data, which day and which code
    produced its tau_0 is not a pre-registration (03_plan S3.1).
    """
    missing = [k for k in PROVENANCE_REQUIRED_KEYS
               if k not in provenance or provenance[k] is None or provenance[k] == ""]
    if missing:
        raise ValueError(
            f"{where}: provenance must give {list(PROVENANCE_REQUIRED_KEYS)}; missing {missing}")
    role = provenance["dataset_role"]
    if role not in DATASET_ROLES:
        raise ValueError(
            f"{where}: provenance dataset_role must be one of {list(DATASET_ROLES)}, got {role!r}")


def _yaml_module() -> Any:
    """PyYAML, imported when a parameters file is written or read and not
    at module level: ``build_rings`` (H1) must not need it, and the two
    other modules that read YAML (``mps_metadata``, ``mps_registration``)
    import it inside the function that needs it for the same reason."""
    import yaml  # type: ignore[import-untyped]
    return yaml


def _yaml_load_strict(text: str, where: str) -> Any:
    """
    ``yaml.safe_load`` that raises ValueError on a repeated key.

    PyYAML keeps the LAST value of a duplicated mapping key without a
    word, so a pre-registered file edited by hand into two ``n_null``
    lines would load as the second one; the loader's promise is that
    an editing error in the frozen file does not pass silently. The
    check walks the node tree ``yaml.compose`` gives (every mapping
    level, the provenance too) before anything is built, comparing
    scalar keys by tag and text (``1`` and ``"1"`` are different keys to
    YAML as well), and names the key and its line.
    """
    yaml = _yaml_module()
    root = yaml.compose(text)
    stack: List[Any] = [] if root is None else [root]
    while stack:
        node = stack.pop()
        if isinstance(node, yaml.MappingNode):
            seen: Set[Tuple[str, str]] = set()
            for key_node, value_node in node.value:
                if isinstance(key_node, yaml.ScalarNode):
                    key = (str(key_node.tag), str(key_node.value))
                    if key in seen:
                        raise ValueError(
                            f"{where}: key {key_node.value!r} appears twice (second at line "
                            f"{key_node.start_mark.line + 1}); a repeated key in the "
                            "pre-registered file is an editing error")
                    seen.add(key)
                stack.append(key_node)
                stack.append(value_node)
        elif isinstance(node, yaml.SequenceNode):
            stack.extend(node.value)
    return yaml.safe_load(text)


def _plain(value: Any) -> Any:
    """A value made of nothing but YAML-native types (dict, list, str,
    int, float, bool, None): NumPy scalars and arrays are converted,
    tuples become lists, so ``safe_dump`` writes no Python tags."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"cannot write a value of type {type(value).__name__} to the parameters file")


def write_columns_params(path: str, params: ColumnsParams) -> None:
    """
    Write ``params`` as YAML: the comment block ``COLUMNS_PARAMS_HEADER``
    first, then the keys in the dataclass order, floats as floats
    (``guard_nm: 0.0``, never ``0``), None as ``null``, lists of scalars
    in flow style (``[20.0, 30.0, ...]``) so the file reads as a table.

    Every value passes through ``_plain`` first: a NumPy float in the
    provenance (a tau component, say) would otherwise be written with a
    Python-specific tag that ``safe_load`` refuses.

    Raises ``ValueError`` when a value is outside its range
    (``_check_columns_ranges``) or the provenance lacks
    ``PROVENANCE_REQUIRED_KEYS`` or names a role outside
    ``DATASET_ROLES`` (``_check_provenance``): what is written is what
    gets frozen, so the file is refused rather than the run.
    """
    yaml = _yaml_module()
    _check_columns_ranges(params, "write_columns_params")
    _check_provenance(params.provenance, "write_columns_params")
    data: Dict[str, Any] = {}
    for f in dataclasses.fields(params):
        value = getattr(params, f.name)
        if f.name in _COLUMNS_FLOAT_FIELDS:
            value = _param_float(f.name, value)
        elif f.name in _COLUMNS_FLOAT_LIST_FIELDS:
            value = _param_float_list(f.name, value)
        elif f.name in _COLUMNS_OPTIONAL_FLOAT_FIELDS:
            value = None if value is None else _param_float(f.name, value)
        elif f.name in _COLUMNS_INT_FIELDS:
            value = _param_int(f.name, value)
        data[f.name] = _plain(value)
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=None,
                          allow_unicode=True, width=100)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(COLUMNS_PARAMS_HEADER) + "\n" + body)


def load_columns_params(path: str) -> ColumnsParams:
    """
    Read a file written by ``write_columns_params`` back, exactly.

    The header comment block is skipped (a '#' line is a comment to the
    YAML reader). Strict on purpose, because the file is pre-registered:
    an unknown key raises ``ValueError`` naming it (a typo must not pass
    silently as "left at the default"), a missing ``tau0_nm``,
    ``tau_grid_nm`` or ``tau_sensitivity_nm`` raises (they have no
    default), a repeated key raises (PyYAML would keep the last value;
    ``_yaml_load_strict``), every value is coerced to the field's
    declared type (ints to int, numbers to float, so ``guard_nm: 0``
    loads as 0.0) or raises, and every value must be in its range
    (``_check_columns_ranges``). Fields with defaults may be absent. The
    provenance is NOT required here (a minimal hand-written file loads
    with an empty one): the writer is where a frozen file is made, and
    it refuses to write one without provenance.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    raw = _yaml_load_strict(text, path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of parameters, got {type(raw).__name__}")
    fields = dataclasses.fields(ColumnsParams)
    names = [f.name for f in fields]
    unknown = [str(k) for k in raw if k not in names]
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown}; the pre-registered file accepts only {names}")
    missing = [f.name for f in fields
               if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
               and f.name not in raw]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")
    kwargs = {name: _coerce_columns_value(name, raw[name]) for name in names if name in raw}
    params = ColumnsParams(**kwargs)
    _check_columns_ranges(params, path)
    return params


def rings_params_from(params: ColumnsParams) -> RingsParams:
    """
    The ``RingsParams`` that H3+ must build the rings with so that both
    files agree: every field the two classes share is copied from
    ``params``; the fields only ``RingsParams`` has (``half_width_nm``,
    ``correct_tilt``, ``axis_min_locs``, ``link_radius_default_nm``) stay
    at their defaults.
    """
    return RingsParams(**{name: getattr(params, name) for name in _COLUMNS_RINGS_SHARED})


def columns_params_from_rings(
    tau0_nm: float,
    rings_params: Optional[RingsParams] = None,
    *,
    sigma_d_nm: float = 0.0,
    tau_grid_nm: Sequence[float] = DEFAULT_TAU_GRID_NM,
    n_null: int = 1999,
    min_shift_fraction: float = 0.5,
    provenance: Optional[Dict[str, Any]] = None,
) -> ColumnsParams:
    """
    The pre-registered parameters for a measured tau_0 (an addition
    beyond the H2 specification's API, for the report script): the grid
    ``DEFAULT_TAU_GRID_NM``, the sensitivity triple [30, tau_0, 100] nm
    and the shared fields of the ``RingsParams`` the rings were built
    with (``rings_params_from`` is its inverse on those fields; None
    means ``RingsParams()``, built here on every call so that no
    instance is shared between calls). ``provenance`` is stored as
    given (dataset role, source files, the date the CALLER passes in --
    nothing here reads the clock -- the tau components, the code
    commit); ``write_columns_params`` is where it is required. The
    ranges of ``_check_columns_ranges`` are enforced, so a negative or
    NaN tau_0 cannot become a parameters object.
    """
    tau0 = _param_float("tau0_nm", tau0_nm)
    rp = RingsParams() if rings_params is None else rings_params
    shared = {name: getattr(rp, name) for name in _COLUMNS_RINGS_SHARED}
    params = ColumnsParams(
        tau0_nm=tau0,
        tau_grid_nm=[float(v) for v in tau_grid_nm],
        tau_sensitivity_nm=[30.0, tau0, 100.0],
        n_null=int(n_null),
        min_shift_fraction=float(min_shift_fraction),
        sigma_d_nm=_param_float("sigma_d_nm", sigma_d_nm),
        provenance={} if provenance is None else dict(provenance),
        **shared,
    )
    _check_columns_ranges(params, "columns_params_from_rings")
    return params


# ============================================================================
# H5-D: the rings without some of their clusters (the lumen cleaning)
# ============================================================================

# Largest distance between a ring analysis' centroid row and the exact
# centroid of the Ring.clusters entry it stands for: the rows are the exact
# means rounded to 0.01 nm (``tools.cluster_quality.good_cluster_centroids``),
# so a larger gap means the analysis and Ring.clusters no longer describe the
# same clusters and nothing may be dropped by position.
CLEAN_ROW_TOL_NM = 0.01
# What ``clean_rings`` records in each rebuilt ring analysis as the discard
# margin (``tools.mps_analysis.without_clusters`` requires one): the margin
# of the widefield rule, ``tools.mps_axoplasm.DEFAULT_MARGIN_NM`` (the
# research port, a private feasibility script, recorded the same).
CLEAN_RECORDED_MARGIN_NM = 250.0


def _whole_index(value: Any, what: str) -> int:
    """``value`` as an int index; booleans, non-integral and non-numeric values are refused."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"clean_rings: the {what} must be an integer, got the boolean {value!r}")
    try:
        return int(operator.index(value))
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        return int(value)
    raise ValueError(f"clean_rings: the {what} must be an integer, got {value!r}")


def _clean_flag_map(res: RingsResult, flagged: Iterable[Tuple[int, int]]) -> Dict[int, Set[int]]:
    """{ring index: positions to drop} from the (ring index, position) pairs, validated against ``res``."""
    by_index = {int(r.index): r for r in res.rings}
    out: Dict[int, Set[int]] = {}
    for pair in flagged:
        try:
            k_raw, i_raw = pair
        except (TypeError, ValueError) as exc:
            raise ValueError(f"clean_rings: every flagged entry must be a (ring index, position) pair, got {pair!r}") from exc
        k = _whole_index(k_raw, "ring index")
        i = _whole_index(i_raw, "cluster position")
        if k not in by_index:
            raise ValueError(f"clean_rings: ring {k} is not a ring of {res.source_name!r} (rings {sorted(by_index)})")
        n = len(by_index[k].clusters)
        if not 0 <= i < n:
            raise ValueError(f"clean_rings: cluster position {i} is outside ring {k}'s 0..{n - 1}")
        out.setdefault(k, set()).add(i)
    return out


def _clean_reasons(reason: Union[str, Mapping[Tuple[int, int], str]]) -> Tuple[str, Dict[Tuple[int, int], str]]:
    """(the reason for every cluster, or the per-(ring, position) reasons) from ``clean_rings``' ``reason``."""
    if isinstance(reason, str):
        return reason, {}
    if not isinstance(reason, Mapping):
        raise ValueError(f"clean_rings: reason must be a str or a mapping (ring index, position) -> str, got {type(reason).__name__}")
    per: Dict[Tuple[int, int], str] = {}
    for key, value in reason.items():
        try:
            k_raw, i_raw = key
        except (TypeError, ValueError) as exc:
            raise ValueError(f"clean_rings: a reason key must be a (ring index, position) pair, got {key!r}") from exc
        per[(_whole_index(k_raw, "ring index"), _whole_index(i_raw, "cluster position"))] = str(value)
    return "", per


def _clean_one_ring(
    ring: Ring,
    drop: Set[int],
    event_id_all: Optional[NDArray[np.int64]],
    lpz: Optional[NDArray[np.float64]],
    margin_nm: float,
    registration: str,
) -> Ring:
    """A copy of ``ring`` (already a private deep copy) without the clusters at positions ``drop``: its analysis by
    ``without_clusters`` (steps 3-6 on the remaining centroid rows, the contour by 2-opt from every start, exactly
    how ``build_rings`` built it), the contour, length and area centroid from that perimeter, the dropped clusters'
    localizations out of ``loc_index`` / ``event_id``, the counts recomputed."""
    k = len(ring.clusters)
    an = ring.analysis
    if an.discard_applied:
        raise ValueError(f"clean_rings: ring {ring.index} was cleaned already (its analysis carries a discard); clean the "
                         "rings build_rings returned, with every cluster to drop at once")
    if int(an.n_clusters_kept) != k:
        raise ValueError(f"clean_rings: ring {ring.index}: the analysis kept {an.n_clusters_kept} clusters, Ring.clusters "
                         f"has {k}")
    # The analysis' centroid rows are the ring's clusters in the same order (_build_clusters).
    rows = np.asarray(an.centroids, dtype=np.float64).reshape(-1, 2)
    exact = (np.array([np.asarray(cl.centroid_nm, dtype=np.float64).reshape(2) for cl in ring.clusters],
                      dtype=np.float64).reshape(-1, 2) if k else np.zeros((0, 2), dtype=np.float64))
    if rows.shape != exact.shape or (rows.size and float(np.abs(rows - exact).max()) > CLEAN_ROW_TOL_NM):
        raise ValueError(f"clean_rings: ring {ring.index}: the analysis' centroid rows do not match Ring.clusters")
    flags = np.zeros(k, dtype=bool)
    if drop:
        flags[sorted(drop)] = True
    new_an = without_clusters(an, flags, margin_nm=float(margin_nm), registration=str(registration))
    per = new_an.perimeter
    kept = [cl for i, cl in enumerate(ring.clusters) if not flags[i]]
    dropped = [cl for i, cl in enumerate(ring.clusters) if flags[i]]
    loc_index = np.asarray(ring.loc_index, dtype=np.intp)
    if dropped:
        gone = np.concatenate([np.asarray(cl.loc_index, dtype=np.intp).reshape(-1) for cl in dropped])
        keep = ~np.isin(loc_index, gone)
    else:
        keep = np.ones(loc_index.size, dtype=bool)
    new_index = loc_index[keep]
    warnings_ = list(ring.warnings)
    if dropped:
        warnings_.append(f"clean_rings: dropped {len(dropped)} cluster(s) at positions {sorted(drop)} "
                         f"(labels {[int(cl.label) for cl in dropped]}, {int((~keep).sum())} localizations)")
    lpz_median = ring.lpz_median_nm
    if lpz is not None:
        lpz_median = _nanmedian_or_nan(lpz, new_index)
    elif dropped:
        warnings_.append("clean_rings: lpz_median_nm kept from before the cleaning (no lpz_nm given)")
    centroid: Optional[NDArray[np.float64]] = None
    contour: Optional[NDArray[np.float64]] = None
    length = float("nan")
    if per is not None:
        contour = np.asarray(per.contour, dtype=np.float64)
        length = float(per.perimeter_nm)
        if per.centre is not None:
            centroid = np.array([per.centre.x_nm, per.centre.y_nm], dtype=np.float64)
        else:
            warnings_.append(f"ring {ring.index}: the rebuilt contour has no area centroid (it crosses itself or "
                             "encloses no area)")
    elif len(kept) < 3:
        warnings_.append(f"clean_rings: ring {ring.index} keeps {len(kept)} cluster(s): no contour (it cannot enter the "
                         "matching)")
    return dataclasses.replace(
        ring,
        clusters=kept,
        analysis=new_an,
        contour_nm=contour,
        length_nm=length,
        centroid_nm=centroid,
        loc_index=new_index,
        event_id=None if ring.event_id is None else np.asarray(ring.event_id, dtype=np.int64)[keep],
        n_locs=int(new_index.size),
        n_events=_count_events(event_id_all, new_index),
        lpz_median_nm=lpz_median,
        warnings=warnings_,
    )


def clean_rings(
    res: RingsResult,
    flagged: Iterable[Tuple[int, int]],
    *,
    reason: Union[str, Mapping[Tuple[int, int], str]] = "",
    lpz_nm: Optional[NDArray[np.float64]] = None,
    force_rebuild: bool = False,
    margin_nm: float = CLEAN_RECORDED_MARGIN_NM,
    registration: str = "",
) -> RingsResult:
    """
    The rings of ``res`` without the ``flagged`` clusters (H5-D, D-35; the
    port of the feasibility study's ``clean_rings`` research script).

    Why it exists (D-32d): a cluster of the axon's lumen that stays in the
    rings deforms every curve fitted through them -- each ring's contour,
    the pooled membranes of the arc test -- by several sd of the arc
    test on simulated axons, and excluding it from the matching alone
    (``analyze_arc_columns(exclude_clusters=...)``) does not remove it from
    the curves. Removing it means rebuilding everything built from it.

    Parameters
    ----------
    res : RingsResult
        The rings as ``build_rings`` returned them (or a result of this
        function whose touched rings are not touched again: see Raises).
        Never modified: the work is done on a deep copy.
    flagged : iterable of (Ring.index, position) pairs
        The clusters to drop, ``position`` being the index in that ring's
        ``Ring.clusters``. A repeated pair counts once.
    reason : str or mapping
        Recorded with every dropped cluster in ``removed_clusters``: one
        string for all, or a mapping (ring index, position) -> str ("" for
        a pair the mapping lacks).
    lpz_nm : (M,) array, optional
        The axial precision of every localization ``build_rings`` received;
        when given, each rebuilt ring's ``lpz_median_nm`` is recomputed over
        the localizations it keeps, otherwise it is kept, with a warning.
    force_rebuild : bool
        Rebuild EVERY ring, dropped clusters or not (the self-check: with
        nothing flagged the geometry must come back bit for bit).
    margin_nm, registration
        What the rebuilt ring analyses record as their discard
        (``tools.mps_analysis.without_clusters``); provenance only.

    Returns
    -------
    RingsResult
        A new result in which, for every ring with a drop:

        * ``Ring.clusters`` loses the flagged clusters (the others keep
          their order and are the same objects' copies, labels and
          centroids unchanged, so ``tools.mps_lumen.stable_cluster_key``
          still names them);
        * ``Ring.analysis`` is ``without_clusters`` of the ring's analysis:
          areas, contour, 1NN and occupancy run again on the remaining
          centroid rows with the analysis' own settings, the contour by
          2-opt from every start, which is exactly how ``build_rings`` got
          it (``analyze_all_segments`` -> ``analyze_axon(all_starts=True)``
          -> ``reconstruct_perimeter(centroids, refine=True,
          all_starts=True)``); the slab arrays and the DBSCAN labels are
          those of the input, the dropped labels recorded as discarded;
        * ``Ring.contour_nm``, ``length_nm`` and ``centroid_nm`` come from
          that perimeter (None / NaN / None with fewer than three clusters
          left, as ``build_rings`` leaves a ring without a perimeter);
        * ``Ring.loc_index`` loses the dropped clusters' localizations (the
          ring's unclustered localizations stay: the localization membrane
          reads them), and ``event_id``, ``n_locs``, ``n_events`` follow.

        ``n_suspect`` is recounted, ``removed_clusters`` is the input's list
        plus one ``RemovedCluster`` per dropped cluster in (ring index,
        position) order, and a warning records what was dropped. ``x_p``,
        ``y_p``, ``z_p``, ``event_id`` (every input localization) and
        ``ms`` are unchanged: the membranes read ``Ring.loc_index`` and the
        matching reads ``Ring.clusters`` and ``Ring.analysis.perimeter``,
        so nothing downstream sees a dropped cluster. Rings without a drop
        are untouched (unless ``force_rebuild``); with nothing flagged the
        result is an unchanged copy of ``res``.

    Raises
    ------
    ValueError
        A ring index that names no ring of ``res``, a position outside
        that ring's clusters, a flagged entry that is not an integer pair,
        an ``lpz_nm`` of the wrong length, a ring analysis whose centroid
        rows do not match ``Ring.clusters``, or a ring to rebuild whose
        analysis already carries a discard (``without_clusters`` rebuilds
        from the analysis of ALL the clusters: clean the rings of
        ``build_rings`` with every cluster to drop at once).
    """
    drop = _clean_flag_map(res, flagged)
    common, per_key = _clean_reasons(reason)
    lpz: Optional[NDArray[np.float64]] = None
    if lpz_nm is not None:
        lpz = np.asarray(lpz_nm, dtype=np.float64).ravel()
        n_input = int(np.asarray(res.x_p).size)
        if lpz.size != n_input:
            raise ValueError(f"clean_rings: lpz_nm has {lpz.size} values for the {n_input} localizations build_rings received")
    previous = list(getattr(res, "removed_clusters", None) or [])
    out = copy.deepcopy(res)
    if not drop and not force_rebuild:
        return dataclasses.replace(out, removed_clusters=list(copy.deepcopy(previous)))
    by_index_in = {int(r.index): r for r in res.rings}
    rings: List[Ring] = []
    for ring in out.rings:
        d = drop.get(int(ring.index), set())
        if d or force_rebuild:
            rings.append(_clean_one_ring(ring, d, out.event_id, lpz, margin_nm, registration))
        else:
            rings.append(ring)
    removed: List[RemovedCluster] = list(copy.deepcopy(previous))
    for k in sorted(drop):
        src = by_index_in[k]
        for i in sorted(drop[k]):
            cl = src.clusters[i]
            removed.append(RemovedCluster(
                ring=int(k), position=int(i), label=int(cl.label),
                centroid_nm=np.array(np.asarray(cl.centroid_nm, dtype=np.float64).reshape(2), dtype=np.float64),
                n_locs=int(cl.n_locs), reason=per_key.get((int(k), int(i)), "") if per_key else str(common)))
    n_dropped = sum(len(v) for v in drop.values())
    note = (f"clean_rings: {n_dropped} cluster(s) dropped "
            f"({', '.join(f'ring {k}: {sorted(v)}' for k, v in sorted(drop.items())) or 'none'}); the contours of the "
            f"{'rings' if force_rebuild else 'touched rings'} rebuilt by 2-opt from every start")
    return dataclasses.replace(out, rings=rings, n_suspect=_count_suspects(rings), warnings=list(out.warnings) + [note],
                               removed_clusters=removed)
