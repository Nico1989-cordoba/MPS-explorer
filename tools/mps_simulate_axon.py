# -*- coding: utf-8 -*-
"""
Localization-level simulator of one axon for the intra-axon H-ECL power
study: milestone H5, module C (01_formalizacion S1.3 the generative
models M1, M2, M3b, M5, M6, M7 and the common measurement stage;
02_investigacion B8 the minimal model of localizations; 03_plan S3.2
``simulate_axon``; DECISIONES D-19 the NeNA ratio by depth, D-25 the
radial offset of the clusters about the smoothed membrane).

Why this exists
---------------
The column analysis (H3) and its leak diagnostics (H4) are decided on
statistics whose false-positive rate under a realistic measurement
(blinking, linkage offsets, an axial precision the file under-reports by
depth, structural width, leak across the ring cuts, background) and
whose power under M5 / M6 / M7 can only be measured on simulated axons
with a known truth per localization. This module makes such axons: a
membrane contour (a measured ring's smoothed contour, a library entry
or an ellipse with a Fourier perturbation), rings at the period,
clusters on the contour under one generative model, and the measurement
stage of 02 B8, every nuisance parameter of which is measurable on a
real axon (``sim_config_from_axon``), pooled over a dataset
(``pool_sim_configs``) and frozen in a strict YAML
(``write_sim_config`` / ``load_sim_config``).

What is here
------------
``SimConfig``, ``SimClusterTable``, ``SimGeometry``, ``SimLocalizations``,
``SimAxon``; ``simulate_axon``, ``cluster_positions``, ``membrane_point``,
``measure``, ``radial_scatter_nm`` (the membrane / scatter split of a
measured ring), ``sim_config_from_axon`` (with its closure calibration
against the pipeline), ``pool_sim_configs``, ``write_sim_config``,
``load_sim_config``, ``contour_library_path``, ``pick_polygon_of_rings``.

Conventions
-----------
* Units in every name (nm, um, deg, frames); inputs and outputs in nm.
* Determinism: ``rng = np.random.default_rng(np.random.SeedSequence(seed))``
  and a FIXED order of draws: the Fourier phases (only when a
  perturbation is requested), the ring offsets eta_k, then ring by ring
  the cluster positions and the radial offsets (geometry), then ring by
  ring the measurement (fluorophores per cluster, epitope disc, linkage,
  structural width, blinking, frames, lateral precision and errors,
  axial precision and errors), then the background. Same seed and
  config: bit-identical output.
* Localizations are returned in FRAME order (as a Picasso file): the
  event linking of ``build_rings`` depends on it.
* The membrane frame at a cluster: the contour is INTERPOLATED
  (``tools.mps_matching.smooth_closed_path``, splprep s = 0) because it
  is already smooth; the tangent is the direction of the table's
  sub-chord at the arc and the outward normal follows the convention of
  ``tools.mps_unroll.project_on_path`` (-sign of the tour's signed area).
* The epitope disc lies in the membrane plane (tangent x axial); the
  linkage offset is N(0, sigma_link) along the tangent, the normal and
  z, drawn once per fluorophore; the axial error of a localization is
  N(0, lpz_reported x scale(z)) with lpz_reported resampled from the
  stratum of its TRUE laboratory z and scale the D-19 NeNA ratio.
* The background is uniform in the membrane's bounding box widened by
  ``background_margin_nm`` (the pick's margin, measured on the real
  axon), NOT in the specification's tangent box: tangent, the box is the
  hull the pipeline's edge criterion falls back to and it cuts the same
  arcs of every ring (``BACKGROUND_MARGIN_DEFAULT_NM``; D-28 candidate).
* No Qt, no matplotlib; nothing here modifies its inputs; the only
  subprocess is a read-only ``git rev-parse`` for the provenance.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import warnings
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.special import gammaln
from scipy.stats import norm

from tools.mps_axial_precision import AxialNena, axial_nena
from tools.mps_columns import (
    Ring,
    RingsParams,
    RingsResult,
    _check_provenance,
    _plain,
    _yaml_load_strict,
    _yaml_module,
    build_rings,
)
from tools.mps_matching import SmoothPath, smooth_closed_path
from tools.mps_membrane import (  # noqa: F401  re-exported: H5-B moved the membrane P-spline there (same objects)
    MEMBRANE_KNOT_SPACING_NM,
    MEMBRANE_NULL_KNOT_SPACING_NM,
    MEMBRANE_PENALTY_REL,
    LocalizationMembrane,
    MembraneFit,
    _periodic_bspline_design,
    fit_membrane_from_localizations,
    localization_membrane_of_rings,
    radial_scatter_nm,
    radial_scatter_on_membrane,
    smooth_polygon_membrane,
)
from tools.mps_randomization import smooth_contour_bspline
from tools.mps_unroll import project_on_path

__all__ = [
    "BACKGROUND_MARGIN_DEFAULT_NM",
    "DECILE_PROBABILITIES",
    "ELLIPSE_VERTICES",
    "LAMBDA_CONVENTION",
    "LENGTH_BASES",
    "LPZ_SAMPLES_PER_BIN_MAX",
    "LP_SAMPLES_MAX",
    "MEASURED_CONTOUR_VERTICES",
    "CONTOUR_SMOOTHING_MODES",
    "MEMBRANE_SOURCES",
    "MODELS",
    "SimAxon",
    "SimClusterTable",
    "SimConfig",
    "SimGeometry",
    "SimLocalizations",
    "TRANSIENT_FIELDS",
    "cluster_positions",
    "contour_library_path",
    "effective_clusters_per_um",
    "load_sim_config",
    "measure",
    "membrane_point",
    "membrane_scale_contour",
    "pick_polygon_of_rings",
    "pool_sim_configs",
    "radial_scatter_nm",
    "sim_config_from_axon",
    "simulate_axon",
    "write_sim_config",
]

# The generative models of 01 S1.3 the simulator knows.
MODELS: Tuple[str, ...] = ("M1", "M2", "M3b", "M5", "M6", "M7")
# Vertices of the ellipse contour: 360 vertices at uniform parametric
# angle are ~22 nm apart on the 1500 x 1000 nm ellipse, so the
# interpolating spline through them is the ellipse to < 1e-3 nm and a
# Fourier harmonic up to m ~ 20 is sampled far above Nyquist.
ELLIPSE_VERTICES = 360
# Vertices of a measured ring's smoothed contour kept in a configuration
# (~40 nm apart on a 7.5 um perimeter; the spline through them is the
# smoothed curve to < 0.1 nm).
MEASURED_CONTOUR_VERTICES = 200
# Empirical precision samples kept per axon: the samples are resampled
# with replacement, so 1000 quantile-thinned values reproduce a
# distribution's percentiles to 0.1 % and keep a pooled YAML of a whole
# dataset under ~2 MB (a full axon carries 10^4-10^5 localizations).
LP_SAMPLES_MAX = 1000
LPZ_SAMPLES_PER_BIN_MAX = 500
# Sequential rejection sampler of the hard core: attempts per cluster
# before the ring is declared jammed (the same budget as
# validate_columns_h3.hard_core_arcs_nm).
HARD_CORE_MAX_ATTEMPTS = 200000
# Calibration of the measured hard core by quantile matching
# (``_hard_core_by_quantile_matching``): rings drawn per evaluation (20
# rings of ~40-60 clusters give ~1000 gaps, a 5th percentile stable to
# ~1 nm), the fixed seed of those draws, the attempt budget per cluster
# before "jammed" counts as "d too large", and the bisection tolerance.
HARD_CORE_CALIBRATION_RINGS = 20
HARD_CORE_CALIBRATION_SEED = 20260925
HARD_CORE_CALIBRATION_ATTEMPTS = 2000
HARD_CORE_CALIBRATION_TOL_NM = 0.25
# A cluster within this distance of a lattice site occupies it (M2).
LATTICE_TOL_NM = 1e-6
# The nine deciles of n_locs per cluster.
DECILE_PROBABILITIES: NDArray[np.float64] = np.linspace(0.1, 0.9, 9)
# ``MEMBRANE_KNOT_SPACING_NM`` and ``MEMBRANE_PENALTY_REL`` (the periodic
# P-spline that separates a measured ring's MEMBRANE from the radial
# SCATTER of its clusters, ``radial_scatter_nm``, ``MembraneFit``) live in
# ``tools.mps_membrane`` since H5-B and are imported above with the same
# names: the argument for 600 nm knots and the 1 % penalty is there.
# Radius around every kept cluster centroid (of every ring, axon frame)
# within which a ring's unclustered localizations are NOT counted as
# background: DBSCAN's border losses of the true clusters and the leak
# children of the neighbouring rings (which sit at the neighbours'
# cluster positions) live there. 100 nm is several cluster widths
# and below half the hard core, so the excluded discs cover a
# small, measurable fraction of the bounding box (``_background_per_um3``).
BACKGROUND_EXCLUSION_NM = 100.0
BACKGROUND_GRID = 200
# Closure calibration of the measured configuration (02 B8 step 1): the
# simulator is run from the measured configuration, its rings are built
# with the same parameters, and the clusters per um, the n_locs deciles
# and the background are rescaled by the ratio of the REAL detected
# quantities to the SIMULATED detected ones, this many times, with this
# seed, each ratio clipped to this range (a fixed point of the
# measurement rather than the measurement itself, because DBSCAN loses
# the small clusters, the windows cut the leak twice and the background
# estimate carries what the detection missed: on a realistic axon the
# uncalibrated round trip re-detected clearly fewer clusters than K,
# review of 2026-09-25).
CALIBRATION_ITERATIONS = 3
CALIBRATION_SEED = 20260926
CALIBRATION_RATIO_CLIP = (0.5, 2.0)
# Simulated axons per calibration iteration, whose DETECTED quantities
# are pooled before the ratio is taken: one axon's detection varies
# more than the binomial noise of K (the closure trajectory swung
# widely per iteration, mostly the shared edge losses of the tangent
# background box, re-review of 2026-09-25; with the box widened the
# factors still moved noticeably per iteration), and two axons halve
# the variance of each factor for
# twice the cost (~8 s per axon on this machine).
CALIBRATION_AXONS_PER_ITERATION = 2
# The n_locs deciles are rescaled by ONE factor, the geometric mean of
# the real / simulated ratios of the deciles from this index up (the
# median and above): the detected size of a small cluster is floored by
# DBSCAN's min_samples, so the low deciles of the DETECTED sizes do not
# move when the true small clusters shrink and a per-decile ratio would
# push them to 0 (in three iterations on a realistic axon).
CALIBRATION_DECILE_FROM = 4
# H5-E (D-40, spec 2.2): the measurement closure versions. "v1" is the
# closure above (value for value); "v2" adds, inside
# ``_calibrate_by_simulation``, F1 -- the axial error split closed on
# what the estimators SEE (the NeNA ratio per stratum of every calibration
# axon, and each ring's component sigma_z), not on the subtraction alone,
# because the NeNA estimator is itself biased in dense geometry (on a
# research harness entry it read the axial scale too high, which left
# sigma_struct well below its truth and the null liberal) -- and F2 --
# common random numbers (the SAME calibration seeds in every iteration, so
# the detection noise does not random-walk the fixed point) over more axons
# and iterations, and the final lambda and decile scale solved from the
# log-log regression of the simulated statistic on the parameter over the
# iterates (the "binding function" of indirect inference) instead of the
# last ratio step, which overshot the compressive detection map on K
# and on the deciles (R-E1, every entry).
CLOSURE_VERSIONS: Tuple[str, ...] = ("v1", "v2")
CALIBRATION_V2_ITERATIONS = 4
CALIBRATION_V2_AXONS = 6
# The binding regression is used only when its slope lies in this range and
# at least this many iterates are usable; otherwise the last iterate stands
# (recorded in the trajectory). The solution is also refused when it lies
# outside CALIBRATION_RATIO_CLIP of the last iterate.
BINDING_SLOPE_RANGE = (0.3, 3.0)
BINDING_MIN_ITERATES = 3
# NeNA strata ratios need this many consecutive-frame pairs (power_columns'
# NENA_MIN_PAIRS) on a calibration axon to enter the F1 closure.
CLOSURE_NENA_MIN_PAIRS = 50
# Rings whose NeNA-calibrated error width (lpz median x scale at the
# ring's depth) is within this fraction of the smallest one are the
# "best-resolved" rings, whose signed structural variances are AVERAGED
# for sigma_struct (``sim_config_from_axon``). The MAX over rings that
# the earlier lower-bound argument used has a floor: at a truth of 0
# the largest of three zero-mean errors (sd ~15 nm after the square
# root) read 10-17 nm (re-review of 2026-09-25). On the realistic geometry
# (error widths growing with depth) only the shallowest ring qualifies --
# the tightest bound, the deeper rings being biased low by the overlap
# and resolving nothing -- and the estimate is the same as the max's;
# with rings of one stratum all three qualify and their mean is
# unbiased.
RESOLVED_LPZ_TOLERANCE = 0.15
# Margin of the background box beyond the membrane's bounding box
# (``SimConfig.background_margin_nm``; measured on a real axon by
# ``sim_config_from_axon`` as the distance between the pick's
# localizations and the membrane). The specification's box is the axon's
# own bounding box, tangent to the membrane: without an ROI the
# pipeline's edge-touching criterion then falls back to the convex hull
# of the localizations, i.e. that box, and flags every cluster within
# eps of its four tangent points on EVERY ring (a sizeable share of the
# kept clusters, the same arcs on every ring; a shared depletion the
# arc-shift null does not reproduce, which shifted z_A upward on a
# leak-free M1 null; re-review of 2026-09-25). A hand-drawn pick leaves a margin
# around the axon, so the default is a margin far above the cluster
# extent (eps 25 nm + 3 sd of a 20-30 nm-wide cluster + 3 sd of the
# radial scatter ~ 130 nm): nothing is flagged at the tangent points,
# exactly as with the reviewer's 1 um ROI, and the measured pick margin
# replaces it in a measured configuration (D-28 candidate).
BACKGROUND_MARGIN_DEFAULT_NM = 500.0
# H5-C (H5C_SPEC S3(a); D-28h, D-29c): the LENGTH ``clusters_per_um`` refers
# to. A lambda measured on a real axon is K / L with L the ring's contour
# length (``Ring.length_nm``: the perimeter through the centroids, which
# carries the centroid scatter and is longer than the 600 nm
# membrane), and ``simulate_axon`` places Poisson(lambda x L_used) clusters
# on the contour it USES -- the resolved contour smoothed at
# ``contour_smoothing_knot_nm`` when that is set. A simnull that
# calibrated lambda on the raw contour and applied it to the much shorter
# smoothed one (D-28h) simulated too few clusters per ring and lost the
# deep ring. ``SimConfig.length_basis`` records which length the
# configuration's lambda was calibrated on: "raw" -- the contour as
# resolved, before any smoothing -- makes ``simulate_axon`` rescale lambda
# by L_raw / L_used when it smooths the contour (``effective_clusters_per_
# um``), so that E[K] = lambda x L_raw whatever the smoothing; "smoothed"
# says lambda was calibrated on the contour as used (``sim_config_from_axon
# (membrane_source="localizations")``: the contour is the localization
# membrane, already at its knot scale, or the calibration ran with the
# smoothing applied) and nothing is rescaled; None (the default, every
# configuration written before H5-C) keeps today's behaviour bit for bit:
# lambda on the contour as used, no rescale (grid v2 compensated by hand
# with a ``clusters_per_um`` override, which v3 no longer needs).
LENGTH_BASES: Tuple[str, ...] = ("raw", "smoothed")
# H5-C (H5C_SPEC S3(b); D-28b, D-29c): where ``sim_config_from_axon`` takes
# the contour and the radial scatter from. "centroid_pspline" (the default,
# today's estimator): the reference ring's 600 nm P-spline through its
# centroids and the dof-corrected residual sd of the centroids about their
# own ring's fit (not always identifiable: 0 with a
# warning, D-28b). "localizations": the membrane fitted to the
# LOCALIZATIONS of every ring with a contour (``tools.mps_membrane.
# localization_membrane_of_rings``, knots every ``MEMBRANE_KNOT_SPACING_NM``)
# is the contour, and the radial scatter is ``radial_scatter_on_membrane``
# of every ring's centroids on it (identifiable: no length-change guard, no
# fallback 0; the old estimate is kept in the provenance as
# ``radial_offset_sd_centroid_pspline_nm``). The same curve is what the
# arc test projects on (``tools.mps_unroll.analyze_arc_columns(
# reference_curve="localization_membrane")``), so the simulated axon's
# scatter about ITS localization membrane is the observed axon's about
# its own: the quantity ``simnull`` must reproduce (D-29c).
MEMBRANE_SOURCES: Tuple[str, ...] = ("centroid_pspline", "localizations")
# H5-C (orchestrator's note of 2026-09-26 on the membrane stage's pilot;
# D-29c): HOW ``contour_smoothing_knot_nm`` smooths the resolved contour.
# "polygon" (None, the default: grid v2 and every configuration written
# before H5-C, bit for bit) is ``smooth_polygon_membrane``: a P-spline in
# the polygon's OWN chord parameter with round(L_chord / knot) knots. The
# library entries are s = K spline smoothings of real rings' centroid
# polygons whose chord runs longer than the membrane they describe (the
# zig-zag of the centroid scatter), so those knots fall closer than
# 600 nm along the resulting curve and the "600 nm" contour keeps
# structure below 600 nm that the arc test's 600 nm localization membrane cannot
# follow. Every simulated ring shares that structure, the membrane misses
# it in the same places for every ring, and the rings' projected arcs
# inherit a common distortion: the membrane stage's pilot measured the arc
# test on the localization membrane clearly liberal WITHOUT leak on such
# a contour, against close to level with the TRUE contour as the
# circle and on a contour re-smoothed until it was smooth at the
# basis's own scale -- the simulator's artefact, D-28b's mechanism one
# scale up. "membrane" makes the smoothed contour the curve the
# localization-membrane estimator itself would return from noise-free
# samples of the resolved contour (``membrane_scale_contour``: the
# polygon P-spline as the initial curve, then ``tools.mps_membrane.
# fit_membrane_from_localizations`` with knots every ``contour_smoothing_
# knot_nm`` of the curve's OWN arc, its two iterations, no robust drop), so
# the simulated membrane lies in the span of the basis the arc test
# estimates it with; applied to a contour that already is such a membrane
# (the localization membranes ``measure`` writes from H5-C on) it changes
# it by a few nm (idempotent up to the penalty's shrinkage, measured in
# the report of this stage). What this cannot do: if REAL membranes carry
# shared radial structure below the knot spacing, the real arc test is
# biased by it the same way and a basis-consistent simulator does not
# reproduce that bias -- not decidable on synthetics (``simnull`` reports
# the sub-basis structure of the observed axon next to its simulations').
CONTOUR_SMOOTHING_MODES: Tuple[str, ...] = ("polygon", "membrane")
# ``membrane_scale_contour``: the resolved contour is sampled every this
# many nm of its interpolating curve (noise-free "localizations"; ~3000
# points on a realistic contour, a 0.05-0.1 s fit), and the robust drop of the
# localization fit is switched off with this constant (the offsets of
# exact samples from the initial curve are the structure the fit must
# absorb, not outliers).
CONTOUR_MEMBRANE_SAMPLE_STEP_NM = 5.0
CONTOUR_MEMBRANE_ROBUST_C = 1.0e6
# The smoothed contour is kept as a polygon at equal arcs, one vertex per
# this many nm (at least MEASURED_CONTOUR_VERTICES): simulate_axon then
# interpolates it (splprep s = 0), which reproduces a curve with 600 nm
# knots to well below 1 nm at a 50 nm spacing.
CONTOUR_MEMBRANE_VERTEX_SPACING_NM = 50.0
# Fields of ``SimConfig`` that are bookkeeping of the loaded instance,
# not parameters: never written to the YAML, rejected as YAML keys,
# excluded from equality; they survive ``dataclasses.replace`` and
# pickling because they ARE fields (a plain attribute did not, and the
# process-wide registry keyed by the relative library name that replaced
# it redirected a replaced copy of one configuration to another YAML's
# library of the same name; re-review of 2026-09-25).
TRANSIENT_FIELDS: Tuple[str, ...] = ("library_dir",)
# The fields H5-C added to ``SimConfig`` (all default None = the behaviour
# before H5-C). ``write_sim_config`` writes them only when set, so that a
# configuration that does not use them is written exactly as before
# (review of 2026-09-26: an existing output must not change); an absent
# key loads as None, so the round trip stays exact.
_H5C_OPTIONAL_FIELDS: Tuple[str, ...] = ("length_basis", "contour_smoothing_mode", "ring_rate_factors")
# H5-D (H5D_SPEC ADDENDUM v2 B "CLI"; D-32c): the field H5-D added (``SimConfig.pick_polygon_nm``, default None =
# the box background of before, bit for bit). Written only when set, like the H5-C fields, so that no existing
# configuration file changes.
_H5D_OPTIONAL_FIELDS: Tuple[str, ...] = ("pick_polygon_nm",)
# H5-D (D-32c, written into the provenance of the configurations that use the pick background): the quantity the
# closure calibration (``_calibrate_by_simulation``) closes lambda on. It is K OWN -- the kept clusters of each ring
# that are NOT axially one-sided (``_axially_one_sided``: the leak children, which the simulation re-creates itself)
# -- ring by ring under ``ring_rate_factors``, never every kept cluster: closing on the total K counts the leak
# copies (which can be a large share of a ring) as clusters of their own and simulates too many independent clusters
# (D-32c: at that density even the true curve was biased without leak). Recorded, not selectable: the code has
# always closed on K own; D-32 asked for the convention to be written where a configuration is read.
LAMBDA_CONVENTION = "K_own"
LAMBDA_CONVENTION_NOTE = ("clusters_per_um (and ring_rate_factors, ring by ring) closed by the calibration on K own: the "
                          "kept clusters of each ring that are NOT axially one-sided (_axially_one_sided; the leak children "
                          "are re-created by the simulation), never on every kept cluster (D-32c)")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GIT_COMMIT: Optional[str] = None

SIM_CONFIG_HEADER: Tuple[str, ...] = (
    "# Simulation configuration of tools/mps_simulate_axon.py (H5, module C).",
    "# Written by write_sim_config; read back by load_sim_config, which is strict:",
    "# an unknown or repeated key raises. Units in every name (nm, um, deg, frames).",
    "# Arrays are lists; contour_nm is a closed polygon [[x, y], ...] in the axon frame;",
    "# lpz_samples_by_bin_nm holds one list per stratum of lpz_bin_edges_lab_nm.",
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SimConfig:
    """
    Everything ``simulate_axon`` decides with (03_plan S3.2; 02 B8).

    Geometry: ``contour_nm`` is a closed polygon (K0, 2) of the membrane
    in the axon frame (a real ring's smoothed contour); None falls back
    to the library entry ``contour_name`` of ``contour_library_npz``
    (an NPZ next to the YAML, one (K, 2) array per axon) and, failing
    that, to the ellipse ``ellipse_semi_axes_nm`` with the relative
    radial harmonics ``fourier_perturbation`` (amplitudes of m = 2, 3,
    ..., phases from the seed). ``n_rings`` rings sit at
    ``z_middle_lab_nm + (k - (n_rings - 1) / 2) period_nm + eta_k`` with
    eta_k ~ N(0, ``sigma_period_nm``) independent per ring. Clusters per
    ring: ``n_clusters_per_ring`` when given, else Poisson(``clusters_per_um``
    x L / 1000); ``d_min_nm`` is the hard core along the arc; each true
    centre sits ``N(0, radial_offset_sd_nm)`` along the outward normal
    from the contour (D-25).

    Model: ``model`` in ``MODELS``; ``f``, ``sigma_col_nm``, ``q`` for M5
    (see ``cluster_positions`` for the joint rule of f and q);
    ``alpha_deg_per_ring`` for M3b; ``jitter_nm`` for M3b / M6 / M2;
    ``m2_*`` for M2; ``m7_z_half_width_nm`` for M7 (None: the whole axial
    span of the rings, min ring z - P/2 to max ring z + P/2).

    Measurement (02 B8 minimal model): fluorophores per cluster ~
    Poisson(``n_fluor_mean``) (M epitopes x p_lab collapsed into one
    mean; ``p_lab`` kept for the record), or, when
    ``n_locs_per_cluster_quantiles`` gives the nine deciles of n_locs
    per cluster, n_fluor = round(sampled n_locs / ``locs_per_fluor_mean``);
    epitopes uniform in a disc of ``epitope_radius_nm`` in the membrane
    plane; linkage N(0, ``sigma_link_nm``) per axis, once per
    fluorophore; localizations per fluorophore ~ Geometric with mean
    ``locs_per_fluor_mean`` in consecutive frames (gaps up to
    ``max_dark_frames``) from a random start in [0, ``n_frames``); lateral
    error N(0, lp) with lp resampled from ``lp_lateral_samples_nm``;
    axial error N(0, lpz x scale) with lpz resampled from the stratum of
    ``lpz_bin_edges_lab_nm`` holding the TRUE laboratory z
    (``lpz_samples_by_bin_nm``; an empty stratum borrows the nearest)
    and scale = ``axial_scale_by_bin`` of that stratum (D-19);
    ``sigma_struct_nm`` per fluorophore in z; ``background_per_um3``
    spurious localizations uniform in the bounding volume, the
    membrane's bounding box widened by ``background_margin_nm`` on every
    side (the pick's margin around the axon; see
    ``BACKGROUND_MARGIN_DEFAULT_NM`` for why the specification's
    tangent box is not used) times the rings' axial span; ``tilt_deg``
    at ``azimuth_deg`` rotates the axon into the laboratory frame;
    ``pixel_size_nm`` is for the record (inputs are nm).
    ``pick_polygon_nm`` (H5-D; None by default) keeps only the
    localizations inside the real pick (the comment at the field).
    ``contour_smoothing_knot_nm`` (H5-B fix; None by default) replaces
    the resolved contour by its P-spline membrane with knots every that
    many nm before the clusters are placed (the comment at the field:
    the contour library carries the real rings' scatter as membrane
    structure, which the simulator would otherwise add twice). Every
    value is validated in ``__post_init__`` (ValueError), lists become
    tuples or arrays, so a loaded and a built configuration compare
    field by field.
    ``library_dir`` is transient bookkeeping (``TRANSIENT_FIELDS``): the
    folder of the YAML the instance was loaded from, where a relative
    ``contour_library_npz`` is looked up; never written, not compared.
    """

    contour_nm: Optional[NDArray[np.float64]] = None
    ellipse_semi_axes_nm: Tuple[float, float] = (1500.0, 1000.0)
    fourier_perturbation: Tuple[float, ...] = ()
    n_rings: int = 3
    period_nm: float = 190.0
    sigma_period_nm: float = 0.0
    z_middle_lab_nm: float = 0.0
    clusters_per_um: float = 4.0
    n_clusters_per_ring: Optional[int] = None
    d_min_nm: float = 80.0
    radial_offset_sd_nm: float = 0.0
    model: str = "M1"
    f: float = 0.0
    sigma_col_nm: float = 10.0
    q: float = 1.0
    alpha_deg_per_ring: float = 0.0
    jitter_nm: float = 5.0
    m2_lattice_nm: float = 200.0
    m2_p_occupied: float = 0.8
    m2_h_inherit: float = 0.0
    m7_z_half_width_nm: Optional[float] = None
    n_fluor_mean: float = 25.0
    p_lab: float = 1.0
    n_locs_per_cluster_quantiles: Optional[Tuple[float, ...]] = None
    epitope_radius_nm: float = 25.0
    sigma_link_nm: float = 12.0
    locs_per_fluor_mean: float = 2.0
    max_dark_frames: int = 0
    n_frames: int = 60000
    lp_lateral_samples_nm: NDArray[np.float64] = field(default_factory=lambda: np.array([8.0]))
    lpz_bin_edges_lab_nm: NDArray[np.float64] = field(default_factory=lambda: np.array([-1000.0, 1000.0]))
    lpz_samples_by_bin_nm: List[NDArray[np.float64]] = field(default_factory=lambda: [np.array([50.0])])
    axial_scale_by_bin: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0]))
    sigma_struct_nm: float = 0.0
    background_per_um3: float = 0.0
    background_margin_nm: float = BACKGROUND_MARGIN_DEFAULT_NM
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0
    pixel_size_nm: float = 113.0
    contour_name: Optional[str] = None
    contour_library_npz: Optional[str] = None
    # H5-B fix (statistics review of 2026-09-25): when set, the resolved
    # contour (config, library or ellipse) is replaced by its P-spline
    # membrane with knots every this many nm (``tools.mps_membrane.
    # smooth_polygon_membrane``, in the polygon's own order) before the
    # clusters are placed, and ``SimAxon.contour_used_nm`` is that curve.
    # The entries of a measured contour library are s = K spline
    # smoothings of real rings' centroid polygons and carry those rings'
    # radial scatter as membrane structure, on top of which the simulator
    # adds ``radial_offset_sd_nm`` -- a scatter measured ABOUT a 600 nm
    # P-spline membrane -- once more; every ring then shares that
    # roughness and every membrane null reads it as columns. 600 nm
    # (``MEMBRANE_KNOT_SPACING_NM``, the scatter estimate's knots) makes
    # the simulated membrane the curve the scatter refers to. None (the
    # default) keeps the contour as resolved, bit for bit.
    contour_smoothing_knot_nm: Optional[float] = None
    # H5-C: the length ``clusters_per_um`` was calibrated on (``LENGTH_BASES``;
    # None = unspecified, today's behaviour: lambda on the contour as used).
    # "raw" makes ``simulate_axon`` rescale lambda by L_raw / L_used when
    # it smooths the contour (``effective_clusters_per_um``).
    length_basis: Optional[str] = None
    # H5-C: how ``contour_smoothing_knot_nm`` smooths (``CONTOUR_SMOOTHING_MODES``;
    # None = "polygon", today's ``smooth_polygon_membrane``, bit for bit; "membrane"
    # = ``membrane_scale_contour``, the curve at the knot spacing of its own arc).
    contour_smoothing_mode: Optional[str] = None
    # H5-C (simnull's fidelity, D-29c "K per ring within 15 %"): relative cluster rates per ring (ring k draws with
    # clusters_per_um x factor_k, and a fixed n_clusters_per_ring x factor_k rounded), one per ring in z order, mean 1.
    # None (the default) = one rate for every ring, today's behaviour. A per-axon null conditions on the real axon's
    # ring-to-ring differences of K (a nuisance under H0, as the arc test's rotation null conditions on each ring's K):
    # with one common rate the observed rings' own spread -- Poisson ~15 % at K 45, plus a depth-dependent detection --
    # cannot be reproduced ring by ring (a harness axon of the research: detected K differing markedly between
    # rings, simulated medians equal).
    ring_rate_factors: Optional[Tuple[float, ...]] = None
    # H5-D (H5D_SPEC ADDENDUM v2 B; D-32c, a feasibility probe of the research): the REAL
    # pick as a closed polygon (P >= 3, 2) in the simulator's axon frame (the frame of ``contour_nm``: build_rings'
    # x', y' of the axon the configuration was measured on). When set, ``measure`` spreads the background over a box
    # that covers it (the contour's box widened by ``background_margin_nm``, grown to the polygon's bounding box when
    # that is larger) and then drops EVERY localization -- background or cluster, at its measured position -- outside
    # the polygon (even-odd rule): a real file is the pick's selection, so its background fills the pick at the
    # configured density and the pipeline's edge criterion bites where the real pick approaches the membrane, not at
    # the four tangent points of a box (D-32c: the box concentrated those losses at the contour's extremes). None (the
    # default) keeps the box background of before, bit for bit.
    pick_polygon_nm: Optional[NDArray[np.float64]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    library_dir: Optional[str] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Coerce every field to its declared type and validate its range."""
        _normalise_config(self)


def _is_number(value: Any) -> bool:
    """True for an int or float (NumPy included), never for a bool."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _as_float(name: str, value: Any, *, lo: Optional[float] = None, hi: Optional[float] = None,
              lo_open: bool = False, hi_open: bool = False) -> float:
    """``value`` as a finite float within [lo, hi] (open ends on request), or ValueError."""
    if not _is_number(value):
        raise ValueError(f"SimConfig.{name}: expected a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"SimConfig.{name}: must be finite, got {value!r}")
    if lo is not None and (v < lo or (lo_open and v <= lo)):
        raise ValueError(f"SimConfig.{name}: must be {'>' if lo_open else '>='} {lo}, got {v}")
    if hi is not None and (v > hi or (hi_open and v >= hi)):
        raise ValueError(f"SimConfig.{name}: must be {'<' if hi_open else '<='} {hi}, got {v}")
    return v


def _as_int(name: str, value: Any, *, lo: Optional[int] = None) -> int:
    """``value`` as an int (an integral float is accepted) >= lo, or ValueError."""
    if not _is_number(value) or float(value) != math.floor(float(value)):
        raise ValueError(f"SimConfig.{name}: expected an integer, got {value!r}")
    v = int(value)
    if lo is not None and v < lo:
        raise ValueError(f"SimConfig.{name}: must be >= {lo}, got {v}")
    return v


def _as_float_tuple(name: str, value: Any) -> Tuple[float, ...]:
    """``value`` (a sequence of numbers) as a tuple of finite floats."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"SimConfig.{name}: expected a sequence of numbers, got {value!r}")
    return tuple(_as_float(name, v) for v in list(value))


def _as_array1d(name: str, value: Any, *, positive: bool = False, allow_empty: bool = False) -> NDArray[np.float64]:
    """``value`` as a finite 1-d float array (a scalar becomes one element)."""
    if isinstance(value, (str, bytes)) or isinstance(value, dict):
        raise ValueError(f"SimConfig.{name}: expected numbers, got {value!r}")
    arr = np.array(value, dtype=np.float64).reshape(-1)
    if arr.size == 0 and not allow_empty:
        raise ValueError(f"SimConfig.{name}: must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"SimConfig.{name}: non-finite value")
    if positive and arr.size and not np.all(arr > 0.0):
        raise ValueError(f"SimConfig.{name}: every value must be positive")
    return arr


def _as_pick_polygon(value: Any) -> NDArray[np.float64]:
    """``SimConfig.pick_polygon_nm`` as a finite (P >= 3, 2) float array
    enclosing a non-zero area (a closing vertex equal to the first is
    allowed), or ValueError."""
    if isinstance(value, (str, bytes, dict)):
        raise ValueError(f"SimConfig.pick_polygon_nm: expected a polygon [[x, y], ...], got {value!r}")
    try:
        poly = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SimConfig.pick_polygon_nm: expected a polygon [[x, y], ...] ({exc})") from exc
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3 or not np.isfinite(poly).all():
        raise ValueError(f"SimConfig.pick_polygon_nm: expected a finite (P >= 3, 2) polygon, got shape {poly.shape}")
    if abs(_signed_area_nm2(poly)) <= 0.0:
        raise ValueError("SimConfig.pick_polygon_nm: the polygon encloses no area")
    return poly


def _inside_polygon(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], polygon_nm: NDArray[np.float64]) -> NDArray[np.bool_]:
    """True for the points (x, y) inside the closed polygon (even-odd
    rule, the last vertex joined to the first; a convex or concave simple
    polygon alike). Points exactly on an edge fall on either side."""
    px = np.asarray(x_nm, dtype=np.float64)
    py = np.asarray(y_nm, dtype=np.float64)
    poly = np.asarray(polygon_nm, dtype=np.float64)
    inside = np.zeros(px.shape, dtype=bool)
    xj, yj = float(poly[-1, 0]), float(poly[-1, 1])
    for xi, yi in poly.tolist():
        crosses = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_edge = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= crosses & (px < x_edge)
        xj, yj = xi, yi
    return inside


def _normalise_config(cfg: SimConfig) -> None:
    """Coerce and validate every field of ``cfg`` in place (``SimConfig.__post_init__``)."""
    if cfg.contour_nm is not None:
        c = np.array(cfg.contour_nm, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 3 or not np.isfinite(c).all():
            raise ValueError(f"SimConfig.contour_nm: expected a finite (K0 >= 3, 2) polygon, got shape {c.shape}")
        cfg.contour_nm = c
    axes = _as_float_tuple("ellipse_semi_axes_nm", cfg.ellipse_semi_axes_nm)
    if len(axes) != 2 or min(axes) <= 0.0:
        raise ValueError(f"SimConfig.ellipse_semi_axes_nm: two positive semi-axes expected, got {axes}")
    cfg.ellipse_semi_axes_nm = (axes[0], axes[1])
    fp = _as_float_tuple("fourier_perturbation", cfg.fourier_perturbation)
    if any(abs(a) >= 1.0 for a in fp):
        raise ValueError(f"SimConfig.fourier_perturbation: relative amplitudes must be < 1 in magnitude, got {fp}")
    cfg.fourier_perturbation = fp
    cfg.n_rings = _as_int("n_rings", cfg.n_rings, lo=2)
    cfg.period_nm = _as_float("period_nm", cfg.period_nm, lo=0.0, lo_open=True)
    cfg.sigma_period_nm = _as_float("sigma_period_nm", cfg.sigma_period_nm, lo=0.0)
    cfg.z_middle_lab_nm = _as_float("z_middle_lab_nm", cfg.z_middle_lab_nm)
    cfg.clusters_per_um = _as_float("clusters_per_um", cfg.clusters_per_um, lo=0.0)
    cfg.n_clusters_per_ring = None if cfg.n_clusters_per_ring is None else _as_int("n_clusters_per_ring", cfg.n_clusters_per_ring, lo=1)
    cfg.d_min_nm = _as_float("d_min_nm", cfg.d_min_nm, lo=0.0)
    cfg.radial_offset_sd_nm = _as_float("radial_offset_sd_nm", cfg.radial_offset_sd_nm, lo=0.0)
    if not isinstance(cfg.model, str) or cfg.model not in MODELS:
        raise ValueError(f"SimConfig.model: must be one of {list(MODELS)}, got {cfg.model!r}")
    cfg.f = _as_float("f", cfg.f, lo=0.0, hi=1.0)
    cfg.sigma_col_nm = _as_float("sigma_col_nm", cfg.sigma_col_nm, lo=0.0)
    cfg.q = _as_float("q", cfg.q, lo=0.0, hi=1.0, lo_open=True)
    cfg.alpha_deg_per_ring = _as_float("alpha_deg_per_ring", cfg.alpha_deg_per_ring)
    cfg.jitter_nm = _as_float("jitter_nm", cfg.jitter_nm, lo=0.0)
    cfg.m2_lattice_nm = _as_float("m2_lattice_nm", cfg.m2_lattice_nm, lo=0.0, lo_open=True)
    cfg.m2_p_occupied = _as_float("m2_p_occupied", cfg.m2_p_occupied, lo=0.0, hi=1.0)
    cfg.m2_h_inherit = _as_float("m2_h_inherit", cfg.m2_h_inherit, lo=0.0, hi=1.0)
    cfg.m7_z_half_width_nm = (None if cfg.m7_z_half_width_nm is None
                              else _as_float("m7_z_half_width_nm", cfg.m7_z_half_width_nm, lo=0.0, lo_open=True))
    cfg.n_fluor_mean = _as_float("n_fluor_mean", cfg.n_fluor_mean, lo=0.0)
    cfg.p_lab = _as_float("p_lab", cfg.p_lab, lo=0.0, hi=1.0)
    if cfg.n_locs_per_cluster_quantiles is not None:
        dec = _as_float_tuple("n_locs_per_cluster_quantiles", cfg.n_locs_per_cluster_quantiles)
        if len(dec) != DECILE_PROBABILITIES.size or any(b < a for a, b in zip(dec, dec[1:])) or dec[0] < 0.0:
            raise ValueError(f"SimConfig.n_locs_per_cluster_quantiles: nine non-decreasing non-negative deciles expected, got {dec}")
        cfg.n_locs_per_cluster_quantiles = dec
    cfg.epitope_radius_nm = _as_float("epitope_radius_nm", cfg.epitope_radius_nm, lo=0.0)
    cfg.sigma_link_nm = _as_float("sigma_link_nm", cfg.sigma_link_nm, lo=0.0)
    cfg.locs_per_fluor_mean = _as_float("locs_per_fluor_mean", cfg.locs_per_fluor_mean, lo=1.0)
    cfg.max_dark_frames = _as_int("max_dark_frames", cfg.max_dark_frames, lo=0)
    cfg.n_frames = _as_int("n_frames", cfg.n_frames, lo=1)
    cfg.lp_lateral_samples_nm = _as_array1d("lp_lateral_samples_nm", cfg.lp_lateral_samples_nm, positive=True)
    edges = _as_array1d("lpz_bin_edges_lab_nm", cfg.lpz_bin_edges_lab_nm)
    if edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError(f"SimConfig.lpz_bin_edges_lab_nm: at least two strictly increasing edges expected, got {edges.tolist()}")
    cfg.lpz_bin_edges_lab_nm = edges
    n_bins = edges.size - 1
    if isinstance(cfg.lpz_samples_by_bin_nm, (str, bytes)) or not isinstance(cfg.lpz_samples_by_bin_nm, (list, tuple)):
        raise ValueError("SimConfig.lpz_samples_by_bin_nm: expected one list of samples per stratum")
    samples = [_as_array1d("lpz_samples_by_bin_nm", v, positive=True, allow_empty=True) for v in cfg.lpz_samples_by_bin_nm]
    if len(samples) != n_bins:
        raise ValueError(f"SimConfig.lpz_samples_by_bin_nm: {len(samples)} strata for {n_bins} bins of lpz_bin_edges_lab_nm")
    if not any(v.size for v in samples):
        raise ValueError("SimConfig.lpz_samples_by_bin_nm: every stratum is empty")
    cfg.lpz_samples_by_bin_nm = samples
    scale = _as_array1d("axial_scale_by_bin", cfg.axial_scale_by_bin, positive=True)
    if scale.size != n_bins:
        raise ValueError(f"SimConfig.axial_scale_by_bin: {scale.size} values for {n_bins} strata")
    cfg.axial_scale_by_bin = scale
    cfg.sigma_struct_nm = _as_float("sigma_struct_nm", cfg.sigma_struct_nm, lo=0.0)
    cfg.background_per_um3 = _as_float("background_per_um3", cfg.background_per_um3, lo=0.0)
    cfg.background_margin_nm = _as_float("background_margin_nm", cfg.background_margin_nm, lo=0.0)
    cfg.tilt_deg = _as_float("tilt_deg", cfg.tilt_deg, lo=0.0, hi=90.0, hi_open=True)
    cfg.azimuth_deg = _as_float("azimuth_deg", cfg.azimuth_deg)
    cfg.pixel_size_nm = _as_float("pixel_size_nm", cfg.pixel_size_nm, lo=0.0, lo_open=True)
    for name in ("contour_name", "contour_library_npz", "library_dir"):
        v = getattr(cfg, name)
        if v is not None and not isinstance(v, str):
            raise ValueError(f"SimConfig.{name}: expected a string or None, got {v!r}")
    cfg.contour_smoothing_knot_nm = (None if cfg.contour_smoothing_knot_nm is None
                                     else _as_float("contour_smoothing_knot_nm", cfg.contour_smoothing_knot_nm, lo=0.0, lo_open=True))
    if cfg.length_basis is not None and (not isinstance(cfg.length_basis, str) or cfg.length_basis not in LENGTH_BASES):
        raise ValueError(f"SimConfig.length_basis: must be None or one of {list(LENGTH_BASES)}, got {cfg.length_basis!r}")
    if cfg.contour_smoothing_mode is not None and (not isinstance(cfg.contour_smoothing_mode, str)
                                                   or cfg.contour_smoothing_mode not in CONTOUR_SMOOTHING_MODES):
        raise ValueError(f"SimConfig.contour_smoothing_mode: must be None or one of {list(CONTOUR_SMOOTHING_MODES)}, "
                         f"got {cfg.contour_smoothing_mode!r}")
    if cfg.ring_rate_factors is not None:
        factors = _as_float_tuple("ring_rate_factors", cfg.ring_rate_factors)
        if len(factors) != cfg.n_rings or any(not (math.isfinite(v) and v > 0.0) for v in factors):
            raise ValueError(f"SimConfig.ring_rate_factors: one positive finite factor per ring ({cfg.n_rings}) expected, got {factors}")
        cfg.ring_rate_factors = factors
    if cfg.pick_polygon_nm is not None:
        cfg.pick_polygon_nm = _as_pick_polygon(cfg.pick_polygon_nm)
    if not isinstance(cfg.provenance, dict):
        raise ValueError(f"SimConfig.provenance: expected a dict, got {type(cfg.provenance).__name__}")


# ============================================================================
# Results
# ============================================================================

@dataclass
class SimClusterTable:
    """
    One row per TRUE cluster (clusters with 0 fluorophores included), in
    ring order: ``ring``, global ``cluster_id``, global ``column_id``
    (shared across rings by M5 / M7 copies, the cluster's own id
    otherwise), arc ``s_nm`` on the contour, true centre ``x_nm``,
    ``y_nm`` (axon frame) and ``z_nm`` (the ring's z), the signed
    ``r_offset_nm`` along the outward normal (D-25), ``n_fluor`` and
    ``n_locs`` counted from the localizations' truth labels. ``tangent``
    and ``normal_out`` are the unit membrane frame at the centre, along
    which the epitope disc and the linkage offset are laid;
    ``is_copy`` marks the clusters ``cluster_positions`` copied from the
    previous ring.
    """

    ring: NDArray[np.int64]
    cluster_id: NDArray[np.int64]
    column_id: NDArray[np.int64]
    s_nm: NDArray[np.float64]
    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    r_offset_nm: NDArray[np.float64]
    n_fluor: NDArray[np.int64]
    n_locs: NDArray[np.int64]
    tangent: NDArray[np.float64]            # (K, 2) unit tangent of the contour at s
    normal_out: NDArray[np.float64]         # (K, 2) unit outward normal at s
    is_copy: NDArray[np.bool_]

    @property
    def n_clusters(self) -> int:
        return int(self.ring.size)


@dataclass
class SimGeometry:
    """The membrane and the rings a simulated axon was built on: the
    contour used (closed polygon, axon frame) and its interpolating
    ``path``, the curve length, the outward sign of the tour, the ring
    z (axon frame), the laboratory axis and the rotation
    ``frame_rotation @ [0, 0, 1] == axis`` (laboratory = rotation @
    axon coordinates), and where the contour came from."""

    contour_used_nm: NDArray[np.float64]
    path: SmoothPath
    length_nm: float
    outward_sign: float
    ring_z_nm: NDArray[np.float64]
    axis: NDArray[np.float64]
    frame_rotation: NDArray[np.float64]
    contour_source: str


@dataclass
class SimLocalizations:
    """The localization table ``measure`` returns, in frame order, with
    the truth per localization and the per-cluster counts (aligned with
    the rows of the ``SimClusterTable`` it was measured from)."""

    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    frame: NDArray[np.int64]
    lp_lateral_nm: NDArray[np.float64]
    lpz_nm: NDArray[np.float64]
    ring_true: NDArray[np.int64]
    cluster_true: NDArray[np.int64]
    fluor_id: NDArray[np.int64]
    column_true: NDArray[np.int64]
    z_true_lab_nm: NDArray[np.float64]
    n_fluor_by_cluster: NDArray[np.int64]
    n_locs_by_cluster: NDArray[np.int64]
    n_background: int
    warnings: List[str] = field(default_factory=list)
    # H5-D: the localizations (background and cluster) dropped outside ``SimConfig.pick_polygon_nm`` (0 without a
    # pick); ``n_locs_by_cluster`` and ``n_background`` count the localizations KEPT.
    n_outside_pick: int = 0


@dataclass
class SimAxon:
    """
    What ``simulate_axon`` returns: the localizations in the LABORATORY
    frame and frame order, noisy (``x_nm``, ``y_nm``, ``z_nm``; what the
    pipeline receives) with ``frame``, ``lp_lateral_nm`` and the
    REPORTED ``lpz_nm`` (before the depth factor, exactly what a Picasso
    file carries); the truth per localization (``ring_true``,
    ``cluster_true``, ``fluor_id``, ``column_true``, -1 for the
    background; ``z_true_lab_nm`` the fluorophore's laboratory z with
    the structural width, before the axial error); the cluster table;
    the axis, the rotation (laboratory = ``frame_rotation`` @ axon), the
    contour used, its curve length, the ring z (axon frame), the
    configuration, the seed and the warnings.
    """

    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    frame: NDArray[np.int64]
    lp_lateral_nm: NDArray[np.float64]
    lpz_nm: NDArray[np.float64]
    ring_true: NDArray[np.int64]
    cluster_true: NDArray[np.int64]
    fluor_id: NDArray[np.int64]
    column_true: NDArray[np.int64]
    z_true_lab_nm: NDArray[np.float64]
    clusters: SimClusterTable
    axis: NDArray[np.float64]
    frame_rotation: NDArray[np.float64]
    contour_used_nm: NDArray[np.float64]
    length_nm: float
    ring_z_nm: NDArray[np.float64]
    config: SimConfig
    seed: int
    warnings: List[str] = field(default_factory=list)
    # H5-C: the length of the contour as RESOLVED (before ``contour_smoothing_knot_nm``; equal to ``length_nm``
    # without smoothing) and the lambda the ring counts were drawn with (``effective_clusters_per_um``: the
    # configuration's, rescaled by length_raw / length_used under ``length_basis`` "raw").
    length_raw_nm: float = float("nan")
    clusters_per_um_used: float = float("nan")
    # H5-D: the localizations ``measure`` dropped outside ``SimConfig.pick_polygon_nm`` (0 without a pick).
    n_outside_pick: int = 0

    @property
    def n_locs(self) -> int:
        return int(self.x_nm.size)


# ============================================================================
# Membrane geometry
# ============================================================================

def _signed_area_nm2(contour: NDArray[np.float64]) -> float:
    """Shoelace signed area of a closed polygon (positive counter-clockwise)."""
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _outward_sign(path: SmoothPath) -> float:
    """The ``outward_sign`` of ``project_on_path`` for this curve: -sign of
    the signed area of its table (interior on the left of a
    counter-clockwise tour)."""
    return -math.copysign(1.0, _signed_area_nm2(np.asarray(path.points_nm, dtype=np.float64)))


def _wrap_arc(s: NDArray[np.float64], length_nm: float) -> NDArray[np.float64]:
    """``s`` reduced to [0, L) with the rounding edge cases closed (``np.mod``
    of a tiny negative number can return exactly L)."""
    out = np.mod(np.asarray(s, dtype=np.float64), length_nm)
    out = np.where(out >= length_nm, out - length_nm, out)
    return np.asarray(np.where(out < 0.0, 0.0, out), dtype=np.float64)


def _frame_at_arc(path: SmoothPath, s_nm: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Point, unit tangent and unit OUTWARD normal of the tabulated curve
    at arcs ``s_nm`` (reduced modulo L): the sub-chord holding s is found
    by bisection in ``cum_nm``, the point interpolated on it, the tangent
    is the sub-chord's direction and the outward normal its left normal
    times ``_outward_sign`` -- the same frame ``project_on_path`` measures
    a signed offset in, so a point placed at C(s) + r n(s) projects back
    with offset r and arc s (to the sub-chord's turn, < 0.003 rad)."""
    pts = np.asarray(path.points_nm, dtype=np.float64)
    edges = np.asarray(path.edges_nm, dtype=np.float64)
    cum = np.asarray(path.cum_nm, dtype=np.float64)
    n = pts.shape[0]
    s = _wrap_arc(np.asarray(s_nm, dtype=np.float64).reshape(-1), float(path.length_nm))
    v = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, n - 1)
    span = np.where(edges[v] > 0.0, edges[v], 1.0)
    t = np.clip((s - cum[v]) / span, 0.0, 1.0)
    start = pts[v]
    chord = pts[(v + 1) % n] - start
    point = start + t[:, None] * chord
    norm_c = np.linalg.norm(chord, axis=1)
    # a zero-length sub-chord (coincident rows) borrows the next one's direction
    for _ in range(2):
        bad = norm_c <= 0.0
        if not bad.any():
            break
        v = np.where(bad, (v + 1) % n, v)
        chord = np.where(bad[:, None], pts[(v + 1) % n] - pts[v], chord)
        norm_c = np.linalg.norm(chord, axis=1)
    tangent = chord / np.where(norm_c > 0.0, norm_c, 1.0)[:, None]
    left = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal_out = left * _outward_sign(path)
    return np.asarray(point, dtype=np.float64), np.asarray(tangent, dtype=np.float64), np.asarray(normal_out, dtype=np.float64)


def membrane_point(
    contour_path: SmoothPath,
    s_nm: NDArray[np.float64],
    r_nm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    (n, 2) points at arc ``s_nm`` of the tabulated contour curve,
    displaced ``r_nm`` along the OUTWARD normal (+ outward, the sign
    convention of ``tools.mps_unroll.project_on_path``, so that
    ``project_on_path(path, membrane_point(path, s, r))`` returns (s, r)
    to the sub-chord's turn). ``contour_path`` is
    ``smooth_closed_path(contour)``: the contour is already smooth (a
    measured ring's smoothed contour or an ellipse), so it is
    INTERPOLATED (splprep s = 0), not smoothed again. ``r_nm`` broadcasts
    against ``s_nm``.
    """
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    r = np.broadcast_to(np.asarray(r_nm, dtype=np.float64), s.shape).astype(np.float64)
    point, _tangent, normal_out = _frame_at_arc(contour_path, s)
    return np.asarray(point + r[:, None] * normal_out, dtype=np.float64)


def _rotation_z_to(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rodrigues rotation taking the z axis onto the unit vector ``u``
    (the identity when u is z to 1e-12): laboratory = R @ axon."""
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    k = np.cross(z, u)
    s = float(np.linalg.norm(k))
    c = float(np.dot(z, u))
    if s < 1e-12:
        return np.eye(3)
    k = k / s
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.asarray(np.eye(3) + s * kx + (1.0 - c) * (kx @ kx), dtype=np.float64)


def _ellipse_contour(cfg: SimConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """The ellipse ``ellipse_semi_axes_nm`` sampled at ``ELLIPSE_VERTICES``
    uniform parametric angles, with the relative radial harmonics
    ``fourier_perturbation`` (m = 2, 3, ...; a phase per harmonic drawn
    uniformly from ``rng``, only when a perturbation is given, so that
    an unperturbed ellipse consumes no draw): (a cos t, b sin t) x
    (1 + sum_m a_m cos(m t + phi_m)). On a circle base t is the polar
    angle and r(theta) / R - 1 has exactly these harmonics."""
    a, b = cfg.ellipse_semi_axes_nm
    t = 2.0 * np.pi * np.arange(ELLIPSE_VERTICES, dtype=np.float64) / ELLIPSE_VERTICES
    rel = np.ones_like(t)
    if cfg.fourier_perturbation:
        phases = rng.uniform(0.0, 2.0 * np.pi, len(cfg.fourier_perturbation))
        for m, (amp, phi) in enumerate(zip(cfg.fourier_perturbation, phases), start=2):
            rel = rel + amp * np.cos(m * t + phi)
    return np.column_stack([a * np.cos(t) * rel, b * np.sin(t) * rel]).astype(np.float64)


def contour_library_path(cfg: SimConfig) -> Optional[str]:
    """
    Where ``cfg.contour_library_npz`` is: None without a library; an
    absolute path as given; a relative one resolved next to the YAML
    the configuration was loaded from (``load_sim_config`` records the
    folder in the transient field ``library_dir``, which a copy made
    with ``dataclasses.replace`` or a pickled instance keeps -- a
    process-wide registry keyed by the relative name sent a replaced
    copy to whichever YAML with that library name was loaded last;
    re-review of 2026-09-25) when the file exists there, else relative
    to the working directory. The field itself is never rewritten (the
    YAML round trip keeps the path as written).
    """
    name = cfg.contour_library_npz
    if name is None:
        return None
    if os.path.isabs(name):
        return name
    if cfg.library_dir is not None:
        candidate = os.path.join(cfg.library_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return os.path.abspath(name)


def _resolve_contour(cfg: SimConfig, rng: np.random.Generator, warnings_: List[str]) -> Tuple[NDArray[np.float64], str]:
    """The contour ``simulate_axon`` uses and its source: ``cfg.contour_nm``
    when given; else the entry ``contour_name`` of the NPZ library; else
    the (perturbed) ellipse. A NAMED contour that cannot be served --
    a name without a library, a library file that does not exist, a
    name missing from it -- raises ValueError: a misspelt name must not
    run silently on the ellipse under that name (review of 2026-09-25).
    A library without a name falls back to the ellipse with a warning
    (nothing was named)."""
    if cfg.contour_nm is not None:
        return np.array(cfg.contour_nm, dtype=np.float64), "config"
    lib = contour_library_path(cfg)
    if cfg.contour_name is not None:
        if lib is None:
            raise ValueError(f"contour_name {cfg.contour_name!r} given without a contour_library_npz to look it up in")
        if not os.path.isfile(lib):
            raise ValueError(f"contour library {lib!r} not found (contour_library_npz={cfg.contour_library_npz!r})")
        with np.load(lib) as store:
            if cfg.contour_name not in store.files:
                raise ValueError(f"contour {cfg.contour_name!r} not in library {lib!r} (entries: {store.files})")
            contour = np.array(store[cfg.contour_name], dtype=np.float64)
        if contour.ndim != 2 or contour.shape[1] != 2 or contour.shape[0] < 3:
            raise ValueError(f"library contour {cfg.contour_name!r} has shape {contour.shape}, expected (K >= 3, 2)")
        return contour, f"library:{cfg.contour_name}"
    if cfg.contour_library_npz is not None:
        warnings_.append("contour_library_npz given without a contour_name: the ellipse is used")
    return _ellipse_contour(cfg, rng), "ellipse"


def _closed_chord_length_nm(contour: NDArray[np.float64]) -> float:
    """The chord length of a closed polygon (the last vertex joined to
    the first): the "raw" length of ``SimConfig.length_basis``, the
    length ``smooth_polygon_membrane`` places its knots on."""
    c = np.asarray(contour, dtype=np.float64)
    return float(np.hypot(*(np.roll(c, -1, axis=0) - c).T).sum())


def _contour_as_used(cfg: SimConfig, contour: NDArray[np.float64], source: str) -> Tuple[NDArray[np.float64], str, float]:
    """The contour ``simulate_axon`` places the clusters on, its source
    label and the RAW length: the resolved ``contour`` as is (raw length
    = its chord length), or, with ``cfg.contour_smoothing_knot_nm``,
    its P-spline membrane at that knot spacing (H5-B fix:
    ``SimConfig.contour_smoothing_knot_nm``; the source suffixed
    "+pspline<knot>nm"), or, with ``contour_smoothing_mode`` "membrane"
    (H5-C), ``membrane_scale_contour`` at that spacing (the source
    suffixed "+membrane<knot>nm")."""
    length_raw = _closed_chord_length_nm(contour)
    if cfg.contour_smoothing_knot_nm is not None:
        knot = float(cfg.contour_smoothing_knot_nm)
        if cfg.contour_smoothing_mode == "membrane":
            contour = membrane_scale_contour(contour, knot_spacing_nm=knot)
            source = f"{source}+membrane{knot:g}nm"
        else:
            contour = smooth_polygon_membrane(contour, knot_spacing_nm=knot)
            source = f"{source}+pspline{knot:g}nm"
    return contour, source, length_raw


def membrane_scale_contour(contour_nm: NDArray[np.float64], *, knot_spacing_nm: float = MEMBRANE_KNOT_SPACING_NM,
                           sample_step_nm: float = CONTOUR_MEMBRANE_SAMPLE_STEP_NM,
                           initial_knot_spacing_nm: float = MEMBRANE_NULL_KNOT_SPACING_NM) -> NDArray[np.float64]:
    """
    A closed contour made consistent with the basis of the arc test's
    localization membrane (H5-C, ``CONTOUR_SMOOTHING_MODES`` "membrane";
    the comment there has the measurement that motivates it): the curve
    ``tools.mps_membrane.fit_membrane_from_localizations`` returns when
    its "localizations" are noise-free samples of the contour.

    Steps: (1) the contour's interpolating curve (``smooth_closed_path``,
    what ``simulate_axon`` would place the clusters on) sampled every
    ``sample_step_nm`` of its arc; (2) the initial curve is the
    ``smooth_polygon_membrane`` of the contour at ``initial_knot_spacing_
    nm`` (400 nm = ``MEMBRANE_NULL_KNOT_SPACING_NM``: the spacing of the
    pooled centroid curve ``localization_membrane_of_rings`` starts from,
    so that the knot count and the first parameterisation are the
    estimator's); (3) ``fit_membrane_from_localizations`` with knots
    every ``knot_spacing_nm`` of the INITIAL curve's length -- that is,
    of the smooth curve's own arc, not of the rough polygon's chord --,
    its default two iterations (the first at the finer parameterisation
    knots) and the robust drop off (``CONTOUR_MEMBRANE_ROBUST_C``: every
    sample is signal); (4) the fitted curve resampled at equal arcs, one
    vertex per ``CONTOUR_MEMBRANE_VERTEX_SPACING_NM`` (at least
    ``MEASURED_CONTOUR_VERTICES``), starting at the fit's parameter 0.
    The result is in the span of the basis the arc test estimates the
    membrane with (up to the re-parameterisation of each fit on its own
    initial curve), so a simulated axon built on it carries no shared
    radial structure below the knot spacing. Deterministic (no random
    draw). ValueError from the fit when the contour is degenerate.
    """
    c = np.asarray(contour_nm, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 4 or not np.isfinite(c).all():
        raise ValueError(f"membrane_scale_contour: expected a finite (K0 >= 4, 2) polygon, got shape {c.shape}")
    knot = float(knot_spacing_nm)
    step = float(sample_step_nm)
    if not (math.isfinite(knot) and knot > 0.0 and math.isfinite(step) and step > 0.0):
        raise ValueError(f"membrane_scale_contour: knot_spacing_nm and sample_step_nm must be positive, got {knot_spacing_nm}, {sample_step_nm}")
    path_0 = smooth_closed_path(c)
    n_samples = max(int(math.ceil(float(path_0.length_nm) / step)), 4 * int(MEASURED_CONTOUR_VERTICES))
    samples = _resample_closed_path(path_0, n_samples)
    initial = smooth_closed_path(smooth_polygon_membrane(c, knot_spacing_nm=float(initial_knot_spacing_nm)))
    fit = fit_membrane_from_localizations(samples[:, 0], samples[:, 1], initial_path=initial, knot_spacing_nm=knot,
                                          robust_c=CONTOUR_MEMBRANE_ROBUST_C)
    n_out = max(int(MEASURED_CONTOUR_VERTICES), int(math.ceil(float(fit.path.length_nm) / float(CONTOUR_MEMBRANE_VERTEX_SPACING_NM))))
    return _resample_closed_path(fit.path, n_out)


def effective_clusters_per_um(cfg: SimConfig, length_raw_nm: float, length_used_nm: float) -> float:
    """
    The lambda the ring counts are drawn with (H5-C, ``LENGTH_BASES``):
    ``cfg.clusters_per_um`` as is, unless the configuration's lambda was
    calibrated on the RAW contour length (``length_basis`` "raw") and the
    contour is smoothed (``contour_smoothing_knot_nm`` set), in which
    case it is rescaled by ``length_raw_nm`` / ``length_used_nm`` so that
    the expected K per ring, lambda x L_used, equals lambda_raw x L_raw
    -- the count the axon was measured with (D-28h: a lambda applied to
    a contour shorter than the one it was measured on simulated too few
    clusters and lost the deep ring). With ``length_basis`` None or
    "smoothed", or without smoothing, nothing changes (today's rule).
    """
    lam = float(cfg.clusters_per_um)
    if (cfg.length_basis == "raw" and cfg.contour_smoothing_knot_nm is not None
            and length_raw_nm > 0.0 and length_used_nm > 0.0 and math.isfinite(length_raw_nm) and math.isfinite(length_used_nm)):
        return lam * float(length_raw_nm) / float(length_used_nm)
    return lam


def _ring_config(cfg: SimConfig, ring: int) -> SimConfig:
    """The configuration ring ``ring`` draws its clusters with under
    ``SimConfig.ring_rate_factors`` (H5-C): ``clusters_per_um`` times the
    ring's factor and, when ``n_clusters_per_ring`` fixes the count,
    that count times the factor, rounded (at least 1)."""
    assert cfg.ring_rate_factors is not None
    factor = float(cfg.ring_rate_factors[int(ring)])
    n_fixed = None if cfg.n_clusters_per_ring is None else max(1, int(round(float(cfg.n_clusters_per_ring) * factor)))
    return dataclasses.replace(cfg, clusters_per_um=float(cfg.clusters_per_um) * factor, n_clusters_per_ring=n_fixed)


def _resample_closed_path(path: SmoothPath, n_vertices: int) -> NDArray[np.float64]:
    """``n_vertices`` points of a tabulated closed curve at equally spaced
    arcs (linear interpolation on the dense table, whose chord is
    ~``tools.mps_matching.PATH_STEP_NM``): the polygon a measured
    localization membrane is kept as (``MEASURED_CONTOUR_VERTICES``)."""
    pts = np.asarray(path.points_nm, dtype=np.float64)
    cum = np.asarray(path.cum_nm, dtype=np.float64)
    closed = np.vstack([pts, pts[:1]])
    arcs = np.linspace(0.0, float(path.length_nm), int(n_vertices), endpoint=False)
    return np.column_stack([np.interp(arcs, cum, closed[:, 0]), np.interp(arcs, cum, closed[:, 1])]).astype(np.float64)


# ============================================================================
# Cluster positions (the generative models)
# ============================================================================

def _cyclic_distance(d: NDArray[np.float64], length_nm: float) -> NDArray[np.float64]:
    """Cyclic distance of arc differences on a circle of length L."""
    m = np.mod(np.asarray(d, dtype=np.float64), length_nm)
    return np.asarray(np.minimum(m, length_nm - m), dtype=np.float64)


def _hard_core_arcs(rng: np.random.Generator, n: int, length_nm: float, d_min_nm: float,
                    fixed: NDArray[np.float64], *, max_attempts: int = HARD_CORE_MAX_ATTEMPTS) -> NDArray[np.float64]:
    """``n`` arc positions uniform on [0, L), each redrawn until it lies
    >= ``d_min_nm`` (cyclic) from every position already placed,
    ``fixed`` included: the sequential rejection sampler of a hard-core
    process, in placement order. RuntimeError when a position cannot be
    placed in ``max_attempts`` draws (the ring is jammed)."""
    placed = np.asarray(fixed, dtype=np.float64).reshape(-1)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        for _attempt in range(int(max_attempts)):
            s = float(rng.uniform(0.0, length_nm))
            if placed.size == 0 or d_min_nm <= 0.0 or float(_cyclic_distance(placed - s, length_nm).min()) >= d_min_nm:
                break
        else:
            raise RuntimeError(f"hard core of {d_min_nm} nm jammed after {placed.size} of {n + int(np.asarray(fixed).size)} "
                               f"clusters on {length_nm:.0f} nm")
        placed = np.append(placed, s)
        out[i] = s
    return out


def _ring_count(cfg: SimConfig, rng: np.random.Generator, length_nm: float) -> int:
    """K of a fresh ring: ``n_clusters_per_ring`` when given, else a Poisson
    draw of ``clusters_per_um`` x L / 1000."""
    if cfg.n_clusters_per_ring is not None:
        return int(cfg.n_clusters_per_ring)
    return int(rng.poisson(cfg.clusters_per_um * length_nm / 1000.0))


def cluster_positions(
    cfg: SimConfig,
    rng: np.random.Generator,
    length_nm: float,
    prev_s_nm: Optional[NDArray[np.float64]],
    prev_columns: Optional[NDArray[np.int64]],
    *,
    prev_is_member: Optional[NDArray[np.bool_]] = None,
    next_id: Optional[int] = None,
    m2_origin_nm: Optional[float] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_]]:
    """
    Arc positions, column ids and copy flags of ONE ring given the
    previous one (``prev_* = None`` for the first ring, which every model
    draws as M1), per ``cfg.model``:

    * M1: K fresh positions under the hard core ``d_min_nm``.
    * M5 (the joint rule of f and q, which the specification leaves
      open): the ring holds n_copy = round(f K) copies. Every COLUMN
      MEMBER of the previous ring (``prev_is_member``: a cluster that
      was itself a copy) continues with probability q, its copy keeping
      the column id at the parent's arc + N(0, ``sigma_col_nm``) along
      the arc; the remaining copies start NEW columns from the
      non-continuing clusters of the previous ring, fresh ones first
      (the copy takes the parent's id, which becomes a column id) and,
      when they run out, the members that died (the copy gets a fresh
      id: a column never revives). The rest of the ring is fresh under
      the hard core against the copies (copies among themselves may sit
      closer than d_min by their jitter). With f = 1, q = 1 every column
      is global; a column that starts at ring 0 and reaches ring 1 has a
      geometric length with continuation q.
    * M6: the midpoints of the previous ring's cyclic gaps + N(0,
      ``jitter_nm``) along the arc (K unchanged).
    * M3b: the previous arcs + alpha / 360 L + N(0, ``jitter_nm``).
    * M2 (01 S1.3 "reticulo canonico s_i = i a + jitter, ocupacion p,
      huecos heredados con probabilidad h"): a lattice of spacing L /
      round(L / ``m2_lattice_nm``) (closing exactly) with origin
      ``m2_origin_nm`` (drawn uniformly in [0, d) when None; the caller
      passes one origin to every ring). The first ring occupies each
      site with probability ``m2_p_occupied``; in the next ring every
      site KEEPS its previous state (occupied or hole) with probability
      ``m2_h_inherit`` and is redrawn with probability p otherwise, so
      that h = 1 gives identical holes ring after ring, h = 0
      independent rings, and the occupancy stays p on every ring
      (h p + (1 - h) p = p; the previous rule redrew every occupied site
      and emptied the lattice ring by ring, 0.8 -> 0.64 -> 0.51 at h = 1,
      review of 2026-09-25); a ring-k hole is a hole in k+1 with
      probability h + (1 - h)(1 - p). Each occupied cluster then sits at
      its site + N(0, ``jitter_nm``) along the arc (the jitter of 01;
      the previous ring's sites are recovered by rounding, so the
      jitter must stay well below d / 2).
    * M7: M5 with f = 1, q = 1 and K fixed to the previous ring's (the
      axial rule is in ``measure``).

    Column ids are global integers: copies share their parent's, every
    other cluster gets a fresh id above ``next_id`` (default: above every
    id of ``prev_columns``, or 0 for the first ring). M6, M3b and M2
    clusters keep their own ids (the specification's "else the cluster's
    own id"). Returns ``(s_nm, column_ids, is_copy)`` of equal length,
    s in [0, L).
    """
    length = float(length_nm)
    if not (math.isfinite(length) and length > 0.0):
        raise ValueError(f"cluster_positions: length_nm must be positive, got {length_nm}")
    prev_s = None if prev_s_nm is None else np.asarray(prev_s_nm, dtype=np.float64).reshape(-1)
    prev_cols = None if prev_columns is None else np.asarray(prev_columns, dtype=np.int64).reshape(-1)
    if (prev_s is None) != (prev_cols is None) or (prev_s is not None and prev_cols is not None and prev_s.size != prev_cols.size):
        raise ValueError("cluster_positions: prev_s_nm and prev_columns must both be given, of one length, or both None")
    start_id = int(next_id) if next_id is not None else (0 if prev_cols is None or prev_cols.size == 0 else int(prev_cols.max()) + 1)
    model = cfg.model
    first = prev_s is None or prev_cols is None

    if model == "M2":
        n_lat = max(1, int(round(length / cfg.m2_lattice_nm)))
        d = length / n_lat
        origin = float(rng.uniform(0.0, d)) if m2_origin_nm is None else float(np.mod(m2_origin_nm, d))
        sites = origin + d * np.arange(n_lat, dtype=np.float64)
        u = rng.uniform(size=n_lat)
        if first or prev_s is None:
            occupied = u < cfg.m2_p_occupied
        else:
            j = np.mod(np.round((prev_s - origin) / d).astype(np.int64), n_lat)
            prev_occ = np.zeros(n_lat, dtype=bool)
            prev_occ[j] = True
            keep_state = rng.uniform(size=n_lat) < cfg.m2_h_inherit
            occupied = np.where(keep_state, prev_occ, u < cfg.m2_p_occupied)
        k = int(np.count_nonzero(occupied))
        s = _wrap_arc(sites[occupied] + rng.normal(0.0, cfg.jitter_nm, k), length)
        return s, start_id + np.arange(k, dtype=np.int64), np.zeros(k, dtype=bool)

    if first or prev_s is None or prev_cols is None:
        k = _ring_count(cfg, rng, length)
        s = _hard_core_arcs(rng, k, length, cfg.d_min_nm, np.zeros(0))
        return s, start_id + np.arange(k, dtype=np.int64), np.zeros(k, dtype=bool)

    k_prev = int(prev_s.size)
    if model == "M1":
        k = _ring_count(cfg, rng, length)
        s = _hard_core_arcs(rng, k, length, cfg.d_min_nm, np.zeros(0))
        return s, start_id + np.arange(k, dtype=np.int64), np.zeros(k, dtype=bool)
    if model == "M6":
        a = np.sort(prev_s)
        gap = np.diff(np.concatenate([a, [a[0] + length]])) if k_prev else np.zeros(0)
        s = _wrap_arc(a + gap / 2.0 + rng.normal(0.0, cfg.jitter_nm, k_prev), length)
        return s, start_id + np.arange(k_prev, dtype=np.int64), np.zeros(k_prev, dtype=bool)
    if model == "M3b":
        s = _wrap_arc(prev_s + cfg.alpha_deg_per_ring / 360.0 * length + rng.normal(0.0, cfg.jitter_nm, k_prev), length)
        return s, start_id + np.arange(k_prev, dtype=np.int64), np.zeros(k_prev, dtype=bool)
    if model not in ("M5", "M7"):
        raise ValueError(f"cluster_positions: unknown model {model!r}")

    # M5 / M7
    f_copy, q_cont = (1.0, 1.0) if model == "M7" else (cfg.f, cfg.q)
    k_next = k_prev if model == "M7" else _ring_count(cfg, rng, length)
    member = np.zeros(k_prev, dtype=bool) if prev_is_member is None else np.asarray(prev_is_member, dtype=bool).reshape(-1)
    if member.size != k_prev:
        raise ValueError(f"cluster_positions: prev_is_member has {member.size} entries for {k_prev} clusters")
    cont = member & (rng.uniform(size=k_prev) < q_cont)
    idx_cont = np.flatnonzero(cont)
    if idx_cont.size > k_next:
        idx_cont = np.sort(rng.choice(idx_cont, k_next, replace=False))
    n_copy = int(round(f_copy * k_next))
    n_new = max(0, min(n_copy - int(idx_cont.size), k_next - int(idx_cont.size)))
    fresh_prev = np.flatnonzero(~member)
    dead_prev = np.flatnonzero(member & ~cont)
    candidates = np.concatenate([rng.permutation(fresh_prev), rng.permutation(dead_prev)]).astype(np.int64)
    idx_new = candidates[:n_new]
    parents = np.concatenate([idx_cont, idx_new]).astype(np.int64)
    n_cp = int(parents.size)
    s_copy = _wrap_arc(prev_s[parents] + rng.normal(0.0, cfg.sigma_col_nm, n_cp), length)
    ids_copy = prev_cols[parents].copy()
    next_free = start_id
    for pos, p in enumerate(idx_new.tolist(), start=int(idx_cont.size)):
        if member[p]:                       # a dead member: its copy starts a NEW column
            ids_copy[pos] = next_free
            next_free += 1
    n_fresh = k_next - n_cp
    s_fresh = _hard_core_arcs(rng, n_fresh, length, cfg.d_min_nm, s_copy)
    ids_fresh = next_free + np.arange(n_fresh, dtype=np.int64)
    s = np.concatenate([s_copy, s_fresh]).astype(np.float64)
    ids = np.concatenate([ids_copy, ids_fresh]).astype(np.int64)
    is_copy = np.concatenate([np.ones(n_cp, dtype=bool), np.zeros(n_fresh, dtype=bool)])
    return s, ids, is_copy


# ============================================================================
# The measurement stage (02 B8)
# ============================================================================

def _lpz_borrow_map(samples: Sequence[NDArray[np.float64]]) -> NDArray[np.int64]:
    """For each stratum the index of the nearest stratum with samples
    (itself when it has any): an empty stratum borrows its neighbour."""
    sizes = np.array([int(np.asarray(v).size) for v in samples])
    nonempty = np.flatnonzero(sizes > 0)
    if nonempty.size == 0:
        raise ValueError("no lpz samples in any stratum")
    idx = np.arange(sizes.size)
    return np.asarray(nonempty[np.argmin(np.abs(idx[:, None] - nonempty[None, :]), axis=1)], dtype=np.int64)


def _stratum_of(z_lab_nm: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.int64]:
    """Index of the stratum holding each laboratory z; outside the edges
    the nearest (first or last) stratum."""
    return np.asarray(np.clip(np.digitize(z_lab_nm, edges) - 1, 0, edges.size - 2), dtype=np.int64)


def _draw_lpz(rng: np.random.Generator, stratum: NDArray[np.int64], samples: Sequence[NDArray[np.float64]],
              borrow: NDArray[np.int64]) -> NDArray[np.float64]:
    """Reported lpz per localization, resampled with replacement from its
    stratum's samples, stratum by stratum in index order (a fixed order
    of draws)."""
    lpz = np.empty(stratum.size, dtype=np.float64)
    for s in range(int(borrow.size)):
        sel = np.flatnonzero(stratum == s)
        if sel.size:
            src = np.asarray(samples[int(borrow[s])], dtype=np.float64)
            lpz[sel] = src[rng.integers(0, src.size, sel.size)]
    return lpz


def _fluorophores_per_cluster(cfg: SimConfig, rng: np.random.Generator, k: int) -> NDArray[np.int64]:
    """n_fluor of ``k`` clusters: Poisson(``n_fluor_mean``), or, with the
    deciles of n_locs per cluster given, round(n_locs / locs_per_fluor)
    with n_locs drawn from the piecewise-linear quantile function
    through the deciles (flat below the first and above the last)."""
    if cfg.n_locs_per_cluster_quantiles is None:
        return np.asarray(rng.poisson(cfg.n_fluor_mean, k), dtype=np.int64)
    u = rng.uniform(size=k)
    n_locs = np.interp(u, DECILE_PROBABILITIES, np.asarray(cfg.n_locs_per_cluster_quantiles, dtype=np.float64))
    return np.asarray(np.maximum(np.round(n_locs / cfg.locs_per_fluor_mean), 0.0), dtype=np.int64)


def _m7_column_map(clusters: SimClusterTable, n_rings: int) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(column index of every table row, (n_columns, n_rings) table row of
    each column in each ring) for M7, where every column has exactly
    one cluster per ring; ValueError otherwise."""
    cols, col_index = np.unique(clusters.column_id, return_inverse=True)
    table = np.full((cols.size, n_rings), -1, dtype=np.int64)
    table[col_index, clusters.ring] = np.arange(clusters.n_clusters)
    if np.any(table < 0):
        raise ValueError("M7: every column must have one cluster in every ring")
    return np.asarray(col_index, dtype=np.int64), table


def _geometry_for_table(cfg: SimConfig, rng: np.random.Generator, clusters: SimClusterTable) -> SimGeometry:
    """The ``SimGeometry`` ``measure`` needs when called without one (the
    specification's three-argument form): the contour resolved from
    the configuration (the Fourier phases are drawn from ``rng`` when a
    perturbation is requested), the ring z read off the table (each
    ring's ``z_nm``; the configuration's nominal z for a ring without a
    cluster) and the laboratory frame of ``tilt_deg`` / ``azimuth_deg``."""
    warn: List[str] = []
    contour, source = _resolve_contour(cfg, rng, warn)
    path = smooth_closed_path(contour)
    n_rings = int(cfg.n_rings)
    ring_z = cfg.z_middle_lab_nm + (np.arange(n_rings, dtype=np.float64) - 0.5 * (n_rings - 1)) * cfg.period_nm
    for k in range(n_rings):
        zs = np.asarray(clusters.z_nm, dtype=np.float64)[np.asarray(clusters.ring) == k]
        if zs.size:
            ring_z[k] = float(zs[0])
    b, ph = math.radians(cfg.tilt_deg), math.radians(cfg.azimuth_deg)
    axis = np.array([math.sin(b) * math.cos(ph), math.sin(b) * math.sin(ph), math.cos(b)], dtype=np.float64)
    rotation = _rotation_z_to(axis) if cfg.tilt_deg > 0.0 else np.eye(3)
    return SimGeometry(contour_used_nm=contour, path=path, length_nm=float(path.length_nm), outward_sign=_outward_sign(path),
                       ring_z_nm=np.asarray(ring_z, dtype=np.float64), axis=axis, frame_rotation=rotation, contour_source=source)


def measure(
    cfg: SimConfig,
    rng: np.random.Generator,
    clusters: SimClusterTable,
    geometry: Optional[SimGeometry] = None,
) -> SimLocalizations:
    """
    The common measurement stage of 02 B8 on a cluster table
    (``geometry`` None rebuilds it from the configuration and the
    table, ``_geometry_for_table``, so that the specification's
    ``measure(cfg, rng, clusters)`` form is valid; ``simulate_axon``
    passes the geometry it built): ring by ring, in this order of
    draws, (1) fluorophores per cluster; (2) the
    epitope disc, uniform in the disc of ``epitope_radius_nm`` in the
    membrane plane (radius R sqrt(u), angle 2 pi u: tangent and axial
    components); (3) the linkage offset N(0, ``sigma_link_nm``) along the
    tangent, the outward normal and z, once per fluorophore; (4) the
    structural width N(0, ``sigma_struct_nm``) in z; for M7 the
    fluorophore's z is uniform over the column's axial span (centre of
    the rings +/- ``m7_z_half_width_nm``) instead of its ring's z, and
    its ring / cluster of truth is the nearest ring's cluster of the
    column (the rings cut the column); (5) localizations per fluorophore
    k ~ Geometric(1 / ``locs_per_fluor_mean``) on {1, 2, ...}, frame gaps
    1 + Uniform{0..``max_dark_frames``} and a uniform start in [0,
    ``n_frames`` - span]; (6) lp per localization resampled from
    ``lp_lateral_samples_nm``; (7) the true position rotated into the
    laboratory frame (``geometry.frame_rotation``) and the lateral error
    N(0, lp) per laboratory axis; (8) the stratum of the TRUE laboratory
    z, the reported lpz resampled from its samples and the axial error
    N(0, lpz x ``axial_scale_by_bin``[stratum]) (D-19: the file's lpz is
    optimistic where the scale is above 1). Then the background:
    Poisson(``background_per_um3`` x V) localizations uniform in the
    bounding box of the contour widened by ``background_margin_nm`` on
    each of its four sides (the pick's margin: tangent to the membrane
    the box made the pipeline's edge criterion cut the same arcs of
    every ring, ``BACKGROUND_MARGIN_DEFAULT_NM``) times [min ring z -
    P/2, max ring z + P/2] (axon frame, rotated to the laboratory), with
    lp and lpz from the tables, uniform frames and -1 ids; the uniform
    box IS their observed distribution (no error is added on top). With
    ``cfg.pick_polygon_nm`` (H5-D) the box grows to cover the polygon's
    bounding box when it does not already, the draws are the same
    otherwise, and every localization whose MEASURED position, rotated
    back to the axon frame, lies outside the polygon is dropped (the
    counts then refer to the localizations kept; ``n_outside_pick``).
    The table is returned in frame order (stable sort).
    """
    if geometry is None:
        geometry = _geometry_for_table(cfg, rng, clusters)
    n_rings = int(cfg.n_rings)
    ring_z = np.asarray(geometry.ring_z_nm, dtype=np.float64)
    rot = np.asarray(geometry.frame_rotation, dtype=np.float64)
    edges = np.asarray(cfg.lpz_bin_edges_lab_nm, dtype=np.float64)
    samples = cfg.lpz_samples_by_bin_nm
    scale = np.asarray(cfg.axial_scale_by_bin, dtype=np.float64)
    borrow = _lpz_borrow_map(samples)
    lp_samples = np.asarray(cfg.lp_lateral_samples_nm, dtype=np.float64)
    warnings_: List[str] = []
    n_rows = clusters.n_clusters
    is_m7 = cfg.model == "M7"
    if is_m7:
        col_index, col_table = _m7_column_map(clusters, n_rings)
        z_centre = float(ring_z.mean())
        hw = cfg.m7_z_half_width_nm if cfg.m7_z_half_width_nm is not None else 0.5 * float(ring_z.max() - ring_z.min()) + 0.5 * cfg.period_nm
    else:
        col_index, col_table = np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.int64)
        z_centre, hw = 0.0, 0.0
    n_fluor_by_cluster = np.zeros(n_rows, dtype=np.int64)
    n_locs_by_cluster = np.zeros(n_rows, dtype=np.int64)
    parts: Dict[str, List[NDArray[Any]]] = {k: [] for k in ("x", "y", "z", "frame", "lp", "lpz", "ring", "cluster", "fluor", "column", "z_true",
                                                            "row")}
    fluor_offset = 0
    p_blink = 1.0 / cfg.locs_per_fluor_mean
    for k in range(n_rings):
        rows = np.flatnonzero(clusters.ring == k)
        n_fl = _fluorophores_per_cluster(cfg, rng, int(rows.size))
        n_f = int(n_fl.sum())
        fl_row = np.repeat(rows, n_fl)
        u1 = rng.uniform(size=n_f)
        u2 = rng.uniform(size=n_f)
        rho = cfg.epitope_radius_nm * np.sqrt(u1)
        phi = 2.0 * np.pi * u2
        d_t, d_z = rho * np.cos(phi), rho * np.sin(phi)
        link = rng.normal(0.0, cfg.sigma_link_nm, (n_f, 3))
        struct = rng.normal(0.0, cfg.sigma_struct_nm, n_f)
        tang = clusters.tangent[fl_row]
        norm_out = clusters.normal_out[fl_row]
        cx = clusters.x_nm[fl_row] + (d_t + link[:, 0]) * tang[:, 0] + link[:, 1] * norm_out[:, 0]
        cy = clusters.y_nm[fl_row] + (d_t + link[:, 0]) * tang[:, 1] + link[:, 1] * norm_out[:, 1]
        if is_m7:
            zc = rng.uniform(z_centre - hw, z_centre + hw, n_f)
        else:
            zc = np.full(n_f, ring_z[k], dtype=np.float64)
        zf = zc + d_z + link[:, 2] + struct
        if is_m7:
            ring_f = np.argmin(np.abs(zf[:, None] - ring_z[None, :]), axis=1).astype(np.int64) if n_f else np.zeros(0, dtype=np.int64)
            row_f = col_table[col_index[fl_row], ring_f]
        else:
            ring_f = np.full(n_f, k, dtype=np.int64)
            row_f = fl_row
        # blinking
        kk = np.asarray(rng.geometric(p_blink, n_f), dtype=np.int64)
        total = int(kk.sum())
        loc_fl = np.repeat(np.arange(n_f, dtype=np.int64), kk)
        starts = np.cumsum(kk) - kk
        steps = np.asarray(rng.integers(1, cfg.max_dark_frames + 2, size=total), dtype=np.int64)
        steps[starts] = 0
        off = np.cumsum(steps)
        off = off - off[starts][loc_fl]
        span = off[starts + kk - 1] + 1 if n_f else np.zeros(0, dtype=np.int64)
        if n_f and int(span.max()) > cfg.n_frames:
            raise ValueError(f"a fluorophore's {int(span.max())} frames exceed n_frames {cfg.n_frames}")
        start_frame = np.asarray(rng.integers(0, cfg.n_frames - span + 1), dtype=np.int64) if n_f else np.zeros(0, dtype=np.int64)
        frame = start_frame[loc_fl] + off
        # lateral
        lp = lp_samples[rng.integers(0, lp_samples.size, total)]
        lab = rot @ np.vstack([cx, cy, zf])
        x = lab[0][loc_fl] + rng.normal(size=total) * lp
        y = lab[1][loc_fl] + rng.normal(size=total) * lp
        z_true = lab[2][loc_fl]
        # axial
        stratum = _stratum_of(z_true, edges)
        lpz = _draw_lpz(rng, stratum, samples, borrow)
        z = z_true + rng.normal(size=total) * lpz * scale[stratum]
        parts["x"].append(x); parts["y"].append(y); parts["z"].append(z); parts["frame"].append(frame)
        parts["lp"].append(lp); parts["lpz"].append(lpz); parts["z_true"].append(z_true)
        parts["ring"].append(ring_f[loc_fl])
        parts["cluster"].append(clusters.cluster_id[row_f][loc_fl])
        parts["column"].append(clusters.column_id[row_f][loc_fl])
        parts["fluor"].append(fluor_offset + loc_fl)
        parts["row"].append(np.asarray(row_f[loc_fl], dtype=np.int64))
        n_fluor_by_cluster += np.bincount(row_f, minlength=n_rows).astype(np.int64)
        n_locs_by_cluster += np.bincount(row_f[loc_fl], minlength=n_rows).astype(np.int64)
        fluor_offset += n_f
    # background
    c = np.asarray(geometry.contour_used_nm, dtype=np.float64)
    margin = float(cfg.background_margin_nm)
    x_lo, x_hi = float(c[:, 0].min() - margin), float(c[:, 0].max() + margin)
    y_lo, y_hi = float(c[:, 1].min() - margin), float(c[:, 1].max() + margin)
    pick: Optional[NDArray[np.float64]] = None
    if cfg.pick_polygon_nm is not None:
        # H5-D: the box must cover the real pick, whose localizations are all kept below (the probe's widened box)
        pick = np.asarray(cfg.pick_polygon_nm, dtype=np.float64)
        x_lo, x_hi = min(x_lo, float(pick[:, 0].min())), max(x_hi, float(pick[:, 0].max()))
        y_lo, y_hi = min(y_lo, float(pick[:, 1].min())), max(y_hi, float(pick[:, 1].max()))
    z_lo, z_hi = float(ring_z.min() - 0.5 * cfg.period_nm), float(ring_z.max() + 0.5 * cfg.period_nm)
    volume_um3 = float((x_hi - x_lo) * (y_hi - y_lo) * (z_hi - z_lo)) / 1e9
    n_bg = int(rng.poisson(cfg.background_per_um3 * volume_um3))
    xb = rng.uniform(x_lo, x_hi, n_bg)
    yb = rng.uniform(y_lo, y_hi, n_bg)
    zb = rng.uniform(z_lo, z_hi, n_bg)
    lab_b = rot @ np.vstack([xb, yb, zb])
    frame_b = np.asarray(rng.integers(0, cfg.n_frames, n_bg), dtype=np.int64)
    lp_b = lp_samples[rng.integers(0, lp_samples.size, n_bg)]
    lpz_b = _draw_lpz(rng, _stratum_of(lab_b[2], edges), samples, borrow)
    minus = np.full(n_bg, -1, dtype=np.int64)
    parts["x"].append(lab_b[0]); parts["y"].append(lab_b[1]); parts["z"].append(lab_b[2]); parts["frame"].append(frame_b)
    parts["lp"].append(lp_b); parts["lpz"].append(lpz_b); parts["z_true"].append(lab_b[2].copy())
    for name in ("ring", "cluster", "column", "fluor", "row"):
        parts[name].append(minus)
    cat: Dict[str, NDArray[Any]] = {name: np.concatenate(v) if v else np.zeros(0) for name, v in parts.items()}
    n_outside = 0
    if pick is not None:
        # H5-D: the real file is the pick's selection -- no localization, background or cluster, lies outside it
        # (the feasibility probe's rule: the measured positions rotated back to the axon frame)
        axon_xyz = rot.T @ np.vstack([cat["x"], cat["y"], cat["z"]])
        keep = _inside_polygon(axon_xyz[0], axon_xyz[1], pick)
        n_outside = int(np.count_nonzero(~keep))
        cat = {name: v[keep] for name, v in cat.items()}
        rows_kept = np.asarray(cat["row"], dtype=np.int64)
        n_locs_by_cluster = np.bincount(rows_kept[rows_kept >= 0], minlength=n_rows).astype(np.int64)
        n_bg = int(np.count_nonzero(rows_kept < 0))
        if keep.size and not keep.any():
            warnings_.append("the pick polygon keeps no localization (is it in the axon frame of the contour?)")
    order = np.argsort(cat["frame"], kind="stable")
    if n_rows and int(n_fluor_by_cluster.sum()) == 0:
        warnings_.append("no fluorophore in any cluster (n_fluor_mean or the deciles give 0)")
    return SimLocalizations(
        x_nm=np.asarray(cat["x"][order], dtype=np.float64), y_nm=np.asarray(cat["y"][order], dtype=np.float64),
        z_nm=np.asarray(cat["z"][order], dtype=np.float64), frame=np.asarray(cat["frame"][order], dtype=np.int64),
        lp_lateral_nm=np.asarray(cat["lp"][order], dtype=np.float64), lpz_nm=np.asarray(cat["lpz"][order], dtype=np.float64),
        ring_true=np.asarray(cat["ring"][order], dtype=np.int64), cluster_true=np.asarray(cat["cluster"][order], dtype=np.int64),
        fluor_id=np.asarray(cat["fluor"][order], dtype=np.int64), column_true=np.asarray(cat["column"][order], dtype=np.int64),
        z_true_lab_nm=np.asarray(cat["z_true"][order], dtype=np.float64),
        n_fluor_by_cluster=n_fluor_by_cluster, n_locs_by_cluster=n_locs_by_cluster, n_background=n_bg, warnings=warnings_,
        n_outside_pick=n_outside)


# ============================================================================
# The simulator
# ============================================================================

def simulate_axon(cfg: SimConfig, seed: int) -> SimAxon:
    """
    One simulated axon (03_plan S3.2): ``rng = default_rng(SeedSequence
    (seed))``; the contour (``_resolve_contour``: config, library or
    ellipse -- the Fourier phases are the first draws when a perturbation
    is requested; then, with ``cfg.contour_smoothing_knot_nm``, its
    P-spline membrane at that knot spacing, ``contour_used_nm`` being the
    smoothed curve and the source suffixed "+pspline<knot>nm") and its
    interpolating curve (``smooth_closed_path``);
    the ring z, ``z_middle_lab_nm + (k - (N - 1) / 2) P + eta_k`` with
    eta_k ~ N(0, sigma_period) drawn for every ring (the middle ring
    sits at z_middle_lab exactly when sigma_period = 0; independent
    offsets, so consecutive ring spacings have sd sigma_period sqrt(2));
    then ring by ring ``cluster_positions`` and the radial offsets
    N(0, ``radial_offset_sd_nm``), the true centre being
    ``membrane_point(path, s, r)``; then ``measure``. The laboratory
    axis is (sin b cos phi, sin b sin phi, cos b) for tilt b and azimuth
    phi and laboratory = ``frame_rotation`` @ axon coordinates. The
    cluster table's ``n_fluor`` / ``n_locs`` are counted from the
    localizations' truth labels (for M7 the nearest ring's cluster of
    the column). Localizations come back in frame order.
    """
    if not isinstance(cfg, SimConfig):
        raise ValueError(f"simulate_axon: expected a SimConfig, got {type(cfg).__name__}")
    seed_int = int(seed)
    rng = np.random.default_rng(np.random.SeedSequence(seed_int))
    warnings_: List[str] = []
    contour, source = _resolve_contour(cfg, rng, warnings_)
    # H5-B fix: the membrane at the scale the radial scatter was measured on (SimConfig.contour_smoothing_knot_nm).
    contour, source, length_raw = _contour_as_used(cfg, contour, source)
    path = smooth_closed_path(contour)
    length = float(path.length_nm)
    if cfg.contour_smoothing_knot_nm is None:
        length_raw = length
    # H5-C: lambda on the length it was calibrated on (SimConfig.length_basis; effective_clusters_per_um). The
    # configuration handed to cluster_positions carries the effective lambda; SimAxon.config stays the caller's.
    lam_used = effective_clusters_per_um(cfg, length_raw, length)
    cfg_draw = cfg if lam_used == float(cfg.clusters_per_um) else dataclasses.replace(cfg, clusters_per_um=lam_used)
    n_rings = int(cfg.n_rings)
    eta = rng.normal(0.0, cfg.sigma_period_nm, n_rings)
    ring_z = cfg.z_middle_lab_nm + (np.arange(n_rings, dtype=np.float64) - 0.5 * (n_rings - 1)) * cfg.period_nm + eta
    b, ph = math.radians(cfg.tilt_deg), math.radians(cfg.azimuth_deg)
    axis = np.array([math.sin(b) * math.cos(ph), math.sin(b) * math.sin(ph), math.cos(b)], dtype=np.float64)
    rotation = _rotation_z_to(axis) if cfg.tilt_deg > 0.0 else np.eye(3)
    geometry = SimGeometry(contour_used_nm=contour, path=path, length_nm=length, outward_sign=_outward_sign(path),
                           ring_z_nm=np.asarray(ring_z, dtype=np.float64), axis=axis, frame_rotation=rotation, contour_source=source)
    # geometry: ring by ring
    rings_s: List[NDArray[np.float64]] = []
    rings_col: List[NDArray[np.int64]] = []
    rings_copy: List[NDArray[np.bool_]] = []
    rings_r: List[NDArray[np.float64]] = []
    prev_s: Optional[NDArray[np.float64]] = None
    prev_col: Optional[NDArray[np.int64]] = None
    prev_member: Optional[NDArray[np.bool_]] = None
    next_id = 0
    m2_origin: Optional[float] = None
    if cfg.model == "M2":
        n_lat = max(1, int(round(length / cfg.m2_lattice_nm)))
        m2_origin = float(rng.uniform(0.0, length / n_lat))
    for k in range(n_rings):
        cfg_k = cfg_draw if cfg.ring_rate_factors is None else _ring_config(cfg_draw, k)
        s, col, is_copy = cluster_positions(cfg_k, rng, length, prev_s, prev_col, prev_is_member=prev_member,
                                            next_id=next_id, m2_origin_nm=m2_origin)
        r = rng.normal(0.0, cfg.radial_offset_sd_nm, s.size)
        rings_s.append(s); rings_col.append(col); rings_copy.append(is_copy); rings_r.append(r)
        if col.size:
            next_id = max(next_id, int(col.max()) + 1)
        prev_s, prev_col, prev_member = s, col, is_copy
        if s.size == 0:
            warnings_.append(f"ring {k} has no cluster")
    ring_of = np.concatenate([np.full(s.size, k, dtype=np.int64) for k, s in enumerate(rings_s)]) if rings_s else np.zeros(0, dtype=np.int64)
    s_all = np.concatenate(rings_s).astype(np.float64)
    col_all = np.concatenate(rings_col).astype(np.int64)
    copy_all = np.concatenate(rings_copy).astype(bool)
    r_all = np.concatenate(rings_r).astype(np.float64)
    point, tangent, normal_out = _frame_at_arc(path, s_all)
    centre = point + r_all[:, None] * normal_out
    n_cl = int(s_all.size)
    clusters = SimClusterTable(
        ring=ring_of, cluster_id=np.arange(n_cl, dtype=np.int64), column_id=col_all, s_nm=s_all,
        x_nm=np.asarray(centre[:, 0], dtype=np.float64), y_nm=np.asarray(centre[:, 1], dtype=np.float64),
        z_nm=np.asarray(ring_z[ring_of], dtype=np.float64), r_offset_nm=r_all,
        n_fluor=np.zeros(n_cl, dtype=np.int64), n_locs=np.zeros(n_cl, dtype=np.int64),
        tangent=tangent, normal_out=normal_out, is_copy=copy_all)
    locs = measure(cfg, rng, clusters, geometry)
    clusters.n_fluor = locs.n_fluor_by_cluster
    clusters.n_locs = locs.n_locs_by_cluster
    warnings_.extend(locs.warnings)
    return SimAxon(
        x_nm=locs.x_nm, y_nm=locs.y_nm, z_nm=locs.z_nm, frame=locs.frame, lp_lateral_nm=locs.lp_lateral_nm, lpz_nm=locs.lpz_nm,
        ring_true=locs.ring_true, cluster_true=locs.cluster_true, fluor_id=locs.fluor_id, column_true=locs.column_true,
        z_true_lab_nm=locs.z_true_lab_nm, clusters=clusters, axis=axis, frame_rotation=rotation, contour_used_nm=contour,
        length_nm=length, ring_z_nm=np.asarray(ring_z, dtype=np.float64), config=cfg, seed=seed_int, warnings=warnings_,
        length_raw_nm=float(length_raw), clusters_per_um_used=float(lam_used), n_outside_pick=int(locs.n_outside_pick))


# ============================================================================
# Measuring a real axon's nuisance parameters (02 B8 step 1)
# ============================================================================

def _git_commit() -> str:
    """The short commit of the repository (read-only ``git rev-parse``),
    cached; "unknown" when git is unavailable."""
    global _GIT_COMMIT
    if _GIT_COMMIT is None:
        try:
            out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, capture_output=True,
                                 text=True, timeout=20, check=False)
            _GIT_COMMIT = out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
        except Exception:  # noqa: BLE001 - provenance only
            _GIT_COMMIT = "unknown"
    return _GIT_COMMIT


def _thin_quantiles(values: NDArray[np.float64], n_max: int) -> NDArray[np.float64]:
    """At most ``n_max`` values: the sorted input itself when it is short
    enough, else its values at ``n_max`` evenly spaced ranks (a
    deterministic quantile thinning that keeps every percentile)."""
    v = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if v.size <= n_max:
        return v
    ranks = np.round(np.linspace(0, v.size - 1, n_max)).astype(np.int64)
    return np.asarray(v[ranks], dtype=np.float64)


def _smoothed_closed_contour(contour: NDArray[np.float64], n_samples: int, warnings_: List[str], label: str) -> NDArray[np.float64]:
    """The s = K periodic smoothing B-spline of a ring's contour
    (``tools.mps_randomization.smooth_contour_bspline``) fitted on the
    CLOSED polygon -- the first vertex appended, because ``splprep(per=1)``
    ignores the last point it is given (it is the period's end), so the
    open contour would leave the last cluster out of the fit -- with
    ``s = K`` (K the ORIGINAL vertices, scipy's rule of thumb for unit
    weights, D-25's pre-registered scale), resampled at ``n_samples``
    points. Falls back to the polygon (with a warning) when the fit
    fails."""
    c = np.asarray(contour, dtype=np.float64)
    closed = np.vstack([c, c[:1]])
    out = smooth_contour_bspline(closed, n_samples=n_samples, smoothing=float(c.shape[0]))
    if out.shape[0] != n_samples or not np.isfinite(out).all():
        warnings_.append(f"{label}: the smoothing spline failed; the polygon is used")
        polygon: NDArray[np.float64] = np.array(c, dtype=np.float64)
        return polygon
    smoothed: NDArray[np.float64] = np.asarray(out, dtype=np.float64)
    return smoothed


def _cluster_lateral_widths(res: RingsResult, rings: Sequence[Ring], min_locs: int = 10) -> Tuple[float, float, int]:
    """Median over the kept clusters (>= ``min_locs`` localizations) of
    the sd of a cluster's localizations about its centroid along the
    ring's TANGENT and along its NORMAL (axon frame; the tangent at a
    cluster is the direction between its two polygon neighbours), with
    the number of clusters: (sd_tangent, sd_normal, n). The two widths
    separate the epitope disc from the linkage: sd_n^2 = sigma_link^2 +
    lp^2 (nothing spreads across the membrane but the linkage and the
    error) and sd_t^2 = R^2 / 4 + sigma_link^2 + lp^2. The MEDIAN over
    clusters, not the pooled sd: a detected cluster that DBSCAN merged
    with a leak child sitting a few tens of nm along the membrane is
    elongated along the tangent, and the pooled sd of the M1 leak axon
    of validate_leak came out 33 nm along the tangent for a truth of
    11.3 (2026-09-25); the median of the per-cluster sd is 11.4.
    (NaN, NaN, 0) without a usable cluster."""
    xp = np.asarray(res.x_p, dtype=np.float64)
    yp = np.asarray(res.y_p, dtype=np.float64)
    sd_t: List[float] = []
    sd_n: List[float] = []
    for r in rings:
        if r.contour_nm is None:
            continue
        c = np.asarray(r.contour_nm, dtype=np.float64)
        k = c.shape[0]
        if k < 3:
            continue
        for cl in r.clusters:
            li = np.asarray(cl.loc_index, dtype=np.intp)
            if li.size < max(2, int(min_locs)):
                continue
            centroid = np.asarray(cl.centroid_nm, dtype=np.float64)
            v = int(np.argmin(np.hypot(c[:, 0] - centroid[0], c[:, 1] - centroid[1])))
            t = c[(v + 1) % k] - c[(v - 1) % k]
            nt = float(np.hypot(t[0], t[1]))
            if nt <= 0.0:
                continue
            t = t / nt
            dx, dy = xp[li] - centroid[0], yp[li] - centroid[1]
            along = dx * t[0] + dy * t[1]
            across = -dx * t[1] + dy * t[0]
            sd_t.append(float(np.std(along, ddof=1)))
            sd_n.append(float(np.std(across, ddof=1)))
    if not sd_t:
        return float("nan"), float("nan"), 0
    return float(np.median(sd_t)), float(np.median(sd_n)), len(sd_t)


def _axially_one_sided(ring: Ring, period_nm: float) -> Tuple[NDArray[np.bool_], float]:
    """Which kept clusters of ``ring`` are AXIALLY ONE-SIDED -- leak
    children rather than clusters of this ring -- and the threshold
    used. A child is the tail of a neighbouring ring's cluster beyond
    the cut at P / 2, so the mean z' of its localizations sits at
    P - E[Z | Z > P / 2] from this ring's centre (Z ~ N(0, sigma_z),
    sigma_z the ring's component width: 46 nm on the M1 leak axon of
    validate_leak, 66 nm at P 190, sigma 65), whereas
    a cluster of this ring sits at 0 +/- sigma_z / sqrt(n) (6 nm at n =
    150). The threshold is half the expected child offset (P / 4 when
    sigma_z is unusable). Nothing is removed from the rings: the mask
    only decides which clusters enter the counts, the hard core and
    the size deciles of the TRUE clusters, because the simulation
    re-creates the children itself through the leak (a lambda measured
    on every kept cluster re-creates them on top of the copies:
    re-detected K + 25 % on that axon, 2026-09-25)."""
    sigma = float(ring.sigma_z_nm)
    a = 0.5 * float(period_nm)
    if math.isfinite(sigma) and sigma > 0.0 and float(norm.sf(a / sigma)) > 0.0:
        tail_mean = sigma * float(norm.pdf(a / sigma) / norm.sf(a / sigma))
        threshold = 0.5 * max(float(period_nm) - tail_mean, 0.0)
        if threshold <= 0.0:
            threshold = 0.25 * float(period_nm)
    else:
        threshold = 0.25 * float(period_nm)
    centre = float(ring.centre_z_nm)
    offsets = np.array([float(np.mean(np.asarray(cl.z_values_nm, dtype=np.float64))) - centre if np.asarray(cl.z_values_nm).size
                        else 0.0 for cl in ring.clusters])
    return np.asarray(np.abs(offsets) > threshold, dtype=bool), threshold


def _gap_quantile_of_hard_core(d_min_nm: float, length_nm: float, k_per_ring: int, probability: float,
                               n_rings: int, seed: int) -> float:
    """The ``probability`` quantile of the cyclic gaps the simulator's own
    hard-core sampler produces with ``k_per_ring`` clusters on
    ``n_rings`` rings of length ``length_nm`` at hard core ``d_min_nm``
    (a fixed seed, so the calibration is deterministic); RuntimeError
    when the sampler jams."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    gaps: List[NDArray[np.float64]] = []
    for _ in range(n_rings):
        s = np.sort(_hard_core_arcs(rng, k_per_ring, length_nm, d_min_nm, np.zeros(0), max_attempts=HARD_CORE_CALIBRATION_ATTEMPTS))
        gaps.append(np.diff(np.concatenate([s, [s[0] + length_nm]])))
    return float(np.percentile(np.concatenate(gaps), 100.0 * probability))


def _hard_core_by_quantile_matching(gaps_nm: NDArray[np.float64], length_nm: float, k_per_ring: float,
                                    probability: float = 0.05) -> float:
    """The hard core d_min such that the simulator's hard-core sampler,
    at the measured clusters per ring and ring length, reproduces the
    observed ``probability`` quantile (the 5th percentile) of the cyclic
    gaps between the clusters of a ring: bisection on d in [0, q_obs]
    (the sampler's quantile is >= d and increases with d) with
    ``_gap_quantile_of_hard_core`` on ``HARD_CORE_CALIBRATION_RINGS``
    rings and a fixed seed, to ``HARD_CORE_CALIBRATION_TOL_NM``.

    Why not the bare 5th percentile the specification names: every gap
    of a hard-core ring is >= d, so any percentile overestimates d by
    the excess at that rank, which depends on the coverage K d / L: on
    the M1 leak axon of validate_leak (true hard core 80 nm, 40 clusters
    on 7.9 um) the 5th percentile of the true clusters' gaps is 97 nm and
    of their nearest-neighbour spacings 92 nm (2026-09-25), and the
    excess is not exponential near the core (a shifted-exponential
    correction still gave 91). The observed minimum would be unbiased
    but is one draw (a single split or merged cluster sets it), which is
    why the specification chose a percentile; matching that percentile
    under the generator keeps its robustness and removes the bias (80
    +/- 2 nm on the same axon). A jammed sampler (coverage too high)
    counts as "d too large"."""
    g = np.asarray(gaps_nm, dtype=np.float64)
    g = g[np.isfinite(g)]
    k = int(round(float(k_per_ring)))
    if g.size < 2 or k < 2 or not (length_nm > 0.0):
        return 0.0
    q_obs = float(np.percentile(g, 100.0 * probability))
    if q_obs <= 0.0:
        return 0.0

    def quantile(d: float) -> float:
        """The sampler's quantile at d, +inf when it jams."""
        try:
            return _gap_quantile_of_hard_core(d, length_nm, k, probability, HARD_CORE_CALIBRATION_RINGS, HARD_CORE_CALIBRATION_SEED)
        except RuntimeError:
            return math.inf

    if quantile(0.0) >= q_obs:
        return 0.0
    lo, hi = 0.0, q_obs
    while hi - lo > HARD_CORE_CALIBRATION_TOL_NM:
        mid = 0.5 * (lo + hi)
        if quantile(mid) < q_obs:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _retained_fraction(ring: Ring) -> float:
    """Fraction of a cluster's localizations the ring's axial window
    [z_lo, z_hi] keeps when its z is N(centre, sigma_z): Phi((z_hi -
    c) / sigma) - Phi((z_lo - c) / sigma); 1 when the width is unusable."""
    sigma = float(ring.sigma_z_nm)
    if not (math.isfinite(sigma) and sigma > 0.0):
        return 1.0
    c = float(ring.centre_z_nm)
    return float(norm.cdf((float(ring.z_hi_nm) - c) / sigma) - norm.cdf((float(ring.z_lo_nm) - c) / sigma))


def _c4(n: int) -> float:
    """The unbiasing constant of a sample standard deviation from ``n``
    values, E[s] = c4(n) sigma: sqrt(2 / (n - 1)) Gamma(n / 2) /
    Gamma((n - 1) / 2) (0.798 at n = 2, 0.886 at n = 3, 0.940 at n = 5);
    1 for n < 2."""
    if n < 2:
        return 1.0
    return float(math.sqrt(2.0 / (n - 1)) * math.exp(gammaln(n / 2.0) - gammaln((n - 1) / 2.0)))


def _scale_at_depth(edges: NDArray[np.float64], scale: NDArray[np.float64], z_lab_nm: float) -> float:
    """The axial scale at a laboratory depth: the NeNA ratio interpolated
    LINEARLY between the stratum centres (held flat beyond the outer
    centres), not the step function of the strata -- a ring sits at
    one depth and the step made sigma_struct jump to 0 when a ring's
    depth crossed a 100 nm edge (large jumps across adjacent
    strata of a measured configuration, review of 2026-09-25); the simulator itself keeps the
    step per localization, which is what the file's lpz table is."""
    centres = 0.5 * (edges[:-1] + edges[1:])
    if centres.size == 1:
        return float(scale[0])
    return float(np.interp(float(z_lab_nm), centres, scale))


def _ring_depth_lab_nm(ring: Ring, z_lab: NDArray[np.float64], z_p: NDArray[np.float64], period_nm: float) -> float:
    """The laboratory depth of a ring: the median laboratory z of its
    localizations within P / 4 of the ring's centre in z' -- the core
    of the ring, which the leak of the neighbouring rings (children
    beyond P / 4) does not pull (the median over every localization sat
    towards the deeper neighbour on a realistic axon,
    review of 2026-09-25); every localization when the core is empty."""
    li = np.asarray(ring.loc_index, dtype=np.intp)
    if li.size == 0:
        return float(ring.centre_z_nm)
    core = np.abs(z_p[li] - float(ring.centre_z_nm)) <= 0.25 * float(period_nm)
    sel = li[core] if np.count_nonzero(core) >= 10 else li
    return float(np.median(z_lab[sel]))


@dataclass
class _DetectedSummary:
    """What a ``RingsResult`` says about the clusters the pipeline
    detected, in the terms the calibration compares: the rings in z
    order, per ring the axially one-sided mask and its threshold, K
    kept, K own (kept and not one-sided), the ring lengths, the n_locs
    of the own clusters pooled, the background density."""

    rings: List[Ring]
    one_sided: List[NDArray[np.bool_]]
    thresholds: List[float]
    k_kept: NDArray[np.float64]
    k_own: NDArray[np.float64]
    lengths: NDArray[np.float64]
    sizes: NDArray[np.float64]
    gaps_nm: NDArray[np.float64]
    background_per_um3: float
    n_far: int
    excluded_fraction: float


def _ring_localization_box_nm(res: RingsResult, rings: Sequence[Ring]) -> Optional[Tuple[float, float, float, float]]:
    """(x0, x1, y0, y1) bounding box, axon frame, of every localization of
    the rings (the pick's extent in the axial windows); None without one."""
    xp = np.asarray(res.x_p, dtype=np.float64)
    yp = np.asarray(res.y_p, dtype=np.float64)
    li = [np.asarray(r.loc_index, dtype=np.intp) for r in rings if np.asarray(r.loc_index).size]
    if not li:
        return None
    idx = np.concatenate(li)
    return float(xp[idx].min()), float(xp[idx].max()), float(yp[idx].min()), float(yp[idx].max())


def pick_polygon_of_rings(res: RingsResult, ring_indices: Optional[Sequence[int]] = None) -> NDArray[np.float64]:
    """
    The real pick of an axon as ``SimConfig.pick_polygon_nm`` (H5-D,
    D-32c): the convex hull, in build_rings' axon frame (``res.x_p``,
    ``res.y_p``), of every localization of the rings (``Ring.loc_index``;
    ``ring_indices`` None = every ring) -- the region the pick's
    localizations fill, as the feasibility study defined it
    (a research script). The hull's vertices
    in counter-clockwise order, (P, 2). ValueError with fewer than three
    localizations or when they span no area.
    """
    from scipy.spatial import ConvexHull, QhullError
    wanted = None if ring_indices is None else {int(k) for k in ring_indices}
    li = [np.asarray(r.loc_index, dtype=np.intp).reshape(-1) for r in res.rings if wanted is None or int(r.index) in wanted]
    idx = np.concatenate(li) if li else np.zeros(0, dtype=np.intp)
    if idx.size < 3:
        raise ValueError(f"pick_polygon_of_rings: {idx.size} localization(s) in the rings; at least 3 are needed")
    pts = np.column_stack([np.asarray(res.x_p, dtype=np.float64)[idx], np.asarray(res.y_p, dtype=np.float64)[idx]])
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError) as exc:
        raise ValueError(f"pick_polygon_of_rings: the rings' localizations span no area ({exc})") from exc
    return np.asarray(pts[hull.vertices], dtype=np.float64)


def _pick_margin_nm(res: RingsResult, rings: Sequence[Ring], contour: Optional[NDArray[np.float64]]) -> Tuple[float, List[float]]:
    """The margin of the pick around the membrane (``SimConfig.
    background_margin_nm``) and its four sides: the distance from the
    bounding box of the rings' localizations (the pick, filled by the
    background) to the bounding box of ``contour`` (the membrane) on
    the left, right, bottom and top; the margin is the median of the
    four, floored at 0. The four sides are where the simulated box is
    tangent to the membrane and where the pipeline's edge criterion
    bites, so the median of the sides is what decides whether simulated
    clusters are flagged as the real ones were (re-review of
    2026-09-25). (``BACKGROUND_MARGIN_DEFAULT_NM``, []) without a contour
    or localizations. Without background the localizations' box is the
    clusters' own extent beyond the contour through their centroids
    (~50-80 nm), which is then what the value says."""
    box = _ring_localization_box_nm(res, rings)
    if contour is None or box is None or contour.shape[0] < 3:
        return float(BACKGROUND_MARGIN_DEFAULT_NM), []
    c = np.asarray(contour, dtype=np.float64)
    sides = [float(c[:, 0].min()) - box[0], box[1] - float(c[:, 0].max()), float(c[:, 1].min()) - box[2], box[3] - float(c[:, 1].max())]
    return float(max(float(np.median(sides)), 0.0)), sides


def _background_per_um3(res: RingsResult, rings: Sequence[Ring]) -> Tuple[float, int, float]:
    """(density per um^3, count, excluded area fraction) of the uniform
    background: the localizations of the rings that are in no kept
    cluster AND farther than ``BACKGROUND_EXCLUSION_NM`` (axon-frame
    x', y') from EVERY kept cluster centroid of EVERY ring, divided by
    the rings' bounding volume (bounding box of the rings'
    LOCALIZATIONS -- the pick, over which the background is spread --
    x the sum of the axial windows) times the fraction of that box
    OUTSIDE the exclusion discs (measured on a ``BACKGROUND_GRID``^2
    grid). The contours' box (the membrane) was the denominator before
    the re-review of 2026-09-25: it left the pick's margin out and
    overstated the density by the area ratio (x2 at a 500 nm margin on
    a 2 um axon) on the real AND the simulated axon alike, which the
    closure ratio cancelled but the bare value carried. Why not every
    unclustered localization (the specification's count): those are
    DBSCAN's border losses of the true clusters and the leak children
    too small to be kept, several percent of a realistic axon with NO
    background, i.e. a sizeable density for a truth of 0, enough to switch the
    pipeline's edge-touching criterion on and lose many of the
    re-simulated clusters (review of 2026-09-25)."""
    xp = np.asarray(res.x_p, dtype=np.float64)
    yp = np.asarray(res.y_p, dtype=np.float64)
    cents = [np.asarray(cl.centroid_nm, dtype=np.float64) for r in rings for cl in r.clusters]
    box = _ring_localization_box_nm(res, rings)
    if box is None:
        return 0.0, 0, 0.0
    x0, x1, y0, y1 = box
    z_span = float(sum(float(r.z_hi_nm) - float(r.z_lo_nm) for r in rings))
    volume = (x1 - x0) * (y1 - y0) * z_span / 1e9
    if volume <= 0.0:
        return 0.0, 0, 0.0
    n_far = 0
    excluded = 0.0
    if cents:
        tree = cKDTree(np.vstack(cents))
        for r in rings:
            li = np.asarray(r.loc_index, dtype=np.intp)
            in_cluster = np.zeros(li.size, dtype=bool)
            pos = {int(i): j for j, i in enumerate(li.tolist())}
            for cl in r.clusters:
                for i in np.asarray(cl.loc_index, dtype=np.intp).tolist():
                    j = pos.get(int(i))
                    if j is not None:
                        in_cluster[j] = True
            free = li[~in_cluster]
            if free.size:
                d, _ = tree.query(np.column_stack([xp[free], yp[free]]), k=1)
                n_far += int(np.count_nonzero(d > BACKGROUND_EXCLUSION_NM))
        gx = np.linspace(x0, x1, BACKGROUND_GRID)
        gy = np.linspace(y0, y1, BACKGROUND_GRID)
        grid = np.column_stack([np.repeat(gx, gy.size), np.tile(gy, gx.size)])
        dg, _ = tree.query(grid, k=1)
        excluded = float(np.mean(dg <= BACKGROUND_EXCLUSION_NM))
    else:
        for r in rings:
            n_far += int(r.n_locs)
    free_fraction = max(1.0 - excluded, 0.05)
    return float(n_far / (volume * free_fraction)), int(n_far), excluded


def _own_cluster_gaps_nm(ring: Ring, one_sided: NDArray[np.bool_]) -> NDArray[np.float64]:
    """The cyclic gaps (along the ring's polygon) between the kept clusters
    of ``ring`` that are not axially one-sided; empty without a contour
    or with fewer than two such clusters."""
    if ring.contour_nm is None or not ring.clusters:
        return np.zeros(0, dtype=np.float64)
    c = np.asarray(ring.contour_nm, dtype=np.float64)
    cents = np.vstack([np.asarray(cl.centroid_nm, dtype=np.float64) for cl in ring.clusters])
    vertex_cluster = np.argmin(np.hypot(c[:, None, 0] - cents[None, :, 0], c[:, None, 1] - cents[None, :, 1]), axis=1)
    keep_v = ~one_sided[vertex_cluster]
    if int(keep_v.sum()) < 2:
        return np.zeros(0, dtype=np.float64)
    edge = np.hypot(*(np.roll(c, -1, axis=0) - c).T)
    cum = np.concatenate([[0.0], np.cumsum(edge)])
    arcs = np.sort(cum[:-1][keep_v])
    return np.asarray(np.diff(np.concatenate([arcs, [arcs[0] + cum[-1]]])), dtype=np.float64)


def _detected_summary(res: RingsResult, period_nm: float) -> _DetectedSummary:
    """The detected quantities of a ``RingsResult`` (see ``_DetectedSummary``)."""
    rings = sorted(res.rings, key=lambda r: float(r.centre_z_nm))
    one_sided: List[NDArray[np.bool_]] = []
    thresholds: List[float] = []
    sizes: List[float] = []
    gaps: List[NDArray[np.float64]] = []
    for r in rings:
        mask, thr = _axially_one_sided(r, period_nm)
        one_sided.append(mask)
        thresholds.append(thr)
        sizes.extend(float(cl.n_locs) for cl, one in zip(r.clusters, mask) if not one)
        gaps.append(_own_cluster_gaps_nm(r, mask))
    k_kept = np.array([len(r.clusters) for r in rings], dtype=np.float64)
    k_own = np.array([int(np.count_nonzero(~m)) for m in one_sided], dtype=np.float64)
    lengths = np.array([float(r.length_nm) for r in rings if math.isfinite(float(r.length_nm)) and float(r.length_nm) > 0.0])
    bg, n_far, excl = _background_per_um3(res, rings)
    return _DetectedSummary(rings=rings, one_sided=one_sided, thresholds=thresholds, k_kept=k_kept, k_own=k_own,
                            lengths=lengths, sizes=np.asarray(sizes, dtype=np.float64),
                            gaps_nm=np.concatenate(gaps) if gaps else np.zeros(0), background_per_um3=bg,
                            n_far=n_far, excluded_fraction=excl)


def _pool_detected(summaries: Sequence[_DetectedSummary]) -> _DetectedSummary:
    """The detected quantities of several simulated axons of one
    configuration as one summary for the calibration ratios: K kept
    and K own averaged ring by ring (the axons share the ring count),
    the sizes and gaps concatenated, the background density averaged,
    the ring lengths concatenated; the rings, masks and thresholds of
    the first axon (the ratios do not read them)."""
    if len(summaries) == 1:
        return summaries[0]
    first = summaries[0]
    return _DetectedSummary(
        rings=first.rings, one_sided=first.one_sided, thresholds=first.thresholds,
        k_kept=np.mean([s.k_kept for s in summaries], axis=0), k_own=np.mean([s.k_own for s in summaries], axis=0),
        lengths=np.concatenate([s.lengths for s in summaries]), sizes=np.concatenate([s.sizes for s in summaries]),
        gaps_nm=np.concatenate([s.gaps_nm for s in summaries]),
        background_per_um3=float(np.mean([s.background_per_um3 for s in summaries])),
        n_far=int(sum(s.n_far for s in summaries)), excluded_fraction=float(np.mean([s.excluded_fraction for s in summaries])))


def _clip_ratio(value: float) -> float:
    """A calibration ratio within ``CALIBRATION_RATIO_CLIP`` (1 when not finite)."""
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    lo, hi = CALIBRATION_RATIO_CLIP
    return float(min(max(value, lo), hi))


def _calibrate_by_simulation(
    cfg: SimConfig,
    target: _DetectedSummary,
    n_frames: int,
    rings_params: RingsParams,
    iterations: int,
    seed: int,
    warnings_: List[str],
    axons_per_iteration: int = CALIBRATION_AXONS_PER_ITERATION,
    radial_scatter_target_nm: Optional[float] = None,
    locs_per_event_target: Optional[float] = None,
    scatter_knot_spacing_nm: Optional[float] = None,
    closure: str = "v1",
    nena_target: Optional[AxialNena] = None,
    sigma_z_target_nm: Optional[Sequence[float]] = None,
) -> Tuple[SimConfig, List[Dict[str, Any]]]:
    """
    Closure of the measurement (02 B8 step 1: the simulator must
    reproduce the K and the cluster sizes of the axon it was measured
    on). ``iterations`` times: simulate ``cfg`` ``axons_per_iteration``
    times (seeds ``seed`` + i x n + j) with a FIXED number of clusters
    per ring, round(clusters_per_um x L) (the Poisson draw of K would
    add 10-15 % of noise per iteration to the K ratio and make the
    fixed point wander), build the rings of each with ``rings_params``
    (the parameters the real rings were built with, the bootstrap
    reduced), summarise what was DETECTED (``_detected_summary``: K own
    per ring, the n_locs deciles and the 5th-percentile gap of the own
    clusters, the background density) POOLED over the simulated axons
    (K own and the density averaged, the sizes and gaps concatenated:
    ``CALIBRATION_AXONS_PER_ITERATION``) and rescale the configuration
    by the ratio real / simulated of each quantity -- clusters_per_um
    by mean K own, the n_locs deciles by ONE factor
    (``CALIBRATION_DECILE_FROM``), d_min_nm by the ratio of the
    5th-percentile gaps, background_per_um3 by the density ratio --
    each ratio clipped to ``CALIBRATION_RATIO_CLIP``; n_fluor_mean
    follows the deciles. A simulated axon whose rings cannot be built
    or whose ring count differs from the real one (the comparison ring
    by ring is then undefined) is skipped with a warning; the
    calibration stops when no axon of an iteration could be
    summarised. Returns the calibrated
    configuration and the trajectory (one record per iteration with
    the simulated detected quantities, the seeds and the factors), for
    the provenance. Why a fixed point and not the measurement itself:
    DBSCAN never detects the clusters below min_samples localizations
    and loses part of the small ones, the ring windows cut the leak of
    a DETECTED size once more on re-simulation, and the background
    estimate carries what the detection missed, so the simulated axons
    of a grid measured without closure showed clearly fewer clusters
    than the real K, and smaller ones (review of 2026-09-25). The detection
    noise of the simulated axons is still in each factor (it is a
    stochastic fixed point): three iterations over two axons give it
    to ~5 % on lambda and ~10 % on the deciles.

    H5-C: with ``radial_scatter_target_nm`` (the observed axon's radial
    scatter on its LOCALIZATION membrane, ``sim_config_from_axon(
    membrane_source="localizations")``) the radial scatter is closed the
    same way: each simulated axon's scatter is measured exactly as the
    observed one (``radial_scatter_on_membrane`` of every centroid of
    its rings with a contour on their ``localization_membrane_of_rings``),
    pooled as an rms over the axons, and ``radial_offset_sd_nm`` is
    updated in VARIANCE by half a step, sd^2 <- max(sd^2 + (target^2 -
    simulated^2) / 2, 0) (the half step damps the two-axon noise):
    the measured scatter is the true one plus the centroid noise
    (linkage and precision over the cluster's localizations) plus the
    membrane's own estimation error, in quadrature, and the simulation
    adds both again on top of the configured sd, so the bare
    measurement re-simulates too wide by exactly that variance (+21 %
    on the M1 leak axon of validate_leak, whose true scatter is 0 and
    measured 3.5 nm; negligible at a scatter of several tens of nm). None (the
    default) leaves the scatter alone, today's calibration.

    Review of 2026-09-26: with ``locs_per_event_target`` (the observed
    axon's localizations per linked event, n_locs / ``RingsResult.
    n_events``, what ``sim_config_from_axon`` measures
    ``locs_per_fluor_mean`` as) the number of localizations per
    fluorophore is closed the same way: each calibration axon's n_locs
    / n_events is measured by the same linking, pooled as a mean, and
    ``locs_per_fluor_mean`` is rescaled by target / simulated (clipped,
    at least 1), after which ``n_fluor_mean`` follows the (closed)
    deciles at the new value. Why: the linking of ``build_rings``
    splits a fluorophore's run of frames into several events (the
    simulator's k ~ Geometric on consecutive frames, lp ~9 nm), so the
    measured ratio is biased low -- two synthetic originals were
    measured well below their true localizations per fluorophore and
    well above their true n_fluor -- and a simulator run
    at the MEASURED value makes its clusters from many more
    fluorophores with fewer localizations each: fewer clumps of
    localizations displaced together in z, a weaker leak (fewer spurious
    children and a lower arcl_z_A on the paired seeds of the original's
    generator; the review's attribution
    run), and so a simulated null centred too low for the leak it is
    meant to calibrate. None (the default) leaves it as measured,
    today's calibration. ``scatter_knot_spacing_nm`` (None = the
    membrane's ``MEMBRANE_KNOT_SPACING_NM``) is the knot spacing of the
    localization membrane each calibration axon's scatter is measured
    on: it must be the one ``radial_scatter_target_nm`` was measured on
    (``sim_config_from_axon(scatter_knot_spacing_nm=...)``).

    H5-E (D-40, ``closure="v2"``; see ``CLOSURE_VERSIONS``): the seeds of
    iteration it are ``seed + j`` for every it (common random numbers);
    with ``nena_target`` (the observed axon's ``AxialNena``) each
    calibration axon's NeNA is measured on the SAME strata and
    ``axial_scale_by_bin`` is multiplied per stratum by observed /
    simulated ratio (clipped; strata where either is undefined keep their
    scale); with ``sigma_z_target_nm`` (the observed rings' component
    sigma_z in z order) sigma_struct^2 moves by the mean over rings of
    observed^2 - simulated^2, the simulated value first advanced by the
    variance the scale step itself adds to that ring (lpz median x old
    scale)^2 x (f^2 - 1), floored at 0. After the last iteration lambda
    and the decile scale (geometric mean from ``CALIBRATION_DECILE_FROM``)
    are solved from log stat_sim = a + b log theta over the iterates at
    the observed statistic (``BINDING_SLOPE_RANGE``,
    ``BINDING_MIN_ITERATES``; the last iterate otherwise). "v1" is
    today's closure value for value.
    """
    if closure not in CLOSURE_VERSIONS:
        raise ValueError(f"closure must be one of {CLOSURE_VERSIONS}, got {closure!r}")
    v2 = closure == "v2"
    binding_lam: List[Tuple[float, float]] = []
    binding_dec: List[Tuple[float, float]] = []
    trajectory: List[Dict[str, Any]] = []
    current = cfg
    n_axons = max(1, int(axons_per_iteration))
    k_real = float(np.mean(target.k_own)) if target.k_own.size else float("nan")
    dec_real = (np.percentile(target.sizes, 100.0 * DECILE_PROBABILITIES) if target.sizes.size else None)
    q5_real = float(np.percentile(target.gaps_nm, 5.0)) if target.gaps_nm.size >= 20 else float("nan")
    probe_rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    contour_c, _source = _resolve_contour(cfg, probe_rng, [])
    # H5-C: the fixed K of the calibration axons is lambda x the length the SIMULATOR uses (the contour smoothed at
    # contour_smoothing_knot_nm when the configuration says so, the effective lambda of its length basis), so that
    # the closure is measured on the geometry the replicates will have; without smoothing this is the old rule.
    contour_used_c, _src, length_raw_c = _contour_as_used(cfg, contour_c, _source)
    length_c = float(smooth_closed_path(contour_used_c).length_nm)
    if cfg.contour_smoothing_knot_nm is None:
        length_raw_c = length_c
    for it in range(int(iterations)):
        k_fixed = max(1, int(round(effective_clusters_per_um(current, length_raw_c, length_c) * length_c / 1000.0)))
        summaries: List[_DetectedSummary] = []
        seeds: List[int] = []
        n_locs_sim: List[int] = []
        scatter_sim: List[float] = []
        lpe_sim: List[float] = []
        nena_sim: List[NDArray[np.float64]] = []
        sz_sim: List[List[Tuple[float, float, float]]] = []
        for j in range(n_axons):
            seed_ij = int(seed) + (j if v2 else it * n_axons + j)
            sim = simulate_axon(dataclasses.replace(current, n_clusters_per_ring=k_fixed), seed_ij)
            try:
                res = build_rings(np.array(sim.x_nm), np.array(sim.y_nm), np.array(sim.z_nm), frame=np.array(sim.frame),
                                  lp_lateral_nm=np.array(sim.lp_lateral_nm), lpz_nm=np.array(sim.lpz_nm), params=rings_params,
                                  source_name=f"calibration-{it}-{j}", pixel_size_nm=float(current.pixel_size_nm),
                                  pixel_size_source="override", n_frames=int(n_frames))
            except Exception as exc:  # noqa: BLE001 - the calibration is best effort; the measurement stands
                warnings_.append(f"calibration iteration {it}, axon {j} (seed {seed_ij}): build_rings failed ({type(exc).__name__}: {exc}); skipped")
                continue
            if len(res.rings) != len(target.rings):
                warnings_.append(f"calibration iteration {it}, axon {j} (seed {seed_ij}): the simulated axon has {len(res.rings)} rings "
                                 f"for {len(target.rings)} real ones; skipped")
                continue
            summaries.append(_detected_summary(res, current.period_nm))
            seeds.append(seed_ij)
            if v2:
                z_all = np.asarray(sim.z_nm, dtype=np.float64)
                if nena_target is not None:
                    nena_sim.append(_nena_on_strata(sim, res, nena_target))
                if sigma_z_target_nm is not None:
                    sz_sim.append([(float(r.sigma_z_nm), float(r.lpz_median_nm),
                                    float(np.median(z_all[np.asarray(r.loc_index, dtype=np.intp)])) if np.asarray(r.loc_index).size else float("nan"))
                                   for r in sorted(res.rings, key=lambda r: float(r.centre_z_nm))])
            n_locs_sim.append(int(sim.n_locs))
            if locs_per_event_target is not None and res.n_events is not None and int(res.n_events) > 0:
                lpe_sim.append(float(sim.n_locs) / float(res.n_events))
            if radial_scatter_target_nm is not None:
                sigma_j = _localization_scatter_nm(res, scatter_knot_spacing_nm)
                if math.isfinite(sigma_j):
                    scatter_sim.append(sigma_j)
                else:
                    warnings_.append(f"calibration iteration {it}, axon {j} (seed {seed_ij}): no localization membrane; "
                                     "the radial scatter is not calibrated on it")
        if not summaries:
            warnings_.append(f"calibration iteration {it}: no simulated axon could be summarised; stopped")
            break
        summary = _pool_detected(summaries)
        k_sim = float(np.mean(summary.k_own)) if summary.k_own.size else float("nan")
        if v2 and k_sim > 0.0 and math.isfinite(k_sim):
            binding_lam.append((float(current.clusters_per_um), k_sim))
        f_lambda = _clip_ratio(k_real / k_sim) if k_sim > 0.0 else 1.0
        kw: Dict[str, Any] = {"clusters_per_um": current.clusters_per_um * f_lambda}
        if current.ring_rate_factors is not None and target.k_own.size == summary.k_own.size == len(current.ring_rate_factors):
            # H5-C: ring by ring (the per-axon null conditions on each ring's K). The common rate takes the full step of
            # the pooled K (lambda <- lambda x sum K_real / sum K_sim, as without factors) and each ring's factor HALF the
            # step of its own ratio relative to the common one (a geometric half step, as d_min's): one ring's detected K
            # over two calibration axons varies noticeably (the deep ring of a research harness axon moved markedly in
            # one iteration), and the full step of the first H5-C version fitted the last iteration's noise -- the
            # re-simulated K_kept of that ring well above the observed one, and the 15 % criterion missed on a draw of
            # the review.
            # The factors are renormalised to mean 1 and the common rate absorbs the mean.
            k_real_sum = float(np.sum(target.k_own))
            k_sim_sum = float(np.sum(summary.k_own))
            f_common = _clip_ratio(k_real_sum / k_sim_sum) if k_sim_sum > 0.0 else 1.0
            ring_ratios = np.array([_clip_ratio(float(a) / float(b)) if b > 0.0 else 1.0 for a, b in zip(target.k_own, summary.k_own)])
            raw = np.asarray(current.ring_rate_factors, dtype=np.float64) * np.sqrt(ring_ratios / f_common)
            norm = float(np.mean(raw))
            f_lambda = f_common * norm
            kw = {"clusters_per_um": current.clusters_per_um * f_lambda, "ring_rate_factors": tuple(float(v) for v in raw / norm)}
        f_dec = 1.0
        if dec_real is not None and current.n_locs_per_cluster_quantiles is not None and summary.sizes.size >= 9:
            dec_sim = np.percentile(summary.sizes, 100.0 * DECILE_PROBABILITIES)
            if v2:
                gm_theta = _geo_mean(np.asarray(current.n_locs_per_cluster_quantiles, dtype=np.float64)[CALIBRATION_DECILE_FROM:])
                gm_sim = _geo_mean(dec_sim[CALIBRATION_DECILE_FROM:])
                if math.isfinite(gm_theta) and math.isfinite(gm_sim):
                    binding_dec.append((gm_theta, gm_sim))
            ratios = [float(a) / float(b) for a, b in zip(dec_real[CALIBRATION_DECILE_FROM:], dec_sim[CALIBRATION_DECILE_FROM:]) if b > 0.0 and a > 0.0]
            if ratios:
                f_dec = _clip_ratio(float(np.exp(np.mean(np.log(ratios)))))
            new_dec = np.asarray(current.n_locs_per_cluster_quantiles, dtype=np.float64) * f_dec
            kw["n_locs_per_cluster_quantiles"] = tuple(float(v) for v in new_dec)
            kw["n_fluor_mean"] = float(np.mean(new_dec) / current.locs_per_fluor_mean)
        f_dmin = 1.0
        q5_sim = float(np.percentile(summary.gaps_nm, 5.0)) if summary.gaps_nm.size >= 20 else float("nan")
        if math.isfinite(q5_real) and math.isfinite(q5_sim) and q5_sim > 0.0 and current.d_min_nm > 0.0:
            # a half step (square root of the ratio): the 5th percentile of ~100-200 gaps is noisy (+/-30 % per
            # iteration on a realistic axon) and a full step made d_min wander between 53 and 108 for 80
            f_dmin = math.sqrt(_clip_ratio(q5_real / q5_sim))
            kw["d_min_nm"] = current.d_min_nm * f_dmin
        f_bg = 1.0
        if current.background_per_um3 > 0.0 and summary.background_per_um3 > 0.0:
            f_bg = _clip_ratio(target.background_per_um3 / summary.background_per_um3)
            kw["background_per_um3"] = current.background_per_um3 * f_bg
        f_lpf = 1.0
        if (locs_per_event_target is not None and lpe_sim and math.isfinite(float(locs_per_event_target))
                and float(locs_per_event_target) > 0.0 and float(np.mean(lpe_sim)) > 0.0):
            # a full step: the pooled n_locs / n_events of two axons of thousands of events is precise to ~1 %
            f_lpf = _clip_ratio(float(locs_per_event_target) / float(np.mean(lpe_sim)))
            new_lpf = max(1.0, float(current.locs_per_fluor_mean) * f_lpf)
            kw["locs_per_fluor_mean"] = new_lpf
            dec_now = kw.get("n_locs_per_cluster_quantiles", current.n_locs_per_cluster_quantiles)
            if dec_now is not None:
                kw["n_fluor_mean"] = float(np.mean(np.asarray(dec_now, dtype=np.float64)) / new_lpf)
            else:
                kw["n_fluor_mean"] = float(current.n_fluor_mean) * float(current.locs_per_fluor_mean) / new_lpf
        scatter_pooled = float(math.sqrt(float(np.mean(np.square(scatter_sim))))) if scatter_sim else float("nan")
        if radial_scatter_target_nm is not None and math.isfinite(scatter_pooled) and math.isfinite(float(radial_scatter_target_nm)):
            # half a step in variance (as d_min's square root): the pooled scatter of two axons is noisy (~5 % in sd,
            # ~10 % in variance), and a full step drove the sd of the M1 leak axon from 2.7 to 0 in the last iteration
            var_new = float(current.radial_offset_sd_nm) ** 2 + 0.5 * (float(radial_scatter_target_nm) ** 2 - scatter_pooled ** 2)
            kw["radial_offset_sd_nm"] = float(math.sqrt(max(var_new, 0.0)))
        f1_record: Dict[str, Any] = {}
        if v2:
            _closure_f1(current, kw, nena_target, nena_sim, sigma_z_target_nm, sz_sim, f1_record)
        trajectory.append({
            "iteration": it, "seed": seeds[0], "seeds": seeds, "n_locs_simulated": n_locs_sim, "K_fixed_per_ring": int(k_fixed),
            "K_kept_simulated": [[int(v) for v in s.k_kept] for s in summaries],
            "K_own_simulated": [[int(v) for v in s.k_own] for s in summaries],
            "deciles_simulated": [float(v) for v in (np.percentile(summary.sizes, 100.0 * DECILE_PROBABILITIES) if summary.sizes.size >= 9 else [])],
            "gap_q5_simulated_nm": q5_sim, "background_simulated_per_um3": float(summary.background_per_um3),
            "factor_lambda": f_lambda, "factor_deciles": f_dec, "factor_d_min": f_dmin, "factor_background": f_bg,
            "clusters_per_um": float(kw["clusters_per_um"]),
            "n_locs_per_cluster_quantiles": [float(v) for v in kw.get("n_locs_per_cluster_quantiles", current.n_locs_per_cluster_quantiles or ())],
            "d_min_nm": float(kw.get("d_min_nm", current.d_min_nm)),
            "background_per_um3": float(kw.get("background_per_um3", current.background_per_um3)),
        })
        if current.ring_rate_factors is not None:
            trajectory[-1]["ring_rate_factors"] = [float(v) for v in kw.get("ring_rate_factors", current.ring_rate_factors)]
        if locs_per_event_target is not None:
            trajectory[-1].update({"locs_per_event_simulated": [float(v) for v in lpe_sim],
                                   "locs_per_event_target": float(locs_per_event_target), "factor_locs_per_fluor": float(f_lpf),
                                   "locs_per_fluor_mean": float(kw.get("locs_per_fluor_mean", current.locs_per_fluor_mean)),
                                   "n_fluor_mean": float(kw.get("n_fluor_mean", current.n_fluor_mean))})
        if radial_scatter_target_nm is not None:
            trajectory[-1].update({"radial_scatter_simulated_nm": [float(v) for v in scatter_sim],
                                   "radial_scatter_target_nm": float(radial_scatter_target_nm),
                                   "radial_offset_sd_nm": float(kw.get("radial_offset_sd_nm", current.radial_offset_sd_nm))})
        if v2:
            trajectory[-1].update({"closure": "v2", **f1_record})
        current = dataclasses.replace(current, **kw)
    if v2:
        current = _closure_binding(current, k_real, dec_real, binding_lam, binding_dec, trajectory)
    return current, trajectory


def _geo_mean(values: NDArray[np.float64]) -> float:
    """Geometric mean of the positive finite values (NaN without any)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0.0)]
    return float(np.exp(np.mean(np.log(v)))) if v.size else float("nan")


def _nena_on_strata(sim: "SimAxon", res: RingsResult, target: AxialNena) -> NDArray[np.float64]:
    """The NeNA ratio of one calibration axon on the observed axon's strata
    (same edges, same pair threshold, the axon's own link radius): what the
    estimator SEES on the simulated localizations (closure v2, F1)."""
    edges = np.asarray(target.z_bin_edges_nm, dtype=np.float64)
    if edges.size < 2:
        return np.zeros(0)
    radius = float(res.link_radius_nm) if res.link_radius_nm is not None else float(target.link_radius_nm)
    n = axial_nena(np.asarray(sim.frame, dtype=np.int64), np.asarray(sim.x_nm, dtype=np.float64), np.asarray(sim.y_nm, dtype=np.float64),
                   np.asarray(sim.z_nm, dtype=np.float64), lpz_nm=np.asarray(sim.lpz_nm, dtype=np.float64), link_radius_nm=radius,
                   z_bin_nm=float(edges[1] - edges[0]), min_pairs=int(target.min_pairs), z_range_nm=(float(edges[0]), float(edges[-1])))
    r = np.asarray(n.ratio, dtype=np.float64)
    return r if r.size == edges.size - 1 else np.full(edges.size - 1, np.nan)


def _closure_f1(current: SimConfig, kw: Dict[str, Any], nena_target: Optional[AxialNena], nena_sim: Sequence[NDArray[np.float64]],
                sigma_z_target_nm: Optional[Sequence[float]], sz_sim: Sequence[Sequence[Tuple[float, float, float]]],
                record: Dict[str, Any]) -> None:
    """F1 of closure v2 (in place on ``kw``): the NeNA scale per stratum and
    sigma_struct (see ``_calibrate_by_simulation``)."""
    scale_old = np.asarray(current.axial_scale_by_bin, dtype=np.float64)
    edges = np.asarray(current.lpz_bin_edges_lab_nm, dtype=np.float64)
    f = np.ones_like(scale_old)
    if nena_target is not None and nena_sim:
        obs = np.asarray(nena_target.ratio, dtype=np.float64)
        stack = np.vstack([v for v in nena_sim if v.size == obs.size]) if any(v.size == obs.size for v in nena_sim) else None
        if stack is not None and obs.size == scale_old.size:
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)     # a stratum no calibration axon reached: NaN, scale kept
                sim_mean = np.nanmean(np.where(np.isfinite(stack) & (stack > 0.0), stack, np.nan), axis=0) if stack.size else np.full(obs.size, np.nan)
            for s in range(obs.size):
                if math.isfinite(float(obs[s])) and float(obs[s]) > 0.0 and math.isfinite(float(sim_mean[s])) and float(sim_mean[s]) > 0.0:
                    f[s] = _clip_ratio(float(obs[s]) / float(sim_mean[s]))
            kw["axial_scale_by_bin"] = scale_old * f
            record["nena_ratio_simulated"] = [float(v) for v in sim_mean]
            record["nena_ratio_target"] = [float(v) for v in obs]
        elif stack is not None:
            record["nena_note"] = f"observed strata {obs.size} != configuration strata {scale_old.size}: scale not closed"
        record["factor_nena_scale"] = [float(v) for v in f]
    if sigma_z_target_nm is not None and sz_sim:
        tgt = [float(v) for v in sigma_z_target_nm]
        diffs: List[float] = []
        for k, t in enumerate(tgt):
            per = [a[k] for a in sz_sim if len(a) == len(tgt)]
            vals = []
            for sig, lpz, depth in per:
                if not (math.isfinite(sig) and math.isfinite(t)):
                    continue
                add = 0.0
                if math.isfinite(lpz) and math.isfinite(depth) and edges.size >= 2:
                    st = int(_stratum_of(np.array([depth]), edges)[0])
                    add = (lpz * float(scale_old[st])) ** 2 * (float(f[st]) ** 2 - 1.0)
                vals.append(sig ** 2 + add)
            if vals:
                diffs.append(t ** 2 - float(np.mean(vals)))
        if diffs:
            var_new = float(current.sigma_struct_nm) ** 2 + float(np.mean(diffs))
            kw["sigma_struct_nm"] = float(math.sqrt(max(var_new, 0.0)))
            record["sigma_z_variance_gap_nm2"] = [float(v) for v in diffs]
        record["sigma_struct_nm"] = float(kw.get("sigma_struct_nm", current.sigma_struct_nm))


def _binding_solve(pairs: Sequence[Tuple[float, float]], stat_obs: float) -> Tuple[Optional[float], Dict[str, Any]]:
    """theta solving a + b log theta = log stat_obs by least squares over the
    iterates (None with the reason when it is not usable)."""
    info: Dict[str, Any] = {"n_iterates": len(pairs)}
    ok = [(t, s) for t, s in pairs if t > 0.0 and s > 0.0 and math.isfinite(t) and math.isfinite(s)]
    if len(ok) < BINDING_MIN_ITERATES or not (math.isfinite(stat_obs) and stat_obs > 0.0):
        info["fallback"] = f"{len(ok)} usable iterate(s) (< {BINDING_MIN_ITERATES}) or no observed statistic"
        return None, info
    lt = np.log(np.array([t for t, _ in ok]))
    ls = np.log(np.array([s for _, s in ok]))
    if float(np.ptp(lt)) < 1e-6:
        info["fallback"] = "the iterates did not move the parameter"
        return None, info
    b, a = np.polyfit(lt, ls, 1)
    info.update(slope=float(b), intercept=float(a))
    lo, hi = BINDING_SLOPE_RANGE
    if not (lo <= float(b) <= hi):
        info["fallback"] = f"slope {float(b):.3f} outside [{lo}, {hi}]"
        return None, info
    theta = float(np.exp((math.log(stat_obs) - float(a)) / float(b)))
    info["theta"] = theta
    return theta, info


def _closure_binding(current: SimConfig, k_real: float, dec_real: Optional[NDArray[np.float64]],
                     binding_lam: Sequence[Tuple[float, float]], binding_dec: Sequence[Tuple[float, float]],
                     trajectory: List[Dict[str, Any]]) -> SimConfig:
    """F2 of closure v2: lambda and the decile scale from the binding
    regression (the last iterate when it is refused; see ``_binding_solve``)."""
    rec: Dict[str, Any] = {"closure": "v2", "binding": True}
    kw: Dict[str, Any] = {}
    lo, hi = CALIBRATION_RATIO_CLIP
    theta, info = _binding_solve(binding_lam, k_real)
    if theta is not None and not (lo <= theta / float(current.clusters_per_um) <= hi):
        info["fallback"] = f"solution {theta:.4g} outside the clip of the last iterate {float(current.clusters_per_um):.4g}"
        theta = None
    if theta is not None:
        kw["clusters_per_um"] = theta
    rec["lambda"] = info
    if dec_real is not None and current.n_locs_per_cluster_quantiles is not None:
        dec_now = np.asarray(current.n_locs_per_cluster_quantiles, dtype=np.float64)
        gm_obs = _geo_mean(np.asarray(dec_real, dtype=np.float64)[CALIBRATION_DECILE_FROM:])
        gm_now = _geo_mean(dec_now[CALIBRATION_DECILE_FROM:])
        theta_d, info_d = _binding_solve(binding_dec, gm_obs)
        if theta_d is not None and not (math.isfinite(gm_now) and gm_now > 0.0 and lo <= theta_d / gm_now <= hi):
            info_d["fallback"] = "solution outside the clip of the last iterate"
            theta_d = None
        if theta_d is not None:
            new_dec = dec_now * (theta_d / gm_now)
            kw["n_locs_per_cluster_quantiles"] = tuple(float(v) for v in new_dec)
            kw["n_fluor_mean"] = float(np.mean(new_dec) / float(current.locs_per_fluor_mean))
        rec["deciles"] = info_d
    rec["clusters_per_um"] = float(kw.get("clusters_per_um", current.clusters_per_um))
    rec["n_locs_per_cluster_quantiles"] = [float(v) for v in kw.get("n_locs_per_cluster_quantiles", current.n_locs_per_cluster_quantiles or ())]
    trajectory.append(rec)
    return dataclasses.replace(current, **kw) if kw else current


def scale_axial_leak(cfg: SimConfig, s: float) -> SimConfig:
    """The R-E1 leak family nu(s): ``axial_scale_by_bin`` and
    ``sigma_struct_nm`` multiplied by ``s`` (the axial spread of every
    localization about its ring scales by s, so the axial leak between
    rings moves monotonically with it), ``s`` recorded in the provenance
    as ``leak_scale``. s = 1 returns a configuration equal field by field
    (the provenance gains the record only)."""
    s = float(s)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError(f"scale_axial_leak: s must be finite and > 0, got {s}")
    prov = dict(cfg.provenance)
    prov["leak_scale"] = s
    return dataclasses.replace(cfg, axial_scale_by_bin=np.asarray(cfg.axial_scale_by_bin, dtype=np.float64) * s,
                               sigma_struct_nm=float(cfg.sigma_struct_nm) * s, provenance=prov)


def axially_one_sided_counts(res: RingsResult, period_nm: float) -> List[int]:
    """Axially one-sided (leak-child) clusters per ring, rings in z order
    (``_axially_one_sided`` at ``period_nm``): the leak-trace fidelity row
    of closure v2 (F4) compares these between the observed axon and its
    simulated null."""
    return [int(np.count_nonzero(_axially_one_sided(r, float(period_nm))[0]))
            for r in sorted(res.rings, key=lambda r: float(r.centre_z_nm))]


def axially_one_sided_clusters(res: RingsResult, period_nm: float) -> List[Tuple[int, int]]:
    """(ring index, cluster index in ``Ring.clusters``) of every axially
    one-sided cluster: the ``exclude_clusters`` of the arccr candidate
    (Q-26, D-38d)."""
    out: List[Tuple[int, int]] = []
    for r in res.rings:
        mask = _axially_one_sided(r, float(period_nm))[0]
        out.extend((int(r.index), int(i)) for i in np.flatnonzero(mask))
    return out


def _localization_scatter_nm(res: RingsResult, knot_spacing_nm: Optional[float] = None) -> float:
    """The radial scatter of an axon as ``sim_config_from_axon(membrane_
    source="localizations")`` measures it: every centroid of the rings
    with a contour about their ``localization_membrane_of_rings`` (knots
    every ``knot_spacing_nm``, None = the function's default; rms about
    zero, ``radial_scatter_on_membrane``); NaN when the membrane cannot
    be fitted or no ring has a contour."""
    rings = [r for r in res.rings if r.contour_nm is not None and np.asarray(r.contour_nm).shape[0] >= 4 and r.clusters]
    if not rings:
        return float("nan")
    try:
        if knot_spacing_nm is None:
            membrane = localization_membrane_of_rings(res, [int(r.index) for r in rings])
        else:
            membrane = localization_membrane_of_rings(res, [int(r.index) for r in rings], knot_spacing_nm=float(knot_spacing_nm))
    except ValueError:
        return float("nan")
    cents = np.vstack([np.asarray(cl.centroid_nm, dtype=np.float64).reshape(1, 2) for r in rings for cl in r.clusters])
    return float(radial_scatter_on_membrane(cents, membrane.path)[1])


def sim_config_from_axon(
    res: RingsResult,
    locs: Any,
    *,
    nena: AxialNena,
    reference_ring: Optional[int] = None,
    correct_retained: bool = False,
    calibration_iterations: int = CALIBRATION_ITERATIONS,
    calibration_seed: int = CALIBRATION_SEED,
    rings_params: Optional[RingsParams] = None,
    calibration_axons_per_iteration: int = CALIBRATION_AXONS_PER_ITERATION,
    membrane_source: str = "centroid_pspline",
    contour_smoothing_knot_nm: Optional[float] = None,
    per_ring_rates: bool = False,
    calibrate_locs_per_fluor: bool = False,
    scatter_knot_spacing_nm: Optional[float] = None,
    contour_knot_spacing_nm: Optional[float] = None,
    pick_polygon_nm: Optional[Any] = None,
    closure: str = "v1",
) -> SimConfig:
    """
    The nuisance parameters of ONE real axon as a ``SimConfig`` (02 B8
    step 1; the measurement process and the geometry, never a
    hypothesis statistic). ``locs`` is any object with ``x_nm, y_nm,
    z_nm, frame, lp_lateral_nm, lpz_nm, n_frames`` (``tools.mps_io.
    Localizations``; ``path`` or ``name`` and ``pixel_size_nm`` are
    used when present); ``nena`` is ``axial_nena`` on the laboratory z.

    H5-C (H5C_SPEC S3(a)/(b); D-29c) -- both keywords default to today's
    estimator, value for value:

    * ``membrane_source`` (``MEMBRANE_SOURCES``): with "localizations"
      the contour is the LOCALIZATION membrane of every ring with a
      contour (``tools.mps_membrane.localization_membrane_of_rings``,
      knots every ``MEMBRANE_KNOT_SPACING_NM``, resampled at
      ``MEASURED_CONTOUR_VERTICES`` equal arcs) and ``radial_offset_sd_nm``
      is ``radial_scatter_on_membrane`` of EVERY ring's centroids on
      that curve, pooled over the rings (the rms about zero; the
      per-ring values are in the provenance) -- identifiable on every
      axon, no length-change guard, no fallback 0, and the same curve
      the arc test projects on, so the simulated axons reproduce the
      observed axon's scatter about ITS membrane; the calibration
      (below) then closes that scatter too (``_calibrate_by_simulation(
      radial_scatter_target_nm=...)``: the measured value holds the
      centroid noise and the membrane's estimation error, which the
      simulation adds again; the bare measured value is in the
      provenance as ``radial_offset_sd_measured_nm``). The centroid P-spline
      estimate is still computed and kept in the provenance
      (``radial_offset_sd_centroid_pspline_nm``). The configuration then
      records ``length_basis`` "smoothed" (lambda is calibrated, below,
      on that membrane, which is at its knot scale already) and the
      membrane's knot spacing in ``provenance["contour_smoothing_knot_nm"]``;
      ``SimConfig.contour_smoothing_knot_nm`` stays None unless given
      (nothing to smooth further). If the membrane cannot be fitted the
      centroid estimator is used with a warning.
    * ``contour_smoothing_knot_nm``: recorded in the configuration, so
      that the calibration (below) simulates on the SMOOTHED contour
      and lambda is calibrated on the smoothed length (``length_basis``
      "smoothed"); None with the centroid estimator leaves
      ``length_basis`` None (today's configuration value for value: the
      v2 ``simnull`` smooths the measured contour afterwards with the
      lambda as measured, and the arc-null validator freezes those rows; a
      caller that knows its lambda refers to the contour as measured
      sets ``length_basis`` "raw" itself and ``simulate_axon`` rescales it
      under a later smoothing, ``effective_clusters_per_um``).
    * ``per_ring_rates`` (default False: one rate for every ring, today's
      configuration): ``SimConfig.ring_rate_factors`` starts at each
      ring's K own over their mean (rings in z order) and the calibration
      closes K own RING BY RING -- the per-axon simulated null
      (``simnull`` rows v3) conditions on the axon's ring-to-ring
      differences of K, which one common rate cannot reproduce (the
      comment at the field). A pooled configuration never carries it.
    * ``calibrate_locs_per_fluor`` (default False: the measured n_locs /
      n_events as is, today's configuration): the calibration also
      closes the localizations per fluorophore against the axon's
      measured n_locs / n_events (``_calibrate_by_simulation(
      locs_per_event_target=...)``: the linking splits events, so the
      measured ratio is biased low and a simulator run at it
      under-produces the leak; review of 2026-09-26). The measured value
      stays in the provenance (``locs_per_event_measured``).
    * ``scatter_knot_spacing_nm`` (default None: the scatter is closed as
      measured about the 600 nm membrane, today's configuration; with
      ``membrane_source`` "localizations" only): the radial scatter the
      calibration closes is the one measured about the localization
      membrane with THESE knots -- ``simnull`` passes the arc test's
      400 nm (``tools.mps_unroll.ARC_LOCALIZATION_KNOT_SPACING_NM``), so
      that the scatter its fidelity table compares (``arcl_radial_scatter_nm``,
      on the arc test's curve) is the quantity the simulator was closed
      on. The contour stays the 600 nm membrane; the target and its
      knots are in the provenance (review of 2026-09-26: on the M1 leak
      axon, whose clusters are ON the membrane, the 400 nm reading of
      the pure centroid noise was 1.9 nm against 2.6 in simulations
      closed at 600 nm).
    * ``contour_knot_spacing_nm`` (default None: the contour is the
      localization membrane at its own ``MEMBRANE_KNOT_SPACING_NM``,
      today's configuration; with ``membrane_source`` "localizations"
      only; re-review of 2026-09-27): the CONTOUR is the localization
      membrane fitted with THESE knots instead (resampled every
      ``CONTOUR_MEMBRANE_VERTEX_SPACING_NM``, at least
      ``MEASURED_CONTOUR_VERTICES`` vertices, so that the simulator's
      interpolating curve keeps the finer structure), while the radial
      scatter stays the one about the 600 nm membrane and is closed by
      the calibration as before. Why: the arc test's 400 nm membrane
      bridges a narrow inward notch of the membrane (notch-shaped
      contours: the clusters of both
      walls project on the bridge at nearly the same arcs), which makes
      the test liberal on such an axon even without leak;
      a null simulated on the 600 nm membrane has no notch and
      re-simulates a level test (a research probe on library contours),
      whereas a contour at 200 nm follows the notch and its simulations
      carry the bias, without creating
      one where there is none (ellipse +0.05). ``power_columns.py
      simnull`` passes ``SIMNULL_MEMBRANE_CONTOUR_KNOT_NM``.
    * ``pick_polygon_nm`` (H5-D, D-32c; default None: the box
      background, today's configuration): the axon's real pick in the
      axon frame (``pick_polygon_of_rings``) becomes
      ``SimConfig.pick_polygon_nm`` BEFORE the calibration, so that the
      calibration axons -- and the background density it closes on the
      real axon's (``_background_per_um3``, the same pick on both sides)
      -- are simulated inside the real pick; the provenance records the
      polygon ("background_model" "pick", its vertices and area) and the
      lambda convention (``LAMBDA_CONVENTION``, K own, D-32c).

    * contour and radial scatter (``radial_scatter_nm``): the reference
      ring's (default: the middle ring; the nearest ring with a contour
      otherwise) MEMBRANE, the periodic B-spline with knots every
      ``MEMBRANE_KNOT_SPACING_NM`` fitted by least squares to its
      centroids, resampled at ``MEASURED_CONTOUR_VERTICES``;
      radial_offset_sd = the dof-corrected residual sd of the centroids
      about their ring's fitted membrane, pooled over the rings
      (variances weighted by their degrees of freedom). The
      specification's quantities -- the s = K smoothing spline as the
      contour and the sd of the centroids' offsets from it -- are
      recorded in the provenance (``radial_offset_sd_s_k_nm``) and NOT
      used: that spline passes within 1 nm rms of the centroids by
      construction, so the offset sd came out 0.97-0.98 nm for a true
      scatter of 15 nm (review of 2026-09-25) and every simulated
      cluster would sit ON the membrane. A ring whose fit is not
      identifiable (K - n_knots < 3) keeps the s = K contour and
      contributes no scatter.
    * period = median spacing of the ring centres (z'); sigma_period =
      sd of the spacings / sqrt(2) (independent per-ring offsets)
      divided by the unbiasing constant c4(n_spacings) (0.80 at n = 2:
      three rings give ONE degree of freedom and the raw sd runs 20 %
      low in expectation; the pooled median over axons is still biased,
      review of 2026-09-25).
    * clusters_per_um = mean K / mean L (``Ring.length_nm``) with K the
      kept clusters that are NOT axially one-sided
      (``_axially_one_sided``: the leak children, which the simulation
      re-creates itself; the specification's "mean K" over every kept
      cluster would count them twice on re-simulation).
    * d_min from the cyclic gaps of those clusters along their ring's
      polygon: the 5th percentile matched under the simulator's own
      hard-core sampler (``_hard_core_by_quantile_matching``; the
      observed minimum is a single draw of the hard core and would set
      d_min from one pair; a child sits at a random arc relative to its
      ring and would break the hard core it does not obey, which is why
      the one-sided clusters are left out). Two true clusters closer
      than DBSCAN separates them are detected as ONE, so the smallest
      gaps are missing and the estimate is an upper bound of the true
      hard core (82-89 nm for 80 on the M1 leak axon of validate_leak).
    * n_locs deciles of the kept, not one-sided clusters as DETECTED (the
      specification's quantity; ``correct_retained`` divides the sizes
      by each ring's ``_retained_fraction`` instead, off by default: the
      Gaussian-window model over-corrected, ninth decile 306 for a truth
      of ~218 on the M1 leak axon), then CALIBRATED (below).
    * epitope_radius_nm and sigma_link_nm from the pooled lateral widths
      of the clusters (``_cluster_lateral_widths``): sigma_link =
      sqrt(max(sd_n^2 - lp^2, 0)) with lp the median lateral precision,
      R = 2 sqrt(max(sd_t^2 - sd_n^2, 0)). The cluster width decides
      whether DBSCAN keeps neighbours at the hard core apart (with the
      default 25 / 12 nm the clusters of an 11 nm-wide axon merged and a
      third of its K was lost on re-simulation), so it must be the
      measured one; the widths are in the provenance.
    * locs_per_fluor_mean = n_locs / n_events over ALL localizations
      (``RingsResult.n_events``, else the NeNA event count): a ring
      window cuts leaked localizations off their events, so the
      per-cluster ratio is biased low.
    * lp samples (all finite lp, quantile-thinned to ``LP_SAMPLES_MAX``)
      and lpz samples per stratum of ``nena.z_bin_edges_nm`` (laboratory
      z; ``LPZ_SAMPLES_PER_BIN_MAX`` each); axial_scale_by_bin =
      ``nena.ratio``, NaN or <= 0 -> 1 with a warning in the provenance.
    * sigma_struct: per ring the SIGNED difference sigma_z^2 - (lpz
      median x scale at the ring's depth)^2 - (R^2 / 4 + sigma_link^2),
      with the scale interpolated between the stratum centres
      (``_scale_at_depth``), the depth the ring's core
      (``_ring_depth_lab_nm``) and R, sigma_link the measured epitope
      disc and linkage (below): the fitted ring width holds the axial
      error, the per-fluorophore spread of the disc (rms R / 2 along z)
      and the linkage, and ``measure`` adds the last two again on top
      of sigma_struct, so leaving them in read 23.5 nm for a truth of 0
      and re-simulated the rings 5-10 % wider than the axon they were
      measured on (re-review of 2026-09-25); sigma_struct = sqrt(max
      (MEAN over the best-resolved rings, 0)), the rings whose error
      width lpz x scale is within ``RESOLVED_LPZ_TOLERANCE`` of the
      smallest. Each ring's difference is a LOWER bound of the
      structural width: lpz x scale is the NeNA-calibrated error width
      (unbiased), whereas the ring's fitted component width is biased
      low when the rings overlap (on a realistic simulated axon the
      fitted width read well below the true axial sd), and a deep ring with
      lpz x scale >= sigma_z resolves nothing; the shallowest ring is
      the tightest bound (close to the truth on two seeds of that
      axon, whose median over rings read far below it) and on that geometry
      it is the only best-resolved ring. The MAX over rings used before
      had a floor where the rings are comparably resolved: at a truth
      of 0 the largest of three zero-mean errors of sd ~15 nm (sqrt)
      read 10-17 nm, whereas their mean is unbiased; the per-ring
      differences and the rings used are in the provenance. The
      specification's median of the per-ring ``structural_width_nm``
      clips each ring at 0 before the median and jumped to 0 whenever a
      ring's depth crossed a stratum edge (review of 2026-09-25).
    * background_per_um3 = the uniform component (``_background_per_um3``:
      unclustered localizations farther than ``BACKGROUND_EXCLUSION_NM``
      from every kept centroid, per free volume of the pick), then
      CALIBRATED; background_margin_nm = the pick's margin around the
      membrane (``_pick_margin_nm``: the median distance between the
      bounding boxes of the rings' localizations and of the contour on
      the four sides), which is where the simulated background box
      ends, so that the pipeline's edge criterion sees on the simulated
      axon the hull the real one had (re-review of 2026-09-25).
    * tilt = ``res.frame.beta_deg`` with its azimuth; z_middle_lab = the
      depth of the middle ring's core.
    * CALIBRATION (``_calibrate_by_simulation``, ``calibration_iterations``
      times over ``calibration_axons_per_iteration`` simulated axons
      from ``calibration_seed`` with ``rings_params`` -- default
      ``res.params`` with at most 20 bootstrap draws): clusters_per_um,
      the n_locs deciles (hence n_fluor_mean) and the background are
      rescaled so that the simulator, run through the same
      ``build_rings``, re-detects the real K own per ring, the real
      detected deciles and the real background density (02 B8 step 1).
      0 iterations returns the bare measurement.
    * provenance: dataset_role "exploratory", date, code_commit, file,
      n_locs, n_events, n_rings, reference ring, K kept and K own per
      ring with the one-sided counts and thresholds, the membrane fits
      (knots, dof, sigma per ring), the retained fractions, the cluster
      widths, the bare (uncalibrated) values, the calibration
      trajectory and the warnings.
    """
    rings = sorted(res.rings, key=lambda r: float(r.centre_z_nm))
    if len(rings) < 2:
        raise ValueError(f"sim_config_from_axon: at least two rings are needed, got {len(rings)}")
    warnings_: List[str] = []
    x_lab = np.asarray(locs.x_nm, dtype=np.float64).reshape(-1)
    z_lab = np.asarray(locs.z_nm, dtype=np.float64).reshape(-1)
    z_p = np.asarray(res.z_p, dtype=np.float64).reshape(-1)
    n_locs = int(x_lab.size)
    centres = np.array([float(r.centre_z_nm) for r in rings])
    spacings = np.diff(centres)
    period = float(np.median(spacings))
    sigma_period = float(np.std(spacings, ddof=1) / math.sqrt(2.0) / _c4(int(spacings.size))) if spacings.size >= 2 else 0.0
    if not (math.isfinite(period) and period > 0.0):
        raise ValueError(f"sim_config_from_axon: the ring centres give no positive period ({centres.tolist()})")
    with_contour = [r for r in rings if r.contour_nm is not None and np.asarray(r.contour_nm).shape[0] >= 4]
    middle = len(rings) // 2
    ref: Optional[Ring] = None
    if reference_ring is not None:
        cands = [r for r in with_contour if int(r.index) == int(reference_ring)]
        if not cands:
            raise ValueError(f"sim_config_from_axon: reference ring {reference_ring} has no contour")
        ref = cands[0]
    elif with_contour:
        ref = min(with_contour, key=lambda r: abs(rings.index(r) - middle))
    # membrane fits: the scatter of every ring with a contour, the contour of the reference ring
    fits: Dict[int, MembraneFit] = {}
    for r in with_contour:
        try:
            fits[id(r)] = radial_scatter_nm(np.asarray(r.contour_nm, dtype=np.float64))
        except ValueError as exc:
            warnings_.append(f"ring {r.index}: no membrane fit ({exc})")
    contour: Optional[NDArray[np.float64]] = None
    if ref is not None:
        fit_ref = fits.get(id(ref))
        if fit_ref is not None and math.isfinite(fit_ref.sigma_nm):
            contour = np.array(fit_ref.curve_nm, dtype=np.float64)
        else:
            warnings_.append(f"ring {ref.index}: the membrane fit is not identifiable; the s = K smoothing spline is the contour")
            contour = _smoothed_closed_contour(np.asarray(ref.contour_nm), MEASURED_CONTOUR_VERTICES, warnings_, f"ring {ref.index}")
    else:
        warnings_.append("no ring has a contour: the ellipse will be used")
    var_num = 0.0
    var_den = 0
    for fit in fits.values():
        if math.isfinite(fit.sigma_nm):
            var_num += fit.sigma_nm ** 2 * fit.dof
            var_den += fit.dof
    radial_sd = math.sqrt(var_num / var_den) if var_den > 0 else 0.0
    if var_den == 0:
        warnings_.append("no identifiable membrane fit: radial_offset_sd left at 0")
    # H5-C: the membrane from the localizations as the contour and the scatter of every ring's centroids about it.
    if str(membrane_source) not in MEMBRANE_SOURCES:
        raise ValueError(f"sim_config_from_axon: membrane_source must be one of {list(MEMBRANE_SOURCES)}, got {membrane_source!r}")
    knot_given = None if contour_smoothing_knot_nm is None else float(contour_smoothing_knot_nm)
    if knot_given is not None and not (math.isfinite(knot_given) and knot_given > 0.0):
        raise ValueError(f"sim_config_from_axon: contour_smoothing_knot_nm must be positive and finite, got {contour_smoothing_knot_nm}")
    radial_sd_centroid = radial_sd
    scatter_target: Optional[float] = None
    membrane_loc: Optional[LocalizationMembrane] = None
    contour_knot_used: Optional[float] = None
    contour_length_used = float("nan")
    scatter_by_ring: List[Dict[str, Any]] = []
    # None with the defaults: today's configuration value for value (simnull's v2 rows, frozen by the arc-null validator,
    # smooth the measured contour afterwards and must keep simulating with the lambda as measured).
    length_basis: Optional[str] = None if knot_given is None else "smoothed"
    knot_recorded: Optional[float] = knot_given
    if str(membrane_source) == "localizations" and with_contour:
        ids_loc = [int(r.index) for r in with_contour]
        try:
            membrane_loc = localization_membrane_of_rings(res, ids_loc)
        except ValueError as exc:
            warnings_.append(f"localization membrane of rings {ids_loc} not fitted ({exc}): the centroid P-spline contour and "
                             "scatter are used")
        if membrane_loc is not None:
            contour = _resample_closed_path(membrane_loc.path, MEASURED_CONTOUR_VERTICES)
            if contour_knot_spacing_nm is not None:
                # Re-review of 2026-09-27: the contour at finer knots than the membrane's (the docstring's argument);
                # the scatter below stays about the membrane's own 600 nm curve.
                knot_c = float(contour_knot_spacing_nm)
                if not (math.isfinite(knot_c) and knot_c > 0.0):
                    raise ValueError(f"sim_config_from_axon: contour_knot_spacing_nm must be positive and finite, got {contour_knot_spacing_nm}")
                try:
                    membrane_c = localization_membrane_of_rings(res, ids_loc, knot_spacing_nm=knot_c)
                    n_vert = max(int(MEASURED_CONTOUR_VERTICES),
                                 int(math.ceil(float(membrane_c.length_nm) / float(CONTOUR_MEMBRANE_VERTEX_SPACING_NM))))
                    contour = _resample_closed_path(membrane_c.path, n_vert)
                    contour_knot_used = knot_c
                    contour_length_used = float(membrane_c.length_nm)
                    warnings_.extend(f"localization membrane of the contour ({knot_c:g} nm knots): {w}" for w in membrane_c.warnings)
                except ValueError as exc:
                    warnings_.append(f"localization membrane at {knot_c:g} nm knots not fitted ({exc}): the contour is the "
                                     f"{float(membrane_loc.knot_spacing_nm):g} nm membrane")
            cents_loc = np.vstack([np.asarray(cl.centroid_nm, dtype=np.float64).reshape(1, 2) for r in with_contour for cl in r.clusters])
            _off_all, radial_sd = radial_scatter_on_membrane(cents_loc, membrane_loc.path)
            for r in with_contour:
                c_r = np.vstack([np.asarray(cl.centroid_nm, dtype=np.float64).reshape(1, 2) for cl in r.clusters]) if r.clusters else np.zeros((0, 2))
                off_r, sig_r = radial_scatter_on_membrane(c_r, membrane_loc.path)
                scatter_by_ring.append({"ring": int(r.index), "K": int(off_r.size), "sigma_nm": float(sig_r),
                                        "mean_offset_nm": float(np.mean(off_r)) if off_r.size else float("nan")})
            length_basis = "smoothed"
            if knot_recorded is None:
                knot_recorded = float(membrane_loc.knot_spacing_nm) if contour_knot_used is None else contour_knot_used
            if scatter_knot_spacing_nm is not None:
                try:
                    m_sc = localization_membrane_of_rings(res, ids_loc, knot_spacing_nm=float(scatter_knot_spacing_nm))
                    scatter_target = float(radial_scatter_on_membrane(cents_loc, m_sc.path)[1])
                except ValueError as exc:
                    warnings_.append(f"localization membrane at {scatter_knot_spacing_nm} nm not fitted ({exc}): the scatter is "
                                     "closed at the membrane's own knots")
            warnings_.extend(f"localization membrane: {w}" for w in membrane_loc.warnings)
    if contour_knot_spacing_nm is not None and str(membrane_source) != "localizations":
        warnings_.append(f"contour_knot_spacing_nm {contour_knot_spacing_nm} ignored: it applies to membrane_source 'localizations' only")
    margin, margin_sides = _pick_margin_nm(res, rings, contour)
    if not margin_sides:
        warnings_.append(f"no contour or localizations to measure the pick's margin: {margin:.0f} nm assumed")
    detected = _detected_summary(res, period)
    one_sided = {id(r): m for r, m in zip(detected.rings, detected.one_sided)}
    thresholds = detected.thresholds
    ks_kept, ks, lengths = detected.k_kept, detected.k_own, detected.lengths
    clusters_per_um = float(ks.mean() / (lengths.mean() / 1000.0)) if lengths.size else 4.0
    if not lengths.size:
        warnings_.append("no ring length: clusters_per_um left at 4")
    # hard core (among the clusters of the ring itself) and D-25's s = K offsets (for the record)
    nn: List[NDArray[np.float64]] = [detected.gaps_nm] if detected.gaps_nm.size else []
    offsets_sk: List[NDArray[np.float64]] = []
    for r in with_contour:
        c = np.asarray(r.contour_nm, dtype=np.float64)
        if len(r.clusters) >= 4:
            sm = _smoothed_closed_contour(c, 2000, warnings_, f"ring {r.index}")
            try:
                path = smooth_closed_path(sm)
            except ValueError as exc:
                warnings_.append(f"ring {r.index}: no smoothed curve ({exc})")
                continue
            cents = np.vstack([np.asarray(cl.centroid_nm, dtype=np.float64) for cl in r.clusters])
            _s, rr = project_on_path(path, cents, outward_sign=_outward_sign(path))
            offsets_sk.append(np.asarray(rr, dtype=np.float64))
    d_min = _hard_core_by_quantile_matching(np.concatenate(nn), float(lengths.mean()), float(ks.mean())) if nn and lengths.size else 80.0
    if not nn:
        warnings_.append("no contour: d_min left at 80 nm")
    off_sk = np.concatenate(offsets_sk) if offsets_sk else np.zeros(0)
    radial_sd_sk = float(np.std(off_sk, ddof=1)) if off_sk.size >= 2 else float("nan")
    # strata of the laboratory z
    edges = np.asarray(nena.z_bin_edges_nm, dtype=np.float64)
    ratio = np.asarray(nena.ratio, dtype=np.float64)
    finite = np.isfinite(ratio) & (ratio > 0.0)
    scale = np.where(finite, ratio, 1.0)
    if not finite.all():
        warnings_.append(f"NeNA ratio NaN or <= 0 in strata {np.flatnonzero(~finite).tolist()}: scale 1 used there")
    # precisions and the cluster widths (the disc and the linkage enter the structural width below)
    lp_all = np.asarray(locs.lp_lateral_nm, dtype=np.float64).reshape(-1)
    lp_all = lp_all[np.isfinite(lp_all) & (lp_all > 0.0)]
    lp_samples = _thin_quantiles(lp_all, LP_SAMPLES_MAX) if lp_all.size else np.array([8.0])
    if not lp_all.size:
        warnings_.append("no lateral precision: 8 nm assumed")
    lp_median = float(np.median(lp_all)) if lp_all.size else 8.0
    sd_t, sd_n, n_width = _cluster_lateral_widths(res, rings)
    if n_width > 0 and math.isfinite(sd_t) and math.isfinite(sd_n):
        sigma_link = math.sqrt(max(sd_n ** 2 - lp_median ** 2, 0.0))
        epitope_radius = 2.0 * math.sqrt(max(sd_t ** 2 - sd_n ** 2, 0.0))
    else:
        sigma_link, epitope_radius = 12.0, 25.0
        warnings_.append("no cluster width measurable: the default epitope disc and linkage are used")
    fluor_axial_var = epitope_radius ** 2 / 4.0 + sigma_link ** 2
    # cluster sizes, structural width, ring depths
    sizes: List[NDArray[np.float64]] = []
    retained: List[float] = []
    struct_diff: List[float] = []
    depths: List[float] = []
    lpz_eff: List[float] = []
    for r in rings:
        depth = _ring_depth_lab_nm(r, z_lab, z_p, period)
        depths.append(depth)
        fr = _retained_fraction(r) if correct_retained else 1.0
        fr = fr if fr > 0.0 else 1.0
        retained.append(fr)
        own = [cl for cl, one in zip(r.clusters, one_sided[id(r)]) if not one]
        if own:
            sizes.append(np.array([float(cl.n_locs) for cl in own]) / fr)
        sig, lpz_r = float(r.sigma_z_nm), float(r.lpz_median_nm)
        if math.isfinite(sig) and math.isfinite(lpz_r) and lpz_r > 0.0:
            eff = lpz_r * _scale_at_depth(edges, scale, depth)
            lpz_eff.append(eff)
            struct_diff.append(sig ** 2 - eff ** 2 - fluor_axial_var)
        else:
            lpz_eff.append(float("nan"))
    all_sizes = np.concatenate(sizes) if sizes else np.zeros(0)
    deciles: Optional[Tuple[float, ...]] = None
    if all_sizes.size:
        deciles = tuple(float(v) for v in np.percentile(all_sizes, 100.0 * DECILE_PROBABILITIES))
    n_events = int(res.n_events) if res.n_events is not None else int(nena.n_events_total)
    if n_events > 0:
        locs_per_fluor = max(1.0, n_locs / n_events)
    else:
        locs_per_fluor = 2.0
        warnings_.append("no event count: locs_per_fluor_mean left at 2")
    n_fluor_mean = float(all_sizes.mean() / locs_per_fluor) if all_sizes.size else 25.0
    struct_rings: List[int] = []
    if struct_diff:
        eff_ok = np.array([v for v in lpz_eff if math.isfinite(v)], dtype=np.float64)
        best = eff_ok <= (1.0 + RESOLVED_LPZ_TOLERANCE) * float(eff_ok.min())
        struct_rings = [int(r.index) for r, ok in zip([r for r, v in zip(rings, lpz_eff) if math.isfinite(v)], best) if ok]
        sigma_struct = math.sqrt(max(float(np.mean(np.asarray(struct_diff, dtype=np.float64)[best])), 0.0))
    else:
        sigma_struct = 0.0
    background = float(detected.background_per_um3)
    lpz_all = np.asarray(locs.lpz_nm, dtype=np.float64).reshape(-1)
    ok = np.isfinite(lpz_all) & (lpz_all > 0.0) & np.isfinite(z_lab)
    strat = _stratum_of(z_lab[ok], edges)
    lpz_by_bin = [_thin_quantiles(lpz_all[ok][strat == s], LPZ_SAMPLES_PER_BIN_MAX) for s in range(edges.size - 1)]
    if not any(v.size for v in lpz_by_bin):
        warnings_.append("no axial precision: 50 nm assumed in every stratum")
        lpz_by_bin = [np.array([50.0]) for _ in range(edges.size - 1)]
    # frame
    beta = float(res.frame.beta_deg)
    u = np.asarray(res.frame.u, dtype=np.float64)
    azimuth = float(math.degrees(math.atan2(u[1], u[0]))) if beta > 1e-9 else 0.0
    z_middle = depths[middle]
    pixel = getattr(locs, "pixel_size_nm", None)
    name = getattr(locs, "path", None) or getattr(locs, "name", None) or ""
    provenance: Dict[str, Any] = {
        "dataset_role": "exploratory", "date": datetime.date.today().isoformat(), "code_commit": _git_commit(),
        "file": os.path.basename(str(name)) if name else "", "n_locs": n_locs, "n_events": n_events,
        "n_rings": len(rings), "reference_ring": None if ref is None else int(ref.index),
        "K_kept_per_ring": [int(k) for k in ks_kept], "K_own_per_ring": [int(k) for k in ks],
        "n_one_sided_per_ring": [int(np.count_nonzero(one_sided[id(r)])) for r in rings],
        "one_sided_threshold_nm": [float(v) for v in thresholds], "length_per_ring_nm": [float(v) for v in lengths],
        "ring_depth_lab_nm": [float(v) for v in depths], "retained_fraction_by_ring": [float(v) for v in retained],
        "ring_sigma_z_nm": [float(r.sigma_z_nm) for r in rings], "ring_lpz_effective_nm": [float(v) for v in lpz_eff],
        "ring_struct_variance_signed_nm2": [float(v) for v in struct_diff], "fluorophore_axial_variance_nm2": float(fluor_axial_var),
        "struct_rings_used": struct_rings, "resolved_lpz_tolerance": float(RESOLVED_LPZ_TOLERANCE),
        "pick_margin_sides_nm": [float(v) for v in margin_sides],
        "membrane_fit_per_ring": [{"ring": int(r.index), "n_knots": int(fits[id(r)].n_knots), "dof": int(fits[id(r)].dof),
                                   "sigma_nm": float(fits[id(r)].sigma_nm)} for r in with_contour if id(r) in fits],
        "membrane_knot_spacing_nm": float(MEMBRANE_KNOT_SPACING_NM), "radial_offset_sd_s_k_nm": radial_sd_sk,
        "n_locs_outside_clusters": int(sum(int(r.n_locs) - int(sum(int(cl.n_locs) for cl in r.clusters)) for r in rings)),
        "n_locs_far_background": int(detected.n_far), "background_excluded_fraction": float(detected.excluded_fraction),
        "background_exclusion_nm": float(BACKGROUND_EXCLUSION_NM), "retained_correction_applied": bool(correct_retained),
        "cluster_sd_tangent_nm": float(sd_t), "cluster_sd_normal_nm": float(sd_n), "n_locs_in_clusters": int(n_width),
        "lp_median_nm": lp_median, "pixel_size_nm": None if pixel is None else float(pixel),
        "pixel_size_source": str(getattr(locs, "pixel_size_source", "unknown")),
        "measured_uncalibrated": {"clusters_per_um": clusters_per_um, "n_locs_per_cluster_quantiles": list(deciles or ()),
                                  "n_fluor_mean": n_fluor_mean, "background_per_um3": background, "d_min_nm": d_min,
                                  "gap_q5_nm": float(np.percentile(detected.gaps_nm, 5.0)) if detected.gaps_nm.size else float("nan")},
        "calibration": [], "warnings": warnings_,
        "note": "nuisance parameters of the measurement process and the geometry only; no hypothesis statistic",
    }
    # The H5-C provenance only when an H5-C option is used (review of 2026-09-26: with the default keywords the
    # configuration, provenance included, is the one this function returned before H5-C).
    if (str(membrane_source) != "centroid_pspline" or knot_given is not None or per_ring_rates or calibrate_locs_per_fluor):
        provenance.update({
            "membrane_source": str(membrane_source), "length_basis": length_basis, "contour_smoothing_knot_nm": knot_recorded,
            "radial_offset_sd_centroid_pspline_nm": float(radial_sd_centroid),
            "radial_offset_sd_measured_nm": float(radial_sd),
            "membrane_localizations": (None if membrane_loc is None else {
                "n_points": int(membrane_loc.n_points), "n_used": int(membrane_loc.n_used), "n_knots": int(membrane_loc.n_knots),
                "knot_spacing_nm": float(membrane_loc.knot_spacing_nm), "length_nm": float(membrane_loc.length_nm),
                "residual_sd_nm": float(membrane_loc.residual_sd_nm), "dof_fit": float(membrane_loc.dof_fit),
                "iterations": int(membrane_loc.iterations), "initial_curve": str(membrane_loc.initial_curve),
                "initial_order": str(membrane_loc.initial_order), "n_self_crossings": int(membrane_loc.n_self_crossings)}),
            "radial_scatter_on_membrane_by_ring": scatter_by_ring,
        })
        if contour_knot_spacing_nm is not None:
            provenance.update({"contour_knot_spacing_nm": contour_knot_used, "contour_knot_spacing_requested_nm": float(contour_knot_spacing_nm),
                               "contour_membrane_length_nm": contour_length_used})
        if scatter_target is not None:
            provenance.update({"radial_scatter_target_nm": float(scatter_target),
                               "scatter_knot_spacing_nm": float(scatter_knot_spacing_nm or 0.0)})
    cfg = SimConfig(
        contour_nm=contour, n_rings=len(rings), period_nm=period, sigma_period_nm=sigma_period, z_middle_lab_nm=z_middle,
        clusters_per_um=clusters_per_um, n_clusters_per_ring=None, d_min_nm=d_min, radial_offset_sd_nm=radial_sd,
        model="M1", n_fluor_mean=n_fluor_mean, n_locs_per_cluster_quantiles=deciles, locs_per_fluor_mean=locs_per_fluor,
        epitope_radius_nm=epitope_radius, sigma_link_nm=sigma_link,
        n_frames=int(locs.n_frames), lp_lateral_samples_nm=lp_samples, lpz_bin_edges_lab_nm=edges,
        lpz_samples_by_bin_nm=lpz_by_bin, axial_scale_by_bin=np.asarray(scale, dtype=np.float64),
        sigma_struct_nm=sigma_struct, background_per_um3=background, background_margin_nm=margin, tilt_deg=beta, azimuth_deg=azimuth,
        pixel_size_nm=float(pixel) if pixel is not None and float(pixel) > 0.0 else 113.0, provenance=provenance,
        contour_smoothing_knot_nm=knot_given, length_basis=length_basis)
    if pick_polygon_nm is not None:
        # H5-D (D-32c): the background inside the real pick, for the calibration axons too; the lambda convention written
        pick = _as_pick_polygon(pick_polygon_nm)
        cfg = dataclasses.replace(cfg, pick_polygon_nm=pick)
        cfg.provenance.update({
            "background_model": "pick", "pick_polygon_n_vertices": int(pick.shape[0]),
            "pick_polygon_area_um2": float(abs(_signed_area_nm2(pick)) / 1e6),
            "lambda_convention": LAMBDA_CONVENTION, "lambda_convention_note": LAMBDA_CONVENTION_NOTE,
        })
    if per_ring_rates:
        k_mean = float(np.mean(ks)) if ks.size else 0.0
        if ks.size == len(rings) and k_mean > 0.0 and bool(np.all(ks > 0.0)):
            cfg = dataclasses.replace(cfg, ring_rate_factors=tuple(float(v) for v in ks / k_mean))
            cfg.provenance["ring_rate_factors_measured"] = [float(v) for v in ks / k_mean]
        else:
            warnings_.append(f"per_ring_rates: K own per ring {ks.tolist()} leaves a ring without clusters; one common rate kept")
    if closure not in CLOSURE_VERSIONS:
        raise ValueError(f"closure must be one of {CLOSURE_VERSIONS}, got {closure!r}")
    closure_kw: Dict[str, Any] = {}
    if closure == "v2" and int(calibration_iterations) > 0:
        # H5-E (D-40, spec 2.2): 4 iterations x 6 axons with common random numbers, F1 on the observed NeNA and sigma_z
        calibration_iterations = CALIBRATION_V2_ITERATIONS
        calibration_axons_per_iteration = CALIBRATION_V2_AXONS
        closure_kw = dict(closure="v2", nena_target=nena, sigma_z_target_nm=[float(r.sigma_z_nm) for r in detected.rings])
        cfg.provenance["closure"] = "v2"
    if int(calibration_iterations) > 0:
        params = rings_params if rings_params is not None else dataclasses.replace(
            res.params, n_bootstrap=min(int(res.params.n_bootstrap), 20))
        cfg, trajectory = _calibrate_by_simulation(cfg, detected, int(locs.n_frames), params, int(calibration_iterations),
                                                   int(calibration_seed), warnings_, axons_per_iteration=int(calibration_axons_per_iteration),
                                                   **closure_kw,
                                                   radial_scatter_target_nm=((float(radial_sd) if scatter_target is None else scatter_target)
                                                                             if membrane_loc is not None else None),
                                                   scatter_knot_spacing_nm=(None if scatter_target is None else float(scatter_knot_spacing_nm or 0.0)),
                                                   locs_per_event_target=(float(locs_per_fluor) if calibrate_locs_per_fluor and n_events > 0 else None))
        cfg.provenance["calibration"] = trajectory
        if calibrate_locs_per_fluor:
            cfg.provenance["locs_per_event_measured"] = float(locs_per_fluor)
        cfg.provenance["calibration_rings_params"] = {f.name: _plain(getattr(params, f.name)) for f in dataclasses.fields(params)}
    return cfg


_POOLED_MEDIAN_FIELDS: Tuple[str, ...] = (
    "period_nm", "sigma_period_nm", "z_middle_lab_nm", "clusters_per_um", "d_min_nm", "radial_offset_sd_nm",
    "n_fluor_mean", "locs_per_fluor_mean", "epitope_radius_nm", "sigma_link_nm", "sigma_struct_nm",
    "background_per_um3", "background_margin_nm", "tilt_deg", "azimuth_deg", "pixel_size_nm", "sigma_col_nm", "jitter_nm",
)


def pool_sim_configs(configs: Sequence[SimConfig], *, contour_from: int) -> SimConfig:
    """
    The pooled configuration of a dataset: the median of every scalar
    nuisance parameter (``_POOLED_MEDIAN_FIELDS``; n_rings and n_frames
    the rounded median, max_dark_frames the maximum), the per-decile
    median of the n_locs deciles, the lp samples concatenated, the lpz
    samples pooled per stratum on the union grid of the configurations'
    strata (all must share one bin width; a configuration's stratum
    goes to the pooled bin holding its centre) with the axial scale
    averaged with the sample counts as weights (1 where no
    configuration contributes), and the contour of
    ``configs[contour_from]`` (its library fields cleared). Model fields
    are left at their defaults (M1). The provenance lists the pooled
    files.
    """
    if not configs:
        raise ValueError("pool_sim_configs: no configuration")
    if not 0 <= int(contour_from) < len(configs):
        raise ValueError(f"pool_sim_configs: contour_from {contour_from} outside 0..{len(configs) - 1}")
    base = configs[int(contour_from)]
    kw: Dict[str, Any] = {}
    for name in _POOLED_MEDIAN_FIELDS:
        kw[name] = float(np.median([float(getattr(c, name)) for c in configs]))
    kw["n_rings"] = int(round(float(np.median([c.n_rings for c in configs]))))
    kw["n_frames"] = int(round(float(np.median([c.n_frames for c in configs]))))
    kw["max_dark_frames"] = int(max(c.max_dark_frames for c in configs))
    decs = [np.asarray(c.n_locs_per_cluster_quantiles, dtype=np.float64) for c in configs if c.n_locs_per_cluster_quantiles is not None]
    kw["n_locs_per_cluster_quantiles"] = tuple(float(v) for v in np.median(np.vstack(decs), axis=0)) if decs else None
    kw["lp_lateral_samples_nm"] = np.concatenate([np.asarray(c.lp_lateral_samples_nm, dtype=np.float64) for c in configs])
    widths = np.concatenate([np.diff(np.asarray(c.lpz_bin_edges_lab_nm, dtype=np.float64)) for c in configs])
    width = float(np.median(widths))
    if width <= 0.0 or np.any(np.abs(widths - width) > 1e-6 * width):
        raise ValueError("pool_sim_configs: the configurations' lpz strata must share one bin width")
    lo = min(float(np.asarray(c.lpz_bin_edges_lab_nm)[0]) for c in configs)
    hi = max(float(np.asarray(c.lpz_bin_edges_lab_nm)[-1]) for c in configs)
    n_bins = max(1, int(round((hi - lo) / width)))
    edges = lo + width * np.arange(n_bins + 1, dtype=np.float64)
    edges[-1] = max(edges[-1], hi)
    pooled: List[List[NDArray[np.float64]]] = [[] for _ in range(n_bins)]
    scale_num = np.zeros(n_bins)
    scale_den = np.zeros(n_bins)
    scale_any: List[List[float]] = [[] for _ in range(n_bins)]
    for c in configs:
        ce = np.asarray(c.lpz_bin_edges_lab_nm, dtype=np.float64)
        centres = 0.5 * (ce[:-1] + ce[1:])
        target = np.clip(np.floor((centres - lo) / width).astype(np.int64), 0, n_bins - 1)
        for s, t in enumerate(target.tolist()):
            v = np.asarray(c.lpz_samples_by_bin_nm[s], dtype=np.float64)
            pooled[t].append(v)
            w = float(v.size)
            sc = float(np.asarray(c.axial_scale_by_bin)[s])
            scale_num[t] += w * sc
            scale_den[t] += w
            scale_any[t].append(sc)
    kw["lpz_bin_edges_lab_nm"] = edges
    kw["lpz_samples_by_bin_nm"] = [np.concatenate(v) if v else np.zeros(0) for v in pooled]
    kw["axial_scale_by_bin"] = np.array([scale_num[t] / scale_den[t] if scale_den[t] > 0.0
                                         else (float(np.mean(scale_any[t])) if scale_any[t] else 1.0) for t in range(n_bins)])
    kw["contour_nm"] = None if base.contour_nm is None else np.array(base.contour_nm, dtype=np.float64)
    kw["ellipse_semi_axes_nm"] = base.ellipse_semi_axes_nm
    # H5-C: the lambda basis and the contour smoothing are carried over when every configuration agrees (a
    # pooled lambda is a median of lambdas that must refer to one kind of length); mixed bases pool to None
    # (today's behaviour) with the per-axon values in the provenance.
    bases = {c.length_basis for c in configs}
    kw["length_basis"] = bases.pop() if len(bases) == 1 else None
    knots = {c.contour_smoothing_knot_nm for c in configs}
    kw["contour_smoothing_knot_nm"] = knots.pop() if len(knots) == 1 else None
    modes = {c.contour_smoothing_mode for c in configs}
    kw["contour_smoothing_mode"] = modes.pop() if len(modes) == 1 else None
    centroid_sds = [float(c.provenance["radial_offset_sd_centroid_pspline_nm"]) for c in configs
                    if "radial_offset_sd_centroid_pspline_nm" in c.provenance]
    knot_recorded = {c.provenance.get("contour_smoothing_knot_nm") for c in configs}
    kw["provenance"] = {
        "dataset_role": "exploratory", "date": datetime.date.today().isoformat(), "code_commit": _git_commit(),
        "pooled_from": [str(c.provenance.get("file", "")) for c in configs], "n_axons": len(configs),
        "contour_from": str(base.provenance.get("file", "")), "contour_from_index": int(contour_from),
        "n_locs_total": int(sum(int(c.provenance.get("n_locs", 0) or 0) for c in configs)),
        "note": "medians of the per-axon nuisance parameters; concatenated precision samples; no hypothesis statistic",
    }
    # The H5-C provenance only when an input configuration carries H5-C information (review of 2026-09-26: under
    # pre-H5-C inputs the pooled provenance keeps exactly its former keys).
    h5c_inputs = any(c.length_basis is not None or "membrane_source" in c.provenance or c.contour_smoothing_mode is not None
                     for c in configs)
    if h5c_inputs:
        kw["provenance"].update({
            "length_basis_per_axon": [c.length_basis for c in configs],
            "membrane_source_per_axon": [c.provenance.get("membrane_source") for c in configs],
            "contour_smoothing_knot_nm": (knot_recorded.pop() if len(knot_recorded) == 1 else None),
            "radial_offset_sd_centroid_pspline_nm": (float(np.median(centroid_sds)) if centroid_sds else None),
            "radial_offset_sd_per_axon_nm": [float(c.radial_offset_sd_nm) for c in configs],
            "locs_per_event_measured_per_axon": [c.provenance.get("locs_per_event_measured") for c in configs],
        })
    return SimConfig(**kw)


# ============================================================================
# YAML
# ============================================================================

def write_sim_config(path: str, cfg: SimConfig) -> None:
    """
    Write ``cfg`` as YAML: the comment block ``SIM_CONFIG_HEADER``, then
    the fields in dataclass order, arrays as lists (the contour as a
    list of [x, y]), tuples as lists, None as null, every value through
    ``tools.mps_columns._plain`` so that no Python tag is written.
    Raises ValueError without a provenance that names ``dataset_role``,
    ``date`` and ``code_commit`` (``_check_provenance``): what is
    written is what a power run is frozen on.
    """
    yaml = _yaml_module()
    _check_provenance(cfg.provenance, "write_sim_config")
    data: Dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        if f.name in TRANSIENT_FIELDS:
            continue
        # The H5-C fields are written only when set (review of 2026-09-26): a configuration that does not use them
        # (None, the default, which ``load_sim_config`` restores for an absent key) is written byte for byte as
        # before H5-C, so the round trip stays exact and no existing file changes.
        if f.name in _H5C_OPTIONAL_FIELDS + _H5D_OPTIONAL_FIELDS and getattr(cfg, f.name) is None:
            continue
        data[f.name] = _plain(getattr(cfg, f.name))
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True, width=120)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(SIM_CONFIG_HEADER) + "\n" + body)


def load_sim_config(path: str) -> SimConfig:
    """
    Read a file written by ``write_sim_config`` back, exactly. Strict:
    an unknown key raises ValueError naming it, a repeated key raises
    (``_yaml_load_strict``; PyYAML would keep the last value), every
    value is coerced to the field's type and range by
    ``SimConfig.__post_init__`` (ValueError). Fields absent from the
    file take their defaults; the transient ``TRANSIENT_FIELDS`` are
    not YAML keys (unknown). ``contour_library_npz`` is kept as
    written; the YAML's folder is recorded in ``library_dir`` so that
    ``contour_library_path`` resolves a relative library next to it.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    raw = _yaml_load_strict(text, path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of parameters, got {type(raw).__name__}")
    names = [f.name for f in dataclasses.fields(SimConfig) if f.name not in TRANSIENT_FIELDS]
    unknown = [str(k) for k in raw if k not in names]
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown}; a simulation configuration accepts only {names}")
    try:
        cfg = SimConfig(**{str(k): v for k, v in raw.items()})
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    cfg.library_dir = os.path.dirname(os.path.abspath(path))
    return cfg
